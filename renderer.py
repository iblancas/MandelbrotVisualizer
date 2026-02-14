"""
Asynchronous Mandelbrot renderer with caching and prefetching.

The MandelbrotRenderer class handles:
- Background (async) computation so UI stays responsive
- Caching of iteration data for fast panning at same zoom level
- Prefetching of adjacent regions for smoother interaction
- Supersampled anti-aliasing (2x by default)
- Margin/overscan for smooth pan transitions
- GPU acceleration when available (via PyTorch)
"""

import numpy as np
import threading
from .compute import (
    compute_mandelbrot,
    compute_mandelbrot_partial,
    compute_mandelbrot_custom,
    prepare_custom_formula,
    apply_colormap_smooth,
    downscale_2x
)
from .colormaps import get_default_colormap

# Try to import GPU compute module
try:
    from .compute_gpu import GPUCompute, get_gpu_compute, is_gpu_available, should_default_to_gpu
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    is_gpu_available = lambda: False
    should_default_to_gpu = lambda: False


class MandelbrotRenderer:
    """
    Handles async Mandelbrot rendering with caching and prefetching.
    
    Usage:
        renderer = MandelbrotRenderer(800, 600, max_iter=256)
        renderer.compute_async(x_min, x_max, y_min, y_max)
        
        # In your game loop:
        result, bounds = renderer.get_result()
        if result is not None:
            # result is the RGB image, bounds are the actual computed bounds
            display(result)
    
    Attributes:
        width, height: Display dimensions
        max_iter: Maximum iteration count
        margin: Extra rendering margin (fraction of width) for smooth panning
        use_gpu: Whether to use GPU acceleration
    """
    
    def __init__(self, width, height, max_iter, supersample=2, margin=0.15,
                 func_id=0, escape_radius=2.0, custom_formula=None, use_gpu=None):
        """
        Initialize the renderer.
        
        Args:
            width, height: Display dimensions in pixels
            max_iter: Maximum iteration count for Mandelbrot computation
            supersample: Supersampling factor for anti-aliasing (default 2x)
            margin: Extra margin around visible area for smooth panning (default 15%)
            func_id: Which iteration function to use (default 0 = z² + c)
            escape_radius: Escape threshold for iteration (default 2.0)
            custom_formula: Custom formula string (overrides func_id if set)
            use_gpu: Whether to use GPU acceleration (None = auto-detect best option)
        """
        self.width = width
        self.height = height
        self.max_iter = max_iter
        self.supersample = supersample
        self.margin = margin
        self.func_id = func_id
        self.escape_radius = escape_radius
        self.custom_formula = None
        self._prepared_formula = None
        if custom_formula:
            self.custom_formula = custom_formula
            self._prepared_formula = prepare_custom_formula(custom_formula)
        
        # GPU settings - auto-detect best default if not specified
        if use_gpu is None:
            # Only default to GPU for CUDA; MPS is slower than Numba for typical renders
            use_gpu = GPU_AVAILABLE and should_default_to_gpu()
        self.use_gpu = use_gpu and GPU_AVAILABLE and is_gpu_available()
        self._gpu_compute = None
        if self.use_gpu:
            self._gpu_compute = get_gpu_compute(prefer_gpu=True)
        
        # With margin, we render a larger image
        self.margin_pixels = int(width * margin)
        self.full_width = width + 2 * self.margin_pixels
        self.full_height = height + 2 * self.margin_pixels
        
        # Render at supersample resolution, then downscale
        self.render_width = self.full_width * supersample
        self.render_height = self.full_height * supersample
        
        # Colormap and output buffers
        self.colormap = get_default_colormap()
        self.rgb_hi = np.zeros((self.render_height, self.render_width, 3), dtype=np.uint8)
        self.rgb = np.zeros((self.full_height, self.full_width, 3), dtype=np.uint8)
        
        # Cache for iteration data (not just RGB) - allows reuse when panning
        self.data_cache = np.zeros((self.render_height, self.render_width), dtype=np.float64)
        self.cache_bounds = None  # (x_min, x_max, y_min, y_max) of cached data
        
        # Async computation state
        self.computing = False
        self.result_ready = False
        self.pending_bounds = None
        self.actual_bounds = None  # The actual computed bounds (with margin)
        self.lock = threading.Lock()
        
        # Prefetch cache: dict mapping direction/zoom to (bounds, data, rgb)
        # Possible keys: 'left', 'right', 'up', 'down', 'zoom_in', 'zoom_out'
        self.prefetch_cache = {}
        self.prefetch_lock = threading.Lock()
        self.prefetching = False
        self.prefetch_base_bounds = None  # Bounds we're prefetching around
    
    def compute_async(self, x_min, x_max, y_min, y_max):
        """
        Start async computation for the given bounds.
        
        The actual rendered region will be expanded by the margin factor.
        If prefetched data is available for these bounds, it will be used
        immediately (returns True).
        
        Args:
            x_min, x_max: Real axis bounds
            y_min, y_max: Imaginary axis bounds
        
        Returns:
            True if prefetch hit (result ready immediately), False otherwise
        """
        # Expand bounds by margin
        w = x_max - x_min
        h = y_max - y_min
        mx = w * self.margin
        my = h * self.margin
        
        expanded_bounds = (x_min - mx, x_max + mx, y_min - my, y_max + my)
        
        # Check if we have prefetched data for these bounds
        prefetch_data, prefetch_rgb, prefetch_bounds = self._check_prefetch(expanded_bounds)
        if prefetch_data is not None:
            # Use prefetched data immediately
            with self.lock:
                self.data_cache[:] = prefetch_data
                self.cache_bounds = prefetch_bounds
                self.actual_bounds = prefetch_bounds
                self.rgb[:] = prefetch_rgb
                self.result_ready = True
                # Start new prefetch around new location
                self._start_prefetch(prefetch_bounds)
            return True
        
        with self.lock:
            self.pending_bounds = expanded_bounds
            if not self.computing:
                self.computing = True
                thread = threading.Thread(target=self._compute_thread)
                thread.daemon = True
                thread.start()
        return False
    
    def _compute_thread(self):
        """Background thread for Mandelbrot computation."""
        while True:
            with self.lock:
                bounds = self.pending_bounds
                self.pending_bounds = None
                old_cache_bounds = self.cache_bounds
            
            if bounds is None:
                with self.lock:
                    self.computing = False
                break
            
            new_x_min, new_x_max, new_y_min, new_y_max = bounds
            new_w = new_x_max - new_x_min
            new_h = new_y_max - new_y_min
            
            # Check if we can reuse cached data (same zoom level = same pixel scale)
            can_reuse = False
            if old_cache_bounds is not None:
                old_x_min, old_x_max, old_y_min, old_y_max = old_cache_bounds
                old_w = old_x_max - old_x_min
                old_h = old_y_max - old_y_min
                
                # Same zoom level if the scale is the same (within tolerance)
                scale_ratio_x = new_w / old_w if old_w > 0 else 0
                scale_ratio_y = new_h / old_h if old_h > 0 else 0
                
                if 0.99 < scale_ratio_x < 1.01 and 0.99 < scale_ratio_y < 1.01:
                    can_reuse = True
            
            # GPU doesn't support incremental rendering (yet), but it's fast enough
            if can_reuse and self._prepared_formula is None and not self.use_gpu:
                data = self._compute_incremental(bounds, old_cache_bounds, new_w, new_h)
            else:
                # Full recompute (using GPU or CPU)
                data = self._compute_full(bounds)
            
            # Update cache
            self.data_cache[:] = data
            
            # Apply colormap and downscale (using GPU or CPU)
            if self.use_gpu and self._gpu_compute is not None:
                self._gpu_compute.apply_colormap_smooth(data, self.max_iter, self.colormap, self.rgb_hi)
                self._gpu_compute.downscale_2x(self.rgb_hi, self.rgb)
            else:
                apply_colormap_smooth(data, self.max_iter, self.colormap, self.rgb_hi)
                downscale_2x(self.rgb_hi, self.rgb)
            
            with self.lock:
                self.cache_bounds = bounds
                self.actual_bounds = bounds
                self.result_ready = True
                if self.pending_bounds is None:
                    self.computing = False
                    # Start prefetching in background
                    self._start_prefetch(bounds)
                    break
    
    def _compute_full(self, bounds):
        """
        Compute full Mandelbrot for bounds using best available method.
        
        Uses GPU if available and enabled, otherwise falls back to CPU.
        """
        if self._prepared_formula is not None:
            # Custom formula: always use CPU (non-JIT)
            return compute_mandelbrot_custom(
                bounds[0], bounds[1], bounds[2], bounds[3],
                self.render_width, self.render_height, self.max_iter,
                self._prepared_formula, self.escape_radius
            )
        elif self.use_gpu and self._gpu_compute is not None:
            # Use GPU acceleration
            return self._gpu_compute.compute_mandelbrot(
                bounds[0], bounds[1], bounds[2], bounds[3],
                self.render_width, self.render_height, self.max_iter,
                self.func_id, self.escape_radius
            )
        else:
            # Use CPU (Numba JIT)
            return compute_mandelbrot(
                bounds[0], bounds[1], bounds[2], bounds[3],
                self.render_width, self.render_height, self.max_iter,
                self.func_id, self.escape_radius
            )
    
    def _compute_incremental(self, new_bounds, old_bounds, new_w, new_h):
        """
        Compute Mandelbrot incrementally by reusing cached data.
        
        Only computes the newly visible strips (edges) instead of
        the entire image. Used when panning at the same zoom level.
        """
        new_x_min, new_x_max, new_y_min, new_y_max = new_bounds
        old_x_min, old_x_max, old_y_min, old_y_max = old_bounds
        
        # Calculate overlap in pixel coordinates
        px_per_unit_x = self.render_width / new_w
        px_per_unit_y = self.render_height / new_h
        
        # Offset of old data relative to new (in pixels)
        offset_x = int((old_x_min - new_x_min) * px_per_unit_x)
        offset_y = int((old_y_min - new_y_min) * px_per_unit_y)
        
        # Create new data array
        new_data = np.full((self.render_height, self.render_width), -1.0, dtype=np.float64)
        
        # Copy overlapping region from cache
        src_x_start = max(0, -offset_x)
        src_y_start = max(0, -offset_y)
        src_x_end = min(self.render_width, self.render_width - offset_x)
        src_y_end = min(self.render_height, self.render_height - offset_y)
        
        dst_x_start = max(0, offset_x)
        dst_y_start = max(0, offset_y)
        dst_x_end = dst_x_start + (src_x_end - src_x_start)
        dst_y_end = dst_y_start + (src_y_end - src_y_start)
        
        if src_x_end > src_x_start and src_y_end > src_y_start:
            new_data[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
                self.data_cache[src_y_start:src_y_end, src_x_start:src_x_end]
        
        # Compute only the regions that need new data
        # Top strip
        if dst_y_start > 0:
            compute_mandelbrot_partial(
                new_x_min, new_x_max, new_y_min, new_y_max,
                self.render_width, self.render_height, self.max_iter,
                new_data, 0, 0, self.render_width, dst_y_start,
                self.func_id, self.escape_radius
            )
        
        # Bottom strip
        if dst_y_end < self.render_height:
            compute_mandelbrot_partial(
                new_x_min, new_x_max, new_y_min, new_y_max,
                self.render_width, self.render_height, self.max_iter,
                new_data, 0, dst_y_end, self.render_width, self.render_height - dst_y_end,
                self.func_id, self.escape_radius
            )
        
        # Left strip (excluding corners already done)
        if dst_x_start > 0:
            compute_mandelbrot_partial(
                new_x_min, new_x_max, new_y_min, new_y_max,
                self.render_width, self.render_height, self.max_iter,
                new_data, 0, dst_y_start, dst_x_start, dst_y_end - dst_y_start,
                self.func_id, self.escape_radius
            )
        
        # Right strip (excluding corners already done)
        if dst_x_end < self.render_width:
            compute_mandelbrot_partial(
                new_x_min, new_x_max, new_y_min, new_y_max,
                self.render_width, self.render_height, self.max_iter,
                new_data, dst_x_end, dst_y_start,
                self.render_width - dst_x_end, dst_y_end - dst_y_start,
                self.func_id, self.escape_radius
            )
        
        return new_data
    
    def _start_prefetch(self, base_bounds):
        """Start background prefetching of adjacent regions."""
        with self.prefetch_lock:
            if self.prefetching:
                return
            self.prefetching = True
            self.prefetch_base_bounds = base_bounds
            # Clear old prefetch cache as we're at new location
            self.prefetch_cache.clear()
        
        thread = threading.Thread(target=self._prefetch_thread, args=(base_bounds,))
        thread.daemon = True
        thread.start()
    
    def _prefetch_thread(self, base_bounds):
        """Prefetch adjacent regions and zoom levels."""
        x_min, x_max, y_min, y_max = base_bounds
        w = x_max - x_min
        h = y_max - y_min
        
        # Define prefetch regions: 50% overlap for panning
        pan_amount = 0.5
        prefetch_targets = [
            ('right', (x_min + w * pan_amount, x_max + w * pan_amount, y_min, y_max)),
            ('left', (x_min - w * pan_amount, x_max - w * pan_amount, y_min, y_max)),
            ('up', (x_min, x_max, y_min + h * pan_amount, y_max + h * pan_amount)),
            ('down', (x_min, x_max, y_min - h * pan_amount, y_max - h * pan_amount)),
            # Zoom: center zoom, 0.85 for in, 1.18 for out
            ('zoom_in', self._zoom_bounds(base_bounds, 0.85)),
            ('zoom_out', self._zoom_bounds(base_bounds, 1.18)),
        ]
        
        for direction, bounds in prefetch_targets:
            # Check if we should stop (new render requested or moved away)
            with self.lock:
                if self.pending_bounds is not None:
                    break
            with self.prefetch_lock:
                if self.prefetch_base_bounds != base_bounds:
                    break
            
            # Compute this region (using GPU or CPU)
            data = self._compute_full(bounds)
            
            rgb_hi = np.zeros((self.render_height, self.render_width, 3), dtype=np.uint8)
            rgb = np.zeros((self.full_height, self.full_width, 3), dtype=np.uint8)
            
            if self.use_gpu and self._gpu_compute is not None:
                self._gpu_compute.apply_colormap_smooth(data, self.max_iter, self.colormap, rgb_hi)
                self._gpu_compute.downscale_2x(rgb_hi, rgb)
            else:
                apply_colormap_smooth(data, self.max_iter, self.colormap, rgb_hi)
                downscale_2x(rgb_hi, rgb)
            
            with self.prefetch_lock:
                if self.prefetch_base_bounds == base_bounds:
                    self.prefetch_cache[direction] = (bounds, data.copy(), rgb.copy())
        
        with self.prefetch_lock:
            self.prefetching = False
    
    def _zoom_bounds(self, bounds, zoom_factor):
        """Calculate zoomed bounds centered on the middle."""
        x_min, x_max, y_min, y_max = bounds
        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2
        w = (x_max - x_min) * zoom_factor
        h = (y_max - y_min) * zoom_factor
        return (cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2)
    
    def _check_prefetch(self, bounds):
        """Check if we have prefetched data that matches the requested bounds."""
        with self.prefetch_lock:
            for direction, (cached_bounds, data, rgb) in self.prefetch_cache.items():
                if self._bounds_match(bounds, cached_bounds):
                    return data.copy(), rgb.copy(), cached_bounds
        return None, None, None
    
    def _bounds_match(self, b1, b2, tolerance=0.02):
        """Check if two bounds are close enough to be considered matching."""
        w1 = b1[1] - b1[0]
        for i in range(4):
            if abs(b1[i] - b2[i]) > w1 * tolerance:
                return False
        return True
    
    def get_result(self):
        """
        Get the latest render result if ready.
        
        Returns:
            Tuple of (image, actual_bounds) if result ready, (None, None) otherwise.
            Image is flipped for pygame coordinate system.
        """
        with self.lock:
            if self.result_ready:
                self.result_ready = False
                return np.flipud(self.rgb).copy(), self.actual_bounds
        return None, None
    
    def update_settings(self, max_iter=None, colormap=None, func_id=None, escape_radius=None,
                         custom_formula=None, use_gpu=None):
        """
        Update rendering settings.
        
        Clears all caches since the visual appearance will change.
        
        Args:
            max_iter: New maximum iteration count (or None to keep current)
            colormap: New colormap array (or None to keep current)
            func_id: New iteration function ID (or None to keep current)
            escape_radius: New escape threshold (or None to keep current)
            custom_formula: Custom formula string (or None to use func_id, '' to clear)
            use_gpu: Enable/disable GPU acceleration (or None to keep current)
        
        Returns:
            True if any setting changed, False otherwise
        """
        changed = False
        if max_iter is not None and max_iter != self.max_iter:
            self.max_iter = max_iter
            changed = True
        if colormap is not None:
            self.colormap = colormap
            changed = True
        if func_id is not None and func_id != self.func_id:
            self.func_id = func_id
            changed = True
        if escape_radius is not None and escape_radius != self.escape_radius:
            self.escape_radius = escape_radius
            changed = True
        # Handle custom formula
        if custom_formula is not None:
            if custom_formula == '':
                # Clear custom formula
                if self.custom_formula is not None:
                    self.custom_formula = None
                    self._prepared_formula = None
                    changed = True
            elif custom_formula != self.custom_formula:
                self.custom_formula = custom_formula
                self._prepared_formula = prepare_custom_formula(custom_formula)
                changed = True
        # Handle GPU toggle
        if use_gpu is not None:
            want_gpu = use_gpu and GPU_AVAILABLE and is_gpu_available()
            if want_gpu != self.use_gpu:
                self.use_gpu = want_gpu
                if want_gpu and self._gpu_compute is None:
                    self._gpu_compute = get_gpu_compute(prefer_gpu=True)
                changed = True
        if changed:
            # Clear caches since settings changed
            self.cache_bounds = None
            with self.prefetch_lock:
                self.prefetch_cache.clear()
                self.prefetch_base_bounds = None
        return changed
    
    def get_gpu_info(self):
        """
        Get information about GPU status.
        
        Returns:
            dict with keys:
            - 'available': bool, whether GPU is available
            - 'enabled': bool, whether GPU is currently enabled
            - 'device': str, device name/description
        """
        if not GPU_AVAILABLE:
            return {
                'available': False,
                'enabled': False,
                'device': 'PyTorch not installed'
            }
        
        gpu_compute = get_gpu_compute(prefer_gpu=True)
        return {
            'available': gpu_compute.is_gpu,
            'enabled': self.use_gpu,
            'device': gpu_compute.get_device_info()
        }
    
    def toggle_gpu(self):
        """
        Toggle GPU acceleration on/off.
        
        Returns:
            bool: New GPU enabled state
        """
        if not GPU_AVAILABLE or not is_gpu_available():
            return False
        
        self.update_settings(use_gpu=not self.use_gpu)
        return self.use_gpu
