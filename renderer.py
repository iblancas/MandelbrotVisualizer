"""Asynchronous Mandelbrot renderer with caching and prefetching."""
import numpy as np
import threading
from .compute import (compute_mandelbrot, compute_mandelbrot_partial, compute_mandelbrot_custom,
                      prepare_custom_formula, apply_colormap_smooth, downscale_2x)
from .colormaps import get_default_colormap

try:
    from .compute_gpu import get_gpu_compute, is_gpu_available, should_default_to_gpu
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    is_gpu_available = lambda: False
    should_default_to_gpu = lambda: False


class MandelbrotRenderer:
    """Async Mandelbrot rendering with caching and prefetching."""
    
    def __init__(self, width, height, max_iter, supersample=2, margin=0.15,
                 func_id=0, escape_radius=2.0, custom_formula=None, use_gpu=None,
                 julia_mode=False, julia_c_real=-0.7, julia_c_imag=0.27015):
        self.width, self.height = width, height
        self.max_iter, self.supersample, self.margin = max_iter, supersample, margin
        self.func_id, self.escape_radius = func_id, escape_radius
        self.custom_formula, self._prepared_formula = None, None
        if custom_formula:
            self.custom_formula = custom_formula
            self._prepared_formula = prepare_custom_formula(custom_formula)
        
        self.julia_mode, self.julia_c_real, self.julia_c_imag = julia_mode, julia_c_real, julia_c_imag
        
        if use_gpu is None:
            use_gpu = GPU_AVAILABLE and should_default_to_gpu()
        self.use_gpu = use_gpu and GPU_AVAILABLE and is_gpu_available()
        self._gpu_compute = get_gpu_compute(prefer_gpu=True) if self.use_gpu else None
        
        self.margin_pixels = int(width * margin)
        self.full_width, self.full_height = width + 2 * self.margin_pixels, height + 2 * self.margin_pixels
        self.render_width, self.render_height = self.full_width * supersample, self.full_height * supersample
        
        self.colormap = get_default_colormap()
        self.rgb_hi = np.zeros((self.render_height, self.render_width, 3), dtype=np.uint8)
        self.rgb = np.zeros((self.full_height, self.full_width, 3), dtype=np.uint8)
        self.data_cache = np.zeros((self.render_height, self.render_width), dtype=np.float64)
        self.cache_bounds = None
        
        self.computing = self.result_ready = False
        self.pending_bounds = self.actual_bounds = None
        self.lock = threading.Lock()
        
        self.prefetch_cache = {}
        self.prefetch_lock = threading.Lock()
        self.prefetching = False
        self.prefetch_base_bounds = None
        self._shutdown = False
    
    def compute_async(self, x_min, x_max, y_min, y_max):
        """Start async computation. Returns True if prefetch hit."""
        w, h = x_max - x_min, y_max - y_min
        expanded_bounds = (x_min - w * self.margin, x_max + w * self.margin,
                          y_min - h * self.margin, y_max + h * self.margin)
        
        prefetch_data, prefetch_rgb, prefetch_bounds = self._check_prefetch(expanded_bounds)
        if prefetch_data is not None:
            with self.lock:
                self.data_cache[:] = prefetch_data
                self.cache_bounds = self.actual_bounds = prefetch_bounds
                self.rgb[:] = prefetch_rgb
                self.result_ready = True
                self._start_prefetch(prefetch_bounds)
            return True
        
        with self.lock:
            self.pending_bounds = expanded_bounds
            if not self.computing:
                self.computing = True
                threading.Thread(target=self._compute_thread, daemon=True).start()
        return False
    
    def cleanup(self):
        """Clean up resources and signal threads to stop."""
        self._shutdown = True
        if self._gpu_compute is not None:
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            except Exception:
                pass
            self._gpu_compute = None
        self.prefetch_cache.clear()
    
    def _compute_thread(self):
        """Background computation thread."""
        while not self._shutdown:
            with self.lock:
                bounds = self.pending_bounds
                self.pending_bounds = None
                old_cache_bounds = self.cache_bounds
            
            if bounds is None:
                with self.lock:
                    self.computing = False
                break
            
            new_w = bounds[1] - bounds[0]
            new_h = bounds[3] - bounds[2]
            
            can_reuse = False
            if old_cache_bounds is not None:
                old_w, old_h = old_cache_bounds[1] - old_cache_bounds[0], old_cache_bounds[3] - old_cache_bounds[2]
                r_x, r_y = new_w / old_w if old_w > 0 else 0, new_h / old_h if old_h > 0 else 0
                can_reuse = 0.99 < r_x < 1.01 and 0.99 < r_y < 1.01
            
            if can_reuse and self._prepared_formula is None and not self.use_gpu:
                data = self._compute_incremental(bounds, old_cache_bounds, new_w, new_h)
            else:
                data = self._compute_full(bounds)
            
            self.data_cache[:] = data
            if self.use_gpu and self._gpu_compute:
                self._gpu_compute.apply_colormap_smooth(data, self.max_iter, self.colormap, self.rgb_hi)
                self._gpu_compute.downscale_2x(self.rgb_hi, self.rgb)
            else:
                apply_colormap_smooth(data, self.max_iter, self.colormap, self.rgb_hi)
                downscale_2x(self.rgb_hi, self.rgb)
            
            with self.lock:
                self.cache_bounds = self.actual_bounds = bounds
                self.result_ready = True
                if self.pending_bounds is None:
                    self.computing = False
                    self._start_prefetch(bounds)
                    break
    
    def _compute_full(self, bounds):
        """Compute full fractal for bounds."""
        args = (bounds[0], bounds[1], bounds[2], bounds[3],
                self.render_width, self.render_height, self.max_iter)
        julia_args = (self.julia_mode, self.julia_c_real, self.julia_c_imag)
        
        if self._prepared_formula is not None:
            return compute_mandelbrot_custom(*args, self._prepared_formula, self.escape_radius, *julia_args)
        elif self.use_gpu and self._gpu_compute:
            return self._gpu_compute.compute_mandelbrot(*args, self.func_id, self.escape_radius, *julia_args)
        return compute_mandelbrot(*args, self.func_id, self.escape_radius, *julia_args)
    
    def _compute_incremental(self, new_bounds, old_bounds, new_w, new_h):
        """Compute incrementally by reusing cached data when panning."""
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
                self.func_id, self.escape_radius,
                self.julia_mode, self.julia_c_real, self.julia_c_imag
            )
        
        # Bottom strip
        if dst_y_end < self.render_height:
            compute_mandelbrot_partial(
                new_x_min, new_x_max, new_y_min, new_y_max,
                self.render_width, self.render_height, self.max_iter,
                new_data, 0, dst_y_end, self.render_width, self.render_height - dst_y_end,
                self.func_id, self.escape_radius,
                self.julia_mode, self.julia_c_real, self.julia_c_imag
            )
        
        # Left strip (excluding corners already done)
        if dst_x_start > 0:
            compute_mandelbrot_partial(
                new_x_min, new_x_max, new_y_min, new_y_max,
                self.render_width, self.render_height, self.max_iter,
                new_data, 0, dst_y_start, dst_x_start, dst_y_end - dst_y_start,
                self.func_id, self.escape_radius,
                self.julia_mode, self.julia_c_real, self.julia_c_imag
            )
        
        # Right strip (excluding corners already done)
        if dst_x_end < self.render_width:
            compute_mandelbrot_partial(
                new_x_min, new_x_max, new_y_min, new_y_max,
                self.render_width, self.render_height, self.max_iter,
                new_data, dst_x_end, dst_y_start,
                self.render_width - dst_x_end, dst_y_end - dst_y_start,
                self.func_id, self.escape_radius,
                self.julia_mode, self.julia_c_real, self.julia_c_imag
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
            # Check if we should stop (shutdown, new render requested, or moved away)
            if self._shutdown:
                break
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
                         custom_formula=None, use_gpu=None, julia_mode=None, 
                         julia_c_real=None, julia_c_imag=None):
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
            julia_mode: Enable/disable Julia mode (or None to keep current)
            julia_c_real: Real part of c for Julia mode (or None to keep current)
            julia_c_imag: Imag part of c for Julia mode (or None to keep current)
        
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
        # Handle Julia mode settings
        if julia_mode is not None and julia_mode != self.julia_mode:
            self.julia_mode = julia_mode
            changed = True
        if julia_c_real is not None and julia_c_real != self.julia_c_real:
            self.julia_c_real = julia_c_real
            changed = True
        if julia_c_imag is not None and julia_c_imag != self.julia_c_imag:
            self.julia_c_imag = julia_c_imag
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
