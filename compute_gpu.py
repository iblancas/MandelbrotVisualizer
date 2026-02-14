"""
GPU-accelerated Mandelbrot computation using PyTorch.

This module provides GPU-accelerated versions of the fractal computation
functions. It auto-detects available hardware:
- CUDA (NVIDIA GPUs)
- MPS (Apple Silicon)
- CPU fallback via PyTorch (still faster than pure Python)

The GPU implementation processes all pixels simultaneously using
tensor operations, achieving massive parallelism.

Usage:
    from compute_gpu import GPUCompute
    
    gpu = GPUCompute()
    if gpu.available:
        result = gpu.compute_mandelbrot(x_min, x_max, y_min, y_max, 
                                         width, height, max_iter)
"""

import numpy as np

# Try to import PyTorch
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None


class GPUCompute:
    """
    GPU-accelerated fractal computation class.
    
    Automatically detects and uses the best available device:
    - CUDA for NVIDIA GPUs (recommended - significant speedup)
    - MPS for Apple Silicon (available but may be slower than CPU for typical renders)
    - CPU as fallback (still uses PyTorch vectorization)
    
    Note: For Apple Silicon (MPS), the Numba JIT CPU implementation is often
    faster due to lower overhead. GPU mode is still available for experimentation
    or for very large renders (4K+) with high iteration counts (2000+).
    
    CUDA GPUs will see better performance due to lower kernel launch overhead.
    """
    
    # Function type constants (same as compute.py)
    FUNC_Z2_PLUS_C = 0      # z² + c
    FUNC_Z3_PLUS_C = 1      # z³ + c
    FUNC_Z4_PLUS_C = 2      # z⁴ + c
    FUNC_Z5_PLUS_C = 3      # z⁵ + c
    FUNC_TRICORN = 4        # (z̄)² + c
    FUNC_EXP = 5            # c·e^z
    FUNC_SIN = 6            # c·sin(z)
    FUNC_Z2_Z_C = 7         # z² + z + c
    FUNC_Z6_PLUS_C = 8      # z⁶ + c
    FUNC_Z7_PLUS_C = 9      # z⁷ + c
    FUNC_Z8_PLUS_C = 10     # z⁸ + c
    FUNC_Z2_CZ = 11         # z² + c·z
    FUNC_Z3_Z_C = 12        # z³ + z + c
    FUNC_Z4_Z_C = 13        # z⁴ + z + c
    FUNC_COS = 14           # c·cos(z)
    FUNC_Z2_MINUS_Z_C = 15  # z² - z + c
    FUNC_CUBIC_JULIA = 16   # z·(z² + c)
    FUNC_Z2_C2 = 17         # z² + c²
    FUNC_RATIONAL = 18      # (z²+c)/(z²-c)
    FUNC_Z3_MINUS_Z_C = 19  # z³ - z + c
    
    def __init__(self, prefer_gpu=True):
        """
        Initialize GPU compute.
        
        Args:
            prefer_gpu: If False, force CPU even if GPU available
        """
        self.available = TORCH_AVAILABLE
        self.device = None
        self.device_name = "None"
        self.is_gpu = False
        self.is_cuda = False
        self.is_mps = False
        self.dtype = torch.float32  # Default dtype
        
        if not TORCH_AVAILABLE:
            return
        
        # Detect best available device
        if prefer_gpu:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                self.device_name = torch.cuda.get_device_name(0)
                self.is_gpu = True
                self.is_cuda = True
                self.dtype = torch.float64  # CUDA supports float64
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device("mps")
                self.device_name = "Apple Silicon GPU (MPS)"
                self.is_gpu = True
                self.is_mps = True
                self.dtype = torch.float32  # MPS only supports float32
            else:
                self.device = torch.device("cpu")
                self.device_name = "CPU (PyTorch)"
                self.is_gpu = False
                self.dtype = torch.float64
        else:
            self.device = torch.device("cpu")
            self.device_name = "CPU (PyTorch)"
            self.is_gpu = False
            self.dtype = torch.float64
    
    def get_device_info(self):
        """Return a string describing the compute device."""
        if not self.available:
            return "PyTorch not available"
        return f"{self.device_name} [{self.device}]"
    
    def _complex_pow_n(self, zr, zi, n):
        """Compute z^n for integer n using repeated squaring (vectorized)."""
        result_r = torch.ones_like(zr)
        result_i = torch.zeros_like(zi)
        base_r, base_i = zr.clone(), zi.clone()
        
        while n > 0:
            if n % 2 == 1:
                new_r = result_r * base_r - result_i * base_i
                new_i = result_r * base_i + result_i * base_r
                result_r, result_i = new_r, new_i
            new_r = base_r * base_r - base_i * base_i
            new_i = 2 * base_r * base_i
            base_r, base_i = new_r, new_i
            n //= 2
        
        return result_r, result_i
    
    def _iterate_function(self, zr, zi, cr, ci, func_id, mask=None):
        """
        Apply one iteration of the selected function (vectorized).
        
        Args:
            zr, zi: Real and imaginary parts of z (tensors)
            cr, ci: Real and imaginary parts of c (tensors)
            func_id: Which function to use
            mask: Optional boolean mask - only compute for True positions
        
        Returns:
            (new_zr, new_zi): The next z values
        """
        if func_id == self.FUNC_Z2_PLUS_C:
            # z² + c
            return zr * zr - zi * zi + cr, 2 * zr * zi + ci
        
        elif func_id == self.FUNC_Z3_PLUS_C:
            # z³ + c
            zr2 = zr * zr
            zi2 = zi * zi
            return zr * (zr2 - 3 * zi2) + cr, zi * (3 * zr2 - zi2) + ci
        
        elif func_id == self.FUNC_Z4_PLUS_C:
            # z⁴ + c
            zr2 = zr * zr
            zi2 = zi * zi
            zr4 = zr2 * zr2 - 6 * zr2 * zi2 + zi2 * zi2
            zi4 = 4 * zr * zi * (zr2 - zi2)
            return zr4 + cr, zi4 + ci
        
        elif func_id == self.FUNC_Z5_PLUS_C:
            # z⁵ + c
            pr, pi = self._complex_pow_n(zr, zi, 5)
            return pr + cr, pi + ci
        
        elif func_id == self.FUNC_TRICORN:
            # (z̄)² + c = conjugate squared
            return zr * zr - zi * zi + cr, -2 * zr * zi + ci
        
        elif func_id == self.FUNC_EXP:
            # c·e^z
            exp_zr = torch.exp(torch.clamp(zr, max=700))
            cos_zi = torch.cos(zi)
            sin_zi = torch.sin(zi)
            return cr * exp_zr * cos_zi - ci * exp_zr * sin_zi, cr * exp_zr * sin_zi + ci * exp_zr * cos_zi
        
        elif func_id == self.FUNC_SIN:
            # c·sin(z)
            sin_zr = torch.sin(zr)
            cos_zr = torch.cos(zr)
            sinh_zi = torch.sinh(torch.clamp(zi, -700, 700))
            cosh_zi = torch.cosh(torch.clamp(zi, -700, 700))
            wr = sin_zr * cosh_zi
            wi = cos_zr * sinh_zi
            return cr * wr - ci * wi, cr * wi + ci * wr
        
        elif func_id == self.FUNC_Z2_Z_C:
            # z² + z + c
            return zr * zr - zi * zi + zr + cr, 2 * zr * zi + zi + ci
        
        elif func_id == self.FUNC_Z6_PLUS_C:
            pr, pi = self._complex_pow_n(zr, zi, 6)
            return pr + cr, pi + ci
        
        elif func_id == self.FUNC_Z7_PLUS_C:
            pr, pi = self._complex_pow_n(zr, zi, 7)
            return pr + cr, pi + ci
        
        elif func_id == self.FUNC_Z8_PLUS_C:
            pr, pi = self._complex_pow_n(zr, zi, 8)
            return pr + cr, pi + ci
        
        elif func_id == self.FUNC_Z2_CZ:
            # z² + c·z
            z2r = zr * zr - zi * zi
            z2i = 2 * zr * zi
            czr = cr * zr - ci * zi
            czi = cr * zi + ci * zr
            return z2r + czr, z2i + czi
        
        elif func_id == self.FUNC_Z3_Z_C:
            # z³ + z + c
            zr2 = zr * zr
            zi2 = zi * zi
            z3r = zr * (zr2 - 3 * zi2)
            z3i = zi * (3 * zr2 - zi2)
            return z3r + zr + cr, z3i + zi + ci
        
        elif func_id == self.FUNC_Z4_Z_C:
            # z⁴ + z + c
            pr, pi = self._complex_pow_n(zr, zi, 4)
            return pr + zr + cr, pi + zi + ci
        
        elif func_id == self.FUNC_COS:
            # c·cos(z)
            cos_zr = torch.cos(zr)
            sin_zr = torch.sin(zr)
            sinh_zi = torch.sinh(torch.clamp(zi, -700, 700))
            cosh_zi = torch.cosh(torch.clamp(zi, -700, 700))
            wr = cos_zr * cosh_zi
            wi = -sin_zr * sinh_zi
            return cr * wr - ci * wi, cr * wi + ci * wr
        
        elif func_id == self.FUNC_Z2_MINUS_Z_C:
            # z² - z + c
            return zr * zr - zi * zi - zr + cr, 2 * zr * zi - zi + ci
        
        elif func_id == self.FUNC_CUBIC_JULIA:
            # z·(z² + c) = z³ + c·z
            zr2 = zr * zr
            zi2 = zi * zi
            z3r = zr * (zr2 - 3 * zi2)
            z3i = zi * (3 * zr2 - zi2)
            czr = cr * zr - ci * zi
            czi = cr * zi + ci * zr
            return z3r + czr, z3i + czi
        
        elif func_id == self.FUNC_Z2_C2:
            # z² + c²
            z2r = zr * zr - zi * zi
            z2i = 2 * zr * zi
            c2r = cr * cr - ci * ci
            c2i = 2 * cr * ci
            return z2r + c2r, z2i + c2i
        
        elif func_id == self.FUNC_RATIONAL:
            # (z² + c) / (z² - c)
            z2r = zr * zr - zi * zi
            z2i = 2 * zr * zi
            num_r = z2r + cr
            num_i = z2i + ci
            den_r = z2r - cr
            den_i = z2i - ci
            den_mag2 = den_r * den_r + den_i * den_i
            # Avoid division by zero
            den_mag2 = torch.clamp(den_mag2, min=1e-10)
            return (num_r * den_r + num_i * den_i) / den_mag2, (num_i * den_r - num_r * den_i) / den_mag2
        
        elif func_id == self.FUNC_Z3_MINUS_Z_C:
            # z³ - z + c
            zr2 = zr * zr
            zi2 = zi * zi
            z3r = zr * (zr2 - 3 * zi2)
            z3i = zi * (3 * zr2 - zi2)
            return z3r - zr + cr, z3i - zi + ci
        
        # Default: z² + c
        return zr * zr - zi * zi + cr, 2 * zr * zi + ci
    
    def _is_transcendental(self, func_id):
        """Check if the function is transcendental."""
        return func_id in (self.FUNC_SIN, self.FUNC_COS, self.FUNC_EXP)
    
    def _get_function_degree(self, func_id):
        """Get the polynomial degree for smooth coloring."""
        degrees = {
            self.FUNC_Z2_PLUS_C: 2.0, self.FUNC_TRICORN: 2.0, self.FUNC_Z2_Z_C: 2.0,
            self.FUNC_Z2_CZ: 2.0, self.FUNC_Z2_MINUS_Z_C: 2.0, self.FUNC_Z2_C2: 2.0,
            self.FUNC_RATIONAL: 2.0,
            self.FUNC_Z3_PLUS_C: 3.0, self.FUNC_Z3_Z_C: 3.0, self.FUNC_CUBIC_JULIA: 3.0,
            self.FUNC_Z3_MINUS_Z_C: 3.0,
            self.FUNC_Z4_PLUS_C: 4.0, self.FUNC_Z4_Z_C: 4.0,
            self.FUNC_Z5_PLUS_C: 5.0,
            self.FUNC_Z6_PLUS_C: 6.0,
            self.FUNC_Z7_PLUS_C: 7.0,
            self.FUNC_Z8_PLUS_C: 8.0,
        }
        return degrees.get(func_id, 2.0)
    
    def compute_mandelbrot(self, x_min, x_max, y_min, y_max, width, height, max_iter,
                           func_id=0, escape_radius=2.0):
        """
        Compute the fractal set using GPU acceleration.
        
        All pixels are processed in parallel using tensor operations.
        Uses a fully vectorized approach for maximum GPU throughput.
        
        Args:
            x_min, x_max: Real axis bounds
            y_min, y_max: Imaginary axis bounds
            width, height: Output dimensions
            max_iter: Maximum iterations
            func_id: Function type (default 0 = z² + c)
            escape_radius: Escape threshold
        
        Returns:
            numpy array (height, width) of float64 smooth iteration counts
        """
        if not self.available:
            raise RuntimeError("PyTorch not available")
        
        # Create coordinate grids on device using appropriate dtype
        x = torch.linspace(x_min, x_max, width, device=self.device, dtype=self.dtype)
        y = torch.linspace(y_min, y_max, height, device=self.device, dtype=self.dtype)
        
        # Create 2D meshgrid: cr[y, x] and ci[y, x]
        cr, ci = torch.meshgrid(x, y, indexing='xy')
        cr = cr.T.contiguous()  # Shape: (height, width)
        ci = ci.T.contiguous()
        
        # Initialize z to 0
        zr = torch.zeros_like(cr)
        zi = torch.zeros_like(ci)
        
        # Escape parameters
        is_trans = self._is_transcendental(func_id)
        if is_trans:
            escape_threshold = 50.0
        else:
            escape_r2 = escape_radius * escape_radius
        
        degree = self._get_function_degree(func_id)
        log_escape = np.log(max(escape_radius, 2.0))
        log_degree = np.log(degree)
        
        # Track iteration counts - initialized to max_iter
        iteration_counts = torch.full((height, width), max_iter, device=self.device, dtype=self.dtype)
        
        # Track which pixels have escaped (for iteration counting only)
        not_escaped = torch.ones((height, width), device=self.device, dtype=torch.bool)
        
        # Run all iterations unconditionally for maximum GPU efficiency
        # This wastes some computation but maximizes parallelism
        check_interval = max(16, max_iter // 16)  # Check escape every N iterations
        
        for iteration in range(max_iter):
            # Always compute iteration for all pixels (no masking)
            new_zr, new_zi = self._iterate_function(zr, zi, cr, ci, func_id)
            zr, zi = new_zr, new_zi
            
            # Check escape condition periodically
            if (iteration + 1) % check_interval == 0 or iteration == max_iter - 1:
                if is_trans:
                    has_escaped = (torch.abs(zi) > escape_threshold) | (torch.abs(zr) > escape_threshold)
                else:
                    has_escaped = (zr * zr + zi * zi) > escape_r2
                
                # Update iteration count for newly escaped pixels
                just_escaped = not_escaped & has_escaped
                iteration_counts = torch.where(just_escaped, torch.tensor(iteration + 1, dtype=self.dtype, device=self.device), iteration_counts)
                not_escaped = not_escaped & ~has_escaped
        
        # Compute smooth iteration count
        result = iteration_counts.clone()
        escaped_mask = iteration_counts < max_iter
        
        if escaped_mask.any():
            zn2 = zr ** 2 + zi ** 2
            valid_smooth = escaped_mask & (zn2 > 1.0)
            
            if valid_smooth.any():
                zn2_clamped = torch.clamp(zn2, min=1.0)  # Avoid log(0)
                log_zn = torch.log(zn2_clamped) * 0.5
                nu = torch.log(torch.clamp(log_zn / log_escape, min=1e-10)) / log_degree
                result = torch.where(valid_smooth, result - nu, result)
        
        # Return as numpy array (always float64 for compatibility)
        return result.cpu().to(torch.float64).numpy()
    
    def compute_mandelbrot_partial(self, x_min, x_max, y_min, y_max, width, height, max_iter,
                                    result_array, start_x, start_y, compute_w, compute_h,
                                    func_id=0, escape_radius=2.0):
        """
        Compute fractal for a sub-region, writing into an existing array.
        
        Args:
            x_min, x_max, y_min, y_max: Full image bounds
            width, height: Full image dimensions
            max_iter: Maximum iterations
            result_array: Output array to write into
            start_x, start_y: Top-left corner of sub-region
            compute_w, compute_h: Size of sub-region
            func_id: Function type
            escape_radius: Escape threshold
        """
        if not self.available:
            raise RuntimeError("PyTorch not available")
        
        # Calculate bounds for sub-region
        dx = (x_max - x_min) / width
        dy = (y_max - y_min) / height
        
        sub_x_min = x_min + dx * start_x
        sub_x_max = x_min + dx * (start_x + compute_w)
        sub_y_min = y_min + dy * start_y
        sub_y_max = y_min + dy * (start_y + compute_h)
        
        # Compute sub-region
        sub_result = self.compute_mandelbrot(
            sub_x_min, sub_x_max, sub_y_min, sub_y_max,
            compute_w, compute_h, max_iter, func_id, escape_radius
        )
        
        # Write into result array
        result_array[start_y:start_y+compute_h, start_x:start_x+compute_w] = sub_result
    
    def apply_colormap_smooth(self, data, max_iter, colormap, out):
        """
        Apply a colormap to iteration data with smooth interpolation.
        
        Args:
            data: numpy array of iteration counts
            max_iter: Maximum iteration value
            colormap: Nx3 numpy array of RGB colors (uint8)
            out: Output RGB image array
        """
        if not self.available:
            raise RuntimeError("PyTorch not available")
        
        height, width = data.shape
        num_colors = colormap.shape[0]
        
        # Convert to tensors - must use the appropriate dtype for the device
        data_t = torch.from_numpy(data.astype(np.float32)).to(self.device)
        colormap_t = torch.from_numpy(colormap.astype(np.float32)).to(self.device)
        
        # Create output tensor
        out_t = torch.zeros((height, width, 3), device=self.device, dtype=torch.float32)
        
        # Points in set (black)
        in_set = data_t >= max_iter - 0.5
        
        # Map values to color indices
        fidx = (data_t / max_iter) * (num_colors - 1)
        idx0 = fidx.long().clamp(0, num_colors - 2)
        idx1 = (idx0 + 1).clamp(0, num_colors - 1)
        t = (fidx - idx0.float()).unsqueeze(-1)
        
        # Interpolate colors
        color0 = colormap_t[idx0]
        color1 = colormap_t[idx1]
        out_t = color0 * (1 - t) + color1 * t
        
        # Set in-set pixels to black
        out_t[in_set] = 0
        
        # Copy to output
        out[:] = out_t.cpu().numpy().astype(np.uint8)
    
    def downscale_2x(self, src, dst):
        """
        Downscale an image by 2x using box filter.
        
        Args:
            src: Source image (2*height, 2*width, 3)
            dst: Destination image (height, width, 3)
        """
        if not self.available:
            raise RuntimeError("PyTorch not available")
        
        # Convert to tensor and reorder to (C, H, W)
        src_t = torch.from_numpy(src).to(self.device).float()
        src_t = src_t.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
        
        # Use avg_pool2d for 2x downscaling
        dst_t = torch.nn.functional.avg_pool2d(src_t, kernel_size=2, stride=2)
        
        # Convert back
        dst_t = dst_t.squeeze(0).permute(1, 2, 0)
        dst[:] = dst_t.cpu().numpy().astype(np.uint8)
    
    def warmup(self, colormap):
        """
        Warm up GPU by running small computations.
        
        Args:
            colormap: A colormap array for testing
        """
        if not self.available:
            return
        
        # Small warmup computation
        _ = self.compute_mandelbrot(-2, 1, -1, 1, 32, 32, 32)
        
        # Warmup colormap
        dummy = np.zeros((32, 32, 3), dtype=np.uint8)
        dummy_data = np.zeros((32, 32), dtype=np.float64)
        self.apply_colormap_smooth(dummy_data, 32, colormap, dummy)
        
        # Warmup downscale
        dummy_hi = np.zeros((64, 64, 3), dtype=np.uint8)
        self.downscale_2x(dummy_hi, dummy)
        
        # Sync to ensure warmup completed
        if self.device.type == 'cuda':
            torch.cuda.synchronize()


# Global instance for easy access
_gpu_compute = None


def get_gpu_compute(prefer_gpu=True):
    """
    Get the global GPU compute instance.
    
    Creates the instance on first call.
    
    Args:
        prefer_gpu: If False, force CPU mode
    
    Returns:
        GPUCompute instance
    """
    global _gpu_compute
    if _gpu_compute is None:
        _gpu_compute = GPUCompute(prefer_gpu=prefer_gpu)
    return _gpu_compute


def is_gpu_available():
    """Check if GPU acceleration is available."""
    return TORCH_AVAILABLE and get_gpu_compute().is_gpu


def is_cuda_available():
    """Check if CUDA GPU is available (recommended for GPU mode)."""
    return TORCH_AVAILABLE and get_gpu_compute().is_cuda


def should_default_to_gpu():
    """
    Check if GPU should be enabled by default.
    
    Returns True only for CUDA GPUs where GPU acceleration provides
    a clear benefit. For MPS (Apple Silicon), the Numba CPU implementation
    is typically faster, so we default to CPU.
    """
    if not TORCH_AVAILABLE:
        return False
    gpu = get_gpu_compute()
    # Only default to GPU for CUDA (where it's clearly beneficial)
    return gpu.is_cuda
