"""GPU-accelerated fractal computation using PyTorch."""
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

# Function IDs (must match compute.py)
F_Z2, F_Z3, F_Z4, F_Z5, F_TRICORN, F_EXP, F_SIN, F_Z2_Z = 0, 1, 2, 3, 4, 5, 6, 7
F_Z6, F_Z7, F_Z8, F_Z2_CZ, F_Z3_Z, F_Z4_Z, F_COS = 8, 9, 10, 11, 12, 13, 14
F_Z2_MINUS_Z, F_CUBIC, F_Z2_C2, F_RATIONAL, F_Z3_MINUS_Z = 15, 16, 17, 18, 19


class GPUCompute:
    """GPU-accelerated fractal computation (CUDA/MPS/CPU fallback)."""
    
    def __init__(self, prefer_gpu=True):
        self.available = TORCH_AVAILABLE
        self.device = None
        self.device_name = "None"
        self.is_gpu = self.is_cuda = self.is_mps = False
        self.dtype = torch.float32 if TORCH_AVAILABLE else None
        
        if not TORCH_AVAILABLE:
            return
        
        if prefer_gpu:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                self.device_name = torch.cuda.get_device_name(0)
                self.is_gpu = self.is_cuda = True
                self.dtype = torch.float64
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device("mps")
                self.device_name = "Apple Silicon GPU (MPS)"
                self.is_gpu = self.is_mps = True
            else:
                self.device = torch.device("cpu")
                self.device_name = "CPU (PyTorch)"
                self.dtype = torch.float64
        else:
            self.device = torch.device("cpu")
            self.device_name = "CPU (PyTorch)"
            self.dtype = torch.float64
    
    def get_device_info(self):
        return "PyTorch not available" if not self.available else f"{self.device_name} [{self.device}]"
    
    def _complex_pow_n(self, zr, zi, n):
        """Compute z^n using repeated squaring (vectorized)."""
        result_r, result_i = torch.ones_like(zr), torch.zeros_like(zi)
        base_r, base_i = zr.clone(), zi.clone()
        while n > 0:
            if n % 2 == 1:
                result_r, result_i = result_r * base_r - result_i * base_i, result_r * base_i + result_i * base_r
            base_r, base_i = base_r * base_r - base_i * base_i, 2 * base_r * base_i
            n //= 2
        return result_r, result_i
    
    def _iterate_function(self, zr, zi, cr, ci, func_id, mask=None):
        """Apply one iteration of the selected function (vectorized)."""
        if func_id == F_Z2:  # z² + c
            return zr * zr - zi * zi + cr, 2 * zr * zi + ci
        elif func_id == F_Z3:  # z³ + c
            zr2, zi2 = zr * zr, zi * zi
            return zr * (zr2 - 3 * zi2) + cr, zi * (3 * zr2 - zi2) + ci
        elif func_id == F_Z4:  # z⁴ + c
            zr2, zi2 = zr * zr, zi * zi
            return zr2 * zr2 - 6 * zr2 * zi2 + zi2 * zi2 + cr, 4 * zr * zi * (zr2 - zi2) + ci
        elif func_id == F_Z5:  # z⁵ + c
            pr, pi = self._complex_pow_n(zr, zi, 5)
            return pr + cr, pi + ci
        elif func_id == F_TRICORN:  # (z̄)² + c
            return zr * zr - zi * zi + cr, -2 * zr * zi + ci
        elif func_id == F_EXP:  # c·e^z
            exp_zr = torch.exp(torch.clamp(zr, max=700))
            cos_zi, sin_zi = torch.cos(zi), torch.sin(zi)
            return cr * exp_zr * cos_zi - ci * exp_zr * sin_zi, cr * exp_zr * sin_zi + ci * exp_zr * cos_zi
        elif func_id == F_SIN:  # c·sin(z)
            sin_zr, cos_zr = torch.sin(zr), torch.cos(zr)
            sinh_zi, cosh_zi = torch.sinh(torch.clamp(zi, -700, 700)), torch.cosh(torch.clamp(zi, -700, 700))
            wr, wi = sin_zr * cosh_zi, cos_zr * sinh_zi
            return cr * wr - ci * wi, cr * wi + ci * wr
        elif func_id == F_Z2_Z:  # z² + z + c
            return zr * zr - zi * zi + zr + cr, 2 * zr * zi + zi + ci
        elif func_id == F_Z6:  # z⁶ + c
            pr, pi = self._complex_pow_n(zr, zi, 6)
            return pr + cr, pi + ci
        elif func_id == F_Z7:  # z⁷ + c
            pr, pi = self._complex_pow_n(zr, zi, 7)
            return pr + cr, pi + ci
        elif func_id == F_Z8:  # z⁸ + c
            pr, pi = self._complex_pow_n(zr, zi, 8)
            return pr + cr, pi + ci
        elif func_id == F_Z2_CZ:  # z² + c·z
            z2r, z2i = zr * zr - zi * zi, 2 * zr * zi
            return z2r + cr * zr - ci * zi, z2i + cr * zi + ci * zr
        elif func_id == F_Z3_Z:  # z³ + z + c
            zr2, zi2 = zr * zr, zi * zi
            return zr * (zr2 - 3 * zi2) + zr + cr, zi * (3 * zr2 - zi2) + zi + ci
        elif func_id == F_Z4_Z:  # z⁴ + z + c
            pr, pi = self._complex_pow_n(zr, zi, 4)
            return pr + zr + cr, pi + zi + ci
        elif func_id == F_COS:  # c·cos(z)
            cos_zr, sin_zr = torch.cos(zr), torch.sin(zr)
            sinh_zi, cosh_zi = torch.sinh(torch.clamp(zi, -700, 700)), torch.cosh(torch.clamp(zi, -700, 700))
            wr, wi = cos_zr * cosh_zi, -sin_zr * sinh_zi
            return cr * wr - ci * wi, cr * wi + ci * wr
        elif func_id == F_Z2_MINUS_Z:  # z² - z + c
            return zr * zr - zi * zi - zr + cr, 2 * zr * zi - zi + ci
        elif func_id == F_CUBIC:  # z³ + c·z
            zr2, zi2 = zr * zr, zi * zi
            z3r, z3i = zr * (zr2 - 3 * zi2), zi * (3 * zr2 - zi2)
            return z3r + cr * zr - ci * zi, z3i + cr * zi + ci * zr
        elif func_id == F_Z2_C2:  # z² + c²
            return zr * zr - zi * zi + cr * cr - ci * ci, 2 * zr * zi + 2 * cr * ci
        elif func_id == F_RATIONAL:  # (z² + c) / (z² - c)
            z2r, z2i = zr * zr - zi * zi, 2 * zr * zi
            num_r, num_i = z2r + cr, z2i + ci
            den_r, den_i = z2r - cr, z2i - ci
            den_mag2 = torch.clamp(den_r * den_r + den_i * den_i, min=1e-10)
            return (num_r * den_r + num_i * den_i) / den_mag2, (num_i * den_r - num_r * den_i) / den_mag2
        elif func_id == F_Z3_MINUS_Z:  # z³ - z + c
            zr2, zi2 = zr * zr, zi * zi
            return zr * (zr2 - 3 * zi2) - zr + cr, zi * (3 * zr2 - zi2) - zi + ci
        return zr * zr - zi * zi + cr, 2 * zr * zi + ci

    def _is_transcendental(self, func_id):
        return func_id in (F_SIN, F_COS, F_EXP)

    def _get_function_degree(self, func_id):
        degrees = (2, 3, 4, 5, 2, 2, 2, 2, 6, 7, 8, 2, 3, 4, 2, 2, 3, 2, 2, 3)
        return float(degrees[func_id]) if func_id < 20 else 2.0

    def compute_mandelbrot(self, x_min, x_max, y_min, y_max, width, height, max_iter,
                           func_id=0, escape_radius=2.0,
                           julia_mode=False, julia_c_real=0.0, julia_c_imag=0.0):
        """Compute fractal using GPU acceleration."""
        if not self.available:
            raise RuntimeError("PyTorch not available")
        
        # Create coordinate grids on device using appropriate dtype
        x = torch.linspace(x_min, x_max, width, device=self.device, dtype=self.dtype)
        y = torch.linspace(y_min, y_max, height, device=self.device, dtype=self.dtype)
        
        # Create 2D meshgrid
        grid_x, grid_y = torch.meshgrid(x, y, indexing='xy')
        grid_x = grid_x.T.contiguous()  # Shape: (height, width)
        grid_y = grid_y.T.contiguous()
        
        # Julia mode: points are z₀, c is fixed
        # Mandelbrot mode: points are c, z₀ is 0
        if julia_mode:
            zr = grid_x.clone()
            zi = grid_y.clone()
            cr = torch.full_like(grid_x, julia_c_real)
            ci = torch.full_like(grid_y, julia_c_imag)
        else:
            zr = torch.zeros_like(grid_x)
            zi = torch.zeros_like(grid_y)
            cr = grid_x.clone()
            ci = grid_y.clone()
        
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
                                    func_id=0, escape_radius=2.0,
                                    julia_mode=False, julia_c_real=0.0, julia_c_imag=0.0):
        """Compute fractal for a sub-region."""
        if not self.available:
            raise RuntimeError("PyTorch not available")
        dx, dy = (x_max - x_min) / width, (y_max - y_min) / height
        sub_result = self.compute_mandelbrot(
            x_min + dx * start_x, x_min + dx * (start_x + compute_w),
            y_min + dy * start_y, y_min + dy * (start_y + compute_h),
            compute_w, compute_h, max_iter, func_id, escape_radius,
            julia_mode, julia_c_real, julia_c_imag
        )
        result_array[start_y:start_y+compute_h, start_x:start_x+compute_w] = sub_result
    
    def apply_colormap_smooth(self, data, max_iter, colormap, out):
        """Apply colormap with smooth interpolation."""
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
        """Downscale image by 2x using box filter."""
        if not self.available:
            raise RuntimeError("PyTorch not available")
        src_t = torch.from_numpy(src).to(self.device).float().permute(2, 0, 1).unsqueeze(0)
        dst_t = torch.nn.functional.avg_pool2d(src_t, kernel_size=2, stride=2).squeeze(0).permute(1, 2, 0)
        dst[:] = dst_t.cpu().numpy().astype(np.uint8)
    
    def warmup(self, colormap):
        """Warm up GPU by running small computations."""
        if not self.available:
            return
        _ = self.compute_mandelbrot(-2, 1, -1, 1, 32, 32, 32)
        dummy, dummy_hi = np.zeros((32, 32, 3), dtype=np.uint8), np.zeros((64, 64, 3), dtype=np.uint8)
        self.apply_colormap_smooth(np.zeros((32, 32), dtype=np.float64), 32, colormap, dummy)
        self.downscale_2x(dummy_hi, dummy)
        if self.device.type == 'cuda':
            torch.cuda.synchronize()


_gpu_compute = None


def get_gpu_compute(prefer_gpu=True):
    """Get the global GPU compute instance (creates on first call)."""
    global _gpu_compute
    if _gpu_compute is None:
        _gpu_compute = GPUCompute(prefer_gpu=prefer_gpu)
    return _gpu_compute


def is_gpu_available():
    """Check if GPU acceleration is available."""
    return TORCH_AVAILABLE and get_gpu_compute().is_gpu


def is_cuda_available():
    return TORCH_AVAILABLE and get_gpu_compute().is_cuda


def should_default_to_gpu():
    """Return True only for CUDA GPUs where GPU provides clear benefit."""
    return TORCH_AVAILABLE and get_gpu_compute().is_cuda
