"""Fractal computation with Numba JIT - CPU backend."""
import numpy as np
from numba import jit, prange

# Function IDs (must be constants for Numba JIT)
F_Z2, F_Z3, F_Z4, F_Z5, F_TRICORN, F_EXP, F_SIN, F_Z2_Z = 0, 1, 2, 3, 4, 5, 6, 7
F_Z6, F_Z7, F_Z8, F_Z2_CZ, F_Z3_Z, F_Z4_Z, F_COS = 8, 9, 10, 11, 12, 13, 14
F_Z2_MINUS_Z, F_CUBIC, F_Z2_C2, F_RATIONAL, F_Z3_MINUS_Z = 15, 16, 17, 18, 19


@jit(nopython=True, cache=True)
def complex_pow_n(zr, zi, n):
    """Compute z^n using repeated squaring."""
    if n == 0: return 1.0, 0.0
    if n == 1: return zr, zi
    result_r, result_i, base_r, base_i = 1.0, 0.0, zr, zi
    while n > 0:
        if n % 2 == 1:
            result_r, result_i = result_r * base_r - result_i * base_i, result_r * base_i + result_i * base_r
        base_r, base_i = base_r * base_r - base_i * base_i, 2 * base_r * base_i
        n //= 2
    return result_r, result_i


@jit(nopython=True, cache=True)
def iterate_function(zr, zi, cr, ci, func_id):
    """Apply one iteration of the selected function."""
    if func_id == F_Z2:  # z² + c
        return zr * zr - zi * zi + cr, 2 * zr * zi + ci
    elif func_id == F_Z3:  # z³ + c
        zr2, zi2 = zr * zr, zi * zi
        return zr * (zr2 - 3 * zi2) + cr, zi * (3 * zr2 - zi2) + ci
    elif func_id == F_Z4:  # z⁴ + c
        zr2, zi2 = zr * zr, zi * zi
        return zr2 * zr2 - 6 * zr2 * zi2 + zi2 * zi2 + cr, 4 * zr * zi * (zr2 - zi2) + ci
    elif func_id == F_Z5:  # z⁵ + c
        pr, pi = complex_pow_n(zr, zi, 5)
        return pr + cr, pi + ci
    elif func_id == F_TRICORN:  # (z̄)² + c
        return zr * zr - zi * zi + cr, -2 * zr * zi + ci
    elif func_id == F_EXP:  # c·e^z
        exp_zr = np.exp(min(zr, 700))
        cos_zi, sin_zi = np.cos(zi), np.sin(zi)
        return cr * exp_zr * cos_zi - ci * exp_zr * sin_zi, cr * exp_zr * sin_zi + ci * exp_zr * cos_zi
    elif func_id == F_SIN:  # c·sin(z)
        sin_zr, cos_zr = np.sin(zr), np.cos(zr)
        sinh_zi, cosh_zi = np.sinh(min(zi, 700)), np.cosh(min(zi, 700))
        wr, wi = sin_zr * cosh_zi, cos_zr * sinh_zi
        return cr * wr - ci * wi, cr * wi + ci * wr
    elif func_id == F_Z2_Z:  # z² + z + c
        return zr * zr - zi * zi + zr + cr, 2 * zr * zi + zi + ci
    elif func_id == F_Z6:  # z⁶ + c
        pr, pi = complex_pow_n(zr, zi, 6)
        return pr + cr, pi + ci
    elif func_id == F_Z7:  # z⁷ + c
        pr, pi = complex_pow_n(zr, zi, 7)
        return pr + cr, pi + ci
    elif func_id == F_Z8:  # z⁸ + c
        pr, pi = complex_pow_n(zr, zi, 8)
        return pr + cr, pi + ci
    elif func_id == F_Z2_CZ:  # z² + c·z
        z2r, z2i = zr * zr - zi * zi, 2 * zr * zi
        return z2r + cr * zr - ci * zi, z2i + cr * zi + ci * zr
    elif func_id == F_Z3_Z:  # z³ + z + c
        zr2, zi2 = zr * zr, zi * zi
        return zr * (zr2 - 3 * zi2) + zr + cr, zi * (3 * zr2 - zi2) + zi + ci
    elif func_id == F_Z4_Z:  # z⁴ + z + c
        pr, pi = complex_pow_n(zr, zi, 4)
        return pr + zr + cr, pi + zi + ci
    elif func_id == F_COS:  # c·cos(z)
        cos_zr, sin_zr = np.cos(zr), np.sin(zr)
        sinh_zi, cosh_zi = np.sinh(min(zi, 700)), np.cosh(min(zi, 700))
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
        den_mag2 = den_r * den_r + den_i * den_i
        if den_mag2 < 1e-10: return 1e10, 0.0
        return (num_r * den_r + num_i * den_i) / den_mag2, (num_i * den_r - num_r * den_i) / den_mag2
    elif func_id == F_Z3_MINUS_Z:  # z³ - z + c
        zr2, zi2 = zr * zr, zi * zi
        return zr * (zr2 - 3 * zi2) - zr + cr, zi * (3 * zr2 - zi2) - zi + ci
    return zr * zr - zi * zi + cr, 2 * zr * zi + ci


@jit(nopython=True, cache=True)
def get_function_degree(func_id):
    """Get the polynomial degree for smooth coloring."""
    degrees = (2, 3, 4, 5, 2, 2, 2, 2, 6, 7, 8, 2, 3, 4, 2, 2, 3, 2, 2, 3)
    return float(degrees[func_id]) if func_id < 20 else 2.0


@jit(nopython=True, cache=True)
def is_transcendental(func_id):
    """Check if function is transcendental (sin, cos, exp)."""
    return func_id == F_SIN or func_id == F_COS or func_id == F_EXP


@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def compute_mandelbrot(x_min, x_max, y_min, y_max, width, height, max_iter,
                       func_id=0, escape_radius=2.0,
                       julia_mode=False, julia_c_real=0.0, julia_c_imag=0.0):
    """Compute fractal with smooth coloring for a given region."""
    result = np.zeros((height, width), dtype=np.float64)
    
    # For transcendental functions, use different escape criteria
    # sinh/cosh grow exponentially, so we check imaginary part separately
    is_trans = is_transcendental(func_id)
    if is_trans:
        # For sin/cos/exp, escape when |Im(z)| > threshold (exponential growth)
        escape_threshold = 50.0  # sinh/cosh grow exponentially past this
        escape_r2 = 1e10  # Effectively unused for transcendental
    else:
        escape_threshold = 1e10  # Not used for polynomial
        escape_r2 = escape_radius * escape_radius
    
    log_escape = np.log(max(escape_radius, 2.0))
    
    # Degree for smooth coloring
    degree = get_function_degree(func_id)
    log_degree = np.log(degree)
    
    # Use float64 for better precision at deep zoom levels
    dx = np.float64(x_max - x_min) / width
    dy = np.float64(y_max - y_min) / height
    
    for py in prange(height):
        y0 = np.float64(y_min) + dy * py
        for px in range(width):
            x0 = np.float64(x_min) + dx * px
            
            # Julia mode: point is z₀, c is fixed
            # Mandelbrot mode: point is c, z₀ is 0
            if julia_mode:
                zr, zi = np.float64(x0), np.float64(y0)
                cr, ci = np.float64(julia_c_real), np.float64(julia_c_imag)
            else:
                zr, zi = np.float64(0.0), np.float64(0.0)
                cr, ci = np.float64(x0), np.float64(y0)
            
            iteration = 0
            
            # Main iteration loop with escape condition based on function type
            if is_trans:
                # For transcendental: escape when |Im(z)| gets large
                while abs(zi) <= escape_threshold and abs(zr) <= escape_threshold and iteration < max_iter:
                    zr, zi = iterate_function(zr, zi, cr, ci, func_id)
                    iteration += 1
            else:
                # For polynomial: standard |z|^2 escape
                while zr * zr + zi * zi <= escape_r2 and iteration < max_iter:
                    zr, zi = iterate_function(zr, zi, cr, ci, func_id)
                    iteration += 1
            
            if iteration < max_iter:
                # Smooth coloring: fractional iteration count
                zn2 = zr * zr + zi * zi
                if zn2 > 1.0:
                    log_zn = np.log(zn2) * 0.5
                    nu = np.log(log_zn / log_escape) / log_degree
                    result[py, px] = iteration + 1 - nu
                else:
                    result[py, px] = iteration
            else:
                result[py, px] = max_iter
    
    return result


@jit(nopython=True, parallel=True, cache=True)
def compute_mandelbrot_partial(x_min, x_max, y_min, y_max, width, height, max_iter,
                                result, start_x, start_y, compute_w, compute_h,
                                func_id=0, escape_radius=2.0,
                                julia_mode=False, julia_c_real=0.0, julia_c_imag=0.0):
    """Compute fractal for a sub-region (used for incremental panning)."""
    is_trans = is_transcendental(func_id)
    escape_threshold = 50.0 if is_trans else 1e10
    escape_r2 = 1e10 if is_trans else escape_radius * escape_radius
    log_escape, degree = np.log(max(escape_radius, 2.0)), get_function_degree(func_id)
    log_degree = np.log(degree)
    dx, dy = np.float64(x_max - x_min) / width, np.float64(y_max - y_min) / height
    
    for py in prange(compute_h):
        actual_py = start_y + py
        y0 = np.float64(y_min) + dy * actual_py
        for px in range(compute_w):
            actual_px = start_x + px
            x0 = np.float64(x_min) + dx * actual_px
            if julia_mode:
                zr, zi = np.float64(x0), np.float64(y0)
                cr, ci = np.float64(julia_c_real), np.float64(julia_c_imag)
            else:
                zr, zi = np.float64(0.0), np.float64(0.0)
                cr, ci = np.float64(x0), np.float64(y0)
            iteration = 0
            if is_trans:
                while abs(zi) <= escape_threshold and abs(zr) <= escape_threshold and iteration < max_iter:
                    zr, zi = iterate_function(zr, zi, cr, ci, func_id)
                    iteration += 1
            else:
                while zr * zr + zi * zi <= escape_r2 and iteration < max_iter:
                    zr, zi = iterate_function(zr, zi, cr, ci, func_id)
                    iteration += 1
            if iteration < max_iter:
                zn2 = zr * zr + zi * zi
                if zn2 > 1.0:
                    result[actual_py, actual_px] = iteration + 1 - np.log(np.log(zn2) * 0.5 / log_escape) / log_degree
                else:
                    result[actual_py, actual_px] = iteration
            else:
                result[actual_py, actual_px] = max_iter


@jit(nopython=True, parallel=True, cache=True)
def apply_colormap_smooth(data, max_iter, colormap, out):
    """Apply colormap with smooth interpolation."""
    height, width = data.shape
    num_colors = colormap.shape[0]
    
    for py in prange(height):
        for px in range(width):
            val = data[py, px]
            if val >= max_iter - 0.5:
                # Points in the set are black
                out[py, px, 0] = 0
                out[py, px, 1] = 0
                out[py, px, 2] = 0
            else:
                # Map value to color index with interpolation
                fidx = (val / max_iter) * (num_colors - 1)
                idx0 = int(fidx)
                idx1 = min(idx0 + 1, num_colors - 1)
                t = fidx - idx0
                
                # Linear interpolation between adjacent colors
                out[py, px, 0] = np.uint8(colormap[idx0, 0] * (1 - t) + colormap[idx1, 0] * t)
                out[py, px, 1] = np.uint8(colormap[idx0, 1] * (1 - t) + colormap[idx1, 1] * t)
                out[py, px, 2] = np.uint8(colormap[idx0, 2] * (1 - t) + colormap[idx1, 2] * t)


@jit(nopython=True, parallel=True, cache=True)
def downscale_2x(src, dst):
    """Downscale image by 2x using box filter (4-pixel average) for anti-aliasing."""
    height, width = dst.shape[:2]
    for y in prange(height):
        y2 = y * 2
        for x in range(width):
            x2 = x * 2
            for c in range(3):
                dst[y, x, c] = (int(src[y2, x2, c]) + int(src[y2, x2 + 1, c]) +
                                int(src[y2 + 1, x2, c]) + int(src[y2 + 1, x2 + 1, c])) // 4


def warmup_jit(colormap):
    """Pre-compile Numba functions to avoid delay on first use."""
    _ = compute_mandelbrot(-2, 1, -1, 1, 10, 10, 10)
    dummy, dummy_hi = np.zeros((10, 10, 3), dtype=np.uint8), np.zeros((20, 20, 3), dtype=np.uint8)
    apply_colormap_smooth(np.zeros((10, 10), dtype=np.float64), 10, colormap, dummy)
    downscale_2x(dummy_hi, dummy)


# Custom Formula Support (non-JIT)
import cmath, math, re

_SAFE_MATH_NAMESPACE = {
    '__builtins__': {}, 'sin': cmath.sin, 'cos': cmath.cos, 'tan': cmath.tan,
    'exp': cmath.exp, 'log': cmath.log, 'sqrt': cmath.sqrt, 'abs': abs,
    'conj': lambda x: complex(x.real, -x.imag), 'sinh': cmath.sinh, 'cosh': cmath.cosh,
    'tanh': cmath.tanh, 'asin': cmath.asin, 'acos': cmath.acos, 'atan': cmath.atan,
    'pi': cmath.pi, 'e': cmath.e, 'i': 1j, 'j': 1j,
}

_MATH_FUNCS = ['sin', 'cos', 'tan', 'exp', 'log', 'sqrt', 'abs', 'conj',
               'sinh', 'cosh', 'tanh', 'asin', 'acos', 'atan']


def prepare_custom_formula(formula_str):
    """Convert mathematical notation to Python syntax for custom formulas."""
    f = formula_str.strip().replace('^', '**')
    f = re.sub(r'(\d)([zc])', r'\1*\2', f)
    f = re.sub(r'([zc])([zc])', r'\1*\2', f)
    f = re.sub(r'\)([zc(])', r')*\1', f)
    for func in _MATH_FUNCS:
        f = f.replace(func + '(', f'__FUNC_{func}__(')
    f = re.sub(r'([zc])(\()', r'\1*\2', f)
    for func in _MATH_FUNCS:
        f = f.replace(f'__FUNC_{func}__(', func + '(')
    return f


def eval_custom_formula(formula_str, z, c):
    """Evaluate a custom formula with given z and c values."""
    namespace = _SAFE_MATH_NAMESPACE.copy()
    namespace['z'], namespace['c'] = z, c
    try:
        return eval(formula_str, namespace)
    except Exception:
        return complex(1e10, 0)  # Return large value to escape


def compute_mandelbrot_custom(x_min, x_max, y_min, y_max, width, height, max_iter,
                               custom_formula, escape_radius=2.0,
                               julia_mode=False, julia_c_real=0.0, julia_c_imag=0.0):
    """Compute fractal with custom formula (non-JIT, slower but flexible)."""
    result = np.zeros((height, width), dtype=np.float64)
    escape_r2, log_escape = escape_radius ** 2, math.log(max(escape_radius, 2.0))
    log_degree = math.log(2.0)  # Default degree
    dx, dy = (x_max - x_min) / width, (y_max - y_min) / height
    julia_c = complex(julia_c_real, julia_c_imag)
    
    for py in range(height):
        y0 = y_min + dy * py
        for px in range(width):
            x0 = x_min + dx * px
            z = complex(x0, y0) if julia_mode else complex(0, 0)
            c = julia_c if julia_mode else complex(x0, y0)
            iteration = 0
            try:
                while abs(z) ** 2 <= escape_r2 and iteration < max_iter:
                    z = eval_custom_formula(custom_formula, z, c)
                    iteration += 1
                    if math.isnan(z.real) or math.isnan(z.imag) or math.isinf(z.real) or math.isinf(z.imag):
                        break
            except Exception:
                pass
            if iteration < max_iter:
                zn2 = abs(z) ** 2
                if zn2 > 1.0:
                    result[py, px] = iteration + 1 - math.log(math.log(zn2) * 0.5 / log_escape) / log_degree
                else:
                    result[py, px] = iteration
            else:
                result[py, px] = max_iter
    return result
