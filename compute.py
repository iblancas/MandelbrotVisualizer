"""
Mandelbrot and fractal computation functions using Numba JIT compilation.

This module contains all the performance-critical computation functions
that are JIT-compiled for speed. These functions handle:
- Fractal iteration computation (full and partial) with multiple function types
- Colormap application with smooth interpolation
- Image downscaling for anti-aliasing

Supported iteration functions:
- 0: z² + c (standard Mandelbrot)
- 1: z³ + c (Multibrot degree 3)
- 2: z⁴ + c (Multibrot degree 4)
- 3: z⁵ + c (Multibrot degree 5)
- 4: (z̄)² + c (Tricorn / Mandelbar)
- 5: c·e^z (Exponential map)
- 6: c·sin(z) (Sine fractal)
- 7: z² + z + c (Variant)
- 8: z⁶ + c
- 9: z⁷ + c
- 10: z⁸ + c
- 11: z² + c·z (Variant)
- 12: z³ + z + c
- 13: z⁴ + z + c
- 14: c·cos(z)
- 15: z² - z + c
- 16: z·(z² + c) [Cubic Julia variant]
- 17: z² + c² [Square c]
- 18: (z²+c)/(z²-c) [Rational]
- 19: z³ - z + c
"""

import numpy as np
from numba import jit, prange


# Function IDs
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


@jit(nopython=True, cache=True)
def complex_pow_n(zr, zi, n):
    """Compute z^n for integer n using repeated squaring."""
    if n == 0:
        return 1.0, 0.0
    if n == 1:
        return zr, zi
    
    # Start with z
    result_r, result_i = 1.0, 0.0
    base_r, base_i = zr, zi
    
    while n > 0:
        if n % 2 == 1:
            # Multiply result by base
            new_r = result_r * base_r - result_i * base_i
            new_i = result_r * base_i + result_i * base_r
            result_r, result_i = new_r, new_i
        # Square the base
        new_r = base_r * base_r - base_i * base_i
        new_i = 2 * base_r * base_i
        base_r, base_i = new_r, new_i
        n //= 2
    
    return result_r, result_i


@jit(nopython=True, cache=True)
def iterate_function(zr, zi, cr, ci, func_id):
    """
    Apply one iteration of the selected function.
    
    Args:
        zr, zi: Real and imaginary parts of z
        cr, ci: Real and imaginary parts of c (parameter)
        func_id: Which function to use (see FUNC_* constants)
    
    Returns:
        (new_zr, new_zi): The next z value
    """
    if func_id == FUNC_Z2_PLUS_C:
        # z² + c
        return zr * zr - zi * zi + cr, 2 * zr * zi + ci
    
    elif func_id == FUNC_Z3_PLUS_C:
        # z³ + c
        zr2 = zr * zr
        zi2 = zi * zi
        return zr * (zr2 - 3 * zi2) + cr, zi * (3 * zr2 - zi2) + ci
    
    elif func_id == FUNC_Z4_PLUS_C:
        # z⁴ + c
        zr2 = zr * zr
        zi2 = zi * zi
        zr4 = zr2 * zr2 - 6 * zr2 * zi2 + zi2 * zi2
        zi4 = 4 * zr * zi * (zr2 - zi2)
        return zr4 + cr, zi4 + ci
    
    elif func_id == FUNC_Z5_PLUS_C:
        # z⁵ + c
        pr, pi = complex_pow_n(zr, zi, 5)
        return pr + cr, pi + ci
    
    elif func_id == FUNC_TRICORN:
        # (z̄)² + c = (zr - i·zi)² + c
        return zr * zr - zi * zi + cr, -2 * zr * zi + ci
    
    elif func_id == FUNC_EXP:
        # c·e^z
        exp_zr = np.exp(min(zr, 700))  # Prevent overflow
        cos_zi = np.cos(zi)
        sin_zi = np.sin(zi)
        return cr * exp_zr * cos_zi - ci * exp_zr * sin_zi, cr * exp_zr * sin_zi + ci * exp_zr * cos_zi
    
    elif func_id == FUNC_SIN:
        # c·sin(z)
        sin_zr = np.sin(zr)
        cos_zr = np.cos(zr)
        sinh_zi = np.sinh(min(zi, 700))
        cosh_zi = np.cosh(min(zi, 700))
        wr = sin_zr * cosh_zi
        wi = cos_zr * sinh_zi
        return cr * wr - ci * wi, cr * wi + ci * wr
    
    elif func_id == FUNC_Z2_Z_C:
        # z² + z + c
        return zr * zr - zi * zi + zr + cr, 2 * zr * zi + zi + ci
    
    elif func_id == FUNC_Z6_PLUS_C:
        # z⁶ + c
        pr, pi = complex_pow_n(zr, zi, 6)
        return pr + cr, pi + ci
    
    elif func_id == FUNC_Z7_PLUS_C:
        # z⁷ + c
        pr, pi = complex_pow_n(zr, zi, 7)
        return pr + cr, pi + ci
    
    elif func_id == FUNC_Z8_PLUS_C:
        # z⁸ + c
        pr, pi = complex_pow_n(zr, zi, 8)
        return pr + cr, pi + ci
    
    elif func_id == FUNC_Z2_CZ:
        # z² + c·z
        z2r = zr * zr - zi * zi
        z2i = 2 * zr * zi
        czr = cr * zr - ci * zi
        czi = cr * zi + ci * zr
        return z2r + czr, z2i + czi
    
    elif func_id == FUNC_Z3_Z_C:
        # z³ + z + c
        zr2 = zr * zr
        zi2 = zi * zi
        z3r = zr * (zr2 - 3 * zi2)
        z3i = zi * (3 * zr2 - zi2)
        return z3r + zr + cr, z3i + zi + ci
    
    elif func_id == FUNC_Z4_Z_C:
        # z⁴ + z + c
        pr, pi = complex_pow_n(zr, zi, 4)
        return pr + zr + cr, pi + zi + ci
    
    elif func_id == FUNC_COS:
        # c·cos(z)
        cos_zr = np.cos(zr)
        sin_zr = np.sin(zr)
        sinh_zi = np.sinh(min(zi, 700))
        cosh_zi = np.cosh(min(zi, 700))
        wr = cos_zr * cosh_zi
        wi = -sin_zr * sinh_zi
        return cr * wr - ci * wi, cr * wi + ci * wr
    
    elif func_id == FUNC_Z2_MINUS_Z_C:
        # z² - z + c
        return zr * zr - zi * zi - zr + cr, 2 * zr * zi - zi + ci
    
    elif func_id == FUNC_CUBIC_JULIA:
        # z·(z² + c) = z³ + c·z
        zr2 = zr * zr
        zi2 = zi * zi
        z3r = zr * (zr2 - 3 * zi2)
        z3i = zi * (3 * zr2 - zi2)
        czr = cr * zr - ci * zi
        czi = cr * zi + ci * zr
        return z3r + czr, z3i + czi
    
    elif func_id == FUNC_Z2_C2:
        # z² + c²
        z2r = zr * zr - zi * zi
        z2i = 2 * zr * zi
        c2r = cr * cr - ci * ci
        c2i = 2 * cr * ci
        return z2r + c2r, z2i + c2i
    
    elif func_id == FUNC_RATIONAL:
        # (z² + c) / (z² - c)
        z2r = zr * zr - zi * zi
        z2i = 2 * zr * zi
        # numerator = z² + c
        num_r = z2r + cr
        num_i = z2i + ci
        # denominator = z² - c
        den_r = z2r - cr
        den_i = z2i - ci
        # complex division
        den_mag2 = den_r * den_r + den_i * den_i
        if den_mag2 < 1e-10:
            return 1e10, 0.0  # Escape immediately
        return (num_r * den_r + num_i * den_i) / den_mag2, (num_i * den_r - num_r * den_i) / den_mag2
    
    elif func_id == FUNC_Z3_MINUS_Z_C:
        # z³ - z + c
        zr2 = zr * zr
        zi2 = zi * zi
        z3r = zr * (zr2 - 3 * zi2)
        z3i = zi * (3 * zr2 - zi2)
        return z3r - zr + cr, z3i - zi + ci
    
    # Default: z² + c
    return zr * zr - zi * zi + cr, 2 * zr * zi + ci


@jit(nopython=True, cache=True)
def get_function_degree(func_id):
    """Get the polynomial degree of a function for smooth coloring."""
    if func_id == FUNC_Z2_PLUS_C or func_id == FUNC_TRICORN or func_id == FUNC_Z2_Z_C or \
       func_id == FUNC_Z2_CZ or func_id == FUNC_Z2_MINUS_Z_C or func_id == FUNC_Z2_C2 or \
       func_id == FUNC_RATIONAL:
        return 2.0
    elif func_id == FUNC_Z3_PLUS_C or func_id == FUNC_Z3_Z_C or func_id == FUNC_CUBIC_JULIA or \
         func_id == FUNC_Z3_MINUS_Z_C:
        return 3.0
    elif func_id == FUNC_Z4_PLUS_C or func_id == FUNC_Z4_Z_C:
        return 4.0
    elif func_id == FUNC_Z5_PLUS_C:
        return 5.0
    elif func_id == FUNC_Z6_PLUS_C:
        return 6.0
    elif func_id == FUNC_Z7_PLUS_C:
        return 7.0
    elif func_id == FUNC_Z8_PLUS_C:
        return 8.0
    else:
        return 2.0  # Default for transcendental functions


@jit(nopython=True, cache=True)
def is_transcendental(func_id):
    """Check if the function is transcendental (sin, cos, exp)."""
    return func_id == FUNC_SIN or func_id == FUNC_COS or func_id == FUNC_EXP


@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def compute_mandelbrot(x_min, x_max, y_min, y_max, width, height, max_iter,
                       func_id=0, escape_radius=2.0):
    """
    Compute the fractal set for a given region with smooth coloring.
    
    Uses the escape time algorithm with smooth iteration count for
    continuous coloring (avoids color banding).
    
    Args:
        x_min, x_max: Real axis bounds in the complex plane
        y_min, y_max: Imaginary axis bounds in the complex plane
        width, height: Output image dimensions in pixels
        max_iter: Maximum iteration count before assuming point is in set
        func_id: Which iteration function to use (default 0 = z² + c)
        escape_radius: Escape threshold (default 2.0)
    
    Returns:
        2D numpy array of float64 containing smooth iteration counts.
        Points in the set have value = max_iter.
    """
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
            
            zr, zi = np.float64(0.0), np.float64(0.0)
            iteration = 0
            
            # Main iteration loop with escape condition based on function type
            if is_trans:
                # For transcendental: escape when |Im(z)| gets large
                while abs(zi) <= escape_threshold and abs(zr) <= escape_threshold and iteration < max_iter:
                    zr, zi = iterate_function(zr, zi, x0, y0, func_id)
                    iteration += 1
            else:
                # For polynomial: standard |z|^2 escape
                while zr * zr + zi * zi <= escape_r2 and iteration < max_iter:
                    zr, zi = iterate_function(zr, zi, x0, y0, func_id)
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
                                func_id=0, escape_radius=2.0):
    """
    Compute fractal for a sub-region, writing into an existing array.
    
    Used for incremental rendering when panning - only compute the
    newly visible strips instead of the entire image.
    
    Args:
        x_min, x_max, y_min, y_max: Full image bounds in complex plane
        width, height: Full image dimensions
        max_iter: Maximum iterations
        result: Output array to write into (modified in place)
        start_x, start_y: Top-left corner of sub-region to compute
        compute_w, compute_h: Size of sub-region to compute
        func_id: Which iteration function to use
        escape_radius: Escape threshold
    """
    # For transcendental functions, use different escape criteria
    is_trans = is_transcendental(func_id)
    if is_trans:
        escape_threshold = 50.0
        escape_r2 = 1e10
    else:
        escape_threshold = 1e10
        escape_r2 = escape_radius * escape_radius
    
    log_escape = np.log(max(escape_radius, 2.0))
    
    degree = get_function_degree(func_id)
    log_degree = np.log(degree)
    
    dx = np.float64(x_max - x_min) / width
    dy = np.float64(y_max - y_min) / height
    
    for py in prange(compute_h):
        actual_py = start_y + py
        y0 = np.float64(y_min) + dy * actual_py
        for px in range(compute_w):
            actual_px = start_x + px
            x0 = np.float64(x_min) + dx * actual_px
            
            zr, zi = np.float64(0.0), np.float64(0.0)
            iteration = 0
            
            if is_trans:
                while abs(zi) <= escape_threshold and abs(zr) <= escape_threshold and iteration < max_iter:
                    zr, zi = iterate_function(zr, zi, x0, y0, func_id)
                    iteration += 1
            else:
                while zr * zr + zi * zi <= escape_r2 and iteration < max_iter:
                    zr, zi = iterate_function(zr, zi, x0, y0, func_id)
                    iteration += 1
            
            if iteration < max_iter:
                zn2 = zr * zr + zi * zi
                if zn2 > 1.0:
                    log_zn = np.log(zn2) * 0.5
                    nu = np.log(log_zn / log_escape) / log_degree
                    result[actual_py, actual_px] = iteration + 1 - nu
                else:
                    result[actual_py, actual_px] = iteration
            else:
                result[actual_py, actual_px] = max_iter


@jit(nopython=True, parallel=True, cache=True)
def apply_colormap_smooth(data, max_iter, colormap, out):
    """
    Apply a colormap to iteration data with smooth interpolation.
    
    Performs linear interpolation between adjacent colormap entries
    for smooth color transitions (no banding).
    
    Args:
        data: 2D array of iteration counts from compute_mandelbrot
        max_iter: Maximum iteration value (points with this value are black)
        colormap: Nx3 array of RGB colors (uint8)
        out: Output RGB image array (modified in place)
    """
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
    """
    Downscale an image by 2x using box filter (4-pixel average).
    
    Used for supersampled anti-aliasing: render at 2x resolution,
    then downscale for smooth edges.
    
    Args:
        src: Source image (2*height, 2*width, 3)
        dst: Destination image (height, width, 3), modified in place
    """
    height, width = dst.shape[:2]
    for y in prange(height):
        y2 = y * 2
        for x in range(width):
            x2 = x * 2
            for c in range(3):
                val = (int(src[y2, x2, c]) + int(src[y2, x2 + 1, c]) +
                       int(src[y2 + 1, x2, c]) + int(src[y2 + 1, x2 + 1, c])) // 4
                dst[y, x, c] = val


def warmup_jit(colormap):
    """
    Warm up JIT compilation with small dummy arrays.
    
    Call this once at startup to pre-compile the Numba functions,
    avoiding a delay on first actual use.
    
    Args:
        colormap: A colormap array to use for warming up apply_colormap_smooth
    """
    _ = compute_mandelbrot(-2, 1, -1, 1, 10, 10, 10)
    dummy = np.zeros((10, 10, 3), dtype=np.uint8)
    dummy_hi = np.zeros((20, 20, 3), dtype=np.uint8)
    apply_colormap_smooth(np.zeros((10, 10), dtype=np.float64), 10, colormap, dummy)
    downscale_2x(dummy_hi, dummy)


# ============================================================================
# Custom Formula Support (non-JIT, slower but flexible)
# ============================================================================

import cmath
import math

# Safe namespace for evaluating custom formulas
_SAFE_MATH_NAMESPACE = {
    '__builtins__': {},
    'sin': cmath.sin,
    'cos': cmath.cos,
    'tan': cmath.tan,
    'exp': cmath.exp,
    'log': cmath.log,
    'sqrt': cmath.sqrt,
    'abs': abs,
    'conj': lambda x: complex(x.real, -x.imag),
    'sinh': cmath.sinh,
    'cosh': cmath.cosh,
    'tanh': cmath.tanh,
    'asin': cmath.asin,
    'acos': cmath.acos,
    'atan': cmath.atan,
    'pi': cmath.pi,
    'e': cmath.e,
    'i': 1j,
    'j': 1j,
}


def prepare_custom_formula(formula_str):
    """
    Prepare a custom formula string for evaluation.
    
    Converts common mathematical notation to Python syntax.
    Returns the prepared formula string.
    """
    import re
    
    f = formula_str.strip()
    
    # Convert ^ to **
    f = f.replace('^', '**')
    
    # List of known function names (don't add * before their parentheses)
    functions = ['sin', 'cos', 'tan', 'exp', 'log', 'sqrt', 'abs', 'conj',
                 'sinh', 'cosh', 'tanh', 'asin', 'acos', 'atan']
    
    # Handle implicit multiplication
    # Number followed by variable (z or c)
    f = re.sub(r'(\d)([zc])', r'\1*\2', f)
    
    # Variable followed by variable (zc -> z*c, cz -> c*z)
    f = re.sub(r'([zc])([zc])', r'\1*\2', f)
    
    # ) followed by variable or (
    f = re.sub(r'\)([zc(])', r')*\1', f)
    
    # Variable followed by ( when it's NOT part of a function
    # First, protect function names by replacing them temporarily
    for func in functions:
        f = f.replace(func + '(', f'__FUNC_{func}__(')
    
    # Now add * between variable and (
    f = re.sub(r'([zc])(\()', r'\1*\2', f)
    
    # Restore function names
    for func in functions:
        f = f.replace(f'__FUNC_{func}__(', func + '(')
    
    return f


def eval_custom_formula(formula_str, z, c):
    """
    Evaluate a custom formula with given z and c values.
    
    Args:
        formula_str: The prepared formula string
        z: Complex number z
        c: Complex number c (parameter)
    
    Returns:
        The result of evaluating the formula
    """
    namespace = _SAFE_MATH_NAMESPACE.copy()
    namespace['z'] = z
    namespace['c'] = c
    try:
        return eval(formula_str, namespace)
    except Exception:
        return complex(1e10, 0)  # Return large value to escape


def compute_mandelbrot_custom(x_min, x_max, y_min, y_max, width, height, max_iter,
                               custom_formula, escape_radius=2.0):
    """
    Compute fractal set using a custom formula (non-JIT, slower but flexible).
    
    This function does NOT use Numba JIT compilation, so it's slower than
    the built-in functions, but allows any user-defined formula.
    
    Args:
        x_min, x_max: Real axis bounds
        y_min, y_max: Imaginary axis bounds  
        width, height: Output dimensions
        max_iter: Maximum iterations
        custom_formula: Prepared formula string (use prepare_custom_formula first)
        escape_radius: Escape threshold
    
    Returns:
        2D numpy array of float64 iteration counts
    """
    result = np.zeros((height, width), dtype=np.float64)
    escape_r2 = escape_radius * escape_radius
    log_escape = math.log(max(escape_radius, 2.0))
    
    # Degree estimation for smooth coloring (default to 2)
    degree = 2.0
    log_degree = math.log(degree)
    
    dx = (x_max - x_min) / width
    dy = (y_max - y_min) / height
    
    for py in range(height):
        y0 = y_min + dy * py
        for px in range(width):
            x0 = x_min + dx * px
            
            z = complex(0, 0)
            c = complex(x0, y0)
            iteration = 0
            
            try:
                while abs(z) ** 2 <= escape_r2 and iteration < max_iter:
                    z = eval_custom_formula(custom_formula, z, c)
                    iteration += 1
                    
                    # Safety check for NaN/Inf
                    if math.isnan(z.real) or math.isnan(z.imag) or \
                       math.isinf(z.real) or math.isinf(z.imag):
                        break
            except Exception:
                pass
            
            if iteration < max_iter:
                zn2 = abs(z) ** 2
                if zn2 > 1.0:
                    log_zn = math.log(zn2) * 0.5
                    nu = math.log(log_zn / log_escape) / log_degree
                    result[py, px] = iteration + 1 - nu
                else:
                    result[py, px] = iteration
            else:
                result[py, px] = max_iter
    
    return result
