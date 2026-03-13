"""Sphere-domain fractal compute kernels for Riemann sphere visualization."""

import math
import numpy as np
from numba import jit, prange

from compute import iterate_function
from custom_formula import eval_prepared_formula

SPHERE_PARAM_MODE_INT = 0
SPHERE_JULIA_MODE_INT = 1


@jit(nopython=True, cache=True, fastmath=True)
def _rotate_point(x, y, z, yaw, pitch):
    """Rotate point on sphere by yaw (Y axis) and pitch (X axis)."""
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    x1 = cos_yaw * x + sin_yaw * z
    z1 = -sin_yaw * x + cos_yaw * z

    cos_pitch = np.cos(pitch)
    sin_pitch = np.sin(pitch)
    y2 = cos_pitch * y - sin_pitch * z1
    z2 = sin_pitch * y + cos_pitch * z1

    return x1, y2, z2


@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def compute_sphere_fractal_predefined(width, height, max_iter, func_id, escape_radius,
                                      mode_int, yaw, pitch, julia_c_real, julia_c_imag):
    """Compute escape-time data over the visible hemisphere using predefined functions."""
    result = np.full((height, width), max_iter, dtype=np.float64)
    escape_r2 = escape_radius * escape_radius
    log_escape = np.log(max(escape_radius, 2.0))
    log_degree = np.log(2.0)

    julia_cr = np.float64(julia_c_real)
    julia_ci = np.float64(julia_c_imag)

    scale_x = 2.0 / width
    scale_y = 2.0 / height

    for py in prange(height):
        sy = 1.0 - (py + 0.5) * scale_y
        for px in range(width):
            sx = (px + 0.5) * scale_x - 1.0
            r2 = sx * sx + sy * sy
            if r2 > 1.0:
                continue

            sz = np.sqrt(max(0.0, 1.0 - r2))
            x, y, z = _rotate_point(sx, sy, sz, yaw, pitch)

            denom = 1.0 - z
            if denom <= 1e-12:
                continue

            plane_r = x / denom
            plane_i = y / denom

            if mode_int == SPHERE_JULIA_MODE_INT:
                zr = plane_r
                zi = plane_i
                cr = julia_cr
                ci = julia_ci
            else:
                zr = 0.0
                zi = 0.0
                cr = plane_r
                ci = plane_i

            iteration = 0
            while zr * zr + zi * zi <= escape_r2 and iteration < max_iter:
                zr, zi = iterate_function(zr, zi, cr, ci, func_id)
                iteration += 1

            if iteration < max_iter:
                zn2 = zr * zr + zi * zi
                if zn2 > 1.0:
                    result[py, px] = iteration + 1 - np.log(np.log(zn2) * 0.5 / log_escape) / log_degree
                else:
                    result[py, px] = iteration
            else:
                result[py, px] = max_iter

    return result


def compute_sphere_fractal_custom(width, height, max_iter, prepared_formula, escape_radius,
                                  mode_int, yaw, pitch, julia_c_real, julia_c_imag):
    """Compute sphere fractal with custom formula (Python fallback)."""
    result = np.full((height, width), max_iter, dtype=np.float64)
    escape_r2 = escape_radius * escape_radius
    log_escape = math.log(max(escape_radius, 2.0))
    log_degree = math.log(2.0)

    julia_c = complex(julia_c_real, julia_c_imag)
    local_scope = {'z': 0j, 'c': 0j}

    scale_x = 2.0 / width
    scale_y = 2.0 / height
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    cos_pitch = math.cos(pitch)
    sin_pitch = math.sin(pitch)

    for py in range(height):
        sy = 1.0 - (py + 0.5) * scale_y
        for px in range(width):
            sx = (px + 0.5) * scale_x - 1.0
            r2 = sx * sx + sy * sy
            if r2 > 1.0:
                continue

            sz = math.sqrt(max(0.0, 1.0 - r2))

            x1 = cos_yaw * sx + sin_yaw * sz
            z1 = -sin_yaw * sx + cos_yaw * sz
            y2 = cos_pitch * sy - sin_pitch * z1
            z2 = sin_pitch * sy + cos_pitch * z1

            denom = 1.0 - z2
            if denom <= 1e-12:
                continue

            plane_point = complex(x1 / denom, y2 / denom)

            if mode_int == SPHERE_JULIA_MODE_INT:
                z = plane_point
                c = julia_c
            else:
                z = complex(0.0, 0.0)
                c = plane_point

            iteration = 0
            while (z.real * z.real + z.imag * z.imag) <= escape_r2 and iteration < max_iter:
                z = eval_prepared_formula(prepared_formula, z, c, local_scope)
                iteration += 1
                if math.isnan(z.real) or math.isnan(z.imag) or math.isinf(z.real) or math.isinf(z.imag):
                    break

            if iteration < max_iter:
                zn2 = z.real * z.real + z.imag * z.imag
                if zn2 > 1.0:
                    result[py, px] = iteration + 1 - math.log(math.log(zn2) * 0.5 / log_escape) / log_degree
                else:
                    result[py, px] = iteration
            else:
                result[py, px] = max_iter

    return result
