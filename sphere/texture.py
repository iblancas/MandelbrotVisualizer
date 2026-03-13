"""Texture-based sphere rendering for smooth interactive rotation."""

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
def build_sphere_texture_predefined(tex_w, tex_h, max_iter, func_id, escape_radius,
                                    mode_int, julia_c_real, julia_c_imag):
    """Build cached sphere texture with escape-time values for predefined formulas."""
    texture = np.full((tex_h, tex_w), max_iter, dtype=np.float64)
    escape_r2 = escape_radius * escape_radius
    log_escape = np.log(max(escape_radius, 2.0))
    log_degree = np.log(2.0)

    two_pi = 2.0 * np.pi

    for ty in prange(tex_h):
        lat = (0.5 - (ty + 0.5) / tex_h) * np.pi
        cos_lat = np.cos(lat)
        sin_lat = np.sin(lat)

        for tx in range(tex_w):
            lon = ((tx + 0.5) / tex_w - 0.5) * two_pi
            x = cos_lat * np.cos(lon)
            y = sin_lat
            z = cos_lat * np.sin(lon)

            denom = 1.0 - z
            if denom <= 1e-12:
                texture[ty, tx] = 1.0
                continue

            plane_r = x / denom
            plane_i = y / denom

            if mode_int == SPHERE_JULIA_MODE_INT:
                zr = plane_r
                zi = plane_i
                cr = julia_c_real
                ci = julia_c_imag
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
                    texture[ty, tx] = iteration + 1 - np.log(np.log(zn2) * 0.5 / log_escape) / log_degree
                else:
                    texture[ty, tx] = iteration
            else:
                texture[ty, tx] = max_iter

    return texture


def build_sphere_texture_custom(tex_w, tex_h, max_iter, prepared_formula, escape_radius,
                                mode_int, julia_c_real, julia_c_imag):
    """Build cached sphere texture with escape-time values for custom formulas."""
    texture = np.full((tex_h, tex_w), max_iter, dtype=np.float64)
    escape_r2 = escape_radius * escape_radius
    log_escape = math.log(max(escape_radius, 2.0))
    log_degree = math.log(2.0)

    julia_c = complex(julia_c_real, julia_c_imag)
    local_scope = {'z': 0j, 'c': 0j}

    two_pi = 2.0 * math.pi

    for ty in range(tex_h):
        lat = (0.5 - (ty + 0.5) / tex_h) * math.pi
        cos_lat = math.cos(lat)
        sin_lat = math.sin(lat)

        for tx in range(tex_w):
            lon = ((tx + 0.5) / tex_w - 0.5) * two_pi
            x = cos_lat * math.cos(lon)
            y = sin_lat
            z = cos_lat * math.sin(lon)

            denom = 1.0 - z
            if denom <= 1e-12:
                texture[ty, tx] = 1.0
                continue

            plane_point = complex(x / denom, y / denom)

            if mode_int == SPHERE_JULIA_MODE_INT:
                z_val = plane_point
                c_val = julia_c
            else:
                z_val = complex(0.0, 0.0)
                c_val = plane_point

            iteration = 0
            while (z_val.real * z_val.real + z_val.imag * z_val.imag) <= escape_r2 and iteration < max_iter:
                z_val = eval_prepared_formula(prepared_formula, z_val, c_val, local_scope)
                iteration += 1
                if (
                    math.isnan(z_val.real)
                    or math.isnan(z_val.imag)
                    or math.isinf(z_val.real)
                    or math.isinf(z_val.imag)
                ):
                    break

            if iteration < max_iter:
                zn2 = z_val.real * z_val.real + z_val.imag * z_val.imag
                if zn2 > 1.0:
                    texture[ty, tx] = iteration + 1 - math.log(math.log(zn2) * 0.5 / log_escape) / log_degree
                else:
                    texture[ty, tx] = iteration
            else:
                texture[ty, tx] = max_iter

    return texture


@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def sample_sphere_texture(texture, out_data, max_iter, yaw, pitch, zoom):
    """Render sphere by sampling cached texture under current orientation."""
    out_h, out_w = out_data.shape
    tex_h, tex_w = texture.shape

    two_pi = 2.0 * np.pi
    inv_two_pi = 1.0 / two_pi

    for py in prange(out_h):
        sy = (1.0 - 2.0 * (py + 0.5) / out_h) / zoom
        for px in range(out_w):
            sx = (2.0 * (px + 0.5) / out_w - 1.0) / zoom
            r2 = sx * sx + sy * sy
            if r2 > 1.0:
                out_data[py, px] = max_iter
                continue

            sz = np.sqrt(max(0.0, 1.0 - r2))
            x, y, z = _rotate_point(sx, sy, sz, yaw, pitch)

            lon = np.arctan2(z, x)
            lat = np.arcsin(max(-1.0, min(1.0, y)))

            u = (lon * inv_two_pi + 0.5) * tex_w
            v = (0.5 - lat / np.pi) * tex_h

            u0 = int(np.floor(u)) % tex_w
            v0 = int(np.floor(v))
            if v0 < 0:
                v0 = 0
            elif v0 >= tex_h:
                v0 = tex_h - 1

            u1 = (u0 + 1) % tex_w
            v1 = v0 + 1
            if v1 >= tex_h:
                v1 = tex_h - 1

            fu = u - np.floor(u)
            fv = v - np.floor(v)

            t00 = texture[v0, u0]
            t01 = texture[v0, u1]
            t10 = texture[v1, u0]
            t11 = texture[v1, u1]

            top = t00 * (1.0 - fu) + t01 * fu
            bottom = t10 * (1.0 - fu) + t11 * fu
            out_data[py, px] = top * (1.0 - fv) + bottom * fv
