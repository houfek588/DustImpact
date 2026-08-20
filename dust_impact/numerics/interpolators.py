# -*- coding: utf-8 -*-
"""
High-performance vectorized Grid-to-Particle interpolators (2D Bilinear and 3D Trilinear).
Supports Numba JIT acceleration with pure NumPy fallback.
"""

import numpy as np

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


@njit(fastmath=True)
def interp_field_2d(x: np.ndarray, y: np.ndarray, x_grid: np.ndarray, y_grid: np.ndarray,
                    dx: float, dy: float, Field_2D: np.ndarray) -> np.ndarray:
    """
    Fast vectorized 2D bilinear grid-to-particle interpolation.
    """
    Nx, Ny = Field_2D.shape
    idx_x = np.clip((x - x_grid[0]) / dx, 0.0, float(Nx - 1.0001))
    idx_y = np.clip((y - y_grid[0]) / dy, 0.0, float(Ny - 1.0001))

    i = np.clip(np.floor(idx_x).astype(np.int64), 0, Nx - 2)
    j = np.clip(np.floor(idx_y).astype(np.int64), 0, Ny - 2)

    tx = idx_x - i
    ty = idx_y - j

    c00 = Field_2D[i, j]
    c10 = Field_2D[i + 1, j]
    c01 = Field_2D[i, j + 1]
    c11 = Field_2D[i + 1, j + 1]

    return (1 - tx) * (1 - ty) * c00 + tx * (1 - ty) * c10 + (1 - tx) * ty * c01 + tx * ty * c11


@njit(fastmath=True)
def interp_field_3d(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                    x_grid: np.ndarray, y_grid: np.ndarray, z_grid: np.ndarray,
                    dx: float, dy: float, dz: float, Field_3D: np.ndarray) -> np.ndarray:
    """
    Fast vectorized 3D trilinear grid-to-particle interpolation.
    """
    Nx, Ny, Nz = Field_3D.shape
    idx_x = np.clip((x - x_grid[0]) / dx, 0.0, float(Nx - 1.0001))
    idx_y = np.clip((y - y_grid[0]) / dy, 0.0, float(Ny - 1.0001))
    idx_z = np.clip((z - z_grid[0]) / dz, 0.0, float(Nz - 1.0001))

    i = np.clip(np.floor(idx_x).astype(np.int64), 0, Nx - 2)
    j = np.clip(np.floor(idx_y).astype(np.int64), 0, Ny - 2)
    k = np.clip(np.floor(idx_z).astype(np.int64), 0, Nz - 2)

    tx = idx_x - i
    ty = idx_y - j
    tz = idx_z - k

    c000 = Field_3D[i, j, k]
    c100 = Field_3D[i + 1, j, k]
    c010 = Field_3D[i, j + 1, k]
    c110 = Field_3D[i + 1, j + 1, k]
    c001 = Field_3D[i, j, k + 1]
    c101 = Field_3D[i + 1, j, k + 1]
    c011 = Field_3D[i, j + 1, k + 1]
    c111 = Field_3D[i + 1, j + 1, k + 1]

    c00 = c000 * (1 - tx) + c100 * tx
    c01 = c001 * (1 - tx) + c101 * tx
    c10 = c010 * (1 - tx) + c110 * tx
    c11 = c011 * (1 - tx) + c111 * tx

    c0 = c00 * (1 - ty) + c10 * ty
    c1 = c01 * (1 - ty) + c11 * ty

    return c0 * (1 - tz) + c1 * tz
