# -*- coding: utf-8 -*-
"""
High-performance vectorized Grid-to-Particle interpolators and Particle-to-Grid
Cloud-in-Cell (CIC) charge deposition schemes.
Supports Numba JIT acceleration with pure NumPy vectorized fallback.
"""

from typing import List, Tuple, Union
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


def trilinear_gradient_3d(
    x: np.ndarray, y: np.ndarray, z: np.ndarray,
    x_grid: np.ndarray, y_grid: np.ndarray, z_grid: np.ndarray,
    dx: float, dy: float, dz: float,
    fields: List[np.ndarray], weights: List[float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Exact gradient of the trilinear interpolant of V = sum_m weights[m] * fields[m] at the particles.

    Inside a cell the trilinear interpolant is a smooth function, so a force F = -q grad V computed this
    way does work q [V(a) - V(b)] along any path between two points of the cell: it is exactly
    conservative, unlike the trilinear interpolation of nodal fields E = -grad V. The cell selection and
    the clipping at the domain edges are those of interp_field_3d.

    Returns
    -------
    (dV/dx, dV/dy, dV/dz) at the particle positions [V/m].
    """
    Nx, Ny, Nz = fields[0].shape
    idx_x = np.clip((x - x_grid[0]) / dx, 0.0, float(Nx - 1.0001))
    idx_y = np.clip((y - y_grid[0]) / dy, 0.0, float(Ny - 1.0001))
    idx_z = np.clip((z - z_grid[0]) / dz, 0.0, float(Nz - 1.0001))
    i = np.clip(np.floor(idx_x).astype(np.int64), 0, Nx - 2)
    j = np.clip(np.floor(idx_y).astype(np.int64), 0, Ny - 2)
    k = np.clip(np.floor(idx_z).astype(np.int64), 0, Nz - 2)
    tx, ty, tz = idx_x - i, idx_y - j, idx_z - k

    def corner(di: int, dj: int, dk: int) -> np.ndarray:
        return sum(w * F[i + di, j + dj, k + dk] for F, w in zip(fields, weights))

    c000, c100, c010, c110 = corner(0, 0, 0), corner(1, 0, 0), corner(0, 1, 0), corner(1, 1, 0)
    c001, c101, c011, c111 = corner(0, 0, 1), corner(1, 0, 1), corner(0, 1, 1), corner(1, 1, 1)

    gx = ((1 - ty) * (1 - tz) * (c100 - c000) + ty * (1 - tz) * (c110 - c010)
          + (1 - ty) * tz * (c101 - c001) + ty * tz * (c111 - c011)) / dx
    gy = ((1 - tx) * (1 - tz) * (c010 - c000) + tx * (1 - tz) * (c110 - c100)
          + (1 - tx) * tz * (c011 - c001) + tx * tz * (c111 - c101)) / dy
    gz = ((1 - tx) * (1 - ty) * (c001 - c000) + tx * (1 - ty) * (c101 - c100)
          + (1 - tx) * ty * (c011 - c010) + tx * ty * (c111 - c110)) / dz
    return gx, gy, gz


def cells_touching_mask(node_mask: np.ndarray) -> np.ndarray:
    """Boolean array (Nx-1, Ny-1, Nz-1) of cells with at least one corner node in node_mask."""
    m = np.asarray(node_mask, dtype=bool)
    cells = np.zeros(tuple(s - 1 for s in m.shape), dtype=bool)
    for di in (0, 1):
        for dj in (0, 1):
            for dk in (0, 1):
                cells |= m[di:m.shape[0] - 1 + di, dj:m.shape[1] - 1 + dj, dk:m.shape[2] - 1 + dk]
    return cells


def particle_cell_indices(
    x: np.ndarray, y: np.ndarray, z: np.ndarray,
    x_grid: np.ndarray, y_grid: np.ndarray, z_grid: np.ndarray,
    dx: float, dy: float, dz: float, shape: Tuple[int, int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cell indices (i, j, k) of the particles, as selected by interp_field_3d."""
    Nx, Ny, Nz = shape
    i = np.clip(np.floor(np.clip((x - x_grid[0]) / dx, 0.0, Nx - 1.0001)).astype(np.int64), 0, Nx - 2)
    j = np.clip(np.floor(np.clip((y - y_grid[0]) / dy, 0.0, Ny - 1.0001)).astype(np.int64), 0, Ny - 2)
    k = np.clip(np.floor(np.clip((z - z_grid[0]) / dz, 0.0, Nz - 1.0001)).astype(np.int64), 0, Nz - 2)
    return i, j, k


if HAS_NUMBA:
    @njit(fastmath=True)
    def _deposit_cic_numba(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                           q_val: float, is_array: bool, q_arr: np.ndarray,
                           x0: float, y0: float, z0: float,
                           dx: float, dy: float, dz: float,
                           Nx: int, Ny: int, Nz: int,
                           rho: np.ndarray) -> None:
        n = len(x)
        for p in range(n):
            u = (x[p] - x0) / dx
            v = (y[p] - y0) / dy
            w = (z[p] - z0) / dz
            if u < 0.0 or u > Nx - 1.0 or v < 0.0 or v > Ny - 1.0 or w < 0.0 or w > Nz - 1.0:
                continue
            i = int(u)
            if i >= Nx - 1:
                i = Nx - 2
            j = int(v)
            if j >= Ny - 1:
                j = Ny - 2
            k = int(w)
            if k >= Nz - 1:
                k = Nz - 2

            du = u - i
            dv = v - j
            dw = w - k
            qp = q_arr[p] if is_array else q_val

            rho[i, j, k] += (1.0 - du) * (1.0 - dv) * (1.0 - dw) * qp
            rho[i + 1, j, k] += du * (1.0 - dv) * (1.0 - dw) * qp
            rho[i, j + 1, k] += (1.0 - du) * dv * (1.0 - dw) * qp
            rho[i + 1, j + 1, k] += du * dv * (1.0 - dw) * qp
            rho[i, j, k + 1] += (1.0 - du) * (1.0 - dv) * dw * qp
            rho[i + 1, j, k + 1] += du * (1.0 - dv) * dw * qp
            rho[i, j + 1, k + 1] += (1.0 - du) * dv * dw * qp
            rho[i + 1, j + 1, k + 1] += du * dv * dw * qp


def deposit_charge_cic_3d(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    q_macro: Union[float, np.ndarray],
    x0: float,
    y0: float,
    z0: float,
    dx: float,
    dy: float,
    dz: float,
    Nx: int,
    Ny: int,
    Nz: int,
) -> np.ndarray:
    """
    Cloud-in-Cell (CIC) trilinear charge deposition onto 3D Cartesian grid nodes.
    Deposits particle charges onto an (Nx, Ny, Nz) node grid using adjoint trilinear weights.

    Parameters:
        x, y, z: 1D particle coordinates.
        q_macro: Macro-particle charge (float or 1D array of length equal to x).
        x0, y0, z0: Origin coordinates of the grid (i.e. x_grid[0], y_grid[0], z_grid[0]).
        dx, dy, dz: Grid cell spacings along each axis.
        Nx, Ny, Nz: Number of grid nodes along each axis.

    Returns:
        np.ndarray: Deposited charge in Coulombs on each grid node, shape (Nx, Ny, Nz).
                    Divide by control volume dV = dx * dy * dz to obtain volume charge density rho (C/m^3).
    """
    rho = np.zeros((Nx, Ny, Nz), dtype=np.float64)
    if len(x) == 0:
        return rho

    is_array = isinstance(q_macro, np.ndarray)
    if HAS_NUMBA:
        q_arr = q_macro if is_array else np.empty(0, dtype=np.float64)
        q_val = float(q_macro) if not is_array else 0.0
        _deposit_cic_numba(
            np.asarray(x, dtype=np.float64),
            np.asarray(y, dtype=np.float64),
            np.asarray(z, dtype=np.float64),
            q_val, is_array, q_arr,
            float(x0), float(y0), float(z0),
            float(dx), float(dy), float(dz),
            int(Nx), int(Ny), int(Nz),
            rho
        )
        return rho

    u = (x - x0) / dx
    v = (y - y0) / dy
    w = (z - z0) / dz

    valid = (u >= 0.0) & (u <= Nx - 1.0) & (v >= 0.0) & (v <= Ny - 1.0) & (w >= 0.0) & (w <= Nz - 1.0)
    if not np.any(valid):
        return rho

    u = u[valid]
    v = v[valid]
    w = w[valid]

    i = np.clip(np.floor(u).astype(np.int64), 0, Nx - 2)
    j = np.clip(np.floor(v).astype(np.int64), 0, Ny - 2)
    k = np.clip(np.floor(w).astype(np.int64), 0, Nz - 2)

    du = np.clip(u - i, 0.0, 1.0)
    dv = np.clip(v - j, 0.0, 1.0)
    dw = np.clip(w - k, 0.0, 1.0)

    q = q_macro[valid] if is_array else q_macro

    w000 = (1.0 - du) * (1.0 - dv) * (1.0 - dw) * q
    w100 = du * (1.0 - dv) * (1.0 - dw) * q
    w010 = (1.0 - du) * dv * (1.0 - dw) * q
    w110 = du * dv * (1.0 - du * 0.0) * (1.0 - dw) * q if False else du * dv * (1.0 - dw) * q
    w001 = (1.0 - du) * (1.0 - dv) * dw * q
    w101 = du * (1.0 - dv) * dw * q
    w011 = (1.0 - du) * dv * dw * q
    w111 = du * dv * dw * q

    np.add.at(rho, (i, j, k), w000)
    np.add.at(rho, (i + 1, j, k), w100)
    np.add.at(rho, (i, j + 1, k), w010)
    np.add.at(rho, (i + 1, j + 1, k), w110)
    np.add.at(rho, (i, j, k + 1), w001)
    np.add.at(rho, (i + 1, j, k + 1), w101)
    np.add.at(rho, (i, j + 1, k + 1), w011)
    np.add.at(rho, (i + 1, j + 1, k + 1), w111)

    return rho
