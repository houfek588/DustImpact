# -*- coding: utf-8 -*-
"""
Surface analysis, ray-tracing, impact intersection, and surface normal calculations.
Supports both PyVista surface mesh ray tracing and discrete 3D voxel grid ray marching.
"""

from typing import Tuple, Optional, Sequence, Any
import numpy as np


def interpolate_field_3d(
    x: float, y: float, z: float,
    field_x: np.ndarray, field_y: np.ndarray, field_z: np.ndarray,
    x_grid: np.ndarray, y_grid: np.ndarray, z_grid: np.ndarray,
    dx: float, dy: float, dz: float
) -> Tuple[float, float, float]:
    """
    Trilinear interpolation of a 3D vector field at arbitrary spatial coordinates (x, y, z).

    Parameters
    ----------
    x, y, z : float
        Query position coordinates.
    field_x, field_y, field_z : np.ndarray
        3D arrays of vector components.
    x_grid, y_grid, z_grid : np.ndarray
        1D coordinate axes arrays.
    dx, dy, dz : float
        Grid cell spacing.

    Returns
    -------
    Tuple[float, float, float]
        Interpolated vector components (vx, vy, vz).
    """
    Nx, Ny, Nz = field_x.shape

    idx_x = (x - x_grid[0]) / dx
    idx_y = (y - y_grid[0]) / dy
    idx_z = (z - z_grid[0]) / dz

    i0 = int(np.floor(idx_x))
    i1 = i0 + 1
    j0 = int(np.floor(idx_y))
    j1 = j0 + 1
    k0 = int(np.floor(idx_z))
    k1 = k0 + 1

    i0 = max(0, min(Nx - 1, i0))
    i1 = max(0, min(Nx - 1, i1))
    j0 = max(0, min(Ny - 1, j0))
    j1 = max(0, min(Ny - 1, j1))
    k0 = max(0, min(Nz - 1, k0))
    k1 = max(0, min(Nz - 1, k1))

    tx = max(0.0, min(1.0, idx_x - i0))
    ty = max(0.0, min(1.0, idx_y - j0))
    tz = max(0.0, min(1.0, idx_z - k0))

    def interp_comp(f: np.ndarray) -> float:
        c000 = f[i0, j0, k0]
        c100 = f[i1, j0, k0]
        c010 = f[i0, j1, k0]
        c110 = f[i1, j1, k0]
        c001 = f[i0, j0, k1]
        c101 = f[i1, j0, k1]
        c011 = f[i0, j1, k1]
        c111 = f[i1, j1, k1]

        c00 = c000 * (1.0 - tx) + c100 * tx
        c10 = c010 * (1.0 - tx) + c110 * tx
        c01 = c001 * (1.0 - tx) + c101 * tx
        c11 = c011 * (1.0 - tx) + c111 * tx

        c0 = c00 * (1.0 - ty) + c10 * ty
        c1 = c01 * (1.0 - ty) + c11 * ty

        return float(c0 * (1.0 - tz) + c1 * tz)

    return interp_comp(field_x), interp_comp(field_y), interp_comp(field_z)


def ray_trace_mesh(
    mesh: Any,
    p_start: np.ndarray,
    dir_u: np.ndarray,
    max_dist: float = 100.0
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Traces a ray against a PyVista/VTK polygonal mesh surface.

    Parameters
    ----------
    mesh : pyvista.PolyData or pyvista.DataSet
        Polygonal surface mesh of the spacecraft.
    p_start : np.ndarray
        Starting point of the ray [x, y, z].
    dir_u : np.ndarray
        Normalized ray direction vector.
    max_dist : float
        Maximum ray propagation distance.

    Returns
    -------
    Tuple[Optional[np.ndarray], Optional[np.ndarray]]
        (intersection_point, outward_surface_normal) or (None, None) if no intersection found.
    """
    if mesh is None or not hasattr(mesh, 'ray_trace'):
        return None, None

    try:
        p_target = p_start + max_dist * dir_u
        points, ind_faces = mesh.ray_trace(p_start, p_target)
        if len(points) > 0:
            intersection_pt = np.array(points[0], dtype=float)
            normal_vec = None
            try:
                norm = mesh.face_normals[ind_faces[0]]
                n_len = np.linalg.norm(norm)
                if n_len > 1e-9:
                    normal_vec = np.array(norm / n_len, dtype=float)
            except Exception:
                pass
            return intersection_pt, normal_vec
    except Exception:
        pass

    return None, None


def ray_march_voxel_grid(
    p_start: np.ndarray,
    dir_u: np.ndarray,
    mask_3d: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    z_grid: np.ndarray,
    dx: float,
    dy: float,
    dz: float,
    max_dist: float = 20.0
) -> Optional[np.ndarray]:
    """
    Marching ray through a 3D boolean voxel grid to find the first cell intersection.

    Parameters
    ----------
    p_start : np.ndarray
        Starting point [x, y, z].
    dir_u : np.ndarray
        Unit direction vector.
    mask_3d : np.ndarray
        3D boolean array where True is solid boundary.
    x_grid, y_grid, z_grid : np.ndarray
        Grid node coordinates.
    dx, dy, dz : float
        Cell spacing.
    max_dist : float
        Maximum ray search distance.

    Returns
    -------
    Optional[np.ndarray]
        Intersection coordinate [x, y, z] or None.
    """
    step_size = 0.25 * min(dx, dy, dz)
    steps = int(max_dist / step_size)
    Nx, Ny, Nz = mask_3d.shape

    for step in range(steps):
        curr_pos = p_start + step * step_size * dir_u
        i = int(round((curr_pos[0] - x_grid[0]) / dx))
        j = int(round((curr_pos[1] - y_grid[0]) / dy))
        k = int(round((curr_pos[2] - z_grid[0]) / dz))

        if 0 <= i < Nx and 0 <= j < Ny and 0 <= k < Nz:
            if mask_3d[i, j, k]:
                return curr_pos

    return None


def compute_surface_normal_from_potential(
    pos: np.ndarray,
    V_field: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    z_grid: np.ndarray,
    dx: float,
    dy: float,
    dz: float
) -> np.ndarray:
    """
    Computes outward surface normal at given position from negative potential gradient (-grad(V)).

    Parameters
    ----------
    pos : np.ndarray
        [x, y, z] position on or near surface.
    V_field : np.ndarray
        3D potential or weighting field.
    x_grid, y_grid, z_grid : np.ndarray
        Grid coordinates.
    dx, dy, dz : float
        Grid steps.

    Returns
    -------
    np.ndarray
        Normalized unit normal vector [nx, ny, nz].
    """
    Ex, Ey, Ez = np.gradient(-V_field, dx, dy, dz)
    gx, gy, gz = interpolate_field_3d(pos[0], pos[1], pos[2], Ex, Ey, Ez, x_grid, y_grid, z_grid, dx, dy, dz)
    normal = np.array([gx, gy, gz], dtype=float)
    norm_n = np.linalg.norm(normal)
    if norm_n > 1e-6:
        return normal / norm_n

    # Radial fallback
    norm_p = np.linalg.norm(pos)
    if norm_p > 1e-6:
        return pos / norm_p

    return np.array([-0.7071, 0.7071, 0.0])


def compute_impact_intersection_and_normal(
    params: Any,
    Vw_body: np.ndarray,
    spacecraft_mask_3d: np.ndarray,
    mesh_body: Any = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates exact dust impact point on spacecraft surface and outward surface normal.
    Updates params.impact_pos and params.impact_normal and returns them.

    Parameters
    ----------
    params : SimulationParams3D or object
        Simulation parameters containing impact_location_xyz_m, impact_direction_vector,
        grid specifications, etc.
    Vw_body : np.ndarray
        3D spacecraft weighting field.
    spacecraft_mask_3d : np.ndarray
        3D spacecraft boolean mask.
    mesh_body : Any, optional
        PyVista PolyData mesh of spacecraft body.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (impact_pos, impact_normal)
    """
    P_start = np.array(
        getattr(params, 'impact_location_xyz_m', getattr(params, 'impact_pos', [-2.0, 2.0, 0.0])),
        dtype=float
    )
    v_dir = np.array(getattr(params, 'impact_direction_vector', [0.0, 0.0, 0.0]), dtype=float)
    v_norm = np.linalg.norm(v_dir)

    x_grid = params.x_grid
    y_grid = params.y_grid
    z_grid = params.z_grid
    dx, dy, dz = params.dx, params.dy, params.dz

    # 1. Zero direction vector [0,0,0]: do not calculate intersection, set impact_pos directly to P_start
    if v_norm < 1e-9:
        print(f"  -> Vektor pohybu prachu je nulový {list(v_dir)}. Bod dopadu je stanoven přímo ve výchozím místě {list(P_start)}.")
        impact_pos = P_start
        normal = compute_surface_normal_from_potential(P_start, Vw_body, x_grid, y_grid, z_grid, dx, dy, dz)
        params.impact_pos = list(impact_pos)
        params.impact_normal = list(normal)
        return impact_pos, normal

    # 2. Non-zero direction vector: calculate ray intersection with spacecraft surface
    dir_u = v_dir / v_norm
    intersection_point, normal_vector = ray_trace_mesh(mesh_body, P_start, dir_u, max_dist=100.0)

    # Grid step ray-casting fallback if PyVista ray_trace didn't return point
    if intersection_point is None:
        max_dist = 2.0 * max(params.L_x, params.L_y, params.L_z)
        intersection_point = ray_march_voxel_grid(
            P_start, dir_u, spacecraft_mask_3d, x_grid, y_grid, z_grid, dx, dy, dz, max_dist=max_dist
        )

    # If NO intersection exists along the trajectory: raise error message!
    if intersection_point is None:
        err_msg = (
            f"\n[CHYBA] Nelze spočítat bod dopadu! Prachová částice z výchozího místa {list(P_start)} "
            f"s vektorem pohybu {list(v_dir)} neprotíná povrch sondy!"
        )
        print(err_msg)
        raise ValueError(err_msg)

    if normal_vector is None:
        normal_vector = compute_surface_normal_from_potential(
            intersection_point, Vw_body, x_grid, y_grid, z_grid, dx, dy, dz
        )

    params.impact_pos = list(intersection_point)
    params.impact_normal = list(normal_vector)

    print(f"  -> Skutečný vypočtený bod dopadu na povrchu sondy: {params.impact_pos}")
    print(f"  -> Vypočtená normála v místě dopadu: {params.impact_normal}")

    return intersection_point, normal_vector
