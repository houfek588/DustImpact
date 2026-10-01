# -*- coding: utf-8 -*-
"""
Voxelizer module for detecting metallic surfaces and converting analytical/mesh geometries
into 3D discrete boolean masks on Cartesian grids.
"""

from typing import Tuple, Sequence, Optional
import numpy as np
import scipy.ndimage as ndimage


def detect_metal_mask_3d(V: np.ndarray, dx_min: float, threshold: float = 0.30) -> np.ndarray:
    """
    Detects metal boundary masks for multi-level potential grids using potential gradient
    and binary hole filling.

    Parameters
    ----------
    V : np.ndarray
        3D array of electric potential or weighting field.
    dx_min : float
        Minimum spatial grid spacing (dx, dy, or dz).
    threshold : float, optional
        Fraction of maximum potential gradient or maximum potential used as cutoff (default: 0.30).

    Returns
    -------
    np.ndarray (bool)
        3D boolean mask where True indicates conductor / metal volume.
    """
    Ex, Ey, Ez = np.gradient(-V, dx_min, dx_min, dx_min)
    grad_mag = np.sqrt(Ex ** 2 + Ey ** 2 + Ez ** 2)

    max_grad = np.nanmax(grad_mag)
    if max_grad < 1e-3:
        v_max = np.nanmax(np.abs(V))
        if v_max > 1e-3:
            return np.abs(V) >= (threshold * v_max)
        return np.zeros_like(V, dtype=bool)

    jump_threshold = threshold * max_grad
    shell = grad_mag > jump_threshold
    mask = ndimage.binary_fill_holes(shell)

    if np.sum(mask) == 0:
        v_max = np.nanmax(np.abs(V))
        if v_max > 1e-3:
            mask = np.abs(V) >= (threshold * v_max)

    return mask


def voxelize_sphere(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    center: Sequence[float] = (0.0, 0.0, 0.0),
    radius: float = 1.0
) -> np.ndarray:
    """
    Generates a 3D boolean mask representing a spherical conductor or boundary.

    Parameters
    ----------
    X, Y, Z : np.ndarray
        3D coordinate meshgrids (e.g. from np.meshgrid(..., indexing='ij')).
    center : Sequence[float]
        (x, y, z) coordinates of sphere center in meters.
    radius : float
        Radius of sphere in meters.

    Returns
    -------
    np.ndarray (bool)
        3D boolean mask where True indicates points inside or on the sphere surface.
    """
    cx, cy, cz = center
    dist_sq = (X - cx)**2 + (Y - cy)**2 + (Z - cz)**2
    return dist_sq <= (radius ** 2)


def voxelize_box(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    min_bounds: Sequence[float],
    max_bounds: Sequence[float]
) -> np.ndarray:
    """
    Generates a 3D boolean mask representing an axis-aligned rectangular conductor or body.

    Parameters
    ----------
    X, Y, Z : np.ndarray
        3D coordinate meshgrids.
    min_bounds : Sequence[float]
        (xmin, ymin, zmin) lower bounds of the box.
    max_bounds : Sequence[float]
        (xmax, ymax, zmax) upper bounds of the box.

    Returns
    -------
    np.ndarray (bool)
        3D boolean mask where True indicates points inside the box.
    """
    return (
        (X >= min_bounds[0]) & (X <= max_bounds[0]) &
        (Y >= min_bounds[1]) & (Y <= max_bounds[1]) &
        (Z >= min_bounds[2]) & (Z <= max_bounds[2])
    )


def voxelize_cylinder(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    p_start: Sequence[float],
    p_end: Sequence[float],
    radius: float
) -> np.ndarray:
    """
    Generates a 3D boolean mask representing a cylindrical conductor (e.g., boom or antenna wire).

    Parameters
    ----------
    X, Y, Z : np.ndarray
        3D coordinate meshgrids.
    p_start : Sequence[float]
        (x, y, z) starting point of cylinder axis.
    p_end : Sequence[float]
        (x, y, z) ending point of cylinder axis.
    radius : float
        Radius of the cylinder in meters.

    Returns
    -------
    np.ndarray (bool)
        3D boolean mask where True indicates points inside the finite cylinder.
    """
    p1 = np.array(p_start, dtype=float)
    p2 = np.array(p_end, dtype=float)
    axis = p2 - p1
    axis_len_sq = np.dot(axis, axis)

    if axis_len_sq < 1e-12:
        return voxelize_sphere(X, Y, Z, center=p1, radius=radius)

    dx = X - p1[0]
    dy = Y - p1[1]
    dz = Z - p1[2]

    # Projection factor t along axis
    t = (dx * axis[0] + dy * axis[1] + dz * axis[2]) / axis_len_sq

    # Distance perpendicular to axis
    closest_x = p1[0] + t * axis[0]
    closest_y = p1[1] + t * axis[1]
    closest_z = p1[2] + t * axis[2]

    dist_perp_sq = (X - closest_x)**2 + (Y - closest_y)**2 + (Z - closest_z)**2

    mask = (t >= 0.0) & (t <= 1.0) & (dist_perp_sq <= (radius ** 2))
    if np.sum(mask) == 0 and X.shape[0] > 1:
        # Thin wire fallback on coarse grids: ensure antenna conductor is represented
        dx_val = abs(X[1, 0, 0] - X[0, 0, 0])
        effective_r = max(radius, 0.75 * dx_val)
        mask = (t >= 0.0) & (t <= 1.0) & (dist_perp_sq <= (effective_r ** 2))

    return mask
