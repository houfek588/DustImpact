# -*- coding: utf-8 -*-
"""
Sub-grid thin-wire model for antennas whose radius is much smaller than the grid spacing.

A wire of radius a << dx cannot be resolved by the grid: holding its voxel nodes at the wire
potential makes it behave like a much thicker wire of some effective radius r_eff ~ dx. Its
weighting potential, capacitance and the induced charge from a passing plasma cloud then depend on the
grid and are wrong by a factor ~ ln(L/r_eff) / ln(L/a).

Model: near a thin wire carrying line charge lambda the potential is logarithmic,
V(rho) = V_wire - lambda / (2 pi eps0) ln(rho / a). The wire nodes on the grid sit at the effective
radius r_eff, so the node potential differs from the wire potential by

    V_wire - V_node = lambda / (2 pi eps0) * ln(r_eff / a).

The wire nodes therefore become free nodes carrying the charge q_n = kappa_n (V_wire - V_n), with
kappa_n = 2 pi eps0 w_n / ln(r_eff / a) and w_n the wire length represented by node n. In the discrete
Poisson equation (-Lap V)_n = q_n / (eps0 dV) this is the diagonal coupling

    (-Lap V)_n + beta_n V_n = beta_n V_wire,     beta_n = 2 pi w_n / (dV ln(r_eff / a)),

which keeps the system symmetric positive definite and is solved by DirichletLaplaceSolver3D (and by
the self-field Poisson solver with V_wire = 0). Green reciprocity holds exactly for the coupled system,
so Ramo-Shockley currents (-q Vw) and the capacitance matrix stay consistent.

r_eff of the voxel node set (any orientation, anisotropic cells) is calibrated numerically on a long
auxiliary wire with the same axis line on the same lattice (see calibrate_effective_radius).
"""

from typing import Callable, Sequence, Tuple

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from dust_impact.physics.constants import eps_0
from dust_impact.numerics.poisson import DirichletLaplaceSolver3D, assemble_negative_laplacian_3d


def thin_wire_voxel_radius(radius: float, spacing: Sequence[float]) -> float:
    """
    Radius used to voxelize a thin wire: the physical radius, but at least 0.75 of the largest grid
    spacing, so that the node chain is connected for any wire orientation (an axis-aligned wire through
    grid nodes still maps to a single line of nodes).
    """
    return max(float(radius), 0.75 * max(float(h) for h in spacing))


def _perpendicular_basis(axis: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Two unit vectors orthogonal to axis and to each other."""
    helper = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(axis, helper)
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(axis, e1)


def calibrate_effective_radius(
    p_start: Sequence[float],
    p_end: Sequence[float],
    voxel_radius: float,
    grid: Tuple[np.ndarray, np.ndarray, np.ndarray],
    calibration_length_cells: float = 40.0,
    margin_cells: int = 10,
) -> float:
    """
    Effective radius r_eff [m] of a voxelized wire on the 7-point Laplace stencil.

    r_eff depends only on how the wire axis crosses the lattice (direction and offset), not on the wire
    length, so it is measured on a long auxiliary wire with the same axis line, on an auxiliary grid
    that shares the lattice of `grid` (same spacing and node positions modulo the spacing). The
    auxiliary wire is held at 1 V in its grounded box; the line charge lambda of its central third
    follows from the discrete Gauss law, and the potential V(rho) is averaged over 8 azimuths and
    5 positions along the central part at rho = 2h and 3h. The midpoint potential of a uniformly
    charged segment of length L, lambda / (2 pi eps0) asinh(L / 2r), then gives
    r_eff = (L/2) / sinh(2 pi eps0 (1 - V(rho)) / lambda + asinh(L / 2 rho)).

    Parameters
    ----------
    p_start, p_end : Sequence[float]
        Wire axis end points [m] (only the axis line is used).
    voxel_radius : float
        Radius used to voxelize the wire nodes [m] (see thin_wire_voxel_radius).
    grid : Tuple[np.ndarray, np.ndarray, np.ndarray]
        Node coordinates (x_grid, y_grid, z_grid) of the uniform PIC grid.
    calibration_length_cells : float
        Length of the auxiliary wire in units of the largest grid spacing.
    margin_cells : int
        Distance of the auxiliary box faces from the wire in cells.

    Returns
    -------
    float
        Radius of the ideal cylinder whose exterior potential matches the voxel wire.
    """
    from dust_impact.geometry.voxelizer import voxelize_cylinder

    spacing = tuple(float(g[1] - g[0]) for g in grid)
    h = max(spacing)
    p1, p2 = np.asarray(p_start, dtype=float), np.asarray(p_end, dtype=float)
    axis = (p2 - p1) / np.linalg.norm(p2 - p1)
    mid = 0.5 * (p1 + p2)
    length = calibration_length_cells * h
    a1, a2 = mid - 0.5 * length * axis, mid + 0.5 * length * axis

    # Auxiliary grid on the same lattice as the PIC grid, enclosing the auxiliary wire with a margin
    aux = []
    for k, g in enumerate(grid):
        lo = min(a1[k], a2[k]) - margin_cells * spacing[k]
        hi = max(a1[k], a2[k]) + margin_cells * spacing[k]
        i_lo = int(np.floor((lo - g[0]) / spacing[k]))
        i_hi = int(np.ceil((hi - g[0]) / spacing[k]))
        aux.append(g[0] + spacing[k] * np.arange(i_lo, i_hi + 1))
    x, y, z = aux
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    wire = voxelize_cylinder(X, Y, Z, a1, a2, voxel_radius)
    shape = wire.shape

    fixed = wire.copy()
    fixed[[0, -1], :, :] = True
    fixed[:, [0, -1], :] = True
    fixed[:, :, [0, -1]] = True
    V = DirichletLaplaceSolver3D(fixed, spacing).solve(wire.astype(np.float64))

    # Line charge of the central third from the discrete Gauss law on the wire nodes
    t = ((X - a1[0]) * axis[0] + (Y - a1[1]) * axis[1] + (Z - a1[2]) * axis[2]) / length
    central = wire & (t >= 1.0 / 3.0) & (t < 2.0 / 3.0)
    q_nodes = eps_0 * float(np.prod(spacing)) * (
        assemble_negative_laplacian_3d(*shape, *spacing) @ V.ravel()
    ).reshape(shape)
    lam = float(np.sum(q_nodes[central])) / (length / 3.0)

    e1, e2 = _perpendicular_basis(axis)
    interp = RegularGridInterpolator((x, y, z), V)
    phi = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    ring = np.cos(phi)[:, None] * e1[None, :] + np.sin(phi)[:, None] * e2[None, :]
    centres = mid[None, :] + np.linspace(-0.1, 0.1, 5)[:, None] * length * axis[None, :]
    estimates = []
    for rho in (2.0 * h, 3.0 * h):
        pts = (centres[:, None, :] + rho * ring[None, :, :]).reshape(-1, 3)
        V_rho = float(np.mean(interp(pts)))
        delta = 2.0 * np.pi * eps_0 * (1.0 - V_rho) / lam
        estimates.append(0.5 * length / np.sinh(delta + np.arcsinh(0.5 * length / rho)))
    return float(np.mean(estimates))


def thin_wire_coupling(
    wire_mask: np.ndarray,
    wire_length: float,
    radius: float,
    effective_radius: float,
    spacing: Sequence[float],
) -> np.ndarray:
    """
    Coupling coefficients beta [1/m^2] of the thin-wire model on the wire nodes (0 elsewhere).

    Parameters
    ----------
    wire_mask : np.ndarray
        Boolean node mask of the wire (nodes that represent it on the grid).
    wire_length : float
        Wire length represented by these nodes [m]; every node carries w = wire_length / N_nodes.
    radius : float
        Physical wire radius a [m].
    effective_radius : float
        Calibrated effective radius r_eff of the node set [m]; must exceed radius.
    spacing : Sequence[float]
        Grid spacings (dx, dy, dz) [m].

    Returns
    -------
    np.ndarray
        Array of the grid shape with beta_n = 2 pi w / (dV ln(r_eff / a)) on the wire nodes.
    """
    if effective_radius <= radius:
        raise ValueError("The thin-wire model needs a wire thinner than its grid representation (r_eff > a).")
    n_nodes = int(np.count_nonzero(wire_mask))
    beta = np.zeros(wire_mask.shape)
    if n_nodes == 0:
        return beta
    w = wire_length / n_nodes
    beta[wire_mask] = 2.0 * np.pi * w / (float(np.prod(spacing)) * np.log(effective_radius / radius))
    return beta


def wire_length_outside(
    p_start: Sequence[float], p_end: Sequence[float], inside: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    samples: int = 2001,
) -> float:
    """Length of the wire axis that lies outside the body described by `inside(x, y, z)` [m]."""
    p1, p2 = np.asarray(p_start, dtype=float), np.asarray(p_end, dtype=float)
    s = np.linspace(0.0, 1.0, samples)
    pts = p1[None, :] + s[:, None] * (p2 - p1)[None, :]
    outside = ~np.asarray(inside(pts[:, 0], pts[:, 1], pts[:, 2]), dtype=bool)
    return float(np.linalg.norm(p2 - p1) * np.mean(outside))
