# -*- coding: utf-8 -*-
"""
Physics of Ramo-Shockley theorem for non-contact induced currents.
I_ind = - sum q_macro * (v . E_w)
"""

import numpy as np
from typing import List, Tuple, Union

from dust_impact.physics.constants import eps_0


def calc_induced_current(q_macro: float, vx: np.ndarray, vy: np.ndarray, Ewx: np.ndarray, Ewy: np.ndarray,
                         vz: Union[np.ndarray, None] = None, Ewz: Union[np.ndarray, None] = None) -> float:
    """
    Calculates instantaneous Ramo-Shockley induced current for active macro-particles.

    I_ind = - q_macro * sum( vx * Ewx + vy * Ewy + vz * Ewz )

    Parameters:
    -----------
    q_macro : float
        Macro-particle charge (with sign).
    vx, vy, vz : np.ndarray
        Particle velocity components [m/s].
    Ewx, Ewy, Ewz : np.ndarray
        Weighting field vectors [1/m].

    Returns:
    --------
    I_ind : float
        Instantaneous induced current [A].
    """
    v_dot_Ew = vx * Ewx + vy * Ewy
    if vz is not None and Ewz is not None:
        v_dot_Ew = v_dot_Ew + vz * Ewz

    return float(- q_macro * np.sum(v_dot_Ew))


def calc_induced_current_from_potential(q_macro: float, Vw_old: np.ndarray, Vw_new: np.ndarray, dt: float) -> float:
    """
    Calculates the Ramo-Shockley induced current from the change of the weighting potential.

    The charge induced on an electrode by a point charge q at position r is Q = q * Vw(r)
    (Vw = 1 on the electrode, 0 on all other conductors), hence over one time step:

        I_ind = q_macro * sum( Vw(r^{n+1}) - Vw(r^n) ) / dt

    Unlike the v . E_w form, this discrete form conserves charge exactly: the time integral
    of the current along any trajectory equals q * [Vw(r_end) - Vw(r_start)], independently of
    the grid gradient error and of the time centering of positions and velocities.

    Parameters:
    -----------
    q_macro : float
        Macro-particle charge (with sign) [C].
    Vw_old, Vw_new : np.ndarray
        Weighting potential sampled at the particle positions before and after the step [-].
    dt : float
        Time step [s].

    Returns:
    --------
    I_ind : float
        Induced current averaged over the time step [A].
    """
    return float(q_macro * np.sum(Vw_new - Vw_old) / dt)


def vacuum_capacitance_matrix(Vw_list: List[np.ndarray], conductor_masks: List[np.ndarray],
                              spacing: Tuple[float, float, float]) -> np.ndarray:
    """
    Vacuum (Maxwell) capacitance matrix of the conductors on the PIC grid, consistent with Ramo-Shockley.

    Vw_list[j] is the weighting potential of conductor j (1 on conductor j, 0 on the other conductors
    and on the grounded outer boundary). By the discrete Gauss law of the 7-point Poisson stencil used
    for the self-field, the charge on conductor i when conductor j is at 1 V and all others at 0 V is

        C_ij = eps_0 * dV * sum_{n in conductor i} (-Lap Vw_j)(n).

    The matrix is symmetrised, which is exact for discrete-harmonic weighting potentials (analytical
    geometry) and removes the small asymmetry of sampled SPIS fields. Together with the induced charge
    -q Vw_i(r) of every free macro-particle, Q_i = sum_j C_ij V_j - sum_p q_p Vw_i(r_p) is the total
    charge on conductor i, which closes the circuit equations of floating conductors.

    Parameters
    ----------
    Vw_list : List[np.ndarray]
        Weighting potentials (Nx, Ny, Nz) of the conductors, in the electrode order of the circuit.
    conductor_masks : List[np.ndarray]
        Boolean node masks of the same conductors (disjoint).
    spacing : Tuple[float, float, float]
        Grid spacings (dx, dy, dz) [m].

    Returns
    -------
    np.ndarray
        Symmetric capacitance matrix of shape (n, n) [F].
    """
    from dust_impact.numerics.poisson import assemble_negative_laplacian_3d

    n = len(Vw_list)
    if n == 0:
        return np.zeros((0, 0))
    shape = Vw_list[0].shape
    A = assemble_negative_laplacian_3d(*shape, *spacing)
    dV = float(np.prod(spacing))
    C = np.zeros((n, n))
    for j, Vw in enumerate(Vw_list):
        charge_density = A @ np.asarray(Vw, dtype=np.float64).ravel()
        for i, mask in enumerate(conductor_masks):
            C[i, j] = eps_0 * dV * float(np.sum(charge_density[np.asarray(mask, dtype=bool).ravel()]))
    return 0.5 * (C + C.T)
