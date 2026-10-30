# -*- coding: utf-8 -*-
"""
Physics of Ramo-Shockley theorem for non-contact induced currents.
I_ind = - sum q_macro * (v . E_w)
"""

import numpy as np
from typing import Union


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
