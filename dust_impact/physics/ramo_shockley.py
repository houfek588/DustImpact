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
