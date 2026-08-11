# -*- coding: utf-8 -*-
"""
Physics of plasma cloud thermal velocity distributions and particle emissions.
Half-Maxwellian / Rayleigh distributions for emitted electrons and ions.
"""

import numpy as np
from typing import Tuple
from dust_impact.physics.constants import e, m_e, amu


def generate_half_maxwellian_velocities(N_particles: int, T_eV: float, mass_kg: float) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Generates thermal velocities for emitted particles in a half-space.

    Parameters:
    -----------
    N_particles : int
        Number of particles.
    T_eV : float
        Temperature in eV.
    mass_kg : float
        Particle mass in kg.

    Returns:
    --------
    vx, vy, v_th : Tuple[np.ndarray, np.ndarray, float]
        Perpendicular positive velocity vx, parallel velocity vy, and thermal velocity v_th.
    """
    v_th = np.sqrt(2.0 * T_eV * e / mass_kg)
    theta = np.random.uniform(-np.pi / 2, np.pi / 2, N_particles)
    v_mag = np.random.normal(v_th, v_th / 2.0, N_particles)
    vx = np.abs(v_mag * np.cos(theta))
    vy = v_mag * np.sin(theta)
    return vx, vy, float(v_th)
