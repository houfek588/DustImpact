# -*- coding: utf-8 -*-
"""
Numerical integrators for kinetic particle motion (Leapfrog pusher).
Supports Numba JIT acceleration with pure NumPy fallback.
"""

import numpy as np
from typing import Tuple, Union

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
def leapfrog_step_2d(x: np.ndarray, y: np.ndarray, vx: np.ndarray, vy: np.ndarray,
                     Ex: np.ndarray, Ey: np.ndarray, q_over_m: float, dt: float,
                     active: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Executes symplectic Leapfrog time step for active 2D particles.

    vx_{n+1/2} = vx_{n-1/2} + (q/m) * Ex * dt
    x_{n+1}    = x_n + vx_{n+1/2} * dt
    """
    vx_new = vx.copy()
    vy_new = vy.copy()
    x_new = x.copy()
    y_new = y.copy()

    vx_new[active] += q_over_m * Ex[active] * dt
    vy_new[active] += q_over_m * Ey[active] * dt

    x_new[active] += vx_new[active] * dt
    y_new[active] += vy_new[active] * dt

    return x_new, y_new, vx_new, vy_new


@njit(fastmath=True)
def leapfrog_step_3d(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                     vx: np.ndarray, vy: np.ndarray, vz: np.ndarray,
                     Ex: np.ndarray, Ey: np.ndarray, Ez: np.ndarray,
                     q_over_m: float, dt: float,
                     active: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Executes symplectic Leapfrog time step for active 3D particles.
    """
    vx_new = vx.copy()
    vy_new = vy.copy()
    vz_new = vz.copy()
    x_new = x.copy()
    y_new = y.copy()
    z_new = z.copy()

    vx_new[active] += q_over_m * Ex[active] * dt
    vy_new[active] += q_over_m * Ey[active] * dt
    vz_new[active] += q_over_m * Ez[active] * dt

    x_new[active] += vx_new[active] * dt
    y_new[active] += vy_new[active] * dt
    z_new[active] += vz_new[active] * dt

    return x_new, y_new, z_new, vx_new, vy_new, vz_new


def check_cfl_condition(dt: float, dx_min: float, v_char: float, safety_factor: float = 1.0) -> Tuple[bool, float, float]:
    """
    Checks the PIC kinetic numerical Courant-Friedrichs-Lewy (CFL) condition:
        v * dt <= dx_min

    Parameters:
    -----------
    dt : float
        Simulation time step [s].
    dx_min : float
        Minimum spatial grid resolution [m].
    v_char : float
        Characteristic maximum particle velocity [m/s].
    safety_factor : float, optional
        Safety multiplier for maximum recommended time step (default 1.0).

    Returns:
    --------
    is_stable : bool
        True if v_char * dt <= dx_min, False otherwise.
    cfl_number : float
        Courant number = (v_char * dt) / dx_min.
    dt_max_recommended : float
        Maximum recommended time step = (dx_min / v_char) * safety_factor [s].
    """
    if v_char <= 0 or dx_min <= 0:
        return True, 0.0, float('inf')

    cfl_number = float((v_char * dt) / dx_min)
    dt_max_recommended = float((dx_min / v_char) * safety_factor)
    is_stable = bool(cfl_number <= 1.0)
    return is_stable, cfl_number, dt_max_recommended
