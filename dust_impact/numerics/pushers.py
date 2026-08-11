# -*- coding: utf-8 -*-
"""
Numerical integrators for kinetic particle motion (Leapfrog pusher).
"""

import numpy as np
from typing import Tuple, Union


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
