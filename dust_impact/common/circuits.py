# -*- coding: utf-8 -*-
"""
Circuit models (RC filter integration) for antenna signal processing.
"""

import numpy as np


def integrate_rc_circuit(tot_curr: np.ndarray, dt: float, R: float, C: float, V_initial: float = 0.0) -> np.ndarray:
    """
    Integrates total current through parallel RC circuit:
      dV/dt = I_tot/C - V/(R*C)

    Parameters:
    -----------
    tot_curr : np.ndarray
        Array of total current at each timestep (1D).
    dt : float
        Timestep size [s].
    R : float
        Resistance [Ohm].
    C : float
        Capacitance [F].
    V_initial : float
        Initial voltage [V].

    Returns:
    --------
    voltage : np.ndarray
        Voltage array over time.
    """
    steps = len(tot_curr)
    voltage = np.zeros(steps)
    voltage[0] = V_initial

    if C <= 0:
        return voltage

    alpha = dt / (R * C) if (R * C > 0) else 0.0
    beta = dt / C

    for step in range(steps - 1):
        voltage[step + 1] = voltage[step] + beta * tot_curr[step] - alpha * voltage[step]

    return voltage
