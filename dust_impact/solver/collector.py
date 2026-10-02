# -*- coding: utf-8 -*-
"""
Antenna Circuit Collector module for 3D PIC Simulation.
Records induced and collected currents for each antenna probe,
integrates the coupled lumped-element RC circuit equations, and formats output signals.
"""

from typing import Dict, List, Any
import numpy as np

from dust_impact.solver.config_loader import SimulationParams3D, SimulationToggles3D


class AntennaCircuitCollector:
    """
    Manages current collection, Ramo-Shockley induction tracking, and RC circuit integration for antennas.
    """

    def __init__(self, params: SimulationParams3D, toggles: SimulationToggles3D, num_antennas: int):
        self.p = params
        self.toggles = toggles
        self.num_antennas = num_antennas

        steps = self.p.steps
        self.ind_curr_e = np.zeros((num_antennas, steps), dtype=np.float64)
        self.ind_curr_i = np.zeros((num_antennas, steps), dtype=np.float64)
        self.col_curr_e = np.zeros((num_antennas, steps), dtype=np.float64)
        self.col_curr_i = np.zeros((num_antennas, steps), dtype=np.float64)
        self.tot_curr = np.zeros((num_antennas, steps), dtype=np.float64)

        self.voltage_ant = np.zeros((num_antennas, steps), dtype=np.float64)
        for a_idx in range(self.num_antennas):
            if getattr(self.toggles, 'enable_antenna_bias_voltage', getattr(self.toggles, 'enable_antenna_bias', True)):
                self.voltage_ant[a_idx, 0] = self.p.V_bias[a_idx] if a_idx < len(self.p.V_bias) else 0.0

    def record_step_currents(
        self,
        step: int,
        col_e: np.ndarray,
        col_i: np.ndarray,
        ind_e: np.ndarray,
        ind_i: np.ndarray,
    ) -> None:
        """Records currents for all antennas at a given step."""
        self.col_curr_e[:, step] = col_e
        self.col_curr_i[:, step] = col_i
        self.ind_curr_e[:, step] = ind_e
        self.ind_curr_i[:, step] = ind_i

    def update_circuit(self, step: int) -> None:
        """Computes total currents and updates RC circuit ODE for step."""
        for a_idx in range(self.num_antennas):
            I_tot = (
                self.ind_curr_e[a_idx, step]
                + self.ind_curr_i[a_idx, step]
                + self.col_curr_e[a_idx, step]
                + self.col_curr_i[a_idx, step]
            )
            self.tot_curr[a_idx, step] = I_tot

            if step > 0:
                c_ant = self.p.C_ant[a_idx] if a_idx < len(self.p.C_ant) else 1e-12
                r_ant = self.p.R_ant[a_idx] if a_idx < len(self.p.R_ant) else 1e6
                dV_dt = I_tot / c_ant
                if getattr(self.toggles, 'enable_rc_circuit_response', True):
                    v_bias = (
                        (self.p.V_bias[a_idx] if a_idx < len(self.p.V_bias) else 0.0)
                        if getattr(self.toggles, 'enable_antenna_bias_voltage', True)
                        else 0.0
                    )
                    dV_dt -= (self.voltage_ant[a_idx, step - 1] - v_bias) / (r_ant * c_ant)

                self.voltage_ant[a_idx, step] = self.voltage_ant[a_idx, step - 1] + dV_dt * self.p.dt

    def build_smoothed_signals(self) -> Dict[str, List[np.ndarray]]:
        """Constructs boxcar-smoothed signals for induced, collected, and total current."""
        k_size = min(100, max(1, self.p.steps))
        kernel = np.ones(k_size) / k_size

        smooth_induced = []
        smooth_collected = []
        smooth_total = []

        for a_idx in range(self.num_antennas):
            smooth_induced.append(
                np.convolve(self.ind_curr_e[a_idx] + self.ind_curr_i[a_idx], kernel, mode='same')
            )
            smooth_collected.append(
                np.convolve(self.col_curr_e[a_idx] + self.col_curr_i[a_idx], kernel, mode='same')
            )
            smooth_total.append(
                np.convolve(self.tot_curr[a_idx], kernel, mode='same')
            )

        return {
            'smooth_induced': smooth_induced,
            'smooth_collected': smooth_collected,
            'smooth_total': smooth_total,
        }
