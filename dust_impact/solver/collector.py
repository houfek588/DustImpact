# -*- coding: utf-8 -*-
"""
Antenna Circuit Collector module for 3D PIC Simulation.
Records induced and collected currents for each electrode (antennas and, with a floating spacecraft,
the spacecraft body), integrates the lumped-element circuit equations, and formats output signals.
"""

from typing import Dict, List, Optional
import numpy as np

from dust_impact.solver.config_loader import SimulationParams3D, SimulationToggles3D


class AntennaCircuitCollector:
    """
    Manages current collection, Ramo-Shockley induction tracking, and circuit integration.

    Electrodes are the antennas (indices 0 .. num_antennas-1) and, when
    toggles.enable_floating_spacecraft is set, the spacecraft body (index num_antennas). The voltages
    obey the coupled circuit equations

        C_tot dV/dt = I(t) - G (V - V_bias),

    where I is the induced plus collected current delivered to each electrode, G = diag(1/R) the bias
    conductances (zero when enable_rc_circuit_response is off or no resistance is given) and
      - fixed spacecraft:    C_tot = diag(C_ant)                    (independent antenna RC circuits),
      - floating spacecraft: C_tot = C_vac + diag(C_ant..., C_sc)   (vacuum capacitance matrix of the
                                                                    conductors plus readout capacitances).
    This follows from charge conservation: Q_i = sum_j C_vac,ij V_j - sum_p q_p Vw_i(r_p) changes only
    by the currents through the readout, so the Ramo-Shockley current q dVw/dt drives the vacuum part.
    """

    def __init__(
        self,
        params: SimulationParams3D,
        toggles: SimulationToggles3D,
        num_antennas: int,
        capacitance_matrix: Optional[np.ndarray] = None,
    ):
        self.p = params
        self.toggles = toggles
        self.num_antennas = num_antennas
        self.floating_spacecraft = bool(getattr(toggles, 'enable_floating_spacecraft', False))
        self.num_electrodes = num_antennas + (1 if self.floating_spacecraft else 0)

        steps = self.p.steps
        n_el = self.num_electrodes
        self.ind_curr_e_all = np.zeros((n_el, steps), dtype=np.float64)
        self.ind_curr_i_all = np.zeros((n_el, steps), dtype=np.float64)
        self.col_curr_e_all = np.zeros((n_el, steps), dtype=np.float64)
        self.col_curr_i_all = np.zeros((n_el, steps), dtype=np.float64)
        self.tot_curr_all = np.zeros((n_el, steps), dtype=np.float64)
        self.voltage = np.zeros((n_el, steps), dtype=np.float64)

        # Antenna views (shared memory) keep the per-antenna interface of the fixed-spacecraft model
        self.ind_curr_e = self.ind_curr_e_all[:num_antennas]
        self.ind_curr_i = self.ind_curr_i_all[:num_antennas]
        self.col_curr_e = self.col_curr_e_all[:num_antennas]
        self.col_curr_i = self.col_curr_i_all[:num_antennas]
        self.tot_curr = self.tot_curr_all[:num_antennas]
        self.voltage_ant = self.voltage[:num_antennas]
        self.voltage_sc = self.voltage[num_antennas] if self.floating_spacecraft else None
        self.current_sc = self.tot_curr_all[num_antennas] if self.floating_spacecraft else None

        self.V_bias_el = self._electrode_biases()
        self.voltage[:, 0] = self.V_bias_el
        self._build_circuit(capacitance_matrix)

    def _electrode_biases(self) -> np.ndarray:
        """Bias (initial and relaxation) potential of each electrode [V]."""
        bias_on = getattr(self.toggles, 'enable_antenna_bias_voltage', getattr(self.toggles, 'enable_antenna_bias', True))
        biases = [
            (self.p.V_bias[a] if a < len(self.p.V_bias) else 0.0) if bias_on else 0.0
            for a in range(self.num_antennas)
        ]
        if self.floating_spacecraft:
            biases.append(self.p.V_sc_bias)
        return np.array(biases, dtype=np.float64)

    def _build_circuit(self, capacitance_matrix: Optional[np.ndarray]) -> None:
        """Assembles C_tot^-1 and the bias conductances G of the electrode circuit."""
        n_ant, n_el = self.num_antennas, self.num_electrodes
        c_ext = [self.p.C_ant[a] if a < len(self.p.C_ant) else 1e-12 for a in range(n_ant)]
        r_bias: List[Optional[float]] = [self.p.R_ant[a] if a < len(self.p.R_ant) else 1e6 for a in range(n_ant)]
        if self.floating_spacecraft:
            c_ext.append(self.p.C_sc)
            r_bias.append(self.p.R_sc)

        if self.floating_spacecraft:
            if capacitance_matrix is None or np.shape(capacitance_matrix) != (n_el, n_el):
                raise ValueError(
                    "enable_floating_spacecraft requires the vacuum capacitance matrix of the antennas and the "
                    f"spacecraft ({n_el}x{n_el}) from the prepared geometry."
                )
            C_tot = np.asarray(capacitance_matrix, dtype=np.float64) + np.diag(c_ext)
        else:
            C_tot = np.diag(np.asarray(c_ext, dtype=np.float64))
        self.C_tot = C_tot
        self.inv_C = np.linalg.inv(C_tot) if n_el > 0 else np.zeros((0, 0))

        rc_enabled = getattr(self.toggles, 'enable_rc_circuit_response', True)
        g = [
            1.0 / r if (rc_enabled and r is not None and np.isfinite(r) and r > 0.0) else 0.0
            for r in r_bias
        ]
        self.G = np.asarray(g, dtype=np.float64)

    def record_step_currents(
        self,
        step: int,
        col_e: np.ndarray,
        col_i: np.ndarray,
        ind_e: np.ndarray,
        ind_i: np.ndarray,
    ) -> None:
        """Records currents for all electrodes at a given step (arrays of length num_electrodes)."""
        self.col_curr_e_all[:, step] = col_e
        self.col_curr_i_all[:, step] = col_i
        self.ind_curr_e_all[:, step] = ind_e
        self.ind_curr_i_all[:, step] = ind_i

    def _rate(self, V: np.ndarray, I: np.ndarray) -> np.ndarray:
        """dV/dt = C_tot^-1 [I - G (V - V_bias)]."""
        return self.inv_C @ (I - self.G * (V - self.V_bias_el))

    def update_circuit(self, step: int) -> None:
        """
        Computes total currents and advances the electrode circuit over the PIC step `step`.

        The currents of step k are step averages (charge delivered between t_k and t_k + dt, see
        ParticleEnsemble.push_species), so they are held constant over that interval and voltage[:, k]
        is the voltage at the end of step k (the start-of-step value is voltage[:, k-1], or the bias for
        k = 0). Without bias resistors this conserves charge exactly: C_tot dV = I_k dt.
        Uses 4th-order Runge-Kutta (RK4) by default, Forward Euler if circuit_solver == 'euler'.
        """
        if self.num_electrodes == 0:
            return
        dt = self.p.dt
        I_step = (
            self.ind_curr_e_all[:, step] + self.ind_curr_i_all[:, step]
            + self.col_curr_e_all[:, step] + self.col_curr_i_all[:, step]
        )
        self.tot_curr_all[:, step] = I_step

        V_prev = self.voltage[:, step - 1] if step > 0 else self.V_bias_el
        solver_type = str(getattr(self.toggles, 'circuit_solver', getattr(self.p, 'circuit_solver', 'rk4'))).lower()
        if solver_type == 'euler':
            self.voltage[:, step] = V_prev + dt * self._rate(V_prev, I_step)
            return

        k1 = self._rate(V_prev, I_step)
        k2 = self._rate(V_prev + 0.5 * dt * k1, I_step)
        k3 = self._rate(V_prev + 0.5 * dt * k2, I_step)
        k4 = self._rate(V_prev + dt * k3, I_step)
        self.voltage[:, step] = V_prev + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

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
