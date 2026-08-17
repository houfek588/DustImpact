# -*- coding: utf-8 -*-
"""
2D PIC Simulation Core Solver.
Specific solver logic for 2D geometry, using shared physics and numerics subpackages.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from typing import Tuple, Dict, Any

from dust_impact.physics.constants import e, m_e, eps_0
from dust_impact.physics.ramo_shockley import calc_induced_current
from dust_impact.numerics.interpolators import interp_field_2d
from dust_impact.numerics.pushers import leapfrog_step_2d
from dust_impact.common.circuits import integrate_rc_circuit
from dust_impact.sim2d.input_data import calc_Ew_2d, _dist_to_segment_sq, SimulationParams2D, SimulationToggles2D


class DustImpactSimulation2D:
    def __init__(self, params: SimulationParams2D, toggles: SimulationToggles2D):
        self.p = params
        self.toggles = toggles
        self.num_antennas = len(self.p.antennas)

        theta_e = np.random.uniform(-np.pi / 2, np.pi / 2, self.p.N_particles)
        v_mag_e = np.random.normal(self.p.v_th_e, self.p.v_th_e / 2, self.p.N_particles)
        self.vx_e = np.abs(v_mag_e * np.cos(theta_e))
        self.vy_e = v_mag_e * np.sin(theta_e)
        self.x_e = np.full(self.p.N_particles, 0.5 * self.p.dx)
        self.y_e = np.full(self.p.N_particles, self.p.y_impact)

        theta_i = np.random.uniform(-np.pi / 2, np.pi / 2, self.p.N_particles)
        v_mag_i = np.random.normal(self.p.v_th_i, self.p.v_th_i / 2, self.p.N_particles)
        self.vx_i = np.abs(v_mag_i * np.cos(theta_i))
        self.vy_i = v_mag_i * np.sin(theta_i)
        self.x_i = np.full(self.p.N_particles, 0.5 * self.p.dx)
        self.y_i = np.full(self.p.N_particles, self.p.y_impact)

        self.active_e = np.zeros(self.p.N_particles, dtype=bool)
        self.active_i = np.zeros(self.p.N_particles, dtype=bool)
        self.was_outside_e = np.zeros((self.num_antennas, self.p.N_particles), dtype=bool)
        self.was_outside_i = np.zeros((self.num_antennas, self.p.N_particles), dtype=bool)

        self.cloud_injected = False

        self.ind_curr_e = np.zeros((self.num_antennas, self.p.steps))
        self.ind_curr_i = np.zeros((self.num_antennas, self.p.steps))
        self.col_curr_e = np.zeros((self.num_antennas, self.p.steps))
        self.col_curr_i = np.zeros((self.num_antennas, self.p.steps))
        self.tot_curr = np.zeros((self.num_antennas, self.p.steps))

        self.voltage_ant = np.zeros((self.num_antennas, self.p.steps))
        for a_idx, ant in enumerate(self.p.antennas):
            if getattr(self.toggles, 'enable_antenna_bias_voltage', True):
                self.voltage_ant[a_idx, 0] = ant.get('V_eq', ant.get('V_bias', 0.0))

        self.V_bg_grid = np.zeros((self.p.Nx, self.p.Ny))
        self.V_self_grid = np.zeros((self.p.Nx, self.p.Ny))
        self.rho_grid = np.zeros((self.p.Nx, self.p.Ny))

        self.ant_masks = []
        self.combined_ant_mask = np.zeros((self.p.Nx, self.p.Ny), dtype=bool)
        for ant in self.p.antennas:
            d_sq, _, _ = _dist_to_segment_sq(self.p.X_mat, self.p.Y_mat, ant['x1'], ant['y1'], ant['x2'], ant['y2'])
            mask = d_sq <= ant['r'] ** 2
            self.ant_masks.append(mask)
            self.combined_ant_mask |= mask

        self._build_poisson_solver()
        self._update_background_field(0)

        self.history = {'V': [], 'rho': [], 't': [],
                        'x_e': [], 'y_e': [], 'x_i': [], 'y_i': [],
                        'vx_e': [], 'vy_e': [], 'vx_i': [], 'vy_i': []}

    def _get_1d_idx(self, i: int, j: int) -> int:
        return i * self.p.Ny + j

    def _build_poisson_solver(self):
        N_tot = self.p.Nx * self.p.Ny
        A_bg = sp.lil_matrix((N_tot, N_tot))
        A_self = sp.lil_matrix((N_tot, N_tot))

        dx2 = self.p.dx ** 2
        dy2 = self.p.dy ** 2

        env_mode = "Plazma pozadí se stíněním" if getattr(self.toggles, 'enable_debye_screening', True) else "Čisté vakuum bez stínění"
        print(f"Sestavuji Laplace/Poissonův řešič 2D (Režim prostředí: {env_mode})")

        for i in range(self.p.Nx):
            for j in range(self.p.Ny):
                idx = self._get_1d_idx(i, j)
                is_boundary = (i == 0) or (i == self.p.Nx - 1) or (j == 0) or (j == self.p.Ny - 1)

                if is_boundary or self.combined_ant_mask[i, j]:
                    A_bg[idx, idx] = 1.0
                    A_self[idx, idx] = 1.0
                else:
                    diag_val = -2 / dx2 - 2 / dy2
                    if getattr(self.toggles, 'enable_debye_screening', True):
                        diag_val -= (1.0 / self.p.debye_length ** 2)

                    A_bg[idx, idx] = diag_val
                    A_bg[idx, self._get_1d_idx(i - 1, j)] = 1 / dx2
                    A_bg[idx, self._get_1d_idx(i + 1, j)] = 1 / dx2
                    A_bg[idx, self._get_1d_idx(i, j - 1)] = 1 / dy2
                    A_bg[idx, self._get_1d_idx(i, j + 1)] = 1 / dy2

                    A_self[idx, idx] = -2 / dx2 - 2 / dy2
                    A_self[idx, self._get_1d_idx(i - 1, j)] = 1 / dx2
                    A_self[idx, self._get_1d_idx(i + 1, j)] = 1 / dx2
                    A_self[idx, self._get_1d_idx(i, j - 1)] = 1 / dy2
                    A_self[idx, self._get_1d_idx(i, j + 1)] = 1 / dy2

        self.solver_bg = spla.factorized(A_bg.tocsc())
        self.solver_self = spla.factorized(A_self.tocsc())

    def _update_background_field(self, step: int = 0):
        Ex_tot = np.zeros((self.p.Nx, self.p.Ny))
        Ey_tot = np.zeros((self.p.Nx, self.p.Ny))

        for a_idx, ant in enumerate(self.p.antennas):
            V_curr = self.voltage_ant[a_idx, max(0, step - 1)] if step > 0 else self.voltage_ant[a_idx, 0]
            V_bias = ant.get('V_eq', ant.get('V_bias', 0.0))
            dV = V_curr - V_bias

            if abs(dV) > 1e-6:
                Ewx, Ewy = calc_Ew_2d(self.p.X_mat, self.p.Y_mat, ant['x1'], ant['y1'], ant['x2'], ant['y2'], ant['w_width'])
                Ex_tot += dV * Ewx
                Ey_tot += dV * Ewy

        if not getattr(self.toggles, 'enable_spis_background_field', True):
            Ex_tot.fill(0.0)
            Ey_tot.fill(0.0)

        self.Ex_bg, self.Ey_bg = Ex_tot, Ey_tot

    def _solve_poisson_equation(self):
        if not self.cloud_injected or not getattr(self.toggles, 'enable_plasma_self_field', True):
            return

        edges_x = np.append(self.p.x_grid - self.p.dx / 2, self.p.x_grid[-1] + self.p.dx / 2)
        edges_y = np.append(self.p.y_grid - self.p.dy / 2, self.p.y_grid[-1] + self.p.dy / 2)

        counts_e, _, _ = np.histogram2d(self.x_e[self.active_e], self.y_e[self.active_e], bins=[edges_x, edges_y])
        counts_i, _, _ = np.histogram2d(self.x_i[self.active_i], self.y_i[self.active_i], bins=[edges_x, edges_y])

        self.rho_grid = (counts_i - counts_e) * self.p.q_macro / (self.p.dx * self.p.dy * self.p.A_sim)

        b_self = np.zeros(self.p.Nx * self.p.Ny)
        for i in range(self.p.Nx):
            for j in range(self.p.Ny):
                idx = self._get_1d_idx(i, j)
                is_boundary = (i == 0) or (i == self.p.Nx - 1) or (j == 0) or (j == self.p.Ny - 1)

                if not (is_boundary or self.combined_ant_mask[i, j]):
                    b_self[idx] = -self.rho_grid[i, j] / eps_0

        V_self_1d = self.solver_self(b_self)
        self.V_self_grid = V_self_1d.reshape((self.p.Nx, self.p.Ny))
        self.Ex_self, self.Ey_self = np.gradient(-self.V_self_grid, self.p.dx, self.p.dy)

    def _interp_field(self, x: np.ndarray, y: np.ndarray, Field_2D: np.ndarray) -> np.ndarray:
        return interp_field_2d(x, y, self.p.x_grid, self.p.y_grid, self.p.dx, self.p.dy, Field_2D)

    def _get_accel(self, x_act: np.ndarray, y_act: np.ndarray, mass: float, phys_charge: float) -> Tuple[
        np.ndarray, np.ndarray]:
        Ex_tot = self._interp_field(x_act, y_act, self.Ex_bg)
        Ey_tot = self._interp_field(x_act, y_act, self.Ey_bg)

        if getattr(self.toggles, 'enable_plasma_self_field', True):
            Ex_tot += self._interp_field(x_act, y_act, self.Ex_self)
            Ey_tot += self._interp_field(x_act, y_act, self.Ey_self)

        return (phys_charge / mass) * Ex_tot, (phys_charge / mass) * Ey_tot

    def _push_species(self, x: np.ndarray, y: np.ndarray, vx: np.ndarray, vy: np.ndarray,
                      active: np.ndarray, was_outside: np.ndarray, mass: float, phys_charge: float,
                      macro_charge: float) -> Tuple[np.ndarray, np.ndarray]:

        col_currs = np.zeros(self.num_antennas)
        ind_currs = np.zeros(self.num_antennas)
        if not np.any(active): return col_currs, ind_currs

        x_act = x[active]
        y_act = y[active]
        ax, ay = self._get_accel(x_act, y_act, mass, phys_charge)

        # Symplectic Leapfrog integration via numerics.pushers
        x[active], y[active], vx[active], vy[active] = leapfrog_step_2d(
            x[active], y[active], vx[active], vy[active], ax, ay, 1.0, self.p.dt, np.ones(np.sum(active), dtype=bool)
        )

        new_active = active.copy()

        for a_idx, ant in enumerate(self.p.antennas):
            x_act_new = x[new_active]
            y_act_new = y[new_active]

            d_sq, _, _ = _dist_to_segment_sq(x_act_new, y_act_new, ant['x1'], ant['y1'], ant['x2'], ant['y2'])
            is_outside = d_sq > ant['r'] ** 2

            was_out_act = was_outside[a_idx, new_active]
            crossed_ant = ~is_outside & was_out_act
            was_outside[a_idx, new_active] = is_outside

            if getattr(self.toggles, 'enable_antenna_particle_collection', True) and np.any(crossed_ant):
                n_crossed = np.sum(crossed_ant)
                eff = ant.get('collection_eff', 0.8)
                absorbed_mask = np.random.rand(n_crossed) < eff

                global_active_idx = np.where(new_active)[0]
                absorbed_global_indices = global_active_idx[crossed_ant][absorbed_mask]

                col_currs[a_idx] = len(absorbed_global_indices) * macro_charge / self.p.dt
                new_active[absorbed_global_indices] = False

            if np.any(new_active):
                Ewx, Ewy = calc_Ew_2d(x[new_active], y[new_active], ant['x1'], ant['y1'], ant['x2'], ant['y2'],
                                      ant['w_width'])
                ind_currs[a_idx] = calc_induced_current(macro_charge, vx[new_active], vy[new_active], Ewx, Ewy)

        new_active[new_active & (x < 0)] = False
        new_active[new_active & (x >= self.p.L_domain)] = False
        new_active[new_active & (y <= -self.p.H_domain)] = False
        new_active[new_active & (y >= self.p.H_domain)] = False

        active[:] = new_active
        return col_currs, ind_currs

    def _save_history(self, current_time: float):
        self.history['V'].append((self.V_bg_grid + self.V_self_grid).copy())
        self.history['rho'].append(self.rho_grid.copy())
        self.history['t'].append(current_time)

        self.history['x_e'].append(np.where(self.active_e, self.x_e, np.nan)[::self.p.plot_stride])
        self.history['y_e'].append(np.where(self.active_e, self.y_e, np.nan)[::self.p.plot_stride])
        self.history['x_i'].append(np.where(self.active_i, self.x_i, np.nan)[::self.p.plot_stride])
        self.history['y_i'].append(np.where(self.active_i, self.y_i, np.nan)[::self.p.plot_stride])

        self.history['vx_e'].append(np.where(self.active_e, self.vx_e, np.nan)[::self.p.plot_stride])
        self.history['vy_e'].append(np.where(self.active_e, self.vy_e, np.nan)[::self.p.plot_stride])
        self.history['vx_i'].append(np.where(self.active_i, self.vx_i, np.nan)[::self.p.plot_stride])
        self.history['vy_i'].append(np.where(self.active_i, self.vy_i, np.nan)[::self.p.plot_stride])

    def run(self) -> Dict[str, Any]:
        print(f"Spouštím 2D PIC Simulaci... ({self.p.steps} kroků na mřížce {self.p.Nx}x{self.p.Ny})")

        for step in range(self.p.steps):
            if not self.cloud_injected and step * self.p.dt >= self.p.t_delay:
                self.active_e[:] = True
                self.active_i[:] = True
                self.was_outside_e[:] = False
                self.was_outside_i[:] = False
                self.cloud_injected = True

            if step > 0:
                self._update_background_field(step)

            self._solve_poisson_equation()

            if self.cloud_injected:
                ce, ie = self._push_species(self.x_e, self.y_e, self.vx_e, self.vy_e, self.active_e, self.was_outside_e,
                                            m_e, -e, -self.p.q_macro)
                ci, ii = self._push_species(self.x_i, self.y_i, self.vx_i, self.vy_i, self.active_i, self.was_outside_i,
                                            self.p.m_i, e, self.p.q_macro)
                self.col_curr_e[:, step] = ce
                self.col_curr_i[:, step] = ci
                self.ind_curr_e[:, step] = ie
                self.ind_curr_i[:, step] = ii

            for a_idx, ant in enumerate(self.p.antennas):
                I_tot = self.ind_curr_e[a_idx, step] + self.ind_curr_i[a_idx, step] + self.col_curr_e[a_idx, step] + \
                        self.col_curr_i[a_idx, step]
                self.tot_curr[a_idx, step] = I_tot

                if step > 0:
                    dV_dt = I_tot / ant['C']
                    if getattr(self.toggles, 'enable_rc_circuit_response', True):
                        v_eq = ant.get('V_eq', ant.get('V_bias', 0.0)) if getattr(self.toggles, 'enable_antenna_bias_voltage', True) else 0.0
                        dV_dt -= (self.voltage_ant[a_idx, step - 1] - v_eq) / (ant['R'] * ant['C'])

                    self.voltage_ant[a_idx, step] = self.voltage_ant[a_idx, step - 1] + dV_dt * self.p.dt

            if step % self.p.save_interval == 0 or step == self.p.steps - 1:
                self._save_history(step * self.p.dt)

            if step % (max(1, self.p.steps // 10)) == 0:
                print(f"  -> Průběh: {int(step / self.p.steps * 100)}% ({step}/{self.p.steps} kroků)")

        print(f"  -> Průběh: 100% ({self.p.steps}/{self.p.steps} kroků)")

        k_size = min(100, max(1, self.p.steps))
        kernel = np.ones(k_size) / k_size
        res = {
            'smooth_induced': [],
            'smooth_collected': [],
            'smooth_total': [],
            'voltage_ant': self.voltage_ant,
            'history': self.history
        }
        for a_idx in range(self.num_antennas):
            res['smooth_induced'].append(
                np.convolve(self.ind_curr_e[a_idx] + self.ind_curr_i[a_idx], kernel, mode='same'))
            res['smooth_collected'].append(
                np.convolve(self.col_curr_e[a_idx] + self.col_curr_i[a_idx], kernel, mode='same'))
            res['smooth_total'].append(np.convolve(self.tot_curr[a_idx], kernel, mode='same'))

        return res
