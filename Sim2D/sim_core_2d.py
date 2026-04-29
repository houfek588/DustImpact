#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MODUL B: Výpočetní jádro 2D (Particle-in-Cell)
Využívá scipy.sparse pro masivně paralelní řešení Poissonovy rovnice ve 2D.
Nyní plně podporuje dynamické a volitelné množství detekčních antén.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from typing import Tuple, Dict, Any

from input_data_2d import calc_Ew_2d, SimulationParams2D, SimulationToggles2D
from input_data_2d import e, m_e, eps_0


class DustImpactSimulation2D:
    def __init__(self, params: SimulationParams2D, toggles: SimulationToggles2D):
        self.p = params
        self.toggles = toggles
        self.num_antennas = len(self.p.antennas)

        # ---------------------------------------------------------------------
        # INICIALIZACE ČÁSTIC
        # ---------------------------------------------------------------------
        self.vx_e = np.abs(np.random.normal(0, self.p.v_th_e, self.p.N_particles))
        self.vy_e = np.random.normal(0, self.p.v_th_e, self.p.N_particles)

        self.vx_i = np.abs(np.random.normal(0, self.p.v_th_i, self.p.N_particles))
        self.vy_i = np.random.normal(0, self.p.v_th_i, self.p.N_particles)

        self.x_e = np.zeros(self.p.N_particles)
        self.y_e = np.zeros(self.p.N_particles)
        self.x_i = np.zeros(self.p.N_particles)
        self.y_i = np.zeros(self.p.N_particles)

        self.active_e = np.zeros(self.p.N_particles, dtype=bool)
        self.active_i = np.zeros(self.p.N_particles, dtype=bool)

        # Sledování dopadů pro KAZDOU anténu zvlášť (matice: num_antennas x N_particles)
        self.was_outside_e = np.zeros((self.num_antennas, self.p.N_particles), dtype=bool)
        self.was_outside_i = np.zeros((self.num_antennas, self.p.N_particles), dtype=bool)

        self.cloud_injected = False

        # Výsledková pole jsou nyní 2D matice (num_antennas x steps)
        self.ind_curr_e = np.zeros((self.num_antennas, self.p.steps))
        self.ind_curr_i = np.zeros((self.num_antennas, self.p.steps))
        self.col_curr_e = np.zeros((self.num_antennas, self.p.steps))
        self.col_curr_i = np.zeros((self.num_antennas, self.p.steps))
        self.tot_curr = np.zeros((self.num_antennas, self.p.steps))
        self.voltage_ant = np.zeros((self.num_antennas, self.p.steps))

        # Mřížková pole (Nx x Ny matice)
        self.rho_grid = np.zeros((self.p.Nx, self.p.Ny))
        self.V_self_grid = np.zeros((self.p.Nx, self.p.Ny))
        self.Ex_self = np.zeros((self.p.Nx, self.p.Ny))
        self.Ey_self = np.zeros((self.p.Nx, self.p.Ny))

        # Sestavení masek všech antén na mřížce
        self.ant_masks = []
        self.combined_ant_mask = np.zeros((self.p.Nx, self.p.Ny), dtype=bool)
        for ant in self.p.antennas:
            r_sq = (self.p.X_mat - ant['x']) ** 2 + (self.p.Y_mat - ant['y']) ** 2
            mask = r_sq <= ant['r'] ** 2
            self.ant_masks.append(mask)
            self.combined_ant_mask |= mask

        # Sestavení řídkých matic pro řešič a předvýpočet pole pozadí
        self._build_poisson_solver()

        self.history = {
            'V': [], 'rho': [], 't': [],
            'x_e': [], 'y_e': [], 'x_i': [], 'y_i': [],
            'vx_e': [], 'vx_i': [], 'vy_e': [], 'vy_i': []
        }

    def _get_1d_idx(self, i: int, j: int) -> int:
        return i * self.p.Ny + j

    def _build_poisson_solver(self):
        Nx, Ny = self.p.Nx, self.p.Ny
        N_tot = Nx * Ny

        A_self = sp.lil_matrix((N_tot, N_tot))
        A_bg = sp.lil_matrix((N_tot, N_tot))

        dx2, dy2 = self.p.dx ** 2, self.p.dy ** 2

        for i in range(Nx):
            for j in range(Ny):
                k = self._get_1d_idx(i, j)

                is_boundary = (i == 0) or (i == Nx - 1) or (j == 0) or (j == Ny - 1)
                is_antenna = self.combined_ant_mask[i, j]

                if is_boundary or is_antenna:
                    A_self[k, k] = 1.0
                    A_bg[k, k] = 1.0
                else:
                    A_self[k, k] = -2 / dx2 - 2 / dy2
                    A_self[k, self._get_1d_idx(i - 1, j)] = 1 / dx2
                    A_self[k, self._get_1d_idx(i + 1, j)] = 1 / dx2
                    A_self[k, self._get_1d_idx(i, j - 1)] = 1 / dy2
                    A_self[k, self._get_1d_idx(i, j + 1)] = 1 / dy2

                    A_bg[k, k] = (-2 / dx2 - 2 / dy2) - (1.0 / self.p.debye_length ** 2)
                    A_bg[k, self._get_1d_idx(i - 1, j)] = 1 / dx2
                    A_bg[k, self._get_1d_idx(i + 1, j)] = 1 / dx2
                    A_bg[k, self._get_1d_idx(i, j - 1)] = 1 / dy2
                    A_bg[k, self._get_1d_idx(i, j + 1)] = 1 / dy2

        self.A_self_csc = A_self.tocsc()
        self.A_bg_csc = A_bg.tocsc()

        self.solver_self = spla.factorized(self.A_self_csc)
        self.solver_bg = spla.factorized(self.A_bg_csc)

        # ---------------------------------------------------------
        # PŘEDVÝPOČET STATICKÉHO POZADÍ
        # ---------------------------------------------------------
        self.V_bg_grid = np.zeros((Nx, Ny))
        self.Ex_bg = np.zeros((Nx, Ny))
        self.Ey_bg = np.zeros((Nx, Ny))

        if self.toggles.enable_background_field:
            b_bg = np.zeros(N_tot)
            for i in range(Nx):
                for j in range(Ny):
                    k = self._get_1d_idx(i, j)
                    y_val = self.p.y_grid[j]

                    if i == 0 and -1.0 <= y_val <= 1.0:
                        b_bg[k] = self.p.Vf
                    elif self.toggles.enable_antenna_bias and self.combined_ant_mask[i, j]:
                        # Zjištění, která z antén pokrývá tento bod
                        for a_idx, mask in enumerate(self.ant_masks):
                            if mask[i, j]:
                                b_bg[k] = self.p.antennas[a_idx]['V_bias']
                                break
                    elif (i == 0) or (i == Nx - 1) or (j == 0) or (j == Ny - 1):
                        b_bg[k] = 0.0

            V_bg_1d = self.solver_bg(b_bg)
            self.V_bg_grid = V_bg_1d.reshape((Nx, Ny))
            self.Ex_bg, self.Ey_bg = np.gradient(-self.V_bg_grid, self.p.dx, self.p.dy)

    def _solve_poisson_equation(self):
        if not self.cloud_injected or not self.toggles.enable_self_field:
            return

        Nx, Ny = self.p.Nx, self.p.Ny

        edges_x = np.append(self.p.x_grid - self.p.dx / 2, self.p.x_grid[-1] + self.p.dx / 2)
        edges_y = np.append(self.p.y_grid - self.p.dy / 2, self.p.y_grid[-1] + self.p.dy / 2)

        counts_e, _, _ = np.histogram2d(self.x_e[self.active_e], self.y_e[self.active_e], bins=(edges_x, edges_y))
        counts_i, _, _ = np.histogram2d(self.x_i[self.active_i], self.y_i[self.active_i], bins=(edges_x, edges_y))

        self.rho_grid = (counts_i - counts_e) * (self.p.q_macro / self.p.A_sim) / (self.p.dx * self.p.dy)

        b_self = np.zeros(Nx * Ny)
        for i in range(Nx):
            for j in range(Ny):
                k = self._get_1d_idx(i, j)
                is_boundary = (i == 0) or (i == Nx - 1) or (j == 0) or (j == Ny - 1)
                if is_boundary or self.combined_ant_mask[i, j]:
                    b_self[k] = 0.0
                else:
                    b_self[k] = -self.rho_grid[i, j] / eps_0

        V_self_1d = self.solver_self(b_self)
        self.V_self_grid = V_self_1d.reshape((Nx, Ny))
        self.Ex_self, self.Ey_self = np.gradient(-self.V_self_grid, self.p.dx, self.p.dy)

    def _interp_field(self, x: np.ndarray, y: np.ndarray, Field_matrix: np.ndarray) -> np.ndarray:
        idx_x = (x - self.p.x_grid[0]) / self.p.dx
        idx_y = (y - self.p.y_grid[0]) / self.p.dy

        i = np.clip(np.floor(idx_x).astype(int), 0, self.p.Nx - 2)
        j = np.clip(np.floor(idx_y).astype(int), 0, self.p.Ny - 2)

        tx = idx_x - i
        ty = idx_y - j

        F00 = Field_matrix[i, j]
        F10 = Field_matrix[i + 1, j]
        F01 = Field_matrix[i, j + 1]
        F11 = Field_matrix[i + 1, j + 1]

        return (1 - tx) * (1 - ty) * F00 + tx * (1 - ty) * F10 + (1 - tx) * ty * F01 + tx * ty * F11

    def _get_accel(self, x_act: np.ndarray, y_act: np.ndarray, mass: float, phys_charge: float) -> Tuple[
        np.ndarray, np.ndarray]:
        Ex_tot = np.zeros(len(x_act))
        Ey_tot = np.zeros(len(y_act))

        if self.toggles.enable_background_field:
            Ex_tot += self._interp_field(x_act, y_act, self.Ex_bg)
            Ey_tot += self._interp_field(x_act, y_act, self.Ey_bg)

        if self.toggles.enable_self_field:
            Ex_tot += self._interp_field(x_act, y_act, self.Ex_self)
            Ey_tot += self._interp_field(x_act, y_act, self.Ey_self)

        return (phys_charge / mass) * Ex_tot, (phys_charge / mass) * Ey_tot

    def _integrate_motion(self, x: np.ndarray, y: np.ndarray, vx: np.ndarray, vy: np.ndarray, active: np.ndarray,
                          mass: float, phys_charge: float, dt: float):

        x_act, y_act = x[active], y[active]
        vx_act, vy_act = vx[active], vy[active]

        if self.p.integrator.lower() in ['euler', 'leapfrog']:
            ax, ay = self._get_accel(x_act, y_act, mass, phys_charge)
            vx_act += ax * dt
            vy_act += ay * dt
            x_act += vx_act * dt
            y_act += vy_act * dt

        x[active], y[active] = x_act, y_act
        vx[active], vy[active] = vx_act, vy_act

    def _push_species(self, x: np.ndarray, y: np.ndarray, vx: np.ndarray, vy: np.ndarray,
                      active: np.ndarray, was_outside: np.ndarray, mass: float, phys_charge: float,
                      macro_charge: float) -> Tuple[np.ndarray, np.ndarray]:

        col_currs = np.zeros(self.num_antennas)
        ind_currs = np.zeros(self.num_antennas)

        if not np.any(active):
            return col_currs, ind_currs

        self._integrate_motion(x, y, vx, vy, active, mass, phys_charge, self.p.dt)

        # Pracujeme s kopií masky active, do které budeme zapisovat smazané částice
        new_active = active.copy()

        for a_idx, ant in enumerate(self.p.antennas):
            x_act = x[new_active]
            y_act = y[new_active]

            r_sq = (x_act - ant['x']) ** 2 + (y_act - ant['y']) ** 2
            is_outside = r_sq > ant['r'] ** 2

            was_out_act = was_outside[a_idx, new_active]
            crossed_ant = ~is_outside & was_out_act
            was_outside[a_idx, new_active] = is_outside

            if self.toggles.enable_antenna_collection and np.any(crossed_ant):
                n_crossed = np.sum(crossed_ant)
                absorbed_mask = np.random.rand(n_crossed) < ant['collection_eff']

                global_active_idx = np.where(new_active)[0]
                global_crossed_idx = global_active_idx[crossed_ant]
                absorbed_global_indices = global_crossed_idx[absorbed_mask]

                dQ_coll = len(absorbed_global_indices) * macro_charge
                col_currs[a_idx] = dQ_coll / self.p.dt
                new_active[absorbed_global_indices] = False

            if np.any(new_active):
                Ewx, Ewy = calc_Ew_2d(x[new_active], y[new_active], ant['x'], ant['y'], ant['w_width'])
                v_dot_Ew = vx[new_active] * Ewx + vy[new_active] * Ewy
                ind_currs[a_idx] = -np.sum(macro_charge * v_dot_Ew)

        new_active[new_active & (x <= 0)] = False
        new_active[new_active & (x >= self.p.L_domain)] = False
        new_active[new_active & (y <= -self.p.H_domain)] = False
        new_active[new_active & (y >= self.p.H_domain)] = False

        active[:] = new_active
        return col_currs, ind_currs

    def _update_circuit(self, step: int):
        for a_idx, ant in enumerate(self.p.antennas):
            I_tot = (self.ind_curr_e[a_idx, step] + self.ind_curr_i[a_idx, step] +
                     self.col_curr_e[a_idx, step] + self.col_curr_i[a_idx, step])
            self.tot_curr[a_idx, step] = I_tot

            if step > 0:
                if self.toggles.enable_rc_circuit:
                    dV_dt = (I_tot / ant['C']) - (self.voltage_ant[a_idx, step - 1] / (ant['R'] * ant['C']))
                else:
                    dV_dt = I_tot / ant['C']
                self.voltage_ant[a_idx, step] = self.voltage_ant[a_idx, step - 1] + dV_dt * self.p.dt

    def _save_history(self, current_time: float):
        self.history['V'].append((self.V_bg_grid + self.V_self_grid).copy())
        self.history['rho'].append(self.rho_grid.copy())
        self.history['t'].append(current_time)

        hx_e = np.where(self.active_e, self.x_e, np.nan)[::self.p.plot_stride]
        hy_e = np.where(self.active_e, self.y_e, np.nan)[::self.p.plot_stride]
        hx_i = np.where(self.active_i, self.x_i, np.nan)[::self.p.plot_stride]
        hy_i = np.where(self.active_i, self.y_i, np.nan)[::self.p.plot_stride]

        self.history['x_e'].append(hx_e)
        self.history['y_e'].append(hy_e)
        self.history['x_i'].append(hx_i)
        self.history['y_i'].append(hy_i)

        hvx_e = np.where(self.active_e, self.vx_e, np.nan)[::self.p.plot_stride]
        hvy_e = np.where(self.active_e, self.vy_e, np.nan)[::self.p.plot_stride]
        hvx_i = np.where(self.active_i, self.vx_i, np.nan)[::self.p.plot_stride]
        hvy_i = np.where(self.active_i, self.vy_i, np.nan)[::self.p.plot_stride]

        self.history['vx_e'].append(hvx_e)
        self.history['vy_e'].append(hvy_e)
        self.history['vx_i'].append(hvx_i)
        self.history['vy_i'].append(hvy_i)

    def _inject_cloud(self):
        self.active_e[:] = True
        self.active_i[:] = True
        self.was_outside_e[:] = False
        self.was_outside_i[:] = False
        self.cloud_injected = True

        if self.p.integrator.lower() == 'leapfrog':
            self._solve_poisson_equation()
            ax_e, ay_e = self._get_accel(self.x_e[self.active_e], self.y_e[self.active_e], m_e, -e)
            ax_i, ay_i = self._get_accel(self.x_i[self.active_i], self.y_i[self.active_i], self.p.m_i, e)

            self.vx_e[self.active_e] -= 0.5 * ax_e * self.p.dt
            self.vy_e[self.active_e] -= 0.5 * ay_e * self.p.dt
            self.vx_i[self.active_i] -= 0.5 * ax_i * self.p.dt
            self.vy_i[self.active_i] -= 0.5 * ay_i * self.p.dt

    def run(self) -> Dict[str, Any]:
        print(f"Spouštím 2D Simulaci... Mřížka: {self.p.Nx}x{self.p.Ny} | Antén: {self.num_antennas}")

        for step in range(self.p.steps):
            current_time = step * self.p.dt

            if not self.cloud_injected and current_time >= self.p.t_delay:
                self._inject_cloud()

            self._solve_poisson_equation()

            if self.cloud_injected:
                c_e, i_e = self._push_species(
                    self.x_e, self.y_e, self.vx_e, self.vy_e, self.active_e, self.was_outside_e,
                    m_e, -e, -self.p.q_macro)
                self.col_curr_e[:, step] = c_e
                self.ind_curr_e[:, step] = i_e

                c_i, i_i = self._push_species(
                    self.x_i, self.y_i, self.vx_i, self.vy_i, self.active_i, self.was_outside_i,
                    self.p.m_i, e, self.p.q_macro)
                self.col_curr_i[:, step] = c_i
                self.ind_curr_i[:, step] = i_i

            self._update_circuit(step)

            if step % self.p.save_interval == 0 or step == self.p.steps - 1:
                self._save_history(current_time)

        return self._post_process()

    def _post_process(self) -> Dict[str, Any]:
        smooth_window = 100
        kernel = np.ones(smooth_window) / smooth_window

        results = {
            'smooth_induced': [],
            'smooth_collected': [],
            'smooth_total': [],
            'voltage_ant': self.voltage_ant,
            'history': self.history
        }

        for a_idx in range(self.num_antennas):
            s_ind = np.convolve(self.ind_curr_e[a_idx] + self.ind_curr_i[a_idx], kernel, mode='same')
            s_col = np.convolve(self.col_curr_e[a_idx] + self.col_curr_i[a_idx], kernel, mode='same')
            s_tot = np.convolve(self.tot_curr[a_idx], kernel, mode='same')
            results['smooth_induced'].append(s_ind)
            results['smooth_collected'].append(s_col)
            results['smooth_total'].append(s_tot)

        return results