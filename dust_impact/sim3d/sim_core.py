# -*- coding: utf-8 -*-
"""
3D PIC Simulation Core Solver.
Specific solver logic for 3D geometry, using shared physics and numerics subpackages.
"""

import sys
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from typing import Dict, Any, Tuple

from dust_impact.physics.constants import e, m_e, eps_0
from dust_impact.physics.ramo_shockley import calc_induced_current
from dust_impact.numerics.interpolators import interp_field_3d
from dust_impact.numerics.pushers import leapfrog_step_3d
from dust_impact.numerics.poisson import build_pyamg_solver, HAS_PYAMG
from dust_impact.common.circuits import integrate_rc_circuit
from dust_impact.sim3d.config_loader import SimulationParams3D, SimulationToggles3D


class DustImpactSimulation3D:
    def __init__(self, params: SimulationParams3D, toggles: SimulationToggles3D,
                 V_bg, Vw_grids, Ex_bg, Ey_bg, Ez_bg, Ewx_list, Ewy_list, Ewz_list, antenna_masks_3d,
                 spacecraft_mask_3d):
        self.p = params
        self.toggles = toggles
        self.num_antennas = len(getattr(self.p.vtk_files, 'antenna_weighting_files', getattr(self.p.vtk_files, 'antenna_weighting', [])))

        self.V_bg_base = V_bg
        self.Ex_bg_base, self.Ey_bg_base, self.Ez_bg_base = Ex_bg, Ey_bg, Ez_bg

        self.Ex_bg, self.Ey_bg, self.Ez_bg = Ex_bg.copy(), Ey_bg.copy(), Ez_bg.copy()

        self.Ewx_list, self.Ewy_list, self.Ewz_list = Ewx_list, Ewy_list, Ewz_list
        self.antenna_masks_3d = antenna_masks_3d
        self.spacecraft_mask_3d = spacecraft_mask_3d

        v_th_e = np.sqrt(2 * e * self.p.T_dust_eV / m_e)
        v_th_i = np.sqrt(2 * e * self.p.T_dust_eV / self.p.m_i)

        impact_normal_val = getattr(self.p, 'impact_normal', [-0.7071, 0.7071, 0.0])
        n = np.array(impact_normal_val, dtype=float)
        n_norm = np.linalg.norm(n)
        if n_norm > 0:
            n = n / n_norm
        else:
            n = np.array([-0.7071, 0.7071, 0.0])

        theta = np.arccos(np.clip(n[2], -1.0, 1.0))
        phi = np.arctan2(n[1], n[0])

        Rz = np.array([
            [np.cos(phi), -np.sin(phi), 0],
            [np.sin(phi), np.cos(phi), 0],
            [0, 0, 1]
        ])
        Ry = np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ])
        R = Rz @ Ry

        v_z_prime_i = np.abs(np.random.normal(v_th_i, v_th_i / 2, self.p.N_particles))
        v_x_prime_i = np.random.normal(0, v_th_i / 2, self.p.N_particles)
        v_y_prime_i = np.random.normal(0, v_th_i / 2, self.p.N_particles)
        v_rot_i = R @ np.vstack([v_x_prime_i, v_y_prime_i, v_z_prime_i])
        self.vx_i, self.vy_i, self.vz_i = v_rot_i[0], v_rot_i[1], v_rot_i[2]

        v_z_prime_e = np.abs(np.random.normal(v_th_e, v_th_e / 2, self.p.N_particles))
        v_x_prime_e = np.random.normal(0, v_th_e / 2, self.p.N_particles)
        v_y_prime_e = np.random.normal(0, v_th_e / 2, self.p.N_particles)
        v_rot_e = R @ np.vstack([v_x_prime_e, v_y_prime_e, v_z_prime_e])
        self.vx_e, self.vy_e, self.vz_e = v_rot_e[0], v_rot_e[1], v_rot_e[2]

        impact_point_val = getattr(self.p, 'impact_point', [0.0, 0.0, 0.0])
        r0 = np.array(impact_point_val, dtype=float)

        grid_step = max(self.p.dx, self.p.dy, self.p.dz)
        r_start = r0 + n * (0.5 * grid_step)
        for step_mult in range(1, 30):
            test_r = r0 + n * (step_mult * 0.2 * grid_step)
            ix = np.clip(int((test_r[0] - self.p.x_grid[0]) / self.p.dx), 0, self.p.Nx - 1)
            iy = np.clip(int((test_r[1] - self.p.y_grid[0]) / self.p.dy), 0, self.p.Ny - 1)
            iz = np.clip(int((test_r[2] - self.p.z_grid[0]) / self.p.dz), 0, self.p.Nz - 1)
            if not self.spacecraft_mask_3d[ix, iy, iz]:
                r_start = test_r + n * (0.5 * grid_step)
                break

        self.x_e = np.full(self.p.N_particles, r_start[0])
        self.y_e = np.full(self.p.N_particles, r_start[1])
        self.z_e = np.full(self.p.N_particles, r_start[2])

        self.x_i = np.full(self.p.N_particles, r_start[0])
        self.y_i = np.full(self.p.N_particles, r_start[1])
        self.z_i = np.full(self.p.N_particles, r_start[2])

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
        for a_idx in range(self.num_antennas):
            if getattr(self.toggles, 'enable_antenna_bias', True):
                self.voltage_ant[a_idx, 0] = self.p.V_bias[a_idx]

        self.V_self_grid = np.zeros((self.p.Nx, self.p.Ny, self.p.Nz))
        self.rho_grid = np.zeros((self.p.Nx, self.p.Ny, self.p.Nz))

        self.combined_mask = self.spacecraft_mask_3d.copy()
        for mask in self.antenna_masks_3d:
            self.combined_mask |= mask

        self._build_poisson_solver()

        self.history = {'V': [], 'rho': [], 't': [],
                        'x_e': [], 'y_e': [], 'z_e': [],
                        'x_i': [], 'y_i': [], 'z_i': [],
                        'vx_e': [], 'vy_e': [], 'vz_e': [],
                        'vx_i': [], 'vy_i': [], 'vz_i': []}

    def _get_1d_idx(self, i, j, k):
        return i * (self.p.Ny * self.p.Nz) + j * self.p.Nz + k

    def _build_poisson_solver(self):
        Nx, Ny, Nz = self.p.Nx, self.p.Ny, self.p.Nz
        N_tot = Nx * Ny * Nz

        dx2, dy2, dz2 = self.p.dx ** 2, self.p.dy ** 2, self.p.dz ** 2
        A_self = sp.lil_matrix((N_tot, N_tot))

        print(f"=== Inicializace Poissonova řešiče (3D) ===")
        print(f"  -> Celkový počet uzlů: {N_tot}")

        for i in range(Nx):
            for j in range(Ny):
                for k in range(Nz):
                    idx = self._get_1d_idx(i, j, k)
                    is_boundary = (i == 0) or (i == Nx - 1) or (j == 0) or (j == Ny - 1) or (k == 0) or (k == Nz - 1)

                    if is_boundary or self.combined_mask[i, j, k]:
                        A_self[idx, idx] = 1.0
                    else:
                        diag_val = -2 / dx2 - 2 / dy2 - 2 / dz2
                        A_self[idx, idx] = diag_val
                        A_self[idx, self._get_1d_idx(i - 1, j, k)] = 1 / dx2
                        A_self[idx, self._get_1d_idx(i + 1, j, k)] = 1 / dx2
                        A_self[idx, self._get_1d_idx(i, j - 1, k)] = 1 / dy2
                        A_self[idx, self._get_1d_idx(i, j + 1, k)] = 1 / dy2
                        A_self[idx, self._get_1d_idx(i, j, k - 1)] = 1 / dz2
                        A_self[idx, self._get_1d_idx(i, j, k + 1)] = 1 / dz2

        A_csr = A_self.tocsr()
        if HAS_PYAMG:
            print("  -> Stavím AMG hierarchii (PyAMG)...")
            self.amg_solver = build_pyamg_solver(A_csr)
        else:
            print("  -> PyAMG nenalezen, používám LU solver (SciPy)...")
            self.lu_solver = spla.factorized(A_self.tocsc())

    def _update_background_fields(self, step: int):
        Ex_tot = self.Ex_bg_base.copy()
        Ey_tot = self.Ey_bg_base.copy()
        Ez_tot = self.Ez_bg_base.copy()

        for a_idx in range(self.num_antennas):
            V_curr = self.voltage_ant[a_idx, max(0, step - 1)] if step > 0 else self.voltage_ant[a_idx, 0]
            V_bias = self.p.V_bias[a_idx]
            dV = V_curr - V_bias

            if abs(dV) > 1e-6:
                Ex_tot += dV * self.Ewx_list[a_idx]
                Ey_tot += dV * self.Ewy_list[a_idx]
                Ez_tot += dV * self.Ewz_list[a_idx]

        if not getattr(self.toggles, 'enable_spis_background_field', True):
            Ex_tot.fill(0.0)
            Ey_tot.fill(0.0)
            Ez_tot.fill(0.0)

        self.Ex_bg, self.Ey_bg, self.Ez_bg = Ex_tot, Ey_tot, Ez_tot

    def _solve_poisson_equation(self):
        if not self.cloud_injected or not getattr(self.toggles, 'enable_plasma_self_field', True):
            return

        Nx, Ny, Nz = self.p.Nx, self.p.Ny, self.p.Nz
        edges_x = np.linspace(-self.p.L_x, self.p.L_x, Nx + 1)
        edges_y = np.linspace(-self.p.L_y, self.p.L_y, Ny + 1)
        edges_z = np.linspace(-self.p.L_z, self.p.L_z, Nz + 1)

        counts_e, _ = np.histogramdd(
            (self.x_e[self.active_e], self.y_e[self.active_e], self.z_e[self.active_e]),
            bins=(edges_x, edges_y, edges_z)
        )
        counts_i, _ = np.histogramdd(
            (self.x_i[self.active_i], self.y_i[self.active_i], self.z_i[self.active_i]),
            bins=(edges_x, edges_y, edges_z)
        )

        dV = self.p.dx * self.p.dy * self.p.dz
        rho_grid = (counts_i - counts_e) * self.p.q_macro / dV
        self.rho_grid = rho_grid

        b_self = np.zeros(Nx * Ny * Nz)
        for i in range(Nx):
            for j in range(Ny):
                for k in range(Nz):
                    idx = self._get_1d_idx(i, j, k)
                    is_boundary = (i == 0) or (i == Nx - 1) or (j == 0) or (j == Ny - 1) or (k == 0) or (k == Nz - 1)
                    if not (is_boundary or self.combined_mask[i, j, k]):
                        b_self[idx] = -rho_grid[i, j, k] / eps_0

        if HAS_PYAMG:
            x0 = self.V_self_grid.ravel()
            V_self_1d = self.amg_solver.solve(b_self, x0=x0, tol=1e-5)
        else:
            V_self_1d = self.lu_solver(b_self)

        self.V_self_grid = V_self_1d.reshape((Nx, Ny, Nz))
        self.Ex_self, self.Ey_self, self.Ez_self = np.gradient(-self.V_self_grid, self.p.dx, self.p.dy, self.p.dz)

    def _interp_3d_fast(self, x, y, z, Field_3D):
        return interp_field_3d(x, y, z, self.p.x_grid, self.p.y_grid, self.p.z_grid,
                               self.p.dx, self.p.dy, self.p.dz, Field_3D)

    def _get_accel(self, x_act, y_act, z_act, mass, phys_charge):
        Ex_tot = self._interp_3d_fast(x_act, y_act, z_act, self.Ex_bg)
        Ey_tot = self._interp_3d_fast(x_act, y_act, z_act, self.Ey_bg)
        Ez_tot = self._interp_3d_fast(x_act, y_act, z_act, self.Ez_bg)

        if getattr(self.toggles, 'enable_plasma_self_field', True):
            Ex_tot += self._interp_3d_fast(x_act, y_act, z_act, self.Ex_self)
            Ey_tot += self._interp_3d_fast(x_act, y_act, z_act, self.Ey_self)
            Ez_tot += self._interp_3d_fast(x_act, y_act, z_act, self.Ez_self)

        return (phys_charge / mass) * Ex_tot, (phys_charge / mass) * Ey_tot, (phys_charge / mass) * Ez_tot

    def _push_species(self, x, y, z, vx, vy, vz, active, was_outside, mass, phys_charge, macro_charge):
        col_currs = np.zeros(self.num_antennas)
        ind_currs = np.zeros(self.num_antennas)
        if not np.any(active): return col_currs, ind_currs

        x_act, y_act, z_act = x[active].copy(), y[active].copy(), z[active].copy()
        ax, ay, az = self._get_accel(x_act, y_act, z_act, mass, phys_charge)

        # Symplectic 3D Leapfrog pusher via numerics.pushers
        x[active], y[active], z[active], vx[active], vy[active], vz[active] = leapfrog_step_3d(
            x[active], y[active], z[active], vx[active], vy[active], vz[active],
            ax, ay, az, 1.0, self.p.dt, np.ones(np.sum(active), dtype=bool)
        )

        new_active = active.copy()

        bad_pos = ~np.isfinite(x[new_active]) | ~np.isfinite(y[new_active]) | ~np.isfinite(z[new_active])
        if np.any(bad_pos):
            global_active_idx = np.where(new_active)[0]
            new_active[global_active_idx[bad_pos]] = False
            valid_mask = ~bad_pos
            x_act, y_act, z_act = x_act[valid_mask], y_act[valid_mask], z_act[valid_mask]

        if not np.any(new_active):
            active[:] = new_active
            return col_currs, ind_currs

        # Continuous Collision Detection (CCD) via trajectory segment midpoint sampling
        x_new, y_new, z_new = x[new_active], y[new_active], z[new_active]
        x_mid = 0.5 * (x_act + x_new)
        y_mid = 0.5 * (y_act + y_new)
        z_mid = 0.5 * (z_act + z_new)

        idx_x_new = np.clip(((x_new - self.p.x_grid[0]) / self.p.dx).astype(int), 0, self.p.Nx - 1)
        idx_y_new = np.clip(((y_new - self.p.y_grid[0]) / self.p.dy).astype(int), 0, self.p.Ny - 1)
        idx_z_new = np.clip(((z_new - self.p.z_grid[0]) / self.p.dz).astype(int), 0, self.p.Nz - 1)

        idx_x_mid = np.clip(((x_mid - self.p.x_grid[0]) / self.p.dx).astype(int), 0, self.p.Nx - 1)
        idx_y_mid = np.clip(((y_mid - self.p.y_grid[0]) / self.p.dy).astype(int), 0, self.p.Ny - 1)
        idx_z_mid = np.clip(((z_mid - self.p.z_grid[0]) / self.p.dz).astype(int), 0, self.p.Nz - 1)

        hit_sc = self.spacecraft_mask_3d[idx_x_new, idx_y_new, idx_z_new] | self.spacecraft_mask_3d[idx_x_mid, idx_y_mid, idx_z_mid]
        global_active_indices = np.where(new_active)[0]
        destroyed_by_sc = global_active_indices[hit_sc]
        new_active[destroyed_by_sc] = False

        for a_idx in range(self.num_antennas):
            idx_x_ant = np.clip(((x[new_active] - self.p.x_grid[0]) / self.p.dx).astype(int), 0, self.p.Nx - 1)
            idx_y_ant = np.clip(((y[new_active] - self.p.y_grid[0]) / self.p.dy).astype(int), 0, self.p.Ny - 1)
            idx_z_ant = np.clip(((z[new_active] - self.p.z_grid[0]) / self.p.dz).astype(int), 0, self.p.Nz - 1)

            is_inside = self.antenna_masks_3d[a_idx][idx_x_ant, idx_y_ant, idx_z_ant]
            is_outside = ~is_inside

            was_out_act = was_outside[a_idx, new_active]
            crossed_ant = is_inside & was_out_act
            was_outside[a_idx, new_active] = is_outside

            if getattr(self.toggles, 'enable_antenna_particle_collection', True) and np.any(crossed_ant):
                n_crossed = np.sum(crossed_ant)
                eff = self.p.collection_eff[a_idx]
                absorbed_mask = np.random.rand(n_crossed) < eff

                global_active_idx = np.where(new_active)[0]
                absorbed_global_indices = global_active_idx[crossed_ant][absorbed_mask]

                col_currs[a_idx] = len(absorbed_global_indices) * macro_charge / self.p.dt
                new_active[absorbed_global_indices] = False

            if np.any(new_active):
                Ewx = self._interp_3d_fast(x[new_active], y[new_active], z[new_active], self.Ewx_list[a_idx])
                Ewy = self._interp_3d_fast(x[new_active], y[new_active], z[new_active], self.Ewy_list[a_idx])
                Ewz = self._interp_3d_fast(x[new_active], y[new_active], z[new_active], self.Ewz_list[a_idx])

                ind_currs[a_idx] = calc_induced_current(macro_charge, vx[new_active], vy[new_active], Ewx, Ewy,
                                                       vz=vz[new_active], Ewz=Ewz)

        new_active[new_active & (x <= -self.p.L_x)] = False
        new_active[new_active & (x >= self.p.L_x)] = False
        new_active[new_active & (y <= -self.p.L_y)] = False
        new_active[new_active & (y >= self.p.L_y)] = False
        new_active[new_active & (z <= -self.p.L_z)] = False
        new_active[new_active & (z >= self.p.L_z)] = False

        active[:] = new_active
        return col_currs, ind_currs

    def _save_history(self, current_time: float):
        self.history['V'].append((self.V_bg_base + self.V_self_grid).copy())
        self.history['rho'].append(self.rho_grid.copy())
        self.history['t'].append(current_time)

        self.history['x_e'].append(np.where(self.active_e, self.x_e, np.nan)[::self.p.plot_stride])
        self.history['y_e'].append(np.where(self.active_e, self.y_e, np.nan)[::self.p.plot_stride])
        self.history['z_e'].append(np.where(self.active_e, self.z_e, np.nan)[::self.p.plot_stride])

        self.history['x_i'].append(np.where(self.active_i, self.x_i, np.nan)[::self.p.plot_stride])
        self.history['y_i'].append(np.where(self.active_i, self.y_i, np.nan)[::self.p.plot_stride])
        self.history['z_i'].append(np.where(self.active_i, self.z_i, np.nan)[::self.p.plot_stride])

        self.history['vx_e'].append(np.where(self.active_e, self.vx_e, np.nan)[::self.p.plot_stride])
        self.history['vy_e'].append(np.where(self.active_e, self.vy_e, np.nan)[::self.p.plot_stride])
        self.history['vz_e'].append(np.where(self.active_e, self.vz_e, np.nan)[::self.p.plot_stride])

        self.history['vx_i'].append(np.where(self.active_i, self.vx_i, np.nan)[::self.p.plot_stride])
        self.history['vy_i'].append(np.where(self.active_i, self.vy_i, np.nan)[::self.p.plot_stride])
        self.history['vz_i'].append(np.where(self.active_i, self.vz_i, np.nan)[::self.p.plot_stride])

    def run(self) -> Dict[str, Any]:
        print(f"Spouštím 3D Simulaci... Mřížka: {self.p.Nx}x{self.p.Ny}x{self.p.Nz} | Počet antén: {self.num_antennas}")

        for step in range(self.p.steps):
            if not self.cloud_injected and step * self.p.dt >= self.p.t_delay:
                self.active_e[:] = True
                self.active_i[:] = True
                self.was_outside_e[:] = False
                self.was_outside_i[:] = False
                self.cloud_injected = True

            if step > 0:
                self._update_background_fields(step)

            self._solve_poisson_equation()

            if self.cloud_injected:
                ce, ie = self._push_species(self.x_e, self.y_e, self.z_e, self.vx_e, self.vy_e, self.vz_e,
                                            self.active_e, self.was_outside_e, m_e, -e, -self.p.q_macro)
                ci, ii = self._push_species(self.x_i, self.y_i, self.z_i, self.vx_i, self.vy_i, self.vz_i,
                                            self.active_i, self.was_outside_i, self.p.m_i, e, self.p.q_macro)

                self.col_curr_e[:, step] = ce
                self.col_curr_i[:, step] = ci
                self.ind_curr_e[:, step] = ie
                self.ind_curr_i[:, step] = ii

            for a_idx in range(self.num_antennas):
                I_tot = self.ind_curr_e[a_idx, step] + self.ind_curr_i[a_idx, step] + self.col_curr_e[a_idx, step] + \
                        self.col_curr_i[a_idx, step]
                self.tot_curr[a_idx, step] = I_tot

                if step > 0:
                    dV_dt = I_tot / self.p.C_ant[a_idx]
                    if getattr(self.toggles, 'enable_rc_circuit_response', True):
                        v_bias = self.p.V_bias[a_idx] if getattr(self.toggles, 'enable_antenna_bias_voltage', True) else 0.0
                        dV_dt -= (self.voltage_ant[a_idx, step - 1] - v_bias) / (
                                    self.p.R_ant[a_idx] * self.p.C_ant[a_idx])

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
