#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from typing import Dict, Any, Tuple
import sys


try:
    import pyamg
except ImportError:
    print("[CHYBA] Knihovna 'pyamg' nenalezena. Pro 3D simulace s vysokým rozlišením je nezbytná.")
    print("Nainstalujte ji pomocí: pip install pyamg")
    sys.exit(1)

from input_data_3d import SimulationParams3D, SimulationToggles3D
from input_data_3d import e, m_e, eps_0


class DustImpactSimulation3D:
    def __init__(self, params: SimulationParams3D, toggles: SimulationToggles3D,
                 V_bg, Vw_grids, Ex_bg, Ey_bg, Ez_bg, Ewx_list, Ewy_list, Ewz_list, antenna_masks_3d,
                 spacecraft_mask_3d):
        self.p = params
        self.toggles = toggles
        self.num_antennas = len(self.p.vtk_files.antenna_weighting)

        # Uložíme základní pole (očištěné o vliv antén, pokud předpokládáme, že V_bg ze SPIS je jen sonda)
        self.V_bg_base = V_bg
        self.Ex_bg_base, self.Ey_bg_base, self.Ez_bg_base = Ex_bg, Ey_bg, Ez_bg

        # Aktuální celkové pole pozadí (bude se měnit)
        self.Ex_bg, self.Ey_bg, self.Ez_bg = Ex_bg.copy(), Ey_bg.copy(), Ez_bg.copy()

        self.Ewx_list, self.Ewy_list, self.Ewz_list = Ewx_list, Ewy_list, Ewz_list
        self.antenna_masks_3d = antenna_masks_3d
        self.spacecraft_mask_3d = spacecraft_mask_3d

        # 3D Sferická maxwellovská emise částic
        v_th_e = np.sqrt(2 * e * self.p.T_dust_eV / m_e)
        v_th_i = np.sqrt(2 * e * self.p.T_dust_eV / self.p.m_i)

        self.vx_e, self.vy_e, self.vz_e = np.random.normal(0, v_th_e, (3, self.p.N_particles))
        self.vx_i, self.vy_i, self.vz_i = np.random.normal(0, v_th_i, (3, self.p.N_particles))

        self.x_e = np.full(self.p.N_particles, self.p.impact_pos[0])
        self.y_e = np.full(self.p.N_particles, self.p.impact_pos[1])
        self.z_e = np.full(self.p.N_particles, self.p.impact_pos[2])
        self.x_i, self.y_i, self.z_i = self.x_e.copy(), self.y_e.copy(), self.z_e.copy()

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

        # Startujeme s anténami nabitými na V_bias
        self.voltage_ant = np.zeros((self.num_antennas, self.p.steps))
        for a_idx in range(self.num_antennas):
            self.voltage_ant[a_idx, 0] = self.p.V_bias[a_idx]

        self.V_self_grid = np.zeros((self.p.Nx, self.p.Ny, self.p.Nz))
        self.Ex_self = np.zeros((self.p.Nx, self.p.Ny, self.p.Nz))
        self.Ey_self = np.zeros((self.p.Nx, self.p.Ny, self.p.Nz))
        self.Ez_self = np.zeros((self.p.Nx, self.p.Ny, self.p.Nz))

        self.combined_mask = np.zeros((self.p.Nx, self.p.Ny, self.p.Nz), dtype=bool)
        self.combined_mask |= self.spacecraft_mask_3d
        for m in self.antenna_masks_3d:
            self.combined_mask |= m

        self._build_3d_poisson_solver()

        self.history = {'V': [], 't': [],
                        'x_e': [], 'y_e': [], 'z_e': [], 'x_i': [], 'y_i': [], 'z_i': [],
                        'vx_e': [], 'vy_e': [], 'vz_e': [], 'vx_i': [], 'vy_i': [], 'vz_i': []}

    def _get_1d_idx(self, i: int, j: int, k: int) -> int:
        return i * (self.p.Ny * self.p.Nz) + j * self.p.Nz + k

    def _build_3d_poisson_solver(self):
        Nx, Ny, Nz = self.p.Nx, self.p.Ny, self.p.Nz
        N_tot = Nx * Ny * Nz

        print(f"=== Inicializace Poissonova řešiče (3D) ===")
        print(f"  -> Celkový počet uzlů: {N_tot}")

        sc_voxels = np.sum(self.spacecraft_mask_3d)
        print(f"  -> Uzly těla sondy: {sc_voxels}")

        for i, m in enumerate(self.antenna_masks_3d):
            ant_voxels = np.sum(m)
            print(f"  -> Uzly antény {i + 1}: {ant_voxels}")

        dx2, dy2, dz2 = self.p.dx ** 2, self.p.dy ** 2, self.p.dz ** 2
        dirichlet_count = 0

        # Optimalizovaná stavba matice pomocí COO formátu (výrazně šetří RAM oproti LIL)
        row = []
        col = []
        data = []

        for i in range(Nx):
            for j in range(Ny):
                for k in range(Nz):
                    idx = self._get_1d_idx(i, j, k)
                    is_boundary = (i == 0) or (i == Nx - 1) or (j == 0) or (j == Ny - 1) or (k == 0) or (k == Nz - 1)
                    if is_boundary or self.combined_mask[i, j, k]:
                        row.append(idx)
                        col.append(idx)
                        data.append(1.0)
                        dirichlet_count += 1
                    else:
                        row.append(idx)
                        col.append(idx)
                        data.append(-2 / dx2 - 2 / dy2 - 2 / dz2)
                        
                        row.append(idx)
                        col.append(self._get_1d_idx(i - 1, j, k))
                        data.append(1 / dx2)
                        
                        row.append(idx)
                        col.append(self._get_1d_idx(i + 1, j, k))
                        data.append(1 / dx2)
                        
                        row.append(idx)
                        col.append(self._get_1d_idx(i, j - 1, k))
                        data.append(1 / dy2)
                        
                        row.append(idx)
                        col.append(self._get_1d_idx(i, j + 1, k))
                        data.append(1 / dy2)
                        
                        row.append(idx)
                        col.append(self._get_1d_idx(i, j, k - 1))
                        data.append(1 / dz2)
                        
                        row.append(idx)
                        col.append(self._get_1d_idx(i, j, k + 1))
                        data.append(1 / dz2)

        print(f"  -> Dirichletovy uzly celkem (vč. okrajů domény): {dirichlet_count}")
        print(f"  -> Stavím matici CSR...")
        A_csr = sp.coo_matrix((data, (row, col)), shape=(N_tot, N_tot)).tocsr()
        
        print(f"  -> Stavím AMG hierarchii (PyAMG)...")
        self.solver_self = pyamg.ruge_stuben_solver(A_csr)
        print(f"  -> AMG řešič připraven. Počet úrovní: {len(self.solver_self.levels)}")
        print(f"  -> Hotovo.")


    def _update_background_field(self, step: int):
        """ Dynamická superpozice polí pozadí podle aktuálního napětí na anténách. """
        self.Ex_bg = self.Ex_bg_base.copy()
        self.Ey_bg = self.Ey_bg_base.copy()
        self.Ez_bg = self.Ez_bg_base.copy()

        for a_idx in range(self.num_antennas):
            V_curr = self.voltage_ant[a_idx, step]
            # Vw_grid v 3D datech je pole při 1V na anténě.
            # Předpokládáme, že základní pole V_bg_base má antény na 0V.
            self.Ex_bg += V_curr * self.Ewx_list[a_idx]
            self.Ey_bg += V_curr * self.Ewy_list[a_idx]
            self.Ez_bg += V_curr * self.Ewz_list[a_idx]

    def _solve_poisson_equation(self):
        if not self.cloud_injected or not self.toggles.enable_self_field:
            return

        Nx, Ny, Nz = self.p.Nx, self.p.Ny, self.p.Nz
        edges_x = np.append(self.p.x_grid - self.p.dx / 2, self.p.x_grid[-1] + self.p.dx / 2)
        edges_y = np.append(self.p.y_grid - self.p.dy / 2, self.p.y_grid[-1] + self.p.dy / 2)
        edges_z = np.append(self.p.z_grid - self.p.dz / 2, self.p.z_grid[-1] + self.p.dz / 2)

        counts_e, _ = np.histogramdd((self.x_e[self.active_e], self.y_e[self.active_e], self.z_e[self.active_e]),
                                     bins=(edges_x, edges_y, edges_z))
        counts_i, _ = np.histogramdd((self.x_i[self.active_i], self.y_i[self.active_i], self.z_i[self.active_i]),
                                     bins=(edges_x, edges_y, edges_z))

        rho_grid = (counts_i - counts_e) * self.p.q_macro / (self.p.dx * self.p.dy * self.p.dz)

        b_self = np.zeros(Nx * Ny * Nz)
        for i in range(Nx):
            for j in range(Ny):
                for k in range(Nz):
                    idx = self._get_1d_idx(i, j, k)
                    is_boundary = (i == 0) or (i == Nx - 1) or (j == 0) or (j == Ny - 1) or (k == 0) or (k == Nz - 1)
                    if not (is_boundary or self.combined_mask[i, j, k]):
                        b_self[idx] = -rho_grid[i, j, k] / eps_0

        # Teplý start: použijeme potenciál z předchozího kroku jako počáteční odhad
        x0 = self.V_self_grid.ravel()
        V_self_1d = self.solver_self.solve(b_self, x0=x0, tol=1e-5)
        
        self.V_self_grid = V_self_1d.reshape((Nx, Ny, Nz))
        self.Ex_self, self.Ey_self, self.Ez_self = np.gradient(-self.V_self_grid, self.p.dx, self.p.dy, self.p.dz)

    def _interp_3d_fast(self, x, y, z, Field_3D):
        """ Rychlá vlastní vektorizovaná trilineární interpolace (nyní zvládá záporné osy) """
        idx_x = (x - self.p.x_grid[0]) / self.p.dx
        idx_y = (y - self.p.y_grid[0]) / self.p.dy
        idx_z = (z - self.p.z_grid[0]) / self.p.dz

        i = np.clip(np.floor(idx_x).astype(int), 0, self.p.Nx - 2)
        j = np.clip(np.floor(idx_y).astype(int), 0, self.p.Ny - 2)
        k = np.clip(np.floor(idx_z).astype(int), 0, self.p.Nz - 2)

        tx, ty, tz = idx_x - i, idx_y - j, idx_z - k

        c000 = Field_3D[i, j, k]
        c100 = Field_3D[i + 1, j, k]
        c010 = Field_3D[i, j + 1, k]
        c110 = Field_3D[i + 1, j + 1, k]
        c001 = Field_3D[i, j, k + 1]
        c101 = Field_3D[i + 1, j, k + 1]
        c011 = Field_3D[i, j + 1, k + 1]
        c111 = Field_3D[i + 1, j + 1, k + 1]

        c00 = c000 * (1 - tx) + c100 * tx
        c10 = c010 * (1 - tx) + c110 * tx
        c01 = c001 * (1 - tx) + c101 * tx
        c11 = c011 * (1 - tx) + c111 * tx

        c0 = c00 * (1 - ty) + c10 * ty
        c1 = c01 * (1 - ty) + c11 * ty

        return c0 * (1 - tz) + c1 * tz

    def _get_accel(self, x_act, y_act, z_act, mass, phys_charge):
        Ex_tot = self._interp_3d_fast(x_act, y_act, z_act, self.Ex_bg)
        Ey_tot = self._interp_3d_fast(x_act, y_act, z_act, self.Ey_bg)
        Ez_tot = self._interp_3d_fast(x_act, y_act, z_act, self.Ez_bg)

        if self.toggles.enable_self_field:
            Ex_tot += self._interp_3d_fast(x_act, y_act, z_act, self.Ex_self)
            Ey_tot += self._interp_3d_fast(x_act, y_act, z_act, self.Ey_self)
            Ez_tot += self._interp_3d_fast(x_act, y_act, z_act, self.Ez_self)

        return (phys_charge / mass) * Ex_tot, (phys_charge / mass) * Ey_tot, (phys_charge / mass) * Ez_tot

    def _push_species(self, x, y, z, vx, vy, vz, active, was_outside, mass, phys_charge, macro_charge):
        col_currs = np.zeros(self.num_antennas)
        ind_currs = np.zeros(self.num_antennas)
        if not np.any(active): return col_currs, ind_currs

        x_act, y_act, z_act = x[active], y[active], z[active]
        ax, ay, az = self._get_accel(x_act, y_act, z_act, mass, phys_charge)

        vx[active] += ax * self.p.dt
        vy[active] += ay * self.p.dt
        vz[active] += az * self.p.dt

        x[active] += vx[active] * self.p.dt
        y[active] += vy[active] * self.p.dt
        z[active] += vz[active] * self.p.dt

        new_active = active.copy()

        # BEZPEČNOSTNÍ POJISTKA: Vyřazení částic s NaN/Inf souřadnicemi (prevence pádů z numerického přetečení)
        bad_pos = ~np.isfinite(x[new_active]) | ~np.isfinite(y[new_active]) | ~np.isfinite(z[new_active])
        if np.any(bad_pos):
            global_active_idx = np.where(new_active)[0]
            new_active[global_active_idx[bad_pos]] = False
            # Ořízneme dočasná pole, abychom dál pokračovali jen se zdravými částicemi
            valid_mask = ~bad_pos
            x_act, y_act, z_act = x_act[valid_mask], y_act[valid_mask], z_act[valid_mask]

        if not np.any(new_active):
            active[:] = new_active
            return col_currs, ind_currs

        # Vypočítáme diskrétní indexy pro mřížku
        idx_x = np.clip(((x[new_active] - self.p.x_grid[0]) / self.p.dx).astype(int), 0, self.p.Nx - 1)
        idx_y = np.clip(((y[new_active] - self.p.y_grid[0]) / self.p.dy).astype(int), 0, self.p.Ny - 1)
        idx_z = np.clip(((z[new_active] - self.p.z_grid[0]) / self.p.dz).astype(int), 0, self.p.Nz - 1)

        # 1. KOLIZE SE SONDOU (Absorpce na trupu)
        hit_sc = self.spacecraft_mask_3d[idx_x, idx_y, idx_z]
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

            if self.toggles.enable_antenna_collection and np.any(crossed_ant):
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

                v_dot_Ew = vx[new_active] * Ewx + vy[new_active] * Ewy + vz[new_active] * Ewz
                ind_currs[a_idx] = -np.sum(macro_charge * v_dot_Ew)

        # Odstranění částic letících mimo 3D doménu (-L až L)
        new_active[new_active & (x <= -self.p.L_x)] = False
        new_active[new_active & (x >= self.p.L_x)] = False
        new_active[new_active & (y <= -self.p.L_y)] = False
        new_active[new_active & (y >= self.p.L_y)] = False
        new_active[new_active & (z <= -self.p.L_z)] = False
        new_active[new_active & (z >= self.p.L_z)] = False

        active[:] = new_active
        return col_currs, ind_currs

    def _save_history(self, current_time: float):
        self.history['V'].append((self.V_bg_base + self.V_self_grid).copy()) # Pozn: Zde by šlo přidat i vliv antén
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

        progress_step = max(1, self.p.steps // 10)

        for step in range(self.p.steps):
            if step % progress_step == 0:
                percent = (step / self.p.steps) * 100
                print(f"  -> Průběh: {percent:.0f}% ({step}/{self.p.steps} kroků)", flush=True)

            if not self.cloud_injected and step * self.p.dt >= self.p.t_delay:
                self.active_e[:], self.active_i[:] = True, True
                self.was_outside_e[:], self.was_outside_i[:] = False, False
                self.cloud_injected = True

            # AKTUALIZACE POZADÍ (Zpětná vazba)
            if step > 0:
                self._update_background_field(step - 1)

            self._solve_poisson_equation()

            if self.cloud_injected:
                ce, ie = self._push_species(self.x_e, self.y_e, self.z_e, self.vx_e, self.vy_e, self.vz_e,
                                            self.active_e, self.was_outside_e, m_e, -e, -self.p.q_macro)
                ci, ii = self._push_species(self.x_i, self.y_i, self.z_i, self.vx_i, self.vy_i, self.vz_i,
                                            self.active_i, self.was_outside_i, self.p.m_i, e, self.p.q_macro)
                self.col_curr_e[:, step], self.ind_curr_e[:, step] = ce, ie
                self.col_curr_i[:, step], self.ind_curr_i[:, step] = ci, ii

            for a_idx in range(self.num_antennas):
                I_tot = self.ind_curr_e[a_idx, step] + self.ind_curr_i[a_idx, step] + self.col_curr_e[a_idx, step] + \
                        self.col_curr_i[a_idx, step]
                self.tot_curr[a_idx, step] = I_tot
                if step > 0:
                    dV_dt = I_tot / self.p.C_ant[a_idx]
                    if self.toggles.enable_rc_circuit:
                        # Relaxace k rovnovážnému potenciálu V_bias
                        dV_dt -= (self.voltage_ant[a_idx, step - 1] - self.p.V_bias[a_idx]) / (self.p.R_ant[a_idx] * self.p.C_ant[a_idx])
                    self.voltage_ant[a_idx, step] = self.voltage_ant[a_idx, step - 1] + dV_dt * self.p.dt

            if step % self.p.save_interval == 0 or step == self.p.steps - 1:
                self._save_history(step * self.p.dt)

        # Post-processing okénkový filtr (proběhne až po skončení hlavní smyčky)
        kernel = np.ones(100) / 100
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