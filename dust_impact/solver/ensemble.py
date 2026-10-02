# -*- coding: utf-8 -*-
"""
Particle Ensemble manager for 3D PIC Simulation.
Handles electron and ion state arrays, injection geometries, phase space tracking,
and continuous collision detection (CCD) against spacecraft and antenna structures.
"""

from typing import Tuple, List, Optional, Any
import numpy as np

from dust_impact.physics.constants import e, m_e
from dust_impact.physics.ramo_shockley import calc_induced_current
from dust_impact.numerics.pushers import leapfrog_step_3d
from dust_impact.solver.config_loader import SimulationParams3D
from dust_impact.geometry.surface import check_wire_collision_3d


class ParticleEnsemble:
    """
    Manages electron and ion macro-particles in 3D Cartesian coordinates.
    """

    def __init__(
        self,
        params: SimulationParams3D,
        spacecraft_mask_3d: np.ndarray,
        num_antennas: int,
        antenna_geometries: Optional[List[Any]] = None,
    ):
        self.p = params
        self.spacecraft_mask_3d = spacecraft_mask_3d
        self.num_antennas = num_antennas
        self.antenna_geometries = antenna_geometries

        v_th_e = np.sqrt(2 * e * self.p.T_dust_eV / m_e)
        v_th_i = np.sqrt(2 * e * self.p.T_dust_eV / self.p.m_i)

        mode = getattr(self.p, 'plasma_injection_mode', 'point_cloud')

        if mode == 'homogeneous':
            # 1. Homogeneous uniform spatial distribution across domain [-Lx, Lx], [-Ly, Ly], [-Lz, Lz]
            def _sample_outside_spacecraft(n_pts):
                pts = np.random.uniform(
                    low=[-self.p.L_x, -self.p.L_y, -self.p.L_z],
                    high=[self.p.L_x, self.p.L_y, self.p.L_z],
                    size=(n_pts, 3)
                )
                for idx in range(n_pts):
                    ix = np.clip(int((pts[idx, 0] - self.p.x_grid[0]) / self.p.dx), 0, self.p.Nx - 1)
                    iy = np.clip(int((pts[idx, 1] - self.p.y_grid[0]) / self.p.dy), 0, self.p.Ny - 1)
                    iz = np.clip(int((pts[idx, 2] - self.p.z_grid[0]) / self.p.dz), 0, self.p.Nz - 1)
                    while self.spacecraft_mask_3d[ix, iy, iz]:
                        pts[idx] = np.random.uniform(
                            low=[-self.p.L_x, -self.p.L_y, -self.p.L_z],
                            high=[self.p.L_x, self.p.L_y, self.p.L_z]
                        )
                        ix = np.clip(int((pts[idx, 0] - self.p.x_grid[0]) / self.p.dx), 0, self.p.Nx - 1)
                        iy = np.clip(int((pts[idx, 1] - self.p.y_grid[0]) / self.p.dy), 0, self.p.Ny - 1)
                        iz = np.clip(int((pts[idx, 2] - self.p.z_grid[0]) / self.p.dz), 0, self.p.Nz - 1)
                return pts

            pts_e = _sample_outside_spacecraft(self.p.N_particles)
            pts_i = pts_e.copy()

            self.x_e, self.y_e, self.z_e = pts_e[:, 0], pts_e[:, 1], pts_e[:, 2]
            self.x_i, self.y_i, self.z_i = pts_i[:, 0], pts_i[:, 1], pts_i[:, 2]

            sigma_v_e = v_th_e / np.sqrt(3.0)
            sigma_v_i = v_th_i / np.sqrt(3.0)

            self.vx_e = np.random.normal(0, sigma_v_e, self.p.N_particles)
            self.vy_e = np.random.normal(0, sigma_v_e, self.p.N_particles)
            self.vz_e = np.random.normal(0, sigma_v_e, self.p.N_particles)

            self.vx_i = np.random.normal(0, sigma_v_i, self.p.N_particles)
            self.vy_i = np.random.normal(0, sigma_v_i, self.p.N_particles)
            self.vz_i = np.random.normal(0, sigma_v_i, self.p.N_particles)

        else:
            # 2. Concentrated point cloud (dust impact expansion) mode
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

            impact_point_val = getattr(self.p, 'impact_pos', getattr(self.p, 'impact_location_xyz_m', [0.0, 0.0, 0.0]))
            r0 = np.array(impact_point_val, dtype=float)

            grid_step = max(self.p.dx, self.p.dy, self.p.dz)
            r_start = r0 + n * (0.2 * grid_step)
            for step_mult in range(1, 50):
                ix = np.clip(int((r_start[0] - self.p.x_grid[0]) / self.p.dx), 0, self.p.Nx - 1)
                iy = np.clip(int((r_start[1] - self.p.y_grid[0]) / self.p.dy), 0, self.p.Ny - 1)
                iz = np.clip(int((r_start[2] - self.p.z_grid[0]) / self.p.dz), 0, self.p.Nz - 1)
                if not self.spacecraft_mask_3d[ix, iy, iz]:
                    break
                r_start = r_start + n * (0.2 * grid_step)

            r_offset = np.random.normal(0, 0.01 * grid_step, (self.p.N_particles, 3))
            self.x_e = r_start[0] + r_offset[:, 0]
            self.y_e = r_start[1] + r_offset[:, 1]
            self.z_e = r_start[2] + r_offset[:, 2]

            self.x_i = r_start[0] + r_offset[:, 0]
            self.y_i = r_start[1] + r_offset[:, 1]
            self.z_i = r_start[2] + r_offset[:, 2]

        self.active_e = np.zeros(self.p.N_particles, dtype=bool)
        self.active_i = np.zeros(self.p.N_particles, dtype=bool)
        self.was_outside_e = np.zeros((self.num_antennas, self.p.N_particles), dtype=bool)
        self.was_outside_i = np.zeros((self.num_antennas, self.p.N_particles), dtype=bool)

        self.cloud_injected = False

    def inject_if_time(self, step: int, dt: float, t_delay: float) -> bool:
        """Activates particles if current simulation time exceeds t_delay."""
        if not self.cloud_injected and step * dt >= t_delay:
            self.active_e[:] = True
            self.active_i[:] = True
            self.was_outside_e[:] = False
            self.was_outside_i[:] = False
            self.cloud_injected = True
        return self.cloud_injected

    def push_species(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        vx: np.ndarray,
        vy: np.ndarray,
        vz: np.ndarray,
        active: np.ndarray,
        was_outside: np.ndarray,
        mass: float,
        phys_charge: float,
        macro_charge: float,
        field_solver,
        antenna_masks_3d: List[np.ndarray],
        Ewx_list: List[np.ndarray],
        Ewy_list: List[np.ndarray],
        Ewz_list: List[np.ndarray],
        enable_collection: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Advances a single species by dt, handles CCD collisions with spacecraft,
        boundary escape, antenna collection, and computes induced / collected currents.
        """
        col_currs = np.zeros(self.num_antennas)
        ind_currs = np.zeros(self.num_antennas)
        if not np.any(active):
            return col_currs, ind_currs

        x_act, y_act, z_act = x[active].copy(), y[active].copy(), z[active].copy()
        ax, ay, az = field_solver.get_accel(x_act, y_act, z_act, mass, phys_charge)

        # Symplectic 3D Leapfrog pusher
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

        idx_x_new = np.clip(np.floor((x_new - self.p.x_grid[0]) / self.p.dx), 0, self.p.Nx - 1).astype(np.int64)
        idx_y_new = np.clip(np.floor((y_new - self.p.y_grid[0]) / self.p.dy), 0, self.p.Ny - 1).astype(np.int64)
        idx_z_new = np.clip(np.floor((z_new - self.p.z_grid[0]) / self.p.dz), 0, self.p.Nz - 1).astype(np.int64)

        idx_x_mid = np.clip(np.floor((x_mid - self.p.x_grid[0]) / self.p.dx), 0, self.p.Nx - 1).astype(np.int64)
        idx_y_mid = np.clip(np.floor((y_mid - self.p.y_grid[0]) / self.p.dy), 0, self.p.Ny - 1).astype(np.int64)
        idx_z_mid = np.clip(np.floor((z_mid - self.p.z_grid[0]) / self.p.dz), 0, self.p.Nz - 1).astype(np.int64)

        hit_sc = self.spacecraft_mask_3d[idx_x_new, idx_y_new, idx_z_new] | self.spacecraft_mask_3d[idx_x_mid, idx_y_mid, idx_z_mid]
        global_active_indices = np.where(new_active)[0]
        destroyed_by_sc = global_active_indices[hit_sc]
        new_active[destroyed_by_sc] = False

        for a_idx in range(self.num_antennas):
            ant_geom = (
                self.antenna_geometries[a_idx]
                if self.antenna_geometries is not None and a_idx < len(self.antenna_geometries)
                else None
            )

            if ant_geom is not None and hasattr(ant_geom, 'p_start') and hasattr(ant_geom, 'p_end'):
                # Exact sub-grid geometric wire collision test
                p_start = ant_geom.p_start
                p_end = ant_geom.p_end
                wire_r = getattr(ant_geom, 'radius', 0.015)
                is_inside = check_wire_collision_3d(
                    x[new_active], y[new_active], z[new_active],
                    p_start, p_end, wire_r
                )
            else:
                # Discrete grid mask lookup (fallback for SPIS meshes)
                idx_x_ant = np.clip(np.floor((x[new_active] - self.p.x_grid[0]) / self.p.dx), 0, self.p.Nx - 1).astype(np.int64)
                idx_y_ant = np.clip(np.floor((y[new_active] - self.p.y_grid[0]) / self.p.dy), 0, self.p.Ny - 1).astype(np.int64)
                idx_z_ant = np.clip(np.floor((z[new_active] - self.p.z_grid[0]) / self.p.dz), 0, self.p.Nz - 1).astype(np.int64)
                is_inside = antenna_masks_3d[a_idx][idx_x_ant, idx_y_ant, idx_z_ant]

            is_outside = ~is_inside
            was_out_act = was_outside[a_idx, new_active]
            crossed_ant = is_inside & was_out_act
            was_outside[a_idx, new_active] = is_outside

            if enable_collection and np.any(crossed_ant):
                global_active_idx = np.where(new_active)[0]
                absorbed_global_indices = global_active_idx[crossed_ant]

                col_currs[a_idx] = len(absorbed_global_indices) * macro_charge / self.p.dt
                new_active[absorbed_global_indices] = False

            if np.any(new_active):
                Ewx = field_solver.interp_field(x[new_active], y[new_active], z[new_active], Ewx_list[a_idx])
                Ewy = field_solver.interp_field(x[new_active], y[new_active], z[new_active], Ewy_list[a_idx])
                Ewz = field_solver.interp_field(x[new_active], y[new_active], z[new_active], Ewz_list[a_idx])

                ind_currs[a_idx] = calc_induced_current(
                    macro_charge, vx[new_active], vy[new_active], Ewx, Ewy,
                    vz=vz[new_active], Ewz=Ewz
                )

        new_active[new_active & (x <= -self.p.L_x)] = False
        new_active[new_active & (x >= self.p.L_x)] = False
        new_active[new_active & (y <= -self.p.L_y)] = False
        new_active[new_active & (y >= self.p.L_y)] = False
        new_active[new_active & (z <= -self.p.L_z)] = False
        new_active[new_active & (z >= self.p.L_z)] = False

        active[:] = new_active
        return col_currs, ind_currs
