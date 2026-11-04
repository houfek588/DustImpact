# -*- coding: utf-8 -*-
"""
Particle Ensemble manager for 3D PIC Simulation.
Handles electron and ion state arrays, injection geometries, phase space tracking,
and continuous collision detection (CCD) against spacecraft and antenna structures.
"""

from typing import Tuple, List, Optional, Any
import numpy as np

from dust_impact.physics.constants import e, m_e
from dust_impact.physics.ramo_shockley import calc_induced_current_from_potential
from dust_impact.physics.emissions import (
    sample_maxwellian_velocities, sample_half_maxwellian_velocities, sample_cone_beam_velocities
)
from dust_impact.numerics.pushers import leapfrog_step_3d
from dust_impact.solver.config_loader import SimulationParams3D
from dust_impact.geometry.surface import check_wire_collision_3d
from dust_impact.geometry.analytical import voxelize_analytical_spacecraft_part


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
        toggles: Optional[Any] = None,
        spacecraft_geometry: Optional[Any] = None,
    ):
        self.p = params
        self.spacecraft_mask_3d = spacecraft_mask_3d
        self.num_antennas = num_antennas
        self.antenna_geometries = antenna_geometries
        self.toggles = toggles
        self.spacecraft_geometry = spacecraft_geometry

        # The active temperatures (solar wind or impact cloud, depending on the injection mode)
        # are encoded in the most probable speeds v_th = sqrt(2 e T / m) set by the configuration.
        T_e_eV = m_e * self.p.v_th_e**2 / (2.0 * e)
        T_i_eV = self.p.m_i * self.p.v_th_i**2 / (2.0 * e)

        mode = getattr(self.p, 'plasma_injection_mode', 'point_cloud')

        if mode == 'homogeneous':
            # 1. Homogeneous uniform spatial distribution across domain [-Lx, Lx], [-Ly, Ly], [-Lz, Lz]
            pts_e = self._sample_outside_spacecraft(self.p.N_particles)
            pts_i = pts_e.copy()

            self.x_e, self.y_e, self.z_e = pts_e[:, 0], pts_e[:, 1], pts_e[:, 2]
            self.x_i, self.y_i, self.z_i = pts_i[:, 0], pts_i[:, 1], pts_i[:, 2]

            # Isotropic Maxwellians of the solar wind temperatures
            v_e = sample_maxwellian_velocities(self.p.N_particles, T_e_eV, m_e)
            v_i = sample_maxwellian_velocities(self.p.N_particles, T_i_eV, self.p.m_i)
            self.vx_e, self.vy_e, self.vz_e = v_e[:, 0], v_e[:, 1], v_e[:, 2]
            self.vx_i, self.vy_i, self.vz_i = v_i[:, 0], v_i[:, 1], v_i[:, 2]

        else:
            # 2. Concentrated point cloud (dust impact expansion) mode
            impact_normal_val = getattr(self.p, 'impact_normal', [-0.7071, 0.7071, 0.0])
            n = np.array(impact_normal_val, dtype=float)
            n_norm = np.linalg.norm(n)
            if n_norm > 0:
                n = n / n_norm
            else:
                n = np.array([-0.7071, 0.7071, 0.0])

            # Electrons: half-Maxwellian of the impact cloud temperature expanding away from the surface.
            # Ions: drifting conical beam around the normal if a drift speed is given, else half-Maxwellian.
            impact_cfg = getattr(self.p, 'impact', None)
            v_drift = float(getattr(impact_cfg, 'ion_drift_speed_m_s', 0.0) or 0.0)
            if v_drift > 0.0:
                v_i = sample_cone_beam_velocities(
                    self.p.N_particles, v_drift, float(getattr(impact_cfg, 'ion_cone_half_angle_deg', 90.0)),
                    T_i_eV, self.p.m_i, n
                )
            else:
                v_i = sample_half_maxwellian_velocities(self.p.N_particles, T_i_eV, self.p.m_i, n)
            v_e = sample_half_maxwellian_velocities(self.p.N_particles, T_e_eV, m_e, n)
            self.vx_i, self.vy_i, self.vz_i = v_i[:, 0], v_i[:, 1], v_i[:, 2]
            self.vx_e, self.vy_e, self.vz_e = v_e[:, 0], v_e[:, 1], v_e[:, 2]

            impact_point_val = getattr(self.p, 'impact_pos', getattr(self.p, 'impact_location_xyz_m', [0.0, 0.0, 0.0]))
            r0 = np.array(impact_point_val, dtype=float)

            grid_step = max(self.p.dx, self.p.dy, self.p.dz)
            r_start = r0 + n * (0.2 * grid_step)
            for step_mult in range(1, 50):
                if not self._inside_spacecraft(r_start[0:1], r_start[1:2], r_start[2:3])[0]:
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

    def _spacecraft_shape(self) -> Optional[Any]:
        """
        Exact analytical description of the spacecraft body (sphere, box, cylinder or composite), or
        None when the body is only known as a conductor mask on the grid (SPIS geometry).
        """
        geom_cfg = getattr(self.p, 'geometry', None)
        if getattr(geom_cfg, 'source', 'analytical') != 'analytical':
            return None
        sc_geom = self.spacecraft_geometry
        if sc_geom is None and geom_cfg is not None:
            sc_geom = getattr(getattr(geom_cfg, 'analytical', None), 'spacecraft', None)
        if sc_geom is None or not hasattr(sc_geom, 'type'):
            return None
        return sc_geom

    def _inside_spacecraft(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        """
        Tests which points lie inside the spacecraft body.

        An analytical body is tested exactly, so that the collision surface coincides with the
        surface the reflection model places particles on. Otherwise the conductor mask is looked up
        at the nearest grid node, i.e. every node represents the cube of half-width dx/2 around it
        (the same region the field solver treats as conductor).

        Args:
            x, y, z: Point coordinates [m], arrays of equal shape.

        Returns:
            Boolean array of the same shape, True inside the spacecraft.
        """
        shape = self._spacecraft_shape()
        if shape is not None:
            return voxelize_analytical_spacecraft_part(x, y, z, shape)
        ix = np.clip(np.rint((x - self.p.x_grid[0]) / self.p.dx), 0, self.p.Nx - 1).astype(np.int64)
        iy = np.clip(np.rint((y - self.p.y_grid[0]) / self.p.dy), 0, self.p.Ny - 1).astype(np.int64)
        iz = np.clip(np.rint((z - self.p.z_grid[0]) / self.p.dz), 0, self.p.Nz - 1).astype(np.int64)
        return self.spacecraft_mask_3d[ix, iy, iz]

    def _sample_outside_spacecraft(self, n_pts: int) -> np.ndarray:
        """
        Samples points uniformly in the domain box, excluding the spacecraft body.

        Rejection sampling is done block-wise: all points falling into masked cells are redrawn
        together until none remains inside the spacecraft.

        Parameters:
        -----------
        n_pts : int
            Number of points.

        Returns:
        --------
        pts : np.ndarray
            Positions of shape (n_pts, 3) in metres.
        """
        low = np.array([-self.p.L_x, -self.p.L_y, -self.p.L_z])
        high = np.array([self.p.L_x, self.p.L_y, self.p.L_z])

        pts = np.random.uniform(low=low, high=high, size=(n_pts, 3))
        redraw = np.arange(n_pts)
        while redraw.size > 0:
            inside = self._inside_spacecraft(pts[redraw, 0], pts[redraw, 1], pts[redraw, 2])
            redraw = redraw[inside]
            pts[redraw] = np.random.uniform(low=low, high=high, size=(redraw.size, 3))
        return pts

    def inject_if_time(self, step: int, dt: float, t_delay: float) -> bool:
        """Activates particles if current simulation time exceeds t_delay."""
        if not self.cloud_injected and step * dt >= t_delay:
            self.active_e[:] = True
            self.active_i[:] = True
            self.was_outside_e[:] = False
            self.was_outside_i[:] = False
            self.cloud_injected = True
        return self.cloud_injected

    def _reflect_particles_from_spacecraft(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        vx: np.ndarray,
        vy: np.ndarray,
        vz: np.ndarray,
        hit_indices: np.ndarray,
        x_old: np.ndarray,
        y_old: np.ndarray,
        z_old: np.ndarray,
    ) -> None:
        """
        Specular elastic reflection of particles colliding with the spacecraft body.
        Preserves kinetic energy exactly and ensures position is pushed outside the surface.
        """
        if len(hit_indices) == 0:
            return

        sc_geom = self._spacecraft_shape()
        geom_type = str(getattr(sc_geom, 'type', '')).lower() if sc_geom is not None else ''
        eps = 0.05 * min(self.p.dx, self.p.dy, self.p.dz)

        if geom_type == 'sphere':
            center = np.array(getattr(sc_geom, 'center', [0.0, 0.0, 0.0]), dtype=float)
            radius = float(getattr(sc_geom, 'radius', 1.0))

            px = x[hit_indices] - center[0]
            py = y[hit_indices] - center[1]
            pz = z[hit_indices] - center[2]
            dist = np.sqrt(px**2 + py**2 + pz**2)

            ox = x_old - center[0]
            oy = y_old - center[1]
            oz = z_old - center[2]
            dist_old = np.sqrt(ox**2 + oy**2 + oz**2)

            nx = np.where(dist > 1e-9, px / np.maximum(dist, 1e-12), np.where(dist_old > 1e-9, ox / np.maximum(dist_old, 1e-12), 1.0))
            ny = np.where(dist > 1e-9, py / np.maximum(dist, 1e-12), np.where(dist_old > 1e-9, oy / np.maximum(dist_old, 1e-12), 0.0))
            nz = np.where(dist > 1e-9, pz / np.maximum(dist, 1e-12), np.where(dist_old > 1e-9, oz / np.maximum(dist_old, 1e-12), 0.0))

            # Reposition particle strictly outside sphere surface
            delta = np.maximum(0.0, radius - dist)
            new_dist = radius + np.maximum(delta, eps)
            x[hit_indices] = center[0] + nx * new_dist
            y[hit_indices] = center[1] + ny * new_dist
            z[hit_indices] = center[2] + nz * new_dist

            # Specular velocity reflection: v' = v - 2(v·n)n
            vn = vx[hit_indices] * nx + vy[hit_indices] * ny + vz[hit_indices] * nz
            inward = vn < 0
            vx[hit_indices] = np.where(inward, vx[hit_indices] - 2.0 * vn * nx, vx[hit_indices])
            vy[hit_indices] = np.where(inward, vy[hit_indices] - 2.0 * vn * ny, vy[hit_indices])
            vz[hit_indices] = np.where(inward, vz[hit_indices] - 2.0 * vn * nz, vz[hit_indices])

        elif geom_type in ('box', 'cube'):
            center = np.array(getattr(sc_geom, 'center', [0.0, 0.0, 0.0]), dtype=float)
            dims = getattr(sc_geom, 'dimensions', None)
            if dims is None and hasattr(sc_geom, 'side_length') and sc_geom.side_length:
                dims = [sc_geom.side_length] * 3
            elif dims is None:
                dims = [1.0, 1.0, 1.0]
            dims = np.array(dims, dtype=float)
            half_dims = dims * 0.5

            px = x[hit_indices] - center[0]
            py = y[hit_indices] - center[1]
            pz = z[hit_indices] - center[2]

            rx = np.abs(px) / (half_dims[0] + 1e-12)
            ry = np.abs(py) / (half_dims[1] + 1e-12)
            rz = np.abs(pz) / (half_dims[2] + 1e-12)

            for i_idx, k in enumerate(hit_indices):
                r_vals = [rx[i_idx], ry[i_idx], rz[i_idx]]
                axis = int(np.argmax(r_vals))
                if axis == 0:
                    sign = 1.0 if px[i_idx] >= 0 else -1.0
                    x[k] = center[0] + sign * (half_dims[0] + eps)
                    if vx[k] * sign < 0:
                        vx[k] = -vx[k]
                elif axis == 1:
                    sign = 1.0 if py[i_idx] >= 0 else -1.0
                    y[k] = center[1] + sign * (half_dims[1] + eps)
                    if vy[k] * sign < 0:
                        vy[k] = -vy[k]
                else:
                    sign = 1.0 if pz[i_idx] >= 0 else -1.0
                    z[k] = center[2] + sign * (half_dims[2] + eps)
                    if vz[k] * sign < 0:
                        vz[k] = -vz[k]

        else:
            # Fallback for arbitrary voxel / SPIS mesh: restore previous valid position outside body
            # and reverse velocity vector
            x[hit_indices] = x_old
            y[hit_indices] = y_old
            z[hit_indices] = z_old
            vx[hit_indices] = -vx[hit_indices]
            vy[hit_indices] = -vy[hit_indices]
            vz[hit_indices] = -vz[hit_indices]

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
        boundary_mode: Optional[str] = None,
        Vw_list: Optional[List[np.ndarray]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Advances a single species by dt, handles CCD collisions with spacecraft,
        boundary escape / periodic wrapping, antenna collection, and computes induced / collected currents.

        Antenna currents are evaluated from the weighting potentials Vw (Ramo-Shockley theorem in
        its charge-conserving form Q_ind = q * Vw(r)), so that the total charge delivered to antenna a
        by one macro-particle over its whole trajectory is exactly q * [Vw_a(r_end) - Vw_a(r_start)]:
          - free particles:                induced  += q * [Vw_a(r^{n+1}) - Vw_a(r^n)] / dt
          - collected by antenna a:        collected += q * [1 - Vw_a(r^n)] / dt (residual charge only)
          - collected by another antenna,
            or absorbed by the spacecraft: induced  += q * [0 - Vw_a(r^n)] / dt
        If Vw_list holds one more grid than there are antennas, that grid is the spacecraft weighting
        potential (floating spacecraft): particles absorbed by the spacecraft are then collected by
        this extra electrode (index num_antennas) and end at Vw_sc = 1.
        The Ewx/Ewy/Ewz lists are kept in the signature for API compatibility only.

        Returns:
            (col_currs, ind_currs): collected and induced currents per electrode [A].
        """
        n_el = len(Vw_list) if Vw_list is not None else self.num_antennas
        sc_electrode = self.num_antennas if n_el > self.num_antennas else None
        col_currs = np.zeros(n_el)
        ind_currs = np.zeros(n_el)
        if not np.any(active):
            return col_currs, ind_currs
        if self.num_antennas > 0 and (Vw_list is None or len(Vw_list) < self.num_antennas):
            raise ValueError("push_species requires one weighting potential grid (Vw_list) per antenna.")

        x_act, y_act, z_act = x[active].copy(), y[active].copy(), z[active].copy()
        ax, ay, az = field_solver.get_accel(x_act, y_act, z_act, mass, phys_charge)

        # Weighting potentials at the pre-push positions r^n, one array per electrode
        vw_old = [
            field_solver.interp_field(x_act, y_act, z_act, Vw_list[a_idx])
            for a_idx in range(n_el)
        ]

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
            vw_old = [v[valid_mask] for v in vw_old]

        if not np.any(new_active):
            active[:] = new_active
            return col_currs, ind_currs

        # Continuous Collision Detection (CCD) via trajectory segment midpoint sampling
        x_new, y_new, z_new = x[new_active], y[new_active], z[new_active]
        x_mid = 0.5 * (x_act + x_new)
        y_mid = 0.5 * (y_act + y_new)
        z_mid = 0.5 * (z_act + z_new)

        hit_sc = self._inside_spacecraft(x_new, y_new, z_new) | self._inside_spacecraft(x_mid, y_mid, z_mid)
        global_active_indices = np.where(new_active)[0]
        destroyed_by_sc = global_active_indices[hit_sc]

        sc_mode = str(
            getattr(self.toggles, 'particle_spacecraft_mode', getattr(self.p, 'particle_spacecraft_mode', 'absorb'))
        ).lower()

        if sc_mode == "reflect":
            if np.any(hit_sc):
                self._reflect_particles_from_spacecraft(
                    x, y, z, vx, vy, vz, destroyed_by_sc,
                    x_act[hit_sc], y_act[hit_sc], z_act[hit_sc]
                )

        # Index of the electrode that collected each particle in this step (-1 = not collected)
        landed_on = np.full(len(global_active_indices), -1, dtype=np.int64)
        if sc_mode != "reflect":
            new_active[destroyed_by_sc] = False
            if sc_electrode is not None:
                landed_on[hit_sc] = sc_electrode

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

                landed_on[np.searchsorted(global_active_indices, absorbed_global_indices)] = a_idx
                new_active[absorbed_global_indices] = False

        # Ramo-Shockley currents from the change of weighting potential over this step.
        # Local arrays below are aligned with global_active_indices (particles active at r^n).
        still_active = new_active[global_active_indices]
        idx_still = global_active_indices[still_active]
        for a_idx in range(n_el):
            vw_new = np.zeros_like(vw_old[a_idx])
            if len(idx_still) > 0:
                vw_new[still_active] = field_solver.interp_field(
                    x[idx_still], y[idx_still], z[idx_still], Vw_list[a_idx]
                )
            # Particles collected by this electrode end at Vw_a = 1; the residual charge is reported as collected
            landed_here = landed_on == a_idx
            vw_new[landed_here] = 1.0
            # Particles absorbed by another conductor end at Vw_a = 0 (vw_new already 0)
            free_or_lost = ~landed_here
            ind_currs[a_idx] = calc_induced_current_from_potential(
                macro_charge, vw_old[a_idx][free_or_lost], vw_new[free_or_lost], self.p.dt
            )
            col_currs[a_idx] = calc_induced_current_from_potential(
                macro_charge, vw_old[a_idx][landed_here], vw_new[landed_here], self.p.dt
            )

        bnd_mode = str(
            boundary_mode
            or getattr(self.toggles, 'particle_boundary_mode', getattr(self.p, 'particle_boundary_mode', 'open'))
        ).lower()

        if bnd_mode == "periodic":
            Lx, Ly, Lz = self.p.L_x, self.p.L_y, self.p.L_z
            act_idx = np.where(new_active)[0]
            if len(act_idx) > 0:
                x[act_idx] = (x[act_idx] + Lx) % (2.0 * Lx) - Lx
                y[act_idx] = (y[act_idx] + Ly) % (2.0 * Ly) - Ly
                z[act_idx] = (z[act_idx] + Lz) % (2.0 * Lz) - Lz
        else:
            new_active[new_active & (x <= -self.p.L_x)] = False
            new_active[new_active & (x >= self.p.L_x)] = False
            new_active[new_active & (y <= -self.p.L_y)] = False
            new_active[new_active & (y >= self.p.L_y)] = False
            new_active[new_active & (z <= -self.p.L_z)] = False
            new_active[new_active & (z >= self.p.L_z)] = False

        active[:] = new_active
        return col_currs, ind_currs
