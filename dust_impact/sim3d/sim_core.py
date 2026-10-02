# -*- coding: utf-8 -*-
"""
3D PIC Simulation Core Solver.
Decomposed architecture orchestrating ParticleEnsemble, FieldSolver3D, and AntennaCircuitCollector.
Maintains 100% backward compatibility for existing callers and test suites.
"""

from typing import Dict, Any, List, Optional
import numpy as np

from dust_impact.physics.constants import e, m_e
from dust_impact.sim3d.config_loader import SimulationParams3D, SimulationToggles3D
from dust_impact.sim3d.ensemble import ParticleEnsemble
from dust_impact.sim3d.field_solver import FieldSolver3D
from dust_impact.sim3d.collector import AntennaCircuitCollector

__all__ = [
    "DustImpactSimulation3D",
    "ParticleEnsemble",
    "FieldSolver3D",
    "AntennaCircuitCollector",
]


class DustImpactSimulation3D:
    """
    Top-level orchestrator for 3D PIC dust impact plasma expansion simulation.
    Integrates particle dynamics, Poisson self-field solving, and probe circuit response.
    """

    def __init__(
        self,
        params: SimulationParams3D,
        toggles: SimulationToggles3D,
        V_bg,
        Vw_grids=None,
        Ex_bg=None,
        Ey_bg=None,
        Ez_bg=None,
        Ewx_list=None,
        Ewy_list=None,
        Ewz_list=None,
        antenna_masks_3d=None,
        spacecraft_mask_3d=None,
    ):
        antenna_geometries = None
        if hasattr(V_bg, 'antenna_masks_3d') and hasattr(V_bg, 'spacecraft_mask_3d'):
            geom = V_bg
            V_bg = geom.V_bg
            Vw_grids = geom.Vw_grids
            Ex_bg, Ey_bg, Ez_bg = geom.Ex_bg, geom.Ey_bg, geom.Ez_bg
            Ewx_list, Ewy_list, Ewz_list = geom.Ewx_list, geom.Ewy_list, geom.Ewz_list
            antenna_masks_3d = geom.antenna_masks_3d
            spacecraft_mask_3d = geom.spacecraft_mask_3d
            antenna_geometries = getattr(geom, 'antenna_geometries', None)

        if antenna_geometries is None and hasattr(params, 'geometry'):
            geom_cfg = params.geometry
            if getattr(geom_cfg, 'source', None) == 'analytical':
                antenna_geometries = getattr(getattr(geom_cfg, 'analytical', None), 'antennas', None)

        self.p = params
        self.toggles = toggles
        self.antenna_masks_3d = antenna_masks_3d
        self.spacecraft_mask_3d = spacecraft_mask_3d
        self.num_antennas = (
            len(antenna_masks_3d)
            if antenna_masks_3d is not None
            else len(getattr(self.p.vtk_files, 'antenna_weighting_files', getattr(self.p.vtk_files, 'antenna_weighting', [])))
        )

        self.combined_mask = self.spacecraft_mask_3d.copy()
        for mask in self.antenna_masks_3d:
            self.combined_mask |= mask

        # Subsystems
        self.particles = ParticleEnsemble(
            params, spacecraft_mask_3d, self.num_antennas, antenna_geometries=antenna_geometries
        )
        self.field_solver = FieldSolver3D(
            params, toggles, V_bg, Ex_bg, Ey_bg, Ez_bg,
            Ewx_list, Ewy_list, Ewz_list, self.combined_mask, self.num_antennas
        )
        self.collector = AntennaCircuitCollector(params, toggles, self.num_antennas)

        # Direct references to particle arrays for zero-overhead backward compatibility
        self.x_e = self.particles.x_e
        self.y_e = self.particles.y_e
        self.z_e = self.particles.z_e
        self.vx_e = self.particles.vx_e
        self.vy_e = self.particles.vy_e
        self.vz_e = self.particles.vz_e
        self.active_e = self.particles.active_e
        self.was_outside_e = self.particles.was_outside_e

        self.x_i = self.particles.x_i
        self.y_i = self.particles.y_i
        self.z_i = self.particles.z_i
        self.vx_i = self.particles.vx_i
        self.vy_i = self.particles.vy_i
        self.vz_i = self.particles.vz_i
        self.active_i = self.particles.active_i
        self.was_outside_i = self.particles.was_outside_i

        # Direct references to current and voltage arrays
        self.ind_curr_e = self.collector.ind_curr_e
        self.ind_curr_i = self.collector.ind_curr_i
        self.col_curr_e = self.collector.col_curr_e
        self.col_curr_i = self.collector.col_curr_i
        self.tot_curr = self.collector.tot_curr
        self.voltage_ant = self.collector.voltage_ant

        # Weighting fields references
        self.Ewx_list = Ewx_list
        self.Ewy_list = Ewy_list
        self.Ewz_list = Ewz_list

        self.history = {
            'V': [], 'rho': [], 't': [],
            'x_e': [], 'y_e': [], 'z_e': [],
            'x_i': [], 'y_i': [], 'z_i': [],
            'vx_e': [], 'vy_e': [], 'vz_e': [],
            'vx_i': [], 'vy_i': [], 'vz_i': [],
        }

    # Backward compatibility properties
    @property
    def cloud_injected(self) -> bool:
        return self.particles.cloud_injected

    @cloud_injected.setter
    def cloud_injected(self, val: bool) -> None:
        self.particles.cloud_injected = val

    @property
    def V_self_grid(self) -> np.ndarray:
        return self.field_solver.V_self_grid

    @V_self_grid.setter
    def V_self_grid(self, val: np.ndarray) -> None:
        self.field_solver.V_self_grid = val

    @property
    def rho_grid(self) -> np.ndarray:
        return self.field_solver.rho_grid

    @rho_grid.setter
    def rho_grid(self, val: np.ndarray) -> None:
        self.field_solver.rho_grid = val

    @property
    def Ex_bg_base(self) -> np.ndarray:
        return self.field_solver.Ex_bg_base

    @property
    def Ey_bg_base(self) -> np.ndarray:
        return self.field_solver.Ey_bg_base

    @property
    def Ez_bg_base(self) -> np.ndarray:
        return self.field_solver.Ez_bg_base

    @property
    def Ex_bg(self) -> np.ndarray:
        return self.field_solver.Ex_bg

    @Ex_bg.setter
    def Ex_bg(self, val: np.ndarray) -> None:
        self.field_solver.Ex_bg = val

    @property
    def Ey_bg(self) -> np.ndarray:
        return self.field_solver.Ey_bg

    @Ey_bg.setter
    def Ey_bg(self, val: np.ndarray) -> None:
        self.field_solver.Ey_bg = val

    @property
    def Ez_bg(self) -> np.ndarray:
        return self.field_solver.Ez_bg

    @Ez_bg.setter
    def Ez_bg(self, val: np.ndarray) -> None:
        self.field_solver.Ez_bg = val

    @property
    def Ex_self(self) -> np.ndarray:
        return self.field_solver.Ex_self

    @property
    def Ey_self(self) -> np.ndarray:
        return self.field_solver.Ey_self

    @property
    def Ez_self(self) -> np.ndarray:
        return self.field_solver.Ez_self

    @property
    def V_bg_base(self) -> np.ndarray:
        return self.field_solver.V_bg_base

    @property
    def amg_solver(self):
        return self.field_solver.amg_solver

    @property
    def lu_solver(self):
        return self.field_solver.lu_solver

    # Delegating helper methods for backward compatibility
    def _build_poisson_solver(self) -> None:
        self.field_solver._build_poisson_solver()

    def _update_background_fields(self, step: int) -> None:
        self.field_solver.update_background_fields(self.collector.voltage_ant, step)

    def _solve_poisson_equation(self) -> None:
        self.field_solver.solve_poisson(self.particles)

    def _interp_3d_fast(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, Field_3D: np.ndarray) -> np.ndarray:
        return self.field_solver.interp_field(x, y, z, Field_3D)

    def _get_accel(self, x_act: np.ndarray, y_act: np.ndarray, z_act: np.ndarray, mass: float, phys_charge: float):
        return self.field_solver.get_accel(x_act, y_act, z_act, mass, phys_charge)

    def _push_species(self, x, y, z, vx, vy, vz, active, was_outside, mass, phys_charge, macro_charge):
        enable_coll = getattr(self.toggles, 'enable_antenna_particle_collection', True)
        return self.particles.push_species(
            x, y, z, vx, vy, vz, active, was_outside, mass, phys_charge, macro_charge,
            self.field_solver, self.antenna_masks_3d,
            self.Ewx_list, self.Ewy_list, self.Ewz_list,
            enable_collection=enable_coll
        )

    def _save_history(self, current_time: float) -> None:
        self.history['V'].append((self.field_solver.V_bg_base + self.field_solver.V_self_grid).copy())
        self.history['rho'].append(self.field_solver.rho_grid.copy())
        self.history['t'].append(current_time)

        stride = self.p.plot_stride
        self.history['x_e'].append(np.where(self.particles.active_e, self.particles.x_e, np.nan)[::stride])
        self.history['y_e'].append(np.where(self.particles.active_e, self.particles.y_e, np.nan)[::stride])
        self.history['z_e'].append(np.where(self.particles.active_e, self.particles.z_e, np.nan)[::stride])

        self.history['x_i'].append(np.where(self.particles.active_i, self.particles.x_i, np.nan)[::stride])
        self.history['y_i'].append(np.where(self.particles.active_i, self.particles.y_i, np.nan)[::stride])
        self.history['z_i'].append(np.where(self.particles.active_i, self.particles.z_i, np.nan)[::stride])

        self.history['vx_e'].append(np.where(self.particles.active_e, self.particles.vx_e, np.nan)[::stride])
        self.history['vy_e'].append(np.where(self.particles.active_e, self.particles.vy_e, np.nan)[::stride])
        self.history['vz_e'].append(np.where(self.particles.active_e, self.particles.vz_e, np.nan)[::stride])

        self.history['vx_i'].append(np.where(self.particles.active_i, self.particles.vx_i, np.nan)[::stride])
        self.history['vy_i'].append(np.where(self.particles.active_i, self.particles.vy_i, np.nan)[::stride])
        self.history['vz_i'].append(np.where(self.particles.active_i, self.particles.vz_i, np.nan)[::stride])

    def get_metadata_dict(self) -> Dict[str, Any]:
        """Generate comprehensive metadata dictionary for simulation provenance."""
        from datetime import datetime
        import dust_impact
        return {
            'package_version': getattr(dust_impact, '__version__', '0.3.0'),
            'timestamp': datetime.now().isoformat(),
            'dt': float(self.p.dt),
            'time_step_s': float(self.p.time_step_s),
            'num_time_steps': int(getattr(self.p, 'num_time_steps', self.p.steps)),
            'simulation_duration_s': float(self.p.simulation_duration_s),
            'steps': int(self.p.steps),
            'grid_shape': [int(self.p.Nx), int(self.p.Ny), int(self.p.Nz)],
            'dx': float(self.p.dx),
            'dy': float(self.p.dy),
            'dz': float(self.p.dz),
            'L_x': float(self.p.L_x),
            'L_y': float(self.p.L_y),
            'L_z': float(self.p.L_z),
            'x_grid': self.p.x_grid.tolist(),
            'y_grid': self.p.y_grid.tolist(),
            'z_grid': self.p.z_grid.tolist(),
            'time_array': self.p.time_array.tolist(),
            'num_antennas': int(self.num_antennas),
            'antenna_capacitance_F': list(self.p.C_ant),
            'antenna_resistance_Ohm': list(self.p.R_ant),
            'antenna_bias_voltage_V': list(self.p.V_bias),
            'impact_location_xyz_m': list(getattr(self.p, 'impact_pos', [0.0, 0.0, 0.0])),
            'impact_normal': list(getattr(self.p, 'impact_normal', [0.0, 0.0, 1.0])),
            'total_impact_charge_C': float(self.p.total_impact_charge_C),
            'ion_mass_amu': float(self.p.ion_mass_amu),
            'solar_wind_electron_temp_eV': float(self.p.solar_wind_electron_temp_eV),
            'solar_wind_density_m3': float(self.p.solar_wind_density_m3),
            'plasma_injection_mode': getattr(self.p, 'plasma_injection_mode', 'point_cloud')
        }

    def _build_results_dict(self) -> Dict[str, Any]:
        """Construct full results dictionary with smoothed signals, voltages, and metadata."""
        smoothed = self.collector.build_smoothed_signals()
        return {
            'smooth_induced': smoothed['smooth_induced'],
            'smooth_collected': smoothed['smooth_collected'],
            'smooth_total': smoothed['smooth_total'],
            'voltage_ant': self.collector.voltage_ant,
            'history': self.history,
            'metadata': self.get_metadata_dict(),
        }

    def step(self, step_idx: int) -> None:
        """Executes a single PIC simulation step."""
        # 1. Injection check
        self.particles.inject_if_time(step_idx, self.p.dt, self.p.t_delay)

        # 2. Update background fields from antenna dynamic voltages
        if step_idx > 0:
            self.field_solver.update_background_fields(self.collector.voltage_ant, step_idx)

        # 3. Poisson equation solve
        self.field_solver.solve_poisson(self.particles)

        # 4. Push species if cloud injected
        if self.particles.cloud_injected:
            enable_coll = getattr(self.toggles, 'enable_antenna_particle_collection', True)
            ce, ie = self.particles.push_species(
                self.particles.x_e, self.particles.y_e, self.particles.z_e,
                self.particles.vx_e, self.particles.vy_e, self.particles.vz_e,
                self.particles.active_e, self.particles.was_outside_e,
                m_e, -e, -self.p.q_macro,
                self.field_solver, self.antenna_masks_3d,
                self.Ewx_list, self.Ewy_list, self.Ewz_list,
                enable_collection=enable_coll
            )
            ci, ii = self.particles.push_species(
                self.particles.x_i, self.particles.y_i, self.particles.z_i,
                self.particles.vx_i, self.particles.vy_i, self.particles.vz_i,
                self.particles.active_i, self.particles.was_outside_i,
                self.p.m_i, e, self.p.q_macro,
                self.field_solver, self.antenna_masks_3d,
                self.Ewx_list, self.Ewy_list, self.Ewz_list,
                enable_collection=enable_coll
            )
            self.collector.record_step_currents(step_idx, ce, ci, ie, ii)

        # 5. Circuit ODE integration
        self.collector.update_circuit(step_idx)

    def run(
        self,
        checkpoint_filepath: Optional[str] = None,
        checkpoint_interval: int = 0,
        h5_writer: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Runs the full 3D PIC simulation over all configured time steps."""
        print(f"Spouštím 3D Simulaci... Mřížka: {self.p.Nx}x{self.p.Ny}x{self.p.Nz} | Počet antén: {self.num_antennas}")

        for step in range(self.p.steps):
            self.step(step)

            # History recording & streaming HDF5 export
            if step % self.p.save_interval == 0 or step == self.p.steps - 1:
                t_cur = step * self.p.dt
                self._save_history(t_cur)
                if h5_writer is not None:
                    part_dict = {
                        'x_e': self.history['x_e'][-1],
                        'y_e': self.history['y_e'][-1],
                        'z_e': self.history['z_e'][-1],
                        'vx_e': self.history['vx_e'][-1],
                        'vy_e': self.history['vy_e'][-1],
                        'vz_e': self.history['vz_e'][-1],
                        'x_i': self.history['x_i'][-1],
                        'y_i': self.history['y_i'][-1],
                        'z_i': self.history['z_i'][-1],
                        'vx_i': self.history['vx_i'][-1],
                        'vy_i': self.history['vy_i'][-1],
                        'vz_i': self.history['vz_i'][-1],
                    }
                    h5_writer.write_history_frame(t_cur, self.history['V'][-1], self.history['rho'][-1], part_dict)

            # Periodic atomic checkpointing
            if checkpoint_filepath and checkpoint_interval > 0 and step > 0 and (
                step % checkpoint_interval == 0 or step == self.p.steps - 1
            ):
                try:
                    from dust_impact.common.io import save_checkpoint
                    inter_res = self._build_results_dict()
                    save_checkpoint(inter_res, checkpoint_filepath, metadata=self.get_metadata_dict())
                except Exception as cp_err:
                    print(f"  [VAROVÁNÍ] Uložení checkpointu selhalo: {cp_err}")

            if step % (max(1, self.p.steps // 10)) == 0:
                print(f"  -> Průběh: {int(step / self.p.steps * 100)}% ({step}/{self.p.steps} kroků)")

        print(f"  -> Průběh: 100% ({self.p.steps}/{self.p.steps} kroků)")

        res = self._build_results_dict()
        if h5_writer is not None:
            h5_writer.finish(res)

        return res
