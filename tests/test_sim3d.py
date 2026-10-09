import unittest
from dust_impact.solver.config_loader import SimulationParams3D, SimulationToggles3D, setup_simulation_parameters_3d
from dust_impact.solver import load_and_interpolate_vtk
from dust_impact.solver.sim_core import DustImpactSimulation3D


class TestSim3D(unittest.TestCase):
    def test_params_initialization(self):
        params, toggles, plot_cfg = setup_simulation_parameters_3d(Vf=-5.0, Vf_antenne=1.0)
        self.assertEqual(params.Vf, -5.0)
        self.assertGreater(params.Nx, 0)
        self.assertGreater(params.Ny, 0)
        self.assertGreater(params.Nz, 0)

    def test_sim_core_3d_synthetic_fallback(self):
        params = SimulationParams3D(
            Vf=-5.0,
            num_macroparticles=50,
            simulation_duration_s=1e-8,
            time_step_s=1e-9,
            grid_nodes_x=8,
            grid_nodes_y=8,
            grid_nodes_z=8,
        )
        toggles = SimulationToggles3D()

        V_bg, Vw_grids, Ex_bg, Ey_bg, Ez_bg, Ewx, Ewy, Ewz, ant_masks, sc_mask = load_and_interpolate_vtk(params)

        sim = DustImpactSimulation3D(
            params, toggles,
            V_bg, Vw_grids, Ex_bg, Ey_bg, Ez_bg, Ewx, Ewy, Ewz, ant_masks, sc_mask
        )
        self.assertGreaterEqual(sim.num_antennas, 1)

    def test_sim_core_3d_homogeneous_mode(self):
        params = SimulationParams3D(
            Vf=-5.0,
            num_macroparticles=100,
            simulation_duration_s=1e-8,
            time_step_s=1e-9,
            grid_nodes_x=8,
            grid_nodes_y=8,
            grid_nodes_z=8,
            plasma_injection_mode="homogeneous"
        )
        toggles = SimulationToggles3D()

        V_bg, Vw_grids, Ex_bg, Ey_bg, Ez_bg, Ewx, Ewy, Ewz, ant_masks, sc_mask = load_and_interpolate_vtk(params)

        sim = DustImpactSimulation3D(
            params, toggles,
            V_bg, Vw_grids, Ex_bg, Ey_bg, Ez_bg, Ewx, Ewy, Ewz, ant_masks, sc_mask
        )
        self.assertEqual(len(sim.x_e), 100)
        self.assertGreater(sim.x_e.max() - sim.x_e.min(), 0.1)

    def test_sim_core_3d_custom_spacecraft_voltage(self):
        from dust_impact.solver.config_loader import VTKFilesConfig
        params = SimulationParams3D(
            Vf=0.0,
            spacecraft_voltage_V=15.0,
            num_macroparticles=50,
            simulation_duration_s=1e-8,
            time_step_s=1e-9,
            grid_nodes_x=8,
            grid_nodes_y=8,
            grid_nodes_z=8,
            vtk_files=VTKFilesConfig(
                spis_background_potential_file="",
                spacecraft_weighting_file="inputs/spis_Vw_body.vtk",
                antenna_weighting_files=["inputs/spis_Vw_ant1.vtk"]
            )
        )
        toggles = SimulationToggles3D()

        V_bg, Vw_grids, Ex_bg, Ey_bg, Ez_bg, Ewx, Ewy, Ewz, ant_masks, sc_mask = load_and_interpolate_vtk(params)
        self.assertGreater(float(V_bg.max()), 10.0)
        self.assertLessEqual(float(V_bg.max()), 15.001)

    def test_particle_boundary_modes(self):
        import numpy as np
        from dust_impact.solver.ensemble import ParticleEnsemble
        from dust_impact.solver.field_solver import FieldSolver3D

        params = SimulationParams3D(
            num_macroparticles=10,
            simulation_duration_s=1e-8,
            time_step_s=1e-9,
            grid_nodes_x=8,
            grid_nodes_y=8,
            grid_nodes_z=8,
        )
        sc_mask = np.zeros((8, 8, 8), dtype=bool)

        # 1. Test open boundary mode
        toggles_open = SimulationToggles3D(particle_boundary_mode="open")
        ens_open = ParticleEnsemble(params, sc_mask, num_antennas=0, toggles=toggles_open)
        ens_open.cloud_injected = True
        field_solver = FieldSolver3D(
            params, toggles_open,
            np.zeros((8, 8, 8)), np.zeros((8, 8, 8)), np.zeros((8, 8, 8)), np.zeros((8, 8, 8)),
            [], [], [], sc_mask, num_antennas=0
        )

        ens_open.x_e[0] = params.L_x - 0.001
        ens_open.y_e[0] = 0.0
        ens_open.z_e[0] = 0.0
        ens_open.vx_e[0] = 1e6  # Moves far past +L_x
        ens_open.vy_e[0] = 0.0
        ens_open.vz_e[0] = 0.0
        ens_open.active_e[0] = True

        ens_open.push_species(
            ens_open.x_e, ens_open.y_e, ens_open.z_e,
            ens_open.vx_e, ens_open.vy_e, ens_open.vz_e,
            ens_open.active_e, ens_open.was_outside_e,
            9.1e-31, -1.6e-19, -1e-15,
            field_solver, [], [], [], [],
            enable_collection=False,
            boundary_mode="open"
        )
        self.assertFalse(ens_open.active_e[0])

        # 2. Test periodic boundary mode
        toggles_periodic = SimulationToggles3D(particle_boundary_mode="periodic")
        ens_periodic = ParticleEnsemble(params, sc_mask, num_antennas=0, toggles=toggles_periodic)
        ens_periodic.cloud_injected = True
        ens_periodic.x_e[0] = params.L_x - 0.001
        ens_periodic.y_e[0] = 0.0
        ens_periodic.z_e[0] = 0.0
        ens_periodic.vx_e[0] = 1e6
        ens_periodic.vy_e[0] = 0.0
        ens_periodic.vz_e[0] = 0.0
        ens_periodic.active_e[0] = True

        ens_periodic.push_species(
            ens_periodic.x_e, ens_periodic.y_e, ens_periodic.z_e,
            ens_periodic.vx_e, ens_periodic.vy_e, ens_periodic.vz_e,
            ens_periodic.active_e, ens_periodic.was_outside_e,
            9.1e-31, -1.6e-19, -1e-15,
            field_solver, [], [], [], [],
            enable_collection=False,
            boundary_mode="periodic"
        )
        self.assertTrue(ens_periodic.active_e[0])
        self.assertGreaterEqual(ens_periodic.x_e[0], -params.L_x)
        self.assertLessEqual(ens_periodic.x_e[0], params.L_x)


if __name__ == "__main__":
    unittest.main()
