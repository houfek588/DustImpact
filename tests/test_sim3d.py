import unittest
from dust_impact.sim3d.config_loader import SimulationParams3D, SimulationToggles3D, setup_simulation_parameters_3d
from dust_impact.sim3d.vtk_reader import load_and_interpolate_vtk
from dust_impact.sim3d.sim_core import DustImpactSimulation3D


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
        from dust_impact.sim3d.config_loader import VTKFilesConfig
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


if __name__ == "__main__":
    unittest.main()
