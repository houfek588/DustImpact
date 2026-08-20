import unittest
from dust_impact.sim2d.input_data import SimulationParams2D, SimulationToggles2D, setup_simulation_parameters_2d
from dust_impact.sim2d.sim_core import DustImpactSimulation2D


class TestSim2D(unittest.TestCase):
    def test_params_initialization(self):
        params, toggles, plot_cfg = setup_simulation_parameters_2d(Vf=-5.0, Vf_antenne=1.0)
        self.assertEqual(params.Vf, -5.0)
        self.assertGreater(params.Nx, 0)
        self.assertGreater(params.Ny, 0)

    def test_sim_core_2d_initialization(self):
        params = SimulationParams2D(
            Vf=-5.0,
            num_macroparticles=50,
            simulation_duration_s=1e-8,
            time_step_s=1e-9,
            grid_nodes_x=10,
            grid_nodes_y=10,
            antennas=[
                {"x1": 1.0, "y1": 0.1, "x2": 1.0, "y2": 0.5, "r": 0.05, "V_bias": 0.0, "w_width": 0.2, "C": 1e-12, "R": 10e3}
            ]
        )
        toggles = SimulationToggles2D()
        sim = DustImpactSimulation2D(params, toggles)
        self.assertEqual(sim.num_antennas, 1)

        # Quick step run test
    def test_sim_core_2d_homogeneous_mode(self):
        params = SimulationParams2D(
            Vf=-5.0,
            num_macroparticles=100,
            simulation_duration_s=1e-8,
            time_step_s=1e-9,
            grid_nodes_x=10,
            grid_nodes_y=10,
            plasma_injection_mode="homogeneous",
            antennas=[
                {"x1": 1.0, "y1": 0.1, "x2": 1.0, "y2": 0.5, "r": 0.05, "V_bias": 0.0, "w_width": 0.2, "C": 1e-12, "R": 10e3}
            ]
        )
        toggles = SimulationToggles2D()
        sim = DustImpactSimulation2D(params, toggles)
        self.assertEqual(len(sim.x_e), 100)
        self.assertGreater(sim.x_e.max() - sim.x_e.min(), 0.1)

        res = sim.run()
        self.assertIn('smooth_total', res)
        self.assertIn('voltage_ant', res)


if __name__ == "__main__":
    unittest.main()
