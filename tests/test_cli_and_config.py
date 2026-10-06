# -*- coding: utf-8 -*-
import os
import tempfile
import unittest
from dust_impact.solver.config_loader import setup_simulation_parameters_3d
from dust_impact.main import main


class TestCLIAndConfig(unittest.TestCase):
    def test_output_dir_redirection_3d(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            params, toggles, plot_cfg = setup_simulation_parameters_3d(
                Vf=-5.0, Vf_antenne=1.0, config_file="config.json", output_dir=tmp_dir
            )
            self.assertTrue(plot_cfg.output_h5_filepath.startswith(os.path.abspath(tmp_dir)))
            self.assertTrue(plot_cfg.output_npz_filepath.startswith(os.path.abspath(tmp_dir)))
            self.assertTrue(plot_cfg.output_csv_filepath.startswith(os.path.abspath(tmp_dir)))
            self.assertTrue(plot_cfg.file_currents.startswith(os.path.abspath(tmp_dir)))
            self.assertTrue(plot_cfg.file_fields_anim.startswith(os.path.abspath(tmp_dir)))
            self.assertTrue(plot_cfg.vtk_output_dir.startswith(os.path.abspath(tmp_dir)))

    def test_template_config_loading(self):
        template_path = os.path.join("inputs", "config_template.json")
        self.assertTrue(os.path.exists(template_path))
        params, toggles, plot_cfg = setup_simulation_parameters_3d(
            Vf=-5.0, Vf_antenne=1.0, config_file=template_path
        )
        self.assertGreater(params.grid_nodes_x, 0)
        self.assertGreater(params.num_macroparticles, 0)
        self.assertEqual(plot_cfg.output_format, "h5")

    def test_zero_antennas_simulation_and_plotting(self):
        debye_config = os.path.join("inputs", "config_analytical_debye.json")
        self.assertTrue(os.path.exists(debye_config))
        with tempfile.TemporaryDirectory() as tmp_dir:
            params, toggles, plot_cfg = setup_simulation_parameters_3d(
                Vf=10.0, Vf_antenne=25.0, config_file=debye_config, output_dir=tmp_dir
            )
            self.assertEqual(len(params.C_ant), 0)
            self.assertEqual(len(params.R_ant), 0)
            self.assertEqual(len(params.V_bias), 0)

            # Test execution with minimal steps
            params.num_time_steps = 3
            params.steps = 3
            params.save_interval = 1
            params.num_macroparticles = 50
            params.N_particles = 50
            plot_cfg.show_interactive_gui_windows = False
            plot_cfg.save_plots_to_disk = False

            from dust_impact.geometry import build_simulation_geometry
            from dust_impact.solver.sim_core import DustImpactSimulation3D
            from dust_impact.solver.plotting import plot_simulation_results_3d

            geom = build_simulation_geometry(params)
            self.assertEqual(geom.num_antennas, 0)
            sim = DustImpactSimulation3D(params, toggles, geom)
            self.assertEqual(sim.num_antennas, 0)

            results = sim.run(stream_to_disk=False)
            self.assertEqual(len(results['smooth_total']), 0)
            # Plotting should not crash on 0 antennas
            plot_simulation_results_3d(results, params, plot_cfg, V_bg=geom.V_bg, Vw_grids=geom.Vw_grids)

    def test_plot_only_missing_file_graceful(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Should not raise exception even if data file is missing, but log error and return
            main(
                config_file="config.json",
                output_dir=tmp_dir,
                plot_only=True,
                visualize_results=False
            )


if __name__ == "__main__":
    unittest.main()
