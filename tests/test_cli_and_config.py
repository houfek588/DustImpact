# -*- coding: utf-8 -*-
import os
import tempfile
import unittest
from dust_impact.sim3d.config_loader import setup_simulation_parameters_3d
from dust_impact.sim2d.input_data import setup_simulation_parameters_2d
from dust_impact.main import main


class TestCLIAndConfig(unittest.TestCase):
    def test_output_dir_redirection_3d(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            params, toggles, plot_cfg = setup_simulation_parameters_3d(
                Vf=-5.0, Vf_antenne=1.0, config_file="config.json", output_dir=tmp_dir
            )
            self.assertTrue(plot_cfg.output_npz_filepath.startswith(os.path.abspath(tmp_dir)))
            self.assertTrue(plot_cfg.output_csv_filepath.startswith(os.path.abspath(tmp_dir)))
            self.assertTrue(plot_cfg.file_currents.startswith(os.path.abspath(tmp_dir)))
            self.assertTrue(plot_cfg.file_fields_anim.startswith(os.path.abspath(tmp_dir)))

    def test_output_dir_redirection_2d(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            params, toggles, plot_cfg = setup_simulation_parameters_2d(
                Vf=-5.0, Vf_antenne=1.0, config_file="config.json", output_dir=tmp_dir
            )
            self.assertTrue(plot_cfg.output_npz_filepath.startswith(os.path.abspath(tmp_dir)))
            self.assertTrue(plot_cfg.output_csv_filepath.startswith(os.path.abspath(tmp_dir)))
            self.assertTrue(plot_cfg.file_currents.startswith(os.path.abspath(tmp_dir)))
            self.assertTrue(plot_cfg.file_weighting.startswith(os.path.abspath(tmp_dir)))

    def test_plot_only_missing_file_graceful(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Should not raise exception even if NPZ is missing, but log error and return
            main(
                config_file="config.json",
                output_dir=tmp_dir,
                plot_only=True,
                dim_override=3,
                visualize_results=False
            )


if __name__ == "__main__":
    unittest.main()
