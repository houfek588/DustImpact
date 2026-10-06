# -*- coding: utf-8 -*-
"""
Unit tests for dust_impact.visualization package and its backward compatibility facades.
"""

import os
import tempfile
import unittest
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Headless test runner
import matplotlib.pyplot as plt

from dust_impact.solver.config_loader import SimulationParams3D, PlottingConfig3D
from dust_impact.common.io import export_time_series_csv
from dust_impact.visualization import (
    VisualizationPipeline,
    plot_simulation_results_3d,
    plot_currents_and_voltage,
    plot_static_vtk_fields,
    animate_fields_slice,
    animate_particles_2d_slice,
    animate_particles_3d,
    animate_velocity_distribution,
    export_simulation_to_paraview,
)
from dust_impact.visualization.styles import safe_set_window_title, get_antenna_colors, get_species_colors, cleanup_animation


class TestVisualization(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.params = SimulationParams3D(
            Vf=10.0,
            num_macroparticles=20,
            simulation_duration_s=1e-7,
            time_step_s=2e-8,
            grid_nodes_x=10,
            grid_nodes_y=10,
            grid_nodes_z=10,
        )
        self.plot_cfg = PlottingConfig3D(
            show_interactive_gui_windows=False,
            save_plots_to_disk=False,
            export_csv_time_series=False,
            file_currents=os.path.join(self.tmp_dir, "currents.png"),
            file_fields_anim=os.path.join(self.tmp_dir, "fields.gif"),
            file_particles_anim=os.path.join(self.tmp_dir, "particles.gif"),
            file_velocity_anim=os.path.join(self.tmp_dir, "velocity.gif"),
        )
        n_steps = len(self.params.time_array)
        n_pts = 20
        self.results = {
            'smooth_total': [np.linspace(1e-9, 0, n_steps)],
            'smooth_induced': [np.linspace(0.5e-9, 0, n_steps)],
            'smooth_collected': [np.linspace(0.5e-9, 0, n_steps)],
            'voltage_ant': [np.linspace(5.0, 0, n_steps)],
            'history': {
                't': list(self.params.time_array[:3]),
                'V': [np.zeros((10, 10, 10)) for _ in range(3)],
                'rho': [np.zeros((10, 10, 10)) for _ in range(3)],
                'x_e': [np.linspace(-1, 1, n_pts) for _ in range(3)],
                'y_e': [np.linspace(-1, 1, n_pts) for _ in range(3)],
                'z_e': [np.linspace(-1, 1, n_pts) for _ in range(3)],
                'vx_e': [np.ones(n_pts) * 1e5 for _ in range(3)],
                'vy_e': [np.zeros(n_pts) for _ in range(3)],
                'vz_e': [np.zeros(n_pts) for _ in range(3)],
                'x_i': [np.linspace(-0.5, 0.5, n_pts) for _ in range(3)],
                'y_i': [np.linspace(-0.5, 0.5, n_pts) for _ in range(3)],
                'z_i': [np.linspace(-0.5, 0.5, n_pts) for _ in range(3)],
                'vx_i': [np.ones(n_pts) * 1e3 for _ in range(3)],
                'vy_i': [np.zeros(n_pts) for _ in range(3)],
                'vz_i': [np.zeros(n_pts) for _ in range(3)],
            }
        }
        self.params.antenna_capacitance_F = [2e-12]
        self.params.antenna_resistance_Ohm = [100e3]

    def tearDown(self):
        plt.close('all')
        import shutil
        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_styles(self):
        fig, ax = plt.subplots()
        safe_set_window_title(fig, "Test Window")
        colors = get_antenna_colors()
        self.assertGreaterEqual(len(colors), 3)
        sp = get_species_colors()
        self.assertIn('e', sp)
        self.assertIn('i', sp)
        plt.close(fig)

    def test_plot_currents_and_voltage(self):
        fig = plot_currents_and_voltage(self.results, self.params, self.plot_cfg)
        self.assertIsNotNone(fig)
        self.assertEqual(len(fig.axes), 2)

        # Test zero antennas skips gracefully
        self.params.antenna_capacitance_F = []
        fig_zero = plot_currents_and_voltage(self.results, self.params, self.plot_cfg)
        self.assertIsNone(fig_zero)

    def test_plot_static_vtk_fields(self):
        V_bg = np.zeros((10, 10, 10))
        Vw_grids = [np.ones((10, 10, 10))]
        fig = plot_static_vtk_fields(self.params, V_bg, Vw_grids, self.plot_cfg)
        self.assertIsNotNone(fig)
        self.assertGreaterEqual(len(fig.axes), 2)

    def test_animate_fields_slice(self):
        fig, anim = animate_fields_slice(self.results['history'], self.params, self.plot_cfg)
        self.assertIsNotNone(fig)
        self.assertIsNotNone(anim)
        cleanup_animation(anim)

    def test_animate_particles_2d_and_3d(self):
        fig2d, anim2d = animate_particles_2d_slice(self.results['history'], self.params, self.plot_cfg)
        self.assertIsNotNone(fig2d)
        self.assertIsNotNone(anim2d)
        cleanup_animation(anim2d)

        fig3d, anim3d = animate_particles_3d(self.results['history'], self.params, self.plot_cfg)
        self.assertIsNotNone(fig3d)
        self.assertIsNotNone(anim3d)
        cleanup_animation(anim3d)

    def test_animate_velocity_distribution(self):
        fig, anim = animate_velocity_distribution(self.results['history'], self.plot_cfg)
        self.assertIsNotNone(fig)
        self.assertIsNotNone(anim)
        cleanup_animation(anim)

    def test_csv_export(self):
        csv_path = os.path.join(self.tmp_dir, "test_signals.csv")
        out = export_time_series_csv(self.results, self.params, csv_path)
        self.assertTrue(os.path.exists(out))
        with open(out, 'r') as f:
            lines = f.readlines()
        self.assertGreater(len(lines), 1)
        self.assertIn("Time_s", lines[0])
        self.assertIn("I_ind_1_A", lines[0])

    def test_pipeline_run_headless(self):
        pipeline = VisualizationPipeline()
        pipeline.run(self.results, self.params, self.plot_cfg)
        self.assertEqual(len(pipeline.figures), 5)
        self.assertEqual(len(pipeline.animations), 4)

    def test_backward_compatibility_facades(self):
        # 1. solver.plotting facade
        import dust_impact.solver.plotting as legacy_plot
        self.assertTrue(callable(legacy_plot.plot_simulation_results_3d))
        self.assertTrue(callable(legacy_plot._plot_currents_and_voltage))
        self.assertTrue(callable(legacy_plot._plot_static_vtk_fields))
        self.assertTrue(callable(legacy_plot._animate_yz_fields_slice))
        self.assertTrue(callable(legacy_plot._export_csv))

        # 2. common.vtk_export facade
        import dust_impact.common.vtk_export as legacy_vtk
        self.assertTrue(callable(legacy_vtk.export_simulation_to_paraview))


if __name__ == "__main__":
    unittest.main()
