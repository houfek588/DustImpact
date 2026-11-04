# -*- coding: utf-8 -*-
"""
Unit tests for the dust_impact.visualization package.
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
    plot_currents_and_voltage,
    plot_monopole_dipole,
    plot_static_vtk_fields,
    animate_fields_slice,
    plot_potential_line_x,
    animate_particles_2d_slice,
    animate_particles_3d,
    animate_velocity_distribution,
)
from dust_impact.visualization.signals import compute_monopole_dipole, resolve_dipole_pairs
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

    def _four_antenna_results(self, floating: bool):
        """Four antennas with constant voltages 1..4 V; a floating spacecraft at 0.5 V if requested."""
        n_steps = len(self.params.time_array)
        res = dict(self.results)
        res['voltage_ant'] = np.array([np.full(n_steps, float(k + 1)) for k in range(4)])
        res['smooth_total'] = [np.zeros(n_steps)] * 4
        res['smooth_induced'] = [np.zeros(n_steps)] * 4
        if floating:
            res['voltage_sc'] = np.full(n_steps, 0.5)
            res['current_sc'] = np.zeros(n_steps)
        self.params.antenna_capacitance_F = [2e-12] * 4
        self.params.antenna_resistance_Ohm = [100e3] * 4
        return res

    def test_monopole_dipole_signals(self):
        """Monopoles are V_k - V_sc (floating history or fixed bias), dipoles V_i - V_j of the chosen pairs."""
        res = self._four_antenna_results(floating=True)
        sig = compute_monopole_dipole(res, self.params, [[3, 4]])
        self.assertTrue(sig['floating_spacecraft'])
        np.testing.assert_allclose(sig['monopoles'][:, 0], [0.5, 1.5, 2.5, 3.5])
        self.assertEqual(sig['pairs'], [(2, 3)])
        np.testing.assert_allclose(sig['dipoles'][0], -1.0)
        np.testing.assert_allclose(sig['t_us'][0], self.params.dt * 1e6)

        res_fixed = self._four_antenna_results(floating=False)
        res_fixed['metadata'] = {'spacecraft_bias_voltage_V': 2.0}
        sig_fixed = compute_monopole_dipole(res_fixed, self.params)
        self.assertFalse(sig_fixed['floating_spacecraft'])
        np.testing.assert_allclose(sig_fixed['monopoles'][:, -1], [-1.0, 0.0, 1.0, 2.0])
        self.assertEqual(sig_fixed['pairs'], [(0, 1), (2, 3)])

    def test_dipole_pair_validation(self):
        self.assertEqual(resolve_dipole_pairs([], 3), [(0, 1)])
        self.assertEqual(resolve_dipole_pairs(None, 1), [])
        with self.assertRaises(ValueError):
            resolve_dipole_pairs([[1, 5]], 4)
        with self.assertRaises(ValueError):
            resolve_dipole_pairs([[2, 2]], 4)

    def test_plot_monopole_dipole(self):
        res = self._four_antenna_results(floating=True)
        self.plot_cfg.dipole_pairs = [[3, 4], [1, 2]]
        fig = plot_monopole_dipole(res, self.params, self.plot_cfg)
        self.assertEqual(len(fig.axes), 2)
        self.assertEqual(len(fig.axes[0].get_lines()), 4 + 2)   # 4 monopoles + t_delay and zero lines
        self.assertEqual(len(fig.axes[1].get_lines()), 2 + 2)

        # One antenna: monopole only; disabled: no figure
        fig_single = plot_monopole_dipole(self.results, self.params, self.plot_cfg.__class__(
            show_interactive_gui_windows=False, save_plots_to_disk=False))
        self.assertEqual(len(fig_single.axes), 1)
        self.plot_cfg.show_monopole_dipole = False
        self.assertIsNone(plot_monopole_dipole(res, self.params, self.plot_cfg))

    def test_plot_currents_includes_floating_spacecraft(self):
        res = self._four_antenna_results(floating=True)
        fig = plot_currents_and_voltage(res, self.params, self.plot_cfg)
        labels = [l.get_label() for l in fig.axes[1].get_lines()]
        self.assertTrue(any(lbl.startswith('Sonda') for lbl in labels))

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

    def test_plot_potential_line_x(self):
        fig = plot_potential_line_x(self.results['history'], self.params, self.plot_cfg)
        self.assertIsNotNone(fig)
        self.assertEqual(len(fig.axes), 4)

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
        self.assertEqual(len(pipeline.figures), 7)
        self.assertEqual(len(pipeline.animations), 4)


if __name__ == "__main__":
    unittest.main()
