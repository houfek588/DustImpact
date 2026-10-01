# -*- coding: utf-8 -*-
"""
Unit tests for modernized simulation I/O:
1. NPZ with embedded metadata and coordinate grids.
2. HDF5 chunked storage and loading.
3. Periodic rotating checkpointing.
4. ParaView VTK/VTI/VTP export and PVD collections.
5. Unified auto-detecting save_results / load_results.
"""

import os
import shutil
import tempfile
import unittest
import numpy as np

from dust_impact.common.io import (
    save_results_npz, load_results_npz,
    save_results_h5, load_results_h5,
    save_results, load_results,
    save_checkpoint, HAS_H5PY
)
from dust_impact.common.vtk_export import export_simulation_to_paraview, HAS_PYVISTA
from dust_impact.main import main


class TestIOFormats(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="test_dust_io_")
        self.num_antennas = 2
        self.num_steps = 50
        self.grid_shape = (8, 8, 8)

        # Create dummy synthetic simulation results
        t_arr = np.linspace(0, 1e-6, self.num_steps)
        self.sample_results = {
            'smooth_induced': [np.sin(t_arr * 1e7), np.cos(t_arr * 1e7)],
            'smooth_collected': [np.zeros(self.num_steps), np.zeros(self.num_steps)],
            'smooth_total': [np.sin(t_arr * 1e7), np.cos(t_arr * 1e7)],
            'voltage_ant': np.zeros((self.num_antennas, self.num_steps)),
            'history': {
                't': [0.0, 5e-7, 1e-6],
                'V': [np.ones(self.grid_shape) * 1.5, np.ones(self.grid_shape) * 2.0, np.ones(self.grid_shape) * 2.5],
                'rho': [np.zeros(self.grid_shape), np.zeros(self.grid_shape), np.zeros(self.grid_shape)],
                'x_e': [np.array([0.1, 0.2]), np.array([0.3, 0.4]), np.array([0.5, 0.6])],
                'y_e': [np.array([0.0, 0.0]), np.array([0.1, 0.1]), np.array([0.2, 0.2])],
                'z_e': [np.array([0.0, 0.0]), np.array([0.0, 0.0]), np.array([0.0, 0.0])],
                'vx_e': [np.array([1e5, 1e5]), np.array([1e5, 1e5]), np.array([1e5, 1e5])],
                'vy_e': [np.array([0.0, 0.0]), np.array([0.0, 0.0]), np.array([0.0, 0.0])],
                'vz_e': [np.array([0.0, 0.0]), np.array([0.0, 0.0]), np.array([0.0, 0.0])],
                'x_i': [np.array([0.1]), np.array([0.15]), np.array([0.2])],
                'y_i': [np.array([0.0]), np.array([0.0]), np.array([0.0])],
                'z_i': [np.array([0.0]), np.array([0.0]), np.array([0.0])],
                'vx_i': [np.array([1e3]), np.array([1e3]), np.array([1e3])],
                'vy_i': [np.array([0.0]), np.array([0.0]), np.array([0.0])],
                'vz_i': [np.array([0.0]), np.array([0.0]), np.array([0.0])]
            }
        }
        self.sample_metadata = {
            'package_version': '0.3.0',
            'dt': 2e-8,
            'dx': 0.2,
            'dy': 0.2,
            'dz': 0.2,
            'L_x': 0.8,
            'L_y': 0.8,
            'L_z': 0.8,
            'x_grid': np.linspace(-0.8, 0.8, 8).tolist(),
            'y_grid': np.linspace(-0.8, 0.8, 8).tolist(),
            'z_grid': np.linspace(-0.8, 0.8, 8).tolist(),
            'time_array': t_arr.tolist(),
            'num_antennas': 2,
            'total_impact_charge_C': 5e-11
        }

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_npz_with_metadata_roundtrip(self):
        """Test saving and loading NPZ archive with embedded JSON metadata and coordinate axes."""
        npz_file = os.path.join(self.test_dir, "test_results.npz")
        save_results_npz(self.sample_results, npz_file, metadata=self.sample_metadata)

        self.assertTrue(os.path.exists(npz_file))
        loaded = load_results_npz(npz_file)

        # Check signals
        self.assertEqual(len(loaded['smooth_total']), 2)
        np.testing.assert_allclose(loaded['smooth_total'][0], self.sample_results['smooth_total'][0])

        # Check metadata restoration
        self.assertIn('metadata', loaded)
        self.assertEqual(loaded['metadata']['package_version'], '0.3.0')
        self.assertEqual(loaded['metadata']['total_impact_charge_C'], 5e-11)
        self.assertIn('x_grid', loaded['metadata'])

        # Check history fields
        self.assertEqual(len(loaded['history']['V']), 3)
        np.testing.assert_allclose(loaded['history']['V'][0], self.sample_results['history']['V'][0])

    @unittest.skipUnless(HAS_H5PY, "h5py is not installed")
    def test_h5_roundtrip(self):
        """Test saving and loading HDF5 archive with chunked fields and signals."""
        h5_file = os.path.join(self.test_dir, "test_results.h5")
        save_results_h5(self.sample_results, h5_file, metadata=self.sample_metadata)

        self.assertTrue(os.path.exists(h5_file))
        loaded = load_results_h5(h5_file)

        # Check signals
        self.assertEqual(len(loaded['smooth_total']), 2)
        np.testing.assert_allclose(loaded['smooth_total'][0], self.sample_results['smooth_total'][0], rtol=1e-5)

        # Check metadata
        self.assertIn('metadata', loaded)
        self.assertEqual(loaded['metadata']['package_version'], '0.3.0')

        # Check fields
        self.assertEqual(len(loaded['history']['V']), 3)
        np.testing.assert_allclose(loaded['history']['V'][0], self.sample_results['history']['V'][0])
        self.assertEqual(len(loaded['history']['t']), 3)

    def test_checkpointing(self):
        """Test atomic rotating checkpoint creation."""
        base_file = os.path.join(self.test_dir, "run_state.h5" if HAS_H5PY else "run_state.npz")
        cp_path = save_checkpoint(self.sample_results, base_file, metadata=self.sample_metadata)

        self.assertTrue(os.path.exists(cp_path))
        self.assertTrue(cp_path.endswith(".checkpoint.h5") or cp_path.endswith(".checkpoint.npz"))

        # Verify load from checkpoint
        loaded = load_results(cp_path)
        self.assertIn('smooth_total', loaded)
        self.assertEqual(len(loaded['history']['t']), 3)

    def test_unified_save_and_load_dispatcher(self):
        """Test unified auto-detection dispatcher for both .h5 and .npz."""
        npz_file = os.path.join(self.test_dir, "auto.npz")
        save_results(self.sample_results, npz_file, metadata=self.sample_metadata, format_type="npz")
        loaded_npz = load_results(npz_file)
        self.assertIn('history', loaded_npz)

        if HAS_H5PY:
            h5_file = os.path.join(self.test_dir, "auto.h5")
            save_results(self.sample_results, h5_file, metadata=self.sample_metadata, format_type="h5")
            loaded_h5 = load_results(h5_file)
            self.assertIn('history', loaded_h5)

    @unittest.skipUnless(HAS_PYVISTA, "pyvista is not installed")
    def test_paraview_vtk_export(self):
        """Test generating ParaView ImageData (.vti), PolyData (.vtp) and XML collections (.pvd)."""
        vtk_dir = os.path.join(self.test_dir, "paraview")
        results = dict(self.sample_results)
        results['metadata'] = self.sample_metadata

        pvd_map = export_simulation_to_paraview(results, params=None, output_dir=vtk_dir)

        self.assertIn('fields', pvd_map)
        self.assertTrue(os.path.exists(pvd_map['fields']))
        self.assertTrue(os.path.exists(os.path.join(vtk_dir, "fields_t0000.vti")))

        self.assertIn('electrons', pvd_map)
        self.assertTrue(os.path.exists(pvd_map['electrons']))
        self.assertTrue(os.path.exists(os.path.join(vtk_dir, "particles_e_t0000.vtp")))

        self.assertIn('ions', pvd_map)
        self.assertTrue(os.path.exists(pvd_map['ions']))
        self.assertTrue(os.path.exists(os.path.join(vtk_dir, "particles_i_t0000.vtp")))

        # Check XML structure of .pvd file
        with open(pvd_map['fields'], 'r', encoding='utf-8') as f:
            content = f.read()
        self.assertIn('<VTKFile type="Collection"', content)
        self.assertIn('fields_t0000.vti', content)


if __name__ == "__main__":
    unittest.main()
