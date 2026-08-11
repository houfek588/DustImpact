import unittest
import os
import shutil
import tempfile
import numpy as np
from dust_impact.common.io import save_results_npz, load_results_npz, ensure_dir


class TestCommonIO(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.filepath = os.path.join(self.test_dir, "test_output.npz")

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_ensure_dir(self):
        nested_file = os.path.join(self.test_dir, "nested", "dir", "file.txt")
        ensure_dir(nested_file)
        self.assertTrue(os.path.exists(os.path.dirname(nested_file)))

    def test_save_load_npz_roundtrip(self):
        dummy_results = {
            'smooth_induced': [1.0, 2.0, 3.0],
            'smooth_collected': [0.1, 0.2, 0.3],
            'smooth_total': [1.1, 2.2, 3.3],
            'voltage_ant': [0.01, 0.02, 0.03],
            'history': {
                'V': [np.array([1, 2]), np.array([3, 4])],
                't': [0.0, 1e-6],
                'x_e': [np.array([0.1]), np.array([0.2])]
            }
        }

        save_results_npz(dummy_results, self.filepath)
        self.assertTrue(os.path.exists(self.filepath))

        loaded = load_results_npz(self.filepath)
        self.assertIn('smooth_induced', loaded)
        self.assertIn('history', loaded)
        self.assertIn('V', loaded['history'])
        np.testing.assert_array_almost_equal(loaded['smooth_induced'], [1.0, 2.0, 3.0])
        np.testing.assert_array_almost_equal(loaded['history']['t'], [0.0, 1e-6])


if __name__ == "__main__":
    unittest.main()
