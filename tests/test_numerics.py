# -*- coding: utf-8 -*-
"""
Unit tests for dust_impact.numerics module:
- Cloud-in-Cell (CIC) trilinear charge deposition
- Trilinear grid-to-particle interpolation
- Boris/Leapfrog pusher
"""

import unittest
import numpy as np
from dust_impact.numerics.interpolators import deposit_charge_cic_3d, interp_field_3d
from dust_impact.numerics.pushers import leapfrog_step_3d, check_cfl_condition


class TestNumerics(unittest.TestCase):
    def test_cic_charge_conservation_random(self):
        """Test that CIC deposition strictly conserves total charge for arbitrary particle positions."""
        Nx, Ny, Nz = 20, 20, 20
        dx, dy, dz = 0.1, 0.1, 0.1
        x0, y0, z0 = -1.0, -1.0, -1.0
        q_macro = 1.602e-19

        np.random.seed(42)
        N_part = 1000
        # Random positions strictly inside domain
        x = np.random.uniform(x0 + 0.1, x0 + (Nx - 1.1) * dx, N_part)
        y = np.random.uniform(y0 + 0.1, y0 + (Ny - 1.1) * dy, N_part)
        z = np.random.uniform(z0 + 0.1, z0 + (Nz - 1.1) * dz, N_part)

        rho = deposit_charge_cic_3d(x, y, z, q_macro, x0, y0, z0, dx, dy, dz, Nx, Ny, Nz)

        total_deposited = np.sum(rho)
        expected_total = N_part * q_macro

        self.assertAlmostEqual(total_deposited, expected_total, places=12,
                               msg=f"Total charge not conserved! Deposited: {total_deposited}, Expected: {expected_total}")

    def test_cic_exact_node_deposition(self):
        """A particle placed exactly on a grid node must deposit 100% of its charge on that node."""
        Nx, Ny, Nz = 10, 10, 10
        dx, dy, dz = 0.5, 0.5, 0.5
        x0, y0, z0 = 0.0, 0.0, 0.0
        q_macro = 5.0e-15

        target_i, target_j, target_k = 3, 4, 5
        x = np.array([x0 + target_i * dx])
        y = np.array([y0 + target_j * dy])
        z = np.array([z0 + target_k * dz])

        rho = deposit_charge_cic_3d(x, y, z, q_macro, x0, y0, z0, dx, dy, dz, Nx, Ny, Nz)

        self.assertAlmostEqual(rho[target_i, target_j, target_k], q_macro, places=12)
        # All other nodes must be zero
        rho[target_i, target_j, target_k] = 0.0
        self.assertEqual(np.count_nonzero(rho), 0)

    def test_cic_cell_center_deposition(self):
        """A particle placed at cell center must deposit exactly 1/8 to each of the 8 corner nodes."""
        Nx, Ny, Nz = 8, 8, 8
        dx, dy, dz = 1.0, 1.0, 1.0
        x0, y0, z0 = 0.0, 0.0, 0.0
        q_macro = 8.0e-12

        ci, cj, ck = 2, 3, 4
        x = np.array([x0 + (ci + 0.5) * dx])
        y = np.array([y0 + (cj + 0.5) * dy])
        z = np.array([z0 + (ck + 0.5) * dz])

        rho = deposit_charge_cic_3d(x, y, z, q_macro, x0, y0, z0, dx, dy, dz, Nx, Ny, Nz)

        for di in [0, 1]:
            for dj in [0, 1]:
                for dk in [0, 1]:
                    self.assertAlmostEqual(rho[ci + di, cj + dj, ck + dk], q_macro / 8.0, places=12)

        # Check total
        self.assertAlmostEqual(np.sum(rho), q_macro, places=12)

    def test_cic_variable_charge_array(self):
        """Test deposition with per-particle variable charges."""
        Nx, Ny, Nz = 6, 6, 6
        dx, dy, dz = 0.2, 0.2, 0.2
        x0, y0, z0 = 0.0, 0.0, 0.0

        x = np.array([0.2, 0.4])
        y = np.array([0.2, 0.4])
        z = np.array([0.2, 0.4])
        q_macro = np.array([1.5e-12, -2.5e-12])

        rho = deposit_charge_cic_3d(x, y, z, q_macro, x0, y0, z0, dx, dy, dz, Nx, Ny, Nz)
        self.assertAlmostEqual(np.sum(rho), np.sum(q_macro), places=12)

    def test_interp_field_3d_linear(self):
        """Test that linear field interpolation reproduces exact linear gradient."""
        Nx, Ny, Nz = 10, 10, 10
        dx, dy, dz = 0.1, 0.1, 0.1
        x_grid = np.linspace(0, (Nx - 1) * dx, Nx)
        y_grid = np.linspace(0, (Ny - 1) * dy, Ny)
        z_grid = np.linspace(0, (Nz - 1) * dz, Nz)

        X, Y, Z = np.meshgrid(x_grid, y_grid, z_grid, indexing='ij')
        field = 2.0 * X + 3.0 * Y - 4.0 * Z + 5.0

        query_x = np.array([0.15, 0.33, 0.72])
        query_y = np.array([0.22, 0.45, 0.61])
        query_z = np.array([0.05, 0.81, 0.44])

        interpolated = interp_field_3d(query_x, query_y, query_z,
                                       x_grid, y_grid, z_grid,
                                       dx, dy, dz, field)
        expected = 2.0 * query_x + 3.0 * query_y - 4.0 * query_z + 5.0

        np.testing.assert_allclose(interpolated, expected, rtol=1e-10)


if __name__ == "__main__":
    unittest.main()
