# -*- coding: utf-8 -*-
"""
Unit tests for dust_impact.numerics module:
- Cloud-in-Cell (CIC) trilinear charge deposition
- Trilinear grid-to-particle interpolation
- Boris/Leapfrog pusher
- Laplace solver with eliminated Dirichlet nodes
"""

import unittest
import numpy as np
from dust_impact.numerics.interpolators import deposit_charge_cic_3d, interp_field_3d
from dust_impact.numerics.pushers import leapfrog_step_3d, check_cfl_condition
from dust_impact.numerics.poisson import DirichletLaplaceSolver3D
from dust_impact.numerics.gradient import ConductorGradient3D


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

    def test_rk4_circuit_exponential_response(self):
        """Test that RK4 circuit integrator matches analytical solution of RC ODE."""
        from dust_impact.solver.collector import AntennaCircuitCollector
        from dust_impact.solver.params import SimulationParams3D, SimulationToggles3D

        dt = 1e-9
        steps = 500
        R = 50e3
        C = 2e-12
        tau = R * C
        I0 = 1e-9

        params = SimulationParams3D(
            time_step_s=dt,
            num_time_steps=steps,
            grid_nodes_x=10,
            grid_nodes_y=10,
            grid_nodes_z=10
        )
        params.antenna_capacitance_F = [C]
        params.antenna_resistance_Ohm = [R]
        params.antenna_bias_voltage_V = [0.0]

        toggles = SimulationToggles3D(
            enable_rc_circuit_response=True,
            enable_antenna_bias_voltage=True,
            circuit_solver="rk4"
        )

        collector = AntennaCircuitCollector(params, toggles, num_antennas=1)
        for s in range(steps):
            collector.record_step_currents(
                s,
                col_e=np.array([I0]),
                col_i=np.array([0.0]),
                ind_e=np.array([0.0]),
                ind_i=np.array([0.0])
            )
            collector.update_circuit(s)

        t_final = (steps - 1) * dt
        v_exact = R * I0 * (1.0 - np.exp(-t_final / tau))
        v_sim = collector.voltage_ant[0, steps - 1]

        # RK4 should be extremely accurate
        self.assertAlmostEqual(v_sim, v_exact, places=8)

    def test_rk4_pure_capacitive_charging(self):
        """With RC circuit response disabled, V(t) should equal Q(t) / C (Simpson rule)."""
        from dust_impact.solver.collector import AntennaCircuitCollector
        from dust_impact.solver.params import SimulationParams3D, SimulationToggles3D

        dt = 2e-9
        steps = 50
        C = 1e-11
        I0 = 2e-9

        params = SimulationParams3D(
            time_step_s=dt,
            num_time_steps=steps,
            grid_nodes_x=10,
            grid_nodes_y=10,
            grid_nodes_z=10
        )
        params.antenna_capacitance_F = [C]
        params.antenna_resistance_Ohm = [1e6]
        params.antenna_bias_voltage_V = [0.0]

        toggles = SimulationToggles3D(
            enable_rc_circuit_response=False,
            enable_antenna_bias_voltage=False,
            circuit_solver="rk4"
        )

        collector = AntennaCircuitCollector(params, toggles, num_antennas=1)
        for s in range(steps):
            collector.record_step_currents(
                s,
                col_e=np.array([I0]),
                col_i=np.array([0.0]),
                ind_e=np.array([0.0]),
                ind_i=np.array([0.0])
            )
            collector.update_circuit(s)

        t_final = (steps - 1) * dt
        v_expected = (I0 * t_final) / C
        v_sim = collector.voltage_ant[0, steps - 1]
        self.assertAlmostEqual(v_sim, v_expected, places=12)


class TestDirichletLaplaceSolver(unittest.TestCase):
    """DirichletLaplaceSolver3D against fields that the 7-point stencil reproduces exactly."""

    def setUp(self):
        N = 21
        x = np.linspace(-1.0, 1.0, N)
        y = np.linspace(-1.5, 1.5, N)
        self.h = (x[1] - x[0], y[1] - y[0], x[1] - x[0])
        self.X, self.Y, self.Z = np.meshgrid(x, y, x, indexing='ij')
        self.fixed = np.zeros((N, N, N), dtype=bool)
        self.fixed[[0, -1], :, :] = True
        self.fixed[:, [0, -1], :] = True
        self.fixed[:, :, [0, -1]] = True
        # Interior conductor: a ball of radius 0.3 m
        self.fixed |= self.X**2 + self.Y**2 + self.Z**2 < 0.3**2
        self.solver = DirichletLaplaceSolver3D(self.fixed, self.h)

    def test_reproduces_discrete_harmonic_polynomial(self):
        """x^2 - z^2 + 3xyz has zero 7-point Laplacian on any grid, so it is the exact discrete solution."""
        V = self.X**2 - self.Z**2 + 3.0 * self.X * self.Y * self.Z
        np.testing.assert_allclose(self.solver.solve(V), V, atol=1e-7)

    def test_enforce_sets_dirichlet_values_and_keeps_laplacian(self):
        """The harmonic correction takes the target on fixed nodes and leaves the discrete Laplacian unchanged."""
        rng = np.random.default_rng(1)
        field = rng.normal(size=self.X.shape)
        target = rng.normal(size=self.X.shape)
        out = self.solver.enforce(field, target)
        np.testing.assert_allclose(out[self.fixed], target[self.fixed], atol=1e-14)

        def lap(V):
            hx, hy, hz = self.h
            return ((V[2:, 1:-1, 1:-1] - 2 * V[1:-1, 1:-1, 1:-1] + V[:-2, 1:-1, 1:-1]) / hx**2
                    + (V[1:-1, 2:, 1:-1] - 2 * V[1:-1, 1:-1, 1:-1] + V[1:-1, :-2, 1:-1]) / hy**2
                    + (V[1:-1, 1:-1, 2:] - 2 * V[1:-1, 1:-1, 1:-1] + V[1:-1, 1:-1, :-2]) / hz**2)

        free = ~self.fixed[1:-1, 1:-1, 1:-1]
        scale = np.max(np.abs(lap(field)[free]))
        np.testing.assert_allclose(lap(out)[free], lap(field)[free], atol=1e-6 * scale)

    def test_rejects_open_outer_boundary(self):
        """Boundary nodes have incomplete stencils, so they must be Dirichlet nodes."""
        fixed = self.fixed.copy()
        fixed[0, 5, 5] = False
        with self.assertRaises(ValueError):
            DirichletLaplaceSolver3D(fixed, self.h)


class TestConductorGradient(unittest.TestCase):
    """E = -grad(V) next to conductors must reproduce the potential drop felt by particles."""

    def test_one_sided_difference_on_conductor_surface(self):
        """A linear potential outside a conductor slab gives the exact field on the slab surface nodes,
        zero field in the slab interior, and the central difference elsewhere."""
        N, h = 12, 0.1
        x = np.arange(N) * h
        V = np.broadcast_to((3.0 - 2.0 * x)[:, None, None], (N, N, N)).copy()
        slab = np.zeros((N, N, N), dtype=bool)
        slab[:4] = True
        V[slab] = V[3, 0, 0]                       # equipotential conductor at the value of its surface node
        Ex, Ey, Ez = ConductorGradient3D(slab, (h, h, h)).electric_field(V)
        np.testing.assert_allclose(Ex[3], 2.0, rtol=1e-12)
        np.testing.assert_allclose(Ex[:3], 0.0, atol=1e-12)
        np.testing.assert_allclose(Ex[4:-1], 2.0, rtol=1e-12)
        np.testing.assert_allclose(Ey, 0.0, atol=1e-12)
        np.testing.assert_allclose(Ez, 0.0, atol=1e-12)

    def test_field_integral_matches_potential_drop_next_to_sphere(self):
        """Along a grid axis from just outside a voxelized sphere, the integral of the trilinearly
        interpolated E equals the drop of the interpolated V. Zeroing E on conductor nodes loses ~8 %
        of the drop in the first cell (R/h = 5), which biases charge escape from the spacecraft."""
        N, L, R, V0 = 61, 0.45, 0.075, 5.0
        g = np.linspace(-L, L, N)
        h = g[1] - g[0]
        X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
        r = np.sqrt(X**2 + Y**2 + Z**2)
        sphere = r <= R + 1e-12
        V = np.where(sphere, V0, V0 * R / np.maximum(r, 1e-12))
        Ex, Ey, Ez = ConductorGradient3D(sphere, (h, h, h)).electric_field(V)

        xs = np.linspace(R + 0.2 * h, 0.3, 4001)
        zeros = np.zeros_like(xs)
        E_line = interp_field_3d(xs, zeros, zeros, g, g, g, h, h, h, Ex)
        V_line = interp_field_3d(xs, zeros, zeros, g, g, g, h, h, h, V)
        drop = V_line[0] - V_line[-1]
        self.assertAlmostEqual(np.trapz(E_line, xs) / drop, 1.0, delta=0.01)


if __name__ == "__main__":
    unittest.main()
