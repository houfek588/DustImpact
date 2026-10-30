"""
Charge-balance tests for the Ramo-Shockley antenna currents.

For a single macro-particle the time integral of (induced + collected) current on antenna a
must equal q * [Vw_a(r_end) - Vw_a(r_start)], where Vw_a(r_end) = 1 when the particle is
collected by antenna a and 0 when it is absorbed by another conductor.
"""

import unittest
import numpy as np

from dust_impact.physics.constants import e, m_e
from dust_impact.physics.ramo_shockley import calc_induced_current_from_potential
from dust_impact.solver.config_loader import SimulationParams3D, SimulationToggles3D
from dust_impact.solver.ensemble import ParticleEnsemble
from dust_impact.solver.field_solver import FieldSolver3D


class TestChargeBalance(unittest.TestCase):
    def setUp(self):
        self.N = 32
        self.params = SimulationParams3D(
            num_macroparticles=1,
            simulation_duration_s=1e-8,
            time_step_s=1e-9,
            grid_nodes_x=self.N,
            grid_nodes_y=self.N,
            grid_nodes_z=self.N,
            domain_half_length_x_m=3.0,
            domain_half_length_y_m=3.0,
            domain_half_length_z_m=3.0,
        )
        p = self.params
        X, Y, Z = np.meshgrid(p.x_grid, p.y_grid, p.z_grid, indexing='ij')
        r = np.sqrt(X**2 + Y**2 + Z**2)

        # Spherical antenna of radius R at the origin: vacuum weighting potential Vw = R / r
        self.R = 0.5
        self.ant_mask = r <= self.R
        self.Vw = np.where(self.ant_mask, 1.0, self.R / np.maximum(r, 1e-12))
        self.sc_mask = np.zeros_like(self.ant_mask)

        zeros = np.zeros((self.N, self.N, self.N))
        self.toggles = SimulationToggles3D(
            enable_spis_background_field=False,
            enable_plasma_self_field=False,
            particle_boundary_mode="open",
        )
        self.field_solver = FieldSolver3D(
            p, self.toggles, zeros, zeros, zeros, zeros,
            [zeros], [zeros], [zeros], self.ant_mask, num_antennas=1
        )

    def _make_ensemble(self, x0, v0):
        """Creates a single active ion at x0 with velocity v0 (no fields act on it)."""
        ens = ParticleEnsemble(self.params, self.sc_mask, num_antennas=1, toggles=self.toggles)
        ens.cloud_injected = True
        ens.x_i[0], ens.y_i[0], ens.z_i[0] = x0
        ens.vx_i[0], ens.vy_i[0], ens.vz_i[0] = v0
        ens.active_i[0] = True
        ens.was_outside_i[:, 0] = True
        return ens

    def _integrate(self, ens, q, max_steps=200):
        """Pushes the ion until it is removed; returns integrated induced and collected charge."""
        Q_ind, Q_col = 0.0, 0.0
        zeros = np.zeros((self.N, self.N, self.N))
        for _ in range(max_steps):
            if not ens.active_i[0]:
                break
            col, ind = ens.push_species(
                ens.x_i, ens.y_i, ens.z_i, ens.vx_i, ens.vy_i, ens.vz_i,
                ens.active_i, ens.was_outside_i,
                self.params.m_i, e, q,
                self.field_solver, [self.ant_mask], [zeros], [zeros], [zeros],
                enable_collection=True, boundary_mode="open",
                Vw_list=[self.Vw],
            )
            Q_ind += ind[0] * self.params.dt
            Q_col += col[0] * self.params.dt
        return Q_ind, Q_col

    def test_collected_particle_delivers_exactly_its_charge(self):
        """A charge moving from r0 onto the antenna delivers q * (1 - Vw(r0)), not q * (1 + Vw)."""
        q = 1e-15
        x0 = (2.5, 0.1, 0.05)
        vw0 = self.field_solver.interp_field(
            np.array([x0[0]]), np.array([x0[1]]), np.array([x0[2]]), self.Vw
        )[0]
        ens = self._make_ensemble(x0, (-1e8, 0.0, 0.0))
        Q_ind, Q_col = self._integrate(ens, q)

        self.assertFalse(ens.active_i[0], "Ion should have been collected by the antenna")
        self.assertGreater(Q_col, 0.0)
        self.assertAlmostEqual((Q_ind + Q_col) / q, 1.0 - vw0, places=10)

    def test_passing_particle_leaves_no_net_charge(self):
        """A charge flying past the antenna and leaving the domain delivers q * [Vw(end) - Vw(start)] ~ 0."""
        q = 1e-15
        x0 = (-2.9, 1.5, 0.0)
        vw0 = self.field_solver.interp_field(
            np.array([x0[0]]), np.array([x0[1]]), np.array([x0[2]]), self.Vw
        )[0]
        ens = self._make_ensemble(x0, (1e8, 0.0, 0.0))
        Q_ind, Q_col = self._integrate(ens, q)

        self.assertFalse(ens.active_i[0], "Ion should have left the domain")
        self.assertEqual(Q_col, 0.0)
        vw_end = self.field_solver.interp_field(
            np.array([ens.x_i[0]]), np.array([ens.y_i[0]]), np.array([ens.z_i[0]]), self.Vw
        )[0]
        self.assertAlmostEqual(Q_ind / q, vw_end - vw0, places=10)
        self.assertLess(abs(Q_ind / q), 0.1)

    def test_electron_sign(self):
        """An electron collected from far away delivers a negative charge of magnitude e * (1 - Vw(r0))."""
        vw_old = np.array([0.2])
        vw_new = np.array([1.0])
        dt = 1e-9
        I = calc_induced_current_from_potential(-e, vw_old, vw_new, dt)
        self.assertAlmostEqual(I * dt / e, -0.8, places=12)

    def test_push_species_requires_weighting_potentials(self):
        ens = self._make_ensemble((2.5, 0.0, 0.0), (-1e8, 0.0, 0.0))
        zeros = np.zeros((self.N, self.N, self.N))
        with self.assertRaises(ValueError):
            ens.push_species(
                ens.x_i, ens.y_i, ens.z_i, ens.vx_i, ens.vy_i, ens.vz_i,
                ens.active_i, ens.was_outside_i,
                m_e, -e, -1e-15,
                self.field_solver, [self.ant_mask], [zeros], [zeros], [zeros],
            )


if __name__ == "__main__":
    unittest.main()
