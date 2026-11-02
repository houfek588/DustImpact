# -*- coding: utf-8 -*-
"""
Unit tests for weighting fields memory optimization (float32 vs float64)
and sub-grid thin-wire continuous collision detection.
"""

import contextlib
import io
import unittest
from types import SimpleNamespace

import numpy as np

from dust_impact.physics.constants import e
from dust_impact.solver.config_loader import (
    setup_simulation_parameters_3d, SimulationParams3D, SimulationToggles3D,
)
from dust_impact.solver.ensemble import ParticleEnsemble
from dust_impact.solver.field_solver import FieldSolver3D
from dust_impact.geometry.prepared import build_simulation_geometry
from dust_impact.geometry.analytical import build_analytical_simulation_geometry
from dust_impact.geometry.surface import check_wire_collision_3d
from dust_impact.solver.sim_core import DustImpactSimulation3D


class TestMemoryAndWireModel(unittest.TestCase):

    def test_float32_precision_and_memory_reduction(self):
        """Verifies that field_precision='float32' cuts memory usage of static grids by 50%."""
        params_f32, toggles_f32, _ = setup_simulation_parameters_3d(
            Vf=-10.0, Vf_antenne=2.0, config_file="inputs/config_analytical_sphere.json"
        )
        params_f32.field_precision = "float32"
        geom_f32 = build_analytical_simulation_geometry(params_f32, params_f32.geometry.analytical)

        params_f64, toggles_f64, _ = setup_simulation_parameters_3d(
            Vf=-10.0, Vf_antenne=2.0, config_file="inputs/config_analytical_sphere.json"
        )
        params_f64.field_precision = "float64"
        geom_f64 = build_analytical_simulation_geometry(params_f64, params_f64.geometry.analytical)

        # 1. Check dtypes
        self.assertEqual(geom_f32.V_bg.dtype, np.float32)
        self.assertEqual(geom_f32.Ex_bg.dtype, np.float32)
        self.assertEqual(geom_f32.Vw_grids[0].dtype, np.float32)
        self.assertEqual(geom_f32.Ewx_list[0].dtype, np.float32)

        self.assertEqual(geom_f64.V_bg.dtype, np.float64)
        self.assertEqual(geom_f64.Ex_bg.dtype, np.float64)
        self.assertEqual(geom_f64.Vw_grids[0].dtype, np.float64)
        self.assertEqual(geom_f64.Ewx_list[0].dtype, np.float64)

        # 2. Check 50% memory footprint reduction
        self.assertEqual(geom_f32.V_bg.nbytes * 2, geom_f64.V_bg.nbytes)
        self.assertEqual(geom_f32.Ex_bg.nbytes * 2, geom_f64.Ex_bg.nbytes)
        self.assertEqual(geom_f32.Ewx_list[0].nbytes * 2, geom_f64.Ewx_list[0].nbytes)

    def test_float32_vs_float64_physical_signal_agreement(self):
        """Verifies that float32 static fields yield results matching float64 to within 0.1%."""
        params_f32, toggles_f32, _ = setup_simulation_parameters_3d(
            Vf=-5.0, Vf_antenne=1.0, config_file="inputs/config_analytical_sphere.json"
        )
        params_f32.field_precision = "float32"
        params_f32.steps = 15
        params_f32.N_particles = 1000

        params_f64, toggles_f64, _ = setup_simulation_parameters_3d(
            Vf=-5.0, Vf_antenne=1.0, config_file="inputs/config_analytical_sphere.json"
        )
        params_f64.field_precision = "float64"
        params_f64.steps = 15
        params_f64.N_particles = 1000

        geom_f32 = build_analytical_simulation_geometry(params_f32, params_f32.geometry.analytical)
        geom_f64 = build_analytical_simulation_geometry(params_f64, params_f64.geometry.analytical)

        # Run identical simulations
        np.random.seed(42)
        sim_f32 = DustImpactSimulation3D(params_f32, toggles_f32, geom_f32)
        res_f32 = sim_f32.run()

        np.random.seed(42)
        sim_f64 = DustImpactSimulation3D(params_f64, toggles_f64, geom_f64)
        res_f64 = sim_f64.run()

        # Induced currents should match closely (within float32 precision limits)
        curr_f32 = np.asarray(res_f32['smooth_induced'])
        curr_f64 = np.asarray(res_f64['smooth_induced'])

        max_val = np.max(np.abs(curr_f64))
        if max_val > 1e-12:
            rel_diff = np.max(np.abs(curr_f32 - curr_f64)) / max_val
            self.assertLess(rel_diff, 1e-3, f"Relative difference {rel_diff:.4e} exceeds 0.1%")

    def test_check_wire_collision_3d_accuracy(self):
        """Verifies exact sub-grid orthogonal distance calculation for wire collision."""
        p_start = [0.0, 0.0, 0.0]
        p_end = [2.0, 0.0, 0.0]
        wire_r = 0.02  # 2 cm wire

        # Test points:
        # Point 1: inside cylinder (x=1.0, y=0.01, z=0.0) -> dist = 0.01 <= 0.02 -> True
        # Point 2: outside cylinder (x=1.0, y=0.05, z=0.0) -> dist = 0.05 > 0.02 -> False
        # Point 3: beyond start endcap (x=-0.1, y=0.0, z=0.0) -> clamped to start -> dist = 0.1 > 0.02 -> False
        # Point 4: beyond end endcap (x=2.01, y=0.0, z=0.0) -> clamped to end -> dist = 0.01 <= 0.02 -> True
        x = np.array([1.0, 1.0, -0.1, 2.01])
        y = np.array([0.01, 0.05, 0.0, 0.0])
        z = np.array([0.0, 0.0, 0.0, 0.0])

        hits = check_wire_collision_3d(x, y, z, p_start, p_end, wire_r)
        self.assertTrue(hits[0])
        self.assertFalse(hits[1])
        self.assertFalse(hits[2])
        self.assertTrue(hits[3])

    def test_analytical_geometry_passes_wire_model_to_ensemble(self):
        """The analytical antennas reach the particle ensemble as exact wire geometries."""
        with contextlib.redirect_stdout(io.StringIO()):
            params, toggles, _ = setup_simulation_parameters_3d(
                Vf=-10.0, Vf_antenne=0.0, config_file="inputs/config_analytical_sphere.json"
            )
            geom = build_analytical_simulation_geometry(params, params.geometry.analytical)
            sim = DustImpactSimulation3D(params, toggles, geom)
        self.assertEqual(sim.particles.num_antennas, geom.num_antennas)
        for ant in sim.particles.antenna_geometries:
            self.assertTrue(hasattr(ant, 'p_start') and hasattr(ant, 'p_end') and hasattr(ant, 'radius'))
            # The wire is thinner than a cell, so the grid mask alone could not resolve it
            self.assertLess(2.0 * ant.radius, params.dx)


class TestSubgridWireCollection(unittest.TestCase):
    """
    Collection of single ions by a thin wire antenna (radius << dx) placed between grid nodes.
    The charge delivered to the antenna must be q * (1 - Vw(r_0)) (Ramo-Shockley with collection).
    """

    def setUp(self):
        N = 21
        self.params = SimulationParams3D(
            num_macroparticles=1, simulation_duration_s=1e-5, time_step_s=1e-8,
            grid_nodes_x=N, grid_nodes_y=N, grid_nodes_z=N,
            domain_half_length_x_m=1.0, domain_half_length_y_m=1.0, domain_half_length_z_m=1.0,
        )
        p = self.params
        # Wire along z through (x, y) = (0.05, 0.05), i.e. in the middle of a cell (dx = 0.1 m)
        self.wire = SimpleNamespace(p_start=[0.05, 0.05, -0.5], p_end=[0.05, 0.05, 0.5], radius=0.01)
        X, Y, Z = np.meshgrid(p.x_grid, p.y_grid, p.z_grid, indexing='ij')
        # Smooth stand-in weighting potential; the charge identity holds for any Vw
        self.Vw = np.exp(-((X - 0.05)**2 + (Y - 0.05)**2) / 0.1)
        self.mask = np.zeros((N, N, N), dtype=bool)
        zeros = np.zeros((N, N, N))
        self.toggles = SimulationToggles3D(
            enable_spis_background_field=False, enable_plasma_self_field=False, particle_boundary_mode="open"
        )
        with contextlib.redirect_stdout(io.StringIO()):
            self.fs = FieldSolver3D(p, self.toggles, zeros, zeros, zeros, zeros,
                                    [zeros], [zeros], [zeros], self.mask, num_antennas=1)

    def _fly(self, x0, v0, max_steps=1000):
        """Moves one force-free ion from x0 with velocity v0; returns (ensemble, delivered charge / q)."""
        p, q = self.params, 1e-15
        zeros = np.zeros_like(self.Vw)
        ens = ParticleEnsemble(p, self.mask, num_antennas=1, antenna_geometries=[self.wire], toggles=self.toggles)
        ens.cloud_injected = True
        ens.x_i[0], ens.y_i[0], ens.z_i[0] = x0
        ens.vx_i[0], ens.vy_i[0], ens.vz_i[0] = v0
        ens.active_i[0] = True
        ens.was_outside_i[:, 0] = True
        Q = 0.0
        for _ in range(max_steps):
            if not ens.active_i[0]:
                break
            col, ind = ens.push_species(
                ens.x_i, ens.y_i, ens.z_i, ens.vx_i, ens.vy_i, ens.vz_i,
                ens.active_i, ens.was_outside_i, p.m_i, e, q,
                self.fs, [self.mask], [zeros], [zeros], [zeros],
                enable_collection=True, boundary_mode="open", Vw_list=[self.Vw],
            )
            Q += (col[0] + ind[0]) * p.dt
        return ens, Q / q

    def _vw0(self, x0):
        return self.fs.interp_field(np.array([x0[0]]), np.array([x0[1]]), np.array([x0[2]]), self.Vw)[0]

    def test_wire_collects_particle_between_grid_nodes(self):
        """An ion aimed at the wire with a step smaller than the wire radius is collected."""
        x0 = (-0.6, 0.053, 0.1)
        ens, Q = self._fly(x0, (2e5, 0.0, 0.0))  # 2 mm per step
        self.assertFalse(ens.active_i[0])
        self.assertLess(abs(ens.x_i[0] - 0.05), self.wire.radius + 2e-3)
        self.assertAlmostEqual(Q, 1.0 - self._vw0(x0), places=10)

    def test_wire_ignores_particle_passing_outside_radius(self):
        """An ion passing at 2 wire radii from the axis is not collected and delivers no net charge."""
        x0 = (-0.6, 0.07, 0.1)
        ens, Q = self._fly(x0, (2e5, 0.0, 0.0))
        self.assertFalse(ens.active_i[0], "Ion should have left the domain")
        self.assertGreater(ens.x_i[0], 0.9)
        vw_end = self._vw0((ens.x_i[0], ens.y_i[0], ens.z_i[0]))
        self.assertAlmostEqual(Q, vw_end - self._vw0(x0), places=10)

    @unittest.expectedFailure
    def test_wire_collects_particle_crossing_within_one_step(self):
        """
        A particle whose straight path crosses the wire during one time step must be collected.

        KNOWN FAILURE: collection only checks whether the end-of-step position lies inside the wire,
        so a particle moving more than the wire diameter per step tunnels through it. With the
        analytical configuration (wire radius 1.6 mm, dt = 10 ns) a 2 eV electron moves ~8 mm per step
        and most crossings are missed.
        """
        # 3 cm per step, wire diameter 2 cm; the end points x = 0.035 and 0.065 straddle the wire
        x0 = (-0.595, 0.053, 0.1)
        ens, _ = self._fly(x0, (3e6, 0.0, 0.0))
        self.assertFalse(ens.active_i[0])
        self.assertLess(abs(ens.x_i[0] - 0.05), self.wire.radius + 0.03)


if __name__ == "__main__":
    unittest.main()
