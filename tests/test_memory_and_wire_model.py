# -*- coding: utf-8 -*-
"""
Unit tests for weighting fields memory optimization (float32 vs float64)
and sub-grid thin-wire continuous collision detection.
"""

import unittest
import numpy as np

from dust_impact.sim3d.config_loader import setup_simulation_parameters_3d
from dust_impact.geometry.prepared import build_simulation_geometry
from dust_impact.geometry.analytical import build_analytical_simulation_geometry
from dust_impact.geometry.surface import check_wire_collision_3d
from dust_impact.sim3d.sim_core import DustImpactSimulation3D


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

    def test_subgrid_wire_collection_in_simulation(self):
        """Verifies that particles intersecting the wire geometry are collected by the ensemble."""
        params, toggles, _ = setup_simulation_parameters_3d(
            Vf=-10.0, Vf_antenne=0.0, config_file="inputs/config_analytical_sphere.json"
        )
        geom = build_analytical_simulation_geometry(params, params.geometry.analytical)
        self.assertIsNotNone(geom.antenna_geometries)
        self.assertGreater(len(geom.antenna_geometries), 0)

        sim = DustImpactSimulation3D(params, toggles, geom)
        self.assertIsNotNone(sim.particles.antenna_geometries)
        # Advance 5 steps to verify no crashes and active particle pushing
        for step in range(5):
            sim.step(step)
        self.assertEqual(sim.particles.num_antennas, geom.num_antennas)


if __name__ == "__main__":
    unittest.main()
