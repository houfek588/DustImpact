# -*- coding: utf-8 -*-
"""
Unit tests for the standalone dust_impact.geometry package.
Validates:
1. Voxelization routines (sphere, box, cylinder, metal detection)
2. Analytical electrostatic models (vacuum sphere, Debye-shielded sphere, box)
3. Surface analysis & ray-tracing (voxel marching, outward surface normal calculation)
4. PreparedGeometry3D container (backward-compatible 10-tuple unpacking, builder orchestration)
"""

import unittest
import numpy as np

from dust_impact.sim3d.config_loader import SimulationParams3D, SimulationToggles3D, VTKFilesConfig
from dust_impact.geometry import (
    PreparedGeometry3D,
    build_simulation_geometry,
    detect_metal_mask_3d,
    voxelize_sphere,
    voxelize_box,
    voxelize_cylinder,
    interpolate_field_3d,
    ray_march_voxel_grid,
    compute_surface_normal_from_potential,
    compute_impact_intersection_and_normal,
    AnalyticalSphere,
    AnalyticalBox,
    generate_synthetic_analytical_fields
)
from dust_impact.sim3d.sim_core import DustImpactSimulation3D


class TestGeometryVoxelizer(unittest.TestCase):
    """ Tests for geometry voxelization routines. """

    def setUp(self):
        self.N = 31
        self.L = 3.0
        self.grid = np.linspace(-self.L, self.L, self.N)
        self.X, self.Y, self.Z = np.meshgrid(self.grid, self.grid, self.grid, indexing='ij')
        self.dx = self.grid[1] - self.grid[0]

    def test_voxelize_sphere(self):
        R = 1.0
        mask = voxelize_sphere(self.X, self.Y, self.Z, center=(0.0, 0.0, 0.0), radius=R)
        self.assertTrue(mask[self.N // 2, self.N // 2, self.N // 2])  # Center is inside
        self.assertFalse(mask[0, 0, 0])  # Corner is outside

        # Volume estimation (discrete sum * dx^3 approx (4/3)*pi*R^3)
        discrete_vol = np.sum(mask) * (self.dx ** 3)
        exact_vol = (4.0 / 3.0) * np.pi * (R ** 3)
        self.assertAlmostEqual(discrete_vol, exact_vol, delta=0.3 * exact_vol)

    def test_voxelize_box(self):
        min_b = (-0.5, -0.5, -0.5)
        max_b = (0.5, 0.5, 0.5)
        mask = voxelize_box(self.X, self.Y, self.Z, min_b, max_b)
        self.assertTrue(mask[self.N // 2, self.N // 2, self.N // 2])
        self.assertFalse(mask[0, 0, 0])

        discrete_vol = np.sum(mask) * (self.dx ** 3)
        exact_vol = 1.0  # 1x1x1 m^3
        self.assertAlmostEqual(discrete_vol, exact_vol, delta=0.2 * exact_vol)

    def test_voxelize_cylinder(self):
        p1 = [0.0, 0.0, -1.0]
        p2 = [0.0, 0.0, 1.0]
        radius = 0.3
        mask = voxelize_cylinder(self.X, self.Y, self.Z, p1, p2, radius)
        # Point on cylinder axis
        self.assertTrue(mask[self.N // 2, self.N // 2, self.N // 2])
        # Point along Z axis within cylinder
        idx_z = int(round((0.5 - self.grid[0]) / self.dx))
        self.assertTrue(mask[self.N // 2, self.N // 2, idx_z])
        # Point far away
        self.assertFalse(mask[0, 0, 0])

    def test_detect_metal_mask_3d(self):
        # Create a potential shell with step jump
        r = np.sqrt(self.X**2 + self.Y**2 + self.Z**2)
        V = np.where(r <= 1.0, 10.0, 10.0 / np.maximum(r, 0.1))
        mask = detect_metal_mask_3d(V, self.dx, threshold=0.3)
        self.assertTrue(mask[self.N // 2, self.N // 2, self.N // 2])
        self.assertFalse(mask[0, 0, 0])


class TestAnalyticalGeometry(unittest.TestCase):
    """ Tests for analytical potential and electric field solutions. """

    def setUp(self):
        self.N = 41
        self.L = 5.0
        self.grid = np.linspace(-self.L, self.L, self.N)
        self.X, self.Y, self.Z = np.meshgrid(self.grid, self.grid, self.grid, indexing='ij')

    def test_analytical_sphere_vacuum(self):
        R = 1.0
        V0 = 25.0
        sphere = AnalyticalSphere(center=(0.0, 0.0, 0.0), radius=R, voltage=V0, debye_length=None)

        mask = sphere.get_mask(self.X, self.Y, self.Z)
        V = sphere.get_potential(self.X, self.Y, self.Z)
        Ex, Ey, Ez = sphere.get_electric_field(self.X, self.Y, self.Z)

        # 1. Inside conductor: V = V0, E = 0
        np.testing.assert_allclose(V[mask], V0, rtol=1e-5)
        np.testing.assert_allclose(Ex[mask], 0.0, atol=1e-10)
        np.testing.assert_allclose(Ey[mask], 0.0, atol=1e-10)
        np.testing.assert_allclose(Ez[mask], 0.0, atol=1e-10)

        # 2. Outside conductor at r = 2.0 m: V should be V0 * (1.0 / 2.0) = 12.5 V
        # With 41 points on [-5, 5], dx = 0.25, so 2.0 is exactly at an integer index
        idx_2m = np.argmin(np.abs(self.grid - 2.0))
        idx_0 = np.argmin(np.abs(self.grid - 0.0))
        r_actual = self.grid[idx_2m]
        v_2m = V[idx_2m, idx_0, idx_0]
        self.assertAlmostEqual(v_2m, V0 * (R / r_actual), places=5)

        # Electric field at r = 2.0 m: E = V0 * R / r^2 = 25 / 4 = 6.25 V/m along +X
        ex_2m = Ex[idx_2m, idx_0, idx_0]
        self.assertAlmostEqual(ex_2m, V0 * R / (r_actual ** 2), places=5)

    def test_analytical_sphere_debye_shielded(self):
        R = 1.0
        V0 = 10.0
        lambda_D = 1.5
        sphere = AnalyticalSphere(center=(0.0, 0.0, 0.0), radius=R, voltage=V0, debye_length=lambda_D)

        V = sphere.get_potential(self.X, self.Y, self.Z)
        idx_2m = np.argmin(np.abs(self.grid - 2.0))
        idx_0 = np.argmin(np.abs(self.grid - 0.0))
        r_actual = self.grid[idx_2m]

        expected_v = V0 * (R / r_actual) * np.exp(-(r_actual - R) / lambda_D)
        self.assertAlmostEqual(V[idx_2m, idx_0, idx_0], expected_v, places=5)


class TestSurfaceAndRayTracing(unittest.TestCase):
    """ Tests for surface normal calculation and ray tracing algorithms. """

    def setUp(self):
        self.N = 31
        self.L = 3.0
        self.x_grid = np.linspace(-self.L, self.L, self.N)
        self.y_grid = np.linspace(-self.L, self.L, self.N)
        self.z_grid = np.linspace(-self.L, self.L, self.N)
        self.dx = self.x_grid[1] - self.x_grid[0]
        self.dy = self.dx
        self.dz = self.dx
        self.X, self.Y, self.Z = np.meshgrid(self.x_grid, self.y_grid, self.z_grid, indexing='ij')

    def test_ray_march_voxel_grid(self):
        # Sphere of radius 1.0 at origin
        mask = (self.X**2 + self.Y**2 + self.Z**2) <= 1.0**2
        p_start = np.array([-2.5, 0.0, 0.0])
        dir_u = np.array([1.0, 0.0, 0.0])

        intersection = ray_march_voxel_grid(
            p_start, dir_u, mask, self.x_grid, self.y_grid, self.z_grid, self.dx, self.dy, self.dz
        )
        self.assertIsNotNone(intersection)
        # Intersection on sphere surface at x approx -1.0
        self.assertAlmostEqual(intersection[0], -1.0, delta=self.dx * 1.5)
        self.assertAlmostEqual(intersection[1], 0.0, delta=1e-5)
        self.assertAlmostEqual(intersection[2], 0.0, delta=1e-5)

    def test_compute_surface_normal_from_potential(self):
        # 1/r potential centered at origin
        r = np.sqrt(self.X**2 + self.Y**2 + self.Z**2)
        V = 10.0 / np.maximum(r, 0.5)

        # Normal on +X axis at (1.0, 0, 0) should point radially outward: [+1, 0, 0]
        norm = compute_surface_normal_from_potential(
            np.array([1.0, 0.0, 0.0]), V,
            self.x_grid, self.y_grid, self.z_grid, self.dx, self.dy, self.dz
        )
        # Outward normal from -grad(V) where V decays outward
        # E = -grad(V) points outward (+X)
        self.assertGreater(norm[0], 0.9)
        self.assertAlmostEqual(norm[1], 0.0, delta=0.1)
        self.assertAlmostEqual(norm[2], 0.0, delta=0.1)


class TestPreparedGeometryAndIntegration(unittest.TestCase):
    """ Tests for PreparedGeometry3D container and solver integration. """

    def test_prepared_geometry_unpacking(self):
        N = 10
        V_bg = np.zeros((N, N, N))
        Vw_grids = [np.ones((N, N, N))]
        Ex_bg, Ey_bg, Ez_bg = np.zeros((N, N, N)), np.zeros((N, N, N)), np.zeros((N, N, N))
        Ewx, Ewy, Ewz = [np.zeros((N, N, N))], [np.zeros((N, N, N))], [np.zeros((N, N, N))]
        ant_masks = [np.zeros((N, N, N), dtype=bool)]
        sc_mask = np.zeros((N, N, N), dtype=bool)

        prep = PreparedGeometry3D(
            V_bg=V_bg, Vw_grids=Vw_grids,
            Ex_bg=Ex_bg, Ey_bg=Ey_bg, Ez_bg=Ez_bg,
            Ewx_list=Ewx, Ewy_list=Ewy, Ewz_list=Ewz,
            antenna_masks_3d=ant_masks, spacecraft_mask_3d=sc_mask
        )

        # Test 10-tuple unpacking
        v_bg_out, vw_out, ex_out, ey_out, ez_out, ewx_out, ewy_out, ewz_out, ant_out, sc_out = prep
        self.assertEqual(v_bg_out.shape, (N, N, N))
        self.assertEqual(len(vw_out), 1)
        self.assertEqual(prep.num_antennas, 1)
        self.assertEqual(len(prep), 10)
        self.assertIn("PreparedGeometry3D", prep.summary())

    def test_build_simulation_geometry_synthetic(self):
        params = SimulationParams3D(
            domain_half_length_x_m=2.0,
            domain_half_length_y_m=2.0,
            domain_half_length_z_m=2.0,
            grid_nodes_x=12,
            grid_nodes_y=12,
            grid_nodes_z=12,
            vtk_files=VTKFilesConfig(
                spis_background_potential_file="",
                spacecraft_weighting_file="",
                antenna_weighting_files=[]
            )
        )

        prep = build_simulation_geometry(params)
        self.assertIsInstance(prep, PreparedGeometry3D)
        self.assertEqual(prep.grid_shape, (12, 12, 12))
        self.assertGreater(np.sum(prep.spacecraft_mask_3d), 0)

        # Test passing directly to DustImpactSimulation3D
        toggles = SimulationToggles3D(
            enable_plasma_self_field=False
        )
        sim = DustImpactSimulation3D(params, toggles, prep)
        self.assertEqual(sim.spacecraft_mask_3d.shape, (12, 12, 12))


if __name__ == '__main__':
    unittest.main()
