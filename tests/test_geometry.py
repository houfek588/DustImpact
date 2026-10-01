# -*- coding: utf-8 -*-
"""
Unit tests for the standalone dust_impact.geometry package.
Validates:
1. Voxelization routines (sphere, box, cylinder, metal detection)
2. Analytical electrostatic models (vacuum sphere, Debye-shielded sphere, box)
3. Surface analysis & ray-tracing (voxel marching, outward surface normal calculation)
4. PreparedGeometry3D container (backward-compatible 10-tuple unpacking, builder orchestration)
"""

import os
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


    def test_exact_ray_intersections(self):
        from dust_impact.geometry import (
            intersect_ray_sphere,
            intersect_ray_box,
            intersect_ray_cylinder,
            intersect_ray_analytical_spacecraft
        )
        from dust_impact.sim3d.config_loader import AnalyticalSpacecraftPart

        # 1. Sphere hit
        p_start = np.array([-3.0, 0.0, 0.0])
        dir_u = np.array([1.0, 0.0, 0.0])
        hit = intersect_ray_sphere(p_start, dir_u, center=(0, 0, 0), radius=1.0)
        self.assertIsNotNone(hit)
        hit_pt, norm, dist = hit
        np.testing.assert_allclose(hit_pt, [-1.0, 0.0, 0.0], atol=1e-7)
        np.testing.assert_allclose(norm, [-1.0, 0.0, 0.0], atol=1e-7)
        self.assertAlmostEqual(dist, 2.0, places=6)

        # 2. Box hit
        hit_box = intersect_ray_box(p_start, dir_u, min_bounds=[-0.5, -0.5, -0.5], max_bounds=[0.5, 0.5, 0.5])
        self.assertIsNotNone(hit_box)
        hit_pt, norm, dist = hit_box
        np.testing.assert_allclose(hit_pt, [-0.5, 0.0, 0.0], atol=1e-7)
        np.testing.assert_allclose(norm, [-1.0, 0.0, 0.0], atol=1e-7)

        # 3. Cylinder hit
        p_cyl_start = np.array([0.0, -3.0, 0.0])
        dir_cyl = np.array([0.0, 1.0, 0.0])
        hit_cyl = intersect_ray_cylinder(p_cyl_start, dir_cyl, p1=[0, 0, -1], p2=[0, 0, 1], radius=0.5)
        self.assertIsNotNone(hit_cyl)
        hit_pt, norm, dist = hit_cyl
        np.testing.assert_allclose(hit_pt, [0.0, -0.5, 0.0], atol=1e-7)
        np.testing.assert_allclose(norm, [0.0, -1.0, 0.0], atol=1e-7)

        # 4. Composite spacecraft part
        composite = AnalyticalSpacecraftPart(
            type="composite",
            parts=[
                AnalyticalSpacecraftPart(type="box", center=[0, 0, 0], dimensions=[1, 1, 1]),
                AnalyticalSpacecraftPart(type="sphere", center=[2, 0, 0], radius=0.5)
            ]
        )
        hit_comp = intersect_ray_analytical_spacecraft(p_start, dir_u, composite)
        self.assertIsNotNone(hit_comp)
        np.testing.assert_allclose(hit_comp[0], [-0.5, 0.0, 0.0], atol=1e-7)

    def test_analytical_laplace_solver_fields(self):
        from dust_impact.geometry import solve_laplace_dirichlet_3d
        N = 15
        dx = 0.5
        dirichlet_mask = np.zeros((N, N, N), dtype=bool)
        center_idx = N // 2
        dirichlet_mask[center_idx, center_idx, center_idx] = True

        rhs = np.zeros((N, N, N), dtype=float)
        rhs[center_idx, center_idx, center_idx] = 10.0

        solutions = solve_laplace_dirichlet_3d(
            N, N, N, dx, dx, dx,
            dirichlet_mask=dirichlet_mask,
            rhs_values=[rhs]
        )
        self.assertEqual(len(solutions), 1)
        V = solutions[0]
        self.assertAlmostEqual(V[center_idx, center_idx, center_idx], 10.0, places=4)
        # Potential decays away from center
        self.assertLess(V[center_idx + 2, center_idx, center_idx], 10.0)
        self.assertGreater(V[center_idx + 2, center_idx, center_idx], 0.0)

    def test_build_analytical_simulation_geometry_sphere_and_box(self):
        from dust_impact.sim3d.config_loader import (
            GeometryConfig, AnalyticalGeometryConfig, AnalyticalSpacecraftPart, AnalyticalAntennaGeometry
        )
        params = SimulationParams3D(
            domain_half_length_x_m=3.0,
            domain_half_length_y_m=3.0,
            domain_half_length_z_m=3.0,
            grid_nodes_x=16,
            grid_nodes_y=16,
            grid_nodes_z=16,
            impact_location_xyz_m=[-2.5, 0.0, 0.0],
            impact_direction_vector=[1.0, 0.0, 0.0],
            geometry=GeometryConfig(
                source="analytical",
                analytical=AnalyticalGeometryConfig(
                    spacecraft=AnalyticalSpacecraftPart(type="sphere", center=[0, 0, 0], radius=1.0),
                    antennas=[
                        AnalyticalAntennaGeometry(p_start=[0, 1, 0], p_end=[0, 2.5, 0], radius=0.05),
                        AnalyticalAntennaGeometry(p_start=[0, -1, 0], p_end=[0, -2.5, 0], radius=0.05)
                    ]
                )
            )
        )

        prep = build_simulation_geometry(params)
        self.assertIsInstance(prep, PreparedGeometry3D)
        self.assertEqual(prep.num_antennas, 2)
        # Impact point exactly at [-1, 0, 0]
        np.testing.assert_allclose(prep.impact_pos, [-1.0, 0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(prep.impact_normal, [-1.0, 0.0, 0.0], atol=1e-6)
        # Weighting field Vw inside antenna 1 is 1.0, inside antenna 2 is 0.0
        self.assertEqual(len(prep.Vw_grids), 2)
        self.assertAlmostEqual(np.max(prep.Vw_grids[0]), 1.0, places=4)

    def test_config_loader_analytical_geometry(self):
        from dust_impact.sim3d.config_loader import setup_simulation_parameters_3d
        params, toggles, plot_cfg = setup_simulation_parameters_3d(
            Vf=10.0,
            config_file="inputs/config_analytical_sphere.json"
        )
        self.assertEqual(params.geometry.source, "analytical")
        self.assertEqual(params.geometry.analytical.spacecraft.type, "sphere")
        self.assertEqual(params.geometry.analytical.spacecraft.voltage_V, 10.0)
        self.assertEqual(params.spacecraft_voltage_V, 10.0)
        self.assertEqual(len(params.geometry.analytical.antennas), 4)
        self.assertEqual(params.geometry.analytical.antennas[0].voltage_V, 5.0)
        self.assertEqual(params.geometry.analytical.antennas[1].voltage_V, -5.0)
        self.assertEqual(params.geometry.analytical.antennas[2].voltage_V, 0.0)
        self.assertEqual(params.geometry.analytical.antennas[3].voltage_V, 2.5)
        self.assertEqual(params.antenna_bias_voltage_V, [5.0, -5.0, 0.0, 2.5])
        self.assertEqual(params.V_bias, [5.0, -5.0, 0.0, 2.5])
        self.assertEqual(params.antenna_capacitance_F, [2e-12, 2e-12, 2e-12, 2e-12])
        self.assertEqual(params.antenna_resistance_Ohm, [100000.0, 100000.0, 100000.0, 100000.0])

    def test_config_loader_spis_unified_geometry(self):
        from dust_impact.sim3d.config_loader import setup_simulation_parameters_3d
        params, toggles, plot_cfg = setup_simulation_parameters_3d(
            Vf=25.0,
            config_file="config.json"
        )
        self.assertEqual(params.geometry.source, "spis")
        self.assertEqual(params.geometry.spis.spacecraft.voltage_V, 25.0)
        self.assertEqual(params.spacecraft_voltage_V, 25.0)
        self.assertEqual(len(params.geometry.spis.antennas), 3)
        self.assertEqual(params.geometry.spis.antennas[0].weighting_file, os.path.abspath("inputs/spis_Vw_ant1.vtk"))
        self.assertEqual(params.antenna_capacitance_F, [2e-12, 2e-12, 2e-12])
        self.assertEqual(params.antenna_resistance_Ohm, [100000.0, 100000.0, 100000.0])
        self.assertEqual(params.geometry.spis.weighting_threshold, 0.85)
        self.assertEqual(params.weighting_threshold, 0.85)

    def test_analytical_antenna_voltage_specification(self):
        from dust_impact.sim3d.config_loader import (
            GeometryConfig, AnalyticalGeometryConfig, AnalyticalSpacecraftPart, AnalyticalAntennaGeometry
        )
        params = SimulationParams3D(
            domain_half_length_x_m=3.0,
            domain_half_length_y_m=3.0,
            domain_half_length_z_m=3.0,
            grid_nodes_x=16,
            grid_nodes_y=16,
            grid_nodes_z=16,
            spacecraft_voltage_V=15.0,
            impact_location_xyz_m=[-2.5, 0.0, 0.0],
            impact_direction_vector=[1.0, 0.0, 0.0],
            geometry=GeometryConfig(
                source="analytical",
                analytical=AnalyticalGeometryConfig(
                    spacecraft=AnalyticalSpacecraftPart(type="sphere", center=[0, 0, 0], radius=1.0),
                    antennas=[
                        AnalyticalAntennaGeometry(p_start=[0, 1.0, 0], p_end=[0, 2.5, 0], radius=0.05, voltage_V=8.0),
                        AnalyticalAntennaGeometry(p_start=[0, -1.0, 0], p_end=[0, -2.5, 0], radius=0.05, voltage_V=-4.0)
                    ]
                )
            )
        )

        self.assertEqual(params.antenna_bias_voltage_V, [8.0, -4.0])
        self.assertEqual(params.V_bias, [8.0, -4.0])

        prep = build_simulation_geometry(params)
        self.assertEqual(params.antenna_bias_voltage_V, [8.0, -4.0])
        self.assertEqual(params.V_bias, [8.0, -4.0])

        # V_bg on spacecraft (center) is V_sc = 15.0
        center_idx = params.grid_nodes_x // 2
        self.assertAlmostEqual(prep.V_bg[center_idx, center_idx, center_idx], 15.0, places=3)

        # Inside antenna 1 ([0, 1.8, 0]), V_bg is 8.0 V
        # Inside antenna 2 ([0, -1.8, 0]), V_bg is -4.0 V
        idx_y_ant1 = int(round((1.8 - params.y_grid[0]) / params.dy))
        idx_y_ant2 = int(round((-1.8 - params.y_grid[0]) / params.dy))
        self.assertAlmostEqual(prep.V_bg[center_idx, idx_y_ant1, center_idx], 8.0, places=3)
        self.assertAlmostEqual(prep.V_bg[center_idx, idx_y_ant2, center_idx], -4.0, places=3)

    def test_analytical_antenna_voltage_omitted_defaults_to_zero(self):
        from dust_impact.sim3d.config_loader import (
            GeometryConfig, AnalyticalGeometryConfig, AnalyticalSpacecraftPart, AnalyticalAntennaGeometry
        )
        ant = AnalyticalAntennaGeometry(p_start=[0, 1.0, 0], p_end=[0, 2.5, 0], radius=0.05)
        self.assertEqual(ant.voltage_V, 0.0)
        self.assertEqual(ant.effective_voltage, 0.0)

        params = SimulationParams3D(
            domain_half_length_x_m=3.0,
            domain_half_length_y_m=3.0,
            domain_half_length_z_m=3.0,
            grid_nodes_x=16,
            grid_nodes_y=16,
            grid_nodes_z=16,
            spacecraft_voltage_V=10.0,
            geometry=GeometryConfig(
                source="analytical",
                analytical=AnalyticalGeometryConfig(
                    spacecraft=AnalyticalSpacecraftPart(type="sphere", center=[0, 0, 0], radius=1.0),
                    antennas=[ant]
                )
            )
        )
        self.assertEqual(params.antenna_bias_voltage_V, [0.0])
        self.assertEqual(params.V_bias, [0.0])

        prep = build_simulation_geometry(params)
        self.assertEqual(params.antenna_bias_voltage_V, [0.0])
        center_idx = params.grid_nodes_x // 2
        idx_y_ant = int(round((1.8 - params.y_grid[0]) / params.dy))
        self.assertAlmostEqual(prep.V_bg[center_idx, idx_y_ant, center_idx], 0.0, places=3)

    def test_physics_and_numeric_config_separation(self):
        from dust_impact.sim3d.config_loader import (
            setup_simulation_parameters_3d, PhysicsConfig, NumericConfig
        )
        params, toggles, plot_cfg = setup_simulation_parameters_3d(
            Vf=25.0, config_file="config.json"
        )
        # Check sub-dataclass attributes
        self.assertEqual(params.physics.ion_mass_amu, 27.0)
        self.assertEqual(params.physics.solar_wind_density_m3, 1e7)
        self.assertEqual(params.physics.total_impact_charge_C, 5e-11)
        self.assertEqual(params.numeric.num_macroparticles, 20000)
        self.assertEqual(params.numeric.grid_nodes, [150, 150, 150])
        self.assertEqual(params.grid_nodes, [150, 150, 150])
        self.assertEqual(params.numeric.domain_half_length_m, [10.0, 10.0, 10.0])
        self.assertEqual(params.domain_half_length_m, [10.0, 10.0, 10.0])
        self.assertEqual(params.numeric.grid_nodes_x, 150)
        self.assertEqual(params.numeric.time_step_s, 2e-9)
        self.assertEqual(params.numeric.num_time_steps, 250)
        self.assertEqual(params.num_time_steps, 250)
        self.assertEqual(params.steps, 250)
        self.assertAlmostEqual(params.simulation_duration_s, 5e-7)

        # Check flat backwards-compatibility access
        self.assertEqual(params.ion_mass_amu, params.physics.ion_mass_amu)
        self.assertEqual(params.num_macroparticles, params.numeric.num_macroparticles)
        self.assertEqual(params.grid_nodes_x, 150)
        self.assertEqual(params.domain_half_length_x_m, 10.0)
        self.assertEqual(params.Nx, 150)
        self.assertEqual(params.L_x, 10.0)
        self.assertEqual(params.time_step_s, params.numeric.time_step_s)

        # Check direct dataclass construction with PhysicsConfig / NumericConfig
        p_custom = SimulationParams3D(
            physics=PhysicsConfig(ion_mass_amu=45.0, total_impact_charge_C=8e-11),
            numeric=NumericConfig(
                num_macroparticles=12345,
                grid_nodes=[30, 40, 50],
                domain_half_length_m=[2.0, 3.0, 4.0],
                num_time_steps=500
            )
        )
        self.assertEqual(p_custom.ion_mass_amu, 45.0)
        self.assertEqual(p_custom.physics.ion_mass_amu, 45.0)
        self.assertEqual(p_custom.total_impact_charge_C, 8e-11)
        self.assertEqual(p_custom.num_macroparticles, 12345)
        self.assertEqual(p_custom.numeric.num_macroparticles, 12345)
        self.assertEqual(p_custom.grid_nodes, [30, 40, 50])
        self.assertEqual(p_custom.numeric.grid_nodes, [30, 40, 50])
        self.assertEqual(p_custom.grid_nodes_x, 30)
        self.assertEqual(p_custom.grid_nodes_y, 40)
        self.assertEqual(p_custom.grid_nodes_z, 50)
        self.assertEqual(p_custom.Nx, 30)
        self.assertEqual(p_custom.Ny, 40)
        self.assertEqual(p_custom.Nz, 50)
        self.assertEqual(p_custom.domain_half_length_m, [2.0, 3.0, 4.0])
        self.assertEqual(p_custom.L_x, 2.0)
        self.assertEqual(p_custom.L_y, 3.0)
        self.assertEqual(p_custom.L_z, 4.0)
        self.assertEqual(p_custom.num_time_steps, 500)
        self.assertEqual(p_custom.steps, 500)
        self.assertAlmostEqual(p_custom.simulation_duration_s, 500 * 2e-9)

    def test_impact_config_structure(self):
        from dust_impact.sim3d.config_loader import (
            setup_simulation_parameters_3d, PhysicsConfig, ImpactConfig
        )
        # Test loading from config.json with unified "impact" object
        params, toggles, plot_cfg = setup_simulation_parameters_3d(
            Vf=25.0, config_file="config.json"
        )
        self.assertEqual(params.physics.impact.location, [-2.0, 2.0, 0.0])
        self.assertEqual(params.physics.impact.direction, [1.0, -1.0, 0.0])
        self.assertEqual(params.physics.impact.time_delay_s, 1e-08)

        self.assertEqual(params.impact.location, [-2.0, 2.0, 0.0])
        self.assertEqual(params.impact.direction, [1.0, -1.0, 0.0])
        self.assertEqual(params.impact.time_delay_s, 1e-08)

        # Backwards-compatibility checks
        self.assertEqual(params.impact_location_xyz_m, [-2.0, 2.0, 0.0])
        self.assertEqual(params.impact_pos, [-2.0, 2.0, 0.0])
        self.assertEqual(params.impact_direction_vector, [1.0, -1.0, 0.0])
        self.assertEqual(params.impact_time_delay_s, 1e-08)
        self.assertEqual(params.t_delay, 1e-08)
        self.assertEqual(params.physics.impact_location_xyz_m, [-2.0, 2.0, 0.0])
        self.assertEqual(params.physics.impact_direction_vector, [1.0, -1.0, 0.0])
        self.assertEqual(params.physics.impact_time_delay_s, 1e-08)

        # Test direct creation via ImpactConfig
        custom_impact = ImpactConfig(location=[1.0, -1.0, 2.0], direction=[0.0, 0.0, 1.0], time_delay_s=5e-9)
        p_custom = SimulationParams3D(impact=custom_impact)
        self.assertEqual(p_custom.impact.location, [1.0, -1.0, 2.0])
        self.assertEqual(p_custom.impact_location_xyz_m, [1.0, -1.0, 2.0])
        self.assertEqual(p_custom.impact_pos, [1.0, -1.0, 2.0])
        self.assertEqual(p_custom.impact.direction, [0.0, 0.0, 1.0])
        self.assertEqual(p_custom.impact_direction_vector, [0.0, 0.0, 1.0])
        self.assertEqual(p_custom.impact.time_delay_s, 5e-9)
        self.assertEqual(p_custom.impact_time_delay_s, 5e-9)
        self.assertEqual(p_custom.t_delay, 5e-9)
        self.assertEqual(p_custom.physics.impact.location, [1.0, -1.0, 2.0])


if __name__ == '__main__':
    unittest.main()
