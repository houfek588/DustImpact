import unittest
import numpy as np

from dust_impact.solver.config_loader import (
    SimulationParams3D, SimulationToggles3D, GeometryConfig,
    AnalyticalGeometryConfig, AnalyticalSpacecraftPart
)
from dust_impact.solver.ensemble import ParticleEnsemble
from dust_impact.solver.field_solver import FieldSolver3D


class TestParticleReflection(unittest.TestCase):
    def setUp(self):
        self.params = SimulationParams3D(
            num_macroparticles=10,
            simulation_duration_s=1e-8,
            time_step_s=1e-9,
            grid_nodes_x=16,
            grid_nodes_y=16,
            grid_nodes_z=16,
            domain_half_length_x_m=3.0,
            domain_half_length_y_m=3.0,
            domain_half_length_z_m=3.0,
            geometry=GeometryConfig(
                source="analytical",
                analytical=AnalyticalGeometryConfig(
                    spacecraft=AnalyticalSpacecraftPart(
                        type="sphere",
                        center=[0.0, 0.0, 0.0],
                        radius=1.0,
                        voltage_V=10.0
                    ),
                    antennas=[]
                )
            )
        )
        self.sc_mask = np.zeros((16, 16, 16), dtype=bool)
        # Mark grid nodes inside sphere as True
        X, Y, Z = np.meshgrid(self.params.x_grid, self.params.y_grid, self.params.z_grid, indexing='ij')
        self.sc_mask = (X**2 + Y**2 + Z**2) <= 1.0**2

    def test_specular_reflection_sphere_physics(self):
        toggles = SimulationToggles3D(
            particle_spacecraft_mode="reflect",
            particle_boundary_mode="periodic"
        )
        sc_geom = self.params.geometry.analytical.spacecraft
        ens = ParticleEnsemble(
            self.params, self.sc_mask, num_antennas=0, toggles=toggles,
            spacecraft_geometry=sc_geom
        )

        # 1. Normal head-on collision at (1.0, 0, 0)
        x = np.array([0.95])  # Penetrated inside sphere of radius 1.0
        y = np.array([0.0])
        z = np.array([0.0])
        vx = np.array([-5e5])  # Inward
        vy = np.array([0.0])
        vz = np.array([0.0])
        x_old = np.array([1.05])
        y_old = np.array([0.0])
        z_old = np.array([0.0])

        speed_orig = np.sqrt(vx[0]**2 + vy[0]**2 + vz[0]**2)

        ens._reflect_particles_from_spacecraft(
            x, y, z, vx, vy, vz, np.array([0]), x_old, y_old, z_old
        )

        # Kinetic energy strictly conserved
        speed_new = np.sqrt(vx[0]**2 + vy[0]**2 + vz[0]**2)
        self.assertAlmostEqual(speed_new, speed_orig, places=10)

        # Velocity reversed outward along X
        self.assertGreater(vx[0], 0.0)
        self.assertAlmostEqual(vy[0], 0.0, places=10)
        self.assertAlmostEqual(vz[0], 0.0, places=10)

        # Position pushed strictly outside sphere
        r_new = np.sqrt(x[0]**2 + y[0]**2 + z[0]**2)
        self.assertGreater(r_new, 1.0)

        # 2. Oblique collision at 45 degrees
        x = np.array([0.7])  # r = sqrt(0.7^2 + 0.7^2) = 0.9899 < 1.0
        y = np.array([0.7])
        z = np.array([0.0])
        vx = np.array([-3e5])
        vy = np.array([-2e5])
        vz = np.array([1e5])
        x_old = np.array([0.8])
        y_old = np.array([0.8])
        z_old = np.array([0.0])

        speed_oblique_orig = np.sqrt(vx[0]**2 + vy[0]**2 + vz[0]**2)

        ens._reflect_particles_from_spacecraft(
            x, y, z, vx, vy, vz, np.array([0]), x_old, y_old, z_old
        )

        speed_oblique_new = np.sqrt(vx[0]**2 + vy[0]**2 + vz[0]**2)
        np.testing.assert_allclose(speed_oblique_new, speed_oblique_orig, rtol=1e-10)

        # Normal at 45 deg is ~ (1/sqrt(2), 1/sqrt(2), 0)
        # Normal component of reflected velocity must be positive
        norm_v = (vx[0] * x[0] + vy[0] * y[0] + vz[0] * z[0]) / np.sqrt(x[0]**2 + y[0]**2 + z[0]**2)
        self.assertGreater(norm_v, 0.0)

        r_oblique_new = np.sqrt(x[0]**2 + y[0]**2 + z[0]**2)
        self.assertGreater(r_oblique_new, 1.0)

    def test_particle_spacecraft_mode_absorb_vs_reflect(self):
        # 1. Mode "absorb": particles penetrating the spacecraft are destroyed
        toggles_absorb = SimulationToggles3D(
            particle_spacecraft_mode="absorb",
            particle_boundary_mode="periodic"
        )
        ens_abs = ParticleEnsemble(
            self.params, self.sc_mask, num_antennas=0, toggles=toggles_absorb,
            spacecraft_geometry=self.params.geometry.analytical.spacecraft
        )
        ens_abs.cloud_injected = True
        field_solver = FieldSolver3D(
            self.params, toggles_absorb,
            np.zeros((16, 16, 16)), np.zeros((16, 16, 16)), np.zeros((16, 16, 16)), np.zeros((16, 16, 16)),
            [], [], [], self.sc_mask, num_antennas=0
        )

        # Place particle just outside moving into the spacecraft
        # With dx = 0.4 m, start at x=1.2 m, move with vx = -7e8 m/s -> x_new = 0.5 m (hits mask)
        ens_abs.x_e[0] = 1.2
        ens_abs.y_e[0] = 0.0
        ens_abs.z_e[0] = 0.0
        ens_abs.vx_e[0] = -7e8
        ens_abs.vy_e[0] = 0.0
        ens_abs.vz_e[0] = 0.0
        ens_abs.active_e[0] = True

        ens_abs.push_species(
            ens_abs.x_e, ens_abs.y_e, ens_abs.z_e,
            ens_abs.vx_e, ens_abs.vy_e, ens_abs.vz_e,
            ens_abs.active_e, ens_abs.was_outside_e,
            9.1e-31, -1.6e-19, -1e-15,
            field_solver, [], [], [], [],
            enable_collection=False,
            boundary_mode="periodic"
        )
        # In absorb mode, particle must be deactivated
        self.assertFalse(ens_abs.active_e[0])

        # 2. Mode "reflect": particles penetrating the spacecraft remain active and bounce
        toggles_reflect = SimulationToggles3D(
            particle_spacecraft_mode="reflect",
            particle_boundary_mode="periodic"
        )
        ens_refl = ParticleEnsemble(
            self.params, self.sc_mask, num_antennas=0, toggles=toggles_reflect,
            spacecraft_geometry=self.params.geometry.analytical.spacecraft
        )
        ens_refl.cloud_injected = True

        ens_refl.x_e[0] = 1.2
        ens_refl.y_e[0] = 0.0
        ens_refl.z_e[0] = 0.0
        ens_refl.vx_e[0] = -7e8
        ens_refl.vy_e[0] = 0.0
        ens_refl.vz_e[0] = 0.0
        ens_refl.active_e[0] = True

        ens_refl.push_species(
            ens_refl.x_e, ens_refl.y_e, ens_refl.z_e,
            ens_refl.vx_e, ens_refl.vy_e, ens_refl.vz_e,
            ens_refl.active_e, ens_refl.was_outside_e,
            9.1e-31, -1.6e-19, -1e-15,
            field_solver, [], [], [], [],
            enable_collection=False,
            boundary_mode="periodic"
        )
        # In reflect mode, particle must remain active!
        self.assertTrue(ens_refl.active_e[0])
        # Position must be outside the sphere
        r_after = np.sqrt(ens_refl.x_e[0]**2 + ens_refl.y_e[0]**2 + ens_refl.z_e[0]**2)
        self.assertGreater(r_after, 1.0)
        # Velocity must be reversed outward
        self.assertGreater(ens_refl.vx_e[0], 0.0)

    def test_specular_reflection_box_physics(self):
        box_geom = AnalyticalSpacecraftPart(
            type="box",
            center=[0.0, 0.0, 0.0],
            dimensions=[2.0, 2.0, 2.0]  # half-extent is 1.0 along x, y, z
        )
        toggles = SimulationToggles3D(particle_spacecraft_mode="reflect")
        ens = ParticleEnsemble(
            self.params, self.sc_mask, num_antennas=0, toggles=toggles,
            spacecraft_geometry=box_geom
        )

        x = np.array([0.95])  # Entered right face (x = 1.0)
        y = np.array([0.2])
        z = np.array([0.3])
        vx = np.array([-4e5])  # Moving in -x direction into right face
        vy = np.array([1e5])
        vz = np.array([0.0])
        x_old = np.array([1.05])
        y_old = np.array([0.2])
        z_old = np.array([0.3])

        speed_orig = np.sqrt(vx[0]**2 + vy[0]**2 + vz[0]**2)

        ens._reflect_particles_from_spacecraft(
            x, y, z, vx, vy, vz, np.array([0]), x_old, y_old, z_old
        )

        speed_new = np.sqrt(vx[0]**2 + vy[0]**2 + vz[0]**2)
        self.assertAlmostEqual(speed_new, speed_orig, places=10)

        # vx reversed to positive, vy and vz untouched
        self.assertGreater(vx[0], 0.0)
        self.assertAlmostEqual(vy[0], 1e5, places=10)
        self.assertAlmostEqual(vz[0], 0.0, places=10)
        self.assertGreater(x[0], 1.0)


if __name__ == "__main__":
    unittest.main()
