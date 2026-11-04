# -*- coding: utf-8 -*-
"""
Physical consistency tests of the production 3D solver components:
1. Initial velocity sampling reproduces the configured plasma temperatures.
2. Energy conservation of a particle moving in a static background potential.
3. Green's reciprocity between the self-field solver and the antenna weighting potential
   (the basis of the Ramo-Shockley theorem): Q_induced = -q * Vw(r).
4. Absence of a numerical self-force on an isolated macro-particle.
"""

import contextlib
import io
import unittest
from types import SimpleNamespace

import numpy as np

from dust_impact.physics.constants import e, m_e, eps_0
from dust_impact.geometry.analytical import solve_laplace_dirichlet_3d
from dust_impact.solver.config_loader import SimulationParams3D, SimulationToggles3D
from dust_impact.solver.params import ImpactConfig
from dust_impact.solver.ensemble import ParticleEnsemble
from dust_impact.solver.field_solver import FieldSolver3D


def _cubic_params(N: int, L: float, **kwargs) -> SimulationParams3D:
    """Simulation parameters on a cubic grid [-L, L]^3 with N nodes per axis."""
    return SimulationParams3D(
        grid_nodes_x=N, grid_nodes_y=N, grid_nodes_z=N,
        domain_half_length_x_m=L, domain_half_length_y_m=L, domain_half_length_z_m=L,
        **kwargs
    )


def _field_solver(params, toggles, V_bg=None, conductor_mask=None) -> FieldSolver3D:
    """FieldSolver3D with an optional static background potential and no antennas."""
    shape = (params.Nx, params.Ny, params.Nz)
    V = np.zeros(shape) if V_bg is None else V_bg
    Ex, Ey, Ez = np.gradient(-V, params.dx, params.dy, params.dz)
    mask = np.zeros(shape, dtype=bool) if conductor_mask is None else conductor_mask
    with contextlib.redirect_stdout(io.StringIO()):
        return FieldSolver3D(params, toggles, V, Ex, Ey, Ez, [], [], [], mask, num_antennas=0)


def _single_ion_ensemble(position: np.ndarray) -> SimpleNamespace:
    """Minimal ensemble stand-in for FieldSolver3D.solve_poisson with one active ion."""
    empty = np.zeros(0)
    return SimpleNamespace(
        cloud_injected=True,
        x_e=empty, y_e=empty, z_e=empty, active_e=np.zeros(0, dtype=bool),
        x_i=np.array([position[0]]), y_i=np.array([position[1]]), z_i=np.array([position[2]]),
        active_i=np.ones(1, dtype=bool),
    )


class TestPhysicsConsistency(unittest.TestCase):

    def test_homogeneous_plasma_temperature(self):
        """
        A Maxwellian with temperature T has <v_x^2> = k_B T / m per component. The temperature
        measured from the sampled velocities must equal the configured solar wind temperatures.
        """
        np.random.seed(3)
        Te_eV, Ti_eV = 15.0, 10.0
        params = _cubic_params(
            8, 1.0,
            num_macroparticles=200000,
            plasma_injection_mode="homogeneous",
            solar_wind_electron_temp_eV=Te_eV,
            solar_wind_ion_temp_eV=Ti_eV,
        )
        ens = ParticleEnsemble(params, np.zeros((8, 8, 8), dtype=bool), num_antennas=0)

        Te_meas = m_e * np.mean(np.concatenate([ens.vx_e, ens.vy_e, ens.vz_e])**2) / e
        Ti_meas = params.m_i * np.mean(np.concatenate([ens.vx_i, ens.vy_i, ens.vz_i])**2) / e

        self.assertAlmostEqual(Te_meas / Te_eV, 1.0, delta=0.02)
        self.assertAlmostEqual(Ti_meas / Ti_eV, 1.0, delta=0.02)

    def test_homogeneous_plasma_avoids_spacecraft(self):
        """
        Homogeneous plasma must not be loaded into the spacecraft. Without an analytical body the
        conductor is the set of mask nodes, each representing the cube of half-width dx/2 around it.
        """
        np.random.seed(5)
        N = 10
        params = _cubic_params(N, 1.0, num_macroparticles=20000, plasma_injection_mode="homogeneous")
        X, Y, Z = np.meshgrid(params.x_grid, params.y_grid, params.z_grid, indexing='ij')
        mask = X**2 + Y**2 + Z**2 <= 0.5**2
        ens = ParticleEnsemble(params, mask, num_antennas=0)

        idx = [
            np.clip(np.rint((c - g[0]) / h).astype(int), 0, N - 1)
            for c, g, h in ((ens.x_e, params.x_grid, params.dx),
                            (ens.y_e, params.y_grid, params.dy),
                            (ens.z_e, params.z_grid, params.dz))
        ]
        self.assertFalse(np.any(mask[idx[0], idx[1], idx[2]]))
        np.testing.assert_array_equal(ens.x_e, ens.x_i)

    def test_impact_cloud_half_maxwellian(self):
        """
        The impact cloud expands into the half-space above the surface. Its velocities must all
        point away from the surface (v . n >= 0), keep m <v_k^2> = k_B T in every component
        (separately for electrons and ions with their own temperatures), and have the
        half-Maxwellian mean expansion velocity <v . n> = sqrt(2 k_B T / (pi m)).
        """
        np.random.seed(7)
        Te_eV, Ti_eV = 4.0, 1.0
        normal = np.array([-1.0, 2.0, 0.5])
        params = _cubic_params(
            8, 1.0,
            num_macroparticles=200000,
            plasma_injection_mode="point_cloud",
            impact=ImpactConfig(electron_temperature_eV=Te_eV, ion_temperature_eV=Ti_eV),
            impact_normal=list(normal),
        )
        ens = ParticleEnsemble(params, np.zeros((8, 8, 8), dtype=bool), num_antennas=0)
        n = normal / np.linalg.norm(normal)

        for label, v, m, T in (
            ("electrons", np.column_stack([ens.vx_e, ens.vy_e, ens.vz_e]), m_e, Te_eV),
            ("ions", np.column_stack([ens.vx_i, ens.vy_i, ens.vz_i]), params.m_i, Ti_eV),
        ):
            with self.subTest(species=label):
                v_n = v @ n
                self.assertGreaterEqual(v_n.min(), 0.0)
                np.testing.assert_allclose(m * np.mean(v**2, axis=0) / e, T, rtol=0.02)
                sigma = np.sqrt(e * T / m)
                self.assertAlmostEqual(np.mean(v_n) / (sigma * np.sqrt(2.0 / np.pi)), 1.0, delta=0.01)

    def test_impact_cloud_ion_cone_beam(self):
        """
        With ion_drift_speed_m_s > 0 the impact ions form a conical beam around the surface normal:
          - cold beam: |v| = v_d, every direction inside the cone, directions uniform in solid angle
            (cos(theta) uniform on [cos(theta_max), 1], hence <cos(theta)> = (1 + cos(theta_max)) / 2);
          - warm beam: the thermal spread adds 3 k_B T / m to <|v|^2> = v_d^2 + 3 k_B T / m (the
            specular reflection at the surface keeps |v|), and all velocities still leave the surface.
        Electrons stay a half-Maxwellian.
        """
        normal = np.array([0.3, -1.0, 0.2])
        n = normal / np.linalg.norm(normal)
        v_d, theta_max = 1.0e4, 30.0
        cos_max = np.cos(np.deg2rad(theta_max))

        def ions(Ti_eV: float, seed: int):
            np.random.seed(seed)
            params = _cubic_params(
                8, 1.0,
                num_macroparticles=200000,
                plasma_injection_mode="point_cloud",
                impact=ImpactConfig(electron_temperature_eV=2.0, ion_temperature_eV=Ti_eV,
                                    ion_drift_speed_m_s=v_d, ion_cone_half_angle_deg=theta_max),
                impact_normal=list(normal),
            )
            ens = ParticleEnsemble(params, np.zeros((8, 8, 8), dtype=bool), num_antennas=0)
            return params, ens, np.column_stack([ens.vx_i, ens.vy_i, ens.vz_i])

        _, ens, v = ions(1e-8, 11)
        speed = np.linalg.norm(v, axis=1)
        cos_t = (v @ n) / speed
        np.testing.assert_allclose(speed, v_d, rtol=1e-3)
        self.assertGreaterEqual(cos_t.min(), cos_max - 1e-3)
        self.assertAlmostEqual(np.mean(cos_t) / (0.5 * (1.0 + cos_max)), 1.0, delta=1e-3)
        self.assertAlmostEqual(np.mean(cos_t < 0.5 * (1.0 + cos_max)), 0.5, delta=0.01)
        v_e_n = np.column_stack([ens.vx_e, ens.vy_e, ens.vz_e]) @ n
        self.assertAlmostEqual(np.mean(v_e_n) / (np.sqrt(e * 2.0 / m_e) * np.sqrt(2.0 / np.pi)), 1.0, delta=0.01)

        Ti_eV = 5.0
        params, _, v = ions(Ti_eV, 13)
        self.assertGreaterEqual((v @ n).min(), 0.0)
        expected = v_d**2 + 3.0 * e * Ti_eV / params.m_i
        self.assertAlmostEqual(np.mean(np.sum(v**2, axis=1)) / expected, 1.0, delta=0.01)

    def test_energy_conservation_in_static_potential(self):
        """
        An ion released at rest in the static potential of a charged sphere, V = V0 R / r, must
        convert potential into kinetic energy: 1/2 m v^2 = q [V(r0) - V(r)]. Tests the background
        field path (E = -grad V on the grid, trilinear interpolation, leapfrog push).
        """
        V0, R = 10.0, 0.3
        params = _cubic_params(41, 2.0, num_macroparticles=1, time_step_s=2e-8, ion_mass_amu=1.0)
        X, Y, Z = np.meshgrid(params.x_grid, params.y_grid, params.z_grid, indexing='ij')
        r = np.sqrt(X**2 + Y**2 + Z**2)
        sphere = r <= R
        V_bg = V0 * R / np.maximum(r, R)

        toggles = SimulationToggles3D(enable_spis_background_field=True, enable_plasma_self_field=False)
        fs = _field_solver(params, toggles, V_bg, sphere)
        ens = ParticleEnsemble(params, sphere, num_antennas=0, toggles=toggles)
        ens.cloud_injected = True
        r0 = np.array([0.55, 0.13, -0.07])
        ens.x_i[0], ens.y_i[0], ens.z_i[0] = r0
        ens.vx_i[0] = ens.vy_i[0] = ens.vz_i[0] = 0.0
        ens.active_i[0] = True

        r_end = np.linalg.norm(r0)
        for _ in range(6000):
            ens.push_species(
                ens.x_i, ens.y_i, ens.z_i, ens.vx_i, ens.vy_i, ens.vz_i,
                ens.active_i, ens.was_outside_i, params.m_i, e, 1e-15,
                fs, [], [], [], [], boundary_mode="open",
            )
            r_end = np.sqrt(ens.x_i[0]**2 + ens.y_i[0]**2 + ens.z_i[0]**2)
            if r_end > 1.6:
                break

        self.assertGreater(r_end, 1.6, "Ion was not accelerated away from the sphere")
        kinetic_eV = 0.5 * params.m_i * (ens.vx_i[0]**2 + ens.vy_i[0]**2 + ens.vz_i[0]**2) / e
        potential_drop_V = V0 * R / np.linalg.norm(r0) - V0 * R / r_end
        self.assertAlmostEqual(kinetic_eV / potential_drop_V, 1.0, delta=0.02)

    def test_green_reciprocity_self_field_vs_weighting_potential(self):
        """
        Green's reciprocity: a point charge q at r near a grounded conductor induces on it the
        charge Q_ind = -q * Vw(r), where Vw is the weighting potential (1 V on the conductor, 0 V on
        all other boundaries). Q_ind is obtained from the self-field computed by
        FieldSolver3D.solve_poisson (discrete Gauss law on the conductor nodes) and compared with
        Vw from the geometry Laplace solver. This ties the plasma self-field to the antenna signal.
        """
        params = _cubic_params(33, 2.0, num_macroparticles=1, total_impact_charge_C=1e-15)
        X, Y, Z = np.meshgrid(params.x_grid, params.y_grid, params.z_grid, indexing='ij')
        conductor = X**2 + Y**2 + Z**2 <= 0.4**2

        toggles = SimulationToggles3D(enable_spis_background_field=False, enable_plasma_self_field=True)
        fs = _field_solver(params, toggles, conductor_mask=conductor)
        with contextlib.redirect_stdout(io.StringIO()):
            Vw = solve_laplace_dirichlet_3d(
                params.Nx, params.Ny, params.Nz, params.dx, params.dy, params.dz,
                conductor, [conductor.astype(float)]
            )[0]

        dV = params.dx * params.dy * params.dz
        for position in ([0.8, 0.05, 0.03], [0.0, 1.1, 0.2]):
            pos = np.array(position)
            fs.solve_poisson(_single_ion_ensemble(pos))
            V = fs.V_self_grid
            lap = np.zeros_like(V)
            for axis, h in ((0, params.dx), (1, params.dy), (2, params.dz)):
                lap[1:-1, 1:-1, 1:-1] += (
                    np.roll(V, 1, axis) + np.roll(V, -1, axis) - 2.0 * V
                )[1:-1, 1:-1, 1:-1] / h**2
            Q_induced = -eps_0 * np.sum(lap[conductor]) * dV
            vw = fs.interp_field(pos[:1], pos[1:2], pos[2:], Vw)[0]
            with self.subTest(position=position):
                self.assertGreater(vw, 0.05)
                self.assertAlmostEqual(Q_induced / params.q_macro, -vw, delta=0.01 * vw)

    def test_no_self_force_on_isolated_particle(self):
        """
        An isolated macro-particle far from all boundaries must not accelerate itself. The self
        field interpolated at its own position must be negligible compared with the Coulomb field
        of the same charge at a distance of one cell.
        """
        params = _cubic_params(33, 2.0, num_macroparticles=1, total_impact_charge_C=1e-15)
        toggles = SimulationToggles3D(enable_spis_background_field=False, enable_plasma_self_field=True)
        fs = _field_solver(params, toggles)
        E_ref = params.q_macro / (4.0 * np.pi * eps_0 * params.dx**2)

        rng = np.random.default_rng(3)
        worst = 0.0
        for _ in range(10):
            pos = rng.uniform(0.0, params.dx, 3)
            fs.solve_poisson(_single_ion_ensemble(pos))
            E_self = [
                fs.interp_field(pos[:1], pos[1:2], pos[2:], F)[0]
                for F in (fs.Ex_self, fs.Ey_self, fs.Ez_self)
            ]
            worst = max(worst, float(np.linalg.norm(E_self)) / E_ref)

        self.assertLess(worst, 0.01)


if __name__ == "__main__":
    unittest.main()
