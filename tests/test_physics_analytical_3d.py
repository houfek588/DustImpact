# -*- coding: utf-8 -*-
"""
Level 3 Rigorous Analytical 3D Physical Verification.

Every test drives the production code path (ParticleEnsemble.push_species, FieldSolver3D,
DustImpactSimulation3D.step) and compares it with an exact analytical result:
1. 3D Ramo-Shockley flyby: induced current of a charge passing a spherical antenna.
2. Gauss's law for the self-field of a Gaussian charge cloud deposited by CIC.
3. Method of Manufactured Solutions (MMS): O(dx^2) convergence of FieldSolver3D.solve_poisson.
4. Poisson solver residual: the linear system must actually be solved.
5. Self-consistent electron plasma (Langmuir) oscillation at omega_pe.
"""

import contextlib
import io
import os
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from dust_impact.physics.constants import e, m_e, eps_0
from dust_impact.common.io import ensure_dir
from dust_impact.solver.config_loader import SimulationParams3D, SimulationToggles3D
from dust_impact.solver.ensemble import ParticleEnsemble
from dust_impact.solver.field_solver import FieldSolver3D
from dust_impact.solver.sim_core import DustImpactSimulation3D


def _cubic_params(N: int, L: float, **kwargs) -> SimulationParams3D:
    """Simulation parameters on a cubic grid [-L, L]^3 with N nodes per axis."""
    return SimulationParams3D(
        grid_nodes_x=N, grid_nodes_y=N, grid_nodes_z=N,
        domain_half_length_x_m=L, domain_half_length_y_m=L, domain_half_length_z_m=L,
        **kwargs
    )


def _field_solver(params: SimulationParams3D, conductor_mask=None, **toggle_kwargs) -> FieldSolver3D:
    """FieldSolver3D without background fields or antennas."""
    shape = (params.Nx, params.Ny, params.Nz)
    zeros = np.zeros(shape)
    mask = np.zeros(shape, dtype=bool) if conductor_mask is None else conductor_mask
    toggles = SimulationToggles3D(
        enable_spis_background_field=False, enable_plasma_self_field=True, **toggle_kwargs
    )
    with contextlib.redirect_stdout(io.StringIO()):
        return FieldSolver3D(params, toggles, zeros, zeros, zeros, zeros, [], [], [], mask, num_antennas=0)


def _ion_only_ensemble(positions: np.ndarray) -> SimpleNamespace:
    """Minimal ensemble stand-in for FieldSolver3D.solve_poisson containing only active ions."""
    empty = np.zeros(0)
    return SimpleNamespace(
        cloud_injected=True,
        x_e=empty, y_e=empty, z_e=empty, active_e=np.zeros(0, dtype=bool),
        x_i=positions[:, 0].copy(), y_i=positions[:, 1].copy(), z_i=positions[:, 2].copy(),
        active_i=np.ones(len(positions), dtype=bool),
    )


class TestPhysicsAnalytical3D(unittest.TestCase):
    """
    Validation test suite comparing the 3D solver directly against exact analytical formulas.
    Saves validation text reports to outputs/tests/
    """

    @classmethod
    def setUpClass(cls):
        cls.report_dir = "outputs/tests"
        cls.report_path = os.path.join(cls.report_dir, "physics_analytical_3d_results.txt")
        ensure_dir(cls.report_path)
        with open(cls.report_path, "w", encoding="utf-8") as f:
            f.write("=================================================================\n")
            f.write("   DUSTIMPACT - 3D RIGOROUS ANALYTICAL VERIFICATION REPORT       \n")
            f.write("=================================================================\n\n")

    def _log_result(self, test_name: str, status: str, details: str):
        with open(self.report_path, "a", encoding="utf-8") as f:
            f.write(f"[{status}] {test_name}\n")
            f.write(f"  {details}\n\n")

    # =========================================================================
    # Test 1: 3D Ramo-Shockley Flyby Induced Current
    # =========================================================================
    def test_ramo_shockley_3d_analytical_flyby(self):
        """
        Validates the induced current computed by ParticleEnsemble.push_species for an ion moving
        along r(t) = (x0 + v t, d, 0) past a spherical antenna of radius a at the origin
        (Vw = a / r). Exact result:
            I_exact(t) = q * dVw/dt = -q * a * v * x(t) / [x(t)^2 + d^2]^(3/2)
        """
        a_ant, d_impact, v_x = 0.1, 0.4, 1.0e5
        q_macro = 1.602e-12
        params = _cubic_params(49, 1.2, num_macroparticles=1, time_step_s=2.0e-8)
        X, Y, Z = np.meshgrid(params.x_grid, params.y_grid, params.z_grid, indexing='ij')
        R = np.sqrt(X**2 + Y**2 + Z**2)
        ant_mask = R <= a_ant
        Vw = np.where(ant_mask, 1.0, a_ant / np.maximum(R, 1e-12))
        zeros = np.zeros_like(Vw)

        toggles = SimulationToggles3D(enable_spis_background_field=False, enable_plasma_self_field=False)
        fs = _field_solver(params, ant_mask)
        ens = ParticleEnsemble(params, np.zeros_like(ant_mask), num_antennas=1, toggles=toggles)
        ens.cloud_injected = True
        ens.x_i[0], ens.y_i[0], ens.z_i[0] = -0.8, d_impact, 0.0
        ens.vx_i[0], ens.vy_i[0], ens.vz_i[0] = v_x, 0.0, 0.0
        ens.active_i[0] = True
        ens.was_outside_i[:, 0] = True

        steps = int(1.6 / (v_x * params.dt))
        I_num = np.zeros(steps)
        x_mid = np.zeros(steps)
        for step in range(steps):
            x_before = ens.x_i[0]
            # Heavy mass: no acceleration, the trajectory is a straight line
            _, ind = ens.push_species(
                ens.x_i, ens.y_i, ens.z_i, ens.vx_i, ens.vy_i, ens.vz_i,
                ens.active_i, ens.was_outside_i, 1.0, e, q_macro,
                fs, [ant_mask], [zeros], [zeros], [zeros],
                enable_collection=True, boundary_mode="open", Vw_list=[Vw],
            )
            I_num[step] = ind[0]
            x_mid[step] = 0.5 * (x_before + ens.x_i[0])

        I_exact = -q_macro * a_ant * v_x * x_mid / (x_mid**2 + d_impact**2)**1.5
        peak_exact = (2.0 / (3.0 * np.sqrt(3.0))) * q_macro * a_ant * v_x / d_impact**2
        rel_peak_err = abs(np.max(np.abs(I_num)) - peak_exact) / peak_exact
        correlation = np.corrcoef(I_num, I_exact)[0, 1]

        self.assertTrue(ens.active_i[0], "The ion must not hit the antenna in a flyby")
        # Measured: correlation 0.9993, peak error 0.24 % (trilinear interpolation of Vw, 49^3 grid)
        self.assertGreater(correlation, 0.998)
        self.assertLess(rel_peak_err, 0.01)

        self._log_result(
            "3D Ramo-Shockley Flyby Induced Current (push_species)",
            "PASS",
            f"Correlation = {correlation:.5f} | Rel Peak Error = {rel_peak_err * 100:.2f}%"
        )

    # =========================================================================
    # Test 2: Gauss's law for the CIC-deposited self-field
    # =========================================================================
    def test_gauss_law_gaussian_cloud(self):
        """
        Validates FieldSolver3D.solve_poisson (CIC deposition + Poisson solve + E = -grad V) via
        Gauss's law: the flux of E_self through any closed box around a charge cloud equals
        Q_enclosed / eps_0, independently of the grounded outer boundary. The smaller box cuts the
        cloud at 3.2 sigma, so the enclosed rather than the total charge is the reference.
        """
        rng = np.random.default_rng(1)
        Np = 200000
        params = _cubic_params(41, 2.0, num_macroparticles=Np, total_impact_charge_C=1e-12)
        fs = _field_solver(params)
        positions = rng.normal(0.0, 0.25, (Np, 3))
        fs.solve_poisson(_ion_only_ensemble(positions))

        centre = params.Nx // 2
        for half_cells in (8, 12):
            i0, i1 = centre - half_cells, centre + half_cells
            sl = slice(i0, i1 + 1)
            w = np.ones(i1 - i0 + 1)
            w[0] = w[-1] = 0.5  # trapezoidal rule on the face
            W = np.outer(w, w) * params.dx * params.dx
            flux = (
                np.sum((fs.Ex_self[i1, sl, sl] - fs.Ex_self[i0, sl, sl]) * W)
                + np.sum((fs.Ey_self[sl, i1, sl] - fs.Ey_self[sl, i0, sl]) * W)
                + np.sum((fs.Ez_self[sl, sl, i1] - fs.Ez_self[sl, sl, i0]) * W)
            )
            # Enclosed charge: macro-particles inside the box
            half_width = half_cells * params.dx
            inside = np.all(np.abs(positions) < half_width, axis=1)
            ratio = flux * eps_0 / (np.sum(inside) * params.q_macro)
            with self.subTest(box_half_width_m=half_cells * params.dx):
                # Measured |ratio - 1| ~ 1e-4
                self.assertAlmostEqual(ratio, 1.0, delta=2e-3)

        self._log_result(
            "Gauss's Law for CIC Self-Field (FieldSolver3D)",
            "PASS",
            f"eps_0 * flux / Q at box half-width {half_cells * params.dx:.2f} m = {ratio:.4f}"
        )

    # =========================================================================
    # Tests 3 and 4: Poisson solver accuracy (production solve_poisson path)
    # =========================================================================
    def _solve_manufactured(self, N: int):
        """
        Solves -Laplacian(V) = rho / eps_0 through FieldSolver3D.solve_poisson for the manufactured
        solution V = V0 cos(kx) cos(ky) cos(kz), k = pi / (2L), which vanishes on the boundary.
        The CIC deposition is replaced by the exact nodal charge so that only the solver is tested.
        """
        L, V0 = 1.0, 10.0
        params = _cubic_params(N, L, num_macroparticles=1)
        fs = _field_solver(params)
        X, Y, Z = np.meshgrid(params.x_grid, params.y_grid, params.z_grid, indexing='ij')
        k = np.pi / (2.0 * L)
        V_exact = V0 * np.cos(k * X) * np.cos(k * Y) * np.cos(k * Z)
        rho = eps_0 * 3.0 * k**2 * V_exact
        q_nodes = rho * params.dx * params.dy * params.dz
        with mock.patch(
            'dust_impact.solver.field_solver.deposit_charge_cic_3d',
            side_effect=[np.zeros_like(q_nodes), q_nodes],
        ):
            fs.solve_poisson(_ion_only_ensemble(np.zeros((0, 3))))
        return params, fs, V_exact, rho

    def test_poisson_solver_residual(self):
        """
        The production Poisson solve must reduce the residual of -Laplacian(V) = rho / eps_0 by at
        least 1e-8, evaluated independently with the 7-point stencil on the full grid.
        """
        params, fs, V_exact, rho = self._solve_manufactured(33)
        h2 = params.dx**2
        V = fs.V_self_grid
        lap = (
            V[2:, 1:-1, 1:-1] + V[:-2, 1:-1, 1:-1] + V[1:-1, 2:, 1:-1] + V[1:-1, :-2, 1:-1]
            + V[1:-1, 1:-1, 2:] + V[1:-1, 1:-1, :-2] - 6.0 * V[1:-1, 1:-1, 1:-1]
        ) / h2
        b = rho[1:-1, 1:-1, 1:-1] / eps_0
        rel_residual = np.linalg.norm(lap + b) / np.linalg.norm(b)
        self.assertLess(rel_residual, 1e-8)

    def test_poisson_solver_reports_non_convergence(self):
        """An unconverged Poisson solve must raise instead of silently returning an inaccurate field."""
        params = _cubic_params(17, 1.0, num_macroparticles=1)
        fs = _field_solver(params)
        if fs.amg_solver is None:
            self.skipTest("Direct LU solver is used; non-convergence cannot occur")
        fs.POISSON_MAXITER = 1
        fs.POISSON_RTOL = 1e-14
        q_nodes = np.zeros((params.Nx, params.Ny, params.Nz))
        q_nodes[8, 8, 8] = 1e-12
        with mock.patch(
            'dust_impact.solver.field_solver.deposit_charge_cic_3d',
            side_effect=[np.zeros_like(q_nodes), q_nodes],
        ):
            with self.assertRaises(RuntimeError):
                fs.solve_poisson(_ion_only_ensemble(np.zeros((0, 3))))

    def test_mms_poisson_3d_convergence_rate(self):
        """
        Validates 2nd-order spatial convergence O(dx^2) of FieldSolver3D.solve_poisson using the
        Method of Manufactured Solutions on grids with N = 17 and N = 33 nodes.
        """
        errors = []
        for N in (17, 33):
            params, fs, V_exact, _ = self._solve_manufactured(N)
            errors.append((params.dx, np.max(np.abs(fs.V_self_grid - V_exact))))
        (h1, err1), (h2, err2) = errors
        conv_order = np.log(err1 / err2) / np.log(h1 / h2)

        self._log_result(
            "Method of Manufactured Solutions (FieldSolver3D)",
            "PASS",
            f"N=17: err = {err1:.4e} | N=33: err = {err2:.4e} | order p = {conv_order:.3f} (theory 2)"
        )
        self.assertGreater(conv_order, 1.85)
        self.assertLess(conv_order, 2.15)

    def test_mms_electric_field_convergence_rate(self):
        """
        The particles are pushed by E = -grad V, not by V. The self-field returned by
        FieldSolver3D.solve_poisson must converge to the exact field of the manufactured solution
        with second order on the whole grid, including the boundary nodes.
        """
        errors = []
        for N in (17, 33):
            params, fs, V_exact, _ = self._solve_manufactured(N)
            X, Y, Z = np.meshgrid(params.x_grid, params.y_grid, params.z_grid, indexing='ij')
            k = np.pi / 2.0
            Ex = 10.0 * k * np.sin(k * X) * np.cos(k * Y) * np.cos(k * Z)
            Ez = 10.0 * k * np.cos(k * X) * np.cos(k * Y) * np.sin(k * Z)
            err = max(np.max(np.abs(fs.Ex_self - Ex)), np.max(np.abs(fs.Ez_self - Ez)))
            errors.append((params.dx, err))
        (h1, err1), (h2, err2) = errors
        conv_order = np.log(err1 / err2) / np.log(h1 / h2)
        self.assertGreater(conv_order, 1.85)
        self.assertLess(err2 / (10.0 * k), 2e-3)

    # =========================================================================
    # Test 5: Self-consistent electron plasma (Langmuir) oscillations
    # =========================================================================
    def test_plasma_oscillation_frequency(self):
        """
        Validates the full PIC cycle (CIC deposition -> Poisson -> E = -grad V -> interpolation ->
        leapfrog push) against the cold-plasma Langmuir frequency
            omega_pe = sqrt(n_0 e^2 / (eps_0 m_e)).
        Cold electrons in a uniform neutralising ion background are displaced by
        delta = delta_0 grad(psi)/k with psi = cos(kx) cos(ky) cos(kz), k = pi / (2L); psi vanishes
        on the grounded walls, so this is an exact electrostatic eigenmode oscillating at omega_pe
        for any wavenumber in a cold plasma.
        """
        np.random.seed(5)
        n0, L, N = 1e12, 1.0, 21
        omega_pe = np.sqrt(n0 * e**2 / (eps_0 * m_e))
        dt = 2.0 * np.pi / omega_pe / 40.0
        params = _cubic_params(
            N, L,
            num_macroparticles=40000,
            plasma_injection_mode="homogeneous",
            solar_wind_density_m3=n0,
            solar_wind_electron_temp_eV=1e-4,
            solar_wind_ion_temp_eV=1e-4,
            ion_mass_amu=1e4,
            time_step_s=dt,
            num_time_steps=120,
            impact_time_delay_s=0.0,
        )
        toggles = SimulationToggles3D(
            enable_spis_background_field=False,
            enable_antenna_particle_collection=False,
            enable_plasma_self_field=True,
        )
        shape = (N, N, N)
        zeros = np.zeros(shape)
        with contextlib.redirect_stdout(io.StringIO()):
            sim = DustImpactSimulation3D(
                params, toggles, zeros, [zeros.copy()], zeros, zeros, zeros,
                [zeros], [zeros], [zeros], [np.zeros(shape, dtype=bool)], np.zeros(shape, dtype=bool)
            )
        ens = sim.particles
        for arr in (ens.vx_e, ens.vy_e, ens.vz_e, ens.vx_i, ens.vy_i, ens.vz_i):
            arr[:] = 0.0

        k, delta_0 = np.pi / (2.0 * L), 0.02
        x0, y0, z0 = ens.x_e.copy(), ens.y_e.copy(), ens.z_e.copy()
        gx = -np.sin(k * x0) * np.cos(k * y0) * np.cos(k * z0)
        gy = -np.cos(k * x0) * np.sin(k * y0) * np.cos(k * z0)
        gz = -np.cos(k * x0) * np.cos(k * y0) * np.sin(k * z0)
        ens.x_e[:] += delta_0 * gx
        ens.y_e[:] += delta_0 * gy
        ens.z_e[:] += delta_0 * gz

        # Projection of the electron displacement onto the eigenmode
        amplitude = np.zeros(params.steps)
        with contextlib.redirect_stdout(io.StringIO()):
            for step in range(params.steps):
                amplitude[step] = np.sum((ens.x_e - x0) * gx + (ens.y_e - y0) * gy + (ens.z_e - z0) * gz)
                sim.step(step)

        from scipy.optimize import curve_fit
        t_arr = np.arange(params.steps) * dt
        popt, _ = curve_fit(
            lambda t, A, w, phi, c: A * np.cos(w * t + phi) + c,
            t_arr, amplitude, p0=[amplitude[0], omega_pe, 0.0, 0.0]
        )
        omega_fit = abs(float(popt[1]))
        rel_freq_err = abs(omega_fit - omega_pe) / omega_pe

        self.assertLess(rel_freq_err, 0.03)

        self._log_result(
            "Self-Consistent Langmuir Oscillation (3D PIC)",
            "PASS",
            f"omega_pe = {omega_pe:.4e} rad/s | fitted = {omega_fit:.4e} rad/s | "
            f"Relative Error = {rel_freq_err * 100:.2f}%"
        )

    # =========================================================================
    # Test 6: Energy conservation of the self-consistent PIC cycle
    # =========================================================================
    def test_energy_conservation_resolved_thermal_plasma(self):
        """
        In a thermal plasma whose Debye length is resolved by the grid (dx / lambda_D ~ 0.4) the total
        energy, kinetic energy of both species plus the field energy eps_0/2 * sum |E|^2 dV, must be
        conserved by the PIC cycle. Particles are kept in the box by the periodic particle boundary
        so that no energy leaves the domain.

        For dx >> lambda_D the same setup heats up by the finite-grid instability (T_e grows from 1 eV
        to 1.7 eV within 5 plasma periods at dx / lambda_D = 13); this is a resolution limit of the
        scheme, not a defect, and dense impact clouds must be judged with it in mind.
        """
        np.random.seed(5)
        n0, Te_eV, L, N = 1e9, 1.0, 1.0, 21
        omega_pe = np.sqrt(n0 * e**2 / (eps_0 * m_e))
        dt = 2.0 * np.pi / omega_pe / 40.0
        params = _cubic_params(
            N, L,
            num_macroparticles=20000,
            plasma_injection_mode="homogeneous",
            solar_wind_density_m3=n0,
            solar_wind_electron_temp_eV=Te_eV,
            solar_wind_ion_temp_eV=Te_eV,
            ion_mass_amu=1.0,
            time_step_s=dt,
            num_time_steps=120,
            impact_time_delay_s=0.0,
        )
        lambda_D = np.sqrt(eps_0 * Te_eV / (n0 * e))
        self.assertLess(params.dx / lambda_D, 0.5)

        toggles = SimulationToggles3D(
            enable_spis_background_field=False,
            enable_antenna_particle_collection=False,
            enable_plasma_self_field=True,
            particle_boundary_mode="periodic",
        )
        shape = (N, N, N)
        zeros = np.zeros(shape)
        with contextlib.redirect_stdout(io.StringIO()):
            sim = DustImpactSimulation3D(
                params, toggles, zeros, [zeros.copy()], zeros, zeros, zeros,
                [zeros], [zeros], [zeros], [np.zeros(shape, dtype=bool)], np.zeros(shape, dtype=bool)
            )
        ens, fs = sim.particles, sim.field_solver
        weight = params.q_macro / e
        dV = params.dx * params.dy * params.dz

        energy = np.zeros(params.steps)
        with contextlib.redirect_stdout(io.StringIO()):
            for step in range(params.steps):
                sim.step(step)
                K_e = 0.5 * weight * m_e * np.sum(ens.vx_e**2 + ens.vy_e**2 + ens.vz_e**2)
                K_i = 0.5 * weight * params.m_i * np.sum(ens.vx_i**2 + ens.vy_i**2 + ens.vz_i**2)
                W = 0.5 * eps_0 * np.sum(fs.Ex_self**2 + fs.Ey_self**2 + fs.Ez_self**2) * dV
                energy[step] = K_e + K_i + W

        self.assertTrue(np.all(ens.active_e) and np.all(ens.active_i))
        rel_dev = np.max(np.abs(energy - energy[0])) / energy[0]
        # Measured 3e-4; a self-field scaled by 3 gives 7e-3
        self.assertLess(rel_dev, 2e-3)

        self._log_result(
            "Energy Conservation, Resolved Thermal Plasma (3D PIC)",
            "PASS",
            f"dx/lambda_D = {params.dx / lambda_D:.2f} | max relative energy deviation = {rel_dev:.2e}"
        )


if __name__ == "__main__":
    unittest.main()
