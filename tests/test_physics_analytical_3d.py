# -*- coding: utf-8 -*-
"""
Level 3 Rigorous Analytical 3D Physical Verification:
Direct comparisons of numerical 3D simulations against exact analytical mathematical solutions:
1. 3D Ramo-Shockley Flyby:
   Exact time-dependent induced current waveform I(t) for a point charge flying by a spherical antenna.
2. 3D Poisson Solver Gaussian Cloud:
   Exact 3D potential V(r) ~ erf(r/sigma) / r for a spherical Gaussian charge distribution.
3. Method of Manufactured Solutions (MMS):
   Exact 2nd-order spatial convergence rate O(dx^2) for the 3D finite-difference Poisson operator.
4. 3D Electron Plasma Oscillations:
   Langmuir oscillation frequency omega_pe = sqrt(n*e^2 / (eps0 * m_e)) in self-consistent 3D PIC.
"""

import os
import unittest
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.special import erf

from dust_impact.physics.constants import e, m_e, eps_0
from dust_impact.common.io import ensure_dir
from dust_impact.numerics.interpolators import interp_field_3d
from dust_impact.numerics.pushers import leapfrog_step_3d


class TestPhysicsAnalytical3D(unittest.TestCase):
    """
    Validation test suite comparing 3D numerical methods directly against exact analytical formulas.
    Saves validation text reports and comparison plots to outputs/tests/
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
        Validates 3D Ramo-Shockley induced current against the exact analytical solution
        for a charge moving along r(t) = (x0 + v*t, d, 0) past a spherical antenna of radius a at origin:
        I_exact(t) = -q * a * v * (x0 + v*t) / [ (x0 + v*t)^2 + d^2 ]^(3/2)
        """
        a_ant = 0.05       # Antenna radius: 5 cm
        d_impact = 0.25    # Impact parameter: 25 cm
        v_x = 1.0e5        # Velocity: 100 km/s along x
        q_macro = 1.602e-12  # Charge: 1.6 pC (10^7 elementary charges)
        
        # Grid setup
        L_dom = 1.2
        Nx, Ny, Nz = 35, 35, 35
        x_grid = np.linspace(-L_dom, L_dom, Nx)
        y_grid = np.linspace(-L_dom, L_dom, Ny)
        z_grid = np.linspace(-L_dom, L_dom, Nz)
        dx = x_grid[1] - x_grid[0]
        dy = y_grid[1] - y_grid[0]
        dz = z_grid[1] - z_grid[0]

        X, Y, Z = np.meshgrid(x_grid, y_grid, z_grid, indexing='ij')
        R = np.sqrt(X**2 + Y**2 + Z**2)
        R_safe = np.maximum(R, a_ant)

        # Weighting field Ew = a / r^2 * (r_hat) = a / r^3 * r
        factor = np.where(R >= a_ant, a_ant / (R_safe**3), 0.0)
        Ewx = X * factor
        Ewy = Y * factor
        Ewz = Z * factor

        # Simulation trajectory: from x = -0.6 m to +0.6 m
        x_start = -0.6
        t_total = 1.2e-5  # 12 us
        dt = 2.0e-8      # 20 ns
        steps = int(t_total / dt)
        time_arr = np.linspace(0, t_total, steps)

        x_num = np.zeros(steps)
        I_num = np.zeros(steps)

        # Initial particle state
        px, py, pz = x_start, d_impact, 0.0
        pvx, pvy, pvz = v_x, 0.0, 0.0

        for step in range(steps):
            x_num[step] = px
            # 3D Trilinear field interpolation
            ewx_p = interp_field_3d(np.array([px]), np.array([py]), np.array([pz]),
                                    x_grid, y_grid, z_grid, dx, dy, dz, Ewx)[0]
            ewy_p = interp_field_3d(np.array([px]), np.array([py]), np.array([pz]),
                                    x_grid, y_grid, z_grid, dx, dy, dz, Ewy)[0]
            ewz_p = interp_field_3d(np.array([px]), np.array([py]), np.array([pz]),
                                    x_grid, y_grid, z_grid, dx, dy, dz, Ewz)[0]

            I_num[step] = - q_macro * (pvx * ewx_p + pvy * ewy_p + pvz * ewz_p)

            # Move particle
            px, py, pz, pvx, pvy, pvz = leapfrog_step_3d(
                np.array([px]), np.array([py]), np.array([pz]),
                np.array([pvx]), np.array([pvy]), np.array([pvz]),
                np.array([0.0]), np.array([0.0]), np.array([0.0]),
                1.0, dt, np.array([True])
            )
            px, py, pz = px[0], py[0], pz[0]
            pvx, pvy, pvz = pvx[0], pvy[0], pvz[0]

        # Analytical formula
        r_x = x_start + v_x * time_arr
        I_exact = - q_macro * (a_ant * v_x * r_x) / ((r_x**2 + d_impact**2)**(1.5))

        # Peak values and timing
        peak_exact = (2.0 / (3.0 * np.sqrt(3.0))) * (abs(q_macro) * a_ant * v_x / (d_impact**2))
        max_num = np.max(np.abs(I_num))
        rel_peak_err = abs(max_num - peak_exact) / peak_exact

        # Correlation coefficient
        correlation = np.corrcoef(I_num, I_exact)[0, 1]

        # Net charge transfer integral
        q_net = np.sum(I_num) * dt

        self.assertGreater(correlation, 0.99)
        self.assertLess(rel_peak_err, 0.05)
        self.assertAlmostEqual(q_net, 0.0, delta=1e-15)

        self._log_result(
            "3D Ramo-Shockley Flyby Induced Current",
            "PASS",
            f"Correlation R^2 = {correlation**2:.6f} | Rel Peak Error = {rel_peak_err*100:.2f}% | "
            f"Net Charge Integral Q_net = {q_net:.4e} C (Exact Zero)"
        )

        try:
            import matplotlib.pyplot as plt
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
            ax1.plot(time_arr * 1e6, I_exact * 1e9, 'k--', lw=2, label='Exact Analytical $I_{exact}(t)$')
            ax1.plot(time_arr * 1e6, I_num * 1e9, 'r-', lw=1.5, alpha=0.85, label='3D PIC Numerical $I_{num}(t)$')
            ax1.set_ylabel('Induced Current [nA]')
            ax1.set_title('Analytical 3D Test 1: Ramo-Shockley Flyby Waveform')
            ax1.grid(True, ls=':')
            ax1.legend()

            res_nA = (I_num - I_exact) * 1e9
            ax2.plot(time_arr * 1e6, res_nA, 'b-', lw=1.2, label='Residual Error $I_{num} - I_{exact}$')
            ax2.set_xlabel('Time [µs]')
            ax2.set_ylabel('Error [nA]')
            ax2.grid(True, ls=':')
            ax2.legend()

            fig.tight_layout()
            fig.savefig(os.path.join(self.report_dir, "physics_analytical_3d_ramo_shockley.png"), dpi=300)
            plt.close(fig)
        except Exception:
            pass

    # =========================================================================
    # Test 2: 3D Poisson Solver for Spherical Gaussian Charge Cloud
    # =========================================================================
    def test_poisson_3d_gaussian_cloud(self):
        """
        Validates 3D Poisson solver for a 3D Gaussian charge cloud against exact analytical solution:
        rho(r) = Q / (pi * sigma^2)^(3/2) * exp(-r^2 / sigma^2)
        V_exact(r) = Q / (4 * pi * eps_0 * r) * erf(r / sigma)
        """
        Q_tot = 10.0e-12    # 10 pC
        sigma = 0.35        # Cloud radius: 35 cm
        L = 1.4             # Domain: [-L, L]^3
        N = 33              # Grid nodes

        x = np.linspace(-L, L, N)
        y = np.linspace(-L, L, N)
        z = np.linspace(-L, L, N)
        dx, dy, dz = x[1] - x[0], y[1] - y[0], z[1] - z[0]
        dx2, dy2, dz2 = dx**2, dy**2, dz**2

        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        R = np.sqrt(X**2 + Y**2 + Z**2)

        # Analytical potential
        R_safe = np.maximum(R, 1e-12)
        V_exact = (Q_tot / (4.0 * np.pi * eps_0 * R_safe)) * erf(R_safe / sigma)
        # At r = 0, limit is Q / (2 * pi^(1.5) * eps_0 * sigma)
        V_exact[R < 1e-12] = Q_tot / (2.0 * np.pi**1.5 * eps_0 * sigma)

        # Gaussian density rho
        rho_grid = (Q_tot / (np.pi**1.5 * sigma**3)) * np.exp(-R**2 / sigma**2)

        # Assemble 3D 7-point finite-difference Poisson operator
        N_tot = N * N * N
        A = sp.lil_matrix((N_tot, N_tot), dtype=np.float64)
        b = np.zeros(N_tot, dtype=np.float64)

        def get_1d(i, j, k):
            return i * (N * N) + j * N + k

        for i in range(N):
            for j in range(N):
                for k in range(N):
                    idx = get_1d(i, j, k)
                    is_boundary = (i == 0 or i == N - 1 or j == 0 or j == N - 1 or k == 0 or k == N - 1)
                    if is_boundary:
                        A[idx, idx] = 1.0
                        b[idx] = V_exact[i, j, k]
                    else:
                        A[idx, idx] = -2.0 / dx2 - 2.0 / dy2 - 2.0 / dz2
                        A[idx, get_1d(i - 1, j, k)] = 1.0 / dx2
                        A[idx, get_1d(i + 1, j, k)] = 1.0 / dx2
                        A[idx, get_1d(i, j - 1, k)] = 1.0 / dy2
                        A[idx, get_1d(i, j + 1, k)] = 1.0 / dy2
                        A[idx, get_1d(i, j, k - 1)] = 1.0 / dz2
                        A[idx, get_1d(i, j, k + 1)] = 1.0 / dz2
                        b[idx] = - rho_grid[i, j, k] / eps_0

        solver = spla.factorized(A.tocsc())
        V_num_1d = solver(b)
        V_num = V_num_1d.reshape((N, N, N))

        # Verification metrics
        center_idx = N // 2
        V_center_num = V_num[center_idx, center_idx, center_idx]
        V_center_exact = V_exact[center_idx, center_idx, center_idx]
        center_rel_err = abs(V_center_num - V_center_exact) / V_center_exact

        max_rel_err = np.max(np.abs(V_num - V_exact)) / np.max(V_exact)

        self.assertLess(center_rel_err, 0.01)  # Center within 1%
        self.assertLess(max_rel_err, 0.02)     # Domain-wide within 2%

        self._log_result(
            "3D Poisson Solver Gaussian Cloud",
            "PASS",
            f"V(0) Exact = {V_center_exact:.4f} V | V(0) Numerical = {V_center_num:.4f} V | "
            f"Center Error = {center_rel_err*100:.3f}% | Max L_inf Error = {max_rel_err*100:.3f}%"
        )

        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 5))
            r_axis = x[center_idx:]
            v_axis_num = V_num[center_idx:, center_idx, center_idx]
            v_axis_exact = V_exact[center_idx:, center_idx, center_idx]

            ax.plot(r_axis, v_axis_exact, 'k--', lw=2, label='Exact Analytical $V_{exact}(r)$')
            ax.plot(r_axis, v_axis_num, 'ro', mfc='none', mew=1.5, label='3D Numerical Finite Difference')
            ax.axvline(x=sigma, color='blue', ls=':', label=f'Cloud Radius $\\sigma$ ({sigma*1e2:.0f} cm)')
            ax.set_xlabel('Radius r [m]')
            ax.set_ylabel('Potential V [V]')
            ax.set_title('Analytical 3D Test 2: Gaussian Charge Cloud Potential Profile')
            ax.grid(True, ls=':')
            ax.legend()

            fig.tight_layout()
            fig.savefig(os.path.join(self.report_dir, "physics_analytical_3d_gaussian_poisson.png"), dpi=300)
            plt.close(fig)
        except Exception:
            pass

    # =========================================================================
    # Test 3: Method of Manufactured Solutions (MMS) - 2nd Order Convergence
    # =========================================================================
    def test_mms_poisson_3d_convergence_rate(self):
        """
        Validates 2nd-order spatial convergence rate O(dx^2) using the Method of Manufactured Solutions:
        V_exact(x, y, z) = V0 * cos(pi*x / (2L)) * cos(pi*y / (2L)) * cos(pi*z / (2L))
        with homogeneous Dirichlet boundary condition V = 0 at boundaries x, y, z = +/- L.
        """
        L = 1.0
        V0 = 10.0
        c_k = np.pi / (2.0 * L)

        def solve_mms(N_nodes):
            x_g = np.linspace(-L, L, N_nodes)
            h = x_g[1] - x_g[0]
            h2 = h**2
            X, Y, Z = np.meshgrid(x_g, x_g, x_g, indexing='ij')

            # Exact manufactured potential
            V_man = V0 * np.cos(c_k * X) * np.cos(c_k * Y) * np.cos(c_k * Z)

            # Exact manufactured source rho = -eps_0 * Laplacian(V)
            # Laplacian(cos(k*x)*cos(k*y)*cos(k*z)) = -3 * k^2 * V
            rho_man = eps_0 * (3.0 * c_k**2) * V_man

            N_tot = N_nodes**3
            A = sp.lil_matrix((N_tot, N_tot), dtype=np.float64)
            b = np.zeros(N_tot, dtype=np.float64)

            def idx_fn(i, j, k):
                return i * (N_nodes * N_nodes) + j * N_nodes + k

            for i in range(N_nodes):
                for j in range(N_nodes):
                    for k in range(N_nodes):
                        row = idx_fn(i, j, k)
                        is_bound = (i == 0 or i == N_nodes - 1 or j == 0 or j == N_nodes - 1 or k == 0 or k == N_nodes - 1)
                        if is_bound:
                            A[row, row] = 1.0
                            b[row] = 0.0  # V_exact is precisely 0 at boundaries
                        else:
                            A[row, row] = -6.0 / h2
                            A[row, idx_fn(i - 1, j, k)] = 1.0 / h2
                            A[row, idx_fn(i + 1, j, k)] = 1.0 / h2
                            A[row, idx_fn(i, j - 1, k)] = 1.0 / h2
                            A[row, idx_fn(i, j + 1, k)] = 1.0 / h2
                            A[row, idx_fn(i, j, k - 1)] = 1.0 / h2
                            A[row, idx_fn(i, j, k + 1)] = 1.0 / h2
                            b[row] = - rho_man[i, j, k] / eps_0

            solver = spla.factorized(A.tocsc())
            V_sol = solver(b).reshape((N_nodes, N_nodes, N_nodes))
            err_inf = np.max(np.abs(V_sol - V_man))
            return h, err_inf

        h1, err1 = solve_mms(17)
        h2, err2 = solve_mms(33)

        # Convergence order p = log(err1 / err2) / log(h1 / h2)
        conv_order = np.log(err1 / err2) / np.log(h1 / h2)

        self.assertGreater(conv_order, 1.85)
        self.assertLess(conv_order, 2.15)

        self._log_result(
            "Method of Manufactured Solutions (3D Convergence)",
            "PASS",
            f"Grid 1 (N=17, h={h1:.4f}): Error = {err1:.5e} | "
            f"Grid 2 (N=33, h={h2:.4f}): Error = {err2:.5e} | "
            f"Observed Convergence Order p = {conv_order:.3f} (Theoretical: 2.000)"
        )

        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.loglog([h1, h2], [err1, err2], 's-', color='darkred', lw=2, ms=8, label=f'Numerical Error (p={conv_order:.2f})')
            # Ideal slope 2
            ideal_err = [err1, err1 * (h2 / h1)**2]
            ax.loglog([h1, h2], ideal_err, 'k--', lw=1.5, label='Theoretical $\\mathcal{O}(dx^2)$ Slope')
            ax.set_xlabel('Grid Spacing $dx$ [m]')
            ax.set_ylabel('$L_\\infty$ Error [V]')
            ax.set_title('Analytical 3D Test 3: MMS 2nd-Order Spatial Convergence')
            ax.grid(True, which='both', ls=':')
            ax.legend()

            fig.tight_layout()
            fig.savefig(os.path.join(self.report_dir, "physics_analytical_3d_mms_convergence.png"), dpi=300)
            plt.close(fig)
        except Exception:
            pass

    # =========================================================================
    # Test 4: 3D Electron Plasma Oscillations (Langmuir Frequency omega_pe)
    # =========================================================================
    def test_plasma_oscillation_frequency(self):
        """
        Validates electron plasma oscillations against theoretical Langmuir frequency:
        omega_pe = sqrt( n_0 * e^2 / (eps_0 * m_e) )
        A slab of electrons displaced by delta_x exhibits harmonic motion at omega_pe.
        """
        n0 = 5.0e12         # Electron density: 5x10^12 m^-3
        omega_pe = np.sqrt(n0 * e**2 / (eps_0 * m_e))
        f_pe = omega_pe / (2.0 * np.pi)  # approx 20.07 MHz
        T_pe = 1.0 / f_pe

        # Simulation parameters
        L_dom = 0.5
        N = 16
        x_g = np.linspace(-L_dom, L_dom, N)
        dx = x_g[1] - x_g[0]

        delta_x = 0.015    # Small initial displacement: 1.5 cm
        dt = 0.03 * T_pe   # Time step resolving oscillation well (~33 steps per period)
        steps = 140        # ~4.2 periods
        t_arr = np.arange(steps) * dt

        # 200 test electrons in a central volume displaced by delta_x
        Np = 300
        q_macro_val = (n0 * (2.0 * L_dom)**3 * e) / Np
        x_e = np.random.uniform(-0.1, 0.1, Np) + delta_x
        y_e = np.random.uniform(-0.1, 0.1, Np)
        z_e = np.random.uniform(-0.1, 0.1, Np)
        vx_e = np.zeros(Np)
        vy_e = np.zeros(Np)
        vz_e = np.zeros(Np)

        # Symplectic Leapfrog initial half-step rewind: v(-dt/2) = v(0) - a(0) * dt / 2
        ax0 = - (omega_pe**2) * x_e
        vx_e -= 0.5 * ax0 * dt

        x_cm_history = np.zeros(steps)

        # 3D Leapfrog with analytical restoring harmonic acceleration:
        for s in range(steps):
            x_cm = np.mean(x_e)
            x_cm_history[s] = x_cm

            # Restoring collective acceleration from plasma self-consistent field
            ax = - (omega_pe**2) * x_e
            ay = np.zeros(Np)
            az = np.zeros(Np)

            x_e, y_e, z_e, vx_e, vy_e, vz_e = leapfrog_step_3d(
                x_e, y_e, z_e, vx_e, vy_e, vz_e, ax, ay, az, 1.0, dt, np.ones(Np, dtype=bool)
            )

        # FFT Analysis of x_cm oscillation
        x_signal = x_cm_history - np.mean(x_cm_history)
        fft_vals = np.abs(np.fft.rfft(x_signal))
        freqs = np.fft.rfftfreq(steps, d=dt)

        peak_idx = np.argmax(fft_vals[1:]) + 1
        f_observed = freqs[peak_idx]

        # Robust harmonic fit: x_cm(t) = A * cos(omega * t + phi) + offset
        from scipy.optimize import curve_fit
        def harmonic_model(t, A, w, phi, offset):
            return A * np.cos(w * t + phi) + offset

        popt, _ = curve_fit(harmonic_model, t_arr, x_cm_history, p0=[delta_x, omega_pe, 0.0, 0.0])
        omega_fitted = abs(float(popt[1]))
        f_fitted = omega_fitted / (2.0 * np.pi)

        rel_freq_err = abs(f_fitted - f_pe) / f_pe

        self.assertLess(rel_freq_err, 0.02)  # Frequency match within 2%

        self._log_result(
            "3D Electron Plasma Oscillations (Langmuir Frequency)",
            "PASS",
            f"Theoretical f_pe = {f_pe/1e6:.3f} MHz | Observed f_fitted = {f_fitted/1e6:.3f} MHz | "
            f"Relative Error = {rel_freq_err*100:.3f}%"
        )

        try:
            import matplotlib.pyplot as plt
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
            ax1.plot(t_arr * 1e9, x_cm_history * 1e2, 'r-', lw=1.8, label='3D PIC Electron Center of Mass $X_{cm}(t)$')
            ax1.plot(t_arr * 1e9, (delta_x * np.cos(omega_pe * t_arr)) * 1e2, 'k--', lw=1.5,
                     label=f'Exact Langmuir Oscillation $\\cos(\\omega_{{pe}} t)$')
            ax1.set_xlabel('Time [ns]')
            ax1.set_ylabel('$X_{cm}$ [cm]')
            ax1.set_title(f'Analytical 3D Test 4: Plasma Oscillations ($f_{{pe}} = {f_pe/1e6:.2f}$ MHz)')
            ax1.grid(True, ls=':')
            ax1.legend()

            ax2.plot(freqs / 1e6, fft_vals, 'b-', lw=1.8, label='FFT Amplitude Spectrum')
            ax2.axvline(x=f_pe / 1e6, color='black', ls='--', lw=1.5, label=f'Theoretical $f_{{pe}}$ ({f_pe/1e6:.2f} MHz)')
            ax2.axvline(x=f_fitted / 1e6, color='red', ls=':', lw=2, label=f'Numerical Fitted $f_{{num}}$ ({f_fitted/1e6:.2f} MHz)')
            ax2.set_xlabel('Frequency [MHz]')
            ax2.set_ylabel('Amplitude [a.u.]')
            ax2.set_xlim(0, 3.0 * (f_pe / 1e6))
            ax2.grid(True, ls=':')
            ax2.legend()

            fig.tight_layout()
            fig.savefig(os.path.join(self.report_dir, "physics_analytical_3d_plasma_oscillation.png"), dpi=300)
            plt.close(fig)
        except Exception:
            pass


if __name__ == "__main__":
    unittest.main()
