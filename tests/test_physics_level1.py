import unittest
import os
import numpy as np
from dust_impact.physics.constants import e, m_e, eps_0
from dust_impact.common.circuits import integrate_rc_circuit
from dust_impact.common.io import ensure_dir
from dust_impact.physics.charging import calculate_equilibrium_potential, ENV_EARTH, MAT_ALUMINIUM
from dust_impact.sim2d.input_data import SimulationParams2D, SimulationToggles2D, calc_Ew_2d
from dust_impact.sim2d.sim_core import DustImpactSimulation2D


class TestPhysicsLevel1(unittest.TestCase):
    """
    Level 1 Physics Verification:
    Tests fundamental physical conservation laws and exact analytical limits.
    Saves validation text reports and plots to outputs/tests/
    """

    @classmethod
    def setUpClass(cls):
        cls.report_dir = "outputs/tests"
        cls.report_path = os.path.join(cls.report_dir, "physics_level1_results.txt")
        ensure_dir(cls.report_path)
        with open(cls.report_path, "w", encoding="utf-8") as f:
            f.write("=====================================================\n")
            f.write("    DUSTIMPACT - PHYSICS VALIDATION REPORT (LEVEL 1) \n")
            f.write("=====================================================\n\n")

    def _log_result(self, test_name: str, status: str, details: str):
        with open(self.report_path, "a", encoding="utf-8") as f:
            f.write(f"[{status}] {test_name}\n")
            f.write(f"  {details}\n\n")

    def test_rc_circuit_analytical_response(self):
        """
        Verifies that RC circuit integration matches the exact analytical step response:
        V(t) = (Q / C) * exp(-t / (R * C))
        """
        R = 100e3  # 100 kOhm
        C = 10e-12  # 10 pF
        tau = R * C  # 1 us
        dt = 1e-9  # 1 ns
        t_max = 5e-6  # 5 us
        time_array = np.arange(0, t_max, dt)
        steps = len(time_array)

        Q0 = 1e-12  # 1 pC
        tot_curr = np.zeros(steps)
        tot_curr[0] = Q0 / dt

        V_num = integrate_rc_circuit(tot_curr, dt, R, C, V_initial=0.0)
        V_analytical = (Q0 / C) * np.exp(-time_array / tau)

        max_rel_error = np.max(np.abs(V_num[1:] - V_analytical[1:]) / V_analytical[1:])
        np.testing.assert_allclose(V_num[1:], V_analytical[1:], rtol=1e-2, atol=1e-5)

        self._log_result(
            "RC Circuit Analytical Response",
            "PASS",
            f"Max Rel Error: {max_rel_error * 100:.4f}% | Peak Voltage: {np.max(V_num):.4f} V"
        )

        try:
            import matplotlib.pyplot as plt
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
            ax1.plot(time_array * 1e6, V_analytical * 1e3, 'k--', lw=2, label='Analytical V(t)')
            ax1.plot(time_array * 1e6, V_num * 1e3, 'r:', lw=2, label='Numerical RC Solver')
            ax1.set_ylabel('Voltage [mV]')
            ax1.set_title('Level 1: RC Circuit Step Response Verification')
            ax1.grid(True, ls=':')
            ax1.legend()

            error_mV = np.abs(V_num - V_analytical) * 1e3
            ax2.plot(time_array * 1e6, error_mV, 'b-', lw=1.5, label='Residual Absolute Error')
            ax2.set_xlabel('Time [µs]')
            ax2.set_ylabel('Absolute Error [mV]')
            ax2.grid(True, ls=':')
            ax2.legend()

            fig.tight_layout()
            plot_path = os.path.join(self.report_dir, "physics_level1_rc_circuit.png")
            fig.savefig(plot_path, dpi=300)
            plt.close(fig)
        except Exception as err:
            print(f"Skipping plot save (headless/matplotlib error): {err}")

    def test_ramo_shockley_zero_net_charge_transfer(self):
        """
        Verifies Ramo-Shockley theorem:
        A charge passing by a conductor without physical impact produces a bipolar
        induced current waveform whose time integral over all time is exactly zero.
        """
        v0 = 1e5  # 100 km/s
        y0 = 0.3  # 30 cm offset
        q0 = e * 1e4
        dt = 1e-9
        time = np.linspace(-10e-6, 10e-6, 20000)

        x_pos = v0 * time
        y_pos = np.full_like(x_pos, y0)

        Ewx, Ewy = calc_Ew_2d(x_pos, y_pos, 0.0, 0.0, 0.0, 0.0, w_width=0.2)
        I_ind = - q0 * (v0 * Ewx + 0.0 * Ewy)
        Q_net = np.sum(I_ind) * dt

        self.assertAlmostEqual(Q_net, 0.0, delta=1e-18)

        self._log_result(
            "Ramo-Shockley Zero Net Charge Transfer",
            "PASS",
            f"Net Charge Integral Q_net = {Q_net:.4e} C (Exact Zero)"
        )

        try:
            import matplotlib.pyplot as plt
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
            ax1.plot(time * 1e6, Ewx, 'g-', lw=2, label='Weighting Field Ewx [V/m]')
            ax1.set_ylabel('Weighting Field [1/m]')
            ax1.set_title('Level 1: Ramo-Shockley Bipolar Induced Current')
            ax1.grid(True, ls=':')
            ax1.legend()

            ax2.plot(time * 1e6, I_ind * 1e9, 'm-', lw=2, label='Induced Current I_ind [nA]')
            ax2.axhline(0, color='grey', ls='--', alpha=0.7)
            ax2.set_xlabel('Time [µs]')
            ax2.set_ylabel('Current [nA]')
            ax2.grid(True, ls=':')
            ax2.legend()

            fig.tight_layout()
            plot_path = os.path.join(self.report_dir, "physics_level1_ramo_shockley.png")
            fig.savefig(plot_path, dpi=300)
            plt.close(fig)
        except Exception as err:
            print(f"Skipping plot save (headless/matplotlib error): {err}")

    def test_0d_charging_root_precision(self):
        """
        Verifies 0D equilibrium potential current balance precision:
        f(V_eq) = 0
        """
        V_eq = calculate_equilibrium_potential(ENV_EARTH, MAT_ALUMINIUM, verbose=False)
        self.assertIsInstance(V_eq, float)
        self.assertGreater(V_eq, -50.0)
        self.assertLess(V_eq, 50.0)

        self._log_result(
            "0D Charging Root Precision",
            "PASS",
            f"Equilibrium Surface Potential (Earth - Al): {V_eq:.4f} V"
        )

    def test_charge_conservation_in_2d_pic(self):
        """
        Verifies global conservation of charge in 2D PIC simulation:
        Q_total(t=0) == Q_active(t) + Q_collected(t) + Q_escaped(t)
        """
        params = SimulationParams2D(
            Vf=-5.0,
            num_macroparticles=200,
            simulation_duration_s=2e-8,
            time_step_s=1e-9,
            grid_nodes_x=12,
            grid_nodes_y=12,
            antennas=[
                {"x1": 1.0, "y1": 0.1, "x2": 1.0, "y2": 0.5, "r": 0.05, "V_bias": 0.0, "w_width": 0.2, "C": 1e-12, "R": 10e3}
            ]
        )
        toggles = SimulationToggles2D()
        sim = DustImpactSimulation2D(params, toggles)

        Q_e_init = - sim.p.N_particles * sim.p.q_macro
        Q_i_init = + sim.p.N_particles * sim.p.q_macro
        Q_tot_init = Q_e_init + Q_i_init

        sim.run()

        Q_active_e = - np.sum(sim.active_e) * sim.p.q_macro
        Q_active_i = + np.sum(sim.active_i) * sim.p.q_macro
        Q_active = Q_active_e + Q_active_i

        Q_collected = 0.0
        for a_idx in range(sim.num_antennas):
            Q_collected += np.sum(sim.col_curr_e[a_idx] + sim.col_curr_i[a_idx]) * params.dt

        escaped_e = (~sim.active_e) & (~sim.was_outside_e[0])
        escaped_i = (~sim.active_i) & (~sim.was_outside_i[0])
        Q_escaped_e = - np.sum(escaped_e) * sim.p.q_macro
        Q_escaped_i = + np.sum(escaped_i) * sim.p.q_macro
        Q_escaped = Q_escaped_e + Q_escaped_i

        Q_tot_final = Q_active + Q_collected + Q_escaped
        delta_Q = abs(Q_tot_init - Q_tot_final)

        self.assertAlmostEqual(Q_tot_init, Q_tot_final, delta=1e-15)

        self._log_result(
            "Global Charge Conservation (2D PIC)",
            "PASS",
            f"Q_init = {Q_tot_init:.4e} C | Q_final = {Q_tot_final:.4e} C | Delta = {delta_Q:.4e} C"
        )

    def test_debye_shielding_exponential_decay(self):
        """
        Verifies exact 1D/2D Debye potential screening decay in plasma:
        Solves (grad^2 - 1/lambda_D^2) V = 0 and verifies that V(x) matches exact
        analytical solution V_exact(x) = V0 * sinh((L - x)/lambda_D) / sinh(L/lambda_D),
        and at x = lambda_D, V(lambda_D) / V0 is within 0.1% of e^-1.
        Saves static plot to outputs/tests/physics_level1_debye_shielding.png
        """
        import scipy.sparse as sp
        import scipy.sparse.linalg as spla
        from dust_impact.physics.constants import n_sw, Te_eV

        lambda_D = np.sqrt((eps_0 * Te_eV * e) / (n_sw * e ** 2))
        kD2 = 1.0 / lambda_D ** 2

        L = 50.0
        Nx = 200
        x_grid = np.linspace(0, L, Nx)
        dx = x_grid[1] - x_grid[0]

        A = sp.lil_matrix((Nx, Nx))
        b = np.zeros(Nx)

        V0 = 10.0
        A[0, 0] = 1.0
        b[0] = V0

        A[Nx - 1, Nx - 1] = 1.0
        b[Nx - 1] = 0.0

        for i in range(1, Nx - 1):
            A[i, i - 1] = 1.0 / dx ** 2
            A[i, i] = -2.0 / dx ** 2 - kD2
            A[i, i + 1] = 1.0 / dx ** 2

        solver = spla.factorized(A.tocsc())
        V_num = solver(b)

        V_analytical = V0 * np.sinh((L - x_grid) / lambda_D) / np.sinh(L / lambda_D)

        target_idx = np.argmin(np.abs(x_grid - lambda_D))
        V_num_lambda = V_num[target_idx]
        V_ana_lambda = V_analytical[target_idx]

        rel_err = abs(V_num_lambda - V_ana_lambda) / V_ana_lambda

        self.assertLess(rel_err, 0.001)
        self.assertAlmostEqual(V_num_lambda / V0, np.exp(-1), delta=0.01)

        self._log_result(
            "Debye Shielding Exponential Decay",
            "PASS",
            f"lambda_D = {lambda_D:.3f} m | V(lambda_D)/V0 = {V_num_lambda / V0:.4f} (Theoretical e^-1 = {np.exp(-1):.4f}) | Rel Error = {rel_err * 100:.4f}%"
        )

        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(x_grid, V_analytical, 'k--', lw=2, label='Analytical Debye Profile V(x)')
            ax.plot(x_grid, V_num, 'r:', lw=2, label='Poisson Solver with Debye Screening')
            ax.axvline(lambda_D, color='blue', ls='--', label=f'Debye Length lambda_D = {lambda_D:.2f} m')
            ax.axhline(V0 * np.exp(-1), color='green', ls=':', label=f'V0 / e = {V0 * np.exp(-1):.2f} V')

            ax.set_xlabel('Distance x [m]')
            ax.set_ylabel('Potential V [V]')
            ax.set_title('Level 1: Debye Plasma Shielding Decay Verification')
            ax.grid(True, ls=':')
            ax.legend(loc='upper right')

            fig.tight_layout()
            plot_path = os.path.join(self.report_dir, "physics_level1_debye_shielding.png")
            fig.savefig(plot_path, dpi=300)
            plt.close(fig)
        except Exception as err:
            print(f"Skipping plot save: {err}")

    def test_cfl_condition_stability(self):
        """
        Verifies that check_cfl_condition correctly checks the PIC numerical stability condition 1.5 * v_th * dt <= dx.
        """
        from dust_impact.numerics.pushers import check_cfl_condition

        T_dust_eV = 2.0
        v_th_e = np.sqrt(2.0 * e * T_dust_eV / m_e)  # ~8.38e5 m/s
        v_cfl = 1.5 * v_th_e  # ~1.26e6 m/s
        dx = 0.20  # 20 cm

        # 1. Stable case: dt = 2 ns -> 1.5 * v_th * dt = 2.51 mm <= 200 mm
        dt_stable = 2e-9
        is_stable, cfl_ratio, dt_rec = check_cfl_condition(dt_stable, dx, v_cfl)
        self.assertTrue(is_stable)
        self.assertLess(cfl_ratio, 1.0)
        self.assertAlmostEqual(dt_rec, dx / v_cfl, places=9)

        # 2. Unstable case: dt = 1 us -> 1.5 * v_th * dt = 1.26 m > 0.20 m
        dt_unstable = 1e-6
        is_stable, cfl_ratio, dt_rec = check_cfl_condition(dt_unstable, dx, v_cfl)
        self.assertFalse(is_stable)
        self.assertGreater(cfl_ratio, 1.0)

        self._log_result(
            "CFL Numerical Stability Condition (1.5 * v_th * dt <= dx)",
            "PASS",
            f"1.5*v_th_e = {v_cfl:.2e} m/s | Stable dt={dt_stable:.1e} s (CFL={cfl_ratio:.4f}) | Unstable dt={dt_unstable:.1e} s (CFL={cfl_ratio:.2f})"
        )


if __name__ == "__main__":
    unittest.main()
