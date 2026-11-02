import contextlib
import io
import unittest
import os
import numpy as np

from dust_impact.common.circuits import integrate_rc_circuit
from dust_impact.common.io import ensure_dir
from dust_impact.physics.charging import (
    calculate_equilibrium_potential, electron_thermal_current_density, f_balance, ENV_EARTH, MAT_ALUMINIUM
)
from dust_impact.solver.config_loader import SimulationParams3D, SimulationToggles3D
from dust_impact.solver.sim_core import DustImpactSimulation3D


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

    def test_0d_charging_current_balance(self):
        """
        Verifies the 0D floating potential: the net current density f(V) = Je - Ji - Jph - Jse must
        vanish at V_eq, and V_eq must be a bracketed root (f changes sign around it).
        """
        V_eq = calculate_equilibrium_potential(ENV_EARTH, MAT_ALUMINIUM, verbose=False)
        J_scale = electron_thermal_current_density(ENV_EARTH.n, ENV_EARTH.Te_eV)

        residual = f_balance(V_eq, ENV_EARTH, MAT_ALUMINIUM) / J_scale
        self.assertLess(abs(residual), 1e-4)
        f_lo = f_balance(V_eq - 0.5, ENV_EARTH, MAT_ALUMINIUM)
        f_hi = f_balance(V_eq + 0.5, ENV_EARTH, MAT_ALUMINIUM)
        self.assertLess(f_lo * f_hi, 0.0, "V_eq is not a bracketed root of the current balance")

        self._log_result(
            "0D Charging Current Balance",
            "PASS",
            f"V_eq (Earth - Al) = {V_eq:.4f} V | f(V_eq)/J_e0 = {residual:.2e}"
        )

    def test_charge_balance_in_3d_pic(self):
        """
        Verifies the charge bookkeeping of the antenna signal in a full self-consistent 3D PIC run.

        Ramo-Shockley theorem: the total charge delivered to an antenna equals
            Q = sum_p q_p * [Vw(r_p,end) - Vw(r_p,start)],
        with Vw(r_end) = 1 for particles collected by the antenna. The cloud is injected after a
        non-zero delay and expands under its own field, so collection, escape and induction all occur.

        The solver accumulates q * dVw per step, so this identity holds to round-off by construction:
        the test guards against double counting, lost steps (injection, removal, collection) and wrong
        signs, not against an incorrect weighting potential. The physical correctness of Vw is covered
        by the Green reciprocity test and by tests/test_weighting_potentials.py.
        """
        rng = np.random.default_rng(11)
        params = SimulationParams3D(
            grid_nodes_x=25, grid_nodes_y=25, grid_nodes_z=25,
            domain_half_length_x_m=1.5, domain_half_length_y_m=1.5, domain_half_length_z_m=1.5,
            num_macroparticles=3000,
            total_impact_charge_C=2e-10,
            time_step_s=1e-8,
            num_time_steps=150,
            impact_time_delay_s=3e-8,
            ion_mass_amu=1.0,
            impact_location_xyz_m=[0.0, 0.0, 0.0],
            impact_normal=[1.0, 0.0, 0.0],
        )
        toggles = SimulationToggles3D(
            enable_spis_background_field=False,
            enable_antenna_particle_collection=True,
            enable_plasma_self_field=True,
        )

        # Spherical antenna (radius R, centre c) with vacuum weighting potential Vw = R / |r - c|
        X, Y, Z = np.meshgrid(params.x_grid, params.y_grid, params.z_grid, indexing='ij')
        c_ant, R_ant = np.array([0.7, 0.0, 0.0]), 0.3
        r_ant = np.sqrt((X - c_ant[0])**2 + (Y - c_ant[1])**2 + (Z - c_ant[2])**2)
        ant_mask = r_ant <= R_ant
        Vw = np.where(ant_mask, 1.0, R_ant / np.maximum(r_ant, 1e-12))
        Ewx, Ewy, Ewz = np.gradient(-Vw, params.dx, params.dy, params.dz)
        shape = (params.Nx, params.Ny, params.Nz)
        zeros = np.zeros(shape)

        with contextlib.redirect_stdout(io.StringIO()):
            sim = DustImpactSimulation3D(
                params, toggles, zeros, [Vw], zeros, zeros, zeros,
                [Ewx], [Ewy], [Ewz], [ant_mask], np.zeros(shape, dtype=bool)
            )
        ens = sim.particles
        # Quasi-neutral Gaussian cloud (sigma = 0.1 m) resolved by the grid
        pos = rng.normal(0.0, 0.1, (params.N_particles, 3))
        for arr, col in zip((ens.x_e, ens.y_e, ens.z_e), pos.T):
            arr[:] = col
        for arr, col in zip((ens.x_i, ens.y_i, ens.z_i), pos.T):
            arr[:] = col

        fs = sim.field_solver
        vw_start_e = fs.interp_field(ens.x_e, ens.y_e, ens.z_e, Vw)
        vw_start_i = fs.interp_field(ens.x_i, ens.y_i, ens.z_i, Vw)

        with contextlib.redirect_stdout(io.StringIO()):
            sim.run()

        def vw_end(x, y, z, active):
            v = fs.interp_field(x, y, z, Vw)
            ix = np.clip(np.floor((x - params.x_grid[0]) / params.dx), 0, params.Nx - 1).astype(np.int64)
            iy = np.clip(np.floor((y - params.y_grid[0]) / params.dy), 0, params.Ny - 1).astype(np.int64)
            iz = np.clip(np.floor((z - params.z_grid[0]) / params.dz), 0, params.Nz - 1).astype(np.int64)
            collected = (~active) & ant_mask[ix, iy, iz]
            v[collected] = 1.0
            return v, collected

        vw_end_e, col_e = vw_end(ens.x_e, ens.y_e, ens.z_e, ens.active_e)
        vw_end_i, col_i = vw_end(ens.x_i, ens.y_i, ens.z_i, ens.active_i)

        Q_expected = params.q_macro * (np.sum(vw_end_i - vw_start_i) - np.sum(vw_end_e - vw_start_e))
        Q_signal = np.sum(sim.tot_curr[0]) * params.dt

        # The scenario must actually exercise collection and escape, otherwise the test is vacuous
        self.assertGreater(np.sum(col_e), 0, "No electrons were collected by the antenna")
        self.assertLess(np.sum(ens.active_e), params.N_particles, "No electrons left the simulation")
        self.assertAlmostEqual(Q_signal / Q_expected, 1.0, places=9)

        self._log_result(
            "Antenna Charge Balance (3D PIC, Ramo-Shockley)",
            "PASS",
            f"Q_signal = {Q_signal:.6e} C | Q_expected = {Q_expected:.6e} C | "
            f"collected e/i = {int(np.sum(col_e))}/{int(np.sum(col_i))}"
        )


if __name__ == "__main__":
    unittest.main()
