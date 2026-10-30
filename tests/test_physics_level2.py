import contextlib
import io
import os
import unittest
import numpy as np

from dust_impact.physics.constants import e, m_e, amu
from dust_impact.common.io import ensure_dir
from dust_impact.solver.config_loader import SimulationParams3D, SimulationToggles3D
from dust_impact.solver.sim_core import DustImpactSimulation3D


class TestPhysicsLevel2(unittest.TestCase):
    """
    Level 2 Physical Cross-Verification:
    Tests collective plasma behaviour of the self-consistent 3D PIC solver.
    Saves validation text reports to outputs/tests/
    """

    @classmethod
    def setUpClass(cls):
        cls.report_dir = "outputs/tests"
        cls.report_path = os.path.join(cls.report_dir, "physics_level2_results.txt")
        ensure_dir(cls.report_path)
        with open(cls.report_path, "w", encoding="utf-8") as f:
            f.write("=====================================================\n")
            f.write("    DUSTIMPACT - PHYSICS VALIDATION REPORT (LEVEL 2) \n")
            f.write("=====================================================\n\n")

    def _log_result(self, test_name: str, status: str, details: str):
        with open(self.report_path, "a", encoding="utf-8") as f:
            f.write(f"[{status}] {test_name}\n")
            f.write(f"  Details: {details}\n\n")

    def _run_expanding_cloud(self, self_field: bool):
        """
        Runs a quasi-neutral Gaussian cloud (sigma = 0.15 m, T_e = 2 eV, cold ions) resolved by the
        grid (lambda_D ~ 0.03 m, dx ~ 0.08 m) for 3 us and returns the final ensemble.
        """
        rng = np.random.default_rng(7)
        Np, Te_eV, Ti_eV, mi_amu = 4000, 2.0, 0.1, 1.0
        params = SimulationParams3D(
            grid_nodes_x=25, grid_nodes_y=25, grid_nodes_z=25,
            domain_half_length_x_m=1.0, domain_half_length_y_m=1.0, domain_half_length_z_m=1.0,
            num_macroparticles=Np,
            total_impact_charge_C=1e-9,
            time_step_s=1e-8,
            num_time_steps=300,
            impact_time_delay_s=0.0,
            ion_mass_amu=mi_amu,
        )
        toggles = SimulationToggles3D(
            enable_spis_background_field=False,
            enable_antenna_particle_collection=False,
            enable_plasma_self_field=self_field,
        )
        shape = (params.Nx, params.Ny, params.Nz)
        zeros = np.zeros(shape)
        with contextlib.redirect_stdout(io.StringIO()):
            sim = DustImpactSimulation3D(
                params, toggles, zeros, [zeros.copy()], zeros, zeros, zeros,
                [zeros], [zeros], [zeros], [np.zeros(shape, dtype=bool)], np.zeros(shape, dtype=bool)
            )
        ens = sim.particles
        pos = rng.normal(0.0, 0.15, (Np, 3))
        for arr, col in zip((ens.x_e, ens.y_e, ens.z_e), pos.T):
            arr[:] = col
        for arr, col in zip((ens.x_i, ens.y_i, ens.z_i), pos.T):
            arr[:] = col
        # Isotropic Maxwellians (set explicitly so that the test does not depend on the injection model)
        sigma_e = np.sqrt(e * Te_eV / m_e)
        sigma_i = np.sqrt(e * Ti_eV / (mi_amu * amu))
        for arr in (ens.vx_e, ens.vy_e, ens.vz_e):
            arr[:] = rng.normal(0.0, sigma_e, Np)
        for arr in (ens.vx_i, ens.vy_i, ens.vz_i):
            arr[:] = rng.normal(0.0, sigma_i, Np)

        with contextlib.redirect_stdout(io.StringIO()):
            for step in range(params.steps):
                sim.step(step)

        c_s = np.sqrt(e * Te_eV / (mi_amu * amu))
        return ens, c_s

    def test_ambipolar_ion_acceleration(self):
        """
        Verifies ambipolar expansion of an impact plasma cloud:
        hot electrons are retained by the self-consistent space-charge field, which in turn
        accelerates the (initially cold) ions radially outward to a fraction of the ion sound speed
        c_s = sqrt(k_B T_e / m_i). Without the self-field neither effect may occur.
        """
        results = {}
        for self_field in (True, False):
            ens, c_s = self._run_expanding_cloud(self_field)
            a = ens.active_i
            r = np.sqrt(ens.x_i[a]**2 + ens.y_i[a]**2 + ens.z_i[a]**2)
            v_r = (ens.vx_i[a] * ens.x_i[a] + ens.vy_i[a] * ens.y_i[a] + ens.vz_i[a] * ens.z_i[a]) / r
            results[self_field] = (np.mean(v_r) / c_s, np.mean(ens.active_e))

        vr_sf, retained_sf = results[True]
        vr_free, retained_free = results[False]

        # Ions are accelerated outward only by the ambipolar field
        self.assertGreater(vr_sf, 0.15)
        self.assertLess(vr_sf, 1.5)
        self.assertLess(abs(vr_free), 0.05)
        # Electrons are electrostatically confined by the positive cloud potential
        self.assertGreater(retained_sf, 3.0 * retained_free)

        self._log_result(
            "Ambipolar Ion Acceleration (3D PIC)",
            "PASS",
            f"<v_r,i>/c_s: self-field = {vr_sf:.3f}, free = {vr_free:.3f} | "
            f"retained electrons: self-field = {retained_sf:.2f}, free = {retained_free:.2f}"
        )


if __name__ == "__main__":
    unittest.main()
