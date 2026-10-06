import unittest
from dust_impact.physics.constants import e, m_e, eps_0, amu, n_sw, Te_eV


class TestConstants(unittest.TestCase):
    def test_values(self):
        self.assertAlmostEqual(e, 1.602176634e-19, delta=1e-25)
        self.assertAlmostEqual(m_e, 9.1093837015e-31, delta=1e-35)
        self.assertAlmostEqual(eps_0, 8.8541878128e-12, delta=1e-16)
        self.assertAlmostEqual(amu, 1.6605390666e-27, delta=1e-32)
        self.assertGreater(n_sw, 0)
        self.assertGreater(Te_eV, 0)

    def test_derived_plasma_parameters(self):
        import numpy as np
        from dust_impact.common.config import BaseSimulationParams
        from dust_impact.solver.params import SimulationParams3D

        params = BaseSimulationParams()
        # Verify Debye length
        expected_debye = np.sqrt((eps_0 * params.solar_wind_electron_temp_eV * e) / (params.solar_wind_density_m3 * e ** 2))
        self.assertAlmostEqual(params.debye_length, expected_debye, places=4)

        # Verify electron plasma frequency
        expected_omega_pe = np.sqrt((params.solar_wind_density_m3 * e ** 2) / (eps_0 * m_e))
        expected_f_pe = expected_omega_pe / (2.0 * np.pi)
        self.assertAlmostEqual(params.omega_pe, expected_omega_pe, places=2)
        self.assertAlmostEqual(params.f_pe, expected_f_pe, places=2)

        # Verify ion plasma frequency
        m_i = params.ion_mass_amu * amu
        expected_omega_pi = np.sqrt((params.solar_wind_density_m3 * e ** 2) / (eps_0 * m_i))
        expected_f_pi = expected_omega_pi / (2.0 * np.pi)
        self.assertAlmostEqual(params.omega_pi, expected_omega_pi, places=2)
        self.assertAlmostEqual(params.f_pi, expected_f_pi, places=2)

        # Verify thermal velocities
        expected_v_th_e = np.sqrt(2.0 * e * params.impact_cloud_temperature_eV / m_e)
        expected_v_th_i = np.sqrt(2.0 * e * params.impact_cloud_temperature_eV / m_i)
        self.assertAlmostEqual(params.v_th_e, expected_v_th_e, places=2)
        self.assertAlmostEqual(params.v_th_i, expected_v_th_i, places=2)

        # Verify 3D simulation params inherits identical derived values
        sim_params = SimulationParams3D()
        self.assertAlmostEqual(sim_params.omega_pe, expected_omega_pe, places=2)
        self.assertAlmostEqual(sim_params.f_pe, expected_f_pe, places=2)
        self.assertAlmostEqual(sim_params.omega_pi, expected_omega_pi, places=2)
        self.assertAlmostEqual(sim_params.f_pi, expected_f_pi, places=2)
        self.assertAlmostEqual(sim_params.v_th_e, expected_v_th_e, places=2)
        self.assertAlmostEqual(sim_params.v_th_i, expected_v_th_i, places=2)


if __name__ == "__main__":
    unittest.main()
