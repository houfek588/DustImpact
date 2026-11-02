import math
import unittest
from dataclasses import replace

from dust_impact.physics.constants import e, amu
from dust_impact.physics.charging import (
    calculate_equilibrium_potential,
    electron_thermal_current_density,
    find_root_bisection,
    find_stable_floating_potentials,
    f_balance,
    Jse,
    Je,
    mean_secondary_yield,
    oml_escape_fraction,
    sey_delta_vaughan_like,
    ENV_EARTH,
    ENV_MARS,
    ENV_JUPITER,
    MAT_ALUMINIUM,
    MAT_ALUMINIUM_ANTENNE,
    MAT_GOLD,
    MAT_AL_OXIDE,
)


class TestChargingModel(unittest.TestCase):
    def test_bisection_root_finder(self):
        func = lambda x: x ** 3 - 8
        root = find_root_bisection(func, 0.0, 5.0, tol=1e-7)
        self.assertAlmostEqual(root, 2.0, places=5)

    def test_bisection_rejects_unbracketed_interval(self):
        """Without a sign change the root finder must fail instead of returning an end point."""
        with self.assertRaises(ValueError):
            find_root_bisection(lambda x: x ** 2 + 1.0, -5.0, 5.0)

    def test_equilibrium_potentials_are_stable_roots(self):
        """For every preset the floating potential is a unique stable root of the current balance."""
        for env in (ENV_EARTH, ENV_MARS, ENV_JUPITER):
            for mat in (MAT_ALUMINIUM, MAT_ALUMINIUM_ANTENNE, MAT_GOLD, MAT_AL_OXIDE):
                with self.subTest(env=env.name, mat=mat.name):
                    V_eq = calculate_equilibrium_potential(env, mat, verbose=False)
                    J_e0 = electron_thermal_current_density(env.n, env.Te_eV)
                    self.assertLess(abs(f_balance(V_eq, env, mat)) / J_e0, 1e-8)
                    self.assertLess(f_balance(V_eq - 0.1, env, mat), 0.0)
                    self.assertGreater(f_balance(V_eq + 0.1, env, mat), 0.0)
                    self.assertEqual(len(find_stable_floating_potentials(env, mat)), 1)
                    # Photoemission dominated sunlit probe in the solar wind floats a few volts positive
                    self.assertGreater(V_eq, 2.0)
                    self.assertLess(V_eq, 20.0)

    def test_dark_probe_without_emission_matches_analytic_potential(self):
        """
        A probe without photo- and secondary emission floats negative where the repelled electron
        flux equals the ion beam flux: J_e0 exp(V/Te) = 1/4 e n v_sw (1 - V/E_ram).
        """
        dark = replace(MAT_ALUMINIUM, Jph0_uA_m2=0.0, sey_delta_max=0.0)
        env = ENV_EARTH
        J_e0 = electron_thermal_current_density(env.n, env.Te_eV)
        J_ram = 0.25 * e * env.n * env.v_sw
        E_ram = 0.5 * amu * env.v_sw ** 2 / e
        V = 0.0
        for _ in range(50):
            V = env.Te_eV * math.log(J_ram * (1.0 - V / E_ram) / J_e0)

        V_eq = calculate_equilibrium_potential(env, dark, verbose=False)
        self.assertLess(V_eq, 0.0)
        self.assertAlmostEqual(V_eq, V, places=6)

    def test_emitted_electrons_are_recollected_at_positive_potential(self):
        """The OML escape fraction suppresses secondary emission at positive potentials."""
        self.assertEqual(oml_escape_fraction(-3.0, 2.0), 1.0)
        self.assertAlmostEqual(oml_escape_fraction(4.0, 2.0), 3.0 * math.exp(-2.0), places=12)
        ratio_at_20V = Jse(20.0, MAT_AL_OXIDE, ENV_EARTH) / Je(20.0, ENV_EARTH)
        self.assertLess(ratio_at_20V, 1e-2)

    def test_mean_secondary_yield_matches_direct_integration(self):
        """Gauss-Laguerre flux average agrees with a direct midpoint-rule integration."""
        env, mat = ENV_EARTH, MAT_ALUMINIUM
        for V in (-5.0, 0.0, 10.0):
            T = env.Te_eV
            dE = 0.01
            total = 0.0
            for k in range(int(40 * T / dE)):
                E = (k + 0.5) * dE
                total += sey_delta_vaughan_like(E + max(V, 0.0), mat) * E / T ** 2 * math.exp(-E / T) * dE
            with self.subTest(V=V):
                self.assertAlmostEqual(mean_secondary_yield(V, mat, env), total, delta=1e-3 * total)


if __name__ == "__main__":
    unittest.main()
