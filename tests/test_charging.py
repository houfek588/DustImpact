import unittest
from dust_impact.physics.constants import n_sw, Te_eV
from dust_impact.physics.charging import (
    calculate_equilibrium_potential,
    ENV_EARTH,
    ENV_MARS,
    ENV_JUPITER,
    MAT_ALUMINIUM,
    MAT_ALUMINIUM_ANTENNE,
    MAT_GOLD,
    find_root_bisection,
)


class TestChargingModel(unittest.TestCase):
    def test_bisection_root_finder(self):
        func = lambda x: x ** 3 - 8
        root = find_root_bisection(func, 0.0, 5.0, tol=1e-7)
        self.assertAlmostEqual(root, 2.0, places=5)

    def test_equilibrium_potentials(self):
        V_earth_al = calculate_equilibrium_potential(ENV_EARTH, MAT_ALUMINIUM, verbose=False)
        self.assertIsInstance(V_earth_al, float)
        self.assertGreater(V_earth_al, -100.0)
        self.assertLess(V_earth_al, 100.0)

        V_earth_ant = calculate_equilibrium_potential(ENV_EARTH, MAT_ALUMINIUM_ANTENNE, verbose=False)
        self.assertIsInstance(V_earth_ant, float)

        V_jup_gold = calculate_equilibrium_potential(ENV_JUPITER, MAT_GOLD, verbose=False)
        self.assertIsInstance(V_jup_gold, float)


if __name__ == "__main__":
    unittest.main()
