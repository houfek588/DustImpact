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


if __name__ == "__main__":
    unittest.main()
