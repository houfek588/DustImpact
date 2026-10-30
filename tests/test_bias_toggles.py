"""
Consistency tests for the antenna bias and static background toggles.

The static background V_bg_base contains each antenna at its reference potential V_ref = p.V_bias.
The external field must equal the field of the background that is actually requested:
  background on  + bias on  -> V_bg_base
  background on  + bias off -> V_bg_base - V_bias * Vw   (bias removed)
  background off + bias on  -> V_bias * Vw               (bias in vacuum)
  background off + bias off -> 0
"""

import unittest
import numpy as np

from dust_impact.solver.config_loader import SimulationParams3D, SimulationToggles3D
from dust_impact.solver.field_solver import FieldSolver3D
from dust_impact.solver.collector import AntennaCircuitCollector


class TestBiasToggles(unittest.TestCase):
    def setUp(self):
        self.N = 12
        self.V_bias = 5.0
        self.V_sc = -3.0
        self.params = SimulationParams3D(
            num_macroparticles=1,
            simulation_duration_s=2e-8,
            time_step_s=1e-9,
            grid_nodes_x=self.N,
            grid_nodes_y=self.N,
            grid_nodes_z=self.N,
        )
        self.params.V_bias = [self.V_bias]
        p = self.params
        X, Y, Z = np.meshgrid(p.x_grid, p.y_grid, p.z_grid, indexing='ij')
        r = np.sqrt((X - 0.5)**2 + Y**2 + Z**2)
        self.Vw = np.exp(-r**2)
        self.Ew = np.gradient(-self.Vw, p.dx, p.dy, p.dz)
        # Static background: a spacecraft-like part plus the antenna held at its bias
        self.V_other = self.V_sc * np.exp(-(X**2 + Y**2 + Z**2))
        self.V_base = self.V_other + self.V_bias * self.Vw
        self.E_base = np.gradient(-self.V_base, p.dx, p.dy, p.dz)
        self.E_other = np.gradient(-self.V_other, p.dx, p.dy, p.dz)

    def _run(self, bg_on: bool, bias_on: bool):
        toggles = SimulationToggles3D(
            enable_spis_background_field=bg_on,
            enable_antenna_bias_voltage=bias_on,
        )
        mask = np.zeros((self.N, self.N, self.N), dtype=bool)
        fs = FieldSolver3D(
            self.params, toggles, self.V_base, *self.E_base,
            [self.Ew[0]], [self.Ew[1]], [self.Ew[2]], mask, num_antennas=1
        )
        col = AntennaCircuitCollector(self.params, toggles, num_antennas=1)
        for step in range(self.params.steps):
            col.update_circuit(step)
        last = self.params.steps - 1
        fs.update_background_fields(col.voltage_ant, last)
        V = fs.background_potential(col.voltage_ant, last, [self.Vw])
        return fs, col, V

    def test_background_on_bias_on(self):
        fs, col, V = self._run(True, True)
        np.testing.assert_allclose(col.voltage_ant[0, -1], self.V_bias)
        np.testing.assert_allclose(fs.Ex_bg, self.E_base[0], atol=1e-12)
        np.testing.assert_allclose(V, self.V_base, atol=1e-12)

    def test_background_on_bias_off_removes_bias(self):
        fs, col, V = self._run(True, False)
        np.testing.assert_allclose(col.voltage_ant[0], 0.0)
        np.testing.assert_allclose(fs.Ex_bg, self.E_other[0], atol=1e-12)
        np.testing.assert_allclose(V, self.V_other, atol=1e-12)

    def test_background_off_bias_on_keeps_bias_in_vacuum(self):
        fs, col, V = self._run(False, True)
        np.testing.assert_allclose(fs.Ex_bg, self.V_bias * self.Ew[0], atol=1e-12)
        np.testing.assert_allclose(V, self.V_bias * self.Vw, atol=1e-12)

    def test_background_off_bias_off_is_field_free(self):
        fs, col, V = self._run(False, False)
        np.testing.assert_allclose(fs.Ex_bg, 0.0, atol=1e-12)
        np.testing.assert_allclose(V, 0.0, atol=1e-12)

    def test_antenna_perturbation_reaches_field_when_background_off(self):
        """With the background disabled the dynamic antenna response must still act on particles."""
        toggles = SimulationToggles3D(enable_spis_background_field=False)
        mask = np.zeros((self.N, self.N, self.N), dtype=bool)
        fs = FieldSolver3D(
            self.params, toggles, self.V_base, *self.E_base,
            [self.Ew[0]], [self.Ew[1]], [self.Ew[2]], mask, num_antennas=1
        )
        voltage = np.full((1, self.params.steps), 2.0)
        fs.update_background_fields(voltage, 3)
        np.testing.assert_allclose(fs.Ex_bg, 2.0 * self.Ew[0], atol=1e-12)


if __name__ == "__main__":
    unittest.main()
