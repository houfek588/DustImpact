# -*- coding: utf-8 -*-
"""
Floating spacecraft body (toggles.enable_floating_spacecraft) against analytical results:
  - the coupled electrode circuit C_tot dV/dt = I - G (V - V_bias) (exact linear ODE solution),
  - charge conservation of an isolated floating sphere: C_vac V_sc = q [Vw_sc(r) - Vw_sc(r0)] for a
    charge leaving it or landing on it (Ramo-Shockley image charge), close to q (R/r - R/r0) / (4 pi eps0 R),
  - the vacuum capacitance of a sphere in a grounded box, between the sphere-in-sphere bounds.
"""

import contextlib
import io
import json
import os
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
from scipy.linalg import expm

from dust_impact.physics.constants import e, eps_0
from dust_impact.solver.config_loader import setup_simulation_parameters_3d
from dust_impact.geometry import build_simulation_geometry
from dust_impact.solver.collector import AntennaCircuitCollector
from dust_impact.solver.sim_core import DustImpactSimulation3D

R_SPHERE = 0.3
L_BOX = 1.5


def _sphere_config(floating: bool, steps: int, num_macroparticles: int = 1) -> dict:
    """Isolated sphere (no antennas, no fields acting on particles) in a grounded box."""
    return {
        "geometry": {"source": "analytical", "analytical": {
            "spacecraft": {"type": "sphere", "center": [0.0, 0.0, 0.0], "radius": R_SPHERE, "voltage_V": 0.0},
            "antennas": []}},
        "toggles": {"enable_spis_background_field": False, "enable_plasma_self_field": False,
                    "enable_antenna_particle_collection": False, "enable_rc_circuit_response": True,
                    "enable_antenna_bias_voltage": False, "particle_boundary_mode": "open",
                    "particle_spacecraft_mode": "absorb", "enable_floating_spacecraft": floating},
        "physics": {"ion_mass_amu": 1.0e6, "total_impact_charge_C": 1e-13 * num_macroparticles,
                    "plasma_injection_mode": "point_cloud",
                    "impact": {"location": [R_SPHERE, 0.0, 0.0], "direction": [-1.0, 0.0, 0.0],
                               "normal": [1.0, 0.0, 0.0], "time_delay_s": 0.0}},
        "numeric": {"num_macroparticles": num_macroparticles, "time_step_s": 1e-7, "num_time_steps": steps,
                    "domain_half_length_m": L_BOX, "grid_nodes": 41, "field_precision": "float64"},
        "plotting": {"enable_checkpointing": False},
    }


def _build_simulation(cfg: dict):
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "cfg.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cfg, f)
        with contextlib.redirect_stdout(io.StringIO()):
            params, toggles, _ = setup_simulation_parameters_3d(0.0, 0.0, config_file=path, output_dir=tmp)
            geom = build_simulation_geometry(params)
            sim = DustImpactSimulation3D(params, toggles, geom)
    return params, geom, sim


def _launch_single_ion(sim, position, velocity) -> None:
    """Replaces the impact cloud by one ion with the given state (electrons inactive)."""
    P = sim.particles
    P.cloud_injected = True
    P.active_e[:] = False
    P.x_i[0], P.y_i[0], P.z_i[0] = position
    P.vx_i[0], P.vy_i[0], P.vz_i[0] = velocity
    P.active_i[:] = False
    P.active_i[0] = True


class TestCoupledCircuit(unittest.TestCase):
    """AntennaCircuitCollector against the exact solution of the linear circuit ODE."""

    def _collector(self, floating: bool, C_vac=None):
        params = SimpleNamespace(
            steps=4001, dt=1e-8, C_ant=[5e-12], R_ant=[2e6], V_bias=[1.0],
            C_sc=20e-12, R_sc=1e6, V_sc_bias=-2.0, circuit_solver="rk4",
        )
        toggles = SimpleNamespace(
            enable_floating_spacecraft=floating, enable_rc_circuit_response=True,
            enable_antenna_bias_voltage=True, circuit_solver="rk4",
        )
        return AntennaCircuitCollector(params, toggles, num_antennas=1, capacitance_matrix=C_vac), params

    def test_floating_circuit_matches_matrix_exponential(self):
        """Constant currents into a coupled antenna/spacecraft pair: V(t) = V_inf + exp(-A t) (V0 - V_inf)."""
        C_vac = np.array([[3e-12, -1e-12], [-1e-12, 8e-12]])
        col, p = self._collector(True, C_vac)
        I = np.array([2e-7, -5e-7])
        for k in range(p.steps):
            col.record_step_currents(k, np.zeros(2), np.zeros(2), I, np.zeros(2))
            col.update_circuit(k)

        C = C_vac + np.diag([5e-12, 20e-12])
        G = np.diag([1.0 / 2e6, 1.0 / 1e6])
        V_b = np.array([1.0, -2.0])
        A = np.linalg.solve(C, G)
        V_inf = V_b + np.linalg.solve(G, I)
        t = p.steps * p.dt
        V_exact = V_inf + expm(-A * t) @ (V_b - V_inf)
        np.testing.assert_allclose(col.voltage[:, -1], V_exact, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(col.voltage_sc, col.voltage[1])

    def test_fixed_spacecraft_keeps_independent_antenna_rc(self):
        """Without the toggle each antenna is an independent RC circuit with C = capacitance_F."""
        col, p = self._collector(False)
        self.assertIsNone(col.voltage_sc)
        I = np.array([3e-7])
        for k in range(p.steps):
            col.record_step_currents(k, np.zeros(1), np.zeros(1), I, np.zeros(1))
            col.update_circuit(k)
        t = p.steps * p.dt
        tau = 2e6 * 5e-12
        expected = 1.0 + 3e-7 * 2e6 * (1.0 - np.exp(-t / tau))
        self.assertAlmostEqual(col.voltage_ant[0, -1], expected, delta=1e-9 * abs(expected))


class TestFloatingSphere(unittest.TestCase):
    """Isolated floating sphere (no bias resistor) charged only by Ramo-Shockley and collected currents."""

    def test_vacuum_capacitance_of_sphere_in_grounded_box(self):
        """C lies between the sphere-in-sphere capacitances for outer radii L and sqrt(3) L, with a margin
        for the voxelized sphere (R/dx = 4, which acts as a smaller sphere)."""
        _, geom, _ = _build_simulation(_sphere_config(floating=True, steps=2))
        C = geom.capacitance_matrix[0, 0]
        c_inner = 4.0 * np.pi * eps_0 / (1.0 / R_SPHERE - 1.0 / (np.sqrt(3.0) * L_BOX))
        c_outer = 4.0 * np.pi * eps_0 / (1.0 / R_SPHERE - 1.0 / L_BOX)
        self.assertGreater(C, 0.85 * c_inner)
        self.assertLess(C, 1.05 * c_outer)

    def _run_single_ion(self, r0: float, v_r: float, steps: int):
        cfg = _sphere_config(floating=True, steps=steps)
        cfg["geometry"]["analytical"]["spacecraft"].pop("voltage_V")
        params, geom, sim = _build_simulation(cfg)
        _launch_single_ion(sim, (r0, 0.0, 0.0), (v_r, 0.0, 0.0))
        with contextlib.redirect_stdout(io.StringIO()):
            sim.run()
        return params, geom, sim

    def test_departing_charge_charges_isolated_sphere(self):
        """C_vac V_sc(t) = q [Vw_sc(r(t)) - Vw_sc(r0)] exactly (charge conservation) and close to the
        analytical image-charge result q (R/r - R/r0) / C for a point charge leaving the sphere."""
        r0, v_r, steps = 0.45, 2.0e4, 301
        params, geom, sim = self._run_single_ion(r0, v_r, steps)
        q = params.q_macro
        C = geom.capacitance_matrix[0, 0]
        x = sim.particles.x_i[0]
        self.assertTrue(sim.particles.active_i[0])

        def vw(px):
            return sim.field_solver.interp_field(np.array([px]), np.zeros(1), np.zeros(1), geom.Vw_sc)[0]

        V_expected = q * (vw(x) - vw(r0)) / C
        self.assertAlmostEqual(sim.collector.voltage_sc[-1] / V_expected, 1.0, delta=1e-6)
        # Analytical sphere image charge; tolerance covers the grounded box and the voxelized sphere
        V_analytic = q * (R_SPHERE / x - R_SPHERE / r0) / (4.0 * np.pi * eps_0 * R_SPHERE)
        self.assertLess(V_expected, 0.0)
        self.assertAlmostEqual(sim.collector.voltage_sc[-1] / V_analytic, 1.0, delta=0.15)

    def test_landing_charge_is_collected_by_sphere(self):
        """An ion absorbed by the sphere leaves it with charge q [1 - Vw_sc(r0)]: V_sc = that / C_vac,
        and the delivered current is recorded as collected current of the spacecraft electrode."""
        r0, v_r, steps = 0.6, -2.0e4, 301
        params, geom, sim = self._run_single_ion(r0, v_r, steps)
        self.assertFalse(sim.particles.active_i[0])
        q = params.q_macro
        C = geom.capacitance_matrix[0, 0]
        vw0 = sim.field_solver.interp_field(np.array([r0]), np.zeros(1), np.zeros(1), geom.Vw_sc)[0]
        self.assertAlmostEqual(sim.collector.voltage_sc[-1] / (q * (1.0 - vw0) / C), 1.0, delta=1e-6)
        collected = np.sum(sim.collector.col_curr_i_all[0]) * params.dt
        induced = np.sum(sim.collector.ind_curr_i_all[0]) * params.dt
        self.assertAlmostEqual((collected + induced) / (q * (1.0 - vw0)), 1.0, delta=1e-9)
        self.assertGreater(collected, 0.0)

    def test_fixed_spacecraft_has_no_spacecraft_signal(self):
        """With the toggle off the spacecraft stays an ideal fixed-potential conductor (no V_sc output)."""
        _, _, sim = _build_simulation(_sphere_config(floating=False, steps=2))
        self.assertIsNone(sim.collector.voltage_sc)
        with contextlib.redirect_stdout(io.StringIO()):
            res = sim.run()
        self.assertNotIn('voltage_sc', res)


if __name__ == "__main__":
    unittest.main()
