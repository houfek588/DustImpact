# -*- coding: utf-8 -*-
"""
Regression tests found by the validation runs of the charge-escape cases:
  - the external force is the exact gradient of the interpolated potential, so the work along any
    straight path from a point next to the staircase sphere to the wall equals the potential
    difference (interpolated nodal fields missed up to 0.4 V of a 4.6 V sheath for oblique paths),
  - a dust grain whose start point lies on the spacecraft surface impacts the near side (previously
    the ray left the body through the far side and the cloud started on the opposite hemisphere).
"""

import contextlib
import io
import json
import os
import tempfile
import unittest

import numpy as np

from dust_impact.solver.config_loader import setup_simulation_parameters_3d
from dust_impact.geometry import build_simulation_geometry
from dust_impact.solver.sim_core import DustImpactSimulation3D

R_SPHERE = 0.075


def _config(location, direction) -> dict:
    return {
        "geometry": {"source": "analytical", "analytical": {
            "spacecraft": {"type": "sphere", "center": [0.0, 0.0, 0.0], "radius": R_SPHERE, "voltage_V": -5.0},
            "antennas": []}},
        "toggles": {"enable_spis_background_field": True, "enable_plasma_self_field": False,
                    "enable_antenna_particle_collection": False, "enable_rc_circuit_response": False,
                    "enable_antenna_bias_voltage": False, "particle_boundary_mode": "open",
                    "particle_spacecraft_mode": "absorb"},
        "physics": {"ion_mass_amu": 56.0, "total_impact_charge_C": 1e-14, "plasma_injection_mode": "point_cloud",
                    "impact": {"location": location, "direction": direction, "time_delay_s": 0.0}},
        "numeric": {"num_macroparticles": 10, "time_step_s": 1e-8, "num_time_steps": 2,
                    "domain_half_length_m": 0.6, "grid_nodes": 41, "field_precision": "float64"},
        "plotting": {"enable_checkpointing": False},
    }


def _simulation(cfg: dict):
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "cfg.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cfg, f)
        with contextlib.redirect_stdout(io.StringIO()):
            params, toggles, _ = setup_simulation_parameters_3d(0.0, 0.0, config_file=path, output_dir=tmp)
            geom = build_simulation_geometry(params)
            sim = DustImpactSimulation3D(params, toggles, geom)
    return params, sim


class TestConservativeExternalField(unittest.TestCase):

    def test_work_equals_potential_difference_along_oblique_paths(self):
        """Work of the external force per unit charge from next to the sphere to the box equals V(a) - V(b)."""
        params, sim = _simulation(_config([0.3, 0.0, 0.0], [-1.0, 0.0, 0.0]))
        fs = sim.field_solver
        fs.update_background_fields(sim.collector.voltage_ant, 0, sim.collector.voltage_sc)
        x0 = np.array([R_SPHERE + 0.003, 0.0, 0.0])
        s = np.linspace(0.0, 0.45, 20001)
        for theta, phi in ((0, 0), (30, 0), (60, 45), (80, 45)):
            d = np.array([np.cos(np.radians(theta)),
                          np.sin(np.radians(theta)) * np.cos(np.radians(phi)),
                          np.sin(np.radians(theta)) * np.sin(np.radians(phi))])
            pts = x0[None, :] + s[:, None] * d[None, :]
            ax, ay, az = fs.get_accel(pts[:, 0], pts[:, 1], pts[:, 2], 1.0, 1.0)
            work = np.trapz(ax * d[0] + ay * d[1] + az * d[2], s)
            V_ends = fs.interp_field(pts[[0, -1], 0], pts[[0, -1], 1], pts[[0, -1], 2], fs.V_bg_base)
            with self.subTest(theta=theta, phi=phi):
                self.assertAlmostEqual(work, V_ends[0] - V_ends[1], delta=2e-3)


class TestImpactPointOnSurface(unittest.TestCase):

    def test_start_on_surface_impacts_near_side(self):
        """location on the surface with direction into the body: the cloud starts there, not on the far side."""
        params, sim = _simulation(_config([0.0, R_SPHERE, 0.0], [0.0, -1.0, 0.0]))
        np.testing.assert_allclose(params.impact_pos, [0.0, R_SPHERE, 0.0], atol=1e-9)
        np.testing.assert_allclose(params.impact_normal, [0.0, 1.0, 0.0], atol=1e-9)
        self.assertGreater(np.min(sim.particles.y_i), R_SPHERE - 1e-9)

    def test_start_outside_impacts_first_crossing(self):
        params, _ = _simulation(_config([0.0, 0.3, 0.0], [0.0, -1.0, 0.0]))
        np.testing.assert_allclose(params.impact_pos, [0.0, R_SPHERE, 0.0], atol=1e-9)


if __name__ == "__main__":
    unittest.main()
