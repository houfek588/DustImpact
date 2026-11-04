# -*- coding: utf-8 -*-
"""
End-to-end tests that drive the solver from the JSON configurations shipped with the project.

Each configuration is shrunk (grid, number of steps and macro-particles) and run through the same
path as a user run: JSON -> schema validation -> geometry construction -> PIC run -> result file.
The physical invariants are then checked on the written results:
  * every conductor node carries its potential: V_sc on the spacecraft and the circuit voltage
    voltage_ant on antenna a (the self-field vanishes on conductors, Vw_a = 1 on antenna a and 0 on
    all other conductors);
  * the deposited charge equals the charge of the active macro-particles;
  * no active particle lies outside the domain or inside the spacecraft;
  * the charge delivered to each antenna satisfies the Ramo-Shockley balance
    Q_a = sum_p q_p [Vw_a(r_end) - Vw_a(r_start)].
"""

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from dust_impact.common.io import load_results, HAS_H5PY
from dust_impact.geometry import build_simulation_geometry
from dust_impact.geometry.analytical import voxelize_analytical_spacecraft_part
from dust_impact.geometry.surface import check_wire_collision_3d
from dust_impact.physics.charging import calculate_equilibrium_potential, ENV_EARTH, MAT_ALUMINIUM
from dust_impact.solver.config_loader import setup_simulation_parameters_3d
from dust_impact.solver.runner import run_3d_simulation
from dust_impact.solver.sim_core import DustImpactSimulation3D

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def write_shrunk_config(source: str, target_dir: str, grid_nodes: int, steps: int, particles: int,
                        output_format: str = "npz", drop_spacecraft_voltage: bool = False) -> str:
    """
    Copies a project JSON configuration into target_dir with a reduced numerical size and all
    interactive and on-disk plotting disabled. Relative input paths keep resolving against the
    repository root (the current working directory of the test run).

    Returns the path of the written configuration file.
    """
    with open(source, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["numeric"].update(grid_nodes=[grid_nodes] * 3, num_time_steps=steps, num_macroparticles=particles)
    cfg["plotting"].update(
        show_interactive_gui_windows=False, save_plots_to_disk=False, export_vtk=False,
        enable_checkpointing=False, output_format=output_format,
    )
    if drop_spacecraft_voltage:
        geom = cfg["geometry"]
        geom[geom["source"]]["spacecraft"].pop("voltage_V", None)
    path = os.path.join(target_dir, "config.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f)
    return path


def run_config(config_path: str, output_dir: str):
    """
    Runs run_3d_simulation on config_path, then rebuilds the parameters and geometry the run used
    (deterministic) and loads the written result file.

    Returns (params, geometry, results).
    """
    with contextlib.redirect_stdout(io.StringIO()):
        run_3d_simulation(config_file=config_path, output_dir=output_dir, visualize_results=False)
        V_eq = calculate_equilibrium_potential(ENV_EARTH, MAT_ALUMINIUM, verbose=False)
        params, _, plot_cfg = setup_simulation_parameters_3d(
            Vf=V_eq, Vf_antenne=0.0, config_file=config_path, output_dir=output_dir
        )
        geometry = build_simulation_geometry(params)
        results = load_results(plot_cfg.primary_output_filepath)
    return params, geometry, results


class _EndToEndRunMixin:
    """Invariant checks shared by all configurations; subclasses set SOURCE and the run size."""

    SOURCE = ""
    GRID_NODES, STEPS, PARTICLES = 30, 60, 2000
    V_SC_EXPECTED = None

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cfg = write_shrunk_config(cls.SOURCE, cls._tmp.name, cls.GRID_NODES, cls.STEPS, cls.PARTICLES)
        cls.params, cls.geom, cls.results = run_config(cfg, cls._tmp.name)
        cls.V = np.asarray(cls.results["history"]["V"])

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_result_file_is_complete(self):
        """All time steps are written, the signals are finite and the result file lands in output_dir."""
        p, r = self.params, self.results
        n_ant = self.geom.num_antennas
        self.assertEqual(np.asarray(r["voltage_ant"]).shape, (n_ant, p.steps))
        self.assertEqual(np.asarray(r["smooth_total"]).shape, (n_ant, p.steps))
        self.assertEqual(self.V.shape, (p.steps, p.Nx, p.Ny, p.Nz))
        for key in ("smooth_induced", "smooth_collected", "smooth_total", "voltage_ant"):
            self.assertTrue(np.all(np.isfinite(np.asarray(r[key]))), key)
        self.assertTrue(np.all(np.isfinite(self.V)))
        files = [f for f in os.listdir(self._tmp.name) if f.endswith((".npz", ".h5"))]
        self.assertEqual(len(files), 1)

    def test_spacecraft_is_at_configured_potential(self):
        """The spacecraft is an equipotential at V_sc during the whole run."""
        V_sc = self.params.spacecraft_voltage_V if self.V_SC_EXPECTED is None else self.V_SC_EXPECTED
        np.testing.assert_allclose(self.V[:, self.geom.spacecraft_mask_3d], V_sc, atol=1e-4)

    def test_antennas_are_at_circuit_voltage(self):
        """Each antenna is an equipotential at the voltage of its RC circuit in every saved frame."""
        voltage = np.asarray(self.results["voltage_ant"])
        for a, mask in enumerate(self.geom.antenna_masks_3d):
            with self.subTest(antenna=a):
                self.assertTrue(mask.any())
                np.testing.assert_allclose(self.V[:, mask], voltage[a][:, None] * np.ones((1, mask.sum())), atol=1e-4)

    def test_deposited_charge_matches_active_particles(self):
        """
        Frame k stores the charge density deposited before the push of step k, i.e. from the
        particles saved in frame k - 1: sum(rho) dV = q_macro (N_i - N_e).
        """
        p, H = self.params, self.results["history"]
        self.assertEqual(p.plot_stride, 1)
        n_e = np.sum(~np.isnan(np.asarray(H["x_e"])), axis=1)
        n_i = np.sum(~np.isnan(np.asarray(H["x_i"])), axis=1)
        rho = np.asarray(H["rho"])
        Q_grid = rho.reshape(rho.shape[0], -1).sum(axis=1) * p.dx * p.dy * p.dz
        injected = (n_e[:-1] + n_i[:-1]) > 0
        self.assertTrue(np.any(injected), "The impact cloud was never injected")
        Q_particles = p.q_macro * (n_i[:-1] - n_e[:-1])
        np.testing.assert_allclose(Q_grid[1:][injected], Q_particles[injected], atol=1e-9 * p.q_macro * p.N_particles)

    def test_active_particles_stay_in_domain_and_outside_spacecraft(self):
        p, H = self.params, self.results["history"]
        for species in ("e", "i"):
            x, y, z = (np.asarray(H[f"{c}_{species}"]) for c in "xyz")
            ok = ~np.isnan(x)
            with self.subTest(species=species):
                self.assertTrue(np.all(np.abs(x[ok]) <= p.L_x))
                self.assertTrue(np.all(np.abs(y[ok]) <= p.L_y))
                self.assertTrue(np.all(np.abs(z[ok]) <= p.L_z))
                self.assertFalse(np.any(self._inside_spacecraft(x[ok], y[ok], z[ok])))

    def _inside_spacecraft(self, x, y, z):
        """Exact body test for analytical geometry, nearest mask node for SPIS geometry."""
        p = self.params
        if p.geometry.source == "analytical":
            return voxelize_analytical_spacecraft_part(x, y, z, p.geometry.analytical.spacecraft)
        ix = np.clip(np.rint((x - p.x_grid[0]) / p.dx), 0, p.Nx - 1).astype(int)
        iy = np.clip(np.rint((y - p.y_grid[0]) / p.dy), 0, p.Ny - 1).astype(int)
        iz = np.clip(np.rint((z - p.z_grid[0]) / p.dz), 0, p.Nz - 1).astype(int)
        return self.geom.spacecraft_mask_3d[ix, iy, iz]


class TestEndToEndAnalyticalBox(_EndToEndRunMixin, unittest.TestCase):
    """inputs/config_analytical_box.json: box spacecraft, three wire antennas."""
    SOURCE = "inputs/config_analytical_box.json"


class TestEndToEndAnalyticalSphere(_EndToEndRunMixin, unittest.TestCase):
    """inputs/config_analytical_sphere.json: small sphere with four thin wire antennas; electrons are collected."""
    SOURCE = "inputs/config_analytical_sphere.json"
    GRID_NODES = 50


class TestEndToEndSpis(_EndToEndRunMixin, unittest.TestCase):
    """
    config.json: production SPIS geometry (VTK background and weighting potentials).

    The configured spacecraft potential (25 V) is overridden by the SPIS background (8.9 V), and the
    thin antennas are under-resolved on the PIC grid, so this run exercises the harmonic correction
    that puts every conductor at its SPIS potential.
    """
    SOURCE = "config.json"


class TestEndToEndFloatingSpacecraft(unittest.TestCase):
    """Without a configured spacecraft voltage the runner floats the spacecraft at the 0D equilibrium."""

    def test_spacecraft_uses_0d_floating_potential(self):
        V_eq = calculate_equilibrium_potential(ENV_EARTH, MAT_ALUMINIUM, verbose=False)
        with tempfile.TemporaryDirectory() as tmp:
            cfg = write_shrunk_config(
                "inputs/config_analytical_box.json", tmp, 24, 5, 200, drop_spacecraft_voltage=True
            )
            params, geom, results = run_config(cfg, tmp)
        V = np.asarray(results["history"]["V"])
        self.assertGreater(V_eq, 2.0)
        np.testing.assert_allclose(V[:, geom.spacecraft_mask_3d], V_eq, atol=1e-4)


class TestEndToEndChargeBalance(unittest.TestCase):
    """
    Ramo-Shockley charge balance for a run built from inputs/config_analytical_sphere.json through the
    public API (JSON -> parameters -> geometry -> DustImpactSimulation3D.run). The result file does not
    record where removed particles ended, so the final particle state of the simulation object is used.
    """

    def test_antenna_charge_balance(self):
        np.random.seed(3)
        with tempfile.TemporaryDirectory() as tmp:
            cfg = write_shrunk_config("inputs/config_analytical_sphere.json", tmp, 50, 60, 2000)
            with contextlib.redirect_stdout(io.StringIO()):
                params, toggles, _ = setup_simulation_parameters_3d(
                    Vf=0.0, Vf_antenne=0.0, config_file=cfg, output_dir=tmp
                )
                geom = build_simulation_geometry(params)
                sim = DustImpactSimulation3D(params, toggles, geom)
                ens, fs = sim.particles, sim.field_solver
                start = {s: (getattr(ens, f"x_{s}").copy(), getattr(ens, f"y_{s}").copy(), getattr(ens, f"z_{s}").copy())
                         for s in ("e", "i")}
                sim.run()

        def delivered(species: str, a: int) -> float:
            """Sum of Vw_a(r_end) - Vw_a(r_start) over the macro-particles of one species."""
            x, y, z = (getattr(ens, f"{c}_{species}") for c in "xyz")
            active = getattr(ens, f"active_{species}")
            vw_start = fs.interp_field(*start[species], geom.Vw_grids[a])
            vw_end = np.where(active, fs.interp_field(x, y, z, geom.Vw_grids[a]), 0.0)
            wire = geom.antenna_geometries[a]
            collected = (~active) & check_wire_collision_3d(x, y, z, wire.p_start, wire.p_end, wire.radius)
            vw_end[collected] = 1.0
            return float(np.sum(vw_end - vw_start)), int(np.sum(collected))

        total_collected = 0
        for a in range(geom.num_antennas):
            d_i, c_i = delivered("i", a)
            d_e, c_e = delivered("e", a)
            total_collected += c_i + c_e
            Q_expected = params.q_macro * (d_i - d_e)
            Q_signal = float(np.sum(sim.tot_curr[a]) * params.dt)
            with self.subTest(antenna=a):
                self.assertAlmostEqual(Q_signal, Q_expected, delta=1e-9 * params.q_macro * params.N_particles)
        self.assertGreater(total_collected, 0, "No particle reached an antenna; the balance would be vacuous")


class TestCommandLine(unittest.TestCase):
    """The documented CLI (python main.py --config ... --no-visualize) runs a JSON configuration."""

    @unittest.skipUnless(HAS_H5PY, "h5py is required for the default HDF5 output")
    def test_main_script_runs_configuration(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = write_shrunk_config("inputs/config_analytical_box.json", tmp, 20, 10, 300, output_format="h5")
            proc = subprocess.run(
                [sys.executable, "main.py", "--config", cfg, "--output-dir", tmp, "--no-visualize", "--no-checkpoint"],
                cwd=REPO_ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
            outputs = [f for f in os.listdir(tmp) if f.endswith(".h5")]
            self.assertEqual(len(outputs), 1)
            with contextlib.redirect_stdout(io.StringIO()):
                results = load_results(os.path.join(tmp, outputs[0]))
        self.assertEqual(np.asarray(results["voltage_ant"]).shape, (3, 10))


if __name__ == "__main__":
    unittest.main()
