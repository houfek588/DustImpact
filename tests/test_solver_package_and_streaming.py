# -*- coding: utf-8 -*-
"""
Unit tests for dust_impact.solver package reorganization,
sim3d backward compatibility, and streaming I/O memory capping.
"""

import os
import tempfile
import unittest
import numpy as np

# Canonical solver package
import dust_impact.solver as solver
from dust_impact.solver.config_loader import setup_simulation_parameters_3d
from dust_impact.solver.sim_core import DustImpactSimulation3D
from dust_impact.solver.runner import run_3d_simulation
from dust_impact.geometry import build_simulation_geometry
from dust_impact.common.io import HDF5SimulationWriter, load_results_h5, HAS_H5PY

# Backward compatibility sim3d package
import dust_impact.sim3d as sim3d
import dust_impact.sim3d.config_loader as sim3d_config_loader
import dust_impact.sim3d.sim_core as sim3d_core
import dust_impact.sim3d.runner as sim3d_runner
import dust_impact.sim3d.vtk_reader as sim3d_vtk
import dust_impact.sim3d.voxelizer as sim3d_vox


class TestSolverPackageAndStreaming(unittest.TestCase):

    def test_solver_package_exports(self):
        """Verifies canonical exports from dust_impact.solver."""
        self.assertTrue(hasattr(solver, "DustImpactSimulation3D"))
        self.assertTrue(hasattr(solver, "ParticleEnsemble"))
        self.assertTrue(hasattr(solver, "FieldSolver3D"))
        self.assertTrue(hasattr(solver, "AntennaCircuitCollector"))
        self.assertTrue(hasattr(solver, "setup_simulation_parameters_3d"))
        self.assertTrue(hasattr(solver, "load_and_validate_config"))
        self.assertTrue(hasattr(solver, "run_3d_simulation"))
        self.assertTrue(hasattr(solver, "load_and_interpolate_vtk"))

    def test_sim3d_backward_compatibility(self):
        """Verifies 100% backward compatibility of dust_impact.sim3d imports."""
        # Top-level sim3d delegates to solver
        self.assertIs(sim3d.DustImpactSimulation3D, solver.DustImpactSimulation3D)
        self.assertIs(sim3d.setup_simulation_parameters_3d, solver.setup_simulation_parameters_3d)
        self.assertIs(sim3d.run_3d_simulation, solver.run_3d_simulation)
        self.assertIs(sim3d.load_and_interpolate_vtk, solver.load_and_interpolate_vtk)

        # Submodule re-exports
        self.assertIs(sim3d_config_loader.setup_simulation_parameters_3d, solver.setup_simulation_parameters_3d)
        self.assertIs(sim3d_core.DustImpactSimulation3D, solver.DustImpactSimulation3D)
        self.assertIs(sim3d_runner.run_3d_simulation, solver.run_3d_simulation)
        self.assertTrue(hasattr(sim3d_vtk, "load_and_interpolate_vtk"))
        self.assertTrue(hasattr(sim3d_vox, "detect_metal_mask_3d"))

    @unittest.skipUnless(HAS_H5PY, "h5py is required for streaming tests")
    def test_streaming_memory_capping(self):
        """Verifies that stream_to_disk=True caps RAM history to 1 frame while disk contains all frames."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            h5_path = os.path.join(tmp_dir, "stream_test.h5")

            params, toggles, _ = setup_simulation_parameters_3d(
                Vf=-5.0, Vf_antenne=1.0, config_file="inputs/config_analytical_sphere.json"
            )
            params.steps = 20
            params.save_interval = 2
            params.N_particles = 500

            geom = build_simulation_geometry(params)
            sim = DustImpactSimulation3D(params, toggles, geom)

            writer = HDF5SimulationWriter(
                filepath=h5_path,
                grid_shape=(params.Nx, params.Ny, params.Nz),
                num_antennas=sim.num_antennas,
                num_steps=params.steps,
                metadata=sim.get_metadata_dict()
            )

            # Run with stream_to_disk=True
            results = sim.run(h5_writer=writer, stream_to_disk=True)

            # 1. In-memory history is strictly capped at length 1 (constant RAM)
            self.assertEqual(len(sim.history['V']), 1, "RAM history['V'] should hold only 1 frame in stream mode")
            self.assertEqual(len(sim.history['rho']), 1, "RAM history['rho'] should hold only 1 frame in stream mode")
            self.assertEqual(len(sim.history['x_e']), 1, "RAM history['x_e'] should hold only 1 frame in stream mode")
            self.assertTrue(results.get('streamed_to_disk', False))

            # 2. File on disk contains all expected saved frames: 20 steps / 2 + 1 (final) = 11 frames
            loaded = load_results_h5(h5_path)
            self.assertIn('history', loaded)
            self.assertIn('V', loaded['history'])
            num_disk_frames = len(loaded['history']['V'])
            self.assertGreaterEqual(num_disk_frames, 10, f"Expected >= 10 frames on disk, got {num_disk_frames}")

            # 3. Signals are successfully persisted in streaming file
            self.assertIn('smooth_total', loaded)
            self.assertIn('voltage_ant', loaded)

    def test_in_memory_history_accumulation(self):
        """Verifies that default stream_to_disk=False accumulates all frames in RAM as before."""
        params, toggles, _ = setup_simulation_parameters_3d(
            Vf=-5.0, Vf_antenne=1.0, config_file="inputs/config_analytical_sphere.json"
        )
        params.steps = 10
        params.save_interval = 2
        params.N_particles = 200

        geom = build_simulation_geometry(params)
        sim = DustImpactSimulation3D(params, toggles, geom)

        # Run standard in-memory
        results = sim.run(stream_to_disk=False)
        self.assertGreater(len(sim.history['V']), 1, "RAM history should accumulate frames when streaming is off")
        self.assertFalse(results.get('streamed_to_disk', False))


if __name__ == "__main__":
    unittest.main()
