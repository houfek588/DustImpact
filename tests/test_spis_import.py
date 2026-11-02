# -*- coding: utf-8 -*-
"""
Import of SPIS VTK data onto the PIC grid.

  * Sampling: VTK ImageData orders its points with x varying fastest, the solver indexes fields as
    [i, j, k] = (x, y, z); grid nodes inside the unmeshed conductors take the conductor potential.
  * Geometry: the conductors of the SPIS model land at their SPIS positions.
  * Background: with "source": "spis" and a background_potential_file, the conductor potentials and
    the potential profile are those of the file.
"""

import contextlib
import io
import json
import os
import tempfile
import unittest

import numpy as np
import pyvista as pv
from scipy.ndimage import binary_dilation

from dust_impact.geometry import build_simulation_geometry
from dust_impact.geometry.spis_loader import sample_mesh_to_grid
from dust_impact.solver.config_loader import setup_simulation_parameters_3d

BACKGROUND_FILE = "inputs/spis_V_bg.vtk"
# Conductor potentials of the SPIS solution (Vw = 1 nodes of the SPIS mesh) [V]
SPIS_SPACECRAFT_V = 8.916
SPIS_ANTENNA_V = (11.641, 11.430, 11.579)


def _image_grid(n, half_length, origin_shift=(0.0, 0.0, 0.0)):
    """Cartesian ImageData with n = (nx, ny, nz) nodes spanning [-L, L] along each axis (shifted)."""
    h = [2.0 * L / (k - 1) for L, k in zip(half_length, n)]
    origin = [-L + s for L, s in zip(half_length, origin_shift)]
    return pv.ImageData(dimensions=n, spacing=h, origin=origin)


class TestSampleMeshToGrid(unittest.TestCase):

    def test_axis_order_matches_solver_indexing(self):
        """A linear field x + 10 y + 100 z is sampled exactly and indexed [i, j, k] = (x, y, z)."""
        source = _image_grid((31, 27, 23), (3.0, 3.0, 3.0))
        P = np.asarray(source.points)
        source.point_data["Potential"] = P[:, 0] + 10.0 * P[:, 1] + 100.0 * P[:, 2]

        target = _image_grid((7, 9, 11), (2.0, 1.5, 1.0), origin_shift=(0.1, -0.2, 0.3))
        V = sample_mesh_to_grid(source, target, "linear")
        self.assertEqual(V.shape, (7, 9, 11))
        x0, y0, z0 = target.origin
        hx, hy, hz = target.spacing
        x = x0 + hx * np.arange(7)
        y = y0 + hy * np.arange(9)
        z = z0 + hz * np.arange(11)
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        np.testing.assert_allclose(V, X + 10.0 * Y + 100.0 * Z, atol=1e-9)

    def test_nodes_inside_conductor_take_its_surface_potential(self):
        """Grid nodes in the unmeshed interior of the spacecraft get its SPIS potential, not 0 V."""
        source = pv.read(BACKGROUND_FILE)
        # Cube well inside the spacecraft body (|x| <= 1.25, |y| <= 1.55, |z| <= 1.35)
        V = sample_mesh_to_grid(source, _image_grid((5, 5, 5), (0.8, 0.8, 0.8)), BACKGROUND_FILE)
        np.testing.assert_allclose(V, SPIS_SPACECRAFT_V, atol=1e-3)


class TestSpisGeometryImport(unittest.TestCase):
    """config.json with weighting_threshold 0.85 on a 61^3 grid (antenna 1 lies on the node line x = z = 0)."""

    GRID_NODES = 61
    # Coarser masks than the config.json value, so the thin antennas are resolved on this grid
    WEIGHTING_THRESHOLD = 0.85

    @classmethod
    def setUpClass(cls):
        with open("config.json", "r", encoding="utf-8") as f:
            cfg = json.load(f)
        cfg["geometry"]["spis"]["weighting_threshold"] = cls.WEIGHTING_THRESHOLD
        cfg["numeric"]["grid_nodes"] = [cls.GRID_NODES] * 3
        cls._tmp = tempfile.TemporaryDirectory()
        path = os.path.join(cls._tmp.name, "config.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cfg, f)
        with contextlib.redirect_stdout(io.StringIO()):
            cls.params, _, _ = setup_simulation_parameters_3d(Vf=0.0, Vf_antenne=0.0, config_file=path)
            cls.geom = build_simulation_geometry(cls.params)
        p = cls.params
        cls.X, cls.Y, cls.Z = np.meshgrid(p.x_grid, p.y_grid, p.z_grid, indexing="ij")

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _centroid(self, mask):
        return np.array([self.X[mask].mean(), self.Y[mask].mean(), self.Z[mask].mean()])

    def test_conductors_land_at_spis_positions(self):
        """
        The SPIS spacecraft is a box with a top plate (z up to 1.85 m, x within +-1.5 m), antenna 1
        points along +y, antennas 2 and 3 along (+-x, -y) in the z = 0 plane. Swapped axes would move
        the top plate to +x and the antennas out of the z = 0 plane.
        """
        sc = self.geom.spacecraft_mask_3d
        h = self.params.dx
        self.assertLess(np.max(np.abs(self.X[sc])), 1.5 + h)
        self.assertGreater(np.max(self.Z[sc]), 1.85 - h)
        c = self._centroid(sc)
        self.assertLess(abs(c[0]), 0.05)
        self.assertGreater(c[2], 0.05)
        expected = [(0.0, 5.0, 0.0), (4.3, -3.15, 0.0), (-4.3, -3.15, 0.0)]
        for a, (mask, centre) in enumerate(zip(self.geom.antenna_masks_3d, expected)):
            with self.subTest(antenna=a):
                self.assertTrue(mask.any())
                np.testing.assert_allclose(self._centroid(mask), centre, atol=0.5)

    def test_conductor_potentials_are_taken_from_background(self):
        """The configured 25 V is ignored; spacecraft and antennas carry their SPIS potentials."""
        self.assertAlmostEqual(self.params.spacecraft_voltage_V, SPIS_SPACECRAFT_V, places=3)
        np.testing.assert_allclose(self.geom.V_bg[self.geom.spacecraft_mask_3d], SPIS_SPACECRAFT_V, atol=1e-3)
        for mask, v in zip(self.geom.antenna_masks_3d, SPIS_ANTENNA_V):
            np.testing.assert_allclose(self.geom.V_bg[mask], v, atol=1e-3)



class TestSpisBackgroundProfile(TestSpisGeometryImport):
    """
    weighting_threshold 0.99: the conductor masks hug the SPIS surfaces (the thin antennas are not
    resolved at this grid spacing), so the background must match the SPIS potential outside them.
    With the default 0.85 the masks reach into the sheath, where the PIC conductor nodes are raised to
    the conductor potential, and the deviation grows to ~0.2 V rms.
    """

    WEIGHTING_THRESHOLD = 0.99
    test_conductors_land_at_spis_positions = None
    test_conductor_potentials_are_taken_from_background = None

    def test_background_reproduces_spis_profile(self):
        """Away from the conductor nodes the background equals the SPIS potential sampled on the grid."""
        p, g = self.params, self.geom
        source = pv.read(BACKGROUND_FILE)
        V_spis = sample_mesh_to_grid(source, _image_grid((p.Nx, p.Ny, p.Nz), (p.L_x, p.L_y, p.L_z)), BACKGROUND_FILE)
        conductors = g.spacecraft_mask_3d.copy()
        for m in g.antenna_masks_3d:
            conductors |= m
        far = ~binary_dilation(conductors, iterations=2)
        dV = np.abs(g.V_bg.astype(np.float64) - V_spis)[far]
        self.assertLess(np.sqrt(np.mean(dV**2)), 0.01)
        self.assertLess(np.max(dV), 0.3)


if __name__ == "__main__":
    unittest.main()
