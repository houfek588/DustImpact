# -*- coding: utf-8 -*-
"""
Consistency of the static electrostatic inputs with the field model of the PIC solver.

The Ramo-Shockley signal q * dVw/dt is only consistent with the self-field when the weighting
potential Vw_a obeys the same boundary conditions as FieldSolver3D.solve_poisson:
  * Vw_a = 1 on antenna a,
  * Vw_a = 0 on the spacecraft and on all other antennas (grounded conductors),
  * Vw_a = 0 on the outer boundary of the PIC domain (grounded box),
  * Laplacian(Vw_a) = 0 in vacuum.
These tests check the weighting potentials produced by both geometry sources (analytical, SPIS VTK)
and the synthetic background potential used when no input file is given.
"""

import contextlib
import io
import unittest

import numpy as np
from scipy.ndimage import binary_dilation

from dust_impact.geometry.analytical import build_analytical_simulation_geometry
from dust_impact.solver import load_and_interpolate_vtk
from dust_impact.solver.config_loader import (
    SimulationParams3D, VTKFilesConfig, setup_simulation_parameters_3d,
)


def _boundary_values(field: np.ndarray) -> np.ndarray:
    """All values on the six outer faces of a 3D grid."""
    return np.concatenate([
        field[0].ravel(), field[-1].ravel(),
        field[:, 0].ravel(), field[:, -1].ravel(),
        field[:, :, 0].ravel(), field[:, :, -1].ravel(),
    ])


def _ramo_violations(Vw_grids, antenna_masks, spacecraft_mask, h: float) -> list:
    """
    Measures how far each weighting potential deviates from the Ramo-Shockley boundary value problem.

    Returns one dict per antenna with the maximum |Vw| on the outer boundary and on the other
    conductors, the number of antenna nodes on the outer boundary, the mean Vw on the own antenna
    and the maximum normalised discrete Laplacian h^2 |Lap Vw| / 6 in vacuum at least two cells
    away from any conductor.
    """
    conductors = spacecraft_mask.copy()
    for m in antenna_masks:
        conductors |= m
    vacuum = ~binary_dilation(conductors, iterations=2)[1:-1, 1:-1, 1:-1]

    report = []
    for a, (Vw, own) in enumerate(zip(Vw_grids, antenna_masks)):
        others = spacecraft_mask.copy()
        for b, m in enumerate(antenna_masks):
            if b != a:
                others |= m
        lap = (
            Vw[2:, 1:-1, 1:-1] + Vw[:-2, 1:-1, 1:-1] + Vw[1:-1, 2:, 1:-1] + Vw[1:-1, :-2, 1:-1]
            + Vw[1:-1, 1:-1, 2:] + Vw[1:-1, 1:-1, :-2] - 6.0 * Vw[1:-1, 1:-1, 1:-1]
        ) / 6.0
        report.append({
            "own_mean": float(np.mean(Vw[own])),
            "boundary_max": float(np.max(np.abs(_boundary_values(Vw)))),
            "antenna_nodes_on_boundary": int(np.sum(_boundary_values(own.astype(float)))),
            "other_conductors_max": float(np.max(np.abs(Vw[others]))) if others.any() else 0.0,
            "laplacian_max": float(np.max(np.abs(lap[vacuum]))),
        })
    return report


class TestWeightingPotentials(unittest.TestCase):

    def _assert_ramo_conditions(self, report, tol: float):
        for a, r in enumerate(report):
            with self.subTest(antenna=a):
                self.assertAlmostEqual(r["own_mean"], 1.0, places=6)
                self.assertEqual(r["antenna_nodes_on_boundary"], 0, "Antenna is truncated by the domain")
                self.assertLess(r["boundary_max"], tol, "Vw does not vanish on the grounded outer boundary")
                self.assertLess(r["other_conductors_max"], tol, "Vw does not vanish on the other conductors")
                self.assertLess(r["laplacian_max"], tol, "Vw is not harmonic in vacuum")

    @unittest.expectedFailure
    def test_analytical_geometry_weighting_potentials(self):
        """
        Weighting potentials of the analytical geometry solve the Ramo-Shockley problem on the PIC grid.

        KNOWN FAILURE: geometry.analytical.solve_laplace_dirichlet_3d keeps the Dirichlet nodes as unit
        rows (non-symmetric matrix) and stops PyAMG-GMRES after 50 iterations at a preconditioned
        tolerance of 1e-5. The local residual next to the antennas reaches h^2 |Lap Vw| / 6 ~ 3e-2 and
        Vw differs from a converged solution by up to 0.055 (about 10 % locally).
        """
        with contextlib.redirect_stdout(io.StringIO()):
            params, _, _ = setup_simulation_parameters_3d(
                Vf=0.0, Vf_antenne=0.0, config_file="inputs/config_analytical_sphere.json"
            )
            geom = build_analytical_simulation_geometry(params, params.geometry.analytical)
        self.assertGreater(geom.num_antennas, 0)
        report = _ramo_violations(
            geom.Vw_grids, geom.antenna_masks_3d, geom.spacecraft_mask_3d, params.dx
        )
        self._assert_ramo_conditions(report, tol=1e-6)

    @classmethod
    def _spis_geometry(cls):
        """SPIS geometry of config.json (|x| <= 10 m, so no conductor is cut by the box) on a 40^3 grid."""
        if not hasattr(cls, "_spis_cache"):
            with contextlib.redirect_stdout(io.StringIO()):
                params = SimulationParams3D(
                    Vf=25.0, spacecraft_voltage_V=25.0,
                    domain_half_length_x_m=10.0, domain_half_length_y_m=10.0, domain_half_length_z_m=10.0,
                    grid_nodes_x=40, grid_nodes_y=40, grid_nodes_z=40,
                    vtk_files=VTKFilesConfig(
                        spis_background_potential_file="inputs/spis_V_bg.vtk",
                        spacecraft_weighting_file="inputs/spis_Vw_body.vtk",
                        antenna_weighting_files=[
                            "inputs/spis_Vw_ant1.vtk", "inputs/spis_Vw_ant2.vtk", "inputs/spis_Vw_ant3.vtk",
                        ],
                    ),
                )
                cls._spis_cache = (params, load_and_interpolate_vtk(params))
        return cls._spis_cache

    def test_spis_weighting_potentials(self):
        """
        Weighting potentials loaded from SPIS VTK files satisfy the Ramo-Shockley boundary conditions on
        the PIC grid, although the SPIS solutions live on a sphere of radius 25 m and the thin antennas
        are under-resolved.

        Harmonicity is not required here: the loader keeps the Laplacian of the sampled SPIS field (a
        piecewise-linear interpolant on a tetrahedral mesh), whose 7-point Laplacian reaches
        h^2 |Lap Vw| / 6 ~ 0.04 next to the antennas.
        """
        params, geom = self._spis_geometry()
        self.assertEqual(geom.num_antennas, 3)
        report = _ramo_violations(geom.Vw_grids, geom.antenna_masks_3d, geom.spacecraft_mask_3d, params.dx)
        for a, r in enumerate(report):
            with self.subTest(antenna=a):
                self.assertAlmostEqual(r["own_mean"], 1.0, places=6)
                self.assertEqual(r["antenna_nodes_on_boundary"], 0, "Antenna is truncated by the domain")
                self.assertLess(r["boundary_max"], 1e-6, "Vw does not vanish on the grounded outer boundary")
                self.assertLess(r["other_conductors_max"], 1e-6, "Vw does not vanish on the other conductors")

    def test_spis_background_holds_conductor_potentials(self):
        """
        The SPIS background puts the spacecraft at the configured V_sc and each antenna at the potential
        of its surface in the SPIS solution, with disjoint conductor masks.
        """
        params, geom = self._spis_geometry()
        masks = [geom.spacecraft_mask_3d] + list(geom.antenna_masks_3d)
        self.assertEqual(int(sum(m.sum() for m in masks)), int(np.logical_or.reduce(masks).sum()),
                         "Conductor masks overlap")
        np.testing.assert_allclose(geom.V_bg[geom.spacecraft_mask_3d], 25.0, rtol=1e-6)
        # Antenna surface potentials of the SPIS solution (Vw = 1 nodes of the SPIS mesh)
        for mask, v_spis, v_bias in zip(geom.antenna_masks_3d, (11.641, 11.430, 11.579), params.V_bias):
            self.assertAlmostEqual(v_bias, v_spis, places=3)
            np.testing.assert_allclose(geom.V_bg[mask], v_bias, rtol=1e-6)

    @unittest.expectedFailure
    def test_synthetic_background_without_input_files(self):
        """
        Without background and weighting files the loader synthesises a spherical spacecraft (R = 1 m)
        at potential V_sc. Its background potential must be continuous at the surface and decay
        outside as the Debye-shielded sphere V_sc (R / r) exp(-(r - R) / lambda_D).

        KNOWN FAILURE: the conductor mask is assigned Vw = 1 before the superposition, so the branch
        with the shielded profile is never reached; V_bg is V_sc inside the mask and exactly 0 outside.
        The resulting E = -grad V_bg is a single-cell spike at the surface. (The unreachable branch
        also uses V_sc exp(-r / lambda_D), which is not equal to V_sc on the surface.)
        """
        V_sc = -10.0
        with contextlib.redirect_stdout(io.StringIO()):
            params = SimulationParams3D(
                Vf=V_sc, spacecraft_voltage_V=V_sc,
                grid_nodes_x=21, grid_nodes_y=21, grid_nodes_z=21,
                vtk_files=VTKFilesConfig(
                    spis_background_potential_file="", spacecraft_weighting_file="", antenna_weighting_files=[]
                ),
            )
            geom = load_and_interpolate_vtk(params)
        X, Y, Z = np.meshgrid(params.x_grid, params.y_grid, params.z_grid, indexing='ij')
        r = np.sqrt(X**2 + Y**2 + Z**2)
        outside = (r >= 1.5) & (r <= 4.0)
        V_ref = V_sc / r[outside] * np.exp(-(r[outside] - 1.0) / params.debye_length)
        np.testing.assert_allclose(geom.V_bg[outside], V_ref, rtol=0.1)


if __name__ == "__main__":
    unittest.main()
