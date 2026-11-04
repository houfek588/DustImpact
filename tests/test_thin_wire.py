# -*- coding: utf-8 -*-
"""
Sub-grid thin-wire model (dust_impact.geometry.thin_wire) against analytical results:
  - the effective radius of a single line of nodes along a grid axis equals the lattice value
    h exp(-gamma) / (2 sqrt 2) = 0.1985 h of the 5-point Laplacian,
  - the capacitance of a thin straight wire matches the thin-cylinder formula
    C = 2 pi eps0 L / (ln(L/a) - (1 - ln 2)) corrected for the grounded box, for wires along a grid axis
    and along diagonals, and does not depend on the grid spacing (the voxel wire held at the wire
    potential overestimates it by 30-100 %),
  - Green reciprocity: the charge induced on the thin wire by a point charge in the self-field Poisson
    problem equals -q Vw(r) of the thin-wire weighting potential (Ramo-Shockley consistency).
"""

import contextlib
import io
import unittest

import numpy as np

from dust_impact.physics.constants import eps_0
from dust_impact.numerics.poisson import DirichletLaplaceSolver3D
from dust_impact.physics.ramo_shockley import vacuum_capacitance_matrix
from dust_impact.geometry.voxelizer import voxelize_cylinder
from dust_impact.geometry.analytical import solve_laplace_dirichlet_3d
from dust_impact.geometry.thin_wire import (
    calibrate_effective_radius, thin_wire_coupling, thin_wire_voxel_radius
)
from dust_impact.solver.config_loader import SimulationParams3D, SimulationToggles3D
from dust_impact.solver.field_solver import FieldSolver3D

L_BOX = 1.5        # half-width of the grounded box [m]
WIRE_LENGTH = 0.5  # [m]
WIRE_RADIUS = 1e-3  # [m]

WIRES = {
    "axis": ((-0.25, 0.0, 0.0), (0.25, 0.0, 0.0)),
    "xy-diagonal": ((-0.1768, -0.1768, 0.013), (0.1768, 0.1768, 0.013)),
    "space-diagonal": ((-0.1443, -0.1443, -0.1443), (0.1443, 0.1443, 0.1443)),
}


def _grid(N: int):
    g = np.linspace(-L_BOX, L_BOX, N)
    h = g[1] - g[0]
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    return g, h, X, Y, Z


def _thin_wire(N: int, p1, p2):
    """Wire nodes, coupling and the thin-wire weighting potential on an N^3 grid."""
    g, h, X, Y, Z = _grid(N)
    spacing = (h, h, h)
    vox_r = thin_wire_voxel_radius(WIRE_RADIUS, spacing)
    mask = voxelize_cylinder(X, Y, Z, p1, p2, vox_r)
    r_eff = calibrate_effective_radius(p1, p2, vox_r, (g, g, g))
    beta = thin_wire_coupling(mask, WIRE_LENGTH, WIRE_RADIUS, r_eff, spacing)
    Vw = solve_laplace_dirichlet_3d(N, N, N, h, h, h, np.zeros_like(mask), [mask.astype(float)], coupling=beta)[0]
    return g, h, mask, beta, r_eff, Vw


def _box_corrected_capacitance() -> tuple:
    """Thin-cylinder capacitance in free space and its bounds in the grounded box (outer radius L .. sqrt3 L)."""
    c_free = 2.0 * np.pi * eps_0 * WIRE_LENGTH / (np.log(WIRE_LENGTH / WIRE_RADIUS) - (1.0 - np.log(2.0)))
    lo = c_free / (1.0 - c_free / (4.0 * np.pi * eps_0 * np.sqrt(3.0) * L_BOX))
    hi = c_free / (1.0 - c_free / (4.0 * np.pi * eps_0 * L_BOX))
    return lo, hi


class TestThinWireModel(unittest.TestCase):

    def test_effective_radius_of_axis_aligned_node_line(self):
        """A line of nodes along a grid axis acts as a cylinder of radius 0.1985 h (lattice Green function)."""
        g, h, _, _, _ = _grid(61)
        r_eff = calibrate_effective_radius(*WIRES["axis"], 0.75 * h, (g, g, g))
        self.assertAlmostEqual(r_eff / h / (np.exp(-np.euler_gamma) / (2.0 * np.sqrt(2.0))), 1.0, delta=0.08)

    def test_wire_capacitance_matches_thin_cylinder_formula(self):
        """C of a 0.5 m wire of radius 1 mm lies within 8 % of the box-corrected analytical value for any
        orientation and two grid spacings (5 cm and 3.3 cm), while the voxel wire is off by > 25 %."""
        lo, hi = _box_corrected_capacitance()
        for N in (61, 91):
            for name, (p1, p2) in WIRES.items():
                with self.subTest(grid=N, wire=name):
                    g, h, mask, beta, r_eff, Vw = _thin_wire(N, p1, p2)
                    C = vacuum_capacitance_matrix([Vw], [mask], (h, h, h))[0, 0]
                    self.assertGreater(C, 0.92 * lo)
                    self.assertLess(C, 1.08 * hi)
                    fixed = mask.copy()
                    fixed[[0, -1]] = fixed[:, [0, -1]] = fixed[:, :, [0, -1]] = True
                    V_vox = DirichletLaplaceSolver3D(fixed, (h, h, h)).solve(mask.astype(float))
                    C_vox = vacuum_capacitance_matrix([V_vox], [mask], (h, h, h))[0, 0]
                    self.assertGreater(C_vox, 1.25 * hi)

    def test_green_reciprocity_with_self_field(self):
        """The charge induced on the grounded thin wire by one ion, computed from the self-field Poisson
        problem (sum of the thin-wire node charges eps0 dV beta (0 - V)), equals -q Vw(r)."""
        N = 41
        g, h, mask, beta, _, Vw = _thin_wire(N, *WIRES["xy-diagonal"])
        params = SimulationParams3D(
            grid_nodes_x=N, grid_nodes_y=N, grid_nodes_z=N,
            domain_half_length_x_m=L_BOX, domain_half_length_y_m=L_BOX, domain_half_length_z_m=L_BOX,
            num_macroparticles=1, total_impact_charge_C=1e-15,
        )
        toggles = SimulationToggles3D(enable_spis_background_field=False, enable_plasma_self_field=True)
        zeros = np.zeros((N, N, N))
        with contextlib.redirect_stdout(io.StringIO()):
            fs = FieldSolver3D(params, toggles, zeros, zeros, zeros, zeros, [], [], [], mask, 0,
                               wire_coupling=beta)
        empty = np.zeros(0)
        for position in ([0.3, -0.2, 0.05], [0.05, 0.12, -0.21]):
            ens = type("Ensemble", (), dict(
                cloud_injected=True,
                x_e=empty, y_e=empty, z_e=empty, active_e=np.zeros(0, dtype=bool),
                x_i=np.array([position[0]]), y_i=np.array([position[1]]), z_i=np.array([position[2]]),
                active_i=np.array([True]),
            ))()
            fs.solve_poisson(ens)
            Q_wire = -eps_0 * h**3 * float(np.sum(beta * fs.V_self_grid))
            vw = fs.interp_field(np.array([position[0]]), np.array([position[1]]), np.array([position[2]]), Vw)[0]
            with self.subTest(position=position):
                self.assertAlmostEqual(Q_wire / (-params.q_macro * vw), 1.0, delta=1e-5)


if __name__ == "__main__":
    unittest.main()
