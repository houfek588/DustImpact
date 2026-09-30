# -*- coding: utf-8 -*-
"""
Validation unit test comparing 3D Poisson/Laplace solver against Variant B
weighting field interpolation for a conducting sphere at (0, 0, 0).
"""

import unittest
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dust_impact.sim3d.config_loader import SimulationParams3D, SimulationToggles3D, VTKFilesConfig
from dust_impact.sim3d.vtk_reader import load_and_interpolate_vtk


class TestSphereElectrostaticValidation(unittest.TestCase):
    """
    Validates electrostatic potential calculation methods for a conducting sphere:
    1. Direct 3D Poisson / Laplace finite difference solver
    2. Variant B (scaling from inputs/sph_potencial.vtk)
    3. Analytical theoretical potential V(r) = V_0 * (R_sph / r)
    """

    def test_sphere_poisson_vs_variant_b(self):
        V_sc = -10
        R_sph = 1.0
        L_box = 10.0
        N = 35

        params = SimulationParams3D(
            Vf=V_sc,
            spacecraft_voltage_V=V_sc,
            domain_half_length_x_m=L_box,
            domain_half_length_y_m=L_box,
            domain_half_length_z_m=L_box,
            grid_nodes_x=N,
            grid_nodes_y=N,
            grid_nodes_z=N,
            vtk_files=VTKFilesConfig(
                spis_background_potential_file="",
                spacecraft_weighting_file="inputs/sph_potencial.vtk",
                antenna_weighting_files=[]
            )
        )

        # 1. Variant B: Interpolating & scaling from VTK weighting field
        V_bg_varB, _, Ex_varB, Ey_varB, Ez_varB, _, _, _, _, sc_mask = load_and_interpolate_vtk(params)

        # 2. Numerical 3D Finite Difference Laplace / Poisson Solver on the grid
        x_grid = params.x_grid
        y_grid = params.y_grid
        z_grid = params.z_grid
        dx, dy, dz = params.dx, params.dy, params.dz
        dx2, dy2, dz2 = dx**2, dy**2, dz**2

        X, Y, Z = np.meshgrid(x_grid, y_grid, z_grid, indexing='ij')
        R = np.sqrt(X**2 + Y**2 + Z**2)

        sphere_mask_geom = R <= R_sph

        N_tot = N * N * N
        A = sp.lil_matrix((N_tot, N_tot))
        b = np.zeros(N_tot)

        def get_idx(i, j, k):
            return i * (N * N) + j * N + k

        for i in range(N):
            for j in range(N):
                for k in range(N):
                    idx = get_idx(i, j, k)
                    is_boundary = (i == 0) or (i == N - 1) or (j == 0) or (j == N - 1) or (k == 0) or (k == N - 1)

                    if sphere_mask_geom[i, j, k]:
                        A[idx, idx] = 1.0
                        b[idx] = V_sc
                    elif is_boundary:
                        A[idx, idx] = 1.0
                        b[idx] = V_sc * (R_sph / R[i, j, k])
                    else:
                        diag = -2 / dx2 - 2 / dy2 - 2 / dz2
                        A[idx, idx] = diag
                        A[idx, get_idx(i - 1, j, k)] = 1 / dx2
                        A[idx, get_idx(i + 1, j, k)] = 1 / dx2
                        A[idx, get_idx(i, j - 1, k)] = 1 / dy2
                        A[idx, get_idx(i, j + 1, k)] = 1 / dy2
                        A[idx, get_idx(i, j, k - 1)] = 1 / dz2
                        A[idx, get_idx(i, j, k + 1)] = 1 / dz2

        solver = spla.factorized(A.tocsc())
        V_poisson_1d = solver(b)
        V_poisson = V_poisson_1d.reshape((N, N, N))

        # 3. Analytical theoretical potential
        R_safe = np.maximum(R, 1e-12)
        V_theory = np.where(R <= R_sph, V_sc, V_sc * (R_sph / R_safe))

        # Evaluate comparison in the physical plasma region outside the sphere
        outer_mask = (R >= 1.2 * R_sph) & (R <= 0.8 * L_box)

        v_poisson_pts = V_poisson[outer_mask]
        v_varB_pts = V_bg_varB[outer_mask]
        v_theory_pts = V_theory[outer_mask]

        # Correlation between numerical Poisson solution and Variant B
        correlation = float(np.corrcoef(v_poisson_pts, v_varB_pts)[0, 1])
        mean_rel_error_varB_poisson = float(np.mean(np.abs(v_poisson_pts - v_varB_pts)) / np.abs(V_sc))
        mean_rel_error_poisson_theory = float(np.mean(np.abs(v_poisson_pts - v_theory_pts)) / np.abs(V_sc))

        # Verification assertions
        self.assertGreater(correlation, 0.99, f"Correlation too low: {correlation:.5f}")
        self.assertLess(mean_rel_error_varB_poisson, 0.10, f"Relative error vs Poisson too high: {mean_rel_error_varB_poisson * 100:.2f}%")
        self.assertLess(mean_rel_error_poisson_theory, 0.05, f"Relative error vs Theory too high: {mean_rel_error_poisson_theory * 100:.2f}%")

        # Save validation plot into outputs/tests/
        os.makedirs("outputs/tests", exist_ok=True)
        mid_idx = N // 2
        r_grid_axis = x_grid[mid_idx:]
        v_poisson_axis = V_poisson[mid_idx:, mid_idx, mid_idx]
        v_varB_axis = V_bg_varB[mid_idx:, mid_idx, mid_idx]
        v_theory_axis = V_theory[mid_idx:, mid_idx, mid_idx]

        plt.figure(figsize=(8, 5))
        plt.plot(r_grid_axis, v_theory_axis, 'k--', label='Analytical $V(r) = V_0 R/r$', linewidth=2)
        plt.plot(r_grid_axis, v_poisson_axis, 'b-o', label='3D Poisson Solver (FDM)', markersize=5)
        plt.plot(r_grid_axis, v_varB_axis, 'r-s', label='Variant B (SPIS VTK $V_{sc} \\cdot V_w$)', markersize=4)
        plt.xlabel('Distance from sphere center $r$ [m]')
        plt.ylabel('Electric Potential $V$ [V]')
        plt.title('Electrostatic Validation: 3D Poisson vs Variant B (Sphere at Origin)')
        plt.grid(True, linestyle=':')
        plt.legend()
        plt.tight_layout()
        plt.savefig("outputs/tests/test_sphere_electrostatic_validation.png", dpi=150)
        plt.close()


if __name__ == "__main__":
    unittest.main()
