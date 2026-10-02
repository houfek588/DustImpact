# -*- coding: utf-8 -*-
"""
Field Solver module for 3D PIC Simulation.
Solves Poisson's equation on regular 3D grids using PyAMG or SciPy LU,
manages static background and antenna bias electrostatic fields, and computes accelerations.
"""

from typing import List, Tuple, Optional
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from dust_impact.physics.constants import eps_0
from dust_impact.numerics.interpolators import interp_field_3d, deposit_charge_cic_3d
from dust_impact.numerics.poisson import build_pyamg_solver, HAS_PYAMG
from dust_impact.solver.config_loader import SimulationParams3D, SimulationToggles3D


class FieldSolver3D:
    """
    Manages electrostatic fields and solves the self-consistent Poisson equation.
    """

    def __init__(
        self,
        params: SimulationParams3D,
        toggles: SimulationToggles3D,
        V_bg: np.ndarray,
        Ex_bg: np.ndarray,
        Ey_bg: np.ndarray,
        Ez_bg: np.ndarray,
        Ewx_list: List[np.ndarray],
        Ewy_list: List[np.ndarray],
        Ewz_list: List[np.ndarray],
        combined_mask: np.ndarray,
        num_antennas: int,
    ):
        self.p = params
        self.toggles = toggles
        self.combined_mask = combined_mask
        self.num_antennas = num_antennas

        self.V_bg_base = V_bg
        self.Ex_bg_base = Ex_bg
        self.Ey_bg_base = Ey_bg
        self.Ez_bg_base = Ez_bg

        self.Ex_bg = Ex_bg.copy()
        self.Ey_bg = Ey_bg.copy()
        self.Ez_bg = Ez_bg.copy()

        self.Ewx_list = Ewx_list
        self.Ewy_list = Ewy_list
        self.Ewz_list = Ewz_list

        Nx, Ny, Nz = self.p.Nx, self.p.Ny, self.p.Nz
        self.V_self_grid = np.zeros((Nx, Ny, Nz), dtype=np.float64)
        self.rho_grid = np.zeros((Nx, Ny, Nz), dtype=np.float64)
        self.Ex_self = np.zeros((Nx, Ny, Nz), dtype=np.float64)
        self.Ey_self = np.zeros((Nx, Ny, Nz), dtype=np.float64)
        self.Ez_self = np.zeros((Nx, Ny, Nz), dtype=np.float64)

        self.amg_solver = None
        self.lu_solver = None
        self._build_poisson_solver()

    def _get_1d_idx(self, i: int, j: int, k: int) -> int:
        return i * (self.p.Ny * self.p.Nz) + j * self.p.Nz + k

    def _build_poisson_solver(self) -> None:
        Nx, Ny, Nz = self.p.Nx, self.p.Ny, self.p.Nz
        N_tot = Nx * Ny * Nz

        dx2, dy2, dz2 = self.p.dx ** 2, self.p.dy ** 2, self.p.dz ** 2
        A_self = sp.lil_matrix((N_tot, N_tot))

        print(f"=== Inicializace Poissonova řešiče (3D) ===")
        print(f"  -> Celkový počet uzlů: {N_tot}")

        for i in range(Nx):
            for j in range(Ny):
                for k in range(Nz):
                    idx = self._get_1d_idx(i, j, k)
                    is_boundary = (i == 0) or (i == Nx - 1) or (j == 0) or (j == Ny - 1) or (k == 0) or (k == Nz - 1)

                    if is_boundary or self.combined_mask[i, j, k]:
                        A_self[idx, idx] = 1.0
                    else:
                        diag_val = 2 / dx2 + 2 / dy2 + 2 / dz2
                        A_self[idx, idx] = diag_val
                        A_self[idx, self._get_1d_idx(i - 1, j, k)] = -1 / dx2
                        A_self[idx, self._get_1d_idx(i + 1, j, k)] = -1 / dx2
                        A_self[idx, self._get_1d_idx(i, j - 1, k)] = -1 / dy2
                        A_self[idx, self._get_1d_idx(i, j + 1, k)] = -1 / dy2
                        A_self[idx, self._get_1d_idx(i, j, k - 1)] = -1 / dz2
                        A_self[idx, self._get_1d_idx(i, j, k + 1)] = -1 / dz2

        A_csr = A_self.tocsr()
        if HAS_PYAMG:
            print("  -> Stavím AMG hierarchii (PyAMG)...")
            self.amg_solver = build_pyamg_solver(A_csr)
        else:
            print("  -> PyAMG nenalezen, používám LU solver (SciPy)...")
            self.lu_solver = spla.factorized(A_self.tocsc())

    def update_background_fields(self, voltage_ant: np.ndarray, step: int) -> None:
        """Superpose antenna dynamic bias into Cartesian background electric field."""
        Ex_tot = self.Ex_bg_base.copy()
        Ey_tot = self.Ey_bg_base.copy()
        Ez_tot = self.Ez_bg_base.copy()

        for a_idx in range(self.num_antennas):
            V_curr = voltage_ant[a_idx, max(0, step - 1)] if step > 0 else voltage_ant[a_idx, 0]
            V_bias = self.p.V_bias[a_idx] if a_idx < len(self.p.V_bias) else 0.0
            dV = V_curr - V_bias

            if abs(dV) > 1e-6:
                Ex_tot += dV * self.Ewx_list[a_idx]
                Ey_tot += dV * self.Ewy_list[a_idx]
                Ez_tot += dV * self.Ewz_list[a_idx]

        if not getattr(self.toggles, 'enable_spis_background_field', True):
            Ex_tot.fill(0.0)
            Ey_tot.fill(0.0)
            Ez_tot.fill(0.0)

        self.Ex_bg, self.Ey_bg, self.Ez_bg = Ex_tot, Ey_tot, Ez_tot

    def solve_poisson(self, ensemble) -> None:
        """
        Deposits charge using Cloud-in-Cell (CIC) trilinear weighting and solves
        the Poisson equation -Laplacian(V) = rho / eps_0.
        """
        if not ensemble.cloud_injected or not getattr(self.toggles, 'enable_plasma_self_field', True):
            return

        Nx, Ny, Nz = self.p.Nx, self.p.Ny, self.p.Nz
        x0, y0, z0 = -self.p.L_x, -self.p.L_y, -self.p.L_z
        dx, dy, dz = self.p.dx, self.p.dy, self.p.dz

        q_e_grid = deposit_charge_cic_3d(
            ensemble.x_e[ensemble.active_e], ensemble.y_e[ensemble.active_e], ensemble.z_e[ensemble.active_e],
            self.p.q_macro, x0, y0, z0, dx, dy, dz, Nx, Ny, Nz
        )
        q_i_grid = deposit_charge_cic_3d(
            ensemble.x_i[ensemble.active_i], ensemble.y_i[ensemble.active_i], ensemble.z_i[ensemble.active_i],
            self.p.q_macro, x0, y0, z0, dx, dy, dz, Nx, Ny, Nz
        )

        dV = dx * dy * dz
        rho_grid = (q_i_grid - q_e_grid) / dV
        self.rho_grid = rho_grid

        # -Laplacian(V) = rho / eps_0 -> A * V = b where A is SPD
        b_3d = rho_grid / eps_0
        b_3d[self.combined_mask] = 0.0
        b_3d[0, :, :] = 0.0
        b_3d[-1, :, :] = 0.0
        b_3d[:, 0, :] = 0.0
        b_3d[:, -1, :] = 0.0
        b_3d[:, :, 0] = 0.0
        b_3d[:, :, -1] = 0.0
        b_self = np.nan_to_num(b_3d.ravel(), nan=0.0, posinf=0.0, neginf=0.0)

        if HAS_PYAMG and self.amg_solver is not None:
            x_init = np.nan_to_num(self.V_self_grid.ravel(), nan=0.0, posinf=0.0, neginf=0.0)
            try:
                V_self_1d = self.amg_solver.solve(b_self, x0=x_init, tol=1e-4, maxiter=20, accel='cg')
            except Exception:
                V_self_1d = self.amg_solver.solve(b_self, tol=1e-3, maxiter=20)
            V_self_1d = np.nan_to_num(V_self_1d, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            V_self_1d = self.lu_solver(b_self)

        self.V_self_grid = V_self_1d.reshape((Nx, Ny, Nz))
        self.Ex_self, self.Ey_self, self.Ez_self = np.gradient(-self.V_self_grid, self.p.dx, self.p.dy, self.p.dz)

    def interp_field(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, field_3d: np.ndarray) -> np.ndarray:
        """Interpolates 3D Cartesian field onto arbitrary particle coordinates."""
        return interp_field_3d(
            x, y, z, self.p.x_grid, self.p.y_grid, self.p.z_grid,
            self.p.dx, self.p.dy, self.p.dz, field_3d
        )

    def get_accel(self, x_act: np.ndarray, y_act: np.ndarray, z_act: np.ndarray,
                  mass: float, phys_charge: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculates total acceleration from background and self-consistent electric fields."""
        Ex_tot = self.interp_field(x_act, y_act, z_act, self.Ex_bg)
        Ey_tot = self.interp_field(x_act, y_act, z_act, self.Ey_bg)
        Ez_tot = self.interp_field(x_act, y_act, z_act, self.Ez_bg)

        if getattr(self.toggles, 'enable_plasma_self_field', True):
            Ex_tot += self.interp_field(x_act, y_act, z_act, self.Ex_self)
            Ey_tot += self.interp_field(x_act, y_act, z_act, self.Ey_self)
            Ez_tot += self.interp_field(x_act, y_act, z_act, self.Ez_self)

        factor = phys_charge / mass
        return factor * Ex_tot, factor * Ey_tot, factor * Ez_tot
