# -*- coding: utf-8 -*-
"""
Numerical sparse matrix assemblers and solvers for Laplace and Poisson equations.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from typing import Tuple, Any, List, Optional

try:
    import pyamg
    HAS_PYAMG = True
except ImportError:
    HAS_PYAMG = False


def solve_poisson_lu(A_sparse: sp.csc_matrix, b_vector: np.ndarray) -> np.ndarray:
    """
    Solves Poisson equation A * x = b via LU decomposition / factorization.
    """
    solver = spla.factorized(A_sparse)
    return solver(b_vector)


def build_pyamg_solver(A_sparse: sp.csr_matrix) -> Any:
    """
    Builds Algebraic Multigrid (PyAMG) solver hierarchy for 3D Poisson equation.
    Falls back to SciPy LU factorization if PyAMG is not installed.
    """
    if HAS_PYAMG:
        return pyamg.ruge_stuben_solver(A_sparse)
    else:
        return spla.factorized(A_sparse.tocsc())


def assemble_negative_laplacian_3d(Nx: int, Ny: int, Nz: int, dx: float, dy: float, dz: float) -> sp.csr_matrix:
    """
    Assembles the 7-point negative Laplacian -Lap on the full Nx x Ny x Nz node grid.

    The matrix acts on fields flattened in row-major (C) order, index = i * Ny * Nz + j * Nz + k,
    so x is the outermost Kronecker factor. Rows of boundary nodes are incomplete stencils; callers
    are expected to treat those nodes as Dirichlet nodes and eliminate them.

    Args:
        Nx, Ny, Nz: Number of grid nodes along x, y, z.
        dx, dy, dz: Grid spacing along x, y, z [m].

    Returns:
        Symmetric sparse matrix of shape (Nx*Ny*Nz, Nx*Ny*Nz) [1/m^2].
    """
    def second_difference(n: int, h: float) -> sp.csr_matrix:
        ones = np.ones(n)
        return sp.diags([-ones[:-1], 2.0 * ones, -ones[:-1]], [-1, 0, 1], format='csr') / h ** 2

    return sp.kronsum(
        sp.kronsum(second_difference(Nz, dz), second_difference(Ny, dy)),
        second_difference(Nx, dx),
        format='csr',
    )


class DirichletLaplaceSolver3D:
    """
    Solves the discrete Laplace equation Lap V = 0 on the free nodes with V prescribed on fixed_mask.

    The Dirichlet nodes are eliminated (A_ff V_f = -A_fd V_d), which keeps the reduced matrix symmetric
    positive definite, so an AMG-preconditioned conjugate gradient solver is used and its convergence
    is checked. The AMG hierarchy is built once and reused by every call of solve(). The outer
    boundary of the grid must be part of fixed_mask, otherwise the boundary rows would be incomplete
    stencils.

    Optional coupling nodes (thin-wire model, see dust_impact.geometry.thin_wire) are free nodes with
    an extra diagonal term beta >= 0 [1/m^2] and the source beta * target:
        (-Lap V)_n + beta_n V_n = beta_n * target_n.
    They represent a conductor thinner than the grid held at potential `target`; the matrix stays
    symmetric positive definite.
    """

    def __init__(self, fixed_mask: np.ndarray, spacing: Tuple[float, float, float],
                 rtol: float = 1e-8, maxiter: int = 500, coupling: Optional[np.ndarray] = None):
        """
        Args:
            fixed_mask: Boolean array (Nx, Ny, Nz) of Dirichlet nodes (must contain the outer faces).
            spacing: Grid spacing (dx, dy, dz) [m].
            rtol: Required relative residual ||b - A_ff V_f|| / ||b|| of every solve.
            maxiter: Iteration cap of the conjugate gradient solver.
            coupling: Optional array (Nx, Ny, Nz) of thin-wire coupling coefficients beta >= 0 [1/m^2],
                nonzero only on free nodes.

        Raises:
            ValueError: If the outer faces of the grid are not all fixed, or coupling nodes are fixed.
        """
        self.fixed_mask = np.asarray(fixed_mask, dtype=bool)
        self.shape = self.fixed_mask.shape
        faces = np.zeros_like(self.fixed_mask)
        faces[[0, -1], :, :] = True
        faces[:, [0, -1], :] = True
        faces[:, :, [0, -1]] = True
        if not np.all(self.fixed_mask[faces]):
            raise ValueError("The outer faces of the grid must be Dirichlet nodes.")
        self.rtol, self.maxiter = rtol, maxiter

        A = assemble_negative_laplacian_3d(*self.shape, *spacing)
        fixed_flat = self.fixed_mask.ravel()
        self.free_idx = np.flatnonzero(~fixed_flat)
        self.fixed_idx = np.flatnonzero(fixed_flat)
        A_free_rows = A[self.free_idx]
        self.A_ff = A_free_rows[:, self.free_idx].tocsr()
        self.A_fd = A_free_rows[:, self.fixed_idx].tocsr()

        self.beta_f = None
        if coupling is not None:
            beta = np.asarray(coupling, dtype=np.float64).ravel()
            if np.any(beta < 0.0) or np.any(beta[self.fixed_idx] != 0.0):
                raise ValueError("Thin-wire coupling must be non-negative and zero on Dirichlet nodes.")
            if np.any(beta > 0.0):
                self.beta_f = beta[self.free_idx]
                self.A_ff = (self.A_ff + sp.diags(self.beta_f)).tocsr()

        self._amg = pyamg.ruge_stuben_solver(self.A_ff) if HAS_PYAMG and len(self.free_idx) else None
        self._lu = spla.factorized(self.A_ff.tocsc()) if not HAS_PYAMG and len(self.free_idx) else None

    def solve(self, fixed_values: np.ndarray) -> np.ndarray:
        """
        Args:
            fixed_values: Field of shape (Nx, Ny, Nz); its values on fixed_mask are the Dirichlet values
                and its values on coupling nodes are the thin-wire potentials (targets).

        Returns:
            Field equal to fixed_values on fixed_mask and discrete-harmonic on the other free nodes.

        Raises:
            RuntimeError: If the solve does not reach the required relative residual.
        """
        V = np.array(fixed_values, dtype=np.float64).ravel()
        b = -(self.A_fd @ V[self.fixed_idx])
        if self.beta_f is not None:
            b = b + self.beta_f * V[self.free_idx]
        b_norm = np.linalg.norm(b)
        if len(self.free_idx) == 0:
            pass
        elif b_norm == 0.0:
            V[self.free_idx] = 0.0
        elif self._amg is not None:
            V_f = self._amg.solve(b, tol=self.rtol, maxiter=self.maxiter, accel='cg')
            rel_res = float(np.linalg.norm(b - self.A_ff @ V_f) / b_norm)
            if not np.isfinite(rel_res) or rel_res > 10.0 * self.rtol:
                raise RuntimeError(
                    f"Laplace solver did not converge: relative residual {rel_res:.3e} (required {self.rtol:.1e})."
                )
            V[self.free_idx] = V_f
        else:
            V[self.free_idx] = self._lu(b)
        return V.reshape(self.shape)

    def enforce(self, field: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        Adds the harmonic correction that makes field equal to target on the Dirichlet nodes.

        The correction c solves Lap c = 0 on the free nodes with c = target - field on fixed_mask, so
        the result keeps the discrete Laplacian (i.e. the space charge) of field on the free nodes and
        takes the prescribed values on the fixed nodes.

        Args:
            field: Field of shape (Nx, Ny, Nz); NaN entries are treated as 0.
            target: Field of shape (Nx, Ny, Nz); only its values on fixed_mask are used.

        Returns:
            Corrected field (float64).
        """
        field = np.nan_to_num(np.asarray(field, dtype=np.float64))
        return field + self.solve(np.where(self.fixed_mask, target - field, 0.0))
