# -*- coding: utf-8 -*-
"""
Numerical sparse matrix assemblers and solvers for Laplace and Poisson equations.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from typing import Tuple, Any

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
