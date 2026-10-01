# -*- coding: utf-8 -*-
"""
Analytical geometry definitions and exact electrostatic solutions for synthetic benchmarks.
Provides exact potentials V and electric fields E for spheres, boxes, and antennas.
"""

from typing import Tuple, Sequence, Optional, List, Any
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from dust_impact.geometry.voxelizer import voxelize_sphere, voxelize_box, voxelize_cylinder
from dust_impact.numerics.poisson import build_pyamg_solver, HAS_PYAMG


class AnalyticalConductor:
    """ Base interface for analytical conductors. """

    def get_mask(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def get_potential(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def get_electric_field(
        self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError


class AnalyticalSphere(AnalyticalConductor):
    """
    Spherical conducting body with exact electrostatic potential and electric field.
    Supports both vacuum (Coulomb/Laplace) and plasma (Debye/Yukawa shielded) solutions.
    """

    def __init__(
        self,
        center: Sequence[float] = (0.0, 0.0, 0.0),
        radius: float = 1.0,
        voltage: float = 1.0,
        debye_length: Optional[float] = None
    ):
        self.center = np.array(center, dtype=float)
        self.radius = float(radius)
        self.voltage = float(voltage)
        self.debye_length = float(debye_length) if (debye_length is not None and debye_length > 0) else None

    def get_mask(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> np.ndarray:
        return voxelize_sphere(X, Y, Z, center=self.center, radius=self.radius)

    def get_potential(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> np.ndarray:
        rx = X - self.center[0]
        ry = Y - self.center[1]
        rz = Z - self.center[2]
        r = np.sqrt(rx**2 + ry**2 + rz**2)
        r_safe = np.maximum(r, 1e-12)

        mask = r <= self.radius
        V = np.empty_like(r)

        if self.debye_length is None:
            # Vacuum Coulomb / Laplace 1/r potential
            V = self.voltage * (self.radius / r_safe)
        else:
            # Debye-shielded Yukawa potential V(r) = V0 * (R / r) * exp(-(r - R) / lambda_D)
            decay = np.exp(-(r_safe - self.radius) / self.debye_length)
            V = self.voltage * (self.radius / r_safe) * decay

        # Equipotential conductor interior
        V[mask] = self.voltage
        return V

    def get_electric_field(
        self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        rx = X - self.center[0]
        ry = Y - self.center[1]
        rz = Z - self.center[2]
        r = np.sqrt(rx**2 + ry**2 + rz**2)
        r_safe = np.maximum(r, 1e-12)

        mask = r <= self.radius

        if self.debye_length is None:
            # E_r = V0 * R / r^2
            Er = self.voltage * self.radius / (r_safe ** 2)
        else:
            # E_r = -dV/dr = V(r) * (1/r + 1/lambda_D)
            V_out = self.voltage * (self.radius / r_safe) * np.exp(-(r_safe - self.radius) / self.debye_length)
            Er = V_out * (1.0 / r_safe + 1.0 / self.debye_length)

        # Vector components E = Er * (r_vec / r)
        Ex = Er * (rx / r_safe)
        Ey = Er * (ry / r_safe)
        Ez = Er * (rz / r_safe)

        # Inside conductor E = 0
        Ex[mask] = 0.0
        Ey[mask] = 0.0
        Ez[mask] = 0.0

        return Ex, Ey, Ez


class AnalyticalBox(AnalyticalConductor):
    """
    Axis-aligned rectangular box conductor.
    """

    def __init__(
        self,
        min_bounds: Sequence[float],
        max_bounds: Sequence[float],
        voltage: float = 1.0
    ):
        self.min_bounds = np.array(min_bounds, dtype=float)
        self.max_bounds = np.array(max_bounds, dtype=float)
        self.voltage = float(voltage)

    def get_mask(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> np.ndarray:
        return voxelize_box(X, Y, Z, self.min_bounds, self.max_bounds)

    def get_potential(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> np.ndarray:
        mask = self.get_mask(X, Y, Z)
        center = 0.5 * (self.min_bounds + self.max_bounds)
        diag = np.linalg.norm(self.max_bounds - self.min_bounds)
        r_eff = 0.5 * diag

        rx = X - center[0]
        ry = Y - center[1]
        rz = Z - center[2]
        r = np.sqrt(rx**2 + ry**2 + rz**2)
        r_safe = np.maximum(r, 1e-12)

        V = self.voltage * (r_eff / np.maximum(r_safe, r_eff))
        V[mask] = self.voltage
        return V

    def get_electric_field(
        self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        V = self.get_potential(X, Y, Z)
        # Using numerical finite difference gradient for complex box geometry
        mask = self.get_mask(X, Y, Z)
        # Assuming uniform grid steps if calculated outside, or approximate
        Ex, Ey, Ez = np.gradient(-V)
        Ex[mask] = 0.0
        Ey[mask] = 0.0
        Ez[mask] = 0.0
        return Ex, Ey, Ez


def generate_synthetic_analytical_fields(params: Any):
    """
    Generates synthetic 3D analytical fields for debugging and testing when VTK files
    are missing or analytical benchmarking mode is active.

    Parameters
    ----------
    params : SimulationParams3D or object
        Parameters defining grid axes (x_grid, y_grid, z_grid), Vf, debye_length, etc.

    Returns
    -------
    Tuple:
        V_bg : np.ndarray
        Vw_grids : List[np.ndarray]
        Ex_bg, Ey_bg, Ez_bg : np.ndarray
        Ewx_list, Ewy_list, Ewz_list : List[np.ndarray]
        antenna_masks : List[np.ndarray]
        spacecraft_mask : np.ndarray
    """
    print("[FALLBACK] Generuji syntetická 3D pole pro ladění...")
    X, Y, Z = np.meshgrid(params.x_grid, params.y_grid, params.z_grid, indexing='ij')

    r_spacecraft = 1.0
    V_sc = getattr(params, 'spacecraft_voltage_V', params.Vf)
    if V_sc is None:
        V_sc = params.Vf

    sphere = AnalyticalSphere(
        center=(0.0, 0.0, 0.0),
        radius=r_spacecraft,
        voltage=V_sc,
        debye_length=getattr(params, 'debye_length', 1.0)
    )

    spacecraft_mask = sphere.get_mask(X, Y, Z)
    V_bg = sphere.get_potential(X, Y, Z)
    Ex_bg, Ey_bg, Ez_bg = sphere.get_electric_field(X, Y, Z)

    # 3 Orthogonal Antennas along X, Y, Z axes
    ant_positions = [[1.5, 0.0, 0.0], [0.0, 1.5, 0.0], [0.0, 0.0, 1.5]]
    Vw_grids, Ewx_list, Ewy_list, Ewz_list, antenna_masks = [], [], [], [], []

    for pos in ant_positions:
        r_sq = (X - pos[0])**2 + (Y - pos[1])**2 + (Z - pos[2])**2
        Vw = np.exp(-r_sq / 0.5**2)
        mask = r_sq <= 0.15**2
        Vw[mask] = 1.0

        Ewx, Ewy, Ewz = np.gradient(-Vw, params.dx, params.dy, params.dz)
        Ewx[mask], Ewy[mask], Ewz[mask] = 0.0, 0.0, 0.0

        Vw_grids.append(Vw)
        Ewx_list.append(Ewx)
        Ewy_list.append(Ewy)
        Ewz_list.append(Ewz)
        antenna_masks.append(mask)

    return V_bg, Vw_grids, Ex_bg, Ey_bg, Ez_bg, Ewx_list, Ewy_list, Ewz_list, antenna_masks, spacecraft_mask


def voxelize_analytical_spacecraft_part(
    X: np.ndarray, Y: np.ndarray, Z: np.ndarray, sc_part: Any
) -> np.ndarray:
    """
    Voxelizes an analytical spacecraft part (primitive or composite) onto coordinate meshgrids.
    """
    sc_type = getattr(sc_part, 'type', 'sphere').lower()

    if sc_type == 'composite':
        mask = np.zeros_like(X, dtype=bool)
        for p in getattr(sc_part, 'parts', []):
            mask |= voxelize_analytical_spacecraft_part(X, Y, Z, p)
        return mask

    if sc_type == 'sphere':
        center = getattr(sc_part, 'center', [0.0, 0.0, 0.0])
        radius = getattr(sc_part, 'radius', 1.0)
        return voxelize_sphere(X, Y, Z, center=center, radius=radius)

    if sc_type in ['box', 'cube']:
        center = np.array(getattr(sc_part, 'center', [0.0, 0.0, 0.0]), dtype=float)
        side_l = getattr(sc_part, 'side_length', None)
        if side_l is not None:
            dims = np.array([side_l, side_l, side_l], dtype=float)
        else:
            dims = np.array(getattr(sc_part, 'dimensions', [1.0, 1.0, 1.0]), dtype=float)
        half_dims = 0.5 * dims
        return voxelize_box(X, Y, Z, center - half_dims, center + half_dims)

    if sc_type == 'cylinder':
        p1 = getattr(sc_part, 'p_start', [0.0, 0.0, -1.0])
        p2 = getattr(sc_part, 'p_end', [0.0, 0.0, 1.0])
        radius = getattr(sc_part, 'radius', 0.5)
        return voxelize_cylinder(X, Y, Z, p1, p2, radius)

    return voxelize_sphere(X, Y, Z)


def solve_laplace_dirichlet_3d(
    Nx: int, Ny: int, Nz: int,
    dx: float, dy: float, dz: float,
    dirichlet_mask: np.ndarray,
    rhs_values: List[np.ndarray],
    debye_length: Optional[float] = None
) -> List[np.ndarray]:
    """
    Solves 3D Laplace/screened-Poisson equation on Cartesian grid for multiple RHS Dirichlet conditions
    sharing the exact same Dirichlet node pattern (conductors + outer boundaries).
    """
    N_tot = Nx * Ny * Nz
    dx2, dy2, dz2 = dx**2, dy**2, dz**2

    # Outer boundary nodes are also Dirichlet (V = 0)
    boundary_mask = np.zeros((Nx, Ny, Nz), dtype=bool)
    boundary_mask[0, :, :] = True
    boundary_mask[Nx - 1, :, :] = True
    boundary_mask[:, 0, :] = True
    boundary_mask[:, Ny - 1, :] = True
    boundary_mask[:, :, 0] = True
    boundary_mask[:, :, Nz - 1] = True

    fixed_mask = boundary_mask | dirichlet_mask
    fixed_indices = np.where(fixed_mask.ravel())[0]

    # 1D second differences
    ex = np.ones(Nx)
    Dxx = sp.diags([-ex / dx2, 2 * ex / dx2, -ex / dx2], [-1, 0, 1], shape=(Nx, Nx), dtype=float)
    ey = np.ones(Ny)
    Dyy = sp.diags([-ey / dy2, 2 * ey / dy2, -ey / dy2], [-1, 0, 1], shape=(Ny, Ny), dtype=float)
    ez = np.ones(Nz)
    Dzz = sp.diags([-ez / dz2, 2 * ez / dz2, -ez / dz2], [-1, 0, 1], shape=(Nz, Nz), dtype=float)

    # 3D Laplacian = Dxx (x) I (x) I + I (x) Dyy (x) I + I (x) I (x) Dzz
    A = sp.kronsum(sp.kronsum(Dxx, Dyy), Dzz, format='lil')

    if debye_length is not None and debye_length > 0:
        helmholtz_term = 1.0 / (debye_length ** 2)
        A.setdiag(A.diagonal() + helmholtz_term)

    # Enforce Dirichlet rows: row becomes unit vector [0 ... 1 ... 0]
    for idx in fixed_indices:
        A.rows[idx] = [idx]
        A.data[idx] = [1.0]

    A_csr = A.tocsr()

    # Build solver hierarchy once
    if HAS_PYAMG:
        solver = build_pyamg_solver(A_csr)
        solve_func = lambda b_vec: solver.solve(b_vec, tol=1e-6)
    else:
        solver = spla.factorized(A.tocsc())
        solve_func = lambda b_vec: solver(b_vec)

    solutions = []
    for rhs_grid in rhs_values:
        b = np.zeros(N_tot, dtype=float)
        b[fixed_indices] = rhs_grid.ravel()[fixed_indices]
        sol_1d = solve_func(b)
        solutions.append(sol_1d.reshape((Nx, Ny, Nz)))

    return solutions


def build_analytical_simulation_geometry(params: Any, analytical_config: Any):
    """
    Builds complete PreparedGeometry3D from analytical geometry specifications,
    solving Laplace equation for exact Ramo-Shockley weighting fields and equilibrium background.
    """
    from dust_impact.geometry.prepared import PreparedGeometry3D
    from dust_impact.geometry.surface import compute_impact_intersection_and_normal

    print("[GEOMETRY] Sestavuji analytickou geometrii a řeším Laplaceova pole...")
    Nx, Ny, Nz = params.Nx, params.Ny, params.Nz
    X, Y, Z = np.meshgrid(params.x_grid, params.y_grid, params.z_grid, indexing='ij')

    # 1. Spacecraft body mask
    sc_part = getattr(analytical_config, 'spacecraft', None)
    if sc_part is not None:
        spacecraft_mask = voxelize_analytical_spacecraft_part(X, Y, Z, sc_part)
    else:
        spacecraft_mask = voxelize_sphere(X, Y, Z, center=(0, 0, 0), radius=1.0)

    # 2. Antenna masks
    ant_geoms = getattr(analytical_config, 'antennas', [])
    antenna_masks = []
    for ant in ant_geoms:
        p_start = getattr(ant, 'p_start', [0.0, 0.0, 0.0])
        p_end = getattr(ant, 'p_end', [1.0, 0.0, 0.0])
        radius = getattr(ant, 'radius', 0.015)
        m = voxelize_cylinder(X, Y, Z, p_start, p_end, radius)
        antenna_masks.append(m)

    num_antennas = len(antenna_masks)

    # 3. Combined conductor mask for Dirichlet boundary conditions
    all_conductors_mask = spacecraft_mask.copy()
    for m in antenna_masks:
        all_conductors_mask |= m

    # 4. Prepare RHS grids for each antenna weighting field and background field
    rhs_list = []

    # Weighting field for each antenna: Vw = 1.0 on target antenna, 0.0 on spacecraft and other antennas
    for i in range(num_antennas):
        rhs_vw = np.zeros((Nx, Ny, Nz), dtype=float)
        rhs_vw[antenna_masks[i]] = 1.0
        rhs_vw[spacecraft_mask] = 0.0
        for j in range(num_antennas):
            if j != i:
                rhs_vw[antenna_masks[j]] = 0.0
        rhs_list.append(rhs_vw)

    # Background potential field: V_bg = V_sc on spacecraft, V_bias on antennas
    V_sc = getattr(params, 'spacecraft_voltage_V', params.Vf)
    if V_sc is None:
        V_sc = params.Vf

    rhs_bg = np.zeros((Nx, Ny, Nz), dtype=float)
    rhs_bg[spacecraft_mask] = float(V_sc)
    effective_biases = []
    for i in range(num_antennas):
        ant = ant_geoms[i]
        v_ant = getattr(ant, 'effective_voltage', None)
        if v_ant is None:
            v_ant = getattr(ant, 'voltage_V', getattr(ant, 'potential_V', getattr(ant, 'bias_voltage_V', None)))
        if v_ant is None:
            v_ant = 0.0
        v_ant = float(v_ant)
        effective_biases.append(v_ant)
        rhs_bg[antenna_masks[i]] = v_ant
    rhs_list.append(rhs_bg)

    # 5. Solve Laplace equation on grid
    debye_len = getattr(params, 'debye_length', None) if getattr(params, 'enable_plasma_self_field', True) else None
    solved_fields = solve_laplace_dirichlet_3d(
        Nx, Ny, Nz, params.dx, params.dy, params.dz,
        dirichlet_mask=all_conductors_mask,
        rhs_values=rhs_list,
        debye_length=None  # Laplace for vacuum/shielded boundary
    )

    Vw_grids = solved_fields[:num_antennas]
    V_bg = solved_fields[num_antennas]

    # Enforce exact equipotentials inside conductors
    for i in range(num_antennas):
        Vw_grids[i][antenna_masks[i]] = 1.0
        Vw_grids[i][spacecraft_mask] = 0.0

    V_bg[spacecraft_mask] = float(V_sc)
    for i in range(num_antennas):
        V_bg[antenna_masks[i]] = effective_biases[i]

    params.antenna_bias_voltage_V = effective_biases
    params.V_bias = effective_biases

    # 6. Compute electric fields (-grad V)
    Ewx_list, Ewy_list, Ewz_list = [], [], []
    for i in range(num_antennas):
        ewx, ewy, ewz = np.gradient(-Vw_grids[i], params.dx, params.dy, params.dz)
        ewx[all_conductors_mask] = 0.0
        ewy[all_conductors_mask] = 0.0
        ewz[all_conductors_mask] = 0.0
        Ewx_list.append(ewx)
        Ewy_list.append(ewy)
        Ewz_list.append(ewz)

    Ex_bg, Ey_bg, Ez_bg = np.gradient(-V_bg, params.dx, params.dy, params.dz)
    Ex_bg[all_conductors_mask] = 0.0
    Ey_bg[all_conductors_mask] = 0.0
    Ez_bg[all_conductors_mask] = 0.0

    # 7. Impact intersection & normal
    impact_pt, normal_vec = compute_impact_intersection_and_normal(
        params, Vw_body=np.zeros_like(V_bg), spacecraft_mask_3d=spacecraft_mask
    )

    print(f"  -> Analytická pole vyřešena pro {num_antennas} antén a těleso sondy (V_sc = {V_sc:.2f} V).")

    return PreparedGeometry3D(
        V_bg=V_bg,
        Vw_grids=Vw_grids,
        Ex_bg=Ex_bg,
        Ey_bg=Ey_bg,
        Ez_bg=Ez_bg,
        Ewx_list=Ewx_list,
        Ewy_list=Ewy_list,
        Ewz_list=Ewz_list,
        antenna_masks_3d=antenna_masks,
        spacecraft_mask_3d=spacecraft_mask,
        impact_pos=list(impact_pt),
        impact_normal=list(normal_vec)
    )
