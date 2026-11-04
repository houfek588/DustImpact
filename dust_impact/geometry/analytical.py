# -*- coding: utf-8 -*-
"""
Analytical geometry definitions and exact electrostatic solutions for synthetic benchmarks.
Provides exact potentials V and electric fields E for spheres, boxes, and antennas.
"""

from typing import Tuple, Sequence, Optional, List, Any
import numpy as np
from dust_impact.geometry.voxelizer import voxelize_sphere, voxelize_box, voxelize_cylinder
from dust_impact.numerics.poisson import DirichletLaplaceSolver3D
from dust_impact.numerics.gradient import ConductorGradient3D
from dust_impact.physics.ramo_shockley import vacuum_capacitance_matrix
from dust_impact.geometry.thin_wire import (
    thin_wire_voxel_radius, calibrate_effective_radius, thin_wire_coupling, wire_length_outside
)


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
    coupling: Optional[np.ndarray] = None,
) -> List[np.ndarray]:
    """
    Solves the discrete 3D Laplace equation for several Dirichlet problems sharing one node pattern.

    The fixed nodes are the conductor nodes (dirichlet_mask, values taken from each rhs grid) and the
    outer faces of the grid (V = 0, the grounded boundary of the PIC self-field). The 7-point stencil
    and the boundary treatment are those of the PIC Poisson solver, which keeps the weighting
    potentials consistent with the Ramo-Shockley theorem on the grid. The solves use
    DirichletLaplaceSolver3D (eliminated Dirichlet nodes, AMG-preconditioned CG with a checked
    residual); the AMG hierarchy is built once and shared by all right-hand sides.

    Parameters
    ----------
    Nx, Ny, Nz : int
        Grid dimensions.
    dx, dy, dz : float
        Grid spacings in metres.
    dirichlet_mask : np.ndarray
        Boolean array (Nx, Ny, Nz), True on conductor nodes.
    rhs_values : List[np.ndarray]
        Conductor potentials for each problem, shape (Nx, Ny, Nz); only values on dirichlet_mask and on
        the thin-wire nodes (coupling > 0) are used.
    coupling : Optional[np.ndarray]
        Thin-wire coupling coefficients beta [1/m^2] (dust_impact.geometry.thin_wire); nonzero nodes are
        thin-wire nodes, free nodes held towards the wire potential given in rhs_values.

    Returns
    -------
    List[np.ndarray]
        Discrete-harmonic fields equal to the prescribed values on the conductors and 0 on the outer faces.
    """
    conductors = np.asarray(dirichlet_mask, dtype=bool)
    fixed_mask = conductors.copy()
    fixed_mask[[0, -1], :, :] = True
    fixed_mask[:, [0, -1], :] = True
    fixed_mask[:, :, [0, -1]] = True

    targets = conductors | (coupling > 0.0) if coupling is not None else conductors

    solver = DirichletLaplaceSolver3D(fixed_mask, (dx, dy, dz), coupling=coupling)
    return [solver.solve(np.where(targets, rhs_grid, 0.0)) for rhs_grid in rhs_values]


def _thin_wire_coupling_for_antennas(
    params: Any, ant_geoms: List[Any], antenna_masks: List[np.ndarray], sc_part: Any,
    spacing: Tuple[float, float, float],
) -> Optional[np.ndarray]:
    """
    Thin-wire coupling coefficients of all antennas thinner than their voxel representation.

    Returns the summed beta array, or None if every antenna is resolved by the grid (r_eff <= radius).
    """
    grid = (params.x_grid, params.y_grid, params.z_grid)

    def inside_spacecraft(px: np.ndarray, py: np.ndarray, pz: np.ndarray) -> np.ndarray:
        if sc_part is None:
            return np.zeros(np.shape(px), dtype=bool)
        return voxelize_analytical_spacecraft_part(px, py, pz, sc_part)

    coupling = np.zeros(antenna_masks[0].shape)
    for i, (ant, mask) in enumerate(zip(ant_geoms, antenna_masks)):
        radius = float(getattr(ant, 'radius', 0.015))
        r_eff = calibrate_effective_radius(ant.p_start, ant.p_end, thin_wire_voxel_radius(radius, spacing), grid)
        if r_eff <= radius:
            print(f"  -> Anténa {i + 1}: drát je rozlišen mřížkou (a = {radius * 1e3:.2f} mm >= "
                  f"r_eff = {r_eff * 1e3:.2f} mm), model tenkého drátu se nepoužije.")
            continue
        length = wire_length_outside(ant.p_start, ant.p_end, inside_spacecraft)
        coupling += thin_wire_coupling(mask, length, radius, r_eff, spacing)
        print(f"  -> Anténa {i + 1}: model tenkého drátu (a = {radius * 1e3:.2f} mm, efektivní poloměr "
              f"uzlů r_eff = {r_eff * 1e3:.2f} mm, {int(mask.sum())} uzlů).")
    return coupling if np.any(coupling > 0.0) else None


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

    # 2. Antenna masks, disjoint from the spacecraft and from each other so that every conductor node
    # has a single owner (Vw_a = 1 only on antenna a, the background takes that conductor's potential)
    ant_geoms = getattr(analytical_config, 'antennas', [])
    spacing = (params.dx, params.dy, params.dz)
    thin_wire = bool(getattr(analytical_config, 'thin_wire_model', False))
    antenna_masks = []
    occupied = spacecraft_mask.copy()
    for ant in ant_geoms:
        p_start = getattr(ant, 'p_start', [0.0, 0.0, 0.0])
        p_end = getattr(ant, 'p_end', [1.0, 0.0, 0.0])
        radius = getattr(ant, 'radius', 0.015)
        vox_radius = thin_wire_voxel_radius(radius, spacing) if thin_wire else radius
        m = voxelize_cylinder(X, Y, Z, p_start, p_end, vox_radius) & ~occupied
        occupied |= m
        antenna_masks.append(m)

    num_antennas = len(antenna_masks)

    # 3. Combined conductor mask; with the thin-wire model, wires thinner than their voxel
    # representation become coupling nodes (free nodes) instead of Dirichlet nodes
    all_conductors_mask = spacecraft_mask.copy()
    for m in antenna_masks:
        all_conductors_mask |= m
    wire_coupling = None
    dirichlet_conductors = all_conductors_mask
    if thin_wire and num_antennas > 0:
        wire_coupling = _thin_wire_coupling_for_antennas(params, ant_geoms, antenna_masks, sc_part, spacing)
        if wire_coupling is not None:
            dirichlet_conductors = all_conductors_mask & ~(wire_coupling > 0.0)

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

    # Weighting potential of the spacecraft body (floating-spacecraft circuit)
    rhs_list.append(spacecraft_mask.astype(float))

    rhs_bg = np.zeros((Nx, Ny, Nz), dtype=float)
    rhs_bg[spacecraft_mask] = float(V_sc)
    effective_biases = []
    for i in range(num_antennas):
        ant = ant_geoms[i]
        v_ant = getattr(ant, 'effective_voltage', None)
        if v_ant is None:
            v_ant = getattr(ant, 'voltage_V', 0.0)
        v_ant = float(v_ant) if v_ant is not None else 0.0
        effective_biases.append(v_ant)
        rhs_bg[antenna_masks[i]] = v_ant
    rhs_list.append(rhs_bg)

    # 5. Solve the vacuum Laplace equation on the grid (conductor values are imposed exactly)
    solved_fields = solve_laplace_dirichlet_3d(
        Nx, Ny, Nz, params.dx, params.dy, params.dz,
        dirichlet_mask=dirichlet_conductors,
        rhs_values=rhs_list,
        coupling=wire_coupling,
    )

    Vw_grids = solved_fields[:num_antennas]
    Vw_sc = solved_fields[num_antennas]
    V_bg = solved_fields[num_antennas + 1]
    capacitance_matrix = vacuum_capacitance_matrix(
        Vw_grids + [Vw_sc], antenna_masks + [spacecraft_mask], (params.dx, params.dy, params.dz)
    )

    params.antenna_bias_voltage_V = effective_biases
    params.V_bias = effective_biases

    # 6. Compute electric fields (-grad V) with one-sided differences on conductor surfaces
    grad = ConductorGradient3D(dirichlet_conductors, (params.dx, params.dy, params.dz))
    Ewx_list, Ewy_list, Ewz_list = [], [], []
    for i in range(num_antennas):
        ewx, ewy, ewz = grad.electric_field(Vw_grids[i])
        Ewx_list.append(ewx)
        Ewy_list.append(ewy)
        Ewz_list.append(ewz)

    Ewx_sc, Ewy_sc, Ewz_sc = grad.electric_field(Vw_sc)
    Ex_bg, Ey_bg, Ez_bg = grad.electric_field(V_bg)

    # 7. Impact intersection & normal (only for localized dust impact)
    is_homogeneous = getattr(params, 'plasma_injection_mode', 'point_cloud') == 'homogeneous'
    if not is_homogeneous:
        impact_pt, normal_vec = compute_impact_intersection_and_normal(
            params, Vw_body=np.zeros_like(V_bg), spacecraft_mask_3d=spacecraft_mask
        )
        impact_pos = list(impact_pt) if impact_pt is not None else None
        impact_normal = list(normal_vec) if normal_vec is not None else None
    else:
        impact_pos = None
        impact_normal = None

    print(f"  -> Analytická pole vyřešena pro {num_antennas} antén a těleso sondy (V_sc = {V_sc:.2f} V).")

    target_dtype = np.float32 if getattr(params, 'field_precision', 'float32') == 'float32' else np.float64
    V_bg = np.ascontiguousarray(V_bg, dtype=target_dtype)
    Ex_bg = np.ascontiguousarray(Ex_bg, dtype=target_dtype)
    Ey_bg = np.ascontiguousarray(Ey_bg, dtype=target_dtype)
    Ez_bg = np.ascontiguousarray(Ez_bg, dtype=target_dtype)
    Vw_grids = [np.ascontiguousarray(v, dtype=target_dtype) for v in Vw_grids]
    Ewx_list = [np.ascontiguousarray(e, dtype=target_dtype) for e in Ewx_list]
    Ewy_list = [np.ascontiguousarray(e, dtype=target_dtype) for e in Ewy_list]
    Ewz_list = [np.ascontiguousarray(e, dtype=target_dtype) for e in Ewz_list]
    Vw_sc, Ewx_sc, Ewy_sc, Ewz_sc = (
        np.ascontiguousarray(f, dtype=target_dtype) for f in (Vw_sc, Ewx_sc, Ewy_sc, Ewz_sc)
    )

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
        impact_pos=impact_pos,
        impact_normal=impact_normal,
        antenna_geometries=ant_geoms,
        spacecraft_geometry=sc_part,
        Vw_sc=Vw_sc, Ewx_sc=Ewx_sc, Ewy_sc=Ewy_sc, Ewz_sc=Ewz_sc,
        capacitance_matrix=capacitance_matrix,
        wire_coupling=wire_coupling,
    )
