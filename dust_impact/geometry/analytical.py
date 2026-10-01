# -*- coding: utf-8 -*-
"""
Analytical geometry definitions and exact electrostatic solutions for synthetic benchmarks.
Provides exact potentials V and electric fields E for spheres, boxes, and antennas.
"""

from typing import Tuple, Sequence, Optional, List, Any
import numpy as np
from dust_impact.geometry.voxelizer import voxelize_sphere, voxelize_box, voxelize_cylinder


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
