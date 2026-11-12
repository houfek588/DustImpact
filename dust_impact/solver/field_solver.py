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
from dust_impact.numerics.interpolators import (
    interp_field_3d, deposit_charge_cic_3d, trilinear_gradient_3d, cells_touching_mask, particle_cell_indices
)
from dust_impact.numerics.poisson import build_pyamg_solver, assemble_negative_laplacian_3d, HAS_PYAMG
from dust_impact.numerics.gradient import ConductorGradient3D
from scipy.ndimage import binary_dilation
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
        Vw_sc: Optional[np.ndarray] = None,
        Ew_sc: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
        wire_coupling: Optional[np.ndarray] = None,
        Vw_list: Optional[List[np.ndarray]] = None,
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

        # Spacecraft weighting potential and field (floating spacecraft only, else None)
        self.Vw_sc = Vw_sc
        self.Ew_sc = Ew_sc
        # Antenna weighting potentials (energy-conserving force in cells touching a conductor)
        self.Vw_list = Vw_list
        self._dV_ant = np.zeros(num_antennas)
        self._dV_sc = 0.0

        Nx, Ny, Nz = self.p.Nx, self.p.Ny, self.p.Nz
        self.V_self_grid = np.zeros((Nx, Ny, Nz), dtype=np.float64)
        self.rho_grid = np.zeros((Nx, Ny, Nz), dtype=np.float64)
        self.Ex_self = np.zeros((Nx, Ny, Nz), dtype=np.float64)
        self.Ey_self = np.zeros((Nx, Ny, Nz), dtype=np.float64)
        self.Ez_self = np.zeros((Nx, Ny, Nz), dtype=np.float64)

        # Thin-wire antenna nodes (beta > 0) are free nodes coupled to the grounded wire (V_wire = 0 for the
        # self-field), the remaining conductor nodes are Dirichlet nodes
        self.wire_coupling = wire_coupling
        thin_nodes = (
            np.asarray(wire_coupling) > 0.0 if wire_coupling is not None
            else np.zeros_like(np.asarray(combined_mask, dtype=bool))
        )
        self.dirichlet_conductors = np.asarray(combined_mask, dtype=bool) & ~thin_nodes

        # Self-field E uses one-sided differences on conductor surface nodes (V_self = 0 there)
        self.gradient = ConductorGradient3D(self.dirichlet_conductors, (self.p.dx, self.p.dy, self.p.dz))

        # Cells within NEAR_CONDUCTOR_LAYERS of a conductor node: there the interpolated nodal fields are
        # not a gradient field (staircase surface), so the force is the exact gradient of the trilinearly
        # interpolated total potential (see get_accel)
        near_nodes = (
            binary_dilation(self.dirichlet_conductors, iterations=self.NEAR_CONDUCTOR_LAYERS)
            if np.any(self.dirichlet_conductors) else self.dirichlet_conductors
        )
        self.near_conductor_cells = cells_touching_mask(near_nodes)

        self.amg_solver = None
        self.lu_solver = None
        # Diagnostics of the most recent Poisson solve
        self.last_poisson_residual = 0.0
        self.last_poisson_iterations = 0
        self._build_poisson_solver()

    # Layers of cells around conductor nodes in which the force is an exact potential gradient
    NEAR_CONDUCTOR_LAYERS = 3

    # Relative residual ||b - A V|| / ||b|| required from the iterative Poisson solve, and its
    # iteration cap; with a warm start from the previous step a few CG iterations usually suffice.
    POISSON_RTOL = 1e-8
    POISSON_MAXITER = 200

    def _build_poisson_solver(self) -> None:
        """
        Assembles the 7-point negative Laplacian restricted to the free (non-Dirichlet) nodes.

        The self-field satisfies V = 0 on the outer boundary and on all conductors (spacecraft and
        antennas). Eliminating these Dirichlet nodes (rather than replacing their rows by identity
        rows) keeps the reduced matrix A_ff symmetric positive definite, which is required by the
        AMG-preconditioned conjugate gradient solver. Assembly is fully vectorised via Kronecker sums.
        """
        Nx, Ny, Nz = self.p.Nx, self.p.Ny, self.p.Nz
        N_tot = Nx * Ny * Nz

        print(f"=== Inicializace Poissonova řešiče (3D) ===")
        print(f"  -> Celkový počet uzlů: {N_tot}")

        A_full = assemble_negative_laplacian_3d(Nx, Ny, Nz, self.p.dx, self.p.dy, self.p.dz)

        dirichlet = np.zeros((Nx, Ny, Nz), dtype=bool)
        dirichlet[[0, -1], :, :] = True
        dirichlet[:, [0, -1], :] = True
        dirichlet[:, :, [0, -1]] = True
        dirichlet |= self.dirichlet_conductors
        self.free_idx = np.flatnonzero(~dirichlet.ravel())

        A_free = A_full[self.free_idx][:, self.free_idx].tocsr()
        if self.wire_coupling is not None:
            A_free = (A_free + sp.diags(np.asarray(self.wire_coupling, dtype=np.float64).ravel()[self.free_idx])).tocsr()
        self.A_free = A_free
        print(f"  -> Volné uzly: {len(self.free_idx)} (Dirichlet: {N_tot - len(self.free_idx)})")

        if HAS_PYAMG:
            print("  -> Stavím AMG hierarchii (PyAMG)...")
            self.amg_solver = build_pyamg_solver(A_free)
        else:
            print("  -> PyAMG nenalezen, používám LU solver (SciPy)...")
            self.lu_solver = spla.factorized(A_free.tocsc())

    def _background_enabled(self) -> bool:
        return bool(getattr(self.toggles, 'enable_spis_background_field', True))

    def antenna_potential_perturbation(self, voltage_ant: np.ndarray, step: int) -> np.ndarray:
        """
        Returns dV_a = V_a(t) - V_a,ref for each antenna, i.e. the antenna potential that is not
        already contained in the static background V_bg_base.

        V_bg_base holds every antenna at its reference potential p.V_bias (configured bias or SPIS
        equilibrium). When the static background is disabled, the reference is 0 V. Combined with the
        circuit target (V_bias if enable_antenna_bias_voltage else 0 V) this gives consistent fields
        for all four toggle combinations, e.g. background on + bias off removes the bias from V_bg.

        Parameters:
            voltage_ant: Antenna voltage history, shape (num_antennas, steps) [V].
            step: Current step; the voltage of the previous step is used (explicit coupling).

        Returns:
            np.ndarray: Potential perturbation per antenna [V].
        """
        col = max(0, step - 1)
        bg_on = self._background_enabled()
        dV = np.zeros(self.num_antennas)
        for a_idx in range(self.num_antennas):
            V_ref = (self.p.V_bias[a_idx] if a_idx < len(self.p.V_bias) else 0.0) if bg_on else 0.0
            dV[a_idx] = voltage_ant[a_idx, col] - V_ref
        return dV

    def spacecraft_potential_perturbation(self, voltage_sc: Optional[np.ndarray], step: int) -> float:
        """
        Returns dV_sc = V_sc(t) - V_sc,ref for a floating spacecraft (0 for a fixed spacecraft).

        V_bg_base holds the spacecraft at its bias potential p.V_sc_bias; without the static
        background the reference is 0 V, mirroring antenna_potential_perturbation.

        Parameters:
            voltage_sc: Spacecraft voltage history, shape (steps,) [V], or None for a fixed spacecraft.
            step: Current step; the voltage of the previous step is used (explicit coupling).

        Returns:
            float: Spacecraft potential perturbation [V].
        """
        if voltage_sc is None or self.Vw_sc is None:
            return 0.0
        V_ref = self.p.V_sc_bias if self._background_enabled() else 0.0
        return float(voltage_sc[max(0, step - 1)] - V_ref)

    def _external_potential_terms(self) -> Tuple[List[np.ndarray], List[float]]:
        """
        Grids and coefficients whose weighted sum is the external potential seen by the particles:
        V_bg_base (if enabled) + sum_a dV_a Vw_a + dV_sc Vw_sc, consistent with update_background_fields.
        """
        fields: List[np.ndarray] = []
        weights: List[float] = []
        if self._background_enabled():
            fields.append(self.V_bg_base)
            weights.append(1.0)
        if self.Vw_list is not None:
            for Vw, dV in zip(self.Vw_list, self._dV_ant):
                if abs(dV) > 1e-6:
                    fields.append(Vw)
                    weights.append(float(dV))
        if self.Vw_sc is not None and abs(self._dV_sc) > 1e-9:
            fields.append(self.Vw_sc)
            weights.append(self._dV_sc)
        return fields, weights

    def update_background_fields(
        self, voltage_ant: np.ndarray, step: int, voltage_sc: Optional[np.ndarray] = None
    ) -> None:
        """
        Sets the external electric field to the static background (if enabled) plus the field of
        the conductor potential perturbations, E = E_bg_base + sum_a dV_a * E_w,a (+ dV_sc * E_w,sc
        for a floating spacecraft).
        """
        if self._background_enabled():
            Ex_tot = self.Ex_bg_base.copy()
            Ey_tot = self.Ey_bg_base.copy()
            Ez_tot = self.Ez_bg_base.copy()
        else:
            Ex_tot = np.zeros_like(self.Ex_bg_base)
            Ey_tot = np.zeros_like(self.Ey_bg_base)
            Ez_tot = np.zeros_like(self.Ez_bg_base)

        dV_all = self.antenna_potential_perturbation(voltage_ant, step)
        for a_idx, dV in enumerate(dV_all):
            if abs(dV) > 1e-6:
                Ex_tot += dV * self.Ewx_list[a_idx]
                Ey_tot += dV * self.Ewy_list[a_idx]
                Ez_tot += dV * self.Ewz_list[a_idx]

        dV_sc = self.spacecraft_potential_perturbation(voltage_sc, step)
        if self.Ew_sc is not None and abs(dV_sc) > 1e-9:
            Ex_tot += dV_sc * self.Ew_sc[0]
            Ey_tot += dV_sc * self.Ew_sc[1]
            Ez_tot += dV_sc * self.Ew_sc[2]

        self.Ex_bg, self.Ey_bg, self.Ez_bg = Ex_tot, Ey_tot, Ez_tot
        self._dV_ant = np.asarray(dV_all, dtype=float)
        self._dV_sc = float(dV_sc) if self.Vw_sc is not None else 0.0

    def background_potential(
        self,
        voltage_ant: np.ndarray,
        step: int,
        Vw_list: Optional[List[np.ndarray]],
        voltage_sc: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        External potential consistent with update_background_fields:
        V = V_bg_base (if enabled) + sum_a dV_a * Vw_a (+ dV_sc * Vw_sc for a floating spacecraft).

        Parameters:
            voltage_ant: Antenna voltage history, shape (num_antennas, steps) [V].
            step: Current step.
            Vw_list: Weighting potentials per antenna; if None, the perturbation is omitted.
            voltage_sc: Spacecraft voltage history (floating spacecraft) or None.

        Returns:
            np.ndarray: External potential on the grid [V].
        """
        V = self.V_bg_base.copy() if self._background_enabled() else np.zeros_like(self.V_bg_base)
        if Vw_list is not None:
            dV_all = self.antenna_potential_perturbation(voltage_ant, step)
            for a_idx, dV in enumerate(dV_all):
                if abs(dV) > 1e-6 and a_idx < len(Vw_list):
                    V += dV * Vw_list[a_idx]
        dV_sc = self.spacecraft_potential_perturbation(voltage_sc, step)
        if abs(dV_sc) > 1e-9:
            V += dV_sc * self.Vw_sc
        return V

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

        # -Laplacian(V) = rho / eps_0 on the free nodes, V = 0 on boundary and conductors: A_ff V_f = b_f
        b_free = np.nan_to_num((rho_grid / eps_0).ravel()[self.free_idx], nan=0.0, posinf=0.0, neginf=0.0)
        V_self_1d = np.zeros(Nx * Ny * Nz, dtype=np.float64)

        b_norm = np.linalg.norm(b_free)
        if b_norm == 0.0:
            self.last_poisson_residual, self.last_poisson_iterations = 0.0, 0
        elif HAS_PYAMG and self.amg_solver is not None:
            # Warm start from the previous step; the solution changes little between PIC steps
            x_init = self.V_self_grid.ravel()[self.free_idx]
            residuals: List[float] = []
            V_free = self.amg_solver.solve(
                b_free, x0=x_init, tol=self.POISSON_RTOL, maxiter=self.POISSON_MAXITER,
                accel='cg', residuals=residuals,
            )
            rel_res = float(np.linalg.norm(b_free - self.A_free @ V_free) / b_norm)
            self.last_poisson_residual = rel_res
            self.last_poisson_iterations = max(0, len(residuals) - 1)
            if not np.isfinite(rel_res) or rel_res > 100.0 * self.POISSON_RTOL:
                raise RuntimeError(
                    f"Poisson solver did not converge: relative residual {rel_res:.3e} after "
                    f"{self.last_poisson_iterations} iterations (required {self.POISSON_RTOL:.1e})."
                )
            V_self_1d[self.free_idx] = V_free
        else:
            V_self_1d[self.free_idx] = self.lu_solver(b_free)
            self.last_poisson_residual, self.last_poisson_iterations = 0.0, 0

        self.V_self_grid = V_self_1d.reshape((Nx, Ny, Nz))
        self.Ex_self, self.Ey_self, self.Ez_self = self.gradient.electric_field(self.V_self_grid)

    def interp_field(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, field_3d: np.ndarray) -> np.ndarray:
        """Interpolates 3D Cartesian field onto arbitrary particle coordinates."""
        return interp_field_3d(
            x, y, z, self.p.x_grid, self.p.y_grid, self.p.z_grid,
            self.p.dx, self.p.dy, self.p.dz, field_3d
        )

    def get_accel(self, x_act: np.ndarray, y_act: np.ndarray, z_act: np.ndarray,
                  mass: float, phys_charge: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculates total acceleration from background and self-consistent electric fields.

        Away from conductors both the external field (static background + conductor potential
        perturbations) and the self-field are trilinear interpolations of their nodal fields, so their
        discretisation errors are the same and cancel where the plasma screens the conductor potential.

        In the NEAR_CONDUCTOR_LAYERS cells around conductor nodes the force is the exact gradient of the
        trilinearly interpolated total potential (external + self). It is conservative, so a particle
        crossing the sheath gains q times the potential difference; interpolated nodal fields next to a
        staircase surface miss up to ~10 % of the sheath potential for oblique paths and bias charge
        escape and recollection. Treating both parts alike keeps the total force consistent there too.
        """
        self_field_on = getattr(self.toggles, 'enable_plasma_self_field', True)
        Ex_tot = self.interp_field(x_act, y_act, z_act, self.Ex_bg)
        Ey_tot = self.interp_field(x_act, y_act, z_act, self.Ey_bg)
        Ez_tot = self.interp_field(x_act, y_act, z_act, self.Ez_bg)
        if self_field_on:
            Ex_tot += self.interp_field(x_act, y_act, z_act, self.Ex_self)
            Ey_tot += self.interp_field(x_act, y_act, z_act, self.Ey_self)
            Ez_tot += self.interp_field(x_act, y_act, z_act, self.Ez_self)

        # Without the antenna weighting potentials the external potential is unknown near conductors;
        # the interpolated nodal fields are then kept everywhere
        external_known = self.Vw_list is not None or self.num_antennas == 0
        if external_known and len(x_act) > 0 and np.any(self.near_conductor_cells):
            grid = (self.p.x_grid, self.p.y_grid, self.p.z_grid)
            spacing = (self.p.dx, self.p.dy, self.p.dz)
            i, j, k = particle_cell_indices(x_act, y_act, z_act, *grid, *spacing, self.near_conductor_cells.shape)
            near = self.near_conductor_cells[i, j, k]
            if np.any(near):
                fields, weights = self._external_potential_terms()
                if self_field_on:
                    fields.append(self.V_self_grid)
                    weights.append(1.0)
                if fields:
                    gx, gy, gz = trilinear_gradient_3d(
                        x_act[near], y_act[near], z_act[near], *grid, *spacing, fields, weights
                    )
                    Ex_tot[near], Ey_tot[near], Ez_tot[near] = -gx, -gy, -gz
                else:
                    Ex_tot[near], Ey_tot[near], Ez_tot[near] = 0.0, 0.0, 0.0

        factor = phys_charge / mass
        return factor * Ex_tot, factor * Ey_tot, factor * Ez_tot
