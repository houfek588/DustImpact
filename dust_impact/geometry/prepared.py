# -*- coding: utf-8 -*-
"""
Prepared simulation geometry container and builder.
Constructs, validates, and packages all 3D fields, conductor masks, and impact metadata
for direct consumption by DustImpactSimulation3D.
"""

import os
from dataclasses import dataclass, field
from typing import List, Tuple, Sequence, Optional, Dict, Any, Iterator
import numpy as np

try:
    import pyvista as pv
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False

from dust_impact.geometry.voxelizer import detect_metal_mask_3d
from dust_impact.geometry.surface import compute_impact_intersection_and_normal
from dust_impact.geometry.analytical import generate_synthetic_analytical_fields
from dust_impact.numerics.poisson import DirichletLaplaceSolver3D
from dust_impact.geometry.spis_loader import (
    read_spis_mesh,
    extract_potential_from_mesh,
    sample_mesh_to_grid,
    extract_enclosed_conductor_mask,
    extract_antenna_bias_from_spis
)


@dataclass
class PreparedGeometry3D:
    """
    Complete, validated 3D geometry and field package for PIC simulations.

    Attributes
    ----------
    V_bg : np.ndarray
        Equilibrium electrostatic background potential grid (Nx, Ny, Nz) [V].
    Vw_grids : List[np.ndarray]
        List of Ramo-Shockley weighting potential grids for each antenna.
    Ex_bg, Ey_bg, Ez_bg : np.ndarray
        Equilibrium electrostatic background electric field components [V/m].
    Ewx_list, Ewy_list, Ewz_list : List[np.ndarray]
        List of weighting field gradients (-grad(Vw)) for each antenna.
    antenna_masks_3d : List[np.ndarray]
        Boolean 3D masks indicating conductor interior for each antenna.
    spacecraft_mask_3d : np.ndarray
        Boolean 3D mask indicating spacecraft conductor interior.
    impact_pos : List[float]
        Exact impact position [x, y, z] on spacecraft surface in meters.
    impact_normal : List[float]
        Outward unit surface normal [nx, ny, nz] at impact point.
    """
    V_bg: np.ndarray
    Vw_grids: List[np.ndarray]
    Ex_bg: np.ndarray
    Ey_bg: np.ndarray
    Ez_bg: np.ndarray
    Ewx_list: List[np.ndarray]
    Ewy_list: List[np.ndarray]
    Ewz_list: List[np.ndarray]
    antenna_masks_3d: List[np.ndarray]
    spacecraft_mask_3d: np.ndarray
    impact_pos: Optional[List[float]] = None
    impact_normal: Optional[List[float]] = None
    antenna_geometries: Optional[List[Any]] = None
    spacecraft_geometry: Optional[Any] = None

    @property
    def num_antennas(self) -> int:
        return len(self.antenna_masks_3d) if self.antenna_masks_3d is not None else 0

    @property
    def grid_shape(self) -> Tuple[int, int, int]:
        return self.V_bg.shape

    def __iter__(self) -> Iterator[Any]:
        """
        Supports seamless 10-tuple unpacking for 100% backward compatibility:
        V_bg, Vw_grids, Ex_bg, Ey_bg, Ez_bg, Ewx, Ewy, Ewz, ant_masks, sc_mask = prep_geom
        """
        yield self.V_bg
        yield self.Vw_grids
        yield self.Ex_bg
        yield self.Ey_bg
        yield self.Ez_bg
        yield self.Ewx_list
        yield self.Ewy_list
        yield self.Ewz_list
        yield self.antenna_masks_3d
        yield self.spacecraft_mask_3d

    def __len__(self) -> int:
        return 10

    def __getitem__(self, idx: int) -> Any:
        return list(self)[idx]

    def summary(self) -> str:
        """ Returns human-readable summary of the prepared geometry. """
        sc_cells = int(np.sum(self.spacecraft_mask_3d))
        sc_frac = sc_cells / float(self.spacecraft_mask_3d.size) * 100.0
        if self.impact_pos is not None:
            impact_desc = f", Impact pos: {self.impact_pos}, Normal: {self.impact_normal}"
        else:
            impact_desc = ", Plasma mode: Homogeneous (no localized impact)"
        return (
            f"PreparedGeometry3D: Grid {self.grid_shape}, Antennas: {self.num_antennas}, "
            f"Spacecraft cells: {sc_cells} ({sc_frac:.2f}%){impact_desc}"
        )


def build_simulation_geometry(params: Any) -> PreparedGeometry3D:
    """
    Builds and prepares all simulation geometry, potential fields, and boundary masks
    for the 3D solver, loading from SPIS VTK meshes or using analytical models.

    Parameters
    ----------
    params : SimulationParams3D
        Full configuration containing grid parameters and geometry/VTK file locations.

    Returns
    -------
    PreparedGeometry3D
        Validated container ready for solver instantiation.
    """
    geom_cfg = getattr(params, 'geometry', None)
    if geom_cfg is not None and getattr(geom_cfg, 'source', '') == 'analytical':
        from dust_impact.geometry.analytical import build_analytical_simulation_geometry
        return build_analytical_simulation_geometry(params, geom_cfg.analytical)

    if not HAS_PYVISTA:
        print("[GEOMETRY] PyVista není k dispozici - používám analytický syntetický generátor.")
        raw_fields = generate_synthetic_analytical_fields(params)
        V_bg, Vw_grids, Ex_bg, Ey_bg, Ez_bg, Ewx, Ewy, Ewz, ant_masks, sc_mask = raw_fields
        impact_pos = getattr(params, 'impact_pos', [-2.0, 2.0, 0.0])
        impact_norm = getattr(params, 'impact_normal', [-0.7071, 0.7071, 0.0])
        return PreparedGeometry3D(
            V_bg=V_bg, Vw_grids=Vw_grids,
            Ex_bg=Ex_bg, Ey_bg=Ey_bg, Ez_bg=Ez_bg,
            Ewx_list=Ewx, Ewy_list=Ewy, Ewz_list=Ewz,
            antenna_masks_3d=ant_masks, spacecraft_mask_3d=sc_mask,
            impact_pos=impact_pos, impact_normal=impact_norm
        )

    print("Mapování SPIS VTK na 3D pravoúhlou mřížku...")
    # VTK ImageData orders its points with x varying fastest, so every sampled point array is
    # reshaped to (Nx, Ny, Nz) in Fortran order to match the [i, j, k] = (x, y, z) indexing of the solver.
    pic_grid = pv.ImageData(
        dimensions=(params.Nx, params.Ny, params.Nz),
        spacing=(params.dx, params.dy, params.dz),
        origin=(-params.L_x, -params.L_y, -params.L_z)
    )

    Nx, Ny, Nz = params.Nx, params.Ny, params.Nz
    dx_min = min(params.dx, params.dy, params.dz)

    if geom_cfg is not None and getattr(geom_cfg, 'source', '') == 'spis':
        spis = geom_cfg.spis
        bg_file = getattr(spis, 'background_potential_file', '')
        sc_file = getattr(spis, 'spacecraft_weighting_file', '')
        ant_files = getattr(spis, 'antenna_weighting_files', [])
        mesh_file = getattr(spis, 'spacecraft_surface_mesh_file', '')
        sc_threshold = getattr(spis, 'weighting_threshold', 0.85)
    else:
        bg_file = getattr(params.vtk_files, 'spis_background_potential_file', getattr(params.vtk_files, 'background_potential', ''))
        sc_file = getattr(params.vtk_files, 'spacecraft_weighting_file', getattr(params.vtk_files, 'spacecraft_weighting', 'inputs/spis_Vw_body.vtk'))
        ant_files = getattr(params.vtk_files, 'antenna_weighting_files', getattr(params.vtk_files, 'antenna_weighting', []))
        mesh_file = ''
        sc_threshold = 0.85

    # 1. Load spacecraft body geometry and weighting field
    mesh_body = None
    Vw_body = np.zeros((Nx, Ny, Nz))
    spacecraft_mask_3d = np.zeros((Nx, Ny, Nz), dtype=bool)

    # Optional external mesh from Gmsh / STL / VTK
    if mesh_file and (os.path.exists(mesh_file) or os.path.exists(os.path.join("inputs", mesh_file))):
        try:
            mesh_body = read_spis_mesh(mesh_file)
            print(f"  -> Načtena explicitní povrchová síť tělesa sondy z {mesh_file}.")
        except Exception as e:
            print(f"[VAROVÁNÍ] Nelze načíst explicitní síť tělesa sondy z {mesh_file}: {e}")

    if sc_file and (os.path.exists(sc_file) or os.path.exists(os.path.join("inputs", sc_file))):
        try:
            sc_mesh = read_spis_mesh(sc_file)
            if mesh_body is None:
                mesh_body = sc_mesh
            Vw_body = sample_mesh_to_grid(sc_mesh, pic_grid, sc_file)

            # Normalize weighting field to 1.0 at conductor surface if unnormalized
            vw_max_abs = np.nanmax(np.abs(Vw_body))
            if vw_max_abs > 1e-6 and abs(vw_max_abs - 1.0) > 0.05:
                vw_peak = Vw_body.ravel()[np.nanargmax(np.abs(Vw_body))]
                Vw_body = Vw_body / vw_peak

            spacecraft_mask_3d = extract_enclosed_conductor_mask(
                mesh_body, pic_grid, Vw_body, dx_min, threshold=sc_threshold
            )
            print(f"  -> Geometrie tělesa sondy načtena z {sc_file} (PyVista select_enclosed_points).")
        except Exception as e:
            print(f"[UPOZORNĚNÍ] Chyba při načítání tělesa sondy ({e}). Používám syntetické těleso.")
            X_mat, Y_mat, Z_mat = np.meshgrid(params.x_grid, params.y_grid, params.z_grid, indexing='ij')
            spacecraft_mask_3d = (X_mat**2 + Y_mat**2 + Z_mat**2) <= 1.0**2
    else:
        print(f"  -> Soubor tělesa sondy nezadán. Používám syntetické kulové těleso.")
        X_mat, Y_mat, Z_mat = np.meshgrid(params.x_grid, params.y_grid, params.z_grid, indexing='ij')
        spacecraft_mask_3d = (X_mat**2 + Y_mat**2 + Z_mat**2) <= 1.0**2

    # Conductor is an equipotential body: Vw inside metal is exactly 1.0
    Vw_body[spacecraft_mask_3d] = 1.0

    # 2. Load weighting fields for all antennas
    Vw_grids = []
    antenna_masks_3d = []
    # Nodes already assigned to a conductor; the conductor masks must be disjoint
    occupied = spacecraft_mask_3d.copy()

    for i, vtk_file in enumerate(ant_files):
        mesh_ant = None
        try:
            mesh_ant = read_spis_mesh(vtk_file)
            Vw = sample_mesh_to_grid(mesh_ant, pic_grid, vtk_file)

            # Normalize weighting field to 1.0 on conductor surface
            vw_ant_max = np.nanmax(np.abs(Vw))
            if vw_ant_max > 1e-6 and abs(vw_ant_max - 1.0) > 0.05:
                peak = Vw.ravel()[np.nanargmax(np.abs(Vw))]
                Vw = Vw / peak
        except (FileNotFoundError, KeyError, Exception) as e:
            print(f"[UPOZORNĚNÍ] VTK antény {i+1} nenalezen ({e}). Používám syntetické pole.")
            Vw = np.zeros((Nx, Ny, Nz))

        threshold = sc_threshold
        if hasattr(params, 'weighting_threshold') and params.weighting_threshold is not None:
            threshold = params.weighting_threshold

        mask = extract_enclosed_conductor_mask(
            mesh_ant, pic_grid, Vw, dx_min, threshold=threshold
        )
        if np.sum(mask) == 0:
            mask = (Vw >= threshold)
        if np.sum(mask) == 0 and np.nanmax(np.abs(Vw)) > 1e-4:
            # Adaptive fallback if global threshold is too strict for sampled thin-wire antenna field
            adaptive_thresh = max(0.2, 0.5 * np.nanmax(np.abs(Vw)))
            mask = (np.abs(Vw) >= adaptive_thresh)

        # A node shared with the spacecraft (or an earlier antenna) stays with that conductor
        overlap = int(np.sum(mask & occupied))
        if overlap:
            print(f"  -> Anténa {i + 1}: {overlap} uzlů masky leží v jiném vodiči, odebírám je z masky antény.")
        mask = mask & ~occupied
        occupied |= mask

        Vw_grids.append(Vw)
        antenna_masks_3d.append(mask)

    # 3. Conductor potentials: spacecraft V_sc and antenna reference biases
    mesh_bg = None
    V_bg_grid = np.zeros((Nx, Ny, Nz))
    V_sc = getattr(params, 'spacecraft_voltage_V', params.Vf)
    if V_sc is None:
        V_sc = params.Vf
    V_sc = float(V_sc)

    if bg_file and (os.path.exists(bg_file) or os.path.exists(os.path.join("inputs", bg_file))):
        try:
            mesh_bg = read_spis_mesh(bg_file)
            V_bg_grid = sample_mesh_to_grid(mesh_bg, pic_grid, bg_file)
            print(f"  -> Pozadí načteno z {bg_file}. Rozsah potenciálu: {np.min(V_bg_grid):.2f} až {np.max(V_bg_grid):.2f} V")
        except Exception as e:
            print(f"[UPOZORNĚNÍ] Nelze načíst pozadí z '{bg_file}' ({e}). Nastavuji počáteční pozadí na 0.0 V.")
            mesh_bg = None
            V_bg_grid = np.zeros((Nx, Ny, Nz))

    # With a SPIS background the conductor potentials belong to that solution: the spacecraft and the
    # antennas take the potential of their surfaces in the background file, and any configured value
    # is ignored (it would be inconsistent with the plasma potential computed by SPIS).
    if mesh_bg is not None and sc_file:
        V_sc_spis = extract_antenna_bias_from_spis(mesh_bg, sc_file, default_bias=None)
        if V_sc_spis is not None:
            if abs(V_sc - V_sc_spis) > 1e-6:
                print(f"[UPOZORNĚNÍ] Potenciál sondy {V_sc:.3f} V z konfigurace ignoruji, "
                      f"pozadí SPIS určuje {V_sc_spis:.3f} V.")
            print(f"  -> Potenciál sondy načten přímo ze SPIS VTK: {V_sc_spis:.3f} V")
            V_sc = V_sc_spis
            params.spacecraft_voltage_V = V_sc_spis

    configured_biases = getattr(params, 'antenna_bias_voltage_V', getattr(params, 'V_bias', []))
    antenna_biases = []
    for i, vtk_file in enumerate(ant_files):
        v_ant = extract_antenna_bias_from_spis(mesh_bg, vtk_file, default_bias=None) if mesh_bg is not None else None
        if v_ant is not None:
            print(f"  -> Anténa {i + 1}: Rovnovážný potenciál načten přímo ze SPIS VTK: {v_ant:.3f} V")
        elif i < len(configured_biases):
            v_ant = configured_biases[i]
            print(f"  -> Anténa {i + 1}: Používám zadaný potenciál z parametrů: {v_ant:.3f} V")
        else:
            v_ant = getattr(params, 'Vf_antenne', 0.0)
            print(f"  -> Anténa {i + 1}: Používám plovoucí potenciál: {v_ant:.3f} V")
        antenna_biases.append(float(v_ant))

    # 4. Weighting potentials on the PIC grid. The self-field solver grounds the outer boundary and all
    # conductors, so Ramo-Shockley consistency requires Vw_a = 1 on antenna a and Vw_a = 0 on the other
    # conductors and on the outer boundary. The sampled SPIS fields violate this (thin conductors are
    # under-resolved and the SPIS domain is larger than the PIC box); a harmonic correction enforces the
    # Dirichlet values and leaves the Laplacian of the sampled field unchanged.
    conductor_nodes = occupied
    fixed_nodes = conductor_nodes.copy()
    fixed_nodes[[0, -1], :, :] = True
    fixed_nodes[:, [0, -1], :] = True
    fixed_nodes[:, :, [0, -1]] = True
    laplace = DirichletLaplaceSolver3D(fixed_nodes, (params.dx, params.dy, params.dz))

    # Vw_body only shapes the background potential (no Ramo signal is taken from the spacecraft), so it
    # keeps its sampled values on the outer boundary and is only made exact on the conductors
    Vw_body = laplace.enforce(Vw_body, np.where(conductor_nodes, spacecraft_mask_3d, Vw_body))
    Vw_grids = [laplace.enforce(Vw, mask.astype(np.float64)) for Vw, mask in zip(Vw_grids, antenna_masks_3d)]

    Ewx_list, Ewy_list, Ewz_list = [], [], []
    for Vw, mask in zip(Vw_grids, antenna_masks_3d):
        Ewx, Ewy, Ewz = np.gradient(-Vw, params.dx, params.dy, params.dz)
        Ewx[mask] = 0.0
        Ewy[mask] = 0.0
        Ewz[mask] = 0.0
        Ewx_list.append(Ewx)
        Ewy_list.append(Ewy)
        Ewz_list.append(Ewz)

    # 5. Background potential: every conductor at its potential, V_bg = V_sc on the spacecraft and the
    # reference bias on each antenna (FieldSolver3D adds only the deviation from this bias).
    conductor_potentials = np.zeros((Nx, Ny, Nz))
    conductor_potentials[spacecraft_mask_3d] = V_sc
    for mask, v_ant in zip(antenna_masks_3d, antenna_biases):
        conductor_potentials[mask] = v_ant

    if mesh_bg is not None:
        # The SPIS potential is kept everywhere. Only the conductor nodes of the PIC grid, where the
        # under-resolved conductors sample the surrounding plasma potential, are set to the SPIS
        # conductor potentials; the harmonic correction keeps the sampled space charge and the sampled
        # values on the outer boundary.
        V_bg_grid = laplace.enforce(V_bg_grid, np.where(conductor_nodes, conductor_potentials, V_bg_grid))
    else:
        # Superposition of the conductor potentials; exact on conductors and zero on the outer boundary
        V_bg_grid = V_sc * Vw_body
        for Vw, v_ant in zip(Vw_grids, antenna_biases):
            V_bg_grid = V_bg_grid + v_ant * Vw
        if abs(V_sc) > 1e-6 or any(abs(v) > 1e-6 for v in antenna_biases):
            print(f"  -> Soubor pozadí nezadán: Elektrostatické pozadí vygenerováno superpozicí tělesa sondy (V_sc = {V_sc:.3f} V) a {len(ant_files)} antén.")
        else:
            print(f"  -> Soubor pozadí nezadán a potenciály jsou 0.0 V. Počáteční elektrostatické pozadí je nastaveno na 0.0 V.")

    # 6. Impact intersection and surface normal (only for localized dust impact)
    is_homogeneous = getattr(params, 'plasma_injection_mode', 'point_cloud') == 'homogeneous'
    if not is_homogeneous:
        impact_pt, normal_vec = compute_impact_intersection_and_normal(
            params, Vw_body, spacecraft_mask_3d, mesh_body
        )
        impact_pos = list(impact_pt) if impact_pt is not None else None
        impact_normal = list(normal_vec) if normal_vec is not None else None
    else:
        impact_pos = None
        impact_normal = None

    # 7. Background electric field
    if np.any(V_bg_grid != 0.0):
        Ex_bg, Ey_bg, Ez_bg = np.gradient(-V_bg_grid, params.dx, params.dy, params.dz)
        Ex_bg[conductor_nodes] = 0.0
        Ey_bg[conductor_nodes] = 0.0
        Ez_bg[conductor_nodes] = 0.0
    else:
        Ex_bg = np.zeros((Nx, Ny, Nz))
        Ey_bg = np.zeros((Nx, Ny, Nz))
        Ez_bg = np.zeros((Nx, Ny, Nz))

    if antenna_biases:
        params.antenna_bias_voltage_V = antenna_biases
        params.V_bias = antenna_biases

    target_dtype = np.float32 if getattr(params, 'field_precision', 'float32') == 'float32' else np.float64
    V_bg_grid = np.ascontiguousarray(V_bg_grid, dtype=target_dtype)
    Ex_bg = np.ascontiguousarray(Ex_bg, dtype=target_dtype)
    Ey_bg = np.ascontiguousarray(Ey_bg, dtype=target_dtype)
    Ez_bg = np.ascontiguousarray(Ez_bg, dtype=target_dtype)
    Vw_grids = [np.ascontiguousarray(v, dtype=target_dtype) for v in Vw_grids]
    Ewx_list = [np.ascontiguousarray(e, dtype=target_dtype) for e in Ewx_list]
    Ewy_list = [np.ascontiguousarray(e, dtype=target_dtype) for e in Ewy_list]
    Ewz_list = [np.ascontiguousarray(e, dtype=target_dtype) for e in Ewz_list]

    return PreparedGeometry3D(
        V_bg=V_bg_grid,
        Vw_grids=Vw_grids,
        Ex_bg=Ex_bg,
        Ey_bg=Ey_bg,
        Ez_bg=Ez_bg,
        Ewx_list=Ewx_list,
        Ewy_list=Ewy_list,
        Ewz_list=Ewz_list,
        antenna_masks_3d=antenna_masks_3d,
        spacecraft_mask_3d=spacecraft_mask_3d,
        impact_pos=impact_pos,
        impact_normal=impact_normal,
        antenna_geometries=getattr(params, 'antenna_geometries', None),
        spacecraft_geometry=getattr(getattr(getattr(params, 'geometry', None), 'analytical', None), 'spacecraft', None)
    )
