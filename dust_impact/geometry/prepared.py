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
from dust_impact.geometry.spis_loader import (
    read_spis_mesh,
    extract_potential_from_mesh,
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
    impact_pos: List[float] = field(default_factory=lambda: [-2.0, 2.0, 0.0])
    impact_normal: List[float] = field(default_factory=lambda: [-0.7071, 0.7071, 0.0])

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
        return (
            f"PreparedGeometry3D: Grid {self.grid_shape}, Antennas: {self.num_antennas}, "
            f"Spacecraft cells: {sc_cells} ({sc_frac:.2f}%), "
            f"Impact pos: {self.impact_pos}, Normal: {self.impact_normal}"
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
            sampled_body = pic_grid.sample(sc_mesh)
            pot_body = extract_potential_from_mesh(sampled_body, sc_file)
            Vw_body = pot_body.reshape((Nx, Ny, Nz))

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
    Ewx_list, Ewy_list, Ewz_list = [], [], []
    antenna_masks_3d = []

    X_mat, Y_mat, Z_mat = np.meshgrid(params.x_grid, params.y_grid, params.z_grid, indexing='ij')

    for i, vtk_file in enumerate(ant_files):
        mesh_ant = None
        try:
            mesh_ant = read_spis_mesh(vtk_file)
            sampled_ant = pic_grid.sample(mesh_ant)
            pot_ant = extract_potential_from_mesh(sampled_ant, vtk_file)
            Vw = pot_ant.reshape((Nx, Ny, Nz))

            # Normalize weighting field to 1.0 on conductor surface
            vw_ant_max = np.nanmax(np.abs(Vw))
            if vw_ant_max > 1e-6 and abs(vw_ant_max - 1.0) > 0.05:
                peak = Vw.ravel()[np.nanargmax(np.abs(Vw))]
                Vw = Vw / peak
        except (FileNotFoundError, KeyError, Exception) as e:
            print(f"[UPOZORNĚNÍ] VTK antény {i+1} nenalezen ({e}). Používám syntetické pole.")
            Vw = np.zeros((Nx, Ny, Nz))

        threshold = sc_threshold
        if hasattr(params, 'antenna_weighting_threshold') and params.antenna_weighting_threshold is not None:
            threshold = params.antenna_weighting_threshold
        elif hasattr(params, 'weighting_threshold') and params.weighting_threshold is not None:
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

        Vw[mask] = 1.0
        Vw_grids.append(Vw)

        Ewx, Ewy, Ewz = np.gradient(-Vw, params.dx, params.dy, params.dz)
        Ewx[mask] = 0.0
        Ewy[mask] = 0.0
        Ewz[mask] = 0.0

        antenna_masks_3d.append(mask)
        Ewx_list.append(Ewx)
        Ewy_list.append(Ewy)
        Ewz_list.append(Ewz)

    # 3. Load or synthesize background field via superposition (spacecraft + antennas)
    mesh_bg = None
    V_bg_grid = np.zeros((Nx, Ny, Nz))
    V_sc = getattr(params, 'spacecraft_voltage_V', params.Vf)
    if V_sc is None:
        V_sc = params.Vf

    if bg_file and (os.path.exists(bg_file) or os.path.exists(os.path.join("inputs", bg_file))):
        try:
            mesh_bg = read_spis_mesh(bg_file)
            sampled_bg = pic_grid.sample(mesh_bg)
            pot_data = extract_potential_from_mesh(sampled_bg, bg_file)
            V_bg_grid = pot_data.reshape((Nx, Ny, Nz))
            V_bg_grid[spacecraft_mask_3d] = float(V_sc)
            print(f"  -> Pozadí načteno z {bg_file}. Rozsah potenciálu: {np.nanmin(V_bg_grid):.2f} až {np.nanmax(V_bg_grid):.2f} V")
        except Exception as e:
            print(f"[UPOZORNĚNÍ] Nelze načíst pozadí z '{bg_file}' ({e}). Nastavuji počáteční pozadí na 0.0 V.")
            mesh_bg = None
            V_bg_grid = np.zeros((Nx, Ny, Nz))
    else:
        # Spacecraft body contribution
        if abs(V_sc) > 1e-6 and np.any(Vw_body != 0.0):
            V_bg_grid += float(V_sc) * Vw_body
        elif abs(V_sc) > 1e-6:
            r_mat = np.sqrt(X_mat**2 + Y_mat**2 + Z_mat**2)
            V_bg_grid += float(V_sc) * np.exp(-r_mat / params.debye_length)

        # Antenna weighting fields contribution
        configured_biases = getattr(params, 'antenna_bias_voltage_V', getattr(params, 'V_bias', []))
        added_antennas = 0
        for i, Vw_ant in enumerate(Vw_grids):
            v_ant = configured_biases[i] if i < len(configured_biases) else getattr(params, 'Vf_antenne', 0.0)
            if abs(v_ant) > 1e-6 and np.any(Vw_ant != 0.0):
                V_bg_grid += float(v_ant) * Vw_ant
                added_antennas += 1

        V_bg_grid[spacecraft_mask_3d] = float(V_sc)
        for i in range(len(antenna_masks_3d)):
            v_ant = configured_biases[i] if i < len(configured_biases) else getattr(params, 'Vf_antenne', 0.0)
            V_bg_grid[antenna_masks_3d[i]] = float(v_ant)

        if abs(V_sc) > 1e-6 or added_antennas > 0:
            print(f"  -> Soubor pozadí nezadán: Elektrostatické pozadí vygenerováno superpozicí tělesa sondy (V_sc = {V_sc:.3f} V) a {len(ant_files)} antén.")
        else:
            print(f"  -> Soubor pozadí nezadán a potenciály jsou 0.0 V. Počáteční elektrostatické pozadí je nastaveno na 0.0 V.")

    # 4. Impact intersection and surface normal
    impact_pt, normal_vec = compute_impact_intersection_and_normal(
        params, Vw_body, spacecraft_mask_3d, mesh_body
    )

    # 5. Background electric field
    if np.any(V_bg_grid != 0.0):
        Ex_bg, Ey_bg, Ez_bg = np.gradient(-V_bg_grid, params.dx, params.dy, params.dz)
        Ex_bg[spacecraft_mask_3d] = 0.0
        Ey_bg[spacecraft_mask_3d] = 0.0
        Ez_bg[spacecraft_mask_3d] = 0.0
        for mask in antenna_masks_3d:
            Ex_bg[mask] = 0.0
            Ey_bg[mask] = 0.0
            Ez_bg[mask] = 0.0
    else:
        Ex_bg = np.zeros((Nx, Ny, Nz))
        Ey_bg = np.zeros((Nx, Ny, Nz))
        Ez_bg = np.zeros((Nx, Ny, Nz))

    # 6. Extract antenna equilibrium potentials
    spis_v_bias = []
    for i, vtk_file in enumerate(ant_files):
        extracted = False
        if mesh_bg is not None:
            v_ant_exact = extract_antenna_bias_from_spis(mesh_bg, vtk_file, default_bias=None)
            if v_ant_exact is not None:
                spis_v_bias.append(v_ant_exact)
                print(f"  -> Anténa {i + 1}: Rovnovážný potenciál načten přímo ze SPIS VTK: {v_ant_exact:.3f} V")
                extracted = True

        if not extracted:
            if i < len(Vw_grids) and np.any(antenna_masks_3d[i]) and np.any(V_bg_grid != 0.0):
                v_ant_val = float(np.nanmax(V_bg_grid[antenna_masks_3d[i]]))
                spis_v_bias.append(v_ant_val)
                print(f"  -> Anténa {i + 1}: Rovnovážný potenciál načten z mřížky: {v_ant_val:.3f} V")
            elif i < len(params.antenna_bias_voltage_V) and params.antenna_bias_voltage_V[i] != 0.0:
                spis_v_bias.append(params.antenna_bias_voltage_V[i])
                print(f"  -> Anténa {i + 1}: Používám zadaný potenciál z parametrů: {params.antenna_bias_voltage_V[i]:.3f} V")
            else:
                v_f_ant = getattr(params, 'Vf_antenne', 0.0)
                spis_v_bias.append(v_f_ant)
                print(f"  -> Anténa {i + 1}: Používám plovoucí potenciál: {v_f_ant:.3f} V")

    if spis_v_bias:
        params.antenna_bias_voltage_V = spis_v_bias
        params.V_bias = spis_v_bias

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
        impact_pos=list(impact_pt),
        impact_normal=list(normal_vec)
    )
