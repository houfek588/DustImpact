# -*- coding: utf-8 -*-
"""
SPIS VTK mesh reader and interpolator for 3D PIC simulations.
Handles VTK unstructued/polygonal grids, sampling onto Cartesian ImageData grids,
and conductor boundary extraction via PyVista.
"""

import os
from typing import Tuple, List, Optional, Any, Dict
import numpy as np

try:
    import pyvista as pv
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False

from dust_impact.geometry.voxelizer import detect_metal_mask_3d


def read_spis_mesh(filepath: str):
    """
    Reads a VTK/VTP/VTU mesh file from disk, checking working directory and 'inputs/' subfolder.

    Parameters
    ----------
    filepath : str
        Relative or absolute path to mesh file.

    Returns
    -------
    pyvista.DataSet
        Loaded PyVista mesh object.
    """
    if not HAS_PYVISTA:
        raise ImportError("PyVista není nainstalováno. Pro načítání SPIS VTK sítí nainstalujte 'pyvista'.")

    if os.path.exists(filepath):
        return pv.read(filepath)
    alt_path = os.path.join("inputs", filepath)
    if os.path.exists(alt_path):
        return pv.read(alt_path)
    return pv.read(filepath)


def extract_potential_from_mesh(sampled_mesh: Any, filename: str = "") -> np.ndarray:
    """
    Extracts the electric potential array from a PyVista mesh by matching common SPIS field names.

    Parameters
    ----------
    sampled_mesh : pyvista.DataSet
        Mesh with point data attributes.
    filename : str
        Source filename for error diagnostics.

    Returns
    -------
    np.ndarray
        1D array of potential values at mesh points.
    """
    possible_names = [
        'Potential', 'Potencial', 'V', 'PlasmaPotential', 'U',
        'plasma pot at t = 1.0 s'
    ]
    available_keys = list(sampled_mesh.point_data.keys())

    for name in possible_names:
        if name in available_keys:
            return sampled_mesh.point_data[name]

    raise KeyError(
        f"\n[CHYBA] V souboru '{filename}' se nepodařilo najít pole s potenciálem.\n"
        f"Dostupná datová pole v tomto souboru jsou: {available_keys}\n"
    )


def sample_mesh_to_grid(mesh: Any, pic_grid: Any, filename: str = "") -> np.ndarray:
    """
    Samples a PyVista mesh onto a Cartesian grid (ImageData) and extracts potential values.

    Parameters
    ----------
    mesh : pyvista.DataSet
        Source mesh.
    pic_grid : pyvista.ImageData
        Target Cartesian grid.
    filename : str
        Source filename for diagnostics.

    Returns
    -------
    np.ndarray
        Array of shape matching pic_grid dimensions.
    """
    sampled = pic_grid.sample(mesh)
    pot = extract_potential_from_mesh(sampled, filename)
    return pot.reshape(pic_grid.dimensions)


def extract_enclosed_conductor_mask(
    mesh: Any,
    pic_grid: Any,
    field_grid: np.ndarray,
    dx_min: float,
    threshold: float = 0.85
) -> np.ndarray:
    """
    Extracts a 3D boolean conductor mask using PyVista contour surface and select_enclosed_points,
    with gradient voxelizer fallback.

    Parameters
    ----------
    mesh : pyvista.DataSet
        Surface or volume mesh of conductor.
    pic_grid : pyvista.ImageData
        Cartesian grid.
    field_grid : np.ndarray
        Reshaped 3D scalar potential grid.
    dx_min : float
        Minimum grid spacing.
    threshold : float
        Contour fraction of peak value (default 0.85).

    Returns
    -------
    np.ndarray (bool)
        3D boolean mask of conductor volume.
    """
    dims = pic_grid.dimensions
    if mesh is not None:
        # 1. Direct enclosure check for surface meshes
        try:
            if not bool(mesh.point_data.keys()):
                enclosed = pic_grid.select_enclosed_points(mesh, tolerance=1e-5)
                mask = enclosed['SelectedPoints'].view(bool).reshape(dims)
                if np.sum(mask) > 0:
                    return mask
        except Exception:
            pass

        # 2. Iso-surface contour extraction from volumetric potential field
        try:
            body_keys = list(mesh.point_data.keys())
            key_body = body_keys[0] if body_keys else 'Potential'
            raw_peak = mesh.point_data[key_body].ravel()[np.nanargmax(np.abs(mesh.point_data[key_body]))]
            contour_val = threshold * raw_peak if abs(raw_peak) > 1e-6 else threshold
            contour_body = mesh.contour([contour_val], scalars=key_body)
            enclosed_body = pic_grid.select_enclosed_points(contour_body, tolerance=1e-5)
            mask = enclosed_body['SelectedPoints'].view(bool).reshape(dims)
            if np.sum(mask) > 0:
                return mask
        except Exception:
            pass

    return detect_metal_mask_3d(field_grid, dx_min, threshold=threshold)


def extract_antenna_bias_from_spis(
    mesh_bg: Any,
    ant_mesh_file: str,
    default_bias: float = 0.0
) -> float:
    """
    Extracts equilibrium potential of an antenna directly by averaging background potential
    over points where antenna weighting field is high (~0.8 of peak).

    Parameters
    ----------
    mesh_bg : pyvista.DataSet
        SPIS background potential mesh.
    ant_mesh_file : str
        Path to antenna weighting field VTK.
    default_bias : float
        Fallback bias if extraction fails.

    Returns
    -------
    float
        Equilibrium floating potential of the antenna.
    """
    if mesh_bg is None or not HAS_PYVISTA:
        return default_bias

    try:
        mesh_w = read_spis_mesh(ant_mesh_file)
        w_keys = list(mesh_w.point_data.keys())
        key_w = w_keys[0] if w_keys else 'Potential'

        vw_pts = mesh_w.point_data[key_w]
        v_max_w = np.nanmax(vw_pts)

        bg_keys = list(mesh_bg.point_data.keys())
        key_bg = bg_keys[0] if bg_keys else 'Potential'

        if v_max_w > 0:
            surf_pts_idx = np.where(vw_pts >= 0.8 * v_max_w)[0]
            if len(surf_pts_idx) > 0 and key_bg in mesh_bg.point_data:
                v_bg_surf = mesh_bg.point_data[key_bg][surf_pts_idx]
                return float(np.nanmean(v_bg_surf))
    except Exception:
        pass

    return default_bias
