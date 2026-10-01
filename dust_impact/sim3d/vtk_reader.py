# -*- coding: utf-8 -*-
"""
VTK Data loader and SPIS mesh interpolator for 3D PIC simulation.
Compatibility bridge delegating to the standalone dust_impact.geometry package.
"""

from typing import Tuple, List, Optional, Any
import numpy as np

from dust_impact.sim3d.config_loader import SimulationParams3D
from dust_impact.geometry import (
    PreparedGeometry3D,
    build_simulation_geometry,
    interpolate_field_3d as _interpolate_field_3d_geom,
    compute_impact_intersection_and_normal as _compute_impact_intersection_geom,
    generate_synthetic_analytical_fields,
    extract_potential_from_mesh as _extract_potential_from_mesh_geom,
    HAS_PYVISTA
)


def _extract_potential_from_mesh(sampled_mesh, filename: str) -> np.ndarray:
    return _extract_potential_from_mesh_geom(sampled_mesh, filename)


def _interpolate_field_3d(x, y, z, field_x, field_y, field_z, params: SimulationParams3D):
    return _interpolate_field_3d_geom(
        x, y, z, field_x, field_y, field_z,
        params.x_grid, params.y_grid, params.z_grid,
        params.dx, params.dy, params.dz
    )


def _compute_impact_intersection_and_normal(
    params: SimulationParams3D, Vw_body: np.ndarray, spacecraft_mask_3d: np.ndarray, mesh_body=None
):
    return _compute_impact_intersection_geom(params, Vw_body, spacecraft_mask_3d, mesh_body)


def _generate_synthetic_fields_3d(params: SimulationParams3D):
    return generate_synthetic_analytical_fields(params)


def load_and_interpolate_vtk(params: SimulationParams3D) -> PreparedGeometry3D:
    """
    Loads SPIS VTK meshes and constructs Cartesian simulation fields and boundary masks.
    Delegates to dust_impact.geometry.build_simulation_geometry.

    Returns a PreparedGeometry3D object that seamlessly unpacks as a 10-tuple:
    (V_bg, Vw_grids, Ex_bg, Ey_bg, Ez_bg, Ewx, Ewy, Ewz, ant_masks, sc_mask)
    """
    return build_simulation_geometry(params)


__all__ = [
    "load_and_interpolate_vtk",
    "PreparedGeometry3D",
    "build_simulation_geometry",
    "_extract_potential_from_mesh",
    "_interpolate_field_3d",
    "_compute_impact_intersection_and_normal",
    "_generate_synthetic_fields_3d",
    "HAS_PYVISTA",
]
