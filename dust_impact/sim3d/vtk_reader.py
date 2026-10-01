# -*- coding: utf-8 -*-
"""
VTK Data loader and SPIS mesh interpolator for 3D PIC simulation.
Backward compatibility bridge delegating to the dust_impact.geometry package.
"""

from dust_impact.geometry import (
    PreparedGeometry3D,
    build_simulation_geometry,
    build_simulation_geometry as load_and_interpolate_vtk,
    extract_potential_from_mesh as _extract_potential_from_mesh,
    interpolate_field_3d as _interpolate_field_3d,
    compute_impact_intersection_and_normal as _compute_impact_intersection_and_normal,
    generate_synthetic_analytical_fields as _generate_synthetic_fields_3d,
    HAS_PYVISTA,
)

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
