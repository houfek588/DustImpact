# -*- coding: utf-8 -*-
"""
Geometry subpackage for the 3D Dust Impact PIC simulation.
Handles:
- Spacecraft geometry processing (PyVista VTK mesh loading, surface extraction).
- Voxelization (gradient metal detection, 3D binary masks for spacecraft and antennas).
- Analytical geometry definitions (exact potentials and electric fields for spheres, boxes, cylinders).
- Surface ray-tracing, dust impact point intersection, and outward normal calculation.
- Geometry preparation and packaging into PreparedGeometry3D for the solver.
"""

from dust_impact.geometry.voxelizer import (
    detect_metal_mask_3d,
    voxelize_sphere,
    voxelize_box,
    voxelize_cylinder
)

from dust_impact.geometry.surface import (
    interpolate_field_3d,
    ray_trace_mesh,
    ray_march_voxel_grid,
    compute_surface_normal_from_potential,
    compute_impact_intersection_and_normal,
    intersect_ray_sphere,
    intersect_ray_box,
    intersect_ray_cylinder,
    intersect_ray_analytical_spacecraft
)

from dust_impact.geometry.analytical import (
    AnalyticalConductor,
    AnalyticalSphere,
    AnalyticalBox,
    generate_synthetic_analytical_fields,
    voxelize_analytical_spacecraft_part,
    solve_laplace_dirichlet_3d,
    build_analytical_simulation_geometry
)

from dust_impact.geometry.spis_loader import (
    read_spis_mesh,
    extract_potential_from_mesh,
    sample_mesh_to_grid,
    extract_enclosed_conductor_mask,
    extract_antenna_bias_from_spis,
    HAS_PYVISTA
)

from dust_impact.geometry.prepared import (
    PreparedGeometry3D,
    build_simulation_geometry
)

__all__ = [
    # Container & Builder
    "PreparedGeometry3D",
    "build_simulation_geometry",
    "build_analytical_simulation_geometry",

    # Voxelizer
    "detect_metal_mask_3d",
    "voxelize_sphere",
    "voxelize_box",
    "voxelize_cylinder",
    "voxelize_analytical_spacecraft_part",

    # Surface & Ray Tracing
    "interpolate_field_3d",
    "ray_trace_mesh",
    "ray_march_voxel_grid",
    "compute_surface_normal_from_potential",
    "compute_impact_intersection_and_normal",
    "intersect_ray_sphere",
    "intersect_ray_box",
    "intersect_ray_cylinder",
    "intersect_ray_analytical_spacecraft",

    # Analytical
    "AnalyticalConductor",
    "AnalyticalSphere",
    "AnalyticalBox",
    "generate_synthetic_analytical_fields",
    "solve_laplace_dirichlet_3d",

    # SPIS Loader
    "read_spis_mesh",
    "extract_potential_from_mesh",
    "sample_mesh_to_grid",
    "extract_enclosed_conductor_mask",
    "extract_antenna_bias_from_spis",
    "HAS_PYVISTA",
]
