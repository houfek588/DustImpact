# -*- coding: utf-8 -*-
"""
Backward-compatibility proxy for dust_impact.visualization.vtk.
Direct all new code imports to `dust_impact.visualization.vtk`.
"""

from dust_impact.visualization.vtk import (
    HAS_PYVISTA,
    _write_pvd_file,
    export_fields_to_vti,
    export_particles_to_vtp,
    export_simulation_to_paraview,
)

__all__ = [
    "HAS_PYVISTA",
    "_write_pvd_file",
    "export_fields_to_vti",
    "export_particles_to_vtp",
    "export_simulation_to_paraview",
]
