# -*- coding: utf-8 -*-
"""
Visualization and post-processing package for DustImpact simulation results.
Provides static plotting, 2D/3D animations, and ParaView VTK export.
"""

import os
import sys

# Ensure headless Agg backend on non-Windows headless environments (servers without X11/Wayland)
if sys.platform != "win32" and not os.environ.get("DISPLAY"):
    try:
        import matplotlib
        matplotlib.use("Agg")
    except Exception:
        pass

from dust_impact.visualization.signals import plot_currents_and_voltage
from dust_impact.visualization.fields import plot_static_vtk_fields, animate_fields_slice
from dust_impact.visualization.particles import animate_particles_2d_slice, animate_particles_3d
from dust_impact.visualization.distributions import animate_velocity_distribution
from dust_impact.visualization.pipeline import VisualizationPipeline, plot_simulation_results_3d
from dust_impact.visualization.vtk import (
    export_simulation_to_paraview,
    export_fields_to_vti,
    export_particles_to_vtp,
)

__all__ = [
    "VisualizationPipeline",
    "plot_simulation_results_3d",
    "plot_currents_and_voltage",
    "plot_static_vtk_fields",
    "animate_fields_slice",
    "animate_particles_2d_slice",
    "animate_particles_3d",
    "animate_velocity_distribution",
    "export_simulation_to_paraview",
    "export_fields_to_vti",
    "export_particles_to_vtp",
]
