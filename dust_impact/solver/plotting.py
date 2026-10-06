# -*- coding: utf-8 -*-
"""
Backward-compatibility proxy for dust_impact.visualization.
Direct all new code imports to `dust_impact.visualization`.
"""

from dust_impact.common.io import export_time_series_csv as _export_csv
from dust_impact.visualization import (
    VisualizationPipeline,
    plot_simulation_results_3d,
    plot_currents_and_voltage,
    plot_static_vtk_fields,
    animate_fields_slice,
    animate_particles_2d_slice,
    animate_particles_3d,
    animate_velocity_distribution,
)

# Aliases for private legacy functions
_plot_currents_and_voltage = plot_currents_and_voltage
_plot_static_vtk_fields = plot_static_vtk_fields
_animate_yz_fields_slice = animate_fields_slice
_animate_yz_particles_slice = animate_particles_2d_slice
_animate_3d_particles = animate_particles_3d
_animate_velocity_distribution = animate_velocity_distribution

__all__ = [
    "VisualizationPipeline",
    "plot_simulation_results_3d",
    "plot_currents_and_voltage",
    "plot_static_vtk_fields",
    "animate_fields_slice",
    "animate_particles_2d_slice",
    "animate_particles_3d",
    "animate_velocity_distribution",
    "_plot_currents_and_voltage",
    "_plot_static_vtk_fields",
    "_animate_yz_fields_slice",
    "_animate_yz_particles_slice",
    "_animate_3d_particles",
    "_animate_velocity_distribution",
    "_export_csv",
]
