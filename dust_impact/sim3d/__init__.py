"""
3D PIC Simulation module for dust impact on spacecraft.
"""

from dust_impact.sim3d.config_loader import (
    SimulationParams3D,
    SimulationToggles3D,
    PlottingConfig3D,
    VTKFilesConfig,
    setup_simulation_parameters_3d,
)
from dust_impact.sim3d.sim_core import DustImpactSimulation3D
from dust_impact.sim3d.vtk_reader import load_and_interpolate_vtk
from dust_impact.sim3d.runner import run_3d_simulation

__all__ = [
    "SimulationParams3D",
    "SimulationToggles3D",
    "PlottingConfig3D",
    "VTKFilesConfig",
    "setup_simulation_parameters_3d",
    "DustImpactSimulation3D",
    "load_and_interpolate_vtk",
    "run_3d_simulation",
]

try:
    from dust_impact.sim3d.plotting import plot_simulation_results_3d
    __all__.append("plot_simulation_results_3d")
except Exception:
    pass
