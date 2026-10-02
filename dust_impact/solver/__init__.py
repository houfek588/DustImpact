# -*- coding: utf-8 -*-
"""
Canonical 3D PIC Simulation solver package for dust impact on spacecraft.
"""

from dust_impact.geometry import build_simulation_geometry as load_and_interpolate_vtk
from dust_impact.solver.config_loader import (
    SimulationParams3D,
    SimulationToggles3D,
    PlottingConfig3D,
    VTKFilesConfig,
    setup_simulation_parameters_3d,
    load_and_validate_config,
)
from dust_impact.solver.sim_core import (
    DustImpactSimulation3D,
    ParticleEnsemble,
    FieldSolver3D,
    AntennaCircuitCollector,
)
from dust_impact.solver.runner import run_3d_simulation

__all__ = [
    "SimulationParams3D",
    "SimulationToggles3D",
    "PlottingConfig3D",
    "VTKFilesConfig",
    "setup_simulation_parameters_3d",
    "load_and_validate_config",
    "DustImpactSimulation3D",
    "ParticleEnsemble",
    "FieldSolver3D",
    "AntennaCircuitCollector",
    "load_and_interpolate_vtk",
    "run_3d_simulation",
]

try:
    from dust_impact.solver.plotting import plot_simulation_results_3d
    __all__.append("plot_simulation_results_3d")
except Exception:
    pass
