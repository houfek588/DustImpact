"""
2D PIC Simulation module for dust impact on spacecraft.
"""

from dust_impact.sim2d.input_data import (
    SimulationParams2D,
    SimulationToggles2D,
    PlottingConfig,
    setup_simulation_parameters_2d,
)
from dust_impact.sim2d.sim_core import DustImpactSimulation2D
from dust_impact.sim2d.runner import run_2d_simulation

__all__ = [
    "SimulationParams2D",
    "SimulationToggles2D",
    "PlottingConfig",
    "setup_simulation_parameters_2d",
    "DustImpactSimulation2D",
    "run_2d_simulation",
]

try:
    from dust_impact.sim2d.plotting import plot_simulation_results_2d
    __all__.append("plot_simulation_results_2d")
except Exception:
    pass
