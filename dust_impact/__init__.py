"""
dust_impact: Modular Kinetic PIC simulation package for dust impacts on spacecraft.
"""

from dust_impact import physics
from dust_impact import numerics
from dust_impact import common
from dust_impact import geometry
from dust_impact import sim3d
from dust_impact.main import main

from dust_impact.physics.constants import e, m_e, eps_0, amu, n_sw, Te_eV
from dust_impact.physics.charging import calculate_equilibrium_potential, ProbeMaterial, PlasmaEnvironment

__version__ = "0.3.0"
__all__ = [
    "main", "physics", "numerics", "common", "geometry", "sim3d",
    "e", "m_e", "eps_0", "amu", "n_sw", "Te_eV",
    "calculate_equilibrium_potential", "ProbeMaterial", "PlasmaEnvironment"
]
