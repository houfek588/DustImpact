"""
Physics subpackage for DustImpact: Physical constants, charging balance, Ramo-Shockley, emissions.
"""

from dust_impact.physics.constants import e, m_e, eps_0, amu, n_sw, Te_eV
from dust_impact.physics.charging import (
    calculate_equilibrium_potential,
    ProbeMaterial,
    PlasmaEnvironment,
    MAT_ALUMINIUM,
    MAT_ALUMINIUM_ANTENNE,
    ENV_EARTH,
)
from dust_impact.physics.ramo_shockley import calc_induced_current, calc_induced_current_from_potential
from dust_impact.physics.emissions import (
    generate_half_maxwellian_velocities,
    maxwellian_sigma,
    sample_maxwellian_velocities,
    sample_half_maxwellian_velocities,
)

__all__ = [
    "e", "m_e", "eps_0", "amu", "n_sw", "Te_eV",
    "calculate_equilibrium_potential", "ProbeMaterial", "PlasmaEnvironment",
    "MAT_ALUMINIUM", "MAT_ALUMINIUM_ANTENNE", "ENV_EARTH",
    "calc_induced_current", "calc_induced_current_from_potential", "generate_half_maxwellian_velocities",
    "maxwellian_sigma", "sample_maxwellian_velocities", "sample_half_maxwellian_velocities",
]
