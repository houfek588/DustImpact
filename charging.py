#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backward compatibility wrapper for charging.py.
Delegates to dust_impact.charging.
"""

from dust_impact.charging import (
    ProbeMaterial,
    PlasmaEnvironment,
    MAT_ALUMINIUM_ANTENNE,
    MAT_ALUMINIUM,
    MAT_AL_OXIDE,
    MAT_GOLD,
    ENV_EARTH,
    ENV_MARS,
    ENV_JUPITER,
    electron_thermal_current_density,
    Je,
    Ji,
    Jph,
    sey_delta_vaughan_like,
    electron_impact_energy_eV,
    Jse,
    f_balance,
    find_root_bisection,
    calculate_equilibrium_potential,
)

if __name__ == "__main__":
    calculate_equilibrium_potential(ENV_EARTH, MAT_ALUMINIUM_ANTENNE)
    calculate_equilibrium_potential(ENV_EARTH, MAT_ALUMINIUM)