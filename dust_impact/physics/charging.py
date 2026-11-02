#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
0D Equilibrium Charging Model in Space Plasma.
Calculates the floating potential of a spherical, sunlit probe from the current balance

  f(V) = Je(V) - Ji(V) - Jph(V) - Jse(V) = 0,

where f is the net electron current density collected by the probe (positive f charges the probe
negatively). All current densities are averaged over the total probe surface. Collection and
escape follow the orbit-motion-limited (OML) theory for a sphere:

  * Je  - thermal solar wind electrons, Boltzmann-repelled for V < 0, OML-attracted for V > 0;
  * Ji  - cold solar wind ion beam (v_sw >> v_th,i) hitting the projected area, OML beam factor
          1 - V / E_ram with the ram energy E_ram = m_i v_sw^2 / (2 e);
  * Jph - photoelectrons from the projected sunlit area, recollected for V > 0;
  * Jse - secondary electrons released by the collected electrons, with the yield averaged over
          the Maxwellian impact-energy distribution and recollected for V > 0.

The physical floating potential is a root of f with df/dV > 0 (a stable equilibrium).
"""

import math
from dataclasses import dataclass
from typing import Callable, List

import numpy as np

from dust_impact.physics.constants import e, m_e, eps_0, amu, n_sw, Te_eV

E_CHARGE: float = e
M_ELECTRON: float = m_e

# Ratio of projected (cross-sectional) to total surface area of a sphere, pi R^2 / (4 pi R^2)
SPHERE_PROJECTED_AREA_FRACTION: float = 0.25

# Gauss-Laguerre nodes and weights for flux-averaging over a Maxwellian energy distribution
_LAGUERRE_NODES, _LAGUERRE_WEIGHTS = np.polynomial.laguerre.laggauss(64)


@dataclass
class ProbeMaterial:
    """Parameters of spacecraft surface material."""
    name: str
    Jph0_uA_m2: float  # Photoemission saturation current density at normal incidence, 1 AU [uA/m^2]
    Tph_eV: float  # Photoelectron temperature in eV
    sey_delta_max: float  # Max secondary electron yield
    sey_Emax_eV: float  # Energy for max SEY in eV
    sey_s_shape: float  # Shape parameter of SEY curve
    sey_E_cutoff_eV: float  # Cutoff energy for SEE
    sey_T_eV: float = 2.0  # Temperature of the emitted secondary electrons in eV


@dataclass
class PlasmaEnvironment:
    """Parameters of plasma environment."""
    name: str
    n: float  # Plasma concentration (m^-3)
    v_sw: float  # Solar wind speed (m/s)
    Te_eV: float  # Electron temperature (eV)
    Ti_eV: float  # Ion temperature (eV); not used by the cold-beam ion current
    distance_au: float  # Distance from Sun in AU
    ion_mass_amu: float = 1.0  # Solar wind ion mass (protons) in amu


# Preset Materials
MAT_ALUMINIUM_ANTENNE = ProbeMaterial(
    name="Aluminium (Antenne)", Jph0_uA_m2=20.0, Tph_eV=2.0,
    sey_delta_max=1.0, sey_Emax_eV=300.0, sey_s_shape=1.35, sey_E_cutoff_eV=15.0
)

MAT_ALUMINIUM = ProbeMaterial(
    name="Aluminium (Structure)", Jph0_uA_m2=30.0, Tph_eV=2.0,
    sey_delta_max=2.5, sey_Emax_eV=300.0, sey_s_shape=1.35, sey_E_cutoff_eV=15.0
)

MAT_AL_OXIDE = ProbeMaterial(
    name="Aluminium Oxide (Al2O3)", Jph0_uA_m2=20.0, Tph_eV=2.0,
    sey_delta_max=3.0, sey_Emax_eV=350.0, sey_s_shape=1.4, sey_E_cutoff_eV=15.0
)

MAT_GOLD = ProbeMaterial(
    name="Gold", Jph0_uA_m2=29.0, Tph_eV=2.0,
    sey_delta_max=1.4, sey_Emax_eV=800.0, sey_s_shape=1.3, sey_E_cutoff_eV=15.0
)

# Preset Environments
ENV_EARTH = PlasmaEnvironment(name="Earth (1 AU)", n=n_sw, v_sw=400e3, Te_eV=Te_eV, Ti_eV=10.0, distance_au=1.0)
ENV_MARS = PlasmaEnvironment(name="Mars (1.52 AU)", n=2.2e6, v_sw=400e3, Te_eV=10.0, Ti_eV=8.0, distance_au=1.52)
ENV_JUPITER = PlasmaEnvironment(name="Jupiter (5.2 AU)", n=0.2e6, v_sw=400e3, Te_eV=7.0, Ti_eV=5.0, distance_au=5.2)


def electron_thermal_current_density(n: float, Te_eV: float) -> float:
    """Random thermal electron current density e n <v> / 4 of a Maxwellian [A/m^2]."""
    v_th_e = math.sqrt(8.0 * E_CHARGE * Te_eV / (math.pi * M_ELECTRON))
    return 0.25 * E_CHARGE * n * v_th_e


def oml_escape_fraction(V: float, T_eV: float) -> float:
    """
    Fraction of electrons emitted with a Maxwellian of temperature T_eV that escape a sphere at
    potential V: 1 for V <= 0, (1 + V/T) exp(-V/T) for V > 0 (OML, Grard 1973).
    """
    if V <= 0:
        return 1.0
    return (1.0 + V / T_eV) * math.exp(-V / T_eV)


def ram_energy_eV(env: PlasmaEnvironment) -> float:
    """Kinetic energy per charge of the solar wind ion beam, m_i v_sw^2 / (2 e) [eV]."""
    return 0.5 * env.ion_mass_amu * amu * env.v_sw**2 / E_CHARGE


def Je(V: float, env: PlasmaEnvironment) -> float:
    """Collected thermal electron current density (OML sphere) [A/m^2]."""
    J_e0 = electron_thermal_current_density(env.n, env.Te_eV)
    if V < 0:
        return J_e0 * math.exp(V / env.Te_eV)
    else:
        return J_e0 * (1.0 + V / env.Te_eV)


def Ji(V: float, env: PlasmaEnvironment,
       projected_area_fraction: float = SPHERE_PROJECTED_AREA_FRACTION) -> float:
    """
    Collected solar wind ion current density [A/m^2].

    The ions form a cold supersonic beam, so they are collected on the projected area with the
    OML beam factor 1 - V / E_ram (enhanced for V < 0, reflected completely for V >= E_ram).
    """
    J_i_ram = projected_area_fraction * E_CHARGE * env.n * env.v_sw
    return J_i_ram * max(0.0, 1.0 - V / ram_energy_eV(env))


def Jph(V: float, mat: ProbeMaterial, env: PlasmaEnvironment,
        projected_area_fraction: float = SPHERE_PROJECTED_AREA_FRACTION) -> float:
    """Escaping photoelectron current density from the projected sunlit area [A/m^2]."""
    J_ph0 = projected_area_fraction * (mat.Jph0_uA_m2 * 1e-6) / (env.distance_au ** 2)
    return J_ph0 * oml_escape_fraction(V, mat.Tph_eV)


def sey_delta_vaughan_like(E_eV: float, mat: ProbeMaterial) -> float:
    """Secondary electron yield for a primary electron of energy E_eV (Vaughan-like fit)."""
    if E_eV <= 0 or E_eV < mat.sey_E_cutoff_eV:
        return 0.0
    E_max = mat.sey_Emax_eV
    delta_max = mat.sey_delta_max
    x = E_eV / E_max
    if x <= 0:
        return 0.0
    elif x < 1.0:
        return delta_max * (x * math.exp(1.0 - x)) ** 0.62
    elif x <= 3.6:
        return delta_max * (x * math.exp(1.0 - x)) ** 0.35
    else:
        return delta_max * 1.125 / (x ** 0.35)


def electron_impact_energy_eV(V: float, env: PlasmaEnvironment) -> float:
    """Mean impact energy 2 Te (+ V for an attracting probe) of the collected electron flux [eV]."""
    E_kin = 2.0 * env.Te_eV
    if V > 0:
        E_kin += V
    return E_kin


def mean_secondary_yield(V: float, mat: ProbeMaterial, env: PlasmaEnvironment) -> float:
    """
    Secondary electron yield averaged over the collected electron flux.

    The flux of a Maxwellian through a surface has the energy distribution (E / Te^2) exp(-E / Te),
    which is unchanged at a repelling surface (V < 0) and shifted by V at an attracting one:

        <delta>(V) = int_0^inf delta(E + max(V, 0)) (E / Te^2) exp(-E / Te) dE,

    The yield vanishes below the cutoff energy, so the integral starts at E_0 = max(E_cut - V, 0)
    and is evaluated with Gauss-Laguerre quadrature after the substitution E = E_0 + Te x, which
    keeps the discontinuity at the cutoff out of the quadrature.
    """
    T = env.Te_eV
    shift = max(V, 0.0)
    E_0 = max(mat.sey_E_cutoff_eV - shift, 0.0)
    E = E_0 + T * _LAGUERRE_NODES
    yields = np.array([sey_delta_vaughan_like(E_k + shift, mat) for E_k in E])
    return float(math.exp(-E_0 / T) * np.sum(_LAGUERRE_WEIGHTS * yields * E / T))


def Jse(V: float, mat: ProbeMaterial, env: PlasmaEnvironment) -> float:
    """Escaping secondary electron current density [A/m^2]."""
    return mean_secondary_yield(V, mat, env) * Je(V, env) * oml_escape_fraction(V, mat.sey_T_eV)


def f_balance(V: float, env: PlasmaEnvironment, mat: ProbeMaterial,
              projected_area_fraction: float = SPHERE_PROJECTED_AREA_FRACTION) -> float:
    """Net collected electron current density Je - Ji - Jph - Jse at probe potential V [A/m^2]."""
    return (
        Je(V, env)
        - Ji(V, env, projected_area_fraction)
        - Jph(V, mat, env, projected_area_fraction)
        - Jse(V, mat, env)
    )


def find_root_bisection(func: Callable[[float], float], a: float = -50.0, b: float = 50.0,
                        tol: float = 1e-5, max_iter: int = 200) -> float:
    """
    Finds a root of func in the bracket [a, b] by bisection.

    Parameters:
    -----------
    func : Callable[[float], float]
        Continuous function with func(a) * func(b) <= 0.
    a, b : float
        Bracket end points.
    tol : float
        Absolute tolerance on the root position.
    max_iter : int
        Maximum number of bisection steps.

    Returns:
    --------
    root : float

    Raises:
    -------
    ValueError
        If func does not change sign on [a, b].
    """
    fa, fb = func(a), func(b)
    if fa == 0.0:
        return float(a)
    if fb == 0.0:
        return float(b)
    if fa * fb > 0:
        raise ValueError(f"Root is not bracketed: f({a}) = {fa:.3e}, f({b}) = {fb:.3e}")

    for _ in range(max_iter):
        c = 0.5 * (a + b)
        fc = func(c)
        if fc == 0.0 or 0.5 * (b - a) < tol:
            return float(c)
        if fa * fc < 0:
            b, fb = c, fc
        else:
            a, fa = c, fc
    return float(0.5 * (a + b))


def find_stable_floating_potentials(env: PlasmaEnvironment, mat: ProbeMaterial,
                                    V_min: float = -1000.0, V_max: float = 1000.0,
                                    step: float = 0.25, tol: float = 1e-9,
                                    projected_area_fraction: float = SPHERE_PROJECTED_AREA_FRACTION
                                    ) -> List[float]:
    """
    Returns all stable floating potentials (roots of f with f changing from - to +) in [V_min, V_max].

    The interval is scanned on a uniform grid and every sign change is refined by bisection.
    Several stable roots can exist when secondary emission dominates (hysteretic charging).
    """
    func = lambda V: f_balance(V, env, mat, projected_area_fraction)
    grid = np.arange(V_min, V_max + step, step)
    values = np.array([func(V) for V in grid])
    up = np.flatnonzero((values[:-1] < 0.0) & (values[1:] >= 0.0))
    return [find_root_bisection(func, grid[k], grid[k + 1], tol=tol) for k in up]


def calculate_equilibrium_potential(env: PlasmaEnvironment = ENV_EARTH,
                                     mat: ProbeMaterial = MAT_ALUMINIUM_ANTENNE,
                                     verbose: bool = True,
                                     projected_area_fraction: float = SPHERE_PROJECTED_AREA_FRACTION
                                     ) -> float:
    """
    Floating potential of a sunlit spherical probe from the 0D current balance.

    If several stable equilibria exist, the one closest to 0 V is returned.

    Raises:
    -------
    ValueError
        If the current balance has no stable root in [-1000, 1000] V.
    """
    roots = find_stable_floating_potentials(env, mat, projected_area_fraction=projected_area_fraction)
    if not roots:
        raise ValueError(f"No stable floating potential found for {mat.name} in {env.name}")
    V_eq = min(roots, key=abs)
    if verbose:
        print(f"=== ROVNOVÁŽNÝ POTENCIÁL PORUCHY PRO: {mat.name} ({env.name}) ===")
        print(f"  Plovoucí potenciál V_eq = {V_eq:.3f} V\n")
        if len(roots) > 1:
            print(f"  Varování: více stabilních rovnováh {[round(r, 3) for r in roots]} V\n")
    return V_eq
