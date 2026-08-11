#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
0D Equilibrium Charging Model in Space Plasma.
Calculates probe floating potential from current balance:
  Je(V) - Ji(V) - Jph(V) - Jse(V) = 0
"""

import math
from dataclasses import dataclass
from dust_impact.physics.constants import e, m_e, eps_0, amu, n_sw, Te_eV

E_CHARGE: float = e
M_ELECTRON: float = m_e


@dataclass
class ProbeMaterial:
    """Parameters of spacecraft surface material."""
    name: str
    Jph0_uA_m2: float  # Photoemission in µA/m^2 at 1 AU
    Tph_eV: float  # Photoelectron temperature in eV
    sey_delta_max: float  # Max secondary electron yield
    sey_Emax_eV: float  # Energy for max SEY in eV
    sey_s_shape: float  # Shape parameter of SEY curve
    sey_E_cutoff_eV: float  # Cutoff energy for SEE


@dataclass
class PlasmaEnvironment:
    """Parameters of plasma environment."""
    name: str
    n: float  # Plasma concentration (m^-3)
    v_sw: float  # Solar wind speed (m/s)
    Te_eV: float  # Electron temperature (eV)
    Ti_eV: float  # Ion temperature (eV)
    distance_au: float  # Distance from Sun in AU


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
    v_th_e = math.sqrt(8.0 * E_CHARGE * Te_eV / (math.pi * M_ELECTRON))
    return 0.25 * E_CHARGE * n * v_th_e


def Je(V: float, env: PlasmaEnvironment) -> float:
    J_e0 = electron_thermal_current_density(env.n, env.Te_eV)
    if V < 0:
        return J_e0 * math.exp(V / env.Te_eV)
    else:
        return J_e0 * (1.0 + V / env.Te_eV)


def Ji(V: float, env: PlasmaEnvironment) -> float:
    J_i_ram = E_CHARGE * env.n * env.v_sw
    if V > 0:
        val = 1.0 - V / env.Ti_eV
        return J_i_ram * max(0.0, val)
    else:
        return J_i_ram * (1.0 - V / env.Ti_eV)


def Jph(V: float, mat: ProbeMaterial, env: PlasmaEnvironment) -> float:
    J_ph0 = (mat.Jph0_uA_m2 * 1e-6) / (env.distance_au ** 2)
    if V <= 0:
        return J_ph0
    else:
        return J_ph0 * math.exp(-V / mat.Tph_eV)


def sey_delta_vaughan_like(E_eV: float, mat: ProbeMaterial) -> float:
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
    E_kin = 2.0 * env.Te_eV
    if V > 0:
        E_kin += V
    return E_kin


def Jse(V: float, mat: ProbeMaterial, env: PlasmaEnvironment) -> float:
    E_imp = electron_impact_energy_eV(V, env)
    delta_se = sey_delta_vaughan_like(E_imp, mat)
    J_e_val = Je(V, env)
    return delta_se * J_e_val


def f_balance(V: float, env: PlasmaEnvironment, mat: ProbeMaterial) -> float:
    return Je(V, env) - Ji(V, env) - Jph(V, mat, env) - Jse(V, mat, env)


def find_root_bisection(func, a: float = -50.0, b: float = 50.0, tol: float = 1e-5, max_iter: int = 100) -> float:
    fa, fb = func(a), func(b)
    if fa * fb > 0:
        for offset in [10.0, 20.0, 50.0]:
            if func(a - offset) * fb <= 0:
                a -= offset
                fa = func(a)
                break
            if fa * func(b + offset) <= 0:
                b += offset
                fb = func(b)
                break

    for _ in range(max_iter):
        c = (a + b) / 2.0
        fc = func(c)
        if abs(fc) < tol or (b - a) / 2.0 < tol:
            return c
        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc
    return (a + b) / 2.0


def calculate_equilibrium_potential(env: PlasmaEnvironment = ENV_EARTH,
                                     mat: ProbeMaterial = MAT_ALUMINIUM_ANTENNE,
                                     verbose: bool = True) -> float:
    V_eq = find_root_bisection(lambda V: f_balance(V, env, mat), -50.0, 50.0)
    if verbose:
        print(f"=== ROVNOVÁŽNÝ POTENCIÁL PORUCHY PRO: {mat.name} ({env.name}) ===")
        print(f"  Plovoucí potenciál V_eq = {V_eq:.3f} V\n")
    return V_eq
