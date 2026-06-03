#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rovnovážný potenciál sondy ve slunečním větru z proudové bilance:
  Je(V) = Ji(V) + Jph(V) + Jse(V)

Tento 0D model odhaduje rovnovážné napětí pro různá tělesa (materiály sondy)
a v různých prostředích (např. oběžné dráhy různých planet).
"""

import math
from dataclasses import dataclass
from global_const import *

# -----------------------------
# Fyzikální konstanty (SI)
# -----------------------------
E_CHARGE = e  # C (elementární náboj)
M_ELECTRON = m_e  # kg (hmotnost elektronu)


# -----------------------------
# Definice datových struktur
# -----------------------------
@dataclass
class ProbeMaterial:
    """Parametry povrchu (tělesa) sondy."""
    name: str
    Jph0_uA_m2: float  # Fotoemise v µA/m^2 při 1 AU
    Tph_eV: float  # Teplota fotoelektronů v eV
    sey_delta_max: float  # Max. výtěžnost sekundární emise (bezrozměrné)
    sey_Emax_eV: float  # Energie pro max. SEY v eV
    sey_s_shape: float  # Tvarový parametr křivky SEY (typicky 1.1-1.8)
    sey_E_cutoff_eV: float  # Mezní energie, pod kterou SEE zanedbáváme


@dataclass
class PlasmaEnvironment:
    """Parametry plazmového prostředí."""
    name: str
    n: float  # Koncentrace plazmatu (m^-3)
    v_sw: float  # Rychlost slunečního větru (m/s)
    Te_eV: float  # Teplota elektronů (eV)
    Ti_eV: float  # Teplota iontů (eV)
    distance_au: float  # Vzdálenost od Slunce v AU (pro škálování fotoemise)


# -----------------------------
# Předdefinovaná tělesa a prostředí
# -----------------------------
# Materiály sond (různá tělesa)
MAT_ALUMINIUM_ANTENNE = ProbeMaterial("Anténa, hliník", 10.0, 2.0, 1.0, 300.0, 1.25, 5.0)
MAT_ALUMINIUM = ProbeMaterial("Sonda, hliník", 10.0, 2.0, 1.0, 300.0, 1.8, 5.0)

MAT_AL_OXIDE = ProbeMaterial("Oxidovaný hliník", 42.0, 2.0, 2.5, 400.0, 1.35, 5.0)
MAT_GOLD = ProbeMaterial("Zlato", 20.0, 2.0, 0.8, 800.0, 1.35, 5.0)

# Prostředí slunečního větru u různých planet
# (Koncentrace klesá zhruba s 1/r^2, rychlost a teplota se mění pomaleji)
ENV_EARTH = PlasmaEnvironment("Země (1.0 AU)", 1e7, 5e5, 10.0, 10.0, 1.0)
ENV_MARS = PlasmaEnvironment("Mars (1.52 AU)", 1e7 / (1.52 ** 2), 5e5, 10.0, 10.0, 1.52)
ENV_JUPITER = PlasmaEnvironment("Jupiter (5.2 AU)", 1e7 / (5.2 ** 2), 4.5e5, 8.0, 8.0, 5.2)


# -----------------------------
# Fyzikální a pomocné funkce
# -----------------------------
def electron_thermal_current_density(n_m3: float, Te_eV: float) -> float:
    """Tepelný proud elektronů: J_e0 = e*n*sqrt((e*Te)/(2π m_e))"""
    return E_CHARGE * n_m3 * math.sqrt((E_CHARGE * Te_eV) / (2.0 * math.pi * M_ELECTRON))


def Je(V: float, Je0: float, Te_eV: float) -> float:
    """Elektronový sběr (zjednodušená inženýrská aproximace)."""
    if V < 0.0:
        return Je0 * math.exp(V / Te_eV)
    return Je0 * (1.0 + V / Te_eV)


def Ji(V: float, Ji0_ram: float, Ti_eV: float) -> float:
    """Iontový sběr ('ram' proud)."""
    if V >= 0.0:
        return Ji0_ram * math.exp(-V / Ti_eV)
    return Ji0_ram


def Jph(V: float, Jph0: float, Tph_eV: float) -> float:
    """Fotoemisní proud unikajících elektronů."""
    if V <= 0.0:
        return Jph0
    x = V / Tph_eV
    return Jph0 * (1.0 + x) * math.exp(-x)


def sey_delta_vaughan_like(E_eV: float, delta_max: float, Emax_eV: float, s: float) -> float:
    """Vaughan-like tvar pro sekundární emisi (SEY)."""
    if E_eV <= 0.0:
        return 0.0
    x = E_eV / Emax_eV
    return delta_max * (x ** s) / ((s - 1.0) + (x ** s))


def electron_impact_energy_eV(V: float, Te_eV: float) -> float:
    """Odhad střední dopadové energie plazmových elektronů."""
    E = 2.0 * Te_eV
    if V > 0.0:
        E += V
    return max(0.0, E)


def Jse(V: float, Je_inc: float, Te_eV: float, mat: ProbeMaterial) -> float:
    """Sekundární emise vyvolaná dopadem elektronů."""
    Eimp = electron_impact_energy_eV(V, Te_eV)
    if Eimp < mat.sey_E_cutoff_eV:
        return 0.0
    delta = sey_delta_vaughan_like(Eimp, mat.sey_delta_max, mat.sey_Emax_eV, mat.sey_s_shape)
    return delta * Je_inc


def f_balance(V: float, Je0: float, Ji0: float, env: PlasmaEnvironment,
              Jph0_local: float, mat: ProbeMaterial) -> float:
    """Rovnice proudové bilance: f(V) = Je - Ji - Jph - Jse = 0"""
    je = Je(V, Je0, env.Te_eV)
    ji = Ji(V, Ji0, env.Ti_eV)
    jph = Jph(V, Jph0_local, mat.Tph_eV)
    jse = Jse(V, je, env.Te_eV, mat)
    return je - ji - jph - jse


def find_root_bisection(func, a: float, b: float, tol: float = 1e-9, max_iter: int = 300) -> float:
    """Hledání kořene metodou bisekce (půlení intervalů)."""
    fa, fb = func(a), func(b)
    if fa == 0.0: return a
    if fb == 0.0: return b
    if fa * fb > 0.0:
        raise ValueError(f"Interval nebracketuje kořen: f(a)={fa:.3e}, f(b)={fb:.3e}")

    for _ in range(max_iter):
        mid = 0.5 * (a + b)
        fm = func(mid)
        if abs(fm) < tol or (b - a) < tol:
            return mid
        if fa * fm <= 0.0:
            b, fb = mid, fm
        else:
            a, fa = mid, fm
    return 0.5 * (a + b)


# -----------------------------
# Hlavní výpočetní funkce
# -----------------------------
def calculate_equilibrium_potential(env: PlasmaEnvironment, mat: ProbeMaterial, verbose: bool = True) -> float:
    """Vypočítá rovnovážný potenciál pro dané těleso (materiál) v daném prostředí."""

    # Bazální proudy
    Je0 = electron_thermal_current_density(env.n, env.Te_eV)
    Ji0 = E_CHARGE * env.n * env.v_sw

    # Úprava fotoemise podle vzdálenosti od Slunce
    Jph0_local = (mat.Jph0_uA_m2 * 1e-6) / (env.distance_au ** 2)

    if verbose:
        print(f"\n{'=' * 40}")
        print(f"Výpočet pro: Sonda '{mat.name}' u '{env.name}'")
        print(f"{'=' * 40}")
        print(f"Plazma: n = {env.n:.2e} m^-3, v_sw = {env.v_sw:.2e} m/s, Te = {env.Te_eV:.1f} eV")
        print(f"Lokální fotoemise: {Jph0_local * 1e6:.2f} µA/m^2 (škálováno pro {env.distance_au} AU)")
        print(f"SEE: δmax = {mat.sey_delta_max:.2f}, Emax = {mat.sey_Emax_eV:.1f} eV")
        print(f"Bazální proudy: Je0 = {Je0 * 1e6:.3f} µA/m^2, Ji0 = {Ji0 * 1e6:.3f} µA/m^2")

    # Definice funkce pro hledání kořene
    func = lambda V: f_balance(V, Je0, Ji0, env, Jph0_local, mat)

    # Automatické nalezení intervalu pro kořen
    a, b = -100.0, 100.0
    if func(a) * func(b) > 0.0:
        for span in (200.0, 400.0, 800.0, 1600.0):
            a, b = -span, span
            if func(a) * func(b) <= 0.0:
                break
        else:
            raise RuntimeError(f"Nepodařilo se najít interval pro kořen (Sonda: {mat.name}).")

    # Samotný výpočet
    Vf = find_root_bisection(func, a, b)

    # Následný rozbor proudů
    je = Je(Vf, Je0, env.Te_eV)
    ji = Ji(Vf, Ji0, env.Ti_eV)
    jph = Jph(Vf, Jph0_local, mat.Tph_eV)
    jse = Jse(Vf, je, env.Te_eV, mat)

    if verbose:
        print(f"\n--- Výsledek ---")
        print(f"Vf = {Vf:.4f} V")
        print(f"Proudy (µA/m^2): Je={je * 1e6:.3f}, Ji={ji * 1e6:.3f}, Jph={jph * 1e6:.3f}, Jse={jse * 1e6:.3f}")
        print(f"Kontrola bilance: {(je - ji - jph - jse) * 1e6:.6f} µA/m^2")

    return Vf


if __name__ == "__main__":
    # Testovací scénáře pro rozlišení různých těles a prostředí

    # 1. Různé materiály sondy na orbitě Země
    calculate_equilibrium_potential(ENV_EARTH, MAT_ALUMINIUM_ANTENNE)
    calculate_equilibrium_potential(ENV_EARTH, MAT_ALUMINIUM)
    # calculate_equilibrium_potential(ENV_EARTH, MAT_AL_OXIDE)
    #
    # # 2. Stejná sonda v různých prostředích (Země vs. Jupiter)
    # # Fotoemise u Jupitera razantně klesá, což se odrazí v zápornějším potenciálu
    # calculate_equilibrium_potential(ENV_JUPITER, MAT_ALUMINIUM)