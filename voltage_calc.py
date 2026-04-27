#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rovnovážný potenciál sondy ve slunečním větru z proudové bilance:
  Je(V) = Ji(V) + Jph(V) + Jse(V)

- Je : sběr plazmových elektronů
- Ji : sběr iontů (ram/drift)
- Jph: fotoemise
- Jse: sekundární emise (SEE) vyvolaná dopadem elektronů

Poznámka: Je to zjednodušený 0D model (bez geometrie/sheath detailů),
ale je výborný pro rychlé odhady a citlivost na materiálové parametry.

Parametry fotoemise Tph ~ 2 eV jsou často uváděny jako typické (viz JAXA JERG-2-211A).
"""

import math
from global_const import *

# -----------------------------
# Fyzikální konstanty (SI)
# -----------------------------
# e = 1.602176634e-19      # C
# m_e = 9.1093837015e-31   # kg

# -----------------------------
# Vstupní parametry prostředí (ZADÁNÍ)
# Input environment parameters (solar wind)
# n_sw = 1e7  # m^-3 (solar wind density, adjusted for SolO near the Sun)
# v_sw = 5e5  # m/s
# Te_eV = 10.0  # eV
# Ti_eV = 10.0  # eV
# # -----------------------------
# n = 1e7          # m^-3
# v_sw = 5e5       # m/s
# Te_eV = 10.0     # eV
# Ti_eV = 10.0     # eV
n = n_sw

# -----------------------------
# Fotoemise: Al (typické startovní hodnoty pro 1 AU)
# -----------------------------
# Zvol si podle povrchu (čistý Al vs. oxidovaný Al2O3 apod.)
Jph0_uA_m2 = 10.0   # µA/m^2 (konzervativní "Al" start)
# Jph0_uA_m2 = 42.0 # µA/m^2 (příklad pro Aluminium Oxide z JAXA JERG-2-211A)

Tph_eV = 2        # eV (typické v design standardech)

Jph0 = Jph0_uA_m2 * 1e-6  # A/m^2

# -----------------------------
# SEE: sekundární emise (parametry pro Al/kovový povrch)
# -----------------------------
# Reálná SEY závisí na oxidaci, kontaminaci a úpravě povrchu.
# Pro kovy je δmax typicky ~0.5–1.7 a Emax ~100–900 eV.
# Zde dáme konzervativní defaulty (snadno přenastavitelné):
sey_delta_max = 1.0   # bezrozměrné (max. výtěžnost)
sey_Emax_eV   = 2.0 # eV (energie, kde je δ = δmax)
sey_s_shape   = 1.35  # tvarový parametr (empirický; 1.1–1.8 bývá OK)

# Volitelné: energie, pod kterou SEE prakticky zanedbáš (pomáhá stabilitě modelu v nízkých eV)
sey_E_cutoff_eV = 100000.0

# -----------------------------
# Pomocné výpočty
# -----------------------------
def electron_thermal_current_density(n_m3: float, Te_eV: float) -> float:
    """J_e0 = e*n*sqrt((e*Te)/(2π m_e))  (Te v eV)"""
    return e * n_m3 * math.sqrt((e * Te_eV) / (2.0 * math.pi * m_e))


def Je(V: float, Je0: float, Te_eV: float) -> float:
    """
    Jednoduchá inženýrská aproximace elektronového sběru:
    V < 0 : Je = Je0 * exp(V/Te)
    V >=0 : Je = Je0 * (1 + V/Te)
    """
    if V < 0.0:
        return Je0 * math.exp(V / Te_eV)
    return Je0 * (1.0 + V / Te_eV)


def Ji(V: float, Ji0_ram: float, Ti_eV: float) -> float:
    """
    Iontový "ram" proud se zjednodušenou repulzí pro V>0:
    V >= 0 : Ji = Ji0 * exp(-V/Ti)
    V < 0  : Ji ~ Ji0 (konzervativní; v realitě by mírně rostl)
    """
    if V >= 0.0:
        return Ji0_ram * math.exp(-V / Ti_eV)
    return Ji0_ram


def Jph(V: float, Jph0: float, Tph_eV: float) -> float:
    """
    Fotoemisní proud (unikající fotoelektrony):
    V <= 0 : Jph = Jph0
    V > 0  : Jph = Jph0*(1+V/Tph)*exp(-V/Tph)
    """
    if V <= 0.0:
        return Jph0
    x = V / Tph_eV
    return Jph0 * (1.0 + x) * math.exp(-x)


def sey_delta_vaughan_like(E_eV: float, delta_max: float, Emax_eV: float, s: float) -> float:
    """
    Jednoduchý "Vaughan-like" tvar pro SEY δ(E), s maximem δmax při E=Emax.
    Je to hladká parametrizace vhodná pro rychlé odhady.
    """
    if E_eV <= 0.0:
        return 0.0
    x = E_eV / Emax_eV
    # hladká křivka s max v x=1, řízená parametrem s
    # (empirický tvar; účel je rychlý odhad, ne přesná metrologie povrchu)
    return delta_max * (x**s) / ((s - 1.0) + (x**s))


def electron_impact_energy_eV(V: float, Te_eV: float) -> float:
    """
    Hrubý odhad střední dopadové energie plazmových elektronů:
    - termální složka ~ 2*Te (řádový odhad pro proud nesený rychlejšími elektrony)
    - pokud je sonda kladná, elektrony jsou urychleny o +V eV
    - pokud je záporná, část elektronů je potlačena; energii nepustíme pod 0
    """
    E = 2.0 * Te_eV
    if V > 0.0:
        E += V
    # pro V<0 je Je už potlačeno exponenciálně, energii necháme na termální složce
    return max(0.0, E)


def Jse(V: float, Je_inc: float, Te_eV: float,
        delta_max: float, Emax_eV: float, s: float,
        E_cutoff_eV: float) -> float:
    """
    Sekundární emise vyvolaná dopadem elektronů:
      Jse = δ(Eimp) * Je_inc
    kde Je_inc je incidentní elektronový proud (tady bereme Je(V)).
    """
    Eimp = electron_impact_energy_eV(V, Te_eV)
    if Eimp < E_cutoff_eV:
        return 0.0
    delta = sey_delta_vaughan_like(Eimp, delta_max, Emax_eV, s)
    return delta * Je_inc


def f_balance(V: float, Je0: float, Ji0: float,
              Te_eV: float, Ti_eV: float,
              Jph0: float, Tph_eV: float,
              delta_max: float, Emax_eV: float, s: float, E_cutoff_eV: float) -> float:
    """
    Bilance:
      f(V) = Je(V) - Ji(V) - Jph(V) - Jse(V)
    Kořen f(V)=0 je rovnovážný potenciál Vf.
    """
    je = Je(V, Je0, Te_eV)
    ji = Ji(V, Ji0, Ti_eV)
    jph = Jph(V, Jph0, Tph_eV)
    jse = Jse(V, je, Te_eV, delta_max, Emax_eV, s, E_cutoff_eV)
    return je - ji - jph - jse


def find_root_bisection(func, a: float, b: float, tol: float = 1e-9, max_iter: int = 300) -> float:
    fa = func(a)
    fb = func(b)
    if fa == 0.0:
        return a
    if fb == 0.0:
        return b
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


def get_spacecraft_voltage():
    Je0 = electron_thermal_current_density(n, Te_eV)
    Ji0 = e * n * v_sw  # ram/drift ion current density

    print("=== Vstupy ===")
    print(f"n      = {n:.3e} m^-3")
    print(f"v_sw   = {v_sw:.3e} m/s")
    print(f"Te,Ti  = {Te_eV:.2f} eV, {Ti_eV:.2f} eV")
    print(f"Fotoemise: Jph0 = {Jph0_uA_m2:.2f} µA/m^2, Tph = {Tph_eV:.2f} eV")
    print(
        f"SEE: δmax = {sey_delta_max:.2f}, Emax = {sey_Emax_eV:.1f} eV, s = {sey_s_shape:.2f}, cutoff = {sey_E_cutoff_eV:.1f} eV")
    print()
    print("=== Bazální proudy ===")
    print(f"Je0 = {Je0 * 1e6:.3f} µA/m^2")
    print(f"Ji0 = {Ji0 * 1e6:.3f} µA/m^2")
    print()

    func = lambda V: f_balance(
        V, Je0, Ji0, Te_eV, Ti_eV,
        Jph0, Tph_eV,
        sey_delta_max, sey_Emax_eV, sey_s_shape, sey_E_cutoff_eV
    )

    # Najdi bracket pro kořen
    a, b = -100.0, 100.0
    fa, fb = func(a), func(b)
    if fa * fb > 0.0:
        for span in (200.0, 400.0, 800.0):
            a, b = -span, span
            fa, fb = func(a), func(b)
            if fa * fb <= 0.0:
                break
        else:
            raise RuntimeError("Nepodařilo se najít interval se změnou znaménka pro kořen.")

    Vf = find_root_bisection(func, a, b)

    je = Je(Vf, Je0, Te_eV)
    ji = Ji(Vf, Ji0, Ti_eV)
    jph = 0
    jse = 0
    Eimp = electron_impact_energy_eV(Vf, Te_eV)


    print("=== Výsledek ===")
    print(f"Vf = {Vf:.4f} V")
    print("Proudy v rovnováze (µA/m^2):")
    print(f"Je   = {je * 1e6:.3f}")
    print(f"Ji   = {ji * 1e6:.3f}")
    print(f"Jph  = {jph * 1e6:.3f}")
    print(f"Jse  = {jse * 1e6:.3f}")
    print(f"Kontrola: Je - Ji - Jph - Jse = {(je - ji - jph - jse) * 1e6:.6f} µA/m^2")

    return Vf

def main():
    Je0 = electron_thermal_current_density(n, Te_eV)
    Ji0 = e * n * v_sw  # ram/drift ion current density

    print("=== Vstupy ===")
    print(f"n      = {n:.3e} m^-3")
    print(f"v_sw   = {v_sw:.3e} m/s")
    print(f"Te,Ti  = {Te_eV:.2f} eV, {Ti_eV:.2f} eV")
    print(f"Fotoemise: Jph0 = {Jph0_uA_m2:.2f} µA/m^2, Tph = {Tph_eV:.2f} eV")
    print(f"SEE: δmax = {sey_delta_max:.2f}, Emax = {sey_Emax_eV:.1f} eV, s = {sey_s_shape:.2f}, cutoff = {sey_E_cutoff_eV:.1f} eV")
    print()
    print("=== Bazální proudy ===")
    print(f"Je0 = {Je0*1e6:.3f} µA/m^2")
    print(f"Ji0 = {Ji0*1e6:.3f} µA/m^2")
    print()

    func = lambda V: f_balance(
        V, Je0, Ji0, Te_eV, Ti_eV,
        Jph0, Tph_eV,
        sey_delta_max, sey_Emax_eV, sey_s_shape, sey_E_cutoff_eV
    )

    # Najdi bracket pro kořen
    a, b = -100.0, 100.0
    fa, fb = func(a), func(b)
    if fa * fb > 0.0:
        for span in (200.0, 400.0, 800.0):
            a, b = -span, span
            fa, fb = func(a), func(b)
            if fa * fb <= 0.0:
                break
        else:
            raise RuntimeError("Nepodařilo se najít interval se změnou znaménka pro kořen.")

    Vf = find_root_bisection(func, a, b)

    je = Je(Vf, Je0, Te_eV)
    ji = Ji(Vf, Ji0, Ti_eV)
    # jph = Jph(Vf, Jph0, Tph_eV)
    jph = 0
    # jse = Jse(Vf, je, Te_eV, sey_delta_max, sey_Emax_eV, sey_s_shape, sey_E_cutoff_eV)
    jse = 0
    Eimp = electron_impact_energy_eV(Vf, Te_eV)
    delta = sey_delta_vaughan_like(Eimp, sey_delta_max, sey_Emax_eV, sey_s_shape) if Eimp >= sey_E_cutoff_eV else 0.0

    print("=== Výsledek ===")
    print(f"Vf = {Vf:.4f} V")
    print(f"E_imp(e-) ~ {Eimp:.2f} eV, δ(E_imp) ~ {delta:.3f}")
    print("Proudy v rovnováze (µA/m^2):")
    print(f"Je   = {je*1e6:.3f}")
    print(f"Ji   = {ji*1e6:.3f}")
    print(f"Jph  = {jph*1e6:.3f}")
    print(f"Jse  = {jse*1e6:.3f}")
    print(f"Kontrola: Je - Ji - Jph - Jse = {(je-ji-jph-jse)*1e6:.6f} µA/m^2")


if __name__ == "__main__":
    main()
