from input_data import *
from global_const import *

Jph0_uA_m2 = 40.0  # µA/m^2 (Strong photoemission near the Sun)
Tph_eV = 2.0  # eV

def electron_thermal_current_density(n_m3: float, Te_eV: float) -> float:
    return e * n_m3 * math.sqrt((e * Te_eV) / (2.0 * math.pi * m_e))


def Je(V: float, Je0: float, Te_eV: float) -> float:
    if V < 0.0:
        return Je0 * math.exp(V / Te_eV)
    return Je0 * (1.0 + V / Te_eV)


def Ji(V: float, Ji0_ram: float, Ti_eV: float) -> float:
    if V >= 0.0:
        return Ji0_ram * math.exp(-V / Ti_eV)
    return Ji0_ram


def Jph(V: float, Jph0: float, Tph_eV: float) -> float:
    if V <= 0.0:
        return Jph0
    x = V / Tph_eV
    return Jph0 * (1.0 + x) * math.exp(-x)


def f_balance(V: float, Je0: float, Ji0: float) -> float:
    # Simplified balance without SEE for demo speedup
    return Je(V, Je0, Te_eV) - Ji(V, Ji0, Ti_eV) - Jph(V, Jph0_uA_m2 * 1e-6, Tph_eV)


def get_equilibrium_potential():
    Je0 = electron_thermal_current_density(n_sw, Te_eV)
    Ji0 = e * n_sw * v_sw

    a, b = -50.0, 50.0
    func = lambda V: f_balance(V, Je0, Ji0)

    # Bisection
    for _ in range(100):
        mid = 0.5 * (a + b)
        if abs(func(mid)) < 1e-9 or (b - a) < 1e-9:
            return mid
        if func(a) * func(mid) <= 0.0:
            b = mid
        else:
            a = mid
    return 0.5 * (a + b)