#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple

# Základní konstanty (stejné jako u 1D)
e = 1.602176634e-19  # Elementární náboj [C]
m_e = 9.1093837015e-31  # Hmotnost elektronu [kg]
eps_0 = 8.8541878128e-12  # Permitivita vakua [F/m]
amu = 1.66053906660e-27  # Atomová hmotnostní jednotka [kg]

# Parametry slunečního větru (pro výpočet Debyeovy délky)
n_sw = 1e7  # Hustota slunečního větru [m^-3]
Te_eV = 15.0  # Teplota elektronů slunečního větru [eV]


# =============================================================================
# POMOCNÉ FUNKCE PRO VÁHOVÉ POLE (2D)
# =============================================================================

def calc_Vw_2d(x: np.ndarray, y: np.ndarray, x_ant: float, y_ant: float, w_width: float) -> np.ndarray:
    """ 2D Gaussovský váhový potenciál centrovaný na pozici antény """
    r_sq = (x - x_ant) ** 2 + (y - y_ant) ** 2
    return np.exp(-r_sq / (w_width ** 2))


def calc_Ew_2d(x: np.ndarray, y: np.ndarray, x_ant: float, y_ant: float, w_width: float) -> Tuple[
    np.ndarray, np.ndarray]:
    """ 2D Váhové pole jako záporný gradient Vw(x, y) """
    r_sq = (x - x_ant) ** 2 + (y - y_ant) ** 2
    factor = (2.0 / w_width ** 2) * np.exp(-r_sq / (w_width ** 2))
    Ewx = (x - x_ant) * factor
    Ewy = (y - y_ant) * factor
    return Ewx, Ewy


# =============================================================================
# NASTAVENÍ SIMULACE
# =============================================================================

@dataclass
class SimulationToggles2D:
    enable_background_field: bool = True
    enable_self_field: bool = True
    enable_antenna_bias: bool = True
    enable_antenna_collection: bool = True
    enable_rc_circuit: bool = True


@dataclass
class SimulationParams2D:
    Vf: float

    # Fyzikální konstanty
    m_i: float = 27 * amu  # Hliníkový iont
    T_dust_eV: float = 2.0
    N_particles: int = 15000  # Ve 2D je potřeba trochu více částic pro hladkost

    # Čas a prostor (2D Doména: x in [0, L], y in [-H, H])
    dt: float = 1e-9
    t_max: float = 20e-6
    t_delay: float = 1e-6
    L_domain: float = 5.0  # Osa X (Délka do prostoru)
    H_domain: float = 2.5  # Osa Y (Výška od -2.5 do 2.5)

    # Anténa (Umístěna v prostoru jako kruh/bod)
    x_antenna: float = 2.5
    y_antenna: float = 0.0
    r_antenna: float = 0.05  # Poloměr detekční oblasti antény [m]
    V_ant_bias: float = 8.0
    w_width: float = 0.4  # Šířka citlivosti pro Ramo-Shockley ve 2D
    collection_efficiency: float = 0.80
    C_ant: float = 2e-12
    R_ant: float = 100e3

    # Mřížka (Grid)
    Nx: int = 100
    Ny: int = 100
    A_sim: float = 1.0  # Expanzní rozměr v ose Z (hloubka 1m)
    integrator: str = 'leapfrog'  # Leapfrog je pro 2D optimální kompromis

    # Předpočítané interní hodnoty
    debye_length: float = field(init=False)
    v_th_e: float = field(init=False)
    v_th_i: float = field(init=False)
    q_macro: float = field(init=False)
    steps: int = field(init=False)
    time_array: np.ndarray = field(init=False)
    dx: float = field(init=False)
    dy: float = field(init=False)
    x_grid: np.ndarray = field(init=False)
    y_grid: np.ndarray = field(init=False)
    X_mat: np.ndarray = field(init=False)
    Y_mat: np.ndarray = field(init=False)
    plot_stride: int = field(init=False)
    save_interval: int = field(init=False)

    def __post_init__(self):
        self.debye_length = np.sqrt((eps_0 * Te_eV * e) / (n_sw * e ** 2))
        self.v_th_e = np.sqrt(2 * e * self.T_dust_eV / m_e)
        self.v_th_i = np.sqrt(2 * e * self.T_dust_eV / self.m_i)

        # Makronáboj (např. 50 pC celkem)
        self.q_macro = 50e-12 / self.N_particles

        self.steps = int(self.t_max / self.dt)
        self.time_array = np.linspace(0, self.t_max, self.steps)

        # Konstrukce 2D mřížky
        self.x_grid = np.linspace(0, self.L_domain, self.Nx)
        self.y_grid = np.linspace(-self.H_domain, self.H_domain, self.Ny)
        self.dx = self.x_grid[1] - self.x_grid[0]
        self.dy = self.y_grid[1] - self.y_grid[0]

        self.X_mat, self.Y_mat = np.meshgrid(self.x_grid, self.y_grid, indexing='ij')

        self.plot_stride = max(1, self.N_particles // 1500)
        self.save_interval = max(1, self.steps // 50)


def setup_simulation_parameters_2d(Vf: float) -> Tuple[SimulationParams2D, SimulationToggles2D]:
    toggles = SimulationToggles2D()
    params = SimulationParams2D(Vf=Vf)
    return params, toggles