#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from global_const import *
import numpy as np
from dataclasses import dataclass, field, asdict, fields
from typing import Tuple
import json
import os


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
class PlottingConfig:
    # Přepínače, které grafy/animace vůbec generovat a zobrazit
    show_currents: bool = True
    show_fields_anim: bool = True
    show_weighting_field: bool = True
    show_particles_anim: bool = True
    show_velocity_anim: bool = True
    show_phase_space_anim: bool = False

    # Přepínače uložení na disk
    save_plots: bool = False
    export_data_csv: bool = False

    # Názvy výstupních souborů
    file_currents: str = "../outputs/out_2d_proudy_napeti.png"
    file_fields_anim: str = "../outputs/out_2d_animace_pole_potencial.gif"
    file_weighting: str = "out_2d_graf_vahove_pole.png"
    file_particles_anim: str = "out_2d_animace_pozice_castic.gif"
    file_velocity_anim: str = "../outputs/out_2d_animace_rychlosti.gif"
    file_phase_space: str = "out_2d_animace_fazovy_prostor.gif"
    file_csv: str = "out_2d_vysledky_simulace.csv"


@dataclass
class SimulationParams2D:
    Vf: float
    V_ant_bias: float

    # Fyzikální konstanty
    m_i_amu: float = 27.0  # Hliníkový iont (zadáno v AMU pro snazší zápis do JSONu)
    T_dust_eV: float = 2.0
    N_particles: int = 15000

    # Čas a prostor
    dt: float = 1e-9
    t_max: float = 20e-6
    t_delay: float = 1e-6
    L_domain: float = 5.0
    H_domain: float = 2.5

    # Anténa
    x_antenna: float = 2.5
    y_antenna: float = 0.0
    r_antenna: float = 0.05
    w_width: float = 0.4
    collection_efficiency: float = 0.80
    C_ant: float = 2e-12
    R_ant: float = 100e3

    # Mřížka (Grid)
    Nx: int = 100
    Ny: int = 100
    A_sim: float = 1.0
    integrator: str = 'leapfrog'

    # ================== Interní kalkulované proměnné ==================
    m_i: float = field(init=False)
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
        # Převod z AMU na kg
        self.m_i = self.m_i_amu * amu

        self.debye_length = np.sqrt((eps_0 * Te_eV * e) / (n_sw * e ** 2))
        self.v_th_e = np.sqrt(2 * e * self.T_dust_eV / m_e)
        self.v_th_i = np.sqrt(2 * e * self.T_dust_eV / self.m_i)

        self.q_macro = 50e-12 / self.N_particles

        self.steps = int(self.t_max / self.dt)
        self.time_array = np.linspace(0, self.t_max, self.steps)

        self.x_grid = np.linspace(0, self.L_domain, self.Nx)
        self.y_grid = np.linspace(-self.H_domain, self.H_domain, self.Ny)
        self.dx = self.x_grid[1] - self.x_grid[0]
        self.dy = self.y_grid[1] - self.y_grid[0]

        self.X_mat, self.Y_mat = np.meshgrid(self.x_grid, self.y_grid, indexing='ij')

        self.plot_stride = max(1, self.N_particles // 1500)
        self.save_interval = max(1, self.steps // 50)


# =============================================================================
# MANAŽER KONFIGURAČNÍHO SOUBORU
# =============================================================================

def setup_simulation_parameters_2d(Vf: float, Vf_antenne: float, config_file: str = "config_2d.json") -> Tuple[
    SimulationParams2D, SimulationToggles2D, PlottingConfig]:
    """
    Načte nastavení ze souboru JSON. Pokud soubor neexistuje, automaticky ho vytvoří
    s výchozími hodnotami, aby ho uživatel mohl následně editovat.
    """
    if not os.path.exists(config_file):
        print(f"Konfigurační soubor '{config_file}' nenalezen. Vytvářím výchozí šablonu...")

        # Extrakce pouze konfigurovatelných parametrů (init=True) a bez fixních napětí (Vf, V_ant_bias),
        # protože ty se počítají dynamicky před startem simulace.
        dummy_params = SimulationParams2D(Vf=0.0, V_ant_bias=0.0)
        params_dict = {
            f.name: getattr(dummy_params, f.name)
            for f in fields(SimulationParams2D)
            if f.init and f.name not in ['Vf', 'V_ant_bias']
        }

        default_config = {
            "toggles": asdict(SimulationToggles2D()),
            "params": params_dict,
            "plotting": asdict(PlottingConfig())
        }

        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(default_config, f, indent=4, ensure_ascii=False)

    # Načtení dat z konfiguračního souboru
    with open(config_file, 'r', encoding='utf-8') as f:
        config_data = json.load(f)

    # 1. Toggles
    toggles = SimulationToggles2D(**config_data.get('toggles', {}))

    # 2. Params (napětí z výpočtu, zbytek ze souboru)
    params_kwargs = config_data.get('params', {})
    params_kwargs['Vf'] = Vf
    params_kwargs['V_ant_bias'] = Vf_antenne
    params = SimulationParams2D(**params_kwargs)

    # 3. Nastavení vizualizací a exportů
    plot_config = PlottingConfig(**config_data.get('plotting', {}))

    return params, toggles, plot_config