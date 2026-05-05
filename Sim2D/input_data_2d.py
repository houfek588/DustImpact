#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from global_const import *
import numpy as np
from dataclasses import dataclass, field, asdict, fields
from typing import Tuple, List, Dict, Any
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

    # Názvy výstupních datových souborů
    file_results_npz: str = "out_2d_vysledky.npz"
    file_csv: str = "out_2d_vysledky_simulace.csv"

    # Názvy výstupních grafů
    file_currents: str = "out_2d_proudy_napeti.png"
    file_fields_anim: str = "out_2d_animace_pole_potencial.gif"
    file_weighting: str = "out_2d_graf_vahove_pole.png"
    file_particles_anim: str = "out_2d_animace_pozice_castic.gif"
    file_velocity_anim: str = "out_2d_animace_rychlosti.gif"
    file_phase_space: str = "out_2d_animace_fazovy_prostor.gif"


@dataclass
class SimulationParams2D:
    Vf: float

    # Seznam definující libovolný počet antén a jejich lokálních parametrů
    antennas: List[Dict[str, float]] = field(default_factory=list)

    # Fyzikální konstanty
    m_i_amu: float = 27.0
    T_dust_eV: float = 2.0
    N_particles: int = 15000

    # Čas a prostor
    dt: float = 1e-9
    t_max: float = 20e-6
    t_delay: float = 1e-6
    L_domain: float = 5.0
    H_domain: float = 2.5
    y_impact: float = 0.0

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
# MANAŽER KONFIGURAČNÍHO SOUBORU A DATOVÝ WORFKLOW
# =============================================================================

def setup_simulation_parameters_2d(Vf: float, Vf_antenne: float, config_file: str = "config_2d.json") -> Tuple[
    SimulationParams2D, SimulationToggles2D, PlottingConfig]:
    if not os.path.exists(config_file):
        print(f"Konfigurační soubor '{config_file}' nenalezen. Vytvářím výchozí šablonu...")

        # Výchozí stav se DVĚMA anténami
        default_antennas = [
            {"x": 2.5, "y": 0.5, "r": 0.05, "V_bias": Vf_antenne, "w_width": 0.4, "collection_eff": 0.80, "C": 2e-12,
             "R": 100e3},
            {"x": 2.5, "y": -0.5, "r": 0.05, "V_bias": Vf_antenne, "w_width": 0.4, "collection_eff": 0.80, "C": 2e-12,
             "R": 100e3}
        ]

        dummy_params = SimulationParams2D(Vf=0.0, antennas=default_antennas)
        params_dict = {
            f.name: getattr(dummy_params, f.name)
            for f in fields(SimulationParams2D)
            if f.init and f.name != 'Vf'
        }

        default_config = {
            "toggles": asdict(SimulationToggles2D()),
            "params": params_dict,
            "plotting": asdict(PlottingConfig())
        }

        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(default_config, f, indent=4, ensure_ascii=False)

    with open(config_file, 'r', encoding='utf-8') as f:
        config_data = json.load(f)

    toggles = SimulationToggles2D(**config_data.get('toggles', {}))
    params_kwargs = config_data.get('params', {})
    params_kwargs['Vf'] = Vf
    params = SimulationParams2D(**params_kwargs)
    plot_config = PlottingConfig(**config_data.get('plotting', {}))

    return params, toggles, plot_config


def save_results_npz(results: Dict[str, Any], filepath: str) -> None:
    """
    Uloží kompletní výsledky (včetně makroskopických polí a historie částic)
    do jediného efektivně zkomprimovaného .npz archivu.
    """
    print(f"Ukládám výsledky do souboru: {filepath} ...")

    # Převod vnitřních seznamů na výkonná vícerozměrná Numpy pole pro extra rychlý zápis
    np.savez_compressed(
        filepath,
        smooth_induced=np.array(results['smooth_induced']),
        smooth_collected=np.array(results['smooth_collected']),
        smooth_total=np.array(results['smooth_total']),
        voltage_ant=np.array(results['voltage_ant']),
        hist_V=np.array(results['history']['V']),
        hist_rho=np.array(results['history']['rho']),
        hist_t=np.array(results['history']['t']),
        hist_x_e=np.array(results['history']['x_e']),
        hist_y_e=np.array(results['history']['y_e']),
        hist_x_i=np.array(results['history']['x_i']),
        hist_y_i=np.array(results['history']['y_i']),
        hist_vx_e=np.array(results['history']['vx_e']),
        hist_vy_e=np.array(results['history']['vy_e']),
        hist_vx_i=np.array(results['history']['vx_i']),
        hist_vy_i=np.array(results['history']['vy_i'])
    )
    print("  [OK] Data úspěšně uložena do binárního archivu.")


def load_results_npz(filepath: str) -> Dict[str, Any]:
    """
    Načte zazipovaný soubor .npz z disku zpět do struktury slovníku
    stejné formy, jako produkuje přímo výpočetní jádro.
    """
    print(f"Načítám výsledky ze souboru: {filepath} ...")

    with np.load(filepath, allow_pickle=True) as data:
        results = {
            'smooth_induced': data['smooth_induced'],
            'smooth_collected': data['smooth_collected'],
            'smooth_total': data['smooth_total'],
            'voltage_ant': data['voltage_ant'],
            'history': {
                'V': data['hist_V'],
                'rho': data['hist_rho'],
                't': data['hist_t'],
                'x_e': data['hist_x_e'],
                'y_e': data['hist_y_e'],
                'x_i': data['hist_x_i'],
                'y_i': data['hist_y_i'],
                'vx_e': data['hist_vx_e'],
                'vy_e': data['hist_vy_e'],
                'vx_i': data['hist_vx_i'],
                'vy_i': data['hist_vy_i']
            }
        }
    print("  [OK] Data úspěšně načtena do operační paměti.")
    return results