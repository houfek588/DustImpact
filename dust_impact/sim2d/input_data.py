# -*- coding: utf-8 -*-
"""
Input data, configuration loader, and geometric weighting functions for 2D PIC simulation.
"""

import os
import json
import numpy as np
from dataclasses import dataclass, field, asdict, fields
from typing import Tuple, List, Dict, Any
from dust_impact.physics.constants import e, m_e, eps_0, amu, n_sw, Te_eV
from dust_impact.common.io import save_results_npz, load_results_npz, ensure_dir


# =============================================================================
# POMOCNÉ FUNKCE PRO VÁHOVÉ POLE A GEOMETRII (2D)
# =============================================================================

def _dist_to_segment_sq(x: np.ndarray, y: np.ndarray, x1: float, y1: float, x2: float, y2: float) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray]:
    """ Vypočítá kvadrát nejkratší vzdálenosti od bodů (x,y) k úsečce (x1,y1)-(x2,y2). """
    px = x - x1
    py = y - y1
    vx = x2 - x1
    vy = y2 - y1

    v_sq = vx ** 2 + vy ** 2
    if v_sq == 0:
        return px ** 2 + py ** 2, np.full_like(x, x1), np.full_like(y, y1)

    t = (px * vx + py * vy) / v_sq
    t = np.clip(t, 0.0, 1.0)

    cx = x1 + t * vx
    cy = y1 + t * vy

    dx = x - cx
    dy = y - cy

    return dx ** 2 + dy ** 2, cx, cy


def calc_Vw_2d(x: np.ndarray, y: np.ndarray, x1: float, y1: float, x2: float, y2: float, w_width: float) -> np.ndarray:
    """ 2D Gaussovský váhový potenciál centrovaný na drátovou anténu """
    r_sq, _, _ = _dist_to_segment_sq(x, y, x1, y1, x2, y2)
    return np.exp(-r_sq / (w_width ** 2))


def calc_Ew_2d(x: np.ndarray, y: np.ndarray, x1: float, y1: float, x2: float, y2: float, w_width: float) -> Tuple[
    np.ndarray, np.ndarray]:
    """ 2D Váhové pole jako záporný gradient Vw(x, y) vůči drátové anténě """
    r_sq, cx, cy = _dist_to_segment_sq(x, y, x1, y1, x2, y2)
    factor = (2.0 / w_width ** 2) * np.exp(-r_sq / (w_width ** 2))
    Ewx = (x - cx) * factor
    Ewy = (y - cy) * factor
    return Ewx, Ewy


# =============================================================================
# NASTAVENÍ SIMULACE (2D)
# =============================================================================

@dataclass
class SimulationToggles2D:
    enable_spis_background_field: bool = True
    enable_plasma_self_field: bool = True
    enable_antenna_particle_collection: bool = True
    enable_rc_circuit_response: bool = True
    enable_debye_screening: bool = True
    enable_antenna_bias_voltage: bool = True


@dataclass
class PlottingConfig:
    run_physical_simulation: bool = True
    show_interactive_gui_windows: bool = False
    save_plots_to_disk: bool = True
    export_csv_time_series: bool = False

    show_currents: bool = True
    show_fields_anim: bool = True
    show_weighting_field: bool = True
    show_particles_anim: bool = True
    show_velocity_anim: bool = True
    show_phase_space_anim: bool = False

    output_npz_filepath: str = "outputs/out_2d_vysledky.npz"
    output_csv_filepath: str = "outputs/out_2d_vysledky_simulace.csv"
    file_currents: str = "outputs/out_2d_proudy_napeti.png"
    file_fields_anim: str = "outputs/out_2d_animace_pole_potencial.gif"
    file_weighting: str = "outputs/out_2d_graf_vahove_pole.png"
    file_particles_anim: str = "outputs/out_2d_animace_pozice_castic.gif"
    file_velocity_anim: str = "outputs/out_2d_animace_rychlosti.gif"
    file_phase_space: str = "outputs/out_2d_animace_fazovy_prostor.gif"


@dataclass
class SimulationParams2D:
    Vf: float = 0.0
    Vf_antenne: float = 0.0
    antennas: List[Dict[str, float]] = field(default_factory=list)

    ion_mass_amu: float = 27.0
    impact_cloud_temperature_eV: float = 2.0
    num_macroparticles: int = 15000

    time_step_s: float = 1e-9
    simulation_duration_s: float = 20e-6
    impact_time_delay_s: float = 1e-6

    domain_length_x_m: float = 5.0
    domain_height_y_m: float = 2.5
    spacecraft_radius: float = 1.0
    impact_location_y_m: float = 0.0

    grid_nodes_x: int = 100
    grid_nodes_y: int = 100
    A_sim: float = 1.0
    solar_wind_electron_temp_eV: float = 15.0
    solar_wind_density_m3: float = 1e7

    # Derived internal parameters
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

    # Convenience aliases
    Nx: int = field(init=False)
    Ny: int = field(init=False)
    dt: float = field(init=False)
    t_max: float = field(init=False)
    t_delay: float = field(init=False)
    L_domain: float = field(init=False)
    H_domain: float = field(init=False)
    y_impact: float = field(init=False)
    N_particles: int = field(init=False)
    T_dust_eV: float = field(init=False)

    def __post_init__(self):
        self.m_i = self.ion_mass_amu * amu
        self.debye_length = np.sqrt((eps_0 * self.solar_wind_electron_temp_eV * e) / (self.solar_wind_density_m3 * e ** 2))
        self.v_th_e = np.sqrt(2 * e * self.impact_cloud_temperature_eV / m_e)
        self.v_th_i = np.sqrt(2 * e * self.impact_cloud_temperature_eV / self.m_i)
        self.q_macro = 50e-12 / self.num_macroparticles

        self.steps = int(self.simulation_duration_s / self.time_step_s)
        self.time_array = np.linspace(0, self.simulation_duration_s, self.steps)

        self.x_grid = np.linspace(0, self.domain_length_x_m, self.grid_nodes_x)
        self.y_grid = np.linspace(-self.domain_height_y_m, self.domain_height_y_m, self.grid_nodes_y)
        self.dx = self.x_grid[1] - self.x_grid[0]
        self.dy = self.y_grid[1] - self.y_grid[0]

        self.Nx = self.grid_nodes_x
        self.Ny = self.grid_nodes_y
        self.dt = self.time_step_s
        self.t_max = self.simulation_duration_s
        self.t_delay = self.impact_time_delay_s
        self.L_domain = self.domain_length_x_m
        self.H_domain = self.domain_height_y_m
        self.y_impact = self.impact_location_y_m
        self.N_particles = self.num_macroparticles
        self.T_dust_eV = self.impact_cloud_temperature_eV

        self.X_mat, self.Y_mat = np.meshgrid(self.x_grid, self.y_grid, indexing='ij')
        self.plot_stride = max(1, self.num_macroparticles // 1500)
        self.save_interval = max(1, self.steps // 50)


def setup_simulation_parameters_2d(Vf: float, Vf_antenne: float, config_file: str = "config.json") -> Tuple[
    SimulationParams2D, SimulationToggles2D, PlottingConfig]:
    if not os.path.exists(config_file) and os.path.exists("inputs/config.json"):
        config_file = "inputs/config.json"

    if not os.path.exists(config_file):
        config_file = "config.json"

    with open(config_file, 'r', encoding='utf-8') as f:
        config_data = json.load(f)

    toggle_kwargs = config_data.get('toggles', {})
    valid_toggle_keys = {f.name for f in fields(SimulationToggles2D)}
    filtered_toggles = {k: v for k, v in toggle_kwargs.items() if k in valid_toggle_keys}
    toggles = SimulationToggles2D(**filtered_toggles)

    params_kwargs = config_data.get('params', {})
    valid_param_keys = {f.name for f in fields(SimulationParams2D)}
    filtered_params = {k: v for k, v in params_kwargs.items() if k in valid_param_keys}
    filtered_params['Vf'] = Vf

    if not filtered_params.get('antennas'):
        filtered_params['antennas'] = [
            {"x1": 2.5, "y1": 0.2, "x2": 2.5, "y2": 1.2, "r": 0.05, "V_bias": Vf_antenne, "w_width": 0.4,
             "collection_eff": 0.80, "C": 2e-12, "R": 100e3},
            {"x1": 2.5, "y1": -0.2, "x2": 2.5, "y2": -1.2, "r": 0.05, "V_bias": Vf_antenne, "w_width": 0.4,
             "collection_eff": 0.80, "C": 2e-12, "R": 100e3}
        ]

    params = SimulationParams2D(**filtered_params)

    plot_kwargs = config_data.get('plotting', {})
    valid_plot_keys = {f.name for f in fields(PlottingConfig)}
    filtered_plot = {k: v for k, v in plot_kwargs.items() if k in valid_plot_keys}
    plot_config = PlottingConfig(**filtered_plot)

    return params, toggles, plot_config
