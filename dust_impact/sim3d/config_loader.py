# -*- coding: utf-8 -*-
"""
Configuration loader and parameter dataclasses for 3D PIC simulation.
"""

import os
import json
import numpy as np
from dataclasses import dataclass, field, asdict, fields
from typing import Tuple, List, Dict, Any
from dust_impact.physics.constants import amu, e, m_e, eps_0
from dust_impact.common.io import ensure_dir


@dataclass
class SimulationToggles3D:
    enable_spis_background_field: bool = True
    enable_plasma_self_field: bool = True
    enable_antenna_particle_collection: bool = True
    enable_rc_circuit_response: bool = True
    enable_debye_screening: bool = True
    enable_antenna_bias_voltage: bool = True


@dataclass
class VTKFilesConfig:
    spis_background_potential_file: str = "inputs/spis_V_bg.vtk"
    spacecraft_weighting_file: str = "inputs/spis_Vw_body.vtk"
    antenna_weighting_files: List[str] = field(default_factory=lambda: [
        "inputs/spis_Vw_ant1.vtk",
        "inputs/spis_Vw_ant2.vtk",
        "inputs/spis_Vw_ant3.vtk"
    ])


@dataclass
class PlottingConfig3D:
    run_physical_simulation: bool = True
    show_interactive_gui_windows: bool = False
    save_plots_to_disk: bool = True
    export_csv_time_series: bool = False

    show_currents: bool = True
    show_fields_slice: bool = True
    show_particles_3d: bool = True
    show_velocity_anim: bool = True

    output_npz_filepath: str = "outputs/out_3d_vysledky.npz"
    output_csv_filepath: str = "outputs/out_3d_vysledky_simulace.csv"
    file_currents: str = "outputs/out_3d_proudy_napeti.png"
    file_fields_anim: str = "outputs/out_3d_animace_pole_potencial.gif"
    file_particles_anim: str = "outputs/out_3d_animace_pozice_castic.gif"
    file_velocity_anim: str = "outputs/out_3d_animace_rychlosti.gif"


@dataclass
class SimulationParams3D:
    Vf: float = 0.0
    Vf_antenne: float = 0.0
    vtk_files: VTKFilesConfig = field(default_factory=VTKFilesConfig)

    ion_mass_amu: float = 27.0
    impact_cloud_temperature_eV: float = 2.0
    num_macroparticles: int = 20000

    time_step_s: float = 2e-9
    simulation_duration_s: float = 20e-6
    impact_time_delay_s: float = 1e-6

    domain_half_length_x_m: float = 5.0
    domain_half_length_y_m: float = 5.0
    domain_half_length_z_m: float = 5.0

    impact_location_xyz_m: List[float] = field(default_factory=lambda: [-2.0, 2.0, 0.0])
    impact_normal: List[float] = field(default_factory=lambda: [-0.7071, 0.7071, 0.0])

    grid_nodes_x: int = 35
    grid_nodes_y: int = 35
    grid_nodes_z: int = 35

    solar_wind_electron_temp_eV: float = 15.0
    solar_wind_density_m3: float = 1e7

    antenna_capacitance_F: List[float] = field(default_factory=lambda: [2e-12, 2e-12, 2e-12])
    antenna_resistance_Ohm: List[float] = field(default_factory=lambda: [100e3, 100e3, 100e3])
    antenna_bias_voltage_V: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    antenna_collection_efficiency: List[float] = field(default_factory=lambda: [0.8, 0.8, 0.8])

    # Derived attributes
    m_i: float = field(init=False)
    debye_length: float = field(init=False)
    q_macro: float = field(init=False)
    steps: int = field(init=False)
    time_array: np.ndarray = field(init=False)
    dx: float = field(init=False)
    dy: float = field(init=False)
    dz: float = field(init=False)
    x_grid: np.ndarray = field(init=False)
    y_grid: np.ndarray = field(init=False)
    z_grid: np.ndarray = field(init=False)
    plot_stride: int = field(init=False)
    save_interval: int = field(init=False)

    # Convenience aliases
    Nx: int = field(init=False)
    Ny: int = field(init=False)
    Nz: int = field(init=False)
    dt: float = field(init=False)
    t_max: float = field(init=False)
    t_delay: float = field(init=False)
    L_x: float = field(init=False)
    L_y: float = field(init=False)
    L_z: float = field(init=False)
    impact_pos: List[float] = field(init=False)
    C_ant: List[float] = field(init=False)
    R_ant: List[float] = field(init=False)
    V_bias: List[float] = field(init=False)
    collection_eff: List[float] = field(init=False)
    N_particles: int = field(init=False)
    T_dust_eV: float = field(init=False)

    def __post_init__(self):
        self.m_i = self.ion_mass_amu * amu
        self.debye_length = np.sqrt((eps_0 * self.solar_wind_electron_temp_eV * e) / (self.solar_wind_density_m3 * e ** 2))
        self.q_macro = 50e-12 / self.num_macroparticles
        self.steps = int(self.simulation_duration_s / self.time_step_s)
        self.time_array = np.linspace(0, self.simulation_duration_s, self.steps)

        self.x_grid = np.linspace(-self.domain_half_length_x_m, self.domain_half_length_x_m, self.grid_nodes_x)
        self.y_grid = np.linspace(-self.domain_half_length_y_m, self.domain_half_length_y_m, self.grid_nodes_y)
        self.z_grid = np.linspace(-self.domain_half_length_z_m, self.domain_half_length_z_m, self.grid_nodes_z)

        self.dx = self.x_grid[1] - self.x_grid[0]
        self.dy = self.y_grid[1] - self.y_grid[0]
        self.dz = self.z_grid[1] - self.z_grid[0]

        self.Nx = self.grid_nodes_x
        self.Ny = self.grid_nodes_y
        self.Nz = self.grid_nodes_z
        self.dt = self.time_step_s
        self.t_max = self.simulation_duration_s
        self.t_delay = self.impact_time_delay_s
        self.L_x = self.domain_half_length_x_m
        self.L_y = self.domain_half_length_y_m
        self.L_z = self.domain_half_length_z_m
        self.impact_pos = self.impact_location_xyz_m
        self.C_ant = self.antenna_capacitance_F
        self.R_ant = self.antenna_resistance_Ohm
        self.V_bias = self.antenna_bias_voltage_V
        self.collection_eff = self.antenna_collection_efficiency
        self.N_particles = self.num_macroparticles
        self.T_dust_eV = self.impact_cloud_temperature_eV

        self.plot_stride = max(1, self.num_macroparticles // 1500)
        self.save_interval = max(1, self.steps // 50)


def setup_simulation_parameters_3d(Vf: float, Vf_antenne: float = 0.0, config_file: str = "config.json") -> Tuple[SimulationParams3D, SimulationToggles3D, PlottingConfig3D]:
    """ Loads configuration for 3D PIC simulation using self-explanatory parameters. """
    if not os.path.exists(config_file) and os.path.exists("inputs/config.json"):
        config_file = "inputs/config.json"

    if not os.path.exists(config_file):
        config_file = "config.json"

    with open(config_file, 'r', encoding='utf-8') as f:
        cfg = json.load(f)

    toggle_kwargs = cfg.get('toggles', {})
    valid_toggle_keys = {f.name for f in fields(SimulationToggles3D)}
    filtered_toggles = {k: v for k, v in toggle_kwargs.items() if k in valid_toggle_keys}
    toggles = SimulationToggles3D(**filtered_toggles)

    p_kwargs = cfg.get('params', {})
    if 'vtk_files' in p_kwargs and isinstance(p_kwargs['vtk_files'], dict):
        valid_vtk_keys = {f.name for f in fields(VTKFilesConfig)}
        filtered_vtk = {k: v for k, v in p_kwargs['vtk_files'].items() if k in valid_vtk_keys}
        p_kwargs['vtk_files'] = VTKFilesConfig(**filtered_vtk)

    valid_param_keys = {f.name for f in fields(SimulationParams3D)}
    filtered_params = {k: v for k, v in p_kwargs.items() if k in valid_param_keys}
    filtered_params['Vf'] = Vf

    params = SimulationParams3D(**filtered_params)

    plot_kwargs = cfg.get('plotting', {})
    valid_plot_keys = {f.name for f in fields(PlottingConfig3D)}
    filtered_plot = {k: v for k, v in plot_kwargs.items() if k in valid_plot_keys}
    plot_config = PlottingConfig3D(**filtered_plot)

    return params, toggles, plot_config
