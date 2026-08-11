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


from dust_impact.common.config import BaseSimulationParams, BaseSimulationToggles


@dataclass
class SimulationToggles3D(BaseSimulationToggles):
    pass


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
class SimulationParams3D(BaseSimulationParams):
    vtk_files: VTKFilesConfig = field(default_factory=VTKFilesConfig)

    domain_half_length_x_m: float = 5.0
    domain_half_length_y_m: float = 5.0
    domain_half_length_z_m: float = 5.0

    impact_location_xyz_m: List[float] = field(default_factory=lambda: [-2.0, 2.0, 0.0])
    impact_normal: List[float] = field(default_factory=lambda: [-0.7071, 0.7071, 0.0])

    grid_nodes_x: int = 35
    grid_nodes_y: int = 35
    grid_nodes_z: int = 35

    antenna_capacitance_F: List[float] = field(default_factory=lambda: [2e-12, 2e-12, 2e-12])
    antenna_resistance_Ohm: List[float] = field(default_factory=lambda: [100e3, 100e3, 100e3])
    antenna_bias_voltage_V: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    antenna_collection_efficiency: List[float] = field(default_factory=lambda: [0.8, 0.8, 0.8])

    # Derived attributes specific to 3D
    dx: float = field(init=False)
    dy: float = field(init=False)
    dz: float = field(init=False)
    x_grid: np.ndarray = field(init=False)
    y_grid: np.ndarray = field(init=False)
    z_grid: np.ndarray = field(init=False)

    # 3D Convenience aliases
    Nx: int = field(init=False)
    Ny: int = field(init=False)
    Nz: int = field(init=False)
    L_x: float = field(init=False)
    L_y: float = field(init=False)
    L_z: float = field(init=False)
    impact_pos: List[float] = field(init=False)
    C_ant: List[float] = field(init=False)
    R_ant: List[float] = field(init=False)
    V_bias: List[float] = field(init=False)
    collection_eff: List[float] = field(init=False)

    def __post_init__(self):
        self._init_base_derived_params()

        self.x_grid = np.linspace(-self.domain_half_length_x_m, self.domain_half_length_x_m, self.grid_nodes_x)
        self.y_grid = np.linspace(-self.domain_half_length_y_m, self.domain_half_length_y_m, self.grid_nodes_y)
        self.z_grid = np.linspace(-self.domain_half_length_z_m, self.domain_half_length_z_m, self.grid_nodes_z)

        self.dx = self.x_grid[1] - self.x_grid[0]
        self.dy = self.y_grid[1] - self.y_grid[0]
        self.dz = self.z_grid[1] - self.z_grid[0]

        self.Nx = self.grid_nodes_x
        self.Ny = self.grid_nodes_y
        self.Nz = self.grid_nodes_z
        self.L_x = self.domain_half_length_x_m
        self.L_y = self.domain_half_length_y_m
        self.L_z = self.domain_half_length_z_m
        self.impact_pos = self.impact_location_xyz_m
        self.C_ant = self.antenna_capacitance_F
        self.R_ant = self.antenna_resistance_Ohm
        self.V_bias = self.antenna_bias_voltage_V
        self.collection_eff = self.antenna_collection_efficiency


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
