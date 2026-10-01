# -*- coding: utf-8 -*-
"""
Configuration loader and parameter dataclasses for 3D PIC simulation.
"""

import os
import json
import numpy as np
from dataclasses import dataclass, field, asdict, fields
from typing import Tuple, List, Dict, Any, Optional
from dust_impact.physics.constants import amu, e, m_e, eps_0
from dust_impact.common.io import ensure_dir


from dust_impact.common.config import BaseSimulationParams, BaseSimulationToggles


@dataclass
class SimulationToggles3D(BaseSimulationToggles):
    pass


@dataclass
class SpisGeometryConfig:
    background_potential_file: str = "inputs/spis_V_bg.vtk"
    spacecraft_weighting_file: str = "inputs/spis_Vw_body.vtk"
    antenna_weighting_files: List[str] = field(default_factory=lambda: [
        "inputs/spis_Vw_ant1.vtk",
        "inputs/spis_Vw_ant2.vtk",
        "inputs/spis_Vw_ant3.vtk"
    ])
    spacecraft_surface_mesh_file: str = ""
    weighting_threshold: float = 0.85

    @property
    def spis_background_potential_file(self) -> str:
        return self.background_potential_file


@dataclass
class AnalyticalSpacecraftPart:
    type: str = "sphere"  # "sphere", "box", "cube", "cylinder", "composite"
    center: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    radius: float = 1.0
    dimensions: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    side_length: Optional[float] = None
    p_start: List[float] = field(default_factory=lambda: [0.0, 0.0, -1.0])
    p_end: List[float] = field(default_factory=lambda: [0.0, 0.0, 1.0])
    parts: List['AnalyticalSpacecraftPart'] = field(default_factory=list)


@dataclass
class AnalyticalAntennaGeometry:
    p_start: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    p_end: List[float] = field(default_factory=lambda: [1.0, 0.0, 0.0])
    radius: float = 0.015
    voltage_V: Optional[float] = 0.0
    potential_V: Optional[float] = None
    bias_voltage_V: Optional[float] = None

    @property
    def effective_voltage(self) -> float:
        if self.potential_V is not None:
            return float(self.potential_V)
        if self.bias_voltage_V is not None:
            return float(self.bias_voltage_V)
        if self.voltage_V is not None:
            return float(self.voltage_V)
        return 0.0


@dataclass
class AnalyticalGeometryConfig:
    spacecraft: AnalyticalSpacecraftPart = field(default_factory=AnalyticalSpacecraftPart)
    antennas: List[AnalyticalAntennaGeometry] = field(default_factory=list)


@dataclass
class GeometryConfig:
    source: str = "spis"  # "spis" | "analytical"
    spis: SpisGeometryConfig = field(default_factory=SpisGeometryConfig)
    analytical: AnalyticalGeometryConfig = field(default_factory=AnalyticalGeometryConfig)


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

    output_format: str = "h5"
    output_h5_filepath: str = "outputs/out_3d_vysledky.h5"
    output_npz_filepath: str = "outputs/out_3d_vysledky.npz"
    output_csv_filepath: str = "outputs/out_3d_vysledky_simulace.csv"
    
    enable_checkpointing: bool = True
    checkpoint_interval_steps: int = 500
    
    export_vtk: bool = False
    vtk_output_dir: str = "outputs/paraview_vtk"

    file_currents: str = "outputs/out_3d_proudy_napeti.png"
    file_fields_anim: str = "outputs/out_3d_animace_pole_potencial.gif"
    file_particles_anim: str = "outputs/out_3d_animace_pozice_castic.gif"
    file_velocity_anim: str = "outputs/out_3d_animace_rychlosti.gif"

    @property
    def primary_output_filepath(self) -> str:
        if getattr(self, 'output_format', 'h5') == 'h5':
            return self.output_h5_filepath
        return self.output_npz_filepath


@dataclass
class SimulationParams3D(BaseSimulationParams):
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    vtk_files: VTKFilesConfig = field(default_factory=VTKFilesConfig)

    domain_half_length_x_m: float = 5.0
    domain_half_length_y_m: float = 5.0
    domain_half_length_z_m: float = 5.0

    impact_location_xyz_m: List[float] = field(default_factory=lambda: [-2.0, 2.0, 0.0])
    impact_direction_vector: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    impact_normal: List[float] = field(default_factory=lambda: [-0.7071, 0.7071, 0.0])

    grid_nodes_x: int = 35
    grid_nodes_y: int = 35
    grid_nodes_z: int = 35

    antenna_capacitance_F: List[float] = field(default_factory=lambda: [2e-12, 2e-12, 2e-12])
    antenna_resistance_Ohm: List[float] = field(default_factory=lambda: [100e3, 100e3, 100e3])
    antenna_bias_voltage_V: List[float] = field(default_factory=list)
    antenna_collection_efficiency: List[float] = field(default_factory=lambda: [0.8, 0.8, 0.8])
    antenna_weighting_threshold: float = 0.2

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

        # Synchronize geometry and vtk_files for bidirectional compatibility
        if getattr(self.geometry, 'source', 'spis') == 'spis':
            default_sc = "inputs/spis_Vw_body.vtk"
            default_bg = "inputs/spis_V_bg.vtk"
            # If vtk_files was customized, sync to geometry.spis
            if (getattr(self.vtk_files, 'spacecraft_weighting_file', '') != default_sc or
                getattr(self.vtk_files, 'spis_background_potential_file', '') != default_bg):
                self.geometry.spis.spacecraft_weighting_file = self.vtk_files.spacecraft_weighting_file
                self.geometry.spis.background_potential_file = self.vtk_files.spis_background_potential_file
                self.geometry.spis.antenna_weighting_files = self.vtk_files.antenna_weighting_files
            else:
                self.vtk_files.spacecraft_weighting_file = self.geometry.spis.spacecraft_weighting_file
                self.vtk_files.spis_background_potential_file = self.geometry.spis.background_potential_file
                self.vtk_files.antenna_weighting_files = self.geometry.spis.antenna_weighting_files

            if not self.antenna_bias_voltage_V:
                self.antenna_bias_voltage_V = [0.0] * len(self.antenna_capacitance_F)
        elif getattr(self.geometry, 'source', 'spis') == 'analytical':
            ant_geoms = getattr(getattr(self.geometry, 'analytical', None), 'antennas', [])
            if ant_geoms:
                self.antenna_bias_voltage_V = [
                    float(getattr(a, 'effective_voltage', 0.0))
                    for a in ant_geoms
                ]
            elif not self.antenna_bias_voltage_V:
                self.antenna_bias_voltage_V = [0.0] * len(self.antenna_capacitance_F)

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

        if getattr(self, 'plasma_injection_mode', 'point_cloud') == 'homogeneous':
            domain_vol = (2.0 * self.L_x) * (2.0 * self.L_y) * (2.0 * self.L_z)
            self.q_macro = (self.solar_wind_density_m3 * domain_vol * e) / self.num_macroparticles


def _resolve_path(base_dir: str, path: str) -> str:
    """Resolve path relative to config file directory if relative, falling back to CWD."""
    if not path or os.path.isabs(path):
        return path
    candidate = os.path.normpath(os.path.join(base_dir, path))
    if os.path.exists(candidate):
        return candidate
    if os.path.exists(path):
        return os.path.abspath(path)
    return candidate


def setup_simulation_parameters_3d(
    Vf: float,
    Vf_antenne: float = 0.0,
    config_file: str = "config.json",
    output_dir: str = None
) -> Tuple[SimulationParams3D, SimulationToggles3D, PlottingConfig3D]:
    """ Loads configuration for 3D PIC simulation using self-explanatory parameters. """
    if not os.path.exists(config_file) and os.path.exists("inputs/config.json"):
        config_file = "inputs/config.json"

    if not os.path.exists(config_file):
        config_file = "config.json"

    config_abs_path = os.path.abspath(config_file)
    base_dir = os.path.dirname(config_abs_path)

    with open(config_file, 'r', encoding='utf-8') as f:
        cfg = json.load(f)

    toggle_kwargs = cfg.get('toggles', {})
    valid_toggle_keys = {f.name for f in fields(SimulationToggles3D)}
    filtered_toggles = {k: v for k, v in toggle_kwargs.items() if k in valid_toggle_keys}
    toggles = SimulationToggles3D(**filtered_toggles)

    p_kwargs = cfg.get('params', {})

    # Parse geometry configuration (top-level "geometry" or inside "params.geometry")
    geom_data = cfg.get('geometry', p_kwargs.get('geometry', None))
    if geom_data and isinstance(geom_data, dict):
        source = str(geom_data.get('source', 'spis')).lower()
        if source == 'analytical':
            ana_data = geom_data.get('analytical', {})
            sc_data = ana_data.get('spacecraft', {})
            parts_list = []
            if 'parts' in sc_data and isinstance(sc_data['parts'], list):
                for p in sc_data['parts']:
                    parts_list.append(AnalyticalSpacecraftPart(
                        **{k: v for k, v in p.items() if k in {f.name for f in fields(AnalyticalSpacecraftPart)}}
                    ))
            sc_kwargs = {k: v for k, v in sc_data.items() if k in {f.name for f in fields(AnalyticalSpacecraftPart)} and k != 'parts'}
            sc_part = AnalyticalSpacecraftPart(parts=parts_list, **sc_kwargs)

            ant_list = []
            extracted_voltages = []
            for a in ana_data.get('antennas', []):
                v_val = a.get('voltage_V', a.get('potential_V', a.get('bias_voltage_V', a.get('voltage', a.get('potential', None)))))
                v_float = float(v_val) if v_val is not None else 0.0
                ant_obj = AnalyticalAntennaGeometry(
                    p_start=a.get('p_start', [0.0, 0.0, 0.0]),
                    p_end=a.get('p_end', [1.0, 0.0, 0.0]),
                    radius=float(a.get('radius', 0.015)),
                    voltage_V=v_float,
                    potential_V=v_float,
                    bias_voltage_V=v_float
                )
                ant_list.append(ant_obj)
                extracted_voltages.append(v_float)

            p_kwargs['antenna_bias_voltage_V'] = extracted_voltages

            geom_config = GeometryConfig(
                source='analytical',
                analytical=AnalyticalGeometryConfig(spacecraft=sc_part, antennas=ant_list)
            )
            p_kwargs['geometry'] = geom_config
            p_kwargs['vtk_files'] = VTKFilesConfig(
                spis_background_potential_file="",
                spacecraft_weighting_file="",
                antenna_weighting_files=[]
            )
        else:
            spis_data = geom_data.get('spis', {})
            bg_f = _resolve_path(base_dir, spis_data.get('background_potential_file', spis_data.get('spis_background_potential_file', 'inputs/spis_V_bg.vtk')))
            sc_f = _resolve_path(base_dir, spis_data.get('spacecraft_weighting_file', 'inputs/spis_Vw_body.vtk'))
            ant_fs = [_resolve_path(base_dir, item) for item in spis_data.get('antenna_weighting_files', [])]
            mesh_f = _resolve_path(base_dir, spis_data.get('spacecraft_surface_mesh_file', ''))
            thresh = float(spis_data.get('weighting_threshold', 0.85))

            spis_cfg = SpisGeometryConfig(
                background_potential_file=bg_f,
                spacecraft_weighting_file=sc_f,
                antenna_weighting_files=ant_fs,
                spacecraft_surface_mesh_file=mesh_f,
                weighting_threshold=thresh
            )
            p_kwargs['geometry'] = GeometryConfig(source='spis', spis=spis_cfg)
            p_kwargs['vtk_files'] = VTKFilesConfig(
                spis_background_potential_file=bg_f,
                spacecraft_weighting_file=sc_f,
                antenna_weighting_files=ant_fs
            )
    elif 'vtk_files' in p_kwargs and isinstance(p_kwargs['vtk_files'], dict):
        vtk_data = p_kwargs['vtk_files']
        bg_f = _resolve_path(base_dir, vtk_data.get('spis_background_potential_file', 'inputs/spis_V_bg.vtk'))
        sc_f = _resolve_path(base_dir, vtk_data.get('spacecraft_weighting_file', 'inputs/spis_Vw_body.vtk'))
        ant_fs = [_resolve_path(base_dir, item) for item in vtk_data.get('antenna_weighting_files', [])]
        vtk_cfg = VTKFilesConfig(
            spis_background_potential_file=bg_f,
            spacecraft_weighting_file=sc_f,
            antenna_weighting_files=ant_fs
        )
        p_kwargs['vtk_files'] = vtk_cfg
        p_kwargs['geometry'] = GeometryConfig(
            source='spis',
            spis=SpisGeometryConfig(
                background_potential_file=bg_f,
                spacecraft_weighting_file=sc_f,
                antenna_weighting_files=ant_fs
            )
        )
    else:
        vtk_cfg = VTKFilesConfig()
        vtk_cfg.spis_background_potential_file = _resolve_path(base_dir, vtk_cfg.spis_background_potential_file)
        vtk_cfg.spacecraft_weighting_file = _resolve_path(base_dir, vtk_cfg.spacecraft_weighting_file)
        vtk_cfg.antenna_weighting_files = [_resolve_path(base_dir, item) for item in vtk_cfg.antenna_weighting_files]
        p_kwargs['vtk_files'] = vtk_cfg
        p_kwargs['geometry'] = GeometryConfig(
            source='spis',
            spis=SpisGeometryConfig(
                background_potential_file=vtk_cfg.spis_background_potential_file,
                spacecraft_weighting_file=vtk_cfg.spacecraft_weighting_file,
                antenna_weighting_files=vtk_cfg.antenna_weighting_files
            )
        )

    valid_param_keys = {f.name for f in fields(SimulationParams3D) if f.init}
    filtered_params = {k: v for k, v in p_kwargs.items() if k in valid_param_keys}
    filtered_params['Vf'] = Vf

    params = SimulationParams3D(**filtered_params)

    plot_kwargs = cfg.get('plotting', {})
    valid_plot_keys = {f.name for f in fields(PlottingConfig3D)}
    filtered_plot = {k: v for k, v in plot_kwargs.items() if k in valid_plot_keys}
    plot_config = PlottingConfig3D(**filtered_plot)

    output_path_keys = [
        'output_h5_filepath',
        'output_npz_filepath',
        'output_csv_filepath',
        'file_currents',
        'file_fields_anim',
        'file_particles_anim',
        'file_velocity_anim'
    ]

    if output_dir:
        abs_output_dir = os.path.abspath(output_dir)
        os.makedirs(abs_output_dir, exist_ok=True)
        for key in output_path_keys:
            val = getattr(plot_config, key, None)
            if val:
                fname = os.path.basename(val)
                setattr(plot_config, key, os.path.join(abs_output_dir, fname))
        if plot_config.vtk_output_dir:
            v_dir_name = os.path.basename(plot_config.vtk_output_dir)
            plot_config.vtk_output_dir = os.path.join(abs_output_dir, v_dir_name)
    else:
        for key in output_path_keys:
            val = getattr(plot_config, key, None)
            if val:
                setattr(plot_config, key, _resolve_path(base_dir, val))
        if plot_config.vtk_output_dir and not os.path.isabs(plot_config.vtk_output_dir):
            plot_config.vtk_output_dir = os.path.join(base_dir, plot_config.vtk_output_dir)

    return params, toggles, plot_config

