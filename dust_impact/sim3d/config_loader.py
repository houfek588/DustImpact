# -*- coding: utf-8 -*-
"""
Configuration loader and parser for 3D PIC simulation.
Parses JSON configuration files, normalizes parameter structures,
resolves file paths, and instantiates typed SimulationParams3D, SimulationToggles3D,
and PlottingConfig3D dataclasses.
"""

import os
import json
from dataclasses import fields
from typing import Tuple, List, Dict, Any, Optional

from dust_impact.geometry.config import (
    SpisSpacecraftConfig,
    SpisAntennaGeometry,
    SpisGeometryConfig,
    AnalyticalSpacecraftPart,
    AnalyticalAntennaGeometry,
    AnalyticalGeometryConfig,
    GeometryConfig,
    VTKFilesConfig,
)
from dust_impact.sim3d.params import (
    SimulationToggles3D,
    PlottingConfig3D,
    ImpactConfig,
    PhysicsConfig,
    PhysicConfig,
    NumericConfig,
    SimulationParams3D,
)

__all__ = [
    "setup_simulation_parameters_3d",
    "SimulationParams3D",
    "SimulationToggles3D",
    "PlottingConfig3D",
    "ImpactConfig",
    "PhysicsConfig",
    "PhysicConfig",
    "NumericConfig",
    "SpisSpacecraftConfig",
    "SpisAntennaGeometry",
    "SpisGeometryConfig",
    "AnalyticalSpacecraftPart",
    "AnalyticalAntennaGeometry",
    "AnalyticalGeometryConfig",
    "GeometryConfig",
    "VTKFilesConfig",
]


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


def _normalize_steps_and_time(p_kwargs: Dict[str, Any]) -> None:
    """Convert simulation_duration_s to num_time_steps if num_time_steps not explicitly set."""
    if 'simulation_duration_s' in p_kwargs and 'num_time_steps' not in p_kwargs:
        dt = float(p_kwargs.get('time_step_s', 2e-9))
        p_kwargs['num_time_steps'] = int(round(float(p_kwargs['simulation_duration_s']) / dt))


def _normalize_grid_and_domain(p_kwargs: Dict[str, Any]) -> None:
    """Normalize grid_nodes and domain_half_length_m vectors and flat aliases."""
    _normalize_steps_and_time(p_kwargs)

    # Normalize grid_nodes
    if 'grid_nodes' in p_kwargs:
        gn = p_kwargs['grid_nodes']
        if isinstance(gn, (list, tuple)) and len(gn) == 3:
            p_kwargs['grid_nodes'] = [int(v) for v in gn]
        elif isinstance(gn, (int, float)):
            p_kwargs['grid_nodes'] = [int(gn), int(gn), int(gn)]
    elif all(k in p_kwargs for k in ('grid_nodes_x', 'grid_nodes_y', 'grid_nodes_z')):
        p_kwargs['grid_nodes'] = [
            int(p_kwargs['grid_nodes_x']),
            int(p_kwargs['grid_nodes_y']),
            int(p_kwargs['grid_nodes_z'])
        ]

    # Normalize domain_half_length_m
    if 'domain_half_length_m' in p_kwargs:
        dhl = p_kwargs['domain_half_length_m']
        if isinstance(dhl, (list, tuple)) and len(dhl) == 3:
            p_kwargs['domain_half_length_m'] = [float(v) for v in dhl]
        elif isinstance(dhl, (int, float)):
            p_kwargs['domain_half_length_m'] = [float(dhl), float(dhl), float(dhl)]
    elif all(
        k in p_kwargs for k in ('domain_half_length_x_m', 'domain_half_length_y_m', 'domain_half_length_z_m')
    ):
        p_kwargs['domain_half_length_m'] = [
            float(p_kwargs['domain_half_length_x_m']),
            float(p_kwargs['domain_half_length_y_m']),
            float(p_kwargs['domain_half_length_z_m'])
        ]

    if 'grid_nodes' in p_kwargs:
        p_kwargs['grid_nodes_x'] = p_kwargs['grid_nodes'][0]
        p_kwargs['grid_nodes_y'] = p_kwargs['grid_nodes'][1]
        p_kwargs['grid_nodes_z'] = p_kwargs['grid_nodes'][2]
    if 'domain_half_length_m' in p_kwargs:
        p_kwargs['domain_half_length_x_m'] = p_kwargs['domain_half_length_m'][0]
        p_kwargs['domain_half_length_y_m'] = p_kwargs['domain_half_length_m'][1]
        p_kwargs['domain_half_length_z_m'] = p_kwargs['domain_half_length_m'][2]


def _parse_impact_config(cfg: Dict[str, Any], p_kwargs: Dict[str, Any]) -> ImpactConfig:
    """Extract and normalize ImpactConfig from JSON dictionary or flat kwargs."""
    impact_data = None
    if 'physics' in cfg and isinstance(cfg['physics'], dict) and 'impact' in cfg['physics']:
        impact_data = cfg['physics']['impact']
    elif 'impact' in cfg and isinstance(cfg['impact'], dict):
        impact_data = cfg['impact']
    elif 'impact' in p_kwargs and isinstance(p_kwargs['impact'], dict):
        impact_data = p_kwargs['impact']

    if impact_data:
        loc = impact_data.get('location', [-2.0, 2.0, 0.0])
        direc = impact_data.get('direction', [0.0, 0.0, 0.0])
        t_del = impact_data.get('time_delay_s', 1e-6)
        norm = impact_data.get('normal', None)

        impact_obj = ImpactConfig(
            location=list(loc), direction=list(direc), time_delay_s=float(t_del),
            normal=list(norm) if norm is not None else None
        )
        p_kwargs['impact'] = impact_obj
        p_kwargs['impact_location_xyz_m'] = list(loc)
        p_kwargs['impact_direction_vector'] = list(direc)
        p_kwargs['impact_time_delay_s'] = float(t_del)
        if norm is not None:
            p_kwargs['impact_normal'] = list(norm)
        return impact_obj

    if any(k in p_kwargs for k in ('impact_location_xyz_m', 'impact_direction_vector', 'impact_time_delay_s', 'impact_normal')):
        loc = p_kwargs.get('impact_location_xyz_m', [-2.0, 2.0, 0.0])
        direc = p_kwargs.get('impact_direction_vector', [0.0, 0.0, 0.0])
        t_del = p_kwargs.get('impact_time_delay_s', 1e-6)
        norm = p_kwargs.get('impact_normal', None)
        impact_obj = ImpactConfig(
            location=list(loc), direction=list(direc), time_delay_s=float(t_del),
            normal=list(norm) if norm is not None else None
        )
        p_kwargs['impact'] = impact_obj
        p_kwargs['impact_location_xyz_m'] = list(loc)
        p_kwargs['impact_direction_vector'] = list(direc)
        p_kwargs['impact_time_delay_s'] = float(t_del)
        if norm is not None:
            p_kwargs['impact_normal'] = list(norm)
        return impact_obj

    default_impact = ImpactConfig()
    p_kwargs['impact'] = default_impact
    return default_impact


def _parse_analytical_geometry(ana_data: Dict[str, Any], p_kwargs: Dict[str, Any]) -> Tuple[GeometryConfig, VTKFilesConfig]:
    """Parse analytical geometry specifications (spacecraft shapes and antenna wires)."""
    sc_data = ana_data.get('spacecraft', {})
    parts_list = []
    if 'parts' in sc_data and isinstance(sc_data['parts'], list):
        for p in sc_data['parts']:
            parts_list.append(AnalyticalSpacecraftPart(
                **{k: v for k, v in p.items() if k in {f.name for f in fields(AnalyticalSpacecraftPart)}}
            ))
    sc_kwargs = {k: v for k, v in sc_data.items() if k in {f.name for f in fields(AnalyticalSpacecraftPart)} and k != 'parts'}
    sc_part = AnalyticalSpacecraftPart(parts=parts_list, **sc_kwargs)
    if sc_part.effective_voltage is not None:
        p_kwargs['spacecraft_voltage_V'] = sc_part.effective_voltage

    ant_list = []
    extracted_voltages = []
    extracted_caps = []
    extracted_res = []
    has_caps = False
    has_res = False
    for a in ana_data.get('antennas', []):
        v_val = a.get('voltage_V', 0.0)
        v_float = float(v_val) if v_val is not None else 0.0
        c_val = a.get('capacitance_F', None)
        r_val = a.get('resistance_Ohm', None)

        ant_obj = AnalyticalAntennaGeometry(
            p_start=a.get('p_start', [0.0, 0.0, 0.0]),
            p_end=a.get('p_end', [1.0, 0.0, 0.0]),
            radius=float(a.get('radius', 0.015)),
            voltage_V=v_float,
            capacitance_F=float(c_val) if c_val is not None else None,
            resistance_Ohm=float(r_val) if r_val is not None else None
        )
        ant_list.append(ant_obj)
        extracted_voltages.append(v_float)
        if c_val is not None:
            has_caps = True
            extracted_caps.append(float(c_val))
        else:
            extracted_caps.append(2e-12)

        if r_val is not None:
            has_res = True
            extracted_res.append(float(r_val))
        else:
            extracted_res.append(100e3)

    p_kwargs['antenna_bias_voltage_V'] = extracted_voltages
    if has_caps:
        p_kwargs['antenna_capacitance_F'] = extracted_caps
    if has_res:
        p_kwargs['antenna_resistance_Ohm'] = extracted_res

    geom_config = GeometryConfig(
        source='analytical',
        analytical=AnalyticalGeometryConfig(spacecraft=sc_part, antennas=ant_list)
    )
    vtk_config = VTKFilesConfig(
        spis_background_potential_file="",
        spacecraft_weighting_file="",
        antenna_weighting_files=[]
    )
    return geom_config, vtk_config


def _parse_spis_geometry(geom_data: Dict[str, Any], base_dir: str, p_kwargs: Dict[str, Any]) -> Tuple[GeometryConfig, VTKFilesConfig]:
    """Parse SPIS VTK mesh configuration and individual antenna circuit definitions."""
    spis_data = geom_data.get('spis', {}) if isinstance(geom_data, dict) else {}
    bg_f = _resolve_path(base_dir, spis_data.get('background_potential_file', 'inputs/spis_V_bg.vtk'))
    thresh = float(spis_data.get('weighting_threshold', p_kwargs.get('weighting_threshold', 0.85)))
    p_kwargs['weighting_threshold'] = thresh

    if 'spacecraft' in spis_data and isinstance(spis_data['spacecraft'], dict):
        sc_data = spis_data['spacecraft']
        sc_f = _resolve_path(base_dir, sc_data.get('weighting_file', 'inputs/spis_Vw_body.vtk'))
        mesh_f = _resolve_path(base_dir, sc_data.get('surface_mesh_file', ''))
        v_sc = sc_data.get('voltage_V', None)
        if v_sc is not None:
            p_kwargs['spacecraft_voltage_V'] = float(v_sc)
        sc_obj = SpisSpacecraftConfig(weighting_file=sc_f, surface_mesh_file=mesh_f, voltage_V=float(v_sc) if v_sc is not None else None)
    else:
        sc_f = _resolve_path(base_dir, 'inputs/spis_Vw_body.vtk')
        mesh_f = ''
        sc_obj = SpisSpacecraftConfig(weighting_file=sc_f, surface_mesh_file=mesh_f)

    if 'antennas' in spis_data and isinstance(spis_data['antennas'], list):
        ant_list = []
        ant_fs = []
        extracted_caps = []
        extracted_res = []
        extracted_volts = []
        has_caps = False
        has_res = False
        has_volts = False
        for a in spis_data['antennas']:
            if isinstance(a, str):
                f_path = _resolve_path(base_dir, a)
                ant_list.append(SpisAntennaGeometry(weighting_file=f_path))
                ant_fs.append(f_path)
            elif isinstance(a, dict):
                f_path = _resolve_path(base_dir, a.get('weighting_file', ''))
                c_val = a.get('capacitance_F', None)
                r_val = a.get('resistance_Ohm', None)
                v_val = a.get('voltage_V', None)

                if c_val is not None:
                    has_caps = True
                    extracted_caps.append(float(c_val))
                else:
                    extracted_caps.append(2e-12)

                if r_val is not None:
                    has_res = True
                    extracted_res.append(float(r_val))
                else:
                    extracted_res.append(100e3)

                if v_val is not None:
                    has_volts = True
                    extracted_volts.append(float(v_val))
                else:
                    extracted_volts.append(0.0)

                ant_obj = SpisAntennaGeometry(
                    weighting_file=f_path,
                    voltage_V=float(v_val) if v_val is not None else None,
                    capacitance_F=float(c_val) if c_val is not None else None,
                    resistance_Ohm=float(r_val) if r_val is not None else None
                )
                ant_list.append(ant_obj)
                ant_fs.append(f_path)
        if has_caps:
            p_kwargs['antenna_capacitance_F'] = extracted_caps
        if has_res:
            p_kwargs['antenna_resistance_Ohm'] = extracted_res
        if has_volts:
            p_kwargs['antenna_bias_voltage_V'] = extracted_volts
    else:
        ant_fs = [_resolve_path(base_dir, item) for item in spis_data.get('antenna_weighting_files', [])]
        ant_list = [SpisAntennaGeometry(weighting_file=f) for f in ant_fs]

    spis_cfg = SpisGeometryConfig(
        background_potential_file=bg_f,
        spacecraft=sc_obj,
        antennas=ant_list,
        weighting_threshold=thresh,
        spacecraft_weighting_file=sc_f,
        spacecraft_surface_mesh_file=mesh_f,
        antenna_weighting_files=ant_fs
    )
    geom_config = GeometryConfig(source='spis', spis=spis_cfg)
    vtk_config = VTKFilesConfig(
        spis_background_potential_file=bg_f,
        spacecraft_weighting_file=sc_f,
        antenna_weighting_files=ant_fs
    )
    return geom_config, vtk_config


def _parse_legacy_vtk_geometry(vtk_data: Dict[str, Any], base_dir: str) -> Tuple[GeometryConfig, VTKFilesConfig]:
    """Parse legacy flat vtk_files dictionary into unified GeometryConfig."""
    bg_f = _resolve_path(base_dir, vtk_data.get('spis_background_potential_file', 'inputs/spis_V_bg.vtk'))
    sc_f = _resolve_path(base_dir, vtk_data.get('spacecraft_weighting_file', 'inputs/spis_Vw_body.vtk'))
    ant_fs = [_resolve_path(base_dir, item) for item in vtk_data.get('antenna_weighting_files', [])]
    vtk_cfg = VTKFilesConfig(
        spis_background_potential_file=bg_f,
        spacecraft_weighting_file=sc_f,
        antenna_weighting_files=ant_fs
    )
    geom_cfg = GeometryConfig(
        source='spis',
        spis=SpisGeometryConfig(
            background_potential_file=bg_f,
            spacecraft_weighting_file=sc_f,
            antenna_weighting_files=ant_fs
        )
    )
    return geom_cfg, vtk_cfg


def _parse_geometry_and_vtk(cfg: Dict[str, Any], base_dir: str, p_kwargs: Dict[str, Any]) -> Tuple[GeometryConfig, VTKFilesConfig]:
    """Resolve geometry configuration from top-level or nested config blocks."""
    geom_data = cfg.get('geometry', p_kwargs.get('geometry', None))
    if geom_data and isinstance(geom_data, dict):
        source = str(geom_data.get('source', 'spis')).lower()
        if source == 'analytical':
            return _parse_analytical_geometry(geom_data.get('analytical', {}), p_kwargs)
        return _parse_spis_geometry(geom_data, base_dir, p_kwargs)

    if (
        ('vtk_files' in cfg and isinstance(cfg['vtk_files'], dict)) or
        ('vtk_files' in p_kwargs and isinstance(p_kwargs['vtk_files'], dict))
    ):
        vtk_data = cfg.get('vtk_files', p_kwargs.get('vtk_files', {}))
        return _parse_legacy_vtk_geometry(vtk_data, base_dir)

    default_vtk = VTKFilesConfig()
    return _parse_legacy_vtk_geometry({
        'spis_background_potential_file': default_vtk.spis_background_potential_file,
        'spacecraft_weighting_file': default_vtk.spacecraft_weighting_file,
        'antenna_weighting_files': default_vtk.antenna_weighting_files,
    }, base_dir)


def _resolve_output_paths(plot_config: PlottingConfig3D, base_dir: str, output_dir: Optional[str]) -> None:
    """Redirect output filenames to output_dir if specified, otherwise resolve relative to base_dir."""
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


def setup_simulation_parameters_3d(
    Vf: float,
    Vf_antenne: float = 0.0,
    config_file: str = "config.json",
    output_dir: Optional[str] = None
) -> Tuple[SimulationParams3D, SimulationToggles3D, PlottingConfig3D]:
    """
    Loads and resolves configuration for 3D PIC simulation.
    Handles legacy and modern config JSON structures, normalizes vector parameters,
    and returns initialized SimulationParams3D, SimulationToggles3D, and PlottingConfig3D.
    """
    if not os.path.exists(config_file) and os.path.exists("inputs/config.json"):
        config_file = "inputs/config.json"

    if not os.path.exists(config_file):
        config_file = "config.json"

    config_abs_path = os.path.abspath(config_file)
    base_dir = os.path.dirname(config_abs_path)

    with open(config_file, 'r', encoding='utf-8') as f:
        cfg = json.load(f)

    # 1. Toggles
    toggle_kwargs = cfg.get('toggles', {})
    valid_toggle_keys = {f.name for f in fields(SimulationToggles3D)}
    filtered_toggles = {k: v for k, v in toggle_kwargs.items() if k in valid_toggle_keys}
    toggles = SimulationToggles3D(**filtered_toggles)

    # 2. Extract params dictionary from physics and numeric sections
    p_kwargs = {}
    if 'physics' in cfg and isinstance(cfg['physics'], dict):
        p_kwargs.update(cfg['physics'])
    if 'numeric' in cfg and isinstance(cfg['numeric'], dict):
        p_kwargs.update(cfg['numeric'])

    # 3. Normalization of steps, grid, domain, and impact
    _normalize_grid_and_domain(p_kwargs)
    _parse_impact_config(cfg, p_kwargs)

    # 4. Geometry and VTK configuration
    geom_cfg, vtk_cfg = _parse_geometry_and_vtk(cfg, base_dir, p_kwargs)
    p_kwargs['geometry'] = geom_cfg
    p_kwargs['vtk_files'] = vtk_cfg

    # 5. Build SimulationParams3D
    valid_param_keys = {f.name for f in fields(SimulationParams3D) if f.init}
    filtered_params = {k: v for k, v in p_kwargs.items() if k in valid_param_keys}
    filtered_params['Vf'] = Vf

    physics_kwargs = {k: v for k, v in p_kwargs.items() if k in {f.name for f in fields(PhysicsConfig)}}
    numeric_kwargs = {k: v for k, v in p_kwargs.items() if k in {f.name for f in fields(NumericConfig)}}
    filtered_params['physics'] = PhysicsConfig(**physics_kwargs)
    filtered_params['numeric'] = NumericConfig(**numeric_kwargs)

    params = SimulationParams3D(**filtered_params)

    # 6. Plotting configuration and path resolution
    plot_kwargs = cfg.get('plotting', {})
    valid_plot_keys = {f.name for f in fields(PlottingConfig3D)}
    filtered_plot = {k: v for k, v in plot_kwargs.items() if k in valid_plot_keys}
    plot_config = PlottingConfig3D(**filtered_plot)
    _resolve_output_paths(plot_config, base_dir, output_dir)

    return params, toggles, plot_config
