# -*- coding: utf-8 -*-
"""
Configuration loader and parser for 3D PIC simulation.
Validates input JSON using declarative Pydantic v2 schemas (fail-fast),
resolves file paths, and instantiates typed SimulationParams3D, SimulationToggles3D,
and PlottingConfig3D dataclasses.
"""

import os
import json
from typing import Tuple, Optional

from pydantic import ValidationError

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
from dust_impact.solver.params import (
    SimulationToggles3D,
    PlottingConfig3D,
    ImpactConfig,
    PhysicsConfig,
    PhysicConfig,
    NumericConfig,
    SimulationParams3D,
)
from dust_impact.solver.schema import (
    SimulationConfigSchema,
    AnalyticalSpacecraftPartSchema,
    AnalyticalGeometrySchema,
    SpisGeometrySchema,
    format_validation_error,
)

__all__ = [
    "setup_simulation_parameters_3d",
    "load_and_validate_config",
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
    "SimulationConfigSchema",
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


def _build_analytical_geometry(ana_schema: AnalyticalGeometrySchema) -> Tuple[GeometryConfig, VTKFilesConfig]:
    """Convert validated AnalyticalGeometrySchema into runtime GeometryConfig and VTKFilesConfig."""
    def _convert_part(p: AnalyticalSpacecraftPartSchema) -> AnalyticalSpacecraftPart:
        subparts = [_convert_part(sp) for sp in p.parts]
        return AnalyticalSpacecraftPart(
            type=p.type,
            center=list(p.center),
            radius=float(p.radius),
            dimensions=list(p.dimensions),
            side_length=float(p.side_length) if p.side_length is not None else None,
            p_start=list(p.p_start),
            p_end=list(p.p_end),
            voltage_V=float(p.voltage_V) if p.voltage_V is not None else None,
            parts=subparts
        )

    sc_part = _convert_part(ana_schema.spacecraft)

    ant_list = []
    for a in ana_schema.antennas:
        ant_list.append(AnalyticalAntennaGeometry(
            p_start=list(a.p_start),
            p_end=list(a.p_end),
            radius=float(a.radius),
            voltage_V=float(a.voltage_V),
            capacitance_F=float(a.capacitance_F) if a.capacitance_F is not None else None,
            resistance_Ohm=float(a.resistance_Ohm) if a.resistance_Ohm is not None else None
        ))

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


def _build_spis_geometry(spis_schema: SpisGeometrySchema, base_dir: str) -> Tuple[GeometryConfig, VTKFilesConfig]:
    """Convert validated SpisGeometrySchema into runtime GeometryConfig and VTKFilesConfig."""
    bg_f = _resolve_path(base_dir, spis_schema.background_potential_file)
    thresh = float(spis_schema.weighting_threshold)

    sc_f = _resolve_path(base_dir, spis_schema.spacecraft.weighting_file)
    mesh_f = _resolve_path(base_dir, spis_schema.spacecraft.surface_mesh_file)
    v_sc = spis_schema.spacecraft.voltage_V
    sc_obj = SpisSpacecraftConfig(
        weighting_file=sc_f,
        surface_mesh_file=mesh_f,
        voltage_V=float(v_sc) if v_sc is not None else None
    )

    ant_list = []
    ant_fs = []
    for a in spis_schema.antennas:
        if isinstance(a, str):
            f_path = _resolve_path(base_dir, a)
            ant_list.append(SpisAntennaGeometry(weighting_file=f_path))
            ant_fs.append(f_path)
        else:
            f_path = _resolve_path(base_dir, a.weighting_file)
            ant_obj = SpisAntennaGeometry(
                weighting_file=f_path,
                voltage_V=float(a.voltage_V) if a.voltage_V is not None else None,
                capacitance_F=float(a.capacitance_F) if a.capacitance_F is not None else None,
                resistance_Ohm=float(a.resistance_Ohm) if a.resistance_Ohm is not None else None
            )
            ant_list.append(ant_obj)
            ant_fs.append(f_path)

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


def load_and_validate_config(config_file: str) -> Tuple[SimulationConfigSchema, str]:
    """
    Reads and validates a JSON simulation configuration using Pydantic v2.
    Raises ValueError with formatted diagnostic output if validation fails.
    Returns (validated_schema, base_dir).
    """
    if not os.path.exists(config_file) and os.path.exists("inputs/config.json"):
        config_file = "inputs/config.json"

    if not os.path.exists(config_file):
        config_file = "config.json"

    config_abs_path = os.path.abspath(config_file)
    base_dir = os.path.dirname(config_abs_path)

    with open(config_file, 'r', encoding='utf-8') as f:
        raw_cfg = json.load(f)

    try:
        validated_cfg = SimulationConfigSchema.model_validate(raw_cfg)
    except ValidationError as err:
        err_msg = format_validation_error(err)
        print(f"\n{err_msg}\n")
        raise ValueError(err_msg) from err

    return validated_cfg, base_dir


def setup_simulation_parameters_3d(
    Vf: float,
    Vf_antenne: float = 0.0,
    config_file: str = "config.json",
    output_dir: Optional[str] = None
) -> Tuple[SimulationParams3D, SimulationToggles3D, PlottingConfig3D]:
    """
    Loads and validates configuration for 3D PIC simulation using declarative Pydantic schemas.
    Builds typed SimulationParams3D, SimulationToggles3D, and PlottingConfig3D.
    """
    cfg, base_dir = load_and_validate_config(config_file)

    # 1. Toggles
    toggles = SimulationToggles3D(**cfg.toggles.model_dump())

    # 2. Geometry and VTK configuration
    if cfg.geometry.source == "analytical":
        geom_cfg, vtk_cfg = _build_analytical_geometry(cfg.geometry.analytical)
    else:
        geom_cfg, vtk_cfg = _build_spis_geometry(cfg.geometry.spis, base_dir)

    # 3. Impact configuration
    imp_s = cfg.physics.impact
    impact_obj = ImpactConfig(
        location=list(imp_s.location),
        direction=list(imp_s.direction),
        time_delay_s=float(imp_s.time_delay_s),
        normal=list(imp_s.normal) if imp_s.normal is not None else None
    )

    # 4. Physics configuration
    phy_s = cfg.physics
    sc_volt = float(phy_s.spacecraft_voltage_V) if phy_s.spacecraft_voltage_V is not None else None
    if sc_volt is None:
        sc_volt = (
            geom_cfg.analytical.spacecraft.voltage_V
            if geom_cfg.source == "analytical"
            else geom_cfg.spis.spacecraft.voltage_V
        )

    physics_cfg = PhysicsConfig(
        ion_mass_amu=float(phy_s.ion_mass_amu),
        impact_cloud_temperature_eV=float(phy_s.impact_cloud_temperature_eV),
        solar_wind_electron_temp_eV=float(phy_s.solar_wind_electron_temp_eV),
        solar_wind_density_m3=float(phy_s.solar_wind_density_m3),
        total_impact_charge_C=float(phy_s.total_impact_charge_C),
        plasma_injection_mode=phy_s.plasma_injection_mode,
        impact=impact_obj,
        spacecraft_voltage_V=sc_volt,
        antenna_capacitance_F=[float(c) for c in phy_s.antenna_capacitance_F] if phy_s.antenna_capacitance_F else None,
        antenna_resistance_Ohm=[float(r) for r in phy_s.antenna_resistance_Ohm] if phy_s.antenna_resistance_Ohm else None,
        antenna_bias_voltage_V=[float(v) for v in phy_s.antenna_bias_voltage_V] if phy_s.antenna_bias_voltage_V else None,
    )

    # 5. Numeric configuration
    num_s = cfg.numeric
    numeric_cfg = NumericConfig(
        num_macroparticles=int(num_s.num_macroparticles),
        time_step_s=float(num_s.time_step_s),
        num_time_steps=int(num_s.num_time_steps),
        simulation_duration_s=float(num_s.simulation_duration_s) if num_s.simulation_duration_s is not None else None,
        domain_half_length_m=[float(d) for d in num_s.domain_half_length_m],
        grid_nodes=[int(n) for n in num_s.grid_nodes],
        field_precision=getattr(num_s, 'field_precision', 'float32'),
    )

    # 6. SimulationParams3D aggregation
    params = SimulationParams3D(
        Vf=Vf,
        Vf_antenne=Vf_antenne,
        geometry=geom_cfg,
        vtk_files=vtk_cfg,
        physics=physics_cfg,
        numeric=numeric_cfg,
        domain_half_length_m=list(numeric_cfg.domain_half_length_m),
        grid_nodes=list(numeric_cfg.grid_nodes),
        impact=impact_obj,
        weighting_threshold=geom_cfg.spis.weighting_threshold if geom_cfg.source == "spis" else 0.85
    )

    # 7. Plotting configuration and path resolution
    plot_config = PlottingConfig3D(**cfg.plotting.model_dump())
    _resolve_output_paths(plot_config, base_dir, output_dir)

    return params, toggles, plot_config
