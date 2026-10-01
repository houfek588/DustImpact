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
class SpisSpacecraftConfig:
    weighting_file: str = "inputs/spis_Vw_body.vtk"
    surface_mesh_file: str = ""
    voltage_V: Optional[float] = None
    potential_V: Optional[float] = None

    @property
    def effective_voltage(self) -> Optional[float]:
        if self.voltage_V is not None:
            return float(self.voltage_V)
        if self.potential_V is not None:
            return float(self.potential_V)
        return None


@dataclass
class SpisAntennaGeometry:
    weighting_file: str = ""
    voltage_V: Optional[float] = None
    potential_V: Optional[float] = None
    capacitance_F: Optional[float] = None
    resistance_Ohm: Optional[float] = None

    @property
    def effective_voltage(self) -> Optional[float]:
        if self.voltage_V is not None:
            return float(self.voltage_V)
        if self.potential_V is not None:
            return float(self.potential_V)
        return None


@dataclass
class SpisGeometryConfig:
    background_potential_file: str = "inputs/spis_V_bg.vtk"
    spacecraft: SpisSpacecraftConfig = field(default_factory=SpisSpacecraftConfig)
    antennas: List[SpisAntennaGeometry] = field(default_factory=lambda: [
        SpisAntennaGeometry(weighting_file="inputs/spis_Vw_ant1.vtk"),
        SpisAntennaGeometry(weighting_file="inputs/spis_Vw_ant2.vtk"),
        SpisAntennaGeometry(weighting_file="inputs/spis_Vw_ant3.vtk")
    ])
    weighting_threshold: float = 0.85
    spacecraft_weighting_file: Optional[str] = None
    spacecraft_surface_mesh_file: Optional[str] = None
    antenna_weighting_files: Optional[List[str]] = None

    def __post_init__(self):
        if self.spacecraft_weighting_file is not None:
            self.spacecraft.weighting_file = self.spacecraft_weighting_file
        else:
            self.spacecraft_weighting_file = self.spacecraft.weighting_file

        if self.spacecraft_surface_mesh_file is not None:
            self.spacecraft.surface_mesh_file = self.spacecraft_surface_mesh_file
        else:
            self.spacecraft_surface_mesh_file = self.spacecraft.surface_mesh_file

        if self.antenna_weighting_files is not None:
            self.antennas = [
                SpisAntennaGeometry(weighting_file=f) for f in self.antenna_weighting_files
            ]
        else:
            self.antenna_weighting_files = [a.weighting_file for a in self.antennas]

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
    voltage_V: Optional[float] = None
    potential_V: Optional[float] = None
    parts: List['AnalyticalSpacecraftPart'] = field(default_factory=list)

    @property
    def effective_voltage(self) -> Optional[float]:
        if self.voltage_V is not None:
            return float(self.voltage_V)
        if self.potential_V is not None:
            return float(self.potential_V)
        return None


@dataclass
class AnalyticalAntennaGeometry:
    p_start: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    p_end: List[float] = field(default_factory=lambda: [1.0, 0.0, 0.0])
    radius: float = 0.015
    voltage_V: Optional[float] = 0.0
    potential_V: Optional[float] = None
    bias_voltage_V: Optional[float] = None
    capacitance_F: Optional[float] = None
    resistance_Ohm: Optional[float] = None

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
class ImpactConfig:
    location: List[float] = field(default_factory=lambda: [-2.0, 2.0, 0.0])
    direction: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    time_delay_s: float = 1e-6
    normal: Optional[List[float]] = None

    @property
    def location_xyz_m(self) -> List[float]:
        return self.location

    @property
    def direction_vector(self) -> List[float]:
        return self.direction


@dataclass
class PhysicConfig:
    ion_mass_amu: float = 27.0
    impact_cloud_temperature_eV: float = 2.0
    solar_wind_electron_temp_eV: float = 15.0
    solar_wind_density_m3: float = 1e7
    total_impact_charge_C: float = 50e-12
    plasma_injection_mode: str = "point_cloud"
    impact: ImpactConfig = field(default_factory=ImpactConfig)
    spacecraft_voltage_V: Optional[float] = None
    antenna_capacitance_F: Optional[List[float]] = None
    antenna_resistance_Ohm: Optional[List[float]] = None
    antenna_bias_voltage_V: Optional[List[float]] = None

    # Backward compatibility flat fields
    impact_location_xyz_m: Optional[List[float]] = None
    impact_direction_vector: Optional[List[float]] = None
    impact_normal: Optional[List[float]] = None
    impact_time_delay_s: Optional[float] = None

    def __post_init__(self):
        if isinstance(self.impact, dict):
            loc = self.impact.get('location', self.impact.get('location_xyz_m', self.impact.get('pos', [-2.0, 2.0, 0.0])))
            direc = self.impact.get('direction', self.impact.get('direction_vector', [0.0, 0.0, 0.0]))
            t_del = float(self.impact.get('time_delay_s', self.impact.get('time_delay', 1e-6)))
            norm = self.impact.get('normal', self.impact.get('impact_normal', None))
            self.impact = ImpactConfig(location=list(loc), direction=list(direc), time_delay_s=t_del, normal=list(norm) if norm is not None else None)

        if self.impact_location_xyz_m is not None:
            self.impact.location = list(self.impact_location_xyz_m)
        else:
            self.impact_location_xyz_m = list(self.impact.location)

        if self.impact_direction_vector is not None:
            self.impact.direction = list(self.impact_direction_vector)
        else:
            self.impact_direction_vector = list(self.impact.direction)

        if self.impact_time_delay_s is not None:
            self.impact.time_delay_s = float(self.impact_time_delay_s)
        else:
            self.impact_time_delay_s = float(self.impact.time_delay_s)

        if self.impact_normal is not None:
            self.impact.normal = list(self.impact_normal)
        elif self.impact.normal is not None:
            self.impact_normal = list(self.impact.normal)


@dataclass
class NumericConfig:
    num_macroparticles: int = 20000
    time_step_s: float = 2e-9
    num_time_steps: int = 10000
    simulation_duration_s: Optional[float] = None
    domain_half_length_m: List[float] = field(default_factory=lambda: [5.0, 5.0, 5.0])
    grid_nodes: List[int] = field(default_factory=lambda: [35, 35, 35])
    antenna_weighting_threshold: Optional[float] = None  # Deprecated: use geometry.spis.weighting_threshold

    # Backward compatibility flat fields
    domain_half_length_x_m: Optional[float] = None
    domain_half_length_y_m: Optional[float] = None
    domain_half_length_z_m: Optional[float] = None
    grid_nodes_x: Optional[int] = None
    grid_nodes_y: Optional[int] = None
    grid_nodes_z: Optional[int] = None

    def __post_init__(self):
        if self.grid_nodes_x is not None or self.grid_nodes_y is not None or self.grid_nodes_z is not None:
            gx = self.grid_nodes_x if self.grid_nodes_x is not None else self.grid_nodes[0]
            gy = self.grid_nodes_y if self.grid_nodes_y is not None else self.grid_nodes[1]
            gz = self.grid_nodes_z if self.grid_nodes_z is not None else self.grid_nodes[2]
            self.grid_nodes = [int(gx), int(gy), int(gz)]
        elif isinstance(self.grid_nodes, (list, tuple)):
            self.grid_nodes = [int(v) for v in self.grid_nodes]
        elif isinstance(self.grid_nodes, (int, float)):
            v = int(self.grid_nodes)
            self.grid_nodes = [v, v, v]

        if self.domain_half_length_x_m is not None or self.domain_half_length_y_m is not None or self.domain_half_length_z_m is not None:
            lx = self.domain_half_length_x_m if self.domain_half_length_x_m is not None else self.domain_half_length_m[0]
            ly = self.domain_half_length_y_m if self.domain_half_length_y_m is not None else self.domain_half_length_m[1]
            lz = self.domain_half_length_z_m if self.domain_half_length_z_m is not None else self.domain_half_length_m[2]
            self.domain_half_length_m = [float(lx), float(ly), float(lz)]
        elif isinstance(self.domain_half_length_m, (list, tuple)):
            self.domain_half_length_m = [float(v) for v in self.domain_half_length_m]
        elif isinstance(self.domain_half_length_m, (int, float)):
            v = float(self.domain_half_length_m)
            self.domain_half_length_m = [v, v, v]

        self.grid_nodes_x = self.grid_nodes[0]
        self.grid_nodes_y = self.grid_nodes[1]
        self.grid_nodes_z = self.grid_nodes[2]
        self.domain_half_length_x_m = self.domain_half_length_m[0]
        self.domain_half_length_y_m = self.domain_half_length_m[1]
        self.domain_half_length_z_m = self.domain_half_length_m[2]


@dataclass
class SimulationParams3D(BaseSimulationParams):
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    vtk_files: VTKFilesConfig = field(default_factory=VTKFilesConfig)
    physic: PhysicConfig = field(default_factory=PhysicConfig)
    numeric: NumericConfig = field(default_factory=NumericConfig)

    domain_half_length_m: List[float] = field(default_factory=lambda: [5.0, 5.0, 5.0])
    grid_nodes: List[int] = field(default_factory=lambda: [35, 35, 35])

    domain_half_length_x_m: Optional[float] = None
    domain_half_length_y_m: Optional[float] = None
    domain_half_length_z_m: Optional[float] = None

    impact: ImpactConfig = field(default_factory=ImpactConfig)
    impact_location_xyz_m: Optional[List[float]] = None
    impact_direction_vector: Optional[List[float]] = None
    impact_normal: Optional[List[float]] = None
    impact_time_delay_s: Optional[float] = None

    grid_nodes_x: Optional[int] = None
    grid_nodes_y: Optional[int] = None
    grid_nodes_z: Optional[int] = None

    antenna_capacitance_F: List[float] = field(default_factory=lambda: [2e-12, 2e-12, 2e-12])
    antenna_resistance_Ohm: List[float] = field(default_factory=lambda: [100e3, 100e3, 100e3])
    antenna_bias_voltage_V: List[float] = field(default_factory=list)
    antenna_collection_efficiency: Optional[List[float]] = None
    weighting_threshold: float = 0.85
    antenna_weighting_threshold: Optional[float] = None

    @property
    def physics(self) -> PhysicConfig:
        return self.physic

    @property
    def numerics(self) -> NumericConfig:
        return self.numeric

    @property
    def Nx(self) -> int:
        return int(self.grid_nodes[0])

    @Nx.setter
    def Nx(self, val: int) -> None:
        self.grid_nodes[0] = int(val)
        self.grid_nodes_x = self.grid_nodes[0]

    @property
    def Ny(self) -> int:
        return int(self.grid_nodes[1])

    @Ny.setter
    def Ny(self, val: int) -> None:
        self.grid_nodes[1] = int(val)
        self.grid_nodes_y = self.grid_nodes[1]

    @property
    def Nz(self) -> int:
        return int(self.grid_nodes[2])

    @Nz.setter
    def Nz(self, val: int) -> None:
        self.grid_nodes[2] = int(val)
        self.grid_nodes_z = self.grid_nodes[2]

    @property
    def L_x(self) -> float:
        return float(self.domain_half_length_m[0])

    @L_x.setter
    def L_x(self, val: float) -> None:
        self.domain_half_length_m[0] = float(val)
        self.domain_half_length_x_m = self.domain_half_length_m[0]

    @property
    def L_y(self) -> float:
        return float(self.domain_half_length_m[1])

    @L_y.setter
    def L_y(self, val: float) -> None:
        self.domain_half_length_m[1] = float(val)
        self.domain_half_length_y_m = self.domain_half_length_m[1]

    @property
    def L_z(self) -> float:
        return float(self.domain_half_length_m[2])

    @L_z.setter
    def L_z(self, val: float) -> None:
        self.domain_half_length_m[2] = float(val)
        self.domain_half_length_z_m = self.domain_half_length_m[2]

    @property
    def x_grid(self) -> np.ndarray:
        return np.linspace(-self.L_x, self.L_x, self.Nx)

    @property
    def y_grid(self) -> np.ndarray:
        return np.linspace(-self.L_y, self.L_y, self.Ny)

    @property
    def z_grid(self) -> np.ndarray:
        return np.linspace(-self.L_z, self.L_z, self.Nz)

    @property
    def dx(self) -> float:
        return float(2.0 * self.L_x / (self.Nx - 1)) if self.Nx > 1 else 1.0

    @property
    def dy(self) -> float:
        return float(2.0 * self.L_y / (self.Ny - 1)) if self.Ny > 1 else 1.0

    @property
    def dz(self) -> float:
        return float(2.0 * self.L_z / (self.Nz - 1)) if self.Nz > 1 else 1.0

    @property
    def impact_pos(self) -> List[float]:
        return self.impact.location

    @impact_pos.setter
    def impact_pos(self, val: List[float]) -> None:
        self.impact.location = list(val)
        self.impact_location_xyz_m = list(val)

    @property
    def t_delay(self) -> float:
        return float(self.impact.time_delay_s)

    @t_delay.setter
    def t_delay(self, val: float) -> None:
        self.impact.time_delay_s = float(val)
        self.impact_time_delay_s = float(val)

    @property
    def C_ant(self) -> List[float]:
        return self.antenna_capacitance_F

    @C_ant.setter
    def C_ant(self, val: List[float]) -> None:
        self.antenna_capacitance_F = list(val)

    @property
    def R_ant(self) -> List[float]:
        return self.antenna_resistance_Ohm

    @R_ant.setter
    def R_ant(self, val: List[float]) -> None:
        self.antenna_resistance_Ohm = list(val)

    @property
    def V_bias(self) -> List[float]:
        return self.antenna_bias_voltage_V

    @V_bias.setter
    def V_bias(self, val: List[float]) -> None:
        self.antenna_bias_voltage_V = list(val)

    @property
    def collection_eff(self) -> List[float]:
        return self.antenna_collection_efficiency or [1.0] * len(self.antenna_capacitance_F)

    @collection_eff.setter
    def collection_eff(self, val: List[float]) -> None:
        self.antenna_collection_efficiency = list(val)

    def __post_init__(self):
        # 1. Synchronize sub-configs (physic, numeric) with self
        skip_physic = {'impact_location_xyz_m', 'impact_direction_vector', 'impact_normal', 'impact_time_delay_s', 'impact'}
        for f in fields(PhysicConfig):
            if f.name in skip_physic:
                continue
            sub_val = getattr(self.physic, f.name)
            if hasattr(self, f.name):
                self_val = getattr(self, f.name)
                d_val = f.default if f.default is not field else None
                if f.default_factory is not None and d_val is None:
                    try:
                        d_val = f.default_factory()
                    except Exception:
                        pass
                if sub_val is not None and sub_val != d_val and (self_val == d_val or self_val is None):
                    setattr(self, f.name, sub_val)

        skip_numeric = {'grid_nodes_x', 'grid_nodes_y', 'grid_nodes_z', 'domain_half_length_x_m', 'domain_half_length_y_m', 'domain_half_length_z_m', 'grid_nodes', 'domain_half_length_m', 'antenna_weighting_threshold'}
        for f in fields(NumericConfig):
            if f.name in skip_numeric:
                continue
            sub_val = getattr(self.numeric, f.name)
            if hasattr(self, f.name):
                self_val = getattr(self, f.name)
                d_val = f.default if f.default is not field else None
                if f.default_factory is not None and d_val is None:
                    try:
                        d_val = f.default_factory()
                    except Exception:
                        pass
                if sub_val is not None and sub_val != d_val and (self_val == d_val or self_val is None):
                    setattr(self, f.name, sub_val)

        # 2. Resolve vector vs scalar/flat grid_nodes and domain_half_length_m
        if self.grid_nodes_x is not None or self.grid_nodes_y is not None or self.grid_nodes_z is not None:
            gx = self.grid_nodes_x if self.grid_nodes_x is not None else self.grid_nodes[0]
            gy = self.grid_nodes_y if self.grid_nodes_y is not None else self.grid_nodes[1]
            gz = self.grid_nodes_z if self.grid_nodes_z is not None else self.grid_nodes[2]
            self.grid_nodes = [int(gx), int(gy), int(gz)]
        elif self.numeric.grid_nodes != [35, 35, 35] and self.grid_nodes == [35, 35, 35]:
            self.grid_nodes = [int(v) for v in self.numeric.grid_nodes]
        elif isinstance(self.grid_nodes, (list, tuple)):
            self.grid_nodes = [int(v) for v in self.grid_nodes]
        elif isinstance(self.grid_nodes, (int, float)):
            v = int(self.grid_nodes)
            self.grid_nodes = [v, v, v]

        if self.domain_half_length_x_m is not None or self.domain_half_length_y_m is not None or self.domain_half_length_z_m is not None:
            lx = self.domain_half_length_x_m if self.domain_half_length_x_m is not None else self.domain_half_length_m[0]
            ly = self.domain_half_length_y_m if self.domain_half_length_y_m is not None else self.domain_half_length_m[1]
            lz = self.domain_half_length_z_m if self.domain_half_length_z_m is not None else self.domain_half_length_m[2]
            self.domain_half_length_m = [float(lx), float(ly), float(lz)]
        elif self.numeric.domain_half_length_m != [5.0, 5.0, 5.0] and self.domain_half_length_m == [5.0, 5.0, 5.0]:
            self.domain_half_length_m = [float(v) for v in self.numeric.domain_half_length_m]
        elif isinstance(self.domain_half_length_m, (list, tuple)):
            self.domain_half_length_m = [float(v) for v in self.domain_half_length_m]
        elif isinstance(self.domain_half_length_m, (int, float)):
            v = float(self.domain_half_length_m)
            self.domain_half_length_m = [v, v, v]

        self.grid_nodes_x = self.grid_nodes[0]
        self.grid_nodes_y = self.grid_nodes[1]
        self.grid_nodes_z = self.grid_nodes[2]
        self.domain_half_length_x_m = self.domain_half_length_m[0]
        self.domain_half_length_y_m = self.domain_half_length_m[1]
        self.domain_half_length_z_m = self.domain_half_length_m[2]

        # 3. Resolve impact sub-config / flat fields
        if isinstance(self.impact, dict):
            loc = self.impact.get('location', self.impact.get('location_xyz_m', self.impact.get('pos', [-2.0, 2.0, 0.0])))
            direc = self.impact.get('direction', self.impact.get('direction_vector', [0.0, 0.0, 0.0]))
            t_del = float(self.impact.get('time_delay_s', self.impact.get('time_delay', 1e-6)))
            norm = self.impact.get('normal', self.impact.get('impact_normal', None))
            self.impact = ImpactConfig(location=list(loc), direction=list(direc), time_delay_s=t_del, normal=list(norm) if norm is not None else None)
        elif self.impact == ImpactConfig() and self.physic.impact != ImpactConfig():
            self.impact = self.physic.impact

        if self.impact_location_xyz_m is not None:
            self.impact.location = list(self.impact_location_xyz_m)
        else:
            self.impact_location_xyz_m = list(self.impact.location)

        if self.impact_direction_vector is not None:
            self.impact.direction = list(self.impact_direction_vector)
        else:
            self.impact_direction_vector = list(self.impact.direction)

        if self.impact_time_delay_s is not None:
            self.impact.time_delay_s = float(self.impact_time_delay_s)
        else:
            self.impact_time_delay_s = float(self.impact.time_delay_s)

        if self.impact_normal is not None:
            self.impact.normal = list(self.impact_normal)
        elif self.impact.normal is not None:
            self.impact_normal = list(self.impact.normal)
        else:
            self.impact_normal = [-0.7071, 0.7071, 0.0]
            self.impact.normal = list(self.impact_normal)

        # 4. Base derived params
        self._init_base_derived_params()

        # 5. Synchronize geometry and vtk_files for bidirectional compatibility
        source = getattr(self.geometry, 'source', 'spis')
        if source == 'spis':
            default_sc = "inputs/spis_Vw_body.vtk"
            default_bg = "inputs/spis_V_bg.vtk"
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
        elif source == 'analytical':
            ant_geoms = getattr(getattr(self.geometry, 'analytical', None), 'antennas', [])
            if ant_geoms:
                self.antenna_bias_voltage_V = [
                    float(getattr(a, 'effective_voltage', 0.0))
                    for a in ant_geoms
                ]
            elif not self.antenna_bias_voltage_V:
                self.antenna_bias_voltage_V = [0.0] * len(self.antenna_capacitance_F)

        # Spacecraft voltage sync
        geom_obj = getattr(self.geometry, source, None)
        sc = getattr(geom_obj, 'spacecraft', None)
        if sc and getattr(sc, 'effective_voltage', None) is not None:
            self.spacecraft_voltage_V = sc.effective_voltage

        # Antenna circuit params sync
        ant_geoms_list = getattr(geom_obj, 'antennas', [])
        if ant_geoms_list:
            if any(getattr(a, 'capacitance_F', None) is not None for a in ant_geoms_list):
                self.antenna_capacitance_F = [
                    float(a.capacitance_F) if getattr(a, 'capacitance_F', None) is not None else 2e-12
                    for a in ant_geoms_list
                ]
            if any(getattr(a, 'resistance_Ohm', None) is not None for a in ant_geoms_list):
                self.antenna_resistance_Ohm = [
                    float(a.resistance_Ohm) if getattr(a, 'resistance_Ohm', None) is not None else 100e3
                    for a in ant_geoms_list
                ]

        # Weighting threshold sync
        if source == 'spis':
            spis_obj = getattr(self.geometry, 'spis', None)
            if spis_obj and hasattr(spis_obj, 'weighting_threshold'):
                self.weighting_threshold = float(spis_obj.weighting_threshold)
        if self.antenna_weighting_threshold is None:
            self.antenna_weighting_threshold = self.weighting_threshold
        else:
            self.weighting_threshold = float(self.antenna_weighting_threshold)
            if source == 'spis' and hasattr(self.geometry.spis, 'weighting_threshold'):
                self.geometry.spis.weighting_threshold = self.weighting_threshold

        # Homogeneous macroparticle charge
        if getattr(self, 'plasma_injection_mode', 'point_cloud') == 'homogeneous':
            domain_vol = (2.0 * self.L_x) * (2.0 * self.L_y) * (2.0 * self.L_z)
            self.q_macro = (self.solar_wind_density_m3 * domain_vol * e) / self.num_macroparticles

        # Keep PhysicConfig and NumericConfig synchronized with self
        for f in fields(PhysicConfig):
            if hasattr(self, f.name):
                setattr(self.physic, f.name, getattr(self, f.name))
        for f in fields(NumericConfig):
            if hasattr(self, f.name):
                setattr(self.numeric, f.name, getattr(self, f.name))
        self.physic.impact = self.impact
        self.physic.impact_location_xyz_m = list(self.impact_location_xyz_m)
        self.physic.impact_direction_vector = list(self.impact_direction_vector)
        self.physic.impact_time_delay_s = float(self.impact_time_delay_s)
        if self.impact_normal is not None:
            self.physic.impact_normal = list(self.impact_normal)
        self.numeric.grid_nodes = list(self.grid_nodes)
        self.numeric.domain_half_length_m = list(self.domain_half_length_m)


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

    p_kwargs = {}
    if 'params' in cfg and isinstance(cfg['params'], dict):
        p_kwargs.update(cfg['params'])
    for k in ('physic', 'physics'):
        if k in cfg and isinstance(cfg[k], dict):
            p_kwargs.update(cfg[k])
    for k in ('numeric', 'numerics'):
        if k in cfg and isinstance(cfg[k], dict):
            p_kwargs.update(cfg[k])

    # Convert simulation_duration_s to num_time_steps if num_time_steps not explicitly set
    if 'num_time_steps' not in p_kwargs:
        for alt_key in ('n_steps', 'steps', 'num_steps', 'time_steps'):
            if alt_key in p_kwargs:
                p_kwargs['num_time_steps'] = int(p_kwargs[alt_key])
                break
    if 'simulation_duration_s' in p_kwargs and 'num_time_steps' not in p_kwargs:
        dt = float(p_kwargs.get('time_step_s', 2e-9))
        p_kwargs['num_time_steps'] = int(round(float(p_kwargs['simulation_duration_s']) / dt))

    # Normalize grid_nodes
    if 'grid_nodes' in p_kwargs:
        gn = p_kwargs['grid_nodes']
        if isinstance(gn, (list, tuple)) and len(gn) == 3:
            p_kwargs['grid_nodes'] = [int(v) for v in gn]
        elif isinstance(gn, (int, float)):
            p_kwargs['grid_nodes'] = [int(gn), int(gn), int(gn)]
    elif all(k in p_kwargs for k in ('grid_nodes_x', 'grid_nodes_y', 'grid_nodes_z')):
        p_kwargs['grid_nodes'] = [int(p_kwargs['grid_nodes_x']), int(p_kwargs['grid_nodes_y']), int(p_kwargs['grid_nodes_z'])]

    # Normalize domain_half_length_m
    domain_keys = ('domain_half_length_m', 'domain_half_lengths_m', 'domain_half_length_xyz_m', 'domain_half_length')
    for dk in domain_keys:
        if dk in p_kwargs:
            dhl = p_kwargs[dk]
            if isinstance(dhl, (list, tuple)) and len(dhl) == 3:
                p_kwargs['domain_half_length_m'] = [float(v) for v in dhl]
            elif isinstance(dhl, (int, float)):
                p_kwargs['domain_half_length_m'] = [float(dhl), float(dhl), float(dhl)]
            break
    if 'domain_half_length_m' not in p_kwargs and all(k in p_kwargs for k in ('domain_half_length_x_m', 'domain_half_length_y_m', 'domain_half_length_z_m')):
        p_kwargs['domain_half_length_m'] = [float(p_kwargs['domain_half_length_x_m']), float(p_kwargs['domain_half_length_y_m']), float(p_kwargs['domain_half_length_z_m'])]

    if 'grid_nodes' in p_kwargs:
        p_kwargs['grid_nodes_x'] = p_kwargs['grid_nodes'][0]
        p_kwargs['grid_nodes_y'] = p_kwargs['grid_nodes'][1]
        p_kwargs['grid_nodes_z'] = p_kwargs['grid_nodes'][2]
    if 'domain_half_length_m' in p_kwargs:
        p_kwargs['domain_half_length_x_m'] = p_kwargs['domain_half_length_m'][0]
        p_kwargs['domain_half_length_y_m'] = p_kwargs['domain_half_length_m'][1]
        p_kwargs['domain_half_length_z_m'] = p_kwargs['domain_half_length_m'][2]

    # Normalize impact
    impact_data = None
    if 'impact' in cfg and isinstance(cfg['impact'], dict):
        impact_data = cfg['impact']
    elif 'impact' in p_kwargs and isinstance(p_kwargs['impact'], dict):
        impact_data = p_kwargs['impact']
    elif 'physic' in cfg and isinstance(cfg['physic'], dict) and 'impact' in cfg['physic']:
        impact_data = cfg['physic']['impact']
    elif 'physics' in cfg and isinstance(cfg['physics'], dict) and 'impact' in cfg['physics']:
        impact_data = cfg['physics']['impact']
    elif 'params' in cfg and isinstance(cfg['params'], dict) and 'impact' in cfg['params']:
        impact_data = cfg['params']['impact']

    if impact_data:
        loc = impact_data.get('location', impact_data.get('location_xyz_m', impact_data.get('pos', p_kwargs.get('impact_location_xyz_m', [-2.0, 2.0, 0.0]))))
        direc = impact_data.get('direction', impact_data.get('direction_vector', p_kwargs.get('impact_direction_vector', [0.0, 0.0, 0.0])))
        t_del = impact_data.get('time_delay_s', impact_data.get('time_delay', p_kwargs.get('impact_time_delay_s', 1e-6)))
        norm = impact_data.get('normal', impact_data.get('impact_normal', p_kwargs.get('impact_normal', None)))

        impact_obj = ImpactConfig(location=list(loc), direction=list(direc), time_delay_s=float(t_del), normal=list(norm) if norm is not None else None)
        p_kwargs['impact'] = impact_obj
        p_kwargs['impact_location_xyz_m'] = list(loc)
        p_kwargs['impact_direction_vector'] = list(direc)
        p_kwargs['impact_time_delay_s'] = float(t_del)
        if norm is not None:
            p_kwargs['impact_normal'] = list(norm)
    elif any(k in p_kwargs for k in ('impact_location_xyz_m', 'impact_direction_vector', 'impact_time_delay_s', 'impact_normal')):
        loc = p_kwargs.get('impact_location_xyz_m', [-2.0, 2.0, 0.0])
        direc = p_kwargs.get('impact_direction_vector', [0.0, 0.0, 0.0])
        t_del = p_kwargs.get('impact_time_delay_s', 1e-6)
        norm = p_kwargs.get('impact_normal', None)
        impact_obj = ImpactConfig(location=list(loc), direction=list(direc), time_delay_s=float(t_del), normal=list(norm) if norm is not None else None)
        p_kwargs['impact'] = impact_obj
        p_kwargs['impact_location_xyz_m'] = list(loc)
        p_kwargs['impact_direction_vector'] = list(direc)
        p_kwargs['impact_time_delay_s'] = float(t_del)
        if norm is not None:
            p_kwargs['impact_normal'] = list(norm)

    # Parse geometry configuration (top-level "geometry" or inside "params.geometry" or "physic.geometry")
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
            if sc_part.effective_voltage is not None:
                p_kwargs['spacecraft_voltage_V'] = sc_part.effective_voltage

            ant_list = []
            extracted_voltages = []
            extracted_caps = []
            extracted_res = []
            has_caps = False
            has_res = False
            for a in ana_data.get('antennas', []):
                v_val = a.get('voltage_V', a.get('potential_V', a.get('bias_voltage_V', a.get('voltage', a.get('potential', None)))))
                v_float = float(v_val) if v_val is not None else 0.0
                c_val = a.get('capacitance_F', a.get('antenna_capacitance_F', a.get('C_ant', None)))
                r_val = a.get('resistance_Ohm', a.get('antenna_resistance_Ohm', a.get('R_ant', None)))

                ant_obj = AnalyticalAntennaGeometry(
                    p_start=a.get('p_start', [0.0, 0.0, 0.0]),
                    p_end=a.get('p_end', [1.0, 0.0, 0.0]),
                    radius=float(a.get('radius', 0.015)),
                    voltage_V=v_float,
                    potential_V=v_float,
                    bias_voltage_V=v_float,
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
            p_kwargs['geometry'] = geom_config
            p_kwargs['vtk_files'] = VTKFilesConfig(
                spis_background_potential_file="",
                spacecraft_weighting_file="",
                antenna_weighting_files=[]
            )
        else:
            spis_data = geom_data.get('spis', {})
            bg_f = _resolve_path(base_dir, spis_data.get('background_potential_file', spis_data.get('spis_background_potential_file', 'inputs/spis_V_bg.vtk')))
            thresh = float(spis_data.get('weighting_threshold', p_kwargs.get('weighting_threshold', p_kwargs.get('antenna_weighting_threshold', 0.85))))
            p_kwargs['weighting_threshold'] = thresh
            p_kwargs['antenna_weighting_threshold'] = thresh

            if 'spacecraft' in spis_data and isinstance(spis_data['spacecraft'], dict):
                sc_data = spis_data['spacecraft']
                sc_f = _resolve_path(base_dir, sc_data.get('weighting_file', sc_data.get('spacecraft_weighting_file', 'inputs/spis_Vw_body.vtk')))
                mesh_f = _resolve_path(base_dir, sc_data.get('surface_mesh_file', sc_data.get('spacecraft_surface_mesh_file', '')))
                v_sc = sc_data.get('voltage_V', sc_data.get('potential_V', sc_data.get('spacecraft_voltage_V', None)))
                if v_sc is not None:
                    p_kwargs['spacecraft_voltage_V'] = float(v_sc)
                sc_obj = SpisSpacecraftConfig(weighting_file=sc_f, surface_mesh_file=mesh_f, voltage_V=float(v_sc) if v_sc is not None else None)
            else:
                sc_f = _resolve_path(base_dir, spis_data.get('spacecraft_weighting_file', 'inputs/spis_Vw_body.vtk'))
                mesh_f = _resolve_path(base_dir, spis_data.get('spacecraft_surface_mesh_file', ''))
                sc_obj = SpisSpacecraftConfig(weighting_file=sc_f, surface_mesh_file=mesh_f)

            if 'antennas' in spis_data and isinstance(spis_data['antennas'], list):
                ant_list = []
                ant_fs = []
                extracted_caps = []
                extracted_res = []
                extracted_effs = []
                extracted_volts = []
                has_caps = False
                has_res = False
                has_effs = False
                has_volts = False
                for a in spis_data['antennas']:
                    if isinstance(a, str):
                        f_path = _resolve_path(base_dir, a)
                        ant_list.append(SpisAntennaGeometry(weighting_file=f_path))
                        ant_fs.append(f_path)
                    elif isinstance(a, dict):
                        f_path = _resolve_path(base_dir, a.get('weighting_file', a.get('file', a.get('antenna_weighting_file', ''))))
                        c_val = a.get('capacitance_F', a.get('antenna_capacitance_F', a.get('C_ant', None)))
                        r_val = a.get('resistance_Ohm', a.get('antenna_resistance_Ohm', a.get('R_ant', None)))
                        v_val = a.get('voltage_V', a.get('potential_V', None))

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
            p_kwargs['geometry'] = GeometryConfig(source='spis', spis=spis_cfg)
            p_kwargs['vtk_files'] = VTKFilesConfig(
                spis_background_potential_file=bg_f,
                spacecraft_weighting_file=sc_f,
                antenna_weighting_files=ant_fs
            )
    elif ('vtk_files' in cfg and isinstance(cfg['vtk_files'], dict)) or ('vtk_files' in p_kwargs and isinstance(p_kwargs['vtk_files'], dict)):
        vtk_data = cfg.get('vtk_files', p_kwargs.get('vtk_files', {}))
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

    physic_kwargs = {k: v for k, v in p_kwargs.items() if k in {f.name for f in fields(PhysicConfig)}}
    numeric_kwargs = {k: v for k, v in p_kwargs.items() if k in {f.name for f in fields(NumericConfig)}}
    filtered_params['physic'] = PhysicConfig(**physic_kwargs)
    filtered_params['numeric'] = NumericConfig(**numeric_kwargs)

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

