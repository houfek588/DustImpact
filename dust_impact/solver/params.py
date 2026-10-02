# -*- coding: utf-8 -*-
"""
Simulation parameters and model dataclasses for 3D PIC simulation.
Defines typed configuration structures for physical constants, grid dimensions,
antenna probe circuits, impact parameters, and runtime toggles.
"""

from dataclasses import dataclass, field, fields
from typing import List, Optional
import numpy as np

from dust_impact.physics.constants import e
from dust_impact.common.config import BaseSimulationParams, BaseSimulationToggles
from dust_impact.geometry.config import GeometryConfig, VTKFilesConfig


@dataclass
class SimulationToggles3D(BaseSimulationToggles):
    """Runtime toggles for selectively enabling physical simulation mechanisms in 3D."""
    pass


@dataclass
class PlottingConfig3D:
    """Controls visualization, plotting, file paths, and periodic checkpointing."""
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
    """Detailed specifications for dust impact location, velocity vector, and timing."""
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
class PhysicsConfig:
    """Physics parameters of the impact plasma cloud and ambient solar wind."""
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
            loc = self.impact.get('location', [-2.0, 2.0, 0.0])
            direc = self.impact.get('direction', [0.0, 0.0, 0.0])
            t_del = float(self.impact.get('time_delay_s', 1e-6))
            norm = self.impact.get('normal', None)
            self.impact = ImpactConfig(
                location=list(loc), direction=list(direc), time_delay_s=t_del,
                normal=list(norm) if norm is not None else None
            )

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


# Backward-compatibility alias
PhysicConfig = PhysicsConfig


@dataclass
class NumericConfig:
    """Numerical discretization settings: grid nodes, domain bounds, time steps, macroparticles."""
    num_macroparticles: int = 20000
    time_step_s: float = 2e-9
    num_time_steps: int = 10000
    simulation_duration_s: Optional[float] = None
    domain_half_length_m: List[float] = field(default_factory=lambda: [5.0, 5.0, 5.0])
    grid_nodes: List[int] = field(default_factory=lambda: [35, 35, 35])
    field_precision: str = "float32"

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

        if (self.domain_half_length_x_m is not None or
            self.domain_half_length_y_m is not None or
            self.domain_half_length_z_m is not None):
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
    """
    Main 3D PIC simulation parameters.
    Aggregates geometry, physical constants, numerical domain, and probe circuit configs.
    Derives Cartesian grid linspaces, differential steps, and convenience aliases dynamically via @property.
    """
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    vtk_files: VTKFilesConfig = field(default_factory=VTKFilesConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
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

    @property
    def field_precision(self) -> str:
        return getattr(self.numeric, 'field_precision', 'float32')

    @field_precision.setter
    def field_precision(self, val: str) -> None:
        if hasattr(self, 'numeric') and self.numeric is not None:
            self.numeric.field_precision = str(val)

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
        # 1. Synchronize sub-configs (physics, numeric) with self
        skip_physics = {'impact_location_xyz_m', 'impact_direction_vector', 'impact_normal', 'impact_time_delay_s', 'impact'}
        for f in fields(PhysicsConfig):
            if f.name in skip_physics:
                continue
            sub_val = getattr(self.physics, f.name)
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

        skip_numeric = {'grid_nodes_x', 'grid_nodes_y', 'grid_nodes_z', 'domain_half_length_x_m', 'domain_half_length_y_m', 'domain_half_length_z_m', 'grid_nodes', 'domain_half_length_m'}
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

        if (self.domain_half_length_x_m is not None or
            self.domain_half_length_y_m is not None or
            self.domain_half_length_z_m is not None):
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
            loc = self.impact.get('location', [-2.0, 2.0, 0.0])
            direc = self.impact.get('direction', [0.0, 0.0, 0.0])
            t_del = float(self.impact.get('time_delay_s', 1e-6))
            norm = self.impact.get('normal', None)
            self.impact = ImpactConfig(
                location=list(loc), direction=list(direc), time_delay_s=t_del,
                normal=list(norm) if norm is not None else None
            )
        elif self.impact == ImpactConfig() and self.physics.impact != ImpactConfig():
            self.impact = self.physics.impact

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

        # Homogeneous macroparticle charge
        if getattr(self, 'plasma_injection_mode', 'point_cloud') == 'homogeneous':
            domain_vol = (2.0 * self.L_x) * (2.0 * self.L_y) * (2.0 * self.L_z)
            self.q_macro = (self.solar_wind_density_m3 * domain_vol * e) / self.num_macroparticles

        # Keep PhysicsConfig and NumericConfig synchronized with self
        for f in fields(PhysicsConfig):
            if hasattr(self, f.name):
                setattr(self.physics, f.name, getattr(self, f.name))
        for f in fields(NumericConfig):
            if hasattr(self, f.name):
                setattr(self.numeric, f.name, getattr(self, f.name))
        self.physics.impact = self.impact
        self.physics.impact_location_xyz_m = list(self.impact_location_xyz_m)
        self.physics.impact_direction_vector = list(self.impact_direction_vector)
        self.physics.impact_time_delay_s = float(self.impact_time_delay_s)
        if self.impact_normal is not None:
            self.physics.impact_normal = list(self.impact_normal)
        self.numeric.grid_nodes = list(self.grid_nodes)
        self.numeric.domain_half_length_m = list(self.domain_half_length_m)
