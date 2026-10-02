# -*- coding: utf-8 -*-
"""
Declarative validation schemas for DustImpact 3D PIC simulation configuration.
Uses Pydantic v2 for strong typing, range verification, strict field validation,
and automated JSON Schema generation.
"""

from typing import List, Optional, Literal, Union, Any, Annotated
from pydantic import (
    BaseModel,
    Field,
    ConfigDict,
    field_validator,
    model_validator,
    ValidationError
)

__all__ = [
    "StrictBaseModel",
    "SpisSpacecraftSchema",
    "SpisAntennaSchema",
    "SpisGeometrySchema",
    "AnalyticalSpacecraftPartSchema",
    "AnalyticalAntennaSchema",
    "AnalyticalGeometrySchema",
    "GeometrySchema",
    "VTKFilesSchema",
    "TogglesSchema",
    "ImpactSchema",
    "PhysicsSchema",
    "NumericSchema",
    "PlottingSchema",
    "SimulationConfigSchema",
    "format_validation_error",
]


class StrictBaseModel(BaseModel):
    """Base model that forbids unknown fields to prevent typos."""
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_default=True,
        validate_assignment=True
    )


class SpisSpacecraftSchema(StrictBaseModel):
    weighting_file: str = Field(
        "inputs/spis_Vw_body.vtk",
        description="Path to spacecraft Ramo-Shockley weighting potential VTK file"
    )
    surface_mesh_file: str = Field(
        "",
        description="Optional path to explicit surface mesh file (Gmsh, STL, VTK)"
    )
    voltage_V: Optional[float] = Field(
        None,
        description="Equilibrium floating potential of spacecraft body [V]"
    )


class SpisAntennaSchema(StrictBaseModel):
    weighting_file: str = Field(
        ...,
        description="Path to antenna weighting potential VTK file"
    )
    voltage_V: Optional[float] = Field(
        None,
        description="Antenna bias potential [V]"
    )
    capacitance_F: Optional[Annotated[float, Field(gt=0.0)]] = Field(
        None,
        description="Antenna input capacitance [F]"
    )
    resistance_Ohm: Optional[Annotated[float, Field(gt=0.0)]] = Field(
        None,
        description="Antenna input resistance [Ohm]"
    )


class SpisGeometrySchema(StrictBaseModel):
    background_potential_file: str = Field(
        "inputs/spis_V_bg.vtk",
        description="VTK file with ambient electrostatic background potential"
    )
    spacecraft: SpisSpacecraftSchema = Field(
        default_factory=SpisSpacecraftSchema,
        description="Spacecraft body weighting field and mesh"
    )
    antennas: List[Union[SpisAntennaSchema, str]] = Field(
        default_factory=lambda: [
            SpisAntennaSchema(weighting_file="inputs/spis_Vw_ant1.vtk"),
            SpisAntennaSchema(weighting_file="inputs/spis_Vw_ant2.vtk"),
            SpisAntennaSchema(weighting_file="inputs/spis_Vw_ant3.vtk"),
        ],
        description="List of antenna definitions or VTK file paths"
    )
    weighting_threshold: Annotated[float, Field(gt=0.0, le=1.0)] = Field(
        0.85,
        description="Conductor voxelization threshold from weighting potential (0.0 < threshold <= 1.0)"
    )
    spacecraft_weighting_file: Optional[str] = None
    spacecraft_surface_mesh_file: Optional[str] = None
    antenna_weighting_files: Optional[List[str]] = None

    @field_validator("antennas", mode="before")
    @classmethod
    def _normalize_antennas(cls, v: Any) -> Any:
        if isinstance(v, list):
            res = []
            for item in v:
                if isinstance(item, str):
                    res.append({"weighting_file": item})
                else:
                    res.append(item)
            return res
        return v


class AnalyticalSpacecraftPartSchema(StrictBaseModel):
    type: Literal["sphere", "box", "cube", "cylinder", "composite"] = Field(
        "sphere",
        description="Geometric primitive shape"
    )
    center: Annotated[List[float], Field(min_length=3, max_length=3)] = Field(
        default_factory=lambda: [0.0, 0.0, 0.0],
        description="Center coordinates [x0, y0, z0] in meters"
    )
    radius: Annotated[float, Field(gt=0.0)] = Field(
        1.0,
        description="Radius in meters (for sphere and cylinder)"
    )
    dimensions: Annotated[List[float], Field(min_length=3, max_length=3)] = Field(
        default_factory=lambda: [1.0, 1.0, 1.0],
        description="Side lengths [a, b, c] in meters (for box)"
    )
    side_length: Optional[Annotated[float, Field(gt=0.0)]] = Field(
        None,
        description="Equal side length for cube"
    )
    p_start: Annotated[List[float], Field(min_length=3, max_length=3)] = Field(
        default_factory=lambda: [0.0, 0.0, -1.0],
        description="Start point along cylinder axis"
    )
    p_end: Annotated[List[float], Field(min_length=3, max_length=3)] = Field(
        default_factory=lambda: [0.0, 0.0, 1.0],
        description="End point along cylinder axis"
    )
    voltage_V: Optional[float] = Field(
        None,
        description="Electrostatic potential of this part [V]"
    )
    parts: List["AnalyticalSpacecraftPartSchema"] = Field(
        default_factory=list,
        description="Sub-parts for composite spacecraft"
    )

    @field_validator("dimensions")
    @classmethod
    def _validate_dimensions(cls, v: List[float]) -> List[float]:
        if any(d <= 0.0 for d in v):
            raise ValueError(f"Box dimensions must all be positive: {v}")
        return v


class AnalyticalAntennaSchema(StrictBaseModel):
    p_start: Annotated[List[float], Field(min_length=3, max_length=3)] = Field(
        default_factory=lambda: [0.0, 0.0, 0.0],
        description="Antenna base coordinates [x1, y1, z1] in meters"
    )
    p_end: Annotated[List[float], Field(min_length=3, max_length=3)] = Field(
        default_factory=lambda: [1.0, 0.0, 0.0],
        description="Antenna tip coordinates [x2, y2, z2] in meters"
    )
    radius: Annotated[float, Field(gt=0.0)] = Field(
        0.015,
        description="Wire conductor physical radius in meters"
    )
    voltage_V: float = Field(
        0.0,
        description="Antenna bias potential [V]"
    )
    capacitance_F: Optional[Annotated[float, Field(gt=0.0)]] = Field(
        None,
        description="RC circuit input capacitance [F]"
    )
    resistance_Ohm: Optional[Annotated[float, Field(gt=0.0)]] = Field(
        None,
        description="RC circuit input resistance [Ohm]"
    )

    @model_validator(mode="after")
    def _validate_endpoints(self) -> "AnalyticalAntennaSchema":
        if self.p_start == self.p_end:
            raise ValueError(f"Antenna endpoints p_start and p_end cannot be identical: {self.p_start}")
        return self


class AnalyticalGeometrySchema(StrictBaseModel):
    spacecraft: AnalyticalSpacecraftPartSchema = Field(
        default_factory=AnalyticalSpacecraftPartSchema,
        description="Analytical spacecraft geometry definition"
    )
    antennas: List[AnalyticalAntennaSchema] = Field(
        default_factory=list,
        description="List of analytical wire antennas"
    )


class GeometrySchema(StrictBaseModel):
    source: Literal["spis", "analytical"] = Field(
        "spis",
        description="Geometry source mode: 'spis' (VTK fields) or 'analytical' (pure shapes)"
    )
    spis: SpisGeometrySchema = Field(
        default_factory=SpisGeometrySchema,
        description="SPIS VTK geometry configuration"
    )
    analytical: AnalyticalGeometrySchema = Field(
        default_factory=AnalyticalGeometrySchema,
        description="Analytical geometry configuration"
    )


class VTKFilesSchema(StrictBaseModel):
    spis_background_potential_file: str = "inputs/spis_V_bg.vtk"
    spacecraft_weighting_file: str = "inputs/spis_Vw_body.vtk"
    antenna_weighting_files: List[str] = Field(
        default_factory=lambda: [
            "inputs/spis_Vw_ant1.vtk",
            "inputs/spis_Vw_ant2.vtk",
            "inputs/spis_Vw_ant3.vtk",
        ]
    )


class TogglesSchema(StrictBaseModel):
    enable_spis_background_field: bool = Field(
        True,
        description="Include static background electric field in particle motion"
    )
    enable_plasma_self_field: bool = Field(
        True,
        description="Solve self-consistent Poisson space-charge potential"
    )
    enable_antenna_particle_collection: bool = Field(
        True,
        description="Collect particles hitting antenna conductors and compute I_col"
    )
    enable_rc_circuit_response: bool = Field(
        True,
        description="Integrate RC circuit differential equations for antenna voltages"
    )
    enable_antenna_bias_voltage: bool = Field(
        True,
        description="Initialize antenna bias voltages at t=0"
    )


class ImpactSchema(StrictBaseModel):
    location: Annotated[List[float], Field(min_length=3, max_length=3)] = Field(
        default_factory=lambda: [-2.0, 2.0, 0.0],
        description="Initial coordinates of dust grain before impact [m]"
    )
    direction: Annotated[List[float], Field(min_length=3, max_length=3)] = Field(
        default_factory=lambda: [0.0, 0.0, 0.0],
        description="Flight direction vector of dust grain [vx, vy, vz]"
    )
    time_delay_s: Annotated[float, Field(ge=0.0)] = Field(
        1e-6,
        description="Time delay before cloud expansion begins [s]"
    )
    normal: Optional[Annotated[List[float], Field(min_length=3, max_length=3)]] = Field(
        None,
        description="Target surface normal vector [nx, ny, nz]"
    )


class PhysicsSchema(StrictBaseModel):
    ion_mass_amu: Annotated[float, Field(gt=0.0)] = Field(
        27.0,
        description="Ion mass in atomic mass units (e.g. 27.0 for Al)"
    )
    impact_cloud_temperature_eV: Annotated[float, Field(gt=0.0)] = Field(
        2.0,
        description="Initial kinetic temperature of impact cloud [eV]"
    )
    solar_wind_electron_temp_eV: Annotated[float, Field(gt=0.0)] = Field(
        15.0,
        description="Ambient solar wind electron temperature [eV]"
    )
    solar_wind_density_m3: Annotated[float, Field(gt=0.0)] = Field(
        1e7,
        description="Ambient solar wind plasma density [m^-3]"
    )
    total_impact_charge_C: Annotated[float, Field(gt=0.0)] = Field(
        50e-12,
        description="Total charge released by dust impact [C]"
    )
    plasma_injection_mode: Literal["point_cloud", "homogeneous"] = Field(
        "point_cloud",
        description="Injection pattern: 'point_cloud' (impact spot) or 'homogeneous' (solar wind domain)"
    )
    impact: ImpactSchema = Field(
        default_factory=ImpactSchema,
        description="Dust grain impact trajectory and timing"
    )
    spacecraft_voltage_V: Optional[float] = Field(
        None,
        description="Spacecraft body potential override [V]"
    )
    antenna_capacitance_F: Optional[List[Annotated[float, Field(gt=0.0)]]] = Field(
        None,
        description="Capacitances per antenna [F]"
    )
    antenna_resistance_Ohm: Optional[List[Annotated[float, Field(gt=0.0)]]] = Field(
        None,
        description="Sheath resistances per antenna [Ohm]"
    )
    antenna_bias_voltage_V: Optional[List[float]] = Field(
        None,
        description="Bias potentials per antenna [V]"
    )


class NumericSchema(StrictBaseModel):
    num_macroparticles: Annotated[int, Field(gt=0)] = Field(
        20000,
        description="Total number of macroparticles in PIC simulation"
    )
    time_step_s: Annotated[float, Field(gt=0.0)] = Field(
        2e-9,
        description="Discrete simulation time step Delta t [s]"
    )
    num_time_steps: Optional[Annotated[int, Field(gt=0)]] = Field(
        10000,
        description="Total number of simulation steps"
    )
    simulation_duration_s: Optional[Annotated[float, Field(gt=0.0)]] = Field(
        None,
        description="Physical simulation duration in seconds (T = N_steps * dt)"
    )
    domain_half_length_m: Annotated[List[float], Field(min_length=3, max_length=3)] = Field(
        default_factory=lambda: [5.0, 5.0, 5.0],
        description="Domain half-lengths [Lx, Ly, Lz] in meters (or single float for isotropic)"
    )
    grid_nodes: Annotated[List[int], Field(min_length=3, max_length=3)] = Field(
        default_factory=lambda: [35, 35, 35],
        description="Rectilinear grid resolution [Nx, Ny, Nz] (or single integer)"
    )
    field_precision: Literal["float32", "float64"] = Field(
        "float32",
        description="Floating-point precision for 3D static background and weighting grids ('float32' saves 50% RAM)"
    )

    @field_validator("domain_half_length_m", mode="before")
    @classmethod
    def _normalize_domain_half_length(cls, v: Any) -> Any:
        if isinstance(v, (int, float)):
            val = float(v)
            return [val, val, val]
        return v

    @field_validator("grid_nodes", mode="before")
    @classmethod
    def _normalize_grid_nodes(cls, v: Any) -> Any:
        if isinstance(v, (int, float)):
            val = int(v)
            return [val, val, val]
        return v

    @field_validator("domain_half_length_m")
    @classmethod
    def _validate_domain_half_length(cls, v: List[float]) -> List[float]:
        if any(l <= 0.0 for l in v):
            raise ValueError(f"Domain half-lengths must all be positive: {v}")
        return v

    @field_validator("grid_nodes")
    @classmethod
    def _validate_grid_nodes(cls, v: List[int]) -> List[int]:
        if any(n < 2 for n in v):
            raise ValueError(f"Grid nodes in each dimension must be at least 2: {v}")
        return v

    @model_validator(mode="after")
    def _resolve_steps_and_duration(self) -> "NumericSchema":
        if self.num_time_steps is None and self.simulation_duration_s is not None:
            self.num_time_steps = int(round(self.simulation_duration_s / self.time_step_s))
        elif self.num_time_steps is None:
            self.num_time_steps = 10000
        return self


class PlottingSchema(StrictBaseModel):
    run_physical_simulation: bool = Field(True, description="Run PIC simulation solver")
    show_interactive_gui_windows: bool = Field(False, description="Open interactive Matplotlib windows")
    save_plots_to_disk: bool = Field(True, description="Save static PNG/GIF plots to disk")
    export_csv_time_series: bool = Field(False, description="Export antenna currents/voltages to CSV")

    show_currents: bool = Field(True, description="Plot antenna currents and voltages")
    show_fields_slice: bool = Field(True, description="Generate potential slice animation")
    show_particles_3d: bool = Field(True, description="Generate 3D particle animation")
    show_velocity_anim: bool = Field(True, description="Generate velocity phase space animation")

    output_format: Literal["h5", "npz"] = Field("h5", description="Primary data storage format ('h5' or 'npz')")
    output_h5_filepath: str = "outputs/out_3d_vysledky.h5"
    output_npz_filepath: str = "outputs/out_3d_vysledky.npz"
    output_csv_filepath: str = "outputs/out_3d_vysledky_simulace.csv"

    enable_checkpointing: bool = Field(True, description="Enable periodic simulation state checkpoints")
    checkpoint_interval_steps: Annotated[int, Field(gt=0)] = 500

    export_vtk: bool = Field(False, description="Export time series to ParaView VTK/PVD files")
    vtk_output_dir: str = "outputs/paraview_vtk"

    file_currents: str = "outputs/out_3d_proudy_napeti.png"
    file_fields_anim: str = "outputs/out_3d_animace_pole_potencial.gif"
    file_particles_anim: str = "outputs/out_3d_animace_pozice_castic.gif"
    file_velocity_anim: str = "outputs/out_3d_animace_rychlosti.gif"


class SimulationConfigSchema(StrictBaseModel):
    """
    Root configuration schema representing the complete config.json file.
    Validates all sections with fail-fast strictness and type checking.
    """
    schema_: Optional[str] = Field(default=None, alias="$schema", description="URI to JSON Schema")
    geometry: GeometrySchema = Field(default_factory=GeometrySchema, description="Geometry setup")
    toggles: TogglesSchema = Field(default_factory=TogglesSchema, description="Physics feature flags")
    physics: PhysicsSchema = Field(default_factory=PhysicsSchema, description="Plasma and impact parameters")
    numeric: NumericSchema = Field(default_factory=NumericSchema, description="Discretization and time stepping")
    plotting: PlottingSchema = Field(default_factory=PlottingSchema, description="Outputs and visualization")
    vtk_files: Optional[VTKFilesSchema] = None

    @model_validator(mode="after")
    def _sync_legacy_vtk(self) -> "SimulationConfigSchema":
        if self.vtk_files is not None:
            self.geometry.spis.background_potential_file = self.vtk_files.spis_background_potential_file
            self.geometry.spis.spacecraft.weighting_file = self.vtk_files.spacecraft_weighting_file
            self.geometry.spis.spacecraft_weighting_file = self.vtk_files.spacecraft_weighting_file
            self.geometry.spis.antenna_weighting_files = list(self.vtk_files.antenna_weighting_files)
            self.geometry.spis.antennas = [
                SpisAntennaSchema(weighting_file=f)
                for f in self.vtk_files.antenna_weighting_files
            ]
        return self


def format_validation_error(err: ValidationError) -> str:
    """Format Pydantic ValidationError into an ASCII-safe, readable diagnostic message."""
    lines = ["[CHYBA KONFIGURACE] Neplatne nastaveni v konfiguracnim souboru:"]
    for error in err.errors():
        loc = " -> ".join(str(p) for p in error.get("loc", []))
        msg = error.get("msg", "")
        inp = error.get("input", None)
        inp_str = f" (predana hodnota: {inp!r})" if inp is not None else ""
        lines.append(f"  * Pole [{loc}]: {msg}{inp_str}")
    return "\n".join(lines)
