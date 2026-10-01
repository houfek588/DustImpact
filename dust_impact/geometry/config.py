# -*- coding: utf-8 -*-
"""
Configuration dataclasses for 3D Geometry definitions (SPIS VTK meshes and analytical shapes).
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class SpisSpacecraftConfig:
    weighting_file: str = "inputs/spis_Vw_body.vtk"
    surface_mesh_file: str = ""
    voltage_V: Optional[float] = None

    @property
    def effective_voltage(self) -> Optional[float]:
        return float(self.voltage_V) if self.voltage_V is not None else None


@dataclass
class SpisAntennaGeometry:
    weighting_file: str = ""
    voltage_V: Optional[float] = None
    capacitance_F: Optional[float] = None
    resistance_Ohm: Optional[float] = None

    @property
    def effective_voltage(self) -> Optional[float]:
        return float(self.voltage_V) if self.voltage_V is not None else None


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
    parts: List['AnalyticalSpacecraftPart'] = field(default_factory=list)

    @property
    def effective_voltage(self) -> Optional[float]:
        return float(self.voltage_V) if self.voltage_V is not None else None


@dataclass
class AnalyticalAntennaGeometry:
    p_start: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    p_end: List[float] = field(default_factory=lambda: [1.0, 0.0, 0.0])
    radius: float = 0.015
    voltage_V: float = 0.0
    capacitance_F: Optional[float] = None
    resistance_Ohm: Optional[float] = None

    @property
    def effective_voltage(self) -> float:
        return float(self.voltage_V) if self.voltage_V is not None else 0.0


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
