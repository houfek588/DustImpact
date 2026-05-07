#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pyvista as pv
from dataclasses import dataclass, field, asdict, fields
from typing import Tuple, List, Dict, Any
import json
import os

# Fyzikální konstanty
amu = 1.66053906660e-27  # kg
e = 1.602176634e-19  # C
m_e = 9.1093837015e-31  # kg
eps_0 = 8.8541878128e-12  # F/m

# Parametry solárního větru pro výpočet Debyeovy délky
Te_eV = 15.0
n_sw = 1e7


@dataclass
class SimulationToggles3D:
    enable_background_field: bool = True
    enable_self_field: bool = True
    enable_antenna_collection: bool = True
    enable_rc_circuit: bool = True


@dataclass
class VTKFilesConfig:
    background_potential: str = "spis_V_bg.vtk"
    antenna_weighting: List[str] = field(default_factory=lambda: [
        "spis_Vw_ant1.vtk",
        "spis_Vw_ant2.vtk",
        "spis_Vw_ant3.vtk"
    ])


@dataclass
class PlottingConfig3D:
    show_currents: bool = True
    show_fields_slice: bool = True
    show_particles_3d: bool = True
    show_velocity_anim: bool = True
    show_phase_space_anim: bool = False
    save_plots: bool = False
    export_data_csv: bool = False
    file_results_npz: str = "out_3d_vysledky.npz"
    file_csv: str = "out_3d_vysledky_simulace.csv"
    file_currents: str = "out_3d_proudy_napeti.png"
    file_fields_anim: str = "out_3d_animace_pole_potencial.gif"
    file_particles_anim: str = "out_3d_animace_pozice_castic.gif"
    file_velocity_anim: str = "out_3d_animace_rychlosti.gif"
    file_phase_space: str = "out_3d_animace_fazovy_prostor.gif"


@dataclass
class SimulationParams3D:
    Vf: float
    vtk_files: VTKFilesConfig = field(default_factory=VTKFilesConfig)

    m_i_amu: float = 27.0
    T_dust_eV: float = 2.0
    N_particles: int = 20000

    dt: float = 2e-9
    t_max: float = 20e-6
    t_delay: float = 1e-6

    # Velikost domény (rozsah bude od -L do +L)
    L_x: float = 5.0
    L_y: float = 5.0
    L_z: float = 5.0

    # Výchozí pozice dopadu (Změněno na 0,0,0 pro centrovanou sondu)
    impact_pos: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])

    # 3D Mřížka: 35x35x35 = 42 875 uzlů (rozumný kompromis paměti a rychlosti)
    Nx: int = 35
    Ny: int = 35
    Nz: int = 35

    C_ant: List[float] = field(default_factory=lambda: [2e-12, 2e-12, 2e-12])
    R_ant: List[float] = field(default_factory=lambda: [100e3, 100e3, 100e3])
    collection_eff: List[float] = field(default_factory=lambda: [0.8, 0.8, 0.8])

    # Interní proměnné
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

    def __post_init__(self):
        self.m_i = self.m_i_amu * amu
        self.debye_length = np.sqrt((eps_0 * Te_eV * e) / (n_sw * e ** 2))
        self.q_macro = 50e-12 / self.N_particles
        self.steps = int(self.t_max / self.dt)
        self.time_array = np.linspace(0, self.t_max, self.steps)

        # Mřížka je nyní od -L do +L ve všech dimenzích
        self.x_grid = np.linspace(-self.L_x, self.L_x, self.Nx)
        self.y_grid = np.linspace(-self.L_y, self.L_y, self.Ny)
        self.z_grid = np.linspace(-self.L_z, self.L_z, self.Nz)

        self.dx = self.x_grid[1] - self.x_grid[0]
        self.dy = self.y_grid[1] - self.y_grid[0]
        self.dz = self.z_grid[1] - self.z_grid[0]

        self.plot_stride = max(1, self.N_particles // 1500)
        self.save_interval = max(1, self.steps // 50)


def setup_simulation_parameters_3d(Vf: float, config_file: str = "config_3d.json") -> Tuple[
    SimulationParams3D, SimulationToggles3D, PlottingConfig3D]:
    """ Načte nebo vytvoří JSON konfiguraci pro 3D. """
    if not os.path.exists(config_file):
        print(f"Vytvářím výchozí šablonu konfigurace 3D: {config_file}")
        dummy_params = SimulationParams3D(Vf=0.0)
        params_dict = {f.name: getattr(dummy_params, f.name) for f in fields(SimulationParams3D) if
                       f.init and f.name != 'Vf'}
        params_dict['vtk_files'] = asdict(dummy_params.vtk_files)

        default_config = {
            "toggles": asdict(SimulationToggles3D()),
            "params": params_dict,
            "plotting": asdict(PlottingConfig3D())
        }
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(default_config, f, indent=4, ensure_ascii=False)

    with open(config_file, 'r', encoding='utf-8') as f:
        cfg = json.load(f)

    toggles = SimulationToggles3D(**cfg.get('toggles', {}))

    p_kwargs = cfg.get('params', {})
    p_kwargs['Vf'] = Vf
    if 'vtk_files' in p_kwargs:
        p_kwargs['vtk_files'] = VTKFilesConfig(**p_kwargs['vtk_files'])

    params = SimulationParams3D(**p_kwargs)
    plot_config = PlottingConfig3D(**cfg.get('plotting', {}))

    return params, toggles, plot_config


def _extract_potential_from_mesh(sampled_mesh, filename: str) -> np.ndarray:
    possible_names = [
        'Potential', 'Potencial', 'V', 'PlasmaPotential', 'U',
        'plasma pot at t = 1.0 s'
    ]
    available_keys = list(sampled_mesh.point_data.keys())

    for name in possible_names:
        if name in available_keys:
            return sampled_mesh.point_data[name]

    raise KeyError(
        f"\n[CHYBA] V souboru '{filename}' se nepodařilo najít pole s potenciálem.\n"
        f"Dostupná datová pole v tomto souboru jsou: {available_keys}\n"
    )


def load_and_interpolate_vtk(params: SimulationParams3D):
    """ Voxelizace SPIS mřížky a příprava polí. """
    print("Mapování SPIS VTK na 3D pravoúhlou mřížku...")

    # Počátek (origin) nyní začíná v záporných souřadnicích (-L_x, -L_y, -L_z)
    pic_grid = pv.ImageData(
        dimensions=(params.Nx, params.Ny, params.Nz),
        spacing=(params.dx, params.dy, params.dz),
        origin=(-params.L_x, -params.L_y, -params.L_z)
    )

    Nx, Ny, Nz = params.Nx, params.Ny, params.Nz
    X, Y, Z = np.meshgrid(params.x_grid, params.y_grid, params.z_grid, indexing='ij')

    # 1. Pole pozadí (V_bg)
    try:
        mesh_bg = pv.read(params.vtk_files.background_potential)
        sampled_bg = pic_grid.sample(mesh_bg)
        pot_data = _extract_potential_from_mesh(sampled_bg, params.vtk_files.background_potential)
        V_bg_grid = pot_data.reshape((Nx, Ny, Nz))
        print(f"  -> Pozadí načteno a voxelizováno.")
    except FileNotFoundError:
        print("  -> Upozornění: VTK pozadí nenalezeno. Vytvářím syntetické analytické pole...")
        V_bg_grid = np.zeros((Nx, Ny, Nz))
        # Syntetická sonda uprostřed prostoru
        r_sq = X ** 2 + Y ** 2 + Z ** 2
        mask_sonda = r_sq < 0.5 ** 2
        V_bg_grid[mask_sonda] = params.Vf

    Ex_bg, Ey_bg, Ez_bg = np.gradient(-V_bg_grid, params.dx, params.dy, params.dz)

    # NOVÉ: VYTVOŘENÍ 3D MASKY PRO TĚLO SONDY
    # Kov sondy je nabitý na Vf. Kvůli interpolaci může být hodnota mírně nepřesná,
    # použijeme proto drobnou toleranci 1% pro detekci kovu.
    tolerance = abs(params.Vf) * 0.01 if params.Vf != 0 else 0.01
    spacecraft_mask_3d = np.abs(V_bg_grid - params.Vf) <= tolerance

    # 2. Váhová pole a Voxelizace antén
    Vw_grids = []
    Ewx_list, Ewy_list, Ewz_list = [], [], []
    antenna_masks_3d = []

    synth_pos = [(2.5, 0.0, 0.0), (-1.25, 2.16, 0.0), (-1.25, -2.16, 0.0)]

    for i, vtk_file in enumerate(params.vtk_files.antenna_weighting):
        try:
            mesh_w = pv.read(vtk_file)
            sampled_w = pic_grid.sample(mesh_w)
            pot_data = _extract_potential_from_mesh(sampled_w, vtk_file)
            Vw = pot_data.reshape((Nx, Ny, Nz))
            print(f"  -> Váhové pole antény {i + 1} načteno.")
        except FileNotFoundError:
            print(f"  -> Upozornění: VTK váhy {vtk_file} nenalezeno. Vytvářím syntetickou anténu {i + 1}...")
            pos = synth_pos[i % len(synth_pos)]
            r_sq = (X - pos[0]) ** 2 + (Y - pos[1]) ** 2 + (Z - pos[2]) ** 2
            Vw = np.exp(-r_sq / 0.4 ** 2)

        Vw_grids.append(Vw)
        Ewx, Ewy, Ewz = np.gradient(-Vw, params.dx, params.dy, params.dz)
        Ewx_list.append(Ewx)
        Ewy_list.append(Ewy)
        Ewz_list.append(Ewz)

        mask = Vw > 0.95
        antenna_masks_3d.append(mask)

    return V_bg_grid, Vw_grids, Ex_bg, Ey_bg, Ez_bg, Ewx_list, Ewy_list, Ewz_list, antenna_masks_3d, spacecraft_mask_3d


def save_results_npz(results: Dict[str, Any], filepath: str) -> None:
    print(f"Ukládám 3D výsledky do: {filepath}")
    np.savez_compressed(
        filepath,
        smooth_induced=np.array(results['smooth_induced']),
        smooth_collected=np.array(results['smooth_collected']),
        smooth_total=np.array(results['smooth_total']),
        voltage_ant=np.array(results['voltage_ant']),
        hist_V=np.array(results['history']['V']),
        hist_t=np.array(results['history']['t']),
        hist_x_e=np.array(results['history']['x_e']),
        hist_y_e=np.array(results['history']['y_e']),
        hist_z_e=np.array(results['history']['z_e']),
        hist_x_i=np.array(results['history']['x_i']),
        hist_y_i=np.array(results['history']['y_i']),
        hist_z_i=np.array(results['history']['z_i']),
        hist_vx_e=np.array(results['history']['vx_e']),
        hist_vy_e=np.array(results['history']['vy_e']),
        hist_vz_e=np.array(results['history']['vz_e']),
        hist_vx_i=np.array(results['history']['vx_i']),
        hist_vy_i=np.array(results['history']['vy_i']),
        hist_vz_i=np.array(results['history']['vz_i'])
    )


def load_results_npz(filepath: str) -> Dict[str, Any]:
    print(f"Načítám 3D výsledky z: {filepath}")
    with np.load(filepath, allow_pickle=True) as d:
        return {
            'smooth_induced': d['smooth_induced'], 'smooth_collected': d['smooth_collected'],
            'smooth_total': d['smooth_total'], 'voltage_ant': d['voltage_ant'],
            'history': {
                'V': d['hist_V'], 't': d['hist_t'],
                'x_e': d['hist_x_e'], 'y_e': d['hist_y_e'], 'z_e': d['hist_z_e'],
                'x_i': d['hist_x_i'], 'y_i': d['hist_y_i'], 'z_i': d['hist_z_i'],
                'vx_e': d['hist_vx_e'], 'vy_e': d['hist_vy_e'], 'vz_e': d['hist_vz_e'],
                'vx_i': d['hist_vx_i'], 'vy_i': d['hist_vy_i'], 'vz_i': d['hist_vz_i']
            }
        }