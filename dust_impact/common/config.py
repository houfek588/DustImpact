# -*- coding: utf-8 -*-
"""
Common base dataclasses and configuration structures for DustImpact simulations (2D and 3D).
"""

import numpy as np
from dataclasses import dataclass, field
from dust_impact.physics.constants import amu, e, eps_0, m_e


@dataclass
class BaseSimulationToggles:
    enable_spis_background_field: bool = True
    enable_plasma_self_field: bool = True
    enable_antenna_particle_collection: bool = True
    enable_rc_circuit_response: bool = True
    enable_debye_screening: bool = True
    enable_antenna_bias_voltage: bool = True


@dataclass
class BaseSimulationParams:
    Vf: float = 0.0
    Vf_antenne: float = 0.0

    ion_mass_amu: float = 27.0
    impact_cloud_temperature_eV: float = 2.0
    num_macroparticles: int = 20000

    time_step_s: float = 2e-9
    simulation_duration_s: float = 20e-6
    impact_time_delay_s: float = 1e-6

    solar_wind_electron_temp_eV: float = 15.0
    solar_wind_density_m3: float = 1e7

    # Derived internal parameters
    m_i: float = field(init=False)
    debye_length: float = field(init=False)
    q_macro: float = field(init=False)
    steps: int = field(init=False)
    time_array: np.ndarray = field(init=False)
    plot_stride: int = field(init=False)
    save_interval: int = field(init=False)

    # Convenience aliases
    N_particles: int = field(init=False)
    T_dust_eV: float = field(init=False)
    dt: float = field(init=False)
    t_max: float = field(init=False)
    t_delay: float = field(init=False)

    def _init_base_derived_params(self):
        self.m_i = self.ion_mass_amu * amu
        self.debye_length = np.sqrt((eps_0 * self.solar_wind_electron_temp_eV * e) / (self.solar_wind_density_m3 * e ** 2))
        self.q_macro = 50e-12 / self.num_macroparticles
        self.steps = int(self.simulation_duration_s / self.time_step_s)
        self.time_array = np.linspace(0, self.simulation_duration_s, self.steps)

        self.N_particles = self.num_macroparticles
        self.T_dust_eV = self.impact_cloud_temperature_eV
        self.dt = self.time_step_s
        self.t_max = self.simulation_duration_s
        self.t_delay = self.impact_time_delay_s

        self.plot_stride = max(1, self.num_macroparticles // 1500)
        self.save_interval = max(1, self.steps // 50)
