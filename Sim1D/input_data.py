import numpy as np
from dataclasses import dataclass, field
from typing import Tuple
from global_const import *

# =============================================================================
# PART 1: Helper Functions for Weighting Field
# =============================================================================

def calc_Vw(x: np.ndarray, x_antenna: float, w_width: float) -> np.ndarray:
    """ Narrow Gaussian weighting potential centered at the antenna """
    return np.exp(-((x - x_antenna) / w_width) ** 2)

def calc_Ew(x: np.ndarray, x_antenna: float, w_width: float) -> np.ndarray:
    """ Weighting field Ew(x) = - d/dx Vw(x) """
    return np.exp(-((x - x_antenna) / w_width) ** 2)

# =============================================================================
# MODULE A: Setup Data and Parameters
# =============================================================================

@dataclass
class SimulationToggles:
    enable_background_field: bool = True  # Field from spacecraft charging (Vf)
    enable_self_field: bool = True        # Poisson solver for plasma cloud space-charge
    enable_antenna_collection: bool = True # Physical impacts on antenna grid
    enable_rc_circuit: bool = True        # RC discharge of collected current
    enable_antenna_bias: bool = True

@dataclass
class SimulationParams:
    Vf: float
    
    # Physics constants
    m_i: float = 1 * amu
    T_dust_eV: float = 2.0
    N_particles: int = 10000
    
    # Domain and Time
    dt: float = 1e-9
    t_max: float = 30e-6
    t_delay: float = 1e-6
    L_domain: float = 5.0
    
    # Antenna & RC
    x_antenna: float = 2.5
    V_ant_bias: float = 8.0  # Konstantní hodnota nabití antény
    w_width: float = 0.2
    collection_efficiency: float = 0.80
    C_ant: float = 2e-12
    R_ant: float = 100e3
    
    # Grid
    N_grid: int = 1000
    A_sim: float = 20.0
    integrator: str = 'rk4'
    
    # Calculated properties
    debye_length: float = field(init=False)
    v_th_e: float = field(init=False)
    v_th_i: float = field(init=False)
    q_macro: float = field(init=False)
    steps: int = field(init=False)
    time_array: np.ndarray = field(init=False)
    bin_edges: np.ndarray = field(init=False)
    x_grid: np.ndarray = field(init=False)
    dx: float = field(init=False)
    plot_stride: int = field(init=False)
    save_interval: int = field(init=False)

    def __post_init__(self):
        self.debye_length = np.sqrt((eps_0 * Te_eV * e) / (n_sw * e ** 2))
        self.v_th_e = np.sqrt(2 * e * self.T_dust_eV / m_e)
        self.v_th_i = np.sqrt(2 * e * self.T_dust_eV / self.m_i)
        self.q_macro = 1e-12 / self.N_particles
        
        self.steps = int(self.t_max / self.dt)
        self.time_array = np.linspace(0, self.t_max, self.steps)
        
        self.bin_edges = np.linspace(0, self.L_domain, self.N_grid + 1)
        self.x_grid = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2.0
        self.dx = self.L_domain / self.N_grid
        
        self.plot_stride = max(1, self.N_particles // 2000)
        self.save_interval = max(1, self.steps // 50)

def setup_simulation_parameters(Vf: float) -> Tuple[SimulationParams, SimulationToggles]:
    """ Initializes and returns simulation parameters and physics toggles. """
    toggles = SimulationToggles()
    params = SimulationParams(Vf=Vf)
    return params, toggles
