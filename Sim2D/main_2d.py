#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from charging import calculate_equilibrium_potential, ENV_EARTH, MAT_ALUMINIUM, MAT_ALUMINIUM_ANTENNE
from input_data_2d import setup_simulation_parameters_2d
import sim_core_2d as core
import plotting_2d as plt



if __name__ == "__main__":
    V_equilibrium = calculate_equilibrium_potential(ENV_EARTH, MAT_ALUMINIUM)
    V_equilibrium_antenne = calculate_equilibrium_potential(ENV_EARTH, MAT_ALUMINIUM_ANTENNE)
    print(f"=== KROK 1: Rovnovážné nabití ===")
    print(f"Plovoucí potenciál sondy: {V_equilibrium:.3f} V")

    # Inicializace 2D Datových struktur a konfigurace z JSONu
    sim_params, sim_toggles, plot_config = setup_simulation_parameters_2d(V_equilibrium, V_equilibrium_antenne)

    # Vytvoření instance a výpočet 2DPIC
    sim = core.DustImpactSimulation2D(sim_params, sim_toggles)
    sim_results = sim.run()

    # Vykreslování
    plt.plot_simulation_results_2d(sim_results, sim_params, plot_config)