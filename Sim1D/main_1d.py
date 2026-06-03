#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from charging import calculate_equilibrium_potential, ENV_EARTH, MAT_ALUMINIUM
import input_data_1d as input_data
import sim_core_1d as core
import plotting_1d as plt




# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # 1. Spočítat rovnováhu
    # V_equilibrium = get_equilibrium_potential()
    V_equilibrium = calculate_equilibrium_potential(ENV_EARTH, MAT_ALUMINIUM)
    print(f"=== STEP 1: Equilibrium charging ===")
    print(f"Calculated spacecraft potential: {V_equilibrium:.3f} V")

    # 2. Setup (Toggles can be changed inside the setup function)
    sim_params, sim_toggles = input_data.setup_simulation_parameters(V_equilibrium)
    # sim_params, sim_toggles = one_file.setup_simulation_parameters(V_equilibrium)

    # 3. Kinetický výpočet dopadu (Core Solver module)
    sim = core.DustImpactSimulation(sim_params, sim_toggles)
    # sim = one_file.DustImpactSimulation(sim_params, sim_toggles)

    sim_results = sim.run()

    # 4. Vizualizace (Plotting module)
    plt.plot_simulation_results(sim_results, sim_params)