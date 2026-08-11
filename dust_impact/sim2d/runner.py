#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main runner for 2D PIC Simulation using self-explanatory parameters.
"""

from dust_impact.physics.charging import calculate_equilibrium_potential, ENV_EARTH, MAT_ALUMINIUM, MAT_ALUMINIUM_ANTENNE
from dust_impact.common.io import save_results_npz, load_results_npz
from dust_impact.sim2d.input_data import setup_simulation_parameters_2d
from dust_impact.sim2d.sim_core import DustImpactSimulation2D
from dust_impact.sim2d.plotting import plot_simulation_results_2d


def run_2d_simulation(config_file: str = "config.json", run_calc: bool = True, visualize_results: bool = None):
    V_equilibrium = calculate_equilibrium_potential(ENV_EARTH, MAT_ALUMINIUM)
    V_equilibrium_antenne = calculate_equilibrium_potential(ENV_EARTH, MAT_ALUMINIUM_ANTENNE)
    print(f"=== KROK 1: Rovnovážné nabití ===")
    print(f"Plovoucí potenciál sondy: {V_equilibrium:.3f} V")

    print(f"Načítám konfiguraci z: {config_file}")
    sim_params, sim_toggles, plot_config = setup_simulation_parameters_2d(
        V_equilibrium, V_equilibrium_antenne, config_file=config_file
    )

    if visualize_results is not None:
        plot_config.show_interactive_gui_windows = visualize_results

    PROVEST_VYPOCET = run_calc and getattr(plot_config, 'run_physical_simulation', True)

    if PROVEST_VYPOCET:
        print("\n=== KROK 2: Spouštím fyzikální 2D PIC simulaci ===")
        sim = DustImpactSimulation2D(sim_params, sim_toggles)
        sim_results = sim.run()

        print("\n=== KROK 3: Ukládání fyzikálních dat na disk ===")
        save_results_npz(sim_results, plot_config.output_npz_filepath)
    else:
        print("\n=== KROK 2 & 3: PŘESKOČEN (Simulace vypnuta) ===")

    if getattr(plot_config, 'show_interactive_gui_windows', True) or getattr(plot_config, 'save_plots_to_disk', False):
        print("\n=== KROK 4: Načítání a Vizualizace ===")
        try:
            loaded_results = load_results_npz(plot_config.output_npz_filepath)
            plot_simulation_results_2d(loaded_results, sim_params, plot_config)
        except FileNotFoundError:
            print(f"[CHYBA] Soubor {plot_config.output_npz_filepath} nebyl nalezen.")


if __name__ == "__main__":
    run_2d_simulation()
