#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from input_data_2d import setup_simulation_parameters_2d
import sim_core_2d as core
import plotting_2d as plt

# Zjednodušená "mock" funkce pro napětí sondy, aby kód okamžitě fungoval
def get_spacecraft_voltage() -> float:
    return 5.0

if __name__ == "__main__":
    V_equilibrium = get_spacecraft_voltage()
    print(f"=== KROK 1: Rovnovážné nabití ===")
    print(f"Plovoucí potenciál sondy: {V_equilibrium:.3f} V")

    # Inicializace 2D Datových struktur
    sim_params, sim_toggles = setup_simulation_parameters_2d(V_equilibrium)

    # Vytvoření instance a výpočet 2DPIC
    sim = core.DustImpactSimulation2D(sim_params, sim_toggles)
    sim_results = sim.run()

    # # Vykreslování
    # plt.plot_simulation_results_2d(sim_results, sim_params)

    # Vykreslování a uložení na disk
    moje_viz_nastaveni = {
        'show_currents': True,
        'show_fields_anim': True,
        'show_particles_anim': True,
        'show_velocity_anim': True,
        'save_plots': True  # Tímto zapnete vygenerování a uložení .png a .gif souborů
    }

    plt.plot_simulation_results_2d(sim_results, sim_params, plot_toggles=moje_viz_nastaveni)
