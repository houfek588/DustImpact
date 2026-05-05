#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from charging import calculate_equilibrium_potential, ENV_EARTH, MAT_ALUMINIUM, MAT_ALUMINIUM_ANTENNE
from input_data_2d import setup_simulation_parameters_2d, save_results_npz, load_results_npz
import sim_core_2d as core
import plotting_2d as plt



if __name__ == "__main__":
    V_equilibrium = calculate_equilibrium_potential(ENV_EARTH, MAT_ALUMINIUM)
    V_equilibrium_antenne = calculate_equilibrium_potential(ENV_EARTH, MAT_ALUMINIUM_ANTENNE)
    print(f"=== KROK 1: Rovnovážné nabití ===")
    print(f"Plovoucí potenciál sondy: {V_equilibrium:.3f} V")

    # 1. Inicializace Datových struktur a konfigurace z JSONu
    sim_params, sim_toggles, plot_config = setup_simulation_parameters_2d(V_equilibrium, V_equilibrium_antenne)

    # ---------------------------------------------------------
    # WORKFLOW PŘEPÍNAČ
    # ---------------------------------------------------------
    # Změňte na False, pokud máte data již spočítaná v souboru
    # a chcete se věnovat jen interaktivnímu prohlížení/ladění grafů.
    PROVEST_VYPOCET = True

    if PROVEST_VYPOCET:
        print("\n=== KROK 2: Spouštím fyzikální PIC simulaci ===")
        sim = core.DustImpactSimulation2D(sim_params, sim_toggles)
        sim_results = sim.run()

        print("\n=== KROK 3: Ukládání fyzikálních dat na disk ===")
        save_results_npz(sim_results, plot_config.file_results_npz)
    else:
        print("\n=== KROK 2 & 3: PŘESKOČEN (Fyzikální simulace vypnuta) ===")

    # ---------------------------------------------------------
    # NEZÁVISLÝ VIZUALIZAČNÍ BLOK (Načítá vždy z disku)
    # ---------------------------------------------------------
    print("\n=== KROK 4: Načítání a Vizualizace ===")
    try:
        loaded_results = load_results_npz(plot_config.file_results_npz)
        plt.plot_simulation_results_2d(loaded_results, sim_params, plot_config)
    except FileNotFoundError:
        print(f"[CHYBA] Soubor {plot_config.file_results_npz} nebyl nalezen. Prosím, nastavte PROVEST_VYPOCET = True pro jeho prvotní vygenerování.")