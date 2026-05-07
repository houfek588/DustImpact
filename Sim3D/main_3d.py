#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from input_data_3d import setup_simulation_parameters_3d, load_and_interpolate_vtk
from input_data_3d import save_results_npz, load_results_npz
import sim_core_3d as core
import plotting_3d as plt

if __name__ == "__main__":
    V_equilibrium = 5.0  # Můžete propojit s Vaším modelem pro výpočet nabíjení

    print(f"=== KROK 1: Rovnovážné nabití ===")
    print(f"Plovoucí potenciál sondy: {V_equilibrium:.3f} V")

    # 1. Konfigurace a načtení VTK (SPIS data nebo Syntetický fallback)
    sim_params, sim_toggles, plot_config = setup_simulation_parameters_3d(V_equilibrium)

    # NOVÉ: Vytažení masky sondy ze čtečky dat
    V_bg, Vw_grids, Ex_bg, Ey_bg, Ez_bg, Ewx, Ewy, Ewz, ant_masks, sc_mask = load_and_interpolate_vtk(sim_params)

    # ---------------------------------------------------------
    # WORKFLOW PŘEPÍNAČ
    # ---------------------------------------------------------
    # Pro pouhé přehrání již spočítaných dat změňte na False
    PROVEST_VYPOCET = True

    if PROVEST_VYPOCET:
        print("\n=== KROK 2: Spouštím 3D fyzikální PIC simulaci ===")
        # NOVÉ: Předání masky sondy do simulačního jádra
        sim = core.DustImpactSimulation3D(
            sim_params, sim_toggles,
            V_bg, Vw_grids, Ex_bg, Ey_bg, Ez_bg, Ewx, Ewy, Ewz, ant_masks, sc_mask
        )
        sim_results = sim.run()

        print("\n=== KROK 3: Ukládání fyzikálních dat na disk ===")
        save_results_npz(sim_results, plot_config.file_results_npz)
    else:
        print("\n=== KROK 2 & 3: PŘESKOČEN (Fyzikální simulace vypnuta) ===")

    # ---------------------------------------------------------
    # VIZUALIZACE A POST-PROCESSING
    # ---------------------------------------------------------
    print("\n=== KROK 4: Načítání a Vizualizace ===")
    try:
        loaded_results = load_results_npz(plot_config.file_results_npz)
        plt.plot_simulation_results_3d(loaded_results, sim_params, plot_config)
    except FileNotFoundError:
        print(
            f"[CHYBA] Soubor {plot_config.file_results_npz} nebyl nalezen. Nastavte PROVEST_VYPOCET = True pro jeho vytvoření.")