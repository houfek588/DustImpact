#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os

# Přidání kořenového adresáře do cesty pro import 'charging.py'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config_loader_3d import setup_simulation_parameters_3d
from vtk_reader_3d import load_and_interpolate_vtk
from io_utils_3d import save_results_npz, load_results_npz
import sim_core_3d as core
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D PIC Simulátor dopadu prachu.")
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config_3d.json",
        help="Cesta ke konfiguračnímu JSON souboru (výchozí: config_3d.json)"
    )
    args = parser.parse_args()

    from charging import calculate_equilibrium_potential, ENV_EARTH, MAT_ALUMINIUM, MAT_ALUMINIUM_ANTENNE
    
    V_equilibrium = calculate_equilibrium_potential(ENV_EARTH, MAT_ALUMINIUM)
    V_equilibrium_antenne = calculate_equilibrium_potential(ENV_EARTH, MAT_ALUMINIUM_ANTENNE)

    print(f"=== KROK 1: Rovnovážné nabití ===")
    print(f"Plovoucí potenciál sondy: {V_equilibrium:.3f} V")
    print(f"Plovoucí potenciál antény: {V_equilibrium_antenne:.3f} V")

    # 1. Konfigurace a načtení VTK (SPIS data nebo Syntetický fallback)
    print(f"Načítám konfiguraci z: {args.config}")
    sim_params, sim_toggles, plot_config = setup_simulation_parameters_3d(V_equilibrium, V_equilibrium_antenne, config_file=args.config)
    print(f"Vypočtená Debyeova délka: {sim_params.debye_length:.3f} m")
    print(f"Fyzická velikost elementu mřížky: dx = {sim_params.dx:.3f} m, dy = {sim_params.dy:.3f} m, dz = {sim_params.dz:.3f} m")

    # NOVÉ: Vytažení masky sondy ze čtečky dat
    V_bg, Vw_grids, Ex_bg, Ey_bg, Ez_bg, Ewx, Ewy, Ewz, ant_masks, sc_mask = load_and_interpolate_vtk(sim_params)

    # ---------------------------------------------------------
    # WORKFLOW PŘEPÍNAČ
    # ---------------------------------------------------------
    # Načteno z configu (visualize_results povolí/zakáže Krok 4)
    PROVEST_VYPOCET = plot_config.run_simulation

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
    # VIZUALIZACE A POST-PROCESSING (Krok 4)
    # ---------------------------------------------------------
    if plot_config.visualize_results:
        import plotting_3d as plt

        print("\n=== KROK 4: Načítání a Vizualizace ===")
        try:
            loaded_results = load_results_npz(plot_config.file_results_npz)
            # Předáme i načtená VTK data a masky pro zobrazení
            plt.plot_simulation_results_3d(
                loaded_results, sim_params, plot_config,
                V_bg=V_bg, Vw_grids=Vw_grids,
                spacecraft_mask=sc_mask, antenna_masks=ant_masks
            )
        except FileNotFoundError:
            print(
                f"[CHYBA] Soubor {plot_config.file_results_npz} nebyl nalezen. Nastavte run_simulation = True v configu.")
    else:
        print("\n=== KROK 4: PŘESKOČEN (Vizualizace vypnuta v configu) ===")

    print("\n=== ZPRACOVÁNÍ DOKONČENO ===")