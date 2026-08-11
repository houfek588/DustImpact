#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main runner for 3D PIC Simulation using self-explanatory parameters.
"""

import argparse
from dust_impact.physics.charging import calculate_equilibrium_potential, ENV_EARTH, MAT_ALUMINIUM, MAT_ALUMINIUM_ANTENNE
from dust_impact.common.io import save_results_npz, load_results_npz
from dust_impact.sim3d.config_loader import setup_simulation_parameters_3d
from dust_impact.sim3d.vtk_reader import load_and_interpolate_vtk
from dust_impact.sim3d.sim_core import DustImpactSimulation3D
from dust_impact.sim3d.plotting import plot_simulation_results_3d


def run_3d_simulation(config_file: str = "config.json", visualize_results: bool = None):
    V_equilibrium = calculate_equilibrium_potential(ENV_EARTH, MAT_ALUMINIUM)
    V_equilibrium_antenne = calculate_equilibrium_potential(ENV_EARTH, MAT_ALUMINIUM_ANTENNE)

    print(f"=== KROK 1: Rovnovážné nabití ===")
    print(f"Plovoucí potenciál sondy: {V_equilibrium:.3f} V")
    print(f"Plovoucí potenciál antény: {V_equilibrium_antenne:.3f} V")

    print(f"Načítám konfiguraci z: {config_file}")
    sim_params, sim_toggles, plot_config = setup_simulation_parameters_3d(V_equilibrium, V_equilibrium_antenne, config_file=config_file)

    if visualize_results is not None:
        plot_config.show_interactive_gui_windows = visualize_results

    print(f"Vypočtená Debyeova délka: {sim_params.debye_length:.3f} m")
    print(f"Fyzická velikost elementu mřížky: dx = {sim_params.dx:.3f} m, dy = {sim_params.dy:.3f} m, dz = {sim_params.dz:.3f} m")

    V_bg, Vw_grids, Ex_bg, Ey_bg, Ez_bg, Ewx, Ewy, Ewz, ant_masks, sc_mask = load_and_interpolate_vtk(sim_params)

    PROVEST_VYPOCET = getattr(plot_config, 'run_physical_simulation', True)

    if PROVEST_VYPOCET:
        print("\n=== KROK 2: Spouštím 3D fyzikální PIC simulaci ===")
        sim = DustImpactSimulation3D(
            sim_params, sim_toggles,
            V_bg, Vw_grids, Ex_bg, Ey_bg, Ez_bg, Ewx, Ewy, Ewz, ant_masks, sc_mask
        )
        sim_results = sim.run()

        print("\n=== KROK 3: Ukládání fyzikálních dat na disk ===")
        save_results_npz(sim_results, plot_config.output_npz_filepath)
    else:
        print("\n=== KROK 2 & 3: PŘESKOČEN (Fyzikální simulace vypnuta) ===")

    if getattr(plot_config, 'show_interactive_gui_windows', True) or getattr(plot_config, 'save_plots_to_disk', False):
        print("\n=== KROK 4: Načítání a Vizualizace ===")
        try:
            loaded_results = load_results_npz(plot_config.output_npz_filepath)
            plot_simulation_results_3d(
                loaded_results, sim_params, plot_config,
                V_bg=V_bg, Vw_grids=Vw_grids,
                spacecraft_mask=sc_mask, antenna_masks=ant_masks
            )
        except FileNotFoundError:
            print(f"[CHYBA] Soubor {plot_config.output_npz_filepath} nebyl nalezen.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D PIC Simulátor dopadu prachu.")
    parser.add_argument("--config", "-c", type=str, default="config.json", help="Cesta ke konfiguračnímu JSON souboru")
    args = parser.parse_args()

    run_3d_simulation(config_file=args.config)
