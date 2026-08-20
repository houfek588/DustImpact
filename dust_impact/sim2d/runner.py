#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main runner for 2D PIC Simulation using self-explanatory parameters.
"""

import time
import numpy as np
from dust_impact.physics.constants import e, m_e
from dust_impact.physics.charging import calculate_equilibrium_potential, ENV_EARTH, MAT_ALUMINIUM, MAT_ALUMINIUM_ANTENNE
from dust_impact.common.io import save_results_npz, load_results_npz
from dust_impact.numerics.pushers import check_cfl_condition
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

    # Kontrola numerické stability: CFL podmínka (1.5 * v_th * dt <= dx)
    v_th_e = np.sqrt(2.0 * e * sim_params.T_dust_eV / m_e)
    v_cfl = 1.5 * v_th_e
    dx_min = min(sim_params.dx, sim_params.dy)
    is_stable, cfl_ratio, dt_max_rec = check_cfl_condition(sim_params.dt, dx_min, v_cfl)
    if is_stable:
        print(f"Kontrola CFL podmínky (1.5*v_th * dt <= dx): [OK] Splněna (CFL poměr: {cfl_ratio:.4f} <= 1.0, posun: {v_cfl * sim_params.dt * 1e3:.2f} mm / buňka: {dx_min * 1e3:.2f} mm)")
    else:
        print(f"[NUMERICKÉ VAROVÁNÍ] Nesplněna CFL podmínka (1.5*v_th * dt > dx)! Poměr: {cfl_ratio:.2f} > 1.0\n"
              f"  -> Rychlé elektrony (1.5*v_th) urazí za krok {v_cfl * sim_params.dt * 1e3:.2f} mm, což přesahuje velikost buňky {dx_min * 1e3:.2f} mm.\n"
              f"  -> Doporučený maximální časový krok: dt <= {dt_max_rec:.2e} s")

    PROVEST_VYPOCET = run_calc and getattr(plot_config, 'run_physical_simulation', True)

    if PROVEST_VYPOCET:
        print("\n=== KROK 2: Spouštím fyzikální 2D PIC simulaci ===")
        t_pic_start = time.perf_counter()
        sim = DustImpactSimulation2D(sim_params, sim_toggles)
        sim_results = sim.run()
        t_pic_elapsed = time.perf_counter() - t_pic_start
        print(f"  [OK] 2D PIC výpočet dokončen za {t_pic_elapsed:.2f} s")

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
