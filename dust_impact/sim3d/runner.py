#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main runner for 3D PIC Simulation using self-explanatory parameters.
"""

import os
import argparse
import time
import numpy as np
from dust_impact.physics.constants import e, m_e
from dust_impact.physics.charging import calculate_equilibrium_potential, ENV_EARTH, MAT_ALUMINIUM, MAT_ALUMINIUM_ANTENNE
from dust_impact.common.io import save_results, load_results
from dust_impact.common.vtk_export import export_simulation_to_paraview
from dust_impact.numerics.pushers import check_cfl_condition
from dust_impact.sim3d.config_loader import setup_simulation_parameters_3d
from dust_impact.geometry import build_simulation_geometry, PreparedGeometry3D
from dust_impact.sim3d.sim_core import DustImpactSimulation3D
from dust_impact.sim3d.plotting import plot_simulation_results_3d


def run_3d_simulation(
    config_file: str = "config.json",
    output_dir: str = None,
    plot_only: bool = False,
    visualize_results: bool = None,
    output_format: str = None,
    export_vtk: bool = None,
    enable_checkpointing: bool = None
):
    V_equilibrium = calculate_equilibrium_potential(ENV_EARTH, MAT_ALUMINIUM)
    V_equilibrium_antenne = calculate_equilibrium_potential(ENV_EARTH, MAT_ALUMINIUM_ANTENNE)

    print(f"=== KROK 1: Rovnovážné nabití ===")
    print(f"Plovoucí potenciál sondy: {V_equilibrium:.3f} V")
    print(f"Plovoucí potenciál antény: {V_equilibrium_antenne:.3f} V")

    print(f"Načítám konfiguraci z: {config_file}")
    sim_params, sim_toggles, plot_config = setup_simulation_parameters_3d(
        V_equilibrium, V_equilibrium_antenne, config_file=config_file, output_dir=output_dir
    )

    if visualize_results is not None:
        plot_config.show_interactive_gui_windows = visualize_results
    if output_format is not None:
        plot_config.output_format = output_format.lower()
    if export_vtk is not None:
        plot_config.export_vtk = export_vtk
    if enable_checkpointing is not None:
        plot_config.enable_checkpointing = enable_checkpointing

    primary_output_path = plot_config.primary_output_filepath
    print(f"Zvolený formát ukládání dat: {plot_config.output_format.upper()} ({primary_output_path})")

    print(f"Vypočtená Debyeova délka: {sim_params.debye_length:.3f} m")
    print(f"Fyzická velikost elementu mřížky: dx = {sim_params.dx:.3f} m, dy = {sim_params.dy:.3f} m, dz = {sim_params.dz:.3f} m")

    # Kontrola numerické stability: CFL podmínka (1.5 * v_th * dt <= dx)
    v_th_e = np.sqrt(2.0 * e * sim_params.T_dust_eV / m_e)
    v_cfl = 1.5 * v_th_e
    dx_min = min(sim_params.dx, sim_params.dy, sim_params.dz)
    is_stable, cfl_ratio, dt_max_rec = check_cfl_condition(sim_params.dt, dx_min, v_cfl)
    if is_stable:
        print(f"Kontrola CFL podmínky (1.5*v_th * dt <= dx): [OK] Splněna (CFL poměr: {cfl_ratio:.4f} <= 1.0, posun: {v_cfl * sim_params.dt * 1e3:.2f} mm / buňka: {dx_min * 1e3:.2f} mm)")
    else:
        print(f"[NUMERICKÉ VAROVÁNÍ] Nesplněna CFL podmínka (1.5*v_th * dt > dx)! Poměr: {cfl_ratio:.2f} > 1.0\n"
              f"  -> Rychlé elektrony (1.5*v_th) urazí za krok {v_cfl * sim_params.dt * 1e3:.2f} mm, což přesahuje velikost buňky {dx_min * 1e3:.2f} mm.\n"
              f"  -> Doporučený maximální časový krok: dt <= {dt_max_rec:.2e} s")

    prep_geom = build_simulation_geometry(sim_params)
    print(f"  [OK] Geometrie připravena: {prep_geom.summary()}")

    PROVEST_VYPOCET = (not plot_only) and getattr(plot_config, 'run_physical_simulation', True)

    if PROVEST_VYPOCET:
        print("\n=== KROK 2: Spouštím 3D fyzikální PIC simulaci ===")
        t_pic_start = time.perf_counter()
        sim = DustImpactSimulation3D(
            sim_params, sim_toggles, prep_geom
        )

        cp_file = primary_output_path if plot_config.enable_checkpointing else None
        cp_interval = plot_config.checkpoint_interval_steps if plot_config.enable_checkpointing else 0

        sim_results = sim.run(checkpoint_filepath=cp_file, checkpoint_interval=cp_interval)
        t_pic_elapsed = time.perf_counter() - t_pic_start
        print(f"  [OK] 3D PIC výpočet dokončen za {t_pic_elapsed:.2f} s")

        print("\n=== KROK 3: Ukládání fyzikálních dat na disk ===")
        save_results(
            sim_results, primary_output_path,
            metadata=sim.get_metadata_dict(),
            format_type=plot_config.output_format
        )
    else:
        if plot_only:
            print("\n=== KROK 2 & 3: PŘESKOČEN (Aktivován režim --plot-only) ===")
        else:
            print("\n=== KROK 2 & 3: PŘESKOČEN (Fyzikální simulace vypnuta) ===")

    # Krok 4: Načítání, ParaView VTK export a vizualizace
    should_visualize = getattr(plot_config, 'show_interactive_gui_windows', True) or getattr(plot_config, 'save_plots_to_disk', False)
    should_export_vtk = getattr(plot_config, 'export_vtk', False)

    if should_visualize or should_export_vtk:
        print("\n=== KROK 4: Načítání a Vizualizace ===")
        try:
            loaded_results = load_results(primary_output_path)
        except Exception as err:
            print(f"[CHYBA] Soubor s výsledky nebyl nalezen nebo jej nelze načíst ({primary_output_path}): {err}")
            if plot_only:
                print("  -> V režimu --plot-only musí existovat předchozí vypočtená data.")
            return

        if should_export_vtk:
            try:
                export_simulation_to_paraview(loaded_results, sim_params, output_dir=plot_config.vtk_output_dir)
            except Exception as vtk_err:
                print(f"[VAROVÁNÍ] Export do ParaView VTK selhal: {vtk_err}")

        if should_visualize:
            try:
                plot_simulation_results_3d(
                    loaded_results, sim_params, plot_config,
                    V_bg=V_bg, Vw_grids=Vw_grids,
                    spacecraft_mask=sc_mask, antenna_masks=ant_masks
                )
            except Exception as err:
                print(f"[CHYBA] Selhalo vykreslení výsledků ze souboru {primary_output_path}: {err}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D PIC Simulátor dopadu prachu.")
    parser.add_argument("--config", "-c", type=str, default="config.json", help="Cesta ke konfiguračnímu JSON souboru")
    parser.add_argument("--output-dir", "-o", type=str, default=None, help="Cílová složka pro uložení všech výstupů")
    parser.add_argument("--plot-only", action="store_true", default=False, help="Přeskočit PIC výpočet a pouze vygenerovat grafy z existujících dat")
    parser.add_argument("--format", type=str, choices=["h5", "npz"], default=None, help="Formát uložení výsledků simulace (výchozí: h5)")
    parser.add_argument("--export-vtk", action="store_true", default=None, help="Vygenerovat 3D ParaView data (.vti, .vtp, .pvd)")
    parser.add_argument("--no-checkpoint", action="store_true", default=False, help="Vypnout průběžné ukládání kontrolních bodů (checkpointing)")
    args = parser.parse_args()

    run_3d_simulation(
        config_file=args.config,
        output_dir=args.output_dir,
        plot_only=args.plot_only,
        output_format=args.format,
        export_vtk=args.export_vtk,
        enable_checkpointing=False if args.no_checkpoint else None
    )

