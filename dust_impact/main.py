# -*- coding: utf-8 -*-
"""
Main entry point for DustImpact PIC simulator.
Reads JSON configuration and executes the 3D kinetic PIC solver runner.
"""

import sys
import os
import argparse
import time
from datetime import datetime

# Vynucení neinteraktivního Agg backendu pro matplotlib před importem modulů s pyplot,
# pokud je explicitně zadán headless režim nebo na serveru bez displaye (X11/Wayland).
if "--no-visualize" in sys.argv or "--plot-only" in sys.argv or (
    sys.platform != "win32" and not os.environ.get("DISPLAY") and "--visualize" not in sys.argv
):
    import matplotlib
    matplotlib.use("Agg")

from dust_impact.sim3d.runner import run_3d_simulation


def main(
    config_file: str = "config.json",
    output_dir: str = None,
    plot_only: bool = False,
    visualize_results: bool = None
):
    """
    Simulation launcher for DustImpact.

    Parameters:
    -----------
    config_file : str
        Path to JSON configuration file.
    output_dir : str, optional
        Target directory to redirect all simulation outputs (.npz, .csv, plots, animations).
    plot_only : bool
        If True, skip physical simulation and only load and visualize existing results.
    visualize_results : bool, optional
        Override whether to render plots/visualizations.
    """
    start_dt = datetime.now()
    t_start = time.perf_counter()

    if visualize_results is False or (visualize_results is None and sys.platform != "win32" and not os.environ.get("DISPLAY")):
        import matplotlib
        matplotlib.use("Agg")

    if not os.path.exists(config_file):
        alt_config = os.path.join("inputs", "config.json")
        if os.path.exists(alt_config):
            config_file = alt_config

    print("=====================================================")
    print(f"   DUSTIMPACT 3D PIC SIMULATOR")
    print(f"   Konfigurační soubor: {config_file}")
    if output_dir:
        print(f"   Výstupní adresář: {os.path.abspath(output_dir)}")
    if plot_only:
        print(f"   Režim běhu: POUZE VIZUALIZACE (--plot-only)")
    print(f"   Čas zahájení: {start_dt.strftime('%Y-%m-%d %H:%M:%S')}")
    if visualize_results is not None:
        print(f"   Interaktivní okna grafů: {'ZAPNUTA' if visualize_results else 'VYPNUTA (headless)'}")
    print("=====================================================\n")

    run_3d_simulation(
        config_file=config_file,
        output_dir=output_dir,
        plot_only=plot_only,
        visualize_results=visualize_results
    )

    end_dt = datetime.now()
    elapsed_sec = time.perf_counter() - t_start
    print("\n=====================================================")
    print(f"   Čas dokončení: {end_dt.strftime('%Y-%m-%d %H:%M:%S')}")
    if elapsed_sec >= 60.0:
        print(f"   Celková doba trvání výpočtu: {elapsed_sec:.2f} s ({elapsed_sec / 60.0:.2f} min)")
    else:
        print(f"   Celková doba trvání výpočtu: {elapsed_sec:.2f} s")
    print("=====================================================")


def cli_entrypoint():
    """Main CLI entrypoint for terminal execution and console script."""
    parser = argparse.ArgumentParser(description="DustImpact 3D PIC Simulator")
    parser.add_argument("--config", "-c", type=str, default="config.json", help="Cesta ke konfiguračnímu souboru JSON")
    parser.add_argument("--output-dir", "-o", type=str, default=None, help="Cílová složka pro uložení všech výstupů (.npz, .csv, grafy)")
    parser.add_argument("--plot-only", action="store_true", default=False, help="Přeskočit fyzikální simulaci a pouze vygenerovat grafy z existujících dat")
    parser.add_argument("--visualize", dest="visualize_results", action="store_true", default=None, help="Zapnout interaktivní okna grafů")
    parser.add_argument("--no-visualize", dest="visualize_results", action="store_false", help="Vypnout interaktivní okna grafů (headless režim pro servery)")
    args = parser.parse_args()

    main(
        config_file=args.config,
        output_dir=args.output_dir,
        plot_only=args.plot_only,
        visualize_results=args.visualize_results
    )


if __name__ == "__main__":
    cli_entrypoint()
