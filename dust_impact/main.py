# -*- coding: utf-8 -*-
"""
Main unified entry point for DustImpact PIC simulator.
Reads unified config.json, inspects dimension parameter ('dim': 2 or 3),
and executes either 2D or 3D kinetic PIC solver runner.
"""

import sys
import os
import json
import argparse
from dust_impact.sim2d.runner import run_2d_simulation
from dust_impact.sim3d.runner import run_3d_simulation


import time
from datetime import datetime


def main(config_file: str = "config.json", dim_override: int = None, visualize_results: bool = None):
    """
    Unified simulation launcher for DustImpact.

    Parameters:
    -----------
    config_file : str
        Path to unified JSON configuration file.
    dim_override : int, optional
        Override dimension from CLI (--dim 2 or --dim 3).
    visualize_results : bool, optional
        Override whether to render plots/visualizations.
    """
    start_dt = datetime.now()
    t_start = time.perf_counter()

    if not os.path.exists(config_file):
        alt_config = os.path.join("inputs", "config.json")
        if os.path.exists(alt_config):
            config_file = alt_config

    dim = 3  # Default dimension
    if os.path.exists(config_file):
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                dim = data.get("simulation_dimension", 3)
        except Exception as err:
            print(f"[UPOZORNĚNÍ] Nelze přečíst 'dim' z {config_file}: {err}. Používám výchozí dim=3.")

    if dim_override in [2, 3]:
        dim = dim_override

    print("=====================================================")
    print(f"   DUSTIMPACT SIMULATOR (Dimenze výpočtu: {dim}D)")
    print(f"   Konfigurační soubor: {config_file}")
    print(f"   Čas zahájení: {start_dt.strftime('%Y-%m-%d %H:%M:%S')}")
    if visualize_results is not None:
        print(f"   Vizualizace výsledků: {'ZAPNUTA' if visualize_results else 'VYPNUTA'}")
    print("=====================================================\n")

    if dim == 2:
        run_2d_simulation(config_file=config_file, visualize_results=visualize_results)
    elif dim == 3:
        run_3d_simulation(config_file=config_file, visualize_results=visualize_results)
    else:
        raise ValueError(f"Podporované dimenze výpočtu jsou pouze 2 nebo 3 (zadáno: {dim}).")

    end_dt = datetime.now()
    elapsed_sec = time.perf_counter() - t_start
    print("\n=====================================================")
    print(f"   Čas dokončení: {end_dt.strftime('%Y-%m-%d %H:%M:%S')}")
    if elapsed_sec >= 60.0:
        print(f"   Celková doba trvání výpočtu: {elapsed_sec:.2f} s ({elapsed_sec / 60.0:.2f} min)")
    else:
        print(f"   Celková doba trvání výpočtu: {elapsed_sec:.2f} s")
    print("=====================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DustImpact Unified PIC Simulator")
    parser.add_argument("--config", type=str, default="config.json", help="Path to unified JSON config file")
    parser.add_argument("--dim", type=int, choices=[2, 3], default=None, help="Override simulation dimension (2 or 3)")
    parser.add_argument("--visualize", dest="visualize_results", action="store_true", default=None, help="Enable visualization")
    parser.add_argument("--no-visualize", dest="visualize_results", action="store_false", help="Disable visualization")
    args = parser.parse_args()

    main(config_file=args.config, dim_override=args.dim, visualize_results=args.visualize_results)
