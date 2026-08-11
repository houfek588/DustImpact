#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DustImpact - Hlavní vstupní skript pro spouštění simulací.
"""

import sys
import argparse
from dust_impact.main import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DustImpact Unified PIC Simulator")
    parser.add_argument("--config", type=str, default="config.json", help="Cesta ke konfiguračnímu souboru JSON")
    parser.add_argument("--dim", type=int, choices=[2, 3], default=None, help="Vynutit dimenzi výpočtu (2 nebo 3)")
    parser.add_argument("--visualize", dest="visualize_results", action="store_true", default=None, help="Zapnout vizualizaci grafů")
    parser.add_argument("--no-visualize", dest="visualize_results", action="store_false", help="Vypnout vizualizaci grafů")
    args = parser.parse_args()

    main(config_file=args.config, dim_override=args.dim, visualize_results=args.visualize_results)
