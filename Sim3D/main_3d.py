#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backward compatibility wrapper for Sim3D/main_3d.py.
Delegates to dust_impact.sim3d.main.
"""
import sys
from dust_impact.sim3d.main import run_3d_simulation

if __name__ == "__main__":
    config = "config_3d.json"
    if len(sys.argv) > 1:
        config = sys.argv[1]
    run_3d_simulation(config_file=config)