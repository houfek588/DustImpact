#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backward compatibility wrapper for Sim2D/main_2d.py.
Delegates to dust_impact.sim2d.main.
"""
from dust_impact.sim2d.main import run_2d_simulation

if __name__ == "__main__":
    run_2d_simulation()