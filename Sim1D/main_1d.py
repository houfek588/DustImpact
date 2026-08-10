#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backward compatibility wrapper for Sim1D/main_1d.py.
Delegates to dust_impact.sim1d.main.
"""
from dust_impact.sim1d.main import run_1d_simulation

if __name__ == "__main__":
    run_1d_simulation()