# -*- coding: utf-8 -*-
"""
Backward compatibility bridge package for dust_impact.sim3d.
Delegates everything to dust_impact.solver and dust_impact.geometry.
"""

from dust_impact.solver import *
from dust_impact.geometry import build_simulation_geometry as load_and_interpolate_vtk
