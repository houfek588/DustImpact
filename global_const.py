# -*- coding: utf-8 -*-
"""
Backward compatibility wrapper for global_const.py.
Delegates to dust_impact.constants.
"""
from dust_impact.constants import e, m_e, eps_0, amu, n_sw, Te_eV

__all__ = ["e", "m_e", "eps_0", "amu", "n_sw", "Te_eV"]