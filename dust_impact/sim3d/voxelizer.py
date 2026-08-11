# -*- coding: utf-8 -*-
"""
Voxelizer for detecting metallic surfaces and antennas in 3D grid fields.
"""

import numpy as np
import scipy.ndimage as ndimage


def detect_metal_mask_3d(V: np.ndarray, dx_min: float) -> np.ndarray:
    """ Detects metal boundary masks for multi-level potential grids. """
    Ex, Ey, Ez = np.gradient(-V, dx_min, dx_min, dx_min)
    grad_mag = np.sqrt(Ex ** 2 + Ey ** 2 + Ez ** 2)

    max_grad = np.nanmax(grad_mag)
    if max_grad < 1e-3:
        return np.zeros_like(V, dtype=bool)

    jump_threshold = 0.30 * max_grad
    shell = grad_mag > jump_threshold
    mask = ndimage.binary_fill_holes(shell)

    return mask
