# -*- coding: utf-8 -*-
"""
Voxelizer compatibility module.
Delegates to the standalone dust_impact.geometry.voxelizer package.
"""

from dust_impact.geometry.voxelizer import (
    detect_metal_mask_3d,
    voxelize_sphere,
    voxelize_box,
    voxelize_cylinder
)

__all__ = [
    "detect_metal_mask_3d",
    "voxelize_sphere",
    "voxelize_box",
    "voxelize_cylinder"
]
