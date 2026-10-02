# -*- coding: utf-8 -*-
"""Backward compatibility re-export module."""
from dust_impact.geometry.voxelizer import (
    detect_metal_mask_3d,
    voxelize_sphere,
    voxelize_box,
    voxelize_cylinder,
)

__all__ = [
    "detect_metal_mask_3d",
    "voxelize_sphere",
    "voxelize_box",
    "voxelize_cylinder",
]
