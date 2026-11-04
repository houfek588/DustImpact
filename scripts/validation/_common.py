# -*- coding: utf-8 -*-
"""
Shared helpers of the validation scripts: loading a finished run (configuration + results file) and
spherical-shell averaging around an analytical spacecraft sphere.
"""

import contextlib
import io
import os
import sys
import tempfile
from dataclasses import dataclass
from typing import Any, Tuple

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from dust_impact.solver.config_loader import setup_simulation_parameters_3d  # noqa: E402
from dust_impact.geometry import build_simulation_geometry  # noqa: E402
from dust_impact.common.io import load_results  # noqa: E402


@dataclass
class SphereRun:
    """A finished run around an analytical sphere: parameters, geometry and potential history."""
    params: Any
    geometry: Any
    t: np.ndarray            # frame times [s]
    V: np.ndarray            # potential frames (n_frames, Nx, Ny, Nz) [V]
    center: np.ndarray       # sphere centre [m]
    radius: float            # sphere radius [m]
    r: np.ndarray            # distance of every grid node from the centre [m]

    @property
    def t_pe(self) -> float:
        """Electron plasma period 2 pi / omega_pe [s]."""
        return 2.0 * np.pi / self.params.omega_pe


def load_sphere_run(config_file: str, results_file: str) -> SphereRun:
    """
    Loads a run of an analytical-sphere configuration.

    The geometry is rebuilt from the configuration (spacecraft mask, background); the results file
    (HDF5 or NPZ written by main.py) must come from the same configuration.
    """
    with contextlib.redirect_stdout(io.StringIO()):
        params, _, _ = setup_simulation_parameters_3d(
            Vf=0.0, Vf_antenne=0.0, config_file=config_file, output_dir=tempfile.mkdtemp()
        )
        geometry = build_simulation_geometry(params)
        results = load_results(results_file)

    sphere = params.geometry.analytical.spacecraft
    if getattr(sphere, 'type', '') != 'sphere':
        raise ValueError("The validation scripts expect an analytical spherical spacecraft.")
    hist = results['history']
    V = np.asarray(hist['V'], dtype=np.float64)
    if V.shape[1:] != (params.Nx, params.Ny, params.Nz):
        raise ValueError(f"Grid of the results {V.shape[1:]} differs from the configuration "
                         f"{(params.Nx, params.Ny, params.Nz)}.")
    center = np.asarray(sphere.center, dtype=float)
    X, Y, Z = np.meshgrid(params.x_grid - center[0], params.y_grid - center[1], params.z_grid - center[2],
                          indexing='ij')
    return SphereRun(
        params=params, geometry=geometry, t=np.asarray(hist['t'], dtype=float), V=V,
        center=center, radius=float(sphere.radius), r=np.sqrt(X**2 + Y**2 + Z**2),
    )


def shell_bins(run: SphereRun, r_max: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Spherical shells of one cell width starting at the sphere surface.

    Returns
    -------
    (bin_index, r_mean, counts)
        bin_index: shell index of every grid node (-1 outside the shells or inside the sphere),
        r_mean: mean radius of the nodes of every non-empty shell, counts: nodes per shell.
    """
    h = run.params.dx
    edges = np.arange(run.radius, r_max + h, h)
    idx = np.digitize(run.r.ravel(), edges) - 1
    valid = (idx >= 0) & (idx < len(edges) - 1) & ~run.geometry.spacecraft_mask_3d.ravel()
    idx = np.where(valid, idx, -1)
    counts = np.bincount(idx[valid], minlength=len(edges) - 1)
    keep = counts > 0
    r_sum = np.bincount(idx[valid], weights=run.r.ravel()[valid], minlength=len(edges) - 1)
    # Renumber to the non-empty shells only
    new_index = np.full(len(edges) - 1, -1)
    new_index[keep] = np.arange(np.count_nonzero(keep))
    idx = np.where(valid, new_index[np.clip(idx, 0, None)], -1)
    return idx, r_sum[keep] / counts[keep], counts[keep]


def shell_average(field: np.ndarray, bin_index: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Average of a grid field over every shell of shell_bins."""
    valid = bin_index >= 0
    return np.bincount(bin_index[valid], weights=np.asarray(field).ravel()[valid], minlength=len(counts)) / counts
