# -*- coding: utf-8 -*-
"""
Electric field E = -grad(V) on the PIC grid with conductor-aware differences at conductor surfaces.

np.gradient uses central differences everywhere. On a conductor node next to free space the central
difference mixes the surface value with the (equal) value inside the conductor and returns only about
half of the surface field; zeroing the field on conductor nodes is worse. Particles in the first cell
next to a conductor then feel a potential drop that is too small (by ~8 % of V_sc for a sphere with
R/dx = 5), which biases the escape and recollection of impact-plasma charges. This module replaces the
derivative on conductor surface nodes by a one-sided difference towards the free neighbour, so that
trilinear interpolation of E reproduces the potential drop across the first cell.
"""

from typing import Sequence, Tuple

import numpy as np


class ConductorGradient3D:
    """
    Precomputes the conductor-surface stencils for a fixed conductor mask and evaluates E = -grad(V).

    Per axis, a conductor node uses:
      - a forward one-sided difference if only its +1 neighbour is free,
      - a backward one-sided difference if only its -1 neighbour is free,
      - the central difference if both neighbours are free (conductor one node thick),
      - zero if no neighbour along the axis is free (conductor interior).
    Free nodes keep the central difference of np.gradient (one-sided on the outer domain faces).

    Parameters
    ----------
    conductor_mask : np.ndarray
        Boolean array (Nx, Ny, Nz), True on all conductor nodes (spacecraft and antennas).
    spacing : Sequence[float]
        Grid spacings (dx, dy, dz) in metres.
    """

    def __init__(self, conductor_mask: np.ndarray, spacing: Sequence[float]):
        mask = np.asarray(conductor_mask, dtype=bool)
        self.shape = mask.shape
        self.spacing = tuple(float(h) for h in spacing)
        strides = (self.shape[1] * self.shape[2], self.shape[2], 1)
        self._stencils = []
        for axis in range(3):
            free_plus = np.zeros_like(mask)
            free_minus = np.zeros_like(mask)
            lo = [slice(None)] * 3
            hi = [slice(None)] * 3
            lo[axis], hi[axis] = slice(0, -1), slice(1, None)
            free_plus[tuple(lo)] = ~mask[tuple(hi)]      # neighbour at +1 exists and is free
            free_minus[tuple(hi)] = ~mask[tuple(lo)]     # neighbour at -1 exists and is free
            forward = np.flatnonzero((mask & free_plus & ~free_minus).ravel())
            backward = np.flatnonzero((mask & free_minus & ~free_plus).ravel())
            interior = np.flatnonzero((mask & ~free_plus & ~free_minus).ravel())
            self._stencils.append((forward, backward, interior, strides[axis]))

    def electric_field(self, V: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns (Ex, Ey, Ez) = -grad(V) with conductor-surface one-sided differences.

        Parameters
        ----------
        V : np.ndarray
            Potential on the grid, shape (Nx, Ny, Nz), in volts.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            Electric field components in V/m.
        """
        E = list(np.gradient(-V, *self.spacing))
        v = np.ascontiguousarray(V).ravel()
        for axis, (forward, backward, interior, stride) in enumerate(self._stencils):
            Ea = E[axis].reshape(-1)
            h = self.spacing[axis]
            Ea[forward] = -(v[forward + stride] - v[forward]) / h
            Ea[backward] = -(v[backward] - v[backward - stride]) / h
            Ea[interior] = 0.0
        return E[0], E[1], E[2]
