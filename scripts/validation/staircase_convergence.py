# -*- coding: utf-8 -*-
"""
Staircase error of a voxelized sphere: the linear screened potential (-Lap V + V / lambda^2 = 0,
V = V0 on the voxelized sphere, 0 on the box) on grids of increasing resolution compared with the open
Debye-Hueckel profile V0 R/r exp(-(r - R)/lambda). It shows how much of the deviation of a Debye run
from theory is caused only by the grid representation of the sphere (no particles involved).

Usage:
    python scripts/validation/staircase_convergence.py --radius 1.0 --half-length 7.0 \
        --debye-length 1.051 --nodes 71 101 141
"""

import argparse
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp
import pyamg

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from dust_impact.numerics.poisson import assemble_negative_laplacian_3d  # noqa: E402
from dust_impact.geometry.analytical import voxelize_analytical_spacecraft_part  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--radius', type=float, default=1.0, help='Sphere radius R [m]')
    ap.add_argument('--half-length', type=float, default=7.0, help='Box half-length L [m]')
    ap.add_argument('--debye-length', type=float, default=1.051, help='Screening length lambda [m]')
    ap.add_argument('--voltage', type=float, default=0.5, help='Sphere potential V0 [V]')
    ap.add_argument('--nodes', type=int, nargs='+', default=[71, 101, 141], help='Grid nodes per axis')
    args = ap.parse_args()

    R, L, lam, V0 = args.radius, args.half_length, args.debye_length, args.voltage
    sphere = SimpleNamespace(type='sphere', center=[0.0, 0.0, 0.0], radius=R)
    shells = [(1.0, 1.3), (1.3, 1.7), (1.7, 2.5), (2.5, 4.0)]
    print('relative deviation from open DH in shells r/R = ' + ', '.join(f'{a}-{b}' for a, b in shells))
    for N in args.nodes:
        t0 = time.time()
        x = np.linspace(-L, L, N)
        h = x[1] - x[0]
        X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
        r = np.sqrt(X**2 + Y**2 + Z**2)
        sc = voxelize_analytical_spacecraft_part(X.ravel(), Y.ravel(), Z.ravel(), sphere).reshape(N, N, N)
        fixed = sc.copy()
        fixed[[0, -1]] = fixed[:, [0, -1]] = fixed[:, :, [0, -1]] = True
        A = (assemble_negative_laplacian_3d(N, N, N, h, h, h) + sp.identity(N**3) / lam**2).tocsr()
        fr, fx = np.flatnonzero(~fixed.ravel()), np.flatnonzero(fixed.ravel())
        V = np.where(sc, V0, 0.0).ravel()
        V[fr] = pyamg.ruge_stuben_solver(A[fr][:, fr].tocsr()).solve(-(A[fr][:, fx] @ V[fx]), tol=1e-10, accel='cg')
        V = V.reshape(N, N, N)
        dh = V0 * R / np.maximum(r, 1e-9) * np.exp(-(r - R) / lam)
        dev = []
        for a, b in shells:
            m = (r >= a * R) & (r < b * R) & ~sc
            dev.append(f'{100.0 * np.mean(V[m] - dh[m]) / np.mean(dh[m]):+.1f} %')
        print(f'  N = {N:3d}, dx = {h:.3f} m, R/dx = {R / h:5.1f}: ' + '  '.join(dev) + f'  ({time.time() - t0:.0f} s)')


if __name__ == '__main__':
    main()
