# -*- coding: utf-8 -*-
"""
Debye-sphere validation: shell- and time-averaged potential around a charged sphere compared with
  - the thermal-equilibrium (Boltzmann) potential of the same closed, grounded box on the same grid
    (frozen uniform ions, or Boltzmann ions when they respond during the run), and
  - the open-space Debye-Hueckel profile V0 R/r exp(-(r - R)/lambda).

Usage (from the repository root):
    python scripts/validation/debye_profile.py --config inputs/config_analytical_debye2.json \
        --results outputs/out_analytical_debye2.h5 [--plot outputs/debye2_profil.png]

The closed-box reference uses the PIC normalisation: the box holds n0 * V_box particles of each species
outside the sphere, so electron number conservation produces a positive far-field offset that the open
Debye-Hueckel profile does not have.
"""

import argparse
import os
import sys

import numpy as np
import scipy.sparse as sp
import pyamg

sys.path.insert(0, os.path.dirname(__file__))
from _common import load_sphere_run, shell_bins, shell_average  # noqa: E402
from dust_impact.physics.constants import e, eps_0  # noqa: E402
from dust_impact.numerics.poisson import assemble_negative_laplacian_3d  # noqa: E402


def node_volume_weights(run) -> np.ndarray:
    """Volume represented by every node (halved on faces, zero inside the sphere) [m^3]."""
    p = run.params
    w = np.full((p.Nx, p.Ny, p.Nz), p.dx * p.dy * p.dz)
    w[[0, -1]] *= 0.5
    w[:, [0, -1]] *= 0.5
    w[:, :, [0, -1]] *= 0.5
    w[run.geometry.spacecraft_mask_3d] = 0.0
    return w


def closed_box_equilibrium(run, V0: float, boltzmann_ions: bool, max_iter: int = 60) -> np.ndarray:
    """
    Boltzmann equilibrium of the closed box: -Lap phi = e/eps0 (n_i - n_e), n_e = C_e exp(phi / T_e),
    n_i uniform (frozen) or C_i exp(-phi / T_i), with C_e, C_i fixed by particle number conservation;
    phi = V0 on the sphere and 0 on the grounded box faces. Solved by Newton iterations with AMG-CG.
    """
    p = run.params
    shape = (p.Nx, p.Ny, p.Nz)
    sc = run.geometry.spacecraft_mask_3d
    Te, Ti = p.solar_wind_electron_temp_eV, p.solar_wind_ion_temp_eV
    w = node_volume_weights(run)
    N_total = p.solar_wind_density_m3 * (2 * p.L_x) * (2 * p.L_y) * (2 * p.L_z)

    fixed = sc.copy()
    fixed[[0, -1]] = fixed[:, [0, -1]] = fixed[:, :, [0, -1]] = True
    A = assemble_negative_laplacian_3d(*shape, p.dx, p.dy, p.dz).tocsr()
    fr, fx = np.flatnonzero(~fixed.ravel()), np.flatnonzero(fixed.ravel())
    A_ff = A[fr][:, fr].tocsr()
    b_d = -(A[fr][:, fx] @ np.where(sc, V0, 0.0).ravel()[fx])

    phi = np.where(sc, V0, 0.0).ravel()
    n_uniform = N_total / np.sum(w)
    for _ in range(max_iter):
        p3 = phi.reshape(shape)
        ex_e = np.exp(p3 / Te)
        n_e = N_total / np.sum(w * ex_e) * ex_e
        if boltzmann_ions:
            ex_i = np.exp(-p3 / Ti)
            n_i = N_total / np.sum(w * ex_i) * ex_i
            dn_dphi = n_e / Te + n_i / Ti
        else:
            n_i = np.full(shape, n_uniform)
            dn_dphi = n_e / Te
        F = A_ff @ phi[fr] - b_d - (e / eps_0) * (n_i - n_e).ravel()[fr]
        J = (A_ff + sp.diags((e / eps_0) * dn_dphi.ravel()[fr])).tocsr()
        d = pyamg.ruge_stuben_solver(J).solve(-F, tol=1e-10, accel='cg', maxiter=200)
        phi[fr] += d
        if np.abs(d).max() < 1e-8:
            break
    return phi.reshape(shape)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', required=True, help='Configuration JSON of the run')
    ap.add_argument('--results', required=True, help='Results file of the run (HDF5 or NPZ)')
    ap.add_argument('--t-start', type=float, default=5.0,
                    help='Frames after this many electron plasma periods are averaged (default 5)')
    ap.add_argument('--ions', choices=['auto', 'frozen', 'boltzmann'], default='auto',
                    help='Ion response of the reference (auto: Boltzmann if the run lasts > 2 ion plasma periods)')
    ap.add_argument('--r-max', type=float, default=None, help='Largest shell radius [m]')
    ap.add_argument('--plot', default=None, help='Optional PNG file with the profile plot (put it under outputs/)')
    args = ap.parse_args()

    run = load_sphere_run(args.config, args.results)
    p = run.params
    V0, R = float(p.V_sc_bias), run.radius
    t_pi = 2.0 * np.pi / p.omega_pi
    boltzmann_ions = args.ions == 'boltzmann' or (args.ions == 'auto' and run.t[-1] > 2.0 * t_pi)
    lam_e = p.debye_length
    lam_i = np.sqrt(eps_0 * p.solar_wind_ion_temp_eV / (p.solar_wind_density_m3 * e))
    lam = 1.0 / np.sqrt(1.0 / lam_e**2 + 1.0 / lam_i**2) if boltzmann_ions else lam_e
    r_max = args.r_max or min(0.65 * min(p.L_x, p.L_y, p.L_z), R + 5.0 * lam)

    late = run.t > args.t_start * run.t_pe
    if not np.any(late):
        raise SystemExit(f"No frame after {args.t_start} T_pe (run ends at {run.t[-1] / run.t_pe:.1f} T_pe).")
    V_avg = run.V[late].mean(axis=0)
    idx, r_shell, counts = shell_bins(run, r_max)
    sim = shell_average(V_avg, idx, counts)
    frame_shells = np.array([shell_average(V, idx, counts) for V in run.V[late]])

    print(f"Run: {args.results}")
    print(f"  {late.sum()} of {len(run.t)} frames averaged (t > {args.t_start} T_pe, t_end = {run.t[-1] / run.t_pe:.1f} T_pe)")
    print(f"  ions in the reference: {'Boltzmann' if boltzmann_ions else 'frozen (uniform)'}, "
          f"lambda_De = {lam_e:.3f} m, lambda used for DH = {lam:.3f} m")
    print("  solving the closed-box equilibrium ...", flush=True)
    ref = closed_box_equilibrium(run, V0, boltzmann_ions)
    ref_shell = shell_average(ref, idx, counts)
    dh = V0 * R / r_shell * np.exp(-(r_shell - R) / lam)

    print(f"\n {'r [m]':>6} {'sim':>8} {'box eq.':>8} {'DH open':>8} {'vacuum':>8} {'noise':>7}")
    for k in range(len(r_shell)):
        print(f" {r_shell[k]:6.2f} {sim[k]:8.4f} {ref_shell[k]:8.4f} {dh[k]:8.4f} "
              f"{V0 * R / r_shell[k]:8.4f} {frame_shells[:, k].std():7.4f}")
    free = ~run.geometry.spacecraft_mask_3d & (run.r < r_max)
    diff = V_avg[free] - ref[free]
    print(f"\n rms(sim - box equilibrium) = {np.sqrt(np.mean(diff**2)) * 1e3:.1f} mV, "
          f"max = {np.abs(diff).max() * 1e3:.1f} mV (r < {r_max:.2f} m)")
    print(f" node noise of single frames: {(run.V[late] - V_avg)[:, free].std() * 1e3:.1f} mV")
    k_probe = int(np.argmin(np.abs(r_shell - (R + 0.5 * lam))))
    evo = [shell_average(V, idx, counts)[k_probe] for V in run.V]
    print(f" V(r = {r_shell[k_probe]:.2f} m) vs time [T_pe]: " +
          ' '.join(f"{ti / run.t_pe:.1f}:{v:.3f}" for ti, v in list(zip(run.t, evo))[::5]))

    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(r_shell, sim, 'o-', color='#2ca02c', ms=3, label='Simulace (průměr přes slupky a čas)')
        ax.plot(r_shell, ref_shell, '--', color='#1f77b4', label='Rovnováha v uzavřeném boxu (stejná mřížka)')
        ax.plot(r_shell, dh, ':', color='#ff7f0e', lw=2, label=f'Debye-Hückel ($\\lambda$ = {lam:.2f} m)')
        ax.set_xlabel('r [m]')
        ax.set_ylabel('V [V]')
        ax.grid(True, ls=':')
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.plot, dpi=150)
        print(f" plot saved: {args.plot}")


if __name__ == '__main__':
    main()
