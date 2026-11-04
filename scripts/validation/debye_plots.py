# -*- coding: utf-8 -*-
"""
Debye-sphere plots: potential along the x axis (y = z = 0) together with its spherical-shell average
and the Debye-Hueckel profile, as a static plot of the last frame (t = T) and as an animation of all
saved frames.

Usage (from the repository root):
    python scripts/validation/debye_plots.py --config inputs/config_analytical_debye2.json \
        --results outputs/out_analytical_debye2.h5 --out-prefix outputs/debye2

writes <prefix>_potencial_osa_x_slupky_T.png and <prefix>_potencial_osa_x_slupky_animace.gif.
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

sys.path.insert(0, os.path.dirname(__file__))
from _common import load_sphere_run, shell_bins, shell_average  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', required=True, help='Configuration JSON of the run')
    ap.add_argument('--results', required=True, help='Results file of the run (HDF5 or NPZ)')
    ap.add_argument('--out-prefix', required=True, help='Output path prefix (put it under outputs/)')
    ap.add_argument('--fps', type=int, default=6, help='Animation frames per second (default 6)')
    args = ap.parse_args()

    run = load_sphere_run(args.config, args.results)
    p = run.params
    V0, R, lam = float(p.V_sc_bias), run.radius, p.debye_length
    x = p.x_grid
    jy = int(np.argmin(np.abs(p.y_grid - run.center[1])))
    kz = int(np.argmin(np.abs(p.z_grid - run.center[2])))

    # Shell average of every frame, shown at x = centre +- r
    idx, r_shell, counts = shell_bins(run, x[-1] - run.center[0])
    r_plot = np.concatenate([[R], r_shell])
    x_shell = np.concatenate([-r_plot[::-1], r_plot]) + run.center[0]
    V_axis = run.V[:, :, jy, kz]
    V_shell = np.empty((len(run.t), len(x_shell)))
    for k, V in enumerate(run.V):
        vs = np.concatenate([[V0], shell_average(V, idx, counts)])
        V_shell[k] = np.concatenate([vs[::-1], vs])

    r_x = np.abs(x - run.center[0])
    V_dh = np.where(r_x <= R, V0, V0 * R / np.maximum(r_x, 1e-9) * np.exp(-(r_x - R) / lam))
    sc_x = x[r_x <= R]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.axhline(0.0, color='gray', linestyle=':', linewidth=0.8, alpha=0.7)
    ax.axvspan(sc_x.min(), sc_x.max(), color='#d62728', alpha=0.15, label='Sonda')
    line_axis, = ax.plot(x, V_axis[-1], color='#1f77b4', linewidth=1.5, alpha=0.6,
                         label='Simulovaný potenciál $V(x)$ na ose')
    line_shell, = ax.plot(x_shell, V_shell[-1], color='#2ca02c', linewidth=2.2,
                          label='Průměr přes kulové slupky $\\langle V \\rangle(r)$')
    ax.plot(x, V_dh, color='#ff7f0e', linestyle='--', linewidth=1.5, alpha=0.85,
            label=f'Debyeovo stínění ($\\lambda_D = {lam:.2f}\\,\\mathrm{{m}}$)')
    title = ax.set_title('', fontsize=11, weight='bold')
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(min(-0.06 * abs(V0), V_axis.min() - 0.02 * abs(V0)), 1.05 * max(V0, V_axis.max()))
    ax.set_xlabel('Poloha $x$ [m] (osa $y = z = 0$; slupkový průměr v $x = \\pm r$)', fontsize=10)
    ax.set_ylabel('Potenciál $V$ [V]', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper right', fontsize=8, framealpha=0.85)
    fig.suptitle('Elektrický potenciál podél osy X', fontsize=13, weight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    def update(k: int):
        line_axis.set_ydata(V_axis[k])
        line_shell.set_ydata(V_shell[k])
        title.set_text(f'$t = {run.t[k] * 1e6:.2f}\\,\\mu s$  ($t = {run.t[k] / run.t_pe:.1f}\\,T_{{pe}}$)')
        return line_axis, line_shell, title

    update(len(run.t) - 1)
    png = f'{args.out_prefix}_potencial_osa_x_slupky_T.png'
    fig.savefig(png, dpi=150, bbox_inches='tight')
    gif = f'{args.out_prefix}_potencial_osa_x_slupky_animace.gif'
    FuncAnimation(fig, update, frames=len(run.t), blit=False).save(gif, writer=PillowWriter(fps=args.fps), dpi=110)
    print(f'saved: {png}\nsaved: {gif} ({len(run.t)} frames)')


if __name__ == '__main__':
    main()
