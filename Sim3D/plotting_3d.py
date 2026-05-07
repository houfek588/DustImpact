#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, Any

from input_data_3d import SimulationParams3D, PlottingConfig3D


def _get_v_mag_3d(hist: Dict[str, Any], species: str, i: int) -> np.ndarray:
    vx = hist[f'vx_{species}'][i]
    vy = hist[f'vy_{species}'][i]
    vz = hist[f'vz_{species}'][i]
    mask = ~np.isnan(vx)
    return np.sqrt(vx[mask] ** 2 + vy[mask] ** 2 + vz[mask] ** 2)


def _get_v_signed(hist: Dict[str, Any], species: str, i: int) -> np.ndarray:
    v_x_curr = hist[f'vx_{species}'][i]
    return v_x_curr[~np.isnan(v_x_curr)]


def _plot_currents_and_voltage(results: Dict, params: SimulationParams3D, plot_cfg: PlottingConfig3D):
    if not plot_cfg.show_currents: return
    fig1, (ax1a, ax1b) = plt.subplots(2, 1, figsize=(10, 8))
    fig1.canvas.manager.set_window_title('3D Signály: Proudy a Napětí')
    colors = ['blue', 'red', 'green', 'orange', 'purple']

    for a_idx in range(len(params.C_ant)):
        c = colors[a_idx % len(colors)]
        ax1a.plot(params.time_array * 1e6, results['smooth_total'][a_idx] * 1e9, color=c, lw=2,
                  label=f'Celkový (Anténa {a_idx + 1})')
        ax1a.plot(params.time_array * 1e6, results['smooth_induced'][a_idx] * 1e9, color=c, ls=':', alpha=0.5)
        tau_us = params.R_ant[a_idx] * params.C_ant[a_idx] * 1e6
        ax1b.plot(params.time_array * 1e6, results['voltage_ant'][a_idx] * 1e3, color=c, lw=2.5,
                  label=f'Anténa {a_idx + 1} ($\\tau$={tau_us:.1f} µs)')

    for ax in (ax1a, ax1b):
        ax.axvline(x=params.t_delay * 1e6, color='grey', ls='--', alpha=0.7)
        ax.grid(True, ls=':')
        ax.legend()

    ax1a.set_ylabel("Proud [nA]")
    ax1b.set_ylabel("Napětí [mV]")
    ax1b.set_xlabel("Čas [µs]")
    fig1.tight_layout()
    if plot_cfg.save_plots: fig1.savefig(plot_cfg.file_currents, dpi=300, bbox_inches='tight')


def _animate_3d_fields_slice(hist: Dict, params: SimulationParams3D, plot_cfg: PlottingConfig3D, anims: list):
    if not plot_cfg.show_fields_slice: return

    fig2, ax_V = plt.subplots(figsize=(8, 6))
    fig2.canvas.manager.set_window_title('3D Pole: 2D Řez uprostřed (z = 0)')

    mid_z = params.Nz // 2
    V_max = np.max([np.max(V[:, :, mid_z]) for V in hist['V']])
    V_min = np.min([np.min(V[:, :, mid_z]) for V in hist['V']])
    if V_max == V_min: V_max += 1.0; V_min -= 1.0

    X, Y = np.meshgrid(params.x_grid, params.y_grid, indexing='ij')
    pcm = ax_V.pcolormesh(X, Y, hist['V'][0][:, :, mid_z], shading='gouraud', cmap='viridis', vmin=V_min, vmax=V_max)
    fig2.colorbar(pcm, ax=ax_V, label='Potenciál [V]')

    try:
        V_bg_slice = hist['V'][0][:, :, mid_z]

        # Detekce hranice sondy pomocí velikosti změny (gradientu) potenciálu
        grad_x, grad_y = np.gradient(V_bg_slice)
        grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)

        # Vykreslení kontury přesně v místě nejprudšího skoku (mezi V_max a 0)
        # 40 % maximálního gradientu bezpečně odfiltruje pozvolné Debyeovo stínění plazmatu
        grad_threshold = grad_mag.max() * 0.4
        ax_V.contour(X, Y, grad_mag, levels=[grad_threshold], colors='white', linewidths=2, linestyles='dashed')
    except Exception:
        pass

    ax_V.set_xlabel("x [m]")
    ax_V.set_ylabel("y [m]")

    time_text = ax_V.text(0.02, 0.90, '', transform=ax_V.transAxes, color='white', weight='bold')

    def update(i):
        pcm.set_array(hist['V'][i][:, :, mid_z].ravel())
        time_text.set_text(f"Čas: {hist['t'][i] * 1e6:.2f} µs")
        return pcm, time_text

    anim = animation.FuncAnimation(fig2, update, frames=len(hist['t']), interval=100, blit=False)
    anims.append(anim)
    if plot_cfg.save_plots: anim.save(plot_cfg.file_fields_anim, writer='pillow', fps=15)


def _animate_3d_particles(hist: Dict, params: SimulationParams3D, plot_cfg: PlottingConfig3D, anims: list):
    if not plot_cfg.show_particles_3d: return

    fig3 = plt.figure(figsize=(10, 8))
    ax = fig3.add_subplot(111, projection='3d')
    fig3.canvas.manager.set_window_title('3D Kinetika Částic a Geometrie')

    V_bg = hist['V'][0]

    # Výpočet 3D gradientu (změny) potenciálu
    grad_x, grad_y, grad_z = np.gradient(V_bg)
    grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2 + grad_z ** 2)

    # Maska sondy nyní hledá místa s extrémně prudkou změnou potenciálu.
    # Uvnitř je 0, na povrchu V_max. Gradient zde má absolutní maximum.
    # Vezmeme horních 30 % nejsilnějších změn, což dokonale obkreslí tenkou skořápku kovu.
    sonda_mask = grad_mag > (grad_mag.max() * 0.3)

    X, Y, Z = np.meshgrid(params.x_grid, params.y_grid, params.z_grid, indexing='ij')
    sc_x = X[sonda_mask]
    sc_y = Y[sonda_mask]
    sc_z = Z[sonda_mask]

    if len(sc_x) > 0:
        ax.scatter(sc_x, sc_y, sc_z, color='gray', alpha=0.15, s=15, marker='s', label='Trup sondy')

    scat_e = ax.scatter([], [], [], s=1, color='red', alpha=0.3, label='Elektrony')
    scat_i = ax.scatter([], [], [], s=2, color='blue', alpha=0.3, label='Ionty')

    # Nastavení os na nové parametry -L až L
    ax.set_xlim(-params.L_x, params.L_x)
    ax.set_ylim(-params.L_y, params.L_y)
    ax.set_zlim(-params.L_z, params.L_z)

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.legend(loc='upper right')

    ax.scatter(*params.impact_pos, color='orange', s=100, marker='*', label='Impakt')
    time_text = ax.text2D(0.02, 0.95, '', transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.8))

    def update(i):
        me, mi = ~np.isnan(hist['x_e'][i]), ~np.isnan(hist['x_i'][i])
        scat_e._offsets3d = (hist['x_e'][i][me], hist['y_e'][i][me], hist['z_e'][i][me])
        scat_i._offsets3d = (hist['x_i'][i][mi], hist['y_i'][i][mi], hist['z_i'][i][mi])
        time_text.set_text(f"Čas: {hist['t'][i] * 1e6:.2f} µs")
        return scat_e, scat_i, time_text

    anim = animation.FuncAnimation(fig3, update, frames=len(hist['t']), interval=100, blit=False)
    anims.append(anim)
    if plot_cfg.save_plots: anim.save(plot_cfg.file_particles_anim, writer='pillow', fps=15)


def _animate_velocity_distribution(hist: Dict[str, Any], plot_cfg: PlottingConfig3D, anims: list) -> None:
    if not plot_cfg.show_velocity_anim: return

    fig4, ax_v = plt.subplots(figsize=(10, 4))
    fig4.canvas.manager.set_window_title('Termodynamika: Celková rychlost 3D')

    line_ve, = ax_v.plot([], [], lw=2, color='red', drawstyle='steps-mid', label='Elektrony')
    line_vi, = ax_v.plot([], [], lw=2, color='blue', drawstyle='steps-mid', label='Ionty')
    ax_v.set_xlabel("Velikost rychlosti |v| [m/s]")
    ax_v.grid(True, ls=':')
    ax_v.legend()

    v_max = 0.0
    for sp in ['e', 'i']:
        for i in range(len(hist['t'])):
            v = _get_v_mag_3d(hist, sp, i)
            if len(v) > 0: v_max = max(v_max, np.max(v))

    pad = v_max * 0.05 if v_max != 0 else 1e3
    ax_v.set_xlim(0, v_max + pad)

    def update(i):
        ve, vi = _get_v_mag_3d(hist, 'e', i), _get_v_mag_3d(hist, 'i', i)
        my = 10
        if len(ve) > 1:
            ce, be = np.histogram(ve, bins=50, range=(0, v_max + pad))
            line_ve.set_data((be[:-1] + be[1:]) / 2, ce)
            my = max(my, ce.max())
        if len(vi) > 1:
            ci, bi = np.histogram(vi, bins=50, range=(0, v_max + pad))
            line_vi.set_data((bi[:-1] + bi[1:]) / 2, ci)
            my = max(my, ci.max())
        ax_v.set_ylim(0, my * 1.1)
        return line_ve, line_vi

    anim = animation.FuncAnimation(fig4, update, frames=len(hist['t']), interval=100, blit=False)
    anims.append(anim)
    if plot_cfg.save_plots: anim.save(plot_cfg.file_velocity_anim, writer='pillow', fps=15)


def _export_csv(results: Dict, params: SimulationParams3D, plot_cfg: PlottingConfig3D):
    if not plot_cfg.export_data_csv: return
    try:
        header = ["Cas_s"]
        cols = [params.time_array]
        for a_idx in range(len(params.C_ant)):
            header.extend([f"I_ind_{a_idx + 1}_A", f"I_col_{a_idx + 1}_A", f"V_{a_idx + 1}_V"])
            cols.extend(
                [results['smooth_induced'][a_idx], results['smooth_collected'][a_idx], results['voltage_ant'][a_idx]])
        np.savetxt(plot_cfg.file_csv, np.column_stack(cols), delimiter=",", header=",".join(header), comments="")
        print(f"  [OK] CSV uloženo: {plot_cfg.file_csv}")
    except Exception as e:
        print(f"  [CHYBA] CSV export selhal: {e}")


def plot_simulation_results_3d(results: Dict, params: SimulationParams3D, plot_cfg: PlottingConfig3D):
    print("=====================================================")
    print("Generuji 3D Vizualizace a připravuji výstupy...")
    print("=====================================================")

    anims = []
    _plot_currents_and_voltage(results, params, plot_cfg)
    _animate_3d_fields_slice(results['history'], params, plot_cfg, anims)
    _animate_3d_particles(results['history'], params, plot_cfg, anims)
    _animate_velocity_distribution(results['history'], plot_cfg, anims)
    _export_csv(results, params, plot_cfg)

    print("=====================================================")
    if anims: print(f"Zobrazuji interaktivní okna ({len(anims)}).")
    plt.show()