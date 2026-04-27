#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MODUL C: Vizualizace 2D Simulace a export do souborů
Obsahuje tepelné mapy (heatmaps) pro 2D makroskopická pole a 2D scatter pro částice.
Nyní plně podporuje volitelné ukládání grafů a animací přes 'plot_toggles'.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import Dict, Any, Optional

from input_data_2d import SimulationParams2D


def _plot_currents_and_voltage(results: Dict[str, Any], params: SimulationParams2D, plot_toggles: Dict[str, bool]):
    if not plot_toggles.get('show_currents', True):
        return

    fig1, (ax1a, ax1b) = plt.subplots(2, 1, figsize=(10, 8))
    fig1.canvas.manager.set_window_title('2D Signály: Proudy a Napětí')

    ax1a.plot(params.time_array * 1e6, results['smooth_induced'] * 1e9, color='blue', lw=1.5,
              label='Indukovaný (Ramo-Shockley)')
    ax1a.plot(params.time_array * 1e6, results['smooth_collected'] * 1e9, color='red', lw=1.5,
              label='Nasbíraný (Dopady)')
    ax1a.plot(params.time_array * 1e6, results['smooth_total'] * 1e9, color='black', lw=2, linestyle=':',
              label='Celkový')

    ax1a.axvline(x=params.t_delay * 1e6, color='grey', linestyle='--', alpha=0.7)
    ax1a.set_title("Proudy tekoucí do 2D antény")
    ax1a.set_ylabel("Proud [nA]")
    ax1a.legend()
    ax1a.grid(True, linestyle=':')

    tau_us = params.R_ant * params.C_ant * 1e6
    ax1b.plot(params.time_array * 1e6, results['voltage_ant'] * 1e3, color='green', lw=2.5,
              label=f'Napětí ($\\tau$ = {tau_us:.1f} µs)')
    ax1b.axvline(x=params.t_delay * 1e6, color='grey', linestyle='--', alpha=0.7)

    ax1b.set_title("Odezva napětí (RC Obvod)")
    ax1b.set_xlabel("Čas [µs]")
    ax1b.set_ylabel("Napětí [mV]")
    ax1b.legend()
    ax1b.grid(True, linestyle=':')
    fig1.tight_layout()

    if plot_toggles.get('save_plots', False):
        fig1.savefig('out_2d_proudy_napeti.png', dpi=300, bbox_inches='tight')
        print("  [OK] Uloženo: out_2d_proudy_napeti.png")


def _animate_2d_fields(hist: Dict[str, Any], params: SimulationParams2D, plot_toggles: Dict[str, bool],
                       animations: list):
    if not plot_toggles.get('show_fields_anim', True):
        return

    fig2, (ax_V, ax_rho) = plt.subplots(2, 1, figsize=(10, 10))
    fig2.canvas.manager.set_window_title('2D Makroskopická Pole')

    # Přednastavení barevných škál pro zafixování os (prevence blikání barev)
    V_max = np.max([np.max(V) for V in hist['V']])
    V_min = np.min([np.min(V) for V in hist['V']])
    if V_max == V_min: V_max += 1.0; V_min -= 1.0

    rho_max = np.max([np.max(np.abs(rho)) for rho in hist['rho']])
    if rho_max == 0: rho_max = 1e-12

    # Používáme pcolormesh pro rychlé zobrazení 2D mřížky
    pcm_V = ax_V.pcolormesh(params.X_mat, params.Y_mat, hist['V'][0], shading='gouraud', cmap='viridis', vmin=V_min,
                            vmax=V_max)
    fig2.colorbar(pcm_V, ax=ax_V, label='Potenciál [V]')
    ax_V.set_title("2D Elektrický Potenciál")
    ax_V.set_ylabel("y [m]")

    pcm_rho = ax_rho.pcolormesh(params.X_mat, params.Y_mat, hist['rho'][0], shading='gouraud', cmap='seismic',
                                vmin=-rho_max, vmax=rho_max)
    fig2.colorbar(pcm_rho, ax=ax_rho, label='Hustota náboje [C/m³]')
    ax_rho.set_title("2D Hustota Prostorového Náboje")
    ax_rho.set_xlabel("x [m]")
    ax_rho.set_ylabel("y [m]")

    # Nakreslení pozice antény a sondy
    circle_V = plt.Circle((params.x_antenna, params.y_antenna), params.r_antenna, color='white', fill=False, ls='--')
    circle_rho = plt.Circle((params.x_antenna, params.y_antenna), params.r_antenna, color='black', fill=False, ls='--')
    ax_V.add_patch(circle_V)
    ax_rho.add_patch(circle_rho)

    ax_V.axvline(x=0, color='white', lw=4, alpha=0.5, label='Povrch sondy')
    ax_rho.axvline(x=0, color='black', lw=4, alpha=0.5, label='Povrch sondy')

    time_text = ax_V.text(0.02, 0.90, '', transform=ax_V.transAxes, color='white', weight='bold')

    def animate_fields(i):
        # Aktualizace dat v tepelné mapě (nutno zploštit pole funkcí ravel)
        pcm_V.set_array(hist['V'][i].ravel())
        pcm_rho.set_array(hist['rho'][i].ravel())
        time_text.set_text(f"Čas: {hist['t'][i] * 1e6:.2f} µs")
        return pcm_V, pcm_rho, time_text

    anim = animation.FuncAnimation(fig2, animate_fields, frames=len(hist['t']), interval=100, blit=False)
    animations.append(anim)
    fig2.tight_layout()

    if plot_toggles.get('save_plots', False):
        print("  -> Ukládám 2D animaci polí (generování heat-map může trvat déle)...")
        anim.save('out_2d_animace_pole_potencial.gif', writer='pillow', fps=15)
        print("  [OK] Uloženo: out_2d_animace_pole_potencial.gif")


def _animate_2d_particles(hist: Dict[str, Any], params: SimulationParams2D, plot_toggles: Dict[str, bool],
                          animations: list):
    if not plot_toggles.get('show_particles_anim', True):
        return

    fig3, ax_pos = plt.subplots(figsize=(10, 6))
    fig3.canvas.manager.set_window_title('2D Kinetika: Pozice částic')

    scat_e = ax_pos.scatter([], [], s=2, color='red', alpha=0.3, label='Elektrony')
    scat_i = ax_pos.scatter([], [], s=2, color='blue', alpha=0.3, label='Ionty')

    circle = plt.Circle((params.x_antenna, params.y_antenna), params.r_antenna, color='black', fill=False, ls='--',
                        lw=2, label='Detekční Anténa')
    ax_pos.add_patch(circle)
    ax_pos.axvline(x=0, color='grey', lw=4, alpha=0.5, label='Místo dopadu')

    ax_pos.set_xlim(0, params.L_domain)
    ax_pos.set_ylim(-params.H_domain, params.H_domain)
    ax_pos.set_title("Expanze 2D Plazmatického Oblaku")
    ax_pos.set_xlabel("Vzdálenost x [m]")
    ax_pos.set_ylabel("Vzdálenost y [m]")
    ax_pos.legend(loc='upper right')

    time_text = ax_pos.text(0.02, 0.90, '', transform=ax_pos.transAxes, bbox=dict(facecolor='white', alpha=0.8))

    def animate_particles(i):
        mask_e = ~np.isnan(hist['x_e'][i])
        mask_i = ~np.isnan(hist['x_i'][i])

        data_e = np.column_stack((hist['x_e'][i][mask_e], hist['y_e'][i][mask_e]))
        data_i = np.column_stack((hist['x_i'][i][mask_i], hist['y_i'][i][mask_i]))

        if len(data_e) > 0: scat_e.set_offsets(data_e)
        if len(data_i) > 0: scat_i.set_offsets(data_i)

        time_text.set_text(f"Čas: {hist['t'][i] * 1e6:.2f} µs")
        return scat_e, scat_i, time_text

    anim = animation.FuncAnimation(fig3, animate_particles, frames=len(hist['t']), interval=100, blit=False)
    animations.append(anim)
    fig3.tight_layout()

    if plot_toggles.get('save_plots', False):
        print("  -> Ukládám 2D animaci pozic částic...")
        anim.save('out_2d_animace_pozice.gif', writer='pillow', fps=15)
        print("  [OK] Uloženo: out_2d_animace_pozice.gif")


def _animate_velocity_distribution(hist: Dict[str, Any], plot_toggles: Dict[str, bool], animations: list):
    if not plot_toggles.get('show_velocity_anim', True):
        return

    fig4, ax_v = plt.subplots(figsize=(10, 4))
    fig4.canvas.manager.set_window_title('Termodynamika: Rychlosti vx')

    line_ve, = ax_v.plot([], [], lw=2, color='red', drawstyle='steps-mid', label='Elektrony')
    line_vi, = ax_v.plot([], [], lw=2, color='blue', drawstyle='steps-mid', label='Ionty')
    ax_v.set_title("Rozložení dopředné rychlosti (v_x)")
    ax_v.set_xlabel("Rychlost v_x [m/s]")
    ax_v.grid(True, linestyle=':')
    ax_v.legend()

    # Nalezení globálních extrémů v_x pro fixaci osy
    valid_vxe = [v[~np.isnan(v)] for v in hist['vx_e'] if len(v[~np.isnan(v)]) > 0]
    valid_vxi = [v[~np.isnan(v)] for v in hist['vx_i'] if len(v[~np.isnan(v)]) > 0]

    vx_min, vx_max = 0, 0
    if valid_vxe and valid_vxi:
        vx_min = min(np.min([np.min(v) for v in valid_vxe]), np.min([np.min(v) for v in valid_vxi]))
        vx_max = max(np.max([np.max(v) for v in valid_vxe]), np.max([np.max(v) for v in valid_vxi]))
        # Mírný padding osy
        pad = (vx_max - vx_min) * 0.05
        ax_v.set_xlim(vx_min - pad, vx_max + pad)

    def animate_vel(i):
        vxe = hist['vx_e'][i][~np.isnan(hist['vx_e'][i])]
        vxi = hist['vx_i'][i][~np.isnan(hist['vx_i'][i])]

        max_y = 10
        if len(vxe) > 1:
            ce, be = np.histogram(vxe, bins=50, range=(vx_min, vx_max))
            line_ve.set_data((be[:-1] + be[1:]) / 2, ce)
            max_y = max(max_y, ce.max())
        if len(vxi) > 1:
            ci, bi = np.histogram(vxi, bins=50, range=(vx_min, vx_max))
            line_vi.set_data((bi[:-1] + bi[1:]) / 2, ci)
            max_y = max(max_y, ci.max())

        ax_v.set_ylim(0, max_y * 1.1)
        return line_ve, line_vi

    anim = animation.FuncAnimation(fig4, animate_vel, frames=len(hist['t']), interval=100, blit=False)
    animations.append(anim)
    fig4.tight_layout()

    if plot_toggles.get('save_plots', False):
        print("  -> Ukládám animaci rozložení rychlostí...")
        anim.save('out_2d_animace_rychlosti.gif', writer='pillow', fps=15)
        print("  [OK] Uloženo: out_2d_animace_rychlosti.gif")


def plot_simulation_results_2d(results: Dict[str, Any], params: SimulationParams2D,
                               plot_toggles: Optional[Dict[str, bool]] = None):
    # Pokud není specifikováno, systém povolí grafiku, ale ukládání je ve výchozím stavu vypnuto
    if plot_toggles is None:
        plot_toggles = {
            'show_currents': True,
            'show_fields_anim': True,
            'show_particles_anim': True,
            'show_velocity_anim': True,
            'save_plots': False  # Výchozí stav ukládání
        }

    print("=====================================================")
    print("Generuji 2D Vizualizace a připravuji výstupy...")
    print("=====================================================")

    animations = []
    hist = results['history']

    _plot_currents_and_voltage(results, params, plot_toggles)
    _animate_2d_fields(hist, params, plot_toggles, animations)
    _animate_2d_particles(hist, params, plot_toggles, animations)
    _animate_velocity_distribution(hist, plot_toggles, animations)

    print("=====================================================")
    print("Vizualizace dokončena.")
    if animations:
        print(f"Spouštím okna s grafy (Aktivní 2D animace: {len(animations)}).")
    plt.show()