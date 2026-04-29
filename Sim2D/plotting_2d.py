#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MODUL C: Vizualizace 2D Simulace a export do souborů
Zahrnuje čtení konfigurace 'PlottingConfig' z JSON souboru pro řízení výstupů.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import Dict, Any, Tuple

from input_data_2d import calc_Ew_2d, calc_Vw_2d, SimulationParams2D, PlottingConfig


def _get_v_mag_2d(hist: Dict[str, Any], species: str, i: int) -> np.ndarray:
    vx = hist[f'vx_{species}'][i]
    vy = hist[f'vy_{species}'][i]
    mask = ~np.isnan(vx)
    return np.sqrt(vx[mask] ** 2 + vy[mask] ** 2)


def _get_v_signed(hist: Dict[str, Any], species: str, i: int) -> np.ndarray:
    v_x_curr = hist[f'vx_{species}'][i]
    return v_x_curr[~np.isnan(v_x_curr)]


def _plot_currents_and_voltage(results: Dict[str, Any], params: SimulationParams2D, plot_cfg: PlottingConfig) -> None:
    if not plot_cfg.show_currents:
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
    ax1a.set_title(f"Proudy tekoucí do 2D antény (x = {params.x_antenna} m)")
    ax1a.set_ylabel("Proud [nA]")
    ax1a.legend(loc='upper right')
    ax1a.grid(True, linestyle=':')

    tau_us = params.R_ant * params.C_ant * 1e6
    ax1b.plot(params.time_array * 1e6, results['voltage_ant'] * 1e3, color='green', lw=2.5,
              label=f'Napětí ($\\tau$ = {tau_us:.1f} µs)')
    ax1b.axvline(x=params.t_delay * 1e6, color='grey', linestyle='--', alpha=0.7)
    ax1b.set_title("Odezva napětí (RC Obvod)")
    ax1b.set_xlabel("Čas [µs]")
    ax1b.set_ylabel("Napětí [mV]")
    ax1b.legend(loc='upper right')
    ax1b.grid(True, linestyle=':')
    fig1.tight_layout()

    if plot_cfg.save_plots:
        fig1.savefig(plot_cfg.file_currents, dpi=300, bbox_inches='tight')
        print(f"  [OK] Uloženo statické zobrazení: {plot_cfg.file_currents}")


def _animate_2d_fields(hist: Dict[str, Any], params: SimulationParams2D, plot_cfg: PlottingConfig,
                       animations: list) -> None:
    if not plot_cfg.show_fields_anim:
        return

    fig2, (ax_V, ax_rho) = plt.subplots(2, 1, figsize=(10, 10))
    fig2.canvas.manager.set_window_title('2D Makroskopická Pole')

    V_max = np.max([np.max(V) for V in hist['V']])
    V_min = np.min([np.min(V) for V in hist['V']])
    if V_max == V_min: V_max += 1.0; V_min -= 1.0

    rho_max = np.max([np.max(np.abs(rho)) for rho in hist['rho']])
    if rho_max == 0: rho_max = 1e-12

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

    circle_V = plt.Circle((params.x_antenna, params.y_antenna), params.r_antenna, color='white', fill=False, ls='--')
    circle_rho = plt.Circle((params.x_antenna, params.y_antenna), params.r_antenna, color='black', fill=False, ls='--')
    ax_V.add_patch(circle_V)
    ax_rho.add_patch(circle_rho)

    ax_V.axvline(x=0, ymin=0.3, ymax=0.7, color='white', lw=4, alpha=0.5, label='Povrch sondy')
    ax_rho.axvline(x=0, ymin=0.3, ymax=0.7, color='black', lw=4, alpha=0.5, label='Povrch sondy')

    time_text = ax_V.text(0.02, 0.90, '', transform=ax_V.transAxes, color='white', weight='bold')

    def animate_fields(i):
        pcm_V.set_array(hist['V'][i].ravel())
        pcm_rho.set_array(hist['rho'][i].ravel())
        time_text.set_text(f"Čas: {hist['t'][i] * 1e6:.2f} µs")
        return pcm_V, pcm_rho, time_text

    anim = animation.FuncAnimation(fig2, animate_fields, frames=len(hist['t']), interval=100, blit=False)
    animations.append(anim)
    fig2.tight_layout()

    if plot_cfg.save_plots:
        print(f"  -> Ukládám 2D animaci polí...")
        anim.save(plot_cfg.file_fields_anim, writer='pillow', fps=15)
        print(f"  [OK] Uloženo: {plot_cfg.file_fields_anim}")


def _plot_weighting_field(params: SimulationParams2D, plot_cfg: PlottingConfig) -> None:
    if not plot_cfg.show_weighting_field:
        return

    fig4, ax4 = plt.subplots(figsize=(10, 4))
    fig4.canvas.manager.set_window_title('Ramo-Shockley: Citlivost antény')

    y_slice = np.full_like(params.x_grid, params.y_antenna)
    Vw_1d = calc_Vw_2d(params.x_grid, y_slice, params.x_antenna, params.y_antenna, params.w_width)
    ax4.plot(params.x_grid, Vw_1d, lw=2, color='orange', linestyle='--', label='Váhový potenciál $V_w(x)$')

    Ewx, _ = calc_Ew_2d(params.x_grid, y_slice, params.x_antenna, params.y_antenna, params.w_width)
    ax4.plot(params.x_grid, Ewx, lw=2, color='magenta', label='Váhové pole $E_{w,x}(x)$')

    ax4.axvline(x=params.x_antenna, color='red', linestyle='-', alpha=0.6, lw=2, label='Umístění antény')
    ax4.axhline(y=0, color='black', lw=1, alpha=0.5)
    ax4.set_title("1D řez váhovou funkcí přes střed antény (Bezkontaktní měření)")
    ax4.set_xlabel("Vzdálenost x [m]")
    ax4.set_ylabel("Amplituda citlivosti")
    ax4.legend(loc='upper left')
    ax4.grid(True, linestyle=':', alpha=0.7)
    fig4.tight_layout()

    if plot_cfg.save_plots:
        fig4.savefig(plot_cfg.file_weighting, dpi=300, bbox_inches='tight')
        print(f"  [OK] Uloženo statické zobrazení: {plot_cfg.file_weighting}")


def _animate_2d_particles(hist: Dict[str, Any], params: SimulationParams2D, plot_cfg: PlottingConfig,
                          animations: list) -> None:
    if not plot_cfg.show_particles_anim:
        return

    fig3, ax_pos = plt.subplots(figsize=(10, 6))
    fig3.canvas.manager.set_window_title('2D Kinetika: Pozice částic')

    scat_e = ax_pos.scatter([], [], s=2, color='red', alpha=0.3, label='Elektrony')
    scat_i = ax_pos.scatter([], [], s=2, color='blue', alpha=0.3, label='Ionty')

    circle = plt.Circle((params.x_antenna, params.y_antenna), params.r_antenna, color='black', fill=False, ls='--',
                        lw=2, label='Detekční Anténa')
    ax_pos.add_patch(circle)
    ax_pos.axvline(x=0, ymin=0.3, ymax=0.7, color='grey', lw=4, alpha=0.5, label='Povrch sondy')

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

    if plot_cfg.save_plots:
        print(f"  -> Ukládám 2D animaci pozic částic...")
        anim.save(plot_cfg.file_particles_anim, writer='pillow', fps=15)
        print(f"  [OK] Uloženo: {plot_cfg.file_particles_anim}")


def _animate_velocity_distribution(hist: Dict[str, Any], plot_cfg: PlottingConfig, animations: list) -> None:
    if not plot_cfg.show_velocity_anim:
        return

    fig4, ax_v = plt.subplots(figsize=(10, 4))
    fig4.canvas.manager.set_window_title('Termodynamika: Celková velikost rychlosti')

    line_ve, = ax_v.plot([], [], lw=2, color='red', drawstyle='steps-mid', label='Elektrony')
    line_vi, = ax_v.plot([], [], lw=2, color='blue', drawstyle='steps-mid', label='Ionty')
    ax_v.set_title("Rozložení celkové velikosti rychlosti (|v|)")
    ax_v.set_xlabel("Velikost rychlosti |v| [m/s]")
    ax_v.grid(True, linestyle=':')
    ax_v.legend()

    v_mag_max = 0.0
    for species in ['e', 'i']:
        for i in range(len(hist['t'])):
            v_mag = _get_v_mag_2d(hist, species, i)
            if len(v_mag) > 0:
                v_mag_max = max(v_mag_max, np.max(v_mag))

    v_mag_min = 0.0
    pad = v_mag_max * 0.05 if v_mag_max != 0 else 1e3
    ax_v.set_xlim(v_mag_min, v_mag_max + pad)

    def animate_vel(i):
        v_mag_e = _get_v_mag_2d(hist, 'e', i)
        v_mag_i = _get_v_mag_2d(hist, 'i', i)

        max_y = 10
        if len(v_mag_e) > 1:
            ce, be = np.histogram(v_mag_e, bins=150, range=(v_mag_min, v_mag_max + pad))
            line_ve.set_data((be[:-1] + be[1:]) / 2, ce)
            max_y = max(max_y, ce.max())
        if len(v_mag_i) > 1:
            ci, bi = np.histogram(v_mag_i, bins=150, range=(v_mag_min, v_mag_max + pad))
            line_vi.set_data((bi[:-1] + bi[1:]) / 2, ci)
            max_y = max(max_y, ci.max())

        ax_v.set_ylim(0, max_y * 1.1)
        return line_ve, line_vi

    anim = animation.FuncAnimation(fig4, animate_vel, frames=len(hist['t']), interval=100, blit=False)
    animations.append(anim)
    fig4.tight_layout()

    if plot_cfg.save_plots:
        print(f"  -> Ukládám animaci histogramu rychlostí...")
        anim.save(plot_cfg.file_velocity_anim, writer='pillow', fps=15)
        print(f"  [OK] Uloženo: {plot_cfg.file_velocity_anim}")


def _animate_phase_space(hist: Dict[str, Any], params: SimulationParams2D, plot_cfg: PlottingConfig,
                         animations: list) -> None:
    if not plot_cfg.show_phase_space_anim:
        return

    fig7, (ax_ps_e, ax_ps_i) = plt.subplots(2, 1, figsize=(10, 9))
    fig7.canvas.manager.set_window_title('PIC Analýza: Fázový prostor (x-v)')

    scat_ps_e = ax_ps_e.scatter([], [], s=1, color='red', alpha=0.3, edgecolors='none')
    ax_ps_e.set_title("Fázový prostor elektronů (x, v_{x,e})")
    ax_ps_e.set_xlabel("Vzdálenost x [m]")
    ax_ps_e.set_ylabel("Dopředná rychlost $v_{x,e}$ [m/s]")
    ax_ps_e.set_xlim(0, params.L_domain)
    ax_ps_e.axvline(x=params.x_antenna, color='black', linestyle='--', alpha=0.5)
    ax_ps_e.grid(True, linestyle=':', alpha=0.5)

    scat_ps_i = ax_ps_i.scatter([], [], s=1, color='blue', alpha=0.3, edgecolors='none')
    ax_ps_i.set_title("Fázový prostor iontů (x, v_{x,i})")
    ax_ps_i.set_xlabel("Vzdálenost x [m]")
    ax_ps_i.set_ylabel("Dopředná rychlost $v_{x,i}$ [m/s]")
    ax_ps_i.set_xlim(0, params.L_domain)
    ax_ps_i.axvline(x=params.x_antenna, color='black', linestyle='--', alpha=0.5)
    ax_ps_i.grid(True, linestyle=':', alpha=0.5)

    # Zjištění osy rychlostí
    v_e_signed_min, v_e_signed_max = 0.0, 0.0
    v_i_signed_min, v_i_signed_max = 0.0, 0.0
    for i in range(len(hist['t'])):
        v_e_s = _get_v_signed(hist, 'e', i)
        v_i_s = _get_v_signed(hist, 'i', i)
        if len(v_e_s) > 0:
            v_e_signed_min = min(v_e_signed_min, np.min(v_e_s))
            v_e_signed_max = max(v_e_signed_max, np.max(v_e_s))
        if len(v_i_s) > 0:
            v_i_signed_min = min(v_i_signed_min, np.min(v_i_s))
            v_i_signed_max = max(v_i_signed_max, np.max(v_i_s))

    pad_e = (v_e_signed_max - v_e_signed_min) * 0.05 if v_e_signed_max != v_e_signed_min else 1e5
    pad_i = (v_i_signed_max - v_i_signed_min) * 0.05 if v_i_signed_max != v_i_signed_min else 1e3

    ax_ps_e.set_ylim(v_e_signed_min - pad_e, v_e_signed_max + pad_e)
    ax_ps_i.set_ylim(v_i_signed_min - pad_i, v_i_signed_max + pad_i)

    time_text_fig7 = ax_ps_e.text(0.02, 0.85, '', transform=ax_ps_e.transAxes, fontsize=11,
                                  bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9, edgecolor='gray'))

    def animate_ps(i):
        v_e_sign = _get_v_signed(hist, 'e', i)
        v_i_sign = _get_v_signed(hist, 'i', i)
        mask_e = ~np.isnan(hist['x_e'][i])
        mask_i = ~np.isnan(hist['x_i'][i])

        data_e = np.column_stack((hist['x_e'][i][mask_e], v_e_sign))
        data_i = np.column_stack((hist['x_i'][i][mask_i], v_i_sign))

        if len(data_e) > 0: scat_ps_e.set_offsets(data_e)
        if len(data_i) > 0: scat_ps_i.set_offsets(data_i)
        time_text_fig7.set_text(f"Čas: {hist['t'][i] * 1e6:.2f} µs")
        return scat_ps_e, scat_ps_i, time_text_fig7

    anim_ps = animation.FuncAnimation(fig7, animate_ps, frames=len(hist['t']), interval=100, blit=False)
    animations.append(anim_ps)
    fig7.tight_layout()

    if plot_cfg.save_plots:
        print(f"  -> Ukládám animaci fázového prostoru...")
        anim_ps.save(plot_cfg.file_phase_space, writer='pillow', fps=15)
        print(f"  [OK] Uloženo: {plot_cfg.file_phase_space}")


def _export_data_csv(results: Dict[str, Any], params: SimulationParams2D, plot_cfg: PlottingConfig) -> None:
    if not plot_cfg.export_data_csv:
        return

    print("  -> Exportuji makroskopická data do CSV formátu...")
    try:
        export_matrix = np.column_stack((
            params.time_array,
            results['smooth_induced'],
            results['smooth_collected'],
            results['voltage_ant']
        ))
        header = "Cas_s,Indukovany_proud_A,Nasbirany_proud_A,Napeti_antena_V"
        np.savetxt(plot_cfg.file_csv, export_matrix, delimiter=",", header=header, comments="")
        print(f"  [OK] Data úspěšně uložena do: {plot_cfg.file_csv}")
    except Exception as e:
        print(f"  [CHYBA] Nepodařilo se exportovat CSV. Detail: {e}")


def plot_simulation_results_2d(results: Dict[str, Any], params: SimulationParams2D, plot_cfg: PlottingConfig) -> None:
    print("=====================================================")
    print("Generuji 2D Vizualizace a připravuji výstupy...")
    print("=====================================================")

    animations = []
    hist = results['history']

    _plot_currents_and_voltage(results, params, plot_cfg)
    _animate_2d_fields(hist, params, plot_cfg, animations)
    _plot_weighting_field(params, plot_cfg)
    _animate_2d_particles(hist, params, plot_cfg, animations)
    _animate_velocity_distribution(hist, plot_cfg, animations)
    _animate_phase_space(hist, params, plot_cfg, animations)
    _export_data_csv(results, params, plot_cfg)

    print("=====================================================")
    print("Post-processing ukončen. Soubory byly aktualizovány.")
    if animations:
        print(f"Zobrazuji interaktivní okna ({len(animations)}). Zavřete je pro ukončení programu.")
    plt.show()