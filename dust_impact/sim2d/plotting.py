# -*- coding: utf-8 -*-
"""
Visualization module for 2D PIC simulation.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import Dict, Any, Tuple

from dust_impact.sim2d.input_data import calc_Ew_2d, calc_Vw_2d, SimulationParams2D, PlottingConfig


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
    fig1.canvas.manager.set_window_title('2D Signals: Currents and Voltages')

    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']

    for a_idx, ant in enumerate(params.antennas):
        c = colors[a_idx % len(colors)]
        ax1a.plot(params.time_array * 1e6, results['smooth_total'][a_idx] * 1e9, color=c, lw=2,
                  label=f'Total Current (Antenna {a_idx + 1})')
        ax1a.plot(params.time_array * 1e6, results['smooth_induced'][a_idx] * 1e9, color=c, ls=':', alpha=0.5)

        tau_us = ant['R'] * ant['C'] * 1e6
        ax1b.plot(params.time_array * 1e6, results['voltage_ant'][a_idx] * 1e3, color=c, lw=2.5,
                  label=f'Voltage Antenna {a_idx + 1} ($\\tau$ = {tau_us:.1f} µs)')

    ax1a.axvline(x=params.t_delay * 1e6, color='grey', linestyle='--', alpha=0.7)
    ax1a.set_title("Total Currents to Individual Antennas")
    ax1a.set_ylabel("Current [nA]")
    ax1a.legend(loc='upper right')
    ax1a.grid(True, linestyle=':')

    ax1b.axvline(x=params.t_delay * 1e6, color='grey', linestyle='--', alpha=0.7)
    ax1b.set_title("Voltage Response of Parallel RC Circuits")
    ax1b.set_xlabel("Time [µs]")
    ax1b.set_ylabel("Voltage [mV]")
    ax1b.legend(loc='upper right')
    ax1b.grid(True, linestyle=':')
    fig1.tight_layout()

    if plot_cfg.save_plots:
        fig1.savefig(plot_cfg.file_currents, dpi=300, bbox_inches='tight')
        print(f"  [OK] Saved static plot: {plot_cfg.file_currents}")


def _animate_2d_fields(hist: Dict[str, Any], params: SimulationParams2D, plot_cfg: PlottingConfig,
                       animations: list) -> None:
    if not plot_cfg.show_fields_anim:
        return

    fig2, (ax_V, ax_rho) = plt.subplots(2, 1, figsize=(10, 10))
    fig2.canvas.manager.set_window_title('2D Macroscopic Fields')

    V_max = np.max([np.max(V) for V in hist['V']])
    V_min = np.min([np.min(V) for V in hist['V']])
    if V_max == V_min: V_max += 1.0; V_min -= 1.0

    rho_max = np.max([np.max(np.abs(rho)) for rho in hist['rho']])
    if rho_max == 0: rho_max = 1e-12

    pcm_V = ax_V.pcolormesh(params.X_mat, params.Y_mat, hist['V'][0], shading='gouraud', cmap='viridis', vmin=V_min,
                            vmax=V_max)
    fig2.colorbar(pcm_V, ax=ax_V, label='Potential [V]')
    ax_V.set_title("2D Electric Potential")
    ax_V.set_ylabel("y [m]")

    pcm_rho = ax_rho.pcolormesh(params.X_mat, params.Y_mat, hist['rho'][0], shading='gouraud', cmap='seismic',
                                vmin=-rho_max, vmax=rho_max)
    fig2.colorbar(pcm_rho, ax=ax_rho, label='Charge Density [C/m³]')
    ax_rho.set_title("2D Space Charge Density")
    ax_rho.set_xlabel("x [m]")
    ax_rho.set_ylabel("y [m]")

    ax_V.axvline(x=0, ymin=0.3, ymax=0.7, color='white', lw=4, alpha=0.5, label='Spacecraft surface')
    ax_rho.axvline(x=0, ymin=0.3, ymax=0.7, color='black', lw=4, alpha=0.5, label='Spacecraft surface')

    for a_idx, ant in enumerate(params.antennas):
        ax_V.plot([ant['x1'], ant['x2']], [ant['y1'], ant['y2']], color='white', lw=2, ls='--')
        ax_rho.plot([ant['x1'], ant['x2']], [ant['y1'], ant['y2']], color='black', lw=2, ls='--')
        if a_idx == 0:
            ax_rho.plot([], [], color='black', lw=2, ls='--', label='Wire Antennas')

    time_text = ax_V.text(0.02, 0.90, '', transform=ax_V.transAxes, color='white', weight='bold')

    def animate_fields(i):
        pcm_V.set_array(hist['V'][i].ravel())
        pcm_rho.set_array(hist['rho'][i].ravel())
        time_text.set_text(f"Time: {hist['t'][i] * 1e6:.2f} µs")
        return pcm_V, pcm_rho, time_text

    anim = animation.FuncAnimation(fig2, animate_fields, frames=len(hist['t']), interval=100, blit=False)
    animations.append(anim)
    fig2.tight_layout()

    if plot_cfg.save_plots:
        print(f"  -> Saving 2D fields animation...")
        anim.save(plot_cfg.file_fields_anim, writer='pillow', fps=15)
        print(f"  [OK] Saved: {plot_cfg.file_fields_anim}")


def _plot_weighting_field(params: SimulationParams2D, plot_cfg: PlottingConfig) -> None:
    if not plot_cfg.show_weighting_field:
        return

    fig4, ax4 = plt.subplots(figsize=(10, 4))
    fig4.canvas.manager.set_window_title('Ramo-Shockley: Antenna Sensitivity')
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']

    for a_idx, ant in enumerate(params.antennas):
        c = colors[a_idx % len(colors)]

        y_mid = (ant['y1'] + ant['y2']) / 2.0
        y_slice = np.full_like(params.x_grid, y_mid)

        Vw_1d = calc_Vw_2d(params.x_grid, y_slice, ant['x1'], ant['y1'], ant['x2'], ant['y2'], ant['w_width'])
        ax4.plot(params.x_grid, Vw_1d, color=c, lw=2, linestyle='--', label=f'$V_w$ (Antenna {a_idx + 1})')

        Ewx, _ = calc_Ew_2d(params.x_grid, y_slice, ant['x1'], ant['y1'], ant['x2'], ant['y2'], ant['w_width'])
        ax4.plot(params.x_grid, Ewx, color=c, lw=2, label=f'$E_{{w,x}}$ (Antenna {a_idx + 1})')

        x_mid = (ant['x1'] + ant['x2']) / 2.0
        ax4.axvline(x=x_mid, color=c, linestyle='-', alpha=0.3, lw=2)

    ax4.axhline(y=0, color='black', lw=1, alpha=0.5)
    ax4.set_title("1D Slices of Weighting Function for All Antennas")
    ax4.set_xlabel("Distance x [m]")
    ax4.set_ylabel("Sensitivity Amplitude")
    if len(params.antennas) > 0:
        ax4.legend(loc='upper left', ncol=max(1, min(3, len(params.antennas))))
    ax4.grid(True, linestyle=':', alpha=0.7)
    fig4.tight_layout()

    if plot_cfg.save_plots:
        fig4.savefig(plot_cfg.file_weighting, dpi=300, bbox_inches='tight')
        print(f"  [OK] Saved static plot: {plot_cfg.file_weighting}")


def _animate_2d_particles(hist: Dict[str, Any], params: SimulationParams2D, plot_cfg: PlottingConfig,
                          animations: list) -> None:
    if not plot_cfg.show_particles_anim:
        return

    fig3, ax_pos = plt.subplots(figsize=(10, 6))
    fig3.canvas.manager.set_window_title('2D Kinetics: Particle Positions')

    scat_e = ax_pos.scatter([], [], s=12, color='cyan', alpha=0.7, label='Electrons')
    scat_i = ax_pos.scatter([], [], s=16, color='blue', alpha=0.7, label='Ions')

    ax_pos.axvline(x=0, ymin=0.3, ymax=0.7, color='grey', lw=4, alpha=0.5, label='Spacecraft surface')
    ax_pos.scatter([0], [params.y_impact], color='orange', marker='*', s=200, label='Impact Location', zorder=5)

    for a_idx, ant in enumerate(params.antennas):
        lbl = 'Detection Antenna' if a_idx == 0 else ""
        ax_pos.plot([ant['x1'], ant['x2']], [ant['y1'], ant['y2']], color='black', lw=2, ls='--', label=lbl)

    ax_pos.set_xlim(0, params.L_domain)
    ax_pos.set_ylim(-params.H_domain, params.H_domain)
    ax_pos.set_title("Expansion of 2D Plasma Cloud")
    ax_pos.set_xlabel("Distance x [m]")
    ax_pos.set_ylabel("Distance y [m]")
    ax_pos.legend(loc='upper right')

    time_text = ax_pos.text(0.02, 0.90, '', transform=ax_pos.transAxes, bbox=dict(facecolor='white', alpha=0.8))

    def animate_particles(i):
        mask_e = ~np.isnan(hist['x_e'][i])
        mask_i = ~np.isnan(hist['x_i'][i])

        data_e = np.column_stack((hist['x_e'][i][mask_e], hist['y_e'][i][mask_e]))
        data_i = np.column_stack((hist['x_i'][i][mask_i], hist['y_i'][i][mask_i]))

        if len(data_e) > 0: scat_e.set_offsets(data_e)
        if len(data_i) > 0: scat_i.set_offsets(data_i)

        time_text.set_text(f"Time: {hist['t'][i] * 1e6:.2f} µs")
        return scat_e, scat_i, time_text

    anim = animation.FuncAnimation(fig3, animate_particles, frames=len(hist['t']), interval=100, blit=False)
    animations.append(anim)
    fig3.tight_layout()

    if plot_cfg.save_plots:
        print(f"  -> Saving 2D particle positions animation...")
        anim.save(plot_cfg.file_particles_anim, writer='pillow', fps=15)
        print(f"  [OK] Saved: {plot_cfg.file_particles_anim}")


def _animate_velocity_distribution(hist: Dict[str, Any], plot_cfg: PlottingConfig, animations: list) -> None:
    if not plot_cfg.show_velocity_anim:
        return

    fig4, ax_v = plt.subplots(figsize=(10, 4))
    fig4.canvas.manager.set_window_title('Thermodynamics: Total Velocity Magnitude')

    line_ve, = ax_v.plot([], [], lw=2, color='red', drawstyle='steps-mid', label='Electrons')
    line_vi, = ax_v.plot([], [], lw=2, color='blue', drawstyle='steps-mid', label='Ions')
    ax_v.set_title("Distribution of Total Velocity Magnitude (|v|)")
    ax_v.set_xlabel("Velocity Magnitude |v| [m/s]")
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
            ce, be = np.histogram(v_mag_e, bins=200, range=(v_mag_min, v_mag_max + pad))
            line_ve.set_data((be[:-1] + be[1:]) / 2, ce)
            max_y = max(max_y, ce.max())
        if len(v_mag_i) > 1:
            ci, bi = np.histogram(v_mag_i, bins=200, range=(v_mag_min, v_mag_max + pad))
            line_vi.set_data((bi[:-1] + bi[1:]) / 2, ci)
            max_y = max(max_y, ci.max())

        ax_v.set_ylim(0, max_y * 1.1)
        return line_ve, line_vi

    anim = animation.FuncAnimation(fig4, animate_vel, frames=len(hist['t']), interval=100, blit=False)
    animations.append(anim)
    fig4.tight_layout()

    if plot_cfg.save_plots:
        print(f"  -> Saving velocity histogram animation...")
        anim.save(plot_cfg.file_velocity_anim, writer='pillow', fps=15)
        print(f"  [OK] Saved: {plot_cfg.file_velocity_anim}")


def _animate_phase_space(hist: Dict[str, Any], params: SimulationParams2D, plot_cfg: PlottingConfig,
                          animations: list) -> None:
    if not plot_cfg.show_phase_space_anim:
        return

    fig7, (ax_ps_e, ax_ps_i) = plt.subplots(2, 1, figsize=(10, 9))
    fig7.canvas.manager.set_window_title('PIC Analysis: Phase Space (x-v)')

    scat_ps_e = ax_ps_e.scatter([], [], s=1, color='red', alpha=0.3, edgecolors='none')
    ax_ps_e.set_title("Phase Space of Electrons (x, v_{x,e})")
    ax_ps_e.set_xlabel("Distance x [m]")
    ax_ps_e.set_ylabel("Forward Velocity $v_{x,e}$ [m/s]")
    ax_ps_e.set_xlim(0, params.L_domain)
    for ant in params.antennas:
        x_min, x_max = min(ant['x1'], ant['x2']), max(ant['x1'], ant['x2'])
        if x_min == x_max:
            ax_ps_e.axvline(x=x_min, color='black', linestyle='--', alpha=0.5)
        else:
            ax_ps_e.axvspan(x_min, x_max, color='black', alpha=0.1)
    ax_ps_e.grid(True, linestyle=':', alpha=0.5)

    scat_ps_i = ax_ps_i.scatter([], [], s=1, color='blue', alpha=0.3, edgecolors='none')
    ax_ps_i.set_title("Phase Space of Ions (x, v_{x,i})")
    ax_ps_i.set_xlabel("Distance x [m]")
    ax_ps_i.set_ylabel("Forward Velocity $v_{x,i}$ [m/s]")
    ax_ps_i.set_xlim(0, params.L_domain)
    for ant in params.antennas:
        x_min, x_max = min(ant['x1'], ant['x2']), max(ant['x1'], ant['x2'])
        if x_min == x_max:
            ax_ps_i.axvline(x=x_min, color='black', linestyle='--', alpha=0.5)
        else:
            ax_ps_i.axvspan(x_min, x_max, color='black', alpha=0.1)
    ax_ps_i.grid(True, linestyle=':', alpha=0.5)

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
        time_text_fig7.set_text(f"Time: {hist['t'][i] * 1e6:.2f} µs")
        return scat_ps_e, scat_ps_i, time_text_fig7

    anim_ps = animation.FuncAnimation(fig7, animate_ps, frames=len(hist['t']), interval=100, blit=False)
    animations.append(anim_ps)
    fig7.tight_layout()

    if plot_cfg.save_plots:
        print(f"  -> Saving phase space animation...")
        anim_ps.save(plot_cfg.file_phase_space, writer='pillow', fps=15)
        print(f"  [OK] Saved: {plot_cfg.file_phase_space}")


def _export_data_csv(results: Dict[str, Any], params: SimulationParams2D, plot_cfg: PlottingConfig) -> None:
    if not plot_cfg.export_data_csv:
        return

    print("  -> Exporting macroscopic data to CSV format...")
    try:
        header_cols = ["Time_s"]
        cols = [params.time_array]

        for a_idx in range(len(params.antennas)):
            header_cols.extend([f"Induced_current_Ant{a_idx + 1}_A",
                                f"Collected_current_Ant{a_idx + 1}_A",
                                f"Voltage_Ant{a_idx + 1}_V"])
            cols.extend([results['smooth_induced'][a_idx],
                         results['smooth_collected'][a_idx],
                         results['voltage_ant'][a_idx]])

        export_matrix = np.column_stack(cols)
        header = ",".join(header_cols)
        np.savetxt(plot_cfg.file_csv, export_matrix, delimiter=",", header=header, comments="")
        print(f"  [OK] Data successfully saved to: {plot_cfg.file_csv}")
    except Exception as e:
        print(f"  [ERROR] Failed to export CSV: {e}")


def plot_simulation_results_2d(results: Dict[str, Any], params: SimulationParams2D, plot_cfg: PlottingConfig) -> None:
    print("=====================================================")
    print("Generating 2D Visualizations and preparing outputs...")
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
    print("Post-processing finished.")
    if getattr(plot_cfg, 'show_interactive_gui_windows', True):
        if animations:
            print(f"Showing interactive windows ({len(animations)}). Close them to exit the program.")
        plt.show()
    else:
        plt.close('all')
        print("  [OK] Headless mode: Plots saved to disk without interactive GUI windows.")
