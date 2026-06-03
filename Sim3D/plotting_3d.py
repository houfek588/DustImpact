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
    fig1.canvas.manager.set_window_title('3D Signals: Currents and Voltages')
    colors = ['blue', 'red', 'green', 'orange', 'purple']

    for a_idx in range(len(params.C_ant)):
        c = colors[a_idx % len(colors)]
        ax1a.plot(params.time_array * 1e6, results['smooth_total'][a_idx] * 1e9, color=c, lw=2,
                  label=f'Total (Antenna {a_idx + 1})')
        ax1a.plot(params.time_array * 1e6, results['smooth_induced'][a_idx] * 1e9, color=c, ls=':', alpha=0.5)
        tau_us = params.R_ant[a_idx] * params.C_ant[a_idx] * 1e6
        ax1b.plot(params.time_array * 1e6, results['voltage_ant'][a_idx] * 1e3, color=c, lw=2.5,
                  label=f'Antenna {a_idx + 1} ($\\tau$={tau_us:.1f} µs)')

    for ax in (ax1a, ax1b):
        ax.axvline(x=params.t_delay * 1e6, color='grey', ls='--', alpha=0.7)
        ax.grid(True, ls=':')
        ax.legend()

    ax1a.set_ylabel("Current [nA]")
    ax1b.set_ylabel("Voltage [mV]")
    ax1b.set_xlabel("Time [µs]")
    fig1.tight_layout()
    if plot_cfg.save_plots: fig1.savefig(plot_cfg.file_currents, dpi=300, bbox_inches='tight')


def _animate_yz_fields_slice(hist: Dict, params: SimulationParams3D, plot_cfg: PlottingConfig3D, anims: list,
                            spacecraft_mask: np.ndarray = None, antenna_masks: list = None):
    if not plot_cfg.show_fields_slice: return

    fig2, ax_V = plt.subplots(figsize=(8, 6))
    fig2.canvas.manager.set_window_title('3D Fields: 2D YZ Slice (x = 0)')

    mid_x = params.Nx // 2
    V_max = np.max([np.max(V[mid_x, :, :]) for V in hist['V']])
    V_min = np.min([np.min(V[mid_x, :, :]) for V in hist['V']])
    if V_max == V_min: V_max += 1.0; V_min -= 1.0

    # Horizontal axis is Y, Vertical axis is Z
    Y, Z = np.meshgrid(params.y_grid, params.z_grid, indexing='ij')
    pcm = ax_V.pcolormesh(Y, Z, hist['V'][0][mid_x, :, :], shading='gouraud', cmap='viridis', vmin=V_min, vmax=V_max)
    fig2.colorbar(pcm, ax=ax_V, label='Potential [V]')

    # Plot exact geometry from masks (slice at mid_x)
    if spacecraft_mask is not None:
        sc_slice = spacecraft_mask[mid_x, :, :]
        if np.any(sc_slice):
            ax_V.contour(Y, Z, sc_slice.astype(float), levels=[0.5], colors='white', linewidths=2)

    if antenna_masks is not None:
        for i, m in enumerate(antenna_masks):
            ant_slice = m[mid_x, :, :]
            if np.any(ant_slice):
                ax_V.contour(Y, Z, ant_slice.astype(float), levels=[0.5], colors='white', linewidths=1, linestyles='dashed')

    ax_V.set_xlabel("Y [m]")
    ax_V.set_ylabel("Z [m]")

    time_text = ax_V.text(0.02, 0.90, '', transform=ax_V.transAxes, color='white', weight='bold')

    def update(i):
        pcm.set_array(hist['V'][i][mid_x, :, :].ravel())
        time_text.set_text(f"Time: {hist['t'][i] * 1e6:.2f} µs")
        return pcm, time_text

    anim = animation.FuncAnimation(fig2, update, frames=len(hist['t']), interval=100, blit=False)
    anims.append(anim)
    if plot_cfg.save_plots: anim.save(plot_cfg.file_fields_anim, writer='pillow', fps=15)


def _animate_yz_particles_slice(hist: Dict, params: SimulationParams3D, plot_cfg: PlottingConfig3D, anims: list,
                               spacecraft_mask: np.ndarray = None, antenna_masks: list = None):
    if not plot_cfg.show_particles_3d: return

    fig_slice, ax_pos = plt.subplots(figsize=(8, 6))
    fig_slice.canvas.manager.set_window_title('3D Particles: 2D YZ Slice (x >= 0)')

    # Plot geometry first (static background for the slice)
    mid_x = params.Nx // 2
    Y_mesh, Z_mesh = np.meshgrid(params.y_grid, params.z_grid, indexing='ij')
    
    if spacecraft_mask is not None:
        sc_slice = spacecraft_mask[mid_x, :, :]
        ax_pos.scatter(Y_mesh[sc_slice], Z_mesh[sc_slice], color='gray', s=10, alpha=0.3, marker='s')

    if antenna_masks is not None:
        colors = ['red', 'green', 'orange', 'purple']
        for i, m in enumerate(antenna_masks):
            ant_slice = m[mid_x, :, :]
            c = colors[i % len(colors)]
            ax_pos.scatter(Y_mesh[ant_slice], Z_mesh[ant_slice], color=c, s=5, alpha=0.5)

    scat_e = ax_pos.scatter([], [], s=2, color='red', alpha=0.3, label='Electrons')
    scat_i = ax_pos.scatter([], [], s=2, color='blue', alpha=0.3, label='Ions')

    ax_pos.set_xlim(-params.L_y, params.L_y)
    ax_pos.set_ylim(-params.L_z, params.L_z)
    ax_pos.set_xlabel("Y [m]")
    ax_pos.set_ylabel("Z [m]")
    ax_pos.set_title("Expanding Plasma Cloud (YZ projection, x $\\geq$ 0)")
    ax_pos.legend(loc='upper right')

    time_text = ax_pos.text(0.02, 0.90, '', transform=ax_pos.transAxes, bbox=dict(facecolor='white', alpha=0.8))

    def update(i):
        me = ~np.isnan(hist['x_e'][i])
        mi = ~np.isnan(hist['x_i'][i])

        # Filter particles to only show those in the YZ plane or behind it (x >= 0)
        me_slice = me & (hist['x_e'][i] >= 0)
        mi_slice = mi & (hist['x_i'][i] >= 0)

        data_e = np.column_stack((hist['y_e'][i][me_slice], hist['z_e'][i][me_slice]))
        data_i = np.column_stack((hist['y_i'][i][mi_slice], hist['z_i'][i][mi_slice]))

        if len(data_e) > 0: scat_e.set_offsets(data_e)
        if len(data_i) > 0: scat_i.set_offsets(data_i)

        time_text.set_text(f"Time: {hist['t'][i] * 1e6:.2f} µs")
        return scat_e, scat_i, time_text

    anim = animation.FuncAnimation(fig_slice, update, frames=len(hist['t']), interval=100, blit=False)
    anims.append(anim)
    if plot_cfg.save_plots: 
        anim.save(plot_cfg.file_particles_anim.replace('.gif', '_yz.gif'), writer='pillow', fps=15)


def _animate_3d_particles(hist: Dict, params: SimulationParams3D, plot_cfg: PlottingConfig3D, anims: list,
                          spacecraft_mask: np.ndarray = None, antenna_masks: list = None):
    if not plot_cfg.show_particles_3d: return

    fig3 = plt.figure(figsize=(10, 8))
    ax = fig3.add_subplot(111, projection='3d')
    fig3.canvas.manager.set_window_title('3D Particle Kinetics and Geometry')

    X, Y, Z = np.meshgrid(params.x_grid, params.y_grid, params.z_grid, indexing='ij')

    # Plot exactly what the physics engine sees as the spacecraft and antennas
    if spacecraft_mask is not None:
        sc_x = X[spacecraft_mask]
        sc_y = Y[spacecraft_mask]
        sc_z = Z[spacecraft_mask]
        if len(sc_x) > 0:
            ax.scatter(sc_x, sc_y, sc_z, color='gray', alpha=0.3, s=15, marker='s', label='Spacecraft Body')

    if antenna_masks is not None:
        colors = ['red', 'green', 'orange', 'purple']
        for i, m in enumerate(antenna_masks):
            c = colors[i % len(colors)]
            ant_x = X[m]
            ant_y = Y[m]
            ant_z = Z[m]
            if len(ant_x) > 0:
                 ax.scatter(ant_x, ant_y, ant_z, color=c, alpha=0.5, s=10, marker='o', label=f'Antenna {i+1}')

    scat_e = ax.scatter([], [], [], s=1, color='red', alpha=0.3, label='Electrons')
    scat_i = ax.scatter([], [], [], s=2, color='blue', alpha=0.3, label='Ions')

    # Nastavení os na nové parametry -L až L
    ax.set_xlim(-params.L_x, params.L_x)
    ax.set_ylim(-params.L_y, params.L_y)
    ax.set_zlim(-params.L_z, params.L_z)

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.legend(loc='upper right')

    ax.scatter(*params.impact_pos, color='orange', s=100, marker='*', label='Impact')
    time_text = ax.text2D(0.02, 0.95, '', transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.8))

    def update(i):
        me, mi = ~np.isnan(hist['x_e'][i]), ~np.isnan(hist['x_i'][i])
        scat_e._offsets3d = (hist['x_e'][i][me], hist['y_e'][i][me], hist['z_e'][i][me])
        scat_i._offsets3d = (hist['x_i'][i][mi], hist['y_i'][i][mi], hist['z_i'][i][mi])
        time_text.set_text(f"Time: {hist['t'][i] * 1e6:.2f} µs")
        return scat_e, scat_i, time_text

    anim = animation.FuncAnimation(fig3, update, frames=len(hist['t']), interval=100, blit=False)
    anims.append(anim)
    if plot_cfg.save_plots: anim.save(plot_cfg.file_particles_anim, writer='pillow', fps=15)


def _animate_velocity_distribution(hist: Dict[str, Any], plot_cfg: PlottingConfig3D, anims: list) -> None:
    if not plot_cfg.show_velocity_anim: return

    fig4, ax_v = plt.subplots(figsize=(10, 4))
    fig4.canvas.manager.set_window_title('Thermodynamics: Total 3D Velocity')

    line_ve, = ax_v.plot([], [], lw=2, color='red', drawstyle='steps-mid', label='Electrons')
    line_vi, = ax_v.plot([], [], lw=2, color='blue', drawstyle='steps-mid', label='Ions')
    ax_v.set_xlabel("Velocity Magnitude |v| [m/s]")
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
        header = ["Time_s"]
        cols = [params.time_array]
        for a_idx in range(len(params.C_ant)):
            header.extend([f"I_ind_{a_idx + 1}_A", f"I_col_{a_idx + 1}_A", f"V_{a_idx + 1}_V"])
            cols.extend(
                [results['smooth_induced'][a_idx], results['smooth_collected'][a_idx], results['voltage_ant'][a_idx]])
        np.savetxt(plot_cfg.file_csv, np.column_stack(cols), delimiter=",", header=",".join(header), comments="")
        print(f"  [OK] CSV saved: {plot_cfg.file_csv}")
    except Exception as e:
        print(f"  [ERROR] CSV export failed: {e}")


def _plot_static_vtk_fields(params: SimulationParams3D, V_bg: np.ndarray, Vw_grids: list):
    """ Plots YZ slices of the loaded VTK fields (Background and Weighting). """
    num_ant = len(Vw_grids)
    # Background + Antennas
    n_plots = 1 + num_ant
    cols = 2
    rows = (n_plots + 1) // 2

    fig, axes = plt.subplots(rows, cols, figsize=(12, 5 * rows))
    fig.canvas.manager.set_window_title('Static Input Fields (VTK/Synthetic)')
    axes = axes.flatten()

    mid_x = params.Nx // 2
    Y, Z = np.meshgrid(params.y_grid, params.z_grid, indexing='ij')

    # Plot Background Potential
    pcm0 = axes[0].pcolormesh(Y, Z, V_bg[mid_x, :, :], shading='gouraud', cmap='viridis')
    axes[0].set_title("Background Potential ($V_{bg}$)")
    axes[0].set_xlabel("Y [m]")
    axes[0].set_ylabel("Z [m]")
    fig.colorbar(pcm0, ax=axes[0], label="Potential [V]")
    
    # Overlay calculation grid lines
    axes[0].set_xticks(params.y_grid, minor=True)
    axes[0].set_yticks(params.z_grid, minor=True)
    axes[0].grid(which='minor', color='white', linestyle='-', linewidth=0.3, alpha=0.4)

    # Plot Weighting Fields
    for i in range(num_ant):
        ax_idx = i + 1
        pcm = axes[ax_idx].pcolormesh(Y, Z, Vw_grids[i][mid_x, :, :], shading='gouraud', cmap='plasma')
        axes[ax_idx].set_title(f"Antenna {i+1} Weighting Field ($V_w$)")
        axes[ax_idx].set_xlabel("Y [m]")
        axes[ax_idx].set_ylabel("Z [m]")
        fig.colorbar(pcm, ax=axes[ax_idx], label="Sensitivity [0-1]")
        
        # Overlay calculation grid lines
        axes[ax_idx].set_xticks(params.y_grid, minor=True)
        axes[ax_idx].set_yticks(params.z_grid, minor=True)
        axes[ax_idx].grid(which='minor', color='white', linestyle='-', linewidth=0.3, alpha=0.4)

    # Hide unused axes
    for j in range(n_plots, len(axes)):
        axes[j].axis('off')

    fig.tight_layout()


def plot_simulation_results_3d(results: Dict, params: SimulationParams3D, plot_cfg: PlottingConfig3D, 
                               V_bg: np.ndarray = None, Vw_grids: list = None,
                               spacecraft_mask: np.ndarray = None, antenna_masks: list = None):
    print("=====================================================")
    print("Generating 3D Visualizations and preparing outputs...")
    print("=====================================================")

    anims = []

    # Plot static fields if provided
    if V_bg is not None and Vw_grids is not None:
        _plot_static_vtk_fields(params, V_bg, Vw_grids)

    _plot_currents_and_voltage(results, params, plot_cfg)
    _animate_yz_fields_slice(results['history'], params, plot_cfg, anims, 
                             spacecraft_mask=spacecraft_mask, antenna_masks=antenna_masks)
    _animate_yz_particles_slice(results['history'], params, plot_cfg, anims, 
                                spacecraft_mask=spacecraft_mask, antenna_masks=antenna_masks)
    _animate_3d_particles(results['history'], params, plot_cfg, anims, 
                          spacecraft_mask=spacecraft_mask, antenna_masks=antenna_masks)
    _animate_velocity_distribution(results['history'], plot_cfg, anims)
    _export_csv(results, params, plot_cfg)

    print("=====================================================")
    if anims: print(f"Showing interactive windows ({len(anims) + (1 if V_bg is not None else 0)}).")
    plt.show()