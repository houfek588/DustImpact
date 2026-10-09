# -*- coding: utf-8 -*-
"""
Particle visualization module: 2D projection slices and 3D kinetic cloud animations.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from dust_impact.visualization.styles import safe_set_window_title, get_antenna_colors, get_species_colors


def animate_particles_2d_slice(
    hist: Dict[str, Any],
    params: Any,
    plot_cfg: Any,
    anims: Optional[List[Any]] = None,
    spacecraft_mask: Optional[np.ndarray] = None,
    antenna_masks: Optional[List[np.ndarray]] = None
) -> Tuple[Optional[plt.Figure], Optional[animation.FuncAnimation]]:
    """
    Generate 2D YZ projection slice of expanding plasma particles and conductor boundaries.

    Parameters
    ----------
    hist : Dict[str, Any]
        History dictionary containing particle coordinates (x_e, y_e, z_e, etc.) and time 't'.
    params : Any
        Simulation parameters providing domain half lengths and grid nodes.
    plot_cfg : Any
        Plotting configuration specifying show_particles_3d, save_plots_to_disk, file_particles_anim.
    anims : Optional[List[Any]]
        Optional list to append animation reference to.
    spacecraft_mask : Optional[np.ndarray]
        3D mask of spacecraft.
    antenna_masks : Optional[List[np.ndarray]]
        List of 3D antenna masks.

    Returns
    -------
    Tuple[Optional[plt.Figure], Optional[animation.FuncAnimation]]
        (fig, anim) or (None, None) if disabled.
    """
    if not getattr(plot_cfg, 'show_particles_3d', True):
        return None, None

    if 't' not in hist or len(hist['t']) == 0:
        return None, None

    fig_slice, ax_pos = plt.subplots(figsize=(8, 6))
    safe_set_window_title(fig_slice, '3D Particles: 2D YZ Slice (x >= 0)')

    mid_x = params.Nx // 2
    Y_mesh, Z_mesh = np.meshgrid(params.y_grid, params.z_grid, indexing='ij')

    # Semi-transparent filled geometry contour + crisp boundary outline for Spacecraft
    if spacecraft_mask is not None:
        sc_slice = spacecraft_mask[mid_x, :, :]
        if np.any(sc_slice):
            ax_pos.contourf(Y_mesh, Z_mesh, sc_slice.astype(float), levels=[0.5, 1.0], colors=['#7f7f7f'], alpha=0.35)
            ax_pos.contour(Y_mesh, Z_mesh, sc_slice.astype(float), levels=[0.5], colors=['#333333'], linewidths=2.0)

    # Antenna geometry contours
    if antenna_masks is not None:
        colors = get_antenna_colors()
        for i, m in enumerate(antenna_masks):
            ant_slice = m[mid_x, :, :]
            if np.any(ant_slice):
                c = colors[i % len(colors)]
                ax_pos.contourf(Y_mesh, Z_mesh, ant_slice.astype(float), levels=[0.5, 1.0], colors=[c], alpha=0.4)
                ax_pos.contour(Y_mesh, Z_mesh, ant_slice.astype(float), levels=[0.5], colors=[c], linewidths=1.5)

    sp_colors = get_species_colors()
    scat_e = ax_pos.scatter([], [], s=12, color=sp_colors['e'], alpha=0.75, label='Electrons (e⁻)')
    scat_i = ax_pos.scatter([], [], s=18, color=sp_colors['i'], alpha=0.85, label='Ions (i⁺)')

    ax_pos.set_xlim(-params.L_y, params.L_y)
    ax_pos.set_ylim(-params.L_z, params.L_z)
    ax_pos.set_xlabel("Y [m]")
    ax_pos.set_ylabel("Z [m]")
    ax_pos.set_title("Expanding Plasma Cloud & Spacecraft Geometry (YZ Projection)")
    ax_pos.legend(loc='upper right', framealpha=0.9)
    ax_pos.grid(True, linestyle=':', alpha=0.6)

    time_text = ax_pos.text(0.02, 0.90, '', transform=ax_pos.transAxes, bbox=dict(facecolor='white', alpha=0.85, edgecolor='gray'))

    def update(i):
        me_slice = ~np.isnan(hist['x_e'][i])
        mi_slice = ~np.isnan(hist['x_i'][i])

        data_e = np.column_stack((hist['y_e'][i][me_slice], hist['z_e'][i][me_slice]))
        data_i = np.column_stack((hist['y_i'][i][mi_slice], hist['z_i'][i][mi_slice]))

        if len(data_e) > 0:
            scat_e.set_offsets(data_e)
        else:
            scat_e.set_offsets(np.empty((0, 2)))

        if len(data_i) > 0:
            scat_i.set_offsets(data_i)
        else:
            scat_i.set_offsets(np.empty((0, 2)))

        time_text.set_text(f"Time: {hist['t'][i] * 1e6:.2f} µs")
        return scat_e, scat_i, time_text

    anim = animation.FuncAnimation(fig_slice, update, frames=len(hist['t']), interval=100, blit=False)
    if anims is not None:
        anims.append(anim)

    should_save = getattr(plot_cfg, 'save_plots_to_disk', getattr(plot_cfg, 'save_plots', False))
    if should_save and hasattr(plot_cfg, 'file_particles_anim'):
        target_file = plot_cfg.file_particles_anim.replace('.gif', '_yz.gif')
        anim.save(target_file, writer='pillow', fps=15)

    return fig_slice, anim


def animate_particles_3d(
    hist: Dict[str, Any],
    params: Any,
    plot_cfg: Any,
    anims: Optional[List[Any]] = None,
    spacecraft_mask: Optional[np.ndarray] = None,
    antenna_masks: Optional[List[np.ndarray]] = None
) -> Tuple[Optional[plt.Figure], Optional[animation.FuncAnimation]]:
    """
    Generate full 3D animated particle kinetics cloud with conductor boundaries and impact point.

    Parameters
    ----------
    hist : Dict[str, Any]
        History dictionary containing particle coordinate arrays.
    params : Any
        Simulation parameters providing domain dimensions and impact location.
    plot_cfg : Any
        Plotting configuration specifying show_particles_3d, save_plots_to_disk, file_particles_anim.
    anims : Optional[List[Any]]
        Optional list to append animation reference to.
    spacecraft_mask : Optional[np.ndarray]
        3D mask of spacecraft.
    antenna_masks : Optional[List[np.ndarray]]
        List of 3D antenna masks.

    Returns
    -------
    Tuple[Optional[plt.Figure], Optional[animation.FuncAnimation]]
        (fig, anim) or (None, None) if disabled.
    """
    if not getattr(plot_cfg, 'show_particles_3d', True):
        return None, None

    if 't' not in hist or len(hist['t']) == 0:
        return None, None

    fig3 = plt.figure(figsize=(10, 8))
    ax = fig3.add_subplot(111, projection='3d')
    safe_set_window_title(fig3, '3D Particle Kinetics and Geometry')

    X, Y, Z = np.meshgrid(params.x_grid, params.y_grid, params.z_grid, indexing='ij')

    # Render spacecraft body as semi-transparent 3D wireframe / shell
    if spacecraft_mask is not None:
        sc_x = X[spacecraft_mask]
        sc_y = Y[spacecraft_mask]
        sc_z = Z[spacecraft_mask]
        if len(sc_x) > 0:
            ax.scatter(sc_x, sc_y, sc_z, color='#000000', alpha=0.12, s=8, marker='s', label='Spacecraft Structure')

    # Render antennas as distinct colored wireframes
    if antenna_masks is not None:
        colors = get_antenna_colors()
        for i, m in enumerate(antenna_masks):
            c = colors[i % len(colors)]
            ant_x = X[m]
            ant_y = Y[m]
            ant_z = Z[m]
            if len(ant_x) > 0:
                ax.scatter(ant_x, ant_y, ant_z, color=c, alpha=0.45, s=6, marker='o', label=f'Antenna {i+1}')

    sp_colors = get_species_colors()
    scat_e = ax.scatter([], [], [], s=12, color=sp_colors['e'], alpha=0.7, label='Electrons (e⁻)')
    scat_i = ax.scatter([], [], [], s=18, color=sp_colors['i'], alpha=0.85, label='Ions (i⁺)')

    ax.set_xlim(-params.L_x, params.L_x)
    ax.set_ylim(-params.L_y, params.L_y)
    ax.set_zlim(-params.L_z, params.L_z)

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.legend(loc='upper right', framealpha=0.9)

    impact_pos = getattr(params, 'impact_pos', None)
    if impact_pos is not None:
        ax.scatter(*impact_pos, color=sp_colors['impact'], s=120, marker='*', edgecolor='black', linewidth=0.5, label='Impact Point')
    time_text = ax.text2D(0.02, 0.95, '', transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.85, edgecolor='gray'))

    def update(i):
        me, mi = ~np.isnan(hist['x_e'][i]), ~np.isnan(hist['x_i'][i])
        scat_e._offsets3d = (hist['x_e'][i][me], hist['y_e'][i][me], hist['z_e'][i][me])
        scat_i._offsets3d = (hist['x_i'][i][mi], hist['y_i'][i][mi], hist['z_i'][i][mi])
        time_text.set_text(f"Time: {hist['t'][i] * 1e6:.2f} µs")
        return scat_e, scat_i, time_text

    anim = animation.FuncAnimation(fig3, update, frames=len(hist['t']), interval=100, blit=False)
    if anims is not None:
        anims.append(anim)

    should_save = getattr(plot_cfg, 'save_plots_to_disk', getattr(plot_cfg, 'save_plots', False))
    if should_save and hasattr(plot_cfg, 'file_particles_anim'):
        anim.save(plot_cfg.file_particles_anim, writer='pillow', fps=15)

    return fig3, anim
