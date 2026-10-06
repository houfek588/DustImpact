# -*- coding: utf-8 -*-
"""
Distribution visualization module: velocity magnitude histograms and kinetic spectra.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from dust_impact.visualization.styles import safe_set_window_title


def get_v_mag_3d(hist: Dict[str, Any], species: str, i: int) -> np.ndarray:
    """Extract magnitude of 3D velocity |v| for active particles at step i."""
    vx = hist[f'vx_{species}'][i]
    vy = hist[f'vy_{species}'][i]
    vz = hist[f'vz_{species}'][i]
    mask = ~np.isnan(vx)
    return np.sqrt(vx[mask] ** 2 + vy[mask] ** 2 + vz[mask] ** 2)


def get_v_signed(hist: Dict[str, Any], species: str, i: int) -> np.ndarray:
    """Extract signed velocity component v_x for active particles at step i."""
    v_x_curr = hist[f'vx_{species}'][i]
    return v_x_curr[~np.isnan(v_x_curr)]


def animate_velocity_distribution(
    hist: Dict[str, Any],
    plot_cfg: Any,
    anims: Optional[List[Any]] = None
) -> Tuple[Optional[plt.Figure], Optional[animation.FuncAnimation]]:
    """
    Generate animated speed distribution histograms for electron and ion populations.

    Parameters
    ----------
    hist : Dict[str, Any]
        History dictionary containing velocity components 'vx_e', 'vy_e', etc. and time 't'.
    plot_cfg : Any
        Plotting configuration specifying show_velocity_anim, save_plots_to_disk, file_velocity_anim.
    anims : Optional[List[Any]]
        Optional list to append animation reference to.

    Returns
    -------
    Tuple[Optional[plt.Figure], Optional[animation.FuncAnimation]]
        (fig, anim) or (None, None) if disabled.
    """
    if not getattr(plot_cfg, 'show_velocity_anim', True):
        return None, None

    if 't' not in hist or len(hist['t']) == 0:
        return None, None

    fig4, ax_v = plt.subplots(figsize=(10, 4))
    safe_set_window_title(fig4, 'Thermodynamics: Total 3D Velocity')

    line_ve, = ax_v.plot([], [], lw=2, color='red', drawstyle='steps-mid', label='Electrons')
    line_vi, = ax_v.plot([], [], lw=2, color='blue', drawstyle='steps-mid', label='Ions')
    ax_v.set_xlabel("Velocity Magnitude |v| [m/s]")
    ax_v.set_ylabel("Particle Count")
    ax_v.grid(True, ls=':')
    ax_v.legend()

    v_max = 0.0
    for sp in ['e', 'i']:
        for i in range(len(hist['t'])):
            v = get_v_mag_3d(hist, sp, i)
            if len(v) > 0:
                v_max = max(v_max, float(np.max(v)))

    pad = v_max * 0.05 if v_max != 0 else 1e3
    ax_v.set_xlim(0, v_max + pad)

    def update(i):
        ve, vi = get_v_mag_3d(hist, 'e', i), get_v_mag_3d(hist, 'i', i)
        my = 10
        if len(ve) > 1:
            ce, be = np.histogram(ve, bins=50, range=(0, v_max + pad))
            line_ve.set_data((be[:-1] + be[1:]) / 2, ce)
            my = max(my, int(ce.max()))
        if len(vi) > 1:
            ci, bi = np.histogram(vi, bins=50, range=(0, v_max + pad))
            line_vi.set_data((bi[:-1] + bi[1:]) / 2, ci)
            my = max(my, int(ci.max()))
        ax_v.set_ylim(0, my * 1.1)
        return line_ve, line_vi

    anim = animation.FuncAnimation(fig4, update, frames=len(hist['t']), interval=100, blit=False)
    if anims is not None:
        anims.append(anim)

    should_save = getattr(plot_cfg, 'save_plots_to_disk', getattr(plot_cfg, 'save_plots', False))
    if should_save and hasattr(plot_cfg, 'file_velocity_anim'):
        anim.save(plot_cfg.file_velocity_anim, writer='pillow', fps=15)

    return fig4, anim
