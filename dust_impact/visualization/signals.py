# -*- coding: utf-8 -*-
"""
Signal visualization module: antenna currents, voltages, and time-series.
"""

from typing import Dict, Any, Optional
import matplotlib.pyplot as plt
from dust_impact.visualization.styles import safe_set_window_title, get_antenna_colors


def plot_currents_and_voltage(
    results: Dict[str, Any],
    params: Any,
    plot_cfg: Any
) -> Optional[plt.Figure]:
    """
    Plot antenna currents (total and induced) and RC voltages over time.

    Parameters
    ----------
    results : Dict[str, Any]
        Dictionary containing 'smooth_total', 'smooth_induced', and 'voltage_ant'.
    params : Any
        Simulation parameters providing time_array, C_ant, R_ant, and t_delay.
    plot_cfg : Any
        Plotting configuration specifying show_currents, save_plots_to_disk, and file_currents.

    Returns
    -------
    Optional[plt.Figure]
        The created Matplotlib figure, or None if plotting was skipped.
    """
    if not getattr(plot_cfg, 'show_currents', True):
        return None

    c_ant = getattr(params, 'C_ant', [])
    if len(c_ant) == 0:
        return None

    fig, (ax1a, ax1b) = plt.subplots(2, 1, figsize=(10, 8))
    safe_set_window_title(fig, '3D Signals: Currents and Voltages')

    colors = get_antenna_colors()
    time_us = params.time_array * 1e6

    for a_idx in range(len(c_ant)):
        c = colors[a_idx % len(colors)]
        tot_nA = results['smooth_total'][a_idx] * 1e9
        ind_nA = results['smooth_induced'][a_idx] * 1e9
        v_mV = results['voltage_ant'][a_idx] * 1e3
        tau_us = params.R_ant[a_idx] * params.C_ant[a_idx] * 1e6

        ax1a.plot(time_us, tot_nA, color=c, lw=2, label=f'Total (Antenna {a_idx + 1})')
        ax1a.plot(time_us, ind_nA, color=c, ls=':', alpha=0.5)
        ax1b.plot(time_us, v_mV, color=c, lw=2.5, label=f'Antenna {a_idx + 1} ($\\tau$={tau_us:.1f} µs)')

    t_delay_us = getattr(params, 't_delay', 1e-6) * 1e6
    for ax in (ax1a, ax1b):
        ax.axvline(x=t_delay_us, color='grey', ls='--', alpha=0.7)
        ax.grid(True, ls=':')
        ax.legend()

    ax1a.set_ylabel("Current [nA]")
    ax1b.set_ylabel("Voltage [mV]")
    ax1b.set_xlabel("Time [µs]")
    fig.tight_layout()

    should_save = getattr(plot_cfg, 'save_plots_to_disk', getattr(plot_cfg, 'save_plots', False))
    if should_save and hasattr(plot_cfg, 'file_currents'):
        fig.savefig(plot_cfg.file_currents, dpi=300, bbox_inches='tight')

    return fig
