# -*- coding: utf-8 -*-
"""
Signal visualization module: electrode currents and voltages computed by the simulation, and the
derived monopole (V_antenna - V_spacecraft) and dipole (V_i - V_j) signals measured by antenna
instruments.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from dust_impact.visualization.styles import safe_set_window_title, get_antenna_colors

SPACECRAFT_COLOR = 'black'
# Alternating line styles keep identical signals (e.g. symmetric antennas) visible on top of each other
LINE_STYLES = ['-', '--', '-.', ':']


def _metadata(results: Dict[str, Any]) -> Dict[str, Any]:
    meta = results.get('metadata', {})
    return meta if isinstance(meta, dict) else {}


def _voltage_time_us(params: Any, n_steps: int) -> np.ndarray:
    """Times of the stored voltages [µs]: voltage[k] is the value at the end of step k, t = (k + 1) dt."""
    return (np.arange(n_steps) + 1) * params.dt * 1e6


def _current_time_us(params: Any, n_steps: int) -> np.ndarray:
    """Times of the stored step currents [µs]: current[k] is the average over step k, plotted at t = k dt."""
    return np.arange(n_steps) * params.dt * 1e6


def _spacecraft_voltage(results: Dict[str, Any], params: Any, n_steps: int) -> Tuple[np.ndarray, bool]:
    """
    Spacecraft potential history [V] and whether it floats.

    A floating spacecraft has its own voltage history ('voltage_sc'); a fixed spacecraft stays at its
    bias potential (metadata 'spacecraft_bias_voltage_V', else params.V_sc_bias, else 0 V).
    """
    v_sc = results.get('voltage_sc')
    if v_sc is not None and np.size(v_sc) == n_steps:
        return np.asarray(v_sc, dtype=float), True
    v_fixed = _metadata(results).get('spacecraft_bias_voltage_V', getattr(params, 'V_sc_bias', 0.0))
    return np.full(n_steps, float(v_fixed if v_fixed is not None else 0.0)), False


def _circuit_time_constants(results: Dict[str, Any], params: Any, n_ant: int) -> List[Optional[float]]:
    """
    Discharge time constants R * C_tot [s] of the electrodes (antennas, then a floating spacecraft).

    C_tot is the diagonal of the circuit capacitance matrix stored in the metadata (vacuum plus readout
    capacitance for a floating spacecraft); older results fall back to the configured antenna capacitance.
    """
    meta = _metadata(results)
    c_matrix = meta.get('circuit_capacitance_matrix_F')
    c_diag = list(np.diag(np.asarray(c_matrix, dtype=float))) if c_matrix is not None else list(params.C_ant)
    r_bias = list(params.R_ant[:n_ant])
    if len(c_diag) > n_ant:
        r_sc = meta.get('spacecraft_resistance_Ohm', getattr(params, 'R_sc', None))
        r_bias.append(r_sc)
    return [
        (r * c) if (r is not None and i < len(c_diag)) else None
        for i, (r, c) in enumerate(zip(r_bias, c_diag))
    ]


def _smooth(signal: np.ndarray, k_size: int = 100) -> np.ndarray:
    """Boxcar smoothing with the kernel used for the antenna currents."""
    k_size = min(k_size, max(1, len(signal)))
    return np.convolve(signal, np.ones(k_size) / k_size, mode='same')


def _tau_label(tau: Optional[float]) -> str:
    return f' ($\\tau$={tau * 1e6:.1f} µs)' if tau is not None else ' (bez svodu)'


def _save(fig: plt.Figure, plot_cfg: Any, attr: str) -> None:
    should_save = getattr(plot_cfg, 'save_plots_to_disk', getattr(plot_cfg, 'save_plots', False))
    if should_save and getattr(plot_cfg, attr, None):
        fig.savefig(getattr(plot_cfg, attr), dpi=300, bbox_inches='tight')


def plot_currents_and_voltage(
    results: Dict[str, Any],
    params: Any,
    plot_cfg: Any
) -> Optional[plt.Figure]:
    """
    Plot the quantities computed directly by the simulation: electrode currents (total and induced)
    and electrode voltages relative to the grounded boundary.

    The antennas are shown in colour, a floating spacecraft (results 'voltage_sc', 'current_sc') in black.

    Parameters
    ----------
    results : Dict[str, Any]
        Dictionary containing 'smooth_total', 'smooth_induced', 'voltage_ant' and, for a floating
        spacecraft, 'voltage_sc' and 'current_sc'.
    params : Any
        Simulation parameters providing time_array, dt, C_ant, R_ant and t_delay.
    plot_cfg : Any
        Plotting configuration specifying show_currents, save_plots_to_disk, and file_currents.

    Returns
    -------
    Optional[plt.Figure]
        The created Matplotlib figure, or None if plotting was skipped.
    """
    if not getattr(plot_cfg, 'show_currents', True):
        return None

    n_ant = len(getattr(params, 'C_ant', []))
    v_sc = results.get('voltage_sc')
    if n_ant == 0 and v_sc is None:
        return None

    fig, (ax1a, ax1b) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    safe_set_window_title(fig, '3D Signals: Currents and Voltages')

    colors = get_antenna_colors()
    taus = _circuit_time_constants(results, params, n_ant)

    for a_idx in range(n_ant):
        c = colors[a_idx % len(colors)]
        tot_nA = results['smooth_total'][a_idx] * 1e9
        ind_nA = results['smooth_induced'][a_idx] * 1e9
        v_mV = np.asarray(results['voltage_ant'][a_idx]) * 1e3

        ax1a.plot(_current_time_us(params, len(tot_nA)), tot_nA, color=c, lw=2, ls=LINE_STYLES[a_idx % len(LINE_STYLES)],
                  label=f'Celkový (anténa {a_idx + 1})')
        ax1a.plot(_current_time_us(params, len(ind_nA)), ind_nA, color=c, lw=1, alpha=0.5,
                  ls=LINE_STYLES[a_idx % len(LINE_STYLES)])
        ax1b.plot(_voltage_time_us(params, len(v_mV)), v_mV, color=c, lw=2.5, ls=LINE_STYLES[a_idx % len(LINE_STYLES)],
                  label=f'Anténa {a_idx + 1}' + _tau_label(taus[a_idx] if a_idx < len(taus) else None))

    if v_sc is not None:
        i_sc = results.get('current_sc')
        if i_sc is not None:
            ax1a.plot(_current_time_us(params, len(i_sc)), _smooth(np.asarray(i_sc, dtype=float)) * 1e9,
                      color=SPACECRAFT_COLOR, lw=2, label='Celkový (sonda)')
        tau_sc = taus[n_ant] if len(taus) > n_ant else None
        v_sc = np.asarray(v_sc, dtype=float)
        ax1b.plot(_voltage_time_us(params, len(v_sc)), v_sc * 1e3, color=SPACECRAFT_COLOR, lw=2.5,
                  label='Sonda' + _tau_label(tau_sc))

    t_delay_us = getattr(params, 't_delay', 1e-6) * 1e6
    for ax in (ax1a, ax1b):
        ax.axvline(x=t_delay_us, color='grey', ls='--', alpha=0.7)
        ax.grid(True, ls=':')
        ax.legend()

    ax1a.set_title('Proudy do elektrod (silně: celkový, slabě: indukovaný)')
    ax1b.set_title('Napětí elektrod vůči uzemněné hranici')
    ax1a.set_ylabel("Proud [nA]")
    ax1b.set_ylabel("Napětí [mV]")
    ax1b.set_xlabel("Čas [µs]")
    fig.tight_layout()
    _save(fig, plot_cfg, 'file_currents')
    return fig


def resolve_dipole_pairs(pairs: Optional[List[List[int]]], n_ant: int) -> List[Tuple[int, int]]:
    """
    Dipole antenna pairs as 0-based index tuples.

    Parameters
    ----------
    pairs : Optional[List[List[int]]]
        Configured 1-based pairs [i, j]; empty or None selects consecutive pairs (1, 2), (3, 4), ...
    n_ant : int
        Number of antennas.

    Returns
    -------
    List[Tuple[int, int]]
        Valid pairs (i, j) with i != j, both below n_ant.

    Raises
    ------
    ValueError
        If a configured pair refers to a missing antenna or repeats an antenna.
    """
    if not pairs:
        return [(i, i + 1) for i in range(0, n_ant - 1, 2)]
    resolved = []
    for p in pairs:
        i, j = int(p[0]) - 1, int(p[1]) - 1
        if not (0 <= i < n_ant and 0 <= j < n_ant) or i == j:
            raise ValueError(f"Neplatný dipól {list(p)}: simulace má {n_ant} antén (indexy od 1).")
        resolved.append((i, j))
    return resolved


def compute_monopole_dipole(
    results: Dict[str, Any], params: Any, pairs: Optional[List[List[int]]] = None
) -> Dict[str, Any]:
    """
    Derived instrument signals from the electrode voltages.

    Monopole k: V_k - V_sc (antenna relative to the spacecraft body); dipole (i, j): V_i - V_j.

    Parameters
    ----------
    results : Dict[str, Any]
        Results with 'voltage_ant' (num_antennas, steps) and optionally 'voltage_sc' (steps).
    params : Any
        Simulation parameters (dt, V_sc_bias).
    pairs : Optional[List[List[int]]]
        1-based dipole pairs (see resolve_dipole_pairs).

    Returns
    -------
    Dict[str, Any]
        't_us' [µs], 'monopoles' (num_antennas, steps) [V], 'dipoles' (num_pairs, steps) [V],
        'pairs' (0-based), 'floating_spacecraft' (bool).
    """
    V_ant = np.atleast_2d(np.asarray(results.get('voltage_ant', np.zeros((0, 0))), dtype=float))
    n_ant, n_steps = V_ant.shape if V_ant.size else (0, 0)
    V_sc, floating = _spacecraft_voltage(results, params, n_steps)
    resolved = resolve_dipole_pairs(pairs, n_ant)
    return {
        't_us': _voltage_time_us(params, n_steps),
        'monopoles': V_ant - V_sc[None, :] if n_ant else np.zeros((0, n_steps)),
        'dipoles': np.array([V_ant[i] - V_ant[j] for i, j in resolved]).reshape(len(resolved), n_steps),
        'pairs': resolved,
        'floating_spacecraft': floating,
    }


def plot_monopole_dipole(
    results: Dict[str, Any],
    params: Any,
    plot_cfg: Any
) -> Optional[plt.Figure]:
    """
    Plot the monopole signals V_k - V_sc of all antennas and the dipole signals V_i - V_j.

    These are the voltages measured by antenna instruments (e.g. Shen et al. 2023: monopoles
    V_ANT,1 - V_SC, V_ANT,2 - V_SC and the dipole V_ANT,3 - V_ANT,4). For a fixed spacecraft the
    monopole is taken relative to its constant bias potential.

    Parameters
    ----------
    results : Dict[str, Any]
        Results with 'voltage_ant' and optionally 'voltage_sc'.
    params : Any
        Simulation parameters (dt, t_delay, V_sc_bias).
    plot_cfg : Any
        Plotting configuration: show_monopole_dipole, dipole_pairs, save_plots_to_disk,
        file_monopole_dipole.

    Returns
    -------
    Optional[plt.Figure]
        The created figure, or None if disabled or there are no antennas.
    """
    if not getattr(plot_cfg, 'show_monopole_dipole', True):
        return None
    signals = compute_monopole_dipole(results, params, getattr(plot_cfg, 'dipole_pairs', None))
    n_ant = signals['monopoles'].shape[0]
    if n_ant == 0:
        return None

    has_dipoles = len(signals['pairs']) > 0
    fig, axes = plt.subplots(2 if has_dipoles else 1, 1, figsize=(10, 8 if has_dipoles else 4.5), sharex=True)
    axes = np.atleast_1d(axes)
    safe_set_window_title(fig, '3D Signals: Monopoles and Dipoles')
    colors = get_antenna_colors()
    t_us = signals['t_us']

    ax_m = axes[0]
    for k in range(n_ant):
        ax_m.plot(t_us, signals['monopoles'][k] * 1e3, color=colors[k % len(colors)], lw=2,
                  ls=LINE_STYLES[k % len(LINE_STYLES)],
                  label=f'$V_{{\\mathrm{{ANT}},{k + 1}}} - V_{{\\mathrm{{SC}}}}$')
    sc_desc = 'plovoucí sonda' if signals['floating_spacecraft'] else 'sonda na pevném potenciálu'
    ax_m.set_title(f'Monopóly (anténa vůči tělesu sondy; {sc_desc})')
    ax_m.set_ylabel('Napětí [mV]')

    if has_dipoles:
        ax_d = axes[1]
        for d_idx, (i, j) in enumerate(signals['pairs']):
            ax_d.plot(t_us, signals['dipoles'][d_idx] * 1e3, color=colors[(n_ant + d_idx) % len(colors)], lw=2,
                      ls=LINE_STYLES[d_idx % len(LINE_STYLES)],
                      label=f'$V_{{\\mathrm{{ANT}},{i + 1}}} - V_{{\\mathrm{{ANT}},{j + 1}}}$')
        ax_d.set_title('Dipóly (rozdíl napětí dvou antén)')
        ax_d.set_ylabel('Napětí [mV]')

    t_delay_us = getattr(params, 't_delay', 1e-6) * 1e6
    for ax in axes:
        ax.axvline(x=t_delay_us, color='grey', ls='--', alpha=0.7)
        ax.axhline(0.0, color='grey', lw=0.8, alpha=0.6)
        ax.grid(True, ls=':')
        ax.legend()
    axes[-1].set_xlabel('Čas [µs]')
    fig.tight_layout()
    _save(fig, plot_cfg, 'file_monopole_dipole')
    return fig
