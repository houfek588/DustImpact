# -*- coding: utf-8 -*-
"""
Field visualization module: 2D static slices and time-dependent animations of 3D scalar fields.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from dust_impact.visualization.styles import safe_set_window_title


def plot_static_vtk_fields(
    params: Any,
    V_bg: np.ndarray,
    Vw_grids: List[np.ndarray],
    plot_cfg: Optional[Any] = None
) -> Optional[plt.Figure]:
    """
    Plot static 2D slices of input electrostatic fields (V_bg and antenna weighting potentials Vw).

    Parameters
    ----------
    params : Any
        Simulation parameters providing grid axes (y_grid, z_grid, Nx).
    V_bg : np.ndarray
        3D array of background potential.
    Vw_grids : List[np.ndarray]
        List of 3D arrays of antenna weighting potentials.
    plot_cfg : Optional[Any]
        Optional plotting configuration.

    Returns
    -------
    Optional[plt.Figure]
        The created Matplotlib figure.
    """
    num_ant = len(Vw_grids)
    n_plots = 1 + num_ant
    cols = 2 if n_plots > 1 else 1
    rows = (n_plots + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols if cols > 1 else 7, 5 * rows))
    safe_set_window_title(fig, 'Static Input Fields (VTK/Synthetic)')
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.flatten()

    mid_x = params.Nx // 2
    Y, Z = np.meshgrid(params.y_grid, params.z_grid, indexing='ij')

    pcm0 = axes[0].pcolormesh(Y, Z, V_bg[mid_x, :, :], shading='gouraud', cmap='viridis')
    axes[0].set_title("Background Potential ($V_{bg}$)")
    axes[0].set_xlabel("Y [m]")
    axes[0].set_ylabel("Z [m]")
    fig.colorbar(pcm0, ax=axes[0], label="Potential [V]")

    axes[0].set_xticks(params.y_grid, minor=True)
    axes[0].set_yticks(params.z_grid, minor=True)
    axes[0].grid(which='minor', color='white', linestyle='-', linewidth=0.3, alpha=0.4)

    for i in range(num_ant):
        ax_idx = i + 1
        pcm = axes[ax_idx].pcolormesh(Y, Z, Vw_grids[i][mid_x, :, :], shading='gouraud', cmap='plasma')
        axes[ax_idx].set_title(f"Antenna {i+1} Weighting Field ($V_w$)")
        axes[ax_idx].set_xlabel("Y [m]")
        axes[ax_idx].set_ylabel("Z [m]")
        fig.colorbar(pcm, ax=axes[ax_idx], label="Sensitivity [0-1]")

        axes[ax_idx].set_xticks(params.y_grid, minor=True)
        axes[ax_idx].set_yticks(params.z_grid, minor=True)
        axes[ax_idx].grid(which='minor', color='white', linestyle='-', linewidth=0.3, alpha=0.4)

    for j in range(n_plots, len(axes)):
        axes[j].axis('off')

    fig.tight_layout()
    return fig


def animate_fields_slice(
    hist: Dict[str, Any],
    params: Any,
    plot_cfg: Any,
    anims: Optional[List[Any]] = None,
    plane: str = 'yz',
    spacecraft_mask: Optional[np.ndarray] = None,
    antenna_masks: Optional[List[np.ndarray]] = None
) -> Tuple[Optional[plt.Figure], Optional[animation.FuncAnimation]]:
    """
    Generate 2D animated slice of the 3D electrostatic potential field.

    Parameters
    ----------
    hist : Dict[str, Any]
        History dictionary containing list of 3D potential fields 'V' and time steps 't'.
    params : Any
        Simulation parameters providing grid vectors and dimensions.
    plot_cfg : Any
        Configuration dict/dataclass with show_fields_slice, save_plots_to_disk, file_fields_anim.
    anims : Optional[List[Any]]
        Optional list to append the resulting animation instance to (preserves reference).
    plane : str
        Slice plane ('yz' supported).
    spacecraft_mask : Optional[np.ndarray]
        3D boolean mask of spacecraft structure.
    antenna_masks : Optional[List[np.ndarray]]
        List of 3D boolean masks for antenna wires.

    Returns
    -------
    Tuple[Optional[plt.Figure], Optional[animation.FuncAnimation]]
        (fig, anim) or (None, None) if slice animation is disabled.
    """
    if not getattr(plot_cfg, 'show_fields_slice', True):
        return None, None

    if 'V' not in hist or len(hist['V']) == 0:
        return None, None

    fig2, ax_V = plt.subplots(figsize=(8, 6))
    safe_set_window_title(fig2, f'3D Fields: 2D {plane.upper()} Slice (x = 0)')

    mid_x = params.Nx // 2

    all_V_slices = [V[mid_x, :, :] for V in hist['V']]
    flat_V = np.concatenate([v.ravel() for v in all_V_slices])
    finite_V = flat_V[np.isfinite(flat_V)]

    if len(finite_V) > 0:
        V_min, V_max = np.min(finite_V), np.max(finite_V)
    else:
        V_min, V_max = -1.0, 1.0

    if V_max == V_min:
        V_max += 1.0
        V_min -= 1.0

    Y, Z = np.meshgrid(params.y_grid, params.z_grid, indexing='ij')

    V0_clean = np.nan_to_num(hist['V'][0][mid_x, :, :], nan=0.0)
    pcm = ax_V.pcolormesh(Y, Z, V0_clean, shading='auto', cmap='viridis', vmin=V_min, vmax=V_max)
    fig2.colorbar(pcm, ax=ax_V, label='Potential [V]')

    if spacecraft_mask is not None:
        sc_slice = spacecraft_mask[mid_x, :, :]
        if np.any(sc_slice):
            ax_V.contour(Y, Z, sc_slice.astype(float), levels=[0.5], colors='white', linewidths=2)

    if antenna_masks is not None:
        for m in antenna_masks:
            ant_slice = m[mid_x, :, :]
            if np.any(ant_slice):
                ax_V.contour(Y, Z, ant_slice.astype(float), levels=[0.5], colors='white', linewidths=1, linestyles='dashed')

    ax_V.set_xlabel("Y [m]")
    ax_V.set_ylabel("Z [m]")

    time_text = ax_V.text(0.02, 0.90, '', transform=ax_V.transAxes, color='white', weight='bold')

    def update(i):
        V_clean = np.nan_to_num(hist['V'][i][mid_x, :, :], nan=0.0)
        pcm.set_array(V_clean.ravel())
        time_text.set_text(f"Time: {hist['t'][i] * 1e6:.2f} µs")
        return pcm, time_text

    anim = animation.FuncAnimation(fig2, update, frames=len(hist['t']), interval=100, blit=False)
    if anims is not None:
        anims.append(anim)

    should_save = getattr(plot_cfg, 'save_plots_to_disk', getattr(plot_cfg, 'save_plots', False))
    if should_save and hasattr(plot_cfg, 'file_fields_anim'):
        anim.save(plot_cfg.file_fields_anim, writer='pillow', fps=15)

    return fig2, anim


def plot_potential_line_x(
    hist: Dict[str, Any],
    params: Any,
    plot_cfg: Optional[Any] = None,
    spacecraft_mask: Optional[np.ndarray] = None
) -> Optional[plt.Figure]:
    """
    Plot 1D electric potential profile along the X-axis (y=0, z=0) at 4 distinct time instances:
    t = 0, t = T/2, t = 3T/4, and t = T.

    Parameters
    ----------
    hist : Dict[str, Any]
        History dictionary containing list of 3D potential fields 'V' and time steps 't'.
    params : Any
        Simulation parameters providing grid vectors (x_grid, y_grid, z_grid) and dimensions.
    plot_cfg : Optional[Any]
        Optional plotting configuration with show_potential_line_x, save_plots_to_disk,
        and file_potential_line_x.
    spacecraft_mask : Optional[np.ndarray]
        3D boolean mask of spacecraft structure to highlight on the 1D profiles.

    Returns
    -------
    Optional[plt.Figure]
        The created Matplotlib figure with 4 subplots, or None if disabled or data is missing.
    """
    if plot_cfg is not None and not getattr(plot_cfg, 'show_potential_line_x', True):
        return None

    if 'V' not in hist or len(hist['V']) == 0:
        return None

    times = np.array(hist.get('t', []))
    if len(times) == 0:
        return None

    V_list = hist['V']
    if len(V_list) == 0:
        return None

    # Determine X-line indices for y = 0 and z = 0
    y_idx = int(np.argmin(np.abs(params.y_grid - 0.0)))
    z_idx = int(np.argmin(np.abs(params.z_grid - 0.0)))
    x_coords = params.x_grid

    # Total time duration T
    T = float(times[-1]) if float(times[-1]) > 0 else float(getattr(params, 'simulation_duration_s', 1.0))

    # Target times: t = 0, t = T/2, t = 3T/4, t = T
    time_targets = [
        (0.0, "t = 0"),
        (0.5 * T, "t = T/2"),
        (0.75 * T, "t = 3T/4"),
        (T, "t = T")
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True)
    safe_set_window_title(fig, 'Electric Potential along X-axis (y=0, z=0)')
    axes_flat = axes.flatten()

    # Identify spacecraft region along X-axis if mask is available
    sc_span = None
    if spacecraft_mask is not None:
        sc_line = spacecraft_mask[:, y_idx, z_idx]
        if np.any(sc_line):
            sc_x = x_coords[sc_line]
            sc_span = (float(np.min(sc_x)), float(np.max(sc_x)))

    # Theoretical Debye profile (if analytical spherical geometry & parameters available)
    V_debye_ref = None
    debye_lbl = None
    lambda_D = getattr(params, 'debye_length', None)
    V_sc = getattr(params, 'spacecraft_voltage_V', getattr(params, 'Vf', 0.0))
    if lambda_D is not None and lambda_D > 0 and V_sc is not None and abs(V_sc) > 0:
        geom_source = getattr(getattr(params, 'geometry', None), 'source', '')
        if geom_source == 'analytical':
            sc_part = getattr(getattr(params.geometry, 'analytical', None), 'spacecraft', None)
            if sc_part and getattr(sc_part, 'type', '') == 'sphere':
                R_sph = getattr(sc_part, 'radius', 0.3)
                r_arr = np.abs(x_coords)
                with np.errstate(divide='ignore', invalid='ignore'):
                    V_debye_ref = np.where(
                        r_arr <= R_sph,
                        V_sc,
                        V_sc * (R_sph / np.maximum(r_arr, 1e-6)) * np.exp(-(r_arr - R_sph) / lambda_D)
                    )
                debye_lbl = f'Debyeovo stínění ($\\lambda_D = {lambda_D:.2f}\\,\\mathrm{{m}}$)'

    for idx_plot, (t_tgt, label_txt) in enumerate(time_targets):
        ax = axes_flat[idx_plot]

        # Find closest recorded frame
        frame_idx = int(np.argmin(np.abs(times - t_tgt)))
        t_actual = float(times[frame_idx])
        V_frame = V_list[frame_idx]
        V_line = np.nan_to_num(V_frame[:, y_idx, z_idx], nan=0.0)

        # Plot numerical potential along X axis
        ax.plot(x_coords, V_line, color='#1f77b4', linewidth=2.0, label='Simulovaný potenciál $V(x)$')

        # Reference zero line
        ax.axhline(0.0, color='gray', linestyle=':', linewidth=0.8, alpha=0.7)

        # Theoretical Debye curve if available
        if V_debye_ref is not None:
            ax.plot(x_coords, V_debye_ref, color='#ff7f0e', linestyle='--', linewidth=1.5, alpha=0.85, label=debye_lbl)

        # Highlight spacecraft body on X axis
        if sc_span is not None:
            ax.axvspan(sc_span[0], sc_span[1], color='#d62728', alpha=0.15, label='Sonda' if idx_plot == 0 else None)

        # Format actual time
        if T < 1e-6:
            t_str = f"{t_actual * 1e9:.1f} ns"
        else:
            t_str = f"{t_actual * 1e6:.2f} µs"

        ax.set_title(f"{label_txt}  ($t = {t_str}$)", fontsize=11, weight='bold')
        ax.set_ylabel("Potenciál $V$ [V]", fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(loc='upper right', fontsize=8, framealpha=0.85)

        # Add x-axis label on bottom row
        if idx_plot >= 2:
            ax.set_xlabel("Poloha $x$ [m] (podél osy $y=0, z=0$)", fontsize=10)

    fig.suptitle('Elektrický potenciál podél osy X ($y=0, z=0$)', fontsize=13, weight='bold', y=0.98)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])

    should_save = getattr(plot_cfg, 'save_plots_to_disk', getattr(plot_cfg, 'save_plots', False))
    if should_save:
        out_file = getattr(plot_cfg, 'file_potential_line_x', 'outputs/out_3d_potencial_osa_x.png')
        try:
            fig.savefig(out_file, dpi=150, bbox_inches='tight')
            print(f"  [OK] Graf potenciálu podél osy X uložen: {out_file}")
        except Exception as save_err:
            print(f"  [VAROVÁNÍ] Uložení grafu potenciálu podél osy X selhalo: {save_err}")

    return fig
