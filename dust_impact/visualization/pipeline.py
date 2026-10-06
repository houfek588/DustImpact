# -*- coding: utf-8 -*-
"""
Visualization pipeline and orchestration coordinator for 3D PIC simulation results.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import matplotlib.pyplot as plt

from dust_impact.common.io import export_time_series_csv
from dust_impact.visualization.styles import cleanup_animation
from dust_impact.visualization.signals import plot_currents_and_voltage
from dust_impact.visualization.fields import plot_static_vtk_fields, animate_fields_slice
from dust_impact.visualization.particles import animate_particles_2d_slice, animate_particles_3d
from dust_impact.visualization.distributions import animate_velocity_distribution


class VisualizationPipeline:
    """
    Orchestrates the generation of static figures, 2D/3D animations, and data exports.
    Manages animation object lifecycles to prevent prematurely garbage collected animations.
    """

    def __init__(self):
        self.figures: List[plt.Figure] = []
        self.animations: List[Any] = []

    def run(
        self,
        results: Dict[str, Any],
        params: Any,
        plot_cfg: Any,
        V_bg: Optional[np.ndarray] = None,
        Vw_grids: Optional[List[np.ndarray]] = None,
        spacecraft_mask: Optional[np.ndarray] = None,
        antenna_masks: Optional[List[np.ndarray]] = None
    ) -> None:
        """
        Execute full visualization suite according to PlottingConfig3D flags.
        """
        print("=====================================================")
        print("Generating 3D Visualizations and preparing outputs...")
        print("=====================================================")

        data_ant_count = len(results.get('smooth_total', []))
        cfg_ant_count = len(getattr(params, 'C_ant', []))

        if data_ant_count != cfg_ant_count:
            print("\n[CHYBA] Nesoulad v počtu antén!")
            print(f"  -> Konfigurace očekává: {cfg_ant_count}")
            print(f"  -> Načtená data obsahují: {data_ant_count}")
            return

        self.figures.clear()
        self.animations.clear()

        # 1. Static input fields (VTK / synthetic background & weighting)
        if V_bg is not None and Vw_grids is not None:
            fig_vtk = plot_static_vtk_fields(params, V_bg, Vw_grids, plot_cfg)
            if fig_vtk:
                self.figures.append(fig_vtk)

        # 2. Time-series: Antenna currents and voltages
        fig_curr = plot_currents_and_voltage(results, params, plot_cfg)
        if fig_curr:
            self.figures.append(fig_curr)

        # 3. 2D YZ Slice animation of potential fields
        hist = results.get('history', {})
        fig_f, anim_f = animate_fields_slice(
            hist, params, plot_cfg, anims=self.animations,
            plane='yz', spacecraft_mask=spacecraft_mask, antenna_masks=antenna_masks
        )
        if fig_f:
            self.figures.append(fig_f)

        # 4. 2D YZ Slice animation of particle positions
        fig_p2d, anim_p2d = animate_particles_2d_slice(
            hist, params, plot_cfg, anims=self.animations,
            spacecraft_mask=spacecraft_mask, antenna_masks=antenna_masks
        )
        if fig_p2d:
            self.figures.append(fig_p2d)

        # 5. Full 3D kinetic cloud animation
        fig_p3d, anim_p3d = animate_particles_3d(
            hist, params, plot_cfg, anims=self.animations,
            spacecraft_mask=spacecraft_mask, antenna_masks=antenna_masks
        )
        if fig_p3d:
            self.figures.append(fig_p3d)

        # 6. Velocity magnitude distribution animation
        fig_v, anim_v = animate_velocity_distribution(hist, plot_cfg, anims=self.animations)
        if fig_v:
            self.figures.append(fig_v)

        # 7. CSV time series export
        should_export_csv = getattr(plot_cfg, 'export_csv_time_series', getattr(plot_cfg, 'export_data_csv', False))
        if should_export_csv:
            csv_path = getattr(plot_cfg, 'output_csv_filepath', getattr(plot_cfg, 'file_csv', 'outputs/out_3d_vysledky_simulace.csv'))
            try:
                export_time_series_csv(results, params, csv_path)
            except Exception as e:
                print(f"  [ERROR] CSV export failed: {e}")

        print("=====================================================")
        if getattr(plot_cfg, 'show_interactive_gui_windows', True):
            if self.animations:
                print(f"Showing interactive windows ({len(self.animations) + (1 if V_bg is not None else 0)}).")
            plt.show()
        else:
            for anim in self.animations:
                cleanup_animation(anim)
            plt.close('all')
            print("  [OK] Headless mode: 3D Plots saved to disk without interactive GUI windows.")


def plot_simulation_results_3d(
    results: Dict[str, Any],
    params: Any,
    plot_cfg: Any,
    V_bg: Optional[np.ndarray] = None,
    Vw_grids: Optional[List[np.ndarray]] = None,
    spacecraft_mask: Optional[np.ndarray] = None,
    antenna_masks: Optional[List[np.ndarray]] = None
) -> None:
    """
    Main visualization entry point: renders and saves all requested figures and animations.
    Compatible drop-in replacement for legacy dust_impact.solver.plotting.plot_simulation_results_3d.
    """
    pipeline = VisualizationPipeline()
    pipeline.run(
        results, params, plot_cfg,
        V_bg=V_bg, Vw_grids=Vw_grids,
        spacecraft_mask=spacecraft_mask,
        antenna_masks=antenna_masks
    )
