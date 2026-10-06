# -*- coding: utf-8 -*-
"""
Visualization styling, color palettes, and Matplotlib canvas utilities.
"""

from typing import List, Dict, Any
import matplotlib.pyplot as plt


def safe_set_window_title(fig: Any, title: str) -> None:
    """Safely set figure window title without crashing on headless / non-GUI backends."""
    if hasattr(fig, 'canvas') and hasattr(fig.canvas, 'manager') and fig.canvas.manager is not None:
        if hasattr(fig.canvas.manager, 'set_window_title'):
            try:
                fig.canvas.manager.set_window_title(title)
            except Exception:
                pass


def get_antenna_colors() -> List[str]:
    """Default distinct palette for antennas."""
    return ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b', '#e377c2']


def get_species_colors() -> Dict[str, str]:
    """Default color mappings for particle species."""
    return {
        'e': '#00d2ff',  # Electrons (e⁻)
        'i': '#ff4500',  # Ions (i⁺)
        'sc': '#7f7f7f', # Spacecraft structure
        'impact': '#ffd700', # Impact point
    }


def cleanup_animation(anim: Any) -> None:
    """Safely mark animation as handled to prevent Matplotlib's unrendered warning on headless close."""
    if anim is not None:
        try:
            anim._draw_was_started = True
            if hasattr(anim, 'event_source') and anim.event_source is not None:
                anim.event_source.stop()
        except Exception:
            pass

