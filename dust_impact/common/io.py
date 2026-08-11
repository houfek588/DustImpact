# -*- coding: utf-8 -*-
"""
Unified I/O utilities for saving and loading PIC simulation results.
"""

import os
import numpy as np
from typing import Dict, Any


def ensure_dir(filepath: str) -> None:
    """Ensure directory containing filepath exists."""
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def save_results_npz(results: Dict[str, Any], filepath: str) -> None:
    """Compress and save simulation results dictionary to NPZ file."""
    ensure_dir(filepath)
    print(f"Ukládám výsledky do souboru: {filepath} ...")

    npz_dict = {}
    for key, value in results.items():
        if key == 'history' and isinstance(value, dict):
            for h_key, h_val in value.items():
                npz_dict[f'hist_{h_key}'] = np.array(h_val)
        else:
            npz_dict[key] = np.array(value)

    np.savez_compressed(filepath, **npz_dict)
    print("  [OK] Data úspěšně uložena do binárního archivu.")


def load_results_npz(filepath: str) -> Dict[str, Any]:
    """Load simulation results from NPZ archive into structured dict."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Soubor s výsledky nebyl nalezen: {filepath}")

    print(f"Načítám výsledky ze souboru: {filepath} ...")
    results: Dict[str, Any] = {'history': {}}

    with np.load(filepath, allow_pickle=True) as data:
        for key in data.files:
            if key.startswith('hist_'):
                h_key = key[5:]
                results['history'][h_key] = data[key]
            else:
                results[key] = data[key]

    print("  [OK] Data úspěšně načtena do operační paměti.")
    return results
