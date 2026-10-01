# -*- coding: utf-8 -*-
"""
Unified I/O utilities for saving and loading PIC simulation results.
Supports:
1. Self-contained NPZ format with embedded JSON metadata and coordinate grids.
2. High-performance HDF5 format (.h5) with chunked datasets and streaming capabilities.
3. Robust rotating checkpointing for simulation crash recovery.
"""

import os
import json
import numpy as np
from typing import Dict, Any, Optional

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False


def ensure_dir(filepath: str) -> None:
    """Ensure directory containing filepath exists."""
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


# =====================================================================
# 1. NPZ Format (with Embedded Metadata & Coordinates)
# =====================================================================

def save_results_npz(results: Dict[str, Any], filepath: str, metadata: Optional[Dict[str, Any]] = None) -> None:
    """
    Compress and save simulation results dictionary to self-contained NPZ file.
    
    Parameters:
    -----------
    results : Dict[str, Any]
        Dictionary of simulation outputs (signals, history, voltages).
    filepath : str
        Target destination path for .npz archive.
    metadata : Optional[Dict[str, Any]]
        Optional metadata containing configuration, domain parameters, and grid vectors.
    """
    ensure_dir(filepath)
    print(f"Ukládám výsledky do souboru: {filepath} ...")

    npz_dict = {}
    
    # Pack metadata if provided or found inside results
    meta_to_save = metadata or results.get('metadata')
    if meta_to_save is not None:
        try:
            # We filter non-serializable objects and store as JSON string
            serializable_meta = _make_serializable(meta_to_save)
            npz_dict['metadata_json'] = np.array(json.dumps(serializable_meta, indent=2))
        except Exception as e:
            print(f"  [VAROVÁNÍ] Nelze serializovat metadata do JSON: {e}")

        # Explicitly store coordinate axes if present in metadata
        for grid_key in ['x_grid', 'y_grid', 'z_grid', 'time_array']:
            if grid_key in meta_to_save:
                npz_dict[f'meta_{grid_key}'] = np.array(meta_to_save[grid_key])

    # Pack results and history
    for key, value in results.items():
        if key == 'metadata':
            continue
        elif key == 'history' and isinstance(value, dict):
            for h_key, h_val in value.items():
                npz_dict[f'hist_{h_key}'] = np.array(h_val)
        else:
            npz_dict[key] = np.array(value)

    np.savez_compressed(filepath, **npz_dict)
    print("  [OK] Data úspěšně uložena do binárního archivu (NPZ).")


def load_results_npz(filepath: str) -> Dict[str, Any]:
    """
    Load simulation results from NPZ archive into structured dict with metadata.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Soubor s výsledky nebyl nalezen: {filepath}")

    print(f"Načítám výsledky ze souboru (NPZ): {filepath} ...")
    results: Dict[str, Any] = {'history': {}}

    with np.load(filepath, allow_pickle=True) as data:
        for key in data.files:
            if key == 'metadata_json':
                try:
                    results['metadata'] = json.loads(str(data[key]))
                except Exception:
                    results['metadata'] = str(data[key])
            elif key.startswith('meta_'):
                if 'metadata' not in results or not isinstance(results['metadata'], dict):
                    results['metadata'] = {}
                results['metadata'][key[5:]] = data[key]
            elif key.startswith('hist_'):
                h_key = key[5:]
                results['history'][h_key] = data[key]
            else:
                results[key] = data[key]

    print("  [OK] Data úspěšně načtena do operační paměti.")
    return results


# =====================================================================
# 2. HDF5 Format (Hierarchical, Chunked, Streaming-ready)
# =====================================================================

def save_results_h5(results: Dict[str, Any], filepath: str, metadata: Optional[Dict[str, Any]] = None) -> None:
    """
    Save simulation results to HDF5 (.h5) archive with gzip compression and chunking.
    """
    if not HAS_H5PY:
        raise ImportError("Knihovna 'h5py' není nainstalována. Nainstalujte ji příkazem: pip install h5py")

    ensure_dir(filepath)
    print(f"Ukládám výsledky do souboru: {filepath} ...")

    meta_to_save = metadata or results.get('metadata')

    with h5py.File(filepath, 'w') as h5f:
        # 1. Metadata group
        meta_grp = h5f.create_group('metadata')
        if meta_to_save is not None:
            try:
                s_meta = _make_serializable(meta_to_save)
                meta_grp.attrs['config_json'] = json.dumps(s_meta, indent=2)
            except Exception as e:
                print(f"  [VAROVÁNÍ] Nelze serializovat metadata do HDF5 atributu: {e}")

            for k in ['x_grid', 'y_grid', 'z_grid', 'time_array']:
                if k in meta_to_save:
                    h5f.create_dataset(f'grid/{k}', data=np.array(meta_to_save[k]), compression='gzip')

        # 2. Signals group
        sig_grp = h5f.create_group('signals')
        for sig_key in ['smooth_induced', 'smooth_collected', 'smooth_total', 'voltage_ant']:
            if sig_key in results:
                sig_grp.create_dataset(sig_key, data=np.array(results[sig_key]), compression='gzip')

        # 3. History group (Fields and Particles)
        hist = results.get('history', {})
        if hist:
            hist_grp = h5f.create_group('history')
            if 't' in hist:
                hist_grp.create_dataset('t', data=np.array(hist['t']), compression='gzip')

            # Fields
            fields_grp = hist_grp.create_group('fields')
            for f_key in ['V', 'rho']:
                if f_key in hist and len(hist[f_key]) > 0:
                    arr = np.array(hist[f_key])
                    # arr shape: (N_frames, Nx, Ny, Nz)
                    chunks = (1, arr.shape[1], arr.shape[2], arr.shape[3]) if arr.ndim == 4 else None
                    fields_grp.create_dataset(f_key, data=arr, chunks=chunks, compression='gzip')

            # Particles
            part_grp = hist_grp.create_group('particles')
            for p_key in ['x_e', 'y_e', 'z_e', 'vx_e', 'vy_e', 'vz_e',
                          'x_i', 'y_i', 'z_i', 'vx_i', 'vy_i', 'vz_i']:
                if p_key in hist and len(hist[p_key]) > 0:
                    arr = np.array(hist[p_key])
                    chunks = (1, arr.shape[1]) if arr.ndim == 2 else None
                    part_grp.create_dataset(p_key, data=arr, chunks=chunks, compression='gzip')

    print("  [OK] Data úspěšně uložena do binárního archivu (HDF5).")


def load_results_h5(filepath: str) -> Dict[str, Any]:
    """
    Load simulation results from HDF5 archive into dictionary compatible with plotting pipeline.
    """
    if not HAS_H5PY:
        raise ImportError("Knihovna 'h5py' není nainstalována. Nainstalujte ji příkazem: pip install h5py")

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Soubor s výsledky nebyl nalezen: {filepath}")

    print(f"Načítám výsledky ze souboru (HDF5): {filepath} ...")
    results: Dict[str, Any] = {'history': {}}

    with h5py.File(filepath, 'r') as h5f:
        # Load metadata
        if 'metadata' in h5f:
            meta_grp = h5f['metadata']
            if 'config_json' in meta_grp.attrs:
                try:
                    results['metadata'] = json.loads(meta_grp.attrs['config_json'])
                except Exception:
                    results['metadata'] = str(meta_grp.attrs['config_json'])

        # Load grid
        if 'grid' in h5f:
            if 'metadata' not in results or not isinstance(results['metadata'], dict):
                results['metadata'] = {}
            for k in h5f['grid']:
                results['metadata'][k] = h5f['grid'][k][:]

        # Load signals
        if 'signals' in h5f:
            for k in h5f['signals']:
                results[k] = h5f['signals'][k][:]

        # Load history
        if 'history' in h5f:
            hist_grp = h5f['history']
            if 't' in hist_grp:
                results['history']['t'] = list(hist_grp['t'][:])

            if 'fields' in hist_grp:
                for f_key in hist_grp['fields']:
                    results['history'][f_key] = [hist_grp['fields'][f_key][i] for i in range(len(hist_grp['fields'][f_key]))]

            if 'particles' in hist_grp:
                for p_key in hist_grp['particles']:
                    results['history'][p_key] = [hist_grp['particles'][p_key][i] for i in range(len(hist_grp['particles'][p_key]))]

    print("  [OK] Data úspěšně načtena do operační paměti.")
    return results


class HDF5SimulationWriter:
    """
    Streaming HDF5 writer that writes 3D history frames directly to disk during simulation.
    Drastically minimizes RAM footprint for fine grids.
    """
    def __init__(self, filepath: str, grid_shape: tuple, num_antennas: int, num_steps: int,
                 metadata: Optional[Dict[str, Any]] = None):
        if not HAS_H5PY:
            raise ImportError("h5py is required for HDF5SimulationWriter.")
        ensure_dir(filepath)
        self.filepath = filepath
        self.grid_shape = grid_shape
        self.num_antennas = num_antennas
        self.num_steps = num_steps
        self.metadata = metadata

        self.h5f = h5py.File(filepath, 'w')
        self._init_file()

    def _init_file(self):
        # Metadata
        meta_grp = self.h5f.create_group('metadata')
        if self.metadata:
            try:
                s_meta = _make_serializable(self.metadata)
                meta_grp.attrs['config_json'] = json.dumps(s_meta, indent=2)
            except Exception:
                pass
            for k in ['x_grid', 'y_grid', 'z_grid', 'time_array']:
                if k in self.metadata:
                    self.h5f.create_dataset(f'grid/{k}', data=np.array(self.metadata[k]), compression='gzip')

        # History group with resizable/extendable datasets
        self.hist_grp = self.h5f.create_group('history')
        self.fields_grp = self.hist_grp.create_group('fields')
        self.part_grp = self.hist_grp.create_group('particles')

        Nx, Ny, Nz = self.grid_shape
        self.ds_V = self.fields_grp.create_dataset(
            'V', shape=(0, Nx, Ny, Nz), maxshape=(None, Nx, Ny, Nz),
            chunks=(1, Nx, Ny, Nz), dtype='float64', compression='gzip'
        )
        self.ds_rho = self.fields_grp.create_dataset(
            'rho', shape=(0, Nx, Ny, Nz), maxshape=(None, Nx, Ny, Nz),
            chunks=(1, Nx, Ny, Nz), dtype='float64', compression='gzip'
        )
        self.ds_t = self.hist_grp.create_dataset(
            't', shape=(0,), maxshape=(None,), dtype='float64', compression='gzip'
        )
        self.part_datasets = {}
        self.frame_count = 0

    def write_history_frame(self, t: float, V: np.ndarray, rho: np.ndarray, particles_dict: Dict[str, np.ndarray]):
        """Append a single history time snapshot to HDF5 on disk."""
        self.frame_count += 1
        new_len = self.frame_count

        self.ds_t.resize((new_len,))
        self.ds_t[new_len - 1] = t

        self.ds_V.resize((new_len, *self.grid_shape))
        self.ds_V[new_len - 1] = V

        self.ds_rho.resize((new_len, *self.grid_shape))
        self.ds_rho[new_len - 1] = rho

        for p_name, p_data in particles_dict.items():
            if p_name not in self.part_datasets:
                n_pts = len(p_data)
                self.part_datasets[p_name] = self.part_grp.create_dataset(
                    p_name, shape=(0, n_pts), maxshape=(None, n_pts),
                    chunks=(1, n_pts), dtype='float64', compression='gzip'
                )
            ds = self.part_datasets[p_name]
            ds.resize((new_len, ds.shape[1]))
            ds[new_len - 1] = p_data

        self.h5f.flush()

    def finish(self, signals_dict: Dict[str, Any]):
        """Write final voltage and currents arrays and close file."""
        sig_grp = self.h5f.create_group('signals')
        for k, v in signals_dict.items():
            if k in ['smooth_induced', 'smooth_collected', 'smooth_total', 'voltage_ant']:
                sig_grp.create_dataset(k, data=np.array(v), compression='gzip')
        self.h5f.close()


# =====================================================================
# 3. Checkpointing & Atomic State Persistence
# =====================================================================

def save_checkpoint(results: Dict[str, Any], filepath: str, metadata: Optional[Dict[str, Any]] = None,
                    format_type: str = "npz") -> str:
    """
    Save atomic rotating checkpoint file to prevent simulation data loss upon crashes.
    """
    base, ext = os.path.splitext(filepath)
    ext_clean = ext if ext in ['.npz', '.h5'] else ('.h5' if format_type == 'h5' else '.npz')
    checkpoint_path = f"{base}.checkpoint{ext_clean}"
    tmp_path = f"{checkpoint_path}.tmp"

    if ext_clean == '.h5' and HAS_H5PY:
        save_results_h5(results, tmp_path, metadata=metadata)
    else:
        save_results_npz(results, tmp_path, metadata=metadata)

    # Atomic replace
    if os.path.exists(checkpoint_path):
        try:
            os.remove(checkpoint_path)
        except OSError:
            pass
    os.replace(tmp_path, checkpoint_path)
    return checkpoint_path


# =====================================================================
# 4. Unified Dispatchers
# =====================================================================

def save_results(results: Dict[str, Any], filepath: str, metadata: Optional[Dict[str, Any]] = None,
                 format_type: str = "auto") -> None:
    """
    Save simulation results dispatching to HDF5 or NPZ based on extension or format_type.
    """
    ext = os.path.splitext(filepath)[1].lower()
    if format_type == "h5" or (format_type == "auto" and ext == ".h5"):
        if HAS_H5PY:
            save_results_h5(results, filepath, metadata=metadata)
        else:
            print("  [VAROVÁNÍ] h5py není dostupné, ukládám do formátu NPZ.")
            npz_path = os.path.splitext(filepath)[0] + ".npz"
            save_results_npz(results, npz_path, metadata=metadata)
    else:
        save_results_npz(results, filepath, metadata=metadata)


def load_results(filepath: str) -> Dict[str, Any]:
    """
    Load simulation results automatically detecting whether file is HDF5 or NPZ.
    """
    if not os.path.exists(filepath):
        # Try alternate extension (.h5 <-> .npz)
        base, ext = os.path.splitext(filepath)
        alt = base + ('.npz' if ext == '.h5' else '.h5')
        if os.path.exists(alt):
            filepath = alt
        else:
            raise FileNotFoundError(f"Soubor s výsledky nebyl nalezen: {filepath}")

    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".h5" and HAS_H5PY:
        return load_results_h5(filepath)
    elif ext == ".npz":
        return load_results_npz(filepath)

    # Fallback inspection by magic header
    with open(filepath, 'rb') as f:
        header = f.read(8)
    if header.startswith(b'\x89HDF') and HAS_H5PY:
        return load_results_h5(filepath)
    return load_results_npz(filepath)


# =====================================================================
# Helper Functions
# =====================================================================

def _make_serializable(obj: Any) -> Any:
    """Convert numpy types and dataclasses into standard JSON-serializable Python types."""
    if hasattr(obj, '__dataclass_fields__'):
        from dataclasses import asdict
        return _make_serializable(asdict(obj))
    elif isinstance(obj, dict):
        return {str(k): _make_serializable(v) for k, v in obj.items() if not str(k).startswith('_')}
    elif isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        if obj.size <= 50:
            return obj.tolist()
        return f"<ndarray shape={obj.shape} dtype={obj.dtype}>"
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj
