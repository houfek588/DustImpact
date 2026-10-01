# -*- coding: utf-8 -*-
"""
ParaView VTK/VTI/VTP Exporter for 3D PIC Simulation Results.
Exports:
1. 3D Fields (Potential V, Charge Density rho) as ImageData (.vti).
2. Species Particles (positions, velocity vectors, speed) as PolyData (.vtp).
3. XML Collection files (.pvd) with time steps for seamless ParaView playback.
"""

import os
import numpy as np
from typing import Dict, Any, Optional

try:
    import pyvista as pv
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False


def _write_pvd_file(pvd_filepath: str, file_entries: list) -> None:
    """
    Write ParaView Collection XML (.pvd) file.
    
    file_entries: list of tuples (timestep: float, relative_filename: str)
    """
    with open(pvd_filepath, 'w', encoding='utf-8') as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n')
        f.write('  <Collection>\n')
        for t_val, rel_fname in file_entries:
            f.write(f'    <DataSet timestep="{t_val:.6e}" group="" part="0" file="{rel_fname}"/>\n')
        f.write('  </Collection>\n')
        f.write('</VTKFile>\n')


def export_fields_to_vti(
    hist_V: list,
    hist_rho: list,
    hist_t: list,
    Nx: int, Ny: int, Nz: int,
    dx: float, dy: float, dz: float,
    origin: tuple,
    output_dir: str
) -> str:
    """
    Export time series of 3D scalar fields to .vti files and generate fields.pvd.
    """
    os.makedirs(output_dir, exist_ok=True)
    pvd_entries = []

    for idx, t_val in enumerate(hist_t):
        fname = f"fields_t{idx:04d}.vti"
        fpath = os.path.join(output_dir, fname)

        V_arr = np.nan_to_num(hist_V[idx], nan=0.0)
        rho_arr = np.nan_to_num(hist_rho[idx], nan=0.0)

        grid = pv.ImageData(
            dimensions=(Nx, Ny, Nz),
            spacing=(dx, dy, dz),
            origin=origin
        )
        grid.point_data["Potential_V"] = V_arr.flatten(order="F")
        grid.point_data["ChargeDensity_C_m3"] = rho_arr.flatten(order="F")
        grid.save(fpath)

        pvd_entries.append((t_val, fname))

    pvd_path = os.path.join(output_dir, "fields.pvd")
    _write_pvd_file(pvd_path, pvd_entries)
    return pvd_path


def export_particles_to_vtp(
    hist_dict: Dict[str, Any],
    species: str,
    output_dir: str
) -> str:
    """
    Export particle species cloud (positions, velocity vectors) to .vtp and generate species.pvd.
    """
    os.makedirs(output_dir, exist_ok=True)
    hist_t = hist_dict.get('t', [])
    pvd_entries = []

    for idx, t_val in enumerate(hist_t):
        fname = f"particles_{species}_t{idx:04d}.vtp"
        fpath = os.path.join(output_dir, fname)

        x = hist_dict[f'x_{species}'][idx]
        y = hist_dict[f'y_{species}'][idx]
        z = hist_dict[f'z_{species}'][idx]
        vx = hist_dict[f'vx_{species}'][idx]
        vy = hist_dict[f'vy_{species}'][idx]
        vz = hist_dict[f'vz_{species}'][idx]

        mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        if np.any(mask):
            pts = np.column_stack([x[mask], y[mask], z[mask]])
            poly = pv.PolyData(pts)
            vel = np.column_stack([vx[mask], vy[mask], vz[mask]])
            poly.point_data["velocity"] = vel
            poly.point_data["speed"] = np.linalg.norm(vel, axis=1)
        else:
            poly = pv.PolyData()

        poly.save(fpath)
        pvd_entries.append((t_val, fname))

    pvd_path = os.path.join(output_dir, f"particles_{species}.pvd")
    _write_pvd_file(pvd_path, pvd_entries)
    return pvd_path


def export_simulation_to_paraview(
    results: Dict[str, Any],
    params: Optional[Any] = None,
    output_dir: str = "outputs/paraview_vtk"
) -> Dict[str, str]:
    """
    Convert simulation results (fields and particles) into ParaView .vti / .vtp / .pvd format.
    
    Parameters:
    -----------
    results : Dict[str, Any]
        Loaded results dictionary containing 'history' and optionally 'metadata'.
    params : Optional[Any]
        SimulationParams3D instance or None (will fallback to results['metadata']).
    output_dir : str
        Target directory to place generated files.
        
    Returns:
    --------
    Dict[str, str] mapping collection names ('fields', 'electrons', 'ions') to .pvd paths.
    """
    if not HAS_PYVISTA:
        raise ImportError("Knihovna 'pyvista' je nutná pro VTK export. Nainstalujte ji: pip install pyvista")

    os.makedirs(output_dir, exist_ok=True)
    hist = results.get('history', {})
    if not hist or 't' not in hist or len(hist['t']) == 0:
        print("  [VAROVÁNÍ] Žádná data historie k exportu do ParaView.")
        return {}

    # Extract spatial grid parameters
    meta = results.get('metadata', {})
    if params is not None:
        Nx, Ny, Nz = params.Nx, params.Ny, params.Nz
        dx, dy, dz = params.dx, params.dy, params.dz
        origin = (-params.L_x, -params.L_y, -params.L_z)
    elif isinstance(meta, dict) and 'x_grid' in meta and 'y_grid' in meta and 'z_grid' in meta:
        xg = np.array(meta['x_grid'])
        yg = np.array(meta['y_grid'])
        zg = np.array(meta['z_grid'])
        Nx, Ny, Nz = len(xg), len(yg), len(zg)
        dx = xg[1] - xg[0] if Nx > 1 else 1.0
        dy = yg[1] - yg[0] if Ny > 1 else 1.0
        dz = zg[1] - zg[0] if Nz > 1 else 1.0
        origin = (xg[0], yg[0], zg[0])
    else:
        # Deduce from 3D field shape
        V0 = np.array(hist['V'][0])
        Nx, Ny, Nz = V0.shape
        dx, dy, dz = 0.1, 0.1, 0.1
        origin = (-dx * (Nx - 1) / 2, -dy * (Ny - 1) / 2, -dz * (Nz - 1) / 2)

    pvd_paths = {}
    print(f"\nExportuji data simulace do formátu ParaView VTK/PVD -> {output_dir} ...")

    # 1. Export 3D Fields
    if 'V' in hist and 'rho' in hist:
        pvd_f = export_fields_to_vti(
            hist['V'], hist['rho'], hist['t'],
            Nx, Ny, Nz, dx, dy, dz, origin, output_dir
        )
        pvd_paths['fields'] = pvd_f
        print(f"  [OK] 3D Pole (V, rho) uložena: {pvd_f}")

    # 2. Export Electrons
    if 'x_e' in hist and len(hist['x_e']) > 0:
        pvd_e = export_particles_to_vtp(hist, 'e', output_dir)
        pvd_paths['electrons'] = pvd_e
        print(f"  [OK] Elektrony uloženy: {pvd_e}")

    # 3. Export Ions
    if 'x_i' in hist and len(hist['x_i']) > 0:
        pvd_i = export_particles_to_vtp(hist, 'i', output_dir)
        pvd_paths['ions'] = pvd_i
        print(f"  [OK] Ionty uloženy: {pvd_i}")

    print("  [OK] ParaView export kompletní. Otevřete soubory .pvd v aplikaci ParaView.")
    return pvd_paths
