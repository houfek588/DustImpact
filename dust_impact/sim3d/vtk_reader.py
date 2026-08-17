# -*- coding: utf-8 -*-
"""
VTK Data loader and SPIS mesh interpolator for 3D PIC simulation.
"""

import os
import numpy as np
try:
    import pyvista as pv
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False

from scipy.spatial import cKDTree

from dust_impact.sim3d.config_loader import SimulationParams3D
from dust_impact.sim3d.voxelizer import detect_metal_mask_3d


def _extract_potential_from_mesh(sampled_mesh, filename: str) -> np.ndarray:
    possible_names = [
        'Potential', 'Potencial', 'V', 'PlasmaPotential', 'U',
        'plasma pot at t = 1.0 s'
    ]
    available_keys = list(sampled_mesh.point_data.keys())

    for name in possible_names:
        if name in available_keys:
            return sampled_mesh.point_data[name]

    raise KeyError(
        f"\n[CHYBA] V souboru '{filename}' se nepodařilo najít pole s potenciálem.\n"
        f"Dostupná datová pole v tomto souboru jsou: {available_keys}\n"
    )


def _interpolate_field_3d(x, y, z, field_x, field_y, field_z, params: SimulationParams3D):
    """ Trilinear interpolation of 3D vector field. """
    idx_x = (x - params.x_grid[0]) / params.dx
    idx_y = (y - params.y_grid[0]) / params.dy
    idx_z = (z - params.z_grid[0]) / params.dz

    i0 = int(np.floor(idx_x))
    i1 = i0 + 1
    j0 = int(np.floor(idx_y))
    j1 = j0 + 1
    k0 = int(np.floor(idx_z))
    k1 = k0 + 1

    i0 = max(0, min(params.Nx - 1, i0))
    i1 = max(0, min(params.Nx - 1, i1))
    j0 = max(0, min(params.Ny - 1, j0))
    j1 = max(0, min(params.Ny - 1, j1))
    k0 = max(0, min(params.Nz - 1, k0))
    k1 = max(0, min(params.Nz - 1, k1))

    tx = max(0.0, min(1.0, idx_x - i0))
    ty = max(0.0, min(1.0, idx_y - j0))
    tz = max(0.0, min(1.0, idx_z - k0))

    def interp_comp(f):
        c000 = f[i0, j0, k0]
        c100 = f[i1, j0, k0]
        c010 = f[i0, j1, k0]
        c110 = f[i1, j1, k0]
        c001 = f[i0, j0, k1]
        c101 = f[i1, j0, k1]
        c011 = f[i0, j1, k1]
        c111 = f[i1, j1, k1]

        c00 = c000 * (1 - tx) + c100 * tx
        c10 = c010 * (1 - tx) + c110 * tx
        c01 = c001 * (1 - tx) + c101 * tx
        c11 = c011 * (1 - tx) + c111 * tx

        c0 = c00 * (1 - ty) + c10 * ty
        c1 = c01 * (1 - ty) + c11 * ty

        return c0 * (1 - tz) + c1 * tz

    return interp_comp(field_x), interp_comp(field_y), interp_comp(field_z)


def _compute_impact_intersection_and_normal(params: SimulationParams3D, Vw_body: np.ndarray, spacecraft_mask_3d: np.ndarray, mesh_body=None):
    P_start = np.array(params.impact_location_xyz_m if hasattr(params, 'impact_location_xyz_m') else params.impact_pos, dtype=float)
    v_dir = np.array(getattr(params, 'impact_direction_vector', [0.0, 0.0, 0.0]), dtype=float)
    v_norm = np.linalg.norm(v_dir)

    # 1. Zero direction vector [0,0,0]: do not calculate intersection, set impact_pos directly to P_start
    if v_norm < 1e-9:
        print(f"  -> Vektor pohybu prachu je nulový {list(v_dir)}. Bod dopadu je stanoven přímo ve výchozím místě {list(P_start)}.")
        params.impact_pos = list(P_start)

        Ex_body, Ey_body, Ez_body = np.gradient(-Vw_body, params.dx, params.dy, params.dz)
        gx, gy, gz = _interpolate_field_3d(P_start[0], P_start[1], P_start[2], Ex_body, Ey_body, Ez_body, params)
        normal = np.array([gx, gy, gz], dtype=float)
        norm_n = np.linalg.norm(normal)
        if norm_n > 1e-6:
            normal /= norm_n
        else:
            norm_p = np.linalg.norm(P_start)
            normal = P_start / norm_p if norm_p > 1e-6 else np.array([-0.7071, 0.7071, 0.0])
        params.impact_normal = list(normal)
        return

    # 2. Non-zero direction vector: calculate ray intersection with spacecraft surface
    dir_u = v_dir / v_norm
    intersection_point = None
    normal_vector = None

    # Analytical ray_trace via PyVista mesh_body if available
    if mesh_body is not None:
        try:
            P_target = P_start + 100.0 * dir_u
            points, ind_faces = mesh_body.ray_trace(P_start, P_target)
            if len(points) > 0:
                intersection_point = points[0]
                try:
                    norm = mesh_body.face_normals[ind_faces[0]]
                    normal_vector = norm / np.linalg.norm(norm)
                except Exception:
                    pass
        except Exception:
            pass

    # Grid step ray-casting fallback if PyVista ray_trace didn't return point
    if intersection_point is None:
        step_size = 0.25 * min(params.dx, params.dy, params.dz)
        max_dist = 2.0 * max(params.L_x, params.L_y, params.L_z)
        steps = int(max_dist / step_size)

        for step in range(steps):
            curr_pos = P_start + step * step_size * dir_u
            i = int(round((curr_pos[0] - params.x_grid[0]) / params.dx))
            j = int(round((curr_pos[1] - params.y_grid[0]) / params.dy))
            k = int(round((curr_pos[2] - params.z_grid[0]) / params.dz))

            if 0 <= i < params.Nx and 0 <= j < params.Ny and 0 <= k < params.Nz:
                if spacecraft_mask_3d[i, j, k]:
                    intersection_point = curr_pos
                    break

    # If NO intersection exists along the trajectory: raise error message!
    if intersection_point is None:
        err_msg = (
            f"\n[CHYBA] Nelze spočítat bod dopadu! Prachová částice z výchozího místa {list(P_start)} "
            f"s vektorem pohybu {list(v_dir)} neprotíná povrch sondy!"
        )
        print(err_msg)
        raise ValueError(err_msg)

    if normal_vector is None:
        Ex_body, Ey_body, Ez_body = np.gradient(-Vw_body, params.dx, params.dy, params.dz)
        gx, gy, gz = _interpolate_field_3d(intersection_point[0], intersection_point[1], intersection_point[2],
                                           Ex_body, Ey_body, Ez_body, params)
        normal = np.array([gx, gy, gz], dtype=float)
        norm_n = np.linalg.norm(normal)
        if norm_n > 1e-6:
            normal_vector = normal / norm_n
        else:
            normal_vector = np.array([-0.7071, 0.7071, 0.0])

    params.impact_pos = list(intersection_point)
    params.impact_normal = list(normal_vector)

    print(f"  -> Skutečný vypočtený bod dopadu na povrchu sondy: {params.impact_pos}")
    print(f"  -> Vypočtená normála v místě dopadu: {params.impact_normal}")


def _generate_synthetic_fields_3d(params: SimulationParams3D):
    """ Synthetic 3D analytical Gaussian field fallback if VTK files are missing or pyvista is unavailable. """
    print("[FALLBACK] Generuji syntetická 3D pole pro ladění...")
    Nx, Ny, Nz = params.Nx, params.Ny, params.Nz
    X, Y, Z = np.meshgrid(params.x_grid, params.y_grid, params.z_grid, indexing='ij')

    r_spacecraft = 1.0
    spacecraft_mask = (X**2 + Y**2 + Z**2) <= r_spacecraft**2

    V_bg = params.Vf * np.exp(-np.sqrt(X**2 + Y**2 + Z**2) / params.debye_length)
    Ex_bg, Ey_bg, Ez_bg = np.gradient(-V_bg, params.dx, params.dy, params.dz)
    Ex_bg[spacecraft_mask] = 0.0
    Ey_bg[spacecraft_mask] = 0.0
    Ez_bg[spacecraft_mask] = 0.0

    ant_positions = [[1.5, 0, 0], [0, 1.5, 0], [0, 0, 1.5]]
    Vw_grids, Ewx_list, Ewy_list, Ewz_list, antenna_masks = [], [], [], [], []

    for pos in ant_positions:
        r_sq = (X - pos[0])**2 + (Y - pos[1])**2 + (Z - pos[2])**2
        Vw = np.exp(-r_sq / 0.5**2)
        mask = r_sq <= 0.15**2
        Ewx, Ewy, Ewz = np.gradient(-Vw, params.dx, params.dy, params.dz)
        Ewx[mask], Ewy[mask], Ewz[mask] = 0.0, 0.0, 0.0
        Vw_grids.append(Vw)
        Ewx_list.append(Ewx)
        Ewy_list.append(Ewy)
        Ewz_list.append(Ewz)
        antenna_masks.append(mask)

    return V_bg, Vw_grids, Ex_bg, Ey_bg, Ez_bg, Ewx_list, Ewy_list, Ewz_list, antenna_masks, spacecraft_mask


def load_and_interpolate_vtk(params: SimulationParams3D):
    if not HAS_PYVISTA:
        return _generate_synthetic_fields_3d(params)

    print("Mapování SPIS VTK na 3D pravoúhlou mřížku...")
    pic_grid = pv.ImageData(
        dimensions=(params.Nx, params.Ny, params.Nz),
        spacing=(params.dx, params.dy, params.dz),
        origin=(-params.L_x, -params.L_y, -params.L_z)
    )

    Nx, Ny, Nz = params.Nx, params.Ny, params.Nz
    dx_min = min(params.dx, params.dy, params.dz)

    def _read_mesh(filepath: str):
        if os.path.exists(filepath):
            return pv.read(filepath)
        alt_path = os.path.join("inputs", filepath)
        if os.path.exists(alt_path):
            return pv.read(alt_path)
        return pv.read(filepath)

    bg_file = getattr(params.vtk_files, 'spis_background_potential_file', getattr(params.vtk_files, 'background_potential', 'inputs/spis_V_bg.vtk'))
    sc_file = getattr(params.vtk_files, 'spacecraft_weighting_file', getattr(params.vtk_files, 'spacecraft_weighting', 'inputs/spis_Vw_body.vtk'))
    ant_files = getattr(params.vtk_files, 'antenna_weighting_files', getattr(params.vtk_files, 'antenna_weighting', []))

    try:
        mesh_bg = _read_mesh(bg_file)
        sampled_bg = pic_grid.sample(mesh_bg)
        pot_data = _extract_potential_from_mesh(sampled_bg, bg_file)
        V_bg_grid = pot_data.reshape((Nx, Ny, Nz))
        print(f"  -> Pozadí načteno. Rozsah potenciálu: {np.nanmin(V_bg_grid):.2f} až {np.nanmax(V_bg_grid):.2f} V")

        mesh_body = _read_mesh(sc_file)
        sampled_body = pic_grid.sample(mesh_body)
        pot_body = _extract_potential_from_mesh(sampled_body, sc_file)
        Vw_body = pot_body.reshape((Nx, Ny, Nz))

        # 1. Spacecraft Surface Mask via PyVista select_enclosed_points
        try:
            body_keys = list(mesh_body.point_data.keys())
            key_body = body_keys[0] if body_keys else 'Potential'
            contour_body = mesh_body.contour([0.5], scalars=key_body)
            enclosed_body = pic_grid.select_enclosed_points(contour_body, tolerance=1e-5)
            spacecraft_mask_3d = enclosed_body['SelectedPoints'].view(bool).reshape((Nx, Ny, Nz))
            if np.sum(spacecraft_mask_3d) == 0:
                spacecraft_mask_3d = detect_metal_mask_3d(Vw_body, dx_min)
        except Exception:
            spacecraft_mask_3d = detect_metal_mask_3d(Vw_body, dx_min)

        print(f"  -> Geometrie tělesa sondy načtena z {sc_file} (PyVista select_enclosed_points).")
    except (FileNotFoundError, KeyError, Exception) as e:
        print(f"[UPOZORNĚNÍ] Chyba při načítání VTK ({e}). Přecházím na syntetická pole.")
        return _generate_synthetic_fields_3d(params)

    _compute_impact_intersection_and_normal(params, Vw_body, spacecraft_mask_3d, mesh_body)

    Ex_bg, Ey_bg, Ez_bg = np.gradient(-V_bg_grid, params.dx, params.dy, params.dz)
    Ex_bg[spacecraft_mask_3d] = 0.0
    Ey_bg[spacecraft_mask_3d] = 0.0
    Ez_bg[spacecraft_mask_3d] = 0.0

    Vw_grids = []
    Ewx_list, Ewy_list, Ewz_list = [], [], []
    antenna_masks_3d = []

    X_mat, Y_mat, Z_mat = np.meshgrid(params.x_grid, params.y_grid, params.z_grid, indexing='ij')
    all_grid_coords = np.column_stack([X_mat.ravel(), Y_mat.ravel(), Z_mat.ravel()])

    for i, vtk_file in enumerate(ant_files):
        try:
            mesh_w = _read_mesh(vtk_file)
            sampled_w = pic_grid.sample(mesh_w)
            pot_data = _extract_potential_from_mesh(sampled_w, vtk_file)
            Vw = pot_data.reshape((Nx, Ny, Nz))
        except (FileNotFoundError, KeyError, Exception) as e:
            print(f"[UPOZORNĚNÍ] VTK antény {i+1} nenalezen. Používám syntetické pole.")
            Vw = np.zeros((Nx, Ny, Nz))

        Vw_grids.append(Vw)
        Ewx, Ewy, Ewz = np.gradient(-Vw, params.dx, params.dy, params.dz)

        # 2. Thin Antenna Mask via Distance Field from wire axis
        w_threshold = getattr(params, 'antenna_weighting_threshold', 0.2)
        wire_indices = np.where(Vw >= w_threshold)
        if len(wire_indices[0]) > 0:
            wire_coords = np.column_stack([
                params.x_grid[wire_indices[0]],
                params.y_grid[wire_indices[1]],
                params.z_grid[wire_indices[2]]
            ])
            tree = cKDTree(wire_coords)
            dists, _ = tree.query(all_grid_coords, k=1)
            configured_r = getattr(params, 'antenna_mask_radius_m', 0.15)
            r_ant = max(configured_r, 0.5 * dx_min)
            mask = (dists.reshape((Nx, Ny, Nz)) <= r_ant)
        else:
            mask = detect_metal_mask_3d(Vw, dx_min)

        antenna_masks_3d.append(mask)

        Ewx[mask] = 0.0
        Ewy[mask] = 0.0
        Ewz[mask] = 0.0

        Ewx_list.append(Ewx)
        Ewy_list.append(Ewy)
        Ewz_list.append(Ewz)

    # Automaticky načteme přesný rovnovážný potenciál antén přímo z originální VTK sítě SPISu (nezávisle na rozlišení mřížky)
    spis_v_bias = []
    bg_keys = list(mesh_bg.point_data.keys())
    key_bg = bg_keys[0] if bg_keys else 'Potential'

    for i, vtk_file in enumerate(ant_files):
        extracted = False
        try:
            mesh_w = _read_mesh(vtk_file)
            w_keys = list(mesh_w.point_data.keys())
            key_w = w_keys[0] if w_keys else 'Potential'

            vw_pts = mesh_w.point_data[key_w]
            v_max_w = np.nanmax(vw_pts)

            if v_max_w > 0:
                surf_pts_idx = np.where(vw_pts >= 0.8 * v_max_w)[0]
                if len(surf_pts_idx) > 0 and key_bg in mesh_bg.point_data:
                    v_bg_surf = mesh_bg.point_data[key_bg][surf_pts_idx]
                    v_ant_exact = float(np.nanmean(v_bg_surf))
                    spis_v_bias.append(v_ant_exact)
                    print(f"  -> Anténa {i + 1}: Rovnovážný potenciál načten přímo ze SPIS VTK: {v_ant_exact:.3f} V")
                    extracted = True
        except Exception:
            pass

        if not extracted:
            if i < len(Vw_grids) and np.any(antenna_masks_3d[i]):
                v_ant_val = float(np.nanmax(V_bg_grid[antenna_masks_3d[i]]))
                spis_v_bias.append(v_ant_val)
                print(f"  -> Anténa {i + 1}: Rovnovážný potenciál načten z mřížky: {v_ant_val:.3f} V")
            elif i < len(params.antenna_bias_voltage_V):
                spis_v_bias.append(params.antenna_bias_voltage_V[i])

    if spis_v_bias:
        params.antenna_bias_voltage_V = spis_v_bias
        params.V_bias = spis_v_bias

    return V_bg_grid, Vw_grids, Ex_bg, Ey_bg, Ez_bg, Ewx_list, Ewy_list, Ewz_list, antenna_masks_3d, spacecraft_mask_3d
