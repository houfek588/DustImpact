# -*- coding: utf-8 -*-
"""
Charge-escape validation: impact-cloud test particles released next to a biased sphere (no self-field)
either escape to the grounded box or return to the sphere. For a half-Maxwellian of temperature T
released at potential V_start, a species of charge q escapes if its energy exceeds the potential-energy
barrier dU = q (0 - V_start), so the escaping fraction is

    P = erfc(sqrt(x)) + 2 sqrt(x / pi) exp(-x),   x = max(0, dU / T).

The fraction measured in the simulation must match P. The script also prints P for the potential drop
that particles actually feel (integral of the interpolated E from the start point to the wall), which
equals V_start when the electric field next to the conductor is consistent with the potential.

The results file of main.py does not record where each particle ended, therefore this script runs the
simulation itself (configs inputs/config_validation_escape_*.json):
    python scripts/validation/escape_check.py inputs/config_validation_escape_p5.json --steps 1000
"""

import argparse
import contextlib
import io
import os
import sys
import tempfile
import time

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.special import erfc

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from dust_impact.physics.constants import m_e  # noqa: E402
from dust_impact.solver.config_loader import setup_simulation_parameters_3d  # noqa: E402
from dust_impact.geometry import build_simulation_geometry  # noqa: E402
from dust_impact.solver.sim_core import DustImpactSimulation3D  # noqa: E402


def escape_fraction(x: float) -> float:
    """Fraction of a (half-)Maxwellian with energy above x k_B T (3D energy distribution)."""
    x = max(0.0, x)
    return float(erfc(np.sqrt(x)) + 2.0 * np.sqrt(x / np.pi) * np.exp(-x))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('config', help='Configuration JSON (inputs/config_validation_escape_*.json)')
    ap.add_argument('--steps', type=int, default=None, help='Number of time steps (default: from the config)')
    ap.add_argument('--seed', type=int, default=2, help='Random seed (default 2)')
    args = ap.parse_args()

    np.random.seed(args.seed)
    with contextlib.redirect_stdout(io.StringIO()):
        p, toggles, _ = setup_simulation_parameters_3d(0.0, 0.0, config_file=args.config, output_dir=tempfile.mkdtemp())
        if args.steps is not None:
            p.steps = args.steps
        geom = build_simulation_geometry(p)
        sim = DustImpactSimulation3D(p, toggles, geom)

    P = sim.particles
    grid = (p.x_grid, p.y_grid, p.z_grid)
    x0 = np.array([P.x_e[0], P.y_e[0], P.z_e[0]])
    sphere = p.geometry.analytical.spacecraft
    centre, R = np.asarray(sphere.center, dtype=float), float(sphere.radius)
    V_start = float(RegularGridInterpolator(grid, geom.V_bg)(x0[None, :])[0])

    # Potential drop felt by the particles along the outward ray from the start point to the wall
    n_hat = (x0 - centre) / np.linalg.norm(x0 - centre)
    s_wall = min((p.x_grid[-1] - centre[0]) / max(abs(n_hat[0]), 1e-12),
                 (p.y_grid[-1] - centre[1]) / max(abs(n_hat[1]), 1e-12),
                 (p.z_grid[-1] - centre[2]) / max(abs(n_hat[2]), 1e-12))
    s = np.linspace(np.linalg.norm(x0 - centre), s_wall, 20001)
    pts = centre[None, :] + s[:, None] * n_hat[None, :]
    E_par = sum(RegularGridInterpolator(grid, E)(pts) * n_hat[k]
                for k, E in enumerate((geom.Ex_bg, geom.Ey_bg, geom.Ez_bg)))
    V_felt = float(np.trapz(E_par, s))

    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        sim.run()

    print(f"{args.config}: V_sc = {p.V_sc_bias:+.2f} V, V_start = {V_start:+.3f} V, "
          f"int E dl to the wall = {V_felt:+.3f} V, {p.steps} steps ({p.steps * p.dt * 1e6:.1f} us), "
          f"{time.time() - t0:.0f} s")
    for name, q, T, xs, ys, zs, active in (
        ('electrons', -1.0, p.impact_electron_temp_eV, P.x_e, P.y_e, P.z_e, P.active_e),
        ('ions', +1.0, p.impact_ion_temp_eV, P.x_i, P.y_i, P.z_i, P.active_i),
    ):
        r = np.sqrt((xs - centre[0])**2 + (ys - centre[1])**2 + (zs - centre[2])**2)
        returned = np.mean(~active & (r < R + 2.0 * p.dx))
        flying = np.mean(active)
        escaped = 1.0 - returned - flying
        p_exact = escape_fraction(q * (0.0 - V_start) / T)
        p_felt = escape_fraction(q * (0.0 - V_felt) / T)
        print(f"  {name:9s} T = {T:.1f} eV: returned {returned:.4f}, still flying {flying:.4f}, "
              f"escaped {escaped:.4f} | analytical {p_exact:.4f} (with felt drop {p_felt:.4f})")
    print("  escaped + still flying is an upper bound of the final escaped fraction "
          "(slow particles may still return).")


if __name__ == '__main__':
    main()
