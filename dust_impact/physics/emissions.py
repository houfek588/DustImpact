# -*- coding: utf-8 -*-
"""
Physics of plasma cloud thermal velocity distributions and particle emissions.
Maxwellian / half-Maxwellian distributions for plasma and emitted electrons and ions.

Convention: a Maxwellian of temperature T has m <v_k^2> = k_B T in every Cartesian component k,
i.e. the standard deviation sigma = sqrt(k_B T / m) = v_th / sqrt(2), where
v_th = sqrt(2 k_B T / m) is the most probable speed used throughout the configuration.
The half-Maxwellian keeps m <v_k^2> = k_B T (second moment about zero) of its parent Maxwellian.
"""

import numpy as np
from typing import Tuple
from dust_impact.physics.constants import e, m_e, amu


def maxwellian_sigma(T_eV: float, mass_kg: float) -> float:
    """
    Standard deviation of one Cartesian velocity component of a Maxwellian.

    Parameters:
    -----------
    T_eV : float
        Kinetic temperature in eV.
    mass_kg : float
        Particle mass in kg.

    Returns:
    --------
    sigma : float
        sqrt(e * T_eV / mass_kg) in m/s.
    """
    return float(np.sqrt(e * T_eV / mass_kg))


def sample_maxwellian_velocities(N_particles: int, T_eV: float, mass_kg: float) -> np.ndarray:
    """
    Samples velocities from an isotropic, non-drifting 3D Maxwellian with temperature T_eV.

    Every Cartesian component is Gaussian with zero mean and standard deviation
    sqrt(e * T_eV / mass_kg), so that m <v_k^2> = e * T_eV and 1/2 m <|v|^2> = 3/2 e * T_eV.

    Parameters:
    -----------
    N_particles : int
        Number of particles.
    T_eV : float
        Kinetic temperature in eV.
    mass_kg : float
        Particle mass in kg.

    Returns:
    --------
    v : np.ndarray
        Velocities of shape (N_particles, 3) in m/s.
    """
    return np.random.normal(0.0, maxwellian_sigma(T_eV, mass_kg), (N_particles, 3))


def sample_half_maxwellian_velocities(
    N_particles: int, T_eV: float, mass_kg: float, normal: np.ndarray
) -> np.ndarray:
    """
    Samples velocities from a 3D Maxwellian restricted to the half-space v . n >= 0.

    Models a plasma cloud of temperature T_eV released from a surface with outward unit
    normal n: an isotropic Maxwellian is sampled and every velocity pointing into the surface
    is specularly reflected, v -> v - 2 (v . n) n. The reflection keeps the tangential components
    Maxwellian and turns the normal component into |N(0, sigma)|, hence m <v_k^2> = e * T_eV holds
    for every Cartesian component and the mean expansion velocity along n is sigma * sqrt(2 / pi).

    Parameters:
    -----------
    N_particles : int
        Number of particles.
    T_eV : float
        Kinetic temperature in eV.
    mass_kg : float
        Particle mass in kg.
    normal : np.ndarray
        Outward surface normal of shape (3,); it does not need to be normalised.

    Returns:
    --------
    v : np.ndarray
        Velocities of shape (N_particles, 3) in m/s, all with v . n >= 0.
    """
    n = np.asarray(normal, dtype=float)
    n_norm = np.linalg.norm(n)
    if n_norm == 0.0:
        raise ValueError("Surface normal must be a non-zero vector")
    n = n / n_norm

    v = sample_maxwellian_velocities(N_particles, T_eV, mass_kg)
    v_n = v @ n
    v -= 2.0 * np.minimum(v_n, 0.0)[:, None] * n[None, :]
    return v


def sample_cone_beam_velocities(
    N_particles: int, drift_speed: float, half_angle_deg: float, T_eV: float, mass_kg: float, normal: np.ndarray
) -> np.ndarray:
    """
    Samples a drifting conical beam leaving a surface, the ion part of an impact plasma cloud.

    Following the electrostatic impact-plasma model (Shen et al. 2021, 2023), impact ions expand from
    the impact site as a beam of speed v_d inside a cone around the outward surface normal n. Each
    particle gets a drift direction drawn uniformly in solid angle inside the cone of half-angle
    theta_max (cos(theta) uniform on [cos(theta_max), 1], azimuth uniform), plus an isotropic thermal
    velocity of temperature T_eV in the beam frame. Velocities that would point into the surface
    (possible only when the thermal spread exceeds the drift) are specularly reflected, v -> v - 2 (v.n) n.

    The mean velocity along n of the drift part is v_d (1 + cos(theta_max)) / 2.

    Parameters:
    -----------
    N_particles : int
        Number of particles.
    drift_speed : float
        Beam (expansion) speed v_d in m/s.
    half_angle_deg : float
        Cone half-angle theta_max in degrees (0 = pencil beam, 90 = hemisphere).
    T_eV : float
        Thermal spread (temperature in the beam frame) in eV; 0 gives a mono-energetic beam.
    mass_kg : float
        Particle mass in kg.
    normal : np.ndarray
        Outward surface normal of shape (3,); it does not need to be normalised.

    Returns:
    --------
    v : np.ndarray
        Velocities of shape (N_particles, 3) in m/s, all with v . n >= 0.
    """
    n = np.asarray(normal, dtype=float)
    n_norm = np.linalg.norm(n)
    if n_norm == 0.0:
        raise ValueError("Surface normal must be a non-zero vector")
    n = n / n_norm
    # Orthonormal basis (e1, e2, n) of the beam frame
    helper = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(n, helper)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)

    cos_max = np.cos(np.deg2rad(half_angle_deg))
    cos_t = 1.0 - (1.0 - cos_max) * np.random.random_sample(N_particles)
    sin_t = np.sqrt(np.clip(1.0 - cos_t**2, 0.0, None))
    phi = 2.0 * np.pi * np.random.random_sample(N_particles)
    directions = (
        cos_t[:, None] * n[None, :]
        + (sin_t * np.cos(phi))[:, None] * e1[None, :]
        + (sin_t * np.sin(phi))[:, None] * e2[None, :]
    )
    v = drift_speed * directions
    if T_eV > 0.0:
        v += sample_maxwellian_velocities(N_particles, T_eV, mass_kg)
    v_n = v @ n
    v -= 2.0 * np.minimum(v_n, 0.0)[:, None] * n[None, :]
    return v


def generate_half_maxwellian_velocities(N_particles: int, T_eV: float, mass_kg: float) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Generates thermal velocities for emitted particles in a half-space.

    Parameters:
    -----------
    N_particles : int
        Number of particles.
    T_eV : float
        Temperature in eV.
    mass_kg : float
        Particle mass in kg.

    Returns:
    --------
    vx, vy, v_th : Tuple[np.ndarray, np.ndarray, float]
        Perpendicular positive velocity vx, parallel velocity vy, and thermal velocity v_th.
    """
    v_th = np.sqrt(2.0 * T_eV * e / mass_kg)
    theta = np.random.uniform(-np.pi / 2, np.pi / 2, N_particles)
    v_mag = np.random.normal(v_th, v_th / 2.0, N_particles)
    vx = np.abs(v_mag * np.cos(theta))
    vy = v_mag * np.sin(theta)
    return vx, vy, float(v_th)
