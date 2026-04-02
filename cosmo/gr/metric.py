"""
Metric tensors for GR formulation of the External-Node Model.

Schwarzschild, FRW, and Minkowski metrics in SI units.
Convention: signature (-,+,+,+), index order (t,r,theta,phi) = (0,1,2,3).
Schwarzschild evaluated at theta=pi/2 (equatorial plane).
Explicit c factors throughout (no geometric units).
"""

import numpy as np

from cosmo.constants import CosmologicalConstants

G = CosmologicalConstants.G
c = CosmologicalConstants.c


def schwarzschild_radius_m(M_kg: float) -> float:
    """Schwarzschild radius r_s = 2GM/c^2 in meters."""
    return 2 * G * M_kg / c**2


def schwarzschild_metric(r_m: float, M_kg: float) -> np.ndarray:
    """
    Diagonal Schwarzschild metric g_uv at theta=pi/2.

    ds^2 = -(1 - r_s/r)c^2 dt^2 + (1 - r_s/r)^{-1} dr^2 + r^2 dphi^2

    Returns 4x4 numpy array. theta components set for theta=pi/2:
    g_theta_theta = r^2, but since we fix theta=pi/2, sin^2(theta)=1
    so g_phi_phi = r^2 as well.
    """
    r_s = schwarzschild_radius_m(M_kg)
    f = 1 - r_s / r_m

    g = np.zeros((4, 4))
    g[0, 0] = -f * c**2        # g_tt
    g[1, 1] = 1.0 / f          # g_rr
    g[2, 2] = r_m**2            # g_theta_theta
    g[3, 3] = r_m**2            # g_phi_phi (sin^2(theta)=1 at theta=pi/2)
    return g


def frw_metric(a: float, k: float = 0, r_m: float = 1) -> np.ndarray:
    """
    Friedmann-Robertson-Walker metric at theta=pi/2.

    ds^2 = -c^2 dt^2 + a^2 [ dr^2/(1-kr^2) + r^2 (dtheta^2 + sin^2 theta dphi^2) ]

    For k=0 (flat): ds^2 = -c^2 dt^2 + a^2 [dr^2 + r^2 dtheta^2 + r^2 dphi^2]
    Evaluated at theta=pi/2.
    """
    g = np.zeros((4, 4))
    g[0, 0] = -c**2
    g[1, 1] = a**2 / (1 - k * r_m**2)
    g[2, 2] = a**2 * r_m**2
    g[3, 3] = a**2 * r_m**2  # sin^2(pi/2) = 1
    return g


def minkowski_metric() -> np.ndarray:
    """Minkowski metric eta = diag(-c^2, 1, 1, 1)."""
    eta = np.zeros((4, 4))
    eta[0, 0] = -c**2
    eta[1, 1] = 1.0
    eta[2, 2] = 1.0
    eta[3, 3] = 1.0
    return eta


def inverse_metric(g: np.ndarray) -> np.ndarray:
    """Inverse metric g^{mu nu} via matrix inversion."""
    return np.linalg.inv(g)


def metric_determinant(g: np.ndarray) -> float:
    """Determinant det(g)."""
    return np.linalg.det(g)


def linearized_metric(h: np.ndarray, eta: np.ndarray = None) -> np.ndarray:
    """
    Linearized metric g = eta + h.

    If eta is None, uses Minkowski metric.
    """
    if eta is None:
        eta = minkowski_metric()
    return eta + h
