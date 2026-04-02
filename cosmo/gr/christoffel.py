"""
Christoffel symbols for GR formulation of the External-Node Model.

Analytic Christoffel symbols for Schwarzschild and FRW spacetimes,
plus numerical computation from metric + derivatives.
Convention: Gamma^mu_nu_rho with index order (t,r,theta,phi) = (0,1,2,3).
SI units throughout with explicit c factors.
"""

import numpy as np

from cosmo.constants import CosmologicalConstants

G = CosmologicalConstants.G
c = CosmologicalConstants.c


def christoffel_schwarzschild(r_m: float, M_kg: float) -> np.ndarray:
    """
    Analytic Christoffel symbols for Schwarzschild metric at theta=pi/2.

    Returns Gamma^mu_nu_rho as a (4,4,4) array.
    Non-zero components (textbook Schwarzschild, equatorial plane):
      Gamma^t_tr = Gamma^t_rt = GM / (r^2 c^2 (1 - r_s/r))
      Gamma^r_tt = GM c^2 (1 - r_s/r) / r^2  (note: GM/r^2 * (1-r_s/r) * c^2... let me be precise)
      Gamma^r_rr = -GM / (r^2 c^2 (1 - r_s/r))   [= -Gamma^t_tr]
      Gamma^r_theta_theta = -(r - r_s)
      Gamma^r_phi_phi = -(r - r_s)   [at theta=pi/2]
      Gamma^theta_r_theta = Gamma^theta_theta_r = 1/r
      Gamma^phi_r_phi = Gamma^phi_phi_r = 1/r
    """
    r_s = 2 * G * M_kg / c**2
    f = 1 - r_s / r_m

    gamma = np.zeros((4, 4, 4))

    # Gamma^t_tr = Gamma^t_rt = (r_s / 2) / (r^2 f)
    # = GM / (r^2 c^2 f)
    val_ttr = G * M_kg / (r_m**2 * c**2 * f)
    gamma[0, 0, 1] = val_ttr  # Gamma^t_tr
    gamma[0, 1, 0] = val_ttr  # Gamma^t_rt

    # Gamma^r_tt = GM * f / r^2
    # Derivation: -(1/2) g^rr * dg_tt/dr = -(1/2) * f * (-c^2 r_s/r^2) = f c^2 r_s/(2r^2) = f GM/r^2
    gamma[1, 0, 0] = G * M_kg * f / r_m**2

    # Gamma^r_rr = -GM / (r^2 c^2 f)
    # = -(r_s/2) / (r^2 f)
    gamma[1, 1, 1] = -G * M_kg / (r_m**2 * c**2 * f)

    # Gamma^r_theta_theta = -(r - r_s)
    gamma[1, 2, 2] = -(r_m - r_s)

    # Gamma^r_phi_phi = -(r - r_s) * sin^2(theta) = -(r - r_s) at theta=pi/2
    gamma[1, 3, 3] = -(r_m - r_s)

    # Gamma^theta_r_theta = Gamma^theta_theta_r = 1/r
    gamma[2, 1, 2] = 1.0 / r_m
    gamma[2, 2, 1] = 1.0 / r_m

    # Gamma^phi_r_phi = Gamma^phi_phi_r = 1/r
    gamma[3, 1, 3] = 1.0 / r_m
    gamma[3, 3, 1] = 1.0 / r_m

    return gamma


def christoffel_frw(a: float, adot: float, k: float = 0) -> np.ndarray:
    """
    Analytic Christoffel symbols for FRW metric at theta=pi/2, r=1.

    Non-zero components (flat FRW, k=0):
      Gamma^t_rr = a * adot / c^2
      Gamma^t_theta_theta = a * adot / c^2   (times r^2, but r=1)
      Gamma^t_phi_phi = a * adot / c^2       (times r^2 sin^2 theta, r=1, theta=pi/2)
      Gamma^r_tr = Gamma^r_rt = adot / a
      Gamma^theta_t_theta = Gamma^theta_theta_t = adot / a
      Gamma^phi_t_phi = Gamma^phi_phi_t = adot / a
      Gamma^theta_r_theta = Gamma^theta_theta_r = 1/r  (=1 for r=1)
      Gamma^phi_r_phi = Gamma^phi_phi_r = 1/r  (=1 for r=1)
      Gamma^r_theta_theta = -r = -1 for r=1
      Gamma^r_phi_phi = -r sin^2(theta) = -1 for r=1, theta=pi/2

    For general k, spatial Christoffel symbols get k-dependent corrections.
    We evaluate at comoving r=1 for simplicity.
    """
    gamma = np.zeros((4, 4, 4))

    H = adot / a if a != 0 else 0.0  # Hubble parameter (physical)

    # Gamma^t_ii = a * adot / c^2 (spatial diagonal, with metric factors)
    gamma[0, 1, 1] = a * adot / c**2                     # Gamma^t_rr (times 1/(1-kr^2) for general k)
    gamma[0, 2, 2] = a * adot / c**2                     # Gamma^t_theta_theta (r=1)
    gamma[0, 3, 3] = a * adot / c**2                     # Gamma^t_phi_phi (r=1, theta=pi/2)

    # Gamma^i_ti = Gamma^i_it = adot/a = H
    gamma[1, 0, 1] = H  # Gamma^r_tr
    gamma[1, 1, 0] = H  # Gamma^r_rt
    gamma[2, 0, 2] = H  # Gamma^theta_t_theta
    gamma[2, 2, 0] = H  # Gamma^theta_theta_t
    gamma[3, 0, 3] = H  # Gamma^phi_t_phi
    gamma[3, 3, 0] = H  # Gamma^phi_phi_t

    # Purely spatial Christoffel symbols (at r=1, theta=pi/2)
    gamma[2, 1, 2] = 1.0  # Gamma^theta_r_theta = 1/r = 1
    gamma[2, 2, 1] = 1.0  # Gamma^theta_theta_r
    gamma[3, 1, 3] = 1.0  # Gamma^phi_r_phi = 1/r = 1
    gamma[3, 3, 1] = 1.0  # Gamma^phi_phi_r

    gamma[1, 2, 2] = -1.0  # Gamma^r_theta_theta = -r = -1
    gamma[1, 3, 3] = -1.0  # Gamma^r_phi_phi = -r sin^2(theta) = -1

    return gamma


def christoffel_from_metric(g: np.ndarray, dg: np.ndarray) -> np.ndarray:
    """
    Numerical Christoffel symbols from metric and its derivatives.

    Gamma^mu_nu_rho = (1/2) g^{mu sigma} (dg_{sigma nu}/dx^rho
                                          + dg_{sigma rho}/dx^nu
                                          - dg_{nu rho}/dx^sigma)

    Args:
        g: Metric tensor g_{mu nu}, shape (4,4)
        dg: Metric derivatives dg_{mu nu}/dx^rho, shape (4,4,4)
            where dg[mu,nu,rho] = partial_rho g_{mu nu}

    Returns:
        Christoffel symbols Gamma^mu_nu_rho, shape (4,4,4)
    """
    g_inv = np.linalg.inv(g)
    gamma = np.zeros((4, 4, 4))

    for mu in range(4):
        for nu in range(4):
            for rho in range(4):
                s = 0.0
                for sigma in range(4):
                    s += g_inv[mu, sigma] * (
                        dg[sigma, nu, rho]
                        + dg[sigma, rho, nu]
                        - dg[nu, rho, sigma]
                    )
                gamma[mu, nu, rho] = 0.5 * s

    return gamma


def christoffel_symmetry_check(gamma: np.ndarray) -> float:
    """
    Check torsion-free symmetry: max|Gamma^m_nl - Gamma^m_ln|.

    For a Levi-Civita connection, Christoffel symbols are symmetric
    in lower indices: Gamma^mu_nu_rho = Gamma^mu_rho_nu.
    """
    max_asym = 0.0
    for mu in range(4):
        for nu in range(4):
            for rho in range(4):
                diff = abs(gamma[mu, nu, rho] - gamma[mu, rho, nu])
                if diff > max_asym:
                    max_asym = diff
    return max_asym
