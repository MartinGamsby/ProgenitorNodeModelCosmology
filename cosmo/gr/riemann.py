"""
Curvature tensors for GR formulation of the External-Node Model.

Riemann tensor, Ricci tensor/scalar, Einstein tensor, Weyl tensor,
tidal tensor, and Kretschner scalar. SI units throughout with explicit c factors.
Convention: R^mu_nu_rho_sigma (first index up, rest down).
Index order (t,r,theta,phi) = (0,1,2,3).
"""

import numpy as np

from cosmo.constants import CosmologicalConstants

G = CosmologicalConstants.G
c = CosmologicalConstants.c


def riemann_schwarzschild(r_m: float, M_kg: float) -> np.ndarray:
    """
    Analytic Riemann tensor R^mu_nu_rho_sigma for Schwarzschild at theta=pi/2.

    All 6 independent component pairs derived from Christoffel symbol algebra
    with the metric ds^2 = -fc^2 dt^2 + dr^2/f + r^2 dOmega^2.
    Antisymmetric in last two indices: R^m_n_rs = -R^m_n_sr.

    Returns (4,4,4,4) array.
    """
    r_s = 2 * G * M_kg / c**2
    f = 1 - r_s / r_m
    r = r_m

    R = np.zeros((4, 4, 4, 4))

    # --- (tr) pair ---
    # R^t_rtr = r_s / (r^2 (r - r_s))
    R[0, 1, 0, 1] = r_s / (r**2 * (r - r_s))
    R[0, 1, 1, 0] = -R[0, 1, 0, 1]
    # R^r_trt = -r_s c^2 (r - r_s) / r^4
    R[1, 0, 1, 0] = -r_s * c**2 * (r - r_s) / r**4
    R[1, 0, 0, 1] = -R[1, 0, 1, 0]

    # --- (t theta) pair ---
    # R^t_theta_t_theta = -r_s / (2r)
    R[0, 2, 0, 2] = -r_s / (2 * r)
    R[0, 2, 2, 0] = -R[0, 2, 0, 2]
    # R^theta_t_theta_t = r_s c^2 (r - r_s) / (2 r^4)
    R[2, 0, 2, 0] = r_s * c**2 * (r - r_s) / (2 * r**4)
    R[2, 0, 0, 2] = -R[2, 0, 2, 0]

    # --- (t phi) pair ---
    R[0, 3, 0, 3] = -r_s / (2 * r)
    R[0, 3, 3, 0] = -R[0, 3, 0, 3]
    R[3, 0, 3, 0] = r_s * c**2 * (r - r_s) / (2 * r**4)
    R[3, 0, 0, 3] = -R[3, 0, 3, 0]

    # --- (r theta) pair ---
    # R^r_theta_r_theta = -r_s / (2r)
    R[1, 2, 1, 2] = -r_s / (2 * r)
    R[1, 2, 2, 1] = -R[1, 2, 1, 2]
    # R^theta_r_theta_r = -r_s / (2 r^2 (r - r_s))
    R[2, 1, 2, 1] = -r_s / (2 * r**2 * (r - r_s))
    R[2, 1, 1, 2] = -R[2, 1, 2, 1]

    # --- (r phi) pair ---
    R[1, 3, 1, 3] = -r_s / (2 * r)
    R[1, 3, 3, 1] = -R[1, 3, 1, 3]
    R[3, 1, 3, 1] = -r_s / (2 * r**2 * (r - r_s))
    R[3, 1, 1, 3] = -R[3, 1, 3, 1]

    # --- (theta phi) pair ---
    # R^theta_phi_theta_phi = r_s / r
    R[2, 3, 2, 3] = r_s / r
    R[2, 3, 3, 2] = -R[2, 3, 2, 3]
    # R^phi_theta_phi_theta = r_s / r
    R[3, 2, 3, 2] = r_s / r
    R[3, 2, 2, 3] = -R[3, 2, 3, 2]

    return R


def riemann_from_christoffel(gamma: np.ndarray, dgamma: np.ndarray) -> np.ndarray:
    """
    Riemann tensor from Christoffel symbols and their derivatives.

    R^mu_nu_rho_sigma = d_rho Gamma^mu_nu_sigma - d_sigma Gamma^mu_nu_rho
                      + Gamma^mu_lam_rho Gamma^lam_nu_sigma
                      - Gamma^mu_lam_sigma Gamma^lam_nu_rho

    Args:
        gamma: Christoffel symbols Gamma^mu_nu_rho, shape (4,4,4)
        dgamma: Derivatives d_sigma Gamma^mu_nu_rho, shape (4,4,4,4)
                dgamma[mu,nu,rho,sigma] = partial_sigma Gamma^mu_nu_rho

    Returns:
        Riemann tensor R^mu_nu_rho_sigma, shape (4,4,4,4)
    """
    R = np.zeros((4, 4, 4, 4))

    for mu in range(4):
        for nu in range(4):
            for rho in range(4):
                for sigma in range(4):
                    # Partial derivative terms
                    val = dgamma[mu, nu, sigma, rho] - dgamma[mu, nu, rho, sigma]
                    # Connection terms
                    for lam in range(4):
                        val += (gamma[mu, lam, rho] * gamma[lam, nu, sigma]
                                - gamma[mu, lam, sigma] * gamma[lam, nu, rho])
                    R[mu, nu, rho, sigma] = val

    return R


def ricci_tensor(riemann: np.ndarray) -> np.ndarray:
    """
    Ricci tensor R_mu_nu = R^lambda_mu_lambda_nu (contract 1st and 3rd indices).

    Args:
        riemann: Riemann tensor R^mu_nu_rho_sigma, shape (4,4,4,4)

    Returns:
        Ricci tensor R_mu_nu, shape (4,4)
    """
    # R_mu_nu = R^lam_mu_lam_nu = sum over lam of riemann[lam, mu, lam, nu]
    return np.einsum('imin->mn', riemann, optimize=True)


def ricci_scalar(ricci: np.ndarray, g_inv: np.ndarray) -> float:
    """
    Ricci scalar R = g^{mu nu} R_{mu nu}.

    Args:
        ricci: Ricci tensor R_mu_nu, shape (4,4)
        g_inv: Inverse metric g^{mu nu}, shape (4,4)

    Returns:
        Ricci scalar R
    """
    return np.einsum('mn,mn->', g_inv, ricci)


def einstein_tensor(ricci: np.ndarray, R_scalar: float, g: np.ndarray) -> np.ndarray:
    """
    Einstein tensor G_mu_nu = R_mu_nu - (1/2) g_mu_nu R.

    Args:
        ricci: Ricci tensor R_mu_nu, shape (4,4)
        R_scalar: Ricci scalar
        g: Metric tensor g_mu_nu, shape (4,4)

    Returns:
        Einstein tensor G_mu_nu, shape (4,4)
    """
    return ricci - 0.5 * g * R_scalar


def weyl_tensor(riemann: np.ndarray, ricci: np.ndarray, R_scalar: float,
                g: np.ndarray) -> np.ndarray:
    """
    Weyl (conformal) tensor C^mu_nu_rho_sigma in 4D.

    C^mu_nu_rho_sigma = R^mu_nu_rho_sigma
        - 1/(n-2) (delta^mu_rho R_nu_sigma - delta^mu_sigma R_nu_rho
                    - g_nu_rho R^mu_sigma + g_nu_sigma R^mu_rho)
        + R / ((n-1)(n-2)) (delta^mu_rho g_nu_sigma - delta^mu_sigma g_nu_rho)

    where n=4. Uses mixed index form: first index up, rest down.
    R^mu_sigma = g^{mu lam} R_{lam sigma} (raised Ricci).

    Args:
        riemann: R^mu_nu_rho_sigma, shape (4,4,4,4)
        ricci: R_mu_nu, shape (4,4)
        R_scalar: Ricci scalar
        g: Metric tensor g_mu_nu, shape (4,4)

    Returns:
        Weyl tensor C^mu_nu_rho_sigma, shape (4,4,4,4)
    """
    n = 4
    g_inv = np.linalg.inv(g)
    delta = np.eye(4)

    # Raise first index of Ricci: R^mu_nu = g^{mu lam} R_{lam nu}
    ricci_up = np.einsum('ml,ln->mn', g_inv, ricci)

    C = np.copy(riemann)

    for mu in range(4):
        for nu in range(4):
            for rho in range(4):
                for sigma in range(4):
                    # Second term: -1/(n-2) * (...)
                    term2 = (delta[mu, rho] * ricci[nu, sigma]
                             - delta[mu, sigma] * ricci[nu, rho]
                             - g[nu, rho] * ricci_up[mu, sigma]
                             + g[nu, sigma] * ricci_up[mu, rho])
                    # Third term: R/((n-1)(n-2)) * (...)
                    term3 = R_scalar * (delta[mu, rho] * g[nu, sigma]
                                        - delta[mu, sigma] * g[nu, rho])

                    C[mu, nu, rho, sigma] -= term2 / (n - 2)
                    C[mu, nu, rho, sigma] += term3 / ((n - 1) * (n - 2))

    return C


def tidal_tensor_from_riemann(riemann: np.ndarray) -> np.ndarray:
    """
    Electrogravitic (tidal) tensor E_ij = R^i_0j0.

    Extracts the spatial 3x3 tidal tensor from the mixed Riemann tensor.
    For a static observer in Schwarzschild:
    - E_rr < 0 (radial stretching)
    - E_theta_theta > 0, E_phi_phi > 0 (transverse compression)
    - Traceless in vacuum (Tr(E) = R_00 = 0)

    Args:
        riemann: R^mu_nu_rho_sigma, shape (4,4,4,4)

    Returns:
        Tidal tensor E_ij, shape (3,3) with i,j in {1,2,3} = {r,theta,phi}
    """
    return riemann[1:, 0, 1:, 0]


def kretschner_scalar(riemann: np.ndarray, g: np.ndarray) -> float:
    """
    Kretschner scalar K = R_{mu nu rho sigma} R^{mu nu rho sigma}.

    For Schwarzschild: K = 48 G^2 M^2 / (c^4 r^6).

    Computes by fully lowering Riemann, then fully raising a second copy,
    and contracting.

    Args:
        riemann: R^mu_nu_rho_sigma, shape (4,4,4,4)
        g: Metric tensor g_mu_nu, shape (4,4)

    Returns:
        Kretschner scalar
    """
    g_inv = np.linalg.inv(g)

    # Lower first index: R_{alpha nu rho sigma} = g_{alpha mu} R^mu_nu_rho_sigma
    R_down = np.einsum('am,mnrs->anrs', g, riemann)

    # Raise all indices of the second copy:
    # R^{alpha beta gamma delta} = g^{alpha a} g^{beta b} g^{gamma c} g^{delta d} R_{abcd}
    R_up = np.einsum('aA,bB,cC,dD,ABCD->abcd', g_inv, g_inv, g_inv, g_inv, R_down)

    return np.einsum('abcd,abcd->', R_down, R_up)
