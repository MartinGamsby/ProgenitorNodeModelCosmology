"""
Cosmological Distance Kernel

Pure-function module computing:
  - H(z): Hubble parameter at redshift z
  - Comoving distance D_C(z)
  - Transverse comoving distance D_M(z) (curvature-corrected)
  - Luminosity distance d_L(z)
  - Distance modulus mu(z)

And a model-curve helper that builds the three curves used by the
Hubble-diagram comparison (external_node, lcdm, matter_only).

No I/O, no plotting.
"""

import numpy as np
from scipy.integrate import cumulative_trapezoid

from .constants import CosmologicalConstants, LambdaCDMParameters, SimulationParameters


# ---------------------------------------------------------------------------
# Derived constants (computed once at import time for reuse)
# ---------------------------------------------------------------------------

def _c_km_s() -> float:
    """Speed of light in km/s."""
    return CosmologicalConstants.c / 1000.0


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def hubble_z(
    z: float | np.ndarray,
    Omega_m: float,
    Omega_de: float,
    H0: float,
) -> float | np.ndarray:
    """
    Hubble parameter H(z) for a flat-or-curved FRW universe.

    Omega_k is derived as 1 - Omega_m - Omega_de (no flat assumption).

    E(z) = sqrt(Omega_m*(1+z)^3 + Omega_k*(1+z)^2 + Omega_de)

    Args:
        z:        Redshift, scalar or array.
        Omega_m:  Present-day matter density parameter.
        Omega_de: Present-day dark-energy density parameter.
        H0:       Hubble constant in km/s/Mpc.

    Returns:
        H(z) in km/s/Mpc, same shape as z.
    """
    z = np.asarray(z, dtype=float)
    Omega_k = 1.0 - Omega_m - Omega_de
    E_sq = Omega_m * (1.0 + z) ** 3 + Omega_k * (1.0 + z) ** 2 + Omega_de
    if np.any(E_sq < 0):
        raise ValueError(
            "E(z)^2 < 0 for some z values: the FRW model with these density "
            f"parameters (Omega_m={Omega_m}, Omega_de={Omega_de}) has a "
            "turnaround before the requested redshift. "
            "Reduce z_max or check your cosmological parameters."
        )
    E_z = np.sqrt(E_sq)
    return H0 * E_z


def comoving_distance(
    z: float | np.ndarray,
    Omega_m: float,
    Omega_de: float,
    H0: float,
    n_steps: int = 2048,
) -> float | np.ndarray:
    """
    Comoving distance D_C(z) in Mpc.

    D_C = D_H * integral_0^z dz' / E(z')

    where D_H = c / H0 (Hubble distance in Mpc).

    The integral is evaluated on a fine uniform grid in [0, z_max] using
    cumulative trapezoid, then interpolated to the requested z values.
    Accepts scalar or array z.

    Args:
        z:        Redshift, scalar or array. Must be >= 0.
        Omega_m:  Present-day matter density parameter.
        Omega_de: Present-day dark-energy density parameter.
        H0:       Hubble constant in km/s/Mpc.
        n_steps:  Number of integration grid points (default 2048).

    Returns:
        D_C in Mpc, same shape as z.
    """
    scalar_input = np.ndim(z) == 0
    z = np.atleast_1d(np.asarray(z, dtype=float))

    z_max = float(np.max(z))
    if z_max <= 0.0:
        result = np.zeros_like(z)
        return float(result[0]) if scalar_input else result

    # Build fine integration grid from 0 to z_max
    z_grid = np.linspace(0.0, z_max, n_steps)

    # Integrand: 1/E(z')
    E_grid = hubble_z(z_grid, Omega_m, Omega_de, H0) / H0  # E(z) = H(z)/H0
    integrand = 1.0 / E_grid

    # Cumulative trapezoid integral -> D_C grid in Mpc
    D_H = _c_km_s() / H0  # Hubble distance in Mpc
    D_C_grid = D_H * cumulative_trapezoid(integrand, z_grid, initial=0.0)

    # Interpolate to requested z values
    D_C = np.interp(z, z_grid, D_C_grid)

    return float(D_C[0]) if scalar_input else D_C


def transverse_comoving_distance(
    z: float | np.ndarray,
    Omega_m: float,
    Omega_de: float,
    H0: float,
    n_steps: int = 2048,
) -> float | np.ndarray:
    """
    Transverse comoving distance D_M(z) in Mpc, corrected for spatial curvature.

    Curvature branches (Omega_k = 1 - Omega_m - Omega_de):
      - Omega_k > 0  (open):   D_M = D_H/sqrt(Omega_k) * sinh(sqrt(Omega_k)*D_C/D_H)
      - Omega_k == 0 (flat):   D_M = D_C
      - Omega_k < 0  (closed): D_M = D_H/sqrt(|Omega_k|) * sin(sqrt(|Omega_k|)*D_C/D_H)

    Args:
        z:        Redshift, scalar or array.
        Omega_m:  Present-day matter density parameter.
        Omega_de: Present-day dark-energy density parameter.
        H0:       Hubble constant in km/s/Mpc.
        n_steps:  Passed through to comoving_distance.

    Returns:
        D_M in Mpc, same shape as z.
    """
    scalar_input = np.ndim(z) == 0
    z = np.atleast_1d(np.asarray(z, dtype=float))

    D_C = np.atleast_1d(comoving_distance(z, Omega_m, Omega_de, H0, n_steps=n_steps))
    D_H = _c_km_s() / H0
    Omega_k = 1.0 - Omega_m - Omega_de

    if abs(Omega_k) < 1e-8:
        # Flat
        D_M = D_C
    elif Omega_k > 0.0:
        # Open: sinh branch
        sqrt_Ok = np.sqrt(Omega_k)
        D_M = (D_H / sqrt_Ok) * np.sinh(sqrt_Ok * D_C / D_H)
    else:
        # Closed: sin branch
        sqrt_Okabs = np.sqrt(abs(Omega_k))
        D_M = (D_H / sqrt_Okabs) * np.sin(sqrt_Okabs * D_C / D_H)

    return float(D_M[0]) if scalar_input else D_M


def luminosity_distance(
    z: float | np.ndarray,
    Omega_m: float,
    Omega_de: float,
    H0: float,
    n_steps: int = 2048,
) -> float | np.ndarray:
    """
    Luminosity distance d_L(z) in Mpc.

    d_L = (1 + z) * D_M

    Args:
        z:        Redshift, scalar or array.
        Omega_m:  Present-day matter density parameter.
        Omega_de: Present-day dark-energy density parameter.
        H0:       Hubble constant in km/s/Mpc.
        n_steps:  Passed through to distance integrals.

    Returns:
        d_L in Mpc, same shape as z.
    """
    scalar_input = np.ndim(z) == 0
    z = np.atleast_1d(np.asarray(z, dtype=float))

    D_M = np.atleast_1d(
        transverse_comoving_distance(z, Omega_m, Omega_de, H0, n_steps=n_steps)
    )
    d_L = (1.0 + z) * D_M

    return float(d_L[0]) if scalar_input else d_L


def distance_modulus(
    z: float | np.ndarray,
    Omega_m: float,
    Omega_de: float,
    H0: float,
    n_steps: int = 2048,
) -> float | np.ndarray:
    """
    Distance modulus mu(z) in magnitudes.

    mu = 5 * log10(d_L / Mpc) + 25

    Edge case: z <= 0 gives d_L = 0 -> log10 undefined. Those entries are
    returned as -inf (consistent with numpy's log10(0) behaviour).  Callers
    that work with real SNe Ia should use z > 0.

    Args:
        z:        Redshift, scalar or array.
        Omega_m:  Present-day matter density parameter.
        Omega_de: Present-day dark-energy density parameter.
        H0:       Hubble constant in km/s/Mpc.
        n_steps:  Passed through to distance integrals.

    Returns:
        mu in magnitudes, same shape as z.
    """
    scalar_input = np.ndim(z) == 0
    z = np.atleast_1d(np.asarray(z, dtype=float))

    d_L = np.atleast_1d(luminosity_distance(z, Omega_m, Omega_de, H0, n_steps=n_steps))

    # Suppress divide-by-zero warning; result is -inf which is the correct sentinel
    with np.errstate(divide="ignore"):
        mu = 5.0 * np.log10(d_L) + 25.0

    return float(mu[0]) if scalar_input else mu


# ---------------------------------------------------------------------------
# Model-curve helper
# ---------------------------------------------------------------------------

_VALID_MODELS = {"external_node", "lcdm", "matter_only"}


def model_distance_modulus(
    z: float | np.ndarray,
    model: str,
    sim_params: SimulationParameters | None = None,
    H0: float | None = None,
) -> np.ndarray:
    """
    Distance modulus mu(z) for one of three named cosmological models.

    Models
    ------
    "lcdm"
        Standard ΛCDM: Omega_m = 0.3, Omega_de = 0.7.
    "matter_only"
        Matter-only (open): Omega_m = 0.3, Omega_de = 0.0, Omega_k = 0.7.
        Uses the sinh curvature branch.
    "external_node"
        External-Node Model: Omega_m = 0.3,
        Omega_de = sim_params.external_params.Omega_Lambda_eff.
        Requires sim_params; raises ValueError if None.

    H0 defaults to LambdaCDMParameters().H0_km_s_Mpc (70.0 km/s/Mpc).
    The absolute offset set by H0 is later marginalized in the Hubble-diagram
    fit (Section 3), so H0 here only controls a constant additive shift in mu.

    Args:
        z:          Redshift, scalar or array.
        model:      One of "lcdm", "matter_only", "external_node".
        sim_params: Required for "external_node". Ignored otherwise.
        H0:         Hubble constant in km/s/Mpc. Defaults to 70.0.

    Returns:
        mu in magnitudes, as np.ndarray aligned to z.
    """
    if model not in _VALID_MODELS:
        raise ValueError(f"model must be one of {sorted(_VALID_MODELS)}, got {model!r}")

    lcdm_defaults = LambdaCDMParameters()
    if H0 is None:
        H0 = lcdm_defaults.H0_km_s_Mpc

    Omega_m = lcdm_defaults.Omega_m  # 0.3 for all models

    if model == "lcdm":
        Omega_de = lcdm_defaults.Omega_Lambda  # 0.7

    elif model == "matter_only":
        Omega_de = 0.0

    else:  # "external_node"
        if sim_params is None:
            raise ValueError(
                "sim_params is required for model='external_node'. "
                "Pass a SimulationParameters instance."
            )
        Omega_de = sim_params.external_params.Omega_Lambda_eff

    mu = distance_modulus(z, Omega_m, Omega_de, H0)
    return np.atleast_1d(np.asarray(mu))
