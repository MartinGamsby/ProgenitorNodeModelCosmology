"""
Hubble-Diagram Comparison Engine

Implements offset marginalization + chi^2/R^2 statistics for comparing
cosmological models against real Pantheon+SH0ES supernova data.

The key physics: SN distance moduli carry an unknown absolute offset
(M_B + 5*log10(c/H0) + 25) that is degenerate with the Hubble constant.
For a fair model comparison, this constant offset is fit analytically
by inverse-variance-weighted least squares for each model independently,
then chi^2 and R^2 are computed on the residuals after offset subtraction.
This tests the SHAPE of H(z) and marginalizes the M_B/H0 calibration
degeneracy.

No I/O, no plotting. Pure numpy math on in-memory arrays.
"""

import numpy as np

from .analysis import calculate_r_squared
from .constants import SimulationParameters
from .distances import model_distance_modulus


# ---------------------------------------------------------------------------
# Core analytic functions
# ---------------------------------------------------------------------------

def fit_offset(
    mu_obs: np.ndarray,
    mu_model: np.ndarray,
    sigma: np.ndarray,
) -> tuple[float, np.ndarray]:
    """
    Fit the inverse-variance-weighted additive offset between observed and
    model distance moduli.

    The offset DeltaM is the single nuisance parameter that absorbs the
    M_B + 5*log10(c/H0) + 25 calibration degeneracy.  Its best-fit value
    is the weighted mean of (mu_obs - mu_model):

        w_i = 1 / sigma_i^2
        DeltaM = sum(w_i * (mu_obs_i - mu_model_i)) / sum(w_i)

    This is the analytic (closed-form) solution to min_DeltaM chi^2 —
    no numerical optimizer is needed.

    Args:
        mu_obs:   Observed distance moduli, shape (n,).
        mu_model: Model-predicted distance moduli, same shape as mu_obs.
        sigma:    Diagonal uncertainties (1-sigma), strictly positive,
                  same shape as mu_obs.

    Returns:
        (DeltaM, mu_fit) where:
            DeltaM  – float, the best-fit additive offset in magnitudes.
            mu_fit  – np.ndarray, mu_model + DeltaM (offset-corrected model).

    Raises:
        ValueError: If any sigma <= 0, or arrays have mismatched lengths,
                    or fewer than 2 data points.
    """
    mu_obs = np.asarray(mu_obs, dtype=float)
    mu_model = np.asarray(mu_model, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    _validate_arrays(mu_obs, mu_model, sigma)

    w = 1.0 / sigma ** 2
    residuals = mu_obs - mu_model
    DeltaM = float(np.sum(w * residuals) / np.sum(w))

    mu_fit = mu_model + DeltaM
    return DeltaM, mu_fit


def evaluate_model(
    z: np.ndarray,
    mu_obs: np.ndarray,
    sigma: np.ndarray,
    model: str,
    sim_params: SimulationParameters | None = None,
    H0: float = 70.0,
) -> dict:
    """
    Evaluate one cosmological model against observed distance moduli.

    Steps:
        1. Compute mu_model(z) via distances.model_distance_modulus.
        2. Fit the constant offset DeltaM analytically (fit_offset).
        3. Compute chi^2, dof, reduced chi^2, R^2, and residuals.

    The offset is fit and applied per-model so that external_node, lcdm,
    and matter_only are all compared fairly against the same observed data —
    only the SHAPE of each model's H(z) determines the outcome.

    Turnaround guard: if model_distance_modulus raises ValueError (e.g.
    closed external_node model with E^2 < 0 over part of the z range),
    the error is caught and re-raised with a descriptive message that
    surfaces the issue rather than silently crashing.

    Args:
        z:          Redshift array (n,), sorted ascending, all > 0.
        mu_obs:     Observed distance moduli (n,).
        sigma:      Diagonal 1-sigma uncertainties (n,), strictly positive.
        model:      One of 'lcdm', 'matter_only', 'external_node'.
        sim_params: Required when model='external_node'. Ignored otherwise.
        H0:         Hubble constant used to compute mu_model, in km/s/Mpc.
                    The additive H0-dependent offset is absorbed into DeltaM.

    Returns:
        dict with keys:
            'model'     – str, the model name.
            'DeltaM'    – float, the best-fit additive offset (magnitudes).
            'chi2'      – float, chi-squared after offset subtraction.
            'dof'       – int, degrees of freedom = n - 1.
            'chi2_dof'  – float, reduced chi-squared = chi2 / dof.
            'R2'        – float, coefficient of determination in [0, 1].
                          Computed as calculate_r_squared(mu_obs, mu_fit).
            'residuals' – np.ndarray (n,), mu_obs - mu_fit.
            'mu_fit'    – np.ndarray (n,), mu_model + DeltaM.

    Raises:
        ValueError: Propagated from fit_offset or model_distance_modulus
                    (including turnaround in closed models).
    """
    z = np.asarray(z, dtype=float)
    mu_obs = np.asarray(mu_obs, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    # Explicit empty-data guard so an empty array (e.g. an over-aggressive
    # z_min cut leaving no SNe) yields an actionable message instead of the
    # turnaround error model_distance_modulus would otherwise raise on np.max
    # of a zero-size array.
    if len(z) == 0:
        raise ValueError(
            "No data points to evaluate (empty z array). "
            "Check the z_min cut and the loaded Pantheon+ data."
        )

    # Compute model distance moduli — may raise ValueError for turnaround
    try:
        mu_model = model_distance_modulus(z, model, sim_params=sim_params, H0=H0)
    except ValueError as exc:
        raise ValueError(
            f"model_distance_modulus failed for model={model!r} "
            f"(Omega_Lambda_eff may cause turnaround within the data z-range). "
            f"Original error: {exc}"
        ) from exc

    DeltaM, mu_fit = fit_offset(mu_obs, mu_model, sigma)

    residuals = mu_obs - mu_fit
    chi2 = float(np.sum((residuals / sigma) ** 2))
    n = len(mu_obs)
    dof = n - 1  # one free parameter: DeltaM
    chi2_dof = chi2 / dof
    R2 = float(calculate_r_squared(mu_obs, mu_fit))

    return {
        "model": model,
        "DeltaM": DeltaM,
        "chi2": chi2,
        "dof": dof,
        "chi2_dof": chi2_dof,
        "R2": R2,
        "residuals": residuals,
        "mu_fit": mu_fit,
    }


def compare_all_models(
    data: dict,
    sim_params: SimulationParameters | None = None,
    H0: float = 70.0,
) -> dict:
    """
    Compare all three cosmological models against Pantheon+SH0ES data.

    Evaluates each model using offset marginalization so chi^2 reflects only
    the SHAPE of H(z), not the absolute calibration.  Each model receives its
    own independently fit DeltaM.

    Models compared:
        'external_node' — requires sim_params (uses Omega_Lambda_eff from it).
        'lcdm'          — standard flat ΛCDM (Omega_m=0.3, Omega_Lambda=0.7).
        'matter_only'   — matter-only open universe (Omega_m=0.3, Omega_de=0.0).

    If a model's H(z) is undefined over part of the z-range (e.g. closed
    external_node with a turnaround), evaluate_model raises a descriptive
    ValueError that is propagated to the caller — it is surfaced rather than
    silently skipped.

    Args:
        data:       dict from load_pantheon() with keys 'z', 'mu', 'sigma', 'n'.
        sim_params: SimulationParameters instance for the external_node model.
                    Required; if None, external_node evaluation will raise.
        H0:         Hubble constant for model distance moduli, km/s/Mpc.

    Returns:
        dict with keys 'external_node', 'lcdm', 'matter_only', each mapping
        to the result dict from evaluate_model().
    """
    z = data["z"]
    mu_obs = data["mu"]
    sigma = data["sigma"]

    results = {}
    for model in ("external_node", "lcdm", "matter_only"):
        results[model] = evaluate_model(
            z, mu_obs, sigma,
            model=model,
            sim_params=sim_params,
            H0=H0,
        )
    return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_arrays(
    mu_obs: np.ndarray,
    mu_model: np.ndarray,
    sigma: np.ndarray,
) -> None:
    """Validate input arrays for fit_offset / evaluate_model."""
    n = len(mu_obs)

    if n < 2:
        raise ValueError(
            f"At least 2 data points are required (got {n}). "
            "dof = n - 1 must be >= 1."
        )
    if len(mu_model) != n:
        raise ValueError(
            f"mu_model length {len(mu_model)} != mu_obs length {n}."
        )
    if len(sigma) != n:
        raise ValueError(
            f"sigma length {len(sigma)} != mu_obs length {n}."
        )
    if np.any(sigma <= 0):
        bad = int(np.sum(sigma <= 0))
        raise ValueError(
            f"{bad} sigma value(s) are <= 0. All uncertainties must be strictly positive."
        )
