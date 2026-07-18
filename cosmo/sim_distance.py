"""
Simulation-Based Distance Kernel

Pure-function module that converts a simulation's scale-factor history a(t)
into a distance-modulus curve mu(z) on a requested redshift grid.

This is the seam that replaces the circular constant-Omega_Lambda_eff shortcut:
instead of model_distance_modulus(z, "external_node") — which uses the analytic
Omega_Lambda_eff and is numerically identical to LCDM — the model curve comes
from the real N-body a(t).

Physics:
  D_C(t_i) = c * integral_{t_i}^{t_today} dt / a(t)   [flat; no curvature term]
  d_L      = (1 + z) * D_C
  mu       = 5 * log10(d_L / Mpc) + 25

H0 is NOT needed: integrating physical c*dt/a in meters already carries the
correct scale from the simulation's time axis. The offset marginalization in
evaluate_model removes any constant mu shift; only the SHAPE matters.

No I/O, no plotting, no simulation execution. Pure numpy/scipy math.
"""

import numpy as np
from scipy.integrate import cumulative_trapezoid

from .constants import CosmologicalConstants

# ---------------------------------------------------------------------------
# Module-level constants (derived once)
# ---------------------------------------------------------------------------

_C_M_S: float = CosmologicalConstants.c          # speed of light [m/s]
_GYR_TO_S: float = CosmologicalConstants.Gyr_to_s  # 3.1536e16 s/Gyr
_MPC_TO_M: float = CosmologicalConstants.Mpc_to_m  # 3.0857e22 m/Mpc

_TODAY_GYR: float = 13.8   # nominal age of the universe in Gyr
_TODAY_TOL_DEFAULT: float = 0.2  # default tolerance in Gyr


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def sim_to_distance_modulus(
    z_target: np.ndarray,
    a: np.ndarray,
    t_Gyr: np.ndarray,
    t_start_Gyr: float,
    today_index: int = -1,
    today_tol_Gyr: float = _TODAY_TOL_DEFAULT,
) -> dict:
    """
    Convert a simulation's scale-factor history to a distance-modulus curve.

    Inputs come from the simulation's expansion history (see
    cosmo.factories.run_external_node_simulation return dict):
      - ``a``:           Scale factor array, normalized so a[0] = 1 at t_start.
                         Length = n_snapshots.
      - ``t_Gyr``:       Relative time array starting at 0.0 (sim frame), Gyr.
      - ``t_start_Gyr``: Absolute start time, Gyr (e.g. 5.8).

    Steps:
      1. Build absolute time in seconds.
      2. Assert the snapshot at ``today_index`` is within ``today_tol_Gyr``
         of 13.8 Gyr; raise ValueError otherwise — the caller must run the sim
         so that t_start_Gyr + t_duration_Gyr == 13.8.
      3. Renormalize: a_today = a / a[today_index], so a_today[today_index] = 1.
      4. Compute per-snapshot redshift: z_snap = 1/a_today - 1  (z=0 today).
      5. Comoving distance by direct integration (no differentiation of a):
         D_C(t_i) = c * integral_{t_i}^{t_today} dt / a_today(t).
         Implemented with scipy.integrate.cumulative_trapezoid on the snapshot
         grid integrating (1/a_today) over t_abs_s.
      6. Luminosity distance (flat): d_L = (1 + z_snap) * D_C_Mpc.
      7. Distance modulus at snapshots: mu = 5*log10(d_L) + 25  (d_L in Mpc).
      8. Interpolate onto z_target; clip to the sim-covered z range (no
         extrapolation).

    H0 note: D_C is computed as c * integral(dt/a), which already carries the
    correct physical scale from the sim's time axis. H0 is therefore NOT a
    parameter. The absolute mu offset is marginalized in the Hubble-diagram fit
    (evaluate_model), so only the shape of mu(z) matters.

    Args:
        z_target:       Redshift values at which to evaluate mu (1-D array).
        a:              Scale factor array from the simulation (a[0] = 1 at
                        t_start, NOT normalized to today). Length n_snapshots.
        t_Gyr:          Relative time array, starting at 0.0, in Gyr.
                        Length must equal len(a).
        t_start_Gyr:    Absolute start time of the simulation in Gyr.
        today_index:    Snapshot index corresponding to "today" (default -1,
                        i.e. the last snapshot).
        today_tol_Gyr:  Tolerance in Gyr for the "today" assertion
                        (default 0.2 Gyr).

    Returns:
        dict with keys:
          ``'z'``        : z_target values within the sim-covered range
                           (ascending, 1-D array).
          ``'mu'``       : Model distance modulus at those z (1-D array, mag).
          ``'in_range'`` : Boolean mask aligned to the original z_target
                           (True where the z value was in range).
          ``'z_cover'``  : Tuple (z_min_cover, z_max_cover) — the z range the
                           sim covers (excludes z=0 today point).
          ``'a_today'``  : Scale factor renormalized so a_today[today_index]=1
                           (diagnostic).
          ``'z_snap'``   : Per-snapshot redshift array (diagnostic).

    Raises:
        ValueError: If the absolute time at ``today_index`` differs from 13.8
                    Gyr by more than ``today_tol_Gyr``; or if all z_target
                    values fall outside the sim-covered range.
    """
    # ---- Input coercion ----
    a = np.asarray(a, dtype=float)
    t_Gyr = np.asarray(t_Gyr, dtype=float)
    z_target = np.atleast_1d(np.asarray(z_target, dtype=float))

    if a.ndim != 1 or t_Gyr.ndim != 1:
        raise ValueError("a and t_Gyr must be 1-D arrays.")
    if len(a) != len(t_Gyr):
        raise ValueError(
            f"a (len={len(a)}) and t_Gyr (len={len(t_Gyr)}) must have the same length."
        )

    # ---- Step 1: Absolute time in seconds ----
    t_abs_Gyr = t_start_Gyr + t_Gyr        # absolute time in Gyr
    t_abs_s = t_abs_Gyr * _GYR_TO_S        # absolute time in seconds

    # ---- Step 2: Assert "today" snapshot ----
    t_today_abs_Gyr = t_abs_Gyr[today_index]
    if abs(t_today_abs_Gyr - _TODAY_GYR) > today_tol_Gyr:
        raise ValueError(
            f"The snapshot at today_index={today_index} has absolute time "
            f"{t_today_abs_Gyr:.4f} Gyr, which differs from the expected "
            f"{_TODAY_GYR} Gyr by {abs(t_today_abs_Gyr - _TODAY_GYR):.4f} Gyr "
            f"(tolerance={today_tol_Gyr} Gyr). "
            f"Ensure t_start_Gyr + t_duration_Gyr == {_TODAY_GYR}. "
            f"Current: t_start_Gyr={t_start_Gyr}, last t_abs={t_today_abs_Gyr:.4f}."
        )

    # ---- Step 3: Renormalize a to today ----
    a_at_today = a[today_index]
    if a_at_today <= 0:
        raise ValueError(
            f"a[today_index={today_index}] = {a_at_today} is non-positive; "
            "cannot renormalize."
        )
    a_today = a / a_at_today   # a_today[today_index] == 1.0 exactly

    # ---- Step 4: Redshift at each snapshot ----
    # Guard against a_today <= 0 (should not happen for valid sims)
    if np.any(a_today <= 0):
        raise ValueError(
            "Renormalized scale factor a_today contains non-positive values. "
            "Check the simulation output."
        )
    z_snap = 1.0 / a_today - 1.0   # z=0 at today, increases toward the past

    # ---- Step 5: Comoving distance D_C(t_i) = c * integral_{t_i}^{t_today} dt/a ----
    # cumulative_trapezoid integrates from index 0 forward.
    # D_C(t_i) = c * [integral_0^{t_today} (1/a)dt  -  integral_0^{t_i} (1/a)dt]
    #           = c * [cum[today_index] - cum[i]]
    integrand = 1.0 / a_today   # 1/a_today at each snapshot

    # Cumulative integral from index 0 (initial=0 pads so output has same length as input)
    cum_integral = cumulative_trapezoid(integrand, t_abs_s, initial=0.0)

    cum_at_today = cum_integral[today_index]
    D_C_m = _C_M_S * (cum_at_today - cum_integral)   # meters; positive for t < t_today

    # Convert to Mpc
    D_C_Mpc = D_C_m / _MPC_TO_M

    # ---- Step 6: Luminosity distance (flat cosmology) ----
    d_L_Mpc = (1.0 + z_snap) * D_C_Mpc

    # ---- Step 7: Distance modulus ----
    # Suppress divide-by-zero at z=0 (today); result is -inf, excluded below.
    with np.errstate(divide="ignore", invalid="ignore"):
        mu_snap = 5.0 * np.log10(d_L_Mpc) + 25.0

    # ---- Step 8: Sort by z (ascending), exclude today point (z=0, mu=-inf) ----
    # Identify positive-z snapshots (exclude today and any future points)
    valid_mask = (z_snap > 0.0) & np.isfinite(mu_snap)
    if not np.any(valid_mask):
        raise ValueError(
            "No valid (z > 0, finite mu) snapshots found. "
            "The simulation may not cover any redshift range before today."
        )

    z_valid = z_snap[valid_mask]
    mu_valid = mu_snap[valid_mask]

    # Sort ascending in z
    sort_idx = np.argsort(z_valid)
    z_sorted = z_valid[sort_idx]
    mu_sorted = mu_valid[sort_idx]

    # Coverage
    z_min_cover = float(z_sorted[0])
    z_max_cover = float(z_sorted[-1])

    # ---- Step 9: Clip z_target to covered range and interpolate ----
    in_range = (z_target >= z_min_cover) & (z_target <= z_max_cover)
    z_in_range = z_target[in_range]

    if len(z_in_range) == 0:
        raise ValueError(
            f"All z_target values fall outside the sim-covered range "
            f"[{z_min_cover:.4f}, {z_max_cover:.4f}]. "
            f"z_target range: [{z_target.min():.4f}, {z_target.max():.4f}]."
        )

    mu_model = np.interp(z_in_range, z_sorted, mu_sorted)

    return {
        "z": z_in_range,
        "mu": mu_model,
        "in_range": in_range,
        "z_cover": (z_min_cover, z_max_cover),
        "a_today": a_today,
        "z_snap": z_snap,
    }
