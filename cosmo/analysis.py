"""
Cosmology Analysis Utilities

Shared functions for:
- Solving Friedmann equations (ΛCDM, matter-only)
- Computing initial conditions
- Comparing expansion histories
- Detecting numerical instabilities
"""

from typing import Optional, Dict
import numpy as np
from scipy.integrate import odeint
from scipy.ndimage import gaussian_filter1d
from .constants import CosmologicalConstants, LambdaCDMParameters


def friedmann_equation(a: float, t: float, H0_si: float, Omega_m: float, Omega_Lambda: float) -> float:
    """
    Friedmann equation: da/dt = H(a) * a where H(a) = H0_si * sqrt(Ω_m/a³ + Ω_Λ)

    Note: t parameter unused but required by odeint signature.
    """
    if a <= 0:
        return 1e-10
    H_si = H0_si * np.sqrt(Omega_m / a**3 + Omega_Lambda)
    return H_si * a


def solve_friedmann_at_times(t_array_Gyr: np.ndarray, Omega_Lambda: Optional[float] = None) -> Dict[str, np.ndarray]:
    """
    Solve Friedmann equation at specific time points.

    Ensures exact time alignment with N-body simulations by evaluating
    the analytic solution at the exact same times as simulation snapshots.
    This eliminates the need to force t[0]=0 or interpolate misaligned grids.

    If Omega_Lambda is None, uses ΛCDM value. Set to 0.0 for matter-only.
    """
    const = CosmologicalConstants()
    lcdm = LambdaCDMParameters()

    if Omega_Lambda is None:
        Omega_Lambda = lcdm.Omega_Lambda

    # Solve from Big Bang to max requested time with fine resolution for interpolation
    t_max_Gyr = np.max(t_array_Gyr) + 1.0  # Add buffer
    n_solve = 1000  # Fine resolution for accurate interpolation
    t_solve = np.linspace(0, t_max_Gyr * 1e9 * 365.25 * 24 * 3600, n_solve)

    a0 = 0.001
    a_solve = odeint(
        friedmann_equation, a0, t_solve,
        args=(lcdm.H0_si, lcdm.Omega_m, Omega_Lambda)
    ).flatten()

    t_solve_Gyr = t_solve / (1e9 * 365.25 * 24 * 3600)

    # Interpolate to requested time points
    a = np.interp(t_array_Gyr, t_solve_Gyr, a_solve)

    # Calculate Hubble parameter at requested times
    H_si = lcdm.H0_si * np.sqrt(lcdm.Omega_m / a**3 + Omega_Lambda)
    H_hubble = H_si * const.Mpc_to_m / 1000  # Convert to km/s/Mpc

    return {
        't_Gyr': t_array_Gyr,
        'a': a,
        'H_hubble': H_hubble
    }


def solve_friedmann_equation(t_start_Gyr: float, t_end_Gyr: float, Omega_Lambda: Optional[float] = None, n_points: int = 400) -> Dict[str, np.ndarray]:
    """
    Solve Friedmann equation for ΛCDM or matter-only cosmology.

    Note: For exact alignment with N-body simulations, use solve_friedmann_at_times() instead.

    If Omega_Lambda is None, uses ΛCDM value. Set to 0.0 for matter-only.
    """
    const = CosmologicalConstants()
    lcdm = LambdaCDMParameters()

    if Omega_Lambda is None:
        Omega_Lambda = lcdm.Omega_Lambda

    # Solve from early time with buffer
    a0 = 0.001
    t_max = t_end_Gyr + 3.0  # Buffer beyond end time
    t_span = np.linspace(0, t_max * 1e9 * 365.25 * 24 * 3600, n_points)

    a_solution = odeint(
        friedmann_equation, a0, t_span,
        args=(lcdm.H0_si, lcdm.Omega_m, Omega_Lambda)
    )
    a_solution = a_solution.flatten()

    # Convert time to Gyr
    t_Gyr_full = t_span / (1e9 * 365.25 * 24 * 3600)

    # Extract requested window
    mask = (t_Gyr_full >= t_start_Gyr) & (t_Gyr_full <= t_end_Gyr)
    t_Gyr = t_Gyr_full[mask]
    a = a_solution[mask]

    # Calculate Hubble parameter
    H_si = lcdm.H0_si * np.sqrt(lcdm.Omega_m / a**3 + Omega_Lambda)
    H_hubble = H_si * const.Mpc_to_m / 1000  # Convert to km/s/Mpc

    return {
        't_Gyr': t_Gyr,
        'a': a,
        'H_hubble': H_hubble,
        '_t_Gyr_full': t_Gyr_full,
        '_a_full': a_solution
    }


def calculate_initial_conditions(t_start_Gyr: float, reference_size_today_Gpc: float = 14.5) -> Dict[str, float]:
    """
    Calculate initial scale factor and box size at given start time.

    Uses ΛCDM cosmology to determine scale factor at t_start,
    then computes box size by scaling from present-day reference.
    """
    const = CosmologicalConstants()
    lcdm = LambdaCDMParameters()

    # Solve ΛCDM to get scale factors at both t_start and t_today=13.8 Gyr
    # Must solve to at least 13.8 Gyr to get accurate a_today
    t_end_solve = max(t_start_Gyr, 13.8) + 1.0  # Add buffer
    solution = solve_friedmann_equation(0.0, t_end_solve, n_points=400)

    # Find scale factor at t_start
    idx_start = np.argmin(np.abs(solution['_t_Gyr_full'] - t_start_Gyr))
    a_start = solution['_a_full'][idx_start]

    # Find scale factor today (13.8 Gyr)
    idx_today = np.argmin(np.abs(solution['_t_Gyr_full'] - 13.8))
    a_today = solution['_a_full'][idx_today]

    # Scale box size from today
    box_size_Gpc = reference_size_today_Gpc * (a_start / a_today)

    # Calculate Hubble parameter at start
    H_start = lcdm.H_at_time(a_start)
    H_start_hubble = H_start * const.Mpc_to_m / 1000

    return {
        'a_start': a_start,
        'box_size_Gpc': box_size_Gpc,
        'H_start_hubble': H_start_hubble
    }


def normalize_to_initial_size(a_array: np.ndarray, initial_size_Gpc: float) -> np.ndarray:
    """Convert scale factor array to physical size array in Gpc."""
    a_normalized = a_array / a_array[0]
    return initial_size_Gpc * a_normalized


def calculate_r_squared(y_actual, y_predicted):
    """
    Calculate R² (coefficient of determination).

    R² = 1 - (SS_res / SS_tot)
    where SS_res = sum of squared residuals
          SS_tot = total sum of squares

    Args:
        y_actual: Reference values (ΛCDM baseline)
        y_predicted: Model values (External-Node model)

    Returns:
        R² value (float):
        - 1.0 = perfect fit
        - 0.0 = model explains no variance (predicts mean)
        - Negative = worse than predicting mean
        - Edge case: if y_actual is constant, returns 1.0 if match else 0.0

    Physics interpretation: Fraction of ΛCDM variance explained by External-Node model.
    """
    y_actual = np.asarray(y_actual)
    y_predicted = np.asarray(y_predicted)

    # Handle scalar inputs (convert to 1-element array for uniform processing)
    if y_actual.ndim == 0:
        y_actual = y_actual.reshape(1)
    if y_predicted.ndim == 0:
        y_predicted = y_predicted.reshape(1)

    # Calculate residuals
    residuals = y_actual - y_predicted
    ss_res = np.sum(residuals**2)

    # Calculate total sum of squares
    y_mean = np.mean(y_actual)
    ss_tot = np.sum((y_actual - y_mean)**2)

    # Edge case: constant y_actual (SS_tot = 0)
    if ss_tot == 0:
        if ss_res == 0:
            return 1.0  # Perfect match of constant curves
        else:
            return 0.0  # Curves differ but baseline is constant

    # Standard R² calculation
    r_squared = 1.0 - (ss_res / ss_tot)
    return r_squared

def compare_expansion_history(size1, size):
    diff = np.abs(size1 - size) / size * 100
    return 100 - diff

def compare_expansion_histories(size_ext, size_lcdm, return_array: bool = False,
                                  use_r_squared: bool = True, r_square_times_100: bool = True,
                                    return_diagnostics: bool = False):
    """
    Calculate match quality between two expansion histories.

    Uses R² (coefficient of determination) by default. Option to use legacy
    percentage match for backward compatibility.

    Returns:
        - If return_diagnostics=False: scalar R² (default) or percentage match
        - If return_diagnostics=True: dict with R², errors, RMSE
        - If return_array=True: per-timestep percentage error array
    """
    # Convert to arrays for uniform processing
    is_array = isinstance(size_ext, np.ndarray) and isinstance(size_lcdm, np.ndarray)

    if not is_array:
        size_ext = np.asarray([size_ext])
        size_lcdm = np.asarray([size_lcdm])

    # Calculate percentage errors (used for diagnostics and backward compat)
    diff_pct = np.abs(size_ext - size_lcdm) / size_lcdm * 100
    match_pct_array = 100 - diff_pct

    # Per-timestep array output
    if return_array and is_array:
        return match_pct_array

    # Calculate metrics
    mean_error_pct = np.mean(diff_pct)
    max_error_pct = np.max(diff_pct)
    match_pct_scalar = np.mean(match_pct_array)

    # RMSE
    rmse = np.sqrt(np.mean((size_ext - size_lcdm)**2))
    rmse_pct = rmse / np.mean(size_lcdm) * 100

    # R² or percentage match
    if use_r_squared:
        r_squared = calculate_r_squared(size_lcdm, size_ext)
        r_squared = r_squared * 100 if r_square_times_100 else r_squared
        primary_metric = r_squared
    else:
        primary_metric = match_pct_scalar

    # Return diagnostics if requested
    if return_diagnostics:
        diagnostics = {
            'r_squared': r_squared if use_r_squared else None,
            'match_pct': match_pct_scalar if not use_r_squared else None,
            'max_error_pct': max_error_pct,
            'mean_error_pct': mean_error_pct,
            'rmse': rmse,
            'rmse_pct': rmse_pct
        }
        return diagnostics
    return primary_metric


def detect_runaway_particles(max_distance_Gpc: float, rms_size_Gpc: float, threshold: float = 2.0) -> Dict[str, float]:
    """
    Detect runaway particles indicating numerical instability.

    When max particle distance >> RMS size, it indicates particles
    being "shot out" due to leapfrog instability or force errors.
    """
    ratio = max_distance_Gpc / rms_size_Gpc
    detected = ratio > threshold

    return {
        'detected': detected,
        'ratio': ratio,
        'threshold': threshold
    }


def calculate_today_marker(t_start_Gyr: float, t_duration_Gyr: float, today_Gyr: float = 13.8) -> Optional[float]:
    """
    Calculate position of "today" marker in simulation time coordinates.

    Returns time coordinate for "today" in simulation frame, or None if
    today is outside simulation window.
    """
    t_end_Gyr = t_start_Gyr + t_duration_Gyr

    if t_start_Gyr < today_Gyr < t_end_Gyr:
        return today_Gyr - t_start_Gyr
    return None


def extract_expansion_history(sim, key: str) -> np.ndarray:
    """Extract a specific field from simulation expansion history as numpy array."""
    import numpy as np
    return np.array([h[key] for h in sim.expansion_history])


def check_com_drift_quality(expansion_history: list, drift_threshold: float = 0.5) -> Dict[str, float]:
    """
    Detect excessive center-of-mass drift as simulation quality metric.

    Large COM drift indicates asymmetric tidal forces from external nodes,
    suggesting problematic simulation parameters (M_ext_kg too large and/or
    S too small). The drift is physically valid but indicates the system
    is being pulled strongly toward external nodes.

    Drift > threshold × final_rms indicates problematic parameters.
    """
    # Extract COM positions over time (stored as 3D vectors in meters)
    com_positions = np.array([h['com'] for h in expansion_history])

    # Calculate total drift from initial to final position
    initial_com = com_positions[0]
    final_com = com_positions[-1]
    com_drift_m = np.linalg.norm(final_com - initial_com)

    # Convert to Gpc
    const = CosmologicalConstants()
    com_drift_Gpc = com_drift_m / const.Gpc_to_m

    # Get final RMS size (stored in meters, diameter = 2*RMS)
    final_diameter_m = expansion_history[-1]['diameter_m']
    final_rms_m = final_diameter_m / 2.0
    final_rms_Gpc = final_rms_m / const.Gpc_to_m

    # Calculate ratio
    drift_to_size_ratio = com_drift_Gpc / final_rms_Gpc

    # Detect excessive drift
    is_excessive = drift_to_size_ratio > drift_threshold

    return {
        'com_drift_Gpc': com_drift_Gpc,
        'final_rms_Gpc': final_rms_Gpc,
        'drift_to_size_ratio': drift_to_size_ratio,
        'is_excessive': is_excessive,
        'threshold': drift_threshold
    }

def calculate_hubble_parameters(t_ext, a_ext, smooth_sigma=0.0):
    """
    Calculate Hubble parameters from scale factors.
    """
    const = CosmologicalConstants()

    # Optional smoothing (default: no smoothing per user request)
    if smooth_sigma > 0:
        a_smooth = gaussian_filter1d(a_ext, sigma=smooth_sigma)
    else:
        a_smooth = a_ext

    # External-Node Hubble parameter
    H_ext = np.gradient(a_smooth, t_ext * 1e9 * 365.25 * 24 * 3600) / a_smooth
    H_hubble = H_ext * const.Mpc_to_m / 1000

    # Fix boundary points: np.gradient uses forward/backward differences at edges
    # which are less accurate. Replace first and last points with NaN to exclude them
    # from plots, or use second-order accurate formulas
    if len(H_hubble) > 2:
        # Second-order forward difference for first point: f'(0) ≈ (-3f(0) + 4f(1) - f(2)) / (2h)
        dt_0 = (t_ext[1] - t_ext[0]) * 1e9 * 365.25 * 24 * 3600
        H_hubble[0] = (-3*a_smooth[0] + 4*a_smooth[1] - a_smooth[2]) / (2*dt_0 * a_smooth[0]) * const.Mpc_to_m / 1000

        # Second-order backward difference for last point: f'(n) ≈ (3f(n) - 4f(n-1) + f(n-2)) / (2h)
        dt_n = (t_ext[-1] - t_ext[-2]) * 1e9 * 365.25 * 24 * 3600
        H_hubble[-1] = (3*a_smooth[-1] - 4*a_smooth[-2] + a_smooth[-3]) / (2*dt_n * a_smooth[-1]) * const.Mpc_to_m / 1000

    return H_hubble
