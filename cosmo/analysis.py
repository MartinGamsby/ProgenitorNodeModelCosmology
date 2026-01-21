"""
Cosmology Analysis Utilities

Shared functions for:
- Solving Friedmann equations (ΛCDM, matter-only)
- Computing initial conditions
- Comparing expansion histories
- Detecting numerical instabilities
"""

import numpy as np
from scipy.integrate import odeint
from .constants import CosmologicalConstants, LambdaCDMParameters


def friedmann_equation(a, t, H0, Omega_m, Omega_Lambda):
    """
    Friedmann equation for cosmic scale factor evolution.

    da/dt = H(a) * a where H(a) = H0 * sqrt(Ω_m/a³ + Ω_Λ)

    Parameters:
    -----------
    a : float
        Scale factor
    t : float
        Time (not used, required by odeint signature)
    H0 : float
        Hubble constant [1/s]
    Omega_m : float
        Matter density parameter
    Omega_Lambda : float
        Dark energy density parameter

    Returns:
    --------
    float
        da/dt
    """
    if a <= 0:
        return 1e-10
    H = H0 * np.sqrt(Omega_m / a**3 + Omega_Lambda)
    return H * a


def solve_friedmann_equation(t_start_Gyr, t_end_Gyr, Omega_Lambda=None, n_points=400):
    """
    Solve Friedmann equation for ΛCDM or matter-only cosmology.

    Parameters:
    -----------
    t_start_Gyr : float
        Start time since Big Bang [Gyr]
    t_end_Gyr : float
        End time since Big Bang [Gyr]
    Omega_Lambda : float or None
        Dark energy density. If None, uses ΛCDM value. Set to 0.0 for matter-only.
    n_points : int
        Number of time points to solve for

    Returns:
    --------
    dict with keys:
        't_Gyr' : ndarray
            Time array [Gyr]
        'a' : ndarray
            Scale factor array
        'H_hubble' : ndarray
            Hubble parameter [km/s/Mpc]
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
        args=(lcdm.H0, lcdm.Omega_m, Omega_Lambda)
    )
    a_solution = a_solution.flatten()

    # Convert time to Gyr
    t_Gyr_full = t_span / (1e9 * 365.25 * 24 * 3600)

    # Extract requested window
    mask = (t_Gyr_full >= t_start_Gyr) & (t_Gyr_full <= t_end_Gyr)
    t_Gyr = t_Gyr_full[mask]
    a = a_solution[mask]

    # Calculate Hubble parameter
    H_raw = lcdm.H0 * np.sqrt(lcdm.Omega_m / a**3 + Omega_Lambda)
    H_hubble = H_raw * const.Mpc_to_m / 1000  # Convert to km/s/Mpc

    return {
        't_Gyr': t_Gyr,
        'a': a,
        'H_hubble': H_hubble,
        '_t_Gyr_full': t_Gyr_full,
        '_a_full': a_solution
    }


def calculate_initial_conditions(t_start_Gyr, reference_size_today_Gpc=14.5):
    """
    Calculate initial scale factor and box size at given start time.

    Uses ΛCDM cosmology to determine scale factor at t_start,
    then computes box size by scaling from present-day reference.

    Parameters:
    -----------
    t_start_Gyr : float
        Start time since Big Bang [Gyr]
    reference_size_today_Gpc : float
        RMS radius of observable universe today [Gpc]

    Returns:
    --------
    dict with keys:
        'a_start' : float
            Scale factor at t_start
        'box_size_Gpc' : float
            Initial box size [Gpc]
        'H_start_hubble' : float
            Hubble parameter at t_start [km/s/Mpc]
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


def normalize_to_initial_size(a_array, initial_size_Gpc):
    """
    Convert scale factor array to physical size array.

    Parameters:
    -----------
    a_array : ndarray
        Scale factor evolution (normalized to a[0])
    initial_size_Gpc : float
        Initial box size [Gpc]

    Returns:
    --------
    ndarray
        Physical size evolution [Gpc]
    """
    a_normalized = a_array / a_array[0]
    return initial_size_Gpc * a_normalized


def compare_expansion_histories(size_ext, size_lcdm):
    """
    Calculate match percentage between two expansion histories.

    Compares full expansion curves by computing average percentage difference
    across all timesteps. This ensures we match the entire evolution, not just
    the final state.

    Parameters:
    -----------
    size_ext : float or ndarray
        External-Node final size or size history [Gpc]
    size_lcdm : float or ndarray
        ΛCDM final size or size history [Gpc]

    Returns:
    --------
    float
        Match percentage (100% = perfect match, based on mean deviation across all timesteps)
    """
    # If arrays, compare full curve
    if isinstance(size_ext, np.ndarray) and isinstance(size_lcdm, np.ndarray):
        # Calculate percentage difference at each timestep
        diff_pct = np.abs(size_ext - size_lcdm) / size_lcdm * 100
        # Average across all timesteps
        mean_diff_pct = np.mean(diff_pct)
        return 100 - mean_diff_pct
    else:
        # Scalar comparison (backward compatibility)
        diff = np.abs(size_ext - size_lcdm) / size_lcdm * 100
        return 100 - diff


def detect_runaway_particles(max_distance_Gpc, rms_size_Gpc, threshold=2.0):
    """
    Detect runaway particles indicating numerical instability.

    When max particle distance >> RMS size, it indicates particles
    being "shot out" due to leapfrog instability or force errors.

    Parameters:
    -----------
    max_distance_Gpc : float
        Maximum particle distance from center [Gpc]
    rms_size_Gpc : float
        RMS radius of particle distribution [Gpc]
    threshold : float
        Ratio threshold for detection (default: 2.0)

    Returns:
    --------
    dict with keys:
        'detected' : bool
            True if runaway particles detected
        'ratio' : float
            Max/RMS ratio
        'threshold' : float
            Detection threshold used
    """
    ratio = max_distance_Gpc / rms_size_Gpc
    detected = ratio > threshold

    return {
        'detected': detected,
        'ratio': ratio,
        'threshold': threshold
    }


def calculate_today_marker(t_start_Gyr, t_duration_Gyr, today_Gyr=13.8):
    """
    Calculate position of "today" marker in simulation time coordinates.

    Parameters:
    -----------
    t_start_Gyr : float
        Simulation start time since Big Bang [Gyr]
    t_duration_Gyr : float
        Simulation duration [Gyr]
    today_Gyr : float
        Age of universe today [Gyr]

    Returns:
    --------
    float or None
        Time coordinate for "today" in simulation frame [Gyr],
        or None if today is outside simulation window
    """
    t_end_Gyr = t_start_Gyr + t_duration_Gyr

    if t_start_Gyr < today_Gyr < t_end_Gyr:
        return today_Gyr - t_start_Gyr
    return None


def extract_expansion_history(sim, key):
    """
    Extract a specific field from simulation expansion history as numpy array.

    Parameters:
    -----------
    sim : CosmologicalSimulation
        Simulation object with expansion_history attribute
    key : str
        Field to extract (e.g., 'time_Gyr', 'scale_factor', 'size')

    Returns:
    --------
    ndarray
        Array of values for the specified key
    """
    import numpy as np
    return np.array([h[key] for h in sim.expansion_history])
