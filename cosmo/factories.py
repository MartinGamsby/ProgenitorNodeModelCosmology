"""
Simulation Utility Functions

Shared functions for running simulations and computing baselines.
Used by both run_simulation.py and parameter_sweep.py for consistency.
"""

from typing import Dict, Tuple
import numpy as np
from .simulation import CosmologicalSimulation
from .constants import CosmologicalConstants, LambdaCDMParameters, SimulationParameters
from .analysis import solve_friedmann_at_times, calculate_initial_conditions, calculate_hubble_parameters


def run_and_extract_results(sim: CosmologicalSimulation, t_duration_Gyr: float,
                            n_steps: int, save_interval: int = 10,
                            damping: float = None) -> Dict:
    """
    Run simulation and extract common results.

    Args:
        sim: Initialized CosmologicalSimulation
        t_duration_Gyr: Duration in Gyr
        n_steps: Number of timesteps
        save_interval: Save every N steps
        damping: Velocity damping factor (None = auto-calculate)

    Returns dict with keys: 't_Gyr', 'a', 'diameter_Gpc', 'max_radius_Gpc', 'sim'.
    """
    from .constants import CosmologicalConstants
    from .analysis import extract_expansion_history

    sim.run(t_end_Gyr=t_duration_Gyr, n_steps=n_steps, save_interval=save_interval, damping=damping)

    const = CosmologicalConstants()

    t_Gyr = extract_expansion_history(sim, 'time_Gyr')
    a = extract_expansion_history(sim, 'scale_factor')
    diameter_m = extract_expansion_history(sim, 'diameter_m')
    max_radius_m = extract_expansion_history(sim, 'max_particle_distance')
    diameter_Gpc = diameter_m / const.Gpc_to_m
    max_radius_Gpc = max_radius_m / const.Gpc_to_m

    # Calculate Hubble parameter from scale factor
    H_hubble = calculate_hubble_parameters(t_Gyr, a, smooth_sigma=0.0)

    return {
        't_Gyr': t_Gyr,
        'a': a,
        'diameter_Gpc': diameter_Gpc,
        'max_radius_Gpc': max_radius_Gpc,
        'H_hubble': H_hubble,
        'sim': sim
    }


def solve_lcdm_baseline(sim_params: SimulationParameters, box_size_Gpc: float,
                        a_start: float, save_interval: int = 10) -> Dict:
    """
    Solve analytic ΛCDM evolution at N-body simulation snapshot times.

    Uses exact time alignment with N-body snapshots to eliminate interpolation artifacts.
    This is the single source of truth for LCDM baseline computation.

    Args:
        sim_params: Simulation parameters (for t_start, t_duration, n_steps)
        box_size_Gpc: Initial box size in Gpc (diameter)
        a_start: Scale factor at t_start
        save_interval: Save every N steps (must match N-body save_interval)

    Returns dict with keys:
        't': Time relative to simulation start (starts at 0.0)
        'a': Scale factor array
        'diameter_Gpc': Diameter (2×RMS radius) in Gpc
        'max_diameter_Gpc': Max diameter scaled for uniform sphere
        'H_hubble': Hubble parameter in km/s/Mpc
    """
    lcdm_params = LambdaCDMParameters()

    # Compute time array matching N-body snapshots
    snapshot_steps = np.arange(0, sim_params.n_steps + 1, save_interval)
    t_relative_Gyr = (snapshot_steps / sim_params.n_steps) * sim_params.t_duration_Gyr
    t_absolute_Gyr = sim_params.t_start_Gyr + t_relative_Gyr

    # Solve ΛCDM at exact N-body snapshot times
    lcdm_solution = solve_friedmann_at_times(t_absolute_Gyr, Omega_Lambda=lcdm_params.Omega_Lambda)
    a_lcdm = lcdm_solution['a']
    H_lcdm_hubble = lcdm_solution['H_hubble']

    # Normalize using the EXACT a_start for consistency with N-body sims
    diameter_lcdm_Gpc = box_size_Gpc * (a_lcdm / a_start)

    # Scale for uniform sphere: RMS radius = R * sqrt(3/5)
    max_diameter_lcdm_Gpc = diameter_lcdm_Gpc / np.sqrt(3/5)

    return {
        't': t_relative_Gyr,
        'a': a_lcdm,
        'diameter_Gpc': diameter_lcdm_Gpc,
        'max_diameter_Gpc': max_diameter_lcdm_Gpc,
        'H_hubble': H_lcdm_hubble
    }


def run_external_node_simulation(sim_params: SimulationParameters, box_size_Gpc: float,
                                  a_start: float, save_interval: int = 10) -> Dict:
    """
    Run External-Node N-body simulation.

    Args:
        sim_params: Simulation parameters
        box_size_Gpc: Initial box size in Gpc
        a_start: Scale factor at t_start
        save_interval: Save every N steps

    Returns dict with keys: 't_Gyr', 'a', 'diameter_Gpc', 'max_radius_Gpc', 'H_hubble', 'sim'
    """
    sim = CosmologicalSimulation(sim_params, box_size_Gpc, a_start,
                                  use_external_nodes=True, use_dark_energy=False)
    return run_and_extract_results(sim, sim_params.t_duration_Gyr, sim_params.n_steps,
                                    save_interval, damping=sim_params.damping_factor)


def run_matter_only_simulation(sim_params: SimulationParameters, box_size_Gpc: float,
                                a_start: float, save_interval: int = 10) -> Dict:
    """
    Run matter-only N-body simulation (no external nodes, no dark energy).

    Args:
        sim_params: Simulation parameters
        box_size_Gpc: Initial box size in Gpc
        a_start: Scale factor at t_start
        save_interval: Save every N steps

    Returns dict with keys: 't_Gyr', 'a', 'diameter_Gpc', 'max_radius_Gpc', 'H_hubble', 'sim'
    """
    sim = CosmologicalSimulation(sim_params, box_size_Gpc, a_start,
                                  use_external_nodes=False, use_dark_energy=False)
    return run_and_extract_results(sim, sim_params.t_duration_Gyr, sim_params.n_steps,
                                    save_interval, damping=sim_params.damping_factor)


def setup_simulation_context(t_start_Gyr: float, t_duration_Gyr: float,
                              n_steps: int, save_interval: int = 10) -> Tuple[float, float, Dict]:
    """
    Calculate initial conditions and LCDM baseline for a simulation run.

    Combines calculate_initial_conditions + solve_lcdm_baseline into one call.
    This is the standard setup for both run_simulation.py and parameter_sweep.py.

    Args:
        t_start_Gyr: Start time in Gyr
        t_duration_Gyr: Duration in Gyr
        n_steps: Number of timesteps
        save_interval: Save every N steps

    Returns:
        (box_size_Gpc, a_start, baseline_dict)
        where baseline_dict has keys: 't', 'a', 'diameter_Gpc', 'max_diameter_Gpc', 'H_hubble'
    """
    # Calculate initial conditions
    initial_conditions = calculate_initial_conditions(t_start_Gyr)
    box_size_Gpc = initial_conditions['box_size_Gpc']
    a_start = initial_conditions['a_start']

    # Create minimal sim_params for baseline computation
    baseline_params = SimulationParameters(
        M_value=1, S_value=1,  # Unused for baseline
        t_start_Gyr=t_start_Gyr,
        t_duration_Gyr=t_duration_Gyr,
        n_steps=n_steps
    )

    # Compute LCDM baseline
    baseline = solve_lcdm_baseline(baseline_params, box_size_Gpc, a_start, save_interval)

    return box_size_Gpc, a_start, baseline


def results_to_sim_result(ext_results: Dict, sim_params: SimulationParameters):
    """
    Convert factory results dict to SimResult for parameter_sweep compatibility.

    Args:
        ext_results: Dict from run_external_node_simulation
        sim_params: SimulationParameters used for the run

    Returns:
        SimResult dataclass instance
    """
    # Import here to avoid circular dependency
    from .parameter_sweep import SimResult

    return SimResult(
        size_curve_Gpc=ext_results['diameter_Gpc'],
        hubble_curve=ext_results['H_hubble'],
        size_final_Gpc=ext_results['diameter_Gpc'][-1],
        radius_max_Gpc=ext_results['max_radius_Gpc'][-1],
        a_final=ext_results['a'][-1],
        t_Gyr=ext_results['t_Gyr'],
        params=sim_params.external_params
    )
