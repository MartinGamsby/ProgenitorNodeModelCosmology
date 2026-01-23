"""
Simulation Utility Functions

Utility functions for running and extracting simulation results.
"""

from typing import Dict
import numpy as np
from .simulation import CosmologicalSimulation


def run_and_extract_results(sim: CosmologicalSimulation, t_duration_Gyr: float,
                            n_steps: int, save_interval: int = 10) -> Dict:
    """
    Run simulation and extract common results.

    Returns dict with keys: 't_Gyr', 'a', 'diameter_Gpc', 'max_radius_Gpc', 'sim'.
    """
    from .constants import CosmologicalConstants
    from .analysis import extract_expansion_history

    sim.run(t_end_Gyr=t_duration_Gyr, n_steps=n_steps, save_interval=save_interval)

    const = CosmologicalConstants()

    t_Gyr = extract_expansion_history(sim, 'time_Gyr')
    a = extract_expansion_history(sim, 'scale_factor')
    diameter_m = extract_expansion_history(sim, 'diameter_m')
    max_radius_m = extract_expansion_history(sim, 'max_particle_distance')
    diameter_Gpc = diameter_m / const.Gpc_to_m
    max_radius_Gpc = max_radius_m / const.Gpc_to_m

    return {
        't_Gyr': t_Gyr,
        'a': a,
        'diameter_Gpc': diameter_Gpc,
        'max_radius_Gpc': max_radius_Gpc,
        'sim': sim
    }
