"""
Simulation Factory Functions

Factory functions for creating common simulation configurations.
Reduces boilerplate and ensures consistent setup across scripts.
"""

import numpy as np
from .simulation import CosmologicalSimulation
from .constants import SimulationParameters


def create_external_node_simulation(sim_params, box_size_Gpc, a_start, seed=None):
    """
    Create an External-Node simulation with standard configuration.

    Parameters:
    -----------
    sim_params : SimulationParameters
        Simulation parameters (M_value, S_value, etc.)
    box_size_Gpc : float
        Initial box size [Gpc]
    a_start : float
        Initial scale factor
    seed : int or None
        Random seed (if None, uses sim_params.seed)

    Returns:
    --------
    CosmologicalSimulation
        Configured External-Node simulation
    """
    if seed is None:
        seed = sim_params.seed

    np.random.seed(seed)

    return CosmologicalSimulation(
        n_particles=sim_params.n_particles,
        box_size_Gpc=box_size_Gpc,
        use_external_nodes=True,
        external_node_params=sim_params.external_params,
        t_start_Gyr=sim_params.t_start_Gyr,
        a_start=a_start,
        use_dark_energy=False,
        damping_factor=sim_params.damping_factor
    )


def create_matter_only_simulation(sim_params, box_size_Gpc, a_start, seed=None):
    """
    Create a Matter-only simulation (no external nodes, no dark energy).

    Parameters:
    -----------
    sim_params : SimulationParameters
        Simulation parameters
    box_size_Gpc : float
        Initial box size [Gpc]
    a_start : float
        Initial scale factor
    seed : int or None
        Random seed (if None, uses sim_params.seed)

    Returns:
    --------
    CosmologicalSimulation
        Configured Matter-only simulation
    """
    if seed is None:
        seed = sim_params.seed

    np.random.seed(seed)

    return CosmologicalSimulation(
        n_particles=sim_params.n_particles,
        box_size_Gpc=box_size_Gpc,
        use_external_nodes=False,
        external_node_params=None,
        t_start_Gyr=sim_params.t_start_Gyr,
        a_start=a_start,
        use_dark_energy=False,
        damping_factor=sim_params.damping_factor
    )


def create_lcdm_simulation(n_particles, box_size_Gpc, t_start_Gyr, a_start, seed=42):
    """
    Create a ΛCDM simulation (with dark energy, no external nodes).

    Parameters:
    -----------
    n_particles : int
        Number of simulation particles
    box_size_Gpc : float
        Initial box size [Gpc]
    t_start_Gyr : float
        Start time since Big Bang [Gyr]
    a_start : float
        Initial scale factor
    seed : int
        Random seed

    Returns:
    --------
    CosmologicalSimulation
        Configured ΛCDM simulation
    """
    np.random.seed(seed)

    return CosmologicalSimulation(
        n_particles=n_particles,
        box_size_Gpc=box_size_Gpc,
        use_external_nodes=False,
        external_node_params=None,
        t_start_Gyr=t_start_Gyr,
        a_start=a_start,
        use_dark_energy=True,
        damping_factor=None  # Auto-calculate for ΛCDM
    )


def run_and_extract_results(sim, t_duration_Gyr, n_steps, save_interval=10):
    """
    Run simulation and extract common results.

    Parameters:
    -----------
    sim : CosmologicalSimulation
        Simulation to run
    t_duration_Gyr : float
        Simulation duration [Gyr]
    n_steps : int
        Number of timesteps
    save_interval : int
        Save expansion history every N steps

    Returns:
    --------
    dict with keys:
        't_Gyr' : ndarray
            Time array [Gyr]
        'a' : ndarray
            Scale factor array
        'size_Gpc' : ndarray
            Physical size array [Gpc]
        'sim' : CosmologicalSimulation
            The simulation object (for further analysis)
    """
    from .constants import CosmologicalConstants
    from .analysis import extract_expansion_history

    sim.run(t_end_Gyr=t_duration_Gyr, n_steps=n_steps, save_interval=save_interval)

    const = CosmologicalConstants()

    t_Gyr = extract_expansion_history(sim, 'time_Gyr')
    a = extract_expansion_history(sim, 'scale_factor')
    size_m = extract_expansion_history(sim, 'size')
    size_Gpc = size_m / const.Gpc_to_m

    return {
        't_Gyr': t_Gyr,
        'a': a,
        'size_Gpc': size_Gpc,
        'sim': sim
    }
