"""
Command-line interface utilities for cosmological simulations.
Provides shared argument parsing for run_simulation.py and visualize_3d.py.
"""

import argparse
from typing import Optional
from .constants import SimulationParameters


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Add simulation arguments shared across all CLI scripts.

    Arguments added:
    - --M: External mass parameter (multiple of observable mass)
    - --S: Node separation distance (Gpc)
    - --particles: Number of simulation particles
    - --seed: Random seed
    - --t-start: Start time (Gyr)
    - --t-duration: Duration (Gyr)
    - --n-steps: Number of timesteps
    - --damping: Initial velocity damping factor
    - --center-node-mass: Central node mass (multiple of M_observable)
    - --compare: Enable comparison mode (3-way visualization)
    """
    # External-Node Model parameters
    parser.add_argument('--M', type=float, default=855,
                        help='External mass parameter (in units of observable mass)')
    parser.add_argument('--S', type=float, default=25.0,
                        help='Node separation distance (in Gpc)')

    # Simulation setup
    parser.add_argument('--particles', type=int, default=200,
                        help='Number of simulation particles (200 matches parameter_sweep quick mode)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    # Time parameters
    parser.add_argument('--t-start', type=float, default=5.8,
                        help='Simulation start time since Big Bang (in Gyr)')
    parser.add_argument('--t-duration', type=float, default=8.0,
                        help='Simulation duration (in Gyr)')
    parser.add_argument('--n-steps', type=int, default=250,
                        help='Number of simulation timesteps (250 matches parameter_sweep quick mode)')

    # Physics parameters
    parser.add_argument('--damping', type=float, default=None,
                        help='Initial velocity damping factor (0-1). Auto-calculated if not specified.')
    parser.add_argument('--center-node-mass', type=float, default=1.0,
                        help='Central (progenitor) node mass as multiple of M_observable. '
                             'Affects total_mass_kg and softening_m scaling.')
    parser.add_argument('--mass-randomize', type=float, default=0.0,
                        help='Particle mass randomization (0.0=equal, 1.0=0 to 2x mean). '
                             'Total mass is preserved. Default 0.0 for deterministic results.')

    # Mode flags
    parser.add_argument('--compare', action='store_true',
                        help='Enable comparison mode (External-Node vs Matter-only vs LCDM)')


def parse_arguments(description: str = 'Run External-Node Cosmology Simulation',
                    add_output_dir: bool = True) -> argparse.Namespace:
    """
    Create parser with common arguments and parse command line.

    Args:
        description: Help text description for the parser
        add_output_dir: If True, adds --output-dir argument

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    if add_output_dir:
        parser.add_argument('--output-dir', type=str, default='./results',
                            help='Output directory for simulation results')

    add_common_arguments(parser)

    return parser.parse_args()


def args_to_sim_params(args: argparse.Namespace) -> SimulationParameters:
    """
    Convert parsed arguments to SimulationParameters object.

    Args:
        args: Parsed argument namespace from argparse

    Returns:
        SimulationParameters configured from CLI args
    """
    return SimulationParameters(
        M_value=args.M,
        S_value=args.S,
        n_particles=args.particles,
        seed=args.seed,
        t_start_Gyr=args.t_start,
        t_duration_Gyr=args.t_duration,
        n_steps=args.n_steps,
        damping_factor=args.damping,
        center_node_mass=args.center_node_mass,
        mass_randomize=args.mass_randomize
    )
