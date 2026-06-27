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
                        help='Outer-mass multiplier: total simulated mass / inner observable mass, '
                             '>= 1.0; particles scale linearly (centerM=2 doubles particle count '
                             'by adding outer-shell matter at the same density); '
                             'default 1.0 = observable sphere only (no outer matter).')
    parser.add_argument('--outer-density-ceiling', type=float, default=1.0,
                        help='Multiplier on the inner EdS-critical density for outer-shell '
                             'particles (>= 0, clipped to MAX_OUTER_DENSITY_CEILING). '
                             'default 1.0 = outer density == inner density (EdS critical).')
    parser.add_argument('--mass-randomize', type=float, default=0.0,
                        help='Particle mass randomization (0.0=equal, 1.0=0 to 2x mean). '
                             'Total mass is preserved. Default 0.0 for deterministic results.')
    parser.add_argument('--node-mass-seed', type=int, default=0,
                        help='RNG seed for per-node mass distribution. '
                             'Independent of particle RNG. Default 0.')
    parser.add_argument('--node-mass-amplitude', type=float, default=0.0,
                        help='Log-normal width of per-node mass distribution. '
                             '0.0 (default) = all 26 nodes uniform = M_ext_kg (backward compatible).')
    parser.add_argument('--node-s-amplitude', type=float, default=0.0,
                        help='Log-normal width of per-node RADIAL position perturbation. '
                             '0.0 (default) = perfect symmetric lattice (backward compatible). '
                             'Reuses --node-mass-seed. Mean radial scale (S) preserved.')

    # Particle initialisation
    parser.add_argument('--init-distribution', type=str, default='uniform_sphere',
                        choices=['uniform_sphere', 'grf'],
                        help='Initial particle position distribution. '
                             '"uniform_sphere" (default) is backward-compatible. '
                             '"grf" uses a Gaussian random field with BBKS LCDM P(k) '
                             '+ Zel\'dovich displacement for realistic large-scale structure.')

    # Node geometry (WS3)
    parser.add_argument('--node-geometry', type=str, default='cube26',
                        choices=['cube26', 'cube_dense', 'fcc', 'bcc', 'virialized'],
                        help='HMEA node geometry (must be volume-filling / virialized). '
                             '"cube26" (default) = 3×3×3-1 cubic lattice (26 nodes, '
                             'backward-compatible). Alternatives: "cube_dense" '
                             '(5×5×5-1, 124 nodes), "fcc" / "bcc" (close-packed lattices), '
                             '"virialized" (COUPLED mass-segregated grid; see --vir-* flags). '
                             'Hollow spherical shells are excluded (opposite of virialized). '
                             'For a fair Omega_Lambda_eff comparison across geometries, '
                             'scale --M so M*26/n_nodes is constant.')

    # Virialized-grid parameters (consumed only when --node-geometry virialized)
    parser.add_argument('--vir-n-nodes', type=int, default=26,
                        help='Virialized node count (default 26, parity with cube26).')
    parser.add_argument('--vir-extent', type=float, default=1.0,
                        help='Virialized radius multiplier; raw outer radius ~ '
                             'vir_extent*S before NN-spacing rescale (default 1.0).')
    parser.add_argument('--vir-mass-rule', type=str, default='radial',
                        choices=['radial', 'massfunc'],
                        help='Virialized mass<->position rule: "radial" (deterministic '
                             'mass ~ f(r), default) or "massfunc" (log-normal draw + '
                             'spatial segregation).')
    parser.add_argument('--vir-mass-spread', type=float, default=0.0,
                        help='Virialized node-mass distribution amplitude. 0.0 (default) '
                             '= uniform masses (the falsifiable knob).')
    parser.add_argument('--vir-segregation', type=float, default=1.0,
                        help='Virialized mass<->radius coupling strength (default 1.0). '
                             '0.0 = mass/radius decoupled (no segregation).')
    parser.add_argument('--vir-s-metric', type=str, default='median',
                        choices=['median', 'mean'],
                        help='Virialized NN-spacing target metric: "median" (default) '
                             'or "mean".')
    parser.add_argument('--vir-relax-steps', type=int, default=1,
                        help='Virialized BALANCE LEVEL (default 1). 0 = realistic '
                             '(not force-balanced) Fibonacci layout; >= 1 = '
                             'force-balanced cubic-lattice ball (inner nodes feel '
                             '~zero net force).')

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
        outer_density_ceiling=getattr(args, 'outer_density_ceiling', 1.0),
        mass_randomize=args.mass_randomize,
        node_mass_seed=args.node_mass_seed,
        node_mass_amplitude=args.node_mass_amplitude,
        node_s_amplitude=getattr(args, 'node_s_amplitude', 0.0),
        init_distribution=args.init_distribution,
        node_geometry=getattr(args, 'node_geometry', 'cube26'),
        vir_n_nodes=getattr(args, 'vir_n_nodes', 26),
        vir_extent=getattr(args, 'vir_extent', 1.0),
        vir_mass_rule=getattr(args, 'vir_mass_rule', 'radial'),
        vir_mass_spread=getattr(args, 'vir_mass_spread', 0.0),
        vir_segregation=getattr(args, 'vir_segregation', 1.0),
        vir_s_metric=getattr(args, 'vir_s_metric', 'median'),
        vir_relax_steps=getattr(args, 'vir_relax_steps', 1),
    )
