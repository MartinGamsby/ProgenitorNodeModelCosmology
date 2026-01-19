#!/usr/bin/env python3
"""
External-Node Cosmology Simulation
Main script to run cosmology simulations with configurable parameters.
"""

import sys
import os
import argparse
import numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt

from cosmo.constants import CosmologicalConstants, LambdaCDMParameters, SimulationParameters
from cosmo.simulation import CosmologicalSimulation
from cosmo.analysis import (
    solve_friedmann_equation,
    calculate_initial_conditions,
    normalize_to_initial_size,
    compare_expansion_histories,
    detect_runaway_particles,
    calculate_today_marker,
    extract_expansion_history
)
from cosmo.visualization import (
    generate_output_filename,
    format_simulation_title,
    create_comparison_plot
)
from cosmo.factories import (
    create_external_node_simulation,
    create_matter_only_simulation,
    run_and_extract_results
)


def run_simulation(output_dir, sim_params):
    """Run cosmological simulation with specified parameters

    Parameters:
    -----------
    output_dir : str
        Directory to save output files
    sim_params : SimulationParameters
        Simulation configuration parameters
    """

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    print("="*70)
    print(f"SIMULATION CONFIGURATION - M={sim_params.M_value}, S={sim_params.S_value}, {sim_params.n_particles} particles")
    print("="*70)

    # Set random seed for reproducibility
    np.random.seed(sim_params.seed)

    const = CosmologicalConstants()
    lcdm_params = LambdaCDMParameters()

    # Calculate initial conditions
    initial_conditions = calculate_initial_conditions(sim_params.t_start_Gyr)
    a_at_start = initial_conditions['a_start']
    lcdm_initial_size = initial_conditions['box_size_Gpc']
    H_start_hubble = initial_conditions['H_start_hubble']

    print(f"H(t_start={sim_params.t_start_Gyr} Gyr) = {H_start_hubble:.1f} km/s/Mpc")

    # Solve ΛCDM evolution
    lcdm_solution = solve_friedmann_equation(
        sim_params.t_start_Gyr,
        sim_params.t_end_Gyr,
        Omega_Lambda=lcdm_params.Omega_Lambda
    )
    t_lcdm = lcdm_solution['t_Gyr'] - sim_params.t_start_Gyr
    a_lcdm = lcdm_solution['a']
    size_lcdm = normalize_to_initial_size(a_lcdm, lcdm_initial_size)
    H_lcdm_hubble = lcdm_solution['H_hubble']

    print(f"LCDM: {lcdm_initial_size:.3f} -> {size_lcdm[-1]:.2f} Gpc")

    # Solve matter-only evolution
    matter_solution = solve_friedmann_equation(
        sim_params.t_start_Gyr,
        sim_params.t_end_Gyr,
        Omega_Lambda=0.0  # Matter-only
    )
    a_matter = matter_solution['a']
    size_matter = normalize_to_initial_size(a_matter, lcdm_initial_size)
    H_matter_hubble = matter_solution['H_hubble']

    print(f"Matter-only: {lcdm_initial_size:.3f} -> {size_matter[-1]:.2f} Gpc")

    # Set up External-Node simulation
    ext_initial_size = lcdm_initial_size

    print(f"\nM={sim_params.M_value}, S={sim_params.S_value}, Omega_Lambda_eff={sim_params.external_params.Omega_Lambda_eff:.3f}")
    print(f"{sim_params.n_particles} particles, seed={sim_params.seed}")

    # Run External-Node simulation
    print("\nRunning External-Node simulation...")
    sim = create_external_node_simulation(sim_params, ext_initial_size, a_at_start)
    ext_results = run_and_extract_results(sim, sim_params.t_duration_Gyr, sim_params.n_steps)

    t_ext = ext_results['t_Gyr']
    a_ext = ext_results['a']
    size_ext = ext_results['size_Gpc']

    # Run matter-only simulation
    print("\nRunning Matter-only simulation...")
    sim_matter = create_matter_only_simulation(sim_params, ext_initial_size, a_at_start)
    matter_results = run_and_extract_results(sim_matter, sim_params.t_duration_Gyr, sim_params.n_steps)

    t_matter = matter_results['t_Gyr']
    a_matter_sim = matter_results['a']
    size_matter_sim = matter_results['size_Gpc']

    # Calculate match statistics
    size_ext_final = size_ext[-1]
    size_lcdm_final = size_lcdm[-1]
    size_matter_sim_final = size_matter_sim[-1]

    ext_match = compare_expansion_histories(size_ext_final, size_lcdm_final)
    matter_match = compare_expansion_histories(size_matter_sim_final, size_lcdm_final)

    print(f"\n{size_ext_final:.2f} Gpc")
    print(f"Match: {ext_match:.2f}%")

    # Check for runaway particles
    max_ext_final = sim.expansion_history[-1]['max_particle_distance'] / const.Gpc_to_m
    max_matter_final = sim_matter.expansion_history[-1]['max_particle_distance'] / const.Gpc_to_m

    print(f"Max particle: {max_ext_final:.1f} Gpc")

    ext_runaway = detect_runaway_particles(max_ext_final, size_ext_final)
    if ext_runaway['detected']:
        print(f"WARNING: Runaway particles detected! Max/RMS ratio = {ext_runaway['ratio']:.1f}× (should be < {ext_runaway['threshold']}×)")
        print(f"         This indicates numerical instability - particles being shot out")

    # Create visualization
    # Calculate Hubble parameters from smoothed scale factors
    a_ext_smooth = gaussian_filter1d(a_ext, sigma=2)
    H_ext = np.gradient(a_ext_smooth, t_ext * 1e9 * 365.25 * 24 * 3600) / a_ext_smooth
    H_ext_hubble = H_ext * const.Mpc_to_m / 1000

    a_matter_sim_smooth = gaussian_filter1d(a_matter_sim, sigma=2)
    H_matter_sim = np.gradient(a_matter_sim_smooth, t_matter * 1e9 * 365.25 * 24 * 3600) / a_matter_sim_smooth
    H_matter_sim_hubble = H_matter_sim * const.Mpc_to_m / 1000

    today = calculate_today_marker(sim_params.t_start_Gyr, sim_params.t_duration_Gyr)

    fig = create_comparison_plot(
        sim_params,
        t_lcdm, a_lcdm, size_lcdm, H_lcdm_hubble,
        t_ext, a_ext, size_ext, H_ext_hubble,
        t_matter, a_matter_sim, size_matter_sim, H_matter_sim_hubble,
        today=today
    )

    # Save outputs using standard naming
    plot_path = generate_output_filename('figure_simulation_results', sim_params, 'png', output_dir)
    sim_path = generate_output_filename('simulation', sim_params, 'pkl', output_dir)

    plt.savefig(plot_path, dpi=150)
    sim.save(sim_path)

    print(f"\nFiles saved to {output_dir}/")
    print(f"  - {os.path.basename(plot_path)}")
    print(f"  - {os.path.basename(sim_path)}")

    return sim, sim_matter, size_ext_final, size_lcdm_final, size_matter_sim_final, ext_match, matter_match, max_ext_final, max_matter_final


def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='Run External-Node Cosmology Simulation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='./results',
        help='Output directory for simulation results'
    )

    parser.add_argument(
        '--M',
        type=float,
        default=800,
        help='External mass parameter (in units of observable mass)'
    )

    parser.add_argument(
        '--S',
        type=float,
        default=24.0,
        help='Node separation distance (in Gpc)'
    )

    parser.add_argument(
        '--particles',
        type=int,
        default=300,
        help='Number of simulation particles'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )

    parser.add_argument(
        '--t-start',
        type=float,
        default=10.8,
        help='Simulation start time since Big Bang (in Gyr)'
    )

    parser.add_argument(
        '--damping',
        type=float,
        default=None,
        help='Initial velocity damping factor (0-1). If not specified, auto-calculated.'
    )

    parser.add_argument(
        '--t-duration',
        type=float,
        default=6.0,
        help='Simulation duration (in Gyr)'
    )

    parser.add_argument(
        '--n-steps',
        type=int,
        default=150,
        help='Number of simulation timesteps'
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    print(f"Output directory: {os.path.abspath(args.output_dir)}\n")

    # Create simulation parameters from command-line arguments
    sim_params = SimulationParameters(
        M_value=args.M,
        S_value=args.S,
        n_particles=args.particles,
        seed=args.seed,
        t_start_Gyr=args.t_start,
        t_duration_Gyr=args.t_duration,
        n_steps=args.n_steps,
        damping_factor=args.damping
    )

    sim, sim_matter, ext_final, lcdm_final, matter_final, ext_match, matter_match, max_ext, max_matter = run_simulation(args.output_dir, sim_params)

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"LCDM (with dark energy):        {lcdm_final:.2f} Gpc")
    print(f"External-Node (M={sim_params.M_value}, S={sim_params.S_value}): {ext_final:.2f} Gpc  ({ext_match:+.1f}% vs LCDM)")
    print(f"  Max particle distance: {max_ext:.1f} Gpc (ratio: {max_ext/ext_final:.2f}×)")

    # Check for runaway particles in External-Node
    if max_ext / ext_final > 2.0:
        print(f"  WARNING: Runaway particles detected in External-Node!")

    print(f"Matter-only (no dark energy):   {matter_final:.2f} Gpc  ({matter_match:+.1f}% vs LCDM)")
    print(f"  Max particle distance: {max_matter:.1f} Gpc (ratio: {max_matter/matter_final:.2f}×)")

    # Check for runaway particles in Matter-only
    if max_matter / matter_final > 2.0:
        print(f"  WARNING: Runaway particles detected in Matter-only!")

    print("\nSimulation complete!")
