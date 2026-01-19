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


def solve_lcdm_baseline(sim_params, lcdm_initial_size, a_start):
    """
    Solve analytic ΛCDM and matter-only evolution.

    Parameters:
    -----------
    sim_params : SimulationParameters
        Simulation configuration
    lcdm_initial_size : float
        Initial box size [Gpc]
    a_start : float
        Initial scale factor (for consistent normalization)

    Returns:
    --------
    dict with keys:
        'lcdm': dict with t, a, size, H arrays for ΛCDM
        'matter': dict with t, a, size, H arrays for matter-only
    """
    lcdm_params = LambdaCDMParameters()

    # Solve ΛCDM evolution
    lcdm_solution = solve_friedmann_equation(
        sim_params.t_start_Gyr,
        sim_params.t_end_Gyr,
        Omega_Lambda=lcdm_params.Omega_Lambda
    )
    t_lcdm = lcdm_solution['t_Gyr'] - sim_params.t_start_Gyr
    a_lcdm = lcdm_solution['a']
    # Normalize using the EXACT a_start for consistency with N-body sims
    size_lcdm = lcdm_initial_size * (a_lcdm / a_start)
    H_lcdm_hubble = lcdm_solution['H_hubble']

    print(f"LCDM: {lcdm_initial_size:.3f} -> {size_lcdm[-1]:.2f} Gpc")

    # Solve matter-only evolution
    matter_solution = solve_friedmann_equation(
        sim_params.t_start_Gyr,
        sim_params.t_end_Gyr,
        Omega_Lambda=0.0
    )
    a_matter = matter_solution['a']
    # Normalize using the EXACT a_start for consistency with N-body sims
    size_matter = lcdm_initial_size * (a_matter / a_start)
    H_matter_hubble = matter_solution['H_hubble']

    print(f"Matter-only: {lcdm_initial_size:.3f} -> {size_matter[-1]:.2f} Gpc")

    return {
        'lcdm': {
            't': t_lcdm,
            'a': a_lcdm,
            'size': size_lcdm,
            'H_hubble': H_lcdm_hubble
        },
        'matter': {
            't': t_lcdm,  # Use same time array for consistency
            'a': a_matter,
            'size': size_matter,
            'H_hubble': H_matter_hubble
        }
    }


def run_nbody_simulations(sim_params, box_size, a_start):
    """
    Run External-Node and Matter-only N-body simulations.

    Parameters:
    -----------
    sim_params : SimulationParameters
        Simulation configuration
    box_size : float
        Initial box size [Gpc]
    a_start : float
        Initial scale factor

    Returns:
    --------
    dict with keys:
        'ext': dict with sim, t, a, size arrays for External-Node
        'matter': dict with sim, t, a, size arrays for Matter-only
    """
    print(f"\nM={sim_params.M_value}, S={sim_params.S_value}, Omega_Lambda_eff={sim_params.external_params.Omega_Lambda_eff:.3f}")
    print(f"{sim_params.n_particles} particles, seed={sim_params.seed}")

    # Run External-Node simulation
    print("\nRunning External-Node simulation...")
    sim_ext = create_external_node_simulation(sim_params, box_size, a_start)
    ext_results = run_and_extract_results(sim_ext, sim_params.t_duration_Gyr, sim_params.n_steps)

    # Run matter-only simulation
    print("\nRunning Matter-only simulation...")
    sim_matter = create_matter_only_simulation(sim_params, box_size, a_start)
    matter_results = run_and_extract_results(sim_matter, sim_params.t_duration_Gyr, sim_params.n_steps)

    return {
        'ext': {
            'sim': ext_results['sim'],
            't': ext_results['t_Gyr'],
            'a': ext_results['a'],
            'size': ext_results['size_Gpc']
        },
        'matter': {
            'sim': matter_results['sim'],
            't': matter_results['t_Gyr'],
            'a': matter_results['a'],
            'size': matter_results['size_Gpc']
        }
    }


def calculate_hubble_parameters(t_ext, a_ext, t_matter, a_matter_sim):
    """
    Calculate Hubble parameters from smoothed scale factors.

    Parameters:
    -----------
    t_ext : ndarray
        External-Node time array [Gyr]
    a_ext : ndarray
        External-Node scale factor array
    t_matter : ndarray
        Matter-only time array [Gyr]
    a_matter_sim : ndarray
        Matter-only scale factor array

    Returns:
    --------
    dict with keys:
        'H_ext_hubble': Hubble parameter for External-Node [km/s/Mpc]
        'H_matter_hubble': Hubble parameter for Matter-only [km/s/Mpc]
    """
    const = CosmologicalConstants()

    # External-Node Hubble parameter
    a_ext_smooth = gaussian_filter1d(a_ext, sigma=2)
    H_ext = np.gradient(a_ext_smooth, t_ext * 1e9 * 365.25 * 24 * 3600) / a_ext_smooth
    H_ext_hubble = H_ext * const.Mpc_to_m / 1000

    # Matter-only Hubble parameter
    a_matter_sim_smooth = gaussian_filter1d(a_matter_sim, sigma=2)
    H_matter_sim = np.gradient(a_matter_sim_smooth, t_matter * 1e9 * 365.25 * 24 * 3600) / a_matter_sim_smooth
    H_matter_sim_hubble = H_matter_sim * const.Mpc_to_m / 1000

    return {
        'H_ext_hubble': H_ext_hubble,
        'H_matter_hubble': H_matter_sim_hubble
    }


def print_results_summary(sim_params, size_ext_final, size_lcdm_final, size_matter_final,
                         ext_match, matter_match, max_ext_final, max_matter_final):
    """
    Print summary of simulation results.

    Parameters:
    -----------
    sim_params : SimulationParameters
        Simulation configuration
    size_ext_final : float
        Final External-Node size [Gpc]
    size_lcdm_final : float
        Final ΛCDM size [Gpc]
    size_matter_final : float
        Final Matter-only size [Gpc]
    ext_match : float
        External-Node match percentage
    matter_match : float
        Matter-only match percentage
    max_ext_final : float
        Max External-Node particle distance [Gpc]
    max_matter_final : float
        Max Matter-only particle distance [Gpc]
    """
    print(f"\n{size_ext_final:.2f} Gpc")
    print(f"Match: {ext_match:.2f}%")
    print(f"Max particle: {max_ext_final:.1f} Gpc")

    # Check for runaway particles in External-Node
    ext_runaway = detect_runaway_particles(max_ext_final, size_ext_final)
    if ext_runaway['detected']:
        print(f"WARNING: Runaway particles detected! Max/RMS ratio = {ext_runaway['ratio']:.1f}× (should be < {ext_runaway['threshold']}×)")
        print(f"         This indicates numerical instability - particles being shot out")


def run_simulation(output_dir, sim_params):
    """Run cosmological simulation with specified parameters

    Parameters:
    -----------
    output_dir : str
        Directory to save output files
    sim_params : SimulationParameters
        Simulation configuration parameters

    Returns:
    --------
    tuple
        (sim_ext, sim_matter, size_ext_final, size_lcdm_final, size_matter_final,
         ext_match, matter_match, max_ext_final, max_matter_final)
    """
    # Setup
    os.makedirs(output_dir, exist_ok=True)
    print("="*70)
    print(f"SIMULATION CONFIGURATION - M={sim_params.M_value}, S={sim_params.S_value}, {sim_params.n_particles} particles")
    print("="*70)

    np.random.seed(sim_params.seed)
    const = CosmologicalConstants()

    # Calculate initial conditions
    initial_conditions = calculate_initial_conditions(sim_params.t_start_Gyr)
    a_at_start = initial_conditions['a_start']
    box_size = initial_conditions['box_size_Gpc']
    print(f"H(t_start={sim_params.t_start_Gyr} Gyr) = {initial_conditions['H_start_hubble']:.1f} km/s/Mpc")

    # Solve analytic baselines (ΛCDM and matter-only)
    baseline = solve_lcdm_baseline(sim_params, box_size, a_at_start)

    # Run N-body simulations (External-Node and Matter-only)
    nbody = run_nbody_simulations(sim_params, box_size, a_at_start)

    # Calculate match statistics
    size_ext_final = nbody['ext']['size'][-1]
    size_lcdm_final = baseline['lcdm']['size'][-1]
    size_matter_final = nbody['matter']['size'][-1]

    ext_match = compare_expansion_histories(size_ext_final, size_lcdm_final)
    matter_match = compare_expansion_histories(size_matter_final, size_lcdm_final)

    # Check for runaway particles
    max_ext_final = nbody['ext']['sim'].expansion_history[-1]['max_particle_distance'] / const.Gpc_to_m
    max_matter_final = nbody['matter']['sim'].expansion_history[-1]['max_particle_distance'] / const.Gpc_to_m

    print_results_summary(
        sim_params, size_ext_final, size_lcdm_final, size_matter_final,
        ext_match, matter_match, max_ext_final, max_matter_final
    )

    # Calculate Hubble parameters for plotting
    hubble = calculate_hubble_parameters(
        nbody['ext']['t'], nbody['ext']['a'],
        nbody['matter']['t'], nbody['matter']['a']
    )

    # Create visualization
    today = calculate_today_marker(sim_params.t_start_Gyr, sim_params.t_duration_Gyr)
    fig = create_comparison_plot(
        sim_params,
        baseline['lcdm']['t'], baseline['lcdm']['a'], baseline['lcdm']['size'], baseline['lcdm']['H_hubble'],
        nbody['ext']['t'], nbody['ext']['a'], nbody['ext']['size'], hubble['H_ext_hubble'],
        nbody['matter']['t'], nbody['matter']['a'], nbody['matter']['size'], hubble['H_matter_hubble'],
        today=today
    )

    # Save outputs
    plot_path = generate_output_filename('figure_simulation_results', sim_params, 'png', output_dir)
    sim_path = generate_output_filename('simulation', sim_params, 'pkl', output_dir)

    plt.savefig(plot_path, dpi=150)
    nbody['ext']['sim'].save(sim_path)

    print(f"\nFiles saved to {output_dir}/")
    print(f"  - {os.path.basename(plot_path)}")
    print(f"  - {os.path.basename(sim_path)}")

    return (nbody['ext']['sim'], nbody['matter']['sim'],
            size_ext_final, size_lcdm_final, size_matter_final,
            ext_match, matter_match, max_ext_final, max_matter_final)


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
