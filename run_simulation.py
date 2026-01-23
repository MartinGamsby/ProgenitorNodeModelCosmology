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
    solve_friedmann_at_times,
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
from cosmo.factories import run_and_extract_results


def solve_lcdm_baseline(sim_params, lcdm_initial_size, a_start, save_interval=10):
    """
    Solve analytic ΛCDM and matter-only evolution at N-body simulation times.

    Uses exact time alignment with N-body snapshots to eliminate interpolation artifacts.

    Parameters:
    -----------
    sim_params : SimulationParameters
        Simulation configuration
    lcdm_initial_size : float
        Initial box size [Gpc]
    a_start : float
        Initial scale factor (for consistent normalization)
    save_interval : int
        Save interval used in N-body simulation (default: 10)

    Returns:
    --------
    dict with keys:
        'lcdm': dict with t, a, diameter_m, H arrays for ΛCDM
        'matter': dict with t, a, diameter_m, H arrays for matter-only
    """
    lcdm_params = LambdaCDMParameters()

    # Compute time array matching N-body snapshots
    # N-body saves initial snapshot + every save_interval steps
    n_snapshots = (sim_params.n_steps // save_interval) + 1  # +1 for initial snapshot

    # Time points: 0, dt*save_interval, 2*dt*save_interval, ..., t_duration
    # This matches exactly what the N-body simulation records
    snapshot_steps = np.arange(0, sim_params.n_steps + 1, save_interval)
    t_relative_Gyr = (snapshot_steps / sim_params.n_steps) * sim_params.t_duration_Gyr
    t_absolute_Gyr = sim_params.t_start_Gyr + t_relative_Gyr

    # Solve ΛCDM at exact N-body snapshot times
    lcdm_solution = solve_friedmann_at_times(t_absolute_Gyr, Omega_Lambda=lcdm_params.Omega_Lambda)
    a_lcdm = lcdm_solution['a']
    H_lcdm_hubble = lcdm_solution['H_hubble']

    # Normalize using the EXACT a_start for consistency with N-body sims
    # NOTE: lcdm_initial_size = box_size (diameter), matching N-body convention
    diameter_lcdm_Gpc = lcdm_initial_size * (a_lcdm / a_start)

    print(f"LCDM: {lcdm_initial_size:.3f} -> {diameter_lcdm_Gpc[-1]:.2f} Gpc")

    # Solve matter-only at same time points
    matter_solution = solve_friedmann_at_times(t_absolute_Gyr, Omega_Lambda=0.0)
    a_matter = matter_solution['a']
    H_matter_hubble = matter_solution['H_hubble']

    # Normalize using the EXACT a_start
    diameter_matter_Gpc = lcdm_initial_size * (a_matter / a_start)

    print(f"Matter-only: {lcdm_initial_size:.3f} -> {diameter_matter_Gpc[-1]:.2f} Gpc")

    return {
        'lcdm': {
            't': t_relative_Gyr,  # Time relative to simulation start (starts at exactly 0.0)
            'a': a_lcdm,
            'diameter_m': diameter_lcdm_Gpc,  # Diameter in Gpc (matches N-body convention)
            'H_hubble': H_lcdm_hubble
        },
        'matter': {
            't': t_relative_Gyr,  # Same time array (exact alignment with N-body)
            'a': a_matter,
            'diameter_m': diameter_matter_Gpc,  # Diameter in Gpc (matches N-body convention)
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
        'ext': dict with sim, t, a, diameter_m arrays for External-Node
        'matter': dict with sim, t, a, diameter_m arrays for Matter-only
    """
    print(f"\nM={sim_params.M_value}, S={sim_params.S_value}, Omega_Lambda_eff={sim_params.external_params.Omega_Lambda_eff:.3f}")
    print(f"{sim_params.n_particles} particles, seed={sim_params.seed}")

    # Run External-Node simulation
    print("\nRunning External-Node simulation...")
    sim_ext = CosmologicalSimulation(sim_params, box_size, a_start,
                                     use_external_nodes=True, use_dark_energy=False)
    ext_results = run_and_extract_results(sim_ext, sim_params.t_duration_Gyr, sim_params.n_steps)

    # Run matter-only simulation
    print("\nRunning Matter-only simulation...")
    sim_matter = CosmologicalSimulation(sim_params, box_size, a_start,
                                        use_external_nodes=False, use_dark_energy=False)
    matter_results = run_and_extract_results(sim_matter, sim_params.t_duration_Gyr, sim_params.n_steps)

    return {
        'ext': {
            'sim': ext_results['sim'],
            't': ext_results['t_Gyr'],
            'a': ext_results['a'],
            'diameter_m': ext_results['diameter_Gpc']
        },
        'matter': {
            'sim': matter_results['sim'],
            't': matter_results['t_Gyr'],
            'a': matter_results['a'],
            'diameter_m': matter_results['diameter_Gpc']
        }
    }


def calculate_hubble_parameters(t_ext, a_ext, t_matter, a_matter_sim, smooth_sigma=0.0):
    """
    Calculate Hubble parameters from scale factors.

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
    smooth_sigma : float
        Gaussian smoothing sigma (default: 0.0 = no smoothing).
        Use 1-2 to reduce numerical noise in derivatives.

    Returns:
    --------
    dict with keys:
        'H_ext_hubble': Hubble parameter for External-Node [km/s/Mpc]
        'H_matter_hubble': Hubble parameter for Matter-only [km/s/Mpc]
    """
    const = CosmologicalConstants()

    # Optional smoothing (default: no smoothing per user request)
    if smooth_sigma > 0:
        a_ext_smooth = gaussian_filter1d(a_ext, sigma=smooth_sigma)
        a_matter_sim_smooth = gaussian_filter1d(a_matter_sim, sigma=smooth_sigma)
    else:
        a_ext_smooth = a_ext
        a_matter_sim_smooth = a_matter_sim

    # External-Node Hubble parameter
    H_ext = np.gradient(a_ext_smooth, t_ext * 1e9 * 365.25 * 24 * 3600) / a_ext_smooth
    H_ext_hubble = H_ext * const.Mpc_to_m / 1000

    # Fix boundary points: np.gradient uses forward/backward differences at edges
    # which are less accurate. Replace first and last points with NaN to exclude them
    # from plots, or use second-order accurate formulas
    if len(H_ext_hubble) > 2:
        # Second-order forward difference for first point: f'(0) ≈ (-3f(0) + 4f(1) - f(2)) / (2h)
        dt_0 = (t_ext[1] - t_ext[0]) * 1e9 * 365.25 * 24 * 3600
        H_ext_hubble[0] = (-3*a_ext_smooth[0] + 4*a_ext_smooth[1] - a_ext_smooth[2]) / (2*dt_0 * a_ext_smooth[0]) * const.Mpc_to_m / 1000

        # Second-order backward difference for last point: f'(n) ≈ (3f(n) - 4f(n-1) + f(n-2)) / (2h)
        dt_n = (t_ext[-1] - t_ext[-2]) * 1e9 * 365.25 * 24 * 3600
        H_ext_hubble[-1] = (3*a_ext_smooth[-1] - 4*a_ext_smooth[-2] + a_ext_smooth[-3]) / (2*dt_n * a_ext_smooth[-1]) * const.Mpc_to_m / 1000

    # Matter-only Hubble parameter
    H_matter_sim = np.gradient(a_matter_sim_smooth, t_matter * 1e9 * 365.25 * 24 * 3600) / a_matter_sim_smooth
    H_matter_sim_hubble = H_matter_sim * const.Mpc_to_m / 1000

    # Fix boundary points for matter-only as well
    if len(H_matter_sim_hubble) > 2:
        dt_0 = (t_matter[1] - t_matter[0]) * 1e9 * 365.25 * 24 * 3600
        H_matter_sim_hubble[0] = (-3*a_matter_sim_smooth[0] + 4*a_matter_sim_smooth[1] - a_matter_sim_smooth[2]) / (2*dt_0 * a_matter_sim_smooth[0]) * const.Mpc_to_m / 1000

        dt_n = (t_matter[-1] - t_matter[-2]) * 1e9 * 365.25 * 24 * 3600
        H_matter_sim_hubble[-1] = (3*a_matter_sim_smooth[-1] - 4*a_matter_sim_smooth[-2] + a_matter_sim_smooth[-3]) / (2*dt_n * a_matter_sim_smooth[-1]) * const.Mpc_to_m / 1000

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
    size_ext_final = nbody['ext']['diameter_m'][-1]
    size_lcdm_final = baseline['lcdm']['diameter_m'][-1]
    size_matter_final = nbody['matter']['diameter_m'][-1]

    ext_match = compare_expansion_histories(size_ext_final, size_lcdm_final)
    matter_match = compare_expansion_histories(size_matter_final, size_lcdm_final)

    # Check for runaway particles
    max_ext_final = nbody['ext']['sim'].expansion_history[-1]['max_particle_distance'] / const.Gpc_to_m
    max_matter_final = nbody['matter']['sim'].expansion_history[-1]['max_particle_distance'] / const.Gpc_to_m

    print_results_summary(
        sim_params, size_ext_final, size_lcdm_final, size_matter_final,
        ext_match, matter_match, max_ext_final, max_matter_final
    )

    # Calculate Hubble parameters for plotting (no smoothing by default per user request)
    hubble = calculate_hubble_parameters(
        nbody['ext']['t'], nbody['ext']['a'],
        nbody['matter']['t'], nbody['matter']['a'],
        smooth_sigma=0.0
    )

    # Create visualization
    today = calculate_today_marker(sim_params.t_start_Gyr, sim_params.t_duration_Gyr)
    fig = create_comparison_plot(
        sim_params,
        baseline['lcdm']['t'], baseline['lcdm']['a'], baseline['lcdm']['diameter_m'], baseline['lcdm']['H_hubble'],
        nbody['ext']['t'], nbody['ext']['a'], nbody['ext']['diameter_m'], hubble['H_ext_hubble'],
        nbody['matter']['t'], nbody['matter']['a'], nbody['matter']['diameter_m'], hubble['H_matter_hubble'],
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
