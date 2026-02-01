#!/usr/bin/env python3
"""
External-Node Cosmology Simulation
Main script to run cosmology simulations with configurable parameters.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

from cosmo.constants import CosmologicalConstants, LambdaCDMParameters, SimulationParameters
from cosmo.cli import parse_arguments, args_to_sim_params
from cosmo.simulation import CosmologicalSimulation
from cosmo.analysis import (
    solve_friedmann_at_times,
    calculate_initial_conditions,
    compare_expansion_history,
    compare_expansion_histories,
    detect_runaway_particles,
    calculate_today_marker,
    calculate_hubble_parameters
)
from cosmo.visualization import (
    generate_output_filename,
    create_comparison_plot
)
from cosmo.factories import run_and_extract_results


def solve_lcdm_baseline(sim_params, lcdm_initial_size, a_start, save_interval=10):
    """
    Solve analytic ΛCDM and matter-only evolution at N-body simulation times.

    Uses exact time alignment with N-body snapshots to eliminate interpolation artifacts.
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

    # Scale box_size so that the RMS radius matches the target
    # For a uniform sphere of radius R, RMS radius = R * sqrt(3/5) ≈ 0.775*R
    # We want RMS = size_lcdm_Gpc, so R_sphere = size_lcdm_Gpc / 0.775
    # This means we need to use a sphere of radius: size_lcdm_Gpc/2 / sqrt(3/5)
    max_diameter_lcdm_Gpc = diameter_lcdm_Gpc / np.sqrt(3/5)
    
    return {
        't': t_relative_Gyr,  # Time relative to simulation start (starts at exactly 0.0)
        'a': a_lcdm,
        'diameter_Gpc': diameter_lcdm_Gpc,  # Diameter in Gpc (matches N-body convention)
        'max_diameter_Gpc': max_diameter_lcdm_Gpc,
        'H_hubble': H_lcdm_hubble
    }


def run_nbody_simulations(sim_params, box_size, a_start):
    """
    Run External-Node and Matter-only N-body simulations.
    """
    print(f"\nM={sim_params.M_value}, S={sim_params.S_value}, Omega_Lambda_eff={sim_params.external_params.Omega_Lambda_eff:.3f}")
    print(f"{sim_params.n_particles} particles, seed={sim_params.seed}")

    # Run External-Node simulation
    print("\nRunning External-Node simulation...")
    sim_ext = CosmologicalSimulation(sim_params, box_size, a_start,
                                     use_external_nodes=True, use_dark_energy=False)
    ext_results = run_and_extract_results(sim_ext, sim_params.t_duration_Gyr, sim_params.n_steps,
                                          damping=sim_params.damping_factor)

    # Run matter-only simulation
    print("\nRunning Matter-only simulation...")
    sim_matter = CosmologicalSimulation(sim_params, box_size, a_start,
                                        use_external_nodes=False, use_dark_energy=False)
    matter_results = run_and_extract_results(sim_matter, sim_params.t_duration_Gyr, sim_params.n_steps,
                                             damping=sim_params.damping_factor)

    return {
        'ext': {
            'sim': ext_results['sim'],
            't': ext_results['t_Gyr'],
            'a': ext_results['a'],
            'diameter_Gpc': ext_results['diameter_Gpc'],
            'max_diameter_Gpc': ext_results['max_radius_Gpc']*2
        },
        'matter': {
            'sim': matter_results['sim'],
            't': matter_results['t_Gyr'],
            'a': matter_results['a'],
            'diameter_Gpc': matter_results['diameter_Gpc'],
            'max_diameter_Gpc': matter_results['max_radius_Gpc']*2
        }
    }

def print_results_summary(size_ext_final, size_lcdm_final, size_matter_final,
                         ext_match, matter_match, match_ext_curve_pct, match_ext_hubble_curve_pct, 
                         match_matter_curve_pct, match_matter_hubble_curve_pct, max_ext_final):
    """
    Print summary of simulation results.
    """

    print("="*30)
    print("Simulation Results Summary:")
    print("="*30)
    print("Parameters used:")
    print(f"\tM={sim_params.M_value},\n\tS={sim_params.S_value},\n\tParticles={sim_params.n_particles}\n\tSeed={sim_params.seed}\n\tStart={sim_params.t_start_Gyr} Gyr,\n\tDuration={sim_params.t_duration_Gyr} Gyr,\n\tSteps={sim_params.n_steps},\n\tDamping={sim_params.damping_factor}")

    print("\nExternal Nodes:")
    print(f"\t{size_ext_final:.2f} Gpc")
    print(f"\tend size match: {ext_match:.2f}%")
    print(f"\t(last 1/2) size R2 match: {(match_ext_curve_pct/100):.4f}")
    print(f"\t(last 1/2) expansion rate R2 match: {(match_ext_hubble_curve_pct/100):.4f}")
    print(f"\tMax particle: {max_ext_final:.1f} Gpc")
    
    print("\nMatter-only:")
    print(f"\t{size_matter_final:.2f} Gpc")
    print(f"\tend size match: {matter_match:.2f}%")
    print(f"\t(last 1/2) size R2 match: {(match_matter_curve_pct/100):.4f}")
    print(f"\t(last 1/2) expansion rate R2 match: {(match_matter_hubble_curve_pct/100):.4f}")

    print("\nLCMD Summary:")
    print(f"\t{size_lcdm_final:.2f} Gpc")

    # Check for runaway particles in External-Node
    ext_runaway = detect_runaway_particles(max_ext_final, size_ext_final)
    if ext_runaway['detected']:
        print(f"WARNING: Runaway particles detected! Max/RMS ratio = {ext_runaway['ratio']:.1f}× (should be < {ext_runaway['threshold']}×)")
        print(f"         This indicates numerical instability - particles being shot out")


def run_simulation(output_dir, sim_params, use_max_radius=False):
    """Run cosmological simulation with specified parameters
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
    # TODO: Maybe slightly bigger initial size?? (for non-lcdm) (Because it accelerates slower at the beginning...)
    # (NOT THAT VALUE! THIS IS ONLY TO TEST!)
    # PROBABLY DEPENDS ON INITIAL VALUES, M, CenterM, start year, ETC.
    #box_size*= 1.003
    nbody = run_nbody_simulations(sim_params, box_size, a_at_start)

    # Calculate match statistics
    size_key = 'max_diameter_Gpc' if use_max_radius else 'diameter_Gpc'
    size_ext_final = nbody['ext'][size_key][-1]
    size_lcdm_final = baseline[size_key][-1]
    size_matter_final = nbody['matter'][size_key][-1]

    ext_match = compare_expansion_history(size_ext_final, size_lcdm_final)
    matter_match = compare_expansion_history(size_matter_final, size_lcdm_final)

    size_lcdm_curve = baseline[size_key]
    hubble_ext = calculate_hubble_parameters(nbody['ext']['t'], nbody['ext']['a'], smooth_sigma=0.0)
    H_lcdm_hubble = baseline['H_hubble']

    half_point = len(size_lcdm_curve)//2

    match_ext_curve_pct = compare_expansion_histories(nbody['ext'][size_key][half_point:], size_lcdm_curve[half_point:])
    match_ext_hubble_curve_pct = compare_expansion_histories(hubble_ext[half_point:], H_lcdm_hubble[half_point:])
    match_matter_curve_pct = compare_expansion_histories(nbody['matter'][size_key][half_point:], size_lcdm_curve[half_point:])
    match_matter_hubble_curve_pct = compare_expansion_histories(
        calculate_hubble_parameters(nbody['matter']['t'], nbody['matter']['a'], smooth_sigma=0.0)[half_point:], 
        H_lcdm_hubble[half_point:]
    )


    # Check for runaway particles
    max_ext_final = nbody['ext']['sim'].expansion_history[-1]['max_particle_distance'] / const.Gpc_to_m
    max_matter_final = nbody['matter']['sim'].expansion_history[-1]['max_particle_distance'] / const.Gpc_to_m

    print_results_summary(
        size_ext_final, size_lcdm_final, size_matter_final,
        ext_match, matter_match, match_ext_curve_pct, match_ext_hubble_curve_pct, 
        match_matter_curve_pct, match_matter_hubble_curve_pct, max_ext_final
    )

    # Calculate Hubble parameters for plotting (no smoothing by default per user request)
    H_ext_hubble = calculate_hubble_parameters(nbody['ext']['t'], nbody['ext']['a'], smooth_sigma=0.0)
    H_matter_hubble = calculate_hubble_parameters(nbody['matter']['t'], nbody['matter']['a'], smooth_sigma=0.0)

    # Create visualization
    today = calculate_today_marker(sim_params.t_start_Gyr, sim_params.t_duration_Gyr)
    fig = create_comparison_plot(
        sim_params,
        baseline['t'], baseline['a'], baseline[size_key], baseline['H_hubble'],
        nbody['ext']['t'], nbody['ext']['a'], nbody['ext'][size_key], H_ext_hubble,
        nbody['matter']['t'], nbody['matter']['a'], nbody['matter'][size_key], H_matter_hubble,
        today=today
    )

    # Save outputs
    plot_path = generate_output_filename('sim_plots', sim_params, 'png', output_dir)
    sim_path = generate_output_filename('simulation', sim_params, 'pkl', output_dir)

    plt.savefig(plot_path, dpi=150)
    nbody['ext']['sim'].save(sim_path)

    print(f"\nFiles saved to {output_dir}/")
    print(f"  - {os.path.basename(plot_path)}")
    print(f"  - {os.path.basename(sim_path)}")

    return (nbody['ext']['sim'], nbody['matter']['sim'],
            size_ext_final, size_lcdm_final, size_matter_final,
            ext_match, matter_match, max_ext_final, max_matter_final)


if __name__ == "__main__":
    args = parse_arguments(
        description='Run External-Node Cosmology Simulation',
        add_output_dir=True
    )

    print(f"Output directory: {os.path.abspath(args.output_dir)}\n")

    # Create simulation parameters from command-line arguments
    sim_params = args_to_sim_params(args)

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
