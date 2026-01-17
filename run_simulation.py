#!/usr/bin/env python3
"""
External-Node Cosmology Simulation
Main script to run cosmology simulations with configurable parameters.
"""

import sys
import os
import argparse
import numpy as np
from datetime import datetime
from scipy.integrate import odeint
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt

from cosmo.constants import CosmologicalConstants, LambdaCDMParameters, SimulationParameters
from cosmo.simulation import CosmologicalSimulation


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
    
    # Define Friedmann equation for ΛCDM
    def friedmann_equation(a, t, H0, Omega_m, Omega_Lambda):
        if a <= 0:
            return 1e-10
        H = H0 * np.sqrt(Omega_m / a**3 + Omega_Lambda)
        return H * a
    
    # Solve ΛCDM evolution
    a0 = 0.001
    t_max = sim_params.t_end_Gyr + 3.0  # Add buffer beyond end time
    t_span_full = np.linspace(0, t_max * 1e9 * 365.25 * 24 * 3600, 400)
    t_Gyr_full = t_span_full / (1e9 * 365.25 * 24 * 3600)

    a_full = odeint(friedmann_equation, a0, t_span_full,
                    args=(lcdm_params.H0, lcdm_params.Omega_m, lcdm_params.Omega_Lambda))
    a_full = a_full.flatten()

    # Find initial conditions at start time
    idx_start = np.argmin(np.abs(t_Gyr_full - sim_params.t_start_Gyr))
    a_at_start = a_full[idx_start]
    idx_today = np.argmin(np.abs(t_Gyr_full - 13.8))

    lcdm_initial_size = 14.5 * (a_at_start / a_full[idx_today])

    # Extract simulation window
    mask = (t_Gyr_full >= sim_params.t_start_Gyr) & (t_Gyr_full <= sim_params.t_end_Gyr)
    t_lcdm = t_Gyr_full[mask] - sim_params.t_start_Gyr
    a_lcdm = a_full[mask]
    a_lcdm_normalized = a_lcdm / a_lcdm[0]
    size_lcdm = lcdm_initial_size * a_lcdm_normalized

    H_lcdm_raw = lcdm_params.H0 * np.sqrt(lcdm_params.Omega_m / a_lcdm**3 + lcdm_params.Omega_Lambda)
    H_lcdm_hubble = H_lcdm_raw * const.Mpc_to_m / 1000

    print(f"LCDM: {lcdm_initial_size:.3f} -> {size_lcdm[-1]:.2f} Gpc")

    # Solve matter-only evolution (Omega_Lambda=0)
    a_matter_full = odeint(friedmann_equation, a0, t_span_full,
                           args=(lcdm_params.H0, lcdm_params.Omega_m, 0.0))  # Omega_Lambda=0
    a_matter_full = a_matter_full.flatten()

    # Extract matter-only window
    a_matter = a_matter_full[mask]
    a_matter_normalized = a_matter / a_matter[0]
    size_matter = lcdm_initial_size * a_matter_normalized

    H_matter_raw = lcdm_params.H0 * np.sqrt(lcdm_params.Omega_m / a_matter**3)
    H_matter_hubble = H_matter_raw * const.Mpc_to_m / 1000

    print(f"Matter-only: {lcdm_initial_size:.3f} -> {size_matter[-1]:.2f} Gpc")
    
    # Set up External-Node simulation
    ext_initial_size = lcdm_initial_size

    print(f"\nM={sim_params.M_value}, S={sim_params.S_value}, Omega_Lambda_eff={sim_params.external_params.Omega_Lambda_eff:.3f}")
    print(f"{sim_params.n_particles} particles, seed={sim_params.seed}")

    # Calculate and display H at start time for verification
    H_start = lcdm_params.H_at_time(a_at_start)
    H_start_hubble = H_start * const.Mpc_to_m / 1000
    print(f"H(t_start={sim_params.t_start_Gyr} Gyr) = {H_start_hubble:.1f} km/s/Mpc")

    # Reset seed before External-Node simulation for reproducibility
    np.random.seed(sim_params.seed)
    sim = CosmologicalSimulation(
        n_particles=sim_params.n_particles,
        box_size_Gpc=ext_initial_size,
        use_external_nodes=True,
        external_node_params=sim_params.external_params,
        t_start_Gyr=sim_params.t_start_Gyr,
        a_start=a_at_start,
        use_dark_energy=False,  # Explicitly disable dark energy for matter-only
        damping_factor=sim_params.damping_factor
    )

    print("\nRunning External-Node simulation...")
    sim.run(t_end_Gyr=sim_params.t_duration_Gyr, n_steps=sim_params.n_steps, save_interval=10)

    # Extract External-Node results
    t_ext = np.array([h['time_Gyr'] for h in sim.expansion_history])
    a_ext = np.array([h['scale_factor'] for h in sim.expansion_history])
    size_ext = ext_initial_size * a_ext

    # Run matter-only simulation (no external nodes, no dark energy)
    print("\nRunning Matter-only simulation...")
    # Reset seed to ensure identical initial conditions as External-Node
    np.random.seed(sim_params.seed)
    sim_matter = CosmologicalSimulation(
        n_particles=sim_params.n_particles,
        box_size_Gpc=ext_initial_size,
        use_external_nodes=False,
        external_node_params=None,
        t_start_Gyr=sim_params.t_start_Gyr,
        a_start=a_at_start,
        use_dark_energy=False,  # Explicitly disable dark energy for matter-only
        damping_factor=sim_params.damping_factor
    )
    sim_matter.run(t_end_Gyr=sim_params.t_duration_Gyr, n_steps=sim_params.n_steps, save_interval=10)

    # Extract Matter-only results
    t_matter = np.array([h['time_Gyr'] for h in sim_matter.expansion_history])
    a_matter_sim = np.array([h['scale_factor'] for h in sim_matter.expansion_history])
    size_matter_sim = ext_initial_size * a_matter_sim
    
    size_ext_final = size_ext[-1]
    size_lcdm_final = size_lcdm[-1]
    size_diff = abs(size_ext_final - size_lcdm_final) / size_lcdm_final * 100

    print(f"\n{size_ext_final:.2f} Gpc")
    print(f"Match: {100-size_diff:.2f}%")
    
    max_r_gpc = np.max(np.sqrt(np.sum(sim.snapshots[-1]['positions']**2, axis=1))) / const.Gpc_to_m
    print(f"Max particle: {max_r_gpc:.1f} Gpc")
    
    # Create visualization
    a_ext_smooth = gaussian_filter1d(a_ext, sigma=2)
    H_ext = np.gradient(a_ext_smooth, t_ext * 1e9 * 365.25 * 24 * 3600) / a_ext_smooth
    H_ext_hubble = H_ext * const.Mpc_to_m / 1000

    a_matter_sim_smooth = gaussian_filter1d(a_matter_sim, sigma=2)
    H_matter_sim = np.gradient(a_matter_sim_smooth, t_matter * 1e9 * 365.25 * 24 * 3600) / a_matter_sim_smooth
    H_matter_sim_hubble = H_matter_sim * const.Mpc_to_m / 1000

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Cosmology Comparison (M={sim_params.M_value}, S={sim_params.S_value}, {sim_params.n_particles}p)',
                 fontsize=16, fontweight='bold')


    today = None
    if (sim_params.t_start_Gyr < 13.8) and ((sim_params.t_start_Gyr + sim_params.t_duration_Gyr) > 13.8):
        today = 13.8 - sim_params.t_start_Gyr
    ax1 = axes[0, 0]
    ax1.plot(t_lcdm, a_lcdm_normalized, 'b-', label='LCDM (with dark energy)', linewidth=2)
    ax1.plot(t_ext, a_ext, 'r--', label='External-Node', linewidth=2)
    ax1.plot(t_matter, a_matter_sim, 'g:', label='Matter-only (no dark energy)', linewidth=2)
    if today:
        ax1.axvline(x=today, color='gray', linestyle=':', alpha=0.5, label='Today')
    ax1.set_xlabel('Time [Gyr]', fontsize=11)
    ax1.set_ylabel('Scale Factor', fontsize=11)
    ax1.set_title('Cosmic Expansion', fontsize=13)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2 = axes[0, 1]
    ax2.plot(t_lcdm, H_lcdm_hubble, 'b-', label='LCDM', linewidth=2)
    ax2.plot(t_ext, H_ext_hubble, 'r--', label='External-Node', linewidth=2)
    ax2.plot(t_matter, H_matter_sim_hubble, 'g:', label='Matter-only', linewidth=2)
    if today:
        ax2.axvline(x=today, color='gray', linestyle=':', alpha=0.5)
    ax2.set_xlabel('Time [Gyr]', fontsize=11)
    ax2.set_ylabel('Hubble Parameter [km/s/Mpc]', fontsize=11)
    ax2.set_title('Expansion Rate', fontsize=13)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    ax3 = axes[1, 0]
    size_lcdm_interp = np.interp(t_ext, t_lcdm, size_lcdm)
    size_ratio_ext = size_ext / size_lcdm_interp
    size_matter_interp = np.interp(t_matter, t_lcdm, size_lcdm)
    size_ratio_matter = size_matter_sim / size_matter_interp
    ax3.plot(t_ext, size_ratio_ext, 'r--', label='External-Node', linewidth=2)
    ax3.plot(t_matter, size_ratio_matter, 'g:', label='Matter-only', linewidth=2)
    ax3.axhline(1.0, color='black', linestyle='--', label='LCDM')
    if today:
        ax3.axvline(x=today, color='gray', linestyle=':', alpha=0.5)
    ax3.fill_between(t_ext, 0.90, 1.10, color='blue', alpha=0.1)
    ax3.set_xlabel('Time [Gyr]', fontsize=11)
    ax3.set_ylabel('Ratio to LCDM', fontsize=11)
    ax3.set_title('Relative Expansion', fontsize=13)
    ax3.legend(fontsize=9)
    ax3.set_ylim([0.85, 1.15])
    ax3.grid(True, alpha=0.3)

    ax4 = axes[1, 1]
    ax4.plot(t_lcdm, size_lcdm, 'b-', label='LCDM', linewidth=2)
    ax4.plot(t_ext, size_ext, 'r--', label='External-Node', linewidth=2)
    ax4.plot(t_matter, size_matter_sim, 'g:', label='Matter-only', linewidth=2)
    ax4.axhline(sim_params.S_value, color='orange', linestyle='--', label=f'Nodes ({sim_params.S_value} Gpc)', linewidth=2)
    if today:
        ax4.axvline(x=today, color='gray', linestyle=':', alpha=0.5)
    ax4.set_xlabel('Time [Gyr]', fontsize=11)
    ax4.set_ylabel('Universe Radius [Gpc]', fontsize=11)
    ax4.set_title('Physical Size', fontsize=13)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save outputs
    plot_path = os.path.join(output_dir, f'figure_simulation_results_{datetime.now().strftime("%Y-%m-%d_%H.%M.%S")}_{sim_params.n_particles}p_{sim_params.t_start_Gyr}-{sim_params.t_start_Gyr+sim_params.t_duration_Gyr}Gyr_{sim_params.M_value}M_{sim_params.S_value}S_{sim_params.n_steps}steps_{sim_params.damping_factor}d.png')
    sim_path = os.path.join(output_dir, 'simulation.pkl')

    plt.savefig(plot_path, dpi=150)
    sim.save(sim_path)
    
    print(f"\nFiles saved to {output_dir}/")
    print(f"  - {os.path.basename(plot_path)}")
    print(f"  - {os.path.basename(sim_path)}")

    # Calculate final sizes and comparisons
    size_matter_sim_final = size_matter_sim[-1]
    ext_vs_lcdm_diff = 100 - size_diff
    matter_vs_lcdm_diff = (1 - size_matter_sim_final / size_lcdm_final) * 100

    return sim, sim_matter, size_ext_final, size_lcdm_final, size_matter_sim_final, ext_vs_lcdm_diff, matter_vs_lcdm_diff


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

    sim, sim_matter, ext_final, lcdm_final, matter_final, ext_match, matter_match = run_simulation(args.output_dir, sim_params)

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"LCDM (with dark energy):        {lcdm_final:.2f} Gpc")
    print(f"External-Node (M={sim_params.M_value}, S={sim_params.S_value}): {ext_final:.2f} Gpc  ({ext_match:+.1f}% vs LCDM)")
    print(f"Matter-only (no dark energy):   {matter_final:.2f} Gpc  ({matter_match:+.1f}% vs LCDM)")
    print("\nSimulation complete!")
