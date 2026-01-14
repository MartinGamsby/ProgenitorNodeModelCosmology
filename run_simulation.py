#!/usr/bin/env python3
"""
External-Node Cosmology Simulation
Main script to run the final optimized configuration (M=800, S=24, 300 particles)

Usage:
    python run_simulation.py [output_dir]
    
    output_dir: Optional output directory (default: current directory)
"""

import sys
import os
import numpy as np
from scipy.integrate import odeint
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt

from cosmo.constants import CosmologicalConstants, LambdaCDMParameters, ExternalNodeParameters
from cosmo.simulation import CosmologicalSimulation


def run_final_simulation(output_dir):
    """Run the final optimized simulation configuration"""
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*70)
    print("FINAL CONFIGURATION - M=800, S=24, Fixed Seed")
    print("="*70)
    
    # Set fixed random seed for reproducibility
    np.random.seed(42)
    
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
    t_span_full = np.linspace(0, 16.8e9 * 365.25 * 24 * 3600, 400)
    t_Gyr_full = t_span_full / (1e9 * 365.25 * 24 * 3600)
    
    a_full = odeint(friedmann_equation, a0, t_span_full,
                    args=(lcdm_params.H0, lcdm_params.Omega_m, lcdm_params.Omega_Lambda))
    a_full = a_full.flatten()
    
    # Find initial conditions at t = 10.8 Gyr
    idx_10_8 = np.argmin(np.abs(t_Gyr_full - 10.8))
    a_at_10_8 = a_full[idx_10_8]
    a_at_13_8 = np.argmin(np.abs(t_Gyr_full - 13.8))
    
    lcdm_initial_size = 14.5 * (a_at_10_8 / a_full[a_at_13_8])
    
    # Extract simulation window
    mask = (t_Gyr_full >= 10.8) & (t_Gyr_full <= 16.8)
    t_lcdm = t_Gyr_full[mask] - 10.8
    a_lcdm = a_full[mask]
    a_lcdm_normalized = a_lcdm / a_lcdm[0]
    size_lcdm = lcdm_initial_size * a_lcdm_normalized
    
    H_lcdm_raw = lcdm_params.H0 * np.sqrt(lcdm_params.Omega_m / a_lcdm**3 + lcdm_params.Omega_Lambda)
    H_lcdm_hubble = H_lcdm_raw * const.Mpc_to_m / 1000
    
    print(f"ΛCDM: {lcdm_initial_size:.3f} → {size_lcdm[-1]:.2f} Gpc")
    
    # Set up External-Node simulation
    ext_initial_size = lcdm_initial_size
    
    M = 800 * const.M_observable
    S = 24.0 * const.Gpc_to_m
    params = ExternalNodeParameters(M_ext=M, S=S)
    
    print(f"\nM=800, S=24, Ω_Λ_eff={params.Omega_Lambda_eff:.3f}")
    print(f"300 particles, fixed seed=42")
    
    sim = CosmologicalSimulation(
        n_particles=300,
        box_size_Gpc=ext_initial_size,
        use_external_nodes=True,
        external_node_params=params
    )
    
    print("\nRunning simulation...")
    sim.run(t_end_Gyr=6.0, n_steps=150, save_interval=10)
    
    # Extract results
    t_ext = np.array([h['time_Gyr'] for h in sim.expansion_history])
    a_ext = np.array([h['scale_factor'] for h in sim.expansion_history])
    size_ext = ext_initial_size * a_ext
    
    size_ext_final = size_ext[-1]
    size_lcdm_final = size_lcdm[-1]
    size_diff = abs(size_ext_final - size_lcdm_final) / size_lcdm_final * 100
    
    print(f"\nFinal: {size_ext_final:.2f} Gpc")
    print(f"Match: {100-size_diff:.2f}%")
    
    max_r_gpc = np.max(np.sqrt(np.sum(sim.snapshots[-1]['positions']**2, axis=1))) / const.Gpc_to_m
    print(f"Max particle: {max_r_gpc:.1f} Gpc")
    
    # Create visualization
    a_ext_smooth = gaussian_filter1d(a_ext, sigma=2)
    H_ext = np.gradient(a_ext_smooth, t_ext * 1e9 * 365.25 * 24 * 3600) / a_ext_smooth
    H_ext_hubble = H_ext * const.Mpc_to_m / 1000
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'ΛCDM vs External-Node (M=800, S=24, 300p, seed=42) - {100-size_diff:.1f}% match', 
                 fontsize=16, fontweight='bold')
    
    ax1 = axes[0, 0]
    ax1.plot(t_lcdm, a_lcdm_normalized, 'b-', label='ΛCDM', linewidth=2)
    ax1.plot(t_ext, a_ext, 'r--', label='External-Node', linewidth=2)
    ax1.axvline(x=3.0, color='gray', linestyle=':', alpha=0.5, label='Today')
    ax1.set_xlabel('Time [Gyr]', fontsize=11)
    ax1.set_ylabel('Scale Factor', fontsize=11)
    ax1.set_title('Cosmic Expansion', fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[0, 1]
    ax2.plot(t_lcdm, H_lcdm_hubble, 'b-', label='ΛCDM', linewidth=2)
    ax2.plot(t_ext, H_ext_hubble, 'r--', label='External-Node', linewidth=2)
    ax2.axvline(x=3.0, color='gray', linestyle=':', alpha=0.5)
    ax2.set_xlabel('Time [Gyr]', fontsize=11)
    ax2.set_ylabel('Hubble Parameter [km/s/Mpc]', fontsize=11)
    ax2.set_title('Expansion Rate', fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    ax3 = axes[1, 0]
    size_lcdm_interp = np.interp(t_ext, t_lcdm, size_lcdm)
    size_ratio = size_ext / size_lcdm_interp
    ax3.plot(t_ext, size_ratio, 'g-', linewidth=2)
    ax3.axhline(1.0, color='black', linestyle='--')
    ax3.axvline(x=3.0, color='gray', linestyle=':', alpha=0.5)
    ax3.fill_between(t_ext, 0.95, 1.05, color='green', alpha=0.2, label='±5%')
    ax3.set_xlabel('Time [Gyr]', fontsize=11)
    ax3.set_ylabel('Ratio: External-Node / ΛCDM', fontsize=11)
    ax3.set_title(f'Size Agreement ({100-size_diff:.1f}%)', fontsize=13)
    ax3.legend(fontsize=10)
    ax3.set_ylim([0.95, 1.05])
    ax3.grid(True, alpha=0.3)
    
    ax4 = axes[1, 1]
    ax4.plot(t_lcdm, size_lcdm, 'b-', label='ΛCDM', linewidth=2)
    ax4.plot(t_ext, size_ext, 'r--', label='External-Node', linewidth=2)
    ax4.axhline(24.0, color='orange', linestyle='--', label='Nodes (24 Gpc)', linewidth=2)
    ax4.axvline(x=3.0, color='gray', linestyle=':', alpha=0.5)
    ax4.set_xlabel('Time [Gyr]', fontsize=11)
    ax4.set_ylabel('Universe Radius [Gpc]', fontsize=11)
    ax4.set_title('Physical Size vs Nodes', fontsize=13)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save outputs
    plot_path = os.path.join(output_dir, 'figure_simulation_results.png')
    sim_path = os.path.join(output_dir, 'simulation_final.pkl')
    
    plt.savefig(plot_path, dpi=150)
    sim.save(sim_path)
    
    print(f"\n✓ Files saved to {output_dir}/")
    print(f"  - {os.path.basename(plot_path)}")
    print(f"  - {os.path.basename(sim_path)}")
    
    return sim, size_ext_final, size_lcdm_final, 100-size_diff


if __name__ == "__main__":
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "./results"
    
    print(f"Output directory: {os.path.abspath(output_dir)}\n")
    
    sim, ext_final, lcdm_final, match = run_final_simulation(output_dir)
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"External-Node final: {ext_final:.2f} Gpc")
    print(f"ΛCDM final: {lcdm_final:.2f} Gpc")
    print(f"Match: {match:.2f}%")
    print("\n Simulation complete!")
