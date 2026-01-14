"""
Visualization Tools
Create plots and animations of simulation results
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D

from cosmo.constants import CosmologicalConstants


def plot_particle_distribution_3d(snapshot, hmea_grid=None, save_path=None):
    """
    Plot 3D particle distribution at a given snapshot
    
    Parameters:
    -----------
    snapshot : dict
        Snapshot containing positions, velocities
    hmea_grid : HMEAGrid, optional
        HMEA nodes to plot
    save_path : str, optional
        Path to save figure
    """
    const = CosmologicalConstants()
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Get positions in Gpc
    positions = snapshot['positions'] / const.Gpc_to_m
    
    # Plot particles
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
              c='blue', marker='o', s=20, alpha=0.6, label='Galaxies')
    
    # Plot HMEA nodes if provided
    if hmea_grid is not None:
        node_pos = hmea_grid.get_positions() / const.Gpc_to_m
        ax.scatter(node_pos[:, 0], node_pos[:, 1], node_pos[:, 2],
                  c='red', marker='*', s=500, alpha=0.8, 
                  edgecolors='black', linewidths=2,
                  label='HMEA Nodes')
        
        # Draw lines from origin to nodes
        for pos in node_pos:
            ax.plot([0, pos[0]], [0, pos[1]], [0, pos[2]], 
                   'r--', alpha=0.3, linewidth=1)
    
    # Plot origin
    ax.scatter([0], [0], [0], c='yellow', marker='x', s=200, 
              linewidths=3, label='Big Bang Origin')
    
    ax.set_xlabel('X [Gpc]', fontsize=12)
    ax.set_ylabel('Y [Gpc]', fontsize=12)
    ax.set_zlabel('Z [Gpc]', fontsize=12)
    ax.set_title(f"Universe at t = {snapshot['time']/(1e9*365.25*24*3600):.2f} Gyr", 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    
    # Equal aspect ratio
    max_range = np.max(np.abs(positions))
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved 3D plot to {save_path}")
    
    return fig


def plot_velocity_distribution(snapshot, save_path=None):
    """
    Plot velocity distribution histogram
    
    Parameters:
    -----------
    snapshot : dict
        Snapshot containing velocities
    save_path : str, optional
        Path to save figure
    """
    velocities = snapshot['velocities']
    v_mag = np.linalg.norm(velocities, axis=1)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Velocity magnitude histogram
    ax1 = axes[0]
    ax1.hist(v_mag / 1000, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Velocity [km/s]', fontsize=12)
    ax1.set_ylabel('Number of Particles', fontsize=12)
    ax1.set_title('Velocity Magnitude Distribution', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Velocity components
    ax2 = axes[1]
    ax2.hist(velocities[:, 0]/1000, bins=30, alpha=0.5, label='v_x', color='red')
    ax2.hist(velocities[:, 1]/1000, bins=30, alpha=0.5, label='v_y', color='green')
    ax2.hist(velocities[:, 2]/1000, bins=30, alpha=0.5, label='v_z', color='blue')
    ax2.set_xlabel('Velocity Component [km/s]', fontsize=12)
    ax2.set_ylabel('Number of Particles', fontsize=12)
    ax2.set_title('Velocity Component Distributions', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved velocity plot to {save_path}")
    
    return fig


def plot_radial_profiles(snapshot, n_bins=20, save_path=None):
    """
    Plot radial density and velocity profiles
    
    Parameters:
    -----------
    snapshot : dict
        Snapshot containing positions, velocities
    n_bins : int
        Number of radial bins
    save_path : str, optional
        Path to save figure
    """
    const = CosmologicalConstants()
    
    positions = snapshot['positions']
    velocities = snapshot['velocities']
    
    # Calculate distances from origin
    r = np.linalg.norm(positions, axis=1)
    r_Gpc = r / const.Gpc_to_m
    
    # Radial velocities
    v_radial = np.sum(positions * velocities, axis=1) / r
    
    # Create bins
    r_max = np.max(r_Gpc)
    bins = np.linspace(0, r_max, n_bins + 1)
    r_centers = (bins[:-1] + bins[1:]) / 2
    
    # Density profile (counts per bin)
    counts, _ = np.histogram(r_Gpc, bins=bins)
    
    # Velocity profile (mean radial velocity per bin)
    v_profile = []
    for i in range(n_bins):
        mask = (r_Gpc >= bins[i]) & (r_Gpc < bins[i+1])
        if np.sum(mask) > 0:
            v_profile.append(np.mean(v_radial[mask]) / 1000)  # km/s
        else:
            v_profile.append(0)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Density profile
    ax1 = axes[0]
    ax1.bar(r_centers, counts, width=r_max/n_bins*0.8, color='steelblue', 
           alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Radius [Gpc]', fontsize=12)
    ax1.set_ylabel('Number of Particles', fontsize=12)
    ax1.set_title('Radial Density Profile', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Velocity profile
    ax2 = axes[1]
    ax2.plot(r_centers, v_profile, 'o-', color='darkred', 
            linewidth=2, markersize=8, alpha=0.7)
    ax2.axhline(0, color='black', linestyle='--', linewidth=1)
    ax2.set_xlabel('Radius [Gpc]', fontsize=12)
    ax2.set_ylabel('Mean Radial Velocity [km/s]', fontsize=12)
    ax2.set_title('Radial Velocity Profile', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved radial profiles to {save_path}")
    
    return fig


def create_animation(snapshots, hmea_grid=None, output_path='universe_evolution.gif', fps=10):
    """
    Create animated GIF of universe evolution
    
    Parameters:
    -----------
    snapshots : list
        List of snapshot dictionaries
    hmea_grid : HMEAGrid, optional
        HMEA nodes to include
    output_path : str
        Path to save animation
    fps : int
        Frames per second
    """
    const = CosmologicalConstants()
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Determine plot limits
    all_positions = np.vstack([snap['positions'] for snap in snapshots])
    max_range = np.max(np.abs(all_positions)) / const.Gpc_to_m * 1.1
    
    def update(frame):
        ax.clear()
        
        snapshot = snapshots[frame]
        positions = snapshot['positions'] / const.Gpc_to_m
        
        # Plot particles
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                  c='blue', marker='o', s=20, alpha=0.6)
        
        # Plot HMEA nodes
        if hmea_grid is not None:
            node_pos = hmea_grid.get_positions() / const.Gpc_to_m
            ax.scatter(node_pos[:, 0], node_pos[:, 1], node_pos[:, 2],
                      c='red', marker='*', s=500, alpha=0.8,
                      edgecolors='black', linewidths=2)
        
        # Origin
        ax.scatter([0], [0], [0], c='yellow', marker='x', s=200, linewidths=3)
        
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([-max_range, max_range])
        ax.set_xlabel('X [Gpc]', fontsize=12)
        ax.set_ylabel('Y [Gpc]', fontsize=12)
        ax.set_zlabel('Z [Gpc]', fontsize=12)
        
        t_Gyr = snapshot['time'] / (1e9 * 365.25 * 24 * 3600)
        ax.set_title(f"Universe Evolution - t = {t_Gyr:.2f} Gyr", 
                    fontsize=14, fontweight='bold')
    
    anim = FuncAnimation(fig, update, frames=len(snapshots), interval=1000/fps)
    
    print(f"Creating animation with {len(snapshots)} frames...")
    writer = PillowWriter(fps=fps)
    anim.save(output_path, writer=writer)
    print(f"Saved animation to {output_path}")
    
    plt.close()


def analyze_simulation(simulation_data, output_dir=''):
    """
    Generate comprehensive analysis plots for a simulation
    
    Parameters:
    -----------
    simulation_data : dict
        Loaded simulation data
    output_dir : str
        Directory to save plots
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    snapshots = simulation_data['snapshots']
    model_type = 'External-Node' if simulation_data['use_external_nodes'] else 'ΛCDM'
    
    print(f"\n{'='*60}")
    print(f"ANALYZING {model_type} SIMULATION")
    print(f"{'='*60}")
    print(f"Total snapshots: {len(snapshots)}")
    print(f"Particles: {simulation_data['n_particles']}")
    
    # Initial state
    print("\n1. Plotting initial state...")
    plot_particle_distribution_3d(
        snapshots[0], 
        save_path=f'{output_dir}/{model_type.lower()}_initial_3d.png'
    )
    
    # Final state
    print("2. Plotting final state...")
    plot_particle_distribution_3d(
        snapshots[-1],
        save_path=f'{output_dir}/{model_type.lower()}_final_3d.png'
    )
    
    # Velocity distributions
    print("3. Plotting velocity distributions...")
    plot_velocity_distribution(
        snapshots[-1],
        save_path=f'{output_dir}/{model_type.lower()}_velocities.png'
    )
    
    # Radial profiles
    print("4. Plotting radial profiles...")
    plot_radial_profiles(
        snapshots[-1],
        save_path=f'{output_dir}/{model_type.lower()}_radial.png'
    )
    
    print(f"\n{'='*60}")
    print("Analysis complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # Test visualization with dummy data
    print("Testing visualization tools...")
    
    # Create dummy snapshot
    np.random.seed(42)
    dummy_snapshot = {
        'time': 13.8e9 * 365.25 * 24 * 3600,
        'positions': np.random.randn(100, 3) * 1e25,
        'velocities': np.random.randn(100, 3) * 1e5,
    }
    
    plot_particle_distribution_3d(dummy_snapshot, 
                                 save_path='test_3d.png')
    plot_velocity_distribution(dummy_snapshot,
                              save_path='test_velocity.png')
    plot_radial_profiles(dummy_snapshot,
                        save_path='test_radial.png')
    
    print("Visualization test complete!")
