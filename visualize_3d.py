#!/usr/bin/env python3
"""
3D Visualization of External-Node Cosmology Simulation

Creates:
1. Interactive 3D plot showing particles and external nodes
2. Animated video showing universe expansion
3. Multi-panel snapshots at different times

Usage:
    python visualize_3d.py [simulation.pkl] [output_dir]
    
    If no simulation file provided, runs a quick simulation first.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D
import pickle

# Try to import simulation code if available
try:
    from cosmo.constants import CosmologicalConstants, ExternalNodeParameters
    from cosmo.simulation import CosmologicalSimulation
    HAS_SIM = True
except ImportError:
    HAS_SIM = False
    print("Note: Simulation modules not found. Provide a .pkl file to visualize.")


def load_or_run_simulation(sim_file=None, output_dir="."):
    """Load existing simulation or run a quick one"""
    
    if sim_file and os.path.exists(sim_file):
        print(f"Loading simulation from {sim_file}")
        with open(sim_file, 'rb') as f:
            sim_data = pickle.load(f)            
        return sim_data
    
    if not HAS_SIM:
        print("ERROR: Cannot run simulation without source modules")
        print("Please provide a .pkl file or ensure src/ is in Python path")
        sys.exit(1)
    
    print("Running quick simulation (100 particles)...")
    np.random.seed(42)
    
    const = CosmologicalConstants()
    
    # Quick simulation parameters
    M = 800 * const.M_observable
    S = 24.0 * const.Gpc_to_m
    params = ExternalNodeParameters(M_ext=M, S=S)
    
    sim = CosmologicalSimulation(
        n_particles=100,  # Fewer particles for faster viz
        box_size_Gpc=11.59,
        use_external_nodes=True,
        external_node_params=params
    )
    
    sim.run(t_end_Gyr=6.0, n_steps=120, save_interval=5)  # More frequent snapshots
    
    # Save for future use
    sim_file = os.path.join(output_dir, 'visualization_sim.pkl')
    sim.save(sim_file)
    print(f"Saved simulation to {sim_file}")
    
    # Convert to data dict
    sim_data = {
        'snapshots': sim.snapshots,
        'expansion_history': sim.expansion_history,
        'params': {
            'M_ext': M,
            'S': S / const.Gpc_to_m,  # Convert to Gpc
            'initial_size': 11.59
        }
    }
    
    return sim_data


def get_node_positions(S_Gpc):
    """Get positions of 26 external nodes in a 3×3×3 grid"""
    positions = []
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            for k in [-1, 0, 1]:
                if i == 0 and j == 0 and k == 0:
                    continue  # Skip center (our universe)
                positions.append([i * S_Gpc, j * S_Gpc, k * S_Gpc])
    return np.array(positions)


def create_3d_snapshot(sim_data, snapshot_idx, output_dir="."):
    """Create a single 3D visualization at a given time"""
    
    snapshot = sim_data['snapshots'][snapshot_idx]
    history = sim_data['expansion_history'][snapshot_idx]
    
    # Extract parameters
    S_Gpc = sim_data['params']['S']
    initial_size = sim_data['params']['initial_size']
    
    # Get particle positions (convert to Gpc)
    const_val = 3.086e25  # Gpc to meters
    positions = snapshot['positions'] / const_val
    
    # Current universe size
    scale_factor = history['scale_factor']
    current_size = initial_size * scale_factor
    time_Gyr = history['time_Gyr']
    
    # Get node positions
    node_positions = get_node_positions(S_Gpc)
    
    # Create figure
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot particles
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
               c='blue', s=20, alpha=0.6, label='Particles')
    
    # Plot external nodes
    ax.scatter(node_positions[:, 0], node_positions[:, 1], node_positions[:, 2],
               c='red', s=200, marker='*', alpha=0.8, 
               edgecolors='darkred', linewidth=1.5,
               label=f'External Nodes (26)')
    
    # Draw sphere representing universe boundary
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x_sphere = current_size * np.outer(np.cos(u), np.sin(v))
    y_sphere = current_size * np.outer(np.sin(u), np.sin(v))
    z_sphere = current_size * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.1, color='cyan')
    
    # Draw cube representing node grid
    def draw_cube_edges(ax, size, color='orange', alpha=0.3):
        # Define cube vertices
        r = [-size, size]
        for s, e in [[[r[0], r[0], r[0]], [r[1], r[0], r[0]]],
                     [[r[0], r[1], r[0]], [r[1], r[1], r[0]]],
                     [[r[0], r[0], r[1]], [r[1], r[0], r[1]]],
                     [[r[0], r[1], r[1]], [r[1], r[1], r[1]]],
                     [[r[0], r[0], r[0]], [r[0], r[1], r[0]]],
                     [[r[1], r[0], r[0]], [r[1], r[1], r[0]]],
                     [[r[0], r[0], r[1]], [r[0], r[1], r[1]]],
                     [[r[1], r[0], r[1]], [r[1], r[1], r[1]]],
                     [[r[0], r[0], r[0]], [r[0], r[0], r[1]]],
                     [[r[1], r[0], r[0]], [r[1], r[0], r[1]]],
                     [[r[0], r[1], r[0]], [r[0], r[1], r[1]]],
                     [[r[1], r[1], r[0]], [r[1], r[1], r[1]]]]:
            ax.plot3D(*zip(s, e), color=color, alpha=alpha, linewidth=1)
    
    draw_cube_edges(ax, S_Gpc)
    
    # Set limits
    lim = S_Gpc * 1.2
    ax.set_xlim([-lim, lim])
    ax.set_ylim([-lim, lim])
    ax.set_zlim([-lim, lim])
    
    # Labels
    ax.set_xlabel('X [Gpc]', fontsize=11)
    ax.set_ylabel('Y [Gpc]', fontsize=11)
    ax.set_zlabel('Z [Gpc]', fontsize=11)
    
    # Title with info
    ax.set_title(
        f'External-Node Cosmology (t = {time_Gyr:.2f} Gyr)\n'
        f'Universe Radius: {current_size:.1f} Gpc | Node Distance: {S_Gpc:.0f} Gpc',
        fontsize=13, fontweight='bold'
    )
    
    ax.legend(loc='upper right', fontsize=10)
    
    # Set viewing angle
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    
    # Save
    filename = os.path.join(output_dir, f'3d_snapshot_t{time_Gyr:.1f}Gyr.png')
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved {filename}")
    
    return filename


def create_multi_panel_evolution(sim_data, output_dir="."):
    """Create a multi-panel figure showing evolution over time"""
    
    # Select 6 evenly spaced snapshots
    n_snapshots = len(sim_data['snapshots'])
    indices = np.linspace(0, n_snapshots-1, 6, dtype=int)
    
    # Extract parameters
    S_Gpc = sim_data['params']['S']
    initial_size = sim_data['params']['initial_size']
    const_val = 3.086e25
    
    # Get node positions (same for all times)
    node_positions = get_node_positions(S_Gpc)
    
    # Create figure
    fig = plt.figure(figsize=(18, 12))
    
    for panel_idx, snap_idx in enumerate(indices):
        snapshot = sim_data['snapshots'][snap_idx]
        history = sim_data['expansion_history'][snap_idx]
        
        positions = snapshot['positions'] / const_val
        scale_factor = history['scale_factor']
        current_size = initial_size * scale_factor
        time_Gyr = history['time_Gyr']
        
        # Create subplot
        ax = fig.add_subplot(2, 3, panel_idx + 1, projection='3d')
        
        # Plot particles
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                   c='blue', s=10, alpha=0.5)
        
        # Plot nodes
        ax.scatter(node_positions[:, 0], node_positions[:, 1], node_positions[:, 2],
                   c='red', s=100, marker='*', alpha=0.8, edgecolors='darkred')
        
        # Universe boundary
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 30)
        x_sphere = current_size * np.outer(np.cos(u), np.sin(v))
        y_sphere = current_size * np.outer(np.sin(u), np.sin(v))
        z_sphere = current_size * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.1, color='cyan')
        
        # Set limits
        lim = S_Gpc * 1.1
        ax.set_xlim([-lim, lim])
        ax.set_ylim([-lim, lim])
        ax.set_zlim([-lim, lim])
        
        ax.set_xlabel('X [Gpc]', fontsize=9)
        ax.set_ylabel('Y [Gpc]', fontsize=9)
        ax.set_zlabel('Z [Gpc]', fontsize=9)
        
        ax.set_title(f't = {time_Gyr:.1f} Gyr\nR = {current_size:.1f} Gpc',
                     fontsize=11, fontweight='bold')
        
        ax.view_init(elev=20, azim=45)
        ax.set_box_aspect([1,1,1])
    
    fig.suptitle('Universe Expansion Over 6 Billion Years', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    filename = os.path.join(output_dir, '3d_evolution_multipanel.png')
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved {filename}")
    
    return filename


def create_animation(sim_data, output_dir=".", fps=10):
    """Create animated GIF of universe expansion"""
    
    print("Creating animation...")
    
    # Extract parameters
    S_Gpc = sim_data['params']['S']
    initial_size = sim_data['params']['initial_size']
    const_val = 3.086e25
    
    # Get node positions
    node_positions = get_node_positions(S_Gpc)
    
    # Create figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Set limits (fixed for all frames)
    lim = S_Gpc * 1.2
    ax.set_xlim([-lim, lim])
    ax.set_ylim([-lim, lim])
    ax.set_zlim([-lim, lim])
    
    ax.set_xlabel('X [Gpc]', fontsize=11)
    ax.set_ylabel('Y [Gpc]', fontsize=11)
    ax.set_zlabel('Z [Gpc]', fontsize=11)
    
    # Initialize empty plots
    particles_plot = ax.scatter([], [], [], c='blue', s=20, alpha=0.6)
    nodes_plot = ax.scatter(node_positions[:, 0], node_positions[:, 1], node_positions[:, 2],
                            c='red', s=200, marker='*', alpha=0.8, 
                            edgecolors='darkred', linewidth=1.5)
    
    title = ax.text2D(0.5, 0.95, '', transform=ax.transAxes,
                      ha='center', fontsize=13, fontweight='bold')
    
    # Prepare sphere for universe boundary
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 30)
    u_mesh, v_mesh = np.meshgrid(u, v)
    
    sphere_plot = None
    
    def init():
        particles_plot._offsets3d = ([], [], [])
        return particles_plot, nodes_plot, title
    
    def update(frame):
        nonlocal sphere_plot
        
        snapshot = sim_data['snapshots'][frame]
        history = sim_data['expansion_history'][frame]
        
        positions = snapshot['positions'] / const_val
        scale_factor = history['scale_factor']
        current_size = initial_size * scale_factor
        time_Gyr = history['time_Gyr']
        
        # Update particles
        particles_plot._offsets3d = (positions[:, 0], positions[:, 1], positions[:, 2])
        
        # Update universe boundary
        if sphere_plot is not None:
            sphere_plot.remove()
        
        x_sphere = current_size * np.outer(np.cos(u), np.sin(v))
        y_sphere = current_size * np.outer(np.sin(u), np.sin(v))
        z_sphere = current_size * np.outer(np.ones(np.size(u)), np.cos(v))
        sphere_plot = ax.plot_surface(x_sphere, y_sphere, z_sphere, 
                                       alpha=0.1, color='cyan')
        
        # Update title
        title.set_text(
            f'External-Node Cosmology: t = {time_Gyr:.2f} Gyr\n'
            f'Universe Radius: {current_size:.1f} Gpc | Nodes at {S_Gpc:.0f} Gpc'
        )
        
        # Rotate view slightly
        ax.view_init(elev=20, azim=45 + frame * 2)
        
        return particles_plot, nodes_plot, title
    
    # Create animation
    n_frames = len(sim_data['snapshots'])
    anim = FuncAnimation(fig, update, init_func=init, frames=n_frames,
                         interval=1000//fps, blit=False)
    
    # Save as GIF
    filename = os.path.join(output_dir, '3d_animation.gif')
    writer = PillowWriter(fps=fps)
    anim.save(filename, writer=writer)
    plt.close()
    
    print(f"✓ Saved {filename}")
    print(f"  {n_frames} frames at {fps} fps = {n_frames/fps:.1f} seconds")
    
    return filename


def main():
    """Main visualization function"""
    
    # Parse arguments
    if len(sys.argv) > 1:
        sim_file = sys.argv[1]
    else:
        sim_file = None
    
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    else:
        output_dir = "."
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*70)
    print("3D VISUALIZATION OF EXTERNAL-NODE COSMOLOGY")
    print("="*70)
    print(f"Output directory: {os.path.abspath(output_dir)}\n")
    
    # Load or run simulation
    sim_data = load_or_run_simulation(sim_file, output_dir)
    
    print(f"\nSimulation has {len(sim_data['snapshots'])} snapshots")
    print(f"Time range: 0 to {sim_data['expansion_history'][-1]['time_Gyr']:.1f} Gyr")
    
    # Create visualizations
    print("\n" + "="*70)
    print("CREATING VISUALIZATIONS")
    print("="*70)
    
    # 1. Multi-panel evolution
    print("\n1. Multi-panel evolution figure...")
    create_multi_panel_evolution(sim_data, output_dir)
    
    # 2. Key snapshots
    print("\n2. Creating key snapshots...")
    n_snapshots = len(sim_data['snapshots'])
    for idx in [0, n_snapshots//4, n_snapshots//2, 3*n_snapshots//4, n_snapshots-1]:
        create_3d_snapshot(sim_data, idx, output_dir)
    
    # 3. Animation
    print("\n3. Creating animation (this may take a minute)...")
    create_animation(sim_data, output_dir, fps=10)
    
    print("\n" + "="*70)
    print("✨ VISUALIZATION COMPLETE!")
    print("="*70)
    print(f"\nAll files saved to: {os.path.abspath(output_dir)}")
    print("\nGenerated files:")
    print("  - 3d_evolution_multipanel.png  (6-panel overview)")
    print("  - 3d_snapshot_t*.png           (individual snapshots)")
    print("  - 3d_animation.gif             (animated expansion)")
    
    print("\nTips:")
    print("  - View the GIF in a browser or image viewer")
    print("  - Use multipanel for presentations")
    print("  - Individual snapshots show detail at specific times")


if __name__ == "__main__":
    main()
