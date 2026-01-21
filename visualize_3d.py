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
    from cosmo.constants import CosmologicalConstants, ExternalNodeParameters, SimulationParameters
    from cosmo.simulation import CosmologicalSimulation
    from cosmo.analysis import calculate_initial_conditions
    from cosmo.visualization import (
        get_node_positions,
        draw_universe_sphere,
        draw_cube_edges,
        setup_3d_axes
    )
    HAS_SIM = True
except ImportError:
    HAS_SIM = False
    print("Note: Simulation modules not found. Provide a .pkl file to visualize.")


START_TIME = 3.8  # Gyr

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

    # Calculate initial conditions using shared function
    sim_params = SimulationParameters(
        M_value=1000,
        S_value=30.0,
        n_particles=200,
        seed=42,
        t_start_Gyr=START_TIME,
        t_duration_Gyr=10.0*4/3,
        n_steps=1000,
        damping_factor=0.9
    )

    initial_conditions = calculate_initial_conditions(sim_params.t_start_Gyr)

    sim = CosmologicalSimulation(
        sim_params=sim_params,
        box_size_Gpc=initial_conditions['box_size_Gpc'],
        a_start=initial_conditions['a_start'],
        use_external_nodes=True,
        use_dark_energy=False
    )

    sim.run(t_end_Gyr=sim_params.t_duration_Gyr, n_steps=sim_params.n_steps, save_interval=20)

    # Save for future use
    sim_file = os.path.join(output_dir, 'visualization_sim.pkl')
    sim.save(sim_file)
    print(f"Saved simulation to {sim_file}")

    # Convert to data dict
    sim_data = {
        'snapshots': sim.snapshots,
        'expansion_history': sim.expansion_history,
        'params': {
            'M_ext': sim_params.M_value * CosmologicalConstants().M_observable,
            'S': sim_params.S_value,
            'initial_size': initial_conditions['box_size_Gpc']
        }
    }

    return sim_data


def create_3d_snapshot(sim_data, snapshot_idx, output_dir="."):
    """Create a single 3D visualization at a given time"""

    snapshot = sim_data['snapshots'][snapshot_idx]
    history = sim_data['expansion_history'][snapshot_idx]

    # Extract parameters
    S_Gpc = sim_data['params']['S']
    const = CosmologicalConstants()

    # Get particle positions (convert to Gpc)
    positions = snapshot['positions'] / const.Gpc_to_m

    # Current universe size
    current_size = history['size'] / const.Gpc_to_m
    current_max_size = history['max_particle_distance'] / const.Gpc_to_m
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

    # Draw spheres representing universe boundaries
    draw_universe_sphere(ax, current_max_size)
    draw_universe_sphere(ax, current_size/2)

    # Draw cube representing node grid
    draw_cube_edges(ax, S_Gpc)

    # Setup axes and labels
    title = (f'External-Node Cosmology (t = {(time_Gyr+START_TIME):.2f} Gyr)\n'
             f'Universe Radius: {current_size:.1f} Gpc | Node Distance: {S_Gpc:.0f} Gpc')
    setup_3d_axes(ax, S_Gpc * 1.2, title=title)

    ax.legend(loc='upper right', fontsize=10)

    plt.tight_layout()

    # Save
    filename = os.path.join(output_dir, f'3d_snapshot_t{(time_Gyr+START_TIME):.1f}Gyr.png')
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
    const = CosmologicalConstants()

    # Get node positions (same for all times)
    node_positions = get_node_positions(S_Gpc)

    # Create figure
    fig = plt.figure(figsize=(18, 12))

    for panel_idx, snap_idx in enumerate(indices):
        snapshot = sim_data['snapshots'][snap_idx]
        history = sim_data['expansion_history'][snap_idx]

        positions = snapshot['positions'] / const.Gpc_to_m
        current_size = history['size'] / const.Gpc_to_m
        current_max_size = history['max_particle_distance'] / const.Gpc_to_m
        time_Gyr = history['time_Gyr']

        # Create subplot
        ax = fig.add_subplot(2, 3, panel_idx + 1, projection='3d')

        # Plot particles
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                   c='blue', s=10, alpha=0.5)

        # Plot nodes
        ax.scatter(node_positions[:, 0], node_positions[:, 1], node_positions[:, 2],
                   c='red', s=100, marker='*', alpha=0.8, edgecolors='darkred')

        # Universe boundaries
        draw_universe_sphere(ax, current_max_size, resolution=30)
        draw_universe_sphere(ax, current_size/2, resolution=30)

        # Setup axes
        title = f't = {(time_Gyr+START_TIME):.1f} Gyr\nR = {current_size:.1f} Gpc'
        setup_3d_axes(ax, S_Gpc * 1.1, title=title)

        ax.set_xlabel('X [Gpc]', fontsize=9)
        ax.set_ylabel('Y [Gpc]', fontsize=9)
        ax.set_zlabel('Z [Gpc]', fontsize=9)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_box_aspect([1,1,1])

    fig.suptitle(f'Universe Expansion Over {sim_data["expansion_history"][-1]["time_Gyr"]:.1f} Billion Years',
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
    const = CosmologicalConstants()

    # Get node positions
    node_positions = get_node_positions(S_Gpc)

    # Create figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Setup axes (fixed for all frames)
    setup_3d_axes(ax, S_Gpc * 1.2)

    # Initialize empty plots
    particles_plot = ax.scatter([], [], [], c='blue', s=20, alpha=0.6)
    nodes_plot = ax.scatter(node_positions[:, 0], node_positions[:, 1], node_positions[:, 2],
                            c='red', s=200, marker='*', alpha=0.8,
                            edgecolors='darkred', linewidth=1.5)

    title = ax.text2D(0.5, 0.95, '', transform=ax.transAxes,
                      ha='center', fontsize=13, fontweight='bold')

    # Prepare sphere mesh
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 30)

    sphere_plot = None

    def init():
        particles_plot._offsets3d = ([], [], [])
        return particles_plot, nodes_plot, title

    def update(frame):
        nonlocal sphere_plot

        snapshot = sim_data['snapshots'][frame]
        history = sim_data['expansion_history'][frame]

        positions = snapshot['positions'] / const.Gpc_to_m
        current_size = history['size'] / const.Gpc_to_m
        current_max_size = history['max_particle_distance'] / const.Gpc_to_m
        time_Gyr = history['time_Gyr']

        # Update particles
        particles_plot._offsets3d = (positions[:, 0], positions[:, 1], positions[:, 2])

        # Update universe boundaries
        if sphere_plot is not None:
            sphere_plot.remove()

        sphere_plot = draw_universe_sphere(ax, current_max_size)
        draw_universe_sphere(ax, current_size/2)

        # Update title
        title.set_text(
            f'External-Node Cosmology: t = {(time_Gyr+START_TIME):.2f} Gyr\n'
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
    print(f"Time range: {START_TIME} to {(sim_data['expansion_history'][-1]['time_Gyr']+START_TIME):.1f} Gyr")

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
