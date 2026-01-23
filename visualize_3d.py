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

# Calculate initial conditions using shared function
sim_params = SimulationParameters(
    M_value=644,
    S_value=21,
    n_particles=140,
    seed=42,
    t_start_Gyr=START_TIME,
    t_duration_Gyr=10.0*5/4,
    n_steps=1000,
    damping_factor=0.92
)

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

    initial_conditions = calculate_initial_conditions(sim_params.t_start_Gyr)

    sim = CosmologicalSimulation(
        sim_params=sim_params,
        box_size_Gpc=initial_conditions['box_size_Gpc'],
        a_start=initial_conditions['a_start'],
        use_external_nodes=True,
        use_dark_energy=False
    )

    sim.run(t_end_Gyr=sim_params.t_duration_Gyr, n_steps=sim_params.n_steps, save_interval=50)

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
    com_Gpc = history['com'] / const.Gpc_to_m

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

    # Draw spheres representing universe boundaries centered on COM
    # Outer sphere: max particle distance (shows extent including outliers)
    draw_universe_sphere(ax, current_max_size, center_Gpc=com_Gpc, alpha=0.03, color='red')
    # Inner sphere: RMS-based radius (shows typical particle distribution)
    draw_universe_sphere(ax, current_size/2, center_Gpc=com_Gpc, alpha=0.08, color='cyan')

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

    print(f"OK Saved {filename}")

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
        com_Gpc = history['com'] / const.Gpc_to_m

        # Create subplot
        ax = fig.add_subplot(2, 3, panel_idx + 1, projection='3d')

        # Plot particles
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                   c='blue', s=10, alpha=0.5)

        # Plot nodes
        ax.scatter(node_positions[:, 0], node_positions[:, 1], node_positions[:, 2],
                   c='red', s=100, marker='*', alpha=0.8, edgecolors='darkred')

        # Universe boundaries centered on COM
        # Outer: max distance, Inner: RMS radius
        draw_universe_sphere(ax, current_max_size, center_Gpc=com_Gpc, alpha=0.03, color='red', resolution=30)
        draw_universe_sphere(ax, current_size/2, center_Gpc=com_Gpc, alpha=0.08, color='cyan', resolution=30)

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

    print(f"OK Saved {filename}")

    return filename


def create_animation(sim_data, output_dir=".", fps=5):
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
        com_Gpc = history['com'] / const.Gpc_to_m

        # Update particles
        particles_plot._offsets3d = (positions[:, 0], positions[:, 1], positions[:, 2])

        # Update universe boundaries centered on COM
        if sphere_plot is not None:
            sphere_plot.remove()

        # Draw both spheres (max and RMS)
        draw_universe_sphere(ax, current_max_size, center_Gpc=com_Gpc, alpha=0.03, color='red')
        sphere_plot = draw_universe_sphere(ax, current_size/2, center_Gpc=com_Gpc, alpha=0.08, color='cyan')

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

    print(f"OK Saved {filename}")
    print(f"  {n_frames} frames at {fps} fps = {n_frames/fps:.1f} seconds")

    return filename


def generate_sphere_positions(radius_m, n_points=100, seed=42):
    """
    Generate randomly distributed points on a sphere surface.

    Uses random spherical coordinates with uniform distribution.

    Parameters:
    -----------
    radius_m : float
        Sphere radius in meters
    n_points : int
        Number of points to generate on sphere surface
    seed : int
        Random seed for reproducibility

    Returns:
    --------
    ndarray
        Array of shape (n_points, 3) with 3D positions in meters
    """
    rng = np.random.RandomState(seed)

    # Generate uniformly distributed points on sphere using spherical coordinates
    # For uniform distribution on sphere surface:
    # - theta (azimuthal angle) uniform in [0, 2π)
    # - phi (polar angle) distributed as arccos(uniform(-1, 1))

    theta = rng.uniform(0, 2 * np.pi, n_points)  # Azimuthal angle
    u = rng.uniform(-1, 1, n_points)  # For uniform distribution on sphere
    phi = np.arccos(u)  # Polar angle

    # Convert spherical to Cartesian coordinates
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)

    # Scale to desired radius and stack into (n_points, 3) array
    positions = np.column_stack([x, y, z]) * radius_m

    return positions


def run_comparison_simulations(output_dir="."):
    """
    Run three simulations for comparison:
    1. External-Node Model (tidal forces, no dark energy)
    2. Matter-Only (gravity only, no dark energy)
    3. ΛCDM equivalent (to show expected size evolution)

    Returns dict with all three simulation results
    """
    if not HAS_SIM:
        print("ERROR: Cannot run simulations without source modules")
        sys.exit(1)

    print("Running three comparison simulations...")
    print("This will take a few moments...\n")

    initial_conditions = calculate_initial_conditions(sim_params.t_start_Gyr)

    # 1. External-Node Model
    print("1/3: Running External-Node simulation...")
    sim_ext = CosmologicalSimulation(
        sim_params=sim_params,
        box_size_Gpc=initial_conditions['box_size_Gpc'],
        a_start=initial_conditions['a_start'],
        use_external_nodes=True,
        use_dark_energy=False
    )
    sim_ext.run(t_end_Gyr=sim_params.t_duration_Gyr, n_steps=sim_params.n_steps, save_interval=50)

    # 2. Matter-Only
    print("2/3: Running Matter-Only simulation...")
    sim_matter = CosmologicalSimulation(
        sim_params=sim_params,
        box_size_Gpc=initial_conditions['box_size_Gpc'],
        a_start=initial_conditions['a_start'],
        use_external_nodes=False,
        use_dark_energy=False
    )
    sim_matter.run(t_end_Gyr=sim_params.t_duration_Gyr, n_steps=sim_params.n_steps, save_interval=50)

    # 3. ΛCDM reference (compute expected sizes from theory)
    print("3/3: Computing ΛCDM reference evolution...")
    from cosmo.analysis import solve_friedmann_equation

    # Solve Friedmann equation to get ΛCDM evolution
    t_start_Gyr = sim_params.t_start_Gyr
    t_end_Gyr = t_start_Gyr + sim_params.t_duration_Gyr
    lcdm_solution = solve_friedmann_equation(t_start_Gyr, t_end_Gyr, n_points=1000)

    # Get initial scale factor to compute relative expansion
    a_start = initial_conditions['a_start']

    # Create ΛCDM size evolution and snapshots to match External-Node snapshots
    lcdm_history = []
    lcdm_snapshots = []
    const = CosmologicalConstants()

    for snap in sim_ext.snapshots:
        t_seconds = snap['time_s']
        t_Gyr_offset = t_seconds / (1e9 * 365.25 * 24 * 3600)  # Simulation time (starts at 0)
        t_Gyr_absolute = t_start_Gyr + t_Gyr_offset  # Absolute cosmic time

        # Interpolate scale factor at this time using full arrays
        # (windowed arrays may not cover full simulation time range)
        a_lcdm = np.interp(t_Gyr_absolute, lcdm_solution['_t_Gyr_full'], lcdm_solution['_a_full'])

        # Physical size in ΛCDM: relative expansion from start
        # size = (a(t) / a_start) * initial_box_size
        # This matches how simulations compute: a_relative = rms_current / rms_initial
        a_relative = a_lcdm / a_start
        size_lcdm_Gpc = a_relative * initial_conditions['box_size_Gpc']

        size_m = size_lcdm_Gpc * const.Gpc_to_m  # RMS radius in meters
        # Scale box_size so that the RMS radius matches the target
        # For a uniform sphere of radius R, RMS radius = R * sqrt(3/5) ≈ 0.775*R
        # We want RMS = size_lcdm_Gpc, so R_sphere = size_lcdm_Gpc / 0.775
        # This means we need to use a sphere of radius: size_lcdm_Gpc/2 / sqrt(3/5)
        radius_max = size_m / 2 / np.sqrt(3/5)

        # Generate sphere positions for this snapshot

        sphere_positions = generate_sphere_positions(radius_max, n_points=100)
        #for i in range(50, 90, 10):
        #    pc = int(i/100)
        #    sphere_positions = np.concatenate((sphere_positions, generate_sphere_positions(radius_max*pc, n_points=30*pc, seed=i)))

        lcdm_history.append({
            'time': t_seconds,
            'time_Gyr': t_Gyr_offset,#t_Gyr_absolute,
            'scale_factor': a_relative,  # Store relative scale factor to match simulations
            'size': size_m,  # Size stored as radius to match simulation convention
            'com': np.zeros(3),  # ΛCDM doesn't drift
            'max_particle_distance': radius_max,
        })

        # Create snapshot with sphere positions
        lcdm_snapshots.append({
            'time': t_seconds,
            'positions': sphere_positions,
            'velocities': np.zeros_like(sphere_positions),  # ΛCDM doesn't have particle velocities
            'accelerations': np.zeros_like(sphere_positions),  # ΛCDM doesn't have accelerations
        })

    print("OK All three simulations complete!\n")

    # Package results
    comparison_data = {
        'external_node': {
            'name': 'External-Node',
            'snapshots': sim_ext.snapshots,
            'expansion_history': sim_ext.expansion_history,
            'color': 'red',
            'linestyle': '--'
        },
        'matter_only': {
            'name': 'Matter-Only',
            'snapshots': sim_matter.snapshots,
            'expansion_history': sim_matter.expansion_history,
            'color': 'green',
            'linestyle': ':'
        },
        'lcdm': {
            'name': 'ΛCDM',
            'snapshots': lcdm_snapshots,  # Generated sphere positions
            'expansion_history': lcdm_history,
            'color': 'blue',
            'linestyle': '-'
        },
        'params': {
            'M_ext': sim_params.M_value * const.M_observable,
            'S': sim_params.S_value,
            'initial_size': initial_conditions['box_size_Gpc']
        }
    }

    return comparison_data


def create_comparison_multipanel(comparison_data, output_dir="."):
    """Create 3-way comparison showing External-Node, Matter-Only, and ΛCDM side-by-side"""

    # Select 6 evenly spaced snapshots
    n_snapshots = len(comparison_data['external_node']['snapshots'])
    indices = np.linspace(0, n_snapshots-1, 6, dtype=int)

    const = CosmologicalConstants()
    S_Gpc = comparison_data['params']['S']
    node_positions = get_node_positions(S_Gpc)

    # Create figure with 6 rows (times) × 3 columns (models)
    fig = plt.figure(figsize=(18, 24))

    model_names = ['external_node', 'matter_only', 'lcdm']

    for row_idx, snap_idx in enumerate(indices):
        for col_idx, model_name in enumerate(model_names):
            model_data = comparison_data[model_name]
            snapshot = model_data['snapshots'][snap_idx]
            history = model_data['expansion_history'][snap_idx]

            positions = snapshot['positions'] / const.Gpc_to_m
            current_size = history['size'] / const.Gpc_to_m
            current_max_size = history['max_particle_distance'] / const.Gpc_to_m
            time_Gyr = history['time_Gyr']
            com_Gpc = history['com'] / const.Gpc_to_m

            # Create subplot
            ax = fig.add_subplot(6, 3, row_idx * 3 + col_idx + 1, projection='3d')

            # Plot particles
            ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                      c=model_data['color'], s=10, alpha=0.5)

            # Plot nodes (only for external-node model)
            if model_name == 'external_node':
                ax.scatter(node_positions[:, 0], node_positions[:, 1], node_positions[:, 2],
                          c='orange', s=100, marker='*', alpha=0.8, edgecolors='darkorange')

            # Universe boundaries centered on COM
            # External-Node and Matter-Only: show both max and RMS
            draw_universe_sphere(ax, current_max_size, center_Gpc=com_Gpc,
                                alpha=0.05, color='red', resolution=20)
            draw_universe_sphere(ax, current_size/2, center_Gpc=com_Gpc,
                                alpha=0.1, color=model_data['color'], resolution=20)

            # Setup axes
            title = f'{model_data["name"]}\nt = {(time_Gyr+START_TIME):.1f} Gyr, R = {current_size:.1f} Gpc'
            setup_3d_axes(ax, S_Gpc * 1.1, title=title)

            ax.set_xlabel('X [Gpc]', fontsize=8)
            ax.set_ylabel('Y [Gpc]', fontsize=8)
            ax.set_zlabel('Z [Gpc]', fontsize=8)
            ax.set_title(title, fontsize=10, fontweight='bold')
            ax.set_box_aspect([1,1,1])

    fig.suptitle(f'3-Way Comparison: External-Node vs Matter-Only vs ΛCDM\n'
                f'M={comparison_data["params"]["M_ext"]/const.M_observable:.0f}×M_obs, '
                f'S={S_Gpc:.0f} Gpc',
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    filename = os.path.join(output_dir, '3d_comparison_multipanel.png')
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"OK Saved {filename}")
    return filename


def create_comparison_animations(comparison_data, output_dir=".", fps=10):
    """Create three separate animations for External-Node, Matter-Only, and ΛCDM"""

    const = CosmologicalConstants()
    S_Gpc = comparison_data['params']['S']
    node_positions = get_node_positions(S_Gpc)

    model_names = ['external_node', 'matter_only', 'lcdm']
    filenames = []

    for model_name in model_names:
        model_data = comparison_data[model_name]
        print(f"Creating {model_data['name']} animation...")

        # Create figure
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        setup_3d_axes(ax, S_Gpc * 1.2)

        # Initialize plots
        particles_plot = ax.scatter([], [], [], c=model_data['color'], s=20, alpha=0.6)

        # Plot nodes for external-node model
        if model_name == 'external_node':
            ax.scatter(node_positions[:, 0], node_positions[:, 1], node_positions[:, 2],
                      c='orange', s=200, marker='*', alpha=0.8,
                      edgecolors='darkorange', linewidth=1.5,
                      label='External Nodes')

        title = ax.text2D(0.5, 0.95, '', transform=ax.transAxes,
                         ha='center', fontsize=13, fontweight='bold')

        sphere_plot = None

        def init():
            particles_plot._offsets3d = ([], [], [])
            return particles_plot, title

        def update(frame):
            nonlocal sphere_plot

            snapshot = model_data['snapshots'][frame]
            history = model_data['expansion_history'][frame]

            positions = snapshot['positions'] / const.Gpc_to_m
            current_size = history['size'] / const.Gpc_to_m
            current_max_size = history['max_particle_distance'] / const.Gpc_to_m
            time_Gyr = history['time_Gyr']
            com_Gpc = history['com'] / const.Gpc_to_m

            # Update particles
            particles_plot._offsets3d = (positions[:, 0], positions[:, 1], positions[:, 2])

            # Update universe boundaries
            if sphere_plot is not None:
                sphere_plot.remove()

            # External-Node and Matter-Only: both max and RMS
            draw_universe_sphere(ax, current_max_size, center_Gpc=com_Gpc,
                                alpha=0.05, color='red')
            sphere_plot = draw_universe_sphere(ax, current_size/2, center_Gpc=com_Gpc,
                                                alpha=0.1, color=model_data['color'])

            # Update title
            title_text = f'{model_data["name"]} Model: t = {(time_Gyr+START_TIME):.2f} Gyr\n'
            title_text += f'Universe Radius: {current_size:.1f} Gpc'
            if model_name == 'external_node':
                title_text += f' | Nodes at {S_Gpc:.0f} Gpc'
            title.set_text(title_text)

            # Rotate view
            ax.view_init(elev=20, azim=45 + frame * 2)

            return particles_plot, title

        # Create animation
        n_frames = len(model_data['snapshots'])
        anim = FuncAnimation(fig, update, init_func=init, frames=n_frames,
                           interval=1000//fps, blit=False)

        # Save
        filename = os.path.join(output_dir, f'3d_animation_{model_name}.gif')
        writer = PillowWriter(fps=fps)
        anim.save(filename, writer=writer)
        plt.close()

        print(f"OK Saved {filename}")
        filenames.append(filename)

    return filenames


def main():
    """Main visualization function"""

    # Parse arguments
    # Usage: python visualize_3d.py [sim_file] [output_dir]
    # Or:    python visualize_3d.py --compare [output_dir]
    if len(sys.argv) > 1 and sys.argv[1] == '--compare':
        # Comparison mode
        compare_mode = True
        sim_file = None
        output_dir = sys.argv[2] if len(sys.argv) > 2 else "."
    else:
        # Single simulation mode
        compare_mode = False
        sim_file = sys.argv[1] if len(sys.argv) > 1 else None
        output_dir = sys.argv[2] if len(sys.argv) > 2 else "."

    os.makedirs(output_dir, exist_ok=True)

    print("="*70)
    if compare_mode:
        print("3D COMPARISON: EXTERNAL-NODE vs MATTER-ONLY vs ΛCDM")
    else:
        print("3D VISUALIZATION OF EXTERNAL-NODE COSMOLOGY")
    print("="*70)
    print(f"Output directory: {os.path.abspath(output_dir)}\n")

    if compare_mode:
        # Run all three simulations and create comparisons
        comparison_data = run_comparison_simulations(output_dir)

        print("\n" + "="*70)
        print("CREATING COMPARISON VISUALIZATIONS")
        print("="*70)

        print("\n1. Three-way comparison multipanel...")
        create_comparison_multipanel(comparison_data, output_dir)

        print("\n2. Creating separate animations for each model...")
        create_comparison_animations(comparison_data, output_dir, fps=10)

        print("\n" + "="*70)
        print("*** COMPARISON COMPLETE!")
        print("="*70)
        print(f"\nAll files saved to: {os.path.abspath(output_dir)}")
        print("\nGenerated files:")
        print("  - 3d_comparison_multipanel.png     (3-way side-by-side comparison)")
        print("  - 3d_animation_external_node.gif   (External-Node animation)")
        print("  - 3d_animation_matter_only.gif     (Matter-Only animation)")
        print("  - 3d_animation_lcdm.gif             (ΛCDM animation)")

    else:
        # Single simulation mode
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
        print("*** VISUALIZATION COMPLETE!")
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
        print("  - Run with --compare flag for 3-way model comparison")


if __name__ == "__main__":
    main()
