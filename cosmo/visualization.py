"""
Visualization Utilities

Shared functions for:
- Node position calculation
- 3D plotting helpers (spheres, cubes, axes setup)
- Filename generation
"""

import os
import numpy as np
from datetime import datetime


def get_node_positions(S_Gpc):
    """
    Get positions of 26 external nodes in 3×3×3 cubic lattice.

    Nodes placed at corners, edges, and faces of cube with spacing S.
    Center (0,0,0) is our observable universe - excluded.

    Parameters:
    -----------
    S_Gpc : float
        Node separation distance [Gpc]

    Returns:
    --------
    ndarray
        (26, 3) array of node positions [Gpc]
    """
    positions = []
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            for k in [-1, 0, 1]:
                if i == 0 and j == 0 and k == 0:
                    continue  # Skip center (our universe)
                positions.append([i * S_Gpc, j * S_Gpc, k * S_Gpc])
    return np.array(positions)


def draw_universe_sphere(ax, radius_Gpc, alpha=0.05, color='cyan', resolution=30):
    """
    Draw sphere representing universe boundary on 3D axes.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes3D
        3D axes to draw on
    radius_Gpc : float
        Sphere radius [Gpc]
    alpha : float
        Transparency (0=invisible, 1=opaque)
    color : str
        Sphere color
    resolution : int
        Number of points for sphere mesh

    Returns:
    --------
    matplotlib surface object
        The drawn sphere surface
    """
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)

    x_sphere = radius_Gpc * np.outer(np.cos(u), np.sin(v))
    y_sphere = radius_Gpc * np.outer(np.sin(u), np.sin(v))
    z_sphere = radius_Gpc * np.outer(np.ones(np.size(u)), np.cos(v))

    return ax.plot_surface(x_sphere, y_sphere, z_sphere,
                          alpha=alpha, color=color)


def draw_cube_edges(ax, half_size_Gpc, color='orange', alpha=0.3, linewidth=1):
    """
    Draw edges of cube representing HMEA node grid.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes3D
        3D axes to draw on
    half_size_Gpc : float
        Half-width of cube (distance from center to face) [Gpc]
    color : str
        Edge color
    alpha : float
        Edge transparency
    linewidth : float
        Edge line width
    """
    r = [-half_size_Gpc, half_size_Gpc]

    # Define 12 edges of cube
    edges = [
        # Bottom face
        [[r[0], r[0], r[0]], [r[1], r[0], r[0]]],
        [[r[0], r[1], r[0]], [r[1], r[1], r[0]]],
        [[r[0], r[0], r[0]], [r[0], r[1], r[0]]],
        [[r[1], r[0], r[0]], [r[1], r[1], r[0]]],
        # Top face
        [[r[0], r[0], r[1]], [r[1], r[0], r[1]]],
        [[r[0], r[1], r[1]], [r[1], r[1], r[1]]],
        [[r[0], r[0], r[1]], [r[0], r[1], r[1]]],
        [[r[1], r[0], r[1]], [r[1], r[1], r[1]]],
        # Vertical edges
        [[r[0], r[0], r[0]], [r[0], r[0], r[1]]],
        [[r[1], r[0], r[0]], [r[1], r[0], r[1]]],
        [[r[0], r[1], r[0]], [r[0], r[1], r[1]]],
        [[r[1], r[1], r[0]], [r[1], r[1], r[1]]],
    ]

    for start, end in edges:
        ax.plot3D(*zip(start, end), color=color, alpha=alpha, linewidth=linewidth)


def setup_3d_axes(ax, lim_Gpc, title="", elev=20, azim=45):
    """
    Configure 3D axes with standard settings.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes3D
        3D axes to configure
    lim_Gpc : float
        Axis limits (±lim_Gpc on all axes) [Gpc]
    title : str
        Plot title
    elev : float
        Viewing elevation angle [degrees]
    azim : float
        Viewing azimuth angle [degrees]
    """
    ax.set_xlim([-lim_Gpc, lim_Gpc])
    ax.set_ylim([-lim_Gpc, lim_Gpc])
    ax.set_zlim([-lim_Gpc, lim_Gpc])

    ax.set_xlabel('X [Gpc]', fontsize=11)
    ax.set_ylabel('Y [Gpc]', fontsize=11)
    ax.set_zlabel('Z [Gpc]', fontsize=11)

    if title:
        ax.set_title(title, fontsize=13, fontweight='bold')

    ax.view_init(elev=elev, azim=azim)


def generate_output_filename(
    base_name,
    sim_params,
    extension='png',
    output_dir='.',
    include_timestamp=True
):
    """
    Generate standardized output filename with parameters.

    Format: {base_name}_{timestamp}_{particles}p_{time_range}_{M}M_{S}S_{steps}steps_{damping}d.{ext}

    Parameters:
    -----------
    base_name : str
        Base filename (e.g., 'simulation', 'figure')
    sim_params : SimulationParameters
        Simulation configuration
    extension : str
        File extension without dot (e.g., 'png', 'pkl')
    output_dir : str
        Output directory path
    include_timestamp : bool
        Whether to include timestamp in filename

    Returns:
    --------
    str
        Full path to output file
    """
    parts = [base_name]

    if include_timestamp:
        parts.append(datetime.now().strftime("%Y-%m-%d_%H.%M.%S"))

    parts.append(f"{sim_params.n_particles}p")
    parts.append(f"{sim_params.t_start_Gyr}-{sim_params.t_end_Gyr}Gyr")
    parts.append(f"{sim_params.M_value}M")
    parts.append(f"{sim_params.S_value}S")
    parts.append(f"{sim_params.n_steps}steps")

    damping_str = f"{sim_params.damping_factor}" if sim_params.damping_factor else "Auto"
    parts.append(f"{damping_str}d")

    filename = "_".join(parts) + f".{extension}"
    return os.path.join(output_dir, filename)


def format_simulation_title(sim_params, include_particles=True):
    """
    Generate standardized title for plots.

    Parameters:
    -----------
    sim_params : SimulationParameters
        Simulation configuration
    include_particles : bool
        Whether to include particle count in title

    Returns:
    --------
    str
        Formatted title string
    """
    parts = [f'M={sim_params.M_value}, S={sim_params.S_value}']

    if include_particles:
        parts.append(f'{sim_params.n_particles}p')

    return f'Cosmology Comparison ({", ".join(parts)})'
