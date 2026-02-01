"""
Visualization Utilities

Shared functions for:
- Node position calculation
- 3D plotting helpers (spheres, cubes, axes setup)
- Filename generation
"""

from typing import Optional
import os
import numpy as np
from datetime import datetime


def get_node_positions(S_Gpc: float) -> np.ndarray:
    """
    Get positions of 26 external nodes in 3×3×3 cubic lattice.

    Nodes placed at corners, edges, and faces of cube with spacing S.
    Center (0,0,0) is our observable universe - excluded.

    Returns (26, 3) array of node positions in Gpc.
    """
    positions = []
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            for k in [-1, 0, 1]:
                if i == 0 and j == 0 and k == 0:
                    continue  # Skip center (our universe)
                positions.append([i * S_Gpc, j * S_Gpc, k * S_Gpc])
    return np.array(positions)


def draw_universe_sphere(ax, radius_Gpc: float, center_Gpc: Optional[np.ndarray] = None,
                          alpha: float = 0.05, color: str = 'cyan', resolution: int = 30):
    """Draw sphere representing universe boundary on 3D axes."""
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)

    x_sphere = radius_Gpc * np.outer(np.cos(u), np.sin(v))
    y_sphere = radius_Gpc * np.outer(np.sin(u), np.sin(v))
    z_sphere = radius_Gpc * np.outer(np.ones(np.size(u)), np.cos(v))

    if center_Gpc is not None:
        x_sphere += center_Gpc[0]
        y_sphere += center_Gpc[1]
        z_sphere += center_Gpc[2]

    return ax.plot_surface(x_sphere, y_sphere, z_sphere,
                          alpha=alpha, color=color)


def draw_cube_edges(ax, half_size_Gpc: float, color: str = 'orange', alpha: float = 0.3, linewidth: float = 1):
    """Draw edges of cube representing HMEA node grid."""
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


def setup_3d_axes(ax, lim_Gpc: float, title: str = "", elev: float = 20, azim: float = 45):
    """Configure 3D axes with standard settings."""
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
    include_timestamp=False
):
    """Generate standardized output filename with parameters."""
    parts = [base_name]

    if include_timestamp:
        parts.append(datetime.now().strftime("%Y-%m-%d_%H.%M.%S"))

    parts.append(f"{sim_params.n_particles}p")
    parts.append(f"{sim_params.t_start_Gyr}-{sim_params.t_end_Gyr}Gyr")
    parts.append(f"{sim_params.M_value}M")
    parts.append(f"{sim_params.center_node_mass}centerM")
    parts.append(f"{sim_params.S_value}S")
    parts.append(f"{sim_params.n_steps}steps")

    damping_str = f"{sim_params.damping_factor}" if sim_params.damping_factor else "Auto"
    parts.append(f"{damping_str}d")

    filename = "_".join(parts) + f".{extension}"
    return os.path.join(output_dir, filename)


def format_simulation_title(sim_params, include_particles: bool = True) -> str:
    """Generate standardized title for plots."""
    parts = [f'M={sim_params.M_value}, S={sim_params.S_value}']

    if include_particles:
        parts.append(f'{sim_params.n_particles}p')

    return f'Cosmology Comparison ({", ".join(parts)})'


def create_comparison_plot(
    sim_params,
    t_lcdm, a_lcdm, size_lcdm, H_lcdm_hubble,
    t_ext, a_ext, size_ext, H_ext_hubble,
    t_matter, a_matter_sim, size_matter_sim, H_matter_sim_hubble,
    today=None
):
    """
    Create 4-panel comparison plot: Scale Factor, Hubble Parameter, Relative Expansion, Physical Size.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(format_simulation_title(sim_params), fontsize=16, fontweight='bold')

    # Panel 1: Scale Factor
    ax1 = axes[0, 0]
    ax1.plot(t_lcdm, a_lcdm / a_lcdm[0], 'b-', label='LCDM (with dark energy)', linewidth=2)
    ax1.plot(t_ext, a_ext, 'r--', label='External-Node', linewidth=2)
    ax1.plot(t_matter, a_matter_sim, 'g:', label='Matter-only (no dark energy)', linewidth=2)
    if today:
        ax1.axvline(x=today, color='gray', linestyle=':', alpha=0.5, label='Today')
    ax1.set_xlabel('Time [Gyr]', fontsize=11)
    ax1.set_ylabel('Scale Factor', fontsize=11)
    ax1.set_title('Cosmic Expansion', fontsize=13)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Hubble Parameter
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

    # Panel 3: Relative Expansion
    ax3 = axes[1, 0]
    import numpy as np
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

    # Panel 4: Physical Size
    ax4 = axes[1, 1]
    ax4.plot(t_lcdm, size_lcdm, 'b-', label='LCDM', linewidth=2)
    ax4.plot(t_ext, size_ext, 'r--', label='External-Node', linewidth=2)
    ax4.plot(t_matter, size_matter_sim, 'g:', label='Matter-only', linewidth=2)
    ax4.axhline(sim_params.S_value, color='orange', linestyle='--',
                label=f'Nodes ({sim_params.S_value} Gpc)', linewidth=2)
    if today:
        ax4.axvline(x=today, color='gray', linestyle=':', alpha=0.5)
    ax4.set_xlabel('Time [Gyr]', fontsize=11)
    ax4.set_ylabel('Universe Radius [Gpc]', fontsize=11)
    ax4.set_title('Physical Size', fontsize=13)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    return fig

