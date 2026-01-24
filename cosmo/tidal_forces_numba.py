"""
Numba-accelerated tidal force calculations for external HMEA nodes.

Provides significant speedup for large N simulations with external nodes.
"""

import numpy as np
from numba import jit


@jit(nopython=True, cache=True)
def calculate_tidal_forces_numba(
    particle_positions: np.ndarray,
    node_positions: np.ndarray,
    node_masses: np.ndarray,
    G: float
) -> np.ndarray:
    """
    Calculate tidal acceleration from external nodes using Numba JIT.

    Args:
        particle_positions: (N, 3) particle positions in meters
        node_positions: (M, 3) external node positions in meters
        node_masses: (M,) external node masses in kg
        G: Gravitational constant

    Returns:
        (N, 3) accelerations in m/s²
    """
    N = len(particle_positions)
    M = len(node_positions)
    accelerations = np.zeros((N, 3))

    # Loop over particles
    for i in range(N):
        # Loop over external nodes
        for j in range(M):
            # Vector from particle to node
            dx = node_positions[j, 0] - particle_positions[i, 0]
            dy = node_positions[j, 1] - particle_positions[i, 1]
            dz = node_positions[j, 2] - particle_positions[i, 2]

            # Distance
            r2 = dx*dx + dy*dy + dz*dz
            r = np.sqrt(r2)

            # Avoid singularities
            if r < 1e10:
                r = 1e10
                r2 = r * r

            # Tidal acceleration (attractive toward node)
            r3 = r2 * r
            a_mag = G * node_masses[j] / r3

            accelerations[i, 0] += a_mag * dx
            accelerations[i, 1] += a_mag * dy
            accelerations[i, 2] += a_mag * dz

    return accelerations
