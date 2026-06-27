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
    G: float,
    softening_m: float = 0.0,
) -> np.ndarray:
    """
    Calculate tidal acceleration from external nodes using Numba JIT.

    Singularity handling (node softening — Section 4 slingshot fix):
        * softening_m == 0.0 (DEFAULT): the LEGACY behaviour — a hard
          ``r < 1e10 m`` floor (~3e-13 Gpc, effectively NO softening at Gpc
          scales). This is byte-identical to the pre-softening force, so the
          default leaves cube26 a(t) and every existing cache entry unchanged.
        * softening_m > 0.0: a PHYSICAL Plummer softening
          ``r_soft^2 = r^2 + softening_m^2`` (same convention as the internal
          particle-particle force in cosmo.integrator.calculate_internal_forces),
          which caps the close-pass kick and tames the runaway slingshot. The
          1e10 m floor is NOT applied in this branch (the Plummer term already
          keeps the force finite at r -> 0).

    Args:
        particle_positions: (N, 3) particle positions in meters
        node_positions: (M, 3) external node positions in meters
        node_masses: (M,) external node masses in kg
        G: Gravitational constant
        softening_m: Plummer node-softening length in meters (>= 0). 0.0 keeps
            the legacy hard 1e10 m floor (byte-identical default).

    Returns:
        (N, 3) accelerations in m/s²
    """
    N = len(particle_positions)
    M = len(node_positions)
    accelerations = np.zeros((N, 3))

    eps2 = softening_m * softening_m

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

            if eps2 > 0.0:
                # Plummer softening: finite at r -> 0, no hard floor needed.
                r2 = r2 + eps2
                r = np.sqrt(r2)
            else:
                # LEGACY hard floor (byte-identical default).
                r = np.sqrt(r2)
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
