"""
Numba-accelerated tidal force calculations for external HMEA nodes.

Provides significant speedup for large N simulations with external nodes.
"""

import numpy as np
from numba import jit


# Close-range force-law codes (kept in sync with cosmo.constants.NODE_FORCE_LAW_CODES
# and the numpy fallback in cosmo.particles.HMEAGrid.calculate_tidal_acceleration_batch).
#   0 == "plummer"  : the legacy/Plummer law (DEFAULT, byte-identical when eps==0).
#   1 == "bounded"  : the "can't cross the midpoint" regularized law — below the
#                     softening length the per-node ACCELERATION MAGNITUDE is capped
#                     at its value at r == softening_m, so the force never keeps
#                     growing toward the node (distinct from the blunt Plummer floor,
#                     which still lets sub-softening pairs barely interact).
NODE_FORCE_LAW_PLUMMER = 0
NODE_FORCE_LAW_BOUNDED = 1


@jit(nopython=True, cache=True)
def calculate_tidal_forces_numba(
    particle_positions: np.ndarray,
    node_positions: np.ndarray,
    node_masses: np.ndarray,
    G: float,
    softening_m: float = 0.0,
    force_law: int = 0,
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

    Close-range force law (``force_law``):
        * 0 == "plummer" (DEFAULT): the behaviour described above. With
          softening_m == 0 it is byte-identical to the legacy force.
        * 1 == "bounded": a more PHYSICAL close-encounter law that addresses the
          "1 Gpc Plummer floor is too coarse" complaint. The Newtonian
          ``a = G m / r^2`` is used down to the softening length, but for
          ``r < softening_m`` the per-node acceleration MAGNITUDE is CAPPED at
          its value at ``r == softening_m`` (``a_cap = G m / softening_m^2``)
          rather than continuing to grow OR collapsing to the Plummer floor.
          This is the "they still attract but can't be flung across the pair"
          constraint: the close-range kick is bounded, so a single coarse step
          cannot produce a runaway near-node slingshot, yet sub-softening bodies
          still feel the FULL ``softening_m`` attraction (not a softened-down
          fraction). ``softening_m`` must be > 0 for this law to differ from the
          unsoftened force; with softening_m == 0 it falls back to the legacy
          hard floor (byte-identical default).

    Args:
        particle_positions: (N, 3) particle positions in meters
        node_positions: (M, 3) external node positions in meters
        node_masses: (M,) external node masses in kg
        G: Gravitational constant
        softening_m: Plummer node-softening length in meters (>= 0). 0.0 keeps
            the legacy hard 1e10 m floor (byte-identical default).
        force_law: 0 == Plummer/legacy (default), 1 == bounded close-range law.

    Returns:
        (N, 3) accelerations in m/s²
    """
    N = len(particle_positions)
    M = len(node_positions)
    accelerations = np.zeros((N, 3))

    eps2 = softening_m * softening_m
    # Capped acceleration magnitude at r == softening_m for the bounded law.
    # Only meaningful when softening_m > 0; guarded by the (force_law == 1 and
    # eps2 > 0) branch below.
    bounded = (force_law == NODE_FORCE_LAW_BOUNDED) and (eps2 > 0.0)

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

            if bounded:
                # BOUNDED "can't cross the midpoint" law: true 1/r^2 outside the
                # softening length, capped magnitude inside it.
                r = np.sqrt(r2)
                if r < softening_m:
                    # Cap the per-unit-vector magnitude at a = G m / eps^2.
                    # a_vec = a_cap * (dr / r); the (dx,dy,dz)/r is the unit
                    # vector toward the node. Guard r == 0 (use a centred zero).
                    if r > 0.0:
                        a_cap = G * node_masses[j] / eps2
                        inv_r = 1.0 / r
                        accelerations[i, 0] += a_cap * dx * inv_r
                        accelerations[i, 1] += a_cap * dy * inv_r
                        accelerations[i, 2] += a_cap * dz * inv_r
                    # r == 0: symmetric, zero net contribution from this node.
                    continue
                # Far field: exact Newtonian 1/r^2 (no Plummer offset).
                r3 = r2 * r
                a_mag = G * node_masses[j] / r3
                accelerations[i, 0] += a_mag * dx
                accelerations[i, 1] += a_mag * dy
                accelerations[i, 2] += a_mag * dz
                continue

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
