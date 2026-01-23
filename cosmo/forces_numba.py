"""
Numba-accelerated force calculations.

These JIT-compiled versions provide significant speedup over pure NumPy
by compiling to machine code.
"""

import numpy as np
from numba import jit


@jit(nopython=True, cache=True)
def calculate_internal_forces_numba(
    positions_m: np.ndarray,
    masses_kg: np.ndarray,
    softening_m: float,
    G: float
) -> np.ndarray:
    """
    JIT-compiled internal forces calculation.

    Same O(N²) algorithm as direct method, but compiled to machine code for speed.

    Args:
        positions_m: Particle positions, shape (N, 3)
        masses_kg: Particle masses, shape (N,)
        softening_m: Gravitational softening length
        G: Gravitational constant

    Returns:
        Accelerations array, shape (N, 3) in m/s²
    """
    N = len(positions_m)
    accelerations = np.zeros((N, 3))

    for i in range(N):
        for j in range(N):
            if i == j:
                continue

            # Vector from i to j
            r_vec = positions_m[j] - positions_m[i]
            r = np.sqrt(r_vec[0]**2 + r_vec[1]**2 + r_vec[2]**2)
            r_soft = np.sqrt(r**2 + softening_m**2)

            # Newton's law
            a_mag = G * masses_kg[j] / r_soft**2

            # Add to acceleration (pointing toward j)
            for k in range(3):
                accelerations[i, k] += a_mag * (r_vec[k] / r_soft)

    return accelerations
