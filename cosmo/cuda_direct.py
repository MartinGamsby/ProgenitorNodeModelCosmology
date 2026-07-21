"""
Numba-CUDA direct O(N^2) gravitational force calculation (WS7 GPU scale-out).

The GPU counterpart of ``cosmo.numba_direct``: one thread per particle sums the
exact pairwise 1/r^2 (Plummer-softened) acceleration over every other particle.
Direct-sum on the GPU beats CPU Barnes-Hut up to ~100k+ particles AND is
bit-checkable against ``NumbaDirectSolver`` (same math, only the hardware and the
summation order differ, ~1e-13 relative) -- the PF22-style falsification gate.

Selected via ``force_method="cuda_direct"`` (an EXPLICIT opt-in; the "auto" path
is unchanged, so default runs and every existing cache key are byte-identical).
The tidal / external-node force stays on the CPU numba path (O(N*M), M~300 nodes
is negligible), so a ``cuda_direct`` run == GPU internal gravity + the identical
CPU tidal path.

Correctness is validated WITHOUT a physical GPU via the CUDA simulator
(``NUMBA_ENABLE_CUDASIM=1``); real-GPU performance validation happens on the
first run on CUDA hardware.
"""

import math
import os

import numpy as np

try:  # numba.cuda import is cheap and always present with numba; guard anyway.
    from numba import cuda
    _CUDA_IMPORT_OK = True
except Exception:  # pragma: no cover - numba is a hard dep, but stay defensive.
    cuda = None
    _CUDA_IMPORT_OK = False


def cuda_backend_available() -> bool:
    """True when a CUDA kernel can actually be launched here.

    Real GPU (``cuda.is_available()``) OR the CUDA SIMULATOR
    (``NUMBA_ENABLE_CUDASIM=1``), which runs kernels on the CPU for correctness
    testing. Used by the solver to fail LOUDLY (not silently fall back to the
    CPU) when ``force_method="cuda_direct"`` is requested on a machine with no
    CUDA backend -- an explicit method means the user wants the GPU path.
    """
    if os.environ.get("NUMBA_ENABLE_CUDASIM", "0") == "1":
        return True
    if not _CUDA_IMPORT_OK:
        return False
    try:
        return bool(cuda.is_available())
    except Exception:  # pragma: no cover - driver probe can throw on odd setups.
        return False


if _CUDA_IMPORT_OK:

    @cuda.jit(cache=True)
    def _cuda_direct_kernel(positions, masses, softening, G, accel):
        """One thread per particle i; exact Plummer-softened O(N^2) sum.

        Mirrors ``cosmo.numba_direct.calculate_forces_direct_numba`` term for
        term: r_soft = sqrt(r^2 + eps^2), a_i += G*m_j*(r_j - r_i)/r_soft^3.
        """
        i = cuda.grid(1)
        n = positions.shape[0]
        if i >= n:
            return

        xi = positions[i, 0]
        yi = positions[i, 1]
        zi = positions[i, 2]
        soft2 = softening * softening

        ax = 0.0
        ay = 0.0
        az = 0.0
        for j in range(n):
            if j == i:
                continue
            dx = positions[j, 0] - xi
            dy = positions[j, 1] - yi
            dz = positions[j, 2] - zi
            r2 = dx * dx + dy * dy + dz * dz + soft2
            inv_r = 1.0 / math.sqrt(r2)
            inv_r3 = inv_r * inv_r * inv_r
            f = G * masses[j] * inv_r3
            ax += f * dx
            ay += f * dy
            az += f * dz

        accel[i, 0] = ax
        accel[i, 1] = ay
        accel[i, 2] = az


class CudaDirectSolver:
    """Direct O(N^2) gravity on the GPU. API-compatible with NumbaDirectSolver.

    ``build_tree`` stores positions/masses (name kept for solver-interface
    parity); ``calculate_all_accelerations`` launches the kernel and returns the
    host (N, 3) acceleration array in m/s^2.
    """

    def __init__(self, softening_m: float = 1e24, G: float = 6.674e-11,
                 threads_per_block: int = 128):
        self.softening_m = float(softening_m)
        self.G = float(G)
        self.threads_per_block = int(threads_per_block)
        self.positions_m = None
        self.masses_kg = None

    def build_tree(self, positions_m: np.ndarray, masses_kg: np.ndarray) -> None:
        """Store positions and masses (API compatible with the tree solvers)."""
        self.positions_m = np.ascontiguousarray(positions_m, dtype=np.float64)
        self.masses_kg = np.ascontiguousarray(masses_kg, dtype=np.float64)

    def calculate_all_accelerations(self) -> np.ndarray:
        """Launch the CUDA direct-sum kernel; return (N, 3) accel in m/s^2."""
        if not cuda_backend_available():
            raise RuntimeError(
                "force_method='cuda_direct' requires a CUDA GPU (numba.cuda "
                "reports no device). Run on CUDA hardware, or set "
                "NUMBA_ENABLE_CUDASIM=1 for a (slow) CPU correctness check."
            )
        positions = self.positions_m
        masses = self.masses_kg
        n = len(positions)
        accel = np.zeros((n, 3), dtype=np.float64)
        if n == 0:
            return accel

        d_pos = cuda.to_device(positions)
        d_mass = cuda.to_device(masses)
        d_acc = cuda.to_device(accel)

        tpb = self.threads_per_block
        bpg = (n + tpb - 1) // tpb
        _cuda_direct_kernel[bpg, tpb](d_pos, d_mass, self.softening_m, self.G, d_acc)

        return d_acc.copy_to_host()
