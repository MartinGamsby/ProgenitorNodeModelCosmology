"""Regression tests for the chunked potential_energy() memory fix.

The PE diagnostic is chunked to avoid the full (N,N,3) pairwise array. The block
size used to be picked by ELEMENT count (`1e7 // N`), which ignored the xyz factor
on `diff` (b,N,3) AND the ~6 (b,N) temporaries -> ~719 MB transient per call at
N=24000, mass-crashing a parallel sweep with numpy _ArrayMemoryError.

These pin BOTH halves of the fix:
  * the VALUE is blocking-invariant (chunking must never change the physics), and
  * the PEAK ALLOCATION stays inside the byte budget.
See lode/numerics/memory-and-parallelism.md.
"""

import tracemalloc

import numpy as np
import pytest

from cosmo import integrator as integ
from cosmo.constants import CosmologicalConstants
from cosmo.integrator import LeapfrogIntegrator, _PE_BLOCK_BYTES
from cosmo.particles import ParticleSystem


def _make_integrator(n_particles: int) -> LeapfrogIntegrator:
    const = CosmologicalConstants()
    particles = ParticleSystem(
        n_particles=n_particles,
        box_size_m=20.0 * const.Gpc_to_m,
        total_mass_kg=10 * 1e53,
    )
    return LeapfrogIntegrator(
        particles, use_external_nodes=False, use_dark_energy=False,
        force_method='numba_direct',
    )


def _dense_pe(ig: LeapfrogIntegrator) -> float:
    """Dense upper-triangle PE reference, independent of any blocking."""
    const = CosmologicalConstants()
    pos = ig.particles.get_positions()
    m = ig.particles.get_masses()
    diff = pos[:, None, :] - pos[None, :, :]
    r = np.sqrt(np.einsum('ijk,ijk->ij', diff, diff) + ig.softening_m ** 2)
    pe = -const.G * (m[:, None] * m[None, :]) / r
    return float(np.sum(np.triu(pe, k=1)))


def test_pe_matches_dense_reference():
    """Chunked PE == dense upper-triangle sum (the correctness pin)."""
    ig = _make_integrator(300)
    assert ig.potential_energy() == pytest.approx(_dense_pe(ig), rel=1e-12)


def test_pe_value_is_invariant_to_block_size(monkeypatch):
    """Changing the memory budget must NOT change the PE value.

    This is the regression that matters: the fix altered block boundaries, so the
    value has to be blocking-invariant or we silently changed the physics.
    """
    ig = _make_integrator(300)
    reference = ig.potential_energy()

    for budget in (4 * 1024, 256 * 1024, 8 * 1024 * 1024, 512 * 1024 * 1024):
        monkeypatch.setattr(integ, "_PE_BLOCK_BYTES", budget)
        assert ig.potential_energy() == pytest.approx(reference, rel=1e-12), (
            f"PE changed with block budget {budget}"
        )


@pytest.mark.parametrize("n", [2_000, 8_000, 24_000, 100_000])
def test_pe_block_respects_byte_budget(n):
    """The chosen block keeps diff + the (b,N) temporaries inside the budget.

    Mirrors the sizing in potential_energy: b*N*8*(3+6) bytes. The OLD heuristic
    (1e7 // n) blows this for every n >= ~2000, which is exactly the bug.
    """
    block = max(1, int(_PE_BLOCK_BYTES // (max(1, n) * 8 * 9)))
    assert block >= 1
    assert block * n * 8 * 9 <= _PE_BLOCK_BYTES

    old_block = max(1, int(1e7 // n))
    assert old_block * n * 8 * 9 > _PE_BLOCK_BYTES, (
        "old element-count heuristic should exceed the budget (regression guard)"
    )


def test_pe_peak_allocation_is_bounded():
    """Actual peak allocation during one PE call stays near the budget.

    N=3000 would have used block=3333 under the old heuristic (~720 MB); the fix
    caps it. Allow generous headroom for numpy scratch so this is not flaky.
    """
    ig = _make_integrator(3_000)
    ig.potential_energy()  # warm up any lazy allocation

    tracemalloc.start()
    try:
        ig.potential_energy()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert peak < 4 * _PE_BLOCK_BYTES, (
        f"PE peak {peak/1e6:.0f} MB exceeds 4x the {_PE_BLOCK_BYTES/1e6:.0f} MB budget"
    )
