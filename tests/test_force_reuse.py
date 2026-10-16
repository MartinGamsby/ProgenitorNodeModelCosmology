"""KDK force reuse: one force evaluation per leapfrog step, bit-identical physics.

A kick-drift-kick step needs the acceleration at the START of the step, which is
exactly the acceleration computed at the END of the previous step (positions are
unchanged in between). The integrator reuses it instead of recomputing.

These pin both halves:
  * the trajectory is BIT-IDENTICAL to the textbook two-evaluation KDK (the
    oracle below), for every force method, with and without sub-stepping, and
    when something outside the integrator resets positions or toggles forces;
  * the evaluation count per step drops from 2 to 1.
See lode/physics/integration.md.
"""

import numpy as np
import pytest

from cosmo.constants import CosmologicalConstants, ExternalNodeParameters
from cosmo.integrator import LeapfrogIntegrator
from cosmo.particles import HMEAGrid, ParticleSystem

CONST = CosmologicalConstants()
DT_S = 2e15  # ~63 Myr


class _TwoEvalLeapfrog(LeapfrogIntegrator):
    """Oracle: the textbook KDK that recomputes forces at the start of every step."""

    def _kdk(self, dt_s):
        a = self.calculate_total_forces()
        self.particles.set_accelerations(a)
        self.particles.update_velocities(dt_s / 2)
        self.particles.update_positions(dt_s)
        a = self.calculate_total_forces()
        self.particles.set_accelerations(a)
        self.particles.update_velocities(dt_s / 2)


def _make(cls, force_method="numba_direct", n=64, **kwargs):
    """Identical ICs for every call (global np.random seeded), with HMEA nodes on."""
    np.random.seed(7)
    particles = ParticleSystem(
        n_particles=n,
        box_size_m=10.0 * CONST.Gpc_to_m,
        total_mass_kg=1e53,
        use_dark_energy=False,
    )
    grid = HMEAGrid(ExternalNodeParameters(M_ext_kg=500 * 1e53, S=20.0 * CONST.Gpc_to_m))
    return cls(particles, grid, softening_per_Mobs_m=CONST.Gpc_to_m,
               use_external_nodes=True, use_dark_energy=False,
               force_method=force_method, **kwargs)


def _state(ig):
    return ig.particles.get_positions(), ig.particles.get_velocities()


def _assert_same_state(a, b):
    (pa, va), (pb, vb) = _state(a), _state(b)
    assert np.array_equal(pa, pb), "positions diverged from the two-evaluation KDK"
    assert np.array_equal(va, vb), "velocities diverged from the two-evaluation KDK"


@pytest.mark.parametrize("force_method", ["direct", "numba_direct", "barnes_hut"])
def test_trajectory_bit_identical_to_two_evaluation_kdk(force_method):
    new, old = _make(LeapfrogIntegrator, force_method), _make(_TwoEvalLeapfrog, force_method)
    for _ in range(12):
        new.step(DT_S)
        old.step(DT_S)
    _assert_same_state(new, old)


def test_substepping_bit_identical_to_two_evaluation_kdk():
    # A huge threshold makes every step a "close pass", so every step sub-steps.
    kw = dict(node_substep_threshold=1e6, node_substeps=4)
    new, old = _make(LeapfrogIntegrator, **kw), _make(_TwoEvalLeapfrog, **kw)
    assert new._substep_active() and new._in_close_pass()
    for _ in range(5):
        new.step(DT_S)
        old.step(DT_S)
    _assert_same_state(new, old)


def test_one_force_evaluation_per_step():
    ig = _make(LeapfrogIntegrator)
    calls = []
    real = ig.calculate_total_forces
    ig.calculate_total_forces = lambda: calls.append(1) or real()
    n_steps = 10
    for _ in range(n_steps):
        ig.step(DT_S)
    # One evaluation to start the first step, then one per step.
    assert len(calls) == n_steps + 1


def test_external_position_reset_invalidates_reuse():
    """Calibration-style reset (set_positions between steps) must not reuse stale forces."""
    new, old = _make(LeapfrogIntegrator), _make(_TwoEvalLeapfrog)
    x0 = new.particles.get_positions()
    for ig in (new, old):
        ig.step(DT_S)
        ig.particles.set_positions(x0.copy())
        ig.step(DT_S)
    _assert_same_state(new, old)


def test_toggling_external_nodes_invalidates_reuse():
    """The legacy calibration turns the nodes off, steps, then back on."""
    new, old = _make(LeapfrogIntegrator), _make(_TwoEvalLeapfrog)
    for ig in (new, old):
        ig.step(DT_S)
        ig.use_external_nodes = False
        ig.step(DT_S)
        ig.use_external_nodes = True
        ig.step(DT_S)
    _assert_same_state(new, old)
