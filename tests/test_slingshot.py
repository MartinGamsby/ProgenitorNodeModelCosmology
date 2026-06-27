"""Slingshot root-cause unit tests (WS8, Section 3).

These tests prove, with TINY deterministic sims:

  1. test_runaway_config_has_slingshot
        A runaway config (cube26, M=1000, S=10) DOES produce a slingshot:
        the inner-particle displacement tail metric max/median exceeds a
        comfortable threshold. ("Validate there's one.")

  2. test_tame_config_has_no_big_slingshot
        A tame config (M=5, S=20) has a modest tail (max/median ~ 1), so the
        slingshot test above is a real contrast, not trivially always-true.

  3. test_slingshot_from_nodes_not_particles
        Re-running the runaway config with external HMEA nodes OFF (matter-only,
        same seed/N/steps) COLLAPSES the tail to tame levels — pinning the root
        cause to the node close-pass, not particle-particle encounters.

  4. test_node_softening_reduces_slingshot
        A DIAGNOSTIC Plummer node-softening (~1 Gpc) on the SAME runaway config
        tames the tail by orders of magnitude — motivating the Section 4 fix.

Determinism / speed
-------------------
- Fixed seed; small N (200) so the sims run in a few seconds each.
- resolve_n_steps keeps dt < 0.05 Gyr (the validator ceiling).
- MEASUREMENT ONLY: no product-physics is changed. Test 4 monkeypatches a single
  grid instance with a diagnostic softened tidal force (softened_node_acceleration);
  it never touches a product sim path.

Measured baselines (seed=42, N=200, t_start=5.8, n_steps->161) that motivate the
thresholds (kept loose so the tests are not flaky):
    runaway nodes ON  : max/median ~ 514,  tail_frac ~ 0.285
    tame    nodes ON  : max/median ~ 1.3,  tail_frac ~ 0.0
    runaway nodes OFF : max/median ~ 1.3,  tail_frac ~ 0.0
    runaway soft 1 Gpc: max/median ~ 2.3,  tail_frac ~ 0.0
"""

import numpy as np
import pytest

from cosmo.constants import CosmologicalConstants, SimulationParameters
from cosmo.factories import (
    run_external_node_simulation,
    run_matter_only_simulation,
    setup_simulation_context,
)
from _generate_ws8_figs import (
    displacement_magnitudes,
    slingshot_metrics,
    softened_node_acceleration,
    resolve_n_steps,
)
from cosmo.simulation import CosmologicalSimulation

_TODAY_GYR = 13.8
_T_START = 5.8
_N = 200
_N_STEPS = 80          # bumped by resolve_n_steps to keep dt < 0.05 Gyr
_SEED = 42
_G = CosmologicalConstants.Gpc_to_m

# Comfortably between the measured tame (~1.3) and runaway (~514) values.
_SLINGSHOT_THRESHOLD = 20.0
_TAME_BOUND = 5.0


def _make_params(M, S_gpc, n_particles=_N, n_steps=_N_STEPS, geometry="cube26"):
    t_dur = _TODAY_GYR - _T_START
    n_steps = resolve_n_steps(t_dur, n_steps)
    box, a_start, _ = setup_simulation_context(
        _T_START, t_dur, n_steps, save_interval=max(1, n_steps // 4))
    sp = SimulationParameters(
        M_value=M, S_value=S_gpc, n_particles=n_particles, seed=_SEED,
        t_start_Gyr=_T_START, t_duration_Gyr=t_dur, n_steps=n_steps,
        damping_factor=None, center_node_mass=1.0, mass_randomize=0.0,
        node_mass_seed=_SEED, init_distribution="uniform_sphere",
        node_geometry=geometry,
    )
    return sp, box, a_start, n_steps


def _disp_from_sim(sim) -> np.ndarray:
    mask = np.asarray(sim.particles.get_observable_mask(), dtype=bool)
    p0 = sim.snapshots[0]["positions"][mask] / _G
    p1 = sim.snapshots[-1]["positions"][mask] / _G
    return displacement_magnitudes(p0, p1)


def _run_factory(M, S_gpc, nodes=True):
    sp, box, a_start, n_steps = _make_params(M, S_gpc)
    fn = run_external_node_simulation if nodes else run_matter_only_simulation
    ext = fn(sp, box, a_start, save_interval=max(1, n_steps // 4))
    return _disp_from_sim(ext["sim"])


# ---------------------------------------------------------------------------
# 1. Runaway config DOES slingshot.
# ---------------------------------------------------------------------------

def test_runaway_config_has_slingshot():
    disp = _run_factory(M=1000, S_gpc=10, nodes=True)
    m = slingshot_metrics(disp)
    assert m["n"] > 0
    assert np.isfinite(m["max_over_median"])
    assert m["max_over_median"] > _SLINGSHOT_THRESHOLD, (
        f"expected a slingshot (max/median > {_SLINGSHOT_THRESHOLD}), "
        f"got {m['max_over_median']:.2f}"
    )
    # A heavy tail also shows up as a nonzero tail fraction.
    assert m["tail_fraction"] > 0.0


# ---------------------------------------------------------------------------
# 2. Tame config does NOT slingshot (contrast — keeps test 1 meaningful).
# ---------------------------------------------------------------------------

def test_tame_config_has_no_big_slingshot():
    disp = _run_factory(M=5, S_gpc=20, nodes=True)
    m = slingshot_metrics(disp)
    assert m["max_over_median"] < _TAME_BOUND, (
        f"tame config should have a modest tail (< {_TAME_BOUND}), "
        f"got {m['max_over_median']:.2f}"
    )
    assert m["tail_fraction"] == 0.0


# ---------------------------------------------------------------------------
# 3. ROOT CAUSE: the slingshot comes from the NODES, not particle-particle.
# ---------------------------------------------------------------------------

def test_slingshot_from_nodes_not_particles():
    disp_on = _run_factory(M=1000, S_gpc=10, nodes=True)
    disp_off = _run_factory(M=1000, S_gpc=10, nodes=False)
    m_on = slingshot_metrics(disp_on)
    m_off = slingshot_metrics(disp_off)

    # Nodes ON: a big slingshot. Nodes OFF: tame.
    assert m_on["max_over_median"] > _SLINGSHOT_THRESHOLD
    assert m_off["max_over_median"] < _TAME_BOUND
    # The node close-pass dominates: ON tail is at least 10x the OFF tail.
    assert m_on["max_over_median"] > 10.0 * m_off["max_over_median"]
    # And the runaway maximum displacement is far larger with nodes on.
    assert m_on["max"] > 10.0 * m_off["max"]


# ---------------------------------------------------------------------------
# 4. A diagnostic NODE softening tames the slingshot (motivates Section 4).
# ---------------------------------------------------------------------------

def test_node_softening_reduces_slingshot():
    # Unsoftened (product behaviour) — big slingshot.
    sp, box, a_start, n_steps = _make_params(M=1000, S_gpc=10)
    sim_hard = CosmologicalSimulation(
        sp, box, a_start, use_external_nodes=True, use_dark_energy=False)
    sim_hard.run(t_end_Gyr=sp.t_duration_Gyr, n_steps=n_steps,
                 save_interval=max(1, n_steps // 4))
    m_hard = slingshot_metrics(_disp_from_sim(sim_hard))
    assert m_hard["max_over_median"] > _SLINGSHOT_THRESHOLD

    # Softened node force (~1 Gpc) via diagnostic monkeypatch — tamed.
    sp2, box2, a2, n2 = _make_params(M=1000, S_gpc=10)
    sim_soft = CosmologicalSimulation(
        sp2, box2, a2, use_external_nodes=True, use_dark_energy=False)
    grid = sim_soft.hmea_grid
    npos = grid.get_positions()
    nmass = grid.get_masses()
    soft_m = 1.0 * _G
    Gc = CosmologicalConstants.G

    def _softened_batch(positions, use_numba=True):
        return softened_node_acceleration(positions, npos, nmass, soft_m, Gc)

    grid.calculate_tidal_acceleration_batch = _softened_batch
    sim_soft.run(t_end_Gyr=sp2.t_duration_Gyr, n_steps=n2,
                 save_interval=max(1, n2 // 4))
    m_soft = slingshot_metrics(_disp_from_sim(sim_soft))

    # Softening collapses the tail to tame levels and shrinks the max enormously.
    assert m_soft["max_over_median"] < _TAME_BOUND
    assert m_soft["max"] < 0.1 * m_hard["max"]


# ---------------------------------------------------------------------------
# 5. Pure-helper test for the diagnostic softened node force.
# ---------------------------------------------------------------------------

def test_softened_node_acceleration_matches_newton_far_field():
    # Far from a single node, softening is negligible: a ~ G m / r^2 toward node.
    Gc = CosmologicalConstants.G
    node = np.array([[1.0e25, 0.0, 0.0]])
    mass = np.array([1.0e53])
    part = np.array([[0.0, 0.0, 0.0]])
    r = 1.0e25
    a_soft = softened_node_acceleration(part, node, mass, softening_m=1.0e10, G=Gc)
    expected = Gc * mass[0] / r**2
    # Points toward the node (+x), magnitude ~ Newtonian.
    assert a_soft[0, 0] > 0.0
    assert abs(a_soft[0, 0] - expected) / expected < 1e-6


def test_softened_node_acceleration_finite_at_node():
    # AT a node, the hard 1/r^2 force diverges; softened stays finite.
    Gc = CosmologicalConstants.G
    node = np.array([[0.0, 0.0, 0.0]])
    mass = np.array([1.0e53])
    part = np.array([[0.0, 0.0, 0.0]])  # exactly on the node
    a = softened_node_acceleration(part, node, mass, softening_m=1.0e24, G=Gc)
    assert np.all(np.isfinite(a))
    assert np.allclose(a, 0.0)  # symmetric -> zero net at the centre


def test_larger_softening_gives_smaller_close_force():
    # Closer than the softening length, a larger eps means a smaller force.
    Gc = CosmologicalConstants.G
    node = np.array([[1.0e23, 0.0, 0.0]])  # very close (< 1 Gpc)
    mass = np.array([1.0e53])
    part = np.array([[0.0, 0.0, 0.0]])
    a_small = softened_node_acceleration(part, node, mass, softening_m=1.0e24, G=Gc)
    a_big = softened_node_acceleration(part, node, mass, softening_m=5.0e24, G=Gc)
    assert np.linalg.norm(a_big) < np.linalg.norm(a_small)
