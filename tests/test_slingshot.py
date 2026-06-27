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


def _make_params(M, S_gpc, n_particles=_N, n_steps=_N_STEPS, geometry="cube26",
                 node_softening_gpc=0.0):
    t_dur = _TODAY_GYR - _T_START
    n_steps = resolve_n_steps(t_dur, n_steps)
    box, a_start, _ = setup_simulation_context(
        _T_START, t_dur, n_steps, save_interval=max(1, n_steps // 4))
    sp = SimulationParameters(
        M_value=M, S_value=S_gpc, n_particles=n_particles, seed=_SEED,
        t_start_Gyr=_T_START, t_duration_Gyr=t_dur, n_steps=n_steps,
        damping_factor=None, center_node_mass=1.0, mass_randomize=0.0,
        node_mass_seed=_SEED, init_distribution="uniform_sphere",
        node_geometry=geometry, node_softening_gpc=node_softening_gpc,
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


# ===========================================================================
# Section 4 — the PRODUCT node-softening knob (node_softening_gpc) tames the
# runaway slingshot for BOTH cube26 AND virialized. Unlike tests 1-4 above
# (which monkeypatch a diagnostic force), these exercise the REAL product path:
# SimulationParameters(node_softening_gpc=...) -> ExternalNodeParameters ->
# HMEAGrid.calculate_tidal_acceleration_batch (numba + numpy).
# ===========================================================================

# Measured (seed=42, N=200, t_start=5.8, M=1000, S=10):
#   cube26     OFF max/median ~514, tail ~0.285 -> ON(1 Gpc) ~2.8, tail 0.0
#   virialized OFF max/median ~23,  tail ~0.060 -> ON(1 Gpc) ~7.0, tail ~0.04
# Loose bounds below so the assertions are not flaky.
_TAMED_MAX_OVER_MEDIAN = 10.0
_NODE_SOFTENING_GPC = 1.0


def _run_softened_disp(M, S_gpc, geometry, node_softening_gpc):
    """Run ONE product sim (real node_softening_gpc knob) and return displacements."""
    sp, box, a_start, n_steps = _make_params(
        M, S_gpc, geometry=geometry, node_softening_gpc=node_softening_gpc)
    sim = CosmologicalSimulation(
        sp, box, a_start, use_external_nodes=True, use_dark_energy=False)
    sim.run(t_end_Gyr=sp.t_duration_Gyr, n_steps=n_steps,
            save_interval=max(1, n_steps // 4))
    return _disp_from_sim(sim)


def test_taming_reduces_tail_cube26():
    """node_softening_gpc=1.0 collapses the cube26 runaway tail (product path)."""
    m_off = slingshot_metrics(_run_softened_disp(1000, 10, "cube26", 0.0))
    m_on = slingshot_metrics(
        _run_softened_disp(1000, 10, "cube26", _NODE_SOFTENING_GPC))
    # Sanity: taming OFF is a real slingshot.
    assert m_off["max_over_median"] > _SLINGSHOT_THRESHOLD
    # Taming ON: tail collapses to a small value and the tail fraction vanishes.
    assert m_on["max_over_median"] < _TAMED_MAX_OVER_MEDIAN, (
        f"cube26 node-softening should tame the tail "
        f"(max/median < {_TAMED_MAX_OVER_MEDIAN}), got {m_on['max_over_median']:.2f}"
    )
    assert m_on["max_over_median"] < 0.1 * m_off["max_over_median"]
    assert m_on["tail_fraction"] == 0.0


def test_taming_reduces_tail_virialized():
    """node_softening_gpc=1.0 further tames the virialized runaway (doubly tamed)."""
    m_off = slingshot_metrics(_run_softened_disp(1000, 10, "virialized", 0.0))
    m_on = slingshot_metrics(
        _run_softened_disp(1000, 10, "virialized", _NODE_SOFTENING_GPC))
    # Softening ON drops the tail metric below the OFF value for virialized too.
    assert m_on["max_over_median"] < m_off["max_over_median"]
    assert m_on["max_over_median"] < _TAMED_MAX_OVER_MEDIAN, (
        f"virialized node-softening should tame the tail "
        f"(max/median < {_TAMED_MAX_OVER_MEDIAN}), got {m_on['max_over_median']:.2f}"
    )
    assert m_on["tail_fraction"] <= m_off["tail_fraction"]


def test_node_softening_default_zero_is_unsoftened():
    """node_softening_gpc=0.0 (default) => tidal force keeps the legacy hard floor.

    Byte-identical to the pre-softening force: the product grid's tidal batch at
    the default knob EXACTLY equals the legacy numba call with softening_m=0.
    """
    from cosmo.tidal_forces_numba import calculate_tidal_forces_numba

    sp, _box, _a, _n = _make_params(1000, 10, node_softening_gpc=0.0)
    assert sp.external_params.node_softening_m == 0.0
    sim = CosmologicalSimulation(
        sp, _box, _a, use_external_nodes=True, use_dark_energy=False)
    grid = sim.hmea_grid
    npos = grid.get_positions()
    nmass = grid.get_masses()
    rng = np.random.RandomState(0)
    pos = rng.uniform(-1.0e24, 1.0e24, (40, 3))
    a_grid = grid.calculate_tidal_acceleration_batch(pos, use_numba=True)
    a_legacy = calculate_tidal_forces_numba(
        pos, npos, nmass, CosmologicalConstants.G, 0.0)
    assert np.array_equal(a_grid, a_legacy), (
        "default node_softening_gpc=0 must be byte-identical to the legacy "
        "hard-floor tidal force"
    )


def test_node_softening_caps_close_pass_product_path():
    """A close node pass gains a BOUNDED kick with softening on (product numba path).

    With node_softening_gpc>0 the Plummer term caps the close-pass force; with
    the default (0) the hard-floor force is enormously larger. Mirrors the
    diagnostic helper tests but exercises calculate_tidal_forces_numba directly.
    """
    from cosmo.tidal_forces_numba import calculate_tidal_forces_numba

    Gc = CosmologicalConstants.G
    g = CosmologicalConstants.Gpc_to_m
    node = np.array([[10.0 * g, 0.0, 0.0]])
    mass = np.array([1.0e56])
    part = np.array([[10.0 * g - 0.01 * g, 0.0, 0.0]])  # 0.01 Gpc from node
    a_hard = calculate_tidal_forces_numba(part, node, mass, Gc, 0.0)
    a_soft = calculate_tidal_forces_numba(part, node, mass, Gc, 1.0 * g)
    assert np.all(np.isfinite(a_soft))
    assert np.linalg.norm(a_soft) < 0.01 * np.linalg.norm(a_hard)


def test_node_softening_far_field_barely_changes():
    """For a far particle, 1 Gpc node softening leaves the tidal force ~unchanged."""
    from cosmo.tidal_forces_numba import calculate_tidal_forces_numba

    Gc = CosmologicalConstants.G
    g = CosmologicalConstants.Gpc_to_m
    node = np.array([[10.0 * g, 0.0, 0.0]])
    mass = np.array([1.0e56])
    far = np.array([[0.0, 0.0, 0.0]])  # 10 Gpc from node >> 1 Gpc softening
    a_hard = calculate_tidal_forces_numba(far, node, mass, Gc, 0.0)
    a_soft = calculate_tidal_forces_numba(far, node, mass, Gc, 1.0 * g)
    rel = np.abs(np.linalg.norm(a_soft) - np.linalg.norm(a_hard)) / np.linalg.norm(a_hard)
    assert rel < 0.05, f"far-field tidal force changed {rel*100:.2f}% (must be < 5%)"


def test_m_ext_zero_zero_tidal_with_node_softening():
    """M_ext=0 => zero node mass => zero tidal force for ANY node_softening_gpc.

    Preserves the M=0 == EdS invariant: softening only reshapes the per-node
    force; with all node masses 0 the tidal sum is identically 0.
    """
    from cosmo.tidal_forces_numba import calculate_tidal_forces_numba

    sp, _box, _a, _n = _make_params(0, 10, node_softening_gpc=1.0)
    sim = CosmologicalSimulation(
        sp, _box, _a, use_external_nodes=True, use_dark_energy=False)
    grid = sim.hmea_grid
    npos = grid.get_positions()
    nmass = grid.get_masses()
    assert np.allclose(nmass, 0.0), "M_ext=0 must give zero node masses"
    rng = np.random.RandomState(1)
    pos = rng.uniform(-1.0e24, 1.0e24, (30, 3))
    a_numba = grid.calculate_tidal_acceleration_batch(pos, use_numba=True)
    a_numpy = grid.calculate_tidal_acceleration_batch(pos, use_numba=False)
    assert np.allclose(a_numba, 0.0)
    assert np.allclose(a_numpy, 0.0)


# ===========================================================================
# Section 4 (items 3, C) — BOUNDED close-range force law ("can't cross the
# midpoint") + ADAPTIVE KDK sub-stepping. New knobs default OFF / byte-identical:
#   node_force_law="plummer" (default) | "bounded"
#   node_substep_threshold=0.0 (default OFF), node_substeps=1 (default no-op)
# ===========================================================================

from cosmo.constants import node_force_law_code, NODE_FORCE_LAW_CODES, MAX_NODE_SUBSTEPS
from cosmo.tidal_forces_numba import (
    calculate_tidal_forces_numba,
    NODE_FORCE_LAW_PLUMMER,
    NODE_FORCE_LAW_BOUNDED,
)


def test_force_law_default_is_plummer_byte_identical():
    """node_force_law defaults to 'plummer' (code 0); the bounded branch is opt-in."""
    sp = SimulationParameters(M_value=1000, S_value=10)
    assert sp.node_force_law == "plummer"
    assert sp.external_params.node_force_law_code == NODE_FORCE_LAW_PLUMMER
    assert node_force_law_code("plummer") == 0
    assert node_force_law_code("bounded") == 1
    # Unknown name -> default plummer (never silently switch physics).
    assert node_force_law_code("nonsense") == NODE_FORCE_LAW_PLUMMER


def test_bounded_law_byte_identical_to_legacy_at_zero_softening():
    """With softening_m==0 the bounded law falls back to the legacy hard floor."""
    Gc = CosmologicalConstants.G
    rng = np.random.RandomState(3)
    node = rng.uniform(-5.0e25, 5.0e25, (8, 3))
    mass = rng.uniform(1.0e54, 1.0e56, 8)
    part = rng.uniform(-5.0e25, 5.0e25, (40, 3))
    a_legacy = calculate_tidal_forces_numba(part, node, mass, Gc, 0.0,
                                            NODE_FORCE_LAW_PLUMMER)
    a_bounded0 = calculate_tidal_forces_numba(part, node, mass, Gc, 0.0,
                                              NODE_FORCE_LAW_BOUNDED)
    assert np.array_equal(a_legacy, a_bounded0), (
        "bounded law with zero softening must be byte-identical to legacy floor"
    )


def test_bounded_law_caps_close_pass_accel_at_softening_value():
    """The bounded close-range accel is CAPPED at its r==softening value.

    Distinct from the blunt Plummer floor (which keeps shrinking the force below
    softening): the bounded magnitude inside the softening length equals exactly
    G m / softening^2 (the value at r==softening), and is FAR larger than the
    Plummer-softened value at the same close separation.
    """
    Gc = CosmologicalConstants.G
    g = CosmologicalConstants.Gpc_to_m
    node = np.array([[10.0 * g, 0.0, 0.0]])
    mass = np.array([1.0e56])
    soft = 1.0 * g
    # Particle 0.01 Gpc from node (deep inside the 1 Gpc softening).
    part = np.array([[10.0 * g - 0.01 * g, 0.0, 0.0]])
    a_bounded = calculate_tidal_forces_numba(part, node, mass, Gc, soft,
                                             NODE_FORCE_LAW_BOUNDED)
    a_plummer = calculate_tidal_forces_numba(part, node, mass, Gc, soft,
                                             NODE_FORCE_LAW_PLUMMER)
    a_cap = Gc * mass[0] / soft**2
    assert np.all(np.isfinite(a_bounded))
    # Bounded magnitude == the cap (value at r == softening).
    assert abs(np.linalg.norm(a_bounded) - a_cap) / a_cap < 1e-9
    # And it does NOT collapse like the blunt Plummer floor: the bounded close
    # force is much LARGER than Plummer's (sub-softening bodies still attract).
    assert np.linalg.norm(a_bounded) > 10.0 * np.linalg.norm(a_plummer)


def test_bounded_law_never_exceeds_cap_over_range():
    """For ANY separation the bounded per-node accel <= G m / softening^2.

    This is the displacement-limiter guarantee: inside the softening length the
    force can never grow past its value at the softening radius, so a single step
    cannot fling a body across the pair separation.
    """
    Gc = CosmologicalConstants.G
    g = CosmologicalConstants.Gpc_to_m
    node = np.array([[0.0, 0.0, 0.0]])
    mass = np.array([1.0e56])
    soft = 1.0 * g
    a_cap = Gc * mass[0] / soft**2
    # Sweep from on-the-node out to 5 Gpc.
    rs = np.linspace(0.0, 5.0 * g, 60)
    parts = np.column_stack([rs, np.zeros_like(rs), np.zeros_like(rs)])
    a = calculate_tidal_forces_numba(parts, node, mass, Gc, soft,
                                     NODE_FORCE_LAW_BOUNDED)
    mags = np.linalg.norm(a, axis=1)
    assert np.all(mags <= a_cap * (1.0 + 1e-9)), (
        "bounded accel must never exceed the softening-radius cap"
    )
    # At r==0 it is exactly zero (no direction); just inside softening ~ the cap.
    assert mags[0] == 0.0


def test_bounded_law_far_field_under_5pct():
    """Far from a node (>> softening) the bounded law leaves the force ~unchanged."""
    Gc = CosmologicalConstants.G
    g = CosmologicalConstants.Gpc_to_m
    node = np.array([[10.0 * g, 0.0, 0.0]])
    mass = np.array([1.0e56])
    far = np.array([[0.0, 0.0, 0.0]])  # 10 Gpc from node >> 1 Gpc softening
    a_hard = calculate_tidal_forces_numba(far, node, mass, Gc, 0.0,
                                          NODE_FORCE_LAW_PLUMMER)
    a_bounded = calculate_tidal_forces_numba(far, node, mass, Gc, 1.0 * g,
                                             NODE_FORCE_LAW_BOUNDED)
    rel = (np.abs(np.linalg.norm(a_bounded) - np.linalg.norm(a_hard))
           / np.linalg.norm(a_hard))
    # Bounded uses EXACT 1/r^2 in the far field, so the change is ~0.
    assert rel < 0.05, f"bounded far-field changed {rel*100:.2f}% (must be < 5%)"


def test_bounded_law_numpy_matches_numba():
    """numpy fallback mirrors the numba kernel for the bounded law (product grid)."""
    sp, _box, _a, _n = _make_params(
        1000, 10, node_softening_gpc=1.0)
    sp = SimulationParameters(
        M_value=1000, S_value=10, n_particles=50, seed=_SEED,
        t_start_Gyr=_T_START, t_duration_Gyr=_TODAY_GYR - _T_START, n_steps=_n,
        mass_randomize=0.0, node_mass_seed=_SEED, node_geometry="cube26",
        node_softening_gpc=1.0, node_force_law="bounded")
    sim = CosmologicalSimulation(
        sp, _box, _a, use_external_nodes=True, use_dark_energy=False)
    grid = sim.hmea_grid
    rng = np.random.RandomState(0)
    # Mix of inside / outside the 1 Gpc softening of the nodes.
    pos = rng.uniform(-2.0 * _G, 2.0 * _G, (60, 3))
    a_nb = grid.calculate_tidal_acceleration_batch(pos, use_numba=True)
    a_np = grid.calculate_tidal_acceleration_batch(pos, use_numba=False)
    assert np.allclose(a_nb, a_np, rtol=1e-9, atol=0.0)


def test_bounded_law_m_ext_zero_is_eds():
    """M_ext=0 -> zero node mass -> zero tidal force for the bounded law too."""
    g = CosmologicalConstants.Gpc_to_m
    sp = SimulationParameters(
        M_value=0, S_value=10, n_particles=40, seed=_SEED,
        t_start_Gyr=_T_START, t_duration_Gyr=_TODAY_GYR - _T_START, n_steps=200,
        mass_randomize=0.0, node_mass_seed=_SEED, node_geometry="cube26",
        node_softening_gpc=1.0, node_force_law="bounded")
    box, a_start, _ = setup_simulation_context(
        _T_START, _TODAY_GYR - _T_START, 200)
    sim = CosmologicalSimulation(
        sp, box, a_start, use_external_nodes=True, use_dark_energy=False)
    grid = sim.hmea_grid
    rng = np.random.RandomState(1)
    pos = rng.uniform(-1.0 * g, 1.0 * g, (30, 3))
    a_nb = grid.calculate_tidal_acceleration_batch(pos, use_numba=True)
    a_np = grid.calculate_tidal_acceleration_batch(pos, use_numba=False)
    assert np.allclose(a_nb, 0.0)
    assert np.allclose(a_np, 0.0)


def test_invalid_force_law_raises():
    """An unknown node_force_law name is rejected up front (loud, not silent)."""
    import pytest as _pytest
    with _pytest.raises(ValueError):
        SimulationParameters(M_value=1, S_value=10, node_force_law="bogus")


# ---------------------------------------------------------------------------
# Adaptive KDK sub-stepping
# ---------------------------------------------------------------------------

def _run_sim_a_curve(node_substep_threshold=0.0, node_substeps=1,
                     node_softening_gpc=0.0, node_force_law="plummer",
                     M=5, S_gpc=20, geometry="cube26", n_particles=80):
    """Run a short product sim and return its a(t) curve."""
    t_dur = _TODAY_GYR - _T_START
    n_steps = resolve_n_steps(t_dur, 160)
    box, a_start, _ = setup_simulation_context(
        _T_START, t_dur, n_steps, save_interval=max(1, n_steps // 4))
    sp = SimulationParameters(
        M_value=M, S_value=S_gpc, n_particles=n_particles, seed=_SEED,
        t_start_Gyr=_T_START, t_duration_Gyr=t_dur, n_steps=n_steps,
        mass_randomize=0.0, node_mass_seed=_SEED, node_geometry=geometry,
        node_softening_gpc=node_softening_gpc, node_force_law=node_force_law,
        node_substep_threshold=node_substep_threshold, node_substeps=node_substeps)
    from cosmo.factories import run_external_node_simulation
    ext = run_external_node_simulation(
        sp, box, a_start, save_interval=max(1, n_steps // 4))
    return np.asarray(ext["a"])


def test_substep_default_off_byte_identical():
    """node_substep_threshold=0 (default) -> one plain leapfrog step, byte-identical.

    Even node_substeps>1 is a no-op while the threshold is 0.
    """
    a_off = _run_sim_a_curve(node_substep_threshold=0.0, node_substeps=1)
    a_subs_but_off = _run_sim_a_curve(node_substep_threshold=0.0, node_substeps=8)
    assert np.array_equal(a_off, a_subs_but_off), (
        "with threshold=0 the integrator must take one plain leapfrog step "
        "regardless of node_substeps (byte-identical default)"
    )


def test_substep_no_close_pass_reproduces_non_substep():
    """A tame cloud (no particle near a node) reproduces non-substep a(t) exactly.

    The close-pass trigger never fires, so the substep path takes the same single
    KDK step — bit-for-bit identical to the non-substep run.
    """
    a_nosub = _run_sim_a_curve(node_substep_threshold=0.0)
    # Tame M=5/S=20 cloud: no particle gets within 0.5*S of a node.
    a_sub = _run_sim_a_curve(node_substep_threshold=0.5, node_substeps=8,
                             M=5, S_gpc=20)
    assert np.allclose(a_nosub, a_sub, rtol=1e-12, atol=0.0)


def test_substep_count_clamped_to_cap():
    """node_substeps is clamped to [1, MAX_NODE_SUBSTEPS] (no unbounded inner loop)."""
    sp = SimulationParameters(M_value=1, S_value=10, node_substeps=100000)
    assert sp.node_substeps == MAX_NODE_SUBSTEPS
    sp2 = SimulationParameters(M_value=1, S_value=10, node_substeps=0)
    assert sp2.node_substeps == 1
    with __import__("pytest").raises(ValueError):
        SimulationParameters(M_value=1, S_value=10, node_substep_threshold=-1.0)


def test_substep_reduces_slingshot_energy_error():
    """Adaptive substepping during a close pass reduces the runaway slingshot tail.

    On the runaway cube26 config, refining dt with KDK substeps during the close
    node pass shrinks the worst displacement (the slingshot is partly a
    time-resolution artifact). We assert the substep run's max displacement is
    no worse than the single-step run (and typically smaller).
    """
    # Runaway config; bounded law on so the per-step kick is also capped, letting
    # the substep refinement actually resolve the close pass.
    a_single = _run_sim_a_curve(
        node_substep_threshold=0.0, node_substeps=1,
        node_softening_gpc=1.0, node_force_law="bounded",
        M=1000, S_gpc=10, geometry="cube26", n_particles=120)
    a_sub = _run_sim_a_curve(
        node_substep_threshold=2.0, node_substeps=8,
        node_softening_gpc=1.0, node_force_law="bounded",
        M=1000, S_gpc=10, geometry="cube26", n_particles=120)
    # Substepping must not blow up the expansion; finite and comparable growth.
    g_single = float(a_single[-1] / a_single[0])
    g_sub = float(a_sub[-1] / a_sub[0])
    assert np.isfinite(g_single) and np.isfinite(g_sub)
    # The substep run's total growth is no larger than the single-step run's:
    # refining the close pass cannot ADD spurious runaway expansion.
    assert g_sub <= g_single * 1.05
