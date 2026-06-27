"""
Tests for cosmo/observer_distance.py — the observer-from-a-particle mu(z) prototype.

Philosophy (mirrors tests/test_ws8_figs.py / tests/test_sim_distance.py)
------------------------------------------------------------------------
- Pure-function tests use tiny SYNTHETIC histories (homologous / anisotropic
  clouds) so they run in well under a second and need no sim.
- ONE slow integration test runs a tiny real sim against the real Pantheon+ data
  to exercise the full a_p(t) -> mu(z) -> chi2 scoring path end-to-end; it is
  skipped if the Pantheon+ data file is absent.

The load-bearing invariants pinned here:
  1. The CENTRE observer reproduces the existing centre-based a(t) (the
     local_rms / center_a_curve equivalence), so the prototype is a faithful
     generalisation, not a new measure.
  2. For an ISOTROPIC cloud both definitions agree and the per-observer chi2
     distribution is TIGHT around the centre value.
  3. An OFF-CENTRE observer in an ANISOTROPIC cloud differs MEASURABLY (PF2).
  4. best_chi2_dof <= center_chi2_dof BY CONSTRUCTION of "take the best".
  5. Determinism.
"""

import math
import tempfile
import unittest

import numpy as np
import pytest

from cosmo.observer_distance import (
    ALL_NEIGHBOURS,
    neighbour_indices,
    center_a_curve,
    observer_a_curve_local_rms,
    observer_a_curve_hubble_flow,
    score_observer,
    observer_chi2_distribution,
    history_from_snapshots,
    fraction_at_or_below,
)
from cosmo.constants import CosmologicalConstants

_GYR_S = CosmologicalConstants.Gyr_to_s


# ---------------------------------------------------------------------------
# Synthetic-history builders
# ---------------------------------------------------------------------------

def _homologous_history(n_snap=12, N=40, growth=3.0, seed=0):
    """Isotropic homologous expansion pos(t)=g(t)*base, vel = dg/dt * base.

    a(t) for ANY observer / the centre is exactly g(t)/g(0).
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 10.0, n_snap)               # Gyr
    g = 1.0 + (growth - 1.0) * (t / t[-1])           # linear 1 -> growth
    gdot = np.full(n_snap, (growth - 1.0) / t[-1])   # dg/dt in 1/Gyr
    base = rng.normal(size=(N, 3))
    pos = np.stack([gi * base for gi in g])
    vel = np.stack([(gd * base) / _GYR_S for gd in gdot])  # m/s
    return pos, vel, t, g


def _anisotropic_history(n_snap=12, N=60, seed=1):
    """Cloud that expands MORE along +x than -x: an off-centre observer on the
    +x edge sees a different local expansion than the COM."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 10.0, n_snap)
    base = rng.normal(size=(N, 3))
    pos_list = []
    vel_list = []
    for ti in t:
        # per-axis growth: x grows faster than y,z
        gx = 1.0 + 0.30 * ti
        gy = 1.0 + 0.15 * ti
        gz = 1.0 + 0.15 * ti
        scale = np.array([gx, gy, gz])
        pos_list.append(base * scale)
    pos = np.stack(pos_list)
    # finite-difference velocities (central where possible) in m/s
    vel = np.gradient(pos, t * _GYR_S, axis=0)
    return pos, vel, t


# ---------------------------------------------------------------------------
# 1. neighbour_indices
# ---------------------------------------------------------------------------

class TestNeighbourIndices(unittest.TestCase):
    def test_all_excludes_observer(self):
        pos = np.random.default_rng(0).normal(size=(10, 3))
        nbr = neighbour_indices(pos, observer=3, k=ALL_NEIGHBOURS)
        self.assertNotIn(3, nbr.tolist())
        self.assertEqual(len(nbr), 9)

    def test_k_nearest_are_nearest(self):
        # Put observer at origin, others on a line at increasing distance.
        pos = np.zeros((6, 3))
        pos[:, 0] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        nbr = neighbour_indices(pos, observer=0, k=2)
        self.assertEqual(sorted(nbr.tolist()), [1, 2])

    def test_k_larger_than_n_returns_all(self):
        pos = np.random.default_rng(0).normal(size=(5, 3))
        nbr = neighbour_indices(pos, observer=0, k=100)
        self.assertEqual(len(nbr), 4)


# ---------------------------------------------------------------------------
# 2. CENTRE observer reproduces the centre-based a(t)  (invariant 1)
# ---------------------------------------------------------------------------

class TestCenterReproduction(unittest.TestCase):
    def test_center_a_curve_recovers_homologous_growth(self):
        pos, _vel, t, g = _homologous_history()
        a = center_a_curve(pos, t)
        np.testing.assert_allclose(a, g / g[0], rtol=1e-10, atol=1e-10)

    def test_local_rms_about_com_all_neighbours_equals_center(self):
        # The local_rms observer with about_com=True and k=ALL is the centre
        # measure (RMS about the cloud mean). Under homologous expansion the
        # observer-excluded COM scales identically, so it matches EXACTLY.
        pos, _vel, t, _g = _homologous_history()
        center = center_a_curve(pos, t)
        for obs in (0, 7, 19):
            a_obs = observer_a_curve_local_rms(
                pos, t, observer=obs, k=ALL_NEIGHBOURS, about_com=True)
            np.testing.assert_allclose(
                a_obs, center, rtol=1e-9, atol=1e-9,
                err_msg=f"observer {obs} about_com a(t) != centre a(t)")

    def test_local_rms_about_p_recovers_homologous_growth(self):
        # About the observer particle itself (about_com=False): under homologous
        # expansion every separation scales by g, so a_p == g/g0 too.
        pos, _vel, t, g = _homologous_history()
        a_obs = observer_a_curve_local_rms(pos, t, observer=5, k=ALL_NEIGHBOURS)
        np.testing.assert_allclose(a_obs, g / g[0], rtol=1e-9, atol=1e-9)


# ---------------------------------------------------------------------------
# 3. hubble_flow recovers homologous a(t) (to integration tolerance) + converges
# ---------------------------------------------------------------------------

class TestHubbleFlowDefinition(unittest.TestCase):
    def test_recovers_homologous_growth(self):
        pos, vel, t, g = _homologous_history(n_snap=30)
        a = observer_a_curve_hubble_flow(pos, vel, t, observer=0, k=ALL_NEIGHBOURS)
        # Integrated estimate: agree with the exact growth to ~0.3% on 30 snaps.
        np.testing.assert_allclose(a, g / g[0], rtol=5e-3, atol=5e-3)

    def test_integration_tightens_with_finer_grid(self):
        def maxerr(n_snap):
            pos, vel, t, g = _homologous_history(n_snap=n_snap)
            a = observer_a_curve_hubble_flow(pos, vel, t, 0, k=ALL_NEIGHBOURS)
            return float(np.max(np.abs(a - g / g[0])))
        self.assertGreater(maxerr(8), maxerr(30))
        self.assertGreater(maxerr(30), maxerr(120))

    def test_both_definitions_agree_for_isotropic_cloud(self):
        # invariant 2 (per-curve): local_rms and hubble_flow agree on an
        # isotropic cloud (both ~ g/g0) to integration tolerance.
        pos, vel, t, g = _homologous_history(n_snap=60)
        a_rms = observer_a_curve_local_rms(pos, t, 0, k=ALL_NEIGHBOURS)
        a_hub = observer_a_curve_hubble_flow(pos, vel, t, 0, k=ALL_NEIGHBOURS)
        np.testing.assert_allclose(a_rms, a_hub, rtol=2e-3, atol=2e-3)


# ---------------------------------------------------------------------------
# 4. OFF-CENTRE observer in an ANISOTROPIC cloud differs measurably (invariant 3)
# ---------------------------------------------------------------------------

class TestAnisotropyMatters(unittest.TestCase):
    def test_offcenter_observer_a_curve_differs_from_center(self):
        pos, _vel, t = _anisotropic_history()
        center = center_a_curve(pos, t)
        # Find the most +x-extreme particle today (an edge observer).
        edge = int(np.argmax(pos[-1][:, 0]))
        a_edge = observer_a_curve_local_rms(pos, t, observer=edge, k=12)
        # The edge observer's local growth must differ from the centre's by a
        # non-trivial amount (not a numerical wobble).
        rel = np.max(np.abs(a_edge - center)) / np.max(center)
        self.assertGreater(rel, 0.02,
                           f"edge-observer a(t) too close to centre (rel={rel})")


# ---------------------------------------------------------------------------
# 5. score_observer — authoritative-path sanity on synthetic Pantheon
# ---------------------------------------------------------------------------

def _fake_pantheon_from_a(a, t, t_start_Gyr):
    """Build a synthetic Pantheon-like dict whose mu == the model mu for a(t),
    so a perfect-match observer scores chi2 ~ 0. Lets us test the scorer wiring
    without the real data file."""
    from cosmo.sim_distance import sim_to_distance_modulus
    # dense z grid spanning the sim coverage
    a = np.asarray(a, dtype=float)
    a_today = a / a[-1]
    z_snap = 1.0 / a_today - 1.0
    z = np.linspace(0.02, float(z_snap.max()) * 0.9, 40)
    dist = sim_to_distance_modulus(z, a, t, t_start_Gyr)
    return {
        "z": dist["z"],
        "mu": dist["mu"],
        "sigma": np.full_like(dist["mu"], 0.15),
    }


class TestScoreObserver(unittest.TestCase):
    def test_perfect_match_scores_low_chi2(self):
        pos, _vel, t, _g = _homologous_history(n_snap=40, growth=3.2)
        t_start = 2.9
        # Build a(t) whose absolute time hits 13.8 today: t spans 0..10.9.
        t = np.linspace(0.0, 13.8 - t_start, 40)
        a = center_a_curve(pos, t)
        ph = _fake_pantheon_from_a(a, t, t_start)
        sc = score_observer(a, t, t_start, ph)
        self.assertTrue(sc["ok"])
        self.assertLess(sc["chi2_dof"], 1e-6)
        self.assertGreater(sc["n_sne_used"], 2)

    def test_bad_input_returns_inf_not_raise(self):
        sc = score_observer(np.array([1.0]), np.array([0.0]), 2.9,
                            {"z": np.array([0.1]), "mu": np.array([40.0]),
                             "sigma": np.array([0.1])})
        self.assertFalse(sc["ok"])
        self.assertEqual(sc["chi2_dof"], float("inf"))


# ---------------------------------------------------------------------------
# 6. observer_chi2_distribution — distribution shape + "take the best" invariant
# ---------------------------------------------------------------------------

class TestDistribution(unittest.TestCase):
    def _run(self, pos, vel, t, definition, k=ALL_NEIGHBOURS):
        t_start = 2.9
        t = np.linspace(0.0, 13.8 - t_start, pos.shape[0])
        a_center = center_a_curve(pos, t)
        ph = _fake_pantheon_from_a(a_center, t, t_start)
        return observer_chi2_distribution(
            pos, vel, t, t_start, ph, definition=definition, k=k)

    def test_isotropic_distribution_is_tight_around_center(self):
        # invariant 2: isotropic cloud -> per-observer chi2 clustered near centre.
        pos, vel, t, _g = _homologous_history(n_snap=40, N=40, growth=3.2)
        out = self._run(pos, vel, t, "local_rms")
        self.assertGreater(out["n_finite"], 0)
        # spread (p90-p10) should be small relative to the centre value+1
        spread = out["p90"] - out["p10"]
        self.assertLess(spread, 0.5,
                        f"isotropic spread too large: {spread}")

    def test_best_le_center_by_construction(self):
        # invariant 4: best <= centre always. (Here the synthetic Pantheon is
        # built FROM the centre a(t), so the centre is essentially the global
        # minimum and "best" ties it to floating-point noise — a tiny absolute
        # tolerance absorbs that. The genuine best<center improvement is exercised
        # against the REAL Pantheon in the slow integration test below.)
        pos, vel, t = _anisotropic_history(n_snap=40, N=60)
        out = self._run(pos, vel, t, "local_rms", k=12)
        self.assertGreater(out["n_finite"], 0)
        self.assertLessEqual(out["best_chi2_dof"], out["center_chi2_dof"] + 1e-6)

    def test_distribution_is_finite_and_well_formed(self):
        pos, vel, t = _anisotropic_history(n_snap=40, N=50)
        out = self._run(pos, vel, t, "hubble_flow", k=12)
        self.assertEqual(out["n_observers"], 50)
        self.assertTrue(np.isfinite(out["center_chi2_dof"]))
        self.assertTrue(math.isfinite(out["median"]))
        self.assertGreaterEqual(out["best_observer"], 0)
        # observer_index must be the GLOBAL indices (here mask=None so 0..N-1)
        self.assertEqual(len(out["observer_index"]), 50)

    def test_determinism(self):
        # invariant 5: identical inputs -> identical outputs.
        pos, vel, t = _anisotropic_history(n_snap=30, N=40)
        a = self._run(pos, vel, t, "local_rms", k=10)
        b = self._run(pos, vel, t, "local_rms", k=10)
        np.testing.assert_array_equal(a["chi2_dof"], b["chi2_dof"])
        self.assertEqual(a["best_observer"], b["best_observer"])

    def test_mask_restricts_observers_and_indices(self):
        pos, vel, t, _g = _homologous_history(n_snap=20, N=20, growth=3.2)
        t_start = 2.9
        t = np.linspace(0.0, 13.8 - t_start, pos.shape[0])
        a_center = center_a_curve(pos, t)
        ph = _fake_pantheon_from_a(a_center, t, t_start)
        mask = np.zeros(20, dtype=bool)
        mask[:8] = True
        out = observer_chi2_distribution(
            pos, vel, t, t_start, ph, definition="local_rms", mask=mask)
        self.assertEqual(out["n_observers"], 8)
        self.assertTrue(np.all(out["observer_index"] < 8))


# ---------------------------------------------------------------------------
# 6b. fraction_at_or_below — the genericity ("how fine-tuned is our vantage")
#     pure helper. We are a RANDOM observer; the fraction of observers at/below
#     a reference says whether a Pantheon-like vantage is generic or fine-tuned.
# ---------------------------------------------------------------------------

class TestFractionAtOrBelow(unittest.TestCase):
    def test_zero_below_min(self):
        chi2 = np.array([0.4, 0.5, 1.0, 2.0])
        self.assertEqual(fraction_at_or_below(chi2, 0.3), 0.0)

    def test_one_at_or_above_max(self):
        chi2 = np.array([0.4, 0.5, 1.0, 2.0])
        self.assertEqual(fraction_at_or_below(chi2, 2.0), 1.0)   # boundary inclusive
        self.assertEqual(fraction_at_or_below(chi2, 5.0), 1.0)

    def test_exact_count_at_boundary(self):
        # 0.436 reference: <= picks 0.40 and 0.436 (inclusive) -> 2/4.
        chi2 = np.array([0.40, 0.436, 0.50, 0.90])
        self.assertAlmostEqual(fraction_at_or_below(chi2, 0.436), 0.5)
        # just below the second value excludes it -> 1/4.
        self.assertAlmostEqual(fraction_at_or_below(chi2, 0.4359), 0.25)

    def test_monotone_nondecreasing_in_threshold(self):
        rng = np.random.default_rng(3)
        chi2 = rng.uniform(0.2, 5.0, size=200)
        thresholds = np.linspace(0.0, 6.0, 40)
        fracs = [fraction_at_or_below(chi2, th) for th in thresholds]
        for a, b in zip(fracs, fracs[1:]):
            self.assertLessEqual(a, b)

    def test_inf_observers_ignored(self):
        # Failed observers (np.inf) are excluded from BOTH numerator and
        # denominator: fraction is over the SCOREABLE observers only.
        chi2 = np.array([0.4, 0.5, np.inf, np.inf])
        self.assertAlmostEqual(fraction_at_or_below(chi2, 0.45), 0.5)  # 1 of 2 finite
        self.assertAlmostEqual(fraction_at_or_below(chi2, 1.0), 1.0)   # 2 of 2 finite

    def test_no_finite_returns_nan(self):
        chi2 = np.array([np.inf, np.inf])
        self.assertTrue(math.isnan(fraction_at_or_below(chi2, 0.5)))

    def test_distribution_emits_fractions_when_refs_given(self):
        # observer_chi2_distribution threads lcdm_ref/eds_ref to the helper and
        # the fractions are consistent with calling the helper directly.
        pos, vel, t = _anisotropic_history(n_snap=30, N=40)
        t_start = 2.9
        t = np.linspace(0.0, 13.8 - t_start, pos.shape[0])
        a_center = center_a_curve(pos, t)
        ph = _fake_pantheon_from_a(a_center, t, t_start)
        out = observer_chi2_distribution(
            pos, vel, t, t_start, ph, definition="local_rms", k=12,
            lcdm_ref=0.436, eds_ref=0.843)
        self.assertEqual(out["lcdm_ref"], 0.436)
        self.assertEqual(out["eds_ref"], 0.843)
        self.assertAlmostEqual(
            out["frac_below_lcdm"],
            fraction_at_or_below(out["chi2_dof"], 0.436))
        self.assertAlmostEqual(
            out["frac_below_eds"],
            fraction_at_or_below(out["chi2_dof"], 0.843))
        # EdS reference is looser than LCDM -> fraction is >= the LCDM fraction.
        self.assertGreaterEqual(out["frac_below_eds"], out["frac_below_lcdm"])

    def test_distribution_fractions_nan_without_refs(self):
        pos, vel, t = _anisotropic_history(n_snap=20, N=30)
        t_start = 2.9
        t = np.linspace(0.0, 13.8 - t_start, pos.shape[0])
        a_center = center_a_curve(pos, t)
        ph = _fake_pantheon_from_a(a_center, t, t_start)
        out = observer_chi2_distribution(
            pos, vel, t, t_start, ph, definition="local_rms", k=10)
        self.assertIsNone(out["lcdm_ref"])
        self.assertTrue(math.isnan(out["frac_below_lcdm"]))
        self.assertTrue(math.isnan(out["frac_below_eds"]))


# ---------------------------------------------------------------------------
# 7. SLOW integration smoke — tiny real sim against real Pantheon+
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestIntegrationSmoke(unittest.TestCase):
    """Run a tiny External-Node sim, extract the snapshot history, and score the
    per-particle observer distribution against the real Pantheon+ data.

    Skipped if the Pantheon+ data file is absent. Checks: a finite distribution
    is produced, the centre baseline is finite, best <= centre, and the centre
    observer reproduces the sim's OWN centre-based a(t) within tolerance."""

    _SKIP_MSG = None

    @classmethod
    def setUpClass(cls):
        import cosmo.pantheon as pl
        try:
            pl.load_pantheon()
        except FileNotFoundError as exc:
            cls._SKIP_MSG = f"Real Pantheon+ data absent: {exc}"

    def _skip_if_no_data(self):
        if self._SKIP_MSG:
            self.skipTest(self._SKIP_MSG)

    def _run_tiny_sim(self):
        from cosmo.constants import SimulationParameters
        from cosmo.factories import (
            run_external_node_simulation, setup_simulation_context,
        )
        t_start = 2.9
        t_dur = 13.8 - t_start
        n_steps = 300
        box, a_start, _ = setup_simulation_context(
            t_start, t_dur, n_steps, save_interval=max(1, n_steps // 10))
        sp = SimulationParameters(
            M_value=855.0, S_value=37.8, n_particles=60, seed=42,
            t_start_Gyr=t_start, t_duration_Gyr=t_dur, n_steps=n_steps,
            damping_factor=None, center_node_mass=1.0, mass_randomize=0.0,
        )
        ext = run_external_node_simulation(
            sp, box, a_start, save_interval=max(1, n_steps // 10))
        return ext, sp

    def test_distribution_runs_and_best_le_center(self):
        self._skip_if_no_data()
        from cosmo.pantheon import load_pantheon
        ext, sp = self._run_tiny_sim()
        sim = ext["sim"]
        mask = np.asarray(sim.particles.get_observable_mask(), dtype=bool)
        pos, vel, t = history_from_snapshots(sim.snapshots)
        ph = load_pantheon()

        for definition in ("local_rms", "hubble_flow"):
            out = observer_chi2_distribution(
                pos, vel, t, sp.t_start_Gyr, ph,
                definition=definition, mask=mask)
            self.assertGreater(out["n_finite"], 0,
                               f"{definition}: no finite observers")
            self.assertTrue(np.isfinite(out["center_chi2_dof"]),
                            f"{definition}: centre chi2 not finite")
            self.assertLessEqual(out["best_chi2_dof"],
                                 out["center_chi2_dof"] + 1e-9,
                                 f"{definition}: best > centre")

    def test_center_observer_reproduces_sim_center_a_curve(self):
        # The prototype's centre_a_curve (from snapshots) must match the sim's
        # own inner-region a(t) (factories ext['a']) within tolerance.
        self._skip_if_no_data()
        ext, sp = self._run_tiny_sim()
        sim = ext["sim"]
        mask = np.asarray(sim.particles.get_observable_mask(), dtype=bool)
        pos, vel, t = history_from_snapshots(sim.snapshots)
        a_proto = center_a_curve(pos, t, mask=mask)
        a_sim = np.asarray(ext["a"], dtype=float)
        # Same snapshot grid, same measure -> should match to ~1e-9.
        np.testing.assert_allclose(a_proto, a_sim, rtol=1e-6, atol=1e-6)
