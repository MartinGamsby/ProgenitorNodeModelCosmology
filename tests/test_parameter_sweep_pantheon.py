"""
Unit tests for the from-data Pantheon+ chi^2 scoring mode (Stage 3).

All tests are hermetic: they use synthetic a(t) curves and the small synthetic
Pantheon+ fixture at tests/fixtures/pantheon_synthetic.dat. No real N-body
simulation is run.
"""
import unittest
import pathlib
import numpy as np

import cosmo.parameter_sweep as ps
from cosmo.parameter_sweep import (
    SearchMethod, SweepConfig, MatchWeights, SimResult, SimSimpleResult, LCDMBaseline,
    compute_pantheon_metrics, compute_match_metrics, run_sweep, worst_callback, SKIP_CACHE,
)
from cosmo.pantheon import load_pantheon
from cosmo.analysis import solve_friedmann_at_times

# Path to the tiny synthetic fixture (shipped with the repo)
_FIXTURE_PATH = pathlib.Path(__file__).resolve().parent / "fixtures" / "pantheon_synthetic.dat"

# Standard sim timing for these tests: t_start=5.8, t_duration=8.0 -> t_end=13.8
_T_START = 5.8
_T_DURATION = 8.0
_N_POINTS = 81  # Enough resolution for the integral


def _make_lcdm_a_curve(n_points: int = _N_POINTS, t_start: float = _T_START,
                        t_duration: float = _T_DURATION) -> tuple:
    """
    Build an analytic LCDM a(t) on [t_start, t_start+t_duration] grid.

    Returns (a_curve, t_Gyr) where t_Gyr is RELATIVE (starts at 0.0) and
    a_curve is normalized so a[0]=1 (as the N-body outputs it).
    """
    t_abs = np.linspace(t_start, t_start + t_duration, n_points)
    sol = solve_friedmann_at_times(t_abs)
    a_abs = sol['a']
    # Normalize so a[0] == 1 (simulation frame convention)
    a_curve = a_abs / a_abs[0]
    t_Gyr = t_abs - t_start   # relative, starts at 0.0
    return a_curve, t_Gyr


def _make_sim_result_with_a_curve(n_points: int = _N_POINTS) -> SimResult:
    """Build a SimResult whose a_curve is analytic LCDM (covers z ~ 0..2.2)."""
    a_curve, t_Gyr = _make_lcdm_a_curve(n_points)
    n = len(a_curve)
    return SimResult(
        size_curve_Gpc=np.linspace(10.0, 14.5, n),
        hubble_curve=np.linspace(75.0, 68.0, n),
        t_Gyr=t_Gyr,
        params=None,
        results=SimSimpleResult(
            size_final_Gpc=14.5,
            radius_max_Gpc=9.4,
            a_final=float(a_curve[-1]),
        ),
        a_curve=a_curve,
    )


def _load_synthetic_pantheon():
    return load_pantheon(path=_FIXTURE_PATH, z_min=0.01, exclude_calibrators=True)


class TestSimResultACurveOptional(unittest.TestCase):
    """SimResult can be constructed without a_curve (backwards-compatible)."""

    def test_default_a_curve_is_none(self):
        result = SimResult(
            size_curve_Gpc=np.linspace(10, 14, 5),
            hubble_curve=np.linspace(75, 68, 5),
            t_Gyr=np.linspace(0, 8, 5),
            params=None,
            results=SimSimpleResult(14.0, 9.0, 1.0),
        )
        self.assertIsNone(result.a_curve)

    def test_explicit_a_curve_stored(self):
        a = np.linspace(1.0, 2.0, 10)
        result = SimResult(
            size_curve_Gpc=np.zeros(10),
            hubble_curve=np.zeros(10),
            t_Gyr=np.zeros(10),
            params=None,
            results=SimSimpleResult(0.0, 0.0, 0.0),
            a_curve=a,
        )
        np.testing.assert_array_equal(result.a_curve, a)

    def test_results_to_sim_result_populates_a_curve(self):
        """results_to_sim_result must populate a_curve from the ext_results dict."""
        from cosmo.factories import results_to_sim_result
        from cosmo.constants import SimulationParameters

        # Build a minimal ext_results dict (no real sim needed)
        n = 11
        a = np.linspace(1.0, 1.5, n)
        ext_results = {
            'a': a,
            't_Gyr': np.linspace(0, 8, n),
            'diameter_Gpc': np.linspace(10, 14, n),
            'max_radius_Gpc': np.linspace(5, 7, n),
            'H_hubble': np.linspace(75, 68, n),
        }
        sim_params = SimulationParameters(
            M_value=1, S_value=1,
            t_start_Gyr=5.8, t_duration_Gyr=8.0,
            n_steps=100,
        )
        result = results_to_sim_result(ext_results, sim_params)
        self.assertIsNotNone(result.a_curve)
        np.testing.assert_array_equal(result.a_curve, a)


class TestPantheonScorerFinite(unittest.TestCase):
    """Scorer returns finite chi2/R2 for a valid sim against the synthetic fixture."""

    def setUp(self):
        self.pantheon = _load_synthetic_pantheon()
        self.sim_result = _make_sim_result_with_a_curve()

    def test_returns_finite_chi2(self):
        metrics = compute_pantheon_metrics(self.sim_result, self.pantheon, _T_START)
        self.assertTrue(np.isfinite(metrics['chi2']), f"chi2 not finite: {metrics['chi2']}")
        self.assertTrue(np.isfinite(metrics['chi2_dof']), f"chi2_dof not finite: {metrics['chi2_dof']}")

    def test_returns_finite_R2(self):
        metrics = compute_pantheon_metrics(self.sim_result, self.pantheon, _T_START)
        self.assertTrue(np.isfinite(metrics['R2']), f"R2 not finite: {metrics['R2']}")

    def test_match_avg_pct_in_range(self):
        metrics = compute_pantheon_metrics(self.sim_result, self.pantheon, _T_START)
        self.assertGreaterEqual(metrics['match_avg_pct'], 0.0)
        self.assertLessEqual(metrics['match_avg_pct'], 100.0)

    def test_n_sne_used_positive(self):
        metrics = compute_pantheon_metrics(self.sim_result, self.pantheon, _T_START)
        self.assertGreater(metrics['n_sne_used'], 0)

    def test_diff_pct_complement(self):
        metrics = compute_pantheon_metrics(self.sim_result, self.pantheon, _T_START)
        self.assertAlmostEqual(metrics['match_avg_pct'] + metrics['diff_pct'], 100.0, places=8)

    def test_match_metric_keys_present(self):
        """Scorer must include all MATCH_METRIC_KEYS (zeroed) for CSV compatibility."""
        from cosmo.parameter_sweep import MATCH_METRIC_KEYS
        metrics = compute_pantheon_metrics(self.sim_result, self.pantheon, _T_START)
        for k in MATCH_METRIC_KEYS:
            self.assertIn(k, metrics, f"Missing MATCH_METRIC_KEY: {k}")


class TestPantheonScorerWorstCaseOnEmpty(unittest.TestCase):
    """Scorer returns worst-case score when no SNe are in range (no exception)."""

    def test_no_overlap_returns_worst_case(self):
        """a_curve with near-zero z coverage -> worst-case score, no exception.

        Strategy: use an a_curve that changes by only 0.1% from start to end,
        so after renormalization to today (a[-1]=1), all earlier a values are
        ~0.999 and z_snap = 1/0.999 - 1 ~ 0.001. The Pantheon fixture has
        z_min=0.01 (after the z>=0.01 cut), so all z_snap < z_min_data -> no
        in-range SNe -> worst-case.
        """
        n = 81
        # Very slow expansion: a goes from 1.0 to 1.001 over the whole sim
        a_curve = np.linspace(1.0, 1.001, n)
        t_Gyr = np.linspace(0.0, _T_DURATION, n)

        sim_result = SimResult(
            size_curve_Gpc=np.zeros(n),
            hubble_curve=np.zeros(n),
            t_Gyr=t_Gyr,
            params=None,
            results=SimSimpleResult(0.0, 0.0, 0.0),
            a_curve=a_curve,
        )
        pantheon = _load_synthetic_pantheon()
        # Should not raise; must return worst-case (0 SNe in range)
        metrics = compute_pantheon_metrics(sim_result, pantheon, _T_START)
        self.assertEqual(metrics['match_avg_pct'], 0.0)
        self.assertEqual(metrics['n_sne_used'], 0)

    def test_none_a_curve_returns_worst_case(self):
        """SimResult with a_curve=None -> worst-case score, no exception."""
        sim_result = SimResult(
            size_curve_Gpc=np.zeros(5),
            hubble_curve=np.zeros(5),
            t_Gyr=np.zeros(5),
            params=None,
            results=SimSimpleResult(0.0, 0.0, 0.0),
            a_curve=None,
        )
        pantheon = _load_synthetic_pantheon()
        metrics = compute_pantheon_metrics(sim_result, pantheon, _T_START)
        self.assertEqual(metrics['match_avg_pct'], 0.0)
        self.assertEqual(metrics['n_sne_used'], 0)


class TestLCDMObjectiveUnchanged(unittest.TestCase):
    """
    A tiny LCDM-objective sweep behaves exactly as before (additive, no regressions).
    The test replicates the dummy-callback pattern from test_parameter_sweep.py.

    Cache is disabled at the MODULE level (ps.SKIP_CACHE) so the dummy-callback
    metrics are never served from a warm real cache left by an earlier (real-sim)
    test in a full cross-file run. Without this the test flakes depending on test
    order (the shared data/*.csv cache leaks across files).
    """

    def setUp(self):
        self._saved_skip = ps.SKIP_CACHE
        ps.SKIP_CACHE = True

    def tearDown(self):
        ps.SKIP_CACHE = self._saved_skip

    def _make_baseline(self, n=31):
        return LCDMBaseline(
            t_Gyr=np.linspace(5.8, 13.8, n),
            size_Gpc=np.linspace(10.0, 14.5, n),
            H_hubble=np.linspace(75.0, 68.0, n),
            size_final_Gpc=14.5,
            radius_max_Gpc=9.4,
            a_final=1.0,
        )

    def _make_unimodal_callback(self, optimal_S: int, optimal_M: int = 500):
        def callback(M, S, centerM, seeds):
            distance = abs(S - optimal_S) / 20.0 + abs(M - optimal_M) / 2000.0
            quality = 0.5 + 0.5 * np.exp(-distance)
            n = 31
            offset = -(1.0 - quality) * 0.5
            return [
                SimResult(
                    size_curve_Gpc=np.linspace(10 + offset, 14.5 + offset, n),
                    hubble_curve=np.linspace(75 + offset * 2, 68 + offset * 2, n),
                    t_Gyr=np.linspace(5.8, 13.8, n),
                    params=None,
                    results=SimSimpleResult(14.5 + offset, 9.4 + offset, 1.0),
                )
                for _ in seeds
            ]
        return callback

    def test_lcdm_objective_selects_unimodal_optimum(self):
        optimal_S = 30
        callback = self._make_unimodal_callback(optimal_S=optimal_S, optimal_M=500)
        baseline = self._make_baseline()

        config = SweepConfig(
            search_center_mass=False,
            s_min_gpc=20,
            s_max_gpc=40,
            objective="lcdm",   # explicit; must be default behaviour
        )

        results = run_sweep(
            config, SearchMethod.BRUTE_FORCE, callback, baseline,
            seeds=[42],
            pantheon_data=None,  # lcdm path must work without pantheon_data
        )

        self.assertGreater(len(results), 0)
        best = max(results, key=lambda r: r['match_avg_pct'])
        # The unimodal peak is at S=30; allow ±5 tolerance for coarse M grid
        self.assertLessEqual(abs(best['S_gpc'] - optimal_S), 5,
                             f"Best S={best['S_gpc']} not near optimal {optimal_S}")


class TestPantheonObjectiveSweepEndToEnd(unittest.TestCase):
    """
    Run a tiny BRUTE_FORCE sweep with objective="pantheon" end-to-end, using a
    dummy callback that returns SimResults carrying an analytic-LCDM a_curve.

    This exercises the from-data path that compute_pantheon_metrics alone does
    NOT cover: the objective branch in worst_callback, and pantheon_data being
    threaded through run_sweep -> brute_force_search -> worst_callback.

    Cache is disabled at the MODULE level (ps.SKIP_CACHE) so the test is fully
    hermetic and never touches data/*.csv. (Reassigning the imported SKIP_CACHE
    name would NOT disable the cache — worst_callback reads the module global.)
    """

    def setUp(self):
        self._saved_skip = ps.SKIP_CACHE
        ps.SKIP_CACHE = True
        self.pantheon = _load_synthetic_pantheon()

    def tearDown(self):
        ps.SKIP_CACHE = self._saved_skip

    def _a_curve_callback(self):
        """Callback returning SimResults whose a_curve is analytic LCDM."""
        a_curve, t_Gyr = _make_lcdm_a_curve()

        def callback(M, S, centerM, seeds):
            n = len(a_curve)
            return [
                SimResult(
                    size_curve_Gpc=np.linspace(10.0, 14.5, n),
                    hubble_curve=np.linspace(75.0, 68.0, n),
                    t_Gyr=t_Gyr,
                    params=None,
                    results=SimSimpleResult(14.5, 9.4, float(a_curve[-1])),
                    a_curve=a_curve,
                )
                for _ in seeds
            ]
        return callback

    def test_pantheon_sweep_returns_finite_chi2_results(self):
        """A pantheon-objective sweep produces results with finite chi2_dof."""
        config = SweepConfig(
            search_center_mass=False,
            s_min_gpc=20,
            s_max_gpc=30,
            objective="pantheon",
        )
        results = run_sweep(
            config, SearchMethod.BRUTE_FORCE,
            self._a_curve_callback(),
            baseline=None,                 # pantheon objective needs no LCDM baseline
            seeds=[42],
            pantheon_data=self.pantheon,
        )
        self.assertGreater(len(results), 0)
        # Every config scored with the pantheon scorer -> chi2_dof present & finite
        for r in results:
            self.assertIn('chi2_dof', r)
            self.assertTrue(np.isfinite(r['chi2_dof']),
                            f"chi2_dof not finite for {r.get('desc')}: {r['chi2_dof']}")
            self.assertGreater(r['n_sne_used'], 0)

    def test_pantheon_sweep_baseline_none_does_not_crash(self):
        """objective='pantheon' must work with baseline=None (no LCDM ref needed)."""
        config = SweepConfig(
            search_center_mass=False,
            s_min_gpc=20, s_max_gpc=25,
            objective="pantheon",
        )
        # Should not raise even though baseline is None
        results = run_sweep(
            config, SearchMethod.TERNARY_SEARCH,
            self._a_curve_callback(),
            baseline=None,
            seeds=[42],
            pantheon_data=self.pantheon,
        )
        self.assertGreater(len(results), 0)


class TestGrowthAnchor(unittest.TestCase):
    """
    The physical expansion anchor rejects configs whose total expansion
    a(today)/a(t_start) is not the real ~1+z(t_start), even if their
    renormalized shape could fit the SN window.
    """

    def setUp(self):
        self.pantheon = _load_synthetic_pantheon()

    def test_expected_growth_factor_is_physical(self):
        from cosmo.parameter_sweep import expected_growth_factor
        g58 = expected_growth_factor(5.8)
        g29 = expected_growth_factor(2.9)
        # Universe expands from t_start to today: growth > 1, and starting earlier
        # (2.9 Gyr) means MORE total expansion than starting at 5.8 Gyr.
        self.assertGreater(g58, 1.0)
        self.assertGreater(g29, g58)

    def test_runaway_growth_rejected(self):
        """An a_curve that over-expands (growth >> physical) scores worst-case."""
        from cosmo.parameter_sweep import expected_growth_factor
        n = 81
        # Growth of 100x over [5.8,13.8] is wildly above the physical ~1.7x.
        a_curve = np.linspace(1.0, 100.0, n)
        sim_result = SimResult(
            size_curve_Gpc=np.zeros(n), hubble_curve=np.zeros(n),
            t_Gyr=np.linspace(0.0, _T_DURATION, n),
            params=None, results=SimSimpleResult(0.0, 0.0, 100.0),
            a_curve=a_curve,
        )
        metrics = compute_pantheon_metrics(sim_result, self.pantheon, _T_START)
        self.assertEqual(metrics['match_avg_pct'], 0.0)
        self.assertEqual(metrics['n_sne_used'], 0)
        # Diagnostic fields are surfaced on rejection
        self.assertAlmostEqual(metrics['growth_factor'], 100.0, places=3)
        self.assertAlmostEqual(metrics['growth_target'],
                               expected_growth_factor(_T_START), places=6)

    def test_physical_growth_accepted(self):
        """The analytic-LCDM a_curve (correct growth) is NOT rejected."""
        sim_result = _make_sim_result_with_a_curve()
        metrics = compute_pantheon_metrics(sim_result, self.pantheon, _T_START)
        self.assertGreater(metrics['n_sne_used'], 0)
        self.assertTrue(np.isfinite(metrics['chi2_dof']))


class TestCacheKeyObjectiveIsolation(unittest.TestCase):
    """
    The cache key built in worst_callback MUST include the objective so that
    lcdm and pantheon scores for the SAME (M, S, centerM, seeds, timing) config
    never collide in the shared cache file.

    This guards the exact regression the cache-key change could have introduced:
    a pantheon score being served from an lcdm cache entry (or vice versa).

    Strategy: monkeypatch cosmo.parameter_sweep.Cache with a spy that records
    every key passed to add_cached_value, run one lcdm and one pantheon
    evaluation of an identical config, and assert the recorded key sets are
    disjoint and carry the expected '<objective>obj' suffix.
    """

    def setUp(self):
        self._saved_skip = ps.SKIP_CACHE
        self._saved_cache_cls = ps.Cache
        self._saved_cache_singleton = ps.CACHE
        ps.SKIP_CACHE = False          # we WANT the cache path to run (spied)
        ps.CACHE = None
        self.pantheon = _load_synthetic_pantheon()

    def tearDown(self):
        ps.SKIP_CACHE = self._saved_skip
        ps.Cache = self._saved_cache_cls
        ps.CACHE = self._saved_cache_singleton

    def _install_spy_cache(self):
        recorded_keys = []

        class _SpyCache:
            def __init__(self, name, *a, **k):
                self.name = name

            def get_cached_value(self, key, data_type):
                return None  # always a miss so the scorer runs

            def add_cached_value(self, key, data_type, value, save_interval_s=5):
                recorded_keys.append(key)

        ps.Cache = _SpyCache
        ps.CACHE = None
        return recorded_keys

    def _eval_once(self, objective):
        """Run worst_callback once for a fixed config under the given objective."""
        a_curve, t_Gyr = _make_lcdm_a_curve()
        n = len(a_curve)

        def callback(M, S, centerM, seeds):
            return [
                SimResult(
                    size_curve_Gpc=np.linspace(10.0, 14.5, n),
                    hubble_curve=np.linspace(75.0, 68.0, n),
                    t_Gyr=t_Gyr,
                    params=None,
                    results=SimSimpleResult(14.5, 9.4, float(a_curve[-1])),
                    a_curve=a_curve,
                )
                for _ in seeds
            ]

        config = SweepConfig(objective=objective)
        baseline = LCDMBaseline(
            t_Gyr=np.linspace(5.8, 13.8, n),
            size_Gpc=np.linspace(10.0, 14.5, n),
            H_hubble=np.linspace(75.0, 68.0, n),
            size_final_Gpc=14.5, radius_max_Gpc=9.4, a_final=1.0,
        )
        ps.CACHE = None  # force a fresh spy cache instance
        worst_callback(
            callback, config, M_factor=500, S_val=25, centerM=1,
            seeds=[42], baseline=baseline, weights=MatchWeights(),
            pantheon_data=self.pantheon,
        )

    def test_lcdm_and_pantheon_keys_are_disjoint(self):
        recorded = self._install_spy_cache()

        self._eval_once("lcdm")
        lcdm_keys = set(recorded)

        recorded.clear()
        self._eval_once("pantheon")
        pantheon_keys = set(recorded)

        self.assertTrue(lcdm_keys, "lcdm path recorded no cache keys")
        self.assertTrue(pantheon_keys, "pantheon path recorded no cache keys")

        # The objective suffix must make the key sets disjoint.
        self.assertEqual(
            lcdm_keys & pantheon_keys, set(),
            f"lcdm and pantheon cache keys collide: {lcdm_keys & pantheon_keys}",
        )
        # Every key must carry its objective slug (now followed by the trailing
        # physics-version token, so it is no longer the final part).
        self.assertTrue(all("lcdmobj" in k for k in lcdm_keys),
                        f"lcdm keys missing 'lcdmobj' slug: {lcdm_keys}")
        self.assertTrue(all("pantheonobj" in k for k in pantheon_keys),
                        f"pantheon keys missing 'pantheonobj' slug: {pantheon_keys}")
        # And every key must end with the physics-version token (defense-in-depth:
        # the token is appended last and applies to both objectives).
        from cosmo.parameter_sweep import PHYSICS_CACHE_VERSION
        tok = f"phys{PHYSICS_CACHE_VERSION}"
        self.assertTrue(all(k.endswith(tok) for k in lcdm_keys | pantheon_keys),
                        f"keys missing trailing physics token {tok!r}: "
                        f"{lcdm_keys | pantheon_keys}")


class TestPhysicsCacheVersionToken(unittest.TestCase):
    """The cache key MUST embed a physics-version token so that a change to the
    simulation physics (e.g. EdS-consistent ICs, the pre-start tidal boost)
    invalidates every pre-change entry instead of silently reusing it.

    Guards the exact regression the user flagged: a stale parameter-only cache
    entry computed under old physics being served for matching parameters after
    the physics changed.
    """

    def _key(self, **cfg_kwargs):
        from cosmo.parameter_sweep import build_cache_name
        cfg = SweepConfig(**cfg_kwargs)
        return build_cache_name(cfg, M_factor=500, S_val=25, centerM=1, seeds=[42])

    def test_token_is_part_of_every_key(self):
        from cosmo.parameter_sweep import PHYSICS_CACHE_VERSION
        tok = f"phys{PHYSICS_CACHE_VERSION}"
        for obj in ("lcdm", "pantheon"):
            key = self._key(objective=obj)
            self.assertIn(tok, key, f"physics token {tok!r} missing from {key!r}")
            self.assertTrue(key.endswith(tok),
                            f"physics token must be the LAST part of {key!r}")

    def test_different_version_does_not_collide_with_old_key(self):
        """Bumping PHYSICS_CACHE_VERSION must produce a key disjoint from the old
        one for the SAME parameters (so old entries become unreachable)."""
        import cosmo.parameter_sweep as _ps
        saved = _ps.PHYSICS_CACHE_VERSION
        try:
            _ps.PHYSICS_CACHE_VERSION = "v2"
            key_v2 = self._key(objective="pantheon")
            _ps.PHYSICS_CACHE_VERSION = "v3"
            key_v3 = self._key(objective="pantheon")
        finally:
            _ps.PHYSICS_CACHE_VERSION = saved
        self.assertNotEqual(key_v2, key_v3,
                            "different physics versions produced the SAME key")
        # Only the trailing token differs; everything before it is identical.
        self.assertEqual(key_v2.rsplit("_", 1)[0], key_v3.rsplit("_", 1)[0])

    def test_legacy_physics_flags_give_distinct_key(self):
        """A legacy (eds_consistent=False) or boost-off config must NOT share a
        key with the current-physics defaults at the same parameters."""
        default_key = self._key(objective="pantheon")
        no_eds_key = self._key(objective="pantheon", eds_consistent=False)
        no_boost_key = self._key(objective="pantheon", pre_start_tidal_boost=False)
        self.assertNotEqual(default_key, no_eds_key)
        self.assertNotEqual(default_key, no_boost_key)
        self.assertNotEqual(no_eds_key, no_boost_key)

    def test_token_round_trips_through_real_cache(self):
        """The token must survive Cache._split_key/_join_key (CSV column round
        trip) so a cached key reads back byte-identically."""
        from cosmo.cache import Cache
        for obj in ("lcdm", "pantheon"):
            key = self._key(objective=obj)
            cols = Cache._split_key(key)
            rejoined = Cache._join_key(cols)
            self.assertEqual(key, rejoined,
                             f"key did not round-trip: {key!r} -> {rejoined!r}")


if __name__ == '__main__':
    unittest.main()
