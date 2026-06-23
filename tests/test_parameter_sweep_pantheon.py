"""
Unit tests for the from-data Pantheon+ chi^2 scoring mode (Stage 3).

All tests are hermetic: they use synthetic a(t) curves and the small synthetic
Pantheon+ fixture at tests/fixtures/pantheon_synthetic.dat. No real N-body
simulation is run.
"""
import unittest
import pathlib
import numpy as np

from cosmo.parameter_sweep import (
    SearchMethod, SweepConfig, MatchWeights, SimResult, SimSimpleResult, LCDMBaseline,
    compute_pantheon_metrics, compute_match_metrics, run_sweep, SKIP_CACHE,
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
    """

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


if __name__ == '__main__':
    unittest.main()
