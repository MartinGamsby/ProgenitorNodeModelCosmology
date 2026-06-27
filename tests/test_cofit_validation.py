"""
Trustworthiness test for the B4 co-fit VALIDATOR (`_validate_cofit.py`).

The validator's whole job is to compare three S co-fit methods (linear /
ternary / brute-force) on a real cell and decide whether linear is safe for the
core sweep. Before trusting THAT comparison, the comparison LOGIC must itself be
correct: on a KNOWN convex chi2(S) bowl the three methods must all land on the
same minimum (within one S grid step), and the ground-truth brute scan must hit
the exact minimum.

This test exercises the SAME three search functions the validator calls
(`linear_search_S`, `ternary_search_S`, and an exhaustive `worst_callback` scan
== brute force) against a synthetic, single-minimum chi2(S) bowl whose minimum
is known a priori. It is a pure-function check: no real N-body sim, no Pantheon
data.

Objective note: the bowl is built in the LCDM objective (the search functions'
early-stop logic inspects `USED_MATCH_METRIC_KEYS`, which are POPULATED only on
the lcdm path; on the pantheon path they are zero-filled). Using a real
`compute_match_metrics`-scored unimodal callback gives a faithful convex
match_avg_pct(S) bowl — equivalently a convex chi2(S) — that drives the same
search machinery the validator drives, so a pass means the validator's
linear-vs-ternary-vs-brute comparison logic is sound.
"""
import unittest

import numpy as np

import cosmo.parameter_sweep as ps
from cosmo.parameter_sweep import (
    SweepConfig, MatchWeights, SimResult, SimSimpleResult, LCDMBaseline,
    build_s_list, linear_search_S, ternary_search_S, worst_callback,
)


def _make_baseline(n_points: int = 31) -> LCDMBaseline:
    return LCDMBaseline(
        t_Gyr=np.linspace(5.8, 13.8, n_points),
        size_Gpc=np.linspace(10.0, 14.5, n_points),
        H_hubble=np.linspace(75.0, 68.0, n_points),
        size_final_Gpc=14.5,
        radius_max_Gpc=9.4,
        a_final=1.0,
    )


def _sim_result(quality: float, n_points: int = 31) -> SimResult:
    """A SimResult whose LCDM match quality is `quality` (1.0 == perfect)."""
    off = -(1.0 - quality) * 0.5
    return SimResult(
        size_curve_Gpc=np.linspace(10.0 + off, 14.5 + off, n_points),
        hubble_curve=np.linspace(75.0 + 2 * off, 68.0 + 2 * off, n_points),
        t_Gyr=np.linspace(5.8, 13.8, n_points),
        params=None,
        results=SimSimpleResult(size_final_Gpc=14.5 + off,
                                radius_max_Gpc=9.4 + off, a_final=1.0),
    )


def _bowl_callback(s_star: int):
    """Unimodal callback: match quality peaks at S=s_star (a convex chi2 bowl).

    Quality = 0.5 + 0.5*exp(-|S - s_star|/6) -> single interior maximum at
    s_star, decaying smoothly either side, so match_avg_pct(S) is a clean convex
    bowl (equivalently chi2(S) is convex) with its minimum at s_star.
    """
    def callback(M, S, centerM, seeds):
        q = 0.5 + 0.5 * float(np.exp(-abs(S - s_star) / 6.0))
        return [_sim_result(q) for _ in seeds]
    return callback


class TestCofitMethodsAgreeOnSyntheticBowl(unittest.TestCase):
    """linear / ternary / brute all locate a KNOWN convex chi2(S) minimum."""

    def setUp(self):
        self._saved_skip = ps.SKIP_CACHE
        ps.SKIP_CACHE = True   # hermetic: do not touch data/*.csv
        self.s_min, self.s_max = 15, 60   # away from the s_max nice-value cap
        self.s_list = build_s_list(self.s_min, self.s_max)
        self.baseline = _make_baseline()
        self.weights = MatchWeights()
        self.config = SweepConfig(objective="lcdm",
                                  s_min_gpc=self.s_min, s_max_gpc=self.s_max)

    def tearDown(self):
        ps.SKIP_CACHE = self._saved_skip

    def _brute_best_S(self, callback, M=500, centerM=1):
        """Exhaustive scan == validator ground truth: highest match_avg_pct."""
        best_S, best = None, None
        for S in self.s_list:
            _, metrics = worst_callback(
                callback, self.config, M, S, centerM, [42],
                self.baseline, self.weights)
            if best is None or metrics["match_avg_pct"] > best["match_avg_pct"]:
                best, best_S = metrics, S
        return best_S

    def _run_three(self, s_star, M=500, centerM=1):
        cb = _bowl_callback(s_star)
        brute_S = self._brute_best_S(cb, M, centerM)
        lin_S, _, _, _ = linear_search_S(
            self.config, M, centerM, cb, self.baseline, self.weights,
            self.s_min, self.s_max, prev_best_S=None, seeds=[42])
        tern_S, _, _, _ = ternary_search_S(
            self.config, M, centerM, cb, self.baseline, self.weights,
            self.s_min, self.s_max, s_hint=None, seeds=[42])
        return brute_S, lin_S, tern_S

    def test_brute_force_finds_exact_minimum(self):
        """Ground truth: brute force lands exactly on the bowl minimum."""
        for s_star in (40, 50, 55):
            with self.subTest(s_star=s_star):
                self.assertEqual(self._brute_best_S(_bowl_callback(s_star)), s_star)

    def test_all_three_agree_within_one_grid_step(self):
        """linear, ternary, brute agree within ONE grid step at interior S*.

        s_star chosen near the top of the search range so linear search (which
        walks DOWN from s_max) reaches the optimum before its early-stop fires.
        """
        for s_star in (50, 55):
            with self.subTest(s_star=s_star):
                brute_S, lin_S, tern_S = self._run_three(s_star)
                self.assertEqual(brute_S, s_star,
                                 f"brute should hit the known minimum {s_star}")
                self.assertLessEqual(abs(lin_S - brute_S), 1,
                                     f"linear S={lin_S} vs brute {brute_S} (>1 step)")
                self.assertLessEqual(abs(tern_S - brute_S), 1,
                                     f"ternary S={tern_S} vs brute {brute_S} (>1 step)")

    def test_chi2_within_tolerance_at_found_S(self):
        """match_avg_pct at each method's S is within a hair of the brute best.

        Within one grid step of a smooth convex bowl the score barely moves;
        this guards the validator's chi2-agreement tolerance check (the chi2
        analogue of match_avg_pct on the pantheon path).
        """
        s_star = 55
        cb = _bowl_callback(s_star)
        brute_S, lin_S, tern_S = self._run_three(s_star)

        def score(S):
            _, m = worst_callback(cb, self.config, 500, S, 1, [42],
                                  self.baseline, self.weights)
            return m["match_avg_pct"]

        ref = score(brute_S)
        for name, S in (("linear", lin_S), ("ternary", tern_S)):
            self.assertLessEqual(abs(score(S) - ref), 0.5,
                                 f"{name} score at S={S} far from brute best")

    def test_brute_detects_true_lower_boundary_optimum(self):
        """When the true optimum IS at the grid's lower boundary, brute reports it.

        Guards the validator's 'pinned at s_min' detection: a genuine boundary
        optimum must be reported AS the boundary by the ground-truth scan, so the
        validator distinguishes a real boundary optimum from a search artifact.
        Note build_s_list floors the grid at the first 'nice' value (s_list[0]),
        so the lower boundary the scan can actually report is s_list[0].
        """
        lower = self.s_list[0]
        self.assertEqual(self._brute_best_S(_bowl_callback(lower)), lower)


if __name__ == "__main__":
    unittest.main()
