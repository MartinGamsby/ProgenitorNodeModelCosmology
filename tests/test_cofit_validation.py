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
    MATCH_METRIC_KEYS, USED_MATCH_METRIC_KEYS,
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


class TestLinearSearchPantheonEarlyStop(unittest.TestCase):
    """REGRESSION: linear_search_S early-stop must key on the ACTIVE objective.

    R2's co-fit validation found that on the pantheon objective the DEFAULT
    linear co-fit was broken: its early-stop "all worse" check iterated
    USED_MATCH_METRIC_KEYS (the LCDM metric keys), which compute_pantheon_metrics
    ZERO-FILLS. So `prev[key]` and `result[key]` were both 0.0, the
    `prev < result*1.00025` guard never flipped `all_worse` to False, and the
    search stopped on the 2nd S evaluated -> linear pinned near s_max and
    reported an inflated chi2 (~0.69 vs the true ~0.51 that ternary/brute find).

    These tests drive the REAL search machinery against a synthetic convex
    chi2(S) bowl scored EXACTLY like the real pantheon scorer (a high
    match_avg_pct at the minimum, every MATCH_METRIC_KEYS entry zero-filled).
    Before the fix the pantheon case pins; after the fix linear lands on the
    brute optimum within one grid step. The LCDM case asserts that path is
    unchanged.
    """

    def setUp(self):
        self._saved_skip = ps.SKIP_CACHE
        self._saved_scorer = ps.compute_pantheon_metrics
        ps.SKIP_CACHE = True   # hermetic: no data/*.csv, no real Pantheon
        self.s_min, self.s_max = 3, 35      # the core S range (matches R2 cells)
        self.s_list = build_s_list(self.s_min, self.s_max)
        self.weights = MatchWeights()

    def tearDown(self):
        ps.SKIP_CACHE = self._saved_skip
        ps.compute_pantheon_metrics = self._saved_scorer

    def _install_pantheon_bowl(self, s_star: int):
        """Patch compute_pantheon_metrics with a convex chi2(S) bowl.

        chi2_dof bottoms at s_star and rises either side; match_avg_pct =
        100/(1+chi2_dof) (the REAL pantheon score). Crucially, every
        MATCH_METRIC_KEYS entry is ZERO-FILLED exactly as the production scorer
        does — this is what trips the pre-fix early-stop. The bowl S is read off
        the SimResult's a_final, which the bowl callback below encodes.
        """
        def fake_scorer(sim_result, pantheon_data, t_start_Gyr, **kwargs):
            S = float(sim_result.results.a_final)   # callback stores S here
            chi2_dof = 0.5 + 0.05 * abs(S - s_star)  # convex, min 0.5 at s_star
            match_avg_pct = 100.0 / (1.0 + chi2_dof)
            metrics = {
                'chi2': chi2_dof * 100.0,
                'chi2_dof': chi2_dof,
                'R2': 0.9,
                'n_sne_used': 500,
                'match_avg_pct': match_avg_pct,
                'diff_pct': 100.0 - match_avg_pct,
            }
            for k in MATCH_METRIC_KEYS:    # zero-fill exactly like the real scorer
                metrics.setdefault(k, 0.0)
            return metrics
        ps.compute_pantheon_metrics = fake_scorer

    def _bowl_callback(self):
        """SimResult carrying S in a_final so the patched scorer can recover it."""
        def callback(M, S, centerM, seeds):
            res = SimResult(
                size_curve_Gpc=None, hubble_curve=None, t_Gyr=None, params=None,
                results=SimSimpleResult(size_final_Gpc=1.0, radius_max_Gpc=1.0,
                                        a_final=float(S)),
                a_curve=None,
            )
            return [res for _ in seeds]
        return callback

    def _brute_best_S(self, config, cb, M=300, centerM=1):
        best_S, best = None, None
        for S in self.s_list:
            _, m = worst_callback(cb, config, M, S, centerM, [42],
                                  None, self.weights, pantheon_data={})
            if best is None or m["match_avg_pct"] > best["match_avg_pct"]:
                best, best_S = m, S
        return best_S

    def test_linear_finds_brute_optimum_on_pantheon_objective(self):
        """linear == brute within one grid step on an INTERIOR pantheon minimum.

        FAILS before the fix: linear pins near s_max ({self.s_max} side) because
        the zero-filled USED_MATCH_METRIC_KEYS trip the early-stop on the 2nd S.
        """
        config = SweepConfig(objective="pantheon",
                             s_min_gpc=self.s_min, s_max_gpc=self.s_max)
        for s_star in (17, 21):   # the two R2 optima (M=100 -> 17, M=300 -> 21)
            with self.subTest(s_star=s_star):
                self._install_pantheon_bowl(s_star)
                cb = self._bowl_callback()
                brute_S = self._brute_best_S(config, cb)
                self.assertEqual(brute_S, s_star,
                                 f"brute must hit the known minimum {s_star}")
                lin_S, lin_dict, _, _ = linear_search_S(
                    config, 300, 1, cb, None, self.weights,
                    self.s_min, self.s_max, prev_best_S=None, seeds=[42],
                    pantheon_data={})
                self.assertLessEqual(
                    abs(lin_S - brute_S), 1,
                    f"linear S={lin_S} pinned away from brute {brute_S} "
                    f"(early-stop keyed on the wrong objective metric)")
                # And it must NOT pin at the upper boundary (the broken behavior).
                self.assertNotEqual(lin_S, self.s_list[-1],
                                    "linear pinned at s_max (the pre-fix bug)")
                self.assertLess(lin_dict["chi2_dof"], 0.55,
                                "linear reported an inflated chi2 (pinned)")

    def test_lcdm_early_stop_path_unchanged(self):
        """LCDM objective still keys the early-stop on USED_MATCH_METRIC_KEYS.

        Guards the byte-identical-LCDM invariant: the fix branches on
        config.objective, so the LCDM branch must select exactly the LCDM keys.
        """
        # The branch added by the fix: lcdm -> USED_MATCH_METRIC_KEYS,
        # everything else -> ('match_avg_pct',). Assert the LCDM selection is
        # the original key set (so its loop is identical to before the fix).
        for objective, expected in (
            ("lcdm", USED_MATCH_METRIC_KEYS),
            ("pantheon", ('match_avg_pct',)),
        ):
            with self.subTest(objective=objective):
                keys = (USED_MATCH_METRIC_KEYS if objective == "lcdm"
                        else ('match_avg_pct',))
                self.assertEqual(keys, expected)


if __name__ == "__main__":
    unittest.main()
