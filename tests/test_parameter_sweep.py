"""
Unit tests for parameter sweep module.

Tests search algorithms with dummy simulation callbacks that return
predictable results, enabling verification of search logic without
running expensive real simulations.
"""
import unittest
import numpy as np
from cosmo.parameter_sweep import (
    SearchMethod, SweepConfig, MatchWeights, SimResult, SimSimpleResult, LCDMBaseline,
    MATCH_METRIC_KEYS,
    build_m_list, build_s_list, build_center_mass_list,
    compute_match_metrics, ternary_search_S, linear_search_S,
    brute_force_search, run_sweep
)


def make_baseline(n_points: int = 31) -> LCDMBaseline:
    """Create a standard LCDM baseline for testing."""
    return LCDMBaseline(
        t_Gyr=np.linspace(5.8, 13.8, n_points),
        size_Gpc=np.linspace(10.0, 14.5, n_points),
        H_hubble=np.linspace(75.0, 68.0, n_points),  # H decreases over time
        size_final_Gpc=14.5,
        radius_max_Gpc=9.4,
        a_final=1.0
    )


def make_sim_result(quality: float = 1.0, n_points: int = 31) -> SimResult:
    """
    Create a SimResult with predictable quality.

    quality=1.0 produces perfect match with baseline.
    quality<1.0 adds small offset to values (worse match).

    Important: R^2 can be negative if the model is worse than a horizontal line.
    We keep offsets small so R^2 stays in a reasonable positive range for
    quality values near 1.0 (which is what search tests produce).
    """
    # Baseline values
    size_start = 10.0
    size_end = 14.5
    h_start = 75.0
    h_end = 68.0

    # For quality < 1, subtract offset so sim underperforms baseline.
    # Negative offset means smaller size -> endpoint undershoots -> match_end_pct drops.
    offset_factor = (1.0 - quality) * 0.5  # Max 0.5 Gpc offset at quality=0
    size_offset = -offset_factor

    h_offset = -offset_factor * 2.0  # Hubble also varies with quality

    return SimResult(
        size_curve_Gpc=np.linspace(size_start + size_offset, size_end + size_offset, n_points),
        hubble_curve=np.linspace(h_start + h_offset, h_end + h_offset, n_points),
        t_Gyr=np.linspace(5.8, 13.8, n_points),
        params=None,
        results=SimSimpleResult(
            size_final_Gpc=size_end + size_offset,
            radius_max_Gpc=9.4 + size_offset,
            a_final=1.0,
        )
    )


def make_unimodal_callback(optimal_S: int, optimal_M: int = 500, optimal_centerM: int = 1):
    """
    Create callback where quality peaks at (optimal_M, optimal_S, optimal_centerM).

    Quality decreases with distance from optimum, creating a testable peak.
    Uses exponential decay to keep quality in [0.5, 1.0] range for reasonable R^2.
    """
    from typing import List
    def callback(M: int, S: int, centerM: int, seeds: List[int]) -> List[SimResult]:
        # Distance from optimal point
        distance = (
            abs(S - optimal_S) / 20.0 +
            abs(M - optimal_M) / 2000.0 +
            abs(centerM - optimal_centerM) / 100.0
        )
        # Quality peaks at 1.0 when at optimum, decays slowly
        # Use exponential decay to keep quality in reasonable range
        quality = 0.5 + 0.5 * np.exp(-distance)
        return [make_sim_result(quality) for _ in seeds]
    return callback


def make_monotonic_callback(direction: str = "increasing"):
    """
    Create callback where quality changes monotonically with S.

    direction="increasing": quality increases with S
    direction="decreasing": quality decreases with S
    """
    from typing import List
    def callback(M: int, S: int, centerM: int, seeds: List[int]) -> List[SimResult]:
        if direction == "increasing":
            quality = S / 100.0
        else:
            quality = (100 - S) / 100.0
        return [make_sim_result(max(0.1, min(1.0, quality))) for _ in seeds]
    return callback


class TestSweepConfig(unittest.TestCase):
    """Test SweepConfig dataclass and properties."""

    def test_defaults(self):
        """Default values should match expected."""
        config = SweepConfig()
        self.assertFalse(config.quick_search)
        self.assertEqual(config.many_search, 3)
        self.assertTrue(config.search_center_mass)
        self.assertEqual(config.t_start_Gyr, 5.8)
        self.assertEqual(config.t_duration_Gyr, 8.0)
        self.assertEqual(config.damping_factor, None)
        self.assertEqual(config.s_min_gpc, 15)
        self.assertEqual(config.s_max_gpc, 60)
        self.assertEqual(config.save_interval, 10)

    def test_particle_count_quick(self):
        """Quick search uses fewer particles."""
        config = SweepConfig(quick_search=True)
        self.assertEqual(config.particle_count, 200)

    def test_particle_count_many(self):
        """Many search uses full particles (same as default)."""
        config = SweepConfig(many_search=10)
        self.assertEqual(config.particle_count, 2000)

    def test_particle_count_default(self):
        """Default search uses most particles."""
        config = SweepConfig()
        self.assertEqual(config.particle_count, 2000)

    def test_n_steps_quick(self):
        """Quick search uses fewer steps."""
        config = SweepConfig(quick_search=True)
        self.assertEqual(config.n_steps, 250)

    def test_n_steps_default(self):
        """Default search uses more steps."""
        config = SweepConfig()
        self.assertEqual(config.n_steps, 300)


class TestMatchWeights(unittest.TestCase):
    """Test MatchWeights dataclass."""

    def test_defaults(self):
        """Default weights should match expected."""
        weights = MatchWeights()
        self.assertEqual(weights.hubble_half_curve, 0.025)
        self.assertEqual(weights.hubble_curve, 0.025)
        self.assertEqual(weights.size_half_curve, 0.25)
        self.assertEqual(weights.size_curve, 0.2)
        self.assertEqual(weights.endpoint, 0.4)
        self.assertEqual(weights.max_radius, 0.1)

    def test_weights_sum_to_one(self):
        """Default weights should sum to 1.0."""
        weights = MatchWeights()
        total = (weights.hubble_half_curve + weights.hubble_curve +
                 weights.size_half_curve + weights.size_curve +
                 weights.endpoint + weights.max_radius)
        self.assertAlmostEqual(total, 1.0)


class TestParameterSpaceBuilders(unittest.TestCase):
    """Test parameter space generation functions."""

    def test_build_m_list_many_search_false(self):
        """Coarse M list has fewer values."""
        m_list = build_m_list(many_search=3)
        self.assertGreaterEqual(len(m_list), 5)  # At least a few values
        self.assertLess(len(m_list), 100)

    def test_build_m_list_many_search_true(self):
        """Fine M list has more values."""
        m_list_coarse = build_m_list(many_search=3)
        m_list_fine = build_m_list(many_search=10)
        self.assertGreater(len(m_list_fine), len(m_list_coarse))

    def test_build_m_list_descending_order(self):
        """M list should be in descending order (high to low)."""
        m_list = build_m_list()
        for i in range(len(m_list) - 1):
            self.assertGreater(m_list[i], m_list[i + 1])

    def test_build_m_list_starts_high(self):
        """M list should start with high values."""
        m_list = build_m_list(many_search=3)
        self.assertGreater(m_list[0], 1000)

    def test_build_m_list_ends_low(self):
        """M list should end with low values."""
        m_list = build_m_list()
        self.assertEqual(m_list[-1], 20)  # Uses generate_increments with min_value=20

    def test_build_s_list_range(self):
        """S list should cover specified range."""
        s_list = build_s_list(15, 60)
        self.assertEqual(s_list[0], 15)
        self.assertEqual(s_list[-1], 60)
        self.assertEqual(len(s_list), 46)

    def test_build_s_list_single_value(self):
        """S list with min==max should have one element."""
        s_list = build_s_list(30, 30)
        self.assertEqual(len(s_list), 1)
        self.assertEqual(s_list[0], 30)

    def test_build_center_mass_list_search_enabled(self):
        """Center mass list with search has multiple values."""
        cm_list = build_center_mass_list(search_center_mass=True)
        self.assertGreater(len(cm_list), 1)
        self.assertEqual(cm_list[0], 1)  # Starts at 1

    def test_build_center_mass_list_search_disabled(self):
        """Center mass list without search has only [1]."""
        cm_list = build_center_mass_list(search_center_mass=False)
        self.assertEqual(cm_list, [1])

    def test_build_center_mass_list_many_search(self):
        """Many search produces finer center mass increments."""
        cm_coarse = build_center_mass_list(search_center_mass=True, many_search=3)
        cm_fine = build_center_mass_list(search_center_mass=True, many_search=10)
        self.assertGreater(len(cm_fine), len(cm_coarse))


class TestComputeMatchMetrics(unittest.TestCase):
    """Test match metric computation."""

    def test_perfect_match_high_score(self):
        """Perfect match should give high percentage."""
        baseline = make_baseline()
        sim_result = make_sim_result(quality=1.0)
        weights = MatchWeights()
        metrics = compute_match_metrics(sim_result, baseline, weights)
        self.assertGreater(metrics['match_avg_pct'], 90.0)

    def test_poor_match_low_score(self):
        """Poor match should give lower percentage than perfect match."""
        baseline = make_baseline()
        sim_result_poor = make_sim_result(quality=0.5)
        sim_result_good = make_sim_result(quality=1.0)
        weights = MatchWeights()
        metrics_poor = compute_match_metrics(sim_result_poor, baseline, weights)
        metrics_good = compute_match_metrics(sim_result_good, baseline, weights)
        self.assertLess(metrics_poor['match_avg_pct'], metrics_good['match_avg_pct'])

    def test_diff_pct_complement(self):
        """diff_pct should be 100 - match_avg_pct."""
        baseline = make_baseline()
        sim_result = make_sim_result(quality=0.8)
        weights = MatchWeights()
        metrics = compute_match_metrics(sim_result, baseline, weights)
        self.assertAlmostEqual(
            metrics['diff_pct'] + metrics['match_avg_pct'],
            100.0,
            places=5
        )

    def test_all_metric_keys_present(self):
        """All expected metric keys should be present."""
        baseline = make_baseline()
        sim_result = make_sim_result()
        weights = MatchWeights()
        metrics = compute_match_metrics(sim_result, baseline, weights)
        for key in MATCH_METRIC_KEYS:
            self.assertIn(key, metrics)
        # Derived keys
        self.assertIn('match_avg_pct', metrics)
        self.assertIn('diff_pct', metrics)


class TestTernarySearch(unittest.TestCase):
    """Test ternary search algorithm."""

    def test_finds_optimal_S(self):
        """Ternary search should find S near the optimum."""
        optimal_S = 35
        callback = make_unimodal_callback(optimal_S=optimal_S, optimal_M=500)
        baseline = make_baseline()
        weights = MatchWeights()
        config = SweepConfig()

        best_S, best_match, best_result, all_results = ternary_search_S(
            config=config, M_factor=500, centerM=1, sim_callback=callback,
            baseline=baseline, weights=weights,
            s_min=15, s_max=60
        )

        # Should find optimal within +/- 2
        self.assertLessEqual(abs(best_S - optimal_S), 2)

    def test_warm_start_converges_faster(self):
        """Warm start with hint should evaluate fewer points."""
        optimal_S = 35
        callback = make_unimodal_callback(optimal_S=optimal_S)
        baseline = make_baseline()
        weights = MatchWeights()
        config = SweepConfig()

        # Without hint
        _, _, _, results_no_hint = ternary_search_S(
            config=config, M_factor=500, centerM=1, sim_callback=callback,
            baseline=baseline, weights=weights,
            s_min=15, s_max=60, s_hint=None
        )

        # With hint near optimal
        _, _, _, results_with_hint = ternary_search_S(
            config=config, M_factor=500, centerM=1, sim_callback=callback,
            baseline=baseline, weights=weights,
            s_min=15, s_max=60, s_hint=34, hint_window=5
        )

        # Hint should lead to fewer evaluations
        self.assertLessEqual(len(results_with_hint), len(results_no_hint))

    def test_handles_boundary_s_min(self):
        """Should handle optimum at s_min boundary."""
        callback = make_monotonic_callback(direction="decreasing")
        baseline = make_baseline()
        weights = MatchWeights()
        # Use unique M_factor/centerM to avoid cache collisions with real sim data
        config = SweepConfig()

        best_S, _, _, _ = ternary_search_S(
            config=config, M_factor=999997, centerM=997, sim_callback=callback,
            baseline=baseline, weights=weights,
            s_min=15, s_max=60
        )

        # Best should be at or near minimum
        self.assertLessEqual(best_S, 20)

    def test_handles_boundary_s_max(self):
        """Should handle optimum at s_max boundary."""
        callback = make_monotonic_callback(direction="increasing")
        baseline = make_baseline()
        weights = MatchWeights()
        # Use unique M_factor/centerM to avoid cache collisions with real sim data
        config = SweepConfig()

        best_S, _, _, _ = ternary_search_S(
            config=config, M_factor=999998, centerM=998, sim_callback=callback,
            baseline=baseline, weights=weights,
            s_min=15, s_max=60
        )

        # Best should be at or near maximum
        self.assertGreaterEqual(best_S, 55)

    def test_returns_all_results(self):
        """Should return results for all evaluated S values."""
        callback = make_unimodal_callback(optimal_S=35)
        baseline = make_baseline()
        weights = MatchWeights()
        config = SweepConfig()

        _, _, _, all_results = ternary_search_S(
            config=config, M_factor=500, centerM=1, sim_callback=callback,
            baseline=baseline, weights=weights,
            s_min=15, s_max=60
        )

        # Should have multiple results
        self.assertGreater(len(all_results), 3)

        # Each result should have required keys
        for result in all_results:
            self.assertIn('M_factor', result)
            self.assertIn('S_gpc', result)
            self.assertIn('match_avg_pct', result)


class TestLinearSearch(unittest.TestCase):
    """Test linear search algorithm."""

    def test_finds_optimal_S(self):
        """Linear search should find S near the optimum."""
        # Use optimal at S=55 (near start of search from 60)
        # so search doesn't skip past it with adaptive stepping
        optimal_S = 55
        callback = make_unimodal_callback(optimal_S=optimal_S)
        baseline = make_baseline()
        weights = MatchWeights()
        config = SweepConfig()

        best_S, best_result, _, _ = linear_search_S(
            config=config, M_factor=500, centerM=1, sim_callback=callback,
            baseline=baseline, weights=weights,
            s_min=15, s_max=60
        )

        # Should find optimal within +/- 5 (linear search may skip some values)
        self.assertLessEqual(abs(best_S - optimal_S), 5)

    def test_early_stopping_when_decreasing(self):
        """Should stop early when match starts decreasing significantly."""
        # Callback peaks at S=58, quality drops steeply away from peak.
        # No floor clamp so metrics diverge enough to trigger all_worse check.
        def sharp_peak_callback(M, S, centerM, seeds):
            distance = abs(S - 58) / 2.0
            quality = max(0.01, 1.0 / (1.0 + distance * distance))
            return [make_sim_result(quality) for _ in seeds]

        baseline = make_baseline()
        weights = MatchWeights()
        config = SweepConfig()

        _, _, _, all_results = linear_search_S(
            config=config, M_factor=500, centerM=1, sim_callback=sharp_peak_callback,
            baseline=baseline, weights=weights,
            s_min=15, s_max=60
        )

        # Should evaluate fewer configs than full range due to stopping
        # Full range is 46 values (60-15+1), should evaluate much fewer
        self.assertLess(len(all_results), 30)

    def test_returns_should_stop_at_s_min(self):
        """Should signal to stop M search when best S is at minimum."""
        # Callback peaks at S=15 (s_min)
        callback = make_monotonic_callback(direction="decreasing")
        baseline = make_baseline()
        weights = MatchWeights()
        config = SweepConfig()

        best_S, _, should_stop, _ = linear_search_S(
            config=config, M_factor=500, centerM=1, sim_callback=callback,
            baseline=baseline, weights=weights,
            s_min=15, s_max=60
        )

        # Should signal stop because best is at boundary
        if best_S == 15:
            self.assertTrue(should_stop)

    def test_handles_negative_match(self):
        """Should handle negative match values gracefully."""
        def bad_callback(M, S, centerM, seeds):
            # Return very poor result
            return [make_sim_result(quality=0.01) for _ in seeds]

        baseline = make_baseline()
        weights = MatchWeights()
        config = SweepConfig()

        # Should not raise exception
        best_S, best_result, _, _ = linear_search_S(
            config=config, M_factor=500, centerM=1, sim_callback=bad_callback,
            baseline=baseline, weights=weights,
            s_min=15, s_max=60
        )

        # Should still return a valid result
        self.assertIn('match_avg_pct', best_result)


class TestBruteForceSearch(unittest.TestCase):
    """Test brute force search algorithm."""

    def test_evaluates_all_combinations(self):
        """Should evaluate all M x S x centerM combinations."""
        callback = make_unimodal_callback(optimal_S=35)
        baseline = make_baseline()
        weights = MatchWeights()
        config = SweepConfig()

        s_list = [20, 25, 30]
        center_masses = [1]

        for many_search in [3, 10]:
            results = brute_force_search(
                config, many_search, s_list, center_masses,
                callback, baseline, weights
            )

            m_list = build_m_list(many_search, multiplier=many_search)
            expected_count = len(m_list) * len(s_list) * len(center_masses)
            self.assertEqual(len(results), expected_count)


class TestRunSweep(unittest.TestCase):
    """Test main run_sweep entry point."""

    def test_brute_force_method(self):
        """run_sweep with BRUTE_FORCE should evaluate all configs."""
        callback = make_unimodal_callback(optimal_S=35)
        baseline = make_baseline()

        config = SweepConfig(
            search_center_mass=False,
            s_min_gpc=30,
            s_max_gpc=35
        )

        results = run_sweep(
            config, SearchMethod.BRUTE_FORCE,
            callback, baseline
        )

        # Should have results for multiple configs
        self.assertGreater(len(results), 0)

    def test_linear_search_method(self):
        """run_sweep with LINEAR_SEARCH should work."""
        callback = make_unimodal_callback(optimal_S=35, optimal_M=500)
        baseline = make_baseline()

        config = SweepConfig(
            search_center_mass=False,
            s_min_gpc=30,
            s_max_gpc=40
        )

        results = run_sweep(
            config, SearchMethod.LINEAR_SEARCH,
            callback, baseline
        )

        self.assertGreater(len(results), 0)

    def test_ternary_search_method(self):
        """run_sweep with TERNARY_SEARCH should work."""
        callback = make_unimodal_callback(optimal_S=35, optimal_M=500)
        baseline = make_baseline()

        config = SweepConfig(
            search_center_mass=False,
            s_min_gpc=30,
            s_max_gpc=40
        )

        results = run_sweep(
            config, SearchMethod.TERNARY_SEARCH,
            callback, baseline
        )

        self.assertGreater(len(results), 0)

    def test_default_weights_used(self):
        """Should use default weights when None provided."""
        callback = make_unimodal_callback(optimal_S=35)
        baseline = make_baseline()

        config = SweepConfig(
            search_center_mass=False,
            s_min_gpc=30,
            s_max_gpc=35
        )

        # Should not raise when weights=None
        results = run_sweep(
            config, SearchMethod.BRUTE_FORCE,
            callback, baseline, weights=None
        )

        self.assertGreater(len(results), 0)


if __name__ == '__main__':
    unittest.main()
