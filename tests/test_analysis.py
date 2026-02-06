"""
Unit tests for cosmo.analysis module

Tests shared analysis functions for:
- Friedmann equation solving
- Initial conditions calculation
- Expansion history comparison
- Runaway particle detection
"""

import unittest
import numpy as np
from cosmo.analysis import (
    friedmann_equation,
    solve_friedmann_equation,
    calculate_initial_conditions,
    normalize_to_initial_size,
    calculate_r_squared,
    compare_expansion_histories,
    compare_expansion_history,
    detect_runaway_particles,
    calculate_today_marker
)
from cosmo.constants import LambdaCDMParameters


class TestFriedmannEquation(unittest.TestCase):
    """Test Friedmann equation function"""

    def test_friedmann_equation_basic(self):
        """Test Friedmann equation returns positive da/dt"""
        lcdm = LambdaCDMParameters()
        a = 1.0
        t = 0.0

        result = friedmann_equation(a, t, lcdm.H0_si, lcdm.Omega_m, lcdm.Omega_Lambda)

        # Should be positive (universe expanding)
        self.assertGreater(result, 0)

    def test_friedmann_equation_zero_scale_factor(self):
        """Test Friedmann equation handles a=0 gracefully"""
        lcdm = LambdaCDMParameters()
        a = 0.0
        t = 0.0

        result = friedmann_equation(a, t, lcdm.H0_si, lcdm.Omega_m, lcdm.Omega_Lambda)

        # Should return small positive value, not divide by zero
        self.assertGreater(result, 0)
        self.assertLess(result, 1e-9)

    def test_friedmann_equation_matter_only_vs_lcdm(self):
        """Test matter-only expands slower than ΛCDM at late times"""
        lcdm = LambdaCDMParameters()
        a = 1.0  # Today
        t = 0.0

        da_dt_lcdm = friedmann_equation(a, t, lcdm.H0_si, lcdm.Omega_m, lcdm.Omega_Lambda)
        da_dt_matter = friedmann_equation(a, t, lcdm.H0_si, lcdm.Omega_m, 0.0)

        # ΛCDM should expand faster (dark energy accelerates)
        self.assertGreater(da_dt_lcdm, da_dt_matter)


class TestSolveFriedmannEquation(unittest.TestCase):
    """Test Friedmann equation solver"""

    def test_solve_friedmann_returns_dict(self):
        """Test solver returns dictionary with expected keys"""
        result = solve_friedmann_equation(10.0, 16.0)

        self.assertIsInstance(result, dict)
        self.assertIn('t_Gyr', result)
        self.assertIn('a', result)
        self.assertIn('H_hubble', result)

    def test_solve_friedmann_scale_factor_increases(self):
        """Test scale factor increases with time"""
        result = solve_friedmann_equation(10.0, 16.0)

        # Scale factor should increase monotonically
        self.assertTrue(np.all(np.diff(result['a']) > 0))

    def test_solve_friedmann_matter_only_slower(self):
        """Test matter-only expansion is slower than ΛCDM"""
        lcdm_result = solve_friedmann_equation(10.0, 16.0, Omega_Lambda=None)
        matter_result = solve_friedmann_equation(10.0, 16.0, Omega_Lambda=0.0)

        # Final scale factor should be smaller for matter-only
        self.assertLess(matter_result['a'][-1], lcdm_result['a'][-1])

    def test_solve_friedmann_time_window(self):
        """Test solution is properly windowed to requested times"""
        t_start = 10.0
        t_end = 15.0
        result = solve_friedmann_equation(t_start, t_end)

        # Time array should be within requested window
        self.assertGreaterEqual(result['t_Gyr'][0], t_start)
        self.assertLessEqual(result['t_Gyr'][-1], t_end)


class TestCalculateInitialConditions(unittest.TestCase):
    """Test initial conditions calculation"""

    def test_calculate_initial_conditions_returns_dict(self):
        """Test function returns dictionary with expected keys"""
        result = calculate_initial_conditions(10.8)

        self.assertIsInstance(result, dict)
        self.assertIn('a_start', result)
        self.assertIn('box_size_Gpc', result)
        self.assertIn('H_start_hubble', result)

    def test_calculate_initial_conditions_scale_factor_range(self):
        """Test scale factor is in reasonable range"""
        result = calculate_initial_conditions(10.8)

        # At t=10.8 Gyr, scale factor should be ~0.8 (before today at a=1)
        self.assertGreater(result['a_start'], 0.5)
        self.assertLess(result['a_start'], 1.0)

    def test_calculate_initial_conditions_box_size_scales(self):
        """Test box size scales properly with time"""
        early = calculate_initial_conditions(8.0)
        late = calculate_initial_conditions(12.0)

        # Earlier time should have smaller box
        self.assertLess(early['box_size_Gpc'], late['box_size_Gpc'])

    def test_calculate_initial_conditions_hubble_decreases(self):
        """Test Hubble parameter decreases with time"""
        early = calculate_initial_conditions(8.0)
        late = calculate_initial_conditions(12.0)

        # Hubble parameter should decrease (expansion slowing in matter era)
        self.assertGreater(early['H_start_hubble'], late['H_start_hubble'])


class TestNormalizeToInitialSize(unittest.TestCase):
    """Test scale factor to physical size conversion"""

    def test_normalize_preserves_initial_size(self):
        """Test first element equals initial size"""
        a_array = np.array([0.8, 0.9, 1.0, 1.1])
        initial_size = 12.0

        result = normalize_to_initial_size(a_array, initial_size)

        # First element should equal initial size
        self.assertAlmostEqual(result[0], initial_size, places=10)

    def test_normalize_scales_linearly(self):
        """Test size scales linearly with scale factor"""
        a_array = np.array([1.0, 2.0, 3.0])
        initial_size = 10.0

        result = normalize_to_initial_size(a_array, initial_size)

        # Should scale linearly
        np.testing.assert_array_almost_equal(result, [10.0, 20.0, 30.0])


class TestCalculateRSquared(unittest.TestCase):
    """Test R² calculation function"""

    def test_calculate_r_squared_perfect_match(self):
        """Test perfect match gives R² = 1.0"""
        y_actual = np.array([1.0, 2.0, 3.0, 4.0])
        y_predicted = np.array([1.0, 2.0, 3.0, 4.0])

        r2 = calculate_r_squared(y_actual, y_predicted)

        self.assertAlmostEqual(r2, 1.0, places=10)

    def test_calculate_r_squared_constant_offset(self):
        """Test constant offset reduces R²"""
        y_actual = np.array([1.0, 2.0, 3.0, 4.0])
        y_predicted = np.array([2.0, 3.0, 4.0, 5.0])  # offset by +1

        r2 = calculate_r_squared(y_actual, y_predicted)

        # Should be less than 1 due to bias
        self.assertLess(r2, 1.0)
        self.assertGreater(r2, 0.0)

    def test_calculate_r_squared_scaled(self):
        """Test scaling reduces R² (can be negative for large scale errors)"""
        y_actual = np.array([1.0, 2.0, 3.0, 4.0])
        y_predicted = np.array([2.0, 4.0, 6.0, 8.0])  # scaled by 2x

        r2 = calculate_r_squared(y_actual, y_predicted)

        # Should be less than 1 due to scaling
        self.assertLess(r2, 1.0)
        # Large systematic error can make R² negative
        # Just verify it's computed (don't enforce specific range)

    def test_calculate_r_squared_random_noise(self):
        """Test small noise gives R² > 0.99"""
        np.random.seed(42)
        y_actual = np.linspace(1, 10, 100)
        y_predicted = y_actual + np.random.normal(0, 0.01, 100)

        r2 = calculate_r_squared(y_actual, y_predicted)

        # Small noise should still give high R²
        self.assertGreater(r2, 0.99)

    def test_calculate_r_squared_uncorrelated(self):
        """Test uncorrelated data gives R² ≈ 0 or negative"""
        y_actual = np.array([1.0, 2.0, 3.0, 4.0])
        y_predicted = np.array([4.0, 1.0, 3.0, 2.0])  # uncorrelated

        r2 = calculate_r_squared(y_actual, y_predicted)

        # Should be around 0 or negative
        self.assertLess(r2, 0.5)

    def test_calculate_r_squared_constant_baseline(self):
        """Test constant baseline with perfect match gives R² = 1.0"""
        y_actual = np.array([5.0, 5.0, 5.0, 5.0])
        y_predicted = np.array([5.0, 5.0, 5.0, 5.0])

        r2 = calculate_r_squared(y_actual, y_predicted)

        # Edge case: perfect match of constant curves
        self.assertAlmostEqual(r2, 1.0, places=10)

    def test_calculate_r_squared_constant_baseline_mismatch(self):
        """Test constant baseline with mismatch gives R² = 0.0"""
        y_actual = np.array([5.0, 5.0, 5.0, 5.0])
        y_predicted = np.array([4.0, 4.0, 4.0, 4.0])

        r2 = calculate_r_squared(y_actual, y_predicted)

        # Edge case: different constant curves
        self.assertAlmostEqual(r2, 0.0, places=10)

    def test_calculate_r_squared_scalar_inputs(self):
        """Test scalar inputs are handled correctly"""
        y_actual = 5.0
        y_predicted = 5.0

        r2 = calculate_r_squared(y_actual, y_predicted)

        # Scalars should work, perfect match
        self.assertAlmostEqual(r2, 1.0, places=10)


class TestCompareExpansionHistories(unittest.TestCase):
    """Test expansion history comparison"""

    def test_compare_identical_history(self):
        size_ext = 20.0
        size_lcdm = 20.0
        r2 = compare_expansion_history(size_ext, size_lcdm)
        self.assertAlmostEqual(r2, 100.0, places=10)

    def test_compare_different_history(self):
        size_ext = 18.0
        size_lcdm = 20.0
        r2 = compare_expansion_history(size_ext, size_lcdm)
        self.assertAlmostEqual(r2, 90.0, places=10)

    def test_compare_identical_histories(self):
        """Test identical histories give R² = 1.0 (new default) or 100% match (old)"""
        size_ext = 20.0
        size_lcdm = 20.0

        # Test new default: R²
        r2 = compare_expansion_histories(size_ext, size_lcdm, times_100=False)
        self.assertAlmostEqual(r2, 1.0, places=10)
        r2 = compare_expansion_histories(size_ext, size_lcdm)
        self.assertAlmostEqual(r2, 100.0, places=10)

        # Test backward compatibility: percentage
        match = compare_expansion_histories(size_ext, size_lcdm, use_r_squared=False)
        self.assertAlmostEqual(match, 100.0, places=10)

    def test_compare_different_histories(self):
        """Test different histories: R² = 0.0 for different scalars (edge case), 90% match in percentage mode"""
        size_ext = 18.0
        size_lcdm = 20.0

        # Test new default: R² (scalar inputs with constant baseline → edge case)
        r2 = compare_expansion_histories(size_ext, size_lcdm)
        # Single scalar comparison: baseline is constant, mismatch → R² = 0.0
        self.assertAlmostEqual(r2, 0.0, places=10)

        # Test backward compatibility: percentage
        match = compare_expansion_histories(size_ext, size_lcdm, use_r_squared=False)
        # 18/20 = 0.9, so 10% diff, 90% match
        self.assertAlmostEqual(match, 90.0, places=1)

    def test_compare_works_with_arrays(self):
        """Test comparison works with array inputs"""
        size_ext = np.array([18.0, 19.0, 20.0])
        size_lcdm = np.array([20.0, 20.0, 20.0])

        match = compare_expansion_histories(size_ext, size_lcdm, return_array=True)

        # Should return array of percentage matches (not R²)
        self.assertEqual(len(match), 3)
        self.assertAlmostEqual(match[-1], 100.0, places=10)

    def test_r_squared_default_behavior(self):
        """Test default returns R² value in 0-1 range"""
        size_ext = np.linspace(10, 26, 50)
        size_lcdm = np.linspace(10, 26, 50) + 0.1  # small constant offset

        r2 = compare_expansion_histories(size_ext, size_lcdm, times_100=False)

        # Should be in 0-1 range
        self.assertGreater(r2, 0.0)
        self.assertLess(r2, 1.0)

        r2 = compare_expansion_histories(size_ext, size_lcdm)
        self.assertGreater(r2, 0.0)
        self.assertLess(r2, 100.0)

    def test_backward_compatibility_percentage(self):
        """Test percentage mode matches old behavior"""
        size_ext = np.array([18.0, 19.0, 20.0])
        size_lcdm = np.array([20.0, 20.0, 20.0])

        match = compare_expansion_histories(size_ext, size_lcdm, use_r_squared=False)

        # Manually calculate expected percentage
        expected = 100 - np.mean(np.abs(size_ext - size_lcdm) / size_lcdm * 100)
        self.assertAlmostEqual(match, expected, places=10)

    def test_return_diagnostics(self):
        """Test diagnostics return dict with multiple metrics"""
        size_ext = np.array([18.0, 19.0, 20.0])
        size_lcdm = np.array([20.0, 20.0, 20.0])

        diagnostics = compare_expansion_histories(size_ext, size_lcdm, return_diagnostics=True)

        # Should be dict with expected keys
        self.assertIsInstance(diagnostics, dict)
        self.assertIn('r_squared', diagnostics)
        self.assertIn('match_pct', diagnostics)
        self.assertIn('max_error_pct', diagnostics)
        self.assertIn('mean_error_pct', diagnostics)
        self.assertIn('rmse', diagnostics)
        self.assertIn('rmse_pct', diagnostics)

        # Both R² and match_pct are always computed in diagnostics mode
        self.assertIsNotNone(diagnostics['r_squared'])
        self.assertIsNotNone(diagnostics['match_pct'])

    def test_diagnostics_with_percentage_mode(self):
        """Test diagnostics in percentage mode"""
        size_ext = np.array([18.0, 19.0, 20.0])
        size_lcdm = np.array([20.0, 20.0, 20.0])

        diagnostics = compare_expansion_histories(
            size_ext, size_lcdm, use_r_squared=False, return_diagnostics=True
        )

        # Both metrics are always computed in diagnostics mode
        self.assertIsNotNone(diagnostics['r_squared'])
        self.assertIsNotNone(diagnostics['match_pct'])

    def test_realistic_cosmology_curves(self):
        """Test R² with realistic cosmology curves"""
        np.random.seed(42)
        size_lcdm = np.linspace(10, 26, 250)  # realistic Gpc range
        size_ext = size_lcdm + 0.01 * np.random.randn(250)  # small noise

        r2 = compare_expansion_histories(size_ext, size_lcdm)

        # Very close match should give R² > 0.9999
        self.assertGreater(r2, 0.9999)

        # Verify corresponds to high percentage match
        match_pct = compare_expansion_histories(size_ext, size_lcdm, use_r_squared=False)
        self.assertGreater(match_pct, 99.0)

    def test_return_array_returns_percentage_not_r_squared(self):
        """Test return_array returns percentage errors, not R²"""
        size_ext = np.array([18.0, 19.0, 20.0])
        size_lcdm = np.array([20.0, 20.0, 20.0])

        result = compare_expansion_histories(size_ext, size_lcdm, return_array=True, use_r_squared=True)

        # Should be array of percentages (0-100 range), not R² (0-1 range)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(len(result), 3)
        # Values should be in percentage range
        self.assertTrue(np.all(result >= 0))
        self.assertTrue(np.all(result <= 100))

    def test_edge_case_zero_values(self):
        """Test handling of edge cases without division by zero errors"""
        # Test with very small values (should not crash)
        size_ext = np.array([1e-10, 2e-10, 3e-10])
        size_lcdm = np.array([1e-10, 2e-10, 3e-10])

        r2 = compare_expansion_histories(size_ext, size_lcdm, times_100=False)

        # Perfect match should give R² = 1.0
        self.assertAlmostEqual(r2, 1.0, places=5)

        r2 = compare_expansion_histories(size_ext, size_lcdm)
        self.assertAlmostEqual(r2, 100.0, places=5)


class TestDetectRunawayParticles(unittest.TestCase):
    """Test runaway particle detection"""

    def test_detect_no_runaway(self):
        """Test no detection when ratio is below threshold"""
        max_distance = 18.0
        rms_size = 15.0

        result = detect_runaway_particles(max_distance, rms_size)

        # Ratio is 1.2, below threshold of 2.0
        self.assertFalse(result['detected'])
        self.assertAlmostEqual(result['ratio'], 1.2, places=1)

    def test_detect_runaway(self):
        """Test detection when ratio exceeds threshold"""
        max_distance = 35.0
        rms_size = 15.0

        result = detect_runaway_particles(max_distance, rms_size)

        # Ratio is 2.33, above threshold of 2.0
        self.assertTrue(result['detected'])
        self.assertAlmostEqual(result['ratio'], 35.0/15.0, places=1)

    def test_detect_custom_threshold(self):
        """Test detection with custom threshold"""
        max_distance = 20.0
        rms_size = 15.0

        result = detect_runaway_particles(max_distance, rms_size, threshold=1.2)

        # Ratio is 1.33, above custom threshold of 1.2
        self.assertTrue(result['detected'])
        self.assertEqual(result['threshold'], 1.2)


class TestCalculateTodayMarker(unittest.TestCase):
    """Test 'today' marker calculation"""

    def test_today_within_window(self):
        """Test 'today' marker when within simulation window"""
        t_start = 10.0
        t_duration = 8.0  # Ends at 18.0
        today_Gyr = 13.8

        result = calculate_today_marker(t_start, t_duration, today_Gyr)

        # Should return offset from start: 13.8 - 10.0 = 3.8
        self.assertAlmostEqual(result, 3.8, places=10)

    def test_today_before_window(self):
        """Test 'today' marker when before simulation start"""
        t_start = 14.0
        t_duration = 6.0
        today_Gyr = 13.8

        result = calculate_today_marker(t_start, t_duration, today_Gyr)

        # Should return None (today is before simulation)
        self.assertIsNone(result)

    def test_today_after_window(self):
        """Test 'today' marker when after simulation end"""
        t_start = 8.0
        t_duration = 4.0  # Ends at 12.0
        today_Gyr = 13.8

        result = calculate_today_marker(t_start, t_duration, today_Gyr)

        # Should return None (today is after simulation)
        self.assertIsNone(result)


if __name__ == '__main__':
    unittest.main()
