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
    compare_expansion_histories,
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

        result = friedmann_equation(a, t, lcdm.H0, lcdm.Omega_m, lcdm.Omega_Lambda)

        # Should be positive (universe expanding)
        self.assertGreater(result, 0)

    def test_friedmann_equation_zero_scale_factor(self):
        """Test Friedmann equation handles a=0 gracefully"""
        lcdm = LambdaCDMParameters()
        a = 0.0
        t = 0.0

        result = friedmann_equation(a, t, lcdm.H0, lcdm.Omega_m, lcdm.Omega_Lambda)

        # Should return small positive value, not divide by zero
        self.assertGreater(result, 0)
        self.assertLess(result, 1e-9)

    def test_friedmann_equation_matter_only_vs_lcdm(self):
        """Test matter-only expands slower than ΛCDM at late times"""
        lcdm = LambdaCDMParameters()
        a = 1.0  # Today
        t = 0.0

        da_dt_lcdm = friedmann_equation(a, t, lcdm.H0, lcdm.Omega_m, lcdm.Omega_Lambda)
        da_dt_matter = friedmann_equation(a, t, lcdm.H0, lcdm.Omega_m, 0.0)

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


class TestCompareExpansionHistories(unittest.TestCase):
    """Test expansion history comparison"""

    def test_compare_identical_histories(self):
        """Test identical histories give 100% match"""
        size_ext = 20.0
        size_lcdm = 20.0

        match = compare_expansion_histories(size_ext, size_lcdm)

        self.assertAlmostEqual(match, 100.0, places=10)

    def test_compare_different_histories(self):
        """Test different histories give less than 100% match"""
        size_ext = 18.0
        size_lcdm = 20.0

        match = compare_expansion_histories(size_ext, size_lcdm)

        # 18/20 = 0.9, so 10% diff, 90% match
        self.assertAlmostEqual(match, 90.0, places=1)

    def test_compare_works_with_arrays(self):
        """Test comparison works with array inputs"""
        size_ext = np.array([18.0, 19.0, 20.0])
        size_lcdm = np.array([20.0, 20.0, 20.0])

        match = compare_expansion_histories(size_ext, size_lcdm)

        # Should return array of matches
        self.assertEqual(len(match), 3)
        self.assertAlmostEqual(match[-1], 100.0, places=10)


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
