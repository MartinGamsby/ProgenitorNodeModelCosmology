"""
Unit tests for solve_friedmann_at_times function.
Ensures exact time alignment for LCDM baseline calculations.
"""

import unittest
import numpy as np
from cosmo.analysis import solve_friedmann_at_times, solve_friedmann_equation


class TestSolveFriedmannAtTimes(unittest.TestCase):
    """Test solve_friedmann_at_times for exact time alignment"""

    def test_first_point_is_exactly_input_time(self):
        """First output time should exactly match first input time"""
        t_array = np.array([10.8, 11.0, 11.2, 11.4, 11.6])
        result = solve_friedmann_at_times(t_array)

        self.assertEqual(result['t_Gyr'][0], t_array[0],
                        "First time point should exactly match input")

    def test_all_times_match_input(self):
        """All output times should exactly match input times"""
        t_array = np.linspace(10.8, 11.8, 16)
        result = solve_friedmann_at_times(t_array)

        np.testing.assert_array_equal(result['t_Gyr'], t_array,
                                      "Output times should exactly match input times")

    def test_scale_factor_increases_monotonically(self):
        """Scale factor should increase monotonically"""
        t_array = np.linspace(5.0, 13.8, 20)
        result = solve_friedmann_at_times(t_array)

        diffs = np.diff(result['a'])
        self.assertTrue(np.all(diffs > 0),
                       "Scale factor should increase monotonically")

    def test_matter_only_slower_than_lcdm(self):
        """Matter-only should expand slower than LCDM at late times"""
        t_array = np.linspace(10.0, 13.8, 10)

        lcdm_result = solve_friedmann_at_times(t_array, Omega_Lambda=None)
        matter_result = solve_friedmann_at_times(t_array, Omega_Lambda=0.0)

        # At late times, LCDM accelerates due to dark energy
        # So a_lcdm[-1] > a_matter[-1]
        self.assertGreater(lcdm_result['a'][-1], matter_result['a'][-1],
                          "LCDM should expand more than matter-only at late times")

    def test_consistent_with_solve_friedmann_equation(self):
        """Should give same results as solve_friedmann_equation at matching times"""
        t_start = 10.8
        t_end = 11.8

        # Old method (returns relative times)
        old_result = solve_friedmann_equation(t_start, t_end, n_points=50)
        # t_old is already ABSOLUTE times (from _t_Gyr_full after masking)
        # But the returned 't_Gyr' field is the masked absolute times
        t_absolute_old = old_result['t_Gyr']
        a_old = old_result['a']

        # New method at same ABSOLUTE times
        new_result = solve_friedmann_at_times(t_absolute_old)
        a_new = new_result['a']

        # Scale factors should match within interpolation error (~0.1%)
        rel_diff = np.abs(a_new - a_old) / a_old
        self.assertTrue(np.all(rel_diff < 0.001),
                       f"Scale factors should match within 0.1%, max diff: {np.max(rel_diff)*100:.3f}%")

    def test_handles_single_time_point(self):
        """Should work with a single time point"""
        t_array = np.array([13.8])
        result = solve_friedmann_at_times(t_array)

        self.assertEqual(len(result['a']), 1)
        self.assertEqual(result['t_Gyr'][0], 13.8)
        self.assertGreater(result['a'][0], 0.9,
                          "Scale factor at t~13.8 Gyr should be near 1.0")

    def test_hubble_parameter_positive(self):
        """Hubble parameter should be positive"""
        t_array = np.linspace(1.0, 13.8, 20)
        result = solve_friedmann_at_times(t_array)

        self.assertTrue(np.all(result['H_hubble'] > 0),
                       "Hubble parameter should always be positive")

    def test_hubble_parameter_decreases_for_matter_only(self):
        """Hubble parameter should decrease for matter-only cosmology"""
        t_array = np.linspace(5.0, 13.8, 10)
        result = solve_friedmann_at_times(t_array, Omega_Lambda=0.0)

        diffs = np.diff(result['H_hubble'])
        self.assertTrue(np.all(diffs < 0),
                       "Hubble parameter should decrease for matter-only")

    def test_matches_n_body_snapshot_times(self):
        """Should exactly match N-body snapshot timing"""
        # Simulate N-body snapshot times
        t_start_Gyr = 10.8
        t_duration_Gyr = 1.0
        n_steps = 150
        save_interval = 10

        # Compute exact snapshot times (as done in solve_lcdm_baseline)
        snapshot_steps = np.arange(0, n_steps + 1, save_interval)
        t_relative_Gyr = (snapshot_steps / n_steps) * t_duration_Gyr
        t_absolute_Gyr = t_start_Gyr + t_relative_Gyr

        # Solve at these times
        result = solve_friedmann_at_times(t_absolute_Gyr)

        # First time should be exactly t_start
        self.assertEqual(result['t_Gyr'][0], t_start_Gyr,
                        "First time should exactly match t_start")

        # Last time should be exactly t_end
        self.assertAlmostEqual(result['t_Gyr'][-1], t_start_Gyr + t_duration_Gyr, places=10,
                              msg="Last time should exactly match t_end")

        # Should have correct number of snapshots
        expected_snapshots = (n_steps // save_interval) + 1
        self.assertEqual(len(result['t_Gyr']), expected_snapshots,
                        f"Should have {expected_snapshots} snapshots")


if __name__ == '__main__':
    unittest.main()
