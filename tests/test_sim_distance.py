"""
Unit tests for cosmo.sim_distance — the simulation-based distance-modulus kernel.

All tests are hermetic: no real data files read, no N-body simulation run.
An analytic LCDM a(t) (from cosmo.analysis.solve_friedmann_at_times) stands in
for the simulation's scale-factor history.

Tests:
  1. LCDM round-trip: kernel mu(z) from analytic a(t) matches model_distance_modulus
     (lcdm) after mean subtraction (offset marginalization proxy).
  2. Normalization: a_today[today_index] == 1.0; z_snap[today_index] == 0.
  3. z-mapping: earliest snapshot z ~ analytic 1/a_lcdm(t_start)/a_lcdm(today) - 1.
  4. Range clipping: z_target above z_max_cover → flagged out-of-range.
  5. today_tol guard: last absolute time ≠ 13.8 Gyr → ValueError with message.
  6. Non-positive a → ValueError.
  7. Mismatched array lengths → ValueError.
  8. d_L monotonically increasing (z_snap → d_L_Mpc).
"""

import unittest
import numpy as np

from cosmo.sim_distance import sim_to_distance_modulus
from cosmo.analysis import solve_friedmann_at_times
from cosmo.distances import model_distance_modulus

# ---------------------------------------------------------------------------
# Shared test fixture helpers
# ---------------------------------------------------------------------------

_T_START_GYR = 5.8
_T_END_GYR = 13.8          # Must end at 13.8 so today_index=-1 passes the guard
_N_SNAPSHOTS = 250          # Fine enough for the integral error < 0.03 mag


def _make_lcdm_inputs(n: int = _N_SNAPSHOTS, t_start: float = _T_START_GYR):
    """
    Build analytic LCDM a(t) inputs that mimic what the sim produces:
      - t_Gyr: relative, starts at 0.
      - a: renormalized so a[0] = 1 (matches sim normalization).
    """
    t_abs_grid = np.linspace(t_start, _T_END_GYR, n)   # absolute Gyr
    sol = solve_friedmann_at_times(t_abs_grid)
    a_abs = sol["a"]                                     # a from LCDM Friedmann solve

    # Mimic sim normalization: a[0] = 1 at t_start
    a_sim = a_abs / a_abs[0]
    t_Gyr_rel = t_abs_grid - t_start                    # relative time, starts at 0

    return a_sim, t_Gyr_rel, t_start


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------

class TestLCDMRoundTrip(unittest.TestCase):
    """
    Test 1: mu(z) from the integral kernel should reproduce the analytic
    LCDM mu(z) curve after removing the constant offset (mean subtraction).
    """

    def test_shape_matches_lcdm_after_offset_removal(self):
        a_sim, t_Gyr, t_start = _make_lcdm_inputs()

        # Target redshifts in the sim-covered range
        # With t_start=5.8 Gyr, a_lcdm(5.8)/a_lcdm(13.8) - 1 ~ 0.8 (approx)
        # Use a safe interior range.
        z_target = np.linspace(0.05, 0.7, 60)

        result = sim_to_distance_modulus(
            z_target=z_target,
            a=a_sim,
            t_Gyr=t_Gyr,
            t_start_Gyr=t_start,
        )

        z_in = result["z"]
        mu_sim = result["mu"]

        # Analytic LCDM reference on the same z grid
        mu_lcdm = model_distance_modulus(z_in, "lcdm")

        # Offset-marginalization proxy: subtract means before comparing
        mu_sim_c = mu_sim - np.mean(mu_sim)
        mu_lcdm_c = mu_lcdm - np.mean(mu_lcdm)

        max_residual = np.max(np.abs(mu_sim_c - mu_lcdm_c))
        self.assertLess(
            max_residual, 0.03,
            f"Max |residual| after offset subtraction = {max_residual:.4f} mag "
            f"(tolerance 0.03 mag). Discretization error too large."
        )

    def test_in_range_mask_aligned_to_z_target(self):
        """The 'in_range' mask must have the same length as z_target."""
        a_sim, t_Gyr, t_start = _make_lcdm_inputs()
        z_target = np.linspace(0.05, 0.7, 30)
        result = sim_to_distance_modulus(z_target, a_sim, t_Gyr, t_start)
        self.assertEqual(len(result["in_range"]), len(z_target))


class TestNormalization(unittest.TestCase):
    """Test 2: normalization invariants."""

    def setUp(self):
        self.a_sim, self.t_Gyr, self.t_start = _make_lcdm_inputs()
        z_target = np.linspace(0.05, 0.6, 20)
        self.result = sim_to_distance_modulus(
            z_target, self.a_sim, self.t_Gyr, self.t_start
        )

    def test_a_today_at_today_index_is_one(self):
        """a_today[-1] must equal 1.0 exactly (float division)."""
        a_today = self.result["a_today"]
        self.assertAlmostEqual(
            float(a_today[-1]), 1.0, places=14,
            msg="a_today at today_index should be exactly 1.0"
        )

    def test_z_snap_at_today_index_is_zero(self):
        """z_snap[-1] must be 0 (a_today=1 => z=1/1-1=0)."""
        z_snap = self.result["z_snap"]
        self.assertAlmostEqual(
            float(z_snap[-1]), 0.0, places=14,
            msg="z_snap at today_index should be 0"
        )


class TestZMapping(unittest.TestCase):
    """Test 3: z at the earliest snapshot matches the analytic LCDM value."""

    def test_earliest_snapshot_z_matches_analytic(self):
        a_sim, t_Gyr, t_start = _make_lcdm_inputs()
        z_target = np.linspace(0.05, 0.6, 20)
        result = sim_to_distance_modulus(z_target, a_sim, t_Gyr, t_start)

        z_snap = result["z_snap"]
        z_earliest_kernel = float(z_snap[0])   # highest z, earliest snapshot

        # Analytic reference: a(t_start) / a(today) - 1
        sol = solve_friedmann_at_times(np.array([t_start, _T_END_GYR]))
        a_t_start_analytic = sol["a"][0]
        a_today_analytic = sol["a"][1]
        z_earliest_analytic = a_today_analytic / a_t_start_analytic - 1.0

        self.assertAlmostEqual(
            z_earliest_kernel, z_earliest_analytic, delta=0.01,
            msg=(
                f"Earliest z mismatch: kernel={z_earliest_kernel:.4f}, "
                f"analytic={z_earliest_analytic:.4f}"
            )
        )


class TestRangeClipping(unittest.TestCase):
    """Test 4: z_target beyond z_max_cover is excluded from results."""

    def test_out_of_range_z_excluded(self):
        a_sim, t_Gyr, t_start = _make_lcdm_inputs()

        # Get the sim's covered range first
        z_probe = np.linspace(0.05, 0.6, 10)
        result_probe = sim_to_distance_modulus(z_probe, a_sim, t_Gyr, t_start)
        z_max = result_probe["z_cover"][1]

        # Now add z values beyond z_max
        z_beyond = np.array([z_max * 0.5, z_max * 0.9, z_max * 1.1, z_max * 1.5])
        result = sim_to_distance_modulus(z_beyond, a_sim, t_Gyr, t_start)

        # Only the first two z values should be in range
        in_range = result["in_range"]
        self.assertTrue(in_range[0], "z = 0.5*z_max should be in range")
        self.assertTrue(in_range[1], "z = 0.9*z_max should be in range")
        self.assertFalse(in_range[2], "z = 1.1*z_max should be out of range")
        self.assertFalse(in_range[3], "z = 1.5*z_max should be out of range")

        # Returned 'z' must not contain out-of-range entries
        self.assertEqual(len(result["z"]), int(np.sum(in_range)))
        self.assertTrue(np.all(result["z"] <= z_max + 1e-10))

    def test_all_out_of_range_raises(self):
        """ValueError if every z_target is outside the sim's z range."""
        a_sim, t_Gyr, t_start = _make_lcdm_inputs()

        # Use a z way beyond any plausible z_max from t_start=5.8 Gyr
        z_target = np.array([50.0, 100.0])
        with self.assertRaises(ValueError):
            sim_to_distance_modulus(z_target, a_sim, t_Gyr, t_start)


class TestTodayTolGuard(unittest.TestCase):
    """Test 5: ValueError when last absolute time is not ~13.8 Gyr."""

    def test_wrong_end_time_raises_value_error(self):
        """
        t_Gyr ending at 3.0 Gyr relative, with t_start=5.8, gives absolute
        end of 8.8 Gyr — well outside the 0.2 Gyr tolerance around 13.8.
        """
        t_Gyr_bad = np.linspace(0.0, 3.0, 100)   # ends at 5.8+3.0 = 8.8 Gyr
        a_bad = np.linspace(1.0, 1.4, 100)        # plausible monotonic a

        with self.assertRaises(ValueError) as ctx:
            sim_to_distance_modulus(
                z_target=np.array([0.1, 0.3]),
                a=a_bad,
                t_Gyr=t_Gyr_bad,
                t_start_Gyr=5.8,
            )

        msg = str(ctx.exception)
        # Message should mention 13.8 so caller knows what to fix
        self.assertIn("13.8", msg,
                      "ValueError message should mention 13.8 Gyr")
        self.assertIn("t_start_Gyr", msg,
                      "ValueError message should mention t_start_Gyr")

    def test_end_time_at_16_raises_value_error(self):
        """Absolute end time of 16.8 Gyr (a common off-by-3 mistake) → ValueError."""
        t_Gyr_bad = np.linspace(0.0, 11.0, 100)  # ends at 5.8+11.0 = 16.8 Gyr
        a_bad = np.linspace(1.0, 1.8, 100)

        with self.assertRaises(ValueError):
            sim_to_distance_modulus(
                z_target=np.array([0.1, 0.3]),
                a=a_bad,
                t_Gyr=t_Gyr_bad,
                t_start_Gyr=5.8,
            )

    def test_valid_today_does_not_raise(self):
        """No error when last absolute time == 13.8 Gyr exactly."""
        a_sim, t_Gyr, t_start = _make_lcdm_inputs()
        # Should not raise
        try:
            sim_to_distance_modulus(
                z_target=np.linspace(0.05, 0.5, 10),
                a=a_sim,
                t_Gyr=t_Gyr,
                t_start_Gyr=t_start,
            )
        except ValueError as e:
            self.fail(f"Unexpected ValueError for valid inputs: {e}")


class TestInputValidation(unittest.TestCase):
    """Tests 6-7: bad input arrays raise ValueError."""

    def test_nonpositive_a_raises(self):
        """a containing zeros should raise ValueError during renormalization."""
        a_bad = np.array([1.0, 1.1, 0.0, 1.2])   # zero in the middle
        t_Gyr = np.linspace(0.0, _T_END_GYR - _T_START_GYR, 4)

        with self.assertRaises(ValueError):
            sim_to_distance_modulus(
                z_target=np.array([0.1]),
                a=a_bad,
                t_Gyr=t_Gyr,
                t_start_Gyr=_T_START_GYR,
            )

    def test_mismatched_lengths_raises(self):
        """a and t_Gyr of different lengths → ValueError."""
        a = np.ones(50)
        t_Gyr = np.linspace(0.0, 8.0, 60)   # different length

        with self.assertRaises(ValueError) as ctx:
            sim_to_distance_modulus(
                z_target=np.array([0.1]),
                a=a,
                t_Gyr=t_Gyr,
                t_start_Gyr=_T_START_GYR,
            )
        self.assertIn("same length", str(ctx.exception))

    def test_2d_a_raises(self):
        """2-D a array → ValueError."""
        a_2d = np.ones((10, 2))
        t_Gyr = np.linspace(0.0, 8.0, 10)

        with self.assertRaises(ValueError):
            sim_to_distance_modulus(
                z_target=np.array([0.1]),
                a=a_2d,
                t_Gyr=t_Gyr,
                t_start_Gyr=_T_START_GYR,
            )


class TestDLMonotonicity(unittest.TestCase):
    """Test 8: luminosity distance (proxied via mu) should be monotonically
    increasing with z over the snapshots."""

    def test_mu_monotonically_increasing_with_z(self):
        """mu(z) should increase monotonically with z (farther = dimmer)."""
        a_sim, t_Gyr, t_start = _make_lcdm_inputs()

        # Dense z grid well inside the covered range
        z_target = np.linspace(0.05, 0.7, 100)
        result = sim_to_distance_modulus(z_target, a_sim, t_Gyr, t_start)

        mu = result["mu"]
        z = result["z"]

        # mu should be increasing
        diffs = np.diff(mu)
        self.assertTrue(
            np.all(diffs >= 0),
            f"mu(z) is not monotonically non-decreasing. "
            f"Negative diffs at z: {z[1:][diffs < 0]}"
        )


if __name__ == "__main__":
    unittest.main()
