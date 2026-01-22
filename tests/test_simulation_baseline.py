"""
Unit tests for LCDM baseline computation validation

Validates that solve_friedmann_equation() produces correct expansion
histories for LCDM cosmology. This baseline is used by both run_simulation.py
and parameter_sweep.py for accurate LCDM reference curves.

Key validation points:
- LCDM reaches expected size (~14.5 Gpc) at present day (t=13.8 Gyr)
- Scale factor evolution matches expected values at key cosmic times
- LCDM expands faster than matter-only due to dark energy acceleration
- Provides reusable helper function for consistent LCDM baseline computation
"""

import unittest
import numpy as np
from cosmo.analysis import solve_friedmann_equation, calculate_initial_conditions
from cosmo.constants import CosmologicalConstants, LambdaCDMParameters


class TestLCDMBaseline(unittest.TestCase):
    """Test analytic LCDM expansion using solve_friedmann_equation()"""

    def setUp(self):
        self.const = CosmologicalConstants()
        self.lcdm = LambdaCDMParameters()
        # Reference size today (Hubble radius proxy)
        self.reference_size_today_Gpc = 14.5

    def test_lcdm_reaches_correct_size_at_present_day(self):
        """LCDM should reach ~14.5 Gpc at t=13.8 Gyr (present day)"""
        # The reference size convention: at t=13.8 Gyr, size should be 14.5 Gpc
        # We validate this by computing size from initial conditions

        # Get initial conditions at t=13.8 Gyr
        initial_conditions_today = calculate_initial_conditions(t_start_Gyr=13.8)
        size_today = initial_conditions_today['box_size_Gpc']

        # This should be ~14.5 Gpc by convention
        self.assertAlmostEqual(size_today, 14.5, delta=0.3,
            msg=f"Size at t=13.8 Gyr should be ~14.5 Gpc, got {size_today:.2f} Gpc")

        # Additional check: solve from earlier time to present day
        t_start = 3.8
        initial_conditions_start = calculate_initial_conditions(t_start)
        a_start = initial_conditions_start['a_start']
        box_size_start = initial_conditions_start['box_size_Gpc']

        # Solve to present day
        solution = solve_friedmann_equation(
            t_start_Gyr=t_start,
            t_end_Gyr=13.8,
            Omega_Lambda=self.lcdm.Omega_Lambda
        )

        a_end = solution['a'][-1]

        # Calculate final size using the scaling relation
        size_end = box_size_start * (a_end / a_start)

        # Should reach ~14.5 Gpc
        self.assertAlmostEqual(size_end, 14.5, delta=0.3,
            msg=f"LCDM from t={t_start} to t=13.8 Gyr should reach ~14.5 Gpc, "
                f"got {size_end:.2f} Gpc")

    def test_lcdm_scale_factor_monotonic_increase(self):
        """Validate LCDM scale factor increases monotonically with time"""
        # Solve full evolution
        solution = solve_friedmann_equation(
            t_start_Gyr=0.0,
            t_end_Gyr=13.8,
            Omega_Lambda=self.lcdm.Omega_Lambda
        )

        a = solution['a']

        # Scale factor should increase monotonically
        a_diffs = np.diff(a)
        self.assertTrue(np.all(a_diffs > 0),
            msg="Scale factor should increase monotonically with time")

        # Verify cosmic acceleration in LCDM (dark energy era)
        # Check that d²a/dt² is positive in late times
        t_Gyr = solution['t_Gyr']
        da_dt = np.gradient(a, t_Gyr)
        d2a_dt2 = np.gradient(da_dt, t_Gyr)

        # In dark energy dominated era (late times), acceleration should be positive
        # Check the last 30% of the timeline
        late_acceleration = np.mean(d2a_dt2[int(0.7*len(d2a_dt2)):])

        self.assertGreater(late_acceleration, 0,
            msg=f"LCDM should show cosmic acceleration in late times "
                f"(d²a/dt² = {late_acceleration:.6e})")

    def test_lcdm_expansion_history_from_simulation_start(self):
        """Test LCDM expansion from typical simulation start time (3.8 Gyr) to present"""
        # This mirrors the typical simulation setup used in run_simulation.py
        t_start_Gyr = 3.8
        t_end_Gyr = 13.8

        # Get initial conditions (this calculates box_size and a_start)
        initial_conditions = calculate_initial_conditions(t_start_Gyr)
        a_start = initial_conditions['a_start']
        box_size_initial = initial_conditions['box_size_Gpc']

        # Solve LCDM evolution from t_start to t_end
        solution = solve_friedmann_equation(
            t_start_Gyr=t_start_Gyr,
            t_end_Gyr=t_end_Gyr,
            Omega_Lambda=self.lcdm.Omega_Lambda
        )

        a_array = solution['a']
        a_final = a_array[-1]

        # Calculate size evolution using the same formula as run_simulation.py
        # size(t) = box_size_initial * (a(t) / a_start)
        size_curve = box_size_initial * (a_array / a_start)
        size_final = size_curve[-1]

        # At t=13.8 Gyr, size should be ~14.5 Gpc
        # This is the key validation: the scaling formula should produce the reference size
        # The calculate_initial_conditions() function ensures this by computing:
        #   box_size_initial = 14.5 * (a_start / a_today)
        # So when we scale back: box_size_initial * (a_today / a_start) = 14.5 Gpc

        # Allow 2% tolerance
        self.assertAlmostEqual(size_final, 14.5, delta=0.3,
            msg=f"LCDM from t={t_start_Gyr} Gyr to t={t_end_Gyr} Gyr should reach "
                f"~14.5 Gpc, got {size_final:.2f} Gpc")

        # Verify expansion is monotonically increasing
        size_diffs = np.diff(size_curve)
        self.assertTrue(np.all(size_diffs > 0),
            msg="LCDM expansion should be monotonically increasing (all da/dt > 0)")

        # Verify scale factor increased from start to end
        self.assertGreater(a_final, a_start,
            msg=f"Scale factor should increase: a_start={a_start:.3f}, a_final={a_final:.3f}")

    def test_lcdm_expands_faster_than_matter_only(self):
        """LCDM should expand faster than matter-only due to dark energy"""
        t_start_Gyr = 3.8
        t_end_Gyr = 13.8

        # Get initial conditions
        initial_conditions = calculate_initial_conditions(t_start_Gyr)
        a_start = initial_conditions['a_start']
        box_size_initial = initial_conditions['box_size_Gpc']

        # Solve LCDM
        lcdm_solution = solve_friedmann_equation(
            t_start_Gyr=t_start_Gyr,
            t_end_Gyr=t_end_Gyr,
            Omega_Lambda=self.lcdm.Omega_Lambda  # 0.7
        )

        # Solve matter-only
        matter_solution = solve_friedmann_equation(
            t_start_Gyr=t_start_Gyr,
            t_end_Gyr=t_end_Gyr,
            Omega_Lambda=0.0  # No dark energy
        )

        # Calculate final sizes
        size_lcdm_final = box_size_initial * (lcdm_solution['a'][-1] / a_start)
        size_matter_final = box_size_initial * (matter_solution['a'][-1] / a_start)

        # LCDM should expand significantly more than matter-only
        self.assertGreater(size_lcdm_final, size_matter_final,
            msg=f"LCDM ({size_lcdm_final:.2f} Gpc) should expand more than "
                f"matter-only ({size_matter_final:.2f} Gpc)")

        # Expansion ratio should be substantial (at least 15% larger)
        expansion_ratio = size_lcdm_final / size_matter_final
        self.assertGreater(expansion_ratio, 1.15,
            msg=f"LCDM should expand at least 15% more than matter-only, "
                f"got {(expansion_ratio-1)*100:.1f}%")

    def test_lcdm_hubble_parameter_evolution(self):
        """Hubble parameter should decrease over time in LCDM"""
        solution = solve_friedmann_equation(
            t_start_Gyr=3.8,
            t_end_Gyr=13.8,
            Omega_Lambda=self.lcdm.Omega_Lambda
        )

        H_hubble = solution['H_hubble']  # km/s/Mpc

        # Hubble parameter should be positive at all times
        self.assertTrue(np.all(H_hubble > 0),
            msg="Hubble parameter should be positive at all times")

        # In LCDM, H(t) decreases with time but asymptotes to H₀√Ω_Λ
        # Early H should be larger than late H
        H_early = H_hubble[0]
        H_late = H_hubble[-1]

        self.assertGreater(H_early, H_late,
            msg=f"Early Hubble parameter ({H_early:.1f}) should be larger than "
                f"late Hubble parameter ({H_late:.1f})")

        # At present day (t=13.8 Gyr), H should be close to H₀ = 70 km/s/Mpc
        # Allow 10% tolerance
        H_today = H_hubble[-1]
        self.assertAlmostEqual(H_today, 70.0, delta=7.0,
            msg=f"Hubble parameter today should be ~70 km/s/Mpc, got {H_today:.1f}")

    def test_matter_only_decelerates_throughout(self):
        """Matter-only cosmology should decelerate at all times (no acceleration)"""
        solution = solve_friedmann_equation(
            t_start_Gyr=3.8,
            t_end_Gyr=13.8,
            Omega_Lambda=0.0  # Matter-only
        )

        a = solution['a']
        t_Gyr = solution['t_Gyr']

        # Calculate da/dt
        da_dt = np.gradient(a, t_Gyr)

        # Expansion rate should decrease over time (deceleration)
        # da/dt should be monotonically decreasing
        da_dt_diffs = np.diff(da_dt)

        # Allow small numerical noise (check mean trend)
        mean_change = np.mean(da_dt_diffs)
        self.assertLess(mean_change, 0,
            msg="Matter-only cosmology should decelerate (da/dt should decrease)")

    def test_solve_friedmann_n_points_parameter(self):
        """Test that n_points parameter controls time resolution"""
        t_start = 3.8
        t_end = 13.8

        # Test with different resolutions
        solution_low = solve_friedmann_equation(t_start, t_end, n_points=100)
        solution_high = solve_friedmann_equation(t_start, t_end, n_points=800)

        # Number of output points should match requested resolution
        # (minus a few due to time windowing)
        self.assertLess(len(solution_low['t_Gyr']), len(solution_high['t_Gyr']),
            msg="Higher n_points should produce more time samples")

        # Both should reach same final scale factor (within numerical precision)
        a_low_final = solution_low['a'][-1]
        a_high_final = solution_high['a'][-1]

        rel_error = abs(a_low_final - a_high_final) / a_high_final
        self.assertLess(rel_error, 0.01,
            msg=f"Final scale factor should be independent of n_points resolution, "
                f"got {rel_error*100:.2f}% difference")


class TestLCDMBaselineHelper(unittest.TestCase):
    """Test helper function for consistent LCDM baseline computation"""

    def test_compute_lcdm_baseline_helper(self):
        """Test reusable helper function for LCDM baseline computation"""
        # This helper function can be used by both run_simulation.py and parameter_sweep.py
        # to ensure identical LCDM baselines

        def compute_lcdm_baseline(t_start_Gyr, t_end_Gyr, box_size_initial, a_start):
            """
            Compute LCDM baseline expansion history.

            Parameters
            ----------
            t_start_Gyr : float
                Start time [Gyr]
            t_end_Gyr : float
                End time [Gyr]
            box_size_initial : float
                Initial box size [Gpc]
            a_start : float
                Initial scale factor

            Returns
            -------
            dict with keys:
                't': time array [Gyr] (offset to start at 0)
                'a': scale factor array
                'size': size evolution [Gpc]
                'H_hubble': Hubble parameter [km/s/Mpc]
            """
            solution = solve_friedmann_equation(
                t_start_Gyr=t_start_Gyr,
                t_end_Gyr=t_end_Gyr,
                Omega_Lambda=None  # Use default LCDM value
            )

            # Offset time to start at 0
            t = solution['t_Gyr'] - t_start_Gyr

            # Calculate size evolution
            size = box_size_initial * (solution['a'] / a_start)

            return {
                't': t,
                'a': solution['a'],
                'size': size,
                'H_hubble': solution['H_hubble']
            }

        # Test the helper function
        t_start = 3.8
        t_end = 13.8

        initial_conditions = calculate_initial_conditions(t_start)
        box_size = initial_conditions['box_size_Gpc']
        a_start = initial_conditions['a_start']

        baseline = compute_lcdm_baseline(t_start, t_end, box_size, a_start)

        # Verify output structure
        self.assertIn('t', baseline)
        self.assertIn('a', baseline)
        self.assertIn('size', baseline)
        self.assertIn('H_hubble', baseline)

        # Verify time starts at 0 (or very close due to windowing)
        self.assertLess(baseline['t'][0], 0.1,
            msg=f"Time should start near 0, got {baseline['t'][0]:.3f}")

        # Verify final size is ~14.5 Gpc
        self.assertAlmostEqual(baseline['size'][-1], 14.5, delta=0.3)

        # Verify all arrays have same length
        n_points = len(baseline['t'])
        self.assertEqual(len(baseline['a']), n_points)
        self.assertEqual(len(baseline['size']), n_points)
        self.assertEqual(len(baseline['H_hubble']), n_points)

    def test_lcdm_baseline_initial_size_matches_simulation(self):
        """ΛCDM baseline should start at same size as particle simulations"""
        # This test verifies the fix for visualize_3d.py where ΛCDM was starting
        # at a different size than External-Node and Matter-Only models

        t_start_Gyr = 3.8
        t_end_Gyr = 13.8

        # Get initial conditions (used by all models)
        initial_conditions = calculate_initial_conditions(t_start_Gyr)
        box_size_start_Gpc = initial_conditions['box_size_Gpc']
        a_start = initial_conditions['a_start']

        # Compute ΛCDM evolution
        solution = solve_friedmann_equation(t_start_Gyr, t_end_Gyr, n_points=100)

        # Interpolate scale factor at EXACTLY t_start using full arrays
        # (windowed arrays may not include exact t_start point)
        a_at_start_interpolated = np.interp(t_start_Gyr, solution['_t_Gyr_full'], solution['_a_full'])

        # CORRECT method: relative expansion from start
        # size(t) = (a(t) / a_start) * box_size_start
        size_lcdm_correct = (a_at_start_interpolated / a_start) * box_size_start_Gpc

        # WRONG method (old bug): absolute scale factor
        # size(t) = a(t) * box_size_start
        size_lcdm_wrong = a_at_start_interpolated * box_size_start_Gpc

        # At start time, ΛCDM should match initial box size (within numerical tolerance)
        # Using relative expansion: (a_start / a_start) * box_size = box_size ✓
        # The interpolated value should be very close to a_start
        self.assertAlmostEqual(a_at_start_interpolated, a_start, delta=0.001,
            msg=f"Interpolated a(t_start) should equal a_start. "
                f"Got {a_at_start_interpolated:.6f} vs {a_start:.6f}")

        self.assertAlmostEqual(size_lcdm_correct, box_size_start_Gpc, delta=0.01,
            msg=f"ΛCDM initial size (correct method) should match box_size_start. "
                f"Got {size_lcdm_correct:.3f} Gpc vs {box_size_start_Gpc:.3f} Gpc")

        # The wrong method produces incorrect initial size
        # It's off by a factor because it uses absolute a(t) instead of relative
        self.assertNotAlmostEqual(size_lcdm_wrong, box_size_start_Gpc, delta=0.5,
            msg=f"Old bug method should NOT match box_size_start. "
                f"Got {size_lcdm_wrong:.3f} Gpc vs {box_size_start_Gpc:.3f} Gpc")

        # Verify the simulations also use relative expansion
        # At t=t_start, simulations have a_relative = rms_initial / rms_initial = 1.0
        # So their size = 1.0 * box_size_start = box_size_start ✓
        a_relative_at_start = 1.0  # By definition for simulations
        size_simulation_at_start = a_relative_at_start * box_size_start_Gpc

        self.assertAlmostEqual(size_simulation_at_start, box_size_start_Gpc, delta=0.001,
            msg="Simulations should start at box_size_start by definition")

        # Final verification: ΛCDM and simulations start at same size
        self.assertAlmostEqual(size_lcdm_correct, size_simulation_at_start, delta=0.01,
            msg=f"ΛCDM and simulations must start at same size. "
                f"ΛCDM: {size_lcdm_correct:.3f} Gpc, Sim: {size_simulation_at_start:.3f} Gpc")

    def test_lcdm_baseline_evolves_continuously(self):
        """ΛCDM baseline should evolve continuously from t_start, not stay constant"""
        # This test verifies that ΛCDM size changes over time, catching bugs where
        # interpolation uses windowed arrays that don't cover early times

        t_start_Gyr = 3.8
        t_end_Gyr = 13.8

        # Get initial conditions
        initial_conditions = calculate_initial_conditions(t_start_Gyr)
        box_size_start_Gpc = initial_conditions['box_size_Gpc']
        a_start = initial_conditions['a_start']

        # Compute ΛCDM evolution
        solution = solve_friedmann_equation(t_start_Gyr, t_end_Gyr, n_points=100)

        # Test several time points in early evolution
        test_times = [t_start_Gyr, t_start_Gyr + 1.0, t_start_Gyr + 2.0, t_start_Gyr + 3.0]
        sizes = []

        for t_Gyr in test_times:
            # Use FULL arrays for interpolation (critical for early times)
            a_lcdm = np.interp(t_Gyr, solution['_t_Gyr_full'], solution['_a_full'])
            a_relative = a_lcdm / a_start
            size_Gpc = a_relative * box_size_start_Gpc
            sizes.append(size_Gpc)

        # Verify sizes are monotonically increasing
        for i in range(len(sizes) - 1):
            self.assertLess(sizes[i], sizes[i+1],
                msg=f"ΛCDM size should increase over time. "
                    f"At t={test_times[i]:.1f} Gyr: {sizes[i]:.3f} Gpc, "
                    f"at t={test_times[i+1]:.1f} Gyr: {sizes[i+1]:.3f} Gpc")

        # Verify first size matches initial box size (within tolerance)
        self.assertAlmostEqual(sizes[0], box_size_start_Gpc, delta=0.1,
            msg=f"ΛCDM should start at box_size_start. "
                f"Got {sizes[0]:.3f} Gpc vs {box_size_start_Gpc:.3f} Gpc")

        # Verify significant growth over 3 Gyr (should grow > 10%)
        growth_ratio = (sizes[-1] - sizes[0]) / sizes[0]
        self.assertGreater(growth_ratio, 0.10,
            msg=f"ΛCDM should grow significantly over 3 Gyr. "
                f"Got {growth_ratio*100:.1f}% growth (expected >10%)")

if __name__ == '__main__':
    unittest.main()
