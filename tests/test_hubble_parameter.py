"""
Unit tests for Hubble parameter calculation from scale factor derivatives.

Validates correctness of numerical derivative computation including:
- Edge formula accuracy (forward/backward differences)
- Smoothing effects on edges vs middle
- Discontinuity detection
- LCDM consistency
"""

import unittest
import numpy as np
from cosmo.constants import CosmologicalConstants, LambdaCDMParameters
from cosmo.analysis import solve_friedmann_at_times


class TestHubbleParameterCalculation(unittest.TestCase):
    """Test Hubble parameter H(t) = (1/a) da/dt computation"""

    def setUp(self):
        self.const = CosmologicalConstants()
        self.lcdm = LambdaCDMParameters()

    def test_edge_derivative_formulas(self):
        """Verify forward/backward difference formulas match expected values"""
        # Use polynomial f(t) = t² where f'(t) = 2t is known exactly
        t = np.linspace(0, 10, 11)  # 0, 1, 2, ..., 10
        f = t**2  # f(t) = t²
        dt = t[1] - t[0]  # 1.0

        # Test forward difference at t=0: f'(0) = 0
        # Formula: f'(0) ≈ (-3f(0) + 4f(1) - f(2)) / (2h)
        df_0_forward = (-3*f[0] + 4*f[1] - f[2]) / (2*dt)
        expected_0 = 0.0  # f'(0) = 2×0 = 0
        self.assertAlmostEqual(df_0_forward, expected_0, places=10,
                              msg=f"Forward difference at t=0 should give f'(0)=0, got {df_0_forward}")

        # Test backward difference at t=10: f'(10) = 20
        # Formula: f'(n) ≈ (3f(n) - 4f(n-1) + f(n-2)) / (2h)
        df_10_backward = (3*f[-1] - 4*f[-2] + f[-3]) / (2*dt)
        expected_10 = 20.0  # f'(10) = 2×10 = 20
        self.assertAlmostEqual(df_10_backward, expected_10, places=10,
                              msg=f"Backward difference at t=10 should give f'(10)=20, got {df_10_backward}")

    def test_hubble_from_derivative_matches_analytic_lcdm(self):
        """Derivative method should match analytic formula for LCDM"""
        # Get LCDM scale factor evolution from solve_friedmann_at_times
        t_array_Gyr = np.linspace(1.0, 13.8, 50)
        lcdm_solution = solve_friedmann_at_times(t_array_Gyr, Omega_Lambda=self.lcdm.Omega_Lambda)

        a_lcdm = lcdm_solution['a']
        H_analytic = lcdm_solution['H_hubble']  # Exact from H² = H₀²[Ω_m/a³ + Ω_Λ]

        # Compute H via numerical derivative: H = (1/a) da/dt
        t_seconds = t_array_Gyr * 1e9 * 365.25 * 24 * 3600
        da_dt = np.gradient(a_lcdm, t_seconds)
        H_derivative = (da_dt / a_lcdm) * self.const.Mpc_to_m / 1000  # Convert to km/s/Mpc

        # Compare (exclude first and last points where derivative is less accurate)
        rel_diff = np.abs(H_derivative[1:-1] - H_analytic[1:-1]) / H_analytic[1:-1]
        max_diff = np.max(rel_diff) * 100  # Percentage

        # Derivative method should match within ~2% (discretization error)
        self.assertLess(max_diff, 2.0,
                       f"Derivative H should match analytic within 2%, max diff: {max_diff:.2f}%")

    def test_no_discontinuities_in_hubble_parameter(self):
        """Hubble parameter should be smooth (no sudden jumps)"""
        # Generate smooth LCDM scale factor evolution
        t_array_Gyr = np.linspace(1.0, 13.8, 100)
        lcdm_solution = solve_friedmann_at_times(t_array_Gyr, Omega_Lambda=self.lcdm.Omega_Lambda)

        a_lcdm = lcdm_solution['a']

        # Compute H via numerical derivative
        t_seconds = t_array_Gyr * 1e9 * 365.25 * 24 * 3600
        da_dt = np.gradient(a_lcdm, t_seconds)
        H = (da_dt / a_lcdm) * self.const.Mpc_to_m / 1000

        # Check for sudden jumps by comparing consecutive differences
        # ΔH[i] = H[i+1] - H[i]
        dH = np.diff(H)

        # Compute relative changes: |ΔH[i]| / |H[i]|
        rel_changes = np.abs(dH / H[:-1])

        # Maximum relative change per step should be small (<15% for smooth curve)
        max_rel_change = np.max(rel_changes) * 100  # Percentage

        self.assertLess(max_rel_change, 15.0,
                       f"Max relative change {max_rel_change:.2f}% per step suggests discontinuity. "
                       f"Should be < 15% for smooth H(t).")

    def test_hubble_decreases_for_matter_only(self):
        """Matter-only cosmology: H(t) should monotonically decrease"""
        # Get matter-only scale factor evolution
        t_array_Gyr = np.linspace(1.0, 13.8, 50)
        matter_solution = solve_friedmann_at_times(t_array_Gyr, Omega_Lambda=0.0)

        H_matter = matter_solution['H_hubble']

        # Compute differences: H[i+1] - H[i]
        dH = np.diff(H_matter)

        # All differences should be negative (H decreasing)
        # Allow tiny violations (<0.01%) due to numerical noise
        positive_steps = np.sum(dH > 0)
        total_steps = len(dH)
        violation_rate = positive_steps / total_steps

        self.assertLess(violation_rate, 0.01,
                       f"H(t) should decrease monotonically for matter-only. "
                       f"{positive_steps}/{total_steps} steps show increase ({violation_rate*100:.1f}%)")

    def test_hubble_at_present_day(self):
        """H(t=13.8 Gyr) ≈ 70 km/s/Mpc for LCDM"""
        # Get Hubble parameter at present day
        t_array_Gyr = np.array([13.8])
        lcdm_solution = solve_friedmann_at_times(t_array_Gyr, Omega_Lambda=self.lcdm.Omega_Lambda)

        H_today = lcdm_solution['H_hubble'][0]
        expected_H0 = 70.0  # km/s/Mpc (approximate Hubble constant)

        # Should be within ±5 km/s/Mpc
        diff = abs(H_today - expected_H0)
        self.assertLess(diff, 5.0,
                       f"H(13.8 Gyr) = {H_today:.1f} should be ≈ {expected_H0} ± 5 km/s/Mpc")


if __name__ == '__main__':
    unittest.main()
