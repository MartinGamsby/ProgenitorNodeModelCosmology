"""
Unit tests for physical constants and parameter classes
Testing basic physics before complex simulations
"""

import unittest
import numpy as np
from cosmo.constants import CosmologicalConstants, LambdaCDMParameters, ExternalNodeParameters, SimulationParameters


class TestCosmologicalConstants(unittest.TestCase):
    """Test physical constants and unit conversions"""

    def setUp(self):
        self.const = CosmologicalConstants()

    def test_gravitational_constant(self):
        """G should be 6.674e-11 m^3/(kg*s^2)"""
        self.assertAlmostEqual(self.const.G, 6.674e-11, places=13)

    def test_speed_of_light(self):
        """c should be 299792458 m/s"""
        self.assertEqual(self.const.c, 299792458)

    def test_gpc_to_meters(self):
        """1 Gpc should be ~3.086e25 meters"""
        # 1 Gpc = 1e9 pc = 1e9 * 3.086e16 m = 3.086e25 m
        expected = 3.086e25
        self.assertAlmostEqual(self.const.Gpc_to_m, expected, delta=expected*0.01)

    def test_gyr_to_seconds(self):
        """1 Gyr should be ~3.154e16 seconds"""
        # 1 Gyr = 1e9 years * 365.25 days/year * 24 hours/day * 3600 s/hour
        expected = 1e9 * 365.25 * 24 * 3600
        self.assertAlmostEqual(self.const.Gyr_to_s, expected, delta=expected*0.01)

    def test_observable_universe_mass(self):
        """M_observable should be ~1e53 kg"""
        self.assertGreater(self.const.M_observable, 1e52)
        self.assertLess(self.const.M_observable, 1e54)


class TestLambdaCDMParameters(unittest.TestCase):
    """Test ΛCDM cosmological parameters"""

    def setUp(self):
        self.lcdm = LambdaCDMParameters()

    def test_hubble_constant_si_units(self):
        """H0 in SI should be ~2.3e-18 s^-1"""
        # H0 = 70 km/s/Mpc
        # Convert: 70 * 1000 m/s / (3.086e22 m) ≈ 2.27e-18 s^-1
        expected = 2.3e-18
        self.assertAlmostEqual(self.lcdm.H0, expected, delta=expected*0.1)

    def test_density_parameters_sum(self):
        """Ω_m + Ω_Λ should equal 1 (flat universe)"""
        total = self.lcdm.Omega_m + self.lcdm.Omega_Lambda
        self.assertAlmostEqual(total, 1.0, places=5)

    def test_omega_lambda(self):
        """Ω_Λ should be ~0.7"""
        self.assertAlmostEqual(self.lcdm.Omega_Lambda, 0.7, places=1)

    def test_omega_matter(self):
        """Ω_m should be ~0.3"""
        self.assertAlmostEqual(self.lcdm.Omega_m, 0.3, places=1)

    def test_hubble_at_present(self):
        """H(a=1) should equal H0"""
        H_present = self.lcdm.H_at_time(1.0)
        self.assertAlmostEqual(H_present, self.lcdm.H0, places=15)

    def test_hubble_at_early_time(self):
        """H(a=0.5) should be larger than H0 (matter dominated)"""
        H_early = self.lcdm.H_at_time(0.5)
        self.assertGreater(H_early, self.lcdm.H0)

    def test_hubble_at_future(self):
        """H(a=2) should approach H0*sqrt(Omega_Lambda) for late times"""
        H_future = self.lcdm.H_at_time(2.0)
        # At very late times, H → H0 * sqrt(Omega_Lambda)
        # At a=2, still some matter contribution, so H > H0*sqrt(Omega_Lambda)
        H_limit = self.lcdm.H0 * np.sqrt(self.lcdm.Omega_Lambda)
        self.assertGreater(H_future, H_limit * 0.9)


class TestExternalNodeParameters(unittest.TestCase):
    """Test External-Node model parameters"""

    def setUp(self):
        self.const = CosmologicalConstants()

    def test_default_mass(self):
        """Default M_ext should be ~5e55 kg"""
        params = ExternalNodeParameters()
        self.assertAlmostEqual(params.M_ext, 5e55, delta=1e55)

    def test_default_spacing(self):
        """Default S should be ~31.6 Gpc in meters"""
        params = ExternalNodeParameters()
        expected_gpc = 31.6
        expected_m = expected_gpc * self.const.Gpc_to_m
        self.assertAlmostEqual(params.S, expected_m, delta=expected_m*0.1)

    def test_omega_lambda_eff_calculation(self):
        """Ω_Λ_eff = G*M_ext/(S^3*H0^2) should be computed correctly"""
        params = ExternalNodeParameters()
        lcdm = LambdaCDMParameters()

        # Manual calculation
        expected = self.const.G * params.M_ext / (params.S**3 * lcdm.H0**2)

        self.assertAlmostEqual(params.Omega_Lambda_eff, expected, places=5)

    def test_custom_mass_and_spacing(self):
        """Should accept custom M_ext and S values"""
        M_custom = 8e55
        S_custom_gpc = 24.0
        S_custom_m = S_custom_gpc * self.const.Gpc_to_m

        params = ExternalNodeParameters(M_ext=M_custom, S=S_custom_m)

        self.assertEqual(params.M_ext, M_custom)
        self.assertAlmostEqual(params.S / self.const.Gpc_to_m, S_custom_gpc, places=1)

    def test_calculate_required_spacing(self):
        """calculate_required_spacing should return S for target Ω_Λ"""
        params = ExternalNodeParameters(M_ext=5e55)

        # Request Ω_Λ = 0.7 (actual parameter name is Omega_Lambda_target)
        S_required = params.calculate_required_spacing(Omega_Lambda_target=0.7)

        # Verify: G*M/(S^3*H0^2) = 0.7
        lcdm = LambdaCDMParameters()
        Omega_check = self.const.G * params.M_ext / (S_required**3 * lcdm.H0**2)

        self.assertAlmostEqual(Omega_check, 0.7, places=3)


class TestSimulationParameters(unittest.TestCase):
    """Test unified simulation parameter class"""

    def test_default_parameters(self):
        """Default parameters should be reasonable"""
        params = SimulationParameters()

        self.assertGreater(params.M_value, 0)
        self.assertGreater(params.S_value, 0)
        self.assertGreater(params.n_particles, 0)
        self.assertGreater(params.t_start_Gyr, 0)
        self.assertGreater(params.t_duration_Gyr, 0)

    def test_custom_parameters(self):
        """Should accept custom parameter values"""
        params = SimulationParameters(
            M_value=800,
            S_value=24.0,
            n_particles=500,
            t_start_Gyr=11.0,
            t_duration_Gyr=5.0,
            seed=123
        )

        self.assertEqual(params.M_value, 800)
        self.assertEqual(params.S_value, 24.0)
        self.assertEqual(params.n_particles, 500)
        self.assertEqual(params.t_start_Gyr, 11.0)
        self.assertEqual(params.t_duration_Gyr, 5.0)
        self.assertEqual(params.seed, 123)

    def test_external_params_created(self):
        """Should create ExternalNodeParameters from M and S"""
        params = SimulationParameters(M_value=800, S_value=24.0)

        self.assertIsNotNone(params.external_params)
        self.assertIsInstance(params.external_params, ExternalNodeParameters)

    def test_t_end_calculation(self):
        """t_end_Gyr should be t_start + t_duration"""
        params = SimulationParameters(t_start_Gyr=10.0, t_duration_Gyr=6.0)

        self.assertAlmostEqual(params.t_end_Gyr, 16.0, places=10)


if __name__ == '__main__':
    unittest.main()
