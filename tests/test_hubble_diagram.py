"""
Unit tests for cosmo.hubble_diagram module.

Tests:
1. fit_offset recovers an injected constant offset exactly (within 1e-9).
2. Perfect match (no noise): chi2 ~ 0, R2 ~ 1 after offset application.
3. Unit-sigma gaussian noise: chi2_dof ~ 1 (0.5 < chi2_dof < 1.8, fixed seed).
4. compare_all_models returns three keys with all documented fields, correct dof.
5. external_node with Omega_Lambda_eff=0.7 gives chi2 nearly equal to lcdm
   on LCDM-generated data (within a small tolerance).

All synthetic data is generated from LCDM model_distance_modulus — no real
Pantheon+ file is needed.
"""

import types
import unittest

import numpy as np

from cosmo.distances import model_distance_modulus
from cosmo.hubble_diagram import compare_all_models, evaluate_model, fit_offset


# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------

def _make_lcdm_data(n: int = 200, seed: int = 0) -> tuple:
    """
    Build synthetic (z, mu_lcdm, sigma) arrays using LCDM model_distance_modulus.

    Returns (z, mu_lcdm, sigma) where sigma is uniform 0.15 mag.
    """
    rng = np.random.default_rng(seed)
    z = np.sort(rng.uniform(0.02, 2.0, n))
    mu_lcdm = model_distance_modulus(z, "lcdm")
    sigma = np.full(n, 0.15)
    return z, mu_lcdm, sigma


def _make_forced_sim_params(Omega_Lambda_eff: float):
    """Return a minimal namespace with external_params.Omega_Lambda_eff set."""
    sim = types.SimpleNamespace()
    sim.external_params = types.SimpleNamespace()
    sim.external_params.Omega_Lambda_eff = Omega_Lambda_eff
    return sim


# ---------------------------------------------------------------------------
# Test 1: fit_offset recovers a known injected constant offset
# ---------------------------------------------------------------------------

class TestFitOffsetRecovery(unittest.TestCase):
    """fit_offset must recover a known injected constant offset exactly."""

    def test_equal_sigma_recovers_offset(self):
        """
        With equal sigma, DeltaM == weighted mean of residuals == simple mean.
        Injected offset 5.0 must be recovered within 1e-9.
        """
        z, mu_model, sigma = _make_lcdm_data(n=100, seed=1)
        injected = 5.0
        mu_obs = mu_model + injected

        DeltaM, mu_fit = fit_offset(mu_obs, mu_model, sigma)

        self.assertAlmostEqual(DeltaM, injected, delta=1e-9,
                               msg=f"DeltaM={DeltaM} != injected={injected}")

    def test_unequal_sigma_recovers_offset(self):
        """
        With varying sigma, offset is still recovered (weighted mean == true
        offset when there is no noise, regardless of weights).
        """
        z, mu_model, _ = _make_lcdm_data(n=50, seed=2)
        injected = -3.25
        mu_obs = mu_model + injected
        rng = np.random.default_rng(42)
        sigma = rng.uniform(0.05, 0.5, len(z))

        DeltaM, mu_fit = fit_offset(mu_obs, mu_model, sigma)

        self.assertAlmostEqual(DeltaM, injected, delta=1e-9,
                               msg=f"DeltaM={DeltaM} != injected={injected}")

    def test_mu_fit_equals_mu_obs_after_noiseless_offset(self):
        """mu_fit must exactly equal mu_obs when there is no noise."""
        z, mu_model, sigma = _make_lcdm_data(n=30, seed=3)
        injected = 2.71828
        mu_obs = mu_model + injected

        _, mu_fit = fit_offset(mu_obs, mu_model, sigma)

        np.testing.assert_allclose(mu_fit, mu_obs, atol=1e-9,
                                   err_msg="mu_fit should equal mu_obs (no noise)")

    def test_large_positive_offset(self):
        """Large positive offset is recovered faithfully."""
        z, mu_model, sigma = _make_lcdm_data(n=50)
        injected = 50.0
        mu_obs = mu_model + injected

        DeltaM, _ = fit_offset(mu_obs, mu_model, sigma)
        self.assertAlmostEqual(DeltaM, injected, delta=1e-9)

    def test_negative_offset(self):
        """Negative offset is recovered faithfully."""
        z, mu_model, sigma = _make_lcdm_data(n=50)
        injected = -10.0
        mu_obs = mu_model + injected

        DeltaM, _ = fit_offset(mu_obs, mu_model, sigma)
        self.assertAlmostEqual(DeltaM, injected, delta=1e-9)


# ---------------------------------------------------------------------------
# Test 2: Perfect match → chi2 ≈ 0, R2 ≈ 1
# ---------------------------------------------------------------------------

class TestPerfectMatchStatistics(unittest.TestCase):
    """With a pure constant offset (no noise), chi2 and R2 must be ideal."""

    def test_chi2_near_zero_for_noiseless_data(self):
        """
        mu_obs = mu_model + constant, no noise.
        After offset subtraction, mu_fit == mu_obs exactly, so chi2 ~ 0.
        """
        z, mu_model, sigma = _make_lcdm_data(n=100, seed=10)
        mu_obs = mu_model + 3.0

        result = evaluate_model(z, mu_obs, sigma, model="lcdm")

        self.assertAlmostEqual(result["chi2"], 0.0, delta=1e-9,
                               msg=f"chi2={result['chi2']:.2e} should be ~0 for noiseless data")

    def test_r2_near_one_for_noiseless_data(self):
        """
        mu_obs = mu_model + constant, no noise.
        After offset, model explains all variance → R2 ~ 1.
        """
        z, mu_model, sigma = _make_lcdm_data(n=100, seed=10)
        mu_obs = mu_model + 3.0

        result = evaluate_model(z, mu_obs, sigma, model="lcdm")

        self.assertAlmostEqual(result["R2"], 1.0, delta=1e-9,
                               msg=f"R2={result['R2']:.6f} should be ~1 for noiseless data")

    def test_residuals_near_zero_for_noiseless_data(self):
        """All residuals should be near zero for noiseless data."""
        z, mu_model, sigma = _make_lcdm_data(n=50, seed=11)
        mu_obs = mu_model + 1.5

        result = evaluate_model(z, mu_obs, sigma, model="lcdm")

        max_residual = np.max(np.abs(result["residuals"]))
        self.assertLess(max_residual, 1e-9,
                        f"Max residual {max_residual:.2e} should be < 1e-9 for noiseless data")

    def test_dof_is_n_minus_one(self):
        """dof must equal n - 1 (one free parameter: DeltaM)."""
        z, mu_model, sigma = _make_lcdm_data(n=77, seed=12)
        mu_obs = mu_model + 0.0

        result = evaluate_model(z, mu_obs, sigma, model="lcdm")

        self.assertEqual(result["dof"], 76)


# ---------------------------------------------------------------------------
# Test 3: Unit-sigma gaussian noise → chi2_dof ≈ 1
# ---------------------------------------------------------------------------

class TestNoisyDataChiSquared(unittest.TestCase):
    """With unit-sigma gaussian noise, chi2_dof should be ~ 1."""

    def test_chi2_dof_near_one_with_unit_sigma_noise(self):
        """
        Inject unit-sigma gaussian noise. Expect 0.5 < chi2_dof < 1.8.

        n=200, seed=42 gives a statistically stable result while remaining
        well within the expected range.
        """
        n = 200
        seed = 42
        rng = np.random.default_rng(seed)

        z, mu_model, _ = _make_lcdm_data(n=n, seed=seed)
        sigma = np.ones(n)  # unit sigma
        noise = rng.standard_normal(n)  # unit-sigma noise
        mu_obs = mu_model + noise

        result = evaluate_model(z, mu_obs, sigma, model="lcdm")

        chi2_dof = result["chi2_dof"]
        self.assertGreater(chi2_dof, 0.5,
                           f"chi2_dof={chi2_dof:.3f} < 0.5 (unexpectedly low)")
        self.assertLess(chi2_dof, 1.8,
                        f"chi2_dof={chi2_dof:.3f} > 1.8 (unexpectedly high)")


# ---------------------------------------------------------------------------
# Test 4: compare_all_models returns correct structure
# ---------------------------------------------------------------------------

class TestCompareAllModels(unittest.TestCase):
    """compare_all_models must return the three models with documented fields."""

    _REQUIRED_KEYS = {"model", "DeltaM", "chi2", "dof", "chi2_dof", "R2", "residuals", "mu_fit"}

    def setUp(self):
        """Build a small synthetic dataset with external_node tuned to Omega_Lambda_eff=0.7."""
        n = 50
        seed = 5
        self.z, self.mu_lcdm, self.sigma = _make_lcdm_data(n=n, seed=seed)
        # Inject a small offset so DeltaM != 0
        self.mu_obs = self.mu_lcdm + 0.5
        self.data = {"z": self.z, "mu": self.mu_obs, "sigma": self.sigma, "n": n}
        # Use Omega_Lambda_eff = 0.7 so external_node can cover the full z-range
        self.sim_params = _make_forced_sim_params(0.7)

    def test_returns_three_model_keys(self):
        """Result must contain exactly external_node, lcdm, matter_only."""
        results = compare_all_models(self.data, sim_params=self.sim_params)
        self.assertSetEqual(set(results.keys()), {"external_node", "lcdm", "matter_only"})

    def test_each_model_has_required_fields(self):
        """Each model result must contain all documented fields."""
        results = compare_all_models(self.data, sim_params=self.sim_params)
        for model_name, result in results.items():
            missing = self._REQUIRED_KEYS - set(result.keys())
            self.assertFalse(missing,
                             f"Model {model_name!r} missing fields: {missing}")

    def test_model_field_matches_key(self):
        """The 'model' field inside each result must match its dict key."""
        results = compare_all_models(self.data, sim_params=self.sim_params)
        for model_name, result in results.items():
            self.assertEqual(result["model"], model_name)

    def test_dof_equals_n_minus_one(self):
        """dof must equal n - 1 for all models."""
        n = len(self.data["z"])
        results = compare_all_models(self.data, sim_params=self.sim_params)
        for model_name, result in results.items():
            self.assertEqual(result["dof"], n - 1,
                             f"Model {model_name!r}: dof={result['dof']} != {n - 1}")

    def test_residuals_shape_matches_data(self):
        """residuals and mu_fit arrays must have the same length as input data."""
        n = len(self.data["z"])
        results = compare_all_models(self.data, sim_params=self.sim_params)
        for model_name, result in results.items():
            self.assertEqual(len(result["residuals"]), n,
                             f"residuals length mismatch for {model_name!r}")
            self.assertEqual(len(result["mu_fit"]), n,
                             f"mu_fit length mismatch for {model_name!r}")

    def test_chi2_is_non_negative(self):
        """chi2 must be >= 0 for all models."""
        results = compare_all_models(self.data, sim_params=self.sim_params)
        for model_name, result in results.items():
            self.assertGreaterEqual(result["chi2"], 0.0,
                                    f"chi2 < 0 for model {model_name!r}")

    def test_r2_in_sensible_range(self):
        """R2 should be in a reasonable range (not wildly negative on small synthetic data)."""
        results = compare_all_models(self.data, sim_params=self.sim_params)
        for model_name, result in results.items():
            self.assertLessEqual(result["R2"], 1.0 + 1e-9,
                                 f"R2 > 1 for model {model_name!r}")


# ---------------------------------------------------------------------------
# Test 5: external_node with Omega_Lambda_eff=0.7 ~ LCDM on LCDM data
# ---------------------------------------------------------------------------

class TestExternalNodeMatchesLCDM(unittest.TestCase):
    """
    external_node with Omega_Lambda_eff forced to 0.7 is mathematically
    identical to LCDM (same Omega_m=0.3, same Omega_de=0.7).  On LCDM-
    generated data, both models must produce the same chi2 within 1e-6.
    """

    def test_chi2_nearly_equal_to_lcdm(self):
        """
        With Omega_Lambda_eff = 0.7, external_node chi2 == lcdm chi2 exactly
        (same mu_model curves, same data, same offset fit).
        """
        n = 150
        seed = 99
        rng = np.random.default_rng(seed)
        z, mu_lcdm, sigma_base = _make_lcdm_data(n=n, seed=seed)
        # Add unit-sigma noise so chi2 is non-trivial
        sigma = np.ones(n)
        noise = rng.standard_normal(n)
        mu_obs = mu_lcdm + noise

        sim_forced = _make_forced_sim_params(0.7)

        data = {"z": z, "mu": mu_obs, "sigma": sigma, "n": n}
        results = compare_all_models(data, sim_params=sim_forced)

        chi2_ext = results["external_node"]["chi2"]
        chi2_lcdm = results["lcdm"]["chi2"]

        self.assertAlmostEqual(
            chi2_ext, chi2_lcdm, delta=1e-6,
            msg=(f"external_node chi2={chi2_ext:.6f} should equal "
                 f"lcdm chi2={chi2_lcdm:.6f} when Omega_Lambda_eff=0.7")
        )

    def test_delta_m_nearly_equal_for_matched_config(self):
        """DeltaM must also be the same when models are identical."""
        n = 100
        seed = 77
        z, mu_lcdm, sigma = _make_lcdm_data(n=n, seed=seed)
        mu_obs = mu_lcdm + 0.3  # constant offset, no noise
        sim_forced = _make_forced_sim_params(0.7)

        data = {"z": z, "mu": mu_obs, "sigma": sigma, "n": n}
        results = compare_all_models(data, sim_params=sim_forced)

        self.assertAlmostEqual(
            results["external_node"]["DeltaM"],
            results["lcdm"]["DeltaM"],
            delta=1e-9,
            msg="DeltaM should be identical when Omega_Lambda_eff=0.7"
        )


# ---------------------------------------------------------------------------
# Edge-case and validation tests
# ---------------------------------------------------------------------------

class TestValidationErrors(unittest.TestCase):
    """fit_offset and evaluate_model must reject bad inputs."""

    def test_zero_sigma_raises(self):
        """Any zero sigma must raise ValueError."""
        mu_obs = np.array([43.0, 44.0, 45.0])
        mu_model = np.array([43.1, 43.9, 44.8])
        sigma = np.array([0.1, 0.0, 0.1])  # zero in middle

        with self.assertRaises(ValueError):
            fit_offset(mu_obs, mu_model, sigma)

    def test_negative_sigma_raises(self):
        """Negative sigma must raise ValueError."""
        mu_obs = np.array([43.0, 44.0, 45.0])
        mu_model = np.array([43.1, 43.9, 44.8])
        sigma = np.array([0.1, -0.05, 0.1])

        with self.assertRaises(ValueError):
            fit_offset(mu_obs, mu_model, sigma)

    def test_mismatched_lengths_raises(self):
        """Arrays of different lengths must raise ValueError."""
        mu_obs = np.array([43.0, 44.0, 45.0])
        mu_model = np.array([43.1, 43.9])  # wrong length
        sigma = np.array([0.1, 0.1, 0.1])

        with self.assertRaises(ValueError):
            fit_offset(mu_obs, mu_model, sigma)

    def test_single_point_raises(self):
        """n=1 must raise ValueError (dof = 0 is undefined)."""
        mu_obs = np.array([43.0])
        mu_model = np.array([43.1])
        sigma = np.array([0.1])

        with self.assertRaises(ValueError):
            fit_offset(mu_obs, mu_model, sigma)

    def test_empty_data_raises_descriptive_error(self):
        """
        Empty z (e.g. an over-aggressive z_min cut leaving no SNe) must raise a
        ValueError whose message names the empty/z_min problem, not a turnaround.
        """
        empty = np.array([])
        with self.assertRaises(ValueError) as ctx:
            evaluate_model(empty, empty, empty, model="lcdm")
        msg = str(ctx.exception).lower()
        self.assertTrue(
            "empty" in msg or "z_min" in msg,
            f"Empty-data error should mention empty/z_min, got: {ctx.exception}",
        )

    def test_two_points_is_valid_minimum(self):
        """n=2 must succeed (dof = 1, minimum valid)."""
        mu_obs = np.array([43.0, 44.0])
        mu_model = np.array([43.1, 43.9])
        sigma = np.array([0.1, 0.1])

        result = evaluate_model(
            np.array([0.1, 0.5]), mu_obs, sigma, model="lcdm"
        )
        self.assertEqual(result["dof"], 1)

    def test_evaluate_model_turnaround_raises_descriptive_error(self):
        """
        A closed external_node model with a turnaround inside the data z-range
        must raise ValueError with a descriptive message (not crash silently).

        Default SimulationParameters gives Omega_Lambda_eff ~ 2.55 which causes
        E^2 < 0 at z > ~0.32.  Requesting z up to 1.0 triggers the error.
        """
        from cosmo.constants import SimulationParameters
        sim = SimulationParameters()  # Omega_Lambda_eff ~ 2.55

        z = np.linspace(0.01, 1.0, 30)
        mu_obs = model_distance_modulus(np.linspace(0.01, 0.3, 30), "lcdm")
        sigma = np.full(30, 0.15)

        with self.assertRaises(ValueError):
            evaluate_model(z, mu_obs, sigma, model="external_node", sim_params=sim)


# ---------------------------------------------------------------------------
# Return-type sanity tests
# ---------------------------------------------------------------------------

class TestReturnTypes(unittest.TestCase):
    """evaluate_model and fit_offset must return the documented types."""

    def setUp(self):
        self.z, self.mu_model, self.sigma = _make_lcdm_data(n=20, seed=55)
        self.mu_obs = self.mu_model + 1.0

    def test_fit_offset_returns_float_and_ndarray(self):
        DeltaM, mu_fit = fit_offset(self.mu_obs, self.mu_model, self.sigma)
        self.assertIsInstance(DeltaM, float)
        self.assertIsInstance(mu_fit, np.ndarray)

    def test_evaluate_model_residuals_are_ndarray(self):
        result = evaluate_model(self.z, self.mu_obs, self.sigma, model="lcdm")
        self.assertIsInstance(result["residuals"], np.ndarray)
        self.assertIsInstance(result["mu_fit"], np.ndarray)

    def test_evaluate_model_scalars_are_float(self):
        result = evaluate_model(self.z, self.mu_obs, self.sigma, model="lcdm")
        for key in ("DeltaM", "chi2", "chi2_dof", "R2"):
            self.assertIsInstance(result[key], float,
                                  f"result[{key!r}] should be float")

    def test_evaluate_model_dof_is_int(self):
        result = evaluate_model(self.z, self.mu_obs, self.sigma, model="lcdm")
        self.assertIsInstance(result["dof"], int)


if __name__ == "__main__":
    unittest.main()
