"""
Tests for hubble_diagram_nbody.py and the additive evaluate_precomputed.

Test groups
-----------
1. Unit: evaluate_precomputed vs evaluate_model — must produce identical
   statistics when given the same mu_model (proves refactor is behavior-
   preserving). Uses fully synthetic LCDM data; no Pantheon+ file needed.

2. Deviation-metric sanity: the returned max/RMS deviation metrics must be
   finite and >= 0 even for a trivial synthetic case.

3. Integration smoke: run the full hubble_diagram_nbody.run() with a tiny
   simulation (n_particles=40, n_steps=300, t_start=5.8) against the real
   Pantheon+ file. Asserts finite chi^2/R^2 and a non-empty in-range subset.
   Skipped (with an informative message) if the real data file is absent.
   Marked as slow via pytest.mark.slow.
"""

import types
import unittest

import numpy as np
import pytest

from cosmo.distances import model_distance_modulus
from cosmo.hubble_diagram import evaluate_model, evaluate_precomputed


# ---------------------------------------------------------------------------
# Shared fixtures (mirrors test_hubble_diagram.py pattern)
# ---------------------------------------------------------------------------

def _make_lcdm_data(n: int = 200, seed: int = 0, z_max: float = 1.2) -> tuple:
    """
    Build synthetic (z, mu_lcdm, sigma) arrays using LCDM model_distance_modulus.

    Returns (z, mu_lcdm, sigma) where sigma is uniform 0.15 mag.
    z range is limited to z_max to mirror the sim-covered range.
    """
    rng = np.random.default_rng(seed)
    z = np.sort(rng.uniform(0.02, z_max, n))
    mu_lcdm = model_distance_modulus(z, "lcdm")
    sigma = np.full(n, 0.15)
    return z, mu_lcdm, sigma


# ---------------------------------------------------------------------------
# 1. evaluate_precomputed vs evaluate_model (behavior-preserving refactor)
# ---------------------------------------------------------------------------

class TestEvaluatePrecomputedMatchesModel(unittest.TestCase):
    """
    Feed mu_model = model_distance_modulus(z, 'lcdm') into evaluate_precomputed
    and compare against evaluate_model(..., model='lcdm').
    chi2, DeltaM, and R2 must match exactly (same computation path).
    """

    def setUp(self):
        self.z, self.mu_obs, self.sigma = _make_lcdm_data(n=150, seed=7)
        # Add a small noise so DeltaM != 0
        rng = np.random.default_rng(77)
        self.mu_obs = self.mu_obs + rng.standard_normal(150) * 0.10
        self.mu_model_lcdm = model_distance_modulus(self.z, "lcdm")

    def test_chi2_matches_evaluate_model(self):
        r_model = evaluate_model(self.z, self.mu_obs, self.sigma, model="lcdm")
        r_pre = evaluate_precomputed(
            self.z, self.mu_obs, self.sigma, self.mu_model_lcdm
        )
        self.assertAlmostEqual(
            r_model["chi2"], r_pre["chi2"], delta=1e-9,
            msg=f"chi2 mismatch: model={r_model['chi2']:.8f}, precomputed={r_pre['chi2']:.8f}",
        )

    def test_deltaM_matches_evaluate_model(self):
        r_model = evaluate_model(self.z, self.mu_obs, self.sigma, model="lcdm")
        r_pre = evaluate_precomputed(
            self.z, self.mu_obs, self.sigma, self.mu_model_lcdm
        )
        self.assertAlmostEqual(
            r_model["DeltaM"], r_pre["DeltaM"], delta=1e-9,
            msg=(f"DeltaM mismatch: model={r_model['DeltaM']:.8f}, "
                 f"precomputed={r_pre['DeltaM']:.8f}"),
        )

    def test_R2_matches_evaluate_model(self):
        r_model = evaluate_model(self.z, self.mu_obs, self.sigma, model="lcdm")
        r_pre = evaluate_precomputed(
            self.z, self.mu_obs, self.sigma, self.mu_model_lcdm
        )
        self.assertAlmostEqual(
            r_model["R2"], r_pre["R2"], delta=1e-9,
            msg=f"R2 mismatch: model={r_model['R2']:.8f}, precomputed={r_pre['R2']:.8f}",
        )

    def test_model_name_stored_correctly(self):
        """evaluate_precomputed stores the supplied model_name."""
        r_pre = evaluate_precomputed(
            self.z, self.mu_obs, self.sigma, self.mu_model_lcdm,
            model_name="my_custom_model",
        )
        self.assertEqual(r_pre["model"], "my_custom_model")

    def test_default_model_name_is_external_node_nbody(self):
        r_pre = evaluate_precomputed(
            self.z, self.mu_obs, self.sigma, self.mu_model_lcdm
        )
        self.assertEqual(r_pre["model"], "external_node_nbody")

    def test_dof_matches_evaluate_model(self):
        n = len(self.z)
        r_model = evaluate_model(self.z, self.mu_obs, self.sigma, model="lcdm")
        r_pre = evaluate_precomputed(
            self.z, self.mu_obs, self.sigma, self.mu_model_lcdm
        )
        self.assertEqual(r_pre["dof"], n - 1)
        self.assertEqual(r_pre["dof"], r_model["dof"])

    def test_residuals_shape_matches_input(self):
        n = len(self.z)
        r_pre = evaluate_precomputed(
            self.z, self.mu_obs, self.sigma, self.mu_model_lcdm
        )
        self.assertEqual(len(r_pre["residuals"]), n)
        self.assertEqual(len(r_pre["mu_fit"]), n)

    def test_noiseless_data_chi2_near_zero(self):
        """With exact LCDM mu_obs (+ constant offset, no noise), chi2 ~ 0."""
        z, mu_lcdm, sigma = _make_lcdm_data(n=100, seed=42)
        mu_obs = mu_lcdm + 3.14  # pure offset, no noise
        r_pre = evaluate_precomputed(z, mu_obs, sigma, mu_lcdm)
        self.assertAlmostEqual(r_pre["chi2"], 0.0, delta=1e-9,
                               msg=f"chi2={r_pre['chi2']:.2e} should be ~0")

    def test_empty_array_raises_value_error(self):
        empty = np.array([])
        with self.assertRaises(ValueError):
            evaluate_precomputed(empty, empty, empty, empty)


# ---------------------------------------------------------------------------
# 2. Deviation-metric sanity
# ---------------------------------------------------------------------------

class TestDeviationMetricSanity(unittest.TestCase):
    """
    Simulate what run() does for the deviation metrics using a synthetic
    a(t) that is exact LCDM — deviation must be finite and ~ 0.
    """

    def _run_deviation(self, a_perturbed: np.ndarray, t_Gyr: np.ndarray,
                       t_start: float) -> tuple:
        """
        Helper: compute deviation between from-sim mu and analytic LCDM mu
        on a small synthetic z grid.
        """
        from cosmo.sim_distance import sim_to_distance_modulus

        # Small synthetic z targets
        z_target = np.linspace(0.05, 1.0, 50)

        sim_dist = sim_to_distance_modulus(z_target, a_perturbed, t_Gyr, t_start)
        in_range = sim_dist["in_range"]
        z_in = z_target[in_range]
        mu_in = model_distance_modulus(z_in, "lcdm")   # synthetic "observations"
        sigma_in = np.full(len(z_in), 0.15)

        r_sim = evaluate_precomputed(z_in, mu_in, sigma_in, sim_dist["mu"])
        r_lcdm = evaluate_model(z_in, mu_in, sigma_in, model="lcdm")

        # Compute deviation (scalar metric)
        z_dense = np.linspace(sim_dist["z_cover"][0] + 1e-4,
                              sim_dist["z_cover"][1] - 1e-4, 200)

        mu_sim_dense = np.interp(
            z_dense, sim_dist["z"], sim_dist["mu"], left=np.nan, right=np.nan
        )
        mu_lcdm_dense = model_distance_modulus(z_dense, "lcdm")

        diff = (mu_sim_dense + r_sim["DeltaM"]) - (mu_lcdm_dense + r_lcdm["DeltaM"])
        valid = np.isfinite(diff)
        dev_max = float(np.max(np.abs(diff[valid]))) if np.any(valid) else float("nan")
        dev_rms = float(np.sqrt(np.mean(diff[valid] ** 2))) if np.any(valid) else float("nan")

        return dev_max, dev_rms

    def _make_lcdm_at(self) -> tuple:
        """
        Build a synthetic a(t) from LCDM Friedmann solution over
        t_start=5.8 to t_end=13.8 Gyr.
        """
        from cosmo.analysis import solve_friedmann_at_times

        t_start = 5.8
        t_end = 13.8
        n = 300
        t_Gyr_rel = np.linspace(0.0, t_end - t_start, n)
        t_abs = t_start + t_Gyr_rel

        sol = solve_friedmann_at_times(t_abs)
        a_lcdm = sol["a"]
        # Normalize so a[0] = 1 (sim convention)
        a_lcdm = a_lcdm / a_lcdm[0]

        return a_lcdm, t_Gyr_rel, t_start

    def test_deviation_metrics_are_finite(self):
        """Deviation metrics must be finite for a valid synthetic a(t)."""
        a_lcdm, t_Gyr, t_start = self._make_lcdm_at()
        dev_max, dev_rms = self._run_deviation(a_lcdm, t_Gyr, t_start)
        self.assertTrue(np.isfinite(dev_max),
                        f"dev_max={dev_max} should be finite")
        self.assertTrue(np.isfinite(dev_rms),
                        f"dev_rms={dev_rms} should be finite")

    def test_deviation_metrics_are_non_negative(self):
        """Deviation metrics must be >= 0."""
        a_lcdm, t_Gyr, t_start = self._make_lcdm_at()
        dev_max, dev_rms = self._run_deviation(a_lcdm, t_Gyr, t_start)
        self.assertGreaterEqual(dev_max, 0.0,
                                f"dev_max={dev_max:.6f} must be >= 0")
        self.assertGreaterEqual(dev_rms, 0.0,
                                f"dev_rms={dev_rms:.6f} must be >= 0")

    def test_exact_lcdm_atob_gives_near_zero_deviation(self):
        """
        When the from-sim a(t) is exact LCDM, the deviation from the analytic
        LCDM curve should be near zero (shape is identical).
        The absolute mu offset from H0 is marginalized, so only shape matters.
        """
        a_lcdm, t_Gyr, t_start = self._make_lcdm_at()
        dev_max, dev_rms = self._run_deviation(a_lcdm, t_Gyr, t_start)
        # LCDM a(t) -> same mu(z) shape -> residuals ~ 0 after offset margin.
        # The Friedmann ODE -> cumulative integral chain introduces ~0.04 mag
        # numerical noise at 300 steps; allow up to 0.10 mag as the threshold.
        self.assertLess(dev_max, 0.10,
                        f"LCDM a(t) should give dev_max < 0.10 mag, got {dev_max:.5f}")


# ---------------------------------------------------------------------------
# 3. Integration smoke test (requires real Pantheon+ data)
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestIntegrationSmoke(unittest.TestCase):
    """
    End-to-end smoke test: run hubble_diagram_nbody.run() with a tiny sim
    against the real Pantheon+ data.

    Skipped if the real data file is absent (FileNotFoundError from load_pantheon).
    n_particles=40, n_steps=300, t_start=5.8 -> t_duration=8.0 Gyr.
    """

    _SKIP_MSG = None  # set in setUpClass if data absent

    @classmethod
    def setUpClass(cls):
        import cosmo.pantheon as pl
        try:
            pl.load_pantheon()
        except FileNotFoundError as exc:
            cls._SKIP_MSG = f"Real Pantheon+ data absent: {exc}"

    def _skip_if_no_data(self):
        if self._SKIP_MSG:
            self.skipTest(self._SKIP_MSG)

    def _make_tiny_sim_params(self):
        from cosmo.constants import SimulationParameters
        t_start = 5.8
        t_duration = 13.8 - t_start  # == 8.0 Gyr
        return SimulationParameters(
            M_value=855.0,
            S_value=37.8,
            n_particles=40,
            seed=42,
            t_start_Gyr=t_start,
            t_duration_Gyr=t_duration,
            n_steps=300,
            damping_factor=None,
            center_node_mass=1.0,
            mass_randomize=0.0,
        )

    def test_run_returns_finite_chi2(self):
        """run() must return finite chi2/R2 for the from-sim entry."""
        self._skip_if_no_data()
        import tempfile
        import hubble_diagram_nbody as hdn

        sim_params = self._make_tiny_sim_params()
        with tempfile.TemporaryDirectory() as tmp:
            result = hdn.run(
                sim_params=sim_params,
                output_dir=tmp,
                z_min=0.01,
                n_bins=15,
            )

        r_nbody = result["results_in_range"]["external_node_nbody"]
        self.assertTrue(np.isfinite(r_nbody["chi2"]),
                        f"chi2={r_nbody['chi2']} should be finite")
        self.assertTrue(np.isfinite(r_nbody["R2"]),
                        f"R2={r_nbody['R2']} should be finite")

    def test_run_has_nonempty_in_range_subset(self):
        """n_in_range must be > 0."""
        self._skip_if_no_data()
        import tempfile
        import hubble_diagram_nbody as hdn

        sim_params = self._make_tiny_sim_params()
        with tempfile.TemporaryDirectory() as tmp:
            result = hdn.run(
                sim_params=sim_params,
                output_dir=tmp,
                z_min=0.01,
                n_bins=15,
            )

        self.assertGreater(result["n_in_range"], 0,
                           "n_in_range must be positive")

    def test_run_deviation_metrics_are_finite_and_nonneg(self):
        """Deviation metrics from a real sim run must be finite and >= 0."""
        self._skip_if_no_data()
        import tempfile
        import hubble_diagram_nbody as hdn

        sim_params = self._make_tiny_sim_params()
        with tempfile.TemporaryDirectory() as tmp:
            result = hdn.run(
                sim_params=sim_params,
                output_dir=tmp,
                z_min=0.01,
                n_bins=15,
            )

        dev_max = result["deviation_max"]
        dev_rms = result["deviation_rms"]
        self.assertTrue(np.isfinite(dev_max),
                        f"deviation_max={dev_max} should be finite")
        self.assertTrue(np.isfinite(dev_rms),
                        f"deviation_rms={dev_rms} should be finite")
        self.assertGreaterEqual(dev_max, 0.0)
        self.assertGreaterEqual(dev_rms, 0.0)


if __name__ == "__main__":
    unittest.main()
