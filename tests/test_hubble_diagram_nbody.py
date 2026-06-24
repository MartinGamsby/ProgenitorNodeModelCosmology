"""
Tests for hubble_diagram_nbody.py and the additive evaluate_precomputed.

Test groups
-----------
1. Unit: evaluate_precomputed vs evaluate_model — must produce identical
   statistics when given the same mu_model (proves refactor is behavior-
   preserving). Uses fully synthetic LCDM data; no Pantheon+ file needed.

2. Deviation-metric sanity: the returned max/RMS deviation metrics must be
   finite and >= 0 even for a trivial synthetic case.

3. load_best_config: --from-best-config loader, column mapping, precedence
   of explicit --M/--S flags, and error handling for missing file/columns.

4. JSON sidecar: _build_sidecar() produces a dict with the expected top-level
   keys and correct per-model entries; values round-trip through json.dumps.

5. Residual panel: _make_figure() includes a LCDM zero-reference line in the
   bottom panel (label contains "LCDM" and "zero reference").

6. Integration smoke: run the full hubble_diagram_nbody.run() with a tiny
   simulation (n_particles=40, n_steps=300, t_start=5.8) against the real
   Pantheon+ file. Asserts finite chi^2/R^2 and a non-empty in-range subset.
   Also checks that the return dict has the new keys (growth_factor,
   growth_target, anchor_ok, out_path, sidecar_path) and the sidecar was
   actually written.
   Skipped (with an informative message) if the real data file is absent.
   Marked as slow via pytest.mark.slow.
"""

import csv
import io
import json
import os
import tempfile
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
# 3. load_best_config — --from-best-config loader
# ---------------------------------------------------------------------------

def _write_sweep_csv(tmp_dir: str, rows: list[dict], fname: str = "sweep.csv") -> str:
    """Write a minimal sweep CSV to a temp directory and return its path."""
    path = os.path.join(tmp_dir, fname)
    if not rows:
        # Write header-only (empty)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("M_factor,S_gpc,centerM,chi2_dof,chi2,R2\n")
        return path
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


class TestLoadBestConfig(unittest.TestCase):
    """Unit tests for hubble_diagram_nbody.load_best_config."""

    def _import(self):
        import hubble_diagram_nbody as hdn
        return hdn

    def test_picks_row_with_lowest_chi2_dof(self):
        hdn = self._import()
        rows = [
            {"M_factor": "500", "S_gpc": "30", "centerM": "1",
             "chi2_dof": "1.5", "chi2": "900", "R2": "0.99"},
            {"M_factor": "855", "S_gpc": "37", "centerM": "1",
             "chi2_dof": "0.9", "chi2": "600", "R2": "0.998"},  # best
            {"M_factor": "200", "S_gpc": "20", "centerM": "2",
             "chi2_dof": "2.1", "chi2": "1200", "R2": "0.95"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_sweep_csv(tmp, rows)
            cfg = hdn.load_best_config(path)

        self.assertAlmostEqual(cfg["M"], 855.0, places=5)
        self.assertAlmostEqual(cfg["S"], 37.0, places=5)
        self.assertAlmostEqual(cfg["centerM"], 1.0, places=5)
        self.assertAlmostEqual(cfg["chi2_dof"], 0.9, places=5)

    def test_returns_expected_keys(self):
        hdn = self._import()
        rows = [
            {"M_factor": "855", "S_gpc": "37.8", "centerM": "1",
             "chi2_dof": "0.88", "chi2": "550", "R2": "0.999"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_sweep_csv(tmp, rows)
            cfg = hdn.load_best_config(path)

        for key in ("M", "S", "centerM", "chi2_dof", "chi2", "R2"):
            self.assertIn(key, cfg, f"Expected key {key!r} in result")

    def test_missing_file_raises_file_not_found(self):
        hdn = self._import()
        with self.assertRaises(FileNotFoundError):
            hdn.load_best_config("/nonexistent/path/sweep.csv")

    def test_empty_csv_raises_value_error(self):
        hdn = self._import()
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_sweep_csv(tmp, [])  # header only, no rows
            with self.assertRaises(ValueError):
                hdn.load_best_config(path)

    def test_missing_required_columns_raises_value_error(self):
        hdn = self._import()
        # CSV with wrong column names
        rows = [
            {"wrong_col": "123", "another": "456"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_sweep_csv(tmp, rows)
            with self.assertRaises(ValueError):
                hdn.load_best_config(path)

    def test_m_value_is_float(self):
        """M_factor must be returned as float."""
        hdn = self._import()
        rows = [
            {"M_factor": "1234", "S_gpc": "55", "centerM": "3",
             "chi2_dof": "1.0"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_sweep_csv(tmp, rows)
            cfg = hdn.load_best_config(path)
        self.assertIsInstance(cfg["M"], float)
        self.assertIsInstance(cfg["S"], float)

    def test_single_row_file_returns_that_row(self):
        """With a single row the loader must return exactly that config."""
        hdn = self._import()
        rows = [
            {"M_factor": "777", "S_gpc": "42.0", "centerM": "5",
             "chi2_dof": "0.72"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_sweep_csv(tmp, rows)
            cfg = hdn.load_best_config(path)
        self.assertAlmostEqual(cfg["M"], 777.0, places=5)
        self.assertAlmostEqual(cfg["S"], 42.0, places=5)


# ---------------------------------------------------------------------------
# 4. JSON sidecar — _build_sidecar contents
# ---------------------------------------------------------------------------

class TestBuildSidecar(unittest.TestCase):
    """
    Test _build_sidecar() returns the expected structure and is JSON-round-trippable.
    Uses fully synthetic results dicts — no simulation needed.
    """

    def _make_fake_results(self):
        """Return a minimal results dict that mirrors what run() builds."""
        def fake_entry(chi2=1.0, dof=100, chi2_dof=0.01, R2=0.999, DeltaM=0.5):
            return {
                "chi2": chi2, "dof": dof, "chi2_dof": chi2_dof,
                "R2": R2, "DeltaM": DeltaM,
                "residuals": np.zeros(dof + 1),
                "mu_fit": np.zeros(dof + 1),
                "model": "test",
            }

        return {
            "external_node_nbody": fake_entry(chi2=650, dof=100, chi2_dof=0.52,
                                               R2=0.997, DeltaM=0.12),
            "lcdm":                fake_entry(chi2=600, dof=100, chi2_dof=0.48,
                                               R2=0.998, DeltaM=0.10),
            "einstein_de_sitter":  fake_entry(chi2=2000, dof=100, chi2_dof=2.0,
                                               R2=0.85, DeltaM=-1.5),
            "analytic_shortcut":   None,
        }

    def _make_fake_sim_params(self):
        from cosmo.constants import SimulationParameters
        return SimulationParameters(
            M_value=855.0, S_value=37.8,
            n_particles=80, seed=42,
            t_start_Gyr=5.8, t_duration_Gyr=8.0,
            n_steps=300, damping_factor=None,
            center_node_mass=1.0, mass_randomize=0.0,
        )

    def test_sidecar_has_top_level_keys(self):
        import hubble_diagram_nbody as hdn
        results = self._make_fake_results()
        sim_params = self._make_fake_sim_params()
        sidecar = hdn._build_sidecar(
            sim_params=sim_params,
            results=results,
            n_in_range=500,
            n_dropped=50,
            z_cover=(0.01, 1.2),
            dev_max=0.03,
            dev_rms=0.01,
            typical_sigma=0.15,
            model_growth=2.38,
            target_growth=2.38,
            anchor_ok=True,
        )
        for key in ("config", "coverage", "models", "deviation", "growth_anchor"):
            self.assertIn(key, sidecar, f"Missing top-level key: {key!r}")

    def test_config_block_has_correct_values(self):
        import hubble_diagram_nbody as hdn
        results = self._make_fake_results()
        sim_params = self._make_fake_sim_params()
        sidecar = hdn._build_sidecar(
            sim_params=sim_params, results=results,
            n_in_range=500, n_dropped=50, z_cover=(0.01, 1.2),
            dev_max=0.03, dev_rms=0.01, typical_sigma=0.15,
            model_growth=2.38, target_growth=2.38, anchor_ok=True,
        )
        cfg = sidecar["config"]
        self.assertAlmostEqual(cfg["M"], 855.0, places=5)
        self.assertAlmostEqual(cfg["S"], 37.8, places=5)
        self.assertAlmostEqual(cfg["centerM"], 1.0, places=5)
        self.assertAlmostEqual(cfg["t_start"], 5.8, places=5)
        self.assertEqual(cfg["particles"], 80)
        self.assertEqual(cfg["n_steps"], 300)

    def test_per_model_entries_present(self):
        import hubble_diagram_nbody as hdn
        results = self._make_fake_results()
        sim_params = self._make_fake_sim_params()
        sidecar = hdn._build_sidecar(
            sim_params=sim_params, results=results,
            n_in_range=500, n_dropped=50, z_cover=(0.01, 1.2),
            dev_max=0.03, dev_rms=0.01, typical_sigma=0.15,
            model_growth=2.38, target_growth=2.38, anchor_ok=True,
        )
        models = sidecar["models"]
        self.assertIn("external_node_nbody", models)
        self.assertIn("lcdm", models)
        self.assertIn("einstein_de_sitter", models)
        # analytic_shortcut is None (turnaround) — must still be present as None
        self.assertIn("analytic_shortcut", models)
        self.assertIsNone(models["analytic_shortcut"])

    def test_per_model_entry_has_expected_sub_keys(self):
        import hubble_diagram_nbody as hdn
        results = self._make_fake_results()
        sim_params = self._make_fake_sim_params()
        sidecar = hdn._build_sidecar(
            sim_params=sim_params, results=results,
            n_in_range=500, n_dropped=50, z_cover=(0.01, 1.2),
            dev_max=0.03, dev_rms=0.01, typical_sigma=0.15,
            model_growth=2.38, target_growth=2.38, anchor_ok=True,
        )
        entry = sidecar["models"]["external_node_nbody"]
        for key in ("chi2", "dof", "chi2_dof", "R2", "DeltaM"):
            self.assertIn(key, entry, f"Missing model sub-key: {key!r}")

    def test_sidecar_values_match_run_return(self):
        """chi2_dof in sidecar must match the results dict."""
        import hubble_diagram_nbody as hdn
        results = self._make_fake_results()
        sim_params = self._make_fake_sim_params()
        sidecar = hdn._build_sidecar(
            sim_params=sim_params, results=results,
            n_in_range=500, n_dropped=50, z_cover=(0.01, 1.2),
            dev_max=0.03, dev_rms=0.01, typical_sigma=0.15,
            model_growth=2.38, target_growth=2.38, anchor_ok=True,
        )
        self.assertAlmostEqual(
            sidecar["models"]["lcdm"]["chi2_dof"],
            results["lcdm"]["chi2_dof"],
            places=8,
        )
        self.assertAlmostEqual(
            sidecar["deviation"]["deviation_max"], 0.03, places=8,
        )
        self.assertTrue(sidecar["growth_anchor"]["anchor_ok"])

    def test_sidecar_is_json_serializable(self):
        """_build_sidecar result must round-trip through json.dumps/loads."""
        import hubble_diagram_nbody as hdn
        results = self._make_fake_results()
        sim_params = self._make_fake_sim_params()
        sidecar = hdn._build_sidecar(
            sim_params=sim_params, results=results,
            n_in_range=500, n_dropped=50, z_cover=(0.01, 1.2),
            dev_max=0.03, dev_rms=0.01, typical_sigma=0.15,
            model_growth=2.38, target_growth=2.38, anchor_ok=True,
        )
        serialized = json.dumps(sidecar)
        reloaded = json.loads(serialized)
        self.assertEqual(reloaded["config"]["M"], 855.0)
        self.assertEqual(reloaded["models"]["external_node_nbody"]["dof"], 100)


# ---------------------------------------------------------------------------
# 5. Residual panel: LCDM zero reference present in _make_figure
# ---------------------------------------------------------------------------

class TestResidualPanelLcdmReference(unittest.TestCase):
    """
    _make_figure must draw a LCDM zero-reference line in the residual panel.
    We call _make_figure with synthetic data and inspect the bottom axis lines.
    """

    def _build_synthetic_inputs(self):
        """Build the minimal synthetic inputs that _make_figure expects."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from cosmo.constants import SimulationParameters
        from cosmo.hubble_diagram import evaluate_model, evaluate_precomputed

        rng = np.random.default_rng(0)
        n = 80
        z = np.sort(rng.uniform(0.05, 1.0, n))
        sigma = np.full(n, 0.15)
        mu_lcdm = model_distance_modulus(z, "lcdm")
        mu_obs = mu_lcdm + rng.standard_normal(n) * 0.12

        in_range_mask = np.ones(n, dtype=bool)

        results = {
            "external_node_nbody": evaluate_precomputed(z, mu_obs, sigma, mu_lcdm),
            "lcdm":                evaluate_model(z, mu_obs, sigma, model="lcdm"),
            "einstein_de_sitter":  evaluate_model(z, mu_obs, sigma,
                                                   model="einstein_de_sitter"),
            "analytic_shortcut":   None,
        }

        z_dense = np.linspace(0.06, 0.99, 200)
        mu_lcdm_dense = model_distance_modulus(z_dense, "lcdm")
        deltaM_sim = results["external_node_nbody"]["DeltaM"]
        deltaM_lcdm = results["lcdm"]["DeltaM"]
        mu_sim_shifted = mu_lcdm_dense + deltaM_sim
        mu_lcdm_shifted = mu_lcdm_dense + deltaM_lcdm

        sim_params = SimulationParameters(
            M_value=855.0, S_value=37.8,
            n_particles=80, seed=42,
            t_start_Gyr=5.8, t_duration_Gyr=8.0,
            n_steps=300, damping_factor=None,
            center_node_mass=1.0, mass_randomize=0.0,
        )

        data = {
            "z": z, "mu": mu_obs, "sigma": sigma, "n": n,
        }
        z_cover = (float(z.min()), float(z.max()))

        return (data, in_range_mask, results, z_dense,
                mu_sim_shifted, mu_lcdm_shifted, sim_params,
                z_cover, False)

    def test_residual_panel_has_lcdm_zero_reference_line(self):
        """
        The bottom residual panel must contain at least one labelled line
        whose label references 'LCDM' and 'zero reference'.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import hubble_diagram_nbody as hdn

        inputs = self._build_synthetic_inputs()
        fig = hdn._make_figure(*inputs, n_bins=8)

        # The bottom axis is the second subplot
        ax_res = fig.axes[1]
        labels = [line.get_label() for line in ax_res.get_lines()]
        plt.close(fig)

        # At least one label must mention LCDM and zero reference
        lcdm_ref_labels = [
            lbl for lbl in labels
            if "LCDM" in lbl and "zero reference" in lbl
        ]
        self.assertTrue(
            len(lcdm_ref_labels) > 0,
            f"No LCDM zero-reference line found in residual panel. "
            f"Labels found: {labels}"
        )

    def test_residual_panel_ylabel_mentions_lcdm(self):
        """The y-axis label of the residual panel must mention LCDM."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import hubble_diagram_nbody as hdn

        inputs = self._build_synthetic_inputs()
        fig = hdn._make_figure(*inputs, n_bins=8)

        ax_res = fig.axes[1]
        ylabel = ax_res.get_ylabel()
        plt.close(fig)

        self.assertIn("LCDM", ylabel,
                      f"Residual panel y-label {ylabel!r} should mention LCDM")


# ---------------------------------------------------------------------------
# 6. Integration smoke test (requires real Pantheon+ data)
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestIntegrationSmoke(unittest.TestCase):
    """
    End-to-end smoke test: run hubble_diagram_nbody.run() with a tiny sim
    against the real Pantheon+ data.

    Skipped if the real data file is absent (FileNotFoundError from load_pantheon).
    n_particles=40, n_steps=300, t_start=5.8 -> t_duration=8.0 Gyr.

    Checks:
    - finite chi2/R2 for the from-sim entry
    - non-empty in-range subset
    - finite deviation metrics
    - new return keys present (growth_factor, growth_target, anchor_ok,
      out_path, sidecar_path)
    - JSON sidecar file was actually written
    - sidecar has correct structure (config.M matches sim_params.M_value)
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

    def test_run_returns_new_growth_anchor_keys(self):
        """run() return dict must contain growth_factor, growth_target, anchor_ok."""
        self._skip_if_no_data()
        import hubble_diagram_nbody as hdn

        sim_params = self._make_tiny_sim_params()
        with tempfile.TemporaryDirectory() as tmp:
            result = hdn.run(
                sim_params=sim_params,
                output_dir=tmp,
                z_min=0.01,
                n_bins=15,
            )

        for key in ("growth_factor", "growth_target", "anchor_ok"):
            self.assertIn(key, result, f"Missing new return key: {key!r}")
        self.assertIsInstance(result["growth_factor"], float)
        self.assertIsInstance(result["anchor_ok"], bool)

    def test_run_returns_path_keys(self):
        """run() return dict must contain out_path and sidecar_path."""
        self._skip_if_no_data()
        import hubble_diagram_nbody as hdn

        sim_params = self._make_tiny_sim_params()
        with tempfile.TemporaryDirectory() as tmp:
            result = hdn.run(
                sim_params=sim_params,
                output_dir=tmp,
                z_min=0.01,
                n_bins=15,
            )
            # Check keys present and PNG/sidecar files exist
            self.assertIn("out_path", result)
            self.assertIn("sidecar_path", result)
            self.assertTrue(os.path.isfile(result["out_path"]),
                            f"PNG not found: {result['out_path']}")
            self.assertTrue(os.path.isfile(result["sidecar_path"]),
                            f"Sidecar not found: {result['sidecar_path']}")

    def test_sidecar_has_correct_structure(self):
        """The JSON sidecar written by run() must be valid and match config."""
        self._skip_if_no_data()
        import hubble_diagram_nbody as hdn

        sim_params = self._make_tiny_sim_params()
        with tempfile.TemporaryDirectory() as tmp:
            result = hdn.run(
                sim_params=sim_params,
                output_dir=tmp,
                z_min=0.01,
                n_bins=15,
            )
            with open(result["sidecar_path"], encoding="utf-8") as fh:
                sidecar = json.load(fh)

        # Top-level structure
        for key in ("config", "coverage", "models", "deviation", "growth_anchor"):
            self.assertIn(key, sidecar, f"Sidecar missing key: {key!r}")

        # Config matches sim_params
        self.assertAlmostEqual(sidecar["config"]["M"], sim_params.M_value, places=5)
        self.assertAlmostEqual(sidecar["config"]["S"], sim_params.S_value, places=5)
        self.assertEqual(sidecar["config"]["n_steps"], sim_params.n_steps)

        # lcdm model entry present and finite
        lcdm_entry = sidecar["models"]["lcdm"]
        self.assertIsNotNone(lcdm_entry)
        self.assertTrue(np.isfinite(lcdm_entry["chi2_dof"]),
                        f"lcdm chi2_dof={lcdm_entry['chi2_dof']} not finite")


if __name__ == "__main__":
    unittest.main()
