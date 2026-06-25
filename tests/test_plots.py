"""
Tests for cosmo/plots.py — WS2 shared plotting module.

Philosophy
----------
- No long sims in tests: feed tiny synthetic arrays and DataFrames.
- Every test asserts that the target PNG was written with non-zero size.
- The Agg backend is imported before pyplot via cosmo.plots (guaranteed by the
  module's own ``matplotlib.use("Agg")`` at import time).
- Tests run in < 5 s total on any developer machine.

Test groups
-----------
1. figure_path — dir creation; correct extension and sub-directory nesting.
2. plot_ms_heatmap — writes a non-empty PNG from a tiny synthetic DataFrame;
   handles a pandas DataFrame; marks the best cell; bad column raises ValueError.
3. plot_growth_map — writes PNG; runaway cells (anchor_ok=False) accepted.
4. plot_eds_overlay — writes PNG from a tiny synthetic a(t) / EdS arrays.
5. plot_mu_z_panel — writes PNG from synthetic sim_dist + results dicts.
6. plot_shear_vs_lever — writes PNG.
7. plot_dipole_vs_lever — writes PNG.
8. plot_runaway_boundary — writes PNG; bad column raises ValueError.
9. plots_from_csv — reads a real CSV from results/ (if present) OR a synthetic
   one and emits heatmaps; returns non-empty list of existing files.
"""

import io
import os
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Synthetic-data helpers
# ---------------------------------------------------------------------------

def _make_sweep_df(n_M: int = 3, n_S: int = 3):
    """Return a tiny pandas DataFrame mimicking a sweep results CSV."""
    import pandas as pd

    M_vals = [50, 500, 5000][:n_M]
    S_vals = [20, 30, 40][:n_S]
    rows = []
    rng = np.random.default_rng(0)
    for M in M_vals:
        for S in S_vals:
            anchor = S > 22  # cheap rule: small S => runaway
            rows.append({
                "M_factor": float(M),
                "S_gpc": float(S),
                "chi2_dof": 0.4 + 0.1 * rng.uniform(),
                "growth_factor": 3.0 + 0.3 * rng.uniform(),
                "anchor_ok": anchor,
                "node_mass_amplitude": 0.0,
                "init_distribution": "uniform_sphere",
            })
    return pd.DataFrame(rows)


def _make_sim_arrays(n_steps: int = 50, t_start: float = 5.8):
    """Return (a_sim, t_Gyr) for a tiny synthetic simulation."""
    t_duration = 13.8 - t_start
    t_Gyr = np.linspace(0.0, t_duration, n_steps + 1)
    # EdS-like a(t) ∝ t^(2/3) but starting at 1.0
    t_abs = t_start + t_Gyr
    a_sim = (t_abs / t_start) ** (2.0 / 3.0)
    a_sim = a_sim / a_sim[0]   # normalize to 1 at t_start
    return a_sim, t_Gyr


def _make_sim_dist(a_sim, t_Gyr, t_start):
    """Run sim_to_distance_modulus on a dense z grid."""
    from cosmo.sim_distance import sim_to_distance_modulus
    z_target = np.linspace(0.02, 0.8, 30)
    return sim_to_distance_modulus(z_target, a_sim, t_Gyr, t_start_Gyr=t_start)


def _make_results(z_in, mu_sim_in):
    """Build a minimal results dict (no real Pantheon data needed)."""
    from cosmo.distances import model_distance_modulus
    from cosmo.hubble_diagram import evaluate_precomputed, evaluate_model

    mu_lcdm_in = model_distance_modulus(z_in, "lcdm")
    sigma = np.full_like(z_in, 0.15)

    # Fake observed = sim
    mu_obs = mu_sim_in + 0.02  # slight offset
    return {
        "external_node_nbody": evaluate_precomputed(
            z_in, mu_obs, sigma, mu_sim_in, model_name="external_node_nbody"
        ),
        "lcdm": evaluate_model(z_in, mu_obs, sigma, model="lcdm"),
        "einstein_de_sitter": evaluate_model(
            z_in, mu_obs, sigma, model="einstein_de_sitter"
        ),
    }


# ---------------------------------------------------------------------------
# Helper: assert a PNG was written and is non-empty
# ---------------------------------------------------------------------------

def _assert_png(path: str) -> None:
    assert os.path.isfile(path), f"Expected PNG at {path!r} but file not found."
    assert os.path.getsize(path) > 0, f"PNG at {path!r} is empty."


# ---------------------------------------------------------------------------
# 1. figure_path
# ---------------------------------------------------------------------------

class TestFigurePath(unittest.TestCase):
    def test_creates_directory_and_returns_png(self):
        from cosmo.plots import figure_path, _RESULTS_ROOT

        # Temporarily redirect _RESULTS_ROOT to a temp dir so we don't
        # pollute the real results/figures/ during testing.
        with tempfile.TemporaryDirectory() as tmp:
            import cosmo.plots as _plots
            orig = _plots._RESULTS_ROOT
            _plots._RESULTS_ROOT = tmp
            try:
                path = figure_path("test_ws", "test_figure")
            finally:
                _plots._RESULTS_ROOT = orig

        # The path should end with .png and be nested correctly
        assert path.endswith(".png"), path
        assert os.sep + "test_ws" + os.sep in path, path
        assert "test_figure.png" in path, path

    def test_returns_string(self):
        from cosmo.plots import figure_path
        import cosmo.plots as _plots
        with tempfile.TemporaryDirectory() as tmp:
            orig = _plots._RESULTS_ROOT
            _plots._RESULTS_ROOT = tmp
            try:
                result = figure_path("ws_unit", "unit_fig")
            finally:
                _plots._RESULTS_ROOT = orig
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# 2. plot_ms_heatmap
# ---------------------------------------------------------------------------

class TestMsHeatmap(unittest.TestCase):
    def setUp(self):
        import cosmo.plots as _plots
        self._tmp = tempfile.TemporaryDirectory()
        self._orig = _plots._RESULTS_ROOT
        _plots._RESULTS_ROOT = self._tmp.name
        self._plots = _plots

    def tearDown(self):
        self._plots._RESULTS_ROOT = self._orig
        self._tmp.cleanup()

    def test_writes_nonempty_png(self):
        from cosmo.plots import plot_ms_heatmap
        df = _make_sweep_df()
        path = plot_ms_heatmap(df, "chi2_dof", "ws_test", "heatmap_chi2")
        _assert_png(path)

    def test_returns_path_string(self):
        from cosmo.plots import plot_ms_heatmap
        df = _make_sweep_df()
        path = plot_ms_heatmap(df, "chi2_dof", "ws_test", "heatmap_chi2b")
        assert isinstance(path, str)

    def test_missing_column_raises(self):
        from cosmo.plots import plot_ms_heatmap
        import pandas as pd
        df = pd.DataFrame({"M_factor": [50], "S_gpc": [30]})
        with self.assertRaises(ValueError):
            plot_ms_heatmap(df, "chi2_dof", "ws_test", "heatmap_bad")

    def test_mark_best_false(self):
        """mark_best=False should still write a valid PNG."""
        from cosmo.plots import plot_ms_heatmap
        df = _make_sweep_df()
        path = plot_ms_heatmap(
            df, "chi2_dof", "ws_test", "heatmap_nobest", mark_best=False
        )
        _assert_png(path)

    def test_dict_input_accepted(self):
        """Should accept a plain dict as well as a DataFrame."""
        from cosmo.plots import plot_ms_heatmap
        data = {
            "M_factor": [50, 500],
            "S_gpc": [30, 40],
            "chi2_dof": [0.5, 0.6],
        }
        path = plot_ms_heatmap(data, "chi2_dof", "ws_test", "heatmap_dict")
        _assert_png(path)


# ---------------------------------------------------------------------------
# 3. plot_growth_map
# ---------------------------------------------------------------------------

class TestGrowthMap(unittest.TestCase):
    def setUp(self):
        import cosmo.plots as _plots
        self._tmp = tempfile.TemporaryDirectory()
        self._orig = _plots._RESULTS_ROOT
        _plots._RESULTS_ROOT = self._tmp.name

    def tearDown(self):
        import cosmo.plots as _plots
        _plots._RESULTS_ROOT = self._orig
        self._tmp.cleanup()

    def test_writes_nonempty_png(self):
        from cosmo.plots import plot_growth_map
        df = _make_sweep_df()
        path = plot_growth_map(df, "ws_test", "growth_map")
        _assert_png(path)

    def test_missing_column_raises(self):
        from cosmo.plots import plot_growth_map
        import pandas as pd
        df = pd.DataFrame({"M_factor": [50], "S_gpc": [30]})
        with self.assertRaises(ValueError):
            plot_growth_map(df, "ws_test", "growth_bad")

    def test_target_growth_line(self):
        """target_growth kwarg should not crash."""
        from cosmo.plots import plot_growth_map
        df = _make_sweep_df()
        path = plot_growth_map(df, "ws_test", "growth_map_tgt", target_growth=3.2)
        _assert_png(path)


# ---------------------------------------------------------------------------
# 4. plot_eds_overlay
# ---------------------------------------------------------------------------

class TestEdsOverlay(unittest.TestCase):
    def setUp(self):
        import cosmo.plots as _plots
        self._tmp = tempfile.TemporaryDirectory()
        self._orig = _plots._RESULTS_ROOT
        _plots._RESULTS_ROOT = self._tmp.name

    def tearDown(self):
        import cosmo.plots as _plots
        _plots._RESULTS_ROOT = self._orig
        self._tmp.cleanup()

    def test_writes_nonempty_png(self):
        from cosmo.plots import plot_eds_overlay
        a_sim, t_Gyr = _make_sim_arrays(n_steps=60, t_start=5.8)
        path = plot_eds_overlay(
            a_sim, t_Gyr, t_start_Gyr=5.8,
            workstream="ws_test", name="eds_overlay_synth",
        )
        _assert_png(path)

    def test_early_start(self):
        """Should work with t_start=2.9 (the full-coverage default)."""
        from cosmo.plots import plot_eds_overlay
        a_sim, t_Gyr = _make_sim_arrays(n_steps=80, t_start=2.9)
        path = plot_eds_overlay(
            a_sim, t_Gyr, t_start_Gyr=2.9,
            workstream="ws_test", name="eds_overlay_early",
        )
        _assert_png(path)


# ---------------------------------------------------------------------------
# 5. plot_mu_z_panel
# ---------------------------------------------------------------------------

class TestMuZPanel(unittest.TestCase):
    def setUp(self):
        import cosmo.plots as _plots
        self._tmp = tempfile.TemporaryDirectory()
        self._orig = _plots._RESULTS_ROOT
        _plots._RESULTS_ROOT = self._tmp.name

    def tearDown(self):
        import cosmo.plots as _plots
        _plots._RESULTS_ROOT = self._orig
        self._tmp.cleanup()

    def _build_inputs(self, t_start=5.8):
        a_sim, t_Gyr = _make_sim_arrays(n_steps=60, t_start=t_start)
        sd = _make_sim_dist(a_sim, t_Gyr, t_start)
        results = _make_results(sd["z"], sd["mu"])
        # Minimal fake sim_params
        sp = types.SimpleNamespace(M_value=855, S_value=37.8)
        return sd, results, sp

    def test_writes_nonempty_png(self):
        from cosmo.plots import plot_mu_z_panel
        sd, results, sp = self._build_inputs()
        path = plot_mu_z_panel(sd, results, sp, "ws_test", "mu_z_synth")
        _assert_png(path)

    def test_no_data_overlay(self):
        """data=None should still produce a valid figure."""
        from cosmo.plots import plot_mu_z_panel
        sd, results, sp = self._build_inputs()
        path = plot_mu_z_panel(sd, results, sp, "ws_test", "mu_z_nodata", data=None)
        _assert_png(path)


# ---------------------------------------------------------------------------
# 6. plot_shear_vs_lever
# ---------------------------------------------------------------------------

class TestShearVsLever(unittest.TestCase):
    def setUp(self):
        import cosmo.plots as _plots
        self._tmp = tempfile.TemporaryDirectory()
        self._orig = _plots._RESULTS_ROOT
        _plots._RESULTS_ROOT = self._tmp.name

    def tearDown(self):
        import cosmo.plots as _plots
        _plots._RESULTS_ROOT = self._orig
        self._tmp.cleanup()

    def test_writes_nonempty_png(self):
        from cosmo.plots import plot_shear_vs_lever
        levers = [0.0, 0.25, 0.5, 0.75, 1.0]
        shears = [0.01, 0.12, 0.30, 0.55, 0.90]
        path = plot_shear_vs_lever(
            levers, shears, "node_mass_amplitude",
            "ws_test", "shear_vs_nma",
        )
        _assert_png(path)


# ---------------------------------------------------------------------------
# 7. plot_dipole_vs_lever
# ---------------------------------------------------------------------------

class TestDipoleVsLever(unittest.TestCase):
    def setUp(self):
        import cosmo.plots as _plots
        self._tmp = tempfile.TemporaryDirectory()
        self._orig = _plots._RESULTS_ROOT
        _plots._RESULTS_ROOT = self._tmp.name

    def tearDown(self):
        import cosmo.plots as _plots
        _plots._RESULTS_ROOT = self._orig
        self._tmp.cleanup()

    def test_writes_nonempty_png(self):
        from cosmo.plots import plot_dipole_vs_lever
        levers = [0.0, 0.25, 0.5, 0.75, 1.0]
        dipoles = [0.005, 0.04, 0.09, 0.16, 0.25]
        path = plot_dipole_vs_lever(
            levers, dipoles, "node_mass_amplitude",
            "ws_test", "dipole_vs_nma",
        )
        _assert_png(path)


# ---------------------------------------------------------------------------
# 8. plot_runaway_boundary
# ---------------------------------------------------------------------------

class TestRunawayBoundary(unittest.TestCase):
    def setUp(self):
        import cosmo.plots as _plots
        self._tmp = tempfile.TemporaryDirectory()
        self._orig = _plots._RESULTS_ROOT
        _plots._RESULTS_ROOT = self._tmp.name

    def tearDown(self):
        import cosmo.plots as _plots
        _plots._RESULTS_ROOT = self._orig
        self._tmp.cleanup()

    def test_writes_nonempty_png(self):
        from cosmo.plots import plot_runaway_boundary
        df = _make_sweep_df()
        path = plot_runaway_boundary(df, "ws_test", "runaway_boundary")
        _assert_png(path)

    def test_missing_column_raises(self):
        from cosmo.plots import plot_runaway_boundary
        import pandas as pd
        df = pd.DataFrame({"M_factor": [50], "S_gpc": [30]})
        with self.assertRaises(ValueError):
            plot_runaway_boundary(df, "ws_test", "runaway_bad")

    def test_string_boolean_values(self):
        """anchor_ok as 'True'/'False' strings should be parsed correctly."""
        from cosmo.plots import plot_runaway_boundary
        import pandas as pd
        df = pd.DataFrame({
            "M_factor": [50, 50, 500, 500],
            "S_gpc": [20, 30, 20, 30],
            "anchor_ok": ["False", "True", "False", "True"],
        })
        path = plot_runaway_boundary(df, "ws_test", "runaway_str_bool")
        _assert_png(path)


# ---------------------------------------------------------------------------
# 9. plots_from_csv — read from the real sweep CSV if present
# ---------------------------------------------------------------------------

class TestPlotsFromCsv(unittest.TestCase):
    """Regenerate heatmaps from either the real CSV or a synthetic one."""

    def setUp(self):
        import cosmo.plots as _plots
        self._tmp = tempfile.TemporaryDirectory()
        self._orig = _plots._RESULTS_ROOT
        _plots._RESULTS_ROOT = self._tmp.name

    def tearDown(self):
        import cosmo.plots as _plots
        _plots._RESULTS_ROOT = self._orig
        self._tmp.cleanup()

    def _make_synthetic_csv(self) -> str:
        """Write a tiny sweep CSV to a temp file and return its path."""
        import csv as _csv
        path = os.path.join(self._tmp.name, "synthetic_sweep.csv")
        rows = [
            {
                "M_factor": M,
                "S_gpc": S,
                "chi2_dof": 0.5 + 0.01 * (i + j),
                "growth_factor": 3.1 + 0.05 * (i + j),
                "anchor_ok": (S > 22),
            }
            for i, M in enumerate([50, 500, 5000])
            for j, S in enumerate([20, 30, 40])
        ]
        with open(path, "w", newline="") as fh:
            writer = _csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_synthetic_csv(self):
        from cosmo.plots import plots_from_csv
        csv_path = self._make_synthetic_csv()
        out_paths = plots_from_csv(csv_path, workstream="ws_test")
        assert len(out_paths) > 0, "plots_from_csv returned empty list"
        for p in out_paths:
            _assert_png(p)

    def test_real_csv_if_present(self):
        """Use the real sweep CSV if it exists (optional, skipped otherwise)."""
        from cosmo.plots import plots_from_csv

        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        real_csv = os.path.join(repo_root, "results", "sweep_results_pantheon.csv")
        if not os.path.isfile(real_csv):
            pytest.skip("Real sweep CSV not present; skipping real-data test.")

        out_paths = plots_from_csv(real_csv, workstream="ws_test_real")
        assert len(out_paths) > 0
        for p in out_paths:
            _assert_png(p)


if __name__ == "__main__":
    unittest.main()
