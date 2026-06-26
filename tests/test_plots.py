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
# 8b. plot_geometry_comparison (F12)
# ---------------------------------------------------------------------------

class TestGeometryComparison(unittest.TestCase):
    def setUp(self):
        import cosmo.plots as _plots
        self._tmp = tempfile.TemporaryDirectory()
        self._orig = _plots._RESULTS_ROOT
        _plots._RESULTS_ROOT = self._tmp.name

    def tearDown(self):
        import cosmo.plots as _plots
        _plots._RESULTS_ROOT = self._orig
        self._tmp.cleanup()

    def _df(self):
        import pandas as pd
        rows = []
        for geom in ["cube26", "cube_dense", "fcc", "bcc"]:
            for M in [855, 1500]:
                for S in [30, 40]:
                    rows.append({"M_factor": M, "S_gpc": S, "node_geometry": geom,
                                 "chi2_dof": 0.50 + 0.02 * len(geom) % 0.1})
        return pd.DataFrame(rows)

    def test_writes_nonempty_png(self):
        from cosmo.plots import plot_geometry_comparison
        path = plot_geometry_comparison(self._df(), "ws_test", "geom_cmp",
                                        lcdm_ref=0.436, eds_ref=0.844)
        _assert_png(path)

    def test_missing_column_raises(self):
        from cosmo.plots import plot_geometry_comparison
        import pandas as pd
        df = pd.DataFrame({"M_factor": [855], "S_gpc": [30]})  # no node_geometry/metric
        with self.assertRaises(ValueError):
            plot_geometry_comparison(df, "ws_test", "geom_bad")


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


# ---------------------------------------------------------------------------
# 10. WS4 figure-script pure helpers (_generate_ws4_figs.py)
# ---------------------------------------------------------------------------

class TestWs4FigHelpers(unittest.TestCase):
    """Unit tests for the pure (no-sim) helpers in _generate_ws4_figs.py.

    These tests import the module directly and exercise data-loading,
    best-row selection, and annotation formatting — no simulation required.
    """

    def _make_rows(self):
        """Synthetic sweep rows covering centerM {1, 2, 3} and M {1, 5}."""
        rows = []
        for M in [1, 5]:
            for cm in [1.0, 2.0, 3.0]:
                rows.append({
                    "M_factor": str(M),
                    "S_gpc": "25.0",
                    "centerM": str(cm),
                    "chi2_dof": str(0.55 - 0.02 * cm + 0.01 * M),
                    "anchor_ok": "True",
                    "growth_factor": "3.1",
                    "outer_density_ceiling": "1.0",
                    "node_mass_amplitude": "0.0",
                    "init_distribution": "uniform_sphere",
                    "node_geometry": "cube26",
                })
        # Add an anchor_ok=False row with better chi2 (should be deprioritised)
        rows.append({
            "M_factor": "1",
            "S_gpc": "20.0",
            "centerM": "1.0",
            "chi2_dof": "0.10",   # very good but unphysical
            "anchor_ok": "False",
            "growth_factor": "99.0",
            "outer_density_ceiling": "1.0",
            "node_mass_amplitude": "0.0",
            "init_distribution": "uniform_sphere",
            "node_geometry": "cube26",
        })
        return rows

    def test_select_best_row_prefers_anchor_ok(self):
        """select_best_row should pick the best anchor_ok row, not the runaway."""
        from _generate_ws4_figs import select_best_row
        rows = self._make_rows()
        best = select_best_row(rows)
        assert best is not None
        # The runaway (anchor_ok=False) row has chi2=0.10 which is lower, but
        # select_best_row must prefer anchor_ok=True rows.
        assert str(best.get("anchor_ok", "")).lower() in ("true", "1"), (
            f"Expected anchor_ok=True, got {best}"
        )

    def test_select_best_row_returns_min_chi2_among_ok(self):
        """Among anchor_ok rows the one with the lowest chi2/dof is selected."""
        from _generate_ws4_figs import select_best_row
        rows = self._make_rows()
        best = select_best_row(rows)
        assert best is not None
        ok_rows = [r for r in rows if str(r.get("anchor_ok", "")).lower() in ("true", "1")]
        min_chi2 = min(float(r["chi2_dof"]) for r in ok_rows)
        assert abs(float(best["chi2_dof"]) - min_chi2) < 1e-9

    def test_select_best_row_empty(self):
        """Empty list returns None."""
        from _generate_ws4_figs import select_best_row
        assert select_best_row([]) is None

    def test_select_best_row_all_nan(self):
        """All-NaN chi2 rows returns None."""
        from _generate_ws4_figs import select_best_row
        rows = [{"chi2_dof": "nan", "anchor_ok": "True"}]
        assert select_best_row(rows) is None

    def test_select_best_row_fallback_to_all_rows(self):
        """When no anchor_ok rows exist, falls back to all rows."""
        from _generate_ws4_figs import select_best_row
        rows = [
            {"chi2_dof": "0.7", "anchor_ok": "False"},
            {"chi2_dof": "0.6", "anchor_ok": "False"},
        ]
        best = select_best_row(rows)
        assert best is not None
        assert abs(float(best["chi2_dof"]) - 0.6) < 1e-9

    def test_format_chi2_annotation_finite(self):
        """Finite chi2 is formatted with 3 decimal places."""
        from _generate_ws4_figs import format_chi2_annotation
        s = format_chi2_annotation(0.4360, "LCDM")
        assert "LCDM" in s
        assert "0.436" in s

    def test_format_chi2_annotation_nan(self):
        """Non-finite chi2 shows n/a."""
        from _generate_ws4_figs import format_chi2_annotation
        s = format_chi2_annotation(float("nan"), "Sim")
        assert "n/a" in s

    def test_load_sweep_csv_roundtrip(self):
        """load_sweep_csv reads back rows written by csv.DictWriter."""
        import csv as _csv_mod
        import tempfile
        from _generate_ws4_figs import load_sweep_csv

        rows_in = [{"M_factor": "1", "S_gpc": "25.0", "chi2_dof": "0.55",
                    "centerM": "2.0"}]
        with tempfile.NamedTemporaryFile(
                mode="w", suffix=".csv", delete=False, encoding="utf-8", newline=""
        ) as fh:
            path = fh.name
            writer = _csv_mod.DictWriter(fh, fieldnames=list(rows_in[0].keys()))
            writer.writeheader()
            writer.writerows(rows_in)

        try:
            rows_out = load_sweep_csv(path)
        finally:
            os.unlink(path)

        assert len(rows_out) == 1
        assert rows_out[0]["M_factor"] == "1"
        assert rows_out[0]["chi2_dof"] == "0.55"

    def test_fig_b_pure_plot_no_sim(self):
        """generate_fig_b can plot from synthetic rows without running a sim
        when at_curves are provided as a pre-filled dict (mocked path).

        This is a lighter smoke test that only exercises the matplotlib path
        by patching run_external_node_simulation with a tiny fake.
        """
        import tempfile
        import types
        import unittest.mock as mock

        import cosmo.plots as _plots
        from _generate_ws4_figs import select_best_row

        rows = [
            {"M_factor": "1", "S_gpc": "25.0", "centerM": "1.0",
             "chi2_dof": "0.55", "anchor_ok": "True",
             "growth_factor": "3.1", "outer_density_ceiling": "1.0",
             "node_mass_amplitude": "0.0", "init_distribution": "uniform_sphere",
             "node_geometry": "cube26"},
            {"M_factor": "1", "S_gpc": "25.0", "centerM": "2.0",
             "chi2_dof": "0.53", "anchor_ok": "True",
             "growth_factor": "3.1", "outer_density_ceiling": "1.0",
             "node_mass_amplitude": "0.0", "init_distribution": "uniform_sphere",
             "node_geometry": "cube26"},
        ]

        cfg = {
            "t_start_Gyr": 5.8,
            "n_steps": 30,
            "particle_count": 50,
            "centerM": [1.0, 2.0],
            "outer_density_ceiling": 1.0,
            "outer_density_ceilings": [1.0],
        }

        # Tiny fake a(t) so the sim is never actually run
        n_snaps = 4
        t_fake = np.linspace(0.0, 13.8 - 5.8, n_snaps)
        t_abs_fake = 5.8 + t_fake
        a_fake = (t_abs_fake / 5.8) ** (2.0 / 3.0)
        a_fake /= a_fake[0]
        fake_ext = {"a": a_fake, "t_Gyr": t_fake,
                    "diameter_Gpc": np.ones(n_snaps),
                    "max_radius_Gpc": np.ones(n_snaps),
                    "H_hubble": np.ones(n_snaps)}

        with tempfile.TemporaryDirectory() as tmp:
            orig = _plots._RESULTS_ROOT
            _plots._RESULTS_ROOT = tmp
            path = None
            try:
                with mock.patch(
                    "_generate_ws4_figs.run_external_node_simulation",
                    return_value=fake_ext,
                ), mock.patch(
                    "_generate_ws4_figs.setup_simulation_context",
                    return_value=(10.0, 0.35, {
                        "t": t_fake, "diameter_Gpc": np.ones(n_snaps),
                        "H_hubble": np.ones(n_snaps), "a": a_fake,
                    }),
                ):
                    from _generate_ws4_figs import generate_fig_b
                    path = generate_fig_b(rows, cfg, out_tag="test_ws4")
                # Assert while tmp dir still exists
                _assert_png(path)
            finally:
                _plots._RESULTS_ROOT = orig


if __name__ == "__main__":
    unittest.main()
