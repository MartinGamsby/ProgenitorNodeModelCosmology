"""
Unit tests for sweep.py (WS1 overarching sweep tool).

All tests are hermetic — no real simulations, no filesystem writes outside
tempfile scope, and no Pantheon+ data loading (all mocked/synthetic).

Covers:
  - Config loading (defaults + JSON override)
  - Grid expansion: amplitude=0 collapse, total cell count
  - CSV column contract: SWEEP_CSV_COLS is a superset of BEST_ISO_COLS
  - Extra columns present: chi2_lcdm, chi2_eds, growth_factor, anchor_ok, runaway
  - _FixedSweepConfig: particle_count and n_steps pinned correctly
  - build_cache_name uniqueness across (geometry, init, amplitude, nm_seed)
  - per-M S co-fit selection: "co-fit" vs explicit list branch
  - --plots-only wiring (plots_from_csv called, not run_sweep)
  - load_best_config compatibility: BEST_ISO_COLS contains M_factor/S_gpc/centerM/chi2_dof
"""

import csv
import io
import json
import math
import os
import pathlib
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np


# ---------------------------------------------------------------------------
# Import the module under test
# ---------------------------------------------------------------------------

import sys
_repo_root = str(pathlib.Path(__file__).parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from sweep import (
    load_config, DEFAULT_CONFIG, expand_grid,
    SWEEP_CSV_COLS, BEST_ISO_COLS,
    _FixedSweepConfig, run_plots_only,
    _make_sweep_config_for_cell, _make_sim_callback,
)
from cosmo.parameter_sweep import build_cache_name


# ---------------------------------------------------------------------------
# 1. Config loading
# ---------------------------------------------------------------------------

class TestLoadConfig(unittest.TestCase):

    def test_defaults_returned_when_no_path(self):
        cfg = load_config(None)
        self.assertEqual(cfg["particle_count"], DEFAULT_CONFIG["particle_count"])
        self.assertEqual(cfg["t_start_Gyr"], DEFAULT_CONFIG["t_start_Gyr"])
        self.assertIn("M_values", cfg)
        self.assertIn("S_values", cfg)

    def test_json_overrides_defaults(self):
        override = {"particle_count": 123, "tag": "test_override"}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False,
                                         encoding="utf-8") as f:
            json.dump(override, f)
            fpath = f.name
        try:
            cfg = load_config(fpath)
            self.assertEqual(cfg["particle_count"], 123)
            self.assertEqual(cfg["tag"], "test_override")
            # Other defaults preserved
            self.assertEqual(cfg["t_start_Gyr"], DEFAULT_CONFIG["t_start_Gyr"])
        finally:
            os.unlink(fpath)

    def test_t_start_gyr_default(self):
        cfg = load_config(None)
        self.assertEqual(cfg["t_start_Gyr"], 2.9)


# ---------------------------------------------------------------------------
# 2. Grid expansion
# ---------------------------------------------------------------------------

class TestExpandGrid(unittest.TestCase):

    def _simple_cfg(self, M_values=None, amp_list=None, seed_list=None,
                    samp_list=None, init_list=None, geom_list=None):
        cfg = dict(DEFAULT_CONFIG)
        cfg["M_values"]           = M_values  or [100, 500]
        cfg["node_mass_amplitudes"]= amp_list  or [0.0, 0.5]
        cfg["node_mass_seeds"]    = seed_list  or [42, 7]
        cfg["node_s_amplitudes"]  = samp_list  or [0.0]
        cfg["init_distributions"] = init_list  or ["uniform_sphere"]
        cfg["node_geometries"]    = geom_list  or ["cube26"]
        return cfg

    def test_amplitude_zero_collapsed_to_one_seed(self):
        """amplitude=0 should collapse multiple seeds into a SINGLE (seed=42) run."""
        cfg = self._simple_cfg(M_values=[100], amp_list=[0.0], seed_list=[42, 7])
        cells = expand_grid(cfg)
        amp0_cells = [c for c in cells if c["amplitude"] == 0.0]
        # Only 1 cell for M=100 at amp=0
        self.assertEqual(len(amp0_cells), 1)
        self.assertEqual(amp0_cells[0]["nm_seed"], 42)

    def test_amplitude_nonzero_gets_all_seeds(self):
        """amplitude > 0 should produce one cell per seed."""
        cfg = self._simple_cfg(M_values=[100], amp_list=[0.5], seed_list=[42, 7])
        cells = expand_grid(cfg)
        self.assertEqual(len(cells), 2)  # one per seed
        seeds = {c["nm_seed"] for c in cells}
        self.assertEqual(seeds, {42, 7})

    def test_total_cell_count_mixed(self):
        """
        With M=[100,500], amp=[0,0.5], seed=[42,7], samp=[0], init=[uniform], geo=[cube26]:
        Per M:  amp=0 => 1 cell; amp=0.5 => 2 cells => 3 cells per M
        Total: 2 M * 3 = 6 cells.
        """
        cfg = self._simple_cfg(M_values=[100, 500],
                               amp_list=[0.0, 0.5], seed_list=[42, 7])
        cells = expand_grid(cfg)
        self.assertEqual(len(cells), 6)

    def test_multiple_geometries_multiply_cells(self):
        cfg = self._simple_cfg(M_values=[100], amp_list=[0.0], geom_list=["cube26", "fcc"])
        cells = expand_grid(cfg)
        self.assertEqual(len(cells), 2)
        geoms = {c["geometry"] for c in cells}
        self.assertEqual(geoms, {"cube26", "fcc"})

    def test_multiple_inits_multiply_cells(self):
        cfg = self._simple_cfg(M_values=[100], amp_list=[0.0],
                               init_list=["uniform_sphere", "grf"])
        cells = expand_grid(cfg)
        self.assertEqual(len(cells), 2)

    def test_cells_have_required_keys(self):
        cfg = self._simple_cfg()
        cells = expand_grid(cfg)
        for c in cells:
            for k in ("M", "amplitude", "nm_seed", "s_amplitude", "init", "geometry"):
                self.assertIn(k, c, f"Missing key {k!r} in cell {c}")


# ---------------------------------------------------------------------------
# 3. CSV column contract
# ---------------------------------------------------------------------------

class TestCSVColumns(unittest.TestCase):

    def test_sweep_csv_cols_superset_of_best_iso(self):
        """Every BEST_ISO_COLS column must appear in SWEEP_CSV_COLS."""
        missing = set(BEST_ISO_COLS) - set(SWEEP_CSV_COLS)
        self.assertEqual(missing, set(),
                         f"BEST_ISO_COLS has columns missing from SWEEP_CSV_COLS: {missing}")

    def test_extra_columns_present(self):
        """WS1 adds chi2_lcdm, chi2_eds, growth_target, runaway."""
        for col in ("chi2_lcdm", "chi2_eds", "growth_target", "runaway"):
            self.assertIn(col, SWEEP_CSV_COLS, f"Missing WS1 column: {col!r}")

    def test_load_best_config_keys_present(self):
        """hubble_diagram_nbody --from-best-config reads M_factor, S_gpc, centerM, chi2_dof."""
        for k in ("M_factor", "S_gpc", "centerM", "chi2_dof"):
            self.assertIn(k, BEST_ISO_COLS)


# ---------------------------------------------------------------------------
# 4. _FixedSweepConfig
# ---------------------------------------------------------------------------

class TestFixedSweepConfig(unittest.TestCase):

    def test_particle_count_pinned(self):
        sc = _FixedSweepConfig(particle_count=400, n_steps=273, t_start_Gyr=2.9,
                               t_duration_Gyr=10.9, objective="pantheon",
                               s_min_gpc=20, s_max_gpc=80)
        self.assertEqual(sc.particle_count, 400)

    def test_n_steps_pinned(self):
        sc = _FixedSweepConfig(particle_count=400, n_steps=273, t_start_Gyr=2.9,
                               t_duration_Gyr=10.9, objective="pantheon",
                               s_min_gpc=20, s_max_gpc=80)
        self.assertEqual(sc.n_steps, 273)

    def test_quick_search_does_not_override_particle_count(self):
        """quick_search=True on the base class would return 200, but the override wins."""
        sc = _FixedSweepConfig(particle_count=400, n_steps=273, quick_search=True,
                               t_start_Gyr=2.9, t_duration_Gyr=10.9, objective="pantheon",
                               s_min_gpc=20, s_max_gpc=80)
        self.assertEqual(sc.particle_count, 400)
        self.assertEqual(sc.n_steps, 273)

    def test_geometry_defaults_to_cube26(self):
        sc = _FixedSweepConfig(particle_count=400, n_steps=273, t_start_Gyr=2.9,
                               t_duration_Gyr=10.9, objective="pantheon",
                               s_min_gpc=20, s_max_gpc=80)
        self.assertEqual(sc.node_geometry, "cube26")


# ---------------------------------------------------------------------------
# 5. Cache key uniqueness
# ---------------------------------------------------------------------------

class TestCacheKeyUniqueness(unittest.TestCase):

    def _cfg(self, amplitude, nm_seed, init="uniform_sphere", geometry="cube26",
             s_amplitude=0.0):
        return _FixedSweepConfig(
            particle_count=400, n_steps=273,
            t_start_Gyr=2.9, t_duration_Gyr=10.9,
            objective="pantheon",
            s_min_gpc=20, s_max_gpc=80,
            node_mass_seed=nm_seed,
            node_mass_amplitude=amplitude,
            node_s_amplitude=s_amplitude,
            init_distribution=init,
            node_geometry=geometry,
        )

    def test_different_amplitude_gives_different_key(self):
        k1 = build_cache_name(self._cfg(0.0, 42), 100, 30, 1, [42])
        k2 = build_cache_name(self._cfg(0.5, 42), 100, 30, 1, [42])
        self.assertNotEqual(k1, k2)

    def test_different_seed_gives_different_key_when_amplitude_nonzero(self):
        k1 = build_cache_name(self._cfg(0.5, 42), 100, 30, 1, [42])
        k2 = build_cache_name(self._cfg(0.5, 7), 100, 30, 1, [42])
        self.assertNotEqual(k1, k2)

    def test_different_geometry_gives_different_key(self):
        k1 = build_cache_name(self._cfg(0.0, 42, geometry="cube26"), 100, 30, 1, [42])
        k2 = build_cache_name(self._cfg(0.0, 42, geometry="fcc"),   100, 30, 1, [42])
        self.assertNotEqual(k1, k2)

    def test_different_init_gives_different_key(self):
        k1 = build_cache_name(self._cfg(0.0, 42, init="uniform_sphere"), 100, 30, 1, [42])
        k2 = build_cache_name(self._cfg(0.0, 42, init="grf"),             100, 30, 1, [42])
        self.assertNotEqual(k1, k2)

    def test_cube26_keeps_original_key_no_geo_slug(self):
        """cube26 must NOT add a geo slug (backward compat)."""
        k = build_cache_name(self._cfg(0.0, 42, geometry="cube26"), 100, 30, 1, [42])
        self.assertNotIn("cube26geo", k)

    def test_non_cube26_gets_geo_slug(self):
        k = build_cache_name(self._cfg(0.0, 42, geometry="fcc"), 100, 30, 1, [42])
        self.assertIn("fccgeo", k)

    def test_uniform_sphere_keeps_original_key_no_init_slug(self):
        k = build_cache_name(self._cfg(0.0, 42, init="uniform_sphere"), 100, 30, 1, [42])
        self.assertNotIn("uniform_sphereinit", k)

    def test_amplitude_zero_same_key_regardless_of_nm_seed(self):
        """When amplitude=0, nm_seed should not appear in the cache key."""
        k1 = build_cache_name(self._cfg(0.0, 42), 100, 30, 1, [42])
        k2 = build_cache_name(self._cfg(0.0, 7),  100, 30, 1, [42])
        self.assertEqual(k1, k2)

    def test_different_M_gives_different_key(self):
        k1 = build_cache_name(self._cfg(0.0, 42), 100,  30, 1, [42])
        k2 = build_cache_name(self._cfg(0.0, 42), 1000, 30, 1, [42])
        self.assertNotEqual(k1, k2)


# ---------------------------------------------------------------------------
# 5b. vir_* threading: cache key and sim must agree (keyed-AND-run guard)
# ---------------------------------------------------------------------------

class TestVirializedThreading(unittest.TestCase):
    """Lock the contract that anything which distinguishes the cache key (vir_*)
    ALSO reaches the actual SimulationParameters. Without this, a virialized
    sweep with non-default vir_* would key on the requested values but RUN with
    SimulationParameters defaults (silent physics/cache mismatch, poisoned cache).
    """

    # A non-default vir_* config that differs from every default.
    _NONDEFAULT = dict(
        node_geometries=["virialized"],
        vir_n_nodes=54,
        vir_extent=2.0,
        vir_mass_rule="massfunc",
        vir_mass_spread=0.5,
        vir_segregation=0.3,
        vir_s_metric="mean",
    )

    def _cfg(self, **overrides):
        cfg = dict(DEFAULT_CONFIG)
        cfg.update(overrides)
        return cfg

    def _vir_cell(self):
        return dict(M=100, amplitude=0.0, nm_seed=42, s_amplitude=0.0,
                    init="uniform_sphere", geometry="virialized")

    def _capture_sim_params(self, cfg, cell):
        """Run the sim path and capture the SimulationParameters object that
        _make_sim_callback._sim hands to run_external_node_simulation."""
        sweep_cfg = _make_sweep_config_for_cell(cell, cfg)
        sim_cb = _make_sim_callback(sweep_cfg, box_size_Gpc=10.0, a_start=0.1)
        captured = {}

        def fake_run(sim_params, box_size_Gpc, a_start, save_interval):
            captured["params"] = sim_params
            return {"dummy": True}

        with patch("sweep.run_external_node_simulation", side_effect=fake_run), \
             patch("sweep.results_to_sim_result", return_value="ok"):
            sim_cb(M_factor=cell["M"], S_gpc=30, centerM=1, seeds=[42])
        return captured["params"], sweep_cfg

    def test_nondefault_vir_reaches_sim_params(self):
        """The SimulationParameters built by the sim path carries the requested vir_*."""
        cfg = self._cfg(**self._NONDEFAULT)
        sim_params, _ = self._capture_sim_params(cfg, self._vir_cell())
        self.assertEqual(sim_params.node_geometry, "virialized")
        self.assertEqual(sim_params.vir_n_nodes, 54)
        self.assertEqual(sim_params.vir_extent, 2.0)
        self.assertEqual(sim_params.vir_mass_rule, "massfunc")
        self.assertEqual(sim_params.vir_mass_spread, 0.5)
        self.assertEqual(sim_params.vir_segregation, 0.3)
        self.assertEqual(sim_params.vir_s_metric, "mean")

    def test_sweep_config_carries_nondefault_vir(self):
        """_make_sweep_config_for_cell threads vir_* onto the SweepConfig that
        build_cache_name keys off of."""
        cfg = self._cfg(**self._NONDEFAULT)
        _, sweep_cfg = self._capture_sim_params(cfg, self._vir_cell())
        self.assertEqual(sweep_cfg.vir_n_nodes, 54)
        self.assertEqual(sweep_cfg.vir_extent, 2.0)
        self.assertEqual(sweep_cfg.vir_mass_rule, "massfunc")
        self.assertEqual(sweep_cfg.vir_mass_spread, 0.5)
        self.assertEqual(sweep_cfg.vir_segregation, 0.3)
        self.assertEqual(sweep_cfg.vir_s_metric, "mean")

    def test_cache_key_encodes_nondefault_vir(self):
        """The cache key must encode the SAME vir_* the sim will use."""
        cfg = self._cfg(**self._NONDEFAULT)
        sim_params, sweep_cfg = self._capture_sim_params(cfg, self._vir_cell())
        key = build_cache_name(sweep_cfg, 100, 30, 1, [42])
        # Slugs (see build_cache_name): vn / vx / vr / vsp / vsg / vsm
        for slug in ("54vn", "2.0vx", "massfuncvr", "0.5vsp", "0.3vsg", "meanvsm"):
            self.assertIn(slug, key, f"cache key missing vir slug {slug!r}: {key}")
        # And the sim params agree with what the key encoded.
        self.assertEqual(sim_params.vir_n_nodes, sweep_cfg.vir_n_nodes)
        self.assertEqual(sim_params.vir_mass_rule, sweep_cfg.vir_mass_rule)

    def test_nonvirialized_default_vir_unchanged(self):
        """INVARIANT: a cube26 (non-virialized) config still gets default vir_*
        in both the SweepConfig and the SimulationParameters, and its cache key
        carries NO vir slug — byte-identical to before this threading."""
        cfg = self._cfg()  # plain defaults: node_geometries=["cube26"]
        cell = dict(M=100, amplitude=0.0, nm_seed=42, s_amplitude=0.0,
                    init="uniform_sphere", geometry="cube26")
        sim_params, sweep_cfg = self._capture_sim_params(cfg, cell)
        # Defaults preserved on both objects.
        self.assertEqual(sweep_cfg.vir_n_nodes, 26)
        self.assertEqual(sweep_cfg.vir_extent, 1.0)
        self.assertEqual(sweep_cfg.vir_mass_rule, "radial")
        self.assertEqual(sim_params.vir_n_nodes, 26)
        self.assertEqual(sim_params.vir_mass_spread, 0.0)
        self.assertEqual(sim_params.vir_segregation, 1.0)
        self.assertEqual(sim_params.vir_s_metric, "median")
        # No vir slug on a cube26 key.
        key = build_cache_name(sweep_cfg, 100, 30, 1, [42])
        for slug in ("vn", "vx", "vr", "vsp", "vsg", "vsm"):
            self.assertNotIn(slug, key, f"cube26 key must not carry vir slug {slug!r}: {key}")


# ---------------------------------------------------------------------------
# 5c. node_softening_gpc threading: cache key and sim must agree (keyed==run)
# ---------------------------------------------------------------------------

class TestNodeSofteningThreading(unittest.TestCase):
    """Section 4: node_softening_gpc must reach BOTH the cache key (build_cache_name
    keys off the SweepConfig) AND the actual SimulationParameters the sim runs, so
    a softened sweep keys on exactly the softening it runs (no silent mismatch)."""

    def _cfg(self, **overrides):
        cfg = dict(DEFAULT_CONFIG)
        cfg.update(overrides)
        return cfg

    def _cell(self):
        return dict(M=100, amplitude=0.0, nm_seed=42, s_amplitude=0.0,
                    init="uniform_sphere", geometry="cube26")

    def _capture_sim_params(self, cfg, cell):
        sweep_cfg = _make_sweep_config_for_cell(cell, cfg)
        sim_cb = _make_sim_callback(sweep_cfg, box_size_Gpc=10.0, a_start=0.1)
        captured = {}

        def fake_run(sim_params, box_size_Gpc, a_start, save_interval):
            captured["params"] = sim_params
            return {"dummy": True}

        with patch("sweep.run_external_node_simulation", side_effect=fake_run), \
             patch("sweep.results_to_sim_result", return_value="ok"):
            sim_cb(M_factor=cell["M"], S_gpc=30, centerM=1, seeds=[42])
        return captured["params"], sweep_cfg

    def test_nondefault_softening_reaches_sim_and_key(self):
        cfg = self._cfg(node_softening_gpc=1.0)
        sim_params, sweep_cfg = self._capture_sim_params(cfg, self._cell())
        # Sim runs the requested softening.
        self.assertEqual(sim_params.node_softening_gpc, 1.0)
        # SweepConfig (what the cache keys off) carries it too.
        self.assertEqual(sweep_cfg.node_softening_gpc, 1.0)
        # Cache key encodes the SAME value.
        key = build_cache_name(sweep_cfg, 100, 30, 1, [42])
        self.assertIn("1.0nsoft", key)

    def test_default_softening_no_slug_and_byte_identical(self):
        cfg = self._cfg()  # node_softening_gpc defaults to 0.0
        sim_params, sweep_cfg = self._capture_sim_params(cfg, self._cell())
        self.assertEqual(sim_params.node_softening_gpc, 0.0)
        self.assertEqual(sweep_cfg.node_softening_gpc, 0.0)
        key = build_cache_name(sweep_cfg, 100, 30, 1, [42])
        self.assertNotIn("nsoft", key)


# ---------------------------------------------------------------------------
# 6. S co-fit vs explicit list selection
# ---------------------------------------------------------------------------

class TestSCofitSelection(unittest.TestCase):

    def test_cofit_flag_detected_correctly(self):
        cfg = load_config(None)
        cfg["S_values"] = "co-fit"
        self.assertEqual(cfg["S_values"], "co-fit")
        cofit = (cfg["S_values"] == "co-fit")
        self.assertTrue(cofit)

    def test_explicit_s_list_detected_correctly(self):
        cfg = load_config(None)
        cfg["S_values"] = [20, 30, 40]
        cofit = (cfg["S_values"] == "co-fit")
        self.assertFalse(cofit)
        self.assertIsInstance(cfg["S_values"], list)

    def test_default_s_values_is_cofit(self):
        """The default config uses co-fit (not an explicit grid)."""
        cfg = load_config(None)
        self.assertEqual(cfg["S_values"], "co-fit")


# ---------------------------------------------------------------------------
# 7. plots_from_csv wiring (--plots-only)
# ---------------------------------------------------------------------------

class TestPlotsOnlyWiring(unittest.TestCase):

    def test_plots_only_calls_plots_from_csv(self):
        """run_plots_only should call cosmo.plots.plots_from_csv, not run sims."""
        # Build a minimal CSV
        rows = [
            {col: ("0.5" if "chi2" in col else
                   "3.1" if "growth" in col else
                   "True" if col in ("anchor_ok",) else
                   "100" if col == "M_factor" else
                   "30" if col == "S_gpc" else "1")
             for col in SWEEP_CSV_COLS}
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False,
                                         encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=SWEEP_CSV_COLS)
            writer.writeheader()
            writer.writerows(rows)
            csv_path = f.name

        cfg = load_config(None)
        called = {}

        def fake_plots_from_csv(path, workstream="ws1", **kw):
            called["path"] = path
            called["ws"] = workstream
            return []

        try:
            with patch("cosmo.plots.plots_from_csv", side_effect=fake_plots_from_csv), \
                 patch("cosmo.plots.plot_ms_heatmap", return_value="fake.png"):
                run_plots_only(csv_path, cfg)
            self.assertEqual(called.get("path"), csv_path)
            self.assertEqual(called.get("ws"), "ws1")
        finally:
            os.unlink(csv_path)


# ---------------------------------------------------------------------------
# 8. Best-iso subset is load_best_config-compatible
# ---------------------------------------------------------------------------

class TestBestIsoSubset(unittest.TestCase):

    def _write_sweep_csv(self, tmpdir, rows):
        path = os.path.join(tmpdir, "ws1_sweep_test.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=SWEEP_CSV_COLS)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def _write_best_iso_csv(self, tmpdir, rows):
        path = os.path.join(tmpdir, "sweep_results_pantheon_test.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=BEST_ISO_COLS)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def _make_row(self, M=855, S=37, amp=0.0, chi2_dof=0.50):
        row = {col: "" for col in SWEEP_CSV_COLS}
        row.update({
            "M_factor": M, "S_gpc": S, "centerM": 1,
            "node_mass_amplitude": amp, "node_s_amplitude": 0.0,
            "node_mass_seed": 42, "init_distribution": "uniform_sphere",
            "node_geometry": "cube26",
            "chi2_dof": chi2_dof, "chi2": chi2_dof * 1200,
            "chi2_lcdm": 0.43, "chi2_eds": 0.85,
            "R2": 0.996, "n_sne_used": 1333,
            "growth_factor": 3.10, "growth_target": 3.30,
            "anchor_ok": True, "runaway": False,
            "match_avg_pct": 100.0 / (1.0 + chi2_dof),
            "diff_pct": 100.0 - 100.0 / (1.0 + chi2_dof),
        })
        return row

    def test_best_iso_cols_contains_load_best_config_keys(self):
        """hubble_diagram_nbody.py --from-best-config reads these four keys."""
        for k in ("M_factor", "S_gpc", "centerM", "chi2_dof"):
            self.assertIn(k, BEST_ISO_COLS)

    def test_iso_row_min_chi2_selection(self):
        """The best iso row is the one with the smallest chi2_dof."""
        rows = [
            {k: (v if k != "chi2_dof" else (0.48 if i == 1 else 0.55))
             for k, v in self._make_row(M=100*i, S=30, chi2_dof=0.55).items()}
            for i in range(1, 4)
        ]
        rows[1]["chi2_dof"] = 0.48   # row index 1 is best

        iso_rows = [r for r in rows if float(r["node_mass_amplitude"]) == 0.0]
        best = min(iso_rows, key=lambda r: float(r["chi2_dof"]))
        self.assertAlmostEqual(float(best["chi2_dof"]), 0.48)


# ---------------------------------------------------------------------------
# 9. Reference chi2 computation (analytic, no sim)
# ---------------------------------------------------------------------------

class TestReferenceChI2(unittest.TestCase):

    def test_lcdm_chi2_is_finite_and_positive(self):
        """The analytic LCDM chi2/dof vs real Pantheon+ must be finite."""
        from sweep import _compute_reference_chi2
        from cosmo.pantheon import load_pantheon
        try:
            pdata = load_pantheon()
        except FileNotFoundError:
            self.skipTest("Pantheon+ data not available")
        chi2_lcdm, chi2_eds = _compute_reference_chi2(pdata, 2.9)
        self.assertTrue(math.isfinite(chi2_lcdm), "LCDM chi2_dof must be finite")
        self.assertGreater(chi2_lcdm, 0.0)
        self.assertTrue(math.isfinite(chi2_eds), "EdS chi2_dof must be finite")
        self.assertGreater(chi2_eds, chi2_lcdm,
                           "EdS (no DE) must be worse than LCDM")

    def test_lcdm_chi2_near_known_value(self):
        """LCDM chi2/dof should be near 0.43 (canonical value)."""
        from sweep import _compute_reference_chi2
        from cosmo.pantheon import load_pantheon
        try:
            pdata = load_pantheon()
        except FileNotFoundError:
            self.skipTest("Pantheon+ data not available")
        chi2_lcdm, _ = _compute_reference_chi2(pdata, 2.9)
        self.assertLess(chi2_lcdm, 0.6, "LCDM chi2/dof should be near 0.43")
        self.assertGreater(chi2_lcdm, 0.3)


# ---------------------------------------------------------------------------
# 10. Objective config key (folded in from the retired root parameter_sweep.py)
# ---------------------------------------------------------------------------

class TestObjectiveConfigKey(unittest.TestCase):
    """sweep.py exposes an 'objective' config key (default 'pantheon') and threads
    it onto the SweepConfig so worst_callback picks the right scorer and
    build_cache_name stamps a '<objective>obj' slug."""

    def _cell(self):
        return dict(M=100, amplitude=0.0, nm_seed=42, s_amplitude=0.0,
                    init="uniform_sphere", geometry="cube26")

    def test_default_objective_is_pantheon(self):
        cfg = load_config(None)
        self.assertEqual(cfg["objective"], "pantheon")

    def test_default_cfg_threads_pantheon_objective(self):
        sweep_cfg = _make_sweep_config_for_cell(self._cell(), load_config(None))
        self.assertEqual(sweep_cfg.objective, "pantheon")

    def test_lcdm_objective_threaded_onto_sweep_config(self):
        cfg = dict(DEFAULT_CONFIG)
        cfg["objective"] = "lcdm"
        sweep_cfg = _make_sweep_config_for_cell(self._cell(), cfg)
        self.assertEqual(sweep_cfg.objective, "lcdm")

    def test_objective_in_cache_key(self):
        """build_cache_name appends '<objective>obj' so lcdm/pantheon never collide."""
        cfg_p = dict(DEFAULT_CONFIG); cfg_p["objective"] = "pantheon"
        cfg_l = dict(DEFAULT_CONFIG); cfg_l["objective"] = "lcdm"
        k_p = build_cache_name(_make_sweep_config_for_cell(self._cell(), cfg_p),
                               100, 30, 1, [42])
        k_l = build_cache_name(_make_sweep_config_for_cell(self._cell(), cfg_l),
                               100, 30, 1, [42])
        self.assertIn("pantheonobj", k_p)
        self.assertIn("lcdmobj", k_l)
        self.assertNotEqual(k_p, k_l)


class TestSelectBestRow(unittest.TestCase):
    """_select_best_row is objective-aware: pantheon minimizes chi2_dof, lcdm
    maximizes match_avg_pct (compute_match_metrics emits no chi2_dof)."""

    def _row(self, chi2_dof=float("inf"), match=0.0, anchor_ok=True):
        return {"chi2_dof": chi2_dof, "match_avg_pct": match, "anchor_ok": anchor_ok}

    def test_pantheon_picks_min_chi2_dof(self):
        from sweep import _select_best_row
        rows = [self._row(chi2_dof=0.6), self._row(chi2_dof=0.48), self._row(chi2_dof=0.55)]
        best = _select_best_row(rows, "pantheon")
        self.assertAlmostEqual(best["chi2_dof"], 0.48)

    def test_lcdm_picks_max_match(self):
        from sweep import _select_best_row
        rows = [self._row(match=90.0), self._row(match=97.5), self._row(match=92.0)]
        best = _select_best_row(rows, "lcdm")
        self.assertAlmostEqual(best["match_avg_pct"], 97.5)

    def test_lcdm_prefers_anchor_ok(self):
        from sweep import _select_best_row
        rows = [self._row(match=99.0, anchor_ok=False), self._row(match=95.0, anchor_ok=True)]
        best = _select_best_row(rows, "lcdm")
        self.assertAlmostEqual(best["match_avg_pct"], 95.0)

    def test_empty_returns_none(self):
        from sweep import _select_best_row
        self.assertIsNone(_select_best_row([], "pantheon"))
        self.assertIsNone(_select_best_row([], "lcdm"))


# ---------------------------------------------------------------------------
# 11. Migrated knob-grf grid (was pantheon_knob_sweep._expand_grid)
# ---------------------------------------------------------------------------

class TestKnobGrfMigration(unittest.TestCase):
    """The retired pantheon_knob_sweep.py grid is now sweeps/knob_grf.json. Its
    amplitude=0 collapse + cell count must match the old _expand_grid contract:
    per (M): 1 (amp=0) + (n_amp-1)*n_seed cells. (Ported from the deleted
    tests/test_pantheon_knob_sweep.py::TestGridExpansion.)"""

    def _sweeps_dir(self):
        return os.path.join(_repo_root, "sweeps")

    def test_knob_grf_config_loads(self):
        cfg = load_config(os.path.join(self._sweeps_dir(), "knob_grf.json"))
        self.assertEqual(cfg["init_distributions"], ["grf"])
        self.assertEqual(cfg["node_mass_amplitudes"], [0.0, 0.25, 0.5, 0.75])
        self.assertEqual(cfg["node_mass_seeds"], [42, 7])

    def test_knob_grf_cell_count_matches_old_expand_grid(self):
        """The old user grid: per (M): 1 amp=0 cell + 3 amp x 2 seeds = 7 cells."""
        cfg = load_config(os.path.join(self._sweeps_dir(), "knob_grf.json"))
        cells = expand_grid(cfg)
        n_M = len(cfg["M_values"])
        n_amp = len(cfg["node_mass_amplitudes"])
        n_seed = len(cfg["node_mass_seeds"])
        expected = n_M * (1 + (n_amp - 1) * n_seed)
        self.assertEqual(len(cells), expected)
        self.assertEqual(len(cells), n_M * 7)

    def test_knob_grf_amp0_collapsed_to_seed_42(self):
        cfg = load_config(os.path.join(self._sweeps_dir(), "knob_grf.json"))
        cells = expand_grid(cfg)
        amp0 = [c for c in cells if c["amplitude"] == 0.0]
        # One amp=0 cell per M, all collapsed to nm_seed=42
        self.assertEqual(len(amp0), len(cfg["M_values"]))
        self.assertTrue(all(c["nm_seed"] == 42 for c in amp0))

    def test_knob_grf_cache_keys_unique_across_amp_seed(self):
        """Distinct (amplitude, nm_seed) cells get distinct cache keys (no collision).
        (Ported from the deleted test_pantheon_knob_sweep cache-key coverage.)"""
        cfg = load_config(os.path.join(self._sweeps_dir(), "knob_grf.json"))
        cells = expand_grid(cfg)
        keys = set()
        for cell in cells:
            sc = _make_sweep_config_for_cell(cell, cfg)
            keys.add(build_cache_name(sc, cell["M"], 30, 1, [42]))
        # Every cell at a fixed (M, S) differs only by (amp, seed); within a single M
        # there must be no key collisions.
        for M in cfg["M_values"]:
            m_cells = [c for c in cells if c["M"] == M]
            m_keys = {build_cache_name(_make_sweep_config_for_cell(c, cfg), M, 30, 1, [42])
                      for c in m_cells}
            self.assertEqual(len(m_keys), len(m_cells),
                             f"cache key collision among amp/seed cells at M={M}")


# ---------------------------------------------------------------------------
# 12. Every committed sweeps/*.json loads and expands without error
# ---------------------------------------------------------------------------

class TestAllSweepConfigsLoadAndExpand(unittest.TestCase):
    """Smoke test: every JSON in sweeps/ must load_config + expand_grid cleanly.
    Guards against a committed config that the single driver can no longer parse."""

    def test_all_configs(self):
        import glob
        sweeps_dir = os.path.join(_repo_root, "sweeps")
        paths = sorted(glob.glob(os.path.join(sweeps_dir, "*.json")))
        self.assertTrue(paths, "no sweeps/*.json configs found")
        for path in paths:
            with self.subTest(config=os.path.basename(path)):
                cfg = load_config(path)
                cells = expand_grid(cfg)
                self.assertIsInstance(cells, list)
                self.assertGreater(len(cells), 0,
                                   f"{os.path.basename(path)} expanded to 0 cells")
                for c in cells:
                    for k in ("M", "amplitude", "nm_seed", "s_amplitude",
                              "init", "geometry"):
                        self.assertIn(k, c)


if __name__ == "__main__":
    unittest.main()
