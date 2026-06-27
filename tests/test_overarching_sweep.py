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
    _build_sim_params, _cell_from_best_row,
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
        vir_extent_couples_nodes=True,
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
        self.assertTrue(sim_params.vir_extent_couples_nodes)

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
        self.assertTrue(sweep_cfg.vir_extent_couples_nodes)

    def test_cache_key_encodes_nondefault_vir(self):
        """The cache key must encode the SAME vir_* the sim will use."""
        cfg = self._cfg(**self._NONDEFAULT)
        sim_params, sweep_cfg = self._capture_sim_params(cfg, self._vir_cell())
        key = build_cache_name(sweep_cfg, 100, 30, 1, [42])
        # Slugs (see build_cache_name): vn / vx / vr / vsp / vsg / vsm / vxcouple
        for slug in ("54vn", "2.0vx", "massfuncvr", "0.5vsp", "0.3vsg", "meanvsm",
                     "vxcouple"):
            self.assertIn(slug, key, f"cache key missing vir slug {slug!r}: {key}")
        # And the sim params agree with what the key encoded.
        self.assertEqual(sim_params.vir_n_nodes, sweep_cfg.vir_n_nodes)
        self.assertEqual(sim_params.vir_mass_rule, sweep_cfg.vir_mass_rule)
        self.assertEqual(sim_params.vir_extent_couples_nodes,
                         sweep_cfg.vir_extent_couples_nodes)

    def test_extent_coupling_changes_built_node_count_through_sweep(self):
        """Keyed == RUN through the sweep: with coupling ON, the grid the sim path
        builds has round(vir_n_nodes * extent^3) nodes (the knob changes the BUILT
        grid, not just the cache key)."""
        from cosmo.node_geometry import extent_coupled_n_nodes
        from cosmo.particles import HMEAGrid
        cfg = self._cfg(node_geometries=["virialized"], vir_n_nodes=32,
                        vir_extent=2.0, vir_extent_couples_nodes=True)
        sim_params, _ = self._capture_sim_params(cfg, self._vir_cell())
        grid = HMEAGrid(node_params=sim_params.external_params)
        self.assertEqual(len(grid.nodes), extent_coupled_n_nodes(32, 2.0))
        self.assertEqual(len(grid.nodes), 256)
        # Coupling OFF -> count stays at the base regardless of extent.
        cfg_off = self._cfg(node_geometries=["virialized"], vir_n_nodes=32,
                            vir_extent=2.0)
        sim_off, _ = self._capture_sim_params(cfg_off, self._vir_cell())
        grid_off = HMEAGrid(node_params=sim_off.external_params)
        self.assertEqual(len(grid_off.nodes), 32)

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
# 5b2. Virialized GEOMETRY-SEED axis (B3a): the node_mass_seed cache-collision
# bug + the expand_grid amp=0 seed-collapse bug.
# ---------------------------------------------------------------------------

class TestVirializedSeedAxis(unittest.TestCase):
    """B3a REGRESSION GUARD for a real bug.

    THE BUG (now fixed): for a virialized + massfunc + spread>0 run with
    node_mass_amplitude==0, the geometry seed (node_mass_seed) MATERIALLY changes
    the realized grid (it drives the log-normal mass draw + segregation
    permutation, cosmo/node_geometry.py:583-586), yet:
      (a) build_cache_name only put the seed in the cache key when an anisotropy
          amplitude was non-zero, so two seeds COLLIDED on one cache entry and
          silently returned the SAME a(t) (run-but-not-keyed inversion); and
      (b) expand_grid COLLAPSED the seed list to a single nm_seed=42 at amp=0, so a
          multi-seed config never even emitted distinct cells.
    Together these made any virialized seed comparison a silent no-op.

    The fix gates a "virseed" cache token + a no-collapse expand_grid branch on
    EXACTLY node_geometry=="virialized" AND vir_mass_rule=="massfunc" AND
    vir_mass_spread>0 — every genuine no-op case (radial rule, spread==0,
    non-virialized) stays byte-identical.
    """

    # ---- build_cache_name ----

    def _key(self, nm_seed, *, geometry="virialized", vir_mass_rule="massfunc",
             vir_mass_spread=0.8):
        sweep_cfg = _FixedSweepConfig(
            particle_count=400, n_steps=273,
            t_start_Gyr=2.9, t_duration_Gyr=10.9,
            objective="pantheon", s_min_gpc=20, s_max_gpc=80,
            node_mass_seed=nm_seed,
            node_mass_amplitude=0.0,   # the bug case: anisotropy OFF
            node_s_amplitude=0.0,
            init_distribution="uniform_sphere",
            node_geometry=geometry,
            vir_n_nodes=80,
            vir_mass_rule=vir_mass_rule,
            vir_mass_spread=vir_mass_spread,
        )
        return build_cache_name(sweep_cfg, 100, 30, 1, [42])

    def test_virialized_massfunc_spread_distinct_keys_per_seed(self):
        """THE BUG: two virialized massfunc (spread>0) runs differing ONLY by
        node_mass_seed must get DISTINCT cache keys. (Before B3a they were EQUAL.)"""
        k42 = self._key(42)
        k7 = self._key(7)
        self.assertNotEqual(
            k42, k7,
            "virialized massfunc spread>0 runs differing only by node_mass_seed "
            "MUST get distinct cache keys (regression: the seed-collision bug)")
        self.assertIn("42virseed", k42)
        self.assertIn("7virseed", k7)

    def test_virialized_radial_keys_unchanged_across_seeds(self):
        """No-op case: radial rule is deterministic (no RNG) -> seed must NOT change
        the key (byte-identical, no virseed token)."""
        k42 = self._key(42, vir_mass_rule="radial")
        k7 = self._key(7, vir_mass_rule="radial")
        self.assertEqual(k42, k7)
        self.assertNotIn("virseed", k42)

    def test_virialized_spread_zero_keys_unchanged_across_seeds(self):
        """No-op case: spread==0 draws raw_masses=ones (no RNG) -> seed must NOT
        change the key (byte-identical, no virseed token)."""
        k42 = self._key(42, vir_mass_spread=0.0)
        k7 = self._key(7, vir_mass_spread=0.0)
        self.assertEqual(k42, k7)
        self.assertNotIn("virseed", k42)

    def test_nonvirialized_keys_unchanged_across_seeds(self):
        """No-op case: non-virialized (cube26) at amp=0 -> seed must NOT change the
        key (byte-identical, no virseed token). Guards against the new token leaking
        out of the virialized branch."""
        k42 = self._key(42, geometry="cube26")
        k7 = self._key(7, geometry="cube26")
        self.assertEqual(k42, k7)
        self.assertNotIn("virseed", k42)

    # ---- expand_grid ----

    def _cfg(self, **overrides):
        cfg = dict(DEFAULT_CONFIG)
        cfg.update(overrides)
        return cfg

    def test_expand_grid_emits_one_cell_per_seed_virialized_massfunc(self):
        """expand_grid must emit one cell per seed for virialized+massfunc+spread>0
        even at node_mass_amplitude==0 (before B3a it collapsed to a single seed=42)."""
        cfg = self._cfg(
            M_values=[100], node_mass_amplitudes=[0.0],
            node_mass_seeds=[42, 7, 123], node_s_amplitudes=[0.0],
            init_distributions=["uniform_sphere"],
            node_geometries=["virialized"],
            vir_mass_rule="massfunc", vir_mass_spread=0.8,
        )
        cells = expand_grid(cfg)
        self.assertEqual(len(cells), 3)
        self.assertEqual({c["nm_seed"] for c in cells}, {42, 7, 123})

    def test_expand_grid_collapses_radial_to_one_seed(self):
        """No-op: radial rule still collapses to a single seed=42 cell at amp=0."""
        cfg = self._cfg(
            M_values=[100], node_mass_amplitudes=[0.0],
            node_mass_seeds=[42, 7, 123], node_s_amplitudes=[0.0],
            init_distributions=["uniform_sphere"],
            node_geometries=["virialized"],
            vir_mass_rule="radial", vir_mass_spread=0.8,
        )
        cells = expand_grid(cfg)
        self.assertEqual(len(cells), 1)
        self.assertEqual(cells[0]["nm_seed"], 42)

    def test_expand_grid_collapses_spread_zero_to_one_seed(self):
        """No-op: spread==0 still collapses to a single seed=42 cell at amp=0."""
        cfg = self._cfg(
            M_values=[100], node_mass_amplitudes=[0.0],
            node_mass_seeds=[42, 7, 123], node_s_amplitudes=[0.0],
            init_distributions=["uniform_sphere"],
            node_geometries=["virialized"],
            vir_mass_rule="massfunc", vir_mass_spread=0.0,
        )
        cells = expand_grid(cfg)
        self.assertEqual(len(cells), 1)
        self.assertEqual(cells[0]["nm_seed"], 42)

    def test_expand_grid_collapses_nonvirialized_to_one_seed(self):
        """No-op: a non-virialized geometry still collapses to seed=42 even if the
        config carries massfunc/spread>0 (the bypass is virialized-only)."""
        cfg = self._cfg(
            M_values=[100], node_mass_amplitudes=[0.0],
            node_mass_seeds=[42, 7, 123], node_s_amplitudes=[0.0],
            init_distributions=["uniform_sphere"],
            node_geometries=["cube26"],
            vir_mass_rule="massfunc", vir_mass_spread=0.8,
        )
        cells = expand_grid(cfg)
        self.assertEqual(len(cells), 1)
        self.assertEqual(cells[0]["nm_seed"], 42)

    # ---- keyed == run: the seed in the key == the seed the grid actually uses ----

    def _capture_sim_params(self, cfg, cell):
        """Capture the SimulationParameters the sim path runs (same pattern as
        TestVirializedThreading / TestForceLawSubstepRelaxModeThreading)."""
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

    def test_keyed_equals_run_seed_in_key_is_seed_in_grid(self):
        """The seed encoded in the cache key is the SAME seed the sim/grid uses.

        For each of two seeds: (i) build the params via the sim path and assert
        SimulationParameters.node_mass_seed == the requested seed; (ii) assert the
        cache key contains '<seed>virseed'; (iii) assert the two realized grids
        actually DIFFER (the seed reshapes the grid, not just the key)."""
        cfg = self._cfg(
            node_geometries=["virialized"], vir_n_nodes=80,
            vir_mass_rule="massfunc", vir_mass_spread=0.8,
        )
        cell42 = dict(M=100, amplitude=0.0, nm_seed=42, s_amplitude=0.0,
                      init="uniform_sphere", geometry="virialized")
        cell7 = dict(cell42, nm_seed=7)

        p42, sc42 = self._capture_sim_params(cfg, cell42)
        p7, sc7 = self._capture_sim_params(cfg, cell7)

        # (i) the sim runs the requested seed (keyed == run on the seed value).
        self.assertEqual(p42.node_mass_seed, 42)
        self.assertEqual(p7.node_mass_seed, 7)

        # (ii) the cache key encodes the SAME seed the sim runs.
        self.assertIn("42virseed", build_cache_name(sc42, 100, 30, 1, [42]))
        self.assertIn("7virseed", build_cache_name(sc7, 100, 30, 1, [42]))

        # (iii) the realized grids actually differ -> the seed is run-but-WAS-not-keyed.
        pos42, mass42 = p42.external_params.build_virialized()
        pos7, mass7 = p7.external_params.build_virialized()
        self.assertEqual(mass42.shape, mass7.shape)
        self.assertFalse(
            np.allclose(np.sort(mass42), np.sort(mass7)),
            "different node_mass_seeds must realize different virialized masses "
            "(the grid the cache key now correctly distinguishes)")


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
# 5c2. node_force_law + adaptive substep + vir_relax_mode threading (keyed==run)
# ---------------------------------------------------------------------------

class TestForceLawSubstepRelaxModeThreading(unittest.TestCase):
    """Section 4 (node_force_law / adaptive KDK sub-stepping) + Section 2 Option B
    (vir_relax_mode='gradient'). Each axis is keyed by build_cache_name off the
    SweepConfig, so it MUST also reach the SimulationParameters the sim runs, or a
    config that sets it would key on a value it never runs (a dead axis / poisoned
    cache). These knobs were keyed in build_cache_name but NOT threaded by sweep.py
    until this section — without the threading the comparison_v2 bounded-law and
    Option-B arms would silently run plummer/lattice.
    """

    def _cfg(self, **overrides):
        cfg = dict(DEFAULT_CONFIG)
        cfg.update(overrides)
        return cfg

    def _cell(self, geometry="cube26"):
        return dict(M=100, amplitude=0.0, nm_seed=42, s_amplitude=0.0,
                    init="uniform_sphere", geometry=geometry)

    def _capture(self, cfg, cell):
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

    def test_bounded_force_law_reaches_sim_and_key(self):
        cfg = self._cfg(node_softening_gpc=1.0, node_force_law="bounded")
        sim_params, sweep_cfg = self._capture(cfg, self._cell())
        self.assertEqual(sim_params.node_force_law, "bounded")
        self.assertEqual(sweep_cfg.node_force_law, "bounded")
        key = build_cache_name(sweep_cfg, 100, 30, 1, [42])
        self.assertIn("boundednlaw", key)

    def test_substep_reaches_sim_and_key(self):
        cfg = self._cfg(node_softening_gpc=1.0, node_substep_threshold=0.5,
                        node_substeps=8)
        sim_params, sweep_cfg = self._capture(cfg, self._cell())
        self.assertEqual(sim_params.node_substep_threshold, 0.5)
        self.assertEqual(sim_params.node_substeps, 8)
        self.assertEqual(sweep_cfg.node_substep_threshold, 0.5)
        self.assertEqual(sweep_cfg.node_substeps, 8)
        key = build_cache_name(sweep_cfg, 100, 30, 1, [42])
        self.assertIn("0.5nsubth", key)
        self.assertIn("8nsub", key)

    def test_gradient_relax_mode_reaches_sim_and_key(self):
        cfg = self._cfg(node_geometries=["virialized"], vir_n_nodes=40,
                        vir_relax_steps=20, vir_relax_mode="gradient",
                        vir_relax_rate=0.1, vir_hold_outer_frac=0.3)
        sim_params, sweep_cfg = self._capture(cfg, self._cell("virialized"))
        self.assertEqual(sim_params.vir_relax_mode, "gradient")
        self.assertEqual(sim_params.vir_relax_steps, 20)
        self.assertEqual(sim_params.vir_relax_rate, 0.1)
        self.assertEqual(sim_params.vir_hold_outer_frac, 0.3)
        self.assertEqual(sweep_cfg.vir_relax_mode, "gradient")
        key = build_cache_name(sweep_cfg, 100, 30, 1, [42])
        self.assertIn("gradientvrm", key)

    def test_gradient_grid_differs_from_lattice_through_sweep(self):
        """Keyed == RUN: the Option-B grid the sim path builds actually DIFFERS from
        the Option-A lattice grid (the mode changes the BUILT positions, not just the
        key). Built via the SAME external_params.build_virialized() the sim uses, so
        this guards against a future regression dropping vir_relax_mode."""
        common = dict(node_geometries=["virialized"], vir_n_nodes=40,
                      vir_mass_rule="massfunc", vir_mass_spread=0.8)
        lat_p, _ = self._capture(self._cfg(vir_relax_steps=1, vir_relax_mode="lattice",
                                           **common), self._cell("virialized"))
        grad_p, _ = self._capture(self._cfg(vir_relax_steps=20, vir_relax_mode="gradient",
                                            **common), self._cell("virialized"))
        lat_pos, _ = lat_p.external_params.build_virialized()
        grad_pos, _ = grad_p.external_params.build_virialized()
        self.assertEqual(lat_pos.shape, grad_pos.shape)
        self.assertFalse(np.allclose(lat_pos, grad_pos),
                         "gradient (Option B) grid must differ from lattice (Option A)")

    def test_defaults_no_slugs_byte_identical(self):
        """INVARIANT: default plummer / no-substep / lattice add NO cache slug and
        reach the sim as defaults — byte-identical to before this threading."""
        cfg = self._cfg(node_geometries=["virialized"], vir_n_nodes=40)
        sim_params, sweep_cfg = self._capture(cfg, self._cell("virialized"))
        self.assertEqual(sim_params.node_force_law, "plummer")
        self.assertEqual(sim_params.node_substep_threshold, 0.0)
        self.assertEqual(sim_params.node_substeps, 1)
        self.assertEqual(sim_params.vir_relax_mode, "lattice")
        key = build_cache_name(sweep_cfg, 100, 30, 1, [42])
        for slug in ("nlaw", "nsubth", "nsub", "vrm", "vrr", "vho"):
            self.assertNotIn(slug, key, f"default key must not carry {slug!r}: {key}")


# ---------------------------------------------------------------------------
# 5d. mu(z) panel params == sim-callback params (figure<->CSV chi2 reconciliation)
# ---------------------------------------------------------------------------

class TestMuZPanelParamsMatchSim(unittest.TestCase):
    """Section 1 (chi2 conflict fix): the mu(z) figure must re-run the SAME full
    config the sweep cell ran, or its annotated chi2 diverges from the CSV value.

    `_generate_mu_z_panel` now builds SimulationParameters via the SAME machinery
    as `_make_sim_callback` (`_cell_from_best_row` -> `_make_sweep_config_for_cell`
    -> `_build_sim_params`). These tests assert field-by-field that the panel's
    params equal the sim-callback's params for a virialized cell (the bug case),
    so the figure can never again silently drop geometry/vir_*/softening/start-size
    and re-run a different a(t). Hermetic — no real simulation runs.

    Fields that MUST match: every knob threaded by _make_sim_callback._sim.
    """

    # Mirror TestVirializedGridThreading._NONDEFAULT: a config differing from every
    # default on the knobs the old panel dropped.
    _NONDEFAULT = dict(
        node_geometries=["virialized"],
        geometry_kwargs={"foo": 1},
        vir_n_nodes=54,
        vir_extent=2.0,
        vir_mass_rule="massfunc",
        vir_mass_spread=0.5,
        vir_segregation=0.3,
        vir_s_metric="mean",
        vir_extent_couples_nodes=True,
        node_softening_gpc=1.0,
        start_size_scale=1.5,
    )

    # Knobs the old _generate_mu_z_panel omitted (the bug) + the always-threaded ones.
    _FIELDS = (
        "M_value", "S_value", "n_particles", "t_start_Gyr", "t_duration_Gyr",
        "n_steps", "center_node_mass", "node_mass_seed", "node_mass_amplitude",
        "node_s_amplitude", "init_distribution",
        "node_geometry", "geometry_kwargs",
        "vir_n_nodes", "vir_extent", "vir_mass_rule", "vir_mass_spread",
        "vir_segregation", "vir_s_metric", "vir_relax_steps",
        "vir_extent_couples_nodes",
        "node_softening_gpc", "start_size_scale",
    )

    def _cfg(self, **overrides):
        cfg = dict(DEFAULT_CONFIG)
        cfg.update(overrides)
        return cfg

    def _vir_cell(self):
        return dict(M=100, amplitude=0.0, nm_seed=42, s_amplitude=0.0,
                    init="uniform_sphere", geometry="virialized")

    def _best_row_for(self, cell, M=100, S=30, centerM=1):
        """The CSV row the sweep would emit for this cell (see _cofit_S_for_cell)."""
        return dict(
            M_factor=M, S_gpc=S, centerM=centerM,
            node_mass_amplitude=cell["amplitude"],
            node_s_amplitude=cell["s_amplitude"],
            node_mass_seed=cell["nm_seed"],
            init_distribution=cell["init"],
            node_geometry=cell["geometry"],
            chi2_dof=0.903,
        )

    def _sim_callback_params(self, cfg, cell, M, S, centerM):
        """The SimulationParameters the REAL run (sim callback) hands to the sim."""
        sweep_cfg = _make_sweep_config_for_cell(cell, cfg)
        sim_cb = _make_sim_callback(sweep_cfg, box_size_Gpc=10.0, a_start=0.1)
        captured = {}

        def fake_run(sim_params, box_size_Gpc, a_start, save_interval):
            captured["params"] = sim_params
            return {"dummy": True}

        with patch("sweep.run_external_node_simulation", side_effect=fake_run), \
             patch("sweep.results_to_sim_result", return_value="ok"):
            sim_cb(M_factor=M, S_gpc=S, centerM=centerM, seeds=[42])
        return captured["params"]

    def _panel_params(self, cfg, best_row, M, S, centerM):
        """The SimulationParameters the FIGURE panel builds (post-fix path)."""
        cell = _cell_from_best_row(best_row)
        sweep_cfg = _make_sweep_config_for_cell(cell, cfg)
        return _build_sim_params(sweep_cfg, M, S, centerM, seed=42)

    def test_cell_from_best_row_roundtrips(self):
        """_cell_from_best_row reconstructs the exact cell that produced the row."""
        cell = self._vir_cell()
        row = self._best_row_for(cell)
        self.assertEqual(_cell_from_best_row(row), cell)

    def test_panel_params_match_sim_params_virialized(self):
        """THE regression: panel params == sim-callback params, field-by-field,
        for a virialized cell with every dropped knob non-default."""
        cfg = self._cfg(**self._NONDEFAULT)
        cell = self._vir_cell()
        row = self._best_row_for(cell, M=100, S=30, centerM=1)
        sim_p = self._sim_callback_params(cfg, cell, M=100, S=30, centerM=1)
        panel_p = self._panel_params(cfg, row, M=100, S=30, centerM=1)
        for f in self._FIELDS:
            self.assertEqual(
                getattr(panel_p, f), getattr(sim_p, f),
                f"panel param {f!r} ({getattr(panel_p, f)!r}) != "
                f"sim param ({getattr(sim_p, f)!r}) -> figure would re-run a "
                f"different a(t) and its chi2 would diverge from the CSV")

    def test_panel_runs_virialized_not_cube26(self):
        """Direct guard against the exact bug: the old panel left node_geometry at
        the cube26 default and dropped softening -> 0.52 vs 0.903."""
        cfg = self._cfg(**self._NONDEFAULT)
        cell = self._vir_cell()
        row = self._best_row_for(cell)
        panel_p = self._panel_params(cfg, row, M=100, S=30, centerM=1)
        self.assertEqual(panel_p.node_geometry, "virialized")
        self.assertEqual(panel_p.node_softening_gpc, 1.0)
        self.assertEqual(panel_p.start_size_scale, 1.5)
        self.assertEqual(panel_p.vir_n_nodes, 54)

    def test_cube26_default_panel_is_noop(self):
        """INVARIANT: for a plain cube26 default config the panel params equal the
        sim params AND carry only defaults (the fix is a no-op for cube26)."""
        cfg = self._cfg()  # plain defaults: cube26, all knobs default
        cell = dict(M=100, amplitude=0.0, nm_seed=42, s_amplitude=0.0,
                    init="uniform_sphere", geometry="cube26")
        row = self._best_row_for(cell)
        sim_p = self._sim_callback_params(cfg, cell, M=100, S=30, centerM=1)
        panel_p = self._panel_params(cfg, row, M=100, S=30, centerM=1)
        for f in self._FIELDS:
            self.assertEqual(getattr(panel_p, f), getattr(sim_p, f), f)
        # Defaults intact: no virialized leakage.
        self.assertEqual(panel_p.node_geometry, "cube26")
        self.assertEqual(panel_p.node_softening_gpc, 0.0)
        self.assertEqual(panel_p.start_size_scale, 1.0)


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


# ---------------------------------------------------------------------------
# 13. core_v3 family: the redesigned headline comparison (FEWER, HIGHER-quality)
# ---------------------------------------------------------------------------

class TestCoreV3Family(unittest.TestCase):
    """The sweeps/core_v3/ family SUPERSEDES comparison_v2: FEWER but HIGHER-quality
    sims (2000p/546, M down to 1, S floor 3, ternary co-fit, GRF + a uniform_sphere
    control on cube26 only), isolating cube-vs-virialized-vs-softening cleanly at a
    MATCHED close-range treatment. This pins the family structure so a future edit
    cannot silently drop an axis or regress the quality knobs.

    12 core arms = 3 GRF geometries {cube26, virialized Option A lattice, Option B
    gradient} x 3 close-range treatments {none, bounded+substep, Plummer 1 Gpc}
    (9 arms) PLUS cube26 uniform_sphere x the 3 treatments (3 arms). Each arm has a
    single geometry/init/softening (config-wide scalars), 7 M values => 84 core cells.
    Arm 13 is the B3a seed sweep (3 M x 5 seeds = 15 cells).
    """

    _N_CORE_ARMS = 12
    _M_PER_ARM = 7  # M_values = [1,5,10,35,100,300,1000]
    _CORE_CELLS = _N_CORE_ARMS * _M_PER_ARM  # 84

    def _core_arm_paths(self):
        import glob
        # The 12 numbered core arms (01..12); exclude the seed arm and _manifest.json.
        d = os.path.join(_repo_root, "sweeps", "core_v3")
        paths = sorted(glob.glob(os.path.join(d, "[0-1][0-9]_*.json")))
        return [p for p in paths if "seedsweep" not in os.path.basename(p)]

    def _seed_arm_path(self):
        return os.path.join(_repo_root, "sweeps", "core_v3",
                            "13_virA_grf_bounded_seedsweep.json")

    def _core_cfgs(self):
        return [load_config(p) for p in self._core_arm_paths()]

    def test_twelve_core_arms_load_and_expand_to_84_cells(self):
        paths = self._core_arm_paths()
        self.assertEqual(len(paths), self._N_CORE_ARMS,
                         f"expected {self._N_CORE_ARMS} core_v3 arms, found {len(paths)}")
        total_cells = 0
        for p in paths:
            with self.subTest(arm=os.path.basename(p)):
                cfg = load_config(p)
                cells = expand_grid(cfg)
                self.assertEqual(len(cells), self._M_PER_ARM,
                                 f"{os.path.basename(p)}: expected {self._M_PER_ARM} "
                                 f"cells (one per M), got {len(cells)}")
                total_cells += len(cells)
        self.assertEqual(total_cells, self._CORE_CELLS,
                         f"core_v3 cell total should be {self._CORE_CELLS}, got {total_cells}")

    def test_unique_tags_per_arm(self):
        tags = [c["tag"] for c in self._core_cfgs()] + [load_config(self._seed_arm_path())["tag"]]
        self.assertEqual(len(tags), len(set(tags)),
                         "every core_v3 arm must have a UNIQUE tag (its own CSV)")

    def test_quality_knobs_on_every_core_arm(self):
        """Items 2/3: 2000p/546, M down to 1, S floor=3, ternary co-fit, pantheon
        objective — on EVERY core arm (the whole point of the redesign)."""
        for c in self._core_cfgs():
            self.assertEqual(c["particle_count"], 2000,
                             "core_v3 must run 2000 particles (not the old 400)")
            self.assertEqual(c["n_steps"], 546,
                             "core_v3 must run 546 steps (dt~20 Myr, not the old 273)")
            self.assertEqual(min(c["M_values"]), 1,
                             "core_v3 M grid must reach the low floor M=1")
            self.assertLessEqual(max(c["M_values"]), 1000,
                                 "core_v3 M grid caps at 1000")
            self.assertEqual(c["s_min_gpc"], 3, "core_v3 S floor must be 3 Gpc")
            self.assertEqual(c["s_max_gpc"], 35, "core_v3 S ceiling must be 35 Gpc")
            self.assertEqual(c["s_cofit_method"], "ternary",
                             "core_v3 must use ternary co-fit (linear is broken on "
                             "pantheon; ternary matches brute)")
            self.assertEqual(c["objective"], "pantheon")

    def test_geometry_x_treatment_triad_present(self):
        """Item 9: cube26 control AND virialized Option A (lattice) AND Option B
        (gradient), each crossed with all three matched close-range treatments."""
        cfgs = self._core_cfgs()
        geoms = set()
        relax_modes = set()
        softenings = set()
        force_laws = set()
        for c in cfgs:
            geoms.update(c["node_geometries"])
            if "virialized" in c["node_geometries"]:
                relax_modes.add(c.get("vir_relax_mode", "lattice"))
            softenings.add(c.get("node_softening_gpc", 0.0))
            force_laws.add(c.get("node_force_law", "plummer"))
        self.assertIn("cube26", geoms)
        self.assertIn("virialized", geoms)
        # Option A (lattice) AND Option B (gradient).
        self.assertIn("lattice", relax_modes)
        self.assertIn("gradient", relax_modes)
        # The three matched treatments: none (soft=0) AND a softened one (1.0).
        self.assertIn(0.0, softenings)
        self.assertIn(1.0, softenings)
        # plummer (none + Plummer 1 Gpc) AND bounded (bounded+substep).
        self.assertIn("plummer", force_laws)
        self.assertIn("bounded", force_laws)

    def test_bounded_arms_carry_validated_substep_values(self):
        """The bounded+substep arms must use the Section-4 VALIDATED values
        (node_force_law='bounded', 1 Gpc cap, threshold=2.0, substeps=8) — read from
        _generate_ws8_close_encounter.py, not invented."""
        bounded = [c for c in self._core_cfgs()
                   if c.get("node_force_law") == "bounded"]
        # 4 bounded arms: cube26-grf, virA-grf, virB-grf, and cube26-uniform.
        self.assertEqual(len(bounded), 4,
                         "expected 4 bounded+substep arms (cube26-grf, virA-grf, "
                         f"virB-grf, cube26-uniform); got {len(bounded)}")
        for c in bounded:
            self.assertEqual(c["node_softening_gpc"], 1.0)
            self.assertEqual(c["node_substep_threshold"], 2.0)
            self.assertEqual(c["node_substeps"], 8)

    def test_grf_headline_and_cube26_uniform_control(self):
        """init: GRF (sphere support) is the headline on every geometry; uniform_sphere
        appears ONLY on cube26 (the attribution control for the GRF clustering cost)."""
        grf_geoms = set()
        uniform_geoms = set()
        for c in self._core_cfgs():
            inits = c["init_distributions"]
            geom = c["node_geometries"][0]
            if "grf" in inits:
                grf_geoms.add(geom)
                self.assertEqual(c.get("grf_support"), "sphere",
                                 "GRF arms must confine the cloud to the sphere support")
            if "uniform_sphere" in inits:
                uniform_geoms.add(geom)
        # GRF runs on all three geometries.
        self.assertEqual(grf_geoms, {"cube26", "virialized"})
        # uniform_sphere ONLY on cube26 (not on virialized).
        self.assertEqual(uniform_geoms, {"cube26"})

    def test_virialized_arms_use_finer_deeper_node_count(self):
        """vir_n_nodes bumped above 80 (=150) for a finer mass function + deeper
        interior (decisions-v2)."""
        for c in self._core_cfgs():
            if "virialized" in c["node_geometries"]:
                self.assertGreater(c["vir_n_nodes"], 80,
                                   "virialized core arms must use vir_n_nodes>80")
                self.assertEqual(c["vir_n_nodes"], 150)

    def test_core_arms_have_distinct_cache_keys(self):
        """keyed==run across arms: every core arm yields a DISTINCT cache key at a
        common (M,S) — geometry/softening/init/force-law all reach the key."""
        keys = {}
        for p in self._core_arm_paths():
            cfg = load_config(p)
            cell = next(c for c in expand_grid(cfg) if c["M"] == 100)
            keys[os.path.basename(p)] = build_cache_name(
                _make_sweep_config_for_cell(cell, cfg), 100, 30, 1, [42])
        self.assertEqual(len(set(keys.values())), len(keys),
                         f"two core arms share a cache key: {keys}")

    # ---- the B3a seed arm ----

    def test_seed_arm_expands_to_15_cells_distinct_keys(self):
        """The seed arm expands to 3 M x 5 seeds = 15 cells (the R1 cache-collision
        fix makes the realizations REAL), and each (M, seed) gets a DISTINCT cache
        key carrying its <seed>virseed token (keyed==run on the geometry seed)."""
        cfg = load_config(self._seed_arm_path())
        self.assertEqual(cfg["node_mass_seeds"], [42, 7, 123, 2024, 99])
        self.assertEqual(cfg["M_values"], [35, 100, 300])
        cells = expand_grid(cfg)
        self.assertEqual(len(cells), 15,
                         "seed arm must emit 3 M x 5 seeds = 15 cells (no amp=0 collapse "
                         "for virialized+massfunc+spread>0)")
        from collections import defaultdict
        by_M = defaultdict(list)
        for c in cells:
            key = build_cache_name(_make_sweep_config_for_cell(c, cfg),
                                   c["M"], 30, 1, [42])
            by_M[c["M"]].append((c["nm_seed"], key))
        for M, lst in by_M.items():
            keys = [k for _, k in lst]
            self.assertEqual(len(set(keys)), len(keys),
                             f"seed cache keys collide at M={M} (the R1 bug would do this)")
            for seed, key in lst:
                self.assertIn(f"{seed}virseed", key,
                              f"key for M={M} seed={seed} missing virseed token: {key}")

    def test_seed_arm_is_virA_grf_bounded(self):
        """The seed arm is virialized Option A (lattice) GRF with the bounded+substep
        treatment (decisions-v2: pick the bounded+substep treatment)."""
        cfg = load_config(self._seed_arm_path())
        self.assertEqual(cfg["node_geometries"], ["virialized"])
        self.assertEqual(cfg["vir_relax_mode"], "lattice")
        self.assertEqual(cfg["init_distributions"], ["grf"])
        self.assertEqual(cfg["node_force_law"], "bounded")
        self.assertEqual(cfg["node_substeps"], 8)

    def test_smoke_config_exercises_core_knobs(self):
        """The core_v3 smoke config must parse AND set the new core knobs (so the
        smoke RUN proves keyed==run end-to-end), and stay tiny (pipeline check)."""
        p = os.path.join(_repo_root, "sweeps", "core_v3_smoke.json")
        cfg = load_config(p)
        cells = expand_grid(cfg)
        self.assertGreater(len(cells), 0)
        self.assertEqual(cfg["s_cofit_method"], "ternary")
        self.assertEqual(cfg["node_softening_gpc"], 1.0)
        self.assertEqual(cfg["node_force_law"], "bounded")
        self.assertEqual(cfg["node_substeps"], 8)
        self.assertEqual(cfg["init_distributions"], ["grf"])
        self.assertEqual(cfg.get("grf_support"), "sphere")
        self.assertIn("virialized", cfg["node_geometries"])
        self.assertIn("cube26", cfg["node_geometries"])
        self.assertLessEqual(cfg["particle_count"], 200)


# ---------------------------------------------------------------------------
# 14. GRF support cache key + keyed==run (WS5 §8 box->sphere default fix)
# ---------------------------------------------------------------------------

class TestGRFSupportCacheKey(unittest.TestCase):
    """The sample_grf default changed box->sphere (confine the GRF cloud to the
    uniform_sphere radius). That changed a(t) for the fixed tuple
    init_distribution="grf", so the grf cache key MUST distinguish the two supports
    or a pre-fix box cache would be served stale for the new sphere default. The fix
    encodes support ONLY for grf runs and ONLY for the NEW "sphere" support:

      - support="box"    -> bare pre-existing "grfinit" token (old caches stay box)
      - support="sphere" -> adds a "sphsup" discriminator (new default recomputes)

    PHYSICS_CACHE_VERSION stays "v3" (no bump) and uniform_sphere / non-grf keys are
    completely unchanged.
    """

    def _cfg(self, init="grf", grf_support="sphere"):
        return _FixedSweepConfig(
            particle_count=400, n_steps=273,
            t_start_Gyr=2.9, t_duration_Gyr=10.9,
            objective="pantheon",
            s_min_gpc=20, s_max_gpc=80,
            init_distribution=init,
            grf_support=grf_support,
        )

    def test_grf_box_key_is_the_preexisting_bare_grf_key(self):
        """REGRESSION: a grf-box key equals the bare grf key that existed BEFORE the
        sphere discriminator (so any pre-fix on-disk cache stays correctly addressed
        as box). The bare key carries the 'grfinit' init slug and NO support token."""
        k_box = build_cache_name(self._cfg(grf_support="box"), 100, 30, 1, [42])
        self.assertIn("grfinit", k_box)
        self.assertNotIn("sphsup", k_box)

    def test_grf_sphere_key_differs_from_box_key(self):
        """The new sphere default MUST force a recompute (distinct key) vs box, so the
        old box cache is never reused for the sphere a(t)."""
        k_box = build_cache_name(self._cfg(grf_support="box"), 100, 30, 1, [42])
        k_sph = build_cache_name(self._cfg(grf_support="sphere"), 100, 30, 1, [42])
        self.assertNotEqual(k_box, k_sph)
        self.assertIn("sphsup", k_sph)
        # The sphere key is the box key + the discriminator (box is the bare baseline).
        self.assertNotIn("sphsup", k_box)

    def test_uniform_sphere_key_unchanged_no_support_token(self):
        """uniform_sphere (and every non-grf) key is COMPLETELY unchanged: no grf
        init slug, and the support discriminator never leaks onto a non-grf key."""
        k_uni = build_cache_name(self._cfg(init="uniform_sphere"), 100, 30, 1, [42])
        self.assertNotIn("grfinit", k_uni)
        self.assertNotIn("sphsup", k_uni)
        # grf_support is irrelevant for uniform_sphere: key invariant to it.
        k_uni_box = build_cache_name(
            self._cfg(init="uniform_sphere", grf_support="box"), 100, 30, 1, [42])
        self.assertEqual(k_uni, k_uni_box)

    def test_uniform_sphere_key_byte_identical_to_no_grf_support_field(self):
        """The uniform_sphere key must be byte-identical to a config that predates the
        grf_support field entirely (defaults round-trip). Build a SweepConfig WITHOUT
        passing grf_support to confirm the default 'sphere' adds nothing for non-grf."""
        cfg_default = _FixedSweepConfig(
            particle_count=400, n_steps=273, t_start_Gyr=2.9, t_duration_Gyr=10.9,
            objective="pantheon", s_min_gpc=20, s_max_gpc=80,
            init_distribution="uniform_sphere",
        )
        k_default = build_cache_name(cfg_default, 100, 30, 1, [42])
        k_explicit = build_cache_name(self._cfg(init="uniform_sphere"), 100, 30, 1, [42])
        self.assertEqual(k_default, k_explicit)

    def test_physics_cache_version_unchanged_v3(self):
        """The fix must NOT bump PHYSICS_CACHE_VERSION (that would invalidate the
        byte-identical uniform_sphere v3 caches)."""
        from cosmo.parameter_sweep import PHYSICS_CACHE_VERSION
        self.assertEqual(PHYSICS_CACHE_VERSION, "v3")


class TestGRFSupportKeyedEqualsRun(unittest.TestCase):
    """keyed == run: the grf_support value that goes into the cache key MUST be the
    SAME value the sampler actually uses. The sweep threads grf_support into the sim
    via init_kwargs={"support": ...}, and build_cache_name keys off the same field.
    """

    def _cfg(self, **overrides):
        cfg = dict(DEFAULT_CONFIG)
        cfg.update(overrides)
        return cfg

    def _grf_cell(self):
        return dict(M=100, amplitude=0.0, nm_seed=42, s_amplitude=0.0,
                    init="grf", geometry="cube26")

    def _uniform_cell(self):
        return dict(M=100, amplitude=0.0, nm_seed=42, s_amplitude=0.0,
                    init="uniform_sphere", geometry="cube26")

    def _capture(self, cfg, cell):
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

    def test_grf_sphere_support_reaches_sim_init_kwargs_and_key(self):
        cfg = self._cfg(init_distributions=["grf"], grf_support="sphere")
        sim_params, sweep_cfg = self._capture(cfg, self._grf_cell())
        # Sampler sees the SAME support that keys the cache.
        self.assertEqual(sim_params.init_kwargs.get("support"), "sphere")
        self.assertEqual(sweep_cfg.grf_support, "sphere")
        key = build_cache_name(sweep_cfg, 100, 30, 1, [42])
        self.assertIn("sphsup", key)

    def test_grf_box_support_reaches_sim_init_kwargs_and_bare_key(self):
        cfg = self._cfg(init_distributions=["grf"], grf_support="box")
        sim_params, sweep_cfg = self._capture(cfg, self._grf_cell())
        self.assertEqual(sim_params.init_kwargs.get("support"), "box")
        self.assertEqual(sweep_cfg.grf_support, "box")
        key = build_cache_name(sweep_cfg, 100, 30, 1, [42])
        self.assertIn("grfinit", key)
        self.assertNotIn("sphsup", key)

    def test_uniform_sphere_init_kwargs_untouched(self):
        """For uniform_sphere the run must be byte-identical: init_kwargs stays empty
        (the sampler ignores support), regardless of grf_support."""
        cfg = self._cfg(init_distributions=["uniform_sphere"], grf_support="box")
        sim_params, _ = self._capture(cfg, self._uniform_cell())
        # init_kwargs is None -> SimulationParameters stores {} (no support injected).
        self.assertEqual(sim_params.init_kwargs, {})


if __name__ == "__main__":
    unittest.main()
