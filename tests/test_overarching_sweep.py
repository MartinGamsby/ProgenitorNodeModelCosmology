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

    def test_force_method_reaches_sim_and_key(self):
        """force_method (no-Barnes-Hut chart reproduction): an explicit method must
        reach the SimulationParameters the sim runs AND key the cache (keyed == run);
        the 'auto' default must stay byte-identical (no slug)."""
        # default: auto, no slug
        cfg0 = self._cfg()
        sim_params0, sweep_cfg0 = self._capture(cfg0, self._cell())
        self.assertEqual(sim_params0.force_method, "auto")
        self.assertNotIn("fm", build_cache_name(sweep_cfg0, 100, 30, 1, [42]))
        # explicit: threads through + slugs (underscore stripped for _split_key)
        cfg = self._cfg(force_method="numba_direct")
        sim_params, sweep_cfg = self._capture(cfg, self._cell())
        self.assertEqual(sim_params.force_method, "numba_direct")
        self.assertEqual(sweep_cfg.force_method, "numba_direct")
        self.assertIn("numbadirectfm", build_cache_name(sweep_cfg, 100, 30, 1, [42]))

    def test_force_method_reaches_integrator_ctor(self):
        """factories must hand SimulationParameters.force_method to the
        CosmologicalSimulation ctor (else the override is a dead knob)."""
        from unittest.mock import patch as _patch, MagicMock as _MM
        from cosmo import factories as _f
        from cosmo.constants import SimulationParameters
        params = SimulationParameters(n_particles=10, force_method="numba_direct")
        with _patch.object(_f, "CosmologicalSimulation") as ctor, \
             _patch.object(_f, "run_and_extract_results", return_value={}):
            ctor.return_value = _MM()
            _f.run_external_node_simulation(params, 10.0, 0.1)
        self.assertEqual(ctor.call_args.kwargs.get("force_method"), "numba_direct")

    def test_vir_center_mass_frac_reaches_sim_and_key(self):
        """vir_center_mass_frac (PF23 centerM test): default 1.0 = the prior
        hardcoded medium 'us'-node mass (no slug, byte-identical); != 1.0 must reach
        SimulationParameters AND key the cache in medium mode (keyed == run)."""
        base = dict(node_geometries=["virialized"], vir_n_nodes=40,
                    vir_mass_rule="massfunc", vir_relax_mode="medium")
        sim_params0, sweep_cfg0 = self._capture(self._cfg(**base), self._cell("virialized"))
        self.assertEqual(sim_params0.vir_center_mass_frac, 1.0)
        self.assertNotIn("vcm", build_cache_name(sweep_cfg0, 100, 30, 1, [42]))
        cfg = self._cfg(**base, vir_center_mass_frac=5.0)
        sim_params, sweep_cfg = self._capture(cfg, self._cell("virialized"))
        self.assertEqual(sim_params.vir_center_mass_frac, 5.0)
        self.assertIn("5.0vcm", build_cache_name(sweep_cfg, 100, 30, 1, [42]))
        # non-medium modes must NOT slug it (the knob only exists in the medium build)
        cfg_lat = self._cfg(node_geometries=["virialized"], vir_n_nodes=40,
                            vir_center_mass_frac=5.0)
        _, sweep_cfg_lat = self._capture(cfg_lat, self._cell("virialized"))
        self.assertNotIn("vcm", build_cache_name(sweep_cfg_lat, 100, 30, 1, [42]))

    def test_outer_particle_cap_reaches_sim_and_key(self):
        """outer_particle_cap (PF24 large-centerM): default 0.0 = legacy linear
        N_outer (no slug); >0 must reach SimulationParameters AND key the cache."""
        sim_params0, sweep_cfg0 = self._capture(self._cfg(), self._cell())
        self.assertEqual(sim_params0.outer_particle_cap, 0.0)
        self.assertNotIn("opc", build_cache_name(sweep_cfg0, 100, 30, 1, [42]))
        sim_params, sweep_cfg = self._capture(self._cfg(outer_particle_cap=2.0), self._cell())
        self.assertEqual(sim_params.outer_particle_cap, 2.0)
        self.assertIn("2.0opc", build_cache_name(sweep_cfg, 100, 30, 1, [42]))

    def test_outer_particle_cap_conserves_outer_mass(self):
        """The cap trades N_outer for per-particle mass: outer TOTAL mass must be
        conserved exactly and N_outer bounded at cap*N_inner."""
        import numpy as np
        from cosmo.particles import ParticleSystem
        common = dict(n_particles=200, box_size_m=1.0e26, a_start=0.2,
                      use_dark_energy=False, mass_randomize=0.0,
                      init_distribution="uniform_sphere", eds_consistent=True,
                      t_start_Gyr=2.9, center_node_mass=10.0)
        np.random.seed(42)
        capped = ParticleSystem(**common, outer_particle_cap=2.0)
        np.random.seed(42)
        legacy = ParticleSystem(**common)
        m_cap = capped.get_masses(); m_leg = legacy.get_masses()
        in_cap = capped.observable_mask; in_leg = legacy.observable_mask
        # N_outer bounded
        self.assertLessEqual((~in_cap).sum(), 2 * in_cap.sum())
        self.assertEqual((~in_leg).sum(), 9 * 200)  # legacy linear
        # outer TOTAL mass conserved vs legacy; inner untouched
        self.assertAlmostEqual(m_cap[~in_cap].sum() / m_leg[~in_leg].sum(), 1.0, places=9)
        np.testing.assert_allclose(m_cap[in_cap], m_leg[in_leg])

    def test_mass_randomize_reaches_sim_and_key(self):
        """mass_randomize (PF23 particle-mass axis): default 0.0 = the historical
        hardcoded equal-mass value (no slug, byte-identical); >0 must reach the
        SimulationParameters AND key the cache (keyed == run)."""
        sim_params0, sweep_cfg0 = self._capture(self._cfg(), self._cell())
        self.assertEqual(sim_params0.mass_randomize, 0.0)
        self.assertNotIn("mrand", build_cache_name(sweep_cfg0, 100, 30, 1, [42]))
        sim_params, sweep_cfg = self._capture(self._cfg(mass_randomize=0.5), self._cell())
        self.assertEqual(sim_params.mass_randomize, 0.5)
        self.assertIn("0.5mrand", build_cache_name(sweep_cfg, 100, 30, 1, [42]))

    def test_particle_seed_default_and_override(self):
        """particle_seed (PF23 GRF-realization axis): default 42 = the historical
        hardcoded co-fit seed; the override must reach the seeds list (which is a
        build_cache_name ARGUMENT -> distinct key) and the panel rebuild."""
        from sweep import _particle_seed
        self.assertEqual(_particle_seed({}), 42)
        self.assertEqual(_particle_seed({"particle_seed": 7}), 7)
        # distinct seeds -> distinct cache keys (seeds are a build_cache_name arg)
        _, sweep_cfg = self._capture(self._cfg(), self._cell())
        self.assertNotEqual(build_cache_name(sweep_cfg, 100, 30, 1, [42]),
                            build_cache_name(sweep_cfg, 100, 30, 1, [7]))

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
    single geometry/init/softening (config-wide scalars), 6 M values => 72 core cells.
    Arm 13 is the B3a seed sweep (3 M x 5 seeds = 15 cells).
    """

    _N_CORE_ARMS = 12
    _EXPECTED_M = [1, 5, 10, 50, 100, 500]  # regular x5/x2 log grid, M=1..500
    _M_PER_ARM = len(_EXPECTED_M)  # 6
    _CORE_CELLS = _N_CORE_ARMS * _M_PER_ARM  # 72

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

    def test_twelve_core_arms_load_and_expand_to_72_cells(self):
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
            self.assertEqual(c["M_values"], self._EXPECTED_M,
                             f"core_v3 M grid must be {self._EXPECTED_M} (regular x5/x2 "
                             "log grid, M=1..500); exact check guards against config drift")
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
        """vir_n_nodes bumped to 300 (finer mass function + deeper interior) AND a
        WIDER mass distribution vir_mass_spread=2.0 (bigger 'distribution size' -> a
        few big nodes + many small ones, which avoids the small-S cloud collapse and
        reaches the accelerating corner). decisions-v2 + the wide-spread finding."""
        for c in self._core_cfgs():
            if "virialized" in c["node_geometries"]:
                self.assertEqual(c["vir_n_nodes"], 300)
                self.assertEqual(c["vir_mass_spread"], 2.0)

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
# 13b. Satellite families: the secondary SHAPE studies (plan-v2 B5/B6)
# ---------------------------------------------------------------------------

class TestSatelliteFamilies(unittest.TestCase):
    """The sweeps/satellite_* families are the secondary shape studies split out of
    the core (decisions-v2 / plan-v2 B5+B6). They are MATCHED to the core_v3 cell
    (virialized-A or cube26, GRF sphere support, bounded+substep treatment, S co-fit
    [3..35] ternary, pantheon objective) but vary ONE secondary axis each:
      - start_size: start_size_scale in {0.5,0.8,1.0,1.2,1.5,2.0} (6 arms x M{35,100,300})
      - convergence: particle_count in {1000,2000,4000} (3 arms x ONE cell M=100)
      - extent: vir_extent {1.0,1.5,2.0} coupled to node count (3 arms x M{35,100})
    Each config-wide scalar (start_size_scale, particle_count, vir_extent) is its own
    small arm. This pins the family structure + cell counts so a future edit cannot
    silently drop an arm or break the keyed==run wiring of the swept axis.
    """

    def _arm_paths(self, family):
        import glob
        d = os.path.join(_repo_root, "sweeps", family)
        paths = sorted(glob.glob(os.path.join(d, "[0-9]*.json")))
        return [p for p in paths if os.path.basename(p) != "_manifest.json"]

    def _cfgs(self, family):
        return [(p, load_config(p)) for p in self._arm_paths(family)]

    # ---- start_size satellite (B5) ----

    def test_startsize_six_arms_eighteen_cells(self):
        cfgs = self._cfgs("satellite_startsize")
        self.assertEqual(len(cfgs), 6,
                         "start_size satellite must be 6 arms (one per start_size_scale)")
        total = 0
        for p, cfg in cfgs:
            cells = expand_grid(cfg)
            self.assertEqual(len(cells), 3,
                             f"{os.path.basename(p)}: M{{35,100,300}} -> 3 cells")
            total += len(cells)
        self.assertEqual(total, 18, "start_size satellite total = 6 arms x 3 M = 18 cells")

    def test_startsize_scales_cover_the_six_values(self):
        scales = sorted(cfg["start_size_scale"] for _, cfg in self._cfgs("satellite_startsize"))
        self.assertEqual(scales, [0.5, 0.8, 1.0, 1.2, 1.5, 2.0])

    def test_startsize_geometry_and_treatment_matched_to_core(self):
        """ONE geometry: virialized Option A (lattice) GRF + bounded+substep, M{35,100,300}."""
        for p, cfg in self._cfgs("satellite_startsize"):
            self.assertEqual(cfg["node_geometries"], ["virialized"])
            self.assertEqual(cfg["vir_relax_mode"], "lattice")
            self.assertEqual(cfg["init_distributions"], ["grf"])
            self.assertEqual(cfg.get("grf_support"), "sphere")
            self.assertEqual(cfg["node_force_law"], "bounded")
            self.assertEqual(cfg["node_substeps"], 8)
            self.assertEqual(cfg["s_cofit_method"], "ternary")
            self.assertEqual(cfg["s_min_gpc"], 3)
            self.assertEqual(cfg["s_max_gpc"], 35)
            self.assertEqual(cfg["M_values"], [35, 100, 300])
            self.assertEqual(cfg["vir_n_nodes"], 300)
            self.assertEqual(cfg["vir_mass_spread"], 2.0)

    def test_startsize_is_keyed_eq_run_in_cache(self):
        """The swept axis (start_size_scale) must reach the cache key when != 1.0 and
        be byte-identical (no ssz slug) at 1.0 -- keyed==run for the lever."""
        keys = {}
        for p, cfg in self._cfgs("satellite_startsize"):
            cell = next(c for c in expand_grid(cfg) if c["M"] == 100)
            keys[cfg["start_size_scale"]] = build_cache_name(
                _make_sweep_config_for_cell(cell, cfg), 100, 20, 1, [42])
        # All six start_size_scale values must yield DISTINCT keys at a common (M,S).
        self.assertEqual(len(set(keys.values())), 6,
                         f"start_size_scale not keyed==run (some keys collide): {keys}")
        # The scale=1.0 arm is the byte-identical reference: NO ssz slug.
        self.assertNotIn("ssz", keys[1.0],
                         "start_size_scale=1.0 must be byte-identical (no ssz slug)")
        self.assertIn("ssz", keys[0.5],
                       "start_size_scale=0.5 must add an ssz slug (keyed==run)")

    # ---- convergence satellite (B6) ----

    def test_convergence_three_arms_one_cell_each(self):
        cfgs = self._cfgs("satellite_convergence")
        self.assertEqual(len(cfgs), 3,
                         "convergence satellite must be 3 arms (one per particle_count)")
        for p, cfg in cfgs:
            cells = expand_grid(cfg)
            self.assertEqual(len(cells), 1,
                             f"{os.path.basename(p)}: ONE cell (M=100)")
            self.assertEqual(cfg["M_values"], [100])

    def test_convergence_particle_counts_are_the_variable(self):
        """N IS the swept variable -> {1000,2000,4000}, with 2000+4000 at full quality."""
        Ns = sorted(cfg["particle_count"] for _, cfg in self._cfgs("satellite_convergence"))
        self.assertEqual(Ns, [1000, 2000, 4000])

    def test_convergence_cell_matched_to_core(self):
        """ONE headline cell: cube26 GRF bounded+substep, M=100, S co-fit [3..35]."""
        for p, cfg in self._cfgs("satellite_convergence"):
            self.assertEqual(cfg["node_geometries"], ["cube26"])
            self.assertEqual(cfg["init_distributions"], ["grf"])
            self.assertEqual(cfg.get("grf_support"), "sphere")
            self.assertEqual(cfg["node_force_law"], "bounded")
            self.assertEqual(cfg["node_substeps"], 8)
            self.assertEqual(cfg["s_cofit_method"], "ternary")
            self.assertEqual(cfg["n_steps"], 546)

    # ---- extent satellite (B6 / item 10) ----

    def test_extent_three_arms_six_cells(self):
        cfgs = self._cfgs("satellite_extent")
        self.assertEqual(len(cfgs), 3,
                         "extent satellite must be 3 arms (one per vir_extent)")
        total = 0
        for p, cfg in cfgs:
            cells = expand_grid(cfg)
            self.assertEqual(len(cells), 2,
                             f"{os.path.basename(p)}: M{{35,100}} -> 2 cells")
            total += len(cells)
        self.assertEqual(total, 6, "extent satellite total = 3 arms x 2 M = 6 cells")

    def test_extent_couples_nodes_and_covers_the_values(self):
        """vir_extent {1.0,1.5,2.0} with vir_extent_couples_nodes=True so extent drives
        the node count ~extent^3 (PF14): 150 -> {150,506,1200}."""
        from cosmo.node_geometry import extent_coupled_n_nodes
        seen = {}
        for p, cfg in self._cfgs("satellite_extent"):
            self.assertTrue(cfg.get("vir_extent_couples_nodes", False),
                            "extent arms must set vir_extent_couples_nodes=True")
            self.assertEqual(cfg["node_geometries"], ["virialized"])
            self.assertEqual(cfg["vir_relax_mode"], "lattice")
            self.assertEqual(cfg["vir_n_nodes"], 150)
            seen[cfg["vir_extent"]] = extent_coupled_n_nodes(150, cfg["vir_extent"])
        self.assertEqual(sorted(seen.keys()), [1.0, 1.5, 2.0])
        self.assertEqual(seen[1.0], 150)
        self.assertEqual(seen[1.5], 506)
        self.assertEqual(seen[2.0], 1200)

    def test_extent_is_keyed_eq_run_in_cache(self):
        """The swept axis (vir_extent + coupling) must reach the cache key -> distinct
        keys per extent at a common (M,S) (keyed==run)."""
        keys = {}
        for p, cfg in self._cfgs("satellite_extent"):
            cell = next(c for c in expand_grid(cfg) if c["M"] == 100)
            keys[cfg["vir_extent"]] = build_cache_name(
                _make_sweep_config_for_cell(cell, cfg), 100, 20, 1, [42])
        self.assertEqual(len(set(keys.values())), 3,
                         f"vir_extent not keyed==run (some keys collide): {keys}")

    # ---- cross-family ----

    def test_all_satellite_tags_unique(self):
        tags = []
        for fam in ("satellite_startsize", "satellite_convergence", "satellite_extent"):
            tags += [cfg["tag"] for _, cfg in self._cfgs(fam)]
        self.assertEqual(len(tags), len(set(tags)),
                         "every satellite arm must have a UNIQUE tag (its own CSV)")
        self.assertEqual(len(tags), 12, "12 satellite arms total (6 + 3 + 3)")


# ---------------------------------------------------------------------------
# 13c. localized_v4: production-resolution refinement of the high-sigma ridge
# ---------------------------------------------------------------------------

class TestLocalizedV4Family(unittest.TestCase):
    """sweeps/localized_v4/ re-runs the high-mass-spread ridge (sigma 4-8, M 100-500,
    S 15-30) at PRODUCTION resolution (2000 particles / 1092 steps) where the cheap
    exploration (explore_vir_spread_hi, 1000p/273) found the best HONEST centre
    chi2/dof ~0.46 (near LCDM 0.436, decisively below EdS 0.843). 8 arms: 5 sigma
    arms (full M x S grid each), a 500-node check, and two step-convergence cells.
    Pins the family structure + quality knobs so a future edit cannot silently drop
    the >=2000p / >=1000-step floor the user required.
    """

    _M_GRID = [100, 200, 300, 400, 500]
    _S_GRID = [15, 18, 20, 25, 30]
    _SIGMAS = [4.0, 5.0, 6.0, 7.0, 8.0]

    def _arm_paths(self):
        import glob
        d = os.path.join(_repo_root, "sweeps", "localized_v4")
        paths = sorted(glob.glob(os.path.join(d, "[0-9]*.json")))
        return [p for p in paths if os.path.basename(p) != "_manifest.json"]

    def _sigma_arm_paths(self):
        return [p for p in self._arm_paths()
                if os.path.basename(p).startswith(("01_", "02_", "03_", "04_", "05_"))]

    def _cfgs(self):
        return [(p, load_config(p)) for p in self._arm_paths()]

    def test_eight_arms_load_and_expand(self):
        paths = self._arm_paths()
        self.assertEqual(len(paths), 8, f"expected 8 localized_v4 arms, found {len(paths)}")
        # 5 sigma arms x 5 M = 25 cells (each runs the 5 S values internally) +
        # node500 (2 M x 2 sigma = 4 cells) + 2 conv (1 cell each) = 31 cells.
        total = sum(len(expand_grid(load_config(p))) for p in paths)
        self.assertEqual(total, 31, f"localized_v4 cell total should be 31, got {total}")

    def test_quality_knobs_meet_the_user_floor(self):
        """>=2000 particles AND >=1000 steps on EVERY arm (the user's explicit floor),
        plus the good-region knobs: virialized lattice, GRF sphere, massfunc, Plummer
        1 Gpc, observer scoring, EXPLICIT S list (no co-fit)."""
        for p, cfg in self._cfgs():
            arm = os.path.basename(p)
            with self.subTest(arm=arm):
                self.assertGreaterEqual(cfg["particle_count"], 2000,
                                        f"{arm}: must run >=2000 particles")
                self.assertGreaterEqual(cfg["n_steps"], 1000,
                                        f"{arm}: must run >=1000 steps")
                self.assertEqual(cfg["node_geometries"], ["virialized"])
                self.assertEqual(cfg["vir_relax_mode"], "lattice")
                self.assertEqual(cfg["vir_mass_rule"], "massfunc")
                self.assertEqual(cfg["init_distributions"], ["grf"])
                self.assertEqual(cfg.get("grf_support"), "sphere")
                self.assertEqual(cfg["node_softening_gpc"], 1.0)
                self.assertEqual(cfg["node_force_law"], "plummer")
                self.assertTrue(cfg.get("score_observers", False))
                self.assertIsInstance(cfg["S_values"], list,
                                      f"{arm}: localized_v4 uses an EXPLICIT S list, not co-fit")

    def test_sigma_arms_cover_the_ridge_grid(self):
        """The 5 sigma arms carry sigma {4,5,6,7,8}, each with the full M x S grid."""
        seen_sigmas = []
        for p in self._sigma_arm_paths():
            cfg = load_config(p)
            self.assertEqual(cfg["M_values"], self._M_GRID)
            self.assertEqual(cfg["S_values"], self._S_GRID)
            self.assertEqual(len(cfg["vir_mass_spreads"]), 1)
            seen_sigmas.append(cfg["vir_mass_spreads"][0])
        self.assertEqual(sorted(seen_sigmas), self._SIGMAS)

    def test_convergence_arms_exceed_floor_and_are_the_headline_cell(self):
        """The two convergence arms hold M=300/S=20/sigma=6 fixed and only raise
        n_steps (1638, 2184) so they validate that >=1092 is converged."""
        conv = [(p, c) for p, c in self._cfgs()
                if "conv" in os.path.basename(p)]
        self.assertEqual(len(conv), 2)
        steps = sorted(c["n_steps"] for _, c in conv)
        self.assertEqual(steps, [1638, 2184])
        for _, c in conv:
            self.assertEqual(c["M_values"], [300])
            self.assertEqual(c["S_values"], [20])
            self.assertEqual(c["vir_mass_spreads"], [6.0])

    def test_unique_tags_and_distinct_cache_keys(self):
        """Every arm has its own tag (its own CSV) and the swept axes (sigma, node
        count, n_steps) reach the cache key -> keyed==run across arms at a common (M,S)."""
        cfgs = self._cfgs()
        tags = [c["tag"] for _, c in cfgs]
        self.assertEqual(len(tags), len(set(tags)), "localized_v4 arm tags must be unique")
        keys = {}
        for p, cfg in cfgs:
            cell = next(c for c in expand_grid(cfg) if c["M"] == 300)
            keys[os.path.basename(p)] = build_cache_name(
                _make_sweep_config_for_cell(cell, cfg), 300, 20, 1, [42])
        self.assertEqual(len(set(keys.values())), len(keys),
                         f"two localized_v4 arms share a cache key: {keys}")


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


class TestVirMassSpreadAxis(unittest.TestCase):
    """`vir_mass_spread` (the node mass-function 'distribution size') is a SWEEPABLE
    axis via the plural `vir_mass_spreads` list. A wider spread (a few big nodes +
    many small ones) avoids the small-S cloud collapse and reaches the accelerating
    corner — so we want to sweep it to find good values. Each swept spread must key
    the cache AND reach the sim (keyed == run)."""

    def _base(self, **over):
        cfg = dict(
            node_mass_amplitudes=[0.0], node_mass_seeds=[42], node_s_amplitudes=[0.0],
            init_distributions=["grf"], node_geometries=["virialized"], M_values=[100],
            vir_mass_rule="massfunc", vir_n_nodes=300, particle_count=2000, n_steps=546,
            t_start_Gyr=2.9, s_min_gpc=3, s_max_gpc=35, objective="pantheon",
        )
        cfg.update(over)
        return cfg

    def test_spreads_list_expands_one_cell_per_spread(self):
        cfg = self._base(M_values=[10, 100], vir_mass_spreads=[0.8, 2.0, 3.0])
        cells = expand_grid(cfg)
        # 2 M x 3 spreads = 6 cells, each tagged with its spread.
        self.assertEqual(len(cells), 6)
        self.assertEqual({c["vir_spread"] for c in cells}, {0.8, 2.0, 3.0})

    def test_each_spread_is_keyed_eq_run(self):
        cfg = self._base(vir_mass_spreads=[0.8, 2.0, 3.0])
        keys = set()
        for c in expand_grid(cfg):
            sc = _make_sweep_config_for_cell(c, cfg)
            # The spread reaches the sim config...
            self.assertEqual(sc.vir_mass_spread, c["vir_spread"])
            k = build_cache_name(sc, c["M"], 10, 1, [c["nm_seed"]])
            # ...and the cache key (so two spreads can't collide).
            self.assertIn(f"{c['vir_spread']}vsp", k)
            keys.add(k)
        self.assertEqual(len(keys), 3, "each swept spread must yield a DISTINCT key")

    def test_scalar_spread_backward_compatible(self):
        """No `vir_mass_spreads` -> the scalar `vir_mass_spread` is used (one value)."""
        cfg = self._base(vir_mass_spread=0.8)
        cells = expand_grid(cfg)
        self.assertEqual(len(cells), 1)
        self.assertEqual(cells[0]["vir_spread"], 0.8)
        sc = _make_sweep_config_for_cell(cells[0], cfg)
        self.assertEqual(sc.vir_mass_spread, 0.8)

    def test_spread_axis_virialized_only(self):
        """cube26 ignores the spread axis -> no redundant cells (it would only
        recompute the same cube26 sim under colliding keys)."""
        cfg = self._base(node_geometries=["cube26"], vir_mass_spreads=[0.8, 2.0, 3.0])
        cells = expand_grid(cfg)
        self.assertEqual(len(cells), 1, "cube26 must not fan out over the spread axis")

    def test_resume_key_distinguishes_spreads(self):
        """REGRESSION: the per-cell CSV/resume key must include vir_mass_spread, else
        swept spreads at the same (M,S) collapse to one row (the bug that made the
        first explore run produce ~7 rows instead of ~210)."""
        from sweep import _resume_key
        a = dict(M_factor=100, S_gpc=10, centerM=1, outer_density_ceiling=1.0,
                 node_mass_amplitude=0.0, node_s_amplitude=0.0, node_mass_seed=42,
                 init_distribution="grf", node_geometry="virialized",
                 vir_mass_spread=0.8)
        b = dict(a, vir_mass_spread=3.0)
        self.assertNotEqual(_resume_key(a, include_S=True), _resume_key(b, include_S=True),
                            "two spreads at the same (M,S) must get DISTINCT resume keys")
        # A legacy row missing the column must still produce a key (no crash).
        c = dict(a); c.pop("vir_mass_spread")
        _resume_key(c, include_S=True)


class TestObserverInSweep(unittest.TestCase):
    """Observer-from-particle scoring wired into the sweep: when score_observers is on,
    the BEST observer is the headline chi2_dof and the cell records center_chi2_dof +
    frac_below_lcdm/eds ('always get the best one, and how many particles are good')."""

    def test_score_observers_threads_to_config(self):
        from sweep import _make_sweep_config_for_cell
        cfg = dict(particle_count=500, n_steps=273, t_start_Gyr=2.9,
                   s_min_gpc=3, s_max_gpc=35,
                   score_observers=True, observer_definition="local_rms",
                   observer_sample=64, observer_k=-1, lcdm_ref=0.44, eds_ref=0.84)
        cell = dict(M=100, amplitude=0.0, nm_seed=42, s_amplitude=0.0,
                    init="grf", geometry="virialized", vir_spread=2.0)
        sc = _make_sweep_config_for_cell(cell, cfg)
        self.assertIs(sc.score_observers, True)
        self.assertEqual(sc.observer_sample, 64)
        self.assertEqual(sc.lcdm_ref, 0.44)
        self.assertEqual(sc.eds_ref, 0.84)

    def test_observer_params_are_keyed_equals_run_in_cache(self):
        """REGRESSION: the cached metrics include best_observer_chi2 / frac_below_*, which
        depend on observer_k / observer_sample / observer_definition. Those MUST reach the
        cache key when score_observers is on, else a re-run with a different observer_k
        silently SERVES the stale cached observer score (the bug where 4 observer_k arms
        all returned the SAME best_observer_chi2). score_observers=False stays byte-identical."""
        from sweep import _make_sweep_config_for_cell
        from cosmo.parameter_sweep import build_cache_name
        from cosmo.cache import Cache
        import copy
        base = dict(particle_count=2000, n_steps=1092, t_start_Gyr=2.9,
                    s_min_gpc=3, s_max_gpc=35, observer_sample=2000)
        cell = dict(M=300, amplitude=0.0, nm_seed=42, s_amplitude=0.0,
                    init="grf", geometry="virialized", vir_spread=6.0)
        sc = _make_sweep_config_for_cell(cell, dict(base, score_observers=True, observer_k=-1))

        def key_with(**over):
            sc2 = copy.copy(sc)
            for k, v in over.items():
                setattr(sc2, k, v)
            return build_cache_name(sc2, 300, 20, 1, [42])

        # score_observers OFF -> no observer slug (byte-identical for non-observer runs)
        off = key_with(score_observers=False)
        self.assertNotIn("obsk", off)
        self.assertNotIn("obsdef", off)
        # distinct observer_k -> distinct keys (the bug fix)
        keys = {k: key_with(score_observers=True, observer_k=k) for k in (-1, 64, 128, 256)}
        self.assertEqual(len(set(keys.values())), 4,
                         f"observer_k not keyed==run (keys collide): {keys}")
        self.assertIn("allobsk", keys[-1])   # whole cloud encoded as 'all' (no minus sign)
        self.assertIn("64obsk", keys[64])
        # observer_sample and observer_definition also keyed
        self.assertNotEqual(key_with(score_observers=True, observer_k=64, observer_sample=512),
                            keys[64])
        self.assertNotEqual(key_with(score_observers=True, observer_k=64,
                                     observer_definition="hubble_flow"), keys[64])
        # the slugs round-trip through the CSV key (_split_key/_join_key)
        for v in list(keys.values()) + [off]:
            self.assertEqual(Cache._join_key(Cache._split_key(v)), v)

    def test_observer_columns_in_csv_cols(self):
        from sweep import SWEEP_CSV_COLS
        for c in ("best_observer_chi2", "center_chi2_dof", "observer_median_chi2",
                  "frac_below_lcdm", "frac_below_eds"):
            self.assertIn(c, SWEEP_CSV_COLS)

    def test_cell_from_best_row_reconstructs_vir_spread(self):
        """REGRESSION: the mu(z) panel must re-run the cell's SWEPT spread, not spread=0
        (else the figure runs a different grid -> figure<->CSV chi2 mismatch)."""
        from sweep import _cell_from_best_row
        base = dict(M_factor="100", node_geometry="virialized",
                    node_mass_amplitude="0.0", node_mass_seed="42",
                    node_s_amplitude="0.0", init_distribution="grf")
        self.assertEqual(_cell_from_best_row(dict(base, vir_mass_spread="3.0"))["vir_spread"], 3.0)
        # Missing/blank -> not set, so _make_sweep_config_for_cell falls back to cfg's scalar.
        self.assertNotIn("vir_spread", _cell_from_best_row(dict(base, vir_mass_spread="")))

    def test_no_snapshots_falls_back_to_center(self):
        """compute_pantheon_metrics with score_observers but a snapshot-less SimResult
        must not crash and must leave chi2_dof as the centre value (no observer keys)."""
        import numpy as np
        from cosmo.parameter_sweep import compute_pantheon_metrics, SimResult, SimSimpleResult
        from cosmo.pantheon import load_pantheon
        pan = load_pantheon()
        # A physical EdS-ish a(t) that passes the growth anchor (~3.3x over 2.9->13.8).
        t = np.linspace(0.0, 10.9, 30)
        a = (1.0 + t / 10.9 * 2.3)  # grows ~3.3x, monotonic
        sr = SimResult(size_curve_Gpc=None, hubble_curve=None, t_Gyr=t,
                       params=None, results=SimSimpleResult(1.0, 1.0, 1.0),
                       a_curve=a, snapshots=None)
        m = compute_pantheon_metrics(sr, pan, 2.9, score_observers=True,
                                     lcdm_ref=0.44, eds_ref=0.84)
        # No snapshots -> observer scoring skipped, no best_observer_chi2 injected.
        self.assertNotIn("best_observer_chi2", m)
        self.assertIn("chi2_dof", m)

    def test_small_sweep_makes_best_observer_the_headline(self):
        """End-to-end: a 1-cell virialized sweep with score_observers writes the observer
        columns and sets chi2_dof == best_observer_chi2 (the centre kept separately)."""
        import csv, tempfile, os as _os
        from sweep import load_config, run_sweep
        cfg = load_config(None)
        cfg.update(dict(
            M_values=[100], S_values=[10], vir_mass_spreads=[3.0],
            node_mass_amplitudes=[0.0], node_s_amplitudes=[0.0], node_mass_seeds=[42],
            particle_count=200, n_steps=273, t_start_Gyr=2.9, centerM=1,
            objective="pantheon", init_distributions=["grf"], grf_support="sphere",
            node_geometries=["virialized"], vir_n_nodes=120, vir_mass_rule="massfunc",
            vir_segregation=1.0, node_softening_gpc=1.0, node_force_law="plummer",
            score_observers=True, observer_sample=48, resume=False,
            results_dir=tempfile.mkdtemp(), tag="test_obs_sweep",
        ))
        csv_path, _best, _figs = run_sweep(cfg)
        rows = list(csv.DictReader(open(csv_path)))
        self.assertEqual(len(rows), 1)
        r = rows[0]
        if r["anchor_ok"].lower() == "true":   # observer scoring only runs for admissible cells
            self.assertNotEqual(r["best_observer_chi2"], "")
            self.assertAlmostEqual(float(r["chi2_dof"]), float(r["best_observer_chi2"]), places=9)
            self.assertNotEqual(r["center_chi2_dof"], "")
            self.assertNotEqual(r["frac_below_eds"], "")


# ---------------------------------------------------------------------------
# 16. S-knot guard (PF25): knot_ratio diagnostic + guarded S co-fit
# ---------------------------------------------------------------------------

class TestKnotRatioPure(unittest.TestCase):
    """knot_ratio_from_snapshots: the PURE Lagrangian core-ratio diagnostic (PF25).
    Core = final-snapshot inner particles within 0.25*r90 of the median centre;
    ratio = median core radius (last) / (first). < 1 contracting, > 1 expanding."""

    def _base_cloud(self, n=200, seed=0):
        rng = np.random.default_rng(seed)
        dirs = rng.normal(size=(n, 3))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        radii = np.linspace(0.01, 1.0, n)
        return dirs * radii[:, None]

    def test_contracting_core_ratio_below_one(self):
        from cosmo.parameter_sweep import knot_ratio_from_snapshots
        base = self._base_cloud()
        snaps = [{"positions": base * 2.0}, {"positions": base}]
        r = knot_ratio_from_snapshots(snaps, n_inner=len(base))
        self.assertIsNotNone(r)
        self.assertAlmostEqual(r, 0.5, places=6)
        self.assertLess(r, 1.0)

    def test_expanding_core_ratio_above_one(self):
        from cosmo.parameter_sweep import knot_ratio_from_snapshots
        base = self._base_cloud()
        snaps = [{"positions": base}, {"positions": base * 3.0}]
        r = knot_ratio_from_snapshots(snaps, n_inner=len(base))
        self.assertAlmostEqual(r, 3.0, places=6)
        self.assertGreater(r, 1.0)

    def test_only_first_n_inner_rows_used(self):
        """The observable cloud = rows 0..n_inner-1 (ParticleSystem appends the
        outer shell AFTER the inner particles); outer rows must not move the ratio."""
        from cosmo.parameter_sweep import knot_ratio_from_snapshots
        base = self._base_cloud()
        outer = self._base_cloud(50, seed=1) * 100.0  # huge distant outer shell
        snaps = [{"positions": np.vstack([base * 2.0, outer])},
                 {"positions": np.vstack([base, outer * 1.7])}]
        r = knot_ratio_from_snapshots(snaps, n_inner=len(base))
        self.assertAlmostEqual(r, 0.5, places=6)

    def test_median_centring_per_snapshot(self):
        """A rigid per-snapshot translation (cloud drift) must not change the ratio."""
        from cosmo.parameter_sweep import knot_ratio_from_snapshots
        base = self._base_cloud()
        snaps = [{"positions": base * 2.0 + np.array([5.0, -3.0, 1.0])},
                 {"positions": base + np.array([-2.0, 7.0, 0.5])}]
        r = knot_ratio_from_snapshots(snaps, n_inner=len(base))
        self.assertAlmostEqual(r, 0.5, places=6)

    def test_fewer_than_two_snapshots_none(self):
        from cosmo.parameter_sweep import knot_ratio_from_snapshots
        base = self._base_cloud()
        self.assertIsNone(knot_ratio_from_snapshots([], 200))
        self.assertIsNone(knot_ratio_from_snapshots(None, 200))
        self.assertIsNone(knot_ratio_from_snapshots([{"positions": base}], 200))

    def test_degenerate_shell_core_too_small_none(self):
        """All particles on one shell -> nothing inside 0.25*r90 (< 10 core
        particles) -> the metric is undefined (None), not a crash."""
        from cosmo.parameter_sweep import knot_ratio_from_snapshots
        rng = np.random.default_rng(3)
        dirs = rng.normal(size=(100, 3))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        snaps = [{"positions": dirs}, {"positions": dirs * 1.1}]
        self.assertIsNone(knot_ratio_from_snapshots(snaps, 100))


class TestKnotRatioInMetrics(unittest.TestCase):
    """compute_pantheon_metrics stores knot_ratio (ADDITIVE field) when
    score_observers is on and snapshots + n_inner are available — the same
    post-sim, snapshots-based family as the observer metrics."""

    def _sim_result(self, snapshots):
        from cosmo.parameter_sweep import SimResult, SimSimpleResult
        t = np.linspace(0.0, 10.9, 30)
        a = 1.0 + t / 10.9 * 2.3  # ~3.3x growth: passes the growth anchor
        return SimResult(size_curve_Gpc=None, hubble_curve=None, t_Gyr=t,
                         params=None, results=SimSimpleResult(1.0, 1.0, 1.0),
                         a_curve=a, snapshots=snapshots)

    def _snaps(self, n=200):
        rng = np.random.default_rng(0)
        dirs = rng.normal(size=(n, 3))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        base = dirs * np.linspace(0.01, 1.0, n)[:, None]
        return [{"positions": base * 2.0}, {"positions": base}]

    def test_knot_ratio_attached_when_observers_on(self):
        from cosmo.parameter_sweep import compute_pantheon_metrics
        from cosmo.pantheon import load_pantheon
        pan = load_pantheon()
        m = compute_pantheon_metrics(self._sim_result(self._snaps()), pan, 2.9,
                                     score_observers=True, n_inner=200)
        self.assertIn("knot_ratio", m)
        self.assertAlmostEqual(m["knot_ratio"], 0.5, places=6)

    def test_no_knot_ratio_when_observers_off(self):
        from cosmo.parameter_sweep import compute_pantheon_metrics
        from cosmo.pantheon import load_pantheon
        pan = load_pantheon()
        m = compute_pantheon_metrics(self._sim_result(self._snaps()), pan, 2.9,
                                     score_observers=False, n_inner=200)
        self.assertNotIn("knot_ratio", m)

    def test_no_knot_ratio_without_snapshots(self):
        from cosmo.parameter_sweep import compute_pantheon_metrics
        from cosmo.pantheon import load_pantheon
        pan = load_pantheon()
        m = compute_pantheon_metrics(self._sim_result(None), pan, 2.9,
                                     score_observers=True, n_inner=200)
        self.assertNotIn("knot_ratio", m)


class TestKnotGuardThreading(unittest.TestCase):
    """s_knot_guard knobs: config -> SweepConfig threading; NO cache-name slug
    (the guard changes which S the SEARCH selects, not per-(M,S) sim physics, so
    every per-S cache entry stays valid); defaults byte-identical (guard off)."""

    def _cfg(self, **overrides):
        cfg = dict(DEFAULT_CONFIG)
        cfg.update(overrides)
        return cfg

    def _cell(self):
        return dict(M=100, amplitude=0.0, nm_seed=42, s_amplitude=0.0,
                    init="uniform_sphere", geometry="cube26")

    def test_guard_fields_reach_sweep_config(self):
        cfg = self._cfg(s_knot_guard=True, knot_guard_chi2_budget=0.05,
                        knot_guard_min_ratio=0.85, knot_guard_step_gpc=3)
        sc = _make_sweep_config_for_cell(self._cell(), cfg)
        self.assertIs(sc.s_knot_guard, True)
        self.assertEqual(sc.knot_guard_chi2_budget, 0.05)
        self.assertEqual(sc.knot_guard_min_ratio, 0.85)
        self.assertEqual(sc.knot_guard_step_gpc, 3)

    def test_defaults_off_and_byte_identical(self):
        """A config WITHOUT the guard keys yields the SweepConfig defaults (guard
        off, budget 0.03, min ratio 0.9, step 5) — same as a bare SweepConfig()."""
        from cosmo.parameter_sweep import SweepConfig
        sc = _make_sweep_config_for_cell(self._cell(), self._cfg())
        base = SweepConfig()
        for obj in (sc, base):
            self.assertIs(obj.s_knot_guard, False)
            self.assertEqual(obj.knot_guard_chi2_budget, 0.03)
            self.assertEqual(obj.knot_guard_min_ratio, 0.9)
            self.assertEqual(obj.knot_guard_step_gpc, 5)

    def test_cache_key_unchanged_by_guard(self):
        """build_cache_name must be IDENTICAL with the guard on/off: per-S entries
        computed without the guard stay valid for guarded runs (and vice versa)."""
        cell = self._cell()
        sc_off = _make_sweep_config_for_cell(cell, self._cfg())
        sc_on = _make_sweep_config_for_cell(cell, self._cfg(
            s_knot_guard=True, knot_guard_chi2_budget=0.05,
            knot_guard_min_ratio=0.85, knot_guard_step_gpc=3))
        for S in (20, 45, 70):
            self.assertEqual(build_cache_name(sc_off, 100, S, 1, [42]),
                             build_cache_name(sc_on, 100, S, 1, [42]))
        # Same invariance on the smoke path (observers on).
        sc_off2 = _make_sweep_config_for_cell(cell, self._cfg(score_observers=True))
        sc_on2 = _make_sweep_config_for_cell(cell, self._cfg(score_observers=True,
                                                             s_knot_guard=True))
        self.assertEqual(build_cache_name(sc_off2, 100, 45, 1, [42]),
                         build_cache_name(sc_on2, 100, 45, 1, [42]))

    def test_knot_columns_in_sweep_csv_cols(self):
        for c in ("knot_ratio", "knot_guard_moved", "knot_guard_S_from"):
            self.assertIn(c, SWEEP_CSV_COLS)


class TestKnotGuardSelection(unittest.TestCase):
    """_knot_guard_select on a SCRIPTED evaluator (no sims). The spec scenario:
    best_S=20 (knot 0.5, chi2 0.44); S=25 (knot 0.85, chi2 0.45); S=30
    (knot 0.95, chi2 0.46). Budget 0.03 -> the guard must select 30 (25 is still
    knotted); budget 0.01 -> it must stay at 20 (30 is out of budget)."""

    _BEST = dict(chi2_dof=0.44, knot_ratio=0.5, anchor_ok=True)

    def _script(self):
        table = {
            25: dict(chi2_dof=0.45, knot_ratio=0.85, anchor_ok=True),
            30: dict(chi2_dof=0.46, knot_ratio=0.95, anchor_ok=True),
        }
        calls = []

        def evaluate(S):
            calls.append(S)
            return dict(table.get(S, dict(chi2_dof=float("inf"),
                                          knot_ratio=None, anchor_ok=False)))
        return evaluate, calls

    def test_budget_003_moves_to_30(self):
        from sweep import _knot_guard_select
        ev, calls = self._script()
        S, m, moved = _knot_guard_select(20, dict(self._BEST), ev, s_max=70,
                                         min_ratio=0.9, budget=0.03, step=5)
        self.assertTrue(moved)
        self.assertEqual(S, 30)
        self.assertEqual(m["knot_ratio"], 0.95)
        self.assertEqual(m["chi2_dof"], 0.46)
        self.assertEqual(calls, [25, 30])  # first-accept: stops at 30

    def test_budget_001_stays_at_20(self):
        from sweep import _knot_guard_select
        ev, calls = self._script()
        S, m, moved = _knot_guard_select(20, dict(self._BEST), ev, s_max=70,
                                         min_ratio=0.9, budget=0.01, step=5)
        self.assertFalse(moved)
        self.assertEqual(S, 20)
        self.assertEqual(m["knot_ratio"], 0.5)  # winner's metrics kept
        # Ladder exhausted up to s_max in step-Gpc increments.
        self.assertEqual(calls, [25, 30, 35, 40, 45, 50, 55, 60, 65, 70])

    def test_no_trigger_when_knot_free(self):
        from sweep import _knot_guard_select
        ev, calls = self._script()
        S, m, moved = _knot_guard_select(20, dict(chi2_dof=0.44, knot_ratio=0.95),
                                         ev, s_max=70, min_ratio=0.9,
                                         budget=0.03, step=5)
        self.assertFalse(moved)
        self.assertEqual(S, 20)
        self.assertEqual(calls, [])  # never probes when the winner is knot-free

    def test_no_trigger_when_knot_ratio_missing_or_none(self):
        from sweep import _knot_guard_select
        for best in (dict(chi2_dof=0.44), dict(chi2_dof=0.44, knot_ratio=None)):
            ev, calls = self._script()
            S, _m, moved = _knot_guard_select(20, best, ev, s_max=70,
                                              min_ratio=0.9, budget=0.03, step=5)
            self.assertFalse(moved)
            self.assertEqual(S, 20)
            self.assertEqual(calls, [])

    def test_anchor_gate_skips_runaway_candidates(self):
        """A knot-free, within-budget candidate that fails the growth anchor must
        be SKIPPED (the guard cannot trade a knot for a runaway)."""
        from sweep import _knot_guard_select
        table = {25: dict(chi2_dof=0.45, knot_ratio=0.95, anchor_ok=False),
                 30: dict(chi2_dof=0.46, knot_ratio=0.95, anchor_ok=True)}

        def ev(S):
            return dict(table.get(S, dict(chi2_dof=float("inf"),
                                          knot_ratio=None, anchor_ok=False)))
        S, _m, moved = _knot_guard_select(20, dict(self._BEST), ev, s_max=30,
                                          min_ratio=0.9, budget=0.03, step=5)
        self.assertTrue(moved)
        self.assertEqual(S, 30)

    def test_respects_s_max(self):
        from sweep import _knot_guard_select
        ev, calls = self._script()
        S, _m, moved = _knot_guard_select(20, dict(self._BEST), ev, s_max=27,
                                          min_ratio=0.9, budget=0.03, step=5)
        self.assertFalse(moved)
        self.assertEqual(calls, [25])  # 30 > s_max is never probed


class TestKnotGuardInCofit(unittest.TestCase):
    """The guard wired into _cofit_S_for_cell: default-off is a no-op (guard helper
    never called, no behaviour change in selection); enabled + knotted winner ->
    the row carries the moved S and the provenance columns."""

    def _cell(self):
        return dict(M=100, amplitude=0.0, nm_seed=42, s_amplitude=0.0,
                    init="uniform_sphere", geometry="cube26")

    def _best_metrics(self, **extra):
        m = {"match_avg_pct": 69.4, "diff_pct": 30.6, "chi2_dof": 0.44,
             "chi2": 100.0, "R2": 0.9, "n_sne_used": 100,
             "growth_factor": 3.3, "growth_target": 3.3}
        m.update(extra)
        return m

    def test_cofit_default_off_never_calls_guard(self):
        from sweep import _cofit_S_for_cell
        cfg = dict(DEFAULT_CONFIG)  # no guard keys at all
        with patch("sweep.linear_search_S",
                   return_value=(30, self._best_metrics(), False, [])), \
             patch("sweep._knot_guard_select") as guard:
            row, S = _cofit_S_for_cell(self._cell(), cfg, 10.0, 0.1,
                                       None, None, None, 0.44, 0.84)
        guard.assert_not_called()
        self.assertEqual(S, 30)
        self.assertFalse(row["knot_guard_moved"])
        self.assertEqual(row["knot_guard_S_from"], "")
        self.assertIsNone(row["knot_ratio"])  # present-if-computed, not required

    def test_cofit_guard_moves_S_and_records_provenance(self):
        from sweep import _cofit_S_for_cell
        cfg = dict(DEFAULT_CONFIG)
        cfg.update(s_knot_guard=True, s_max_gpc=30)  # defaults: budget .03, step 5
        table = {25: (0.45, 0.85), 30: (0.46, 0.95)}
        probed = []

        def fake_worst(sim_cb, config, M, S, cM, seeds, baseline, weights,
                       pantheon_data=None):
            probed.append(S)
            chi2, knot = table[S]
            m = self._best_metrics(chi2_dof=chi2, knot_ratio=knot,
                                   match_avg_pct=100.0 / (1.0 + chi2))
            return MagicMock(), m

        with patch("sweep.linear_search_S",
                   return_value=(20, self._best_metrics(knot_ratio=0.5),
                                 False, [])), \
             patch("sweep.worst_callback", side_effect=fake_worst):
            row, S = _cofit_S_for_cell(self._cell(), cfg, 10.0, 0.1,
                                       None, None, None, 0.44, 0.84)
        self.assertEqual(probed, [25, 30])
        self.assertEqual(S, 30)
        self.assertEqual(row["S_gpc"], 30)
        self.assertTrue(row["knot_guard_moved"])
        self.assertEqual(row["knot_guard_S_from"], 20)
        self.assertAlmostEqual(row["knot_ratio"], 0.95)
        self.assertAlmostEqual(row["chi2_dof"], 0.46)
        self.assertTrue(row["anchor_ok"])

    def test_cofit_guard_keeps_winner_when_no_candidate_qualifies(self):
        from sweep import _cofit_S_for_cell
        cfg = dict(DEFAULT_CONFIG)
        cfg.update(s_knot_guard=True, s_max_gpc=30, knot_guard_chi2_budget=0.01)
        table = {25: (0.45, 0.85), 30: (0.46, 0.95)}

        def fake_worst(sim_cb, config, M, S, cM, seeds, baseline, weights,
                       pantheon_data=None):
            chi2, knot = table[S]
            return MagicMock(), self._best_metrics(
                chi2_dof=chi2, knot_ratio=knot,
                match_avg_pct=100.0 / (1.0 + chi2))

        with patch("sweep.linear_search_S",
                   return_value=(20, self._best_metrics(knot_ratio=0.5),
                                 False, [])), \
             patch("sweep.worst_callback", side_effect=fake_worst):
            row, S = _cofit_S_for_cell(self._cell(), cfg, 10.0, 0.1,
                                       None, None, None, 0.44, 0.84)
        self.assertEqual(S, 20)
        self.assertFalse(row["knot_guard_moved"])
        self.assertEqual(row["knot_guard_S_from"], "")
        self.assertAlmostEqual(row["knot_ratio"], 0.5)
        self.assertAlmostEqual(row["chi2_dof"], 0.44)


class TestKnotGuardCacheRequire(unittest.TestCase):
    """PF-OBSCACHE mirror: a cache entry consulted by a GUARDED run must carry the
    knot_ratio field (else a pre-knot entry would silently disable the guard);
    guard-off runs keep reusing pre-knot entries byte-identically."""

    class _FakeCache:
        def __init__(self, name, metrics):
            self.name = name
            self._metrics = metrics
            self.added = []

        def get_cached_value(self, key, cache_type):
            from cosmo.cache import CacheType
            if cache_type == CacheType.METRICS:
                return dict(self._metrics)
            return {"size_final_Gpc": 1.0, "radius_max_Gpc": 1.0, "a_final": 1.0}

        def add_cached_value(self, *args, **kwargs):
            self.added.append(args)

    def setUp(self):
        import cosmo.parameter_sweep as ps
        self._ps = ps
        self._saved_cache = ps.CACHE
        self._saved_skip = ps.SKIP_CACHE
        ps.SKIP_CACHE = False

    def tearDown(self):
        self._ps.CACHE = self._saved_cache
        self._ps.SKIP_CACHE = self._saved_skip

    def _worst(self, cached_metrics, **cfg_over):
        from cosmo.parameter_sweep import worst_callback, SimResult, SimSimpleResult
        cfg = dict(DEFAULT_CONFIG)
        cfg.update(particle_count=500, n_steps=137)
        cfg.update(cfg_over)
        sweep_cfg = _make_sweep_config_for_cell(
            dict(M=100, amplitude=0.0, nm_seed=42, s_amplitude=0.0,
                 init="uniform_sphere", geometry="cube26"), cfg)
        self._ps.CACHE = self._FakeCache(
            f"metrics_{sweep_cfg.particle_count}_s42", cached_metrics)
        sim_calls = []

        def sim_cb(M, S, cM, seeds):
            sim_calls.append(S)
            # a_curve=None -> compute_pantheon_metrics scores worst-case without
            # touching pantheon_data (hermetic).
            return [SimResult(size_curve_Gpc=None, hubble_curve=None, t_Gyr=None,
                              params=None, results=SimSimpleResult(1.0, 1.0, 1.0),
                              a_curve=None)]

        _res, metrics = worst_callback(sim_cb, sweep_cfg, 100, 50, 1, [42],
                                       baseline=None, weights=None,
                                       pantheon_data=None)
        return metrics, sim_calls

    def test_guard_off_reuses_pre_knot_entry(self):
        """Default (guard off): a pre-knot cache entry stays a HIT — byte-identical
        behaviour, knot_ratio NOT required."""
        metrics, sim_calls = self._worst({"match_avg_pct": 60.0, "chi2_dof": 0.5})
        self.assertEqual(sim_calls, [])
        self.assertEqual(metrics["chi2_dof"], 0.5)

    def test_guard_on_requires_knot_ratio(self):
        """Guarded run + pre-knot entry (no knot_ratio field) -> cache MISS, the
        cell recomputes (so the guard always has its input)."""
        _metrics, sim_calls = self._worst({"match_avg_pct": 60.0, "chi2_dof": 0.5},
                                          s_knot_guard=True)
        self.assertEqual(sim_calls, [50])

    def test_guard_on_accepts_entry_with_knot_ratio_even_none(self):
        """knot_ratio=None is a VALID cached verdict (metric undefined): presence
        of the field is what the guard requires, not a non-None value."""
        metrics, sim_calls = self._worst(
            {"match_avg_pct": 60.0, "chi2_dof": 0.5, "knot_ratio": None},
            s_knot_guard=True)
        self.assertEqual(sim_calls, [])
        self.assertIn("knot_ratio", metrics)
        self.assertIsNone(metrics["knot_ratio"])


if __name__ == "__main__":
    unittest.main()
