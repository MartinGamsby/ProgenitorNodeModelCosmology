"""
Unit tests for pantheon_knob_sweep.py harness.

All tests are hermetic (no real N-body simulation, no filesystem writes unless
explicitly temp-dir scoped). Tests cover:
  - Config expansion / amplitude=0 collapse
  - CSV column contract (_BEST_ISO_COLS, _KNOB_SUMMARY_COLS)
  - Cache-key uniqueness across (amplitude, init_distribution)
  - init_distribution threading into SweepConfig and build_cache_name
  - load_best_config compatibility with the sweep-emitted CSV
"""
import csv
import io
import math
import pathlib
import tempfile
import unittest

import numpy as np


# ---------------------------------------------------------------------------
# Helpers shared across test classes
# ---------------------------------------------------------------------------

def _make_iso_row(M=855, S=37, centerM=1, chi2_dof=0.50, chi2=666.0, R2=0.996):
    """Build a minimal row dict matching _BEST_ISO_COLS."""
    return {
        "M_factor": M,
        "S_gpc": S,
        "centerM": centerM,
        "chi2_dof": chi2_dof,
        "chi2": chi2,
        "R2": R2,
        "n_sne_used": 1333,
        "growth_factor": 3.10,
        "anchor_ok": True,
        "node_mass_amplitude": 0.0,
        "node_mass_seed": 42,
        "init_distribution": "grf",
        "match_avg_pct": 100.0 / (1.0 + chi2_dof),
        "diff_pct": 100.0 - 100.0 / (1.0 + chi2_dof),
    }


def _make_knob_row(**kwargs):
    """Build a minimal knob-summary row; keyword args override defaults."""
    base = {
        "M_factor": 855,
        "S_gpc": 37,
        "centerM": 1,
        "node_mass_amplitude": 0.0,
        "node_mass_seed": 42,
        "init_distribution": "grf",
        "chi2_dof": 0.50,
        "chi2": 666.0,
        "R2": 0.996,
        "n_sne_used": 1333,
        "growth_factor": 3.10,
        "anchor_ok": True,
        "match_avg_pct": 66.7,
        "diff_pct": 33.3,
    }
    base.update(kwargs)
    return base


# ---------------------------------------------------------------------------
# Test 1: Config expansion / amplitude=0 collapse
# ---------------------------------------------------------------------------

class TestGridExpansion(unittest.TestCase):
    """_expand_grid must collapse amplitude=0 to a SINGLE nm_seed run."""

    def setUp(self):
        from pantheon_knob_sweep import _expand_grid
        self._expand = _expand_grid

    def test_amp0_collapsed_to_one_seed(self):
        combos = self._expand(
            m_list=[100, 500],
            s_list=[30, 40],
            amp_list=[0.0],
            seed_list=[42, 7],
            init="grf",
        )
        # amplitude=0 collapses to 1 seed => 2 M x 2 S x 1 = 4
        self.assertEqual(len(combos), 4)
        # All nm_seeds must be the single collapsed value (42)
        for _, _, amp, nm_seed in combos:
            self.assertEqual(amp, 0.0)
            self.assertEqual(nm_seed, 42)

    def test_amp_nonzero_expands_both_seeds(self):
        combos = self._expand(
            m_list=[100],
            s_list=[30],
            amp_list=[0.25, 0.5],
            seed_list=[42, 7],
            init="grf",
        )
        # 1 M x 1 S x 2 amp x 2 seeds = 4
        self.assertEqual(len(combos), 4)

    def test_mixed_amp_total_count(self):
        """amp=[0.0, 0.25, 0.5] with seeds=[42,7] -> 1 + 2 + 2 = 5 per (M,S)."""
        combos = self._expand(
            m_list=[100],
            s_list=[30],
            amp_list=[0.0, 0.25, 0.5],
            seed_list=[42, 7],
            init="grf",
        )
        self.assertEqual(len(combos), 5)

    def test_user_grid_total(self):
        """The user grid: 10 M x 13 S x (1 amp0 + 3 amp x 2 seeds) = 10 x 13 x 7 = 910."""
        from pantheon_knob_sweep import M_LIST, S_LIST, AMP_LIST, SEED_LIST, INIT_DISTRIBUTION
        combos = self._expand(M_LIST, S_LIST, AMP_LIST, SEED_LIST, INIT_DISTRIBUTION)
        # amp=[0.0,0.25,0.5,0.75] -> 1 + 3*2 = 7 per (M,S)
        expected = len(M_LIST) * len(S_LIST) * (1 + (len(AMP_LIST) - 1) * len(SEED_LIST))
        self.assertEqual(len(combos), expected)


# ---------------------------------------------------------------------------
# Test 2: CSV column contract
# ---------------------------------------------------------------------------

class TestCSVColumns(unittest.TestCase):
    """Verify _BEST_ISO_COLS and _KNOB_SUMMARY_COLS contain the required columns."""

    def test_best_iso_has_load_best_config_columns(self):
        """load_best_config requires M_factor, S_gpc, centerM, chi2_dof."""
        from pantheon_knob_sweep import _BEST_ISO_COLS
        required = {"M_factor", "S_gpc", "centerM", "chi2_dof"}
        missing = required - set(_BEST_ISO_COLS)
        self.assertEqual(missing, set(), f"Missing from _BEST_ISO_COLS: {missing}")

    def test_best_iso_has_extended_columns(self):
        from pantheon_knob_sweep import _BEST_ISO_COLS
        for col in ("chi2", "R2", "growth_factor", "anchor_ok",
                    "node_mass_amplitude", "init_distribution"):
            self.assertIn(col, _BEST_ISO_COLS, f"{col} missing from _BEST_ISO_COLS")

    def test_knob_summary_has_knob_columns(self):
        from pantheon_knob_sweep import _KNOB_SUMMARY_COLS
        for col in ("node_mass_amplitude", "node_mass_seed", "init_distribution",
                    "chi2_dof", "chi2", "R2", "n_sne_used", "growth_factor"):
            self.assertIn(col, _KNOB_SUMMARY_COLS, f"{col} missing from _KNOB_SUMMARY_COLS")

    def test_best_iso_csv_round_trip(self):
        """Write _BEST_ISO_COLS rows to a temp CSV and read them back cleanly."""
        from pantheon_knob_sweep import _BEST_ISO_COLS
        row = _make_iso_row()
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=_BEST_ISO_COLS, extrasaction="ignore")
        writer.writeheader()
        writer.writerow(row)
        buf.seek(0)
        reader = csv.DictReader(buf)
        rows = list(reader)
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(float(rows[0]["chi2_dof"]), 0.50, places=6)
        self.assertEqual(rows[0]["M_factor"], "855")

    def test_knob_summary_csv_round_trip(self):
        from pantheon_knob_sweep import _KNOB_SUMMARY_COLS
        row = _make_knob_row(node_mass_amplitude=0.25, node_mass_seed=7)
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=_KNOB_SUMMARY_COLS, extrasaction="ignore")
        writer.writeheader()
        writer.writerow(row)
        buf.seek(0)
        reader = csv.DictReader(buf)
        rows = list(reader)
        self.assertEqual(rows[0]["node_mass_amplitude"], "0.25")
        self.assertEqual(rows[0]["node_mass_seed"], "7")
        self.assertEqual(rows[0]["init_distribution"], "grf")


# ---------------------------------------------------------------------------
# Test 3: load_best_config compatibility
# ---------------------------------------------------------------------------

class TestLoadBestConfigCompatibility(unittest.TestCase):
    """
    The CSV emitted by the sweep must be directly consumable by
    hubble_diagram_nbody.load_best_config (finds lowest chi2_dof row).
    """

    def _write_iso_csv(self, rows, tmpdir):
        from pantheon_knob_sweep import _BEST_ISO_COLS
        path = os.path.join(tmpdir, "sweep_results_pantheon.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_BEST_ISO_COLS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_load_best_config_picks_lowest_chi2_dof(self):
        import os
        from hubble_diagram_nbody import load_best_config
        rows = [
            _make_iso_row(M=100, S=30, chi2_dof=0.55),
            _make_iso_row(M=500, S=45, chi2_dof=0.48),  # best
            _make_iso_row(M=900, S=60, chi2_dof=0.52),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._write_iso_csv(rows, tmpdir)
            cfg = load_best_config(path)
        self.assertAlmostEqual(cfg["M"], 500.0, places=3)
        self.assertAlmostEqual(cfg["S"], 45.0, places=3)
        self.assertAlmostEqual(cfg["chi2_dof"], 0.48, places=6)

    def test_load_best_config_has_required_keys(self):
        import os
        from hubble_diagram_nbody import load_best_config
        rows = [_make_iso_row()]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._write_iso_csv(rows, tmpdir)
            cfg = load_best_config(path)
        for key in ("M", "S", "centerM", "chi2_dof"):
            self.assertIn(key, cfg, f"load_best_config result missing key: {key}")


# Needed for the _write_iso_csv helper
import os


# ---------------------------------------------------------------------------
# Test 4: Cache-key uniqueness across knobs
# ---------------------------------------------------------------------------

class TestCacheKeyUniqueness(unittest.TestCase):
    """
    build_cache_name must produce distinct keys for distinct (amplitude,
    init_distribution, nm_seed) combinations; and must keep the
    uniform_sphere / amplitude=0 key UNCHANGED (backward compatible).
    """

    def _config(self, amplitude=0.0, nm_seed=42, init="uniform_sphere"):
        from cosmo.parameter_sweep import SweepConfig
        cfg = SweepConfig(
            quick_search=True,   # 200p / 250steps so particle_count is small
            t_start_Gyr=2.9,
            t_duration_Gyr=10.9,
            objective="pantheon",
            node_mass_amplitude=amplitude,
            node_mass_seed=nm_seed,
            init_distribution=init,
        )
        return cfg

    def _key(self, amplitude=0.0, nm_seed=42, init="uniform_sphere"):
        from cosmo.parameter_sweep import build_cache_name
        cfg = self._config(amplitude=amplitude, nm_seed=nm_seed, init=init)
        return build_cache_name(cfg, M_factor=500, S_val=35, centerM=1, seeds=[42])

    def test_amp0_uniform_key_unchanged(self):
        """amplitude=0, uniform_sphere key must NOT contain 'nmseed', 'nmamp', or 'init'."""
        key = self._key(amplitude=0.0, init="uniform_sphere")
        self.assertNotIn("nmseed", key)
        self.assertNotIn("nmamp", key)
        self.assertNotIn("init", key)

    def test_amp0_grf_key_contains_init_slug(self):
        """amplitude=0, grf key MUST include the init slug to distinguish from uniform_sphere."""
        key_grf = self._key(amplitude=0.0, init="grf")
        key_uni = self._key(amplitude=0.0, init="uniform_sphere")
        self.assertIn("grfinit", key_grf)
        self.assertNotEqual(key_grf, key_uni)

    def test_different_amplitudes_produce_different_keys(self):
        key_0   = self._key(amplitude=0.0)
        key_025 = self._key(amplitude=0.25)
        key_05  = self._key(amplitude=0.5)
        key_075 = self._key(amplitude=0.75)
        all_keys = {key_0, key_025, key_05, key_075}
        self.assertEqual(len(all_keys), 4, f"Colliding keys: {all_keys}")

    def test_different_seeds_produce_different_keys(self):
        key_42 = self._key(amplitude=0.5, nm_seed=42)
        key_7  = self._key(amplitude=0.5, nm_seed=7)
        self.assertNotEqual(key_42, key_7)

    def test_grf_and_uniform_keys_disjoint(self):
        """No grf key should equal any uniform_sphere key for the same (M,S,amp)."""
        from cosmo.parameter_sweep import build_cache_name
        keys_uni = set()
        keys_grf = set()
        for amp in [0.0, 0.25, 0.5]:
            for nm_seed in [42, 7]:
                cfg_u = self._config(amplitude=amp, nm_seed=nm_seed, init="uniform_sphere")
                cfg_g = self._config(amplitude=amp, nm_seed=nm_seed, init="grf")
                keys_uni.add(build_cache_name(cfg_u, 500, 35, 1, [42]))
                keys_grf.add(build_cache_name(cfg_g, 500, 35, 1, [42]))
        self.assertEqual(keys_uni & keys_grf, set(),
                         f"Colliding keys between uniform and grf: {keys_uni & keys_grf}")


# ---------------------------------------------------------------------------
# Test 5: init_distribution threading into SweepConfig
# ---------------------------------------------------------------------------

class TestSweepConfigInitDistribution(unittest.TestCase):
    """SweepConfig must carry init_distribution and default to 'uniform_sphere'."""

    def test_default_is_uniform_sphere(self):
        from cosmo.parameter_sweep import SweepConfig
        cfg = SweepConfig()
        self.assertEqual(cfg.init_distribution, "uniform_sphere")

    def test_grf_is_stored(self):
        from cosmo.parameter_sweep import SweepConfig
        cfg = SweepConfig(init_distribution="grf")
        self.assertEqual(cfg.init_distribution, "grf")

    def test_getattr_fallback_on_missing_attr(self):
        """build_cache_name uses getattr(..., 'uniform_sphere') for older configs."""
        from cosmo.parameter_sweep import build_cache_name, SweepConfig
        cfg = SweepConfig()
        # Deliberately delete the attribute to simulate an old pickled config
        del cfg.init_distribution
        # Should not raise; should fall back to uniform_sphere behavior (no init slug)
        key = build_cache_name(cfg, 500, 35, 1, [42])
        self.assertNotIn("init", key)


# ---------------------------------------------------------------------------
# Test 6: _make_sweep_config returns correct SweepConfig fields
# ---------------------------------------------------------------------------

class TestMakeSweepConfig(unittest.TestCase):
    """_make_sweep_config wires amplitude/seed/init into SweepConfig correctly."""

    def test_amplitude_wired(self):
        from pantheon_knob_sweep import _make_sweep_config
        cfg = _make_sweep_config(amplitude=0.75, seed=7, init="grf")
        self.assertAlmostEqual(cfg.node_mass_amplitude, 0.75)
        self.assertEqual(cfg.node_mass_seed, 7)
        self.assertEqual(cfg.init_distribution, "grf")
        self.assertEqual(cfg.objective, "pantheon")

    def test_particle_count(self):
        from pantheon_knob_sweep import _make_sweep_config, PARTICLES
        cfg = _make_sweep_config(0.0, 42, "grf")
        # _SweepConfigFixed overrides particle_count to PARTICLES (400)
        self.assertEqual(cfg.particle_count, PARTICLES)

    def test_n_steps(self):
        from pantheon_knob_sweep import _make_sweep_config, N_STEPS
        cfg = _make_sweep_config(0.0, 42, "grf")
        # _SweepConfigFixed overrides n_steps to N_STEPS (273)
        self.assertEqual(cfg.n_steps, N_STEPS)

    def test_timing_adds_to_today(self):
        from pantheon_knob_sweep import _make_sweep_config, T_START, T_DURATION
        cfg = _make_sweep_config(0.0, 42, "grf")
        self.assertAlmostEqual(cfg.t_start_Gyr + cfg.t_duration_Gyr, 13.8, places=5)


# ---------------------------------------------------------------------------
# Test 7: One-row knob summary CSV produced without crashes
# ---------------------------------------------------------------------------

class TestKnobSummaryOneSyntheticRow(unittest.TestCase):
    """
    Write a single synthetic row to the knob-summary CSV and verify the columns.
    No real sim is run.
    """

    def test_write_and_read_back(self):
        from pantheon_knob_sweep import _KNOB_SUMMARY_COLS
        rows = [
            _make_knob_row(M_factor=850, S_gpc=37, node_mass_amplitude=0.75,
                           node_mass_seed=42, chi2_dof=0.501),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "knob_sweep_summary.csv")
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=_KNOB_SUMMARY_COLS,
                                        extrasaction="ignore")
                writer.writeheader()
                writer.writerows(rows)
            with open(path, newline="", encoding="utf-8") as f:
                read_rows = list(csv.DictReader(f))

        self.assertEqual(len(read_rows), 1)
        self.assertEqual(set(read_rows[0].keys()), set(_KNOB_SUMMARY_COLS))
        self.assertAlmostEqual(float(read_rows[0]["chi2_dof"]), 0.501, places=6)
        self.assertEqual(read_rows[0]["node_mass_amplitude"], "0.75")


if __name__ == "__main__":
    unittest.main()
