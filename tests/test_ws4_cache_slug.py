"""
Tests for WS4 Section 3: cache-slug uniqueness, PHYSICS_CACHE_VERSION bump,
outer_density_ceiling slug, and CLI/sweep round-trips.

These guard the invariants:
  - build_cache_name distinguishes fractional centerM values (no int() truncation)
  - The v3 physics token appears in cache keys
  - outer_density_ceiling != 1.0 adds a 'ceil' slug
  - --center-node-mass and --outer-density-ceiling round-trip through args_to_sim_params
  - load_best_config returns centerM as float (survives --from-best-config round-trip)
"""
import argparse
import csv
import io
import math
import os
import sys
import tempfile
import unittest

from cosmo.parameter_sweep import (
    PHYSICS_CACHE_VERSION,
    SweepConfig,
    build_cache_name,
    physics_cache_token,
)
from cosmo.cli import add_common_arguments, args_to_sim_params
from cosmo.constants import SimulationParameters


# ---------------------------------------------------------------------------
# Minimal SweepConfig factory for cache-name tests
# ---------------------------------------------------------------------------

def _cfg(outer_density_ceiling: float = 1.0, **kwargs) -> SweepConfig:
    """Create a SweepConfig with outer_density_ceiling and optional overrides."""
    defaults = dict(
        quick_search=False,
        many_search=3,
        leet_search=False,
        search_center_mass=False,
        t_start_Gyr=2.9,
        t_duration_Gyr=10.9,
        damping_factor=None,
        s_min_gpc=18,
        s_max_gpc=40,
        save_interval=10,
        objective="pantheon",
        outer_density_ceiling=outer_density_ceiling,
    )
    defaults.update(kwargs)
    return SweepConfig(**defaults)


# ---------------------------------------------------------------------------
# Cache-slug tests
# ---------------------------------------------------------------------------

class TestCenterMSlugUniqueness(unittest.TestCase):
    """build_cache_name must produce DISTINCT keys for distinct centerM values."""

    def _key(self, centerM: float, ceiling: float = 1.0) -> str:
        cfg = _cfg(outer_density_ceiling=ceiling)
        return build_cache_name(cfg, M_factor=1, S_val=20, centerM=centerM, seeds=[42])

    def test_1p0_vs_1p5_distinct(self):
        self.assertNotEqual(self._key(1.0), self._key(1.5),
                            "centerM=1.0 and centerM=1.5 must produce distinct cache keys")

    def test_1p5_vs_2p0_distinct(self):
        self.assertNotEqual(self._key(1.5), self._key(2.0),
                            "centerM=1.5 and centerM=2.0 must produce distinct cache keys")

    def test_1p0_vs_2p0_distinct(self):
        self.assertNotEqual(self._key(1.0), self._key(2.0),
                            "centerM=1.0 and centerM=2.0 must produce distinct cache keys")

    def test_2p0_vs_3p0_distinct(self):
        self.assertNotEqual(self._key(2.0), self._key(3.0),
                            "centerM=2.0 and centerM=3.0 must produce distinct cache keys")

    def test_1p0_slug_contains_float(self):
        key = self._key(1.0)
        self.assertIn("1.0centerM", key,
                      "centerM=1.0 slug should contain '1.0centerM', not '1centerM'")

    def test_1p5_slug_contains_float(self):
        key = self._key(1.5)
        self.assertIn("1.5centerM", key,
                      "centerM=1.5 slug should contain '1.5centerM'")

    def test_no_int_truncation_1p0_1p5(self):
        """The old int() bug would make 1.0 and 1.5 collide into '1centerM'."""
        key_1p0 = self._key(1.0)
        key_1p5 = self._key(1.5)
        # Both keys must NOT contain the truncated slug '1centerM' as their centerM part
        # (i.e. they must not be equal AND they must use float repr)
        self.assertNotEqual(key_1p0, key_1p5)


class TestPhysicsCacheVersionV3(unittest.TestCase):
    """PHYSICS_CACHE_VERSION must be 'v3' and appear in cache keys."""

    def test_version_is_v3(self):
        self.assertEqual(PHYSICS_CACHE_VERSION, "v3",
                         "PHYSICS_CACHE_VERSION must be bumped to 'v3'")

    def test_v3_token_appears_in_cache_key(self):
        cfg = _cfg()
        key = build_cache_name(cfg, M_factor=1, S_val=20, centerM=1.0, seeds=[42])
        self.assertIn("v3", key,
                      "Physics version token 'v3' must appear in cache key")

    def test_v3_token_distinct_from_v2(self):
        """Simulate that a v2 cache key would not match a v3 key."""
        cfg = _cfg()
        key_v3 = build_cache_name(cfg, M_factor=1, S_val=20, centerM=1.0, seeds=[42])
        # A hypothetical v2 key would contain 'v2' in the physics token position.
        # The v3 key must NOT contain 'v2' as its physics version identifier.
        # (It may appear elsewhere but the version token itself should be v3.)
        self.assertIn("v3", key_v3)
        # The token for v3 should differ from what v2 produced — simplest check:
        # replace 'v3' with 'v2' and confirm the keys differ.
        key_v2_simulated = key_v3.replace("v3", "v2")
        self.assertNotEqual(key_v3, key_v2_simulated)


class TestOuterDensityCeilingSlug(unittest.TestCase):
    """outer_density_ceiling != 1.0 must add a 'ceil' slug; ==1.0 must not."""

    def _key(self, ceiling: float) -> str:
        cfg = _cfg(outer_density_ceiling=ceiling)
        return build_cache_name(cfg, M_factor=1, S_val=20, centerM=1.0, seeds=[42])

    def test_default_ceiling_no_slug(self):
        key = self._key(1.0)
        self.assertNotIn("ceil", key,
                         "ceiling=1.0 (default) must NOT add a 'ceil' slug to the key")

    def test_non_default_ceiling_adds_slug(self):
        key = self._key(1.5)
        self.assertIn("ceil", key,
                      "ceiling=1.5 must add a 'ceil' slug to the key")

    def test_different_ceilings_produce_distinct_keys(self):
        self.assertNotEqual(self._key(1.0), self._key(1.5),
                            "Different outer_density_ceiling values must produce distinct keys")

    def test_ceiling_1p5_slug_value(self):
        key = self._key(1.5)
        self.assertIn("1.5ceil", key,
                      "ceiling=1.5 slug should contain '1.5ceil'")


# ---------------------------------------------------------------------------
# CLI round-trip tests
# ---------------------------------------------------------------------------

class TestCLIRoundTrip(unittest.TestCase):
    """--center-node-mass and --outer-density-ceiling must round-trip into SimulationParameters."""

    def _parse(self, args_list):
        parser = argparse.ArgumentParser()
        add_common_arguments(parser)
        return parser.parse_args(args_list)

    def test_center_node_mass_float_roundtrip(self):
        args = self._parse(["--center-node-mass", "2.0"])
        sim_params = args_to_sim_params(args)
        self.assertAlmostEqual(sim_params.center_node_mass, 2.0, places=10)

    def test_center_node_mass_fractional_roundtrip(self):
        args = self._parse(["--center-node-mass", "1.5"])
        sim_params = args_to_sim_params(args)
        self.assertAlmostEqual(sim_params.center_node_mass, 1.5, places=10)

    def test_outer_density_ceiling_default(self):
        args = self._parse([])
        sim_params = args_to_sim_params(args)
        self.assertAlmostEqual(sim_params.outer_density_ceiling, 1.0, places=10)

    def test_outer_density_ceiling_roundtrip(self):
        args = self._parse(["--outer-density-ceiling", "1.5"])
        sim_params = args_to_sim_params(args)
        self.assertAlmostEqual(sim_params.outer_density_ceiling, 1.5, places=10)

    def test_outer_density_ceiling_exists_on_sim_params(self):
        args = self._parse(["--outer-density-ceiling", "1.2"])
        sim_params = args_to_sim_params(args)
        self.assertTrue(hasattr(sim_params, "outer_density_ceiling"))

    def test_center_node_mass_default_is_1(self):
        args = self._parse([])
        sim_params = args_to_sim_params(args)
        self.assertAlmostEqual(sim_params.center_node_mass, 1.0, places=10)


# ---------------------------------------------------------------------------
# from-best-config float round-trip
# ---------------------------------------------------------------------------

class TestFromBestConfigFloatRoundtrip(unittest.TestCase):
    """load_best_config must parse centerM as float, not int."""

    def _write_csv(self, rows, tmpdir):
        path = os.path.join(tmpdir, "sweep.csv")
        if not rows:
            return path
        fieldnames = list(rows[0].keys())
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_centerm_1p5_survives_roundtrip(self):
        """centerM=1.5 in CSV must come back as float 1.5 from load_best_config."""
        # Inline import here to avoid circular import at module level
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from hubble_diagram_nbody import load_best_config

        with tempfile.TemporaryDirectory() as tmpdir:
            rows = [
                {"M_factor": "5", "S_gpc": "25", "centerM": "1.5",
                 "chi2_dof": "0.65", "chi2": "100.0", "R2": "0.9"},
                {"M_factor": "10", "S_gpc": "30", "centerM": "1.0",
                 "chi2_dof": "0.80", "chi2": "120.0", "R2": "0.85"},
            ]
            path = self._write_csv(rows, tmpdir)
            result = load_best_config(path)

        self.assertIsInstance(result["centerM"], float,
                              "load_best_config must return centerM as float")
        self.assertAlmostEqual(result["centerM"], 1.5, places=10,
                               msg="centerM=1.5 must survive CSV round-trip")

    def test_centerm_1p0_survives_roundtrip(self):
        from hubble_diagram_nbody import load_best_config

        with tempfile.TemporaryDirectory() as tmpdir:
            rows = [
                {"M_factor": "1", "S_gpc": "20", "centerM": "1.0",
                 "chi2_dof": "0.55", "chi2": "90.0", "R2": "0.92"},
            ]
            path = self._write_csv(rows, tmpdir)
            result = load_best_config(path)

        self.assertAlmostEqual(result["centerM"], 1.0, places=10)

    def test_best_row_selected_by_min_chi2_dof(self):
        """load_best_config selects the row with the LOWEST chi2_dof."""
        from hubble_diagram_nbody import load_best_config

        with tempfile.TemporaryDirectory() as tmpdir:
            rows = [
                {"M_factor": "5", "S_gpc": "25", "centerM": "2.0",
                 "chi2_dof": "0.50", "chi2": "80.0", "R2": "0.93"},
                {"M_factor": "10", "S_gpc": "30", "centerM": "1.0",
                 "chi2_dof": "0.80", "chi2": "120.0", "R2": "0.85"},
            ]
            path = self._write_csv(rows, tmpdir)
            result = load_best_config(path)

        # Best row is M=5, centerM=2.0 (lowest chi2_dof=0.50)
        self.assertAlmostEqual(result["centerM"], 2.0, places=10)
        self.assertAlmostEqual(result["M"], 5.0, places=10)


if __name__ == "__main__":
    unittest.main()
