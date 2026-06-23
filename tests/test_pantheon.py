"""
Unit tests for cosmo.pantheon loader module.

Tests use the synthetic fixture at tests/fixtures/pantheon_synthetic.dat.
The fixture has 15 rows:
  - 2 rows with zHD < 0.01  (SN001: z=0.005, SN002: z=0.008)
  - 2 rows with IS_CALIBRATOR == 1  (SN003: z=0.015, SN004: z=0.020)
  - 11 remaining Hubble-flow rows with z in [0.03, 1.5]
After default cuts (z_min=0.01, exclude_calibrators=True): 11 rows remain.
After z_min only (exclude_calibrators=False): 13 rows remain.
After no cuts (z_min=0, exclude_calibrators=False): 15 rows remain.
"""

import pathlib
import unittest

import numpy as np

from cosmo.pantheon import load_pantheon, bin_for_plot, DEFAULT_PATH

# Path to the committed synthetic fixture
_FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "pantheon_synthetic.dat"

# Counts derived from the fixture header comment above
_TOTAL_ROWS = 15
_SUB_ZMIN_ROWS = 2      # z < 0.01
_CALIBRATOR_ROWS = 2    # IS_CALIBRATOR == 1 AND z >= 0.01
_EXPECTED_DEFAULT = _TOTAL_ROWS - _SUB_ZMIN_ROWS - _CALIBRATOR_ROWS  # 11


class TestLoadPantheon(unittest.TestCase):
    """Core loader tests using the synthetic fixture."""

    def test_loads_and_returns_dict(self):
        """Loader returns a dict with z, mu, sigma, n keys."""
        result = load_pantheon(path=_FIXTURE)
        self.assertIsInstance(result, dict)
        for key in ("z", "mu", "sigma", "n"):
            self.assertIn(key, result, f"Missing key: {key!r}")

    def test_equal_length_arrays(self):
        """z, mu, sigma arrays have equal length and match n."""
        result = load_pantheon(path=_FIXTURE)
        n = result["n"]
        self.assertEqual(len(result["z"]), n)
        self.assertEqual(len(result["mu"]), n)
        self.assertEqual(len(result["sigma"]), n)

    def test_default_cuts_row_count(self):
        """After default cuts (z_min=0.01, exclude_calibrators=True): 11 rows."""
        result = load_pantheon(path=_FIXTURE)
        self.assertEqual(result["n"], _EXPECTED_DEFAULT)

    def test_z_min_cut_removes_sub_threshold_rows(self):
        """z_min cut removes the 2 rows with z < 0.01."""
        # Without calibrator exclusion: 15 - 2 sub-zmin = 13
        result = load_pantheon(path=_FIXTURE, z_min=0.01, exclude_calibrators=False)
        self.assertEqual(result["n"], _TOTAL_ROWS - _SUB_ZMIN_ROWS)
        # All returned z values must be >= 0.01
        self.assertTrue(np.all(result["z"] >= 0.01))

    def test_z_min_zero_returns_all_rows(self):
        """z_min=0 with exclude_calibrators=False returns all rows."""
        result = load_pantheon(path=_FIXTURE, z_min=0.0, exclude_calibrators=False)
        self.assertEqual(result["n"], _TOTAL_ROWS)

    def test_exclude_calibrators_removes_calibrator_rows(self):
        """Calibrator flag removes the 2 IS_CALIBRATOR==1 rows (that pass z_min)."""
        # With z_min=0 so no z cut, exclude calibrators: 15 - 2 = 13
        result_excl = load_pantheon(path=_FIXTURE, z_min=0.0, exclude_calibrators=True)
        result_incl = load_pantheon(path=_FIXTURE, z_min=0.0, exclude_calibrators=False)
        self.assertEqual(result_excl["n"], result_incl["n"] - _CALIBRATOR_ROWS)

    def test_z_sorted_ascending(self):
        """Returned z array is sorted ascending."""
        result = load_pantheon(path=_FIXTURE)
        z = result["z"]
        self.assertTrue(
            np.all(np.diff(z) >= 0),
            "z array is not sorted ascending",
        )

    def test_all_sigma_positive(self):
        """All returned sigma values must be strictly positive."""
        result = load_pantheon(path=_FIXTURE)
        self.assertTrue(
            np.all(result["sigma"] > 0),
            "Some sigma values are <= 0",
        )

    def test_arrays_are_ndarrays(self):
        """z, mu, sigma are np.ndarray instances."""
        result = load_pantheon(path=_FIXTURE)
        for key in ("z", "mu", "sigma"):
            self.assertIsInstance(result[key], np.ndarray, f"{key!r} is not ndarray")


class TestMissingFile(unittest.TestCase):
    """Error-handling tests for missing data file."""

    def test_missing_file_raises_file_not_found(self):
        """Non-existent path raises FileNotFoundError."""
        bad_path = "/nonexistent/path/Pantheon+SH0ES.dat"
        with self.assertRaises(FileNotFoundError):
            load_pantheon(path=bad_path)

    def test_missing_file_error_mentions_readme(self):
        """FileNotFoundError message references the README acquisition path."""
        bad_path = "/nonexistent/path/Pantheon+SH0ES.dat"
        try:
            load_pantheon(path=bad_path)
        except FileNotFoundError as exc:
            msg = str(exc).lower()
            self.assertIn("readme", msg, "Error message should mention README")
        else:
            self.fail("FileNotFoundError was not raised")

    def test_default_path_points_to_real_file_location(self):
        """DEFAULT_PATH ends with the expected filename (sanity check)."""
        self.assertEqual(DEFAULT_PATH.name, "Pantheon+SH0ES.dat")


class TestBinForPlot(unittest.TestCase):
    """Tests for the optional bin_for_plot helper."""

    def _load(self):
        return load_pantheon(path=_FIXTURE)

    def test_returns_dict_with_expected_keys(self):
        """bin_for_plot returns a dict with z, mu, err, n_sne."""
        d = self._load()
        binned = bin_for_plot(d["z"], d["mu"], d["sigma"])
        for key in ("z", "mu", "err", "n_sne"):
            self.assertIn(key, binned, f"Missing key: {key!r}")

    def test_at_most_n_bins(self):
        """Number of returned bins <= n_bins (empty bins are dropped)."""
        d = self._load()
        n_bins = 8
        binned = bin_for_plot(d["z"], d["mu"], d["sigma"], n_bins=n_bins)
        self.assertLessEqual(len(binned["z"]), n_bins)

    def test_bin_z_monotonic(self):
        """Bin-centre z values are monotonically non-decreasing."""
        d = self._load()
        binned = bin_for_plot(d["z"], d["mu"], d["sigma"], n_bins=5)
        if len(binned["z"]) > 1:
            self.assertTrue(
                np.all(np.diff(binned["z"]) >= 0),
                "Binned z values are not monotonic",
            )

    def test_bin_err_positive(self):
        """All bin uncertainties are strictly positive."""
        d = self._load()
        binned = bin_for_plot(d["z"], d["mu"], d["sigma"])
        self.assertTrue(np.all(binned["err"] > 0))

    def test_bin_arrays_equal_length(self):
        """All output arrays in binned dict have the same length."""
        d = self._load()
        binned = bin_for_plot(d["z"], d["mu"], d["sigma"])
        n = len(binned["z"])
        self.assertEqual(len(binned["mu"]), n)
        self.assertEqual(len(binned["err"]), n)
        self.assertEqual(len(binned["n_sne"]), n)

    def test_empty_input(self):
        """Empty input returns empty output without error."""
        empty = np.array([])
        binned = bin_for_plot(empty, empty, empty)
        self.assertEqual(len(binned["z"]), 0)


if __name__ == "__main__":
    unittest.main()
