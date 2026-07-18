"""
Smoke tests for the top-level hubble_diagram.py script.

These tests exercise the script's run() callable end-to-end using the
committed synthetic fixture (tests/fixtures/pantheon_synthetic.dat) so
no real Pantheon+SH0ES data file is required.

Matplotlib is forced to the non-interactive Agg backend via the script's
own module-level call, so these tests are safe in headless / CI environments.

Tests
-----
1. run() produces a PNG file in a tmp dir.
2. run() returns a results dict with exactly 3 model keys.
3. run() exits with FileNotFoundError when the data file is absent.
4. CLI --help exits 0 (parser is functional).
5. Default M/S give Omega_Lambda_eff ~ 0.70 (not the raw SimulationParameters
   default of ~2.55 which would crash on the full z range).
"""

import importlib
import pathlib
import subprocess
import sys
import types
import unittest

import numpy as np


# ---------------------------------------------------------------------------
# Shared fixture path
# ---------------------------------------------------------------------------

_FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "pantheon_synthetic.dat"
_SCRIPT = pathlib.Path(__file__).parent.parent / "hubble_diagram.py"


def _make_sim_params(M: float = 855.0, S: float = 37.8, Omega_Lambda_eff: float | None = None):
    """
    Build a minimal sim_params namespace compatible with the script's run().

    Uses the real SimulationParameters so Omega_Lambda_eff is derived from M/S,
    unless overridden explicitly for tests that need a specific value.
    """
    from cosmo.constants import SimulationParameters
    sim = SimulationParameters(M_value=M, S_value=S)
    if Omega_Lambda_eff is not None:
        sim.external_params.Omega_Lambda_eff = Omega_Lambda_eff
    return sim


# ---------------------------------------------------------------------------
# Test 1 + 2: run() produces a PNG and returns 3-model results dict
# ---------------------------------------------------------------------------

class TestRunProducesPNG(unittest.TestCase):
    """run() must save a PNG and return a 3-model results dict."""

    def setUp(self):
        self.sim_params = _make_sim_params()

    def test_png_is_created(self):
        """A PNG file must exist in the tmp output dir after run()."""
        import tempfile
        # Import via importlib so we pick up the script's module-level
        # matplotlib.use("Agg") call.
        import hubble_diagram as hd_script

        with tempfile.TemporaryDirectory() as tmp:
            hd_script.run(
                sim_params=self.sim_params,
                output_dir=tmp,
                pantheon_path=str(_FIXTURE),
                z_min=0.01,
                n_bins=5,
            )
            pngs = list(pathlib.Path(tmp).glob("hubble_diagram*.png"))
            self.assertEqual(
                len(pngs), 1,
                f"Expected 1 PNG, found {len(pngs)}: {pngs}",
            )

    def test_results_has_three_models(self):
        """run() must return a dict with exactly 3 cosmological model keys."""
        import tempfile
        import hubble_diagram as hd_script

        with tempfile.TemporaryDirectory() as tmp:
            results = hd_script.run(
                sim_params=self.sim_params,
                output_dir=tmp,
                pantheon_path=str(_FIXTURE),
                z_min=0.01,
                n_bins=5,
            )
        self.assertSetEqual(
            set(results.keys()),
            {"lcdm", "external_node", "matter_only"},
        )

    def test_each_result_has_chi2_and_r2(self):
        """Each model result must contain chi2, chi2_dof, and R2 keys."""
        import tempfile
        import hubble_diagram as hd_script

        with tempfile.TemporaryDirectory() as tmp:
            results = hd_script.run(
                sim_params=self.sim_params,
                output_dir=tmp,
                pantheon_path=str(_FIXTURE),
                z_min=0.01,
                n_bins=5,
            )
        for model_name, r in results.items():
            for key in ("chi2", "chi2_dof", "R2", "DeltaM", "dof"):
                self.assertIn(
                    key, r,
                    f"Model {model_name!r} missing key {key!r}",
                )


# ---------------------------------------------------------------------------
# Test 3: missing data file raises FileNotFoundError
# ---------------------------------------------------------------------------

class TestMissingDataFile(unittest.TestCase):
    """run() must raise FileNotFoundError when the data file is absent."""

    def test_missing_file_raises(self):
        import tempfile
        import hubble_diagram as hd_script

        sim_params = _make_sim_params()
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                hd_script.run(
                    sim_params=sim_params,
                    output_dir=tmp,
                    pantheon_path="/nonexistent/Pantheon+SH0ES.dat",
                    z_min=0.01,
                )


# ---------------------------------------------------------------------------
# Test 4: CLI --help exits 0
# ---------------------------------------------------------------------------

class TestCLIHelp(unittest.TestCase):
    """python hubble_diagram.py --help must exit with code 0."""

    def test_help_exits_zero(self):
        result = subprocess.run(
            [sys.executable, str(_SCRIPT), "--help"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            result.returncode, 0,
            f"--help exited {result.returncode}. stderr: {result.stderr[:200]}",
        )
        self.assertIn("Hubble", result.stdout)


# ---------------------------------------------------------------------------
# Test 5: default M/S give Omega_Lambda_eff ~ 0.70
# ---------------------------------------------------------------------------

class TestDefaultOmegaLambda(unittest.TestCase):
    """
    The default CLI parameters (M=855, S=37.8) must yield an Omega_Lambda_eff
    that keeps the external-node model well within the flat/mildly-open regime
    (0.5 < Omega_Lambda_eff < 1.0), NOT the raw SimulationParameters default
    (M=800, S=24 -> ~2.55).
    """

    def test_default_params_give_sensible_omega(self):
        # M=855, S=37.8 is the script default; must yield Omega_Lambda_eff ~ 0.70.
        sim_params = _make_sim_params(M=855.0, S=37.8)
        Omega = sim_params.external_params.Omega_Lambda_eff
        self.assertGreater(Omega, 0.5, f"Omega_Lambda_eff={Omega:.4f} too low")
        self.assertLess(Omega, 1.0, f"Omega_Lambda_eff={Omega:.4f} too high (closed?)")
        # Tighter check: within 5% of 0.70
        self.assertAlmostEqual(
            Omega, 0.70, delta=0.05,
            msg=f"Omega_Lambda_eff={Omega:.4f} not near 0.70"
        )


if __name__ == "__main__":
    unittest.main()
