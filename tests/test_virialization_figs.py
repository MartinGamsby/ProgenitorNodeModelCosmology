"""
Tests for the center-only virialization study script (_generate_virialization_figs).

Covers the PURE measurement helpers (no figure rendering, no I/O):
  - measure_row returns the center-only + legacy + objective fields and a sane
    pass flag;
  - build_residual_table covers {A_lattice, realistic, B_relax_*} x rule x size;
  - optionB_descent's force-residual objective f is MONOTONE non-increasing in
    relaxation step (the true-descent guarantee) and the realistic start (step 0)
    has a far-larger center residual than Option A;
  - _best_B_center picks the minimum over swept steps.

The figure-writing functions are NOT exercised here (matplotlib Agg render is slow
and not logic-bearing); the script's __main__ path renders them.
"""
import importlib

import numpy as np
import pytest

vfig = importlib.import_module("_generate_virialization_figs")
from cosmo.node_geometry import (  # noqa: E402
    build_virialized_grid,
    _gradient_relax_positions,
    _force_residual_objective_and_grad,
)
from cosmo.constants import CosmologicalConstants  # noqa: E402


# Small sizes keep the O(N^2) relaxation fast.
SMALL_SIZES = (26, 60)


class TestMeasureRow:

    def test_keys_and_types(self):
        pos, m = vfig._build("radial", 60)
        row = vfig.measure_row("A_lattice", "radial", 60, pos, m)
        for k in ("builder", "rule", "n", "max_residual_center",
                  "median_residual_center", "n_center", "a_ref",
                  "max_residual_inner_frac", "median_residual_inner_frac",
                  "f_objective", "passes_center_tol"):
            assert k in row
        assert isinstance(row["passes_center_tol"], bool)
        assert row["n"] == 60
        assert row["n_center"] >= 2

    def test_lattice_passes_center_tol(self):
        """Option A (lattice) center residual is ~machine-precision => passes TOL."""
        pos, m = vfig._build("radial", 100)
        row = vfig.measure_row("A_lattice", "radial", 100, pos, m)
        assert row["passes_center_tol"] is True
        assert row["max_residual_center"] < vfig._TOL

    def test_realistic_fails_center_tol(self):
        """The un-relaxed realistic blob is far from balanced => fails TOL."""
        pos, m = vfig._build("radial", 100, steps=0)
        row = vfig.measure_row("realistic", "radial", 100, pos, m)
        assert row["passes_center_tol"] is False
        assert row["max_residual_center"] > 1.0


class TestResidualTable:

    def test_covers_all_builders_rules_sizes(self):
        rows = vfig.build_residual_table(SMALL_SIZES)
        builders = {r["builder"] for r in rows}
        assert "A_lattice" in builders
        assert "realistic" in builders
        assert any(b.startswith("B_relax_") for b in builders)
        assert {r["rule"] for r in rows} == set(vfig._RULES)
        assert {r["n"] for r in rows} == set(SMALL_SIZES)

    def test_lattice_far_below_realistic_both_rules(self):
        rows = vfig.build_residual_table((100,))
        for rule in vfig._RULES:
            a = next(r for r in rows
                     if r["builder"] == "A_lattice" and r["rule"] == rule)
            real = next(r for r in rows
                        if r["builder"] == "realistic" and r["rule"] == rule)
            assert a["max_residual_center"] < real["max_residual_center"]
            assert a["max_residual_center"] <= vfig._TOL
            assert real["max_residual_center"] > 1.0


class TestOptionBDescent:

    def test_objective_monotone_within_a_descent(self):
        """Within a SINGLE relaxation run the force-residual objective f decreases
        monotonically (the backtracking line-search guarantee). Measured on the raw
        relaxed positions BEFORE the per-build NN-spacing rescale — the rescale is a
        global factor that changes f's absolute scale, so cross-build comparison of f
        is not meaningful, but the descent itself is strictly downhill.
        """
        pos0, m0 = vfig._build("radial", 80, steps=0)  # realistic start
        cm = float(np.mean(m0))
        G = CosmologicalConstants.G
        f_prev = None
        for steps in (0, 3, 8, 15, 30):
            relaxed = _gradient_relax_positions(
                pos0, m0, n_steps=steps, rate=vfig._RELAX_RATE,
                hold_outer_frac=vfig._RELAX_HOLD)
            f, _ = _force_residual_objective_and_grad(relaxed, m0, cm, G)
            if f_prev is not None:
                assert f <= f_prev + 1e-9 * abs(f_prev), (
                    f"objective rose at steps={steps}: {f} > {f_prev}")
            f_prev = f

    def test_relaxation_reduces_center_residual_from_start(self):
        """A moderate Option-B step count lowers the center residual below the
        realistic (step-0) start — true relaxation makes genuine progress."""
        curve = vfig.optionB_descent("radial", 80, (10,))
        start = curve["center_residual"][0]   # step 0 = realistic
        relaxed = curve["center_residual"][1]  # step 10
        assert relaxed < start

    def test_both_rules_measured(self):
        for rule in vfig._RULES:
            curve = vfig.optionB_descent(rule, 60, (5,))
            assert np.all(np.isfinite(curve["f_objective"]))
            assert np.all(np.isfinite(curve["center_residual"]))


class TestBestB:

    def test_best_b_is_minimum_over_steps(self):
        rows = vfig.build_residual_table((100,))
        best = vfig._best_B_center(rows, "radial", 100)
        bvals = [r["max_residual_center"] for r in rows
                 if r["builder"].startswith("B_relax_") and r["rule"] == "radial"
                 and r["n"] == 100]
        assert best == min(bvals)
