"""
Tests for the PURE (no-sim, no-I/O) helpers in _generate_ws8_figs.py.

Philosophy (mirrors tests/test_plots.py)
----------------------------------------
- No full sims in tests: feed tiny synthetic arrays / call the deterministic
  build_virialized_grid generator directly (it does no I/O and is cheap).
- Tests run in well under a second.

Covered helpers
---------------
1. resolve_n_steps        — bumps n_steps so dt stays below the 0.05 Gyr ceiling.
2. displacement_magnitudes — per-particle |final-initial|; shape-mismatch raises.
3. slingshot_metrics       — tail metrics; heavy-tail vs uniform; empty/zero edge.
4. mass_radius_stats       — correlation/slope/NN/mean-preservation on real grids.
5. _pearson / _spearman    — basic correlation sanity.
"""

import math

import numpy as np
import pytest

from _generate_ws8_figs import (
    resolve_n_steps,
    displacement_magnitudes,
    slingshot_metrics,
    mass_radius_stats,
    _pearson,
    _spearman,
)
from cosmo.node_geometry import build_virialized_grid


# ---------------------------------------------------------------------------
# 1. resolve_n_steps
# ---------------------------------------------------------------------------

class TestResolveNSteps:
    def test_bumps_up_when_dt_too_big(self):
        # t_duration=8.0, requested 50 -> dt=0.16 (too big). Must bump up.
        n = resolve_n_steps(8.0, 50, min_dt_Gyr=0.05)
        assert 8.0 / n < 0.05
        assert n > 50

    def test_keeps_large_request(self):
        # A request that already yields dt<ceiling is left at least as large.
        n = resolve_n_steps(8.0, 1000, min_dt_Gyr=0.05)
        assert n >= 1000
        assert 8.0 / n < 0.05

    def test_dt_strictly_below_ceiling(self):
        for t_dur in (8.0, 10.9, 6.0):
            n = resolve_n_steps(t_dur, 1, min_dt_Gyr=0.05)
            assert t_dur / n < 0.05


# ---------------------------------------------------------------------------
# 2. displacement_magnitudes
# ---------------------------------------------------------------------------

class TestDisplacementMagnitudes:
    def test_simple_displacement(self):
        p0 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        p1 = np.array([[3.0, 4.0, 0.0], [1.0, 0.0, 0.0]])
        d = displacement_magnitudes(p0, p1)
        assert d.shape == (2,)
        assert abs(d[0] - 5.0) < 1e-12   # 3-4-5
        assert abs(d[1] - 0.0) < 1e-12

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            displacement_magnitudes(np.zeros((3, 3)), np.zeros((4, 3)))

    def test_nonnegative(self):
        rng = np.random.default_rng(0)
        p0 = rng.standard_normal((50, 3))
        p1 = rng.standard_normal((50, 3))
        d = displacement_magnitudes(p0, p1)
        assert np.all(d >= 0.0)


# ---------------------------------------------------------------------------
# 3. slingshot_metrics
# ---------------------------------------------------------------------------

class TestSlingshotMetrics:
    def test_uniform_has_small_ratio(self):
        # All equal -> max/median == 1, no tail.
        d = np.full(100, 2.0)
        m = slingshot_metrics(d, tail_factor=5.0)
        assert abs(m["max_over_median"] - 1.0) < 1e-12
        assert m["tail_fraction"] == 0.0
        assert m["n"] == 100

    def test_heavy_tail_detected(self):
        # 99 small + 1 huge -> big max/median and a nonzero tail fraction.
        d = np.concatenate([np.full(99, 1.0), np.array([100.0])])
        m = slingshot_metrics(d, tail_factor=5.0)
        assert m["max_over_median"] > 50.0
        assert m["tail_fraction"] > 0.0          # the one runaway counts
        assert m["max"] == 100.0

    def test_heavier_tail_has_larger_metric(self):
        light = np.concatenate([np.full(99, 1.0), np.array([3.0])])
        heavy = np.concatenate([np.full(99, 1.0), np.array([50.0])])
        m_light = slingshot_metrics(light)
        m_heavy = slingshot_metrics(heavy)
        assert m_heavy["max_over_median"] > m_light["max_over_median"]

    def test_empty_returns_nan(self):
        m = slingshot_metrics(np.array([]))
        assert m["n"] == 0
        assert math.isnan(m["median"])

    def test_all_zero_ratio_nan(self):
        m = slingshot_metrics(np.zeros(10))
        assert m["median"] == 0.0
        assert math.isnan(m["max_over_median"])


# ---------------------------------------------------------------------------
# 4. _pearson / _spearman
# ---------------------------------------------------------------------------

class TestCorrelation:
    def test_perfect_positive(self):
        x = np.arange(10, dtype=float)
        y = 2.0 * x + 1.0
        assert abs(_pearson(x, y) - 1.0) < 1e-9
        assert abs(_spearman(x, y) - 1.0) < 1e-9

    def test_perfect_negative(self):
        x = np.arange(10, dtype=float)
        y = -3.0 * x
        assert abs(_pearson(x, y) + 1.0) < 1e-9
        assert abs(_spearman(x, y) + 1.0) < 1e-9

    def test_constant_is_nan(self):
        x = np.ones(5)
        y = np.arange(5, dtype=float)
        assert math.isnan(_pearson(x, y))

    def test_spearman_monotonic_nonlinear(self):
        # Monotonic but non-linear -> Spearman == 1, Pearson < 1.
        x = np.arange(1, 11, dtype=float)
        y = x ** 3
        assert abs(_spearman(x, y) - 1.0) < 1e-9
        assert _pearson(x, y) < 1.0


# ---------------------------------------------------------------------------
# 5. mass_radius_stats (on real, deterministic virialized grids)
# ---------------------------------------------------------------------------

class TestMassRadiusStats:
    def test_radial_rule_mean_preserved_and_segregated(self):
        pos, mass = build_virialized_grid(
            20.0, n_nodes=26, M_ext_kg=1.0, vir_mass_rule="radial",
            vir_mass_spread=0.6, vir_segregation=1.0, seed=42,
        )
        s = mass_radius_stats(pos, mass, M_ext_kg=1.0)
        # Mean-preserving contract: mean(mass) == M_ext_kg.
        assert s["mean_preserved"] < 1e-9
        assert s["n"] == 26
        # Radial rule: mass increases monotonically with radius -> positive corr.
        assert s["pearson"] > 0.5
        assert s["seg_slope"] > 0.0
        # NN spacings finite and positive; median and mean generally differ.
        assert s["nn_median"] > 0.0
        assert s["nn_mean"] > 0.0

    def test_massfunc_rule_mean_preserved(self):
        pos, mass = build_virialized_grid(
            20.0, n_nodes=26, M_ext_kg=1.0, vir_mass_rule="massfunc",
            vir_mass_spread=0.6, vir_segregation=1.0, seed=42,
        )
        s = mass_radius_stats(pos, mass, M_ext_kg=1.0)
        assert s["mean_preserved"] < 1e-9
        # Segregation by mass rank -> positive mass-radius correlation.
        assert s["spearman"] > 0.0

    def test_spread_zero_uniform_mass(self):
        pos, mass = build_virialized_grid(
            20.0, n_nodes=26, M_ext_kg=1.0, vir_mass_rule="radial",
            vir_mass_spread=0.0, seed=42,
        )
        s = mass_radius_stats(pos, mass, M_ext_kg=1.0)
        # spread=0 -> all masses equal -> std ~ 0, correlation NaN (constant y).
        assert s["mass_std"] < 1e-9
        assert math.isnan(s["pearson"])
