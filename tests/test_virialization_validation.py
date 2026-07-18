"""
Virialization VALIDATION tests + force-balance metric (W1).

The physical criterion (the user's, verbatim intent): a TRUE virialized grid's
INNER nodes feel ~NET-ZERO gravitational force from all the OTHER nodes PLUS a
central node of mass centerM (==one unit node here). "The inner nodes of a big
enough virialized grid should not move. If they do, then it's not virialized."

This file DEFINES and TESTS that criterion via two PURE helpers in
cosmo.node_geometry:
  - node_net_accelerations(positions, masses, *, center_mass_kg, G)
      mirrors the sim's tidal law (incl. the 1e10 m singularity floor),
  - virialization_residual(positions, masses, *, inner_frac, center_mass_kg, G,
      reference) -> dict with a DIMENSIONLESS per-inner-node residual
      |net_accel| / a_ref (well-virialized => residual << 1).

WHICH RULE IS ACTUALLY VIRIALIZED (measured, not guessed)
---------------------------------------------------------
The DEFAULT virialized generator (vir_relax_steps>=1) is the FORCE-BALANCED mode:
an exact cubic-lattice ball with a node at the origin and masses assigned by radius
shell. Its big-grid inner residual is at MACHINE PRECISION (~1e-30) for BOTH rules,
well below VIRIALIZATION_TOL — so the default grid satisfies the user's criterion
(inner nodes feel ~zero net force) for radial AND massfunc.

The REALISTIC mode (vir_relax_steps=0, the old Fibonacci layout) is NOT force-
balanced: its big-grid inner residual is O(20-30) for both rules (massfunc the
lesser offender). A continuous position relaxation cannot reach balance on a finite
canvas, so Section 2 implemented the analytic lattice instead (Option A). The
balanced-vs-unbalanced contrast is asserted below.
"""
import numpy as np
import pytest

from cosmo.node_geometry import (
    build_virialized_grid,
    build_node_positions,
    node_net_accelerations,
    virialization_residual,
    _force_residual_objective_and_grad,
)
from cosmo.constants import CosmologicalConstants

const = CosmologicalConstants()
G = const.G
M_EXT_KG = 5e55
S_DEFAULT_M = 30.0 * const.Gpc_to_m

# THE tunable knob: an inner node counts as "virialized" when its net force is
# below this fraction of a characteristic single-neighbour pull. 0.25 = the inner
# residual must be under 25% of the reference pull (directional pulls cancel).
VIRIALIZATION_TOL = 0.25

# The "big enough" grid the criterion is asserted on (multi-layer, segregated).
BIG_N = 100
BIG_EXTENT = 2.5
BIG_SPREAD = 0.8
BIG_SEG = 1.0
BIG_SEED = 12

def _radii(positions):
    return np.linalg.norm(positions, axis=1)


def _big_grid(rule, n=BIG_N, relax_steps=1):
    """The big grid. relax_steps>=1 (default) = force-balanced; 0 = realistic."""
    return build_virialized_grid(
        S_DEFAULT_M, n_nodes=n, M_ext_kg=M_EXT_KG, vir_mass_rule=rule,
        vir_mass_spread=BIG_SPREAD, vir_segregation=BIG_SEG,
        vir_extent=BIG_EXTENT, vir_relax_steps=relax_steps, seed=BIG_SEED,
    )


# ---------------------------------------------------------------------------
# 1. node_net_accelerations — shape/dtype + hand checks
# ---------------------------------------------------------------------------

class TestNodeNetAccelerations:

    def test_shape_dtype(self):
        pos, masses = _big_grid("radial", n=40)
        accel = node_net_accelerations(pos, masses, center_mass_kg=M_EXT_KG, G=G)
        assert accel.shape == (40, 3)
        assert accel.dtype == np.float64

    def test_symmetric_ring_plus_centre_is_null(self):
        """A single ring of EQUAL masses + an equal-mass centre at the origin:
        the origin node's net accel is ~0 by symmetry (opposing pulls cancel)."""
        nring = 12
        ang = np.linspace(0.0, 2.0 * np.pi, nring, endpoint=False)
        R = 1.0e24
        ring = np.stack([R * np.cos(ang), R * np.sin(ang), np.zeros(nring)], axis=1)
        pos = np.vstack([np.zeros((1, 3)), ring])  # node 0 = origin (symmetric centre)
        m = np.ones(pos.shape[0])
        accel = node_net_accelerations(pos, m, center_mass_kg=1.0, G=G)
        # Reference single-pull on the origin node from one ring member.
        a_one = G * 1.0 / R**2
        assert np.linalg.norm(accel[0]) < 1e-6 * a_one

    def test_two_node_points_toward_other(self):
        """A 2-node asymmetric case: net accel on node0 is finite and points
        toward node1 (attractive, sign-correct)."""
        pos = np.array([[0.0, 0.0, 0.0], [1.0e24, 0.0, 0.0]])
        m = np.array([1.0, 5.0])
        # No central node so the 2-body result is clean.
        accel = node_net_accelerations(pos, m, center_mass_kg=0.0, G=G)
        assert accel[0, 0] > 0.0                 # toward +x (node1)
        assert abs(accel[0, 1]) < 1e-30
        assert abs(accel[0, 2]) < 1e-30
        # Newton's third law: |a0|*m0 == |a1|*m1.
        np.testing.assert_allclose(
            abs(accel[0, 0]) * m[0], abs(accel[1, 0]) * m[1], rtol=1e-12)

    def test_singularity_floor_no_inf(self):
        """Two coincident nodes do not blow up (1e10 m floor mirrors the sim)."""
        pos = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        m = np.array([1.0, 1.0])
        accel = node_net_accelerations(pos, m, center_mass_kg=0.0, G=G)
        assert np.all(np.isfinite(accel))


# ---------------------------------------------------------------------------
# 2. virialization_residual — reads ~null on a symmetric ground truth
# ---------------------------------------------------------------------------

class TestResidualSymmetricGroundTruth:

    def test_cube26_plus_centre_node_residual_near_zero(self):
        """cube26 lattice + a node AT the origin (its symmetric centre) +
        uniform masses + equal central source -> the centre node's residual is
        ~0. This is the metric's ground truth: it reads near-null when it should.
        (Plain cube26 has NO node at its centre; the origin node is the one
        symmetric inner node, so we add it explicitly.)"""
        cube = build_node_positions("cube26", 1.0e24)
        pos = np.vstack([np.zeros((1, 3)), cube])  # node 0 = symmetric centre
        m = np.ones(pos.shape[0])
        res = virialization_residual(pos, m, inner_frac=0.5, center_mass_kg=1.0)
        assert res["n_inner"] >= 1
        assert 0 in res["inner_idx"]             # the origin node is "inner"
        assert res["max_residual"] < 1e-6        # essentially null
        assert res["max_residual"] <= VIRIALIZATION_TOL

    def test_returns_expected_keys(self):
        pos, masses = _big_grid("radial", n=40)
        res = virialization_residual(pos, masses, inner_frac=0.5)
        for key in ("residual_per_node", "inner_idx", "max_residual",
                    "median_residual", "a_ref", "n_inner"):
            assert key in res
        assert res["residual_per_node"].shape == (res["n_inner"],)
        assert res["a_ref"] > 0.0


# ---------------------------------------------------------------------------
# 3. THE criterion, BOTH rules (xfail on the current un-balanced generator)
# ---------------------------------------------------------------------------

class TestVirializationCriterion:
    """Build the BIG grid, compute the residual, assert inner nodes are balanced.

    The DEFAULT generator (vir_relax_steps>=1) is force-balanced, so this now PASSES
    for BOTH rules (it was xfail on the old un-balanced generator). The contrast with
    the realistic (un-balanced) mode is asserted in test_unbalanced_mode_is_large.
    """

    @pytest.mark.parametrize("rule", ["radial", "massfunc"])
    def test_big_grid_inner_residual_below_tol(self, rule):
        pos, masses = _big_grid(rule)
        res = virialization_residual(
            pos, masses, inner_frac=0.5, center_mass_kg=M_EXT_KG)
        # Printed so the empirical comparison is captured in -s output.
        print(f"\n[virialization] rule={rule} n={BIG_N} "
              f"n_inner={res['n_inner']} max_residual={res['max_residual']:.3e} "
              f"median={res['median_residual']:.3e} (TOL={VIRIALIZATION_TOL})")
        assert res["max_residual"] <= VIRIALIZATION_TOL, (
            f"{rule}: inner nodes are NOT force-balanced "
            f"(max_residual={res['max_residual']:.3e} > {VIRIALIZATION_TOL})"
        )

    @pytest.mark.parametrize("rule", ["radial", "massfunc"])
    def test_unbalanced_mode_is_large(self, rule):
        """The REALISTIC mode (vir_relax_steps=0) is far from virialized for BOTH
        rules (residual O(20-30) >> TOL), so the force-balanced default is a genuine
        improvement, not a no-op."""
        pos, masses = _big_grid(rule, relax_steps=0)
        res = virialization_residual(
            pos, masses, inner_frac=0.5, center_mass_kg=M_EXT_KG)
        assert res["max_residual"] > 1.0, (
            f"{rule}: unbalanced residual unexpectedly small "
            f"({res['max_residual']:.3f})")

    @pytest.mark.parametrize("rule", ["radial", "massfunc"])
    def test_balanced_far_below_unbalanced(self, rule):
        """balanced (steps>=1) max_residual <= TOL << unbalanced (steps=0)."""
        rb = virialization_residual(
            *_big_grid(rule, relax_steps=1), inner_frac=0.5, center_mass_kg=M_EXT_KG)
        ru = virialization_residual(
            *_big_grid(rule, relax_steps=0), inner_frac=0.5, center_mass_kg=M_EXT_KG)
        assert rb["max_residual"] <= VIRIALIZATION_TOL
        assert rb["max_residual"] < ru["max_residual"]


# ---------------------------------------------------------------------------
# 4. Bigger grid -> smaller residual (a property of a VIRIALIZED structure)
# ---------------------------------------------------------------------------

class TestBigEnough:

    def test_bigger_grid_smaller_residual(self):
        """A virialized (force-balanced) structure's inner nodes feel ~zero net
        force at EVERY size: the inner residual stays below TOL as the grid grows
        from n=40 to n=120 (it does not blow up with N). On the force-balanced
        lattice both are at machine precision; the meaningful invariant is that the
        bigger grid remains balanced.
        """
        res40 = virialization_residual(
            *_big_grid("radial", n=40), inner_frac=0.5, center_mass_kg=M_EXT_KG)
        res120 = virialization_residual(
            *_big_grid("radial", n=120), inner_frac=0.5, center_mass_kg=M_EXT_KG)
        assert res40["max_residual"] <= VIRIALIZATION_TOL
        assert res120["max_residual"] <= VIRIALIZATION_TOL, (
            f"residual should stay balanced as the grid grows: "
            f"n=40 -> {res40['max_residual']:.3e}, "
            f"n=120 -> {res120['max_residual']:.3e}"
        )


# ---------------------------------------------------------------------------
# 5. Determinism — same seed/params -> identical residuals
# ---------------------------------------------------------------------------

class TestDeterminism:

    @pytest.mark.parametrize("rule", ["radial", "massfunc"])
    def test_same_seed_identical_residual(self, rule):
        r1 = virialization_residual(
            *_big_grid(rule), inner_frac=0.5, center_mass_kg=M_EXT_KG)
        r2 = virialization_residual(
            *_big_grid(rule), inner_frac=0.5, center_mass_kg=M_EXT_KG)
        np.testing.assert_array_equal(
            r1["residual_per_node"], r2["residual_per_node"])
        assert r1["max_residual"] == r2["max_residual"]
        np.testing.assert_array_equal(r1["inner_idx"], r2["inner_idx"])


# ---------------------------------------------------------------------------
# 6. CENTER-ONLY selector — isolate the DEEP INTERIOR of a LARGE grid (item 1)
# ---------------------------------------------------------------------------

class TestCenterOnlySelector:
    """The user's actual criterion: only the DEEP-INTERIOR nodes of a LARGE grid
    should be net-force ~null (outer nodes feel an expected inward pull). center_k /
    center_frac select nodes by distance to the CENTROID, independent of r_max, so a
    bigger grid genuinely deepens the interior being tested."""

    def test_center_k_isolates_innermost_nodes(self):
        """center_k=K returns exactly the K nodes closest to the centroid."""
        pos, masses = _big_grid("radial", n=100)
        res = virialization_residual(
            pos, masses, center_k=10, center_mass_kg=M_EXT_KG)
        assert res["n_inner"] == 10
        assert res["selector"] == "center_k=10"
        centroid = pos.mean(axis=0)
        d = np.linalg.norm(pos - centroid, axis=1)
        expected = set(np.argsort(d, kind="stable")[:10].tolist())
        assert set(res["inner_idx"].tolist()) == expected

    def test_center_frac_scales_with_grid(self):
        """center_frac selects ceil(frac*N) nodes (scales with grid size)."""
        pos, masses = _big_grid("radial", n=120)
        res = virialization_residual(
            pos, masses, center_frac=0.25, center_mass_kg=M_EXT_KG)
        assert res["n_inner"] == int(np.ceil(0.25 * 120))
        assert "center_frac" in res["selector"]

    def test_center_k_takes_priority_over_inner_frac(self):
        pos, masses = _big_grid("radial", n=80)
        res = virialization_residual(
            pos, masses, inner_frac=0.5, center_k=7, center_mass_kg=M_EXT_KG)
        assert res["n_inner"] == 7
        assert res["selector"].startswith("center_k")

    def test_legacy_inner_frac_unchanged_default(self):
        """With no center_* given, the selector is the legacy inner_frac path."""
        pos, masses = _big_grid("radial", n=80)
        res = virialization_residual(pos, masses, center_mass_kg=M_EXT_KG)
        assert res["selector"] == "inner_frac=0.5"

    def test_deep_center_of_big_lattice_is_near_zero(self):
        """On the FORCE-BALANCED lattice the deep-center (center_k) residual is at
        machine precision and stays there as the grid grows from n=100 to n=500 —
        the size-independent selector confirms the interior is genuinely balanced,
        not an artifact of 'inner half of a small ball'."""
        for n in (100, 500):
            pos, masses = _big_grid("radial", n=n)
            res = virialization_residual(
                pos, masses, center_k=15, center_mass_kg=M_EXT_KG)
            assert res["max_residual"] <= VIRIALIZATION_TOL
            assert res["max_residual"] < 1e-6

    def test_deep_center_of_realistic_blob_is_large(self):
        """The realistic (un-relaxed) blob's deep center is FAR from balanced for
        both rules — re-confirming PF8 under the corrected center-only metric."""
        for rule in ("radial", "massfunc"):
            pos, masses = _big_grid(rule, n=100, relax_steps=0)
            res = virialization_residual(
                pos, masses, center_k=15, center_mass_kg=M_EXT_KG)
            assert res["max_residual"] > 1.0


# ---------------------------------------------------------------------------
# 7. Force-residual objective gradient (Option B's descent direction)
# ---------------------------------------------------------------------------

class TestForceResidualGradient:
    """Option B descends f = sum_i |a_i|^2. Its analytic gradient must match a
    finite-difference gradient (the descent is only correct if the gradient is)."""

    def test_analytic_gradient_matches_finite_difference(self):
        rng = np.random.default_rng(0)
        pos = rng.standard_normal((9, 3)) * 1.0e24
        masses = np.abs(rng.standard_normal(9)) + 0.5
        cm = float(np.mean(masses))
        f0, grad = _force_residual_objective_and_grad(pos, masses, cm, G)
        eps = 1.0e16
        gnum = np.zeros_like(pos)
        for i in range(pos.shape[0]):
            for d in range(3):
                p2 = pos.copy()
                p2[i, d] += eps
                f2, _ = _force_residual_objective_and_grad(p2, masses, cm, G)
                gnum[i, d] = (f2 - f0) / eps
        rel = np.abs(grad - gnum) / (np.abs(gnum) + 1e-300)
        assert np.max(rel) < 1e-3, f"gradient mismatch (max rel err {np.max(rel):.2e})"

    def test_objective_is_nonnegative(self):
        pos, masses = _big_grid("radial", n=40, relax_steps=0)
        f, _ = _force_residual_objective_and_grad(
            pos, masses, float(np.mean(masses)), G)
        assert f >= 0.0


# ---------------------------------------------------------------------------
# 8. OPTION B vs OPTION A — true relaxation reduces, but does not crystallize
# ---------------------------------------------------------------------------

class TestOptionBvsOptionA:
    """Build Option B (gradient relaxation) and COMPARE to Option A (lattice) on the
    center-only metric. The honest result: B's monotone descent reduces the center
    residual but never reaches the lattice's machine-precision balance."""

    def _build(self, rule, n, *, mode="lattice", steps=1, rate=0.1):
        return build_virialized_grid(
            S_DEFAULT_M, n_nodes=n, M_ext_kg=M_EXT_KG, vir_mass_rule=rule,
            vir_mass_spread=BIG_SPREAD, vir_segregation=BIG_SEG,
            vir_extent=BIG_EXTENT, vir_relax_mode=mode, vir_relax_steps=steps,
            vir_relax_rate=rate, seed=BIG_SEED)

    @pytest.mark.parametrize("rule", ["radial", "massfunc"])
    def test_option_b_reduces_center_residual(self, rule):
        """A moderate Option-B relaxation lowers the deep-center residual below the
        realistic (un-relaxed) start, for BOTH mass rules."""
        pos0, m0 = self._build(rule, 100, steps=0)            # realistic start
        posB, mB = self._build(rule, 100, mode="gradient", steps=20)
        r0 = virialization_residual(pos0, m0, center_k=12, center_mass_kg=M_EXT_KG)
        rB = virialization_residual(posB, mB, center_k=12, center_mass_kg=M_EXT_KG)
        assert rB["max_residual"] < r0["max_residual"]

    @pytest.mark.parametrize("rule", ["radial", "massfunc"])
    def test_option_a_beats_option_b_at_center(self, rule):
        """Option A (lattice) center residual is orders of magnitude below Option B
        (relaxed): the crystal is balanced; the realistic relaxed blob is not."""
        posA, mA = self._build(rule, 100, mode="lattice", steps=1)
        posB, mB = self._build(rule, 100, mode="gradient", steps=20)
        rA = virialization_residual(posA, mA, center_k=12, center_mass_kg=M_EXT_KG)
        rB = virialization_residual(posB, mB, center_k=12, center_mass_kg=M_EXT_KG)
        assert rA["max_residual"] <= VIRIALIZATION_TOL
        assert rB["max_residual"] > VIRIALIZATION_TOL
        assert rA["max_residual"] < rB["max_residual"]

    def test_option_b_objective_decreases_with_steps(self):
        """Within a single relaxation run the force-residual objective f decreases
        monotonically (backtracking guarantee). Measured on the RAW relaxed positions
        before the per-build NN-spacing rescale (a global factor that would otherwise
        change f's absolute scale across rebuilds)."""
        from cosmo.node_geometry import _gradient_relax_positions
        pos0, m0 = self._build("radial", 80, steps=0)  # realistic start
        cm = float(np.mean(m0))
        f_prev = None
        for steps in (0, 5, 15, 30):
            relaxed = _gradient_relax_positions(
                pos0, m0, n_steps=steps, rate=0.1, hold_outer_frac=0.3)
            f, _ = _force_residual_objective_and_grad(relaxed, m0, cm, G)
            if f_prev is not None:
                assert f <= f_prev + 1e-9 * abs(f_prev)
            f_prev = f

    def test_option_b_preserves_mean_and_spacing(self):
        """Option B keeps the mass + spacing contracts (only positions relax)."""
        from cosmo.node_geometry import nearest_neighbour_spacing
        pos, m = self._build("radial", 100, mode="gradient", steps=10)
        np.testing.assert_allclose(m.mean(), M_EXT_KG, rtol=1e-12)
        np.testing.assert_allclose(
            nearest_neighbour_spacing(pos, "median"), S_DEFAULT_M, rtol=1e-6)
        assert len(np.unique(np.round(np.linalg.norm(pos, axis=1), 3))) >= 2

    def test_option_b_deterministic(self):
        a = self._build("massfunc", 80, mode="gradient", steps=12)
        b = self._build("massfunc", 80, mode="gradient", steps=12)
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[1], b[1])

    def test_option_b_m_ext_zero_is_eds(self):
        """M_ext=0 => zero masses, no NaN, positions finite (M=0==EdS preserved)."""
        pos, m = build_virialized_grid(
            S_DEFAULT_M, n_nodes=60, M_ext_kg=0.0, vir_mass_rule="radial",
            vir_mass_spread=0.5, vir_relax_mode="gradient", vir_relax_steps=10,
            seed=1)
        assert np.all(np.isfinite(pos))
        np.testing.assert_array_equal(m, np.zeros(60))
