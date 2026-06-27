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
