"""
Tests for the VIRIALIZED node grid (Section 2).

The virialized geometry is the only one that returns COUPLED (positions, masses):
a more massive node sits FURTHER from the centre (mass segregation, like a relaxed
cluster). It is reached via cosmo.node_geometry.build_virialized_grid(...) or by an
HMEAGrid whose ExternalNodeParameters.node_geometry == "virialized" (which takes the
coupled branch in _create_grid).

Invariants proven here:
  - Generator returns (N,3) positions + (N,) masses, N == vir_n_nodes, float64.
  - Mean-preserving: mean(masses) == M_ext_kg (rtol 1e-12); sum == N*M_ext_kg; all>0.
  - Falsifiable: vir_mass_spread=0 -> uniform masses (both rules);
                 vir_segregation=0 -> mass/radius decoupled (~0 correlation).
  - Segregation present (spread>0, segregation>0): mass-radius correlation POSITIVE.
  - vir_extent: outer radius scales ~linearly with vir_extent.
  - vir_s_metric: median & mean NN spacing both finite/positive; the realized
    spacing equals the targeted S per metric.
  - Determinism per (seed, params); different seed differs.
  - RNG isolation: independent of global np.random.
  - Volume-filling: >= 2 distinct radii (not a hollow shell).
  - Threading: HMEAGrid("virialized") -> len(nodes)==vir_n_nodes; mean==M_ext_kg;
    positions match the generator (before node_s) / compose with node_s_amplitude.
  - SimulationParameters threads vir_* into external_params; SweepConfig defaults;
    cache slug distinct for virialized & across vir_* values; non-virialized keys
    carry NO vir slug (regression).
  - cube26 byte-identical (opt-in invariant); M=0 builds & total mass 0.
  - build_node_positions("virialized") raises a helpful error (coupled-only).
"""
import numpy as np
import pytest

from cosmo.node_geometry import (
    build_virialized_grid,
    build_node_positions,
    list_geometries,
    nearest_neighbour_spacing,
    virialization_residual,
)
from cosmo.constants import ExternalNodeParameters, SimulationParameters, CosmologicalConstants
from cosmo.particles import HMEAGrid


const = CosmologicalConstants()
M_EXT_KG = 5e55
S_DEFAULT_M = 30.0 * const.Gpc_to_m

RULES = ["radial", "massfunc"]
METRICS = ["median", "mean"]
N_VALUES = [12, 26, 54]


def _radii(positions):
    return np.linalg.norm(positions, axis=1)


# ---------------------------------------------------------------------------
# A. Generator: shapes / dtype / count
# ---------------------------------------------------------------------------

class TestGeneratorShapes:

    @pytest.mark.parametrize("rule", RULES)
    @pytest.mark.parametrize("n", N_VALUES)
    def test_shapes_dtype_count(self, rule, n):
        pos, masses = build_virialized_grid(
            S_DEFAULT_M, n_nodes=n, M_ext_kg=M_EXT_KG, vir_mass_rule=rule,
            vir_mass_spread=0.5, seed=42,
        )
        assert pos.shape == (n, 3)
        assert masses.shape == (n,)
        assert pos.dtype == np.float64
        assert masses.dtype == np.float64

    def test_registered_in_list_geometries(self):
        assert "virialized" in list_geometries()


# ---------------------------------------------------------------------------
# B. Mean-preservation (both rules)
# ---------------------------------------------------------------------------

class TestMeanPreservation:

    @pytest.mark.parametrize("rule", RULES)
    @pytest.mark.parametrize("spread", [0.0, 0.3, 0.8])
    @pytest.mark.parametrize("n", N_VALUES)
    def test_mean_equals_M_ext_kg(self, rule, spread, n):
        _, masses = build_virialized_grid(
            S_DEFAULT_M, n_nodes=n, M_ext_kg=M_EXT_KG, vir_mass_rule=rule,
            vir_mass_spread=spread, seed=7,
        )
        np.testing.assert_allclose(masses.mean(), M_EXT_KG, rtol=1e-12)

    @pytest.mark.parametrize("rule", RULES)
    @pytest.mark.parametrize("n", N_VALUES)
    def test_total_is_N_times_M_ext_and_positive(self, rule, n):
        _, masses = build_virialized_grid(
            S_DEFAULT_M, n_nodes=n, M_ext_kg=M_EXT_KG, vir_mass_rule=rule,
            vir_mass_spread=0.6, seed=7,
        )
        np.testing.assert_allclose(masses.sum(), n * M_EXT_KG, rtol=1e-12)
        assert np.all(masses > 0)


# ---------------------------------------------------------------------------
# C. Falsifiable reductions
# ---------------------------------------------------------------------------

class TestFalsifiableReductions:

    @pytest.mark.parametrize("rule", RULES)
    def test_spread_zero_gives_uniform_masses(self, rule):
        """vir_mass_spread == 0 -> every node mass == M_ext_kg (both rules)."""
        _, masses = build_virialized_grid(
            S_DEFAULT_M, n_nodes=26, M_ext_kg=M_EXT_KG, vir_mass_rule=rule,
            vir_mass_spread=0.0, vir_segregation=1.0, seed=3,
        )
        np.testing.assert_allclose(masses, np.full(26, M_EXT_KG), rtol=1e-12)

    @pytest.mark.parametrize("rule", RULES)
    def test_segregation_zero_decouples_mass_and_radius(self, rule):
        """vir_segregation == 0 -> mass<->radius correlation ~ 0 (decoupled).

        Asserted on the REALISTIC (vir_relax_steps=0) layout where every node has a
        distinct radius, so a decoupled (random) mass assignment yields ~0
        correlation. The force-balanced lattice has only a handful of radius SHELLS
        (antipodes share a mass to keep the force balance), so shell-level random
        assignment cannot fully decouple at small shell counts — decoupling is a
        property of the realistic, per-node-radius mode.
        """
        pos, masses = build_virialized_grid(
            S_DEFAULT_M, n_nodes=54, M_ext_kg=M_EXT_KG, vir_mass_rule=rule,
            vir_mass_spread=0.8, vir_segregation=0.0, vir_relax_steps=0, seed=11,
        )
        r = _radii(pos)
        if masses.std() == 0.0 or r.std() == 0.0:
            pytest.skip("degenerate (no spread) -> correlation undefined")
        corr = np.corrcoef(masses, r)[0, 1]
        assert abs(corr) < 0.35, f"{rule}: segregation=0 should decouple, corr={corr:.3f}"

    def test_spread_zero_and_segregation_zero_clean_null(self):
        """spread=0 AND segregation=0 -> uniform masses, isotropic-ish positions."""
        pos, masses = build_virialized_grid(
            S_DEFAULT_M, n_nodes=26, M_ext_kg=M_EXT_KG, vir_mass_rule="massfunc",
            vir_mass_spread=0.0, vir_segregation=0.0, seed=5,
        )
        np.testing.assert_allclose(masses, np.full(26, M_EXT_KG), rtol=1e-12)
        assert pos.shape == (26, 3)


# ---------------------------------------------------------------------------
# D. Segregation present (mass-radius correlation POSITIVE)
# ---------------------------------------------------------------------------

class TestSegregationPositive:

    @pytest.mark.parametrize("rule", RULES)
    def test_bigger_mass_larger_radius(self, rule):
        pos, masses = build_virialized_grid(
            S_DEFAULT_M, n_nodes=54, M_ext_kg=M_EXT_KG, vir_mass_rule=rule,
            vir_mass_spread=0.8, vir_segregation=1.0, seed=21,
        )
        r = _radii(pos)
        corr = np.corrcoef(masses, r)[0, 1]
        assert corr > 0.5, f"{rule}: expected positive mass-radius correlation, got {corr:.3f}"


# ---------------------------------------------------------------------------
# E. vir_extent scaling
# ---------------------------------------------------------------------------

class TestExtent:

    def test_spacing_always_rescaled_to_target_S(self):
        """Whatever the extent, the realized NN spacing is rescaled to the target S."""
        for extent in (1.0, 2.0, 3.0):
            pos, _ = build_virialized_grid(
                S_DEFAULT_M, n_nodes=40, M_ext_kg=M_EXT_KG, vir_mass_rule="radial",
                vir_mass_spread=0.0, vir_extent=extent, seed=1,
            )
            np.testing.assert_allclose(
                nearest_neighbour_spacing(pos, "median"), S_DEFAULT_M, rtol=1e-6)

    def test_extent_widens_radial_range(self):
        """Larger vir_extent -> larger outer-to-inner radius ratio.

        vir_extent shapes the REALISTIC (vir_relax_steps=0) Fibonacci layout, whose
        radii span [0.5S, (0.5+extent)S]; the force-balanced lattice mode ignores
        vir_extent (it is a fixed cubic ball), so this property is asserted on the
        realistic mode. The NN-spacing rescale fixes the spacing to S (a global
        factor that cancels in any radius RATIO), so the scale-invariant signature
        of vir_extent is the radial spread r_max / r_min = 1 + 2*extent.
        """
        def span(extent):
            pos, _ = build_virialized_grid(
                S_DEFAULT_M, n_nodes=40, M_ext_kg=M_EXT_KG, vir_mass_rule="radial",
                vir_mass_spread=0.0, vir_extent=extent, vir_relax_steps=0, seed=1,
            )
            r = _radii(pos)
            return r.max() / r.min()
        s1, s2, s4 = span(1.0), span(2.0), span(4.0)
        assert s2 > s1 and s4 > s2, f"radial span not increasing: {s1}, {s2}, {s4}"
        # Roughly 1 + 2*extent (rescale cancels): extent=1 -> ~3, extent=2 -> ~5.
        np.testing.assert_allclose(s1, 3.0, rtol=1e-6)
        np.testing.assert_allclose(s2, 5.0, rtol=1e-6)
        np.testing.assert_allclose(s4, 9.0, rtol=1e-6)


# ---------------------------------------------------------------------------
# E2. vir_extent -> node-count coupling (item 10): a larger radial extent
#     auto-raises the node count to hold the ball DENSITY ~constant. OFF by
#     default (byte-identical); a no-op at the default extent == 1.0 even when ON.
# ---------------------------------------------------------------------------

class TestExtentNodeCoupling:

    BASE_N = 64  # an exact-ish cube count so the lattice ball is well-formed

    def test_density_law_count_scales_as_extent_cubed(self):
        """extent_coupled_n_nodes derives round(base_n * extent^3) (volume law)."""
        from cosmo.node_geometry import extent_coupled_n_nodes
        base = 64
        assert extent_coupled_n_nodes(base, 1.0) == 64        # 64 * 1^3
        assert extent_coupled_n_nodes(base, 2.0) == 512       # 64 * 8
        assert extent_coupled_n_nodes(base, 3.0) == 1728      # 64 * 27
        assert extent_coupled_n_nodes(base, 1.5) == 216       # 64 * 3.375
        assert extent_coupled_n_nodes(base, 0.5) == 8         # 64 * 0.125
        # Never degenerate: clamped to >= 1 for a tiny extent.
        assert extent_coupled_n_nodes(base, 0.0) == 1

    def test_default_extent_reproduces_base_count_exactly(self):
        """Coupling ON at extent == 1.0 yields EXACTLY the base node count."""
        for couples in (False, True):
            pos, _ = build_virialized_grid(
                S_DEFAULT_M, n_nodes=self.BASE_N, M_ext_kg=M_EXT_KG,
                vir_extent=1.0, vir_extent_couples_nodes=couples, seed=0,
            )
            assert pos.shape[0] == self.BASE_N

    def test_coupling_off_is_byte_identical_to_omitting_flag(self):
        """Default OFF leaves node count AND the grid byte-identical (no behaviour)."""
        for extent in (1.0, 2.0, 3.0):
            p_off, m_off = build_virialized_grid(
                S_DEFAULT_M, n_nodes=self.BASE_N, M_ext_kg=M_EXT_KG,
                vir_extent=extent, vir_extent_couples_nodes=False, seed=0,
            )
            p_omit, m_omit = build_virialized_grid(
                S_DEFAULT_M, n_nodes=self.BASE_N, M_ext_kg=M_EXT_KG,
                vir_extent=extent, seed=0,
            )
            assert p_off.shape[0] == self.BASE_N
            np.testing.assert_array_equal(p_off, p_omit)
            np.testing.assert_array_equal(m_off, m_omit)

    def test_on_at_default_extent_is_noop_vs_off(self):
        """ON at extent == 1.0 must be byte-identical to OFF (factor exactly 1)."""
        p_on, m_on = build_virialized_grid(
            S_DEFAULT_M, n_nodes=self.BASE_N, M_ext_kg=M_EXT_KG,
            vir_extent=1.0, vir_extent_couples_nodes=True, seed=0,
        )
        p_off, m_off = build_virialized_grid(
            S_DEFAULT_M, n_nodes=self.BASE_N, M_ext_kg=M_EXT_KG,
            vir_extent=1.0, vir_extent_couples_nodes=False, seed=0,
        )
        np.testing.assert_array_equal(p_on, p_off)
        np.testing.assert_array_equal(m_on, m_off)

    def test_larger_extent_yields_proportionally_more_nodes(self):
        """Coupling ON: realized node count == round(base_n * extent^3)."""
        from cosmo.node_geometry import extent_coupled_n_nodes
        for extent in (1.0, 1.5, 2.0, 3.0):
            pos, _ = build_virialized_grid(
                S_DEFAULT_M, n_nodes=self.BASE_N, M_ext_kg=M_EXT_KG,
                vir_extent=extent, vir_extent_couples_nodes=True, seed=0,
            )
            assert pos.shape[0] == extent_coupled_n_nodes(self.BASE_N, extent)
        # And monotonically increasing.
        counts = [
            build_virialized_grid(
                S_DEFAULT_M, n_nodes=self.BASE_N, M_ext_kg=M_EXT_KG,
                vir_extent=e, vir_extent_couples_nodes=True, seed=0)[0].shape[0]
            for e in (1.0, 1.5, 2.0, 3.0)
        ]
        assert counts == sorted(counts) and counts[0] < counts[-1]

    def test_nn_spacing_still_equals_S(self):
        """The NN-spacing contract holds at every coupled extent (realized NN == S)."""
        for extent in (1.0, 2.0, 3.0):
            pos, _ = build_virialized_grid(
                S_DEFAULT_M, n_nodes=self.BASE_N, M_ext_kg=M_EXT_KG,
                vir_extent=extent, vir_extent_couples_nodes=True, seed=0,
            )
            np.testing.assert_allclose(
                nearest_neighbour_spacing(pos, "median"), S_DEFAULT_M, rtol=1e-6)

    def test_mean_preserved_at_every_coupled_extent(self):
        """mean(masses) == M_ext_kg for every coupled extent (mass contract held)."""
        for extent in (1.0, 2.0, 3.0):
            _, masses = build_virialized_grid(
                S_DEFAULT_M, n_nodes=self.BASE_N, M_ext_kg=M_EXT_KG,
                vir_mass_spread=0.5, vir_extent=extent,
                vir_extent_couples_nodes=True, seed=0,
            )
            np.testing.assert_allclose(masses.mean(), M_EXT_KG, rtol=1e-12)

    def test_density_roughly_constant_across_extents(self):
        """N / reach^3 (the ball density) stays ~constant as extent grows.

        The coupling is designed so a larger extent reaches further (more nodes)
        WITHOUT thinning out: density = N / (radial reach)^3 should be roughly
        flat. The lattice ball's reach/N relation is not perfectly cubic at small
        N, so we allow a generous band but require it does NOT drift by an order of
        magnitude (i.e. node density really is being held, not the radius fixed).
        """
        densities = []
        for extent in (1.0, 1.5, 2.0, 3.0):
            pos, _ = build_virialized_grid(
                S_DEFAULT_M, n_nodes=self.BASE_N, M_ext_kg=M_EXT_KG,
                vir_extent=extent, vir_extent_couples_nodes=True, seed=0,
            )
            n_eff = pos.shape[0]
            reach = np.linalg.norm(pos, axis=1).max() / S_DEFAULT_M
            densities.append(n_eff / reach ** 3)
        densities = np.array(densities)
        # Density stays within +/-30% of its mean across a 1->3 extent range
        # (a fixed-node-count grid would instead drop ~27x in density at extent 3).
        rel_spread = (densities.max() - densities.min()) / densities.mean()
        assert rel_spread < 0.3, f"density not ~constant: {densities}"

    def test_coupling_makes_extent_meaningful_in_lattice_mode(self):
        """In the force-balanced lattice mode vir_extent is a no-op UNLESS coupled.

        Without coupling, the force-balanced lattice ball ignores vir_extent (its
        radius derives from node count): the grid is byte-identical across extents.
        With coupling ON, a larger extent raises the node count and the realized
        radial reach grows -> the knob is meaningful again.
        """
        # No coupling: extent has NO effect on the force-balanced lattice.
        p1, _ = build_virialized_grid(
            S_DEFAULT_M, n_nodes=self.BASE_N, M_ext_kg=M_EXT_KG,
            vir_extent=1.0, vir_relax_steps=1, seed=0)
        p2, _ = build_virialized_grid(
            S_DEFAULT_M, n_nodes=self.BASE_N, M_ext_kg=M_EXT_KG,
            vir_extent=3.0, vir_relax_steps=1, seed=0)
        np.testing.assert_array_equal(p1, p2)  # extent inert without coupling
        # With coupling: extent now drives reach (more nodes -> larger ball).
        reach1 = np.linalg.norm(build_virialized_grid(
            S_DEFAULT_M, n_nodes=self.BASE_N, M_ext_kg=M_EXT_KG, vir_extent=1.0,
            vir_extent_couples_nodes=True, seed=0)[0], axis=1).max()
        reach3 = np.linalg.norm(build_virialized_grid(
            S_DEFAULT_M, n_nodes=self.BASE_N, M_ext_kg=M_EXT_KG, vir_extent=3.0,
            vir_extent_couples_nodes=True, seed=0)[0], axis=1).max()
        assert reach3 > 1.5 * reach1

    def test_center_stays_virialized_as_extent_grows(self):
        """The deep-center force residual stays tiny as the coupled grid grows.

        Growing the grid via extent must not break the force balance the lattice
        mode guarantees: the center-only residual stays << 1 (machine-precision-ish)
        at every extent, so a bigger virialized ball is still virialized at its core.
        """
        for extent in (1.0, 2.0, 3.0):
            pos, masses = build_virialized_grid(
                S_DEFAULT_M, n_nodes=self.BASE_N, M_ext_kg=M_EXT_KG,
                vir_extent=extent, vir_extent_couples_nodes=True, seed=0,
            )
            res = virialization_residual(pos, masses, center_k=8)
            assert res["max_residual"] < 1e-3, (
                f"center not virialized at extent={extent}: {res['max_residual']}")


# ---------------------------------------------------------------------------
# F. vir_s_metric (median / mean)
# ---------------------------------------------------------------------------

class TestSMetric:

    @pytest.mark.parametrize("metric", METRICS)
    def test_realized_spacing_equals_target_S(self, metric):
        pos, _ = build_virialized_grid(
            S_DEFAULT_M, n_nodes=54, M_ext_kg=M_EXT_KG, vir_mass_rule="radial",
            vir_mass_spread=0.4, vir_s_metric=metric, seed=9,
        )
        realized = nearest_neighbour_spacing(pos, metric)
        np.testing.assert_allclose(realized, S_DEFAULT_M, rtol=1e-6)

    def test_median_and_mean_both_finite_positive(self):
        pos, _ = build_virialized_grid(
            S_DEFAULT_M, n_nodes=54, M_ext_kg=M_EXT_KG, vir_mass_rule="radial",
            vir_mass_spread=0.4, seed=9,
        )
        med = nearest_neighbour_spacing(pos, "median")
        mean = nearest_neighbour_spacing(pos, "mean")
        assert np.isfinite(med) and med > 0
        assert np.isfinite(mean) and mean > 0

    def test_metric_switch_changes_targeted_S_layout(self):
        """Targeting median vs mean produces different absolute layouts in general.

        Asserted on the REALISTIC (vir_relax_steps=0) Fibonacci layout, whose
        irregular NN distribution makes median != mean. The force-balanced lattice
        mode is a near-uniform crystal (median ~ mean), so the two metrics yield the
        same rescale there; the meaningful difference lives in the realistic mode.
        """
        pos_med, _ = build_virialized_grid(
            S_DEFAULT_M, n_nodes=54, M_ext_kg=M_EXT_KG, vir_mass_rule="radial",
            vir_mass_spread=0.4, vir_s_metric="median", vir_relax_steps=0, seed=9,
        )
        pos_mean, _ = build_virialized_grid(
            S_DEFAULT_M, n_nodes=54, M_ext_kg=M_EXT_KG, vir_mass_rule="radial",
            vir_mass_spread=0.4, vir_s_metric="mean", vir_relax_steps=0, seed=9,
        )
        assert not np.array_equal(pos_med, pos_mean), \
            "median vs mean s-metric should produce different absolute layouts"


# ---------------------------------------------------------------------------
# G. Determinism & RNG isolation
# ---------------------------------------------------------------------------

class TestDeterminismAndRNG:

    def test_same_seed_identical(self):
        a = build_virialized_grid(
            S_DEFAULT_M, n_nodes=26, M_ext_kg=M_EXT_KG, vir_mass_rule="massfunc",
            vir_mass_spread=0.6, seed=42,
        )
        b = build_virialized_grid(
            S_DEFAULT_M, n_nodes=26, M_ext_kg=M_EXT_KG, vir_mass_rule="massfunc",
            vir_mass_spread=0.6, seed=42,
        )
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[1], b[1])

    def test_different_seed_differs(self):
        p1, m1 = build_virialized_grid(
            S_DEFAULT_M, n_nodes=26, M_ext_kg=M_EXT_KG, vir_mass_rule="massfunc",
            vir_mass_spread=0.6, seed=1,
        )
        p2, m2 = build_virialized_grid(
            S_DEFAULT_M, n_nodes=26, M_ext_kg=M_EXT_KG, vir_mass_rule="massfunc",
            vir_mass_spread=0.6, seed=2,
        )
        assert not (np.array_equal(p1, p2) and np.array_equal(m1, m2))

    def test_independent_of_global_rng(self):
        np.random.seed(999); _ = np.random.rand(1000)
        before = build_virialized_grid(
            S_DEFAULT_M, n_nodes=26, M_ext_kg=M_EXT_KG, vir_mass_rule="massfunc",
            vir_mass_spread=0.6, seed=42,
        )
        np.random.seed(0); _ = np.random.rand(5000)
        after = build_virialized_grid(
            S_DEFAULT_M, n_nodes=26, M_ext_kg=M_EXT_KG, vir_mass_rule="massfunc",
            vir_mass_spread=0.6, seed=42,
        )
        np.testing.assert_array_equal(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])


# ---------------------------------------------------------------------------
# G2. vir_relax_steps BALANCE LEVEL (force-balanced lattice mode)
# ---------------------------------------------------------------------------

_BAL_TOL = 0.25  # mirrors VIRIALIZATION_TOL in test_virialization_validation.

def _legacy_fibonacci_grid(rule, n=100, **kw):
    """The realistic (un-balanced) layout = vir_relax_steps=0."""
    return build_virialized_grid(
        S_DEFAULT_M, n_nodes=n, M_ext_kg=M_EXT_KG, vir_mass_rule=rule,
        vir_mass_spread=0.8, vir_segregation=1.0, vir_extent=2.5,
        vir_relax_steps=0, seed=12, **kw,
    )


class TestRelaxBalanceLevel:
    """vir_relax_steps reframed as a BALANCE LEVEL: 0 = realistic Fibonacci layout
    (byte-identical to the legacy generator), >= 1 = force-balanced lattice ball."""

    def test_default_is_force_balanced(self):
        """The DEFAULT (vir_relax_steps unset == 1) is force-balanced: inner
        max_residual << TOL for BOTH rules (the headline criterion)."""
        for rule in RULES:
            pos, masses = build_virialized_grid(
                S_DEFAULT_M, n_nodes=100, M_ext_kg=M_EXT_KG, vir_mass_rule=rule,
                vir_mass_spread=0.8, vir_segregation=1.0, vir_extent=2.5, seed=12,
            )
            res = virialization_residual(
                pos, masses, inner_frac=0.5, center_mass_kg=M_EXT_KG)
            assert res["max_residual"] <= _BAL_TOL, (
                f"{rule}: default grid not force-balanced "
                f"(max_residual={res['max_residual']:.3e})")

    @pytest.mark.parametrize("rule", RULES)
    def test_balanced_residual_far_below_unbalanced(self, rule):
        """balanced (steps>=1) max_residual <= TOL << unbalanced (steps=0)."""
        pos_b, m_b = build_virialized_grid(
            S_DEFAULT_M, n_nodes=100, M_ext_kg=M_EXT_KG, vir_mass_rule=rule,
            vir_mass_spread=0.8, vir_segregation=1.0, vir_extent=2.5,
            vir_relax_steps=1, seed=12)
        pos_u, m_u = _legacy_fibonacci_grid(rule)
        rb = virialization_residual(pos_b, m_b, inner_frac=0.5, center_mass_kg=M_EXT_KG)
        ru = virialization_residual(pos_u, m_u, inner_frac=0.5, center_mass_kg=M_EXT_KG)
        assert rb["max_residual"] <= _BAL_TOL
        assert ru["max_residual"] > 1.0  # unbalanced is far from balanced
        assert rb["max_residual"] < ru["max_residual"]

    def test_steps_zero_reproduces_legacy_generator(self):
        """vir_relax_steps=0 reproduces the realistic Fibonacci layout: no node at
        the origin and it obeys the extent radial-range law (1+2*extent), which the
        lattice ball does NOT — proving steps=0 is the legacy generator unchanged."""
        pos, _ = build_virialized_grid(
            S_DEFAULT_M, n_nodes=40, M_ext_kg=M_EXT_KG, vir_mass_rule="radial",
            vir_mass_spread=0.0, vir_extent=2.0, vir_relax_steps=0, seed=1)
        r = _radii(pos)
        assert r.min() > 0.0, "legacy layout has no node at the origin"
        np.testing.assert_allclose(r.max() / r.min(), 5.0, rtol=1e-6)

    def test_balanced_has_node_at_origin(self):
        """The force-balanced lattice ball includes a node AT the origin (the
        symmetric centre that makes antipodal pulls cancel)."""
        pos, _ = build_virialized_grid(
            S_DEFAULT_M, n_nodes=100, M_ext_kg=M_EXT_KG, vir_mass_rule="radial",
            vir_mass_spread=0.8, seed=12)
        r = _radii(pos)
        assert np.min(r) == 0.0

    @pytest.mark.parametrize("rule", RULES)
    def test_balanced_preserves_segregation_mean_spacing(self, rule):
        """Post-balance: positive mass-radius correlation, mean(masses)==M_ext_kg,
        NN-spacing==S, >=2 distinct radii — all the realistic-mode contracts hold."""
        pos, masses = build_virialized_grid(
            S_DEFAULT_M, n_nodes=100, M_ext_kg=M_EXT_KG, vir_mass_rule=rule,
            vir_mass_spread=0.8, vir_segregation=1.0, seed=12)
        r = _radii(pos)
        np.testing.assert_allclose(masses.mean(), M_EXT_KG, rtol=1e-12)
        np.testing.assert_allclose(
            nearest_neighbour_spacing(pos, "median"), S_DEFAULT_M, rtol=1e-6)
        assert len(np.unique(np.round(r, 3))) >= 2
        corr = np.corrcoef(masses, r)[0, 1]
        assert corr > 0.5, f"{rule}: balanced grid lost segregation (corr={corr:.3f})"

    @pytest.mark.parametrize("rule", RULES)
    def test_balanced_determinism(self, rule):
        """Same (seed, params, steps) -> byte-identical balanced grid."""
        a = build_virialized_grid(
            S_DEFAULT_M, n_nodes=80, M_ext_kg=M_EXT_KG, vir_mass_rule=rule,
            vir_mass_spread=0.7, seed=5)
        b = build_virialized_grid(
            S_DEFAULT_M, n_nodes=80, M_ext_kg=M_EXT_KG, vir_mass_rule=rule,
            vir_mass_spread=0.7, seed=5)
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[1], b[1])

    def test_balanced_m_ext_zero_no_nan(self):
        """M_ext_kg=0 with balancing on -> all masses 0, no NaN (M=0==EdS)."""
        pos, masses = build_virialized_grid(
            S_DEFAULT_M, n_nodes=80, M_ext_kg=0.0, vir_mass_rule="radial",
            vir_mass_spread=0.5, seed=1)
        assert np.all(np.isfinite(pos))
        np.testing.assert_array_equal(masses, np.zeros(80))

    def test_balanced_global_rng_isolation(self):
        """Balancing adds no global-RNG draws (massfunc uses default_rng(seed))."""
        np.random.seed(123); _ = np.random.rand(777)
        before = build_virialized_grid(
            S_DEFAULT_M, n_nodes=80, M_ext_kg=M_EXT_KG, vir_mass_rule="massfunc",
            vir_mass_spread=0.6, seed=42)
        np.random.seed(0); _ = np.random.rand(4242)
        after = build_virialized_grid(
            S_DEFAULT_M, n_nodes=80, M_ext_kg=M_EXT_KG, vir_mass_rule="massfunc",
            vir_mass_spread=0.6, seed=42)
        np.testing.assert_array_equal(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])


# ---------------------------------------------------------------------------
# G3. Option B — vir_relax_mode="gradient" true iterative relaxation
# ---------------------------------------------------------------------------

class TestRelaxModeGradient:
    """vir_relax_mode='gradient' (Option B): a realistic blob relaxed toward force
    balance by gradient descent. Opt-in (default 'lattice' is byte-identical)."""

    def test_default_mode_is_lattice_byte_identical(self):
        """Default vir_relax_mode='lattice' => byte-identical to the prior generator
        (no gradient relaxation runs unless explicitly requested)."""
        a = build_virialized_grid(
            S_DEFAULT_M, n_nodes=80, M_ext_kg=M_EXT_KG, vir_mass_rule="radial",
            vir_mass_spread=0.7, seed=5)
        b = build_virialized_grid(
            S_DEFAULT_M, n_nodes=80, M_ext_kg=M_EXT_KG, vir_mass_rule="radial",
            vir_mass_spread=0.7, vir_relax_mode="lattice", seed=5)
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[1], b[1])

    def test_bad_mode_raises(self):
        with pytest.raises(ValueError, match="vir_relax_mode"):
            build_virialized_grid(
                S_DEFAULT_M, n_nodes=26, M_ext_kg=M_EXT_KG,
                vir_relax_mode="bogus")

    @pytest.mark.parametrize("rule", RULES)
    def test_gradient_differs_from_lattice_and_realistic(self, rule):
        """Gradient mode produces a DISTINCT layout from both the lattice (A) and the
        realistic (steps=0) start — it actually moves the interior nodes."""
        lat, _ = build_virialized_grid(
            S_DEFAULT_M, n_nodes=80, M_ext_kg=M_EXT_KG, vir_mass_rule=rule,
            vir_mass_spread=0.7, vir_relax_mode="lattice", vir_relax_steps=1, seed=5)
        real, _ = build_virialized_grid(
            S_DEFAULT_M, n_nodes=80, M_ext_kg=M_EXT_KG, vir_mass_rule=rule,
            vir_mass_spread=0.7, vir_relax_mode="lattice", vir_relax_steps=0, seed=5)
        grad, _ = build_virialized_grid(
            S_DEFAULT_M, n_nodes=80, M_ext_kg=M_EXT_KG, vir_mass_rule=rule,
            vir_mass_spread=0.7, vir_relax_mode="gradient", vir_relax_steps=10, seed=5)
        assert not np.array_equal(grad, lat)
        assert not np.array_equal(grad, real)

    def test_gradient_zero_steps_is_realistic_start(self):
        """vir_relax_steps=0 in gradient mode == the realistic Fibonacci layout
        (no relaxation iterations run)."""
        grad, _ = build_virialized_grid(
            S_DEFAULT_M, n_nodes=60, M_ext_kg=M_EXT_KG, vir_mass_rule="radial",
            vir_mass_spread=0.7, vir_relax_mode="gradient", vir_relax_steps=0, seed=5)
        real, _ = build_virialized_grid(
            S_DEFAULT_M, n_nodes=60, M_ext_kg=M_EXT_KG, vir_mass_rule="radial",
            vir_mass_spread=0.7, vir_relax_mode="lattice", vir_relax_steps=0, seed=5)
        np.testing.assert_array_equal(grad, real)

    @pytest.mark.parametrize("rule", RULES)
    def test_gradient_preserves_mean_spacing_volume(self, rule):
        pos, masses = build_virialized_grid(
            S_DEFAULT_M, n_nodes=100, M_ext_kg=M_EXT_KG, vir_mass_rule=rule,
            vir_mass_spread=0.8, vir_segregation=1.0, vir_extent=2.5,
            vir_relax_mode="gradient", vir_relax_steps=15, seed=12)
        r = _radii(pos)
        np.testing.assert_allclose(masses.mean(), M_EXT_KG, rtol=1e-12)
        np.testing.assert_allclose(
            nearest_neighbour_spacing(pos, "median"), S_DEFAULT_M, rtol=1e-6)
        assert len(np.unique(np.round(r, 3))) >= 2

    def test_gradient_global_rng_isolation(self):
        """Gradient relaxation adds no global-RNG draws (massfunc uses default_rng)."""
        np.random.seed(123); _ = np.random.rand(777)
        before = build_virialized_grid(
            S_DEFAULT_M, n_nodes=60, M_ext_kg=M_EXT_KG, vir_mass_rule="massfunc",
            vir_mass_spread=0.6, vir_relax_mode="gradient", vir_relax_steps=8, seed=42)
        np.random.seed(0); _ = np.random.rand(4242)
        after = build_virialized_grid(
            S_DEFAULT_M, n_nodes=60, M_ext_kg=M_EXT_KG, vir_mass_rule="massfunc",
            vir_mass_spread=0.6, vir_relax_mode="gradient", vir_relax_steps=8, seed=42)
        np.testing.assert_array_equal(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])

    def test_gradient_m_ext_zero_no_nan(self):
        pos, masses = build_virialized_grid(
            S_DEFAULT_M, n_nodes=60, M_ext_kg=0.0, vir_mass_rule="radial",
            vir_mass_spread=0.5, vir_relax_mode="gradient", vir_relax_steps=10, seed=1)
        assert np.all(np.isfinite(pos))
        np.testing.assert_array_equal(masses, np.zeros(60))


# ---------------------------------------------------------------------------
# H. Volume-filling (not a hollow shell)
# ---------------------------------------------------------------------------

class TestVolumeFilling:

    @pytest.mark.parametrize("rule", RULES)
    def test_at_least_two_distinct_radii(self, rule):
        pos, _ = build_virialized_grid(
            S_DEFAULT_M, n_nodes=26, M_ext_kg=M_EXT_KG, vir_mass_rule=rule,
            vir_mass_spread=0.5, seed=4,
        )
        r = _radii(pos)
        assert len(np.unique(np.round(r, 6))) >= 2, "virialized grid must not be a hollow shell"


# ---------------------------------------------------------------------------
# I. Positions-only path raises (coupled-only)
# ---------------------------------------------------------------------------

class TestPositionsOnlyRaises:

    def test_build_node_positions_virialized_raises(self):
        with pytest.raises(ValueError, match="build_virialized_grid"):
            build_node_positions("virialized", S_DEFAULT_M)


# ---------------------------------------------------------------------------
# J. NN-spacing helper sanity
# ---------------------------------------------------------------------------

class TestNNSpacingHelper:

    def test_known_spacing(self):
        # Two nodes a distance d apart: NN spacing == d for both metrics.
        d = 7.0
        pos = np.array([[0.0, 0.0, 0.0], [d, 0.0, 0.0]])
        assert nearest_neighbour_spacing(pos, "median") == pytest.approx(d)
        assert nearest_neighbour_spacing(pos, "mean") == pytest.approx(d)

    def test_needs_two_nodes(self):
        with pytest.raises(ValueError):
            nearest_neighbour_spacing(np.zeros((1, 3)), "median")

    def test_bad_metric(self):
        with pytest.raises(ValueError):
            nearest_neighbour_spacing(np.zeros((2, 3)), "bogus")


# ---------------------------------------------------------------------------
# K. Threading: HMEAGrid coupled branch
# ---------------------------------------------------------------------------

def _vir_params(*, seed=0, n_nodes=26, spread=0.5, segregation=1.0,
                rule="radial", metric="median", s_amp=0.0, M_ext_kg=M_EXT_KG,
                extent=1.0, S=S_DEFAULT_M, relax_steps=1):
    return ExternalNodeParameters(
        M_ext_kg=M_ext_kg, S=S,
        node_mass_seed=seed,
        node_s_amplitude=s_amp,
        node_geometry="virialized",
        vir_n_nodes=n_nodes, vir_extent=extent, vir_mass_rule=rule,
        vir_mass_spread=spread, vir_segregation=segregation, vir_s_metric=metric,
        vir_relax_steps=relax_steps,
    )


class TestHMEAGridThreading:

    def test_node_count_matches_vir_n_nodes(self):
        grid = HMEAGrid(node_params=_vir_params(n_nodes=40, seed=2))
        assert len(grid.nodes) == 40
        assert grid.get_positions().shape == (40, 3)
        assert grid.get_masses().shape == (40,)

    def test_grid_masses_mean_preserving(self):
        grid = HMEAGrid(node_params=_vir_params(n_nodes=26, spread=0.7, seed=2))
        np.testing.assert_allclose(grid.get_masses().mean(), M_EXT_KG, rtol=1e-12)
        np.testing.assert_allclose(grid.get_masses().sum(), 26 * M_EXT_KG, rtol=1e-12)
        assert np.all(grid.get_masses() > 0)

    def test_positions_match_generator_when_no_s_perturbation(self):
        """node_s_amplitude=0 -> grid positions/masses byte-identical to generator."""
        params = _vir_params(n_nodes=26, spread=0.5, seed=2, s_amp=0.0)
        grid = HMEAGrid(node_params=params)
        pos, masses = params.build_virialized()
        np.testing.assert_array_equal(grid.get_positions(), pos)
        np.testing.assert_array_equal(grid.get_masses(), masses)

    def test_masses_use_coupled_not_node_masses(self):
        """Virialized grid masses come from the coupled generator, NOT node_masses().

        node_mass_amplitude is IGNORED for virialized; vir_mass_spread owns the
        distribution. Setting node_mass_amplitude must not change the grid masses.
        """
        p0 = _vir_params(n_nodes=26, spread=0.5, seed=2)
        p_amp = ExternalNodeParameters(
            M_ext_kg=M_EXT_KG, S=S_DEFAULT_M, node_mass_seed=2,
            node_mass_amplitude=0.9,  # would matter for other geometries
            node_geometry="virialized",
            vir_n_nodes=26, vir_mass_spread=0.5, vir_segregation=1.0,
        )
        np.testing.assert_array_equal(
            HMEAGrid(node_params=p0).get_masses(),
            HMEAGrid(node_params=p_amp).get_masses(),
        )

    def test_node_s_amplitude_composes_mean_radial_preserved(self):
        """node_s_amplitude>0 perturbs radii but mean radial scale is preserved and
        masses are unchanged (the s-perturbation acts on positions only).

        Uses the realistic layout (relax_steps=0) so every node has a non-zero
        radius; the force-balanced lattice puts one node AT the origin (r=0), where
        the per-node radial RATIO rs/r0 is undefined, so this mechanics check runs
        on the all-positive-radius realistic mode.
        """
        p0 = _vir_params(n_nodes=26, spread=0.5, seed=13, s_amp=0.0, relax_steps=0)
        ps = _vir_params(n_nodes=26, spread=0.5, seed=13, s_amp=0.4, relax_steps=0)
        g0 = HMEAGrid(node_params=p0)
        gs = HMEAGrid(node_params=ps)
        r0 = _radii(g0.get_positions())
        rs = _radii(gs.get_positions())
        assert not np.array_equal(g0.get_positions(), gs.get_positions())
        np.testing.assert_allclose((rs / r0).mean(), 1.0, rtol=1e-12)
        # Masses are untouched by the position perturbation.
        np.testing.assert_array_equal(g0.get_masses(), gs.get_masses())

    def test_m_ext_zero_builds_and_total_zero(self):
        """M_ext_kg = 0 still builds; all node masses 0 (mean-preserving of 0)."""
        params = ExternalNodeParameters(
            M_ext_kg=0.0, S=S_DEFAULT_M, node_mass_seed=1,
            node_geometry="virialized", vir_n_nodes=26, vir_mass_spread=0.5,
        )
        grid = HMEAGrid(node_params=params)
        assert len(grid.nodes) == 26
        np.testing.assert_allclose(grid.get_masses().sum(), 0.0, atol=0.0)

    @pytest.mark.parametrize("rule", RULES)
    def test_m_ext_zero_gives_zero_tidal_force_eds_invariant(self, rule):
        """M_ext=0 == EdS for virialized: with all node masses 0 the virialized grid
        exerts EXACTLY zero tidal acceleration on any cloud, so the dynamics reduce
        to pure matter (Einstein-de Sitter) just like every other geometry (PF1).

        This asserts the invariant at the force-path level (fast, no full sim): the
        tidal sum is the only channel through which the nodes act, and it vanishes
        identically when M_ext_kg=0, independent of segregation/spread/rule.
        """
        params = ExternalNodeParameters(
            M_ext_kg=0.0, S=S_DEFAULT_M, node_mass_seed=1,
            node_geometry="virialized", vir_n_nodes=26,
            vir_mass_rule=rule, vir_mass_spread=0.5, vir_segregation=1.0,
        )
        grid = HMEAGrid(node_params=params)
        # A small off-centre test cloud (meters).
        cloud = np.array([
            [0.0, 0.0, 0.0],
            [1.0e24, -2.0e24, 3.0e24],
            [-4.0e24, 5.0e24, -6.0e24],
        ], dtype=np.float64)
        for use_numba in (True, False):
            accel = grid.calculate_tidal_acceleration_batch(cloud, use_numba=use_numba)
            assert accel.shape == cloud.shape
            np.testing.assert_array_equal(accel, np.zeros_like(cloud))


# ---------------------------------------------------------------------------
# K2. Particle-realization independence (confound guard, plan S2)
# ---------------------------------------------------------------------------

class TestParticleRealizationIndependence:
    """Building a virialized grid must NOT perturb the particle-cloud realization.

    Mirrors the node_mass_amplitude confound guard
    (test_particle_realization_independent_of_amplitude): the virialized generator
    draws from its own default_rng(node_mass_seed) and must NEVER touch the global
    np.random state. So a cloud sampled before the grid is built (the real ordering:
    cloud first, then HMEAGrid) is byte-identical regardless of vir_* values, and the
    global RNG stream is not advanced by the (massfunc) virialized draws.
    """

    def _cloud_then_grid(self, *, rule, spread, segregation):
        from cosmo.particles import ParticleSystem
        np.random.seed(20240601)
        ps = ParticleSystem(
            n_particles=40, eds_consistent=True, t_start_Gyr=2.9, mass_randomize=0.0,
        )
        cloud_pos = ps.get_positions().copy()
        params = ExternalNodeParameters(
            M_ext_kg=M_EXT_KG, S=S_DEFAULT_M, node_mass_seed=7,
            node_geometry="virialized", vir_n_nodes=26, vir_mass_rule=rule,
            vir_mass_spread=spread, vir_segregation=segregation,
        )
        HMEAGrid(node_params=params)  # build consumes default_rng(seed) only
        post_grid_global = np.random.rand(4)  # global stream after grid build
        return cloud_pos, post_grid_global

    def test_cloud_byte_identical_across_vir_params(self):
        c1, _ = self._cloud_then_grid(rule="massfunc", spread=0.3, segregation=1.0)
        c2, _ = self._cloud_then_grid(rule="massfunc", spread=0.9, segregation=0.4)
        c3, _ = self._cloud_then_grid(rule="radial", spread=0.7, segregation=1.0)
        np.testing.assert_array_equal(c1, c2)
        np.testing.assert_array_equal(c1, c3)

    def test_global_rng_not_advanced_by_grid_build(self):
        """Even the massfunc RNG draws use default_rng(seed), never global np.random,
        so the global stream after a grid build is identical across vir_* values."""
        _, g1 = self._cloud_then_grid(rule="massfunc", spread=0.3, segregation=1.0)
        _, g2 = self._cloud_then_grid(rule="massfunc", spread=0.9, segregation=0.4)
        np.testing.assert_array_equal(g1, g2)


# ---------------------------------------------------------------------------
# L. cube26 opt-in invariant (virialized never the default; cube26 unchanged)
# ---------------------------------------------------------------------------

class TestCube26Unchanged:

    def test_default_geometry_is_cube26(self):
        assert SimulationParameters().node_geometry == "cube26"
        assert ExternalNodeParameters().node_geometry == "cube26"

    def test_cube26_positions_byte_identical(self):
        """Adding the virialized path must not change cube26 grid positions."""
        params = ExternalNodeParameters(M_ext_kg=M_EXT_KG, S=S_DEFAULT_M)
        grid = HMEAGrid(node_params=params)
        expected = build_node_positions("cube26", S_DEFAULT_M)
        np.testing.assert_array_equal(grid.get_positions(), expected)
        np.testing.assert_array_equal(grid.get_masses(), np.full(26, M_EXT_KG))


# ---------------------------------------------------------------------------
# M. SimulationParameters / SweepConfig threading
# ---------------------------------------------------------------------------

class TestSimParamsThreading:

    def test_defaults_exist(self):
        p = SimulationParameters()
        assert p.vir_n_nodes == 26
        assert p.vir_extent == 1.0
        assert p.vir_mass_rule == "radial"
        assert p.vir_mass_spread == 0.0
        assert p.vir_segregation == 1.0
        assert p.vir_s_metric == "median"
        assert p.vir_relax_steps == 1  # DEFAULT = force-balanced
        assert p.vir_extent_couples_nodes is False  # item-10 coupling OFF by default

    def test_external_params_receives_vir_fields(self):
        p = SimulationParameters(
            node_geometry="virialized", vir_n_nodes=40, vir_extent=2.0,
            vir_mass_rule="massfunc", vir_mass_spread=0.7, vir_segregation=0.5,
            vir_s_metric="mean", vir_relax_steps=0, vir_extent_couples_nodes=True,
        )
        ep = p.external_params
        assert ep.node_geometry == "virialized"
        assert ep.vir_n_nodes == 40
        assert ep.vir_extent == 2.0
        assert ep.vir_mass_rule == "massfunc"
        assert ep.vir_mass_spread == 0.7
        assert ep.vir_segregation == 0.5
        assert ep.vir_s_metric == "mean"
        assert ep.vir_relax_steps == 0
        assert ep.vir_extent_couples_nodes is True

    def test_sim_params_build_grid_uses_vir(self):
        p = SimulationParameters(
            M_value=500, S_value=25.0, n_particles=5, seed=1,
            node_geometry="virialized", vir_n_nodes=30, vir_mass_spread=0.6,
        )
        grid = HMEAGrid(node_params=p.external_params)
        assert len(grid.nodes) == 30
        np.testing.assert_allclose(grid.get_masses().mean(), p.M_ext_kg, rtol=1e-12)

    def test_extent_coupling_changes_built_node_count(self):
        """Keyed == RUN: the coupling actually changes the built grid's node count.

        With the coupling ON and a non-default extent, the HMEAGrid built from these
        params must have round(vir_n_nodes * extent^3) nodes, not vir_n_nodes — the
        knob changes the BUILT grid, not just the cache key.
        """
        from cosmo.node_geometry import extent_coupled_n_nodes
        base = 32
        p = SimulationParameters(
            M_value=500, S_value=25.0, n_particles=5, seed=1,
            node_geometry="virialized", vir_n_nodes=base, vir_extent=2.0,
            vir_extent_couples_nodes=True,
        )
        grid = HMEAGrid(node_params=p.external_params)
        assert len(grid.nodes) == extent_coupled_n_nodes(base, 2.0) == 256
        # Default OFF: count stays at the base regardless of extent.
        p_off = SimulationParameters(
            M_value=500, S_value=25.0, n_particles=5, seed=1,
            node_geometry="virialized", vir_n_nodes=base, vir_extent=2.0,
        )
        assert len(HMEAGrid(node_params=p_off.external_params).nodes) == base


class TestSweepConfigFields:

    def test_defaults(self):
        from cosmo.parameter_sweep import SweepConfig
        cfg = SweepConfig()
        assert cfg.vir_n_nodes == 26
        assert cfg.vir_extent == 1.0
        assert cfg.vir_mass_rule == "radial"
        assert cfg.vir_mass_spread == 0.0
        assert cfg.vir_segregation == 1.0
        assert cfg.vir_s_metric == "median"
        assert cfg.vir_relax_steps == 1  # DEFAULT = force-balanced
        assert cfg.vir_extent_couples_nodes is False  # item-10 coupling OFF by default

    def test_custom(self):
        from cosmo.parameter_sweep import SweepConfig
        cfg = SweepConfig(vir_n_nodes=12, vir_mass_rule="massfunc", vir_mass_spread=0.4)
        assert cfg.vir_n_nodes == 12
        assert cfg.vir_mass_rule == "massfunc"
        assert cfg.vir_mass_spread == 0.4


# ---------------------------------------------------------------------------
# N. Cache slug behaviour
# ---------------------------------------------------------------------------

class TestCacheSlug:

    def _key(self, **kwargs):
        from cosmo.parameter_sweep import SweepConfig, build_cache_name
        cfg = SweepConfig(quick_search=True, objective="pantheon", **kwargs)
        return build_cache_name(cfg, M_factor=800, S_val=25, centerM=1, seeds=[42])

    def test_virialized_key_has_vir_slugs(self):
        key = self._key(node_geometry="virialized")
        assert "virializedgeo" in key
        for slug in ("vn", "vx", "vr", "vsp", "vsg", "vsm", "vrx"):
            assert slug in key, f"missing {slug!r} sub-slug for virialized"

    def test_virialized_distinct_from_cube26(self):
        kv = self._key(node_geometry="virialized")
        kc = self._key()  # cube26 default
        assert kv != kc

    def test_distinct_across_vir_values(self):
        k1 = self._key(node_geometry="virialized", vir_mass_spread=0.3)
        k2 = self._key(node_geometry="virialized", vir_mass_spread=0.7)
        k3 = self._key(node_geometry="virialized", vir_mass_rule="massfunc")
        k4 = self._key(node_geometry="virialized", vir_segregation=0.0)
        k5 = self._key(node_geometry="virialized", vir_extent=2.0)
        k6 = self._key(node_geometry="virialized", vir_n_nodes=12)
        k7 = self._key(node_geometry="virialized", vir_s_metric="mean")
        k8 = self._key(node_geometry="virialized", vir_relax_steps=0)
        k9 = self._key(node_geometry="virialized", vir_extent_couples_nodes=True)
        keys = [k1, k2, k3, k4, k5, k6, k7, k8, k9]
        assert len(set(keys)) == len(keys), "vir_* values must yield distinct cache keys"

    def test_extent_coupling_slug_only_when_on(self):
        """The extent-coupling slug appears ONLY when the flag is True (keyed==run).

        Default virialized runs (flag False) must keep their existing cache key, so
        the slug is absent; a coupled run gets a distinct key.
        """
        k_off = self._key(node_geometry="virialized")
        k_on = self._key(node_geometry="virialized", vir_extent_couples_nodes=True)
        assert "vxcouple" not in k_off, "default virialized key must NOT bump"
        assert "vxcouple" in k_on
        assert k_off != k_on

    def test_non_virialized_keys_carry_no_vir_slug(self):
        """Regression: cube26/cube_dense/fcc/bcc keys must NOT pick up vir slugs."""
        for geo in ("cube26", "cube_dense", "fcc", "bcc"):
            kwargs = {} if geo == "cube26" else {"node_geometry": geo}
            # Even requesting the coupling on a non-virialized geometry must not slug.
            key = self._key(vir_extent_couples_nodes=True, **kwargs)
            for slug in ("vn", "vx", "vr", "vsp", "vsg", "vsm", "vrx", "vxcouple"):
                # Guard against substring collisions by checking the suffix tokens.
                assert not any(part.endswith(slug) for part in key.split("_")), \
                    f"{geo}: unexpected vir sub-slug {slug!r} in key {key!r}"

    # --- Node softening (Section 4) cache slug ---

    def test_node_softening_default_has_no_slug(self):
        """node_softening_gpc=0.0 (default) => NO nsoft slug (byte-identical key)."""
        key = self._key()  # default node_softening_gpc == 0.0
        assert "nsoft" not in key

    def test_node_softening_nonzero_has_slug(self):
        """node_softening_gpc != 0 => the value appears as a {value}nsoft slug."""
        key = self._key(node_softening_gpc=1.0)
        assert "1.0nsoft" in key

    def test_node_softening_distinct_keys(self):
        """Different node_softening_gpc values yield distinct cache keys; 0.0 == default."""
        k0 = self._key()  # default 0.0 (no slug)
        k_half = self._key(node_softening_gpc=0.5)
        k_one = self._key(node_softening_gpc=1.0)
        assert k0 != k_half != k_one and k0 != k_one
        # The explicit 0.0 must equal the default (no slug either way).
        assert self._key(node_softening_gpc=0.0) == k0
