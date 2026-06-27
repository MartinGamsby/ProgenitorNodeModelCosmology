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
        """vir_segregation == 0 -> mass<->radius correlation ~ 0 (decoupled)."""
        pos, masses = build_virialized_grid(
            S_DEFAULT_M, n_nodes=54, M_ext_kg=M_EXT_KG, vir_mass_rule=rule,
            vir_mass_spread=0.8, vir_segregation=0.0, seed=11,
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

        The NN-spacing rescale fixes the spacing to S (a global factor that cancels
        in any radius RATIO), so the scale-invariant signature of vir_extent is the
        radial spread r_max / r_min: with radii in [0.5S, (0.5+extent)S] this ratio
        is 1 + 2*extent, growing strongly and predictably with extent.
        """
        def span(extent):
            pos, _ = build_virialized_grid(
                S_DEFAULT_M, n_nodes=40, M_ext_kg=M_EXT_KG, vir_mass_rule="radial",
                vir_mass_spread=0.0, vir_extent=extent, seed=1,
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
        """Targeting median vs mean produces different absolute layouts in general."""
        pos_med, _ = build_virialized_grid(
            S_DEFAULT_M, n_nodes=54, M_ext_kg=M_EXT_KG, vir_mass_rule="radial",
            vir_mass_spread=0.4, vir_s_metric="median", seed=9,
        )
        pos_mean, _ = build_virialized_grid(
            S_DEFAULT_M, n_nodes=54, M_ext_kg=M_EXT_KG, vir_mass_rule="radial",
            vir_mass_spread=0.4, vir_s_metric="mean", seed=9,
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
                extent=1.0, S=S_DEFAULT_M):
    return ExternalNodeParameters(
        M_ext_kg=M_ext_kg, S=S,
        node_mass_seed=seed,
        node_s_amplitude=s_amp,
        node_geometry="virialized",
        vir_n_nodes=n_nodes, vir_extent=extent, vir_mass_rule=rule,
        vir_mass_spread=spread, vir_segregation=segregation, vir_s_metric=metric,
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
        masses are unchanged (the s-perturbation acts on positions only)."""
        p0 = _vir_params(n_nodes=26, spread=0.5, seed=13, s_amp=0.0)
        ps = _vir_params(n_nodes=26, spread=0.5, seed=13, s_amp=0.4)
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

    def test_external_params_receives_vir_fields(self):
        p = SimulationParameters(
            node_geometry="virialized", vir_n_nodes=40, vir_extent=2.0,
            vir_mass_rule="massfunc", vir_mass_spread=0.7, vir_segregation=0.5,
            vir_s_metric="mean",
        )
        ep = p.external_params
        assert ep.node_geometry == "virialized"
        assert ep.vir_n_nodes == 40
        assert ep.vir_extent == 2.0
        assert ep.vir_mass_rule == "massfunc"
        assert ep.vir_mass_spread == 0.7
        assert ep.vir_segregation == 0.5
        assert ep.vir_s_metric == "mean"

    def test_sim_params_build_grid_uses_vir(self):
        p = SimulationParameters(
            M_value=500, S_value=25.0, n_particles=5, seed=1,
            node_geometry="virialized", vir_n_nodes=30, vir_mass_spread=0.6,
        )
        grid = HMEAGrid(node_params=p.external_params)
        assert len(grid.nodes) == 30
        np.testing.assert_allclose(grid.get_masses().mean(), p.M_ext_kg, rtol=1e-12)


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
        for slug in ("vn", "vx", "vr", "vsp", "vsg", "vsm"):
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
        keys = [k1, k2, k3, k4, k5, k6, k7]
        assert len(set(keys)) == len(keys), "vir_* values must yield distinct cache keys"

    def test_non_virialized_keys_carry_no_vir_slug(self):
        """Regression: cube26/cube_dense/fcc/bcc keys must NOT pick up vir slugs."""
        for geo in ("cube26", "cube_dense", "fcc", "bcc"):
            kwargs = {} if geo == "cube26" else {"node_geometry": geo}
            key = self._key(**kwargs)
            for slug in ("vn", "vx", "vr", "vsp", "vsg", "vsm"):
                # Guard against substring collisions by checking the suffix tokens.
                assert not any(part.endswith(slug) for part in key.split("_")), \
                    f"{geo}: unexpected vir sub-slug {slug!r} in key {key!r}"
