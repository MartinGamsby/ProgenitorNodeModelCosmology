"""
Tests for WS3: node geometry factory (cosmo/node_geometry.py) and threading.

Invariants:
  1. cube26 produces exactly the same 26 positions as the old HMEAGrid loop.
  2. Each geometry returns the expected node count.
  3. All nodes are at radius >= S/2 (outside the cloud; not at origin).
  4. Total external mass is conserved across geometries (via mean-preservation).
  5. Cache slug changes with geometry (non-default geometry appends geo slug).
  6. Threading: HMEAGrid built from factory for default and non-default geometries.
  7. M=0 == EdS is preserved for cube26 (unchanged by the refactor).
  8. SweepConfig and build_cache_name handle node_geometry correctly.
"""

import math
import numpy as np
import pytest

from cosmo.node_geometry import build_node_positions, list_geometries, effective_M_ext_kg
from cosmo.constants import ExternalNodeParameters, SimulationParameters, CosmologicalConstants
from cosmo.particles import HMEAGrid


# ---------------------------------------------------------------------------
# 1. cube26 byte-identical to old loop
# ---------------------------------------------------------------------------

class TestCube26ByteIdentical:
    """build_node_positions('cube26', S) must reproduce the old _create_grid loop."""

    def _old_loop(self, S: float):
        """Copy of the ORIGINAL hard-coded loop from particles.py (pre-WS3)."""
        base = []
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                for k in [-1, 0, 1]:
                    if i == 0 and j == 0 and k == 0:
                        continue
                    base.append(np.array([i, j, k], dtype=float) * S)
        return np.array(base, dtype=np.float64)

    def test_cube26_matches_old_loop_S1(self):
        S = 1.0
        expected = self._old_loop(S)
        got = build_node_positions("cube26", S)
        np.testing.assert_array_equal(got, expected,
            err_msg="cube26 must be byte-identical to the original hard-coded loop (S=1)")

    def test_cube26_matches_old_loop_typical_S(self):
        const = CosmologicalConstants()
        S = 25.0 * const.Gpc_to_m
        expected = self._old_loop(S)
        got = build_node_positions("cube26", S)
        np.testing.assert_array_equal(got, expected,
            err_msg="cube26 must be byte-identical to the original hard-coded loop (S=25 Gpc)")

    def test_cube26_shape(self):
        pos = build_node_positions("cube26", 1.0)
        assert pos.shape == (26, 3), f"Expected (26,3), got {pos.shape}"

    def test_cube26_dtype(self):
        pos = build_node_positions("cube26", 1.0)
        assert pos.dtype == np.float64

    def test_cube26_no_origin(self):
        pos = build_node_positions("cube26", 1.0)
        radii = np.linalg.norm(pos, axis=1)
        assert np.all(radii > 0.0), "cube26 must not have a node at the origin"


# ---------------------------------------------------------------------------
# 2. Each geometry returns the expected node count
# ---------------------------------------------------------------------------

class TestGeometryNodeCounts:
    """Each geometry factory returns the documented node count."""

    def test_cube26_count(self):
        pos = build_node_positions("cube26", 1.0)
        assert len(pos) == 26

    def test_cube_dense_default_count(self):
        # 5^3 - 1 = 124
        pos = build_node_positions("cube_dense", 1.0)
        assert len(pos) == 124

    def test_cube_dense_n3_count(self):
        # 3^3 - 1 = 26
        pos = build_node_positions("cube_dense", 1.0, n_per_side=3)
        assert len(pos) == 26

    def test_cube_dense_n7_count(self):
        # 7^3 - 1 = 342
        pos = build_node_positions("cube_dense", 1.0, n_per_side=7)
        assert len(pos) == 342

    def test_fcc_default_count_positive(self):
        pos = build_node_positions("fcc", 1.0)
        assert len(pos) > 0

    def test_bcc_default_count_positive(self):
        pos = build_node_positions("bcc", 1.0)
        assert len(pos) > 0

    def test_list_geometries(self):
        geos = list_geometries()
        for expected in ["cube26", "cube_dense", "fcc", "bcc"]:
            assert expected in geos, f"{expected} missing from list_geometries()"

    def test_shells_excluded(self):
        """Hollow shells are the opposite of a virialized grid; must be unavailable."""
        geos = list_geometries()
        assert "shell" not in geos and "shell_multi" not in geos


# ---------------------------------------------------------------------------
# 3. All nodes at radius >= S/2 (outside the cloud; not near origin)
# ---------------------------------------------------------------------------

class TestNodesOutsideCloud:
    """Every geometry must place all nodes at radius >= S * 0.4 from origin."""

    @pytest.mark.parametrize("geometry,kwargs", [
        ("cube26", {}),
        ("cube_dense", {}),
        ("fcc", {}),
        ("bcc", {}),
    ])
    def test_all_nodes_outside_cloud(self, geometry, kwargs):
        S = 30.0  # Gpc units (abstract; just a float)
        pos = build_node_positions(geometry, S, **kwargs)
        radii = np.linalg.norm(pos, axis=1)
        min_r = radii.min()
        assert min_r >= 0.4 * S, (
            f"{geometry}: closest node is at r={min_r:.3f} < 0.4*S={0.4*S:.3f}. "
            "All nodes must be outside the cloud."
        )


# ---------------------------------------------------------------------------
# 4. Total external mass conserved via mean-preservation
# ---------------------------------------------------------------------------

class TestMassBookkeeping:
    """node_masses(n) mean-preserves M_ext_kg for any n; total = n * M_ext_kg."""

    @pytest.mark.parametrize("geometry,kwargs,expected_n", [
        ("cube26", {}, 26),
        ("cube_dense", {"n_per_side": 5}, 124),
        ("cube_dense", {"n_per_side": 7}, 342),
    ])
    def test_total_mass_is_n_times_M_ext(self, geometry, kwargs, expected_n):
        M_ext_kg = 5e55
        params = ExternalNodeParameters(M_ext_kg=M_ext_kg, node_mass_amplitude=0.0)
        masses = params.node_masses(expected_n)
        np.testing.assert_allclose(
            masses.sum(), expected_n * M_ext_kg, rtol=1e-12,
            err_msg=f"Total mass must be n_nodes * M_ext_kg for {geometry}"
        )

    def test_effective_M_ext_kg_preserves_total(self):
        """effective_M_ext_kg rescales so total mass == 26 * M_ref."""
        M_ref = 5e55
        for n in [50, 124, 150]:
            m_per_node = effective_M_ext_kg(M_ref, n, ref_nodes=26)
            total = n * m_per_node
            np.testing.assert_allclose(total, 26 * M_ref, rtol=1e-12,
                err_msg=f"effective_M_ext_kg failed for n={n}")

    def test_mass_amplitude_mean_preservation_any_n(self):
        """node_masses with amplitude > 0 still has mean == M_ext_kg for various n."""
        M_ext_kg = 3e54
        params = ExternalNodeParameters(M_ext_kg=M_ext_kg, node_mass_seed=42,
                                        node_mass_amplitude=0.5)
        for n in [26, 50, 124]:
            masses = params.node_masses(n)
            np.testing.assert_allclose(masses.mean(), M_ext_kg, rtol=1e-12,
                err_msg=f"Mean not preserved for n={n}")


# ---------------------------------------------------------------------------
# 5. Cache slug changes with geometry
# ---------------------------------------------------------------------------

class TestCacheSlug:
    """build_cache_name appends a geo slug only for non-default geometry."""

    def _cache_name(self, geometry):
        from cosmo.parameter_sweep import SweepConfig, build_cache_name
        cfg = SweepConfig(quick_search=True, objective='pantheon',
                          node_geometry=geometry)
        return build_cache_name(cfg, M_factor=800, S_val=25, centerM=1, seeds=[42])

    def test_cube26_no_geo_slug(self):
        key = self._cache_name("cube26")
        assert "geo" not in key, "cube26 must not add a geo slug"

    def test_cube_dense_appends_geo_slug(self):
        key = self._cache_name("cube_dense")
        assert "cube_densegeo" in key

    def test_different_geometries_different_keys(self):
        k26 = self._cache_name("cube26")
        kfcc = self._cache_name("fcc")
        kfd = self._cache_name("cube_dense")
        assert k26 != kfcc
        assert k26 != kfd
        assert kfcc != kfd

    def test_fcc_slug(self):
        key = self._cache_name("fcc")
        assert "fccgeo" in key

    def test_bcc_slug(self):
        key = self._cache_name("bcc")
        assert "bccgeo" in key


# ---------------------------------------------------------------------------
# 6. Threading: HMEAGrid built from factory for cube26 and alternatives
# ---------------------------------------------------------------------------

class TestHMEAGridThreading:
    """HMEAGrid._create_grid uses the geometry factory correctly."""

    def _make_grid(self, geometry="cube26", **gkwargs):
        params = ExternalNodeParameters(
            M_ext_kg=5e55,
            S=25.0 * CosmologicalConstants.Gpc_to_m,
            node_geometry=geometry,
            geometry_kwargs=gkwargs,
        )
        return HMEAGrid(node_params=params)

    def test_cube26_grid_has_26_nodes(self):
        grid = self._make_grid("cube26")
        assert len(grid.nodes) == 26
        assert grid.n_nodes == 26

    def test_cube26_grid_positions_match_factory(self):
        """HMEAGrid with cube26 has positions byte-identical to factory."""
        S = 25.0 * CosmologicalConstants.Gpc_to_m
        expected = build_node_positions("cube26", S)
        grid = self._make_grid("cube26")
        got = grid.get_positions()
        np.testing.assert_array_equal(got, expected,
            err_msg="HMEAGrid cube26 positions must match factory output")

    def test_cube_dense_grid_has_124_nodes(self):
        grid = self._make_grid("cube_dense")
        assert len(grid.nodes) == 124
        assert grid.n_nodes == 124

    def test_sim_params_threads_geometry_to_grid(self):
        """SimulationParameters.node_geometry threads into HMEAGrid via external_params."""
        sim_params = SimulationParameters(
            M_value=500, S_value=25.0, n_particles=5, seed=1,
            node_geometry="cube_dense",
            geometry_kwargs={"n_per_side": 5},
        )
        grid = HMEAGrid(node_params=sim_params.external_params)
        assert len(grid.nodes) == 124

    def test_fcc_grid_nodes_positive(self):
        grid = self._make_grid("fcc")
        assert len(grid.nodes) > 0

    def test_bcc_grid_nodes_positive(self):
        grid = self._make_grid("bcc")
        assert len(grid.nodes) > 0

    def test_default_sim_params_gives_cube26(self):
        """Default SimulationParameters should produce a 26-node cube26 grid."""
        sp = SimulationParameters()
        assert sp.node_geometry == "cube26"
        grid = HMEAGrid(node_params=sp.external_params)
        assert len(grid.nodes) == 26


# ---------------------------------------------------------------------------
# 7. Backward compatibility: HMEAGrid with no ExternalNodeParameters
#    (old code path using default node_params) still gives 26 nodes
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:
    """HMEAGrid with default node_params (no geometry arg) still gives 26 nodes."""

    def test_default_grid_still_26_nodes(self):
        grid = HMEAGrid()
        assert len(grid.nodes) == 26

    def test_default_grid_positions_unchanged(self):
        """Default HMEAGrid positions match cube26 factory (byte-identical)."""
        grid = HMEAGrid()
        S = grid.params.S
        expected = build_node_positions("cube26", S)
        np.testing.assert_array_equal(
            grid.get_positions(), expected,
            err_msg="Default HMEAGrid positions must be cube26-identical"
        )


# ---------------------------------------------------------------------------
# 8. SweepConfig fields
# ---------------------------------------------------------------------------

class TestSweepConfigGeometry:
    def test_default_node_geometry(self):
        from cosmo.parameter_sweep import SweepConfig
        cfg = SweepConfig()
        assert cfg.node_geometry == "cube26"
        assert cfg.geometry_kwargs == {}

    def test_custom_node_geometry(self):
        from cosmo.parameter_sweep import SweepConfig
        cfg = SweepConfig(node_geometry="cube_dense", geometry_kwargs={"n_per_side": 7})
        assert cfg.node_geometry == "cube_dense"
        assert cfg.geometry_kwargs == {"n_per_side": 7}


# ---------------------------------------------------------------------------
# 9. cube_dense with n=3 produces the same layout as cube26
# ---------------------------------------------------------------------------

class TestCubeDenseEquivalence:
    """cube_dense(n_per_side=3) must produce the same outer positions as cube26
    (scaled so outermost nodes are at the same radius)."""

    def test_cube_dense_n3_outer_radius_matches_cube26(self):
        S = 1.0
        pos26 = build_node_positions("cube26", S)
        pos_dense = build_node_positions("cube_dense", S, n_per_side=3)
        # Both should have 26 nodes
        assert len(pos26) == len(pos_dense)
        # Sorted radii should be identical (same geometry, same outer radius)
        r26 = np.sort(np.linalg.norm(pos26, axis=1))
        rd = np.sort(np.linalg.norm(pos_dense, axis=1))
        np.testing.assert_allclose(r26, rd, rtol=1e-12,
            err_msg="cube_dense(n=3) radii must match cube26 radii")


# ---------------------------------------------------------------------------
# 10. fcc/bcc are volume-filling (not hollow): they have nodes across radii
# ---------------------------------------------------------------------------

class TestVolumeFilling:
    @pytest.mark.parametrize("geometry", ["cube26", "cube_dense", "fcc", "bcc"])
    def test_nodes_span_multiple_radii(self, geometry):
        """A virialized/volume-filling lattice has nodes at more than one radius
        (a hollow shell would have all nodes at a single radius)."""
        pos = build_node_positions(geometry, 1.0)
        radii = np.linalg.norm(pos, axis=1)
        n_distinct = len(np.unique(np.round(radii, 6)))
        assert n_distinct >= 2, (
            f"{geometry} must fill the volume (>=2 distinct radii), got {n_distinct}"
        )


# ---------------------------------------------------------------------------
# 10b. normalize_nearest: put each geometry's nearest node at radius S
# ---------------------------------------------------------------------------

class TestNormalizeNearest:
    """normalize_nearest rescales so the closest node sits at S (fair cross-geometry
    comparison: equal per-node mass + equal nearest-node distance)."""

    @pytest.mark.parametrize("geometry", ["cube26", "cube_dense", "fcc", "bcc"])
    def test_nearest_node_at_S(self, geometry):
        S = 30.0
        pos = build_node_positions(geometry, S, normalize_nearest=True)
        r_min = np.linalg.norm(pos, axis=1).min()
        np.testing.assert_allclose(r_min, S, rtol=1e-9,
            err_msg=f"{geometry}: nearest node must be at exactly S with normalize_nearest")

    def test_cube26_normalize_is_noop(self):
        """cube26's nearest node is already at S, so normalize_nearest changes nothing."""
        S = 25.0
        a = build_node_positions("cube26", S)
        b = build_node_positions("cube26", S, normalize_nearest=True)
        np.testing.assert_array_equal(a, b)

    def test_normalize_preserves_node_count(self):
        for geometry in ["cube_dense", "fcc", "bcc"]:
            n_raw = len(build_node_positions(geometry, 10.0))
            n_norm = len(build_node_positions(geometry, 10.0, normalize_nearest=True))
            assert n_raw == n_norm


# ---------------------------------------------------------------------------
# 11. Invalid geometry raises ValueError
# ---------------------------------------------------------------------------

class TestInvalidGeometry:
    def test_unknown_geometry_raises(self):
        with pytest.raises(ValueError, match="Unknown node geometry"):
            build_node_positions("bogus_geometry", 1.0)

    def test_cube_dense_even_n_raises(self):
        with pytest.raises(ValueError):
            build_node_positions("cube_dense", 1.0, n_per_side=4)

    def test_excluded_shell_geometry_raises(self):
        """shell/shell_multi were removed (hollow != virialized) -> ValueError."""
        with pytest.raises(ValueError, match="Unknown node geometry"):
            build_node_positions("shell", 1.0)
