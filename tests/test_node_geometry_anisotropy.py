"""
Geometry x amplitude unit coverage for HMEA node POSITIONS + MASSES.

The existing tests (tests/test_node_masses.py, tests/test_node_s_amplitude.py) are
geometry-BLIND: they only exercise the DEFAULT cube26 (n=26) layout. This file
generalizes the node-state invariants ACROSS ALL geometries
(cube26, cube_dense, fcc, bcc), with the geometry's ACTUAL node count N, and
TIGHTLY tests the node state AFTER the amplitude parameters are applied
(perturbed masses + perturbed positions), not just the unperturbed placement.

Invariants proven here (for EACH geometry, with its own N):
  - POSITIONS (factory): (N,3) float64, N == len(build_node_positions(...)) > 0,
    every node at radius > 0 (none at origin).
  - MASSES (node_masses, AFTER perturbation):
      * amplitude==0 -> exactly np.full(N, M_ext_kg) (byte-identical uniform);
      * amplitude>0  -> mean == M_ext_kg (rtol 1e-12, MEAN-PRESERVING for ANY N),
        all > 0, std > 0 (really spread);
      * total external mass == N * M_ext_kg exactly for amplitude 0 AND > 0;
      * deterministic per (geometry, seed, amplitude); different seed differs;
      * independent of the global numpy RNG.
  - node_mass_amplitude THROUGH HMEAGrid: get_masses() length == N, equals
    node_masses(N) exactly; mean-preserving; sum == N * M_ext_kg.
  - node_s_amplitude (positions, AFTER perturbation):
      * s_amp==0 -> grid positions byte-identical to build_node_positions(...);
      * s_amp>0  -> positions change; per-node UNIT direction unchanged (ray
        preserved generically); mean(r_perturbed/r_unperturbed) == 1.0; all r>0;
        deterministic per seed; different seed differs.
  - Cross-knob independence: enabling node_s_amplitude does NOT change node_masses
    (separate default_rng draws).
  - effective_M_ext_kg parity: total N * effective == 26 * M_ref for each N.

These are tests only; NO production code is touched. cube26/cube_dense/fcc/bcc
node POSITIONS from build_node_positions are unchanged by this section.
M=0==EdS is NOT re-tested here (covered elsewhere).
"""
import numpy as np
import pytest

from cosmo.node_geometry import build_node_positions, effective_M_ext_kg
from cosmo.constants import ExternalNodeParameters, CosmologicalConstants
from cosmo.particles import HMEAGrid


const = CosmologicalConstants()
M_EXT_KG = 5e55
S_DEFAULT_M = 30.0 * const.Gpc_to_m

# Geometry axis: name + the kwargs forwarded to the factory (use defaults).
GEOMETRIES = ["cube26", "cube_dense", "fcc", "bcc"]
GEOMETRY_KWARGS = {
    "cube26": {},
    "cube_dense": {},
    "fcc": {},
    "bcc": {},
}


def _n_nodes(geometry: str) -> int:
    """Actual node count for a geometry (queried, never hardcoded for fcc/bcc)."""
    return len(build_node_positions(geometry, S_DEFAULT_M, **GEOMETRY_KWARGS[geometry]))


def _params(geometry, *, seed=0, nmamp=0.0, nsamp=0.0, M_ext_kg=M_EXT_KG, S=S_DEFAULT_M):
    return ExternalNodeParameters(
        M_ext_kg=M_ext_kg, S=S,
        node_mass_seed=seed,
        node_mass_amplitude=nmamp,
        node_s_amplitude=nsamp,
        node_geometry=geometry,
        geometry_kwargs=GEOMETRY_KWARGS[geometry],
    )


# ---------------------------------------------------------------------------
# A. Node POSITIONS per geometry (factory sanity)
# ---------------------------------------------------------------------------

class TestPositionsFactory:
    """Light per-geometry sanity on the unperturbed factory positions.

    (Counts/byte-identity already covered in tests/test_node_geometry.py — this
    is just the float64/(N,3)/no-origin contract that the amplitude tests rely on.)
    """

    @pytest.mark.parametrize("geometry", GEOMETRIES)
    def test_shape_dtype_and_count(self, geometry):
        pos = build_node_positions(geometry, S_DEFAULT_M, **GEOMETRY_KWARGS[geometry])
        n = _n_nodes(geometry)
        assert n > 0
        assert pos.shape == (n, 3), f"{geometry}: expected ({n},3), got {pos.shape}"
        assert pos.dtype == np.float64, f"{geometry}: dtype {pos.dtype} != float64"

    @pytest.mark.parametrize("geometry", GEOMETRIES)
    def test_no_node_at_origin(self, geometry):
        pos = build_node_positions(geometry, S_DEFAULT_M, **GEOMETRY_KWARGS[geometry])
        radii = np.linalg.norm(pos, axis=1)
        assert np.all(radii > 0.0), f"{geometry}: a node sits at the origin"


# ---------------------------------------------------------------------------
# B. Node MASSES per geometry (mean-preserving, AFTER perturbation)
# ---------------------------------------------------------------------------

class TestNodeMassesPerGeometry:
    """node_masses(N) for each geometry's ACTUAL N — the key generalization is
    that mean-preservation holds for ANY N, not just 26."""

    @pytest.mark.parametrize("geometry", GEOMETRIES)
    def test_amplitude_zero_is_exact_uniform(self, geometry):
        """amplitude=0 -> exactly np.full(N, M_ext_kg) (byte-identical uniform)."""
        n = _n_nodes(geometry)
        masses = _params(geometry, seed=12345, nmamp=0.0).node_masses(n)
        assert masses.shape == (n,)
        np.testing.assert_array_equal(masses, np.full(n, M_EXT_KG))

    @pytest.mark.parametrize("geometry", GEOMETRIES)
    @pytest.mark.parametrize("amplitude", [0.3, 0.6, 1.0])
    def test_mean_preservation_for_any_N(self, geometry, amplitude):
        """amplitude>0 -> mean == M_ext_kg (rtol 1e-12) for the geometry's actual N."""
        n = _n_nodes(geometry)
        masses = _params(geometry, seed=42, nmamp=amplitude).node_masses(n)
        np.testing.assert_allclose(
            masses.mean(), M_EXT_KG, rtol=1e-12,
            err_msg=f"{geometry} (N={n}): mean not preserved at amplitude={amplitude}")

    @pytest.mark.parametrize("geometry", GEOMETRIES)
    @pytest.mark.parametrize("amplitude", [0.3, 0.6, 1.0])
    def test_strict_positivity_and_spread(self, geometry, amplitude):
        """amplitude>0 -> all masses > 0 and genuinely spread (std > 0)."""
        n = _n_nodes(geometry)
        masses = _params(geometry, seed=42, nmamp=amplitude).node_masses(n)
        assert np.all(masses > 0), f"{geometry}: non-positive mass at amplitude={amplitude}"
        assert masses.std() > 0, f"{geometry}: amplitude={amplitude} did not spread masses"

    @pytest.mark.parametrize("geometry", GEOMETRIES)
    @pytest.mark.parametrize("amplitude", [0.0, 0.6])
    def test_total_mass_is_N_times_M_ext(self, geometry, amplitude):
        """Total external mass == N * M_ext_kg exactly for amplitude 0 AND > 0."""
        n = _n_nodes(geometry)
        masses = _params(geometry, seed=7, nmamp=amplitude).node_masses(n)
        np.testing.assert_allclose(
            masses.sum(), n * M_EXT_KG, rtol=1e-12,
            err_msg=f"{geometry} (N={n}): total != N*M_ext at amplitude={amplitude}")

    @pytest.mark.parametrize("geometry", GEOMETRIES)
    def test_determinism_same_seed(self, geometry):
        n = _n_nodes(geometry)
        m1 = _params(geometry, seed=7, nmamp=0.6).node_masses(n)
        m2 = _params(geometry, seed=7, nmamp=0.6).node_masses(n)
        np.testing.assert_array_equal(m1, m2)

    @pytest.mark.parametrize("geometry", GEOMETRIES)
    def test_different_seed_different_masses(self, geometry):
        n = _n_nodes(geometry)
        m1 = _params(geometry, seed=1, nmamp=0.6).node_masses(n)
        m2 = _params(geometry, seed=2, nmamp=0.6).node_masses(n)
        assert not np.array_equal(m1, m2), \
            f"{geometry}: different seeds produced identical mass vectors"

    @pytest.mark.parametrize("geometry", GEOMETRIES)
    def test_independent_of_global_rng(self, geometry):
        n = _n_nodes(geometry)
        p = _params(geometry, seed=42, nmamp=0.6)
        np.random.seed(999); _ = np.random.rand(1000)
        before = p.node_masses(n)
        np.random.seed(0); _ = np.random.rand(5000)
        after = p.node_masses(n)
        np.testing.assert_array_equal(before, after,
            err_msg=f"{geometry}: node_masses depends on global numpy RNG")


# ---------------------------------------------------------------------------
# C. node_mass_amplitude THROUGH HMEAGrid per geometry (AFTER-amplitude state)
# ---------------------------------------------------------------------------

class TestHMEAGridMassesPerGeometry:

    def _grid(self, geometry, *, seed=0, nmamp=0.0):
        return HMEAGrid(node_params=_params(geometry, seed=seed, nmamp=nmamp))

    @pytest.mark.parametrize("geometry", GEOMETRIES)
    def test_grid_mass_count_matches_geometry_N(self, geometry):
        n = _n_nodes(geometry)
        grid = self._grid(geometry, seed=42, nmamp=0.6)
        assert grid.get_masses().shape == (n,)
        assert len(grid.nodes) == n

    @pytest.mark.parametrize("geometry", GEOMETRIES)
    def test_grid_masses_match_node_masses_method(self, geometry):
        """get_masses() == params.node_masses(N) exactly."""
        n = _n_nodes(geometry)
        params = _params(geometry, seed=42, nmamp=0.6)
        grid = HMEAGrid(node_params=params)
        np.testing.assert_array_equal(grid.get_masses(), params.node_masses(n))

    @pytest.mark.parametrize("geometry", GEOMETRIES)
    def test_grid_amplitude_zero_all_uniform(self, geometry):
        n = _n_nodes(geometry)
        grid = self._grid(geometry, seed=42, nmamp=0.0)
        np.testing.assert_array_equal(grid.get_masses(), np.full(n, M_EXT_KG))

    @pytest.mark.parametrize("geometry", GEOMETRIES)
    def test_grid_amplitude_nonzero_mean_preserving_and_total(self, geometry):
        n = _n_nodes(geometry)
        grid = self._grid(geometry, seed=13, nmamp=0.6)
        masses = grid.get_masses()
        np.testing.assert_allclose(masses.mean(), M_EXT_KG, rtol=1e-12)
        np.testing.assert_allclose(masses.sum(), n * M_EXT_KG, rtol=1e-12)
        assert np.all(masses > 0)


# ---------------------------------------------------------------------------
# D. node_s_amplitude (position perturbation) per geometry (AFTER-amplitude)
# ---------------------------------------------------------------------------

class TestHMEAGridPositionsPerGeometry:

    def _grid(self, geometry, *, seed=0, nsamp=0.0):
        return HMEAGrid(node_params=_params(geometry, seed=seed, nsamp=nsamp))

    @pytest.mark.parametrize("geometry", GEOMETRIES)
    def test_s_amplitude_zero_byte_identical_to_factory(self, geometry):
        """s_amp=0 -> grid positions byte-identical to build_node_positions(...)."""
        grid = self._grid(geometry, seed=42, nsamp=0.0)
        expected = build_node_positions(geometry, S_DEFAULT_M, **GEOMETRY_KWARGS[geometry])
        np.testing.assert_array_equal(grid.get_positions(), expected)

    @pytest.mark.parametrize("geometry", GEOMETRIES)
    @pytest.mark.parametrize("nsamp", [0.3, 0.6])
    def test_s_amplitude_changes_positions(self, geometry, nsamp):
        g0 = self._grid(geometry, seed=42, nsamp=0.0)
        gp = self._grid(geometry, seed=42, nsamp=nsamp)
        assert not np.array_equal(g0.get_positions(), gp.get_positions()), \
            f"{geometry}: node_s_amplitude={nsamp} did not change positions"

    @pytest.mark.parametrize("geometry", GEOMETRIES)
    @pytest.mark.parametrize("nsamp", [0.3, 0.6])
    def test_ray_preserved_unit_direction_unchanged(self, geometry, nsamp):
        """Each perturbed node lies on the SAME ray as its unperturbed counterpart
        (unit vector unchanged) — generic over ALL geometries, since
        node_scale_factors multiplies each (1,3) position by a scalar."""
        gsym = self._grid(geometry, seed=42, nsamp=0.0)
        gper = self._grid(geometry, seed=42, nsamp=nsamp)
        p_sym = gsym.get_positions()
        p_per = gper.get_positions()
        for a, b in zip(p_sym, p_per):
            ua = a / np.linalg.norm(a)
            ub = b / np.linalg.norm(b)
            np.testing.assert_allclose(ua, ub, atol=1e-12,
                err_msg=f"{geometry}: node direction changed at nsamp={nsamp}")

    @pytest.mark.parametrize("geometry", GEOMETRIES)
    @pytest.mark.parametrize("nsamp", [0.3, 0.6])
    def test_mean_radial_scale_preserved(self, geometry, nsamp):
        """mean over nodes of (r_perturbed / r_unperturbed) == 1.0 (rtol 1e-12)."""
        gsym = self._grid(geometry, seed=13, nsamp=0.0)
        gper = self._grid(geometry, seed=13, nsamp=nsamp)
        r_sym = np.linalg.norm(gsym.get_positions(), axis=1)
        r_per = np.linalg.norm(gper.get_positions(), axis=1)
        np.testing.assert_allclose((r_per / r_sym).mean(), 1.0, rtol=1e-12,
            err_msg=f"{geometry}: mean radial scale not preserved at nsamp={nsamp}")

    @pytest.mark.parametrize("geometry", GEOMETRIES)
    @pytest.mark.parametrize("nsamp", [0.3, 0.6])
    def test_all_radii_positive(self, geometry, nsamp):
        gper = self._grid(geometry, seed=42, nsamp=nsamp)
        radii = np.linalg.norm(gper.get_positions(), axis=1)
        assert np.all(radii > 0), f"{geometry}: a node crossed the origin at nsamp={nsamp}"

    @pytest.mark.parametrize("geometry", GEOMETRIES)
    def test_determinism_same_seed(self, geometry):
        g1 = self._grid(geometry, seed=7, nsamp=0.6)
        g2 = self._grid(geometry, seed=7, nsamp=0.6)
        np.testing.assert_array_equal(g1.get_positions(), g2.get_positions())

    @pytest.mark.parametrize("geometry", GEOMETRIES)
    def test_different_seed_different_positions(self, geometry):
        g1 = self._grid(geometry, seed=1, nsamp=0.6)
        g2 = self._grid(geometry, seed=2, nsamp=0.6)
        assert not np.array_equal(g1.get_positions(), g2.get_positions()), \
            f"{geometry}: different seeds produced identical perturbed positions"


# ---------------------------------------------------------------------------
# E. Cross-knob independence per geometry (separate default_rng draws)
# ---------------------------------------------------------------------------

class TestCrossKnobIndependencePerGeometry:
    """Enabling node_s_amplitude must NOT change node_masses, and vice versa —
    the two knobs are SEPARATE default_rng draws (parametrized over geometry)."""

    @pytest.mark.parametrize("geometry", GEOMETRIES)
    def test_s_amplitude_does_not_change_masses(self, geometry):
        n = _n_nodes(geometry)
        both = _params(geometry, seed=42, nmamp=0.6, nsamp=0.6)
        mass_only = _params(geometry, seed=42, nmamp=0.6, nsamp=0.0)
        np.testing.assert_array_equal(both.node_masses(n), mass_only.node_masses(n))

    @pytest.mark.parametrize("geometry", GEOMETRIES)
    def test_mass_amplitude_does_not_change_scale_factors(self, geometry):
        n = _n_nodes(geometry)
        both = _params(geometry, seed=42, nmamp=0.6, nsamp=0.6)
        scale_only = _params(geometry, seed=42, nmamp=0.0, nsamp=0.6)
        np.testing.assert_array_equal(
            both.node_scale_factors(n), scale_only.node_scale_factors(n))


# ---------------------------------------------------------------------------
# F. effective_M_ext_kg parity per geometry's actual N
# ---------------------------------------------------------------------------

class TestEffectiveMassParity:
    """effective_M_ext_kg(M_ref, N) keeps total external mass == 26 * M_ref."""

    @pytest.mark.parametrize("geometry", GEOMETRIES)
    def test_total_effective_equals_cube26_reference(self, geometry):
        M_ref = 5e55
        n = _n_nodes(geometry)
        m_per_node = effective_M_ext_kg(M_ref, n, ref_nodes=26)
        np.testing.assert_allclose(n * m_per_node, 26 * M_ref, rtol=1e-12,
            err_msg=f"{geometry} (N={n}): effective total != 26*M_ref")
