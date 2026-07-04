"""Unit tests for the INFINITE virialized meta-structure (cosmo/virialized_medium.py).

These pin the physical claims the structure must satisfy to be "virialized in an infinite
theoretical universe":
  * VIRIAL EQUILIBRIUM: 2K/|U| ~ 1 (order unity, bounded) -- held up by velocity dispersion, NOT
    static force-balance (a static net-zero-force structure is a crystal/glass, not virialized).
  * HOMOGENEOUS, NOT concentrated: uniform on average (density std O(1)); does NOT pile up at the
    centre (unlike a finite relaxed halo) -- the "similar density everywhere when zoomed out" claim.
  * INFINITE / no special centre: the periodic (min-image) field is translation-invariant -- every
    node is virialized by its surroundings on all sides; there is no privileged origin.
  * "US" / the progenitor node (centerM) is taken into account: node 0 sits at the origin with mass
    center_mass_frac x mean, participates in the dynamics, and changes the virial state.
"""
import numpy as np
import pytest

from cosmo.virialized_medium import (
    relax_virialized_medium, virial_ratio, _mi_accel_pot, density_std, central_concentration,
)


@pytest.fixture(scope="module")
def medium():
    """A small, fast relaxed medium reused across tests (N=125, ~2-3 s)."""
    return relax_virialized_medium(n_nodes=125, sigma=1.5, seed=0, n_steps=800, return_history=True)


def test_reaches_and_stays_virial_equilibrium(medium):
    """2K/|U| is ORDER UNITY and BOUNDED over the relaxation -> virialized (not a static crystal at
    ~0, not unbound at ->inf). This is the 'virialized in an infinite universe' criterion."""
    vr = medium["virial_ratio"]
    assert 0.5 < vr < 2.5, f"virial ratio {vr} not order-unity (virialized)"
    hist = medium["history"]
    assert hist is not None and len(hist) > 3
    assert hist.min() > 0.3, "medium collapsed (virial ratio -> 0, would be a static clump)"
    assert hist.max() < 3.0, "medium evaporated/unbound (virial ratio blew up)"


def test_homogeneous_not_centrally_concentrated(medium):
    """Uniform on average (density std O(1)) and NOT piled up at the centre -- distinguishes the
    infinite medium from the (wrong) finite relaxed halo, whose concentration would be >> uniform."""
    assert medium["density_std"] < 3.0, "density too clumped (collapsed, not a homogeneous medium)"
    # uniform expectation = 0.25^3 = 0.0156; a centrally-concentrated halo gives >> 0.15.
    assert medium["concentration"] < 0.15, (
        f"medium is centrally concentrated (conc {medium['concentration']:.3f}) -- that is a halo, "
        f"not a homogeneous infinite medium")


def test_us_progenitor_node_at_origin(medium):
    """Node 0 is 'us' / the progenitor: at the origin, mass == center_mass_frac (default 1.0)."""
    assert medium["center_index"] == 0
    assert np.linalg.norm(medium["pos"][0]) < 1e-9, "the 'us' node is not at the origin"
    assert medium["mass"][0] == pytest.approx(1.0), "default center_mass_frac should be 1.0 (typical node)"
    assert medium["mass"][1:].mean() == pytest.approx(1.0, abs=0.1), "non-central nodes should have mean mass ~1"


def test_centerM_is_taken_into_account():
    """The 'us'/centerM node PARTICIPATES: scaling center_mass_frac scales node 0's mass AND changes
    the virial state (so the virialization accounts for us, not ignores us)."""
    light = relax_virialized_medium(n_nodes=125, sigma=1.5, seed=0, center_mass_frac=1.0, n_steps=800)
    heavy = relax_virialized_medium(n_nodes=125, sigma=1.5, seed=0, center_mass_frac=12.0, n_steps=800)
    assert heavy["mass"][0] == pytest.approx(12.0)
    assert light["mass"][0] == pytest.approx(1.0)
    # a 12x central mass measurably changes the medium's virial ratio (it is in the potential sum).
    assert abs(heavy["virial_ratio"] - light["virial_ratio"]) > 1e-3, (
        "center_mass_frac had no effect -> the central (us) node is NOT accounted for")


def test_periodic_field_is_translation_invariant():
    """The 'infinite homogeneous universe' property: shifting ALL node positions by a constant (with
    periodic wrap) leaves every node's acceleration unchanged -- there is no special centre; each
    node is virialized by its surroundings on every side. A FINITE (open) structure lacks this."""
    rng = np.random.default_rng(3)
    N, L = 200, 1.0
    pos = rng.random((N, 3)) * L
    mass = np.exp(1.5 * rng.standard_normal(N)); mass /= mass.mean()
    eps = 0.5 * L / N ** (1 / 3)
    a0, _ = _mi_accel_pot(pos, mass, L, eps)
    shift = rng.random(3) * L
    a1, _ = _mi_accel_pot((pos + shift) % L, mass, L, eps)
    np.testing.assert_allclose(a0, a1, rtol=1e-10, atol=1e-8)


def test_deterministic_per_seed():
    """Same (n_nodes, sigma, seed) -> identical structure (reproducible boundary conditions)."""
    a = relax_virialized_medium(n_nodes=100, sigma=1.5, seed=7, n_steps=400)
    b = relax_virialized_medium(n_nodes=100, sigma=1.5, seed=7, n_steps=400)
    np.testing.assert_array_equal(a["pos"], b["pos"])
    np.testing.assert_array_equal(a["mass"], b["mass"])
    assert a["virial_ratio"] == b["virial_ratio"]


def test_different_seed_differs():
    a = relax_virialized_medium(n_nodes=100, sigma=1.5, seed=1, n_steps=400)
    b = relax_virialized_medium(n_nodes=100, sigma=1.5, seed=2, n_steps=400)
    assert not np.array_equal(a["pos"], b["pos"])


def test_virial_ratio_signs():
    """virial_ratio is positive and finite for a real configuration; density/concentration helpers
    behave (uniform -> low concentration)."""
    r = relax_virialized_medium(n_nodes=100, sigma=1.0, seed=5, n_steps=400)
    assert np.isfinite(r["virial_ratio"]) and r["virial_ratio"] > 0
    assert 0.0 <= r["concentration"] < 1.0
    assert r["density_std"] >= 0.0


# --------------------------------------------------------------------------------------------------
# Integration: build_virialized_grid(vir_relax_mode="medium") as the HMEA node geometry.
# The medium is relaxed WITH "us" (node 0) but "us" is DROPPED from the returned HMEAs (in the sim
# the cloud is us), the HMEA NN spacing is rescaled to S, and masses are mean-preserving.
# --------------------------------------------------------------------------------------------------
_S_M = None
try:
    from cosmo.node_geometry import build_virialized_grid, nearest_neighbour_spacing
    from cosmo.constants import CosmologicalConstants
    _S_M = 30.0 * CosmologicalConstants.Gpc_to_m
except Exception:  # pragma: no cover
    pass


@pytest.fixture(scope="module")
def medium_grid():
    return build_virialized_grid(_S_M, n_nodes=60, M_ext_kg=1.0, vir_mass_spread=1.5,
                                 vir_s_metric="median", vir_relax_mode="medium", seed=0)


def test_medium_grid_drops_us(medium_grid):
    """n_nodes HMEAs are returned (the extra 'us' node is relaxed in the medium then dropped)."""
    pos, mass = medium_grid
    assert len(pos) == 60 and len(mass) == 60, "us (node 0) should be dropped from the HMEAs"


def test_medium_grid_nn_equals_S(medium_grid):
    pos, _ = medium_grid
    assert nearest_neighbour_spacing(pos, "median") == pytest.approx(_S_M, rel=1e-6)


def test_medium_grid_mean_preserving():
    _, mass = build_virialized_grid(_S_M, n_nodes=60, M_ext_kg=7.0, vir_mass_spread=1.5,
                                    vir_relax_mode="medium", seed=0)
    assert mass.mean() == pytest.approx(7.0, rel=1e-9), "mean(HMEA masses) must equal M_ext_kg"


def test_medium_grid_M0_is_eds():
    """M_ext_kg=0 -> all HMEA masses 0 (M=0 == Einstein-de Sitter invariant preserved)."""
    _, mass = build_virialized_grid(_S_M, n_nodes=60, M_ext_kg=0.0, vir_mass_spread=1.5,
                                    vir_relax_mode="medium", seed=0)
    assert np.all(mass == 0.0)


def test_medium_grid_homogeneous_not_halo(medium_grid):
    """The HMEAs are homogeneous around us, NOT centrally concentrated (unlike a finite halo)."""
    pos, _ = medium_grid
    r = np.linalg.norm(pos, axis=1)
    conc = np.mean(r < 0.25 * r.max())          # uniform ~ 0.25^3 = 0.016; a halo would be >> 0.1
    assert conc < 0.12, f"HMEA field is centrally concentrated (conc {conc:.3f}) -- not a medium"


def test_medium_grid_deterministic():
    a = build_virialized_grid(_S_M, n_nodes=50, M_ext_kg=1.0, vir_mass_spread=1.5,
                              vir_relax_mode="medium", seed=3)
    b = build_virialized_grid(_S_M, n_nodes=50, M_ext_kg=1.0, vir_mass_spread=1.5,
                              vir_relax_mode="medium", seed=3)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])


def test_nbody_mode_removed():
    """The wrong finite 'nbody' cluster mode is gone; only lattice/gradient/medium are valid."""
    with pytest.raises(ValueError):
        build_virialized_grid(_S_M, n_nodes=50, M_ext_kg=1.0, vir_mass_spread=1.5,
                              vir_relax_mode="nbody", seed=0)
