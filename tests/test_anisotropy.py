"""
Unit tests for cosmo.anisotropy — Deliverable B: directional shear / Hubble-dipole
diagnostic.

All tests are hermetic: no N-body simulation is run. Synthetic position and velocity
arrays exercise the four public functions.

Test groups
-----------
1. shape_tensor         – isotropic sphere, stretched cloud, degenerate inputs.
2. axis_rms             – isotropic and anisotropic clouds, degenerate.
3. hubble_dipole        – isotropic Hubble flow, constructed dipole, sign flip.
4. expansion_anisotropy – isotropic and anisotropic expansion.
5. anisotropy_summary   – wrapper smoke test.
6. Determinism          – same arrays in, same arrays out.
"""

import numpy as np
import pytest

from cosmo.anisotropy import (
    shape_tensor,
    axis_rms,
    hubble_dipole,
    expansion_anisotropy,
    anisotropy_summary,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RNG = np.random.default_rng(0)


def _isotropic_sphere(N: int = 500, R: float = 1.0, rng_seed: int = 0) -> np.ndarray:
    """Random positions uniformly distributed in a sphere, COM-centred at 0."""
    rng = np.random.default_rng(rng_seed)
    # Rejection sample uniform sphere
    pts = []
    while len(pts) < N:
        batch = rng.uniform(-R, R, (N * 2, 3))
        inside = batch[np.linalg.norm(batch, axis=1) <= R]
        pts.extend(inside.tolist())
    pos = np.array(pts[:N])
    return pos - pos.mean(axis=0)   # ensure exact COM = 0


def _isotropic_hubble_velocities(positions: np.ndarray, H: float = 70e3 / 3.086e22) -> np.ndarray:
    """Pure Hubble flow: v_i = H * r_i (no peculiar velocity component)."""
    return H * positions


# ---------------------------------------------------------------------------
# 1. shape_tensor
# ---------------------------------------------------------------------------

class TestShapeTensor:

    def test_isotropic_sphere_shear_near_zero(self):
        """A uniform isotropic sphere should have shear_index << 1."""
        pos = _isotropic_sphere(N=2000, rng_seed=1)
        result = shape_tensor(pos)
        assert not result["degenerate"]
        # For a perfect sphere eigenvalues are equal; statistical noise on 2000 pts keeps this small
        assert result["shear_index"] < 0.2, (
            f"Expected shear_index < 0.2 for isotropic sphere, got {result['shear_index']:.4f}"
        )

    def test_stretched_cloud_principal_axis_is_x(self):
        """Stretch x by 5x: principal axis should be x, shear_index >> 0."""
        rng = np.random.default_rng(99)
        pos = rng.standard_normal((500, 3))
        pos[:, 0] *= 5.0   # stretch x by 5
        pos -= pos.mean(axis=0)

        result = shape_tensor(pos)
        assert not result["degenerate"]
        # The principal axis (largest eigenvalue) should point mostly along x
        ax = result["principal_axis"]
        assert abs(ax[0]) > 0.9, (
            f"Principal axis should be ~x; got {ax}"
        )
        assert result["shear_index"] > 1.0, (
            f"Expected shear_index > 1 for 5x stretch; got {result['shear_index']:.4f}"
        )

    def test_eigenvalues_ascending(self):
        """Eigenvalues should be sorted ascending."""
        pos = _isotropic_sphere(N=200)
        result = shape_tensor(pos)
        ev = result["eigenvalues"]
        assert ev[0] <= ev[1] <= ev[2]

    def test_degenerate_single_particle(self):
        """Single particle => degenerate flag, zero shear."""
        pos = np.array([[1.0, 2.0, 3.0]])
        result = shape_tensor(pos)
        assert result["degenerate"]
        assert result["shear_index"] == 0.0

    def test_degenerate_all_identical(self):
        """All identical positions => degenerate flag."""
        pos = np.tile([1.0, 1.0, 1.0], (10, 1))
        result = shape_tensor(pos)
        assert result["degenerate"]

    def test_principal_axis_is_unit_vector(self):
        """Principal axis must be a unit vector."""
        pos = _isotropic_sphere(N=100)
        result = shape_tensor(pos)
        ax = result["principal_axis"]
        np.testing.assert_allclose(np.linalg.norm(ax), 1.0, atol=1e-12)


# ---------------------------------------------------------------------------
# 2. axis_rms
# ---------------------------------------------------------------------------

class TestAxisRms:

    def test_isotropic_sphere_ratio_near_one(self):
        """Isotropic sphere: max/min ratio close to 1."""
        pos = _isotropic_sphere(N=2000)
        result = axis_rms(pos)
        assert not result["degenerate"]
        assert result["max_min_ratio"] < 1.3, (
            f"Expected max_min_ratio < 1.3 for sphere; got {result['max_min_ratio']:.4f}"
        )

    def test_stretched_cloud_ratio_matches_stretch(self):
        """Stretch x by factor k; max_min_ratio should be ~ k."""
        rng = np.random.default_rng(7)
        pos = rng.standard_normal((1000, 3))
        k = 4.0
        pos[:, 0] *= k
        pos -= pos.mean(axis=0)

        result = axis_rms(pos)
        assert not result["degenerate"]
        # Ratio between largest and smallest RMS; should be close to k=4
        assert abs(result["max_min_ratio"] - k) < 0.5, (
            f"Expected max_min_ratio ~ {k}; got {result['max_min_ratio']:.4f}"
        )

    def test_rms_shape(self):
        """Returns (3,) array."""
        pos = _isotropic_sphere(N=50)
        result = axis_rms(pos)
        assert result["rms"].shape == (3,)

    def test_degenerate_one_particle(self):
        """N=1 => degenerate."""
        pos = np.array([[0.0, 0.0, 0.0]])
        result = axis_rms(pos)
        assert result["degenerate"]
        assert result["max_min_ratio"] == 1.0


# ---------------------------------------------------------------------------
# 3. hubble_dipole
# ---------------------------------------------------------------------------

class TestHubbleDipole:

    def test_pure_hubble_flow_dipole_near_zero(self):
        """Pure isotropic Hubble flow should give dipole ~ 0."""
        pos = _isotropic_sphere(N=500, rng_seed=2)
        H = 70e3 / 3.086e22   # 70 km/s/Mpc in SI
        vel = _isotropic_hubble_velocities(pos, H)
        result = hubble_dipole(pos, vel)
        assert not result["degenerate"]
        # Dipole should be small; statistical noise on 500 particles
        assert abs(result["dipole"]) < 0.3, (
            f"Expected |dipole| < 0.3 for isotropic flow; got {result['dipole']:.4f}"
        )

    def test_constructed_dipole_along_x_positive(self):
        """Give +x hemisphere 20% higher radial speed: dipole > 0, best_axis ~ x."""
        pos = _isotropic_sphere(N=1000, rng_seed=3)
        H = 70e3 / 3.086e22
        vel = _isotropic_hubble_velocities(pos, H)

        # Boost +x hemisphere by 20%
        boost_mask = pos[:, 0] >= 0
        vel[boost_mask] *= 1.20

        axis_x = np.array([1.0, 0.0, 0.0])
        result = hubble_dipole(pos, vel, axis=axis_x)
        assert not result["degenerate"]
        assert result["dipole"] > 0.05, (
            f"Expected dipole > 0.05 for +x boost; got {result['dipole']:.4f}"
        )
        # best_axis should be approximately x (|dot| > 0.7)
        dot = abs(result["best_axis"] @ axis_x)
        assert dot > 0.7, (
            f"Expected best_axis ~ x (|dot| > 0.7); got best_axis={result['best_axis']}"
        )

    def test_dipole_sign_flips_when_boost_reversed(self):
        """Boost -x hemisphere instead: dipole should flip sign."""
        pos = _isotropic_sphere(N=1000, rng_seed=3)
        H = 70e3 / 3.086e22
        vel_plus = _isotropic_hubble_velocities(pos, H)
        vel_minus = _isotropic_hubble_velocities(pos, H)

        # Boost +x for one, -x for the other
        vel_plus[pos[:, 0] >= 0] *= 1.20
        vel_minus[pos[:, 0] < 0] *= 1.20

        axis_x = np.array([1.0, 0.0, 0.0])
        d_plus  = hubble_dipole(pos, vel_plus,  axis=axis_x)["dipole"]
        d_minus = hubble_dipole(pos, vel_minus, axis=axis_x)["dipole"]

        assert d_plus * d_minus < 0, (
            f"Expected opposite-sign dipoles; got d_plus={d_plus:.4f}, d_minus={d_minus:.4f}"
        )

    def test_degenerate_few_particles(self):
        """N=3 => degenerate."""
        pos = np.eye(3)
        vel = np.zeros((3, 3))
        result = hubble_dipole(pos, vel)
        assert result["degenerate"]

    def test_starved_hemisphere_marks_degenerate_not_zero(self):
        """A lopsided cloud that starves one hemisphere along the probe axis must
        report degenerate=True, NOT a confident dipole==0 (false isotropy).

        Regression for the bug where a hemisphere with < 2 particles fell back to
        H_global on BOTH sides, forcing dipole = (H+ - H-)/H_mean to exactly 0
        with degenerate=False — masking real anisotropy as clean isotropy.
        """
        # N >= 4 (so the N<4 guard does not trip) but only ONE particle ends up in
        # the -x hemisphere after COM-centering: the split along x is 4/1, so the
        # -x hemisphere is starved (< 2 particles) and cannot be slope-fit.
        H = 70e3 / 3.086e22
        pos = np.array([
            [10.0, 0.01, -0.02],
            [10.0, -0.03, 0.04],
            [10.0, 0.02, 0.01],
            [10.0, -0.01, 0.03],
            [-10.0, 0.0, 0.0],   # the lone -x particle starves that hemisphere
        ])
        vel = H * pos
        axis_x = np.array([1.0, 0.0, 0.0])
        result = hubble_dipole(pos, vel, axis=axis_x)

        assert result["degenerate"], (
            "Starved hemisphere must be reported as degenerate (could-not-measure), "
            f"got degenerate=False with dipole={result['dipole']:.4f}"
        )
        # And it must NOT masquerade as a confident non-degenerate zero.
        assert not (result["dipole"] != 0.0 and not result["degenerate"])

    def test_well_populated_isotropic_is_not_degenerate(self):
        """The intended (N>=2000) well-populated isotropic case stays
        degenerate=False with ~0 dipole — behaviour unchanged by the starved-
        hemisphere guard."""
        pos = _isotropic_sphere(N=2000, rng_seed=8)
        H = 70e3 / 3.086e22
        vel = _isotropic_hubble_velocities(pos, H)
        result = hubble_dipole(pos, vel)
        assert not result["degenerate"]
        assert abs(result["dipole"]) < 0.3, (
            f"Well-populated isotropic flow should give ~0 dipole; got {result['dipole']:.4f}"
        )

    def test_global_hubble_slope_positive(self):
        """H_global should be positive for an expanding cloud."""
        pos = _isotropic_sphere(N=200, rng_seed=5)
        H = 70e3 / 3.086e22
        vel = _isotropic_hubble_velocities(pos, H)
        result = hubble_dipole(pos, vel)
        assert result["H_global"] > 0

    def test_shape_mismatch_raises(self):
        """Mismatched positions/velocities shapes => ValueError."""
        pos = np.zeros((10, 3))
        vel = np.zeros((8, 3))
        with pytest.raises(ValueError):
            hubble_dipole(pos, vel)


# ---------------------------------------------------------------------------
# 4. expansion_anisotropy
# ---------------------------------------------------------------------------

class TestExpansionAnisotropy:

    def test_isotropic_expansion_spread_near_one(self):
        """Isotropic uniform expansion: spread should be ~ 1."""
        pos_i = _isotropic_sphere(N=500, rng_seed=10)
        scale = 2.5   # expand uniformly
        pos_f = pos_i * scale

        result = expansion_anisotropy(pos_i, pos_f)
        assert not result["degenerate"]
        np.testing.assert_allclose(result["spread"], 1.0, atol=0.02)
        np.testing.assert_allclose(result["growth_factor"], scale, atol=0.01)

    def test_anisotropic_expansion_spread_gt_one(self):
        """Stretch x more than y/z: spread > 1."""
        pos_i = _isotropic_sphere(N=500, rng_seed=11)
        pos_f = pos_i.copy()
        pos_f[:, 0] *= 4.0   # x grows by 4
        pos_f[:, 1] *= 2.0   # y grows by 2
        pos_f[:, 2] *= 2.0   # z grows by 2

        result = expansion_anisotropy(pos_i, pos_f)
        assert not result["degenerate"]
        # growth_factor[0] should be ~ 4 and the others ~ 2
        assert result["spread"] > 1.5, (
            f"Expected spread > 1.5; got {result['spread']:.4f}"
        )

    def test_degenerate_initial_cloud(self):
        """Single initial point => degenerate."""
        pos_i = np.array([[0.0, 0.0, 0.0]])
        pos_f = np.array([[1.0, 1.0, 1.0]])
        result = expansion_anisotropy(pos_i, pos_f)
        assert result["degenerate"]


# ---------------------------------------------------------------------------
# 5. anisotropy_summary
# ---------------------------------------------------------------------------

class TestAnisotropySummary:

    def test_returns_all_keys(self):
        """Summary returns shape, axis_rms, hubble_dipole, expansion."""
        pos = _isotropic_sphere(N=100)
        H = 70e3 / 3.086e22
        vel = _isotropic_hubble_velocities(pos, H)
        pos_i = pos / 2.0   # pretend initial was half-size

        result = anisotropy_summary(pos, vel, positions_initial=pos_i, positions_final=pos)
        assert "shape" in result
        assert "axis_rms" in result
        assert "hubble_dipole" in result
        assert "expansion" in result
        assert result["expansion"] is not None

    def test_expansion_none_if_no_initial(self):
        """Without positions_initial, expansion key is None."""
        pos = _isotropic_sphere(N=50)
        vel = np.zeros_like(pos)
        result = anisotropy_summary(pos, vel)
        assert result["expansion"] is None


# ---------------------------------------------------------------------------
# 6. Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:

    def test_same_input_same_output(self):
        """All four functions must be deterministic (no randomness)."""
        pos = _isotropic_sphere(N=300, rng_seed=42)
        H = 70e3 / 3.086e22
        vel = _isotropic_hubble_velocities(pos, H)

        for _ in range(3):
            r_st  = shape_tensor(pos)
            r_ar  = axis_rms(pos)
            r_hd  = hubble_dipole(pos, vel)
            r_ea  = expansion_anisotropy(pos / 2.0, pos)

            np.testing.assert_array_equal(r_st["eigenvalues"],  shape_tensor(pos)["eigenvalues"])
            np.testing.assert_array_equal(r_ar["rms"],          axis_rms(pos)["rms"])
            assert r_hd["dipole"] == hubble_dipole(pos, vel)["dipole"]
            np.testing.assert_array_equal(r_ea["growth_factor"], expansion_anisotropy(pos / 2.0, pos)["growth_factor"])


# ---------------------------------------------------------------------------
# 7. Isotropic vs anisotropic: the key deliverable check
# ---------------------------------------------------------------------------

class TestIsotropicVsAnisotropic:
    """Verify the core deliverable: amplitude=0 -> ~zero signal; amplitude>0 -> measurable signal."""

    def _make_isotropic_cloud(self, N: int = 500) -> tuple[np.ndarray, np.ndarray]:
        pos = _isotropic_sphere(N=N, rng_seed=0)
        H = 70e3 / 3.086e22
        vel = _isotropic_hubble_velocities(pos, H)
        return pos, vel

    def _make_anisotropic_cloud(self, N: int = 500, stretch: float = 3.0) -> tuple[np.ndarray, np.ndarray]:
        """Cloud stretched along x by `stretch`; +x hemisphere has higher Hubble rate."""
        rng = np.random.default_rng(77)
        pos = rng.standard_normal((N, 3))
        pos[:, 0] *= stretch
        pos -= pos.mean(axis=0)

        # +x hemisphere gets stretch-fold higher Hubble rate (mimics anisotropic node masses)
        H = 70e3 / 3.086e22
        vel = H * pos
        boost = pos[:, 0] >= 0
        vel[boost, 0] *= stretch   # faster expansion along x in +x hemisphere
        vel -= vel.mean(axis=0)    # remove COM velocity
        return pos, vel

    def test_isotropic_shear_small(self):
        pos, _ = self._make_isotropic_cloud(N=2000)
        result = shape_tensor(pos)
        assert result["shear_index"] < 0.2, (
            f"Isotropic cloud: shear_index={result['shear_index']:.4f}, expected < 0.2"
        )

    def test_anisotropic_shear_large(self):
        pos, _ = self._make_anisotropic_cloud(N=500, stretch=3.0)
        result = shape_tensor(pos)
        assert result["shear_index"] > 1.0, (
            f"Anisotropic cloud: shear_index={result['shear_index']:.4f}, expected > 1.0"
        )

    def test_isotropic_dipole_small(self):
        pos, vel = self._make_isotropic_cloud(N=1000)
        result = hubble_dipole(pos, vel)
        assert abs(result["best_dipole"]) < 0.25, (
            f"Isotropic cloud: best_dipole={result['best_dipole']:.4f}, expected < 0.25"
        )

    def test_anisotropic_dipole_nonzero(self):
        pos, vel = self._make_anisotropic_cloud(N=500, stretch=3.0)
        result = hubble_dipole(pos, vel, axis=np.array([1.0, 0.0, 0.0]))
        assert abs(result["dipole"]) > 0.10, (
            f"Anisotropic cloud: dipole={result['dipole']:.4f}, expected > 0.10"
        )

    def test_expansion_isotropic_near_one(self):
        pos, _ = self._make_isotropic_cloud(N=500)
        result = expansion_anisotropy(pos / 2.0, pos)
        assert result["spread"] < 1.15, (
            f"Isotropic expansion: spread={result['spread']:.4f}, expected < 1.15"
        )

    def test_expansion_anisotropic_above_one(self):
        pos_i = _isotropic_sphere(N=500, rng_seed=20)
        pos_f = pos_i.copy()
        pos_f[:, 0] *= 3.0   # x expands 3x more
        result = expansion_anisotropy(pos_i, pos_f)
        assert result["spread"] > 2.0, (
            f"Anisotropic expansion: spread={result['spread']:.4f}, expected > 2.0"
        )
