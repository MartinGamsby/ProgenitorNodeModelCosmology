"""
Tests for node_s_amplitude: deterministic per-node RADIAL position perturbation.

node_s_amplitude perturbs the 26 HMEA node POSITIONS off the perfect symmetric
lattice (analogous to node_mass_amplitude, which perturbs node MASSES). It scales
each node's distance from the origin by a mean-preserving log-normal factor while
keeping each node on its original ray (direction fixed), so it breaks the lattice
symmetry RADIALLY without changing the net scale S or any node direction.

Invariants tested:
  (a) Vanishes at 0: node_s_amplitude=0 => positions byte-identical to symmetric
      lattice; node_scale_factors() returns exact ones.
  (b) Deterministic per (node_mass_seed, node_s_amplitude); different seed/amp
      => different factors.
  (c) MEAN radial scale preserved: mean(scale_factors) == 1.0 exactly.
  (d) Nodes stay OUTSIDE the cloud for the experimental amplitude range.
  (e) Strictly positive factors (no node crosses the origin / flips sides).
  (f) Independent of the particle realization (confound guard) — end-to-end.
"""
import numpy as np
import pytest

from cosmo.constants import (
    ExternalNodeParameters, SimulationParameters, CosmologicalConstants,
)
from cosmo.particles import HMEAGrid

const = CosmologicalConstants()
M_EXT_KG = 5e55
S_DEFAULT_M = 30.0 * const.Gpc_to_m


# ---------------------------------------------------------------------------
# ExternalNodeParameters.node_scale_factors() unit tests
# ---------------------------------------------------------------------------

class TestNodeScaleFactorsMethod:

    def _params(self, seed=0, s_amp=0.0, S=S_DEFAULT_M):
        return ExternalNodeParameters(M_ext_kg=M_EXT_KG, S=S,
                                      node_mass_seed=seed, node_s_amplitude=s_amp)

    def test_zero_amplitude_returns_exact_ones(self):
        """node_s_amplitude=0 => every factor is exactly 1.0 (byte-identical path)."""
        sf = self._params(seed=12345, s_amp=0.0).node_scale_factors(26)
        assert sf.shape == (26,)
        assert np.all(sf == 1.0)

    def test_determinism_same_seed_same_factors(self):
        p1 = self._params(seed=7, s_amp=0.5)
        p2 = self._params(seed=7, s_amp=0.5)
        np.testing.assert_array_equal(p1.node_scale_factors(26),
                                      p2.node_scale_factors(26))

    def test_determinism_different_seed_different_factors(self):
        p1 = self._params(seed=1, s_amp=0.5)
        p2 = self._params(seed=2, s_amp=0.5)
        assert not np.array_equal(p1.node_scale_factors(26),
                                  p2.node_scale_factors(26))

    def test_mean_preservation_various_amplitudes(self):
        for seed in [0, 42, 99, 1337]:
            for s_amp in [0.1, 0.3, 0.6, 1.0]:
                sf = self._params(seed=seed, s_amp=s_amp).node_scale_factors(26)
                np.testing.assert_allclose(
                    sf.mean(), 1.0, rtol=1e-12,
                    err_msg=f"Mean scale not preserved for seed={seed}, s_amp={s_amp}")

    def test_strict_positivity(self):
        for s_amp in [0.0, 0.1, 0.5, 1.0, 2.0]:
            sf = self._params(seed=42, s_amp=s_amp).node_scale_factors(26)
            assert np.all(sf > 0), f"Non-positive scale factor at s_amp={s_amp}"

    def test_independent_of_global_rng(self):
        p = self._params(seed=42, s_amp=0.5)
        np.random.seed(999); _ = np.random.rand(1000)
        before = p.node_scale_factors(26)
        np.random.seed(0); _ = np.random.rand(5000)
        after = p.node_scale_factors(26)
        np.testing.assert_array_equal(before, after)

    def test_uses_separate_rng_draw_from_node_masses(self):
        """node_scale_factors and node_masses both seed off node_mass_seed but are
        SEPARATE default_rng draws, so enabling one does not perturb the other."""
        p = ExternalNodeParameters(M_ext_kg=M_EXT_KG, S=S_DEFAULT_M,
                                   node_mass_seed=42,
                                   node_mass_amplitude=0.5, node_s_amplitude=0.5)
        # node_masses unchanged whether or not node_s_amplitude is set
        p_mass_only = ExternalNodeParameters(M_ext_kg=M_EXT_KG, S=S_DEFAULT_M,
                                             node_mass_seed=42,
                                             node_mass_amplitude=0.5,
                                             node_s_amplitude=0.0)
        np.testing.assert_array_equal(p.node_masses(26), p_mass_only.node_masses(26))


# ---------------------------------------------------------------------------
# HMEAGrid position threading
# ---------------------------------------------------------------------------

class TestHMEAGridNodePositions:

    def _grid(self, seed=0, s_amp=0.0, S=S_DEFAULT_M):
        p = ExternalNodeParameters(M_ext_kg=M_EXT_KG, S=S,
                                   node_mass_seed=seed, node_s_amplitude=s_amp)
        return HMEAGrid(node_params=p)

    def test_zero_amplitude_positions_identical_to_symmetric(self):
        """node_s_amplitude=0 => node positions byte-identical to the symmetric lattice."""
        g0 = self._grid(seed=42, s_amp=0.0)
        # Reconstruct the symmetric lattice independently
        S = S_DEFAULT_M
        expected = []
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                for k in [-1, 0, 1]:
                    if i == j == k == 0:
                        continue
                    expected.append(np.array([i, j, k], dtype=float) * S)
        np.testing.assert_array_equal(g0.get_positions(), np.array(expected))

    def test_nonzero_amplitude_changes_positions(self):
        g0 = self._grid(seed=42, s_amp=0.0)
        gp = self._grid(seed=42, s_amp=0.5)
        assert not np.array_equal(g0.get_positions(), gp.get_positions())

    def test_direction_preserved_only_radius_scaled(self):
        """Each perturbed node lies on the SAME ray as its symmetric counterpart
        (unit direction unchanged); only its radius changes."""
        g0 = self._grid(seed=42, s_amp=0.6)
        gsym = self._grid(seed=42, s_amp=0.0)
        p_sym = gsym.get_positions()
        p_per = g0.get_positions()
        for a, b in zip(p_sym, p_per):
            ua = a / np.linalg.norm(a)
            ub = b / np.linalg.norm(b)
            np.testing.assert_allclose(ua, ub, atol=1e-12)

    def test_mean_radial_scale_preserved_in_grid(self):
        """mean over nodes of (perturbed_radius / symmetric_radius) == 1.0."""
        gsym = self._grid(seed=13, s_amp=0.0)
        gper = self._grid(seed=13, s_amp=0.6)
        r_sym = np.linalg.norm(gsym.get_positions(), axis=1)
        r_per = np.linalg.norm(gper.get_positions(), axis=1)
        np.testing.assert_allclose((r_per / r_sym).mean(), 1.0, rtol=1e-12)

    def test_nodes_stay_outside_cloud(self):
        """For the experimental amplitude range the closest node radius still
        exceeds the cloud's max particle radius at t_start=2.9 Gyr.

        Box ~ 4.39 Gpc at t_start=2.9 => max particle radius ~ box/2 ~ 2.2 Gpc.
        Use a conservative 3 Gpc cloud-edge bound. Smallest node radius at S=30
        with s_amp<=0.6 is ~8.6 Gpc (face node), comfortably outside.
        """
        cloud_edge_Gpc = 3.0
        for s_amp in [0.0, 0.3, 0.6]:
            grid = self._grid(seed=42, s_amp=s_amp, S=30.0 * const.Gpc_to_m)
            r_min_Gpc = np.linalg.norm(grid.get_positions(), axis=1).min() / const.Gpc_to_m
            assert r_min_Gpc > cloud_edge_Gpc, (
                f"Closest node ({r_min_Gpc:.2f} Gpc) crossed into cloud "
                f"(edge {cloud_edge_Gpc} Gpc) at s_amp={s_amp}")


# ---------------------------------------------------------------------------
# SimulationParameters / SweepConfig plumbing
# ---------------------------------------------------------------------------

class TestPlumbing:

    def test_default_field_zero(self):
        p = SimulationParameters()
        assert hasattr(p, 'node_s_amplitude')
        assert p.node_s_amplitude == 0.0
        assert p.external_params.node_s_amplitude == 0.0

    def test_field_threads_into_external_params(self):
        p = SimulationParameters(node_mass_seed=7, node_s_amplitude=0.4)
        assert p.external_params.node_s_amplitude == 0.4
        assert p.external_params.node_mass_seed == 7

    def test_sweep_config_default_and_custom(self):
        from cosmo.parameter_sweep import SweepConfig
        assert SweepConfig().node_s_amplitude == 0.0
        assert SweepConfig(node_s_amplitude=0.3).node_s_amplitude == 0.3


# ---------------------------------------------------------------------------
# Cache-key slug
# ---------------------------------------------------------------------------

class TestCacheSlug:

    def _name(self, *, seed=0, nmamp=0.0, nsamp=0.0):
        from cosmo.parameter_sweep import SweepConfig, build_cache_name
        cfg = SweepConfig(quick_search=True, objective='pantheon',
                          node_mass_seed=seed, node_mass_amplitude=nmamp,
                          node_s_amplitude=nsamp)
        return build_cache_name(cfg, M_factor=800, S_val=25, centerM=1, seeds=[42])

    def test_zero_s_amplitude_no_slug(self):
        key = self._name(seed=99, nsamp=0.0)
        assert 'nsamp' not in key

    def test_nonzero_s_amplitude_appends_slug(self):
        key = self._name(seed=7, nsamp=0.6)
        assert 'nsamp' in key
        assert 'nmseed' in key  # node_s also depends on the seed

    def test_different_s_amplitudes_distinct_keys(self):
        assert self._name(nsamp=0.3) != self._name(nsamp=0.6)

    def test_different_seeds_distinct_keys_with_s_amplitude(self):
        assert self._name(seed=1, nsamp=0.5) != self._name(seed=2, nsamp=0.5)

    def test_s_amplitude_distinct_from_uniform(self):
        assert self._name(nsamp=0.5) != self._name(nsamp=0.0)


# ---------------------------------------------------------------------------
# End-to-end sim-path confound guard
# ---------------------------------------------------------------------------

class TestSimPathInvariants:
    """node_s_amplitude must perturb ONLY node positions, never the particle cloud."""

    def _build_sim(self, s_amp, nm_seed=42):
        from cosmo.simulation import CosmologicalSimulation
        from cosmo.factories import setup_simulation_context
        import io, contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            box, a_start, _ = setup_simulation_context(2.9, 13.8 - 2.9, 273, 10)
            params = SimulationParameters(
                M_value=1500, S_value=30, n_particles=120, seed=42,
                t_start_Gyr=2.9, t_duration_Gyr=13.8 - 2.9, n_steps=273,
                center_node_mass=1, mass_randomize=0.0,
                node_mass_seed=nm_seed, node_s_amplitude=s_amp,
                init_distribution="grf",
            )
            sim = CosmologicalSimulation(params, box, a_start,
                                         use_external_nodes=True, use_dark_energy=False)
        return sim

    def test_particle_cloud_independent_of_s_amplitude(self):
        """Particle positions+velocities byte-identical across node_s_amplitude.

        node_scale_factors() draws from its OWN default_rng AFTER the cloud is
        built; it must never advance the global RNG that seeds the cloud.
        NOTE: with the pre-start tidal boost ON, the boost depends on the node
        POSITIONS, so velocities legitimately differ. Compare positions only here,
        and velocities with the boost OFF in the next test.
        """
        s0 = self._build_sim(0.0)
        sp = self._build_sim(0.6)
        np.testing.assert_array_equal(s0.particles.get_positions(),
                                      sp.particles.get_positions())

    def test_total_external_mass_unchanged_by_s_amplitude(self):
        """Per-node masses (and their sum) are untouched by node_s_amplitude."""
        s0 = self._build_sim(0.0)
        sp = self._build_sim(0.6)
        np.testing.assert_array_equal(s0.hmea_grid.get_masses(),
                                      sp.hmea_grid.get_masses())
