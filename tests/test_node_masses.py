"""
Tests for Deliverable B: deterministic seed-driven HMEA node masses.

Invariants tested:
  (a) Deterministic & reproducible: same (seed, amplitude) => identical vector.
  (b) All masses strictly positive.
  (c) MEAN-PRESERVING: mean(m_i) == M_ext_kg exactly (within float tolerance).
  (d) amplitude == 0.0 => all 26 nodes uniform == M_ext_kg (backward compatible).
"""
import numpy as np
import pytest

from cosmo.constants import ExternalNodeParameters, SimulationParameters
from cosmo.particles import HMEAGrid


M_EXT_KG = 5e55  # default ExternalNodeParameters value


# ---------------------------------------------------------------------------
# ExternalNodeParameters.node_masses() unit tests
# ---------------------------------------------------------------------------

class TestNodeMassesMethod:
    """Direct tests of ExternalNodeParameters.node_masses()."""

    def _make_params(self, seed=0, amplitude=0.0, M_ext_kg=M_EXT_KG):
        return ExternalNodeParameters(M_ext_kg=M_ext_kg, node_mass_seed=seed,
                                      node_mass_amplitude=amplitude)

    def test_uniform_default_all_equal_M_ext_kg(self):
        """amplitude=0.0 => every node mass equals M_ext_kg (backward compatible)."""
        params = self._make_params(seed=0, amplitude=0.0)
        masses = params.node_masses(26)
        assert masses.shape == (26,)
        np.testing.assert_array_equal(masses, M_EXT_KG)

    def test_uniform_default_n_nodes_3(self):
        """amplitude=0.0 works for any n_nodes."""
        params = self._make_params(amplitude=0.0)
        masses = params.node_masses(3)
        np.testing.assert_array_equal(masses, M_EXT_KG)

    def test_determinism_same_seed_same_vector(self):
        """Same (seed, amplitude) => identical 26-vector across two constructions."""
        p1 = self._make_params(seed=7, amplitude=0.5)
        p2 = self._make_params(seed=7, amplitude=0.5)
        np.testing.assert_array_equal(p1.node_masses(26), p2.node_masses(26))

    def test_determinism_different_seed_different_vector(self):
        """Different seed, same amplitude > 0 => different mass vector."""
        p1 = self._make_params(seed=1, amplitude=0.5)
        p2 = self._make_params(seed=2, amplitude=0.5)
        assert not np.array_equal(p1.node_masses(26), p2.node_masses(26)), \
            "Different seeds must produce different mass distributions"

    def test_mean_preservation_various_amplitudes(self):
        """mean(m_i) == M_ext_kg within float tol for several amplitudes/seeds."""
        for seed in [0, 42, 99, 1337]:
            for amplitude in [0.1, 0.3, 0.5, 1.0, 2.0]:
                params = self._make_params(seed=seed, amplitude=amplitude)
                masses = params.node_masses(26)
                np.testing.assert_allclose(
                    masses.mean(), M_EXT_KG, rtol=1e-12,
                    err_msg=f"Mean not preserved for seed={seed}, amplitude={amplitude}"
                )

    def test_strict_positivity(self):
        """All node masses > 0 for amplitudes up to 2.0."""
        for amplitude in [0.0, 0.1, 0.5, 1.0, 2.0]:
            params = self._make_params(seed=42, amplitude=amplitude)
            masses = params.node_masses(26)
            assert np.all(masses > 0), \
                f"Not all masses positive for amplitude={amplitude}: min={masses.min()}"

    def test_amplitude_zero_returns_uniform_array_not_computed(self):
        """amplitude=0.0 returns np.full (fast path), not the computed path."""
        params = self._make_params(seed=12345, amplitude=0.0)
        masses = params.node_masses(26)
        # All should be exactly equal (not just mean-equal)
        assert np.all(masses == M_EXT_KG)

    def test_custom_M_ext_kg_mean_preserved(self):
        """Mean preservation holds for non-default M_ext_kg values."""
        M = 3.7e54
        params = ExternalNodeParameters(M_ext_kg=M, node_mass_seed=5, node_mass_amplitude=0.7)
        masses = params.node_masses(26)
        np.testing.assert_allclose(masses.mean(), M, rtol=1e-12)

    def test_independent_of_global_rng(self):
        """node_masses() is independent of the global numpy RNG state."""
        params = self._make_params(seed=42, amplitude=0.5)
        # Draw from global RNG to change its state
        np.random.seed(999)
        _ = np.random.rand(1000)
        masses_before = params.node_masses(26)

        np.random.seed(0)
        _ = np.random.rand(5000)
        masses_after = params.node_masses(26)

        np.testing.assert_array_equal(masses_before, masses_after,
            err_msg="node_masses() should be independent of global numpy RNG")


# ---------------------------------------------------------------------------
# SimulationParameters field plumbing tests
# ---------------------------------------------------------------------------

class TestSimulationParametersFields:
    """SimulationParameters correctly stores and threads the new fields."""

    def test_default_fields_exist(self):
        p = SimulationParameters()
        assert hasattr(p, 'node_mass_seed')
        assert hasattr(p, 'node_mass_amplitude')
        assert p.node_mass_seed == 0
        assert p.node_mass_amplitude == 0.0

    def test_custom_fields_stored(self):
        p = SimulationParameters(node_mass_seed=77, node_mass_amplitude=0.8)
        assert p.node_mass_seed == 77
        assert p.node_mass_amplitude == 0.8

    def test_external_params_receives_fields(self):
        """SimulationParameters._calculate_derived threads fields into ExternalNodeParameters."""
        p = SimulationParameters(node_mass_seed=99, node_mass_amplitude=0.4)
        assert p.external_params.node_mass_seed == 99
        assert p.external_params.node_mass_amplitude == 0.4

    def test_external_params_default_amplitude_zero(self):
        """Default SimulationParameters => ExternalNodeParameters has amplitude 0."""
        p = SimulationParameters()
        assert p.external_params.node_mass_amplitude == 0.0


# ---------------------------------------------------------------------------
# HMEAGrid threading tests
# ---------------------------------------------------------------------------

class TestHMEAGridNodeMasses:
    """HMEAGrid._create_grid assigns masses from node_masses()."""

    def _make_grid(self, seed=0, amplitude=0.0, M_ext_kg=M_EXT_KG):
        params = ExternalNodeParameters(M_ext_kg=M_ext_kg, node_mass_seed=seed,
                                        node_mass_amplitude=amplitude)
        return HMEAGrid(node_params=params)

    def test_default_grid_uniform_masses(self):
        """amplitude=0.0 => all 26 grid node masses equal M_ext_kg."""
        grid = self._make_grid(amplitude=0.0)
        masses = grid.get_masses()
        assert masses.shape == (26,)
        np.testing.assert_array_equal(masses, M_EXT_KG)

    def test_grid_count_is_26(self):
        """HMEAGrid always creates 26 nodes."""
        grid = self._make_grid(amplitude=0.5)
        assert len(grid.nodes) == 26
        assert grid.get_masses().shape == (26,)

    def test_grid_masses_match_node_masses_method(self):
        """get_masses() == external_params.node_masses(26) for same (seed, amplitude)."""
        seed, amplitude = 42, 0.6
        params = ExternalNodeParameters(M_ext_kg=M_EXT_KG, node_mass_seed=seed,
                                        node_mass_amplitude=amplitude)
        grid = HMEAGrid(node_params=params)
        expected = params.node_masses(26)
        np.testing.assert_array_equal(grid.get_masses(), expected)

    def test_grid_mean_preservation_amplitude_nonzero(self):
        """mean(grid.get_masses()) == M_ext_kg for amplitude > 0."""
        grid = self._make_grid(seed=13, amplitude=1.0)
        np.testing.assert_allclose(grid.get_masses().mean(), M_EXT_KG, rtol=1e-12)

    def test_grid_strict_positivity(self):
        """All grid node masses > 0."""
        grid = self._make_grid(seed=5, amplitude=1.5)
        assert np.all(grid.get_masses() > 0)

    def test_sim_params_threads_into_grid(self):
        """SimulationParameters(node_mass_seed, node_mass_amplitude) -> HMEAGrid.get_masses()."""
        sim_params = SimulationParameters(
            M_value=500, S_value=25.0, n_particles=5, seed=1,
            node_mass_seed=7, node_mass_amplitude=0.4,
        )
        grid = HMEAGrid(node_params=sim_params.external_params)
        expected = sim_params.external_params.node_masses(26)
        np.testing.assert_array_equal(grid.get_masses(), expected)


# ---------------------------------------------------------------------------
# End-to-end sim-path invariants (confound guards)
# ---------------------------------------------------------------------------

class TestSimPathNodeMassInvariants:
    """Guard the invariants that make the node-mass knobs a CLEAN experiment.

    These protect against a future refactor that would let node_mass_amplitude /
    node_mass_seed perturb anything OTHER than the node masses (e.g. the particle
    realization), which would turn the measured amplitude->growth effect into a
    particle-cloud confound. Verified end-to-end through CosmologicalSimulation,
    not just the node_masses() unit.
    """

    def _build_sim(self, amplitude, nm_seed):
        from cosmo.constants import SimulationParameters
        from cosmo.simulation import CosmologicalSimulation
        from cosmo.factories import setup_simulation_context
        import io, contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            box, a_start, _ = setup_simulation_context(2.9, 13.8 - 2.9, 273, 10)
            params = SimulationParameters(
                M_value=1000, S_value=50, n_particles=120, seed=42,
                t_start_Gyr=2.9, t_duration_Gyr=13.8 - 2.9, n_steps=273,
                center_node_mass=1, mass_randomize=0.0,
                node_mass_seed=nm_seed, node_mass_amplitude=amplitude,
                init_distribution="grf",
            )
            sim = CosmologicalSimulation(params, box, a_start,
                                         use_external_nodes=True, use_dark_energy=False)
        return sim

    def test_total_external_mass_fixed_across_amplitude(self):
        """get_masses() sums to 26*M_ext_kg EXACTLY for amp=0 and amp>0 (no bug)."""
        s0 = self._build_sim(0.0, 42)
        s5 = self._build_sim(0.5, 42)
        M_ext = s0.sim_params.external_params.M_ext_kg
        np.testing.assert_allclose(s0.hmea_grid.get_masses().sum(), 26 * M_ext, rtol=1e-12)
        np.testing.assert_allclose(s5.hmea_grid.get_masses().sum(), 26 * M_ext, rtol=1e-12)
        # amplitude>0 really does spread the masses (not a no-op)
        assert s5.hmea_grid.get_masses().std() > 0

    def test_particle_realization_independent_of_amplitude(self):
        """Particle positions+velocities are byte-identical across node_mass_amplitude.

        node_masses() draws from its OWN default_rng AFTER the particle system is
        built, so it must not consume/advance the global RNG that seeds the cloud.
        """
        s0 = self._build_sim(0.0, 42)
        s5 = self._build_sim(0.5, 42)
        np.testing.assert_array_equal(s0.particles.get_positions(),
                                      s5.particles.get_positions())
        np.testing.assert_array_equal(s0.particles.get_velocities(),
                                      s5.particles.get_velocities())

    def test_particle_realization_independent_of_node_mass_seed(self):
        """Particle cloud is byte-identical across node_mass_seed at fixed amplitude.

        This is what makes a 'seed=42 vs seed=7' comparison a node-mass effect and
        NOT a lucky particle realization (confound guard).
        """
        s42 = self._build_sim(0.5, 42)
        s7 = self._build_sim(0.5, 7)
        np.testing.assert_array_equal(s42.particles.get_positions(),
                                      s7.particles.get_positions())
        np.testing.assert_array_equal(s42.particles.get_velocities(),
                                      s7.particles.get_velocities())


# ---------------------------------------------------------------------------
# Cache-key slug tests
# ---------------------------------------------------------------------------

class TestCacheKeySlug:
    """Distinct (seed, amplitude) specs produce distinct cache keys."""

    def _make_cache_name(self, seed, amplitude):
        """Call the REAL production cache-name builder so these tests guard the
        actual slug logic (not a hand-rolled copy that could drift from it)."""
        from cosmo.parameter_sweep import SweepConfig, build_cache_name
        config = SweepConfig(
            quick_search=True,
            objective='pantheon',
            node_mass_seed=seed,
            node_mass_amplitude=amplitude,
        )
        return build_cache_name(
            config, M_factor=800, S_val=25, centerM=1, seeds=[42],
        )

    def test_different_seeds_produce_different_cache_keys(self):
        """Two different seeds with same amplitude>0 => distinct cache keys."""
        key1 = self._make_cache_name(seed=1, amplitude=0.5)
        key2 = self._make_cache_name(seed=2, amplitude=0.5)
        assert key1 != key2, "Different seeds must produce different cache keys"

    def test_different_amplitudes_produce_different_cache_keys(self):
        """Two different amplitudes => distinct cache keys."""
        key1 = self._make_cache_name(seed=0, amplitude=0.3)
        key2 = self._make_cache_name(seed=0, amplitude=0.7)
        assert key1 != key2

    def test_uniform_amplitude_zero_no_slug_appended(self):
        """amplitude=0.0 does NOT append nmseed/nmamp slugs."""
        key = self._make_cache_name(seed=99, amplitude=0.0)
        assert 'nmseed' not in key
        assert 'nmamp' not in key

    def test_nonzero_amplitude_appends_slugs(self):
        """amplitude>0 appends nmseed and nmamp slugs."""
        key = self._make_cache_name(seed=7, amplitude=0.5)
        assert 'nmseed' in key
        assert 'nmamp' in key

    def test_anisotropic_key_differs_from_uniform_key(self):
        """amplitude>0 run can never reuse a uniform (amplitude=0) cache entry."""
        uniform_key = self._make_cache_name(seed=0, amplitude=0.0)
        aniso_key = self._make_cache_name(seed=0, amplitude=0.5)
        assert uniform_key != aniso_key


# ---------------------------------------------------------------------------
# SweepConfig field tests
# ---------------------------------------------------------------------------

class TestSweepConfigFields:
    """SweepConfig has node_mass_seed and node_mass_amplitude with correct defaults."""

    def test_default_fields(self):
        from cosmo.parameter_sweep import SweepConfig
        cfg = SweepConfig()
        assert cfg.node_mass_seed == 0
        assert cfg.node_mass_amplitude == 0.0

    def test_custom_fields(self):
        from cosmo.parameter_sweep import SweepConfig
        cfg = SweepConfig(node_mass_seed=42, node_mass_amplitude=0.3)
        assert cfg.node_mass_seed == 42
        assert cfg.node_mass_amplitude == 0.3
