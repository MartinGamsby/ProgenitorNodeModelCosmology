"""Unit tests for the MASS-WEIGHTED GRF init (init_distribution="grfmass").

The PF23 idea (user's): carry the BBKS density contrast in per-particle MASSES on
quasi-uniform positions, instead of Zel'dovich-crowding particle POSITIONS — removing the
geometric seed of the central-knot collapse while preserving the density field.
Contracts pinned here: mass-preserving total (PF1 M=0==EdS depends on the total),
positive weights, deterministic per seed, same sphere-support geometry as grf,
positions genuinely LESS crowded than grf, keyed==run (distinct cache key + support
sub-slug + init_kwargs threading), and mass_randomize mutual exclusion.
"""
import numpy as np
import pytest

from cosmo.initial_distributions import sample_grf, sample_grf_mass

BOX = 1.0e26  # m


class TestSampler:
    def test_deterministic_and_shapes(self):
        p1, w1 = sample_grf_mass(500, BOX, seed=11)
        p2, w2 = sample_grf_mass(500, BOX, seed=11)
        np.testing.assert_array_equal(p1, p2)
        np.testing.assert_array_equal(w1, w2)
        assert p1.shape == (500, 3) and w1.shape == (500,)
        p3, _ = sample_grf_mass(500, BOX, seed=12)
        assert not np.array_equal(p1, p3)

    def test_weights_positive_mean_near_one(self):
        _, w = sample_grf_mass(2000, BOX, seed=3, delta_rms=0.5)
        assert np.all(w > 0)
        # mean ~1 (weights = 1+delta over support cells; exact normalisation is the
        # caller's total-mass rescale)
        assert 0.7 < w.mean() < 1.3
        assert w.std() > 0.1, "weights should actually carry contrast"

    def test_sphere_support_radius_contract(self):
        p, _ = sample_grf_mass(2000, BOX, seed=5, support="sphere")
        r = np.linalg.norm(p, axis=1)
        assert r.max() <= (BOX / 2.0) / np.sqrt(3.0 / 5.0) * (1 + 1e-12)

    def test_positions_less_crowded_than_grf(self):
        """The whole point: grfmass positions are quasi-uniform (no Zel'dovich
        crowding). At FULL grid occupancy (no Poisson subsampling noise — n close
        to the available cell count) the coarse-cell occupancy variance must sit
        well below grf's (whose displacement crowds cells)."""
        n, Ng = 25000, 32          # ~all sphere-masked cells of the 32^3 grid
        pm, _ = sample_grf_mass(n, BOX, seed=7, support="sphere", Ng=Ng)
        pg = sample_grf(n, BOX, seed=7, support="sphere", Ng=Ng)

        def occupancy_cv(p, ncell=16):
            idx = np.floor((p / BOX + 0.5) * ncell).astype(int).clip(0, ncell - 1)
            flat = idx[:, 0] * ncell * ncell + idx[:, 1] * ncell + idx[:, 2]
            cnt = np.bincount(flat, minlength=ncell ** 3).astype(float)
            cnt = cnt[cnt > 0]
            return cnt.std() / cnt.mean()

        # grf's Zel'dovich displacement is deliberately MILD (~0.5 cell RMS), so the
        # positional-crowding gap is real but not dramatic: measured 0.368 vs 0.486
        # (24% lower) at ncell=16. Assert a stable subset of that margin.
        assert occupancy_cv(pm) < 0.85 * occupancy_cv(pg), (
            "grfmass positions are not meaningfully less crowded than grf")

    def test_bad_support_raises(self):
        with pytest.raises(ValueError):
            sample_grf_mass(100, BOX, seed=0, support="cube")


class TestParticleSystem:
    def _ps(self, **kw):
        from cosmo.particles import ParticleSystem
        args = dict(n_particles=300, box_size_m=BOX, total_mass_kg=1.0e53,
                    a_start=0.2, use_dark_energy=False, mass_randomize=0.0,
                    init_distribution="grfmass", eds_consistent=True,
                    t_start_Gyr=2.9)
        args.update(kw)
        np.random.seed(42)
        return ParticleSystem(**args)

    def test_total_mass_matches_uniform_sphere_twin(self):
        """PF1 contract: on the EdS-consistent path the cloud TOTAL is the
        EdS-critical mass (re-pinned after the RMS normalisation) — grfmass must
        change only the SPLIT, never the TOTAL. So its total must equal a
        uniform_sphere twin's total exactly, and the split must carry contrast."""
        m_gm = self._ps().get_masses()
        m_uni = self._ps(init_distribution="uniform_sphere").get_masses()
        assert m_gm.sum() == pytest.approx(m_uni.sum(), rel=1e-9), (
            "grfmass changed the cloud TOTAL mass (PF1 violation)")
        assert m_gm.std() / m_gm.mean() > 0.1, "masses should carry the density contrast"
        assert np.all(m_gm > 0)
        assert m_uni.std() == 0.0, "twin check: uniform_sphere masses are equal"

    def test_mass_randomize_conflict_raises(self):
        with pytest.raises(ValueError):
            self._ps(mass_randomize=0.5)

    def test_deterministic_per_seed(self):
        a = self._ps().get_masses()
        b = self._ps().get_masses()
        np.testing.assert_array_equal(a, b)


class TestKeyedEqualsRun:
    def test_cache_key_distinct_and_support_slugged(self):
        from cosmo.parameter_sweep import SweepConfig, build_cache_name
        k_grf = build_cache_name(SweepConfig(init_distribution="grf"), 100, 30, 1, [42])
        k_gm = build_cache_name(SweepConfig(init_distribution="grfmass"), 100, 30, 1, [42])
        k_uni = build_cache_name(SweepConfig(), 100, 30, 1, [42])
        assert "grfmassinit" in k_gm and k_gm != k_grf and k_gm != k_uni
        assert "sphsup" in k_gm, "grfmass must key the grf_support like grf does"

    def test_sweep_threads_init_kwargs(self):
        import sweep as sw
        cfg = dict(sw.DEFAULT_CONFIG) if hasattr(sw, "DEFAULT_CONFIG") else {}
        # go through the real cell->SweepConfig->SimulationParameters path
        from tests.test_overarching_sweep import DEFAULT_CONFIG
        c = dict(DEFAULT_CONFIG)
        c["init_distributions"] = ["grfmass"]
        cell = dict(M=100, amplitude=0.0, nm_seed=42, s_amplitude=0.0,
                    init="grfmass", geometry="cube26")
        sweep_cfg = sw._make_sweep_config_for_cell(cell, c)
        params = sw._build_sim_params(sweep_cfg, 100, 30, 1, 42)
        assert params.init_distribution == "grfmass"
        assert params.init_kwargs == {"support": "sphere"}, (
            "grfmass must thread grf_support via init_kwargs (keyed == run)")
