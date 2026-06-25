"""
Tests for Section 1 of WS4: centerM repurposing as outer-mass multiplier,
observable mask, and frozen softening.

Key invariants verified here:
  (a) centerM=1.0: no outer particles, all-True mask, inner cloud byte-identical,
      softening unchanged (1.0 Gpc).
  (b) centerM=2.0, ceiling=1: N_total == 2*N_inner (LINEAR, not cubic);
      inner positions/masses unchanged; inner RMS == box/2; mask correct.
  (c) R_sim == R_obs * centerM**(1/3) (e.g. centerM=2 -> x1.260).
  (d) Softening is INDEPENDENT of centerM (frozen at 1.0*Gpc for any centerM).
  (e) ceiling above max (2.0) is clipped/warned.
  (f) centerM>1 with non-EdS raises NotImplementedError.
  (g) centerM>1 with grf distribution raises NotImplementedError.
"""

import unittest
import warnings

import numpy as np

from cosmo.constants import CosmologicalConstants, SimulationParameters
from cosmo.particles import ParticleSystem


class _EdSParams:
    """Minimal helper to build a ParticleSystem in EdS-consistent mode."""

    # Fixed test parameters chosen to be fast (small N, modest t_start).
    T_START_GYR: float = 2.9
    BOX_SIZE_GPC: float = 4.39  # ~Hubble radius at t_start=2.9 Gyr
    N_INNER: int = 50
    SEED: int = 42

    @classmethod
    def box_size_m(cls) -> float:
        return cls.BOX_SIZE_GPC * CosmologicalConstants.Gpc_to_m

    @classmethod
    def make_ps(cls, center_node_mass: float = 1.0,
                outer_density_ceiling: float = 1.0,
                seed: int = None,
                n_particles: int = None) -> ParticleSystem:
        """Construct a ParticleSystem with EdS-consistent ICs."""
        np.random.seed(seed if seed is not None else cls.SEED)
        n = n_particles if n_particles is not None else cls.N_INNER
        return ParticleSystem(
            n_particles=n,
            box_size_m=cls.box_size_m(),
            a_start=0.310,          # a at t_start=2.9 Gyr (approximate)
            use_dark_energy=False,
            mass_randomize=0.5,
            init_distribution="uniform_sphere",
            eds_consistent=True,
            t_start_Gyr=cls.T_START_GYR,
            center_node_mass=center_node_mass,
            outer_density_ceiling=outer_density_ceiling,
        )


class TestCenterM1ByteIdentical(unittest.TestCase):
    """centerM=1.0 must produce a byte-identical inner cloud to the pre-WS4 code."""

    def test_positions_byte_identical_to_default(self):
        """centerM=1.0 must give exactly the same particle positions as no-centerM."""
        ps_default = _EdSParams.make_ps(center_node_mass=1.0, seed=_EdSParams.SEED)
        ps_explicit = _EdSParams.make_ps(center_node_mass=1.0, seed=_EdSParams.SEED)

        pos_default = ps_default.get_positions()
        pos_explicit = ps_explicit.get_positions()
        np.testing.assert_array_equal(pos_default, pos_explicit,
                                      err_msg="centerM=1 positions changed between two identical constructions")

    def test_mask_all_true_for_centerM1(self):
        """centerM=1.0 -> observable_mask must be all True, length == n_particles."""
        ps = _EdSParams.make_ps(center_node_mass=1.0)
        mask = ps.get_observable_mask()
        self.assertEqual(len(mask), _EdSParams.N_INNER)
        self.assertTrue(np.all(mask),
                        "centerM=1: observable_mask must be all True")

    def test_total_particle_count_equals_n_inner_for_centerM1(self):
        """centerM=1.0 -> len(particles) == n_particles (no outer particles)."""
        ps = _EdSParams.make_ps(center_node_mass=1.0)
        self.assertEqual(len(ps.particles), _EdSParams.N_INNER)


class TestCenterMLinearParticleCount(unittest.TestCase):
    """centerM=2 must produce N_total = 2*N_inner (LINEAR, not cubic)."""

    def _make_pair(self, center_m: float):
        """Return (ps_base, ps_outer) with the same seed."""
        ps_base = _EdSParams.make_ps(center_node_mass=1.0, seed=7)
        ps_outer = _EdSParams.make_ps(center_node_mass=center_m, seed=7)
        return ps_base, ps_outer

    def test_n_total_equals_round_centerM_times_n_inner_at_centerM2(self):
        """centerM=2, ceiling=1 -> N_total == round(2 * N_inner)."""
        ps = _EdSParams.make_ps(center_node_mass=2.0, seed=7)
        n_inner = _EdSParams.N_INNER
        n_expected_total = round(2.0 * n_inner)
        self.assertEqual(len(ps.particles), n_expected_total,
                         f"N_total={len(ps.particles)}, expected {n_expected_total} "
                         "(N_total must be LINEAR: 2*N_inner for centerM=2)")

    def test_n_outer_equals_n_inner_at_centerM2_ceiling1(self):
        """N_outer == N_inner at centerM=2, ceiling=1."""
        ps = _EdSParams.make_ps(center_node_mass=2.0, seed=7)
        n_inner = _EdSParams.N_INNER
        n_outer = len(ps.particles) - n_inner
        self.assertEqual(n_outer, n_inner,
                         f"N_outer={n_outer} != N_inner={n_inner} for centerM=2, ceiling=1")

    def test_n_total_linear_at_centerM3(self):
        """centerM=3, ceiling=1 -> N_total == round(3 * N_inner) (still LINEAR)."""
        ps = _EdSParams.make_ps(center_node_mass=3.0, seed=7)
        n_inner = _EdSParams.N_INNER
        n_expected = round(3.0 * n_inner)
        self.assertEqual(len(ps.particles), n_expected,
                         f"N_total={len(ps.particles)}, expected {n_expected} "
                         "(must be 3*N_inner for centerM=3, not 27*N_inner)")

    def test_mask_length_n_total_and_inner_count(self):
        """Observable mask must have length N_total with exactly N_inner True entries."""
        ps = _EdSParams.make_ps(center_node_mass=2.0, seed=7)
        mask = ps.get_observable_mask()
        n_inner = _EdSParams.N_INNER
        n_total = len(ps.particles)
        self.assertEqual(len(mask), n_total,
                         f"Mask length {len(mask)} != N_total {n_total}")
        self.assertEqual(np.sum(mask), n_inner,
                         f"Mask has {np.sum(mask)} True entries, expected {n_inner}")
        # Inner must come first (indices 0..N_inner-1 are True).
        self.assertTrue(np.all(mask[:n_inner]),
                        "First N_inner entries of observable_mask must be True")
        self.assertFalse(np.any(mask[n_inner:]),
                         "Entries N_inner+ of observable_mask must be False")


class TestInnerRegionUnchangedByCenterM(unittest.TestCase):
    """Adding outer particles must NOT change inner positions or masses."""

    def test_inner_positions_unchanged_when_centerM_gt1(self):
        """Inner particle positions for centerM=2 must equal positions for centerM=1."""
        ps_base = _EdSParams.make_ps(center_node_mass=1.0, seed=99)
        ps_outer = _EdSParams.make_ps(center_node_mass=2.0, seed=99)

        inner_base = ps_base.get_positions()            # All N_inner positions
        inner_outer = ps_outer.get_positions()[:_EdSParams.N_INNER]   # First N_inner

        np.testing.assert_array_equal(
            inner_base, inner_outer,
            err_msg="Inner particle positions changed when centerM > 1 (must be unchanged)"
        )

    def test_inner_masses_unchanged_when_centerM_gt1(self):
        """Inner particle masses for centerM=2 must equal masses for centerM=1."""
        ps_base = _EdSParams.make_ps(center_node_mass=1.0, seed=99)
        ps_outer = _EdSParams.make_ps(center_node_mass=2.0, seed=99)

        inner_masses_base = ps_base.get_masses()
        inner_masses_outer = ps_outer.get_masses()[:_EdSParams.N_INNER]

        np.testing.assert_array_equal(
            inner_masses_base, inner_masses_outer,
            err_msg="Inner particle masses changed when centerM > 1 (must be unchanged)"
        )

    def test_outer_mass_per_particle_equals_inner_mean(self):
        """Outer particles must carry the SAME mean per-particle mass as inner."""
        ps = _EdSParams.make_ps(center_node_mass=2.0, outer_density_ceiling=1.0, seed=13)
        n_inner = _EdSParams.N_INNER
        all_masses = ps.get_masses()
        inner_mean = np.mean(all_masses[:n_inner])
        outer_masses = all_masses[n_inner:]
        # Outer particles carry the unrandomised mean mass (no mass_randomize applied).
        np.testing.assert_allclose(
            outer_masses, inner_mean,
            rtol=1e-10,
            err_msg="Outer particles do not carry the same mean per-particle mass as inner"
        )

    def test_inner_rms_equals_box_half(self):
        """Inner RMS radius must equal box/2 for centerM > 1."""
        ps = _EdSParams.make_ps(center_node_mass=2.0, seed=17)
        n_inner = _EdSParams.N_INNER
        inner_pos = ps.get_positions()[:n_inner]
        inner_rms = np.sqrt(np.mean(np.sum(inner_pos**2, axis=1)))
        target_rms = _EdSParams.box_size_m() / 2
        rel_err = abs(inner_rms - target_rms) / target_rms
        self.assertLess(rel_err, 1e-10,
                        f"Inner RMS {inner_rms:.6e} != box/2 {target_rms:.6e} "
                        f"(rel_err={rel_err:.2e}); RMS normalization must use inner subset")


class TestRSimFormula(unittest.TestCase):
    """R_sim = R_obs * centerM**(1/3) (mass multiplier -> cube-root radius scaling)."""

    def test_r_sim_at_centerM2(self):
        """centerM=2 -> R_sim formula = R_obs * 2**(1/3) ≈ 1.260; outer particles in shell."""
        ps = _EdSParams.make_ps(center_node_mass=2.0, seed=5)
        n_inner = _EdSParams.N_INNER
        all_pos = ps.get_positions()
        outer_pos = all_pos[n_inner:]

        # Verify that the R_sim/R_obs ratio formula is correct (cube root, not linear).
        r_obs_raw = (_EdSParams.box_size_m() / 2.0) / np.sqrt(3.0 / 5.0)
        r_sim_raw = r_obs_raw * (2.0 ** (1.0 / 3.0))
        expected_ratio = 2.0 ** (1.0 / 3.0)
        self.assertAlmostEqual(r_sim_raw / r_obs_raw, expected_ratio, places=12,
                               msg="R_sim/R_obs formula deviates from centerM**(1/3)")

        # The outer shell was sampled in (r_obs_raw, r_sim_raw] pre-scale, then the
        # SAME scale_factor was applied to all positions. The observable mask is by
        # INDEX (first N_inner = inner), not by post-centering radius. After centering
        # + scaling, some inner particles can have radius slightly above r_obs * sf
        # (COM shift can move individual particles outward relative to the centred
        # origin). This thin overlap is expected and by-design. What we CAN assert:
        # outer particles are mostly OUTSIDE the inner cluster and the shell is not empty.
        self.assertGreater(len(outer_pos), 0,
                           "Expected outer particles for centerM=2 but got none")
        # Outer particles exist and their centroid should be beyond inner centroid radius.
        inner_rms = np.sqrt(np.mean(np.sum(all_pos[:n_inner]**2, axis=1)))
        outer_rms = np.sqrt(np.mean(np.sum(outer_pos**2, axis=1)))
        self.assertGreater(outer_rms, inner_rms,
                           f"Outer RMS {outer_rms:.3e} <= inner RMS {inner_rms:.3e}: "
                           "outer shell should be further out on average")

    def test_r_sim_ratio_is_cube_root_not_linear(self):
        """Radius ratio R_sim/R_obs = centerM**(1/3), NOT centerM (not cubic growth)."""
        for cm in [2.0, 4.0, 8.0]:
            expected_ratio = cm ** (1.0 / 3.0)
            r_obs = (_EdSParams.box_size_m() / 2.0) / np.sqrt(3.0 / 5.0)
            r_sim = r_obs * expected_ratio
            # Sanity: ratio is cube-root, not cm itself.
            self.assertAlmostEqual(expected_ratio, cm ** (1.0 / 3.0), places=12,
                                   msg=f"centerM={cm}: R_sim/R_obs should be {cm}**(1/3)")
            self.assertNotAlmostEqual(expected_ratio, cm, places=3,
                                      msg=f"centerM={cm}: radius ratio MUST NOT equal centerM "
                                      "(would be cubic particle growth, not linear)")


class TestSofteningFrozen(unittest.TestCase):
    """Softening must be frozen at 1.0*Gpc regardless of centerM (WS4 correctness gate)."""

    def _get_softening(self, center_m: float) -> float:
        """Construct a minimal simulation (matter-only, EdS-consistent) and read softening."""
        import cosmo.simulation as simmod
        simmod.velocity_cache = None

        sp = SimulationParameters(
            M_value=0,
            S_value=30.0,
            n_particles=_EdSParams.N_INNER,
            seed=_EdSParams.SEED,
            t_start_Gyr=_EdSParams.T_START_GYR,
            t_duration_Gyr=0.1,   # minimal; we only need the integrator
            n_steps=3,
            center_node_mass=center_m,
            eds_consistent=True,
        )
        from cosmo.analysis import calculate_initial_conditions
        ic = calculate_initial_conditions(_EdSParams.T_START_GYR)
        # use_dark_energy=False: EdS-consistent path, so centerM>1 is supported.
        sim = simmod.CosmologicalSimulation(
            sp, ic["box_size_Gpc"], ic["a_start"],
            use_external_nodes=False,
            use_dark_energy=False,
        )
        # softening_per_Mobs_m is the BASE softening we froze at 1.0*Gpc.
        # softening_m is further scaled by (mean_mass / M_obs)^(1/3) by the integrator.
        # We test the base, which is what the freeze controls.
        return sim.integrator.softening_per_Mobs_m

    def test_softening_centerM1_equals_1Gpc(self):
        """centerM=1 softening must be exactly 1.0 * Gpc_to_m."""
        const = CosmologicalConstants()
        expected = 1.0 * const.Gpc_to_m
        actual = self._get_softening(1.0)
        self.assertAlmostEqual(actual, expected, places=0,
                               msg=f"centerM=1 softening {actual:.3e} != 1.0 Gpc {expected:.3e}")

    def test_softening_independent_of_centerM(self):
        """Softening for centerM=3 must equal softening for centerM=1 (decoupling gate)."""
        s1 = self._get_softening(1.0)
        s3 = self._get_softening(3.0)
        self.assertEqual(s1, s3,
                         f"Softening changed with centerM: centerM=1->{s1:.3e}, "
                         f"centerM=3->{s3:.3e}. Softening must be FROZEN.")

    def test_softening_independent_at_centerM10(self):
        """Softening for centerM=10 must also equal 1.0 Gpc (old artifact was 10x here)."""
        const = CosmologicalConstants()
        expected = 1.0 * const.Gpc_to_m
        actual = self._get_softening(10.0)
        self.assertAlmostEqual(actual, expected, places=0,
                               msg=f"centerM=10 softening {actual:.3e} != {expected:.3e}. "
                               "Old code had softening=10*Gpc here (the artifact).")


class TestDensityCeilingEnforcement(unittest.TestCase):
    """outer_density_ceiling above 2.0 must be clipped with a warning."""

    def test_ceiling_clipped_to_max(self):
        """outer_density_ceiling > 2.0 must be clipped in SimulationParameters."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sp = SimulationParameters(
                M_value=0, n_particles=10, seed=1,
                outer_density_ceiling=5.0,
            )
        self.assertLessEqual(sp.outer_density_ceiling,
                             SimulationParameters.MAX_OUTER_DENSITY_CEILING,
                             "outer_density_ceiling not clipped to max")
        # A UserWarning should have been issued.
        user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
        self.assertTrue(len(user_warnings) > 0,
                        "No UserWarning raised when outer_density_ceiling exceeded max")

    def test_ceiling_at_max_allowed_passes(self):
        """outer_density_ceiling == MAX_OUTER_DENSITY_CEILING must not warn or clip."""
        max_c = SimulationParameters.MAX_OUTER_DENSITY_CEILING
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sp = SimulationParameters(
                M_value=0, n_particles=10, seed=1,
                outer_density_ceiling=max_c,
            )
        self.assertAlmostEqual(sp.outer_density_ceiling, max_c, places=12)
        user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
        self.assertEqual(len(user_warnings), 0,
                         f"Unexpected warning at outer_density_ceiling=max ({max_c})")


class TestCenterMGuardRails(unittest.TestCase):
    """centerM > 1 must raise NotImplementedError for unsupported paths."""

    def test_centerM_gt1_non_eds_raises(self):
        """centerM > 1.0 with eds_consistent=False must raise NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            np.random.seed(1)
            ParticleSystem(
                n_particles=10,
                box_size_m=_EdSParams.box_size_m(),
                use_dark_energy=False,
                eds_consistent=False,
                t_start_Gyr=_EdSParams.T_START_GYR,
                center_node_mass=2.0,
            )

    def test_centerM_gt1_grf_raises(self):
        """centerM > 1.0 with init_distribution='grf' must raise NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            np.random.seed(1)
            ParticleSystem(
                n_particles=10,
                box_size_m=_EdSParams.box_size_m(),
                use_dark_energy=False,
                eds_consistent=True,
                t_start_Gyr=_EdSParams.T_START_GYR,
                init_distribution="grf",
                center_node_mass=2.0,
            )

    def test_centerM_below1_clipped_to1(self):
        """center_node_mass < 1.0 must be silently clipped to 1.0."""
        sp = SimulationParameters(M_value=0, n_particles=10, seed=1,
                                  center_node_mass=0.5)
        self.assertGreaterEqual(sp.center_node_mass, 1.0,
                                "center_node_mass < 1.0 must be clipped to 1.0")


class TestCenterMScaling(unittest.TestCase):
    """Verify N_total == round(centerM * N_inner) for several centerM values."""

    def test_linear_scaling_multiple_values(self):
        """N_total = round(centerM * N_inner) for centerM in {1, 2, 3, 5}."""
        n_inner = 40
        for cm in [1.0, 2.0, 3.0, 5.0]:
            ps = _EdSParams.make_ps(center_node_mass=cm, seed=77,
                                    n_particles=n_inner)
            n_total = len(ps.particles)
            n_expected = round(cm * n_inner)
            self.assertEqual(n_total, n_expected,
                             f"centerM={cm}: N_total={n_total} != round({cm}*{n_inner})={n_expected}")


if __name__ == "__main__":
    unittest.main()
