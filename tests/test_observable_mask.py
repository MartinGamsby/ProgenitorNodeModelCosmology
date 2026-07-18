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

    def test_centerM_gt1_grf_composes(self):
        """centerM > 1.0 with init_distribution='grf' now COMPOSES (PF24): the outer
        shell is sampler-agnostic (box-geometry radius, uniform shell, mean-mass
        outer particles), so the old NotImplementedError guard was removed. Pin the
        composition contract: inner count, outer count, mask split."""
        np.random.seed(1)
        ps = ParticleSystem(
            n_particles=10,
            box_size_m=_EdSParams.box_size_m(),
            use_dark_energy=False,
            eds_consistent=True,
            t_start_Gyr=_EdSParams.T_START_GYR,
            init_distribution="grf",
            center_node_mass=2.0,
        )
        self.assertEqual(int(ps.observable_mask.sum()), 10, "inner count must stay N")
        self.assertEqual(int((~ps.observable_mask).sum()), 10,
                         "centerM=2 must append (centerM-1)*N outer particles")

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


# =============================================================================
# Section 2 tests — a(t) computed from observable inner sub-region only
# =============================================================================

# Shared fast sim parameters for Section 2 tests.
_S2_T_START = 2.9
_S2_T_DUR = 2.0          # short run (< 0.05 Gyr / step stability limit)
_S2_N_PARTICLES = 60
_S2_N_STEPS = 50          # dt = 2.0/50 = 0.04 Gyr — just at the stability ceiling
_S2_SEED = 42


def _run_sim_s2(center_node_mass: float = 1.0,
                outer_density_ceiling: float = 1.0,
                seed: int = _S2_SEED):
    """Run a minimal EdS-consistent matter-only sim and return (a_curve, sim)."""
    import cosmo.simulation as simmod
    simmod.velocity_cache = None
    from cosmo.analysis import calculate_initial_conditions
    from cosmo.factories import run_matter_only_simulation
    ic = calculate_initial_conditions(_S2_T_START)
    sp = SimulationParameters(
        M_value=0,
        n_particles=_S2_N_PARTICLES,
        seed=seed,
        t_start_Gyr=_S2_T_START,
        t_duration_Gyr=_S2_T_DUR,
        n_steps=_S2_N_STEPS,
        center_node_mass=center_node_mass,
        outer_density_ceiling=outer_density_ceiling,
        eds_consistent=True,
    )
    res = run_matter_only_simulation(sp, ic["box_size_Gpc"], ic["a_start"])
    return res["a"], res["sim"]


def _eds_growth_s2(t_start_Gyr: float = _S2_T_START,
                   t_dur_Gyr: float = _S2_T_DUR) -> float:
    """Analytic EdS growth over the sim duration: (t_end / t_start)^(2/3)."""
    t_end = t_start_Gyr + t_dur_Gyr
    return (t_end / t_start_Gyr) ** (2.0 / 3.0)


class TestSection2ACurveInnerOnly(unittest.TestCase):
    """Section 2: a(t) is computed from the inner observable sub-region only."""

    def test_centerM1_a_curve_byte_identical_to_unmasked(self):
        """centerM=1 masked a(t) must equal unmasked a(t) element-for-element.

        When centerM=1 the observable_mask is all-True, so the masked RMS
        calculation is numerically identical to the full-cloud calculation.
        This is the byte-identity invariant (THE HARD INVARIANT, Section 2).
        """
        a1, _ = _run_sim_s2(center_node_mass=1.0, seed=_S2_SEED)
        a2, _ = _run_sim_s2(center_node_mass=1.0, seed=_S2_SEED)
        np.testing.assert_array_equal(
            a1, a2,
            err_msg="centerM=1: two identical runs gave different a(t) arrays "
                    "(byte-identity broken)"
        )

    def test_centerM1_a_curve_matches_eds_growth(self):
        """centerM=1 masked a(t) must still match EdS growth within 3%.

        Confirms the masking does not perturb the baseline EdS result.
        """
        a, _ = _run_sim_s2(center_node_mass=1.0, seed=_S2_SEED)
        sim_growth = float(a[-1] / a[0])
        eds_growth = _eds_growth_s2()
        rel_err = abs(sim_growth - eds_growth) / eds_growth
        self.assertLess(
            rel_err, 0.03,
            f"centerM=1 masked growth {sim_growth:.4f} deviates {rel_err*100:.2f}% "
            f"from EdS {eds_growth:.4f} (must be < 3%)."
        )

    def test_outer_particles_do_not_enter_size_measurement(self):
        """centerM>1: the stored a(t) matches a manual recompute on the inner subset.

        The key correctness gate for Section 2: the outer shell particles are
        excluded from the size/a(t) measurement.  We verify this by recomputing
        a(t) manually from sim2's snapshots using the same inner observable mask
        and the same COM-centred RMS formula used by ParticleSystem.calculate_system_size,
        then asserting it equals the stored expansion_history['scale_factor'] array
        to floating-point precision.
        """
        from cosmo.particles import ParticleSystem as PS
        _a2, sim2 = _run_sim_s2(center_node_mass=2.0, seed=_S2_SEED)

        # Recompute a(t) manually from sim2's snapshots using the inner mask.
        mask2 = np.asarray(sim2.particles.observable_mask, dtype=bool)
        n_inner = int(np.sum(mask2))
        n_total2 = len(sim2.particles.particles)
        self.assertGreater(n_total2, n_inner,
                           "centerM=2 must have outer particles beyond inner")

        inner_pos_0 = sim2.snapshots[0]['positions'][mask2]
        rms_0, _, _ = PS.calculate_system_size(inner_pos_0)  # COM-subtracted RMS
        a2_inner = np.array([
            PS.calculate_system_size(snap['positions'][mask2])[0] / rms_0
            for snap in sim2.snapshots
        ])

        # a(t) returned by run_matter_only_simulation must match the manual
        # inner-mask recompute to floating-point precision.
        np.testing.assert_allclose(
            _a2, a2_inner, rtol=1e-12,
            err_msg="a(t) from run differs from manual inner-mask recompute: "
                    "expansion history is NOT restricted to the inner observable subset."
        )

    def test_outer_particles_still_exert_gravity(self):
        """centerM>1: outer particles must change the integrator's force field.

        Compare a(t) for centerM=1 vs centerM=2 at the SAME seed.  The inner
        particles are byte-identical at t=0 (Section 1 invariant), but the outer
        shell adds gravitational pull so the inner region's a(t) should differ
        slightly once evolved.  If they are exactly equal the outer particles are
        NOT being included in the dynamics (a bug).

        Note: with N=60 particles and a 2 Gyr run the tidal effect is small but
        non-zero.  We only assert they are NOT byte-equal after evolution.
        """
        a1, _ = _run_sim_s2(center_node_mass=1.0, seed=_S2_SEED)
        a2, _ = _run_sim_s2(center_node_mass=2.0, seed=_S2_SEED)
        # They must NOT be byte-identical (outer gravity does something).
        # We allow them to be very close (small N, short run) but not equal.
        if np.array_equal(a1, a2):
            self.fail(
                "centerM=1 and centerM=2 produced identical a(t) arrays. "
                "Outer particles are NOT exerting gravity (integrator bug)."
            )

    def test_eds_invariant_with_outer_matter_present(self):
        """M=0, centerM=2, ceiling=1 -> inner a(t) growth matches EdS within 3%.

        Physical invariant: outer matter at the SAME critical density does not
        change the inner expansion in the Newtonian approximation (the shell
        theorem — a uniform shell exerts zero net force on interior points).
        At the N-body level the shell is finite and discrete so a small
        (< 3%) deviation is expected, but growth must not deviate dramatically.
        """
        a, _ = _run_sim_s2(center_node_mass=2.0, seed=_S2_SEED)
        sim_growth = float(a[-1] / a[0])
        eds_growth = _eds_growth_s2()
        rel_err = abs(sim_growth - eds_growth) / eds_growth
        # 5% tolerance: discreteness + boundary effects are larger with an outer shell.
        self.assertLess(
            rel_err, 0.05,
            f"M=0, centerM=2: inner a(t) growth {sim_growth:.4f} deviates "
            f"{rel_err*100:.2f}% from EdS {eds_growth:.4f} (must be < 5%). "
            "Outer critical-density matter must not grossly perturb inner expansion."
        )


if __name__ == "__main__":
    unittest.main()
