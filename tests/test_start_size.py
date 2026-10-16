"""
Section 6 — start_size_scale (falsifiable initial-size lever).

start_size_scale multiplies the LCDM-implied INITIAL cloud size BEFORE particles
are built (CosmologicalSimulation.__init__). The subtlety this section relies on:

  * a(t) is computed as an RMS RATIO (rms(t)/rms(0)), so a UNIFORM rescale of the
    whole cloud cancels -> a pure size offset is a no-op (and the mu(z) pipeline
    divides out the absolute size too).
  * Under eds_consistent the EdS-critical cloud MASS = rho_crit * V(box), so
    scaling the box scales the mass with VOLUME -> the DENSITY is unchanged ->
    at M_ext=0 the dynamics are IDENTICAL EdS for ANY size (M=0 == EdS preserved).
  * The FALSIFIABLE effect appears only at M_ext>0: the nodes keep their UNSCALED
    spacing S (and the softening is frozen), so a bigger/smaller cloud spans a
    different fraction of S -> different differential tidal shear across it ->
    the a(t) SHAPE moves (NOT just an offset).

Invariants tested here:
  1. start_size_scale=1.0 -> byte-identical a(t) / positions / masses to NOT
     passing the knob (the headline backward-compat test).
  2. M_ext=0 -> a(t) ~ EdS for ANY size in {0.5, 1.0, 2.0}.
  3. M_ext>0 -> start_size_scale changes the NORMALIZED a(t) SHAPE (not just a
     multiplicative offset) -> it is a real physical lever.
  4. Cache slug: 'ssz' appears only when != 1.0; default keys unchanged; distinct
     values -> distinct keys.
  5. Threading: SimulationParameters stores it; SweepConfig default; sweep.py
     _make_sweep_config_for_cell / _make_sim_callback pass it; CLI round-trip.
  6. Invalid (<= 0) raises ValueError.
"""

import unittest

import numpy as np

import cosmo.simulation as simmod
from cosmo.constants import SimulationParameters
from cosmo.analysis import calculate_initial_conditions
from cosmo.factories import (
    run_matter_only_simulation,
    run_external_node_simulation,
)


def _eds_growth(t_start_Gyr: float, t_today_Gyr: float = 13.8) -> float:
    """Analytic EdS scale-factor growth a(today)/a(t_start) = (t_today/t_start)^(2/3)."""
    return (t_today_Gyr / t_start_Gyr) ** (2.0 / 3.0)


def _tiny_params(M_value, start_size_scale=1.0, t_start=2.9, n_particles=60,
                 seed=42, n_steps=None):
    """Build a TINY SimulationParameters for fast a(t) tests (dt < 0.05 Gyr)."""
    t_dur = 13.8 - t_start
    if n_steps is None:
        # dt = t_dur / n_steps must stay < 0.05 Gyr (leapfrog stability).
        n_steps = int(np.ceil(t_dur / 0.04))
    return SimulationParameters(
        M_value=M_value,
        S_value=24.0,
        n_particles=n_particles,
        seed=seed,
        t_start_Gyr=t_start,
        t_duration_Gyr=t_dur,
        n_steps=n_steps,
        mass_randomize=0.0,
        start_size_scale=start_size_scale,
        # eds_consistent defaults True -> self-consistent EdS ICs.
    )


def _run_matter_only(start_size_scale=1.0, t_start=2.9, n_particles=60, seed=42):
    """Run an M_ext=0 (matter-only) sim and return (a, t_Gyr)."""
    simmod.velocity_cache = None  # fresh cache: no stale calibration can leak in
    ic = calculate_initial_conditions(t_start)
    box_size_Gpc = ic["box_size_Gpc"]
    a_start = ic["a_start"]
    sim_params = _tiny_params(0, start_size_scale=start_size_scale,
                              t_start=t_start, n_particles=n_particles, seed=seed)
    res = run_matter_only_simulation(sim_params, box_size_Gpc, a_start, save_interval=10)
    return res["a"], res["t_Gyr"]


def _run_external(M_value, start_size_scale=1.0, t_start=2.9, n_particles=60, seed=42):
    """Run an External-Node (M_ext>0) sim and return (a, t_Gyr)."""
    simmod.velocity_cache = None
    ic = calculate_initial_conditions(t_start)
    box_size_Gpc = ic["box_size_Gpc"]
    a_start = ic["a_start"]
    sim_params = _tiny_params(M_value, start_size_scale=start_size_scale,
                              t_start=t_start, n_particles=n_particles, seed=seed)
    res = run_external_node_simulation(sim_params, box_size_Gpc, a_start, save_interval=10)
    return res["a"], res["t_Gyr"]


class TestDefaultByteIdentical(unittest.TestCase):
    """start_size_scale=1.0 must be byte-identical to the pre-knob behaviour."""

    def test_default_a_t_matches_omitting_the_knob(self):
        """a(t), box, positions and masses must match the no-knob path EXACTLY."""
        t_start = 2.9
        ic = calculate_initial_conditions(t_start)
        box_size_Gpc = ic["box_size_Gpc"]
        a_start = ic["a_start"]
        t_dur = 13.8 - t_start
        n_steps = int(np.ceil(t_dur / 0.04))

        # (a) Params WITHOUT passing the knob (uses the dataclass default).
        p_default = SimulationParameters(
            M_value=200, S_value=24.0, n_particles=60, seed=42,
            t_start_Gyr=t_start, t_duration_Gyr=t_dur, n_steps=n_steps,
            mass_randomize=0.0,
        )
        # (b) Params WITH start_size_scale explicitly = 1.0.
        p_explicit = SimulationParameters(
            M_value=200, S_value=24.0, n_particles=60, seed=42,
            t_start_Gyr=t_start, t_duration_Gyr=t_dur, n_steps=n_steps,
            mass_randomize=0.0, start_size_scale=1.0,
        )

        simmod.velocity_cache = None
        sim_a = simmod.CosmologicalSimulation(p_default, box_size_Gpc, a_start,
                                              use_external_nodes=True, use_dark_energy=False)
        pos_a = sim_a.particles.get_positions()
        mass_a = sim_a.particles.get_masses()

        simmod.velocity_cache = None
        sim_b = simmod.CosmologicalSimulation(p_explicit, box_size_Gpc, a_start,
                                              use_external_nodes=True, use_dark_energy=False)
        pos_b = sim_b.particles.get_positions()
        mass_b = sim_b.particles.get_masses()

        self.assertEqual(sim_a.box_size_Gpc, box_size_Gpc,
                         "default start_size_scale must leave box_size_Gpc unchanged")
        self.assertEqual(sim_b.box_size_Gpc, box_size_Gpc,
                         "start_size_scale=1.0 must leave box_size_Gpc unchanged")
        np.testing.assert_array_equal(
            pos_a, pos_b,
            "start_size_scale=1.0 must produce byte-identical particle positions")
        np.testing.assert_array_equal(
            mass_a, mass_b,
            "start_size_scale=1.0 must produce byte-identical particle masses")

    def test_default_full_run_a_t_identical(self):
        """Full-run a(t) at scale=1.0 must equal the no-knob run exactly."""
        a_default, _ = _run_external(200, start_size_scale=1.0)
        # Re-run with the same config but constructed from the dataclass default.
        simmod.velocity_cache = None
        t_start = 2.9
        ic = calculate_initial_conditions(t_start)
        t_dur = 13.8 - t_start
        n_steps = int(np.ceil(t_dur / 0.04))
        p = SimulationParameters(
            M_value=200, S_value=24.0, n_particles=60, seed=42,
            t_start_Gyr=t_start, t_duration_Gyr=t_dur, n_steps=n_steps,
            mass_randomize=0.0,
        )
        res = run_external_node_simulation(p, ic["box_size_Gpc"], ic["a_start"], 10)
        np.testing.assert_array_equal(
            a_default, res["a"],
            "scale=1.0 a(t) must be byte-identical to the no-knob a(t)")


class TestM0EdSForAnySize(unittest.TestCase):
    """M_ext=0 must reproduce EdS growth for ANY initial size."""

    def test_m0_growth_eds_for_scales(self):
        t_start = 2.9
        eds = _eds_growth(t_start)
        for scale in (0.5, 1.0, 2.0):
            a, _ = _run_matter_only(start_size_scale=scale, t_start=t_start)
            sim_growth = a[-1] / a[0]
            rel_err = abs(sim_growth - eds) / eds
            self.assertLess(
                rel_err, 0.03,
                f"At start_size_scale={scale}, M_ext=0 growth {sim_growth:.4f} "
                f"deviates {rel_err*100:.2f}% from EdS {eds:.4f} (must be < 3%). "
                f"Scaling the box scales the EdS-critical mass with volume so the "
                f"density stays critical -> M=0 == EdS must hold at any size.")

    def test_m0_canonical_scale1_byte_identical(self):
        """The canonical M=0==EdS at scale=1.0 must equal the no-knob a(t)."""
        a_scaled, _ = _run_matter_only(start_size_scale=1.0)
        # Build the no-knob equivalent.
        simmod.velocity_cache = None
        t_start = 2.9
        ic = calculate_initial_conditions(t_start)
        t_dur = 13.8 - t_start
        n_steps = int(np.ceil(t_dur / 0.04))
        p = SimulationParameters(
            M_value=0, S_value=24.0, n_particles=60, seed=42,
            t_start_Gyr=t_start, t_duration_Gyr=t_dur, n_steps=n_steps,
            mass_randomize=0.0,
        )
        res = run_matter_only_simulation(p, ic["box_size_Gpc"], ic["a_start"], 10)
        np.testing.assert_array_equal(
            a_scaled, res["a"],
            "M=0 a(t) at scale=1.0 must be byte-identical to the no-knob run")


class TestChangesAtShapeAtMPositive(unittest.TestCase):
    """At M_ext>0, start_size_scale must change the NORMALIZED a(t) SHAPE."""

    @staticmethod
    def _shape(a):
        """Normalized-to-today curve a(t)/a(t)[-1] (divides out any pure offset)."""
        a = np.asarray(a, dtype=float)
        return a / a[-1]

    def test_shape_differs_between_0p8_1p0_1p2(self):
        M = 2000  # strong enough tidal coupling to register at small N
        a08, _ = _run_external(M, start_size_scale=0.8)
        a10, _ = _run_external(M, start_size_scale=1.0)
        a12, _ = _run_external(M, start_size_scale=1.2)

        s08 = self._shape(a08)
        s10 = self._shape(a10)
        s12 = self._shape(a12)

        # The normalized shapes must DIFFER beyond float noise: if start_size_scale
        # were a pure (divided-out) offset, these would be identical.
        d_08 = float(np.max(np.abs(s08 - s10)))
        d_12 = float(np.max(np.abs(s12 - s10)))
        self.assertGreater(
            d_08, 1e-4,
            f"start_size_scale=0.8 must move the normalized a(t) SHAPE vs 1.0 "
            f"(max |Δ shape| = {d_08:.3e}); if not, the knob is a no-op offset.")
        self.assertGreater(
            d_12, 1e-4,
            f"start_size_scale=1.2 must move the normalized a(t) SHAPE vs 1.0 "
            f"(max |Δ shape| = {d_12:.3e}); if not, the knob is a no-op offset.")

    def test_growth_factor_changes_with_size(self):
        """The total growth a[-1]/a[0] must also respond to the size lever."""
        M = 2000
        a08, _ = _run_external(M, start_size_scale=0.8)
        a10, _ = _run_external(M, start_size_scale=1.0)
        a12, _ = _run_external(M, start_size_scale=1.2)
        g08, g10, g12 = a08[-1] / a08[0], a10[-1] / a10[0], a12[-1] / a12[0]
        self.assertNotAlmostEqual(
            g08, g10, places=4,
            msg=f"growth at scale=0.8 ({g08:.5f}) must differ from scale=1.0 ({g10:.5f})")
        self.assertNotAlmostEqual(
            g12, g10, places=4,
            msg=f"growth at scale=1.2 ({g12:.5f}) must differ from scale=1.0 ({g10:.5f})")


class TestCacheSlug(unittest.TestCase):
    """The 'ssz' cache slug must appear only when start_size_scale != 1.0."""

    def _cfg(self, start_size_scale=1.0):
        from cosmo.parameter_sweep import SweepConfig
        return SweepConfig(
            quick_search=False, many_search=3, leet_search=False,
            search_center_mass=False, t_start_Gyr=2.9, t_duration_Gyr=10.9,
            damping_factor=None, s_min_gpc=18, s_max_gpc=40, save_interval=10,
            objective="pantheon", start_size_scale=start_size_scale,
        )

    def _key(self, start_size_scale=1.0):
        from cosmo.parameter_sweep import build_cache_name
        return build_cache_name(self._cfg(start_size_scale),
                                M_factor=1, S_val=20, centerM=1.0, seeds=[42])

    def test_default_no_slug(self):
        self.assertNotIn("ssz", self._key(1.0),
                         "start_size_scale=1.0 (default) must NOT add an 'ssz' slug")

    def test_non_default_adds_slug(self):
        self.assertIn("ssz", self._key(1.2),
                      "start_size_scale=1.2 must add an 'ssz' slug")

    def test_slug_value(self):
        self.assertIn("1.2ssz", self._key(1.2),
                      "start_size_scale=1.2 slug should contain '1.2ssz'")

    def test_distinct_values_distinct_keys(self):
        self.assertNotEqual(self._key(0.8), self._key(1.2),
                            "Different start_size_scale values must produce distinct keys")
        self.assertNotEqual(self._key(1.0), self._key(1.2),
                            "start_size_scale=1.0 and 1.2 must produce distinct keys")

    def test_default_key_unchanged_vs_absent(self):
        """A config with the field defaulted must key identically to one without it."""
        from cosmo.parameter_sweep import build_cache_name, SweepConfig
        cfg_default = SweepConfig(
            quick_search=False, many_search=3, leet_search=False,
            search_center_mass=False, t_start_Gyr=2.9, t_duration_Gyr=10.9,
            damping_factor=None, s_min_gpc=18, s_max_gpc=40, save_interval=10,
            objective="pantheon",
        )
        key_default = build_cache_name(cfg_default, M_factor=1, S_val=20, centerM=1.0, seeds=[42])
        self.assertEqual(key_default, self._key(1.0),
                         "Defaulted start_size_scale must not change the cache key")


class TestThreading(unittest.TestCase):
    """start_size_scale must thread end-to-end (constants, sweep, CLI)."""

    def test_sim_params_stores_it(self):
        p = SimulationParameters(start_size_scale=1.3)
        self.assertAlmostEqual(p.start_size_scale, 1.3, places=10)

    def test_sim_params_default_is_1(self):
        p = SimulationParameters()
        self.assertAlmostEqual(p.start_size_scale, 1.0, places=10)

    def test_sweepconfig_default_is_1(self):
        from cosmo.parameter_sweep import SweepConfig
        self.assertAlmostEqual(SweepConfig().start_size_scale, 1.0, places=10)

    def test_sweep_cell_config_threads_it(self):
        """_make_sweep_config_for_cell must read start_size_scale off cfg."""
        import sweep as sweep_mod
        cell = dict(M=100, amplitude=0.0, nm_seed=42, s_amplitude=0.0,
                    init="uniform_sphere", geometry="cube26")
        cfg = dict(sweep_mod.DEFAULT_CONFIG)
        cfg["start_size_scale"] = 1.4
        sc = sweep_mod._make_sweep_config_for_cell(cell, cfg)
        self.assertAlmostEqual(sc.start_size_scale, 1.4, places=10)

    def test_sweep_sim_callback_passes_it(self):
        """_make_sim_callback must build SimulationParameters with the scale.

        We monkeypatch run_external_node_simulation to capture sim_params and
        confirm the start_size_scale propagated from the SweepConfig.
        """
        import sweep as sweep_mod
        cell = dict(M=100, amplitude=0.0, nm_seed=42, s_amplitude=0.0,
                    init="uniform_sphere", geometry="cube26")
        cfg = dict(sweep_mod.DEFAULT_CONFIG)
        cfg["start_size_scale"] = 1.6
        cfg["t_start_Gyr"] = 2.9
        sc = sweep_mod._make_sweep_config_for_cell(cell, cfg)

        captured = {}

        def _fake_run(sim_params, box, a_start, save_interval):
            captured["scale"] = sim_params.start_size_scale
            # Return a minimal object that results_to_sim_result can consume.
            return {
                "diameter_Gpc": np.array([1.0, 1.1]),
                "H_hubble": np.array([70.0, 65.0]),
                "max_radius_Gpc": np.array([1.0, 1.1]),
                "a": np.array([1.0, 1.1]),
                "t_Gyr": np.array([0.0, 1.0]),
            }

        orig = sweep_mod.run_external_node_simulation
        try:
            sweep_mod.run_external_node_simulation = _fake_run
            cb = sweep_mod._make_sim_callback(sc, box_size_Gpc=1.0, a_start=0.3)
            cb(100, 24, 1.0, [42])
        finally:
            sweep_mod.run_external_node_simulation = orig

        self.assertAlmostEqual(captured.get("scale"), 1.6, places=10,
                               msg="sweep _make_sim_callback must pass start_size_scale to the sim")

    def test_cli_roundtrip(self):
        import argparse
        from cosmo.cli import add_common_arguments, args_to_sim_params
        parser = argparse.ArgumentParser()
        add_common_arguments(parser)
        args = parser.parse_args(["--start-size-scale", "0.7"])
        p = args_to_sim_params(args)
        self.assertAlmostEqual(p.start_size_scale, 0.7, places=10)

    def test_cli_default(self):
        import argparse
        from cosmo.cli import add_common_arguments, args_to_sim_params
        parser = argparse.ArgumentParser()
        add_common_arguments(parser)
        args = parser.parse_args([])
        p = args_to_sim_params(args)
        self.assertAlmostEqual(p.start_size_scale, 1.0, places=10)


class TestInvalidSizeRejected(unittest.TestCase):
    """start_size_scale <= 0 must raise ValueError (documented behaviour)."""

    def test_zero_raises(self):
        with self.assertRaises(ValueError):
            SimulationParameters(start_size_scale=0.0)

    def test_negative_raises(self):
        with self.assertRaises(ValueError):
            SimulationParameters(start_size_scale=-1.0)


if __name__ == "__main__":
    unittest.main()
