"""
Pre-t_start HMEA tidal velocity boost.

The EdS-consistent ICs set v_i = H_EdS(t_start) * r_i (pure matter-only Hubble
flow). For M_ext > 0 the HMEA nodes have been pulling the cloud since the Big
Bang, so the cloud should ARRIVE at t_start moving slightly FASTER (a net outward
radial boost). CosmologicalSimulation._apply_pre_start_tidal_boost adds the
velocity the node-sum tidal field would have imparted over [Big Bang, t_start],
derived from the SAME node sum the integrator uses:

    dv_r(particle) = g_r(t_start) * (3/5) * t_start_seconds ,
    g_r = radial component of sum_nodes G m_node (r - r_node)/|r - r_node|^3 .

HARD INVARIANTS this guards:
  * The boost VANISHES as M_ext -> 0 (g_tid is linear in node mass), so the
    M_ext=0 == Einstein-de Sitter invariant is preserved EXACTLY.
  * The boost is monotone-increasing in M_ext (stronger nodes -> larger boost),
    so the initial RMS radial speed at M>0 exceeds the M=0 EdS value and grows
    with M.
"""

import io
import contextlib
import unittest

import numpy as np

import cosmo.simulation as simmod
from cosmo.constants import SimulationParameters
from cosmo.analysis import calculate_initial_conditions
from cosmo.simulation import CosmologicalSimulation


T_START = 2.9


def _build_sim(M_value, boost=True, n_particles=300, seed=42):
    """Construct a sim (which applies the boost in __init__) and return it.

    Stdout is suppressed (the sim is chatty). M_value=0 with use_external_nodes
    True still has zero tidal field, so the boost must be a no-op there.
    """
    simmod.velocity_cache = None
    ic = calculate_initial_conditions(T_START)
    sp = SimulationParameters(
        M_value=M_value,
        S_value=30.0,
        n_particles=n_particles,
        seed=seed,
        t_start_Gyr=T_START,
        t_duration_Gyr=13.8 - T_START,
        n_steps=280,
        mass_randomize=0.0,
        pre_start_tidal_boost=boost,
    )
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        sim = CosmologicalSimulation(
            sp, ic["box_size_Gpc"], ic["a_start"],
            use_external_nodes=(M_value > 0), use_dark_energy=False,
        )
    return sim


def _rms_radial_speed(sim):
    """RMS of the radial velocity component |v . r_hat| over all particles."""
    pos = sim.particles.get_positions()
    vel = sim.particles.get_velocities()
    r = np.linalg.norm(pos, axis=1, keepdims=True)
    r_hat = pos / np.where(r > 0, r, 1.0)
    v_r = np.sum(vel * r_hat, axis=1)
    return float(np.sqrt(np.mean(v_r ** 2)))


class TestPreStartTidalBoost(unittest.TestCase):
    def test_boost_vanishes_at_M0(self):
        """At M_ext=0 the boost ON and OFF velocities must be IDENTICAL.

        The pre-start tidal boost is only applied when use_external_nodes is True;
        with M=0 there is no external field, so the EdS ICs must be untouched and
        the M=0==EdS invariant preserved exactly. We compare the boost-on matter-
        only velocities to boost-off and require bit-for-bit equality.
        """
        sim_off = _build_sim(0, boost=False)
        sim_on = _build_sim(0, boost=True)
        v_off = sim_off.particles.get_velocities()
        v_on = sim_on.particles.get_velocities()
        # M=0 => no external nodes => boost is a strict no-op.
        self.assertFalse(
            sim_on.pre_start_tidal_boost,
            "Boost must NOT be active at M=0 (no external tidal field).",
        )
        np.testing.assert_allclose(
            v_on, v_off, rtol=0, atol=0,
            err_msg="At M=0 the boost changed the velocities -- it must vanish, "
                    "otherwise the M=0 == EdS invariant is broken.",
        )

    def test_boost_increases_initial_radial_speed_and_grows_with_M(self):
        """RMS radial speed must exceed the M=0 EdS value and grow with M_ext.

        The boost adds a net outward radial velocity that scales with the node
        masses, so RMS radial speed(M>0, boost on) > RMS radial speed(M=0), and is
        monotone-increasing in M.
        """
        v0 = _rms_radial_speed(_build_sim(0, boost=True))  # EdS baseline
        v_small = _rms_radial_speed(_build_sim(855, boost=True))
        v_mid = _rms_radial_speed(_build_sim(3000, boost=True))
        v_big = _rms_radial_speed(_build_sim(6000, boost=True))

        self.assertGreater(
            v_small, v0,
            f"M=855 RMS radial speed {v_small:.3e} must exceed the M=0 EdS value "
            f"{v0:.3e} (boost adds outward velocity).",
        )
        self.assertGreater(v_mid, v_small, "Boost must grow from M=855 to M=3000.")
        self.assertGreater(v_big, v_mid, "Boost must grow from M=3000 to M=6000.")

    def test_boost_off_matches_pure_eds_flow(self):
        """With the boost OFF, an M>0 run starts on pure EdS Hubble flow.

        Sanity check that the flag actually gates the term: boost-off RMS radial
        speed for M>0 must equal the M=0 (EdS) value to within peculiar-velocity
        noise, confirming no boost leaked in.
        """
        v0 = _rms_radial_speed(_build_sim(0, boost=True))
        v_off = _rms_radial_speed(_build_sim(3000, boost=False))
        rel = abs(v_off - v0) / v0
        self.assertLess(
            rel, 0.02,
            f"Boost-off M=3000 RMS radial speed {v_off:.3e} should match the EdS "
            f"value {v0:.3e} (rel {rel*100:.2f}%); a large gap means the flag did "
            f"not gate the boost.",
        )


if __name__ == "__main__":
    unittest.main()
