"""
THE matter-only / Einstein-de Sitter consistency invariant.

Core physics validation for the External-Node toy model: the particle cloud is a
Newtonian comoving patch of a homogeneous universe. With NO external tidal forces
(M_ext = 0) it MUST reproduce the analytic matter-only Einstein-de Sitter (EdS,
Omega_m = 1) expansion. This is the prerequisite for trusting every downstream
number: if M_ext=0 does not equal EdS, the expansion is coming from tuned initial
conditions, not physics.

This holds BY CONSTRUCTION when the initial conditions are self-consistent
(standard Newtonian cosmology):
  (i)  Hubble flow v_i = H_EdS(t_start) * r_i with H_EdS = 2/(3 t_start), and
  (ii) cloud density = EdS critical density rho_crit = 3 H_EdS^2 / (8 pi G).
Both are enabled together by SimulationParameters(eds_consistent=True) (the
default) whenever dark energy is off — see cosmo/particles.py.

Tests:
  * test_matter_only_growth_matches_eds: from-sim a(today)/a(t_start) matches the
    analytic EdS growth (t_today/t_start)^(2/3) to a tight tolerance.
  * test_matter_only_mu_matches_eds_not_lcdm: from-sim mu(z) sits on the EdS null
    (small RMS) and is clearly FARTHER from LCDM — the physically correct ordering.
  * test_eds_invariant_holds_at_lower_t_start: the invariant is t_start-independent
    (it follows from the self-consistent ICs), so it also holds at t_start=2.0 Gyr.
"""

import unittest

import numpy as np

import cosmo.simulation as simmod
from cosmo.constants import SimulationParameters
from cosmo.analysis import calculate_initial_conditions
from cosmo.factories import run_matter_only_simulation
from cosmo.sim_distance import sim_to_distance_modulus
from cosmo.distances import model_distance_modulus


def _eds_growth(t_start_Gyr: float, t_today_Gyr: float = 13.8) -> float:
    """Analytic EdS scale-factor growth a(today)/a(t_start) = (t_today/t_start)^(2/3)."""
    return (t_today_Gyr / t_start_Gyr) ** (2.0 / 3.0)


def _run_matter_only(t_start_Gyr: float, n_particles: int = 300, seed: int = 42,
                     init_distribution: str = "uniform_sphere"):
    """Run an M_ext=0 (matter-only) sim end-to-end and return (a, t_Gyr).

    init_distribution: "uniform_sphere" (default) or "grf". The EdS invariant must
    hold for GRF too (M=0 == EdS is geometry/clustering-independent: a(t) is the
    RMS RATIO, and the cloud carries the EdS critical density regardless of how the
    particles are arranged)."""
    # Fresh velocity cache so a stale calibration entry can never leak in.
    simmod.velocity_cache = None
    t_dur = 13.8 - t_start_Gyr
    ic = calculate_initial_conditions(t_start_Gyr)
    box_size_Gpc = ic["box_size_Gpc"]
    a_start = ic["a_start"]
    # dt = t_dur / n_steps must stay < 0.05 Gyr (leapfrog stability).
    n_steps = int(np.ceil(t_dur / 0.04))
    init_kwargs = {"Ng": 32} if init_distribution == "grf" else {}
    sim_params = SimulationParameters(
        M_value=0,
        n_particles=n_particles,
        seed=seed,
        t_start_Gyr=t_start_Gyr,
        t_duration_Gyr=t_dur,
        n_steps=n_steps,
        init_distribution=init_distribution,
        init_kwargs=init_kwargs,
        # eds_consistent defaults True -> self-consistent EdS ICs, no calibration.
    )
    res = run_matter_only_simulation(sim_params, box_size_Gpc, a_start, save_interval=10)
    return res["a"], res["t_Gyr"]


class TestMatterOnlyEdSConsistency(unittest.TestCase):
    """M_ext=0 must reproduce the analytic Einstein-de Sitter expansion."""

    def test_matter_only_growth_matches_eds(self):
        """From-sim total growth must match analytic EdS within 2%.

        Tolerance rationale: the self-consistent uniform-sphere ICs reproduce EdS
        EXACTLY in the continuum limit (verified analytically). The residual is
        N-body discreteness: finite N, gravitational softening, 100 km/s peculiar
        velocity noise and finite leapfrog dt. At N=300 these contribute well
        under 1%; 2% is a safe ceiling that still rules out the pre-fix ~9%
        over-expansion (which sat on LCDM, not EdS).
        """
        t_start = 2.9
        a, _t = _run_matter_only(t_start)
        sim_growth = a[-1] / a[0]
        eds_growth = _eds_growth(t_start)
        rel_err = abs(sim_growth - eds_growth) / eds_growth
        self.assertLess(
            rel_err, 0.02,
            f"M_ext=0 growth {sim_growth:.4f} deviates {rel_err*100:.2f}% from "
            f"analytic EdS {eds_growth:.4f} (must be < 2%). If this fails, the "
            f"matter-only sim is NOT reproducing Einstein-de Sitter -- the ICs "
            f"are not self-consistent."
        )

    def test_matter_only_mu_matches_eds_not_lcdm(self):
        """From-sim mu(z) must sit on the EdS null and be FARTHER from LCDM.

        The distance modulus offset is a free parameter (marginalized downstream),
        so we compare offset-removed RMS residuals. The physically correct result
        is: small residual vs EdS, larger residual vs LCDM. The pre-fix sim had
        this ordering inverted (closer to LCDM) -- the bug this test guards.
        """
        t_start = 2.9
        a, t = _run_matter_only(t_start)
        z_target = np.linspace(0.05, 2.0, 40)
        out = sim_to_distance_modulus(z_target, a, t, t_start)
        z = out["z"]
        mu_sim = out["mu"]

        def rms_vs(model: str) -> float:
            mu_model = model_distance_modulus(z, model)
            offset = np.mean(mu_sim - mu_model)  # marginalize the free mu offset
            return float(np.sqrt(np.mean((mu_sim - mu_model - offset) ** 2)))

        rms_eds = rms_vs("einstein_de_sitter")
        rms_lcdm = rms_vs("lcdm")

        # Must lie on the EdS null...
        self.assertLess(
            rms_eds, 0.05,
            f"M_ext=0 mu(z) RMS vs EdS = {rms_eds:.4f} mag (must be < 0.05). "
            f"The matter-only sim must reproduce the einstein_de_sitter curve."
        )
        # ...and clearly NOT on LCDM (correct physical ordering).
        self.assertGreater(
            rms_lcdm, 2.0 * rms_eds,
            f"M_ext=0 must be closer to EdS than LCDM: RMS vs EdS={rms_eds:.4f}, "
            f"vs LCDM={rms_lcdm:.4f}. If LCDM is closer, the expansion is coming "
            f"from tuned velocities (the original bug), not matter-only physics."
        )

    def test_eds_invariant_holds_at_lower_t_start(self):
        """The invariant is t_start-independent: also holds at t_start=2.0 Gyr.

        Self-consistent ICs reproduce EdS for ANY t_start, so the t_start fudge
        (previously floored at 2.9 Gyr to mask the velocity-calibration overshoot)
        can be lowered. We assert the growth invariant at an earlier start.
        """
        t_start = 2.0
        a, _t = _run_matter_only(t_start)
        sim_growth = a[-1] / a[0]
        eds_growth = _eds_growth(t_start)
        rel_err = abs(sim_growth - eds_growth) / eds_growth
        self.assertLess(
            rel_err, 0.03,
            f"At t_start={t_start} Gyr, M_ext=0 growth {sim_growth:.4f} deviates "
            f"{rel_err*100:.2f}% from EdS {eds_growth:.4f} (must be < 3%). The EdS "
            f"invariant should hold at lower t_start with self-consistent ICs."
        )

    def test_eds_invariant_holds_for_grf_init(self):
        """M_ext=0 == EdS must hold for the GRF initial distribution too.

        a(t) is the RMS RATIO and the cloud carries the EdS critical density
        regardless of HOW the particles are arranged, so a clustered (GRF) cloud
        with no external tidal forces still reproduces EdS. This guards the
        section-8 invariant (M=0 == EdS preserved for GRF). Tolerance is slightly
        looser than uniform_sphere because GRF clustering adds a touch more N-body
        discreteness noise at fixed N.
        """
        t_start = 2.9
        a, _t = _run_matter_only(t_start, init_distribution="grf")
        sim_growth = a[-1] / a[0]
        eds_growth = _eds_growth(t_start)
        rel_err = abs(sim_growth - eds_growth) / eds_growth
        self.assertLess(
            rel_err, 0.03,
            f"GRF M_ext=0 growth {sim_growth:.4f} deviates {rel_err*100:.2f}% from "
            f"analytic EdS {eds_growth:.4f} (must be < 3%). M=0 == EdS must be "
            f"distribution-independent."
        )


if __name__ == "__main__":
    unittest.main()
