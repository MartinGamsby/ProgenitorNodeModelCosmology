"""
Unit tests for GR Christoffel symbols (Phase 2).
Tests analytic Schwarzschild/FRW and numerical computation.
"""

import unittest
import numpy as np

from cosmo.constants import CosmologicalConstants
from cosmo.gr.metric import schwarzschild_metric, schwarzschild_radius_m
from cosmo.gr.christoffel import (
    christoffel_schwarzschild,
    christoffel_frw,
    christoffel_from_metric,
    christoffel_symmetry_check,
)

c = CosmologicalConstants.c
G = CosmologicalConstants.G
M_sun = CosmologicalConstants.M_sun

AU_m = 1.496e11


class TestChristoffelSchwarzschild(unittest.TestCase):
    """Tests for analytic Schwarzschild Christoffel symbols."""

    def setUp(self):
        self.r = 10 * AU_m
        self.M = M_sun
        self.gamma = christoffel_schwarzschild(self.r, self.M)
        self.r_s = schwarzschild_radius_m(self.M)

    def test_gamma_t_tr_textbook(self):
        """Gamma^t_tr = GM / (r^2 c^2 (1 - r_s/r))."""
        f = 1 - self.r_s / self.r
        expected = G * self.M / (self.r**2 * c**2 * f)
        self.assertAlmostEqual(
            self.gamma[0, 0, 1] / expected, 1.0, places=12
        )

    def test_gamma_r_tt_textbook(self):
        """Gamma^r_tt = GM (1-r_s/r) / r^2."""
        f = 1 - self.r_s / self.r
        expected = G * self.M * f / self.r**2
        self.assertAlmostEqual(
            self.gamma[1, 0, 0] / expected, 1.0, places=12
        )

    def test_lower_index_symmetry(self):
        """Torsion-free: Gamma^m_nl = Gamma^m_ln."""
        asym = christoffel_symmetry_check(self.gamma)
        self.assertEqual(asym, 0.0)

    def test_gravitational_vanish_large_r(self):
        """Mass-dependent Christoffel symbols -> 0 for r -> infinity.

        Pure coordinate Christoffels (1/r, -r) from spherical coords persist,
        so we check only M-dependent components: Gamma^t_tr, Gamma^r_tt, Gamma^r_rr.
        """
        r_huge = 1e30
        gamma = christoffel_schwarzschild(r_huge, self.M)
        self.assertAlmostEqual(gamma[0, 0, 1], 0.0, places=50)  # Gamma^t_tr ~ GM/(r^2 c^2)
        self.assertAlmostEqual(gamma[1, 0, 0], 0.0, places=20)  # Gamma^r_tt ~ GM c^2/r^2
        self.assertAlmostEqual(gamma[1, 1, 1], 0.0, places=50)  # Gamma^r_rr ~ GM/(r^2 c^2)

    def test_nonzero_independent_count(self):
        """Correct count of non-zero independent components.

        Schwarzschild at theta=pi/2 has these independent non-zero:
        Gamma^t_tr, Gamma^r_tt, Gamma^r_rr, Gamma^r_theta_theta,
        Gamma^r_phi_phi, Gamma^theta_r_theta, Gamma^phi_r_phi = 7 independent.
        With symmetry in lower indices: 10 non-zero entries total
        (Gamma^t_tr + Gamma^t_rt, Gamma^r_tt, Gamma^r_rr,
         Gamma^r_theta_theta, Gamma^r_phi_phi,
         Gamma^theta_r_theta + Gamma^theta_theta_r,
         Gamma^phi_r_phi + Gamma^phi_phi_r).
        """
        nonzero = np.count_nonzero(self.gamma)
        self.assertEqual(nonzero, 10)


class TestChristoffelFRW(unittest.TestCase):
    """Tests for analytic FRW Christoffel symbols."""

    def test_gamma_r_tr(self):
        """FRW: Gamma^r_tr = adot/a."""
        a, adot = 2.0, 0.5
        gamma = christoffel_frw(a, adot)
        self.assertAlmostEqual(gamma[1, 0, 1], adot / a, places=14)

    def test_frw_symmetry(self):
        """FRW Christoffel symbols are symmetric in lower indices."""
        gamma = christoffel_frw(a=1.5, adot=0.3)
        asym = christoffel_symmetry_check(gamma)
        self.assertEqual(asym, 0.0)

    def test_static_flat_all_zero(self):
        """Static flat FRW (a=1, adot=0, k=0) -> all zero except spatial."""
        gamma = christoffel_frw(a=1.0, adot=0.0, k=0)
        # With adot=0, time-space mixing terms vanish.
        # Only purely spatial Christoffel symbols from spherical coords remain.
        # Gamma^theta_r_theta=1, Gamma^phi_r_phi=1, Gamma^r_theta_theta=-1, Gamma^r_phi_phi=-1
        # Check that time-related components are zero
        self.assertEqual(gamma[0, 1, 1], 0.0)  # Gamma^t_rr
        self.assertEqual(gamma[1, 0, 1], 0.0)  # Gamma^r_tr
        self.assertEqual(gamma[2, 0, 2], 0.0)  # Gamma^theta_t_theta


class TestChristoffelNumerical(unittest.TestCase):
    """Tests for numerical Christoffel computation."""

    def test_numerical_matches_analytic_schwarzschild(self):
        """Numerical Christoffel matches analytic for Schwarzschild."""
        M = M_sun
        r_s = schwarzschild_radius_m(M)
        r = 100 * r_s  # stronger field for better numerical accuracy
        dr = r * 1e-7  # small perturbation for numerical derivative

        g = schwarzschild_metric(r, M)

        # Build dg[mu,nu,rho] = partial_rho g_{mu nu}
        # Only r-derivatives (rho=1) are non-zero for Schwarzschild
        dg = np.zeros((4, 4, 4))

        g_plus = schwarzschild_metric(r + dr, M)
        g_minus = schwarzschild_metric(r - dr, M)

        for mu in range(4):
            for nu in range(4):
                dg[mu, nu, 1] = (g_plus[mu, nu] - g_minus[mu, nu]) / (2 * dr)

        gamma_num = christoffel_from_metric(g, dg)
        gamma_ana = christoffel_schwarzschild(r, M)

        # Compare non-zero components with relative tolerance
        for mu in range(4):
            for nu in range(4):
                for rho in range(4):
                    ana = gamma_ana[mu, nu, rho]
                    num = gamma_num[mu, nu, rho]
                    if abs(ana) > 1e-30:
                        rel_err = abs(num - ana) / abs(ana)
                        self.assertLess(
                            rel_err, 1e-5,
                            f"Gamma^{mu}_{nu}{rho}: ana={ana:.6e}, num={num:.6e}, rel_err={rel_err:.2e}"
                        )
                    else:
                        self.assertAlmostEqual(num, 0.0, places=20)

    def test_flat_space_numerical_zero(self):
        """Flat (Minkowski) space: all Christoffel symbols vanish."""
        from cosmo.gr.metric import minkowski_metric

        g = minkowski_metric()
        dg = np.zeros((4, 4, 4))  # constant metric -> zero derivatives

        gamma = christoffel_from_metric(g, dg)
        np.testing.assert_array_almost_equal(gamma, np.zeros((4, 4, 4)), decimal=15)


if __name__ == '__main__':
    unittest.main()
