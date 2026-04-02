"""
Unit tests for GR curvature tensors (Phase 3).
Tests Riemann, Ricci, Einstein, Weyl, tidal, and Kretschner for Schwarzschild.
"""

import unittest
import numpy as np

from cosmo.constants import CosmologicalConstants
from cosmo.gr.metric import (
    schwarzschild_metric,
    schwarzschild_radius_m,
    inverse_metric,
)
from cosmo.gr.christoffel import christoffel_schwarzschild
from cosmo.gr.riemann import (
    riemann_schwarzschild,
    riemann_from_christoffel,
    ricci_tensor,
    ricci_scalar,
    einstein_tensor,
    weyl_tensor,
    tidal_tensor_from_riemann,
    kretschner_scalar,
)

c = CosmologicalConstants.c
G = CosmologicalConstants.G
M_sun = CosmologicalConstants.M_sun

AU_m = 1.496e11


class TestRiemannSchwarzschild(unittest.TestCase):
    """Tests for Schwarzschild Riemann tensor."""

    def setUp(self):
        self.r = 10 * AU_m
        self.M = M_sun
        self.R = riemann_schwarzschild(self.r, self.M)
        self.r_s = schwarzschild_radius_m(self.M)
        self.g = schwarzschild_metric(self.r, self.M)
        self.g_inv = inverse_metric(self.g)

    def test_flat_space_riemann_vanishes(self):
        """Riemann -> 0 for r >> r_s (flat space limit)."""
        r_huge = 1e40
        R = riemann_schwarzschild(r_huge, self.M)
        self.assertLess(np.max(np.abs(R)), 1e-30)

    def test_R_r_trt_textbook(self):
        """R^r_trt = -r_s c^2 (r - r_s) / r^4 for Schwarzschild."""
        f = 1 - self.r_s / self.r
        expected = -self.r_s * c**2 * (self.r - self.r_s) / self.r**4
        val = self.R[1, 0, 1, 0]
        self.assertAlmostEqual(val / expected, 1.0, places=10)

    def test_antisymmetry_last_two_indices(self):
        """R^mu_nu_rho_sigma = -R^mu_nu_sigma_rho."""
        for mu in range(4):
            for nu in range(4):
                for rho in range(4):
                    for sigma in range(4):
                        self.assertAlmostEqual(
                            self.R[mu, nu, rho, sigma],
                            -self.R[mu, nu, sigma, rho],
                            places=15,
                            msg=f"Antisymmetry failed for R^{mu}_{nu}{rho}{sigma}"
                        )


class TestRicciSchwarzschild(unittest.TestCase):
    """Tests for Ricci tensor/scalar in vacuum Schwarzschild."""

    def setUp(self):
        self.r = 10 * AU_m
        self.M = M_sun
        self.R = riemann_schwarzschild(self.r, self.M)
        self.g = schwarzschild_metric(self.r, self.M)
        self.g_inv = inverse_metric(self.g)
        self.Ric = ricci_tensor(self.R)
        self.R_scalar = ricci_scalar(self.Ric, self.g_inv)

    def test_ricci_zero_vacuum(self):
        """Ricci tensor = 0 for vacuum Schwarzschild."""
        max_val = np.max(np.abs(self.Ric))
        self.assertLess(max_val, 1e-15,
                        f"Ricci not zero in vacuum: max component = {max_val:.2e}")

    def test_ricci_symmetric(self):
        """Ricci tensor is symmetric: R_mu_nu = R_nu_mu."""
        for mu in range(4):
            for nu in range(4):
                self.assertAlmostEqual(
                    self.Ric[mu, nu], self.Ric[nu, mu], places=10,
                    msg=f"Ricci asymmetry at ({mu},{nu})"
                )

    def test_ricci_scalar_zero_vacuum(self):
        """Ricci scalar = 0 for vacuum Schwarzschild."""
        self.assertAlmostEqual(self.R_scalar, 0.0, places=10,
                               msg=f"Ricci scalar = {self.R_scalar:.2e}")


class TestEinsteinTensor(unittest.TestCase):
    """Tests for Einstein tensor in vacuum Schwarzschild."""

    def setUp(self):
        self.r = 10 * AU_m
        self.M = M_sun
        R = riemann_schwarzschild(self.r, self.M)
        self.g = schwarzschild_metric(self.r, self.M)
        self.g_inv = inverse_metric(self.g)
        Ric = ricci_tensor(R)
        R_sc = ricci_scalar(Ric, self.g_inv)
        self.G = einstein_tensor(Ric, R_sc, self.g)

    def test_einstein_zero_vacuum(self):
        """Einstein tensor = 0 for vacuum Schwarzschild."""
        max_val = np.max(np.abs(self.G))
        self.assertLess(max_val, 1e-15,
                        f"Einstein not zero in vacuum: max = {max_val:.2e}")

    def test_einstein_symmetric(self):
        """Einstein tensor is symmetric: G_mu_nu = G_nu_mu."""
        for mu in range(4):
            for nu in range(4):
                self.assertAlmostEqual(
                    self.G[mu, nu], self.G[nu, mu], places=10,
                    msg=f"Einstein asymmetry at ({mu},{nu})"
                )

    def test_einstein_trace(self):
        """Trace: g^{mu nu} G_mu_nu = -R."""
        r = 10 * AU_m
        M = M_sun
        R = riemann_schwarzschild(r, M)
        g = schwarzschild_metric(r, M)
        g_inv = inverse_metric(g)
        Ric = ricci_tensor(R)
        R_sc = ricci_scalar(Ric, g_inv)
        G = einstein_tensor(Ric, R_sc, g)

        trace_G = np.einsum('mn,mn->', g_inv, G)
        # In 4D: g^mn G_mn = g^mn R_mn - 1/2 * 4 * R = R - 2R = -R
        self.assertAlmostEqual(trace_G, -R_sc, places=10)


class TestWeylTensor(unittest.TestCase):
    """Tests for Weyl tensor in vacuum Schwarzschild."""

    def test_weyl_equals_riemann_in_vacuum(self):
        """In vacuum (Ricci=0): Weyl = Riemann."""
        r = 10 * AU_m
        M = M_sun
        R = riemann_schwarzschild(r, M)
        g = schwarzschild_metric(r, M)
        g_inv = inverse_metric(g)
        Ric = ricci_tensor(R)
        R_sc = ricci_scalar(Ric, g_inv)
        W = weyl_tensor(R, Ric, R_sc, g)

        # Since Ricci=0 in vacuum, Weyl should equal Riemann
        np.testing.assert_array_almost_equal(
            W, R, decimal=6,
            err_msg="Weyl != Riemann in vacuum"
        )


class TestTidalTensor(unittest.TestCase):
    """Tests for tidal tensor extracted from Riemann."""

    def setUp(self):
        self.r = 10 * AU_m
        self.M = M_sun
        R = riemann_schwarzschild(self.r, self.M)
        self.E = tidal_tensor_from_riemann(R)

    def test_tidal_symmetric(self):
        """Tidal tensor is symmetric: E_ij = E_ji."""
        for i in range(3):
            for j in range(3):
                self.assertAlmostEqual(
                    self.E[i, j], self.E[j, i], places=15,
                    msg=f"Tidal asymmetry at ({i},{j})"
                )

    def test_tidal_traceless_vacuum(self):
        """Tidal tensor is traceless in vacuum: Tr(E) = R_00 = 0."""
        trace = np.trace(self.E)
        self.assertAlmostEqual(trace, 0.0, places=10,
                               msg=f"Tidal trace = {trace:.2e}")

    def test_E_rr_negative(self):
        """E_rr < 0 (radial stretching)."""
        self.assertLess(self.E[0, 0], 0,
                        f"E_rr = {self.E[0, 0]:.2e} should be negative")

    def test_E_theta_theta_positive(self):
        """E_theta_theta > 0 (transverse compression)."""
        self.assertGreater(self.E[1, 1], 0,
                           f"E_theta_theta = {self.E[1, 1]:.2e} should be positive")


class TestKretschnerScalar(unittest.TestCase):
    """Tests for Kretschner scalar."""

    def test_kretschner_schwarzschild(self):
        """Kretschner = 48 G^2 M^2 / (c^4 r^6) for Schwarzschild."""
        r = 10 * AU_m
        M = M_sun
        R = riemann_schwarzschild(r, M)
        g = schwarzschild_metric(r, M)

        K = kretschner_scalar(R, g)
        K_expected = 48 * G**2 * M**2 / (c**4 * r**6)

        rel_err = abs(K - K_expected) / K_expected
        self.assertLess(rel_err, 1e-4,
                        f"Kretschner: got {K:.6e}, expected {K_expected:.6e}, "
                        f"rel_err = {rel_err:.2e}")

    def test_kretschner_vanishes_large_r(self):
        """Kretschner -> 0 for large r."""
        r = 1e30
        M = M_sun
        R = riemann_schwarzschild(r, M)
        g = schwarzschild_metric(r, M)

        K = kretschner_scalar(R, g)
        self.assertAlmostEqual(K, 0.0, places=30,
                               msg=f"Kretschner at r=1e30: {K:.2e}")


class TestRiemannNumerical(unittest.TestCase):
    """Tests for numerical Riemann computation from Christoffel symbols."""

    def test_numerical_matches_analytic_schwarzschild(self):
        """riemann_from_christoffel matches analytic riemann_schwarzschild.

        Computes Christoffel r-derivatives via finite difference and adds
        analytic theta-derivatives for the two components that are zero at
        theta=pi/2 but have non-zero theta-derivatives:
          d_theta Gamma^theta_phi_phi = 1
          d_theta Gamma^phi_theta_phi = d_theta Gamma^phi_phi_theta = -1
        """
        M = M_sun
        r_s = schwarzschild_radius_m(M)
        r = 100 * r_s
        dr = r * 1e-7

        gamma = christoffel_schwarzschild(r, M)
        gamma_plus = christoffel_schwarzschild(r + dr, M)
        gamma_minus = christoffel_schwarzschild(r - dr, M)

        dgamma = np.zeros((4, 4, 4, 4))
        # r-derivatives (sigma=1)
        dgamma[:, :, :, 1] = (gamma_plus - gamma_minus) / (2 * dr)
        # theta-derivatives (sigma=2) — non-zero at theta=pi/2
        dgamma[2, 3, 3, 2] = 1.0    # d_theta Gamma^theta_phi_phi
        dgamma[3, 2, 3, 2] = -1.0   # d_theta Gamma^phi_theta_phi
        dgamma[3, 3, 2, 2] = -1.0   # d_theta Gamma^phi_phi_theta

        R_num = riemann_from_christoffel(gamma, dgamma)
        R_ana = riemann_schwarzschild(r, M)

        for mu in range(4):
            for nu in range(4):
                for rho in range(4):
                    for sigma in range(4):
                        ana = R_ana[mu, nu, rho, sigma]
                        num = R_num[mu, nu, rho, sigma]
                        if abs(ana) > 1e-30:
                            rel_err = abs(num - ana) / abs(ana)
                            self.assertLess(
                                rel_err, 1e-5,
                                f"R^{mu}_{nu}{rho}{sigma}: ana={ana:.6e}, "
                                f"num={num:.6e}, rel_err={rel_err:.2e}"
                            )
                        else:
                            self.assertAlmostEqual(
                                num, 0.0, places=15,
                                msg=f"R^{mu}_{nu}{rho}{sigma}: expected ~0, got {num:.2e}"
                            )


if __name__ == '__main__':
    unittest.main()
