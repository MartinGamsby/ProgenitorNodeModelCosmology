"""
Unit tests for GR metric tensors (Phase 1).
Tests Schwarzschild, FRW, Minkowski metrics, inverse, determinant, linearized.
"""

import unittest
import numpy as np

from cosmo.constants import CosmologicalConstants
from cosmo.gr.metric import (
    schwarzschild_metric,
    schwarzschild_radius_m,
    frw_metric,
    minkowski_metric,
    inverse_metric,
    metric_determinant,
    linearized_metric,
)

c = CosmologicalConstants.c
G = CosmologicalConstants.G
M_sun = CosmologicalConstants.M_sun

# 1 AU in meters
AU_m = 1.496e11


class TestSchwarzschildMetric(unittest.TestCase):
    """Tests for Schwarzschild metric."""

    def setUp(self):
        self.r = 10 * AU_m  # well outside r_s for solar mass
        self.M = M_sun
        self.g = schwarzschild_metric(self.r, self.M)

    def test_signature(self):
        """Schwarzschild has signature (-,+,+,+) for r > r_s."""
        self.assertLess(self.g[0, 0], 0)
        self.assertGreater(self.g[1, 1], 0)
        self.assertGreater(self.g[2, 2], 0)
        self.assertGreater(self.g[3, 3], 0)

    def test_gtt_large_r(self):
        """g_tt -> -c^2 as r -> infinity."""
        r_large = 1e20  # very far from mass
        g = schwarzschild_metric(r_large, self.M)
        self.assertAlmostEqual(g[0, 0] / (-c**2), 1.0, places=10)

    def test_grr_large_r(self):
        """g_rr -> 1 as r -> infinity."""
        r_large = 1e20
        g = schwarzschild_metric(r_large, self.M)
        self.assertAlmostEqual(g[1, 1], 1.0, places=10)

    def test_grr_diverges_at_rs(self):
        """g_rr diverges as r -> r_s (horizon)."""
        r_s = schwarzschild_radius_m(self.M)
        # Approach from outside
        r_near = r_s * 1.001
        g = schwarzschild_metric(r_near, self.M)
        self.assertGreater(g[1, 1], 1000)

    def test_off_diagonal_zero(self):
        """Schwarzschild metric is diagonal."""
        for i in range(4):
            for j in range(4):
                if i != j:
                    self.assertEqual(self.g[i, j], 0.0)

    def test_known_values_solar(self):
        """Known values at r=1 AU, M=M_sun."""
        g = schwarzschild_metric(AU_m, self.M)
        r_s = schwarzschild_radius_m(self.M)
        f = 1 - r_s / AU_m
        self.assertAlmostEqual(g[0, 0], -f * c**2, places=5)
        self.assertAlmostEqual(g[1, 1], 1.0 / f, places=15)
        self.assertAlmostEqual(g[2, 2], AU_m**2, places=5)
        self.assertAlmostEqual(g[3, 3], AU_m**2, places=5)


class TestFRWMetric(unittest.TestCase):
    """Tests for FRW metric."""

    def test_k0_diagonal(self):
        """FRW k=0 has correct diagonal form."""
        g = frw_metric(a=2.0, k=0, r_m=3.0)
        self.assertAlmostEqual(g[0, 0], -c**2)
        self.assertAlmostEqual(g[1, 1], 4.0)       # a^2
        self.assertAlmostEqual(g[2, 2], 4.0 * 9.0)  # a^2 * r^2
        self.assertAlmostEqual(g[3, 3], 4.0 * 9.0)

    def test_a1_matches_minkowski_spatial(self):
        """FRW a=1, k=0, r=1 spatial part matches Minkowski."""
        g_frw = frw_metric(a=1.0, k=0, r_m=1.0)
        eta = minkowski_metric()
        # Spatial diagonal should match
        for i in range(1, 4):
            self.assertAlmostEqual(g_frw[i, i], eta[i, i])

    def test_spatial_proportional_to_a2(self):
        """Spatial components scale as a^2."""
        a1, a2 = 1.5, 3.0
        g1 = frw_metric(a=a1, k=0, r_m=2.0)
        g2 = frw_metric(a=a2, k=0, r_m=2.0)
        ratio = (a2 / a1) ** 2
        for i in range(1, 4):
            self.assertAlmostEqual(g2[i, i] / g1[i, i], ratio, places=12)


class TestMinkowskiMetric(unittest.TestCase):
    """Tests for Minkowski metric."""

    def test_diagonal_values(self):
        eta = minkowski_metric()
        self.assertAlmostEqual(eta[0, 0], -c**2)
        self.assertAlmostEqual(eta[1, 1], 1.0)
        self.assertAlmostEqual(eta[2, 2], 1.0)
        self.assertAlmostEqual(eta[3, 3], 1.0)


class TestInverseMetric(unittest.TestCase):
    """Tests for metric inversion."""

    def test_schwarzschild_inverse(self):
        """g * g^{-1} = I for Schwarzschild."""
        g = schwarzschild_metric(10 * AU_m, M_sun)
        g_inv = inverse_metric(g)
        product = g @ g_inv
        np.testing.assert_array_almost_equal(product, np.eye(4), decimal=10)

    def test_frw_inverse(self):
        """g * g^{-1} = I for FRW."""
        g = frw_metric(a=2.0, k=0, r_m=5.0)
        g_inv = inverse_metric(g)
        product = g @ g_inv
        np.testing.assert_array_almost_equal(product, np.eye(4), decimal=10)


class TestMetricDeterminant(unittest.TestCase):
    """Tests for metric determinant."""

    def test_schwarzschild_determinant(self):
        """det(Schwarzschild) = -c^2 * r^4 at theta=pi/2."""
        r = 5 * AU_m
        g = schwarzschild_metric(r, M_sun)
        det = metric_determinant(g)
        # For Schwarzschild at theta=pi/2:
        # det = g_tt * g_rr * g_theta_theta * g_phi_phi
        # = (-f c^2)(1/f)(r^2)(r^2) = -c^2 r^4
        expected = -c**2 * r**4
        self.assertAlmostEqual(det / expected, 1.0, places=10)

    def test_minkowski_determinant(self):
        """det(Minkowski) = -c^2."""
        eta = minkowski_metric()
        det = metric_determinant(eta)
        self.assertAlmostEqual(det / (-c**2), 1.0, places=10)


class TestLinearizedMetric(unittest.TestCase):
    """Tests for linearized metric."""

    def test_h_zero_gives_eta(self):
        """Linearized with h=0 gives Minkowski."""
        h = np.zeros((4, 4))
        g = linearized_metric(h)
        eta = minkowski_metric()
        np.testing.assert_array_almost_equal(g, eta)

    def test_additive(self):
        """Linearized is additive: g = eta + h."""
        eta = minkowski_metric()
        h = np.diag([0.01, -0.02, 0.03, -0.04])
        g = linearized_metric(h, eta)
        np.testing.assert_array_almost_equal(g, eta + h)


class TestSchwarzschildRadius(unittest.TestCase):
    """Tests for Schwarzschild radius calculation."""

    def test_solar_mass(self):
        """schwarzschild_radius_m(M_sun) ~ 2954 m."""
        r_s = schwarzschild_radius_m(M_sun)
        # 2GM_sun/c^2 ≈ 2953.25 m
        self.assertAlmostEqual(r_s, 2 * G * M_sun / c**2, places=5)
        self.assertAlmostEqual(r_s, 2953.25, delta=2.0)


if __name__ == '__main__':
    unittest.main()
