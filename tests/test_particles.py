"""Tests for particle system, including mass randomization."""
import unittest
import numpy as np
from cosmo.particles import ParticleSystem
from cosmo.constants import CosmologicalConstants


class TestMassRandomization(unittest.TestCase):
    """Tests for particle mass randomization feature."""

    def setUp(self):
        """Set up test fixtures."""
        self.const = CosmologicalConstants()
        np.random.seed(42)

    def test_equal_masses_when_randomize_zero(self):
        """mass_randomize=0.0 should give equal masses to all particles."""
        ps = ParticleSystem(n_particles=50, mass_randomize=0.0)
        masses = ps.get_masses()

        # All masses should be equal
        self.assertTrue(np.allclose(masses, masses[0]))

    def test_total_mass_preserved(self):
        """Total mass should be preserved regardless of randomization level."""
        for randomize in [0.0, 0.25, 0.5, 0.75, 1.0]:
            ps = ParticleSystem(n_particles=100, mass_randomize=randomize)
            total = np.sum(ps.get_masses())
            # Use relative tolerance since masses are ~1e53 kg
            rel_error = abs(total - ps.total_mass_kg) / ps.total_mass_kg
            self.assertLess(
                rel_error, 1e-10,
                msg=f"Total mass not preserved for randomize={randomize}"
            )

    def test_mass_range_scales_with_randomize(self):
        """Mass range should scale with randomize parameter."""
        # Higher randomization should give wider mass range
        ps_low = ParticleSystem(n_particles=100, mass_randomize=0.2)
        ps_high = ParticleSystem(n_particles=100, mass_randomize=0.8)

        masses_low = ps_low.get_masses()
        masses_high = ps_high.get_masses()

        range_low = np.max(masses_low) - np.min(masses_low)
        range_high = np.max(masses_high) - np.min(masses_high)

        self.assertGreater(range_high, range_low)

    def test_mass_randomize_clamped_to_valid_range(self):
        """mass_randomize should be clamped to [0, 1]."""
        ps_negative = ParticleSystem(n_particles=10, mass_randomize=-0.5)
        ps_over_one = ParticleSystem(n_particles=10, mass_randomize=1.5)

        # Should be clamped
        self.assertEqual(ps_negative.mass_randomize, 0.0)
        self.assertEqual(ps_over_one.mass_randomize, 1.0)

    def test_no_negative_masses(self):
        """All masses should be positive even at max randomization."""
        ps = ParticleSystem(n_particles=200, mass_randomize=1.0)
        masses = ps.get_masses()

        self.assertTrue(np.all(masses > 0))

    def test_mean_mass_close_to_expected(self):
        """Mean mass should be close to total_mass / n_particles."""
        ps = ParticleSystem(n_particles=100, mass_randomize=0.5)
        masses = ps.get_masses()
        expected_mean = ps.total_mass_kg / ps.n_particles

        self.assertAlmostEqual(np.mean(masses), expected_mean, places=5)

    def test_default_randomize_is_half(self):
        """Default mass_randomize should be 0.5."""
        ps = ParticleSystem(n_particles=10)
        self.assertEqual(ps.mass_randomize, 0.5)

    def test_reproducibility_with_seed(self):
        """Same seed should give same mass distribution."""
        np.random.seed(123)
        ps1 = ParticleSystem(n_particles=50, mass_randomize=0.7)
        masses1 = ps1.get_masses().copy()

        np.random.seed(123)
        ps2 = ParticleSystem(n_particles=50, mass_randomize=0.7)
        masses2 = ps2.get_masses()

        np.testing.assert_array_almost_equal(masses1, masses2)

    def test_single_particle_equal_to_total_mass(self):
        """Single particle should have total mass regardless of randomize."""
        for randomize in [0.0, 0.5, 1.0]:
            ps = ParticleSystem(n_particles=1, mass_randomize=randomize)
            masses = ps.get_masses()

            self.assertEqual(len(masses), 1)
            self.assertAlmostEqual(masses[0], ps.total_mass_kg, places=5)

    def test_randomize_zero_five_gives_moderate_spread(self):
        """mass_randomize=0.5 should give masses in [0.5*mean, 1.5*mean] approximately."""
        np.random.seed(42)
        ps = ParticleSystem(n_particles=500, mass_randomize=0.5)
        masses = ps.get_masses()
        mean_mass = ps.total_mass_kg / ps.n_particles

        # With 500 particles and uniform distribution, extremes should be near bounds
        # Allow some slack due to normalization
        min_ratio = np.min(masses) / mean_mass
        max_ratio = np.max(masses) / mean_mass

        # Should be approximately in [0.5, 1.5] range
        self.assertGreater(min_ratio, 0.4)  # Allow some normalization wiggle
        self.assertLess(min_ratio, 0.7)
        self.assertGreater(max_ratio, 1.3)
        self.assertLess(max_ratio, 1.6)


class TestParticleSystemBasics(unittest.TestCase):
    """Basic tests for ParticleSystem."""

    def test_particle_count(self):
        """Should create correct number of particles."""
        for n in [10, 50, 100]:
            ps = ParticleSystem(n_particles=n, mass_randomize=0.0)
            self.assertEqual(len(ps.particles), n)
            self.assertEqual(ps.n_particles, n)

    def test_get_masses_returns_array(self):
        """get_masses should return numpy array."""
        ps = ParticleSystem(n_particles=20, mass_randomize=0.5)
        masses = ps.get_masses()

        self.assertIsInstance(masses, np.ndarray)
        self.assertEqual(len(masses), 20)

    def test_get_positions_shape(self):
        """get_positions should return (N, 3) array."""
        ps = ParticleSystem(n_particles=30, mass_randomize=0.5)
        positions = ps.get_positions()

        self.assertEqual(positions.shape, (30, 3))


if __name__ == '__main__':
    unittest.main()
