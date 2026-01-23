"""
Unit tests for Barnes-Hut octree force calculation.

Tests tree construction, COM calculations, force accuracy, and comparison
with the direct O(N²) method.
"""

import unittest
import numpy as np
from cosmo.constants import CosmologicalConstants
from cosmo.particles import Particle, ParticleSystem
from cosmo.integrator import Integrator
from cosmo.barnes_hut import OctreeNode, BarnesHutTree


class TestOctreeConstruction(unittest.TestCase):
    """Test octree data structure and tree construction"""

    def setUp(self):
        self.const = CosmologicalConstants()

    def test_single_particle_tree(self):
        """Single particle should create root-only tree"""
        positions = np.array([[0.0, 0.0, 0.0]])
        masses = np.array([1e53])

        tree = BarnesHutTree(theta=0.5, softening_m=1e24)
        tree.build_tree(positions, masses)

        # Root should be a leaf
        self.assertTrue(tree.root.is_leaf())
        self.assertEqual(tree.root.particle_idx, 0)
        self.assertAlmostEqual(tree.root.total_mass_kg, 1e53)
        np.testing.assert_array_almost_equal(
            tree.root.center_of_mass_m, [0.0, 0.0, 0.0]
        )

    def test_two_particle_tree(self):
        """Two particles should create root with 2 child leaves"""
        positions = np.array([
            [0.0, 0.0, 0.0],
            [1e25, 0.0, 0.0]  # 1 Gpc apart on x-axis
        ])
        masses = np.array([1e53, 2e53])

        tree = BarnesHutTree(theta=0.5, softening_m=1e24)
        tree.build_tree(positions, masses)

        # Root should NOT be a leaf
        self.assertFalse(tree.root.is_leaf())
        self.assertIsNotNone(tree.root.children)

        # Root's COM should be weighted average
        expected_com = (0.0 * 1e53 + 1e25 * 2e53) / (1e53 + 2e53)
        np.testing.assert_array_almost_equal(
            tree.root.center_of_mass_m, [expected_com, 0.0, 0.0]
        )

        # Root's total mass should be sum
        self.assertAlmostEqual(tree.root.total_mass_kg, 3e53)

    def test_center_of_mass_calculation(self):
        """Verify COM = Σ(m_i × r_i) / Σm_i for 3 particles"""
        positions = np.array([
            [0.0, 0.0, 0.0],
            [1e25, 0.0, 0.0],
            [0.0, 1e25, 0.0]
        ])
        masses = np.array([1e53, 2e53, 3e53])

        tree = BarnesHutTree(theta=0.5, softening_m=1e24)
        tree.build_tree(positions, masses)

        # Calculate expected COM manually
        total_mass = np.sum(masses)
        expected_com = np.sum(positions * masses[:, np.newaxis], axis=0) / total_mass

        np.testing.assert_array_almost_equal(
            tree.root.center_of_mass_m, expected_com
        )
        self.assertAlmostEqual(tree.root.total_mass_kg, total_mass)

    def test_octree_bounds_correct(self):
        """Verify spatial subdivision is correct"""
        positions = np.array([
            [-1e25, -1e25, -1e25],
            [1e25, 1e25, 1e25]
        ])
        masses = np.array([1e53, 1e53])

        tree = BarnesHutTree(theta=0.5, softening_m=1e24)
        tree.build_tree(positions, masses)

        # Root bounds should contain all particles
        self.assertTrue(np.all(tree.root.min_corner_m <= -1e25))
        self.assertTrue(np.all(tree.root.max_corner_m >= 1e25))

        # Root should have children (2 particles)
        self.assertIsNotNone(tree.root.children)
        self.assertEqual(len(tree.root.children), 8)


class TestBarnesHutForceAccuracy(unittest.TestCase):
    """Test force calculation accuracy"""

    def setUp(self):
        self.const = CosmologicalConstants()

    def test_two_particles_matches_direct(self):
        """For N=2, Barnes-Hut should match direct method exactly"""
        # Create 2-particle system
        particles = ParticleSystem(
            n_particles=2,
            box_size_m=10.0 * self.const.Gpc_to_m,
            total_mass_kg=2e53,
            damping_factor_override=0.0
        )

        # Place particles 5 Gpc apart
        particles.particles[0].pos = np.array([0.0, 0.0, 0.0])
        particles.particles[1].pos = np.array([5e25, 0.0, 0.0])
        particles.particles[0].mass_kg = 1e53
        particles.particles[1].mass_kg = 1e53

        # Direct method
        integrator_direct = Integrator(
            particles,
            softening_per_Mobs_m=1e24,
            use_external_nodes=False,
            use_dark_energy=False,
            force_method='direct'
        )
        a_direct = integrator_direct.calculate_internal_forces()

        # Barnes-Hut method with theta=0 (exact)
        integrator_bh = Integrator(
            particles,
            softening_per_Mobs_m=1e24,
            use_external_nodes=False,
            use_dark_energy=False,
            force_method='barnes_hut',
            barnes_hut_theta=0.0
        )
        a_bh = integrator_bh.calculate_internal_forces_barnes_hut()

        # Should match exactly (or very close)
        np.testing.assert_array_almost_equal(a_direct, a_bh, decimal=10)

    def test_theta_zero_equals_direct(self):
        """theta=0 should give direct method results"""
        # Create small system
        particles = ParticleSystem(
            n_particles=5,
            box_size_m=10.0 * self.const.Gpc_to_m,
            total_mass_kg=5e53,
            damping_factor_override=0.0
        )

        # Random positions
        np.random.seed(42)
        for p in particles.particles:
            p.pos = np.random.uniform(-1e25, 1e25, size=3)

        # Direct method
        integrator_direct = Integrator(
            particles,
            softening_per_Mobs_m=1e24,
            use_external_nodes=False,
            use_dark_energy=False,
            force_method='direct'
        )
        a_direct = integrator_direct.calculate_internal_forces()

        # Barnes-Hut with theta=0
        tree = BarnesHutTree(theta=0.0, softening_m=integrator_direct.softening_m)
        tree.build_tree(particles.get_positions(), particles.get_masses())
        a_bh = tree.calculate_all_accelerations()

        # Should be very close (accounting for floating point)
        for i in range(len(particles)):
            np.testing.assert_array_almost_equal(
                a_direct[i], a_bh[i], decimal=8,
                err_msg=f"Particle {i} acceleration mismatch"
            )

    def test_newton_third_law(self):
        """Forces should be equal and opposite (F_ij = -F_ji)"""
        # Create 2-particle system
        positions = np.array([
            [0.0, 0.0, 0.0],
            [1e25, 0.0, 0.0]
        ])
        masses = np.array([1e53, 1e53])

        tree = BarnesHutTree(theta=0.5, softening_m=1e24)
        tree.build_tree(positions, masses)

        a0 = tree.calculate_acceleration(0)
        a1 = tree.calculate_acceleration(1)

        # Accelerations should point toward each other
        self.assertGreater(a0[0], 0)  # Particle 0 accelerates in +x
        self.assertLess(a1[0], 0)     # Particle 1 accelerates in -x

        # Magnitudes should be equal (same mass)
        np.testing.assert_almost_equal(np.linalg.norm(a0), np.linalg.norm(a1))

    def test_softening_applied(self):
        """Close particles should have finite force (no singularity)"""
        positions = np.array([
            [0.0, 0.0, 0.0],
            [1e20, 0.0, 0.0]  # 100 km apart (very close!)
        ])
        masses = np.array([1e53, 1e53])

        softening_m = 1e24  # 1 Gpc softening >> 100 km
        tree = BarnesHutTree(theta=0.5, softening_m=softening_m)
        tree.build_tree(positions, masses)

        a0 = tree.calculate_acceleration(0)

        # Force should be finite
        self.assertTrue(np.isfinite(np.linalg.norm(a0)))

        # With softening >> r, force should be approximately G*m/ε²
        expected_max = self.const.G * 1e53 / softening_m**2
        self.assertLess(np.linalg.norm(a0), expected_max * 10)


class TestBarnesHutComparison(unittest.TestCase):
    """Compare Barnes-Hut with direct method for various N"""

    def setUp(self):
        self.const = CosmologicalConstants()

    def _compare_methods(self, n_particles: int, theta: float, tolerance: float):
        """Helper to compare direct vs Barnes-Hut"""
        # Create particle system
        particles = ParticleSystem(
            n_particles=n_particles,
            box_size_m=20.0 * self.const.Gpc_to_m,
            total_mass_kg=n_particles * 1e53,
            damping_factor_override=0.0
        )

        # Random positions
        np.random.seed(42)
        for p in particles.particles:
            p.pos = np.random.uniform(-5e25, 5e25, size=3)

        # Direct method
        integrator_direct = Integrator(
            particles,
            softening_per_Mobs_m=1e24,
            use_external_nodes=False,
            use_dark_energy=False,
            force_method='direct'
        )
        a_direct = integrator_direct.calculate_internal_forces()

        # Barnes-Hut method
        tree = BarnesHutTree(theta=theta, softening_m=integrator_direct.softening_m)
        tree.build_tree(particles.get_positions(), particles.get_masses())
        a_bh = tree.calculate_all_accelerations()

        # Calculate per-particle errors
        errors = []
        for i in range(n_particles):
            a_direct_mag = np.linalg.norm(a_direct[i])
            if a_direct_mag > 1e-20:  # Avoid division by tiny numbers
                error = np.linalg.norm(a_bh[i] - a_direct[i]) / a_direct_mag
                errors.append(error)

        rms_error = np.sqrt(np.mean(np.array(errors)**2))
        max_error = np.max(errors)

        print(f"\nN={n_particles}, θ={theta}: RMS error={rms_error:.3f}, max error={max_error:.3f}")

        # Check errors within tolerance
        self.assertLess(rms_error, tolerance,
                       f"RMS error {rms_error:.3f} exceeds tolerance {tolerance}")

        return rms_error, max_error

    def test_N10_within_5_percent(self):
        """N=10 particles should match within 5% (RMS)"""
        rms_error, _ = self._compare_methods(n_particles=10, theta=0.5, tolerance=0.05)

    def test_N50_within_10_percent(self):
        """N=50 particles should match within 10% (RMS)"""
        rms_error, _ = self._compare_methods(n_particles=50, theta=0.5, tolerance=0.10)

    def test_N100_within_15_percent(self):
        """N=100 particles should match within 15% (RMS)"""
        rms_error, _ = self._compare_methods(n_particles=100, theta=0.5, tolerance=0.15)

    def test_theta_effect_on_accuracy(self):
        """Lower theta should give better accuracy"""
        n_particles = 50

        rms_03, _ = self._compare_methods(n_particles=n_particles, theta=0.3, tolerance=0.08)
        rms_05, _ = self._compare_methods(n_particles=n_particles, theta=0.5, tolerance=0.12)
        rms_07, _ = self._compare_methods(n_particles=n_particles, theta=0.7, tolerance=0.20)

        # Lower theta should give lower error
        self.assertLess(rms_03, rms_05)
        self.assertLess(rms_05, rms_07)


if __name__ == '__main__':
    unittest.main()
