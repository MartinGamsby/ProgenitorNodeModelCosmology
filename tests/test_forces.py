"""
Unit tests for force calculations
Testing fundamental physics of gravity and tidal forces
"""

import unittest
import numpy as np
from cosmo.constants import CosmologicalConstants, LambdaCDMParameters, ExternalNodeParameters
from cosmo.particles import Particle, ParticleSystem, HMEAGrid
from cosmo.integrator import Integrator


class TestGravitationalForces(unittest.TestCase):
    """Test basic gravitational force calculations"""

    def setUp(self):
        self.const = CosmologicalConstants()

    def test_two_particle_attraction(self):
        """Two particles should attract each other"""
        # Create simple 2-particle system
        particles = ParticleSystem(n_particles=2, box_size_Gpc=1.0)

        # Place particles along x-axis
        particles.particles[0].position = np.array([0.0, 0.0, 0.0])
        particles.particles[1].position = np.array([1e25, 0.0, 0.0])  # 1 Gpc apart

        # Equal masses
        m = 1e53  # kg
        particles.particles[0].mass = m
        particles.particles[1].mass = m

        # Create integrator
        integrator = Integrator(particles, softening=0, use_external_nodes=False)

        # Calculate forces
        accelerations = integrator.calculate_internal_forces()

        # Both should accelerate toward each other (opposite directions)
        a0 = accelerations[0]
        a1 = accelerations[1]

        # Particle 0 should accelerate in +x direction (toward particle 1)
        self.assertGreater(a0[0], 0)
        self.assertAlmostEqual(a0[1], 0, places=20)
        self.assertAlmostEqual(a0[2], 0, places=20)

        # Particle 1 should accelerate in -x direction (toward particle 0)
        self.assertLess(a1[0], 0)
        self.assertAlmostEqual(a1[1], 0, places=20)
        self.assertAlmostEqual(a1[2], 0, places=20)

        # Magnitudes should be equal (Newton's 3rd law)
        self.assertAlmostEqual(np.abs(a0[0]), np.abs(a1[0]), places=15)

    def test_gravitational_force_magnitude(self):
        """F = GMm/r^2 for two particles"""
        particles = ParticleSystem(n_particles=2, box_size_Gpc=1.0)

        # Known configuration
        r = 1e24  # 1/3 Gpc separation
        particles.particles[0].position = np.array([0.0, 0.0, 0.0])
        particles.particles[1].position = np.array([r, 0.0, 0.0])

        m1 = 1e53
        m2 = 2e53
        particles.particles[0].mass = m1
        particles.particles[1].mass = m2

        integrator = Integrator(particles, softening=0, use_external_nodes=False)
        accelerations = integrator.calculate_internal_forces()

        # Expected acceleration on particle 0: a = G*m2/r^2
        expected_a = self.const.G * m2 / r**2

        actual_a = np.linalg.norm(accelerations[0])

        self.assertAlmostEqual(actual_a, expected_a, places=5)

    def test_softening_prevents_singularity(self):
        """Softening should prevent infinite force at r=0"""
        particles = ParticleSystem(n_particles=2, box_size_Gpc=1.0)

        # Place particles very close (but not exactly on top)
        particles.particles[0].position = np.array([0.0, 0.0, 0.0])
        particles.particles[1].position = np.array([1e10, 0.0, 0.0])  # 10 km apart

        m = 1e53
        particles.particles[0].mass = m
        particles.particles[1].mass = m

        softening = 1e21  # 1 Mpc softening
        integrator = Integrator(particles, softening=softening, use_external_nodes=False)

        accelerations = integrator.calculate_internal_forces()

        # Force should be finite
        a = np.linalg.norm(accelerations[0])
        self.assertTrue(np.isfinite(a))

        # With softening >> r, force should be approximately G*m/epsilon^2
        expected_max = self.const.G * m / softening**2
        self.assertLess(a, expected_max * 10)  # Within order of magnitude


class TestTidalForces(unittest.TestCase):
    """Test external tidal force calculations"""

    def setUp(self):
        self.const = CosmologicalConstants()

    def test_single_external_node_force_direction(self):
        """Particle should be pulled toward external node"""
        # Create HMEA grid with single node
        M_ext = 1e56
        S = 30 * self.const.Gpc_to_m

        # Manually create single-node grid
        grid = HMEAGrid(M_ext=M_ext, S=S, irregularity=0.0)
        # Keep only one node at +x
        grid.nodes = [(M_ext, np.array([S, 0.0, 0.0]))]

        # Test particle at origin
        position = np.array([[0.0, 0.0, 0.0]])

        acceleration = grid.calculate_tidal_acceleration_batch(position)

        # Should pull in +x direction (toward node)
        self.assertGreater(acceleration[0, 0], 0)
        self.assertAlmostEqual(acceleration[0, 1], 0, places=20)
        self.assertAlmostEqual(acceleration[0, 2], 0, places=20)

    def test_tidal_acceleration_linear_approximation(self):
        """For R << S, tidal acceleration should be approximately linear in R"""
        M_ext = 1e56
        S = 30 * self.const.Gpc_to_m

        grid = HMEAGrid(M_ext=M_ext, S=S, irregularity=0.0)
        grid.nodes = [(M_ext, np.array([S, 0.0, 0.0]))]

        # Test at small R << S
        R1 = 0.1 * self.const.Gpc_to_m
        R2 = 0.2 * self.const.Gpc_to_m

        pos1 = np.array([[R1, 0.0, 0.0]])
        pos2 = np.array([[R2, 0.0, 0.0]])

        a1 = grid.calculate_tidal_acceleration_batch(pos1)[0, 0]
        a2 = grid.calculate_tidal_acceleration_batch(pos2)[0, 0]

        # a should be approximately proportional to R
        # a2/a1 ≈ R2/R1 = 2
        ratio = a2 / a1
        expected_ratio = R2 / R1

        # Allow 10% deviation (it's approximate for finite R/S)
        self.assertAlmostEqual(ratio, expected_ratio, delta=expected_ratio*0.1)

    def test_symmetric_grid_cancellation_at_origin(self):
        """At exact center of symmetric grid, net force should be zero"""
        M_ext = 1e56
        S = 30 * self.const.Gpc_to_m

        # Create symmetric 3x3x3 grid
        grid = HMEAGrid(M_ext=M_ext, S=S, irregularity=0.0)

        # Test at exact origin
        position = np.array([[0.0, 0.0, 0.0]])

        acceleration = grid.calculate_tidal_acceleration_batch(position)

        # Should be approximately zero (small numerical errors allowed)
        a_magnitude = np.linalg.norm(acceleration[0])

        # Typical node force: G*M/S^2 ~ 10^-11 m/s^2
        # Cancellation should reduce this by many orders
        typical_force = self.const.G * M_ext / S**2
        self.assertLess(a_magnitude, typical_force * 0.01)  # Should cancel to < 1%


class TestDarkEnergyForces(unittest.TestCase):
    """Test dark energy acceleration (ΛCDM)"""

    def setUp(self):
        self.const = CosmologicalConstants()
        self.lcdm = LambdaCDMParameters()

    def test_dark_energy_radial_direction(self):
        """Dark energy should push particles radially outward"""
        particles = ParticleSystem(n_particles=1, box_size_Gpc=10.0)

        # Place particle at arbitrary position
        particles.particles[0].position = np.array([1e25, 2e25, 3e25])

        integrator = Integrator(particles, use_dark_energy=True, use_external_nodes=False)

        acceleration = integrator.calculate_dark_energy_forces()

        # Should point in same direction as position (outward)
        pos = particles.particles[0].position
        pos_normalized = pos / np.linalg.norm(pos)
        acc_normalized = acceleration[0] / np.linalg.norm(acceleration[0])

        # Directions should be parallel
        dot_product = np.dot(pos_normalized, acc_normalized)
        self.assertAlmostEqual(dot_product, 1.0, places=5)

    def test_dark_energy_magnitude(self):
        """a_Λ = H0^2 * Omega_Lambda * R"""
        particles = ParticleSystem(n_particles=1, box_size_Gpc=10.0)

        R = 10 * self.const.Gpc_to_m
        particles.particles[0].position = np.array([R, 0.0, 0.0])

        integrator = Integrator(particles, use_dark_energy=True, use_external_nodes=False)

        acceleration = integrator.calculate_dark_energy_forces()

        # Expected: H0^2 * Omega_Lambda * R
        expected = self.lcdm.H0**2 * self.lcdm.Omega_Lambda * R

        actual = np.linalg.norm(acceleration[0])

        self.assertAlmostEqual(actual, expected, places=5)

    def test_dark_energy_disabled_by_default(self):
        """use_dark_energy=False should return zero acceleration"""
        particles = ParticleSystem(n_particles=1, box_size_Gpc=10.0)
        particles.particles[0].position = np.array([1e25, 0.0, 0.0])

        integrator = Integrator(particles, use_dark_energy=False, use_external_nodes=False)

        acceleration = integrator.calculate_dark_energy_forces()

        self.assertAlmostEqual(np.linalg.norm(acceleration[0]), 0.0, places=20)


class TestHubbleDrag(unittest.TestCase):
    """Test Hubble drag calculation"""

    def setUp(self):
        self.const = CosmologicalConstants()
        self.lcdm = LambdaCDMParameters()

    def test_hubble_drag_opposes_velocity(self):
        """Hubble drag should oppose particle velocity"""
        particles = ParticleSystem(n_particles=1, box_size_Gpc=10.0)

        # Give particle velocity in +x direction
        particles.particles[0].velocity = np.array([1e6, 0.0, 0.0])  # 1000 km/s

        integrator = Integrator(particles, use_dark_energy=True, use_external_nodes=False)

        acceleration = integrator.calculate_hubble_drag()

        # Should accelerate in -x direction (opposite to velocity)
        self.assertLess(acceleration[0, 0], 0)
        self.assertAlmostEqual(acceleration[0, 1], 0, places=20)
        self.assertAlmostEqual(acceleration[0, 2], 0, places=20)

    def test_hubble_drag_magnitude(self):
        """a_drag = -2 * H * v"""
        particles = ParticleSystem(n_particles=1, box_size_Gpc=10.0)

        v = np.array([5e5, 3e5, 1e5])  # m/s
        particles.particles[0].velocity = v

        integrator = Integrator(particles, use_dark_energy=True, use_external_nodes=False)

        acceleration = integrator.calculate_hubble_drag()

        # Expected: -2 * H0 * v
        expected = -2.0 * self.lcdm.H0 * v

        np.testing.assert_array_almost_equal(acceleration[0], expected, decimal=5)

    def test_hubble_drag_disabled_without_dark_energy(self):
        """use_dark_energy=False should disable Hubble drag"""
        particles = ParticleSystem(n_particles=1, box_size_Gpc=10.0)
        particles.particles[0].velocity = np.array([1e6, 0.0, 0.0])

        integrator = Integrator(particles, use_dark_energy=False, use_external_nodes=False)

        acceleration = integrator.calculate_hubble_drag()

        self.assertAlmostEqual(np.linalg.norm(acceleration[0]), 0.0, places=20)


if __name__ == '__main__':
    unittest.main()
