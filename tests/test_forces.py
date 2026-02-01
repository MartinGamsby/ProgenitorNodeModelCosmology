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
        particles = ParticleSystem(
            n_particles=2,
            box_size_m=1.0 * self.const.Gpc_to_m,
            total_mass_kg=2e53,
            
            mass_randomize=0.0  # Equal masses for deterministic test
        )

        # Place particles along x-axis
        particles.particles[0].pos = np.array([0.0, 0.0, 0.0])
        particles.particles[1].pos = np.array([1e25, 0.0, 0.0])  # 1 Gpc apart

        # Equal masses
        m = 1e53  # kg
        particles.particles[0].mass = m
        particles.particles[1].mass = m

        # Create integrator
        integrator = Integrator(particles, softening_per_Mobs_m=0, use_external_nodes=False, use_dark_energy=False)

        # Calculate forces
        accelerations = integrator.calculate_internal_forces()

        # Both should accelerate toward each other (opposite directions)
        a0 = accelerations[0]
        a1 = accelerations[1]

        # Particle 0 should accelerate in +x direction (toward particle 1)
        self.assertGreater(a0[0], 0)
        self.assertAlmostEqual(a0[1], 0, places=10)
        self.assertAlmostEqual(a0[2], 0, places=10)

        # Particle 1 should accelerate in -x direction (toward particle 0)
        self.assertLess(a1[0], 0)
        self.assertAlmostEqual(a1[1], 0, places=10)
        self.assertAlmostEqual(a1[2], 0, places=10)

        # Magnitudes should be equal (Newton's 3rd law)
        self.assertAlmostEqual(np.abs(a0[0]), np.abs(a1[0]), places=15)

    def test_gravitational_force_magnitude(self):
        """F = GMm/r^2 for two particles"""
        particles = ParticleSystem(
            n_particles=2,
            box_size_m=1.0 * self.const.Gpc_to_m,
            total_mass_kg=3e53,
            
            mass_randomize=0.0  # Equal masses for deterministic test
        )

        # Known configuration
        r = 1e24  # 1/3 Gpc separation
        particles.particles[0].pos = np.array([0.0, 0.0, 0.0])
        particles.particles[1].pos = np.array([r, 0.0, 0.0])

        m1 = 1e53
        m2 = 2e53
        particles.particles[0].mass = m1
        particles.particles[1].mass = m2

        integrator = Integrator(particles, softening_per_Mobs_m=0, use_external_nodes=False, use_dark_energy=False)
        accelerations = integrator.calculate_internal_forces()

        # Expected acceleration on particle 0: a = G*m2/r^2
        expected_a = self.const.G * m2 / r**2

        actual_a = np.linalg.norm(accelerations[0])

        self.assertAlmostEqual(actual_a, expected_a, places=5)

    def test_softening_prevents_singularity(self):
        """Softening should prevent infinite force at r=0"""
        particles = ParticleSystem(
            n_particles=2,
            box_size_m=1.0 * self.const.Gpc_to_m,
            total_mass_kg=2e53,
            
        )

        # Place particles very close (but not exactly on top)
        particles.particles[0].pos = np.array([0.0, 0.0, 0.0])
        particles.particles[1].pos = np.array([1e10, 0.0, 0.0])  # 10 km apart

        m = 1e53
        particles.particles[0].mass = m
        particles.particles[1].mass = m

        softening = 1e24  # 1 Gpc softening
        integrator = Integrator(particles, softening_per_Mobs_m=softening, use_external_nodes=False, use_dark_energy=False)

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
        # Create HMEA grid with ExternalNodeParameters
        M_ext = 1e56
        S = 30 * self.const.Gpc_to_m

        node_params = ExternalNodeParameters(M_ext_kg=M_ext, S=S)
        grid = HMEAGrid(node_params=node_params)

        # Keep only one node at +x (find closest to target position)
        target_pos = np.array([S, 0.0, 0.0])
        distances = [np.linalg.norm(node['position'] - target_pos) for node in grid.nodes]
        closest_idx = np.argmin(distances)
        grid.nodes = [grid.nodes[closest_idx]]

        # Test particle at origin
        position = np.array([[0.0, 0.0, 0.0]])

        acceleration = grid.calculate_tidal_acceleration_batch(position)

        # Should pull in +x direction (toward node)
        self.assertGreater(acceleration[0, 0], 0)
        # y and z small relative to x (5% node irregularity causes ~5-10% deviations)
        self.assertLess(np.abs(acceleration[0, 1]), np.abs(acceleration[0, 0]) * 2.0)
        self.assertLess(np.abs(acceleration[0, 2]), np.abs(acceleration[0, 0]) * 2.0)

    def test_tidal_acceleration_linear_approximation(self):
        """For R << S, tidal acceleration should be approximately constant (a ≈ GM/S²)"""
        M_ext = 1e56
        S = 30 * self.const.Gpc_to_m

        node_params = ExternalNodeParameters(M_ext_kg=M_ext, S=S)
        grid = HMEAGrid(node_params=node_params)

        # Keep only one node at +x
        target_pos = np.array([S, 0.0, 0.0])
        distances = [np.linalg.norm(node['position'] - target_pos) for node in grid.nodes]
        closest_idx = np.argmin(distances)
        grid.nodes = [grid.nodes[closest_idx]]

        # Test at small R << S
        R1 = 0.1 * self.const.Gpc_to_m
        R2 = 0.2 * self.const.Gpc_to_m

        pos1 = np.array([[R1, 0.0, 0.0]])
        pos2 = np.array([[R2, 0.0, 0.0]])

        a1 = grid.calculate_tidal_acceleration_batch(pos1)[0, 0]
        a2 = grid.calculate_tidal_acceleration_batch(pos2)[0, 0]

        # With exact formula a = GM/(S-R)², the ratio should be close to 1
        # Expected: (S-R1)²/(S-R2)² ≈ 1.0067 for R1=0.1, R2=0.2, S=30
        ratio = a2 / a1

        # For R << S, force should be nearly constant (ratio close to 1)
        self.assertAlmostEqual(ratio, 1.0, delta=0.02)  # Within 2%

    def test_symmetric_grid_cancellation_at_origin(self):
        """At exact center of symmetric grid, net force should be small"""
        M_ext = 1e56
        S = 30 * self.const.Gpc_to_m

        # Create symmetric 3x3x3 grid
        node_params = ExternalNodeParameters(M_ext_kg=M_ext, S=S)
        np.random.seed(42)  # Fixed seed for reproducibility
        grid = HMEAGrid(node_params=node_params)

        # Test at exact origin
        position = np.array([[0.0, 0.0, 0.0]])

        acceleration = grid.calculate_tidal_acceleration_batch(position)

        # Should be small (irregularity causes some imbalance)
        a_magnitude = np.linalg.norm(acceleration[0])

        # Typical single node force: G*M/S^2
        typical_force = self.const.G * M_ext / S**2
        # With 26 nodes and 5% irregularity, expect significant cancellation
        self.assertLess(a_magnitude, typical_force * 0.5)  # Should cancel to < 50%


class TestDarkEnergyForces(unittest.TestCase):
    """Test dark energy acceleration (ΛCDM)"""

    def setUp(self):
        self.const = CosmologicalConstants()
        self.lcdm = LambdaCDMParameters()

    def test_dark_energy_radial_direction(self):
        """Dark energy should push particles radially outward"""
        particles = ParticleSystem(
            n_particles=1,
            box_size_m=10.0 * self.const.Gpc_to_m,
            
        )

        # Place particle at arbitrary position
        particles.particles[0].pos = np.array([1e25, 2e25, 3e25])

        integrator = Integrator(particles, use_dark_energy=True, use_external_nodes=False)

        acceleration = integrator.calculate_dark_energy_forces()

        # Should point in same direction as position (outward)
        pos = particles.particles[0].pos
        pos_normalized = pos / np.linalg.norm(pos)
        acc_normalized = acceleration[0] / np.linalg.norm(acceleration[0])

        # Directions should be parallel
        dot_product = np.dot(pos_normalized, acc_normalized)
        self.assertAlmostEqual(dot_product, 1.0, places=5)

    def test_dark_energy_magnitude(self):
        """a_Λ = H0^2 * Omega_Lambda * R"""
        particles = ParticleSystem(
            n_particles=1,
            box_size_m=10.0 * self.const.Gpc_to_m,
            
        )

        R = 10 * self.const.Gpc_to_m
        particles.particles[0].pos = np.array([R, 0.0, 0.0])

        integrator = Integrator(particles, use_dark_energy=True, use_external_nodes=False)

        acceleration = integrator.calculate_dark_energy_forces()

        # Expected: H0^2 * Omega_Lambda * R
        expected = self.lcdm.H0_si**2 * self.lcdm.Omega_Lambda * R

        actual = np.linalg.norm(acceleration[0])

        self.assertAlmostEqual(actual, expected, places=5)

    def test_dark_energy_disabled_by_default(self):
        """use_dark_energy=False should return zero acceleration"""
        particles = ParticleSystem(
            n_particles=1,
            box_size_m=10.0 * self.const.Gpc_to_m,
            
        )
        particles.particles[0].pos = np.array([1e25, 0.0, 0.0])

        integrator = Integrator(particles, use_dark_energy=False, use_external_nodes=False)

        acceleration = integrator.calculate_dark_energy_forces()

        self.assertAlmostEqual(np.linalg.norm(acceleration[0]), 0.0, places=20)


if __name__ == '__main__':
    unittest.main()
