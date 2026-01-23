"""
Unit tests for Leapfrog integration algorithm.
Tests the step() method, energy conservation, and symplectic properties.
"""

import unittest
import numpy as np
from cosmo.particles import ParticleSystem
from cosmo.integrator import LeapfrogIntegrator
from cosmo.constants import CosmologicalConstants


class TestLeapfrogIntegrator(unittest.TestCase):
    """Test Leapfrog integration correctness"""

    def test_single_step_updates_positions_and_velocities(self):
        """Leapfrog.step() should update both position and velocity"""
        particles = ParticleSystem(
            n_particles=10,
            box_size=1e26,
            total_mass_kg=1e54,
            damping_factor_override=0.0
        )

        integrator = LeapfrogIntegrator(
            particles,
            use_external_nodes=False,
            use_dark_energy=False
        )

        # Save initial state
        pos_initial = particles.get_positions().copy()
        vel_initial = particles.get_velocities().copy()

        # Take one step
        dt_s = 1e15  # ~31.7 Myr
        integrator.step(dt_s)

        # Verify positions changed
        pos_final = particles.get_positions()
        self.assertFalse(
            np.allclose(pos_initial, pos_final),
            "Positions should change after leapfrog step"
        )

        # Verify velocities changed (due to gravity)
        vel_final = particles.get_velocities()
        self.assertFalse(
            np.allclose(vel_initial, vel_final),
            "Velocities should change after leapfrog step"
        )

    def test_energy_conservation_short_term(self):
        """Leapfrog should conserve energy over short integration"""
        particles = ParticleSystem(
            n_particles=20,
            box_size=1e26,
            total_mass_kg=1e54,
            damping_factor_override=0.0
        )

        integrator = LeapfrogIntegrator(
            particles,
            use_external_nodes=False,
            use_dark_energy=False
        )

        # Initial energy
        E_initial = integrator.total_energy()

        # Evolve for 10 steps
        dt_s = 5e14  # Small timestep
        for _ in range(10):
            integrator.step(dt_s)

        # Final energy
        E_final = integrator.total_energy()

        # Energy should be conserved within ~0.1% for symplectic integrator
        relative_error = abs(E_final - E_initial) / abs(E_initial)
        self.assertLess(
            relative_error, 0.001,
            msg=f"Energy conservation failed: {relative_error*100:.2f}% drift"
        )

    def test_step_increments_time(self):
        """Leapfrog step should increment particle system time"""
        particles = ParticleSystem(
            n_particles=10,
            box_size=1e26,
            total_mass_kg=1e54,
            damping_factor_override=0.0
        )

        integrator = LeapfrogIntegrator(
            particles,
            use_external_nodes=False,
            use_dark_energy=False
        )

        # Initial time
        t_initial = particles.time

        # Take step
        dt_s = 1e15
        integrator.step(dt_s)

        # Time should have increased by dt
        t_final = particles.time
        self.assertAlmostEqual(
            t_final, t_initial + dt_s, places=5,
            msg=f"Time should increment by dt. Got {t_final:.3e}, expected {t_initial + dt_s:.3e}"
        )

    def test_zero_timestep_changes_nothing(self):
        """Step with dt=0 should not change state"""
        particles = ParticleSystem(
            n_particles=10,
            box_size=1e26,
            total_mass_kg=1e54,
            damping_factor_override=0.0
        )

        integrator = LeapfrogIntegrator(
            particles,
            use_external_nodes=False,
            use_dark_energy=False
        )

        # Save initial state
        pos_initial = particles.get_positions().copy()
        vel_initial = particles.get_velocities().copy()
        t_initial = particles.time

        # Step with dt=0
        integrator.step(0.0)

        # Nothing should change
        pos_final = particles.get_positions()
        vel_final = particles.get_velocities()
        t_final = particles.time

        np.testing.assert_array_almost_equal(
            pos_initial, pos_final,
            err_msg="Positions should not change with dt=0"
        )
        np.testing.assert_array_almost_equal(
            vel_initial, vel_final,
            err_msg="Velocities should not change with dt=0"
        )
        self.assertAlmostEqual(
            t_initial, t_final,
            msg="Time should not change with dt=0"
        )

    def test_reversibility(self):
        """Leapfrog should be time-reversible"""
        particles = ParticleSystem(
            n_particles=10,
            box_size=1e26,
            total_mass_kg=1e54,
            damping_factor_override=0.0
        )

        integrator = LeapfrogIntegrator(
            particles,
            use_external_nodes=False,
            use_dark_energy=False
        )

        # Save initial state
        pos_initial = particles.get_positions().copy()
        vel_initial = particles.get_velocities().copy()

        # Step forward
        dt_s = 1e15
        integrator.step(dt_s)

        # Step backward
        integrator.step(-dt_s)

        # Should return to initial state (within numerical precision)
        pos_final = particles.get_positions()
        vel_final = particles.get_velocities()

        np.testing.assert_array_almost_equal(
            pos_initial, pos_final, decimal=5,
            err_msg="Forward-backward step should return to initial position"
        )
        np.testing.assert_array_almost_equal(
            vel_initial, vel_final, decimal=5,
            err_msg="Forward-backward step should return to initial velocity"
        )

    def test_calculate_total_forces_combines_all_forces(self):
        """Total forces should be sum of internal + external + dark energy"""
        particles = ParticleSystem(
            n_particles=10,
            box_size=1e26,
            total_mass_kg=1e54,
            damping_factor_override=0.0,
            use_dark_energy=True
        )

        integrator = LeapfrogIntegrator(
            particles,
            use_external_nodes=True,
            use_dark_energy=True
        )

        # Calculate individual components
        a_internal = integrator.calculate_internal_forces()
        a_external = integrator.calculate_external_forces()
        a_dark = integrator.calculate_dark_energy_forces()

        # Calculate total
        a_total = integrator.calculate_total_forces()

        # Total should equal sum (within numerical precision)
        expected_total = a_internal + a_external + a_dark

        np.testing.assert_array_almost_equal(
            a_total, expected_total, decimal=10,
            err_msg="Total forces should equal sum of components"
        )

    def test_internal_forces_are_newtonian_gravity(self):
        """Internal forces should follow Newton's law of gravitation"""
        const = CosmologicalConstants()

        # Create 2-particle system at known separation
        particles = ParticleSystem(
            n_particles=2,
            box_size=1e25,
            total_mass_kg=2e53,
            damping_factor_override=0.0
        )

        # Place particles at specific locations
        r_sep = 5e24  # 5 Gpc separation
        particles.particles[0].pos = np.array([0.0, 0.0, 0.0])
        particles.particles[1].pos = np.array([r_sep, 0.0, 0.0])

        m1 = 1e53  # kg
        m2 = 1e53  # kg
        particles.particles[0].mass = m1
        particles.particles[1].mass = m2

        integrator = LeapfrogIntegrator(
            particles,
            use_external_nodes=False,
            use_dark_energy=False
        )

        # Calculate forces
        accelerations = integrator.calculate_internal_forces()

        # Acceleration on particle 0 should point toward particle 1 (+x direction)
        # Magnitude should be G*m2/r² (with softening)
        a0 = accelerations[0]

        # Check direction (should be +x)
        self.assertGreater(
            a0[0], 0,
            "Acceleration on particle 0 should point toward particle 1 (+x)"
        )
        self.assertAlmostEqual(
            a0[1], 0, places=10,
            msg="Acceleration should only be in x-direction"
        )
        self.assertAlmostEqual(
            a0[2], 0, places=10,
            msg="Acceleration should only be in x-direction"
        )

        # Check magnitude (approximately G*m2/r²)
        # Note: softening affects this, so we check order of magnitude
        expected_mag = const.G * m2 / r_sep**2
        actual_mag = np.linalg.norm(a0)

        # Should be within factor of 2 (softening affects exact value)
        self.assertLess(
            abs(actual_mag - expected_mag) / expected_mag, 1.0,
            msg=f"Acceleration magnitude should be ~G*m/r². Got {actual_mag:.3e}, expected {expected_mag:.3e}"
        )


if __name__ == '__main__':
    unittest.main()
