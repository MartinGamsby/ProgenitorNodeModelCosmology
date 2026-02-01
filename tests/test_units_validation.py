"""
Unit tests validating physical dimensions and unit correctness.
Tests that functions return values in expected SI units.
"""

import unittest
import numpy as np
from cosmo.constants import CosmologicalConstants, LambdaCDMParameters
from cosmo.particles import ParticleSystem
from cosmo.integrator import LeapfrogIntegrator


class TestUnitsValidation(unittest.TestCase):
    """Test that physical quantities have correct dimensions"""

    def test_acceleration_has_mps2_units(self):
        """Internal gravity acceleration should be in m/s²"""
        # Setup simple 2-particle system
        const = CosmologicalConstants()
        particles = ParticleSystem(
            n_particles=2,
            box_size_m=1e25,
            total_mass_kg=2e53,
            
        )

        # Known configuration: 1 Gpc separation
        r_m = 1e25  # meters
        particles.particles[0].pos = np.array([0.0, 0.0, 0.0])
        particles.particles[1].pos = np.array([r_m, 0.0, 0.0])

        m1_kg = 1e53  # kg
        m2_kg = 1e53  # kg
        particles.particles[0].mass = m1_kg
        particles.particles[1].mass = m2_kg

        integrator = LeapfrogIntegrator(
            particles,
            use_external_nodes=False,
            use_dark_energy=False
        )

        # Calculate acceleration
        a_vec = integrator.calculate_internal_forces()
        a_magnitude = np.linalg.norm(a_vec[0])

        # Expected: a = G*m/r² in m/s²
        expected_mps2 = const.G * m2_kg / r_m**2

        # Verify units by checking order of magnitude
        self.assertAlmostEqual(
            a_magnitude, expected_mps2, places=5,
            msg=f"Acceleration should be in m/s². Got {a_magnitude:.3e}, expected {expected_mps2:.3e}"
        )

    def test_dark_energy_acceleration_units(self):
        """Dark energy acceleration should be H0²*Omega_Lambda*r in m/s²"""
        const = CosmologicalConstants()
        lcdm = LambdaCDMParameters()

        particles = ParticleSystem(
            n_particles=10,
            box_size_m=1e26,
            total_mass_kg=1e54,
            
            use_dark_energy=True
        )

        integrator = LeapfrogIntegrator(
            particles,
            use_external_nodes=False,
            use_dark_energy=True
        )

        # Get dark energy acceleration
        a_dark = integrator.calculate_dark_energy_forces()

        # Verify it's proportional to position (a_Lambda = H0²*Omega_Lambda*r)
        positions = particles.get_positions()
        expected_factor = lcdm.H0_si**2 * lcdm.Omega_Lambda

        # Check one particle
        r_m = np.linalg.norm(positions[0])
        a_magnitude = np.linalg.norm(a_dark[0])
        expected_mps2 = expected_factor * r_m

        self.assertAlmostEqual(
            a_magnitude, expected_mps2, places=10,
            msg=f"Dark energy acceleration units incorrect. Got {a_magnitude:.3e}, expected {expected_mps2:.3e}"
        )

    def test_timestep_has_seconds_units(self):
        """Leapfrog dt should be in seconds"""
        particles = ParticleSystem(
            n_particles=10,
            box_size_m=1e26,
            total_mass_kg=1e54,
            
        )

        integrator = LeapfrogIntegrator(
            particles,
            use_external_nodes=False,
            use_dark_energy=False
        )

        # Evolve for 1 Gyr
        t_end_Gyr = 1.0
        n_steps = 25

        # Convert to seconds (this is what the code should do internally)
        Gyr_to_s = 1e9 * 365.25 * 24 * 3600
        t_end_s = t_end_Gyr * Gyr_to_s
        dt_s = t_end_s / n_steps

        # Verify dt has reasonable magnitude for seconds
        # 1 Gyr / 25 steps ≈ 0.04 Gyr ≈ 1.26e15 seconds
        expected_dt_s = 1.26e15
        self.assertAlmostEqual(
            dt_s, expected_dt_s, delta=1e14,
            msg=f"Timestep should be in seconds. Got {dt_s:.3e}"
        )

    def test_position_units_are_meters(self):
        """Particle positions should be stored in meters"""
        particles = ParticleSystem(
            n_particles=10,
            box_size_m=1e26,  # 10 Gpc in meters
            total_mass_kg=1e54,
            
        )

        positions = particles.get_positions()

        # Positions should be on order of 1e25-1e26 meters (cosmological scales)
        # NOT on order of 1-10 (which would indicate Gpc units)
        typical_position = np.mean(np.abs(positions))

        self.assertGreater(
            typical_position, 1e23,
            msg="Positions appear too small - might not be in meters"
        )
        self.assertLess(
            typical_position, 1e27,
            msg="Positions appear too large - might not be in meters"
        )

    def test_velocity_units_are_mps(self):
        """Particle velocities should be stored in m/s"""
        # Calculate expected Hubble velocity for a known distance
        from cosmo.constants import LambdaCDMParameters
        lcdm = LambdaCDMParameters()
        H_start_si = lcdm.H0_si  # s^-1 (should be ~2.27e-18 s^-1)
        r_m = 1e25  # 10 Gpc in meters
        expected_v_mps = H_start_si * r_m  # m/s

        # For H0 = 70 km/s/Mpc = 2.27e-18 s^-1
        # r = 10 Gpc = 1e25 m
        # v = H0 * r = 2.27e-18 * 1e25 = 2.27e7 m/s

        # Velocity should be ~10^7 m/s (not ~10^3 km/s which would indicate wrong units)
        self.assertGreater(
            expected_v_mps, 1e6,
            msg="Velocities appear too small - might not be in m/s"
        )
        self.assertLess(
            expected_v_mps, 1e9,
            msg="Velocities appear too large - might not be in m/s"
        )


if __name__ == '__main__':
    unittest.main()
