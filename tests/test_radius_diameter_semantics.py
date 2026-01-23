"""
Unit tests validating radius vs diameter semantics.
Ensures 'size' variables correctly represent diameter (not radius).
"""

import unittest
import numpy as np
from cosmo.particles import ParticleSystem
from cosmo.simulation import CosmologicalSimulation
from cosmo.constants import SimulationParameters


class TestRadiusDiameterSemantics(unittest.TestCase):
    """Test that radius/diameter semantics are correctly implemented"""

    def test_size_equals_twice_rms_radius(self):
        """Expansion history 'size' should equal 2 × rms_radius"""
        # Create simple particle system
        np.random.seed(42)
        sim_params = SimulationParameters(
            n_particles=50,
            seed=42,
            t_start_Gyr=0.0,
            t_duration_Gyr=0.5,
            n_steps=100,  # Enough steps for stability
            damping_factor=0.0
        )

        sim = CosmologicalSimulation(
            sim_params,
            box_size_Gpc=10.0,
            a_start=0.8,
            use_external_nodes=False,
            use_dark_energy=False
        )

        # Run brief simulation
        # NOTE: sim.run treats t_end_Gyr as duration (bug in code naming)
        sim.run(t_end_Gyr=0.5, n_steps=100, save_interval=20)

        # Check expansion history entries
        # Expansion history is calculated FROM snapshots
        self.assertGreater(len(sim.expansion_history), 0, "Should have expansion history")

        for i, history_entry in enumerate(sim.expansion_history):
            # Get corresponding snapshot
            snapshot = sim.snapshots[i]
            positions = snapshot['positions']

            # Calculate RMS radius manually
            com = np.mean(positions, axis=0)
            r_vectors = positions - com
            r_distances = np.linalg.norm(r_vectors, axis=1)
            rms_radius_m = np.sqrt(np.mean(r_distances**2))

            # Get 'size' from expansion history
            size_m = history_entry['size']

            # Verify: size = 2 × rms_radius (diameter)
            expected_diameter_m = 2.0 * rms_radius_m

            self.assertAlmostEqual(
                size_m, expected_diameter_m, places=5,
                msg=f"Entry {i}: 'size' should equal 2*rms_radius. "
                    f"Got {size_m:.3e}, expected {expected_diameter_m:.3e}"
            )

    def test_calculate_system_size_returns_radius_not_diameter(self):
        """ParticleSystem.calculate_system_size should return RMS radius, not diameter"""
        # Create particles in a known geometry
        # Place particles uniformly in a sphere of radius R
        R_m = 1e26  # 10 Gpc
        n_particles = 100

        # Generate particles uniformly in sphere
        np.random.seed(123)
        positions = []
        for _ in range(n_particles):
            # Rejection sampling for uniform sphere
            while True:
                x = np.random.uniform(-R_m, R_m)
                y = np.random.uniform(-R_m, R_m)
                z = np.random.uniform(-R_m, R_m)
                if x**2 + y**2 + z**2 <= R_m**2:
                    positions.append([x, y, z])
                    break

        positions = np.array(positions)

        # Calculate system size using ParticleSystem method
        from cosmo.particles import ParticleSystem
        rms_radius_m, max_radius_m, com = ParticleSystem.calculate_system_size(positions)

        # For uniform sphere, RMS radius = sqrt(3/5) * R ≈ 0.775 * R
        # max_radius should be ≈ R
        expected_rms_m = np.sqrt(3.0/5.0) * R_m

        self.assertAlmostEqual(
            rms_radius_m, expected_rms_m, delta=R_m*0.1,
            msg=f"RMS radius should be ~0.775*R for uniform sphere. "
                f"Got {rms_radius_m:.3e}, expected {expected_rms_m:.3e}"
        )

        # Max radius should be close to R (not 2R)
        self.assertLess(
            max_radius_m, R_m * 1.1,
            msg=f"Max radius should be ≤R, not diameter. Got {max_radius_m:.3e}"
        )
        self.assertGreater(
            max_radius_m, R_m * 0.9,
            msg=f"Max radius should be ≈R. Got {max_radius_m:.3e}"
        )

    def test_rms_radius_mathematical_definition(self):
        """RMS radius should match sqrt(mean(r²)) definition"""
        # Create simple configuration
        positions = np.array([
            [1e25, 0, 0],
            [-1e25, 0, 0],
            [0, 1e25, 0],
            [0, -1e25, 0]
        ])

        # All particles at distance 1e25 from origin
        # RMS radius should be exactly 1e25
        snapshot = {'positions': positions}
        rms_radius_m, _, com = CosmologicalSimulation.calculate_system_size(snapshot)

        # COM should be at origin
        np.testing.assert_array_almost_equal(
            com, np.zeros(3),
            decimal=5,
            err_msg="COM should be at origin for symmetric config"
        )

        # RMS radius should be 1e25
        expected_rms_m = 1e25
        self.assertAlmostEqual(
            rms_radius_m, expected_rms_m, places=5,
            msg=f"RMS radius should match sqrt(mean(r²)). "
                f"Got {rms_radius_m:.3e}, expected {expected_rms_m:.3e}"
        )

    def test_size_is_not_radius(self):
        """Expansion history 'size' field should NOT be equal to RMS radius"""
        # This test explicitly verifies that size is diameter, not radius
        np.random.seed(999)
        sim_params = SimulationParameters(
            n_particles=30,
            seed=999,
            t_start_Gyr=0.0,
            t_duration_Gyr=0.4,
            n_steps=100,  # Enough steps for stability
            damping_factor=0.0
        )

        sim = CosmologicalSimulation(
            sim_params,
            box_size_Gpc=8.0,
            a_start=0.9,
            use_external_nodes=False,
            use_dark_energy=False
        )

        # Run brief simulation
        # NOTE: sim.run treats t_end_Gyr as duration (bug in code naming)
        sim.run(t_end_Gyr=0.4, n_steps=100, save_interval=20)

        # For at least one snapshot, verify size ≠ rms_radius
        self.assertGreater(len(sim.expansion_history), 0, "Should have expansion history")

        snapshot = sim.snapshots[0]
        positions = snapshot['positions']

        # Calculate RMS radius
        com = np.mean(positions, axis=0)
        r_vectors = positions - com
        r_distances = np.linalg.norm(r_vectors, axis=1)
        rms_radius_m = np.sqrt(np.mean(r_distances**2))

        # Get size
        size_m = sim.expansion_history[0]['size']

        # size should NOT equal rms_radius (it should be 2x)
        self.assertNotAlmostEqual(
            size_m, rms_radius_m, places=3,
            msg="'size' should NOT equal rms_radius (it should be diameter = 2*rms_radius)"
        )

        # Verify it's actually 2x
        ratio = size_m / rms_radius_m
        self.assertAlmostEqual(
            ratio, 2.0, places=3,
            msg=f"size/rms_radius ratio should be 2.0. Got {ratio:.4f}"
        )


if __name__ == '__main__':
    unittest.main()
