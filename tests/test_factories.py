"""
Unit tests for simulation factory functions.
Tests run_and_extract_results and other utilities.
"""

import unittest
import numpy as np
from cosmo.constants import SimulationParameters
from cosmo.simulation import CosmologicalSimulation
from cosmo.factories import run_and_extract_results


class TestFactories(unittest.TestCase):
    """Test factory functions for running simulations"""

    def test_run_and_extract_results_returns_correct_keys(self):
        """run_and_extract_results should return dict with expected keys"""
        np.random.seed(42)
        sim_params = SimulationParameters(
            n_particles=20,
            seed=42,
            t_start_Gyr=12.0,
            t_duration_Gyr=1.0,
            n_steps=25,
            damping_factor=0.0
        )

        sim = CosmologicalSimulation(
            sim_params,
            box_size_Gpc=10.0,
            a_start=0.9,
            use_external_nodes=False,
            use_dark_energy=False
        )

        results = run_and_extract_results(
            sim,
            t_duration_Gyr=1.0,
            n_steps=25,
            save_interval=5
        )

        # Verify keys
        self.assertIn('t_Gyr', results)
        self.assertIn('a', results)
        self.assertIn('size_Gpc', results)
        self.assertIn('max_radius_Gpc', results)
        self.assertIn('sim', results)

        # Verify types
        self.assertIsInstance(results['t_Gyr'], np.ndarray)
        self.assertIsInstance(results['a'], np.ndarray)
        self.assertIsInstance(results['size_Gpc'], np.ndarray)
        self.assertIsInstance(results['sim'], CosmologicalSimulation)

    def test_run_and_extract_results_size_is_in_Gpc(self):
        """Extracted size should be in Gpc (not meters)"""
        np.random.seed(42)
        sim_params = SimulationParameters(
            n_particles=20,
            seed=42,
            t_start_Gyr=12.0,
            t_duration_Gyr=0.5,
            n_steps=25,
            damping_factor=0.0
        )

        sim = CosmologicalSimulation(
            sim_params,
            box_size_Gpc=10.0,
            a_start=0.9,
            use_external_nodes=False,
            use_dark_energy=False
        )

        results = run_and_extract_results(
            sim,
            t_duration_Gyr=0.5,
            n_steps=25,
            save_interval=5
        )

        size_Gpc = results['size_Gpc']

        # Size should be ~10-20 Gpc (reasonable for cosmological scales)
        # NOT ~1e26 meters (which would indicate unit error)
        self.assertGreater(
            size_Gpc[0], 1.0,
            msg="Size should be in Gpc, not meters"
        )
        self.assertLess(
            size_Gpc[0], 100.0,
            msg="Size should be in Gpc, not meters"
        )

    def test_run_and_extract_results_arrays_same_length(self):
        """All returned arrays should have same length"""
        np.random.seed(99)
        sim_params = SimulationParameters(
            n_particles=15,
            seed=99,
            t_start_Gyr=11.0,
            t_duration_Gyr=1.0,
            n_steps=20,
            damping_factor=0.0
        )

        sim = CosmologicalSimulation(
            sim_params,
            box_size_Gpc=8.0,
            a_start=0.85,
            use_external_nodes=False,
            use_dark_energy=False
        )

        results = run_and_extract_results(
            sim,
            t_duration_Gyr=1.0,
            n_steps=20,
            save_interval=4
        )

        # All arrays should have same length
        n_t = len(results['t_Gyr'])
        n_a = len(results['a'])
        n_size = len(results['size_Gpc'])
        n_max = len(results['max_radius_Gpc'])

        self.assertEqual(n_t, n_a, "t_Gyr and a should have same length")
        self.assertEqual(n_t, n_size, "t_Gyr and size_Gpc should have same length")
        self.assertEqual(n_t, n_max, "t_Gyr and max_radius_Gpc should have same length")

    def test_run_and_extract_results_time_is_monotonic(self):
        """Extracted times should be monotonically increasing"""
        np.random.seed(123)
        sim_params = SimulationParameters(
            n_particles=20,
            seed=123,
            t_start_Gyr=10.0,
            t_duration_Gyr=1.0,
            n_steps=100,  # dt < 0.05 Gyr required
            damping_factor=0.0
        )

        sim = CosmologicalSimulation(
            sim_params,
            box_size_Gpc=12.0,
            a_start=0.8,
            use_external_nodes=False,
            use_dark_energy=False
        )

        results = run_and_extract_results(
            sim,
            t_duration_Gyr=1.0,
            n_steps=100,
            save_interval=20
        )

        t_Gyr = results['t_Gyr']

        # Times should be strictly increasing
        for i in range(len(t_Gyr) - 1):
            self.assertLess(
                t_Gyr[i], t_Gyr[i+1],
                msg=f"Time should be monotonic increasing. t[{i}]={t_Gyr[i]}, t[{i+1}]={t_Gyr[i+1]}"
            )

    def test_run_and_extract_results_scale_factor_is_monotonic(self):
        """Scale factor should be monotonically increasing (for expanding universe)"""
        np.random.seed(456)
        sim_params = SimulationParameters(
            n_particles=20,
            seed=456,
            t_start_Gyr=10.0,
            t_duration_Gyr=1.0,
            n_steps=100,  # dt < 0.05 Gyr required
            damping_factor=0.0
        )

        sim = CosmologicalSimulation(
            sim_params,
            box_size_Gpc=12.0,
            a_start=0.8,
            use_external_nodes=False,
            use_dark_energy=True  # With dark energy, universe should expand
        )

        results = run_and_extract_results(
            sim,
            t_duration_Gyr=1.0,
            n_steps=100,
            save_interval=20
        )

        a = results['a']

        # Scale factor should be increasing
        for i in range(len(a) - 1):
            self.assertLessEqual(
                a[i], a[i+1],
                msg=f"Scale factor should increase. a[{i}]={a[i]:.4f}, a[{i+1}]={a[i+1]:.4f}"
            )

    def test_run_and_extract_results_sim_object_has_snapshots(self):
        """Returned simulation object should have snapshots"""
        np.random.seed(789)
        sim_params = SimulationParameters(
            n_particles=10,
            seed=789,
            t_start_Gyr=12.0,
            t_duration_Gyr=0.5,
            n_steps=15,
            damping_factor=0.0
        )

        sim = CosmologicalSimulation(
            sim_params,
            box_size_Gpc=8.0,
            a_start=0.9,
            use_external_nodes=False,
            use_dark_energy=False
        )

        results = run_and_extract_results(
            sim,
            t_duration_Gyr=0.5,
            n_steps=15,
            save_interval=3
        )

        sim_obj = results['sim']

        # Simulation should have snapshots
        self.assertGreater(
            len(sim_obj.snapshots), 0,
            msg="Simulation should have at least one snapshot"
        )

        # Each snapshot should have required keys
        snapshot = sim_obj.snapshots[0]
        self.assertIn('positions', snapshot)
        self.assertIn('velocities', snapshot)
        self.assertIn('time', snapshot)


if __name__ == '__main__':
    unittest.main()
