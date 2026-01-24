"""
Unit tests for Numba-accelerated tidal force calculations.
"""

import unittest
import time
import numpy as np
from cosmo.particles import HMEAGrid, ParticleSystem
from cosmo.constants import CosmologicalConstants, ExternalNodeParameters


class TestTidalForcesNumba(unittest.TestCase):
    """Test Numba JIT tidal forces match NumPy version"""

    def setUp(self):
        self.const = CosmologicalConstants()

    def test_numba_matches_numpy_small_N(self):
        """Numba should match NumPy exactly for small N"""
        # Create grid
        ext_params = ExternalNodeParameters()
        grid = HMEAGrid(ext_params)

        # Create test positions
        positions = np.random.uniform(-1e26, 1e26, size=(10, 3))

        # Calculate with both methods
        a_numpy = grid.calculate_tidal_acceleration_batch(positions, use_numba=False)
        a_numba = grid.calculate_tidal_acceleration_batch(positions, use_numba=True)

        # Should match to floating point precision (small rounding differences OK)
        np.testing.assert_allclose(a_numba, a_numpy, rtol=1e-10,
                                   err_msg="Numba tidal forces don't match NumPy")

    def test_numba_matches_numpy_large_N(self):
        """Numba should match NumPy for large N"""
        ext_params = ExternalNodeParameters()
        grid = HMEAGrid(ext_params)

        np.random.seed(42)  # Fixed seed for reproducibility
        positions = np.random.uniform(-1e26, 1e26, size=(500, 3))

        a_numpy = grid.calculate_tidal_acceleration_batch(positions, use_numba=False)
        a_numba = grid.calculate_tidal_acceleration_batch(positions, use_numba=True)

        # Tolerance accounts for different floating point operation order
        np.testing.assert_allclose(a_numba, a_numpy, rtol=1e-9)

    def test_numba_faster_for_large_N(self):
        """Numba should be faster than NumPy for N=1000"""
        ext_params = ExternalNodeParameters()
        grid = HMEAGrid(ext_params)

        positions = np.random.uniform(-1e26, 1e26, size=(1000, 3))

        # Time NumPy
        t0 = time.perf_counter()
        a_numpy = grid.calculate_tidal_acceleration_batch(positions, use_numba=False)
        t_numpy = time.perf_counter() - t0

        # Time Numba (with warmup)
        _ = grid.calculate_tidal_acceleration_batch(positions[:10], use_numba=True)  # Warmup
        t0 = time.perf_counter()
        a_numba = grid.calculate_tidal_acceleration_batch(positions, use_numba=True)
        t_numba = time.perf_counter() - t0

        speedup = t_numpy / t_numba
        print(f"\nTidal forces N=1000: NumPy {t_numpy*1000:.1f}ms, Numba {t_numba*1000:.1f}ms, Speedup: {speedup:.1f}x")

        # Should be faster
        self.assertGreater(speedup, 2.0,
                          f"Numba tidal forces only {speedup:.1f}x faster, expected >2x")

    def test_symmetry_preserved(self):
        """Grid symmetry should be preserved with Numba"""
        ext_params = ExternalNodeParameters()
        grid = HMEAGrid(ext_params)

        # Acceleration at origin should be zero (perfect symmetry)
        origin = np.array([[0.0, 0.0, 0.0]])

        a_origin_numpy = grid.calculate_tidal_acceleration_batch(origin, use_numba=False)
        a_origin_numba = grid.calculate_tidal_acceleration_batch(origin, use_numba=True)

        # Both should be ~zero
        np.testing.assert_allclose(a_origin_numpy, 0.0, atol=1e-20)
        np.testing.assert_allclose(a_origin_numba, 0.0, atol=1e-20)

    def test_default_uses_numba(self):
        """Default behavior should use Numba"""
        ext_params = ExternalNodeParameters()
        grid = HMEAGrid(ext_params)

        positions = np.random.uniform(-1e26, 1e26, size=(10, 3))

        # Default call (should use Numba)
        a_default = grid.calculate_tidal_acceleration_batch(positions)

        # Explicit Numba
        a_numba = grid.calculate_tidal_acceleration_batch(positions, use_numba=True)

        np.testing.assert_array_equal(a_default, a_numba,
                                     err_msg="Default doesn't use Numba")


if __name__ == '__main__':
    unittest.main()
