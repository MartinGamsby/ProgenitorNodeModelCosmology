"""
Unit tests for Barnes-Hut force calculation (Numba JIT implementation).

Tests force accuracy and comparison with direct O(N²) method.
"""

import unittest
import numpy as np
from cosmo.constants import CosmologicalConstants
from cosmo.particles import ParticleSystem
from cosmo.integrator import LeapfrogIntegrator


class TestBarnesHutPerformance(unittest.TestCase):
    """Test Barnes-Hut gives correct results"""

    def setUp(self):
        self.const = CosmologicalConstants()

    def test_barnes_hut_matches_direct_small_N(self):
        """Barnes-Hut should match direct method for small N"""
        # Create small system
        particles = ParticleSystem(
            n_particles=10,
            box_size_m=20.0 * self.const.Gpc_to_m,
            total_mass_kg=10 * 1e53,
            damping_factor_override=0.0
        )

        # Direct method
        integrator_direct = LeapfrogIntegrator(
            particles,
            use_external_nodes=False,
            use_dark_energy=False,
            force_method='direct'
        )
        a_direct = integrator_direct.calculate_internal_forces()

        # Barnes-Hut method
        integrator_bh = LeapfrogIntegrator(
            particles,
            use_external_nodes=False,
            use_dark_energy=False,
            force_method='barnes_hut',
            barnes_hut_theta=0.5
        )
        a_bh = integrator_bh.calculate_internal_forces_barnes_hut()

        # Should match to machine precision
        relative_error = np.linalg.norm(a_direct - a_bh) / np.linalg.norm(a_direct)
        self.assertLess(relative_error, 1e-10,
                       f"Barnes-Hut error {relative_error:.2e} too large")

    def test_barnes_hut_faster_than_direct(self):
        """Barnes-Hut should be faster for N=300"""
        import time

        particles = ParticleSystem(
            n_particles=300,
            box_size_m=20.0 * self.const.Gpc_to_m,
            total_mass_kg=300 * 1e53,
            damping_factor_override=0.0
        )

        # Time direct method
        integrator_direct = LeapfrogIntegrator(
            particles,
            use_external_nodes=False,
            use_dark_energy=False,
            force_method='direct'
        )
        t0 = time.perf_counter()
        a_direct = integrator_direct.calculate_internal_forces()
        t_direct = time.perf_counter() - t0

        # Time Barnes-Hut (with warmup)
        integrator_bh = LeapfrogIntegrator(
            particles,
            use_external_nodes=False,
            use_dark_energy=False,
            force_method='barnes_hut',
            barnes_hut_theta=0.5
        )
        _ = integrator_bh.calculate_internal_forces_barnes_hut()  # Warmup JIT

        t0 = time.perf_counter()
        a_bh = integrator_bh.calculate_internal_forces_barnes_hut()
        t_bh = time.perf_counter() - t0

        # Should be faster
        speedup = t_direct / t_bh
        print(f"\nBarnes-Hut speedup: {speedup:.1f}x (direct: {t_direct*1000:.1f}ms, BH: {t_bh*1000:.1f}ms)")
        self.assertGreater(speedup, 5.0,
                          f"Barnes-Hut speedup {speedup:.1f}x less than expected")

        # Should still match closely
        relative_error = np.linalg.norm(a_direct - a_bh) / np.linalg.norm(a_direct)
        self.assertLess(relative_error, 1e-10,
                       f"Barnes-Hut error {relative_error:.2e} too large")


if __name__ == '__main__':
    unittest.main()
