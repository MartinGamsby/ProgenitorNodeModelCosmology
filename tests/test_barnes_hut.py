"""
Unit tests for force calculation methods:
- numba_direct: Numba JIT O(N²) direct summation (exact, fast)
- barnes_hut: Real Barnes-Hut octree O(N log N) (approximate, faster for large N)
"""

import unittest
import numpy as np
from cosmo.constants import CosmologicalConstants
from cosmo.particles import ParticleSystem
from cosmo.integrator import LeapfrogIntegrator


class TestNumbaDirectPerformance(unittest.TestCase):
    """Test Numba JIT direct method accuracy and speed"""

    def setUp(self):
        self.const = CosmologicalConstants()

    def test_numba_direct_matches_numpy_direct(self):
        """Numba direct O(N²) should match NumPy direct O(N²) exactly"""
        particles = ParticleSystem(
            n_particles=10,
            box_size_m=20.0 * self.const.Gpc_to_m,
            total_mass_kg=10 * 1e53,
            damping_factor_override=0.0
        )

        integrator_numpy = LeapfrogIntegrator(
            particles, use_external_nodes=False, use_dark_energy=False,
            force_method='direct'
        )
        a_numpy = integrator_numpy.calculate_internal_forces()

        integrator_numba = LeapfrogIntegrator(
            particles, use_external_nodes=False, use_dark_energy=False,
            force_method='numba_direct'
        )
        a_numba = integrator_numba.calculate_internal_forces_numba_direct()

        relative_error = np.linalg.norm(a_numpy - a_numba) / np.linalg.norm(a_numpy)
        self.assertLess(relative_error, 1e-10,
                       f"Numba direct error {relative_error:.2e} too large (should be machine precision)")

    def test_numba_direct_faster_than_numpy(self):
        """Numba direct should be faster than NumPy for N=300"""
        import time

        particles = ParticleSystem(
            n_particles=300,
            box_size_m=20.0 * self.const.Gpc_to_m,
            total_mass_kg=300 * 1e53,
            damping_factor_override=0.0
        )

        # Time NumPy direct
        integrator_numpy = LeapfrogIntegrator(
            particles, use_external_nodes=False, use_dark_energy=False,
            force_method='direct'
        )
        t0 = time.perf_counter()
        a_numpy = integrator_numpy.calculate_internal_forces()
        t_numpy = time.perf_counter() - t0

        # Time Numba direct (with warmup)
        integrator_numba = LeapfrogIntegrator(
            particles, use_external_nodes=False, use_dark_energy=False,
            force_method='numba_direct'
        )
        _ = integrator_numba.calculate_internal_forces_numba_direct()  # Warmup JIT

        t0 = time.perf_counter()
        a_numba = integrator_numba.calculate_internal_forces_numba_direct()
        t_numba = time.perf_counter() - t0

        speedup = t_numpy / t_numba
        print(f"\nNumba direct speedup: {speedup:.1f}x (NumPy: {t_numpy*1000:.1f}ms, Numba: {t_numba*1000:.1f}ms)")
        self.assertGreater(speedup, 5.0,
                          f"Numba direct speedup {speedup:.1f}x less than expected")

        # Should match exactly
        relative_error = np.linalg.norm(a_numpy - a_numba) / np.linalg.norm(a_numpy)
        self.assertLess(relative_error, 1e-10,
                       f"Numba direct error {relative_error:.2e} too large")


class TestBarnesHutOctree(unittest.TestCase):
    """Test real Barnes-Hut octree accuracy"""

    def setUp(self):
        self.const = CosmologicalConstants()

    def test_barnes_hut_close_to_direct(self):
        """Barnes-Hut octree should be close to direct method (theta=0.5)"""
        particles = ParticleSystem(
            n_particles=50,
            box_size_m=20.0 * self.const.Gpc_to_m,
            total_mass_kg=50 * 1e53,
            damping_factor_override=0.0
        )

        integrator_direct = LeapfrogIntegrator(
            particles, use_external_nodes=False, use_dark_energy=False,
            force_method='direct'
        )
        a_direct = integrator_direct.calculate_internal_forces()

        integrator_bh = LeapfrogIntegrator(
            particles, use_external_nodes=False, use_dark_energy=False,
            force_method='barnes_hut', barnes_hut_theta=0.5
        )
        a_bh = integrator_bh.calculate_internal_forces_barnes_hut()

        # Octree approximation: expect <5% relative error with theta=0.5
        relative_error = np.linalg.norm(a_direct - a_bh) / np.linalg.norm(a_direct)
        self.assertLess(relative_error, 0.05,
                       f"Barnes-Hut error {relative_error:.4f} > 5% for theta=0.5")

    def test_barnes_hut_accuracy_improves_with_smaller_theta(self):
        """Smaller theta should give more accurate results"""
        particles = ParticleSystem(
            n_particles=30,
            box_size_m=20.0 * self.const.Gpc_to_m,
            total_mass_kg=30 * 1e53,
            damping_factor_override=0.0
        )

        integrator_direct = LeapfrogIntegrator(
            particles, use_external_nodes=False, use_dark_energy=False,
            force_method='direct'
        )
        a_direct = integrator_direct.calculate_internal_forces()

        errors = []
        for theta in [0.8, 0.5, 0.3]:
            integrator_bh = LeapfrogIntegrator(
                particles, use_external_nodes=False, use_dark_energy=False,
                force_method='barnes_hut', barnes_hut_theta=theta
            )
            a_bh = integrator_bh.calculate_internal_forces_barnes_hut()
            err = np.linalg.norm(a_direct - a_bh) / np.linalg.norm(a_direct)
            errors.append(err)

        # Smaller theta should give smaller error
        self.assertLessEqual(errors[2], errors[0],
                           f"theta=0.3 error ({errors[2]:.4f}) should be <= theta=0.8 error ({errors[0]:.4f})")

    def test_barnes_hut_correct_direction_two_particles(self):
        """Barnes-Hut should give correct force direction for 2 particles"""
        particles = ParticleSystem(
            n_particles=2,
            box_size_m=1e25,
            total_mass_kg=2e53,
            damping_factor_override=0.0
        )

        r_sep = 5e24
        particles.particles[0].pos = np.array([0.0, 0.0, 0.0])
        particles.particles[1].pos = np.array([r_sep, 0.0, 0.0])
        particles.particles[0].mass_kg = 1e53
        particles.particles[1].mass_kg = 1e53

        integrator_bh = LeapfrogIntegrator(
            particles, use_external_nodes=False, use_dark_energy=False,
            force_method='barnes_hut', barnes_hut_theta=0.5
        )
        a_bh = integrator_bh.calculate_internal_forces_barnes_hut()

        # Particle 0 should be pulled toward +x (toward particle 1)
        self.assertGreater(a_bh[0, 0], 0, "Particle 0 should be accelerated toward +x")
        # Particle 1 should be pulled toward -x (toward particle 0)
        self.assertLess(a_bh[1, 0], 0, "Particle 1 should be accelerated toward -x")


if __name__ == '__main__':
    unittest.main()
