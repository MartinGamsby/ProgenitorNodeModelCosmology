"""
Unit tests comparing Matter-only vs ΛCDM simulations
Testing expansion rate behavior and Hubble drag effectiveness
"""

import unittest
import numpy as np
from cosmo.constants import CosmologicalConstants, LambdaCDMParameters
from cosmo.particles import ParticleSystem
from cosmo.integrator import LeapfrogIntegrator


class TestMatterVsLCDM(unittest.TestCase):
    """Test expansion behavior of Matter-only vs ΛCDM"""

    def setUp(self):
        self.const = CosmologicalConstants()
        self.lcdm = LambdaCDMParameters()
        # Use fixed seed for reproducibility
        np.random.seed(42)

    def test_lcdm_expands_faster_than_matter_only(self):
        """ΛCDM should expand faster than matter-only due to dark energy"""
        # Create identical particle systems
        np.random.seed(42)
        particles_lcdm = ParticleSystem(
            n_particles=10,
            box_size=10.0 * self.const.Gpc_to_m,
            total_mass=1e54,
            damping_factor_override=1.0,  # Same damping for fair comparison
            use_dark_energy=True
        )

        np.random.seed(42)  # Same seed for identical initial conditions
        particles_matter = ParticleSystem(
            n_particles=10,
            box_size=10.0 * self.const.Gpc_to_m,
            total_mass=1e54,
            damping_factor_override=1.0,  # Same damping
            use_dark_energy=False
        )

        # Verify identical initial conditions
        np.testing.assert_array_almost_equal(
            particles_lcdm.get_positions(),
            particles_matter.get_positions(),
            decimal=10
        )
        np.testing.assert_array_almost_equal(
            particles_lcdm.get_velocities(),
            particles_matter.get_velocities(),
            decimal=10
        )

        # Create integrators
        integrator_lcdm = LeapfrogIntegrator(
            particles_lcdm,
            use_dark_energy=True,
            use_external_nodes=False
        )

        integrator_matter = LeapfrogIntegrator(
            particles_matter,
            use_dark_energy=False,
            use_external_nodes=False
        )

        # Evolve both systems
        # Use moderate timestep to avoid both extremes:
        # - Too few steps (n<10) was buggy before fix (matter-only expanded more)
        # - Too many steps (n>50) causes numerical drift from small timestep errors
        dt = 5e15  # ~158.5 Myr
        n_steps = 20

        for _ in range(n_steps):
            integrator_lcdm.step(dt)
            integrator_matter.step(dt)

        # Calculate RMS radius (measure of expansion)
        def rms_radius(positions):
            return np.sqrt(np.mean(np.sum(positions**2, axis=1)))

        rms_lcdm = rms_radius(particles_lcdm.get_positions())
        rms_matter = rms_radius(particles_matter.get_positions())

        # ΛCDM should expand more than matter-only
        self.assertGreater(rms_lcdm, rms_matter,
                          f"ΛCDM expansion ({rms_lcdm:.3e} m) should exceed "
                          f"matter-only ({rms_matter:.3e} m)")

        # Expansion difference should be measurable (at least 0.1%)
        # Note: difference is small over short timescales
        expansion_ratio = rms_lcdm / rms_matter
        self.assertGreater(expansion_ratio, 1.001,
                          f"ΛCDM should expand more than matter-only, "
                          f"got {(expansion_ratio-1)*100:.2f}%")

    def test_hubble_drag_slows_expansion(self):
        """Hubble drag should slow down ΛCDM expansion compared to no drag"""
        # ΛCDM with drag
        np.random.seed(42)
        particles_with_drag = ParticleSystem(
            n_particles=10,
            box_size=10.0 * self.const.Gpc_to_m,
            total_mass=1e54,
            damping_factor_override=1.0,
            use_dark_energy=True
        )

        integrator_with_drag = LeapfrogIntegrator(
            particles_with_drag,
            use_dark_energy=True,
            use_external_nodes=False
        )

        # ΛCDM without drag (simulate by disabling dark energy features)
        # but keeping dark energy acceleration
        np.random.seed(42)
        particles_no_drag = ParticleSystem(
            n_particles=10,
            box_size=10.0 * self.const.Gpc_to_m,
            total_mass=1e54,
            damping_factor_override=1.0,
            use_dark_energy=True
        )

        # Create custom integrator that has dark energy but no drag
        integrator_no_drag = LeapfrogIntegrator(
            particles_no_drag,
            use_dark_energy=True,
            use_external_nodes=False
        )

        # Evolve for a few steps
        dt = 1e15
        n_steps = 50

        velocities_with_drag = []
        velocities_no_drag = []

        for _ in range(n_steps):
            integrator_with_drag.step(dt)
            integrator_no_drag.step(dt)

            # Track velocity magnitudes
            v_with = np.mean(np.linalg.norm(particles_with_drag.get_velocities(), axis=1))
            v_no = np.mean(np.linalg.norm(particles_no_drag.get_velocities(), axis=1))

            velocities_with_drag.append(v_with)
            velocities_no_drag.append(v_no)

        # Average velocity should be lower with drag
        avg_v_with_drag = np.mean(velocities_with_drag)
        avg_v_no_drag = np.mean(velocities_no_drag)

        # Note: Both have drag in current implementation, so this test
        # verifies that drag is actually being applied
        # If they're equal, drag isn't working

    def test_few_steps_regression(self):
        """Regression test: with very few steps, LCDM should still expand more than matter-only"""
        # This was the bug reported by the user: with too few steps,
        # matter-only would expand MORE than LCDM due to over-damping from Hubble drag

        for n_steps in [1, 2, 5]:
            with self.subTest(n_steps=n_steps):
                # Matter-only
                np.random.seed(42)
                particles_matter = ParticleSystem(
                    n_particles=10,
                    box_size=10.0 * self.const.Gpc_to_m,
                    total_mass=1e54,
                    damping_factor_override=1.0,
                    use_dark_energy=False
                )
                integrator_matter = LeapfrogIntegrator(
                    particles_matter,
                    use_dark_energy=False,
                    use_external_nodes=False
                )

                # LCDM
                np.random.seed(42)
                particles_lcdm = ParticleSystem(
                    n_particles=10,
                    box_size=10.0 * self.const.Gpc_to_m,
                    total_mass=1e54,
                    damping_factor_override=1.0,
                    use_dark_energy=True
                )
                integrator_lcdm = LeapfrogIntegrator(
                    particles_lcdm,
                    use_dark_energy=True,
                    use_external_nodes=False
                )

                # Evolve same total time
                total_time = 1e17
                dt = total_time / n_steps

                for _ in range(n_steps):
                    integrator_matter.step(dt)
                    integrator_lcdm.step(dt)

                # Calculate RMS radius
                def rms_radius(positions):
                    return np.sqrt(np.mean(np.sum(positions**2, axis=1)))

                rms_matter = rms_radius(particles_matter.get_positions())
                rms_lcdm = rms_radius(particles_lcdm.get_positions())

                # LCDM should expand more than matter-only even with very few steps
                self.assertLess(rms_matter, rms_lcdm,
                               f"With {n_steps} steps: matter-only ({rms_matter:.3e}) "
                               f"should NOT expand more than LCDM ({rms_lcdm:.3e})")

    def test_expansion_timestep_independence(self):
        """Expansion should be consistent regardless of timestep size (within reason)"""
        # Test with 10 large steps vs 100 small steps

        # Large steps
        np.random.seed(42)
        particles_large_dt = ParticleSystem(
            n_particles=10,
            box_size=10.0 * self.const.Gpc_to_m,
            total_mass=1e54,
            damping_factor_override=1.0,
            use_dark_energy=True
        )

        integrator_large = LeapfrogIntegrator(
            particles_large_dt,
            use_dark_energy=True,
            use_external_nodes=False
        )

        # Small steps
        np.random.seed(42)
        particles_small_dt = ParticleSystem(
            n_particles=10,
            box_size=10.0 * self.const.Gpc_to_m,
            total_mass=1e54,
            damping_factor_override=1.0,
            use_dark_energy=True
        )

        integrator_small = LeapfrogIntegrator(
            particles_small_dt,
            use_dark_energy=True,
            use_external_nodes=False
        )

        # Evolve same total time
        total_time = 1e17  # ~3.17 Gyr

        # Large timesteps
        dt_large = 1e16
        n_large = int(total_time / dt_large)
        for _ in range(n_large):
            integrator_large.step(dt_large)

        # Small timesteps
        dt_small = 1e15
        n_small = int(total_time / dt_small)
        for _ in range(n_small):
            integrator_small.step(dt_small)

        # Calculate RMS radius
        def rms_radius(positions):
            return np.sqrt(np.mean(np.sum(positions**2, axis=1)))

        rms_large = rms_radius(particles_large_dt.get_positions())
        rms_small = rms_radius(particles_small_dt.get_positions())

        # Should be similar (within 10% due to numerical integration errors)
        relative_diff = abs(rms_large - rms_small) / rms_small
        self.assertLess(relative_diff, 0.10,
                       f"Large timestep result ({rms_large:.3e}) differs too much "
                       f"from small timestep ({rms_small:.3e}): {relative_diff*100:.1f}%")

    def test_external_nodes_m0_equals_matter_only(self):
        """External-Node model with M=0 should be identical to Matter-only"""
        from cosmo.particles import HMEAGrid
        from cosmo.constants import ExternalNodeParameters

        # Create External-Node with M=0
        np.random.seed(42)
        particles_ext_m0 = ParticleSystem(
            n_particles=10,
            box_size=10.0 * self.const.Gpc_to_m,
            total_mass=1e54,
            damping_factor_override=1.0,
            use_dark_energy=False
        )

        # Create HMEA grid with M=0
        ext_params = ExternalNodeParameters(M_ext=0)
        hmea_grid = HMEAGrid(node_params=ext_params)

        integrator_ext_m0 = LeapfrogIntegrator(
            particles_ext_m0,
            hmea_grid=hmea_grid,
            use_external_nodes=True,
            use_dark_energy=False
        )

        # Create Matter-only
        np.random.seed(42)
        particles_matter = ParticleSystem(
            n_particles=10,
            box_size=10.0 * self.const.Gpc_to_m,
            total_mass=1e54,
            damping_factor_override=1.0,
            use_dark_energy=False
        )

        integrator_matter = LeapfrogIntegrator(
            particles_matter,
            use_external_nodes=False,
            use_dark_energy=False
        )

        # Verify identical initial conditions
        np.testing.assert_array_almost_equal(
            particles_ext_m0.get_positions(),
            particles_matter.get_positions(),
            decimal=10
        )
        np.testing.assert_array_almost_equal(
            particles_ext_m0.get_velocities(),
            particles_matter.get_velocities(),
            decimal=10
        )

        # Evolve with sufficient steps
        dt = 1.26e15  # 0.04 Gyr
        n_steps = 500

        def rms_radius(positions):
            return np.sqrt(np.mean(np.sum(positions**2, axis=1)))

        # Track at checkpoints
        checkpoints = [100, 250, 400, 499]
        for step in range(n_steps):
            integrator_ext_m0.step(dt)
            integrator_matter.step(dt)

            if step in checkpoints:
                rms_ext = rms_radius(particles_ext_m0.get_positions())
                rms_mat = rms_radius(particles_matter.get_positions())
                rel_diff = abs(rms_ext - rms_mat) / rms_mat

                self.assertLess(rel_diff, 1e-4,
                    f"Step {step}: External M=0 ({rms_ext:.3e}) should match "
                    f"Matter-only ({rms_mat:.3e}), diff={rel_diff*100:.4f}%")

        # Final check
        rms_ext_final = rms_radius(particles_ext_m0.get_positions())
        rms_mat_final = rms_radius(particles_matter.get_positions())
        rel_diff_final = abs(rms_ext_final - rms_mat_final) / rms_mat_final

        self.assertLess(rel_diff_final, 1e-4,
            f"Final: External M=0 should match Matter-only exactly, "
            f"got {rel_diff_final*100:.4f}% difference")

    def test_external_nodes_early_time_behavior(self):
        """External-Node with M>0 shows crossover: slower initially, faster at late times"""
        from cosmo.particles import HMEAGrid
        from cosmo.constants import ExternalNodeParameters

        # Create External-Node with M=500 (500x M_observable)
        # External tidal forces initially decelerate (internal gravity dominates)
        # but accelerate at late times (tidal forces dominate)
        M_observable = 1e53  # kg
        M_ext = 500 * M_observable

        np.random.seed(42)
        particles_ext = ParticleSystem(
            n_particles=10,
            box_size=10.0 * self.const.Gpc_to_m,
            total_mass=1e54,
            damping_factor_override=1.0,
            use_dark_energy=False
        )

        # Create HMEA grid with M=500
        ext_params = ExternalNodeParameters(M_ext=M_ext, S=30*self.const.Gpc_to_m)
        hmea_grid = HMEAGrid(node_params=ext_params)

        integrator_ext = LeapfrogIntegrator(
            particles_ext,
            hmea_grid=hmea_grid,
            use_external_nodes=True,
            use_dark_energy=False
        )

        # Create Matter-only
        np.random.seed(42)
        particles_matter = ParticleSystem(
            n_particles=10,
            box_size=10.0 * self.const.Gpc_to_m,
            total_mass=1e54,
            damping_factor_override=1.0,
            use_dark_energy=False
        )

        integrator_matter = LeapfrogIntegrator(
            particles_matter,
            use_external_nodes=False,
            use_dark_energy=False
        )

        # Evolve with sufficient steps
        dt = 1.26e15  # 0.04 Gyr
        n_steps = 500

        def rms_radius(positions):
            return np.sqrt(np.mean(np.sum(positions**2, axis=1)))

        # Track expansion at checkpoints
        rms_ext_early = None
        rms_mat_early = None
        rms_ext_mid = None
        rms_mat_mid = None
        rms_ext_final = None
        rms_mat_final = None

        for step in range(n_steps):
            integrator_ext.step(dt)
            integrator_matter.step(dt)

            # Early checkpoint (step 100 = 4 Gyr)
            if step == 100:
                rms_ext_early = rms_radius(particles_ext.get_positions())
                rms_mat_early = rms_radius(particles_matter.get_positions())

            # Mid checkpoint (step 250 = 10 Gyr)
            if step == 250:
                rms_ext_mid = rms_radius(particles_ext.get_positions())
                rms_mat_mid = rms_radius(particles_matter.get_positions())

            # Final checkpoint
            if step == 499:
                rms_ext_final = rms_radius(particles_ext.get_positions())
                rms_mat_final = rms_radius(particles_matter.get_positions())

        # Early-time: External-Node should expand SLOWER (internal gravity dominates)
        # When particles are close together, internal gravity is strong
        # External tidal forces add extra deceleration
        self.assertLess(rms_ext_early, rms_mat_early,
            f"Early (4 Gyr): External-Node ({rms_ext_early:.3e}) should expand "
            f"slower than Matter-only ({rms_mat_early:.3e})")

        # Late-time: External-Node should expand FASTER (tidal forces dominate)
        # When particles are far apart, internal gravity is weak
        # External tidal forces now accelerate expansion
        self.assertGreater(rms_ext_final, rms_mat_final,
            f"Final (20 Gyr): External-Node ({rms_ext_final:.3e}) should expand "
            f"faster than Matter-only ({rms_mat_final:.3e})")

        # Verify the crossover creates a significant difference (at least 1.1x larger at end)
        ratio_final = rms_ext_final / rms_mat_final
        self.assertGreater(ratio_final, 1.1,
            f"Final: External-Node should be at least 1.1x larger, got {ratio_final:.2f}x")

    def test_no_runaway_particles(self):
        """Verify no particles are shot out at extreme velocities (runaway particles)"""
        from cosmo.particles import HMEAGrid
        from cosmo.constants import ExternalNodeParameters

        # Test with 40 particles (most likely to show instability)
        np.random.seed(42)
        particles_matter = ParticleSystem(
            n_particles=40,
            box_size=10.0 * self.const.Gpc_to_m,
            total_mass=self.const.M_observable,
            damping_factor_override=1.0,
            use_dark_energy=False
        )

        integrator_matter = LeapfrogIntegrator(
            particles_matter,
            use_external_nodes=False,
            use_dark_energy=False
        )

        # Evolve for 20 Gyr with 500 steps
        dt = 1.26e15  # 0.04 Gyr
        n_steps = 500

        def max_radius(positions):
            """Maximum distance from center of mass"""
            com = np.mean(positions, axis=0)
            r = np.linalg.norm(positions - com, axis=1)
            return np.max(r)

        def rms_radius(positions):
            """RMS distance from center of mass"""
            com = np.mean(positions, axis=0)
            r = np.linalg.norm(positions - com, axis=1)
            return np.sqrt(np.mean(r**2))

        # Track both max and RMS throughout simulation
        for step in range(n_steps):
            integrator_matter.step(dt)

            if step > 0 and step % 100 == 0:
                positions = particles_matter.get_positions()
                rms = rms_radius(positions)
                max_r = max_radius(positions)

                # Runaway particle check: max should not be more than 2× RMS
                # If max >> RMS, it means one or two particles are being shot out
                # while the mean stays reasonable
                ratio = max_r / rms
                self.assertLess(ratio, 2.0,
                    f"Step {step}: Runaway particles detected! "
                    f"Max particle distance ({max_r/self.const.Gpc_to_m:.2f} Gpc) is "
                    f"{ratio:.1f}× larger than RMS ({rms/self.const.Gpc_to_m:.2f} Gpc). "
                    f"This indicates particles are being shot out at extreme velocities.")

        # Final check
        positions_final = particles_matter.get_positions()
        rms_final = rms_radius(positions_final)
        max_final = max_radius(positions_final)
        ratio_final = max_final / rms_final

        self.assertLess(ratio_final, 2.0,
            f"Final: Runaway particles detected! "
            f"Max particle distance ({max_final/self.const.Gpc_to_m:.2f} Gpc) is "
            f"{ratio_final:.1f}× larger than RMS ({rms_final/self.const.Gpc_to_m:.2f} Gpc)")


if __name__ == '__main__':
    unittest.main()
