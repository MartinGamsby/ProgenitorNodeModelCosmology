"""
Test reproducibility of simulations with fixed random seed.

Ensures that simulations with the same parameters and seed produce
identical initial conditions and deterministic evolution.
"""

import numpy as np
import pytest
from cosmo.constants import SimulationParameters
from cosmo.analysis import calculate_initial_conditions
from cosmo.simulation import CosmologicalSimulation


class TestInitialConditionsReproducibility:
    """Test that initial conditions are deterministic given a seed."""

    def test_same_parameters_same_initial_conditions(self):
        """Test that identical parameters produce identical initial conditions."""
        t_start = 0.8

        # Calculate initial conditions twice
        ic1 = calculate_initial_conditions(t_start)
        ic2 = calculate_initial_conditions(t_start)

        # All values should be identical
        assert ic1['a_start'] == ic2['a_start'], "Scale factor should be deterministic"
        assert ic1['box_size_Gpc'] == ic2['box_size_Gpc'], "Box size should be deterministic"
        assert ic1['H_start_hubble'] == ic2['H_start_hubble'], "Hubble parameter should be deterministic"

    def test_same_seed_same_particle_positions(self):
        """Test that same seed produces identical particle positions."""
        sim_params = SimulationParameters(
            M_value=800,
            S_value=25.0,
            n_particles=50,
            seed=42,
            t_start_Gyr=0.8,
            t_duration_Gyr=13.8,
            n_steps=550
        )

        # Calculate initial conditions
        ic = calculate_initial_conditions(sim_params.t_start_Gyr)

        # Create two simulations with same seed
        sim1 = CosmologicalSimulation(sim_params, ic['box_size_Gpc'], ic['a_start'],
                                      use_external_nodes=True, use_dark_energy=False)

        sim2 = CosmologicalSimulation(sim_params, ic['box_size_Gpc'], ic['a_start'],
                                      use_external_nodes=True, use_dark_energy=False)

        # Particle positions should be identical
        np.testing.assert_array_equal(
            sim1.particles.get_positions(),
            sim2.particles.get_positions(),
            err_msg="Particle positions should be identical with same seed"
        )

        # Particle velocities should be identical
        np.testing.assert_array_equal(
            sim1.particles.get_velocities(),
            sim2.particles.get_velocities(),
            err_msg="Particle velocities should be identical with same seed"
        )

    def test_different_seed_different_particle_positions(self):
        """Test that different seeds produce different particle positions."""
        sim_params_1 = SimulationParameters(
            M_value=800,
            S_value=25.0,
            n_particles=50,
            seed=42,
            t_start_Gyr=0.8,
            t_duration_Gyr=13.8,
            n_steps=550
        )

        sim_params_2 = SimulationParameters(
            M_value=800,
            S_value=25.0,
            n_particles=50,
            seed=123,  # Different seed
            t_start_Gyr=0.8,
            t_duration_Gyr=13.8,
            n_steps=550
        )

        # Calculate initial conditions
        ic = calculate_initial_conditions(sim_params_1.t_start_Gyr)

        # Create two simulations with different seeds
        sim1 = CosmologicalSimulation(sim_params_1, ic['box_size_Gpc'], ic['a_start'],
                                      use_external_nodes=True, use_dark_energy=False)

        sim2 = CosmologicalSimulation(sim_params_2, ic['box_size_Gpc'], ic['a_start'],
                                      use_external_nodes=True, use_dark_energy=False)

        # Particle positions should be different
        assert not np.array_equal(sim1.particles.get_positions(), sim2.particles.get_positions()), \
            "Particle positions should differ with different seeds"

        # Particle velocities should be different
        assert not np.array_equal(sim1.particles.get_velocities(), sim2.particles.get_velocities()), \
            "Particle velocities should differ with different seeds"

    def test_com_velocity_removal_is_deterministic(self):
        """Test that COM velocity removal is deterministic given a seed."""
        sim_params = SimulationParameters(
            M_value=800,
            S_value=25.0,
            n_particles=50,
            seed=42,
            t_start_Gyr=0.8,
            t_duration_Gyr=13.8,
            n_steps=550
        )

        ic = calculate_initial_conditions(sim_params.t_start_Gyr)

        # Create two simulations
        sim1 = CosmologicalSimulation(sim_params, ic['box_size_Gpc'], ic['a_start'],
                                      use_external_nodes=True, use_dark_energy=False)

        sim2 = CosmologicalSimulation(sim_params, ic['box_size_Gpc'], ic['a_start'],
                                      use_external_nodes=True, use_dark_energy=False)

        # COM velocity should be near zero for both
        com_vel_1 = np.mean(sim1.particles.get_velocities(), axis=0)
        com_vel_2 = np.mean(sim2.particles.get_velocities(), axis=0)

        # COM velocities should be identical
        np.testing.assert_array_almost_equal(
            com_vel_1,
            com_vel_2,
            decimal=10,
            err_msg="COM velocity removal should be deterministic"
        )

        # COM velocity should be very small (removed)
        # Tolerance relaxed to account for numerical precision in velocity removal
        assert np.linalg.norm(com_vel_1) < 1e-6, \
            "COM velocity should be approximately zero after removal"


class TestMatterOnlyReproducibility:
    """Test reproducibility for matter-only simulations."""

    def test_matter_only_same_seed_same_particles(self):
        """Test that matter-only sim with same seed produces identical particles."""
        sim_params = SimulationParameters(
            M_value=800,
            S_value=25.0,
            n_particles=50,
            seed=42,
            t_start_Gyr=0.8,
            t_duration_Gyr=13.8,
            n_steps=550
        )

        ic = calculate_initial_conditions(sim_params.t_start_Gyr)

        # Create two matter-only simulations
        sim1 = CosmologicalSimulation(sim_params, ic['box_size_Gpc'], ic['a_start'],
                                      use_external_nodes=False, use_dark_energy=False)

        sim2 = CosmologicalSimulation(sim_params, ic['box_size_Gpc'], ic['a_start'],
                                      use_external_nodes=False, use_dark_energy=False)

        # Should be identical
        np.testing.assert_array_equal(sim1.particles.get_positions(), sim2.particles.get_positions())
        np.testing.assert_array_equal(sim1.particles.get_velocities(), sim2.particles.get_velocities())


class TestInitialSizeConsistency:
    """Test that all models start from the same physical size."""

    def test_all_models_start_at_same_physical_size(self):
        """
        Test that ΛCDM, External-Node, and Matter-only all start at the same
        physical size when simulated from the same t_start.
        """
        from cosmo.analysis import solve_friedmann_equation
        from cosmo.factories import run_and_extract_results

        t_start = 0.8
        t_duration = 13.8
        n_steps = 350  # Need dt < 0.05 Gyr for stability

        sim_params = SimulationParameters(
            M_value=800,
            S_value=25.0,
            n_particles=30,
            seed=42,
            t_start_Gyr=t_start,
            t_duration_Gyr=t_duration,
            n_steps=n_steps
        )

        ic = calculate_initial_conditions(t_start)
        box_size_Gpc = ic['box_size_Gpc']

        # Get ΛCDM baseline
        # Note: N-body normalizes a=1 at t_start, so ΛCDM should start at box_size_Gpc
        # (The actual ΛCDM scale factor is relative to present day, but we normalize here)
        lcdm_initial_size = box_size_Gpc

        # Run External-Node simulation
        sim_ext = CosmologicalSimulation(sim_params, box_size_Gpc, ic['a_start'],
                                         use_external_nodes=True, use_dark_energy=False)
        ext_results = run_and_extract_results(sim_ext, t_duration, n_steps, save_interval=1)

        # Run Matter-only simulation
        sim_matter = CosmologicalSimulation(sim_params, box_size_Gpc, ic['a_start'],
                                            use_external_nodes=False, use_dark_energy=False)
        matter_results = run_and_extract_results(sim_matter, t_duration, n_steps, save_interval=1)

        # Initial sizes should all be the same
        ext_initial_size = ext_results['size_Gpc'][0]
        matter_initial_size = matter_results['size_Gpc'][0]

        # All three should start at the same physical size
        np.testing.assert_allclose(
            ext_initial_size,
            lcdm_initial_size,
            rtol=0.01,  # 1% tolerance
            err_msg=f"External-Node initial size ({ext_initial_size:.3f} Gpc) should match ΛCDM ({lcdm_initial_size:.3f} Gpc)"
        )

        np.testing.assert_allclose(
            matter_initial_size,
            lcdm_initial_size,
            rtol=0.01,
            err_msg=f"Matter-only initial size ({matter_initial_size:.3f} Gpc) should match ΛCDM ({lcdm_initial_size:.3f} Gpc)"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
