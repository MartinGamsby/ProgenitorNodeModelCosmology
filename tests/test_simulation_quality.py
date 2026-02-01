"""
Unit tests for simulation quality metrics.

Tests for detecting problematic simulation outcomes:
- Excessive center-of-mass drift (asymmetric tidal forces)
- Numerical instabilities
- Parameter regime validation
"""

import pytest
import numpy as np
from cosmo.constants import SimulationParameters
from cosmo.analysis import calculate_initial_conditions, check_com_drift_quality
from cosmo.simulation import CosmologicalSimulation


class TestCOMDriftQuality:
    """Tests for center-of-mass drift quality detection."""

    def test_minimal_drift_with_symmetric_grid(self):
        """Symmetric external node grid produces minimal COM drift."""
        # Parameters: M=800, S=24
        # With symmetric grid, drift should be negligible regardless of M/S
        sim_params = SimulationParameters(
            M_value=800,
            S_value=24.0,
            n_particles=100,
            seed=42,
            t_start_Gyr=0.5,
            t_duration_Gyr=13.3,
            n_steps=1000
        )

        ic = calculate_initial_conditions(sim_params.t_start_Gyr)
        sim = CosmologicalSimulation(sim_params, ic['box_size_Gpc'], ic['a_start'],
                                      use_external_nodes=True, use_dark_energy=False)
        sim.run()

        # Check drift quality
        quality = check_com_drift_quality(sim.expansion_history)

        # Should have negligible drift with symmetric grid (observed: ~0.01× RMS)
        assert quality['drift_to_size_ratio'] < 0.05, (
            f"Expected negligible drift (<0.05× RMS), got {quality['drift_to_size_ratio']:.3f}"
        )

        # With default threshold of 0.5, should NOT be flagged as excessive
        assert not quality['is_excessive'], (
            f"Drift of {quality['drift_to_size_ratio']:.3f}× RMS should not be flagged"
        )

        # Drift should be very small (<0.2 Gpc)
        assert quality['com_drift_Gpc'] < 0.2, (
            f"Expected tiny drift (<0.2 Gpc), got {quality['com_drift_Gpc']:.2f} Gpc"
        )

    def test_extreme_parameters_still_minimal_drift(self):
        """Even extreme parameters show minimal drift with symmetric grid."""
        # Previously problematic parameters: M=2584, S=41
        # With symmetric grid, drift should still be negligible
        sim_params = SimulationParameters(
            M_value=2584,
            S_value=41.0,
            n_particles=100,
            seed=42,
            t_start_Gyr=0.5,
            t_duration_Gyr=13.3,
            n_steps=1000
        )

        ic = calculate_initial_conditions(sim_params.t_start_Gyr)
        sim = CosmologicalSimulation(sim_params, ic['box_size_Gpc'], ic['a_start'],
                                      use_external_nodes=True, use_dark_energy=False)
        sim.run()

        # Check drift quality
        quality = check_com_drift_quality(sim.expansion_history)

        # Even extreme parameters should have minimal drift with symmetric grid
        assert quality['drift_to_size_ratio'] < 0.05, (
            f"Expected minimal drift even with extreme parameters. "
            f"Got {quality['drift_to_size_ratio']:.3f} × RMS, "
            f"drift={quality['com_drift_Gpc']:.2f} Gpc, "
            f"RMS={quality['final_rms_Gpc']:.2f} Gpc"
        )
        assert not quality['is_excessive'], (
            "Symmetric grid should not trigger excessive drift flag"
        )

        # Drift should be tiny
        assert quality['com_drift_Gpc'] < 0.2, (
            f"Expected tiny drift (<0.2 Gpc), got {quality['com_drift_Gpc']:.2f} Gpc"
        )

    def test_drift_ratio_calculation(self):
        """Verify drift_to_size_ratio is calculated correctly."""
        from cosmo.constants import CosmologicalConstants
        const = CosmologicalConstants()

        # Mock expansion history with known drift
        # COM drift: from (0,0,0) to (3,4,0) Gpc = 5.0 Gpc
        # Final size: 20.0 Gpc diameter (10.0 Gpc RMS)
        mock_history = [
            {'com': np.array([0.0, 0.0, 0.0]), 'diameter_m': 2.0 * const.Gpc_to_m},
            {'com': np.array([1.0, 2.0, 2.0]) * const.Gpc_to_m, 'diameter_m': 10.0 * const.Gpc_to_m},
            {'com': np.array([3.0, 4.0, 0.0]) * const.Gpc_to_m, 'diameter_m': 20.0 * const.Gpc_to_m},  # final
        ]

        quality = check_com_drift_quality(mock_history, drift_threshold=0.3)

        # COM drift: from (0,0,0) to (3,4,0) = 5.0 Gpc
        expected_drift = 5.0
        assert np.isclose(quality['com_drift_Gpc'], expected_drift), (
            f"Expected drift={expected_drift}, got {quality['com_drift_Gpc']}"
        )

        # Final RMS = 20.0 / 2 = 10.0 Gpc
        assert np.isclose(quality['final_rms_Gpc'], 10.0), (
            f"Expected RMS=10.0, got {quality['final_rms_Gpc']}"
        )

        # Ratio = 5.0 / 10.0 = 0.5
        expected_ratio = 0.5
        assert np.isclose(quality['drift_to_size_ratio'], expected_ratio), (
            f"Expected ratio={expected_ratio}, got {quality['drift_to_size_ratio']}"
        )

        # Should exceed threshold of 0.3
        assert quality['is_excessive']
        assert quality['threshold'] == 0.3

    def test_custom_threshold(self):
        """Custom drift thresholds should be respected."""
        from cosmo.constants import CosmologicalConstants
        const = CosmologicalConstants()

        # Drift = 6.0 Gpc, RMS = 10.0 Gpc, ratio = 0.6
        mock_history = [
            {'com': np.array([0.0, 0.0, 0.0]), 'diameter_m': 20.0 * const.Gpc_to_m},
            {'com': np.array([6.0, 0.0, 0.0]) * const.Gpc_to_m, 'diameter_m': 20.0 * const.Gpc_to_m},
        ]

        # With default threshold (0.5), should be excessive
        quality_default = check_com_drift_quality(mock_history)
        assert quality_default['is_excessive']
        assert quality_default['threshold'] == 0.5

        # With high threshold (0.8), should NOT be excessive
        quality_high = check_com_drift_quality(mock_history, drift_threshold=0.8)
        assert not quality_high['is_excessive']
        assert quality_high['threshold'] == 0.8

        # With low threshold (0.3), should be excessive
        quality_low = check_com_drift_quality(mock_history, drift_threshold=0.3)
        assert quality_low['is_excessive']
        assert quality_low['threshold'] == 0.3

    def test_zero_drift_edge_case(self):
        """Perfectly centered simulation should have zero drift."""
        from cosmo.constants import CosmologicalConstants
        const = CosmologicalConstants()

        mock_history = [
            {'com': np.array([0.0, 0.0, 0.0]), 'diameter_m': 10.0 * const.Gpc_to_m},
            {'com': np.array([0.0, 0.0, 0.0]), 'diameter_m': 20.0 * const.Gpc_to_m},
        ]

        quality = check_com_drift_quality(mock_history)

        assert quality['com_drift_Gpc'] == 0.0
        assert quality['drift_to_size_ratio'] == 0.0
        assert not quality['is_excessive']
