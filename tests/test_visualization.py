"""
Unit tests for cosmo.visualization module

Tests shared visualization utilities for:
- Node position calculation
- 3D plotting helpers
- Filename generation
"""

import unittest
import numpy as np
import tempfile
import os
from cosmo.visualization import (
    get_node_positions,
    generate_output_filename,
    format_simulation_title
)
from cosmo.constants import SimulationParameters


class TestGetNodePositions(unittest.TestCase):
    """Test HMEA node position calculation"""

    def test_node_count(self):
        """Test 26 nodes returned (3×3×3 - 1 center)"""
        S_Gpc = 24.0
        positions = get_node_positions(S_Gpc)

        # Should have 26 nodes (27 - 1 for center)
        self.assertEqual(positions.shape, (26, 3))

    def test_node_positions_centered(self):
        """Test nodes are centered around origin"""
        S_Gpc = 24.0
        positions = get_node_positions(S_Gpc)

        # Center of mass should be at origin
        com = np.mean(positions, axis=0)
        np.testing.assert_array_almost_equal(com, [0, 0, 0], decimal=10)

    def test_node_positions_symmetric(self):
        """Test node positions are symmetric"""
        S_Gpc = 24.0
        positions = get_node_positions(S_Gpc)

        # For each position, negative should also exist
        for pos in positions:
            neg_pos = -pos
            # Check if negative position exists in array
            distances = np.linalg.norm(positions - neg_pos, axis=1)
            min_distance = np.min(distances)
            # Should find exact match (distance ~ 0)
            self.assertLess(min_distance, 1e-10)

    def test_node_spacing(self):
        """Test nodes are spaced by S_Gpc"""
        S_Gpc = 24.0
        positions = get_node_positions(S_Gpc)

        # All coordinates should be multiples of S_Gpc
        unique_coords = np.unique(positions)
        expected_coords = np.array([-S_Gpc, 0, S_Gpc])

        # Check that all unique coordinates match expected
        self.assertEqual(len(unique_coords), 3)
        np.testing.assert_array_almost_equal(sorted(unique_coords), expected_coords)

    def test_no_center_node(self):
        """Test center position (0,0,0) is excluded"""
        S_Gpc = 24.0
        positions = get_node_positions(S_Gpc)

        # Check no node at origin
        distances_from_origin = np.linalg.norm(positions, axis=1)
        min_distance = np.min(distances_from_origin)

        # Minimum distance should be at least S_Gpc (face nodes)
        self.assertGreaterEqual(min_distance, S_Gpc - 1e-10)

    def test_scaling_with_S(self):
        """Test positions scale linearly with S_Gpc"""
        S1 = 10.0
        S2 = 20.0

        pos1 = get_node_positions(S1)
        pos2 = get_node_positions(S2)

        # pos2 should be exactly 2× pos1
        np.testing.assert_array_almost_equal(pos2, pos1 * 2)


class TestGenerateOutputFilename(unittest.TestCase):
    """Test output filename generation"""

    def test_filename_has_extension(self):
        """Test generated filename has correct extension"""
        sim_params = SimulationParameters(
            M_value=800,
            S_value=24.0,
            n_particles=300,
            seed=42,
            t_start_Gyr=10.8,
            t_duration_Gyr=6.0,
            n_steps=150
        )

        filename = generate_output_filename(
            'test',
            sim_params,
            extension='png',
            output_dir='.',
            include_timestamp=False
        )

        # Should end with .png
        self.assertTrue(filename.endswith('.png'))

    def test_filename_contains_parameters(self):
        """Test filename contains key parameters"""
        sim_params = SimulationParameters(
            M_value=800,
            S_value=24.0,
            n_particles=300,
            seed=42,
            t_start_Gyr=10.8,
            t_duration_Gyr=6.0,
            n_steps=150
        )

        filename = generate_output_filename(
            'test',
            sim_params,
            extension='pkl',
            output_dir='.',
            include_timestamp=False
        )

        # Should contain key parameters
        self.assertIn('300p', filename)  # particles
        self.assertIn('800M', filename)  # M value
        self.assertIn('24.0S', filename)  # S value
        self.assertIn('150steps', filename)  # steps
        self.assertIn('10.8-16.8Gyr', filename)  # time range

    def test_filename_output_dir(self):
        """Test filename includes output directory"""
        sim_params = SimulationParameters(
            M_value=800,
            S_value=24.0,
            n_particles=300,
            seed=42,
            t_start_Gyr=10.8,
            t_duration_Gyr=6.0,
            n_steps=150
        )

        output_dir = '/tmp/test'
        filename = generate_output_filename(
            'test',
            sim_params,
            extension='png',
            output_dir=output_dir,
            include_timestamp=False
        )

        # Should start with output_dir
        self.assertTrue(filename.startswith(output_dir))

    def test_filename_without_timestamp(self):
        """Test filename generation without timestamp"""
        sim_params = SimulationParameters(
            M_value=800,
            S_value=24.0,
            n_particles=300,
            seed=42,
            t_start_Gyr=10.8,
            t_duration_Gyr=6.0,
            n_steps=150
        )

        filename = generate_output_filename(
            'test',
            sim_params,
            extension='png',
            output_dir='.',
            include_timestamp=False
        )

        # Should not contain date pattern (YYYY-MM-DD)
        self.assertNotIn('202', filename)  # Simple check for year

    def test_filename_with_timestamp(self):
        """Test filename generation with timestamp"""
        sim_params = SimulationParameters(
            M_value=800,
            S_value=24.0,
            n_particles=300,
            seed=42,
            t_start_Gyr=10.8,
            t_duration_Gyr=6.0,
            n_steps=150
        )

        filename = generate_output_filename(
            'test',
            sim_params,
            extension='png',
            output_dir='.',
            include_timestamp=True
        )

        # Should contain date pattern
        # Format: YYYY-MM-DD_HH.MM.SS
        import re
        pattern = r'\d{4}-\d{2}-\d{2}_\d{2}\.\d{2}\.\d{2}'
        self.assertIsNotNone(re.search(pattern, filename))

    def test_filename_damping_auto(self):
        """Test filename shows 'Auto' for auto-calculated damping"""
        sim_params = SimulationParameters(
            M_value=800,
            S_value=24.0,
            n_particles=300,
            seed=42,
            t_start_Gyr=10.8,
            t_duration_Gyr=6.0,
            n_steps=150,
            damping_factor=None  # Auto-calculate
        )

        filename = generate_output_filename(
            'test',
            sim_params,
            extension='png',
            output_dir='.',
            include_timestamp=False
        )

        # Should contain 'Auto' for damping
        self.assertIn('Autod', filename)

    def test_filename_damping_explicit(self):
        """Test filename shows value for explicit damping"""
        sim_params = SimulationParameters(
            M_value=800,
            S_value=24.0,
            n_particles=300,
            seed=42,
            t_start_Gyr=10.8,
            t_duration_Gyr=6.0,
            n_steps=150,
            damping_factor=0.91
        )

        filename = generate_output_filename(
            'test',
            sim_params,
            extension='png',
            output_dir='.',
            include_timestamp=False
        )

        # Should contain damping value
        self.assertIn('0.91d', filename)


class TestFormatSimulationTitle(unittest.TestCase):
    """Test simulation title formatting"""

    def test_title_contains_parameters(self):
        """Test title contains M and S parameters"""
        sim_params = SimulationParameters(
            M_value=800,
            S_value=24.0,
            n_particles=300,
            seed=42,
            t_start_Gyr=10.8,
            t_duration_Gyr=6.0,
            n_steps=150
        )

        title = format_simulation_title(sim_params)

        # Should contain M and S values
        self.assertIn('800', title)
        self.assertIn('24.0', title)

    def test_title_with_particles(self):
        """Test title includes particle count when requested"""
        sim_params = SimulationParameters(
            M_value=800,
            S_value=24.0,
            n_particles=300,
            seed=42,
            t_start_Gyr=10.8,
            t_duration_Gyr=6.0,
            n_steps=150
        )

        title = format_simulation_title(sim_params, include_particles=True)

        # Should contain particle count
        self.assertIn('300p', title)

    def test_title_without_particles(self):
        """Test title excludes particle count when not requested"""
        sim_params = SimulationParameters(
            M_value=800,
            S_value=24.0,
            n_particles=300,
            seed=42,
            t_start_Gyr=10.8,
            t_duration_Gyr=6.0,
            n_steps=150
        )

        title = format_simulation_title(sim_params, include_particles=False)

        # Should not contain 'p' for particles
        self.assertNotIn('300p', title)


if __name__ == '__main__':
    unittest.main()
