"""
Unit tests for CLI argument parsing module.
"""

import unittest
import sys
import argparse
from unittest.mock import patch
from cosmo.cli import add_common_arguments, parse_arguments, args_to_sim_params
from cosmo.constants import SimulationParameters, CosmologicalConstants


class TestAddCommonArguments(unittest.TestCase):
    """Test add_common_arguments function"""

    def test_adds_all_expected_arguments(self):
        """All common arguments should be added to parser"""
        parser = argparse.ArgumentParser()
        add_common_arguments(parser)

        # Parse with defaults
        args = parser.parse_args([])

        # Check all expected attributes exist
        self.assertTrue(hasattr(args, 'M'))
        self.assertTrue(hasattr(args, 'S'))
        self.assertTrue(hasattr(args, 'particles'))
        self.assertTrue(hasattr(args, 'seed'))
        self.assertTrue(hasattr(args, 't_start'))
        self.assertTrue(hasattr(args, 't_duration'))
        self.assertTrue(hasattr(args, 'n_steps'))
        self.assertTrue(hasattr(args, 'damping'))
        self.assertTrue(hasattr(args, 'center_node_mass'))
        self.assertTrue(hasattr(args, 'compare'))

    def test_default_values(self):
        """Default values should match expected"""
        parser = argparse.ArgumentParser()
        add_common_arguments(parser)
        args = parser.parse_args([])

        self.assertEqual(args.M, 855)
        self.assertEqual(args.S, 25.0)
        self.assertEqual(args.particles, 200)
        self.assertEqual(args.seed, 42)
        self.assertEqual(args.t_start, 5.8)
        self.assertEqual(args.t_duration, 8.0)
        self.assertEqual(args.n_steps, 250)
        self.assertEqual(args.damping, None)
        self.assertEqual(args.center_node_mass, 1.0)
        self.assertFalse(args.compare)

    def test_custom_values_override_defaults(self):
        """Custom CLI values should override defaults"""
        parser = argparse.ArgumentParser()
        add_common_arguments(parser)
        args = parser.parse_args([
            '--M', '1000',
            '--S', '30.0',
            '--particles', '500',
            '--seed', '123',
            '--center-node-mass', '2.5',
            '--compare'
        ])

        self.assertEqual(args.M, 1000)
        self.assertEqual(args.S, 30.0)
        self.assertEqual(args.particles, 500)
        self.assertEqual(args.seed, 123)
        self.assertEqual(args.center_node_mass, 2.5)
        self.assertTrue(args.compare)

    def test_time_parameters(self):
        """Time parameters should be configurable"""
        parser = argparse.ArgumentParser()
        add_common_arguments(parser)
        args = parser.parse_args([
            '--t-start', '5.0',
            '--t-duration', '8.0',
            '--n-steps', '200'
        ])

        self.assertEqual(args.t_start, 5.0)
        self.assertEqual(args.t_duration, 8.0)
        self.assertEqual(args.n_steps, 200)

    def test_damping_custom_value(self):
        """Damping factor should be configurable"""
        parser = argparse.ArgumentParser()
        add_common_arguments(parser)
        args = parser.parse_args(['--damping', '0.85'])

        self.assertEqual(args.damping, 0.85)


class TestArgsToSimParams(unittest.TestCase):
    """Test args_to_sim_params conversion"""

    def test_converts_args_to_sim_params(self):
        """Should create SimulationParameters from parsed args"""
        parser = argparse.ArgumentParser()
        add_common_arguments(parser)
        args = parser.parse_args(['--M', '800', '--S', '24.0', '--center-node-mass', '1.5'])

        sim_params = args_to_sim_params(args)

        self.assertIsInstance(sim_params, SimulationParameters)
        self.assertEqual(sim_params.M_value, 800)
        self.assertEqual(sim_params.S_value, 24.0)
        self.assertEqual(sim_params.center_node_mass, 1.5)

    def test_center_node_mass_derived_values(self):
        """center_node_mass_kg should be calculated correctly"""
        parser = argparse.ArgumentParser()
        add_common_arguments(parser)
        args = parser.parse_args(['--center-node-mass', '2.0'])

        sim_params = args_to_sim_params(args)

        const = CosmologicalConstants()
        expected_kg = 2.0 * const.M_observable_kg

        self.assertAlmostEqual(sim_params.center_node_mass_kg, expected_kg, places=5)

    def test_all_args_transferred(self):
        """All CLI args should transfer to SimulationParameters"""
        parser = argparse.ArgumentParser()
        add_common_arguments(parser)
        args = parser.parse_args([
            '--M', '900',
            '--S', '26.0',
            '--particles', '400',
            '--seed', '99',
            '--t-start', '4.0',
            '--t-duration', '9.0',
            '--n-steps', '180',
            '--damping', '0.9',
            '--center-node-mass', '1.2'
        ])

        sim_params = args_to_sim_params(args)

        self.assertEqual(sim_params.M_value, 900)
        self.assertEqual(sim_params.S_value, 26.0)
        self.assertEqual(sim_params.n_particles, 400)
        self.assertEqual(sim_params.seed, 99)
        self.assertEqual(sim_params.t_start_Gyr, 4.0)
        self.assertEqual(sim_params.t_duration_Gyr, 9.0)
        self.assertEqual(sim_params.n_steps, 180)
        self.assertEqual(sim_params.damping_factor, 0.9)
        self.assertEqual(sim_params.center_node_mass, 1.2)


class TestParseArguments(unittest.TestCase):
    """Test parse_arguments function"""

    def test_with_output_dir(self):
        """Should include --output-dir when add_output_dir=True"""
        with patch.object(sys, 'argv', ['prog', '--output-dir', '/tmp/test']):
            args = parse_arguments(add_output_dir=True)
            self.assertEqual(args.output_dir, '/tmp/test')

    def test_without_output_dir(self):
        """Should not include --output-dir when add_output_dir=False"""
        with patch.object(sys, 'argv', ['prog']):
            args = parse_arguments(add_output_dir=False)
            self.assertFalse(hasattr(args, 'output_dir'))

    def test_default_description(self):
        """Default description should be used"""
        with patch.object(sys, 'argv', ['prog']):
            # Just verify it doesn't raise an error
            args = parse_arguments(add_output_dir=False)
            self.assertIsNotNone(args)

    def test_custom_description(self):
        """Custom description should be accepted"""
        with patch.object(sys, 'argv', ['prog']):
            args = parse_arguments(
                description='Custom Description',
                add_output_dir=False
            )
            self.assertIsNotNone(args)


if __name__ == '__main__':
    unittest.main()
