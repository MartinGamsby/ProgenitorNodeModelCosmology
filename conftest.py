"""Root pytest configuration."""
import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: mark test as slow (heavy N-body run, skipped by default). "
        "Run with:  pytest -m slow"
    )
