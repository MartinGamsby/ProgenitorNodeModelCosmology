"""Root pytest configuration."""
import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: mark test as slow (heavy N-body run, skipped by default). "
        "Run with:  pytest -m slow"
    )


def pytest_collection_modifyitems(config, items):
    """Skip slow tests unless the run selects markers explicitly (e.g. -m slow)."""
    if config.getoption("markexpr"):
        return
    skip_slow = pytest.mark.skip(reason="slow; run with  pytest -m slow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
