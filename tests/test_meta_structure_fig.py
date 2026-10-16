"""
Tests for the PURE (no-I/O) helper in _gen_fig_meta_structure.py.

Philosophy (mirrors tests/test_ws8_figs.py): no sims, no rendering; tiny
synthetic arrays; deterministic; sub-second.
"""

import numpy as np

from _gen_fig_meta_structure import nearest_indices


def test_nearest_indices_returns_closest_first():
    r = np.array([5.0, 1.0, 3.0, 9.0])
    assert list(nearest_indices(r, 2)) == [1, 2]      # r=1 then r=3


def test_nearest_indices_clamps_k_to_size():
    r = np.array([2.0, 1.0])
    assert list(nearest_indices(r, 10)) == [1, 0]     # all, nearest first


def test_nearest_indices_zero_or_negative_k_is_empty():
    r = np.array([1.0, 2.0, 3.0])
    assert nearest_indices(r, 0).size == 0
    assert nearest_indices(r, -1).size == 0
