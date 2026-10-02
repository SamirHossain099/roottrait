"""Pinhole filling: fills what it should, leaves what it should not. See F25 and C12."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from prmi_branching import junction_nodes, prune  # noqa: E402
from prmi_clean import enclosed_holes, fill_pinholes  # noqa: E402
from traits import skeleton  # noqa: E402


def _root_with_pinhole():
    m = np.zeros((40, 60), bool)
    m[17:24, 5:55] = True              # a 7 px wide horizontal root
    m[20, 30] = False                  # one-pixel hole inside it
    return m


def test_a_one_pixel_hole_is_filled():
    m = _root_with_pinhole()
    assert enclosed_holes(m) == [1]
    assert enclosed_holes(fill_pinholes(m, 10)) == []


def test_the_pinhole_is_what_creates_the_junctions():
    """The mechanism behind C12: a hole splits the skeleton into a loop, and pruning cannot remove it."""
    m = _root_with_pinhole()
    with_hole = junction_nodes(prune(skeleton(m), 15))
    filled = junction_nodes(prune(skeleton(fill_pinholes(m, 10)), 15))
    assert with_hole > filled == 0


def test_large_holes_and_open_background_are_kept():
    m = np.zeros((60, 60), bool)
    m[10:50, 10:50] = True
    m[20:40, 20:40] = False            # 400 px hole: a real gap enclosed by roots, keep it
    out = fill_pinholes(m, 10)
    assert enclosed_holes(out) == [400]
    assert not out[0, 0]               # background touching the border is never filled


def test_zero_disables_and_input_is_not_modified():
    m = _root_with_pinhole()
    before = m.copy()
    assert np.array_equal(fill_pinholes(m, 0), m)
    fill_pinholes(m, 10)
    assert np.array_equal(m, before)
