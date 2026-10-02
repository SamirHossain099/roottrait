"""Tests for the trait extractors and the degradation ladder.

The fractal estimator gets the most attention because the brief calls it the technical long pole,
and because a box-counting implementation can be wrong by 0.05-0.10 while still looking sensible.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from degrade import (  # noqa: E402
    DEGRADATIONS,
    blob,
    boundary,
    junction_break,
    speckle,
    synth_root,
)
from traits import (  # noqa: E402
    box_counting_dimension,
    dice,
    iou,
    lacunarity,
    n_branch_points,
    n_components,
    n_holes,
    n_tips,
    pixel_area,
    total_length,
)
from validate_fractal import KNOWN  # noqa: E402


# ------------------------------------------------------------------- overlap
def test_dice_and_iou_identity():
    m = synth_root(128, seed=0)
    assert dice(m, m) == 1.0
    assert iou(m, m) == 1.0


def test_dice_of_disjoint_masks_is_zero():
    a = np.zeros((32, 32), bool)
    b = np.zeros((32, 32), bool)
    a[:16] = True
    b[16:] = True
    assert dice(a, b) == 0.0


def test_dice_of_two_empty_masks_is_one():
    z = np.zeros((16, 16), bool)
    assert dice(z, z) == 1.0


def test_dice_matches_hand_computation():
    a = np.zeros((10, 10), bool)
    b = np.zeros((10, 10), bool)
    a[:, :6] = True          # 60 px
    b[:, 4:] = True          # 60 px, overlap 20
    assert dice(a, b) == pytest.approx(2 * 20 / 120)


# --------------------------------------------------------------- basic traits
def test_pixel_area_counts_foreground():
    m = np.zeros((20, 20), bool)
    m[3:7, 2:5] = True
    assert pixel_area(m) == 12.0


def test_line_has_two_tips_and_no_branches():
    m = np.zeros((41, 41), bool)
    m[20, 5:36] = True
    assert n_tips(m) == 2.0
    assert n_branch_points(m) == 0.0


def test_y_shape_has_a_branch_point_and_three_tips():
    m = np.zeros((61, 61), bool)
    m[30, 10:31] = True
    for k in range(20):
        m[30 - k, 30 + k] = True
        m[30 + k, 30 + k] = True
    assert n_branch_points(m) >= 1
    assert n_tips(m) == 3.0


def test_component_and_hole_counts():
    m = np.zeros((40, 40), bool)
    m[2:8, 2:8] = True
    m[20:38, 20:38] = True
    m[25:33, 25:33] = False          # one hole
    assert n_components(m) == 2.0
    assert n_holes(m) == 1.0


def test_total_length_of_a_straight_line():
    m = np.zeros((41, 41), bool)
    m[20, 5:36] = True
    assert total_length(m) == pytest.approx(31, abs=2)


# ------------------------------------------------------ fractal dimension
@pytest.mark.parametrize("name", list(KNOWN))
def test_box_counting_recovers_known_dimension(name):
    """The long pole: every analytic structure within 0.07 at the finest resolution."""
    fn, truth = KNOWN[name]
    m = fn(6) if name not in ("line", "filled_square") else fn(729)
    est = box_counting_dimension(m)
    assert abs(est - truth) < 0.07, f"{name}: got {est:.4f}, truth {truth:.4f}"


def test_dimension_ordering_is_respected():
    """A line must measure lower-dimensional than a carpet, which must be below a square."""
    from validate_fractal import filled_square, line, sierpinski_carpet
    d_line = box_counting_dimension(line(729))
    d_carpet = box_counting_dimension(sierpinski_carpet(6))
    d_square = box_counting_dimension(filled_square(729))
    assert d_line < d_carpet < d_square


def test_dimension_of_empty_mask_is_nan():
    assert np.isnan(box_counting_dimension(np.zeros((64, 64), bool)))


def test_dimension_returns_nan_rather_than_fitting_two_points():
    assert np.isnan(box_counting_dimension(np.ones((4, 4), bool)))


# ---------------------------------------------------------------- lacunarity
def test_lacunarity_of_a_full_mask_is_exactly_one():
    """Zero variance in box mass: the definitional anchor."""
    assert lacunarity(np.ones((243, 243), bool)) == pytest.approx(1.0)


def test_lacunarity_increases_with_gappiness():
    from validate_fractal import cantor_dust, filled_square
    assert lacunarity(filled_square(243)) < lacunarity(cantor_dust(5))


def test_lacunarity_of_empty_mask_is_nan():
    assert np.isnan(lacunarity(np.zeros((64, 64), bool)))


# ---------------------------------------------------------- degradation ladder
def test_zero_severity_is_the_identity():
    m = synth_root(160, seed=1)
    for fn, grid in DEGRADATIONS.values():
        assert grid[0] == 0
        assert dice(m, fn(m, 0, np.random.default_rng(0))) == 1.0


def test_erosion_shrinks_and_dilation_grows():
    m = synth_root(160, seed=2)
    assert boundary(m, -2).sum() < m.sum() < boundary(m, 2).sum()


def test_junction_break_removes_material_at_branch_points():
    m = synth_root(256, seed=3)
    out = junction_break(m, 0.8, np.random.default_rng(0))
    assert out.sum() < m.sum()
    assert n_components(out) > n_components(m), "breaking junctions must fragment the mask"


def test_junction_break_is_cheap_in_dice_but_expensive_in_topology():
    """The paper's central mechanism, asserted directly."""
    m = synth_root(256, seed=4)
    out = junction_break(m, 0.8, np.random.default_rng(0))
    d = dice(m, out)
    topo = abs(n_components(out) - n_components(m)) / max(n_components(m), 1)
    assert d > 0.9, f"junction breaks should barely move Dice, got {d:.3f}"
    assert topo > d - 0.9, "topology change should dwarf the Dice change"


def test_speckle_adds_only_foreground():
    m = synth_root(160, seed=5)
    out = speckle(m, 0.002, np.random.default_rng(0))
    assert out.sum() > m.sum()
    assert (m & ~out).sum() == 0, "speckle must not remove true foreground"


def test_blob_adds_only_foreground():
    m = synth_root(160, seed=6)
    out = blob(m, 0.5, np.random.default_rng(0))
    assert (m & ~out).sum() == 0


def test_severity_is_monotone_in_dice():
    """A harder degradation must not score better overlap than a milder one."""
    m = synth_root(256, seed=7)
    for name, (fn, grid) in DEGRADATIONS.items():
        ds = [dice(m, fn(m, s, np.random.default_rng(0))) for s in grid]
        assert ds == sorted(ds, reverse=True), f"{name} is not monotone: {ds}"


# ------------------------------------------------------------- synthetic roots
def test_synth_root_is_reproducible_and_branching():
    a = synth_root(200, seed=11)
    b = synth_root(200, seed=11)
    assert (a == b).all()
    assert a.any()
    assert n_branch_points(a) > 0, "a root generator that does not branch is useless here"
    assert n_tips(a) >= 3


def test_synth_root_differs_by_seed():
    assert not (synth_root(200, seed=1) == synth_root(200, seed=2)).all()
