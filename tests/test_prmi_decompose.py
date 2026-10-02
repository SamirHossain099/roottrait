"""The pixel decomposition must recover pure failure modes AND known mixtures in their proportions.

The second half is the point. The nearest-mode fingerprint passed a positive control on pure
degradations and then mislabelled every mixture it was shown -- including the area-neutral mixture
that matches what real segmenters do. A method validated only on pure inputs is validated on the
wrong inputs.
"""
import os
import sys

import numpy as np
import pytest

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
sys.path.insert(0, SRC)

from calibrate import SEVERITY_BOUNDS, solve_severity  # noqa: E402
from degrade import DEGRADATIONS  # noqa: E402
from morphology import MORPHOLOGIES, generate  # noqa: E402
from prmi_decompose import KINDS, decompose, decompose_set, dominant, shares  # noqa: E402
from prmi_fingerprint import FAMILIES  # noqa: E402


@pytest.fixture(scope="module")
def gts():
    return [generate(m, size=192, seed=s, thickness=5) for m in MORPHOLOGIES for s in range(3)]


def _one(name, g, dose, rs):
    fn = DEGRADATIONS[name][0]
    lo, hi = SEVERITY_BOUNDS[name]
    s, _ = solve_severity(fn, g, dose, lo, hi, rng_seed=rs)
    return fn(g, s, np.random.default_rng(rs)) if np.isfinite(s) else g.copy()


# ------------------------------------------------------------------ unit behaviour
def test_a_perfect_prediction_has_no_errors():
    g = generate("taproot", size=128, seed=0, thickness=5)
    assert sum(decompose(g, g.copy()).values()) == 0


def test_every_wrong_pixel_is_counted_exactly_once():
    """The four kinds partition the error: they must sum to |pred XOR gt|."""
    g = generate("taproot", size=128, seed=0, thickness=5)
    p = _one("speckle", g, 0.05, 1) | _one("boundary_erode", g, 0.05, 2)
    c = decompose(g, p)
    assert sum(c.values()) == int((g ^ p).sum())


def test_shares_sum_to_one():
    s = shares({"over_inclusive": 3, "false_positive": 1, "under_inclusive": 4,
                "lost_structure": 2})
    assert sum(s.values()) == pytest.approx(1.0)
    assert shares(dict.fromkeys(KINDS, 0)) == dict.fromkeys(KINDS, 0.0)


def test_attached_and_detached_false_positives_are_separated():
    g = np.zeros((40, 40), bool)
    g[5:35, 19:21] = True                      # a vertical root
    p = g.copy()
    p[5:35, 21] = True                         # touching it: over-inclusive
    p[2, 2] = True                             # far away: a spurious false positive
    c = decompose(g, p)
    assert c["over_inclusive"] == 30
    assert c["false_positive"] == 1


def test_boundary_loss_and_centreline_loss_are_separated():
    g = np.zeros((40, 40), bool)
    g[5:35, 15:25] = True                      # a thick root, centreline near column 19-20
    p = g.copy()
    p[5:35, 15] = False                        # shave one boundary column: under-inclusive
    c1 = decompose(g, p)
    assert c1["under_inclusive"] > 0 and c1["lost_structure"] == 0
    p2 = g.copy()
    p2[18:21, :] = False                       # cut straight through: lost structure
    c2 = decompose(g, p2)
    assert c2["lost_structure"] > 0


# ------------------------------------------------------------------ pure positive control
@pytest.mark.parametrize("name", sorted(DEGRADATIONS))
def test_a_pure_degradation_lands_in_its_own_family(name, gts):
    r = decompose_set(gts, [_one(name, g, 0.05, i) for i, g in enumerate(gts)])
    assert dominant(r["pixel_weighted"]) == FAMILIES[name], r["pixel_weighted"]


# ------------------------------------------------------------------ mixtures: the real test
@pytest.mark.parametrize("a,b", [
    ("boundary_dilate", "boundary_erode"),       # the area-neutral case real models resemble
    ("boundary_erode", "speckle"),               # the fingerprint called this thin_dropout
    ("boundary_erode", "junction_break"),
])
def test_a_half_and_half_mixture_is_recovered_in_roughly_equal_shares(a, b, gts):
    """Both components must be clearly present, neither swamped by the other.

    Not exactly 50/50: the two degradations are calibrated to the same Dice loss, not to the same
    pixel count, and a few blob/speckle pixels genuinely land touching the root. But a method that
    reports one component and drops the other has failed, which is what nearest-mode matching did.
    """
    preds = [_one(a if i % 2 else b, g, 0.05, i) for i, g in enumerate(gts)]
    r = decompose_set(gts, preds)["pixel_weighted"]
    fa, fb = FAMILIES[a], FAMILIES[b]
    assert r[fa] > 0.25, f"{a} ({fa}) under-represented: {r}"
    assert r[fb] > 0.25, f"{b} ({fb}) under-represented: {r}"


def test_frame_and_pixel_weighting_are_both_reported(gts):
    r = decompose_set(gts, [_one("speckle", g, 0.05, i) for i, g in enumerate(gts)])
    assert set(r["pixel_weighted"]) == set(KINDS)
    assert set(r["frame_weighted"]) == set(KINDS)
    assert r["n_frames"] == len(gts)
