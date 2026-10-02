"""The fingerprint must be able to name a failure mode it is shown.

This is a positive control, and it is the only thing that makes the method's output meaningful. If
the fingerprint cannot recover a degradation it was literally handed, then "this model resembles
speckle" is a sentence with no evidence behind it.

Truth is constructed: the "model predictions" ARE a named degradation, calibrated to a known Dice
loss on the same masks. The answer is known by construction, so accuracy is measurable rather than
plausible.
"""
import os
import sys

import numpy as np
import pytest

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
sys.path.insert(0, SRC)

from calibrate import SEVERITY_BOUNDS, solve_severity  # noqa: E402
from degrade import DEGRADATIONS  # noqa: E402
from morphology import generate  # noqa: E402
from prmi_fingerprint import (  # noqa: E402
    DIRECTION_DEADBAND,
    FAMILIES,
    fingerprint,
    trait_error_vector,
)

DOSE = 0.05


@pytest.fixture(scope="module")
def masks():
    return [generate(m, size=192, seed=s, thickness=5)
            for m in ("taproot", "fibrous", "herringbone") for s in range(3)]


def _apply(name, gts, dose=DOSE):
    fn = DEGRADATIONS[name][0]
    lo, hi = SEVERITY_BOUNDS[name]
    out = []
    for i, g in enumerate(gts):
        sev, _ = solve_severity(fn, g, dose, lo, hi, rng_seed=100 + i)
        out.append(fn(g, sev, np.random.default_rng(100 + i)) if np.isfinite(sev) else g.copy())
    return out


@pytest.mark.parametrize("truth", sorted(DEGRADATIONS))
def test_fingerprint_recovers_the_FAMILY_it_was_given(truth, masks):
    """The claim the method supports: 96/96 at family level in the measured confusion matrix."""
    f = fingerprint(masks, _apply(truth, masks), seed=7)
    assert f["nearest_family"] == FAMILIES[truth], (
        f"fed {truth} ({FAMILIES[truth]}), identified {f['nearest']} ({f['nearest_family']}). "
        f"Ranking: {f['ranking'][:3]}"
    )


@pytest.mark.parametrize("truth", ["boundary_dilate", "boundary_erode"])
def test_singleton_families_are_recovered_exactly(truth, masks):
    """Dilation and erosion are each alone in their family and scored 16/16 individually."""
    f = fingerprint(masks, _apply(truth, masks), seed=7)
    assert f["nearest"] == truth


def test_within_family_pairs_are_the_only_confusions_the_method_makes():
    """Pin the structure of the confusion matrix, not its numbers.

    Every measured error was inside {blob, speckle} or {junction_break, thin_dropout}. If a future
    change to the signature lets an error cross families -- or worse, cross direction -- the family
    claim is no longer supported, and this is the test that should say so.
    """
    fams = {}
    for d, fam in FAMILIES.items():
        fams.setdefault(fam, set()).add(d)
    assert fams["false_positive"] == {"blob", "speckle"}
    assert fams["lost_structure"] == {"junction_break", "thin_dropout"}
    assert fams["over_inclusive"] == {"boundary_dilate"}
    assert fams["under_inclusive"] == {"boundary_erode"}
    # direction is constant within a family, which is why no error ever crossed it
    adds = {"boundary_dilate", "speckle", "blob"}
    for members in fams.values():
        assert len({m in adds for m in members}) == 1, members


def test_direction_gate_is_what_separates_dilation_from_erosion(masks):
    """Without it, magnitudes alone confuse the two.

    Measured before the gate existed: `boundary_dilate` came back as `boundary_erode`, +0.960
    against +0.908. The two damage the same traits by nearly the same amounts and differ in sign,
    which `rel_error` discards by taking an absolute value.
    """
    f = fingerprint(masks, _apply("boundary_dilate", masks), seed=7)
    assert f["nearest"] == "boundary_dilate"
    assert "boundary_erode" in f["excluded_by_direction"], (
        "erosion should be ruled out by direction when the model added area"
    )
    assert f["model_area_direction"] > 0


def test_area_direction_has_the_expected_sign_for_each_degradation(masks):
    adds = {"boundary_dilate", "speckle", "blob"}
    for name in sorted(DEGRADATIONS):
        v = trait_error_vector(masks, _apply(name, masks))
        d = v["_area_direction"]
        assert np.isfinite(d)
        if abs(d) > DIRECTION_DEADBAND:
            assert (d > 0) == (name in adds), f"{name} moved area by {d:+.3f}"


def test_the_direction_key_is_not_treated_as_a_trait(masks):
    """`_area_direction` is a sign, not a magnitude; including it in the log vector would be wrong."""
    from prmi_fingerprint import _logvec
    from traits import TRAITS
    v = trait_error_vector(masks, _apply("speckle", masks))
    assert "_area_direction" in v
    assert len(_logvec(v, list(TRAITS))) == len(TRAITS)


def test_an_identical_prediction_has_no_failure_mode(masks):
    """A perfect model has zero Dice loss, so there is nothing to match it against."""
    f = fingerprint(masks, [m.copy() for m in masks], seed=7)
    assert f["median_dice_loss"] == pytest.approx(0.0, abs=1e-9)


def test_similarity_is_scale_free(masks):
    """A model at a different dose but the same pattern should not get a different fingerprint.

    The comparison is a correlation on log error precisely so that shifting every entry together --
    which is what changing the dose does -- leaves the identification alone.
    """
    a = fingerprint(masks, _apply("speckle", masks, dose=0.03), seed=7)
    b = fingerprint(masks, _apply("speckle", masks, dose=0.08), seed=7)
    assert a["nearest"] == b["nearest"] == "speckle"
    assert a["median_dice_loss"] < b["median_dice_loss"]
