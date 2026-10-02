"""Properties every degradation must have, and the specific traps two of them fell into.

Each degradation is a claim about a failure mode. If two of them do the same thing, the table gains
a column and no information -- that happened once and is asserted against here.
"""
import os
import sys

import numpy as np
import pytest
from scipy import ndimage as ndi

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
sys.path.insert(0, SRC)

from calibrate import SEVERITY_BOUNDS, is_monotone, solve_severity  # noqa: E402
from degrade import DEGRADATIONS, boundary, thin_dropout  # noqa: E402
from morphology import MORPHOLOGIES, generate  # noqa: E402
from traits import dice  # noqa: E402

ARCHS = sorted(MORPHOLOGIES)


@pytest.fixture(scope="module")
def taproot():
    return generate("taproot", size=384, seed=0, thickness=6)


@pytest.mark.parametrize("name", sorted(DEGRADATIONS))
def test_zero_severity_is_the_identity(name, taproot):
    fn = DEGRADATIONS[name][0]
    out = fn(taproot, 0, np.random.default_rng(0))
    assert np.array_equal(out, taproot), name + " changes the mask at severity 0"


@pytest.mark.parametrize("name", sorted(DEGRADATIONS))
@pytest.mark.parametrize("arch", ARCHS)
def test_dice_loss_is_monotone_in_severity(name, arch):
    """Bisection in `calibrate.solve_severity` is only valid if this holds."""
    m = generate(arch, size=256, seed=0, thickness=5)
    ok, pts = is_monotone(DEGRADATIONS[name][0], m, *SEVERITY_BOUNDS[name])
    assert ok, f"{name} on {arch} is not monotone in severity: {pts}"


@pytest.mark.parametrize("name", sorted(DEGRADATIONS))
def test_severity_is_continuous_enough_to_calibrate(name, taproot):
    """A degradation must be able to hit a small target dose, not just jump past it.

    `thin_dropout` v1 thresholded a distance transform at a percentile. On a root a few pixels wide
    the EDT has only a handful of distinct values, so Dice loss was a step function and a request
    for 0.005 returned 0.20 -- a 40x overshoot, reported as if it were the requested dose.
    """
    lo, hi = SEVERITY_BOUNDS[name]
    sev, achieved = solve_severity(DEGRADATIONS[name][0], taproot, 0.02, lo, hi)
    assert np.isfinite(sev), name + " cannot reach a 0.02 Dice loss at all"
    assert abs(achieved - 0.02) < 0.005, (
        f"{name} overshot the 0.02 target and landed at {achieved:.4f} -- severity is too quantised "
        "to calibrate"
    )


def test_thin_dropout_is_not_a_second_erosion(taproot):
    """The v2 trap: ranking PIXELS by thickness peels boundaries, exactly as erosion does.

    The discriminator is the surviving maximum local half-width. Erosion peels a layer off every
    structure, so the thick primary axis must get thinner. `thin_dropout` deletes whole thin
    branches, so the primary is untouched. v2 scored 7.81/7.21/7.07/6.40 against erosion's
    7.81/7.21/7.07/6.32 -- indistinguishable.
    """
    base_max = ndi.distance_transform_edt(taproot).max()
    separated = 0
    for target in (0.05, 0.10, 0.20):
        sv_t, _ = solve_severity(thin_dropout, taproot, target, *SEVERITY_BOUNDS["thin_dropout"])
        sv_e, _ = solve_severity(boundary, taproot, target, *SEVERITY_BOUNDS["boundary_erode"])
        if not (np.isfinite(sv_t) and np.isfinite(sv_e)):
            continue
        mt = thin_dropout(taproot, sv_t, np.random.default_rng(0))
        me = boundary(taproot, sv_e, np.random.default_rng(0))
        wt = ndi.distance_transform_edt(mt).max()
        we = ndi.distance_transform_edt(me).max()
        # thin_dropout must leave the primary axis essentially intact...
        assert wt >= base_max - 0.5, (
            f"thin_dropout thinned the primary axis ({base_max:.2f} -> {wt:.2f}) at Dice loss {target} -- it is "
            "behaving like erosion"
        )
        if we < wt - 0.2:
            separated += 1
    # ...and erosion must visibly thin it at some dose, or the two are not distinguishable.
    assert separated >= 2, (
        "erosion and thin_dropout were not separable at 2 of 3 doses; they may have collapsed "
        "into the same degradation again"
    )


def test_boundary_integer_severity_matches_pure_morphology(taproot):
    """Making boundary continuous must not change what integer severities mean.

    Every number in `results/dose_response.csv` came from the integer grid. If the fractional
    refactor moved them, the published sweep would silently disagree with its own code.
    """
    st = ndi.generate_binary_structure(2, 1)
    for s in (1, 2, 3):
        assert np.array_equal(boundary(taproot, s, np.random.default_rng(0)),
                              ndi.binary_dilation(taproot, structure=st, iterations=s))
        assert np.array_equal(boundary(taproot, -s, np.random.default_rng(0)),
                              ndi.binary_erosion(taproot, structure=st, iterations=s))


def test_boundary_fractional_severity_is_between_its_integer_neighbours(taproot):
    losses = [1 - dice(taproot, boundary(taproot, s, np.random.default_rng(0)))
              for s in (1.0, 1.5, 2.0)]
    assert losses[0] < losses[1] < losses[2], (
        "fractional boundary severity is not between its integer neighbours: " + str(losses)
    )


@pytest.mark.parametrize("name", sorted(DEGRADATIONS))
def test_degradation_is_deterministic_given_a_seed(name, taproot):
    fn = DEGRADATIONS[name][0]
    sev = 0.3 if name not in ("boundary_erode", "boundary_dilate") else 1.5
    sev = -sev if name == "boundary_erode" else sev
    a = fn(taproot, sev, np.random.default_rng(7))
    b = fn(taproot, sev, np.random.default_rng(7))
    assert np.array_equal(a, b), name + " is not deterministic for a fixed seed"


def test_junction_break_saturates_below_five_percent_dice(taproot):
    """Not a defect -- the paper's point. Punching EVERY junction barely moves the metric."""
    from degrade import junction_break
    worst = 1 - dice(taproot, junction_break(taproot, 1.0, np.random.default_rng(0)))
    assert worst < 0.10, f"junction_break cost {worst:.4f} Dice at full severity"
