"""The calibrator must hit its target, refuse what it cannot reach, and never extrapolate.

The failure this guards against is silent: a calibrator that clips an unreachable target to the
nearest achievable value returns a number that looks like a measurement and is not one. Every
"unreachable" path here asserts NaN rather than a clipped value.
"""
import os
import sys

import numpy as np
import pytest

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
sys.path.insert(0, SRC)

import pandas as pd  # noqa: E402

from calibrate import (  # noqa: E402
    DOSE_GRID,
    SEVERITY_BOUNDS,
    calibrate_mask,
    dice_loss_at,
    solve_severity,
)
from deceptiveness import bootstrap_argmax, error_at_dose, table  # noqa: E402
from degrade import DEGRADATIONS, junction_break  # noqa: E402
from morphology import MORPHOLOGIES, generate  # noqa: E402


@pytest.fixture(scope="module")
def taproot():
    return generate("taproot", size=320, seed=0, thickness=6)


@pytest.mark.parametrize("name", sorted(DEGRADATIONS))
@pytest.mark.parametrize("dose", [0.005, 0.02, 0.05])
def test_calibration_hits_its_target(name, dose, taproot):
    lo, hi = SEVERITY_BOUNDS[name]
    sev, achieved = solve_severity(DEGRADATIONS[name][0], taproot, dose, lo, hi)
    if not np.isfinite(sev):
        # Unreachable is a legitimate answer, but only when the bracket genuinely cannot get there.
        f_hi = dice_loss_at(DEGRADATIONS[name][0], taproot, hi)
        assert f_hi < dose, (
            f"{name} reported dose {dose} unreachable, but its maximum Dice loss is {f_hi:.4f}"
        )
        return
    assert abs(achieved - dose) < 0.005, (
        f"{name} calibrated to {achieved:.5f} for a target of {dose}"
    )


def test_unreachable_targets_return_nan_not_a_clipped_value(taproot):
    """junction_break saturates below 0.05 Dice. Asking for 0.20 must fail, not return the max."""
    sev, achieved = solve_severity(junction_break, taproot, 0.20,
                                   *SEVERITY_BOUNDS["junction_break"])
    assert np.isnan(sev), "an unreachable dose returned a severity of " + repr(sev)
    assert achieved < 0.20, "achieved loss should record what the ceiling actually is"


def test_calibrate_mask_marks_reachability_consistently(taproot):
    recs = calibrate_mask(taproot, doses=DOSE_GRID)
    for r in recs:
        assert r["reachable"] == bool(np.isfinite(r["severity"])), (
            "reachable flag disagrees with the severity for " + repr(r)
        )


@pytest.mark.parametrize("arch", sorted(MORPHOLOGIES))
def test_every_architecture_can_be_calibrated_at_the_small_doses(arch):
    """The doses the paper reads the table at must be reachable for most degradations."""
    m = generate(arch, size=256, seed=0, thickness=5)
    recs = calibrate_mask(m, doses=(0.02,))
    reachable = sum(r["reachable"] for r in recs)
    assert reachable >= 5, (
        f"only {reachable}/{len(recs)} degradations reach a 0.02 Dice loss on {arch} -- the matched-dose comparison "
        "would be mostly empty"
    )


# ------------------------------------------------------------------ deceptiveness metric guards
def _toy_curve():
    """Two degradations: `slow` does little damage cheaply, `fast` does a lot at real Dice cost.

    Built so the RATIO statistic prefers `slow` (its denominator is tiny) while the matched-dose
    statistic correctly prefers `fast`. This is the exact shape of the fibrous/speckle artefact.
    """
    rows = []
    for sev, (dl_slow, dl_fast) in enumerate([(0.0005, 0.01), (0.001, 0.02), (0.002, 0.04)], 1):
        rows.append(dict(mask="m", severity=sev, degradation="slow", dice=1 - dl_slow,
                         err_n_branch_points=0.02 * sev))
        rows.append(dict(mask="m", severity=sev, degradation="fast", dice=1 - dl_fast,
                         err_n_branch_points=0.30 * sev))
    return pd.DataFrame(rows)


def test_ratio_statistic_is_fooled_but_matched_dose_is_not():
    df = _toy_curve()
    d = df[df.severity != 0].copy()
    ratio = d.groupby("degradation").apply(
        lambda g: g.err_n_branch_points.mean() / (1 - g.dice).mean(), include_groups=False)
    assert ratio.idxmax() == "slow", "the toy case no longer reproduces the ratio artefact"

    t = table(df, trait="n_branch_points", doses=(0.02,))
    t = t[np.isfinite(t["err_at_dice_loss_0.02"])]
    assert t.loc[t["err_at_dice_loss_0.02"].idxmax(), "degradation"] == "fast", (
        "the matched-dose statistic picked the wrong degradation"
    )


def test_error_at_dose_refuses_to_extrapolate():
    df = _toy_curve()
    slow = df[df.degradation == "slow"]
    # `slow` never reaches a 0.02 Dice loss; asking for it must give NaN, not the endpoint.
    assert np.isnan(error_at_dose(slow, "n_branch_points", 0.02))
    assert np.isfinite(error_at_dose(slow, "n_branch_points", 0.001))


def test_bootstrap_argmax_reports_a_coin_flip_as_a_coin_flip():
    """Two degradations with identical curves must split roughly 50/50, not report a winner."""
    rows = []
    for m in range(6):
        for sev, dl in enumerate([0.01, 0.02, 0.04], 1):
            for deg in ("a", "b"):
                rows.append(dict(mask=f"m{m}", severity=sev, degradation=deg,
                                 dice=1 - dl, err_n_branch_points=0.3 * sev))
    freq, _ = bootstrap_argmax(pd.DataFrame(rows), dose=0.02, n_boot=300, seed=0)
    top = max(freq.values())
    assert top < 0.85, (
        f"two identical degradations produced a {top:.0%} 'winner' -- the bootstrap is not measuring "
        "stability"
    )
