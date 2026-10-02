"""Calibrate a degradation to an exact Dice loss on a given mask.

Each degradation has its own severity unit (a radius, a fraction of branch points, a flip
probability), and the same severity costs different Dice on different masks. Comparing failures at
a fixed severity therefore compares different amounts of error. `solve_severity` bisects on
severity until the degraded mask reaches a target Dice loss, so that six failures applied to one
mask at one Dice differ only in how the pixels are wrong.

Two choices are deliberate:

  * Calibration is per mask, not global, because a fixed severity costs different Dice on a thin
    and a thick root.
  * A target that cannot be reached returns NaN and is never clipped to the nearest achievable
    value. Junction breaks, for example, cannot remove enough pixels to reach a large loss on a
    sparsely branched mask.

Bisection is valid because Dice loss increases monotonically with severity for all six
degradations; `is_monotone` checks this on a given mask.
"""
import numpy as np

from .degrade import DEGRADATIONS, boundary, junction_break
from .traits import dice

# Default Dice losses to calibrate to: small, where a segmentation still looks acceptable.
DOSE_GRID = (0.005, 0.01, 0.02, 0.05, 0.10, 0.20)

# Search bounds per degradation, in that degradation's own units. The sign convention for
# boundary_erode is negative severity, so its bracket is negative.
SEVERITY_BOUNDS = {
    "boundary_erode": (0.0, -6.0),
    "boundary_dilate": (0.0, 6.0),
    "junction_break": (0.0, 1.0),
    "thin_dropout": (0.0, 0.9),
    "speckle": (0.0, 0.05),
    "blob": (0.0, 3.0),
}


def dice_loss_at(fn, mask, severity, rng_seed=0):
    out = fn(mask, severity, np.random.default_rng(rng_seed))
    if out.sum() == 0:
        return 1.0
    return 1.0 - dice(mask, out)


def solve_severity(fn, mask, target_loss, lo, hi, rng_seed=0, tol=0.0015, max_iter=40):
    """Bisect severity to hit `target_loss` Dice loss. Returns NaN if the target is unreachable.

    `lo` is assumed to give ~0 loss and `hi` the maximum. The bracket is checked rather than
    trusted: if `hi` cannot reach the target, that is reported, not extrapolated.
    """
    f_hi = dice_loss_at(fn, mask, hi, rng_seed)
    if f_hi < target_loss:
        return float("nan"), float(f_hi)        # unreachable within the bracket
    f_lo = dice_loss_at(fn, mask, lo, rng_seed)
    if f_lo > target_loss:
        return float("nan"), float(f_lo)        # even the mildest setting overshoots
    a, b = lo, hi
    for _ in range(max_iter):
        mid = (a + b) / 2.0
        f = dice_loss_at(fn, mask, mid, rng_seed)
        if abs(f - target_loss) <= tol:
            return float(mid), float(f)
        if f < target_loss:
            a = mid
        else:
            b = mid
    mid = (a + b) / 2.0
    return float(mid), float(dice_loss_at(fn, mask, mid, rng_seed))


def calibrate_mask(mask, doses=DOSE_GRID, rng_seed=0, degradations=None):
    """Severity per (degradation, dose) for one mask. Returns a list of records."""
    out = []
    names = degradations or list(DEGRADATIONS)
    for name in names:
        fn = DEGRADATIONS[name][0]
        lo, hi = SEVERITY_BOUNDS[name]
        for dose in doses:
            sev, achieved = solve_severity(fn, mask, dose, lo, hi, rng_seed=rng_seed)
            out.append({
                "degradation": name,
                "target_dice_loss": float(dose),
                "severity": sev,
                "achieved_dice_loss": achieved,
                "reachable": bool(np.isfinite(sev)),
            })
    return out


def is_monotone(fn, mask, lo, hi, n=8, rng_seed=0, slack=0.02):
    """Is Dice loss non-decreasing in |severity|? Bisection is only valid if it is.

    `slack` tolerates the small non-monotonicity that stochastic degradations show between adjacent
    points; the test is that the trend is monotone, not that every step is.
    """
    xs = np.linspace(lo, hi, n)
    ys = [dice_loss_at(fn, mask, float(x), rng_seed) for x in xs]
    drops = [ys[i + 1] - ys[i] for i in range(len(ys) - 1)]
    return all(d >= -slack for d in drops), list(zip([float(x) for x in xs], ys))


__all__ = ["DOSE_GRID", "SEVERITY_BOUNDS", "calibrate_mask", "solve_severity",
           "dice_loss_at", "is_monotone", "boundary", "junction_break"]
