"""Where do real segmenters sit in the failure-mode space the calibrated ladder defines?

THE QUESTION F12 LEAVES OPEN
----------------------------
F12 established that at a fixed Dice of 0.99 the error in a trait depends almost entirely on WHICH
failure mode produced that Dice: `n_components` is wrong by 0% under dilation, erosion and
thin-branch loss, and by 15,500% under speckle-like false positives. `pixel_area` is wrong by
2.0-2.3% whichever it was.

That is a statement about a range. It does not say where in the range practice sits, and the paper
is much weaker if it cannot: a reader is entitled to ask whether the catastrophic corner is a real
risk or a theoretical one. Answering it needs real models, which is why this is the GPU arm.

THE METHOD
----------
Both sides are put in the same space, which is the part that makes the comparison mean anything:

  1. Take a sample of real non-empty test masks.
  2. Measure a trained model's mean Dice loss L on them.
  3. Calibrate EACH of the six named degradations to that same L **on those same real masks** --
     not on synthetic roots, and not at some arbitrary severity.
  4. Compute the 12-trait relative-error vector for the model and for each degradation.
  5. Compare the vectors.

So a model and a degradation differ only in *how* they lost that Dice, never in how much, and the
masks are identical. Reading the curves at a matched dose is the same discipline F11 needed when a
near-zero denominator turned a ranking into a coin flip.

The comparison is on log error, because the traits span four orders of magnitude and a correlation
on raw values would be a correlation with `n_components` alone.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# isort: off
import resources  # noqa: F401,E402  MUST load before numpy: caps BLAS threads
# isort: on

import numpy as np  # noqa: E402

from calibrate import SEVERITY_BOUNDS, solve_severity  # noqa: E402
from degrade import DEGRADATIONS  # noqa: E402
from dose_response import rel_error, stable_seed  # noqa: E402
from traits import TRAIT_KIND, TRAITS, dice  # noqa: E402

FLOOR = 1e-4

# Below this |median signed area change| a degradation is treated as directionally neutral and
# stays eligible either way. junction_break removes only a few small disks, so its area change is
# tiny and its sign is not meaningful evidence about direction.
DIRECTION_DEADBAND = 0.02


# Failure-mode FAMILIES, taken from a measured confusion matrix rather than chosen in advance.
#
# Positive control over 96 trials (16 fixtures x 6 degradations; 256 and 320 px; 12 and 20 masks;
# doses 0.03 and 0.05; two replicates each), with the predictions constructed as a known
# degradation so the right answer is known:
#
#                 blob  dilate  erode  junction  speckle  thin    accuracy
#     blob          14      0      0         0        2     0       88%
#     dilate         0     16      0         0        0     0      100%
#     erode          0      0     16         0        0     0      100%
#     junction       0      0      0        16        0     0      100%
#     speckle        1      0      0         0       15     0       94%
#     thin           0      0      0         1        0    15       94%
#
# Individual accuracy 95.8% (92/96). Every error is WITHIN one of two pairs -- {blob, speckle} and
# {junction_break, thin_dropout} -- and none crosses between adding and removing tissue. At the
# family level the method is 96/96.
#
# So the families are the resolution the method actually supports, and reporting the individual
# name as if it were equally certain would overstate it. Both are kept; the family is the claim.
#
# The blob/speckle confusion is NOT a power problem, which is worth knowing before anyone tries to
# fix it with more data: at 192 px it separated at 12 and 20 masks and failed at 32; at 256 px it
# failed at 12, passed at 20 and failed at 32. Non-monotone in sample size means the two signatures
# are close enough that the winner turns on noise. They are the same failure -- spurious foreground
# disconnected from the root -- differing only in the size of the spurious pieces.
FAMILIES = {
    "boundary_dilate": "over_inclusive",
    "boundary_erode": "under_inclusive",
    "speckle": "false_positive",
    "blob": "false_positive",
    "junction_break": "lost_structure",
    "thin_dropout": "lost_structure",
}


def trait_error_vector(gt_masks, other_masks, traits=None):
    """Median relative error per trait, plus the direction of the area change.

    Pairs are positional and must correspond. Non-finite per-frame errors are dropped per trait
    rather than for the whole frame, so one undefined fractal dimension does not remove a frame's
    information about the other eleven traits.

    `_area_direction` is carried alongside because the magnitude vector alone cannot distinguish
    over-segmentation from under-segmentation. Measured: a positive control fed `boundary_dilate`
    at a 0.05 Dice loss came back as `boundary_erode` (+0.960 against +0.908), because dilation and
    erosion damage the same traits by nearly the same amounts and differ only in sign -- which
    `rel_error` takes the absolute value of. Whether a segmenter adds or removes tissue is the most
    basic fact about its failure mode, so it is kept rather than thrown away.
    """
    traits = traits or list(TRAITS)
    acc = {t: [] for t in traits}
    signed_area = []
    for g, o in zip(gt_masks, other_masks):
        base = {t: TRAITS[t](g) for t in traits}
        for t in traits:
            e = rel_error(base[t], TRAITS[t](o))
            if np.isfinite(e):
                acc[t].append(e)
        a0 = base.get("pixel_area", TRAITS["pixel_area"](g))
        if a0 > 0:
            signed_area.append((TRAITS["pixel_area"](o) - a0) / a0)
    out = {t: (float(np.median(v)) if v else np.nan) for t, v in acc.items()}
    out["_area_direction"] = float(np.median(signed_area)) if signed_area else np.nan
    return out


def degradation_vector(gt_masks, target_loss, degradation, seed=0, traits=None):
    """Apply one degradation calibrated to `target_loss` on each real mask; return its error vector.

    Calibration is PER MASK. A single global severity would land at different Dice losses on a
    sparse frame and a dense one, which is exactly the confound F7 removed from the synthetic ladder.
    """
    fn = DEGRADATIONS[degradation][0]
    lo, hi = SEVERITY_BOUNDS[degradation]
    out, kept = [], []
    for i, g in enumerate(gt_masks):
        rs = stable_seed(seed, i, degradation, round(float(target_loss), 6)) % (2 ** 31)
        sev, _ = solve_severity(fn, g, target_loss, lo, hi, rng_seed=rs)
        if not np.isfinite(sev):
            continue                      # this degradation cannot reach that dose on this mask
        out.append(fn(g, sev, np.random.default_rng(rs)))
        kept.append(g)
    if not kept:
        return None, 0
    return trait_error_vector(kept, out, traits=traits), len(kept)


def _logvec(d, traits):
    """Log-magnitude over the named traits only; `_area_direction` is not a magnitude."""
    return np.array([np.log10(max(d.get(t, np.nan), FLOOR))
                     if np.isfinite(d.get(t, np.nan)) else np.nan for t in traits])


def similarity(model_vec, deg_vec, traits):
    """Correlation between two log error vectors over the traits both define.

    Pearson on logs: the question is whether a model damages traits in the same PATTERN as a
    degradation, not whether it damages them by the same amount. A model at a slightly different
    Dice would shift every entry together and should not count as a different fingerprint.
    """
    a, b = _logvec(model_vec, traits), _logvec(deg_vec, traits)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 4 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
        return np.nan, int(ok.sum())
    return float(np.corrcoef(a[ok], b[ok])[0, 1]), int(ok.sum())


def fingerprint(gt_masks, pred_masks, seed=0, traits=None, degradations=None):
    """Which named failure mode does this model's error pattern most resemble?"""
    traits = traits or list(TRAITS)
    degradations = degradations or list(DEGRADATIONS)

    losses = np.array([1.0 - dice(g, p) for g, p in zip(gt_masks, pred_masks)])
    target = float(np.median(losses))
    model_vec = trait_error_vector(gt_masks, pred_masks, traits=traits)

    rows = {}
    for dname in degradations:
        vec, n_ok = degradation_vector(gt_masks, target, dname, seed=seed, traits=traits)
        if vec is None:
            rows[dname] = {"similarity": np.nan, "n_masks": 0, "vector": {}}
            continue
        r, n_traits = similarity(model_vec, vec, traits)
        rows[dname] = {"similarity": r, "n_masks": n_ok, "n_traits": n_traits, "vector": vec}

    # Direction gate. A degradation that moves area the opposite way to the model cannot be its
    # failure mode however well the magnitudes correlate. `DIRECTION_DEADBAND` keeps a degradation
    # that barely moves area (junction_break removes a few disks) eligible either way, so the gate
    # only ever separates genuinely opposed directions.
    md = model_vec.get("_area_direction", np.nan)
    for k, r in rows.items():
        dd = r.get("vector", {}).get("_area_direction", np.nan)
        opposed = (np.isfinite(md) and np.isfinite(dd)
                   and abs(md) > DIRECTION_DEADBAND and abs(dd) > DIRECTION_DEADBAND
                   and np.sign(md) != np.sign(dd))
        r["area_direction"] = None if not np.isfinite(dd) else float(dd)
        r["direction_opposed"] = bool(opposed)

    eligible = [k for k in rows
                if np.isfinite(rows[k]["similarity"]) and not rows[k]["direction_opposed"]]
    ranked = sorted(eligible, key=lambda k: -rows[k]["similarity"])
    all_ranked = sorted((k for k in rows if np.isfinite(rows[k]["similarity"])),
                        key=lambda k: -rows[k]["similarity"])
    return {
        "median_dice_loss": target,
        "mean_dice_loss": float(losses.mean()),
        "n_masks": int(len(gt_masks)),
        "model_area_direction": None if not np.isfinite(md) else float(md),
        "model_vector": model_vec,
        "model_vector_by_kind": {t: TRAIT_KIND[t] for t in traits},
        "degradations": rows,
        "nearest": ranked[0] if ranked else None,
        "nearest_family": FAMILIES.get(ranked[0]) if ranked else None,
        "nearest_ignoring_direction": all_ranked[0] if all_ranked else None,
        "ranking": [(k, rows[k]["similarity"]) for k in ranked],
        "excluded_by_direction": [k for k in rows if rows[k]["direction_opposed"]],
    }
