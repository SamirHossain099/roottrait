"""Which degradation hides the most damage behind the metric -- measured at MATCHED Dice loss.

WHY THIS MODULE EXISTS
----------------------
The first version of this comparison divided mean topology error by mean Dice loss:

    topology_per_dice_lost = mean(err_n_branch_points) / mean(1 - dice)

That statistic is unstable, and the instability is not academic. Across the morphology sweep,
`speckle` has a mean Dice loss of 0.016 (minimum 0.00022) while `junction_break` has 0.049. Any
ratio with a near-zero denominator is dominated by the denominator, so in the `fibrous` architecture
`speckle` overtook `junction_break` (12.37 vs 12.07) while doing SEVEN TIMES LESS absolute topology
damage (branch error 0.055 vs 0.385). Bootstrapped over masks, `speckle` was the argmax in 56.8% of
2000 resamples -- a coin flip reported as a finding.

Taken at face value it would have produced the claim "junction_break is not the most deceptive
degradation in every architecture," which is false, and false in the interesting direction: it would
have looked like a nuanced replication failure rather than a division artifact.

THE FIX
-------
This is a dose-response experiment, so read every curve at a common dose. For each degradation,
interpolate the trait error at a fixed Dice loss and compare those. A degradation whose observed
range does not bracket the target dose is reported as NOT COMPARABLE rather than extrapolated --
extrapolating is how the near-zero denominator got in.

The target doses are deliberately small (0.02 and 0.05 Dice). The paper's claim is about damage
hidden behind a metric that looks fine, so the comparison belongs where the metric still looks fine.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# isort: off
import resources  # noqa: F401,E402  MUST load before numpy: caps BLAS threads
# isort: on

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

DEFAULT_DOSES = (0.02, 0.05)


def error_at_dose(sub, trait, dose):
    """Interpolate `err_<trait>` at Dice loss == `dose`.

    Returns NaN when the degradation's observed Dice-loss range does not bracket `dose`, so a
    degradation that never reaches the dose is excluded rather than extrapolated.
    """
    d = sub.copy()
    d["dice_loss"] = 1.0 - d["dice"]
    d = d[np.isfinite(d["dice_loss"]) & np.isfinite(d["err_" + trait])]
    if len(d) < 2:
        return np.nan
    # Average replicate masks at each severity first, so interpolation runs on the dose-response
    # curve rather than on a scatter of individual masks.
    curve = d.groupby("severity").agg(dice_loss=("dice_loss", "mean"),
                                      err=("err_" + trait, "mean")).sort_values("dice_loss")
    lo, hi = curve.dice_loss.min(), curve.dice_loss.max()
    if not (lo <= dose <= hi):
        return np.nan
    return float(np.interp(dose, curve.dice_loss.values, curve.err.values))


def table(df, trait="n_branch_points", doses=DEFAULT_DOSES, by=None):
    """Trait error at each matched dose, per degradation (optionally within each level of `by`)."""
    keys = ([by] if by else []) + ["degradation"]
    rows = []
    for key, sub in df[df.severity != 0].groupby(keys):
        rec = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
        for dose in doses:
            rec[f"err_at_dice_loss_{dose:g}"] = error_at_dose(sub, trait, dose)
        rec["max_dice_loss"] = float((1 - sub.dice).max())
        rows.append(rec)
    return pd.DataFrame(rows)


def bootstrap_argmax(df, trait="n_branch_points", dose=0.02, by_col="mask",
                     n_boot=2000, seed=0):
    """How often is each degradation the most deceptive, resampling masks with replacement?

    Reports stability rather than a point ranking. A degradation that wins 57% of resamples is a
    coin flip, and saying so is the whole point of this function.
    """
    rng = np.random.default_rng(seed)
    units = df[by_col].unique()
    counts = {}
    n_valid = 0
    for _ in range(n_boot):
        pick = rng.choice(units, size=len(units), replace=True)
        b = pd.concat([df[df[by_col] == u] for u in pick], ignore_index=True)
        scores = {}
        for deg, sub in b[b.severity != 0].groupby("degradation"):
            v = error_at_dose(sub, trait, dose)
            if np.isfinite(v):
                scores[deg] = v
        if not scores:
            continue
        n_valid += 1
        # Break ties AT RANDOM. `max()` returns the first key it meets, which on a tie is decided
        # by groupby's alphabetical order -- so two degradations with identical curves would report
        # a 100% winner for whichever name sorts first. A tie is the one result this function most
        # needs to be able to express.
        best_val = max(scores.values())
        tied = [k for k, v in scores.items() if v >= best_val - 1e-12]
        best = tied[0] if len(tied) == 1 else str(rng.choice(sorted(tied)))
        counts[best] = counts.get(best, 0) + 1
    return {k: v / max(n_valid, 1) for k, v in
            sorted(counts.items(), key=lambda kv: -kv[1])}, n_valid


def most_deceptive(df, trait="n_branch_points", dose=0.02):
    """The degradation with the highest trait error at the matched dose, or None if undecidable."""
    t = table(df, trait=trait, doses=(dose,))
    col = f"err_at_dice_loss_{dose:g}"
    t = t[np.isfinite(t[col])]
    if t.empty:
        return None
    return str(t.loc[t[col].idxmax(), "degradation"])
