"""The calibrated experiment: every degradation compared at the SAME Dice loss.

This is the comparison the paper's claim actually needs and the one the original ladder could not
support. Previously each degradation had its own hand-picked severity grid in its own units, and
those units produced Dice losses spanning 0.016 to 0.524, so "which degradation hides the most
damage behind the metric" could only be answered with a ratio whose denominator went to zero -- and
that ratio was a coin flip (see `deceptiveness.py`).

Here every degradation is solved to a target Dice loss per mask, so a row answers:

    at a Dice loss of exactly 0.02 -- a segmentation any reviewer would wave through --
    how wrong is each trait, and which failure mode hides the most?

Because calibration is per mask, thickness and architecture cannot leak into the dose. A thin
fibrous root and a thick taproot are compared at the same metric cost, not at the same severity.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# isort: off
import resources  # noqa: F401,E402  MUST load before numpy: caps BLAS threads
# isort: on

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from calibrate import DOSE_GRID, SEVERITY_BOUNDS, solve_severity  # noqa: E402
from degrade import DEGRADATIONS  # noqa: E402
from dose_response import rel_error, stable_seed  # noqa: E402
from morphology import MORPHOLOGIES, THICKNESS_LEVELS, generate  # noqa: E402
from traits import TRAIT_KIND, TRAITS, dice  # noqa: E402

KIND_ORDER = ["integral", "fitted", "extremum", "count"]


def run(n_seeds=2, size=384, thicknesses=None, doses=DOSE_GRID, verbose=True):
    thicknesses = thicknesses or list(THICKNESS_LEVELS)
    rows = []
    cells = [(m, t, s) for m in MORPHOLOGIES for t in thicknesses for s in range(n_seeds)]
    t0 = time.time()
    for ci, (morph, tname, seed) in enumerate(cells):
        gt = generate(morph, size=size, seed=seed, thickness=THICKNESS_LEVELS[tname])
        base = {k: fn(gt) for k, fn in TRAITS.items()}
        if verbose:
            print(f"  [{ci + 1}/{len(cells)}] {morph} {tname} seed{seed}  ({time.time() - t0:.0f}s elapsed)", flush=True)
        for dname, (fn, _grid) in DEGRADATIONS.items():
            lo, hi = SEVERITY_BOUNDS[dname]
            for dose in doses:
                key = f"{morph}|{tname}|{seed}"
                rseed = stable_seed(0, key, dname, dose) % (2 ** 31)
                sev, achieved = solve_severity(fn, gt, dose, lo, hi, rng_seed=rseed)
                if not np.isfinite(sev):
                    rows.append(dict(mask=key, morphology=morph, thickness=tname, seed=seed,
                                     degradation=dname, target_dice_loss=float(dose),
                                     severity=np.nan, dice=np.nan, reachable=False))
                    continue
                dm = fn(gt, sev, np.random.default_rng(rseed))
                row = dict(mask=key, morphology=morph, thickness=tname, seed=seed,
                           degradation=dname, target_dice_loss=float(dose),
                           severity=float(sev), dice=float(dice(gt, dm)), reachable=True)
                for k, tfn in TRAITS.items():
                    row["err_" + k] = rel_error(base[k], tfn(dm))
                    row["base_" + k] = base[k]
                rows.append(row)
    return pd.DataFrame(rows)


def fmt3(v):
    return f"{v:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=2)
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--thicknesses", nargs="*", default=None)
    ap.add_argument("--out", default="results/calibrated_sweep.csv")
    a = ap.parse_args()

    df = run(n_seeds=a.n_seeds, size=a.size, thicknesses=a.thicknesses)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    df.to_csv(a.out, index=False)
    ok = df[df.reachable]
    print(f"\nwrote {a.out}  ({len(df)} rows, {len(ok)} reachable)")

    print("\n=== calibration accuracy: achieved vs target Dice loss ===")
    acc = ok.assign(achieved=1 - ok.dice).groupby("target_dice_loss").achieved.agg(
        ["mean", "std", "min", "max"])
    print(acc.to_string(float_format=lambda v: f"{v:.5f}"))

    print("\n=== THE TABLE: trait error at matched Dice loss 0.02 ===")
    at = ok[ok.target_dice_loss == 0.02]
    tab = at.groupby("degradation")[["err_" + t for t in TRAITS]].median().T
    tab.index = [i[4:] for i in tab.index]
    tab.insert(0, "kind", [TRAIT_KIND[t] for t in tab.index])
    tab = tab.sort_values("kind")
    print(tab.to_string(float_format=fmt3))

    print("\n=== which degradation hides the most, at each dose? "
          "(median branch-point error) ===")
    for dose in sorted(ok.target_dice_loss.unique()):
        sub = ok[ok.target_dice_loss == dose]
        r = sub.groupby("degradation").err_n_branch_points.median().sort_values(ascending=False)
        head = ", ".join(f"{k} {v:.3f}" for k, v in r.head(3).items())
        print(f"  Dice loss {dose:<6g} n={len(sub):<3d} winner={r.index[0]:<16s} | top3: {head}")

    print("\n=== does the KIND ordering hold at matched dose, in every architecture? ===")
    kt = (at.melt(id_vars=["morphology"], value_vars=["err_" + t for t in TRAITS],
                  var_name="trait", value_name="err"))
    kt["kind"] = kt.trait.str[4:].map(TRAIT_KIND)
    kk = kt.groupby(["morphology", "kind"]).err.median().unstack()
    kk = kk[[k for k in KIND_ORDER if k in kk.columns]]
    print(kk.to_string(float_format=fmt3))
    holds = bool((kk["integral"] < kk["extremum"]).all() and (kk["integral"] < kk["count"]).all())
    print(f"  integrals below both extrema and counts in every architecture: {holds}")

    print("\n=== junction_break saturation: the most it can ever cost ===")
    jb = df[df.degradation == "junction_break"]
    reach = jb.groupby("target_dice_loss").reachable.mean()
    print(reach.to_string(float_format=lambda v: f"{v:.2f}"))

    summary = {
        "n_rows": int(len(df)),
        "n_reachable": int(len(ok)),
        "doses": list(DOSE_GRID),
        "kind_ordering_holds_at_matched_dose": holds,
        "calibration_max_abs_error": float((1 - ok.dice - ok.target_dice_loss).abs().max()),
        "winner_by_dose": {
            str(d): str(ok[ok.target_dice_loss == d]
                        .groupby("degradation").err_n_branch_points.median().idxmax())
            for d in sorted(ok.target_dice_loss.unique())
        },
    }
    with open(a.out.replace(".csv", "_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print("\nwrote " + a.out.replace(".csv", "_summary.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
