"""What does a model that predicts NOTHING score on PRMI?

The mask census found that a large fraction of PRMI's ground-truth masks are entirely empty --
minirhizotron frames in which no root is visible. Those masks are not a defect; they are real data,
and a segmenter genuinely should output nothing for them.

The problem is what averaging Dice over them does. By the universal convention (and by this
project's own `traits.dice`), two empty masks score **1.0**: no intersection, no union, call it
perfect. So a predictor that outputs an empty mask for every frame scores 1.0 on every empty frame
and 0.0 on every other one, and its mean Dice is exactly the empty rate.

That gives the benchmark a **free floor** that has nothing to do with segmenting roots, and it is
the number every reported mean Dice on PRMI sits on top of. This module measures it, and measures
the alternatives, so a paper can say how much of a headline score is actually earned.

Project 01 ran the same check -- a signal-free null model against a batch-mixing metric -- and it
was the negative result that surfaced the better finding. Same move here.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# isort: off
import resources  # noqa: F401,E402  MUST load before numpy: caps BLAS threads
# isort: on

import argparse  # noqa: E402
import json  # noqa: E402
from collections import defaultdict  # noqa: E402

import numpy as np  # noqa: E402


def _stats(fracs):
    """Scores for the trivial predictors, given each frame's ground-truth foreground fraction.

    For an all-empty prediction P = {} against ground truth G:
        |P| = 0, so Dice = 2*0/(0+|G|) = 0 when G is non-empty, and 1.0 when G is empty too.
    For an all-foreground prediction P = everything:
        Dice = 2|G| / (N + |G|), which is small because roots are a small part of a frame.
    """
    f = np.asarray(fracs, float)
    empty = f == 0
    dice_empty_pred = np.where(empty, 1.0, 0.0)
    dice_full_pred = 2.0 * f / (1.0 + f)
    return {
        "n": int(f.size),
        "empty_rate": float(empty.mean()),
        "mean_dice_predict_nothing": float(dice_empty_pred.mean()),
        "mean_dice_predict_everything": float(dice_full_pred.mean()),
        "mean_dice_nothing_nonempty_only": 0.0,
        "mean_gt_foreground_fraction": float(f.mean()),
    }


def analyse(census_rows):
    out = {"overall": _stats([r["frac"] for r in census_rows])}
    for key in ("config", "official_split", "species"):
        b = defaultdict(list)
        for r in census_rows:
            b[r[key]].append(r["frac"])
        out["by_" + key] = {k: _stats(v) for k, v in sorted(b.items())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--census", default="results/prmi_mask_census_rows.json",
                    help="per-mask rows written by prmi_mask_census.py --save-rows")
    ap.add_argument("--out", default="results/prmi_null_baseline.json")
    a = ap.parse_args()

    with open(a.census) as fh:
        rows = json.load(fh)
    res = analyse(rows)

    o = res["overall"]
    print("=== the free floor ===")
    print(f"  frames                                : {o['n']:,}")
    print(f"  entirely empty ground truth           : {o['empty_rate']:.1%}")
    print(f"  mean Dice of a model predicting NOTHING: **{o['mean_dice_predict_nothing']:.4f}**")
    print(f"  mean Dice of a model predicting ALL    : {o['mean_dice_predict_everything']:.4f}")
    print(f"  mean ground-truth foreground fraction  : {o['mean_gt_foreground_fraction']:.4f}")

    print("\n=== per imaging configuration ===")
    print(f"{'config':32} {'n':>7} {'empty':>8} {'null Dice':>11}")
    for k, v in res["by_config"].items():
        print(f"{k:32} {v['n']:>7} {v['empty_rate']:>7.1%} {v['mean_dice_predict_nothing']:>11.4f}")

    print("\n=== per official split ===")
    for k, v in res["by_official_split"].items():
        print(f"  {k:6} n={v['n']:>7}  empty={v['empty_rate']:.1%}  "
              f"null Dice={v['mean_dice_predict_nothing']:.4f}")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=2)
    print("\nwrote " + a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
