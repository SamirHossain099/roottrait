"""F25: one-pixel holes in PRMI's ground-truth masks, censused at native resolution.

Three questions, each written to `results/prmi_pinholes.json`:

  1. How common are enclosed holes of at most `PINHOLE_PX` pixels, per imaging configuration?
     (Every non-empty mask, native resolution -- no resizing, so this is the dataset, not our cache.)
  2. How much does filling them move Dice between the annotation and its own filled copy?
  3. How much does filling them move the skeleton junction count? (A sample per configuration;
     skeletonising every native frame would take hours and the answer does not need it.)

Question 2 against question 3 is the paper's thesis observed on the annotations themselves: an
edit that leaves Dice essentially at 1 changes a count trait several-fold.
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
from collections import defaultdict  # noqa: E402

import numpy as np  # noqa: E402

from prmi_branching import junction_nodes, prune  # noqa: E402
from prmi_clean import PINHOLE_PX, enclosed_holes, fill_pinholes  # noqa: E402
from prmi_load import _decode  # noqa: E402
from traits import dice, skeleton  # noqa: E402


def load_mask(root, rel):
    a = _decode(open(os.path.join(root, rel.replace("\\", os.sep).replace("/", os.sep)),
                     "rb").read())
    return a > a.min()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--census-rows", default="results/prmi_mask_census_rows.json")
    ap.add_argument("--n-junction-sample", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/prmi_pinholes.json")
    a = ap.parse_args()

    rows = [r for r in json.load(open(a.census_rows)) if r["frac"] > 0]
    by_cfg = defaultdict(list)
    for r in rows:
        by_cfg[r["config"]].append(r)
    print(f"{len(rows)} non-empty masks across {len(by_cfg)} configurations")

    out = {"pinhole_px": PINHOLE_PX, "n_nonempty_masks": len(rows), "configs": {}}
    t0 = time.time()
    all_any, all_n = [], []
    for cfg, rs in sorted(by_cfg.items()):
        n_small, any_small, dice_fill = [], [], []
        for r in rs:
            m = load_mask(a.dir, r["path"])
            h = enclosed_holes(m)
            k = sum(1 for s in h if s <= PINHOLE_PX)
            n_small.append(k)
            any_small.append(k > 0)
            if k:
                dice_fill.append(dice(m, fill_pinholes(m)))
        n_small, any_small = np.array(n_small), np.array(any_small)
        all_any += any_small.tolist()
        all_n += n_small.tolist()
        out["configs"][cfg] = {
            "n_masks": len(rs),
            "share_with_pinholes": float(any_small.mean()),
            "pinholes_per_mask_median": float(np.median(n_small)),
            "pinholes_per_mask_mean": float(n_small.mean()),
            "dice_to_filled_min": float(min(dice_fill)) if dice_fill else None,
            "dice_to_filled_median": float(np.median(dice_fill)) if dice_fill else None,
        }
        print(f"  {cfg:28} n={len(rs):6}  with pinholes {any_small.mean():6.1%}  "
              f"median {np.median(n_small):4.0f}  Dice to filled >= "
              f"{(min(dice_fill) if dice_fill else 1):.4f}  ({(time.time() - t0) / 60:.1f} min)",
              flush=True)
    out["overall"] = {"share_with_pinholes": float(np.mean(all_any)),
                      "pinholes_per_mask_median": float(np.median(all_n)),
                      "pinholes_per_mask_mean": float(np.mean(all_n))}

    rng = np.random.default_rng(a.seed)
    out["junctions"] = {}
    for cfg in ("peanut_640x480_DPI120", "peanut_736x552_DPI150"):
        rs = by_cfg[cfg]
        pick = rng.choice(len(rs), size=min(a.n_junction_sample, len(rs)), replace=False)
        before, after = [], []
        for i in pick:
            m = load_mask(a.dir, rs[int(i)]["path"])
            before.append(junction_nodes(prune(skeleton(m), 15)))
            after.append(junction_nodes(prune(skeleton(fill_pinholes(m)), 15)))
        before, after = np.array(before), np.array(after)
        out["junctions"][cfg] = {
            "n_sample": len(pick),
            "median_before": float(np.median(before)), "median_after": float(np.median(after)),
            "mean_before": float(before.mean()), "mean_after": float(after.mean()),
        }
        print(f"  junctions, native, {cfg}: median {np.median(before):.0f} -> {np.median(after):.0f}"
              f"  mean {before.mean():.1f} -> {after.mean():.1f}", flush=True)

    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print("wrote " + a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
