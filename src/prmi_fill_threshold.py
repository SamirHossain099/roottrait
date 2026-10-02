"""Does the pinhole-filling threshold matter? Junction counts of the annotation at several thresholds.

`prmi_clean.PINHOLE_PX = 10` was chosen by looking at the hole-size histogram, not tuned. The
manuscript's limitations say the junction count does not depend on it; this is the measurement
behind that sentence, written to `results/prmi_fill_threshold.json` so the claim is pinned.

A random sample of non-empty test annotations per configuration, from the 320 px cache the trait
analysis uses, skeleton pruned to 15 px as everywhere else.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# isort: off
import resources  # noqa: F401,E402  MUST load before numpy: caps BLAS threads
# isort: on

import argparse  # noqa: E402
import json  # noqa: E402

import numpy as np  # noqa: E402

from prmi_analysis import load_test  # noqa: E402
from prmi_branching import junction_nodes, prune  # noqa: E402
from prmi_clean import fill_pinholes  # noqa: E402
from traits import skeleton  # noqa: E402

THRESHOLDS = (0, 5, 10, 30, 100, 300)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+",
                    default=["peanut_640x480_DPI120", "peanut_736x552_DPI150"])
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/prmi_fill_threshold.json")
    a = ap.parse_args()
    out = {"thresholds": list(THRESHOLDS), "n_sample": a.n, "configs": {}}
    for cfg in a.configs:
        gt, _meta = load_test("data/cache", cfg, 320)
        ne = np.flatnonzero(gt.reshape(len(gt), -1).any(1))
        pick = np.random.default_rng(a.seed).choice(ne, size=min(a.n, len(ne)), replace=False)
        med = {}
        for t in THRESHOLDS:
            j = [junction_nodes(prune(skeleton(fill_pinholes(gt[i], t) if t else gt[i]), 15))
                 for i in pick]
            med[str(t)] = float(np.median(j))
        filled = [med[str(t)] for t in THRESHOLDS if t >= 10]
        out["configs"][cfg] = {"median_junctions": med,
                               "constant_from_10_to_300": bool(len(set(filled)) == 1)}
        print(cfg, med)
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print("wrote " + a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
