"""What replicates between PRMI's two peanut imaging configurations, and what does not.

Peanut is the one species PRMI images at two configurations (640x480 DPI120 and 736x552 DPI150,
different operators). Training the same six architectures x three seeds on each turns the second
into a replication of the first rather than a different experiment (F16).

This file only COMPARES results already in `results/`; it trains and measures nothing. Everything it
reports is pinned by `tests/test_prmi_replication.py`.
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

from dose_response import spearman  # noqa: E402
from prmi_decompose import KINDS  # noqa: E402

CONFIGS = ("peanut_640x480_DPI120", "peanut_736x552_DPI150")


def _load(kind, config, d="results"):
    with open(os.path.join(d, f"prmi_{kind}_{config}.json")) as fh:
        return json.load(fh)


def per_config(config, d="results"):
    a, f, b = _load("analysis", config, d), _load("floor", config, d), _load("branching", config, d)
    ms = a["models"]
    dall = np.array([m["mean_dice_all_frames"] for m in ms])
    droot = np.array([m["mean_dice_nonempty_only"] for m in ms])
    return {
        "n_models": len(ms),
        "n_test_tubes": a["n_test_tubes"],
        "null_dice_all_frames": a["null_dice_all_frames"],
        "dice_all_frames": {"min": float(dall.min()), "max": float(dall.max())},
        "dice_roots_only": {"min": float(droot.min()), "max": float(droot.max())},
        "inflation_all_minus_roots": {"min": float((dall - droot).min()),
                                      "max": float((dall - droot).max())},
        "worst_model_margin_over_null": float(dall.min() - a["null_dice_all_frames"]),
        "decomposition_mean": {k: float(np.mean([m["decomp_nonempty"][k] for m in ms]))
                               for k in KINDS},
        # Per-architecture means (over seeds), so "the same for every architecture" is a range the
        # paper can quote, not an impression from the configuration mean.
        "decomposition_by_arch": _by_arch(ms),
        "rho": {t: v["spearman_dice_vs_trait_error"] for t, v in a["rank_inversion"].items()},
        "rho_without_two_weakest": {t: v["spearman_without_two_weakest"]
                                    for t, v in a["rank_inversion"].items()},
        "arch_mean_dice": f["arch_mean_dice_nonempty"],
        "ranking": f["ranking"],
        "adjacent_resolved_now": f["n_adjacent_resolved_now"],
        "adjacent_never": f["n_adjacent_never_resolvable"],
        "adjacent_unknown": f["n_adjacent_floor_clipped"],
        "median_paired_floor": f["median_paired_floor"],
        "branching": b["summary"],
        "nearest_mode_families": sorted({fp["nearest_family"]
                                         for fp in a["fingerprints"].values()}),
    }


def _by_arch(ms):
    archs = sorted({m["arch"] for m in ms})
    out = {a: {k: float(np.mean([m["decomp_nonempty"][k] for m in ms if m["arch"] == a]))
               for k in KINDS} for a in archs}
    ol = [v["over_inclusive"] + v["lost_structure"] for v in out.values()]
    fp = [v["false_positive"] for v in out.values()]
    return {"means": out, "over_plus_lost": {"min": min(ol), "max": max(ol)},
            "false_positive": {"min": min(fp), "max": max(fp)}}


def compare(c1, c2):
    archs = sorted(c1["arch_mean_dice"])
    rank_rho = spearman(np.array([c1["arch_mean_dice"][x] for x in archs]),
                        np.array([c2["arch_mean_dice"][x] for x in archs]))
    shared = sorted(set(c1["rho"]) & set(c2["rho"]))
    trait_rho = spearman(np.array([c1["rho"][t] for t in shared]),
                         np.array([c2["rho"][t] for t in shared]))
    return {
        "architecture_rank_spearman": rank_rho,
        "same_best_architecture": c1["ranking"][0] == c2["ranking"][0],
        "same_worst_architecture": c1["ranking"][-1] == c2["ranking"][-1],
        "rank_moves": {x: c2["ranking"].index(x) - c1["ranking"].index(x) for x in archs},
        "trait_rho_profile_spearman": trait_rho,
        "decomposition_max_abs_difference": max(abs(c1["decomposition_mean"][k]
                                                     - c2["decomposition_mean"][k])
                                                 for k in KINDS),
        "over_plus_lost_share": {
            "c1": c1["decomposition_mean"]["over_inclusive"] + c1["decomposition_mean"]["lost_structure"],
            "c2": c2["decomposition_mean"]["over_inclusive"] + c2["decomposition_mean"]["lost_structure"],
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/prmi_replication.json")
    a = ap.parse_args()
    c1, c2 = (per_config(c, a.results) for c in CONFIGS)
    out = {"configs": list(CONFIGS), "c1": c1, "c2": c2, "comparison": compare(c1, c2)}
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out["comparison"], indent=2))
    print("wrote " + a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
