"""The synthetic results the manuscript quotes, derived from the sweep CSVs into one JSON.

F11-F13 were first computed in throwaway snippets against `results/calibrated_sweep.csv`. Standing
rule 4 needs every quoted number to live in `results/` where a test can read it, so this recomputes
them reproducibly. It trains and simulates nothing; it only summarises existing sweeps.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# isort: off
import resources  # noqa: F401,E402  MUST load before numpy: caps BLAS threads
# isort: on

import json  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from traits import TRAIT_KIND, TRAITS  # noqa: E402

R = "results"


def main():
    cal = pd.read_csv(os.path.join(R, "calibrated_sweep.csv"))
    ok = cal[cal.reachable]
    out = {"n_masks": int(cal["mask"].nunique()), "n_cells": int(len(cal)),
           "reachable_share": float(cal.reachable.mean())}

    # Calibration accuracy at the two doses the paper reads.
    for dose in (0.01, 0.02):
        s = ok[ok.target_dice_loss == dose]
        out[f"achieved_loss_at_{dose}"] = {"mean": float((1 - s.dice).mean()),
                                           "sd": float((1 - s.dice).std()), "n": int(len(s))}

    # F12: median relative error per (trait, degradation) at Dice loss 0.01, and the spread.
    at = ok[ok.target_dice_loss == 0.01]
    f12 = {}
    for t in TRAITS:
        med = at.groupby("degradation")["err_" + t].median()
        f12[t] = {"kind": TRAIT_KIND[t], "by_degradation": {k: float(v) for k, v in med.items()},
                  "min": float(med.min()), "max": float(med.max()),
                  "argmax": str(med.idxmax())}
    out["f12_dice_099"] = f12
    out["f12_n_components_baseline_median"] = float(at["base_n_components"].median())

    # F11: dilation against junction_break at Dice loss 0.02, paired per mask.
    a02 = ok[ok.target_dice_loss == 0.02]
    w = a02.pivot_table(index="mask", columns="degradation", values="err_n_branch_points")
    d = (w["boundary_dilate"] - w["junction_break"]).dropna()
    rng = np.random.default_rng(0)
    bs = [np.median(rng.choice(d.values, len(d), replace=True)) for _ in range(5000)]
    winner = a02.groupby(["mask", "degradation"]).err_n_branch_points.median().unstack()
    out["f11"] = {
        "n_masks": int(len(d)),
        "dilate_beats_junction_in": int((d > 0).sum()),
        "median_difference": float(d.median()),
        "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
        "dilate_is_argmax_in": int((winner.idxmax(axis=1) == "boundary_dilate").sum()),
        "winner_by_dose": json.load(open(os.path.join(R, "calibrated_sweep_summary.json")))
        ["winner_by_dose"],
    }

    # F9/F10: kind ordering by architecture, rank agreement, the confound in dex.
    bym = pd.read_csv(os.path.join(R, "morphology_sweep_by_morphology.csv"))
    kind = bym.groupby(["morphology", "kind"]).sensitivity.median().unstack()
    ms = json.load(open(os.path.join(R, "morphology_sweep_summary.json")))
    rhos = [p["rho"] for p in ms["fragility_rank_agreement_between_architectures"]]
    out["f9"] = {
        "kind_median_sensitivity": {m: {k: float(v) for k, v in r.items()}
                                    for m, r in kind.iterrows()},
        "integral_lowest_everywhere": bool(ms["kind_ordering_holds_everywhere"]),
        "trait_rank_agreement": {"min": float(min(rhos)), "median": float(np.median(rhos)),
                                 "max": float(max(rhos))},
    }
    out["f10"] = {"median_morphology_dex": ms["median_morphology_range_dex"],
                  "median_thickness_dex": ms["median_thickness_range_dex"]}

    # How different the four architectures are: branch points per 1,000 skeleton pixels, at the
    # middle thickness, seed 0 (the configuration the morphology test checks).
    from morphology import MORPHOLOGIES, THICKNESS_LEVELS, generate
    from traits import n_branch_points, total_length
    dens = {}
    for m in sorted(MORPHOLOGIES):
        g = generate(m, size=384, seed=0, thickness=THICKNESS_LEVELS["medium"])
        dens[m] = 1000.0 * n_branch_points(g) / total_length(g)
    out["architecture_branch_density"] = dens
    out["architecture_branch_density_ratio"] = float(max(dens.values()) / min(dens.values()))
    morph = pd.read_csv(os.path.join(R, "morphology_sweep.csv"))
    out["morphology_sweep_n_masks"] = int(morph["mask"].nunique())
    out["calibrated_sweep_n_masks"] = int(cal["mask"].nunique())
    integral = [t for t in TRAITS if TRAIT_KIND[t] == "integral"]
    out["f12_integral_max"] = {"value": float(max(f12[t]["max"] for t in integral)),
                               "trait": max(integral, key=lambda t: f12[t]["max"])}

    # What the hand-picked severity grids cost in Dice before calibration. An earlier version of
    # the ladder gave 0.016-0.524, before thin_dropout was rewritten; these are the current grids.
    dr = pd.read_csv(os.path.join(R, "dose_response.csv"))
    grid = (1 - dr.dice).groupby(dr.degradation).mean()
    out["f7_uncalibrated_mean_loss"] = {k: float(v) for k, v in grid.items()}

    # F11's mechanism: dilation fuses neighbouring branches. One fibrous mask, medium thickness,
    # seed 0, at a Dice loss of 0.02. (The earlier ladder gave 557.)
    fus = a02[(a02.degradation == "boundary_dilate") & (a02["mask"] == "fibrous|medium|0")].iloc[0]
    out["f11_fusion_example"] = {
        "mask": "fibrous|medium|0", "dice": float(fus.dice),
        "branch_points_before": int(fus.base_n_branch_points),
        "branch_points_after": int(round(fus.base_n_branch_points * (1 + fus.err_n_branch_points)))}

    # F1: fractal estimator against analytic truth at the finest resolution.
    fv = json.load(open(os.path.join(R, "fractal_validation.json")))["dimension"]
    finest = {s: max(v["runs"], key=lambda r: r["size"]) for s, v in fv.items()}
    out["f1_max_abs_error_finest"] = float(max(r["abs_error"] for r in finest.values()))
    out["f1_n_structures"] = len(finest)

    with open(os.path.join(R, "synthetic_summary.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps({k: v for k, v in out.items() if k != "f12_dice_099"}, indent=1)[:2500])
    for t in ("pixel_area", "n_components", "convex_hull_area", "n_branch_points"):
        print(t, {k: round(v, 4) for k, v in f12[t].items() if k in ("min", "max")},
              f12[t]["argmax"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
