"""The calibrated degradations applied to REAL PRMI annotations, at native resolution.

Answers two limitations at once. The synthetic sweep (F7-F13) used generated architectures, and the
trained-model arm resized frames to 320 px. Here every degradation is calibrated to an exact Dice
loss on real ground-truth masks, at the resolution they were annotated, across every imaging
configuration and all six species, and the twelve traits are read off as in the synthetic sweep.

Masks: a seeded random sample of test-split annotations per configuration with at least
`--min-fg` root pixels, so that a 1% Dice loss is not a handful of pixels. Pinholes are filled first
(`prmi_clean`, C12), as for every other real-data trait in the paper.

Writes `results/prmi_native_calibrated.csv` (one row per mask x degradation x dose) and
`results/prmi_native_calibrated_summary.json`.
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
import pandas as pd  # noqa: E402

from calibrate import SEVERITY_BOUNDS, solve_severity  # noqa: E402
from degrade import DEGRADATIONS  # noqa: E402
from dose_response import rel_error, stable_seed  # noqa: E402
from prmi_clean import fill_pinholes  # noqa: E402
from prmi_pinholes import load_mask  # noqa: E402
from traits import TRAIT_KIND, TRAITS, dice  # noqa: E402

DOSES = (0.01, 0.05)


def sample(rows, n_per_config, min_fg, seed):
    by = defaultdict(list)
    for r in rows:
        if r["official_split"] == "test" and r["n_fg"] >= min_fg:
            by[r["config"]].append(r)
    rng = np.random.default_rng(seed)
    out = []
    for cfg in sorted(by):
        rs = sorted(by[cfg], key=lambda r: r["path"])
        pick = rng.choice(len(rs), size=min(n_per_config, len(rs)), replace=False)
        out += [rs[int(i)] for i in sorted(pick)]
    return out


def summarise(df):
    ok = df[df.reachable]
    out = {"n_masks": int(df["mask"].nunique()), "n_configs": int(df.config.nunique()),
           "n_species": int(df.species.nunique()), "reachable_share": float(df.reachable.mean()),
           "doses": {}}
    for dose in DOSES:
        at = ok[ok.target_dice_loss == dose]
        med = {t: at.groupby("degradation")["err_" + t].median() for t in TRAITS}
        d = {"achieved_loss_mean": float((1 - at.dice).mean()),
             "traits": {t: {"kind": TRAIT_KIND[t],
                            "by_degradation": {k: float(v) for k, v in m.items()},
                            "min": float(m.min()), "max": float(m.max()),
                            "argmax": str(m.idxmax())} for t, m in med.items()}}
        # Does the kind ordering hold in every configuration? For each configuration, the largest
        # median error over integral traits against the largest over count traits.
        per_cfg = {}
        for cfg, sub in at.groupby("config"):
            mx = {t: sub.groupby("degradation")["err_" + t].median().max() for t in TRAITS}
            integ = max(v for t, v in mx.items() if TRAIT_KIND[t] == "integral")
            count = max(v for t, v in mx.items() if TRAIT_KIND[t] == "count")
            per_cfg[cfg] = {"integral_max": float(integ), "count_max": float(count),
                            "pixel_area_span": [float(sub.groupby("degradation").err_pixel_area
                                                      .median().min()),
                                                float(sub.groupby("degradation").err_pixel_area
                                                      .median().max())]}
        d["by_config"] = per_cfg
        d["count_exceeds_integral_in_every_config"] = bool(
            all(v["count_max"] > v["integral_max"] for v in per_cfg.values()))
        # F11 on real masks: dilation against junction breaks on branch-point error, paired.
        w = at.pivot_table(index="mask", columns="degradation", values="err_n_branch_points")
        if {"boundary_dilate", "junction_break"} <= set(w.columns):
            diff = (w["boundary_dilate"] - w["junction_break"]).dropna()
            d["dilate_vs_junction_branch_points"] = {
                "n": int(len(diff)), "dilate_larger_in": int((diff > 0).sum()),
                "median_difference": float(diff.median()) if len(diff) else None}
        out["doses"][str(dose)] = d
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/prmi/doi_10_5061_dryad_2v6wwpzp4__v20220204/PRMI_official")
    ap.add_argument("--census-rows", default="results/prmi_mask_census_rows.json")
    ap.add_argument("--n-per-config", type=int, default=6)
    ap.add_argument("--min-fg", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="first N masks only (timing)")
    ap.add_argument("--out", default="results/prmi_native_calibrated")
    a = ap.parse_args()

    picks = sample(json.load(open(a.census_rows)), a.n_per_config, a.min_fg, a.seed)
    if a.limit:
        picks = picks[: a.limit]
    print(f"{len(picks)} masks across {len({p['config'] for p in picks})} configurations")
    rows, t0 = [], time.time()
    for mi, r in enumerate(picks):
        gt = fill_pinholes(load_mask(a.dir, r["path"]))
        base = {k: fn(gt) for k, fn in TRAITS.items()}
        key = r["path"].replace("\\", "/")
        for dname, (fn, _grid) in DEGRADATIONS.items():
            lo, hi = SEVERITY_BOUNDS[dname]
            for dose in DOSES:
                rseed = stable_seed(0, key, dname, dose) % (2 ** 31)
                sev, _ = solve_severity(fn, gt, dose, lo, hi, rng_seed=rseed)
                row = dict(mask=key, config=r["config"], species=r["species"], n_fg=r["n_fg"],
                           degradation=dname, target_dice_loss=float(dose), reachable=False)
                if np.isfinite(sev):
                    dm = fn(gt, sev, np.random.default_rng(rseed))
                    row.update(reachable=True, severity=float(sev), dice=float(dice(gt, dm)))
                    for k, tfn in TRAITS.items():
                        row["err_" + k] = rel_error(base[k], tfn(dm))
                        row["base_" + k] = base[k]
                rows.append(row)
        print(f"  [{mi + 1}/{len(picks)}] {r['config']:28} n_fg={r['n_fg']:6}  "
              f"({(time.time() - t0) / 60:.1f} min)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(a.out + ".csv", index=False)
    s = summarise(df)
    with open(a.out + "_summary.json", "w") as fh:
        json.dump(s, fh, indent=2)
    print(json.dumps({k: v for k, v in s.items() if k != "doses"}))
    for dose, d in s["doses"].items():
        print(f"dose {dose}: count>integral everywhere {d['count_exceeds_integral_in_every_config']}"
              f"  pixel_area {d['traits']['pixel_area']['min']:.4f}-{d['traits']['pixel_area']['max']:.4f}"
              f"  n_components {d['traits']['n_components']['min']:.3f}-{d['traits']['n_components']['max']:.3f}"
              f"  {d.get('dilate_vs_junction_branch_points')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
