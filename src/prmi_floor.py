"""Can PRMI's test set tell these architectures apart? The PAIRED, architecture-level floor.

WHY THIS FILE EXISTS: THE FIRST VERSION ANSWERED THE WRONG QUESTION
------------------------------------------------------------------
`prmi_analysis.resolution_floor_report` passed all 18 (architecture x seed) units to
`groupeval.decompose_variance` and compared the gaps between adjacent models to
`groupeval.resolution_floor`. That function is documented, correctly, as the half-width **on a single
mean**: `1.96 * sqrt(var_tube / n_tubes)`. It includes the between-tube variance -- some tubes are
simply easier -- and that term CANCELS when two models are compared on the same tubes. Comparing a
difference to a single-mean floor overstates how unresolvable the models are, which is the direction
that flatters the finding. It reported 0.050 and called it "paired".

Two further problems with the first version, both about the unit of comparison:
  * it treated the 18 seed-units as 18 models, so "16 of 17 adjacent gaps below the floor" mostly
    measured seeds of the same architecture against each other, which nobody claims differ;
  * it never used the seeds as replicates, which is the one thing they are for.

THE METHOD, PORTED FROM PROJECT 02's `variance_floor.py`
-------------------------------------------------------
For each pair of architectures (x, y), form the per-tube DIFFERENCE for each seed,

    diff[s][tube] = dice_x[s][tube] - dice_y[s][tube]

and decompose THAT with `groupeval.decompose_variance`. The tube term of the difference is the
architecture x tube interaction -- x is better on some tubes and y on others -- and it is the part of
a paired comparison that does not cancel and does not shrink with more seeds. Its
`resolution_floor` is therefore the paired floor: the narrowest interval on the difference that any
amount of retraining could reach on this fixed test set.

Seeds are paired by index. Nothing is shared between seed i of x and seed i of y except the test set,
so the pairing is arbitrary; it only routes seed noise into the run and residual terms, which vanish
as seeds go to infinity, leaving the interaction term as the floor.

Scores are per-tube means over NON-EMPTY frames. Empty frames carry the 0.45 free floor (F15) and
would compress every difference toward zero.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# isort: off
import resources  # noqa: F401,E402  MUST load before numpy: caps BLAS threads
# isort: on

import argparse  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
from collections import defaultdict  # noqa: E402

import groupeval  # noqa: E402
import numpy as np  # noqa: E402

from prmi_analysis import load_models, load_test  # noqa: E402

PROJECT_SEEDS = (3, 10, 1000)


def tube_scores(models, meta):
    """{arch: {seed: {tube: mean Dice over non-empty frames}}}."""
    nonempty = np.array([not m["gt_empty_resized"] for m in meta])
    groups = np.array([m["group"] for m in meta])
    out = defaultdict(dict)
    for m in models:
        d = np.load(m["dice_path"])
        per = defaultdict(list)
        for i in np.flatnonzero(nonempty):
            per[groups[i]].append(d[i])
        out[m["arch"]][int(m["seed"])] = {g: float(np.mean(v)) for g, v in per.items()}
    return dict(out)


def paired(scores, x, y):
    seeds = sorted(set(scores[x]) & set(scores[y]))
    tubes = sorted(set.intersection(*[set(scores[a][s]) for a in (x, y) for s in seeds]))
    diff = {s: {t: scores[x][s][t] - scores[y][s][t] for t in tubes} for s in seeds}
    rep = groupeval.decompose_variance(diff)
    mean_diff = float(np.mean([v for s in diff.values() for v in s.values()]))
    return {
        "x": x, "y": y, "n_seeds": len(seeds), "n_tubes": len(tubes),
        "mean_diff": mean_diff,
        "halfwidth": {str(n): float(1.96 * rep.se(n)) for n in PROJECT_SEEDS},
        "floor": float(groupeval.resolution_floor(rep)),
        # `decompose_variance` clips a negative variance-component estimate to zero. With 3 seeds
        # and 6-7 tubes that happens, and a floor of exactly 0 then reads as "resolvable with
        # enough seeds" for ANY gap. It is an estimation failure, not a property of the benchmark,
        # so it is flagged and kept out of the "resolvable ever" verdicts.
        "floor_clipped": bool(rep.var_group == 0.0),
        "var_run": rep.var_run, "var_tube": rep.var_group, "var_resid": rep.var_resid,
    }


def analyse(scores):
    archs = sorted(scores)
    arch_mean = {a: float(np.mean([np.mean(list(scores[a][s].values())) for s in scores[a]]))
                 for a in archs}
    ranking = sorted(archs, key=lambda a: -arch_mean[a])
    pairs = {f"{x}|{y}": paired(scores, x, y) for x, y in itertools.combinations(archs, 2)}

    def get(x, y):
        return pairs.get(f"{x}|{y}") or pairs[f"{y}|{x}"]

    adjacent = []
    for x, y in zip(ranking, ranking[1:]):
        p = get(x, y)
        gap = arch_mean[x] - arch_mean[y]
        adjacent.append({
            "upper": x, "lower": y, "gap": gap,
            "floor": p["floor"],
            "floor_clipped": p["floor_clipped"],
            "halfwidth_now": p["halfwidth"]["3"],
            # None, not True, when the floor estimate collapsed: the verdict is unknown.
            "resolvable_ever": None if p["floor_clipped"] else bool(abs(gap) > p["floor"]),
            "resolved_now": bool(abs(gap) > p["halfwidth"]["3"]),
        })
    all_pairs = []
    for p in pairs.values():
        all_pairs.append({**{k: p[k] for k in ("x", "y", "mean_diff", "floor", "floor_clipped")},
                          "resolvable_ever": (None if p["floor_clipped"]
                                              else bool(abs(p["mean_diff"]) > p["floor"])),
                          "resolved_now": bool(abs(p["mean_diff"]) > p["halfwidth"]["3"])})

    # The single-mean floor, kept for context and LABELLED as what it is.
    flat = {f"{a}|s{s}": scores[a][s] for a in archs for s in scores[a]}
    single = float(groupeval.resolution_floor(groupeval.decompose_variance(flat)))

    floors = [p["floor"] for p in pairs.values() if not p["floor_clipped"]]
    return {
        "n_architectures": len(archs),
        "n_tubes": len(next(iter(next(iter(scores.values())).values()))),
        "arch_mean_dice_nonempty": arch_mean,
        "ranking": ranking,
        "adjacent": adjacent,
        "n_adjacent_resolvable_ever": sum(a["resolvable_ever"] is True for a in adjacent),
        "n_adjacent_never_resolvable": sum(a["resolvable_ever"] is False for a in adjacent),
        "n_adjacent_floor_clipped": sum(a["floor_clipped"] for a in adjacent),
        "n_adjacent_resolved_now": sum(a["resolved_now"] for a in adjacent),
        "pairs": all_pairs,
        "n_pairs": len(all_pairs),
        "n_pairs_resolvable_ever": sum(p["resolvable_ever"] is True for p in all_pairs),
        "n_pairs_floor_clipped": sum(p["floor_clipped"] for p in all_pairs),
        "n_pairs_resolved_now": sum(p["resolved_now"] for p in all_pairs),
        "median_paired_floor": float(np.median(floors)) if floors else None,
        "median_adjacent_paired_floor": (float(np.median([a["floor"] for a in adjacent
                                                          if not a["floor_clipped"]]))
                                         if any(not a["floor_clipped"] for a in adjacent)
                                         else None),
        "median_adjacent_gap": float(np.median([abs(a["gap"]) for a in adjacent])),
        "single_mean_floor_NOT_for_comparisons": single,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="peanut_640x480_DPI120")
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--cache-dir", default="data/cache")
    ap.add_argument("--models-dir", default="results/prmi_models")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out_path = a.out or f"results/prmi_floor_{a.config}.json"

    models = load_models(a.models_dir, a.config)
    _gt, meta = load_test(a.cache_dir, a.config, a.size)
    res = analyse(tube_scores(models, meta))
    res["config"] = a.config
    with open(out_path, "w") as fh:
        json.dump(res, fh, indent=2)

    print(f"{res['n_architectures']} architectures x {len(models) // res['n_architectures']} "
          f"seeds, {res['n_tubes']} tubes ({a.config})\n")
    print(f"{'architecture':16} {'mean Dice (roots)':>18}")
    for arch in res["ranking"]:
        print(f"{arch:16} {res['arch_mean_dice_nonempty'][arch]:>18.4f}")
    print(f"\n{'adjacent pair':34} {'gap':>8} {'paired floor':>13} {'now (3 seeds)':>14}  verdict")
    for r in res["adjacent"]:
        v = ("resolved" if r["resolved_now"] else
             "unknown (floor estimate clipped to 0)" if r["resolvable_ever"] is None else
             "resolvable with more seeds" if r["resolvable_ever"] else "NEVER resolvable")
        print(f"{r['upper'] + ' > ' + r['lower']:34} {r['gap']:+8.4f} {r['floor']:13.4f} "
              f"{r['halfwidth_now']:14.4f}  {v}")
    print(f"\nadjacent pairs: resolved now {res['n_adjacent_resolved_now']}, resolvable ever "
          f"{res['n_adjacent_resolvable_ever']}, never {res['n_adjacent_never_resolvable']}, "
          f"unknown {res['n_adjacent_floor_clipped']} (of {len(res['adjacent'])})")
    print(f"all {res['n_pairs']} pairs: resolved now {res['n_pairs_resolved_now']}, resolvable "
          f"ever {res['n_pairs_resolvable_ever']}, floor clipped {res['n_pairs_floor_clipped']}")
    mp = res["median_paired_floor"]
    print(f"median paired floor {f'{mp:.4f}' if mp is not None else 'n/a'} | single-mean floor "
          f"{res['single_mean_floor_NOT_for_comparisons']:.4f} (not a comparison floor)")
    print("wrote " + out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
