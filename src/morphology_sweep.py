"""Does the fragility ordering survive a change of root architecture?

The open gap this closes: every number came from one synthetic morphology, and
projects 01 and 04 both had confident single-source claims collapse on replication. This crosses
four architectures with three thickness levels and asks three questions the single-shape sweep
could not:

  Q1  Does the corrected taxonomy (integrals robust; extrema and counts fragile) hold in EVERY
      architecture, or was it a property of the one shape it was found on?
  Q2  Is fragility driven by architecture or by THICKNESS? `junction_break` punches a fixed 3 px
      disk, so a thin root loses proportionally more of itself at every junction. Architecture and
      thickness are confounded unless crossed, and the confound runs in the direction that would
      manufacture an architecture effect.
  Q3  Does `junction_break` stay the most deceptive degradation everywhere?

Q2 is the one that matters most. Project 04 reported "the dominant analytic choice changes with the
population" on two cohorts that differed 14x in record length, and the claim only survived once the
lengths were matched. This is the same shape of error, pre-empted rather than corrected.
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
import pandas as pd  # noqa: E402

from dose_response import spearman, sweep_masks  # noqa: E402
from morphology import MORPHOLOGIES, THICKNESS_LEVELS, generate  # noqa: E402
from traits import TRAIT_KIND, TRAITS  # noqa: E402

KIND_ORDER = ["integral", "fitted", "extremum", "count"]


def build_masks(n_seeds=3, size=384, seed0=0):
    """Every (architecture, thickness, seed) cell. Fully crossed, so margins are interpretable."""
    for morph in MORPHOLOGIES:
        for tname, tval in THICKNESS_LEVELS.items():
            for s in range(n_seeds):
                meta = dict(mask=f"{morph}|{tname}|{s}", morphology=morph,
                            thickness=tname, thickness_px=tval, seed=s)
                yield meta, generate(morph, size=size, seed=seed0 + s, thickness=tval)


def sensitivity(sub, trait):
    """Mean |relative trait error| per unit of Dice lost, over genuinely degraded rows."""
    e = sub["err_" + trait]
    loss = 1.0 - sub.dice
    ok = np.isfinite(e) & (loss > 1e-9)
    return float((e[ok] / loss[ok]).mean()) if ok.any() else np.nan


def per_group_table(df, by):
    """Trait sensitivity within each level of `by` -- the replication check."""
    rows = []
    for key, sub in df.groupby(by):
        for t in TRAITS:
            rows.append({by: key, "trait": t, "kind": TRAIT_KIND[t],
                         "sensitivity": sensitivity(sub, t),
                         "spearman": spearman(sub.dice, sub["err_" + t])})
    return pd.DataFrame(rows)


def fmt2(v):
    return f"{v:.2f}"


def fmt3(v):
    return f"{v:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--out", default="results/morphology_sweep.csv")
    a = ap.parse_args()

    masks = list(build_masks(n_seeds=a.n_seeds, size=a.size))
    print(f"{len(MORPHOLOGIES)} architectures x {len(THICKNESS_LEVELS)} thicknesses x {a.n_seeds} seeds = {len(masks)} masks")
    df = sweep_masks(masks, seed=0, verbose=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"\nwrote {a.out}  ({len(df)} rows)\n")

    # ------------------------------------------------------------- Q1: does the taxonomy hold?
    by_morph = per_group_table(df, "morphology")
    print("=== Q1. Trait sensitivity by architecture (higher = Dice is a worse proxy) ===")
    piv = by_morph.pivot(index="trait", columns="morphology", values="sensitivity")
    morphs = sorted(MORPHOLOGIES)
    piv = piv[morphs].sort_values(morphs[0], ascending=False)
    piv.insert(0, "kind", [TRAIT_KIND[t] for t in piv.index])
    print(piv.to_string(float_format=fmt2))

    print("\n=== Q1b. Rank agreement of the fragility ordering between architectures ===")
    agree = []
    for i, m1 in enumerate(morphs):
        for m2 in morphs[i + 1:]:
            r = spearman(piv[m1].values, piv[m2].values)
            agree.append({"pair": f"{m1} vs {m2}", "rho": r})
            print(f"  {m1:12s} vs {m2:12s}  rho = {r:+.3f}")
    rhos = [x["rho"] for x in agree]
    print(f"  --> min {min(rhos):+.3f}, median {float(np.median(rhos)):+.3f}")

    print("\n=== Q1c. Does the KIND ordering hold in every architecture? ===")
    kind_tab = by_morph.groupby(["morphology", "kind"]).sensitivity.median().unstack()
    kind_tab = kind_tab[[k for k in KIND_ORDER if k in kind_tab.columns]]
    print(kind_tab.to_string(float_format=fmt2))
    holds = bool((kind_tab["integral"] < kind_tab["extremum"]).all()
                 and (kind_tab["integral"] < kind_tab["count"]).all())
    print(f"  integrals below BOTH extrema and counts in every architecture: {holds}")

    # -------------------------------------------------- Q2: architecture or thickness?
    print("\n=== Q2. The confound: architecture margin vs thickness margin ===")
    by_thick = per_group_table(df, "thickness")
    tt = by_thick.pivot(index="trait", columns="thickness", values="sensitivity")
    tt = tt[[k for k in THICKNESS_LEVELS if k in tt.columns]]
    tt.insert(0, "kind", [TRAIT_KIND[t] for t in tt.index])
    print(tt.to_string(float_format=fmt2))

    # Spread of log-sensitivity attributable to each factor, per trait. Logs because sensitivity
    # spans three orders of magnitude and a raw range would be dominated by n_components alone.
    spread = []
    for t in TRAITS:
        lm = np.log10([sensitivity(sub, t) for _, sub in df.groupby("morphology")])
        lt = np.log10([sensitivity(sub, t) for _, sub in df.groupby("thickness")])
        lm, lt = lm[np.isfinite(lm)], lt[np.isfinite(lt)]
        if len(lm) and len(lt):
            spread.append({"trait": t, "kind": TRAIT_KIND[t],
                           "morphology_range_dex": float(lm.max() - lm.min()),
                           "thickness_range_dex": float(lt.max() - lt.min())})
    sp = pd.DataFrame(spread).sort_values("morphology_range_dex", ascending=False)
    print("\n  spread in log10(sensitivity) attributable to each factor, per trait:")
    print(sp.to_string(index=False, float_format=fmt3))
    print(f"\n  median across traits: morphology {sp.morphology_range_dex.median():.3f} dex, thickness {sp.thickness_range_dex.median():.3f} dex")

    # -------------------------------------------------- Q3: most deceptive degradation
    print("\n=== Q3. Most deceptive degradation, per architecture ===")
    g = (df[df.severity != 0].groupby(["morphology", "degradation"])
         .agg(mean_dice=("dice", "mean"), branches=("err_n_branch_points", "mean")))
    g["topology_per_dice_lost"] = g.branches / (1 - g.mean_dice).clip(lower=1e-9)
    worst = g.groupby("morphology").topology_per_dice_lost.idxmax()
    for m, idx in worst.items():
        print("  {:12s} -> {:16s} ({:.2f} topology error per unit Dice lost)".format(
            m, idx[1], g.loc[idx, "topology_per_dice_lost"]))
    jb_wins = all(idx[1] == "junction_break" for idx in worst.values)
    print(f"  junction_break is most deceptive in EVERY architecture: {jb_wins}")

    # -------------------------------------------------- headline replication
    print("\n=== Headline replication: median error at Dice >= 0.99, per architecture ===")
    near = df[(df.dice >= 0.99) & (df.severity != 0)]
    hl = near.groupby("morphology")[["err_" + t for t in TRAITS]].median().T
    hl.index = [i[4:] for i in hl.index]
    hl.insert(0, "kind", [TRAIT_KIND[t] for t in hl.index])
    print(hl.to_string(float_format=fmt3))
    print(f"  (n = {len(near)} genuinely degraded masks at Dice >= 0.99)")

    summary = {
        "n_rows": int(len(df)),
        "n_masks": int(len(masks)),
        "architectures": morphs,
        "thicknesses": THICKNESS_LEVELS,
        "kind_ordering_holds_everywhere": holds,
        "junction_break_most_deceptive_everywhere": bool(jb_wins),
        "fragility_rank_agreement_between_architectures": agree,
        "median_morphology_range_dex": float(sp.morphology_range_dex.median()),
        "median_thickness_range_dex": float(sp.thickness_range_dex.median()),
    }
    with open(a.out.replace(".csv", "_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    by_morph.to_csv(a.out.replace(".csv", "_by_morphology.csv"), index=False)
    by_thick.to_csv(a.out.replace(".csv", "_by_thickness.csv"), index=False)
    print("\nwrote " + a.out.replace(".csv", "_summary.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
