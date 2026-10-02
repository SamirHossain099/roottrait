"""The core result: does Dice predict trait error?

Sweep every degradation across its severity grid on a population of root masks, and at each point
record Dice alongside the relative error of every trait. Then ask the question the field has never
asked: **given a Dice value, how well can you predict the error in the thing you actually wanted?**

Reported per trait:
  sensitivity   mean |relative trait error| per unit of Dice lost. How fast the measurement
                degrades relative to the metric people report.
  spearman      rank correlation between Dice and trait error across the whole sweep. If Dice were
                a good proxy this would be near -1 for every trait.
  worst_at_99   the trait error still possible at Dice >= 0.99 -- the number that matters, because
                a model at 0.99 Dice is one everybody would accept without question.

The last of these is the paper's headline shape: not "trait error grows" but "trait error can be
large *while Dice says the segmentation is essentially perfect*".
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# isort: off
import resources  # noqa: F401,E402  MUST load before numpy: caps BLAS threads
# isort: on

import argparse  # noqa: E402
import json  # noqa: E402
import zlib  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from degrade import DEGRADATIONS, synth_root  # noqa: E402
from traits import TRAIT_FAMILY, TRAITS, dice  # noqa: E402


def stable_seed(*parts):
    """A seed derived from the arguments that is IDENTICAL across processes.

    The obvious `hash(name)` is not: Python randomises str/bytes hashing per process unless
    PYTHONHASHSEED is set, so a sweep seeded that way silently produces different masks on every
    run and cannot regenerate its own published numbers. This code did exactly that, and it hit
    the three degradations that take an rng -- junction_break, speckle and blob -- one of which
    carries the paper's central claim.

    crc32 is used rather than hash() because it is specified, stable across versions and
    platforms, and cheap. The value is not cryptographic and does not need to be.
    """
    key = "|".join(str(p) for p in parts).encode("utf-8")
    return int(zlib.crc32(key))


def rel_error(ref, got):
    """Relative error, with an absolute fallback when the reference is ~0 (e.g. n_holes)."""
    if not np.isfinite(ref) or not np.isfinite(got):
        return np.nan
    if abs(ref) < 1e-9:
        return abs(got - ref)
    return abs(got - ref) / abs(ref)


def sweep_masks(masks, seed=0, verbose=True):
    """Run the degradation ladder over an iterable of (meta, ground-truth mask) pairs.

    `meta` is a dict of identifying columns copied onto every row, so the same ladder serves the
    synthetic sweep, the morphology x thickness sweep and the real-PRMI sweep without any of them
    reimplementing the loop. `mask` in the output is whatever the caller put in meta, or the index.

    The RNG is derived from (seed, mask key, degradation, severity) via `stable_seed`, so a row is
    a pure function of its identifiers -- rows can be regenerated individually and the sweep can be
    resumed or reordered without changing a number.
    """
    rows = []
    masks = list(masks)
    for mi, (meta, gt) in enumerate(masks):
        base = {k: fn(gt) for k, fn in TRAITS.items()}
        key = meta.get("mask", mi)
        if verbose:
            desc = " ".join(f"{k}={v}" for k, v in meta.items())
            print(f"  [{mi + 1}/{len(masks)}] {desc} area={base['pixel_area']:.0f} "
                  f"tips={base['n_tips']:.0f} branches={base['n_branch_points']:.0f} "
                  f"D={base['fractal_dimension']:.3f}", flush=True)
        for dname, (fn, grid) in DEGRADATIONS.items():
            for sev in grid:
                rng = np.random.default_rng(stable_seed(seed, key, dname, sev))
                dm = fn(gt, sev, rng)
                if dm.sum() == 0:
                    continue
                row = dict(meta)
                row.update(degradation=dname, severity=float(sev), dice=dice(gt, dm))
                for k, tfn in TRAITS.items():
                    row[f"err_{k}"] = rel_error(base[k], tfn(dm))
                    row[f"base_{k}"] = base[k]
                rows.append(row)
    return pd.DataFrame(rows)


def sweep(n_masks=8, size=384, seed=0, verbose=True):
    """The original single-morphology synthetic sweep, now a thin wrapper on `sweep_masks`."""
    src = ((dict(mask=mi), synth_root(size=size, seed=seed + mi)) for mi in range(n_masks))
    df = sweep_masks(src, seed=seed, verbose=verbose)
    return df.drop(columns=[c for c in df.columns if c.startswith("base_")])


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 5:
        return np.nan
    ra = pd.Series(a[ok]).rank().values
    rb = pd.Series(b[ok]).rank().values
    if ra.std() == 0 or rb.std() == 0:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def analyse(df):
    out = []
    for k in TRAITS:
        e = df[f"err_{k}"]
        loss = 1.0 - df.dice
        ok = np.isfinite(e) & (loss > 1e-9)
        sens = float((e[ok] / loss[ok]).mean()) if ok.any() else np.nan
        # EXCLUDE the identity transform. severity == 0 gives Dice exactly 1.0 and zero error
        # by construction; including those rows dragged every "at Dice >= 0.99" median to 0.000
        # and made the table look like nothing ever goes wrong.
        near = df[(df.dice >= 0.99) & (df.severity != 0)]
        out.append(dict(
            trait=k, family=TRAIT_FAMILY[k],
            sensitivity=sens,
            spearman_dice_vs_error=spearman(df.dice, e),
            median_err=float(np.nanmedian(e)),
            worst_at_dice99=float(np.nanmax(near[f"err_{k}"])) if len(near) else np.nan,
            median_err_at_dice99=float(np.nanmedian(near[f"err_{k}"])) if len(near) else np.nan,
        ))
    return pd.DataFrame(out).sort_values("sensitivity", ascending=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-masks", type=int, default=8)
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/dose_response.csv")
    a = ap.parse_args()

    print(f"sweeping {len(DEGRADATIONS)} degradations x {a.n_masks} synthetic root masks")
    df = sweep(n_masks=a.n_masks, size=a.size, seed=a.seed)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"\nwrote {a.out}  ({len(df)} rows)")

    s = analyse(df)
    s.to_csv(a.out.replace(".csv", "_summary.csv"), index=False)

    print("\n=== trait error per unit of Dice lost (higher = Dice is a worse proxy) ===")
    print(s[["trait", "family", "sensitivity", "spearman_dice_vs_error"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print("\n=== by family ===")
    fam = s.groupby("family").sensitivity.agg(["median", "min", "max"])
    print(fam.to_string(float_format=lambda v: f"{v:.3f}"))

    near = df[(df.dice >= 0.99) & (df.severity != 0)]
    print(f"\n=== the headline: what can go wrong while Dice >= 0.99? "
          f"(n={len(near)} genuinely degraded masks) ===")
    hl = s[["trait", "family", "median_err_at_dice99", "worst_at_dice99"]].copy()
    print(hl.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print("\n=== which degradation is most deceptive? (Dice cost vs topology damage) ===")
    g = df[df.severity != 0].groupby("degradation").agg(
        mean_dice=("dice", "mean"),
        tips=("err_n_tips", "mean"), branches=("err_n_branch_points", "mean"),
        area=("err_pixel_area", "mean"), fracdim=("err_fractal_dimension", "mean"))
    g["topology_per_dice_lost"] = g.branches / (1 - g.mean_dice).clip(lower=1e-9)
    print(g.sort_values("topology_per_dice_lost", ascending=False)
          .to_string(float_format=lambda v: f"{v:.3f}"))

    with open(a.out.replace(".csv", "_summary.json"), "w") as fh:
        json.dump(s.to_dict("records"), fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
