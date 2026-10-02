"""How many PRMI ground-truth masks are actually empty, and what does that do to the analysis?

The mask audit sampled 300 masks and reported a MEDIAN foreground fraction of 0.0 for two of the
eight imaging configurations. A median of zero means more than half of those masks contain no root
pixels at all.

That matters more here than it would in most projects:

  * **Dice is degenerate on an empty pair.** Two empty masks have no intersection and no union;
    the conventional fix is to score them 1.0, which silently awards a perfect score for predicting
    nothing. Any Dice averaged over a set containing many empty frames is mostly measuring how often
    the model correctly said "nothing here".
  * **Every trait is degenerate too.** `n_components` is 0, `fractal_dimension` has no structure to
    fit, `mean_diameter` divides by an empty skeleton. The relative-error denominator is zero, so
    the error measure this project is built on cannot be formed at all.
  * **So the effective size of the benchmark is not 63,943 pairs.** It is the number of pairs that
    contain a root, and that is the number the paper must quote when it talks about how much
    evidence the benchmark carries -- the same argument project 02 made about independent polyps.

This module measures the rate exactly rather than estimating it, per configuration and per official
split, and reports the distribution of non-empty masks so the analysable subset is a known quantity
before any trait is computed on it.
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

from prmi_ingest import index_directory  # noqa: E402
from prmi_load import open_source, pair_images_and_masks, to_boolean_mask  # noqa: E402


def census(root, pairs, sample=None, seed=0, verbose=True, report_every=5000):
    """Foreground fraction of every mask (or a random sample). Returns per-pair records."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(pairs))
    if sample and sample < len(pairs):
        idx = rng.permutation(idx)[:sample]
    rows, failures = [], []
    t0 = time.time()
    with open_source(root) as src:
        for n, i in enumerate(idx, 1):
            p = pairs[int(i)]
            rec = p["mask"]
            try:
                m = to_boolean_mask(src.read(rec["path"]), path=rec["path"])
            except Exception as exc:
                failures.append({"path": rec["path"], "error": str(exc)[:200]})
                continue
            rows.append({
                "path": rec["path"],
                "config": rec.get("config"),
                "official_split": rec.get("official_split"),
                "species": rec["species"],
                "group": rec["group"],
                "frac": float(m.mean()),
                "n_fg": int(m.sum()),
                "h": int(m.shape[0]),
                "w": int(m.shape[1]),
            })
            if verbose and n % report_every == 0:
                rate = n / max(time.time() - t0, 1e-9)
                print(f"  {n}/{len(idx)}  ({rate:.0f}/s, {(len(idx) - n) / rate / 60:.1f} min left)",
                      flush=True)
    return rows, failures


def summarise(rows):
    def _stats(sub):
        fr = np.array([r["frac"] for r in sub])
        fg = np.array([r["n_fg"] for r in sub])
        nz = fr > 0
        return {
            "n": len(sub),
            "n_empty": int((~nz).sum()),
            "empty_rate": float((~nz).mean()) if len(sub) else None,
            "median_frac_nonempty": float(np.median(fr[nz])) if nz.any() else None,
            "median_fg_px_nonempty": float(np.median(fg[nz])) if nz.any() else None,
            "max_frac": float(fr.max()) if len(sub) else None,
        }

    out = {"overall": _stats(rows)}
    for key in ("config", "official_split", "species"):
        buckets = defaultdict(list)
        for r in rows:
            buckets[r[key]].append(r)
        out["by_" + key] = {k: _stats(v) for k, v in sorted(buckets.items())}

    # The quantity the paper needs: how many INDEPENDENT tubes still carry a non-empty mask. A
    # benchmark's evidence is bounded by its independent units, not by its frame count -- project
    # 02's F29-F31.
    tubes_all = {r["group"] for r in rows}
    tubes_nonempty = {r["group"] for r in rows if r["frac"] > 0}
    out["groups"] = {
        "n_tubes_total": len(tubes_all),
        "n_tubes_with_any_nonempty_mask": len(tubes_nonempty),
        "tubes_entirely_empty": sorted(tubes_all - tubes_nonempty)[:20],
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--sample", type=int, default=None,
                    help="sample this many masks; omit to census all of them")
    ap.add_argument("--out", default="results/prmi_mask_census.json")
    ap.add_argument("--save-rows", default=None,
                    help="also write the per-mask rows here; needed by prmi_null_baseline.py, "
                         "which reads the full foreground distribution rather than a summary")
    a = ap.parse_args()

    records, _ = index_directory(a.dir)
    pairs, _, _, _ = pair_images_and_masks(records)
    print(f"{len(pairs)} image/mask pairs; "
          f"{'sampling ' + str(a.sample) if a.sample else 'reading all masks'}")

    rows, failures = census(a.dir, pairs, sample=a.sample)
    summary = summarise(rows)
    summary["n_failed"] = len(failures)
    summary["failures"] = failures[:20]
    summary["sampled"] = a.sample

    print("\n=== overall ===")
    print(json.dumps(summary["overall"], indent=2))
    print("\n=== by imaging configuration ===")
    print(f"{'config':32} {'n':>7} {'empty':>7} {'empty %':>8} {'median fg px':>13}")
    for k, v in summary["by_config"].items():
        med = v["median_fg_px_nonempty"]
        print(f"{k:32} {v['n']:>7} {v['n_empty']:>7} {v['empty_rate']:>7.1%} "
              f"{(f'{med:.0f}' if med else '-'):>13}")
    print("\n=== by official split ===")
    for k, v in summary["by_official_split"].items():
        print(f"  {k:6} n={v['n']:>7}  empty={v['empty_rate']:.1%}")
    print("\n=== tubes ===")
    print(json.dumps(summary["groups"], indent=2)[:600])

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(summary, fh, indent=2)
    print("\nwrote " + a.out)
    if a.save_rows:
        with open(a.save_rows, "w") as fh:
            json.dump(rows, fh)
        print("wrote " + a.save_rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
