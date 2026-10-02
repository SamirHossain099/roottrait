"""The three real-data analyses, over the trained models.

  1. **Rank inversion** -- rank models by Dice, rank them by trait error, and ask whether the two
     orders agree. This is the right panel of the brief's section 5 figure, and the only part of
     this paper that needs a GPU.
  2. **The failure-mode fingerprint** -- which named degradation does each real model resemble?
     (`prmi_fingerprint.py` has the method and the reasoning.)
  3. **The resolution floor** -- can a test set of this many independent tubes resolve the
     differences between these models at all? Computed with `groupeval`, this project's sibling
     artifact from project 02, on PRMI's tube grouping.

Every number written here is read back by `tests/test_prmi_analysis.py` from `results/`, never
pinned as a literal in a test.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# isort: off
import resources  # noqa: F401,E402  MUST load before numpy: caps BLAS threads
# isort: on

import argparse  # noqa: E402
import glob  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from collections import defaultdict  # noqa: E402

import numpy as np  # noqa: E402

from dose_response import rel_error, spearman  # noqa: E402
from prmi_clean import PINHOLE_PX, fill_pinholes  # noqa: E402
from prmi_dataset import cache_path  # noqa: E402
from prmi_decompose import KINDS, decompose_set, dominant  # noqa: E402
from prmi_fingerprint import fingerprint  # noqa: E402
from traits import TRAIT_KIND, TRAITS  # noqa: E402


def load_models(outdir, config):
    """Every completed unit for one configuration."""
    out = []
    for s in sorted(glob.glob(os.path.join(outdir, f"{config}__*_summary.json"))):
        with open(s) as fh:
            d = json.load(fh)
        if d.get("failed"):
            continue
        tag = os.path.basename(s)[: -len("_summary.json")]
        pred = os.path.join(outdir, tag + "_pred.npy")
        if not os.path.exists(pred):
            continue
        d["tag"] = tag
        d["pred_path"] = pred
        d["dice_path"] = os.path.join(outdir, tag + "_dice.npy")
        out.append(d)
    return out


def load_test(cache_dir, config, size):
    import torch
    p = cache_path(cache_dir, config, "test", size)
    d = torch.load(p, map_location="cpu", weights_only=False)
    with open(p.replace(".pt", "_meta.json")) as fh:
        meta = json.load(fh)
    gt = (d["masks"].squeeze(1).numpy() > 127)
    return gt, meta


def per_model_traits(gt, pred, idx, traits):
    """Relative trait error per frame, for the frames in `idx`. Returns {trait: array}."""
    acc = {t: [] for t in traits}
    for i in idx:
        g, p = gt[i], pred[i]
        base = {t: TRAITS[t](g) for t in traits}
        for t in traits:
            acc[t].append(rel_error(base[t], TRAITS[t](p)))
    return {t: np.array(v, float) for t, v in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="peanut_640x480_DPI120")
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--cache-dir", default="data/cache")
    ap.add_argument("--models-dir", default="results/prmi_models")
    ap.add_argument("--n-trait-frames", type=int, default=600,
                    help="non-empty test frames used for the trait comparison")
    ap.add_argument("--n-fingerprint-frames", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fill-px", type=int, default=PINHOLE_PX,
                    help="fill enclosed holes up to this size in BOTH annotation and prediction "
                         "before any trait or decomposition (F25); 0 disables")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    suffix = "" if a.fill_px else "_unfilled"
    a.out = a.out or f"results/prmi_analysis_{a.config}{suffix}.json"

    models = load_models(a.models_dir, a.config)
    if not models:
        raise SystemExit(f"no completed models for {a.config} in {a.models_dir}")
    gt, meta = load_test(a.cache_dir, a.config, a.size)
    # F25: PRMI annotations carry one-pixel holes that turn into skeleton loops. Filled identically
    # in annotation and prediction; the Dice values below come from training and are not touched.
    gt = np.stack([fill_pinholes(g, a.fill_px) for g in gt])
    traits = list(TRAITS)

    nonempty = np.array([not m["gt_empty_resized"] for m in meta])
    groups = np.array([m["group"] for m in meta])
    rng = np.random.default_rng(a.seed)
    ne_idx = np.flatnonzero(nonempty)
    trait_idx = np.sort(rng.permutation(ne_idx)[:a.n_trait_frames])
    fp_idx = np.sort(rng.permutation(ne_idx)[:a.n_fingerprint_frames])

    print(f"{len(models)} models | {len(meta)} test frames, {nonempty.sum()} non-empty, "
          f"{len(set(groups))} tubes")
    print(f"traits on {len(trait_idx)} frames, fingerprint on {len(fp_idx)}")

    # ------------------------------------------------------------------ per-model measurement
    rows, fps = [], {}
    t0 = time.time()
    for k, m in enumerate(models, 1):
        pred = np.stack([fill_pinholes(p_, a.fill_px) for p_ in np.load(m["pred_path"])])
        d_all = np.load(m["dice_path"])
        te = per_model_traits(gt, pred, trait_idx, traits)
        row = {
            "tag": m["tag"], "arch": m["arch"], "seed": m["seed"],
            "mean_dice_all_frames": m["mean_dice_all_frames"],
            "mean_dice_nonempty_only": m["mean_dice_nonempty_only"],
            "mean_dice_empty_only": m["mean_dice_empty_only"],
            "pred_empty_rate": m["pred_empty_rate"],
            "dice_on_trait_frames": float(d_all[trait_idx].mean()),
        }
        for t in traits:
            v = te[t]
            v = v[np.isfinite(v)]
            row["err_" + t] = float(np.median(v)) if v.size else np.nan
        rows.append(row)

        # The PRIMARY failure-mode answer: count error pixels by kind (prmi_decompose). On roots only,
        # and separately over every frame -- on an empty-GT frame every predicted pixel is a detached
        # false positive by definition, so the two answer different questions and differ by exactly
        # the empty-frame behaviour F15 is about.
        dec_ne = decompose_set([gt[i] for i in ne_idx], [pred[i] for i in ne_idx])
        dec_all = decompose_set(list(gt), list(pred))
        row["decomp_nonempty"] = dec_ne["pixel_weighted"]
        row["decomp_nonempty_frame_weighted"] = dec_ne["frame_weighted"]
        row["decomp_all_frames"] = dec_all["pixel_weighted"]
        row["dominant_error_nonempty"] = dominant(dec_ne["pixel_weighted"])

        # SECONDARY, kept for the record: nearest single mode. Valid on pure degradations, NOT on
        # mixtures, and real segmenters are mixtures -- see prmi_decompose's docstring.
        fps[m["tag"]] = fingerprint([gt[i] for i in fp_idx], [pred[i] for i in fp_idx],
                                    seed=a.seed, traits=traits)
        sh = dec_ne["pixel_weighted"]
        print(f"  [{k}/{len(models)}] {m['tag'].split('__', 1)[-1]:24} "
              f"dice={row['mean_dice_nonempty_only']:.4f}  "
              + " ".join(f"{kk[:5]}={sh[kk]:.2f}" for kk in KINDS)
              + f"  ({(time.time() - t0) / 60:.1f} min)", flush=True)

    # ------------------------------------------------------------------ 1. rank inversion
    by_dice = sorted(rows, key=lambda r: -r["mean_dice_nonempty_only"])
    inversion, undefined = {}, {}
    dice_vals = np.array([r["mean_dice_nonempty_only"] for r in rows])
    # Robustness: the two lowest-Dice units are worst at nearly everything, so a correlation can be
    # carried by them alone. Measured on the first run: n_components went from -0.40 to -0.04 when
    # they were dropped, i.e. its whole correlation was those two models.
    weakest = set(np.argsort(dice_vals)[:2].tolist())
    keep = np.array([i not in weakest for i in range(len(rows))])
    for t in traits:
        ev = np.array([r["err_" + t] for r in rows], float)
        ok = np.isfinite(ev)
        if ok.sum() < 4:
            continue
        rho = spearman(dice_vals[ok], ev[ok])
        if not np.isfinite(rho):
            # n_holes: every model has the same median error, so there is no ranking to correlate.
            undefined[t] = "no variation in median error across models"
            continue
        v = ev[ok]
        # Dice up should mean trait error down, so a faithful proxy gives rho near -1.
        inversion[t] = {
            "kind": TRAIT_KIND[t],
            "spearman_dice_vs_trait_error": rho,
            "spearman_without_two_weakest": spearman(dice_vals[ok & keep], ev[ok & keep]),
            # A correlation near zero means nothing if the trait error barely varies across models:
            # there is then nothing for Dice to predict. Report the spread so that reading is
            # impossible to make by accident.
            "error_min": float(v.min()), "error_max": float(v.max()),
            "error_spread_ratio": float(v.max() / max(v.min(), 1e-9)),
            "best_by_dice_is_best_by_trait": bool(
                rows[int(np.argmax(dice_vals))]["tag"] == rows[int(np.nanargmin(ev))]["tag"]),
            "best_by_trait_tag": rows[int(np.nanargmin(ev))]["tag"],
        }

    out = {
        "config": a.config,
        "fill_px": a.fill_px,
        "n_models": len(rows),
        "n_test_frames": int(len(meta)),
        "n_test_nonempty": int(nonempty.sum()),
        "n_test_tubes": int(len(set(groups))),
        "null_dice_all_frames": float(1.0 - nonempty.mean()),
        "trait_frames": int(len(trait_idx)),
        "fingerprint_frames": int(len(fp_idx)),
        "models": rows,
        "best_by_dice": by_dice[0]["tag"],
        "rank_inversion": inversion,
        "rank_inversion_undefined": undefined,
        "fingerprints": {k: {kk: vv for kk, vv in v.items() if kk != "degradations"}
                         for k, v in fps.items()},
        "fingerprint_detail": fps,
        # Deliberately NOT computed here. The first version compared gaps between seed-units to
        # groupeval's SINGLE-MEAN floor, overstating the floor about 8x. The paired,
        # architecture-level floor is src/prmi_floor.py.
        "resolution_floor": f"see results/prmi_floor_{a.config}.json (src/prmi_floor.py)",
    }
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    print("\nwrote " + a.out)
    report(out)
    return 0


def report(out):
    print("\n=== 1. rank inversion (Spearman of Dice against trait error; -1 = faithful) ===")
    inv = out["rank_inversion"]
    for t, v in sorted(inv.items(), key=lambda kv: kv[1]["spearman_dice_vs_trait_error"]):
        print(f"  {t:20} {v['kind']:9} rho={v['spearman_dice_vs_trait_error']:+.3f}  "
              f"w/o 2 weakest={v['spearman_without_two_weakest']:+.3f}  "
              f"error {v['error_min']:.3f}-{v['error_max']:.3f} (x{v['error_spread_ratio']:.2f})")
    for t, why in out["rank_inversion_undefined"].items():
        print(f"  {t:20} undefined: {why}")
    print(f"  best by Dice: {out['best_by_dice'].split('__', 1)[-1]}")

    print("\n=== 2a. error decomposition on roots (PRIMARY; pixel shares by kind) ===")
    print(f"  {'model':26} " + " ".join(f"{k:>16}" for k in KINDS))
    for r in out["models"]:
        print(f"  {r['tag'].split('__', 1)[-1]:26} "
              + " ".join(f"{r['decomp_nonempty'][k]:>16.3f}" for k in KINDS))
    mean_sh = {k: float(np.mean([r["decomp_nonempty"][k] for r in out["models"]])) for k in KINDS}
    print(f"  {'MEAN':26} " + " ".join(f"{mean_sh[k]:>16.3f}" for k in KINDS))

    print("\n=== 2b. nearest single mode (SECONDARY; unreliable on mixtures) ===")
    near, fam = defaultdict(int), defaultdict(int)
    for tag, f in out["fingerprints"].items():
        near[f["nearest"]] += 1
        fam[f["nearest_family"]] += 1
        print(f"  {tag.split('__', 1)[-1]:28} loss={f['median_dice_loss']:.4f} "
              f"area_dir={f['model_area_direction']:+.3f} "
              f"family={f['nearest_family']:16} ({f['nearest']})")
    print(f"  --> families: {dict(fam)}")
    print(f"  --> individual: {dict(near)}")

    print("\n(resolution floor: src/prmi_floor.py -- the paired, architecture-level one)")


if __name__ == "__main__":
    sys.exit(main())
