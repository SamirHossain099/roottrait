"""Do real segmenters lose root branching, or does annotation texture just fake it?

The trait table showed `n_branch_points` wrong by ~95% for EVERY model (0.942-0.972), with a rank
correlation against Dice of ~0. A 95% relative error can mean the prediction has almost no branch
points or roughly twice as many -- opposite failures -- and it could also be an artefact: hand-drawn
masks have ragged edges, ragged edges sprout skeleton spurs, and every spur adds a junction.

Three checks separate those readings, and each writes to `results/` so the paper can cite it:

  1. **Sign.** Signed relative error, and the share of frames with fewer / more / zero junctions.
  2. **Spur pruning.** Remove terminal branches shorter than k px from BOTH skeletons, for k up to
     3x the median root half-width, and recount. If the gap were spur texture it would close.
  3. **Junctions as graph nodes, not pixels.** `traits.n_branch_points` counts skeleton PIXELS with
     three or more neighbours, so one junction counts several times. Clusters are the graph nodes.

Plus the matching robustness check on `prmi_decompose`, which uses the ground-truth skeleton to call
a missed pixel "lost structure" rather than "boundary loss": the same spurs could inflate that share,
so it is recomputed with the centreline restricted to increasingly deep pixels (EDT >= d).
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

import numpy as np  # noqa: E402
from scipy import ndimage as ndi  # noqa: E402

from prmi_analysis import load_models, load_test  # noqa: E402
from prmi_clean import PINHOLE_PX, enclosed_holes, fill_pinholes  # noqa: E402
from roottrait.graph import _K, _ST, _nb, junction_nodes, prune  # noqa: E402,F401
from traits import skeleton  # noqa: E402

PRUNE_K = (0, 3, 6, 10, 15)
CENTRELINE_EDT = (0.0, 1.5, 2.0, 3.0, 4.0)


def tip_depth(mask):
    """Median distance-to-boundary at skeleton tips. A spur ending at the boundary sits near 1."""
    sk = skeleton(mask)
    if not sk.any():
        return np.nan
    tips = sk & (_nb(sk) == 1)
    if not tips.any():
        return np.nan
    return float(np.median(ndi.distance_transform_edt(mask)[tips]))


def lost_structure_share(gts, preds, min_edt):
    """Share of false-negative pixels called lost structure, with a centreline of depth >= min_edt."""
    lost = under = 0
    for g, p in zip(gts, preds):
        fn = g & ~p
        if not fn.any():
            continue
        sk = skeleton(g)
        if min_edt > 0:
            sk = sk & (ndi.distance_transform_edt(g) >= min_edt)
        lab, _ = ndi.label(fn, structure=_ST)
        ids = np.unique(lab[sk & fn])
        ids = ids[ids > 0]
        s = np.isin(lab, ids)
        lost += int((fn & s).sum())
        under += int((fn & ~s).sum())
    return lost / (lost + under) if (lost + under) else np.nan


def analyse_model(gts, preds):
    out = {"prune": {}, "centreline": {}}
    sk_g = [skeleton(g) for g in gts]
    sk_p = [skeleton(p) for p in preds]
    for k in PRUNE_K:
        g = np.array([junction_nodes(prune(s, k)) for s in sk_g])
        p = np.array([junction_nodes(prune(s, k)) for s in sk_p])
        ok = g > 0
        out["prune"][str(k)] = {
            "gt_median_junctions": float(np.median(g)),
            "pred_median_junctions": float(np.median(p)),
            "median_signed_rel_err": float(np.median((p[ok] - g[ok]) / g[ok])) if ok.any()
            else None,
            "share_pred_fewer": float((p < g).mean()),
            "share_pred_more": float((p > g).mean()),
            "share_pred_zero_where_gt_has_some": float((p[ok] == 0).mean()) if ok.any() else None,
        }
    for d in CENTRELINE_EDT:
        out["centreline"][str(d)] = lost_structure_share(gts, preds, d)
    out["tip_depth_gt"] = float(np.nanmedian([tip_depth(g) for g in gts]))
    out["tip_depth_pred"] = float(np.nanmedian([tip_depth(p) if p.any() else np.nan
                                                for p in preds]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="peanut_640x480_DPI120")
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--cache-dir", default="data/cache")
    ap.add_argument("--models-dir", default="results/prmi_models")
    ap.add_argument("--n-frames", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fill-px", type=int, default=PINHOLE_PX,
                    help="fill enclosed holes up to this size in both masks first (F25); 0 off")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    suffix = "" if a.fill_px else "_unfilled"
    out_path = a.out or f"results/prmi_branching_{a.config}{suffix}.json"

    models = load_models(a.models_dir, a.config)
    gt, meta = load_test(a.cache_dir, a.config, a.size)
    ne = np.flatnonzero([not m["gt_empty_resized"] for m in meta])
    idx = np.sort(np.random.default_rng(a.seed).choice(ne, min(a.n_frames, len(ne)),
                                                         replace=False))
    holes_before = [len(enclosed_holes(gt[i])) for i in idx]
    gts = [fill_pinholes(gt[i], a.fill_px) for i in idx]
    root_half_width = float(np.median([ndi.distance_transform_edt(g).max() for g in gts]))

    res = {"config": a.config, "fill_px": a.fill_px,
           "gt_holes_per_frame_median": float(np.median(holes_before)),
           "n_frames": int(len(idx)), "prune_k": list(PRUNE_K),
           "centreline_edt": list(CENTRELINE_EDT), "median_root_half_width_px": root_half_width,
           "models": {}}
    t0 = time.time()
    for k, m in enumerate(models, 1):
        pred = np.load(m["pred_path"])
        res["models"][m["tag"]] = analyse_model(gts, [fill_pinholes(pred[i], a.fill_px)
                                                      for i in idx])
        r = res["models"][m["tag"]]
        print(f"  [{k}/{len(models)}] {m['tag'].split('__', 1)[-1]:20} "
              f"junctions GT/pred k=0 {r['prune']['0']['gt_median_junctions']:.0f}/"
              f"{r['prune']['0']['pred_median_junctions']:.0f}  k=15 "
              f"{r['prune']['15']['gt_median_junctions']:.0f}/"
              f"{r['prune']['15']['pred_median_junctions']:.0f}  lost(EDT>=1.5)="
              f"{r['centreline']['1.5']:.3f}  ({(time.time() - t0) / 60:.1f} min)", flush=True)

    # Across-model summary, so the paper cites one number per question rather than eighteen.
    def across(fn):
        v = np.array([fn(r) for r in res["models"].values()], float)
        return {"min": float(np.nanmin(v)), "median": float(np.nanmedian(v)),
                "max": float(np.nanmax(v))}
    res["summary"] = {
        "signed_err_unpruned": across(lambda r: r["prune"]["0"]["median_signed_rel_err"]),
        "signed_err_pruned_15": across(lambda r: r["prune"]["15"]["median_signed_rel_err"]),
        "gt_junctions_unpruned": across(lambda r: r["prune"]["0"]["gt_median_junctions"]),
        "gt_junctions_pruned_15": across(lambda r: r["prune"]["15"]["gt_median_junctions"]),
        "pred_junctions_unpruned": across(lambda r: r["prune"]["0"]["pred_median_junctions"]),
        "pred_junctions_pruned_15": across(lambda r: r["prune"]["15"]["pred_median_junctions"]),
        "share_pred_fewer": across(lambda r: r["prune"]["0"]["share_pred_fewer"]),
        "share_pred_zero": across(lambda r: r["prune"]["0"]["share_pred_zero_where_gt_has_some"]),
        "lost_structure_share_full_skeleton": across(lambda r: r["centreline"]["0.0"]),
        "lost_structure_share_edt_1_5": across(lambda r: r["centreline"]["1.5"]),
        "lost_structure_share_edt_4": across(lambda r: r["centreline"]["4.0"]),
        "tip_depth_gt": res["models"][next(iter(res["models"]))]["tip_depth_gt"],
        "tip_depth_pred": across(lambda r: r["tip_depth_pred"]),
    }
    with open(out_path, "w") as fh:
        json.dump(res, fh, indent=2)
    print(json.dumps(res["summary"], indent=2))
    print("wrote " + out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
