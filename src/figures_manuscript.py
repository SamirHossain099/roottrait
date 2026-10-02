"""Manuscript Figure 1: one real PRMI mask, six failure modes, the same Dice.

The paper's claim in one image. A real ground-truth mask from the 640x480 peanut test split is
degraded six ways, each calibrated (`calibrate.solve_severity`) to the SAME Dice loss on this mask.
Under each panel: the Dice actually reached and four trait readouts. A reader sees the metric hold
still while the traits move.

The mask was chosen by eye from a contact sheet of masks with at least six junctions after pruning
spurs to 15 px. The first choice (index 69) was a single unbranched root whose 19 "junctions" were
spurs from a notched annotation edge -- the artefact F19 had to rule out -- so it was replaced.
The second choice (index 1918, two roots crossing and a lateral) failed once annotation pinholes
were filled: junction breaks could no longer reach the target Dice loss, because its "62 junctions"
were almost all pinhole loops (F25). Index 170 was chosen from masks that keep at least three
junctions AFTER filling: a branching system with fine laterals, not separate roots crossing.

Error pixels are coloured so they are visible at full-frame scale, where the root covers ~4% of the
image: correct root in dark grey, added pixels in orange, missed pixels in blue.

The numbers printed in the artwork are cached to `results/fig01_numbers.json` by the same code that
draws them (a figure's numbers must exist outside the PNG), and the
manuscript test reads them from there.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# isort: off
import resources  # noqa: F401,E402  MUST load before numpy: caps BLAS threads
# isort: on

import argparse  # noqa: E402
import json  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from calibrate import SEVERITY_BOUNDS, solve_severity  # noqa: E402
from degrade import DEGRADATIONS  # noqa: E402
from prmi_analysis import load_test  # noqa: E402
from prmi_branching import junction_nodes, prune  # noqa: E402
from prmi_clean import enclosed_holes, fill_pinholes  # noqa: E402
from traits import dice, n_components, pixel_area, skeleton, total_length  # noqa: E402

ORDER = ["boundary_dilate", "boundary_erode", "thin_dropout", "junction_break", "speckle", "blob"]
TITLE = {"boundary_dilate": "boundary pushed out", "boundary_erode": "boundary pulled in",
         "thin_dropout": "thin branches lost", "junction_break": "junctions cut",
         "speckle": "scattered false pixels", "blob": "false blobs"}


def readout(gt, m):
    a0, l0 = pixel_area(gt), total_length(gt)
    return {
        "dice": float(dice(gt, m)),
        "area_err": float((pixel_area(m) - a0) / a0),
        "length_err": float((total_length(m) - l0) / l0),
        # Pruned to 15 px, the same measure as F19: unpruned counts include skeleton spurs from the
        # notched annotation edge, which is texture, not structure.
        "junctions": int(junction_nodes(prune(skeleton(m), 15))),
        "components": int(n_components(m)),
    }


def overlay(gt, m):
    img = np.ones(gt.shape + (3,))
    img[gt & m] = (0.20, 0.20, 0.20)
    img[m & ~gt] = (0.90, 0.49, 0.13)
    img[gt & ~m] = (0.17, 0.44, 0.73)
    return img


def fig1(config, index, dose, out_png, out_json, size=320, cache_dir="data/cache", seed=0):
    gt_all, meta = load_test(cache_dir, config, size)
    # Fill annotation pinholes first (F25): every real-data trait in the paper is computed after it,
    # and the caption says so. Degrading an unfilled mask would mix annotation texture into Fig 1.
    gt = fill_pinholes(gt_all[index])
    panels = [("ground truth", gt, readout(gt, gt), None)]
    for i, name in enumerate(ORDER):
        fn = DEGRADATIONS[name][0]
        lo, hi = SEVERITY_BOUNDS[name]
        rs = 1000 + i + seed
        sev, _ = solve_severity(fn, gt, dose, lo, hi, rng_seed=rs)
        if not np.isfinite(sev):
            raise SystemExit(f"{name} cannot reach a Dice loss of {dose} on this mask")
        m = fn(gt, sev, np.random.default_rng(rs))
        panels.append((TITLE[name], m, readout(gt, m), name))

    fig, axes = plt.subplots(1, 7, figsize=(17.5, 3.55))
    fig.subplots_adjust(top=0.80, bottom=0.27, wspace=0.08)
    for k, (ax, (title, m, r, _name)) in enumerate(zip(axes, panels)):
        ax.imshow(overlay(gt, m) if k else np.where(gt[..., None], 0.2, 1.0) * np.ones(3),
                  interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(0.02, 0.98, "abcdefg"[k], transform=ax.transAxes, fontsize=13,
                fontweight="bold", va="top", ha="left")
        ax.set_title(title, fontsize=10)
        if k == 0:
            txt = (f"junctions {r['junctions']}\ncomponents {r['components']}\n"
                   f"root pixels {int(pixel_area(gt))}")
        else:
            txt = (f"Dice {r['dice']:.3f}\narea {r['area_err']:+.0%}   "
                   f"length {r['length_err']:+.0%}\n"
                   f"junctions {r['junctions']}   components {r['components']}")
        ax.text(0.5, -0.04, txt, transform=ax.transAxes, fontsize=8.6, ha="center", va="top")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in
               ((0.20, 0.20, 0.20), (0.90, 0.49, 0.13), (0.17, 0.44, 0.73))]
    fig.legend(handles, ["root, correct", "added (false positive)", "missed (false negative)"],
               loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False,
               fontsize=9.5)
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    fig.savefig(os.path.splitext(out_png)[0] + ".pdf", bbox_inches="tight")  # vector copy (section 7)
    plt.close(fig)

    with open(out_json, "w") as fh:
        json.dump({"config": config, "test_index": index, "path": meta[index]["path"],
                   "target_dice_loss": dose,
                   "panels": {(name or "ground_truth"): r for _t, _m, r, name in panels}},
                  fh, indent=2)
    print("  wrote " + out_png)
    print("  wrote " + out_json)


def fig3(config, index, out_png, out_json, crop=(8, 176, 96), size=320, cache_dir="data/cache"):
    """Annotation pinholes and the skeleton junctions they create (F25).

    One annotated mask, a crop where its one-pixel holes are densest, shown with its skeleton and
    junctions before (a) and after (b) filling holes of at most 10 pixels. Frame 1918 was the
    mask whose "62 junctions" turned out to be pinhole loops (see `fig1`), so it is the honest
    illustration of the effect.
    """
    gt_all, meta = load_test(cache_dir, config, size)
    raw = gt_all[index]
    filled = fill_pinholes(raw)
    y0, x0, w = crop

    def junction_mask(m):
        from scipy import ndimage as ndi
        sk = prune(skeleton(m), 15)
        nb = ndi.convolve(sk.astype(int), np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]]),
                          mode="constant")
        return sk, sk & (nb >= 3)

    nums = {}
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.6))
    fig.subplots_adjust(wspace=0.06, top=0.86, bottom=0.18)
    for k, (ax, m, title) in enumerate(zip(axes, (raw, filled),
                                           ("annotation as distributed",
                                            "after filling holes of at most 10 pixels"))):
        sk, jn = junction_mask(m)
        img = np.ones(m.shape + (3,))
        img[m] = (0.72, 0.72, 0.72)
        img[sk] = (0.10, 0.10, 0.10)
        img[jn] = (0.80, 0.10, 0.10)
        ax.imshow(img[y0:y0 + w, x0:x0 + w], interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=10)
        ax.text(0.02, 0.98, "ab"[k], transform=ax.transAxes, fontsize=13, fontweight="bold",
                va="top", ha="left", color="black",
                bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"))
        holes = sum(1 for s_ in enclosed_holes(m) if s_ <= 10)
        jcount = int(junction_nodes(prune(skeleton(m), 15)))
        nums["raw" if k == 0 else "filled"] = {"pinholes_whole_frame": holes,
                                               "junctions_whole_frame": jcount}
        ax.text(0.5, -0.04, f"whole frame: {holes} holes of 10 px or less, {jcount} junctions",
                transform=ax.transAxes, fontsize=9, ha="center", va="top")
    nums["dice_raw_vs_filled"] = float(dice(raw, filled))
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in
               ((0.72, 0.72, 0.72), (0.10, 0.10, 0.10), (0.80, 0.10, 0.10))]
    fig.legend(handles, ["root mask", "skeleton (spurs under 15 px removed)", "junction"],
               loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, fontsize=9)
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    fig.savefig(os.path.splitext(out_png)[0] + ".pdf", bbox_inches="tight")  # vector copy (section 7)
    plt.close(fig)
    nums.update({"config": config, "test_index": index, "path": meta[index]["path"],
                 "crop_y0_x0_size": list(crop)})
    with open(out_json, "w") as fh:
        json.dump(nums, fh, indent=2)
    print("  wrote " + out_png)
    print("  wrote " + out_json)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="peanut_640x480_DPI120")
    ap.add_argument("--index", type=int, default=170)
    ap.add_argument("--dose", type=float, default=0.05)
    ap.add_argument("--outdir", default="figures")
    a = ap.parse_args()
    fig1(a.config, a.index, a.dose, os.path.join(a.outdir, "fig01_same_dice_six_failures.png"),
         "results/fig01_numbers.json")
    fig3(a.config, 1918, os.path.join(a.outdir, "fig03_pinholes.png"), "results/fig03_numbers.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
