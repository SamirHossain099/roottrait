"""Figures for the real-data GPU arm (F18-F24). ONE FIGURE PER FILE.

Each is standalone -- own title, legend and axis labels -- so a manuscript can cite any one of them;
composing them side by side is a layout decision for the document, not for this code.

Every value plotted is read from `results/`; nothing is recomputed here.

  f18_headline_vs_roots_dice.png      all-frames against roots-only Dice, with the do-nothing floor
  f19_branching_lost.png              junctions in annotation against prediction, under spur pruning
  f20_error_decomposition.png         what real segmenters get wrong, per architecture
  f23_architecture_rank_transfer.png  ranking in one configuration against the other
  f24_gaps_against_paired_floor.png   adjacent architecture gaps against their paired floor
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# isort: off
import resources  # noqa: F401,E402  MUST load before numpy: caps BLAS threads
# isort: on

import argparse  # noqa: E402
import json  # noqa: E402
from collections import defaultdict  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from prmi_decompose import KINDS  # noqa: E402

CONFIGS = ("peanut_640x480_DPI120", "peanut_736x552_DPI150")
LABEL = {"peanut_640x480_DPI120": "640 by 480, 120 dpi (7 test tubes)",
         "peanut_736x552_DPI150": "736 by 552, 150 dpi (6 test tubes)"}
SHORT = {"peanut_640x480_DPI120": "640 by 480", "peanut_736x552_DPI150": "736 by 552"}
CCOL = {"peanut_640x480_DPI120": "#2c6fbb", "peanut_736x552_DPI150": "#c0392b"}
KCOL = {"over_inclusive": "#e67e22", "false_positive": "#8e44ad",
        "under_inclusive": "#16a085", "lost_structure": "#c0392b"}
KLAB = {"over_inclusive": "over-inclusive (boundary pushed out)",
        "false_positive": "detached false positive",
        "under_inclusive": "under-inclusive (boundary pulled in)",
        "lost_structure": "lost structure (centreline cut)"}
# The manuscript's names, so figure and text read the same.
ARCH = {"unetpp_r34": "UNet++", "deeplabv3p_r34": "DeepLabV3+", "unet_r34": "U-Net R34",
        "unet_r50": "U-Net R50", "fpn_r34": "FPN", "manet_r34": "MA-Net"}


def _j(kind, c, d):
    with open(os.path.join(d, f"prmi_{kind}_{c}.json")) as fh:
        return json.load(fh)


def _save(fig, out):
    fig.savefig(out, dpi=600, bbox_inches="tight")
    fig.savefig(os.path.splitext(out)[0] + ".pdf", bbox_inches="tight")  # vector copy (section 7)
    plt.close(fig)
    print("  wrote " + out)


def fig_headline(d, out):
    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    for c in CONFIGS:
        a = _j("analysis", c, d)
        x = [m["mean_dice_nonempty_only"] for m in a["models"]]
        y = [m["mean_dice_all_frames"] for m in a["models"]]
        ax.scatter(x, y, s=34, color=CCOL[c], label=LABEL[c], zorder=3, edgecolor="white", lw=.6)
        ax.axhline(a["null_dice_all_frames"], color=CCOL[c], ls=":", lw=1.4)
        ax.text(0.405, a["null_dice_all_frames"] + 0.006,
                f"predicting nothing, {SHORT[c]}: {a['null_dice_all_frames']:.3f}",
                color=CCOL[c], fontsize=8)
    lim = (0.40, 0.88)
    ax.plot(lim, lim, color="#999", lw=1, ls="--")
    ax.text(0.80, 0.785, "equal", color="#888", fontsize=8, rotation=38)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("mean Dice on frames that contain roots")
    ax.set_ylabel("mean Dice over all test frames")
    ax.legend(fontsize=8.5, loc="upper left", title="18 trained models each",
              title_fontsize=8.5)
    ax.grid(alpha=.25)
    # No message title on a manuscript figure: the caption states the result (section 11).
    _save(fig, out)


def fig_branching(d, out):
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    for c in CONFIGS:
        b = _j("branching", c, d)
        ks = b["prune_k"]
        gt = [np.median([m["prune"][str(k)]["gt_median_junctions"] for m in b["models"].values()])
              for k in ks]
        pr = [np.median([m["prune"][str(k)]["pred_median_junctions"]
                         for m in b["models"].values()]) for k in ks]
        ax.plot(ks, gt, "-o", color=CCOL[c], lw=2, label=f"annotation, {SHORT[c]}")
        ax.plot(ks, pr, "--s", color=CCOL[c], lw=2, mfc="white", label=f"prediction, {SHORT[c]}")
    ax.set_xlabel("spurs pruned: terminal branches shorter than k px removed from both skeletons")
    ax.set_ylabel("junctions per frame (median; graph nodes)")
    ax.set_ylim(bottom=-0.4)
    ax.grid(alpha=.25)
    ax.legend(fontsize=8.5, ncol=2, loc="upper right")
    ax.set_title("The annotation keeps its junctions when spurs are pruned;\n"
                 "the median prediction has none", fontsize=11)
    _save(fig, out)


def fig_decomposition(d, out):
    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    archs = list(ARCH)
    w = 0.38
    for ci, c in enumerate(CONFIGS):
        a = _j("analysis", c, d)
        by = defaultdict(list)
        for m in a["models"]:
            by[m["arch"]].append(m["decomp_nonempty"])
        x = np.arange(len(archs)) + (ci - 0.5) * w
        bottom = np.zeros(len(archs))
        for k in KINDS:
            v = np.array([np.mean([s[k] for s in by[ar]]) for ar in archs])
            ax.bar(x, v, w * 0.92, bottom=bottom, color=KCOL[k],
                   label=KLAB[k] if ci == 0 else None, edgecolor="white", lw=.5)
            bottom += v
        for xi in x:
            ax.text(xi, 1.01, "1" if ci == 0 else "2", ha="center", fontsize=7.5, color="#555")
    ax.set_xticks(np.arange(len(archs)))
    ax.set_xticklabels([ARCH[a] for a in archs], fontsize=9)
    ax.set_ylim(0, 1.06)
    ax.set_ylabel("share of error pixels on root frames (mean of 3 seeds)")
    ax.set_xlabel("architecture  (left bar: 640 by 480, right bar: 736 by 552)")
    ax.legend(fontsize=8.5, loc="lower center", bbox_to_anchor=(0.5, -0.33), ncol=2,
              frameon=False)
    _save(fig, out)


def fig_rank_transfer(d, out):
    f1, f2 = _j("floor", CONFIGS[0], d), _j("floor", CONFIGS[1], d)
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    for a in ARCH:
        r1, r2 = f1["ranking"].index(a) + 1, f2["ranking"].index(a) + 1
        col = "#c0392b" if abs(r1 - r2) >= 2 else "#555"
        ax.plot([0, 1], [r1, r2], "-o", color=col, lw=2 if col != "#555" else 1.4)
        ax.text(-0.04, r1, ARCH[a], ha="right", va="center", fontsize=9)
        ax.text(1.04, r2, ARCH[a], ha="left", va="center", fontsize=9)
    ax.set_xlim(-0.75, 1.75)
    ax.set_ylim(6.6, 0.4)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["640 by 480\n120 dpi", "736 by 552\n150 dpi"])
    ax.set_yticks(range(1, 7))
    ax.set_ylabel("rank by mean Dice on root frames (1 = best)")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    _save(fig, out)


def fig_gaps(d, out):
    # Log axis from a fixed lower edge. Every gap and every non-clipped floor is positive, so a
    # symlog axis only added a meaningless negative region; bars start at the left edge instead.
    x0, x1 = 5e-4, 0.2
    fig, ax = plt.subplots(figsize=(8.8, 5.8))
    y = 0
    ticks, labels = [], []
    for c in CONFIGS:
        f = _j("floor", c, d)
        for a in f["adjacent"]:
            if not a["floor_clipped"]:
                ax.plot([x0, a["floor"]], [y, y], color="#bbb", lw=7, solid_capstyle="butt")
            ax.plot([x0, a["halfwidth_now"]], [y, y], color=CCOL[c], lw=1.4, alpha=.7)
            ax.plot(a["gap"], y, "D", color="black", ms=6, zorder=3)
            ticks.append(y)
            tail = "   [floor unknown]" if a["floor_clipped"] else ""
            labels.append(f"{SHORT[c]}: {ARCH[a['upper']]} > {ARCH[a['lower']]}{tail}")
            y += 1
        y += 0.8
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlim(x0, x1)
    ax.set_xlabel("Dice on root frames (log scale)")
    handles = [plt.Line2D([], [], color="#bbb", lw=7, label="paired floor (infinite seeds)"),
               plt.Line2D([], [], color="#555", lw=1.4, label="interval half-width at 3 seeds"),
               plt.Line2D([], [], marker="D", color="black", ls="",
                          label="observed gap between adjacent architectures")]
    ax.legend(handles=handles, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.12),
              ncol=3, frameon=False)
    ax.grid(alpha=.25, axis="x")
    ax.set_title("At three seeds, adjacent architectures are mostly not separated;\n"
                 "a gap left of its grey bar can never be, on this test set", fontsize=11)
    _save(fig, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--outdir", default="figures")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    o = a.outdir
    fig_headline(a.results, os.path.join(o, "f18_headline_vs_roots_dice.png"))
    fig_branching(a.results, os.path.join(o, "f19_branching_lost.png"))
    fig_decomposition(a.results, os.path.join(o, "f20_error_decomposition.png"))
    fig_rank_transfer(a.results, os.path.join(o, "f23_architecture_rank_transfer.png"))
    fig_gaps(a.results, os.path.join(o, "f24_gaps_against_paired_floor.png"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
