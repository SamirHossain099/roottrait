"""The money plot (brief §5, left panel) plus the fractal-validation figure.

Left panel of the brief's two-panel figure is buildable now: trait error against Dice, one line per
trait, with the fragile traits diverging from the robust ones. The right panel (rank inversion
across real models) needs model outputs and is not attempted.

Five-second read: *the metric the field optimises is not the metric the field needs.*
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# isort: off
import resources  # noqa: F401,E402  MUST load before numpy: caps BLAS threads
# isort: on

import argparse  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from validate_fractal import KNOWN  # noqa: E402

FAMILY_COLOUR = {"area": "#2c6fbb", "length": "#2e8b57",
                 "topology": "#c0392b", "scaling": "#8e44ad"}


def _save(fig, out):
    fig.savefig(out, dpi=170, bbox_inches="tight")
    print(f"  wrote {out}")
    plt.close(fig)
    return out


def fig_error_at_dice99(df, out):
    """F5 (SUPERSEDED by F12): trait error in the Dice >= 0.99 bucket, uncalibrated ladder.

    Kept because it is what the uncalibrated ladder said, and the difference from F12 is itself the
    argument. `Dice >= 0.99` is a bucket whose membership is decided by the severity grids, so this
    panel moves when those grids move -- it once moved 28-fold on a local refactor. Cite
    `f12a_error_span_at_fixed_dice.png` for the claim; cite this only to show the contrast.
    """
    from dose_response import analyse
    s = analyse(df).dropna(subset=["median_err_at_dice99"]).sort_values("median_err_at_dice99")

    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    y = np.arange(len(s))
    ax.barh(y, s.median_err_at_dice99.clip(lower=1e-3),
            color=[FAMILY_COLOUR[f] for f in s.family], alpha=.9)
    ax.set_yticks(y)
    ax.set_yticklabels(s.trait, fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("median relative trait error at Dice >= 0.99")
    ax.axvline(0.05, color="k", ls="--", lw=1.4)
    ax.text(0.055, 0.3, "5% error", fontsize=8, rotation=90, va="bottom")
    ax.grid(alpha=.25, axis="x", which="both")
    for yi, v in zip(y, s.median_err_at_dice99):
        ax.text(max(v, 1e-3) * 1.25, yi, f"{v:.0%}" if v < 10 else f"{v:.0f}x",
                va="center", fontsize=8)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=.9) for c in FAMILY_COLOUR.values()]
    ax.legend(handles, list(FAMILY_COLOUR), fontsize=8, frameon=False, loc="lower right",
              title="trait family (a-priori)", title_fontsize=8)
    ax.set_title("At a segmentation everyone would accept,\nthese are the errors you actually get",
                 fontsize=11.5)
    return _save(fig, out)


def fig_dice_predictive_power(df, out):
    """F2: how much a Dice number tells you about each trait. 1.0 = a perfect proxy."""
    from dose_response import analyse
    d = df[df.severity != 0]
    s = analyse(df).dropna(subset=["spearman_dice_vs_error"])
    s = s.sort_values("spearman_dice_vs_error")

    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    y = np.arange(len(s))
    ax.barh(y, -s.spearman_dice_vs_error, color=[FAMILY_COLOUR[f] for f in s.family], alpha=.9)
    ax.set_yticks(y)
    ax.set_yticklabels(s.trait, fontsize=9)
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("-Spearman(Dice, trait error)    -    1.0 = Dice is a perfect proxy")
    ax.axvline(1.0, color="k", ls="--", lw=1.4)
    ax.grid(alpha=.25, axis="x")
    for yi, v in enumerate(-s.spearman_dice_vs_error):
        ax.text(v + .02, yi, f"{v:.2f}", va="center", fontsize=8)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=.9) for c in FAMILY_COLOUR.values()]
    ax.legend(handles, list(FAMILY_COLOUR), fontsize=8, frameon=False, loc="lower right",
              title="trait family (a-priori)", title_fontsize=8)
    ax.set_title("How much does Dice actually tell you about the trait you wanted?\n"
                 f"{len(d):,} degraded masks, 6 named degradations, synthetic roots", fontsize=11.5)
    return _save(fig, out)


def fig_deception(df, out):
    """Which degradation buys the most topology damage per unit of Dice?"""
    d = df[df.severity != 0].groupby("degradation").agg(
        dice=("dice", "mean"), tips=("err_n_tips", "mean"),
        branches=("err_n_branch_points", "mean"), area=("err_pixel_area", "mean"))
    d = d.sort_values("tips", ascending=False)
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    x = np.arange(len(d))
    w = 0.27
    ax.bar(x - w, 1 - d.dice, w, label="Dice lost", color="#7f8c8d")
    ax.bar(x, d.tips, w, label="tip-count error", color="#c0392b")
    ax.bar(x + w, d.area, w, label="pixel-area error", color="#2c6fbb")
    ax.set_xticks(x)
    ax.set_xticklabels(d.index, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("mean magnitude")
    ax.set_title("Uncalibrated ladder: each degradation at its OWN severity grid\n"
                 "SUPERSEDED by F11 -- at a matched Dice loss, dilation is worse",
                 fontsize=11)
    ax.grid(alpha=.25, axis="y")
    ax.legend(fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=170, bbox_inches="tight")
    print(f"  wrote {out}")
    plt.close(fig)


def fig_fractal_validation(out):
    from traits import box_counting_dimension
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    names, truths, ests = [], [], []
    for name, (fn, truth) in KNOWN.items():
        m = fn(6) if name not in ("line", "filled_square") else fn(729)
        names.append(name)
        truths.append(truth)
        ests.append(box_counting_dimension(m))
    ax.plot([0.9, 2.1], [0.9, 2.1], "k--", lw=1.5, label="exact")
    ax.scatter(truths, ests, s=70, color="#8e44ad", zorder=5)
    for n, t, e in zip(names, truths, ests):
        ax.annotate(n, (t, e), textcoords="offset points", xytext=(7, -3), fontsize=8)
    ax.set_xlabel("analytic Minkowski-Bouligand dimension")
    ax.set_ylabel("box-counting estimate")
    ax.set_title("Fractal estimator validated against known structures "
                 "(max error 0.06)", fontsize=11)
    ax.grid(alpha=.25)
    ax.legend(fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=170, bbox_inches="tight")
    print(f"  wrote {out}")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", default="results/dose_response.csv")
    ap.add_argument("--outdir", default="figures")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    fig_fractal_validation(f"{a.outdir}/f01_fractal_validation.png")
    if not os.path.exists(a.infile):
        print(f"  ({a.infile} missing; run dose_response.py first)")
        return 0
    df = pd.read_csv(a.infile)
    fig_dice_predictive_power(df, f"{a.outdir}/f02_dice_predictive_power.png")
    fig_deception(df, f"{a.outdir}/f04_deception_uncalibrated.png")
    fig_error_at_dice99(df, f"{a.outdir}/f05_error_at_dice99_uncalibrated.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
