"""Figures for the calibrated experiment (F9-F13).

ONE FIGURE PER FILE. Nothing here emits a multi-panel composite.

A panel that only exists inside a composite cannot be cited: a manuscript that wants to point at
"the control" has to say "the right-hand panel of Figure 3", captions have to cover two arguments
at once, and a reviewer asking for one panel to change forces a re-render of both. Each figure here
is therefore standalone -- its own title, its own legend, its own axis labels, readable with no
neighbour. Place them side by side in the document if that reads better; that is a layout decision
for the manuscript, not a constraint the plotting code should bake in.

Emitted:
  f09_kind_ordering_by_architecture.png   the taxonomy replicates in all four architectures
  f10_architecture_vs_thickness.png       the confound, both margins reported
  f12a_error_span_at_fixed_dice.png       THE figure: the spread at one fixed Dice
  f12b_control_vs_worst_case.png          pixel_area against n_components, the internal control
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

from traits import TRAIT_KIND, TRAITS  # noqa: E402

KIND_COLOUR = {"integral": "#2e8b57", "fitted": "#8e44ad",
               "extremum": "#2c6fbb", "count": "#c0392b"}
KIND_ORDER = ["integral", "fitted", "extremum", "count"]
DEG_LABEL = {"boundary_dilate": "dilate", "boundary_erode": "erode",
             "junction_break": "junction", "thin_dropout": "thin-drop",
             "speckle": "speckle", "blob": "blob"}
FLOOR = 1e-4
# Table 1's trait names, so the figure and the manuscript use the same words.
TRAIT_NAME = {"pixel_area": "Pixel area", "total_length": "Total length",
              "mean_diameter": "Mean diameter", "lacunarity": "Lacunarity",
              "convex_hull_area": "Convex-hull area", "bounding_depth": "Bounding depth",
              "bounding_width": "Bounding width", "n_tips": "Tips", "n_branch_points": "Branch points",
              "n_components": "Components", "n_holes": "Holes",
              "fractal_dimension": "Fractal dimension"}
DOSE = 0.01          # the dose every figure below is read at: Dice = 0.99


def _save(fig, out):
    fig.savefig(out, dpi=600, bbox_inches="tight")
    fig.savefig(os.path.splitext(out)[0] + ".pdf", bbox_inches="tight")  # vector copy (section 7)
    plt.close(fig)
    print("  wrote " + out)
    return out


def _median_table(cal, dose=DOSE):
    """Median relative error per (trait, degradation) at one calibrated dose."""
    at = cal[(cal.reachable) & (cal.target_dice_loss == dose)]
    med = at.groupby("degradation")[["err_" + t for t in TRAITS]].median().T
    med.index = [i[4:] for i in med.index]
    med["kind"] = [TRAIT_KIND[t] for t in med.index]
    degs = [d for d in DEG_LABEL if d in med.columns]
    med["lo"] = med[degs].min(axis=1)
    med["hi"] = med[degs].max(axis=1)
    med["kord"] = [KIND_ORDER.index(k) for k in med.kind]
    return med.sort_values(["kord", "hi"]), degs, len(at)


# --------------------------------------------------------------------------- F12a
def fig_error_span(cal, out):
    """F12: at ONE fixed Dice, how far apart the failure modes put each trait."""
    med, degs, n = _median_table(cal)

    fig, ax = plt.subplots(figsize=(9.6, 6.2))
    y = np.arange(len(med))
    for yi, (_, r) in zip(y, med.iterrows()):
        c = KIND_COLOUR[r.kind]
        ax.plot([max(r.lo, FLOOR), max(r.hi, FLOOR)], [yi, yi], color=c, lw=3.4, alpha=.45,
                solid_capstyle="round", zorder=1)
        for d in degs:
            ax.plot(max(r[d], FLOOR), yi, "o", ms=5.4, color=c, mec="white", mew=.7, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([TRAIT_NAME[t] for t in med.index], fontsize=9.5)
    ax.set_ylim(-0.6, len(med) + 0.4)
    ax.set_xscale("log")
    ax.set_xlim(FLOOR * 0.7, 6e3)
    ax.axvline(DOSE, color="k", ls=":", lw=1.2)
    # Labels sit in a header row above the data, where they cannot cover a trait's dots.
    ax.text(DOSE, len(med) - 0.15, "Dice loss (0.01)", fontsize=8, va="center", ha="center",
            bbox=dict(fc="white", ec="none", pad=1))
    ax.set_xlabel("median relative trait error   (each dot is one failure mode)")

    # Span labels in a reserved column. An earlier version printed "unbounded" for seven of twelve
    # traits -- correct arithmetic, since the minimum is exactly 0, and useless as a description:
    # it gave bounding_depth, whose worst case is 9.5%, the same label as n_components, whose worst
    # case is 15,500%.
    ax.axvline(500, color="#ccc", lw=.8)
    for yi, (_, r) in zip(y, med.iterrows()):
        if r.lo > 1e-9:
            span = f"x{r.hi / r.lo:.0f}"
        else:
            span = f"0 to {r.hi:.0%}" if r.hi < 5 else f"0 to {r.hi:.0f}x"
        ax.text(700, yi, span, fontsize=8.2, ha="left", va="center", color="#333")
    ax.text(700, len(med) - 0.15, "range across\nfailure modes", fontsize=7.8, ha="left",
            va="center", color="#777", style="italic")

    handles = [plt.Line2D([], [], color=KIND_COLOUR[k], lw=3.4, label=k) for k in KIND_ORDER]
    ax.legend(handles=handles, fontsize=8.5, loc="lower right", bbox_to_anchor=(0.85, 0.01),
              framealpha=.95, title="trait kind", title_fontsize=8.5)
    ax.grid(alpha=.22, axis="x", which="both")
    return _save(fig, out)


# --------------------------------------------------------------------------- F12b
def fig_control_vs_worst(cal, out):
    """F12: the internal positive control. One trait does not span; the other does."""
    med, degs, n = _median_table(cal)
    ctrl, worst = "pixel_area", "n_components"

    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    w = 0.36
    x = np.arange(len(degs))
    cv = [med.loc[ctrl, d] for d in degs]
    wv = [med.loc[worst, d] for d in degs]
    ax.bar(x - w / 2, [max(v, FLOOR) for v in cv], w,
           color=KIND_COLOUR["integral"], label=ctrl + "   (integral)")
    ax.bar(x + w / 2, [max(v, FLOOR) for v in wv], w,
           color=KIND_COLOUR["count"], label=worst + "   (count)")
    # A bar drawn at the log floor looks like a small non-zero value. Where the measurement is
    # exactly zero, say so, or the figure claims an error that was never measured.
    for xi, v in zip(x, wv):
        if v <= 1e-9:
            ax.text(xi + w / 2, FLOOR * 1.35, "exactly\nzero", fontsize=7.4, ha="center",
                    va="bottom", color="#7a1f14")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([DEG_LABEL[d] for d in degs], fontsize=9)
    ax.set_ylabel("median relative trait error")
    ax.set_xlabel("failure mode, each calibrated to the same Dice = 0.99")
    ax.set_ylim(FLOOR * 0.7, 900)
    ax.legend(fontsize=9, loc="upper left", framealpha=.95)
    ax.grid(alpha=.22, axis="y", which="both")
    lo, hi = med.loc[ctrl, "lo"], med.loc[ctrl, "hi"]
    ax.text(0.98, 0.06, f"pixel area spans only x{hi / lo:.1f}\nacross all six failure modes",
            transform=ax.transAxes, fontsize=8.5, ha="right",
            bbox=dict(boxstyle="round,pad=0.4", fc="#f4f4f4", ec="#bbb"))
    ax.set_title("The control and the worst case: same masks, same Dice.\n"
                 "One is flat across failure modes, the other spans four orders of magnitude.",
                 fontsize=11.5)
    return _save(fig, out)


# --------------------------------------------------------------------------- F9
def fig_kind_ordering(morph_csv, out):
    """F9: the kind-level taxonomy replicates in every architecture."""
    by_m = pd.read_csv(morph_csv.replace(".csv", "_by_morphology.csv"))
    k = by_m.groupby(["morphology", "kind"]).sensitivity.median().unstack()
    k = k[[c for c in KIND_ORDER if c in k.columns]]

    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    x = np.arange(len(k.index))
    w = 0.2
    for i, kind in enumerate(k.columns):
        ax.bar(x + (i - (len(k.columns) - 1) / 2) * w, k[kind], w,
               color=KIND_COLOUR[kind], label=kind)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(list(k.index), fontsize=9.5)
    ax.set_xlabel("synthetic root architecture")
    ax.set_ylabel("median trait error per unit of Dice lost")
    ax.legend(fontsize=9, ncol=2, framealpha=.95, title="trait kind", title_fontsize=9)
    ax.grid(alpha=.22, axis="y", which="both")
    ax.set_title("The kind-level ordering holds in all four architectures\n"
                 "(integrals lowest everywhere; the trait-by-trait ranking does not transfer)",
                 fontsize=11.5)
    return _save(fig, out)


# --------------------------------------------------------------------------- F10
def fig_confound(morph_csv, out):
    """F10: architecture versus thickness, both margins, because they are confounded."""
    by_m = pd.read_csv(morph_csv.replace(".csv", "_by_morphology.csv"))
    by_t = pd.read_csv(morph_csv.replace(".csv", "_by_thickness.csv"))

    sp = []
    for t in TRAITS:
        lm = np.log10([v for v in by_m[by_m.trait == t].sensitivity if v and v > 0])
        lt = np.log10([v for v in by_t[by_t.trait == t].sensitivity if v and v > 0])
        if len(lm) and len(lt):
            sp.append((t, TRAIT_KIND[t], lm.max() - lm.min(), lt.max() - lt.min()))
    sp = pd.DataFrame(sp, columns=["trait", "kind", "morph", "thick"])

    fig, ax = plt.subplots(figsize=(8.4, 6.6))
    lim = max(sp.morph.max(), sp.thick.max()) * 1.14
    ax.fill_between([0, lim], [0, lim], [lim, lim], color="#c0392b", alpha=.055)
    ax.fill_between([0, lim], [0, 0], [0, lim], color="#2c6fbb", alpha=.055)
    ax.plot([0, lim], [0, lim], "k--", lw=1.1, alpha=.6)
    ax.text(lim * .62, lim * .70, "equal influence", fontsize=8.5, rotation=38, color="#555")
    for _, r in sp.iterrows():
        ax.plot(r.thick, r.morph, "o", ms=8, color=KIND_COLOUR[r.kind], mec="white", mew=.8,
                zorder=3)
        ax.annotate(r.trait, (r.thick, r.morph), fontsize=7.4, xytext=(6, 3),
                    textcoords="offset points", color="#333", zorder=4)
    ax.text(lim * .06, lim * .93, "architecture matters more", fontsize=9.5, color="#c0392b")
    ax.text(lim * .50, lim * .05, "thickness matters more", fontsize=9.5, color="#2c6fbb")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("spread attributable to THICKNESS   (dex of log10 sensitivity)")
    ax.set_ylabel("spread attributable to ARCHITECTURE   (dex)")
    handles = [plt.Line2D([], [], marker="o", ls="", color=KIND_COLOUR[k], label=k)
               for k in KIND_ORDER]
    ax.legend(handles=handles, fontsize=8.5, loc="center right", framealpha=.95,
              title="trait kind", title_fontsize=8.5)
    ax.grid(alpha=.22)
    ax.set_title("Architecture and thickness are confounded by construction,\n"
                 "so both margins are reported rather than one being assumed",
                 fontsize=11.5)
    return _save(fig, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrated", default="results/calibrated_sweep.csv")
    ap.add_argument("--morphology", default="results/morphology_sweep.csv")
    ap.add_argument("--outdir", default="figures")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    cal = pd.read_csv(a.calibrated)
    o = a.outdir
    fig_kind_ordering(a.morphology, os.path.join(o, "f09_kind_ordering_by_architecture.png"))
    fig_confound(a.morphology, os.path.join(o, "f10_architecture_vs_thickness.png"))
    fig_error_span(cal, os.path.join(o, "f12a_error_span_at_fixed_dice.png"))
    fig_control_vs_worst(cal, os.path.join(o, "f12b_control_vs_worst_case.png"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
