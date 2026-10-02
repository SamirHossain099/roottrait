"""Tables 3 and 4 of the manuscript, generated from `results/` and nowhere else.

The manuscript test imports `table3()` and `table4_rows()` and asserts the text in MANUSCRIPT.md is
exactly what they produce, so a table can only change by a results file changing. `--fill` writes
them into the `{{T3_table}}` and `{{T4_rows}}` placeholders the first time.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse  # noqa: E402
import json  # noqa: E402

R = "results"
CONFIGS = ("peanut_640x480_DPI120", "peanut_736x552_DPI150")
CFG_LABEL = {"peanut_640x480_DPI120": "640 by 480", "peanut_736x552_DPI150": "736 by 552"}
ARCH = {"unetpp_r34": "UNet++", "deeplabv3p_r34": "DeepLabV3+", "unet_r34": "U-Net R34",
        "unet_r50": "U-Net R50", "fpn_r34": "FPN", "manet_r34": "MA-Net"}
# Table 1's order and names, so Tables 1 and 3 read in parallel.
TRAIT_ROWS = [("pixel_area", "Pixel area"), ("total_length", "Total length"),
              ("mean_diameter", "Mean diameter"), ("lacunarity", "Lacunarity"),
              ("convex_hull_area", "Convex-hull area"), ("bounding_depth", "Bounding depth"),
              ("bounding_width", "Bounding width"), ("n_tips", "Tips"),
              ("n_branch_points", "Branch points"), ("n_components", "Components"),
              ("n_holes", "Holes"), ("fractal_dimension", "Fractal dimension")]
KIND = {"pixel_area": "integral", "total_length": "integral", "mean_diameter": "integral",
        "lacunarity": "integral", "convex_hull_area": "extremum", "bounding_depth": "extremum",
        "bounding_width": "extremum", "n_tips": "count", "n_branch_points": "count",
        "n_components": "count", "n_holes": "count", "fractal_dimension": "fitted"}


def _load(kind, cfg, d=R):
    with open(os.path.join(d, f"prmi_{kind}_{cfg}.json"), encoding="utf-8") as fh:
        return json.load(fh)


def _f(x, nd=2):
    return f"{x:.{nd}f}".replace("-0.00", "0.00")


def table3(d=R):
    """Spearman correlation of mean Dice on root frames with median relative trait error."""
    an = {c: _load("analysis", c, d) for c in CONFIGS}
    head = ("| Trait | Kind | All models, 640 by 480 | All models, 736 by 552 "
            "| Without two weakest, 640 by 480 | Without two weakest, 736 by 552 |\n"
            "|---|---|---|---|---|---|")
    lines = [head]
    for t, name in TRAIT_ROWS:
        cells = []
        for key in ("spearman_dice_vs_trait_error", "spearman_without_two_weakest"):
            for c in CONFIGS:
                ri = an[c]["rank_inversion"].get(t)
                cells.append(_f(ri[key]) if ri else "undefined")
        lines.append(f"| {name} | {KIND[t]} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def table4_rows(d=R):
    rows = []
    for c in CONFIGS:
        fl = _load("floor", c, d)
        for a in fl["adjacent"]:
            limit = "not estimable" if a["floor_clipped"] else _f(a["floor"], 3)
            rows.append(f"| {CFG_LABEL[c]} | {ARCH[a['upper']]} over {ARCH[a['lower']]} | "
                        f"{_f(a['gap'], 3)} | {_f(a['halfwidth_now'], 3)} | {limit} |")
    return "\n".join(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fill", action="store_true", help="write into MANUSCRIPT.md placeholders")
    a = ap.parse_args()
    t4 = table4_rows()
    have3 = all(os.path.exists(os.path.join(R, f"prmi_analysis_{c}.json")) for c in CONFIGS)
    t3 = table3() if have3 else None
    print(t3 or "(Table 3 waits on results/prmi_analysis_<config>.json)")
    print(t4)
    if a.fill:
        p = "MANUSCRIPT.md"
        s = open(p, encoding="utf-8").read()
        s = s.replace("{{T4_rows}}", t4)
        if t3:
            s = s.replace("{{T3_table}}", t3)
        open(p, "w", encoding="utf-8").write(s)
    return 0


if __name__ == "__main__":
    sys.exit(main())
