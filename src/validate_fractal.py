"""Validate the fractal/lacunarity code on structures with KNOWN dimension.

The brief calls this the technical long pole, and it is the one step that must happen before any
root mask is touched. Box-counting dimension is easy to implement and easy to get subtly wrong:
partial boxes at the array edge, too narrow a scale range, or fitting a slope through two points
all produce numbers that look like measurements.

So it is checked against structures whose Minkowski-Bouligand dimension is known analytically:

    straight line          D = 1
    filled square          D = 2
    Sierpinski triangle    D = log 3 / log 2  = 1.5850
    Sierpinski carpet      D = log 8 / log 3  = 1.8928
    Cantor dust (2-D)      D = log 4 / log 3  = 1.2619
    Vicsek fractal         D = log 5 / log 3  = 1.4650

If the estimator cannot recover these, no number it reports on a root is worth anything.

A caveat stated up front rather than discovered later: box counting on a *finite raster* is
biased. A structure rendered at 729 px has only ~6 usable octaves, and the estimate approaches the
analytic value slowly. The test therefore asserts a tolerance that reflects measured behaviour, and
this script prints the error at several resolutions so the convergence is visible rather than
asserted.
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

from traits import box_counting_dimension, lacunarity  # noqa: E402


# ----------------------------------------------------------------- generators
def line(n=512):
    m = np.zeros((n, n), bool)
    m[n // 2, :] = True
    return m


def filled_square(n=512):
    return np.ones((n, n), bool)


def sierpinski_triangle(order=8):
    """Chaos-game-free construction: pixel (r, c) is on iff (r & c) == 0 in the Pascal sense."""
    n = 2 ** order
    r = np.arange(n)[:, None]
    c = np.arange(n)[None, :]
    return ((r & c) == 0) & (c <= r)


def _menger_like(order, keep):
    """Generic 3x3 subdivision fractal; `keep` is a 3x3 boolean of retained sub-blocks."""
    m = np.ones((1, 1), bool)
    k = np.asarray(keep, bool)
    for _ in range(order):
        m = np.kron(m, k)
    return m


def sierpinski_carpet(order=6):
    return _menger_like(order, [[1, 1, 1], [1, 0, 1], [1, 1, 1]])


def cantor_dust(order=6):
    return _menger_like(order, [[1, 0, 1], [0, 0, 0], [1, 0, 1]])


def vicsek(order=6):
    return _menger_like(order, [[0, 1, 0], [1, 1, 1], [0, 1, 0]])


KNOWN = {
    "line": (line, 1.0),
    "filled_square": (filled_square, 2.0),
    "sierpinski_triangle": (sierpinski_triangle, np.log(3) / np.log(2)),
    "sierpinski_carpet": (sierpinski_carpet, np.log(8) / np.log(3)),
    "cantor_dust": (cantor_dust, np.log(4) / np.log(3)),
    "vicsek": (vicsek, np.log(5) / np.log(3)),
}


def evaluate(name, orders=(4, 5, 6), n_scales=14):
    """Estimate D at several resolutions so convergence is visible, not assumed."""
    fn, truth = KNOWN[name]
    out = []
    for o in orders:
        m = fn(o) if name not in ("line", "filled_square") else fn(3 ** o)
        est = box_counting_dimension(m, n_scales=n_scales)
        out.append(dict(order=o, size=m.shape[0], estimate=est,
                        error=est - truth, abs_error=abs(est - truth)))
    return truth, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-scales", type=int, default=14)
    ap.add_argument("--out", default="results/fractal_validation.json")
    a = ap.parse_args()

    print("=== box-counting dimension vs analytic truth ===")
    print(f"  {'structure':22s} {'truth':>7s} {'size':>6s} {'estimate':>9s} {'error':>8s}")
    rows = {}
    for name in KNOWN:
        truth, res = evaluate(name, n_scales=a.n_scales)
        rows[name] = dict(truth=truth, runs=res)
        for r in res:
            print(f"  {name:22s} {truth:7.4f} {r['size']:6d} {r['estimate']:9.4f} "
                  f"{r['error']:+8.4f}")
        print()

    finest = {k: v["runs"][-1] for k, v in rows.items()}
    worst = max(finest.items(), key=lambda kv: kv[1]["abs_error"])
    print(f"  worst absolute error at the finest resolution: {worst[0]} "
          f"{worst[1]['abs_error']:.4f}")

    print("\n=== does the estimate improve with resolution? ===")
    for name, v in rows.items():
        errs = [r["abs_error"] for r in v["runs"]]
        trend = "improves" if errs[-1] < errs[0] else "does NOT improve"
        print(f"  {name:22s} {errs[0]:.4f} -> {errs[-1]:.4f}   {trend}")

    print("\n=== lacunarity sanity: homogeneous < structured < sparse ===")
    lac = {
        "filled_square": lacunarity(filled_square(243)),
        "vicsek": lacunarity(vicsek(5)),
        "sierpinski_carpet": lacunarity(sierpinski_carpet(5)),
        "cantor_dust": lacunarity(cantor_dust(5)),
    }
    for k, v in lac.items():
        print(f"  {k:22s} {v:8.4f}")
    print("  (a fully filled mask must give exactly 1.0: zero variance in box mass)")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump({"dimension": rows, "lacunarity": lac}, fh, indent=2)
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
