"""Root traits computed from a binary mask.

Twelve traits that root phenotyping reports, each a function of a boolean 2-D array returning a
float, so they can be swept uniformly over many masks:

  pixel_area, total_length, mean_diameter, lacunarity                      integral
  convex_hull_area, bounding_depth, bounding_width                         extremum
  n_tips, n_branch_points, n_components, n_holes                           count
  fractal_dimension (box counting)                                         fitted

The kinds (`TRAIT_KIND`) describe how a local pixel error enters each trait, which is what decides
how closely the trait follows Dice. `TRAIT_FAMILY` is the intuitive area/length/topology/scaling
grouping, kept for comparison because it does not predict sensitivity.
"""
import numpy as np
from scipy import ndimage as ndi
from skimage.measure import euler_number, label
from skimage.morphology import convex_hull_image, skeletonize

# 3x3 neighbour count kernel, centre excluded
_NEIGH = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)


def _as_bool(mask):
    a = np.asarray(mask)
    if a.dtype != bool:
        a = a > 0
    if a.ndim != 2:
        raise ValueError(f"expected a 2-D mask, got shape {a.shape}")
    return a


# ------------------------------------------------------------------ area-like
def pixel_area(mask):
    return float(_as_bool(mask).sum())


def convex_hull_area(mask):
    m = _as_bool(mask)
    if not m.any():
        return 0.0
    return float(convex_hull_image(m).sum())


def bounding_depth(mask):
    """Vertical extent of the mask (rooting depth)."""
    m = _as_bool(mask)
    if not m.any():
        return 0.0
    rows = np.where(m.any(axis=1))[0]
    return float(rows.max() - rows.min() + 1)


def bounding_width(mask):
    m = _as_bool(mask)
    if not m.any():
        return 0.0
    cols = np.where(m.any(axis=0))[0]
    return float(cols.max() - cols.min() + 1)


# ---------------------------------------------------------------- length-like
def skeleton(mask):
    return skeletonize(_as_bool(mask))


def total_length(mask):
    """Skeleton pixel count, the standard proxy for total root length."""
    return float(skeleton(mask).sum())


def mean_diameter(mask):
    """Area / length. Undefined for an empty skeleton, reported as 0."""
    ln = total_length(mask)
    return float(pixel_area(mask) / ln) if ln else 0.0


# -------------------------------------------------------------- topology-like
def _neighbour_count(skel):
    return ndi.convolve(skel.astype(np.uint8), _NEIGH, mode="constant", cval=0)


def n_tips(mask):
    """Skeleton pixels with exactly one neighbour."""
    s = skeleton(mask)
    return float((s & (_neighbour_count(s) == 1)).sum())


def n_branch_points(mask):
    """Skeleton pixels with three or more neighbours.

    Counts pixels, not graph nodes: a Y-junction can occupy 1-2 adjacent skeleton pixels, so this
    slightly over-counts. It is the measure most phenotyping code uses. `graph.junction_nodes`
    counts graph nodes instead.
    """
    s = skeleton(mask)
    return float((s & (_neighbour_count(s) >= 3)).sum())


def n_components(mask):
    return float(label(_as_bool(mask), connectivity=2).max())


def n_holes(mask):
    """Enclosed background regions, via the Euler characteristic: holes = components - euler."""
    m = _as_bool(mask)
    if not m.any():
        return 0.0
    return float(n_components(m) - euler_number(m, connectivity=2))


# -------------------------------------------------------------------- scaling
def box_counting_dimension(mask, min_box=2, max_box=None, n_scales=14, max_box_frac=4):
    """Minkowski-Bouligand dimension by box counting.

    Boxes tile the array on a grid and are counted if they contain any foreground; the slope of
    log N(eps) against log(1/eps) is the dimension.

    THE ARRAY IS CROPPED TO A MULTIPLE OF THE BOX SIZE, NOT PADDED. That choice was made by
    measurement, not taste. Padding leaves ceil(n/b) boxes per side, which at coarse scales
    materially exceeds n/b and inflates log N exactly where 1/eps is smallest -- flattening the
    slope and biasing D downwards on every structure. Cropping gives exactly floor(n/b) boxes and
    removes the effect, at the cost of discarding a sliver at two edges.

    Measured against six structures with analytic dimensions (see `validate_fractal.py`):

        strategy                       mean |error|   max |error|
        pad,  max_box = n/4                 0.0537        0.0826
        crop, max_box = n/8                 0.0629        0.1272
        crop, max_box = n/16                0.0383        0.0745
        crop, max_box = n/4  (chosen)       0.0327        0.0612

    Restricting the fit to "unsaturated" scales was also tried and rejected: it returns NaN for a
    filled square, which legitimately saturates because it genuinely is 2-dimensional.

    Residual bias is small and slightly POSITIVE (line 1.012 vs 1.000, filled square 2.024 vs
    2.000). Since the project compares trait *sensitivity* across a degradation ladder, a constant
    small offset cancels; what mattered was removing the structure-dependent bias that padding
    introduced.

    Returns NaN when fewer than three usable scales exist, rather than fitting a line through two
    points and returning something that looks like a measurement.
    """
    m = _as_bool(mask)
    if not m.any():
        return float("nan")
    n = max(m.shape)
    if max_box is None:
        max_box = max(min_box + 1, n // max_box_frac)
    sizes = np.unique(np.geomspace(min_box, max(max_box, min_box + 1), n_scales).astype(int))
    sizes = sizes[(sizes >= 2) & (sizes <= max_box)]
    counts, used = [], []
    for b in sizes:
        p = m[: m.shape[0] // b * b, : m.shape[1] // b * b]
        if p.size == 0:
            continue
        occ = p.reshape(p.shape[0] // b, b, p.shape[1] // b, b).any(axis=(1, 3))
        c = int(occ.sum())
        if c == 0:
            continue
        counts.append(c)
        used.append(b)
    if len(counts) < 3:
        return float("nan")
    slope = np.polyfit(np.log(1.0 / np.asarray(used, float)), np.log(counts), 1)[0]
    return float(slope)


def lacunarity(mask, box=None):
    """Gliding-box lacunarity: Lambda = 1 + Var(S)/Mean(S)^2 over box masses.

    1.0 means perfectly homogeneous. Higher means gappier. Standard in root phenotyping as a
    complement to fractal dimension, which cannot distinguish arrangements with the same
    space-filling but different gap structure.
    """
    m = _as_bool(mask).astype(np.float64)
    if not m.any():
        return float("nan")
    if box is None:
        box = max(3, max(m.shape) // 20)
    k = np.ones((box, box))
    s = ndi.convolve(m, k, mode="constant", cval=0.0)
    valid = s[box // 2: m.shape[0] - box // 2 or None,
              box // 2: m.shape[1] - box // 2 or None]
    mu = valid.mean()
    if mu <= 0:
        return float("nan")
    return float(1.0 + valid.var() / (mu ** 2))


# --------------------------------------------------------------------- Dice
def dice(a, b):
    a, b = _as_bool(a), _as_bool(b)
    tot = a.sum() + b.sum()
    if tot == 0:
        return 1.0
    return float(2.0 * (a & b).sum() / tot)


def iou(a, b):
    a, b = _as_bool(a), _as_bool(b)
    u = (a | b).sum()
    return 1.0 if u == 0 else float((a & b).sum() / u)


TRAITS = {
    # area-like
    "pixel_area": pixel_area,
    "convex_hull_area": convex_hull_area,
    "bounding_depth": bounding_depth,
    "bounding_width": bounding_width,
    # length-like
    "total_length": total_length,
    "mean_diameter": mean_diameter,
    # topology-like
    "n_tips": n_tips,
    "n_branch_points": n_branch_points,
    "n_components": n_components,
    "n_holes": n_holes,
    # scaling
    "fractal_dimension": box_counting_dimension,
    "lacunarity": lacunarity,
}

TRAIT_FAMILY = {
    "pixel_area": "area", "convex_hull_area": "area",
    "bounding_depth": "area", "bounding_width": "area",
    "total_length": "length", "mean_diameter": "length",
    "n_tips": "topology", "n_branch_points": "topology",
    "n_components": "topology", "n_holes": "topology",
    "fractal_dimension": "scaling", "lacunarity": "scaling",
}


# The intuitive taxonomy above (TRAIT_FAMILY) fails for half of the traits when measured on
# calibrated degradations: `convex_hull_area` and `bounding_depth` are "area-like" by any ordinary reading and are among
# the MOST fragile traits measured, because a single false-positive blob far from the root moves
# the hull and the bounding box enormously while barely touching Dice.
#
# The taxonomy the DATA gives is integrals versus extrema-and-counts:
#
#   integral   averaged or summed over the whole mask, so a local error contributes in proportion
#              to its size. Robust. Includes lacunarity -- a variance-to-mean ratio over many
#              gliding boxes is an average, which is why a "scaling" trait lands with pixel area.
#   extremum   determined by the most distant foreground pixel. One stray blob relocates it.
#   count      a tally of discrete features. Fragmenting one structure multiplies it, and the
#              baseline is small, so the relative error is unbounded.
#   fitted     a slope estimated across scales. Intermediate: averaging over scales gives partial
#              robustness, but the fit is levered by the extreme scales.
#
# Both dictionaries are kept so the two groupings can be compared.
TRAIT_KIND = {
    "pixel_area": "integral",
    "total_length": "integral",
    "mean_diameter": "integral",
    "lacunarity": "integral",
    "convex_hull_area": "extremum",
    "bounding_depth": "extremum",
    "bounding_width": "extremum",
    "n_tips": "count",
    "n_branch_points": "count",
    "n_components": "count",
    "n_holes": "count",
    "fractal_dimension": "fitted",
}

assert set(TRAIT_KIND) == set(TRAITS), "TRAIT_KIND and TRAITS have drifted apart"
assert set(TRAIT_FAMILY) == set(TRAITS), "TRAIT_FAMILY and TRAITS have drifted apart"


def all_traits(mask):
    return {k: fn(mask) for k, fn in TRAITS.items()}
