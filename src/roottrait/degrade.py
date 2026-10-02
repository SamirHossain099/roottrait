"""Six named segmentation failures, each with one severity parameter.

  boundary_dilate   the whole boundary pushed outward (a threshold set too low)
  boundary_erode    the whole boundary pulled inward (a threshold set too high)
  thin_dropout      whole branches deleted, thinnest first (fine laterals missed)
  junction_break    small disks removed at skeleton branch points (a connection lost at a fork)
  speckle           isolated false-positive pixels
  blob              compact false-positive regions away from the root

`DEGRADATIONS` maps each name to `(function, default severity grid)`. Every function takes
`(mask, severity, rng)` and returns a new boolean mask. Dilation and erosion apply the fractional
part of the severity to a random subset of the pixels the next iteration would change, and
thin-branch dropout removes a fraction of branches ranked by thickness, so severity is continuous
and can be calibrated to an exact Dice loss (see `calibrate`).

`synth_root` draws a simple recursively branching root for quick tests; `morphology` provides the
four architectures used for systematic work.
"""
import numpy as np
from scipy import ndimage as ndi

from .traits import _neighbour_count, skeleton


# --------------------------------------------------------- synthetic roots
def synth_root(size=512, seed=0, n_primary=3, depth=4, thickness=5):
    """A branching root-like structure with a known construction.

    Recursive: each segment spawns 1-2 children with a narrowing angle and shrinking length and
    thickness, which is the qualitative form of a real root system (thick primaries, thin, numerous
    laterals). Drawn by stamping disks along each segment so thickness is controllable.
    """
    rng = np.random.default_rng(seed)
    m = np.zeros((size, size), bool)
    yy, xx = np.mgrid[0:size, 0:size]

    def stamp(y, x, r):
        if 0 <= y < size and 0 <= x < size:
            m[(yy - y) ** 2 + (xx - x) ** 2 <= r * r] = True

    def draw(y, x, ang, length, thick, d):
        steps = max(int(length), 1)
        for t in range(steps):
            ny = y + t * np.cos(ang)
            nx = x + t * np.sin(ang)
            stamp(int(round(ny)), int(round(nx)), max(1, int(round(thick))))
        ey = y + steps * np.cos(ang)
        ex = x + steps * np.sin(ang)
        if d <= 0:
            return
        for _ in range(rng.integers(1, 3)):
            draw(ey, ex, ang + rng.uniform(-0.7, 0.7), length * rng.uniform(0.5, 0.8),
                 thick * 0.65, d - 1)

    for i in range(n_primary):
        x0 = size * (i + 1) / (n_primary + 1)
        draw(2, x0, rng.uniform(-0.25, 0.25), size / (depth + 1), thickness, depth)
    return m


# ------------------------------------------------------------- degradations
def boundary(mask, severity, rng=None):
    """severity > 0 dilates, < 0 erodes; |severity| is the structuring-element radius.

    Severity is CONTINUOUS, not integer. The integer part applies that many full morphological
    iterations; the fractional part applies the next iteration to a random subset of the pixels it
    would have changed.

    That matters for the experiment, not just for tidiness. With integer-only radii the mildest
    possible boundary error is a full one-pixel band, which costs ~0.045-0.05 Dice on these masks --
    so boundary degradations could never be compared against `junction_break` or `speckle` at a
    matched Dice loss below 0.045, and the matched-dose comparison silently dropped four of six
    degradations as "not comparable". A partial band is also the more faithful model: a real
    segmenter's boundary error is ragged, not a uniform annulus.
    """
    if severity == 0:
        return mask.copy()
    rng = rng if rng is not None else np.random.default_rng(0)
    mag = abs(float(severity))
    whole, frac = int(mag), mag - int(mag)
    st = ndi.generate_binary_structure(2, 1)
    fn = ndi.binary_dilation if severity > 0 else ndi.binary_erosion

    out = fn(mask, structure=st, iterations=whole) if whole else mask.copy()
    if frac > 1e-9:
        nxt = fn(out, structure=st, iterations=1)
        changed = np.flatnonzero((nxt ^ out).ravel())
        if changed.size:
            k = int(round(frac * changed.size))
            if k > 0:
                pick = rng.choice(changed, size=min(k, changed.size), replace=False)
                flat = out.ravel().copy()
                flat[pick] = bool(severity > 0)
                out = flat.reshape(mask.shape)
    return out


def junction_break(mask, severity, rng=None):
    """Punch small disks at skeleton branch points.

    `severity` is the fraction of branch points to hit (0-1). Radius is fixed small (3 px) so the
    Dice cost stays tiny while the graph changes; the graph changes while the overlap barely does.
    """
    rng = rng or np.random.default_rng(0)
    if severity <= 0:
        return mask.copy()
    s = skeleton(mask)
    bp = np.argwhere(s & (_neighbour_count(s) >= 3))
    if len(bp) == 0:
        return mask.copy()
    k = max(1, int(round(severity * len(bp))))
    pick = bp[rng.choice(len(bp), size=min(k, len(bp)), replace=False)]
    out = mask.copy()
    r = 3
    yy, xx = np.mgrid[0:mask.shape[0], 0:mask.shape[1]]
    for y, x in pick:
        out[(yy - y) ** 2 + (xx - x) ** 2 <= r * r] = False
    return out


def thin_dropout(mask, severity, rng=None):
    """Delete whole thin BRANCHES, thinnest first. `severity` is the fraction of branches removed.

    This models a segmenter losing fine lateral roots entirely -- a different failure from a
    threshold set slightly too high, which is `boundary_erode`.

    Two earlier versions were wrong, in opposite directions, and both are worth recording because
    each looked reasonable:

      v1  Thresholded the distance transform at a severity percentile. On a root a few pixels wide
          the EDT takes only a handful of distinct values, so Dice loss was a STEP function and the
          degradation could not be calibrated to a dose at all: bisection for a 0.005 target
          returned 0.20.

      v2  Removed the globally thinnest `severity` fraction of pixels. Continuous, but it silently
          became a duplicate of erosion -- measured at four matched Dice losses, the surviving
          maximum local half-width was 7.81/7.21/7.07/6.40 against erosion's 7.81/7.21/7.07/6.32.
          The thinnest pixels in a mask are the boundary pixels of EVERY structure, thick ones
          included, so ranking pixels by thickness peels layers just as erosion does. A degradation
          that is a copy of another degradation adds a column to the table and no information.

    v3, here: rank SKELETON BRANCHES by mean local thickness, take the thinnest `severity` fraction,
    and remove each selected branch at its full width by deleting every mask pixel whose nearest
    skeleton pixel belongs to a selected branch. Continuous in the branch fraction, and it leaves
    the thick primary axis untouched, which is what makes it a distinct failure mode.
    """
    if severity <= 0:
        return mask.copy()
    rng = rng if rng is not None else np.random.default_rng(0)
    skel = skeleton(mask)
    if not skel.any():
        return mask.copy()

    # Branches = skeleton minus its junctions, so each piece is a single unbranched segment.
    junc = skel & (_neighbour_count(skel) >= 3)
    segs, n_seg = ndi.label(skel & ~junc, structure=np.ones((3, 3), np.uint8))
    if n_seg == 0:
        return mask.copy()

    dt = ndi.distance_transform_edt(mask)
    labels = np.arange(1, n_seg + 1)
    thickness = ndi.mean(dt, labels=segs, index=labels)
    # Jitter breaks ties between equally thin branches at random rather than by label order.
    order = np.lexsort((rng.random(n_seg), np.asarray(thickness, float)))
    k = int(round(min(max(severity, 0.0), 1.0) * n_seg))
    if k <= 0:
        return mask.copy()
    doomed = set(labels[order[:k]].tolist())

    # Assign every mask pixel to its nearest skeleton pixel, then delete the pixels owned by a
    # doomed branch. This removes each branch at its full width instead of a fixed radius.
    _, (iy, ix) = ndi.distance_transform_edt(~skel, return_indices=True)
    owner = segs[iy, ix]
    return mask & ~np.isin(owner, list(doomed))


def speckle(mask, severity, rng=None):
    """Random isolated false-positive pixels; `severity` is the fraction of background flipped."""
    rng = rng or np.random.default_rng(0)
    if severity <= 0:
        return mask.copy()
    out = mask.copy()
    bg = ~mask
    n = int(round(severity * bg.sum()))
    if n == 0:
        return out
    idx = rng.choice(np.flatnonzero(bg.ravel()), size=min(n, bg.sum()), replace=False)
    flat = out.ravel()
    flat[idx] = True
    return flat.reshape(mask.shape)


def blob(mask, severity, rng=None):
    """Compact false-positive regions. `severity` scales the number of blobs."""
    rng = rng or np.random.default_rng(0)
    if severity <= 0:
        return mask.copy()
    out = mask.copy()
    n = max(1, int(round(severity * 40)))
    yy, xx = np.mgrid[0:mask.shape[0], 0:mask.shape[1]]
    for _ in range(n):
        cy, cx = rng.integers(0, mask.shape[0]), rng.integers(0, mask.shape[1])
        r = rng.integers(3, 9)
        out[(yy - cy) ** 2 + (xx - cx) ** 2 <= r * r] = True
    return out


DEGRADATIONS = {
    "boundary_erode": (boundary, [0, -1, -2, -3, -4, -5]),
    "boundary_dilate": (boundary, [0, 1, 2, 3, 4, 5]),
    "junction_break": (junction_break, [0, .1, .2, .4, .6, .8]),
    "thin_dropout": (thin_dropout, [0, .05, .1, .2, .3, .4]),
    "speckle": (speckle, [0, .0002, .0005, .001, .002, .005]),
    "blob": (blob, [0, .1, .25, .5, .75, 1.0]),
}
