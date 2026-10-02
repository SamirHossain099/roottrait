"""Fill small enclosed holes in root masks before computing skeleton-based traits.

Hand-drawn root annotations often contain enclosed holes of one or a few pixels. They are not root
structure, but each one splits the skeleton around it into a loop, and a loop adds junctions that
spur pruning cannot remove. On the PRMI minirhizotron annotations, filling holes of at most 10
pixels changes Dice against the original mask by at most 0.017 while changing the median skeleton
junction count several-fold.

Apply `fill_pinholes` identically to annotation and prediction before any trait or decomposition
that reads a skeleton. The default threshold, `PINHOLE_PX = 10`, is not critical: on PRMI the median
junction count is the same for every threshold from 5 to 300 pixels. Holes that touch the image
border are never filled.
"""
import numpy as np
from scipy import ndimage as ndi

PINHOLE_PX = 10


def fill_pinholes(mask, max_px=PINHOLE_PX):
    """Fill enclosed background regions of at most `max_px` pixels. Holes touching the border stay."""
    m = np.asarray(mask, bool)
    if max_px <= 0 or not m.any():
        return m.copy()
    lab, n = ndi.label(~m)
    if n == 0:
        return m.copy()
    sizes = np.bincount(lab.ravel())
    border = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    small = np.flatnonzero(sizes <= max_px)
    small = small[(small > 0) & ~np.isin(small, border)]
    return m | np.isin(lab, small)


def enclosed_holes(mask):
    """Sizes of enclosed background regions (holes not touching the image border)."""
    m = np.asarray(mask, bool)
    lab, n = ndi.label(~m)
    if n == 0:
        return []
    sizes = np.bincount(lab.ravel())
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])).tolist())
    return [int(sizes[k]) for k in range(1, n + 1) if k not in border]
