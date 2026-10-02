"""Decompose a prediction's errors into four kinds, directly from the pixels.

Every wrong pixel is one of:

  over_inclusive   false positive in a connected component that touches the true root
                   (the boundary pushed outward, as dilation does)
  false_positive   false positive in a component detached from the true root
                   (spurious foreground, as speckle and blobs produce)
  under_inclusive  false negative in a component that does not reach the root's skeleton
                   (the boundary pulled inward with structure intact, as erosion does)
  lost_structure   false negative in a component that reaches the skeleton
                   (a branch or connection removed, as junction breaks and branch dropout do)

The skeleton test separates the last two: erosion by less than a root's half-width never touches
the centreline, while removing a branch or cutting a junction does. Because the four shares are
counts, mixtures of failures are represented directly, which matching a prediction to the nearest
single failure cannot do.
"""
import numpy as np
from scipy import ndimage as ndi

from .traits import skeleton

KINDS = ("over_inclusive", "false_positive", "under_inclusive", "lost_structure")
_ST = np.ones((3, 3), bool)


def decompose(gt, pred):
    """Pixel counts of each error kind for one (ground truth, prediction) pair."""
    gt = np.asarray(gt, bool)
    pred = np.asarray(pred, bool)
    out = dict.fromkeys(KINDS, 0)

    fp = pred & ~gt
    if fp.any():
        lab, n = ndi.label(fp, structure=_ST)
        # A false-positive component is "attached" if it is 8-adjacent to the true root.
        touch = ndi.binary_dilation(gt, structure=_ST)
        attached_ids = np.unique(lab[touch & fp])
        attached_ids = attached_ids[attached_ids > 0]
        attached = np.isin(lab, attached_ids)
        out["over_inclusive"] = int((fp & attached).sum())
        out["false_positive"] = int((fp & ~attached).sum())

    fn = gt & ~pred
    if fn.any():
        sk = skeleton(gt)
        lab, n = ndi.label(fn, structure=_ST)
        on_centreline = np.unique(lab[sk & fn])
        on_centreline = on_centreline[on_centreline > 0]
        structural = np.isin(lab, on_centreline)
        out["lost_structure"] = int((fn & structural).sum())
        out["under_inclusive"] = int((fn & ~structural).sum())
    return out


def shares(counts):
    tot = sum(counts.values())
    if tot == 0:
        return dict.fromkeys(KINDS, 0.0)
    return {k: counts[k] / tot for k in KINDS}


def decompose_set(gts, preds):
    """Aggregate over a set of frames. Pixel-weighted and frame-weighted shares are both returned.

    Pixel-weighted is dominated by a few frames with large errors; frame-weighted gives each frame
    one vote. They answer different questions ("where is the error mass?" against "what does a
    typical frame get wrong?"), and if they disagree the disagreement is worth reporting.
    """
    tot = dict.fromkeys(KINDS, 0)
    per_frame = []
    for g, p in zip(gts, preds):
        c = decompose(g, p)
        for k in KINDS:
            tot[k] += c[k]
        if sum(c.values()) > 0:
            per_frame.append(shares(c))
    frame_w = {k: float(np.mean([f[k] for f in per_frame])) if per_frame else 0.0
               for k in KINDS}
    return {
        "pixel_weighted": shares(tot),
        "frame_weighted": frame_w,
        "error_pixels": int(sum(tot.values())),
        "n_frames_with_error": len(per_frame),
        "n_frames": len(gts),
    }


def dominant(share_dict):
    return max(share_dict, key=share_dict.get)
