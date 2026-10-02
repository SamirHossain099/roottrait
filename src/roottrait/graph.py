"""Skeleton graph helpers: spur pruning and junction counting."""
import numpy as np
from scipy import ndimage as ndi

from .traits import skeleton

_K = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
_ST = np.ones((3, 3), bool)


def _nb(sk):
    return ndi.convolve(sk.astype(np.int32), _K, mode="constant")


def prune(sk, k, max_rounds=10):
    """Remove terminal branches shorter than `k` px, repeating until none remain.

    A terminal branch is a junction-free segment that contains a tip AND touches a junction.
    A segment with tips but no junction is a whole unbranched root, not a spur, and is kept --
    pruning it would delete short roots and fake a loss in the opposite direction.
    """
    sk = np.asarray(sk, bool).copy()
    if k <= 0:
        return sk
    for _ in range(max_rounds):
        nb = _nb(sk)
        junc = sk & (nb >= 3)
        tips = sk & (nb == 1)
        if not junc.any():
            break
        seg, _ = ndi.label(sk & ~junc, structure=_ST)
        sizes = np.bincount(seg.ravel())
        tipseg = set(np.unique(seg[tips]).tolist()) - {0}
        touching = set(np.unique(seg[ndi.binary_dilation(junc, structure=_ST) & (seg > 0)])
                       .tolist())
        kill = [s for s in tipseg if s in touching and sizes[s] < k]
        if not kill:
            break
        sk &= ~np.isin(seg, kill)
        sk = skeleton(sk)
    return sk


def junction_nodes(sk):
    """Number of junctions as graph nodes: connected clusters of >=3-neighbour skeleton pixels."""
    sk = np.asarray(sk, bool)
    if not sk.any():
        return 0
    return int(ndi.label(sk & (_nb(sk) >= 3), structure=_ST)[1])
