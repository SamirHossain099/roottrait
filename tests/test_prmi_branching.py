"""Skeleton pruning and junction counting, on constructed shapes with known answers."""
import glob
import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from prmi_branching import junction_nodes, lost_structure_share, prune  # noqa: E402
from traits import skeleton  # noqa: E402

RESULTS = sorted(glob.glob(os.path.join(ROOT, "results", "prmi_branching_*.json")))


def _cross():
    """A plus sign: one junction, four long arms."""
    m = np.zeros((61, 61), bool)
    m[30, 5:56] = True
    m[5:56, 30] = True
    return m


def _cross_with_spur():
    """The plus sign plus a 3 px spur off one arm -- the annotation-texture artefact."""
    m = _cross()
    m[27:30, 15] = True
    return m


def test_a_cross_has_one_junction_node_not_several_pixels():
    """`traits.n_branch_points` counts pixels; graph nodes are clusters. One crossing = one node."""
    assert junction_nodes(skeleton(_cross())) == 1


def test_a_short_spur_adds_a_junction_and_pruning_removes_it():
    sk = skeleton(_cross_with_spur())
    assert junction_nodes(sk) == 2
    assert junction_nodes(prune(sk, 6)) == 1


def test_pruning_keeps_long_arms():
    sk = skeleton(_cross())
    assert junction_nodes(prune(sk, 15)) == 1
    assert prune(sk, 15).sum() > 0.8 * sk.sum()


def test_pruning_never_deletes_an_isolated_short_root():
    """A short root with no junction is a whole root, not a spur. Deleting it would fake a loss."""
    m = np.zeros((40, 40), bool)
    m[10, 5:12] = True                          # 7 px root, no branches
    sk = skeleton(m)
    assert prune(sk, 15).sum() == sk.sum()


def test_prune_k0_is_identity():
    sk = skeleton(_cross_with_spur())
    assert np.array_equal(prune(sk, 0), sk)


def test_lost_structure_share_is_one_when_a_branch_is_cut_through():
    g = np.zeros((40, 40), bool)
    g[18:23, 2:38] = True                       # thick horizontal root
    p = g.copy()
    p[:, 18:22] = False                         # cut clean through it
    assert lost_structure_share([g], [p], 0.0) == 1.0


def test_lost_structure_share_is_zero_for_pure_boundary_shaving():
    g = np.zeros((40, 40), bool)
    g[15:26, 2:38] = True                       # half-width 5
    p = g.copy()
    p[15, :] = False                            # shave one boundary row
    assert lost_structure_share([g], [p], 0.0) == 0.0


@pytest.mark.skipif(not RESULTS, reason="run src/prmi_branching.py first")
@pytest.mark.parametrize("path", RESULTS, ids=[os.path.basename(p) for p in RESULTS])
def test_branching_results_are_internally_consistent(path):
    with open(path) as fh:
        r = json.load(fh)
    for tag, m in r["models"].items():
        prev_gt = None
        for k in map(str, r["prune_k"]):
            p = m["prune"][k]
            assert 0.0 <= p["share_pred_fewer"] + p["share_pred_more"] <= 1.0, tag
            # pruning only ever removes junctions from the ground truth
            if prev_gt is not None:
                assert p["gt_median_junctions"] <= prev_gt + 1e-9, (tag, k)
            prev_gt = p["gt_median_junctions"]
        for d in map(str, r["centreline_edt"]):
            v = m["centreline"][d]
            assert v is None or 0.0 <= v <= 1.0
    s = r["summary"]
    assert s["signed_err_unpruned"]["min"] <= s["signed_err_unpruned"]["max"]
