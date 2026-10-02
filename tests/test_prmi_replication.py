"""Pins the cross-configuration replication, reading `results/` only.

These assert relationships, not literals (standing rule 4), and each corresponds to one sentence the
paper makes. If a re-run changes the data enough to break one, the sentence is wrong and must change.
"""
import json
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REP = os.path.join(ROOT, "results", "prmi_replication.json")

pytestmark = pytest.mark.skipif(not os.path.exists(REP), reason="run src/prmi_replication.py")


@pytest.fixture(scope="module")
def r():
    with open(REP) as fh:
        return json.load(fh)


def test_empty_frames_inflate_the_headline_in_both_configurations(r):
    """'All-frames Dice exceeds roots-only Dice for every model, in both configurations.'"""
    for c in ("c1", "c2"):
        assert r[c]["inflation_all_minus_roots"]["min"] > 0, c


def test_every_model_beats_predicting_nothing_but_not_always_by_much(r):
    for c in ("c1", "c2"):
        assert r[c]["worst_model_margin_over_null"] > 0, c


def test_branching_gap_is_annotation_texture_not_a_headline(r):
    """C12: F19's 'annotation keeps 7 / 3 junctions, prediction 0' was mostly one-pixel holes.

    After filling holes of at most 10 px in both masks, the annotation's median junction count after
    pruning is at most 1 in both configurations, so 'models lose branching' is no longer a claim the
    paper can make. If a re-run brings the annotation's junctions back, F19 must be re-examined.
    Predictions still have fewer junctions than the annotation in most frames.
    """
    for c in ("c1", "c2"):
        b = r[c]["branching"]
        assert b["gt_junctions_pruned_15"]["max"] <= 1, c
        assert b["share_pred_fewer"]["min"] > 0.5, c


def test_over_inclusion_and_lost_structure_dominate_in_both(r):
    for c in ("c1", "c2"):
        d = r[c]["decomposition_mean"]
        assert d["over_inclusive"] + d["lost_structure"] > 0.75, c
        assert max(d, key=d.get) in ("over_inclusive", "lost_structure"), c


def test_dice_does_not_track_branching_in_either_configuration(r):
    """'Dice is uninformative about branch-point error.' |rho| small in both."""
    for c in ("c1", "c2"):
        assert abs(r[c]["rho"]["n_branch_points"]) < 0.2, c


def test_dice_tracks_total_length_in_both(r):
    for c in ("c1", "c2"):
        assert r[c]["rho"]["total_length"] < -0.5, c


def test_architecture_ranking_does_not_transfer_beyond_its_ends(r):
    """'Only the best and worst architecture agree between configurations.'

    If a re-run makes the ranking agree strongly, this sentence must be withdrawn.
    """
    cmp_ = r["comparison"]
    assert cmp_["same_best_architecture"] and cmp_["same_worst_architecture"]
    assert cmp_["architecture_rank_spearman"] < 0.8


def test_no_never_resolvable_claim_rests_on_a_clipped_floor(r):
    for c in ("c1", "c2"):
        f = r[c]
        assert f["adjacent_resolved_now"] + f["adjacent_never"] + f["adjacent_unknown"] <= 5
