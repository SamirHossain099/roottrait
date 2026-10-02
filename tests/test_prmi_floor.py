"""The paired, architecture-level resolution floor: method checks and pinned consistency.

The first floor in this project compared differences between models to a floor on a SINGLE mean,
which includes the between-tube variance that cancels in any paired comparison. It overstated the
floor about 8x (0.050 against a median paired floor of 0.0062), in the direction that made the
benchmark look less able to resolve models than it is. These tests make that impossible to repeat.
"""
import glob
import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from prmi_floor import analyse, paired  # noqa: E402

FLOORS = sorted(glob.glob(os.path.join(ROOT, "results", "prmi_floor_*.json")))


def _synthetic(n_arch=3, n_seed=3, n_tube=7, tube_sd=0.10, inter_sd=0.0, seed_sd=0.002,
               arch_gap=0.02, rng=0):
    """score = tube effect + architecture effect + (optional) arch x tube interaction + seed noise."""
    r = np.random.default_rng(rng)
    tube = r.normal(0, tube_sd, n_tube)
    inter = r.normal(0, inter_sd, (n_arch, n_tube))
    out = {}
    for a in range(n_arch):
        out[f"a{a}"] = {}
        for s in range(n_seed):
            noise = r.normal(0, seed_sd, n_tube)
            out[f"a{a}"][s] = {f"t{t}": 0.7 + tube[t] - a * arch_gap + inter[a, t] + noise[t]
                               for t in range(n_tube)}
    return out


def test_a_shared_tube_effect_cancels_in_the_paired_floor():
    """Large between-tube variance and NO interaction: the paired floor must be near zero.

    This is the case the single-mean floor gets catastrophically wrong: it reports the full tube
    variance even though every architecture sees exactly the same tubes.
    """
    # A small non-zero interaction keeps the components estimable. With exactly zero, the estimate
    # clips to 0 and is now reported as unknown rather than resolvable -- conservative by design.
    res = analyse(_synthetic(tube_sd=0.15, inter_sd=0.003))
    assert res["median_paired_floor"] < 0.01
    assert res["single_mean_floor_NOT_for_comparisons"] > 5 * res["median_paired_floor"]
    # with a 0.02 gap and no interaction, every pair is resolvable
    assert res["n_pairs_resolvable_ever"] == res["n_pairs"]


def test_an_architecture_by_tube_interaction_does_not_cancel():
    """When architectures disagree about which tubes are easy, the paired floor must widen."""
    calm = analyse(_synthetic(inter_sd=0.003, rng=1))
    noisy = analyse(_synthetic(inter_sd=0.05, rng=1))
    assert noisy["median_paired_floor"] > 3 * calm["median_paired_floor"]


def test_a_gap_smaller_than_the_interaction_is_never_resolvable():
    res = analyse(_synthetic(arch_gap=0.001, inter_sd=0.04, rng=2))
    assert res["n_adjacent_resolvable_ever"] < len(res["adjacent"])


def test_a_clipped_floor_is_reported_as_unknown_not_resolvable():
    """True interaction exactly zero -> negative estimate -> clipped -> verdict None, not True."""
    res = analyse(_synthetic(inter_sd=0.0, seed_sd=0.002, rng=4))
    clipped = [a for a in res["adjacent"] if a["floor_clipped"]]
    assert clipped, "fixture did not produce a clipped estimate"
    assert all(a["resolvable_ever"] is None for a in clipped)


def test_paired_difference_has_the_right_sign_and_size():
    p = paired(_synthetic(arch_gap=0.02, inter_sd=0.0, seed_sd=0.0), "a0", "a1")
    assert p["mean_diff"] == pytest.approx(0.02, abs=1e-9)


# ------------------------------------------------------------------ pinned to results/
@pytest.mark.skipif(not FLOORS, reason="run src/prmi_floor.py first")
@pytest.mark.parametrize("path", FLOORS, ids=[os.path.basename(p) for p in FLOORS])
def test_floor_results_are_internally_consistent(path):
    with open(path) as fh:
        r = json.load(fh)
    n = r["n_architectures"]
    assert r["n_pairs"] == n * (n - 1) // 2
    assert len(r["adjacent"]) == n - 1
    assert r["ranking"] == sorted(r["arch_mean_dice_nonempty"],
                                  key=lambda a: -r["arch_mean_dice_nonempty"][a])
    for adj in r["adjacent"]:
        assert adj["gap"] >= 0
        if adj["floor_clipped"]:
            # an estimate clipped to zero must never be read as "resolvable"
            assert adj["floor"] == 0.0 and adj["resolvable_ever"] is None
        else:
            assert adj["resolvable_ever"] == (abs(adj["gap"]) > adj["floor"])
            # resolved now implies resolvable ever
            assert not adj["resolved_now"] or adj["resolvable_ever"]
        assert adj["resolved_now"] == (abs(adj["gap"]) > adj["halfwidth_now"])
        # with finitely many seeds the interval is never narrower than its own floor
        assert adj["halfwidth_now"] >= adj["floor"] - 1e-12
    adj = r["adjacent"]
    assert (r["n_adjacent_resolvable_ever"] + r["n_adjacent_never_resolvable"]
            + r["n_adjacent_floor_clipped"]) == len(adj)
    assert r["n_pairs_resolvable_ever"] + r["n_pairs_floor_clipped"] <= r["n_pairs"]
    for p in r["pairs"]:
        assert (p["resolvable_ever"] is None) == p["floor_clipped"]


@pytest.mark.skipif(not FLOORS, reason="run src/prmi_floor.py first")
@pytest.mark.parametrize("path", FLOORS, ids=[os.path.basename(p) for p in FLOORS])
def test_the_paired_floor_is_narrower_than_the_single_mean_floor(path):
    """Pairing removes the shared tube variance, so it can only help. If not, the decomposition is wrong."""
    with open(path) as fh:
        r = json.load(fh)
    if r["median_paired_floor"] is not None:
        assert r["median_paired_floor"] < r["single_mean_floor_NOT_for_comparisons"]
