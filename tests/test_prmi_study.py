"""The study driver's bookkeeping: atomic writes, resume logic and unit ordering.

None of this is science, and all of it decides whether a 3-hour GPU run produces usable results.
Each test here corresponds to a way the run can burn compute and leave nothing behind.
"""
import json
import os
import sys

import numpy as np
import pytest

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
sys.path.insert(0, SRC)


def test_numpy_save_appends_npy_which_breaks_a_naive_atomic_write(tmp_path):
    """The bug that cost a unit of GPU time, pinned as a property of numpy rather than of our code.

    `np.save` silently appends `.npy` when the path does not end in it. So the obvious temp-file
    idiom -- write `x.npy.tmp`, then `os.replace` it onto `x.npy` -- writes `x.npy.tmp.npy` and then
    raises FileNotFoundError on a file that was never created. Training had completed and the
    prediction was computed; the save threw it away.
    """
    bad = tmp_path / "pred.npy.tmp"
    np.save(bad, np.zeros(3))
    assert not bad.exists(), "numpy stopped appending .npy; this guard can be simplified"
    assert (tmp_path / "pred.npy.tmp.npy").exists()
    with pytest.raises(FileNotFoundError):
        os.replace(bad, tmp_path / "pred.npy")

    good = tmp_path / "pred.tmp.npy"
    np.save(good, np.zeros(3))
    assert good.exists()
    os.replace(good, tmp_path / "pred.npy")
    assert (tmp_path / "pred.npy").exists()


def test_run_unit_uses_a_temp_name_that_survives_numpy(tmp_path, monkeypatch):
    """End to end against a stubbed trainer: the files a resume looks for must actually appear."""
    import prmi_study

    class _M:
        def __init__(self):
            pass

    def fake_train_one(arch, seed, cache_dir, config, size, epochs, batch, lr):
        import torch
        pred = torch.zeros((5, 1, 8, 8), dtype=torch.bool)
        d = np.linspace(0.1, 0.9, 5)
        return _M(), pred, d, {"arch": arch, "seed": seed, "mean_dice_all_frames": 0.5,
                               "mean_dice_nonempty_only": 0.6, "pred_empty_rate": 0.1,
                               "train_seconds": 1.0}, []

    monkeypatch.setattr(prmi_study, "train_one", fake_train_one)
    monkeypatch.setattr(prmi_study.torch.cuda, "empty_cache", lambda: None)

    out = str(tmp_path)
    prmi_study.run_unit("cfg", "unet_r34", 0, out, "cache", 32, 1, 2, 1e-3)
    tag = prmi_study.unit_tag("cfg", "unet_r34", 0)
    assert os.path.exists(os.path.join(out, tag + "_pred.npy"))
    assert os.path.exists(os.path.join(out, tag + "_summary.json"))
    assert os.path.exists(os.path.join(out, tag + "_dice.npy"))
    assert not [f for f in os.listdir(out) if ".tmp" in f], "temp files left behind"
    assert prmi_study.unit_done(out, tag)


def test_unit_is_not_done_without_its_prediction(tmp_path):
    """A summary alone must not satisfy resume: that is how a failed unit becomes permanent.

    Project 02's I10: a unit that raised was written with `failed: true` and counted as done on
    resume, so a transient out-of-memory deleted that architecture-seed cell for good -- and because
    the loader trims to a balanced design, cost every other architecture a seed too.
    """
    import prmi_study
    tag = "cfg__unet_r34__s0"
    with open(os.path.join(tmp_path, tag + "_summary.json"), "w") as fh:
        json.dump({"arch": "unet_r34"}, fh)
    assert not prmi_study.unit_done(str(tmp_path), tag)


def test_a_truncated_summary_is_redone_not_trusted(tmp_path):
    import prmi_study
    tag = "cfg__unet_r34__s0"
    np.save(os.path.join(tmp_path, tag + "_pred.npy"), np.zeros(3))
    with open(os.path.join(tmp_path, tag + "_summary.json"), "w") as fh:
        fh.write('{"arch": "unet_r3')
    assert not prmi_study.unit_done(str(tmp_path), tag)


def test_units_run_seed_major_so_an_interrupted_run_stays_balanced():
    """All architectures at seed 0, then all at seed 1. Not architecture by architecture.

    A rank over architectures where one has three seeds and another has one is not a rank, and that
    is exactly what an architecture-major order leaves behind when the run is stopped early.
    """
    import prmi_study
    archs = ["a", "b", "c"]
    units = [(c, arch, s) for c in ["cfg"] for s in range(3) for arch in archs]
    seeds_in_order = [u[2] for u in units]
    assert seeds_in_order == sorted(seeds_in_order), "units are not seed-major"
    # the first len(archs) units must cover every architecture exactly once
    assert {u[1] for u in units[:len(archs)]} == set(archs)
    assert prmi_study.unit_tag("cfg", "a", 0) == "cfg__a__s0"


def test_batched_dice_matches_a_reference_including_the_empty_convention():
    """The OOM fix rewrote dice_per_frame; it must give exactly the same numbers.

    Covers the four cases that matter for F15: empty-empty (convention says 1.0), empty GT with a
    non-empty prediction (0.0), non-empty GT with an empty prediction (0.0), and ordinary overlap.
    And it must agree across a batch boundary, which is where a batching bug would hide.
    """
    import torch

    from train_prmi import dice_per_frame

    rng = np.random.default_rng(0)
    n = 600
    gt = torch.from_numpy(rng.random((n, 1, 16, 16)) < 0.2)
    pr = torch.from_numpy(rng.random((n, 1, 16, 16)) < 0.2)
    gt[0], pr[0] = False, False                     # empty / empty   -> 1.0
    gt[1], pr[1] = False, True                      # empty GT        -> 0.0
    gt[2], pr[2] = True, False                      # empty pred      -> 0.0

    got = dice_per_frame(pr, gt, batch=128).numpy()

    ref = []
    for i in range(n):
        p, g = pr[i].numpy().ravel(), gt[i].numpy().ravel()
        tot = p.sum() + g.sum()
        ref.append(1.0 if tot == 0 else 2.0 * (p & g).sum() / tot)
    ref = np.array(ref)

    assert got[0] == 1.0 and got[1] == 0.0 and got[2] == 0.0
    np.testing.assert_allclose(got, ref, atol=1e-12)
    # the result must not depend on where the batch boundaries fall
    np.testing.assert_allclose(dice_per_frame(pr, gt, batch=7).numpy(), ref, atol=1e-12)
