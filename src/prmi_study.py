"""Run the (architecture x seed) study over one or more PRMI configurations. Resumable.

Operational design copied from project 02's overnight study, where it earned its keep:

  * **Units run SEED-MAJOR.** An interrupted run then leaves a balanced design across architectures
    rather than some finished and some untouched. Reporting a rank over architectures where one has
    3 seeds and another has 1 is not a rank.
  * **Atomic per-unit writes.** Killing the process loses at most the unit in flight.
  * **A STOP file.** `touch results/prmi_models/STOP` lets the current unit finish and write, then
    exits cleanly. Project 02's F-series was interrupted this way repeatedly without losing a unit.
  * **A failed unit is recorded as failed and retried**, never written as done. Project 02 had a
    latent bug (its I10) where a unit that raised was written with `failed: true` and then counted
    as complete on resume, so a transient out-of-memory would have deleted that architecture-seed
    cell permanently and cost every other architecture a seed, because the loader trims to a
    balanced design.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# isort: off
import resources  # noqa: F401,E402  MUST load before numpy: caps BLAS threads
# isort: on

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from prmi_dataset import DEFAULT_SIZE  # noqa: E402
from train_prmi import ARCHITECTURES, train_one  # noqa: E402


def unit_tag(config, arch, seed):
    return f"{config}__{arch}__s{seed}"


def unit_done(outdir, tag):
    """A unit counts as done only if its summary parses AND its prediction file exists."""
    s = os.path.join(outdir, tag + "_summary.json")
    p = os.path.join(outdir, tag + "_pred.npy")
    if not (os.path.exists(s) and os.path.exists(p)):
        return False
    try:
        with open(s) as fh:
            d = json.load(fh)
        return not d.get("failed", False)
    except Exception:
        return False                      # truncated summary: redo it rather than trust it


def run_unit(config, arch, seed, outdir, cache_dir, size, epochs, batch, lr):
    tag = unit_tag(config, arch, seed)
    _model, pred, d, summary, _meta = train_one(
        arch, seed, cache_dir, config, size, epochs, batch, lr)
    # The temp name must END in `.npy`. `np.save` silently appends `.npy` when the path does not,
    # so `np.save("x_pred.npy.tmp", a)` writes `x_pred.npy.tmp.npy` and the `os.replace` below then
    # fails on a file that was never created. That cost one unit of GPU time: training finished, the
    # prediction was computed, and the save threw it away.
    tmp = os.path.join(outdir, tag + "_pred.tmp.npy")
    np.save(tmp, pred.squeeze(1).cpu().numpy().astype(bool))
    os.replace(tmp, os.path.join(outdir, tag + "_pred.npy"))
    np.save(os.path.join(outdir, tag + "_dice.npy"), d)
    tmp = os.path.join(outdir, tag + "_summary.json.tmp")
    with open(tmp, "w") as fh:
        json.dump(summary, fh, indent=2)
    os.replace(tmp, os.path.join(outdir, tag + "_summary.json"))
    del _model
    torch.cuda.empty_cache()
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=["peanut_640x480_DPI120"])
    ap.add_argument("--archs", nargs="*", default=sorted(ARCHITECTURES))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--size", type=int, default=DEFAULT_SIZE)
    ap.add_argument("--cache-dir", default="data/cache")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--outdir", default="results/prmi_models")
    a = ap.parse_args()

    os.makedirs(a.outdir, exist_ok=True)
    stop = os.path.join(a.outdir, "STOP")

    # SEED-MAJOR: every architecture at seed 0, then every architecture at seed 1, ...
    units = [(c, arch, s)
             for c in a.configs
             for s in range(a.seeds)
             for arch in a.archs]
    todo = [u for u in units if not unit_done(a.outdir, unit_tag(*u))]
    print(f"{len(units)} units, {len(units) - len(todo)} already done, {len(todo)} to run")
    print(f"stop cleanly with:  touch {stop}")

    t0 = time.time()
    for i, (config, arch, seed) in enumerate(todo, 1):
        if os.path.exists(stop):
            print("STOP file present, exiting cleanly")
            break
        tag = unit_tag(config, arch, seed)
        el = time.time() - t0
        eta = (el / max(i - 1, 1)) * (len(todo) - i + 1) / 60 if i > 1 else float("nan")
        print(f"\n[{i}/{len(todo)}] {tag}  ({el / 60:.1f} min elapsed, ~{eta:.0f} min left)",
              flush=True)
        try:
            s = run_unit(config, arch, seed, a.outdir, a.cache_dir, a.size,
                         a.epochs, a.batch, a.lr)
            print(f"  dice all={s['mean_dice_all_frames']:.4f} "
                  f"nonempty={s['mean_dice_nonempty_only']:.4f} "
                  f"pred_empty={s['pred_empty_rate']:.3f} "
                  f"({s['train_seconds'] / 60:.1f} min)", flush=True)
        except Exception:
            # Record the failure for diagnosis, but do NOT write a summary that `unit_done` would
            # accept. The unit stays on the to-do list for the next resume.
            print(f"  FAILED: {traceback.format_exc()[-800:]}", flush=True)
            with open(os.path.join(a.outdir, tag + "_FAILED.txt"), "w") as fh:
                fh.write(traceback.format_exc())
            torch.cuda.empty_cache()

    done = sum(unit_done(a.outdir, unit_tag(*u)) for u in units)
    print(f"\n{done}/{len(units)} units complete in {(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
