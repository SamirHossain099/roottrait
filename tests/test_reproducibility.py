"""Regression test: the sweep must produce identical numbers in a fresh process.

This is the test that was missing. `dose_response.sweep` seeded its per-degradation RNG with
`hash(dname)`, and Python randomises string hashing per process unless PYTHONHASHSEED is set. So
the sweep produced different masks on every run, and the published `results/dose_response.csv`
could not be regenerated -- while every in-process test passed, because within one process the
hash is stable.

That is the whole difficulty: an in-process test CANNOT see this bug. The check has to cross a
process boundary, and it has to let the interpreter randomise its hash seed rather than pinning
PYTHONHASHSEED, which would hide exactly the defect being guarded against.

Project 02 shipped the same bug in a bootstrap. It is habit 13.
"""
import hashlib
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")

_SNIPPET = (
    "import sys; sys.path.insert(0, r'{src}');"
    "from dose_response import sweep;"
    "df = sweep(n_masks=1, size=160, seed=0, verbose=False);"
    "print(df.round(6).to_csv(index=False))"
)


def _run_sweep_in_fresh_process():
    env = dict(os.environ)
    # Let the interpreter randomise its hash seed. Pinning it here would defeat the test.
    env.pop("PYTHONHASHSEED", None)
    r = subprocess.run(
        [sys.executable, "-c", _SNIPPET.format(src=SRC)],
        capture_output=True, text=True, cwd=ROOT, env=env,
    )
    assert r.returncode == 0, f"sweep failed in subprocess:\n{r.stderr[-3000:]}"
    return hashlib.sha256(r.stdout.encode()).hexdigest()


def test_sweep_is_reproducible_across_processes():
    digests = {_run_sweep_in_fresh_process() for _ in range(3)}
    assert len(digests) == 1, (
        "sweep produced different results in different processes: "
        f"{sorted(digests)}. Something in the pipeline is seeded from a "
        "process-varying source (hash(), id(), time, or an unseeded global RNG)."
    )


def test_stable_seed_is_actually_stable_across_processes():
    """Pin the helper itself, so a future 'simplification' back to hash() fails loudly."""
    snippet = (
        f"import sys; sys.path.insert(0, r'{SRC}');"
        "from dose_response import stable_seed;"
        "print(stable_seed(0, 3, 'junction_break', 0.4))"
    )
    env = dict(os.environ)
    env.pop("PYTHONHASHSEED", None)
    vals = set()
    for _ in range(3):
        r = subprocess.run([sys.executable, "-c", snippet], capture_output=True, text=True,
                           cwd=ROOT, env=env)
        assert r.returncode == 0, r.stderr[-2000:]
        vals.add(r.stdout.strip())
    assert len(vals) == 1, f"stable_seed is not stable across processes: {vals}"
    # Pinned value: changing the derivation changes every downstream number, so it should be a
    # deliberate act that updates this line, not a silent drift.
    assert vals.pop() == "1494015008"


def test_stable_seed_separates_its_arguments():
    """Distinct inputs must give distinct seeds; a collision would silently couple two arms."""
    from importlib import import_module
    sys.path.insert(0, SRC)
    stable_seed = import_module("dose_response").stable_seed
    seeds = {
        stable_seed(0, m, d, s)
        for m in range(4)
        for d in ("junction_break", "speckle", "blob")
        for s in (0.1, 0.2, 0.4)
    }
    assert len(seeds) == 4 * 3 * 3, "stable_seed collided on distinct (mask, degradation, severity)"
