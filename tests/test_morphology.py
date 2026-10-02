"""The morphology family must be four genuinely different shapes, with thickness separable.

Two ways this could be useless and look fine:

  1. The four generators produce structures that differ in name only. Then "the taxonomy replicates
     across architectures" is a claim about one shape drawn four times.
  2. Architecture and thickness are entangled -- if `fibrous` is only ever thin and `taproot` only
     ever thick, an apparent architecture effect is a thickness effect. `junction_break` punches a
     fixed 3 px disk, so thinner roots lose proportionally more, and the confound runs in exactly
     the direction that would manufacture the result.
"""
import os
import sys

import numpy as np
import pytest

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
sys.path.insert(0, SRC)

from morphology import MORPHOLOGIES, THICKNESS_LEVELS, generate  # noqa: E402
from traits import all_traits, n_branch_points, pixel_area, total_length  # noqa: E402

ARCHS = sorted(MORPHOLOGIES)


@pytest.mark.parametrize("arch", ARCHS)
def test_generator_produces_a_plausible_root(arch):
    m = generate(arch, size=384, seed=0, thickness=5)
    assert m.dtype == bool and m.ndim == 2
    frac = m.mean()
    assert 0.005 < frac < 0.5, f"{arch} fills {frac:.3f} of the frame"
    assert n_branch_points(m) >= 5, arch + " has almost no branch points"


@pytest.mark.parametrize("arch", ARCHS)
def test_generator_is_deterministic(arch):
    a = generate(arch, size=256, seed=3, thickness=5)
    b = generate(arch, size=256, seed=3, thickness=5)
    assert np.array_equal(a, b), arch + " is not deterministic for a fixed seed"


@pytest.mark.parametrize("arch", ARCHS)
def test_different_seeds_give_different_masks(arch):
    a = generate(arch, size=256, seed=0, thickness=5)
    b = generate(arch, size=256, seed=1, thickness=5)
    assert not np.array_equal(a, b), arch + " ignores its seed"


def test_the_four_architectures_are_actually_different():
    """Branch density per unit skeleton separates them; if it does not, they are one shape."""
    density = {}
    for arch in ARCHS:
        m = generate(arch, size=384, seed=0, thickness=5)
        density[arch] = 1000.0 * n_branch_points(m) / max(total_length(m), 1.0)
    vals = sorted(density.values())
    assert vals[-1] / vals[0] > 2.0, (
        f"branch density spans only {vals[-1] / vals[0]:.2f}x across architectures -- they are not distinct: "
        f"{density}"
    )


@pytest.mark.parametrize("arch", ARCHS)
def test_thickness_moves_area_but_not_skeleton_length(arch):
    """This is what makes the confound control work.

    If thickness also changed the skeleton, the design could not separate "thin roots are fragile"
    from "this architecture is fragile". Area should scale with thickness; skeleton length should
    not.
    """
    lens, areas = [], []
    for t in THICKNESS_LEVELS.values():
        m = generate(arch, size=384, seed=0, thickness=t)
        lens.append(total_length(m))
        areas.append(pixel_area(m))
    assert areas[-1] > 1.5 * areas[0], arch + ": thickness barely changed area"
    span = max(lens) / max(min(lens), 1.0)
    assert span < 1.6, (
        f"{arch}: skeleton length varies {span:.2f}x across thickness levels, so thickness and structure "
        "are entangled"
    )


def test_every_architecture_supports_every_thickness():
    """A missing cell would unbalance the design and make the margins uninterpretable."""
    for arch in ARCHS:
        for tname, t in THICKNESS_LEVELS.items():
            m = generate(arch, size=384, seed=0, thickness=t)
            assert m.any(), f"{arch} at thickness {tname} is empty"


@pytest.mark.parametrize("arch", ARCHS)
def test_all_traits_are_finite_on_every_architecture(arch):
    tr = all_traits(generate(arch, size=384, seed=0, thickness=5))
    bad = {k: v for k, v in tr.items() if not np.isfinite(v)}
    assert not bad, f"{arch} produced non-finite traits: {bad}"


def test_unknown_morphology_raises():
    with pytest.raises(KeyError):
        generate("mycorrhiza", size=64)
