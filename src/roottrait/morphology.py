"""Four synthetic root architectures with an independent thickness parameter.

The shapes follow the topological classification of root systems (Fitter, 1987):

  herringbone   one dominant axis with laterals that do not branch again
  dichotomous   repeated bifurcation, every branch splitting again
  taproot       a dominant vertical primary with laterals that branch once
  fibrous       many roots of similar rank from a crown, no dominant axis

Thickness is a separate argument because the degradations act on pixels: a fixed-size defect
removes proportionally more of a thin root, so architecture and thickness must be varied
independently to tell their effects apart. `THICKNESS_LEVELS` gives the three used in the paper.

Every generator returns a boolean 2-D array and takes `(size, seed, thickness, **kw)`.
"""
import numpy as np


class _Canvas:
    """Stamp disks along segments. Shared by every generator so line thickness means one thing."""

    def __init__(self, size):
        self.size = size
        self.m = np.zeros((size, size), bool)
        self.yy, self.xx = np.mgrid[0:size, 0:size]

    def stamp(self, y, x, r):
        if -r <= y < self.size + r and -r <= x < self.size + r:
            self.m[(self.yy - y) ** 2 + (self.xx - x) ** 2 <= r * r] = True

    def segment(self, y, x, ang, length, thick):
        """Draw from (y, x) along `ang`; return the end point. ang=0 is straight down."""
        steps = max(int(round(length)), 1)
        for t in range(steps + 1):
            self.stamp(int(round(y + t * np.cos(ang))),
                       int(round(x + t * np.sin(ang))),
                       max(1, int(round(thick))))
        return y + steps * np.cos(ang), x + steps * np.sin(ang)


def herringbone(size=384, seed=0, thickness=5, n_laterals=14, lateral_frac=0.30):
    """One primary axis; laterals come off it and do not branch again."""
    rng = np.random.default_rng(seed)
    c = _Canvas(size)
    x0 = size / 2
    main_len = size * 0.92
    c.segment(2, x0, rng.uniform(-0.05, 0.05), main_len, thickness)
    for i in range(n_laterals):
        f = (i + 1) / (n_laterals + 1)
        y = 2 + f * main_len
        side = 1 if i % 2 == 0 else -1
        ang = side * rng.uniform(0.9, 1.35)
        c.segment(y, x0, ang, main_len * lateral_frac * rng.uniform(0.6, 1.0), thickness * 0.5)
    return c.m


def dichotomous(size=384, seed=0, thickness=5, depth=5, split=2):
    """Repeated bifurcation: every branch splits into `split` children, `depth` times."""
    rng = np.random.default_rng(seed)
    c = _Canvas(size)

    def rec(y, x, ang, length, thick, d):
        ey, ex = c.segment(y, x, ang, length, thick)
        if d <= 0:
            return
        for k in range(split):
            spread = 0.55 * (k - (split - 1) / 2) / max(1, (split - 1) / 2 or 1)
            rec(ey, ex, ang + spread + rng.uniform(-0.12, 0.12),
                length * rng.uniform(0.62, 0.78), thick * 0.7, d - 1)

    rec(2, size / 2, rng.uniform(-0.05, 0.05), size / (depth + 1.2), thickness, depth)
    return c.m


def taproot(size=384, seed=0, thickness=6, n_laterals=11, sublaterals=2):
    """A dominant vertical primary; laterals branch once more. The dicot form."""
    rng = np.random.default_rng(seed)
    c = _Canvas(size)
    x0 = size / 2
    main_len = size * 0.94
    c.segment(2, x0, rng.uniform(-0.04, 0.04), main_len, thickness)
    for i in range(n_laterals):
        f = (i + 1) / (n_laterals + 1)
        y = 2 + f * main_len
        side = 1 if i % 2 == 0 else -1
        ang = side * rng.uniform(0.8, 1.25)
        llen = main_len * 0.30 * (1.0 - 0.45 * f) * rng.uniform(0.7, 1.05)
        ey, ex = c.segment(y, x0, ang, llen, thickness * 0.45)
        for _ in range(sublaterals):
            c.segment(ey, ex, ang + rng.uniform(-0.8, 0.8), llen * rng.uniform(0.3, 0.55),
                      thickness * 0.28)
    return c.m


def fibrous(size=384, seed=0, thickness=3, n_primary=13, depth=2):
    """Many similar-rank roots from a crown; no dominant axis, wider and shallower."""
    rng = np.random.default_rng(seed)
    c = _Canvas(size)
    x0 = size / 2

    def rec(y, x, ang, length, thick, d):
        ey, ex = c.segment(y, x, ang, length, thick)
        if d <= 0:
            return
        for _ in range(rng.integers(1, 3)):
            rec(ey, ex, ang + rng.uniform(-0.55, 0.55), length * rng.uniform(0.45, 0.7),
                thick * 0.7, d - 1)

    for i in range(n_primary):
        ang = np.interp(i, [0, n_primary - 1], [-1.05, 1.05]) + rng.uniform(-0.1, 0.1)
        rec(2, x0, ang, size * 0.42 * rng.uniform(0.75, 1.0), thickness, depth)
    return c.m


MORPHOLOGIES = {
    "herringbone": herringbone,
    "dichotomous": dichotomous,
    "taproot": taproot,
    "fibrous": fibrous,
}

# Thickness levels crossed with every architecture, so the two are separable. The junction_break
# wound radius is 3 px, so `thin` sits below it, `medium` at roughly twice it, `thick` well above.
THICKNESS_LEVELS = {"thin": 2, "medium": 5, "thick": 8}


def generate(morphology, size=384, seed=0, thickness=5):
    if morphology not in MORPHOLOGIES:
        raise KeyError(f"unknown morphology {morphology!r}; have {sorted(MORPHOLOGIES)}")
    return MORPHOLOGIES[morphology](size=size, seed=seed, thickness=thickness)
