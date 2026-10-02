# roottrait

Root traits from binary masks, and calibrated mask degradations for testing how segmentation
error reaches them.

Root segmentation models are usually compared by Dice overlap, but the masks are used to measure
traits. `roottrait` measures how much error in each trait a given Dice permits:

- **`traits`**: twelve root traits from a boolean mask (pixel area, total length, mean diameter,
  lacunarity, convex-hull area, bounding depth and width, tips, branch points, components,
  holes, fractal dimension), each labelled with its kind: integral, extremum, count or fitted.
- **`degrade`**: six named segmentation failures (dilation, erosion, thin-branch dropout,
  junction breaks, speckle, false blobs), each with one severity parameter.
- **`calibrate`**: bisection on severity so that a failure reaches an exact Dice loss on a given
  mask. Applied to one mask at one Dice, the six failures then differ only in how the pixels are
  wrong.
- **`decompose`**: assigns every wrong pixel of a prediction to one of four error kinds
  (over-inclusive, detached false positive, under-inclusive, lost structure).
- **`clean.fill_pinholes`**: fills one-pixel annotation holes, which otherwise turn into skeleton
  loops and inflate junction counts.
- **`graph`** (spur pruning, junction counting) and **`morphology`** (four synthetic root
  architectures with independent thickness).

## Install

```
pip install roottrait
```

The library needs only numpy, scipy and scikit-image.

## Example

```python
import numpy as np
from roottrait import DEGRADATIONS, SEVERITY_BOUNDS, TRAITS, dice, generate, solve_severity

mask = generate("taproot", size=256, seed=0, thickness=5)
truth = {name: fn(mask) for name, fn in TRAITS.items()}

for failure in ("boundary_dilate", "speckle"):
    fn = DEGRADATIONS[failure][0]
    lo, hi = SEVERITY_BOUNDS[failure]
    severity, _ = solve_severity(fn, mask, 0.01, lo, hi, rng_seed=1)   # Dice 0.99
    degraded = fn(mask, severity, np.random.default_rng(1))
    print(failure, round(dice(mask, degraded), 3),
          "components", truth["n_components"], "->", TRAITS["n_components"](degraded))
```

Both degraded masks score the same Dice; the component count is unchanged by dilation and
multiplied by speckle.

## Licence

MIT.
