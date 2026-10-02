"""roottrait: root traits from binary masks, and calibrated mask degradations for testing them.

Four parts, usable separately:

  traits      twelve root traits computed from a boolean mask (area, length, diameter, tips,
              branch points, components, holes, convex hull, extents, fractal dimension,
              lacunarity), grouped by how a local pixel error enters them
  degrade     six named segmentation failures (dilation, erosion, thin-branch dropout, junction
              breaks, speckle, false blobs), each with one severity parameter
  calibrate   bisection on severity so that a degradation reaches an exact Dice loss on a given
              mask, which separates how much a mask is wrong from how it is wrong
  decompose   assigns each wrong pixel of a prediction to one of four error kinds

plus `clean.fill_pinholes` (fill one-pixel annotation holes before skeletonising), `graph`
(spur pruning and junction counting) and `morphology` (four synthetic root architectures).
"""
from .calibrate import SEVERITY_BOUNDS, calibrate_mask, solve_severity
from .clean import PINHOLE_PX, enclosed_holes, fill_pinholes
from .decompose import KINDS, decompose, decompose_set, shares
from .degrade import DEGRADATIONS
from .graph import junction_nodes, prune
from .morphology import MORPHOLOGIES, generate
from .traits import TRAIT_KIND, TRAITS, all_traits, dice, iou, skeleton

__version__ = "1.0.0"

__all__ = [
    "DEGRADATIONS", "KINDS", "MORPHOLOGIES", "PINHOLE_PX", "SEVERITY_BOUNDS", "TRAITS",
    "TRAIT_KIND", "all_traits", "calibrate_mask", "decompose", "decompose_set", "dice",
    "enclosed_holes", "fill_pinholes", "generate", "iou", "junction_nodes", "prune", "shares",
    "skeleton", "solve_severity",
]
