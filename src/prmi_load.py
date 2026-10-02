"""Pair PRMI images with their masks and load masks as validated boolean arrays.

`prmi_ingest.py` parses filenames. This turns those records into the thing the trait code actually
consumes: a boolean array per mask, with every assumption checked rather than assumed.

WHY THIS IS A SEPARATE, GUARDED STEP
------------------------------------
Loading a mask looks like one line of `imread`. Four things can go wrong silently, and each would
produce plausible-looking traits from a wrong array:

  1. **Inverted polarity.** If a mask stores root=0 and background=255, every trait is computed on
     the soil. Nothing raises: the array is valid, the traits are finite, and `pixel_area` is simply
     enormous. Guarded by a foreground-fraction check -- roots occupy a small minority of a
     minirhizotron frame, so a mask that is mostly foreground is either inverted or not a mask.
  2. **Not actually binary.** Anti-aliased or JPEG-compressed masks carry intermediate values.
     Thresholding at >0 then counts compression ringing as root, which inflates `n_components`
     without limit -- and `n_components` is the trait this project has already shown is the most
     fragile of all. Guarded by reporting the value set and refusing anything with many levels
     unless a threshold is passed explicitly.
  3. **Wrong pairing.** PRMI ships ~72K images and ~63K masks, so roughly 9K images have no mask.
     A pairing that silently drops or mismatches them would compute Dice between unrelated frames.
     Guarded by pairing on the parsed identity tuple, never on list order, and by reporting
     unpaired counts instead of discarding them.
  4. **Shape mismatch.** An image and mask of different sizes cannot be compared, and a resize
     applied without noticing would change every topology trait.

Project 08's own history is the argument for this much care: a uint8 overflow in project 02 turned a
0.894 AUC into 0.518, and a silently-skipped guard in project 01 produced numbers that were pure
basis rotation. Both were caught by a value that was impossible on its face.

Everything here is tested against synthetic fixtures in `tests/test_prmi_load.py`, so it is known to
work before the 9.75 GB archive arrives.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# isort: off
import resources  # noqa: F401,E402  MUST load before numpy: caps BLAS threads
# isort: on

import io as _io  # noqa: E402
import zipfile  # noqa: E402
from collections import defaultdict  # noqa: E402

import numpy as np  # noqa: E402

from prmi_ingest import SchemaError  # noqa: E402

# An outright ceiling: no root mask is 85% of the frame. This is a last-resort backstop, not the
# inversion test -- see `looks_inverted` below for the one that actually discriminates.
#
# The first version of this guard used 0.60 and it was WRONG, in the direction that matters: it
# rejected two real papaya masks out of 63,943. Measured on the archive, papaya's densest frames
# genuinely reach 64.9% root, and the evidence that they are real rather than inverted is that
# (a) the tube's foreground distribution runs smoothly 46 -> 47 -> 49 -> 53 -> 58 -> 62 -> 65% with
# no gap, (b) the same tube and location across successive dates goes 46.4% -> 64.9% -> 61.8%,
# which is a root growing, and (c) the structural test below says root, not soil.
#
# The lesson is the one this project keeps relearning: a threshold picked from intuition is a claim
# about the data, and it has to be checked against the data like any other claim.
MAX_FOREGROUND_FRACTION = 0.85

# Above this, the fraction alone is no longer decisive and the structural test is applied.
#
# Set LOW on purpose, and the reasoning matters because the obvious choice is wrong. Inverting a
# mask maps fraction f to 1-f. PRMI's real root fractions run 0 to 65% with a non-empty median of
# 2%, so:
#   * an inverted SPARSE mask lands at 98% and the 0.85 ceiling catches it;
#   * an inverted DENSE mask lands at 35% -- a perfectly ordinary-looking fraction.
# So the dangerous band is the MIDDLE, not the top, and a structural test gated on *high* foreground
# could never fire on the case it was written for. A test whose trigger condition excludes its own
# motivating example is worse than no test, because it reads as coverage.
#
# 0.15 sits below the inverted-dense band (0.35+) with margin, and above the bulk of real masks
# (median 2%), so the cost is paid on a small minority of frames.
SUSPICIOUS_FOREGROUND_FRACTION = 0.15

# More distinct grey levels than this and the file is not a clean binary mask.
MAX_LEVELS_FOR_BINARY = 2


def identity(rec):
    """The tuple that identifies a frame, independent of whether it is the image or the mask.

    `tile` is part of the identity and must not be dropped: Switchgrass frames are tiled, so one
    (tube, location, date, time) covers many distinct tiles. Omitting it collapsed 12,537 Switchgrass
    images into a handful of identities, every one of them reported as ambiguous.

    The `GT_` prefix is deliberately NOT part of the identity -- it is exactly what distinguishes the
    mask from the image, and the whole point is that the two share one identity.
    """
    return (rec["species"], rec["tube"], rec["loc"], rec["date"], rec.get("time"),
            rec.get("tile"))


def pair_images_and_masks(records):
    """Match each image to its mask on parsed identity, never on list order.

    Returns (pairs, report). `pairs` is a list of dicts with `image` and `mask` records. Images with
    no mask are counted and returned separately rather than dropped -- PRMI has ~72K images against
    ~63K masks, so unpaired images are expected and their number is a fact worth reporting.
    """
    by_id = defaultdict(lambda: {"images": [], "masks": []})
    for r in records:
        by_id[identity(r)]["masks" if r["is_mask"] else "images"].append(r)

    pairs, unpaired_images, unpaired_masks, ambiguous = [], [], [], []
    for ident, grp in by_id.items():
        imgs, masks = grp["images"], grp["masks"]
        if len(imgs) > 1 or len(masks) > 1:
            ambiguous.append({"identity": ident, "n_images": len(imgs), "n_masks": len(masks)})
            continue
        if imgs and masks:
            pairs.append({"identity": ident, "image": imgs[0], "mask": masks[0]})
        elif imgs:
            unpaired_images.append(imgs[0])
        elif masks:
            unpaired_masks.append(masks[0])

    report = {
        "n_identities": len(by_id),
        "n_pairs": len(pairs),
        "n_images_without_mask": len(unpaired_images),
        "n_masks_without_image": len(unpaired_masks),
        "n_ambiguous_identities": len(ambiguous),
        "ambiguous_examples": ambiguous[:5],
        "pair_rate": len(pairs) / max(len(by_id), 1),
    }
    # A mask with no image is the one direction that should not happen: PRMI annotates a subset of
    # its images, never the reverse. If it does happen the identity tuple is wrong.
    if report["n_masks_without_image"] > 0.02 * max(len(by_id), 1):
        raise SchemaError(
            "{} masks have no matching image ({:.1%} of identities). The identity tuple is not "
            "matching images to masks correctly -- check whether masks carry the same stem. "
            "Examples: {}".format(report["n_masks_without_image"],
                                  report["n_masks_without_image"] / max(len(by_id), 1),
                                  [m["path"] for m in unpaired_masks[:5]])
        )
    return pairs, report, unpaired_images, unpaired_masks


def _decode(raw):
    """Decode image bytes to a 2-D array, without assuming a channel layout."""
    try:
        from imageio.v3 import imread
    except Exception:                                     # pragma: no cover - env dependent
        from imageio import imread
    a = np.asarray(imread(_io.BytesIO(raw)))
    if a.ndim == 3:
        # A greyscale mask saved as RGB: every channel is identical, so take one. If they differ
        # the file is not a binary mask and the level check below will reject it.
        a = a[..., 0]
    if a.ndim != 2:
        raise SchemaError(f"expected a 2-D mask, got shape {a.shape}")
    return a


def looks_inverted(mask):
    """Structural inversion test: is the mask the THIN thing, or the blobby thing around it?

    Two statistics, both of which must agree before anything is rejected:

      max EDT  A root is thin, so the largest circle that fits inside it is small. Soil is blobby,
               so the largest circle fitting in the background is large. On a real flagged mask:
               43.7 px inside the roots against 173.1 px inside the soil.
      border   Soil reaches the frame edge; roots seldom do. Same mask: 6.2% of border pixels are
               foreground.

    HOW THESE WERE CHOSEN, because the first attempt was wrong. The original rule was
    `components(mask) > components(~mask)`, on the reasoning that roots form few connected pieces
    while the soil between them is fragmented. It is a plausible story and it does not survive
    contact with the data: measured against 79 real masks and their inverses, it scored **0/17** on
    the masks it flagged and 51/60 and 44/60 on a random sample -- barely better than chance in the
    direction that matters. It fires on any mask whose roots are scattered rather than connected,
    which is common.

    max EDT and border each scored **79/79 on real masks and 79/79 on their inverses**. Requiring
    both to agree costs nothing on this evidence and makes a false rejection require two independent
    failures.

    The lesson is the project's own habit, applied to a detector rather than a null: before trusting
    a test, measure its error rate on real data where the answer is known. Constructing the negatives
    by inverting real masks makes that possible without any labelling.
    """
    from scipy import ndimage as ndi
    # The mask is the blobby one: a bigger circle fits inside it than inside its complement.
    thin_says_inverted = (ndi.distance_transform_edt(mask).max()
                          > ndi.distance_transform_edt(~mask).max())
    # The mask lines the frame edge. Border fractions of a mask and its complement sum to 1, so
    # "more of the border than the complement has" is exactly "more than half the border".
    border_says_inverted = _border_fraction(mask) > 0.5
    return bool(thin_says_inverted and border_says_inverted)


def _border_fraction(m):
    return float(np.concatenate([m[0], m[-1], m[:, 0], m[:, -1]]).mean())


def to_boolean_mask(raw, path="<bytes>", threshold=None,
                    max_foreground=MAX_FOREGROUND_FRACTION):
    """Decode mask bytes to a boolean array, refusing anything that is not a clean binary mask.

    `threshold` is None by default, which means "the file must already be binary". Pass a number to
    accept a greyscale mask, and know that you are choosing where compression artefacts land.
    """
    a = _decode(raw)
    levels = np.unique(a)
    if threshold is None:
        if levels.size > MAX_LEVELS_FOR_BINARY:
            raise SchemaError(
                f"{path} is not a binary mask: {levels.size} distinct values {levels[:8]}. Anti-aliased or JPEG-compressed "
                "masks must be thresholded explicitly -- thresholding at >0 counts compression "
                "ringing as root and inflates n_components without limit."
            )
        m = a > levels.min() if levels.size == 2 else np.zeros(a.shape, bool)
    else:
        m = a > threshold

    frac = float(m.mean())
    if SUSPICIOUS_FOREGROUND_FRACTION < frac <= max_foreground and looks_inverted(m):
        raise SchemaError(
            f"{path} is {frac:.1%} foreground AND its complement is more root-like than it is "
            "(fewer connected components), so it is almost certainly INVERTED. Nothing downstream "
            "would raise -- it would just compute every trait on the soil."
        )
    if frac > max_foreground:
        raise SchemaError(
            f"{path} is {frac:.1%} foreground, above the {max_foreground:.0%} ceiling. The mask is almost certainly "
            "INVERTED (root stored as 0). Nothing downstream would raise -- it would just compute "
            "every trait on the soil."
        )
    return m


class DirectoryReader:
    """Read entries by relative path from an extracted tree, mirroring ZipFile.read()."""

    def __init__(self, root):
        self.root = root

    def read(self, rel):
        with open(os.path.join(self.root, rel.replace("/", os.sep)), "rb") as fh:
            return fh.read()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def open_source(path):
    """Open either a zip archive or an extracted directory, returning something with .read()."""
    if os.path.isdir(path):
        return DirectoryReader(path)
    return zipfile.ZipFile(path)


def load_pair(zf, pair, threshold=None, check_shape=True):
    """Load one (image, mask) pair. `zf` is a ZipFile or a DirectoryReader."""
    img = _decode(zf.read(pair["image"]["path"]))
    mask = to_boolean_mask(zf.read(pair["mask"]["path"]), path=pair["mask"]["path"],
                           threshold=threshold)
    if check_shape and img.shape[:2] != mask.shape[:2]:
        raise SchemaError(
            "shape mismatch for {}: image {} vs mask {}. Resizing either one would change every "
            "topology trait.".format(pair["identity"], img.shape[:2], mask.shape[:2])
        )
    return img, mask


def audit_masks(source, pairs, limit=200, threshold=None, seed=0):
    """Load a sample of masks and report what they actually contain, before trusting any of them.

    `source` is a zip path or an extracted directory. Returns aggregate statistics plus every
    failure, so a systematic problem (inverted polarity, greyscale masks, a shape convention) shows
    up on a sample instead of after a full run.
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(pairs))[:limit]
    fracs, shapes, failures, by_config = [], [], [], {}
    with open_source(source) as src:
        for i in idx:
            p = pairs[int(i)]
            try:
                _, m = load_pair(src, p, threshold=threshold)
            except Exception as exc:              # report, do not abort the audit
                failures.append({"path": p["mask"]["path"], "error": str(exc)[:300]})
                continue
            f = float(m.mean())
            fracs.append(f)
            shapes.append(m.shape)
            cfg = p["mask"].get("config") or "unknown"
            by_config.setdefault(cfg, []).append(f)

    out = {
        "n_sampled": len(idx),
        "n_ok": len(fracs),
        "n_failed": len(failures),
        "failures": failures[:20],
        "foreground_fraction": {
            "min": float(np.min(fracs)) if fracs else None,
            "median": float(np.median(fracs)) if fracs else None,
            "max": float(np.max(fracs)) if fracs else None,
        },
        "foreground_by_config": {
            k: {"n": len(v), "median": float(np.median(v)), "max": float(np.max(v))}
            for k, v in sorted(by_config.items())
        },
        "distinct_shapes": sorted({tuple(s) for s in shapes})[:10],
        "n_distinct_shapes": len({tuple(s) for s in shapes}),
    }
    # An all-empty sample passes every individual guard and means the masks are unusable.
    if fracs and out["foreground_fraction"]["max"] == 0.0:
        raise SchemaError(
            "every sampled mask is empty. The mask files are not what this code expects."
        )
    return out
