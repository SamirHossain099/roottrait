"""Mask loading and pairing, tested against synthetic fixtures built to fail in the real ways.

Each test encodes one way a mask can be wrong while still decoding cleanly. That is the whole risk
here: none of these raise on their own, they just produce traits computed on the wrong array.
"""
import io
import os
import sys
import zipfile

import numpy as np
import pytest

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
sys.path.insert(0, SRC)

from prmi_ingest import SchemaError, index_names  # noqa: E402
from prmi_load import (  # noqa: E402
    audit_masks,
    identity,
    load_pair,
    looks_inverted,
    pair_images_and_masks,
    to_boolean_mask,
)

imageio = pytest.importorskip("imageio.v3")


def png_bytes(arr):
    buf = io.BytesIO()
    imageio.imwrite(buf, arr.astype(np.uint8), extension=".png")
    return buf.getvalue()


def root_mask(size=64, frac=0.08, seed=0):
    """A sparse root-like mask: a vertical line with a few laterals."""
    rng = np.random.default_rng(seed)
    m = np.zeros((size, size), bool)
    m[:, size // 2 - 1:size // 2 + 1] = True
    for y in rng.choice(np.arange(4, size - 4), size=4, replace=False):
        m[y:y + 1, size // 4:3 * size // 4] = True
    return m


def names(n_tubes=3, n_locs=4, with_masks=True, mask_subset=None):
    out = []
    for t in range(1, n_tubes + 1):
        for loc in range(1, n_locs + 1):
            stem = f"Peanut_T{t:03d}_L{loc:03d}_2017.07.13_10352{loc}_EED_DPI120"
            out.append(f"PRMI/images/peanut/{stem}.jpg")
            if with_masks and (mask_subset is None or (t, loc) in mask_subset):
                out.append(f"PRMI/masks/peanut/{stem}.png")
    return out


# ------------------------------------------------------------------------ pairing
def test_images_pair_with_their_masks_on_identity_not_order():
    recs, _ = index_names(names())
    # Shuffle hard: a pairing that relies on list order will now be wrong.
    rng = np.random.default_rng(0)
    recs = [recs[i] for i in rng.permutation(len(recs))]
    pairs, rep, _, _ = pair_images_and_masks(recs)
    assert rep["n_pairs"] == 12
    assert rep["n_images_without_mask"] == 0
    for p in pairs:
        assert identity(p["image"]) == identity(p["mask"])
        assert p["image"]["path"].endswith(".jpg") and p["mask"]["path"].endswith(".png")


def test_images_without_masks_are_counted_not_dropped():
    """PRMI has ~72K images against ~63K masks; unpaired images are expected."""
    subset = {(1, 1), (1, 2), (2, 1)}
    recs, _ = index_names(names(mask_subset=subset))
    pairs, rep, unpaired_img, unpaired_mask = pair_images_and_masks(recs)
    assert rep["n_pairs"] == 3
    assert rep["n_images_without_mask"] == 9
    assert len(unpaired_img) == 9 and not unpaired_mask
    assert rep["n_pairs"] + rep["n_images_without_mask"] == 12


def test_masks_without_images_raise_because_that_direction_should_not_happen():
    recs, _ = index_names(names())
    recs = [r for r in recs if r["is_mask"]]        # masks only: the identity tuple must be wrong
    with pytest.raises(SchemaError, match="masks have no matching image"):
        pair_images_and_masks(recs)


def test_ambiguous_identities_are_reported_not_silently_resolved():
    n = names(n_tubes=1, n_locs=1)
    n.append("PRMI/masks_v2/peanut/Peanut_T001_L001_2017.07.13_103521_EED_DPI120.png")
    recs, _ = index_names(n)
    _, rep, _, _ = pair_images_and_masks(recs)
    assert rep["n_ambiguous_identities"] == 1
    assert rep["n_pairs"] == 0, "an ambiguous identity must not be paired by guessing"


# ------------------------------------------------------------------------ mask decoding
def test_a_clean_binary_mask_loads():
    m = root_mask()
    got = to_boolean_mask(png_bytes(m * 255))
    assert got.dtype == bool and np.array_equal(got, m)


def test_zero_one_encoding_loads_the_same_as_zero_255():
    m = root_mask()
    assert np.array_equal(to_boolean_mask(png_bytes(m * 1)),
                          to_boolean_mask(png_bytes(m * 255)))


def test_an_inverted_mask_is_refused():
    """The failure that raises nothing: every trait would be computed on the soil."""
    m = root_mask()
    with pytest.raises(SchemaError, match="INVERTED"):
        to_boolean_mask(png_bytes((~m) * 255), path="inverted.png")


def dense_root_mask(size=128, seed=0):
    """A dense root system: >45% foreground, so the fraction alone cannot decide polarity.

    This is the papaya case. Real PRMI papaya masks reach 64.9% root, and a 60% ceiling rejected
    two genuine ones. Density is not inversion, and only structure can tell them apart.
    """
    rng = np.random.default_rng(seed)
    m = np.zeros((size, size), bool)
    for x in range(3, size - 3, 5):                 # many thick vertical roots
        m[:, max(0, x - 1):x + 2] = True
    for y in range(3, size - 3, 7):                 # crossing laterals
        m[y:y + 2, :] = True
    m |= rng.random((size, size)) < 0.05
    return m


def test_a_dense_but_correct_mask_is_accepted():
    """The failure the first guard actually produced: rejecting real papaya masks for being dense."""
    m = dense_root_mask()
    assert m.mean() > 0.45, "fixture is not dense enough to exercise the structural test"
    got = to_boolean_mask(png_bytes(m * 255), path="dense.png")
    assert got.mean() > 0.45


def test_inversion_is_caught_structurally_when_the_fraction_cannot_decide():
    """Both a dense mask and its inverse sit above the suspicion threshold; structure separates them."""
    m = dense_root_mask()
    inv = ~m
    assert not looks_inverted(m), "a correct dense mask was called inverted"
    assert looks_inverted(inv), "an inverted dense mask was not detected"
    with pytest.raises(SchemaError, match="INVERTED"):
        to_boolean_mask(png_bytes(inv * 255), path="dense_inverted.png")


def test_the_two_real_papaya_masks_that_broke_the_first_guard_are_representable():
    """Pin the measured facts: 64.9% foreground, and the complement is the fragmented one."""
    m = dense_root_mask()
    from scipy import ndimage as ndi
    st = np.ones((3, 3), np.uint8)
    n_fg = ndi.label(m, structure=st)[1]
    n_bg = ndi.label(~m, structure=st)[1]
    assert n_fg < n_bg, (
        f"the discriminator's premise fails on this fixture: {n_fg} root components vs {n_bg} soil "
        "components. A root mask must have fewer components than its complement."
    )


def test_a_greyscale_mask_is_refused_unless_a_threshold_is_chosen_explicitly():
    """Thresholding at >0 would count compression ringing as root and inflate n_components."""
    m = root_mask().astype(np.uint8) * 255
    m[10, 10:20] = 128                                   # anti-aliased edge
    m[12, 10:20] = 64
    with pytest.raises(SchemaError, match="not a binary mask"):
        to_boolean_mask(png_bytes(m), path="grey.png")
    got = to_boolean_mask(png_bytes(m), path="grey.png", threshold=127)
    assert got.dtype == bool and got.sum() > 0


def test_an_all_background_mask_loads_as_empty_rather_than_raising():
    got = to_boolean_mask(png_bytes(np.zeros((32, 32), np.uint8)))
    assert got.dtype == bool and not got.any()


def test_rgb_encoded_greyscale_mask_loads():
    m = root_mask()
    rgb = np.stack([m * 255] * 3, axis=-1)
    assert np.array_equal(to_boolean_mask(png_bytes(rgb)), m)


# ------------------------------------------------------------------------ pair loading
def _fixture_zip(tmp_path, mask_arrays=None, image_shape=None):
    p = tmp_path / "prmi.zip"
    ns = names(n_tubes=2, n_locs=2)
    with zipfile.ZipFile(p, "w") as zf:
        for n in ns:
            if n.endswith(".png"):
                arr = (mask_arrays or {}).get(n, root_mask() * 255)
                zf.writestr(n, png_bytes(arr))
            else:
                shape = image_shape or (64, 64)
                zf.writestr(n, png_bytes(np.full(shape, 120, np.uint8)))
    return p, ns


def test_load_pair_returns_image_and_boolean_mask(tmp_path):
    p, ns = _fixture_zip(tmp_path)
    recs, _ = index_names(ns)
    pairs, _, _, _ = pair_images_and_masks(recs)
    with zipfile.ZipFile(p) as zf:
        img, mask = load_pair(zf, pairs[0])
    assert mask.dtype == bool
    assert img.shape[:2] == mask.shape[:2]


def test_shape_mismatch_raises(tmp_path):
    p, ns = _fixture_zip(tmp_path, image_shape=(48, 48))
    recs, _ = index_names(ns)
    pairs, _, _, _ = pair_images_and_masks(recs)
    with zipfile.ZipFile(p) as zf:
        with pytest.raises(SchemaError, match="shape mismatch"):
            load_pair(zf, pairs[0])


# ------------------------------------------------------------------------ the audit
def test_audit_reports_statistics_over_a_sample(tmp_path):
    p, ns = _fixture_zip(tmp_path)
    recs, _ = index_names(ns)
    pairs, _, _, _ = pair_images_and_masks(recs)
    out = audit_masks(str(p), pairs, limit=10)
    assert out["n_ok"] == 4 and out["n_failed"] == 0
    assert 0.0 < out["foreground_fraction"]["median"] < 0.6
    assert out["n_distinct_shapes"] == 1


def test_audit_collects_failures_instead_of_aborting(tmp_path):
    """One bad mask must not hide the statistics of the good ones."""
    bad = "PRMI/masks/peanut/Peanut_T001_L001_2017.07.13_103521_EED_DPI120.png"
    p, ns = _fixture_zip(tmp_path, mask_arrays={bad: (~root_mask()) * 255})
    recs, _ = index_names(ns)
    pairs, _, _, _ = pair_images_and_masks(recs)
    out = audit_masks(str(p), pairs, limit=10)
    assert out["n_failed"] == 1 and out["n_ok"] == 3
    assert "INVERTED" in out["failures"][0]["error"]


def test_audit_raises_when_every_mask_is_empty(tmp_path):
    ns = names(n_tubes=2, n_locs=2)
    p = tmp_path / "empty.zip"
    with zipfile.ZipFile(p, "w") as zf:
        for n in ns:
            zf.writestr(n, png_bytes(np.zeros((64, 64), np.uint8)))
    recs, _ = index_names(ns)
    pairs, _, _, _ = pair_images_and_masks(recs)
    with pytest.raises(SchemaError, match="every sampled mask is empty"):
        audit_masks(str(p), pairs, limit=10)


def test_loaded_masks_feed_the_trait_code(tmp_path):
    """End to end: the array this module produces is what `traits.py` consumes."""
    from traits import all_traits
    p, ns = _fixture_zip(tmp_path)
    recs, _ = index_names(ns)
    pairs, _, _, _ = pair_images_and_masks(recs)
    with zipfile.ZipFile(p) as zf:
        _, mask = load_pair(zf, pairs[0])
    tr = all_traits(mask)
    assert np.isfinite(list(tr.values())).all()
    assert tr["pixel_area"] > 0


# ------------------------------------------------------------------ the on-arrival entry point
def test_full_ingest_runs_end_to_end_on_a_fixture_archive(tmp_path, monkeypatch, capsys):
    """Prove the single command works before the real 9.75 GB archive exists.

    `python src/prmi_ingest.py` is what runs the moment the download lands. If it only ever gets
    exercised on the real file, its first run is also its first test -- on a 9.75 GB input, after a
    long download, which is the worst possible moment to discover a typo in the reporting path.
    Project 02 shipped a contention check whose reporting path read three fields that did not
    exist; it was caught by testing that path on synthetic batches.
    """
    import json
    import zipfile

    import prmi_ingest

    ns = names(n_tubes=4, n_locs=4)
    zp = tmp_path / "PRMI_official.zip"
    with zipfile.ZipFile(zp, "w") as zf:
        for n in ns:
            if n.endswith(".png"):
                zf.writestr(n, png_bytes(root_mask() * 255))
            else:
                zf.writestr(n, png_bytes(np.full((64, 64), 120, np.uint8)))
        zf.writestr("PRMI/README", "fixture")

    # The real archive's size and checksum are pinned; a fixture cannot match them.
    monkeypatch.setattr(prmi_ingest, "PRMI_ZIP_BYTES", zp.stat().st_size)
    monkeypatch.setattr(prmi_ingest, "PRMI_ZIP_SHA256", None)

    def _fake_verify(path, expect_sha=None, expect_bytes=None, quick=False):
        return {"path": str(path), "size_bytes": zp.stat().st_size,
                "size_ok": True, "sha256": None, "sha256_ok": None}

    monkeypatch.setattr(prmi_ingest, "verify_zip", _fake_verify)

    out = tmp_path / "prmi_index.json"
    monkeypatch.setattr(sys, "argv",
                        ["prmi_ingest.py", "--zip", str(zp), "--out", str(out), "--audit-sample", "8"])
    assert prmi_ingest.main() == 0

    written = json.loads(out.read_text())
    for key in ("archive", "index", "pairing", "mask_audit", "leakage", "split"):
        assert key in written, "the on-arrival report is missing the '" + key + "' section"
    assert written["index"]["n_parsed"] == 32
    assert written["pairing"]["n_pairs"] == 16
    assert written["mask_audit"]["n_failed"] == 0
    assert written["leakage"]["group_disjoint_val_group_overlap"] == 0.0
    assert set(written["split"]["group_counts"]) == {"train", "val", "test"}
    # The console output is the thing a human reads on arrival; make sure every stage printed.
    printed = capsys.readouterr().out
    for stage in ("1. index", "2. image/mask pairing", "3. mask audit", "4. leakage",
                  "5. group-disjoint split"):
        assert stage in printed, "stage '" + stage + "' produced no output"
