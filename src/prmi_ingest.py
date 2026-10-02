"""PRMI ingestion: verify the archive, parse the filename schema, build leakage-safe splits.

Runs against `data/prmi/PRMI_official.zip`. Dryad sits behind a proof-of-work bot check so the file
has to be fetched with a browser -- `data/prmi/PROVENANCE.md` has the link and the published
SHA-256. Everything here is tested against synthetic fixtures (`tests/test_prmi_ingest.py`) so it is
known to work before the 9.75 GB lands.

THE FILENAME SCHEMA
-------------------
    Peanut_T024_L051_2017.07.13_103521_EED_DPI120.jpg
    ^^^^^^ ^^^^ ^^^^ ^^^^^^^^^^ ^^^^^^ ^^^ ^^^^^^
    species tube loc  date       time   code resolution

`tube` is the grouping unit that matters: a minirhizotron tube is imaged repeatedly down its length
(`L###`) and across sessions (`date`), so frames from one tube are near-duplicates of each other.

A CORRECTION TO THIS PROJECT'S BRIEF
------------------------------------
The brief says: "Gotcha: tube/date near-duplicates leak random splits. Split by tube, not by image."
The first half is right. The implication that PRMI's *shipped* split is leaky is NOT: PRMI already
splits by tube, roughly 60/20/20 of tubes, with switchgrass held entirely in test.

So the honest framing is not "PRMI leaks" -- it does not -- but "a random re-split of PRMI leaks,
and re-splitting is what people do when they combine datasets or run their own cross-validation."
`leakage_report` quantifies exactly that, by measuring the near-duplicate structure the official
split is protecting against. Project 06 published an accusation against SurvPath that turned out to
be false on inspection; this is the same failure mode, caught before rather than after.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# isort: off
import resources  # noqa: F401,E402  MUST load before numpy: caps BLAS threads
# isort: on

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import zipfile  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402

import numpy as np  # noqa: E402

PRMI_ZIP_SHA256 = "ce5059f92b039855bbcb5d28e346aec6bf9b26db642ab66f2c9851df4d92de5a"
PRMI_ZIP_BYTES = 9748189339
PRMI_DOI = "10.5061/dryad.2v6wwpzp4"
PRMI_LICENCE = "CC0-1.0"

SPECIES = ("cotton", "papaya", "peanut", "sesame", "sunflower", "switchgrass")

# The four filename shapes that actually occur, measured over all 136,511 files:
#
#   Peanut_T001_L001_2017.06.08_082034_EED_DPI120.jpg            image           60,031
#   GT_Peanut_T001_L001_2017.06.08_082034_EED_DPI120.png         mask            60,031
#   Switchgrass_T111_L2_2018.11.06_110530_CLMB_DPI300_1_1.png    image, tiled    12,537
#   GT_Switchgrass_T111_L2_2018.11.06_110530_CLMB_DPI300_1_1.png mask,  tiled     3,912
#
# Three things the first version of this pattern got wrong, all found by running it on the real
# archive rather than on the one example filename the documentation shows:
#   * masks carry a `GT_` prefix, so every mask failed to parse;
#   * Switchgrass frames are TILED and carry a trailing `_<row>_<col>`;
#   * the location field is variable width (`L2` as well as `L001`).
# The original pattern matched 44.0% of files. A parse rate that low is exactly what the guard in
# `index_names` exists to catch.
NAME_RE = re.compile(
    r"^(?P<gt>GT_)?"
    r"(?P<species>[A-Za-z]+)"
    r"_T(?P<tube>[A-Za-z0-9]+)"
    r"_L(?P<loc>[A-Za-z0-9]+)"
    r"_(?P<date>\d{4}[.\-_]\d{2}[.\-_]\d{2})"
    r"(?:_(?P<time>\d{4,6}))?"
    r"(?:_(?P<code>[A-Za-z]+))?"
    r"(?:_DPI(?P<dpi>\d+))?"
    r"(?:_(?P<tile_r>\d+)_(?P<tile_c>\d+))?"
    r"\.(?P<ext>jpg|jpeg|png|tif|tiff)$",
    re.IGNORECASE,
)

# `Peanut_640x480_DPI120` -- the species folder encodes the imaging configuration, not just the
# species. Peanut and Sesame each appear at TWO resolutions with different operator codes, so
# species and device are confounded in this dataset by construction.
FOLDER_RE = re.compile(
    r"^(?P<species>[A-Za-z]+)_(?P<width>\d+)x(?P<height>\d+)_DPI(?P<dpi>\d+)(?P<nomask>_noMask)?$"
)

OFFICIAL_SPLITS = ("train", "val", "test")
IMAGE_DIR, MASK_DIR = "images", "masks_pixel_gt"

IMAGE_EXT = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


class SchemaError(RuntimeError):
    """Raised when the archive does not look like PRMI. Loud, per the project's guard rule."""


def parse_name(path):
    """Parse one PRMI filename into its metadata fields, or return None if it does not match."""
    m = NAME_RE.match(os.path.basename(path))
    if not m:
        return None
    d = m.groupdict()
    d["species"] = d["species"].lower()
    d["date"] = re.sub(r"[.\-_]", "-", d["date"])
    d["is_gt_name"] = bool(d.pop("gt"))
    # Switchgrass frames are tiled; the tile indices are part of the frame's identity, not noise.
    d["tile"] = (f"{d['tile_r']}_{d['tile_c']}" if d.get("tile_r") else None)
    # The grouping key. Tube ids repeat across species (every species has a T001), so the group is
    # (species, tube) and NOT the tube alone -- collapsing them would merge unrelated tubes and
    # make a leaky split look clean.
    d["group"] = f"{d['species']}/T{d['tube']}"
    d["session"] = f"{d['group']}/{d['date']}"
    return d


def parse_folder(name):
    """Parse a species folder like `Peanut_640x480_DPI120` or `Switchgrass_720x510_DPI300_noMask`."""
    m = FOLDER_RE.match(name)
    if not m:
        return None
    d = m.groupdict()
    d["species"] = d["species"].lower()
    d["nomask"] = bool(d["nomask"])
    d["width"], d["height"], d["dpi"] = int(d["width"]), int(d["height"]), int(d["dpi"])
    # The imaging configuration, which is what actually varies. Traits are measured in PIXELS, so
    # two configurations at different DPI are not on a common scale.
    d["config"] = f"{d['species']}_{d['width']}x{d['height']}_DPI{d['dpi']}"
    return d


def verify_zip(path, expect_sha=PRMI_ZIP_SHA256, expect_bytes=PRMI_ZIP_BYTES, quick=False):
    """Check the archive against Dryad's published size and SHA-256 before anything reads it.

    Project 02's F41 found the polyp benchmark archives shipped no checksum, no version and no
    manifest, and had been silently amended once -- so no published number could say which version
    produced it. PRMI publishes both. Refusing to proceed on a mismatch is the whole point of
    that being available.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Dryad is behind a proof-of-work bot check, so download it in a "
            f"browser from https://datadryad.org/dataset/doi:{PRMI_DOI} -- see data/prmi/PROVENANCE.md"

        )
    size = os.path.getsize(path)
    out = {"path": path, "size_bytes": size, "size_ok": size == expect_bytes}
    if not out["size_ok"]:
        raise SchemaError(
            f"size mismatch: {size} bytes on disk, {expect_bytes} published by Dryad. The download is incomplete "
            "or this is a different release."
        )
    if quick:
        out["sha256"] = None
        out["sha256_ok"] = None
        return out
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    out["sha256"] = h.hexdigest()
    out["sha256_ok"] = out["sha256"] == expect_sha
    if not out["sha256_ok"]:
        raise SchemaError(
            "SHA-256 mismatch.\n  got      {}\n  expected {}\nThe archive is corrupt or is not "
            "the version this project was written against.".format(out["sha256"], expect_sha)
        )
    return out


def index_archive(zip_path, limit=None):
    """List a zip archive and parse every image filename. Does not extract anything."""
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if not n.endswith("/")]
    return index_names(names, limit=limit)


def index_directory(root, limit=None):
    """Index an EXTRACTED PRMI tree, capturing what the directory layout encodes.

    The archive's structure carries three things the filenames do not:

      * `train/`, `val/`, `test/` -- PRMI's OFFICIAL split, which is a directory layout rather than
        a metadata column. This is what makes the split tube-disjoint already.
      * `images/` vs `masks_pixel_gt/` -- authoritative, unlike guessing from the filename.
      * `Peanut_640x480_DPI120` -- the imaging configuration. Peanut and Sesame each appear at two
        resolutions, so species and device are confounded in this dataset.

    A `_noMask` folder holds images with no annotation and is flagged rather than silently dropped.
    """
    rows = []
    for split in OFFICIAL_SPLITS:
        for kind in (IMAGE_DIR, MASK_DIR):
            base = os.path.join(root, split, kind)
            if not os.path.isdir(base):
                continue
            for folder in sorted(os.listdir(base)):
                d = os.path.join(base, folder)
                if not os.path.isdir(d):
                    continue
                meta = parse_folder(folder)
                if meta is None:
                    raise SchemaError(
                        f"unrecognised species folder {folder!r} under {split}/{kind}. The layout "
                        "is not what this code was written for."
                    )
                for fn in os.listdir(d):
                    rows.append((os.path.join(split, kind, folder, fn), split, kind, meta))
                    if limit and len(rows) >= limit:
                        break
    return index_rows(rows)


def index_rows(rows):
    """Shared indexer for (relpath, split, kind, folder_meta) tuples."""
    records, unparsed, non_image = [], [], []
    for rel, split, kind, meta in rows:
        if not rel.lower().endswith(IMAGE_EXT):
            non_image.append(rel)
            continue
        d = parse_name(rel)
        if d is None:
            unparsed.append(rel)
            continue
        d["path"] = rel
        d["official_split"] = split
        d["is_mask"] = (kind == MASK_DIR)
        d["config"] = meta["config"]
        d["width"], d["height"], d["dpi_folder"] = meta["width"], meta["height"], meta["dpi"]
        d["no_mask_folder"] = meta["nomask"]
        records.append(d)

    n_img = len(records) + len(unparsed)
    report = _build_report(records, unparsed, non_image, n_img, len(rows))
    # String keys: this dict is written straight to JSON, and a tuple key raises there rather than
    # here, i.e. after the whole index has been built.
    report["official_split_sizes"] = dict(Counter(
        f"{r['official_split']}/{'mask' if r['is_mask'] else 'image'}" for r in records
    ).most_common()) if records else {}
    _check_parse_rate(report, n_img, unparsed)
    return records, report


def index_names(names, limit=None):
    """Index a flat list of paths (zip entries or relative paths), inferring split/kind from them."""
    rows = []
    for n in names:
        parts = n.replace("\\", "/").split("/")
        split = next((p for p in parts if p in OFFICIAL_SPLITS), None)
        kind = MASK_DIR if _looks_like_mask(n) else IMAGE_DIR
        meta = None
        for p in parts:
            meta = parse_folder(p)
            if meta:
                break
        if meta is None:
            meta = {"config": None, "width": None, "height": None, "dpi": None, "nomask": False}
        rows.append((n, split, kind, meta))
        if limit and len(rows) >= limit:
            break
    return index_rows(rows)


def _build_report(records, unparsed, non_image, n_img, n_entries):
    return {
        "n_entries": n_entries,
        "n_images": n_img,
        "n_parsed": len(records),
        "n_unparsed": len(unparsed),
        "n_non_image": len(non_image),
        "parse_rate": (len(records) / n_img) if n_img else 0.0,
        "unparsed_examples": unparsed[:10],
        "species": dict(Counter(r["species"] for r in records)),
        "configs": dict(Counter(r["config"] for r in records if r.get("config"))),
        "n_groups": len({r["group"] for r in records}),
        "n_sessions": len({r["session"] for r in records}),
        "n_tiled": sum(1 for r in records if r.get("tile")),
        "n_in_nomask_folder": sum(1 for r in records if r.get("no_mask_folder")),
    }


def _check_parse_rate(report, n_img, unparsed):
    if n_img and report["parse_rate"] < 0.5:
        raise SchemaError(
            "only {:.1%} of {} image filenames matched the PRMI schema. The layout is not what "
            "this code was written for. Examples that failed: {}".format(
                report["parse_rate"], n_img, unparsed[:5])
        )


def _looks_like_mask(path):
    """Fallback used only when the directory layout is unavailable (e.g. a flat zip listing).

    `index_directory` decides from the `masks_pixel_gt/` folder instead, which is authoritative.
    PRMI masks are additionally `GT_`-prefixed, so that is checked on the basename.
    """
    p = path.replace("\\", "/").lower()
    if os.path.basename(p).startswith("gt_"):
        return True
    return any(k in p for k in ("/mask", "mask/", "_mask", "label", "annotation", "gt/"))


def group_disjoint_split(records, fractions=(0.6, 0.2, 0.2), seed=0, group_key="group"):
    """Assign whole groups (species/tube) to splits. No group appears in two splits, ever.

    Greedy longest-first assignment: repeatedly give the largest remaining group to whichever split
    is furthest below its quota. That balances sizes far better than random assignment when group
    sizes are skewed, which they are -- project 06 measured 1.00-1.02x imbalance from the same
    approach against 1.16-1.42x for stratification.
    """
    by_group = defaultdict(list)
    for r in records:
        by_group[r[group_key]].append(r)
    order = sorted(by_group, key=lambda g: (-len(by_group[g]), g))
    total = len(records)
    names = ["train", "val", "test"]
    quota = [f * total for f in fractions]
    have = [0.0, 0.0, 0.0]
    assign = {}
    for g in order:
        deficit = [quota[i] - have[i] for i in range(3)]
        i = int(np.argmax(deficit))
        assign[g] = names[i]
        have[i] += len(by_group[g])
    for r in records:
        r["split"] = assign[r[group_key]]
    sizes = Counter(r["split"] for r in records)
    return assign, {
        "n_groups": len(order),
        "sizes": dict(sizes),
        "group_counts": dict(Counter(assign.values())),
        "imbalance": (max(sizes.values()) / min(sizes.values())) if len(sizes) == 3 else None,
    }


def random_split(records, fractions=(0.6, 0.2, 0.2), seed=0):
    """A naive per-IMAGE random split -- what people do when they re-split a dataset."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(records))
    n = len(records)
    a, b = int(fractions[0] * n), int((fractions[0] + fractions[1]) * n)
    out = {}
    for rank, i in enumerate(idx):
        out[records[i]["path"]] = "train" if rank < a else ("val" if rank < b else "test")
    return out


def leakage_report(records, seed=0):
    """How much group structure a random re-split destroys.

    Reports the fraction of validation images whose group (species/tube) is also represented in
    train, under a random split and under a group-disjoint one. The group-disjoint number must be
    exactly 0.0; if it is not, the split builder is broken and the guard below says so.
    """
    rnd = random_split(records, seed=seed)
    by_split = defaultdict(set)
    for r in records:
        by_split[rnd[r["path"]]].add(r["group"])
    train_groups = by_split["train"]
    val_recs = [r for r in records if rnd[r["path"]] == "val"]
    rnd_leak = (sum(r["group"] in train_groups for r in val_recs) / len(val_recs)) if val_recs else 0.0

    recs2 = [dict(r) for r in records]
    group_disjoint_split(recs2, seed=seed)
    tg = {r["group"] for r in recs2 if r["split"] == "train"}
    vr = [r for r in recs2 if r["split"] == "val"]
    grp_leak = (sum(r["group"] in tg for r in vr) / len(vr)) if vr else 0.0
    if grp_leak != 0.0:
        raise SchemaError(
            f"group-disjoint split leaked {grp_leak:.4f} of validation images -- the split builder is "
            "broken"
        )

    sess = defaultdict(set)
    for r in records:
        sess[r["group"]].add(r["date"])
    return {
        "n_images": len(records),
        "n_groups": len({r["group"] for r in records}),
        "n_sessions": len({r["session"] for r in records}),
        "median_images_per_group": float(np.median(list(Counter(
            r["group"] for r in records).values()))),
        "median_sessions_per_group": float(np.median([len(v) for v in sess.values()])),
        "random_split_val_group_overlap": rnd_leak,
        "group_disjoint_val_group_overlap": grp_leak,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default="data/prmi/PRMI_official.zip")
    ap.add_argument("--dir", default=None,
                    help="an EXTRACTED PRMI tree (the folder containing train/ val/ test/). Use "
                         "this when the zip has been unpacked and deleted -- the checksum cannot "
                         "then be re-verified, and the report says so rather than implying it was.")
    ap.add_argument("--verify", action="store_true", help="checksum the archive and stop")
    ap.add_argument("--quick", action="store_true", help="skip the SHA-256 (size check only)")
    ap.add_argument("--audit-sample", type=int, default=200,
                    help="how many masks to actually load and check")
    ap.add_argument("--mask-threshold", type=int, default=None,
                    help="threshold greyscale masks at this value; omit to require binary masks")
    ap.add_argument("--out", default="results/prmi_index.json")
    a = ap.parse_args()

    if a.dir:
        # The archive was unpacked and removed, so the published SHA-256 cannot be checked against
        # anything. Say that plainly instead of reporting a verification that did not happen: the
        # whole reason PRMI's checksum is worth having (project 02's F41) is that it lets a paper
        # name the exact bytes it used. An extracted tree cannot make that claim.
        info = {"source": "extracted directory", "path": a.dir, "sha256": None,
                "sha256_ok": None,
                "note": "zip not present; the published SHA-256 was NOT verified against these "
                        "files. Re-download the archive to assert byte-level provenance."}
        print(f"source: extracted tree at {a.dir}")
        print("  NOTE: the zip is absent, so the published SHA-256 could not be verified.")
    else:
        info = verify_zip(a.zip, quick=a.quick)
        info["source"] = "zip"
        print("archive: {} bytes, sha256 {}".format(
            info["size_bytes"], "OK" if info["sha256_ok"] else "not checked"))
    if a.verify:
        return 0

    print("\n=== 1. index ===")
    records, report = index_directory(a.dir) if a.dir else index_archive(a.zip)
    print(json.dumps(report, indent=2)[:2500])

    # Pairing and the mask audit come BEFORE the splits on purpose. A split built on records whose
    # masks turn out to be inverted, greyscale or mismatched is a tidy answer to the wrong question,
    # and both failures are invisible until something loads an actual array.
    from prmi_load import audit_masks, pair_images_and_masks  # noqa: PLC0415 - optional dep

    print("\n=== 2. image/mask pairing ===")
    pairs, pair_rep, _unpaired_img, _unpaired_mask = pair_images_and_masks(records)
    print(json.dumps(pair_rep, indent=2))

    print(f"\n=== 3. mask audit (sample of {a.audit_sample}) ===")
    audit = audit_masks(a.dir or a.zip, pairs, limit=a.audit_sample, threshold=a.mask_threshold)
    print(json.dumps(audit, indent=2)[:2500])

    print("\n=== 4. leakage ===")
    leak = leakage_report(records)
    print(json.dumps(leak, indent=2))

    print("\n=== 5. group-disjoint split ===")
    _, split_info = group_disjoint_split(records)
    print(json.dumps(split_info, indent=2))

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump({"archive": info, "index": report, "pairing": pair_rep, "mask_audit": audit,
                   "leakage": leak, "split": split_info}, fh, indent=2)
    print("\nwrote " + a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
