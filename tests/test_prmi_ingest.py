"""PRMI ingestion, tested against synthetic fixtures so it is known-good before the 9.75 GB lands.

The fixtures reproduce the real filename schema documented for PRMI:

    Peanut_T024_L051_2017.07.13_103521_EED_DPI120.jpg

Building a zip of a few hundred zero-byte entries costs milliseconds and exercises every path the
real archive will take: schema parsing, group derivation, split construction and the leakage
measurement. The one thing it cannot check is that the real archive matches the schema -- which is
why `index_names` raises rather than returning a mostly-empty index.
"""
import os
import sys
import zipfile

import pytest

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
sys.path.insert(0, SRC)

from prmi_ingest import (  # noqa: E402
    NAME_RE,
    SchemaError,
    group_disjoint_split,
    index_names,
    leakage_report,
    parse_name,
    random_split,
    verify_zip,
)

REAL_EXAMPLE = "Peanut_T024_L051_2017.07.13_103521_EED_DPI120.jpg"


def make_names(n_species=3, n_tubes=5, n_sessions=3, n_locs=8):
    """A PRMI-shaped corpus: several species, tubes per species, sessions and depths per tube."""
    species = ["Cotton", "Peanut", "Switchgrass", "Sesame", "Papaya", "Sunflower"][:n_species]
    names = []
    for sp in species:
        for t in range(1, n_tubes + 1):
            for s in range(n_sessions):
                date = f"2017.07.{13 + s:02d}"
                for loc in range(1, n_locs + 1):
                    stem = f"{sp}_T{t:03d}_L{loc:03d}_{date}_10352{s}_EED_DPI120"
                    names.append(f"PRMI/images/{sp.lower()}/{stem}.jpg")
                    names.append(f"PRMI/masks/{sp.lower()}/{stem}.png")
    return names


# ------------------------------------------------------------------------- schema
def test_the_documented_real_filename_parses():
    d = parse_name(REAL_EXAMPLE)
    assert d is not None, "the one real filename we have does not parse"
    assert d["species"] == "peanut"
    assert d["tube"] == "024"
    assert d["loc"] == "051"
    assert d["date"] == "2017-07-13"
    assert d["time"] == "103521"
    assert d["dpi"] == "120"
    assert d["group"] == "peanut/T024"


@pytest.mark.parametrize("name", [
    "Cotton_T001_L002_2018.05.01_090000_ABC_DPI300.jpg",
    "Sunflower_T12_L3_2019-06-02_1200_XY_DPI150.png",
    "Sesame_T007_L010_2017.08.20.jpg",                    # minimal: no time/code/dpi
])
def test_schema_variants_parse(name):
    assert parse_name(name) is not None, name


@pytest.mark.parametrize("name", [
    "README.txt", "notes.csv", "image_without_schema.jpg", "T024_L051_2017.07.13.jpg",
])
def test_non_conforming_names_are_rejected_not_guessed(name):
    assert parse_name(name) is None, name + " should not have parsed"


def test_group_is_species_plus_tube_not_tube_alone():
    """Every species has a T001. Grouping on the tube alone would merge unrelated tubes."""
    a = parse_name("Cotton_T001_L001_2017.07.13_100000_A_DPI120.jpg")
    b = parse_name("Peanut_T001_L001_2017.07.13_100000_A_DPI120.jpg")
    assert a["group"] != b["group"], "two species' T001 collapsed into one group"


def test_index_reports_what_it_could_not_parse():
    names = make_names() + ["PRMI/README", "PRMI/junk_file.jpg", "PRMI/meta.json"]
    recs, rep = index_names(names)
    assert rep["n_unparsed"] == 1 and "junk_file.jpg" in rep["unparsed_examples"][0]
    assert rep["n_non_image"] == 2
    assert rep["parse_rate"] > 0.99
    assert set(rep["species"]) == {"cotton", "peanut", "switchgrass"}


def test_a_changed_layout_raises_instead_of_returning_an_empty_index():
    """The guard rule: fail loudly. A silent empty index would produce clean-looking splits."""
    with pytest.raises(SchemaError, match="did not match|matched the PRMI schema"):
        index_names([f"weird/IMG_{i:05d}.jpg" for i in range(50)])


# ------------------------------------------------------------------------- splits
def test_group_disjoint_split_has_exactly_zero_group_overlap():
    recs, _ = index_names(make_names())
    group_disjoint_split(recs)
    by = {}
    for r in recs:
        by.setdefault(r["group"], set()).add(r["split"])
    multi = {g: s for g, s in by.items() if len(s) > 1}
    assert not multi, f"groups appearing in more than one split: {multi}"


def test_group_disjoint_split_is_reasonably_balanced():
    recs, _ = index_names(make_names(n_species=4, n_tubes=8))
    _, info = group_disjoint_split(recs)
    assert info["imbalance"] < 4.0, f"fold sizes are badly skewed: {info}"
    assert set(info["group_counts"]) == {"train", "val", "test"}


def test_split_is_deterministic():
    recs1, _ = index_names(make_names())
    recs2, _ = index_names(make_names())
    a, _ = group_disjoint_split(recs1)
    b, _ = group_disjoint_split(recs2)
    assert a == b


def test_random_split_leaks_and_group_split_does_not():
    """The number the brief asks for: how much a naive re-split leaks."""
    recs, _ = index_names(make_names(n_species=3, n_tubes=6, n_sessions=4))
    rep = leakage_report(recs)
    assert rep["group_disjoint_val_group_overlap"] == 0.0
    assert rep["random_split_val_group_overlap"] > 0.9, (
        "a random per-image split over many frames per tube should leak almost every tube; got "
        f"{rep['random_split_val_group_overlap']:.3f}"
    )


def test_random_split_covers_every_record_exactly_once():
    recs, _ = index_names(make_names())
    r = random_split(recs)
    assert len(r) == len(recs)
    assert set(r.values()) == {"train", "val", "test"}


# ------------------------------------------------------------------------- archive verification
def test_verify_zip_rejects_a_wrong_size(tmp_path):
    p = tmp_path / "PRMI_official.zip"
    with zipfile.ZipFile(p, "w") as zf:
        zf.writestr("a.txt", "hello")
    with pytest.raises(SchemaError, match="size mismatch"):
        verify_zip(str(p))


def test_verify_zip_rejects_a_wrong_checksum(tmp_path):
    p = tmp_path / "PRMI_official.zip"
    p.write_bytes(b"x")
    with pytest.raises(SchemaError, match="size mismatch"):
        verify_zip(str(p))
    # size matches, hash does not
    with pytest.raises(SchemaError, match="SHA-256 mismatch"):
        verify_zip(str(p), expect_bytes=1)


def test_verify_zip_names_the_manual_download_when_the_file_is_absent(tmp_path):
    with pytest.raises(FileNotFoundError, match="datadryad.org"):
        verify_zip(str(tmp_path / "nope.zip"))


def test_published_constants_match_the_provenance_record():
    """These came from the Dryad API on 2026-09-06 and pin the release this code targets."""
    from prmi_ingest import PRMI_LICENCE, PRMI_ZIP_BYTES, PRMI_ZIP_SHA256
    assert PRMI_ZIP_BYTES == 9748189339
    assert PRMI_ZIP_SHA256 == (
        "ce5059f92b039855bbcb5d28e346aec6bf9b26db642ab66f2c9851df4d92de5a")
    assert PRMI_LICENCE == "CC0-1.0"
    doc = os.path.join(os.path.dirname(SRC), "data", "prmi", "PROVENANCE.md")
    text = open(doc, encoding="utf-8").read()
    assert PRMI_ZIP_SHA256 in text, "PROVENANCE.md and the code disagree about the checksum"


def test_name_regex_is_anchored():
    """An unanchored pattern would match junk with a PRMI-looking substring."""
    assert NAME_RE.match("prefix_" + REAL_EXAMPLE) is None
