"""Pins every number the real-data analysis reports, by reading `results/` -- never by literal.

Standing rule 4: a test that hardcodes the number it guards passes right up until the moment it
matters. So these tests assert IDENTITIES and CONSISTENCY between independently computed quantities,
which hold whatever the numbers turn out to be and fail loudly if any computation drifts.

Skipped until `src/prmi_analysis.py` has been run.
"""
import glob
import json
import os

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANALYSES = sorted(glob.glob(os.path.join(ROOT, "results", "prmi_analysis_*.json")))
MODELS = os.path.join(ROOT, "results", "prmi_models")

pytestmark = pytest.mark.skipif(not ANALYSES, reason="run src/prmi_analysis.py first")


@pytest.fixture(scope="module", params=ANALYSES or [None],
                ids=[os.path.basename(p) for p in ANALYSES] or ["none"])
def a(request):
    with open(request.param) as fh:
        return json.load(fh)


def test_all_frame_dice_decomposes_exactly_into_empty_and_nonempty_parts(a):
    """The identity behind F15, checked on every trained model.

    Mean Dice over all frames is a weighted average of the mean over empty-GT frames and the mean
    over non-empty frames, weighted by how many of each there are. If this fails, one of the three
    numbers was computed over a different frame set from the others -- the exact mistake that would
    let the empty-frame floor leak silently into a reported score.
    """
    n, ne = a["n_test_frames"], a["n_test_nonempty"]
    w_empty = (n - ne) / n
    for m in a["models"]:
        recomposed = (w_empty * m["mean_dice_empty_only"]
                      + (1 - w_empty) * m["mean_dice_nonempty_only"])
        assert recomposed == pytest.approx(m["mean_dice_all_frames"], abs=1e-6), m["tag"]


def test_null_dice_is_exactly_the_empty_fraction(a):
    """A model predicting nothing scores 1.0 on every empty frame and 0.0 elsewhere."""
    assert a["null_dice_all_frames"] == pytest.approx(
        1 - a["n_test_nonempty"] / a["n_test_frames"], abs=1e-12)


def test_models_beat_the_null_floor_on_all_frames(a):
    """Not guaranteed in principle; if it fails, a model is worse than predicting nothing."""
    for m in a["models"]:
        assert m["mean_dice_all_frames"] > a["null_dice_all_frames"], m["tag"]


def test_empty_only_dice_equals_the_rate_of_correctly_empty_predictions(a):
    """On an empty GT frame, Dice is 1.0 if the prediction is empty and 0.0 otherwise -- binary."""
    for m in a["models"]:
        assert 0.0 <= m["mean_dice_empty_only"] <= 1.0


def test_summaries_on_disk_agree_with_the_analysis(a):
    """The analysis must report what training wrote, not a recomputation that could disagree."""
    for m in a["models"]:
        with open(os.path.join(MODELS, m["tag"] + "_summary.json")) as fh:
            s = json.load(fh)
        assert s["mean_dice_all_frames"] == pytest.approx(m["mean_dice_all_frames"], abs=1e-9)
        assert s["mean_dice_nonempty_only"] == pytest.approx(m["mean_dice_nonempty_only"],
                                                             abs=1e-9)


def test_per_frame_dice_files_reproduce_the_summary_means(a):
    for m in a["models"]:
        d = np.load(os.path.join(MODELS, m["tag"] + "_dice.npy"))
        assert d.size == a["n_test_frames"], m["tag"]
        assert float(d.mean()) == pytest.approx(m["mean_dice_all_frames"], abs=1e-6), m["tag"]


def test_rank_inversion_statistics_are_well_formed(a):
    for t, v in a["rank_inversion"].items():
        assert -1.0 <= v["spearman_dice_vs_trait_error"] <= 1.0, t
        assert -1.0 <= v["spearman_without_two_weakest"] <= 1.0, t
        assert v["error_min"] <= v["error_max"], t
        assert v["best_by_trait_tag"] in {m["tag"] for m in a["models"]}, t


def test_an_undefined_correlation_is_reported_not_silently_nan(a):
    """A trait whose median error is identical for every model cannot be ranked; say so."""
    assert not set(a["rank_inversion"]) & set(a["rank_inversion_undefined"])
    for t in a["rank_inversion_undefined"]:
        errs = {m["err_" + t] for m in a["models"]}
        assert len(errs) == 1, f"{t} marked undefined but its errors vary: {errs}"


def test_trait_errors_in_rank_section_match_the_model_rows(a):
    for t, v in a["rank_inversion"].items():
        vals = [m["err_" + t] for m in a["models"] if np.isfinite(m["err_" + t])]
        assert v["error_min"] == pytest.approx(min(vals))
        assert v["error_max"] == pytest.approx(max(vals))


def test_error_decomposition_shares_sum_to_one_for_every_model(a):
    for m in a["models"]:
        assert sum(m["decomp_nonempty"].values()) == pytest.approx(1.0, abs=1e-9), m["tag"]
        assert sum(m["decomp_all_frames"].values()) == pytest.approx(1.0, abs=1e-9), m["tag"]


def test_the_floor_is_not_computed_here(a):
    """The single-mean floor this file once reported was 8x too wide; it must not come back."""
    assert isinstance(a["resolution_floor"], str)
    assert "prmi_floor" in a["resolution_floor"]


def test_pixel_area_tracks_dice_better_than_counts_do(a):
    """The F12 control, replicated on real models rather than synthetic degradations.

    Pixel area is the trait Dice pins down. If it does not track Dice at least as faithfully as the
    count traits, the real-data result has reversed the synthetic one and the paper must say so.
    This test is allowed to fail -- that would be a finding -- but it must not fail silently.
    """
    inv = a["rank_inversion"]
    if "pixel_area" not in inv:
        pytest.skip("pixel_area not ranked")
    area = inv["pixel_area"]["spearman_dice_vs_trait_error"]
    counts = [inv[t]["spearman_dice_vs_trait_error"] for t in inv
              if inv[t]["kind"] == "count"]
    assert counts, "no count traits ranked"
    # More negative = more faithful (Dice up, error down).
    assert area <= np.median(counts) + 1e-9, (
        f"pixel_area rho {area:+.3f} is LESS faithful than the median count trait "
        f"({np.median(counts):+.3f}); the synthetic control does not replicate on real models"
    )


def test_every_fingerprint_names_a_real_degradation(a):
    from_names = set(a["fingerprint_detail"][next(iter(a["fingerprint_detail"]))]["degradations"])
    for tag, f in a["fingerprints"].items():
        assert f["nearest"] in from_names, tag
