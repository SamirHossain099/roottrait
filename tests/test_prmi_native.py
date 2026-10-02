"""Calibrated failures on real PRMI annotations at native resolution: what the paper says of them.

Relationships read from `results/`, not literals. Skipped until `src/prmi_native_calibrated.py`
has been run.
"""
import json
import os

import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUM = os.path.join(ROOT, "results", "prmi_native_calibrated_summary.json")
CSV = os.path.join(ROOT, "results", "prmi_native_calibrated.csv")
pytestmark = pytest.mark.skipif(not os.path.exists(SUM), reason="run prmi_native_calibrated.py")


@pytest.fixture(scope="module")
def s():
    with open(SUM) as fh:
        return json.load(fh)


def test_every_configuration_and_species_is_covered(s):
    df = pd.read_csv(CSV)
    assert s["n_configs"] == 8 and s["n_species"] == 6
    per = df.groupby("config")["mask"].nunique()
    assert per.nunique() == 1, "the same number of masks from every configuration"


def test_calibration_hits_the_target(s):
    for dose, d in s["doses"].items():
        assert abs(d["achieved_loss_mean"] - float(dose)) < 0.25 * float(dose), dose


def test_pixel_area_is_pinned_by_dice_whatever_the_failure(s):
    """At Dice 0.99 pixel area moves by about twice the loss, under every failure."""
    t = s["doses"]["0.01"]["traits"]["pixel_area"]
    assert 0.015 < t["min"] <= t["max"] < 0.03
    assert t["max"] / t["min"] < 1.5


def test_count_traits_are_not(s):
    d = s["doses"]["0.01"]
    comps = d["traits"]["n_components"]
    assert comps["min"] == 0 and comps["max"] > 10
    assert d["count_exceeds_integral_in_every_config"]


def test_dilation_hides_more_branch_point_damage_than_junction_breaks(s):
    dj = s["doses"]["0.01"]["dilate_vs_junction_branch_points"]
    assert dj["dilate_larger_in"] > 0.75 * dj["n"]
    assert dj["median_difference"] > 0
