"""Whole-brain (WB) normalisation in the explorer apps."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("streamlit")
pytest.importorskip("plotly")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "apps"))

import explorer_common as ec  # noqa: E402


def _square(sample, region, y, x, side):
    corners = [(y, x), (y, x + side), (y + side, x + side), (y + side, x)]
    return [
        {"Sample": sample, "Genotype": "wt", "Region": region, "Part": 0, "Vertex": i,
         "Y (µm)": cy, "X (µm)": cx}
        for i, (cy, cx) in enumerate(corners)
    ]


def _features(rows):
    return pd.DataFrame(
        [{"Sample": s, "Genotype": "wt", "Region": r, "Label map": "", "Region area (µm²)": a}
         for s, r, a in rows]
    )


def test_wb_is_the_union_of_the_outlines_not_their_sum():
    # A 100 x 100 lobe with a 50 x 50 nucleus drawn inside it: the brain is the
    # lobe, 10 000 µm², not 12 500.
    features = _features([("A", "Tel", 10_000.0), ("A", "Nucleus", 2_500.0)])
    outlines = pd.DataFrame(_square("A", "Tel", 0, 0, 100) + _square("A", "Nucleus", 25, 25, 50))

    frame = ec.add_whole_brain(features, outlines)

    assert frame[ec.WB_AREA].tolist() == pytest.approx([10_000, 10_000], rel=2e-3)


def test_a_drawn_wb_region_wins_and_the_sum_is_the_fallback():
    features = _features([
        ("drawn", "WB", 8_000.0), ("drawn", "Tel", 2_000.0),
        ("summed", "Tel", 3_000.0), ("summed", "OT", 1_000.0),
    ])

    frame = ec.add_whole_brain(features, pd.DataFrame())

    wb = frame.drop_duplicates("Sample").set_index("Sample")[ec.WB_AREA]
    assert wb["drawn"] == 8_000
    assert wb["summed"] == 4_000


def test_whole_image_rows_get_no_wb():
    frame = ec.add_whole_brain(_features([("bare", ec.NO_REGIONS, 5_000.0)]))
    assert np.isnan(frame[ec.WB_AREA].iloc[0])


def test_normalize_to_offers_wb_first_and_divides_row_by_row():
    frame = ec.add_whole_brain(_features([("A", "Tel", 3_000.0), ("A", "OT", 1_000.0)]))
    frame["Objects"] = [30.0, 0.0]
    numbers = ["Objects", "Region area (µm²)", ec.WB_AREA]

    assert ec.normalise_choices(numbers, "Region area (µm²)") == [
        ec.NOT_NORMALISED, ec.WB_AREA, "Objects",
    ]
    same, column = ec.normalise(frame, "Region area (µm²)", ec.NOT_NORMALISED)
    assert column == "Region area (µm²)" and same is frame

    scaled, column = ec.normalise(frame, "Region area (µm²)", ec.WB_AREA)
    assert column == "Region area (µm²) / WB area (µm²)"
    assert scaled[column].tolist() == [0.75, 0.25]

    per_object, column = ec.normalise(frame, "Region area (µm²)", "Objects")
    assert per_object[column].iloc[0] == 100
    assert np.isnan(per_object[column].iloc[1])  # no objects: no value, not infinity
