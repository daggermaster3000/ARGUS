"""Grouping the explorer apps' comparisons by genotype, a condition, or both."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("streamlit")
pytest.importorskip("plotly")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "apps"))

import explorer_common as ec  # noqa: E402


def _features():
    rows = []
    for sample, genotype, treatment in (("a", "mut", "drug"), ("b", "wt", "DMSO"),
                                        ("c", "wt", "drug"), ("d", "mut", None)):
        for region in ("Tel", "OT"):
            rows.append({"Sample": sample, "Genotype": genotype, "Treatment": treatment,
                         "Region": region, "Label map": "cells", "Objects": 10.0})
    return ec.fill_groups(pd.DataFrame(rows))


def test_conditions_are_offered_alone_and_crossed_with_genotype():
    features = _features()
    assert ec.group_columns(features) == ["Genotype", "Treatment"]
    assert ec.group_choices(features) == ["Genotype", "Treatment", "Genotype × Treatment"]


def test_a_column_that_varies_inside_a_sample_is_not_a_group():
    features = _features().assign(Note=lambda f: f["Region"] + "!")
    assert "Note" not in ec.group_columns(features)


def test_blank_conditions_become_their_own_group():
    assert set(_features()["Treatment"]) == {"drug", "DMSO", ec.NO_VALUE}


def test_crossed_groups_put_the_genotypes_side_by_side_in_each_condition():
    features = _features()
    frame, column = ec.with_group(features, features, "Genotype × Treatment")
    assert column == "Genotype × Treatment"
    assert ec.group_order(frame[column]) == ["wt · DMSO", "wt · drug", "mut · drug", "mut · (none)"]


def test_crossed_groups_keep_their_genotype_colour():
    order = ["wt · DMSO", "mut · DMSO", "wt · drug", "mut · drug"]
    colors = ec.group_colors(order)
    assert colors["wt · DMSO"] == colors["wt · drug"] != colors["mut · drug"]


def test_a_per_sample_table_gets_its_groups_from_the_features():
    features = _features()
    per_sample = pd.DataFrame({"Sample": ["a", "b"], "Genotype": ["mut", "wt"], "Share": [1.0, 2.0]})
    frame, column = ec.with_group(per_sample, features, "Treatment")
    assert column == "Treatment" and list(frame["Treatment"]) == ["drug", "DMSO"]


def test_groups_are_compared_like_genotypes():
    features = _features()
    features["Objects"] = [5.0, 6.0, 1.0, 2.0, 5.5, 6.5, 1.5, 2.5]
    frame, column = ec.with_group(features[features["Region"] == "Tel"], features, "Treatment")
    order = ec.group_order(frame[column])
    values = {g: frame.loc[frame[column] == g, "Objects"].to_numpy(float) for g in order}
    figure = ec.comparison_figure(frame, "Objects", column, "Box", "t")
    assert list(figure.layout.xaxis.ticktext) == order
    assert ec.compare_groups(values, "Welch's ANOVA")["test"]
