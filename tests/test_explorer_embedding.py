"""The UMAP failsafe in the explorer: too few rows fall back instead of crashing."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("streamlit")
pytest.importorskip("sklearn")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "apps"))

import explorer_common as ec  # noqa: E402


@pytest.mark.parametrize("rows", [3, 4, 5])
@pytest.mark.parametrize("dims", [2, 3])
def test_a_few_samples_still_get_an_embedding(rows, dims):
    # Three samples broke UMAP's spectral start ("k >= N") and the app with it.
    matrix = np.random.default_rng(0).normal(size=(rows, 6))

    points, names, _note = ec.embed_umap(matrix, 15, 0.1, dims, 0)

    assert points.shape == (rows, dims)
    assert np.isfinite(points).all()
    assert len(names) == dims


def test_pca_stands_in_when_umap_cannot_run():
    # Two rows: UMAP fails whatever its start.
    matrix = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 3.0]])

    points, names, note = ec.embed_umap(matrix, 15, 0.1, 3, 0)

    assert points.shape == (2, 3)
    assert names[0].startswith("PC1") and names[2] == "PC3 (none)"
    assert "PCA" in note


def test_enough_rows_are_a_plain_umap():
    pytest.importorskip("umap")
    matrix = np.random.default_rng(1).normal(size=(30, 5))

    points, names, note = ec.embed_umap(matrix, 10, 0.1, 2, 0)

    assert note == ""
    assert names == ["UMAP 1", "UMAP 2"]
    assert points.shape == (30, 2)


@pytest.mark.parametrize("dims", [2, 3])
def test_the_app_runs_on_three_samples(tmp_path, dims):
    # The Neighbours slider was slider(2, 2) with three rows, which Streamlit refuses.
    pd = pytest.importorskip("pandas")
    pytest.importorskip("umap")
    from streamlit.testing.v1 import AppTest

    rng = np.random.default_rng(0)
    workbook = tmp_path / "three.xlsx"
    pd.DataFrame([
        {"Sample": f"fish{i}", "Genotype": "wt" if i % 2 else "mut", "Region": "Tel",
         "Label map": "", "Objects": int(rng.integers(10, 99)),
         "Region area (µm²)": float(rng.uniform(1e3, 1e4)),
         "Mean intensity": float(rng.uniform(1, 9))}
        for i in range(3)
    ]).to_excel(workbook, sheet_name=ec.SHEET, index=False)
    app = Path(__file__).resolve().parents[1] / "apps" / "region_explorer.py"

    at = AppTest.from_file(str(app), default_timeout=120)
    at.query_params["workbook"] = str(workbook)
    at.run()
    at.radio(key="umap-dims").set_value(dims).run()

    assert not at.exception
