"""Checks for the measurement-analysis engine. No Qt, no display.

What is being checked is the part that decides what a number means: which column
holds the object id, where the colour scale starts and stops, which label gets
which colour, and which layer a table written by a batch run belongs to. The
panel is a thin layer over these.

Run with::

    python tests/test_analysis.py
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from microscopy_viewer import analysis  # noqa: E402

_failures: list[str] = []


def check(condition: bool, message: str) -> None:
    if condition:
        print(f"  ok   {message}")
    else:
        print(f"  FAIL {message}")
        _failures.append(message)


def _has_pandas() -> bool:
    try:
        import pandas  # noqa: F401
    except ImportError:
        return False
    return True


def _frame(rows: int = 5):
    import pandas as pd

    return pd.DataFrame(
        {
            "Label": list(range(1, rows + 1)),
            "Volume (µm³)": [10.0, 20.0, 30.0, 40.0, 1000.0][:rows],
            "Mean intensity": [100.0, 200.0, 300.0, 400.0, 500.0][:rows],
            "Note": ["a", "b", "c", "d", "e"][:rows],
        }
    )


# ---------------------------------------------------------------------------


def test_reading(directory: Path) -> None:
    print("reading a table")

    frame = _frame()
    csv = directory / "G_07_0.csv"
    frame.to_csv(csv, index=False, encoding="utf-8")

    read = analysis.read_table(csv)
    check(len(read) == 5, f"every row comes back ({len(read)})")
    check(
        "Volume (µm³)" in read.columns,
        f"the µm column name survives the round trip: {list(read.columns)}",
    )

    semicolons = directory / "european.csv"
    semicolons.write_text("Label;Area\n1;5\n2;6\n", encoding="utf-8")
    sniffed = analysis.read_table(semicolons)
    check(
        list(sniffed.columns) == ["Label", "Area"],
        f"a semicolon CSV is sniffed rather than read as one column: {list(sniffed.columns)}",
    )

    try:
        analysis.read_table(directory / "nothing.csv")
        check(False, "a missing file is refused")
    except FileNotFoundError:
        check(True, "a missing file is refused by name, not by traceback")

    empty = directory / "empty.csv"
    empty.write_text("Label,Area\n", encoding="utf-8")
    try:
        analysis.read_table(empty)
        check(False, "an empty table is refused")
    except ValueError as exc:
        check("no rows" in str(exc), f"an empty table says so: {exc}")


def test_column_detection() -> None:
    print("finding the label and the numbers")

    import pandas as pd

    frame = _frame()
    check(analysis.label_column(frame) == "Label", "the exported name is found")
    check(
        analysis.label_column(pd.DataFrame({"ObjectNumber": [1], "x": [2]})) == "ObjectNumber",
        "so is CellProfiler's",
    )
    check(
        analysis.label_column(pd.DataFrame({"labels": [1], "x": [2]})) == "labels",
        "and a differently-cased one",
    )
    check(
        analysis.label_column(pd.DataFrame({"x": [1.0]})) is None,
        "a table with no id column says so rather than guessing at row numbers",
    )

    numeric = analysis.numeric_columns(frame, exclude=["Label"])
    check(
        numeric == ["Volume (µm³)", "Mean intensity"],
        f"text columns are left out and the id is excluded: {numeric}",
    )


def test_scale_and_colours() -> None:
    print("the colour scale")

    values = np.array([10.0, 20.0, 30.0, 40.0, 1000.0])
    low, high = analysis.value_range(values, 1.0, 99.0)
    check(
        high < 1000.0,
        f"the outlier is clipped out of the scale ({high:.0f} rather than 1000)",
    )
    full_low, full_high = analysis.value_range(values, 0.0, 100.0)
    check(
        (full_low, full_high) == (10.0, 1000.0),
        f"0-100 % is the true range ({full_low}, {full_high})",
    )
    flat_low, flat_high = analysis.value_range(np.array([7.0, 7.0, 7.0]))
    check(
        flat_high > flat_low,
        "a column where every object is identical still gives a usable scale",
    )
    check(
        analysis.value_range(np.array([np.nan, np.nan])) == (0.0, 1.0),
        "a column of blanks falls back rather than raising",
    )

    scaled = analysis.normalise(np.array([0.0, 5.0, 10.0, 20.0]), 0.0, 10.0)
    check(
        list(scaled[:3]) == [0.0, 0.5, 1.0] and scaled[3] == 1.0,
        f"values map onto 0-1 and clip at the top ({list(scaled)})",
    )

    colors = analysis.colormap_colors([0.0, 1.0, np.nan], "viridis")
    check(colors.shape == (3, 4), f"one RGBA per value ({colors.shape})")
    check(
        not np.allclose(colors[0], colors[1]),
        "the ends of the scale are different colours",
    )
    check(tuple(colors[2]) == (0.0, 0.0, 0.0, 0.0), "a blank value is transparent")
    check(
        analysis.colormap_colors([0.5], "not-a-colormap").shape == (1, 4),
        "an unknown colormap falls back instead of raising",
    )


def test_label_colours() -> None:
    print("label -> colour")

    labels = [1, 2, 3]
    values = [0.0, 50.0, 100.0]
    mapping, (low, high) = analysis.label_colors(labels, values, "viridis", low=0.0, high=100.0)

    check((low, high) == (0.0, 100.0), "an explicit range is used as given")
    check(set(mapping) == {None, 0, 1, 2, 3}, f"one entry per label plus background: {set(mapping)}")
    check(
        mapping[None] == (0.0, 0.0, 0.0, 0.0) and mapping[0] == (0.0, 0.0, 0.0, 0.0),
        "background and any label with no row are transparent",
    )
    check(
        mapping[1] != mapping[3],
        "the smallest and largest values are not the same colour",
    )
    check(
        all(len(colour) == 4 for colour in mapping.values()),
        "every entry is RGBA, which is what DirectLabelColormap wants",
    )

    zeros, _range = analysis.label_colors([0, -1, 2], [1.0, 2.0, 3.0])
    check(
        set(zeros) == {None, 0, 2},
        f"labels at or below zero are not painted — they are background: {set(zeros)}",
    )

    try:
        analysis.label_colors([1, 2], [1.0])
        check(False, "mismatched lengths are refused")
    except ValueError as exc:
        check("same table" in str(exc), f"mismatched lengths are refused: {exc}")


def test_matching_a_layer() -> None:
    print("matching a table to a layer")

    check(analysis.component_from_name("G_07_0") == "G/07/0", "the image path is read back out")
    check(analysis.component_from_name("B_02_0") == "B/02/0", "and for another well")
    check(analysis.component_from_name("nuclei_summary") == "", "a summary table names no image")

    names = [
        "G/07 :: cycle 1 :: nuclei",
        "B/02 :: cycle 1 :: nuclei",
        "G/07 :: cycle 1 :: Ab1_DAPI",
    ]
    check(
        analysis.match_layer("G_07_0.csv", names) == "G/07 :: cycle 1 :: nuclei",
        "the table finds its own well",
    )
    check(
        analysis.match_layer("B_02_0.csv", names) == "B/02 :: cycle 1 :: nuclei",
        "and a different one",
    )
    check(analysis.match_layer("G_07_0.csv", []) is None, "an empty viewer matches nothing")
    check(
        analysis.match_layer("summary.csv", names) is None,
        "a table that names no well does not guess",
    )
    check(
        analysis.match_layer("C_11_0.csv", names) is None,
        "a well that is not open matches nothing rather than the first layer",
    )


def test_region_selection() -> None:
    print("picking a region out of the plot")

    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    y = np.array([0.0, 1.0, 2.0, 3.0, 4.0])

    inside = analysis.points_in_rectangle(x, y, 0.5, 3.5, 0.5, 3.5)
    check(list(inside) == [False, True, True, True, False], f"a rectangle selects what it covers ({list(inside)})")
    check(
        list(analysis.points_in_rectangle(x, y, 3.5, 0.5, 3.5, 0.5)) == list(inside),
        "dragged the other way it selects the same points",
    )
    check(
        not analysis.points_in_rectangle(x, y, 10.0, 11.0, 10.0, 11.0).any(),
        "a rectangle over empty space selects nothing",
    )

    square = [(0.5, 0.5), (3.5, 0.5), (3.5, 3.5), (0.5, 3.5)]
    check(
        list(analysis.points_in_polygon(x, y, square)) == list(inside),
        "a lasso round the same area selects the same points",
    )
    triangle = [(-0.5, -0.5), (1.5, -0.5), (-0.5, 1.5)]
    check(
        list(analysis.points_in_polygon(x, y, triangle)) == [True, False, False, False, False],
        "and a shape that is not a box selects only what is really inside it",
    )
    check(
        not analysis.points_in_polygon(x, y, [(0.0, 0.0), (1.0, 1.0)]).any(),
        "a lasso with no area selects nothing rather than raising",
    )
    check(
        not analysis.points_in_polygon([], [], square).any(),
        "and neither does an empty plot",
    )


def test_dimming_outside_the_selection() -> None:
    print("fading what was not selected")

    mapping = {
        None: (0.0, 0.0, 0.0, 0.0),
        0: (0.0, 0.0, 0.0, 0.0),
        1: (0.1, 0.2, 0.3, 1.0),
        2: (0.4, 0.5, 0.6, 1.0),
        3: (0.7, 0.8, 0.9, 0.5),
    }
    faded = analysis.dim_unselected(mapping, [1], alpha=0.1)

    check(faded[1] == (0.1, 0.2, 0.3, 1.0), "a selected object is untouched, colour and all")
    check(
        faded[2][:3] == (0.4, 0.5, 0.6) and abs(faded[2][3] - 0.1) < 1e-9,
        f"an unselected one keeps its colour and loses its opacity ({faded[2]})",
    )
    check(
        abs(faded[3][3] - 0.05) < 1e-9,
        f"fading is relative, so an already-faint object stays fainter ({faded[3][3]})",
    )
    check(
        faded[None] == (0.0, 0.0, 0.0, 0.0) and faded[0] == (0.0, 0.0, 0.0, 0.0),
        "background stays transparent rather than being faded twice",
    )
    check(
        analysis.dim_unselected(mapping, []) == mapping,
        "an empty selection means everything, not nothing",
    )
    check(
        analysis.dim_unselected(mapping, [1]) is not mapping,
        "the original mapping is left alone, so Clear has something to put back",
    )


def test_finding_an_object(directory: Path) -> None:
    print("going to one object")

    import pandas as pd

    frame = pd.DataFrame(
        {
            "Label": [1, 2],
            "Centroid Z (\u00b5m)": [0.0, 0.0],
            "Centroid Y (\u00b5m)": [10.0, 500.0],
            "Centroid X (\u00b5m)": [20.0, 900.0],
            "Equivalent diameter (\u00b5m)": [8.0, float("nan")],
        }
    )

    columns = analysis.centroid_columns(frame)
    check(set(columns) == {"z", "y", "x"}, f"the centroid columns are recognised ({columns})")
    check(
        analysis.object_centroid(frame, 1) == (0.0, 500.0, 900.0),
        f"a row gives its centroid in z, y, x ({analysis.object_centroid(frame, 1)})",
    )
    check(
        analysis.object_centroid(pd.DataFrame({"Label": [1]}), 0) is None,
        "a table with no centroid says so rather than guessing at the origin",
    )

    # The centroids are micrometres and a calibrated layer's world coordinates are
    # micrometres, so this is the camera position with nothing done to it.
    two_d = frame.drop(columns=["Centroid Z (\u00b5m)"])
    check(
        analysis.object_centroid(two_d, 0) == (10.0, 20.0),
        f"a 2D table gives (y, x) and does not invent a plane ({analysis.object_centroid(two_d, 0)})",
    )
    other = pd.DataFrame({"label": [1], "centroid-y": [3.0], "centroid_x": [4.0]})
    check(
        analysis.object_centroid(other, 0) == (3.0, 4.0),
        "a table written by something else is still read",
    )

    check(analysis.object_diameter(frame, 0) == 8.0, "the size comes off the table")
    check(
        analysis.object_diameter(frame, 1) == analysis.FALLBACK_DIAMETER_UM,
        "a blank size falls back rather than zooming to infinity",
    )

    zoom = analysis.zoom_for(10.0, 800.0, fill=0.25)
    check(zoom == 20.0, f"an object a quarter of an 800 px canvas is 20 px per um ({zoom})")
    check(
        analysis.zoom_for(20.0, 800.0) < analysis.zoom_for(5.0, 800.0),
        "a bigger object is zoomed out further, not in",
    )
    check(analysis.zoom_for(0.0, 800.0) > 0, "a zero size does not divide by zero")


def test_anndata_export(directory: Path) -> None:
    print("out to AnnData")

    try:
        import anndata as ad
    except ImportError:
        print("  skip anndata is not installed")
        return

    import pandas as pd

    frame = pd.DataFrame(
        {
            "Label": [1, 2, 3],
            "Voxels": [100, 200, 300],
            "Centroid Z (\u00b5m)": [0.0, 0.0, 0.0],
            "Centroid Y (\u00b5m)": [10.0, 20.0, 30.0],
            "Centroid X (\u00b5m)": [40.0, 50.0, 60.0],
            "Mean intensity": [5.0, 6.0, 7.0],
            "Note": ["a", "b", "c"],
        }
    )

    features = analysis.feature_columns(frame, "Label")
    check(
        features == ["Voxels", "Mean intensity"],
        f"measurements are features; the id, the centroids and the text are not: {features}",
    )

    path = analysis.write_anndata(frame, directory / "objects.h5ad", source="G_07_0.csv")
    check(path.exists() and path.stat().st_size > 0, f"a file is written ({path.name})")

    adata = ad.read_h5ad(path)
    check(adata.shape == (3, 2), f"one row per object, one column per feature {adata.shape}")
    check(list(adata.var_names) == features, f"the features are named: {list(adata.var_names)}")
    check(
        "Label" not in adata.var_names,
        "the label is not a feature — clustering on it would cluster on Cellpose's numbering",
    )
    check(list(adata.obs["label"]) == [1, 2, 3], "the label is kept, in obs, as an id")
    check(
        list(adata.obs_names) == ["1", "2", "3"],
        f"and names the observations, so a result joins back to the mask ({list(adata.obs_names)})",
    )
    check("Note" in adata.obs.columns, "a text column is carried through rather than dropped")

    spatial = adata.obsm["spatial"]
    check(spatial.shape == (3, 2), f"the centroid reaches obsm['spatial'] ({spatial.shape})")
    check(
        list(spatial[0]) == [40.0, 10.0],
        f"as (x, y), which is the order squidpy plots in ({list(spatial[0])})",
    )
    check(
        adata.uns["microscopy_viewer"]["image"] == "G/07/0",
        "and the image it came from is recorded, so a folder of these is still readable later",
    )
    check(
        adata.uns["microscopy_viewer"]["spatial_units"] == "micrometer",
        "with the units said out loud",
    )

    # Z is constant on a plate image; a third spatial column would make every
    # neighbour graph a 3D one built on an axis that does not vary.
    check(spatial.shape[1] == 2, "a flat image gives 2D coordinates")
    volume = frame.copy()
    volume["Centroid Z (\u00b5m)"] = [0.0, 5.0, 10.0]
    check(
        analysis.to_anndata(volume, "Label").obsm["spatial"].shape[1] == 3,
        "but a real volume gives 3D ones",
    )

    try:
        analysis.to_anndata(frame[["Label"]], "Label")
        check(False, "a table with nothing to measure is refused")
    except ValueError as exc:
        check("no numeric measurement" in str(exc), f"a table with no features is refused: {exc}")


def test_plot_helpers() -> None:
    print("plot helpers")

    check(analysis.scatter_sample(500) is None, "a small table is plotted whole")
    sample = analysis.scatter_sample(500_000, limit=1000)
    check(
        sample is not None and len(sample) == 1000 and len(set(sample.tolist())) == 1000,
        "a huge one is subsampled without repeats",
    )
    check(
        sorted(sample.tolist()) != list(range(1000)),
        "and the sample is spread over the table rather than its first rows",
    )

    text = analysis.describe_column([1.0, 2.0, 3.0, np.nan])
    check("n = 3" in text and "median 2" in text, f"the summary counts and centres: {text}")
    check("1 blank" in text, f"and says how many were blank: {text}")
    check(
        analysis.describe_column([np.nan]) == "no numeric values",
        "a column of blanks says so",
    )


def main() -> int:
    if not _has_pandas():
        print("pandas is not installed; the analysis checks need it")
        return 0

    directory = Path(tempfile.mkdtemp(prefix="mv-analysis-"))
    try:
        test_reading(directory)
        print()
        test_finding_an_object(directory)
        print()
        test_anndata_export(directory)
        print()
        for test in (
            test_column_detection,
            test_scale_and_colours,
            test_label_colours,
            test_matching_a_layer,
            test_region_selection,
            test_dimming_outside_the_selection,
            test_plot_helpers,
        ):
            test()
            print()
    finally:
        shutil.rmtree(directory, ignore_errors=True)

    if _failures:
        print(f"{len(_failures)} check(s) failed:")
        for failure in _failures:
            print(f"  - {failure}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
