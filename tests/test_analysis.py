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
        for test in (
            test_column_detection,
            test_scale_and_colours,
            test_label_colours,
            test_matching_a_layer,
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
