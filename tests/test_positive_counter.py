"""Hand counts of positive cells: where the dots are, and what is counted from them.

Run with:  python tests/test_positive_counter.py

No Qt, no napari: only :mod:`microscopy_viewer.positive_counter`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from microscopy_viewer import cell_counter as cc  # noqa: E402
from microscopy_viewer import positive_counter as pc  # noqa: E402

_failures: list[str] = []


def check(condition: bool, message: str) -> None:
    status = "ok  " if condition else "FAIL"
    print(f"  {status} {message}")
    if not condition:
        _failures.append(message)


def _dots(shape, centres, diameter_um=6.0, voxel=(1.0, 1.0)):
    """A counter with one dot per centre, put on as the panel puts them."""
    geometry = cc.geometry_of(shape, voxel, (0.0, 0.0))
    data = np.zeros(geometry.shape, dtype=geometry.dtype())
    for label, centre in enumerate(centres, start=1):
        data[cc.dot_indices(centre, diameter_um, geometry, data)] = label
    return data


def test_names() -> None:
    print("what a counter is called")
    check(pc.key_for("AF488") == "AF488 positive manual", "a channel's counter is named after it")
    check(pc.channel_of("AF488 positive manual") == "AF488" and pc.channel_of("fusion region") is None
          and pc.channel_of(" positive manual") is None, "and the channel is read back from the name")
    check(len({pc.colour_for(i) for i in range(4)}) == 4, "each channel has a dot colour of its own")


def test_counting_in_a_region() -> None:
    print("dots, and the region")
    centres = [(20, 20), (20, 60), (70, 30), (70, 90), (95, 95)]
    data = _dots((120, 120), centres)
    labels, found = pc.dot_centres(data)
    check(list(labels) == [1, 2, 3, 4, 5], "every dot is found")
    check(np.abs(found - np.asarray(centres, dtype=float)).max() < 1.0, "where it was put, to within a pixel")

    region = np.zeros((120, 120), dtype=bool)
    region[50:, :] = True
    check(pc.count_in_region(data, region) == 3, "the dots whose centre is in the region are counted")
    check(pc.count_in_region(data, None) is None, "no region gives no count rather than zero")
    check(pc.count_in_region(np.zeros((120, 120), np.uint16), region) == 0, "an empty counter counts nothing")
    # A dot straddling the edge belongs to the side its centre is on.
    edge = _dots((120, 120), [(47, 40), (52, 80)])
    check(pc.count_in_region(edge, region) == 1, "a dot on the region's edge goes by its centre")
    stack = np.stack([np.zeros((120, 120), np.uint16), data])
    check(pc.count_in_region(stack, region) == 3, "a 3D counter is read against the region plane by plane")
    try:
        pc.count_in_region(data, np.ones((60, 60), bool))
        check(False, "a region on another grid is refused")
    except ValueError:
        check(True, "a region on another grid is refused")


def test_cells_marked_in_two_channels() -> None:
    print("the same cell in two channels")
    green = _dots((200, 200), [(30, 30), (30, 100), (150, 150)])
    red = _dots((200, 200), [(33, 32), (100, 100), (152, 149), (160, 160)])
    check(pc.shared(green, red, (1.0, 1.0), within_um=8.0) == 2, "dots within a nucleus of each other are one cell")
    check(pc.shared(green, red, (1.0, 1.0), within_um=1.0) == 0, "and not when they are further apart than asked")
    check(pc.shared(green, red, (0.5, 0.5), within_um=3.0) == 2, "the distance is in micrometres, not pixels")
    check(pc.shared(green, np.zeros_like(red), (1.0, 1.0), 8.0) == 0, "nothing is shared with an empty counter")
    check(pc.shared(red, green, (1.0, 1.0), 20.0) == 2, "each dot is matched once, however wide the reach")


def test_the_table() -> None:
    print("the table")
    region = np.zeros((120, 120), dtype=bool)
    region[50:, :] = True
    counters = {"AF555": _dots((120, 120), [(20, 20), (70, 30), (70, 90)]),
                "AF488": np.zeros((120, 120), np.uint32)}
    tallies = pc.tally(counters, region)
    check([(t.channel, t.count, t.in_region) for t in tallies] == [("AF555", 3, 2), ("AF488", 0, 0)],
          f"a count and a count in the region per channel ({[(t.channel, t.count, t.in_region) for t in tallies]})")
    frame = pc.counts_dataframe("s1.ome.tiff", tallies, region_area_mm2=0.5, dot_diameter_um=5.0, file="x")
    check(list(frame.columns) == list(pc.COLUMNS) and len(frame) == 2, "one row per channel, with its columns")
    check(frame.loc[0, "In region per mm²"] == 4.0 and frame.loc[0, "Label map"] == "AF555 positive manual",
          "with the density in the region and the name of the label map")
    bare = pc.counts_dataframe("s1", pc.tally(counters))
    check(bare["In region"].isna().all() and bare["Positive (manual)"].tolist() == [3, 0],
          "without a region the totals stand and the region columns are empty")


def main() -> int:
    for test in (test_names, test_counting_in_a_region, test_cells_marked_in_two_channels, test_the_table):
        test()
        print()
    if _failures:
        print(f"{len(_failures)} check(s) failed:")
        for failure in _failures:
            print(f"  - {failure}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
