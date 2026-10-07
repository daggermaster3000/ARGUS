"""Checks for the fusion-region geometry: line -> spline -> band -> label map.

No GUI and no files: :mod:`microscopy_viewer.fusion` is plain numpy, and what
matters is that the band lands on the right side of the line, is the width that
was asked for in micrometres, and comes back as a clean label.

Run with::

    python tests/test_fusion.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from microscopy_viewer import fusion as fu  # noqa: E402

_failures: list[str] = []


def check(condition: bool, message: str) -> None:
    if condition:
        print(f"  ok   {message}")
    else:
        print(f"  FAIL {message}")
        _failures.append(message)


def test_the_spline_goes_through_the_clicks() -> None:
    print("fitting the spline")
    clicks = np.array([[10.0, 0.0], [30.0, 40.0], [10.0, 80.0], [30.0, 120.0]])
    curve = fu.fit_spline(clicks, spacing=1.0)
    check(len(curve) >= fu.MIN_SPLINE_POINTS, f"it is resampled finely ({len(curve)} points)")
    check(np.allclose(curve[0], clicks[0]) and np.allclose(curve[-1], clicks[-1]),
          "it starts and ends where the line does")
    nearest = [float(np.min(np.linalg.norm(curve - click, axis=1))) for click in clicks]
    check(max(nearest) < 1.0, f"and passes through every click (furthest {max(nearest):.2f})")
    steps = np.linalg.norm(np.diff(curve, axis=0), axis=1)
    check(steps.max() < 3 * steps.min() + 1e-9, "at an even spacing")

    straight = fu.fit_spline([[0.0, 0.0], [0.0, 100.0]], spacing=1.0)
    check(np.allclose(straight[:, 0], 0.0), "two points stay a straight line")
    check(fu.line_length(straight) == 100.0, "of the same length")

    doubled = fu.fit_spline([[0.0, 0.0], [0.0, 0.0], [0.0, 50.0], [20.0, 80.0]], spacing=1.0)
    check(np.isfinite(doubled).all(), "a repeated click does not break the fit")
    try:
        fu.fit_spline([[5.0, 5.0], [5.0, 5.0]])
        check(False, "a single point is refused")
    except ValueError:
        check(True, "a single point is refused")


def test_the_band_is_on_the_side_asked_for() -> None:
    print("the band beside a straight line")
    # A horizontal line drawn left to right across the middle, 1 µm pixels.
    line = [[100.0, 50.0], [100.0, 150.0]]
    shape = (200, 200)

    left, _ = fu.fusion_region(line, shape, width_um=30.0, gap_um=2.0, side=fu.SIDE_LEFT)
    rows = np.flatnonzero(left.any(axis=1))
    check(rows.max() < 100, f"left of a left-to-right line is above it (rows {rows.min()}–{rows.max()})")
    check(abs((rows.max() - rows.min() + 1) - 30) <= 3, "and the band is as wide as asked, give or take the cleanup")
    check(left.dtype == np.uint8 and set(np.unique(left)) == {0, 1}, "one band is label 1")
    check(left[100].sum() == 0, "the line itself is not in it: the gap is kept")

    right, _ = fu.fusion_region(line, shape, width_um=30.0, gap_um=2.0, side=fu.SIDE_RIGHT)
    check(np.flatnonzero(right.any(axis=1)).min() > 100, "right is below it")
    check(abs(int(right.sum()) - int(left.sum())) <= 200, "and the two sides are the same size")

    both, _ = fu.fusion_region(line, shape, width_um=30.0, gap_um=2.0, side=fu.SIDE_BOTH)
    check(set(np.unique(both)) == {0, 1, 2}, "both sides gives two labels")
    check(np.array_equal(both == 1, left == 1) and np.array_equal(both == 2, right == 1),
          "label 1 on the left, label 2 on the right")


def test_the_outer_caps_splay_the_ends() -> None:
    print("the outer caps")
    line = np.array([[100.0, 80.0], [100.0, 120.0]])
    polygon = fu.band_polygon(line, width=20.0, gap=0.0, side=1, cap=15.0)
    check(polygon.shape == (6, 2), "inner edge, end cap, outer edge, start cap")
    check(np.allclose(polygon[:2], line), "the inner edge is the line when the gap is zero")
    check(np.allclose(polygon[2], [80.0, 135.0]) and np.allclose(polygon[5], [80.0, 65.0]),
          "the outer corners are pushed out along the line by the cap length")
    default = fu.band_polygon(line, width=20.0, gap=0.0, side=1, cap=0.0)
    check(np.allclose(default[2], [80.0, 140.0]), "zero cap means the band width")

    try:
        fu.band_polygon(line, width=0.0)
        check(False, "a band of no width is refused")
    except ValueError:
        check(True, "a band of no width is refused")


def test_the_region_is_in_micrometres() -> None:
    print("calibration")
    line = [[100.0, 50.0], [100.0, 150.0]]  # µm
    coarse, _ = fu.fusion_region(line, (200, 200), (1.0, 1.0), width_um=30.0, side=fu.SIDE_LEFT)
    fine, _ = fu.fusion_region(line, (400, 400), (0.5, 0.5), width_um=30.0, side=fu.SIDE_LEFT)
    area_coarse = coarse.sum() * 1.0
    area_fine = fine.sum() * 0.25
    check(abs(area_coarse - area_fine) / area_coarse < 0.08,
          f"the same band covers the same area at two pixel sizes ({area_coarse:.0f} vs {area_fine:.0f} µm²)")

    moved, _ = fu.fusion_region(
        [[1100.0, 2050.0], [1100.0, 2150.0]], (200, 200), (1.0, 1.0), (1000.0, 2000.0),
        width_um=30.0, side=fu.SIDE_LEFT,
    )
    check(np.array_equal(moved, coarse), "an image that does not start at the origin is followed")

    outside, _ = fu.fusion_region([[-500.0, 0.0], [-500.0, 50.0]], (50, 50), width_um=10.0)
    check(not outside.any(), "a band wholly off the image is an empty label map, not an error")


def test_a_bent_junction_has_no_holes() -> None:
    print("a curved junction")
    # A tight arc: on its inside the outer edge of a wide band crosses itself.
    angles = np.linspace(0.0, np.pi, 9)
    arc = np.column_stack((200.0 - 60.0 * np.sin(angles), 200.0 - 60.0 * np.cos(angles)))
    for side in (fu.SIDE_LEFT, fu.SIDE_RIGHT):
        labels, curve = fu.fusion_region(arc, (400, 400), width_um=100.0, side=side)
        from scipy import ndimage

        mask = labels > 0
        check(mask.any(), f"{side.lower()}: a region comes back")
        check(np.array_equal(ndimage.binary_fill_holes(mask), mask), "with no hole in it")
    check(len(curve) >= fu.MIN_SPLINE_POINTS and fu.line_length(curve) > fu.line_length(arc),
          "built from the smoothed curve, which is a little longer than the chords")

    kinked, raw = fu.fusion_region(arc, (400, 400), width_um=100.0, spline=False)
    check(len(raw) == len(arc), "without the spline the line is used as drawn")


def main() -> int:
    for test in (
        test_the_spline_goes_through_the_clicks,
        test_the_band_is_on_the_side_asked_for,
        test_the_outer_caps_splay_the_ends,
        test_the_region_is_in_micrometres,
        test_a_bent_junction_has_no_holes,
    ):
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
    raise SystemExit(main())
