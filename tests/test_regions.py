"""Checks for the region counting engine. No Qt, no display, no napari.

What is exercised is the part that decides a number: the geometry, the
world-coordinate conversion a Shapes layer goes through, and the rule that every
object belongs to exactly one region.

Run with::

    python tests/test_regions.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from microscopy_viewer import regions as rg  # noqa: E402
from microscopy_viewer import segmentation as seg  # noqa: E402

_failures: list[str] = []


def check(condition: bool, message: str) -> None:
    if condition:
        print(f"  ok   {message}")
    else:
        print(f"  FAIL {message}")
        _failures.append(message)


def _box(y0: float, y1: float, x0: float, x1: float) -> np.ndarray:
    return np.array([[y0, x0], [y0, x1], [y1, x1], [y1, x0]], dtype=float)


def _labelled(centres, shape=(2, 40, 40), radius=1) -> np.ndarray:
    """A label map with one small cube per centre, numbered from 1."""
    masks = np.zeros(shape, dtype=np.int32)
    for index, (y, x) in enumerate(centres, start=1):
        masks[:, y - radius:y + radius + 1, x - radius:x + radius + 1] = index
    return masks


def test_polygon_area() -> None:
    print("polygon area")
    check(rg.polygon_area(_box(0, 10, 0, 20)) == 200.0, "a 10x20 rectangle is 200 um^2")
    # Winding must not matter: napari gives opposite windings for the same
    # rectangle depending on which way it was dragged.
    reversed_box = _box(0, 10, 0, 20)[::-1]
    check(rg.polygon_area(reversed_box) == 200.0, "and the same drawn the other way round")
    check(rg.polygon_area([[0.0, 0.0], [1.0, 1.0]]) == 0.0, "two points enclose nothing")


def test_containment() -> None:
    print("containment")
    outline = _box(0, 10, 0, 10)
    inside = rg.contains_points(outline, np.array([[5.0, 5.0]]))
    outside = rg.contains_points(outline, np.array([[50.0, 5.0]]))
    check(bool(inside[0]), "a point in the middle is inside")
    check(not bool(outside[0]), "a point well outside is not")
    check(rg.contains_points(outline, np.zeros((0, 2))).size == 0, "no points, no answers")


def test_shapes_reach_world_coordinates() -> None:
    """A Shapes layer's own scale is what turns pixels into micrometres."""
    print("shapes to world coordinates")
    # 4 x 4 pixels of a layer at 0.5 um/px, offset 100 um: 2 x 2 um at (100, 100).
    regions = rg.regions_from_shapes(
        [_box(0, 4, 0, 4)], ["rectangle"], ["forebrain"], scale=(0.5, 0.5), translate=(100.0, 100.0)
    )
    check(len(regions) == 1 and regions[0].name == "forebrain", "the name came through")
    check(abs(regions[0].area_um2 - 4.0) < 1e-9, f"4 px at 0.5 um/px is 4 um^2 (got {regions[0].area_um2})")
    check(
        bool(regions[0].contains(np.array([[101.0, 101.0]]))[0]),
        "and it sits where the translate puts it",
    )

    # An ellipse arrives as four bounding corners rather than an outline.
    corners = np.array([[0.0, 0.0], [0.0, 10.0], [10.0, 10.0], [10.0, 0.0]])
    ellipse = rg.regions_from_shapes([corners], ["ellipse"], ["eye"])[0]
    check(ellipse.vertices_world.shape[0] > 4, "an ellipse is expanded into an outline")
    area = ellipse.area_um2
    check(abs(area - np.pi * 25.0) / (np.pi * 25.0) < 0.01, f"with the area of a disc (got {area:.1f})")

    # An unnamed shape is still counted, under a positional name.
    unnamed = rg.regions_from_shapes([_box(0, 4, 0, 4)], ["rectangle"], [""])
    check(unnamed[0].name == "region 1", "an unnamed outline gets a positional name")

    # Too few vertices to enclose anything: dropped rather than counted as empty.
    degenerate = rg.regions_from_shapes([np.array([[0.0, 0.0], [1.0, 1.0]])], ["line"], ["x"])
    check(degenerate == [], "a line is not a region")


def test_every_object_lands_in_one_region() -> None:
    print("assignment")
    masks = _labelled([(6, 6), (6, 30), (30, 30)])
    stats = seg.object_table(masks, voxel_size_um=(1.0, 1.0, 1.0))
    left = rg.Region("left", _box(0, 40, 0, 20))
    right = rg.Region("right", _box(0, 40, 20, 40))

    assignment = rg.assign_objects(stats, [left, right])
    check(assignment == ["left", "right", "right"], f"objects split by centroid ({assignment})")

    counts = rg.count_objects(stats, [left, right])
    check([count.n_objects for count in counts] == [1, 2], "and the counts follow")
    check(
        sum(count.n_objects for count in counts) == len(stats),
        "every object counted exactly once",
    )

    # Empty regions still get a row: a count of zero is a result.
    counts = rg.count_objects(stats, [left, right, rg.Region("tail", _box(100, 140, 100, 140))])
    check(len(counts) == 3 and counts[2].n_objects == 0, "an empty region is still reported")


def test_objects_outside_are_reported() -> None:
    print("objects outside every region")
    masks = _labelled([(6, 6), (36, 36)])
    stats = seg.object_table(masks, voxel_size_um=(1.0, 1.0, 1.0))
    only_left = rg.Region("left", _box(0, 20, 0, 20))

    counts = rg.count_objects(stats, [only_left])
    names = [count.region for count in counts]
    check(rg.UNASSIGNED in names, f"the leftover row is there ({names})")
    check(counts[names.index(rg.UNASSIGNED)].n_objects == 1, "and it has the missing object")

    hidden = rg.count_objects(stats, [only_left], include_unassigned=False)
    check(len(hidden) == 1, "and it can be suppressed when asked")


def test_overlaps_resolve_to_the_first_region() -> None:
    """Hand-drawn outlines overlap. Double-counting would be worse than a rule."""
    print("overlapping regions")
    masks = _labelled([(20, 20)])
    stats = seg.object_table(masks, voxel_size_um=(1.0, 1.0, 1.0))
    first = rg.Region("first", _box(0, 40, 0, 40))
    second = rg.Region("second", _box(10, 30, 10, 30))

    counts = rg.count_objects(stats, [first, second])
    check([count.n_objects for count in counts] == [1, 0], "the earlier region takes it")
    check(
        [count.n_objects for count in rg.count_objects(stats, [second, first])] == [1, 0],
        "and the order is what decides, not the size",
    )
    pairs = rg.overlapping_pairs([first, second])
    check(pairs == [("first", "second")], f"the overlap is reported so it can be said ({pairs})")
    check(rg.overlapping_pairs([first, rg.Region("far", _box(500, 540, 500, 540))]) == [],
          "and separate regions are not reported")


def test_the_label_layers_offset_is_used() -> None:
    """Centroids are relative to the label map; regions are in world space."""
    print("label layer translate")
    masks = _labelled([(6, 6)])
    stats = seg.object_table(masks, voxel_size_um=(1.0, 1.0, 1.0))
    far = rg.Region("far", _box(1000, 1040, 1000, 1040))

    check(rg.assign_objects(stats, [far]) == [rg.UNASSIGNED], "without the offset it is outside")
    check(
        rg.assign_objects(stats, [far], translate=(0.0, 1000.0, 1000.0)) == ["far"],
        "with the layer's translate it is inside",
    )


def test_density_and_the_tables() -> None:
    print("summaries and tables")
    masks = _labelled([(6, 6), (10, 10)])
    signal = np.full(masks.shape, 7.0, dtype=np.float32)
    stats = seg.object_table(masks, signal, voxel_size_um=(1.0, 1.0, 1.0))
    region = rg.Region("forebrain", _box(0, 1000, 0, 1000))  # 1 mm^2 exactly

    count = rg.count_objects(stats, [region])[0]
    check(abs(count.area_um2 - 1e6) < 1e-6, "a 1000 x 1000 um region is 1 mm^2")
    check(abs(count.density_per_mm2 - 2.0) < 1e-9, f"two objects in it is 2/mm^2 (got {count.density_per_mm2})")
    check(abs(count.mean_intensity - 7.0) < 1e-6, "intensities come through")

    frame = rg.counts_dataframe([count])
    check(list(frame.columns)[:2] == ["Region", "Objects"], f"headers are readable ({list(frame.columns)[:2]})")
    check(frame["Objects"].iloc[0] == 2, "and carry the numbers")

    objects = rg.objects_dataframe(stats, [region])
    check("Region" in objects.columns, "the per-object table gains a region column")
    check(set(objects["Region"]) == {"forebrain"}, "filled in for every object")
    check(
        "Volume (µm³)" in objects.columns,
        f"3D labels are labelled as volumes ({[c for c in objects.columns if 'µm' in c][:2]})",
    )

    flat = rg.objects_dataframe(stats, [region], ndim=2)
    check("Area (µm²)" in flat.columns, "and 2D labels as areas when told so")


def test_nothing_to_count() -> None:
    print("empty inputs")
    check(rg.assign_objects([], [rg.Region("a", _box(0, 10, 0, 10))]) == [], "no objects, no rows")
    check(rg.count_objects([], []) == [], "no regions, no counts")
    masks = _labelled([(6, 6)])
    stats = seg.object_table(masks, voxel_size_um=(1.0, 1.0, 1.0))
    check(rg.assign_objects(stats, []) == [rg.UNASSIGNED], "no regions leaves everything outside")


def main() -> int:
    for test in (
        test_polygon_area,
        test_containment,
        test_shapes_reach_world_coordinates,
        test_every_object_lands_in_one_region,
        test_objects_outside_are_reported,
        test_overlaps_resolve_to_the_first_region,
        test_the_label_layers_offset_is_used,
        test_density_and_the_tables,
        test_nothing_to_count,
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
