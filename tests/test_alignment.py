"""Checks for the cycle-registration score: does it tell moved from still?

No Qt and no plate — this is the arithmetic the batch runner calls, against
shifts whose size is known because this file applied them.

Run with::

    python tests/test_alignment.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from microscopy_viewer import alignment as al  # noqa: E402

_failures: list[str] = []

#: A real objective's pixel, so the µm are the µm of an actual experiment.
PIXEL_UM = 0.325
VOXEL = (1.0, PIXEL_UM, PIXEL_UM)


def check(condition: bool, message: str) -> None:
    if condition:
        print(f"  ok   {message}")
    else:
        print(f"  FAIL {message}")
        _failures.append(message)


def _has(name: str) -> bool:
    try:
        __import__(name)
        return True
    except Exception:
        return False


def _field(size: int = 256, spacing: int = 32, radius2: int = 144, seed: int = 0):
    """A field of textured nuclei, and the masks over them.

    Textured rather than flat: a correlation reads the pattern inside an
    object, and a disc of one constant value has no pattern to compare. A flat
    object is its own case, further down.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    image = rng.normal(100.0, 5.0, size=(size, size))
    masks = np.zeros((size, size), dtype=np.int32)
    centres = []
    label = 1
    for row in range(1, size // spacing):
        for col in range(1, size // spacing):
            cy, cx = row * spacing, col * spacing
            disc = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius2
            image[disc] += 800.0 + rng.normal(0.0, 120.0, size=int(disc.sum()))
            masks[disc] = label
            centres.append((label, cy, cx))
            label += 1
    return image, masks, centres


def test_a_shift_is_measured_in_microns() -> None:
    """The number a microscopist asks for first: how far did it move."""
    print("reading a known shift back")

    if not _has("skimage") or not _has("scipy"):
        print("  skip scikit-image and scipy are needed")
        return

    from scipy import ndimage as ndi

    image, masks, _centres = _field()

    still = al.align_cycle(masks, image, image.copy(), 1, VOXEL)
    check(
        abs(still.distance_um) < 1e-9,
        f"an image against itself has not moved ({still.distance_um:.4f} µm)",
    )
    check(
        abs(still.median_r - 1.0) < 1e-9,
        f"and matches itself exactly (r = {still.median_r:.4f})",
    )

    moved = ndi.shift(image, (2.0, 2.0), order=1, mode="nearest")
    two = al.align_cycle(masks, image, moved, 2, VOXEL)
    check(
        all(abs(abs(value) - 2.0 * PIXEL_UM) < 0.02 for value in two.shift_um),
        f"two pixels read back as 2 x {PIXEL_UM} µm on both axes: {two.shift_um}",
    )
    check(
        two.median_r < 0.2,
        f"and a field that moved does not match itself (r = {two.median_r:.3f})",
    )

    # Sub-pixel, which an integer answer would report as no movement at all.
    half = ndi.shift(image, (0.5, 0.0), order=1, mode="nearest")
    fine = al.align_cycle(masks, image, half, 3, VOXEL)
    check(
        abs(abs(fine.shift_um[0]) - 0.5 * PIXEL_UM) < 0.02,
        f"half a pixel is read as half a pixel, not as zero: {fine.shift_um[0]:+.4f} µm",
    )
    check(
        abs(fine.shift_um[1]) < 0.02,
        f"and the axis that did not move reads as not moving: {fine.shift_um[1]:+.4f} µm",
    )


def test_one_corner_moving_is_found_per_object() -> None:
    """The case a single number per image cannot show, and the reason for this.

    An organoid lifts at one edge; the field as a whole has not moved, so the
    shift is zero and the average correlation is fine. The objects over the
    corner are the ones whose later cycles are about different cells.
    """
    print("a field that moved only in one corner")

    if not _has("skimage") or not _has("scipy"):
        print("  skip scikit-image and scipy are needed")
        return

    from scipy import ndimage as ndi

    image, masks, centres = _field()
    patchy = image.copy()
    corner = (slice(0, 110), slice(0, 110))
    patchy[corner] = ndi.shift(image[corner], (4.0, 4.0), order=1, mode="nearest")

    outcome = al.align_cycle(masks, image, patchy, 2, VOXEL)
    check(
        outcome.distance_um < 0.05,
        f"the field as a whole has not moved ({outcome.distance_um:.3f} µm), so a "
        "shift per image would call this aligned",
    )

    inside = [outcome.r[label] for label, cy, cx in centres
              if cy < 100 and cx < 100 and label in outcome.r]
    outside = [outcome.r[label] for label, cy, cx in centres
               if not (cy < 100 and cx < 100) and label in outcome.r]
    check(
        bool(inside) and bool(outside),
        f"both groups have objects ({len(inside)} moved, {len(outside)} still)",
    )
    check(
        max(inside) < min(outside),
        f"every moved object scores below every still one "
        f"({max(inside):.3f} < {min(outside):.3f}), so one cut separates them",
    )
    check(
        float(np.median(outside)) > 0.95,
        f"the still ones match themselves (median r = {np.median(outside):.3f})",
    )

    # Which is what "keep the correctly registered regions" comes to.
    cut = 0.5
    dropped = {label for label, value in outcome.r.items() if value < cut}
    expected = {label for label, cy, cx in centres if cy < 100 and cx < 100}
    check(
        dropped == expected,
        f"a gate at r >= {cut} drops the moved corner and nothing else "
        f"({len(dropped)} dropped, {len(expected)} moved)",
    )


def test_what_cannot_be_judged_gets_no_number() -> None:
    """A confident number about nothing is worse than a blank.

    The gates read a blank as "this gate has no opinion" and abstain, which is
    the honest reading. An r of 1.0 over four pixels would instead read as
    "perfectly registered" and keep an object nothing was checked on.
    """
    print("objects there is not enough of to judge")

    if not _has("skimage"):
        print("  skip scikit-image is needed")
        return

    rng = np.random.default_rng(3)
    noise = rng.normal(0.0, 1.0, size=(32, 32))

    masks = np.zeros((32, 32), dtype=np.int32)
    masks[2:4, 2:4] = 1          # four pixels
    masks[10:20, 10:20] = 2      # a hundred
    scores, blank = al.object_correlation(masks, noise, noise.copy())
    check(1 not in scores, "a four-pixel object gets no correlation")
    check(blank == 1, f"and is counted as unjudged ({blank})")
    check(
        abs(scores.get(2, 0.0) - 1.0) < 1e-9,
        f"while the object big enough to judge is judged ({scores.get(2)})",
    )

    # No variance, so r is 0/0. numpy warns and returns NaN; this says nothing.
    flat = np.zeros((32, 32), dtype=np.int32)
    flat[5:25, 5:25] = 1
    ones = np.ones((32, 32))
    scores, blank = al.object_correlation(flat, ones, ones.copy())
    check(
        not scores and blank == 1,
        f"a saturated flat object has no correlation to give, and says so: "
        f"{scores}, {blank} blank",
    )

    check(
        np.isnan(al._pearson(np.array([1.0]), np.array([2.0]))),
        "one pixel is not a correlation",
    )
    check(
        np.isnan(al._pearson(np.array([1.0, np.nan]), np.array([2.0, 3.0]))),
        "and a column with a blank in it does not become one by ignoring the blank",
    )


def test_the_worst_cycle_is_the_one_that_counts() -> None:
    """An object that moved in one cycle moved, however good the other six are."""
    print("the worst cycle of several")

    good = al.CycleAlignment(cycle=1, r={1: 0.99, 2: 0.98, 3: 0.97})
    bad = al.CycleAlignment(cycle=2, r={1: 0.10, 2: 0.96})
    worst = al.worst_of([good, bad], [1, 2, 3, 4])

    check(worst[1] == 0.10, f"an object bad in one cycle is bad ({worst[1]})")
    check(worst[2] == 0.96, f"one good in both keeps its lowest ({worst[2]})")
    check(
        worst[3] == 0.97,
        f"one judged in only one cycle is judged on that cycle ({worst[3]})",
    )
    check(
        4 not in worst,
        "and one judged in no cycle gets no number, so a gate abstains on it",
    )

    named = al.r_column(3)
    check(named == "Registration r, cycle 3", f"the column names its cycle: {named}")
    check(
        al.worst_column([1, 2, 3]) != al.r_column(1),
        "and the worst-of column is its own, not the first cycle's",
    )


def test_two_grids_are_not_compared() -> None:
    """Refused rather than answered: the answer would look like any other."""
    print("refusing to compare two different grids")

    if not _has("skimage"):
        print("  skip scikit-image is needed")
        return

    masks = np.zeros((16, 16), dtype=np.int32)
    masks[2:10, 2:10] = 1
    try:
        al.object_correlation(masks, np.zeros((16, 16)), np.zeros((8, 8)))
        check(False, "a correlation across two grids is refused")
    except ValueError as error:
        check(
            "one grid" in str(error),
            f"a correlation across two grids is refused, in words ({error})",
        )
    try:
        al.field_shift(np.zeros((16, 16)), np.zeros((8, 8)))
        check(False, "and so is a shift")
    except ValueError as error:
        check(
            "not a shift" in str(error),
            f"and so is a shift ({error})",
        )


def main() -> int:
    for test in (
        test_a_shift_is_measured_in_microns,
        test_one_corner_moving_is_found_per_object,
        test_what_cannot_be_judged_gets_no_number,
        test_the_worst_cycle_is_the_one_that_counts,
        test_two_grids_are_not_compared,
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
