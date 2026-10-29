"""Positive cells that are one labelled body, counted once (cell_bodies)."""

from __future__ import annotations

import sys

import numpy as np

from microscopy_viewer import cell_bodies as cb

FAILURES: list[str] = []


def check(cond, msg):
    print(("ok   " if cond else "FAIL ") + msg)
    if not cond:
        FAILURES.append(msg)


VOXEL = 0.5  # µm a pixel


def disc(shape, centre, radius):
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    return (yy - centre[0]) ** 2 + (xx - centre[1]) ** 2 <= radius ** 2


def scene():
    """A big labelled body over three nuclei, a second body joined to it by a thin bridge
    (one nucleus), a lone body (one nucleus), and an unlabelled nucleus between them."""
    shape = (200, 300)
    stain = disc(shape, (100, 70), 40)                      # body A: 40 px = 20 µm radius
    stain |= disc(shape, (100, 190), 30)                    # body B
    stain[98:102, 105:165] = True                           # a 2 µm bridge A–B
    stain |= disc(shape, (40, 270), 12)                     # body C, apart
    nuclei = np.zeros(shape, dtype=np.int32)
    for label, centre in ((1, (85, 55)), (2, (115, 60)), (3, (100, 90)),   # in A
                          (4, (100, 190)),                                   # in B
                          (5, (40, 270)),                                    # in C
                          (6, (170, 140))):                                  # unlabelled
        nuclei[disc(shape, centre, 6)] = label
    return nuclei, stain


def test_merge():
    nuclei, stain = scene()
    objects = cb.positive_objects(nuclei, stain, [1, 2, 3, 4, 5], VOXEL)
    check(objects[1] == objects[2] == objects[3] == 1, f"three nuclei in one body are one object ({objects})")
    check(objects[4] == 4, "a body joined by a thin bridge stays its own object")
    check(objects[5] == 5, "a body apart stays its own object")
    check(cb.count_objects(objects, [1, 2, 3, 4, 5]) == 3, "five positive cells, three objects")
    check(cb.count_objects(objects, [1, 4]) == 2, "counting a subset (the region's cells)")
    only = cb.positive_objects(nuclei, stain, [1, 3], VOXEL)
    check(only == {1: 1, 3: 1}, "only the positive cells are merged, the others still seed territories")
    check(cb.positive_objects(nuclei, stain, [5], VOXEL) == {5: 5}, "a single positive cell is itself")
    check(cb.positive_objects(nuclei, np.zeros_like(stain), [1, 2], VOXEL) == {1: 1, 2: 2},
          "no stain: nothing merged")


def test_foreground():
    rng = np.random.default_rng(0)
    filtered = rng.normal(0, 10, (200, 200)).astype(np.float32)
    filtered[80:120, 80:120] += 200
    fg = cb.stain_foreground(filtered, None, VOXEL)
    check(fg[100, 100] and not fg[10, 10], "the stain is foreground, the noise is not")
    check(fg.mean() < 0.06, f"little of the noise passes ({fg.mean():.3f})")
    tissue = np.zeros((200, 200), bool)
    tissue[:, :50] = True
    check(not cb.stain_foreground(filtered, tissue, VOXEL)[100, 100], "only inside the tissue")


if __name__ == "__main__":
    test_merge()
    test_foreground()
    print("FAILED" if FAILURES else "all ok")
    sys.exit(1 if FAILURES else 0)
