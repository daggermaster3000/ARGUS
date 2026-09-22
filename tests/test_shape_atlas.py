"""Checks for the mean-outline atlas. No display required.

Run with::

    python tests/test_shape_atlas.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from microscopy_viewer import shape_atlas as sa  # noqa: E402

_failures: list[str] = []


def check(condition: bool, message: str) -> None:
    if condition:
        print(f"  ok   {message}")
    else:
        print(f"  FAIL {message}")
        _failures.append(message)


def _bean(n: int = 90) -> np.ndarray:
    """A lopsided closed outline, so every turn and start point is distinct."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = 1.0 + 0.3 * np.cos(2 * t) + 0.15 * np.sin(3 * t) + 0.12 * np.cos(t)
    return np.column_stack([60 * r * np.cos(t), 150 * r * np.sin(t)])


def _move(points, scale, degrees, shift, start=0, reverse=False):
    a = np.radians(degrees)
    rotation = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    moved = scale * points @ rotation.T + np.asarray(shift)
    moved = np.roll(moved, -start, axis=0)
    return moved[::-1] if reverse else moved


def test_resample() -> None:
    print("resampling")
    square = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
    points = sa.resample_closed(square, 40)
    check(points.shape == (40, 2), "the asked-for number of points")
    steps = np.hypot(*np.diff(np.vstack([points, points[:1]]), axis=0).T)
    check(np.allclose(steps, 1.0), "evenly spaced along the edge")
    check(sa.signed_area(sa.resample_closed(square[::-1], 40)) > 0, "always the same way round")


def test_registration() -> None:
    print("registration")
    bean = _bean()
    outlines = {
        "a": _move(bean, 1.0, 0, (500, 300)),
        "b": _move(bean, 1.3, 25, (800, 100), start=17),
        "c": _move(bean, 0.8, -40, (200, 900), start=50, reverse=True),
    }
    atlas = sa.build_atlas(outlines, warp=False)
    worst = max(atlas.residual.values())
    check(worst < 1.0, f"copies of one shape register onto each other ({worst:.3f} µm)")
    size = np.mean([sa.centroid_size(sa.resample_closed(o)) for o in outlines.values()])
    check(abs(sa.centroid_size(atlas.template) - size) < 1.0, "the template has the mean size")
    check(np.allclose(atlas.template.mean(axis=0), sa.resample_closed(outlines["a"]).mean(axis=0),
                      atol=1.0), "and sits where the reference sample does")
    ratio = atlas.transforms["b"].scale * 1.3 / atlas.transforms["a"].scale
    check(abs(ratio - 1) < 0.01, f"scale undone ({ratio:.3f})")
    turn = atlas.transforms["b"].angle - atlas.transforms["a"].angle
    check(abs(turn + 25) < 2.0, f"turn undone ({turn:.1f}°)")

    # A cell at the same spot of each copy lands at the same spot of the template.
    spot = np.array([[20.0, 40.0]])
    where = [atlas.map_points(name, _move(spot, *args)) for name, args in (
        ("a", (1.0, 0, (500, 300))), ("b", (1.3, 25, (800, 100))), ("c", (0.8, -40, (200, 900))),
    )]
    spread = float(np.ptp(np.vstack(where), axis=0).max())
    check(spread < 1.0, f"cells follow their outline ({spread:.3f} µm apart)")

    rigid = sa.build_atlas(outlines, scale=False, warp=False)
    check(all(abs(t.scale - 1) < 1e-9 for t in rigid.transforms.values()), "rigid keeps sizes")

    mirrored = dict(outlines, m=outlines["a"] * [-1, 1])
    plain = sa.build_atlas(mirrored, warp=False)
    allowed = sa.build_atlas(mirrored, warp=False, reflect=True)
    check(allowed.transforms["m"].mirrored, "a mirrored mount is flipped back when allowed")
    check(allowed.residual["m"] < plain.residual["m"], "and fits better for it")


def test_warp() -> None:
    print("bending onto the template")
    bean = _bean()
    fat = bean * [1.4, 1.0]
    atlas = sa.build_atlas({"thin": bean, "fat": fat}, warp=True)
    bent = atlas.warps["fat"].apply(atlas.registered["fat"])
    check(np.abs(bent - atlas.template).max() < 1e-6, "the outline lands exactly on the template")
    moved = atlas.map_points("fat", fat * 0.5)
    inside = [sa.inside_polygon(atlas.template, np.array([x]), np.array([y]))[0, 0] for x, y in moved]
    check(all(inside), "the inside stays inside")


def test_density() -> None:
    print("density")
    square = sa.resample_closed(np.array([[0, 0], [100, 0], [100, 100], [0, 100]], float), 64)
    rng = np.random.default_rng(0)
    few = rng.uniform(20, 80, size=(50, 2))
    many = rng.uniform(20, 80, size=(500, 2))
    grid = sa.density_maps(square, {"few": few, "many": many}, sigma=5, pixel=1.0)
    check(np.isnan(grid.maps["few"][~grid.inside]).all(), "nothing outside the template")
    total = float(np.nansum(grid.maps["many"])) / 1000.0  # 1 µm² pixels
    check(abs(total - 500) < 5, f"per-area map integrates to the cell count ({total:.1f})")
    share = sa.density_maps(square, {"few": few, "many": many}, sigma=5, pixel=1.0, units=sa.SHARE)
    ratio = np.nansum(share.maps["many"]) / np.nansum(share.maps["few"])
    check(abs(ratio - 1) < 0.02, f"share maps ignore how many cells there are ({ratio:.3f})")
    mean = grid.mean_of(["few", "many"])
    check(np.allclose(mean, (grid.maps["few"] + grid.maps["many"]) / 2, equal_nan=True),
          "a group map weighs each sample the same")


def main() -> int:
    for test in (test_resample, test_registration, test_warp, test_density):
        test()
    print("all shape atlas checks passed" if not _failures else f"{len(_failures)} FAILED")
    return 1 if _failures else 0


if __name__ == "__main__":
    sys.exit(main())
