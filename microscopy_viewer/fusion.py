"""Turn a line drawn along a junction into a band-shaped region beside it.

Two organoids fused into an assembloid meet along an interface, and what is
measured is what has crossed it: the cells in a band of fixed width on one side
of that line. This module builds the band. It is the geometry of the Fiji macro
``Flip_and_duplicate_assembloids`` and nothing else of it — the spline through
the drawn line, the band offset from it, and the cleaned mask. Thresholding,
counting and the QC dialogs are what the rest of the viewer already does with a
label map.

The steps, in the macro's order:

1. **Spline.** The clicked vertices are joined by a smooth curve
   (:func:`fit_spline`), as ImageJ's *Fit Spline* does, so the band has no
   corners where the mouse was clicked.
2. **Normals.** At every point of the curve, the direction perpendicular to it
   (:func:`normals`), averaged over the two segments that meet there.
3. **Band.** The curve pushed sideways by a small gap gives the inner edge and
   pushed by gap + width the outer edge. The outer corners are extended along
   the line by the cap length, which splays the ends outwards
   (:func:`band_polygon`).
4. **Mask.** The polygon is rasterised, dilated once and has its holes filled
   (:func:`band_mask`). The last two are not cosmetic: on the inside of a bend
   the outer edge crosses itself, and an even-odd fill leaves the doubly covered
   part empty.

Everything is in world micrometres, in napari's ``(y, x)`` order, until the
mask; no Qt and no napari here.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .utils import get_logger

logger = get_logger("fusion")

#: The macro's defaults: a 300 µm band starting 2 µm off the line.
DEFAULT_WIDTH_UM = 300.0
DEFAULT_GAP_UM = 2.0

#: Which side of the line the band goes. "Left" and "right" are as seen walking
#: along the line from its first point to its last.
SIDE_LEFT = "Left of the line"
SIDE_RIGHT = "Right of the line"
SIDE_BOTH = "Both sides"
SIDES = (SIDE_LEFT, SIDE_RIGHT, SIDE_BOTH)

#: Points the fitted curve is sampled at, whatever its length.
MIN_SPLINE_POINTS = 50
MAX_SPLINE_POINTS = 1000

#: Key the automatic run stores its region under in the sample.
DEFAULT_KEY = "fusion region"
#: Key a region drawn (or corrected) by hand is stored under, so it never
#: overwrites the automatic one and the two can be compared.
MANUAL_KEY = "manual annotated junction region"

#: How far a band that follows the tissue reaches past its outline, in µm:
#: cells migrating over the organoid's surface sit just outside the outline the
#: nuclear stain gives.
TISSUE_MARGIN_UM = 10.0


def region_keys(key: str) -> tuple[str, ...]:
    """The stored regions to try for *key*, in order.

    Asking for the hand-drawn region falls back to the automatic one, so a
    folder where only some samples were redrawn is counted whole: the redrawn
    ones in their own region, the rest in the automatic band.
    """
    return (MANUAL_KEY, DEFAULT_KEY) if key == MANUAL_KEY else (key,)


def _clean(points: Sequence[Sequence[float]]) -> np.ndarray:
    """The line as an (N, 2) array with repeated points dropped."""
    line = np.asarray(points, dtype=float).reshape(-1, np.shape(points)[-1])[:, -2:]
    if len(line) < 2:
        return line
    keep = np.concatenate(([True], np.any(np.diff(line, axis=0) != 0, axis=1)))
    return line[keep]


def line_length(points: Sequence[Sequence[float]]) -> float:
    """Length of a polyline, in the units of its coordinates."""
    line = _clean(points)
    if len(line) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(line, axis=0), axis=1).sum())


def fit_spline(points: Sequence[Sequence[float]], spacing: float = 1.0) -> np.ndarray:
    """A smooth curve through *points*, sampled about every *spacing*.

    A natural cubic spline through every vertex, parametrised by distance along
    the drawn line — the curve passes through the clicks rather than near them.
    Two points are a straight line and come back as one, resampled.
    """
    line = _clean(points)
    if len(line) < 2:
        raise ValueError("a line needs at least two different points")
    along = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(line, axis=0), axis=1))))
    total = float(along[-1])
    count = int(np.clip(np.ceil(total / max(float(spacing), 1e-9)) + 1,
                        MIN_SPLINE_POINTS, MAX_SPLINE_POINTS))
    samples = np.linspace(0.0, total, count)
    if len(line) == 2:
        return np.column_stack([np.interp(samples, along, line[:, k]) for k in (0, 1)])

    from scipy.interpolate import CubicSpline

    return np.asarray(CubicSpline(along, line, bc_type="natural")(samples), dtype=float)


def _unit(vectors: np.ndarray) -> np.ndarray:
    lengths = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.where(lengths == 0, 1.0, lengths)


def normals(line: np.ndarray) -> np.ndarray:
    """Unit normal at every point of *line*, pointing to its left.

    The end points take the normal of their one segment; every other point the
    mean of its two, which is what keeps the offset edge from kinking at a
    vertex. "Left" is as seen walking along the line on screen, where y grows
    downwards.
    """
    line = np.asarray(line, dtype=float)
    tangent = _unit(np.diff(line, axis=0))
    # (y, x) tangent (ty, tx) turned a quarter turn: (-tx, ty).
    segment = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    out = np.empty_like(line)
    out[0], out[-1] = segment[0], segment[-1]
    if len(line) > 2:
        mean = segment[:-1] + segment[1:]
        # Two segments doubling straight back cancel; fall back on the first.
        flat = np.linalg.norm(mean, axis=1) == 0
        mean[flat] = segment[:-1][flat]
        out[1:-1] = _unit(mean)
    return out


def band_polygon(
    line: np.ndarray,
    width: float,
    gap: float = DEFAULT_GAP_UM,
    side: int = 1,
    cap: float = 0.0,
) -> np.ndarray:
    """Outline of the band beside *line*, as an (M, 2) polygon.

    *side* is +1 for the left of the line and -1 for the right. *cap* is how far
    the two outer corners are pushed out along the line; zero means the band
    width, which is the macro's default.

    Inner edge, outer end corner, outer edge backwards, outer start corner —
    the same order the macro builds its selection in.
    """
    line = np.asarray(line, dtype=float)
    if len(line) < 2:
        raise ValueError("a band needs a line of at least two points")
    if width <= 0:
        raise ValueError("the band width must be greater than zero")
    cap = float(cap) if cap and cap > 0 else float(width)
    sign = 1.0 if side >= 0 else -1.0

    normal = normals(line)
    inner = line + normal * sign * float(gap)
    outer = line + normal * sign * (float(gap) + float(width))
    start = _unit(line[1] - line[0])
    end = _unit(line[-1] - line[-2])
    return np.vstack((
        inner,
        outer[-1] + end * cap,
        outer[::-1],
        outer[0] - start * cap,
    ))


def fill_polygon(pixels: np.ndarray, shape: Sequence[int]) -> np.ndarray:
    """A polygon given in ``(row, column)`` pixel coordinates, as a boolean mask.

    Rasterised by Pillow's scanline fill. ``skimage.draw.polygon`` tests every
    pixel of the bounding box against every edge, which for a band on a slide
    scan — nine million pixels, two thousand vertices — is twenty seconds; a
    scanline fill is a few hundredths of one.
    """
    from PIL import Image, ImageDraw

    height, width = (int(n) for n in tuple(shape)[-2:])
    points = np.asarray(pixels, dtype=float)
    if len(points) < 3 or height <= 0 or width <= 0:
        return np.zeros((height, width), dtype=bool)
    canvas = Image.new("1", (width, height), 0)
    # Pillow wants (x, y), and puts pixel centres at whole numbers as napari does.
    ImageDraw.Draw(canvas).polygon([(float(c), float(r)) for r, c in points], fill=1, outline=1)
    return np.asarray(canvas, dtype=bool)


def band_mask(
    polygon: np.ndarray,
    shape_yx: Sequence[int],
    voxel_yx: Sequence[float] = (1.0, 1.0),
    translate_yx: Sequence[float] = (0.0, 0.0),
) -> np.ndarray:
    """*polygon* (world µm) as a boolean mask on one image's pixel grid.

    Rasterised, dilated by one pixel and hole-filled, as the macro cleans its
    band mask. Only the polygon's bounding box is worked on, so a band on a
    slide scan costs what the band covers and not what the slide does.
    """
    from scipy import ndimage

    shape = tuple(int(n) for n in tuple(shape_yx)[-2:])
    voxel = np.asarray(tuple(voxel_yx)[-2:], dtype=float)
    voxel = np.where(voxel == 0, 1.0, voxel)
    # Pixel centres sit at whole numbers in napari's data coordinates.
    pixels = (np.asarray(polygon, dtype=float) - np.asarray(tuple(translate_yx)[-2:], dtype=float)) / voxel

    mask = np.zeros(shape, dtype=bool)
    low = np.maximum(np.floor(pixels.min(axis=0)).astype(int) - 2, 0)
    high = np.minimum(np.ceil(pixels.max(axis=0)).astype(int) + 3, shape)
    if np.any(high <= low):
        return mask
    box = tuple(int(n) for n in (high - low))
    window = fill_polygon(pixels - low, box)
    window = ndimage.binary_dilation(window, structure=np.ones((3, 3), dtype=bool))
    window = ndimage.binary_fill_holes(window)
    mask[low[0]:high[0], low[1]:high[1]] = window
    return mask


def fusion_region(
    points: Sequence[Sequence[float]],
    shape_yx: Sequence[int],
    voxel_yx: Sequence[float] = (1.0, 1.0),
    translate_yx: Sequence[float] = (0.0, 0.0),
    width_um: float = DEFAULT_WIDTH_UM,
    gap_um: float = DEFAULT_GAP_UM,
    cap_um: float = 0.0,
    side: str = SIDE_LEFT,
    spline: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """The band(s) beside a drawn line, as a label map. Returns ``(labels, curve)``.

    *points* is the line as drawn, in world µm. The label map is 2D, on the
    grid described by *shape_yx*, *voxel_yx* and *translate_yx*: label 1 is the
    band on the left of the line and, for :data:`SIDE_BOTH`, label 2 the band
    on its right — otherwise the one band asked for is label 1. *curve* is the
    line the band was actually built from, for showing and for its length.
    """
    voxel = tuple(float(v) for v in tuple(voxel_yx)[-2:])
    drawn = _clean(points)
    if len(drawn) < 2:
        raise ValueError("draw a line of at least two points along the junction")
    # A few pixels between samples: finer adds vertices without changing a mask.
    curve = fit_spline(drawn, spacing=4.0 * max(voxel)) if spline else drawn

    signs = {SIDE_LEFT: (1,), SIDE_RIGHT: (-1,), SIDE_BOTH: (1, -1)}.get(side)
    if signs is None:
        raise ValueError(f"unknown side {side!r}; expected one of {', '.join(SIDES)}")

    labels = np.zeros(tuple(int(n) for n in tuple(shape_yx)[-2:]), dtype=np.uint8)
    for label, sign in enumerate(signs, start=1):
        polygon = band_polygon(curve, width_um, gap_um, sign, cap_um)
        mask = band_mask(polygon, labels.shape, voxel, translate_yx)
        labels[mask & (labels == 0)] = label
    return labels, curve
