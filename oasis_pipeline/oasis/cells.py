"""Per-cell helpers: label maps, a channel less its local median, robust spread, and where a cell lies.

Column names are those of the per-cell tables in ``data/segmentation/cells/``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from . import geometry as fu
from . import junction as jn
from .scan import _plane, plane  # noqa: F401  (the reader, under the name the analysis code uses)

IN_REGION_COLUMN = "In region"                               #: the cell's centre lies in the junction region
LINE_COLUMN = "Distance to the junction line (µm)"           #: signed, + on the junction region's side
BESIDE_COLUMN = "Beside the junction line"                   #: nearest a point of the line, not one of its ends
EDGE_COLUMN = "Distance to the tissue edge (µm)"             #: + inside the section's outline
FILTERED = "mean (median filter)"                            #: mean over the cell of the channel less its local median
CELL_METRICS = ("mean", "median", "SD", "CV", "P95", "max", "integrated")
MEDIAN_UM = jn.MEDIAN_UM                                     #: local median window (µm)
MEDIAN_STEP = 8                                              #: the local median is taken at 1/8 resolution


def read_labels(path: str | Path) -> np.ndarray:
    """A label map (TIFF) as int32."""
    import tifffile

    return np.asarray(tifffile.imread(path)).astype(np.int32, copy=False)


def robust_centre_and_spread(values: np.ndarray) -> tuple[float, float]:
    """Median, and the standard deviation the MAD stands for (1.4826 × MAD)."""
    data = np.asarray(values, dtype=np.float64)
    data = data[np.isfinite(data)]
    if data.size == 0:
        return float("nan"), float("nan")
    centre = float(np.median(data))
    return centre, float(np.median(np.abs(data - centre)) * 1.4826)


def coarse_median(thumbnail: np.ndarray, voxel_um: float, step: int = MEDIAN_STEP) -> np.ndarray:
    """A channel's local median over :data:`MEDIAN_UM`, from its plane at 1/*step*; still at 1/*step*."""
    from scipy import ndimage

    size = max(3, int(round(MEDIAN_UM / (float(voxel_um) * step))) | 1)
    return ndimage.median_filter(np.asarray(thumbnail, dtype=np.float32), size)


def median_at(coarse: np.ndarray, window: tuple[slice, slice], shape: Sequence[int]) -> np.ndarray:
    """:func:`coarse_median` at full resolution over *window* of an image of *shape* (linear between coarse pixels)."""
    coarse = np.asarray(coarse, dtype=np.float32)

    def weights(start: int, stop: int, size: int, n: int):
        positions = np.arange(start, stop, dtype=np.float64) * ((n - 1) / max(int(size) - 1, 1))
        low = np.clip(np.floor(positions).astype(np.int64), 0, n - 1)
        return low, np.minimum(low + 1, n - 1), (positions - low).astype(np.float32)

    low, high, w = weights(window[0].start, window[0].stop, shape[0], coarse.shape[0])
    rows = coarse[low] * (1 - w)[:, None] + coarse[high] * w[:, None]
    low, high, w = weights(window[1].start, window[1].stop, shape[1], coarse.shape[1])
    return rows[:, low] * (1 - w)[None, :] + rows[:, high] * w[None, :]


def less_local_median(image: np.ndarray, voxel_um: float) -> np.ndarray:
    """*image* less its local median, the median worked out over *image* itself."""
    image = np.asarray(image, dtype=np.float32)
    whole = (slice(0, image.shape[0]), slice(0, image.shape[1]))
    return image - median_at(coarse_median(image[::MEDIAN_STEP, ::MEDIAN_STEP], voxel_um), whole, image.shape)


def label_stats(labels: np.ndarray, image: np.ndarray, ids: Sequence[int]) -> dict[str, np.ndarray]:
    """Per-label statistics of *image* (n and :data:`CELL_METRICS`), for the labels in *ids*, in that order."""
    ids = np.asarray(ids, dtype=np.int64)
    empty = {key: np.full(len(ids), np.nan) for key in ("n", *CELL_METRICS)}
    flat = np.asarray(labels).reshape(-1)
    inside = np.flatnonzero(flat)
    if inside.size == 0 or ids.size == 0:
        empty["n"] = np.zeros(len(ids))
        return empty
    owner = flat[inside].astype(np.int64)
    values = np.asarray(image, dtype=np.float64).reshape(-1)[inside]
    order = np.lexsort((values, owner))
    owner, values = owner[order], values[order]
    first = np.searchsorted(owner, ids, side="left")
    last = np.searchsorted(owner, ids, side="right")
    count = (last - first).astype(np.float64)
    present = count > 0
    running = np.concatenate(([0.0], np.cumsum(values)))
    squares = np.concatenate(([0.0], np.cumsum(values * values)))
    total = running[last] - running[first]
    safe = np.where(present, count, 1.0)
    mean = total / safe
    spread = np.sqrt(np.maximum((squares[last] - squares[first]) / safe - mean * mean, 0.0))

    def quantile(q: float) -> np.ndarray:
        position = first + np.round((safe - 1) * q).astype(np.int64)
        return values[np.clip(position, 0, len(values) - 1)]

    out = {"n": count, "mean": mean, "median": quantile(0.5), "SD": spread,
           "CV": np.divide(spread, mean, out=np.full_like(mean, np.nan), where=mean != 0),
           "P95": quantile(0.95), "max": values[np.clip(last - 1, 0, len(values) - 1)], "integrated": total}
    for key in CELL_METRICS:
        out[key] = np.where(present, out[key], np.nan)
    return out


def place_cells(table, band: np.ndarray, line, voxel: Sequence[float], overview) -> None:
    """Add each row's (cell or object) signed distance to the junction *line* (+ on the band's side), whether it is
    beside the line, and (with *overview*) its distance to the tissue edge."""
    ys = table["Centroid Y (µm)"].to_numpy(dtype=float)
    xs = table["Centroid X (µm)"].to_numpy(dtype=float)
    rows = np.clip(np.round(ys / voxel[0]).astype(int), 0, band.shape[0] - 1)
    cols = np.clip(np.round(xs / voxel[1]).astype(int), 0, band.shape[1] - 1)
    if line is not None and len(np.asarray(line).reshape(-1, 2)) >= 2:
        raw, beside, _length = fu.signed_distance(np.column_stack([ys, xs]), line)
        inside = band[rows, cols]
        chosen = inside & beside if (inside & beside).any() else inside
        flip = np.sign(np.median(np.sign(raw[chosen]))) if chosen.any() else 1.0
        table[LINE_COLUMN] = (raw * (flip or 1.0)).astype(np.float32)
        table[BESIDE_COLUMN] = beside
    else:
        table[LINE_COLUMN] = np.float32(np.nan)
        table[BESIDE_COLUMN] = False
    if overview is not None:
        table[EDGE_COLUMN] = jn.edge_distance(overview, ys / voxel[0], xs / voxel[1])
