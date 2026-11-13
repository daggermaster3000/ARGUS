"""Distances to the junction line: a smooth curve through its points, its normals, and each point's signed distance.

The line is (y, x) in µm. A natural cubic spline passes through every point, sampled about every 2 µm; a point's
distance is to the nearest sample, positive on the curve's left (as seen walking along it on screen, y down).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

MIN_SPLINE_POINTS = 50            #: the smoothed line has at least this many samples …
MAX_SPLINE_POINTS = 1000          #: … and at most this many


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
    """A smooth curve through *points* (natural cubic spline by distance along the line), sampled about every *spacing*."""
    line = _clean(points)
    if len(line) < 2:
        raise ValueError("a line needs at least two different points")
    along = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(line, axis=0), axis=1))))
    total = float(along[-1])
    count = int(np.clip(np.ceil(total / max(float(spacing), 1e-9)) + 1, MIN_SPLINE_POINTS, MAX_SPLINE_POINTS))
    samples = np.linspace(0.0, total, count)
    if len(line) == 2:
        return np.column_stack([np.interp(samples, along, line[:, k]) for k in (0, 1)])

    from scipy.interpolate import CubicSpline

    return np.asarray(CubicSpline(along, line, bc_type="natural")(samples), dtype=float)


def _unit(vectors: np.ndarray) -> np.ndarray:
    lengths = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.where(lengths == 0, 1.0, lengths)


def normals(line: np.ndarray) -> np.ndarray:
    """Unit normal at every point of *line*, pointing to its left (y down)."""
    line = np.asarray(line, dtype=float)
    tangent = _unit(np.diff(line, axis=0))
    segment = np.column_stack((-tangent[:, 1], tangent[:, 0]))     # (ty, tx) turned a quarter turn
    out = np.empty_like(line)
    out[0], out[-1] = segment[0], segment[-1]
    if len(line) > 2:
        mean = segment[:-1] + segment[1:]
        flat = np.linalg.norm(mean, axis=1) == 0                     # two segments doubling back: the first
        mean[flat] = segment[:-1][flat]
        out[1:-1] = _unit(mean)
    return out


def signed_distance(points: Sequence[Sequence[float]], line: Sequence[Sequence[float]],
                    spacing: float = 2.0) -> tuple[np.ndarray, np.ndarray, float]:
    """``(distance, beside, length)``: each point's distance to the smoothed *line* (+ on its left), whether it is
    beside the line rather than past one of its ends, and the smoothed line's length."""
    from scipy.spatial import cKDTree

    curve = fit_spline(line, spacing)
    normal = normals(curve)
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    distance, nearest = cKDTree(curve).query(pts)
    side = np.einsum("ij,ij->i", pts - curve[nearest], normal[nearest])
    return distance * np.where(side < 0, -1.0, 1.0), (nearest > 0) & (nearest < len(curve) - 1), line_length(curve)
