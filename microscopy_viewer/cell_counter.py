"""Counting cells by hand: one dot per cell, in a label map the analysis can read.

A dot is a small disc (a ball, in 3D) painted with a label of its own, so the
map is an ordinary label map: one object per counted cell. Saved into the
sample's ``.ims`` like a segmentation result, it goes through the Analysis panel
the same way — counts per region, densities, the intensity under each dot.

The map covers the image at full resolution, because that is what the analysis
measures against. In 2D it is one plane over the whole stack — a dot shows on
every plane, so a cell counted while scrolling through Z is not counted again,
and the analysis reads it against the maximum projection. In 3D it has a plane
per plane of the stack, which costs memory, so it is only offered when the
stack is small enough.

No Qt here; the panel is :mod:`microscopy_viewer.widgets.cell_counter_widget`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .projection import z_axis_of

#: Name the counts are stored and shown under.
DEFAULT_KEY = "Manual counts"
#: Largest 3D counter map offered, in voxels (2 bytes each).
MAX_3D_VOXELS = 250_000_000
#: Default dot diameter, µm: about a nucleus.
DEFAULT_DIAMETER_UM = 5.0


@dataclass
class Geometry:
    """Where a counter map sits: its shape, voxel size and offset, ZYX or YX."""

    shape: tuple[int, ...]
    scale: tuple[float, ...]
    translate: tuple[float, ...]

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def voxels(self) -> int:
        return int(np.prod(self.shape))

    def dtype(self):
        # 2D maps are small, so there is room for any number of cells; a 3D map
        # is large, and 65 535 hand-placed dots is more than anyone will click.
        return np.uint32 if self.ndim == 2 else np.uint16


def geometry_of(shape: Sequence[int], scale: Sequence[float], translate: Sequence[float],
                axes: str = "", three_d: bool = False) -> Geometry:
    """The counter map for an image of *shape*: its YX, and its Z too if *three_d*.

    Every other axis (time, typically) is dropped: a count is of one stack.
    """
    shape = tuple(int(n) for n in shape)
    ndim = len(shape)
    scale = tuple(float(v) for v in scale) or (1.0,) * ndim
    translate = tuple(float(v) for v in translate) or (0.0,) * ndim
    keep = [ndim - 2, ndim - 1]
    if three_d:
        z = z_axis_of(axes, ndim) if ndim >= 3 else None
        if z is None:
            raise ValueError("This image has no Z axis to count through.")
        keep = [z, *keep]
    return Geometry(
        shape=tuple(shape[a] for a in keep),
        scale=tuple(scale[a] if a < len(scale) else 1.0 for a in keep),
        translate=tuple(translate[a] if a < len(translate) else 0.0 for a in keep),
    )


def to_data(position_world: Sequence[float], geometry: Geometry) -> np.ndarray:
    """A world position (the viewer's, any number of axes) in the map's pixel coordinates."""
    world = np.asarray(position_world, dtype=float)[-geometry.ndim:]
    return (world - np.asarray(geometry.translate)) / np.asarray(geometry.scale)


def dot_indices(centre: Sequence[float], diameter_um: float, geometry: Geometry,
                data: np.ndarray | None = None) -> tuple[np.ndarray, ...]:
    """Index arrays of the disc (ball, in 3D) a dot at *centre* covers.

    *centre* is in pixel coordinates. Pixels already holding another dot are
    left out when *data* is given, so two touching cells stay two.
    """
    radius = [max(0.5, 0.5 * float(diameter_um) / s) for s in geometry.scale]
    centre = np.asarray(centre, dtype=float)
    ranges = []
    for axis, (c, r, n) in enumerate(zip(centre, radius, geometry.shape)):
        low, high = max(0, int(np.floor(c - r))), min(n, int(np.ceil(c + r)) + 1)
        if high <= low:
            return tuple(np.empty(0, dtype=np.intp) for _ in geometry.shape)
        ranges.append(np.arange(low, high))
    grids = np.meshgrid(*ranges, indexing="ij")
    inside = sum(((g - c) / r) ** 2 for g, c, r in zip(grids, centre, radius)) <= 1.0
    if not inside.any():
        # A dot smaller than a pixel still marks the pixel it was put on.
        nearest = tuple(np.array([min(n - 1, max(0, int(round(c))))]) for c, n in zip(centre, geometry.shape))
        return nearest
    indices = tuple(g[inside] for g in grids)
    if data is not None:
        free = np.asarray(data[indices]) == 0
        indices = tuple(i[free] for i in indices)
    return indices


def label_near(data: np.ndarray, centre: Sequence[float], radius_px: float = 3.0) -> int:
    """The dot under *centre*, or the nearest one within *radius_px*; 0 if none."""
    centre = np.asarray(centre, dtype=float)
    point = tuple(int(round(c)) for c in centre)
    if all(0 <= p < n for p, n in zip(point, data.shape)):
        here = int(data[point])
        if here:
            return here
    slices = tuple(slice(max(0, int(c - radius_px)), min(n, int(c + radius_px) + 1))
                   for c, n in zip(centre, data.shape))
    window = np.asarray(data[slices])
    found = np.nonzero(window)
    if not found[0].size:
        return 0
    offsets = np.stack([f + s.start - c for f, s, c in zip(found, slices, centre)])
    nearest = int(np.argmin((offsets ** 2).sum(axis=0)))
    return int(window[tuple(f[nearest] for f in found)])


def indices_of(data: np.ndarray, label: int, centre: Sequence[float], reach_px: float) -> tuple:
    """Where *label* is, searched in a window around *centre* rather than the whole map."""
    slices = tuple(slice(max(0, int(c - reach_px)), min(n, int(c + reach_px) + 1))
                   for c, n in zip(centre, data.shape))
    found = np.nonzero(np.asarray(data[slices]) == label)
    return tuple(f + s.start for f, s in zip(found, slices))


def count(data: np.ndarray) -> int:
    """How many cells are marked: distinct labels other than 0."""
    values = np.unique(np.asarray(data))
    return int((values != 0).sum())
