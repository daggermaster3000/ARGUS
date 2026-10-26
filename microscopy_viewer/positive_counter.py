"""Counting positive cells by hand, one counter per channel.

The automatic count (:mod:`microscopy_viewer.junction_analysis`) calls a cell
positive against a threshold. This is the count a person makes instead, or to
check it: look at a channel, click every cell that is positive in it. Each
channel has a counter of its own — a label map of dots, as the Cell counter
panel makes (:mod:`microscopy_viewer.cell_counter`) — so a sample ends up with
"AF488 positive manual", "AF555 positive manual", and each is stored with the sample as a label
map the Analysis panel and a script can read like any other.

What this module adds to the dots is the arithmetic on them: where each dot is,
how many lie inside the junction region, how many are marked in two channels at
once, and the table that goes to a workbook.

No Qt and no napari here; the panel is
:mod:`microscopy_viewer.widgets.positive_counter_widget`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

#: What a channel's counter is called, on screen and in the sample. Plain
#: words: the store keeps letters, digits and spaces in a name and rewrites the
#: rest, so a "+" would come back as something else.
KEY_SUFFIX = " positive manual"

#: Dot colours, one per channel in order. None is a colour a channel is shown
#: in, so a dot is never lost in the stain it marks.
DOT_COLOURS = ("yellow", "cyan", "white", "orange", "magenta", "lime")

COLUMNS = (
    "Sample", "Channel", "Positive (manual)", "In region", "Region area (mm²)",
    "In region per mm²", "Dot diameter (µm)", "Label map", "File",
)


def key_for(channel: str) -> str:
    """The name a channel's counter is stored under: ``AF488 positive manual``."""
    return f"{str(channel).strip()}{KEY_SUFFIX}"


def channel_of(key: str) -> str | None:
    """The channel a stored counter belongs to, or ``None`` if it is not one."""
    text = str(key)
    return text[: -len(KEY_SUFFIX)] if text.endswith(KEY_SUFFIX) and len(text) > len(KEY_SUFFIX) else None


def colour_for(index: int) -> str:
    return DOT_COLOURS[index % len(DOT_COLOURS)]


def dot_centres(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Every dot's label and its centre, in pixels. ``(labels, centres)``.

    Centres are ``(N, ndim)``. A dot is a small disc, so its centre of mass is
    where it was clicked to within a pixel.

    Only the marked pixels are looked at after one pass to find them: a counter
    over a slide scan is tens of millions of pixels and a few thousand of them
    are dots, and sorting or labelling the whole map is what made a recount
    take a second.
    """
    array = np.asarray(data)
    flat = array.reshape(-1)
    marked = np.flatnonzero(flat)
    if not marked.size:
        return np.zeros(0, dtype=np.int64), np.zeros((0, array.ndim))
    ids = flat[marked].astype(np.int64)
    labels, inverse, sizes = np.unique(ids, return_inverse=True, return_counts=True)
    coordinates = np.unravel_index(marked, array.shape)
    centres = np.column_stack([
        np.bincount(inverse, weights=axis.astype(np.float64), minlength=len(labels)) / sizes
        for axis in coordinates
    ])
    return labels, centres


def count_in_region(data: np.ndarray, region: np.ndarray | None) -> int | None:
    """How many dots have their centre inside *region*; ``None`` without one.

    *region* is a mask on the dots' own YX grid. A 3D counter is read against
    it plane by plane: the region is the same on every plane.
    """
    if region is None:
        return None
    if tuple(np.shape(data)[-2:]) != tuple(np.shape(region)[-2:]):
        raise ValueError(f"the region is {tuple(np.shape(region)[-2:])} and the counter {tuple(np.shape(data)[-2:])}")
    _labels, centres = dot_centres(data)
    return centres_in_region(centres, region)


def centres_in_region(centres: np.ndarray, region: np.ndarray | None) -> int | None:
    """How many of *centres* (pixels, YX last) fall inside *region*.

    Looks up one pixel per dot; the region is never converted or scanned, so
    it costs nothing however large the image.
    """
    if region is None:
        return None
    points = np.asarray(centres, dtype=float).reshape(-1, np.shape(centres)[-1] if np.ndim(centres) == 2 else 2)
    if not len(points):
        return 0
    mask = np.asarray(region)
    plane = mask.reshape(mask.shape[-2:])
    rows = np.clip(np.round(points[:, -2]).astype(int), 0, plane.shape[0] - 1)
    cols = np.clip(np.round(points[:, -1]).astype(int), 0, plane.shape[1] - 1)
    return int((plane[rows, cols] > 0).sum())


def shared(first: np.ndarray, second: np.ndarray, voxel_yx: Sequence[float], within_um: float) -> int:
    """How many dots of *first* have a dot of *second* within *within_um*.

    The same cell marked in two channels is two dots a nucleus apart at most,
    never the same pixel — hence a distance rather than an overlap. Each dot of
    *second* is matched once, so two cells sitting close do not count as three.
    """
    _a, one = dot_centres(first)
    _b, two = dot_centres(second)
    if not len(one) or not len(two):
        return 0
    voxel = np.asarray(tuple(voxel_yx)[-2:], dtype=float)
    one, two = one[:, -2:] * voxel, two[:, -2:] * voxel
    distance = np.linalg.norm(one[:, None, :] - two[None, :, :], axis=2)
    pairs = 0
    # Closest pairs first, each dot used once.
    order = np.dstack(np.unravel_index(np.argsort(distance, axis=None), distance.shape))[0]
    used_one: set[int] = set()
    used_two: set[int] = set()
    for i, j in order:
        if distance[i, j] > within_um:
            break
        if int(i) in used_one or int(j) in used_two:
            continue
        used_one.add(int(i))
        used_two.add(int(j))
        pairs += 1
    return pairs


@dataclass
class Tally:
    """One channel's hand count in one sample."""

    channel: str
    count: int
    in_region: int | None = None


def tally(counters: Mapping[str, np.ndarray], region: np.ndarray | None = None) -> list[Tally]:
    """The count of every channel's counter, and how many are in *region*."""
    from .cell_counter import count as count_dots

    return [
        Tally(channel=channel, count=count_dots(np.asarray(data)), in_region=count_in_region(data, region))
        for channel, data in counters.items()
    ]


def counts_dataframe(
    sample: str,
    tallies: Sequence[Tally],
    region_area_mm2: float | None = None,
    dot_diameter_um: float = float("nan"),
    file: str = "",
):
    """The table of one sample's hand counts, a row per channel."""
    import pandas as pd

    area = float(region_area_mm2) if region_area_mm2 else float("nan")
    rows = []
    for item in tallies:
        inside = float("nan") if item.in_region is None else item.in_region
        rows.append({
            "Sample": sample,
            "Channel": item.channel,
            "Positive (manual)": item.count,
            "In region": inside,
            "Region area (mm²)": area,
            "In region per mm²": inside / area if area and np.isfinite(area) and area > 0 else float("nan"),
            "Dot diameter (µm)": dot_diameter_um,
            "Label map": key_for(item.channel),
            "File": file,
        })
    return pd.DataFrame(rows, columns=list(COLUMNS))
