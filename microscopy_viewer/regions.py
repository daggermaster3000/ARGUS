"""Name anatomical regions, then count segmented objects inside each one.

The question this answers is the one asked after every segmentation of a whole
brain: *how many of those are in the cerebellum?* The label map already knows
where every object is; what it does not know is what the parts of the specimen
are called. So the outlines are drawn by hand — a Shapes layer, one shape per
region, each carrying a name — and each object is attributed to the region its
centroid falls in.

Three decisions worth knowing about:

* **Objects are counted by their centroid, not by overlap.** An object straddling
  a boundary belongs to exactly one region, so the per-region counts sum to the
  total and a nucleus is never counted twice. Overlap-weighted counting is the
  alternative and it does not have that property.
* **Outlines are kept in world coordinates**, in µm, exactly as
  :mod:`microscopy_viewer.intensity` keeps its ROIs. The regions are therefore
  independent of which layer they were drawn on, and the same set can be applied
  to a label map at a different pixel size — which is what makes them reusable
  across the samples of an experiment.
* **Regions are tested in order and the first match wins.** Outlines drawn by
  hand overlap slightly at every boundary; silently double-counting there would
  be worse than a rule that can be stated. :func:`overlapping_pairs` reports
  regions that overlap so the panel can say so.

No Qt and no napari here: the panel converts its Shapes layer into
:class:`Region` objects and everything below that is numpy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

from .utils import get_logger

logger = get_logger("regions")

#: Name of the Shapes layer the panel draws regions on.
REGION_LAYER_NAME = "Brain regions"

#: Offered as starting names in a new region set. Zebrafish neuroanatomy at the
#: resolution a 20x stack resolves — the level at which counts are usually
#: reported — and left/right split because that is the comparison being made.
SUGGESTED_REGIONS = (
    "forebrain",
    "midbrain",
    "hindbrain",
    "cerebellum left",
    "cerebellum right",
)

#: Row for objects whose centroid is in none of the regions. Always reported
#: rather than dropped: a large count here means the outlines missed something,
#: and that is worth seeing rather than quietly losing.
UNASSIGNED = "(outside every region)"

#: Columns of the per-region table, in display order.
REGION_COLUMNS = (
    "region",
    "n_objects",
    "area_um2",
    "density_per_mm2",
    "total_volume_um3",
    "median_diameter_um",
    "mean_intensity",
)

REGION_HEADERS = {
    "region": "Region",
    "n_objects": "Objects",
    "area_um2": "Area (µm²)",
    "density_per_mm2": "Objects per mm²",
    "total_volume_um3": "Total object volume (µm³)",
    "median_diameter_um": "Median diameter (µm)",
    "mean_intensity": "Mean intensity",
}


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


@dataclass
class Region:
    """One named outline, in world (row, column) micrometres."""

    name: str
    vertices_world: np.ndarray  # (N, 2), world Y/X in µm

    def __post_init__(self) -> None:
        self.vertices_world = np.asarray(self.vertices_world, dtype=float)

    @property
    def area_um2(self) -> float:
        return polygon_area(self.vertices_world)

    @property
    def valid(self) -> bool:
        return self.vertices_world.ndim == 2 and self.vertices_world.shape[0] >= 3

    def contains(self, points_world: np.ndarray) -> np.ndarray:
        return contains_points(self.vertices_world, points_world)


def polygon_area(vertices: Sequence[Sequence[float]]) -> float:
    """Area enclosed by a polygon, by the shoelace formula.

    Absolute, so the answer does not depend on which way round the outline was
    drawn — napari gives a rectangle dragged up-and-left the opposite winding to
    one dragged down-and-right, and they are the same rectangle.
    """
    points = np.asarray(vertices, dtype=float)
    if points.ndim != 2 or points.shape[0] < 3:
        return 0.0
    rows = points[:, 0]
    cols = points[:, 1]
    return float(abs(np.dot(rows, np.roll(cols, -1)) - np.dot(cols, np.roll(rows, -1))) / 2.0)


def contains_points(vertices: Sequence[Sequence[float]], points: np.ndarray) -> np.ndarray:
    """Which of *points* lie inside the outline. Both in the same coordinates.

    Matplotlib's path test rather than a hand-rolled ray cast: it is the same one
    :func:`microscopy_viewer.intensity.polygon_mask` uses to rasterise a ROI, so
    a region and a ROI drawn on the same outline agree about their boundary.
    """
    from matplotlib.path import Path

    outline = np.asarray(vertices, dtype=float)
    samples = np.asarray(points, dtype=float).reshape(-1, 2)
    if outline.ndim != 2 or outline.shape[0] < 3 or samples.size == 0:
        return np.zeros(samples.shape[0], dtype=bool)
    return np.asarray(Path(outline).contains_points(samples), dtype=bool)


def shape_to_polygon(shape_data: np.ndarray, shape_type: str) -> np.ndarray:
    """The outline of one napari shape as (N, 2) layer-space Y/X vertices.

    Ellipses arrive as four corners of a bounding parallelogram rather than an
    outline, so they are expanded; everything else already is one. The leading
    axes of a shape drawn on a stack are dropped — a region is an outline in the
    imaging plane and applies through the whole depth.
    """
    from .intensity import ellipse_to_polygon

    vertices = np.asarray(shape_data, dtype=float)
    if vertices.ndim != 2 or vertices.shape[0] < 3:
        return np.zeros((0, 2), dtype=float)
    if str(shape_type).lower() == "ellipse" and vertices.shape[0] == 4:
        vertices = ellipse_to_polygon(vertices[:, -2:])
    return vertices[:, -2:]


def regions_from_shapes(
    shape_data: Sequence[np.ndarray],
    shape_types: Sequence[str],
    names: Sequence[str],
    scale: Sequence[float] = (1.0, 1.0),
    translate: Sequence[float] = (0.0, 0.0),
) -> list[Region]:
    """Build regions from a Shapes layer's raw contents.

    Takes the layer apart rather than taking the layer, so this module never
    imports napari. *names* is parallel to *shape_data*; an unnamed shape gets a
    positional fallback so it is still counted rather than silently ignored.
    """
    from .intensity import world_vertices

    regions: list[Region] = []
    for index, data in enumerate(shape_data):
        kind = shape_types[index] if index < len(shape_types) else "polygon"
        polygon = shape_to_polygon(data, kind)
        if polygon.shape[0] < 3:
            logger.debug("shape %d has too few vertices to be a region", index)
            continue
        raw = str(names[index]).strip() if index < len(names) else ""
        regions.append(
            Region(
                name=raw or f"region {index + 1}",
                vertices_world=world_vertices(polygon, scale, translate),
            )
        )
    return regions


def overlapping_pairs(regions: Sequence[Region], samples: int = 400) -> list[tuple[str, str]]:
    """Pairs of regions that share area, so the panel can name them.

    Tested on a scatter of points inside the first region rather than by exact
    polygon intersection: the answer only has to be good enough to warn with, and
    hand-drawn outlines that genuinely overlap do so over a large area, not a
    sliver. Ordered as the count is resolved — the earlier region wins.
    """
    rng = np.random.default_rng(0)
    found: list[tuple[str, str]] = []
    for i, first in enumerate(regions):
        if not first.valid:
            continue
        box = first.vertices_world
        points = rng.uniform(box.min(axis=0), box.max(axis=0), size=(samples, 2))
        inside_first = first.contains(points)
        if not inside_first.any():
            continue
        candidates = points[inside_first]
        for second in regions[i + 1:]:
            if second.valid and second.contains(candidates).any():
                found.append((first.name, second.name))
    return found


# ---------------------------------------------------------------------------
# Counting
# ---------------------------------------------------------------------------


@dataclass
class RegionCount:
    """What was found in one region."""

    region: str
    n_objects: int = 0
    area_um2: float = 0.0
    density_per_mm2: float = float("nan")
    total_volume_um3: float = 0.0
    median_diameter_um: float = float("nan")
    mean_intensity: float = float("nan")

    def as_row(self) -> dict[str, Any]:
        return {column: getattr(self, column) for column in REGION_COLUMNS}


def object_points(stats: Iterable[Any], translate: Sequence[float] = (0.0, 0.0)) -> np.ndarray:
    """Object centroids as world (row, column) µm, ready to test against regions.

    :class:`~microscopy_viewer.segmentation.ObjectStat` centroids are in µm from
    the label map's own origin, and the regions are in world coordinates, so the
    layer's translate has to be added. It is nearly always zero — and silently
    wrong by a whole field of view when it is not.
    """
    offset = np.asarray(tuple(translate), dtype=float)[-2:]
    if offset.size < 2:
        offset = np.zeros(2, dtype=float)
    points = np.array(
        [[float(stat.centroid_um[-2]), float(stat.centroid_um[-1])] for stat in stats],
        dtype=float,
    ).reshape(-1, 2)
    return points + offset


def assign_objects(
    stats: Sequence[Any],
    regions: Sequence[Region],
    translate: Sequence[float] = (0.0, 0.0),
) -> list[str]:
    """The region each object belongs to, parallel to *stats*.

    First match wins, so overlapping outlines split their shared objects
    deterministically instead of double-counting them. Objects in no region get
    :data:`UNASSIGNED`.
    """
    assignment = [UNASSIGNED] * len(stats)
    if not len(stats) or not regions:
        return assignment

    points = object_points(stats, translate)
    remaining = np.ones(points.shape[0], dtype=bool)
    for region in regions:
        if not region.valid or not remaining.any():
            continue
        hit = np.zeros(points.shape[0], dtype=bool)
        hit[remaining] = region.contains(points[remaining])
        for index in np.flatnonzero(hit):
            assignment[int(index)] = region.name
        remaining &= ~hit
    return assignment


def count_objects(
    stats: Sequence[Any],
    regions: Sequence[Region],
    translate: Sequence[float] = (0.0, 0.0),
    include_unassigned: bool = True,
) -> list[RegionCount]:
    """Per-region counts and summaries, in the order the regions were given.

    Every region gets a row even when it is empty — a count of zero in the
    cerebellum is a result, and a missing row looks like a mistake.
    """
    assignment = assign_objects(stats, regions, translate)
    grouped: dict[str, list[Any]] = {}
    for name, stat in zip(assignment, stats):
        grouped.setdefault(name, []).append(stat)

    counts: list[RegionCount] = []
    for region in regions:
        counts.append(summarise_region(region.name, grouped.get(region.name, []), region.area_um2))
    leftover = grouped.get(UNASSIGNED, [])
    if include_unassigned and leftover:
        counts.append(summarise_region(UNASSIGNED, leftover, 0.0))
    return counts


def summarise_region(name: str, stats: Sequence[Any], area_um2: float = 0.0) -> RegionCount:
    """Count and summarise one region's objects.

    Public because the batch workbook summarises a sample that carries no
    outlines the same way it summarises one that does, and duplicating the
    arithmetic is how two tables come to disagree about the same numbers.
    """
    count = RegionCount(region=name, n_objects=len(stats), area_um2=float(area_um2))
    if area_um2 > 0:
        # Per mm², because per µm² of a brain section is a number with five
        # leading zeros.
        count.density_per_mm2 = float(len(stats) / (area_um2 / 1e6))
    if not stats:
        return count
    volumes = np.array([float(stat.volume_um3) for stat in stats], dtype=float)
    diameters = np.array([float(stat.equivalent_diameter_um) for stat in stats], dtype=float)
    means = np.array([float(stat.mean) for stat in stats], dtype=float)
    count.total_volume_um3 = float(volumes.sum())
    count.median_diameter_um = float(np.median(diameters))
    count.mean_intensity = float(np.mean(means))
    return count


def counts_dataframe(counts: Sequence[RegionCount]):
    """The per-region table as a DataFrame, for the workbook writer."""
    import pandas as pd

    frame = pd.DataFrame([count.as_row() for count in counts], columns=list(REGION_COLUMNS))
    return frame.rename(columns=REGION_HEADERS)


def objects_dataframe(
    stats: Sequence[Any],
    regions: Sequence[Region],
    translate: Sequence[float] = (0.0, 0.0),
    ndim: int = 3,
):
    """The per-object table with a *Region* column added.

    The counts say how many; this says which, so a suspicious count can be
    traced back to the objects that produced it.

    *ndim* is the label map's, and only picks the size column's heading — passed
    in for the same reason :func:`segmentation.object_dataframe` takes it rather
    than inferring it: a genuine 3D run can have every object at z=0.
    """
    from .segmentation import OBJECT_COLUMNS, object_headers

    import pandas as pd

    rows = []
    for stat, name in zip(stats, assign_objects(stats, regions, translate)):
        row = stat.as_row()
        row["region"] = name
        rows.append(row)
    frame = pd.DataFrame(rows, columns=["region", *OBJECT_COLUMNS])
    headers = {"region": "Region", **object_headers(ndim)}
    return frame.rename(columns=headers)
