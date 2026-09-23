"""Describe every brain region of every sample as numbers, ready for PCA.

The batch workbook (:mod:`microscopy_viewer.experiment`) answers *how many
objects, where*. This answers the next question — *how do the regions differ* —
by putting one row per sample and region into a table whose columns are all
numbers: the shape of the outline, the objects found inside it, and how bright
every channel is there.

Everything is read back out of the files, not out of the session. A sample that
was segmented last week, outlined yesterday and never opened today is analysed
the same as one on screen: its label map and outlines are in ``/ARGUS`` inside
the ``.ims`` (see :mod:`microscopy_viewer.ims_store`), and its channels are the
file's own.

Three decisions worth knowing about:

* **Region channel intensities are measured in 2D.** A stack is reduced to its
  maximum-intensity projection first (every leading axis — Z, and T if there is
  one — is collapsed), and each channel's statistics are taken inside the
  outline on that plane. Planes of empty space above and below the tissue then
  do not dilute the numbers. Objects and region volumes are still measured in
  3D.
* **Channel statistics can be read off a coarser pyramid level.** A whole brain
  at full resolution is a few gigabytes per channel; level 1 is an eighth of
  that and its means and percentiles are the same to a few percent. Level 0 is
  the default because it is the honest one.
* **Every cell is measured in every channel.** On the label map's own grid:
  a 3D label map against each channel's full-resolution stack, a 2D one
  against each channel's maximum-intensity projection, the same plane its
  cells were segmented on. The *Cell intensities* sheet has the result.
* **Channels are matched by name across samples.** The wide sheets name their
  columns after the channel, so a file that calls it "Confocal - GFP" and one
  that calls it "GFP" produce different columns. The long sheet does not care.

No Qt and no napari: the panel is a view onto this.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from . import experiment as ex
from . import ims_store, naming
from .utils import get_logger

logger = get_logger("analysis")

#: Region name for the union of a sample's outlines, in the long intensity sheet.
ALL_REGIONS = "(all regions)"

#: Percentiles reported per region and channel. The tails, not the extremes:
#: a single hot pixel moves the maximum and not the 99th percentile.
PERCENTILES = (5.0, 25.0, 75.0, 95.0, 99.0)

#: Columns of the long per-region, per-channel sheet.
INTENSITY_COLUMNS = (
    "Sample",
    "Genotype",
    "Region",
    "Channel",
    "Voxels",
    "Mean",
    "Median",
    "SD",
    "CV",
    "Min",
    "P5",
    "P25",
    "P75",
    "P95",
    "P99",
    "Max",
    "Integrated",
    "Level",
)

#: The per-channel statistics that make it into the wide sheets.
WIDE_INTENSITY = ("Mean", "Median", "SD", "CV", "P5", "P95", "P99", "Integrated")


@dataclass
class AnalysisOptions:
    """What an analysis run reads out of each file."""

    #: Which stored label map to use, matched as a substring of its key, case
    #: insensitively. Empty takes the first one the file carries.
    label_key: str = ""
    #: Channel the per-object intensities are read from. ``None`` means the
    #: channel the label map was segmented on, as recorded when it was written.
    measure_channel: int | str | None = None
    #: Pyramid level the per-region channel statistics are read at.
    intensity_level: int = 0
    #: Channels to describe. Empty means every channel in the file.
    channels: Sequence[int | str] = ()
    #: Trace every object's outline and measure its shape. Costs a few seconds
    #: per ten thousand objects.
    cell_outlines: bool = True
    #: Measure every described channel inside every object, not only the one
    #: it was segmented on. One more full-resolution read per channel.
    cell_channels: bool = True


@dataclass
class ChannelStat:
    """One channel's intensities inside one region."""

    region: str
    channel: str
    n_voxels: int = 0
    mean: float = float("nan")
    median: float = float("nan")
    std: float = float("nan")
    minimum: float = float("nan")
    maximum: float = float("nan")
    integrated: float = 0.0
    percentiles: dict[float, float] = field(default_factory=dict)
    level: int = 0

    @property
    def cv(self) -> float:
        if not np.isfinite(self.mean) or self.mean == 0:
            return float("nan")
        return float(self.std / self.mean)


@dataclass
class RegionShape:
    """The outline itself, measured: size, elongation, irregularity, position."""

    region: str
    area_um2: float = 0.0
    perimeter_um: float = 0.0
    circularity: float = float("nan")
    solidity: float = float("nan")
    major_axis_um: float = float("nan")
    minor_axis_um: float = float("nan")
    aspect_ratio: float = float("nan")
    eccentricity: float = float("nan")
    orientation_deg: float = float("nan")
    centroid_y_um: float = float("nan")
    centroid_x_um: float = float("nan")
    bbox_height_um: float = 0.0
    bbox_width_um: float = 0.0
    #: Outline area times the stack's depth — the volume the channel statistics
    #: were taken over, and what object volumes are a fraction of.
    volume_um3: float = 0.0


@dataclass
class AnalysisOutcome(ex.BatchOutcome):
    """A batch outcome read back from a file, plus what the analysis adds.

    A subclass rather than a new type so every sheet the batch export writes is
    written from it unchanged — two implementations of the same counts is how
    two workbooks come to disagree.
    """

    shapes: list[RegionShape] = field(default_factory=list)
    channel_stats: list[ChannelStat] = field(default_factory=list)
    channel_names: list[str] = field(default_factory=list)
    #: ``label -> (N, 2)`` outline in µm (Y, X), one per object.
    cell_outlines: dict[int, np.ndarray] = field(default_factory=dict)
    #: ``label -> RegionShape`` of that outline, plus its pixel-counted area.
    cell_shapes: dict[int, tuple[RegionShape, float]] = field(default_factory=dict)
    #: ``channel -> statistic -> values indexed by label`` (see
    #: :func:`object_intensities`).
    cell_intensities: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
    depth_um: float = 0.0


# ---------------------------------------------------------------------------
# Region geometry
# ---------------------------------------------------------------------------


def polygon_perimeter(vertices: np.ndarray) -> float:
    points = np.asarray(vertices, dtype=float).reshape(-1, 2)
    if points.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(points - np.roll(points, -1, axis=0), axis=1).sum())


def _hull_area(vertices: np.ndarray) -> float:
    """Convex hull area, by the monotone chain. No scipy needed for a polygon."""
    from .regions import polygon_area

    points = sorted(set(map(tuple, np.asarray(vertices, dtype=float).reshape(-1, 2))))
    if len(points) < 3:
        return 0.0

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list = []
    for point in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: list = []
    for point in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return polygon_area(np.asarray(lower[:-1] + upper[:-1]))


def _polygon_moments(vertices: np.ndarray) -> tuple[float, float, float, float, float, float]:
    """Area, centroid and central second moments of a simple polygon.

    Exact, from the outline (Green's theorem), rather than from a rasterised
    mask — so the shape descriptors do not depend on which pyramid level the
    intensities were read at.
    """
    points = np.asarray(vertices, dtype=float).reshape(-1, 2)
    y, x = points[:, 0], points[:, 1]
    y1, x1 = np.roll(y, -1), np.roll(x, -1)
    cross = x * y1 - x1 * y
    area = cross.sum() / 2.0
    if area == 0:
        return 0.0, float("nan"), float("nan"), 0.0, 0.0, 0.0
    cx = ((x + x1) * cross).sum() / (6.0 * area)
    cy = ((y + y1) * cross).sum() / (6.0 * area)
    # Second moments about the origin, then shifted to the centroid.
    mxx = ((x * x + x * x1 + x1 * x1) * cross).sum() / 12.0
    myy = ((y * y + y * y1 + y1 * y1) * cross).sum() / 12.0
    mxy = ((x * y1 + 2 * x * y + 2 * x1 * y1 + x1 * y) * cross).sum() / 24.0
    mu_xx = mxx / area - cx * cx
    mu_yy = myy / area - cy * cy
    mu_xy = mxy / area - cx * cy
    return abs(area), float(cy), float(cx), float(mu_yy), float(mu_xx), float(mu_xy)


def region_shape(region, depth_um: float = 0.0) -> RegionShape:
    """Morphometrics of one outline, in micrometres."""
    vertices = np.asarray(region.vertices_world, dtype=float).reshape(-1, 2)
    shape = RegionShape(region=region.name)
    if vertices.shape[0] < 3:
        return shape
    area, cy, cx, mu_yy, mu_xx, mu_xy = _polygon_moments(vertices)
    shape.area_um2 = float(area)
    shape.perimeter_um = polygon_perimeter(vertices)
    if shape.perimeter_um > 0:
        shape.circularity = float(4.0 * math.pi * area / shape.perimeter_um**2)
    hull = _hull_area(vertices)
    if hull > 0:
        shape.solidity = float(area / hull)
    shape.centroid_y_um, shape.centroid_x_um = cy, cx
    # Axes of the ellipse with the same second moments, as scikit-image does it.
    spread = math.sqrt(max(((mu_xx - mu_yy) / 2.0) ** 2 + mu_xy**2, 0.0))
    major = (mu_xx + mu_yy) / 2.0 + spread
    minor = (mu_xx + mu_yy) / 2.0 - spread
    if major > 0:
        shape.major_axis_um = float(4.0 * math.sqrt(major))
        shape.minor_axis_um = float(4.0 * math.sqrt(max(minor, 0.0)))
        shape.eccentricity = float(math.sqrt(max(1.0 - max(minor, 0.0) / major, 0.0)))
        if shape.minor_axis_um > 0:
            shape.aspect_ratio = float(shape.major_axis_um / shape.minor_axis_um)
        # Angle of the major axis from the image's X axis, anticlockwise on
        # screen (rows grow downwards, hence the sign).
        shape.orientation_deg = float(
            math.degrees(-0.5 * math.atan2(2.0 * mu_xy, mu_xx - mu_yy))
        )
    extent = vertices.max(axis=0) - vertices.min(axis=0)
    shape.bbox_height_um = float(extent[0])
    shape.bbox_width_um = float(extent[1])
    shape.volume_um3 = float(area * depth_um) if depth_um > 0 else float(area)
    return shape


# ---------------------------------------------------------------------------
# Cell outlines
# ---------------------------------------------------------------------------

#: Vertices each cell outline is resampled to. Enough to show a cell's shape at
#: plot size; few enough that a hundred thousand cells stay a small file.
CONTOUR_POINTS = 24


def resample_closed(points: np.ndarray, n: int = CONTOUR_POINTS) -> np.ndarray:
    """*n* points evenly spaced along a closed outline."""
    ring = np.vstack([points, points[:1]])
    steps = np.linalg.norm(np.diff(ring, axis=0), axis=1)
    along = np.concatenate([[0.0], np.cumsum(steps)])
    if along[-1] <= 0:
        return np.repeat(points[:1], n, axis=0)
    targets = np.linspace(0.0, along[-1], n, endpoint=False)
    return np.column_stack([np.interp(targets, along, ring[:, i]) for i in range(2)])


def cell_outlines(masks: np.ndarray, voxel_yx, progress: Callable[[str], None] | None = None):
    """Outline and shape of every object in a label map, in µm.

    Returns ``(outlines, shapes)``, both keyed by label. A 3D object is outlined
    by its footprint — every (y, x) it covers in any plane — which is what it
    looks like from above and what the projection-based measurements see.
    """
    from scipy import ndimage
    from skimage.measure import find_contours

    from .regions import Region

    labels = np.asarray(masks)
    dy, dx = (float(v) for v in tuple(voxel_yx)[-2:])
    scale = np.array([dy, dx])
    outlines: dict[int, np.ndarray] = {}
    shapes: dict[int, tuple[RegionShape, float]] = {}
    boxes = ndimage.find_objects(labels)
    total = sum(box is not None for box in boxes)
    done = 0
    for index, box in enumerate(boxes):
        if box is None:
            continue
        label = index + 1
        crop = labels[box] == label
        footprint = crop.any(axis=tuple(range(crop.ndim - 2))) if crop.ndim > 2 else crop
        contours = find_contours(np.pad(footprint, 1).astype(np.uint8), 0.5)
        done += 1
        if progress is not None and done % 5000 == 0:
            progress(f"outlined {done} of {total} objects")
        if not contours:
            continue
        longest = max(contours, key=len)
        origin = np.array([box[-2].start, box[-1].start], dtype=float) - 1.0
        outline = resample_closed((longest + origin) * scale).astype(np.float32)
        outlines[label] = outline
        shape = region_shape(Region(name=str(label), vertices_world=outline))
        shapes[label] = (shape, float(footprint.sum()) * dy * dx)
    return outlines, shapes


# ---------------------------------------------------------------------------
# Channel intensities
# ---------------------------------------------------------------------------


def _level(spec, level: int):
    """The array at pyramid *level* (clamped), and its (Y, X) voxel size."""
    levels = list(spec.data) if getattr(spec, "multiscale", False) else [spec.data]
    chosen = levels[min(max(int(level), 0), len(levels) - 1)]
    full = np.shape(levels[0])
    here = np.shape(chosen)
    scale = tuple(float(v) for v in spec.scale)[-2:]
    voxel_yx = tuple(
        scale[i] * float(full[-2 + i]) / max(float(here[-2 + i]), 1.0) for i in range(2)
    )
    return chosen, voxel_yx, min(max(int(level), 0), len(levels) - 1)


def project(array) -> np.ndarray:
    """Maximum-intensity projection over every leading axis, one plane at a time.

    Plane by plane rather than ``array.max(axis=0)``: a lazily read whole-brain
    channel would otherwise be materialised whole to take its maximum.
    """
    shape = tuple(int(n) for n in np.shape(array))
    if len(shape) <= 2:
        return np.asarray(array)
    result = None
    for index in range(int(np.prod(shape[:-2]))):
        position = tuple(int(i) for i in np.unravel_index(index, shape[:-2]))
        plane = np.asarray(array[position])
        result = plane.copy() if result is None else np.maximum(result, plane, out=result)
    return result


#: Per-object statistics of each channel, in the *Cell intensities* sheet.
CELL_INTENSITY_STATS = ("Mean", "SD", "Max", "Integrated")


def object_intensities(masks: np.ndarray, signal) -> dict[str, np.ndarray]:
    """Each object's intensity statistics in *signal*, indexed by label.

    *signal* must have the shape of *masks*; it may be lazy (an HDF5 dataset),
    since it is read one plane at a time and only the labelled voxels of each
    plane are kept. Labels with no voxels come back as NaN.
    """
    masks = np.asarray(masks)
    highest = int(masks.max()) if masks.size else 0
    count = np.zeros(highest + 1)
    total = np.zeros(highest + 1)
    squares = np.zeros(highest + 1)
    maxima = np.full(highest + 1, -np.inf)
    planes = masks.reshape(-1, *masks.shape[-2:]) if masks.ndim > 2 else masks[None]
    for index, plane in enumerate(planes):
        where = plane > 0
        if not where.any():
            continue
        if masks.ndim > 2:
            position = np.unravel_index(index, masks.shape[:-2])
            values = np.asarray(signal[tuple(int(i) for i in position)])[where]
        else:
            values = np.asarray(signal)[where]
        ids = plane[where].astype(np.int64, copy=False)
        values = values.astype(np.float64)
        count += np.bincount(ids, minlength=highest + 1)
        total += np.bincount(ids, weights=values, minlength=highest + 1)
        squares += np.bincount(ids, weights=values * values, minlength=highest + 1)
        np.maximum.at(maxima, ids, values)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = total / count
        sd = np.sqrt(np.maximum(squares / count - mean**2, 0.0))
    empty = count == 0
    maxima[empty] = np.nan
    return {"Mean": mean, "SD": sd, "Max": maxima, "Integrated": np.where(empty, np.nan, total)}


def region_masks(regions, shape_yx, voxel_yx) -> dict[str, np.ndarray]:
    """One boolean (Y, X) mask per region, plus their union, on one pixel grid."""
    from .intensity import polygon_mask, world_to_data

    masks: dict[str, np.ndarray] = {}
    union = np.zeros(tuple(int(n) for n in shape_yx), dtype=bool)
    for region in regions:
        pixels = world_to_data(region.vertices_world, voxel_yx, (0.0, 0.0))
        mask = polygon_mask(pixels, union.shape)
        # Same name twice (two outlines of one region) is one region, as the
        # counts treat it.
        masks[region.name] = masks[region.name] | mask if region.name in masks else mask
        union |= mask
    if masks:
        masks[ALL_REGIONS] = union
    return masks


def summarise_values(region: str, channel: str, values: np.ndarray, level: int = 0) -> ChannelStat:
    """Intensity statistics of one region's voxels in one channel."""
    stat = ChannelStat(region=region, channel=channel, level=level)
    data = np.asarray(values).ravel()
    stat.n_voxels = int(data.size)
    if data.size == 0:
        return stat
    stat.mean = float(data.mean(dtype=np.float64))
    stat.std = float(data.std(dtype=np.float64))
    stat.minimum = float(data.min())
    stat.maximum = float(data.max())
    stat.integrated = float(data.sum(dtype=np.float64))
    wanted = (50.0, *PERCENTILES)
    found = np.percentile(data, wanted)
    stat.median = float(found[0])
    stat.percentiles = {p: float(v) for p, v in zip(PERCENTILES, found[1:])}
    return stat


def channel_stats(
    array,
    masks: dict[str, np.ndarray],
    channel: str,
    level: int = 0,
) -> list[ChannelStat]:
    """Statistics of *array* inside every mask, reading one plane at a time.

    A whole-brain channel is gigabytes; the masked voxels of a region are a
    fraction of that, kept in the file's own dtype. Reading plane by plane keeps
    the peak at one plane plus what the regions cover.
    """
    shape = tuple(int(n) for n in np.shape(array))
    if len(shape) < 2:
        return []
    planes = int(np.prod(shape[:-2])) if len(shape) > 2 else 1
    collected: dict[str, list[np.ndarray]] = {name: [] for name in masks}
    for index in range(planes):
        position = np.unravel_index(index, shape[:-2]) if len(shape) > 2 else ()
        plane = np.asarray(array[tuple(int(i) for i in position)])
        for name, mask in masks.items():
            if mask.shape != plane.shape:
                continue
            collected[name].append(plane[mask])
    stats = []
    for name, chunks in collected.items():
        values = np.concatenate(chunks) if chunks else np.zeros(0)
        stats.append(summarise_values(name, channel, values, level))
    return stats


# ---------------------------------------------------------------------------
# One sample
# ---------------------------------------------------------------------------


def pick_label_key(keys: Sequence[str], wanted: str = "") -> str:
    """The stored label map *wanted* names, or the first one; empty if none."""
    if not keys:
        return ""
    needle = str(wanted or "").strip().lower()
    if not needle:
        return keys[0]
    for key in keys:
        if needle in key.lower():
            return key
    return ""


def _channels_to_describe(names: Sequence[str], wanted: Sequence[int | str]) -> list[int]:
    if not wanted:
        return list(range(len(names)))
    picked: list[int] = []
    for spec in wanted:
        index = ex.pick_channel(names, spec)
        if index is not None and index not in picked:
            picked.append(index)
    return picked


def analyse_sample(
    path: str | Path,
    options: AnalysisOptions | None = None,
    progress: Callable[[str], None] | None = None,
) -> AnalysisOutcome:
    """Read one file's labels, outlines and channels, and describe each region.

    Never raises: a file that cannot be read comes back with ``error`` set, so a
    folder of thirty is not stopped by one stub.
    """
    from . import loaders
    from . import segmentation as sg

    options = options or AnalysisOptions()
    target = Path(path)
    outcome = AnalysisOutcome(
        path=target, name=target.stem, genotype=naming.genotype_from_name(target.name)
    )
    say = progress or (lambda _text: None)
    started = time.perf_counter()
    # Reading does not need the file to itself, and releasing the handle of a
    # file that is on screen would break its layers. Only give back what this
    # opened.
    was_open = loaders.is_open(target)
    try:
        say("reading")
        specs = loaders.load_path(target)
        if not specs:
            raise ValueError(f"{target.name} has no readable channels")
        names = [spec.channel_name or spec.name for spec in specs]
        outcome.channel_names = list(names)
        first = specs[0]
        full = np.shape(first.data[0] if first.multiscale else first.data)
        scale = tuple(float(v) for v in first.scale)
        if len(full) >= 3 and len(scale) >= 3:
            outcome.depth_um = float(full[-3]) * scale[-3]

        outcome.region_rois = ims_store.load_rois(target)
        regions = ex.regions_of(outcome)

        # -- objects ------------------------------------------------------------
        keys = ims_store.list_labels(target)
        key = pick_label_key(keys, options.label_key)
        if not key:
            outcome.warnings.append(
                "no stored label map"
                + (f" matching {options.label_key!r}" if options.label_key else "")
                + (f" (it has: {', '.join(keys)})" if keys else "")
            )
        else:
            say(f"reading labels “{key}”")
            masks, attrs = ims_store.load_labels(target, key)
            if masks is None:
                raise OSError(f"could not read the label map {key!r}")
            outcome.label_key = key
            outcome.saved = True
            outcome.ndim = int(masks.ndim)
            outcome.channel_name = ims_store._text(attrs["channel"]) if "channel" in attrs else ""
            voxel = tuple(float(v) for v in np.asarray(attrs.get("voxel_size_um", ())).ravel())
            if len(voxel) != masks.ndim:
                voxel = scale[-masks.ndim:]

            wanted = options.measure_channel
            if wanted is None and outcome.channel_name:
                wanted = outcome.channel_name
            signal = None
            index = ex.pick_channel(names, wanted) if wanted is not None else None
            if index is not None:
                spec = specs[index]
                say(f"reading “{names[index]}” for per-object intensities")
                signal = np.asarray(spec.data[0] if spec.multiscale else spec.data)
                if signal.ndim > masks.ndim and masks.ndim == 2:
                    signal = sg.max_projection(signal.reshape(-1, *signal.shape[-2:]))
                if signal.shape != masks.shape:
                    outcome.warnings.append(
                        f"“{names[index]}” is {signal.shape} and the labels are "
                        f"{masks.shape}; per-object intensities were not measured"
                    )
                    signal = None
                else:
                    outcome.channel_name = names[index]
            elif wanted is not None:
                outcome.warnings.append(f"no channel matching {wanted!r} to measure objects on")
            say("measuring objects")
            outcome.stats = sg.object_table(masks, signal, voxel)
            outcome.n_objects = len(outcome.stats)
            del signal
            if options.cell_outlines and outcome.n_objects:
                say(f"outlining {outcome.n_objects} objects")
                outcome.cell_outlines, outcome.cell_shapes = cell_outlines(
                    masks, voxel[-2:], say
                )
            if options.cell_channels and outcome.n_objects:
                for index in _channels_to_describe(names, options.channels):
                    spec = specs[index]
                    array = spec.data[0] if spec.multiscale else spec.data
                    shape = tuple(int(n) for n in np.shape(array))
                    say(f"measuring every cell in “{names[index]}”")
                    if masks.ndim == 2 and len(shape) > 2 and shape[-2:] == masks.shape:
                        # The plane the cells were segmented on.
                        array = project(array)
                    elif shape != masks.shape:
                        outcome.warnings.append(
                            f"“{names[index]}” is {shape} and the labels are {masks.shape}; "
                            "its per-cell intensities were not measured"
                        )
                        continue
                    outcome.cell_intensities[names[index]] = object_intensities(masks, array)
                    del array
            del masks

        # -- regions ------------------------------------------------------------
        outcome.shapes = [region_shape(region, outcome.depth_um) for region in regions]

        for index in _channels_to_describe(names, options.channels):
            array, voxel_yx, level = _level(specs[index], options.intensity_level)
            say(f"projecting “{names[index]}”")
            plane = project(array)
            if regions:
                masks_by_region = region_masks(regions, plane.shape, voxel_yx)
            else:
                masks_by_region = {ex.NO_REGIONS: np.ones(plane.shape, dtype=bool)}
            say(f"describing “{names[index]}” in {len(masks_by_region)} region(s)")
            outcome.channel_stats.extend(
                channel_stats(plane, masks_by_region, names[index], level)
            )
            del plane
    except Exception as exc:
        logger.warning("analysis failed on %s: %s", target.name, exc)
        logger.debug("analysis traceback", exc_info=True)
        outcome.error = str(exc)
    finally:
        outcome.elapsed_s = time.perf_counter() - started
        if not was_open:
            ex._release(target)
    return outcome


def analyse(
    paths: Sequence[str | Path],
    options: AnalysisOptions | None = None,
    progress: Callable[[str], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> list[AnalysisOutcome]:
    """:func:`analyse_sample` over every path, one file at a time."""
    outcomes: list[AnalysisOutcome] = []
    for number, entry in enumerate(paths, start=1):
        if should_cancel is not None and should_cancel():
            break
        path = Path(entry)
        prefix = f"{path.stem} ({number} of {len(paths)})"

        def _relay(text: str, _prefix=prefix) -> None:
            if progress is not None:
                progress(f"{_prefix}: {text}")

        outcome = analyse_sample(path, options, _relay)
        outcomes.append(outcome)
        if progress is not None:
            if outcome.error:
                progress(f"{prefix}: failed — {outcome.error}")
            else:
                done = f"{outcome.n_objects} object(s), {len(outcome.shapes)} region(s)"
                if outcome.warnings:
                    done += " — " + "; ".join(outcome.warnings)
                progress(f"{prefix}: {done}")
    return outcomes


# ---------------------------------------------------------------------------
# The workbook
# ---------------------------------------------------------------------------


def intensity_dataframe(outcomes: Sequence[AnalysisOutcome]):
    """Long table: one row per sample, region and channel."""
    import pandas as pd

    rows = []
    for outcome in outcomes:
        for stat in outcome.channel_stats:
            rows.append(
                {
                    "Sample": outcome.name,
                    "Genotype": outcome.genotype,
                    "Region": stat.region,
                    "Channel": stat.channel,
                    "Voxels": stat.n_voxels,
                    "Mean": ex._round(stat.mean, 3),
                    "Median": ex._round(stat.median, 3),
                    "SD": ex._round(stat.std, 3),
                    "CV": ex._round(stat.cv, 4),
                    "Min": stat.minimum,
                    **{
                        f"P{int(p)}": ex._round(stat.percentiles.get(p, float("nan")), 3)
                        for p in PERCENTILES
                    },
                    "Max": stat.maximum,
                    "Integrated": ex._round(stat.integrated, 1),
                    "Level": stat.level,
                }
            )
    return pd.DataFrame(rows, columns=list(INTENSITY_COLUMNS))


def _shape_row(shape: RegionShape | None) -> dict[str, float]:
    empty = RegionShape(region="")
    source = shape or empty
    return {
        "Region area (µm²)": ex._round(source.area_um2, 1) if shape else float("nan"),
        "Region volume (µm³)": ex._round(source.volume_um3, 1) if shape else float("nan"),
        "Perimeter (µm)": ex._round(source.perimeter_um, 2) if shape else float("nan"),
        "Circularity": ex._round(source.circularity, 4),
        "Solidity": ex._round(source.solidity, 4),
        "Major axis (µm)": ex._round(source.major_axis_um, 2),
        "Minor axis (µm)": ex._round(source.minor_axis_um, 2),
        "Aspect ratio": ex._round(source.aspect_ratio, 4),
        "Eccentricity": ex._round(source.eccentricity, 4),
        "Orientation (°)": ex._round(source.orientation_deg, 2),
        "Centroid Y (µm)": ex._round(source.centroid_y_um, 2),
        "Centroid X (µm)": ex._round(source.centroid_x_um, 2),
        "Bounding height (µm)": ex._round(source.bbox_height_um, 2) if shape else float("nan"),
        "Bounding width (µm)": ex._round(source.bbox_width_um, 2) if shape else float("nan"),
    }


def _object_row(
    stats: Sequence[Any], shape: RegionShape | None, has_labels: bool, is_stack: bool
) -> dict[str, float]:
    """Object counts and morphometrics for one region, with 3D-aware densities."""
    nan = float("nan")
    count = len(stats) if has_labels else nan
    area = shape.area_um2 if shape else 0.0
    volume = shape.volume_um3 if shape else 0.0
    sizes = np.array([float(stat.volume_um3) for stat in stats], dtype=float)
    diameters = np.array([float(stat.equivalent_diameter_um) for stat in stats], dtype=float)
    means = np.array([float(stat.mean) for stat in stats], dtype=float)

    def _stat(values, fn, digits):
        return ex._round(float(fn(values)), digits) if values.size else nan

    return {
        "Objects": count,
        "Objects per mm²": ex._round(count / (area / 1e6), 2) if has_labels and area > 0 else nan,
        "Objects per mm³": ex._round(count / (volume / 1e9), 2) if has_labels and is_stack and volume > 0 else nan,
        "Object volume fraction": ex._round(float(sizes.sum()) / volume, 6) if has_labels and volume > 0 else nan,
        "Mean object diameter (µm)": _stat(diameters, np.mean, 3),
        "Median object diameter (µm)": _stat(diameters, np.median, 3),
        "SD object diameter (µm)": _stat(diameters, np.std, 3),
        "Mean object size": _stat(sizes, np.mean, 3),
        "Median object size": _stat(sizes, np.median, 3),
        "SD object size": _stat(sizes, np.std, 3),
        "Mean object intensity": _stat(means, np.mean, 2),
        "SD object intensity": _stat(means, np.std, 2),
    }


def features_dataframe(outcomes: Sequence[AnalysisOutcome]):
    """Wide table: one row per sample and region, every column a number.

    The table PCA is run on. Rows are regions as drawn — the objects outside
    every outline and the union of all outlines are left out, because neither is
    an anatomical unit and both would dominate the first component. A sample
    with no outlines contributes one whole-image row, marked as such.
    """
    import pandas as pd

    rows = []
    channels: list[str] = []
    for outcome in outcomes:
        for stat in outcome.channel_stats:
            if stat.channel not in channels:
                channels.append(stat.channel)

    for outcome in outcomes:
        if outcome.error:
            continue
        regions = ex.regions_of(outcome)
        grouped = ex._stats_by_region(outcome, regions)
        shapes = {shape.region: shape for shape in outcome.shapes}
        by_key = {(stat.region, stat.channel): stat for stat in outcome.channel_stats}
        names = [region.name for region in regions] or [ex.NO_REGIONS]
        seen: list[str] = []
        for name in names:
            if name in seen:
                continue
            seen.append(name)
            row: dict[str, Any] = {
                "Sample": outcome.name,
                "Genotype": outcome.genotype,
                "Region": name,
                "Label map": outcome.label_key,
            }
            row.update(_shape_row(shapes.get(name)))
            row.update(
                _object_row(
                    grouped.get(name, []),
                    shapes.get(name),
                    bool(outcome.label_key),
                    outcome.depth_um > 0,
                )
            )
            for channel in channels:
                stat = by_key.get((name, channel))
                for column in WIDE_INTENSITY:
                    row[f"{channel} {column}"] = _wide_value(stat, column)
            rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=["Sample", "Genotype", "Region", "Label map"])
    return frame.rename(columns=_size_headers(outcomes))


def _wide_value(stat: ChannelStat | None, column: str) -> float:
    if stat is None:
        return float("nan")
    value = {
        "Mean": stat.mean,
        "Median": stat.median,
        "SD": stat.std,
        "CV": stat.cv,
        "Integrated": stat.integrated,
    }.get(column)
    if value is None and column.startswith("P"):
        value = stat.percentiles.get(float(column[1:]), float("nan"))
    return ex._round(float(value), 4 if column == "CV" else 3)


def _size_headers(outcomes: Sequence[AnalysisOutcome]) -> dict[str, str]:
    ndim = next((outcome.ndim for outcome in outcomes if outcome.stats), 3)
    unit = "(µm³)" if ndim >= 3 else "(µm²)"
    return {
        "Mean object size": f"Mean object size {unit}",
        "Median object size": f"Median object size {unit}",
        "SD object size": f"SD object size {unit}",
    }


def pca_matrix(features) -> Any:
    """One row per sample, one numeric column per region × feature.

    The shape PCA over samples wants: ``Sample`` and ``Genotype`` first, then
    every numeric feature of every region as ``<region> | <feature>``. Columns
    that are the same for every sample (a constant carries no variance) and
    identifying columns (centroids, orientation) are kept — dropping them is a
    modelling decision, and it belongs in the analysis, not the export.
    """
    import pandas as pd

    if features is None or features.empty:
        return pd.DataFrame(columns=["Sample", "Genotype"])
    identity = ["Sample", "Genotype", "Region", "Label map"]
    numeric = [
        column for column in features.columns
        if column not in identity and pd.api.types.is_numeric_dtype(features[column])
    ]
    samples = features[["Sample", "Genotype"]].drop_duplicates("Sample").set_index("Sample")
    wide = features.pivot_table(
        index="Sample", columns="Region", values=numeric, aggfunc="first", dropna=False
    )
    # Region first, then feature, in the order the regions were drawn.
    regions = list(dict.fromkeys(features["Region"]))
    ordered = [(feature, region) for region in regions for feature in numeric
               if (feature, region) in wide.columns]
    wide = wide.reindex(columns=pd.MultiIndex.from_tuples(ordered))
    wide.columns = [f"{region} | {feature}" for feature, region in wide.columns]
    wide = samples.join(wide).reset_index()
    return wide


#: Columns of the per-cell shape sheet.
CELL_SHAPE_COLUMNS = (
    "Sample", "Genotype", "Label", "Footprint area (µm²)", "Perimeter (µm)",
    "Circularity", "Solidity", "Major axis (µm)", "Minor axis (µm)",
    "Aspect ratio", "Eccentricity", "Orientation (°)",
)


def cell_shapes_dataframe(outcomes: Sequence[AnalysisOutcome]):
    """One row per object: the shape of its outline seen from above.

    Joins the *Objects* sheet on Sample and Label. Footprint area is counted in
    pixels; the rest is measured on the traced outline, so a cell a few pixels
    across has a coarse perimeter and circularity.
    """
    import pandas as pd

    rows = []
    for outcome in outcomes:
        for label, (shape, area) in sorted(outcome.cell_shapes.items()):
            rows.append({
                "Sample": outcome.name,
                "Genotype": outcome.genotype,
                "Label": int(label),
                "Footprint area (µm²)": ex._round(area, 3),
                "Perimeter (µm)": ex._round(shape.perimeter_um, 3),
                "Circularity": ex._round(shape.circularity, 4),
                "Solidity": ex._round(shape.solidity, 4),
                "Major axis (µm)": ex._round(shape.major_axis_um, 3),
                "Minor axis (µm)": ex._round(shape.minor_axis_um, 3),
                "Aspect ratio": ex._round(shape.aspect_ratio, 4),
                "Eccentricity": ex._round(shape.eccentricity, 4),
                "Orientation (°)": ex._round(shape.orientation_deg, 2),
            })
    return pd.DataFrame(rows, columns=list(CELL_SHAPE_COLUMNS))


def cell_intensities_dataframe(outcomes: Sequence[AnalysisOutcome]):
    """One row per object, every channel's intensity inside it.

    Joins the *Objects* sheet on Sample and Label. Columns are
    ``<channel> <statistic>``, channels matched by name across samples as in
    the wide region sheets; a sample without a channel leaves its cells blank.
    """
    import pandas as pd

    channels: list[str] = []
    for outcome in outcomes:
        for channel in outcome.cell_intensities:
            if channel not in channels:
                channels.append(channel)
    columns = ["Sample", "Genotype", "Label"] + [
        f"{channel} {stat}" for channel in channels for stat in CELL_INTENSITY_STATS
    ]
    rows = []
    for outcome in outcomes:
        if not outcome.cell_intensities:
            continue
        for object_stat in outcome.stats:
            label = int(object_stat.label)
            row = {"Sample": outcome.name, "Genotype": outcome.genotype, "Label": label}
            for channel, values in outcome.cell_intensities.items():
                for stat in CELL_INTENSITY_STATS:
                    column = values[stat]
                    value = float(column[label]) if label < len(column) else float("nan")
                    row[f"{channel} {stat}"] = ex._round(value, 3)
            rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def write_cluster_labels(
    path: str | Path,
    source_key: str,
    cluster_of: dict[int, int],
    key: str,
    colors: dict[int, str] | None = None,
    names: dict[int, str] | None = None,
    attrs: dict[str, Any] | None = None,
) -> tuple[str, int]:
    """Store a copy of the label map *source_key* with every cell recoloured.

    *cluster_of* maps a cell's label to its cluster number (1, 2, …); cells
    not in it become background. *colors* and *names* (cluster number -> hex
    colour / name) are kept with the map as JSON, so the viewer draws each
    cluster in the colour the explorer gave it. Returns the key written and how
    many cells were placed.
    """
    import json

    masks, source_attrs = ims_store.load_labels(path, source_key)
    if masks is None:
        raise ValueError(f"{Path(path).name} has no label map {source_key!r}")
    highest = int(masks.max()) if masks.size else 0
    lookup = np.zeros(highest + 1, dtype=np.uint16)
    placed = 0
    for label, cluster in cluster_of.items():
        if 0 < int(label) <= highest:
            lookup[int(label)] = int(cluster)
            placed += 1
    extra = {
        "source_labels": source_key,
        "label_colors": json.dumps({str(k): v for k, v in (colors or {}).items()}),
        "label_names": json.dumps({str(k): v for k, v in (names or {}).items()}),
        **(attrs or {}),
    }
    if "channel" in source_attrs:
        extra["channel"] = ims_store._text(source_attrs["channel"])
    written = ims_store.save_labels(
        path, key, lookup[masks], source_attrs.get("voxel_size_um", ()), attrs=extra
    )
    return written, placed


def save_cell_outlines(outcomes: Sequence[AnalysisOutcome], path: str | Path) -> int:
    """Write every cell outline to one ``.npz``. Returns how many were written.

    Arrays: ``samples`` (names), and per outline ``sample_index``, ``label``,
    ``start`` into ``points`` and ``count``; ``points`` is (N, 2) Y/X in µm.
    Not a sheet: a hundred thousand cells of 24 vertices is past what a
    spreadsheet is for.
    """
    samples: list[str] = []
    sample_index, labels, starts, counts, chunks = [], [], [], [], []
    offset = 0
    for outcome in outcomes:
        if not outcome.cell_outlines:
            continue
        samples.append(outcome.name)
        for label, outline in sorted(outcome.cell_outlines.items()):
            sample_index.append(len(samples) - 1)
            labels.append(int(label))
            starts.append(offset)
            counts.append(len(outline))
            chunks.append(np.asarray(outline, dtype=np.float32))
            offset += len(outline)
    if not chunks:
        return 0
    np.savez_compressed(
        str(path),
        samples=np.asarray(samples, dtype=str),
        sample_index=np.asarray(sample_index, dtype=np.int32),
        label=np.asarray(labels, dtype=np.int64),
        start=np.asarray(starts, dtype=np.int64),
        count=np.asarray(counts, dtype=np.int32),
        points=np.vstack(chunks),
    )
    return len(labels)


def load_cell_outlines(path) -> dict[tuple[str, int], np.ndarray]:
    """Read :func:`save_cell_outlines` back as ``(sample, label) -> outline``."""
    with np.load(path, allow_pickle=False) as data:
        samples = [str(name) for name in data["samples"]]
        points = data["points"]
        return {
            (samples[int(i)], int(label)): points[int(start):int(start) + int(count)]
            for i, label, start, count in zip(
                data["sample_index"], data["label"], data["start"], data["count"]
            )
        }


#: Columns of the outline sheet.
OUTLINE_COLUMNS = ("Sample", "Genotype", "Region", "Part", "Vertex", "Y (µm)", "X (µm)")


def outlines_dataframe(outcomes: Sequence[AnalysisOutcome]):
    """Every region outline as vertices, one row per vertex.

    So the outlines travel with the numbers: a plot of the regions can draw each
    one as its own shape without the ``.ims`` files at hand. *Part* numbers the
    outlines of a region drawn in more than one piece.
    """
    import pandas as pd

    rows = []
    for outcome in outcomes:
        if outcome.error:
            continue
        parts: dict[str, int] = {}
        for region in ex.regions_of(outcome):
            part = parts.get(region.name, 0)
            parts[region.name] = part + 1
            for vertex, (y, x) in enumerate(np.asarray(region.vertices_world, dtype=float)):
                rows.append({
                    "Sample": outcome.name,
                    "Genotype": outcome.genotype,
                    "Region": region.name,
                    "Part": part,
                    "Vertex": vertex,
                    "Y (µm)": round(float(y), 3),
                    "X (µm)": round(float(x), 3),
                })
    return pd.DataFrame(rows, columns=list(OUTLINE_COLUMNS))


def workbook_sheets(outcomes: Sequence[AnalysisOutcome]) -> dict[str, Any]:
    """Every sheet of the analysis workbook, in order.

    The first three are exactly the batch export's; the rest are what the
    analysis adds.
    """
    features = features_dataframe(outcomes)
    return {
        "Samples": ex.batch_dataframe(outcomes),
        "Regions": ex.regions_dataframe(outcomes),
        "Objects": ex.objects_dataframe(outcomes),
        "Region features": features,
        "Region intensities": intensity_dataframe(outcomes),
        "PCA matrix": pca_matrix(features),
        "Region outlines": outlines_dataframe(outcomes),
        "Cell shapes": cell_shapes_dataframe(outcomes),
        "Cell intensities": cell_intensities_dataframe(outcomes),
    }
