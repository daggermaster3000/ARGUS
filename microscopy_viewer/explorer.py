"""Browse a plate before opening it: what is in it, what has been segmented, what it looks like.

A plate is not a file you open, it is a few hundred images you open a handful of.
Dropping the whole store on the window builds every mosaic of every cycle, which
on a 4i plate is tens of gigabytes of lazy graph for a look at one well; opening
one well folder means knowing the path and typing it. This module is the middle:
it reads the plate's metadata, says what each image is and whether anything has
already segmented it, renders a miniature small enough to draw in a list, and
builds the layers for the images actually chosen.

It also finds the tables. A batch run writes its object tables into a folder
beside the plate — ``<plate>_nuclei_objects`` next to ``<plate>.zarr`` — and
those folders are what the measurement analysis panel wants to be pointed at.

The survey itself is :func:`microscopy_viewer.batch.survey_plate`: the panel that
runs a plate and the panel that browses it disagreeing about what is in it would
be worse than either of them being wrong.

No Qt here; everything below is testable headless.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from .batch import ImageJob, PlateSurvey, survey_plate  # noqa: F401 - re-exported
from .loaders.layer_spec import LayerSpec
from .loaders.ome_zarr import (
    LABELS_GROUP,
    _label_names,
    _read_image,
    child_group,
    open_group,
)
from .utils import get_logger

logger = get_logger("explorer")

#: Longest edge of a miniature, in pixels. Big enough to tell an empty well from
#: a full one and to see where the organoids sit; small enough that a hundred of
#: them cost less than one full-resolution tile.
THUMBNAIL_PX = 160

#: Percentiles the miniature is stretched between. A plate image is mostly
#: background, so scaling to the true maximum leaves a black square with three
#: bright specks in it.
THUMBNAIL_PERCENTILES = (2.0, 99.5)

#: Suffixes counted as measurement tables when looking for analysis folders.
TABLE_SUFFIXES = (".csv", ".tsv", ".xlsx", ".xls")

#: How deep a sibling folder is searched for tables. Batch output is flat, and
#: walking an arbitrary neighbour of the plate to the bottom is not worth it.
ANALYSIS_DEPTH = 2


# ---------------------------------------------------------------------------
# What is in the plate
# ---------------------------------------------------------------------------


def label_sets(job: ImageJob) -> tuple[str, ...]:
    """Names of the segmentations stored inside *job*'s image, newest first on disk order.

    Read from the store rather than from a run's report: the point of the panel is
    to show what is there now, including label sets written by a run that happened
    last week or by somebody else's pipeline.
    """
    try:
        group = open_group(job.path)
    except Exception:  # pragma: no cover - an image listed but not readable
        logger.debug("%s: could not be opened for its label sets", job.component, exc_info=True)
        return ()
    if child_group(group, LABELS_GROUP) is None:
        return ()
    try:
        return tuple(_label_names(group))
    except Exception:  # pragma: no cover - a malformed labels group
        logger.debug("%s: label sets could not be listed", job.component, exc_info=True)
        return ()


def describe_job(job: ImageJob, labels: Sequence[str] | None = None) -> dict:
    """One row of the image list: where it is, how big, and what has been done to it."""
    names = tuple(labels) if labels is not None else label_sets(job)
    level = job.levels[0] if job.levels else None
    shape = "" if level is None else " x ".join(str(int(n)) for n in level.shape)
    return {
        "well": job.well,
        "acquisition": "" if job.acquisition is None else int(job.acquisition),
        "field": job.field_path,
        "component": job.component,
        "channels": len(job.channels),
        "channel_names": ", ".join(
            channel.label or channel.wavelength_id or str(channel.index)
            for channel in job.channels
        ),
        "shape": shape,
        "levels": len(job.levels),
        "segmentations": ", ".join(names),
        "n_segmentations": len(names),
    }


def describe_plate(survey: PlateSurvey) -> list[dict]:
    """A row per image of the plate, in the order the plate lists them."""
    return [describe_job(job) for job in survey.jobs]


# ---------------------------------------------------------------------------
# The miniature
# ---------------------------------------------------------------------------


def _coarsest(job: ImageJob) -> int:
    """Index of the smallest pyramid level, which is the one a miniature reads."""
    return max(0, len(job.levels) - 1)


def thumbnail(
    job: ImageJob,
    channel_index: int | None = None,
    max_px: int = THUMBNAIL_PX,
) -> np.ndarray:
    """A small 2D preview of one image, read from the coarsest pyramid level.

    Reading the bottom of the pyramid is the whole point: a 12000 x 12000 image
    has a 375 x 375 level sitting in the same store, so a preview costs a few
    hundred kilobytes rather than 288 MB. A Z stack is projected at maximum, which
    is what makes an organoid visible in one plane.
    """
    if not job.levels:
        raise ValueError(f"{job.component} has no pyramid levels")
    index = 0 if channel_index is None else int(channel_index)
    group = open_group(job.path)
    array = group[job.level(_coarsest(job)).path]

    selector = tuple(
        min(index, array.shape[axis_index] - 1) if axis == "C" else slice(None)
        for axis_index, axis in enumerate(job.axes)
    )
    plane = np.asarray(array[selector])
    while plane.ndim > 2:
        plane = plane.max(axis=0)
    if plane.ndim < 2:
        raise ValueError(f"{job.component} is not an image: {plane.ndim}D after projection")

    step = max(1, int(max(plane.shape) // max(1, int(max_px))))
    return plane[::step, ::step]


def stretch(plane: np.ndarray, percentiles: Sequence[float] = THUMBNAIL_PERCENTILES) -> np.ndarray:
    """A 2D array as 8-bit grey, contrast-stretched between *percentiles*."""
    array = np.asarray(plane, dtype=np.float32)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return np.zeros(array.shape, dtype=np.uint8)
    low, high = (float(v) for v in np.percentile(finite, list(percentiles)))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low, high = float(finite.min()), float(finite.max())
    if high <= low:
        return np.zeros(array.shape, dtype=np.uint8)
    scaled = (array - low) / (high - low)
    return (np.clip(scaled, 0.0, 1.0) * 255.0).astype(np.uint8)


# ---------------------------------------------------------------------------
# Opening what was chosen
# ---------------------------------------------------------------------------


def layer_name(job: ImageJob, fields_in_well: int = 1) -> str:
    """What the image's layers are called: ``G/07 :: cycle 1``.

    The same shape the well reader uses, so a table written by a batch run still
    finds its layer through :func:`microscopy_viewer.analysis.match_layer`, and so
    that opening a well through this panel and by dropping its folder give the
    layers the same names.
    """
    parts = [job.well]
    if job.acquisition is not None:
        parts.append(f"cycle {job.acquisition}")
    if fields_in_well > 1:
        parts.append(f"field {job.field_path}")
    return " :: ".join(parts)


def specs_for(
    jobs: Sequence[ImageJob],
    plate_path: Path | None = None,
    with_labels: bool = True,
) -> tuple[list[LayerSpec], list[str]]:
    """Layers for the chosen images, and a note per image that could not be read.

    Returns ``(specs, problems)`` rather than raising: a plate with one unreadable
    cycle in it should open the other six.
    """
    fields: dict[str, int] = {}
    for job in jobs:
        fields[f"{job.well}/{job.acquisition}"] = fields.get(f"{job.well}/{job.acquisition}", 0) + 1

    specs: list[LayerSpec] = []
    problems: list[str] = []
    for job in jobs:
        root = plate_path or _plate_root(job)
        try:
            group = open_group(job.path)
            built = _read_image(
                root,
                group,
                layer_name(job, fields.get(f"{job.well}/{job.acquisition}", 1)),
                file_format="OME-Zarr (HCS plate)",
                keep_name=True,
            )
        except Exception as exc:  # noqa: BLE001 - reported per image
            logger.exception("explorer: %s could not be read", job.component)
            problems.append(f"{job.component}: {type(exc).__name__}: {exc}")
            continue
        if not with_labels:
            built = [spec for spec in built if spec.layer_type != "labels"]
        for spec in built:
            spec.metadata.extra.setdefault("Well", job.well)
            if job.acquisition is not None:
                spec.metadata.extra.setdefault("Acquisition", str(job.acquisition))
        specs.extend(built)
    return specs, problems


def _plate_root(job: ImageJob) -> Path:
    """The ``.zarr`` the image sits in, walked up from its path inside the store."""
    root = job.path
    for _component in job.component.split("/"):
        root = root.parent
    return root


# ---------------------------------------------------------------------------
# The tables beside the plate
# ---------------------------------------------------------------------------


def _tables_in(folder: Path, depth: int = ANALYSIS_DEPTH) -> list[Path]:
    """Measurement tables in *folder*, at most *depth* levels down."""
    found: list[Path] = []
    try:
        entries = sorted(folder.iterdir())
    except OSError:
        return found
    for entry in entries:
        if entry.is_file() and entry.suffix.lower() in TABLE_SUFFIXES:
            found.append(entry)
        elif entry.is_dir() and depth > 1 and entry.suffix.lower() != ".zarr":
            found.extend(_tables_in(entry, depth - 1))
    return found


def analysis_folders(plate_path: str | Path) -> list[Path]:
    """Folders of measurement tables sitting beside the plate, best match first.

    A batch run writes ``<plate>_<label set>_objects`` next to ``<plate>.zarr``,
    so the folders that start with the plate's own stem are listed first and any
    other neighbour holding tables after them. Nothing is opened here; this is
    the list the analysis panel offers.
    """
    plate = Path(plate_path)
    parent = plate.parent
    # Both checks matter: without the first, a mistyped plate name lists whatever
    # folders happen to sit in the directory it would have been in, and offers
    # another plate's tables as though they were this one's.
    if not plate.exists() or not parent.is_dir():
        return []

    stem = plate.stem
    scored: list[tuple[int, str, Path]] = []
    for entry in sorted(parent.iterdir()):
        if not entry.is_dir() or entry.resolve() == plate.resolve():
            continue
        if entry.suffix.lower() == ".zarr":
            continue
        if not _tables_in(entry):
            continue
        # Folders this plate's own runs wrote come first; a neighbour that merely
        # holds CSVs might belong to a different plate entirely.
        scored.append((0 if entry.name.startswith(stem) else 1, entry.name.lower(), entry))

    folders = [entry for _rank, _name, entry in sorted(scored, key=lambda item: item[:2])]
    logger.info("found %d analysis folder(s) beside %s", len(folders), plate.name)
    return folders


def analysis_tables(folder: str | Path) -> list[Path]:
    """Measurement tables inside one analysis folder, per-image tables first.

    A run writes one table per image plus a single ``*_summary.csv``; the summary
    is one row per image rather than one row per object, so it sorts last — it is
    not what someone opening the folder is usually after.
    """
    tables = _tables_in(Path(folder))
    return sorted(tables, key=lambda path: (path.stem.endswith("_summary"), path.name.lower()))
