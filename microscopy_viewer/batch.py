"""Segment every image of an HCS plate in one unattended run.

A high-content plate is not one image, it is a few hundred: wells across the
plate, acquisitions (4i cycles) within a well, fields within an acquisition. The
Segmentation panel does one at a time, which is right for choosing settings and
wrong for using them — nobody is going to sit through 320 clicks.

This module is that loop, and it is deliberately Qt-free so the same code can be
driven from a panel, a notebook or a script:

* :func:`survey_plate` walks the store and returns one :class:`ImageJob` per
  image, with its channels, shape and calibrated scale already read.
* :class:`ChannelPick` says *which* channel to segment in a way that survives the
  plate. Channel **labels change between 4i cycles** — ``Ab1_DAPI``, ``Ab2_DAPI``,
  … — so matching by name has to be a substring, while ``wavelength_id`` stays
  put and is the reliable key.
* :func:`iter_batch` runs them, yielding one :class:`ImageOutcome` as each image
  finishes. A generator rather than a callback because that is what lets a GUI
  show progress, and ``worker.quit()`` stop the run, without either side knowing
  about the other.

Masks are written back into the plate as NGFF ``labels`` groups — the same layout
Fractal and ome-zarr-py expect — so the result is readable by anything that reads
the plate, this viewer included.

Nothing here overwrites pixel data. The only writes are new ``labels`` subgroups,
and an existing label set is skipped rather than replaced unless
:attr:`BatchSettings.overwrite` says otherwise.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Mapping
from typing import Any, Callable, Iterator, Sequence

import numpy as np

from . import segmentation as sg
from .loaders.ome_zarr import (
    LABELS_GROUP,
    _dataset_scale,
    _multiscales,
    child_group,
    ngff_attrs,
    open_group,
)
from .utils import get_logger, normalise_axes

logger = get_logger("batch")

#: Match on the ``wavelength_id`` in the ``omero`` block: ``A01_C01``. The one key
#: that is stable across 4i cycles, and the default for that reason.
BY_WAVELENGTH = "wavelength"
#: Match on the channel label, case-insensitively and as a substring, so ``dapi``
#: finds ``Ab1_DAPI`` in cycle 1 and ``Ab7_DAPI`` in cycle 7.
BY_LABEL = "label"
#: Match on position in the channel axis. Fast, and wrong the moment one cycle
#: was acquired with the channels in a different order.
BY_INDEX = "index"

CHANNEL_KEYS = (BY_WAVELENGTH, BY_LABEL, BY_INDEX)

#: Default name of the label set written into each image.
DEFAULT_LABEL_NAME = "nuclei"

#: NGFF version written on the label groups. 0.4 is what the plates we read carry,
#: and what the readers in the wild still expect.
LABEL_VERSION = "0.4"

STATUS_DONE = "done"
STATUS_SKIPPED = "skipped"
STATUS_FAILED = "failed"


# ---------------------------------------------------------------------------
# What is in the plate
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChannelInfo:
    """One channel of one image, as the ``omero`` block describes it."""

    index: int
    label: str = ""
    wavelength_id: str = ""

    def describe(self) -> str:
        if self.label and self.wavelength_id:
            return f"{self.wavelength_id} — {self.label}"
        return self.label or self.wavelength_id or f"channel {self.index}"


@dataclass(frozen=True)
class ChannelPick:
    """How to find the wanted channel in an image whose channel list may differ.

    Unset (``value`` empty, or an index below zero) means "no channel", which is
    how the optional nuclear and measurement channels say they were left alone.
    """

    key: str = BY_WAVELENGTH
    value: str = ""

    @property
    def is_set(self) -> bool:
        return bool(str(self.value).strip())

    def resolve(self, channels: Sequence[ChannelInfo]) -> ChannelInfo | None:
        """The channel this pick names, or ``None`` when the image has no such channel."""
        if not self.is_set:
            return None
        wanted = str(self.value).strip()
        if self.key == BY_INDEX:
            try:
                index = int(wanted)
            except ValueError:
                return None
            return next((channel for channel in channels if channel.index == index), None)
        if self.key == BY_WAVELENGTH:
            lowered = wanted.lower()
            return next(
                (channel for channel in channels if channel.wavelength_id.lower() == lowered), None
            )
        lowered = wanted.lower()
        # Exact first: a plate can hold both "DAPI" and "DAPI_2", and a substring
        # search would hand back whichever happened to be listed first.
        exact = next((channel for channel in channels if channel.label.lower() == lowered), None)
        if exact is not None:
            return exact
        return next((channel for channel in channels if lowered in channel.label.lower()), None)

    def describe(self) -> str:
        if not self.is_set:
            return "none"
        return f"{self.value} (by {self.key})"


@dataclass(frozen=True)
class LevelInfo:
    """One pyramid level of an image: where it is, how big, and at what scale."""

    path: str
    shape: tuple[int, ...]
    scale: tuple[float, ...]


@dataclass(frozen=True)
class ImageJob:
    """One image of the plate — one well, one acquisition, one field."""

    path: Path
    #: Path of the image group inside the store, e.g. ``B/02/0``.
    component: str
    well: str
    row: str
    column: str
    #: Acquisition id from the well metadata, when the plate records one.
    acquisition: int | None
    #: The image's own path within the well, which for a multi-field plate is the
    #: field index and for a 4i plate is the cycle.
    field_path: str
    channels: tuple[ChannelInfo, ...]
    axes: str
    levels: tuple[LevelInfo, ...] = ()
    chunks: tuple[int, ...] = ()

    @property
    def shape(self) -> tuple[int, ...]:
        return self.levels[0].shape if self.levels else ()

    def level(self, index: int) -> LevelInfo:
        """A pyramid level, clamped — a plate with fewer levels than asked for is
        a reason to segment the smallest one, not to fail the image."""
        if not self.levels:
            raise ValueError(f"{self.component} has no pyramid levels")
        return self.levels[max(0, min(int(index), len(self.levels) - 1))]

    def voxel_size_um(self, level: int = 0) -> tuple[float, float, float]:
        """``(z, y, x)`` in µm at *level*, from the NGFF scale transformation."""
        scale = self.level(level).scale
        by_axis = dict(zip(self.axes, scale))
        return (
            float(by_axis.get("Z", 1.0) or 1.0),
            float(by_axis.get("Y", 1.0) or 1.0),
            float(by_axis.get("X", 1.0) or 1.0),
        )

    def describe(self) -> str:
        if self.acquisition is not None:
            return f"{self.well} · cycle {self.acquisition}"
        return f"{self.well} · {self.field_path}"


@dataclass(frozen=True)
class PlateSurvey:
    """Everything a run needs to know about the plate before it starts."""

    path: Path
    name: str
    jobs: tuple[ImageJob, ...]

    @property
    def wells(self) -> tuple[str, ...]:
        seen: dict[str, None] = {}
        for job in self.jobs:
            seen.setdefault(job.well, None)
        return tuple(seen)

    @property
    def acquisitions(self) -> tuple[int | None, ...]:
        seen: dict[Any, None] = {}
        for job in self.jobs:
            seen.setdefault(job.acquisition, None)
        return tuple(sorted(seen, key=lambda value: (value is None, value)))

    def channels(self) -> tuple[ChannelInfo, ...]:
        """The union of every image's channels, keyed by wavelength then label.

        Built as a union rather than read off the first image because the label
        differs per cycle and the panel has to offer something that holds for all
        of them. The entry keeps the *first* label seen, so the list reads like
        cycle 1 while matching on the stable key.
        """
        seen: dict[str, ChannelInfo] = {}
        for job in self.jobs:
            for channel in job.channels:
                key = channel.wavelength_id or channel.label or str(channel.index)
                seen.setdefault(key, channel)
        return tuple(seen.values())

    def labels_present(self, name: str) -> tuple[ImageJob, ...]:
        """Images that already carry a label set called *name*."""
        return tuple(job for job in self.jobs if has_labels(job, name))


def _channels_of(group) -> tuple[ChannelInfo, ...]:
    """Channel list from the ``omero`` block, or positional entries when absent."""
    omero = ngff_attrs(group).get("omero")
    entries = omero.get("channels") if isinstance(omero, dict) else None
    if not isinstance(entries, list) or not entries:
        return ()
    channels = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            channels.append(ChannelInfo(index=index))
            continue
        channels.append(
            ChannelInfo(
                index=index,
                label=str(entry.get("label", "") or ""),
                wavelength_id=str(entry.get("wavelength_id", "") or ""),
            )
        )
    return tuple(channels)


def _levels_of(group) -> tuple[tuple[LevelInfo, ...], str, tuple[int, ...]]:
    """Pyramid levels, axis letters and level-0 chunking of an image group."""
    entry = _multiscales(group)
    if entry is None:
        raise ValueError("no multiscales metadata")
    datasets = entry.get("datasets") or []

    axes_spec = entry.get("axes")
    letters = ""
    if isinstance(axes_spec, list):
        letters = "".join(
            str(axis.get("name", "q")) if isinstance(axis, dict) else str(axis)
            for axis in axes_spec
        )

    levels: list[LevelInfo] = []
    chunks: tuple[int, ...] = ()
    for dataset in datasets:
        component = str(dataset.get("path", ""))
        try:
            array = group[component]
        except KeyError:
            logger.warning("multiscales references missing array %r", component)
            continue
        shape = tuple(int(n) for n in array.shape)
        scale = _dataset_scale(dataset) or [1.0] * len(shape)
        levels.append(LevelInfo(path=component, shape=shape, scale=tuple(float(s) for s in scale)))
        if not chunks:
            chunks = tuple(int(n) for n in (getattr(array, "chunks", None) or ()))
    if not levels:
        raise ValueError("none of the multiscales datasets could be opened")
    return tuple(levels), normalise_axes(letters), chunks


def survey_plate(path: str | Path) -> PlateSurvey:
    """Walk an HCS plate and describe every image in it.

    Reads metadata only: no pixels are touched, so this is fast even on a plate
    of a few hundred gigabytes, and it is what the panel calls to fill its lists.
    """
    root = Path(path)
    group = open_group(root)
    attrs = ngff_attrs(group)
    plate = attrs.get("plate")
    if not isinstance(plate, dict):
        raise ValueError(f"{root.name} is not an OME-Zarr plate — it has no plate metadata")

    rows = [str(entry.get("name", index)) for index, entry in enumerate(plate.get("rows") or [])]
    columns = [
        str(entry.get("name", index)) for index, entry in enumerate(plate.get("columns") or [])
    ]

    jobs: list[ImageJob] = []
    missing: list[str] = []
    for entry in plate.get("wells") or []:
        if not isinstance(entry, dict):
            continue
        well_path = str(entry.get("path", "")).strip("/")
        if not well_path:
            row_index, column_index = entry.get("rowIndex"), entry.get("columnIndex")
            if row_index is None or column_index is None:
                continue
            well_path = f"{rows[int(row_index)]}/{columns[int(column_index)]}"
        well_group = child_group(group, well_path)
        if well_group is None:
            missing.append(well_path)
            continue

        row, _, column = well_path.partition("/")
        well_block = ngff_attrs(well_group).get("well")
        images = well_block.get("images") if isinstance(well_block, dict) else None
        if not isinstance(images, list) or not images:
            images = [{"path": key} for key, _value in well_group.groups()]

        for image_entry in images:
            if not isinstance(image_entry, dict):
                continue
            field_path = str(image_entry.get("path", "")).strip("/")
            image_group = child_group(well_group, field_path) if field_path else None
            if image_group is None:
                missing.append(f"{well_path}/{field_path}")
                continue
            acquisition = image_entry.get("acquisition")
            try:
                levels, axes, chunks = _levels_of(image_group)
            except ValueError as exc:
                logger.warning("%s/%s: %s", well_path, field_path, exc)
                continue
            jobs.append(
                ImageJob(
                    path=root / well_path / field_path,
                    component=f"{well_path}/{field_path}",
                    well=well_path,
                    row=row,
                    column=column,
                    acquisition=None if acquisition is None else int(acquisition),
                    field_path=field_path,
                    channels=_channels_of(image_group),
                    axes=axes,
                    levels=levels,
                    chunks=chunks,
                )
            )

    if missing:
        shown = ", ".join(missing[:6]) + ("…" if len(missing) > 6 else "")
        logger.info("plate: %d image(s) in the metadata are not in the store (%s)",
                    len(missing), shown)

    name = str(plate.get("name") or "").strip() or root.stem
    logger.info("surveyed %s: %d image(s) in %d well(s)", name, len(jobs), len({j.well for j in jobs}))
    return PlateSurvey(path=root, name=name, jobs=tuple(jobs))


def select_jobs(
    survey: PlateSurvey,
    wells: Sequence[str] | None = None,
    acquisitions: Sequence[int | None] | None = None,
) -> tuple[ImageJob, ...]:
    """The subset of the plate to run. ``None`` for either filter means "all"."""
    chosen = list(survey.jobs)
    if wells is not None:
        wanted = {str(well) for well in wells}
        chosen = [job for job in chosen if job.well in wanted]
    if acquisitions is not None:
        allowed = {None if value is None else int(value) for value in acquisitions}
        chosen = [job for job in chosen if job.acquisition in allowed]
    return tuple(chosen)


# ---------------------------------------------------------------------------
# Reading one image and writing its labels
# ---------------------------------------------------------------------------


def _spatial_axes(axes: str) -> str:
    return "".join(axis for axis in axes if axis != "C")


def read_channel(job: ImageJob, channel: ChannelInfo, level: int = 0) -> np.ndarray:
    """One channel of one image, at *level*, as a plain in-memory array.

    Loaded eagerly rather than as a dask array: Cellpose reads the whole thing
    anyway, and a lazy array crossing a thread boundary buys nothing but a slower
    first touch.
    """
    group = open_group(job.path)
    info = job.level(level)
    array = group[info.path]
    selector = tuple(
        channel.index if axis == "C" else slice(None) for axis in job.axes
    )
    return np.asarray(array[selector])


def has_labels(job: ImageJob, name: str) -> bool:
    """Whether *job* already carries a label set called *name* with data in it."""
    target = job.path / LABELS_GROUP / name
    if not target.is_dir():
        return False
    # A group left behind by an interrupted write has attributes and no arrays;
    # treating that as "already done" would silently skip the image forever.
    return any(child.is_dir() for child in target.iterdir() if not child.name.startswith("."))


def _label_pyramid(
    masks: np.ndarray, job: ImageJob, level: int
) -> list[tuple[LevelInfo, np.ndarray]]:
    """The mask at every pyramid level the image has below *level*.

    Subsampled on a stride rather than averaged: a label map has no meaningful
    mean, and taking every n-th voxel keeps the values exactly as they are. The
    step per axis comes from the image's own scales, so the label pyramid lines
    up with the channel pyramid instead of merely resembling it.
    """
    spatial = _spatial_axes(job.axes)
    base = job.level(level)
    base_scale = dict(zip(job.axes, base.scale))

    built: list[tuple[LevelInfo, np.ndarray]] = []
    for info in job.levels[max(0, min(int(level), len(job.levels) - 1)) :]:
        scale = dict(zip(job.axes, info.scale))
        steps = []
        for axis in spatial:
            reference = float(base_scale.get(axis, 1.0)) or 1.0
            steps.append(max(1, int(round(float(scale.get(axis, 1.0)) / reference))))
        data = masks[tuple(slice(None, None, step) for step in steps)]

        # Trim to the shape the image level actually has: with an odd extent the
        # writer that made the plate may have rounded the other way.
        wanted = tuple(
            int(size) for axis, size in zip(job.axes, info.shape) if axis != "C"
        )
        if len(wanted) == data.ndim and any(a > b for a, b in zip(data.shape, wanted)):
            data = data[tuple(slice(0, min(a, b)) for a, b in zip(data.shape, wanted))]

        label_scale = tuple(float(scale.get(axis, 1.0)) for axis in spatial)
        built.append((LevelInfo(path=info.path, shape=data.shape, scale=label_scale), data))
    return built


def _unit_of(axis: str) -> str:
    return "micrometer" if axis in "ZYX" else ("second" if axis == "T" else "")


def write_labels(
    job: ImageJob,
    name: str,
    masks: np.ndarray,
    level: int = 0,
    overwrite: bool = False,
) -> Path:
    """Write *masks* into the image as an NGFF ``labels`` group and return its path.

    The pixels of the image are never opened for writing; this adds
    ``<image>/labels/<name>`` beside them, with its own multiscale pyramid and the
    ``image-label`` marker that says what it is.
    """
    import zarr

    target = job.path / LABELS_GROUP / name
    if has_labels(job, name) and not overwrite:
        raise FileExistsError(f"{job.component} already has a label set called {name!r}")

    group = zarr.open_group(str(job.path), mode="a")
    labels_group = group.require_group(LABELS_GROUP)
    existing = ngff_attrs(labels_group).get(LABELS_GROUP)
    listed = [str(entry) for entry in existing] if isinstance(existing, list) else []
    if name not in listed:
        listed.append(name)
    labels_group.attrs[LABELS_GROUP] = listed

    label_group = labels_group.require_group(name)
    spatial = _spatial_axes(job.axes)
    chunk_by_axis = dict(zip(job.axes, job.chunks)) if job.chunks else {}

    datasets = []
    for info, data in _label_pyramid(masks, job, level):
        payload = np.ascontiguousarray(data, dtype=np.uint32)
        chunks = tuple(
            max(1, min(int(chunk_by_axis.get(axis, size)), int(size)))
            for axis, size in zip(spatial, payload.shape)
        )
        _create_array(label_group, info.path, payload, chunks)
        datasets.append(
            {
                "path": info.path,
                "coordinateTransformations": [
                    {"type": "scale", "scale": [float(value) for value in info.scale]}
                ],
            }
        )

    label_group.attrs["multiscales"] = [
        {
            "version": LABEL_VERSION,
            "name": name,
            "axes": [
                {"name": axis.lower(), "type": "space" if axis in "ZYX" else "time",
                 **({"unit": _unit_of(axis)} if _unit_of(axis) else {})}
                for axis in spatial
            ],
            "datasets": datasets,
        }
    ]
    # "source.image" is a path relative to the label group: two levels up from
    # labels/<name> is the image the masks were computed from.
    label_group.attrs["image-label"] = {"version": LABEL_VERSION, "source": {"image": "../../"}}
    logger.info("wrote %s labels to %s (%d level(s))", name, job.component, len(datasets))
    return target


def _create_array(group, name: str, data: np.ndarray, chunks):
    """Write one array, on either zarr 2 or zarr 3.

    ``Group.create_dataset`` is gone in zarr 3 and ``create_array`` does not exist
    in zarr 2, and the store's own format is whatever the plate was written with,
    so ask the installed version which one it has.
    """
    if hasattr(group, "create_array"):
        return group.create_array(name, data=data, chunks=chunks, overwrite=True)
    return group.create_dataset(name, data=data, chunks=chunks, overwrite=True)


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BatchSettings:
    """A batch run, on top of the per-image Cellpose settings."""

    segmentation: sg.SegmentationSettings = field(default_factory=sg.SegmentationSettings)
    #: The channel to segment. Required.
    channel: ChannelPick = field(default_factory=ChannelPick)
    #: Optional second stain handed to Cellpose alongside it, for whole cells.
    nuclei: ChannelPick = field(default_factory=ChannelPick)
    #: Channel the per-object intensities are read from; unset means the
    #: segmented channel.
    measure: ChannelPick = field(default_factory=ChannelPick)
    #: Measure every channel of the image, not only the one above, adding a set of
    #: intensity columns per channel named after it. On a 4i cycle that is four
    #: stains per object for one segmentation, and it is the only way to ask
    #: whether the nuclei picked out by DAPI are the ones carrying the reporter.
    measure_all_channels: bool = False
    #: Name of the label set written into each image.
    label_name: str = DEFAULT_LABEL_NAME
    #: Pyramid level segmented. 0 is full resolution; 1 is half, four times
    #: faster, and enough when objects are tens of pixels across.
    level: int = 0
    #: Replace a label set that is already there instead of skipping the image.
    overwrite: bool = False
    #: Write a per-image object table, and a plate-level summary, next to the plate.
    write_tables: bool = True
    #: Also write each object table as ``.h5ad`` beside its CSV, for squidpy and
    #: the rest of the single-cell stack. Costs a second per image and means the
    #: spatial analysis does not start with a folder of CSVs to convert by hand.
    write_anndata: bool = False
    #: Write one ``.h5ad`` for the whole run rather than one per image, with the
    #: well and cycle in ``obs["image"]``. A plate is one experiment, and a folder
    #: of forty-four files is forty-four files to concatenate before anything can
    #: be asked about the plate as a whole.
    anndata_single_file: bool = False

    def anndata_path(self, survey: "PlateSurvey") -> Path:
        """Where the combined ``.h5ad`` for a whole run goes."""
        return self.tables_root(survey) / f"{self.label_name}_objects.h5ad"
    #: Where those tables go. ``None`` puts them beside the plate.
    table_dir: Path | None = None

    def tables_root(self, survey: PlateSurvey) -> Path:
        """Where the object tables go for a whole run."""
        if self.table_dir is not None:
            return Path(self.table_dir)
        return survey.path.parent / f"{survey.path.stem}_{self.label_name}_objects"

    def table_path(self, job: ImageJob) -> Path:
        """Where one image's object table goes.

        Derived from the job rather than the survey so that a single image can be
        run without one — the plate root is however many components up from the
        image group its path inside the store says it is.
        """
        if self.table_dir is not None:
            root = Path(self.table_dir)
        else:
            plate = job.path
            for _component in job.component.split("/"):
                plate = plate.parent
            root = plate.parent / f"{plate.stem}_{self.label_name}_objects"
        return root / f"{_table_name(job)}.csv"


@dataclass
class ImageOutcome:
    """What happened to one image."""

    job: ImageJob
    status: str
    n_objects: int = 0
    #: Objects the solidity filter threw out before they reached the table.
    n_dropped: int = 0
    elapsed_s: float = 0.0
    message: str = ""
    table_path: Path | None = None
    summary: dict[str, float] = field(default_factory=dict)
    #: The channel the unqualified intensity columns were read from, and the
    #: channels that got columns of their own. Recorded per image because a 4i
    #: plate names the same stain differently in every cycle, so "which channel is
    #: this number" cannot be answered from the settings alone.
    measured_channel: str = ""
    measured_extra: tuple[str, ...] = ()
    #: The object table as a DataFrame, kept only while a run is building one
    #: combined AnnData and dropped as soon as it has. Holding every image's
    #: numbers for a whole plate is about sixty megabytes; holding the masks too
    #: would be a hundred gigabytes, which is why this is the frame and not the
    #: result.
    frame: Any = None

    @property
    def ok(self) -> bool:
        return self.status == STATUS_DONE


@dataclass
class BatchReport:
    """The run as a whole."""

    outcomes: list[ImageOutcome] = field(default_factory=list)
    elapsed_s: float = 0.0
    summary_path: Path | None = None
    #: Where the one combined ``.h5ad`` went, when the run was asked for one.
    anndata_path: Path | None = None
    cancelled: bool = False

    def counted(self, status: str) -> int:
        return sum(1 for outcome in self.outcomes if outcome.status == status)

    @property
    def total_objects(self) -> int:
        return sum(outcome.n_objects for outcome in self.outcomes if outcome.ok)

    def describe(self) -> str:
        done, skipped, failed = (
            self.counted(STATUS_DONE),
            self.counted(STATUS_SKIPPED),
            self.counted(STATUS_FAILED),
        )
        parts = [f"{done} image(s) segmented", f"{self.total_objects} object(s)"]
        if skipped:
            parts.append(f"{skipped} skipped")
        if failed:
            parts.append(f"{failed} failed")
        parts.append(f"{self.elapsed_s / 60:.1f} min")
        head = "Cancelled after " if self.cancelled else ""
        return head + ", ".join(parts) + "."


class _LazyChannels(Mapping):
    """The image's other channels, read one at a time as they are measured.

    A 12000x12000 plate image is 288 MB a channel; handing four of them to the
    measurement as a plain dict would hold a gigabyte at once for numbers that are
    consumed one channel at a time. Iterating this reads a channel, measures it
    and lets it go.
    """

    def __init__(self, job: ImageJob, channels: Sequence[ChannelInfo], level: int):
        self._job = job
        self._channels = {_channel_name(channel): channel for channel in channels}
        self._level = level

    def __getitem__(self, key: str) -> np.ndarray:
        return read_channel(self._job, self._channels[key], self._level)

    def __iter__(self):
        return iter(self._channels)

    def __len__(self) -> int:
        return len(self._channels)


def _channel_name(channel: ChannelInfo) -> str:
    """Short, stable column name for a channel: its label, else its wavelength."""
    return channel.label or channel.wavelength_id or f"channel {channel.index}"


def _table_name(job: ImageJob) -> str:
    """A file name that sorts by well and cannot collide across the plate."""
    return job.component.replace("/", "_")


def _write_table(
    stats, path: Path, anndata: bool = False, keep_frame: bool = False
) -> tuple[Path | None, str, Any]:
    """Write one image's object table. Returns ``(path, note)``.

    The AnnData copy is written from the same frame rather than by reading the CSV
    back: a float that has been through a text file is not the float that was
    measured, and the spatial analysis should not start from a rounded one.
    """
    try:
        frame = sg.object_dataframe(stats)
    except ImportError:
        logger.warning("pandas is not installed; the object tables were not written")
        return None, "pandas is not installed, so no object table was written", None
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8")
    kept = frame if keep_frame else None
    if not anndata:
        return path, "", kept
    try:
        from .analysis import anndata_path, write_anndata

        write_anndata(frame, anndata_path(path), label_column="Label", source=path)
    except ImportError:
        return path, "anndata is not installed, so no .h5ad was written beside the table", kept
    except Exception as exc:  # noqa: BLE001 - the CSV is already safely on disk
        logger.exception("could not write the AnnData for %s", path.name)
        return path, f"the .h5ad could not be written ({type(exc).__name__}: {exc})", kept
    return path, "", kept


def segment_job(
    job: ImageJob,
    settings: BatchSettings,
    progress: Callable[[str], None] | None = None,
) -> ImageOutcome:
    """Segment one image and write its labels. Never raises; failures come back
    as an outcome so one bad image does not end a run of three hundred."""
    started = time.perf_counter()
    try:
        channel = settings.channel.resolve(job.channels)
        if channel is None:
            return ImageOutcome(
                job=job,
                status=STATUS_FAILED,
                message=f"no channel matching {settings.channel.describe()}",
                elapsed_s=time.perf_counter() - started,
            )
        if has_labels(job, settings.label_name) and not settings.overwrite:
            return ImageOutcome(
                job=job,
                status=STATUS_SKIPPED,
                message=f"already has labels called {settings.label_name!r}",
                elapsed_s=time.perf_counter() - started,
            )

        if progress is not None:
            progress(f"{job.describe()}: reading {channel.describe()}")
        image = read_channel(job, channel, settings.level)

        nuclei = None
        nuclear = settings.nuclei.resolve(job.channels)
        if nuclear is not None and nuclear.index != channel.index:
            nuclei = read_channel(job, nuclear, settings.level)

        result = sg.segment_volume(
            image,
            job.voxel_size_um(settings.level),
            settings=settings.segmentation,
            progress=progress,
            nuclei=nuclei,
        )

        write_labels(
            job,
            settings.label_name,
            result.masks,
            level=settings.level,
            overwrite=settings.overwrite,
        )

        notes = list(result.warnings)
        signal, measured_channel = image, channel
        wanted = settings.measure
        if wanted.is_set:
            measured = wanted.resolve(job.channels)
            if measured is None:
                # Silence here would put the segmented channel's numbers in a
                # column the user believes holds the reporter's.
                notes.append(
                    f"No channel matching {wanted.describe()} in this image; intensities were "
                    f"measured on {_channel_name(channel)}, the segmented channel."
                )
            elif measured.index != channel.index:
                candidate = read_channel(job, measured, settings.level)
                if candidate.shape != image.shape:
                    notes.append(
                        f"{_channel_name(measured)} is {tuple(candidate.shape)} and "
                        f"{_channel_name(channel)} is {tuple(image.shape)}; intensities were "
                        "measured on the segmented channel instead."
                    )
                else:
                    signal, measured_channel = candidate, measured

        extra = None
        if settings.measure_all_channels:
            others = [
                other for other in job.channels if other.index != measured_channel.index
            ]
            if others:
                if progress is not None:
                    progress(f"{job.describe()}: measuring {len(others)} further channel(s)")
                extra = _LazyChannels(job, others, settings.level)

        stats = sg.object_table(
            result.masks,
            signal,
            result.voxel_size_um[-result.masks.ndim :],
            shapes=result.shapes,
            extra_signals=extra,
        )

        outcome = ImageOutcome(
            job=job,
            status=STATUS_DONE,
            n_objects=result.n_objects,
            n_dropped=result.n_dropped,
            elapsed_s=time.perf_counter() - started,
            message="; ".join(notes),
            summary=sg.count_summary(stats),
            measured_channel=_channel_name(measured_channel),
            measured_extra=tuple(extra) if extra is not None else (),
        )
        if settings.write_tables and stats:
            # One combined file is built from the frames rather than per-image
            # files, so per-image ones are only written when they were asked for.
            combining = settings.write_anndata and settings.anndata_single_file
            outcome.table_path, note, outcome.frame = _write_table(
                stats,
                settings.table_path(job),
                anndata=settings.write_anndata and not combining,
                keep_frame=combining,
            )
            if note:
                outcome.message = "; ".join(part for part in (outcome.message, note) if part)
        return outcome
    except Exception as exc:  # noqa: BLE001 - reported per image, not raised
        logger.exception("batch: %s failed", job.component)
        return ImageOutcome(
            job=job,
            status=STATUS_FAILED,
            message=f"{type(exc).__name__}: {exc}",
            elapsed_s=time.perf_counter() - started,
        )


def iter_batch(
    jobs: Sequence[ImageJob],
    settings: BatchSettings,
    progress: Callable[[str], None] | None = None,
) -> Iterator[ImageOutcome]:
    """Run *jobs* one at a time, yielding each outcome as it lands.

    A generator so that a caller can stop between images — a plate takes hours,
    and the only safe place to interrupt is between one image and the next, with
    the labels of the last one already on disk.
    """
    for index, job in enumerate(jobs, start=1):
        if progress is not None:
            progress(f"[{index}/{len(jobs)}] {job.describe()}")
        yield segment_job(job, settings, progress=progress)


def run_batch(
    jobs: Sequence[ImageJob],
    settings: BatchSettings,
    survey: PlateSurvey | None = None,
    progress: Callable[[str], None] | None = None,
) -> BatchReport:
    """Run the whole batch to completion and write the plate-level summary."""
    started = time.perf_counter()
    report = BatchReport()
    for outcome in iter_batch(jobs, settings, progress=progress):
        report.outcomes.append(outcome)
    report.elapsed_s = time.perf_counter() - started
    if settings.write_tables and survey is not None:
        report.summary_path = write_summary(report, settings, survey)
        if settings.write_anndata and settings.anndata_single_file:
            report.anndata_path = write_combined_anndata(report, settings, survey)
    logger.info("batch finished: %s", report.describe())
    return report


def write_combined_anndata(
    report: BatchReport, settings: BatchSettings, survey: PlateSurvey
) -> Path | None:
    """Write the run's one combined ``.h5ad``, and let go of the frames.

    Built from the frames the run measured rather than by reading the CSVs back:
    a float that has been through a text file is not the float that was measured,
    and the whole point of the combined file is that the spatial analysis starts
    from the numbers rather than from a conversion.
    """
    blocks = [
        (outcome.job.component, outcome.frame)
        for outcome in report.outcomes
        if outcome.ok and outcome.frame is not None and len(outcome.frame)
    ]
    try:
        if not blocks:
            logger.info("nothing to combine; no objects were measured")
            return None
        from .analysis import combine_frames

        combined = combine_frames(blocks, label_column="Label")
        path = settings.anndata_path(survey)
        path.parent.mkdir(parents=True, exist_ok=True)
        combined.write_h5ad(path)
        logger.info(
            "combined AnnData written to %s (%d object(s), %.1f MB)",
            path,
            combined.n_obs,
            path.stat().st_size / 1024 / 1024,
        )
        return path
    except ImportError:
        logger.warning("anndata is not installed; the combined .h5ad was not written")
        return None
    except Exception:  # noqa: BLE001 - the tables are already safely on disk
        logger.exception("could not write the combined AnnData")
        return None
    finally:
        # The tables are on disk; a plate of frames is not worth keeping alive
        # for the rest of the session just because the report is.
        for outcome in report.outcomes:
            outcome.frame = None


def summary_rows(report: BatchReport) -> list[dict[str, Any]]:
    """One row per image: what ran, what came out, and what went wrong."""
    rows = []
    for outcome in report.outcomes:
        job = outcome.job
        totals = outcome.summary
        rows.append(
            {
                "well": job.well,
                "row": job.row,
                "column": job.column,
                "acquisition": "" if job.acquisition is None else job.acquisition,
                "image": job.component,
                "status": outcome.status,
                "n_objects": outcome.n_objects,
                "n_dropped": outcome.n_dropped,
                "measured_channel": outcome.measured_channel,
                "extra_channels": " | ".join(outcome.measured_extra),
                "median_solidity": totals.get("median_solidity", ""),
                "median_volume_um3": totals.get("median_volume_um3", ""),
                "median_diameter_um": totals.get("median_diameter_um", ""),
                "total_volume_um3": totals.get("total_volume_um3", ""),
                "seconds": round(outcome.elapsed_s, 1),
                "message": outcome.message,
            }
        )
    return rows


def write_summary(report: BatchReport, settings: BatchSettings, survey: PlateSurvey) -> Path | None:
    """Write the per-image summary CSV and return its path."""
    rows = summary_rows(report)
    if not rows:
        return None
    try:
        import pandas as pd
    except ImportError:
        logger.warning("pandas is not installed; the batch summary was not written")
        return None
    root = settings.tables_root(survey)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{settings.label_name}_summary.csv"
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8")
    logger.info("batch summary written to %s", path)
    return path
