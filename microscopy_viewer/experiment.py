"""Treat a folder as an experiment: list it, preview it, process all of it.

An acquisition session leaves a folder of ``.ims`` files, and almost everything
done to them afterwards is done to all of them: the same region outlined on every
fish, the same channel segmented with the same settings. Doing that one file at a
time through the viewer means opening thirty datasets, and the layer list stops
being usable somewhere around the fourth.

So this module never puts anything in the viewer. It reads each file far enough
to describe it and to draw a thumbnail, and it runs the batch by opening one file
at a time, taking what it needs, and closing it again. What comes out goes back
into the file it came from, through :mod:`microscopy_viewer.ims_store`.

Qt-free and napari-free, like :mod:`microscopy_viewer.slides` and for the same
reason: the panel is a view onto this, and this is what the checks exercise.

The one thing worth knowing before running a batch: **a file being written to
cannot be open in the viewer**. HDF5 refuses to open for writing what is open for
reading, so :func:`run_batch` releases each file's reader handle before writing —
which invalidates any layer still showing it. The panel closes those layers
first.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from . import ims_store
from .utils import get_logger

logger = get_logger("experiment")

#: Longest edge of a thumbnail, in pixels. Big enough to tell a good mount from a
#: bad one and to recognise which fish is which; small enough that a folder of
#: thirty draws in a couple of seconds off the coarsest pyramid level.
THUMBNAIL_PIXELS = 220

#: Percentiles the thumbnail stretches between. The same pair the contrast tools
#: use, so a preview looks like the image will when it is opened.
LOW_PERCENTILE = 0.5
HIGH_PERCENTILE = 99.5

#: Fallback channel colours, in acquisition order, when a file records none.
FALLBACK_COLORS = (
    (0.0, 1.0, 0.0),
    (1.0, 0.0, 1.0),
    (0.0, 0.6, 1.0),
    (1.0, 0.6, 0.0),
)


class BatchCancelled(Exception):
    """Raised inside :func:`run_batch` when the caller asks it to stop."""


# ---------------------------------------------------------------------------
# Listing a folder
# ---------------------------------------------------------------------------


@dataclass
class SampleEntry:
    """One dataset in the experiment folder, described without opening it fully."""

    path: Path
    name: str
    n_channels: int = 0
    shape: tuple[int, ...] = ()
    voxel_um: tuple[float, ...] = ()
    channel_names: list[str] = field(default_factory=list)
    #: Number of ROIs already stored in the file, and the label maps it carries.
    n_rois: int = 0
    label_keys: list[str] = field(default_factory=list)
    #: Why this file could not be read, if it could not be.
    error: str = ""

    @property
    def readable(self) -> bool:
        return not self.error and bool(self.shape)

    @property
    def is_stack(self) -> bool:
        return len(self.shape) >= 3 and self.shape[0] > 1

    def describe(self) -> str:
        if self.error:
            return self.error
        parts = [" × ".join(str(int(n)) for n in self.shape)]
        if self.n_channels:
            parts.append(f"{self.n_channels} ch")
        stored = ims_store.summary(self.path)
        if stored:
            parts.append(stored)
        return ", ".join(parts)


def list_files(folder: str | Path, recursive: bool = True) -> list[Path]:
    """Readable datasets in *folder*, sorted the way a file browser sorts them.

    Aborted Imaris acquisitions leave ``*_F0.ims`` stubs with no image in them;
    they are left in the list rather than filtered, because a stub among the
    samples is something to notice, and :func:`describe_file` will say what is
    wrong with it.
    """
    from .loaders import is_supported

    root = Path(folder)
    if not root.is_dir():
        return []
    pattern = "**/*" if recursive else "*"
    found = [
        path
        for path in sorted(root.glob(pattern), key=lambda p: (str(p.parent).lower(), p.name.lower()))
        if path.is_file() and is_supported(path)
    ]
    return found


def describe_file(path: str | Path) -> SampleEntry:
    """Read one file's shape, channels and stored extras. Never raises."""
    candidate = Path(path)
    entry = SampleEntry(path=candidate, name=candidate.stem)
    try:
        specs = _read_specs(candidate)
    except Exception as exc:
        entry.error = str(exc)
        logger.info("could not describe %s: %s", candidate.name, exc)
        return entry

    entry.n_channels = len(specs)
    if specs:
        first = specs[0]
        array = first.data[0] if first.multiscale else first.data
        entry.shape = tuple(int(n) for n in np.shape(array))
        entry.voxel_um = tuple(float(v) for v in first.scale)
        entry.channel_names = [spec.channel_name or spec.name for spec in specs]
    entry.n_rois = len(ims_store.load_rois(candidate))
    entry.label_keys = ims_store.list_labels(candidate)
    _release(candidate)
    return entry


def scan_folder(
    folder: str | Path,
    recursive: bool = True,
    progress: Callable[[str], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> list[SampleEntry]:
    """Describe every dataset in *folder*.

    One file at a time, closing each before the next: a folder of thirty stacks
    is a couple of hundred gigabytes of pyramid, and holding thirty HDF5 handles
    open to list them is how a scan turns into a memory problem.
    """
    paths = list_files(folder, recursive)
    entries: list[SampleEntry] = []
    for index, path in enumerate(paths, start=1):
        if should_cancel is not None and should_cancel():
            break
        if progress is not None:
            progress(f"reading {path.name} ({index} of {len(paths)})")
        entries.append(describe_file(path))
    return entries


def _read_specs(path: Path):
    from .loaders import load_path

    return load_path(path)


def _release(path: Path) -> None:
    """Give back the reader's handle on *path*, if it is holding one."""
    from .loaders import ims as ims_reader

    try:
        ims_reader.release(path)
    except Exception:
        logger.debug("could not release the handle on %s", path, exc_info=True)


# ---------------------------------------------------------------------------
# Thumbnails
# ---------------------------------------------------------------------------


def thumbnail(path: str | Path, max_pixels: int = THUMBNAIL_PIXELS) -> np.ndarray | None:
    """A small RGB preview of one dataset: channels merged, flattened over Z.

    Read off the coarsest pyramid level in the file, which for an Imaris stack is
    a few hundred pixels across — so this costs a fraction of a second per file
    rather than the seconds a full-resolution read would take, and a folder can
    be previewed while the user watches.
    """
    try:
        specs = _read_specs(Path(path))
    except Exception:
        logger.debug("no thumbnail for %s", path, exc_info=True)
        return None
    try:
        planes = []
        for index, spec in enumerate(specs):
            array = _preview_level(spec, max_pixels)
            plane = np.asarray(array)
            if plane.ndim > 2:
                plane = plane.reshape(-1, *plane.shape[-2:]).max(axis=0)
            planes.append((plane, _channel_color(spec, index, len(specs))))
        if not planes:
            return None
        return merge_channels(planes, max_pixels)
    finally:
        _release(Path(path))


def _preview_level(spec, max_pixels: int):
    """The coarsest pyramid level still at least *max_pixels* across.

    Not simply the coarsest one: Imaris pyramids bottom out around 64 px, and a
    64 px preview blown up to thumbnail size is a blur that says nothing about
    whether the mount was any good. Not the finest either — that is a 2040²
    read per channel per file, which is the difference between previewing a
    folder in two seconds and in a minute.
    """
    if not getattr(spec, "multiscale", False):
        return spec.data
    levels = list(spec.data)
    chosen = levels[0]
    for level in levels:
        if min(np.shape(level)[-2:]) < max_pixels:
            break
        chosen = level
    return chosen


def _channel_color(spec, index: int, total: int) -> tuple[float, float, float]:
    """The channel's display colour: the file's own, or a readable fallback.

    A single-channel image is drawn grey rather than green — a brightfield or a
    lone DAPI stack tinted a colour it was not acquired in reads as a mistake.
    """
    color = getattr(spec, "color", None)
    if color is not None and len(tuple(color)) >= 3 and any(float(c) > 0 for c in color):
        return tuple(float(c) for c in tuple(color)[:3])  # type: ignore[return-value]
    if total <= 1:
        return (1.0, 1.0, 1.0)
    return FALLBACK_COLORS[index % len(FALLBACK_COLORS)]


def merge_channels(
    planes: Sequence[tuple[np.ndarray, tuple[float, float, float]]],
    max_pixels: int = THUMBNAIL_PIXELS,
) -> np.ndarray:
    """Stretch, tint and add each plane into one RGB image, scaled down."""
    merged: np.ndarray | None = None
    for plane, color in planes:
        small = _downsample(np.asarray(plane), max_pixels)
        stretched = _stretch(small)
        tinted = stretched[..., None] * np.asarray(color, dtype=np.float32)
        merged = tinted if merged is None else merged + tinted
    if merged is None:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    return np.clip(merged * 255.0, 0, 255).astype(np.uint8)


def _stretch(plane: np.ndarray) -> np.ndarray:
    """Percentile contrast stretch to 0-1. Flat images come back black."""
    values = np.asarray(plane, dtype=np.float32)
    if values.size == 0:
        return values
    low, high = np.percentile(values, (LOW_PERCENTILE, HIGH_PERCENTILE))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low, high = float(values.min()), float(values.max())
    if high <= low:
        return np.zeros_like(values)
    return np.clip((values - low) / (high - low), 0.0, 1.0)


def _downsample(plane: np.ndarray, max_pixels: int) -> np.ndarray:
    """Shrink so the longest edge is at most *max_pixels*, by plain striding.

    Striding rather than averaging: this is already the coarsest pyramid level,
    which was produced by averaging, and a thumbnail does not repay a second
    filter pass across a folder of thirty files.
    """
    array = np.asarray(plane)
    if array.ndim != 2 or max_pixels <= 0:
        return array
    longest = max(array.shape)
    if longest <= max_pixels:
        return array
    # Floor, not ceil: a step that overshoots lands *under* the requested size
    # and the preview is then scaled back up, blurred. Coming out slightly larger
    # than asked costs nothing — the widget scales down cleanly.
    step = max(1, int(longest // max_pixels))
    return array[::step, ::step]


# ---------------------------------------------------------------------------
# ROIs across the whole folder
# ---------------------------------------------------------------------------


@dataclass
class RoiOutcome:
    """What happened to one file when ROIs were written to it."""

    path: Path
    name: str
    n_rois: int = 0
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error


def apply_rois(
    paths: Sequence[str | Path],
    rois: Sequence[ims_store.StoredRoi],
    progress: Callable[[str], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> list[RoiOutcome]:
    """Write the same ROI set into every file. One outcome per file.

    This is the "draw once, apply to the folder" case, which is right when the
    samples are mounted the same way and wrong when they are not — the ROIs are
    in micrometres from each image's own origin, so a fish sitting 200 µm further
    along in its field gets an outline 200 µm out of place. Drawing per sample is
    the other workflow: the panel calls this with one path at a time.
    """
    outcomes: list[RoiOutcome] = []
    for index, entry in enumerate(paths, start=1):
        if should_cancel is not None and should_cancel():
            break
        path = Path(entry)
        if progress is not None:
            progress(f"writing ROIs into {path.name} ({index} of {len(paths)})")
        ok, reason = ims_store.can_write(path)
        if not ok:
            outcomes.append(RoiOutcome(path=path, name=path.stem, error=reason))
            continue
        try:
            written = ims_store.save_rois(path, rois)
            outcomes.append(RoiOutcome(path=path, name=path.stem, n_rois=written))
        except Exception as exc:
            logger.exception("could not write ROIs into %s", path.name)
            outcomes.append(RoiOutcome(path=path, name=path.stem, error=str(exc)))
    return outcomes


def roi_mask(
    rois: Sequence[ims_store.StoredRoi],
    shape_yx: tuple[int, int],
    voxel_yx: Sequence[float],
) -> np.ndarray | None:
    """Union of the stored outlines, rasterised onto one image's pixel grid.

    ``None`` when there is nothing to rasterise, which the caller reads as "use
    the whole image" — a file with no ROIs is not a file with an empty ROI.

    The vertices are in micrometres from the image's own origin, so converting
    them needs only the voxel size. That is what makes an outline drawn on the
    20x stack land correctly on the same stack read at a different pyramid level.
    """
    from .intensity import polygon_mask, world_to_data

    outlines = [roi for roi in rois if roi.vertices_um.shape[0] >= 3]
    if not outlines:
        return None
    mask = np.zeros(tuple(int(n) for n in shape_yx), dtype=bool)
    for roi in outlines:
        pixels = world_to_data(roi.vertices_um, voxel_yx, (0.0, 0.0))
        mask |= polygon_mask(pixels, mask.shape)
    return mask if mask.any() else None


def apply_roi_mask(image: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    """Blank everything outside *mask*, broadcasting a 2D mask over a stack.

    Zeroed rather than cropped, so the labels that come back are on the grid that
    went in and drop straight onto the source layer. The cost is that Cellpose's
    percentile normalisation now sees all those zeros; in practice the outlines
    cover the bright part of the field and the low percentile was near zero
    anyway, but it is a real difference from segmenting the whole image.
    """
    if mask is None:
        return image
    array = np.asarray(image)
    if array.ndim == 2:
        return np.where(mask, array, 0)
    return np.where(mask[None, ...], array, 0)


# ---------------------------------------------------------------------------
# Segmenting the whole folder
# ---------------------------------------------------------------------------


@dataclass
class BatchOptions:
    """How a batch segmentation picks its channel and what it does with results."""

    #: Which channel to segment. An integer is an index into the file's channels;
    #: a string is matched against the channel names, case-insensitively, so
    #: "dapi" finds the nuclear channel in every file even when the channel order
    #: differs between acquisitions — which it does.
    channel: int | str = 0
    #: Optional second channel to read per-object intensities from, same rules.
    measure_channel: int | str | None = None
    settings: Any = None  # SegmentationSettings; typed loosely to stay import-free
    #: Write the label map back into the ``.ims`` it came from.
    save_to_file: bool = True
    #: Dataset name inside ``/ARGUS/Labels``. Empty derives it from the channel.
    label_key: str = ""
    #: Blank everything outside the ROIs stored in the file before segmenting.
    #: This is the point of drawing them: Cellpose has no idea that the skin and
    #: the yolk are not brain, and it finds plenty of objects in both. A file
    #: with no stored ROIs is segmented whole.
    restrict_to_rois: bool = False


@dataclass
class BatchOutcome:
    """One file's result."""

    path: Path
    name: str
    n_objects: int = 0
    elapsed_s: float = 0.0
    channel_name: str = ""
    label_key: str = ""
    saved: bool = False
    #: Dimensions of the label map, which decides whether sizes are areas or
    #: volumes. Carried on the outcome rather than guessed from the rows later.
    ndim: int = 3
    error: str = ""
    stats: list = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.error


def pick_channel(names: Sequence[str], wanted: int | str | None) -> int | None:
    """Index of the channel *wanted* names, or ``None``.

    An integer is taken as an index and clamped away from nonsense; a string is
    matched as a substring of the channel name. Matching by name is what makes
    one batch run over files whose channel order differs — and it silently
    matching the wrong channel is worse than not matching, so a string that finds
    nothing returns ``None`` rather than falling back to channel 0.
    """
    if wanted is None:
        return None
    if isinstance(wanted, (int, np.integer)) and not isinstance(wanted, bool):
        index = int(wanted)
        return index if 0 <= index < len(names) else None
    needle = str(wanted).strip().lower()
    if not needle:
        return None
    for index, name in enumerate(names):
        if needle in str(name).lower():
            return index
    return None


def run_batch(
    paths: Sequence[str | Path],
    options: BatchOptions,
    progress: Callable[[str], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> list[BatchOutcome]:
    """Segment one channel of every file, and put the labels back in the file.

    A file that cannot be read, or whose channel cannot be found, is reported and
    skipped: thirty files is long enough that aborting the run on the twenty-ninth
    because one is a stub would be its own kind of failure.
    """
    from . import segmentation as sg

    settings = options.settings or sg.SegmentationSettings()
    outcomes: list[BatchOutcome] = []

    for index, entry in enumerate(paths, start=1):
        path = Path(entry)
        prefix = f"{path.stem} ({index} of {len(paths)})"
        if should_cancel is not None and should_cancel():
            break
        outcome = BatchOutcome(path=path, name=path.stem)
        started = time.perf_counter()
        try:
            if progress is not None:
                progress(f"{prefix}: reading")
            image, measure, voxel, channel_name = _load_channel(path, options)
            outcome.channel_name = channel_name

            def _relay(text: str, _prefix=prefix) -> None:
                if progress is not None:
                    progress(f"{_prefix}: {text}")

            result = sg.segment_volume(
                image, voxel_size_um=voxel, settings=settings, progress=_relay
            )
            outcome.n_objects = int(result.n_objects)
            outcome.ndim = int(result.masks.ndim)
            outcome.warnings = list(result.warnings)

            signal = measure if measure is not None else image
            if result.projected and np.asarray(signal).ndim == 3:
                signal = sg.max_projection(np.asarray(signal))
            outcome.stats = sg.object_table(
                result.masks, signal, result.voxel_size_um[-result.masks.ndim:]
            )

            if options.save_to_file:
                if progress is not None:
                    progress(f"{prefix}: writing into {path.name}")
                key = options.label_key or f"{channel_name or 'channel'} labels"
                outcome.label_key = ims_store.save_labels(
                    path,
                    key,
                    result.masks,
                    result.voxel_size_um[-result.masks.ndim:],
                    attrs={
                        "model": result.model,
                        "mode": result.mode,
                        "device": result.device,
                        "channel": channel_name,
                        "diameter_um": float(settings.diameter_um),
                        "min_diameter_um": float(settings.min_diameter_um),
                        "max_diameter_um": float(settings.max_diameter_um),
                    },
                )
                outcome.saved = True
        except BatchCancelled:
            break
        except Exception as exc:
            logger.exception("batch segmentation failed on %s", path.name)
            outcome.error = str(exc)
        finally:
            outcome.elapsed_s = time.perf_counter() - started
            _release(path)
        outcomes.append(outcome)
        if progress is not None:
            progress(
                f"{prefix}: {outcome.error or f'{outcome.n_objects} object(s)'}"
            )
    return outcomes


def _load_channel(path: Path, options: BatchOptions):
    """The array to segment, the array to measure, the voxel size, the name.

    Materialised into memory before the file is closed. The pyramid is lazy and
    the handle goes away the moment the labels are written back, so anything
    still needed afterwards has to be real by then.
    """
    specs = _read_specs(path)
    if not specs:
        raise ValueError(f"{path.name} has no readable channels")
    names = [spec.channel_name or spec.name for spec in specs]

    index = pick_channel(names, options.channel)
    if index is None:
        raise ValueError(
            f"no channel matching {options.channel!r} in {path.name} "
            f"(it has: {', '.join(names) or 'none'})"
        )
    spec = specs[index]
    image = np.asarray(spec.data[0] if spec.multiscale else spec.data)

    measure = None
    if options.measure_channel is not None:
        other = pick_channel(names, options.measure_channel)
        if other is not None and other != index:
            source = specs[other]
            measure = np.asarray(source.data[0] if source.multiscale else source.data)

    voxel = tuple(float(v) for v in spec.scale)[-image.ndim:]

    if options.restrict_to_rois:
        mask = roi_mask(ims_store.load_rois(path), image.shape[-2:], voxel[-2:])
        if mask is not None:
            image = apply_roi_mask(image, mask)
            if measure is not None:
                measure = apply_roi_mask(measure, mask)
        else:
            logger.info("%s has no stored ROIs; segmenting the whole image", path.name)
    return image, measure, voxel, names[index]


def batch_dataframe(outcomes: Sequence[BatchOutcome]):
    """One row per file: what was segmented, how many, how long, what failed."""
    import pandas as pd

    rows = [
        {
            "Sample": outcome.name,
            "Channel": outcome.channel_name,
            "Objects": outcome.n_objects,
            "Seconds": round(outcome.elapsed_s, 1),
            "Saved as": outcome.label_key if outcome.saved else "",
            "Problem": outcome.error,
            "File": str(outcome.path),
        }
        for outcome in outcomes
    ]
    return pd.DataFrame(
        rows, columns=["Sample", "Channel", "Objects", "Seconds", "Saved as", "Problem", "File"]
    )


def objects_dataframe(outcomes: Sequence[BatchOutcome]):
    """Every object from every file, with the sample it came from."""
    import pandas as pd

    from .segmentation import OBJECT_COLUMNS, object_headers

    rows = []
    ndim = next((outcome.ndim for outcome in outcomes if outcome.stats), 3)
    for outcome in outcomes:
        for stat in outcome.stats:
            row = stat.as_row()
            row["sample"] = outcome.name
            row["channel"] = outcome.channel_name
            rows.append(row)
    frame = pd.DataFrame(rows, columns=["sample", "channel", *OBJECT_COLUMNS])
    headers = {"sample": "Sample", "channel": "Channel", **object_headers(ndim)}
    return frame.rename(columns=headers)
