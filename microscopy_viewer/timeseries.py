"""Fluid playback of time series: local caching, prefetching, and the frame clock.

A time lapse is stored one timepoint at a time and usually lives on a NAS, so
stepping the T slider costs a network round trip per frame. That is what makes
playback stutter: the pixels are not slow to draw, they are slow to arrive.

Two things fix it, and both are here:

* **Local caching.** Every pyramid level of a time series that fits the budget is
  copied once to a local ``.npy`` and memory-mapped, through the same cache the
  3D view uses. Unlike the 3D path this replaces the levels *in place* — the
  layer stays multiscale with the same shapes and the same scale — so nothing
  about how 2D looks changes, only where the bytes come from.
* **Prefetching.** A worker thread walks ahead of the playhead touching one byte
  per page of the timepoints that are about to be shown, so they are resident in
  the operating system's page cache before napari asks for them.

The frame clock is pure arithmetic (:func:`next_index`,
:func:`frames_to_advance`) so playback timing can be tested without a display.
Nothing in this module imports Qt or napari at module level.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Sequence as _AbcSequence
from typing import Callable, Iterable, Sequence

import math
import numpy as np

from . import volume_cache
from .utils import get_logger

logger = get_logger("timeseries")

#: Total bytes of cached time series allowed per layer. A whole time lapse is
#: far bigger than one volume, so this is separate from the 3D voxel budget:
#: levels are cached finest-first until the next one would not fit.
DEFAULT_TIMELINE_BUDGET = 8 * 1024**3  # 8 GB

#: Environment override for that budget, in bytes.
BUDGET_ENV_VAR = "MICROSCOPY_VIEWER_TIMELINE_BUDGET"

#: How many timepoints ahead of the playhead the prefetcher walks.
PREFETCH_AHEAD = 8

#: A timepoint bigger than this is prefetched one Z slab at a time rather than
#: whole, so a thick stack does not have the prefetcher reading gigabytes.
MAX_PREFETCH_FRAME_BYTES = 256 * 1024**2

#: Bytes per memory page. Touching one element per page is what forces a read.
PAGE_BYTES = 4096

# Keys written on ``layer.metadata``.
_CACHED_LEVELS = "mv_timeline_cached"
_PYRAMID = "mv_pyramid"  # shared with rendering.MultiscaleDepthManager


def timeline_budget() -> int:
    """Bytes of local cache one time series may use."""
    raw = os.environ.get(BUDGET_ENV_VAR)
    if raw:
        try:
            value = int(float(raw))
            if value > 0:
                return value
        except ValueError:
            logger.warning("ignoring invalid %s=%r", BUDGET_ENV_VAR, raw)
    return DEFAULT_TIMELINE_BUDGET


# ---------------------------------------------------------------------------
# Finding the time axis
# ---------------------------------------------------------------------------

#: Axis labels that mean "time", as napari or a reader may spell them.
_TIME_LABELS = frozenset({"t", "time", "frame"})


def axes_of(layer) -> str:
    """The axis letters a reader recorded for *layer*, or ``""``."""
    try:
        axes = layer.metadata.get("mv_axes")
    except AttributeError:
        return ""
    return str(axes or "")


def layer_time_axis(layer) -> int | None:
    """Index of the time axis within *layer*'s own dimensions.

    Readers drop singleton T axes, so a layer that carries no ``T`` in its axes
    genuinely has no time dimension and returns ``None`` rather than guessing.
    """
    axes = axes_of(layer).upper()
    if "T" not in axes:
        return None
    index = axes.index("T")
    try:
        if int(np.asarray(shape_of(layer))[index]) < 2:
            return None
    except (IndexError, ValueError, TypeError):
        return None
    return index


def data_levels(data) -> list:
    """A layer's data as a list of pyramid levels, finest first.

    napari does not hand back the list a multiscale layer was built from: it
    wraps it in ``MultiScaleData``, a Sequence that is neither a list nor a tuple
    and that proxies level 0's ``shape`` and ``dtype``. Testing for list-ness
    alone therefore reports a whole pyramid as a single level.
    """
    if isinstance(data, _AbcSequence) and not isinstance(data, (str, bytes)):
        return list(data)
    return [data]


def shape_of(layer) -> tuple[int, ...]:
    """Shape of the layer's full-resolution data, multiscale or not."""
    levels = data_levels(layer.data)
    data = levels[0] if levels else layer.data
    return tuple(int(n) for n in np.asarray(getattr(data, "shape", ()), dtype=int))


def is_image(layer) -> bool:
    """Whether *layer* holds pixels rather than annotations.

    The axes metadata alone is not enough to tell: a ROI Shapes layer is given
    its image's ``mv_axes`` so measurements land on the right slice, and its
    ``data`` is a list of vertex arrays that must never be handed to the cache.
    napari tags every layer class with ``_type_string``; anything without one is
    taken at face value, which is what makes this testable without napari.
    """
    kind = str(getattr(layer, "_type_string", "") or "").lower()
    if kind:
        return kind == "image"
    return type(layer).__name__ != "Shapes" and hasattr(layer, "data")


def has_timeline(layer) -> bool:
    """Whether *layer* is an image with more than one timepoint."""
    return is_image(layer) and layer_time_axis(layer) is not None


def viewer_time_axis(viewer) -> int | None:
    """Index of the time axis in the *viewer's* dimensions.

    napari aligns layers to the right, so a layer's own axis index is not the
    slider index once datasets of different dimensionality are open together.
    The viewer's axis labels are set from the widest dataset when files are
    opened, so they are tried first; otherwise the offset is worked out from a
    layer that has a time axis.
    """
    labels = tuple(getattr(getattr(viewer, "dims", None), "axis_labels", ()) or ())
    for index, label in enumerate(labels):
        if str(label).strip().lower() in _TIME_LABELS:
            return index

    ndim = int(getattr(viewer.dims, "ndim", len(labels)) or 0)
    for layer in getattr(viewer, "layers", ()):
        axis = layer_time_axis(layer)
        if axis is None:
            continue
        offset = ndim - int(getattr(layer, "ndim", axis + 1))
        candidate = offset + axis
        if 0 <= candidate < ndim:
            return candidate
    return None


def timepoint_count(viewer, axis: int | None = None) -> int:
    """Number of timepoints on the viewer's time axis (0 when there is none)."""
    axis = viewer_time_axis(viewer) if axis is None else axis
    if axis is None:
        return 0
    try:
        return int(viewer.dims.nsteps[axis])
    except (AttributeError, IndexError, TypeError, ValueError):
        return 0


def current_index(viewer, axis: int | None = None) -> int:
    axis = viewer_time_axis(viewer) if axis is None else axis
    if axis is None:
        return 0
    try:
        return int(viewer.dims.current_step[axis])
    except (AttributeError, IndexError, TypeError, ValueError):
        return 0


def set_index(viewer, index: int, axis: int | None = None) -> None:
    """Move the viewer to timepoint *index*."""
    axis = viewer_time_axis(viewer) if axis is None else axis
    if axis is None:
        return
    try:
        viewer.dims.set_current_step(axis, int(index))
    except AttributeError:  # pragma: no cover - very old napari
        step = list(viewer.dims.current_step)
        step[axis] = int(index)
        viewer.dims.current_step = tuple(step)


# ---------------------------------------------------------------------------
# The frame clock
# ---------------------------------------------------------------------------


def clamp_range(start: int, stop: int, count: int) -> tuple[int, int]:
    """A playable ``(start, stop)`` inside ``0..count-1``, in order."""
    if count <= 0:
        return (0, 0)
    start = max(0, min(int(start), count - 1))
    stop = max(0, min(int(stop), count - 1))
    if stop < start:
        start, stop = stop, start
    return (start, stop)


def next_index(
    current: int, start: int, stop: int, advance: int = 1, loop: bool = True
) -> int | None:
    """The frame after *current*, or ``None`` when playback should stop.

    *advance* may be more than one frame — that is how a player that has fallen
    behind catches up — and negative to play backwards. Looping wraps within
    ``[start, stop]``; without it, running past either end ends playback.
    """
    span = stop - start + 1
    if span <= 0:
        return None
    if advance == 0:
        return int(current)

    position = int(current) + int(advance)
    if start <= position <= stop:
        return position
    if not loop:
        return None
    return start + (position - start) % span


def frames_to_advance(elapsed_s: float, fps: float, max_skip: int = 4) -> int:
    """How many frames a player should jump given how long the last tick took.

    A timer that fires late — a slow read, a busy machine — would otherwise turn
    a 30 fps request into slow motion. Advancing by the number of frame periods
    that actually elapsed keeps playback at the requested *speed* and drops
    frames instead, which is what every video player does. The jump is capped so
    one long stall does not skip half the series.
    """
    if fps <= 0:
        return 1
    frames = int(round(float(elapsed_s) * float(fps)))
    return max(1, min(frames, max(1, int(max_skip))))


def elapsed_seconds(index: int, interval_s: float | None) -> float | None:
    """Acquisition time at timepoint *index*, if the file recorded an interval."""
    if interval_s is None:
        return None
    try:
        interval = float(interval_s)
    except (TypeError, ValueError):
        return None
    if interval <= 0:
        return None
    return interval * int(index)


def format_timestamp(seconds: float | None) -> str:
    """``h:mm:ss`` for long series, ``m:ss.s`` for short ones, ``""`` for None."""
    if seconds is None:
        return ""
    seconds = float(seconds)
    sign = "-" if seconds < 0 else ""
    seconds = abs(seconds)
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    if hours >= 1:
        return f"{sign}{int(hours)}:{int(minutes):02d}:{int(secs):02d}"
    return f"{sign}{int(minutes)}:{secs:04.1f}"


def time_interval(layer) -> float | None:
    """Seconds between timepoints, from the acquisition metadata."""
    meta = None
    try:
        meta = layer.metadata.get("mv_metadata")
    except AttributeError:
        return None
    return getattr(meta, "time_interval_s", None) if meta is not None else None


# ---------------------------------------------------------------------------
# Level selection and caching
# ---------------------------------------------------------------------------


def pyramid_levels(layer) -> list:
    """The layer's level list — the full pyramid even while 3D has collapsed it.

    :class:`~microscopy_viewer.rendering.MultiscaleDepthManager` parks the real
    list on ``layer.metadata`` while the viewer is in 3D, so that is checked
    first. A layer that is not multiscale reports a single level.
    """
    stored = None
    try:
        stored = layer.metadata.get(_PYRAMID)
    except AttributeError:
        stored = None
    if stored:
        return list(stored)
    return data_levels(layer.data)


def level_bytes(level) -> int:
    shape = tuple(int(n) for n in getattr(level, "shape", ()) or ())
    if not shape:
        return 0
    return math.prod(int(n) for n in shape) * int(np.dtype(level.dtype).itemsize)


def levels_within_budget(levels: Sequence, budget: int) -> list[int]:
    """Indices of the levels worth caching, finest first, that fit *budget*.

    Levels are taken in order and the first one that does not fit is skipped
    along with nothing else: a coarser level is smaller, so it is still tried.
    Caching only the coarser levels of an oversized series is the useful case —
    zoomed-out playback, which is what a time lapse is watched at, comes off
    local disk while a zoomed-in view still reads its small crop from the source.
    """
    chosen: list[int] = []
    used = 0
    for index, level in enumerate(levels):
        size = level_bytes(level)
        if size <= 0 or used + size > budget:
            continue
        chosen.append(index)
        used += size
    return chosen


def is_local(array) -> bool:
    """Whether an array is already backed by memory rather than a slow read."""
    return isinstance(array, np.memmap) or type(array) is np.ndarray


def touch(array) -> None:
    """Read one element per page so *array* becomes resident, cheaply.

    A memory-mapped read only costs a page fault; touching a byte in each page
    forces those faults to happen on this thread instead of on the one drawing
    the next frame. Reading every element instead would move the same bytes but
    burn the CPU copying them for nothing.
    """
    try:
        flat = np.asarray(array).reshape(-1) if array.flags["C_CONTIGUOUS"] else None
    except (AttributeError, ValueError, TypeError):
        flat = None
    if flat is None:
        try:
            np.asarray(array[..., :1])
        except Exception:
            pass
        return
    stride = max(1, PAGE_BYTES // max(1, flat.dtype.itemsize))
    # Summing the sampled elements is what stops the read being optimised away.
    int(np.asarray(flat[::stride]).sum(dtype=np.float64))


class _Prefetcher:
    """Background thread that keeps the frames around the playhead resident.

    Targets are ``(array, time_axis, leading_index)`` triples: the arrays a
    playing viewer is about to read. ``leading_index`` is the position of the
    axes before the displayed plane (the Z slider, typically) so a thick stack
    is warmed one slab at a time.
    """

    def __init__(self, ahead: int = PREFETCH_AHEAD):
        self._ahead = max(1, int(ahead))
        self._targets: list[tuple[object, int, tuple[int, ...]]] = []
        self._index = 0
        self._direction = 1
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stopping = False
        self._thread: threading.Thread | None = None

    # -- control --------------------------------------------------------------

    def set_targets(self, targets: Iterable[tuple[object, int, tuple[int, ...]]]) -> None:
        """Point the thread at new arrays. Does not wake it: only a moving
        playhead is a reason to read anything."""
        with self._lock:
            self._targets = [t for t in targets if is_local(t[0])]

    def update(self, index: int, direction: int = 1) -> None:
        """Tell the thread where the playhead is, and which way it is moving."""
        with self._lock:
            self._index = int(index)
            self._direction = 1 if direction >= 0 else -1
            targets = bool(self._targets)
        if targets:
            self._ensure_thread()
            self._wake.set()

    def stop(self) -> None:
        self._stopping = True
        self._wake.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)
        self._thread = None

    # -- the loop -------------------------------------------------------------

    def _ensure_thread(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stopping = False
        self._thread = threading.Thread(target=self._run, name="mv-prefetch", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stopping:
            # No timeout: the thread does one pass per playhead move and then
            # sleeps. Waking on a timer instead meant it kept re-reading the
            # same timepoints long after playback stopped, evicting the pages
            # the 3D view had just uploaded from.
            self._wake.wait()
            self._wake.clear()
            if self._stopping:
                return
            with self._lock:
                targets = list(self._targets)
                index = self._index
                direction = self._direction
            for offset in range(1, self._ahead + 1):
                if self._stopping or self._wake.is_set():
                    break  # the playhead moved; restart from where it is now
                for array, axis, leading in targets:
                    self._warm(array, axis, leading, index + offset * direction)

    def _warm(self, array, axis: int, leading: tuple[int, ...], index: int) -> None:
        shape = tuple(int(n) for n in getattr(array, "shape", ()) or ())
        if axis >= len(shape):
            return
        count = shape[axis]
        if count <= 0:
            return
        position = index % count  # wrap, because looping playback does
        selector: list[object] = [slice(None)] * len(shape)
        selector[axis] = position
        try:
            frame = array[tuple(selector)]
            if level_bytes(frame) > MAX_PREFETCH_FRAME_BYTES and leading:
                frame = frame[leading[0]]
            touch(frame)
        except Exception:  # pragma: no cover - a failed prefetch is not an error
            logger.debug("prefetch of timepoint %d failed", position, exc_info=True)


class TimelineManager:
    """Keeps every open time series playable without touching the source again.

    Attach one to a viewer. It notices time series as they are added, copies
    their pyramid levels to the local cache on a worker thread, points the layers
    at the copies, and runs the prefetcher ahead of the playhead while the panel
    is playing.

    Every step degrades to "read from the original", so a full disk, a disabled
    cache or an unreadable entry costs speed and nothing else.
    """

    def __init__(
        self,
        viewer,
        budget: int | None = None,
        on_status: Callable[[str], None] | None = None,
        cache: bool = True,
        threaded: bool = True,
    ):
        self._viewer = viewer
        self._budget = timeline_budget() if budget is None else int(budget)
        self._on_status = on_status
        self._cache = cache
        # Off in tests, where there is no event loop to deliver a worker's
        # signals: the copy then happens inline instead.
        self._threaded = threaded
        self._workers: dict[int, object] = {}
        self._pending: set[int] = set()
        self._prefetcher = _Prefetcher()
        self._connect()

    # -- wiring ---------------------------------------------------------------

    def _connect(self) -> None:
        try:
            self._viewer.layers.events.inserted.connect(self._on_event)
            self._viewer.layers.events.removed.connect(self._on_event)
            # Leaving 3D restores the full pyramid from metadata; this picks the
            # cached levels back up afterwards, and entering 3D drops the
            # prefetch targets so nothing reads ahead behind the volume upload.
            self._viewer.dims.events.ndisplay.connect(self._on_event)
        except Exception:  # pragma: no cover - napari event API drift
            logger.warning("could not connect the timeline handler", exc_info=True)

    def _on_event(self, event=None) -> None:
        self.apply()

    def _in_3d(self) -> bool:
        try:
            return int(getattr(self._viewer.dims, "ndisplay", 2)) >= 3
        except (TypeError, ValueError):
            return False

    def _status(self, message: str) -> None:
        if self._on_status is not None:
            self._on_status(message)

    # -- discovery ------------------------------------------------------------

    def timelines(self) -> list:
        """Open image layers that have more than one timepoint."""
        return [layer for layer in list(self._viewer.layers) if has_timeline(layer)]

    def cached_bytes(self, layer) -> int:
        levels = pyramid_levels(layer)
        return sum(level_bytes(level) for level in levels if isinstance(level, np.memmap))

    def is_cached(self, layer) -> bool:
        """Whether the level a player would read is local."""
        levels = pyramid_levels(layer)
        return bool(levels) and any(isinstance(level, np.memmap) for level in levels)

    def describe(self, layer) -> str:
        """One line about where *layer*'s frames are coming from."""
        levels = pyramid_levels(layer)
        local = [index for index, level in enumerate(levels) if is_local(level)]
        if not levels:
            return "no data"
        if len(local) == len(levels):
            return "all levels local"
        if local:
            return f"levels {', '.join(str(i) for i in local)} of {len(levels) - 1} local"
        if id(layer) in self._pending:
            return "caching…"
        return "reading from the source"

    # -- caching --------------------------------------------------------------

    def apply(self) -> None:
        """Adopt anything already cached and re-aim the prefetcher.

        Deliberately does **not** start a copy. Opening a file is exactly when
        the viewer is busiest — the first slice, the first 3D upload — and a
        background thread pulling the whole time lapse across the network at the
        same moment slows down the thing the user is actually looking at. The
        copy is started by :meth:`start_caching` instead, when playback begins
        or the panel's button is pressed.
        """
        for layer in self.timelines():
            try:
                self.adopt_existing(layer)
            except Exception:  # pragma: no cover - never take the viewer down
                logger.exception("could not check the cache for %s", getattr(layer, "name", "?"))
        self.refresh_prefetch()

    def start_caching(self) -> int:
        """Copy every open time series locally. Returns the workers started."""
        started = 0
        for layer in self.timelines():
            try:
                if self.cache_layer(layer) is not None:
                    started += 1
            except Exception:  # pragma: no cover - never take the viewer down
                logger.exception("could not cache the timeline of %s", getattr(layer, "name", "?"))
        self.refresh_prefetch()
        return started

    def adopt_existing(self, layer) -> int:
        """Point *layer* at cache entries that already exist, copying nothing.

        Free: a dataset that has been played or viewed in 3D before is already
        on local disk under the same key, so this is the whole benefit of the
        cache with none of its cost.
        """
        if not self._cache or not volume_cache.enabled():
            return 0
        adopted = 0
        for index, level, key in self._jobs(layer):
            existing = volume_cache.load(key, level.shape, level.dtype)
            if existing is not None:
                self._adopt(layer, index, existing)
                adopted += 1
        if adopted:
            logger.info("%s: reusing %d cached level(s)", getattr(layer, "name", "?"), adopted)
        return adopted

    def _jobs(self, layer) -> list[tuple[int, object, str]]:
        """``(level index, array, cache key)`` for the levels worth caching."""
        levels = pyramid_levels(layer)
        wanted = levels_within_budget(levels, self._budget)
        meta = layer.metadata.get("mv_metadata")
        source = getattr(meta, "file_path", None) if meta is not None else None
        name = getattr(layer, "name", "layer")
        return [
            (
                index,
                levels[index],
                volume_cache.layer_key(
                    source, name, index, levels[index].shape, levels[index].dtype
                ),
            )
            for index in wanted
            if not is_local(levels[index])
        ]

    def cache_layer(self, layer, force: bool = False) -> object | None:
        """Copy *layer*'s levels to local disk, off the main thread.

        Returns the worker when one was started, ``None`` when there was nothing
        to do — already cached, already running, or caching switched off.
        """
        if not force and (not self._cache or not volume_cache.enabled()):
            return None
        if id(layer) in self._pending:
            return None
        if self._in_3d() and not force:
            # The 3D view is already reading this data — and caching a level for
            # it, under the same key. Two threads pulling the same file over the
            # network at once is slower than either alone.
            logger.debug("skipping timeline caching while the viewer is in 3D")
            return None

        jobs = self._jobs(layer)
        if not jobs:
            return None
        name = getattr(layer, "name", "layer")

        # Anything already on disk can be adopted straight away, without a worker.
        remaining = []
        adopted = 0
        for index, level, key in jobs:
            existing = volume_cache.load(key, level.shape, level.dtype)
            if existing is not None:
                self._adopt(layer, index, existing)
                adopted += 1
            else:
                remaining.append((index, level, key))
        if adopted:
            logger.info("%s: reusing %d cached level(s)", name, adopted)
        if not remaining:
            self.refresh_prefetch()
            return None

        total = sum(level_bytes(level) for _index, level, _key in remaining)
        self._status(f"{name}: caching {total / 1e6:.0f} MB of timepoints locally…")

        def _work():
            done = []
            for index, level, key in remaining:
                cached = volume_cache.store(level, key)
                if cached is not None:
                    done.append((index, cached))
            return done

        def _finish(done):
            self._pending.discard(id(layer))
            self._workers.pop(id(layer), None)
            for index, cached in done or ():
                self._adopt(layer, index, cached)
            if done:
                self._status(f"{name}: {len(done)} level(s) cached — playback is local now.")
            self.refresh_prefetch()

        def _failed(exc):
            self._pending.discard(id(layer))
            self._workers.pop(id(layer), None)
            logger.warning("timeline caching failed for %s: %s", name, exc)

        self._pending.add(id(layer))
        try:
            if not self._threaded:
                raise RuntimeError("threading disabled")
            from napari.qt.threading import thread_worker
        except Exception:
            _finish(_work())
            return None

        worker = thread_worker(_work)()
        worker.returned.connect(_finish)
        worker.errored.connect(_failed)
        self._workers[id(layer)] = worker
        worker.start()
        return worker

    def cache_all(self, force: bool = False) -> int:
        """Cache every open time series. Returns how many workers were started."""
        started = 0
        for layer in self.timelines():
            if self.cache_layer(layer, force=force) is not None:
                started += 1
        return started

    def _adopt(self, layer, index: int, cached) -> None:
        """Point one pyramid level at its local copy, leaving the rest alone."""
        if layer not in self._viewer.layers:
            return
        levels = pyramid_levels(layer)
        if index >= len(levels):
            return
        if tuple(levels[index].shape) != tuple(cached.shape):
            logger.warning("cached level %d of %s no longer matches; ignoring it", index, layer.name)
            return
        levels[index] = cached
        try:
            self._replace_levels(layer, levels)
        except Exception:
            logger.warning("could not switch %s onto its cached level", layer.name, exc_info=True)
            return
        layer.metadata[_CACHED_LEVELS] = [
            i for i, level in enumerate(levels) if isinstance(level, np.memmap)
        ]
        logger.info("%s: level %d now reads from the local cache", layer.name, index)

    @staticmethod
    def _replace_levels(layer, levels: list) -> None:
        """Swap the arrays a layer reads without changing its shape or scale.

        Every level keeps its shape, so unlike the 3D path there is no rescaling
        to do and multiscale rendering carries on exactly as before. The full
        list is written back to ``mv_pyramid`` as well, or the 3D manager would
        restore the original, uncached arrays the next time the viewer leaves 3D.
        """
        if layer.metadata.get(_PYRAMID) is not None:
            layer.metadata[_PYRAMID] = list(levels)
        if not getattr(layer, "multiscale", False):
            layer.data = levels[0]
            return
        # In 3D the layer is holding a single collapsed level; leave that alone
        # and let the manager pick the cached array up on its next swap.
        if len(layer.data) == len(levels):
            layer.data = list(levels)

    # -- prefetch -------------------------------------------------------------

    def refresh_prefetch(self) -> None:
        """Point the prefetcher at the arrays the viewer is currently reading.

        Nothing is prefetched in 3D: napari uploads a whole volume per timepoint
        there, so reading ahead would evict the pages the current volume was
        built from instead of saving a read.
        """
        targets: list[tuple[object, int, tuple[int, ...]]] = []
        for layer in ([] if self._in_3d() else self.timelines()):
            axis = layer_time_axis(layer)
            if axis is None:
                continue
            array = self._displayed_array(layer)
            if array is None or not is_local(array):
                continue
            targets.append((array, axis, self._leading_index(layer, axis)))
        self._prefetcher.set_targets(targets)

    def set_playhead(self, index: int, direction: int = 1) -> None:
        """Called by the player on every frame, so the thread stays ahead of it."""
        self._prefetcher.update(index, direction)

    def stop(self) -> None:
        self._prefetcher.stop()

    def _displayed_array(self, layer):
        levels = data_levels(layer.data)
        if not levels:
            return None
        index = int(getattr(layer, "_data_level", 0) or 0)
        return levels[min(index, len(levels) - 1)]

    def _leading_index(self, layer, time_axis: int) -> tuple[int, ...]:
        """The layer's non-displayed slider positions after the time axis."""
        try:
            step = tuple(int(n) for n in self._viewer.dims.current_step)
        except (AttributeError, TypeError, ValueError):
            return ()
        offset = len(step) - int(getattr(layer, "ndim", len(step)))
        after = step[offset + time_axis + 1 :]
        return tuple(after[:-2])  # drop the two displayed axes
