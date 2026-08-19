"""Time-series playback checks: the frame clock, level choice and local caching.

No display and no Qt event loop needed.

Run with::

    python tests/test_timeseries.py
"""

from __future__ import annotations

import gc
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from microscopy_viewer import timeseries, volume_cache  # noqa: E402

_failures: list[str] = []


def check(condition: bool, message: str) -> None:
    if condition:
        print(f"  ok   {message}")
    else:
        print(f"  FAIL {message}")
        _failures.append(message)


# ---------------------------------------------------------------------------
# Stand-ins for the parts of napari the module touches
# ---------------------------------------------------------------------------


class _Event:
    """The one method the manager calls on a napari event."""

    def connect(self, _handler) -> None:
        return None


class _Events:
    def __init__(self, *names):
        for name in names:
            setattr(self, name, _Event())


class _Layers(list):
    def __init__(self, *args):
        super().__init__(*args)
        self.events = _Events("inserted", "removed")


class _Dims:
    def __init__(self, ndim=4, nsteps=(5, 3, 8, 8), axis_labels=("T", "Z", "Y", "X")):
        self.ndim = ndim
        self.nsteps = nsteps
        self.axis_labels = axis_labels
        self.ndisplay = 2
        self.current_step = tuple(0 for _ in range(ndim))
        self.events = _Events("ndisplay", "current_step")

    def set_current_step(self, axis, value):
        step = list(self.current_step)
        step[axis] = int(value)
        self.current_step = tuple(step)


class _Viewer:
    def __init__(self, layers=(), dims=None):
        self.layers = _Layers(layers)
        self.dims = dims or _Dims()


class _Layer:
    """A duck-typed image layer: data, metadata, scale and a level index."""

    def __init__(self, data, axes="TZYX", name="image", multiscale=False, metadata=None):
        self.data = data
        self.name = name
        self.multiscale = multiscale
        self._data_level = 0
        first = data[0] if multiscale else data
        self.ndim = len(first.shape)
        self.metadata = {"mv_axes": axes, "mv_metadata": metadata}


class _Meta:
    def __init__(self, file_path=None, time_interval_s=None):
        self.file_path = file_path
        self.time_interval_s = time_interval_s


def _stack(shape=(6, 2, 32, 32), dtype=np.uint16) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 4000, size=shape, dtype=np.uint64).astype(dtype)


def _lazy(array: np.ndarray):
    """The array as a reader would hand it over: lazy, one chunk per timepoint."""
    import dask.array as da

    return da.from_array(array, chunks=(1,) + array.shape[1:])


# ---------------------------------------------------------------------------


def test_time_axis_detection() -> None:
    print("the time axis is found on the layer and on the viewer")
    layer = _Layer(_stack(), axes="TZYX")
    check(timeseries.layer_time_axis(layer) == 0, "TZYX puts time on axis 0")
    check(timeseries.has_timeline(layer) is True, "six timepoints count as a time series")

    single = _Layer(_stack((1, 2, 32, 32)), axes="TZYX")
    check(
        timeseries.layer_time_axis(single) is None,
        "a single timepoint is not a time series, so no player is offered",
    )
    still = _Layer(_stack((4, 32, 32)), axes="ZYX")
    check(timeseries.layer_time_axis(still) is None, "a z-stack has no time axis")

    viewer = _Viewer([layer])
    check(timeseries.viewer_time_axis(viewer) == 0, "the viewer's T label gives the slider index")
    check(timeseries.timepoint_count(viewer) == 5, "the timepoint count comes from dims.nsteps")

    # A 3D layer open beside a 4D one: napari right-aligns, so the layer's own
    # axis index is not the slider index.
    unlabelled = _Viewer([_Layer(_stack((6, 2, 32, 32)), axes="TZYX")], _Dims(axis_labels=(0, 1, 2, 3)))
    check(
        timeseries.viewer_time_axis(unlabelled) == 0,
        "without axis labels the offset is worked out from the layer",
    )
    check(
        timeseries.viewer_time_axis(_Viewer([], _Dims(2, (8, 8), (0, 1)))) is None,
        "a plain 2D viewer reports no time axis",
    )


def test_frame_clock() -> None:
    print("the frame clock keeps the rate and drops frames instead of slowing down")
    check(timeseries.next_index(3, 0, 9, 1, loop=True) == 4, "one step forward")
    check(timeseries.next_index(9, 0, 9, 1, loop=True) == 0, "looping wraps to the start")
    check(timeseries.next_index(9, 0, 9, 1, loop=False) is None, "without looping the end stops playback")
    check(timeseries.next_index(0, 0, 9, -1, loop=True) == 9, "backwards wraps the other way")
    check(timeseries.next_index(2, 2, 5, 7, loop=True) == 5, "a big jump wraps inside the range")
    check(timeseries.next_index(4, 3, 7, 1, loop=True) == 5, "a sub-range plays inside itself")

    check(timeseries.clamp_range(8, 2, 6) == (2, 5), "a reversed, out-of-bounds range is fixed up")
    check(timeseries.clamp_range(0, 0, 0) == (0, 0), "an empty series clamps to nothing")

    # 10 fps: a tick 300 ms late is three frames of catching up, not one.
    check(timeseries.frames_to_advance(0.1, 10) == 1, "an on-time tick advances one frame")
    check(timeseries.frames_to_advance(0.3, 10) == 3, "a late tick advances by the frames that elapsed")
    check(timeseries.frames_to_advance(0.001, 10) == 1, "an early tick still advances one frame")
    check(
        timeseries.frames_to_advance(10.0, 30, max_skip=4) == 4,
        "one long stall is capped rather than skipping most of the series",
    )


def test_timestamps() -> None:
    print("timepoints are labelled with the acquisition time when the file records one")
    check(timeseries.elapsed_seconds(4, 2.5) == 10.0, "four intervals of 2.5 s is 10 s")
    check(timeseries.elapsed_seconds(4, None) is None, "no interval means no elapsed time")
    check(timeseries.elapsed_seconds(4, 0) is None, "a zero interval is not a time base")
    check(timeseries.format_timestamp(65.4) == "1:05.4", "short series read as minutes and seconds")
    check(timeseries.format_timestamp(3725) == "1:02:05", "long series switch to hours")
    check(timeseries.format_timestamp(None) == "", "no timestamp is an empty label, not a zero")

    layer = _Layer(_stack(), metadata=_Meta(time_interval_s=30.0))
    check(timeseries.time_interval(layer) == 30.0, "the interval comes off the acquisition metadata")


def test_level_budget() -> None:
    print("levels are cached finest-first, and only while they fit the budget")
    levels = [_stack((4, 1, 64, 64)), _stack((4, 1, 32, 32)), _stack((4, 1, 16, 16))]
    sizes = [timeseries.level_bytes(level) for level in levels]
    check(sizes[0] == 4 * 64 * 64 * 2, "level bytes count every timepoint, not one frame")

    check(
        timeseries.levels_within_budget(levels, sum(sizes)) == [0, 1, 2],
        "everything is cached when it all fits",
    )
    check(
        timeseries.levels_within_budget(levels, sizes[1] + sizes[2]) == [1, 2],
        "an oversized level 0 is skipped and the coarser ones still cached",
    )
    check(timeseries.levels_within_budget(levels, 0) == [], "a zero budget caches nothing")

    os.environ[timeseries.BUDGET_ENV_VAR] = "12345"
    try:
        check(timeseries.timeline_budget() == 12345, "the budget can be overridden from the environment")
    finally:
        os.environ.pop(timeseries.BUDGET_ENV_VAR, None)
    check(
        timeseries.timeline_budget() == timeseries.DEFAULT_TIMELINE_BUDGET,
        "and falls back to the default when it is not set",
    )


def test_touch_reads_without_copying() -> None:
    print("prefetching touches a byte per page rather than copying the frame")
    array = _stack((2, 4, 64, 64))
    timeseries.touch(array[0])  # must not raise, and must not change anything
    before = array.copy()
    timeseries.touch(array[1])
    check(np.array_equal(array, before), "touching a frame leaves the data alone")

    # A non-contiguous view takes the fallback path rather than failing.
    timeseries.touch(array[:, ::2])
    check(True, "a non-contiguous slice is handled too")


def test_caching_swaps_levels_in_place(directory: Path) -> None:
    print("caching points the layer at local copies without changing what it shows")
    volume_cache.clear()
    levels = [_stack((6, 2, 64, 64)), _stack((6, 2, 32, 32))]
    layer = _Layer(
        [_lazy(level) for level in levels],
        name="cells",
        multiscale=True,
        metadata=_Meta(file_path=directory / "cells.ims"),
    )
    viewer = _Viewer([layer])
    manager = timeseries.TimelineManager(viewer, budget=10**9, threaded=False)

    check(manager.is_cached(layer) is False, "nothing is local before the copy runs")
    check(manager.describe(layer) == "reading from the source", "and the panel says so")

    # Opening a file must not start a copy: that is when the viewer is busiest
    # reading the first slice, and a background copy of the whole series would
    # be competing with it for the same network share.
    manager.apply()
    check(manager.is_cached(layer) is False, "opening a file copies nothing on its own")
    check(volume_cache.total_size() == 0, "and writes nothing to disk")

    manager.cache_layer(layer)

    check(manager.is_cached(layer) is True, "after caching the layer reads from a memmap")
    check(
        all(isinstance(level, np.memmap) for level in layer.data),
        "every level that fitted the budget was cached",
    )
    check(manager.describe(layer) == "all levels local", "the panel reports the levels that are local")
    check(
        [tuple(level.shape) for level in layer.data] == [tuple(level.shape) for level in levels],
        "shapes are unchanged, so the scale and the multiscale rendering still hold",
    )
    check(
        np.array_equal(np.asarray(layer.data[0]), levels[0]),
        "and the cached pixels are the ones that were read",
    )

    # Running again is a no-op rather than a second copy.
    check(manager.cache_layer(layer) is None, "a second run finds nothing left to do")

    # A layer that has been through the 3D path keeps its pyramid in metadata;
    # the cached arrays have to land there too or leaving 3D restores the
    # originals and playback goes back to the network.
    check(layer.metadata.get("mv_pyramid") is None, "an untouched layer has no parked pyramid")
    layer.metadata["mv_pyramid"] = list(layer.data)
    manager._adopt(layer, 0, layer.data[0])
    check(
        isinstance(layer.metadata["mv_pyramid"][0], np.memmap),
        "when a pyramid is parked for 3D, the cached level is written into it",
    )

    del manager
    gc.collect()


def test_cache_is_shared_with_the_3d_view(directory: Path) -> None:
    print("the 3D view and playback share one cache entry, not one each")
    from microscopy_viewer import volume_cache as vc

    volume_cache.clear()
    level = _stack((6, 2, 64, 64))
    source = directory / "shared.ims"
    layer = _Layer(
        [_lazy(level)], name="shared", multiscale=True, metadata=_Meta(file_path=source)
    )
    viewer = _Viewer([layer])

    manager = timeseries.TimelineManager(viewer, budget=10**9, threaded=False)
    manager.cache_layer(layer)
    check(vc.total_size() > 0, "playback wrote one entry")
    written = vc.total_size()

    # This is the key the 3D path builds for the same level of the same layer.
    key = vc.layer_key(source, "shared", 0, level.shape, level.dtype)
    check(
        vc.load(key, level.shape, level.dtype) is not None,
        "and the 3D view finds it under the key it would have used itself",
    )

    # A fresh manager on a fresh layer adopts it without copying anything again.
    again = _Layer([_lazy(level)], name="shared", multiscale=True, metadata=_Meta(file_path=source))
    second = timeseries.TimelineManager(_Viewer([again]), budget=10**9, threaded=False)
    adopted = second.adopt_existing(again)
    check(adopted == 1, "reopening the file adopts the existing copy")
    check(second.is_cached(again), "so it is local straight away")
    check(vc.total_size() == written, "and nothing was written a second time")

    del manager, second
    gc.collect()


def test_caching_stays_out_of_the_way_in_3d(directory: Path) -> None:
    print("no timeline copy is started while the viewer is rendering a volume")
    volume_cache.clear()
    layer = _Layer(
        [_lazy(_stack((6, 2, 64, 64)))],
        name="vol",
        multiscale=True,
        metadata=_Meta(file_path=directory / "vol.ims"),
    )
    viewer = _Viewer([layer])
    viewer.dims.ndisplay = 3
    manager = timeseries.TimelineManager(viewer, budget=10**9, threaded=False)

    manager.start_caching()
    check(not manager.is_cached(layer), "in 3D the copy is left to the 3D path, which is already doing it")
    check(volume_cache.total_size() == 0, "so the network is not read twice at once")
    check(manager._prefetcher._targets == [], "and nothing is prefetched behind the volume upload")

    # The panel's own button is explicit, so it is honoured either way.
    manager.cache_layer(layer, force=True)
    check(manager.is_cached(layer), "Cache locally still works when the user asks for it")

    viewer.dims.ndisplay = 2
    manager.apply()
    check(manager.is_cached(layer), "and the copy is still in use back in 2D")

    del manager
    gc.collect()


def test_caching_respects_the_budget(directory: Path) -> None:
    print("an oversized time series caches what fits and leaves the rest alone")
    volume_cache.clear()
    levels = [_stack((6, 2, 64, 64)), _stack((6, 2, 32, 32))]
    layer = _Layer(
        [_lazy(level) for level in levels],
        name="big",
        multiscale=True,
        metadata=_Meta(file_path=directory / "big.ims"),
    )
    viewer = _Viewer([layer])
    budget = timeseries.level_bytes(levels[1]) + 1
    manager = timeseries.TimelineManager(viewer, budget=budget, threaded=False)
    manager.cache_layer(layer)

    check(not isinstance(layer.data[0], np.memmap), "the level that did not fit is still the original")
    check(isinstance(layer.data[1], np.memmap), "the coarser level, which playback uses zoomed out, is local")
    check("levels 1 of 1 local" in manager.describe(layer), "and the panel says which levels are local")

    del manager
    gc.collect()


def test_caching_can_be_switched_off(directory: Path) -> None:
    print("MICROSCOPY_VIEWER_NO_CACHE leaves the data where it is")
    volume_cache.clear()
    layer = _Layer(
        _lazy(_stack((6, 2, 64, 64))), name="plain", metadata=_Meta(file_path=directory / "plain.ims")
    )
    viewer = _Viewer([layer])
    os.environ[volume_cache.DISABLE_ENV_VAR] = "1"
    try:
        manager = timeseries.TimelineManager(viewer, threaded=False)
        manager.apply()
        check(not isinstance(layer.data, np.memmap), "no copy is made when caching is disabled")
        check(volume_cache.total_size() == 0, "and nothing is written to disk")
    finally:
        os.environ.pop(volume_cache.DISABLE_ENV_VAR, None)
        del manager
        gc.collect()


def test_prefetcher_walks_ahead() -> None:
    print("the prefetcher warms the timepoints in front of the playhead")
    array = _stack((8, 1, 64, 64))
    prefetcher = timeseries._Prefetcher(ahead=3)
    prefetcher.set_targets([(array, 0, ())])
    prefetcher.update(2, 1)
    try:
        deadline = time.monotonic() + 2.0
        while prefetcher._thread is None and time.monotonic() < deadline:
            time.sleep(0.01)
        check(prefetcher._thread is not None, "a worker thread starts once there is somewhere to look")
        time.sleep(0.1)
        check(prefetcher._thread.is_alive(), "and it stays alive between frames")
    finally:
        prefetcher.stop()
    check(prefetcher._thread is None, "stopping joins the thread")

    # An index past the end wraps, because looping playback does.
    prefetcher._warm(array, 0, (), 11)
    check(True, "warming a wrapped index does not raise")


def test_prefetcher_idles_when_nothing_moves() -> None:
    print("and goes quiet when playback stops, instead of re-reading on a timer")
    array = _stack((8, 1, 64, 64))
    reads: list[int] = []
    original = timeseries.touch
    timeseries.touch = lambda frame: reads.append(1)  # type: ignore[assignment]

    prefetcher = timeseries._Prefetcher(ahead=3)
    try:
        prefetcher.set_targets([(array, 0, ())])
        check(not reads, "pointing it at an array reads nothing on its own")

        prefetcher.update(2, 1)
        deadline = time.monotonic() + 2.0
        while not reads and time.monotonic() < deadline:
            time.sleep(0.01)
        check(len(reads) == 3, f"a playhead move warms the three frames ahead of it ({len(reads)})")

        # Longer than the timer this used to wake on, which kept re-reading the
        # same timepoints and evicted the pages the 3D view had just uploaded.
        after_one_pass = len(reads)
        time.sleep(2.5)
        check(len(reads) == after_one_pass, f"and then it reads nothing more ({len(reads)})")
    finally:
        prefetcher.stop()
        timeseries.touch = original  # type: ignore[assignment]


def main() -> int:
    print("Time-series playback checks\n")
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        os.environ["LOCALAPPDATA"] = str(directory / "appdata")
        # Small test arrays would otherwise be below the cache's size floor.
        original_floor = volume_cache.MIN_CACHE_BYTES
        volume_cache.MIN_CACHE_BYTES = 0
        try:
            test_time_axis_detection()
            test_frame_clock()
            test_timestamps()
            test_level_budget()
            test_touch_reads_without_copying()
            test_caching_swaps_levels_in_place(directory)
            test_cache_is_shared_with_the_3d_view(directory)
            test_caching_stays_out_of_the_way_in_3d(directory)
            test_caching_respects_the_budget(directory)
            test_caching_can_be_switched_off(directory)
            test_prefetcher_walks_ahead()
            test_prefetcher_idles_when_nothing_moves()
        finally:
            volume_cache.MIN_CACHE_BYTES = original_floor
            gc.collect()

    print()
    if _failures:
        print(f"{len(_failures)} check(s) failed:")
        for message in _failures:
            print(f"  - {message}")
        return 1
    print("All time-series checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
