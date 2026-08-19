"""Make 3D / maximum-intensity-projection views show real resolution.

napari renders multiscale images at their *coarsest* pyramid level whenever the
viewer is in 3D — ``napari/layers/image/_slice.py`` picks ``len(self.data) - 1``
outright, with no regard for zoom. For a dataset opened from an Imaris pyramid
that means switching to 3D or MIP produces a small, blocky volume.

The workaround is to hand napari a pyramid with a single level while it is in 3D:
``len(data) - 1`` is then ``0``, so the level it picks is the one we chose. The
full pyramid goes back when the viewer returns to 2D, where multiscale rendering
works properly and is worth having.

A layer's ``multiscale`` flag cannot be changed after construction, but its
``data`` can be reassigned, which is why the level list is swapped rather than the
flag.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from . import gpu, volume_cache
from .utils import get_logger

logger = get_logger("rendering")

#: Kept for callers that want the old constant; the live budget now comes from
#: :func:`microscopy_viewer.gpu.voxel_budget`, which sizes itself to the card.
DEFAULT_VOXEL_BUDGET = gpu.FALLBACK_VOXEL_BUDGET
BUDGET_ENV_VAR = gpu.BUDGET_ENV_VAR

# Keys used on ``layer.metadata`` to remember what a layer looked like in 2D.
_PYRAMID = "mv_pyramid"
_BASE_SCALE = "mv_pyramid_scale"
_LEVEL = "mv_pyramid_level"
_CACHED = "mv_cached_level"


def voxel_budget(bytes_per_voxel: int = 2) -> int:
    """The 3D voxel budget for this GPU."""
    return gpu.voxel_budget(bytes_per_voxel=bytes_per_voxel)


def displayed_voxels(shape: Sequence[int]) -> int:
    """Voxel count of the volume actually uploaded: the last three axes.

    Leading axes such as time are sliced by napari before rendering, so they do
    not contribute to the size of the 3D texture.
    """
    spatial = tuple(int(n) for n in shape[-3:])
    return int(np.prod(spatial)) if spatial else 0


def choose_level(levels: Sequence, budget: int, max_axis: int | None = None) -> int:
    """Index of the finest pyramid level the GPU can actually hold.

    A level qualifies when its volume fits within *budget* and no spatial axis
    exceeds the hardware's maximum 3D texture size — a volume that is small
    enough overall can still be refused for being too long in one direction.
    Falls back to the coarsest level when nothing qualifies: a blocky volume
    beats a failed upload.
    """
    for index, level in enumerate(levels):
        if displayed_voxels(level.shape) > budget:
            continue
        if max_axis is not None and any(int(n) > max_axis for n in tuple(level.shape)[-3:]):
            continue
        return index
    return len(levels) - 1


class MultiscaleDepthManager:
    """Keeps multiscale layers at a usable resolution while the viewer is in 3D.

    Attach one to a viewer; it watches ``dims.ndisplay`` and the layer list, and
    swaps pyramids in and out as needed. ``on_status``, if given, is called with
    a short human-readable note whenever the 3D resolution changes.
    """

    def __init__(
        self,
        viewer,
        budget: int | None = None,
        on_status: Callable[[str], None] | None = None,
        cache: bool = True,
    ):
        self._viewer = viewer
        self._explicit_budget = budget
        self._on_status = on_status
        self._cache = cache
        self._limits = gpu.GpuLimits()
        self._workers: dict[int, object] = {}
        self._connect()

    def _budget_for(self, dtype) -> int:
        """Voxel budget for a layer of this dtype, sized to the GPU."""
        if self._explicit_budget is not None:
            return self._explicit_budget
        # Queried lazily: at construction time napari's canvas may not exist yet.
        if not self._limits.queried:
            self._limits = gpu.query_limits()
        return gpu.voxel_budget(self._limits, bytes_per_voxel=np.dtype(dtype).itemsize)

    @property
    def _max_axis(self) -> int | None:
        return self._limits.max_3d_texture if self._limits.queried else None

    # -- wiring ---------------------------------------------------------------

    def _connect(self) -> None:
        try:
            self._viewer.dims.events.ndisplay.connect(self._on_event)
            self._viewer.layers.events.inserted.connect(self._on_event)
        except Exception:  # pragma: no cover - napari event API drift
            logger.warning("could not connect the 3D resolution handler", exc_info=True)

    def _on_event(self, event=None) -> None:
        self.apply()

    # -- the swap -------------------------------------------------------------

    def apply(self) -> None:
        """Bring every layer in line with the current ``ndisplay``. Idempotent."""
        in_3d = int(getattr(self._viewer.dims, "ndisplay", 2)) >= 3
        notes: list[str] = []
        for layer in list(self._viewer.layers):
            try:
                note = self._apply_to(layer, in_3d)
            except Exception:
                logger.exception("could not adjust %s for 3D display", getattr(layer, "name", "?"))
                continue
            if note:
                notes.append(note)
        if notes and self._on_status is not None:
            self._on_status(" ".join(notes[:2]))

    def _apply_to(self, layer, in_3d: bool) -> str | None:
        levels = self._pyramid(layer)
        if levels is None:
            return None
        return self._enter_3d(layer, levels) if in_3d else self._restore_2d(layer, levels)

    def _pyramid(self, layer) -> list | None:
        """The layer's full level list, registering it on first sight.

        Returns ``None`` for anything that is not a genuine multiscale image.
        """
        from napari.layers import Image

        if not isinstance(layer, Image):
            return None
        stored = layer.metadata.get(_PYRAMID)
        if stored is not None:
            return stored
        if not layer.multiscale or len(layer.data) < 2:
            return None
        levels = list(layer.data)
        layer.metadata[_PYRAMID] = levels
        layer.metadata[_BASE_SCALE] = tuple(float(s) for s in layer.scale)
        layer.metadata[_LEVEL] = None
        return levels

    def _enter_3d(self, layer, levels: list) -> str | None:
        budget = self._budget_for(levels[0].dtype)
        best = choose_level(levels, budget, self._max_axis)
        if layer.metadata.get(_LEVEL) == best:
            return None

        base_scale = np.asarray(layer.metadata[_BASE_SCALE], dtype=float)
        # A coarser level covers the same physical extent with fewer voxels, so
        # its voxel size grows by exactly the shape ratio. Scaling by that keeps
        # world coordinates — and therefore ROIs and the scale bar — correct.
        factor = np.asarray(levels[0].shape, dtype=float) / np.asarray(levels[best].shape, dtype=float)
        self._swap(layer, [levels[best]], tuple(base_scale * factor))
        layer.metadata[_LEVEL] = best
        self._start_caching(layer, levels[best], best)

        if best == 0:
            logger.info(
                "3D view: %s at full resolution (%s, budget %d voxels)",
                layer.name, self._limits.describe(), budget,
            )
            return None
        logger.info(
            "3D view: %s at pyramid level %d of %d — %d voxels exceeds the budget of %d (%s)",
            layer.name, best, len(levels) - 1,
            displayed_voxels(levels[0].shape), budget, self._limits.describe(),
        )
        return (
            f"3D view is showing level {best} of {len(levels) - 1} — the full-resolution "
            f"volume needs more than the {budget // 1_000_000}M voxels this GPU can hold."
        )

    # -- local caching --------------------------------------------------------

    def _start_caching(self, layer, array, level: int) -> None:
        """Copy the displayed volume to local disk, off the main thread.

        Source data typically lives on a NAS, where every re-read costs seconds.
        Once the local copy exists the layer is switched onto it, so rotating,
        re-entering 3D and stepping through time all read locally instead.
        """
        if not self._cache or not volume_cache.enabled():
            return
        if isinstance(array, np.memmap) or layer.metadata.get(_CACHED) == level:
            return

        meta = layer.metadata.get("mv_metadata")
        source = getattr(meta, "file_path", None) if meta is not None else None
        # Same key the time-series cache uses, so a dataset that is both played
        # and viewed in 3D is copied once rather than once per feature.
        key = volume_cache.layer_key(source, layer.name, level, array.shape, array.dtype)

        existing = volume_cache.load(key, array.shape, array.dtype)
        if existing is not None:
            self._adopt_cache(layer, existing, level)
            return

        def _work():
            return volume_cache.store(array, key)

        def _done(cached):
            self._workers.pop(id(layer), None)
            if cached is None:
                return
            self._adopt_cache(layer, cached, level)
            if self._on_status is not None:
                self._on_status(f"{layer.name}: cached locally for faster 3D redraws.")

        try:
            from napari.qt.threading import thread_worker
        except Exception:  # pragma: no cover - no Qt threading available
            _done(_work())
            return

        worker = thread_worker(_work)()
        worker.returned.connect(_done)
        worker.errored.connect(lambda exc: logger.warning("caching failed: %s", exc))
        self._workers[id(layer)] = worker
        worker.start()

    def _adopt_cache(self, layer, cached, level: int) -> None:
        """Point the layer at the local copy, if it is still showing that level."""
        if layer.metadata.get(_LEVEL) != level or layer not in self._viewer.layers:
            return
        try:
            self._swap(layer, [cached], tuple(layer.scale))
        except Exception:
            logger.warning("could not switch %s onto its cached volume", layer.name, exc_info=True)
            return
        layer.metadata[_CACHED] = level
        logger.info("%s is now reading from the local cache", layer.name)

    def _restore_2d(self, layer, levels: list) -> None:
        if layer.metadata.get(_LEVEL) is None:
            return None
        self._swap(layer, levels, layer.metadata[_BASE_SCALE])
        layer.metadata[_LEVEL] = None
        # The cached copy stays on disk; only the layer stops pointing at it, so
        # re-entering 3D picks it straight back up without touching the NAS.
        layer.metadata[_CACHED] = None
        return None

    @staticmethod
    def _thumbnail_level(levels: list) -> int:
        """napari's own thumbnail-level rule, which we have to reapply by hand.

        It picks the coarsest level that still has an axis of at least 64 px, so
        the thumbnail does not come out of a heavily downsampled level. napari
        only computes this in the layer constructor.
        """
        big_enough = [bool(np.any(np.greater_equal(level.shape, 64))) for level in levels]
        return int(np.where(big_enough)[0][-1]) if any(big_enough) else 0

    @classmethod
    def _swap(cls, layer, levels: list, scale) -> None:
        """Replace a layer's level list and scale without tripping napari's slicer.

        ``_data_level`` and ``_thumbnail_level`` both index the level list, and
        both are only ever assigned in the layer constructor. Assigning ``data``
        triggers an immediate reslice, so they have to be brought into range for
        the *new* list first or napari indexes off the end of it.
        """
        layer._data_level = 0
        layer._thumbnail_level = 0
        layer.data = levels
        layer._thumbnail_level = cls._thumbnail_level(levels)
        layer.scale = tuple(scale)
