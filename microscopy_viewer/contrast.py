"""Contrast helpers used by the toolbar's Auto Contrast / Reset Contrast buttons."""

from __future__ import annotations

import numpy as np

from .utils import get_logger

logger = get_logger("contrast")

#: Percentiles used for auto contrast. Clipping the extremes keeps hot pixels
#: and camera offset from flattening the visible range.
LOW_PERCENTILE = 0.5
HIGH_PERCENTILE = 99.5

#: Cap on how many voxels are sampled when computing percentiles, so a lazily
#: loaded multi-gigabyte stack does not stall the GUI.
MAX_SAMPLE = 4_000_000


def current_slice(viewer, layer) -> np.ndarray | None:
    """The data currently displayed for *layer*, as a small in-memory array.

    For multiscale layers the coarsest level is used: it is cheap to read and its
    intensity distribution matches the full-resolution data closely enough for
    setting contrast limits.
    """
    data = layer.data[-1] if layer.multiscale else layer.data
    ndim = int(getattr(data, "ndim", 0))
    if ndim == 0:
        return None

    # napari right-aligns layer dimensions against world dimensions.
    offset = viewer.dims.ndim - ndim
    displayed = set(viewer.dims.displayed)
    index: list[object] = []
    for axis in range(ndim):
        world_axis = axis + offset
        if world_axis in displayed or world_axis < 0:
            index.append(slice(None))
            continue
        step = 0
        if 0 <= world_axis < len(viewer.dims.current_step):
            step = int(viewer.dims.current_step[world_axis])
        # Scaled layers can put the world step beyond this layer's extent.
        index.append(max(0, min(step, int(data.shape[axis]) - 1)))

    try:
        view = data[tuple(index)]
        array = np.asarray(view)
    except Exception:
        logger.debug("could not slice %s for auto contrast", layer.name, exc_info=True)
        return None

    if array.size > MAX_SAMPLE:
        stride = int(np.ceil(np.sqrt(array.size / MAX_SAMPLE)))
        array = array[::stride, ::stride] if array.ndim >= 2 else array[::stride]
    return array


def auto_contrast(viewer, layers=None) -> int:
    """Set contrast limits from the visible data's percentiles.

    Returns the number of layers that were adjusted.
    """
    from napari.layers import Image

    targets = list(layers) if layers is not None else [
        layer for layer in viewer.layers if isinstance(layer, Image) and layer.visible
    ]
    adjusted = 0
    for layer in targets:
        if not isinstance(layer, Image):
            continue
        array = current_slice(viewer, layer)
        if array is None or array.size == 0:
            continue
        finite = array[np.isfinite(array)] if array.dtype.kind == "f" else array
        if finite.size == 0:
            continue
        low, high = (float(v) for v in np.percentile(finite, [LOW_PERCENTILE, HIGH_PERCENTILE]))
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            low, high = float(np.min(finite)), float(np.max(finite))
        if high <= low:
            high = low + 1.0
        # Widen the layer's allowed range first, or napari clamps the new limits.
        range_low, range_high = layer.contrast_limits_range
        layer.contrast_limits_range = (min(range_low, low), max(range_high, high))
        layer.contrast_limits = (low, high)
        adjusted += 1
    return adjusted


def reset_contrast(viewer, layers=None) -> int:
    """Restore each image layer's contrast limits to its full data range."""
    from napari.layers import Image

    targets = list(layers) if layers is not None else [
        layer for layer in viewer.layers if isinstance(layer, Image)
    ]
    adjusted = 0
    for layer in targets:
        if not isinstance(layer, Image):
            continue
        try:
            layer.reset_contrast_limits_range()
            layer.reset_contrast_limits()
        except Exception:
            low, high = layer.contrast_limits_range
            layer.contrast_limits = (low, high)
        adjusted += 1
    return adjusted
