"""Show an image layer less its local median, as the cells analysis compares it.

The positivity rule of the cells analysis (median + 7 SD, *mean (median filter)*)
looks at each channel less its local median: the median over
:data:`~microscopy_viewer.junction_analysis.MEDIAN_UM` (45 µm) round each pixel,
worked out at 1/8 resolution and scaled back up
(:func:`microscopy_viewer.junction_analysis.less_local_median`). This puts that
image on screen for any image layer, with the same settings:

* right-click an image layer in the layer list ▸ **Median filter: less the
  local median (45 µm)** (:func:`install_layer_action`);
* the Cell counter panel's **Show less local median** button.

A stack is flattened to its brightest plane first, as the analysis does. The
work runs off the main thread; the new layer, ``<layer> less local median``, has
the source layer's colour and is tied to its sample.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Callable

import numpy as np

from .. import junction_analysis as ja
from ..utils import get_logger
from .sample_files import sample_of, tie_to_sample

try:  # the layer list the right-click action is handed (resolved by name, so at module level)
    from napari.components import LayerList
except Exception:  # pragma: no cover - napari missing
    LayerList = object

logger = get_logger("local_median")

#: Suffix of the layer added.
SUFFIX = " less local median"
#: Its brightness range: from 0 to this percentile of the image.
PERCENTILE = 99.9
#: The right-click action, in napari's layer list.
ACTION_ID = "argus.layer.less_local_median"
ACTION_TITLE = "Median filter: less the local median (45 µm)"

#: Workers still running, kept so they are not collected half-way.
_RUNNING: set = set()


def finest_plane(layer) -> tuple[np.ndarray, tuple[float, float]]:
    """The layer's finest level flattened to its brightest plane (as the analysis), and its pixel size (y, x)."""
    pyramid = layer.metadata.get("mv_pyramid")
    data = pyramid[0] if pyramid else (layer.data[0] if getattr(layer, "multiscale", False) else layer.data)
    scale = layer.metadata.get("mv_pyramid_scale") or layer.scale
    shape = np.shape(data)
    whole = (slice(0, int(shape[-2])), slice(0, int(shape[-1])))
    plane = ja._plane(SimpleNamespace(data=data, multiscale=False), whole)
    return np.asarray(plane), (float(scale[-2]), float(scale[-1]))


def less_local_median(layer) -> tuple[np.ndarray, tuple[float, float], tuple[float, float]]:
    """The layer less its local median, its pixel size, and the brightness range to show it with."""
    plane, voxel = finest_plane(layer)
    out = ja.less_local_median(plane, voxel[1])
    sample = out[::4, ::4]
    top = float(np.nanpercentile(sample, PERCENTILE)) if sample.size else 1.0
    return out, voxel, (0.0, max(top, 1.0))


def add_less_local_median(viewer, layer, status: Callable[[str], None] | None = None) -> None:
    """Work out *layer* less its local median aside, then add it as a layer above it."""
    from napari.qt.threading import thread_worker

    say = status or (lambda _text: None)
    if layer is None or getattr(layer, "rgb", False):
        say("Choose a single-channel image layer.")
        return
    name = f"{layer.name}{SUFFIX}"
    say(f"Working out {layer.name} less its local median ({ja.MEDIAN_UM:g} µm)…")

    @thread_worker
    def _work():
        return less_local_median(layer)

    def _done(result) -> None:
        data, voxel, limits = result
        if name in viewer.layers:
            viewer.layers.remove(name)
        translate = tuple(float(t) for t in layer.translate)[-2:]
        added = viewer.add_image(data, name=name, scale=voxel, translate=translate, colormap=layer.colormap,
                                 contrast_limits=limits, blending="additive")
        try:
            added.units = tuple(layer.units)[-2:]
        except Exception:
            logger.debug("could not copy the layer's units", exc_info=True)
        tie_to_sample(added, sample_of(layer))
        say(f"“{name}” added: {layer.name} less its local median over {ja.MEDIAN_UM:g} µm, "
            "the image the cells analysis compares.")

    def _failed(exc) -> None:
        logger.exception("less local median failed for %s", layer.name)
        say(f"Could not work out {layer.name} less its local median: {exc}")

    worker = _work()
    _RUNNING.add(worker)
    worker.returned.connect(_done)
    worker.errored.connect(_failed)
    worker.finished.connect(lambda: _RUNNING.discard(worker))
    worker.start()


def install_layer_action(viewer, status: Callable[[str], None] | None = None) -> bool:
    """Add the right-click action on image layers in napari's layer list. Returns whether it was added."""
    try:
        from app_model.types import Action
        from napari._app_model import get_app_model
        from napari._app_model.constants import MenuGroup, MenuId
        from napari._app_model.context import LayerListSelectionContextKeys as LLSCK
        from napari.layers import Image
    except Exception:
        logger.debug("this napari has no layer-list actions to add to", exc_info=True)
        return False

    def _run(ll: LayerList) -> None:
        for layer in [layer for layer in ll.selection if isinstance(layer, Image)]:
            add_less_local_median(viewer, layer, status)

    app = get_app_model()
    if ACTION_ID in app.commands:
        return True
    try:
        app.register_action(Action(
            id=ACTION_ID,
            title=ACTION_TITLE,
            callback=_run,
            menus=[{"id": MenuId.LAYERLIST_CONTEXT, "group": MenuGroup.LAYERLIST_CONTEXT.CONVERSION,
                    "when": LLSCK.num_selected_image_layers >= 1}],
        ))
    except Exception:
        logger.exception("could not add the less-local-median action")
        return False
    return True
