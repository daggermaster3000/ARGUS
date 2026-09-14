"""The contract between readers and the GUI: one :class:`LayerSpec` per napari layer."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ..metadata import AcquisitionMetadata
from ..utils import get_logger

logger = get_logger("layer_spec")

#: Fallback colormaps for multichannel data that carries no colour metadata.
COLORMAP_CYCLE = ("green", "magenta", "cyan", "yellow", "red", "blue", "bop orange", "gray")

#: Long, unambiguous words: matched against the channel name with all separators
#: stripped, so "Bright Field", "bright-field" and "BrightField" all hit.
_BRIGHTFIELD_SUBSTRINGS = (
    "brightfield",
    "transmitted",
    "transmission",
    "phasecontrast",
    "tpmt",  # Zeiss transmitted-light PMT, written "T-PMT"
    "esid",  # Zeiss Airyscan transmitted detector
)

#: Short abbreviations, matched as whole words only. Substring matching here would
#: turn "BFP" (blue fluorescent protein) into a brightfield channel.
_BRIGHTFIELD_TOKENS = frozenset({"bf", "tl", "dic", "ph", "phc", "phase", "trans", "bright"})


def is_brightfield(name: str | None) -> bool:
    """Whether a channel name describes transmitted light rather than fluorescence.

    Brightfield, DIC and phase-contrast channels carry no emission colour, so
    tinting them looks wrong — they belong in grayscale. Microscope software
    still hands them a slot in the channel colour cycle, and vendors often store
    an arbitrary display colour for them, so the name is what we go on.
    """
    if not name:
        return False
    lowered = str(name).lower()
    squashed = re.sub(r"[^a-z0-9]", "", lowered)
    if any(word in squashed for word in _BRIGHTFIELD_SUBSTRINGS):
        return True
    tokens = {token for token in re.split(r"[^a-z0-9]+", lowered) if token}
    return bool(tokens & _BRIGHTFIELD_TOKENS)


def channel_appearance(
    channel_name: str | None,
    color: tuple[float, float, float] | None,
    index: int,
    n_channels: int,
) -> tuple[str, tuple[float, float, float] | None, str]:
    """Pick ``(colormap, color, blending)`` for one channel.

    Brightfield channels are forced to grayscale, overriding any display colour
    stored in the file — a transmitted-light channel tinted green is never what
    was wanted. Fluorescence channels keep the file's colour when there is one,
    and otherwise take the next entry from the cycle.
    """
    if is_brightfield(channel_name):
        return "gray", None, "additive" if n_channels > 1 else "translucent"
    fallback = COLORMAP_CYCLE[index % len(COLORMAP_CYCLE)] if n_channels > 1 else "gray"
    return fallback, color, "additive" if n_channels > 1 else "translucent"


def napari_colormap(color: tuple[float, float, float] | None, name: str):
    """A black-to-*color* napari colormap, or ``None`` to fall back to a named one.

    Imported lazily: this is the only place the loaders touch napari, and it runs
    when a layer is actually being added rather than while parsing a file.
    """
    if color is None or max(color) <= 0:
        return None
    try:
        from napari.utils import Colormap

        red, green, blue = (float(min(max(component, 0.0), 1.0)) for component in color)
        return Colormap(
            colors=[[0.0, 0.0, 0.0, 1.0], [red, green, blue, 1.0]],
            name=f"mv-{name}",
            display_name=name,
        )
    except Exception:  # pragma: no cover - colour is cosmetic, never fatal
        logger.debug("could not build a colormap for %s", name, exc_info=True)
        return None


def supports_layer_units() -> bool:
    """Whether this napari takes a per-axis ``units`` argument on a layer.

    napari moved the scale bar's unit onto the layers: up to 0.6 the overlay had
    its own ``unit`` field, and from 0.8 ``ScaleBarOverlay`` has none at all and
    the bar reads ``layer.units``, which defaults to *pixel*. A calibrated stack
    whose layers never said "µm" therefore gets a scale bar reading "25 pixels",
    which is the wrong answer stated confidently.

    Checked once, by signature, so the same code is right on both.
    """
    global _SUPPORTS_UNITS
    if _SUPPORTS_UNITS is None:
        try:
            import inspect

            from napari.layers import Image

            _SUPPORTS_UNITS = "units" in inspect.signature(Image.__init__).parameters
        except Exception:  # pragma: no cover - napari API drift
            logger.debug("could not tell whether napari takes layer units", exc_info=True)
            _SUPPORTS_UNITS = False
    return bool(_SUPPORTS_UNITS)


_SUPPORTS_UNITS: bool | None = None


#: Units napari gives a layer that was never told any. Dimensionless, which is
#: what makes it clash with a calibrated image rather than simply differ from it.
DEFAULT_UNIT = "pixel"


def units_like(source, ndim: int) -> dict:
    """``units`` for a new layer, copied from the layer it is derived from.

    Right-aligned onto *ndim* axes, because a 2D ROI or label map drawn over a
    4D stack takes the last two of its axes.

    Empty when this napari has no layer units, or when there is nothing to copy —
    in both cases the caller simply does not pass the argument.
    """
    if source is None or not supports_layer_units():
        return {}
    units = getattr(source, "units", None)
    if units is None:
        return {}
    try:
        taken = tuple(units)[-int(ndim):]
    except Exception:  # pragma: no cover - napari API drift
        logger.debug("could not read layer units", exc_info=True)
        return {}
    return {"units": taken} if len(taken) == int(ndim) else {}


def world_units(viewer, ndim: int, exclude=None) -> dict:
    """``units`` for a new layer, matched to the calibrated layers already open.

    Every layer added without this defaults to *pixel*, which is dimensionless.
    napari compares units across layers right-aligned and by dimensionality, so a
    single pixel-unit layer makes the whole list inconsistent — at which point it
    warns, drops units from rendering, and **the scale bar goes back to reading
    pixels over a calibrated image**. Adding a ROI or a region outline should not
    do that.

    The first layer carrying a real unit wins. When nothing is calibrated there is
    nothing to match and the default is already right.

    *exclude* is the layer being matched, when it is already in the list: without
    it a layer asking what the others use is answered with its own units, and can
    never be corrected.
    """
    if viewer is None or not supports_layer_units():
        return {}
    try:
        layers = [layer for layer in viewer.layers if layer is not exclude]
    except Exception:  # pragma: no cover - defensive
        return {}
    for layer in layers:
        units = getattr(layer, "units", None)
        if units is None:
            continue
        # "Real" means dimensioned: a layer still on pixels is what we are
        # avoiding becoming, not something to copy.
        if all(str(unit) == DEFAULT_UNIT for unit in tuple(units)):
            continue
        matched = units_like(layer, ndim)
        if matched:
            return matched
    return {}



@dataclass
class LayerSpec:
    """A ready-to-add napari image layer plus the metadata that produced it.

    ``data`` is either an array-like or, when ``multiscale`` is set, a list of
    array-likes ordered from full resolution downwards. ``axes`` describes the
    axes of ``data`` (channels are always split into separate specs, so ``axes``
    never contains ``C``).
    """

    data: Any
    name: str
    axes: str
    scale: tuple[float, ...]
    metadata: AcquisitionMetadata
    channel_index: int | None = None
    #: Fallback colormap name, used when ``color`` is None.
    colormap: str = "gray"
    #: Channel display colour as RGB in 0-1, when the file specified one. Readers
    #: stay free of napari imports; :func:`napari_colormap` turns this into a real
    #: colormap at the point the layer is added.
    color: tuple[float, float, float] | None = None
    blending: str = "translucent"
    multiscale: bool = False
    contrast_limits: tuple[float, float] | None = None
    units: tuple[str, ...] = ()
    extra_kwargs: dict[str, Any] = field(default_factory=dict)

    @property
    def channel_name(self) -> str:
        if self.channel_index is None:
            return ""
        return self.metadata.channel(self.channel_index).display_name

    def to_kwargs(self) -> dict[str, Any]:
        """Keyword arguments for :meth:`napari.Viewer.add_image`.

        The ``metadata`` dict carries our own objects through to the layer so the
        dock widgets can recover them from ``layer.metadata``.
        """
        kwargs: dict[str, Any] = {
            "name": self.name,
            "scale": tuple(self.scale),
            "colormap": napari_colormap(self.color, self.name) or self.colormap,
            "blending": self.blending,
            "multiscale": self.multiscale,
            "metadata": {
                "mv_metadata": self.metadata,
                "mv_axes": self.axes,
                "mv_channel_index": self.channel_index,
                "mv_channel_name": self.channel_name,
                "mv_units": self.units,
            },
        }
        # The scale bar's unit comes from here on napari 0.8 and later. Passed only
        # when there is one per axis: a partial tuple would be worse than none.
        if self.units and len(self.units) == len(self.scale) and supports_layer_units():
            kwargs["units"] = tuple(self.units)
        if self.contrast_limits is not None:
            low, high = self.contrast_limits
            if high > low:
                kwargs["contrast_limits"] = (float(low), float(high))
        kwargs.update(self.extra_kwargs)
        return kwargs
