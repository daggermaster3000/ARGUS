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


def squeeze_plan(shape: tuple[int, ...], axes: str) -> list[int]:
    """Axis indices worth keeping: singleton ``T``/``Z`` axes give useless sliders.

    The plan is computed once from the full-resolution level and reused for every
    pyramid level, otherwise a level whose Z has collapsed to 1 would end up with
    a different number of dimensions than its parent.
    """
    return [i for i, axis in enumerate(axes) if not (axis in "TZ" and shape[i] == 1)]


def apply_squeeze(array, keep: list[int], ndim: int):
    """Index *array* down to the axes named by :func:`squeeze_plan`."""
    if len(keep) == ndim:
        return array
    return array[tuple(slice(None) if i in keep else 0 for i in range(ndim))]


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
    #: ``image`` or ``labels``. A segmentation stored beside the pixels it came
    #: from is not an image: napari has to be told, or the label values are shown
    #: as grey levels and cannot be picked, hidden or recoloured per object.
    layer_type: str = "image"
    extra_kwargs: dict[str, Any] = field(default_factory=dict)

    @property
    def channel_name(self) -> str:
        if self.channel_index is None:
            return ""
        return self.metadata.channel(self.channel_index).display_name

    def to_kwargs(self) -> dict[str, Any]:
        """Keyword arguments for ``add_image`` — or ``add_labels``, see :attr:`layer_type`.

        The ``metadata`` dict carries our own objects through to the layer so the
        dock widgets can recover them from ``layer.metadata``.
        """
        kwargs: dict[str, Any] = {
            "name": self.name,
            "scale": tuple(self.scale),
            "multiscale": self.multiscale,
            "metadata": {
                "mv_metadata": self.metadata,
                "mv_axes": self.axes,
                "mv_channel_index": self.channel_index,
                "mv_channel_name": self.channel_name,
                "mv_units": self.units,
            },
        }
        # A Labels layer colours itself from the label values and takes neither a
        # colormap nor contrast limits; passing them is a TypeError, not a hint.
        if self.layer_type != "labels":
            kwargs["colormap"] = napari_colormap(self.color, self.name) or self.colormap
            kwargs["blending"] = self.blending
            if self.contrast_limits is not None:
                low, high = self.contrast_limits
                if high > low:
                    kwargs["contrast_limits"] = (float(low), float(high))
        kwargs.update(self.extra_kwargs)
        return kwargs
