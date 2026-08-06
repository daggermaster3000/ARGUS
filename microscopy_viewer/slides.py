"""Build a PowerPoint figure slide from the datasets currently open.

The slide holds one table: a row per sample, a column per channel, and a final
merge column. Each cell contains the rendered image; the header row names the
channels in their own display colour, so the legend and the pictures cannot drift
apart. Those header labels are ordinary PowerPoint text runs, which is the point
— the acquisition names a channel "Alexa 568", and the person making the figure
wants it to say "anti-CD31" instead, either before exporting (the dialog) or
afterwards (in PowerPoint).

Rendering happens here rather than through ``viewer.screenshot`` so that each
channel is isolated cleanly, the projection is the same maximum-intensity one the
measurement panel uses, and nothing depends on the canvas, the camera or on the
window being on screen.

Nothing in this module imports Qt.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from . import intensity as ix
from . import overview as ov
from .utils import get_logger

logger = get_logger("slides")

PRESENTATION_FILTER = "PowerPoint presentation (*.pptx)"

#: Plane modes, reusing the measurement panel's vocabulary so the two panels
#: cannot disagree about what "maximum projection" means.
PLANE_MODES = ix.PLANE_MODES

#: Longest edge, in pixels, of an image embedded in the slide. A 2040² plane is
#: 4 MP per cell; at four channels plus a merge across six samples that is a
#: 120 MB deck that PowerPoint renders down to a few hundred pixels anyway.
DEFAULT_MAX_PIXELS = 900

#: How contrast is decided per channel.
CONTRAST_AS_DISPLAYED = "As displayed"
CONTRAST_AUTO = "Auto per image"
CONTRAST_MODES = (CONTRAST_AS_DISPLAYED, CONTRAST_AUTO)

#: Rows per slide. Four leaves each panel about 1.5 in tall on a 16:9 slide,
#: which is still legible projected; more than that and the images stop being
#: worth looking at, so a folder is split across slides instead of squeezed.
DEFAULT_ROWS_PER_SLIDE = 4
MAX_ROWS_PER_SLIDE = 10

#: Nice round scale-bar lengths in micrometres.
_BAR_STEPS = (0.5, 1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000, 5000)

#: Fraction of the image width a scale bar aims for.
_BAR_FRACTION = 0.2


# ---------------------------------------------------------------------------
# What goes on the slide
# ---------------------------------------------------------------------------


@dataclass
class ChannelView:
    """One channel of one sample, and how it should be drawn."""

    #: Editable legend text. Starts as the channel name from the file; the whole
    #: point is that it becomes the staining or antibody.
    label: str
    #: Display colour as RGB in 0-1, taken from the layer's colormap.
    color: tuple[float, float, float]
    layer_name: str
    data: Any
    axes: str = ""
    contrast_limits: tuple[float, float] | None = None
    #: A napari ``Colormap``; when absent the colour is used as a black-to-colour ramp.
    colormap: Any = None
    current_step: tuple[int, ...] = ()
    channel_index: int | None = None
    #: Whether this channel contributes to the merge. Brightfield defaults to off:
    #: added to a fluorescence merge it washes every colour out to grey.
    in_merge: bool = True
    #: The full pyramid, finest first, when the file has one. Reading a coarse
    #: level costs a quarter of the bytes per step down, which matters when the
    #: source is on a network share.
    levels: list = field(default_factory=list)

    @property
    def key(self) -> str:
        """Groups the same channel across samples into one table column."""
        if self.channel_index is not None:
            return f"{self.channel_index}"
        return self.label.lower()

    @property
    def full_width(self) -> int:
        """Width of the finest level, which is what the pixel size refers to."""
        shape = getattr(self.data, "shape", ())
        return int(shape[-1]) if shape else 0

    def level_for(self, max_pixels: int):
        """The coarsest pyramid level still finer than the exported image.

        The slide holds at most *max_pixels* on the long edge, so reading full
        resolution off a NAS and immediately throwing three quarters of it away is
        pure latency. A level is only accepted while both of its in-plane axes
        stay at or above the target, which keeps the downsampling a shrink.
        """
        if not self.levels or max_pixels <= 0:
            return self.data
        chosen = self.levels[0]
        for level in self.levels:
            shape = tuple(getattr(level, "shape", ()))
            if len(shape) < 2 or min(int(shape[-1]), int(shape[-2])) < max_pixels:
                break
            chosen = level
        return chosen


@dataclass
class SampleSlide:
    """One row of the table: a dataset and its channels."""

    name: str
    channels: list[ChannelView] = field(default_factory=list)
    pixel_size_um: float | None = None
    source: str = ""
    #: Where on the stage this dataset was imaged, ``(x0, x1, y0, y1, z0, z1)`` in
    #: µm, when the file recorded it. Drives the overview slide and nothing else.
    stage_extent: tuple[float, float, float, float, float, float] | None = None


# ---------------------------------------------------------------------------
# Colour
# ---------------------------------------------------------------------------


def colormap_color(colormap, fallback: tuple[float, float, float] = (1.0, 1.0, 1.0)):
    """The colour a colormap maps full intensity to, as RGB in 0-1."""
    if colormap is None:
        return fallback
    try:
        mapped = np.asarray(colormap.map(np.array([1.0], dtype=np.float32)))
        return tuple(float(c) for c in mapped.reshape(-1, mapped.shape[-1])[0][:3])
    except Exception:
        logger.debug("could not read the top colour of %r", colormap, exc_info=True)
        return fallback


def text_color(rgb: Sequence[float]) -> tuple[int, int, int]:
    """A version of *rgb* that stays legible as text on a white slide.

    Grey and yellow channels map to near-white, which is invisible on a light
    background, so anything that bright is darkened until it reads. Hue is kept:
    a green channel's label still comes out green.
    """
    red, green, blue = (float(min(max(c, 0.0), 1.0)) for c in rgb)
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    if luminance <= 1e-6:
        return (0, 0, 0)
    ceiling = 0.55
    if luminance > ceiling:
        factor = ceiling / luminance
        red, green, blue = red * factor, green * factor, blue * factor
    return tuple(int(round(c * 255)) for c in (red, green, blue))


def _ramp(color: Sequence[float], values: np.ndarray) -> np.ndarray:
    """Black-to-*color* ramp, the fallback when there is no napari colormap."""
    tint = np.asarray([float(c) for c in color[:3]], dtype=np.float32)
    return values[..., None] * tint


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _condition(channel: ChannelView, data) -> ix.ConditionSpec:
    """Adapt a channel to the measurement panel's plane extractor."""
    ndim = int(getattr(data, "ndim", 0))
    step = tuple(channel.current_step[:ndim]) if channel.current_step else ()
    step = step + (0,) * (ndim - len(step))
    return ix.ConditionSpec(
        name=channel.label,
        layer_name=channel.layer_name,
        data=data,
        axes=channel.axes,
        scale=(),
        translate=(),
        current_step=step,
        dtype=np.dtype(getattr(data, "dtype", np.float32)),
    )


def auto_limits(plane: np.ndarray) -> tuple[float, float]:
    """Contrast limits from the plane's own percentiles.

    The same 0.5–99.5 rule the Auto Contrast button uses, so a batch export of
    files that were never opened looks like what the viewer would have shown.
    Clipping the extremes keeps hot pixels and camera offset from flattening
    everything into one grey.
    """
    from .contrast import HIGH_PERCENTILE, LOW_PERCENTILE

    finite = plane[np.isfinite(plane)] if plane.dtype.kind == "f" else plane
    if finite.size == 0:
        return 0.0, 0.0
    low, high = (float(v) for v in np.percentile(finite, [LOW_PERCENTILE, HIGH_PERCENTILE]))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low, high = float(np.min(finite)), float(np.max(finite))
    return low, high


def normalized_plane(
    channel: ChannelView,
    mode: str,
    data=None,
    contrast: str = CONTRAST_AS_DISPLAYED,
) -> tuple[np.ndarray, str]:
    """The channel's 2D plane, contrast-stretched into 0-1.

    Clipping happens before any downsampling: shrinking first would let a single
    hot pixel pull a whole neighbourhood up and change the apparent brightness.
    Contrast limits are absolute intensities, so they hold whichever pyramid
    level *data* comes from.
    """
    plane, description = ix.extract_plane(_condition(channel, channel.data if data is None else data), mode)
    plane = np.asarray(plane, dtype=np.float32)

    limits = None if contrast == CONTRAST_AUTO else channel.contrast_limits
    if limits is None:
        # No limits to honour — either the caller asked for auto, or the file
        # never recorded a display range. Percentiles either way, so a single hot
        # pixel does not push the whole image into the black.
        low, high = auto_limits(plane)
    else:
        low, high = float(limits[0]), float(limits[1])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return np.zeros(plane.shape, dtype=np.float32), description

    return np.clip((plane - low) / (high - low), 0.0, 1.0), description


def downsample(plane: np.ndarray, max_pixels: int) -> np.ndarray:
    """Shrink so the longest edge is at most *max_pixels*, keeping the aspect ratio."""
    if max_pixels <= 0:
        return plane
    height, width = plane.shape[:2]
    longest = max(height, width)
    if longest <= max_pixels:
        return plane

    factor = max_pixels / float(longest)
    target = (max(1, int(round(width * factor))), max(1, int(round(height * factor))))
    try:
        from PIL import Image

        return np.asarray(
            Image.fromarray(plane.astype(np.float32), mode="F").resize(target, Image.LANCZOS),
            dtype=np.float32,
        )
    except Exception:
        # Averaging over whole blocks beats plain striding, which drops thin
        # structures outright — exactly the features being looked at.
        logger.debug("no PIL resize available; block-averaging instead", exc_info=True)
        step = int(np.ceil(longest / max_pixels))
        trimmed = plane[: height // step * step, : width // step * step]
        return trimmed.reshape(height // step, step, width // step, step).mean(axis=(1, 3))


def colorize(plane: np.ndarray, channel: ChannelView) -> np.ndarray:
    """Map a normalised plane through the channel's colormap to float RGB."""
    if channel.colormap is not None:
        try:
            mapped = np.asarray(channel.colormap.map(plane.ravel().astype(np.float32)))
            return mapped.reshape(plane.shape + (mapped.shape[-1],))[..., :3].astype(np.float32)
        except Exception:
            logger.debug("colormap failed for %s; using a plain ramp", channel.label, exc_info=True)
    return _ramp(channel.color, plane)


def to_uint8(rgb: np.ndarray) -> np.ndarray:
    return (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)


def render_sample(
    sample: SampleSlide,
    mode: str = "Maximum projection",
    max_pixels: int = DEFAULT_MAX_PIXELS,
    scale_bar: bool = True,
    contrast: str = CONTRAST_AS_DISPLAYED,
) -> tuple[list[np.ndarray], np.ndarray | None, str, str]:
    """Render every channel of a sample plus the merge.

    Returns ``(channel images, merge image, plane description, scale bar label)``.
    Images are ``uint8`` RGB. The merge is an additive composite, which is what
    napari itself does for multichannel data, so the slide matches the screen.
    """
    images: list[np.ndarray] = []
    accumulator: np.ndarray | None = None
    description = ""
    original_width = 0
    rendered_width = 0

    for channel in sample.channels:
        plane, description = normalized_plane(channel, mode, channel.level_for(max_pixels), contrast)
        # The pixel size describes the finest level, so the scale bar has to be
        # measured against that even when a coarser one was read.
        original_width = max(original_width, channel.full_width or int(plane.shape[1]))
        small = downsample(plane, max_pixels)
        rendered_width = max(rendered_width, int(small.shape[1]))
        rgb = colorize(small, channel)
        images.append(to_uint8(rgb))
        if channel.in_merge:
            accumulator = rgb.copy() if accumulator is None else accumulator + rgb

    merge = to_uint8(accumulator) if accumulator is not None else None

    label = ""
    if scale_bar and sample.pixel_size_um and rendered_width:
        shrink = original_width / float(rendered_width) if rendered_width else 1.0
        effective = float(sample.pixel_size_um) * shrink
        for image in images:
            label = draw_scale_bar(image, effective)
        if merge is not None:
            label = draw_scale_bar(merge, effective)

    return images, merge, description, label


def bar_length(pixel_size_um: float, width_px: int) -> tuple[float, int]:
    """A round scale-bar length: ``(micrometres, pixels)``."""
    ideal = pixel_size_um * width_px * _BAR_FRACTION
    micrometres = min(_BAR_STEPS, key=lambda step: abs(step - ideal))
    return float(micrometres), int(round(micrometres / pixel_size_um))


def draw_scale_bar(image: np.ndarray, pixel_size_um: float) -> str:
    """Burn a scale bar into the bottom-right of *image*. Returns its label.

    The bar is white on dark images and black on bright ones. Fluorescence
    channels are nearly black so white is the convention, but a brightfield panel
    is a pale field with a dark specimen on it, and a white bar there is simply
    not visible — which is how it looked before this chose a colour.

    The length is written into the slide as text rather than into the pixels, so
    it stays selectable and does not depend on a font being available.
    """
    if pixel_size_um <= 0 or image.ndim != 3:
        return ""
    height, width = image.shape[:2]
    micrometres, length = bar_length(pixel_size_um, width)
    length = max(4, min(length, width - 8))
    thickness = max(2, int(round(height * 0.012)))
    margin = max(4, int(round(width * 0.04)))

    top = height - margin - thickness
    left = width - margin - length
    if top < 0 or left < 0:
        return ""

    # Judge the background from a band around the bar, not the bar's own strip,
    # so a thin bright feature underneath does not flip the whole decision.
    band = image[max(0, top - thickness) : min(height, top + 2 * thickness), left : left + length]
    value = 0 if float(np.mean(band)) > 127 else 255
    image[top : top + thickness, left : left + length] = value
    return f"{micrometres:g} µm".replace(".0 ", " ")


# ---------------------------------------------------------------------------
# Collecting what is open
# ---------------------------------------------------------------------------


def _levels(layer) -> list:
    """A layer's pyramid, finest first.

    ``mv_pyramid`` is the untouched level list the 3D renderer stashes before it
    swaps a layer down to a single level, so this is right whether the viewer is
    in 2D or 3D.
    """
    pyramid = layer.metadata.get("mv_pyramid")
    if pyramid:
        return list(pyramid)
    if getattr(layer, "multiscale", False):
        return list(layer.data)
    return [layer.data]


def sample_name(meta, fallback: str) -> str:
    """A short row label.

    Imaris records the acquiring machine's own path as the image name, so a raw
    ``image_name`` reads ``D:\\Transfer\\2026-07-27\\TEB-BIOTIN_4.ims`` — useless
    as a figure label. The file's stem is what the experiment is actually called.
    """
    path = getattr(meta, "file_path", None) if meta is not None else None
    if path:
        return Path(path).stem
    name = str(getattr(meta, "image_name", "") or "") if meta is not None else ""
    if name:
        # image_name may still be a path, from any platform.
        return Path(name.replace("\\", "/")).stem or name
    return fallback


def _current_step(viewer, data) -> tuple[int, ...]:
    """The dimension-slider position, right-aligned onto a layer's own axes."""
    ndim = int(getattr(data, "ndim", 0))
    steps = tuple(int(s) for s in viewer.dims.current_step)
    offset = len(steps) - ndim
    return tuple(steps[axis + offset] if 0 <= axis + offset < len(steps) else 0 for axis in range(ndim))


def collect_samples(viewer, visible_only: bool = False) -> list[SampleSlide]:
    """Group the viewer's image layers into one :class:`SampleSlide` per dataset.

    Layers are grouped by source file, so the per-channel layers a reader creates
    from one ``.ims`` come back together as one row. When a channel appears more
    than once — the measurement panel's projection layers sit alongside their
    sources — the visible one wins, since that is the one being looked at.
    """
    from napari.layers import Image

    from .loaders.layer_spec import is_brightfield

    samples: dict[str, SampleSlide] = {}
    order: list[str] = []

    for layer in viewer.layers:
        if not isinstance(layer, Image):
            continue
        if visible_only and not layer.visible:
            continue
        meta = layer.metadata.get("mv_metadata")
        source = str(getattr(meta, "file_path", "") or "") if meta is not None else ""
        name = sample_name(meta, layer.name)
        key = source or name

        sample = samples.get(key)
        if sample is None:
            pixel = getattr(meta, "pixel_size_x_um", None) if meta is not None else None
            sample = SampleSlide(
                name=name,
                pixel_size_um=pixel,
                source=source,
                stage_extent=getattr(meta, "stage_extent", None) if meta is not None else None,
            )
            samples[key] = sample
            order.append(key)

        levels = _levels(layer)
        data = levels[0]
        channel_name = str(layer.metadata.get("mv_channel_name", "") or "") or layer.name
        color = colormap_color(getattr(layer, "colormap", None))
        view = ChannelView(
            label=channel_name,
            color=color,
            layer_name=layer.name,
            data=data,
            axes=str(layer.metadata.get("mv_axes", "")),
            contrast_limits=tuple(float(v) for v in layer.contrast_limits)
            if layer.contrast_limits is not None
            else None,
            colormap=getattr(layer, "colormap", None),
            current_step=_current_step(viewer, data),
            channel_index=layer.metadata.get("mv_channel_index"),
            in_merge=not is_brightfield(channel_name),
            levels=levels,
        )

        existing = next((c for c in sample.channels if c.key == view.key), None)
        if existing is None:
            sample.channels.append(view)
        elif layer.visible and not viewer.layers[existing.layer_name].visible:
            sample.channels[sample.channels.index(existing)] = view

    return [samples[key] for key in order if samples[key].channels]


def _spec_colormap(spec):
    """A napari colormap for a reader's :class:`LayerSpec`.

    The readers deliberately stay free of napari, so a spec carries either an RGB
    display colour from the file or the name of a colormap. Both have to be turned
    into something with ``map`` before anything can be drawn.
    """
    from .loaders.layer_spec import napari_colormap

    colormap = napari_colormap(spec.color, spec.name)
    if colormap is not None:
        return colormap
    try:
        from napari.utils.colormaps import ensure_colormap

        return ensure_colormap(spec.colormap)
    except Exception:
        logger.debug("could not resolve the colormap %r", spec.colormap, exc_info=True)
        return None


def samples_from_paths(paths: Sequence[str | Path]) -> tuple[list[SampleSlide], list[str]]:
    """Read files straight into slide rows, without adding them to the viewer.

    This is what batch mode runs on. Opening a folder of thirty datasets as layers
    would mean a hundred-odd entries in the layer list and every pyramid held
    open, for images nobody is going to look at interactively — the slide is the
    output. Returns ``(samples, error messages)``; a file that cannot be read is
    reported and skipped rather than aborting the batch.
    """
    from .loaders import expand_inputs, load_paths

    candidates = expand_inputs(list(paths))
    specs, errors = load_paths(candidates)

    samples: dict[str, SampleSlide] = {}
    order: list[str] = []
    for spec in specs:
        meta = spec.metadata
        source = str(getattr(meta, "file_path", "") or "")
        name = sample_name(meta, spec.name)
        key = source or name

        sample = samples.get(key)
        if sample is None:
            sample = SampleSlide(
                name=name,
                pixel_size_um=getattr(meta, "pixel_size_x_um", None),
                source=source,
                stage_extent=getattr(meta, "stage_extent", None),
            )
            samples[key] = sample
            order.append(key)

        levels = list(spec.data) if spec.multiscale else [spec.data]
        channel_name = spec.channel_name or spec.name
        colormap = _spec_colormap(spec)
        sample.channels.append(
            ChannelView(
                label=channel_name,
                color=colormap_color(colormap) if spec.color is None else tuple(spec.color),
                layer_name=spec.name,
                data=levels[0],
                axes=spec.axes,
                contrast_limits=spec.contrast_limits,
                colormap=colormap,
                current_step=(),
                channel_index=spec.channel_index,
                in_merge=not _is_brightfield(channel_name),
                levels=levels,
            )
        )

    return [samples[key] for key in order if samples[key].channels], [str(error) for error in errors]


def _is_brightfield(name: str) -> bool:
    from .loaders.layer_spec import is_brightfield

    return is_brightfield(name)


def channel_columns(samples: Sequence[SampleSlide]) -> list[tuple[str, str, tuple[float, float, float]]]:
    """The table's channel columns as ``(key, label, colour)``, in first-seen order.

    Samples may not share every channel — a control imaged without one stain, say
    — so the columns are the union across samples rather than one sample's list.
    """
    columns: list[tuple[str, str, tuple[float, float, float]]] = []
    seen: set[str] = set()
    for sample in samples:
        for channel in sample.channels:
            if channel.key in seen:
                continue
            seen.add(channel.key)
            columns.append((channel.key, channel.label, channel.color))
    return columns


def apply_labels(samples: Iterable[SampleSlide], labels: dict[str, str]) -> None:
    """Rename channels by column key, so one edit renames every sample's channel."""
    for sample in samples:
        for channel in sample.channels:
            text = labels.get(channel.key)
            if text:
                channel.label = text


def apply_merge_selection(samples: Iterable[SampleSlide], keys: Iterable[str]) -> None:
    """Restrict the merge to the given channel column keys."""
    wanted = set(keys)
    for sample in samples:
        for channel in sample.channels:
            channel.in_merge = channel.key in wanted


# ---------------------------------------------------------------------------
# The overview
# ---------------------------------------------------------------------------


def split_overview(samples: Sequence[SampleSlide]) -> tuple[list[SampleSlide], list[SampleSlide]]:
    """Pull a mosaic overview out of a list of samples.

    Returns ``(samples, tiles)``. The tiles are the ``_F####`` fields of an
    overview acquisition: twenty-five of them make twenty-five useless rows in the
    figure, one per field, all of the same slide at low magnification. They belong
    on their own slide, stitched, which is what :func:`build_mosaic` does with
    them.
    """
    return ov.split_tiles(list(samples))


def _tile_plane(sample: SampleSlide, mode: str, max_pixels: int) -> np.ndarray:
    """The plane of one overview field, at no more detail than the mosaic needs."""
    channel = sample.channels[0]
    plane, _description = ix.extract_plane(
        _condition(channel, channel.level_for(max_pixels)), mode
    )
    return np.asarray(plane, dtype=np.float32)


def build_mosaic(
    tiles: Sequence[SampleSlide],
    samples: Sequence[SampleSlide] = (),
    mode: str = "Maximum projection",
    max_pixels: int = ov.DEFAULT_MAX_PIXELS,
) -> ov.Mosaic | None:
    """Stitch the overview fields, wide enough to hold every sample's footprint.

    *samples* only contributes its stage coordinates: a field imaged past the edge
    of the overview would otherwise have to be drawn on the border, which is a
    lie. The canvas is extended instead and the uncovered part left white.
    """
    boxes = [(sample, ov.box_from_extent(sample.stage_extent)) for sample in tiles]
    usable = [(sample, box) for sample, box in boxes if box is not None and sample.channels]
    if not usable:
        return None

    footprints = [
        box
        for box in (ov.box_from_extent(sample.stage_extent) for sample in samples)
        if box is not None
    ]
    bounds = ov.union([box for _sample, box in usable] + footprints)
    if bounds is None or bounds.width <= 0:
        return None

    prepared = []
    for sample, box in usable:
        # Each field only occupies its own share of the mosaic, so reading it at
        # full resolution is throwing away nine tenths of the bytes — which are
        # coming off a network share one field at a time.
        budget = max(64, int(round(max_pixels * box.width / bounds.width)))
        prepared.append(
            ov.Tile(
                box=box,
                read=lambda sample=sample, budget=budget: _tile_plane(sample, mode, budget),
                name=sample.name,
            )
        )

    return ov.stitch(prepared, max_pixels=max_pixels, include=footprints)


def default_stem() -> str:
    return f"figure_slide_{_dt.datetime.now():%Y%m%d_%H%M%S}"


# ---------------------------------------------------------------------------
# The PowerPoint file
# ---------------------------------------------------------------------------

EMU_PER_INCH = 914400

#: 16:9, the shape of every projector and screen this ends up on.
SLIDE_WIDTH_IN = 13.333
SLIDE_HEIGHT_IN = 7.5

_MARGIN_IN = 0.4
_TITLE_HEIGHT_IN = 0.45
_NAME_COLUMN_IN = 1.95
_HEADER_HEIGHT_IN = 0.35
_CELL_INSET_IN = 0.04
_MIN_ROW_IN = 0.45

#: Width of the numbered legend beside the overview.
_LEGEND_COLUMN_IN = 3.1
#: Smallest a marker is drawn. A 1.2 mm field on a 30 mm overview is 4% of the
#: picture; below about a tenth of an inch the outline stops being findable.
_MIN_MARKER_IN = 0.11
#: Marker red. Dark enough to read on the pale background of a brightfield
#: overview, and nothing in a fluorescence panel is this colour.
_MARKER_RGB = (0xC0, 0x20, 0x20)


def _png_bytes(image: np.ndarray):
    """Encode an RGB array as PNG in memory, so no temporary files are needed."""
    import io

    buffer = io.BytesIO()
    try:
        from PIL import Image

        Image.fromarray(image).save(buffer, format="PNG")
    except Exception:
        import imageio.v3 as iio

        buffer = io.BytesIO(iio.imwrite("<bytes>", image, extension=".png"))
    buffer.seek(0)
    return buffer


def _blank_layout(presentation):
    """The template's blank layout, falling back to whatever exists."""
    layouts = presentation.slide_layouts
    for index in (6, 5, len(layouts) - 1):
        try:
            return layouts[index]
        except IndexError:
            continue
    return layouts[0]


def _style_table(table) -> None:
    """Strip PowerPoint's banded blue default, which fights the images."""
    from pptx.dml.color import RGBColor

    table.first_row = False
    table.horz_banding = False
    for row in table.rows:
        for cell in row.cells:
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
            cell.margin_left = cell.margin_right = 0
            cell.margin_top = cell.margin_bottom = 0


def _write_cell(cell, text: str, size: float, bold: bool = False, color=None, align_centre: bool = True):
    from pptx.dml.color import RGBColor
    from pptx.enum.text import PP_ALIGN
    from pptx.util import Pt

    frame = cell.text_frame
    frame.word_wrap = True
    paragraph = frame.paragraphs[0]
    paragraph.alignment = PP_ALIGN.CENTER if align_centre else PP_ALIGN.LEFT
    run = paragraph.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    if color is not None:
        run.font.color.rgb = RGBColor(*color)
    return paragraph


def export_slide(
    samples: Sequence[SampleSlide],
    path: str | Path,
    title: str = "",
    mode: str = "Maximum projection",
    max_pixels: int = DEFAULT_MAX_PIXELS,
    scale_bar: bool = True,
    font_size: float = 11.0,
    contrast: str = CONTRAST_AS_DISPLAYED,
    rows_per_slide: int = DEFAULT_ROWS_PER_SLIDE,
    overview_tiles: Sequence[SampleSlide] = (),
    overview_pixels: int = ov.DEFAULT_MAX_PIXELS,
    progress: Callable[[int, int, str], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> Path:
    """Write a ``.pptx`` figure: samples down the table, channels across it.

    More than *rows_per_slide* samples are split across further slides rather than
    squeezed onto one, and every slide keeps the same columns and row height so the
    deck reads as one figure. Rendering is chunked to match, so a folder of thirty
    datasets never holds more than one slide's worth of images in memory.

    *overview_tiles* are the fields of a mosaic overview, as returned by
    :func:`split_overview`. They are stitched into a single locator slide that
    leads the deck, with a numbered box where each sample was imaged; the numbers
    are the sample's position in the deck, so marker 3 is the third row.

    Raises :class:`ExportCancelled` if *should_cancel* starts returning true; a
    folder of large stacks takes minutes and has to be interruptible.
    """
    from pptx import Presentation
    from pptx.util import Inches

    if not samples:
        raise ValueError("No samples to put on the slide.")

    path = Path(path)
    if path.suffix.lower() != ".pptx":
        path = path.with_suffix(".pptx")
    path.parent.mkdir(parents=True, exist_ok=True)

    # Columns are worked out across every sample, not per slide, so a dataset
    # missing a channel does not shift the grid on its own slide.
    columns = channel_columns(samples)
    if not columns:
        raise ValueError("None of the open datasets have channels to show.")

    per_slide = max(1, min(int(rows_per_slide), MAX_ROWS_PER_SLIDE))
    chunks = [samples[start : start + per_slide] for start in range(0, len(samples), per_slide)]

    presentation = Presentation()
    presentation.slide_width = Inches(SLIDE_WIDTH_IN)
    presentation.slide_height = Inches(SLIDE_HEIGHT_IN)

    # The overview leads the deck, so it is built first — a reader wants to know
    # where the panels came from before looking at them.
    stitched = False
    if overview_tiles:
        if should_cancel is not None and should_cancel():
            raise ExportCancelled("cancelled before the overview was stitched")
        if progress is not None:
            # A negative index marks work that is not one of the numbered samples.
            progress(-1, len(samples), f"overview ({len(overview_tiles)} fields)")
        mosaic = build_mosaic(overview_tiles, samples, mode, overview_pixels)
        if mosaic is None:
            logger.warning("the overview fields could not be stitched; skipping that slide")
        else:
            _build_overview_slide(
                presentation, mosaic, ov.markers(samples, mosaic.covered), title, font_size
            )
            stitched = True

    done = 0
    warnings: list[str] = []
    for number, chunk in enumerate(chunks, start=1):
        # Render a slide's worth at a time. This is the slow part: every channel
        # is a fresh read from wherever the data lives.
        rendered = []
        for sample in chunk:
            if should_cancel is not None and should_cancel():
                raise ExportCancelled(f"cancelled after {done} of {len(samples)} samples")
            if progress is not None:
                progress(done, len(samples), sample.name)
            images, merge, description, bar = render_sample(
                sample, mode, max_pixels, scale_bar, contrast
            )
            by_key = {channel.key: image for channel, image in zip(sample.channels, images)}
            rendered.append((sample, by_key, merge, description, bar))
            done += 1

        heading = title or f"{mode} — {_dt.date.today():%d %b %Y}"
        if len(chunks) > 1:
            heading = f"{heading} ({number} of {len(chunks)})"
        warnings.extend(
            _build_slide(presentation, rendered, columns, heading, font_size, per_slide)
        )

    presentation.save(str(path))
    logger.info(
        "wrote %d slide(s), %d sample(s) x %d channel(s) to %s%s",
        len(chunks) + (1 if stitched else 0), len(samples), len(columns), path,
        f" ({len(warnings)} gap(s))" if warnings else "",
    )
    return path


class ExportCancelled(Exception):
    """The caller asked for the export to stop part-way through."""


def _build_slide(
    presentation,
    rendered: list,
    columns: list,
    heading: str,
    font_size: float,
    rows_per_slide: int,
) -> list[str]:
    """Lay one slide out. Returns notes about channels a sample did not have.

    Every picture is a separate shape sitting over its table cell. PowerPoint has
    no notion of an image *inside* a table cell, so the table supplies the grid
    and the labels while the pictures are positioned on top of it. Both stay
    fully editable: the table can be restyled and the pictures moved or resized.
    """
    from pptx.util import Inches, Pt

    slide = presentation.slides.add_slide(_blank_layout(presentation))

    # -- title ---------------------------------------------------------------
    box = slide.shapes.add_textbox(
        Inches(_MARGIN_IN), Inches(0.22),
        Inches(SLIDE_WIDTH_IN - 2 * _MARGIN_IN), Inches(_TITLE_HEIGHT_IN),
    )
    run = box.text_frame.paragraphs[0].add_run()
    run.text = heading
    run.font.size = Pt(18)
    run.font.bold = True

    # -- geometry ------------------------------------------------------------
    table_left = _MARGIN_IN
    table_top = 0.22 + _TITLE_HEIGHT_IN + 0.08
    table_width = SLIDE_WIDTH_IN - 2 * _MARGIN_IN
    image_column = (table_width - _NAME_COLUMN_IN) / (len(columns) + 1)

    available = SLIDE_HEIGHT_IN - table_top - _HEADER_HEIGHT_IN - _MARGIN_IN
    # Divided by the configured row count rather than this slide's, so a final
    # slide holding one leftover sample matches the full ones instead of blowing
    # that sample up to fill the page.
    per_row = max(_MIN_ROW_IN, available / max(1, rows_per_slide))

    row_heights = []
    for _sample, by_key, merge, _description, _bar in rendered:
        sample_images = list(by_key.values()) + ([merge] if merge is not None else [])
        aspect = _aspect(sample_images)
        natural = (image_column - 2 * _CELL_INSET_IN) / aspect + 2 * _CELL_INSET_IN
        row_heights.append(max(_MIN_ROW_IN, min(per_row, natural)))

    table_height = _HEADER_HEIGHT_IN + sum(row_heights)
    shape = slide.shapes.add_table(
        len(rendered) + 1, len(columns) + 2,
        Inches(table_left), Inches(table_top), Inches(table_width), Inches(table_height),
    )
    table = shape.table

    column_widths = [_NAME_COLUMN_IN] + [image_column] * (len(columns) + 1)
    for index, width in enumerate(column_widths):
        table.columns[index].width = Inches(width)
    table.rows[0].height = Inches(_HEADER_HEIGHT_IN)
    for index, height in enumerate(row_heights, start=1):
        table.rows[index].height = Inches(height)
    _style_table(table)

    # -- header: the legend, in each channel's own colour ---------------------
    _write_cell(table.cell(0, 0), "Sample", font_size, bold=True, align_centre=False)
    for index, (_key, label, color) in enumerate(columns, start=1):
        _write_cell(table.cell(0, index), label, font_size, bold=True, color=text_color(color))
    _write_cell(table.cell(0, len(columns) + 1), "Merge", font_size, bold=True)

    # -- rows -----------------------------------------------------------------
    warnings: list[str] = []
    y = table_top + _HEADER_HEIGHT_IN
    for row_index, (sample, by_key, merge, description, bar) in enumerate(rendered, start=1):
        height = row_heights[row_index - 1]

        caption = sample.name
        notes = [note for note in (description, f"scale bar {bar}" if bar else "") if note]
        if notes:
            caption = f"{caption}\n{' · '.join(notes)}"
        _write_cell(table.cell(row_index, 0), caption, font_size, align_centre=False)

        x = table_left + _NAME_COLUMN_IN
        for column_index, (key, _label, _color) in enumerate(columns, start=1):
            image = by_key.get(key)
            if image is None:
                _write_cell(table.cell(row_index, column_index), "—", font_size)
                warnings.append(f"{sample.name} has no {_label!r} channel.")
            else:
                _place(slide, image, x, y, image_column, height)
            x += image_column

        if merge is not None:
            _place(slide, merge, x, y, image_column, height)
        else:
            # No channel is in the merge — a brightfield-only dataset, typically.
            # Marked, not left blank: an empty cell reads as a failed render.
            _write_cell(table.cell(row_index, len(columns) + 1), "—", font_size)
        y += height

    return warnings


def _legend_size(
    lines: Sequence[tuple[str, bool]], width_in: float, height_in: float, preferred: float
) -> float:
    """The largest font size at which the legend still fits its column.

    Text width is estimated at half the point size per character, which is about
    right for the sans-serif PowerPoint defaults to and errs on the wide side —
    the failure that matters is a legend running off the bottom of the slide.
    """
    for size in [preferred] + [value / 2 for value in range(int(preferred * 2) - 1, 11, -1)]:
        characters = max(8, int(width_in * 72 / (size * 0.5)))
        rows = sum(max(1, -(-len(text) // characters)) for text, _bold in lines)
        if rows * (size * 1.35 / 72) <= height_in:
            return float(size)
    return 6.0


def _build_overview_slide(
    presentation,
    mosaic: ov.Mosaic,
    markers: Sequence[ov.Marker],
    title: str,
    font_size: float,
) -> None:
    """Lay out the locator slide: the stitched overview, marked and numbered.

    The boxes and their numbers are PowerPoint shapes rather than pixels burned
    into the picture, for the same reason the channel headings are real text: a
    marker that lands on top of the thing it is pointing at has to be draggable,
    and the numbers have to stay sharp when the slide is projected.
    """
    from pptx.dml.color import RGBColor
    from pptx.enum.shapes import MSO_SHAPE
    from pptx.enum.text import PP_ALIGN
    from pptx.util import Inches, Pt

    slide = presentation.slides.add_slide(_blank_layout(presentation))

    stitched = ov.Mosaic(
        image=mosaic.image.copy(), box=mosaic.box, um_per_px=mosaic.um_per_px,
        tiles=mosaic.tiles, covered=mosaic.covered,
    )
    image = stitched.image
    # Inside the covered area, not the bottom right of the canvas: a sample imaged
    # off the edge of the overview widens the canvas, and a bar out there would be
    # drawn on blank paper.
    bar = draw_scale_bar(stitched.view_of(stitched.covered), mosaic.um_per_px)

    heading = f"{title} — overview" if title else "Overview"
    box = slide.shapes.add_textbox(
        Inches(_MARGIN_IN), Inches(0.22),
        Inches(SLIDE_WIDTH_IN - 2 * _MARGIN_IN), Inches(_TITLE_HEIGHT_IN),
    )
    run = box.text_frame.paragraphs[0].add_run()
    run.text = heading
    run.font.size = Pt(18)
    run.font.bold = True

    # -- geometry --------------------------------------------------------------
    top = 0.22 + _TITLE_HEIGHT_IN + 0.08
    height = SLIDE_HEIGHT_IN - top - _MARGIN_IN
    width = SLIDE_WIDTH_IN - 2 * _MARGIN_IN - _LEGEND_COLUMN_IN - 0.2

    draw_width = width
    draw_height = draw_width / mosaic.aspect
    if draw_height > height:
        draw_height = height
        draw_width = draw_height * mosaic.aspect
    left = _MARGIN_IN + (width - draw_width) / 2
    picture_top = top + (height - draw_height) / 2

    slide.shapes.add_picture(
        _png_bytes(image), Inches(left), Inches(picture_top), Inches(draw_width), Inches(draw_height)
    )

    # -- markers ---------------------------------------------------------------
    for marker in markers:
        fraction_left, fraction_top, fraction_width, fraction_height = mosaic.fractions_of(marker.box)
        marker_width = max(_MIN_MARKER_IN, fraction_width * draw_width)
        marker_height = max(_MIN_MARKER_IN, fraction_height * draw_height)
        # Grown to the minimum, a marker has to grow about its centre, or it stops
        # pointing at the place it is marking.
        marker_left = left + fraction_left * draw_width - (marker_width - fraction_width * draw_width) / 2
        marker_top = picture_top + fraction_top * draw_height - (marker_height - fraction_height * draw_height) / 2

        outline = slide.shapes.add_shape(
            MSO_SHAPE.RECTANGLE,
            Inches(marker_left), Inches(marker_top), Inches(marker_width), Inches(marker_height),
        )
        outline.fill.background()
        outline.line.color.rgb = RGBColor(*_MARKER_RGB)
        outline.line.width = Pt(1.25)
        outline.shadow.inherit = False

        # The number sits beside the box, not in it: inside a tenth-of-an-inch
        # square it would be unreadable, and it would hide the specimen.
        label_width = 0.4
        to_the_right = marker_left + marker_width + label_width < left + draw_width
        label_left = marker_left + marker_width + 0.02 if to_the_right else marker_left - label_width - 0.02
        label = slide.shapes.add_textbox(
            Inches(label_left), Inches(marker_top + marker_height / 2 - 0.09),
            Inches(label_width), Inches(0.18),
        )
        frame = label.text_frame
        frame.word_wrap = False
        frame.margin_left = frame.margin_right = frame.margin_top = frame.margin_bottom = 0
        paragraph = frame.paragraphs[0]
        paragraph.alignment = PP_ALIGN.LEFT if to_the_right else PP_ALIGN.RIGHT
        number = paragraph.add_run()
        number.text = str(marker.number)
        number.font.size = Pt(max(7.0, font_size - 2))
        number.font.bold = True
        number.font.color.rgb = RGBColor(*_MARKER_RGB)

    # -- legend ----------------------------------------------------------------
    legend_left = SLIDE_WIDTH_IN - _MARGIN_IN - _LEGEND_COLUMN_IN
    legend = slide.shapes.add_textbox(
        Inches(legend_left), Inches(top), Inches(_LEGEND_COLUMN_IN), Inches(height)
    )
    frame = legend.text_frame
    frame.word_wrap = True

    lines: list[tuple[str, bool]] = [(f"{mosaic.tiles} overview fields", True)]
    for marker in markers:
        # The stage coordinates in millimetres: the box says where on the picture,
        # this says where on the microscope, which is what gets typed back in to
        # find the specimen again.
        x_mm, y_mm = (value / 1000.0 for value in marker.box.centre)
        suffix = " — outside the overview" if marker.outside else ""
        lines.append((f"{marker.number}  {marker.name}   X {x_mm:.2f}, Y {y_mm:.2f} mm{suffix}", False))
    notes = [note for note in (f"scale bar {bar}" if bar else "", f"{mosaic.um_per_px:.0f} µm/px") if note]
    if notes:
        lines.append((" · ".join(notes), False))
    if not markers:
        lines.append(("No sample recorded a stage position.", False))

    # Shrink to fit rather than run off the slide: a folder of thirty datasets is
    # thirty legend entries, and the alternative is a second overview slide whose
    # picture would be identical. Entries wrap, so the count of lines is not the
    # count of entries — at a smaller size fewer of them wrap, which is why this
    # tries sizes rather than solving for one.
    size = _legend_size(lines, _LEGEND_COLUMN_IN, height, font_size)
    for index, (text, bold) in enumerate(lines):
        paragraph = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        run = paragraph.add_run()
        run.text = text
        run.font.size = Pt(size)
        run.font.bold = bold
        if index and not bold and index <= len(markers):
            run.font.color.rgb = RGBColor(*_MARKER_RGB)

    logger.info("overview slide: %d field(s), %d marker(s)", mosaic.tiles, len(markers))


def _aspect(images: Sequence[np.ndarray]) -> float:
    """Width/height of the first usable image, defaulting to square."""
    for image in images:
        if image is not None and image.ndim >= 2 and image.shape[0]:
            return float(image.shape[1]) / float(image.shape[0])
    return 1.0


def _place(slide, image: np.ndarray, left: float, top: float, width: float, height: float) -> None:
    """Drop a picture into a table cell, centred and fitted inside it."""
    from pptx.util import Inches

    box_width = width - 2 * _CELL_INSET_IN
    box_height = height - 2 * _CELL_INSET_IN
    aspect = _aspect([image])

    draw_width = box_width
    draw_height = draw_width / aspect
    if draw_height > box_height:
        draw_height = box_height
        draw_width = draw_height * aspect

    slide.shapes.add_picture(
        _png_bytes(image),
        Inches(left + (width - draw_width) / 2),
        Inches(top + (height - draw_height) / 2),
        Inches(draw_width),
        Inches(draw_height),
    )
