"""A movie of a z-stack being acquired: the projection building up, channel by channel.

What a confocal does, played back: the first channel is swept through the stack
from the first plane to the last, and its maximum-intensity projection fills in
as it goes; then the next channel; then the merge is held. Every selected
channel has its own panel beside a merge panel that gains each channel as it is
acquired, so the movie shows both what each stain looks like and how they add up.

Frames are rendered straight from the data — the layer's colour, and either the
contrast it is displayed at or one fitted to the finished projection — not from
screenshots, so the viewer can be doing anything while it runs. They are read
from the coarsest pyramid level that still fills the panel: a whole-brain stack
at full resolution is gigabytes, and every pixel of it would be thrown away.

No Qt here; the panel is :mod:`microscopy_viewer.widgets.acquisition_widget`.
Encoding is :func:`microscopy_viewer.movie.write_movie`, shared with the
time-series export.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Sequence

import numpy as np

from . import slides as sl
from .projection import z_axis_of
from .utils import get_logger

logger = get_logger("acquisition")

#: Layouts of a frame.
LAYOUT_GRID = "Channels + merge"
LAYOUT_MERGE = "Merge only"
LAYOUTS = (LAYOUT_GRID, LAYOUT_MERGE)

#: How each channel's intensities are mapped to the display range.
CONTRAST_AUTO = "Fit to the projection"
CONTRAST_DISPLAYED = "As displayed"
CONTRASTS = (CONTRAST_AUTO, CONTRAST_DISPLAYED)

#: Space between panels, and the frame's background.
GAP = 8
BACKGROUND = 0


class AcquisitionError(RuntimeError):
    """A movie that cannot be made, with a message worth showing."""


class Cancelled(Exception):
    """Raised inside the rendering when the user asks it to stop."""


@dataclass
class AcquisitionSpec:
    """Everything a movie needs: which channels, and how it should look and play."""

    channels: list  # of slides.ChannelView, in acquisition order
    layout: str = LAYOUT_GRID
    contrast: str = CONTRAST_AUTO
    #: How long each channel's sweep through the stack lasts.
    seconds_per_channel: float = 4.0
    #: How long the finished merge stays on screen at the end.
    hold_s: float = 2.0
    fps: float = 30.0
    #: Longest edge of one panel, in pixels.
    panel_pixels: int = 720
    labels: bool = True
    scale_bar: bool = True
    #: Pixel size of the full-resolution image, for the scale bar.
    pixel_size_um: float | None = None
    #: Z spacing of the full-resolution image, for the depth label.
    z_step_um: float | None = None
    #: Sweep from the last plane to the first.
    reverse: bool = False
    quality: int = 8

    def frames_per_channel(self) -> int:
        return max(1, int(round(self.seconds_per_channel * self.fps)))

    def hold_frames(self) -> int:
        return max(0, int(round(self.hold_s * self.fps)))

    def total_frames(self) -> int:
        return self.frames_per_channel() * len(self.channels) + self.hold_frames()


# ---------------------------------------------------------------------------
# Reading the stack
# ---------------------------------------------------------------------------


@dataclass
class Stack:
    """One channel's planes at the level being read, and how to index them."""

    data: object
    z_axis: int | None
    #: Index of every axis but Z, Y and X: the displayed timepoint, typically.
    fixed: tuple = ()
    #: How many full-resolution planes one plane of this level stands for.
    z_factor: float = 1.0

    @property
    def depth(self) -> int:
        return 1 if self.z_axis is None else int(self.data.shape[self.z_axis])

    def plane(self, z: int) -> np.ndarray:
        index = list(self.fixed)
        if self.z_axis is not None:
            index[self.z_axis] = int(z)
        return np.asarray(self.data[tuple(index)], dtype=np.float32)


def stack_of(channel, panel_pixels: int) -> Stack:
    """The planes of *channel* to sweep through, at a level that fills a panel."""
    data = channel.level_for(panel_pixels)
    ndim = int(getattr(data, "ndim", 0))
    if ndim < 2:
        raise AcquisitionError(f"{channel.label} is not an image.")
    z_axis = z_axis_of(channel.axes, ndim) if ndim >= 3 else None
    step = tuple(channel.current_step) if channel.current_step else ()
    step = step + (0,) * (ndim - len(step))
    fixed = []
    for axis in range(ndim):
        if axis >= ndim - 2 or axis == z_axis:
            fixed.append(slice(None))
        else:
            fixed.append(max(0, min(int(step[axis]), int(data.shape[axis]) - 1)))
    full = getattr(channel.data, "shape", ())
    factor = 1.0
    if z_axis is not None and len(full) == ndim and int(data.shape[z_axis]):
        factor = float(full[z_axis]) / float(data.shape[z_axis])
    return Stack(data=data, z_axis=z_axis, fixed=tuple(fixed), z_factor=factor)


def sweep(stack: Stack, frames: int, panel_pixels: int, reverse: bool = False,
          should_cancel: Callable[[], bool] | None = None,
          on_plane: Callable[[int, int], None] | None = None) -> tuple[list[np.ndarray], list[int]]:
    """The running projection after each of *frames* steps, and the plane each reached.

    Every plane is read exactly once. A stack deeper than the number of frames
    advances several planes per frame; a shallower one repeats frames, so each
    channel takes the same time whatever its depth.
    """
    depth = stack.depth
    order = list(range(depth))[::-1] if reverse else list(range(depth))
    # The last plane each frame has reached, 1-based.
    reached = [max(1, math.ceil((k + 1) * depth / frames)) for k in range(frames)]
    snapshots: list[np.ndarray] = []
    running = None
    done = 0
    for count in reached:
        while done < count:
            if should_cancel is not None and should_cancel():
                raise Cancelled()
            plane = sl.downsample(stack.plane(order[done]), panel_pixels)
            running = plane if running is None else np.maximum(running, plane)
            done += 1
            if on_plane is not None:
                on_plane(done, depth)
        snapshots.append(running)
    return snapshots, [order[c - 1] for c in reached]


# ---------------------------------------------------------------------------
# Drawing a frame
# ---------------------------------------------------------------------------


def limits_for(channel, projection: np.ndarray, contrast: str) -> tuple[float, float]:
    """Display range of *channel*: fitted to its finished projection, or as displayed."""
    if contrast == CONTRAST_DISPLAYED and channel.contrast_limits is not None:
        return float(channel.contrast_limits[0]), float(channel.contrast_limits[1])
    return sl.auto_limits(projection)


def colour(plane: np.ndarray, channel, limits: tuple[float, float]) -> np.ndarray:
    """*plane* stretched to *limits* and coloured like its layer, float RGB 0-1."""
    low, high = limits
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        normalised = np.zeros(plane.shape, dtype=np.float32)
    else:
        normalised = np.clip((plane - low) / (high - low), 0.0, 1.0)
    return sl.colorize(normalised, channel)


def grid_shape(panels: int) -> tuple[int, int]:
    """Rows and columns for *panels*: one row up to three, then two rows."""
    rows = 1 if panels <= 3 else 2
    return rows, math.ceil(panels / rows)


def compose(tiles: Sequence[np.ndarray]) -> np.ndarray:
    """Tiles of one size laid out in :func:`grid_shape`, uint8 RGB."""
    height, width = tiles[0].shape[:2]
    rows, columns = grid_shape(len(tiles))
    canvas = np.full((rows * height + (rows - 1) * GAP, columns * width + (columns - 1) * GAP, 3),
                     BACKGROUND, dtype=np.uint8)
    for index, tile in enumerate(tiles):
        row, column = divmod(index, columns)
        top, left = row * (height + GAP), column * (width + GAP)
        canvas[top:top + height, left:left + width] = tile
    return canvas


def _font(size: int):
    from PIL import ImageFont

    for name in ("arial.ttf", "Arial.ttf", "DejaVuSans.ttf", "Helvetica.ttc"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def annotate(tile: np.ndarray, title: str, rgb: Sequence[float], detail: str = "",
             progress: float | None = None) -> np.ndarray:
    """Name a panel in its channel's colour, with the depth reached and a progress bar."""
    from PIL import Image, ImageDraw

    image = Image.fromarray(tile)
    draw = ImageDraw.Draw(image, "RGBA")
    size = max(12, image.height // 22)
    margin = max(6, size // 2)
    ink = tuple(int(round(255 * max(0.35, float(c)))) for c in rgb[:3]) + (255,)
    y = margin
    for text, fill in ((title, ink), (detail, (235, 235, 235, 255))):
        if not text:
            continue
        font = _font(size if fill is ink else int(size * 0.8))
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        pad = max(3, size // 5)
        draw.rectangle((margin - pad, y - pad, margin + right - left + pad, y + bottom - top + pad),
                       fill=(0, 0, 0, 150))
        draw.text((margin - left, y - top), text, font=font, fill=fill)
        y += bottom - top + 2 * pad + 2
    if progress is not None:
        bar = max(3, image.height // 90)
        width = int(round(image.width * min(max(progress, 0.0), 1.0)))
        draw.rectangle((0, image.height - bar, image.width, image.height), fill=(255, 255, 255, 40))
        draw.rectangle((0, image.height - bar, width, image.height), fill=ink)
    return np.asarray(image)


def scale_bar(tile: np.ndarray, pixel_size_um: float) -> np.ndarray:
    """The slides' scale bar, with its length written above it.

    A slide carries the length as text beside the image; a movie has nowhere
    else to put it.
    """
    from PIL import Image, ImageDraw

    label = sl.draw_scale_bar(tile, pixel_size_um)
    if not label:
        return tile
    height, width = tile.shape[:2]
    _micrometres, length = sl.bar_length(pixel_size_um, width)
    length = max(4, min(length, width - 8))
    thickness = max(2, int(round(height * 0.012)))
    margin = max(4, int(round(width * 0.04)))
    image = Image.fromarray(tile)
    draw = ImageDraw.Draw(image)
    font = _font(max(10, height // 30))
    left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
    x = width - margin - length // 2 - (right - left) // 2 - left
    y = height - margin - thickness - (bottom - top) - max(3, thickness) - top
    draw.text((x, y), label, font=font, fill=(255, 255, 255))
    return np.asarray(image)


def depth_text(z: int, depth: int, stack: Stack, z_step_um: float | None) -> str:
    """"z 42 / 120 · 63 µm" — the plane reached, in the file's own numbering."""
    plane = int(round(z * stack.z_factor)) + 1
    total = int(round(depth * stack.z_factor))
    text = f"z {plane} / {total}"
    if z_step_um:
        # Depth below the first plane: 0 µm where the acquisition starts.
        microns = (plane - 1) * z_step_um
        text += f" · {microns:.1f} µm" if microns < 100 else f" · {microns:.0f} µm"
    return text


# ---------------------------------------------------------------------------
# The movie
# ---------------------------------------------------------------------------


@dataclass
class _Done:
    """A channel whose sweep has finished: its projection, coloured."""

    channel: object
    rgb: np.ndarray
    in_merge: bool = True


def frames(spec: AcquisitionSpec, on_progress: Callable[[int, int], None] | None = None,
           should_cancel: Callable[[], bool] | None = None,
           on_status: Callable[[str], None] | None = None) -> Iterator[np.ndarray]:
    """Every frame of the movie, in order, as uint8 RGB. Rendered as they are needed.

    A channel's whole stack is read before its first frame: its contrast is
    fitted to the finished projection. *on_status* hears how that read is going.
    """
    if not spec.channels:
        raise AcquisitionError("Choose at least one channel.")
    total = spec.total_frames()
    per_channel = spec.frames_per_channel()
    emitted = 0
    finished: list[_Done] = []
    shape: tuple[int, int] | None = None

    def tick():
        nonlocal emitted
        emitted += 1
        if on_progress is not None:
            on_progress(emitted, total)

    for channel in spec.channels:
        stack = stack_of(channel, spec.panel_pixels)

        def _reading(done, depth, _label=channel.label):
            if on_status is not None and (done == depth or done % 10 == 0):
                on_status(f"Reading {_label}: plane {done} of {depth}")

        snapshots, reached = sweep(stack, per_channel, spec.panel_pixels, spec.reverse,
                                   should_cancel, _reading)
        if shape is None:
            shape = snapshots[-1].shape[:2]
        limits = limits_for(channel, snapshots[-1], spec.contrast)
        for index, (snapshot, z) in enumerate(zip(snapshots, reached)):
            if should_cancel is not None and should_cancel():
                raise Cancelled()
            current = _Done(channel, colour(_fit(snapshot, shape), channel, limits), channel.in_merge)
            detail = depth_text(z, stack.depth, stack, spec.z_step_um)
            yield _frame(spec, shape, finished, current, detail, (index + 1) / len(snapshots))
            tick()
        finished.append(_Done(channel, colour(_fit(snapshots[-1], shape), channel, limits), channel.in_merge))

    last = _frame(spec, shape, finished, None, "", None)
    for _ in range(spec.hold_frames()):
        yield last
        tick()


def _fit(plane: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """*plane* at *shape*: channels read at different levels can differ by a pixel."""
    if plane.shape[:2] == tuple(shape):
        return plane
    from PIL import Image

    return np.asarray(Image.fromarray(plane.astype(np.float32), mode="F")
                      .resize((shape[1], shape[0]), Image.BILINEAR), dtype=np.float32)


def _frame(spec: AcquisitionSpec, shape, finished: list, current, detail: str,
           progress: float | None) -> np.ndarray:
    black = np.zeros(tuple(shape) + (3,), dtype=np.float32)
    shown = {id(d.channel): d for d in finished}
    if current is not None:
        shown[id(current.channel)] = current
    merge = sum((d.rgb for d in shown.values() if d.in_merge), black.copy())
    merge_tile = sl.to_uint8(merge)
    active = current.channel if current is not None else None

    if spec.scale_bar and spec.pixel_size_um:
        full = max(c.full_width for c in spec.channels) or shape[1]
        merge_tile = scale_bar(merge_tile, spec.pixel_size_um * full / float(shape[1]))

    if spec.layout == LAYOUT_MERGE:
        if spec.labels:
            title = active.label if active is not None else "Merge"
            rgb = active.color if active is not None else (1.0, 1.0, 1.0)
            merge_tile = annotate(merge_tile, title, rgb, detail, progress)
        return merge_tile

    tiles = []
    for channel in spec.channels:
        done = shown.get(id(channel))
        tile = sl.to_uint8(done.rgb if done is not None else black)
        if spec.labels:
            is_active = channel is active
            tile = annotate(tile, channel.label, channel.color,
                            detail if is_active else "",
                            progress if is_active else None)
        tiles.append(tile)
    if spec.labels:
        merge_tile = annotate(merge_tile, "Merge", (1.0, 1.0, 1.0))
    tiles.append(merge_tile)
    return compose(tiles)


def preview(spec: AcquisitionSpec, should_cancel: Callable[[], bool] | None = None) -> np.ndarray:
    """The finished merge frame, without sweeping: one projection per channel."""
    quick = AcquisitionSpec(**{**spec.__dict__, "seconds_per_channel": 1.0, "fps": 1.0,
                               "hold_s": 1.0})
    last = None
    for last in frames(quick, should_cancel=should_cancel):
        pass
    return last


def export(spec: AcquisitionSpec, path: str | Path,
           on_progress: Callable[[int, int], None] | None = None,
           should_cancel: Callable[[], bool] | None = None,
           on_status: Callable[[str], None] | None = None) -> Path:
    """Render and encode the movie to *path* (.mp4, .mov or .gif)."""
    from .movie import MovieCancelled, write_movie

    def _frames():
        try:
            yield from frames(spec, on_progress, should_cancel, on_status)
        except Cancelled as exc:
            raise MovieCancelled() from exc

    written = write_movie(_frames(), path, fps=spec.fps, quality=spec.quality)
    logger.info("acquisition movie: %d channel(s), %d frame(s) at %g fps -> %s",
                len(spec.channels), spec.total_frames(), spec.fps, written)
    return written
