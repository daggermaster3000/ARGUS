"""Write a time series out as a movie for slides.

The frames are canvas screenshots, the same thing **Export Snapshot** saves, so
what lands in the file is what was on screen: the contrast you set, the layers
you had visible, the ROIs, the scale bar, and a 3D view if that is what you were
looking at. Nothing is re-rendered from the data with different settings.

The default container is QuickTime ``.mov`` with H.264 video, which is what
PowerPoint wants for a movie that plays inside a slide rather than opening in an
external player. ``.mp4`` writes the same video stream in the other container —
the safest option on old PowerPoint versions — and ``.gif`` is there for a
loop that has to survive being pasted into a chat window.

Only :func:`capture_frames` touches the GUI. The rest — frame selection, the
timestamp overlay, encoding — is ordinary array work and is tested without a
display.
"""

from __future__ import annotations

import datetime as _dt
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator, Sequence

import numpy as np

from .utils import get_logger

logger = get_logger("movie")

MOVIE_FILTER = ";;".join(
    [
        "QuickTime movie (*.mov)",
        "MP4 video (*.mp4)",
        "Animated GIF (*.gif)",
    ]
)

#: Suffixes that go through ffmpeg rather than the GIF writer.
VIDEO_SUFFIXES = (".mov", ".mp4", ".m4v", ".avi", ".mkv", ".webm")

#: What to tell the user when the encoder is missing, in the same spirit as the
#: slide export's message about python-pptx.
INSTALL_HINT = "python -m pip install imageio-ffmpeg"

#: Longest a single frame is waited on before it is captured anyway. napari 0.5+
#: loads slices asynchronously, so setting the slider returns before the pixels
#: exist and an unguarded screenshot would catch the previous timepoint.
FRAME_TIMEOUT_S = 20.0


class MovieExportError(RuntimeError):
    """Raised when a movie cannot be written, with a message worth showing."""


class MovieCancelled(Exception):
    """Raised inside the export when the user asks it to stop."""


def default_stem(prefix: str = "movie") -> str:
    return f"{prefix}_{_dt.datetime.now():%Y%m%d_%H%M%S}"


# ---------------------------------------------------------------------------
# What gets written
# ---------------------------------------------------------------------------


@dataclass
class MovieSpec:
    """Everything the export needs that is not the viewer itself."""

    path: Path
    #: Playback rate of the written file. Independent of the panel's own rate.
    fps: float = 10.0
    #: Inclusive range of timepoints, and the stride through it.
    start: int = 0
    stop: int = 0
    step: int = 1
    #: Canvas oversampling, as for snapshots: 2 keeps a slide sharp.
    scale: int = 2
    #: 0-10, passed to the encoder. Higher is a bigger file.
    quality: int = 8
    #: Burn the acquisition time into the corner of each frame.
    timestamp: bool = False
    #: Seconds between timepoints, for that stamp.
    interval_s: float | None = None
    #: Prefix put in front of the stamp, for a movie that names its sample.
    label: str = ""

    @property
    def indices(self) -> list[int]:
        return frame_indices(self.start, self.stop, self.step)

    @property
    def duration_s(self) -> float:
        return len(self.indices) / self.fps if self.fps > 0 else 0.0


def frame_indices(start: int, stop: int, step: int = 1) -> list[int]:
    """The timepoints a movie covers: *stop* is inclusive, *step* at least 1.

    Inclusive because the range comes from two spin boxes showing timepoint
    numbers, and a user who types the last timepoint means to include it.
    """
    start = max(0, int(start))
    stop = max(0, int(stop))
    if stop < start:
        start, stop = stop, start
    return list(range(start, stop + 1, max(1, int(step))))


def encoder_available() -> tuple[bool, str]:
    """``(ok, message)`` for the ffmpeg encoder, without raising."""
    try:
        import imageio_ffmpeg

        exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        return (False, f"Writing .mov and .mp4 needs imageio-ffmpeg ({INSTALL_HINT}). {exc}")
    return (True, f"ffmpeg: {exe}")


# ---------------------------------------------------------------------------
# Frames
# ---------------------------------------------------------------------------


def to_rgb(frame: np.ndarray) -> np.ndarray:
    """Drop an alpha channel and make sure the result is 8-bit RGB.

    Canvas screenshots come back RGBA; H.264 has no alpha, and compositing the
    canvas onto white would change what the user was looking at, so the alpha is
    simply discarded — napari's canvas is opaque anyway.
    """
    frame = np.asarray(frame)
    if frame.ndim == 2:
        frame = np.repeat(frame[:, :, None], 3, axis=2)
    if frame.shape[-1] == 4:
        frame = frame[..., :3]
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(frame)


def even_dimensions(frame: np.ndarray) -> np.ndarray:
    """Trim to an even width and height.

    H.264 in the 4:2:0 pixel format every player understands cannot encode an
    odd-sized frame. Cropping a row loses a pixel; letting the encoder rescale
    the whole frame to a multiple of 16 — which is imageio's default — quietly
    resamples the image and invalidates the scale bar, so cropping is right.
    """
    height, width = frame.shape[:2]
    return frame[: height - (height % 2), : width - (width % 2)]


def stamp_text(index: int, interval_s: float | None, label: str = "") -> str:
    """The overlay text for one timepoint."""
    from .timeseries import elapsed_seconds, format_timestamp

    seconds = elapsed_seconds(index, interval_s)
    stamp = format_timestamp(seconds) if seconds is not None else f"t = {int(index)}"
    return f"{label} {stamp}".strip() if label else stamp


def draw_timestamp(frame: np.ndarray, text: str, margin: int = 12) -> np.ndarray:
    """Burn *text* into the top-left corner of *frame*.

    Drawn with a dark plate behind it so it stays readable over a bright field,
    and sized against the frame so it does not vanish on an oversampled export.
    """
    if not text:
        return frame
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:  # pragma: no cover - pillow is a hard dependency
        logger.debug("no pillow; skipping the timestamp overlay", exc_info=True)
        return frame

    image = Image.fromarray(frame)
    size = max(12, int(round(image.height / 28)))
    try:
        font = ImageFont.truetype("arial.ttf", size)
    except Exception:
        font = ImageFont.load_default()

    draw = ImageDraw.Draw(image, "RGBA")
    try:
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    except AttributeError:  # pragma: no cover - pillow < 8
        right, bottom = draw.textsize(text, font=font)
        left = top = 0
    pad = max(3, size // 4)
    draw.rectangle(
        (margin - pad, margin - pad, margin + (right - left) + pad, margin + (bottom - top) + pad),
        fill=(0, 0, 0, 140),
    )
    draw.text((margin - left, margin - top), text, font=font, fill=(255, 255, 255, 255))
    return np.asarray(image)


def prepare_frame(frame: np.ndarray, text: str = "") -> np.ndarray:
    """A raw screenshot turned into something an encoder will accept."""
    return even_dimensions(draw_timestamp(to_rgb(frame), text))


def capture_frames(
    viewer,
    indices: Sequence[int],
    axis: int | None = None,
    scale: int = 2,
    canvas_only: bool = True,
    include_scale_bar: bool = True,
    on_progress: Callable[[int, int], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> Iterator[np.ndarray]:
    """Yield one canvas screenshot per timepoint, in order.

    The viewer is stepped through the range and put back where it started. Each
    frame is waited on before it is grabbed: napari loads slices on a worker
    thread, so a screenshot taken the moment the slider moves shows the *previous*
    timepoint, and a movie made that way is off by one frame throughout.
    """
    from .timeseries import current_index, set_index, viewer_time_axis

    axis = viewer_time_axis(viewer) if axis is None else axis
    if axis is None:
        raise MovieExportError("This dataset has no time axis, so there is nothing to play.")

    oversample = max(1, int(round(float(scale))))
    started_at = current_index(viewer, axis)
    previous_bar = viewer.scale_bar.visible
    if include_scale_bar and canvas_only and not previous_bar:
        viewer.scale_bar.visible = True

    try:
        for position, index in enumerate(indices):
            if should_cancel is not None and should_cancel():
                raise MovieCancelled()
            set_index(viewer, index, axis)
            _settle(viewer)
            frame = viewer.screenshot(canvas_only=canvas_only, flash=False, scale=oversample)
            if on_progress is not None:
                on_progress(position + 1, len(indices))
            yield np.asarray(frame)
    finally:
        viewer.scale_bar.visible = previous_bar
        set_index(viewer, started_at, axis)


def _settle(viewer, timeout: float = FRAME_TIMEOUT_S) -> None:
    """Let Qt repaint and every layer finish loading the slice just requested."""
    try:
        from qtpy.QtWidgets import QApplication
    except Exception:  # pragma: no cover - no Qt (never true in the app)
        return

    app = QApplication.instance()
    deadline = time.monotonic() + timeout
    while True:
        if app is not None:
            app.processEvents()
        if all(bool(getattr(layer, "loaded", True)) for layer in viewer.layers):
            break
        if time.monotonic() > deadline:
            logger.warning("gave up waiting for a slice to load; the frame may be stale")
            break
        time.sleep(0.005)
    if app is not None:
        app.processEvents()


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------


def write_movie(
    frames: Iterable[np.ndarray],
    path: str | Path,
    fps: float = 10.0,
    quality: int = 8,
    stamps: Iterable[str] | None = None,
) -> Path:
    """Encode *frames* to *path*, returning where it was written.

    Frames are consumed lazily, so a long export never holds more than one frame
    plus the encoder's own buffer in memory. All frames must agree on size —
    they do, because they come from one canvas — and the first one sets it.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    labels = iter(stamps) if stamps is not None else None

    def _prepared() -> Iterator[np.ndarray]:
        for frame in frames:
            text = next(labels, "") if labels is not None else ""
            yield prepare_frame(frame, text)

    if suffix == ".gif":
        return _write_gif(_prepared(), path, fps)
    if suffix not in VIDEO_SUFFIXES:
        raise MovieExportError(
            f"{path.suffix or 'That'} is not a movie format this writes. "
            f"Use one of: {', '.join(VIDEO_SUFFIXES)}, .gif"
        )
    return _write_video(_prepared(), path, fps, quality)


def _write_video(frames: Iterator[np.ndarray], path: Path, fps: float, quality: int) -> Path:
    ok, message = encoder_available()
    if not ok:
        raise MovieExportError(message)

    import imageio.v2 as iio

    writer = iio.get_writer(
        str(path),
        fps=max(0.1, float(fps)),
        codec="libx264",
        quality=int(np.clip(quality, 0, 10)),
        # The frames are already even-sized; without this imageio would rescale
        # them to a multiple of 16 and the burnt-in scale bar would be wrong.
        macro_block_size=1,
        # yuv420p is what PowerPoint, QuickTime and every browser can decode.
        pixelformat="yuv420p",
    )
    count = 0
    try:
        try:
            for frame in frames:
                writer.append_data(frame)
                count += 1
        finally:
            writer.close()
    except BaseException:
        # A stopped or failed export must not leave a half-written movie behind
        # for someone to drop into a slide.
        path.unlink(missing_ok=True)
        raise

    if not count:
        path.unlink(missing_ok=True)
        raise MovieExportError("No frames were rendered, so no movie was written.")
    logger.info("wrote %d frame(s) to %s (%.1f MB)", count, path, path.stat().st_size / 1e6)
    return path


def _write_gif(frames: Iterator[np.ndarray], path: Path, fps: float) -> Path:
    """Write an animated GIF with pillow directly.

    Not through imageio: its ``duration`` argument changed units between v2 and
    v3, so a GIF written through it comes out with no frame delay on some
    installs and plays as fast as the viewer can decode it. Pillow's is
    unambiguously milliseconds. Each frame gets its own adaptive palette, which
    a fluorescence image needs — the default web palette posterises it.
    """
    from PIL import Image

    collected = [
        Image.fromarray(frame).convert("P", palette=Image.ADAPTIVE, colors=256)
        for frame in frames
    ]
    if not collected:
        raise MovieExportError("No frames were rendered, so no movie was written.")

    # GIF stores its frame delay in centiseconds, so the rate is rounded to what
    # the format can actually hold rather than silently landing somewhere else.
    milliseconds = max(20, int(round(1000.0 / max(0.1, float(fps)) / 10.0)) * 10)
    collected[0].save(
        str(path),
        save_all=True,
        append_images=collected[1:],
        duration=milliseconds,
        loop=0,  # forever
        disposal=2,
    )
    logger.info("wrote %d frame(s) to %s (%.1f MB)", len(collected), path, path.stat().st_size / 1e6)
    return path


def export_movie(
    viewer,
    spec: MovieSpec,
    on_progress: Callable[[int, int], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> Path:
    """Capture the range described by *spec* and encode it. Returns the path."""
    indices = spec.indices
    if not indices:
        raise MovieExportError("The chosen range holds no timepoints.")

    stamps = (
        [stamp_text(index, spec.interval_s, spec.label) for index in indices]
        if spec.timestamp
        else None
    )
    frames = capture_frames(
        viewer,
        indices,
        scale=spec.scale,
        on_progress=on_progress,
        should_cancel=should_cancel,
    )
    written = write_movie(frames, spec.path, fps=spec.fps, quality=spec.quality, stamps=stamps)
    logger.info(
        "movie: %d frame(s) at %.3g fps (%.1f s) from timepoints %d-%d step %d",
        len(indices), spec.fps, spec.duration_s, spec.start, spec.stop, spec.step,
    )
    return written
