"""Movie export checks: frame selection, the overlay, and the written file.

No display needed — the frames are synthesised rather than screenshotted. The
encoder checks skip themselves when imageio-ffmpeg is not installed.

Run with::

    python tests/test_movie.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from microscopy_viewer import movie  # noqa: E402

_failures: list[str] = []


def check(condition: bool, message: str) -> None:
    if condition:
        print(f"  ok   {message}")
    else:
        print(f"  FAIL {message}")
        _failures.append(message)


def skip(message: str) -> None:
    print(f"  skip {message}")


def _frames(count: int = 8, height: int = 64, width: int = 80) -> list[np.ndarray]:
    """RGBA frames with a moving bar, as a canvas screenshot would arrive."""
    out = []
    for index in range(count):
        frame = np.zeros((height, width, 4), dtype=np.uint8)
        frame[..., 3] = 255
        column = int(index / max(1, count - 1) * (width - 8))
        frame[:, column : column + 8, :3] = 255
        out.append(frame)
    return out


# ---------------------------------------------------------------------------


def test_frame_selection() -> None:
    print("the exported range is inclusive, and a stride shortens a long series")
    check(movie.frame_indices(0, 4) == [0, 1, 2, 3, 4], "the last timepoint is included")
    check(movie.frame_indices(2, 2) == [2], "a single timepoint is a one-frame movie")
    check(movie.frame_indices(0, 9, 3) == [0, 3, 6, 9], "every third timepoint")
    check(movie.frame_indices(7, 3) == [3, 4, 5, 6, 7], "a reversed range is put back in order")
    check(movie.frame_indices(0, 4, 0) == [0, 1, 2, 3, 4], "a zero stride is treated as one")

    spec = movie.MovieSpec(path=Path("x.mov"), fps=5.0, start=0, stop=9)
    check(len(spec.indices) == 10, "the spec reports the frames it will write")
    check(abs(spec.duration_s - 2.0) < 1e-9, "ten frames at 5 fps is two seconds")


def test_frame_preparation() -> None:
    print("screenshots are turned into something an encoder accepts")
    rgba = _frames(1)[0]
    rgb = movie.to_rgb(rgba)
    check(rgb.shape[-1] == 3, "the alpha channel is dropped — H.264 has no alpha")
    check(rgb.dtype == np.uint8, "and the result is 8-bit")
    check(np.array_equal(rgb, rgba[..., :3]), "the colour channels are untouched")

    grey = movie.to_rgb(np.zeros((8, 8), dtype=np.uint8))
    check(grey.shape == (8, 8, 3), "a greyscale frame is broadcast to RGB")

    odd = np.zeros((65, 81, 3), dtype=np.uint8)
    even = movie.even_dimensions(odd)
    check(even.shape == (64, 80, 3), "odd dimensions are cropped, not rescaled")
    check(
        movie.even_dimensions(np.zeros((64, 80, 3), dtype=np.uint8)).shape == (64, 80, 3),
        "an already-even frame is left exactly as it is, so the scale bar stays true",
    )


def test_timestamp_overlay() -> None:
    print("the burnt-in timestamp reads the acquisition interval")
    check(movie.stamp_text(4, 2.5) == "0:10.0", "four intervals of 2.5 s is ten seconds")
    check(movie.stamp_text(4, None) == "t = 4", "without an interval the timepoint number is used")
    check(movie.stamp_text(4, 2.5, "fish 3") == "fish 3 0:10.0", "a label is prefixed")

    frame = np.zeros((64, 80, 3), dtype=np.uint8)
    stamped = movie.draw_timestamp(frame, "0:10.0")
    check(stamped.shape == frame.shape, "the overlay does not change the frame size")
    check(stamped.max() > 0, "and something was actually drawn")
    check(
        np.array_equal(movie.draw_timestamp(frame, ""), frame),
        "an empty stamp leaves the frame alone",
    )

    prepared = movie.prepare_frame(_frames(1)[0], "0:10.0")
    check(prepared.shape[-1] == 3 and prepared.shape[0] % 2 == 0, "prepare_frame does both jobs")


def test_rejects_formats(directory: Path) -> None:
    print("an unusable target is refused with a message worth showing")
    try:
        movie.write_movie(_frames(2), directory / "frames.tiff")
        check(False, "a .tiff target should be refused")
    except movie.MovieExportError as exc:
        check(".mov" in str(exc), f"the error names the formats that do work: {exc}")

    try:
        movie.write_movie(iter([]), directory / "empty.gif")
        check(False, "an empty movie should be refused")
    except movie.MovieExportError:
        check(True, "writing no frames is an error rather than an empty file")


def test_writes_a_gif(directory: Path) -> None:
    print("a GIF is written without any codec")
    path = movie.write_movie(_frames(6), directory / "loop.gif", fps=8)
    check(path.exists() and path.stat().st_size > 0, f"loop.gif written ({path.stat().st_size} bytes)")

    from PIL import Image

    with Image.open(path) as image:
        check(image.n_frames == 6, "and it holds the six frames that went in")
        check(
            image.info.get("duration") == 120,
            f"8 fps is 125 ms a frame, rounded to the 10 ms GIF can store ({image.info.get('duration')})",
        )
        check(image.info.get("loop") == 0, "and it loops forever, which is the point of a GIF")


def test_writes_a_mov(directory: Path) -> None:
    print("a .mov is written as H.264 for PowerPoint")
    ok, message = movie.encoder_available()
    if not ok:
        skip(f"no ffmpeg: {message}")
        return

    frames = _frames(12, height=65, width=81)  # deliberately odd-sized
    path = movie.write_movie(frames, directory / "series.mov", fps=10, quality=6)
    check(path.exists() and path.stat().st_size > 0, f"series.mov written ({path.stat().st_size} bytes)")

    import imageio.v2 as iio

    reader = iio.get_reader(str(path))
    try:
        meta = reader.get_meta_data()
        first = reader.get_data(0)
    finally:
        reader.close()
    check(first.shape[:2] == (64, 80), f"the odd frame was cropped to even, not rescaled ({first.shape[:2]})")
    check(abs(float(meta.get("fps", 0)) - 10.0) < 0.5, f"the file plays at the requested rate ({meta.get('fps')})")

    stamped = movie.write_movie(
        _frames(4), directory / "stamped.mov", fps=4, stamps=["a", "b", "c", "d"]
    )
    check(stamped.exists(), "a stamped movie is written too")


def main() -> int:
    print("Movie export checks\n")
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        test_frame_selection()
        test_frame_preparation()
        test_timestamp_overlay()
        test_rejects_formats(directory)
        test_writes_a_gif(directory)
        test_writes_a_mov(directory)

    print()
    if _failures:
        print(f"{len(_failures)} check(s) failed:")
        for message in _failures:
            print(f"  - {message}")
        return 1
    print("All movie checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
