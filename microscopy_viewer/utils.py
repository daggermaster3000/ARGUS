"""Shared helpers: logging, unit conversion, axis handling, small numeric utilities."""

from __future__ import annotations

import logging
import os
import re
import sys
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

MICRON = "µm"
MICRON_SQ = "µm²"

#: Length units that may appear in file metadata, expressed as micrometres per unit.
_LENGTH_TO_MICRON = {
    "": 1.0,
    "px": 1.0,
    "pixel": 1.0,
    "pixels": 1.0,
    "um": 1.0,
    "µm": 1.0,  # U+00B5 micro sign
    "μm": 1.0,  # U+03BC greek small mu
    "micron": 1.0,
    "microns": 1.0,
    "micrometer": 1.0,
    "micrometre": 1.0,
    "nm": 1e-3,
    "nanometer": 1e-3,
    "mm": 1e3,
    "millimeter": 1e3,
    "millimetre": 1e3,
    "cm": 1e4,
    "centimeter": 1e4,
    "m": 1e6,
    "meter": 1e6,
    "metre": 1e6,
    "inch": 25400.0,
    "in": 25400.0,
}

#: Time units that may appear in file metadata, expressed as seconds per unit.
_TIME_TO_SECOND = {
    "": 1.0,
    "s": 1.0,
    "sec": 1.0,
    "second": 1.0,
    "seconds": 1.0,
    "ms": 1e-3,
    "msec": 1e-3,
    "millisecond": 1e-3,
    "us": 1e-6,
    "µs": 1e-6,
    "microsecond": 1e-6,
    "min": 60.0,
    "minute": 60.0,
    "h": 3600.0,
    "hr": 3600.0,
    "hour": 3600.0,
}

LOG_NAME = "microscopy_viewer"


def log_file() -> Path:
    """Location of the rolling log file, used when the GUI is started without a console."""
    from .runtime import app_data_dir

    directory = app_data_dir()
    directory.mkdir(parents=True, exist_ok=True)
    return directory / "microscopy_viewer.log"


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure the package logger once; safe to call repeatedly."""
    logger = logging.getLogger(LOG_NAME)
    if getattr(logger, "_mv_configured", False):
        return logger
    logger.setLevel(level)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")

    try:
        handler: logging.Handler = logging.FileHandler(log_file(), encoding="utf-8")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    except OSError:
        pass

    # pythonw.exe has no usable stderr; only add a stream handler when one exists.
    if sys.stderr is not None:
        stream = logging.StreamHandler(sys.stderr)
        stream.setFormatter(fmt)
        logger.addHandler(stream)

    logger._mv_configured = True  # type: ignore[attr-defined]
    return logger


def get_logger(suffix: str | None = None) -> logging.Logger:
    return logging.getLogger(LOG_NAME if not suffix else f"{LOG_NAME}.{suffix}")


def length_to_micron(value: float | None, unit: str | None) -> float | None:
    """Convert *value* given in *unit* to micrometres. Returns ``None`` if not convertible."""
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value):
        return None
    factor = _LENGTH_TO_MICRON.get((unit or "").strip().lower())
    if factor is None:
        return None
    return value * factor


def time_to_second(value: float | None, unit: str | None) -> float | None:
    """Convert *value* given in *unit* to seconds. Returns ``None`` if not convertible."""
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value):
        return None
    factor = _TIME_TO_SECOND.get((unit or "").strip().lower())
    if factor is None:
        return None
    return value * factor


def parse_float(text) -> float | None:
    """Pull the first floating point number out of an arbitrary value."""
    if text is None:
        return None
    if isinstance(text, (int, float, np.number)):
        value = float(text)
        return value if np.isfinite(value) else None
    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(text))
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def parse_float_list(text) -> list[float]:
    """Pull every floating point number out of an arbitrary value."""
    if text is None:
        return []
    if isinstance(text, (list, tuple, np.ndarray)):
        out = []
        for item in text:
            value = parse_float(item)
            if value is not None:
                out.append(value)
        return out
    return [float(m) for m in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(text))]


def normalise_axes(axes: str) -> str:
    """Map reader-specific axis codes onto the ``TCZYX`` vocabulary used internally.

    Unknown axes become ``Q`` so they are still counted but never mistaken for a
    spatial or temporal axis.
    """
    mapping = {
        "T": "T",
        "C": "C",
        "Z": "Z",
        "Y": "Y",
        "X": "X",
        "S": "C",  # tifffile sample axis (RGB) treated as channels
        "I": "T",  # generic sequence axis
        "Q": "Q",
    }
    return "".join(mapping.get(a.upper(), "Q") for a in axes)


def guess_axes(shape: Sequence[int]) -> str:
    """Best-effort axis assignment for files that carry no dimension metadata."""
    ndim = len(shape)
    if ndim <= 2:
        return "YX"[-ndim:] if ndim else ""
    if ndim == 3:
        # A small leading axis is far more likely to be channels than Z or T.
        return "CYX" if shape[0] <= 5 else "ZYX"
    if ndim == 4:
        return "TZYX" if shape[1] > 5 else "TCYX"
    if ndim == 5:
        return "TCZYX"
    return "Q" * (ndim - 5) + "TCZYX"


def describe_dimensionality(axes: str, shape: Sequence[int]) -> str:
    """Human readable summary such as ``XYZT (multichannel)``."""
    present = {a: n for a, n in zip(axes, shape)}
    order = [a for a in "XYZT" if present.get(a, 1) > 1]
    label = "".join(order) or "XY"
    channels = present.get("C", 1)
    if channels > 1:
        label += f" (multichannel, {channels} channels)"
    return label


def dedupe_name(name: str, existing: Iterable[str]) -> str:
    """Return *name*, suffixed with ``[n]`` if it already appears in *existing*."""
    taken = set(existing)
    if name not in taken:
        return name
    index = 1
    while f"{name} [{index}]" in taken:
        index += 1
    return f"{name} [{index}]"


def format_number(value, digits: int = 4) -> str:
    """Compact human-friendly number formatting for tables and labels."""
    if value is None:
        return "—"
    if isinstance(value, (list, tuple, np.ndarray)):
        return ", ".join(format_number(v, digits) for v in value)
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(value):
        return "—"
    if value == 0:
        return "0"
    if abs(value) < 1e-3 or abs(value) >= 1e6:
        return f"{value:.{digits}g}"
    return f"{value:.{digits}g}"


def shoelace_area(points: np.ndarray) -> float:
    """Polygon area from ordered 2D vertices (already scaled to physical units)."""
    if points.shape[0] < 3:
        return 0.0
    y = points[:, 0]
    x = points[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def polyline_length(points: np.ndarray, closed: bool = False) -> float:
    """Total length of a polyline whose vertices are already in physical units."""
    if points.shape[0] < 2:
        return 0.0
    pts = np.vstack([points, points[:1]]) if closed else points
    return float(np.sqrt(((np.diff(pts, axis=0)) ** 2).sum(axis=1)).sum())


def ellipse_axes(corners: np.ndarray) -> tuple[float, float]:
    """Semi-axis lengths of a napari ellipse, given its four (scaled) corner points."""
    if corners.shape[0] != 4:
        radii = (corners.max(axis=0) - corners.min(axis=0)) / 2.0
        radii = np.sort(radii)[::-1]
        return float(radii[0]), float(radii[1] if radii.size > 1 else radii[0])
    edge_a = np.linalg.norm(corners[1] - corners[0])
    edge_b = np.linalg.norm(corners[2] - corners[1])
    return float(edge_a / 2.0), float(edge_b / 2.0)


def ramanujan_perimeter(semi_a: float, semi_b: float) -> float:
    """Ramanujan's approximation of an ellipse perimeter (error < 1e-5 for our use)."""
    a, b = float(semi_a), float(semi_b)
    if a <= 0 or b <= 0:
        return 0.0
    h = ((a - b) ** 2) / ((a + b) ** 2)
    return float(np.pi * (a + b) * (1 + (3 * h) / (10 + np.sqrt(4 - 3 * h))))
