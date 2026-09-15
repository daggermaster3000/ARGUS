"""Read an object table back in, and paint the labels it describes.

A segmentation run writes one row per object — size, shape, intensity — and then
the table and the image go their separate ways: the numbers end up in Excel and
the masks stay in the viewer, with nothing tying a row to the object it came
from. This module is the way back. Give it the table a run wrote and it will

* say which column holds the label id, and which columns are worth plotting;
* turn any numeric column into a colour per label, so the segmentation can be
  looked at *as* the measurement — nuclei shaded by area, by mean intensity, by
  solidity — instead of as 18 000 arbitrary colours;
* guess which layer in the viewer the table belongs to, from the file name a
  batch run gave it.

No Qt and no napari here: the widget is a thin layer over these functions, and
everything below is testable headless.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .utils import get_logger

logger = get_logger("analysis")

#: Column names that hold the object id, in the order they are looked for. The
#: first is what this viewer's own export writes; the rest are what the tables
#: from scikit-image, CellProfiler and Fractal call it.
LABEL_COLUMNS = ("Label", "label", "label_id", "ObjectNumber", "object_id", "id")

#: Colormaps offered for label colouring. Perceptually uniform ones first: a
#: measurement painted in jet is a measurement misread.
COLORMAPS = ("viridis", "magma", "plasma", "cividis", "turbo", "coolwarm", "Greys")

#: Percentile range the colour scale is stretched over by default. Object tables
#: routinely have a handful of enormous outliers — two nuclei merged into one —
#: and scaling to the true maximum leaves everything else the same dark blue.
DEFAULT_LOW_PERCENTILE = 1.0
DEFAULT_HIGH_PERCENTILE = 99.0

#: Points above which the scatter plot subsamples. Matplotlib will draw a million
#: points, but it will take seconds to do it and the result is a solid block.
MAX_SCATTER_POINTS = 100_000

TABLE_SUFFIXES = (".csv", ".tsv", ".txt", ".xlsx", ".xls")


def read_table(path: str | Path):
    """Read a measurement table into a DataFrame, from CSV or Excel."""
    import pandas as pd

    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"{source} does not exist")
    suffix = source.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        frame = pd.read_excel(source)
    else:
        separator = "\t" if suffix == ".tsv" else None
        # sep=None asks the sniffer to work it out, which handles the semicolon
        # CSVs a European Excel writes without the user having to know that.
        frame = pd.read_csv(source, sep=separator, engine="python" if separator is None else "c")
    if frame.empty:
        raise ValueError(f"{source.name} has no rows")
    frame.columns = [str(name).strip() for name in frame.columns]
    logger.info("read %s: %d row(s), %d column(s)", source.name, len(frame), len(frame.columns))
    return frame


def label_column(frame) -> str | None:
    """The column holding the object id, or None when the table has no such thing.

    A table without one can still be plotted; it just cannot colour anything, and
    saying so is better than colouring by row number and being quietly wrong.
    """
    for name in LABEL_COLUMNS:
        if name in frame.columns:
            return name
    for name in frame.columns:
        if str(name).strip().lower() in {"label", "labels", "id"}:
            return str(name)
    return None


def numeric_columns(frame, exclude: Sequence[str] = ()) -> list[str]:
    """Columns that can be coloured by or plotted, in table order."""
    import pandas as pd

    skip = {str(name) for name in exclude}
    return [
        str(name)
        for name in frame.columns
        if str(name) not in skip and pd.api.types.is_numeric_dtype(frame[name])
    ]


def value_range(
    values: Sequence[float],
    low_percentile: float = DEFAULT_LOW_PERCENTILE,
    high_percentile: float = DEFAULT_HIGH_PERCENTILE,
) -> tuple[float, float]:
    """The range the colour scale covers, ignoring NaNs and the extreme tails."""
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return 0.0, 1.0
    low = float(np.percentile(finite, max(0.0, min(100.0, low_percentile))))
    high = float(np.percentile(finite, max(0.0, min(100.0, high_percentile))))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low, high = float(finite.min()), float(finite.max())
    if high <= low:  # every object has the same value
        high = low + 1.0
    return low, high


def normalise(values: Sequence[float], low: float, high: float) -> np.ndarray:
    """Values mapped onto 0-1 and clipped, with NaN left as NaN."""
    array = np.asarray(values, dtype=float)
    span = float(high) - float(low)
    if span <= 0:
        span = 1.0
    scaled = (array - float(low)) / span
    return np.clip(scaled, 0.0, 1.0, out=scaled, where=np.isfinite(scaled))


def colormap_colors(fractions: Sequence[float], name: str = "viridis") -> np.ndarray:
    """RGBA for each 0-1 fraction; NaN comes back fully transparent."""
    from matplotlib import colormaps

    values = np.asarray(fractions, dtype=float)
    try:
        table = colormaps[name]
    except KeyError:
        logger.warning("unknown colormap %r; using viridis", name)
        table = colormaps["viridis"]
    colors = np.asarray(table(np.nan_to_num(values, nan=0.0)), dtype=float)
    colors[~np.isfinite(values)] = (0.0, 0.0, 0.0, 0.0)
    return colors


def label_colors(
    labels: Sequence[int],
    values: Sequence[float],
    colormap: str = "viridis",
    low: float | None = None,
    high: float | None = None,
    low_percentile: float = DEFAULT_LOW_PERCENTILE,
    high_percentile: float = DEFAULT_HIGH_PERCENTILE,
) -> tuple[dict, tuple[float, float]]:
    """``({label: rgba}, (low, high))`` for a napari ``DirectLabelColormap``.

    Background and any label the table does not mention map to ``None``, which
    napari draws as transparent — so an object with no row simply is not painted,
    rather than being painted the colour of zero.
    """
    ids = np.asarray(labels)
    numbers = np.asarray(values, dtype=float)
    if ids.size != numbers.size:
        raise ValueError(
            f"{ids.size} label(s) and {numbers.size} value(s); they come from the same table"
        )
    if low is None or high is None:
        low, high = value_range(numbers, low_percentile, high_percentile)

    colors = colormap_colors(normalise(numbers, low, high), colormap)
    mapping: dict[Any, tuple[float, float, float, float]] = {
        None: (0.0, 0.0, 0.0, 0.0),  # background, and every label without a row
        0: (0.0, 0.0, 0.0, 0.0),
    }
    for label, color in zip(ids, colors):
        try:
            key = int(label)
        except (TypeError, ValueError):
            continue
        if key <= 0:
            continue
        mapping[key] = tuple(float(c) for c in color)
    return mapping, (float(low), float(high))


def scatter_sample(count: int, limit: int = MAX_SCATTER_POINTS, seed: int = 0) -> np.ndarray | None:
    """Indices to plot when there are too many points, or None to plot them all.

    Random rather than the first N: a table is written in label order, which on a
    plate runs roughly top-left to bottom-right, so the first N points would be a
    corner of the well rather than a sample of it.
    """
    if count <= limit:
        return None
    return np.random.default_rng(seed).choice(count, size=limit, replace=False)


def describe_column(values: Sequence[float]) -> str:
    """One line of distribution, for under the plot."""
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return "no numeric values"
    low, median, high = np.percentile(finite, [0, 50, 100])
    return (
        f"n = {finite.size}, min {low:.4g}, median {median:.4g}, max {high:.4g}"
        + ("" if finite.size == array.size else f" ({array.size - finite.size} blank)")
    )


def component_from_name(stem: str) -> str:
    """``G_07_0`` -> ``G/07/0``: the image a batch table was written for.

    :func:`microscopy_viewer.batch.run_batch` names each table after the path of
    the image inside the store with the separators flattened, so the path can be
    read back out of the file name and used to find the layer.
    """
    text = str(stem).strip()
    match = re.search(r"([A-Za-z]{1,2})[_\-/](\d{1,3})(?:[_\-/](\d{1,3}))?$", text)
    if not match:
        return ""
    parts = [part for part in match.groups() if part]
    return "/".join(parts)


def match_layer(stem: str, names: Sequence[str]) -> str | None:
    """The layer a table most likely belongs to, or None when nothing fits.

    Matched on the well and field in the file name rather than on the whole name,
    because the layer is called something like ``G/07 :: cycle 1 :: nuclei`` and
    the table ``G_07_0.csv``; what they have in common is the well.
    """
    candidates = [str(name) for name in names]
    if not candidates:
        return None
    component = component_from_name(Path(str(stem)).stem)
    if not component:
        return None

    parts = component.split("/")
    well = "/".join(parts[:2])
    scored: list[tuple[int, str]] = []
    for name in candidates:
        score = 0
        if well and well in name:
            score += 2
        if component in name:
            score += 3
        if score:
            scored.append((score, name))
    if not scored:
        return None
    best = max(score for score, _name in scored)
    return next(name for score, name in scored if score == best)
