"""ROI intensity statistics and distribution comparison.

Deliberately free of Qt and of napari widgets: the panel snapshots what it needs
from the viewer on the main thread, and everything here then runs in a worker
thread. That keeps the GUI responsive over multi-megapixel ROIs and makes the
maths testable without a display.

Nothing here assumes normally distributed intensities. Separation between two
conditions is reported as the Mann-Whitney U / ROC AUC and as a histogram
overlap coefficient, both rank- or density-based.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

from .utils import get_logger

logger = get_logger("intensity")

#: Ceiling on the pixels kept per ROI/condition for histograms and the AUC. A
#: 2040x2040 ROI holds 4M pixels; a few hundred thousand is far more than enough
#: to characterise a distribution and keeps memory and rank sorting bounded.
MAX_SAMPLES = 200_000

#: Random seed for that subsampling, so repeated runs give identical numbers.
SAMPLE_SEED = 12345

PLANE_MODES = ("Current slice", "Maximum projection", "Mean projection")

#: How pixel values are rescaled against each condition's own background before
#: histograms, the AUC and the normalised mean. Dividing is the one that makes
#: separate acquisitions comparable when illumination or gain differed.
NORMALIZATIONS = ("None", "Subtract background", "Divide by background")

#: Marker used in the UI and in RoiSpec.condition for a ROI shared by everything.
ALL_CONDITIONS = "All conditions"

#: Column order for the statistics table and the CSV export.
#: Results first: in a narrow dock the leading columns are the ones on screen, so
#: the layer name and provenance fields go last.
STAT_COLUMNS = (
    "roi_name", "condition", "n_pixels", "mean", "median", "std", "integrated",
    "corrected_mean", "normalized_mean", "signal_to_background", "snr",
    "background_mean", "background_std", "minimum", "maximum",
    "saturated_pixels", "saturated_percent", "offset", "normalization", "plane",
    "roi_source", "layer_name",
)

STAT_LABELS = {
    "roi_name": "ROI",
    "condition": "Condition",
    "layer_name": "Layer",
    "n_pixels": "Pixels",
    "mean": "Mean",
    "median": "Median",
    "std": "Std",
    "integrated": "Integrated",
    "minimum": "Min",
    "maximum": "Max",
    "background_mean": "Background mean",
    "background_std": "Background std",
    "corrected_mean": "Mean − background",
    "normalized_mean": "Normalised mean",
    "signal_to_background": "Signal/background",
    "snr": "SNR",
    "saturated_pixels": "Saturated px",
    "saturated_percent": "Saturated %",
    "offset": "Camera offset",
    "normalization": "Normalisation",
    "plane": "Plane",
    "roi_source": "ROI shape",
}


# ---------------------------------------------------------------------------
# Inputs, snapshotted from the viewer on the main thread
# ---------------------------------------------------------------------------


@dataclass
class ConditionSpec:
    """One named condition and the image layer that provides its pixels."""

    name: str
    layer_name: str
    data: Any                      # full-resolution array (numpy or dask)
    axes: str                      # e.g. "TZYX"; may be "" if unknown
    scale: tuple[float, ...]
    translate: tuple[float, ...]
    current_step: tuple[int, ...]  # per layer axis, already right-aligned
    dtype: Any


@dataclass
class RoiSpec:
    """A ROI as world-space (row, column) vertices.

    Keeping the geometry in world coordinates is what makes a *shared* ROI
    spatially matched: each condition converts the same outline into its own
    pixel grid, so differing pixel sizes or offsets do not shift the region.

    ``condition`` is ``None`` for a shared ROI — measured on every condition — or
    a condition name for a ROI that belongs to one sample only. Per-condition
    ROIs are what you want when the specimens sit in different places in the
    field of view, which is the usual case across separate acquisitions.

    ``label`` is the comparison key: ROIs drawn on different conditions but given
    the same label are treated as the same region for the statistics table and
    the overlaid histograms. It defaults to the ROI's own name, which reproduces
    the shared-ROI behaviour exactly.
    """

    name: str
    vertices_world: np.ndarray     # (N, 2), world Y/X
    is_background: bool = False
    condition: str | None = None
    label: str = ""

    def __post_init__(self) -> None:
        if not self.label:
            self.label = self.name

    def applies_to(self, condition_name: str) -> bool:
        return self.condition is None or self.condition == condition_name


@dataclass
class RoiStats:
    """Statistics for one ROI measured on one condition."""

    roi_name: str = ""      # the comparison label
    roi_source: str = ""    # the shape it came from
    condition: str = ""
    layer_name: str = ""
    n_pixels: int = 0
    mean: float = float("nan")
    median: float = float("nan")
    std: float = float("nan")
    integrated: float = float("nan")
    minimum: float = float("nan")
    maximum: float = float("nan")
    background_mean: float | None = None
    background_std: float | None = None
    corrected_mean: float | None = None
    normalized_mean: float | None = None
    signal_to_background: float | None = None
    snr: float | None = None
    saturated_pixels: int = 0
    saturated_percent: float = 0.0
    offset: float = 0.0
    normalization: str = "None"
    plane: str = ""
    is_background: bool = False

    def as_row(self) -> dict[str, Any]:
        data = asdict(self)
        return {key: data[key] for key in STAT_COLUMNS if key in data}


@dataclass
class ComparisonResult:
    """Everything one measurement run produces."""

    stats: list[RoiStats] = field(default_factory=list)
    #: ``(roi_name, condition)`` -> offset-corrected pixel sample for plotting.
    samples: dict[tuple[str, str], np.ndarray] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    plane_description: str = ""
    normalization: str = "None"

    def conditions(self) -> list[str]:
        seen: list[str] = []
        for entry in self.stats:
            if entry.condition not in seen:
                seen.append(entry.condition)
        return seen

    def series(self, include_background: bool = False) -> list[tuple[str, str]]:
        """Every measured ``(roi label, condition)`` that has pixels to plot.

        Ordered ROI-major so curves of the same region across conditions sit next
        to each other in the list and the legend.
        """
        out: list[tuple[str, str]] = []
        for entry in self.stats:
            key = (entry.roi_name, entry.condition)
            if key in out or key not in self.samples:
                continue
            # Filter on the row's own role, not on the label: a signal ROI that
            # happens to share a label with a background must still be plotted.
            if entry.is_background and not include_background:
                continue
            out.append(key)
        return out

    def roi_names(self, include_background: bool = True) -> list[str]:
        seen: list[str] = []
        for entry in self.stats:
            if not include_background and entry.is_background:
                continue
            if entry.roi_name not in seen:
                seen.append(entry.roi_name)
        return seen


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def ellipse_to_polygon(corners: np.ndarray, n_points: int = 64) -> np.ndarray:
    """Convert napari's four ellipse corner points into a polygon outline.

    The corners describe a (possibly rotated) bounding parallelogram, so the two
    edge vectors from the first corner are the semi-axis directions.
    """
    corners = np.asarray(corners, dtype=float)
    centre = corners.mean(axis=0)
    semi_a = (corners[1] - corners[0]) / 2.0
    semi_b = (corners[-1] - corners[0]) / 2.0
    angles = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)
    return centre + np.outer(np.cos(angles), semi_a) + np.outer(np.sin(angles), semi_b)


def world_vertices(shape_data: np.ndarray, scale: Sequence[float], translate: Sequence[float]) -> np.ndarray:
    """Convert a shape's layer coordinates to world (row, column) coordinates."""
    vertices = np.asarray(shape_data, dtype=float)[:, -2:]
    scale_yx = np.asarray(scale, dtype=float)[-2:]
    translate_yx = np.asarray(translate, dtype=float)[-2:]
    return vertices * scale_yx + translate_yx


def world_to_data(vertices_world: np.ndarray, scale: Sequence[float], translate: Sequence[float]) -> np.ndarray:
    """Convert world (row, column) coordinates into one layer's pixel indices."""
    scale_yx = np.asarray(scale, dtype=float)[-2:]
    translate_yx = np.asarray(translate, dtype=float)[-2:]
    safe = np.where(scale_yx == 0, 1.0, scale_yx)
    return (np.asarray(vertices_world, dtype=float) - translate_yx) / safe


def polygon_mask(vertices: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Rasterise a polygon given in pixel coordinates onto a boolean mask.

    A pixel is inside when its *centre* is inside the outline. That is the
    convention users expect from a ROI tool — a rectangle dragged from row 10 to
    row 30 is 20 pixels tall, not 21 — and it is what keeps pixel counts and
    integrated intensities unbiased. ``skimage.draw.polygon2mask`` is not used
    here precisely because it fills boundary pixels on both edges, which inflates
    a small ROI by several percent.

    Only the polygon's bounding box is tested, so cost scales with the ROI rather
    than with the image.
    """
    from matplotlib.path import Path

    vertices = np.asarray(vertices, dtype=float)
    mask = np.zeros(shape, dtype=bool)
    if vertices.shape[0] < 3:
        return mask

    row_start = max(int(np.floor(vertices[:, 0].min())), 0)
    row_stop = min(int(np.ceil(vertices[:, 0].max())) + 1, shape[0])
    col_start = max(int(np.floor(vertices[:, 1].min())), 0)
    col_stop = min(int(np.ceil(vertices[:, 1].max())) + 1, shape[1])
    if row_stop <= row_start or col_stop <= col_start:
        return mask

    rows = np.arange(row_start, row_stop, dtype=float) + 0.5
    cols = np.arange(col_start, col_stop, dtype=float) + 0.5
    grid_rows, grid_cols = np.meshgrid(rows, cols, indexing="ij")
    points = np.column_stack((grid_rows.ravel(), grid_cols.ravel()))
    inside = Path(vertices).contains_points(points)
    mask[row_start:row_stop, col_start:col_stop] = inside.reshape(
        row_stop - row_start, col_stop - col_start
    )
    return mask


# ---------------------------------------------------------------------------
# Plane extraction
# ---------------------------------------------------------------------------


def _z_axis(axes: str, ndim: int) -> int | None:
    """Index of the Z axis within a layer's own dimensions, if it has one."""
    if axes and len(axes) == ndim and "Z" in axes:
        return axes.index("Z")
    return None


def extract_plane(condition: ConditionSpec, mode: str) -> tuple[np.ndarray, str]:
    """Reduce a condition's layer to the 2D plane that will be measured.

    ``Current slice`` matches exactly what is on screen. The projections reduce
    over Z only, holding every other axis (time in particular) at the slider
    position, so a projection of a 4D stack stays at the displayed timepoint.
    """
    data = condition.data
    ndim = int(getattr(data, "ndim", 0))
    if ndim <= 2:
        return np.asarray(data), "single plane"

    z_axis = _z_axis(condition.axes, ndim)
    project = mode in ("Maximum projection", "Mean projection") and z_axis is not None

    index: list[Any] = []
    described: list[str] = []
    for axis in range(ndim):
        if axis >= ndim - 2:
            index.append(slice(None))
            continue
        if project and axis == z_axis:
            index.append(slice(None))
            continue
        step = 0
        if axis < len(condition.current_step):
            step = int(condition.current_step[axis])
        step = max(0, min(step, int(data.shape[axis]) - 1))
        index.append(step)
        label = condition.axes[axis] if condition.axes and len(condition.axes) == ndim else f"axis{axis}"
        described.append(f"{label}={step}")

    plane = data[tuple(index)]
    if project:
        # After indexing, Z is the only axis left in front of Y and X.
        plane = plane.max(axis=0) if mode == "Maximum projection" else plane.mean(axis=0)
        described.append("max over Z" if mode == "Maximum projection" else "mean over Z")
    elif mode != "Current slice":
        described.append("no Z axis — current slice used")

    return np.asarray(plane), ", ".join(described) or "single plane"


#: Short tag appended to a projected layer's name, per plane mode.
PROJECTION_SUFFIX = {
    "Maximum projection": "MIP",
    "Mean projection": "Mean Z",
    "Current slice": "slice",
}


def has_z_axis(condition: ConditionSpec) -> bool:
    """Whether this condition's layer has a Z axis worth projecting through."""
    ndim = int(getattr(condition.data, "ndim", 0))
    axis = _z_axis(condition.axes, ndim)
    return axis is not None and int(condition.data.shape[axis]) > 1


def projection_layer_name(source_name: str, mode: str) -> str:
    """Name for the flattened layer produced from *source_name*."""
    return f"{source_name} [{PROJECTION_SUFFIX.get(mode, 'projection')}]"


def default_saturation_level(dtype) -> float | None:
    """Value at which pixels of *dtype* are clipped, if that is well defined."""
    dtype = np.dtype(dtype)
    if dtype.kind in "ui":
        return float(np.iinfo(dtype).max)
    return None


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def _subsample(values: np.ndarray, limit: int = MAX_SAMPLES) -> np.ndarray:
    """Deterministically thin *values* so repeated runs agree."""
    if values.size <= limit:
        return values
    rng = np.random.default_rng(SAMPLE_SEED)
    return values[rng.choice(values.size, size=limit, replace=False)]


def _ratio(numerator: float, denominator: float) -> float | None:
    """Guarded division: returns ``None`` rather than an inf when it is undefined."""
    if denominator is None or not np.isfinite(denominator) or abs(denominator) < 1e-12:
        return None
    value = float(numerator) / float(denominator)
    return value if np.isfinite(value) else None


def measure_values(
    values: np.ndarray,
    *,
    roi_name: str,
    condition: str,
    layer_name: str,
    offset: float = 0.0,
    saturation_level: float | None = None,
    raw_for_saturation: np.ndarray | None = None,
    is_background: bool = False,
    plane: str = "",
) -> RoiStats:
    """Statistics for the pixels of one ROI on one condition.

    *values* must already have the camera offset removed; *raw_for_saturation*
    is the uncorrected data, since clipping happens at the sensor and so has to
    be judged before the offset is subtracted.
    """
    values = np.asarray(values).ravel()
    stats = RoiStats(
        roi_name=roi_name,
        condition=condition,
        layer_name=layer_name,
        n_pixels=int(values.size),
        offset=float(offset),
        plane=plane,
        is_background=is_background,
    )
    if values.size == 0:
        return stats

    finite = values[np.isfinite(values)] if values.dtype.kind == "f" else values
    if finite.size == 0:
        return stats

    stats.mean = float(np.mean(finite))
    stats.median = float(np.median(finite))
    stats.std = float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0
    stats.integrated = float(np.sum(finite))
    stats.minimum = float(np.min(finite))
    stats.maximum = float(np.max(finite))

    if saturation_level is not None:
        raw = values if raw_for_saturation is None else np.asarray(raw_for_saturation).ravel()
        saturated = int(np.count_nonzero(raw >= saturation_level))
        stats.saturated_pixels = saturated
        stats.saturated_percent = 100.0 * saturated / max(raw.size, 1)
    return stats


def normalize_values(
    values: np.ndarray, background_mean: float | None, normalization: str
) -> np.ndarray:
    """Rescale pixel values against the background of their own condition.

    ``Subtract background`` puts every condition on a common zero;
    ``Divide by background`` makes them comparable in fold-over-background, which
    is what separate acquisitions need when illumination or gain differed. A
    background at or near zero leaves the values untouched rather than producing
    infinities.
    """
    if normalization == "None" or background_mean is None:
        return values
    if normalization == "Subtract background":
        return values - float(background_mean)
    if normalization == "Divide by background":
        if abs(float(background_mean)) < 1e-12:
            return values
        return values / float(background_mean)
    return values


def normalized_axis_label(normalization: str) -> str:
    """Axis label describing what the plotted pixel values now mean."""
    return {
        "None": "Intensity (offset corrected)",
        "Subtract background": "Intensity − background",
        "Divide by background": "Intensity / background",
    }.get(normalization, "Intensity")


def series_name(roi_name: str, condition: str) -> str:
    """Display name for one measured ROI-on-condition curve."""
    return f"{roi_name} — {condition}"


def rois_for(rois: Sequence[RoiSpec], condition_name: str) -> list[RoiSpec]:
    """The ROIs that should be measured on *condition_name*."""
    return [roi for roi in rois if roi.applies_to(condition_name)]


def background_for(rois: Sequence[RoiSpec], condition_name: str) -> RoiSpec | None:
    """The background ROI governing *condition_name*.

    A background drawn for this condition specifically wins over a shared one, so
    each sample can be normalised against its own nearby background.
    """
    backgrounds = [roi for roi in rois if roi.is_background]
    for roi in backgrounds:
        if roi.condition == condition_name:
            return roi
    for roi in backgrounds:
        if roi.condition is None:
            return roi
    return None


def apply_background(stats: RoiStats, background: RoiStats | None) -> RoiStats:
    """Fill in the background-relative fields of *stats*.

    ``corrected_mean`` is the plain difference of means, ``signal_to_background``
    their ratio, and ``snr`` the difference divided by the background's spread —
    the usual detection-limit form, which is why it uses the background std
    rather than the signal's.
    """
    if background is None or background.n_pixels == 0:
        return stats
    stats.background_mean = background.mean
    stats.background_std = background.std
    stats.corrected_mean = float(stats.mean - background.mean)
    stats.signal_to_background = _ratio(stats.mean, background.mean)
    stats.snr = _ratio(stats.mean - background.mean, background.std)
    return stats


# ---------------------------------------------------------------------------
# Distribution comparison
# ---------------------------------------------------------------------------


@dataclass
class Separation:
    """How distinguishable two conditions' pixel populations are."""

    condition_a: str = ""
    condition_b: str = ""
    auc: float | None = None
    u_statistic: float | None = None
    p_value: float | None = None
    overlap: float | None = None
    n_a: int = 0
    n_b: int = 0
    subsampled: bool = False

    def summary(self) -> str:
        if self.auc is None:
            return "Not enough data for a separation metric."
        # AUC below 0.5 just means the ordering is reversed; report the strength.
        strength = max(self.auc, 1.0 - self.auc)
        parts = [
            f"ROC AUC {self.auc:.3f} (separation {strength:.3f})",
            f"overlap {self.overlap:.3f}" if self.overlap is not None else "",
            f"p {self.p_value:.3g}" if self.p_value is not None else "",
            f"n = {self.n_a:,} vs {self.n_b:,}" + (" (subsampled)" if self.subsampled else ""),
        ]
        return "   ".join(part for part in parts if part)


def mann_whitney_auc(a: np.ndarray, b: np.ndarray, limit: int = MAX_SAMPLES) -> Separation:
    """Rank-based separation between two pixel populations.

    The AUC is the probability that a random pixel from *b* exceeds a random
    pixel from *a* (ties counted as half), which is ``U / (n_a * n_b)``. It makes
    no distributional assumption, which matters because fluorescence intensities
    inside a ROI are typically skewed and multi-modal rather than Gaussian.
    """
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    result = Separation(n_a=int(a.size), n_b=int(b.size))
    if a.size == 0 or b.size == 0:
        return result

    sub_a, sub_b = _subsample(a, limit), _subsample(b, limit)
    result.subsampled = sub_a.size < a.size or sub_b.size < b.size
    result.n_a, result.n_b = int(sub_a.size), int(sub_b.size)

    try:
        from scipy.stats import mannwhitneyu

        outcome = mannwhitneyu(sub_b, sub_a, alternative="two-sided")
        result.u_statistic = float(outcome.statistic)
        result.p_value = float(outcome.pvalue)
        result.auc = float(outcome.statistic) / (sub_a.size * sub_b.size)
    except Exception:
        logger.debug("scipy unavailable or failed; using the numpy rank fallback", exc_info=True)
        result.auc = _auc_from_ranks(sub_a, sub_b)

    result.overlap = overlap_coefficient(sub_a, sub_b)
    return result


def _auc_from_ranks(a: np.ndarray, b: np.ndarray) -> float:
    """AUC via mid-ranks, used when scipy is not importable."""
    combined = np.concatenate([a, b])
    order = np.argsort(combined, kind="mergesort")
    ranks = np.empty(combined.size, dtype=float)
    ranks[order] = np.arange(1, combined.size + 1, dtype=float)

    # Average the ranks within each group of equal values so ties count as half.
    sorted_values = combined[order]
    start = 0
    for index in range(1, sorted_values.size + 1):
        if index == sorted_values.size or sorted_values[index] != sorted_values[start]:
            if index - start > 1:
                ranks[order[start:index]] = ranks[order[start:index]].mean()
            start = index

    rank_sum_b = ranks[a.size :].sum()
    u_b = rank_sum_b - b.size * (b.size + 1) / 2.0
    return float(u_b / (a.size * b.size))


def overlap_coefficient(a: np.ndarray, b: np.ndarray, bins: int = 256) -> float | None:
    """Area shared by two normalised histograms, from 1 (identical) to 0 (disjoint)."""
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    if a.size == 0 or b.size == 0:
        return None
    low = float(min(a.min(), b.min()))
    high = float(max(a.max(), b.max()))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return 1.0
    edges = np.linspace(low, high, bins + 1)
    density_a, _ = np.histogram(a, bins=edges, density=True)
    density_b, _ = np.histogram(b, bins=edges, density=True)
    width = edges[1] - edges[0]
    return float(np.minimum(density_a, density_b).sum() * width)


def shared_bins(samples: Iterable[np.ndarray], bins: int = 128) -> np.ndarray | None:
    """Common bin edges spanning every supplied sample.

    A shared range is what makes the overlaid curves comparable; per-curve bins
    would silently rescale each condition.
    """
    lows: list[float] = []
    highs: list[float] = []
    for values in samples:
        values = np.asarray(values).ravel()
        values = values[np.isfinite(values)]
        if values.size:
            lows.append(float(values.min()))
            highs.append(float(values.max()))
    if not lows:
        return None
    low, high = min(lows), max(highs)
    if high <= low:
        high = low + 1.0
    return np.linspace(low, high, int(bins) + 1)


# ---------------------------------------------------------------------------
# The measurement run
# ---------------------------------------------------------------------------


def run_comparison(
    conditions: Sequence[ConditionSpec],
    rois: Sequence[RoiSpec],
    mode: str = "Current slice",
    offset: float = 0.0,
    saturation_level: float | None = None,
    sample_limit: int = MAX_SAMPLES,
    normalization: str = "None",
) -> ComparisonResult:
    """Measure the ROIs that apply to each condition. Safe to run off-thread.

    Each condition is background-corrected against *its own* background ROI when
    one is assigned to it, falling back to a shared background otherwise, so
    samples imaged in different fields normalise independently.
    """
    result = ComparisonResult()
    result.normalization = normalization
    if not conditions or not rois:
        result.warnings.append("Select at least one condition and draw at least one ROI.")
        return result

    plane_notes: list[str] = []
    any_background = False

    for condition in conditions:
        applicable = rois_for(rois, condition.name)
        if not applicable:
            result.warnings.append(f"{condition.name}: no ROI is assigned to it.")
            continue

        try:
            plane, description = extract_plane(condition, mode)
        except Exception as exc:
            logger.exception("could not extract a plane for %s", condition.name)
            result.warnings.append(f"{condition.name}: could not read the displayed plane ({exc}).")
            continue
        plane_notes.append(description)

        level = saturation_level
        if level is None:
            level = default_saturation_level(condition.dtype)

        # The background has to be measured first: every other ROI on this
        # condition is expressed relative to it.
        background_roi = background_for(rois, condition.name)
        background_stats: RoiStats | None = None
        if background_roi is not None:
            any_background = True
            background_stats = _measure_roi(
                plane, background_roi, condition, offset, level, description
            )
        background_mean = background_stats.mean if background_stats is not None else None

        for roi in applicable:
            if background_roi is not None and roi.name == background_roi.name:
                stats = background_stats
            else:
                stats = _measure_roi(plane, roi, condition, offset, level, description)
            if stats is None:
                continue
            stats.normalization = normalization
            if not roi.is_background:
                apply_background(stats, background_stats)
                stats.normalized_mean = _normalized_mean(stats, background_mean, normalization)
            result.stats.append(stats)

            values = _roi_values(plane, roi, condition)
            if values is not None and values.size:
                corrected = values.astype(np.float64, copy=False) - float(offset)
                corrected = normalize_values(corrected, background_mean, normalization)
                result.samples[(roi.label, condition.name)] = _subsample(corrected, sample_limit)

            if stats.saturated_pixels:
                result.warnings.append(
                    f"{condition.name} / {roi.label}: {stats.saturated_pixels:,} saturated pixel(s) "
                    f"({stats.saturated_percent:.2f}%) — ratios and SNR are not trustworthy."
                )

        if background_roi is None:
            result.warnings.append(
                f"{condition.name}: no background ROI applies to it — "
                "subtraction, signal/background and SNR are unavailable."
            )

    if not any_background and normalization != "None":
        result.warnings.append(
            f"“{normalization}” was requested but no background ROI was found, so values are unchanged."
        )
    result.plane_description = plane_notes[0] if plane_notes else ""
    return result


def _normalized_mean(stats: RoiStats, background_mean: float | None, normalization: str) -> float | None:
    """The ROI mean expressed under the chosen normalisation."""
    if normalization == "None" or background_mean is None:
        return None
    if normalization == "Subtract background":
        return stats.corrected_mean
    if normalization == "Divide by background":
        return _ratio(stats.mean, background_mean)
    return None


def _roi_values(plane: np.ndarray, roi: RoiSpec, condition: ConditionSpec) -> np.ndarray | None:
    """Raw pixel values of *roi* on an already-extracted 2D *plane*."""
    if plane.ndim != 2:
        return None
    vertices = world_to_data(roi.vertices_world, condition.scale, condition.translate)
    mask = polygon_mask(vertices, plane.shape)
    if not mask.any():
        return np.empty(0, dtype=plane.dtype)
    return plane[mask]


def _measure_roi(
    plane: np.ndarray,
    roi: RoiSpec,
    condition: ConditionSpec,
    offset: float,
    saturation_level: float | None,
    plane_description: str,
) -> RoiStats | None:
    raw = _roi_values(plane, roi, condition)
    if raw is None:
        return None
    corrected = raw.astype(np.float64, copy=False) - float(offset)
    stats = measure_values(
        corrected,
        roi_name=roi.label,
        condition=condition.name,
        layer_name=condition.layer_name,
        offset=offset,
        saturation_level=saturation_level,
        raw_for_saturation=raw,
        is_background=roi.is_background,
        plane=plane_description,
    )
    stats.roi_source = roi.name
    return stats


def stats_dataframe(stats: Sequence[RoiStats]):
    """A :class:`pandas.DataFrame` of *stats* with readable column names."""
    import pandas as pd

    if not stats:
        return pd.DataFrame(columns=[STAT_LABELS[c] for c in STAT_COLUMNS])
    frame = pd.DataFrame([entry.as_row() for entry in stats])
    ordered = [column for column in STAT_COLUMNS if column in frame.columns]
    return frame[ordered].rename(columns=STAT_LABELS)
