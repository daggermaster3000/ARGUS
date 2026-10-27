"""Count the cells in the junction band, and say which channels each is positive for.

The band (see :mod:`microscopy_viewer.fusion` and
:mod:`microscopy_viewer.junction`) is where cells that crossed from one organoid
into the other are looked for. This module does the looking:

1. **The band** is read from the sample's store, or found on the spot when the
   sample has none yet.
2. **The cells** are nuclei, segmented on the nuclear stain over the whole
   section — both organoids — in tiles (:func:`segment_section`). The cells
   of the band are those whose centroid is in it; the rest are measured and
   scored too, so a positive cell is placed wherever it is. With
   ``CountOptions.whole_section`` off only the band's bounding box is
   segmented: seconds instead of minutes, and nothing outside the band.
3. **Each cell is measured in every channel**, over its nucleus grown by a few
   micrometres: the label in an assembloid is cytoplasmic as often as nuclear,
   and the nucleus alone would miss it. Sizes stay the nucleus's own.
4. **Positive or not** is decided per channel against a threshold that is
   derived from the data and written down beside the count (see
   :func:`threshold_for`).

No Qt and no napari. Cellpose is reached through
:func:`microscopy_viewer.segmentation.segment_volume`, and can be replaced by
any function of the same shape — which is how the checks run without it.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from . import fusion as fu
from . import junction as jn
from .utils import get_logger

logger = get_logger("junction_analysis")

#: How a channel's threshold is derived.
METHOD_CELLS = "Above the cells: median + k × SD"
METHOD_BACKGROUND = "Above the background: median + k × SD"
METHOD_OTSU = "Otsu split of the cells"
METHOD_FIXED = "Fixed value"
METHOD_RING = "Ring contrast: cell ÷ its 3–6 µm ring"
METHODS = (METHOD_CELLS, METHOD_BACKGROUND, METHOD_OTSU, METHOD_FIXED, METHOD_RING)

#: The ring a cell's local background is read over: from this far out from the
#: nucleus to that far, in µm — beyond the region the cell is measured over, and
#: never into a neighbour (the same seeded growth). As in the validation's
#: ``count_cells.RING_UM``.
RING_UM = (3.0, 6.0)
#: Ring contrast a cell needs, with :data:`METHOD_RING`: its mean over its
#: ring's. 1.5 was the validation's (``positivity_metrics``, 21 Oct): 0.07 %
#: false positives deep in the other organoid, held out by position.
RING_RATIO = 1.5
#: Per-cell column: no pixel of the ring was free of other cells, so the
#: contrast is over the rings of the nearest cells that have one instead.
NO_RING_COLUMN = "No ring"
#: For a cell with no ring: how many of the nearest cells with one stand in,
#: and how far they may be, in µm.
RING_NEIGHBOURS = 8
RING_NEIGHBOUR_UM = 50.0

#: Which cells a threshold is derived from, for the rules that derive it from
#: cells (:data:`CELL_RULES`). The cells *counted* are those in the region
#: either way.
FROM_REGION = "The cells in the region"
FROM_SAMPLE = "All cells of the sample"
THRESHOLD_FROM = (FROM_REGION, FROM_SAMPLE)

#: The rules whose threshold comes from a population of cells.
CELL_RULES = (METHOD_CELLS, METHOD_OTSU)

#: Which per-cell figure is compared with the threshold.
STATISTICS = ("mean", "median", "P95", "max")

#: Per-cell figures reported for every channel, in column order.
CELL_METRICS = ("mean", "median", "SD", "CV", "P95", "max", "integrated")

#: Key the counted cells are stored under in the sample.
CELLS_KEY = "junction cells"

#: Side of the tiles a whole sample is segmented in, in pixels, when its cells
#: are wanted for a threshold. A slide scan handed to Cellpose in one piece —
#: fifty-seven megapixels — does not finish in half an hour; in tiles of this
#: size it is a couple of minutes. A cell cut by a tile edge is measured as two
#: halves, which a median over tens of thousands of cells does not notice.
TILE_PX = 2048
#: How much of its neighbours each tile is segmented with, in pixels: a cell
#: cut by the tile's edge is whole in one of the two, and counted in the one
#: holding its centre.
TILE_OVERLAP_PX = 96

#: A tile is segmented when at least this share of it is brighter, in the
#: nuclear stain, than the split between slide and tissue.
TILE_MIN_TISSUE = 0.02

#: Where a run's results go, inside the experiment folder: one dated folder
#: per run (:func:`run_folder`) holding the workbook (:data:`WORKBOOK_NAME`),
#: every cell of each section (:data:`CELLS_FOLDER`), a picture of each counted
#: region sorted by QC status (:data:`REGION_FOLDER`) and a picture of each
#: whole section with its positive cells (:data:`POSITIVE_FOLDER`).
RESULTS_FOLDER = "cells_analysis"
WORKBOOK_NAME = "counts.xlsx"
CELLS_FOLDER = "cells"
REGION_FOLDER = "region"
POSITIVE_FOLDER = "positive cells"
#: The whole section's label maps, two per sample: ``<sample> nuclei.tif``
#: (every cell segmented) and ``<sample> grown.tif`` (the same grown, as measured).
LABELS_FOLDER = "labels"
NUCLEI_SUFFIX = " nuclei.tif"
GROWN_SUFFIX = " grown.tif"
#: The region the sample was counted in, beside its label maps, and what the
#: sample's segmentation was made with, beside its cells: together with the
#: cells table, all a later run needs to score the sample again without
#: segmenting it (:func:`segmented_samples`, :func:`rescore`).
REGION_SUFFIX = " region.tif"
INFO_SUFFIX = ".json"
README_NAME = "README.txt"

#: Longest side of the pictures, in pixels: the counted region, and the whole
#: section. A 7,600 px slide scan at 2,500 px is ~1 µm a pixel — a nucleus is
#: ten pixels across, its ring clear.
REGION_PICTURE_PX = 2000
SECTION_PICTURE_PX = 2500

#: Threads that measure the tiles already segmented while the GPU segments the
#: next one. Measuring a 2240 px tile in three channels is about four seconds
#: of CPU; Cellpose on it about three of GPU.
MEASURE_THREADS = 3

#: Pixels kept round the band when it is cropped, in µm: room for a cell lying
#: on the band's edge to be segmented whole.
CROP_MARGIN_UM = 30.0


@dataclass
class CountOptions:
    """How the cells in the band are found, measured and called positive."""

    #: The hand-drawn region where a sample has one, else the automatic band.
    band_key: str = fu.MANUAL_KEY
    #: When set, the hand-drawn region is the first stored junction region whose
    #: name contains this (any case) — whoever drew it and whatever they named
    #: it — instead of exactly *band_key*; the automatic one is the fallback.
    region_keyword: str = ""
    #: What a review said about each sample, by name (``Status``, ``Decision``,
    #: ``Note`` of ``review.csv``); carried into the tables.
    review: dict[str, dict[str, str]] = field(default_factory=dict)
    #: The nuclear stain, by name or index; empty picks DAPI / Hoechst / 405.
    nuclear_channel: int | str = ""
    method: str = METHOD_CELLS
    #: Standard deviations above the centre, for the two "median + k × SD" rules.
    k: float = 3.0
    #: With :data:`METHOD_RING`: the contrast a cell needs, cell ÷ its ring.
    ring_ratio: float = RING_RATIO
    statistic: str = "mean"
    #: The cells the threshold is derived from: those in the region, or every
    #: cell of the sample — which means segmenting the whole section.
    threshold_from: str = FROM_REGION
    #: Thresholds for :data:`METHOD_FIXED`, by channel name. A channel that is
    #: not listed falls back to :data:`METHOD_CELLS`.
    fixed: dict[str, float] = field(default_factory=dict)
    #: How far each nucleus is grown to make the region it is measured over.
    expand_um: float = 3.0
    #: Find the junction when the sample has no stored band.
    find_junction: bool = True
    #: Give every cell its distance to the junction line (positive into the
    #: band) and to the tissue edge (positive inside); the edge needs the
    #: section's overview, a few seconds a sample.
    distances: bool = True
    #: Segment the whole section (both organoids) and score every cell in it
    #: against the region's thresholds, with its distances
    #: (``JunctionCounts.section_cells``); the region's cells are taken from
    #: the same pass. Off: only the region's bounding box is segmented — seconds
    #: instead of ~40 s a slide scan, and nothing known outside the region.
    whole_section: bool = True
    #: With *whole_section*: keep every cell of the section, not only those
    #: positive for a channel or in the region. ~100,000 rows a section.
    keep_section: bool = True
    #: Folder each sample's section cells are written to, one CSV per sample
    #: (``<sample>.csv``), as soon as the sample is done; ``None`` writes none.
    cells_dir: Path | None = None
    #: Folder each sample's whole-section picture is written to
    #: (``<sample>.png``: the stains, the region, a ring round every positive
    #: cell); ``None`` writes none.
    pictures_dir: Path | None = None
    #: Folder each sample's whole-section label maps are written to (see
    #: :data:`LABELS_FOLDER`); ``None`` writes none. Implies keeping them while
    #: the sample is analysed.
    labels_dir: Path | None = None
    #: Keep the label maps of the whole section — every nucleus, and grown —
    #: for drawing (``JunctionCounts.section_labels`` / ``section_regions``).
    #: ~250 MB each on a slide scan: for a single sample, not a batch.
    full_labels: bool = False
    #: Store the counted cells with the sample as a label map.
    save_cells: bool = False
    cells_key: str = CELLS_KEY
    #: ``SegmentationSettings`` for the nuclei; ``None`` takes the defaults.
    settings: Any = None
    #: Options for finding the junction, when it has to be found.
    junction: Any = None
    #: Folder for the QC pictures, sorted into ``ok/``, ``check/`` and
    #: ``fail/``; ``None`` writes none.
    qc_dir: Path | None = None


@dataclass
class ChannelCount:
    """One channel's result in one sample."""

    channel: str
    n_cells: int = 0
    n_positive: int = 0
    threshold: float = float("nan")
    method: str = ""
    statistic: str = "mean"
    #: Median and spread of the tissue between the cells, in this channel.
    background: float = float("nan")
    background_sd: float = float("nan")
    #: Centre and spread of the cells the threshold was derived from — what
    #: the default rule uses — which cells those were, and how many.
    cells_median: float = float("nan")
    cells_sd: float = float("nan")
    threshold_from: str = FROM_REGION
    n_threshold_cells: int = 0
    per_mm2: float = float("nan")
    per_mm_junction: float = float("nan")
    #: With the whole section segmented: its cells, those positive anywhere
    #: in it, and where the positive ones lie — the median of their distance to
    #: the junction line (positive on the band's side) and to the tissue edge.
    n_cells_section: int = 0
    n_positive_section: float = float("nan")
    median_line_um: float = float("nan")
    median_edge_um: float = float("nan")
    #: The nuclear stain: every cell was found on it, so it is not scored.
    reference: bool = False

    @property
    def percent(self) -> float:
        return 100.0 * self.n_positive / self.n_cells if self.n_cells else float("nan")

    @property
    def percent_section(self) -> float:
        if not self.n_cells_section or not np.isfinite(self.n_positive_section):
            return float("nan")
        return 100.0 * self.n_positive_section / self.n_cells_section

    @property
    def enrichment(self) -> float:
        """How many times more often a cell is positive in the region than in the whole section."""
        whole = self.percent_section
        return self.percent / whole if np.isfinite(whole) and whole > 0 else float("nan")


@dataclass
class JunctionCounts:
    """Everything counted in one sample's band."""

    path: Path
    name: str
    n_cells: int = 0
    band_area_mm2: float = float("nan")
    junction_length_um: float = float("nan")
    band_source: str = ""
    #: Who drew the region counted in: "by hand", "junction finder", or "on screen".
    region_by: str = ""
    #: The quality status of the region counted in (a hand-drawn one has none).
    status: str = ""
    flags: str = ""
    #: What the junction finder said about the sample, whichever region was
    #: counted in: its status, its flags, and whether its second search (for
    #: sections without a dark line) made the line. Empty when it never ran.
    finder_status: str = ""
    finder_flags: str = ""
    second_search: str = ""
    #: What the review decided, when the samples came from one.
    review_status: str = ""
    review_decision: str = ""
    review_note: str = ""
    channels: list[ChannelCount] = field(default_factory=list)
    #: One row per cell (a DataFrame); ``None`` when the run failed.
    cells: Any = None
    #: The nuclei and their grown regions on the cropped grid, and where the
    #: crop sits in the image: ``(row, column)`` of its first pixel.
    labels: np.ndarray | None = None
    regions: np.ndarray | None = None
    origin: tuple[int, int] = (0, 0)
    voxel_yx: tuple[float, float] = (1.0, 1.0)
    shape_yx: tuple[int, int] = (0, 0)
    error: str = ""
    elapsed_s: float = 0.0
    warnings: list[str] = field(default_factory=list)
    #: The QC picture of this count, when one was written.
    picture: Path | None = None
    #: The failure was the GPU's: no later sample can be segmented either.
    gpu_fault: bool = False
    #: With ``CountOptions.whole_section``: the section's cells (a DataFrame),
    #: with their place; ``None`` otherwise, and after a batch has written them
    #: to *cells_file*.
    section_cells: Any = None
    n_cells_section: int = 0
    cells_file: Path | None = None
    #: With ``CountOptions.full_labels``: every nucleus of the section and the
    #: same grown, on the image's grid, under the labels of *section_cells*.
    section_labels: np.ndarray | None = None
    section_regions: np.ndarray | None = None
    #: The whole-section picture, when one was written.
    section_picture: Path | None = None
    #: The whole section's label maps on disk, nuclei and grown, when written.
    nuclei_file: Path | None = None
    grown_file: Path | None = None

    @property
    def ok(self) -> bool:
        return not self.error

    @property
    def qc(self) -> str:
        """Which QC folder the sample belongs in: ``ok``, ``check`` or ``fail``.

        Failed when it could not be counted or its junction failed; to check
        when the junction was flagged or the region holds no cell. A region
        drawn by hand carries no status of its own and counts as good.
        """
        if self.error or self.status == jn.STATUS_FAIL:
            return jn.STATUS_FAIL
        if self.status == jn.STATUS_CHECK or not self.n_cells:
            return jn.STATUS_CHECK
        return jn.STATUS_OK

    @property
    def qc_reason(self) -> str:
        parts = [self.error, self.flags, "" if self.n_cells or self.error else "no cells in the region"]
        return "; ".join(part for part in parts if part) or "no flags"

    def positive_labels(self, channel: str) -> np.ndarray:
        """Labels of the cells positive for *channel*."""
        column = f"{channel} positive"
        if self.cells is None or column not in self.cells:
            return np.zeros(0, dtype=np.int64)
        return self.cells.loc[self.cells[column], "Label"].to_numpy(dtype=np.int64)


# ---------------------------------------------------------------------------
# Measuring
# ---------------------------------------------------------------------------


def label_stats(labels: np.ndarray, image: np.ndarray, ids: Sequence[int]) -> dict[str, np.ndarray]:
    """Per-label statistics of *image*, for the labels in *ids*, in that order.

    One sort of the labelled pixels gives every figure, the quantiles included,
    so ten thousand cells cost one pass and not ten thousand.
    """
    ids = np.asarray(ids, dtype=np.int64)
    empty = {key: np.full(len(ids), np.nan) for key in ("n", *CELL_METRICS)}
    flat = np.asarray(labels).reshape(-1)
    inside = np.flatnonzero(flat)
    if inside.size == 0 or ids.size == 0:
        empty["n"] = np.zeros(len(ids))
        return empty
    owner = flat[inside].astype(np.int64)
    values = np.asarray(image, dtype=np.float64).reshape(-1)[inside]
    order = np.lexsort((values, owner))
    owner, values = owner[order], values[order]

    first = np.searchsorted(owner, ids, side="left")
    last = np.searchsorted(owner, ids, side="right")
    count = (last - first).astype(np.float64)
    present = count > 0
    running = np.concatenate(([0.0], np.cumsum(values)))
    squares = np.concatenate(([0.0], np.cumsum(values * values)))
    total = running[last] - running[first]
    safe = np.where(present, count, 1.0)
    mean = total / safe
    spread = np.sqrt(np.maximum((squares[last] - squares[first]) / safe - mean * mean, 0.0))

    def quantile(q: float) -> np.ndarray:
        # Sorted within each label, so a quantile is a position in its run.
        position = first + np.round((safe - 1) * q).astype(np.int64)
        return values[np.clip(position, 0, len(values) - 1)]

    out = {
        "n": count,
        "mean": mean,
        "median": quantile(0.5),
        "SD": spread,
        "CV": np.divide(spread, mean, out=np.full_like(mean, np.nan), where=mean != 0),
        "P95": quantile(0.95),
        "max": values[np.clip(last - 1, 0, len(values) - 1)],
        "integrated": total,
    }
    for key in CELL_METRICS:
        out[key] = np.where(present, out[key], np.nan)
    return out


def ring_labels(nuclei: np.ndarray, voxel_yx: Sequence[float], regions: np.ndarray | None = None,
                expand_um: float = 0.0) -> np.ndarray:
    """Each nucleus's ring, :data:`RING_UM` out, under the nucleus's label.

    Grown as the measuring regions are, so a ring never takes a neighbour's
    pixels; a nucleus packed in among others can have no ring at all.
    *regions* grown by *expand_um* are reused when that is the ring's inside.
    """
    from . import segmentation as sg

    inner = (regions if regions is not None and abs(expand_um - RING_UM[0]) < 1e-9
             else sg.expand_labels(nuclei, voxel_yx, RING_UM[0]))
    outer = sg.expand_labels(nuclei, voxel_yx, RING_UM[1])
    return np.where(inner == 0, outer, 0)


def ring_background(ys_um: np.ndarray, xs_um: np.ndarray, ring_mean: np.ndarray) -> np.ndarray:
    """Each cell's local background: its ring's mean, or, with no ring, the median ring of its neighbours.

    A cell packed in among others has no ring of its own; the
    :data:`RING_NEIGHBOURS` nearest cells that have one, within
    :data:`RING_NEIGHBOUR_UM`, stand in — the same kind of measurement, from
    the same neighbourhood. With none that near, NaN: the cell is not called.
    """
    ring = np.asarray(ring_mean, dtype=np.float64)
    have = np.isfinite(ring) & (ring > 0)
    background = np.where(have, ring, np.nan)
    missing = np.flatnonzero(~have)
    if not missing.size or not have.any():
        return background
    from scipy.spatial import cKDTree

    points = np.column_stack([np.asarray(ys_um, float), np.asarray(xs_um, float)])
    tree = cKDTree(points[have])
    k = min(RING_NEIGHBOURS, int(have.sum()))
    distance, index = tree.query(points[missing], k=k, distance_upper_bound=RING_NEIGHBOUR_UM)
    distance, index = distance.reshape(len(missing), k), index.reshape(len(missing), k)
    values = np.where(np.isfinite(distance), ring[have][np.minimum(index, int(have.sum()) - 1)], np.nan)
    with np.errstate(all="ignore"):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            background[missing] = np.nanmedian(values, axis=1)
    return background


def ring_contrast(cell_mean: np.ndarray, background: np.ndarray) -> np.ndarray:
    """Cell over its local background (:func:`ring_background`); NaN where there is none."""
    cell = np.asarray(cell_mean, dtype=np.float64)
    background = np.asarray(background, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return cell / np.where(np.isfinite(background) & (background > 0), background, np.nan)


def add_ring_contrast(table, names: Sequence[str]) -> None:
    """Add ``<channel> ring contrast`` for every channel of *table* that has its ``ring mean``."""
    ys = table["Centroid Y (µm)"].to_numpy(dtype=float)
    xs = table["Centroid X (µm)"].to_numpy(dtype=float)
    for name in names:
        if f"{name} ring mean" not in table:
            continue
        background = ring_background(ys, xs, table[f"{name} ring mean"].to_numpy(dtype=float))
        table[f"{name} ring contrast"] = ring_contrast(table[f"{name} mean"].to_numpy(dtype=float), background)


def robust_centre_and_spread(values: np.ndarray) -> tuple[float, float]:
    """Median, and the standard deviation the MAD stands for.

    Used wherever "k × SD" is asked for: a handful of bright cells is what is
    being looked for, and an ordinary SD is inflated by exactly those.
    """
    data = np.asarray(values, dtype=np.float64)
    data = data[np.isfinite(data)]
    if data.size == 0:
        return float("nan"), float("nan")
    centre = float(np.median(data))
    return centre, float(np.median(np.abs(data - centre)) * 1.4826)


def threshold_for(
    channel: str,
    cell_values: np.ndarray,
    background: tuple[float, float],
    options: CountOptions,
) -> tuple[float, str]:
    """The value a cell has to exceed to be positive, and the rule that gave it.

    * **Above the cells** — median + k robust SDs of the cells' own values.
      Right when most cells in the band are negative, which is the case for
      cells that had to migrate to be there. It adapts to the staining, and to
      the autofluorescence every cell shares.
    * **Above the background** — median + k robust SDs of the tissue between
      the cells. Stricter about what counts as signal at all, and blind to a
      dim haze that every cell carries.
    * **Otsu** — the split that best separates the cells into two groups. Right
      when there really are two; with one, it splits it anyway.
    * **Fixed** — a number, the same for every sample. The only rule under
      which two batches are compared on the same scale.
    """
    values = np.asarray(cell_values, dtype=np.float64)
    values = values[np.isfinite(values)]
    method = options.method
    if method == METHOD_RING:
        # In contrast units, not intensity: see :func:`positive_of`.
        return float(options.ring_ratio), METHOD_RING
    if method == METHOD_FIXED:
        if channel in options.fixed:
            return float(options.fixed[channel]), METHOD_FIXED
        method = METHOD_CELLS
    if method == METHOD_BACKGROUND:
        centre, spread = background
        if np.isfinite(centre) and np.isfinite(spread):
            return float(centre + options.k * spread), METHOD_BACKGROUND
        method = METHOD_CELLS
    if method == METHOD_OTSU:
        if values.size >= 10 and float(values.max()) > float(values.min()):
            from skimage.filters import threshold_otsu

            return float(threshold_otsu(values)), METHOD_OTSU
        method = METHOD_CELLS
    centre, spread = robust_centre_and_spread(values)
    return float(centre + options.k * spread), METHOD_CELLS


def positive_of(table, channel: str, count: "ChannelCount", statistic: str) -> np.ndarray:
    """Which rows of *table* are positive for *channel* under *count*'s rule and threshold."""
    if count.method == METHOD_RING:
        values = table[f"{channel} ring contrast"].to_numpy(dtype=np.float64)
        return np.isfinite(values) & (values >= count.threshold)
    values = table[f"{channel} {statistic}"].to_numpy(dtype=np.float64)
    return np.isfinite(values) & (values > count.threshold)


def measure_cells(
    nuclei: np.ndarray,
    regions: np.ndarray,
    images: dict[str, np.ndarray],
    voxel_yx: Sequence[float],
    band: np.ndarray,
    options: CountOptions | None = None,
    nuclear: str = "",
    origin: Sequence[int] = (0, 0),
    sample_values: dict[str, np.ndarray] | None = None,
):
    """Measure the cells whose centroid is in *band*. Returns ``(table, counts)``.

    *nuclei* are the cells as segmented and *regions* the same labels grown;
    sizes and positions come from the first and intensities from the second.
    *images* maps a channel name to its plane on the same grid. The table has
    one row per cell; *counts* one :class:`ChannelCount` per channel.

    With ``options.threshold_from`` set to :data:`FROM_SAMPLE`, the threshold
    of a channel comes from every cell in *nuclei*, in the region or not, and
    only the cells in the region are counted against it. A region where most
    cells are positive then no longer raises its own bar. The grid has to cover
    the whole sample for that to mean what it says — or *sample_values* gives
    the sample's cells instead: per channel, the compared figure of every cell,
    gathered elsewhere (:func:`sample_cell_values`).
    """
    import pandas as pd
    from scipy import ndimage

    options = options or CountOptions()
    voxel = tuple(float(v) for v in tuple(voxel_yx)[-2:])
    pixel_area = voxel[0] * voxel[1]
    nuclei = np.asarray(nuclei)
    band = np.asarray(band, dtype=bool)

    present = np.unique(nuclei[nuclei > 0]).astype(np.int64)
    if present.size:
        centres = np.asarray(ndimage.center_of_mass(nuclei > 0, nuclei, present)).reshape(-1, 2)
        rows = np.clip(np.round(centres[:, 0]).astype(int), 0, band.shape[0] - 1)
        cols = np.clip(np.round(centres[:, 1]).astype(int), 0, band.shape[1] - 1)
        inside = band[rows, cols]
        ids, centres = present[inside], centres[inside]
    else:
        inside = np.zeros(0, dtype=bool)
        ids, centres = present, np.zeros((0, 2))
    whole = options.threshold_from == FROM_SAMPLE

    area = np.asarray(ndimage.sum(nuclei > 0, nuclei, ids), dtype=np.float64) * pixel_area if ids.size else np.zeros(0)
    table = pd.DataFrame({
        "Label": ids,
        "Centroid Y (µm)": (centres[:, 0] + float(origin[0])) * voxel[0],
        "Centroid X (µm)": (centres[:, 1] + float(origin[1])) * voxel[1],
        "Nucleus area (µm²)": area,
        "Equivalent diameter (µm)": 2.0 * np.sqrt(area / np.pi),
    })

    between = band & (np.asarray(regions) == 0)
    if int(between.sum()) < 200:  # a band packed solid with cells
        between = band
    rings = ring_labels(nuclei, voxel, regions, options.expand_um) if ids.size else np.zeros_like(nuclei)
    ring_size = np.asarray(ndimage.sum(rings > 0, rings, ids)) if ids.size else np.zeros(0)
    table[NO_RING_COLUMN] = ring_size == 0
    counts: list[ChannelCount] = []
    positive_names: list[list[str]] = [[] for _ in range(len(ids))]
    for channel, image in images.items():
        if whole and sample_values is not None and channel in sample_values:
            stats = label_stats(regions, image, ids)
            population = np.asarray(sample_values[channel], dtype=np.float64)
        elif whole:
            everyone = label_stats(regions, image, present)
            stats = {metric: figures[inside] for metric, figures in everyone.items()}
            population = everyone[options.statistic]
        else:
            stats = label_stats(regions, image, ids)
            population = stats[options.statistic]
        for metric in CELL_METRICS:
            table[f"{channel} {metric}"] = stats[metric]
        centre, spread = robust_centre_and_spread(np.asarray(image, dtype=np.float64)[between])
        scale = spread if spread and np.isfinite(spread) and spread > 0 else np.nan
        table[f"{channel} SDs over background"] = (stats["mean"] - centre) / scale
        table[f"{channel} fold over background"] = stats["mean"] / centre if centre else np.nan
        table[f"{channel} ring mean"] = label_stats(rings, image, ids)["mean"] if ids.size else np.zeros(0)
        add_ring_contrast(table, [channel])

        count = ChannelCount(channel=channel, n_cells=int(len(ids)), statistic=options.statistic,
                             background=centre, background_sd=spread, reference=channel == nuclear)
        count.cells_median, count.cells_sd = robust_centre_and_spread(population)
        count.threshold_from = options.threshold_from if whole else FROM_REGION
        count.n_threshold_cells = int(len(population))
        if count.reference:
            # Every cell was found on this stain; scoring it would only show
            # the threshold rule cutting a single population in two.
            count.method = "reference stain — not scored"
        else:
            count.threshold, count.method = threshold_for(channel, population, (centre, spread), options)
            positive = positive_of(table, channel, count, options.statistic)
            table[f"{channel} positive"] = positive
            count.n_positive = int(positive.sum())
            for index in np.flatnonzero(positive):
                positive_names[index].append(channel)
        counts.append(count)
    table["Positive for"] = [" + ".join(names) if names else "none" for names in positive_names]
    return table, counts


# ---------------------------------------------------------------------------
# The QC picture
# ---------------------------------------------------------------------------

#: Colours of the rings round the positive cells, one per scored channel. None
#: is a colour a channel is shown in, so a ring is never lost in its own stain.
MARKER_COLOURS = ((255, 255, 0), (0, 255, 255), (255, 255, 255), (255, 150, 0))

#: Colours for channels whose name states no wavelength, in order.
_FALLBACK_RGB = ((0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.0, 1.0), (0.0, 1.0, 1.0))


def count_picture(result: JunctionCounts, images: dict[str, np.ndarray], band: np.ndarray,
                  size: int = REGION_PICTURE_PX):
    """The counted region as a picture: the stains, the region, the positive cells.

    A PIL image of the crop the cells were counted in. Every channel is shown
    in its wavelength's colour, stretched over the region; what lies outside
    the region is dimmed and its edge drawn white. Each cell positive for a
    channel has a ring in that channel's marker colour — a cell positive for
    two has two, one inside the other — and the caption gives the counts in the
    same colours, under the sample's status and what was flagged.
    """
    from PIL import Image, ImageDraw
    from scipy import ndimage

    from .loaders.layer_spec import color_for_wavelength, wavelength_from_name

    band = np.asarray(band, dtype=bool)
    step = max(1, int(np.ceil(max(band.shape) / float(size))))
    small = band[::step, ::step]
    rgb = np.zeros((*small.shape, 3), dtype=np.float32)
    fallback = iter(_FALLBACK_RGB * 4)
    for name, plane in images.items():
        nm = wavelength_from_name(name)
        colour = color_for_wavelength(nm)[1] if nm is not None else next(fallback)
        shade = jn._stretch(np.asarray(plane[::step, ::step], dtype=np.float32), small)
        rgb += shade[..., None] * np.asarray(colour, dtype=np.float32)
    rgb = np.clip(rgb, 0.0, 1.0)
    rgb[~small] *= 0.4
    pixels = (rgb * 255).astype(np.uint8)
    pixels[ndimage.binary_dilation(small) & ~small] = (255, 255, 255)
    image = Image.fromarray(pixels)

    scored = [count for count in result.channels if not count.reference]
    draw = ImageDraw.Draw(image)
    if result.cells is not None and len(result.cells):
        ys = result.cells["Centroid Y (µm)"].to_numpy() / result.voxel_yx[0] - result.origin[0]
        xs = result.cells["Centroid X (µm)"].to_numpy() / result.voxel_yx[1] - result.origin[1]
        for index, count in enumerate(scored):
            colour = MARKER_COLOURS[index % len(MARKER_COLOURS)]
            radius = 4 + 2 * index
            for row in np.flatnonzero(result.cells[f"{count.channel} positive"].to_numpy()):
                x, y = float(xs[row]) / step, float(ys[row]) / step
                draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=colour)

    strip = jn.CAPTION_PX + 16
    canvas = Image.new("RGB", (max(image.size[0], 640), image.size[1] + strip), "black")
    canvas.paste(image, (0, strip))
    jn.caption(canvas, result.name, result.qc, result.qc_reason)
    draw = ImageDraw.Draw(canvas)
    x = 4
    for text, colour in [(f"{result.n_cells} cells", (220, 220, 220))] + [
        (f"{count.channel}+ {count.n_positive} ({count.percent:.1f} %)", MARKER_COLOURS[index % len(MARKER_COLOURS)])
        for index, count in enumerate(scored)
    ]:
        draw.text((x, jn.CAPTION_PX - 1), text, fill=colour)
        x += int(draw.textlength(text)) + 18
    return canvas


def _thumbnail(spec: Any, step: int) -> np.ndarray:
    """One channel flattened over Z, every *step*-th pixel."""
    array = spec.data[0] if getattr(spec, "multiscale", False) else spec.data
    stack = array.reshape((-1, *array.shape[-2:]))
    best: np.ndarray | None = None
    for z in range(int(stack.shape[0])):
        plane = np.asarray(stack[z][::step, ::step])
        best = plane if best is None else np.maximum(best, plane)
    return best


def section_picture(result: JunctionCounts, specs: Sequence[Any], band: np.ndarray, line=None,
                    size: int = SECTION_PICTURE_PX):
    """The whole section as a picture: the stains, the region, a ring round each positive cell.

    Every channel in its wavelength's colour, stretched over the tissue; the
    region's edge in white and the junction line in yellow; each cell positive
    for a channel ringed in that channel's marker colour (as in
    :func:`count_picture`), wherever it is in the section. The caption gives,
    per channel, the positive cells in the region and in the whole section.
    """
    from PIL import Image, ImageDraw
    from scipy import ndimage

    from .loaders.layer_spec import color_for_wavelength, wavelength_from_name

    band = np.asarray(band, dtype=bool)
    step = max(1, int(np.ceil(max(band.shape) / float(size))))
    small = band[::step, ::step]
    planes = {str(spec.channel_name or spec.name): _thumbnail(spec, step).astype(np.float32) for spec in specs}
    planes = {name: plane[:small.shape[0], :small.shape[1]] for name, plane in planes.items()}
    nuclear = next((name for name in planes if any(h in name.lower() for h in jn.NUCLEAR_HINTS)), next(iter(planes)))
    stain = planes[nuclear]
    tissue = stain > (np.percentile(stain, 50) if stain.size else 0)
    if not tissue.any():
        tissue = np.ones(stain.shape, dtype=bool)
    rgb = np.zeros((*small.shape, 3), dtype=np.float32)
    fallback = iter(_FALLBACK_RGB * 4)
    for name, plane in planes.items():
        nm = wavelength_from_name(name)
        colour = color_for_wavelength(nm)[1] if nm is not None else next(fallback)
        rgb += jn._stretch(plane, tissue)[..., None] * np.asarray(colour, dtype=np.float32)
    pixels = (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)
    pixels[ndimage.binary_dilation(small, iterations=2) & ~small] = (255, 255, 255)
    image = Image.fromarray(pixels)
    draw = ImageDraw.Draw(image)
    voxel = result.voxel_yx
    if line is not None and len(np.asarray(line).reshape(-1, 2)) >= 2:
        points = [(float(x) / voxel[1] / step, float(y) / voxel[0] / step) for y, x in np.asarray(line).reshape(-1, 2)]
        draw.line(points, fill=(255, 230, 0), width=2)

    scored = [count for count in result.channels if not count.reference]
    table = result.section_cells
    if table is not None and len(table):
        ys = table["Centroid Y (µm)"].to_numpy() / voxel[0] / step
        xs = table["Centroid X (µm)"].to_numpy() / voxel[1] / step
        for index, count in enumerate(scored):
            column = f"{count.channel} positive"
            if column not in table:
                continue
            colour = MARKER_COLOURS[index % len(MARKER_COLOURS)]
            radius = 4 + 2 * index
            for row in np.flatnonzero(table[column].to_numpy()):
                x, y = float(xs[row]), float(ys[row])
                draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=colour)

    strip = jn.CAPTION_PX + 16
    canvas = Image.new("RGB", (max(image.size[0], 900), image.size[1] + strip), "black")
    canvas.paste(image, (0, strip))
    jn.caption(canvas, result.name, result.qc,
               f"whole section: {result.n_cells_section} cells, {result.n_cells} in the region (white outline); "
               "junction line yellow")
    draw = ImageDraw.Draw(canvas)
    x = 4
    for index, count in enumerate(scored):
        text = (f"{count.channel}+ {count.n_positive_section:.0f} in section ({count.percent_section:.1f} %), "
                f"{count.n_positive} in region ({count.percent:.1f} %)")
        draw.text((x, jn.CAPTION_PX - 1), text, fill=MARKER_COLOURS[index % len(MARKER_COLOURS)])
        x += int(draw.textlength(text)) + 18
    return canvas


def _write_picture(result: JunctionCounts, folder: Path, images, band) -> None:
    """File the sample's QC picture under its status. Never raises."""
    try:
        if result.error or images is None or band is None:
            picture = jn.problem_picture(result.name, result.error or "nothing was counted")
        else:
            picture = count_picture(result, images, band)
        result.picture = jn.file_picture(folder, result.name, result.qc, picture)
    except Exception:
        logger.exception("could not write the QC picture of %s", result.name)


# ---------------------------------------------------------------------------
# One sample
# ---------------------------------------------------------------------------


def _segment(image: np.ndarray, voxel_yx: Sequence[float], settings: Any, progress=None) -> np.ndarray:
    """Nuclei of one plane, through the segmentation engine."""
    from dataclasses import replace

    from . import segmentation as sg

    settings = settings or sg.SegmentationSettings()
    # A crop is already small, and it is grown here rather than by the engine.
    settings = replace(settings, expand_um=0.0)
    return sg.segment_volume(image, voxel_yx, settings=settings, progress=progress).masks


def _plane(spec: Any, window: tuple[slice, slice]) -> np.ndarray:
    """One channel flattened over Z, inside *window*."""
    array = spec.data[0] if getattr(spec, "multiscale", False) else spec.data
    stack = array.reshape((-1, *array.shape[-2:]))
    best: np.ndarray | None = None
    for z in range(int(stack.shape[0])):
        plane = np.asarray(stack[z][window])
        best = plane if best is None else np.maximum(best, plane)
    return best


def sample_cell_values(
    specs: Sequence[Any],
    names: Sequence[str],
    nuclear: int,
    voxel_yx: Sequence[float],
    options: CountOptions,
    segment: Callable[..., np.ndarray] | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, np.ndarray]:
    """The compared figure of every cell in the sample, per channel (see :func:`section_cells`)."""
    table = section_cells(specs, names, nuclear, voxel_yx, options, segment, progress)
    return {name: table[f"{name} {options.statistic}"].to_numpy(dtype=np.float64) for name in names}


@dataclass
class SectionCells:
    """The whole section's cells: one row each, and their labels over a crop."""

    #: One row per cell: ``Label``, centroid and size, and every figure of
    #: :data:`CELL_METRICS` per channel. Labels are unique over the section.
    table: Any
    #: The nuclei and their grown regions over the crop asked for, with the
    #: section's labels; ``None`` when no crop was asked for.
    labels: np.ndarray | None = None
    regions: np.ndarray | None = None


def section_cells(
    specs: Sequence[Any],
    names: Sequence[str],
    nuclear: int,
    voxel_yx: Sequence[float],
    options: CountOptions,
    segment: Callable[..., np.ndarray] | None = None,
    progress: Callable[[str], None] | None = None,
):
    """Every cell of the section, as a table (see :func:`segment_section`)."""
    return segment_section(specs, names, nuclear, voxel_yx, options, segment, progress).table


def segment_section(
    specs: Sequence[Any],
    names: Sequence[str],
    nuclear: int,
    voxel_yx: Sequence[float],
    options: CountOptions,
    segment: Callable[..., np.ndarray] | None = None,
    progress: Callable[[str], None] | None = None,
    crop: tuple[slice, slice] | None = None,
) -> SectionCells:
    """Every cell of the section: where it is, its size, and its figures in each channel.

    The whole section is segmented, in tiles of :data:`TILE_PX` so that it
    finishes; tiles of bare slide are skipped. Each tile is segmented with
    :data:`TILE_OVERLAP_PX` of its neighbours around it, and keeps the cells
    whose centre is in the tile itself, so a cell cut by a tile's edge is
    counted once and whole. Each cell is measured as the cells in the region
    are — over its nucleus grown by ``options.expand_um`` — giving per channel
    every figure of :data:`CELL_METRICS`. ``Centroid Y (µm)`` /
    ``Centroid X (µm)`` are on the image's grid.

    With *crop*, the cells are also pasted, nuclei and grown regions, onto a
    label map of that window, under the same labels as the table's: the
    region's cells, measured from the same segmentation.
    """
    import pandas as pd
    from scipy import ndimage

    from . import segmentation as sg

    say = progress or (lambda _text: None)
    first = specs[0].data[0] if getattr(specs[0], "multiscale", False) else specs[0].data
    height, width = (int(n) for n in first.shape[-2:])
    whole = (slice(0, height), slice(0, width))
    stain = _plane(specs[nuclear], whole)
    coarse = np.asarray(stain[::8, ::8], dtype=np.float64)
    split = float("-inf")
    if coarse.size and float(coarse.max()) > float(coarse.min()):
        from skimage.filters import threshold_otsu

        split = float(threshold_otsu(coarse))
    tiles = [(slice(y, min(y + TILE_PX, height)), slice(x, min(x + TILE_PX, width)))
             for y in range(0, height, TILE_PX) for x in range(0, width, TILE_PX)]
    tiles = [tile for tile in tiles if float((stain[tile][::8, ::8] > split).mean()) >= TILE_MIN_TISSUE]

    voxel = tuple(float(v) for v in tuple(voxel_yx)[-2:])
    parts: dict[str, list[np.ndarray]] = {}
    pasted = pasted_regions = None
    if crop is not None:
        pasted = np.zeros((crop[0].stop - crop[0].start, crop[1].stop - crop[1].start), dtype=np.int32)
        pasted_regions = np.zeros_like(pasted)

    def add(key, values):
        parts.setdefault(key, []).append(np.asarray(values, dtype=np.float64))

    def measure(tile, window, nuclei, images):
        """One tile's own cells, measured: runs on a worker thread, no file access."""
        ids = np.unique(nuclei[nuclei > 0]).astype(np.int64)
        if not ids.size:
            return None
        centres = np.asarray(ndimage.center_of_mass(nuclei > 0, nuclei, ids)).reshape(-1, 2)
        cy, cx = centres[:, 0] + window[0].start, centres[:, 1] + window[1].start
        own = (cy >= tile[0].start) & (cy < tile[0].stop) & (cx >= tile[1].start) & (cx < tile[1].stop)
        if not own.any():
            return None
        ids, cy, cx = ids[own], cy[own], cx[own]
        regions = sg.expand_labels(nuclei, voxel, options.expand_um) if options.expand_um > 0 else nuclei
        area = np.asarray(ndimage.sum(nuclei > 0, nuclei, ids)) * voxel[0] * voxel[1]
        rings = ring_labels(nuclei, voxel, regions, options.expand_um)
        no_ring = np.asarray(ndimage.sum(rings > 0, rings, ids)) == 0
        figures = {"Centroid Y (µm)": cy * voxel[0], "Centroid X (µm)": cx * voxel[1],
                   "Nucleus area (µm²)": area, "Equivalent diameter (µm)": 2.0 * np.sqrt(area / np.pi),
                   NO_RING_COLUMN: no_ring.astype(np.float64)}
        for name, image in images.items():
            stats = label_stats(regions, image, ids)
            for figure in CELL_METRICS:
                figures[f"{name} {figure}"] = stats[figure]
            figures[f"{name} ring mean"] = label_stats(rings, image, ids)["mean"]
        return ids, figures, (nuclei, regions) if crop is not None else None

    found = 0

    def collect(window, measured) -> None:
        """Number a measured tile's cells after those before it, in tile order, and keep them."""
        nonlocal found
        if measured is None:
            return
        ids, figures, maps = measured
        labels = np.arange(found + 1, found + 1 + ids.size, dtype=np.int64)
        add("Label", labels)
        for key, values in figures.items():
            add(key, values)
        if maps is not None:
            _paste(pasted, pasted_regions, crop, window, maps[0], maps[1], ids, labels)
        found += int(ids.size)

    # The GPU segments on this thread and the files are read here (their
    # handles belong to this thread); the measuring of the tiles already
    # segmented goes on meanwhile on worker threads. Collected in tile order,
    # so the labels are the same as one tile at a time.
    from concurrent.futures import ThreadPoolExecutor

    pending: list[tuple[tuple[slice, slice], Any]] = []
    with ThreadPoolExecutor(max_workers=MEASURE_THREADS) as pool:
        for index, tile in enumerate(tiles, start=1):
            while pending and (pending[0][1].done() or len(pending) > MEASURE_THREADS):
                window_done, future = pending.pop(0)
                collect(window_done, future.result())
            say(f"the section's cells: tile {index} of {len(tiles)} ({found} so far)")
            window = (slice(max(tile[0].start - TILE_OVERLAP_PX, 0), min(tile[0].stop + TILE_OVERLAP_PX, height)),
                      slice(max(tile[1].start - TILE_OVERLAP_PX, 0), min(tile[1].stop + TILE_OVERLAP_PX, width)))
            nuclei = np.asarray((segment or _segment)(stain[window], voxel, options.settings, None))
            if not np.any(nuclei):
                continue
            images = {name: stain[window] if position == nuclear else _plane(specs[position], window)
                      for position, name in enumerate(names)}
            pending.append((window, pool.submit(measure, tile, window, nuclei, images)))
        for window_done, future in pending:
            collect(window_done, future.result())
    columns = ["Label", "Centroid Y (µm)", "Centroid X (µm)", "Nucleus area (µm²)", "Equivalent diameter (µm)",
               NO_RING_COLUMN,
               *(f"{name} {figure}" for name in names for figure in (*CELL_METRICS, "ring mean"))]
    table = pd.DataFrame({key: np.concatenate(parts[key]) if key in parts else np.zeros(0) for key in columns})
    table["Label"] = table["Label"].astype(np.int64)
    table[NO_RING_COLUMN] = table[NO_RING_COLUMN].astype(bool)
    # Over the whole section, so a packed cell's neighbours may lie in the next tile.
    add_ring_contrast(table, names)
    return SectionCells(table=table, labels=pasted, regions=pasted_regions)


def _paste(pasted: np.ndarray, pasted_regions: np.ndarray, crop: tuple[slice, slice], window: tuple[slice, slice],
           nuclei: np.ndarray, regions: np.ndarray, ids: np.ndarray, labels: np.ndarray) -> None:
    """Copy one tile's own cells onto the crop, under their section labels; never over another cell."""
    y0, y1 = max(window[0].start, crop[0].start), min(window[0].stop, crop[0].stop)
    x0, x1 = max(window[1].start, crop[1].start), min(window[1].stop, crop[1].stop)
    if y0 >= y1 or x0 >= x1:
        return
    lut = np.zeros(int(max(nuclei.max(), regions.max())) + 1, dtype=np.int32)
    lut[ids] = labels
    source = (slice(y0 - window[0].start, y1 - window[0].start), slice(x0 - window[1].start, x1 - window[1].start))
    target = (slice(y0 - crop[0].start, y1 - crop[0].start), slice(x0 - crop[1].start, x1 - crop[1].start))
    for onto, cells in ((pasted, nuclei), (pasted_regions, regions)):
        mapped = lut[np.asarray(cells[source])]
        view = onto[target]
        free = (mapped > 0) & (view == 0)
        view[free] = mapped[free]


def region_candidates(path: str | Path, options: CountOptions) -> list[str]:
    """The stored regions to try for a sample, in order: hand-drawn first, then automatic.

    With ``options.region_keyword``, every stored junction region (one that
    keeps its line) whose name contains the keyword, in name order — so with
    several the first is taken — then the automatic one. Without, the exact
    ``options.band_key`` and its fallback (:func:`fusion.region_keys`).
    """
    from . import ims_store

    keyword = options.region_keyword.strip().lower()
    if not keyword:
        return list(fu.region_keys(options.band_key))
    drawn = [key for key in ims_store.list_labels(path)
             if keyword in key.lower() and key != fu.DEFAULT_KEY
             and "line_yx_um" in ims_store.label_attrs(path, key)]
    return [*drawn, fu.DEFAULT_KEY]


def finder_record(path: str | Path) -> dict[str, str]:
    """What the junction finder stored with its region: status, flags, second search."""
    from . import ims_store

    attrs = ims_store.label_attrs(path, fu.DEFAULT_KEY)
    if not attrs:
        return {}
    second = attrs.get("second_search")
    return {"status": ims_store._text(attrs.get("status", "")), "flags": ims_store._text(attrs.get("flags", "")),
            "second_search": "" if second is None else ("yes" if float(second) else "no")}


def _stored_band(path: Path, keys: Sequence[str] | str, shape_yx: tuple[int, int]):
    """The first stored band of *keys* that fits, its junction length and what was said about it."""
    from . import ims_store

    for used in ([keys] if isinstance(keys, str) else keys):
        if not used:
            continue
        labels, attrs = ims_store.load_labels(path, used)
        if labels is not None and tuple(labels.shape[-2:]) == tuple(shape_yx) and np.any(labels):
            break
    else:
        return None, {}
    details: dict[str, Any] = {"source": "stored", "region": used,
                               "by": "junction finder" if used == fu.DEFAULT_KEY else "by hand"}
    if "line_yx_um" in attrs:
        line = np.asarray(attrs["line_yx_um"], dtype=float).reshape(-1, 2)
        if len(line) >= 2:
            details["length"] = fu.line_length(fu.fit_spline(line, 5.0))
            details["line"] = line
    for name in ("status", "flags"):
        if name in attrs:
            details[name] = ims_store._text(attrs[name])
    return np.asarray(labels).reshape(labels.shape[-2:]) > 0, details


#: Per-cell columns giving where the cell is.
LINE_COLUMN = "Distance to the junction line (µm)"
BESIDE_COLUMN = "Beside the junction line"
EDGE_COLUMN = "Distance to the tissue edge (µm)"


#: Per-cell column of the section's cells: whether the cell's centre is in the region.
IN_REGION_COLUMN = "In region"


def score_section(section, counts: Sequence[ChannelCount], band: np.ndarray,
                  voxel: Sequence[float], options: CountOptions, region_labels=None):
    """The section's cells scored against the region's thresholds, as the region's cells were.

    Adds ``<channel> positive`` per scored channel, *Positive for* and *In
    region*, and sets each count's ``n_cells_section`` and
    ``n_positive_section``. A cell is in the region when its label is among
    *region_labels* (the cells counted there, from the same segmentation), or
    else when its centre is in *band*. Keeps the cells in the region and those
    positive for a channel — or all of them with ``options.keep_section``.
    """
    table = section.copy()
    if region_labels is not None and "Label" in table:
        table[IN_REGION_COLUMN] = np.isin(table["Label"].to_numpy(), np.asarray(region_labels))
    else:
        rows = np.clip(np.round(table["Centroid Y (µm)"].to_numpy() / voxel[0]).astype(int), 0, band.shape[0] - 1)
        cols = np.clip(np.round(table["Centroid X (µm)"].to_numpy() / voxel[1]).astype(int), 0, band.shape[1] - 1)
        table[IN_REGION_COLUMN] = band[rows, cols]
    names: list[list[str]] = [[] for _ in range(len(table))]
    anywhere = np.zeros(len(table), dtype=bool)
    for count in counts:
        if count.reference or not np.isfinite(count.threshold):
            continue
        positive = positive_of(table, count.channel, count, options.statistic)
        table[f"{count.channel} positive"] = positive
        count.n_positive_section = int(positive.sum())
        count.n_cells_section = int(len(table))
        anywhere |= positive
        for index in np.flatnonzero(positive):
            names[index].append(count.channel)
    table["Positive for"] = [" + ".join(n) if n else "none" for n in names]
    if not options.keep_section:
        table = table[anywhere | table[IN_REGION_COLUMN].to_numpy()].reset_index(drop=True)
    return table


def place_positives(table, counts: Sequence[ChannelCount]) -> None:
    """Where each channel's positive cells of the section lie: their median distances."""
    if table is None or not len(table):
        return
    for count in counts:
        column = f"{count.channel} positive"
        if count.reference or column not in table:
            continue
        chosen = table[column].to_numpy(dtype=bool)
        for key, attribute in ((LINE_COLUMN, "median_line_um"), (EDGE_COLUMN, "median_edge_um")):
            if key in table and chosen.any():
                values = table.loc[chosen, key].to_numpy(dtype=np.float64)
                values = values[np.isfinite(values)]
                setattr(count, attribute, float(np.median(values)) if values.size else float("nan"))


def write_labels(result: JunctionCounts, folder: Path) -> tuple[Path, Path]:
    """Write the sample's whole-section label maps: ``<sample> nuclei.tif`` and ``<sample> grown.tif``.

    Compressed, 16-bit when the labels fit, with the pixel size in µm; the
    labels are those of the sample's cells table.
    """
    import tifffile

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    written = []
    for suffix, data in ((NUCLEI_SUFFIX, result.section_labels), (GROWN_SUFFIX, result.section_regions)):
        data = np.asarray(data)
        dtype = np.uint16 if data.size and int(data.max()) < 65536 else np.uint32
        target = folder / f"{result.name}{suffix}"
        tifffile.imwrite(target, data.astype(dtype, copy=False), compression="zlib",
                         resolution=(1e4 / result.voxel_yx[1], 1e4 / result.voxel_yx[0]),
                         resolutionunit="CENTIMETER", metadata={"unit": "um", "spacing_yx_um": list(result.voxel_yx)})
        written.append(target)
    return written[0], written[1]


def read_labels(path: str | Path) -> np.ndarray:
    """A label map written by :func:`write_labels`."""
    import tifffile

    return np.asarray(tifffile.imread(path)).astype(np.int32, copy=False)


def labels_files(result: JunctionCounts) -> tuple[Path, Path] | None:
    """The sample's label maps on disk, when both are there."""
    files = (result.nuclei_file, result.grown_file)
    return files if all(f is not None and Path(f).is_file() for f in files) else None


def write_section(result: JunctionCounts, folder: Path) -> Path:
    """Write the sample's section cells to ``<folder>/<sample>.csv``."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"{result.name}.csv"
    frame = result.section_cells.copy()
    frame.insert(0, "Sample", result.name)
    frame.to_csv(target, index=False, encoding="utf-8-sig")
    return target


def place_cells(table, band: np.ndarray, line, voxel: Sequence[float], overview) -> None:
    """Add to *table* each cell's distance to the junction *line* and to the tissue edge.

    The line distance is signed so the band's side is positive (cells on the
    other side, when a band reaches across, are negative); *Beside the junction
    line* is ``False`` for a cell nearest one of the line's ends, past it rather
    than beside it. The edge distance is positive inside the section's outline,
    negative outside it (cells on the rim past the tissue edge). Without a line
    the line columns are left empty.
    """
    ys = table["Centroid Y (µm)"].to_numpy(dtype=float)
    xs = table["Centroid X (µm)"].to_numpy(dtype=float)
    rows = np.clip(np.round(ys / voxel[0]).astype(int), 0, band.shape[0] - 1)
    cols = np.clip(np.round(xs / voxel[1]).astype(int), 0, band.shape[1] - 1)
    if line is not None and len(np.asarray(line).reshape(-1, 2)) >= 2:
        raw, beside, _length = fu.signed_distance(np.column_stack([ys, xs]), line)
        inside = band[rows, cols]
        chosen = inside & beside if (inside & beside).any() else inside
        flip = np.sign(np.median(np.sign(raw[chosen]))) if chosen.any() else 1.0
        table[LINE_COLUMN] = (raw * (flip or 1.0)).astype(np.float32)
        table[BESIDE_COLUMN] = beside
    else:
        table[LINE_COLUMN] = np.float32(np.nan)
        table[BESIDE_COLUMN] = False
    if overview is not None:
        table[EDGE_COLUMN] = jn.edge_distance(overview, ys / voxel[0], xs / voxel[1])


def analyse_sample(
    path: str | Path,
    options: CountOptions | None = None,
    band: np.ndarray | None = None,
    segment: Callable[..., np.ndarray] | None = None,
    progress: Callable[[str], None] | None = None,
) -> JunctionCounts:
    """Count and score the cells in one sample's band. Never raises.

    *band* overrides the stored one — what the panel passes when the region on
    screen has not been saved. *segment* replaces Cellpose: a function of
    ``(plane, voxel_yx, settings, progress)`` returning a label map.
    """
    from . import ims_store, loaders
    from . import segmentation as sg

    options = options or CountOptions()
    target = Path(path)
    result = JunctionCounts(path=target, name=jn.fu_name(target))
    reviewed = options.review.get(result.name, {})
    result.review_status = reviewed.get("Status", "")
    result.review_decision = reviewed.get("Decision", "")
    result.review_note = reviewed.get("Note", "")
    say = progress or (lambda _text: None)
    started = time.perf_counter()
    was_open = loaders.is_open(target)
    images = inside = None
    try:
        say("reading")
        specs = loaders.load_path(target)
        if not specs:
            raise ValueError(f"{target.name} has no readable channels")
        names = [str(spec.channel_name or spec.name) for spec in specs]
        nuclear = jn.pick_channel(names, options.nuclear_channel, jn.NUCLEAR_HINTS)
        if nuclear is None:
            raise ValueError(f"no nuclear channel among: {', '.join(names)}")
        first = specs[0].data[0] if specs[0].multiscale else specs[0].data
        shape_yx = tuple(int(n) for n in first.shape[-2:])
        voxel = tuple(float(v) for v in specs[0].scale)[-2:]
        result.voxel_yx, result.shape_yx = voxel, shape_yx

        # -- the band ---------------------------------------------------------
        details: dict[str, Any] = {}
        seen_overview = None
        if band is not None:
            band = np.asarray(band).reshape(np.shape(band)[-2:]) > 0
            if band.shape != shape_yx:
                raise ValueError(f"the region is {band.shape} and the image {shape_yx}")
            details = {"source": "on screen"}
            _stored, stored = _stored_band(target, region_candidates(target, options), shape_yx)
            details.update({k: v for k, v in stored.items() if k != "source"})
            details["by"] = "on screen"
        else:
            band, details = _stored_band(target, region_candidates(target, options), shape_yx)
        finder = finder_record(target)
        if band is None:
            if not options.find_junction:
                tried = (f"whose name contains “{options.region_keyword}”, nor “{fu.DEFAULT_KEY}”"
                         if options.region_keyword.strip() else
                         " or ".join(f"“{key}”" for key in fu.region_keys(options.band_key)))
                raise ValueError(f"no stored region {tried} — make it in the Fusion region panel")
            say("finding the junction")
            junction = options.junction or jn.JunctionOptions(save=False)
            overview, found = jn.analyse_specs(specs, junction)
            if found.line_um is None or found.band is None:
                raise ValueError("no junction found: " + ("; ".join(found.flags) or "no reason given"))
            band = jn.full_resolution_band(found, overview, junction) > 0
            seen_overview = overview
            details = {"source": "found automatically", "by": "junction finder", "status": found.status,
                       "flags": "; ".join(found.flags), "line": np.asarray(found.line_um, float),
                       "length": found.metrics.get("Junction length (um)", float("nan"))}
            finder = {"status": found.status, "flags": "; ".join(found.flags),
                      "second_search": "yes" if found.metrics.get("No-seam fallback") else "no"}
        if not band.any():
            raise ValueError("the junction region is empty")
        result.band_source = str(details.get("source", ""))
        if details.get("region"):
            # Which stored region it was: the hand-drawn one, or the automatic fallback.
            result.band_source += f" ({details['region']})"
        result.region_by = str(details.get("by", ""))
        result.status, result.flags = str(details.get("status", "")), str(details.get("flags", ""))
        result.finder_status = finder.get("status", "")
        result.finder_flags = finder.get("flags", "")
        result.second_search = finder.get("second_search", "")
        result.junction_length_um = float(details.get("length", float("nan")))
        result.band_area_mm2 = float(band.sum()) * voxel[0] * voxel[1] / 1e6

        # -- the crop ---------------------------------------------------------
        rows, cols = np.flatnonzero(band.any(axis=1)), np.flatnonzero(band.any(axis=0))
        margin = int(round(CROP_MARGIN_UM / max(voxel)))
        y0, y1 = max(int(rows[0]) - margin, 0), min(int(rows[-1]) + 1 + margin, shape_yx[0])
        x0, x1 = max(int(cols[0]) - margin, 0), min(int(cols[-1]) + 1 + margin, shape_yx[1])
        window = (slice(y0, y1), slice(x0, x1))
        result.origin = (y0, x0)
        inside = band[window]
        say(f"reading {len(specs)} channel(s) in the region")
        images = {name: _plane(spec, window) for name, spec in zip(names, specs)}

        # -- the cells --------------------------------------------------------
        if segment is None:
            crowded = sg.crowded_gpu_note()
            if crowded:
                say("slow: " + crowded)
                result.warnings.append(crowded)
        # A threshold from all the sample's cells needs all of them segmented;
        # a rule that takes nothing from the cells does not, whatever was asked.
        sample_values = None
        section = None
        if options.threshold_from == FROM_SAMPLE and options.method not in CELL_RULES:
            options = replace(options, threshold_from=FROM_REGION)
        if options.whole_section:
            # One pass over the section: its cells, and the region's among them.
            everything = (slice(0, shape_yx[0]), slice(0, shape_yx[1]))
            keep_maps = options.full_labels or options.labels_dir is not None
            whole = segment_section(specs, names, nuclear, voxel, options, segment, say,
                                    crop=everything if keep_maps else window)
            section, nuclei, regions = whole.table, whole.labels, whole.regions
            if keep_maps:
                result.section_labels, result.section_regions = nuclei, regions
                nuclei, regions = nuclei[window], regions[window]
        else:
            say(f"segmenting {inside.shape[1]} × {inside.shape[0]} px of {names[nuclear]}")
            nuclei = np.asarray((segment or _segment)(images[names[nuclear]], voxel, options.settings, say))
            if nuclei.shape != inside.shape:
                raise ValueError(f"the segmentation came back {nuclei.shape}, expected {inside.shape}")
            regions = sg.expand_labels(nuclei, voxel, options.expand_um) if options.expand_um > 0 else nuclei
            if options.threshold_from == FROM_SAMPLE:
                section = section_cells(specs, names, nuclear, voxel, options, segment, say)
        if options.threshold_from == FROM_SAMPLE:
            sample_values = {name: section[f"{name} {options.statistic}"].to_numpy(dtype=np.float64)
                             for name in names}
        say("measuring the cells")
        table, counts = measure_cells(nuclei, regions, images, voxel, inside, options,
                                      nuclear=names[nuclear], origin=(y0, x0), sample_values=sample_values)
        overviews: list[Any] = [seen_overview] if seen_overview is not None else []

        def overview_now():
            if not overviews:
                overviews.append(jn.build_overview(specs, options.junction or jn.JunctionOptions(save=False)))
            return overviews[0]

        if options.distances and len(table):
            say("placing the cells: distance to the line and to the tissue edge")
            try:
                place_cells(table, band, details.get("line"), voxel, overview_now())
            except Exception as exc:
                logger.exception("could not place the cells of %s", target.name)
                result.warnings.append(f"no distances: {exc}")
        if options.whole_section and section is not None:
            say(f"scoring the section's {len(section)} cells")
            result.n_cells_section = int(len(section))
            result.section_cells = score_section(section, counts, band, voxel, options,
                                                 region_labels=table["Label"].to_numpy())
            if options.distances and len(result.section_cells):
                try:
                    place_cells(result.section_cells, band, details.get("line"), voxel, overview_now())
                except Exception as exc:
                    logger.exception("could not place the section's cells of %s", target.name)
                    result.warnings.append(f"no distances for the section's cells: {exc}")
            place_positives(result.section_cells, counts)
            if options.cells_dir is not None and len(result.section_cells):
                say("writing the section's cells")
                result.cells_file = write_section(result, Path(options.cells_dir))
            if options.labels_dir is not None and result.section_labels is not None:
                say("writing the section's label maps")
                try:
                    result.nuclei_file, result.grown_file = write_labels(result, Path(options.labels_dir))
                except Exception as exc:
                    logger.exception("could not write the label maps of %s", target.name)
                    result.warnings.append(f"no label maps written: {exc}")
                if not options.full_labels:
                    # Written; a batch does not keep half a gigabyte a sample in memory.
                    result.section_labels = result.section_regions = None
            if options.cells_dir is not None and result.cells_file is not None:
                try:
                    write_sample_info(result, counts, options, band, details.get("line"))
                except Exception as exc:
                    logger.exception("could not write what %s was segmented with", target.name)
                    result.warnings.append(f"cannot be re-scored later: {exc}")
        kept = np.isin(nuclei, table["Label"].to_numpy())
        result.labels = np.where(kept, nuclei, 0)
        result.regions = np.where(np.isin(regions, table["Label"].to_numpy()), regions, 0)
        result.cells, result.channels, result.n_cells = table, counts, int(len(table))
        for count in counts:
            if np.isfinite(result.band_area_mm2) and result.band_area_mm2 > 0:
                count.per_mm2 = count.n_positive / result.band_area_mm2
            if np.isfinite(result.junction_length_um) and result.junction_length_um > 0:
                count.per_mm_junction = count.n_positive / (result.junction_length_um / 1000.0)
        if not result.n_cells:
            result.warnings.append("no cells in the region")
        if options.pictures_dir is not None and result.section_cells is not None:
            say("drawing the section's positive cells")
            try:
                picture = section_picture(result, specs, band, details.get("line"))
                folder = Path(options.pictures_dir)
                folder.mkdir(parents=True, exist_ok=True)
                result.section_picture = folder / f"{result.name}.png"
                picture.save(result.section_picture)
            except Exception as exc:
                logger.exception("could not draw the section of %s", target.name)
                result.warnings.append(f"no section picture: {exc}")

        if options.save_cells and result.n_cells:
            say("storing the cells")
            full = np.zeros(shape_yx, dtype=result.labels.dtype)
            full[window] = result.labels
            attrs = {"source": "junction analysis", "region": str(details.get("region", options.band_key)),
                     "method": options.method, "statistic": options.statistic, "k": float(options.k),
                     "expand_um": float(options.expand_um)}
            for count in counts:
                if not count.reference:
                    attrs[f"threshold {count.channel}"] = float(count.threshold)
                    attrs[f"positive {count.channel}"] = result.positive_labels(count.channel)
            if not was_open or ims_store.uses_sidecar(target):
                ims_store.save_labels(target, options.cells_key, full, voxel, attrs=attrs)
            else:
                result.warnings.append("cells not stored: the .ims is open in the viewer")
    except Exception as exc:
        logger.exception("junction analysis failed on %s", target.name)
        result.error = str(exc)
        result.gpu_fault = sg.is_gpu_fault(exc)
        if result.gpu_fault:
            forget = getattr(sg.get_backend(), "forget_model", None)
            if forget is not None:
                forget()
    finally:
        if segment is None:
            sg.release_gpu_memory()
        if options.qc_dir is not None:
            _write_picture(result, Path(options.qc_dir), images, inside)
        result.elapsed_s = time.perf_counter() - started
        if not was_open:
            try:
                loaders.release(target)
            except Exception:
                logger.debug("could not release %s", target, exc_info=True)
    return result


def run(
    paths: Sequence[str | Path],
    options: CountOptions | None = None,
    progress: Callable[[str], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
    segment: Callable[..., np.ndarray] | None = None,
    keep_labels: bool | None = None,
    segmented: dict[str, "SegmentedSample"] | None = None,
    reuse_any: bool = False,
) -> list[JunctionCounts]:
    """Every sample in turn. One that fails is reported and the run goes on.

    A sample in *segmented* (see :func:`segmented_samples`) whose saved cells
    were segmented with the settings of *options* is not segmented again: it is
    scored afresh from them (:func:`rescore`) — seconds instead of a minute.
    With *reuse_any*, whatever they were segmented with.

    With ``options.qc_dir`` each leaves a picture in that folder's ``ok/``,
    ``check/`` or ``fail/`` — a sample that failed too, saying why.

    Two things end the run early. Being asked to; and a GPU fault
    (:func:`segmentation.is_gpu_fault`), after which nothing more can be
    segmented by this process: the samples not reached are left out rather
    than each failing with the same error, and the progress line says how many.

    The label maps of the cells are kept only for a single sample, where they
    are drawn (*keep_labels* forces it either way): a few hundred slide scans'
    worth is tens of gigabytes, and a batch needs the tables, not the maps.
    For the same reason a batch lets go of each sample's section cells once
    they are written to ``options.cells_dir``.
    """
    keep = len(paths) == 1 if keep_labels is None else bool(keep_labels)
    results: list[JunctionCounts] = []
    for index, path in enumerate(paths, start=1):
        if should_cancel is not None and should_cancel():
            break
        prefix = f"{jn.fu_name(Path(path))} ({index} of {len(paths)})"
        tell = (lambda text, _prefix=prefix: progress(f"{_prefix}: {text}")) if progress else None
        saved = (segmented or {}).get(jn.fu_name(Path(path)))
        if saved is not None and (reuse_any or saved.matches(options or CountOptions())):
            result = rescore(saved, options, progress=tell)
        else:
            result = analyse_sample(path, options, segment=segment, progress=tell)
        if not keep:
            result.labels = result.regions = None
            result.section_labels = result.section_regions = None
            if result.cells_file is not None:
                # Written already; a hundred sections of every cell do not fit in memory.
                result.section_cells = None
        results.append(result)
        if progress is not None:
            progress(f"{prefix}: " + (result.error or summary_line(result)))
        if result.gpu_fault:
            left = len(paths) - index
            logger.error("GPU fault on %s; stopping with %d sample(s) not attempted", result.name, left)
            if progress is not None:
                progress(
                    f"STOPPED: the GPU reported a fault on {result.name} and will not answer again "
                    f"in this session. {left} sample(s) were not attempted. Close the viewer, start "
                    "it again and run the samples that are missing. This happens when the card runs "
                    "out of memory while another program is using it too."
                )
            break
    return results


# ---------------------------------------------------------------------------
# Scoring again, from the cells a run saved
# ---------------------------------------------------------------------------


def segmentation_key(options: CountOptions) -> dict[str, Any]:
    """What a sample's cells and their figures depend on: same key, same cells, same figures."""
    from . import segmentation as sg

    settings = options.settings or sg.SegmentationSettings()
    key = {
        "model": settings.resolved_model(), "custom_model": str(settings.custom_model_path or ""),
        "mode": settings.mode, "diameter_um": float(settings.diameter_um),
        "min_diameter_um": float(settings.min_diameter_um), "max_diameter_um": float(settings.max_diameter_um),
        "flow_threshold": float(settings.flow_threshold), "cellprob_threshold": float(settings.cellprob_threshold),
        "min_size": int(settings.min_size), "normalize": bool(settings.normalize),
        "expand_um": float(options.expand_um), "nuclear_channel": str(options.nuclear_channel),
        "ring_um": [float(v) for v in RING_UM],
    }
    return json.loads(json.dumps(key))


def write_sample_info(result: JunctionCounts, counts: Sequence[ChannelCount], options: CountOptions,
                      band: np.ndarray, line=None) -> Path:
    """Save beside the sample's cells what a later run needs to score it again, and its region."""
    import tifffile

    cells_dir = Path(options.cells_dir)
    region_file = None
    if options.labels_dir is not None:
        labels_dir = Path(options.labels_dir)
        labels_dir.mkdir(parents=True, exist_ok=True)
        region_file = labels_dir / f"{result.name}{REGION_SUFFIX}"
        tifffile.imwrite(region_file, np.asarray(band, dtype=np.uint8), compression="zlib")
    run_folder = cells_dir.parent

    def relative(path):
        return str(Path(path).relative_to(run_folder)) if path is not None else None

    info = {
        "version": 1, "sample": result.name, "file": str(result.path), "written": time.strftime("%Y-%m-%d %H:%M"),
        "shape_yx": [int(v) for v in result.shape_yx], "voxel_yx": [float(v) for v in result.voxel_yx],
        "channels": [count.channel for count in counts],
        "nuclear": next((count.channel for count in counts if count.reference), ""),
        "background": {count.channel: [float(count.background), float(count.background_sd)] for count in counts},
        "band_source": result.band_source, "region_by": result.region_by, "status": result.status,
        "flags": result.flags, "finder_status": result.finder_status, "finder_flags": result.finder_flags,
        "second_search": result.second_search, "junction_length_um": float(result.junction_length_um),
        "line": np.asarray(line, dtype=float).reshape(-1, 2).tolist() if line is not None else None,
        "complete": bool(options.keep_section), "n_cells_section": int(result.n_cells_section),
        "segmentation": segmentation_key(options),
        "cells_file": relative(result.cells_file), "nuclei_file": relative(result.nuclei_file),
        "grown_file": relative(result.grown_file), "region_file": relative(region_file),
    }
    target = cells_dir / f"{result.name}{INFO_SUFFIX}"
    target.write_text(json.dumps(info, indent=1), encoding="utf-8")
    return target


@dataclass
class SegmentedSample:
    """A sample a run segmented: where its cells, label maps and region are, and what they were made with."""

    name: str
    run: Path
    info: dict[str, Any]

    def file(self, key: str) -> Path | None:
        value = self.info.get(key)
        return self.run / value if value else None

    @property
    def path(self) -> Path:
        return Path(self.info.get("file", ""))

    def matches(self, options: CountOptions) -> bool:
        """Whether its cells are what segmenting with *options* would give."""
        return bool(self.info.get("complete")) and self.info.get("segmentation") == segmentation_key(options)


def _rows_in(path: Path) -> int:
    """Rows of a CSV file, without parsing it."""
    with open(path, "rb") as handle:
        return max(sum(chunk.count(b"\n") for chunk in iter(lambda: handle.read(1 << 20), b"")) - 1, 0)


def _sample_file(base: Path, name: str) -> str:
    """The image of sample *name* in *base*, when its run did not say."""
    found = [path for path in Path(base).glob(f"{name}.*")
             if path.is_file() and not path.name.lower().endswith((".h5", ".json", ".csv"))]
    return str(sorted(found, key=lambda path: len(path.name))[0]) if found else ""


def legacy_info(run: Path, cells_file: Path, counts=None, base: Path | None = None) -> dict[str, Any]:
    """What a run made before ``<sample>.json`` existed says about a sample, rebuilt from its tables.

    The channels and nuclear stain, the background per channel, the region and
    the finder's verdict come from the run's *Counts* sheet (*counts*, read
    once per run); the voxel size and image shape are read from the image when
    it is re-scored. No region was saved, so the sample's stored one is used,
    and what the cells were segmented with is unknown (``segmentation`` is
    ``None``): such a sample is re-scored on request, never skipped by Run.
    """
    name = cells_file.stem
    rows = counts[counts["Sample"] == name] if counts is not None and "Sample" in counts else None
    info: dict[str, Any] = {"version": 0, "sample": name, "written": time.strftime(
        "%Y-%m-%d %H:%M", time.localtime(cells_file.stat().st_mtime)), "segmentation": None,
        "cells_file": str(cells_file.relative_to(run)), "line": None}
    labels = run / LABELS_FOLDER
    for key, suffix in (("nuclei_file", NUCLEI_SUFFIX), ("grown_file", GROWN_SUFFIX)):
        candidate = labels / f"{name}{suffix}"
        info[key] = str(candidate.relative_to(run)) if candidate.is_file() else None
    if rows is not None and len(rows):
        first = rows.iloc[0]

        def text(column):
            value = first.get(column, "")
            return "" if value is None or (isinstance(value, float) and not np.isfinite(value)) else str(value)

        info["file"] = text("File")
        info["channels"] = [str(c) for c in rows["Channel"] if str(c)]
        info["nuclear"] = next((str(r["Channel"]) for _i, r in rows.iterrows()
                                if str(r.get("Rule", "")).startswith("reference")), "")
        info["background"] = {str(r["Channel"]): [float(r.get("Background median", np.nan)),
                                                  float(r.get("Background SD", np.nan))] for _i, r in rows.iterrows()}
        for key, column in (("band_source", "Region"), ("region_by", "Region drawn by"), ("status", "Junction status"),
                            ("flags", "Junction flags"), ("finder_status", "Finder status"),
                            ("finder_flags", "Finder flags"), ("second_search", "Finder second search")):
            info[key] = text(column)
        info["junction_length_um"] = float(first.get("Junction length (µm)", np.nan))
        whole = first.get("Cells in the whole section", np.nan)
        info["complete"] = bool(np.isfinite(whole) and _rows_in(cells_file) >= int(whole))
    else:
        import pandas as pd

        columns = list(pd.read_csv(cells_file, nrows=0).columns)
        info["channels"] = [c[:-5] for c in columns if c.endswith(" mean") and not c.endswith(" ring mean")]
        info["nuclear"] = next((c for c in info["channels"] if any(h in c.lower() for h in jn.NUCLEAR_HINTS)), "")
        info["background"] = {}
        info["complete"] = True
    if not info.get("file") and base is not None:
        info["file"] = _sample_file(base, name)
    return info


def segmented_samples(base: str | Path) -> dict[str, SegmentedSample]:
    """Every sample segmented under ``<base>/cells_analysis``, by name: the newest run's of each.

    Only samples whose every cell was kept (``CountOptions.keep_section``)
    and whose cells file is still there. Runs made before ``<sample>.json``
    existed are read too (:func:`legacy_info`).
    """
    folder = Path(base) / RESULTS_FOLDER
    found: dict[str, SegmentedSample] = {}
    runs = sorted(path for path in folder.iterdir() if path.is_dir()) if folder.is_dir() else []
    for run in runs:
        described = {path.stem for path in (run / CELLS_FOLDER).glob(f"*{INFO_SUFFIX}")}
        older = [path for path in sorted((run / CELLS_FOLDER).glob("*.csv")) if path.stem not in described]
        if older:
            counts = None
            if (run / WORKBOOK_NAME).is_file():
                try:
                    import pandas as pd

                    counts = pd.read_excel(run / WORKBOOK_NAME, sheet_name="Counts")
                except Exception:
                    logger.warning("could not read the counts of %s", run, exc_info=True)
            for cells_file in older:
                try:
                    info = legacy_info(run, cells_file, counts, Path(base))
                except Exception:
                    logger.warning("could not read what %s holds", cells_file, exc_info=True)
                    continue
                if info.get("complete") and info.get("file"):
                    found[cells_file.stem] = SegmentedSample(name=cells_file.stem, run=run, info=info)
        for info_file in sorted((run / CELLS_FOLDER).glob(f"*{INFO_SUFFIX}")):
            try:
                info = json.loads(info_file.read_text(encoding="utf-8"))
            except Exception:
                logger.warning("could not read %s", info_file, exc_info=True)
                continue
            sample = SegmentedSample(name=str(info.get("sample", info_file.stem)), run=run, info=info)
            cells = sample.file("cells_file")
            if info.get("complete") and cells is not None and cells.is_file():
                found[sample.name] = sample
    return found


def rescore(
    sample: SegmentedSample,
    options: CountOptions | None = None,
    band: np.ndarray | None = None,
    progress: Callable[[str], None] | None = None,
) -> JunctionCounts:
    """Score a segmented sample again with *options*, from its saved cells: no segmentation. Never raises.

    The region is the one on screen (*band*), else the sample's stored region
    as it is now — redrawn since, the cells are placed against the new one —
    else the one it was counted in. Thresholds, positives, counts, metrics and
    distances to the line are worked out afresh; each cell's figures and its
    distance to the tissue edge are the saved ones. The pictures are drawn
    again from the image (read, not segmented), and the label maps copied, so
    the new run folder holds everything a segmenting run's does.
    """
    import shutil

    import pandas as pd

    from . import loaders

    options = options or CountOptions()
    info = sample.info
    result = JunctionCounts(path=sample.path, name=sample.name)
    reviewed = options.review.get(result.name, {})
    result.review_status = reviewed.get("Status", "")
    result.review_decision = reviewed.get("Decision", "")
    result.review_note = reviewed.get("Note", "")
    say = progress or (lambda _text: None)
    started = time.perf_counter()
    images = inside = None
    try:
        say(f"re-scoring from the cells saved on {info.get('written', '?')} (no segmentation)")
        table = pd.read_csv(sample.file("cells_file"))
        scored = [c for c in table.columns if c.endswith(" positive") or c in ("Positive for", "Sample")]
        table = table.drop(columns=scored)
        if "voxel_yx" not in info or "shape_yx" not in info:
            # A run from before <sample>.json: the image's header says.
            was_open = loaders.is_open(result.path)
            specs = loaders.load_path(result.path)
            try:
                first = specs[0].data[0] if specs[0].multiscale else specs[0].data
                info = {**info, "shape_yx": [int(n) for n in first.shape[-2:]],
                        "voxel_yx": [float(v) for v in specs[0].scale[-2:]]}
            finally:
                if not was_open:
                    loaders.release(result.path)
        voxel = tuple(float(v) for v in info["voxel_yx"])
        shape = tuple(int(v) for v in info["shape_yx"])
        result.voxel_yx, result.shape_yx = voxel, shape
        line = np.asarray(info["line"], dtype=float) if info.get("line") else None
        for key in ("band_source", "region_by", "status", "flags", "finder_status", "finder_flags", "second_search"):
            setattr(result, key, str(info.get(key, "")))
        result.junction_length_um = float(info.get("junction_length_um", float("nan")))

        # -- the region: on screen, stored now, or the one counted in ----------------
        if band is not None:
            band = np.asarray(band).reshape(shape) > 0
            result.band_source, result.region_by = "on screen", "on screen"
        elif result.path.is_file():
            stored, details = _stored_band(result.path, region_candidates(result.path, options), shape)
            if stored is not None:
                band = stored
                result.band_source = f"stored ({details['region']})"
                result.region_by = str(details.get("by", ""))
                result.status, result.flags = str(details.get("status", "")), str(details.get("flags", ""))
                if details.get("line") is not None:
                    line = np.asarray(details["line"], dtype=float)
                    result.junction_length_um = float(details.get("length", result.junction_length_um))
        if band is None:
            region_file = sample.file("region_file")
            if region_file is None or not region_file.is_file():
                raise ValueError("no region to count in: none stored, none saved with the cells")
            band = read_labels(region_file) > 0
        result.band_area_mm2 = float(band.sum()) * voxel[0] * voxel[1] / 1e6

        rows = np.clip(np.round(table["Centroid Y (µm)"].to_numpy() / voxel[0]).astype(int), 0, shape[0] - 1)
        cols = np.clip(np.round(table["Centroid X (µm)"].to_numpy() / voxel[1]).astype(int), 0, shape[1] - 1)
        table[IN_REGION_COLUMN] = band[rows, cols]
        if options.distances and line is not None:
            place_cells(table, band, line, voxel, None)  # the edge distance is the saved one
        region = table[table[IN_REGION_COLUMN].to_numpy(dtype=bool)]

        # -- thresholds, as a segmenting run sets them -------------------------------
        from_sample = options.threshold_from == FROM_SAMPLE and options.method in CELL_RULES
        counts: list[ChannelCount] = []
        for channel in info["channels"]:
            centre, spread = (info.get("background", {}).get(channel) or [float("nan"), float("nan")])
            count = ChannelCount(channel=channel, n_cells=int(len(region)), statistic=options.statistic,
                                 background=float(centre), background_sd=float(spread),
                                 reference=channel == info.get("nuclear"))
            column = f"{channel} {options.statistic}"
            if column not in table or (options.method == METHOD_RING and f"{channel} ring mean" not in table):
                raise ValueError(f"the saved cells have no “{column if column not in table else channel + ' ring mean'}” "
                                 "— they were saved before it was measured; segment the sample again")
            population = (table if from_sample else region)[column].to_numpy(dtype=float)
            count.cells_median, count.cells_sd = robust_centre_and_spread(population)
            count.threshold_from = FROM_SAMPLE if from_sample else FROM_REGION
            count.n_threshold_cells = int(len(population))
            if count.reference:
                count.method = "reference stain — not scored"
            else:
                count.threshold, count.method = threshold_for(channel, population, (count.background, count.background_sd),
                                                              options)
            counts.append(count)
        say(f"scoring the section's {len(table)} cells")
        section = score_section(table, counts, band, voxel, replace(options, keep_section=True),
                                region_labels=region["Label"].to_numpy())
        cells = section[section[IN_REGION_COLUMN].to_numpy(dtype=bool)].reset_index(drop=True)
        for count in counts:
            if count.reference:
                continue
            count.n_positive = int(cells[f"{count.channel} positive"].sum())
            if np.isfinite(count.background) and count.background:
                cells[f"{count.channel} fold over background"] = cells[f"{count.channel} mean"] / count.background
                if np.isfinite(count.background_sd) and count.background_sd > 0:
                    cells[f"{count.channel} SDs over background"] = (
                        (cells[f"{count.channel} mean"] - count.background) / count.background_sd)
            if result.band_area_mm2 > 0:
                count.per_mm2 = count.n_positive / result.band_area_mm2
            if np.isfinite(result.junction_length_um) and result.junction_length_um > 0:
                count.per_mm_junction = count.n_positive / (result.junction_length_um / 1000.0)
        place_positives(section, counts)
        if not options.keep_section:
            anywhere = (section["Positive for"] != "none").to_numpy() | section[IN_REGION_COLUMN].to_numpy(dtype=bool)
            section = section[anywhere].reset_index(drop=True)
        result.cells, result.channels, result.n_cells = cells, counts, int(len(cells))
        result.section_cells, result.n_cells_section = section, int(len(table))
        if not result.n_cells:
            result.warnings.append("no cells in the region")

        # -- the run folder: tables, label maps, info, pictures -------------------------
        if options.cells_dir is not None:
            result.cells_file = write_section(result, Path(options.cells_dir))
        if options.labels_dir is not None:
            Path(options.labels_dir).mkdir(parents=True, exist_ok=True)
            for key, attribute in (("nuclei_file", "nuclei_file"), ("grown_file", "grown_file")):
                source = sample.file(key)
                if source is not None and source.is_file():
                    target = Path(options.labels_dir) / source.name
                    if source.resolve() != target.resolve():
                        shutil.copyfile(source, target)
                    setattr(result, attribute, target)
        if options.cells_dir is not None:
            # Saved again with this run, so the next one finds the newest; same segmentation.
            write_sample_info(result, counts, replace(options, keep_section=bool(info.get("complete"))), band, line)
            saved = json.loads((Path(options.cells_dir) / f"{result.name}{INFO_SUFFIX}").read_text(encoding="utf-8"))
            saved["segmentation"] = info.get("segmentation")
            saved["complete"] = bool(info.get("complete"))
            (Path(options.cells_dir) / f"{result.name}{INFO_SUFFIX}").write_text(json.dumps(saved, indent=1),
                                                                                encoding="utf-8")
        if (options.qc_dir is not None or options.pictures_dir is not None) and result.path.is_file():
            say("drawing the pictures")
            was_open = loaders.is_open(result.path)
            specs = loaders.load_path(result.path)
            try:
                names = [str(spec.channel_name or spec.name) for spec in specs]
                ys, xs = np.flatnonzero(band.any(axis=1)), np.flatnonzero(band.any(axis=0))
                margin = int(round(CROP_MARGIN_UM / max(voxel)))
                y0, y1 = max(int(ys[0]) - margin, 0), min(int(ys[-1]) + 1 + margin, shape[0])
                x0, x1 = max(int(xs[0]) - margin, 0), min(int(xs[-1]) + 1 + margin, shape[1])
                window = (slice(y0, y1), slice(x0, x1))
                result.origin = (y0, x0)
                inside = band[window]
                images = {name: _plane(spec, window) for name, spec in zip(names, specs)}
                if options.pictures_dir is not None:
                    folder = Path(options.pictures_dir)
                    folder.mkdir(parents=True, exist_ok=True)
                    result.section_picture = folder / f"{result.name}.png"
                    section_picture(result, specs, band, line).save(result.section_picture)
            finally:
                if not was_open:
                    loaders.release(result.path)
    except Exception as exc:
        logger.exception("re-scoring %s failed", sample.name)
        result.error = str(exc)
    finally:
        if options.qc_dir is not None:
            _write_picture(result, Path(options.qc_dir), images, inside)
        result.elapsed_s = time.perf_counter() - started
    return result


def summary_line(result: JunctionCounts) -> str:
    """One line for the log: the cells, and the positives per channel."""
    parts = [f"{count.channel}+ {count.n_positive}" for count in result.channels if not count.reference]
    line = f"{result.n_cells} cell(s) in the region; " + (", ".join(parts) or "no channel scored")
    if result.n_cells_section:
        whole = [f"{count.channel}+ {count.n_positive_section:.0f}" for count in result.channels
                 if not count.reference and np.isfinite(count.n_positive_section)]
        line += f" — {result.n_cells_section} in the section; " + ", ".join(whole)
    return line


# ---------------------------------------------------------------------------
# Progress: how far a run is, and how long it has left
# ---------------------------------------------------------------------------

_SAMPLE_OF = re.compile(r"\((\d+) of (\d+)\):")
_TILE_OF = re.compile(r"tile (\d+) of (\d+)")
#: Share of a sample's time spent segmenting, the rest reading and measuring.
#: Rough; the estimate corrects itself as samples finish.
SEGMENT_SHARE = 0.85
#: Spinner frames for the progress line.
SPINNER = "◐◓◑◒"


class RunProgress:
    """Follows a run's progress lines; says how far it is, and how long it has left.

    Reads the lines :func:`run` and :func:`analyse_sample` send: ``"<sample>
    (i of n): …"`` says which sample, ``"tile k of m"`` how far its
    segmentation is, and the sample's summary line that it is done. A single
    sample's lines carry no ``(i of n)`` and count as one of one. The fraction
    never goes back; the time left is the time so far scaled by what is left.
    """

    def __init__(self, samples: int = 1, now: float | None = None):
        self.samples = max(int(samples), 1)
        self.started = time.perf_counter() if now is None else now
        self.sample = 1
        self.within = 0.0
        self.fraction = 0.0
        self.tile = ""

    def update(self, text: str) -> float:
        """Take one progress line; returns the fraction of the run done."""
        found = _SAMPLE_OF.search(text)
        if found:
            sample, total = int(found.group(1)), int(found.group(2))
            if sample < self.sample:
                return self.fraction  # a late line about a sample already passed
            if sample != self.sample:
                self.within, self.tile = 0.0, ""
            self.sample, self.samples = sample, max(total, 1)
        tile = _TILE_OF.search(text)
        if tile:
            k, m = int(tile.group(1)), max(int(tile.group(2)), 1)
            self.within = max(self.within, SEGMENT_SHARE * (k - 1) / m)
            self.tile = f"tile {k}/{m}"
        elif "cell(s) in the region" in text or "STOPPED" in text:
            self.within, self.tile = 1.0, ""
        elif self.tile and ("measuring" in text or "scoring" in text or "placing" in text or "writing" in text):
            self.within = max(self.within, SEGMENT_SHARE)
        done = (self.sample - 1 + min(self.within, 1.0)) / self.samples
        self.fraction = max(self.fraction, min(done, 1.0))
        return self.fraction

    def elapsed(self, now: float | None = None) -> float:
        return (time.perf_counter() if now is None else now) - self.started

    def remaining(self, now: float | None = None) -> float:
        """Seconds left, or NaN until there is enough to go on."""
        if self.fraction < 0.01:
            return float("nan")
        spent = self.elapsed(now)
        return spent * (1.0 - self.fraction) / self.fraction

    def line(self, tick: int = 0, now: float | None = None, width: int = 24) -> str:
        """One line of text: spinner, bar, percent, where it is, time so far and left."""
        left = self.remaining(now)
        parts = [f"{SPINNER[tick % len(SPINNER)]} {progress_bar(self.fraction, width, tick)} {100 * self.fraction:3.0f}%",
                 f"sample {self.sample}/{self.samples}"]
        if self.tile:
            parts.append(self.tile)
        parts.append(f"{duration(self.elapsed(now))} so far")
        parts.append("about " + duration(left) + " left" if np.isfinite(left) else "estimating…")
        return " · ".join(parts)


def progress_bar(fraction: float, width: int = 24, tick: int = 0) -> str:
    """``[█████▓▒░░░░]``: filled to *fraction*, with a shimmer at the front that moves with *tick*."""
    fraction = min(max(float(fraction), 0.0), 1.0)
    full = int(fraction * width)
    if full >= width:
        return "[" + "█" * width + "]"
    head = "▓▒"[tick % 2]
    return "[" + "█" * full + head + "░" * (width - full - 1) + "]"


def duration(seconds: float) -> str:
    """``42 s``, ``3 min 05 s``, ``1 h 12 min``."""
    if not np.isfinite(seconds):
        return "?"
    seconds = int(round(max(seconds, 0.0)))
    if seconds < 60:
        return f"{seconds} s"
    if seconds < 3600:
        return f"{seconds // 60} min {seconds % 60:02d} s"
    return f"{seconds // 3600} h {(seconds % 3600) // 60:02d} min"


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

#: Columns of the per-channel table, as the panel shows them and Excel gets them.
COUNT_COLUMNS = (
    "Sample", "Channel", "Cells in region", "Positive", "Positive (%)",
    "Positive per mm²", "Positive per mm of junction",
    "Cells in the whole section", "Positive in the whole section", "Positive in the whole section (%)",
    "Enrichment in the region", "Positives: median distance to the line (µm)",
    "Positives: median distance to the edge (µm)",
    "Threshold", "Rule", "Compared",
    "Threshold from", "Cells behind the threshold",
    "Cells median", "Cells SD", "Background median", "Background SD",
    "Region area (mm²)", "Junction length (µm)", "Region drawn by", "Region", "Junction status",
    "Junction flags", "Finder status", "Finder flags", "Finder second search",
    "Review status", "Review decision", "Review note", "QC", "Problem", "Cells file", "File",
)


def counts_dataframe(results: Sequence[JunctionCounts]):
    """One row per sample and channel: the count, and everything it rests on."""
    import pandas as pd

    rows = []
    for result in results:
        shared = {
            "Sample": result.name,
            "Cells in region": result.n_cells,
            "Cells in the whole section": result.n_cells_section or float("nan"),
            "Region area (mm²)": result.band_area_mm2,
            "Junction length (µm)": result.junction_length_um,
            "Region drawn by": result.region_by,
            "Region": result.band_source,
            "Junction status": result.status,
            "Junction flags": result.flags,
            "Finder status": result.finder_status,
            "Finder flags": result.finder_flags,
            "Finder second search": result.second_search,
            "Review status": result.review_status,
            "Review decision": result.review_decision,
            "Review note": result.review_note,
            "QC": result.qc,
            "Problem": result.error or "; ".join(result.warnings),
            "Cells file": str(result.cells_file) if result.cells_file else "",
            "File": str(result.path),
        }
        if not result.channels:
            rows.append({**shared, "Channel": ""})
            continue
        for count in result.channels:
            rows.append({
                **shared,
                "Channel": count.channel,
                "Positive": float("nan") if count.reference else count.n_positive,
                "Positive (%)": float("nan") if count.reference else count.percent,
                "Positive per mm²": float("nan") if count.reference else count.per_mm2,
                "Positive per mm of junction": float("nan") if count.reference else count.per_mm_junction,
                "Positive in the whole section": float("nan") if count.reference else count.n_positive_section,
                "Positive in the whole section (%)": float("nan") if count.reference else count.percent_section,
                "Enrichment in the region": float("nan") if count.reference else count.enrichment,
                "Positives: median distance to the line (µm)": float("nan") if count.reference else count.median_line_um,
                "Positives: median distance to the edge (µm)": float("nan") if count.reference else count.median_edge_um,
                "Threshold": count.threshold,
                "Rule": count.method,
                "Compared": "ring contrast" if count.method == METHOD_RING else count.statistic,
                "Threshold from": "" if count.reference or count.method not in CELL_RULES else count.threshold_from,
                "Cells behind the threshold": (float("nan") if count.reference or count.method not in CELL_RULES
                                               else count.n_threshold_cells),
                "Cells median": count.cells_median,
                "Cells SD": count.cells_sd,
                "Background median": count.background,
                "Background SD": count.background_sd,
            })
    return pd.DataFrame(rows, columns=list(COUNT_COLUMNS))


def cells_dataframe(results: Sequence[JunctionCounts]):
    """Every cell of every sample, with the sample it came from."""
    import pandas as pd

    frames = []
    for result in results:
        if result.cells is None or not len(result.cells):
            continue
        frame = result.cells.copy()
        frame.insert(0, "Sample", result.name)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame({"Sample": []})


def section_dataframe(results: Sequence[JunctionCounts]):
    """The section's cells of every sample segmented whole and not written to a CSV, with their sample."""
    import pandas as pd

    frames = []
    for result in results:
        # Written to its own CSV already: a second copy in the workbook only slows it down.
        if result.section_cells is None or not len(result.section_cells) or result.cells_file is not None:
            continue
        frame = result.section_cells.copy()
        frame.insert(0, "Sample", result.name)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else None


def combinations_dataframe(results: Sequence[JunctionCounts]):
    """How many cells are positive for which set of channels, per sample."""
    import pandas as pd

    rows = []
    for result in results:
        if result.cells is None or not len(result.cells):
            continue
        tally = result.cells["Positive for"].value_counts()
        for combination, number in tally.items():
            rows.append({
                "Sample": result.name,
                "Positive for": combination,
                "Cells": int(number),
                "Share of cells (%)": 100.0 * float(number) / max(result.n_cells, 1),
            })
    return pd.DataFrame(rows, columns=["Sample", "Positive for", "Cells", "Share of cells (%)"])


def settings_dataframe(options: CountOptions):
    """What the run was made with, so the workbook explains itself."""
    import pandas as pd

    rows = [
        ("Region", f"first stored junction region whose name contains “{options.region_keyword}”, "
                   f"else “{fu.DEFAULT_KEY}”" if options.region_keyword.strip() else options.band_key),
        ("Samples from a review", "yes" if options.review else "no"),
        ("Positive when", options.method),
        ("k (SDs)", options.k),
        ("Ring contrast at least (cell ÷ ring)", options.ring_ratio if options.method == METHOD_RING else "not used"),
        ("Ring (µm out from the nucleus)", f"{RING_UM[0]:g}–{RING_UM[1]:g}"),
        ("Threshold from", options.threshold_from if options.method in CELL_RULES else "not from cells"),
        ("Compared per cell", options.statistic),
        ("Nuclei grown by (µm)", options.expand_um),
        ("Junction found when none stored", options.find_junction),
        ("Whole section segmented", "yes, every cell kept" if options.whole_section and options.keep_section
         else "yes, positive and in-region cells kept" if options.whole_section else "no, the region only"),
        ("Section cells written to", str(options.cells_dir) if options.cells_dir else "the workbook"),
    ]
    rows.extend((f"Fixed threshold: {name}", value) for name, value in options.fixed.items())
    settings = options.settings
    if settings is not None:
        rows.extend([
            ("Cellpose model", settings.resolved_model()),
            ("Diameter (µm)", settings.diameter_um),
            ("Cell probability", settings.cellprob_threshold),
            ("Flow threshold", settings.flow_threshold),
        ])
    return pd.DataFrame(rows, columns=["Setting", "Value"])


def run_folder(base: str | Path, label: str = "", when: float | None = None) -> Path:
    """A new folder for one run's results: ``<base>/cells_analysis/<date>_<time>[ <label>]``."""
    stamp = time.strftime("%Y-%m-%d_%H%M%S", time.localtime(when))
    name = f"{stamp} {label}".strip() if label else stamp
    return Path(base) / RESULTS_FOLDER / name


def latest_run(base: str | Path) -> Path | None:
    """The newest run folder under ``<base>/cells_analysis``, if there is one."""
    folder = Path(base) / RESULTS_FOLDER
    runs = sorted(path for path in folder.iterdir() if path.is_dir()) if folder.is_dir() else []
    return runs[-1] if runs else None


def results_options(options: CountOptions, folder: str | Path) -> CountOptions:
    """*options* writing into the run folder *folder*: cells, region pictures, section pictures."""
    folder = Path(folder)
    return replace(options, cells_dir=folder / CELLS_FOLDER, qc_dir=folder / REGION_FOLDER,
                   labels_dir=folder / LABELS_FOLDER,
                   pictures_dir=folder / POSITIVE_FOLDER)


def write_readme(folder: str | Path, results: Sequence[JunctionCounts], options: CountOptions) -> Path:
    """Say in the run folder what is in it, and how the run was made."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    tally = ", ".join(f"{sum(r.qc == status for r in results)} {status}" for status in jn.STATUSES)
    lines = [
        f"Cells analysis, {time.strftime('%Y-%m-%d %H:%M')} — {len(results)} sample(s): {tally}.",
        "",
        f"{WORKBOOK_NAME}",
        "    Counts: per sample and channel — cells and positive cells in the junction region",
        "    and in the whole section, %, enrichment in the region, where the positives lie,",
        "    thresholds, who drew the region, the finder's and the review's verdicts.",
        "    Combinations: how many cells are positive for which set of channels.",
        "    Cells: every cell of the region, every figure. Settings: how the run was made.",
        f"{CELLS_FOLDER}/<sample>.csv",
        "    Every cell of the whole section: position, size, every figure per channel,",
        "    positive or not, in the region or not, distance to the junction line and edge.",
        f"{REGION_FOLDER}/ok, check, fail/<sample>.png",
        "    The counted region, a ring round each positive cell, sorted by QC status.",
        "    Look through them in the Junction review panel.",
        f"{LABELS_FOLDER}/<sample>{NUCLEI_SUFFIX}, <sample>{GROWN_SUFFIX}",
        "    The whole section's label maps: every cell segmented, and the same grown by the",
        "    expansion it was measured over. Labels match the 'Label' column of the cells CSV.",
        f"{POSITIVE_FOLDER}/<sample>.png",
        "    The whole section: the region outlined in white, the junction line in yellow,",
        "    a ring round every positive cell (one colour per channel, as in the caption).",
        "",
        "Settings:",
        *(f"    {row.Setting}: {row.Value}" for row in settings_dataframe(options).itertuples()),
    ]
    failed = [r for r in results if not r.ok]
    if failed:
        lines += ["", "Failed:", *(f"    {r.name}: {r.error}" for r in failed)]
    target = folder / README_NAME
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def export(results: Sequence[JunctionCounts], path: str | Path, options: CountOptions | None = None) -> Path:
    """Write the counts, the combinations, every cell and the settings to one workbook."""
    from .exports import export_sheets

    sheets = {
        "Counts": counts_dataframe(results),
        "Combinations": combinations_dataframe(results),
        "Cells": cells_dataframe(results),
        "Settings": settings_dataframe(options or CountOptions()),
    }
    section = section_dataframe(results)
    if section is not None:
        sheets = {**{k: v for k, v in sheets.items() if k != "Settings"}, "Section cells": section,
                  "Settings": sheets["Settings"]}
    return Path(export_sheets(sheets, path))
