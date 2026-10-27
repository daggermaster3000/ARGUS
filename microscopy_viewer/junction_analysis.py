"""Count the cells in the junction band, and say which channels each is positive for.

The band (see :mod:`microscopy_viewer.fusion` and
:mod:`microscopy_viewer.junction`) is where cells that crossed from one organoid
into the other are looked for. This module does the looking:

1. **The band** is read from the sample's store, or found on the spot when the
   sample has none yet.
2. **The cells** are nuclei, segmented on the nuclear stain — but only inside
   the band's bounding box. A slide scan is fifty megapixels and the band is
   three; segmenting the rest would be most of the run and none of the answer.
   A cell belongs to the band when its centroid is in it.
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
METHODS = (METHOD_CELLS, METHOD_BACKGROUND, METHOD_OTSU, METHOD_FIXED)

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

#: A tile is segmented when at least this share of it is brighter, in the
#: nuclear stain, than the split between slide and tissue.
TILE_MIN_TISSUE = 0.02

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
    #: The nuclear stain: every cell was found on it, so it is not scored.
    reference: bool = False

    @property
    def percent(self) -> float:
        return 100.0 * self.n_positive / self.n_cells if self.n_cells else float("nan")


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

        values = stats[options.statistic]
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
            positive = np.asarray(values > count.threshold) & np.isfinite(values)
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


def count_picture(result: JunctionCounts, images: dict[str, np.ndarray], band: np.ndarray, size: int = 900):
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
    """The compared figure of every cell in the sample, per channel.

    The whole section is segmented, in tiles of :data:`TILE_PX` so that it
    finishes; tiles of bare slide are skipped. Each cell is measured as the
    cells in the region are — over its nucleus grown by ``options.expand_um`` —
    and what comes back is ``options.statistic`` of each, which is all a
    threshold needs. Nothing is kept of where the cells were.
    """
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

    values: dict[str, list[np.ndarray]] = {name: [] for name in names}
    found = 0
    for index, tile in enumerate(tiles, start=1):
        say(f"the sample's cells: tile {index} of {len(tiles)} ({found} so far)")
        nuclei = np.asarray((segment or _segment)(stain[tile], voxel_yx, options.settings, None))
        ids = np.unique(nuclei[nuclei > 0]).astype(np.int64)
        if not ids.size:
            continue
        regions = sg.expand_labels(nuclei, voxel_yx, options.expand_um) if options.expand_um > 0 else nuclei
        for position, name in enumerate(names):
            image = stain[tile] if position == nuclear else _plane(specs[position], tile)
            values[name].append(label_stats(regions, image, ids)[options.statistic])
        found += int(ids.size)
    return {name: np.concatenate(parts) if parts else np.zeros(0) for name, parts in values.items()}


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
    for name in ("status", "flags"):
        if name in attrs:
            details[name] = ims_store._text(attrs[name])
    return np.asarray(labels).reshape(labels.shape[-2:]) > 0, details


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
            details = {"source": "found automatically", "by": "junction finder", "status": found.status,
                       "flags": "; ".join(found.flags),
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
        say(f"segmenting {inside.shape[1]} × {inside.shape[0]} px of {names[nuclear]}")
        nuclei = np.asarray((segment or _segment)(images[names[nuclear]], voxel, options.settings, say))
        if nuclei.shape != inside.shape:
            raise ValueError(f"the segmentation came back {nuclei.shape}, expected {inside.shape}")
        regions = sg.expand_labels(nuclei, voxel, options.expand_um) if options.expand_um > 0 else nuclei
        # A threshold from all the sample's cells needs all of them segmented;
        # a rule that takes nothing from the cells does not, whatever was asked.
        sample_values = None
        if options.threshold_from == FROM_SAMPLE:
            if options.method in CELL_RULES:
                sample_values = sample_cell_values(specs, names, nuclear, voxel, options, segment, say)
            else:
                options = replace(options, threshold_from=FROM_REGION)
        say("measuring the cells")
        table, counts = measure_cells(nuclei, regions, images, voxel, inside, options,
                                      nuclear=names[nuclear], origin=(y0, x0), sample_values=sample_values)
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
) -> list[JunctionCounts]:
    """Every sample in turn. One that fails is reported and the run goes on.

    With ``options.qc_dir`` each leaves a picture in that folder's ``ok/``,
    ``check/`` or ``fail/`` — a sample that failed too, saying why.

    Two things end the run early. Being asked to; and a GPU fault
    (:func:`segmentation.is_gpu_fault`), after which nothing more can be
    segmented by this process: the samples not reached are left out rather
    than each failing with the same error, and the progress line says how many.

    The label maps of the cells are kept only for a single sample, where they
    are drawn (*keep_labels* forces it either way): a few hundred slide scans'
    worth is tens of gigabytes, and a batch needs the tables, not the maps.
    """
    keep = len(paths) == 1 if keep_labels is None else bool(keep_labels)
    results: list[JunctionCounts] = []
    for index, path in enumerate(paths, start=1):
        if should_cancel is not None and should_cancel():
            break
        prefix = f"{jn.fu_name(Path(path))} ({index} of {len(paths)})"
        result = analyse_sample(
            path, options, segment=segment,
            progress=(lambda text, _prefix=prefix: progress(f"{_prefix}: {text}")) if progress else None,
        )
        if not keep:
            result.labels = result.regions = None
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


def summary_line(result: JunctionCounts) -> str:
    """One line for the log: the cells, and the positives per channel."""
    parts = [f"{count.channel}+ {count.n_positive}" for count in result.channels if not count.reference]
    return f"{result.n_cells} cell(s) in the region; " + (", ".join(parts) or "no channel scored")


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

#: Columns of the per-channel table, as the panel shows them and Excel gets them.
COUNT_COLUMNS = (
    "Sample", "Channel", "Cells in region", "Positive", "Positive (%)",
    "Positive per mm²", "Positive per mm of junction", "Threshold", "Rule", "Compared",
    "Threshold from", "Cells behind the threshold",
    "Cells median", "Cells SD", "Background median", "Background SD",
    "Region area (mm²)", "Junction length (µm)", "Region drawn by", "Region", "Junction status",
    "Junction flags", "Finder status", "Finder flags", "Finder second search",
    "Review status", "Review decision", "Review note", "QC", "Problem", "File",
)


def counts_dataframe(results: Sequence[JunctionCounts]):
    """One row per sample and channel: the count, and everything it rests on."""
    import pandas as pd

    rows = []
    for result in results:
        shared = {
            "Sample": result.name,
            "Cells in region": result.n_cells,
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
                "Threshold": count.threshold,
                "Rule": count.method,
                "Compared": count.statistic,
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
        ("Threshold from", options.threshold_from if options.method in CELL_RULES else "not from cells"),
        ("Compared per cell", options.statistic),
        ("Nuclei grown by (µm)", options.expand_um),
        ("Junction found when none stored", options.find_junction),
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


def export(results: Sequence[JunctionCounts], path: str | Path, options: CountOptions | None = None) -> Path:
    """Write the counts, the combinations, every cell and the settings to one workbook."""
    from .exports import export_sheets

    sheets = {
        "Counts": counts_dataframe(results),
        "Combinations": combinations_dataframe(results),
        "Cells": cells_dataframe(results),
        "Settings": settings_dataframe(options or CountOptions()),
    }
    return Path(export_sheets(sheets, path))
