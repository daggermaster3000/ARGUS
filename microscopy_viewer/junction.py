"""Find the junction of an assembloid by itself, and say how far to trust it.

An assembloid is two organoids fused: one carries the labelled (GFP) cells, the
other does not, and the measurement is what crossed from the first into the
second. :mod:`microscopy_viewer.fusion` turns a line along the interface into a
band; this module proposes the line, so a folder of two hundred slides does not
need two hundred hand-drawn ones — and it scores each sample, because a torn
section or two organoids that never fused should not reach the statistics just
because a line could be put through them.

Everything here works on an **overview**: every channel flattened over Z and
decimated to about 5 µm per pixel. An organoid is millimetres across; nothing
about where its junction is lives below that scale, and a slide scan is a
gigabyte at full resolution.

How the junction is found:

1. **Tissue** from the nuclear channel and the others (:func:`tissue_mask`).
2. **Where the labelled cells are**: the punctate part of the GFP channel —
   what stands out over its own neighbourhood, which leaves out autofluorescence
   and folds — smoothed into a density (:func:`cell_density`).
3. **Two seeds**, one in each organoid (:func:`find_seeds`). The first is the
   dense core of that density. The second is made the same way from the other
   organoid's own label when it has one (mCherry, the 555 channel): the dense
   core of *its* cells. Without one, it is the low-density tissue furthest from
   the first seed going through the tissue.
4. **A minimum cut** between the seeds (:func:`split_tissue`). This is the
   step that matters. Labelled cells are not spread evenly through their own
   organoid, so thresholding the density draws the line round the brightest
   patch instead of round the organoid. The density only says which side is
   which; where exactly the two meet is read off the nuclear stain, which shows
   a thin dark line between two organoids (:func:`line_cost`). The cut is the
   cheapest way across the tissue between the seeds — short, and cheaper still
   along that line — so it runs through the neck when there is one and along
   the line wherever there is one to follow.
5. **The interface** between the two halves, ordered end to end and smoothed
   (:func:`trace_interface`), is the junction line.
6. **The band** is the part of one organoid within the band width of that line
   (:func:`_build_band`), so its two ends run along the organoid's own outline
   rather than being cut straight across it.

It is a proposal. On a clean pair of lobes it is where a person would draw it;
on an organoid embedded in the other, or one with labelled cells everywhere, it
is a guess, and :func:`quality_flags` says so. The line goes into the same
Shapes layer a hand-drawn one does, so it can be corrected.

No Qt and no napari here.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from . import fusion as fu
from .utils import get_logger

logger = get_logger("junction")

#: Pixel size the overview is decimated to, in µm.
OVERVIEW_UM = 5.0

#: Name of the stored outline the analysis counts inside, and of its label map.
BAND_ROI_NAME = "Fusion band"

#: Channel-name fragments tried, in order, when no channel is named.
NUCLEAR_HINTS = ("dapi", "hoechst", "405")
LABEL_HINTS = ("gfp", "488")
#: The other organoid's own label, when it carries one.
OTHER_HINTS = ("mcherry", "tdtomato", "rfp", "555", "561", "568", "594")

#: How the second seed was made (see :func:`find_seeds`).
SEED_CHANNEL = "dense core of the other organoid's label"
SEED_FAR = "low-density tissue far from the labelled cells"

#: A channel is taken to hold cells when its brightest points stand this many
#: robust SDs over its own texture. Noise alone reaches about three.
CELLS_OVER_NOISE = 6.0

#: The other organoid's core is used as a seed only when it covers at least
#: this share of the tissue: less is a few stray cells, not an organoid.
SEED_MIN_SHARE = 0.02

#: The nuclear stain is compared with its surroundings to find the line between
#: two organoids: smoothed over the first distance, divided by itself smoothed
#: over the second, and raised to the power — which sets how much cheaper a dark
#: line is to cut along than plain tissue (at 6, a line a fifth darker than its
#: surroundings costs a quarter as much).
LINE_SMOOTH_UM = 8.0

#: How far the junction is smoothed along its length, and how far the outlines
#: of the tissue and of the band are rounded off. The cut is found pixel by
#: pixel and wanders by a few of them; an organoid's boundary does not turn a
#: corner in less than this.
JUNCTION_SMOOTH_UM = 150.0
SMOOTH_UM = 25.0
LINE_SURROUND_UM = 60.0
LINE_POWER = 6.0

#: A tear is empty, not dim: where the nuclear stain (smoothed as for the line
#: cost) falls below this share of the tissue's level, or the tissue mask has a
#: hole, the section is torn. Cutting along a tear is priced at
#: ``tear_price`` (see :func:`tear_mask`) instead of being the cheapest thing in
#: the section, which the darkest pixels otherwise are.
TEAR_LEVEL = 0.35
TEAR_PRICE = 0.0
#: How much a cut pays for running where the labelled cells do not change (see
#: :func:`edge_cost`), on top of the dark-line price.
EDGE_WEIGHT = 0.0
#: How strongly the cut is drawn to where the labels change from one organoid's
#: to the other's (see :func:`label_change_price`): the price is multiplied by
#: a factor from 1 (labels flat) down to :data:`LABEL_CHANGE_FLOOR` (steepest
#: change); 0 leaves it out.
LABEL_CHANGE = 0.0
#: Smoothing (µm) of the label densities the change is measured on: finer than
#: the 100 µm density that places the seeds, so the change is a narrow valley.
LABEL_CHANGE_UM = 50.0
LABEL_CHANGE_FLOOR = 0.3

#: How the tissue is split between the seeds: ``"cut"`` (the minimum cut, then
#: straightened), ``"walk"`` (a random walker: each pixel goes to the seed a
#: random walk from it is likelier to reach first, then straightened along the
#: 50:50 line), or ``"auto"`` (the cut, falling back on the walk when the cut
#: looks wrong — see :func:`cut_looks_wrong`).
SPLIT_METHOD = "cut"
#: How strongly the random walk is held back by a change in the cutting price
#: (scikit-image's ``beta``; it is divided by ten times the price's spread).
WALK_BETA = 130.0
#: When "auto" falls back on the walk: a cut cheaper than this per pixel along
#: it runs down a tear rather than a dim line; dearer than this, it found no
#: line to follow; a turn sharper than this, it followed a crack.
AUTO_PRICE_LOW = 0.05
AUTO_PRICE_HIGH = 0.85
AUTO_TURN_DEG = 40.0

#: Find the tissue in every channel, not only the nuclear stain (see
#: :func:`tissue_mask`): an organoid with a faint nuclear stain is otherwise
#: dropped, and the junction stops where the mask does.
TISSUE_ALL_CHANNELS = True

#: Where two organoids meet without a dark line between them, the cheapest cut
#: is the shortest, wherever that is. The cut's mean line price (see
#: :func:`line_cost`; 1 is plain tissue) above this says there was no line to
#: follow: the sample is flagged, and with ``no_seam_price`` set the junction
#: is found again from the labels instead (see :func:`dominant_seeds`).
NO_SEAM_PRICE = 0.4
#: The settings for sections without a dark line, used for the second search
#: (see ``JunctionOptions.no_seam_price``): seeds, their lead, and the price.
NO_SEAM_SEED_RULE = "dominant"
NO_SEAM_LEAD: float | None = None
NO_SEAM_EDGE = 0.5
NO_SEAM_LABEL_CHANGE = 0.0
NO_SEAM_TEAR_PRICE = 0.0
NO_SEAM_BEND_UM = 1000.0
#: How far one organoid's label must lead the other's (both scaled 0–1 over
#: the tissue) for a pixel to join its seed in :func:`dominant_seeds`.
SEED_LEAD = 0.25
#: Which seeds the cut starts from: ``"core"`` (see :func:`find_seeds`),
#: ``"dominant"`` (see :func:`dominant_seeds`), ``"patches"`` (the labelled seed
#: is every patch of labelled cells of at least SEED_MIN_SHARE of the tissue, not
#: only the largest, and is not pulled back) or ``"patches_pulled"`` (the same,
#: each patch pulled back to its own far half).
SEED_RULE = "core"
SEED_DOMINANT = "where each organoid's label clearly leads"

#: Side of the median taken away from a label channel before its cells are
#: counted into a density (see :func:`cell_density`), in µm.
MEDIAN_UM = 45.0

#: What bending the junction costs. A corner of 90° costs as much as cutting
#: this many µm of plain tissue; the same turn spread over a longer stretch
#: costs less, in proportion (bending energy: angle² over length). So the line
#: still curves round an organoid, but no longer turns a corner to follow a
#: crack. 0 leaves the cut as the dark line alone makes it.
BEND_UM = 1000.0
#: The straightening pass works on stations this far apart along the junction,
#: and sideways in steps of this much; per station the line turns at most
#: :data:`BEND_MAX_STEPS` steps sideways (about 55°).
BEND_STATION_UM = 20.0
BEND_OFFSET_UM = 5.0
BEND_MAX_STEPS = 6

#: How many samples a folder run handles at once, each in its own process.
#: About 5 s a sample one after another; on the 32-core workstation twelve at
#: once ran 708 sections in 10 minutes instead of an hour. Each process holds
#: one sample's overview (a few hundred MB at most). 1 = one after another.
RUN_WORKERS = max(1, min(8, (os.cpu_count() or 2) - 2))

STATUS_OK = "ok"
STATUS_CHECK = "check"
STATUS_FAIL = "fail"
#: Every status, which is also the folders a run sorts its pictures into.
STATUSES = (STATUS_OK, STATUS_CHECK, STATUS_FAIL)

#: Flags that make a sample unusable rather than merely worth a look.
FAILING = ("no tissue found", "no junction found", "organoids not fused")


@dataclass
class JunctionOptions:
    """What the automatic junction and its band are built with."""

    #: Channels, as an index or a fragment of the name; empty picks by hint.
    nuclear_channel: int | str = ""
    label_channel: int | str = ""
    #: The channel labelling the *other* organoid, for the second seed. Empty
    #: takes the first channel that is neither of the two above.
    other_channel: int | str = ""
    #: Make the second seed from that channel, as the first is made from the
    #: labelled cells. Off, or with no such channel, it is the low-density
    #: tissue far from the labelled cells.
    seed_from_other_channel: bool = True
    width_um: float = fu.DEFAULT_WIDTH_UM
    gap_um: float = fu.DEFAULT_GAP_UM
    cap_um: float = 0.0
    #: Put the band on the organoid the labelled cells migrate *into*. Off puts
    #: it on the labelled organoid itself.
    into_unlabelled: bool = True
    #: Draw the junction along the dark line the two organoids show between
    #: them in the nuclear stain. Off places it by the shape of the section alone.
    follow_nuclear_line: bool = True
    #: What a 90° corner in the junction costs, in µm of plain tissue cut (see
    #: :data:`BEND_UM`); 0 lets it bend freely.
    bend_um: float = BEND_UM
    #: Price of cutting along a tear (see :data:`TEAR_LEVEL`); 0 leaves tears cheap.
    tear_price: float = TEAR_PRICE
    #: Extra price for cutting where the labelled cells do not change, 0–1
    #: per pixel at most (see :func:`edge_cost`); 0 leaves it out.
    edge_weight: float = EDGE_WEIGHT
    #: Discount for cutting where the labels change, and the smoothing it is
    #: measured at (see :func:`label_change_price`); 0 leaves it out.
    label_change: float = LABEL_CHANGE
    label_change_um: float = LABEL_CHANGE_UM
    #: ``"cut"``, ``"walk"`` or ``"auto"`` (see :data:`SPLIT_METHOD`).
    split_method: str = SPLIT_METHOD
    walk_beta: float = WALK_BETA
    #: Take the local median away from the label channels before their cells
    #: are counted into a density, over this many µm (see :func:`cell_density`).
    use_median: bool = True
    median_um: float = MEDIAN_UM
    #: ``"core"`` or ``"dominant"`` (see :data:`SEED_RULE`).
    seed_rule: str = SEED_RULE
    #: Above this cut price there is no dark line to follow, and the junction
    #: is found again from dominant seeds with ``no_seam_edge`` as the cells'
    #: edge weight (see :data:`NO_SEAM_PRICE`); 0 leaves the first cut. On since
    #: 26 Oct 2026: 172 -> 159 um held out on 138 hand-drawn sections, no change
    #: where there is a dark line (``junction_seeding.ipynb``).
    no_seam_price: float = NO_SEAM_PRICE
    #: The settings of that second search, for sections without a dark line
    #: (see :data:`NO_SEAM_SEED_RULE` and below); ``no_seam_lead`` None is
    #: :data:`SEED_LEAD`.
    no_seam_edge: float = NO_SEAM_EDGE
    no_seam_seed_rule: str = NO_SEAM_SEED_RULE
    no_seam_lead: float | None = NO_SEAM_LEAD
    no_seam_label_change: float = NO_SEAM_LABEL_CHANGE
    no_seam_tear_price: float = NO_SEAM_TEAR_PRICE
    no_seam_bend_um: float = NO_SEAM_BEND_UM
    #: Find the tissue in every channel, not only the nuclear stain.
    tissue_all_channels: bool = TISSUE_ALL_CHANNELS
    #: Keep only the part of the band that is on tissue.
    clip_to_tissue: bool = True
    #: How far past the tissue's outline the band still reaches, in µm, so the
    #: cells migrating over the organoid's side are in it.
    margin_um: float = fu.TISSUE_MARGIN_UM
    #: Store the band with the sample: as a label map and as an outline.
    save: bool = True
    label_key: str = fu.DEFAULT_KEY
    #: Folder the QC pictures and workbook go to; ``None`` writes none.
    qc_dir: Path | None = None


@dataclass
class Overview:
    """One sample, flattened and decimated, with what is needed to go back."""

    #: ``(C, y, x)`` float32, maximum over Z, every *step*-th pixel.
    channels: np.ndarray
    names: list[str]
    #: µm per overview pixel, and per full-resolution pixel.
    um: float
    voxel_yx: tuple[float, float]
    shape_yx: tuple[int, int]
    step: int
    nuclear: int = 0
    label: int = 0
    #: The channel labelling the other organoid, or ``None`` when there is none.
    other: int | None = None
    #: Sharpness of the nuclear channel at full resolution (see :func:`focus_score`).
    focus: float = float("nan")
    saturated: float = 0.0
    #: The nuclear channel again, but *averaged* over each block of pixels
    #: rather than sampled: the line between two organoids is a dozen µm wide,
    #: and one pixel in sixteen of a speckle of nuclei does not show it.
    nuclear_mean: np.ndarray | None = None


@dataclass
class JunctionResult:
    """What was found on one overview."""

    tissue: np.ndarray
    filled: np.ndarray
    density: np.ndarray
    #: 0 outside, 1 the labelled organoid, 2 the other. ``None`` when not split.
    halves: np.ndarray | None = None
    #: The junction in world µm ``(y, x)``, ordered end to end.
    line_um: np.ndarray | None = None
    side: str = fu.SIDE_LEFT
    #: The band on the overview grid, and its outline in µm.
    band: np.ndarray | None = None
    band_outline_um: np.ndarray | None = None
    #: The two seeds the tissue was cut between, and how the second was made.
    seeds: tuple[np.ndarray, np.ndarray] | None = None
    second_seed: str = ""
    metrics: dict[str, float] = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)
    #: The section's outline and labelled half, centred and turned onto its
    #: long axis (see :func:`shape_signature`); for matching serial sections.
    signature: np.ndarray | None = None

    @property
    def status(self) -> str:
        if any(flag in FAILING for flag in self.flags):
            return STATUS_FAIL
        return STATUS_CHECK if self.flags else STATUS_OK


@dataclass
class JunctionOutcome:
    """One file's result in a batch."""

    path: Path
    name: str
    status: str = STATUS_FAIL
    flags: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    line_um: np.ndarray | None = None
    side: str = ""
    second_seed: str = ""
    saved: bool = False
    picture: Path | None = None
    error: str = ""
    elapsed_s: float = 0.0
    #: Which assembloid the section is from (``<slide>/A`` …; see :mod:`assembloids`).
    assembloid: str = ""
    #: Unlike both its neighbours on the slide, which are like each other (torn, folded …).
    odd_section: bool = False
    signature: np.ndarray | None = None


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _disk(radius_px: float) -> np.ndarray:
    radius = max(int(round(radius_px)), 1)
    grid = np.arange(-radius, radius + 1)
    return (grid[:, None] ** 2 + grid[None, :] ** 2) <= radius * radius


def _largest(mask: np.ndarray) -> np.ndarray:
    from scipy import ndimage

    labels, count = ndimage.label(mask)
    if count <= 1:
        return mask.astype(bool)
    return labels == (int(np.argmax(np.bincount(labels.ravel())[1:])) + 1)


def pick_channel(names: Sequence[str], wanted: int | str, hints: Sequence[str]) -> int | None:
    """Index of the channel *wanted* names, else the first one a hint matches."""
    from .experiment import pick_channel as by_name

    if wanted not in ("", None):
        return by_name(names, wanted)
    for hint in hints:
        index = by_name(names, hint)
        if index is not None:
            return index
    return None


# ---------------------------------------------------------------------------
# The overview
# ---------------------------------------------------------------------------


def pick_other_channel(names: Sequence[str], wanted: int | str, nuclear: int, label: int) -> int | None:
    """The channel labelling the other organoid: named, hinted, or what is left."""
    taken = {nuclear, label}
    if wanted not in ("", None):
        index = pick_channel(names, wanted, ())
        return index if index is not None and index not in taken else None
    for hint in OTHER_HINTS:
        index = pick_channel(names, hint, ())
        if index is not None and index not in taken:
            return index
    left = [index for index in range(len(names)) if index not in taken]
    return left[0] if left else None


def focus_score(plane: np.ndarray) -> float:
    """How sharp *plane* is: variance of its Laplacian over its mean squared.

    Dimensionless, so a dim stain and a bright one compare. It means little as
    a number; it is for finding the samples that sit far below their cohort.
    """
    from scipy import ndimage

    data = np.asarray(plane, dtype=np.float32)
    mean = float(data.mean())
    if data.size == 0 or mean <= 0:
        return float("nan")
    return float(ndimage.laplace(data).var() / (mean * mean))


def build_overview(
    specs: Sequence[Any],
    options: JunctionOptions | None = None,
    target_um: float = OVERVIEW_UM,
) -> Overview:
    """Flatten and decimate every channel of one sample.

    *specs* are the reader's layer specs, one per channel. Each plane is read
    whole and every *step*-th pixel kept — not averaged, because the labelled
    cells are a few pixels across at full resolution and averaging a block of
    sixteen would dissolve them into the background they are told apart from.
    """
    options = options or JunctionOptions()
    names = [str(getattr(spec, "channel_name", "") or getattr(spec, "name", "")) for spec in specs]
    nuclear = pick_channel(names, options.nuclear_channel, NUCLEAR_HINTS)
    label = pick_channel(names, options.label_channel, LABEL_HINTS)
    if nuclear is None:
        raise ValueError(f"no nuclear channel among: {', '.join(names) or 'none'}")
    if label is None:
        raise ValueError(f"no labelled-cell channel among: {', '.join(names) or 'none'}")
    other = pick_other_channel(names, options.other_channel, nuclear, label)

    first = specs[0]
    data = first.data[0] if getattr(first, "multiscale", False) else first.data
    shape_yx = tuple(int(n) for n in data.shape[-2:])
    voxel_yx = tuple(float(v) for v in first.scale)[-2:]
    pixel = max(voxel_yx) or 1.0
    step = max(1, int(round(target_um / pixel)))

    planes: list[np.ndarray] = []
    focus, saturated = float("nan"), 0.0
    nuclear_mean: np.ndarray | None = None
    for index, spec in enumerate(specs):
        array = spec.data[0] if getattr(spec, "multiscale", False) else spec.data
        stack = array.reshape((-1, *array.shape[-2:]))
        best: np.ndarray | None = None
        for z in range(int(stack.shape[0])):
            plane = np.asarray(stack[z])
            if index == nuclear and z == int(stack.shape[0]) // 2:
                focus = focus_score(_centre_crop(plane))
                if np.issubdtype(plane.dtype, np.integer):
                    limit = np.iinfo(plane.dtype).max
                    saturated = float((plane[::step, ::step] >= limit).mean())
            small = plane[::step, ::step].astype(np.float32)
            best = small if best is None else np.maximum(best, small)
            if index == nuclear:
                pooled = _block_mean(plane, step, small.shape)
                nuclear_mean = pooled if nuclear_mean is None else np.maximum(nuclear_mean, pooled)
        planes.append(best)
    return Overview(
        channels=np.stack(planes), names=names, um=pixel * step, voxel_yx=voxel_yx,
        shape_yx=shape_yx, step=step, nuclear=nuclear, label=label, other=other,
        focus=focus, saturated=saturated, nuclear_mean=nuclear_mean,
    )


def _block_mean(plane: np.ndarray, step: int, shape: tuple[int, int]) -> np.ndarray:
    """Mean of every *step* × *step* block, on the grid striding gives."""
    rows, cols = (plane.shape[0] // step) * step, (plane.shape[1] // step) * step
    pooled = plane[:rows, :cols].reshape(rows // step, step, cols // step, step).mean(
        axis=(1, 3), dtype=np.float32
    )
    out = np.zeros(shape, dtype=np.float32)
    out[: pooled.shape[0], : pooled.shape[1]] = pooled[: shape[0], : shape[1]]
    return out


def _centre_crop(plane: np.ndarray, size: int = 1536) -> np.ndarray:
    """The middle of the non-empty part of *plane*, where the tissue is."""
    rows = np.flatnonzero(plane[:, :: max(1, plane.shape[1] // 256)].any(axis=1))
    cols = np.flatnonzero(plane[:: max(1, plane.shape[0] // 256), :].any(axis=0))
    if rows.size == 0 or cols.size == 0:
        return plane[:size, :size]
    cy, cx = int(rows.mean()), int(cols.mean())
    y0, x0 = max(cy - size // 2, 0), max(cx - size // 2, 0)
    return plane[y0:y0 + size, x0:x0 + size]


# ---------------------------------------------------------------------------
# Tissue, density, split
# ---------------------------------------------------------------------------


def _above_background(plane: np.ndarray, valid: np.ndarray, um: float) -> np.ndarray | None:
    """Where one channel, smoothed, stands above the slide; ``None`` if it cannot tell.

    The cut sits a fifth of the way from background to tissue rather than at
    Otsu's midpoint: one organoid of a pair is routinely dimmer than the other,
    and the midpoint drops it.
    """
    from scipy import ndimage
    from skimage.filters import threshold_otsu

    smooth = ndimage.gaussian_filter(np.asarray(plane, dtype=np.float32), 10.0 / um)
    values = smooth[valid]
    if values.size < 100 or float(values.max()) <= float(values.min()):
        return None
    midpoint = threshold_otsu(values)
    below, above = values[values < midpoint], values[values >= midpoint]
    if below.size == 0 or above.size == 0:
        return None
    cut = float(np.median(below)) + 0.2 * (float(np.median(above)) - float(np.median(below)))
    return (smooth > cut) & valid


def tissue_mask(nuclear: np.ndarray, valid: np.ndarray, um: float,
                others: Sequence[np.ndarray] = ()) -> np.ndarray:
    """Where the tissue is, from the nuclear stain. Holes are left as holes.

    With *others* (the other channels), tissue is wherever *any* channel stands
    above the slide: an organoid whose nuclear stain is faint — common in one
    of a pair — is still bright in its own label or its autofluorescence, and
    the nuclear stain alone drops it, cutting the section (and the junction
    line) short.
    """
    from scipy import ndimage

    mask = _above_background(nuclear, valid, um)
    if mask is None:
        return np.zeros(np.shape(nuclear), dtype=bool)
    for plane in others:
        more = _above_background(plane, valid, um)
        if more is not None:
            mask |= more
    mask = ndimage.binary_opening(mask, _disk(10.0 / um))
    mask = ndimage.binary_closing(mask, _disk(15.0 / um))
    labels, count = ndimage.label(mask)
    if count == 0:
        return mask
    sizes = np.bincount(labels.ravel())[1:]
    # Debris is dropped; a second organoid lying beside the first is not.
    keep = np.flatnonzero(sizes >= 0.1 * sizes.max()) + 1
    return np.isin(labels, keep)


def cell_density(
    channel: np.ndarray, tissue: np.ndarray, filled: np.ndarray, um: float,
    radius_um: float = 100.0, median=True, median_um: float = MEDIAN_UM,
) -> np.ndarray:
    """Local density of labelled cells, as a smooth map over the tissue.

    A cell is a point brighter than its surroundings, so the map is built from
    what is left after a median of the neighbourhood is taken away. Diffuse
    signal — autofluorescence, a fold, a thick part of the section — goes with
    the median. The smoothing is normalised by the tissue under the kernel, so
    the edge of the section is not read as a fall in density.

    The median is taken over *median_um* (an odd number of pixels, at least
    3). With *median* off the channel is used as it is.
    """
    from scipy import ndimage

    prominence = cell_prominence(channel, um, median, median_um)
    inside = prominence[tissue]
    if inside.size == 0:
        return np.zeros(prominence.shape, dtype=np.float32)
    prominence = np.clip(prominence / (float(np.percentile(inside, 99.5)) + 1e-6), 0, 1)
    sigma = radius_um / um
    amount = ndimage.gaussian_filter(np.where(tissue, prominence, 0).astype(np.float32), sigma)
    weight = ndimage.gaussian_filter(tissue.astype(np.float32), sigma)
    return np.where(filled, amount / np.maximum(weight, 1e-3), 0).astype(np.float32)


def cell_prominence(channel: np.ndarray, um: float, median: bool = True,
                    median_um: float = MEDIAN_UM) -> np.ndarray:
    """What :func:`cell_density` counts: the channel less its local median.

    With *median* off, or a size of 0, the channel itself.
    """
    from scipy import ndimage

    data = np.asarray(channel, dtype=np.float32)
    if not median or median_um <= 0:
        return np.clip(data, 0, None)
    size = max(3, int(round(float(median_um) / um)) | 1)
    return np.clip(data - ndimage.median_filter(data, size), 0, None)


def has_cells(channel: np.ndarray, tissue: np.ndarray, um: float) -> bool:
    """Whether *channel* holds labelled cells at all, or only its own texture.

    :func:`cell_density` scales a channel by its own brightest points, so an
    empty channel comes out looking as full as a labelled one — the noise is
    simply stretched. This asks first whether there is anything to stretch: the
    brightest points over the local median, against the spread of everything
    else over it.
    """
    from scipy import ndimage

    data = np.asarray(channel, dtype=np.float32)
    size = max(3, int(round(45.0 / um)) | 1)
    over = (data - ndimage.median_filter(data, size))[tissue]
    if over.size < 100:
        return False
    spread = float(np.median(np.abs(over - np.median(over)))) * 1.4826
    if spread <= 0:
        return False
    return float(np.percentile(over, 99.9)) / spread > CELLS_OVER_NOISE


def line_cost(nuclear: np.ndarray, filled: np.ndarray, um: float) -> np.ndarray:
    """What it costs to cut the tissue at each pixel: little along a dark line.

    Where two organoids meet the nuclear stain shows it — a thin dark line, or
    the edge where a dim organoid lies against a bright one. This is the stain
    divided by its own neighbourhood: 1 in the middle of either organoid, bright
    or dim alike, and below 1 on anything darker than what surrounds it. Raised
    to a power, so a line a fifth darker than its surroundings is several times
    cheaper to cut along than plain tissue.

    The stain is smoothed just enough to merge single nuclei (~8 µm); the line
    is a dozen µm wide and a wider blur fills it in.
    """
    from scipy import ndimage

    data = np.asarray(nuclear, dtype=np.float32)
    inside = data[filled]
    if inside.size == 0:
        return np.ones(data.shape, dtype=np.float32)
    level = float(np.median(inside))
    # Outside the tissue takes the tissue's own level, so the section's edge is
    # not itself a dark line the cut would rather run along.
    data = np.where(filled, data, level)
    fine = ndimage.gaussian_filter(data, LINE_SMOOTH_UM / um)
    coarse = ndimage.gaussian_filter(data, LINE_SURROUND_UM / um)
    ratio = np.clip(fine / np.maximum(coarse, 1e-6), 0.3, 1.0)
    return (ratio ** LINE_POWER).astype(np.float32)


def tear_mask(nuclear: np.ndarray, tissue: np.ndarray, filled: np.ndarray, um: float) -> np.ndarray:
    """Where the section is torn: holes in the tissue, and stain that is all but gone.

    A junction between two organoids is *dimmer* than its surroundings; a tear
    is *empty*. Both are dark, and the line cost (:func:`line_cost`) makes the
    darkest the cheapest — so a tear is the best road a cut can find. This
    tells them apart by level: holes the tissue mask left, and anything whose
    smoothed stain is under :data:`TEAR_LEVEL` of the tissue's median.
    """
    from scipy import ndimage

    data = np.asarray(nuclear, dtype=np.float32)
    inside = data[filled]
    if inside.size == 0:
        return np.zeros(data.shape, dtype=bool)
    level = float(np.median(data[tissue])) if tissue.any() else float(np.median(inside))
    fine = ndimage.gaussian_filter(np.where(filled, data, level), LINE_SMOOTH_UM / um)
    torn = (filled & ~tissue) | (filled & (fine < TEAR_LEVEL * level))
    return ndimage.binary_dilation(torn) & filled


def edge_cost(
    density: np.ndarray, filled: np.ndarray, hot: np.ndarray, cold: np.ndarray,
    other_density: np.ndarray | None = None,
) -> np.ndarray:
    """0 where the labelled cells change from one organoid's level to the other's, up to 1 away from it.

    The density is put on a scale where the labelled seed is 1 and the other
    seed 0; the junction is where it is halfway, and the cost grows with the
    distance from halfway (squared). With the other organoid's own label the
    same is done for it, the other way round, and the two are averaged. A cut
    that slides towards one pole, or follows a tear across which the cells do
    not change, pays for it.
    """
    def scaled(values, high, low):
        top, bottom = float(np.median(values[high & filled])), float(np.median(values[low & filled]))
        if not np.isfinite(top) or not np.isfinite(bottom) or abs(top - bottom) < 1e-9:
            return None
        return np.clip((values - bottom) / (top - bottom), 0.0, 1.0)

    parts = [part for part in (scaled(density, hot, cold),
                               scaled(other_density, cold, hot) if other_density is not None else None)
             if part is not None]
    if not parts:
        return np.zeros(filled.shape, dtype=np.float32)
    share = np.mean(parts, axis=0)
    return np.where(filled, (2.0 * (share - 0.5)) ** 2, 0.0).astype(np.float32)


def label_change_price(
    label: np.ndarray, tissue: np.ndarray, filled: np.ndarray, um: float,
    other: np.ndarray | None = None, strength: float = 1.0,
    radius_um: float = LABEL_CHANGE_UM, median: bool = True, median_um: float = MEDIAN_UM,
) -> np.ndarray:
    """A factor on the cutting price: 1 where the labels are flat, lower where they change.

    Where two organoids meet, one label falls and the other rises, whether or
    not the nuclear stain shows a dark line there. Each label channel becomes
    a density (:func:`cell_density`, smoothed over *radius_um*) scaled 0–1;
    with the other organoid's label the two are subtracted, so the change is
    twice as clear where one falls as the other rises. The steepness of that
    map, as a share of its 99th percentile over the tissue, times *strength*,
    is taken off 1, down to :data:`LABEL_CHANGE_FLOOR`.

    Outside the tissue the map takes the value of the nearest tissue pixel,
    so the section's own edge is not read as a change.
    """
    from scipy import ndimage

    ones = np.ones(filled.shape, dtype=np.float32)
    share = _normalised(cell_density(label, tissue, filled, um, radius_um, median, median_um), filled)
    if share is None or not filled.any():
        return ones
    if other is not None:
        second = _normalised(cell_density(other, tissue, filled, um, radius_um, median, median_um), filled)
        if second is not None:
            share = share - second
    nearest = ndimage.distance_transform_edt(~filled, return_distances=False, return_indices=True)
    share = share[nearest[0], nearest[1]].astype(np.float32)
    rows, cols = np.gradient(share)
    steep = np.hypot(rows, cols)
    top = float(np.percentile(steep[filled], 99))
    if top <= 0:
        return ones
    factor = 1.0 - float(strength) * np.clip(steep / top, 0, 1)
    return np.where(filled, np.clip(factor, LABEL_CHANGE_FLOOR, 1.0), 1.0).astype(np.float32)


def _min_cut(filled: np.ndarray, hot: np.ndarray, cold: np.ndarray, cost: np.ndarray) -> np.ndarray | None:
    """Cheapest cut of the tissue that separates *hot* from *cold*.

    Pixels are joined to their eight neighbours, each join as costly to cut as
    the mean of the two pixels' *cost*; the minimum cut is then the shortest
    way across the tissue between the seeds, shortened further wherever it can
    run along something cheap. Returns the *hot* side as a mask.

    A sum over the whole cut is what makes this follow a line: a single dark
    speck saves one pixel's worth, a continuous dark line saves all of it.
    """
    from scipy import sparse
    from scipy.sparse import csgraph

    index = np.full(filled.shape, -1, dtype=np.int64)
    count = int(filled.sum())
    index[filled] = np.arange(count)
    source, sink = count, count + 1

    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    caps: list[np.ndarray] = []
    height, width = filled.shape
    # Diagonal joins are longer and so cut less of the boundary each; weighting
    # them down keeps a cut at 45° from costing more than one along the grid.
    for dy, dx, weight in ((0, 1, 1.0), (1, 0, 1.0), (1, 1, 0.7071), (1, -1, 0.7071)):
        a = index[max(0, -dy): height - max(0, dy), max(0, -dx): width - max(0, dx)]
        b = index[max(0, dy):, max(0, dx): width + min(0, dx)] if dx >= 0 else \
            index[max(0, dy):, : width + dx]
        if dx < 0:
            a = index[: height - dy, -dx:]
        ca = cost[max(0, -dy): height - max(0, dy), max(0, -dx): width - max(0, dx)] if dx >= 0 \
            else cost[: height - dy, -dx:]
        cb = cost[max(0, dy):, max(0, dx): width + min(0, dx)] if dx >= 0 else cost[max(0, dy):, : width + dx]
        both = (a >= 0) & (b >= 0)
        capacity = np.maximum(((ca[both] + cb[both]) * 0.5 * weight * 1000).astype(np.int32), 1)
        rows.extend((a[both], b[both]))
        cols.extend((b[both], a[both]))
        caps.extend((capacity, capacity))

    huge = np.int32(2 ** 30)
    for seed, node, outward in ((hot, source, True), (cold, sink, False)):
        nodes = index[seed & filled]
        if nodes.size == 0:
            return None
        ends = np.full(nodes.size, node, dtype=np.int64)
        rows.append(ends if outward else nodes)
        cols.append(nodes if outward else ends)
        caps.append(np.full(nodes.size, huge, dtype=np.int32))

    graph = sparse.csr_matrix(
        (np.concatenate(caps), (np.concatenate(rows), np.concatenate(cols))),
        shape=(count + 2, count + 2), dtype=np.int32,
    )
    flow = csgraph.maximum_flow(graph, source, sink).flow
    residual = graph - flow
    residual.data = np.maximum(residual.data, 0)
    residual.eliminate_zeros()
    reached = csgraph.breadth_first_order(residual, source, return_predecessors=False)
    side = np.zeros(count + 2, dtype=bool)
    side[reached] = True
    out = np.zeros(filled.shape, dtype=bool)
    out[filled] = side[:count]
    return out


def _normalised(density: np.ndarray, filled: np.ndarray) -> np.ndarray | None:
    """*density* from 0 to 1 between its 10th and 98th percentile in the tissue."""
    inside = density[filled]
    if inside.size == 0:
        return None
    low, high = np.percentile(inside, (10, 98))
    if high <= low:
        return None
    return np.clip((density - low) / (high - low), 0, 1)


def _through_tissue(filled: np.ndarray, start: np.ndarray) -> np.ndarray:
    """Distance of every tissue pixel from *start*, going through the tissue.

    Straight-line distance would call the far side of a hole close. Zero
    outside the tissue and where it cannot be reached.
    """
    from skimage.graph import MCP

    walk = MCP(np.where(filled, 1.0, 1e9), fully_connected=True)
    costs, _trace = walk.find_costs([tuple(point) for point in np.argwhere(start)[::7]])
    return np.where(filled & (costs < 1e8), costs, 0.0)


def _far_part(seed: np.ndarray, distance: np.ndarray) -> np.ndarray:
    """The half of *seed* furthest by *distance*; *seed* itself if that is nothing."""
    away = np.where(seed, distance, 0.0)
    part = seed & (away > 0.5 * float(away.max()))
    return _largest(part) if part.any() else seed


def find_seeds(
    filled: np.ndarray, density: np.ndarray, other_density: np.ndarray | None = None,
    patches: str = "",
) -> tuple[np.ndarray, np.ndarray, str] | None:
    """One seed in each organoid: ``(labelled, other, how the second was made)``.

    The **labelled** seed is the dense core of *density*: the largest patch
    where the labelled cells are at more than half their range.

    The **other** seed is made by the same operation on *other_density* — the
    density of the other organoid's own label — when there is one: the largest
    patch where *those* cells are at more than half their range, outside the
    first core and where they outweigh the labelled cells. A seed that is a
    positive sign of the second organoid, rather than the absence of the first.
    It is used when it is a real patch (:data:`SEED_MIN_SHARE` of the tissue).

    Otherwise the other seed is the old one: tissue with few labelled cells,
    further from the labelled core than half the furthest tissue.

    Either way the seeds are then evened out. A dense core reaches right up to
    the junction; left like that, the cut hugs whichever seed is smaller. Each
    core is cut back to its half furthest from the other seed, so both start
    well inside their own organoid.

    ``None`` when there is nothing to seed one of the halves from.
    """
    normal = _normalised(density, filled)
    if normal is None:
        return None
    hot = _largest((normal > 0.5) & filled)
    if not hot.any():
        return None
    if patches:
        # Every sizeable patch of labelled cells, not only the largest: a labelled
        # organoid can have dense patches at both ends (see SEED_RULE "patches").
        from scipy import ndimage

        pieces, count = ndimage.label((normal > 0.5) & filled)
        sizes = np.bincount(pieces.ravel())[1:]
        big = np.flatnonzero(sizes >= SEED_MIN_SHARE * filled.sum()) + 1
        if big.size:
            hot = np.isin(pieces, big)
    from_hot = _through_tissue(filled, hot)

    cold, rule = None, SEED_FAR
    if other_density is not None:
        other = _normalised(other_density, filled)
        if other is not None:
            core = (other > 0.5) & filled & ~hot & (other > normal)
            if core.any():
                core = _largest(core)
                if core.sum() >= SEED_MIN_SHARE * filled.sum():
                    # The same pulling back the labelled core gets below.
                    cold, rule = _far_part(core, from_hot), SEED_CHANNEL
    if cold is None:
        cold = filled & (from_hot > 0.5 * float(from_hot.max())) & (normal < 0.2)
        if not cold.any():
            return None
        cold = _largest(cold)

    if patches == "patches":
        return hot, cold, rule                     # every patch, whole
    from_cold = _through_tissue(filled, cold)
    if patches == "patches_pulled":
        # Each patch pulled back on its own: its half furthest from the other seed.
        from scipy import ndimage

        pieces, count = ndimage.label(hot)
        kept = np.zeros_like(hot)
        for k in range(1, count + 1):
            piece = pieces == k
            away = np.where(piece, from_cold, 0.0)
            kept |= piece & (away > 0.5 * float(away.max()))
        return (kept if kept.any() else hot), cold, rule
    hot = _far_part(hot, from_cold)
    return hot, cold, rule


def dominant_seeds(
    filled: np.ndarray, density: np.ndarray, other_density: np.ndarray | None = None,
    lead: float | None = None,
) -> tuple[np.ndarray, np.ndarray, str] | None:
    """One seed in each organoid, as large as the labels allow: ``(labelled, other, how)``.

    :func:`find_seeds` takes each organoid's densest core. Where the two meet
    without a dark line, the cut is then the shortest way round the smaller
    core — often across a corner of one organoid. Here each seed is all the
    tissue where its own label leads the other's by *lead* (both scaled 0–1
    over the tissue): the cut is left only the mixed strip between them, which
    is where the junction is when nothing else marks it. With no label for the
    other organoid, its seed is where the labelled cells are scarce.

    ``None`` when either seed is not a real patch (:data:`SEED_MIN_SHARE`).
    *lead* defaults to :data:`SEED_LEAD`, read at call time.
    """
    lead = SEED_LEAD if lead is None else lead
    normal = _normalised(density, filled)
    if normal is None:
        return None
    other = _normalised(other_density, filled) if other_density is not None else None
    margin = (normal - other) if other is not None else (2.0 * normal - 1.0)
    hot = _largest(filled & (margin > lead))
    cold = _largest(filled & (margin < -lead))
    least = SEED_MIN_SHARE * filled.sum()
    if hot.sum() < least or cold.sum() < least:
        return None
    return hot, cold, SEED_DOMINANT


def split_tissue(
    filled: np.ndarray, density: np.ndarray, um: float, cost: np.ndarray | None = None,
    other_density: np.ndarray | None = None, seeds: tuple[np.ndarray, np.ndarray] | None = None,
    bend_um: float = 0.0,
) -> np.ndarray | None:
    """The tissue as two halves: 1 the labelled organoid, 2 the other.

    The two seeds (see :func:`find_seeds`; *seeds* passes them in ready-made)
    say roughly where each organoid is. Between them the tissue is cut where
    cutting is cheapest: the shortest way across, bent onto the dark line the
    nuclear stain shows between two organoids when *cost* (see
    :func:`line_cost`) is given. Without it the cut is simply the shortest,
    which finds a neck and nothing subtler.

    With *bend_um*, the cut is then straightened (see :func:`straighten_cut`):
    turning costs, so it no longer follows a crack round a corner.

    ``None`` when there is nothing to seed one of the halves from.
    """
    if seeds is None:
        found = find_seeds(filled, density, other_density)
        if found is None:
            return None
        seeds = found[:2]
    hot, cold = seeds

    price = np.ones(filled.shape, dtype=np.float32) if cost is None else np.asarray(cost, np.float32)
    labelled = _min_cut(filled, hot, cold, price)
    if labelled is None:
        return None
    halves = np.zeros(filled.shape, dtype=np.uint8)
    halves[_largest(labelled & filled)] = 1
    halves[_largest(~labelled & filled)] = 2
    if not (halves == 1).any() or not (halves == 2).any():
        return None
    if bend_um > 0:
        halves = straighten_cut(halves, filled, price, hot, cold, um, bend_um)
    return halves


def straighten_cut(
    halves: np.ndarray, filled: np.ndarray, cost: np.ndarray,
    hot: np.ndarray, cold: np.ndarray, um: float, bend_um: float,
) -> np.ndarray:
    """The cut between the seeds again, this time paying for every turn it takes.

    A minimum cut only sums what it crosses, so a crack in the section is as
    good a road as the line between the organoids, and the cut will turn a
    right angle to follow one. Here the cut is a curve across the tissue,
    chosen to minimise what it crosses (*cost* per µm, as before) plus
    *bend_um* × (turn / 90°)² at every station along it — bending energy, so
    one sharp corner costs a lot and the same turn as a gentle curve little.

    The curve runs across the axis from one seed to the other: at each station
    along the perpendicular to that axis it sits somewhere between the seeds,
    found by dynamic programming over that position and the heading, so its
    angles are exact rather than the pixel grid's. Off the tissue it costs
    nothing, which lets its ends meet the outline wherever is cheapest; it may
    not cross a seed. The search is over the whole section, not near the old
    cut, so a cut that went round a corner is not where this one starts. The
    halves are the two sides of the curve; where that fails — both seeds on
    one side — the old halves are kept.
    """
    from scipy import ndimage

    if not hot.any() or not cold.any():
        return halves
    hot_at = np.argwhere(hot & filled).mean(axis=0)
    cold_at = np.argwhere(cold & filled).mean(axis=0)
    axis = cold_at - hot_at
    span = float(np.hypot(*axis))
    if span * um < 4 * BEND_STATION_UM:
        return halves
    across = axis / span                           # offsets: from the hot seed to the cold
    along = np.array([-across[1], across[0]])      # stations: across the section
    start = hot_at

    # Stations cover the section; offsets run from the hot seed to the cold.
    tissue_px = np.argwhere(filled)
    projected = (tissue_px - start) @ along
    station = BEND_STATION_UM / um
    step = BEND_OFFSET_UM / um
    s = np.arange(float(projected.min()) - station, float(projected.max()) + 1.5 * station, station)
    o = np.arange(0.0, span + step / 2, step)
    n_s, n_o = len(s), len(o)
    if n_s < 4 or n_o < 3:
        return halves

    # Cost per µm along the curve: the cut's own price on tissue, nothing off it.
    price = np.where(filled, cost, 0.0).astype(np.float32)
    blocked = ((hot | cold) & filled).astype(np.float32)

    def sample(ss):
        points = (start[:, None, None] + along[:, None, None] * ss[None, :, None]
                  + across[:, None, None] * o[None, None, :]).reshape(2, -1)
        shape = (len(ss), n_o)
        values = ndimage.map_coordinates(price, points, order=1, cval=0.0).reshape(shape)
        seed = ndimage.map_coordinates(blocked, points, order=0, cval=0.0).reshape(shape) > 0
        return np.where(seed, 1e6, values).astype(np.float32)

    at = sample(s)
    mid = sample(s[:-1] + station / 2)

    shifts = np.arange(-BEND_MAX_STEPS, BEND_MAX_STEPS + 1)
    n_d = len(shifts)
    seg_len_um = np.hypot(BEND_STATION_UM, shifts * BEND_OFFSET_UM)
    heading = np.arctan2(shifts * BEND_OFFSET_UM, BEND_STATION_UM)
    turn = (heading[:, None] - heading[None, :]) / (np.pi / 2)
    bend = (float(bend_um) * turn ** 2).astype(np.float32)  # [arriving, leaving]

    big = np.float32(1e12)
    value = np.zeros((n_o, n_d), dtype=np.float32)  # [offset, arriving heading]
    back = np.zeros((n_s - 1, n_o, n_d), dtype=np.int8)
    for i in range(n_s - 1):
        if i == 0:
            # The first segment has nothing to turn from.
            leave = np.zeros((n_o, n_d), dtype=np.float32)
            choice = np.zeros((n_o, n_d), dtype=np.int8)
        else:
            totals = value[:, :, None] + bend[None, :, :]  # [offset, arriving, leaving]
            choice = totals.argmin(axis=1).astype(np.int8)
            leave = np.take_along_axis(totals, choice[:, None, :].astype(np.int64), axis=1)[:, 0, :]
        back[i] = choice
        new = np.full((n_o, n_d), big, dtype=np.float32)
        for j, d in enumerate(shifts):
            lo, hi = max(0, -d), min(n_o, n_o - d)
            if lo >= hi:
                continue
            k = np.arange(lo, hi)
            # Segment from offset k to k + d: its two ends and its middle.
            half = k + d / 2.0
            middle = 0.5 * (mid[i, np.floor(half).astype(int)] + mid[i, np.ceil(half).astype(int)])
            per_um = (at[i, k] + 2.0 * middle + at[i + 1, k + d]) / 4.0
            new[k + d, j] = leave[k, j] + per_um * seg_len_um[j]
        value = new
    if float(value.min()) >= 1e5:
        return halves

    # Walk back from the cheapest end: the offset at every station.
    k, j = (int(n) for n in np.unravel_index(int(np.argmin(value)), value.shape))
    path = [k]
    for i in range(n_s - 2, -1, -1):
        k_prev = k - int(shifts[j])
        j = int(back[i, k_prev, j])
        k = k_prev
        path.append(k)
    offsets = o[np.asarray(path[::-1])]

    # The two sides of the curve: a pixel's offset against the curve's at its
    # station. Beyond the stations the old halves stand.
    rows, cols = np.nonzero(filled)
    ps = (rows - start[0]) * along[0] + (cols - start[1]) * along[1]
    po = (rows - start[0]) * across[0] + (cols - start[1]) * across[1]
    within = (ps >= s[0]) & (ps <= s[-1])
    left = po < np.interp(ps, s, offsets)
    in_hot, in_cold = hot[rows, cols] & within, cold[rows, cols] & within
    if not in_hot.any() or not in_cold.any():
        return halves
    hot_left, cold_left = float(left[in_hot].mean()), float(left[in_cold].mean())
    if abs(hot_left - cold_left) < 0.9:
        return halves  # the curve does not keep the seeds apart
    hot_side = left if hot_left > 0.5 else ~left
    labels = np.where(within, np.where(hot_side, 1, 2), halves[rows, cols])
    out = np.zeros(filled.shape, dtype=np.uint8)
    out[rows, cols] = labels
    straight = np.zeros(filled.shape, dtype=np.uint8)
    straight[_largest(out == 1)] = 1
    straight[_largest(out == 2)] = 2
    if not (straight == 1).any() or not (straight == 2).any():
        return halves
    return straight


def walk_tissue(filled: np.ndarray, cost: np.ndarray, hot: np.ndarray, cold: np.ndarray,
                beta: float = WALK_BETA) -> np.ndarray | None:
    """For every tissue pixel, the chance a random walk from it reaches *hot* before *cold*.

    The walk moves between neighbouring pixels, freely within plain tissue and
    rarely across a change in *cost* — the edges of a dark line (Grady's random
    walker, as scikit-image implements it). Where the minimum cut takes the
    cheapest way across, which a tear running most of the way is, the walk
    simply goes round the end of the tear: only a barrier that closes keeps the
    two sides apart. ``None`` when it cannot be solved.
    """
    from skimage.segmentation import random_walker

    labels = np.zeros(filled.shape, dtype=np.int32)
    labels[~filled] = -1                       # off the tissue: not part of the walk
    labels[hot & filled] = 1
    labels[cold & filled] = 2
    if not (labels == 1).any() or not (labels == 2).any():
        return None
    data = np.where(filled, cost, 1.0).astype(np.float64)
    try:
        chance = random_walker(data, labels, beta=float(beta), mode="cg_j", tol=1e-4, return_full_prob=True)
    except Exception:
        return None
    return np.where(filled, chance[0], 0.0)


def walk_halves(filled, cost, hot, cold, um: float, beta: float = WALK_BETA, bend_um: float = 0.0):
    """The two halves by random walk, the line then straightened along its 50:50 boundary."""
    chance = walk_tissue(filled, cost, hot, cold, beta)
    if chance is None:
        return None
    halves = np.zeros(filled.shape, dtype=np.uint8)
    halves[_largest(filled & (chance >= 0.5))] = 1
    halves[_largest(filled & (chance < 0.5))] = 2
    if not (halves == 1).any() or not (halves == 2).any():
        return None
    if bend_um > 0:
        # Cheap along the 50:50 boundary, dear where the walk is sure: the
        # bending pass then lays a smooth line along the boundary.
        price = (np.abs(chance - 0.5) * 2.0) ** 2 + 0.02
        halves = straighten_cut(halves, filled, price.astype(np.float32), hot, cold, um, bend_um)
    return halves


def cut_signs(halves: np.ndarray, cost: np.ndarray | None, um: float) -> dict[str, float]:
    """What a cut says about itself: its price per pixel along it, and its sharpest turn."""
    line = trace_interface(halves, um)
    if line is None or len(line) < 3:
        return {}
    rc = np.clip(np.round(line).astype(int), 0, np.array(halves.shape) - 1)
    price = float(np.asarray(cost)[rc[:, 0], rc[:, 1]].mean()) if cost is not None else float("nan")
    v = np.diff(line, axis=0)
    heading = np.unwrap(np.arctan2(v[:, 0], v[:, 1]))
    turn = float(np.degrees(np.abs(np.diff(heading)).max())) if len(heading) > 1 else 0.0
    return {"Cut price": price, "Sharpest turn (deg)": turn}


def cut_looks_wrong(signs: dict[str, float]) -> bool:
    """Too cheap (a tear), too dear (no line) or a sharp turn (a crack): see the AUTO_ constants."""
    if not signs:
        return True
    price, turn = signs.get("Cut price", np.nan), signs.get("Sharpest turn (deg)", 0.0)
    return bool((np.isfinite(price) and (price < AUTO_PRICE_LOW or price > AUTO_PRICE_HIGH)) or turn > AUTO_TURN_DEG)


def trace_interface(halves: np.ndarray, um: float) -> np.ndarray | None:
    """The boundary between the two halves, as ordered ``(row, col)`` points.

    The pixels where one half touches the other are thinned to a curve, and the
    curve is walked from one end to the other: the two ends are the pair of
    points furthest apart along it. Smoothed along its length (a running mean
    over :data:`JUNCTION_SMOOTH_UM`) so neither the line nor the band inherits
    the wander of a cut made pixel by pixel.
    """
    from scipy import ndimage
    from skimage.graph import MCP
    from skimage.morphology import skeletonize

    touching = (halves == 1) & ndimage.binary_dilation(halves == 2)
    if not touching.any():
        return None
    curve = skeletonize(_largest(ndimage.binary_dilation(touching)))
    points = np.argwhere(curve)
    if len(points) < 5:
        return None
    cost = np.where(curve, 1.0, 1e9)

    def furthest(start) -> tuple[tuple[int, int], Any]:
        walk = MCP(cost, fully_connected=True)
        costs, _trace = walk.find_costs([tuple(start)])
        costs = np.where(curve & (costs < 1e8), costs, -1.0)
        return tuple(int(n) for n in np.unravel_index(int(np.argmax(costs)), costs.shape)), walk

    one_end, _walk = furthest(points[0])
    other_end, walk = furthest(one_end)
    path = np.asarray(walk.traceback(other_end), dtype=float)
    if len(path) < 5:
        return None
    window = max(3, int(round(JUNCTION_SMOOTH_UM / um)))
    smooth = np.column_stack([
        ndimage.uniform_filter1d(path[:, axis], window, mode="nearest") for axis in (0, 1)
    ])
    return smooth[:: max(1, int(round(40.0 / um)))]


# ---------------------------------------------------------------------------
# The junction and its band
# ---------------------------------------------------------------------------


def find_junction(overview: Overview, options: JunctionOptions | None = None) -> JunctionResult:
    """Tissue, junction, band and quality figures for one overview."""
    from scipy import ndimage

    options = options or JunctionOptions()
    um = overview.um
    channels = overview.channels
    valid = (channels > 0).all(axis=0)  # stitched scans pad with exact zeros
    tissue = tissue_mask(channels[overview.nuclear], valid, um, _others(channels, overview.nuclear, options.tissue_all_channels))
    filled = ndimage.binary_fill_holes(tissue)
    density = cell_density(channels[overview.label], tissue, filled, um,
                           median=options.use_median, median_um=options.median_um)
    result = JunctionResult(tissue=tissue, filled=filled, density=density)
    result.metrics = tissue_metrics(overview, tissue, filled, valid)
    if not tissue.any():
        result.flags = quality_flags(result.metrics)
        return result

    cost = None
    if options.follow_nuclear_line:
        stain = overview.nuclear_mean if overview.nuclear_mean is not None else channels[overview.nuclear]
        cost = line_cost(stain, filled, um)
    other_density = None
    if (options.seed_from_other_channel and overview.other is not None
            and has_cells(channels[overview.other], tissue, um)):
        other_density = cell_density(channels[overview.other], tissue, filled, um,
                                     median=options.use_median, median_um=options.median_um)
    changes: dict[float, np.ndarray] = {}

    def label_factor(strength):
        """The label-change factor at *strength*, made once."""
        if strength not in changes:
            changes[strength] = label_change_price(
                channels[overview.label], tissue, filled, um,
                channels[overview.other] if other_density is not None else None,
                strength, options.label_change_um, options.use_median, options.median_um)
        return changes[strength]

    torn_cache: list[np.ndarray] = []

    def torn():
        if not torn_cache:
            torn_cache.append(tear_mask(channels[overview.nuclear], tissue, filled, um))
        return torn_cache[0]

    def split(seeds, how):
        """The halves between *seeds* under the settings *how*, and the figures the cut gives about itself."""
        found: dict[str, float] = {}
        price = cost
        if how["tear_price"] > 0 or how["edge_weight"] > 0 or how["label_change"] > 0:
            price = np.ones(filled.shape, np.float32) if cost is None else np.asarray(cost, np.float32).copy()
            if how["tear_price"] > 0:
                price[torn()] = np.maximum(price[torn()], float(how["tear_price"]))
            if how["edge_weight"] > 0:
                price += float(how["edge_weight"]) * edge_cost(density, filled, *seeds[:2], other_density)
            if how["label_change"] > 0:
                price *= label_factor(float(how["label_change"]))
        bend = float(how["bend_um"]) if price is not None else 0.0
        walk_price = price if price is not None else np.ones(filled.shape, np.float32)
        if options.split_method == "walk":
            halves = walk_halves(filled, walk_price, *seeds[:2], um, options.walk_beta, bend)
            found["Split"] = 1.0
        else:
            halves = split_tissue(filled, density, um, price, seeds=seeds[:2], bend_um=bend)
            found["Split"] = 0.0
            if halves is not None:
                found.update(cut_signs(halves, price, um))
            if options.split_method == "auto" and (halves is None or cut_looks_wrong(found)):
                walked = walk_halves(filled, walk_price, *seeds[:2], um, options.walk_beta, bend)
                if walked is not None:
                    halves = walked
                    found["Split"] = 1.0
        if halves is not None and cost is not None:
            # How dark the line under the cut is, whatever else was priced in.
            found["Seam price"] = cut_signs(halves, cost, um).get("Cut price", float("nan"))
        return halves, found

    def seeds_by(rule, lead=None):
        found = None
        if rule == "dominant":
            found = dominant_seeds(filled, density, other_density, lead)
        elif rule in ("patches", "patches_pulled"):
            found = find_seeds(filled, density, other_density, patches=rule)
        return found if found is not None else find_seeds(filled, density, other_density)

    first_how = {"tear_price": options.tear_price, "edge_weight": options.edge_weight,
                 "label_change": options.label_change, "bend_um": options.bend_um}
    seeds = seeds_by(options.seed_rule)
    halves = None
    if seeds is not None:
        result.seeds, result.second_seed = seeds[:2], seeds[2]
        halves, found = split(seeds, first_how)
        result.metrics.update(found)
        if (options.no_seam_price > 0 and halves is not None
                and found.get("Seam price", 0.0) > options.no_seam_price):
            # No dark line: the second set of settings, made for sections without one.
            again = seeds_by(options.no_seam_seed_rule, options.no_seam_lead)
            if again is not None:
                retry, found_again = split(again, {
                    "tear_price": options.no_seam_tear_price, "edge_weight": options.no_seam_edge,
                    "label_change": options.no_seam_label_change, "bend_um": options.no_seam_bend_um})
                if retry is not None:
                    # The first cut's seam price stays: it is why this one was made.
                    found_again.pop("Seam price", None)
                    halves = retry
                    result.seeds, result.second_seed = again[:2], again[2]
                    result.metrics.update(found_again)
                    result.metrics["No-seam fallback"] = 1.0
    if halves is not None:
        result.halves = halves
        line = trace_interface(halves, um)
        # A junction shrunk to a point by the smoothing is no junction.
        if line is not None and len(np.unique(np.round(line, 3), axis=0)) >= 2:
            result.line_um = line * um
            _build_band(result, overview, options)
        labelled, other = halves == 1, halves == 2
        result.metrics["Labelled share of tissue"] = float(labelled.sum() / max(filled.sum(), 1))
        result.metrics["Labelled area (mm2)"] = float(labelled.sum() * um * um / 1e6)
        result.metrics["Other area (mm2)"] = float(other.sum() * um * um / 1e6)
        result.metrics["Density contrast"] = float(
            density[labelled].mean() / max(float(density[other].mean()), 1e-9)
        )
    if filled.any():
        result.signature = shape_signature(_largest(filled), result.halves == 1 if result.halves is not None else None, um)
    result.flags = quality_flags(result.metrics)
    return result


def _smooth_mask(mask: np.ndarray, sigma_px: float) -> np.ndarray:
    """*mask* with its outline rounded off: blurred, and cut again at one half.

    Corners sharper than about *sigma_px* go, as do slivers narrower than it;
    the area stays where it was, which an opening or a closing alone would not
    manage.
    """
    from scipy import ndimage

    if sigma_px <= 0 or not mask.any():
        return mask.astype(bool)
    return ndimage.gaussian_filter(mask.astype(np.float32), sigma_px) > 0.5


def _draw_curve(curve_px: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """The pixels a curve passes through, as a mask."""
    from skimage.draw import line as draw_line

    mask = np.zeros(shape, dtype=bool)
    points = np.round(np.asarray(curve_px)).astype(int)
    for (r0, c0), (r1, c1) in zip(points[:-1], points[1:]):
        rows, cols = draw_line(int(r0), int(c0), int(r1), int(c1))
        keep = (rows >= 0) & (rows < shape[0]) & (cols >= 0) & (cols < shape[1])
        mask[rows[keep], cols[keep]] = True
    return mask


def _side_of_line(result: JunctionResult, curve_px: np.ndarray, into_unlabelled: bool) -> np.ndarray:
    """The tissue on one side of the junction *as drawn*.

    The halves meet on the cut as it was found; the junction is that cut
    smoothed, and the band has to be on one side of the line people see, not of
    the one it came from. So the tissue is divided again by the smoothed line —
    run on a little past both ends, since smoothing pulls them in from the edge
    — and each piece goes to whichever half most of it was in.
    """
    from scipy import ndimage

    wanted = result.halves == (2 if into_unlabelled else 1)
    curve = np.asarray(curve_px, dtype=float)
    reach = 40.0  # overview pixels: enough to cross the outline from a pulled-in end
    head = curve[0] + (curve[0] - curve[min(3, len(curve) - 1)]) / max(
        np.linalg.norm(curve[0] - curve[min(3, len(curve) - 1)]), 1e-9) * reach
    tail = curve[-1] + (curve[-1] - curve[max(-4, -len(curve))]) / max(
        np.linalg.norm(curve[-1] - curve[max(-4, -len(curve))]), 1e-9) * reach
    wall = _draw_curve(np.vstack((head, curve, tail)), result.filled.shape)
    wall = ndimage.binary_dilation(wall, np.ones((3, 3), dtype=bool))
    pieces, count = ndimage.label(result.filled & ~wall)
    if count < 2:
        return wanted
    inside = ndimage.sum(wanted, pieces, index=np.arange(1, count + 1))
    sizes = ndimage.sum(np.ones_like(wanted), pieces, index=np.arange(1, count + 1))
    keep = np.flatnonzero(inside > 0.5 * sizes) + 1
    side = np.isin(pieces, keep)
    # The wall itself belongs to nobody; the gap from the line keeps it out.
    return ndimage.binary_dilation(side, np.ones((3, 3), dtype=bool)) & result.filled


def _build_band(result: JunctionResult, overview: Overview, options: JunctionOptions) -> None:
    """Make the band on the overview grid, and measure it.

    The band is every part of one organoid that lies within the band width of
    the junction. That is not the macro's strip with a cap at either end: a
    strip pushed out sideways from the line ends in two straight cuts across
    the tissue, wherever the line happened to stop. Taking the organoid's own
    tissue by distance instead lets the band end where the organoid does — its
    two extremities run along the outline of the organoid, and the far edge is
    the line the junction would be if it were moved in by the band width.

    Everything is then rounded off (see :func:`_smooth_mask`): the tissue
    outline it is clipped to, and the band itself.
    """
    from scipy import ndimage

    um = overview.um
    shape = result.filled.shape
    curve = fu.fit_spline(result.line_um, spacing=um)
    on_line = _draw_curve(curve / um, shape)
    distance = ndimage.distance_transform_edt(~on_line) * um

    target = _side_of_line(result, curve / um, options.into_unlabelled)
    tissue = _smooth_mask(result.filled, SMOOTH_UM / um) if options.clip_to_tissue else np.ones(shape, bool)
    if options.clip_to_tissue and float(options.margin_um) > 0:
        # A rim outside the outline goes with the organoid it borders: cells
        # migrating over its surface sit just outside what the stain calls tissue.
        # Never into the other organoid's tissue, which is not outside anything.
        rim = (ndimage.distance_transform_edt(~target) * um <= float(options.margin_um)) & ~result.filled
        target = target | rim
        tissue = tissue | rim
    band = target & tissue & (distance <= float(options.gap_um) + float(options.width_um))
    band &= distance >= float(options.gap_um)
    band = _smooth_mask(band, SMOOTH_UM / um) & tissue & (distance >= float(options.gap_um))
    # What smoothing leaves of a band is one piece and its crumbs.
    band = _largest(band) if band.any() else band

    # Which side of the line that is, for the record and for the panel's switch.
    both, _curve = fu.fusion_region(result.line_um, shape, (um, um), width_um=options.width_um,
                                    gap_um=options.gap_um, side=fu.SIDE_BOTH)
    overlap = [int(((both == label) & band).sum()) for label in (1, 2)]
    result.side = fu.SIDE_LEFT if overlap[0] >= overlap[1] else fu.SIDE_RIGHT

    result.band = band
    result.band_outline_um = _outline(band) * um if band.any() else None

    pixel_area = um * um
    channels = overview.channels
    result.metrics["Junction length (um)"] = fu.line_length(curve)
    result.metrics["Band area (mm2)"] = float(band.sum() * pixel_area / 1e6)
    on_tissue = band & result.tissue
    result.metrics["Band tissue area (mm2)"] = float(on_tissue.sum() * pixel_area / 1e6)
    for index, name in enumerate(overview.names):
        values = channels[index][on_tissue]
        result.metrics[f"Band mean {name}"] = float(values.mean()) if values.size else float("nan")
    inside = result.density[on_tissue]
    result.metrics["Band labelled-cell density (a.u.)"] = (
        float(inside.mean()) if inside.size else float("nan")
    )


def _outline(mask: np.ndarray) -> np.ndarray:
    """The largest outline of *mask*, as ``(row, col)`` points, lightly smoothed."""
    from scipy import ndimage
    from skimage.measure import find_contours

    contours = find_contours(np.pad(mask, 1).astype(float), 0.5)
    if not contours:
        return np.zeros((0, 2))
    longest = max(contours, key=len) - 1.0
    if len(longest) > 12:
        # The contour walks the pixel grid; a short running mean (wrapped, as
        # the outline is closed) takes the staircase out of it.
        longest = np.column_stack([
            ndimage.uniform_filter1d(longest[:, axis], 5, mode="wrap") for axis in (0, 1)
        ])
    return longest[:: max(1, len(longest) // 600)]


def full_resolution_band(
    result: JunctionResult, overview: Overview, options: JunctionOptions | None = None
) -> np.ndarray:
    """The band as a label map on the image's own pixel grid.

    Filled from the band's outline rather than enlarged from its overview mask:
    an outline scales to any grid and stays smooth, where a mask blown up
    sixteen times is a staircase of sixteen-pixel steps.
    """
    labels = np.zeros(tuple(int(n) for n in overview.shape_yx), dtype=np.uint8)
    outline = result.band_outline_um
    if outline is None or len(outline) < 3:
        return labels
    voxel = np.asarray(overview.voxel_yx, dtype=float)
    # Overview pixel k covers full-resolution pixels k*step … k*step + step - 1.
    pixels = outline / voxel + (overview.step - 1) / 2.0
    low = np.maximum(np.floor(pixels.min(axis=0)).astype(int) - 1, 0)
    high = np.minimum(np.ceil(pixels.max(axis=0)).astype(int) + 2, labels.shape)
    if np.all(high > low):
        box = tuple(int(n) for n in (high - low))
        labels[low[0]:high[0], low[1]:high[1]] = fu.fill_polygon(pixels - low, box)
    return labels


def _others(channels: np.ndarray, nuclear: int, use: bool) -> list[np.ndarray]:
    """The channels besides the nuclear one, when the tissue is found in all of them."""
    return [plane for k, plane in enumerate(channels) if k != nuclear] if use else []


def tissue_of(overview: Overview, all_channels: bool = TISSUE_ALL_CHANNELS) -> np.ndarray:
    """The section's outline, holes filled, on the overview grid."""
    from scipy import ndimage

    valid = (overview.channels > 0).all(axis=0)
    tissue = tissue_mask(overview.channels[overview.nuclear], valid, overview.um,
                         _others(overview.channels, overview.nuclear, all_channels))
    return ndimage.binary_fill_holes(tissue)


def band_along_tissue(
    overview: Overview,
    filled: np.ndarray,
    line_um: np.ndarray,
    side: str,
    width_um: float,
    gap_um: float = 0.0,
    margin_um: float = fu.TISSUE_MARGIN_UM,
) -> np.ndarray:
    """The band of a line drawn by hand, made the way the automatic run makes its own.

    The organoid's own tissue within *width_um* of the line, on *side* of it —
    so its ends follow the organoid's outline instead of cutting straight
    across, and nothing off the section is in it. *line_um* is in µm from the
    image origin. Returns a label map on the image's own grid: 1 for the band,
    and with :data:`fusion.SIDE_BOTH` 1 left of the line and 2 right of it.
    """
    um = overview.um
    shape = filled.shape
    line = np.asarray(line_um, dtype=float).reshape(-1, 2)
    # Which side is which, as the macro sees it: its two strips, made wide
    # enough to reach across the section, stand in for the automatic run's
    # two organoids, and the tissue is divided between them along the line.
    reach = float(max(shape) * um)
    halves, _curve = fu.fusion_region(line, shape, (um, um), width_um=reach, gap_um=0.0,
                                      side=fu.SIDE_BOTH)
    result = JunctionResult(tissue=filled, filled=filled, density=np.zeros(shape, np.float32),
                            halves=np.where(filled, halves, 0), line_um=line)
    labels = np.zeros(tuple(int(n) for n in overview.shape_yx), dtype=np.uint8)
    sides = {fu.SIDE_LEFT: (1,), fu.SIDE_RIGHT: (2,), fu.SIDE_BOTH: (1, 2)}[side]
    for value in sides:
        options = JunctionOptions(width_um=float(width_um), gap_um=float(gap_um), margin_um=float(margin_um),
                                  into_unlabelled=value == 2, clip_to_tissue=True)
        _build_band(result, overview, options)
        band = full_resolution_band(result, overview, options)
        labels[(band > 0) & (labels == 0)] = value if side == fu.SIDE_BOTH else 1
    return labels


# ---------------------------------------------------------------------------
# Quality
# ---------------------------------------------------------------------------


def tissue_metrics(
    overview: Overview, tissue: np.ndarray, filled: np.ndarray, valid: np.ndarray
) -> dict[str, float]:
    """Figures describing the section itself, before any junction."""
    from scipy import ndimage
    from skimage.morphology import convex_hull_image

    pixel_area = overview.um * overview.um
    metrics: dict[str, float] = {
        "Tissue area (mm2)": float(tissue.sum() * pixel_area / 1e6),
        "Focus": float(overview.focus),
        "Saturated fraction": float(overview.saturated),
    }
    if not tissue.any():
        return metrics
    labels, count = ndimage.label(tissue)
    sizes = np.sort(np.bincount(labels.ravel())[1:])[::-1]
    metrics["Pieces"] = float(count)
    metrics["Second piece share"] = float(sizes[1] / sizes[0]) if count > 1 else 0.0
    metrics["Hole share"] = float((filled.sum() - tissue.sum()) / max(filled.sum(), 1))
    hull = convex_hull_image(_largest(filled))
    metrics["Solidity"] = float(_largest(filled).sum() / max(hull.sum(), 1))
    # Tissue running into where the scan stops is tissue that was not all imaged.
    border = ~ndimage.binary_erosion(valid, iterations=2, border_value=0)
    edge = filled & ~ndimage.binary_erosion(filled)
    metrics["Cut off by the scan"] = float((edge & border).sum() / max(edge.sum(), 1))
    values = overview.channels[overview.nuclear][tissue]
    metrics["Nuclear stain (mean)"] = float(values.mean())
    metrics.update(shape_metrics(_largest(filled), overview.um))
    return metrics


#: The shape signature's grid: this many pixels a side, this many µm each.
SIGNATURE_PX = 64
SIGNATURE_UM = 60.0


def shape_signature(outline: np.ndarray, labelled: np.ndarray | None, um: float) -> np.ndarray:
    """The section as a small picture of fixed scale: ``(2, SIGNATURE_PX, SIGNATURE_PX)`` bool.

    Plane 0 the tissue, plane 1 its labelled half (empty when not split),
    centred on the tissue's centre and turned so its long axis runs along the
    grid's rows, at :data:`SIGNATURE_UM` µm a pixel. Serial sections of one
    assembloid overlap almost entirely on it, up to a half turn or a mirror
    (see :func:`assembloids.signature_distance`); another assembloid does not.
    """
    from scipy import ndimage

    n = SIGNATURE_PX
    out = np.zeros((2, n, n), dtype=bool)
    points = np.argwhere(outline).astype(float)
    if len(points) < 3:
        return out
    centre = points.mean(axis=0)
    values, vectors = np.linalg.eigh(np.cov((points - centre).T))
    major, minor = vectors[:, int(np.argmax(values))], vectors[:, int(np.argmin(values))]
    offsets = (np.arange(n) - (n - 1) / 2) * SIGNATURE_UM / um
    across, along = np.meshgrid(offsets, offsets, indexing="ij")
    coords = centre[:, None, None] + along[None] * major[:, None, None] + across[None] * minor[:, None, None]
    out[0] = ndimage.map_coordinates(outline.astype(np.uint8), coords, order=0, cval=0) > 0
    if labelled is not None:
        out[1] = ndimage.map_coordinates(labelled.astype(np.uint8), coords, order=0, cval=0) > 0
    return out


def shape_metrics(outline: np.ndarray, um: float) -> dict[str, float]:
    """The section's shape, for telling serial sections of one assembloid from another's.

    Size (perimeter, long and short axis) and four numbers that describe the
    shape whatever its size, rotation or mirroring (the first Hu moments, as
    −log10 of their size). Neighbouring sections of one assembloid change
    these a little at a time; a jump means another assembloid.
    """
    from skimage import measure

    if not outline.any():
        return {}
    region = measure.regionprops(outline.astype(np.uint8))[0]
    out = {"Perimeter (um)": float(region.perimeter * um),
           "Major axis (um)": float(region.axis_major_length * um),
           "Minor axis (um)": float(region.axis_minor_length * um)}
    hu = measure.moments_hu(measure.moments_normalized(measure.moments_central(outline.astype(float))))
    for k in range(4):
        out[f"Shape {k + 1}"] = float(-np.sign(hu[k]) * np.log10(abs(hu[k]))) if hu[k] != 0 else float("nan")
    return out


def quality_flags(metrics: dict[str, float]) -> list[str]:
    """Reasons to look at a sample, from its own figures alone.

    Cohort outliers — a section much smaller, dimmer or blurrier than the rest
    of the folder — are added by :func:`flag_cohort`, which needs the folder.
    """
    flags: list[str] = []
    get = metrics.get
    if not get("Tissue area (mm2)", 0.0) > 0:
        return ["no tissue found"]
    if get("Pieces", 1) > 1 and get("Second piece share", 0.0) >= 0.25:
        flags.append("organoids not fused")
    if get("Hole share", 0.0) > 0.10:
        flags.append("large holes or torn")
    if get("Solidity", 1.0) < 0.75:
        flags.append("ragged outline")
    if get("Cut off by the scan", 0.0) > 0.05:
        flags.append("cut off by the scan edge")
    if get("Saturated fraction", 0.0) > 0.01:
        flags.append("saturated")
    if "Junction length (um)" not in metrics:
        flags.append("no junction found")
        return flags
    if get("Junction length (um)", 0.0) < 150:
        flags.append("junction very short")
    if get("Seam price", 0.0) > NO_SEAM_PRICE:
        flags.append("no dark line at the junction")
    if get("Density contrast", np.inf) < 1.8:
        flags.append("labelled cells on both sides (junction uncertain)")
    share = get("Labelled share of tissue", 0.5)
    if share < 0.1 or share > 0.9:
        flags.append("one organoid much smaller than the other")
    if get("Band tissue area (mm2)", 1.0) < 0.02:
        flags.append("hardly any tissue in the band")
    return flags


#: Figures compared across the folder, and what being an outlier in each is called.
COHORT_CHECKS = (
    ("Tissue area (mm2)", "both", "size unlike the rest"),
    ("Nuclear stain (mean)", "both", "staining unlike the rest"),
    ("Focus", "low", "out of focus"),
)

#: Robust z beyond which a figure is an outlier. 3.5 is the usual cut for a
#: median/MAD score; with two hundred samples it flags a handful, not a tenth.
COHORT_Z = 3.5


def flag_cohort(outcomes: Sequence[JunctionOutcome]) -> None:
    """Add the flags that only mean something against the rest of the folder."""
    usable = [outcome for outcome in outcomes if not outcome.error]
    if len(usable) < 8:
        return
    for key, direction, flag in COHORT_CHECKS:
        values = np.array([outcome.metrics.get(key, np.nan) for outcome in usable], dtype=float)
        logs = np.log(np.where(values > 0, values, np.nan))
        centre = np.nanmedian(logs)
        spread = np.nanmedian(np.abs(logs - centre)) * 1.4826
        if not np.isfinite(spread) or spread <= 0:
            continue
        scores = (logs - centre) / spread
        for outcome, score in zip(usable, scores):
            if not np.isfinite(score):
                continue
            if score < -COHORT_Z or (direction == "both" and score > COHORT_Z):
                if flag not in outcome.flags:
                    outcome.flags.append(flag)
    for outcome in usable:
        if any(flag in FAILING for flag in outcome.flags):
            outcome.status = STATUS_FAIL
        else:
            outcome.status = STATUS_CHECK if outcome.flags else STATUS_OK


# ---------------------------------------------------------------------------
# Pictures
# ---------------------------------------------------------------------------


def _stretch(plane: np.ndarray, valid: np.ndarray) -> np.ndarray:
    values = plane[valid]
    if values.size == 0:
        return np.zeros(plane.shape, dtype=np.float32)
    low, high = np.percentile(values, (1.0, 99.8))
    return np.clip((plane - low) / max(float(high - low), 1.0), 0, 1)


def qc_picture(overview: Overview, result: JunctionResult, title: str = "", size: int = 640):
    """The overview with the tissue outline, the junction and the band drawn in.

    A PIL image: nuclear stain in blue, labelled cells in green, the remaining
    channel in red. White is the tissue outline, yellow the junction, magenta
    the band. The caption carries the status and what was flagged.
    """
    from PIL import Image, ImageDraw
    from scipy import ndimage

    channels = overview.channels
    valid = (channels > 0).all(axis=0)
    others = [i for i in range(len(channels)) if i not in (overview.nuclear, overview.label)]
    red = _stretch(channels[others[0]], valid) if others else np.zeros(valid.shape, np.float32)
    rgb = np.stack([red, _stretch(channels[overview.label], valid),
                    _stretch(channels[overview.nuclear], valid)], axis=-1)
    rgb = (rgb * 255).astype(np.uint8)
    if result.filled.any():
        # The outline the band was clipped to, which is the rounded one.
        outline = _smooth_mask(result.filled, SMOOTH_UM / overview.um)
        rgb[ndimage.binary_dilation(outline) & ~outline] = (255, 255, 255)
    if result.band is not None and result.band.any():
        rim = ndimage.binary_dilation(result.band, iterations=2) & ~result.band
        rgb[rim] = (255, 0, 255)
    image = Image.fromarray(rgb)
    if result.line_um is not None:
        curve = fu.fit_spline(result.line_um, spacing=overview.um) / overview.um
        ImageDraw.Draw(image).line([(float(x), float(y)) for y, x in curve],
                                   fill=(255, 255, 0), width=3)
    scale = size / max(image.size)
    image = image.resize((max(1, int(image.size[0] * scale)), max(1, int(image.size[1] * scale))))
    canvas = Image.new("RGB", (size, size + CAPTION_PX), "black")
    canvas.paste(image, (0, CAPTION_PX))
    caption(canvas, title, result.status, "; ".join(result.flags) or "no flags")
    return canvas


#: Height of the strip of text above a QC picture.
CAPTION_PX = 34

STATUS_COLOURS = {STATUS_OK: (120, 220, 120), STATUS_CHECK: (255, 200, 60), STATUS_FAIL: (255, 90, 90)}


def caption(canvas, title: str, status: str, detail: str) -> None:
    """Write the strip above a QC picture: the sample and its status, then why.

    Painted over whatever was there, so a picture whose status changed after it
    was drawn — the folder-wide checks come last — can be captioned again.
    """
    from PIL import ImageDraw

    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, canvas.size[0], CAPTION_PX - 1), fill="black")
    draw.text((4, 3), f"{title}  [{status.upper()}]", fill=STATUS_COLOURS.get(status, (220, 220, 220)))
    draw.text((4, 18), detail, fill=(220, 220, 220))


def problem_picture(title: str, problem: str, size: int = 640):
    """The picture of a sample there is nothing to show of: its name and why."""
    from PIL import Image, ImageDraw

    canvas = Image.new("RGB", (size, 200 + CAPTION_PX), "black")
    caption(canvas, title, STATUS_FAIL, "no picture: the sample could not be processed")
    draw = ImageDraw.Draw(canvas)
    words, lines, line = str(problem).split(), [], ""
    for word in words:
        if len(line) + len(word) + 1 > size // 7:
            lines.append(line)
            line = ""
        line = f"{line} {word}".strip()
    lines.append(line)
    for index, text in enumerate(lines[:10]):
        draw.text((4, CAPTION_PX + 12 + 15 * index), text, fill=STATUS_COLOURS[STATUS_FAIL])
    return canvas


def file_picture(folder: str | Path, name: str, status: str, picture=None) -> Path:
    """Put a sample's QC picture in its status folder. Returns where it is.

    ``<folder>/ok``, ``<folder>/check`` and ``<folder>/fail`` each hold the
    pictures of the samples with that status, so the ones to look at are one
    folder and not a column to filter. A sample has one picture: the one from
    an earlier run, in whichever folder (or loose in *folder*, where they used
    to go), is removed. *picture* is a PIL image to save; without one, the
    sample's existing picture is moved.
    """
    folder = Path(folder)
    for each in STATUSES:
        (folder / each).mkdir(parents=True, exist_ok=True)
    target = folder / status / f"{name}.png"
    elsewhere = [folder / f"{name}.png", *(folder / each / f"{name}.png" for each in STATUSES if each != status)]
    if picture is not None:
        picture.save(target)
    else:
        found = next((path for path in elsewhere if path.is_file()), None)
        if found is not None and not target.is_file():
            found.replace(target)
    for path in elsewhere:
        if path.is_file():
            path.unlink()
    return target


def contact_sheets(pictures: Sequence[Path], folder: Path, columns: int = 5, rows: int = 4) -> list[Path]:
    """Tile the per-sample pictures into pages that can be flicked through."""
    from PIL import Image

    written: list[Path] = []
    per_page = columns * rows
    tile = 400
    for page, start in enumerate(range(0, len(pictures), per_page), start=1):
        batch = list(pictures[start:start + per_page])
        sheet = Image.new("RGB", (columns * tile, rows * (tile + 21)), "black")
        for index, path in enumerate(batch):
            with Image.open(path) as picture:
                small = picture.resize((tile, int(tile * picture.size[1] / picture.size[0])))
            sheet.paste(small, ((index % columns) * tile, (index // columns) * (tile + 21)))
        target = folder / f"junction_QC_sheet_{page:02d}.png"
        sheet.save(target)
        written.append(target)
    return written


# ---------------------------------------------------------------------------
# One sample, and a folder of them
# ---------------------------------------------------------------------------


def analyse_specs(specs: Sequence[Any], options: JunctionOptions | None = None):
    """Overview and junction for one sample already read. Returns both."""
    options = options or JunctionOptions()
    overview = build_overview(specs, options)
    return overview, find_junction(overview, options)


def store(path: Path, overview: Overview, result: JunctionResult, options: JunctionOptions) -> None:
    """Write the band into the sample's store: a label map and an outline.

    The label map is the region itself. The outline is the same region as a
    stored ROI, which is what the batch segmentation restricts to and what the
    Analysis panel counts inside — so the junction band behaves like any region
    drawn by hand. Other stored outlines are kept; an earlier band is replaced.
    """
    from . import ims_store

    labels = full_resolution_band(result, overview, options)
    attrs = {
        "source": "automatic junction",
        "band_width_um": float(options.width_um),
        "start_gap_um": float(options.gap_um),
        "cap_length_um": float(options.cap_um) or float(options.width_um),
        "past_tissue_edge_um": float(options.margin_um),
        "side": result.side,
        "status": result.status,
        "flags": "; ".join(result.flags),
        # How it was found: whether the second search (no dark line) made it.
        "second_search": float(result.metrics.get("No-seam fallback", 0.0)),
        "seam_price": float(result.metrics.get("Seam price", float("nan"))),
        "line_yx_um": np.asarray(result.line_um, dtype=np.float64),
    }
    ims_store.save_labels(path, options.label_key, labels, overview.voxel_yx, attrs=attrs)
    if result.band_outline_um is not None and len(result.band_outline_um) >= 3:
        kept = [roi for roi in ims_store.load_rois(path) if roi.name != BAND_ROI_NAME]
        band = ims_store.StoredRoi(
            name=BAND_ROI_NAME, vertices_um=result.band_outline_um,
            attrs={"source": "automatic junction", "status": result.status},
        )
        ims_store.save_rois(path, [*kept, band])


def process_file(path: str | Path, options: JunctionOptions | None = None) -> JunctionOutcome:
    """Find, store and picture the junction of one file. Never raises."""
    from . import loaders

    options = options or JunctionOptions()
    target = Path(path)
    outcome = JunctionOutcome(path=target, name=fu_name(target))
    started = time.perf_counter()
    was_open = loaders.is_open(target)
    try:
        specs = loaders.load_path(target)
        overview, result = analyse_specs(specs, options)
        outcome.metrics, outcome.flags = dict(result.metrics), list(result.flags)
        outcome.status, outcome.line_um, outcome.side = result.status, result.line_um, result.side
        outcome.second_seed = result.second_seed
        outcome.signature = result.signature
        if options.qc_dir is not None:
            outcome.picture = file_picture(options.qc_dir, outcome.name, outcome.status,
                                           qc_picture(overview, result, outcome.name))
        if options.save and result.line_um is not None and result.band is not None:
            store(target, overview, result, options)
            outcome.saved = True
    except Exception as exc:
        logger.exception("junction detection failed on %s", target.name)
        outcome.error = str(exc)
        outcome.status = STATUS_FAIL
        if options.qc_dir is not None:
            try:
                outcome.picture = file_picture(options.qc_dir, outcome.name, STATUS_FAIL,
                                               problem_picture(outcome.name, outcome.error))
            except Exception:
                logger.exception("could not write the QC picture of %s", target.name)
    finally:
        outcome.elapsed_s = time.perf_counter() - started
        if not was_open:
            try:
                loaders.release(target)
            except Exception:
                logger.debug("could not release %s", target, exc_info=True)
    return outcome


def fu_name(path: Path) -> str:
    """The sample's name: the file name without ``.ome.tiff`` and the like."""
    name = path.name
    for suffix in (".ome.tiff", ".ome.tif"):
        if name.lower().endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def run(
    paths: Sequence[str | Path],
    options: JunctionOptions | None = None,
    progress: Callable[[str], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
    workers: int | None = None,
) -> list[JunctionOutcome]:
    """Every file, then the folder-wide checks, sheets and workbook.

    *workers* files are handled at once, each in its own process
    (:data:`RUN_WORKERS` when ``None``; 1 = one after another). The outcomes
    come back in the order of *paths* whatever order they finish in.

    With ``options.qc_dir`` every run leaves a QC folder: a picture per sample
    sorted into ``ok/``, ``check/`` and ``fail/``, the workbook, and the contact
    sheets.
    """
    options = options or JunctionOptions()
    workers = RUN_WORKERS if workers is None else max(1, int(workers))
    outcomes = _process_all(list(paths), options, progress, should_cancel, workers)
    flag_cohort(outcomes)
    group_assembloids(outcomes)
    if options.qc_dir is not None and outcomes:
        folder = Path(options.qc_dir)
        folder.mkdir(parents=True, exist_ok=True)
        try:
            _refile(outcomes, folder)
            qc_dataframe(outcomes).to_excel(folder / "junction_QC.xlsx", index=False)
            contact_sheets([o.picture for o in outcomes if o.picture is not None], folder)
        except Exception:
            logger.exception("could not write the QC report into %s", folder)
    return outcomes


def _process_all(paths, options, progress, should_cancel, workers) -> list[JunctionOutcome]:
    """:func:`process_file` on every path: in *workers* processes, or one after another.

    A pool that cannot be started (or breaks) is not fatal: what it did not
    finish is done here, one after another. Cancelling stops new samples from
    starting; those already running finish.
    """
    done: dict[int, JunctionOutcome] = {}

    def report(index: int, outcome: JunctionOutcome) -> None:
        done[index] = outcome
        if progress is not None:
            detail = outcome.error or "; ".join(outcome.flags) or "no flags"
            progress(f"{outcome.name} ({len(done)} of {len(paths)}): {outcome.status} — {detail}")

    cancelled = lambda: should_cancel is not None and should_cancel()  # noqa: E731
    if workers > 1 and len(paths) > 1:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor, as_completed

        try:
            with ProcessPoolExecutor(min(workers, len(paths)),
                                     mp_context=multiprocessing.get_context("spawn")) as pool:
                futures = {pool.submit(process_file, path, options): k for k, path in enumerate(paths)}
                for future in as_completed(futures):
                    report(futures[future], future.result())
                    if cancelled():
                        for pending in futures:
                            pending.cancel()
                        break
        except Exception:
            logger.exception("the parallel run failed; doing the rest one after another")
    for k, path in enumerate(paths):
        if k in done:
            continue
        if cancelled():
            break
        report(k, process_file(path, options))
    return [done[k] for k in sorted(done)]


def group_assembloids(outcomes: Sequence[JunctionOutcome]) -> None:
    """Which sections of the folder are from the same assembloid, from their names and shapes."""
    import pandas as pd

    from . import assembloids

    usable = [outcome for outcome in outcomes if not outcome.error]
    if not usable:
        return
    try:
        grouped = assembloids.group_sections(
            pd.DataFrame([{"name": o.name, **o.metrics, "Signature": o.signature} for o in usable]))
    except Exception:
        logger.exception("could not group the sections by assembloid")
        return
    for outcome, name, odd in zip(usable, grouped["Assembloid"], grouped["Odd section"]):
        outcome.assembloid, outcome.odd_section = str(name), bool(odd)


def _refile(outcomes: Sequence[JunctionOutcome], folder: Path) -> None:
    """Move the pictures whose status the folder-wide checks changed.

    A sample is pictured when it is processed, and compared with the rest of
    the folder only at the end; one that turns out unlike the others goes from
    ``ok/`` to ``check/`` then, with its caption saying why.
    """
    from PIL import Image

    for outcome in outcomes:
        picture = outcome.picture
        if picture is None or not picture.is_file() or picture.parent.name == outcome.status:
            continue
        with Image.open(picture) as stored:
            canvas = stored.convert("RGB")
        caption(canvas, outcome.name, outcome.status, outcome.error or "; ".join(outcome.flags) or "no flags")
        outcome.picture = file_picture(folder, outcome.name, outcome.status, canvas)


def qc_dataframe(outcomes: Sequence[JunctionOutcome]):
    """One row per sample: status, flags, and every figure behind them."""
    import pandas as pd

    keys: list[str] = []
    for outcome in outcomes:
        keys.extend(key for key in outcome.metrics if key not in keys)
    rows = []
    for outcome in outcomes:
        row = {
            "Sample": outcome.name,
            "Assembloid": outcome.assembloid,
            "Odd section": outcome.odd_section,
            "Status": outcome.status,
            "Flags": "; ".join(outcome.flags),
            "Band side": outcome.side,
            "Second seed": outcome.second_seed,
            "Saved": outcome.saved,
            "Problem": outcome.error,
            "Picture": (f"{outcome.picture.parent.name}/{outcome.picture.name}"
                        if outcome.picture is not None else ""),
        }
        row.update({key: outcome.metrics.get(key, float("nan")) for key in keys})
        row["File"] = str(outcome.path)
        rows.append(row)
    return pd.DataFrame(rows)
