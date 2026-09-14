"""Find the mosaic overview in a folder and say where each sample sits on it.

Imaris writes a multi-field overview as a numbered series — ``overview_F0000.ims``
through ``overview_F0024.ims``, one file per stage position — and every one of
them records the absolute stage extents it was taken at. So does every ordinary
acquisition. The whole geometry is therefore already in the files: nothing here
registers, correlates or guesses anything, each tile is placed where the stage
says it was and each sample is marked at the coordinates it was imaged from.

Two things follow. The overview stops being twenty-five near-identical rows in
the figure and becomes one picture, and that picture can carry a numbered box per
sample, so a reader can see which fish the panels below came from.

Nothing here imports Qt, napari or pptx. Tiles arrive as :class:`Tile` objects
carrying a callable that returns their plane, so the caller keeps control of how
the pixels are read — which pyramid level, which projection.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from .utils import get_logger

logger = get_logger("overview")

#: The field-series suffix Imaris gives each position of a multi-field
#: acquisition. Three digits at least: a bare ``_F0`` is what the software leaves
#: behind when an acquisition is aborted, and those files are empty stubs that
#: have nothing to do with an overview.
_FIELD_SUFFIX = re.compile(r"^(?P<base>.+)_F(?P<index>\d{3,})$")

#: Fewest fields that count as an overview. Two tiles already make a map worth
#: showing; one is just a file that happens to be named like a field.
MIN_TILES = 2

#: Longest edge, in pixels, of the stitched mosaic. The overview is a locator,
#: not data — it is looked at at slide size, and 1600 px across a 30 mm slide is
#: already about 20 µm per pixel.
DEFAULT_MAX_PIXELS = 1600

#: Fraction of a tile's edge over which its weight ramps from nothing to full.
#: Fields overlap by about 10%, and a hard edge inside that overlap shows up as a
#: visible grid; ramping across it hides the seam without touching the pixels
#: anywhere else.
_FEATHER = 0.12

#: How far a closeup may stick out of the field it was taken from, as a fraction
#: of its own width. The stage repeats to a few micrometres and a field re-centred
#: by eye can end up a hair over the edge of the one it was picked from; a sample
#: that is genuinely half outside was not taken from it.
_NEST_SLACK = 0.05

#: Largest a closeup may be relative to the field it sits in. Two acquisitions of
#: the same field contain each other and neither is a closeup of the other, and
#: two images that merely start at the same corner are not related at all. Every
#: real step down clears this easily: 40x inside 20x is 0.5, and even 60x inside
#: 40x is 0.67.
_NEST_RATIO = 0.7

#: How much of the overview to show around a sample that no other sample
#: contains, as a multiple of the sample's own size. The whole mosaic is already
#: the locator slide — this is meant to be the next step in, close enough that the
#: box is a shape rather than a dot.
_WINDOW = 6.0


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def field_base(name: str) -> str | None:
    """The series name behind a field file, or ``None`` if this is not one.

    ``overview_F0007`` belongs to the series ``overview``. Grouping by that base
    is what keeps two overviews acquired on the same day apart from each other.
    """
    match = _FIELD_SUFFIX.match(str(name or ""))
    return match.group("base") if match else None


def field_index(name: str) -> int:
    """Position of a field within its series; ``-1`` when the name has none."""
    match = _FIELD_SUFFIX.match(str(name or ""))
    return int(match.group("index")) if match else -1


def _is_flat(sample) -> bool:
    """Whether every channel of *sample* is a single plane.

    An overview is a flat map of the slide; the acquisitions taken from it are
    z-stacks. That difference is what stops a genuine multi-position 3D
    experiment, which Imaris also names ``_F####``, from being swallowed as an
    overview.
    """
    for channel in getattr(sample, "channels", ()) or ():
        if int(getattr(getattr(channel, "data", None), "ndim", 0)) > 2:
            return False
    return True


def split_tiles(samples: Sequence) -> tuple[list, list]:
    """Separate overview fields from real samples.

    A sample is an overview field when its name carries the ``_F####`` suffix, it
    records where the stage was, and it is a single plane. Fields are grouped by
    series name and a group is only taken when it holds at least :data:`MIN_TILES`
    of them, so one oddly named file never disappears out of the figure.

    Returns ``(samples, tiles)``, both in a stable order: the samples as they came
    in, the tiles by series and then by field number.
    """
    groups: dict[str, list] = {}
    for sample in samples:
        base = field_base(getattr(sample, "name", ""))
        if base is None or getattr(sample, "stage_extent", None) is None or not _is_flat(sample):
            continue
        groups.setdefault(base, []).append(sample)

    taken = {base: group for base, group in groups.items() if len(group) >= MIN_TILES}
    if not taken:
        return list(samples), []

    if len(taken) > 1:
        # Separate series share one stage frame, so they still stitch together;
        # worth logging because the mosaic will then cover both.
        logger.info("overview series found: %s", ", ".join(sorted(taken)))

    tiles = [sample for base in sorted(taken) for sample in sorted(taken[base], key=lambda s: field_index(s.name))]
    identity = {id(tile) for tile in tiles}
    rest = [sample for sample in samples if id(sample) not in identity]
    logger.info("%d overview field(s) split off from %d sample(s)", len(tiles), len(rest))
    return rest, tiles


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Box:
    """An axis-aligned area of the stage, in micrometres."""

    x0: float
    x1: float
    y0: float
    y1: float

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def centre(self) -> tuple[float, float]:
        return (0.5 * (self.x0 + self.x1), 0.5 * (self.y0 + self.y1))

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    def contains(self, other: "Box", slack: float = 0.0) -> bool:
        """Whether *other* lies inside this box, allowing *slack* µm of overhang."""
        return (
            self.x0 - slack <= other.x0
            and other.x1 <= self.x1 + slack
            and self.y0 - slack <= other.y0
            and other.y1 <= self.y1 + slack
        )


def box_from_extent(extent: Sequence[float] | None) -> Box | None:
    """A :class:`Box` from an ``(x0, x1, y0, y1, z0, z1)`` stage extent."""
    if extent is None or len(extent) < 4:
        return None
    x0, x1, y0, y1 = (float(v) for v in extent[:4])
    return Box(min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1))


def union(boxes: Iterable[Box]) -> Box | None:
    """The smallest box containing every one of *boxes*."""
    boxes = [box for box in boxes if box is not None]
    if not boxes:
        return None
    return Box(
        min(box.x0 for box in boxes),
        max(box.x1 for box in boxes),
        min(box.y0 for box in boxes),
        max(box.y1 for box in boxes),
    )


@dataclass
class Tile:
    """One field of the overview: where it was taken, and how to read it."""

    box: Box
    #: Returns the field's 2D plane. Called once, at stitch time.
    read: Callable[[], np.ndarray]
    name: str = ""


@dataclass
class Marker:
    """One sample's footprint on the overview."""

    number: int
    name: str
    box: Box
    #: True when the sample sits outside the area the overview fields cover, so
    #: the canvas had to be extended to show it.
    outside: bool = False


@dataclass
class Mosaic:
    """The stitched overview and the stage coordinates it spans.

    ``image`` is ``uint8`` RGB. Row 0 is the low-Y edge of ``box`` and column 0
    the low-X edge, which is the convention Imaris stores its data in — verified
    against the overlap between neighbouring fields rather than assumed.
    """

    image: np.ndarray
    box: Box
    um_per_px: float
    tiles: int
    #: Area actually covered by fields, which is smaller than ``box`` when a
    #: sample outside the overview forced the canvas wider.
    covered: Box

    @property
    def aspect(self) -> float:
        return float(self.image.shape[1]) / float(self.image.shape[0])

    def fraction(self, x_um: float, y_um: float) -> tuple[float, float]:
        """Stage coordinates as fractions of the image, measured from its top left."""
        return (
            (float(x_um) - self.box.x0) / self.box.width,
            (float(y_um) - self.box.y0) / self.box.height,
        )

    def fractions_of(self, box: Box) -> tuple[float, float, float, float]:
        """A stage box as ``(left, top, width, height)`` fractions of the image."""
        left, top = self.fraction(box.x0, box.y0)
        return left, top, box.width / self.box.width, box.height / self.box.height

    def view_of(self, box: Box) -> np.ndarray:
        """The part of ``image`` covering a stage box, as a writable view.

        Used to put the scale bar inside the area the fields actually cover: when
        a sample was imaged past the edge of the overview the canvas is wider than
        the picture, and a bar in that blank margin is a bar drawn on nothing.
        """
        left, top, width, height = self.fractions_of(box)
        rows, columns = self.image.shape[:2]
        y0 = max(0, min(rows - 1, int(round(top * rows))))
        x0 = max(0, min(columns - 1, int(round(left * columns))))
        y1 = max(y0 + 1, min(rows, int(round((top + height) * rows))))
        x1 = max(x0 + 1, min(columns, int(round((left + width) * columns))))
        return self.image[y0:y1, x0:x1]


# ---------------------------------------------------------------------------
# Stitching
# ---------------------------------------------------------------------------


def _resize(plane: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resample a plane to an exact size, preferring a proper filter."""
    if plane.shape[:2] == (height, width):
        return plane.astype(np.float32, copy=False)
    try:
        from PIL import Image

        return np.asarray(
            Image.fromarray(plane.astype(np.float32), mode="F").resize((width, height), Image.LANCZOS),
            dtype=np.float32,
        )
    except Exception:
        logger.debug("no PIL resize available; sampling the tile instead", exc_info=True)
        rows = np.clip((np.arange(height) * plane.shape[0] / height).astype(int), 0, plane.shape[0] - 1)
        columns = np.clip((np.arange(width) * plane.shape[1] / width).astype(int), 0, plane.shape[1] - 1)
        return plane.astype(np.float32)[np.ix_(rows, columns)]


def _feather(height: int, width: int) -> np.ndarray:
    """Weights that fall off towards a tile's edges, for blending the overlap."""

    def ramp(size: int) -> np.ndarray:
        distance = np.minimum(np.arange(size), np.arange(size)[::-1]).astype(np.float32)
        span = max(1.0, _FEATHER * size)
        # Never quite zero: a pixel only one tile covers must still count.
        return np.clip(distance / span, 0.0, 1.0) + 1e-3

    return ramp(height)[:, None] * ramp(width)[None, :]


def stitch(
    tiles: Sequence[Tile],
    max_pixels: int = DEFAULT_MAX_PIXELS,
    include: Sequence[Box] = (),
) -> Mosaic | None:
    """Blend the fields into one image, placed by their stage coordinates.

    Fields overlap by about a tenth of their width. They are averaged across that
    overlap with a weight that fades towards each field's edge, so the seam is not
    a visible line. Intensities are otherwise left alone — every field keeps the
    brightness it was acquired with, and the contrast stretch at the end is
    applied to the whole mosaic at once, so a field is never brightened relative
    to its neighbours.

    *include* extends the canvas to cover boxes that lie outside the fields, which
    is what keeps a sample imaged beyond the edge of the overview in its true
    place instead of clamped onto the border.
    """
    if not tiles:
        return None

    covered = union(tile.box for tile in tiles)
    if covered is None or covered.width <= 0 or covered.height <= 0:
        return None
    bounds = union([covered, *[box for box in include if box is not None]]) or covered

    scale = float(max_pixels) / max(bounds.width, bounds.height)
    width = max(1, int(round(bounds.width * scale)))
    height = max(1, int(round(bounds.height * scale)))

    total = np.zeros((height, width), dtype=np.float32)
    weight = np.zeros((height, width), dtype=np.float32)

    for tile in tiles:
        plane = np.asarray(tile.read(), dtype=np.float32)
        if plane.ndim != 2 or plane.size == 0:
            logger.warning("skipping overview field %s: not a 2D plane", tile.name or "?")
            continue
        tile_width = max(1, int(round(tile.box.width * scale)))
        tile_height = max(1, int(round(tile.box.height * scale)))
        left = int(round((tile.box.x0 - bounds.x0) * scale))
        top = int(round((tile.box.y0 - bounds.y0) * scale))

        # Clip rather than trust the arithmetic: a rounded-up tile can overrun the
        # canvas by a pixel at the far edge.
        tile_width = min(tile_width, width - left)
        tile_height = min(tile_height, height - top)
        if tile_width <= 0 or tile_height <= 0:
            continue

        resized = _resize(plane, tile_width, tile_height)
        mask = _feather(tile_height, tile_width)
        total[top : top + tile_height, left : left + tile_width] += resized * mask
        weight[top : top + tile_height, left : left + tile_width] += mask

    painted = weight > 1e-6
    if not painted.any():
        return None

    blended = np.zeros_like(total)
    np.divide(total, weight, out=blended, where=painted)

    from .contrast import HIGH_PERCENTILE, LOW_PERCENTILE

    low, high = (float(v) for v in np.percentile(blended[painted], [LOW_PERCENTILE, HIGH_PERCENTILE]))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low, high = float(blended[painted].min()), float(blended[painted].max())
    if high <= low:
        high = low + 1.0

    grey = np.clip((blended - low) / (high - low), 0.0, 1.0)
    # Anything no field reached is not black data, it is nothing; white says so
    # on a white slide.
    grey[~painted] = 1.0
    image = np.repeat((grey * 255).astype(np.uint8)[..., None], 3, axis=2)

    logger.info(
        "stitched %d field(s) into %dx%d px at %.1f µm/px", len(tiles), width, height, 1.0 / scale
    )
    return Mosaic(image=image, box=bounds, um_per_px=1.0 / scale, tiles=len(tiles), covered=covered)


def markers(samples: Sequence, covered: Box | None = None) -> list[Marker]:
    """One :class:`Marker` per sample that knows where it was imaged.

    Numbering follows the order the samples appear in the figure, so marker 3 on
    the overview is the third row of the deck. Samples with no stage coordinates
    are skipped but still consume their number, which is what keeps that promise
    when one file in a folder came from a different microscope.
    """
    out: list[Marker] = []
    for number, sample in enumerate(samples, start=1):
        box = box_from_extent(getattr(sample, "stage_extent", None))
        if box is None:
            continue
        out.append(
            Marker(
                number=number,
                name=str(getattr(sample, "name", "") or ""),
                box=box,
                outside=covered is not None and not covered.contains(box),
            )
        )
    return out


# ---------------------------------------------------------------------------
# Closeups
# ---------------------------------------------------------------------------


@dataclass
class Closeup:
    """A sample imaged inside the field of a lower-magnification one.

    ``parent`` is the sample the closeup was taken from, or ``None`` when nothing
    else contains it and the overview mosaic stands in. ``parent_box`` is the area
    to draw as the context picture — the parent's whole field, or a window of the
    overview — and ``box`` is the closeup's own footprint inside it.
    """

    child: Any
    parent: Any | None
    box: Box
    parent_box: Box

    @property
    def on_overview(self) -> bool:
        return self.parent is None


def fractions_within(inner: Box, outer: Box) -> tuple[float, float, float, float]:
    """*inner* as ``(left, top, width, height)`` fractions of *outer*.

    Measured from the top left of the picture, with row 0 at the low-Y edge —
    the same convention :class:`Mosaic` places its tiles with, which is how
    Imaris stores a plane.
    """
    if outer.width <= 0 or outer.height <= 0:
        return (0.0, 0.0, 0.0, 0.0)
    return (
        (inner.x0 - outer.x0) / outer.width,
        (inner.y0 - outer.y0) / outer.height,
        inner.width / outer.width,
        inner.height / outer.height,
    )


def _window(box: Box, bounds: Box, factor: float) -> Box:
    """A square-ish view of *bounds* centred on *box*, *factor* times its size.

    Slid back inside *bounds* rather than clipped when it would overhang, so the
    window keeps its size near an edge; a window larger than the overview is the
    overview.
    """
    span = max(box.width, box.height) * float(factor)
    if span >= bounds.width and span >= bounds.height:
        return bounds
    half_x = min(span, bounds.width) / 2.0
    half_y = min(span, bounds.height) / 2.0
    centre_x, centre_y = box.centre
    centre_x = min(max(centre_x, bounds.x0 + half_x), bounds.x1 - half_x)
    centre_y = min(max(centre_y, bounds.y0 + half_y), bounds.y1 - half_y)
    return Box(centre_x - half_x, centre_x + half_x, centre_y - half_y, centre_y + half_y)


def closeups(samples: Sequence, covered: Box | None = None, window: float = _WINDOW) -> list[Closeup]:
    """Pair each sample with the field it is a closeup of.

    The parent is the *smallest* other sample whose stage footprint contains it,
    so a 40x taken inside a 20x taken inside a 10x is shown against the 20x — the
    tightest context is the one that tells a reader where they are. A sample only
    counts as a parent when the closeup is meaningfully smaller than it, which is
    what stops two acquisitions of the same field pairing with each other.

    When nothing contains a sample and *covered* is given — the area the overview
    fields span — the overview stands in as its context, windowed around the
    sample rather than shown whole. Samples with no stage coordinates, and samples
    outside the overview with no parent, are left out.
    """
    boxed = [(sample, box_from_extent(getattr(sample, "stage_extent", None))) for sample in samples]
    boxed = [(sample, box) for sample, box in boxed if box is not None and box.area > 0]

    out: list[Closeup] = []
    for child, box in boxed:
        slack = _NEST_SLACK * max(box.width, box.height)
        parents = [
            (other, other_box)
            for other, other_box in boxed
            if other is not child
            and other_box.contains(box, slack)
            and max(box.width, box.height) <= _NEST_RATIO * max(other_box.width, other_box.height)
        ]
        if parents:
            parent, parent_box = min(parents, key=lambda pair: pair[1].area)
            out.append(Closeup(child=child, parent=parent, box=box, parent_box=parent_box))
        elif covered is not None and covered.contains(box, slack):
            out.append(Closeup(child=child, parent=None, box=box, parent_box=_window(box, covered, window)))

    logger.info(
        "%d closeup(s) of %d sample(s); %d against the overview",
        len(out), len(boxed), sum(1 for pair in out if pair.on_overview),
    )
    return out
