"""How well the cycles of a 4i plate line up, per object and per field.

A 4i experiment stains the same cells over and over, and every cycle is aligned
back onto a reference before anything is measured. The alignment is never
perfect everywhere: an organoid lifts at one edge, a corner of the well dries,
the stage repositions a micron out. Where it failed, the objects found in the
reference cycle sit over different cells in the later ones, and every number
measured there is a number about the wrong cell — with nothing in the table to
say so.

Two measurements of that, both per image and per cycle:

* **A correlation, per object.** The nuclear stain is imaged in every cycle, so
  inside one object it should look like itself. Where it does not, that object
  moved. This is the metric `abbott-features`' ``cycle_registration_quality``
  task uses (`pelkmanslab/abbott-features`), reproduced here: the pixels inside
  one label, in two cycles, through Pearson — see :func:`object_correlation`.
* **A shift, per field.** Phase cross-correlation of the whole nuclear channel
  against the reference, in µm. One number per cycle that says how far the
  field as a whole moved, which is the thing a correlation only implies.

The per-object number is the one that answers "keep the regions that are right":
it lands in the object table as an ordinary column, so it can be gated on in the
Population analysis panel like any other measurement. Nothing here decides what
to keep — a threshold that belongs to the data is the user's to set, against the
distribution, where they can see it.

No Qt, no napari: the panel and the batch runner both call this.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from .utils import get_logger

logger = get_logger("alignment")

#: The per-object column, before the cycle is put on it.
R_COLUMN = "Registration r"
#: The per-field columns of the run summary.
SHIFT_COLUMN = "Registration shift (um)"

#: Fewer pixels than this inside an object and a correlation is noise.
#:
#: Pearson's r over four pixels is almost always near 1 or near -1 and means
#: neither; the number would be a confident answer to a question the data cannot
#: settle. Such objects get a blank, which the gates already treat as "no
#: opinion" rather than as a failure.
MIN_PIXELS = 8

#: Upsampling for the field shift, so it is read to a fraction of a pixel.
#:
#: A 4i misalignment worth finding is often under a pixel at the magnifications
#: this is used at, and an integer-pixel answer would report it as zero.
SHIFT_UPSAMPLE = 10


def r_column(cycle: int | None) -> str:
    """The per-object correlation column for one cycle.

    Named per cycle because there is one answer per cycle: a field can be
    perfect against cycle 2 and gone by cycle 7, and a single column would hide
    which.
    """
    return R_COLUMN if cycle is None else f"{R_COLUMN}, cycle {cycle}"


def worst_column(cycles: Sequence[int | None] = ()) -> str:
    """The column holding each object's worst correlation over every cycle.

    The one to gate on. An object that matches in six cycles and not in the
    seventh is an object that moved, and the measurement from the seventh is as
    wrong as if it had moved in all of them. Gating on the worst is one decision
    instead of seven, and it is the decision people mean.
    """
    return f"{R_COLUMN}, worst of {len(cycles)} cycle(s)" if cycles else f"{R_COLUMN}, worst"


@dataclass(frozen=True)
class CycleAlignment:
    """How one cycle lines up with the reference, for one image."""

    cycle: int | None
    #: Per-object Pearson r, keyed by label. Missing where there was too little
    #: to judge; see :data:`MIN_PIXELS`.
    r: Mapping[int, float] = field(default_factory=dict)
    #: Whole-field displacement in µm, ``(z, y, x)``.
    shift_um: tuple[float, ...] = ()
    #: Pixels the field moved, same order, before the voxel size is applied.
    shift_px: tuple[float, ...] = ()
    #: Objects that could not be judged.
    n_blank: int = 0
    note: str = ""

    @property
    def distance_um(self) -> float:
        """How far the field moved, as one number."""
        if not self.shift_um:
            return float("nan")
        return float(np.sqrt(np.sum(np.square(np.asarray(self.shift_um, dtype=float)))))

    @property
    def median_r(self) -> float:
        values = np.asarray([value for value in self.r.values()], dtype=float)
        finite = values[np.isfinite(values)]
        return float(np.median(finite)) if finite.size else float("nan")

    def describe(self) -> str:
        parts = [f"cycle {self.cycle}"]
        if self.shift_um:
            moved = ", ".join(f"{value:+.2f}" for value in self.shift_um)
            parts.append(f"moved ({moved}) µm = {self.distance_um:.2f} µm")
        if self.r:
            parts.append(f"median r {self.median_r:.3f} over {len(self.r):,} object(s)")
        if self.n_blank:
            parts.append(f"{self.n_blank:,} too small to judge")
        if self.note:
            parts.append(self.note)
        return "; ".join(parts)


def object_correlation(
    masks: np.ndarray,
    reference: np.ndarray,
    moving: np.ndarray,
    min_pixels: int = MIN_PIXELS,
) -> tuple[dict[int, float], int]:
    """Pearson r inside each object, between two cycles of the same stain.

    Returns ``({label: r}, n_blank)``.

    This is `abbott-features`' registration-quality metric. Its
    ``cycle_registration_quality`` task takes the same wavelength in the
    reference image and in the moving one, gathers the pixels inside one ROI
    from both, and correlates them; the correlation over a region that is
    properly aligned is high, because the nuclear stain there is the same
    picture twice.

    Done per **object** rather than per tile because that is the unit the rest
    of this application answers in. A tile grid would put an object that
    straddles a boundary in two tiles with two different answers, and the table
    has one row for it.

    An object with fewer than *min_pixels* pixels gets no number rather than a
    confident one: r over a handful of pixels is near ±1 whatever the data says.
    The blank is what the gates already read as "this gate has no opinion".
    """
    from skimage.measure import regionprops_table

    masks = np.asarray(masks)
    reference = np.asarray(reference, dtype=float)
    moving = np.asarray(moving, dtype=float)
    if reference.shape != masks.shape or moving.shape != masks.shape:
        raise ValueError(
            f"the objects are {tuple(masks.shape)}, the reference is "
            f"{tuple(reference.shape)} and the other cycle is {tuple(moving.shape)}; "
            "a correlation only means anything on one grid"
        )

    # label + slice, so each object is read out of its own bounding box rather
    # than out of the whole plane. On a 12000 x 12000 image with ten thousand
    # objects, the difference is minutes.
    props = regionprops_table(masks, properties=("label", "slice"))
    scores: dict[int, float] = {}
    blank = 0
    for label, window in zip(props["label"], props["slice"]):
        inside = masks[window] == label
        left = reference[window][inside]
        right = moving[window][inside]
        if left.size < int(min_pixels):
            blank += 1
            continue
        value = _pearson(left, right)
        if np.isfinite(value):
            scores[int(label)] = float(value)
        else:
            blank += 1
    return scores, blank


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    """Pearson's r, or NaN where it is undefined.

    Undefined is the normal case rather than the exceptional one here: a nucleus
    whose stain saturates flat in one cycle has no variance, and r is 0/0. numpy
    would warn and return NaN; scipy raises on some inputs and warns on others.
    Deciding it here keeps one answer.
    """
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    good = np.isfinite(left) & np.isfinite(right)
    if good.sum() < 2:
        return float("nan")
    left, right = left[good], right[good]
    left = left - left.mean()
    right = right - right.mean()
    denominator = float(np.sqrt(np.dot(left, left) * np.dot(right, right)))
    if denominator == 0.0:
        return float("nan")
    return float(np.dot(left, right) / denominator)


def field_shift(
    reference: np.ndarray,
    moving: np.ndarray,
    voxel_size_um: Sequence[float] = (1.0, 1.0, 1.0),
    upsample: int = SHIFT_UPSAMPLE,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """``(shift_px, shift_um)`` of *moving* against *reference*, by phase correlation.

    The number a microscopist asks for first: how far did this field move. A
    correlation says a region disagrees without saying by how much or which way,
    and "0.8" is not something anyone has an intuition for; "1.4 µm" is.

    Upsampled, because a 4i misalignment worth finding is often well under a
    pixel and an integer answer would report it as none at all.
    """
    from skimage.registration import phase_cross_correlation

    reference = np.asarray(reference, dtype=float)
    moving = np.asarray(moving, dtype=float)
    if reference.shape != moving.shape:
        raise ValueError(
            f"the reference is {tuple(reference.shape)} and the other cycle is "
            f"{tuple(moving.shape)}; a shift between two different grids is not a shift"
        )

    outcome = phase_cross_correlation(
        reference, moving, upsample_factor=max(1, int(upsample))
    )
    # Across versions this has returned the shift alone and a 3-tuple of
    # (shift, error, phasediff). Take the first element when it is packed.
    shift = outcome[0] if isinstance(outcome, tuple) else outcome
    shift_px = tuple(float(value) for value in np.asarray(shift, dtype=float).ravel())

    sizes = [float(value) for value in voxel_size_um][-len(shift_px) :]
    while len(sizes) < len(shift_px):
        sizes.insert(0, 1.0)
    shift_um = tuple(abs(size) * value for size, value in zip(sizes, shift_px))
    return shift_px, shift_um


def align_cycle(
    masks: np.ndarray,
    reference: np.ndarray,
    moving: np.ndarray,
    cycle: int | None,
    voxel_size_um: Sequence[float] = (1.0, 1.0, 1.0),
    min_pixels: int = MIN_PIXELS,
    with_shift: bool = True,
) -> CycleAlignment:
    """Both measurements, for one cycle against the reference."""
    scores, blank = object_correlation(masks, reference, moving, min_pixels=min_pixels)
    shift_px: tuple[float, ...] = ()
    shift_um: tuple[float, ...] = ()
    note = ""
    if with_shift:
        try:
            shift_px, shift_um = field_shift(reference, moving, voxel_size_um)
        except Exception as error:  # noqa: BLE001 - reported, never fatal
            # A field shift is a diagnostic. Losing it is worth a sentence, not
            # the loss of the per-object answers computed beside it.
            note = f"the field shift could not be read: {error}"
            logger.debug("field shift failed for cycle %s", cycle, exc_info=True)
    outcome = CycleAlignment(
        cycle=cycle,
        r=scores,
        shift_um=shift_um,
        shift_px=shift_px,
        n_blank=blank,
        note=note,
    )
    logger.info("registration: %s", outcome.describe())
    return outcome


def worst_of(alignments: Sequence[CycleAlignment], labels: Sequence[int]) -> dict[int, float]:
    """Each object's lowest correlation across every cycle.

    An object judged in no cycle is absent rather than present with a NaN, so it
    reaches the table as a blank and the gates abstain on it.
    """
    worst: dict[int, float] = {}
    for label in labels:
        scores = [
            alignment.r[int(label)]
            for alignment in alignments
            if int(label) in alignment.r
        ]
        if scores:
            worst[int(label)] = float(min(scores))
    return worst
