"""Flatten a folder of stacks to maximum projections, without opening any of them.

A confocal folder is mostly Z. Most of what happens to it afterwards — figures,
counting, sending a collaborator something they can actually open — happens on
the projection, and making thirty of those one at a time through the viewer is an
afternoon.

Nothing here touches the viewer, and nothing loads a whole stack. The projection
is computed **through the lazy array the reader returns**, so dask reduces it
chunk by chunk and only the finished plane is ever in memory: a 271 × 2040 × 2040
channel is 2.2 GB materialised and about 8 MB projected, and the difference is
whether a thirty-file batch fits in RAM at all.

Channels are chosen by name, because the channel order is not the same in every
acquisition and an index quietly means a different stain from one file to the
next. A name matching nothing in a given file is reported against that file
rather than silently skipped.

Output goes through :mod:`microscopy_viewer.writers`: TIFF for everything else to
read, Imaris to keep the derived image in the same format as its source.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from . import writers
from .utils import get_logger

logger = get_logger("projection")

#: Appended to the source stem so a projection never lands on top of its source.
DEFAULT_SUFFIX = "_MIP"

#: What the panel offers to write. Ordered: TIFF first, since it is the one
#: anything else can open.
OUTPUT_FORMATS = (".tif", ".ims")


@dataclass
class ProjectionOptions:
    """Where the projections go and which channels go into them."""

    #: Folder to write into. ``None`` writes beside each source file.
    output_dir: Path | None = None
    #: Channel names (matched as case-insensitive substrings) or integer indices.
    #: Empty means every channel, which is the usual thing to want.
    channels: tuple[str | int, ...] = ()
    fmt: str = ".tif"
    suffix: str = DEFAULT_SUFFIX
    #: Overwrite an output that is already there. Off, so a re-run after adding
    #: files to a folder costs nothing and cannot destroy an edited projection.
    overwrite: bool = False


@dataclass
class ProjectionOutcome:
    """What happened to one file."""

    path: Path
    name: str
    written: Path | None = None
    channels: list[str] = field(default_factory=list)
    planes: int = 0
    elapsed_s: float = 0.0
    skipped: bool = False
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error


# ---------------------------------------------------------------------------
# Choosing channels
# ---------------------------------------------------------------------------


def select_channels(names: Sequence[str], wanted: Sequence[str | int]) -> tuple[list[int], list[str]]:
    """``(indices, unmatched)`` for the channels *wanted* out of *names*.

    Empty *wanted* selects every channel. Indices are returned in the file's own
    channel order rather than in the order they were asked for, so a projection's
    channel 0 is the file's channel 0 and the colours stay where they belong.

    What is *not* matched comes back too. Over thirty files, a name that silently
    matches nothing is a stack of projections missing a channel nobody notices
    until the figure is made.
    """
    if not wanted:
        return list(range(len(names))), []

    chosen: set[int] = set()
    unmatched: list[str] = []
    for entry in wanted:
        if isinstance(entry, (int, np.integer)) and not isinstance(entry, bool):
            index = int(entry)
            if 0 <= index < len(names):
                chosen.add(index)
            else:
                unmatched.append(str(entry))
            continue
        needle = str(entry).strip().lower()
        if not needle:
            continue
        hits = [index for index, name in enumerate(names) if needle in str(name).lower()]
        if hits:
            chosen.update(hits)
        else:
            unmatched.append(str(entry))
    return sorted(chosen), unmatched


# ---------------------------------------------------------------------------
# Projecting
# ---------------------------------------------------------------------------


def z_axis_of(axes: str, ndim: int) -> int | None:
    """Which axis of an array is Z, from the reader's axis string.

    Positional guessing is the fallback, not the rule: a 3D array is ``ZYX`` and
    a 4D one is ``TZYX`` for every reader here, but a file that says so is
    believed over a file that merely looks usual.
    """
    letters = str(axes or "").upper()
    if len(letters) == ndim:
        # The file said what its axes are, so it is believed in both directions:
        # a stack that says TYX has no Z, and guessing one would project a
        # time-lapse over time and hand back a single frame.
        return letters.index("Z") if "Z" in letters else None
    if ndim == 3:
        return 0
    if ndim == 4:
        return 1
    return None


def project_array(array, axes: str = "") -> np.ndarray:
    """Maximum-project one channel over Z, returning ``(T, Y, X)``.

    *array* may be a dask array, and is left as one until the very end: the
    reduction runs chunk by chunk and only the result is materialised.
    """
    ndim = int(getattr(array, "ndim", np.asarray(array).ndim))
    axis = z_axis_of(axes, ndim)
    if ndim <= 2:
        flat = array
    elif axis is None:
        flat = array
    else:
        flat = array.max(axis=axis)
    result = np.asarray(flat)
    while result.ndim < 3:
        result = result[None, ...]
    return result


def output_path(source: Path, options: ProjectionOptions) -> Path:
    """Where one source file's projection is written."""
    folder = Path(options.output_dir) if options.output_dir else Path(source).parent
    suffix = options.fmt if options.fmt.startswith(".") else f".{options.fmt}"
    return folder / f"{Path(source).stem}{options.suffix}{suffix}"


def project_file(path: str | Path, options: ProjectionOptions) -> ProjectionOutcome:
    """Project one file and write it. Never raises; failures land on the outcome."""
    from .loaders import load_path, release

    source = Path(path)
    outcome = ProjectionOutcome(path=source, name=source.stem)
    started = time.perf_counter()
    try:
        target = output_path(source, options)
        if target.resolve() == source.resolve():
            raise ValueError(
                "the projection would overwrite its own source; give it a suffix "
                "or a different output folder"
            )
        if target.exists() and not options.overwrite:
            outcome.written = target
            outcome.skipped = True
            return outcome

        specs = load_path(source)
        if not specs:
            raise ValueError("no readable channels")
        names = [spec.channel_name or spec.name for spec in specs]
        indices, unmatched = select_channels(names, options.channels)
        if not indices:
            raise ValueError(
                f"no channel matching {', '.join(str(c) for c in options.channels)} "
                f"(it has: {', '.join(names) or 'none'})"
            )

        stacks = []
        for index in indices:
            spec = specs[index]
            data = spec.data[0] if spec.multiscale else spec.data
            stacks.append(project_array(data, getattr(spec, "axes", "")))

        timepoints = max(int(stack.shape[0]) for stack in stacks)
        planes = np.stack(
            [
                np.stack([_broadcast_time(stack, t) for stack in stacks])
                for t in range(timepoints)
            ]
        )

        meta = specs[indices[0]].metadata
        voxel = tuple(float(v) for v in specs[indices[0]].scale)[-3:]
        if len(voxel) < 3:
            voxel = (1.0,) * (3 - len(voxel)) + voxel
        outcome.written = writers.write_stack(
            target,
            planes,
            voxel_um=voxel,
            channel_names=[names[index] for index in indices],
            channel_colors=[getattr(specs[index], "color", None) for index in indices],
            stage_extent=getattr(meta, "stage_extent", None),
            source_name=source.stem,
            time_interval_s=float(getattr(meta, "time_interval_s", 0.0) or 0.0),
        )
        outcome.channels = [names[index] for index in indices]
        outcome.planes = int(timepoints)
        if unmatched:
            outcome.error = f"no channel matching {', '.join(unmatched)}"
    except Exception as exc:
        logger.exception("could not project %s", source.name)
        outcome.error = str(exc)
    finally:
        outcome.elapsed_s = time.perf_counter() - started
        try:
            release(source)
        except Exception:
            logger.debug("could not release %s", source, exc_info=True)
    return outcome


def _broadcast_time(stack: np.ndarray, index: int) -> np.ndarray:
    """One timepoint of a projected channel, repeating a still channel if needed.

    A file can hold a time series alongside a channel imaged once. Repeating the
    still one keeps the output rectangular rather than refusing the file.
    """
    return stack[index] if index < stack.shape[0] else stack[-1]


def run_batch(
    paths: Sequence[str | Path],
    options: ProjectionOptions,
    progress: Callable[[str], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> list[ProjectionOutcome]:
    """Project every file. One outcome each, and one bad file does not stop it."""
    outcomes: list[ProjectionOutcome] = []
    for index, entry in enumerate(paths, start=1):
        if should_cancel is not None and should_cancel():
            break
        source = Path(entry)
        if progress is not None:
            progress(f"{source.stem} ({index} of {len(paths)})")
        outcome = project_file(source, options)
        outcomes.append(outcome)
        if progress is not None:
            if outcome.skipped:
                note = "already there"
            elif outcome.error:
                note = outcome.error
            else:
                note = f"{len(outcome.channels)} channel(s) → {outcome.written.name}"
            progress(f"{source.stem}: {note}")
    return outcomes


def summarise(outcomes: Sequence[ProjectionOutcome]) -> str:
    """The one line the panel shows when a batch finishes."""
    written = [outcome for outcome in outcomes if outcome.ok and not outcome.skipped]
    skipped = [outcome for outcome in outcomes if outcome.skipped]
    failed = [outcome for outcome in outcomes if outcome.error]
    seconds = sum(outcome.elapsed_s for outcome in outcomes)

    parts = [f"Projected {len(written)} file(s) in {seconds:.0f} s."]
    if skipped:
        parts.append(f"{len(skipped)} already existed — tick Overwrite to redo them.")
    if failed:
        names = ", ".join(f"{outcome.name} ({outcome.error})" for outcome in failed[:2])
        parts.append(f"{len(failed)} had problems: {names}")
    return " ".join(parts)


def channel_names(paths: Sequence[str | Path], limit: int = 12) -> list[str]:
    """Every channel name seen across *paths*, in first-seen order.

    What the panel offers as tick boxes. Read from the files rather than typed,
    because the whole point of matching by name is that the names are the thing
    that stays the same between acquisitions.
    """
    from .loaders import load_path, release

    seen: list[str] = []
    for entry in list(paths)[:limit]:
        source = Path(entry)
        try:
            for spec in load_path(source):
                name = spec.channel_name or spec.name
                if name and name not in seen:
                    seen.append(name)
        except Exception:
            logger.debug("could not read channel names from %s", source, exc_info=True)
        finally:
            try:
                release(source)
            except Exception:
                pass
    return seen
