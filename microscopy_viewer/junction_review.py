"""Go through the QC pictures of a run and decide what happens to each sample.

A junction run (:mod:`microscopy_viewer.junction`) and a count
(:mod:`microscopy_viewer.junction_analysis`) each leave a folder of pictures
sorted into ``ok/``, ``check/`` and ``fail/``. Looking at them is how a wrong
line is found; this module keeps what was decided while looking:

* **accept** — the junction is where it should be;
* **redraw** — the sample is good and the line is not: it joins the queue of
  samples to draw by hand;
* **exclude** — the sample is not to be used at all;
* **redrawn** — it was in the queue and its line has been drawn and saved.

The decisions live in ``review.csv`` in the QC folder, written after every
one, so a review of several hundred samples can stop anywhere and be picked up
— and so the list of what was excluded, and why, travels with the pictures.

No Qt and no napari here; the panel is
:mod:`microscopy_viewer.widgets.junction_review_widget`.
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from . import junction as jn
from .utils import get_logger

logger = get_logger("junction_review")

ACCEPT = "accept"
REDRAW = "redraw"
EXCLUDE = "exclude"
REDRAWN = "redrawn"
UNDECIDED = ""
DECISIONS = (ACCEPT, REDRAW, EXCLUDE, REDRAWN)

#: The file the decisions are kept in, inside the QC folder.
REVIEW_FILE = "review.csv"
COLUMNS = ("Sample", "Status", "Decision", "Note", "Reviewed", "File")

#: QC folders a run writes, relative to the experiment folder, with what each
#: shows. The overview is the one a line is judged on.
SOURCES = (
    ("Junction: whole section, line and band", Path("junction_QC")),
    ("Junction analysis: the counted region", Path("junction_analysis") / "QC"),
)


@dataclass
class ReviewItem:
    """One sample's picture, and what was decided about it."""

    name: str
    picture: Path
    #: The folder the run put it in: ``ok``, ``check`` or ``fail``.
    status: str
    #: The sample's own file, when it was found beside the QC folder.
    path: Path | None = None
    decision: str = UNDECIDED
    note: str = ""
    reviewed: str = ""


def find_samples(folder: str | Path) -> dict[str, Path]:
    """The image files in *folder*, by the name a run gives them."""
    from . import loaders

    found: dict[str, Path] = {}
    directory = Path(folder)
    if not directory.is_dir():
        return found
    for child in sorted(directory.iterdir()):
        if child.is_file() and not child.name.endswith(".h5") and loaders.is_supported(child):
            found.setdefault(jn.fu_name(child), child)
    return found


def load_decisions(qc_folder: str | Path) -> dict[str, dict[str, str]]:
    """What ``review.csv`` holds, by sample name. Empty when there is none."""
    target = Path(qc_folder) / REVIEW_FILE
    if not target.is_file():
        return {}
    try:
        with target.open(newline="", encoding="utf-8-sig") as handle:
            return {row["Sample"]: row for row in csv.DictReader(handle) if row.get("Sample")}
    except Exception:
        logger.exception("could not read %s", target)
        return {}


def scan(qc_folder: str | Path, sample_folder: str | Path | None = None) -> list[ReviewItem]:
    """Every picture in the QC folder's status folders, with its decision.

    Ordered ``fail``, ``check``, ``ok`` — the ones most worth looking at first —
    and by name within each. *sample_folder* is where the samples themselves
    are; a picture whose sample is not there can be reviewed but not opened.
    """
    folder = Path(qc_folder)
    samples = find_samples(sample_folder) if sample_folder is not None else {}
    decisions = load_decisions(folder)
    items: list[ReviewItem] = []
    for status in (jn.STATUS_FAIL, jn.STATUS_CHECK, jn.STATUS_OK):
        for picture in sorted((folder / status).glob("*.png")):
            name = picture.stem
            row = decisions.get(name, {})
            decision = str(row.get("Decision", "")).strip().lower()
            path = samples.get(name)
            if path is None and row.get("File") and Path(row["File"]).is_file():
                path = Path(row["File"])
            items.append(ReviewItem(
                name=name, picture=picture, status=status, path=path,
                decision=decision if decision in DECISIONS else UNDECIDED,
                note=str(row.get("Note", "")), reviewed=str(row.get("Reviewed", "")),
            ))
    return items


def save_decisions(qc_folder: str | Path, items: Sequence[ReviewItem]) -> Path:
    """Write ``review.csv``: one row per sample, decided or not."""
    target = Path(qc_folder) / REVIEW_FILE
    partial = target.with_suffix(".csv.part")
    # utf-8-sig so Excel reads a note with an accent in it as written.
    with partial.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(COLUMNS)
        for item in items:
            writer.writerow([item.name, item.status, item.decision, item.note, item.reviewed,
                             str(item.path) if item.path is not None else ""])
    partial.replace(target)
    return target


def decide(item: ReviewItem, decision: str, note: str | None = None) -> None:
    """Record *decision* on *item*, with when."""
    if decision not in (*DECISIONS, UNDECIDED):
        raise ValueError(f"unknown decision {decision!r}")
    item.decision = decision
    if note is not None:
        item.note = note
    item.reviewed = time.strftime("%Y-%m-%d %H:%M") if decision else ""


def redraw_queue(items: Sequence[ReviewItem]) -> list[ReviewItem]:
    """The samples marked to redraw whose file is known, in review order."""
    return [item for item in items if item.decision == REDRAW and item.path is not None]


#: The decisions whose samples a count takes from a review by default: the line
#: was accepted, or it was drawn again by hand.
COUNT_DECISIONS = (ACCEPT, REDRAWN)


@dataclass
class ReviewedSamples:
    """The samples a review file sends on, and those it leaves behind."""

    #: Sample name -> its file, in the file's order.
    paths: dict[str, Path]
    #: Sample name -> what the review said: ``Status``, ``Decision``, ``Note``.
    rows: dict[str, dict[str, str]]
    #: Kept by their decision, but no file was found for them.
    missing: list[str]
    #: How many were left out, by decision (``""`` is undecided).
    left_out: dict[str, int]


def samples_from_review(
    review_file: str | Path, include: Sequence[str] = COUNT_DECISIONS,
    sample_folder: str | Path | None = None,
) -> ReviewedSamples:
    """The samples of a ``review.csv`` whose decision is in *include*, with their files.

    A sample's file is the one the review recorded; failing that, the sample of
    that name in *sample_folder* — by default the experiment folder the QC
    folder sits in (``junction_QC/`` or ``junction_analysis/QC/``).
    """
    target = Path(review_file)
    if target.is_dir():
        target = target / REVIEW_FILE
    if not target.is_file():
        raise FileNotFoundError(f"no review file at {target}")
    rows = _read_rows(target)
    found: dict[str, Path] = {}
    # The experiment folder holds junction_QC/ and junction_analysis/QC/: one or two levels up.
    for folder in ([Path(sample_folder)] if sample_folder is not None
                   else [target.parent.parent, target.parent.parent.parent]):
        found = find_samples(folder)
        if any(name in found for name in rows):
            break
    wanted = {str(d).strip().lower() for d in include}
    paths: dict[str, Path] = {}
    kept_rows: dict[str, dict[str, str]] = {}
    missing: list[str] = []
    left_out: dict[str, int] = {}
    for name, row in rows.items():
        decision = str(row.get("Decision", "")).strip().lower()
        if decision not in wanted:
            left_out[decision] = left_out.get(decision, 0) + 1
            continue
        recorded = Path(row["File"]) if row.get("File") else None
        path = recorded if recorded is not None and recorded.is_file() else found.get(name)
        if path is None:
            missing.append(name)
            continue
        paths[name] = path
        kept_rows[name] = {k: str(row.get(k, "") or "") for k in ("Status", "Decision", "Note")}
    return ReviewedSamples(paths=paths, rows=kept_rows, missing=missing, left_out=left_out)


def _read_rows(target: Path) -> dict[str, dict[str, str]]:
    """The rows of a review table, by sample name, whatever the file is called."""
    try:
        with target.open(newline="", encoding="utf-8-sig") as handle:
            return {row["Sample"]: row for row in csv.DictReader(handle) if row.get("Sample")}
    except Exception:
        logger.exception("could not read %s", target)
        return {}


def tally(items: Sequence[ReviewItem]) -> dict[str, int]:
    """How many samples carry each decision; ``""`` is the undecided."""
    counts = {decision: 0 for decision in (UNDECIDED, *DECISIONS)}
    for item in items:
        counts[item.decision] = counts.get(item.decision, 0) + 1
    return counts


def summary(items: Sequence[ReviewItem]) -> str:
    """One line: how far the review has got."""
    counts = tally(items)
    return (f"{len(items)} sample(s): {counts[UNDECIDED]} to review, {counts[ACCEPT]} accepted, "
            f"{counts[REDRAW]} to redraw, {counts[REDRAWN]} redrawn, {counts[EXCLUDE]} excluded")
