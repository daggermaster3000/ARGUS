"""Which sections come from the same assembloid.

Cryosections are named ``<slide>_s<scene>``, the scanner numbering the pieces
of tissue on a slide one after another. The file does not say which
assembloid a piece is from (every scene records the same stage position), and
the pattern varies:

- one assembloid per slide, a section per scene;
- two assemblies cut together from one block: every section gives two pieces,
  so the scenes **alternate** between them — until the smaller one runs out
  and the other carries on alone;
- two series one after the other on the slide (**blocks**);
- an assembloid that runs on onto the next slide.

The section's outline tells them apart. Serial sections a few µm apart overlap
almost entirely once centred and turned onto their long axis (the junction
finder's :func:`junction.shape_signature`): about 0.2 apart on
:func:`signature_distance`, against 0.75 between two assemblies cut together
(measured on the 2025-09 slides). So the scenes are taken in order and each
is given to one of two **tracks**, choosing the assignment for which each
track changes least from one member to the next (an exact dynamic programme).
Opening a second track costs :data:`SECOND_TRACK`, so a slide carrying one
assembloid stays one track. A track of fewer than :data:`MIN_TRACK` sections is
not an assembloid: its sections join the other track and are marked as odd
ones out (torn, folded …). Where a track at the end of one slide matches a
track at the start of the next, the assembloid is taken to run on.

Where a section has no signature, its shape figures stand in
(:func:`junction.shape_metrics`, the tissue area, the two halves, the nuclear
stain), scaled by :data:`FEATURE_SCALE`; they separate assemblies far less
well. Both come from the junction run at no extra cost. No Qt; numpy and
pandas only.
"""
from __future__ import annotations

import re
import warnings
from typing import Sequence

import numpy as np

#: ``<slide>_s<scene>``.
SECTION_NAME = re.compile(r"^(?P<slide>.+)_s(?P<scene>\d+)$")

#: The figures compared, and whether they are compared on a log scale (sizes, levels).
FEATURES = (
    ("Tissue area (mm2)", True),
    ("Labelled area (mm2)", True),
    ("Other area (mm2)", True),
    ("Major axis (um)", True),
    ("Minor axis (um)", True),
    ("Solidity", False),
    ("Shape 1", False),
    ("Shape 2", False),
    ("Nuclear stain (mean)", True),
)

#: What opening a second track costs, in units of :func:`signature_distance`
#: (0 the same outline, 1 nothing in common).
SECOND_TRACK = 1.0
#: What every change of track costs (same units). Two assemblies cut together
#: alternate every scene and repay it many times over; one assembloid split
#: into two interleaved tracks would gain only noise, and does not.
SWITCH = 0.1
#: Shape figures (mean robust SDs) to signature units, where a signature is missing.
FEATURE_SCALE = 0.4
#: A track this short is odd sections, not an assembloid.
MIN_TRACK = 3
#: A track at the end of a slide runs on into one at the start of the next
#: when their ends are at most this far apart (same units). Across a slide
#: boundary the same assembloid is 0.2–0.6 apart (sections are lost between
#: slides), two different ones about 0.75.
CONTINUE_DISTANCE = 0.6
#: … and when that pairing beats every other open to either track by this much.
#: A block holds two assemblies, so the question at a slide boundary is mostly
#: which of the two is which.
LINK_MARGIN = 0.1
#: How many scenes at a slide's end (start) a track must reach to run on (from the previous).
END_SCENES = 3


def parse(name: str) -> tuple[str, int | None]:
    """``(slide, scene)`` from a section's name; the whole name and ``None`` if it does not follow the pattern."""
    match = SECTION_NAME.match(str(name))
    if not match:
        return str(name), None
    return match["slide"], int(match["scene"])


def _natural(text: str):
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", text)]


def standardised(frame) -> np.ndarray:
    """The figures compared, one column each, in robust SDs over all the sections (missing → 0)."""
    import pandas as pd

    columns = []
    for name, log in FEATURES:
        values = pd.to_numeric(frame[name], errors="coerce") if name in frame else pd.Series(np.nan, index=frame.index)
        values = values.to_numpy(float)
        if log:
            values = np.log(np.where(values > 0, values, np.nan))
        columns.append(values)
    x = np.column_stack(columns)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)       # a figure missing everywhere
        centre = np.nanmedian(x, axis=0)
        spread = np.nanmedian(np.abs(x - centre), axis=0) * 1.4826
    z = (x - centre) / np.where(np.isfinite(spread) & (spread > 0), spread, np.nan)
    return np.where(np.isfinite(z), z, 0.0)


def decode_signature(text: str, px: int = 64) -> np.ndarray | None:
    """A signature stored as hex (``np.packbits`` of the ``(2, px, px)`` array) back as the array."""
    if not isinstance(text, str) or not text:
        return None
    bits = np.unpackbits(np.frombuffer(bytes.fromhex(text), dtype=np.uint8))[: 2 * px * px]
    return bits.reshape(2, px, px).astype(bool)


def signature_distance(a: np.ndarray, b: np.ndarray) -> float:
    """1 − how much two sections overlap on their signatures, at the best of a half turn and a mirror.

    The overlap is the IoU of the tissue, averaged with that of the labelled
    half where both have one. 0 = the same outline; 1 = nothing in common.
    """
    best = 0.0
    for flipped in (b, b[:, ::-1, :], b[:, :, ::-1], b[:, ::-1, ::-1]):
        scores = []
        for plane in (0, 1):
            union = (a[plane] | flipped[plane]).sum()
            if union and a[plane].any() and flipped[plane].any():
                scores.append((a[plane] & flipped[plane]).sum() / union)
        if scores:
            best = max(best, float(np.mean(scores)))
    return 1.0 - best


def distances(z: np.ndarray) -> np.ndarray:
    """How unlike every two sections are: the mean over figures of their difference."""
    return np.abs(z[:, None, :] - z[None, :, :]).mean(axis=2)


def two_tracks(d: np.ndarray, second: float = SECOND_TRACK, switch: float = SWITCH) -> np.ndarray:
    """Each of the sections (in scene order) given to track 0 or 1, so that each track changes least.

    *d* is their distance matrix. The cost is the sum of the distances between
    consecutive members of each track, plus *second* if track 1 is opened and
    *switch* for every change of track.
    Exact: after section *j*, all that matters is which track *j* is on and
    the last member of the other track.
    """
    n = len(d)
    if n <= 1:
        return np.zeros(n, dtype=int)
    big = np.inf
    # cost[k + 1]: best cost with section j on the current track and the other
    # track's last member k (−1: not opened); same[...]: is j on track 0?
    cost = np.full(n + 1, big)
    cost[0] = 0.0                                 # section 0 on track 0, track 1 not opened
    back: list[np.ndarray] = []
    for j in range(n - 1):
        new = np.full(n + 1, big)
        choice = np.full(n + 1, -2, dtype=int)    # where each new state came from (old k + 1)
        for k1 in np.flatnonzero(np.isfinite(cost)):
            k = k1 - 1
            # j + 1 on the same track as j: the other track's last stays k.
            c = cost[k1] + d[j, j + 1]
            if c < new[k1]:
                new[k1], choice[k1] = c, k1
            # j + 1 on the other track: its last becomes j.
            c = cost[k1] + switch + (d[k, j + 1] if k >= 0 else second)
            if c < new[j + 1]:
                new[j + 1], choice[j + 1] = c, -100 - k1   # marks a switch
        back.append(choice)
        cost = new
    # Walk back: which track each section is on, relative to the one before.
    state = int(np.argmin(cost))
    switch = np.zeros(n, dtype=bool)
    for j in range(n - 2, -1, -1):
        came = back[j][state]
        if came <= -100:
            switch[j + 1] = True
            state = -100 - came
        else:
            state = came
    return (np.cumsum(switch) % 2).astype(int)


def section_distances(rows: np.ndarray, z: np.ndarray, signatures) -> np.ndarray:
    """How unlike the sections *rows* are, pairwise: on their signatures where both have one."""
    d = distances(z[rows]) * FEATURE_SCALE
    if signatures is None:
        return d
    for i, a in enumerate(rows):
        for j in range(i + 1, len(rows)):
            b = rows[j]
            if signatures[a] is not None and signatures[b] is not None:
                d[i, j] = d[j, i] = signature_distance(signatures[a], signatures[b])
    return d


def group_sections(frame, name_column: str = "name", series: Sequence[str] | None = None,
                   second_track: float = SECOND_TRACK, continue_distance: float = CONTINUE_DISTANCE,
                   switch: float = SWITCH, signature_column: str = "Signature"):
    """*frame* with the assembloid each section belongs to.

    Adds ``Slide``, ``Scene``, ``Assembloid`` (``<slide>/A`` or ``/B``; one that
    runs on from the previous slide keeps that slide's name), ``Step`` (how
    unlike the section is to the one before it on its track; NaN for a
    track's first), ``Odd section`` (put in a track of its own, too short to be
    an assembloid, and so given to the other), ``Pattern`` (the slide's:
    ``one``, ``alternating``, ``blocks`` or ``mixed``), ``Continues from`` and
    ``Link distance`` (how far apart the two ends were where it was joined on;
    0.2 is as close as neighbouring sections, 0.6 the most that is joined).

    *series* names columns that must match for one slide to run on onto the
    next (e.g. genotype and experiment); slides are taken in name order.
    *signature_column* holds each section's :func:`junction.shape_signature`
    (an array, its hex form, or nothing).
    """
    out = frame.copy()
    signatures = None
    if signature_column in out:
        signatures = [s if isinstance(s, np.ndarray) else decode_signature(s) for s in out[signature_column]]
    parsed = [parse(n) for n in out[name_column]]
    out["Slide"] = [p[0] for p in parsed]
    out["Scene"] = [p[1] if p[1] is not None else -1 for p in parsed]
    z = standardised(out)
    slides = out["Slide"].to_numpy()
    scenes = out["Scene"].to_numpy()
    assembloid = np.empty(len(out), dtype=object)
    step = np.full(len(out), np.nan)
    odd = np.zeros(len(out), dtype=bool)
    pattern = np.empty(len(out), dtype=object)
    continues = np.full(len(out), "", dtype=object)
    link = np.full(len(out), np.nan)
    tracks_of: dict[str, list[np.ndarray]] = {}     # slide → its tracks, each the rows in scene order

    for slide in sorted(set(slides), key=_natural):
        rows = np.flatnonzero(slides == slide)
        order = rows[np.argsort(scenes[rows], kind="stable")]
        d = section_distances(order, z, signatures)
        track = two_tracks(d, second_track, switch)
        # A track too short to be an assembloid: odd sections of the other one.
        for t in (0, 1):
            if 0 < (track == t).sum() < MIN_TRACK and (track != t).sum() >= MIN_TRACK:
                odd[order[track == t]] = True
                track[track == t] = 1 - t
        # Name the tracks by first appearance.
        first = track[0]
        track = np.where(track == first, 0, 1)
        members = [order[track == t] for t in (0, 1) if (track == t).any()]
        tracks_of[slide] = members
        for t, rows_t in enumerate(members):
            for k, row in enumerate(rows_t):
                assembloid[row] = f"{slide}/{'AB'[t]}"
                kept = [r for r in rows_t[:k] if not odd[r]]
                if kept and not odd[row]:
                    i, j = list(order).index(kept[-1]), list(order).index(row)
                    step[row] = d[i, j]
        switches = int(np.abs(np.diff(track)).sum())
        kind = ("one" if len(members) == 1 else "blocks" if switches == 1
                else "alternating" if switches >= len(order) - 2 else "mixed")
        pattern[order] = kind

    # Runs on onto the next slide: a track reaching the end of one slide, and
    # one starting at the start of the next, whose ends match.
    names = sorted(set(slides), key=_natural)
    for before, after in zip(names[:-1], names[1:]):
        if series is not None:
            a = out.iloc[tracks_of[before][0][0]][list(series)].astype(str).to_numpy()
            b = out.iloc[tracks_of[after][0][0]][list(series)].astype(str).to_numpy()
            if not (a == b).all():
                continue
        last_scene = scenes[np.concatenate(tracks_of[before])].max()
        first_scene = scenes[np.concatenate(tracks_of[after])].min()
        ending = [t for t in tracks_of[before] if (scenes[t] > last_scene - END_SCENES * 2).any()]
        starting = [t for t in tracks_of[after] if (scenes[t] < first_scene + END_SCENES * 2).any()]
        pairs = []
        for i, tail in enumerate(ending):
            for j, head in enumerate(starting):
                tail_rows = [r for r in tail if not odd[r]][-2:]
                head_rows = [r for r in head if not odd[r]][:2]
                if tail_rows and head_rows:
                    between = section_distances(np.array(tail_rows + head_rows), z, signatures)
                    pairs.append((float(np.median(between[: len(tail_rows), len(tail_rows):])), i, j))
        table = {(i, j): dist for dist, i, j in pairs}
        for (i, j), dist in sorted(table.items(), key=lambda item: item[1]):
            rivals = [v for (a, b), v in table.items() if (a == i) != (b == j)]
            if dist > continue_distance or (rivals and dist + LINK_MARGIN > min(rivals)):
                continue
            old, new = assembloid[starting[j][0]], assembloid[ending[i][-1]]
            for row in np.flatnonzero(assembloid == old):
                assembloid[row] = new
                continues[row] = new
                link[row] = dist
    out["Assembloid"] = assembloid
    out["Step"] = step
    out["Odd section"] = odd
    out["Pattern"] = pattern
    out["Continues from"] = continues
    out["Link distance"] = link
    return out
