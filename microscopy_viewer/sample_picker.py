"""Pick samples off a table of categories: keep some, draw a few of each.

A prescreen or QC run leaves a table with a row per sample and columns of
categories — *Pass* yes/no, *Ripped*, *Folded*, *Shape* … The picker turns it
into a selection in two steps:

1. **Keep** — tick the category values a sample may have. Unticking *Pass: no*
   leaves out every failing sample. Within a column the ticked values are any
   of; across columns, all of.
2. **Draw** — tick the groups to annotate, say *Ripped: yes* and each *Shape*,
   and *n*. Each ticked group gets its own *n* samples (fewer when it has fewer
   kept samples), none picked twice. Optionally (*share*) a sample picked for
   one group counts for every group it belongs to, so overlapping groups are
   covered with fewer samples. With nothing ticked to draw, every kept sample
   is picked.

The draw is random but seeded: the same table, ticks and seed give the same
picks, and the saved list records the seed.

Rows are matched to the files of the experiment folder by the column holding
file names (``file``), else by any column whose values are file names or
stems. Rows matching no file are left out of the draw — they could not be
opened — and counted.

Pure pandas; the window is :mod:`microscopy_viewer.widgets.sample_picker_window`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

#: Column suffixes that describe another column rather than categorise samples.
SKIP_SUFFIXES = (" source", " confidence", " score", " probability")
#: Columns never offered as categories.
SKIP_NAMES = {"file", "path", "object_id", "notes", "note", "name", "sample", "scene"}
#: A column with more distinct values than this is not a category.
MAX_VALUES = 24
#: Column holding the path the picks are matched on, in the saved list.
PATH_COLUMN = "path"
#: Column of the saved list naming the groups a pick was drawn for.
GROUPS_COLUMN = "picked for"

MISSING = "(empty)"


def read_table(path) -> pd.DataFrame:
    """A CSV or Excel table, every value as text (``(empty)`` for blanks)."""
    path = Path(path)
    if path.suffix.lower() in (".xlsx", ".xlsm", ".xls"):
        table = pd.read_excel(path)
    else:
        table = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
    table.columns = [str(c).strip() for c in table.columns]
    return table


def _text(value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return MISSING
    text = str(value).strip()
    return text or MISSING


def category_columns(table: pd.DataFrame) -> list[str]:
    """Columns that sort samples into a handful of groups, in table order."""
    out = []
    for column in table.columns:
        lower = column.lower()
        if lower in SKIP_NAMES or lower.endswith(SKIP_SUFFIXES):
            continue
        values = table[column].map(_text)
        distinct = values.nunique()
        if 2 <= distinct <= MAX_VALUES and distinct < len(table):
            out.append(column)
    return out


def values_of(table: pd.DataFrame, column: str) -> list[tuple[str, int]]:
    """``(value, rows)`` of *column*, most common first."""
    counts = table[column].map(_text).value_counts()
    return [(str(value), int(n)) for value, n in counts.items()]


def _key(name: str) -> str:
    """A file name without folder and without its image extensions."""
    name = Path(str(name)).name.lower()
    for suffix in (".argus.h5", ".ome.tiff", ".ome.tif", ".tiff", ".tif", ".ims", ".czi", ".nd2", ".lif"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def match_paths(table: pd.DataFrame, paths: Iterable) -> pd.Series:
    """The file of each row among *paths* (``None`` when there is none).

    Tries the column called ``file`` / ``path`` first, then every column, and
    keeps the one that matches the most rows. A full path in the ``file`` /
    ``path`` column that exists counts even when it is not among *paths*.
    """
    by_key: dict[str, Path] = {}
    for path in paths:
        by_key.setdefault(_key(Path(path).name), Path(path))
    preferred = [c for c in table.columns if c.lower() in ("file", "path", "filename", "file name")]
    # Full paths in the table are used as they are, so it works without a folder.
    for column in preferred:
        for value in table[column]:
            if isinstance(value, str) and Path(value).is_absolute() and Path(value).is_file():
                by_key.setdefault(_key(Path(value).name), Path(value))
    best, best_hits = None, 0
    for column in preferred + [c for c in table.columns if c not in preferred]:
        keys = table[column].map(lambda v: _key(v) if isinstance(v, str) else "")
        hits = int(keys.isin(by_key.keys()).sum())
        if hits > best_hits:
            best, best_hits = keys, hits
        if column in preferred and hits == len(table):
            break
    if best is None:
        return pd.Series([None] * len(table), index=table.index, dtype=object)
    return best.map(lambda k: by_key.get(k)).astype(object)


def keep_mask(table: pd.DataFrame, keep: Mapping[str, Iterable[str]]) -> pd.Series:
    """Rows whose value in every column of *keep* is one of the ticked values."""
    mask = pd.Series(True, index=table.index)
    for column, values in keep.items():
        mask &= table[column].map(_text).isin(set(values))
    return mask


@dataclass
class Picks:
    """What a draw picked, and why."""

    rows: list = field(default_factory=list)
    #: Row index → the groups (``"Column: value"``) it was drawn for or counts for.
    groups: dict = field(default_factory=dict)
    #: ``"Column: value"`` → (picked, kept), for each group drawn.
    filled: dict = field(default_factory=dict)
    seed: int = 0


def group_name(column: str, value: str) -> str:
    return f"{column}: {value}"


def draw(
    table: pd.DataFrame,
    eligible: pd.Series,
    groups: Sequence[tuple[str, str]],
    n: int,
    seed: int = 0,
    share: bool = False,
) -> Picks:
    """*n* picks for every ``(column, value)`` of *groups*, among *eligible* rows.

    Each group draws its own *n* samples, none picked twice, so ten per group
    over four groups is forty samples. With *share*, a sample drawn for one
    group also counts for every other group it is in, and a group only draws
    what it still lacks: fewer samples, at least *n* per group.

    With *n* 0 or less, a group takes **all** its eligible samples rather than
    a draw — every passing sample of a genotype, say — in table order.

    With no groups, every eligible row is picked, in table order.
    """
    pool = table.index[eligible.to_numpy(dtype=bool)]
    picks = Picks(seed=int(seed))
    if not len(groups):
        picks.rows = list(pool)
        return picks
    rng = np.random.default_rng(int(seed))
    chosen: list = []
    taken: set = set()
    members_of: dict[str, set] = {}
    drawn_for: dict[str, list] = {}
    for column, value in groups:
        name = group_name(column, value)
        members = [i for i in pool if _text(table.at[i, column]) == value]
        members_of[name] = set(members)
        have = [i for i in chosen if i in members_of[name]] if share else []
        rest = [i for i in members if i not in taken]
        need = len(rest) if int(n) <= 0 else max(0, int(n) - len(have))
        drawn_for[name] = []
        if need and rest:
            order = np.arange(len(rest)) if int(n) <= 0 else rng.permutation(len(rest))[:need]
            for k in sorted(order):
                chosen.append(rest[k])
                taken.add(rest[k])
                drawn_for[name].append(rest[k])
    for name, members in members_of.items():
        inside = [i for i in chosen if i in members] if share else drawn_for[name]
        # How many it got, and how many it could have got: its kept samples,
        # less those already drawn for an earlier group when nothing is shared.
        picks.filled[name] = (len(inside), len(members))
        for i in inside:
            picks.groups.setdefault(i, []).append(name)
    picks.rows = chosen
    return picks


def picks_table(table: pd.DataFrame, picks: Picks, paths: pd.Series) -> pd.DataFrame:
    """The picked rows as a table to save: path, groups, the table's columns, the seed."""
    out = table.loc[picks.rows].copy()
    out.insert(0, GROUPS_COLUMN, ["; ".join(picks.groups.get(i, [])) for i in picks.rows])
    out.insert(0, PATH_COLUMN, [str(paths.get(i)) if paths.get(i) is not None else "" for i in picks.rows])
    out["seed"] = picks.seed
    return out.reset_index(drop=True)
