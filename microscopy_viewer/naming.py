"""Read the things an experiment encodes in its file names.

A confocal folder is a flat list of ``.ims`` files, and the only place the
genotype of each fish is written down is the file name — ``fish03_mut_20x.ims``,
``TUNEL_ift88 mut_Confocal - Blue_2025-04-08.ims``. Every analysis afterwards is
a comparison between genotypes, so the alternative to reading them here is
typing a genotype column into thirty rows of a spreadsheet by hand, which is
where transcription errors come from.

Two ways of reading it, in order:

1. **A pattern**, when one is given. Anything with a capturing group works, so a
   naming scheme this module has never seen can still be read without changing
   any code.
2. **A vocabulary**, otherwise. The name is split on underscores — the delimiter
   in every scheme seen here — and each field is searched for a known genotype
   word. Spelling variants collapse onto one canonical form, so ``MUT``,
   ``mut`` and ``mutant`` all group together in the workbook instead of making
   three columns in a bar chart.

**A name that says nothing gives an empty string, never a guess.** A wrong
genotype silently attached to a sample is worse than a blank one: the blank is
visible in the workbook and gets fixed, the wrong one becomes a result.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Sequence

from .utils import get_logger

logger = get_logger("naming")

#: Canonical genotype → the spellings that mean it, longest first inside each
#: entry so ``wildtype`` is not matched as ``wild``.
#:
#: Deliberately conservative. ``hom`` is not folded into ``mut`` and ``sib`` is
#: not folded into ``wt``, even though a given project usually means the same
#: thing by them, because which one it means is the project's business and
#: guessing it here would silently merge two groups of an experiment.
GENOTYPE_WORDS: dict[str, tuple[str, ...]] = {
    "wt": ("wildtype", "wild-type", "wild type", "wt"),
    "mut": ("mutant", "mut"),
    "het": ("heterozygous", "heterozygote", "het"),
    "hom": ("homozygous", "homozygote", "homo", "hom"),
    "ko": ("knockout", "knock-out", "ko"),
    "ki": ("knockin", "knock-in", "ki"),
    "tg": ("transgenic", "tg"),
    "ctl": ("control", "ctrl", "ctl"),
    "sib": ("sibling", "sib"),
    "mo": ("morpholino", "morphant", "mo"),
}

#: Fields split on this. Underscore is the delimiter in the naming schemes here;
#: the pattern is kept separate so a scheme that uses something else only has to
#: change in one place.
FIELD_SEPARATOR = "_"


def _word_pattern(spellings: Iterable[str]) -> re.Pattern[str]:
    """One regex matching any spelling of a genotype, on word boundaries.

    Boundaries matter more than they look: without them ``ki`` matches the
    ``ki`` in ``kidney`` and every sample in the folder comes back a knock-in.
    """
    alternatives = "|".join(re.escape(word) for word in spellings)
    return re.compile(rf"(?<![0-9a-z]){alternatives}(?![0-9a-z])", re.IGNORECASE)


_PATTERNS: dict[str, re.Pattern[str]] = {
    canonical: _word_pattern(spellings) for canonical, spellings in GENOTYPE_WORDS.items()
}


def fields(name: str) -> list[str]:
    """The underscore-separated fields of a file name, without its suffix."""
    stem = Path(str(name)).stem
    return [field for field in stem.split(FIELD_SEPARATOR) if field.strip()]


def genotype_from_name(name: str, pattern: str = "") -> str:
    """The genotype encoded in *name*, or ``""`` when it encodes none.

    *pattern* is a regular expression with either a group named ``genotype`` or
    one capturing group; it is tried against the whole stem and wins outright
    when it matches. A pattern that does not match falls through to the
    vocabulary rather than failing, so one odd file in a folder does not empty
    the column for the rest.

    Without a pattern the fields are read left to right and the first one
    carrying a known genotype word decides it. Left to right because acquisition
    software appends — channel names, dates, protocol numbers — so what the
    person typed is at the front and what the microscope added is at the back.
    """
    stem = Path(str(name)).stem
    if pattern:
        found = _from_pattern(stem, pattern)
        if found:
            return found

    for field in fields(stem) or [stem]:
        for canonical, matcher in _PATTERNS.items():
            if matcher.search(field):
                return canonical
    return ""


def _from_pattern(stem: str, pattern: str) -> str:
    """The capture *pattern* takes out of *stem*. Empty if it cannot be used."""
    try:
        match = re.search(pattern, stem, re.IGNORECASE)
    except re.error as exc:
        logger.info("genotype pattern %r is not a valid regex: %s", pattern, exc)
        return ""
    if match is None:
        return ""
    named = match.groupdict().get("genotype")
    if named:
        return named.strip()
    if match.groups():
        return (match.group(1) or "").strip()
    return match.group(0).strip()


def genotypes(names: Sequence[str], pattern: str = "") -> list[str]:
    """:func:`genotype_from_name` over a list, for a whole folder at once."""
    return [genotype_from_name(name, pattern) for name in names]
