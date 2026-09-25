"""Group columns of an analysis — genotype, condition, … — read off file names and folders.

An experiment's groups are written down in two places: the file name
(``fish03_mut_20x.ims``) and the folder the file sits in (``DMSO/fish03.ims``).
Each column of the workbook is one :class:`ColumnRule`: which of those texts to
look at, and how to take the value out of it. The first rule is always the
genotype, because every plot and test downstream groups by it; the rest become
extra columns next to it.

Folders also make sample names collide — ``DMSO/fish1.ims`` and
``drug/fish1.ims`` are both "fish1" — and a workbook with two "fish1" cannot be
traced back to the files. :func:`label_samples` gives every sample a unique name
by adding the group values that tell the duplicates apart: ``fish1 (DMSO)`` and
``fish1 (drug)``.

Like :mod:`microscopy_viewer.naming`, **a rule that finds nothing gives an empty
string, never a guess.**

No Qt here: the Analysis panel is only a view of these rules.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from . import naming

GENOTYPE = "Genotype"

#: Where a column's value is read from. Short: they are the choices of a
#: drop-down in a narrow panel; :data:`EXPLAIN` has the long form.
FILE_NAME = "File name"
FOLDER = "Folder"
FOLDER_ABOVE = "Folder above"
WHOLE_PATH = "Whole path"
SOURCES = (FILE_NAME, FOLDER, FOLDER_ABOVE, WHOLE_PATH)

#: How the value is taken out of that text.
GENOTYPE_WORDS = "Genotype"
FIELD = "Part #"
ONE_OF = "One of…"
WHOLE = "All of it"
PATTERN = "Regex"
RULES = (GENOTYPE_WORDS, FIELD, ONE_OF, WHOLE, PATTERN)

#: The long form of every choice, for tooltips.
EXPLAIN = {
    FILE_NAME: "The file's name, without its extension.",
    FOLDER: "The name of the folder the file is in.",
    FOLDER_ABOVE: "The name of the folder above that one.",
    WHOLE_PATH: "Every folder below the experiment folder, then the file name.",
    GENOTYPE_WORDS: "A genotype word — wt, mut, het, hom, ko, ctl, … — with its spelling "
                    "variants folded together (WT, wildtype → wt).",
    FIELD: "One numbered part of the name, split at “_”. The parts of the first "
           "sample are shown under the table.",
    ONE_OF: "Whichever of the words you list appears in the name, written as you typed it.",
    WHOLE: "The whole name, as it is.",
    PATTERN: "A regular expression; the text its first (group) captures.",
}

#: What the argument of each rule is, for the panel's placeholder text.
ARGUMENT_HINTS = {
    GENOTYPE_WORDS: "",
    FIELD: "e.g. 2 (−1 = last)",
    ONE_OF: "e.g. DMSO, drug",
    WHOLE: "",
    PATTERN: "e.g. (\\d+)hpf",
}


@dataclass
class ColumnRule:
    """One group column: its header, where it is read and how."""

    name: str
    source: str = FILE_NAME
    rule: str = GENOTYPE_WORDS
    argument: str = ""

    def text(self, path: Path, root: Path | None = None) -> str:
        """The part of *path* this rule reads."""
        path = Path(path)
        if self.source == FOLDER:
            return path.parent.name
        if self.source == FOLDER_ABOVE:
            return path.parent.parent.name
        if self.source == WHOLE_PATH:
            relative = _relative(path, root)
            return "/".join([*relative.parent.parts, relative.stem])
        return path.stem

    def value(self, path: Path, root: Path | None = None) -> str:
        """This column's value for the file at *path*; ``""`` when not found."""
        text = self.text(path, root)
        if self.rule == WHOLE:
            return text.strip()
        if self.rule == FIELD:
            return field_of(text, self.argument)
        if self.rule == ONE_OF:
            return one_of(text, self.argument)
        if self.rule == PATTERN:
            return naming._from_pattern(text, self.argument) if self.argument.strip() else ""
        # Genotype words: the naming module's vocabulary, spelling variants folded,
        # searched folder by folder and then in the file name.
        found = (naming.genotype_from_name(piece) for piece in text.split("/"))
        return next((genotype for genotype in found if genotype), "")


def default_rules() -> list[ColumnRule]:
    """What the analysis did before the rules could be changed: genotype words in the file name."""
    return [ColumnRule(GENOTYPE)]


def parts(text: str) -> list[str]:
    """*text* split into its numbered parts, the way :data:`FIELD` counts them."""
    return [part for part in str(text).split(naming.FIELD_SEPARATOR) if part.strip()]


def field_of(text: str, argument: str) -> str:
    """Part *argument* (1-based, negative from the end) of *text*."""
    try:
        number = int(str(argument).strip())
    except ValueError:
        return ""
    pieces = parts(text)
    if number > 0 and number <= len(pieces):
        return pieces[number - 1].strip()
    if number < 0 and -number <= len(pieces):
        return pieces[number].strip()
    return ""


def one_of(text: str, argument: str) -> str:
    """The first of the comma-separated words in *argument* found in *text*.

    Matched as a whole word, ignoring case, so ``drug`` is not found inside
    ``nodrug``; returned as typed, so every sample of a group gets the same
    spelling whatever the file said.
    """
    for word in (w.strip() for w in str(argument).split(",")):
        if word and re.search(rf"(?<![0-9a-z]){re.escape(word)}(?![0-9a-z])", text, re.IGNORECASE):
            return word
    return ""


# ---------------------------------------------------------------------------
# Labelling samples
# ---------------------------------------------------------------------------


@dataclass
class SampleLabel:
    """What one file is called in the workbook, and its groups."""

    path: Path
    name: str
    genotype: str = ""
    #: Every column after the genotype, in rule order.
    conditions: dict[str, str] = field(default_factory=dict)


def label_samples(
    paths: Sequence[Path], rules: Sequence[ColumnRule] | None = None, root: Path | None = None
) -> list[SampleLabel]:
    """Name every file and read its group columns, in the order given."""
    rules = list(rules) if rules else default_rules()
    paths = [Path(p) for p in paths]
    headers = column_names(rules)
    rows = [{header: rule.value(path, root) for header, rule in zip(headers, rules)} for path in paths]
    names = unique_names(paths, rows, root)
    labels = []
    for path, name, row in zip(paths, names, rows):
        values = list(row.values())
        labels.append(SampleLabel(
            path=path,
            name=name,
            genotype=values[0] if values else "",
            conditions=dict(list(row.items())[1:]),
        ))
    return labels


def column_names(rules: Sequence[ColumnRule]) -> list[str]:
    """Headers of *rules*: the first is always Genotype, blanks and repeats get numbered."""
    headers: list[str] = []
    for index, rule in enumerate(rules):
        header = GENOTYPE if index == 0 else (rule.name.strip() or f"Condition {index}")
        base, count = header, 2
        while header in headers or (index and header == GENOTYPE):
            header, count = f"{base} {count}", count + 1
        headers.append(header)
    return headers


def unique_names(
    paths: Sequence[Path], rows: Sequence[dict[str, str]], root: Path | None = None
) -> list[str]:
    """File stems, with the differing group values added where two stems collide.

    ``fish1`` in ``DMSO/`` and ``drug/`` become ``fish1 (DMSO)`` and
    ``fish1 (drug)``: only the columns whose values differ inside that set of
    duplicates are added, so the name says what tells them apart and no more.
    When no column does, the folder is used; a number is the last resort.
    """
    stems = [Path(p).stem for p in paths]
    names = list(stems)
    groups: dict[str, list[int]] = {}
    for index, stem in enumerate(stems):
        groups.setdefault(stem, []).append(index)

    for stem, members in groups.items():
        if len(members) < 2:
            continue
        headers = list(rows[members[0]].keys()) if rows else []
        differing = [h for h in headers if len({rows[i].get(h, "") for i in members}) > 1]
        for i in members:
            suffix = ", ".join(rows[i][h] for h in differing if rows[i].get(h))
            names[i] = f"{stem} ({suffix})" if suffix else stem
        _separate_by_folder(members, names, paths, root)

    # Numbers only for what is still the same after all that.
    seen: dict[str, int] = {}
    for index, name in enumerate(names):
        if name in seen:
            seen[name] += 1
            names[index] = f"{name} #{seen[name]}"
        else:
            seen[name] = 1
    return names


def _separate_by_folder(members: list[int], names: list[str], paths, root) -> None:
    """Give duplicates that are still duplicates their folder as well."""
    clashes = [i for i in members if [names[j] for j in members].count(names[i]) > 1]
    if not clashes:
        return
    for i in clashes:
        folder = "/".join(_relative(Path(paths[i]), root).parent.parts) or Path(paths[i]).parent.name
        if not folder:
            continue
        stem = Path(paths[i]).stem
        extra = names[i][len(stem):].strip()
        inside = extra[1:-1] if extra.startswith("(") and extra.endswith(")") else extra
        names[i] = f"{stem} ({', '.join(x for x in (inside, folder) if x)})"


def _relative(path: Path, root: Path | None) -> Path:
    if root is not None:
        try:
            return Path(path).resolve().relative_to(Path(root).resolve())
        except ValueError:
            pass
    return Path(Path(path).parent.name) / Path(path).name


def apply(outcomes: Iterable, labels: Sequence[SampleLabel]) -> None:
    """Give each outcome the name and groups of its file."""
    by_path = {_key(label.path): label for label in labels}
    for outcome in outcomes:
        label = by_path.get(_key(outcome.path))
        if label is None:
            continue
        outcome.name = label.name
        outcome.genotype = label.genotype
        outcome.conditions = dict(label.conditions)


def _key(path) -> str:
    try:
        return str(Path(path).resolve())
    except OSError:
        return str(path)


# ---------------------------------------------------------------------------
# Remembering the rules
# ---------------------------------------------------------------------------


def rules_file() -> Path:
    from .runtime import app_data_dir

    return app_data_dir() / "group_columns.json"


def load_rules(path: Path | None = None) -> list[ColumnRule]:
    """The rules saved last time, or :func:`default_rules`."""
    import json

    target = path or rules_file()
    try:
        data = json.loads(Path(target).read_text(encoding="utf-8"))
        rules = [
            ColumnRule(
                name=str(item.get("name", "")),
                source=item.get("source") if item.get("source") in SOURCES else FILE_NAME,
                rule=item.get("rule") if item.get("rule") in RULES else GENOTYPE_WORDS,
                argument=str(item.get("argument", "")),
            )
            for item in data
        ]
    except (OSError, ValueError, TypeError, AttributeError):
        return default_rules()
    return rules or default_rules()


def save_rules(rules: Sequence[ColumnRule], path: Path | None = None) -> None:
    import json
    from dataclasses import asdict

    target = Path(path or rules_file())
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps([asdict(rule) for rule in rules], indent=1), encoding="utf-8")
    except OSError:
        pass
