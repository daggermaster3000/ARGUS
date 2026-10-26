"""Checks for grouping sections by assembloid.

No files: each section is a row of shape figures and a shape signature drawn
as an ellipse (the tissue) with its left part labelled, sizes drifting slowly
from one section to the next, so which sections belong together is known.

Run with::

    python tests/test_assembloids.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from microscopy_viewer import assembloids as asm  # noqa: E402

_failures: list[str] = []
PX = 64


def check(condition: bool, message: str) -> None:
    if condition:
        print(f"  ok   {message}")
    else:
        print(f"  FAIL {message}")
        _failures.append(message)


def _signature(long_px, short_px, labelled=0.4, notch=0.0):
    """An ellipse on the signature grid, its left *labelled* share labelled; *notch* bites into its top."""
    rows, cols = np.mgrid[:PX, :PX] - (PX - 1) / 2
    tissue = (cols / long_px) ** 2 + (rows / short_px) ** 2 <= 1
    if notch:
        tissue &= ~(((cols - long_px * 0.3) / (notch * long_px)) ** 2 + ((rows + short_px) / (notch * short_px)) ** 2 <= 1)
    left = cols < -long_px + 2 * long_px * labelled
    return np.stack([tissue, tissue & left])


def _section(rng, slide, scene, long_px, short_px, labelled=0.4, notch=0.0):
    long_px *= np.exp(rng.normal(0, 0.02))
    short_px *= np.exp(rng.normal(0, 0.02))
    area = np.pi * long_px * short_px * 60 * 60 / 1e6
    return {"name": f"{slide}_s{scene}", "Tissue area (mm2)": area,
            "Major axis (um)": 120 * long_px, "Minor axis (um)": 120 * short_px,
            "Labelled area (mm2)": area * labelled, "Other area (mm2)": area * (1 - labelled),
            "Signature": _signature(long_px, short_px, labelled, notch)}


def _frame():
    import pandas as pd

    rng = np.random.default_rng(1)
    rows = []
    # Slide 001: one assembloid, growing slowly; scene 7 is torn (a much smaller piece).
    for k, s in enumerate([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]):
        if s == 7:
            rows.append(_section(rng, "Asm_001-001", s, 10, 6, 0.6))
        else:
            rows.append(_section(rng, "Asm_001-001", s, 22 + 0.3 * k, 15 + 0.2 * k))
    # Slide 002: two assemblies cut together — scenes alternate — until the
    # round one runs out (scene 13); the long, notched one carries on alone.
    k_long = k_round = 0
    for s in range(1, 21):
        if s <= 12 and s % 2 == 0:
            rows.append(_section(rng, "Asm_002-001", s, 17 - 0.3 * k_round, 16 - 0.3 * k_round, 0.55))
            k_round += 1
        else:
            rows.append(_section(rng, "Asm_002-001", s, 27 + 0.2 * k_long, 11 + 0.1 * k_long, 0.3, notch=0.35))
            k_long += 1
    # Slide 003: the long one runs on.
    for k, s in enumerate(range(1, 9)):
        rows.append(_section(rng, "Asm_003-002", s, 27 + 0.2 * (k_long + k), 11 + 0.1 * (k_long + k), 0.3, notch=0.35))
    # Slide 004: another assembloid altogether.
    for s in range(1, 11):
        rows.append(_section(rng, "Asm_004-003", s, 20, 19, 0.7))
    return pd.DataFrame(rows).sample(frac=1, random_state=0)  # the order of the rows does not matter


def test_names_are_parsed() -> None:
    print("section names")
    check(asm.parse("2025_09_26_Asssembloid_019-018_s7") == ("2025_09_26_Asssembloid_019-018", 7), "slide and scene")
    check(asm.parse("odd name") == ("odd name", None), "a name without a scene is its own slide")


def test_signatures() -> None:
    print("signatures")
    a = _signature(22, 15)
    check(asm.signature_distance(a, a) == 0.0, "a section is 0 from itself")
    check(asm.signature_distance(a, a[:, ::-1, ::-1]) == 0.0, "and from itself turned half round")
    check(asm.signature_distance(a, _signature(22.5, 15.3)) < 0.15, "a neighbouring section is close")
    check(asm.signature_distance(a, _signature(14, 13, 0.6)) > 0.4, "another assembloid is not")
    text = np.packbits(a).tobytes().hex()
    check(np.array_equal(asm.decode_signature(text), a), "a stored signature reads back")


def test_sections_are_grouped_by_assembloid() -> None:
    print("grouping sections")
    out = asm.group_sections(_frame()).set_index("name")
    one = out[out["Slide"] == "Asm_001-001"]
    check(one["Assembloid"].nunique() == 1 and one["Pattern"].iloc[0] == "one",
          "a slide of one assembloid is one, though one section is torn")
    check(bool(out.loc["Asm_001-001_s7", "Odd section"]) and int(one["Odd section"].sum()) == 1,
          "the torn section is the odd one out, and nothing else is")
    two = out[out["Slide"] == "Asm_002-001"]
    rounds = two.loc[[f"Asm_002-001_s{s}" for s in range(2, 13, 2)], "Assembloid"]
    longs = two.loc[[f"Asm_002-001_s{s}" for s in [1, 3, 5, 11, 13, 14, 20]], "Assembloid"]
    check(rounds.nunique() == 1 and longs.nunique() == 1 and rounds.iloc[0] != longs.iloc[0],
          f"two assemblies cut together are told apart ({rounds.iloc[0]}, {longs.iloc[0]})")
    check(two["Pattern"].iloc[0] in ("alternating", "mixed"), f"the slide's pattern ({two['Pattern'].iloc[0]})")
    check(out.loc["Asm_003-002_s1", "Assembloid"] == longs.iloc[0] and out.loc["Asm_003-002_s1", "Continues from"] != "",
          "the one that carries on alone runs on onto the next slide, under its name")
    check(out.loc["Asm_004-003_s1", "Assembloid"] not in set(out.loc[out["Slide"] != "Asm_004-003", "Assembloid"]),
          "a different one on the next slide does not")
    check(out["Assembloid"].nunique() == 4, f"four assemblies in all ({sorted(out['Assembloid'].unique())})")
    check(out["Step"].dropna().median() < 0.15, "and within each, a section is close to the one before")


def test_series_columns_stop_running_on() -> None:
    print("running on only within a series")
    frame = _frame()
    frame["Genotype"] = np.where(frame["name"].str.startswith("Asm_003"), "B", "A")
    out = asm.group_sections(frame, series=["Genotype"]).set_index("name")
    check(out.loc["Asm_003-002_s1", "Continues from"] == "", "a slide of another genotype never continues an assembloid")


def test_without_signatures_the_figures_stand_in() -> None:
    print("without signatures")
    frame = _frame().drop(columns=["Signature"])
    out = asm.group_sections(frame)
    check(out["Assembloid"].notna().all(), "every section still gets an assembloid")
    one = out[out["Slide"] == "Asm_004-003"]
    check(one["Assembloid"].nunique() == 1, "and a plain slide of one assembloid stays one")


def main() -> int:
    for test in (test_names_are_parsed, test_signatures, test_sections_are_grouped_by_assembloid,
                 test_series_columns_stop_running_on, test_without_signatures_the_figures_stand_in):
        test()
        print()
    if _failures:
        print(f"{len(_failures)} check(s) failed:")
        for failure in _failures:
            print(f"  - {failure}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
