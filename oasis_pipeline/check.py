"""Compare the re-run (``rerun/``) with the shipped data and results, step by step.

    python check.py

* counting: per section, the thresholds and the number of objects of each mask, against ``data/counting/``
  (identical expected);
* cellpose: per section and method-threshold, the self-trained Cellpose counts (all, home, crossed, junction region)
  against ``results/counting/cellpose_counts.csv`` (identical expected with the shipped masks; close after ``--infer``);
* stats: the primary outcome and its sensitivity analyses against ``results/statistics/primary_outcome.csv``, numbers
  to 6 significant digits (only when every section was counted);
* tables: when the notebooks were run (paper package only), every table (``rerun/figures/*.csv`` against
  ``results/tables/``), cell by cell;
* browser: every section's counts in ``rerun/results/section_browser/data.json`` against ``results/section_browser/``,
  for the methods both have.

Prints what differs and exits 1 when anything does.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from oasis import paths  # noqa: E402

SHIPPED_MASKS = paths.DATA / "counting" / "af488_mask"
SHIPPED_TABLES = paths.ROOT / "results" / "tables"
SHIPPED_BROWSER = paths.ROOT / "results" / "section_browser"
SHIPPED_CELLPOSE_COUNTS = paths.ROOT / "results" / "counting" / "cellpose_counts.csv"
SHIPPED_STATISTICS = paths.ROOT / "results" / "statistics" / "primary_outcome.csv"


def check_counting() -> int:
    if not paths.MASKS.exists():
        print("counting: not re-run")
        return 0
    bad, n = [], 0
    for folder in sorted(paths.MASKS.iterdir()):
        mine, ref = folder / "summary.json", SHIPPED_MASKS / folder.name / "summary.json"
        if not mine.exists() or not ref.exists():
            continue
        n += 1
        a, b = json.loads(ref.read_text(encoding="utf-8")), json.loads(mine.read_text(encoding="utf-8"))
        oa = pd.read_parquet(SHIPPED_MASKS / folder.name / "objects.parquet", columns=["mask"])["mask"].value_counts()
        ob = pd.read_parquet(folder / "objects.parquet", columns=["mask"])["mask"].value_counts()
        if a["thresholds"] != b["thresholds"] or not oa.sort_index().equals(ob.sort_index()):
            bad.append(folder.name)
    print(f"counting: {n} sections compared, {len(bad)} differ" + (f": {bad[:10]}" if bad else ""))
    return len(bad)


def same_table(a: pd.DataFrame, b: pd.DataFrame) -> list[str]:
    out = []
    if a.shape != b.shape:
        return [f"shape {a.shape} → {b.shape}"]
    for c in a.columns:
        x, y = a[c], b[c]
        nx, ny = pd.to_numeric(x, errors="coerce"), pd.to_numeric(y, errors="coerce")
        if nx.notna().sum() == x.notna().sum() and x.notna().any():
            ok = np.isclose(nx.to_numpy(float), ny.to_numpy(float), rtol=1e-6, atol=1e-9, equal_nan=True)
        else:
            ok = (x.astype(str) == y.astype(str)).to_numpy()
        if not ok.all():
            i = int(np.flatnonzero(~ok)[0])
            out.append(f"{c}: {int((~ok).sum())} rows, e.g. {x.iloc[i]!r} → {y.iloc[i]!r}")
    return out


def check_tables() -> int:
    if not paths.FIGURES.exists():
        print("tables: not re-run")
        return 0
    bad = 0
    for ref in sorted(SHIPPED_TABLES.glob("*.csv")):
        mine = paths.FIGURES / ref.name
        if not mine.exists():
            print(f"  {ref.name}: missing in the re-run")
            bad += 1
            continue
        diffs = same_table(pd.read_csv(ref), pd.read_csv(mine))
        print(f"  {ref.name}: {'identical' if not diffs else 'differs'}")
        for d in diffs[:5]:
            print(f"      {d}")
        bad += bool(diffs)
    print(f"tables: {bad} differ")
    return bad


def check_cellpose() -> int:
    if not paths.CELLPOSE_COUNTS.exists():
        print("cellpose: not re-run")
        return 0
    mine, ref = pd.read_csv(paths.CELLPOSE_COUNTS), pd.read_csv(SHIPPED_CELLPOSE_COUNTS)
    m = mine.merge(ref, on=["section", "method"], how="left", suffixes=("", " shipped"), indicator=True)
    missing = int((m["_merge"] != "both").sum())
    cols = ("n", "home", "crossed", "region")
    differ = ~np.logical_and.reduce([m[c].fillna(-1).to_numpy() == m[c + " shipped"].fillna(-1).to_numpy() for c in cols])
    bad = sorted(set(m.loc[differ, "section"]))
    print(f"cellpose: {mine['section'].nunique()} sections, {len(mine)} method-thresholds compared, "
          f"{int(differ.sum())} differ in {len(bad)} sections" + (f": {bad[:10]}" if bad else "")
          + (f"; {missing} not in the shipped table" if missing else "") + f" (masks: {paths.cellpose()})")
    return int(differ.sum()) + missing


def check_stats() -> int:
    mine_file = paths.STATISTICS / "primary_outcome.csv"
    if not mine_file.exists():
        print("stats: not re-run")
        return 0
    mine, ref = pd.read_csv(mine_file), pd.read_csv(SHIPPED_STATISTICS)
    if mine["n sections"].max() != ref["n sections"].max():
        print(f"stats: re-run on {mine['n sections'].max()} sections, shipped on {ref['n sections'].max()}: not compared")
        return 0
    diffs = same_table(ref.sort_values(["outcome", "method"]).reset_index(drop=True),
                       mine.sort_values(["outcome", "method"]).reset_index(drop=True))
    p = mine[mine["primary"]].iloc[0]
    print(f"stats: primary outcome {p['method']} WT3/GD1 {p['fold WT3/GD1']:.2f} ({p['CI low']:.2f}–{p['CI high']:.2f}); "
          f"{'identical to the shipped' if not diffs else 'differs from the shipped'}")
    for d in diffs[:5]:
        print(f"      {d}")
    return int(bool(diffs))


def check_browser() -> int:
    mine = paths.BROWSER / "data.json"
    if not mine.exists():
        print("browser: not re-run")
        return 0
    a = {s["name"]: s["counts"] for s in json.loads((SHIPPED_BROWSER / "data.json").read_text(encoding="utf-8"))["sections"]}
    b = {s["name"]: s["counts"] for s in json.loads(mine.read_text(encoding="utf-8"))["sections"]}
    pick = lambda counts, keys: {k: {kk: counts[k].get(kk) for kk in ("n", "crossed", "home", "reg")} for k in keys}  # noqa: E731
    bad = [n for n in b if n in a and pick(a[n], set(a[n]) & set(b[n])) != pick(b[n], set(a[n]) & set(b[n]))]
    print(f"browser: {len(b)} sections compared, {len(bad)} with different counts" + (f": {bad[:10]}" if bad else ""))
    return len(bad)


def main() -> int:
    print(f"shipped: {paths.ROOT}\nre-run:  {paths.OUT}\nsegmentation used: {paths.segmentation()}\n")
    bad = check_counting() + check_cellpose() + check_stats() + check_tables() + check_browser()
    print("\nall identical" if not bad else f"\n{bad} differences (expected only after --segment or --infer)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
