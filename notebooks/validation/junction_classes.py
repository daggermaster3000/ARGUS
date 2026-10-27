"""Settings for sections with and without a dark line at the junction, chosen separately.

Every combination in GRID runs on every hand-drawn section (``annotated_all.csv``),
on the whole section with the no-seam fallback off, so each run is one clean set of
settings -> ``junction_eval/classes.jsonl``. The notebook then splits the sections by
the seam price of today's first cut (``seeding_diagnostics.csv``) and picks, per
class, the combination best on that class's tuning sections.

    python notebooks/validation/junction_classes.py
"""
from __future__ import annotations

import itertools
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

import junction_tuning as jt  # noqa: E402
from microscopy_viewer import junction as jn  # noqa: E402

SAMPLES = jt.OUT / "annotated_all.csv"
RESULTS = jt.OUT / "classes.jsonl"

#: Every combination of these is run; ``seed_lead`` is the module's SEED_LEAD.
GRID = {
    "seed_rule": ["core", "dominant"],
    "label_change": [0.0, 1.0],
    "tear_price": [0.0, 1.0],
    "bend_um": [400.0, 1000.0],
    "edge_weight": [0.0, 0.5],
}
SEED_LEAD = 0.35
#: Under every combination: one cut, no second one.
BASE = {"no_seam_price": 0.0}


def combos() -> list[dict]:
    keys = list(GRID)
    return [dict(zip(keys, values)) for values in itertools.product(*GRID.values())]


def run_one(task):
    name, combo = task
    saved = jn.SEED_LEAD
    jn.SEED_LEAD = SEED_LEAD
    try:
        row = jt.run_case((name, {**jt.defaults(), **BASE, **combo}))
    except Exception as exc:
        row = {"name": name, "found": False, "error": f"{type(exc).__name__}: {exc}"}
    finally:
        jn.SEED_LEAD = saved
    row.pop("line", None)
    return {**row, **combo, "capped": jt.capped(row)}


def main() -> None:
    table = pd.read_csv(SAMPLES)
    tasks = [(n, c) for c in combos() for n in table["name"]]
    print(f"{len(tasks)} runs", flush=True)
    with ProcessPoolExecutor(12) as pool:
        rows = list(pool.map(run_one, tasks, chunksize=4))
    meta = table.set_index("name")[["set", "batch"]]
    with RESULTS.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({**r, **meta.loc[r["name"]].to_dict()}, default=str) + "\n")
    print("written", RESULTS, flush=True)


if __name__ == "__main__":
    main()
