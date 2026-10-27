"""The change between the two labels as part of the cutting price, on every hand-drawn section.

Same sections, split and scoring as ``junction_seeding.py`` (its ``annotated_all.csv``);
"default" here is today's defaults (no-seam fallback on). Every variant sets
``label_change`` (see ``junction.label_change_price``) and perhaps one more thing
-> ``junction_eval/labels.jsonl``.

    python notebooks/validation/junction_labels.py
    python notebooks/validation/junction_labels.py "change 1" ...   # run only these variants again
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

import junction_tuning as jt  # noqa: E402

SAMPLES = jt.OUT / "annotated_all.csv"
RESULTS = jt.OUT / "labels.jsonl"

#: Overrides of today's defaults, by name.
VARIANTS = {
    "default": {},
    "change 0.5": {"label_change": 0.5},
    "change 1": {"label_change": 1.0},
    "change 2": {"label_change": 2.0},
    "change 1, measured at 25 µm": {"label_change": 1.0, "label_change_um": 25.0},
    "change 1, measured at 100 µm": {"label_change": 1.0, "label_change_um": 100.0},
    "change 1, no-seam fallback off": {"label_change": 1.0, "no_seam_price": 0.0},
    "change 1 + bend 400": {"label_change": 1.0, "bend_um": 400.0},
    "change 1 + tears priced": {"label_change": 1.0, "tear_price": 1.0},
    "change 2 + tears priced": {"label_change": 2.0, "tear_price": 1.0},
    # Control: how much of the last two is the tears alone.
    "tears priced": {"tear_price": 1.0},
}


def run_one(task):
    """One section, one variant."""
    name, variant, params = task
    try:
        row = jt.run_case((name, {**jt.defaults(), **params}))
    except Exception as exc:
        row = {"name": name, "found": False, "error": f"{type(exc).__name__}: {exc}"}
    row.pop("line", None)
    return {**row, "variant": variant, "capped": jt.capped(row)}


def main() -> None:
    table = pd.read_csv(SAMPLES)
    only = sys.argv[1:]
    variants = {v: p for v, p in VARIANTS.items() if not only or v in only}
    tasks = [(n, v, p) for v, p in variants.items() for n in table["name"]]
    with ProcessPoolExecutor(12) as pool:
        rows = list(pool.map(run_one, tasks, chunksize=4))
    meta = table.set_index("name")[["set", "batch", "Ripped", "Folded", "Quality", "Shape"]]
    kept = []
    if only and RESULTS.exists():
        kept = [r for r in map(json.loads, RESULTS.read_text(encoding="utf-8").splitlines()) if r["variant"] not in variants]
    with RESULTS.open("w", encoding="utf-8") as f:
        for r in kept + [{**r, **meta.loc[r["name"]].to_dict()} for r in rows]:
            f.write(json.dumps(r, default=str) + "\n")
    print("written", RESULTS, flush=True)


if __name__ == "__main__":
    main()
