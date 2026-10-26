"""Random walker against the minimum cut, on the hand-drawn junctions.

Used by ``junction_evaluation.ipynb``; run on its own to redo it (a few minutes)::

    python notebooks/validation/junction_walk.py

Every variant is run on every annotated sample from the cached overviews (see
``junction_tuning.py``), in parallel, and the rows — with each line — are
written to ``junction_eval/walk.jsonl``.
"""

from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import junction_tuning as jt  # noqa: E402

RESULTS = jt.OUT / "walk.jsonl"

#: What differs from the finder's defaults (cut, bending cost 1000).
VARIANTS = {
    "cut (current)": {},
    "walk": {"split_method": "walk", "bend_um": 0.0},
    "walk + bending": {"split_method": "walk"},
    "walk β 30 + bending": {"split_method": "walk", "walk_beta": 30.0},
    "walk β 1000 + bending": {"split_method": "walk", "walk_beta": 1000.0},
    "auto: cut, walk when it looks wrong": {"split_method": "auto"},
}


def main() -> None:
    samples = jt.annotated_samples()
    jt.build_cache(samples)
    names = list(samples["name"])
    tasks = [(name, params) for params in VARIANTS.values() for name in names]
    with ProcessPoolExecutor(max(1, (os.cpu_count() or 2) - 2)) as pool:
        rows = list(pool.map(jt.run_case, tasks, chunksize=1))
    with RESULTS.open("w", encoding="utf-8") as handle:
        for k, variant in enumerate(VARIANTS):
            for row in rows[k * len(names): (k + 1) * len(names)]:
                handle.write(json.dumps({"variant": variant, **row}) + "\n")
    print(f"{len(rows)} runs written to {RESULTS}")


if __name__ == "__main__":
    main()
