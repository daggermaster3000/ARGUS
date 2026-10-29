"""No clear junction: dominant seeds, the cells' edge, and a fallback when there is no seam.

One run per annotated sample and variant (cached overviews, see
``junction_tuning``), scored against the hand-drawn line; results to
``junction_eval/seam.jsonl`` for ``01_junction_evaluation.ipynb``.

    python notebooks/validation/junction_seam.py
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

import junction_tuning as jt  # noqa: E402

#: Overrides of the defaults, by name.
VARIANTS = {
    "default": {},
    "dominant seeds": {"seed_rule": "dominant"},
    "dominant seeds + edge 0.5": {"seed_rule": "dominant", "edge_weight": 0.5},
    "core seeds + edge 0.5": {"edge_weight": 0.5},
    "no seam > 0.4: dominant seeds": {"no_seam_price": 0.4, "no_seam_edge": 0.0},
    "no seam > 0.4: dominant seeds + edge 0.5": {"no_seam_price": 0.4, "no_seam_edge": 0.5},
    "no seam > 0.3: dominant seeds + edge 0.5": {"no_seam_price": 0.3, "no_seam_edge": 0.5},
}
OUTFILE = jt.OUT / "seam.jsonl"


def main() -> None:
    tune, held = jt.split(jt.annotated_samples(keep_excluded=True))
    names = tune + held
    tasks = [(variant, (n, {**jt.defaults(), **override})) for variant, override in VARIANTS.items() for n in names]
    with ProcessPoolExecutor(12) as pool:
        rows = list(pool.map(jt.run_case, [t for _v, t in tasks]))
    with OUTFILE.open("w", encoding="utf-8") as f:
        for (variant, (name, _p)), row in zip(tasks, rows):
            row = {k: v for k, v in row.items() if k != "line"}
            f.write(json.dumps({**row, "variant": variant, "set": "held out" if name in held else "tuning",
                                "capped": jt.capped(row)}) + "\n")
    print("written", OUTFILE)


if __name__ == "__main__":
    main()
