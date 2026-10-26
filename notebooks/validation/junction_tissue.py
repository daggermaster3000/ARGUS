"""The finder with the tissue found from the nuclear stain only vs from every channel.

One run per annotated sample and variant (cached overviews, see
``junction_tuning``), scored against the hand-drawn line; results to
``junction_eval/tissue.jsonl`` for ``junction_evaluation.ipynb``.

    python notebooks/validation/junction_tissue.py
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
VARIANTS = {"nuclear only": {"tissue_all_channels": False}, "all channels": {"tissue_all_channels": True}}
OUTFILE = jt.OUT / "tissue.jsonl"


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
