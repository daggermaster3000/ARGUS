"""The object map's exclusions as a kept / excluded flag per object, for the browser's filtered counting method.

Reads the exclusions exported from ``object_map.html`` (:data:`EXCLUSIONS`: the feature ranges kept and the objects
excluded by hand, each as section id and object index) and the object features (``object_features.py``), and writes

- ``object_filter.parquet``: per object of ``gauss3-8`` (section id, object index) whether it is kept;
- ``object_filter.json``: the rules, the number of objects excluded by hand and in all, when.

``section_browser.py`` turns these into the method ``gfilt`` (8 SD): the ``gauss3-8`` objects that are kept, with their
outlines, counts and the square comparisons, and the figure builder writes its Methods sentence from the rules. The
paper package ships both files, so its pipeline gets the same counts without the map.

    python notebooks/official_analysis/object_filter.py [exclusions.json]

No Qt.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import object_features as of  # noqa: E402
import section_browser as sb  # noqa: E402

EXCLUSIONS = sb.OUT / "object_exclusions.json"       #: saved from object_map.html ("export JSON")
FLAGS = sb.OUT / "object_filter.parquet"
META = sb.OUT / "object_filter.json"


def kept_flags(exclusions: dict, t: pd.DataFrame) -> np.ndarray:
    """True for the objects inside every feature range and not excluded by hand."""
    keep = np.ones(len(t), bool)
    for r in exclusions.get("rules", []):
        v = t[r["f"]].to_numpy(float)
        if r.get("lo") is not None:
            keep &= v >= r["lo"]
        if r.get("hi") is not None:
            keep &= v <= r["hi"]
    hand = {(sid, int(i)) for sid, i in exclusions.get("excluded", [])}
    if hand:
        keep &= ~np.array([(s, int(i)) in hand for s, i in zip(t["section"], t["index"])])
    return keep


def main(argv: list[str]) -> int:
    src = Path(argv[0]) if argv else EXCLUSIONS
    exclusions = json.loads(src.read_text(encoding="utf-8"))
    if exclusions.get("key", of.KEY) != of.KEY:
        raise SystemExit(f"{src} is for {exclusions['key']}, the objects are {of.KEY}")
    cols = ["section", "index"] + sorted({r["f"] for r in exclusions.get("rules", [])})
    t = pd.concat([pd.read_parquet(f, columns=cols) for f in sorted(of.OUT.glob("*.parquet"))], ignore_index=True)
    keep = kept_flags(exclusions, t)
    pd.DataFrame({"section": t["section"], "index": t["index"].astype(np.int32), "kept": keep}).to_parquet(FLAGS, index=False)
    meta = {"key": of.KEY, "rules": exclusions.get("rules", []), "n_hand": len(exclusions.get("excluded", [])),
            "n_objects": int(len(t)), "n_kept": int(keep.sum()), "source": src.name,
            "map_generated": exclusions.get("generated"), "written": time.strftime("%Y-%m-%d %H:%M")}
    META.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"{meta['n_kept']} of {meta['n_objects']} objects kept ({meta['n_hand']} excluded by hand; rules "
          f"{meta['rules']}) -> {FLAGS}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
