"""Does the automatic band change the cell counts? The hand-drawn band against it.

Used by ``junction_evaluation.ipynb``; run on its own to redo it (Cellpose on the
GPU, a few minutes a sample)::

    python notebooks/validation/junction_downstream.py            # every annotated sample
    python notebooks/validation/junction_downstream.py 1          # just the first, to time it

Each annotated sample is counted twice by the junction analysis with its default
settings: once in the hand-drawn region stored with it, once in the band the
finder draws with its defaults (from the cached overview, with the hand-drawn
band's width, gap and rim, as in ``junction_tuning.run_case``). Each count sets its
own threshold from its own cells — as a real run would. Where a second person
has drawn the region too (``junction_tuning.SECOND_KEY``) it is counted as a
third band, for the agreement between two annotators. A row per sample and
band, with the per-channel counts, goes to ``junction_eval/downstream.jsonl``;
a band already there is skipped, so rerunning picks up new second annotations.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import junction_tuning as jt  # noqa: E402

from microscopy_viewer import fusion as fu  # noqa: E402
from microscopy_viewer import ims_store  # noqa: E402
from microscopy_viewer import junction as jn  # noqa: E402
from microscopy_viewer import junction_analysis as ja  # noqa: E402

RESULTS = jt.OUT / "downstream.jsonl"


def automatic_band(name: str) -> np.ndarray | None:
    """The finder's band at full resolution, with its default settings."""
    overview, _band, _line, base = jt.load_case(name)
    options = jn.JunctionOptions(save=False, **base)
    result = jn.find_junction(overview, options)
    if result.line_um is None or result.band is None:
        return None
    band = jn.full_resolution_band(result, overview, options) > 0
    return band if band.any() else None


def row_of(name: str, which: str, counts: ja.JunctionCounts) -> dict:
    row = {"name": name, "band": which, "source": counts.band_source, "error": counts.error,
           "n_cells": counts.n_cells, "area_mm2": counts.band_area_mm2, "seconds": counts.elapsed_s}
    for channel in counts.channels:
        if channel.reference:
            continue
        row[f"{channel.channel} positive"] = channel.n_positive
        row[f"{channel.channel} %"] = channel.percent
        row[f"{channel.channel} threshold"] = channel.threshold
    return row


def main(limit: int | None = None) -> None:
    samples = jt.annotated_samples()
    jt.build_cache(samples)
    done = set()
    if RESULTS.exists():
        done = {(r["name"], r["band"]) for r in map(json.loads, RESULTS.read_text(encoding="utf-8").splitlines())}
    wanted = []
    for row in samples.itertuples():
        bands = ["manual", "automatic"]
        if jt.SECOND_KEY in ims_store.list_labels(row.path):
            bands.append("second")
        if any((row.name, band) not in done for band in bands):
            wanted.append((row, [band for band in bands if (row.name, band) not in done]))
    for index, (row, bands) in enumerate(wanted[:limit], start=1):
        print(f"{index}/{len(wanted[:limit])} {row.name}: {', '.join(bands)}", flush=True)
        rows = []
        for which in bands:
            if which == "automatic":
                band = automatic_band(row.name)
                if band is None:
                    rows.append({"name": row.name, "band": which, "error": "no junction found"})
                    continue
                counts = ja.analyse_sample(row.path, ja.CountOptions(find_junction=False), band=band)
            else:
                key = fu.MANUAL_KEY if which == "manual" else jt.SECOND_KEY
                counts = ja.analyse_sample(row.path, ja.CountOptions(band_key=key, find_junction=False))
            rows.append(row_of(row.name, which, counts))
        with RESULTS.open("a", encoding="utf-8") as handle:
            for r in rows:
                handle.write(json.dumps(r) + "\n")
        print("   ", "; ".join(f"{r['band']}: {r.get('error') or r.get('n_cells')} cells" for r in rows), flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
