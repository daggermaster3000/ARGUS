"""The classical AF488 mask on every section of a cells-analysis run, for ``04_genotype_by_rule.ipynb``.

The mask is the one ``notebooks/validation/04_classical_vs_cellpose.ipynb`` validated
(``classical_vs_cellpose.classical_masks``): AF488 flattened over z, smoothed 0.5 µm,
its local background (a 20 µm Gaussian) taken away, and cut inside the tissue at
the median + k robust SDs of what is left, for k = 3, 5 and 8 (``kmad3`` …);
objects under 20 µm² dropped, holes under 10 µm² filled. On 39 scans ``kmad5``
called almost the same cells as the run's rule (κ 0.79).

Per section, into ``OUT/<sample>/``:

* ``cells.parquet`` — every cell of the run (``Label``) and the share of its grown
  region each mask covers (``cover kmad3`` …); a cell is AF488+ by a mask when the
  share is at least 25 % (the validation's ``COVER``);
* ``objects.parquet`` — every object of each ``kmad`` mask, the mask's own cells:
  area, centroid, distance to the junction line (positive on the region's side,
  as the run's cells) and whether it is beside the line or past its ends;
* ``summary.json`` — the thresholds and the noise SD.

Run from the repository root (CPU only, ~20 s a section, a few sections at a time)::

    python notebooks/dataset/af488_masks.py          # every section not done yet
    python notebooks/dataset/af488_masks.py 3        # just three, to time it
"""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "validation"))
sys.path.insert(0, str(HERE.parents[1]))
import classical_vs_cellpose as cv  # noqa: E402

from microscopy_viewer import junction_analysis as ja  # noqa: E402

RUN = Path(r"D:\Slidescanner extracted\cells_analysis\2026-10-28_090440 re-scored")
OUT = Path(r"D:\Slidescanner extracted\run_analysis_20261028\af488_mask")
#: The masks kept: the k × robust SD ones (triangle and Otsu were no better in the validation).
METHODS = tuple(f"kmad{k:g}" for k in cv.K_MADS)
#: Processes at a time: each holds a slide scan's planes and masks (~2 GB).
WORKERS = 4
VERSION = 1


def run_one(name: str) -> str:
    """The masks of one section, their cover of each cell, and their objects."""
    import pandas as pd
    from scipy import ndimage
    from skimage.measure import label, regionprops_table

    cv.below_normal_priority()
    started = time.perf_counter()
    info = json.loads((RUN / "cells" / f"{name}.json").read_text(encoding="utf-8"))
    af488, nuclear, voxel = cv.read_planes(Path(info["file"]))
    tissue = cv.tissue_mask(nuclear)
    del nuclear
    masks, meta = cv.classical_masks(af488, tissue, voxel)
    del af488
    grown = ja.read_labels(RUN / info["grown_file"])
    ids = np.unique(grown[grown > 0]).astype(np.int64)
    size = np.asarray(ndimage.sum(grown > 0, grown, ids), dtype=float)
    cells = pd.DataFrame({"Label": ids})
    for method in METHODS:
        cells[f"cover {method}"] = np.asarray(ndimage.sum(masks[method], grown, ids)) / np.where(size > 0, size, np.nan)
    del grown

    band = ja.read_labels(RUN / info["region_file"]) > 0 if info.get("region_file") else None
    line = np.asarray(info["line"], dtype=float) if info.get("line") else None
    objects = []
    for method in METHODS:
        labelled = label(masks[method], connectivity=2)
        if not labelled.max():
            continue
        props = pd.DataFrame(regionprops_table(labelled, properties=("label", "area", "centroid")))
        table = pd.DataFrame({"mask": method, "Object": props["label"], "Area (µm²)": props["area"] * voxel ** 2,
                              "Centroid Y (µm)": props["centroid-0"] * voxel, "Centroid X (µm)": props["centroid-1"] * voxel})
        if band is not None and line is not None:
            ja.place_cells(table, band, line, (voxel, voxel), None)
            rows = np.clip(np.round(props["centroid-0"]).astype(int), 0, band.shape[0] - 1)
            cols = np.clip(np.round(props["centroid-1"]).astype(int), 0, band.shape[1] - 1)
            table[ja.IN_REGION_COLUMN] = band[rows, cols]
        objects.append(table)

    folder = OUT / name
    folder.mkdir(parents=True, exist_ok=True)
    cells.to_parquet(folder / "cells.parquet", index=False)
    (pd.concat(objects, ignore_index=True) if objects else pd.DataFrame()).to_parquet(folder / "objects.parquet",
                                                                                       index=False)
    summary = {"version": VERSION, "sample": name, "file": info["file"], "voxel_um": voxel,
               "thresholds": {m: meta["thresholds"][m] for m in METHODS}, "noise_sd": meta["noise_sd"],
               "background_centre": meta["background_centre"], "seconds": round(time.perf_counter() - started, 1)}
    (folder / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return f"{name}: {summary['seconds']:.0f} s"


def done(name: str) -> bool:
    target = OUT / name / "summary.json"
    try:
        return json.loads(target.read_text(encoding="utf-8")).get("version", 0) >= VERSION
    except Exception:
        return False


def main(argv: list[str]) -> int:
    names = sorted(p.stem for p in (RUN / "cells").glob("*.json"))
    todo = [n for n in names if not done(n)]
    if argv:
        todo = todo[: int(argv[0])]
    print(f"{len(names)} sections, {len(todo)} to do -> {OUT}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with ProcessPoolExecutor(WORKERS) as pool:
        futures = {pool.submit(run_one, n): n for n in todo}
        for index, future in enumerate(as_completed(futures), start=1):
            try:
                line = future.result()
            except Exception as exc:
                line = f"{futures[future]}: FAILED {exc!r}"
            elapsed = time.perf_counter() - started
            print(f"[{index}/{len(todo)}] {line} — {elapsed / 60:.0f} min, "
                  f"~{elapsed / index * (len(todo) - index) / 60:.0f} min left", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
