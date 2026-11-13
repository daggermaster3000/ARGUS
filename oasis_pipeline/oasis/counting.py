"""The counting method on every analysed section: the AF488 mask's objects, each cell's cover, the thresholds.

The mask (``cv.classical_masks``): AF488 (maximum over z) smoothed 0.5 µm, less its 20 µm Gaussian background, cut
inside the tissue at the median + k robust SD of the section's own tissue, k = 3, 5, 8 (``kmad3`` …; ``kmad8`` is
the counting method); objects < 20 µm² removed, holes < 10 µm² filled. Objects outside the section are then taken
out of each mask (``section_outline.drop_outside``). One connected object (8-connectivity) is one AF488⁺ cell.

Per section, into ``rerun/counting/af488_mask/<section>/``:

* ``objects.parquet`` — every object of each mask: area, centroid (µm), signed distance to the junction line
  (+ on the junction region's side), beside the line or past its ends, in the junction region;
* ``cells.parquet`` — every segmented cell (``Label``) and the share of its grown region each mask covers;
* ``summary.json`` — the thresholds, the noise SD, the objects dropped as outside the section;
* ``outline.npz`` — the section's outline (``section_outline``).

About 20 s a section on a CPU, :data:`WORKERS` at a time (each holds a scan's planes, ~2 GB).
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from . import cells as ja
from . import cv, paths, section_outline

WORKERS = 4
VERSION = 2


def run_one(name: str) -> str:
    """The masks of one section, their cover of each cell, and their objects."""
    import pandas as pd
    from scipy import ndimage
    from skimage.measure import label, regionprops_table

    cv.below_normal_priority()
    started = time.perf_counter()
    info = paths.section_info(name)
    af488, nuclear, voxel = cv.read_planes(paths.scan_path(name))
    tissue = cv.tissue_mask(nuclear)
    del nuclear
    masks, meta = cv.classical_masks(af488, tissue, voxel)
    del af488
    folder = paths.MASKS / name
    outline = section_outline.outline_for(folder, paths.scan_path(name))
    line = section_outline.line_um(info)
    outside = {}
    for method in cv.METHODS:
        masks[method], outside[method] = section_outline.drop_outside(masks[method], outline, line, voxel)
    grown = ja.read_labels(paths.grown_tif(name))
    ids = np.unique(grown[grown > 0]).astype(np.int64)
    size = np.asarray(ndimage.sum(grown > 0, grown, ids), dtype=float)
    cells = pd.DataFrame({"Label": ids})
    for method in cv.METHODS:
        cells[f"cover {method}"] = np.asarray(ndimage.sum(masks[method], grown, ids)) / np.where(size > 0, size, np.nan)
    del grown

    band = ja.read_labels(paths.region_tif(name)) > 0 if paths.region_tif(name).exists() else None
    objects = []
    for method in cv.METHODS:
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

    folder.mkdir(parents=True, exist_ok=True)
    cells.to_parquet(folder / "cells.parquet", index=False)
    (pd.concat(objects, ignore_index=True) if objects else pd.DataFrame()).to_parquet(folder / "objects.parquet",
                                                                                       index=False)
    summary = {"version": VERSION, "sample": name, "file": paths.scan_path(name).name, "voxel_um": voxel,
               "thresholds": {m: meta["thresholds"][m] for m in cv.METHODS}, "noise_sd": meta["noise_sd"],
               "background_centre": meta["background_centre"], "outside": outside,
               "seconds": round(time.perf_counter() - started, 1)}
    (folder / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return f"{name}: {summary['seconds']:.0f} s, outside the section {outside}"


def done(name: str) -> bool:
    try:
        return json.loads((paths.MASKS / name / "summary.json").read_text(encoding="utf-8")).get("version", 0) >= VERSION
    except Exception:
        return False


def main(names: list[str] | None = None, workers: int = WORKERS) -> int:
    names = names or paths.section_names()
    todo = [n for n in names if not done(n)]
    print(f"counting: {len(names)} sections, {len(todo)} to do -> {paths.MASKS}", flush=True)
    paths.MASKS.mkdir(parents=True, exist_ok=True)
    started, failed = time.perf_counter(), 0
    with ProcessPoolExecutor(workers) as pool:
        futures = {pool.submit(run_one, n): n for n in todo}
        for index, future in enumerate(as_completed(futures), start=1):
            try:
                line = future.result()
            except Exception as exc:
                failed += 1
                line = f"{futures[future]}: FAILED {exc!r}"
            elapsed = time.perf_counter() - started
            print(f"[{index}/{len(todo)}] {line} — {elapsed / 60:.0f} min, "
                  f"~{elapsed / index * (len(todo) - index) / 60:.0f} min left", flush=True)
    return 1 if failed else 0
