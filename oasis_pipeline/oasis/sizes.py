"""Per-section sizes and the slim cell cache the notebooks read.

* **Tissue area** (mm²): the tissue the AF488 count is thresholded in (the nuclear stain at 1/8, Otsu, closed, holes
  filled, widened two steps; ``cv.tissue_mask``), counted on its 1/8 grid → ``section_tissue_area.csv``.
* **Assembloid length** (µm): the spread of the segmented cells beside the junction line across it, 0.5th to 99.5th
  percentile of their signed distance to the line → ``section_scales.csv``.
* **Cell cache**: each section's cells in the junction region or beside the line within 650 µm of it, a few columns
  → ``cell_cache/<section>.parquet``.
"""
from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from . import cells as ja
from . import junction as jn
from . import paths, scan

STEP = 8                          #: resolution step of the tissue mask (cv.tissue_mask's default)
KEEP_UM = 650.0                   #: cells further than this from the line are not kept in the cache
WORKERS = 4
CHANNELS = ("AF488", "AF555")
CACHE_COLUMNS = {
    "Label": "label", "Centroid Y (µm)": "y", "Centroid X (µm)": "x", "Nucleus area (µm²)": "area",
    ja.IN_REGION_COLUMN: "region", ja.LINE_COLUMN: "d", ja.BESIDE_COLUMN: "beside", ja.EDGE_COLUMN: "edge",
    **{f"{ch} {what}": f"{ch} {what}" for ch in CHANNELS
       for what in ("mean", "P95", "ring mean", "ring contrast", "mean (median filter)", "positive")},
}


def small_tissue(nuclear: np.ndarray, step: int = STEP) -> np.ndarray:
    """cv.tissue_mask, kept at 1/*step*."""
    from scipy import ndimage
    from skimage.filters import threshold_otsu

    small = np.asarray(nuclear[::step, ::step], dtype=np.float32)
    mask = small > threshold_otsu(small)
    mask = ndimage.binary_closing(mask, iterations=3)
    mask = ndimage.binary_fill_holes(mask)
    return ndimage.binary_dilation(mask, iterations=2)


def tissue_area(name: str) -> dict:
    specs = scan.load_path(paths.scan_path(name))
    names = [str(s.channel_name or s.name) for s in specs]
    spec = specs[jn.pick_channel(names, "", jn.NUCLEAR_HINTS)]
    voxel = float(spec.scale[-1])
    nuclear = scan.plane(spec)
    mask = small_tissue(nuclear)
    rows = np.minimum(STEP, nuclear.shape[0] - np.arange(mask.shape[0]) * STEP)
    cols = np.minimum(STEP, nuclear.shape[1] - np.arange(mask.shape[1]) * STEP)
    pixels = float(rows @ mask.astype(np.int64) @ cols)
    return {"name": name, "tissue area (mm²)": pixels * voxel ** 2 / 1e6, "voxel (µm)": voxel}


def assembloid_length(name: str) -> dict:
    t = pd.read_csv(paths.cells_csv(name), usecols=[ja.LINE_COLUMN, ja.BESIDE_COLUMN], engine="pyarrow")
    away = t[ja.LINE_COLUMN].to_numpy(float)[t[ja.BESIDE_COLUMN].to_numpy(bool)]
    away = away[np.isfinite(away)]
    length = np.percentile(away, 99.5) - np.percentile(away, 0.5) if len(away) else np.nan
    return {"name": name, "assembloid length (µm)": length}


def slim(name: str) -> None:
    header = pd.read_csv(paths.cells_csv(name), nrows=0).columns
    use = [c for c in CACHE_COLUMNS if c in header]
    table = pd.read_csv(paths.cells_csv(name), usecols=use, engine="pyarrow").rename(columns=CACHE_COLUMNS)
    keep = table["region"].astype(bool) | (table["beside"].astype(bool) & (table["d"].abs() <= KEEP_UM))
    table = table[keep].reset_index(drop=True)
    for c in table.columns:
        if table[c].dtype == np.float64:
            table[c] = table[c].astype(np.float32)
    paths.CACHE.mkdir(parents=True, exist_ok=True)
    table.to_parquet(paths.CACHE / f"{name}.parquet", index=False)


def main(names: list[str] | None = None, workers: int = WORKERS) -> int:
    names = names or paths.section_names()
    paths.DERIVED.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    rows = []
    with ProcessPoolExecutor(workers) as pool:
        futures = {pool.submit(tissue_area, n): n for n in names}
        for k, future in enumerate(as_completed(futures), start=1):
            rows.append(future.result())
            if k % 100 == 0:
                print(f"tissue area: {k}/{len(names)}", flush=True)
    pd.DataFrame(rows).sort_values("name").to_csv(paths.TISSUE_AREA_CSV, index=False)
    pd.DataFrame([assembloid_length(n) for n in names]).to_csv(paths.SCALES_CSV, index=False)
    for n in names:
        slim(n)
    print(f"sizes and cell cache: {len(names)} sections ({time.perf_counter() - started:.0f} s)", flush=True)
    return 0
