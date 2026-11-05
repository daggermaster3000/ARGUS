"""The area of each analysed section's tissue (the assembloid), for ``02_biology.ipynb``.

The tissue is the one the AF488 count is thresholded in: the nuclear stain at 1/8 resolution,
Otsu, closed, holes filled, dilated by two steps (``classical_vs_cellpose.tissue_mask``). Its
area is counted on that 1/8 grid, so it is the same mask without blowing it up to full size.

Writes ``run_analysis_20261028/section_tissue_area.csv`` (name, tissue area in mm², pixel size);
sections already there are skipped. About 2 s a section, 4 at a time. No Qt.

    python notebooks/official_analysis/tissue_area.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parent / "validation"))

RUN = Path(r"D:\Slidescanner extracted\cells_analysis\2026-10-28_090440 re-scored")
OUT = Path(r"D:\Slidescanner extracted\run_analysis_20261028") / "section_tissue_area.csv"
#: Resolution step of the tissue mask, as classical_vs_cellpose.tissue_mask's default.
STEP = 8
WORKERS = 4


def small_tissue(nuclear: np.ndarray, step: int = STEP) -> np.ndarray:
    """classical_vs_cellpose.tissue_mask, kept at 1/*step* (its last step only scales it up)."""
    from scipy import ndimage
    from skimage.filters import threshold_otsu

    small = np.asarray(nuclear[::step, ::step], dtype=np.float32)
    mask = small > threshold_otsu(small)
    mask = ndimage.binary_closing(mask, iterations=3)
    mask = ndimage.binary_fill_holes(mask)
    return ndimage.binary_dilation(mask, iterations=2)


def run_one(name: str) -> dict:
    """The section's tissue area (mm²) from its nuclear stain."""
    import classical_vs_cellpose as cv
    from microscopy_viewer import junction as jn
    from microscopy_viewer import junction_analysis as ja
    from microscopy_viewer import loaders

    logging.getLogger("tifffile").setLevel(logging.ERROR)
    cv.below_normal_priority()
    started = time.perf_counter()
    info = json.loads((RUN / "cells" / f"{name}.json").read_text(encoding="utf-8"))
    path = Path(info["file"])
    specs = loaders.load_path(path)
    try:
        names = [str(spec.channel_name or spec.name) for spec in specs]
        spec = specs[jn.pick_channel(names, "", jn.NUCLEAR_HINTS)]
        voxel = float(spec.scale[-1])
        first = spec.data[0] if spec.multiscale else spec.data
        whole = (slice(0, int(first.shape[-2])), slice(0, int(first.shape[-1])))
        nuclear = ja._plane(spec, whole)
    finally:
        loaders.release(path)
    mask = small_tissue(nuclear)
    # The full-size mask is the small one repeated STEP × STEP times, cut to the plane: count it that way.
    rows = np.minimum(STEP, nuclear.shape[0] - np.arange(mask.shape[0]) * STEP)
    cols = np.minimum(STEP, nuclear.shape[1] - np.arange(mask.shape[1]) * STEP)
    pixels = float(rows @ mask.astype(np.int64) @ cols)
    return {"name": name, "tissue area (mm²)": pixels * voxel ** 2 / 1e6, "voxel (µm)": voxel,
            "seconds": round(time.perf_counter() - started, 1)}


def main() -> int:
    names = sorted(p.stem for p in (RUN / "cells").glob("*.json"))
    done = pd.read_csv(OUT) if OUT.exists() else pd.DataFrame(columns=["name"])
    todo = [n for n in names if n not in set(done["name"])]
    print(f"{len(names)} sections, {len(todo)} to do", flush=True)
    rows = done.to_dict("records")
    with ProcessPoolExecutor(WORKERS) as pool:
        futures = {pool.submit(run_one, n): n for n in todo}
        for k, future in enumerate(as_completed(futures), start=1):
            try:
                rows.append(future.result())
            except Exception as exc:  # one bad file must not stop the rest
                print(f"FAILED {futures[future]}: {exc}", flush=True)
                continue
            if k % 25 == 0 or k == len(todo):
                pd.DataFrame(rows).to_csv(OUT, index=False)
                print(f"[{k}/{len(todo)}] {rows[-1]['name']}: {rows[-1]['tissue area (mm²)']:.2f} mm², "
                      f"{rows[-1]['seconds']} s", flush=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
