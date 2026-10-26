"""Every cell of every annotated sample, measured every way the counts are compared.

Used by the count notebooks (through ``count_common.py``); run on its own to redo it (Cellpose on the
GPU, a few minutes a sample)::

    python notebooks/validation/count_cells.py            # every annotated sample not done yet
    python notebooks/validation/count_cells.py 1          # just the first, to time it

Each sample is segmented **once**, on the nuclear stain, over the whole section
in tiles (as :func:`junction_analysis.sample_cell_values` does), and each cell is
measured in every scored channel:

* over its nucleus grown by 0, 3 (the analysis default) and 5 µm (the seeded
  watershed of :func:`segmentation.expand_labels`);
* in the channel as it is, and with its local median taken away (45 µm, as the
  junction finder's ``cell_prominence``; computed at 1/8 resolution and scaled
  back up, which on a 300-megapixel scan is the difference between seconds and
  an hour);
* as the mean, the median and the 95th percentile of those pixels;
* and over a **ring** round it — from 3 to 6 µm out, the seeded watershed
  again, so a ring never takes a neighbour's pixels — the cell's own local
  background, for contrast rules (cell over ring).

What is kept, per sample, in ``junction_eval/cells/<name>.npz``: each cell's
centroid (full-resolution pixels), nucleus area, and one column per
``channel|expand|filter|statistic``. Which band a cell is in, where it sits
against each annotator's line, and every threshold rule are worked out in the
notebook from these — so a new annotation needs no new segmentation.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import junction_tuning as jt  # noqa: E402

from microscopy_viewer import junction as jn  # noqa: E402
from microscopy_viewer import junction_analysis as ja  # noqa: E402
from microscopy_viewer import loaders  # noqa: E402
from microscopy_viewer import segmentation as sg  # noqa: E402

OUT = jt.OUT / "cells"
EXPANDS_UM = (0.0, 3.0, 5.0)
FILTERS = ("raw", "median")
STATISTICS = ("mean", "median", "P95")
#: The ring a cell's local background is read over: from this far out to that far, in µm.
RING_UM = (3.0, 6.0)
#: Files written before the ring and P95 existed are measured again.
VERSION = 2
#: The local median taken away by the "median" filter, in µm, and the step it is computed at.
MEDIAN_UM = jn.MEDIAN_UM
MEDIAN_STEP = 8


def column(channel: str, expand: float, filtered: str, statistic: str) -> str:
    return f"{channel}|{expand:g}|{filtered}|{statistic}"


def ring_column(channel: str, filtered: str, statistic: str) -> str:
    return f"{channel}|ring|{filtered}|{statistic}"


def less_local_median(plane: np.ndarray, um: float) -> np.ndarray:
    """The plane less its local median over :data:`MEDIAN_UM`.

    Not clipped at 0, unlike the finder's ``cell_prominence``: clipped, most
    cells' median is 0, their spread too, and every cell above 0 is "positive".
    """
    from scipy import ndimage

    coarse = np.asarray(plane[::MEDIAN_STEP, ::MEDIAN_STEP], dtype=np.float32)
    size = max(3, int(round(MEDIAN_UM / (um * MEDIAN_STEP))) | 1)
    local = ndimage.median_filter(coarse, size)
    zoom = (plane.shape[0] / local.shape[0], plane.shape[1] / local.shape[1])
    local = ndimage.zoom(local, zoom, order=1)[: plane.shape[0], : plane.shape[1]]
    return np.asarray(plane, dtype=np.float32) - local


def measure_sample(path: str) -> dict[str, np.ndarray]:
    from scipy import ndimage
    from skimage.filters import threshold_otsu

    target = Path(path)
    specs = loaders.load_path(target)
    try:
        names = [str(spec.channel_name or spec.name) for spec in specs]
        nuclear = jn.pick_channel(names, "", jn.NUCLEAR_HINTS)
        first = specs[0].data[0] if specs[0].multiscale else specs[0].data
        height, width = (int(n) for n in first.shape[-2:])
        voxel = tuple(float(v) for v in specs[0].scale)[-2:]
        whole = (slice(0, height), slice(0, width))
        planes = {name: ja._plane(spec, whole) for name, spec in zip(names, specs)}
    finally:
        loaders.release(target)
    stain = planes[names[nuclear]]
    scored = [name for k, name in enumerate(names) if k != nuclear]
    images = {(name, "raw"): planes[name] for name in scored}
    images.update({(name, "median"): less_local_median(planes[name], voxel[0]) for name in scored})

    coarse = np.asarray(stain[::8, ::8], dtype=np.float64)
    split = float(threshold_otsu(coarse)) if float(coarse.max()) > float(coarse.min()) else float("-inf")
    tiles = [(slice(y, min(y + ja.TILE_PX, height)), slice(x, min(x + ja.TILE_PX, width)))
             for y in range(0, height, ja.TILE_PX) for x in range(0, width, ja.TILE_PX)]
    tiles = [t for t in tiles if float((stain[t][::8, ::8] > split).mean()) >= ja.TILE_MIN_TISSUE]

    parts: dict[str, list[np.ndarray]] = {}

    def add(key, values):
        parts.setdefault(key, []).append(np.asarray(values, dtype=np.float32))

    for index, tile in enumerate(tiles, start=1):
        print(f"    tile {index}/{len(tiles)}", end="\r", flush=True)
        nuclei = np.asarray(ja._segment(stain[tile], voxel, None, None))
        ids = np.unique(nuclei[nuclei > 0]).astype(np.int64)
        if not ids.size:
            continue
        centres = np.asarray(ndimage.center_of_mass(nuclei > 0, nuclei, ids)).reshape(-1, 2)
        add("y", centres[:, 0] + tile[0].start)
        add("x", centres[:, 1] + tile[1].start)
        add("area_um2", np.asarray(ndimage.sum(nuclei > 0, nuclei, ids)) * voxel[0] * voxel[1])
        grown = {}
        for expand in EXPANDS_UM:
            regions = sg.expand_labels(nuclei, voxel, expand) if expand > 0 else nuclei
            grown[expand] = regions
            for (name, filtered), image in images.items():
                stats = ja.label_stats(regions, image[tile], ids)
                for statistic in STATISTICS:
                    add(column(name, expand, filtered, statistic), stats[statistic])
        inner = grown.get(RING_UM[0])
        inner = inner if inner is not None else sg.expand_labels(nuclei, voxel, RING_UM[0])
        outer = sg.expand_labels(nuclei, voxel, RING_UM[1])
        ring = np.where(inner == 0, outer, 0)
        for (name, filtered), image in images.items():
            stats = ja.label_stats(ring, image[tile], ids)
            for statistic in ("mean", "median"):
                add(ring_column(name, filtered, statistic), stats[statistic])
    sg.release_gpu_memory()
    out = {key: np.concatenate(values) for key, values in parts.items()}
    out["voxel_yx"] = np.asarray(voxel)
    out["shape_yx"] = np.asarray((height, width))
    out["channels"] = np.asarray(scored)
    out["version"] = np.asarray(VERSION)
    return out


def main(limit: int | None = None) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    samples = jt.annotated_samples()
    def done(name):
        target = OUT / f"{name}.npz"
        if not target.exists():
            return False
        with np.load(target) as data:
            return "version" in data.files and int(data["version"]) >= VERSION
    todo = [row for row in samples.itertuples() if not done(row.name)][:limit]
    for index, row in enumerate(todo, start=1):
        started = time.perf_counter()
        print(f"{index}/{len(todo)} {row.name}", flush=True)
        try:
            cells = measure_sample(row.path)
        except Exception as exc:  # one bad sample should not stop the run
            print(f"    FAILED: {exc}", flush=True)
            continue
        np.savez_compressed(OUT / f"{row.name}.npz", **cells)
        print(f"    {len(cells['y']):,} cells in {time.perf_counter() - started:.0f} s", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
