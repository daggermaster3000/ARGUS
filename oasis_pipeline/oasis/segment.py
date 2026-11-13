"""Optional: segment every section again with Cellpose (GPU), as the shipped segmentation was made.

The shipped segmentation (``data/segmentation/``) is what the paper used; this re-makes it from the raw scans so that
step can be checked too. Settings of the run (``data/segmentation/run_README.txt``): Cellpose 4, model ``cpsam_v2``
(downloaded by Cellpose on first use), flow threshold 0.4, cell probability 0, minimum size 15 px, normalised, no
diameter given; the nuclear stain (maximum over z) of the whole section in 2048 px tiles with 96 px of their
neighbours round them, tiles with under 2 % tissue skipped, each tile keeping the nuclei whose centre lies in it;
nuclei grown by 6 µm (each into its share of the gap; ``expand_labels``) to approximate cells, and each cell measured
over its grown region in every channel (mean, median, SD, CV, P95, max, integrated, and the mean of the channel less
its 45 µm local median).

Each cell is then placed as in the shipped tables: in the junction region (centre in the region), its signed distance
to the junction line, beside the line or past its ends, its distance to the tissue edge. *AF488 positive*: the cell's
median-filtered AF488 mean above the median + 7 robust SD of the junction region's cells (the run's rule).

Cellpose on a GPU is not bit-for-bit repeatable, so the cells differ slightly from the shipped ones; the count itself
(the AF488 mask) does not use the segmentation, only the secondary outcomes and the section browser's per-nucleus
methods do. Writes ``rerun/segmentation/`` (cells/<s>.csv, cells/<s>.json, labels/<s> nuclei.tif, grown.tif) and,
when every section is done, ``COMPLETE``: the later steps then use it instead of the shipped one.

About 2–3 min a section on an RTX-class GPU (~1 day for 614).
"""
from __future__ import annotations

import json
import time

import numpy as np

from . import cells as ja
from . import junction as jn
from . import paths, scan

MODEL = "cpsam_v2"
FLOW_THRESHOLD = 0.4
CELLPROB_THRESHOLD = 0.0
MIN_SIZE = 15
EXPAND_UM = 6.0                   #: nuclei grown by this to approximate cells
TILE_PX, TILE_OVERLAP_PX, TILE_MIN_TISSUE = 2048, 96, 0.02
POSITIVE_K = 7.0                  #: AF488 positive: median-filtered mean > median + k robust SD of the region's cells
CHANNEL = "AF488"

_MODEL = None


def model():
    global _MODEL
    if _MODEL is None:
        from cellpose import models

        try:
            import torch

            gpu = torch.cuda.is_available()
        except Exception:
            gpu = False
        _MODEL = models.CellposeModel(gpu=gpu, pretrained_model=MODEL)
    return _MODEL


def expand_labels(masks: np.ndarray, voxel: tuple[float, float], distance_um: float) -> np.ndarray:
    """Grow every object outwards by up to *distance_um* (a seeded watershed over the distance to the nearest object)."""
    from scipy import ndimage
    from skimage.segmentation import watershed

    if distance_um <= 0 or not masks.any():
        return masks.copy()
    distance = ndimage.distance_transform_edt(masks == 0, sampling=voxel).astype(np.float32, copy=False)
    return watershed(distance, markers=masks, mask=distance <= float(distance_um)).astype(masks.dtype, copy=False)


def segment_tile(image: np.ndarray) -> np.ndarray:
    out = model().eval(image, batch_size=8, normalize=True, flow_threshold=FLOW_THRESHOLD,
                       cellprob_threshold=CELLPROB_THRESHOLD, min_size=MIN_SIZE, do_3D=False)
    return np.asarray(out[0]).astype(np.int32, copy=False)


def run_one(name: str) -> str:
    import pandas as pd
    import tifffile
    from scipy import ndimage
    from skimage.filters import threshold_otsu

    started = time.perf_counter()
    info = paths.section_info(name)
    specs = scan.load_path(paths.scan_path(name))
    names = [str(s.channel_name or s.name) for s in specs]
    nuclear = jn.pick_channel(names, "", jn.NUCLEAR_HINTS)
    voxel = (float(specs[0].scale[-2]), float(specs[0].scale[-1]))
    height, width = specs[0].shape_yx
    planes = {n: scan.plane(s) for n, s in zip(names, specs)}
    stain = planes[names[nuclear]]
    medians = {n: ja.coarse_median(p[::ja.MEDIAN_STEP, ::ja.MEDIAN_STEP], voxel[1]) for n, p in planes.items()}
    filtered = {n: p.astype(np.float32) - ja.median_at(medians[n], (slice(0, height), slice(0, width)), (height, width))
                for n, p in planes.items()}
    coarse = np.asarray(stain[::8, ::8], dtype=np.float64)
    split = float(threshold_otsu(coarse)) if float(coarse.max()) > float(coarse.min()) else float("-inf")
    tiles = [(slice(y, min(y + TILE_PX, height)), slice(x, min(x + TILE_PX, width)))
             for y in range(0, height, TILE_PX) for x in range(0, width, TILE_PX)]
    tiles = [t for t in tiles if float((stain[t][::8, ::8] > split).mean()) >= TILE_MIN_TISSUE]
    nuclei_full = np.zeros((height, width), np.int32)
    grown_full = np.zeros((height, width), np.int32)
    parts, found = {}, 0
    for tile in tiles:
        window = (slice(max(tile[0].start - TILE_OVERLAP_PX, 0), min(tile[0].stop + TILE_OVERLAP_PX, height)),
                  slice(max(tile[1].start - TILE_OVERLAP_PX, 0), min(tile[1].stop + TILE_OVERLAP_PX, width)))
        nuclei = segment_tile(stain[window])
        ids = np.unique(nuclei[nuclei > 0]).astype(np.int64)
        if not ids.size:
            continue
        centres = np.asarray(ndimage.center_of_mass(nuclei > 0, nuclei, ids)).reshape(-1, 2)
        cy, cx = centres[:, 0] + window[0].start, centres[:, 1] + window[1].start
        own = (cy >= tile[0].start) & (cy < tile[0].stop) & (cx >= tile[1].start) & (cx < tile[1].stop)
        if not own.any():
            continue
        ids, cy, cx = ids[own], cy[own], cx[own]
        regions = expand_labels(nuclei, voxel, EXPAND_UM)
        area = np.asarray(ndimage.sum(nuclei > 0, nuclei, ids)) * voxel[0] * voxel[1]
        labels = np.arange(found + 1, found + 1 + ids.size, dtype=np.int64)
        figures = {"Label": labels, "Centroid Y (µm)": cy * voxel[0], "Centroid X (µm)": cx * voxel[1],
                   "Nucleus area (µm²)": area, "Equivalent diameter (µm)": 2.0 * np.sqrt(area / np.pi)}
        for n in names:
            stats = ja.label_stats(regions, planes[n][window], ids)
            for figure in ja.CELL_METRICS:
                figures[f"{n} {figure}"] = stats[figure]
            figures[f"{n} {ja.FILTERED}"] = np.asarray(ndimage.mean(filtered[n][window], regions, ids), dtype=np.float64)
        for key, values in figures.items():
            parts.setdefault(key, []).append(np.asarray(values))
        lut = np.zeros(int(nuclei.max()) + 1, np.int32)
        lut[ids] = labels
        for full, local in ((nuclei_full, nuclei), (grown_full, regions)):
            mapped = lut[local]
            target = full[window]
            target[mapped > 0] = mapped[mapped > 0]
        found += int(ids.size)
    table = pd.DataFrame({k: np.concatenate(v) for k, v in parts.items()})
    band = ja.read_labels(paths.region_tif(name)) > 0
    rows = np.clip(np.round(table["Centroid Y (µm)"].to_numpy() / voxel[0]).astype(int), 0, height - 1)
    cols = np.clip(np.round(table["Centroid X (µm)"].to_numpy() / voxel[1]).astype(int), 0, width - 1)
    table[ja.IN_REGION_COLUMN] = band[rows, cols]
    region = table[ja.IN_REGION_COLUMN].to_numpy(bool)
    values = table[f"{CHANNEL} {ja.FILTERED}"].to_numpy(float)
    centre, spread = ja.robust_centre_and_spread(values[region])
    table[f"{CHANNEL} positive"] = values > centre + POSITIVE_K * spread
    line = info.get("line")
    line = np.asarray(json.loads(line) if isinstance(line, str) else line, float) if line else None
    ja.place_cells(table, band, line, voxel, jn.build_overview(specs))
    folder = paths.RERUN_SEGMENTATION
    (folder / "cells").mkdir(parents=True, exist_ok=True)
    (folder / "labels").mkdir(parents=True, exist_ok=True)
    table.insert(0, "Sample", name)
    table.to_csv(folder / "cells" / f"{name}.csv", index=False, encoding="utf-8-sig")
    tifffile.imwrite(folder / "labels" / f"{name} nuclei.tif", nuclei_full, compression="zlib")
    tifffile.imwrite(folder / "labels" / f"{name} grown.tif", grown_full, compression="zlib")
    record = {"n_cells_section": int(len(table)), "model": MODEL, "seconds": round(time.perf_counter() - started, 1)}
    (folder / "cells" / f"{name}.json").write_text(json.dumps(record), encoding="utf-8")
    return f"{name}: {len(table)} cells, {record['seconds']:.0f} s"


def done(name: str) -> bool:
    return (paths.RERUN_SEGMENTATION / "cells" / f"{name}.json").exists()


def main(names: list[str] | None = None) -> int:
    """Segment every section not done yet (one at a time: the GPU); mark the segmentation complete when all are."""
    every = paths.section_names()
    names = names or every
    todo = [n for n in names if not done(n)]
    print(f"segmentation: {len(names)} sections, {len(todo)} to do -> {paths.RERUN_SEGMENTATION}", flush=True)
    started, failed = time.perf_counter(), 0
    for index, name in enumerate(todo, start=1):
        try:
            line = run_one(name)
        except Exception as exc:
            failed += 1
            line = f"{name}: FAILED {exc!r}"
        elapsed = time.perf_counter() - started
        print(f"[{index}/{len(todo)}] {line} — {elapsed / 60:.0f} min, "
              f"~{elapsed / index * (len(todo) - index) / 60:.0f} min left", flush=True)
    if all(done(n) for n in every):
        (paths.RERUN_SEGMENTATION / "COMPLETE").write_text("every section segmented\n", encoding="utf-8")
    return 1 if failed else 0
