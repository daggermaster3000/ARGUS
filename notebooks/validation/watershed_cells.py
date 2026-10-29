"""Whole cells from a nuclei-seeded watershed on the median-filtered AF488, and their ring contrast.

Used by ``05_watershed_cells.ipynb``; run on its own to (re)compute the cache::

    python notebooks/validation/watershed_cells.py        # every sample not done yet
    python notebooks/validation/watershed_cells.py 2      # just the first two, to time it

The current rule measures each cell over its nucleus grown by 3 µm, and calls it
AF488-positive at **median + 7 SD (median filter)**: the cell's mean of AF488 less
its local median, above the median + 7 robust SDs of the region's cells. A labelled
cell's body and processes often lie outside those 3 µm. Here each cell is grown
over the label itself instead:

1. AF488 less its **local median** over 45 µm (as the analysis does:
   ``junction_analysis.coarse_median`` / ``median_at``), smoothed by
   :data:`SMOOTH_UM`;
2. **foreground**: the pixels :data:`FG_K` robust SDs above the tissue's centre
   of that, plus every nucleus;
3. **seeded watershed**: the Cellpose nuclei are the seeds, the landscape is the
   foreground's **distance transform** (deepest in the middle of a labelled
   body), so neighbouring cells split along the narrowings between them; no
   pixel further than :data:`MAX_UM` from a nucleus. A cell with no foreground
   round it keeps its nucleus only. This is the cell's **territory**;
4. its **ring**: the pixels 3–6 µm out from the territory that are more than
   3 µm from every territory (tissue, not cells); a cell with none takes the
   median ring of its 8 nearest cells within 50 µm (``ring_background``).

Per cell, over the territory: area, AF488 mean (raw), AF488 mean (median filter),
ring mean and **ring contrast** (territory mean ÷ ring mean). The notebook calls
cells positive from those — *watershed ring contrast ≥ r* and *watershed median +
k SD (median filter)* — and sets them against the nucleus + 3 µm rules.

Samples: the 39 slide scans of ``classical_vs_cellpose`` (nuclei from their
cells-analysis run) → ``junction_eval/watershed/<name>/``; and the 12 samples
Quillan clicked, segmented here with the Cells panel's defaults (GPU, ~40 s each),
with their nucleus + 3 µm figures too → ``junction_eval/watershed_clicked/<name>/``.
Each folder: ``cells.parquet``, ``territories.tif``, ``summary.json``.
"""

from __future__ import annotations

import os

#: Threads numpy / scipy may use per process.
THREADS = 2
for _variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_variable, str(THREADS))

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import classical_vs_cellpose as cv  # noqa: E402
import junction_tuning as jt  # noqa: E402

from microscopy_viewer import junction as jn  # noqa: E402
from microscopy_viewer import junction_analysis as ja  # noqa: E402
from microscopy_viewer import loaders  # noqa: E402

OUT = jt.OUT / "watershed"
OUT_CLICKED = jt.OUT / "watershed_clicked"
VERSION = 1
LABEL_HINT = "488"
#: Smoothing of the median-filtered AF488 before the foreground is cut, in µm.
SMOOTH_UM = 0.5
#: Robust SDs above the tissue's centre a pixel needs to be foreground.
FG_K = 3.0
#: Furthest a territory reaches from its nucleus, in µm.
MAX_UM = 20.0
#: The ring round a territory, in µm out from it.
RING_UM = (3.0, 6.0)
#: Overlap of the tiles the watershed runs in, in µm: more than a territory and its ring reach.
OVERLAP_UM = 45.0
#: Processes for the slide scans (CPU).
WORKERS = 4


def below_normal_priority() -> None:
    cv.below_normal_priority()


def read_planes(path: Path):
    """AF488 and the nuclear stain over the whole section, and the pixel size."""
    specs = loaders.load_path(path)
    try:
        names = [str(spec.channel_name or spec.name) for spec in specs]
        label = next(i for i, name in enumerate(names) if LABEL_HINT in name)
        nuclear = jn.pick_channel(names, "", jn.NUCLEAR_HINTS)
        first = specs[0].data[0] if specs[0].multiscale else specs[0].data
        whole = (slice(0, int(first.shape[-2])), slice(0, int(first.shape[-1])))
        return (ja._plane(specs[label], whole).astype(np.float32), ja._plane(specs[nuclear], whole),
                float(specs[0].scale[-1]), names[label])
    finally:
        loaders.release(path)


def median_filtered(af488: np.ndarray, voxel: float) -> np.ndarray:
    """AF488 less its local median, as the analysis takes it (in strips, to spare memory)."""
    shape = af488.shape
    coarse = ja.coarse_median(af488[::ja.MEDIAN_STEP, ::ja.MEDIAN_STEP], voxel)
    out = np.empty(shape, dtype=np.float32)
    for y in range(0, shape[0], ja.TILE_PX):
        window = (slice(y, min(y + ja.TILE_PX, shape[0])), slice(0, shape[1]))
        out[window] = af488[window] - ja.median_at(coarse, window, shape)
    return out


def foreground(filtered: np.ndarray, nuclear: np.ndarray, voxel: float) -> tuple[np.ndarray, dict]:
    """Pixels clearly labelled: the smoothed median-filtered AF488 :data:`FG_K` SDs over the tissue's centre."""
    from scipy import ndimage

    step = 8
    tissue = ja.tissue_of(np.asarray(nuclear[::step, ::step]))
    tissue = np.repeat(np.repeat(tissue, step, axis=0), step, axis=1)[: filtered.shape[0], : filtered.shape[1]]
    smooth = ndimage.gaussian_filter(filtered, SMOOTH_UM / voxel)
    centre, spread = ja.robust_centre_and_spread(smooth[tissue][::20])
    cut = centre + FG_K * spread
    return (smooth > cut) & tissue, {"fg_centre": centre, "fg_sd": spread, "fg_cut": cut,
                                     "tissue_mm2": float(tissue.sum()) * voxel * voxel / 1e6}


def tiles(shape, overlap: int):
    for y in range(0, shape[0], ja.TILE_PX):
        for x in range(0, shape[1], ja.TILE_PX):
            core = (slice(y, min(y + ja.TILE_PX, shape[0])), slice(x, min(x + ja.TILE_PX, shape[1])))
            window = (slice(max(y - overlap, 0), min(core[0].stop + overlap, shape[0])),
                      slice(max(x - overlap, 0), min(core[1].stop + overlap, shape[1])))
            yield core, window


def territories(nuclei: np.ndarray, fg: np.ndarray, centres_px: np.ndarray, labels: np.ndarray,
                voxel: float) -> np.ndarray:
    """Each nucleus grown over the foreground by a distance-transform watershed (see the module)."""
    from scipy import ndimage
    from skimage.segmentation import watershed

    out = np.zeros(nuclei.shape, dtype=np.int32)
    reach = MAX_UM / voxel
    for core, window in tiles(nuclei.shape, int(round(OVERLAP_UM / voxel))):
        seeds = nuclei[window]
        if not seeds.any():
            continue
        cy, cx = centres_px[:, 0], centres_px[:, 1]
        own = labels[(cy >= core[0].start) & (cy < core[0].stop) & (cx >= core[1].start) & (cx < core[1].stop)]
        if not own.size:
            continue
        near = ndimage.distance_transform_edt(seeds == 0) <= reach
        mask = (fg[window] & near) | (seeds > 0)
        landscape = ndimage.distance_transform_edt(mask)
        grown = watershed(-landscape, markers=seeds, mask=mask)
        keep = np.isin(grown, own)
        view = out[window]
        view[keep & (view == 0)] = grown[keep & (view == 0)]
    return out


def measure(territory: np.ndarray, af488: np.ndarray, filtered: np.ndarray, labels: np.ndarray,
            centres_um: np.ndarray, voxel: float) -> dict[str, np.ndarray]:
    """Per cell, over its territory: area, raw and median-filtered means, ring mean, ring contrast."""
    from scipy import ndimage
    from skimage.segmentation import expand_labels

    top = int(max(int(territory.max()), int(labels.max()) if labels.size else 0))
    sums = {key: np.zeros(top + 1) for key in ("n", "raw", "filtered", "ring n", "ring")}
    inner_px, outer_px = RING_UM[0] / voxel, RING_UM[1] / voxel
    for core, window in tiles(territory.shape, int(round((RING_UM[1] + 2) / voxel))):
        here = territory[window]
        if not here.any():
            continue
        owned = here[(slice(core[0].start - window[0].start, core[0].stop - window[0].start),
                      slice(core[1].start - window[1].start, core[1].stop - window[1].start))]
        flat = owned.reshape(-1).astype(np.int64)
        inside = (slice(core[0].start, core[0].stop), slice(core[1].start, core[1].stop))
        sums["n"] += np.bincount(flat, minlength=top + 1)
        sums["raw"] += np.bincount(flat, weights=af488[inside].reshape(-1), minlength=top + 1)
        sums["filtered"] += np.bincount(flat, weights=filtered[inside].reshape(-1), minlength=top + 1)
        # The ring: within 6 µm of a territory, more than 3 µm from every territory.
        outer = expand_labels(here, outer_px)
        inner = expand_labels(here, inner_px)
        ring = np.where(inner == 0, outer, 0)
        ring_core = ring[(slice(core[0].start - window[0].start, core[0].stop - window[0].start),
                          slice(core[1].start - window[1].start, core[1].stop - window[1].start))]
        flat = ring_core.reshape(-1).astype(np.int64)
        sums["ring n"] += np.bincount(flat, minlength=top + 1)
        sums["ring"] += np.bincount(flat, weights=af488[inside].reshape(-1), minlength=top + 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        n = sums["n"][labels]
        raw = np.where(n > 0, sums["raw"][labels] / n, np.nan)
        filt = np.where(n > 0, sums["filtered"][labels] / n, np.nan)
        rn = sums["ring n"][labels]
        ring = np.where(rn > 0, sums["ring"][labels] / rn, np.nan)
    background = ja.ring_background(centres_um[:, 0], centres_um[:, 1], ring)
    return {"WS area (µm²)": n * voxel * voxel, "WS AF488 mean": raw, "WS AF488 mean (median filter)": filt,
            "WS ring mean": ring, "WS no ring": ~(rn > 0), "WS ring contrast": ja.ring_contrast(raw, background)}


def run_one(name: str, path: Path, nuclei: np.ndarray, table, folder: Path, extra: dict | None = None) -> dict:
    """Territories and their figures for one sample; *table* has ``Label`` and the centroids in µm."""
    import pandas as pd
    import tifffile

    timings = {}
    started = time.perf_counter()
    af488, nuclear, voxel, channel = read_planes(path)
    timings["read"] = time.perf_counter() - started
    filtered = median_filtered(af488, voxel)
    fg, info = foreground(filtered, nuclear, voxel)
    del nuclear
    timings["foreground"] = time.perf_counter() - started - sum(timings.values())
    labels = table["Label"].to_numpy(dtype=np.int64)
    centres_um = table[["Centroid Y (µm)", "Centroid X (µm)"]].to_numpy(dtype=float)
    territory = territories(nuclei, fg, centres_um / voxel, labels, voxel)
    timings["watershed"] = time.perf_counter() - started - sum(timings.values())
    figures = measure(territory, af488, filtered, labels, centres_um, voxel)
    timings["measure"] = time.perf_counter() - started - sum(timings.values())
    cells = table.copy()
    nucleus_area = np.asarray(cells["Nucleus area (µm²)"], dtype=float) if "Nucleus area (µm²)" in cells else np.nan
    for key, values in figures.items():
        cells[key] = values
    cells["WS beyond the nucleus (µm²)"] = cells["WS area (µm²)"] - nucleus_area
    folder.mkdir(parents=True, exist_ok=True)
    cells.to_parquet(folder / "cells.parquet")
    dtype = np.uint16 if int(territory.max()) < 65536 else np.uint32
    tifffile.imwrite(folder / "territories.tif", territory.astype(dtype), compression="zlib")
    summary = {"version": VERSION, "sample": name, "file": str(path), "voxel_um": voxel, "channel": channel,
               "shape": [int(v) for v in territory.shape], "n_cells": int(len(cells)),
               "fg_share_of_tissue": float(fg.sum() / max(info["tissue_mm2"] * 1e6 / voxel / voxel, 1)),
               **info, **(extra or {}), "timings_s": {k: round(v, 1) for k, v in timings.items()}}
    (folder / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary


#: How deep a dip in the stain must be, in µm, to part two bodies (h of the h-maxima), tried in turn.
BODY_H_UM = (0.5, 1.0, 2.0, 3.0)


def stain_depth(fg: np.ndarray, voxel: float) -> np.ndarray:
    """How deep each pixel lies in the AF488 foreground, in µm: its distance transform, nuclei not added.

    The territories were grown with the nuclei in the mask; for telling one body
    from two only the stain counts — in packed tissue the nuclei alone fill the
    mask and every border would look thick.
    """
    from scipy import ndimage

    return (ndimage.distance_transform_edt(fg) * voxel).astype(np.float32)


def bodies(fg: np.ndarray, depth: np.ndarray, territory: np.ndarray, hs=BODY_H_UM):
    """Which labelled body each cell's territory lies in, for each h of :data:`BODY_H_UM`.

    The classic distance-transform watershed, on the stain alone: in each
    connected piece of foreground, the maxima of :func:`stain_depth` at least *h*
    µm higher than the dip between them are the bodies, and the piece is split
    between them by a watershed. A cell belongs to the body holding most of its
    territory. Cells that share a body are one labelled body cut in several by
    the nuclei. Columns ``piece`` (the connected piece: the coarsest grouping)
    and ``body h=<h>``; a cell outside the foreground, or alone in its piece,
    gets an id of its own (negative).
    """
    import pandas as pd
    from scipy import ndimage
    from skimage.morphology import h_maxima
    from skimage.segmentation import watershed

    pieces, _n = ndimage.label(fg)
    inside = (pieces > 0) & (territory > 0)
    pairs = pd.DataFrame({"t": territory[inside], "p": pieces[inside]}).value_counts().reset_index(name="n")
    shared = pairs.groupby("p")["t"].nunique()
    shared = shared[shared >= 2].index.to_numpy()
    boxes = ndimage.find_objects(pieces)
    records = {h: [] for h in hs}
    for piece in shared:
        box = boxes[int(piece) - 1]
        here = pieces[box] == piece
        d = np.where(here, depth[box], 0)
        ter = territory[box]
        sel = here & (ter > 0)
        for h in hs:
            markers, _m = ndimage.label(h_maxima(d, h))
            basins = watershed(-d, markers, mask=here)
            ok = sel & (basins > 0)
            records[h].append(pd.DataFrame({"t": ter[ok], "b": int(piece) * 100_000 + basins[ok].astype(np.int64)}))
    labels = np.unique(territory[territory > 0]).astype(np.int64)
    out = pd.DataFrame({"Label": labels})
    best_piece = pairs.sort_values("n", ascending=False).drop_duplicates("t").set_index("t")["p"]
    out["piece"] = out["Label"].map(best_piece).where(lambda s: s.isin(shared), -out["Label"]).astype(np.int64)
    for h in hs:
        if records[h]:
            counts = pd.concat(records[h]).value_counts().reset_index(name="n")
            best = counts.sort_values("n", ascending=False).drop_duplicates("t").set_index("t")["b"]
        else:
            best = pd.Series(dtype=np.int64)
        out[f"body h={h:g}"] = out["Label"].map(best).fillna(-out["Label"]).astype(np.int64)
    return out


def run_bodies(name: str, path: str, folder: str) -> str:
    """The labelled bodies of one sample's cells → ``bodies.parquet`` beside its cells."""
    import tifffile

    below_normal_priority()
    started = time.perf_counter()
    folder = Path(folder)
    af488, nuclear, voxel, _channel = read_planes(Path(path))
    fg, _info = foreground(median_filtered(af488, voxel), nuclear, voxel)
    del af488, nuclear
    depth = stain_depth(fg, voxel)
    territory = tifffile.imread(folder / "territories.tif").astype(np.int32)
    table = bodies(fg, depth, territory)
    table.to_parquet(folder / "bodies.parquet")
    merged = int((table["body h=1"] > 0).sum())
    return f"{name}: {merged} of {len(table)} cells share a piece of stain with another, {time.perf_counter() - started:.0f} s"


def done(folder: Path) -> bool:
    target = folder / "summary.json"
    try:
        return target.is_file() and json.loads(target.read_text(encoding="utf-8")).get("version") == VERSION
    except Exception:
        return False


def run_scan(name: str) -> str:
    """One of the 39 slide scans: nuclei from its cells-analysis run, cells from the classical cache."""
    import pandas as pd

    below_normal_priority()
    sample = ja.segmented_samples(cv.DATA)[name]
    keep = ["Label", "Centroid Y (µm)", "Centroid X (µm)", "Nucleus area (µm²)", ja.IN_REGION_COLUMN,
            ja.LINE_COLUMN, ja.EDGE_COLUMN, "AF488 mean", "AF488 ring contrast", "AF488 mean (median filter)",
            "median + 3 SD", "ring ≥ 1.5", "median + 7 SD (median filter)", "cover kmad5"]
    table = pd.read_parquet(cv.OUT / name / "cells.parquet")
    table = table[[c for c in keep if c in table.columns]]
    nuclei = ja.read_labels(sample.file("nuclei_file"))
    started = time.perf_counter()
    summary = run_one(name, sample.path, nuclei, table, OUT / name)
    return f"{name}: {time.perf_counter() - started:.0f} s {summary['timings_s']}"


def segment_clicked(name: str, path: Path):
    """A clicked sample segmented as the Cells panel does: nuclei, and every cell's nucleus + 3 µm figures."""
    specs = loaders.load_path(path)
    try:
        names = [str(spec.channel_name or spec.name) for spec in specs]
        nuclear = jn.pick_channel(names, "", jn.NUCLEAR_HINTS)
        first = specs[0].data[0] if specs[0].multiscale else specs[0].data
        shape = (int(first.shape[-2]), int(first.shape[-1]))
        voxel = tuple(float(v) for v in specs[0].scale[-2:])
        whole = ja.segment_section(specs, names, nuclear, voxel, ja.CountOptions(), crop=(slice(0, shape[0]), slice(0, shape[1])))
    finally:
        loaders.release(path)
    channel = next(n for n in names if LABEL_HINT in n)
    table = whole.table.rename(columns={f"{channel} {ja.FILTERED}": "AF488 mean (median filter)",
                                        f"{channel} ring contrast": "AF488 ring contrast", f"{channel} mean": "AF488 mean"})
    keep = ["Label", "Centroid Y (µm)", "Centroid X (µm)", "Nucleus area (µm²)", "AF488 mean",
            "AF488 mean (median filter)", "AF488 ring contrast"]
    return whole.labels, table[keep]


def run_clicked_ws(name: str, path: str) -> str:
    import pandas as pd

    below_normal_priority()
    folder = OUT_CLICKED / name
    nuclei = ja.read_labels(folder / "nuclei.tif")
    table = pd.read_parquet(folder / "nucleus_cells.parquet")
    started = time.perf_counter()
    summary = run_one(name, Path(path), nuclei, table, folder)
    return f"{name} (clicked): {time.perf_counter() - started:.0f} s {summary['timings_s']}"


def scans() -> list[str]:
    """The 39 slide scans of ``classical_vs_cellpose``."""
    names = []
    for folder in sorted(cv.OUT.iterdir()):
        summary = folder / "summary.json"
        if summary.is_file() and json.loads(summary.read_text()).get("version") == cv.VERSION:
            names.append(folder.name)
    return names


def main(argv: list[str]) -> int:
    from concurrent.futures import ProcessPoolExecutor, as_completed

    import tifffile

    if argv and argv[0] == "bodies":
        # The labelled bodies of every sample done, for merging cells cut in several.
        saved = ja.segmented_samples(cv.DATA)
        jobs = [(n, str(saved[n].path), str(OUT / n))
                for n in scans() if done(OUT / n) and not (OUT / n / "bodies.parquet").is_file()]
        jobs += [(n, str(p), str(OUT_CLICKED / n))
                 for n, p in cv.clicked_samples()
                 if done(OUT_CLICKED / n) and not (OUT_CLICKED / n / "bodies.parquet").is_file()]
        print(f"{len(jobs)} sample(s) to find the bodies of", flush=True)
        with ProcessPoolExecutor(max_workers=WORKERS) as pool:
            for future in as_completed([pool.submit(run_bodies, *job) for job in jobs]):
                try:
                    print(future.result(), flush=True)
                except Exception as exc:
                    print(f"FAILED {exc!r}", flush=True)
        return 0
    todo = [name for name in scans() if not done(OUT / name)]
    if argv:
        todo = todo[: int(argv[0])]
    clicked = [(n, p) for n, p in cv.clicked_samples() if not done(OUT_CLICKED / n)]
    if argv:
        clicked = clicked[: int(argv[0])]
    print(f"{len(todo)} scan(s) -> {OUT}; {len(clicked)} clicked -> {OUT_CLICKED}", flush=True)
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(run_scan, name) for name in todo]
        # The clicked samples: segmented here, on the GPU, one after the other; grown in the pool.
        for name, path in clicked:
            folder = OUT_CLICKED / name
            folder.mkdir(parents=True, exist_ok=True)
            if not (folder / "nuclei.tif").is_file():
                started = time.perf_counter()
                nuclei, table = segment_clicked(name, path)
                dtype = np.uint16 if int(nuclei.max()) < 65536 else np.uint32
                tifffile.imwrite(folder / "nuclei.tif", nuclei.astype(dtype), compression="zlib")
                table.to_parquet(folder / "nucleus_cells.parquet")
                print(f"{name}: segmented, {len(table)} cells, {time.perf_counter() - started:.0f} s", flush=True)
            futures.append(pool.submit(run_clicked_ws, name, str(path)))
        for future in as_completed(futures):
            try:
                print(future.result(), flush=True)
            except Exception as exc:
                print(f"FAILED {exc!r}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
