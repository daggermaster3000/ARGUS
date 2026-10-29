"""AF488 segmented directly, by classical thresholds, against Cellpose nuclei scored positive.

Used by ``04_classical_vs_cellpose.ipynb``; run on its own to (re)compute the cache::

    python notebooks/validation/classical_vs_cellpose.py        # every sample not done yet
    python notebooks/validation/classical_vs_cellpose.py 2      # just the first two, to time it
    python notebooks/validation/classical_vs_cellpose.py cached # only bring the samples done up to date

The Cellpose side is not run again: each sample's cells, nuclei and grown
regions are read from the newest cells-analysis run that saved them
(:func:`junction_analysis.segmented_samples`). Each cell is called AF488-positive
by the two rules the analysis offers — **median + 3 SD** of the region's cells
(the default) and **ring contrast ≥ 1.5** (cell over its 3–6 µm ring).

The classical side segments the AF488 channel itself, at full resolution:

1. the channel flattened over Z, smoothed (Gaussian, :data:`SMOOTH_UM`);
2. its **local background** taken away: a Gaussian of :data:`BACKGROUND_UM`,
   computed at 1/4 resolution and scaled back up;
3. thresholded inside the tissue (:data:`METHODS`): *k × robust SD* of the
   background-subtracted tissue for k = 3, 5 and 8 (``kmad3`` …), *triangle*
   (made for a histogram with one big peak and a long tail) and *Otsu*;
4. objects smaller than :data:`MIN_OBJECT_UM2` dropped, small holes filled.

Each cell is also called by **median + 7 SD on the median-filtered channel**
(the rule ``positivity_metrics`` found best; ``count_cells.less_local_median``).
For the samples Quillan clicked (the earlier annotated set), only the masks are
made, in ``junction_eval/classical_clicked/<name>/``; the notebook scores them
and the Cellpose rules against the clicks.

What is kept, per sample, in ``junction_eval/classical/<name>/``:

* ``cells.parquet`` — every Cellpose cell: place, in the region or not,
  distance to the line, AF488 mean and ring contrast, positive by each rule,
  and the share of its grown region each classical mask covers;
* ``objects_<method>.parquet`` — every classical object: area, centroid,
  solidity, eccentricity, long axis, skeleton length, nuclei it touches, in
  the region or not, distance to the line;
* ``<method>.tif`` — the mask (compressed), for pictures;
* ``summary.json`` — thresholds, noise, timings.

The machine may be segmenting with Cellpose meanwhile: this runs on the CPU
only, at below-normal priority, on :data:`THREADS` threads.
"""

from __future__ import annotations

import os

#: Threads numpy / scipy may use: the GPU job running beside this keeps the rest.
THREADS = 4
for _variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_variable, str(THREADS))

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import count_cells as cc  # noqa: E402
import junction_tuning as jt  # noqa: E402

from microscopy_viewer import fusion as fu  # noqa: E402,F401
from microscopy_viewer import junction as jn  # noqa: E402
from microscopy_viewer import junction_analysis as ja  # noqa: E402
from microscopy_viewer import loaders  # noqa: E402

#: Where the samples and their cells-analysis runs are, and where this writes.
DATA = Path(r"D:\Slidescanner extracted")
OUT = jt.OUT / "classical"
#: Bump to recompute what is cached. Version 2 adds median + 7 SD on the
#: median-filtered channel (version-1 caches are upgraded in place).
VERSION = 2
#: Masks of the samples Quillan clicked (from the earlier annotated set, whose
#: Cellpose cells are ``count_cells.py``'s): masks only, scored in the notebook.
OUT_CLICKED = jt.OUT / "classical_clicked"
MANUAL_COUNTS = "Manual counts Quillan"

LABEL_HINT = "488"
SMOOTH_UM = 0.5
BACKGROUND_UM = 20.0
#: Robust SDs above the local background, for the ``kmad<k>`` methods.
K_MADS = (3.0, 5.0, 8.0)
MIN_OBJECT_UM2 = 20.0
MAX_HOLE_UM2 = 10.0
METHODS = (*(f"kmad{k:g}" for k in K_MADS), "triangle", "otsu")
#: A Cellpose cell counts as covered by a classical mask from this share of
#: its grown region on (the notebook shows the other shares too).
COVER = 0.25
#: The two Cellpose rules compared.
RING_RATIO = ja.RING_RATIO
K_SD = 3.0
#: The rule ``positivity_metrics`` found best (21 Oct): median + 7 SD of the
#: region's cells, on the channel less its local median (``count_cells``).
K_SD_FILTERED = 7.0


def below_normal_priority() -> None:
    """Let the Cellpose run beside this have the machine first (Windows)."""
    try:
        import ctypes

        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
    except Exception:
        pass


def tissue_mask(nuclear: np.ndarray, step: int = 8) -> np.ndarray:
    """The section, from the nuclear stain at 1/*step*: Otsu, closed, holes filled, at full size."""
    from scipy import ndimage
    from skimage.filters import threshold_otsu

    small = np.asarray(nuclear[::step, ::step], dtype=np.float32)
    mask = small > threshold_otsu(small)
    mask = ndimage.binary_closing(mask, iterations=3)
    mask = ndimage.binary_fill_holes(mask)
    mask = ndimage.binary_dilation(mask, iterations=2)
    full = np.repeat(np.repeat(mask, step, axis=0), step, axis=1)
    return full[: nuclear.shape[0], : nuclear.shape[1]]


def local_background(image: np.ndarray, sigma_px: float, step: int = 4) -> np.ndarray:
    """A Gaussian of *sigma_px* over *image*, computed at 1/*step* and scaled back up."""
    from scipy import ndimage

    height, width = image.shape
    small = image[: height // step * step, : width // step * step]
    small = small.reshape(height // step, step, width // step, step).mean(axis=(1, 3))
    small = ndimage.gaussian_filter(small, sigma_px / step)
    full = ndimage.zoom(small, step, order=1)
    out = np.empty(image.shape, dtype=np.float32)
    out[: full.shape[0], : full.shape[1]] = full[: height, : width]
    if full.shape[0] < height:
        out[full.shape[0]:, :] = out[full.shape[0] - 1: full.shape[0], :]
    if full.shape[1] < width:
        out[:, full.shape[1]:] = out[:, full.shape[1] - 1: full.shape[1]]
    return out


def classical_masks(af488: np.ndarray, tissue: np.ndarray, voxel: float) -> tuple[dict[str, np.ndarray], dict]:
    """The AF488 masks of every method in :data:`METHODS`, and their thresholds."""
    from scipy import ndimage
    from skimage.filters import threshold_otsu, threshold_triangle
    from skimage.morphology import remove_small_holes, remove_small_objects

    smooth = ndimage.gaussian_filter(af488, SMOOTH_UM / voxel)
    signal = smooth - local_background(smooth, BACKGROUND_UM / voxel)
    del smooth
    values = signal[tissue][::20]
    centre = float(np.median(values))
    spread = float(1.4826 * np.median(np.abs(values - centre)))
    thresholds = {
        **{f"kmad{k:g}": centre + k * spread for k in K_MADS},
        "triangle": float(threshold_triangle(values)),
        "otsu": float(threshold_otsu(values)),
    }
    min_px = int(round(MIN_OBJECT_UM2 / voxel ** 2))
    hole_px = int(round(MAX_HOLE_UM2 / voxel ** 2))
    masks = {}
    import warnings

    for method in METHODS:
        mask = (signal > thresholds[method]) & tissue
        with warnings.catch_warnings():
            # scikit-image 0.26 renames these arguments; the notebook's environment has 0.25.
            warnings.simplefilter("ignore", FutureWarning)
            mask = remove_small_objects(mask, min_px)
            masks[method] = remove_small_holes(mask, hole_px)
    return masks, {"thresholds": thresholds, "background_centre": centre, "noise_sd": spread}


def cellpose_cells(sample: ja.SegmentedSample, voxel: tuple[float, float]):
    """The sample's Cellpose cells, with the two rules' calls."""
    import pandas as pd

    table = pd.read_csv(sample.file("cells_file"))
    keep = ["Label", "Centroid Y (µm)", "Centroid X (µm)", "Nucleus area (µm²)", ja.IN_REGION_COLUMN,
            ja.LINE_COLUMN, ja.EDGE_COLUMN, "AF488 mean", "AF488 ring contrast"]
    table = table[[c for c in keep if c in table.columns]].copy()
    region = table[table[ja.IN_REGION_COLUMN].astype(bool)]
    centre, spread = ja.robust_centre_and_spread(region["AF488 mean"].to_numpy(dtype=float))
    table["median + 3 SD"] = table["AF488 mean"] > centre + K_SD * spread
    table["ring ≥ 1.5"] = table["AF488 ring contrast"].fillna(0) >= RING_RATIO
    return table, {"median_3sd_threshold": centre + K_SD * spread}


def add_median_filtered(cells, af488: np.ndarray, grown: np.ndarray, voxel: float) -> float:
    """Add each cell's AF488 mean on the median-filtered channel and the median + 7 SD call; return the cut."""
    from scipy import ndimage

    filtered = cc.less_local_median(af488, voxel)
    ids = cells["Label"].to_numpy(dtype=np.int64)
    cells["AF488 mean (median filter)"] = np.asarray(ndimage.mean(filtered, grown, ids), dtype=float)
    region = cells[ja.IN_REGION_COLUMN].astype(bool).to_numpy()
    centre, spread = ja.robust_centre_and_spread(cells.loc[region, "AF488 mean (median filter)"].to_numpy(dtype=float))
    cut = centre + K_SD_FILTERED * spread
    cells["median + 7 SD (median filter)"] = cells["AF488 mean (median filter)"] > cut
    return float(cut)


def read_planes(path: Path):
    """AF488 and the nuclear stain, flattened over Z, and the pixel size."""
    specs = loaders.load_path(path)
    try:
        names = [str(spec.channel_name or spec.name) for spec in specs]
        label_index = next(i for i, name in enumerate(names) if LABEL_HINT in name)
        nuclear_index = jn.pick_channel(names, "", jn.NUCLEAR_HINTS)
        voxel = float(specs[0].scale[-1])
        first = specs[0].data[0] if specs[0].multiscale else specs[0].data
        whole = (slice(0, int(first.shape[-2])), slice(0, int(first.shape[-1])))
        return ja._plane(specs[label_index], whole).astype(np.float32), ja._plane(specs[nuclear_index], whole), voxel
    finally:
        loaders.release(path)


def object_table(mask: np.ndarray, nuclei: np.ndarray, voxel: float, band: np.ndarray | None, line):
    """Every object of *mask*: size, shape, skeleton, nuclei touched, place."""
    import pandas as pd
    from scipy import ndimage
    from skimage.measure import label, regionprops_table
    from skimage.morphology import skeletonize

    objects = label(mask, connectivity=2)
    count = int(objects.max())
    if not count:
        return pd.DataFrame(), objects
    props = pd.DataFrame(regionprops_table(objects, properties=(
        "label", "area", "centroid", "solidity", "eccentricity", "major_axis_length")))
    ids = props["label"].to_numpy()
    skeleton = skeletonize(mask)
    length = np.asarray(ndimage.sum(skeleton, objects, ids)) * voxel
    both = (objects > 0) & (nuclei > 0)
    pairs = np.unique(objects[both].astype(np.int64) * (int(nuclei.max()) + 1) + nuclei[both])
    touched = np.bincount(pairs // (int(nuclei.max()) + 1), minlength=count + 1)
    table = pd.DataFrame({
        "Object": ids,
        "Area (µm²)": props["area"].to_numpy() * voxel ** 2,
        "Centroid Y (µm)": props["centroid-0"].to_numpy() * voxel,
        "Centroid X (µm)": props["centroid-1"].to_numpy() * voxel,
        "Solidity": props["solidity"].to_numpy(),
        "Eccentricity": props["eccentricity"].to_numpy(),
        "Long axis (µm)": props["major_axis_length"].to_numpy() * voxel,
        "Skeleton (µm)": length,
        "Nuclei touched": touched[ids],
    })
    if band is not None:
        rows = np.clip(props["centroid-0"].round().astype(int), 0, band.shape[0] - 1)
        cols = np.clip(props["centroid-1"].round().astype(int), 0, band.shape[1] - 1)
        table[ja.IN_REGION_COLUMN] = band[rows, cols]
        if line is not None:
            ja.place_cells(table, band, line, (voxel, voxel), None)
    return table, objects


def run_sample(sample: ja.SegmentedSample) -> dict:
    import pandas as pd
    import tifffile
    from scipy import ndimage

    folder = OUT / sample.name
    folder.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    timings = {}
    path = sample.path
    af488, nuclear, voxel = read_planes(path)
    timings["read"] = time.perf_counter() - started

    t = time.perf_counter()
    tissue = tissue_mask(nuclear)
    del nuclear
    masks, info = classical_masks(af488, tissue, voxel)
    timings["masks"] = time.perf_counter() - t

    t = time.perf_counter()
    nuclei = ja.read_labels(sample.file("nuclei_file"))
    grown = ja.read_labels(sample.file("grown_file"))
    region_file = sample.file("region_file")
    band = ja.read_labels(region_file) > 0 if region_file is not None and region_file.is_file() else None
    line = np.asarray(sample.info["line"], dtype=float) if sample.info.get("line") else None
    cells, rules = cellpose_cells(sample, (voxel, voxel))
    ids = cells["Label"].to_numpy(dtype=np.int64)
    size = np.asarray(ndimage.sum(grown > 0, grown, ids), dtype=float)
    for method, mask in masks.items():
        cells[f"cover {method}"] = np.asarray(ndimage.sum(mask, grown, ids)) / np.where(size > 0, size, np.nan)
    rules["median_7sd_filtered_threshold"] = add_median_filtered(cells, af488, grown, voxel)
    timings["cells"] = time.perf_counter() - t

    t = time.perf_counter()
    outside = {}
    for method, mask in masks.items():
        objects, _labelled = object_table(mask, nuclei, voxel, band, line)
        objects.to_parquet(folder / f"objects_{method}.parquet")
        outside[method] = {
            "mask_area_um2": float(mask.sum()) * voxel ** 2,
            "outside_cells_um2": float((mask & (grown == 0)).sum()) * voxel ** 2,
            "outside_nuclei_um2": float((mask & (nuclei == 0)).sum()) * voxel ** 2,
        }
        tifffile.imwrite(folder / f"{method}.tif", mask.astype(np.uint8), compression="zlib")
    timings["objects"] = time.perf_counter() - t
    cells.to_parquet(folder / "cells.parquet")

    summary = {
        "version": VERSION, "sample": sample.name, "file": str(path), "run": str(sample.run),
        "voxel_um": voxel, "shape": [int(v) for v in af488.shape], "tissue_mm2": float(tissue.sum()) * voxel ** 2 / 1e6,
        **info, **rules, "mask_areas": outside, "n_cells": int(len(cells)),
        "timings_s": {k: round(v, 1) for k, v in timings.items()},
    }
    (folder / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary


def cached_version(name: str) -> int:
    target = OUT / name / "summary.json"
    try:
        return int(json.loads(target.read_text(encoding="utf-8")).get("version", 0)) if target.is_file() else 0
    except Exception:
        return 0


def upgrade(sample: ja.SegmentedSample) -> None:
    """A version-1 cache: add median + 7 SD (median filter) to its cells, nothing else redone."""
    import pandas as pd

    folder = OUT / sample.name
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    cells = pd.read_parquet(folder / "cells.parquet")
    specs = loaders.load_path(sample.path)
    try:
        names = [str(spec.channel_name or spec.name) for spec in specs]
        spec = specs[next(i for i, name in enumerate(names) if LABEL_HINT in name)]
        first = spec.data[0] if spec.multiscale else spec.data
        af488 = ja._plane(spec, (slice(0, int(first.shape[-2])), slice(0, int(first.shape[-1])))).astype(np.float32)
    finally:
        loaders.release(sample.path)
    grown = ja.read_labels(sample.file("grown_file"))
    summary["median_7sd_filtered_threshold"] = add_median_filtered(cells, af488, grown, summary["voxel_um"])
    summary["version"] = VERSION
    cells.to_parquet(folder / "cells.parquet")
    (folder / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")


def clicked_samples() -> list[tuple[str, Path]]:
    """The annotated samples Quillan clicked AF488-positive cells in."""
    from microscopy_viewer import ims_store

    table = jt.annotated_samples()
    return [(row.name, Path(row.path)) for row in table.itertuples()
            if Path(row.path).is_file() and MANUAL_COUNTS in ims_store.list_labels(row.path)]


def run_masks_only(name: str, path: Path) -> dict:
    """The classical masks of a clicked sample, for the notebook to score against the clicks."""
    import tifffile

    folder = OUT_CLICKED / name
    folder.mkdir(parents=True, exist_ok=True)
    af488, nuclear, voxel = read_planes(path)
    masks, info = classical_masks(af488, tissue_mask(nuclear), voxel)
    for method, mask in masks.items():
        tifffile.imwrite(folder / f"{method}.tif", mask.astype(np.uint8), compression="zlib")
    summary = {"version": VERSION, "sample": name, "file": str(path), "voxel_um": voxel,
               "shape": [int(v) for v in af488.shape], **info}
    (folder / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary


def samples() -> list[ja.SegmentedSample]:
    """Every sample with saved Cellpose cells, nuclei, grown regions and ring contrast."""
    import pandas as pd

    found = []
    for sample in ja.segmented_samples(DATA).values():
        nuclei, grown, cells = sample.file("nuclei_file"), sample.file("grown_file"), sample.file("cells_file")
        if not all(f is not None and f.is_file() for f in (nuclei, grown, cells)) or not sample.path.is_file():
            continue
        if "AF488 ring contrast" not in pd.read_csv(cells, nrows=0).columns:
            continue
        found.append(sample)
    return sorted(found, key=lambda s: s.name)


def main(argv: list[str]) -> int:
    below_normal_priority()
    todo = [sample for sample in samples() if cached_version(sample.name) < VERSION]
    if argv and argv[0] == "cached":
        # Bring the samples done already up to date (and mask the clicked ones); no new samples.
        todo = [sample for sample in todo if cached_version(sample.name) >= 1]
    elif argv:
        todo = todo[: int(argv[0])]
    print(f"{len(todo)} sample(s) to do -> {OUT}", flush=True)
    for index, sample in enumerate(todo, start=1):
        started = time.perf_counter()
        try:
            if cached_version(sample.name) == 1:
                upgrade(sample)
                print(f"[{index}/{len(todo)}] {sample.name}: median + 7 SD added, {time.perf_counter() - started:.0f} s",
                      flush=True)
            else:
                summary = run_sample(sample)
                print(f"[{index}/{len(todo)}] {sample.name}: {time.perf_counter() - started:.0f} s {summary['timings_s']}",
                      flush=True)
        except Exception as exc:
            print(f"[{index}/{len(todo)}] {sample.name}: FAILED {exc!r}", flush=True)
    clicked = [(name, path) for name, path in clicked_samples() if not (OUT_CLICKED / name / "summary.json").is_file()]
    print(f"{len(clicked)} clicked sample(s) to mask -> {OUT_CLICKED}", flush=True)
    for index, (name, path) in enumerate(clicked, start=1):
        started = time.perf_counter()
        try:
            run_masks_only(name, path)
            print(f"[{index}/{len(clicked)}] {name}: {time.perf_counter() - started:.0f} s", flush=True)
        except Exception as exc:
            print(f"[{index}/{len(clicked)}] {name}: FAILED {exc!r}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
