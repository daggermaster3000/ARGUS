"""Classical AF488 masks on James's 14 hand-counted sections: background × threshold.

The mask is that of ``af488_masks.py`` (``classical_vs_cellpose.classical_masks``),
with the local background and the threshold each varied:

* **background** — ``none`` (AF488 flattened over z and smoothed 0.5 µm, nothing
  taken away); ``Gaussian 20 µm`` (the smoothed channel less its 20 µm Gaussian,
  the existing mask); ``median 45 µm`` (the channel less its 45 µm local median,
  exactly the run's filter ``junction_analysis.less_local_median``, then smoothed
  0.5 µm);
* **threshold**, inside the tissue — the median + k robust SDs (1.4826 × MAD) for
  k = 3, 5, 7, 8 and 10, and Otsu's and the triangle threshold, as
  ``classical_vs_cellpose.METHODS``.

Every mask is cleaned the same way: objects under 20 µm² dropped, holes under
10 µm² filled. Variants are named ``mask · <background> · <threshold>``, e.g.
``mask · median 45 µm · 5 SD``.

Into ``james_counts/methods/`` (the layout of the methods brief):

* ``cells/mask/<sample>.parquet`` — every cell of the run (``Label``) and the share
  of its grown region each mask covers (``cover <variant>``); a cell is AF488+ by
  a mask when the share is at least 25 % (``classical_vs_cellpose.COVER``);
* ``objects/mask/<sample>.parquet`` — every connected object of every mask
  (``method``, ``object``, centroid ``y``, ``x`` in µm, ``area`` in µm²) placed like
  the run's cells (``james_clicks.place``: ``region``, ``d``, ``beside``, ``edge``);
* ``hits/mask/<sample>.parquet`` — per James dot (``dot``) and per variant, the id
  of the object the dot falls in (the nearest object pixel within 2 µm), 0 for none;
* ``mask/settings.json`` — the parameters, and per section the thresholds and time.

Run from the repository root (CPU only, a few sections at a time)::

    python notebooks/dataset/james_masks.py          # every section not done yet
    python notebooks/dataset/james_masks.py 1        # just one, to time it

No Qt.
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
sys.path.insert(0, str(HERE))
import classical_vs_cellpose as cv  # noqa: E402
import james_clicks as jc  # noqa: E402

from microscopy_viewer import junction_analysis as ja  # noqa: E402

RUN = jc.RUN
CLICKS = jc.TARGET / "clicks.parquet"
OUT = jc.TARGET / "methods"
FAMILY = "mask"
#: The local backgrounds taken away before the threshold.
BACKGROUNDS = ("none", f"Gaussian {cv.BACKGROUND_UM:g} µm", "median 45 µm")
#: The k of the median + k robust SD thresholds.
K_SDS = (3.0, 5.0, 7.0, 8.0, 10.0)
#: The thresholds: k robust SDs, then the automatic ones.
THRESHOLDS = (*(f"{k:g} SD" for k in K_SDS), "Otsu", "triangle")
#: A dot this close to an object's pixel hits it (µm).
HIT_UM = 2.0
#: Processes at a time: each holds a slide scan's planes and masks (~3 GB); others share the machine.
WORKERS = 3
VERSION = 1


def variant(background: str, threshold: str) -> str:
    """The variant's readable name, e.g. ``mask · median 45 µm · 5 SD``."""
    return f"{FAMILY} · {background} · {threshold}"


VARIANTS = tuple(variant(b, t) for b in BACKGROUNDS for t in THRESHOLDS)


def signal_of(af488: np.ndarray, background: str, voxel: float) -> np.ndarray:
    """AF488 smoothed 0.5 µm with *background* taken away (the order as in the existing masks)."""
    from scipy import ndimage

    sigma = cv.SMOOTH_UM / voxel
    if background == "median 45 µm":
        return ndimage.gaussian_filter(ja.less_local_median(af488, voxel), sigma)
    smooth = ndimage.gaussian_filter(af488, sigma)
    if background == "none":
        return smooth
    smooth -= cv.local_background(smooth, cv.BACKGROUND_UM / voxel)
    return smooth


def thresholds_of(signal: np.ndarray, tissue: np.ndarray) -> tuple[dict[str, float], float, float]:
    """Every threshold of :data:`THRESHOLDS` on *signal* inside *tissue*, its median and robust SD."""
    from skimage.filters import threshold_otsu, threshold_triangle

    values = signal[tissue][::20]
    centre = float(np.median(values))
    spread = float(1.4826 * np.median(np.abs(values - centre)))
    cuts = {f"{k:g} SD": centre + k * spread for k in K_SDS}
    cuts["Otsu"] = float(threshold_otsu(values))
    cuts["triangle"] = float(threshold_triangle(values))
    return cuts, centre, spread


def cleaned(mask: np.ndarray, voxel: float) -> np.ndarray:
    """*mask* without objects under 20 µm² and with holes under 10 µm² filled."""
    import warnings

    from skimage.morphology import remove_small_holes, remove_small_objects

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        mask = remove_small_objects(mask, int(round(cv.MIN_OBJECT_UM2 / voxel ** 2)))
        return remove_small_holes(mask, int(round(cv.MAX_HOLE_UM2 / voxel ** 2)))


def hits_of(labelled: np.ndarray, dots_px: np.ndarray, radius_px: float) -> np.ndarray:
    """Per dot, the object whose nearest pixel is within *radius_px* of it, 0 for none."""
    out = np.zeros(len(dots_px), dtype=np.int64)
    r = int(np.ceil(radius_px))
    height, width = labelled.shape
    for i, (y, x) in enumerate(dots_px):
        cy, cx = int(round(y)), int(round(x))
        y0, y1, x0, x1 = max(cy - r, 0), min(cy + r + 1, height), max(cx - r, 0), min(cx + r + 1, width)
        window = labelled[y0:y1, x0:x1]
        rows, cols = np.nonzero(window)
        if not len(rows):
            continue
        dist = np.hypot(rows + y0 - y, cols + x0 - x)
        best = int(np.argmin(dist))
        if dist[best] <= radius_px:
            out[i] = int(window[rows[best], cols[best]])
    return out


def run_one(name: str) -> dict:
    """Every mask of one section: its cover of each cell, its objects, and the dots it hits."""
    import pandas as pd
    from scipy import ndimage
    from skimage.measure import label, regionprops_table

    cv.below_normal_priority()
    started = time.perf_counter()
    info = jc.section_info(name)
    af488, nuclear, voxel = cv.read_planes(Path(info["file"]))
    tissue = cv.tissue_mask(nuclear)
    del nuclear
    grown = ja.read_labels(RUN / info["grown_file"])
    ids = np.unique(grown[grown > 0]).astype(np.int64)
    size = np.asarray(ndimage.sum(grown > 0, grown, ids), dtype=float)
    size = np.where(size > 0, size, np.nan)
    dots = pd.read_parquet(CLICKS)
    dots = dots[dots["name"] == name]
    dots_px = dots[["y", "x"]].to_numpy(float) / voxel

    cells = pd.DataFrame({"Label": ids})
    hits = pd.DataFrame({"dot": dots["dot"].to_numpy()})
    objects, cuts, noise, seconds = [], {}, {}, {}
    for background in BACKGROUNDS:
        tick = time.perf_counter()
        signal = signal_of(af488, background, voxel)
        found, centre, spread = thresholds_of(signal, tissue)
        noise[background] = {"median": centre, "robust_sd": spread}
        for threshold, cut in found.items():
            name_v = variant(background, threshold)
            cuts[name_v] = cut
            mask = cleaned((signal > cut) & tissue, voxel)
            cells[f"cover {name_v}"] = np.asarray(ndimage.sum(mask, grown, ids)) / size
            labelled = label(mask, connectivity=2)
            del mask
            hits[name_v] = hits_of(labelled, dots_px, HIT_UM / voxel)
            if labelled.max():
                props = regionprops_table(labelled, properties=("label", "area", "centroid"))
                objects.append(pd.DataFrame({
                    "method": name_v, "object": props["label"].astype(np.int64),
                    "y": props["centroid-0"] * voxel, "x": props["centroid-1"] * voxel,
                    "area": props["area"] * voxel ** 2}))
            del labelled
        del signal
        seconds[background] = round(time.perf_counter() - tick, 1)
    del af488, grown, tissue

    objects = pd.concat(objects, ignore_index=True) if objects else pd.DataFrame(
        columns=["method", "object", "y", "x", "area"])
    if len(objects):
        placed = jc.place(info, jc.run_cells(name), objects[["y", "x"]].to_numpy(float))
        for column in placed:
            objects[column] = placed[column].to_numpy()

    for kind, table in (("cells", cells), ("objects", objects), ("hits", hits)):
        folder = OUT / kind / FAMILY
        folder.mkdir(parents=True, exist_ok=True)
        table.to_parquet(folder / f"{name}.parquet", index=False)
    return {"sample": name, "voxel_um": voxel, "thresholds": cuts, "noise": noise,
            "seconds_per_background": seconds, "seconds": round(time.perf_counter() - started, 1)}


def settings_path() -> Path:
    return OUT / FAMILY / "settings.json"


def read_settings() -> dict:
    try:
        return json.loads(settings_path().read_text(encoding="utf-8"))
    except Exception:
        return {}


def main(argv: list[str]) -> int:
    import pandas as pd

    names = sorted(pd.read_parquet(CLICKS, columns=["name"])["name"].unique())
    settings = read_settings()
    if settings.get("version", 0) < VERSION:
        settings = {}
    sections = settings.get("sections", {})
    todo = [n for n in names if n not in sections]
    if argv:
        todo = todo[: int(argv[0])]
    print(f"{len(names)} sections, {len(todo)} to do -> {OUT}", flush=True)
    settings.update({
        "version": VERSION, "family": FAMILY, "variants": list(VARIANTS), "backgrounds": list(BACKGROUNDS),
        "thresholds": list(THRESHOLDS), "smooth_um": cv.SMOOTH_UM, "gaussian_background_um": cv.BACKGROUND_UM,
        "median_background": "junction_analysis.less_local_median (45 µm), then smoothed",
        "threshold_values": "AF488 inside the tissue (nuclear Otsu outline), every 20th pixel",
        "min_object_um2": cv.MIN_OBJECT_UM2, "max_hole_um2": cv.MAX_HOLE_UM2, "cover": cv.COVER,
        "hit_um": HIT_UM, "run": str(RUN), "sections": sections})
    started = time.perf_counter()
    with ProcessPoolExecutor(min(WORKERS, max(len(todo), 1))) as pool:
        futures = {pool.submit(run_one, n): n for n in todo}
        for index, future in enumerate(as_completed(futures), start=1):
            try:
                result = future.result()
                sections[result["sample"]] = result
                settings_path().parent.mkdir(parents=True, exist_ok=True)
                settings_path().write_text(json.dumps(settings, indent=1, ensure_ascii=False), encoding="utf-8")
                line = f"{result['sample']}: {result['seconds']:.0f} s"
            except Exception as exc:
                line = f"{futures[future]}: FAILED {exc!r}"
            elapsed = time.perf_counter() - started
            print(f"[{index}/{len(todo)}] {line} — {elapsed / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
