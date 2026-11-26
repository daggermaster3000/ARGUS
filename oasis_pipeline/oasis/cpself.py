"""The paper's counting method: AF488⁺ cells segmented by a self-trained Cellpose model (cpdino), counted by their signal.

Cellpose (model ``cpdino``, DINO backbone, fine-tuned on this dataset without hand labels; ``data/cellpose/model/``)
is shown each section's AF488 and nuclear stain (maximum over z), each scaled once for the whole section to its
1st–99.9th percentile in the section's tissue and clipped to 0–1.5 (flow threshold 0.4, cell-probability threshold 0).
Per mask: centroid (µm, index × voxel), area, mean background-subtracted AF488 (Gaussian 0.5 µm less a Gaussian
20 µm, as the AF488 mask of ``counting.py``), ``snr`` = that ÷ the section's noise SD (the AF488 mask's robust SD,
``counting/af488_mask/<s>/summary.json``), and whether it lies in the section (``section_outline.in_section``).

A mask is an AF488⁺ cell when ``snr`` ≥ k (k = 3, 5, 8; **5** in the paper) and its area is more than the size filter
(20 µm²; 10, 5 and 3 µm² as sensitivity variants). Each cell's compartment, as for every method: home or crossed by
its side of the junction interface extended past its ends (``sides.away``; the home side from the AF488 mask's 8 SD
object area, ``browser.section_browser.section_sign``), and whether it lies in the junction region.

Two steps:

* ``infer`` (GPU, optional): the masks of every section again with the shipped model, into
  ``rerun/counting/cellpose/<id>.parquet`` (+ the label map ``<id>.npz``). Without it the shipped masks
  (``data/counting/cellpose/``) are used. Cellpose on a GPU is not bit-for-bit repeatable, so re-made masks differ
  slightly from the shipped ones.
* ``count`` (CPU, seconds): every method-threshold's counts per section from the masks, into
  ``rerun/counting/cellpose_counts.csv`` (section, method, n, home, crossed, region).

No Qt. Torch and Cellpose are imported only by ``infer``.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import cells as ja
from . import cv, paths, section_outline, sides

#: method → size filter: masks of this area (µm²) or less are not counted
SIZES = {"cpself": 20.0, "cpself10": 10.0, "cpself5": 5.0, "cpself3": 3.0}
KS = (3, 5, 8)                    #: the signal thresholds, in noise SD
DEFAULT = "cpself-5"              #: the paper's counting method
MIN_AREA_UM2 = 3.0                #: masks kept at inference (every size filter can be applied afterwards)
NORM_PCT = (1.0, 99.9)            #: each channel scaled to these percentiles of the section's tissue
FLOW_THRESHOLD = 0.4
CELLPROB_THRESHOLD = 0.0
BATCH = 16
VERSION = 1
PREFIX = "2025_09_26_Asssembloid_"


def short(name: str) -> str:
    return name.replace(PREFIX, "")


def keys() -> list[str]:
    return [f"{m}-{k}" for m in SIZES for k in KS]


# ---------------------------------------------------------------- infer (GPU)
_MODEL = None


def model():
    global _MODEL
    if _MODEL is None:
        from cellpose import models

        _MODEL = models.CellposeModel(gpu=True, pretrained_model=str(paths.CELLPOSE_MODEL))
    return _MODEL


def scaled(plane: np.ndarray, tissue: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(plane[tissue][::7], NORM_PCT)
    return np.clip((plane - lo) / max(hi - lo, 1e-6), 0, 1.5).astype(np.float32)


def infer_one(name: str) -> str:
    """One section's Cellpose masks and their signal."""
    from scipy import ndimage
    from skimage.measure import regionprops_table

    logging.getLogger("tifffile").setLevel(logging.ERROR)
    started = time.perf_counter()
    info = paths.section_info(name)
    af488, nuclear, voxel = cv.read_planes(paths.scan_path(name))
    nuclear = nuclear.astype(np.float32)
    outline = section_outline.outline_for(paths.af488_mask(name), paths.scan_path(name))
    tissue = section_outline.stats_mask(outline, af488.shape)
    image = np.stack([scaled(af488, tissue), scaled(nuclear, tissue)], axis=-1)
    del nuclear
    out = model().eval(image, batch_size=BATCH, normalize=False, flow_threshold=FLOW_THRESHOLD,
                       cellprob_threshold=CELLPROB_THRESHOLD, min_size=int(round(MIN_AREA_UM2 / voxel ** 2)))
    labels = np.asarray(out[0]).astype(np.int32)
    del image
    smooth = ndimage.gaussian_filter(af488, cv.SMOOTH_UM / voxel)
    signal = smooth - cv.local_background(smooth, cv.BACKGROUND_UM / voxel)
    del smooth
    noise = float(json.loads((paths.af488_mask(name) / "summary.json").read_text(encoding="utf-8"))["noise_sd"])
    p = regionprops_table(labels, intensity_image=np.stack([af488, signal], axis=-1),
                          properties=("label", "centroid", "area", "intensity_mean"))
    t = pd.DataFrame({"label": p["label"], "y": p["centroid-0"] * voxel, "x": p["centroid-1"] * voxel,
                      "area": p["area"] * voxel ** 2, "mean": p["intensity_mean-0"], "signal": p["intensity_mean-1"]})
    t["snr"] = t["signal"] / noise
    t["in_section"] = (section_outline.in_section(outline, section_outline.line_um(info), t[["y", "x"]].to_numpy(float), voxel)
                       if len(t) else np.zeros(0, bool))
    t["version"] = VERSION
    paths.RERUN_CELLPOSE.mkdir(parents=True, exist_ok=True)
    t.to_parquet(paths.RERUN_CELLPOSE / f"{short(name)}.parquet", index=False)
    np.savez_compressed(paths.RERUN_CELLPOSE / f"{short(name)}.npz", labels=labels)
    return f"{short(name)}: {len(t)} masks, {time.perf_counter() - started:.0f} s"


def infer(names: list[str] | None = None) -> int:
    """Cellpose on every section (one at a time on the GPU); needs the ``counting`` step (outline, noise SD) first."""
    names = names or paths.section_names()
    todo = [n for n in names if not (paths.RERUN_CELLPOSE / f"{short(n)}.parquet").exists()]
    print(f"cellpose: {len(todo)} sections -> {paths.RERUN_CELLPOSE} (model {paths.CELLPOSE_MODEL.name})", flush=True)
    started, failed = time.perf_counter(), 0
    for i, n in enumerate(todo, 1):
        try:
            line = infer_one(n)
        except Exception as exc:
            failed += 1
            line = f"{short(n)}: FAILED {exc!r}"
        el = time.perf_counter() - started
        print(f"[{i}/{len(todo)}] {line} | {el / 60:.1f} min, ~{el / i * (len(todo) - i) / 60:.0f} min left", flush=True)
    if not failed and all((paths.RERUN_CELLPOSE / f"{short(n)}.parquet").exists() for n in paths.section_names()):
        (paths.RERUN_CELLPOSE / "COMPLETE").write_text("every section\n", encoding="utf-8")
    return 1 if failed else 0


# ---------------------------------------------------------------- count (CPU)
SIDE_MASK = "kmad8"               #: the AF488 mask whose object area decides the home side (the same for every method)


def home_sign(name: str, line) -> float | None:
    """The section's home sign (``sides.home_sign``) from the AF488 mask's 8 SD objects' area; None without a line."""
    if line is None:
        return None
    o = pd.read_parquet(paths.af488_mask(name) / "objects.parquet")
    o = o[o["mask"] == SIDE_MASK]
    raw = sides.raw_side(o[["Centroid Y (µm)", "Centroid X (µm)"]].to_numpy(float), line)
    return sides.home_sign(raw, o["Area (µm²)"].to_numpy(float))


def placed(name: str) -> tuple[pd.DataFrame, float | None]:
    """The section's masks in the section, with ``away`` (µm past the junction interface; NaN without one) and ``reg``
    (in the junction region), and the home sign."""
    t = pd.read_parquet(paths.cellpose() / f"{short(name)}.parquet")
    t = t[t["in_section"]].reset_index(drop=True)
    info = paths.section_info(name)
    line = np.asarray(info["line"], float) if info.get("line") else None
    sign = home_sign(name, line)
    yx = t[["y", "x"]].to_numpy(float)
    t["away"] = sides.away(yx, line, sign) if sign is not None and len(yx) else np.full(len(yx), np.nan)
    if paths.region_tif(name).exists() and len(yx):
        band = ja.read_labels(paths.region_tif(name)) > 0
        voxel = float(info["voxel_yx"][0])
        rows = np.clip(np.round(yx[:, 0] / voxel).astype(int), 0, band.shape[0] - 1)
        cols = np.clip(np.round(yx[:, 1] / voxel).astype(int), 0, band.shape[1] - 1)
        t["reg"] = band[rows, cols]
    else:
        t["reg"] = False
    return t, sign


def counts_of(t: pd.DataFrame, has_line: bool) -> list[dict]:
    """Per method-threshold: all cells, home, crossed, in the junction region."""
    out = []
    for m, size in SIZES.items():
        for k in KS:
            sel = t[(t["snr"] >= k) & (t["area"] > size)]
            a = sel["away"].to_numpy(float)
            out.append({"method": f"{m}-{k}", "n": int(len(sel)), "home": int((a <= 0).sum()) if has_line else None,
                        "crossed": int((a > 0).sum()) if has_line else None, "region": int(sel["reg"].astype(bool).sum())})
    return out


def count(names: list[str] | None = None) -> int:
    names = names or paths.section_names()
    rows, failed = [], 0
    for n in names:
        try:
            t, sign = placed(n)
            rows += [{"section": short(n), **r} for r in counts_of(t, sign is not None)]
        except Exception as exc:
            failed += 1
            print(f"{short(n)}: FAILED {exc!r}", flush=True)
    table = pd.DataFrame(rows)
    paths.CELLPOSE_COUNTS.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(paths.CELLPOSE_COUNTS, index=False)
    d = table[table["method"] == DEFAULT]
    print(f"cellpose counts: {d['section'].nunique()} sections from {paths.cellpose()} -> {paths.CELLPOSE_COUNTS}; "
          f"{DEFAULT}: {int(d['n'].sum())} cells, {int(d['crossed'].sum())} crossed", flush=True)
    return 1 if failed else 0


def main(names: list[str] | None = None, workers: int = 1, infer_first: bool = False) -> int:
    code = infer(names) if infer_first else 0
    return code or count(names)
