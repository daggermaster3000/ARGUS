"""AF488⁺ cells segmented directly by Cellpose (the DINO-backbone model ``cpdino``), as another positivity method.

Instead of thresholding AF488 and counting connected objects, Cellpose is shown the AF488 image (with the nuclear
stain as a second channel) and asked for cell bodies; each mask is a candidate AF488⁺ cell. Positivity is then a
threshold on the mask's mean background-subtracted AF488 in noise SDs of the section (the counting mask's own signal
and noise: ``snr``), so the same masks give a count at any threshold.

Per section:

- AF488 and the nuclear stain (or AF488 alone, :data:`INPUT`), maximum-intensity projection over z, each scaled once for the whole section to its
  :data:`NORM_PCT` percentiles in the section's tissue (not per tile: an empty tile would otherwise be stretched until
  its noise looked like cells), clipped to 0–1.5;
- Cellpose ``cpdino`` (:data:`MODEL`) on the two-channel image, :data:`FLOW_THRESHOLD`, :data:`CELLPROB_THRESHOLD`,
  masks of :data:`MIN_AREA_UM2` or more;
- per mask: centroid (µm, index × voxel), area, mean raw AF488, mean background-subtracted signal (Gaussian 0.5 µm
  less a Gaussian 20 µm, as the counting mask), ``snr`` = that ÷ the section's noise SD (``af488_mask/<s>/summary.json``),
  and whether it lies in the section (``section_outline.in_section``, as every method).

Written to ``run_analysis_20261028/cellpose_af488/<section>.parquet`` and the label map ``<section>.npz``.

    python notebooks/official_analysis/cellpose_af488.py --squares        # the 48 sections of the blinded squares
    python notebooks/official_analysis/cellpose_af488.py                  # every section (GPU, one at a time)
    python notebooks/official_analysis/cellpose_af488.py --squares --af488-only   # AF488 alone → cellpose_af488_only/

Torch is imported after the scan is read (no Qt here, but keep the order the viewer needs). No Qt.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "notebooks" / "validation"))
sys.path.insert(0, str(ROOT / "notebooks" / "dataset"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
import section_browser as sb  # noqa: E402

# ---------------------------------------------------------------- Settings
MODEL = "cpdino"
#: the image Cellpose sees: "two" AF488 + nuclear stain (it then finds every cell; positivity is the SNR threshold),
#: "af488" AF488 alone (it finds green cell bodies); chosen with --af488-only
INPUT = "af488" if "--af488-only" in sys.argv else "two"
OUT = sb.OUT / ("cellpose_af488" if INPUT == "two" else "cellpose_af488_only")
NORM_PCT = (1.0, 99.9)            #: each channel scaled to these percentiles of the section's tissue
FLOW_THRESHOLD = 0.4
CELLPROB_THRESHOLD = 0.0
MIN_AREA_UM2 = 20.0               #: smaller masks dropped (the counting mask's size filter)
BATCH = 16
VERSION = 1
# --------------------------------------------------------------------------

_MODEL = None


def model():
    global _MODEL
    if _MODEL is None:
        from cellpose import models

        _MODEL = models.CellposeModel(gpu=True, pretrained_model=MODEL)
    return _MODEL


def done(name: str) -> bool:
    f = OUT / f"{sb.short(name)}.parquet"
    try:
        return pd.read_parquet(f, columns=["version"])["version"].min() >= VERSION
    except Exception:
        return False


def scaled(plane: np.ndarray, tissue: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(plane[tissue][::7], NORM_PCT)
    return np.clip((plane - lo) / max(hi - lo, 1e-6), 0, 1.5).astype(np.float32)


def run_one(name: str) -> str:
    from scipy import ndimage
    from skimage.measure import regionprops_table

    import classical_vs_cellpose as cv
    import section_outline

    logging.getLogger("tifffile").setLevel(logging.ERROR)
    started = time.perf_counter()
    info = json.loads((sb.RUN / "cells" / f"{name}.json").read_text(encoding="utf-8"))
    af488, nuclear, voxel = cv.read_planes(Path(info["file"]))
    nuclear = nuclear.astype(np.float32)
    section = section_outline.outline_for(sb.MASKS / name, Path(info["file"]))
    tissue = section_outline.stats_mask(section, af488.shape)
    image = (np.stack([scaled(af488, tissue), scaled(nuclear, tissue)], axis=-1) if INPUT == "two"
             else scaled(af488, tissue))
    del nuclear
    t_read = time.perf_counter() - started
    out = model().eval(image, batch_size=BATCH, normalize=False, flow_threshold=FLOW_THRESHOLD,
                       cellprob_threshold=CELLPROB_THRESHOLD, min_size=int(round(MIN_AREA_UM2 / voxel ** 2)))
    labels = np.asarray(out[0]).astype(np.int32)
    del image
    t_seg = time.perf_counter() - started - t_read
    smooth = ndimage.gaussian_filter(af488, cv.SMOOTH_UM / voxel)
    signal = smooth - cv.local_background(smooth, cv.BACKGROUND_UM / voxel)
    del smooth
    noise = float(json.loads((sb.MASKS / name / "summary.json").read_text(encoding="utf-8"))["noise_sd"])
    p = regionprops_table(labels, intensity_image=np.stack([af488, signal], axis=-1),
                          properties=("label", "centroid", "area", "intensity_mean"))
    t = pd.DataFrame({"label": p["label"], "y": p["centroid-0"] * voxel, "x": p["centroid-1"] * voxel,
                      "area": p["area"] * voxel ** 2, "mean": p["intensity_mean-0"], "signal": p["intensity_mean-1"]})
    t["snr"] = t["signal"] / noise
    t["in_section"] = section_outline.in_section(section, section_outline.line_um(info), t[["y", "x"]].to_numpy(float), voxel) \
        if len(t) else np.zeros(0, bool)
    t["version"] = VERSION
    OUT.mkdir(parents=True, exist_ok=True)
    t.to_parquet(OUT / f"{sb.short(name)}.parquet", index=False)
    np.savez_compressed(OUT / f"{sb.short(name)}.npz", labels=labels)
    n5 = int(((t["snr"] >= 5) & t["in_section"]).sum())
    return (f"{sb.short(name)}: {len(t)} masks ({n5} at SNR >= 5), read {t_read:.0f} s, Cellpose {t_seg:.0f} s, "
            f"total {time.perf_counter() - started:.0f} s")


def main(argv: list[str]) -> int:
    if "--squares" in argv:
        names = sorted(pd.read_csv(sb.SQUARES_KEY)["name"].unique())
    else:
        names = list(sb.section_table()["name"])
    limit = next((int(a) for a in argv if a.isdigit()), None)
    todo = [n for n in names if not done(n)][:limit]
    print(f"{len(todo)} sections -> {OUT} ({MODEL})", flush=True)
    started = time.perf_counter()
    for i, n in enumerate(todo, 1):
        try:
            line = run_one(n)
        except Exception as exc:
            line = f"{sb.short(n)}: FAILED {exc!r}"
        el = time.perf_counter() - started
        print(f"[{i}/{len(todo)}] {line} | {el / 60:.1f} min, ~{el / i * (len(todo) - i) / 60:.0f} min left", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
