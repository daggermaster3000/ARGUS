"""Shape, intensity and nucleus features of every AF488 mask object, to tell cell bodies from neurites and debris.

The objects are those of the counting mask (Gaussian 20 µm background, median + 8 robust SD of the section's tissue)
with the smallest size filter (objects > 3 µm², the browser's ``gauss3-8``), so that the size filter becomes one more
feature: keeping area > 20 µm² gives the counting method's objects exactly. The mask is re-cut here exactly as
``section_browser.run_masks`` cuts it (same functions, same stored threshold), so the objects come in the order of
``parts/masks/<id>__gauss3-8.js``; a section whose object count differs from the stored one is reported and skipped.

Per object (one row; µm and the raw maximum-intensity projection over z):

- **shape**: area, perimeter, circularity (4π·area / perimeter², 1 for a disc), solidity (area / convex area),
  extent, eccentricity, major and minor axis, aspect ratio, thickness (twice the largest inscribed radius), skeleton
  length, mean width (area / skeleton length), skeleton end points and branch points;
- **intensity**: mean, max and integrated AF488 (raw), mean and max of the background-subtracted signal the mask was
  cut from, the mean signal in noise SDs, and the contrast: mean raw AF488 over that of a :data:`RING_UM` ring round
  the object (other objects left out);
- **nucleus**: the share of the object lying on a segmented nucleus (the run's nuclei label map), the number of
  nuclei it touches, and how much of its main nucleus it covers;
- **place**: centroid (µm, index × voxel as the run), side (1 crossed, 0 home, -1 no line), in the junction region
  and distance past the junction interface, from the stored outline file;
- **outline**: the object's outer contour relative to its centroid, in :data:`OUTLINE_STEP_UM` steps, as signed bytes
  (base64), for drawing it on the object map.

    object_features/<section>.parquet

    python notebooks/official_analysis/object_features.py          # every section not done yet (~1 h, 4 at a time)
    python notebooks/official_analysis/object_features.py 3        # just three

No Qt.
"""
from __future__ import annotations

import base64
import json
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
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
KEY = "gauss3-8"                  #: the objects: counting mask, 8 SD, objects > 3 µm²
OUT = sb.OUT / "object_features"
RING_UM = (3.0, 6.0)              #: the local background ring round an object (as count_cells.py)
OUTLINE_STEP_UM = 0.5             #: outline coordinates in these steps (signed bytes: ±63 µm round the centroid)
OUTLINE_POINTS = 48               #: at most this many outline points (resampled along the contour)
WORKERS = 4
VERSION = 1
# --------------------------------------------------------------------------


def out_file(name: str) -> Path:
    return OUT / f"{sb.short(name)}.parquet"


def done(name: str) -> bool:
    try:
        return pd.read_parquet(out_file(name), columns=["version"])["version"].min() >= VERSION
    except Exception:
        return False


def skeleton_points(skel: np.ndarray) -> tuple[int, int]:
    """End points (one neighbour) and branch points (three or more) of a skeleton, 8-connected."""
    from scipy import ndimage

    n = ndimage.convolve(skel.astype(np.uint8), np.ones((3, 3), np.uint8), mode="constant") - skel
    return int((skel & (n == 1)).sum()), int((skel & (n >= 3)).sum())


def outline_code(img: np.ndarray, cy: float, cx: float, top: int, left: int, voxel: float) -> str:
    """The outer contour round the centroid, resampled to at most OUTLINE_POINTS, as base64 signed bytes (x, y)."""
    from skimage import measure

    contours = measure.find_contours(np.pad(img, 1).astype(np.float32), 0.5)
    if not contours:
        return ""
    c = max(contours, key=len)
    if len(c) > OUTLINE_POINTS:
        c = c[np.linspace(0, len(c) - 1, OUTLINE_POINTS).astype(int)]
    ys = (c[:, 0] - 1 + top - cy) * voxel / OUTLINE_STEP_UM
    xs = (c[:, 1] - 1 + left - cx) * voxel / OUTLINE_STEP_UM
    xy = np.clip(np.round(np.column_stack([xs, ys])), -127, 127).astype(np.int8)
    return base64.b64encode(xy.tobytes()).decode("ascii")


def run_one(name: str) -> str:
    from scipy import ndimage
    from skimage.measure import label, regionprops
    from skimage.morphology import skeletonize

    import classical_vs_cellpose as cv
    import section_outline
    from microscopy_viewer import junction_analysis as ja

    logging.getLogger("tifffile").setLevel(logging.ERROR)
    cv.below_normal_priority()
    started = time.perf_counter()
    sid = sb.short(name)
    info = json.loads((sb.RUN / "cells" / f"{name}.json").read_text(encoding="utf-8"))
    af488, _nuclear, voxel = cv.read_planes(Path(info["file"]))
    del _nuclear
    # the mask, exactly as section_browser.run_masks cuts the size variants
    section = section_outline.outline_for(sb.MASKS / name, Path(info["file"]))
    domain = section_outline.count_domain(section, af488.shape)
    line_um = section_outline.line_um(info)
    stored = json.loads((sb.MASKS / name / "summary.json").read_text(encoding="utf-8"))
    cut = float(stored["thresholds"]["kmad8"])
    noise = float(stored["noise_sd"])
    smooth = ndimage.gaussian_filter(af488, cv.SMOOTH_UM / voxel)
    signal = smooth - cv.local_background(smooth, cv.BACKGROUND_UM / voxel)
    del smooth
    mask = section_outline.drop_outside(sb.cleaned((signal > cut) & domain, voxel, sb.min_um2(KEY)), section, line_um, voxel)[0]
    labels = label(mask, connectivity=2)
    del mask
    props = regionprops(labels)
    # side, region and distance from the stored outline file (same objects, same order)
    text = sb.mask_file(sb.BROWSER, name, KEY).read_text(encoding="utf-8")
    shapes = json.loads(text[text.index(",{") + 1:text.rindex(");")])
    if len(shapes["side"]) != len(props):
        return f"{sid}: SKIPPED, {len(props)} objects against {len(shapes['side'])} stored"
    nuclei = ja.read_labels(sb.RUN / info["nuclei_file"])
    r_in, r_out = RING_UM[0] / voxel, RING_UM[1] / voxel
    pad = int(np.ceil(r_out)) + 1
    H, W = labels.shape
    rows = []
    for i, p in enumerate(props):
        r0, c0, r1, c1 = p.bbox
        img = p.image
        area = float(p.area)
        # shape
        perim = float(p.perimeter) or 1.0
        skel = skeletonize(img)
        skel_len = float(skel.sum())
        ends, branches = skeleton_points(skel)
        inscribed = float(ndimage.distance_transform_edt(np.pad(img, 1)).max())
        # the object with a margin: ring, nuclei
        R0, C0, R1, C1 = max(r0 - pad, 0), max(c0 - pad, 0), min(r1 + pad, H), min(c1 + pad, W)
        lab = labels[R0:R1, C0:C1]
        me = lab == p.label
        dist = ndimage.distance_transform_edt(~me)
        ring = (dist > r_in) & (dist <= r_out) & (lab == 0)
        raw = af488[R0:R1, C0:C1]
        sig = signal[R0:R1, C0:C1]
        nuc = nuclei[R0:R1, C0:C1]
        on = nuc[me]
        on_nuc = on[on > 0]
        main = int(np.bincount(on_nuc).argmax()) if on_nuc.size else 0
        if main:
            whole = int((nuclei[max(r0 - 3 * pad, 0):min(r1 + 3 * pad, H), max(c0 - 3 * pad, 0):min(c1 + 3 * pad, W)] == main).sum())
            cover = float((on == main).sum()) / max(whole, 1)
        else:
            cover = 0.0
        mean_in = float(raw[me].mean())
        mean_ring = float(raw[ring].mean()) if ring.any() else np.nan
        cy, cx = p.centroid
        rows.append({
            "section": sid, "index": i,
            "y": cy * voxel, "x": cx * voxel,
            "area": area * voxel ** 2, "perimeter": perim * voxel,
            "circularity": min(4 * np.pi * area / perim ** 2, 1.0), "solidity": float(p.solidity), "extent": float(p.extent),
            "eccentricity": float(p.eccentricity),
            "major": float(p.axis_major_length) * voxel, "minor": float(p.axis_minor_length) * voxel,
            "aspect": float(p.axis_major_length) / max(float(p.axis_minor_length), 1.0),
            "thickness": 2 * inscribed * voxel, "skeleton": skel_len * voxel,
            "width": area / max(skel_len, 1.0) * voxel, "ends": ends, "branches": branches,
            "mean": mean_in, "max": float(raw[me].max()), "integrated": float(raw[me].sum()),
            "signal": float(sig[me].mean()), "signal_max": float(sig[me].max()), "snr": float(sig[me].mean()) / noise,
            "contrast": mean_in / mean_ring if mean_ring and np.isfinite(mean_ring) and mean_ring > 0 else np.nan,
            "on_nucleus": float(on_nuc.size) / area, "nuclei": int(np.unique(on_nuc).size), "nucleus_cover": cover,
            "side": int(shapes["side"][i]), "reg": int(shapes["reg"][i]),
            "away": shapes["away"][i] if shapes["away"][i] is not None else np.nan,
            "outline": outline_code(img, cy, cx, r0, c0, voxel), "version": VERSION,
        })
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out_file(name), index=False)
    return f"{sid}: {len(rows)} objects, {time.perf_counter() - started:.0f} s"


def main(argv: list[str]) -> int:
    table = sb.section_table()
    limit = int(argv[0]) if argv else None
    todo = [n for n in table["name"] if not done(n)][:limit]
    print(f"{len(todo)} sections to do -> {OUT}", flush=True)
    started = time.perf_counter()
    with ProcessPoolExecutor(min(WORKERS, max(len(todo), 1))) as pool:
        futures = {pool.submit(run_one, n): n for n in todo}
        for i, f in enumerate(as_completed(futures), 1):
            try:
                line = f.result()
            except Exception as exc:
                line = f"{sb.short(futures[f])}: FAILED {exc!r}"
            el = time.perf_counter() - started
            print(f"[{i}/{len(todo)}] {line} | {el / 60:.1f} min, ~{el / i * (len(todo) - i) / 60:.0f} min left", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
