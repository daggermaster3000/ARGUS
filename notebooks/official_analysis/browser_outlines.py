"""Full-resolution outlines for the section browser: the AF488 mask objects traced at the scan's pixels.

``section_browser.py`` stores every method's outlines simplified to 0.5 overview pixels (~0.65 µm), which is right
for the overview but cuts corners once the page shows the full-resolution tiles (``browser_tiles.py``). This script
traces the objects of the Gaussian 20 µm background mask (:data:`KEYS`: 3, 5 and 8 SD; 8 SD is the counting method)
again, at ~0.325 µm per pixel, without simplifying (only points on a straight run are dropped), so an outline
follows the pixels of the mask that was counted.

The mask is made exactly as in ``section_browser.run_masks`` (the stored threshold of ``af488_mask/<section>``,
objects outside the section dropped by ``section_outline``), so the objects come in the same order as in ``parts/masks/<id>__<key>.js``; the page takes side, region, area and
distance from that file and only the rings from this one. A section whose object count differs from the stored
one is reported and not written.

    parts/fine/<id>__<key>.js      window.FINE_LOADED(id, key, {"q": 2·f, "rings": [...]})

Each ring is flat integers ``[x0, y0, dx1, dy1, dx2, dy2, …]``: the first point, then steps, in half scan pixels
(``q`` of them per overview pixel). A section with all its files is done, so a rerun skips it.

    python notebooks/official_analysis/browser_outlines.py            # every section not done yet
    python notebooks/official_analysis/browser_outlines.py 3          # just three, to time it

No Qt.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "notebooks" / "validation"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

# ---------------------------------------------------------------- Settings
DATA = Path(r"D:\Slidescanner extracted")
RUN = DATA / "cells_analysis" / "2026-10-28_090440 re-scored"     # the cells-analysis run
BROWSER = DATA / "run_analysis_20261028" / "section_browser"      # section_browser.py's folder
KEYS = ("gauss-3", "gauss-5", "gauss-8")   #: the Gaussian mask's thresholds traced at full resolution
STRAIGHT_PX = 0.05                #: points within this many scan pixels of a straight run are dropped
WORKERS = 4                       #: each holds a slide scan's planes (~2 GB)
VERSION = 2                       #: 2: objects outside the section dropped
# --------------------------------------------------------------------------


def fine_file(out: Path, sid: str, key: str) -> Path:
    return out / "parts" / "fine" / f"{sid}__{key}.js"


def fine_done(out: Path, sid: str) -> bool:
    try:
        return all(f'"version":{VERSION},' in fine_file(out, sid, k).read_text(encoding="utf-8")[:200] for k in KEYS)
    except OSError:
        return False


def fine_rings(crop: np.ndarray, top: int, left: int) -> list[list[int]]:
    """The rings round the True pixels of *crop* (corner at scan pixel (top, left)), in half scan pixels from the
    picture's corner, each ``[x0, y0, dx, dy, …]``."""
    from skimage import measure

    rings = []
    for contour in measure.find_contours(np.pad(crop, 1).astype(np.float32), 0.5):
        contour = measure.approximate_polygon(contour, STRAIGHT_PX)
        # pixel-centre coordinates of the padded crop → the scan's pixel edges (+0.5), doubled: whole numbers
        ys = np.round(2 * (contour[:, 0] - 1 + top) + 1).astype(np.int64)
        xs = np.round(2 * (contour[:, 1] - 1 + left) + 1).astype(np.int64)
        if len(xs) > 1 and xs[0] == xs[-1] and ys[0] == ys[-1]:
            xs, ys = xs[:-1], ys[:-1]                 # the polygon closes itself
        if not len(xs):
            continue
        steps = np.column_stack([np.diff(xs, prepend=0), np.diff(ys, prepend=0)])
        steps[0] = (xs[0], ys[0])
        rings.append(steps.ravel().tolist())
    return rings


def run_fine(name: str, out: str) -> str:
    """One section's full-resolution outlines for every key → parts/fine/."""
    from scipy import ndimage
    from skimage.measure import label, regionprops

    import classical_vs_cellpose as cv
    import section_browser as sb

    logging.getLogger("tifffile").setLevel(logging.ERROR)
    cv.below_normal_priority()
    out = Path(out)
    started = time.perf_counter()
    sid = sb.short(name)
    stored_counts = json.loads(sb.methods_file(out, name).read_text(encoding="utf-8"))["counts"]
    info = json.loads((RUN / "cells" / f"{name}.json").read_text(encoding="utf-8"))
    af488, nuclear, voxel = cv.read_planes(Path(info["file"]))
    f = max(1, int(round(sb.TARGET_UM / voxel)))
    tissue = cv.tissue_mask(nuclear)
    del nuclear
    outline = sb.section_outline.outline_for(sb.MASKS / name, Path(info["file"]))
    line_um = sb.section_outline.line_um(info)
    # as section_browser.run_masks: smoothed, less the 20 µm Gaussian, cut at the stored threshold in the tissue
    smooth = ndimage.gaussian_filter(af488, cv.SMOOTH_UM / voxel)
    del af488
    signal = smooth - cv.local_background(smooth, cv.BACKGROUND_UM / voxel)
    del smooth
    stored = json.loads((sb.MASKS / name / "summary.json").read_text(encoding="utf-8"))["thresholds"]
    cuts = sb.robust_cuts(signal, tissue, sb.METHODS["gauss"]["ks"])[0]
    (out / "parts" / "fine").mkdir(parents=True, exist_ok=True)
    said = []
    for key in KEYS:
        k = int(key.split("-")[1])
        cut = float(stored.get(f"kmad{k:g}", cuts[k]))
        mask = sb.section_outline.drop_outside(sb.cleaned((signal > cut) & tissue, voxel), outline, line_um, voxel)[0]
        labelled = label(mask, connectivity=2)
        rings = [fine_rings(p.image, p.bbox[0], p.bbox[1]) for p in regionprops(labelled)]
        if len(rings) != stored_counts[key]["n"]:
            said.append(f"{key} MISMATCH {len(rings)} vs {stored_counts[key]['n']}")
            continue
        text = json.dumps({"version": VERSION, "q": 2 * f, "rings": rings}, separators=(",", ":"))
        fine_file(out, sid, key).write_text(f"window.FINE_LOADED({json.dumps(sid)},{json.dumps(key)},{text});\n",
                                            encoding="utf-8")
        said.append(f"{key} {len(rings)}")
    return f"{sid}: {', '.join(said)}, {time.perf_counter() - started:.1f} s"


def main(argv: list[str]) -> int:
    import section_browser as sb

    out = BROWSER
    limit = int(argv[0]) if argv and argv[0].isdigit() else None
    names = [json.loads(p.read_text(encoding="utf-8"))["name"] for p in sorted((out / "parts" / "methods").glob("*.json"))]
    todo = [n for n in names if not fine_done(out, sb.short(n))]
    if limit:
        todo = todo[:limit]
    print(f"{len(names)} sections, {len(todo)} to trace", flush=True)
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(run_fine, n, str(out)): n for n in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                msg = fut.result()
            except Exception as err:          # one bad file does not stop the rest
                msg = f"{sb.short(futures[fut])}: FAILED {err!r}"
            print(f"[{i}/{len(todo)}, {(time.perf_counter() - started) / 60:.1f} min] {msg}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
