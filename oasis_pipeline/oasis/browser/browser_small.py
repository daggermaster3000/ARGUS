"""The objects the size filter removes, for the section browser: AF488 mask objects too small to be counted.

The counting mask (``section_browser.cleaned``, as ``counting.py``) removes objects of 20 µm² or smaller
(:data:`cv.MIN_OBJECT_UM2`, 4-connected, ``cv.remove_small_objects``) before the objects are counted. Here the
Gaussian 20 µm background mask is made again exactly as ``browser_outlines.run_fine`` does (the stored threshold of
``af488_mask/<section>``, inside the tissue mask), and the pixels that filter takes away are traced at the scan's
pixels as objects of their own (8-connected, as objects are counted). Each is marked as in or outside the section
(``section_outline``: the rule that drops counted objects outside it).

    parts/small/<id>__<key>.js   window.SMALL_LOADED(id, key, {"q": 2·f, "rings": [...], "area": [µm²], "in": [0/1]})
    parts/small/index.js         window.SMALL_INDEX = {id: {key: number of small objects in the section}}

Rings as ``browser_outlines.fine_rings``; ``key`` is ``gauss-3``, ``gauss-5`` or ``gauss-8`` (the counting method),
or the same for the variants with a 10 or 5 µm² size filter (``gauss10-8`` …: what that filter removes).
The page's "too small to count" toggle draws them. A section with all its files is done, so a rerun skips it; the
index is rebuilt from the files at the end.

    python run.py --steps browser      (after section_browser.py)

No Qt.
"""
from __future__ import annotations

import json
import logging
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from .. import paths

# ---------------------------------------------------------------- Settings
BROWSER = paths.BROWSER                                          # section_browser.py's folder
KEYS = tuple(f"{m}-{k}" for m in ("gauss", "gauss10", "gauss5") for k in (3, 5, 8))   #: the Gaussian mask (and its size variants)
WORKERS = 4                       #: each holds a slide scan's planes (~2 GB)
VERSION = 1
# --------------------------------------------------------------------------


def small_file(out: Path, sid: str, key: str) -> Path:
    return out / "parts" / "small" / f"{sid}__{key}.js"


def key_done(out: Path, sid: str, key: str) -> bool:
    try:
        return f'"version":{VERSION},' in small_file(out, sid, key).read_text(encoding="utf-8")[:200]
    except OSError:
        return False


def small_done(out: Path, sid: str) -> bool:
    return all(key_done(out, sid, k) for k in KEYS)


def removed_by_size(mask: np.ndarray, voxel: float, min_area_um2: float = 20.0) -> np.ndarray:
    """The pixels of *mask* that ``section_browser.cleaned``'s size filter (*min_area_um2*) removes (same rule)."""
    from .. import cv

    return mask & ~cv.remove_small_objects(mask, int(round(min_area_um2 / voxel ** 2)))


def run_small(name: str, out: str) -> str:
    """One section's removed objects for every key → parts/small/."""
    from scipy import ndimage
    from skimage.measure import label, regionprops

    from .. import cv
    from . import browser_outlines as bo
    from . import section_browser as sb

    logging.getLogger("tifffile").setLevel(logging.ERROR)
    cv.below_normal_priority()
    out = Path(out)
    started = time.perf_counter()
    sid = sb.short(name)
    info = paths.section_info(name)
    af488, nuclear, voxel = cv.read_planes(paths.scan_path(name))
    f = max(1, int(round(sb.TARGET_UM / voxel)))
    tissue = cv.tissue_mask(nuclear)
    del nuclear
    outline = sb.section_outline.outline_for(sb.MASKS / name, paths.scan_path(name))
    line_um = sb.section_outline.line_um(info)
    # as browser_outlines.run_fine: smoothed, less the 20 µm Gaussian, cut at the stored threshold in the tissue
    smooth = ndimage.gaussian_filter(af488, cv.SMOOTH_UM / voxel)
    del af488
    signal = smooth - cv.local_background(smooth, cv.BACKGROUND_UM / voxel)
    del smooth
    stored = json.loads((sb.MASKS / name / "summary.json").read_text(encoding="utf-8"))["thresholds"]
    cuts = sb.robust_cuts(signal, tissue, sb.METHODS["gauss"]["ks"])[0]
    (out / "parts" / "small").mkdir(parents=True, exist_ok=True)
    said = []
    for key in [k for k in KEYS if not key_done(out, sid, k)]:         # only what is missing
        k = int(key.split("-")[1])
        cut = float(stored.get(f"kmad{k:g}", cuts[k]))
        small = removed_by_size((signal > cut) & tissue, voxel, sb.min_um2(key))
        props = regionprops(label(small, connectivity=2))
        yx = np.array([p.centroid for p in props], float).reshape(-1, 2) * voxel
        inside = sb.section_outline.in_section(outline, line_um, yx, voxel) if len(yx) else np.zeros(0, bool)
        payload = {"version": VERSION, "q": 2 * f,
                   "rings": [bo.fine_rings(p.image, p.bbox[0], p.bbox[1]) for p in props],
                   "area": [round(float(p.area * voxel ** 2), 1) for p in props],
                   "in": inside.astype(int).tolist()}
        text = json.dumps(payload, separators=(",", ":"))
        small_file(out, sid, key).write_text(f"window.SMALL_LOADED({json.dumps(sid)},{json.dumps(key)},{text});\n",
                                             encoding="utf-8")
        said.append(f"{key} {int(inside.sum())}/{len(props)}")
    return f"{sid}: {', '.join(said)} (in the section / all), {time.perf_counter() - started:.1f} s"


def write_index(out: Path) -> int:
    """``parts/small/index.js`` from the files there: per section and key, the small objects in the section."""
    index: dict[str, dict[str, int]] = {}
    for path in sorted((out / "parts" / "small").glob("*__gauss*-*.js")):
        sid, key = path.stem.rsplit("__", 1)
        flags = re.search(r'"in":\[([01,]*)\]', path.read_text(encoding="utf-8"))
        index.setdefault(sid, {})[key] = flags.group(1).count("1") if flags else 0
    (out / "parts" / "small").mkdir(parents=True, exist_ok=True)
    (out / "parts" / "small" / "index.js").write_text(
        "window.SMALL_INDEX = " + json.dumps(index, separators=(",", ":")) + ";\n", encoding="utf-8")
    return len(index)


def main(argv: list[str]) -> int:
    from . import section_browser as sb

    out = BROWSER
    limit = int(argv[0]) if argv and argv[0].isdigit() else None
    names = [json.loads(p.read_text(encoding="utf-8"))["name"] for p in sorted((out / "parts" / "methods").glob("*.json"))]
    todo = [n for n in names if not small_done(out, sb.short(n))]
    if limit:
        todo = todo[:limit]
    print(f"{len(names)} sections, {len(todo)} with too-small objects to trace", flush=True)
    started, failed = time.perf_counter(), 0
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(run_small, n, str(out)): n for n in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                msg = fut.result()
            except Exception as err:          # one bad file does not stop the rest
                failed += 1
                msg = f"{sb.short(futures[fut])}: FAILED {err!r}"
            print(f"[{i}/{len(todo)}, {(time.perf_counter() - started) / 60:.1f} min] {msg}", flush=True)
    print(f"too-small index: {write_index(out)} sections", flush=True)
    return 1 if failed else 0
