"""Full-resolution tiles for the section browser: the page shows them when you zoom in past its ~1.3 µm overview.

For each section of the browser (``parts/<id>.json``, written by ``section_browser.py``): the AF488 and AF555
maximum-intensity projections over z, at the scan's resolution (level 0, ~0.325 µm per pixel) and halved (level 1,
~0.65 µm), cut into :data:`TILE` px greyscale JPEGs. The display range is the overview's own (the tissue's
1st–99.7th percentile, from ``parts/<id>.json`` and ``parts/<id>_red.json``), so a tile matches the picture under
it; the page colours them green and red and applies its brightness sliders. Tiles without tissue are not written.

    tiles/<id>/<channel>/<level>/<row>_<col>.jpg       channel g (AF488) or r (AF555)
    tiles/<id>/tiles.js                                 which tiles exist; calls window.TILES_LOADED(id, data)

A section with its ``tiles.js`` is done, so a rerun skips it.

    python notebooks/official_analysis/browser_tiles.py            # every section not done yet
    python notebooks/official_analysis/browser_tiles.py 3          # just three, to time it

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
TILE = 512                        #: tile side, pixels of its level
LEVELS = (1, 2)                   #: downsampling of each level (1 = the scan's pixels); the overview is 4
JPEG_QUALITY = 82
TISSUE_MARGIN_PX = 8              #: overview pixels round the tissue still tiled (the outlines reach a little past it)
RED_HINT = "555"
WORKERS = 4                       #: each holds three full-resolution planes (~1.5 GB)
VERSION = 1
# --------------------------------------------------------------------------


def tiles_done(out: Path, sid: str) -> bool:
    return (out / "tiles" / sid / "tiles.js").exists()


def grey8(plane: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.round(np.clip((plane - lo) / max(hi - lo, 1e-9), 0, 1) * 255).astype(np.uint8)


def shrink(image: np.ndarray, d: int) -> np.ndarray:
    """Mean of each d × d block (edges cut), as the overview's block_mean."""
    if d == 1:
        return image
    h, w = image.shape[0] // d * d, image.shape[1] // d * d
    return image[:h, :w].reshape(h // d, d, w // d, d).mean(axis=(1, 3), dtype=np.float32)


def run_tiles(name: str, out: str) -> str:
    """One section's tiles, both channels and every level → tiles/<id>/."""
    from PIL import Image
    from scipy import ndimage

    import classical_vs_cellpose as cv
    from microscopy_viewer import junction as jn
    from microscopy_viewer import junction_analysis as ja
    from microscopy_viewer import loaders
    import section_browser as sb

    logging.getLogger("tifffile").setLevel(logging.ERROR)
    cv.below_normal_priority()
    out = Path(out)
    started = time.perf_counter()
    sid = sb.short(name)
    part = json.loads(sb.part_file(out, name).read_text(encoding="utf-8"))
    red_part = sb.red_file(out, name)
    red_display = json.loads(red_part.read_text(encoding="utf-8"))["display"] if red_part.exists() else None
    info = json.loads((RUN / "cells" / f"{name}.json").read_text(encoding="utf-8"))
    path = Path(info["file"])
    specs = loaders.load_path(path)
    try:
        names = [str(spec.channel_name or spec.name) for spec in specs]
        green_index = next(i for i, n in enumerate(names) if cv.LABEL_HINT in n)
        nuclear_index = jn.pick_channel(names, "", jn.NUCLEAR_HINTS)
        red_index = next((i for i, n in enumerate(names) if RED_HINT in n), None)
        first = specs[0].data[0] if specs[0].multiscale else specs[0].data
        whole = (slice(0, int(first.shape[-2])), slice(0, int(first.shape[-1])))
        planes = {"g": ja._plane(specs[green_index], whole).astype(np.float32)}
        if red_index is not None and red_display:
            planes["r"] = ja._plane(specs[red_index], whole).astype(np.float32)
        nuclear = ja._plane(specs[nuclear_index], whole)
    finally:
        loaders.release(path)
    read_s = time.perf_counter() - started
    f = int(part["f"])
    # the tissue on the overview's grid, widened a little: a tile is written when it touches it
    tissue = ndimage.binary_dilation(sb.small_tissue(sb.block_mean(nuclear, f)), iterations=TISSUE_MARGIN_PX)
    del nuclear
    displays = {"g": part["display"], "r": red_display}
    full_h, full_w = planes["g"].shape
    levels = []
    for d in LEVELS:
        have = {}
        for ch, plane in planes.items():
            small = grey8(shrink(plane, d), *displays[ch])
            rows, cols = -(-small.shape[0] // TILE), -(-small.shape[1] // TILE)
            kept = []
            for r in range(rows):
                for c in range(cols):
                    y0, x0 = r * TILE, c * TILE
                    y1, x1 = min(y0 + TILE, small.shape[0]), min(x0 + TILE, small.shape[1])
                    # the tile's footprint on the overview grid
                    ty0, ty1 = y0 * d // f, -(-y1 * d // f)
                    tx0, tx1 = x0 * d // f, -(-x1 * d // f)
                    if not tissue[ty0:ty1, tx0:tx1].any():
                        continue
                    target = out / "tiles" / sid / ch / str(d)
                    target.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(small[y0:y1, x0:x1], mode="L").save(target / f"{r}_{c}.jpg", quality=JPEG_QUALITY)
                    kept.append([r, c, x1 - x0, y1 - y0])
            have[ch] = kept
        levels.append({"d": d, "tile": TILE, "have": have})
    index = {"version": VERSION, "id": sid, "f": f, "full": [full_w, full_h], "levels": levels}
    (out / "tiles" / sid / "tiles.js").write_text(
        "window.TILES_LOADED(" + json.dumps(sid) + "," + json.dumps(index, separators=(",", ":")) + ");", encoding="utf-8")
    n = sum(len(v) for lv in levels for v in lv["have"].values())
    return f"{sid}: {n} tiles, {time.perf_counter() - started:.1f} s (reading {read_s:.1f} s)"


def main(argv: list[str]) -> int:
    import section_browser as sb

    out = BROWSER
    limit = int(argv[0]) if argv and argv[0].isdigit() else None
    names = [json.loads(p.read_text(encoding="utf-8"))["name"] for p in sorted((out / "parts").glob("*.json"))
             if not p.name.endswith("_red.json")]
    todo = [n for n in names if not tiles_done(out, sb.short(n))]
    if limit:
        todo = todo[:limit]
    print(f"{len(names)} sections, {len(todo)} to tile", flush=True)
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(run_tiles, n, str(out)): n for n in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                msg = fut.result()
            except Exception as err:          # one bad file does not stop the rest
                msg = f"{sb.short(futures[fut])}: FAILED {err!r}"
            print(f"[{i}/{len(todo)}, {(time.perf_counter() - started) / 60:.1f} min] {msg}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
