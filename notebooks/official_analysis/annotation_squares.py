"""A blinded re-annotation set: random 120 µm squares for two annotators to count AF488⁺ cells in, the same squares each.

Why: the first human counts (annotator 1: 14 whole sections; annotator 2: 13 sections near the junction) were counted
on different sections and areas, with no shared protocol, and turned out inconsistent. This set lets both annotators
count the *same* squares under one written protocol (``PROTOCOL.md``), blind to genotype, experiment, section and the
automated result, so that inter-rater agreement can be measured and the automated count compared with a consensus.

Design (stratified random sample, fixed seed):

- per genotype (WT3, GD1) × experiment (8, 9, 10): :data:`SECTIONS_PER_CELL` sections drawn at random from the 614
  analysed sections (``section_browser/data.json``);
- per section one square in each compartment (:data:`COMPARTMENTS`): *home* (home side, outside the junction region),
  *junction* (centre in the junction region) and *crossed* (far side of the junction interface, outside the region).
  The square's centre is drawn at random among the overview pixels of that compartment whose whole square lies in the
  section (outline eroded by half a square; relaxed to a quarter when nothing fits). Squares are not chosen for their
  content, so empty squares are part of the sample;
- each square is shown with a :data:`MARGIN_UM` margin of context around the 120 µm counting frame.

Written into ``OUT`` (``run_analysis_20261028/annotation_v2``):

- ``images/<code>_488.png`` and ``<code>_405.png``: AF488 (the section's display range, as the browser) and the nuclear
  stain (its own 1st–99.8th percentile), maximum-intensity projection over z, at the scan's pixels (0.325 µm);
- ``squares.js``: ``window.SQUARES`` — per square only its random code, image names and geometry, in a shuffled order;
- ``../annotation_v2_key.csv`` (:data:`KEY`, outside the folder): code → section, genotype, experiment, compartment,
  square position (µm). **Not for the annotators.**
- ``annotate.html`` and ``PROTOCOL.md``, copied from beside this script.

    python notebooks/official_analysis/annotation_squares.py

No Qt.
"""
from __future__ import annotations

import json
import random
import shutil
import string
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "notebooks" / "validation"))
sys.path.insert(0, str(ROOT / "notebooks" / "dataset"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

# ---------------------------------------------------------------- Settings
DATA = Path(r"D:\Slidescanner extracted")
RUN = DATA / "cells_analysis" / "2026-10-28_090440 re-scored"
ANALYSIS = DATA / "run_analysis_20261028"
MASKS = ANALYSIS / "af488_mask"
BROWSER = ANALYSIS / "section_browser"
OUT = ANALYSIS / "annotation_v2"                 #: the folder to hand to the annotators
KEY = ANALYSIS / "annotation_v2_key.csv"         #: code -> section; kept outside it, so the set stays blind
SEED = 20261123
SQUARE_UM = 120.0                 #: the counting frame
MARGIN_UM = 20.0                  #: context shown round it (not counted)
SECTIONS_PER_CELL = 8             #: per genotype × experiment: 2 × 3 × 8 = 48 sections, 144 squares
COMPARTMENTS = ("home", "junction", "crossed")
# --------------------------------------------------------------------------


def compartment_map(name: str, info: dict, sign: float, outline: dict) -> dict[str, np.ndarray]:
    """Per compartment, the overview pixels (rows, cols) a square's centre may take."""
    from scipy import ndimage
    import section_outline
    import sides
    from microscopy_viewer import junction_analysis as ja

    filled, step, um = outline["filled"], outline["step"], outline["um"]
    voxel = float(info["voxel_yx"][0])
    band = ja.read_labels(RUN / info["region_file"]) > 0
    rows, cols = np.mgrid[0:filled.shape[0], 0:filled.shape[1]]
    centre_px = (np.stack([rows, cols], -1) * step + (step - 1) / 2)              # overview pixel -> scan pixel
    in_band = band[np.clip(centre_px[..., 0].astype(int), 0, band.shape[0] - 1),
                   np.clip(centre_px[..., 1].astype(int), 0, band.shape[1] - 1)]
    line = section_outline.line_um(info)
    away = sides.away((centre_px * voxel).reshape(-1, 2), line, sign).reshape(filled.shape)
    out = {}
    for shrink in (0.5, 0.25):
        inside = ndimage.binary_erosion(filled, iterations=max(1, int(round(SQUARE_UM * shrink / um))))
        cand = {"junction": inside & in_band, "crossed": inside & ~in_band & (away > 0),
                "home": inside & ~in_band & (away <= 0)}
        for k, m in cand.items():
            if k not in out and m.any():
                out[k] = np.argwhere(m)
    return out


def crop(plane: np.ndarray, top: int, left: int, size: int) -> np.ndarray:
    out = np.zeros((size, size), dtype=np.float32)
    t, l = max(top, 0), max(left, 0)
    b, r = min(top + size, plane.shape[0]), min(left + size, plane.shape[1])
    if b > t and r > l:
        out[t - top:b - top, l - left:r - left] = plane[t:b, l:r]
    return out


def to_png(img: np.ndarray, lo: float, hi: float, path: Path) -> None:
    from PIL import Image

    Image.fromarray((np.clip((img - lo) / max(hi - lo, 1e-9), 0, 1) * 255).astype(np.uint8)).save(path)


def main() -> int:
    import classical_vs_cellpose as cv
    import section_outline

    rng = random.Random(SEED)
    data = json.loads((BROWSER / "data.json").read_text(encoding="utf-8"))
    sections = data["sections"]
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    chosen = []
    for g in ("WT3", "GD1"):
        for e in ("8", "9", "10"):
            pool = [s for s in sections if s["genotype"] == g and str(s["experiment"]) == e and s["sign"] is not None]
            chosen += rng.sample(pool, min(SECTIONS_PER_CELL, len(pool)))
    codes = set()
    squares, key = [], []
    for n_s, s in enumerate(chosen, 1):
        name = s["name"]
        info = json.loads((RUN / "cells" / f"{name}.json").read_text(encoding="utf-8"))
        outline = section_outline.load_outline(MASKS / name)
        where = compartment_map(name, info, float(s["sign"]), outline)
        af488, nuclear, voxel = cv.read_planes(Path(info["file"]))
        side = int(round((SQUARE_UM + 2 * MARGIN_UM) / voxel))
        margin = int(round(MARGIN_UM / voxel))
        lo, hi = s["display"]
        for comp in COMPARTMENTS:
            if comp not in where:
                print(f"{name}: no {comp} square fits", flush=True)
                continue
            r, c = where[comp][rng.randrange(len(where[comp]))]
            cy = (r * outline["step"] + (outline["step"] - 1) / 2)                # scan pixels
            cx = (c * outline["step"] + (outline["step"] - 1) / 2)
            top = int(round(cy - side / 2))
            left = int(round(cx - side / 2))
            code = None
            while code is None or code in codes:
                code = "".join(rng.choice(string.ascii_uppercase) for _ in range(2)) + str(rng.randrange(100, 1000))
            codes.add(code)
            a = crop(af488, top, left, side)
            nu = crop(nuclear.astype(np.float32), top, left, side)
            to_png(a, lo, hi, OUT / "images" / f"{code}_488.png")
            nlo, nhi = np.percentile(nu[nu > 0], (1, 99.8)) if (nu > 0).any() else (0, 1)
            to_png(nu, nlo, nhi, OUT / "images" / f"{code}_405.png")
            squares.append({"code": code, "img488": f"images/{code}_488.png", "img405": f"images/{code}_405.png",
                            "px": side, "frame": [margin, margin, side - margin, side - margin], "um_per_px": voxel})
            key.append({"code": code, "name": name, "id": s["id"], "genotype": s["genotype"],
                        "experiment": s["experiment"], "slide": s["slide"], "compartment": comp,
                        "top_um": round(top * voxel, 2), "left_um": round(left * voxel, 2),
                        "frame_top_um": round((top + margin) * voxel, 2), "frame_left_um": round((left + margin) * voxel, 2),
                        "frame_um": SQUARE_UM})
        del af488, nuclear
        print(f"[{n_s}/{len(chosen)}] {s['id']} ({s['genotype']} e{s['experiment']}): {len(where)} compartments", flush=True)
    rng.shuffle(squares)                                    # the order annotators see: no section or genotype runs
    (OUT / "squares.js").write_text("window.SQUARES = " + json.dumps(squares, separators=(",", ":")) + ";\n",
                                    encoding="utf-8")
    import csv
    with open(KEY, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(key[0]))
        w.writeheader()
        w.writerows(key)
    for f in ("annotate.html", "PROTOCOL.md"):
        shutil.copy(HERE / f, OUT / f)
    print(f"{len(squares)} squares from {len(chosen)} sections -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
