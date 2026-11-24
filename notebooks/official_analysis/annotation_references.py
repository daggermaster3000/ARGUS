"""Reference cells for the counting page: each section's brightest AF488⁺ cells, shown beside its squares.

An annotator judging a faint cell compares it with what a clearly positive cell looks like *in the same section*
(the same staining, scan and display range). For every section of the blinded square set (``annotation_squares.py``)
this picks the :data:`N_REF` brightest objects of the counting mask (``af488_mask/<section>/objects.parquet``, mask
``kmad8``) and crops each, :data:`REF_UM` across, centred on it:

- brightness: the mean AF488 (maximum-intensity projection over z) within :data:`DISC_UM` of the object's centroid;
- candidates: objects of :data:`AREA_UM2` (a soma, not a speck nor a bright blob of debris), not inside any of the
  section's squares (crop and margin: the reference must not show a cell the annotator is asked to count) and at
  least half a crop from the scan's edge;
- the AF488 crop takes the section's display range, as its squares do; the nuclear crop its own 1st–99.8th percentile.

Written into ``annotation_v2``: ``images/ref_<token>_<i>_488.png`` / ``_405.png`` (``token`` a random name per section,
so file names say nothing about the section) and, in ``squares.js``, a ``refs`` list on every square of that section.
The squares themselves (codes, order, images, frames) are left exactly as they were, so earlier counts stay valid.

    python notebooks/official_analysis/annotation_references.py

No Qt.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import annotation_squares as aq  # noqa: E402

N_REF = 4                         #: reference cells per section
REF_UM = 30.0                     #: side of a reference crop (µm)
DISC_UM = 2.5                     #: brightness: mean AF488 within this radius of the centroid
AREA_UM2 = (20.0, 300.0)          #: a reference is a soma-sized object
SEED = 20261124


def brightness(plane: np.ndarray, y_px: float, x_px: float, r_px: float) -> float:
    """Mean of *plane* in a disc of *r_px* round (y, x)."""
    y0, x0 = int(round(y_px)), int(round(x_px))
    r = int(np.ceil(r_px))
    win = plane[max(y0 - r, 0):y0 + r + 1, max(x0 - r, 0):x0 + r + 1].astype(np.float32)
    yy, xx = np.mgrid[max(y0 - r, 0):max(y0 - r, 0) + win.shape[0], max(x0 - r, 0):max(x0 - r, 0) + win.shape[1]]
    inside = (yy - y_px) ** 2 + (xx - x_px) ** 2 <= r_px ** 2
    return float(win[inside].mean()) if inside.any() else np.nan


def main() -> int:
    import classical_vs_cellpose as cv

    rng = random.Random(SEED)
    key = pd.read_csv(aq.KEY)
    data = json.loads((aq.BROWSER / "data.json").read_text(encoding="utf-8"))
    display = {s["name"]: s["display"] for s in data["sections"]}
    js = aq.OUT / "squares.js"
    text = js.read_text(encoding="utf-8")
    squares = json.loads(text[text.index("=") + 1:].strip().rstrip(";"))
    refs: dict[str, list] = {}
    for n_s, (name, g) in enumerate(key.groupby("name"), 1):
        info = json.loads((aq.RUN / "cells" / f"{name}.json").read_text(encoding="utf-8"))
        af488, nuclear, voxel = cv.read_planes(Path(info["file"]))
        o = pd.read_parquet(aq.MASKS / name / "objects.parquet")
        o = o[(o["mask"] == "kmad8") & o["Area (µm²)"].between(*AREA_UM2)]
        y, x = o["Centroid Y (µm)"].to_numpy(float), o["Centroid X (µm)"].to_numpy(float)
        half = REF_UM / 2
        keep = (y > half) & (x > half) & (y < af488.shape[0] * voxel - half) & (x < af488.shape[1] * voxel - half)
        for _, k in g.iterrows():                           # not in any of the section's squares (crop + margin)
            side = k["frame_um"] + 2 * aq.MARGIN_UM
            keep &= ~((y > k["top_um"] - half) & (y < k["top_um"] + side + half) &
                      (x > k["left_um"] - half) & (x < k["left_um"] + side + half))
        y, x = y[keep], x[keep]
        b = np.array([brightness(af488, yy / voxel, xx / voxel, DISC_UM / voxel) for yy, xx in zip(y, x)])
        order = np.argsort(-b)[:N_REF]
        token = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(6))
        size = int(round(REF_UM / voxel))
        lo, hi = display[name]
        out = []
        for i, j in enumerate(order):
            top, left = int(round(y[j] / voxel - size / 2)), int(round(x[j] / voxel - size / 2))
            a = aq.crop(af488, top, left, size)
            nu = aq.crop(nuclear.astype(np.float32), top, left, size)
            stem = f"images/ref_{token}_{i}"
            aq.to_png(a, lo, hi, aq.OUT / f"{stem}_488.png")
            nlo, nhi = np.percentile(nu[nu > 0], (1, 99.8)) if (nu > 0).any() else (0, 1)
            aq.to_png(nu, nlo, nhi, aq.OUT / f"{stem}_405.png")
            out.append({"img488": f"{stem}_488.png", "img405": f"{stem}_405.png", "px": size, "um_per_px": voxel})
        refs[name] = out
        del af488, nuclear
        print(f"[{n_s}/{key['name'].nunique()}] {name}: {len(out)} references of {len(b)} candidates", flush=True)
    by_code = key.set_index("code")["name"]
    for q in squares:
        q["refs"] = refs.get(by_code[q["code"]], [])
    js.write_text("window.SQUARES = " + json.dumps(squares, separators=(",", ":")) + ";\n", encoding="utf-8")
    print(f"refs on {sum(bool(q['refs']) for q in squares)} of {len(squares)} squares -> {js}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
