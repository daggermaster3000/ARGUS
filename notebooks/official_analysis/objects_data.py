"""Every counted AF488⁺ object, for the figure builder's distribution panel: ``section_browser/objects.js``.

One row per object of the counting method (AF488 mask, Gaussian 20 µm background, median + 5 robust SD: the
stored ``af488_mask/<section>/objects.parquet`` rows with ``mask`` == "kmad5") of every section in the section
browser's ``data.json`` (WT3 and GD1), with:

- ``away`` — signed distance past the junction line (µm, ``sides.py``: the line extended past both ends; > 0
  crossed, in the organoid where AF488 is not at home; the section's home sign as in ``data.json``);
- ``area`` — the object's area (µm²);
- ``reg`` — 1 when its centroid lies in the junction region (the stored ``In region``);
- ``edge`` — distance to the organoid edge (µm): that of the nearest segmented cell of the run's cell table
  (``Distance to the tissue edge (µm)``), as ``02_biology.ipynb`` does for the crossed objects;
- ``edge_rel`` — ``edge`` ÷ the organoid's local half-width at the same depth past the line (99th percentile of
  the cells' distance to the edge per 25 µm step over the far side, ≥ 20 cells; as 02_biology), clipped to 0–1,
  for crossed objects only.

Per section (``sections``, ``home_depth``): its id and the distance from the line to the centroid of its
home-side objects (µm; the mean of their centroids, None when it does not lie on the home side), so that ``away``
can be divided by how deep the home organoid's labelled cells sit.

The file is ``window.OBJECT_DATA = {...}`` with column arrays (distances rounded to 0.1 µm) and ``sec``, each
object's index into ``sections``. The figure builder loads it only when its distribution panel shows objects.

    python notebooks/official_analysis/objects_data.py [--out <section_browser folder>]

``section_browser.py`` runs it at the end of every rebuild (after it writes ``data.json``), so the file follows the
browser's sections; run it alone after changing only the objects or the junction lines (~1 min).

No Qt.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sides  # noqa: E402

# ---------------------------------------------------------------- Settings
DATA = Path(r"D:\Slidescanner extracted")
RUN = DATA / "cells_analysis" / "2026-10-28_090440 re-scored"     # the cells-analysis run (614 sections)
OUT = DATA / "run_analysis_20261028"
MASKS = OUT / "af488_mask"
BROWSER = OUT / "section_browser"
METHOD_MASK = "kmad5"                     #: the counting method's stored objects
EDGE_COLUMN = "Distance to the tissue edge (µm)"
WIDTH_STEP_UM, WIDTH_MIN_CELLS = 25.0, 20  #: local half-width per 25 µm step past the line (02_biology)
# --------------------------------------------------------------------------


def half_widths(away: np.ndarray, edge: np.ndarray):
    """Per 25 µm step past the line, over the far side: the 99th percentile of the cells' distance to the edge."""
    steps = np.arange(0, max(np.nanmax(away, initial=0), 0) + 2 * WIDTH_STEP_UM, WIDTH_STEP_UM)
    step = np.digitize(away, steps) - 1
    out = np.full(len(steps) - 1, np.nan)
    for k in range(len(out)):
        here = (away > 0) & (step == k) & (edge > 0)
        if here.sum() >= WIDTH_MIN_CELLS:
            out[k] = np.percentile(edge[here], 99)
    return steps, out


def width_at(away: np.ndarray, steps: np.ndarray, widths: np.ndarray) -> np.ndarray:
    step = np.digitize(away, steps) - 1
    ok = (step >= 0) & (step < len(widths))
    out = np.full(len(away), np.nan)
    out[ok] = widths[step[ok]]
    return out


def section_objects(name: str, sign: float | None) -> tuple[pd.DataFrame, float]:
    """One section's objects with away, area, reg, edge, edge_rel; and its home depth (µm, NaN if none)."""
    info = json.loads((RUN / "cells" / f"{name}.json").read_text(encoding="utf-8"))
    line = info.get("line")
    line = np.asarray(json.loads(line) if isinstance(line, str) else line, float) if line else None
    o = pd.read_parquet(MASKS / name / "objects.parquet")
    o = o[o["mask"] == METHOD_MASK].reset_index(drop=True)
    yx = o[["Centroid Y (µm)", "Centroid X (µm)"]].to_numpy(float)
    frame = pd.DataFrame({"area": o["Area (µm²)"].to_numpy(float), "reg": o["In region"].fillna(False).astype(int)})
    if line is None or sign is None or not len(o):
        frame["away"] = np.nan
        frame["edge"], frame["edge_rel"] = np.nan, np.nan
        return frame, np.nan
    away = sides.away(yx, line, sign)
    frame["away"] = away
    cells = pd.read_csv(RUN / "cells" / f"{name}.csv", usecols=["Centroid Y (µm)", "Centroid X (µm)", EDGE_COLUMN],
                        engine="pyarrow")
    cyx = cells[["Centroid Y (µm)", "Centroid X (µm)"]].to_numpy(float)
    cedge = cells[EDGE_COLUMN].to_numpy(float)
    _, nearest = cKDTree(cyx).query(yx)
    frame["edge"] = cedge[nearest]
    steps, widths = half_widths(sides.away(cyx, line, sign), cedge)
    rel = np.clip(frame["edge"].to_numpy() / width_at(away, steps, widths), 0, 1)
    frame["edge_rel"] = np.where(away > 0, rel, np.nan)
    home = away < 0
    depth = np.nan
    if home.any():
        centre = yx[home].mean(axis=0)
        d = -float(sides.away(centre[None, :], line, sign)[0])
        depth = d if d > 0 else np.nan
    return frame, depth


def rounded(values, digits: int) -> list:
    return [None if not np.isfinite(v) else round(float(v), digits) for v in np.asarray(values, float)]


def main(argv: list[str]) -> int:
    out = Path(argv[argv.index("--out") + 1]) if "--out" in argv else BROWSER
    data = json.loads((out / "data.json").read_text(encoding="utf-8"))
    started = time.perf_counter()
    ids, depths, parts = [], [], []
    for index, s in enumerate(data["sections"]):
        frame, depth = section_objects(s["name"], s["sign"])
        frame["sec"] = index
        ids.append(s["id"])
        depths.append(depth)
        parts.append(frame)
    table = pd.concat(parts, ignore_index=True)
    payload = {
        "generated": time.strftime("%Y-%m-%d %H:%M"), "method": "gauss-5", "mask": METHOD_MASK,
        "sections": ids, "home_depth": rounded(depths, 1),
        "sec": table["sec"].astype(int).tolist(), "away": rounded(table["away"], 1),
        "area": rounded(table["area"], 1), "reg": table["reg"].astype(int).tolist(),
        "edge": rounded(table["edge"], 1), "edge_rel": rounded(table["edge_rel"], 3),
    }
    text = json.dumps(payload, separators=(",", ":"), allow_nan=False)
    target = out / "objects.js"
    target.write_text("window.OBJECT_DATA = " + text + ";\n", encoding="utf-8")
    print(f"{len(table)} objects in {len(ids)} sections -> {target} ({target.stat().st_size / 1e6:.1f} MB, "
          f"{time.perf_counter() - started:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
