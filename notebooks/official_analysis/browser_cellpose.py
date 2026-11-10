"""The retrained Cellpose's cells for the section browser: its outlines on the human-counted sections.

``notebooks/dataset/cellpose488_retrain.py`` fine-tuned Cellpose on the annotators' AF488⁺ dots and predicted every
human-counted section (27) whole, twice: by a model that never saw the section (cross-validated) and by one trained
on the other annotator's sections. Its label maps (``cellpose_retrained/labels/<section> <variant>.tif``) hold the
objects it kept (centre in the tissue); here each is traced at the scan's pixels (as ``browser_outlines.py``) and
placed: crossed or home (``sides.py``, the section's home sign from ``parts/methods/<id>.json``), in the junction
region (the stored ``region``), area, and whether it lies in the section (``section_outline``: objects outside it
are drawn but not counted anywhere).

    parts/cellpose/<id>__<variant>.js   window.CP_LOADED(id, variant, {"q", "rings", "side", "reg", "area", "away"})
    parts/cellpose/index.js             window.CP_INDEX = {id: {variant: number of cells}}

``variant`` is ``cv`` (cross-validated) or ``other`` (trained on the other annotator). ``side``: 1 crossed, 0 home,
-1 no junction line, -2 outside the section. Rings as ``browser_outlines.fine_rings``. ~1 min for all.

    python notebooks/official_analysis/browser_cellpose.py

No Qt.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "notebooks" / "dataset"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
import browser_outlines  # noqa: E402
import section_outline  # noqa: E402
import sides  # noqa: E402

# ---------------------------------------------------------------- Settings
DATA = Path(r"D:\Slidescanner extracted")
RUN = DATA / "cells_analysis" / "2026-10-28_090440 re-scored"
OUT = DATA / "run_analysis_20261028"
LABELS = OUT / "cellpose_retrained" / "labels"
MASKS = OUT / "af488_mask"
BROWSER = OUT / "section_browser"
ANNOTATOR_FOLDERS = (OUT / "james_counts", OUT / "quillan_counts")
#: label-map suffix → (the browser's key, the method's name in the objects tables)
VARIANTS = {
    "cross-validated": ("cv", "Cellpose AF488 · retrained (cross-validated)"),
    "other annotator": ("other", "Cellpose AF488 · retrained on the other annotator"),
}
PREFIX = "2025_09_26_Asssembloid_"
VERSION = 1
# --------------------------------------------------------------------------


def objects_table(name: str) -> pd.DataFrame | None:
    """The retrained Cellpose's kept objects of *name* (both variants), from whichever annotator counted it."""
    for folder in ANNOTATOR_FOLDERS:
        path = folder / "methods" / "objects" / "cellpose488_retrained" / f"{name}.parquet"
        if path.exists():
            return pd.read_parquet(path)
    return None


def one_section(name: str, out: Path) -> dict[str, int]:
    import tifffile
    from skimage.measure import regionprops

    sid = name.replace(PREFIX, "")
    info = json.loads((RUN / "cells" / f"{name}.json").read_text(encoding="utf-8"))
    voxel = float(info["voxel_yx"][0])
    f = int(json.loads((out / "parts" / f"{sid}.json").read_text(encoding="utf-8"))["f"])   # the overview's step
    methods = json.loads((out / "parts" / "methods" / f"{sid}.json").read_text(encoding="utf-8"))
    sign = methods.get("sign")
    line = section_outline.line_um(info)
    outline = section_outline.load_outline(MASKS / name)
    table = objects_table(name)
    counts = {}
    for suffix, (key, method) in VARIANTS.items():
        path = LABELS / f"{name} {suffix}.tif"
        if not path.exists() or table is None:
            continue
        rows = table[table["method"] == method].set_index("object")
        labels = tifffile.imread(path)
        props = [p for p in regionprops(labels) if p.label in rows.index]
        yx = np.array([p.centroid for p in props], float).reshape(-1, 2) * voxel
        away = sides.away(yx, line, sign) if sign is not None and line is not None and len(yx) else np.full(len(yx), np.nan)
        inside = section_outline.in_section(outline, line, yx, voxel)
        side = np.where(np.isnan(away), -1, (away > 0).astype(int))
        side = np.where(inside, side, -2)
        payload = {
            "version": VERSION, "q": 2 * f,
            "rings": [browser_outlines.fine_rings(p.image, p.bbox[0], p.bbox[1]) for p in props],
            "side": side.astype(int).tolist(),
            "reg": [int(bool(rows.at[p.label, "region"])) for p in props],
            "area": [int(round(p.area * voxel ** 2)) for p in props],
            "away": [None if not np.isfinite(a) else round(float(a), 1) for a in away],
        }
        target = out / "parts" / "cellpose" / f"{sid}__{key}.js"
        target.write_text(f"window.CP_LOADED({json.dumps(sid)},{json.dumps(key)},"
                          f"{json.dumps(payload, separators=(',', ':'))});\n", encoding="utf-8")
        counts[key] = len(props)
    return counts


def main(argv: list[str]) -> int:
    out = Path(argv[argv.index("--out") + 1]) if "--out" in argv else BROWSER
    (out / "parts" / "cellpose").mkdir(parents=True, exist_ok=True)
    names = sorted({p.stem[: -len(suffix) - 1] for p in LABELS.glob("*.tif") for suffix in VARIANTS
                    if p.stem.endswith(" " + suffix)})
    started = time.perf_counter()
    index = {}
    for name in names:
        if not (out / "parts" / "methods" / f"{name.replace(PREFIX, '')}.json").exists():
            continue
        counts = one_section(name, out)
        if counts:
            index[name.replace(PREFIX, "")] = counts
            print(name.replace(PREFIX, ""), counts, flush=True)
    (out / "parts" / "cellpose" / "index.js").write_text(
        "window.CP_INDEX = " + json.dumps(index, separators=(",", ":")) + ";\n", encoding="utf-8")
    print(f"{len(index)} sections -> {out / 'parts' / 'cellpose'} ({time.perf_counter() - started:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
