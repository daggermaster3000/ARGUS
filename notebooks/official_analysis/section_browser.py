"""Data and pictures for the section browser: every analysed section, what the counting method labelled.

For each of the run's sections (WT3 and GD1 only):

- a picture of the whole section, AF488 (brightest plane over z) at ~1.3 µm per pixel, green on
  black, its display range from the tissue's 1st–99.7th percentile; and a ~300 px thumbnail;
- the overlays, in picture pixels: the junction line, the junction region's outline, every object
  of the AF488 mask (median + 5 robust SD) with its radius and whether it crossed or is at home,
  and the human annotators' dots where they exist;
- the section's numbers: objects in total, in the junction region, crossed and at home, crossed
  per mm of junction, and on the human-counted sections the human and the automated total in the
  area the human counted (as ``01_validation.ipynb``).

A crossed object lies beside the junction line, 2–300 µm past it into the organoid where AF488 is
not at home; a home object 2–300 µm before it. Home is the side holding most of the run's AF488+
cells beside the line (``notebooks/dataset/04``).

Writes into *out* (default ``run_analysis_20261028/section_browser``) ``data.json``, the same as
``data.js`` (so the page opens from disk too), ``img/`` and ``parts/`` (one file per section, so a
rerun skips what is done), beside a copy of ``section_browser.html``.

    python notebooks/official_analysis/section_browser.py            # every section not done yet
    python notebooks/official_analysis/section_browser.py 5          # just five, to time it
    python notebooks/official_analysis/section_browser.py --out <folder>

Then ``python -m http.server`` in *out* and open http://localhost:8000/section_browser.html, or open
``section_browser.html`` directly.
"""
from __future__ import annotations

import json
import logging
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "notebooks" / "validation"))
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------- Settings
DATA = Path(r"D:\Slidescanner extracted")
RUN = DATA / "cells_analysis" / "2026-10-28_090440 re-scored"     # the cells-analysis run (614 sections)
OUT = DATA / "run_analysis_20261028"                               # caches written by notebooks/dataset
MASKS = OUT / "af488_mask"                                         # af488_masks.py
CACHE = OUT / "cache"                                              # the run's cells near the line (dataset/03)
DESIGN_CSV = DATA / "Image_File_List.csv"
BROWSER = OUT / "section_browser"
GENOTYPES = ("WT3", "GD1")
CHANNEL = "AF488"
METHOD_MASK = "kmad5"             #: the counting method: AF488 mask objects, median + 5 robust SD
GAP_UM, NEAR_UM = 2.0, 300.0      #: crossed = beside the line, 2–300 µm past it; home = 2–300 µm before it
AREA_2_UM = -50.0                 #: annotator 2 counted beside the line from 50 µm before it to the far edge
ANNOTATORS = {"human annotator 1": OUT / "james_counts", "human annotator 2": OUT / "quillan_counts"}
TARGET_UM = 1.3                   #: picture pixel size (whole pixels of the scan are averaged)
THUMB_PX = 300                    #: longest side of a thumbnail
DISPLAY_PCT = (1.0, 99.7)         #: display range: these percentiles of AF488 inside the tissue
JPEG_QUALITY = 85
WORKERS = 4                       #: each holds a slide scan's two planes (~0.5 GB)
VERSION = 1
PREFIX = "2025_09_26_Asssembloid_"
# --------------------------------------------------------------------------


def short(name: str) -> str:
    return name.replace(PREFIX, "")


def home_sign(name: str) -> float:
    """+1 or −1 so that distance × sign is + past the line, in the organoid where AF488 is not at home."""
    c = pd.read_parquet(CACHE / f"{name}.parquet", columns=["d", "beside", f"{CHANNEL} positive"])
    pos = c[f"{CHANNEL} positive"].to_numpy(bool) & c["beside"].to_numpy(bool)
    return -1.0 if pos.sum() >= 5 and np.median(c.loc[pos, "d"]) > 0 else 1.0


def block_mean(image: np.ndarray, f: int) -> np.ndarray:
    """*image* shrunk by *f*, each picture pixel the mean of an f × f block (edges cut)."""
    h, w = image.shape[0] // f * f, image.shape[1] // f * f
    return image[:h, :w].reshape(h // f, f, w // f, f).mean(axis=(1, 3), dtype=np.float32)


def small_tissue(nuclear_small: np.ndarray) -> np.ndarray:
    """The section from the shrunk nuclear stain: Otsu, closed, holes filled (as classical_vs_cellpose.tissue_mask)."""
    from scipy import ndimage
    from skimage.filters import threshold_otsu

    mask = nuclear_small > threshold_otsu(nuclear_small)
    mask = ndimage.binary_closing(mask, iterations=6)
    return ndimage.binary_fill_holes(mask)


def to_px(y_um, x_um, voxel: float, f: int):
    """µm (index × voxel, as the run stores centroids) → picture pixel coordinates (0 at the picture's corner)."""
    return (np.asarray(y_um, float) / voxel + 0.5) / f, (np.asarray(x_um, float) / voxel + 0.5) / f


def part_file(out: Path, name: str) -> Path:
    return out / "parts" / f"{short(name)}.json"


def done(out: Path, name: str) -> bool:
    try:
        part = json.loads(part_file(out, name).read_text(encoding="utf-8"))
        return part.get("version", 0) >= VERSION and (out / "img" / f"{part['id']}.jpg").exists()
    except Exception:
        return False


def run_one(name: str, out: str) -> str:
    """Picture, thumbnail and overlays of one section → img/ and parts/."""
    from PIL import Image
    from skimage import measure

    import classical_vs_cellpose as cv
    from microscopy_viewer import junction_analysis as ja

    logging.getLogger("tifffile").setLevel(logging.ERROR)
    cv.below_normal_priority()
    out = Path(out)
    started = time.perf_counter()
    info = json.loads((RUN / "cells" / f"{name}.json").read_text(encoding="utf-8"))
    af488, nuclear, voxel = cv.read_planes(Path(info["file"]))
    read_s = time.perf_counter() - started
    f = max(1, int(round(TARGET_UM / voxel)))
    picture = block_mean(af488, f)
    del af488
    tissue = small_tissue(block_mean(nuclear, f))
    del nuclear
    lo, hi = np.percentile(picture[tissue] if tissue.any() else picture, DISPLAY_PCT)
    grey = np.clip((picture - lo) / max(hi - lo, 1e-9), 0, 1)
    rgb = np.zeros((*grey.shape, 3), np.uint8)
    rgb[..., 1] = np.round(grey * 255).astype(np.uint8)
    sid = short(name)
    image = Image.fromarray(rgb)
    image.save(out / "img" / f"{sid}.jpg", quality=JPEG_QUALITY)
    thumb = image.copy()
    thumb.thumbnail((THUMB_PX, THUMB_PX), Image.Resampling.LANCZOS)
    thumb.save(out / "img" / f"{sid}_t.jpg", quality=JPEG_QUALITY)

    def path_px(points_um):
        y, x = to_px(points_um[:, 0], points_um[:, 1], voxel, f)
        return [[round(float(a), 1), round(float(b), 1)] for a, b in zip(x, y)]       # stored as (x, y)

    line = path_px(np.asarray(info["line"], float)) if info.get("line") else None
    region = []
    if info.get("region_file") and (RUN / info["region_file"]).exists():
        band = ja.read_labels(RUN / info["region_file"]) > 0
        small = block_mean(band.astype(np.float32), f) >= 0.5
        for contour in measure.find_contours(np.pad(small, 1).astype(float), 0.5):
            contour = measure.approximate_polygon(contour - 1, 0.8)
            if len(contour) >= 4:
                # find_contours runs through pixel centres: +0.5 to the picture's pixel-edge coordinates
                region.append([[round(float(c) + 0.5, 1), round(float(r) + 0.5, 1)] for r, c in contour])

    objects = pd.read_parquet(MASKS / name / "objects.parquet")
    objects = objects[objects["mask"] == METHOD_MASK]
    y, x = to_px(objects["Centroid Y (µm)"], objects["Centroid X (µm)"], voxel, f)
    radius = np.sqrt(objects["Area (µm²)"].to_numpy(float) / np.pi) / (voxel * f)
    sign = home_sign(name)
    away = objects["Distance to the junction line (µm)"].to_numpy(float) * sign
    beside = objects["Beside the junction line"].fillna(False).to_numpy(bool)
    kind = np.where(beside & (away > GAP_UM) & (away <= NEAR_UM), 1,
                    np.where(beside & (away >= -NEAR_UM) & (away < -GAP_UM), 2, 0))   # 1 crossed, 2 home, 0 other
    part = {
        "version": VERSION, "id": sid, "name": name, "voxel": voxel, "f": f, "um_per_px": round(voxel * f, 4),
        "width": int(rgb.shape[1]),
        "height": int(rgb.shape[0]), "display": [round(float(lo), 1), round(float(hi), 1)], "sign": sign,
        "line": line, "region": region,
        # objects as columns: x, y, radius (px), kind, in region, away (µm)
        "objects": {"x": np.round(x, 1).tolist(), "y": np.round(y, 1).tolist(), "r": np.round(radius, 2).tolist(),
                    "k": kind.tolist(), "reg": objects["In region"].fillna(False).astype(int).tolist(),
                    "away": [None if not np.isfinite(a) else round(float(a), 1) for a in away]},
        "seconds": round(time.perf_counter() - started, 1), "read_seconds": round(read_s, 1),
    }
    part_file(out, name).write_text(json.dumps(part, separators=(",", ":")), encoding="utf-8")
    return f"{sid}: {part['seconds']:.1f} s (reading {read_s:.1f} s), {len(objects)} objects"


RED_HINT = "555"                  #: the second label's channel (AF555), shown in red


def red_file(out: Path, name: str) -> Path:
    return out / "parts" / f"{short(name)}_red.json"


def run_red(name: str, out: str) -> str:
    """The AF555 picture of one section (red on black, same grid as the AF488 one) → img/<id>_r.jpg."""
    from PIL import Image

    import classical_vs_cellpose as cv
    from microscopy_viewer import junction as jn
    from microscopy_viewer import junction_analysis as ja
    from microscopy_viewer import loaders

    logging.getLogger("tifffile").setLevel(logging.ERROR)
    cv.below_normal_priority()
    out = Path(out)
    started = time.perf_counter()
    info = json.loads((RUN / "cells" / f"{name}.json").read_text(encoding="utf-8"))
    path = Path(info["file"])
    specs = loaders.load_path(path)
    try:
        names = [str(spec.channel_name or spec.name) for spec in specs]
        red_index = next(i for i, n in enumerate(names) if RED_HINT in n)
        nuclear_index = jn.pick_channel(names, "", jn.NUCLEAR_HINTS)
        voxel = float(specs[0].scale[-1])
        first = specs[0].data[0] if specs[0].multiscale else specs[0].data
        whole = (slice(0, int(first.shape[-2])), slice(0, int(first.shape[-1])))
        red = ja._plane(specs[red_index], whole).astype(np.float32)
        nuclear = ja._plane(specs[nuclear_index], whole)
    finally:
        loaders.release(path)
    f = max(1, int(round(TARGET_UM / voxel)))          # the same grid as the AF488 picture
    picture = block_mean(red, f)
    del red
    tissue = small_tissue(block_mean(nuclear, f))
    del nuclear
    lo, hi = np.percentile(picture[tissue] if tissue.any() else picture, DISPLAY_PCT)
    grey = np.clip((picture - lo) / max(hi - lo, 1e-9), 0, 1)
    rgb = np.zeros((*grey.shape, 3), np.uint8)
    rgb[..., 0] = np.round(grey * 255).astype(np.uint8)
    sid = short(name)
    Image.fromarray(rgb).save(out / "img" / f"{sid}_r.jpg", quality=JPEG_QUALITY)
    red_file(out, name).write_text(json.dumps({"version": VERSION, "channel": names[red_index],
                                               "display": [round(float(lo), 1), round(float(hi), 1)]}),
                                   encoding="utf-8")
    return f"{sid} (AF555): {time.perf_counter() - started:.1f} s"


def section_table() -> pd.DataFrame:
    """The run's sections in the study, with design, junction length and the run's QC columns."""
    design = pd.read_csv(DESIGN_CSV)
    design["name"] = design["File_Name"].str.replace(".ome.tiff", "", regex=False)
    design = design[design["Genotype"].isin(GENOTYPES)].copy()
    design["slide"] = design["Genotype"] + " e" + design["Experiment"].astype(str) + " s" + design["Slide"].astype(str)
    design["Experiment"] = design["Experiment"].astype(str)
    names = sorted(p.stem for p in (RUN / "cells").glob("*.json"))
    rows = []
    for n in names:
        info = json.loads((RUN / "cells" / f"{n}.json").read_text(encoding="utf-8"))
        rows.append({"name": n, "length_um": info.get("junction_length_um"), "status": info.get("status"),
                     "region_by": info.get("region_by"), "n_nuclei": info.get("n_cells_section")})
    table = pd.DataFrame(rows).merge(design[["name", "Genotype", "Experiment", "slide"]], on="name")
    counts = pd.read_excel(RUN / "counts.xlsx", sheet_name="Counts")
    counts = counts[counts["Channel"] == CHANNEL].drop_duplicates("Sample").set_index("Sample")
    for column, key in (("Region drawn by", "region_drawn_by"), ("QC", "qc"), ("Review status", "review"),
                        ("Review decision", "decision")):
        table[key] = table["name"].map(counts[column]).where(lambda s: s.notna(), None)
    # Size of the assembloid: its tissue area (tissue_area.py) and its length across the line (dataset/04).
    area = pd.read_csv(OUT / "section_tissue_area.csv").set_index("name")["tissue area (mm²)"]
    length = pd.read_csv(OUT / "section_scales.csv").set_index("name")["assembloid length (µm)"]
    table["area_mm2"] = table["name"].map(area)
    table["assembloid_um"] = table["name"].map(length)
    return table


def human_numbers(table: pd.DataFrame) -> dict[str, dict]:
    """Per section: each annotator's dots (picture pixels added later), the human total and the
    automated total in the counted area (annotator 1 the whole section, annotator 2 beside the line, ≥ −50 µm)."""
    out: dict[str, dict] = {}
    for index, (annotator, folder) in enumerate(ANNOTATORS.items(), start=1):
        dots = pd.read_parquet(folder / "clicks.parquet")
        for name, d in dots.groupby("name"):
            if name not in set(table["name"]):
                continue
            sign = home_sign(name)
            o = pd.read_parquet(MASKS / name / "objects.parquet")
            o = o[o["mask"] == METHOD_MASK]
            if index == 1:
                human, automated = len(d), len(o)
                in_area_dots = np.ones(len(d), bool)
            else:
                in_area_dots = d["beside"].fillna(False).to_numpy(bool) & (d["d"].to_numpy(float) * sign >= AREA_2_UM)
                o_in = (o["Beside the junction line"].fillna(False).to_numpy(bool)
                        & (o["Distance to the junction line (µm)"].to_numpy(float) * sign >= AREA_2_UM))
                human, automated = int(in_area_dots.sum()), int(o_in.sum())
            out.setdefault(name, {})[annotator] = {
                "n": index, "human": int(human), "automated": int(automated),
                "ratio": automated / human if human else None,
                "area": "whole section" if index == 1 else "beside the line, from 50 µm before it to the far edge",
                "_dots": d[["y", "x"]].to_numpy(float), "_in": in_area_dots}
    return out


def _clean(value):
    """NaN / inf → None, numpy scalars → Python, all the way down (JSON has no NaN)."""
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if value is pd.NA or value is pd.NaT:
        return None
    return value


def assemble(out: Path, table: pd.DataFrame) -> int:
    """data.json and data.js from the parts, the table and the human numbers."""
    humans = human_numbers(table)
    sections = []
    for _, s in table.iterrows():
        target = part_file(out, s["name"])
        if not target.exists():
            continue
        part = json.loads(target.read_text(encoding="utf-8"))
        k = np.asarray(part["objects"]["k"], int)
        reg = np.asarray(part["objects"]["reg"], int)
        length_mm = float(s["length_um"]) / 1000 if pd.notna(s["length_um"]) else np.nan
        crossed, home = int((k == 1).sum()), int((k == 2).sum())
        red = json.loads(red_file(out, s["name"]).read_text(encoding="utf-8")) if red_file(out, s["name"]).exists() else None
        entry = {
            **{key: part[key] for key in ("id", "name", "um_per_px", "width", "height", "display", "line", "region",
                                          "objects")},
            "genotype": s["Genotype"], "experiment": s["Experiment"], "slide": s["slide"],
            "length_um": round(length_mm * 1000, 1),
            "n_objects": int(len(k)), "n_region": int(reg.sum()), "n_crossed": crossed, "n_home": home,
            "crossed_per_mm": crossed / length_mm if length_mm > 0 else None,
            "share_crossed": crossed / (crossed + home) if crossed + home else None,
            "region_per_mm": int(reg.sum()) / length_mm if length_mm > 0 else None,
            "home_per_mm": home / length_mm if length_mm > 0 else None,
            "area_mm2": s["area_mm2"], "assembloid_um": s["assembloid_um"], "n_nuclei": s["n_nuclei"],
            "objects_per_mm2": len(k) / s["area_mm2"] if s["area_mm2"] and s["area_mm2"] > 0 else None,
            "objects_per_1000_nuclei": 1000 * len(k) / s["n_nuclei"] if s["n_nuclei"] else None,
            "region_by": s["region_drawn_by"] or s["region_by"], "qc": s["qc"], "review": s["review"],
            "decision": s["decision"], "status": s["status"], "humans": [],
            "red": None if red is None else {"channel": red["channel"], "display": red["display"]},
        }
        for annotator, h in humans.get(s["name"], {}).items():
            dots_um, inside = h["_dots"], h["_in"]
            y, x = to_px(dots_um[:, 0], dots_um[:, 1], part["voxel"], part["f"])
            entry["humans"].append({**{k2: v for k2, v in h.items() if not k2.startswith("_")}, "label": annotator,
                                    "dots": {"x": np.round(x, 1).tolist(), "y": np.round(y, 1).tolist(),
                                             "in": inside.astype(int).tolist()}})
        sections.append(entry)
    payload = _clean({"generated": time.strftime("%Y-%m-%d %H:%M"), "method": METHOD_MASK, "gap_um": GAP_UM,
                      "near_um": NEAR_UM, "sections": sections})
    text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    (out / "data.json").write_text(text, encoding="utf-8")
    (out / "data.js").write_text("window.SECTION_DATA = " + text + ";\n", encoding="utf-8")
    shutil.copy(HERE / "section_browser.html", out / "section_browser.html")
    return len(sections)


def main(argv: list[str]) -> int:
    out = BROWSER
    if "--out" in argv:
        i = argv.index("--out")
        out = Path(argv[i + 1])
        argv = argv[:i] + argv[i + 2:]
    (out / "img").mkdir(parents=True, exist_ok=True)
    (out / "parts").mkdir(parents=True, exist_ok=True)
    table = section_table()
    todo = [n for n in table["name"] if not done(out, n)]
    if argv:
        todo = todo[: int(argv[0])]
    print(f"{len(table)} sections, {len(todo)} to do -> {out}", flush=True)
    started = time.perf_counter()
    if todo:
        with ProcessPoolExecutor(WORKERS) as pool:
            futures = {pool.submit(run_one, n, str(out)): n for n in todo}
            for index, future in enumerate(as_completed(futures), start=1):
                try:
                    line = future.result()
                except Exception as exc:
                    line = f"{short(futures[future])}: FAILED {exc!r}"
                elapsed = time.perf_counter() - started
                print(f"[{index}/{len(todo)}] {line} | {elapsed / 60:.1f} min, "
                      f"~{elapsed / index * (len(todo) - index) / 60:.0f} min left", flush=True)
    red_todo = [n for n in table["name"] if not red_file(out, n).exists() or not (out / "img" / f"{short(n)}_r.jpg").exists()]
    if argv:
        red_todo = red_todo[: int(argv[0])]
    if red_todo:
        print(f"AF555 pictures: {len(red_todo)} to do", flush=True)
        started = time.perf_counter()
        with ProcessPoolExecutor(WORKERS) as pool:
            futures = {pool.submit(run_red, n, str(out)): n for n in red_todo}
            for index, future in enumerate(as_completed(futures), start=1):
                try:
                    line = future.result()
                except Exception as exc:
                    line = f"{short(futures[future])} (AF555): FAILED {exc!r}"
                if index % 25 == 0 or index == len(red_todo) or "FAILED" in line:
                    elapsed = time.perf_counter() - started
                    print(f"[{index}/{len(red_todo)}] {line} | {elapsed / 60:.1f} min", flush=True)
    n = assemble(out, table)
    print(f"written {out / 'data.json'}: {n} sections", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
