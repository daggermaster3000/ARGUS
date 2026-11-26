"""Data and pictures for the section browser: every analysed section, what each positivity method labelled.

For each of the run's sections (WT3 and GD1 only):

- a picture of the whole section, AF488 (maximum-intensity projection over z) at ~1.3 µm per pixel, green on
  black, its display range from the tissue's 1st–99.7th percentile; the AF555 picture (red) on the
  same grid; and a ~300 px thumbnail;
- the junction line and the junction region's outline, in picture pixels, and the human annotators'
  dots where they exist;
- per positivity method and threshold (:data:`METHODS`), the outline of every AF488⁺ object or cell
  (one small ``parts/masks/<id>__<method>-<k>.js`` file each, loaded by the page when it is shown)
  and the section's numbers: in total, in the junction region, crossed and at home, crossed per mm
  of junction, and on the human-counted sections the automated count in the area the human counted.

The methods:

- **AF488 mask, Gaussian 20 µm background** (3, 5, 8 SD) — the mask of ``counting.py``:
  AF488 smoothed 0.5 µm, less its 20 µm Gaussian, cut inside the tissue at the median + k robust SD,
  objects of 20 µm² or less dropped, holes of 10 µm² or less filled. One object = one AF488⁺ cell body. 8 SD is the
  counting method of the official analysis (the page's default; until 9 Nov 2026 it was 5 SD). The median and
  robust SD are each section's own, from its tissue pixels, so the threshold is set per section.
- **The same mask with a smaller size filter** (:data:`SIZE_VARIANTS`: objects > 10, > 5 and > 3 µm² kept; 3, 5,
  8 SD): the same thresholds and hole filling, only fewer small objects dropped, to see what the 20 µm² filter
  does to the counts and the statistics.
- **The same mask with a shared threshold spread** (``gpool``, ``gslide``; 3, 5, 8 SD): the section's own median + k × the
  robust SD pooled over all sections (the median of the sections' own SDs) or over the sections of its slide
  (:func:`write_spreads`, ``parts/spreads.json``), so that a noisy section does not raise its own bar nor a quiet, empty
  one lower it.
- **AF488 mask, local median 45 µm background** (3, 5, 7, 8 SD) — as a local-median mask:
  AF488 less its 45 µm local median (``junction_analysis.less_local_median``), smoothed 0.5 µm, the
  same threshold and cleanup.
- **Per-nucleus rule** on the run's cells, median-filtered or raw AF488 mean of the grown region
  (3, 5, 7 SD) — a cell is AF488⁺ above the median + k robust SD of the junction region's cells; the
  median-filtered 7 SD rule is the run's own. A cell is drawn as the outline of its grown region.

Every method counts only what lies in the section (``section_outline.py``): an object or cell
whose centroid is more than 10 µm outside the run's outline of the section, or in a piece of it the junction line
does not pass through (debris, bubble rims, a neighbouring organoid, a bright strip at the scan's edge), is not
counted nor drawn.

Sides (``sides.py``): crossed = on the far side of the junction line extended past both ends, in the
organoid where AF488 is not at home, at any distance; home = every other. Home is the side holding
more of the Gaussian 8 SD mask's object area (``counting.py``'s objects), so a section's
side is the same for every method. (With the 5 SD mask's area, the rule before 9 Nov 2026, every one of the 614
sections has the same home side.) Objects and cells are placed by their centroid.

Writes into ``rerun/results/section_browser/`` (``paths.BROWSER``) ``data.json``, the same as ``data.js`` (so the
page opens from disk too), ``img/`` and ``parts/`` (one file per section, so a rerun skips what is done), beside a
copy of ``section_browser.html`` and of ``figure_builder.html`` (the paper's graphs with any normalisation; it also
reads ``sections_table.js``, which the last cell of ``02_biology.ipynb`` writes beside it), and ``objects.js``
(every counted AF488⁺ object, for the figure builder's distribution panel; ``objects_data.py``).

    python run.py --steps browser

Then open ``section_browser.html`` there. ``browser_tiles.py`` adds the full-resolution tiles and
``browser_outlines.py`` the full-resolution outlines (both run by the same step).
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

from .. import paths, section_outline, sides  # noqa: E402

# ---------------------------------------------------------------- Settings
MASKS = paths.MASKS                                                # counting.py
DESIGN_CSV = paths.DESIGN_CSV
BROWSER = paths.BROWSER
GENOTYPES = ("WT3", "GD1")
CHANNEL = "AF488"
SIDE_MASK = "kmad8"               #: the stored mask whose object area decides the home side (the counting method's)
#: The positivity methods: key → label, kind ("mask" or "cell"), the k of median + k robust SD, the default k.
METHODS = {
    "gauss": {"label": "AF488 mask, Gaussian 20 µm background", "kind": "mask", "ks": (3, 5, 8), "default": 8,
              "min_um2": 20.0},
    # the same mask, only a smaller size filter ("of": the method whose mask and thresholds it shares)
    "gauss10": {"label": "AF488 mask, Gaussian 20 µm background, objects > 10 µm²", "kind": "mask", "ks": (3, 5, 8),
                "default": 8, "min_um2": 10.0, "of": "gauss"},
    "gauss5": {"label": "AF488 mask, Gaussian 20 µm background, objects > 5 µm²", "kind": "mask", "ks": (3, 5, 8),
               "default": 8, "min_um2": 5.0, "of": "gauss"},
    "gauss3": {"label": "AF488 mask, Gaussian 20 µm background, objects > 3 µm²", "kind": "mask", "ks": (3, 5, 8),
               "default": 8, "min_um2": 3.0, "of": "gauss"},
    # the same mask, the threshold's robust SD not the section's own (its median stays the section's): "spread"
    "gpool": {"label": "AF488 mask, Gaussian 20 µm background, SD pooled over all sections", "kind": "mask", "ks": (3, 5, 8),
              "default": 8, "min_um2": 20.0, "of": "gauss", "spread": "pooled"},
    "gslide": {"label": "AF488 mask, Gaussian 20 µm background, SD pooled over the slide's sections", "kind": "mask",
               "ks": (3, 5, 8), "default": 8, "min_um2": 20.0, "of": "gauss", "spread": "slide"},
    "median": {"label": "AF488 mask, local median 45 µm background", "kind": "mask", "ks": (3, 5, 7, 8), "default": 5,
               "min_um2": 20.0},
    "cellmf": {"label": "Per-nucleus rule, median-filtered mean", "kind": "cell", "ks": (3, 5, 7), "default": 7,
               "column": "AF488 mean (median filter)"},
    "cellraw": {"label": "Per-nucleus rule, raw mean", "kind": "cell", "ks": (3, 5, 7), "default": 7,
                "column": "AF488 mean"},
}
DEFAULT = "gauss-8"               #: the counting method of the official analysis (median + 8 robust SD)
#: the Gaussian mask's variants with another size filter
#: the object map's filter (object_filter.py): the 8 SD objects (> 3 µm²) kept by its feature ranges and hand exclusions,
#: a method only once a filter has been saved
OBJECT_FILTER = paths.OBJECT_FILTER
OBJECT_FILTER_META = paths.OBJECT_FILTER_META
if OBJECT_FILTER.exists() and OBJECT_FILTER_META.exists():
    METHODS["gfilt"] = {"label": "AF488 mask, Gaussian 20 µm background, filtered on the object map", "kind": "mask",
                        "ks": (8,), "default": 8, "min_um2": 3.0, "of": "gauss", "filtered": "gauss3-8"}
#: the pages' menus: each method's family and its variant within the family (method → (family, variant))
MENU = {
    "gauss": ("AF488 mask, Gaussian 20 µm background", "section's SD, objects > 20 µm²"),
    "gauss10": ("AF488 mask, Gaussian 20 µm background", "section's SD, objects > 10 µm²"),
    "gauss5": ("AF488 mask, Gaussian 20 µm background", "section's SD, objects > 5 µm²"),
    "gauss3": ("AF488 mask, Gaussian 20 µm background", "section's SD, objects > 3 µm²"),
    "gpool": ("AF488 mask, Gaussian 20 µm background", "SD pooled over all sections, objects > 20 µm²"),
    "gslide": ("AF488 mask, Gaussian 20 µm background", "SD pooled over the slide, objects > 20 µm²"),
    "gfilt": ("AF488 mask, Gaussian 20 µm background", "filtered on the object map"),
    "median": ("AF488 mask, local median 45 µm background", "objects > 20 µm²"),
    "cellmf": ("Per-nucleus rule", "median-filtered mean"),
    "cellraw": ("Per-nucleus rule", "raw mean"),
    "cpself": ("Cellpose cpdino, self-trained", "masks > 20 µm²"),
    "cpself10": ("Cellpose cpdino, self-trained", "masks > 10 µm²"),
    "cpself5": ("Cellpose cpdino, self-trained", "masks > 5 µm²"),
    "cpself3": ("Cellpose cpdino, self-trained", "masks > 3 µm²"),
}
SIZE_VARIANTS = tuple(m for m, spec in METHODS.items() if spec.get("of") == "gauss" and not spec.get("filtered"))
MEDIAN_BACKGROUND_UM = 45.0       #: junction_analysis.less_local_median's window (for the label only)
SIMPLIFY_PX = 0.5                 #: outlines simplified to within this many picture pixels
AREA_2_UM = -50.0                 #: annotator 2 counted from 50 µm before the line to the far edge
ANNOTATORS = paths.ANNOTATORS
#: the blinded count squares (annotation_squares.py): the key (code → section, frame) and the annotators' exports
SQUARES_KEY = paths.SQUARES_KEY
SQUARES_COUNTS = paths.SQUARES_COUNTS
#: who counted the squares (the name typed into annotate.html, lower case) → their number as an annotator above; the
#: browser shows the number, never the name. Squares of annotator a are the human count n = 2 + a ("squares, annotator a").
SQUARE_ANNOTATORS = {"james": 1, "quillan": 2, "annotator 3": 3, "annotator 4": 4, "ref annotator": 5}
TARGET_UM = 1.3                   #: picture pixel size (whole pixels of the scan are averaged)
THUMB_PX = 300                    #: longest side of a thumbnail
DISPLAY_PCT = (1.0, 99.7)         #: display range: these percentiles of AF488 inside the tissue
JPEG_QUALITY = 85
WORKERS = 4                       #: each holds a slide scan's planes and one mask at a time (~2 GB)
VERSION = 1                       #: of the pictures (parts/<id>.json)
MASK_VERSION = 4                  #: of the outlines and counts (parts/methods/<id>.json, parts/masks/); 2: in the section
                                  #: only; 3: masks cut inside the tissue or the outline (as counting.py 3)
PREFIX = "2025_09_26_Asssembloid_"
# --------------------------------------------------------------------------


def short(name: str) -> str:
    return name.replace(PREFIX, "")


def method_keys() -> list[str]:
    """Every method-threshold key, e.g. ``gauss-8``."""
    return [f"{m}-{k}" for m, spec in METHODS.items() for k in spec["ks"]]


def min_um2(key: str) -> float:
    """The size filter of a method or method-threshold key: objects of this area (µm²) or less are dropped."""
    return float(METHODS[key.split("-")[0]].get("min_um2", 20.0))


def block_mean(image: np.ndarray, f: int) -> np.ndarray:
    """*image* shrunk by *f*, each picture pixel the mean of an f × f block (edges cut)."""
    h, w = image.shape[0] // f * f, image.shape[1] // f * f
    return image[:h, :w].reshape(h // f, f, w // f, f).mean(axis=(1, 3), dtype=np.float32)


def small_tissue(nuclear_small: np.ndarray) -> np.ndarray:
    """The section from the shrunk nuclear stain: Otsu, closed, holes filled (as cv.tissue_mask)."""
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


def methods_file(out: Path, name: str) -> Path:
    return out / "parts" / "methods" / f"{short(name)}.json"


def mask_file(out: Path, name: str, key: str) -> Path:
    return out / "parts" / "masks" / f"{short(name)}__{key}.js"


def done(out: Path, name: str) -> bool:
    try:
        part = json.loads(part_file(out, name).read_text(encoding="utf-8"))
        return part.get("version", 0) >= VERSION and (out / "img" / f"{part['id']}.jpg").exists()
    except Exception:
        return False


def masks_done(out: Path, name: str) -> bool:
    try:
        part = json.loads(methods_file(out, name).read_text(encoding="utf-8"))
        return part.get("version", 0) >= MASK_VERSION and all(mask_file(out, name, k).exists() for k in method_keys()
                                                              if not METHODS[k.split("-")[0]].get("filtered"))
    except Exception:
        return False


def spread_file(out: Path) -> Path:
    return out / "parts" / "spreads.json"


def write_spreads(out: Path, table: pd.DataFrame) -> dict:
    """The robust SD of the threshold variants with a shared spread: the median over all sections of each section's own
    (``pooled``) and the median over each slide's sections (``slide``: section → value), from the stored counting
    summaries. A section's threshold is then its own median + k × that SD."""
    noise = {n: json.loads((MASKS / n / "summary.json").read_text(encoding="utf-8"))["noise_sd"] for n in table["name"]}
    t = table.assign(noise=table["name"].map(noise))
    by_slide = t.groupby("slide")["noise"].median()
    spreads = {"pooled": float(t["noise"].median()), "slide": {n: float(by_slide[s]) for n, s in zip(t["name"], t["slide"])}}
    spread_file(out).parent.mkdir(parents=True, exist_ok=True)
    spread_file(out).write_text(json.dumps(spreads), encoding="utf-8")
    return spreads


def variant_cut(name: str, method: str, k: float, stored_summary: dict, spreads: dict | None) -> float | None:
    """The threshold of a variant with a shared spread (None for the others): the section's median + k × the SD."""
    spread = METHODS[method].get("spread")
    if not spread:
        return None
    sd = spreads["pooled"] if spread == "pooled" else spreads["slide"][name]
    return float(stored_summary["background_centre"]) + k * sd


def run_one(name: str, out: str) -> str:
    """Picture, thumbnail, junction line and region of one section → img/ and parts/."""
    from PIL import Image
    from skimage import measure

    from .. import cv
    from .. import cells as ja

    logging.getLogger("tifffile").setLevel(logging.ERROR)
    cv.below_normal_priority()
    out = Path(out)
    started = time.perf_counter()
    info = paths.section_info(name)
    af488, nuclear, voxel = cv.read_planes(paths.scan_path(name))
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
    if paths.region_tif(name).exists():
        band = ja.read_labels(paths.region_tif(name)) > 0
        small = block_mean(band.astype(np.float32), f) >= 0.5
        for contour in measure.find_contours(np.pad(small, 1).astype(float), 0.5):
            contour = measure.approximate_polygon(contour - 1, 0.8)
            if len(contour) >= 4:
                # find_contours runs through pixel centres: +0.5 to the picture's pixel-edge coordinates
                region.append([[round(float(c) + 0.5, 1), round(float(r) + 0.5, 1)] for r, c in contour])

    part = {
        "version": VERSION, "id": sid, "name": name, "voxel": voxel, "f": f, "um_per_px": round(voxel * f, 4),
        "width": int(rgb.shape[1]),
        "height": int(rgb.shape[0]), "display": [round(float(lo), 1), round(float(hi), 1)],
        "line": line, "region": region,
        "seconds": round(time.perf_counter() - started, 1), "read_seconds": round(read_s, 1),
    }
    part_file(out, name).write_text(json.dumps(part, separators=(",", ":")), encoding="utf-8")
    return f"{sid}: {part['seconds']:.1f} s (reading {read_s:.1f} s)"


RED_HINT = "555"                  #: the second label's channel (AF555), shown in red


def red_file(out: Path, name: str) -> Path:
    return out / "parts" / f"{short(name)}_red.json"


def run_red(name: str, out: str) -> str:
    """The AF555 picture of one section (red on black, same grid as the AF488 one) → img/<id>_r.jpg."""
    from PIL import Image

    from .. import cv
    from .. import junction as jn
    from .. import cells as ja
    from .. import scan as loaders

    logging.getLogger("tifffile").setLevel(logging.ERROR)
    cv.below_normal_priority()
    out = Path(out)
    started = time.perf_counter()
    info = paths.section_info(name)
    path = paths.scan_path(name)
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


# ---------------------------------------------------------------- outlines and counts per method
def section_sign(name: str, line) -> float | None:
    """The section's home sign (``sides.home_sign``) from the stored Gaussian 8 SD objects' area; None without a line."""
    if line is None:
        return None
    o = pd.read_parquet(MASKS / name / "objects.parquet")
    o = o[o["mask"] == SIDE_MASK]
    raw = sides.raw_side(o[["Centroid Y (µm)", "Centroid X (µm)"]].to_numpy(float), line)
    return sides.home_sign(raw, o["Area (µm²)"].to_numpy(float))


def outline(crop: np.ndarray, top: int, left: int, voxel: float, f: int) -> list[list[float]]:
    """The rings round the True pixels of *crop* (whose corner is full-size pixel (top, left)),
    in picture pixels, simplified, each flat ``[x0, y0, x1, y1, …]``."""
    from skimage import measure

    rings = []
    for contour in measure.find_contours(np.pad(crop, 1).astype(np.float32), 0.5):
        # find_contours: pixel-centre coordinates of the padded crop → full size → picture pixels
        ys = (contour[:, 0] - 1 + top + 0.5) / f
        xs = (contour[:, 1] - 1 + left + 0.5) / f
        ring = measure.approximate_polygon(np.column_stack([xs, ys]), SIMPLIFY_PX)
        if len(ring) > 1 and np.allclose(ring[0], ring[-1]):
            ring = ring[:-1]                          # the polygon closes itself
        if len(ring):                                 # a sliver (< 3 points) is kept too: no object is dropped
            rings.append(np.round(ring, 1).ravel().tolist())
    return rings


def labelled_shapes(labelled: np.ndarray, voxel: float, f: int) -> pd.DataFrame:
    """Every object of a label image: label, area (µm²), centroid (µm, index × voxel) and its outline rings."""
    from skimage.measure import regionprops

    rows = []
    for p in regionprops(labelled):
        r0, c0, _r1, _c1 = p.bbox
        cy, cx = p.centroid
        rows.append({"label": int(p.label), "area": float(p.area) * voxel ** 2, "y": cy * voxel, "x": cx * voxel,
                     "rings": outline(p.image, r0, c0, voxel, f)})
    return pd.DataFrame(rows, columns=["label", "area", "y", "x", "rings"])


def mask_objects(mask: np.ndarray, voxel: float, f: int) -> pd.DataFrame:
    """The objects of one mask (8-connected, as counting.py)."""
    from skimage.measure import label

    return labelled_shapes(label(mask, connectivity=2), voxel, f)


def robust_cuts(signal: np.ndarray, tissue: np.ndarray, ks) -> tuple[dict[int, float], float, float]:
    """Median + k robust SD of *signal* in the tissue (every 20th pixel, as the masks were made)."""
    values = signal[tissue][::20]
    centre = float(np.median(values))
    spread = float(1.4826 * np.median(np.abs(values - centre)))
    return {k: centre + k * spread for k in ks}, centre, spread


def cleaned(mask: np.ndarray, voxel: float, min_area_um2: float | None = None) -> np.ndarray:
    """*mask* without objects of *min_area_um2* or less (default 20 µm², cv.py) and with holes of 10 µm² or less
    filled."""
    from .. import cv

    limit = cv.MIN_OBJECT_UM2 if min_area_um2 is None else min_area_um2
    mask = cv.remove_small_objects(mask, int(round(limit / voxel ** 2)))
    return cv.remove_small_holes(mask, int(round(cv.MAX_HOLE_UM2 / voxel ** 2)))


def write_mask_js(target: Path, sid: str, key: str, shapes: pd.DataFrame) -> None:
    """``window.MASKS_LOADED(id, key, {...})``: per object its rings, side, in region, area, distance past the line."""
    payload = {
        "rings": shapes["rings"].tolist(),
        "side": shapes["side"].astype(int).tolist(),                   # 1 crossed, 0 home, -1 no line
        "reg": shapes["reg"].astype(int).tolist(),
        "area": np.round(shapes["area"].to_numpy(float), 0).astype(int).tolist(),
        "away": [None if not np.isfinite(a) else round(float(a), 1) for a in shapes["away"].to_numpy(float)],
    }
    text = json.dumps(payload, separators=(",", ":"), allow_nan=False)
    target.write_text(f"window.MASKS_LOADED({json.dumps(sid)},{json.dumps(key)},{text});\n", encoding="utf-8")


def run_masks(name: str, out: str) -> str:
    """Every method-threshold's AF488⁺ outlines and counts for one section → parts/masks/ and parts/methods/."""
    from scipy import ndimage

    from .. import cv
    from .. import cells as ja

    logging.getLogger("tifffile").setLevel(logging.ERROR)
    cv.below_normal_priority()
    out = Path(out)
    for folder in ("methods", "masks"):
        (out / "parts" / folder).mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    seconds = {}
    tick = time.perf_counter()
    info = paths.section_info(name)
    sid = short(name)
    line = np.asarray(info["line"], float) if info.get("line") else None
    sign = section_sign(name, line)
    band = ja.read_labels(paths.region_tif(name)) > 0 if paths.region_tif(name).exists() else None
    af488, nuclear, voxel = cv.read_planes(paths.scan_path(name))
    f = max(1, int(round(TARGET_UM / voxel)))
    del nuclear
    # objects and cells outside the section are not counted (section_outline; the outline counting.py cached)
    section = section_outline.outline_for(MASKS / name, paths.scan_path(name))     # not `outline`: that is the ring tracer
    # thresholds over the section's tissue from every channel; masks cut inside the outline widened 10 µm (counting.py version 4)
    tissue = section_outline.stats_mask(section, af488.shape)
    domain = section_outline.count_domain(section, af488.shape)
    line_um = section_outline.line_um(info)

    def in_section(mask: np.ndarray) -> np.ndarray:
        return section_outline.drop_outside(mask, section, line_um, voxel)[0]

    seconds["read"] = round(time.perf_counter() - tick, 1)
    stored_summary = json.loads((MASKS / name / "summary.json").read_text(encoding="utf-8"))
    stored_objects = pd.read_parquet(MASKS / name / "objects.parquet", columns=["mask"])["mask"].value_counts()

    def place(shapes: pd.DataFrame, in_region=None) -> pd.DataFrame:
        yx = shapes[["y", "x"]].to_numpy(float)
        shapes["away"] = sides.away(yx, line, sign) if sign is not None and len(yx) else np.full(len(yx), np.nan)
        shapes["side"] = np.where(np.isnan(shapes["away"]), -1, (shapes["away"] > 0).astype(int))
        if in_region is not None:
            shapes["reg"] = in_region
        elif band is not None and len(shapes):
            rows = np.clip(np.round(yx[:, 0] / voxel).astype(int), 0, band.shape[0] - 1)
            cols = np.clip(np.round(yx[:, 1] / voxel).astype(int), 0, band.shape[1] - 1)
            shapes["reg"] = band[rows, cols]
        else:
            shapes["reg"] = False
        return shapes

    counts, check = {}, {}

    def record(key: str, shapes: pd.DataFrame, cut: float) -> None:
        a = shapes["away"].to_numpy(float)
        has_line = sign is not None
        counts[key] = {"n": int(len(shapes)), "reg": int(shapes["reg"].astype(bool).sum()),
                       "crossed": int((a > 0).sum()) if has_line else None,
                       "home": int((a <= 0).sum()) if has_line else None,
                       "area2": int((a >= AREA_2_UM).sum()) if has_line else None,
                       "thr": round(float(cut), 3)}
        write_mask_js(mask_file(out, name, key), sid, key, shapes)

    # AF488 mask, Gaussian 20 µm background (cv.classical_masks)
    tick = time.perf_counter()
    smooth = ndimage.gaussian_filter(af488, cv.SMOOTH_UM / voxel)
    signal = smooth - cv.local_background(smooth, cv.BACKGROUND_UM / voxel)
    del smooth
    cuts, _c, _s = robust_cuts(signal, tissue, METHODS["gauss"]["ks"])
    stored = stored_summary["thresholds"]
    spreads = json.loads(spread_file(out).read_text(encoding="utf-8"))
    for m in ("gauss",) + SIZE_VARIANTS:                              # the same mask, each with its size filter or spread
        for k in METHODS[m]["ks"]:
            cut = variant_cut(name, m, k, stored_summary, spreads)
            if cut is None:
                cut = float(stored.get(f"kmad{k:g}", cuts[k]))    # the stored threshold (equal to the recomputed one)
            shapes = place(mask_objects(in_section(cleaned((signal > cut) & domain, voxel, min_um2(m))), voxel, f))
            record(f"{m}-{k}", shapes, cut)
            if m == "gauss":
                check[f"gauss-{k}"] = {"recomputed_threshold": cuts[k], "stored_threshold": stored.get(f"kmad{k:g}"),
                                       "objects": len(shapes), "stored_objects": int(stored_objects.get(f"kmad{k:g}", -1))}
    del signal
    seconds["gauss"] = round(time.perf_counter() - tick, 1)

    # AF488 mask, local median 45 µm background (less the local median, then smoothed)
    tick = time.perf_counter()
    signal = ndimage.gaussian_filter(ja.less_local_median(af488, voxel), cv.SMOOTH_UM / voxel)
    del af488
    cuts, _c, _s = robust_cuts(signal, tissue, METHODS["median"]["ks"])
    for k in METHODS["median"]["ks"]:
        shapes = place(mask_objects(in_section(cleaned((signal > cuts[k]) & domain, voxel)), voxel, f))
        record(f"median-{k}", shapes, cuts[k])
    del signal, tissue
    seconds["median"] = round(time.perf_counter() - tick, 1)

    # Per-nucleus rules on the run's cells: the grown region of each positive cell
    tick = time.perf_counter()
    columns = ["Label", "Centroid Y (µm)", "Centroid X (µm)", ja.IN_REGION_COLUMN] + [
        METHODS[m]["column"] for m in ("cellmf", "cellraw")]
    cells = pd.read_csv(paths.cells_csv(name), usecols=columns)
    region = cells[ja.IN_REGION_COLUMN].fillna(False).astype(bool).to_numpy()
    counted = section_outline.in_section(section, line_um, cells[["Centroid Y (µm)", "Centroid X (µm)"]].to_numpy(float),
                                         voxel)
    positive, cell_cuts = {}, {}
    for m in ("cellmf", "cellraw"):
        values = cells[METHODS[m]["column"]].to_numpy(float)
        centre, spread = ja.robust_centre_and_spread(values[region])
        for k in METHODS[m]["ks"]:
            cell_cuts[f"{m}-{k}"] = centre + k * spread
            positive[f"{m}-{k}"] = (values > centre + k * spread) & counted     # outside the section: not counted
    wanted = np.zeros(len(cells), bool)
    for p in positive.values():
        wanted |= p
    grown = ja.read_labels(paths.grown_tif(name))
    labels = cells["Label"].to_numpy(np.int64)
    rings: dict[int, list] = {}
    areas: dict[int, float] = {}
    boxes = ndimage.find_objects(grown)
    for lab in labels[wanted]:
        box = boxes[lab - 1] if 0 < lab <= len(boxes) else None
        if box is None:
            rings[lab], areas[lab] = [], np.nan
            continue
        crop = grown[box] == lab
        rings[lab] = outline(crop, box[0].start, box[1].start, voxel, f)
        areas[lab] = float(crop.sum()) * voxel ** 2
    del grown
    for key, p in positive.items():
        chosen = cells[p]
        shapes = pd.DataFrame({"label": chosen["Label"].to_numpy(np.int64),
                               "area": [areas[int(lab)] for lab in chosen["Label"]],
                               "y": chosen["Centroid Y (µm)"].to_numpy(float),
                               "x": chosen["Centroid X (µm)"].to_numpy(float),
                               "rings": [rings[int(lab)] for lab in chosen["Label"]]})
        shapes = place(shapes, in_region=region[p])
        record(key, shapes, cell_cuts[key])
    seconds["cells"] = round(time.perf_counter() - tick, 1)

    summary = {"version": MASK_VERSION, "id": sid, "name": name, "sign": sign, "counts": counts, "check": check,
               "seconds": {**seconds, "total": round(time.perf_counter() - started, 1)}}
    methods_file(out, name).write_text(json.dumps(_clean(summary), separators=(",", ":")), encoding="utf-8")
    g = check[DEFAULT]
    return (f"{sid}: {summary['seconds']['total']:.0f} s ({', '.join(f'{k} {v:.0f}' for k, v in seconds.items())}); "
            f"{DEFAULT} {g['objects']} objects (stored {g['stored_objects']}), crossed {counts[DEFAULT]['crossed']}")


# ---------------------------------------------------------------- the table and data.js
def section_table() -> pd.DataFrame:
    """The run's sections in the study, with design, junction length and the run's QC columns."""
    design = pd.read_csv(DESIGN_CSV)
    design["name"] = design["File_Name"].str.replace(".ome.tiff", "", regex=False)
    design = design[design["Genotype"].isin(GENOTYPES)].copy()
    design["slide"] = design["Genotype"] + " e" + design["Experiment"].astype(str) + " s" + design["Slide"].astype(str)
    design["Experiment"] = design["Experiment"].astype(str)
    names = paths.section_names()
    rows = []
    for n in names:
        info = paths.section_info(n)
        rows.append({"name": n, "length_um": info.get("junction_length_um"), "status": info.get("status"),
                     "region_by": info.get("region_by"), "n_nuclei": info.get("n_cells_section"),
                     "line": info.get("line")})
    table = pd.DataFrame(rows).merge(design[["name", "Genotype", "Experiment", "slide"]], on="name")
    counts = pd.read_excel(paths.review_xlsx(), sheet_name="Counts")
    counts = counts[counts["Channel"] == CHANNEL].drop_duplicates("Sample").set_index("Sample")
    for column, key in (("Region drawn by", "region_drawn_by"), ("QC", "qc"), ("Review status", "review"),
                        ("Review decision", "decision")):
        table[key] = table["name"].map(counts[column]).where(lambda s: s.notna(), None)
    # Size of the assembloid: its tissue area (tissue_area.py) and its length across the line (dataset/04).
    area = pd.read_csv(paths.TISSUE_AREA_CSV).set_index("name")["tissue area (mm²)"]
    length = pd.read_csv(paths.SCALES_CSV).set_index("name")["assembloid length (µm)"]
    table["area_mm2"] = table["name"].map(area)
    table["assembloid_um"] = table["name"].map(length)
    return table


def human_numbers(table: pd.DataFrame, methods: dict[str, dict]) -> dict[str, dict]:
    """Per section: each annotator's dots, the human total and, per method, the automated total in the
    counted area (annotator 1 the whole section; annotator 2 from 50 µm before the extended line on)."""
    lines = table.set_index("name")["line"]
    out: dict[str, dict] = {}
    for index, (annotator, folder) in enumerate(ANNOTATORS.items(), start=1):
        dots = pd.read_parquet(folder / "clicks.parquet")
        for name, d in dots.groupby("name"):
            if name not in methods:
                continue
            m = methods[name]
            if index == 1:
                in_area_dots = np.ones(len(d), bool)
                automated = {key: c["n"] for key, c in m["counts"].items()}
            else:
                if m["sign"] is None:
                    continue
                away = sides.away(d[["y", "x"]].to_numpy(float), lines[name], m["sign"])
                in_area_dots = away >= AREA_2_UM
                automated = {key: c["area2"] for key, c in m["counts"].items()}
            out.setdefault(name, {})[annotator] = {
                "n": index, "human": int(in_area_dots.sum()), "automated": automated,
                "area": "whole section" if index == 1 else "from 50 µm before the junction line (extended) to the far edge",
                "_dots": d[["y", "x"]].to_numpy(float), "_in": in_area_dots}
    return out


def ring_centroid(ring: list[float]) -> tuple[float, float, float]:
    """(x, y, |area|) of a flat ``[x0, y0, x1, y1, …]`` polygon (its vertices' mean when it has no area)."""
    x, y = np.asarray(ring[0::2], float), np.asarray(ring[1::2], float)
    cross = x * np.roll(y, -1) - np.roll(x, -1) * y
    a = cross.sum() / 2
    if abs(a) < 1e-9:
        return float(x.mean()), float(y.mean()), 0.0
    return (float(((x + np.roll(x, -1)) * cross).sum() / (6 * a)), float(((y + np.roll(y, -1)) * cross).sum() / (6 * a)),
            abs(float(a)))


def mask_centroids(out: Path, name: str, key: str) -> np.ndarray:
    """(x, y) in picture pixels of every object drawn for a method-threshold (its largest ring's centroid)."""
    text = mask_file(out, name, key).read_text(encoding="utf-8")
    payload = json.loads(text[text.index(",{") + 1:text.rindex(");")])
    points = [max((ring_centroid(r) for r in rings), key=lambda c: c[2])[:2] for rings in payload["rings"] if rings]
    return np.asarray(points, float).reshape(-1, 2)


def square_numbers(out: Path, parts: dict[str, dict]) -> tuple[dict[str, dict], dict[str, list]]:
    """The blinded count squares, per section: (1) per annotator one human count — the dots in the section's squares
    (one in each of home, junction region and crossed), the automated count per method-threshold of the objects whose
    centroid lies in the same squares, the dots in picture pixels; (2) each square's frame (picture pixels), its
    compartment and per annotator its count and per method-threshold the automated count. A square counts when the
    annotator saw it: marked Done, or before the last square dotted or marked Done in the page's order (key ``shown``)."""
    files = sorted(SQUARES_COUNTS.glob("square_counts_*.csv")) if SQUARES_KEY.exists() else []
    if not files:
        return {}, {}
    key = pd.read_csv(SQUARES_KEY)
    counts = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    counts["who"] = counts["annotator"].str.strip().str.lower().map(
        lambda w: SQUARE_ANNOTATORS.get(w, len(SQUARE_ANNOTATORS) + 1))
    sq = counts[counts["kind"] == "square"].copy()
    sq["pos"] = sq["code"].map(key.set_index("code")["shown"])
    reached = sq[(sq["done"] == 1) | (sq["n_dots"] > 0)].groupby("who")["pos"].max()
    seen = sq[(sq["done"] == 1) | (sq["pos"] <= sq["who"].map(reached))]
    dots = counts[counts["kind"] == "dot"]
    keys = method_keys()
    humans: dict[str, dict] = {}
    frames: dict[str, list] = {}
    for name, g in key.groupby("name"):
        if name not in parts:
            continue
        voxel, f = parts[name]["voxel"], parts[name]["f"]
        centroids = {k: mask_centroids(out, name, k) for k in keys if mask_file(out, name, k).exists()}
        for _, k in g.sort_values("compartment").iterrows():
            # frame and dots: µm from the scan's corner (pixel edges) → picture pixels
            x0, y0, side = k["frame_left_um"] / voxel / f, k["frame_top_um"] / voxel / f, k["frame_um"] / voxel / f
            auto = {m: int(((c[:, 0] >= x0) & (c[:, 0] < x0 + side) & (c[:, 1] >= y0) & (c[:, 1] < y0 + side)).sum())
                    for m, c in centroids.items()}
            entry = {"code": k["code"], "compartment": k["compartment"], "x": round(x0, 1), "y": round(y0, 1), "size": round(side, 1),
                     "human": {}, "automated": auto}
            for who, s in seen[seen["code"] == k["code"]].groupby("who"):
                d = dots[(dots["code"] == k["code"]) & (dots["who"] == who)]
                n = 2 + int(who)
                entry["human"][str(n)] = int(s["n_dots"].iloc[0])
                h = humans.setdefault(name, {}).setdefault(f"squares, annotator {who}", {
                    "n": n, "human": 0, "automated": {m: 0 for m in auto}, "squares": 0,
                    "area": "blinded 120 µm squares, one each at home, in the junction region and crossed",
                    "_dots": [], "_in": []})
                h["human"] += entry["human"][str(n)]
                h["squares"] += 1
                for m, v in auto.items():
                    h["automated"][m] += v
                h["_dots"] += [[k["frame_top_um"] + y, k["frame_left_um"] + x] for x, y in zip(d["x_um"], d["y_um"])]
            frames.setdefault(name, []).append(entry)
    for per in humans.values():
        for h in per.values():
            h["_dots"] = np.asarray(h["_dots"], float).reshape(-1, 2)
            h["_in"] = np.ones(len(h["_dots"]), bool)
            h["_edges"] = True                            # the dots are µm from the scan's corner, not pixel centres
    return humans, frames


def apply_object_filter(out: Path, names) -> int:
    """The ``gfilt`` method: of each section's ``gauss3-8`` objects those the object map's filter kept (the same order),
    written as their outline file and counts; nothing when no filter has been saved."""
    if "gfilt" not in METHODS:
        return 0
    base = METHODS["gfilt"]["filtered"]
    flags = pd.read_parquet(OBJECT_FILTER)
    by = {sid: g.sort_values("index")["kept"].to_numpy(bool) for sid, g in flags.groupby("section")}
    n = 0
    for name in names:
        sid = short(name)
        target = methods_file(out, name)
        if sid not in by or not target.exists() or not mask_file(out, name, base).exists():
            continue
        text = mask_file(out, name, base).read_text(encoding="utf-8")
        payload = json.loads(text[text.index(",{") + 1:text.rindex(");")])
        keep = by[sid]
        if len(keep) != len(payload["side"]):
            print(f"{sid}: object filter has {len(keep)} objects, {base} {len(payload['side'])}: skipped", flush=True)
            continue
        kept = {k: [v for v, ok in zip(vals, keep) if ok] for k, vals in payload.items()}
        body = json.dumps(kept, separators=(",", ":"), allow_nan=False)
        mask_file(out, name, "gfilt-8").write_text(f"window.MASKS_LOADED({json.dumps(sid)},{json.dumps('gfilt-8')},{body});\n",
                                                   encoding="utf-8")
        summary = json.loads(target.read_text(encoding="utf-8"))
        has_line = summary.get("sign") is not None
        a = np.array([np.nan if v is None else v for v in kept["away"]], float)
        summary["counts"]["gfilt-8"] = {
            "n": int(keep.sum()), "reg": int(np.sum(kept["reg"])),
            "crossed": int((a > 0).sum()) if has_line else None, "home": int((a <= 0).sum()) if has_line else None,
            "area2": int((a >= AREA_2_UM).sum()) if has_line else None, "thr": summary["counts"][base]["thr"]}
        target.write_text(json.dumps(summary, separators=(",", ":")), encoding="utf-8")
        n += 1
    print(f"object filter: {n} sections", flush=True)
    return n


def _clean(value):
    """NaN / inf → None, numpy scalars → Python, all the way down (JSON has no NaN)."""
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if value is pd.NA or value is pd.NaT:
        return None
    return value


def assemble(out: Path, table: pd.DataFrame) -> int:
    """data.json and data.js from the parts, the table and the human numbers."""
    methods = {}
    for name in table["name"]:
        if methods_file(out, name).exists():
            methods[name] = json.loads(methods_file(out, name).read_text(encoding="utf-8"))
    humans = human_numbers(table, methods)
    parts = {n: json.loads(part_file(out, n).read_text(encoding="utf-8")) for n in methods if part_file(out, n).exists()}
    square_humans, squares = square_numbers(out, parts)
    for name, per in square_humans.items():
        humans.setdefault(name, {}).update(per)
    sections = []
    for _, s in table.iterrows():
        if s["name"] not in parts:
            continue
        part = parts[s["name"]]
        length_mm = float(s["length_um"]) / 1000 if pd.notna(s["length_um"]) else np.nan
        red = json.loads(red_file(out, s["name"]).read_text(encoding="utf-8")) if red_file(out, s["name"]).exists() else None
        entry = {
            **{key: part[key] for key in ("id", "name", "um_per_px", "width", "height", "display", "line", "region")},
            "genotype": s["Genotype"], "experiment": s["Experiment"], "slide": s["slide"],
            "length_um": round(length_mm * 1000, 1), "sign": methods[s["name"]]["sign"],
            "counts": methods[s["name"]]["counts"],          # per method-threshold: n, reg, crossed, home, area2, thr
            "area_mm2": s["area_mm2"], "assembloid_um": s["assembloid_um"], "n_nuclei": s["n_nuclei"],
            "region_by": s["region_drawn_by"] or s["region_by"], "qc": s["qc"], "review": s["review"],
            "decision": s["decision"], "status": s["status"], "humans": [],
            "red": None if red is None else {"channel": red["channel"], "display": red["display"]},
            "squares": squares.get(s["name"], []),          # the blinded count squares on this section (frames)
        }
        for annotator, h in humans.get(s["name"], {}).items():
            dots_um, inside = h["_dots"], h["_in"]
            if h.get("_edges"):
                y, x = dots_um[:, 0] / part["voxel"] / part["f"], dots_um[:, 1] / part["voxel"] / part["f"]
            else:
                y, x = to_px(dots_um[:, 0], dots_um[:, 1], part["voxel"], part["f"])
            entry["humans"].append({**{k2: v for k2, v in h.items() if not k2.startswith("_")}, "label": annotator,
                                    "dots": {"x": np.round(x, 1).tolist(), "y": np.round(y, 1).tolist(),
                                             "in": inside.astype(int).tolist()}})
        sections.append(entry)
    catalogue = [{"key": m, "label": spec["label"], "kind": spec["kind"], "ks": list(spec["ks"]),
                  "default": spec["default"], "min_um2": spec.get("min_um2"), "of": spec.get("of"),
                  "family": MENU.get(m, (spec["label"], ""))[0], "variant": MENU.get(m, (spec["label"], ""))[1],
                  **({"filter": json.loads(OBJECT_FILTER_META.read_text(encoding="utf-8"))} if spec.get("filtered") else {})}
                 for m, spec in METHODS.items()]
    payload = _clean({"generated": time.strftime("%Y-%m-%d %H:%M"), "methods": catalogue, "default": DEFAULT,
                      "area_2_um": AREA_2_UM, "sections": sections})
    text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    (out / "data.json").write_text(text, encoding="utf-8")
    (out / "data.js").write_text("window.SECTION_DATA = " + text + ";\n", encoding="utf-8")
    shutil.copy(HERE / "section_browser.html", out / "section_browser.html")
    # the figure builder reads data.js and the biology notebook's sections_table.js (02_biology, last cell)
    shutil.copy(HERE / "figure_builder.html", out / "figure_builder.html")
    return len(sections)


def pool_run(label: str, worker, todo: list[str], out: Path, every: int = 1) -> None:
    """*worker*(name, out) over *todo*, WORKERS at a time, with progress."""
    if not todo:
        return
    print(f"{label}: {len(todo)} to do", flush=True)
    started = time.perf_counter()
    with ProcessPoolExecutor(min(WORKERS, len(todo))) as pool:
        futures = {pool.submit(worker, n, str(out)): n for n in todo}
        for index, future in enumerate(as_completed(futures), start=1):
            try:
                line = future.result()
            except Exception as exc:
                line = f"{short(futures[future])}: FAILED {exc!r}"
            if index % every == 0 or index == len(todo) or "FAILED" in line:
                elapsed = time.perf_counter() - started
                print(f"[{index}/{len(todo)}] {line} | {elapsed / 60:.1f} min, "
                      f"~{elapsed / index * (len(todo) - index) / 60:.0f} min left", flush=True)


def main(argv: list[str]) -> int:
    out = BROWSER
    if "--out" in argv:
        i = argv.index("--out")
        out = Path(argv[i + 1])
        argv = argv[:i] + argv[i + 2:]
    for folder in ("img", "parts", "parts/methods", "parts/masks"):
        (out / folder).mkdir(parents=True, exist_ok=True)
    table = section_table()
    limit = int(argv[0]) if argv else None
    print(f"{len(table)} sections -> {out}", flush=True)
    todo = [n for n in table["name"] if not done(out, n)][:limit]
    pool_run("AF488 pictures", run_one, todo, out)
    todo = [n for n in table["name"] if not red_file(out, n).exists() or not (out / "img" / f"{short(n)}_r.jpg").exists()]
    pool_run("AF555 pictures", run_red, todo[:limit], out, every=25)
    todo = [n for n in table["name"] if not masks_done(out, n)][:limit]
    write_spreads(out, table)                   # the shared robust SDs of the gpool / gslide thresholds
    pool_run("outlines and counts", run_masks, todo, out)
    apply_object_filter(out, table["name"])      # the object map's filter, when one is shipped (data/counting/object_filter.*)
    n = assemble(out, table)
    print(f"written {out / 'data.json'}: {n} sections", flush=True)
    from . import objects_data                  # every counted object, for the figure builder (objects.js)
    objects_data.main(["--out", str(out)])
    return 0



