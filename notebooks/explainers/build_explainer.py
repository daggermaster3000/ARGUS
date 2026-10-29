"""Build the data of the pipeline explainer, and the explainer itself.

Run from the repository root::

    python notebooks/explainers/build_explainer.py            # reuse the slow parts cached in _build/cache
    python notebooks/explainers/build_explainer.py --fresh    # redo everything

What it does:

1. computes every array and picture the explainer shows with the package's own
   code (``microscopy_viewer.junction``, ``fusion``, ``junction_analysis``,
   ``segmentation``) and the validation helpers (``notebooks/validation``), as
   one dict ``DATA`` (schema: the explainer spec; summarised in
   :func:`check_schema`);
2. writes it to ``_build/explainer_data.json``;
3. if ``explainer_template.html`` exists, writes ``pipeline_explainer.html`` with
   the line ``const DATA = /*__DATA__*/ null;`` replaced by the data.

Nothing is written outside ``notebooks/explainers``. The one segmentation (a
150 µm crop of nuclei) runs Cellpose on the **CPU**: the GPU is left alone.
The slow parts — reading whole scans for the organoid gallery and the cells
crop, and that segmentation — are cached in ``_build/cache``.
"""

from __future__ import annotations

import os

# Never touch the GPU from here, whatever imports torch later.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse  # noqa: E402
import base64  # noqa: E402
import datetime as dt  # noqa: E402
import gzip  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BUILD = HERE / "_build"
CACHE = BUILD / "cache"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "notebooks" / "validation"))

import microscopy_viewer  # noqa: E402,F401  (first, before anything else of the package)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import ndimage  # noqa: E402

from microscopy_viewer import fusion as fu  # noqa: E402
from microscopy_viewer import ims_store  # noqa: E402
from microscopy_viewer import junction as jn  # noqa: E402
from microscopy_viewer import junction_analysis as ja  # noqa: E402
from microscopy_viewer import loaders  # noqa: E402

import junction_tuning as jt  # noqa: E402
import count_common as C  # noqa: E402

PRESCREEN = Path(r"C:\Users\qfavey\Downloads\oasis_prescreen_211026.csv")
SCAN_DIR = jt.PICKS.parent
PREFIX = "2025_09_26_Asssembloid_"
TEMPLATE = HERE / "explainer_template.html"
OUTPUT = HERE / "pipeline_explainer.html"
PLACEHOLDER = "const DATA = /*__DATA__*/ null;"

#: The real sample of steps 2-9, the cells crop and the positivity section.
REAL = PREFIX + "034-033_s13"
#: The section the finder gets wrong, from the same slide: status ok, line far from the hand-drawn one.
FAILED = PREFIX + "034-033_s17"
#: Bend costs the bending section switches between (µm of plain tissue per 90° turn).
BENDS = (0, 400, 1000)
COST_POWERS = (2, 6, 10)
SIGMAS_UM = (5, 10, 20)
CELL_CROP_UM = 150.0
PROFILE_BIN_UM = 25.0
PROFILE_MAX_UM = 600.0
POSITIVITY_RULES = ("band median + 7 SD, median filter", "band median + 3 SD (current)", "ring contrast ≥ 1.5")

# Meaning colours (the spec's).
TEAL, CORAL, NAVY, AMBER, INK, PURPLE, PAPER = "#2a9d8f", "#e76f51", "#264653", "#e9b949", "#1f2430", "#8d6cab", "#faf7f2"
#: The nuclear stain in a fluorescence composite on black: navy itself would not show.
NUCLEAR_ON_BLACK = "#5a7fc4"

T0 = time.perf_counter()


def log(message: str) -> None:
    print(f"[{time.perf_counter() - T0:6.1f} s] {message}", flush=True)


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------


def quantise(values, levels: int = 4096) -> np.ndarray:
    """Float32 rounded to *levels* steps of its finite range: identical to the eye, and it gzips."""
    a = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(a)
    if not finite.any():
        return a
    low, high = float(a[finite].min()), float(a[finite].max())
    step = (high - low) / levels if high > low else 0.0
    if step <= 0:
        return a
    out = a.copy()
    out[finite] = (np.round((a[finite] - low) / step) * step + low).astype(np.float32)
    return out


def enc(values, dtype: str, levels: int | None = 4096) -> dict:
    """``{shape, dtype, gz}``: base64 of gzip of the raw little-endian C-order bytes."""
    a = np.asarray(values)
    if dtype == "float32":
        a = quantise(a, levels) if levels else a.astype(np.float32)
    a = np.ascontiguousarray(a.astype("<" + np.dtype(dtype).str[1:]))
    raw = gzip.compress(a.tobytes(), compresslevel=9, mtime=0)
    return {"shape": list(a.shape), "dtype": dtype, "gz": base64.b64encode(raw).decode("ascii")}


def dec(item: dict) -> np.ndarray:
    raw = gzip.decompress(base64.b64decode(item["gz"]))
    return np.frombuffer(raw, dtype="<" + np.dtype(item["dtype"]).str[1:]).reshape(item["shape"])


def mask(values) -> dict:
    return enc(np.asarray(values, dtype=bool), "uint8")


def f32(values, levels: int = 4096) -> dict:
    return enc(values, "float32", levels)


def points(line, digits: int = 1) -> list | None:
    if line is None:
        return None
    return [[round(float(y), digits), round(float(x), digits)] for y, x in np.asarray(line, dtype=float).reshape(-1, 2)]


def data_url(rgb, fmt: str = "JPEG", quality: int = 88) -> str:
    from PIL import Image

    a = np.asarray(rgb)
    if a.dtype != np.uint8:
        a = (np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8)
    image = Image.fromarray(a)
    buffer = io.BytesIO()
    if fmt.upper() == "JPEG":
        image.convert("RGB").save(buffer, "JPEG", quality=quality, optimize=True)
        kind = "jpeg"
    else:
        image.save(buffer, "PNG", optimize=True)
        kind = "png"
    return f"data:image/{kind};base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def clean(value):
    """Plain JSON: numpy scalars to Python, NaN and inf to null."""
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        v = float(value)
        return v if np.isfinite(v) else None
    return value


def rgb_of(colour: str) -> np.ndarray:
    colour = colour.lstrip("#")
    return np.array([int(colour[k:k + 2], 16) / 255.0 for k in (0, 2, 4)])


def bright(colour: str) -> np.ndarray:
    """The colour scaled so its brightest component is 1: same hue, full strength."""
    c = rgb_of(colour)
    return c / c.max()


def stretch(plane, valid=None, low: float = 1.0, high: float = 99.7) -> np.ndarray:
    a = np.asarray(plane, dtype=np.float32)
    sel = a[valid] if valid is not None else a[a > 0]
    if sel.size == 0:
        sel = a.ravel()
    lo, hi = np.percentile(sel, (low, high))
    return np.clip((a - lo) / max(float(hi - lo), 1e-6), 0, 1)


def composite(planes: dict) -> np.ndarray:
    """Additive fluorescence composite on black: ``{colour: plane 0..1}``."""
    first = next(iter(planes.values()))
    out = np.zeros((*first.shape, 3), dtype=np.float32)
    for colour, plane in planes.items():
        out += plane[..., None] * bright(colour)[None, None, :]
    return np.clip(out, 0, 1)


def overview_composite(ov, valid=None) -> np.ndarray:
    ch = ov.channels
    valid = (ch > 0).all(axis=0) if valid is None else valid
    # Nuclei dimmed, and each label stretched from its median, so the labelled cells — not the haze — show.
    tissue = jn.tissue_mask(ch[ov.nuclear], valid, ov.um, jn._others(ch, ov.nuclear, jn.TISSUE_ALL_CHANNELS))
    region = tissue if tissue.sum() > 100 else valid
    planes = {NUCLEAR_ON_BLACK: 0.6 * stretch(ch[ov.nuclear], region, 1, 99.5),
              TEAL: 1.3 * stretch(ch[ov.label], region, 50, 99.9) ** 0.8}
    if ov.other is not None:
        planes[CORAL] = 1.3 * stretch(ch[ov.other], region, 50, 99.9) ** 0.8
    return composite(planes)


# ---------------------------------------------------------------------------
# The finder, step by step (the same for both real sections)
# ---------------------------------------------------------------------------


def pipeline(ov, options) -> tuple[dict, dict]:
    """Every intermediate the explainer shows, computed as ``junction.py`` does.

    Returns ``(data, raw)``: the encoded entries, and a few plain arrays the
    caller needs further.
    """
    um = float(ov.um)
    ch = ov.channels
    nuc_raw = np.asarray(ch[ov.nuclear], dtype=np.float32)
    lab_raw = np.asarray(ch[ov.label], dtype=np.float32)
    oth_raw = np.asarray(ch[ov.other], dtype=np.float32) if ov.other is not None else np.zeros_like(nuc_raw)
    valid = (ch > 0).all(axis=0)
    n_scale = float(np.percentile(nuc_raw[valid], 99.8))
    l_scale = float(np.percentile(lab_raw[valid], 99.8))
    o_scale = float(np.percentile(oth_raw[valid], 99.8)) if ov.other is not None else 1.0

    # Step 1: the tissue, operation by operation (tissue_mask, unrolled): the
    # nuclear stain above its cut, or any other channel above its own.
    from skimage.filters import threshold_otsu

    smooth = ndimage.gaussian_filter(nuc_raw, 10.0 / um)
    values = smooth[valid]
    midpoint = float(threshold_otsu(values))
    background_level = float(np.median(values[values < midpoint]))
    tissue_level = float(np.median(values[values >= midpoint]))
    cut = background_level + 0.2 * (tissue_level - background_level)
    raw_nuclear = (smooth > cut) & valid
    others = jn._others(ch, ov.nuclear, options.tissue_all_channels)
    raw = raw_nuclear.copy()
    for plane in others:
        more = jn._above_background(plane, valid, um)
        if more is not None:
            raw |= more
    opened = ndimage.binary_opening(raw, jn._disk(10.0 / um))
    closed = ndimage.binary_closing(opened, jn._disk(15.0 / um))
    pieces, _count = ndimage.label(closed)
    sizes = np.bincount(pieces.ravel())[1:]
    keep = np.flatnonzero(sizes >= 0.1 * sizes.max()) + 1
    tissue = np.isin(pieces, keep)
    assert np.array_equal(tissue, jn.tissue_mask(nuc_raw, valid, um, others)), "tissue steps differ from tissue_mask"
    filled = ndimage.binary_fill_holes(tissue)
    counts, edges = np.histogram(values / n_scale, bins=80)

    # Step 2: labelled cells.
    size = max(3, int(round(options.median_um / um)) | 1)
    local_median = ndimage.median_filter(lab_raw, size)
    prominence = jn.cell_prominence(lab_raw, um, True, options.median_um)
    prominence_off = jn.cell_prominence(lab_raw, um, False)
    density = jn.cell_density(lab_raw, tissue, filled, um, median=True, median_um=options.median_um)
    density_off = jn.cell_density(lab_raw, tissue, filled, um, median=False)

    # Step 3: the seeds, mask by mask (find_seeds, unrolled).
    other_density = None
    if options.seed_from_other_channel and ov.other is not None and jn.has_cells(oth_raw, tissue, um):
        other_density = jn.cell_density(oth_raw, tissue, filled, um, median=True, median_um=options.median_um)
    normal = jn._normalised(density, filled)
    hot_all = (normal > 0.5) & filled
    hot = jn._largest(hot_all)
    from_hot = jn._through_tissue(filled, hot)
    # (name, mask, which organoid it is about, note, kept on screen in later steps)
    seed_steps = [
        ("AF488 density above half its range", hot_all, "labelled",
         "The AF488 density, scaled from 0 (its 10th percentile) to 1 (its 98th). Every pixel above 0.5 is "
         "where labelled cells are dense.", False),
        ("Its largest piece: the labelled core", hot, "labelled",
         "Only the largest connected piece is kept; smaller patches (grey) are dropped.", True)]
    cold, rule = None, jn.SEED_FAR
    other_normal = jn._normalised(other_density, filled) if other_density is not None else None
    if other_normal is not None:
        core_all = (other_normal > 0.5) & filled & ~hot & (other_normal > normal)
        seed_steps += [
            ("AF555 density above half its range", (other_normal > 0.5) & filled, "other",
             "Now the other organoid's own label (AF555), scaled the same way: every pixel above 0.5.", False),
            ("… where AF555 outweighs AF488, outside the labelled core", core_all, "other",
             "Kept only where AF555 is denser than AF488, and outside the labelled core (dashed teal); "
             "the rest (grey) is dropped.", False)]
        if core_all.any():
            core = jn._largest(core_all)
            seed_steps.append(("Its largest piece: the other core", core, "other",
                               "Again only the largest piece is kept.", True))
            if core.sum() >= jn.SEED_MIN_SHARE * filled.sum():
                cold, rule = jn._far_part(core, from_hot), jn.SEED_CHANNEL
    if cold is None:
        cold = jn._largest(filled & (from_hot > 0.5 * float(from_hot.max())) & (normal < 0.2))
        seed_steps.append(("Other seed: AF488-poor tissue far from the labelled core", cold, "other",
                           "No usable AF555 core, so the other seed is tissue with few labelled cells, further "
                           "from the labelled core (through the tissue) than half the furthest tissue.", True))
    else:
        seed_steps.append(("Pull back: the other seed", cold, "other",
                           "The half of the other core furthest, through the tissue, from the labelled core. "
                           "This is the other seed: it starts well inside its own organoid.", True))
    hot_seed = jn._far_part(hot, jn._through_tissue(filled, cold))
    seed_steps.append(("Pull back: the labelled seed", hot_seed, "labelled",
                       "The same for the labelled core: the half furthest from the other seed. This is the "
                       "labelled seed.", True))
    found = jn.find_seeds(filled, density, other_density)
    assert np.array_equal(found[0], hot_seed) and np.array_equal(found[1], cold), "seed steps differ from find_seeds"

    # Step 4: the price.
    stain = ov.nuclear_mean if ov.nuclear_mean is not None else nuc_raw
    cost = jn.line_cost(stain, filled, um)
    level = float(np.median(stain[filled]))
    flat = np.where(filled, stain, level)
    fine = ndimage.gaussian_filter(flat, jn.LINE_SMOOTH_UM / um)
    coarse = ndimage.gaussian_filter(flat, jn.LINE_SURROUND_UM / um)
    ratio = np.clip(fine / np.maximum(coarse, 1e-6), 0.3, 1.0)
    assert np.allclose((ratio ** jn.LINE_POWER).astype(np.float32), cost), "price differs from line_cost"

    # Steps 5-6: the cut at several bend costs, and the line traced along each.
    cuts, lines = {}, {}
    for bend in BENDS:
        halves = jn.split_tissue(filled, density, um, cost, seeds=(hot_seed, cold), bend_um=float(bend))
        cuts[str(bend)] = halves if halves is not None else np.zeros(filled.shape, np.uint8)
        line = jn.trace_interface(halves, um) if halves is not None else None
        lines[str(bend)] = line * um if line is not None else None
    result = jn.find_junction(ov, options)
    if result.halves is not None and not np.array_equal(result.halves, cuts[str(int(options.bend_um))]):
        log("  note: find_junction's halves differ from the unrolled cut")

    # The animation: water from each seed, slow across the cheap (dark) line.
    from skimage.graph import MCP_Geometric

    speed_cost = np.where(filled, 1.0 / np.maximum(cost, 0.05), np.inf)
    arrival = []
    for seed in (hot_seed & filled, cold & filled):
        times, _ = MCP_Geometric(speed_cost).find_costs(np.argwhere(seed))
        times = np.where(filled & np.isfinite(times), times, -1.0)
        arrival.append(times * um)          # µm of plain tissue

    band = result.band if result.band is not None else np.zeros(filled.shape, bool)
    data = {
        "um": um, "shape": list(filled.shape),
        "nuclear": f32(np.clip(nuc_raw / n_scale, 0, 1)),
        "label": f32(np.clip(lab_raw / l_scale, 0, 1)),
        "other": f32(np.clip(oth_raw / o_scale, 0, 1)),
        "smooth": f32(smooth / n_scale),
        "hist": {"edges": [round(float(e), 5) for e in edges], "counts": [int(c) for c in counts],
                 "otsu": midpoint / n_scale, "cut": cut / n_scale,
                 "background": background_level / n_scale, "tissue": tissue_level / n_scale},
        "tissue_steps": [{"name": n, "mask": mask(m)} for n, m in (
            ("1. nuclear stain above its cut", raw_nuclear), ("2. or any other channel above its own cut", raw),
            ("3. opened: specks removed", opened), ("4. closed: small gaps bridged", closed),
            ("5. big pieces kept, debris dropped", tissue), ("6. holes filled", filled))],
        "sigmas": {str(s): f32(ndimage.gaussian_filter(nuc_raw, s / um) / n_scale) for s in SIGMAS_UM},
        "median_local": f32(local_median / l_scale),
        "prominence": f32(prominence / l_scale),
        "prominence_off": f32(prominence_off / l_scale),
        "density": f32(density), "density_off": f32(density_off),
        "seed_steps": [{"name": n, "mask": mask(m), "kind": kind, "note": note, "keep": keep}
                       for n, m, kind, note, keep in seed_steps],
        "other_density": f32(other_density) if other_density is not None else None,
        "hot": mask(hot_seed & filled), "cold": mask(cold & filled), "second_seed": rule,
        "cost": f32(cost),
        "cost_powers": {str(p): f32(ratio ** p) for p in COST_POWERS},
        "arrival_hot": f32(arrival[0]), "arrival_cold": f32(arrival[1]),
        "cuts": {k: enc(v, "uint8") for k, v in cuts.items()},
        "lines": {k: points(v) for k, v in lines.items()},
        "band": mask(band), "tissue": mask(filled), "tissue_holes": mask(tissue),
        "rim_um": float(options.margin_um), "gap_um": float(options.gap_um), "width_um": float(options.width_um),
        "line": points(result.line_um), "band_outline": points(result.band_outline_um),
        "side": result.side, "status": result.status, "flags": list(result.flags),
        "metrics": {k: round(float(v), 4) for k, v in result.metrics.items()},
        "settings": {"median_um": float(options.median_um), "line_smooth_um": jn.LINE_SMOOTH_UM,
                     "line_surround_um": jn.LINE_SURROUND_UM, "line_power": jn.LINE_POWER,
                     "junction_smooth_um": jn.JUNCTION_SMOOTH_UM, "bend_um": float(options.bend_um),
                     "density_radius_um": 100.0, "tissue_smooth_um": 10.0, "open_um": 10.0, "close_um": 15.0},
    }
    return data, {"filled": filled, "valid": valid, "result": result}


# ---------------------------------------------------------------------------
# The minimum-cut toy: a hand-made price grid, an illustration of the operation
# ---------------------------------------------------------------------------


def toy_presets() -> dict:
    h, w = 7, 11
    uniform = np.ones((h, w))
    slanted = uniform.copy()
    for r in range(h):
        slanted[r, 3 + (4 * r) // (h - 1)] = 0.1
    speck = uniform.copy()
    speck[3, 5] = 0.05
    gappy = slanted.copy()
    gappy[3, 5] = gappy[2, 4] = 1.0
    # Checked against the package's own cut, so the front-end's max-flow can be compared.
    presets, cuts = {}, {}
    for name, price in (("uniform", uniform), ("slanted", slanted), ("speck", speck), ("gappy", gappy)):
        region = np.ones((h, w), bool)
        left = np.zeros((h, w), bool); left[:, 0] = True
        right = np.zeros((h, w), bool); right[:, -1] = True
        side = jn._min_cut(region, left, right, price.astype(np.float32))
        presets[name] = enc(price, "float32", None)
        cuts[name] = mask(side)
    return {"presets": presets, "cuts_8_neighbour": cuts,
            "note": "cuts_8_neighbour: the package's own cut (8 neighbours, diagonals x0.7071) — the "
                    "explainer's 4-neighbour cut may differ on ties"}


# ---------------------------------------------------------------------------
# The real sample
# ---------------------------------------------------------------------------


def scan_path(name: str) -> Path:
    return SCAN_DIR / f"{name}.ome.tiff"


def real_sample(name: str, fullres_crop: dict | None) -> dict:
    """One annotated section through the finder, with its hand-drawn line and how far the finder is from it."""
    ov, _manual_band, manual_line, base = jt.load_case(name)
    options = jn.JunctionOptions(save=False, width_um=base["width_um"], gap_um=base["gap_um"])
    data, raw = pipeline(ov, options)
    second = annotator_lines(name).get("second")
    result = raw["result"]
    error = None
    if result.line_um is not None:
        region = ndimage.binary_dilation(result.filled, iterations=max(1, int(np.ceil(base["margin_um"] / ov.um))))
        error = round(float(jt.line_distance(manual_line, result.line_um, region, ov.um)[0]), 1)
    data.update({
        "name": name, "short": C.short(name),
        "composite": data_url(overview_composite(ov, raw["valid"]), "JPEG", 90),
        "manual_line": points(manual_line), "second_line": points(second),
        "truth": points(manual_line),
        "error_um": error,
    })
    if fullres_crop is not None:
        data["fullres"] = fullres_crop
    return data


# ---------------------------------------------------------------------------
# Validation tables, read without ever writing beside the scans
# ---------------------------------------------------------------------------


def private_tables() -> None:
    """Point ``count_common`` at a copy of its geometry cache inside ``_build``.

    ``load_sample`` rebuilds and *writes* the cache when it is stale; with the
    copy here, a rebuild would land in ``_build`` and never beside the scans.
    """
    source, target = C.TABLES, BUILD / "cell_tables"
    target.mkdir(parents=True, exist_ok=True)
    for f in source.glob("*.*"):
        if f.suffix in (".json", ".parquet"):
            t = target / f.name
            if not t.exists() or t.stat().st_mtime < f.stat().st_mtime:
                shutil.copy2(f, t)
    C.TABLES = target


_SCORES = None


def scores() -> pd.DataFrame:
    global _SCORES
    if _SCORES is None:
        s = pd.read_csv(jt.OUT / "scores.csv")
        _SCORES = s[s["variant"] == "bend 1000"].set_index("name")
    return _SCORES


def score_of(name: str) -> dict:
    s = scores()
    if name not in s.index:
        return {}
    row = s.loc[name]
    err = float(row["mean_dist"]) if pd.notna(row["mean_dist"]) else None
    flags = [f.strip() for f in str(row["flags"]).split(";")] if pd.notna(row["flags"]) else []
    return {"status": str(row["status"]), "flags": flags, "error_um": round(err, 1) if err is not None else None,
            "dice": round(float(row["dice"]), 3) if pd.notna(row["dice"]) else None}


def annotator_lines(name: str) -> dict:
    """The hand-drawn lines (µm) by annotator: first, second (where drawn)."""
    path = str(scan_path(name))
    out = {}
    for who, key in C.ANNOTATORS.items():
        _labels, attrs = (None, {})
        if key in ims_store.list_labels(path):
            _labels, attrs = ims_store.load_labels(path, key)
        if "line_yx_um" in attrs:
            out[who] = np.asarray(attrs["line_yx_um"], dtype=float).reshape(-1, 2)
    return out


# ---------------------------------------------------------------------------
# Galleries
# ---------------------------------------------------------------------------


def crop_box(mask_, margin: int, shape) -> tuple[slice, slice]:
    rows, cols = np.flatnonzero(mask_.any(axis=1)), np.flatnonzero(mask_.any(axis=0))
    if rows.size == 0:
        return slice(0, shape[0]), slice(0, shape[1])
    return (slice(max(rows[0] - margin, 0), min(rows[-1] + margin + 1, shape[0])),
            slice(max(cols[0] - margin, 0), min(cols[-1] + margin + 1, shape[1])))


def render_lines(background: np.ndarray, um: float, box, lines: list, width_px: int = 420) -> str:
    """A picture of *background* (RGB 0..1, overview grid) cropped to *box*, with lines on it.

    *lines*: ``(points µm, colour, dashed, width)``.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    crop = background[box]
    h, w = crop.shape[:2]
    scale = width_px / w
    fig = plt.figure(figsize=(width_px / 100, h * scale / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(crop, interpolation="bilinear", extent=(-0.5, w - 0.5, h - 0.5, -0.5))
    for line, colour, dashed, lw in lines:
        if line is None or len(line) < 2:
            continue
        curve = fu.fit_spline(np.asarray(line, float), spacing=um) / um
        ax.plot(curve[:, 1] - box[1].start, curve[:, 0] - box[0].start, color=colour, lw=lw,
                ls=(0, (5, 3)) if dashed else "-", solid_capstyle="round")
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)
    ax.set_axis_off()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=100)
    plt.close(fig)
    from PIL import Image

    image = np.asarray(Image.open(io.BytesIO(buffer.getvalue())).convert("RGB"))
    return data_url(image, "JPEG", 85)


def paper_nuclear(ov, filled) -> np.ndarray:
    """Nuclear stain as dark ink on paper, so dark and coloured lines both read on it."""
    stain = ov.nuclear_mean if ov.nuclear_mean is not None else ov.channels[ov.nuclear]
    valid = (ov.channels > 0).all(axis=0)
    s = stretch(stain, valid & filled if filled.any() else valid, 2, 99.5)
    paper, ink = rgb_of(PAPER), rgb_of(NAVY)
    return paper[None, None, :] * (1 - s[..., None] * 0.92) + ink[None, None, :] * (s[..., None] * 0.92)


JUNCTION_LEGEND = ("Nuclear stain, dark on light. Dashed dark line: drawn by hand (first annotator); "
                   "purple: drawn by a second annotator; amber: the automatic junction finder.")

#: The junction cases, with what each shows, in plain words.
JUNCTION_CASES = [
    ("good_junction", PREFIX + "055-054_s15", "A clean junction",
     "Two organoids pressed together, with a clear dark seam in the nuclear stain between them. The finder's "
     "line runs along the seam, within about 10 µm of the hand-drawn one: as close as two people are to each other."),
    ("good_junction", PREFIX + "034-033_s13", "A clean junction, despite cracks",
     "The section has cracks running across both organoids, but none crosses from one to the other. The price on "
     "bending keeps the cut on the seam, and the line lands within 10 µm of both annotators'."),
    ("good_junction", PREFIX + "050-049_s5", "A narrow neck",
     "Where two organoids meet at a narrow neck, the shortest cut and the dark seam agree. The finder and the "
     "people agree to about 12 µm."),
    ("failure", PREFIX + "033-032_s12", "A tear taken for the junction",
     "Deep cracks run through this damaged section, and the bottom of it is cut off by the edge of the scan. A crack "
     "is the darkest, cheapest thing to cut along, so the finder follows one instead of the seam the annotators "
     "drew, nearly 400 µm away. Triage catches it: its shape is Damaged."),
    ("failure", PREFIX + "034-033_s17", "Looks fine, is wrong",
     "The worst kind: the finder reports nothing wrong (status ok), yet its line turns away along a crack, about "
     "350 µm from where both annotators agree the junction is. Only a look at the QC picture catches this."),
    ("failure", PREFIX + "041-040_s1", "Damaged shape",
     "The section is torn and misshapen, so the labelled cells and the dark seam do not mark one clear border. "
     "The finder cuts across the wrong part of the tissue, more than 300 µm off. Triage sends Damaged sections to "
     "be drawn by hand."),
    ("failure", PREFIX + "052-051_s23", "Not fused: no junction to find",
     "The two organoids barely touch: there is no shared border, so the finder stops with 'organoids not fused' "
     "and draws nothing. This is a correct refusal; a person decides whether a junction can be drawn at all."),
    ("failure", PREFIX + "052-051_s14", "Cut off by the scan edge",
     "The section lies against the edge of the scanned area, and a straight edge of the scan is not an organoid "
     "outline. The finder flags 'cut off by the scan edge' and finds no junction; the annotators drew it by hand."),
    ("failure", PREFIX + "037-036_s25", "Two people disagree",
     "On this damaged section the two annotators' lines are about 130 µm apart: even people do not agree where "
     "the junction is. The finder's line lies near the second annotator's. When people disagree this much, "
     "no automatic line can be judged right or wrong against just one of them."),
]


def picks_classes() -> pd.DataFrame:
    p = pd.read_csv(jt.PICKS, encoding="utf-8-sig")
    p["name"] = [Path(x).name.replace(".ome.tiff", "") for x in p["path"]]
    return p.set_index("name")


def classes_of(row) -> dict:
    return {k: (None if pd.isna(row[k]) else str(row[k])) for k in ("Pass", "Shape", "Ripped", "Folded", "Quality")}


def junction_gallery() -> list[dict]:
    picks = picks_classes()
    items = []
    for category, name, title, why in JUNCTION_CASES:
        ov, _band, manual_line, base = jt.load_case(name)
        result = jn.find_junction(ov, jn.JunctionOptions(save=False, **base))
        filled = jn.tissue_of(ov)
        lines = annotator_lines(name)
        background = paper_nuclear(ov, filled)
        box = crop_box(filled | ((ov.channels > 0).all(axis=0) & False), int(60 / ov.um), filled.shape)
        drawn = [(lines.get("second"), PURPLE, False, 2.0), (manual_line, INK, True, 2.2),
                 (result.line_um, AMBER, False, 3.0)]
        s = score_of(name)
        items.append({
            "id": C.short(name), "name": name, "category": category, "title": title,
            "classes": classes_of(picks.loc[name]) if name in picks.index else None,
            "status": result.status, "flags": list(result.flags), "error_um": s.get("error_um"),
            "second_annotator": "second" in lines, "why": why,
            "img": render_lines(background, ov.um, box, drawn), "legend": JUNCTION_LEGEND,
        })
        log(f"  gallery {C.short(name)}: {result.status}, error {s.get('error_um')}")
    return items


#: Pre-screen classes shown in the organoid gallery: key, title, filter, why.
ORGANOID_CLASSES = [
    ("good-1", "Pass: a fused pair, good quality",
     lambda d: (d.Pass == "yes") & (d.Shape == "Fused pair") & (d.Quality == "Good") & (d.Ripped == "no") & (d.Folded == "no"),
     "Two organoids grown into one, cut cleanly: one end full of labelled (teal) cells, the other of the second "
     "label (coral). This is what the analysis is made for."),
    ("good-2", "Pass: a fused pair, good quality",
     lambda d: (d.Pass == "yes") & (d.Shape == "Fused pair") & (d.Quality == "Good") & (d.Ripped == "no") & (d.Folded == "no"),
     "Another clean fused pair. The two organoids need not be the same size or equally bright."),
    ("damaged", "Shape: damaged",
     lambda d: d.Shape == "Damaged",
     "The section is broken up or misshapen. A junction may still be there, but cracks and missing pieces can "
     "look like one, so the junction is drawn by hand."),
    ("separated", "Shape: separated",
     lambda d: d.Shape == "Separated",
     "Two organoids that never fused, or came apart: there is no junction for cells to cross."),
    ("empty", "Shape: empty or debris",
     lambda d: d.Shape == "Empty / debris",
     "No organoid worth measuring in this scan: debris, a fragment, or nothing at all."),
    ("ripped", "Ripped",
     lambda d: (d.Ripped == "yes") & (d.Shape == "Fused pair"),
     "A tear runs through the section. Tears are dark, like the junction, so they can mislead the junction finder."),
    ("folded", "Folded",
     lambda d: (d.Folded == "yes") & (d.Shape == "Fused pair") & (d.Ripped == "no"),
     "Part of the section lies folded over itself: two layers of tissue on top of each other look brighter and "
     "denser than they are."),
    ("bad", "Quality: bad",
     lambda d: (d.Quality == "Bad") & (d.Shape == "Fused pair") & (d.Ripped == "no") & (d.Folded == "no"),
     "The tissue is there, but the staining or the focus is poor: weak labels, blur, or uneven brightness."),
]

#: Hand-picked rows (object_id) overriding the first match of a class, after looking at them.
ORGANOID_OVERRIDES: dict[str, str] = {}

ORGANOID_LEGEND = ("Fluorescence composite: teal = AF488 (the labelled organoid), coral = AF555 (the other "
                   "organoid), blue = nuclear stain.")


def organoid_gallery(fresh: bool) -> list[dict]:
    table = pd.read_csv(PRESCREEN)
    table = table[[Path(p).exists() for p in table["path"]]]
    used, items = set(), []
    for key, title, rule, why in ORGANOID_CLASSES:
        rows = table[rule(table) & ~table["object_id"].isin(used)]
        if key in ORGANOID_OVERRIDES:
            rows = table[table["object_id"] == ORGANOID_OVERRIDES[key]]
        if rows.empty:
            log(f"  organoid {key}: no row")
            continue
        # Prefer rows a person classified, on every column the class rests on.
        manual = (rows[["Pass source", "Shape source", "Quality source"]] == "manual").sum(axis=1)
        rows = rows.assign(_manual=manual).sort_values(["_manual", "object_id"], ascending=[False, True])
        row = rows.iloc[0]
        used.add(row["object_id"])
        cached = CACHE / f"organoid_{row['object_id']}.jpg.txt"
        if cached.exists() and not fresh:
            img = cached.read_text(encoding="ascii")
        else:
            img = render_organoid(Path(row["path"]))
            cached.write_text(img, encoding="ascii")
        items.append({
            "id": f"organoid-{key}", "name": str(row["object_id"]), "category": "organoid", "title": title,
            "classes": classes_of(row), "status": None, "flags": [], "error_um": None, "why": why,
            "img": img, "legend": ORGANOID_LEGEND,
        })
        log(f"  organoid {key}: {row['object_id']}")
    return items


def render_organoid(path: Path, width_px: int = 420) -> str:
    from PIL import Image

    specs = loaders.load_path(path)
    try:
        ov = jn.build_overview(specs, jn.JunctionOptions(save=False))
    finally:
        loaders.release(path)
    valid = (ov.channels > 0).all(axis=0)
    rgb = overview_composite(ov, valid)
    tissue = jn.tissue_mask(ov.channels[ov.nuclear], valid, ov.um)
    box = crop_box(tissue, int(80 / ov.um), tissue.shape) if tissue.any() else (slice(None), slice(None))
    crop = (rgb[box] * 255 + 0.5).astype(np.uint8)
    h, w = crop.shape[:2]
    scale = width_px / max(w, 1)
    image = Image.fromarray(crop).resize((width_px, max(1, int(round(h * scale)))), Image.LANCZOS)
    return data_url(np.asarray(image), "JPEG", 85)


# ---------------------------------------------------------------------------
# Cells: positivity, profile, and the crop
# ---------------------------------------------------------------------------


def load_cells():
    cells, meta = C.load_all(progress=True)
    dots = C.load_dots(cells, meta)
    return cells, meta, dots


def oriented(df, label, band="first") -> np.ndarray:
    """Signed distance to *band*'s line, negative on the side where *label* is at home."""
    return C.away_distance(df, label, band)


def positivity(df, meta_row, label) -> dict:
    band = "first"
    away = oriented(df, label, band)
    beside = df[f"beside {band}"].to_numpy()
    in_band = df[f"in {band}"].to_numpy()
    keep = in_band | (beside & (np.abs(away) <= PROFILE_MAX_UM))
    sub = df[keep]
    raw = sub[C.column(label, 3.0, "raw", "mean")].to_numpy(float)
    filtered = sub[C.column(label, 3.0, "median", "mean")].to_numpy(float)
    ring = sub[C.ring_column(label)].to_numpy(float)
    contrast = C.contrast(sub, label)
    centre_raw, spread_raw = ja.robust_centre_and_spread(df[C.column(label, 3.0, "raw", "mean")].to_numpy()[in_band])
    centre_f, spread_f = ja.robust_centre_and_spread(df[C.column(label, 3.0, "median", "mean")].to_numpy()[in_band])
    neg_threshold = None
    values_all = df[C.column(label)].to_numpy(float)
    for deep in (C.DEEP_UM, 200.0, 100.0):
        neg = C.negatives(df, label, band, deep) & np.isfinite(values_all)
        if neg.sum() >= 50:
            neg_threshold = float(np.percentile(values_all[neg], 99.0))
            break
    rules = {
        "band median + 7 SD, median filter": centre_f + 7 * spread_f,
        "band median + 3 SD (current)": centre_raw + 3 * spread_raw,
        "ring contrast ≥ 1.5": 1.5,
        "negative control 99th pct": neg_threshold,
    }
    # The package's own calls, to check the thresholds against.
    called = {name: int(C.rules(meta_row["channels"], label)[name](df, band)[keep].sum())
              for name in ("band median + 7 SD, median filter", "band median + 3 SD (current)", "ring contrast ≥ 1.5")}
    line, flip = meta_row["lines"][band]
    path = meta_row["path"]
    labels, _ = ims_store.load_labels(path, C.ANNOTATORS[band])
    labels = np.asarray(labels).reshape(np.shape(labels)[-2:])
    voxel = np.asarray(meta_row["voxel"], float)
    from skimage.measure import find_contours

    small = labels[::8, ::8] > 0
    contour = max(find_contours(np.pad(small, 1).astype(float), 0.5), key=len) - 1.0
    band_outline = contour[:: max(1, len(contour) // 500)] * 8 * voxel
    ov = jt.load_case(meta_row.name)[0]
    tissue_outline = jn._outline(jn.tissue_of(ov)) * ov.um
    cols = {
        "y_um": enc(sub["y (um)"].to_numpy(), "float32", None),
        "x_um": enc(sub["x (um)"].to_numpy(), "float32", None),
        "raw_mean": enc(np.round(raw, 1), "float32", None),
        "filtered_mean": enc(np.round(filtered, 1), "float32", None),
        "ring": enc(np.round(ring, 1), "float32", None),
        "contrast": enc(np.round(contrast, 3), "float32", None),
        "in_band": enc(sub[f"in {band}"].to_numpy(), "uint8"),
        "from_line_um": enc(np.round(df[f"from {band} (um)"].to_numpy(float)[keep], 1), "float32", None),
        "edge_um": enc(np.round(sub["edge (um)"].to_numpy(float), 1), "float32", None),
        "clicked": enc(sub["clicked"].to_numpy() if "clicked" in sub else np.zeros(len(sub), bool), "uint8"),
    }
    return {
        "name": meta_row.name, "channel": label, "cells": int(len(sub)),
        "line": points(line), "band_outline": points(band_outline), "tissue_outline": points(tissue_outline),
        "cols": cols, "rules": rules,
        "rule_columns": {"band median + 7 SD, median filter": "filtered_mean", "band median + 3 SD (current)": "raw_mean",
                         "ring contrast ≥ 1.5": "contrast", "negative control 99th pct": "raw_mean"},
        "band_stats": {"raw_mean": {"median": centre_raw, "sd": spread_raw},
                       "filtered_mean": {"median": centre_f, "sd": spread_f}},
        "called": called,
        "note": "from_line_um: signed distance to the first annotator's line, + into the band's side (where AF488 "
                "is not at home, in 30 of 30 samples); "
                "SD = 1.4826 x MAD over the band's cells",
    }


def profile(cells, meta) -> dict:
    edges = np.arange(-PROFILE_MAX_UM, PROFILE_MAX_UM + PROFILE_BIN_UM, PROFILE_BIN_UM)
    centres = (edges[:-1] + edges[1:]) / 2
    out = {r: {"median": None, "per_sample": []} for r in POSITIVITY_RULES}
    names, counts = [], []
    for name, df in cells.items():
        m = meta.loc[name]
        if "first" not in m["bands"]:
            continue
        label = C.label_channel(meta)
        rules = C.rules(m["channels"], label)
        away = oriented(df, label)
        beside = df["beside first"].to_numpy()
        which = np.digitize(away, edges) - 1
        ok = beside & (which >= 0) & (which < len(centres))
        n = np.bincount(which[ok], minlength=len(centres)).astype(float)
        for r in POSITIVITY_RULES:
            pos = rules[r](df, "first")
            k = np.bincount(which[ok], weights=pos[ok].astype(float), minlength=len(centres))
            pct = np.where(n >= 20, 100.0 * k / np.maximum(n, 1), np.nan)
            out[r]["per_sample"].append([round(float(v), 3) if np.isfinite(v) else None for v in pct])
        names.append(C.short(name))
        counts.append(int(n.sum()))
    for r in POSITIVITY_RULES:
        a = np.array([[np.nan if v is None else v for v in row] for row in out[r]["per_sample"]], float)
        med = np.nanmedian(a, axis=0)
        out[r]["median"] = [round(float(v), 3) if np.isfinite(v) else None for v in med]
    return {"distance_um": [float(c) for c in centres], "bin_um": PROFILE_BIN_UM, "samples": names,
            "cells_per_sample": counts, "rules": out,
            "note": "% of AF488 cells positive per 25 µm bin of signed distance to the first annotator's line "
                    "(negative = AF488's home side); bins with < 20 cells are null"}


def choose_crop(df, dots_here, voxel, shape) -> tuple[int, int]:
    """Centre (full-res px) of a 150 µm window holding the most clicked cells near the line."""
    half = CELL_CROP_UM / 2
    pts = dots_here[["y (um)", "x (um)"]].to_numpy()
    best, best_score = None, -1.0
    for y, x in pts:
        inside = (np.abs(pts[:, 0] - y) < half * 0.85) & (np.abs(pts[:, 1] - x) < half * 0.85)
        score = inside.sum()
        if score > best_score:
            best, best_score = (y, x), score
    if best is None:                                        # no clicks: the band's middle
        b = df[df["in first"]]
        best = (float(b["y (um)"].median()), float(b["x (um)"].median()))
    cy, cx = int(round(best[0] / voxel[0])), int(round(best[1] / voxel[1]))
    n = int(round(CELL_CROP_UM / voxel[0]))
    cy = int(np.clip(cy, n // 2, shape[0] - n // 2 - 1))
    cx = int(np.clip(cx, n // 2, shape[1] - n // 2 - 1))
    return cy, cx


def cells_crop(cells, meta, dots, fresh: bool) -> tuple[dict, dict]:
    """The cells section: one crop, segmented on the CPU, grown, ringed and measured."""
    from microscopy_viewer import segmentation as sg

    name = REAL
    df, m = cells[name], meta.loc[name]
    label = C.label_channel(meta)
    voxel = np.asarray(m["voxel"], float)
    shape = tuple(int(v) for v in m["shape"])
    dots_here = dots[dots["name"] == name] if len(dots) else pd.DataFrame(columns=["y (um)", "x (um)"])
    cy, cx = choose_crop(df, dots_here, voxel, shape)
    n = int(round(CELL_CROP_UM / voxel[0]))
    y0, x0 = cy - n // 2, cx - n // 2
    pad = int(round(120.0 / voxel[0]))
    py0, px0 = max(y0 - pad, 0), max(x0 - pad, 0)
    py1, px1 = min(y0 + n + pad, shape[0]), min(x0 + n + pad, shape[1])
    cache = CACHE / f"cells_{C.short(name)}_{y0}_{x0}_{n}.npz"
    if cache.exists() and not fresh:
        d = np.load(cache)
        dapi_pad, af488_pad, nuclei = d["dapi"], d["af488"], d["nuclei"]
    else:
        path = scan_path(name)
        specs = loaders.load_path(path)
        try:
            names = [str(s.channel_name or s.name) for s in specs]
            nuc = jn.pick_channel(names, "", jn.NUCLEAR_HINTS)
            lab = names.index(label)
            window = (slice(py0, py1), slice(px0, px1))
            dapi_pad = np.asarray(ja._plane(specs[nuc], window))
            af488_pad = np.asarray(ja._plane(specs[lab], window))
        finally:
            loaders.release(path)
        inner = (slice(y0 - py0, y0 - py0 + n), slice(x0 - px0, x0 - px0 + n))
        log(f"  segmenting a {n} x {n} px crop on the CPU ...")
        settings = sg.SegmentationSettings(use_gpu=False, expand_um=0.0)
        nuclei = np.asarray(sg.segment_volume(dapi_pad[inner], tuple(voxel), settings=settings).masks).astype(np.int32)
        np.savez_compressed(cache, dapi=dapi_pad, af488=af488_pad, nuclei=nuclei)
    inner = (slice(y0 - py0, y0 - py0 + n), slice(x0 - px0, x0 - px0 + n))
    dapi = dapi_pad[inner].astype(np.float32)
    af488 = af488_pad[inner].astype(np.float32)
    filtered = C.cc.less_local_median(af488_pad, float(voxel[0]))[inner]

    from microscopy_viewer import segmentation as sg  # noqa: F811

    grown = sg.expand_labels(nuclei, tuple(voxel), 3.0).astype(np.int32)
    outer = sg.expand_labels(nuclei, tuple(voxel), 6.0)
    ring = np.where(grown == 0, outer, 0).astype(np.int32)
    ids = np.unique(nuclei[nuclei > 0]).astype(np.int64)
    s_cell = ja.label_stats(grown, af488, ids)
    s_ring = ja.label_stats(ring, af488, ids)
    s_filt = ja.label_stats(grown, filtered, ids)
    centres = np.asarray(ndimage.center_of_mass(nuclei > 0, nuclei, ids)).reshape(-1, 2)
    area = np.asarray(ndimage.sum(nuclei > 0, nuclei, ids)) * voxel[0] * voxel[1]
    cell, ring_mean, filt = s_cell["mean"], s_ring["mean"], s_filt["mean"]
    local = cell - filt
    background = np.where(np.isfinite(ring_mean) & (ring_mean > 0), ring_mean, local)
    contrast = cell / np.where(background > 0, background, np.nan)

    gy, gx = centres[:, 0] + y0, centres[:, 1] + x0
    pts_um = np.column_stack([gy * voxel[0], gx * voxel[1]])
    line, flip = m["lines"]["first"]
    from_line = C._side(pts_um, line)[0] * flip        # + into the band's side, as count_common
    edge = C._edge(name, gy, gx)
    band_labels, _ = ims_store.load_labels(m["path"], C.ANNOTATORS["first"])
    band_labels = np.asarray(band_labels).reshape(np.shape(band_labels)[-2:])
    rr = np.clip(np.round(gy).astype(int), 0, shape[0] - 1)
    cc_ = np.clip(np.round(gx).astype(int), 0, shape[1] - 1)
    in_band = band_labels[rr, cc_] > 0
    band_crop = band_labels[y0:y0 + n, x0:x0 + n] > 0
    clicked = np.zeros(len(ids), bool)
    if len(dots_here):
        from scipy.spatial import cKDTree

        dist, idx = cKDTree(pts_um).query(dots_here[["y (um)", "x (um)"]].to_numpy())
        for d_, j in sorted(zip(dist, idx)):
            if d_ <= C.MATCH_UM:
                clicked[j] = True
    table = []
    for k, cid in enumerate(ids):
        table.append({
            "id": int(cid), "y": round(float(centres[k, 0]), 1), "x": round(float(centres[k, 1]), 1),
            "area_um2": round(float(area[k]), 1), "mean": round(float(cell[k]), 1),
            "ring": round(float(ring_mean[k]), 1) if np.isfinite(ring_mean[k]) else None,
            "contrast": round(float(contrast[k]), 3) if np.isfinite(contrast[k]) else None,
            "filtered_mean": round(float(filt[k]), 1), "in_band": bool(in_band[k]),
            "from_line_um": round(float(from_line[k]), 1), "edge_um": round(float(edge[k]), 1),
            "clicked": bool(clicked[k]),
        })
    d_grey = stretch(dapi, None, 1, 99.7)
    a_str = stretch(af488, None, 1, 99.8)
    dots_crop = []
    for y, x in dots_here[["y (um)", "x (um)"]].to_numpy():
        py, px = y / voxel[0] - y0, x / voxel[1] - x0
        if 0 <= py < n and 0 <= px < n:
            dots_crop.append([round(float(py), 1), round(float(px), 1)])
    out = {
        "name": name, "um_per_px": float(voxel[0]), "size_px": [n, n],
        "origin_um": [round(y0 * voxel[0], 2), round(x0 * voxel[1], 2)], "channel": label,
        "dapi": data_url(d_grey, "PNG"),
        "af488": data_url(composite({TEAL: a_str}), "PNG"),
        "composite": data_url(composite({NUCLEAR_ON_BLACK: 0.85 * d_grey, TEAL: a_str}), "PNG"),
        "nuclei": enc(nuclei, "int32"), "grown": enc(grown, "int32"), "ring": enc(ring, "int32"),
        "band": mask(band_crop), "dots": dots_crop,
        "grow_um": 3.0, "ring_um": [3.0, 6.0],
        "table": table,
        "note": "table y, x: centroid in crop pixels; from_line_um + into the band's side; "
                "nuclei by Cellpose (CPU) on this crop",
    }
    # The overview step on full-resolution pixels: a window of 12 x 12 overview blocks of the crop.
    step = int(jt.load_case(name)[0].step)
    k = min(12 * step, n)
    fullres = {"um_per_px": float(voxel[0]), "step": step,
               "origin_um": [round(y0 * voxel[0], 2), round(x0 * voxel[1], 2)],
               "nuclear": enc(dapi_pad[inner][:k, :k], "uint16"), "label": enc(af488_pad[inner][:k, :k], "uint16")}
    log(f"  cells crop: {len(ids)} nuclei, {int(clicked.sum())} clicked, {int(in_band.sum())} in the band")
    return out, fullres


# ---------------------------------------------------------------------------
# Key numbers (hard-coded from the validation notebooks' "What it showed" cells, 21 October 2026)
# ---------------------------------------------------------------------------


NUMBERS = {
    "triage": {
        "source": "01_junction_evaluation.ipynb, 21 October 2026",
        "n_samples": 30, "excluded": "028-027_s20 (organoids separated)",
        "failed_over_100_um": 8,
        "check_status_sends": 21, "check_status_fine": 14,
        "both_sides_flag_good": 12, "both_sides_flag_bad": 3,
        "rule": "status is fail, or Shape is Damaged → draw the junction by hand",
        "rule_sends": 7, "rule_real_failures": 7, "rule_good_sent": 0,
        "missed": 1, "missed_sample": "034-033_s17", "missed_why": "status ok, line wrong: only a look at the QC picture catches it",
        "damaged": 5, "kept": 23,
        "downstream_mean_diff_pp": {"AF488": 0.04, "AF555": -0.20},
        "downstream_median_abs_diff_pp": {"AF488": 0.5, "AF555": 0.75},
        "downstream_spearman": {"AF488": 0.95, "AF555": 0.93},
        "limits_of_agreement_pp": 2.0, "between_sample_sd_pp": [3.0, 3.5],
        "workflow": {"mean_diff_pp": {"AF488": 0.03, "AF555": -0.18}, "median_abs_diff_pp": {"AF488": 0.25, "AF555": 0.54},
                     "worst_pp": 3.1, "drawn_by_hand": 7},
        "decision": ["Use the finder on pass samples; draw by hand where status is fail or Shape is Damaged; glance at "
                     "every other QC picture (accept / redraw).",
                     "Fine for group comparisons; for a single-sample claim, or a 1-2 pp expected difference, draw by hand."],
    },
    "annotators": {
        "source": "01_junction_evaluation.ipynb, 21 October 2026 (27 samples drawn by both)",
        "samples_both": 27, "line_distance_median_um": 15, "line_distance_mean_um": 28, "dice": 0.95,
        "over_100_um": 1, "over_100_sample": "037-036_s25", "over_100_um_value": 128,
        "af488_pp_range": [-0.9, 0.8], "af488_ccc": 0.99, "af555_pp_range": [-2.3, 2.0], "af555_ccc": 0.95,
        "kept_samples": 22,
        "table": [
            {"pair": "second vs first", "line_median_um": 14, "over_100_um": 0, "af488_loa_pp": 0.85, "af555_loa_pp": 2.2},
            {"pair": "finder vs first", "line_median_um": 25, "over_100_um": 1, "af488_loa_pp": 2.1, "af555_loa_pp": 2.0},
            {"pair": "finder vs second", "line_median_um": 17, "over_100_um": 1, "af488_loa_pp": 1.9, "af555_loa_pp": 2.0},
        ],
        "finder_vs_people": "the finder is about 1.2-1.8x as far from a person as two people are from each other",
        "band_jaccard": {"first-second": 0.92, "first-automatic": 0.81, "second-automatic": 0.87},
    },
    "positivity": {
        "source": "03_positivity_metrics.ipynb, 09_manual_vs_auto.ipynb, 21 October 2026 (30 samples, 920 000 cells; clicks on 12)",
        "cells": 920000, "samples": 30, "clicked_samples": 12, "clicks": 822,
        "chosen": "band median + 7 SD, median filter",
        "check": "ring contrast ≥ 1.5",
        "table": [
            {"rule": "median + 7 SD, median filter", "false_pos_pct": 0.03, "step_at_line": 2.4, "migrated_per_mm": 11, "f1_clicks": 0.38, "spearman_clicks": 0.73},
            {"rule": "ring contrast ≥ 1.5", "false_pos_pct": 0.07, "step_at_line": 1.5, "migrated_per_mm": 8, "f1_clicks": 0.25, "spearman_clicks": 0.53},
            {"rule": "median + 3 SD (current)", "false_pos_pct": 0.54, "step_at_line": 2.0, "migrated_per_mm": 80, "f1_clicks": 0.13, "spearman_clicks": 0.03},
            {"rule": "ring contrast > negative 99th, no autofluorescence", "false_pos_pct": 0.52, "step_at_line": 1.9, "migrated_per_mm": 45, "f1_clicks": 0.17, "spearman_clicks": 0.59},
            {"rule": "negative control 99th pct", "false_pos_pct": 2.13, "step_at_line": 1.9, "migrated_per_mm": 121, "f1_clicks": 0.09, "spearman_clicks": 0.28},
            {"rule": "negative control 99.9th pct", "false_pos_pct": 0.67, "step_at_line": 1.7, "migrated_per_mm": 23, "f1_clicks": 0.21, "spearman_clicks": 0.18},
            {"rule": "GMM on log intensity", "false_pos_pct": 0.43, "step_at_line": 1.4, "migrated_per_mm": 66, "f1_clicks": 0.02, "spearman_clicks": 0.20},
            {"rule": "Otsu on log intensity", "false_pos_pct": 31.0, "step_at_line": 1.0, "migrated_per_mm": 2384, "f1_clicks": 0.01, "spearman_clicks": 0.06},
        ],
        "current_overcounts_x": 7, "current_spearman_clicks": 0.03,
        "cell_by_cell": {"calibrated": {"precision": 0.29, "recall": 0.55, "f1": 0.38},
                         "current": {"precision": 0.07, "recall": 0.57, "f1": 0.13, "called_not_clicked": 4320}},
        "per_sample_vs_clicks": {"current_spearman": -0.01, "current_x_clicks": "4-7x",
                                 "calibrated_spearman": 0.78, "calibrated_x_clicks": 0.87,
                                 "contrast_spearman": 0.61, "contrast_x_clicks": 1.8},
        "clicks_where": {"median_from_line_um": [95, 154], "median_from_edge_um": [118, 284],
                         "clicks_on_a_nucleus_pct": 87},
        "why_not_current": "median + 3 SD of the band's own cells sets the bar by the band's background, so it calls "
                           "about 7x as many cells as were clicked, spread evenly through the band, and does not rank "
                           "the samples like the clicks (Spearman 0.03).",
        "why_chosen": "median + 7 SD on the median-filtered channel has the fewest false positives deep in the other "
                      "organoid (0.03 %), the largest step at the line (2.4x), and agrees best with the clicks "
                      "(F1 0.38, Spearman 0.73).",
        "caveats": ["The AF488 label is dim and patchy per cell; no rule separates cleanly (every step at the line <= 2.4).",
                    "Negative-control rules do not deliver their promised rate (99th pct gives 2.1 %).",
                    "AF488 is not bimodal over a section: Otsu calls half the cells."],
    },
    "results": {
        "source": "06_counts_in_band.ipynb, 07_counts_crossed.ipynb, 08_band_choice.ipynb, 09_manual_vs_auto.ipynb, 21 October 2026",
        "samples": 30, "gd1": 19, "wt3": 11, "slides": {"GD1": 11, "WT3": 8},
        "in_band_per_mm": {"current": 85, "calibrated": 12, "contrast": 53},
        "in_band_pct": {"current": 1.5, "calibrated": 0.3, "contrast": 1.0},
        "by_depth_pct": {"bins_um": ["0-50", "50-100", "100-200", "200-300"], "calibrated": [0.49, 0.29, 0.18, 0.07]},
        "edge_pct": {"within_20_um": 0.48, "deeper": 0.25},
        "genotype_in_band": [
            {"measure": "median + 7 SD filtered, per mm", "samples": "4.0x (p 0.001)", "slides": "2.7x (p 0.020)"},
            {"measure": "median + 7 SD filtered, % positive", "samples": "4.7x (p 0.002)", "slides": "3.4x (p 0.016)"},
            {"measure": "ring contrast, no autofl., per mm", "samples": "3.2x (p 0.003)", "slides": "2.7x (p 0.009)"},
            {"measure": "median + 3 SD (current), per mm", "samples": "1.4x (p 0.55)", "slides": "1.4x (p 0.72)"},
        ],
        "batch_per_mm": {"experiment 8": [18.8, 5.7], "experiment 9": [106, 15], "experiment 10": [21.2, 5.9],
                         "order": ["WT3", "GD1"]},
        "crossed_per_mm": {"le_50_um": 3, "le_100_um": 6, "le_300_um": 11, "any": 15, "current_le_300_um": 80},
        "crossed_lines_ccc_300_um": 0.97, "crossed_rel_diff_50_um_pct": 38,
        "crossed_depth_um": {"median": [105, 112], "p90": [330, 370]},
        "surface": {"crossed_edge_um": 157, "other_edge_um": 242, "closer_samples": "17 of 24", "p": 0.03},
        "genotype_crossed": {"calibrated": "3.5x (p 0.002 samples; 2.7x, p 0.026 slides)",
                             "contrast": "3.5x (p 0.003; 2.9x, p 0.016 slides)", "current": "1.2x (p 0.55)"},
        "orientation": {"opposite_homes": "28 of 30", "band_into_af488_away": "30 of 30"},
        "manual_genotype": {"clicks_per_mm": "1.64x (p 0.15)", "calibrated_12": "4.6x (p 0.11)",
                            "current_12": "0.60x (p 0.34)"},
        "band_depth": {"positives_first_150_um_pct": 75, "cells_first_150_um_pct": 47, "spearman_150_vs_300": 0.95,
                       "positives_past_outline": 33},
        "normalise": "per mm² of band or per mm of the hand-drawn line, not of the finder's line",
        "caveats": ["Slides, not sections, are the honest unit (n = 8 vs 11).", "30 of ~1 450 sections; WT2 not annotated.",
                    "Experiment 9 WT3 (3 samples, 2 of one slide) is much higher than the rest."],
        "headline": "WT3 sends more AF488 cells across the junction than GD1 (about 3.5-4x), not further.",
    },
}


# ---------------------------------------------------------------------------
# Schema check
# ---------------------------------------------------------------------------


def triage_samples() -> list[dict]:
    """The 30 annotated samples: finder status, whether its line is bad (> 100 µm off or none), Damaged."""
    picks = picks_classes()
    out = []
    for name, row in scores().iterrows():
        if name in jt.EXCLUDE:
            continue
        err = float(row["mean_dist"]) if pd.notna(row["mean_dist"]) else None
        out.append({"name": C.short(name), "status": str(row["status"]),
                    "error_um": round(err, 1) if err is not None else None,
                    "bad": bool(err is None or err > 100.0),
                    "damaged": bool(name in picks.index and picks.loc[name, "Shape"] == "Damaged")})
    return sorted(out, key=lambda r: r["name"])


def _is_array(v) -> bool:
    return isinstance(v, dict) and set(v) >= {"shape", "dtype", "gz"}


def _check_array(path, v, shape=None, problems=None):
    if not _is_array(v):
        problems.append(f"{path}: not an encoded array")
        return
    try:
        a = dec(v)
    except Exception as exc:  # noqa: BLE001
        problems.append(f"{path}: does not decode ({exc})")
        return
    if list(a.shape) != list(v["shape"]):
        problems.append(f"{path}: shape {a.shape} != {v['shape']}")
    if shape is not None and list(a.shape) != list(shape):
        problems.append(f"{path}: shape {list(a.shape)} != expected {list(shape)}")


def check_schema(data: dict) -> list[str]:
    problems: list[str] = []

    def need(obj, keys, where):
        for k in keys:
            if k not in obj:
                problems.append(f"{where}: missing {k}")

    need(data, ["meta", "real", "failed", "toy", "cells", "positivity", "profile", "gallery", "numbers"], "DATA")
    need(data["meta"], ["built", "source"], "meta")
    pixel_keys = ["nuclear", "label", "other", "smooth", "median_local", "prominence", "prominence_off", "density",
                  "density_off", "hot", "cold", "cost", "arrival_hot", "arrival_cold", "band", "tissue"]
    for part in ("real", "failed"):
        s = data[part]
        need(s, ["um", "shape", *pixel_keys, "hist", "tissue_steps", "sigmas", "seed_steps", "cost_powers", "cuts",
                 "lines", "truth", "rim_um", "gap_um", "width_um"], part)
        shape = s["shape"]
        for k in pixel_keys:
            if k in s:
                _check_array(f"{part}.{k}", s[k], shape, problems)
        for group in ("sigmas", "cost_powers", "cuts"):
            for k, v in s.get(group, {}).items():
                _check_array(f"{part}.{group}.{k}", v, shape, problems)
        for group in ("tissue_steps", "seed_steps"):
            for i, step in enumerate(s.get(group, [])):
                need(step, ["name", "mask"], f"{part}.{group}[{i}]")
                _check_array(f"{part}.{group}[{i}]", step["mask"], shape, problems)
        need(s.get("hist", {}), ["edges", "counts", "otsu", "cut"], f"{part}.hist")
        if sorted(s.get("sigmas", {})) != sorted(str(x) for x in SIGMAS_UM):
            problems.append(f"{part}.sigmas keys")
        if sorted(s.get("cuts", {})) != sorted(str(b) for b in BENDS) or sorted(s.get("lines", {})) != sorted(str(b) for b in BENDS):
            problems.append(f"{part}.cuts/lines keys")
    for part in ("real", "failed"):
        need(data[part], ["name", "composite", "manual_line", "second_line", "error_um"], part)
    for name in ("uniform", "slanted", "speck", "gappy"):
        if name not in data["toy"]["presets"]:
            problems.append(f"toy.presets: missing {name}")
        else:
            _check_array(f"toy.presets.{name}", data["toy"]["presets"][name], [7, 11], problems)
    c = data["cells"]
    need(c, ["name", "um_per_px", "size_px", "origin_um", "dapi", "af488", "composite", "nuclei", "grown", "ring", "table"], "cells")
    for k in ("nuclei", "grown", "ring"):
        _check_array(f"cells.{k}", c[k], c["size_px"], problems)
    for row in c["table"][:1]:
        need(row, ["id", "y", "x", "area_um2", "mean", "ring", "contrast", "filtered_mean", "in_band", "from_line_um",
                   "edge_um", "clicked"], "cells.table[0]")
    p = data["positivity"]
    need(p, ["name", "line", "band_outline", "tissue_outline", "cols", "rules"], "positivity")
    lengths = set()
    for k in ("y_um", "x_um", "raw_mean", "filtered_mean", "ring", "contrast", "in_band", "from_line_um", "edge_um", "clicked"):
        if k not in p["cols"]:
            problems.append(f"positivity.cols: missing {k}")
            continue
        _check_array(f"positivity.cols.{k}", p["cols"][k], None, problems)
        lengths.add(tuple(p["cols"][k]["shape"]))
    if len(lengths) != 1:
        problems.append(f"positivity.cols: lengths differ {lengths}")
    pr = data["profile"]
    need(pr, ["distance_um", "rules"], "profile")
    for r, v in pr["rules"].items():
        need(v, ["median", "per_sample"], f"profile.rules[{r}]")
        if len(v["median"]) != len(pr["distance_um"]):
            problems.append(f"profile.rules[{r}].median length")
    for i, g in enumerate(data["gallery"]):
        need(g, ["id", "category", "title", "classes", "status", "flags", "error_um", "why", "img", "legend"], f"gallery[{i}]")
        if g.get("category") not in ("organoid", "failure", "good_junction"):
            problems.append(f"gallery[{i}].category {g.get('category')}")
        if not str(g.get("img", "")).startswith("data:image/"):
            problems.append(f"gallery[{i}].img")
    need(data["numbers"], ["triage", "positivity", "results", "annotators"], "numbers")
    return problems


def size_report(data: dict) -> None:
    total = len(json.dumps(data, ensure_ascii=False).encode("utf-8"))
    print(f"\nJSON size {total / 1e6:.2f} MB")
    for key, value in data.items():
        size = len(json.dumps(value, ensure_ascii=False).encode("utf-8"))
        print(f"  {key:12s} {size / 1e6:7.3f} MB")
        if isinstance(value, dict) and size > 1e6:
            parts = sorted(((len(json.dumps(v, ensure_ascii=False).encode('utf-8')), k) for k, v in value.items()), reverse=True)
            print("      " + ", ".join(f"{k} {s / 1e6:.2f}" for s, k in parts[:8]))


# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fresh", action="store_true", help="redo the cached slow parts (scans, segmentation)")
    args = parser.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    private_tables()

    log("cells of the 30 annotated samples")
    cells, meta, dots = load_cells()
    log("cells crop")
    crop, fullres = cells_crop(cells, meta, dots, args.fresh)
    log(f"the section that worked, {REAL}")
    real = real_sample(REAL, fullres)
    log(f"the section that failed, {FAILED}")
    failed = real_sample(FAILED, None)
    log("positivity")
    label = C.label_channel(meta)
    pos = positivity(cells[REAL], meta.loc[REAL], label)
    log("profile")
    prof = profile(cells, meta)
    log("made-up failure modes")
    import explainer_code_check
    import explainer_failures

    synth_failures = explainer_failures.synthetic_failures(sys.modules[__name__])
    log("junction gallery")
    gallery = junction_gallery()
    log("organoid gallery")
    gallery = organoid_gallery(args.fresh) + gallery

    data = {
        "meta": {"built": dt.datetime.now().isoformat(timespec="seconds"),
                 "source": "notebooks/explainers/build_explainer.py; microscopy_viewer.junction / fusion / "
                           "junction_analysis / segmentation; notebooks/validation (junction_eval caches, 30 annotated "
                           "samples of 2025_09_26_Asssembloid); OASIS pre-screen 21 Oct 2026",
                 "real_sample": REAL, "failed_sample": FAILED, "colours": {"af488": TEAL, "af555": CORAL, "nuclear": NAVY, "line": AMBER,
                                                   "manual": INK, "second": PURPLE}},
        "real": real, "failed": failed, "toy": toy_presets(), "cells": crop,
        "positivity": pos, "profile": prof, "gallery": gallery, "numbers": NUMBERS,
        "synthetic_failures": synth_failures,
        "code": explainer_code_check.blocks(),
    }
    data["numbers"]["triage"]["samples"] = triage_samples()
    data = clean(data)
    problems = check_schema(data)
    if problems:
        print("SCHEMA PROBLEMS:\n  " + "\n  ".join(problems))
    else:
        print("schema check: every documented key present, every array decodes to its shape")
    text = json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    BUILD.mkdir(parents=True, exist_ok=True)
    (BUILD / "explainer_data.json").write_text(text, encoding="utf-8")
    size_report(data)
    if TEMPLATE.exists():
        template = TEMPLATE.read_text(encoding="utf-8")
        if PLACEHOLDER not in template:
            print(f"! {TEMPLATE.name} has no line '{PLACEHOLDER}': explainer not written")
        else:
            html = template.replace(PLACEHOLDER, "const DATA = " + text.replace("</", "<\\/") + ";", 1)
            OUTPUT.write_text(html, encoding="utf-8")
            print(f"wrote {OUTPUT} ({len(html.encode('utf-8')) / 1e6:.2f} MB)")
    else:
        print(f"no {TEMPLATE.name} yet: wrote the data only")
    log("done")


if __name__ == "__main__":
    main()
