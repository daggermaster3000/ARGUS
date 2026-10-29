"""Data and pictures for the junction browser: every hand-drawn section, its errors, its lines.

For each section in ``junction_eval/annotated_all.csv``:

- two pictures of the overview cropped to the tissue (nuclear stain; the two labels
  over a dim nuclear stain), JPEG at the overview's resolution;
- the hand-drawn line, today's line and seeds, and the line of the per-class
  settings of ``02_junction_seeding.ipynb`` section 7, in picture pixels;
- the errors of several settings (from the sweeps' JSONL, and from the two runs here);
- the finder's figures, and **polarity** figures: inside each organoid (as split by
  the hand-drawn line) whether its own label sits near the junction or at the far
  pole, at the rim, and how much of it lies across the line.

Writes ``data.json`` and ``img/`` into *out* (default ``junction_eval/browser``), beside a copy of
``junction_browser.html``; open it through a web server or publish the folder (the page fetches its data).

    python notebooks/validation/junction_browser.py <out folder>
"""
from __future__ import annotations

import json
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

import junction_tuning as jt  # noqa: E402
from microscopy_viewer import junction as jn  # noqa: E402

OUT = jt.OUT
MARGIN_UM = 150.0
#: The settings each class chose in 02_junction_seeding.ipynb section 7 (seeds' lead 0.35).
PER_CLASS = {"seed_rule": "dominant", "label_change": 1.0, "tear_price": 1.0, "bend_um": 1000.0, "edge_weight": 0.0,
             "no_seam_seed_rule": "dominant", "no_seam_label_change": 1.0, "no_seam_tear_price": 1.0,
             "no_seam_bend_um": 400.0, "no_seam_edge": 0.0, "no_seam_lead": 0.35}
PER_CLASS_LEAD = 0.35


def _stretch(a, mask, low=1.0, high=99.7):
    lo, hi = np.percentile(a[mask], (low, high)) if mask.any() else (a.min(), a.max())
    return np.clip((a - lo) / max(hi - lo, 1e-9), 0, 1)


def _side(shape, line_px, extend_px=80.0):
    """+1 / -1 for every pixel: which side of *line_px* (rows, cols) it lies on, the line run on past its ends."""
    from scipy.spatial import cKDTree

    line = np.asarray(line_px, float)
    d0 = line[0] - line[min(3, len(line) - 1)]
    d1 = line[-1] - line[max(0, len(line) - 4)]
    d0 /= np.linalg.norm(d0) or 1.0
    d1 /= np.linalg.norm(d1) or 1.0
    ext = np.vstack([line[0] + d0 * extend_px, line, line[-1] + d1 * extend_px])
    dense = [ext[0]]
    for a, b in zip(ext[:-1], ext[1:]):
        n = max(int(np.ceil(np.linalg.norm(b - a))), 1)
        dense.extend(a + (b - a) * t for t in np.linspace(0, 1, n + 1)[1:])
    dense = np.asarray(dense)
    tangent = np.gradient(dense, axis=0)
    rows, cols = np.indices(shape)
    pts = np.column_stack([rows.ravel(), cols.ravel()]).astype(float)
    dist, k = cKDTree(dense).query(pts)
    rel = pts - dense[k]
    cross = tangent[k, 0] * rel[:, 1] - tangent[k, 1] * rel[:, 0]
    return np.where(cross >= 0, 1, -1).reshape(shape), dist.reshape(shape)


def polarity(ov, tissue, filled, manual_line_um, manual_band):
    """Where each organoid's own label sits, inside the organoid the hand-drawn line gives it."""
    from scipy import ndimage, stats

    um = ov.um
    side, dist = _side(filled.shape, np.asarray(manual_line_um) / um)
    band_side = np.sign(np.sum(side[manual_band & filled])) or 1
    other_half = filled & (side == band_side)
    labelled_half = filled & (side != band_side)
    out = {}
    dens = {"AF488": jn.cell_density(ov.channels[ov.label], tissue, filled, um)}
    if ov.other is not None:
        dens["AF555"] = jn.cell_density(ov.channels[ov.other], tissue, filled, um)
    rim = filled & ~ndimage.binary_erosion(filled, iterations=max(1, int(round(100 / um))))
    for name, d in dens.items():
        home, away = (labelled_half, other_half) if name == "AF488" else (other_half, labelled_half)
        if home.sum() < 50:
            continue
        rho = stats.spearmanr(d[home], dist[home])[0]
        out[f"{name}: toward the far pole (Spearman with distance from the line)"] = float(rho)
        inner = home & ~rim
        out[f"{name}: rim ÷ interior"] = float(d[home & rim].mean() / max(d[inner].mean(), 1e-9)) if inner.any() else np.nan
        out[f"{name}: share across the line"] = float(d[away].sum() / max(d[filled].sum(), 1e-9))
    out["Labelled half: share of tissue"] = float(labelled_half.sum() / max(filled.sum(), 1))
    return out


def one(name):
    from scipy import ndimage
    from skimage import measure

    ov, manual_band, manual_line, base = jt.load_case(name)
    um = ov.um
    ch = ov.channels
    valid = (ch > 0).all(axis=0)
    tissue = jn.tissue_mask(ch[ov.nuclear], valid, um, jn._others(ch, ov.nuclear, True))
    filled = ndimage.binary_fill_holes(tissue)
    rows, cols = np.nonzero(filled)
    m = int(MARGIN_UM / um)
    r0, r1 = max(rows.min() - m, 0), min(rows.max() + m + 1, filled.shape[0])
    c0, c1 = max(cols.min() - m, 0), min(cols.max() + m + 1, filled.shape[1])
    crop = (slice(r0, r1), slice(c0, c1))

    def px(line_um):
        if line_um is None:
            return None
        a = np.asarray(line_um, float) / um - [r0, c0]
        return [[round(float(y), 1), round(float(x), 1)] for y, x in a[:: max(1, len(a) // 400)]]

    # today's settings: the line, its seeds, and the figures
    today = jn.find_junction(ov, jn.JunctionOptions(save=False, width_um=base["width_um"], gap_um=base["gap_um"]))
    row_today = jt.run_case((name, jt.defaults()))
    saved = jn.SEED_LEAD
    jn.SEED_LEAD = PER_CLASS_LEAD
    try:
        row_class = jt.run_case((name, {**jt.defaults(), **PER_CLASS}))
    finally:
        jn.SEED_LEAD = saved
    seeds = []
    if today.seeds is not None:
        for mask in today.seeds[:2]:
            cs = measure.find_contours(mask[crop].astype(float), 0.5)
            seeds.append([[[round(float(y), 1), round(float(x), 1)] for y, x in c[::2]] for c in cs if len(c) > 8])
    return {
        "name": name,
        "crop": [int(r0), int(c0), int(r1 - r0), int(c1 - c0)],
        "um": float(um),
        "lines": {"hand": px(manual_line), "today": px(row_today.get("line")), "per class": px(row_class.get("line"))},
        "seeds": seeds,
        "today_capped": jt.capped(row_today), "class_capped": jt.capped(row_class),
        "status": today.status, "flags": "; ".join(today.flags), "fallback": float(today.metrics.get("No-seam fallback", 0.0)),
        "metrics": {k: float(today.metrics[k]) for k in ("Seam price", "Density contrast", "Labelled share of tissue",
                                                        "Solidity", "Hole share", "Tissue area (mm2)") if k in today.metrics},
        "polarity": polarity(ov, tissue, filled, manual_line, manual_band),
        "_pictures": (ov.nuclear_mean[crop] if ov.nuclear_mean is not None and np.size(ov.nuclear_mean) else ch[ov.nuclear][crop],
                      ch[ov.label][crop], ch[ov.other][crop] if ov.other is not None else None, filled[crop]),
    }


def pictures(entry, folder):
    from PIL import Image

    nuc, af488, af555, mask = entry.pop("_pictures")
    g = _stretch(nuc.astype(float), mask, 0.5, 99.8)
    Image.fromarray((g * 255).astype(np.uint8)).save(folder / f"{entry['id']}_n.jpg", quality=88)
    a = _stretch(af488.astype(float), mask, 2, 99.7)
    b = _stretch(af555.astype(float), mask, 2, 99.7) if af555 is not None else np.zeros_like(a)
    dim = 0.35 * g
    rgb = np.stack([np.clip(dim + b, 0, 1), np.clip(dim + a, 0, 1), np.clip(dim + b, 0, 1)], -1)
    Image.fromarray((rgb * 255).astype(np.uint8)).save(folder / f"{entry['id']}_l.jpg", quality=86)


def errors_from_sweeps(names):
    """Capped error per section for the settings worth comparing, from the sweeps already run."""
    out = {n: {} for n in names}
    for file, variant, label in (("seeding.jsonl", "seeds: dominant (lead 0.35)", "dominant seeds everywhere"),
                                 ("seeding.jsonl", "seeds: all labelled patches", "all AF488 patches as seed"),
                                 ("labels.jsonl", "change 1 + tears priced", "label change + tears priced"),
                                 ("seeding.jsonl", "default", "before 26 Oct (no fallback)")):
        r = pd.read_json(OUT / file, lines=True)
        s = r[r["variant"] == variant].set_index("name")["capped"]
        for n in names:
            if n in s.index:
                out[n][label] = float(s[n])
    return out


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT / "browser"
    (out / "img").mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(OUT / "annotated_all.csv")
    diag = pd.read_csv(OUT / "seeding_diagnostics.csv").set_index("name")
    names = list(table["name"])
    with ProcessPoolExecutor(12) as pool:
        entries = list(pool.map(one, names))
    sweeps = errors_from_sweeps(names)
    meta = table.set_index("name")
    for i, e in enumerate(entries):
        e["id"] = f"s{i:03d}"
        pictures(e, out / "img")
        n = e["name"]
        e["short"] = n.replace("2025_09_26_Asssembloid_", "")
        e["errors"] = {"today": e.pop("today_capped"), "per-class settings": e.pop("class_capped"), **sweeps[n]}
        for c in ("set", "batch", "Ripped", "Folded", "Quality", "Shape"):
            e[c] = None if pd.isna(meta.loc[n, c]) else str(meta.loc[n, c])
        d = diag.loc[n] if n in diag.index else None
        if d is not None:
            for c in ("AF488 patches", "seed distance (um)", "labelled seed in the hand-drawn band"):
                if c in d and pd.notna(d[c]):
                    e["metrics"][c] = float(d[c])
            e["first cut seam price"] = float(d["Seam price"]) if pd.notna(d["Seam price"]) else None
    clean = json.loads(json.dumps(entries, default=float).replace("NaN", "null"))
    (out / "data.json").write_text(json.dumps({"sections": clean}, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    shutil.copy(HERE / "junction_browser.html", out / "junction_browser.html")
    print("written", out, len(entries), "sections", flush=True)


if __name__ == "__main__":
    main()
