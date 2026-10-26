"""The junction pipeline over the whole dataset: junction, cells, positivity, pictures.

Run from the repository root (GPU; about two minutes a sample; resumes where it
stopped)::

    python notebooks/dataset/run_dataset.py                    # every passing sample not done
    python notebooks/dataset/run_dataset.py --limit 2          # two, to time it
    python notebooks/dataset/run_dataset.py --retry            # also redo the ones that failed
    python notebooks/dataset/run_dataset.py --report           # only the cohort flags and workbook

**Which samples.** Every row of the pre-screen table (OASIS) with ``Pass`` =
``yes``, except ``Shape`` = ``Separated`` (no junction to find). ``Damaged``
samples and those whose junction fails are run too but marked *routed*: the
validation found the finder unreliable on them, so their counts stand only
where a hand-drawn region exists.

**Per sample:**

1. **Junction** — the finder with its defaults (``junction.process_file``): the
   band is stored in the sample as ``fusion region`` (so the Junction review
   panel can open it) and its QC picture filed under ``junction_QC/ok|check|fail``.
2. **Bands** — ``first`` (the hand-drawn region, where there is one),
   ``second`` (Quillan's), ``automatic`` (the finder's). The *primary* band is
   the first annotator's where it exists, else the finder's.
3. **Cells** — Cellpose on the nuclear stain, in 2048-px tiles, only where the
   tissue lies within :data:`NEAR_UM` of a line. Each cell measured over its
   nucleus grown by 3 µm, raw and less its 45 µm local median, and over a ring
   3–6 µm out (its local background) — what the rules chosen in
   ``notebooks/validation/positivity_metrics.ipynb`` need.
4. **Where each cell is** — in which band, signed distance to each line
   (+ into the band's side), distance to the tissue edge.
5. **Positive** for AF488 and AF555 under the rules compared, and the counts in
   the band, near the line and past it (crossed), per mm of junction and per
   mm² of band.
6. **Picture** of the counts: the section, the lines, the band, the positive
   cells in the band and those that crossed (``count_QC/kept|routed``).

Out, in :data:`OUT_DIR`: ``samples.csv`` (the selection), ``summary.jsonl``
(one row per sample), ``cells/<name>.parquet`` (every measured cell),
``junction_QC/`` (pictures, workbook, contact sheets), ``count_QC/``, ``run.log``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "notebooks" / "validation"))

import count_common as C  # noqa: E402  (also sets the SSL guard)
import count_cells as cc  # noqa: E402

from microscopy_viewer import fusion as fu  # noqa: E402
from microscopy_viewer import ims_store, loaders  # noqa: E402
from microscopy_viewer import junction as jn  # noqa: E402
from microscopy_viewer import junction_analysis as ja  # noqa: E402
from microscopy_viewer import segmentation as sg  # noqa: E402

PRESCREEN = Path(r"C:\Users\qfavey\Downloads\oasis_prescreen_211026.csv")
DATA = Path(r"D:\Slidescanner extracted")
OUT_DIR = DATA / "dataset_analysis_20261021"
#: Tissue within this distance of a line is segmented; the rest is never needed.
NEAR_UM = 800.0
RING_UM = (3.0, 6.0)
#: The rules counted (name → (function of df, band, channel), short key).
RULES = {
    "calibrated": ("band median + 7 SD, median filter", lambda df, b, ch: C.rule_band_ksd(df, ch, b, 7.0, "median")),
    "contrast": ("ring contrast ≥ 1.5", lambda df, b, ch: C.rule_contrast(df, ch, b, 1.5)),
    "current": ("band median + 3 SD (current)", lambda df, b, ch: C.rule_band_ksd(df, ch, b, 3.0)),
}

log = logging.getLogger("run_dataset")


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def selection() -> pd.DataFrame:
    table = pd.read_csv(PRESCREEN, encoding="utf-8-sig")
    table["name"] = [jn.fu_name(Path(p)) for p in table["path"]]
    keep = (table["Pass"] == "yes") & (table["Shape"] != "Separated")
    out = table[keep].copy()
    out = out.join(C.design(), on="name")
    out["exists"] = [Path(p).exists() for p in out["path"]]
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Geometry on the sample's own overview
# ---------------------------------------------------------------------------


def edge_distance(overview, y_px, x_px):
    from scipy import ndimage

    filled = jn.tissue_of(overview)
    um, step = overview.um, overview.step
    inside = ndimage.distance_transform_edt(filled) * um - um / 2
    outside = ndimage.distance_transform_edt(~filled) * um - um / 2
    signed = np.where(filled, inside, -outside)
    coords = np.vstack([(y_px - (step - 1) / 2) / step, (x_px - (step - 1) / 2) / step])
    return ndimage.map_coordinates(signed, coords, order=1, mode="nearest").astype(np.float32), filled


def near_lines_mask(overview, filled, lines):
    """Overview pixels of tissue within NEAR_UM of any line."""
    from scipy import ndimage

    on = np.zeros(filled.shape, bool)
    for line in lines:
        curve = fu.fit_spline(np.asarray(line, float).reshape(-1, 2), overview.um) / overview.um
        on |= jn._draw_curve(curve, filled.shape) > 0
    if not on.any():
        return filled
    distance = ndimage.distance_transform_edt(~on) * overview.um
    return filled & (distance <= NEAR_UM)


# ---------------------------------------------------------------------------
# Cells
# ---------------------------------------------------------------------------


def measure(specs, names, nuclear, voxel, wanted_overview, step):
    """Segment the tiles that touch *wanted_overview* and measure every cell in them."""
    from scipy import ndimage

    first = specs[0].data[0] if specs[0].multiscale else specs[0].data
    height, width = (int(n) for n in first.shape[-2:])
    whole = (slice(0, height), slice(0, width))
    planes = {name: ja._plane(spec, whole) for name, spec in zip(names, specs)}
    stain = planes[names[nuclear]]
    scored = [n for k, n in enumerate(names) if k != nuclear]
    filtered = {n: cc.less_local_median(planes[n], voxel[0]) for n in scored}
    tiles = []
    for y in range(0, height, ja.TILE_PX):
        for x in range(0, width, ja.TILE_PX):
            tile = (slice(y, min(y + ja.TILE_PX, height)), slice(x, min(x + ja.TILE_PX, width)))
            box = wanted_overview[y // step: (tile[0].stop + step - 1) // step, x // step: (tile[1].stop + step - 1) // step]
            if box.size and box.mean() >= ja.TILE_MIN_TISSUE:
                tiles.append(tile)
    parts: dict[str, list] = {}

    def add(key, values):
        parts.setdefault(key, []).append(np.asarray(values, dtype=np.float32))

    for tile in tiles:
        nuclei = np.asarray(ja._segment(stain[tile], voxel, None, None))
        ids = np.unique(nuclei[nuclei > 0]).astype(np.int64)
        if not ids.size:
            continue
        centres = np.asarray(ndimage.center_of_mass(nuclei > 0, nuclei, ids)).reshape(-1, 2)
        add("y", centres[:, 0] + tile[0].start)
        add("x", centres[:, 1] + tile[1].start)
        add("area_um2", np.asarray(ndimage.sum(nuclei > 0, nuclei, ids)) * voxel[0] * voxel[1])
        grown = sg.expand_labels(nuclei, voxel, RING_UM[0])
        outer = sg.expand_labels(nuclei, voxel, RING_UM[1])
        ring = np.where(grown == 0, outer, 0)
        for n in scored:
            raw = ja.label_stats(grown, planes[n][tile], ids)
            add(C.column(n, 3.0, "raw", "mean"), raw["mean"])
            add(C.column(n, 3.0, "raw", "median"), raw["median"])
            add(C.column(n, 3.0, "median", "mean"), ja.label_stats(grown, filtered[n][tile], ids)["mean"])
            add(C.ring_column(n, "raw", "mean"), ja.label_stats(ring, planes[n][tile], ids)["mean"])
    sg.release_gpu_memory()
    if not parts:
        return pd.DataFrame(), scored, len(tiles)
    return pd.DataFrame({k: np.concatenate(v) for k, v in parts.items()}), scored, len(tiles)


# ---------------------------------------------------------------------------
# One sample
# ---------------------------------------------------------------------------


def bands_of(path, overview, result, options):
    """name → (full-resolution band, line in µm) for every band the sample has."""
    out = {}
    keys = ims_store.list_labels(path)
    for name, key in C.ANNOTATORS.items():
        if key in keys:
            labels, attrs = ims_store.load_labels(path, key)
            if "line_yx_um" in attrs:
                out[name] = (np.asarray(labels).reshape(np.shape(labels)[-2:]) > 0,
                             np.asarray(attrs["line_yx_um"], float).reshape(-1, 2))
    if result.line_um is not None and result.band is not None:
        band = jn.full_resolution_band(result, overview, options) > 0
        if band.any():
            out["automatic"] = (band, np.asarray(result.line_um, float))
    return out


def place(df, bands, voxel, shape):
    rr = np.clip(np.round(df["y"].to_numpy()).astype(int), 0, shape[0] - 1)
    cols = np.clip(np.round(df["x"].to_numpy()).astype(int), 0, shape[1] - 1)
    points = np.column_stack([df["y"].to_numpy() * voxel[0], df["x"].to_numpy() * voxel[1]])
    lengths, areas = {}, {}
    for name, (band, line) in bands.items():
        inside = band[rr, cols]
        raw, beside, length = C._side(points, line)
        flip = np.sign(np.median(np.sign(raw[inside & beside]))) if (inside & beside).any() else 1.0
        df[f"in {name}"] = inside
        df[f"from {name} (um)"] = (raw * (flip or 1.0)).astype(np.float32)
        df[f"beside {name}"] = beside
        lengths[name] = float(length)
        areas[name] = float(band.sum()) * voxel[0] * voxel[1] / 1e6
    return lengths, areas


def count(df, band, channels, length_um, area_mm2):
    out = {}
    for ch in channels:
        home_band = C.home_is_band_side(df, ch, band)
        out[f"{ch} home on band side"] = bool(home_band)
        regions = {
            "band": C.region(df, "band", band, ch),
            "crossed <=300": C.region(df, "crossed", band, ch, 300.0),
            "crossed any": C.region(df, "crossed", band, ch),
            "near+crossed": C.region(df, "near + crossed", band, ch),
        }
        out["cells in band"] = int(regions["band"].sum())
        for key, (_label, rule) in RULES.items():
            positive = rule(df, band, ch)
            df[f"{ch} positive {key}"] = positive
            for region, mask in regions.items():
                n = int((positive & mask).sum())
                out[f"{ch} {key} {region}"] = n
                out[f"{ch} {key} {region} per mm"] = n / (length_um / 1000) if length_um else np.nan
            if regions["band"].any():
                out[f"{ch} {key} band %"] = 100 * (positive & regions["band"]).sum() / regions["band"].sum()
                out[f"{ch} {key} band per mm2"] = (positive & regions["band"]).sum() / area_mm2 if area_mm2 else np.nan
    return out


def picture(target, overview, bands, primary, df, label, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    um = overview.um
    rgb = np.zeros((*overview.channels.shape[1:], 3), np.float32)
    order = [overview.label, overview.nuclear, overview.other]
    for k, index in enumerate(order):
        if index is None:
            continue
        plane = overview.channels[index].astype(np.float32)
        valid = plane[plane > 0]
        low, top = (np.percentile(valid, [5, 99.5]) if valid.size else (0.0, 1.0))
        rgb[..., [1, 2, 0][k]] = np.clip((plane - low) / max(top - low, 1e-6), 0, 1)
    fig, ax = plt.subplots(figsize=(8, 8 * rgb.shape[0] / max(rgb.shape[1], 1)))
    extent = (0, rgb.shape[1] * um, rgb.shape[0] * um, 0)
    ax.imshow(rgb, extent=extent)
    colours = {"first": "white", "second": "orange", "automatic": "magenta"}
    from skimage.measure import find_contours

    band_small = bands[primary][0][:: overview.step, :: overview.step]
    for contour in find_contours(np.pad(band_small, 1).astype(float), 0.5):
        ax.plot((contour[:, 1] - 1) * um, (contour[:, 0] - 1) * um, color=colours.get(primary, "white"),
                lw=0.8, ls="--")
    for name, (_band, line) in bands.items():
        curve = fu.fit_spline(line, um)
        ax.plot(curve[:, 1], curve[:, 0], color=colours.get(name, "white"), lw=1.5 if name == primary else 0.8,
                label=f"{name} line" + (" (primary)" if name == primary else ""))
    if not df.empty and f"{label} positive calibrated" in df:
        pos = df[f"{label} positive calibrated"].to_numpy()
        in_band = pos & df[f"in {primary}"].to_numpy()
        crossed = pos & C.region(df, "crossed", primary, label) & ~df[f"in {primary}"].to_numpy()
        ax.scatter(df.loc[in_band, "x (um)"], df.loc[in_band, "y (um)"], s=6, c="yellow", label=f"{label}+ in band")
        ax.scatter(df.loc[crossed, "x (um)"], df.loc[crossed, "y (um)"], s=6, c="cyan", label=f"{label}+ crossed, outside band")
    ax.set_title(title, fontsize=8)
    ax.legend(fontsize=6, loc="lower right")
    ax.set_axis_off()
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=110, bbox_inches="tight")
    plt.close(fig)


def run_one(row) -> dict:
    path = Path(row.path)
    name = row.name
    started = time.perf_counter()
    out = {"name": name, "path": str(path), "Genotype": row.Genotype, "Experiment": row.Experiment,
           "slide": row.slide, "Shape": row.Shape, "Ripped": row.Ripped, "Folded": row.Folded,
           "Quality": row.Quality, "error": ""}
    options = jn.JunctionOptions(save=True, qc_dir=OUT_DIR / "junction_QC")
    specs = loaders.load_path(path)
    try:
        overview, result = jn.analyse_specs(specs, options)
        out.update(status=result.status, flags="; ".join(result.flags), side=result.side,
                   **{f"junction {k}": float(v) for k, v in result.metrics.items()})
        jn.file_picture(options.qc_dir, name, result.status, jn.qc_picture(overview, result, name))
        if result.line_um is not None and result.band is not None:
            jn.store(path, overview, result, options)
        out["routed"] = bool(result.status == jn.STATUS_FAIL or row.Shape == "Damaged")
        bands = bands_of(path, overview, result, options)
        out["bands"] = ",".join(bands)
        primary = "first" if "first" in bands else ("automatic" if "automatic" in bands else "")
        out["primary band"] = primary
        out["counted"] = bool(primary) and (primary != "automatic" or not out["routed"])
        if not bands:
            out["error"] = "no junction: none found and none drawn"
            return out
        names = [str(s.channel_name or s.name) for s in specs]
        nuclear = jn.pick_channel(names, "", jn.NUCLEAR_HINTS)
        voxel = tuple(float(v) for v in specs[0].scale)[-2:]
        first = specs[0].data[0] if specs[0].multiscale else specs[0].data
        shape = tuple(int(n) for n in first.shape[-2:])
        y_dummy = np.zeros(0)
        _edge0, filled = edge_distance(overview, y_dummy, y_dummy)
        wanted = near_lines_mask(overview, filled, [line for _b, line in bands.values()])
        t = time.perf_counter()
        df, scored, n_tiles = measure(specs, names, nuclear, voxel, wanted, overview.step)
        out["segment s"], out["tiles"] = time.perf_counter() - t, n_tiles
        if df.empty:
            out["error"] = "no cells found"
            return out
        df["y (um)"], df["x (um)"] = df["y"] * voxel[0], df["x"] * voxel[1]
        df["edge (um)"], _ = edge_distance(overview, df["y"].to_numpy(), df["x"].to_numpy())
        lengths, areas = place(df, bands, voxel, shape)
        out.update({f"length {b} (um)": v for b, v in lengths.items()})
        out.update({f"band area {b} (mm2)": v for b, v in areas.items()})
        out["cells measured"] = len(df)
        label = scored[jn.pick_channel(scored, "", jn.LABEL_HINTS)] if jn.pick_channel(scored, "", jn.LABEL_HINTS) is not None else scored[0]
        out["label channel"] = label
        out.update(count(df, primary, scored, lengths[primary], areas[primary]))
        # Every band too, for the calibrated rule on the label: how much the choice matters.
        for b in bands:
            if b != primary:
                pos = C.rule_band_ksd(df, label, b, 7.0, "median")
                out[f"{label} calibrated band [{b}]"] = int((pos & df[f"in {b}"].to_numpy()).sum())
                out[f"{label} calibrated crossed <=300 [{b}] per mm"] = int(
                    (pos & C.region(df, "crossed", b, label, 300.0)).sum()) / (lengths[b] / 1000)
        (OUT_DIR / "cells").mkdir(parents=True, exist_ok=True)
        keep = df.drop(columns=[c for c in df.columns if c in ("y", "x")])
        keep.astype({c: np.float32 for c in keep.columns if keep[c].dtype == np.float64}).to_parquet(
            OUT_DIR / "cells" / f"{name}.parquet", index=False)
        title = (f"{name} — {row.Genotype} e{row.Experiment} — junction {result.status}"
                 f"{' (routed)' if out['routed'] else ''} — {label}+ calibrated: "
                 f"band {out.get(f'{label} calibrated band', 0)}, crossed ≤300 µm {out.get(f'{label} calibrated crossed <=300', 0)}")
        picture(OUT_DIR / "count_QC" / ("routed" if out["routed"] else "kept") / f"{name}.png",
                overview, bands, primary, df, label, title)
    finally:
        loaders.release(path)
        out["seconds"] = time.perf_counter() - started
    return out


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


def done_names(retry: bool) -> set[str]:
    summary = OUT_DIR / "summary.jsonl"
    if not summary.exists():
        return set()
    rows = [json.loads(l) for l in summary.read_text(encoding="utf-8").splitlines() if l.strip()]
    return {r["name"] for r in rows if not (retry and r.get("error", "").startswith("crashed"))}


def append(row: dict) -> None:
    with (OUT_DIR / "summary.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, default=lambda v: v.item() if hasattr(v, "item") else str(v)) + "\n")


def report() -> None:
    """The cohort flags, the junction workbook and contact sheets, from the summary."""
    rows = [json.loads(l) for l in (OUT_DIR / "summary.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    latest = {r["name"]: r for r in rows}
    outcomes = []
    for r in latest.values():
        o = jn.JunctionOutcome(path=Path(r["path"]), name=r["name"], status=r.get("status", jn.STATUS_FAIL),
                               flags=[f for f in str(r.get("flags", "")).split("; ") if f],
                               metrics={k[len("junction "):]: v for k, v in r.items() if k.startswith("junction ")},
                               error=r.get("error", ""), elapsed_s=float(r.get("seconds", 0)))
        status_dir = OUT_DIR / "junction_QC" / o.status
        o.picture = status_dir / f"{o.name}.png" if (status_dir / f"{o.name}.png").exists() else None
        outcomes.append(o)
    jn.flag_cohort(outcomes)
    folder = OUT_DIR / "junction_QC"
    jn._refile(outcomes, folder)
    jn.qc_dataframe(outcomes).to_excel(folder / "junction_QC.xlsx", index=False)
    jn.contact_sheets([o.picture for o in outcomes if o.picture is not None], folder)
    pd.DataFrame(list(latest.values())).to_csv(OUT_DIR / "summary.csv", index=False)
    log.info("report: %d samples; workbook and contact sheets in %s", len(outcomes), folder)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--retry", action="store_true")
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(OUT_DIR / "run.log", encoding="utf-8"), logging.StreamHandler()])
    for noisy in ("cellpose", "microscopy_viewer", "count_common"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    if args.report:
        report()
        return
    table = selection()
    table.to_csv(OUT_DIR / "samples.csv", index=False)
    done = done_names(args.retry)
    todo = table[table["exists"] & ~table["name"].isin(done)]
    if args.limit:
        todo = todo.head(args.limit)
    log.info("%d selected (Pass = yes, not Separated); %d missing on disk; %d done; %d to run",
             len(table), int((~table["exists"]).sum()), len(done), len(todo))
    for k, row in enumerate(todo.itertuples(), start=1):
        try:
            out = run_one(row)
        except Exception as exc:
            out = {"name": row.name, "path": row.path, "error": f"crashed: {exc}",
                   "traceback": traceback.format_exc()[-2000:]}
            if sg.is_gpu_fault(exc):
                append(out)
                log.error("GPU fault on %s: stopping; start again to resume", row.name)
                break
        append(out)
        log.info("%d/%d %s: %s %s cells, %.0f s %s", k, len(todo), row.name, out.get("status", "?"),
                 out.get("cells measured", 0), out.get("seconds", 0), out.get("error", ""))
    report()


if __name__ == "__main__":
    main()
