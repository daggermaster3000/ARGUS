"""Positive cells merged into labelled bodies, and whole-cell watershed territories, on James's sections.

Two method families for ``05_james_counts.ipynb``, on the 14 sections James
counted (the ``name`` s of ``james_counts/clicks.parquet``), written under
``james_counts/methods/``:

* **bodies** — the run's merging of positive cells that are one labelled body
  (``microscopy_viewer.cell_bodies``, as ``junction_analysis.merge_section`` does
  it: foreground on the channel less its 45 µm local median, territories seeded
  by every nucleus within 20 µm, bodies parted at dips of h = 1 µm), for the
  positive cells of each rule in :data:`RULES` (median + k robust SD of the
  junction region's cells, as ``setup_cell.py``). Per section:

  - ``objects/bodies/<name>.parquet`` — one row per body: ``method`` (the
    variant), ``object`` (named after its smallest cell label), ``y``, ``x``
    (centroid of the body's pixels, µm), ``area`` (µm², the union of its cells'
    territories), ``n_cells``, and ``region``, ``d``, ``beside``, ``edge``
    (``james_clicks.place``);
  - ``hits/bodies/<name>.parquet`` — per James dot: per variant, the body of the
    cell the dot is matched to (``clicks.parquet`` ``label``); if that cell is in
    none, the nearest body pixel within :data:`HIT_UM`; 0 for none. ``<variant> by``
    says which (``cell``, ``pixel`` or empty);
  - ``cells/bodies/<name>.parquet`` — ``Label`` and, per variant, the body id of
    each cell (0 when the cell is not positive).

* **watershed** — each cell's nuclei-seeded watershed territory over the same
  foreground (``validation/watershed_cells.territories`` / ``measure``, as
  ``05_watershed_cells.ipynb``): ``cells/watershed/<name>.parquet`` with ``Label``,
  ``watershed area (µm²)``, ``watershed AF488 mean``, ``watershed AF488 mean
  (median filter)``, ``watershed ring mean``, ``watershed no ring``, ``watershed
  ring contrast``, and the calls of :data:`WS_RULES` (05's best territory rule,
  median + 7 SD (median filter), its 5 and 10 SD neighbours, and ring contrast ≥ 1.5).

Each family's ``settings.json`` holds its parameters and the time per section.
Run it with ``python notebooks/dataset/james_bodies.py`` (``1`` to time one
section first; ``--force`` to redo the sections done). Pure numpy / scipy /
scikit-image, no Qt.
"""

from __future__ import annotations

import os

#: Threads numpy / scipy may use per process.
THREADS = 2
for _variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_variable, str(THREADS))

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "validation"))
import james_clicks as jc  # noqa: E402
import watershed_cells as ws  # noqa: E402

from microscopy_viewer import cell_bodies as cb  # noqa: E402
from microscopy_viewer import junction_analysis as ja  # noqa: E402
from microscopy_viewer import loaders  # noqa: E402

METHODS = jc.TARGET / "methods"
CHANNEL = "AF488"
RAW, FILTERED = f"{CHANNEL} mean", f"{CHANNEL} mean (median filter)"
#: The rules whose positive cells are merged: variant → (cell value, k robust SDs over the region's cells).
RULES = {
    "bodies · filtered 7 SD (run)": (FILTERED, 7.0),
    "bodies · filtered 5 SD": (FILTERED, 5.0),
    "bodies · filtered 3 SD": (FILTERED, 3.0),
    "bodies · raw 7 SD": (RAW, 7.0),
    "bodies · raw 3 SD": (RAW, 3.0),
}
#: The territory rules of 05: name → (territory value, k SDs over the region's territories) or ring ratio.
WS_RULES = {
    "watershed median + 7 SD (median filter)": ("watershed AF488 mean (median filter)", 7.0),
    "watershed median + 5 SD (median filter)": ("watershed AF488 mean (median filter)", 5.0),
    "watershed median + 10 SD (median filter)": ("watershed AF488 mean (median filter)", 10.0),
    "watershed ring contrast ≥ 1.5": ("watershed ring contrast", None),
}
#: A dot whose cell is in no body still hits one whose pixels lie this close (µm).
HIT_UM = 2.0
#: Processes (another job has the GPU, another the CPU).
WORKERS = 3


def robust(values) -> tuple[float, float]:
    """Median and robust SD (1.4826 × MAD) of the finite values."""
    return ja.robust_centre_and_spread(np.asarray(values, dtype=np.float64))


def positive(cells: pd.DataFrame, value: str, k: float) -> np.ndarray:
    """Cells above the median + *k* robust SDs of the junction region's cells (as ``setup_cell.positive``)."""
    centre, sd = robust(cells.loc[cells["region"].to_numpy(bool), value])
    return (cells[value] > centre + k * sd).to_numpy()


def read_section(info: dict):
    """AF488 raw and less its local median over the whole section, and the section's outline."""
    path = Path(info["file"])
    specs = loaders.load_path(path)
    try:
        names = [str(spec.channel_name or spec.name) for spec in specs]
        spec = specs[names.index(CHANNEL)]
        voxel = float(info["voxel_yx"][-1])
        shape = tuple(int(v) for v in info["shape_yx"])
        whole = (slice(0, shape[0]), slice(0, shape[1]))
        raw = ja._plane(spec, whole).astype(np.float32)
        filtered = ja.less_local_median(raw, voxel)
        tissue = ja.full_tissue(specs[names.index(info["nuclear"])], shape)
    finally:
        loaders.release(path, own_thread_only=True)
    return raw, filtered, tissue


def body_centres(body_map: np.ndarray, ids: np.ndarray, voxel: float) -> tuple[np.ndarray, np.ndarray]:
    """Per body id: centroid of its pixels (µm, NaN without pixels) and its area (µm²)."""
    rows, cols = np.nonzero(body_map)
    owner = body_map[rows, cols]
    index = np.searchsorted(ids, owner)
    n = np.bincount(index, minlength=len(ids)).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        y = np.bincount(index, weights=rows, minlength=len(ids)) / n
        x = np.bincount(index, weights=cols, minlength=len(ids)) / n
    return np.column_stack([y, x]) * voxel, n * voxel * voxel


def pixel_hits(body_map: np.ndarray, dots_px: np.ndarray, radius_px: float) -> np.ndarray:
    """Per dot, the body whose nearest pixel lies within *radius_px*, 0 for none."""
    out = np.zeros(len(dots_px), dtype=np.int64)
    r = int(np.ceil(radius_px))
    for i, (y, x) in enumerate(np.round(dots_px).astype(int)):
        y0, x0 = max(y - r, 0), max(x - r, 0)
        patch = body_map[y0: y + r + 1, x0: x + r + 1]
        hit = np.nonzero(patch)
        if not hit[0].size:
            continue
        dist = np.hypot(hit[0] + y0 - y, hit[1] + x0 - x)
        best = int(np.argmin(dist))
        if dist[best] <= radius_px:
            out[i] = int(patch[hit[0][best], hit[1][best]])
    return out


def ws_calls(table: pd.DataFrame, region: np.ndarray) -> dict[str, np.ndarray]:
    """The territory rules of :data:`WS_RULES`, the bar from the junction region's territories."""
    calls = {}
    for name, (value, k) in WS_RULES.items():
        values = table[value].to_numpy(float)
        if k is None:
            calls[name] = values >= ja.RING_RATIO
        else:
            centre, sd = robust(values[region])
            calls[name] = values > centre + k * sd
    return calls


def run_section(name: str) -> dict:
    """Both families for one section; returns its timings and counts."""
    ws.below_normal_priority()
    timings: dict[str, float] = {}
    started = time.perf_counter()

    def lap(key: str) -> None:
        timings[key] = round(time.perf_counter() - started - sum(timings.values()), 1)

    info = jc.section_info(name)
    voxel = float(info["voxel_yx"][-1])
    cells = jc.run_cells(name)
    labels = cells["label"].to_numpy(np.int64)
    nuclei = ja.read_labels(jc.RUN / info["nuclei_file"]).astype(np.int32)
    raw, filtered, tissue = read_section(info)
    lap("read")
    fg = cb.stain_foreground(filtered, tissue, voxel)
    del tissue
    lap("foreground")

    # Watershed: every cell's territory and its figures.
    centres_um = cells[["y", "x"]].to_numpy(float)
    territory = ws.territories(nuclei, fg, centres_um / voxel, labels, voxel)
    lap("territories")
    figures = ws.measure(territory, raw, filtered, labels, centres_um, voxel)
    del raw, filtered
    wtable = pd.DataFrame({"Label": labels})
    for key, values in figures.items():
        wtable[key.replace("WS ", "watershed ")] = values
    for key, values in ws_calls(wtable, cells["region"].to_numpy(bool)).items():
        wtable[key] = values
    lap("measure")

    # Bodies: each rule's positive cells merged.
    clicks = pd.read_parquet(jc.TARGET / "clicks.parquet")
    clicks = clicks[clicks["name"] == name].reset_index(drop=True)
    dots_px = clicks[["y", "x"]].to_numpy(float) / voxel
    run_check = int((positive(cells, FILTERED, 7.0) != cells[f"{CHANNEL} positive"].to_numpy(bool)).sum())
    btable = pd.DataFrame({"Label": labels})
    hits = pd.DataFrame({"name": name, "dot": clicks["dot"].to_numpy()})
    objects, counts = [], {}
    top = int(max(territory.max(), labels.max()))
    for variant, (value, k) in RULES.items():
        calls = positive(cells, value, k)
        chosen = labels[calls]
        mapping = cb.positive_objects(nuclei, fg, chosen, voxel)
        ids = np.zeros(len(labels), dtype=np.int64)
        ids[calls] = [mapping.get(int(label), int(label)) for label in chosen]
        btable[variant] = ids
        lut = np.zeros(top + 1, dtype=np.int64)
        lut[labels] = ids
        body_map = lut[territory]
        bodies, n_cells = np.unique(ids[ids > 0], return_counts=True)
        centre_um, area = body_centres(body_map, bodies, voxel)
        # A body with no pixel (cannot happen while every nucleus seeds its territory): its cells' mean.
        missing = ~np.isfinite(centre_um[:, 0])
        if missing.any():
            frame = pd.DataFrame({"o": ids, "y": centres_um[:, 0], "x": centres_um[:, 1]})
            mean = frame[frame["o"] > 0].groupby("o")[["y", "x"]].mean()
            centre_um[missing] = mean.loc[bodies[missing]].to_numpy()
        placed = jc.place(info, cells, centre_um) if len(bodies) else pd.DataFrame(
            columns=["region", "d", "beside", "edge"])
        objects.append(pd.DataFrame({"method": variant, "object": bodies, "y": centre_um[:, 0],
                                     "x": centre_um[:, 1], "area": area, "n_cells": n_cells,
                                     **{c: placed[c].to_numpy() for c in placed}}))
        by_cell = clicks["label"].map(pd.Series(ids, index=labels)).fillna(0).astype(np.int64).to_numpy()
        by_pixel = pixel_hits(body_map, dots_px, HIT_UM / voxel)
        hits[variant] = np.where(by_cell > 0, by_cell, by_pixel)
        hits[f"{variant} by"] = np.where(by_cell > 0, "cell", np.where(by_pixel > 0, "pixel", ""))
        counts[variant] = {"positive cells": int(calls.sum()), "bodies": int(len(bodies)),
                           "dots in a body": int((hits[variant] > 0).sum()),
                           "dots by pixel only": int(((by_cell == 0) & (by_pixel > 0)).sum())}
        del body_map
    lap("bodies")

    for family, kind, frame in (("watershed", "cells", wtable), ("bodies", "cells", btable),
                                ("bodies", "objects", pd.concat(objects, ignore_index=True)),
                                ("bodies", "hits", hits)):
        folder = METHODS / kind / family
        folder.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(folder / f"{name}.parquet", index=False)
    lap("write")
    timings["total"] = round(time.perf_counter() - started, 1)
    ws_counts = {key: int(wtable[key].sum()) for key in WS_RULES}
    return {"name": name, "timings_s": timings, "n_cells": int(len(cells)), "n_dots": int(len(clicks)),
            "run rule recomputed, cells called differently": run_check,
            "foreground share of tissue": float(fg.mean()), "bodies": counts, "watershed": ws_counts}


def done(name: str) -> bool:
    return all((METHODS / kind / family / f"{name}.parquet").is_file()
               for kind, family in (("cells", "watershed"), ("cells", "bodies"), ("objects", "bodies"),
                                    ("hits", "bodies")))


def write_settings(records: list[dict]) -> None:
    """Each family's parameters and the time per section (merged with the sections done before)."""
    shared = {"sections": "unique name of james_counts/clicks.parquet", "run": str(jc.RUN),
              "channel": CHANNEL, "local median (µm)": ja.MEDIAN_UM,
              "foreground": {"smooth (µm)": cb.SMOOTH_UM, "k robust SD over the tissue": cb.FG_K,
                             "tissue": "junction_analysis.full_tissue (Otsu of the nuclear stain at 1/8)"},
              "processes": WORKERS, "threads per process": THREADS}
    families = {
        "bodies": {**shared, "engine": "microscopy_viewer.cell_bodies.positive_objects",
                   "reach (µm)": cb.REACH_UM, "h (µm)": cb.H_UM,
                   "rules": {k: {"value": v, "k robust SD over the region's cells": s} for k, (v, s) in RULES.items()},
                   "body pixels": "union of its cells' watershed territories (watershed_cells.territories)",
                   "hit": f"the matched cell's body, else the nearest body pixel within {HIT_UM} µm",
                   "object id": "smallest cell label in the body"},
        "watershed": {**shared, "engine": "notebooks/validation/watershed_cells.territories / measure",
                      "reach (µm)": ws.MAX_UM, "ring (µm)": list(ws.RING_UM), "tile overlap (µm)": ws.OVERLAP_UM,
                      "rules": {k: {"value": v, "k robust SD over the region's cells" if s else "ring ratio ≥":
                                    s if s else ja.RING_RATIO} for k, (v, s) in WS_RULES.items()},
                      "best rule in 05 (29 Oct 2026)": "watershed median + 7 SD (median filter)"},
    }
    for family, settings in families.items():
        target = METHODS / family / "settings.json"
        old = {}
        if target.is_file():
            old = json.loads(target.read_text(encoding="utf-8")).get("sections", {})
        for record in records:
            old[record["name"]] = {"timings_s": record["timings_s"], "n_cells": record["n_cells"],
                                   "counts": record[family]}
        settings["sections"] = old
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(settings, indent=1, ensure_ascii=False), encoding="utf-8")


def main(argv: list[str]) -> int:
    from concurrent.futures import ProcessPoolExecutor, as_completed

    force = "--force" in argv
    argv = [a for a in argv if a != "--force"]
    names = sorted(pd.read_parquet(jc.TARGET / "clicks.parquet", columns=["name"])["name"].unique())
    todo = [n for n in names if force or not done(n)]
    if argv:
        todo = todo[: int(argv[0])]
    print(f"{len(todo)} of {len(names)} section(s) -> {METHODS}", flush=True)
    records = []
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=min(WORKERS, max(len(todo), 1))) as pool:
        futures = {pool.submit(run_section, name): name for name in todo}
        for future in as_completed(futures):
            try:
                record = future.result()
            except Exception as exc:
                print(f"FAILED {futures[future]}: {exc!r}", flush=True)
                continue
            records.append(record)
            run = record["bodies"]["bodies · filtered 7 SD (run)"]
            print(f"{record['name']}: {record['timings_s']}; run rule {run['positive cells']} cells -> "
                  f"{run['bodies']} bodies, recomputed differs on {record['run rule recomputed, cells called differently']}",
                  flush=True)
    write_settings(records)
    print(f"done in {time.perf_counter() - started:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
