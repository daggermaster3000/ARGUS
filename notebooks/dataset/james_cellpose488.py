"""Cellpose on the AF488 channel itself, for James's 14 sections: the labelled cells directly.

The alternative to the run's nuclei + a positivity rule: Cellpose (``cpsam_v2``,
the run's model and thresholds, diameter left to the model) segments the AF488
plane — the brightest plane over z, the whole section — in the run's tiles
(:data:`junction_analysis.TILE_PX` with :data:`junction_analysis.TILE_OVERLAP_PX`
of overlap, each object kept in the tile holding its centre; tiles of bare slide
skipped). Two variants (:data:`VARIANTS`):

* ``Cellpose AF488 · raw`` — the AF488 plane as read;
* ``Cellpose AF488 · less local median`` — the plane less its 45 µm local median
  (``junction_analysis.less_local_median``, the run's filter), clipped at 0.

Cellpose outlines anything cell-shaped, so dim, unlabelled cells come out too:
each object's AF488 mean (raw and less the local median) is kept, so that a
threshold can be put on them afterwards (``05_james_counts.ipynb``). Objects whose
centre is outside the section (``classical_vs_cellpose.tissue_mask``) are dropped.

Into ``james_counts/methods/`` (the layout of the brief; *family* ``cellpose488``):

* ``objects/cellpose488/<name>.parquet`` — one row per object: ``method``,
  ``object``, ``y``, ``x`` (µm), ``area`` (µm²), ``AF488 mean``,
  ``AF488 mean (median filter)``, and ``region``, ``d``, ``beside``, ``edge``
  placed like the run's cells (``james_clicks.place``);
* ``hits/cellpose488/<name>.parquet`` — one row per James dot: per variant, the
  object it falls in (nearest object pixel within :data:`HIT_UM`), 0 for none;
* ``cells/cellpose488/<name>.parquet`` — one row per run cell (``Label``): the share
  of its grown region each variant's objects cover (``cover <variant>``);
* ``cellpose488/labels/<name> <variant>.tif`` — the label maps (compressed), for pictures;
* ``cellpose488/settings.json`` — the settings and the time per section.

Run from the repository root with the Python that has cellpose and a CUDA torch
(one GPU process; ~a few minutes a section)::

    python notebooks/dataset/james_cellpose488.py        # every section not done yet
    python notebooks/dataset/james_cellpose488.py 1      # just one, to time it

No Qt.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "validation"))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
import classical_vs_cellpose as cv  # noqa: E402
import james_clicks as jc  # noqa: E402

from microscopy_viewer import junction_analysis as ja  # noqa: E402
from microscopy_viewer import segmentation as sg  # noqa: E402

FAMILY = "cellpose488"
METHODS = jc.TARGET / "methods"
#: The two inputs Cellpose is given, by variant name.
RAW = "Cellpose AF488 · raw"
LESS_MEDIAN = "Cellpose AF488 · less local median"
VARIANTS = (RAW, LESS_MEDIAN)
#: A dot hits an object when the nearest object pixel is this close (µm).
HIT_UM = 2.0
#: The run's segmentation (``cells/<name>.json`` → ``segmentation``), less the growing:
#: ``cpsam_v2``, diameter left to the model, flow 0.4, cell probability 0, min size 15 px.
SETTINGS = sg.SegmentationSettings(model="cpsam_v2", mode=sg.MODE_STITCH, diameter_um=0.0,
                                   flow_threshold=0.4, cellprob_threshold=0.0, min_size=sg.DEFAULT_MIN_SIZE,
                                   normalize=True, use_gpu=True, expand_um=0.0)
#: Version 2: each tile handed over as a copy (version 1 let Cellpose normalise the plane in place).
VERSION = 2


def tiles_of(tissue: np.ndarray) -> list[tuple[slice, slice]]:
    """The run's tiles (:data:`ja.TILE_PX`) that hold at least :data:`ja.TILE_MIN_TISSUE` of tissue."""
    height, width = tissue.shape
    tiles = [(slice(y, min(y + ja.TILE_PX, height)), slice(x, min(x + ja.TILE_PX, width)))
             for y in range(0, height, ja.TILE_PX) for x in range(0, width, ja.TILE_PX)]
    return [t for t in tiles if float(tissue[t][::8, ::8].mean()) >= ja.TILE_MIN_TISSUE]


def segment_tiled(image: np.ndarray, tiles: list[tuple[slice, slice]], voxel: float,
                  settings: sg.SegmentationSettings | None = None) -> np.ndarray:
    """Cellpose over *tiles* of *image*, pasted into one label map of the whole plane.

    Each tile is segmented with :data:`ja.TILE_OVERLAP_PX` of its neighbours and keeps
    the objects whose centre is in the tile itself; an object is pasted only where
    no earlier object lies, so labels never overlap. *settings* default to
    :data:`SETTINGS` (``cellpose488_retrain.py`` hands over its trained models).
    """
    settings = settings or SETTINGS
    height, width = image.shape
    out = np.zeros((height, width), dtype=np.int32)
    found = 0
    pad = ja.TILE_OVERLAP_PX
    for tile in tiles:
        window = (slice(max(tile[0].start - pad, 0), min(tile[0].stop + pad, height)),
                  slice(max(tile[1].start - pad, 0), min(tile[1].stop + pad, width)))
        # A copy: Cellpose normalises its input in place, and the tiles overlap.
        masks = sg.segment_volume(np.array(image[window]), (voxel, voxel), settings=settings).masks
        ids = np.unique(masks[masks > 0])
        if not ids.size:
            continue
        centres = np.asarray(ndimage.center_of_mass(masks > 0, masks, ids)).reshape(-1, 2)
        cy, cx = centres[:, 0] + window[0].start, centres[:, 1] + window[1].start
        own = ids[(cy >= tile[0].start) & (cy < tile[0].stop) & (cx >= tile[1].start) & (cx < tile[1].stop)]
        lookup = np.zeros(int(masks.max()) + 1, dtype=np.int32)
        lookup[own] = np.arange(found + 1, found + 1 + own.size, dtype=np.int32)
        found += int(own.size)
        mapped = lookup[masks]
        target = out[window]
        free = (target == 0) & (mapped > 0)
        target[free] = mapped[free]
    return out


def keep_in_tissue(labels: np.ndarray, tissue: np.ndarray) -> tuple[np.ndarray, int, int]:
    """*labels* without the objects whose centre is outside *tissue*, renumbered; and the counts before / after."""
    ids = np.unique(labels[labels > 0])
    if not ids.size:
        return labels, 0, 0
    centres = np.asarray(ndimage.center_of_mass(labels > 0, labels, ids)).reshape(-1, 2)
    rows = np.clip(np.round(centres[:, 0]).astype(int), 0, tissue.shape[0] - 1)
    cols = np.clip(np.round(centres[:, 1]).astype(int), 0, tissue.shape[1] - 1)
    inside = ids[tissue[rows, cols]]
    lookup = np.zeros(int(labels.max()) + 1, dtype=np.int32)
    lookup[inside] = np.arange(1, inside.size + 1, dtype=np.int32)
    return lookup[labels], int(ids.size), int(inside.size)


def object_table(labels: np.ndarray, af488: np.ndarray, filtered: np.ndarray, voxel: float, info: dict,
                 cells: pd.DataFrame, variant: str) -> pd.DataFrame:
    """Every object of *labels*: centroid and area in µm, its AF488 means, and its place."""
    ids = np.arange(1, int(labels.max()) + 1)
    if not ids.size:
        return pd.DataFrame()
    centres = np.asarray(ndimage.center_of_mass(labels > 0, labels, ids)).reshape(-1, 2) * voxel
    table = pd.DataFrame({
        "method": variant, "object": ids, "y": centres[:, 0], "x": centres[:, 1],
        "area": np.asarray(ndimage.sum(labels > 0, labels, ids), dtype=float) * voxel ** 2,
        "AF488 mean": np.asarray(ndimage.mean(af488, labels, ids), dtype=float),
        "AF488 mean (median filter)": np.asarray(ndimage.mean(filtered, labels, ids), dtype=float),
    })
    placed = jc.place(info, cells, centres)
    for column in placed:
        table[column] = placed[column].to_numpy()
    return table


def dot_hits(labels: np.ndarray, dots_um: np.ndarray, voxel: float) -> np.ndarray:
    """Per dot, the object under it, or the nearest object pixel within :data:`HIT_UM`; 0 for none."""
    reach = int(np.ceil(HIT_UM / voxel))
    height, width = labels.shape
    hits = np.zeros(len(dots_um), dtype=np.int64)
    for k, (y, x) in enumerate(np.asarray(dots_um, float) / voxel):
        r, c = int(round(y)), int(round(x))
        if not (0 <= r < height and 0 <= c < width):
            continue
        if labels[r, c]:
            hits[k] = labels[r, c]
            continue
        top, left = max(r - reach, 0), max(c - reach, 0)
        patch = labels[top: r + reach + 1, left: c + reach + 1]
        rr, cc = np.nonzero(patch)
        if not rr.size:
            continue
        dist = np.hypot(rr + top - y, cc + left - x) * voxel
        best = int(np.argmin(dist))
        if dist[best] <= HIT_UM:
            hits[k] = patch[rr[best], cc[best]]
    return hits


def run_one(name: str) -> dict:
    """Both variants of one section, and their tables."""
    import tifffile

    started = time.perf_counter()
    info = jc.section_info(name)
    af488, nuclear, voxel = cv.read_planes(Path(info["file"]))
    tissue = cv.tissue_mask(nuclear)
    del nuclear
    filtered = ja.less_local_median(af488, voxel)
    tiles = tiles_of(tissue)
    read_s = time.perf_counter() - started

    cells = jc.run_cells(name)
    clicks = pd.read_parquet(jc.TARGET / "clicks.parquet")
    clicks = clicks[clicks["name"] == name]
    grown = ja.read_labels(jc.RUN / info["grown_file"])
    labels_of_run = np.unique(grown[grown > 0]).astype(np.int64)
    size = np.asarray(ndimage.sum(grown > 0, grown, labels_of_run), dtype=float)

    objects, cover = [], pd.DataFrame({"Label": labels_of_run})
    hits = pd.DataFrame({"name": name, "dot": clicks["dot"].to_numpy()})
    timing = {"read_s": round(read_s, 1), "tiles": len(tiles)}
    folder = METHODS / FAMILY / "labels"
    folder.mkdir(parents=True, exist_ok=True)
    for variant in VARIANTS:
        begun = time.perf_counter()
        image = af488 if variant == RAW else np.clip(filtered, 0, None)
        labels = segment_tiled(image, tiles, voxel)
        del image
        segment_s = time.perf_counter() - begun
        labels, before, after = keep_in_tissue(labels, tissue)
        objects.append(object_table(labels, af488, filtered, voxel, info, cells, variant))
        hits[variant] = dot_hits(labels, clicks[["y", "x"]].to_numpy(float), voxel)
        cover[f"cover {variant}"] = (np.asarray(ndimage.sum(labels > 0, grown, labels_of_run))
                                     / np.where(size > 0, size, np.nan))
        slug = "raw" if variant == RAW else "less local median"
        tifffile.imwrite(folder / f"{name} {slug}.tif", labels, compression="zlib")
        timing[variant] = {"segment_s": round(segment_s, 1), "objects_found": before, "objects_in_tissue": after}
        del labels
        sg.release_gpu_memory()

    for kind, table in (("objects", pd.concat(objects, ignore_index=True)), ("hits", hits), ("cells", cover)):
        target = METHODS / kind / FAMILY
        target.mkdir(parents=True, exist_ok=True)
        table.to_parquet(target / f"{name}.parquet", index=False)
    timing["seconds"] = round(time.perf_counter() - started, 1)
    return timing


def settings_file() -> Path:
    return METHODS / FAMILY / "settings.json"


def read_settings() -> dict:
    try:
        return json.loads(settings_file().read_text(encoding="utf-8"))
    except Exception:
        return {}


def main(argv: list[str]) -> int:
    names = sorted(pd.read_parquet(jc.TARGET / "clicks.parquet", columns=["name"])["name"].unique())
    record = read_settings()
    sections = record.get("sections", {}) if record.get("version") == VERSION else {}
    todo = [n for n in names if n not in sections]
    if argv:
        todo = todo[: int(argv[0])]
    print(f"{len(names)} sections, {len(todo)} to do -> {METHODS}", flush=True)
    settings_file().parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    for index, name in enumerate(todo, start=1):
        try:
            sections[name] = run_one(name)
            line = f"{name}: {sections[name]['seconds']:.0f} s, " + ", ".join(
                f"{v.split('· ')[-1]} {sections[name][v]['objects_in_tissue']} objects" for v in VARIANTS)
        except Exception as exc:
            line = f"{name}: FAILED {exc!r}"
        record = {"version": VERSION, "family": FAMILY, "variants": list(VARIANTS),
                  "segmentation": asdict(SETTINGS), "tile_px": ja.TILE_PX, "tile_overlap_px": ja.TILE_OVERLAP_PX,
                  "tile_min_tissue": ja.TILE_MIN_TISSUE, "median_um": ja.MEDIAN_UM, "hit_um": HIT_UM,
                  "plane": "brightest over z (junction_analysis._plane), whole section",
                  "tissue": "classical_vs_cellpose.tissue_mask (nuclear Otsu at 1/8); objects kept by centre",
                  "less local median": "junction_analysis.less_local_median, clipped at 0 for Cellpose",
                  "sections": sections}
        settings_file().write_text(json.dumps(record, indent=1, ensure_ascii=False), encoding="utf-8")
        elapsed = time.perf_counter() - started
        print(f"[{index}/{len(todo)}] {line} — {elapsed / 60:.0f} min, "
              f"~{elapsed / index * (len(todo) - index) / 60:.0f} min left", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
