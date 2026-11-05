"""Cellpose fine-tuned on the AF488+ cells the annotators clicked, each section predicted by a model that never saw it.

``james_cellpose488.py`` runs Cellpose untrained on the AF488 plane: it outlines
anything cell-shaped, dim unlabelled cells too. Here the model (:data:`BASE_MODEL`,
the run's) is fine-tuned to find only the cells James and Quillan called AF488+,
from their dots (``james_clicks.py``: one 5 µm dot per cell, not an outline).

1. **Training masks from dots.** Each dot becomes the AF488 body under it: a
   watershed seeded at the dot on the AF488 plane less its local median
   (``junction_analysis.less_local_median``) smoothed by ``cell_bodies.SMOOTH_UM``,
   over the stain foreground (``cell_bodies.stain_foreground``: median + 3 SD in
   the tissue), never more than :data:`BODY_REACH_UM` from the dot, every dot's seed
   competing. A dot with no foreground within :data:`SEED_REACH_UM` (or a body under
   :data:`MIN_BODY_UM2`) becomes a disk of radius :data:`DISK_UM`. Everything else
   **inside the area the annotator counted** is background: for James the whole
   section's tissue, for Quillan his zone (beside the run's line, from
   :data:`ZONE_HOME_UM` on the home side to :data:`ZONE_AWAY_UM` past it, ``away``
   with the sign of ``05_james_counts.ipynb``).
2. **Tiles.** :data:`TILE` px squares of the raw AF488 plane (brightest over z,
   as ``james_cellpose488`` reads it, so Cellpose sees at training what it sees
   at prediction: the raw plane, its own percentile normalisation), each wholly
   inside the counted area: tiles placed to hold every dot that can be held,
   and as many random tiles without a dot (negatives, nothing within
   :data:`BODY_REACH_UM`). Into ``cellpose_retrained/tiles/<section>/`` in the
   layout of ``microscopy_viewer.cellpose_training.TrainingSet``.
3. **Trainings** (:func:`trainings`): 2-fold cross-validation by section, stratified
   by annotator × genotype (seed :data:`SEED`), and the other annotator's
   sections (all of James → Quillan's, all of Quillan → James's) — 4 models,
   ``cellpose_training.train`` with its defaults and :data:`VALIDATION_SHARE` of
   the tiles held out for a validation loss. A tile with fewer than 5 objects
   is kept (``cellpose_training.MIN_OBJECTS`` is set to 0 here): a negative
   tile is the point.
4. **Prediction.** Every section, whole, in the run's tiles
   (``james_cellpose488.segment_tiled``), once by each of its held-out models
   (:data:`CROSS`, :data:`OTHER`); objects kept by centre in the tissue, measured
   and placed as ``james_cellpose488`` does. Into ``<annotator>_counts/methods/``
   ``objects|hits|cells/cellpose488_retrained/<section>.parquet``.

Everything else (folds.json, settings.json, models, training curves, label maps,
overlays of training pairs) goes to ``run_analysis_20261028/cellpose_retrained/``.

Run from the repository root with the Python that has cellpose and a CUDA torch::

    python notebooks/dataset/cellpose488_retrain.py tiles      # training tiles + overlays (minutes)
    python notebooks/dataset/cellpose488_retrain.py train      # the 4 trainings not done yet
    python notebooks/dataset/cellpose488_retrain.py train cv1  # just one, to time it
    python notebooks/dataset/cellpose488_retrain.py predict    # every section not done yet
    python notebooks/dataset/cellpose488_retrain.py numbers    # the quick comparison table

No Qt.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "validation"))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
import classical_vs_cellpose as cv  # noqa: E402
import james_cellpose488 as jcp  # noqa: E402
import james_clicks as jc  # noqa: E402

from microscopy_viewer import cell_bodies as cb  # noqa: E402
from microscopy_viewer import cellpose_training as ct  # noqa: E402
from microscopy_viewer import junction_analysis as ja  # noqa: E402
from microscopy_viewer import segmentation as sg  # noqa: E402

FAMILY = "cellpose488_retrained"
#: The two predictions every section gets, by method name.
CROSS = "Cellpose AF488 · retrained (cross-validated)"
OTHER = "Cellpose AF488 · retrained on the other annotator"
METHOD_NAMES = (CROSS, OTHER)
SLUGS = {CROSS: "cross-validated", OTHER: "other annotator"}
#: Where the training sets, models, folds and label maps go.
FOLDER = jc.OUT / "cellpose_retrained"
DESIGN_CSV = Path(r"D:\Slidescanner extracted\Image_File_List.csv")

#: The model fine-tuned: the run's own (``james_cellpose488.SETTINGS``), so "untrained" compares with it.
BASE_MODEL = jcp.SETTINGS.model
#: Tile side, px (0.325 µm/px: 166 µm).
TILE = 512
#: Furthest a body reaches from its dot (µm).
BODY_REACH_UM = 8.0
#: The seed is the foreground pixel nearest the dot within this (µm): the dot's own radius.
SEED_REACH_UM = 2.5
#: A dot without foreground, or with a body smaller than MIN_BODY_UM2, is a disk this radius (µm).
DISK_UM = 4.0
#: A body smaller than this (µm²) is not a cell body: the disk instead.
MIN_BODY_UM2 = 10.0
#: Quillan's zone: beside the line, from this far on the home side (µm) …
ZONE_HOME_UM = 50.0
#: … to this far past it (µm).
ZONE_AWAY_UM = 300.0
#: Crossed, as ``05_james_counts.ipynb``: beside, between GAP_UM and NEAR_UM past the line.
GAP_UM, NEAR_UM = 2.0, 300.0
#: Seed of the folds, the tile placement and the training / validation split.
SEED = 0
#: Share of a training's tiles held out for its validation loss.
VALIDATION_SHARE = 0.1
#: Training, as ``cellpose_training``'s defaults, one 256 px crop per step: a batch of 8
#: does not fit beside the viewer on the 24 GB card (spills into shared memory, > 10 min an
#: epoch), 4 was slower per tile than 1, 2 only 15 % faster for half the steps.
#: Epochs: 30, not cellpose_training's 100. In a first run of fold 1 at 100 epochs the validation
#: loss stopped falling by epoch 10 (0.041 at 5, 0.040 at 10, 0.044 at 20, 0.040 at 30) and each
#: epoch takes ~52 s, so 100 would cost ~1.5 h a training for nothing.
N_EPOCHS = 30
LEARNING_RATE = ct.LEARNING_RATE
WEIGHT_DECAY = ct.WEIGHT_DECAY
BATCH_SIZE = ct.TrainingSettings().batch_size
#: Shifts (px) tried round a dot to fit a tile inside the counted area.
SHIFTS = tuple(range(-240, 241, 48))
#: Random draws tried per section when looking for negative tiles.
NEGATIVE_DRAWS = 4000
#: Training pictures saved per section (dot tiles; plus one negative).
OVERLAYS = 2


# ---------------------------------------------------------------------------
# Sections, annotators, folds
# ---------------------------------------------------------------------------


def sections() -> pd.DataFrame:
    """Every annotated section: ``name``, ``annotator``, ``genotype``."""
    design = pd.read_csv(DESIGN_CSV)
    design["name"] = design["File_Name"].str.replace(".ome.tiff", "", regex=False)
    genotype = design.set_index("name")["Genotype"]
    rows = []
    for annotator, (_counts, target) in jc.ANNOTATORS.items():
        for name in sorted(pd.read_parquet(target / "clicks.parquet", columns=["name"])["name"].unique()):
            rows.append({"name": name, "annotator": annotator, "genotype": str(genotype.get(name, ""))})
    return pd.DataFrame(rows)


def folds(table: pd.DataFrame, seed: int = SEED) -> dict[str, int]:
    """Each section's fold (1 or 2): shuffled within annotator × genotype, dealt alternately.

    The dealing carries on from one stratum to the next, so odd strata do not all
    give their extra section to the same fold.
    """
    rng = np.random.default_rng(seed)
    out, turn = {}, 0
    for _key, group in table.groupby(["annotator", "genotype"], sort=True):
        for name in rng.permutation(group["name"].to_numpy()):
            out[str(name)] = 1 + turn % 2
            turn += 1
    return out


def trainings(table: pd.DataFrame) -> dict[str, dict]:
    """The 4 trainings: which sections each trains on, which it predicts, and for which method."""
    fold = folds(table)
    names = table["name"].tolist()
    by = table.set_index("name")["annotator"]
    plan = {}
    for k in (1, 2):
        plan[f"cv{k}"] = {"method": CROSS, "test": [n for n in names if fold[n] == k],
                          "train": [n for n in names if fold[n] != k]}
    plan["on_james"] = {"method": OTHER, "train": [n for n in names if by[n] == "james"],
                        "test": [n for n in names if by[n] == "quillan"]}
    plan["on_quillan"] = {"method": OTHER, "train": [n for n in names if by[n] == "quillan"],
                          "test": [n for n in names if by[n] == "james"]}
    return plan


def write_folds(table: pd.DataFrame) -> dict:
    fold = folds(table)
    record = {"seed": SEED, "stratified by": "annotator × genotype", "base_model": BASE_MODEL,
              "sections": {r.name: {"annotator": r.annotator, "genotype": r.genotype, "fold": fold[r.name]}
                           for r in table.itertuples()},
              "trainings": trainings(table)}
    FOLDER.mkdir(parents=True, exist_ok=True)
    (FOLDER / "folds.json").write_text(json.dumps(record, indent=1, ensure_ascii=False), encoding="utf-8")
    return record


# ---------------------------------------------------------------------------
# The area each annotator counted
# ---------------------------------------------------------------------------


def home_sign(cells: pd.DataFrame) -> float:
    """The sign that turns the run's ``d`` into ``away`` (+ = past the line), as ``05_james_counts.ipynb``."""
    pos = cells["AF488 positive"].to_numpy(bool) & cells["beside"].to_numpy(bool)
    return -1.0 if pos.sum() >= 5 and np.median(cells.loc[pos, "d"]) > 0 else 1.0


def zone_of(info: dict, cells: pd.DataFrame, tissue: np.ndarray, sign: float, step: int = 8) -> np.ndarray:
    """Quillan's zone at full size: beside the line, ``-ZONE_HOME_UM ≤ away ≤ ZONE_AWAY_UM``, in the tissue.

    Worked out on a grid of every *step* pixels and repeated back up.
    """
    voxel = float(info["voxel_yx"][0])
    small = tissue[::step, ::step]
    rows, cols = np.nonzero(small)
    centres = (np.stack([rows, cols], axis=1) * step + step / 2) * voxel
    placed = jc.place(info, cells, centres)
    away = placed["d"].to_numpy(float) * sign
    inside = placed["beside"].to_numpy(bool) & (away >= -ZONE_HOME_UM) & (away <= ZONE_AWAY_UM)
    zone_small = np.zeros(small.shape, dtype=bool)
    zone_small[rows[inside], cols[inside]] = True
    full = np.repeat(np.repeat(zone_small, step, axis=0), step, axis=1)[: tissue.shape[0], : tissue.shape[1]]
    return full & tissue


def counted_area(annotator: str, info: dict, cells: pd.DataFrame, tissue: np.ndarray) -> np.ndarray:
    """Where the annotator looked: James the whole tissue, Quillan his zone."""
    if annotator == "james":
        return tissue
    return zone_of(info, cells, tissue, home_sign(cells))


# ---------------------------------------------------------------------------
# Training masks from dots
# ---------------------------------------------------------------------------


def dot_bodies(smooth: np.ndarray, foreground: np.ndarray, dots_px: np.ndarray, voxel: float) -> tuple[np.ndarray, int]:
    """One label per dot (dot *k* → label *k* + 1): its AF488 body, or a disk; and how many are disks.

    *smooth* is the smoothed AF488 less its local median and *foreground* its
    stain foreground, over a window; *dots_px* the dots in that window, (row, col).
    """
    from skimage.segmentation import watershed

    height, width = smooth.shape
    labels = np.zeros((height, width), dtype=np.int32)
    if not len(dots_px):
        return labels, 0
    dots_px = np.asarray(dots_px, float)
    reach, seed_reach, disk = BODY_REACH_UM / voxel, SEED_REACH_UM / voxel, DISK_UM / voxel
    point = np.ones((height, width), dtype=bool)
    rr = np.clip(np.round(dots_px[:, 0]).astype(int), 0, height - 1)
    cc = np.clip(np.round(dots_px[:, 1]).astype(int), 0, width - 1)
    point[rr, cc] = False
    near = ndimage.distance_transform_edt(point) <= reach

    markers = np.zeros((height, width), dtype=np.int32)
    seeded = np.zeros(len(dots_px), dtype=bool)
    r = int(np.ceil(seed_reach))
    for k, (y, x) in enumerate(dots_px):
        top, left = max(int(round(y)) - r, 0), max(int(round(x)) - r, 0)
        patch = foreground[top: int(round(y)) + r + 1, left: int(round(x)) + r + 1]
        py, px = np.nonzero(patch)
        if not py.size:
            continue
        dist = np.hypot(py + top - y, px + left - x)
        best = int(np.argmin(dist))
        if dist[best] <= seed_reach and markers[py[best] + top, px[best] + left] == 0:
            markers[py[best] + top, px[best] + left] = k + 1
            seeded[k] = True
    if seeded.any():
        grown = watershed(-smooth, markers, mask=foreground & near)
    else:
        grown = np.zeros_like(labels)

    disks = 0
    min_px = MIN_BODY_UM2 / voxel ** 2
    R = int(np.ceil(max(reach, disk))) + 1
    for k, (y, x) in enumerate(dots_px):
        top, left = max(int(round(y)) - R, 0), max(int(round(x)) - R, 0)
        window = (slice(top, min(int(round(y)) + R + 1, height)), slice(left, min(int(round(x)) + R + 1, width)))
        yy, xx = np.mgrid[window[0], window[1]]
        dist = np.hypot(yy - y, xx - x)
        body = (grown[window] == k + 1) & (dist <= reach)
        if body.any():
            pieces, _n = ndimage.label(body)
            seed_at = np.argwhere(markers[window] == k + 1)
            if seed_at.size:
                body = pieces == pieces[tuple(seed_at[0])]
            body = ndimage.binary_fill_holes(body)
        if body.sum() < min_px:
            body = body | (dist <= disk)
            disks += 1
        target = labels[window]
        target[body & (target == 0)] = k + 1
    return labels, disks


# ---------------------------------------------------------------------------
# Tiles
# ---------------------------------------------------------------------------


def _integral(mask: np.ndarray) -> np.ndarray:
    out = np.zeros((mask.shape[0] + 1, mask.shape[1] + 1), dtype=np.int32)
    np.cumsum(np.cumsum(mask, axis=0, dtype=np.int32), axis=1, out=out[1:, 1:])
    return out


def _inside(integral: np.ndarray, top: int, left: int) -> bool:
    """The TILE square at (*top*, *left*) lies wholly in the mask (and in the image)."""
    height, width = integral.shape[0] - 1, integral.shape[1] - 1
    if top < 0 or left < 0 or top + TILE > height or left + TILE > width:
        return False
    total = (integral[top + TILE, left + TILE] - integral[top, left + TILE]
             - integral[top + TILE, left] + integral[top, left])
    return int(total) == TILE * TILE


def place_tiles(area: np.ndarray, dots_px: np.ndarray, voxel: float,
                rng: np.random.Generator) -> tuple[list, list, int]:
    """Top-left corners of the dot tiles and the negative tiles, and how many dots the dot tiles hold.

    Dots are taken in random order; a dot not yet in a tile gets the tile, among
    those round it (:data:`SHIFTS`) lying wholly in *area*, holding most dots not
    yet in a tile. Negatives: random squares in *area* with no dot within
    :data:`BODY_REACH_UM`, overlapping another negative by less than half a tile
    each way (they may overlap dot tiles); as many as the dot tiles.
    """
    integral = _integral(area)
    dots_px = np.asarray(dots_px, float).reshape(-1, 2)
    held = np.zeros(len(dots_px), dtype=bool)
    chosen: list[tuple[int, int]] = []
    for k in rng.permutation(len(dots_px)):
        if held[k]:
            continue
        y, x = dots_px[k]
        best, gain = None, 0
        for dy in SHIFTS:
            for dx in SHIFTS:
                top, left = int(round(y)) - TILE // 2 + dy, int(round(x)) - TILE // 2 + dx
                if not _inside(integral, top, left):
                    continue
                inside = ((dots_px[:, 0] >= top) & (dots_px[:, 0] < top + TILE)
                          & (dots_px[:, 1] >= left) & (dots_px[:, 1] < left + TILE))
                n = int((inside & ~held).sum())
                if n > gain:
                    best, gain = (top, left, inside), n
        if best is not None:
            chosen.append(best[:2])
            held |= best[2]

    margin = BODY_REACH_UM / voxel
    rows, cols = np.nonzero(area[::16, ::16])
    negatives: list[tuple[int, int]] = []
    taken: list[tuple[int, int]] = []
    for _ in range(NEGATIVE_DRAWS if rows.size else 0):
        if len(negatives) >= len(chosen):
            break
        j = int(rng.integers(rows.size))
        top, left = int(rows[j] * 16 - TILE // 2), int(cols[j] * 16 - TILE // 2)
        if not _inside(integral, top, left):
            continue
        if len(dots_px) and np.any((dots_px[:, 0] >= top - margin) & (dots_px[:, 0] < top + TILE + margin)
                                   & (dots_px[:, 1] >= left - margin) & (dots_px[:, 1] < left + TILE + margin)):
            continue
        if any(abs(top - t) < TILE // 2 and abs(left - l) < TILE // 2 for t, l in taken):
            continue
        negatives.append((top, left))
        taken.append((top, left))
    return chosen, negatives, int(held.sum())


def tile_set(name: str) -> ct.TrainingSet:
    return ct.TrainingSet(FOLDER / "tiles" / name)


def as_stored(plane: np.ndarray) -> np.ndarray:
    """The AF488 plane in its own integer type when it is one (it is read as uint16 and made float)."""
    if plane.size and float(plane.min()) >= 0 and float(plane.max()) < 65536 and np.all(plane == np.round(plane)):
        return plane.astype(np.uint16)
    return plane.astype(np.float32)


def save_overlay(image: np.ndarray, masks: np.ndarray, dots: np.ndarray, path: Path, title: str) -> None:
    """The tile in green (1–99.5 percentile), each mask's outline in magenta, the dots as white rings."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from skimage.segmentation import find_boundaries

    low, high = np.percentile(image, (1, 99.5))
    green = np.clip((image.astype(np.float32) - low) / max(high - low, 1e-6), 0, 1)
    rgb = np.zeros((*image.shape, 3), dtype=np.float32)
    rgb[..., 1] = green
    edge = find_boundaries(masks, mode="inner")
    rgb[edge] = (1.0, 0.0, 1.0)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.2))
    axes[0].imshow(green, cmap="gray")
    axes[0].set_title("AF488 (raw plane)")
    axes[1].imshow(rgb)
    if len(dots):
        axes[1].scatter(dots[:, 1], dots[:, 0], s=60, facecolors="none", edgecolors="white", linewidths=0.8)
    axes[1].set_title(f"{int(masks.max())} masks, {len(dots)} dots")
    for ax in axes:
        ax.set_axis_off()
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=90)
    plt.close(fig)


def make_tiles(name: str, annotator: str, seed: int) -> dict:
    """One section's training tiles (dot tiles and negatives) into :func:`tile_set`."""
    import shutil

    started = time.perf_counter()
    info = jc.section_info(name)
    af488, nuclear, voxel = cv.read_planes(Path(info["file"]))
    tissue = cv.tissue_mask(nuclear)
    del nuclear
    filtered = ja.less_local_median(af488, voxel)
    foreground = cb.stain_foreground(filtered, tissue, voxel)
    smooth = ndimage.gaussian_filter(filtered, cb.SMOOTH_UM / voxel)
    del filtered
    cells = jc.run_cells(name)
    area = counted_area(annotator, info, cells, tissue)

    clicks = pd.read_parquet(jc.ANNOTATORS[annotator][1] / "clicks.parquet")
    clicks = clicks[clicks["name"] == name]
    dots_px = clicks[["y", "x"]].to_numpy(float) / voxel
    in_area = area[np.clip(np.round(dots_px[:, 0]).astype(int), 0, area.shape[0] - 1),
                   np.clip(np.round(dots_px[:, 1]).astype(int), 0, area.shape[1] - 1)]
    rng = np.random.default_rng(seed)
    chosen, negatives, held = place_tiles(area, dots_px, voxel, rng)

    store = tile_set(name)
    if store.folder.exists():
        shutil.rmtree(store.folder)
    pad = int(np.ceil(BODY_REACH_UM / voxel)) + 4
    disks = bodies = 0
    overlays = []
    for kind, corners in (("dots", chosen), ("negative", negatives)):
        for top, left in corners:
            tile = (slice(top, top + TILE), slice(left, left + TILE))
            window = (slice(max(top - pad, 0), top + TILE + pad), slice(max(left - pad, 0), left + TILE + pad))
            wy, wx = window[0].start, window[1].start
            near = ((dots_px[:, 0] >= window[0].start) & (dots_px[:, 0] < window[0].stop)
                    & (dots_px[:, 1] >= window[1].start) & (dots_px[:, 1] < window[1].stop))
            labels, n_disks = dot_bodies(smooth[window], foreground[window], dots_px[near] - (wy, wx), voxel)
            masks = labels[top - wy: top - wy + TILE, left - wx: left - wx + TILE]
            image = as_stored(af488[tile])
            pair = store.add(image, masks, source=name, annotator=annotator, kind=kind, top=top, left=left,
                             tile_px=TILE, voxel_um=voxel, dots=int(np.sum(
                                 (dots_px[:, 0] >= top) & (dots_px[:, 0] < top + TILE)
                                 & (dots_px[:, 1] >= left) & (dots_px[:, 1] < left + TILE))))
            if kind == "dots":
                disks += n_disks
                bodies += int(near.sum())
            if (kind == "dots" and len([o for o in overlays if o[0] == "dots"]) < OVERLAYS) or (
                    kind == "negative" and not any(o[0] == "negative" for o in overlays)):
                inside = near & (dots_px[:, 0] >= top) & (dots_px[:, 0] < top + TILE) & (
                    dots_px[:, 1] >= left) & (dots_px[:, 1] < left + TILE)
                path = FOLDER / "overlays" / f"{name} {pair.image.stem} {kind}.png"
                save_overlay(af488[tile], ct.relabel(masks), dots_px[inside] - (top, left), path,
                             f"{name} ({annotator}) tile {pair.image.stem} {kind} at ({top}, {left})")
                overlays.append((kind, str(path)))
    return {"annotator": annotator, "dots": len(clicks), "dots in area": int(in_area.sum()), "dots in tiles": held,
            "dot tiles": len(chosen), "negative tiles": len(negatives), "dot bodies drawn": bodies,
            "of which disks": disks, "area (mm²)": round(float(area.sum()) * voxel ** 2 / 1e6, 3),
            "seconds": round(time.perf_counter() - started, 1)}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


class SectionsSet(ct.TrainingSet):
    """The tiles of several sections' sets, read as one; *folder* keeps the run record."""

    def __init__(self, folder: Path, names: list[str]):
        super().__init__(folder)
        self.names = list(names)

    def pairs(self) -> list[ct.Pair]:
        return [pair for name in self.names for pair in tile_set(name).pairs()]


def train_one(key: str, plan: dict) -> dict:
    """One training of the plan; its model, curves and record into ``FOLDER/trainings/<key>/``."""
    folder = FOLDER / "trainings" / key
    folder.mkdir(parents=True, exist_ok=True)
    models = FOLDER / "models"
    models.mkdir(parents=True, exist_ok=True)
    stale = models / f"af488_{key}"
    if stale.exists():
        stale.unlink()
    settings = ct.TrainingSettings(base_model=BASE_MODEL, model_name=f"af488_{key}", n_epochs=N_EPOCHS,
                                   learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
                                   test_share=VALIDATION_SHARE, batch_size=BATCH_SIZE, seed=SEED)
    training_set = SectionsSet(folder, plan["train"])
    log = (folder / "log.txt").open("w", encoding="utf-8")

    def say(text: str) -> None:
        log.write(text + "\n")
        log.flush()
        if "train_loss" in text or "training" in text or "loading" in text:
            print(f"  {key}: {text}", flush=True)

    try:
        result = ct.train(training_set, settings, progress=say, destination=models)
    finally:
        log.close()
    tested = {i: v for i, v in enumerate(result.test_losses) if i == 5 or i % 10 == 0}
    record = {"key": key, "method": plan["method"], "train": plan["train"], "test": plan["test"],
              "model_path": str(result.model_path), "n_train_tiles": result.n_train,
              "n_validation_tiles": result.n_test, "elapsed_s": round(result.elapsed_s, 1),
              "device": result.device, "settings": asdict(settings),
              "train_losses": result.train_losses, "validation_losses": tested}
    (folder / "curves.json").write_text(json.dumps(record, indent=1, ensure_ascii=False), encoding="utf-8")
    return record


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


def model_settings(model_path: str) -> sg.SegmentationSettings:
    """The untrained run's settings (``james_cellpose488.SETTINGS``) with a trained model."""
    return replace(jcp.SETTINGS, custom_model_path=str(model_path))


def predict_one(name: str, annotator: str, models: dict[str, str]) -> dict:
    """One section by each of its held-out models (*models*: method → model path)."""
    import tifffile

    started = time.perf_counter()
    info = jc.section_info(name)
    af488, nuclear, voxel = cv.read_planes(Path(info["file"]))
    tissue = cv.tissue_mask(nuclear)
    del nuclear
    filtered = ja.less_local_median(af488, voxel)
    tiles = jcp.tiles_of(tissue)
    cells = jc.run_cells(name)
    target = jc.ANNOTATORS[annotator][1]
    clicks = pd.read_parquet(target / "clicks.parquet")
    clicks = clicks[clicks["name"] == name]
    grown = ja.read_labels(jc.RUN / info["grown_file"])
    labels_of_run = np.unique(grown[grown > 0]).astype(np.int64)
    size = np.asarray(ndimage.sum(grown > 0, grown, labels_of_run), dtype=float)

    objects, cover = [], pd.DataFrame({"Label": labels_of_run})
    hits = pd.DataFrame({"name": name, "dot": clicks["dot"].to_numpy()})
    timing = {"annotator": annotator, "tiles": len(tiles), "models": models}
    folder = FOLDER / "labels"
    folder.mkdir(parents=True, exist_ok=True)
    for method in METHOD_NAMES:
        begun = time.perf_counter()
        labels = jcp.segment_tiled(af488, tiles, voxel, settings=model_settings(models[method]))
        segment_s = time.perf_counter() - begun
        labels, before, after = jcp.keep_in_tissue(labels, tissue)
        objects.append(jcp.object_table(labels, af488, filtered, voxel, info, cells, method))
        hits[method] = jcp.dot_hits(labels, clicks[["y", "x"]].to_numpy(float), voxel)
        cover[f"cover {method}"] = (np.asarray(ndimage.sum(labels > 0, grown, labels_of_run))
                                    / np.where(size > 0, size, np.nan))
        tifffile.imwrite(folder / f"{name} {SLUGS[method]}.tif", labels, compression="zlib")
        timing[method] = {"segment_s": round(segment_s, 1), "objects_found": before, "objects_in_tissue": after}
        del labels
        sg.release_gpu_memory()

    for kind, table in (("objects", pd.concat(objects, ignore_index=True)), ("hits", hits), ("cells", cover)):
        out = target / "methods" / kind / FAMILY
        out.mkdir(parents=True, exist_ok=True)
        table.to_parquet(out / f"{name}.parquet", index=False)
    timing["seconds"] = round(time.perf_counter() - started, 1)
    return timing


# ---------------------------------------------------------------------------
# The quick numbers
# ---------------------------------------------------------------------------


def zones(table: pd.DataFrame, sign: float, annotator: str) -> dict[str, np.ndarray]:
    """Which rows (dots or objects, placed) are in each zone: counted, region, crossed."""
    away = table["d"].to_numpy(float) * sign
    beside = table["beside"].to_numpy(bool)
    counted = (np.ones(len(table), bool) if annotator == "james"
               else beside & (away >= -ZONE_HOME_UM) & (away <= ZONE_AWAY_UM))
    return {"counted": counted, "region": table["region"].to_numpy(bool),
            "crossed": beside & (away > GAP_UM) & (away <= NEAR_UM)}


def numbers() -> pd.DataFrame:
    """Per method, annotator and zone: objects and dots per section, recall, precision, F1."""
    table = sections()
    rows = []
    for r in table.itertuples():
        target = jc.ANNOTATORS[r.annotator][1]
        sign = home_sign(jc.run_cells(r.name))
        clicks = pd.read_parquet(target / "clicks.parquet")
        dots = clicks[clicks["name"] == r.name].reset_index(drop=True)
        sources = [(FAMILY, METHOD_NAMES)]
        if (target / "methods" / "objects" / "cellpose488" / f"{r.name}.parquet").exists():
            sources.append(("cellpose488", (jcp.RAW,)))
        for family, methods in sources:
            path_o = target / "methods" / "objects" / family / f"{r.name}.parquet"
            path_h = target / "methods" / "hits" / family / f"{r.name}.parquet"
            if not (path_o.exists() and path_h.exists()):
                continue
            objects, hits = pd.read_parquet(path_o), pd.read_parquet(path_h)
            for method in methods:
                obj = objects[objects["method"] == method]
                hit = hits.set_index("dot").loc[dots["dot"], method].to_numpy()
                dot_zone, obj_zone = zones(dots, sign, r.annotator), zones(obj, sign, r.annotator)
                taken = set(hit[hit > 0].tolist())
                for zone in ("counted", "region", "crossed"):
                    in_obj = obj[obj_zone[zone]]
                    rows.append({"method": method, "annotator": r.annotator, "name": r.name, "zone": zone,
                                 "dots": int(dot_zone[zone].sum()), "objects": len(in_obj),
                                 "dots hit": int((hit[dot_zone[zone]] > 0).sum()),
                                 "objects with a dot": int(in_obj["object"].isin(taken).sum())})
    per = pd.DataFrame(rows)
    sums = per.groupby(["method", "annotator", "zone"], sort=False)[
        ["dots", "objects", "dots hit", "objects with a dot"]].sum()
    sums["sections"] = per.groupby(["method", "annotator", "zone"], sort=False)["name"].nunique()
    sums["objects / dots"] = sums["objects"] / sums["dots"]
    sums["recall"] = sums["dots hit"] / sums["dots"]
    sums["precision"] = sums["objects with a dot"] / sums["objects"]
    sums["F1"] = 2 * sums["recall"] * sums["precision"] / (sums["recall"] + sums["precision"])
    per.to_csv(FOLDER / "numbers_per_section.csv", index=False)
    sums.reset_index().to_csv(FOLDER / "numbers.csv", index=False)
    return sums.reset_index()


# ---------------------------------------------------------------------------
# Settings record and the command line
# ---------------------------------------------------------------------------


def read_settings() -> dict:
    try:
        return json.loads((FOLDER / "settings.json").read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_settings(record: dict) -> None:
    record.update({
        "family": FAMILY, "methods": list(METHOD_NAMES), "base_model": BASE_MODEL,
        "masks": {"body": "watershed on -(AF488 less local median, smoothed SMOOTH_UM) from the dot's seed, "
                          "over cell_bodies.stain_foreground (median + FG_K SD in tissue), all dots competing",
                  "smooth_um": cb.SMOOTH_UM, "fg_k": cb.FG_K, "body_reach_um": BODY_REACH_UM,
                  "seed_reach_um": SEED_REACH_UM, "disk_um (radius)": DISK_UM, "min_body_um2": MIN_BODY_UM2},
        "counted area": {"james": "classical_vs_cellpose.tissue_mask of the whole section",
                         "quillan": f"beside the run's line, -{ZONE_HOME_UM:g} ≤ away ≤ {ZONE_AWAY_UM:g} µm, in tissue"},
        "tiles": {"tile_px": TILE, "image": "raw AF488 plane, brightest over z (junction_analysis._plane)",
                  "negatives": "as many as dot tiles, random, no dot within body_reach_um, < half a tile overlap between negatives",
                  "shifts_px": list(SHIFTS)},
        "training": {"n_epochs": N_EPOCHS, "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
                     "batch_size": BATCH_SIZE, "validation_share": VALIDATION_SHARE, "seed": SEED,
                     "min_objects": 0},
        "prediction": {"segmentation": asdict(jcp.SETTINGS), "tile_px": ja.TILE_PX,
                       "tile_overlap_px": ja.TILE_OVERLAP_PX, "tile_min_tissue": ja.TILE_MIN_TISSUE,
                       "hit_um": jcp.HIT_UM, "tissue": "objects kept by centre (james_cellpose488.keep_in_tissue)"},
    })
    FOLDER.mkdir(parents=True, exist_ok=True)
    (FOLDER / "settings.json").write_text(json.dumps(record, indent=1, ensure_ascii=False, default=str),
                                          encoding="utf-8")


def main(argv: list[str]) -> int:
    stage = argv[0] if argv else ""
    # Negative tiles hold no object: keep them (``TrainingSet.add``, ``train`` and Cellpose refuse
    # crops under MIN_OBJECTS).
    ct.MIN_OBJECTS = 0
    table = sections()
    record = read_settings()
    plan = write_folds(table)["trainings"]
    by = table.set_index("name")["annotator"]
    if stage == "tiles":
        done = record.setdefault("tiles", {})
        for index, name in enumerate(table["name"], start=1):
            done[name] = make_tiles(name, by[name], SEED + index)
            print(f"[{index}/{len(table)}] {name}: {done[name]}", flush=True)
            write_settings(record)
    elif stage == "train":
        done = record.setdefault("trainings", {})
        keys = argv[1:] or [k for k in plan if k not in done]
        for key in keys:
            print(f"training {key}: {len(plan[key]['train'])} sections → {len(plan[key]['test'])}", flush=True)
            curves = train_one(key, plan[key])
            done[key] = {k: v for k, v in curves.items() if k not in ("train_losses", "validation_losses")}
            print(f"{key}: {curves['elapsed_s'] / 60:.1f} min, {curves['n_train_tiles']} tiles, "
                  f"last train loss {curves['train_losses'][-1]:.3f}", flush=True)
            write_settings(record)
    elif stage == "predict":
        trained = record.get("trainings", {})
        done = record.setdefault("predicted", {})
        todo = [n for n in table["name"] if n not in done]
        if argv[1:]:
            todo = todo[: int(argv[1])]
        started = time.perf_counter()
        for index, name in enumerate(todo, start=1):
            models = {plan[k]["method"]: trained[k]["model_path"] for k in plan if name in plan[k]["test"]}
            done[name] = predict_one(name, by[name], models)
            write_settings(record)
            elapsed = time.perf_counter() - started
            print(f"[{index}/{len(todo)}] {name}: {done[name]['seconds']:.0f} s, "
                  + ", ".join(f"{SLUGS[m]} {done[name][m]['objects_in_tissue']}" for m in METHOD_NAMES)
                  + f" — ~{elapsed / index * (len(todo) - index) / 60:.0f} min left", flush=True)
    elif stage == "numbers":
        pd.set_option("display.width", 200)
        print(numbers().round(3).to_string(index=False))
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
