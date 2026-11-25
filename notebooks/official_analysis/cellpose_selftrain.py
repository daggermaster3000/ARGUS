"""Fine-tune Cellpose ``cpdino`` on AF488 with the cells the counting mask is surest of (self-training).

No hand labels: the training cells are the counting mask's own objects (Gaussian 20 µm background, median + 8 robust
SD, objects > 20 µm²) that are surely cells — per section the top :data:`TOP_SHARE` by mean background-subtracted
signal, among those of soma shape (circularity ≥ :data:`MIN_CIRCULARITY`, so that bright neurites and debris are not
taught as cells). Ranking within each section lets dim sections contribute their own brightest cells.

So that a real but dimmer cell is never taught as background, every other object of the mask at a low threshold
(:data:`UNSURE_K` SD, objects > 3 µm²), grown by :data:`UNSURE_GROW_PX`, is painted over with the local background (the
Gaussian 20 µm background the mask subtracts) in the training image. Crops of :data:`CROP_PX` pixels round sure cells,
AF488 and the nuclear stain each scaled per section as ``cellpose_af488.py`` scales them (normalize off).

The sections of the blinded count squares are not used, so ``cellpose_af488.py --model <name> --squares`` scores the
model on sections it has not seen.

    python notebooks/official_analysis/cellpose_selftrain.py          # prepare crops, train, save as cpdino_af488_self
    python notebooks/official_analysis/cellpose_selftrain.py --reuse  # train again on the crops already prepared

No Qt.
"""
from __future__ import annotations

import json
import logging
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "notebooks" / "validation"))
sys.path.insert(0, str(ROOT / "notebooks" / "dataset"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
import section_browser as sb  # noqa: E402

# ---------------------------------------------------------------- Settings
BASE_MODEL = "cpdino"
MODEL_NAME = "cpdino_af488_self"
N_SECTIONS = 120                  #: training sections, drawn at random (seed) from those without count squares
TOP_SHARE = 0.20                  #: per section, the brightest share of the counting mask's soma-shaped objects
MIN_CIRCULARITY = 0.5
UNSURE_K = 3                      #: every other object of the 3 SD mask is painted over with background
UNSURE_GROW_PX = 3
CROP_PX = 256
CROPS_PER_SECTION = 4
N_EPOCHS, LEARNING_RATE, WEIGHT_DECAY, BATCH = 100, 1e-5, 0.1, 4      #: batch 8 filled the 24 GB card and slowed to a crawl
SAVE_EVERY = 20                   #: a checkpoint every so many epochs
SEED = 20261125
WORKERS = 4
WORK = sb.OUT / "cellpose_selftrain"
NORM_PCT = (1.0, 99.9)            #: as cellpose_af488.py
# --------------------------------------------------------------------------


def scaled(plane: np.ndarray, tissue: np.ndarray, value: np.ndarray | None = None) -> np.ndarray:
    lo, hi = np.percentile(plane[tissue][::7], NORM_PCT)
    v = plane if value is None else value
    return np.clip((v - lo) / max(hi - lo, 1e-6), 0, 1.5).astype(np.float32)


def crops_of(name: str) -> tuple[list[np.ndarray], list[np.ndarray], str]:
    from scipy import ndimage
    from skimage.measure import label, regionprops

    import classical_vs_cellpose as cv
    import section_outline

    logging.getLogger("tifffile").setLevel(logging.ERROR)
    rng = random.Random(f"{SEED}-{name}")
    info = json.loads((sb.RUN / "cells" / f"{name}.json").read_text(encoding="utf-8"))
    af488, nuclear, voxel = cv.read_planes(Path(info["file"]))
    nuclear = nuclear.astype(np.float32)
    section = section_outline.outline_for(sb.MASKS / name, Path(info["file"]))
    tissue = section_outline.stats_mask(section, af488.shape)
    domain = section_outline.count_domain(section, af488.shape)
    line_um = section_outline.line_um(info)
    stored = json.loads((sb.MASKS / name / "summary.json").read_text(encoding="utf-8"))["thresholds"]
    smooth = ndimage.gaussian_filter(af488, cv.SMOOTH_UM / voxel)
    background = cv.local_background(smooth, cv.BACKGROUND_UM / voxel)
    signal = smooth - background
    del smooth
    sure_mask = section_outline.drop_outside(sb.cleaned((signal > float(stored["kmad8"])) & domain, voxel, 20.0), section, line_um, voxel)[0]
    labels = label(sure_mask, connectivity=2)
    props = [p for p in regionprops(labels, intensity_image=signal)
             if 4 * np.pi * p.area / max(p.perimeter, 1) ** 2 >= MIN_CIRCULARITY]
    if not props:
        return [], [], f"{sb.short(name)}: no soma-shaped object"
    props.sort(key=lambda p: -p.intensity_mean)
    sure = props[:max(1, int(round(TOP_SHARE * len(props))))]
    keep = np.zeros(labels.max() + 1, bool)
    keep[[p.label for p in sure]] = True
    sure_labels = np.where(keep[labels], labels, 0)
    unsure = sb.cleaned((signal > float(stored[f"kmad{UNSURE_K}"])) & domain, voxel, 3.0) & ~(sure_labels > 0)
    unsure = ndimage.binary_dilation(unsure, iterations=UNSURE_GROW_PX) & ~ndimage.binary_dilation(sure_labels > 0, iterations=1)
    a = scaled(af488, tissue)
    a[unsure] = scaled(af488, tissue, background)[unsure]          # unsure objects: painted over with background
    n = scaled(nuclear, tissue)
    del af488, nuclear, signal, background
    images, masks = [], []
    H, W = labels.shape
    for p in rng.sample(sure, min(CROPS_PER_SECTION, len(sure))):
        cy, cx = (int(v) for v in p.centroid)
        top = int(np.clip(cy - CROP_PX // 2 + rng.randint(-60, 60), 0, H - CROP_PX))
        left = int(np.clip(cx - CROP_PX // 2 + rng.randint(-60, 60), 0, W - CROP_PX))
        m = sure_labels[top:top + CROP_PX, left:left + CROP_PX]
        if not m.any():
            continue
        _, inv = np.unique(m, return_inverse=True)
        images.append(np.stack([a[top:top + CROP_PX, left:left + CROP_PX], n[top:top + CROP_PX, left:left + CROP_PX]]))
        masks.append(inv.reshape(m.shape).astype(np.int32))
    return images, masks, f"{sb.short(name)}: {len(sure)} sure of {len(props)} soma-shaped, {len(images)} crops"


def main() -> int:
    WORK.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout, force=True)
    if "--reuse" in sys.argv and (WORK / "crops.npz").exists():   # the crops of an earlier run
        z = np.load(WORK / "crops.npz")
        images, masks = list(z["images"]), list(z["masks"])
        started = time.perf_counter()
        print(f"{len(images)} crops reused", flush=True)
        return train_on(images, masks, started)
    held = set(pd.read_csv(sb.SQUARES_KEY)["name"])
    pool = [n for n in sb.section_table()["name"] if n not in held]
    names = random.Random(SEED).sample(pool, min(N_SECTIONS, len(pool)))
    images, masks = [], []
    started = time.perf_counter()
    with ProcessPoolExecutor(WORKERS) as ex:
        futures = {ex.submit(crops_of, n): n for n in names}
        for i, f in enumerate(as_completed(futures), 1):
            try:
                im, ma, line = f.result()
                images += im; masks += ma
            except Exception as exc:
                line = f"{sb.short(futures[f])}: FAILED {exc!r}"
            print(f"[{i}/{len(names)}] {line}", flush=True)
    print(f"{len(images)} crops, {sum(int(m.max()) for m in masks)} sure cells, {time.perf_counter() - started:.0f} s", flush=True)
    np.savez_compressed(WORK / "crops.npz", images=np.stack(images), masks=np.stack(masks))
    (WORK / "sections.json").write_text(json.dumps(names, indent=1), encoding="utf-8")
    return train_on(images, masks, started)


def train_on(images: list, masks: list, started: float) -> int:
    from cellpose import models, train

    order = list(range(len(images)))
    random.Random(SEED).shuffle(order)
    n_test = max(1, len(order) // 10)
    test, tr = order[:n_test], order[n_test:]
    net = models.CellposeModel(gpu=True, pretrained_model=BASE_MODEL).net
    path, train_losses, test_losses = train.train_seg(
        net, train_data=[images[k] for k in tr], train_labels=[masks[k] for k in tr],
        test_data=[images[k] for k in test], test_labels=[masks[k] for k in test], channel_axis=0,
        normalize=False, n_epochs=N_EPOCHS, learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY, batch_size=BATCH,
        min_train_masks=1, save_path=str(WORK), model_name=MODEL_NAME, save_every=SAVE_EVERY)
    record = {"model": str(path), "base": BASE_MODEL, "n_train": len(tr), "n_test": len(test), "epochs": N_EPOCHS,
              "train_loss_last": float(np.ravel(train_losses)[-1]), "test_loss_last": float(np.ravel(test_losses)[-1]),
              "minutes": round((time.perf_counter() - started) / 60, 1)}
    (WORK / "training.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    print(record, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
