"""Retrain Cellpose on corrected segmentations: the training set and the run.

The loop this supports, in the viewer:

1. **Segment** a sample with the model you have (Segmentation panel, Setup).
2. **Correct** the Labels layer by hand — paint the nuclei it missed, erase what
   is not a nucleus, split what it merged — inside one or more rectangles.
3. **Add** those rectangles to a *training set*: each becomes an image crop and
   its label map, written to a folder in the layout the Cellpose GUI and command
   line read too (``0001.tif`` beside ``0001_masks.tif``).
4. **Train** a new model from the set, starting from the current one. It is
   saved into Cellpose's own model folder, so the Segmentation panel offers it
   at the next Rescan, by name, like any other model.

Inside a rectangle every pixel counts: a nucleus left unlabelled is taught as
background. That is the one thing to get right when correcting.

No Qt and no napari here. Cellpose is imported only by :func:`train`, and the
training call itself can be replaced (``trainer=``), which is how the checks run
without it.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from .utils import get_logger

logger = get_logger("cellpose_training")

#: Suffix Cellpose expects on the label map beside each training image.
MASK_SUFFIX = "_masks"

#: A crop with fewer objects than this teaches little, and Cellpose's own
#: trainer skips it (``min_train_masks``); it is refused here instead, with a reason.
MIN_OBJECTS = 5

#: A crop larger than this, per side, is refused: Cellpose trains on 256-pixel
#: tiles drawn from each image, so a huge crop is mostly never seen — and is
#: usually a rectangle drawn over a whole uncorrected section by mistake.
MAX_SIDE_PX = 4096

#: Defaults of the run: Cellpose 4's own for fine-tuning cpsam.
N_EPOCHS = 100
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.1
#: Share of the crops held out to report a test loss, when there are enough.
TEST_SHARE = 0.2
#: Fewer crops than this and none are held out: every one is needed to train.
MIN_FOR_TEST = 5


def default_root() -> Path:
    """Where training sets are kept unless another folder is chosen."""
    from .runtime import app_data_dir

    return app_data_dir() / "cellpose training"


# ---------------------------------------------------------------------------
# Cropping
# ---------------------------------------------------------------------------


def world_box_to_pixels(
    top_left_um: Sequence[float], bottom_right_um: Sequence[float],
    scale: Sequence[float], translate: Sequence[float], shape: Sequence[int],
) -> tuple[slice, slice]:
    """The (row, column) slices of a layer covering a box given in world µm.

    *scale* and *translate* are the layer's own, last two axes; the box is
    clipped to the layer.
    """
    low = (np.minimum(top_left_um, bottom_right_um) - np.asarray(translate[-2:], float)) / np.asarray(scale[-2:], float)
    high = (np.maximum(top_left_um, bottom_right_um) - np.asarray(translate[-2:], float)) / np.asarray(scale[-2:], float)
    start = np.clip(np.floor(low).astype(int), 0, np.asarray(shape[-2:]))
    stop = np.clip(np.ceil(high).astype(int), 0, np.asarray(shape[-2:]))
    return slice(int(start[0]), int(stop[0])), slice(int(start[1]), int(stop[1]))


def resample_nearest(labels: np.ndarray, shape: Sequence[int]) -> np.ndarray:
    """*labels* stretched to *shape* by nearest neighbour — a label is never blended."""
    labels = np.asarray(labels)
    if tuple(labels.shape) == tuple(shape):
        return labels
    rows = np.minimum((np.arange(shape[0]) + 0.5) * labels.shape[0] / shape[0], labels.shape[0] - 1).astype(int)
    cols = np.minimum((np.arange(shape[1]) + 0.5) * labels.shape[1] / shape[1], labels.shape[1] - 1).astype(int)
    return labels[np.ix_(rows, cols)]


def relabel(labels: np.ndarray) -> np.ndarray:
    """Labels renumbered 1..n, in a type Cellpose reads; objects cut to nothing are gone."""
    labels = np.asarray(labels)
    ids, inverse = np.unique(labels, return_inverse=True)
    if ids.size and ids[0] == 0:
        out = inverse.reshape(labels.shape)
    else:
        out = inverse.reshape(labels.shape) + 1
    return out.astype(np.uint16 if out.max() < 65535 else np.uint32)


def training_pair(
    image: np.ndarray, image_scale: Sequence[float], image_translate: Sequence[float],
    labels: np.ndarray, labels_scale: Sequence[float], labels_translate: Sequence[float],
    top_left_um: Sequence[float], bottom_right_um: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    """The image and its labels inside one box, both on the image's own pixel grid.

    Both are 2D planes here (the caller picks the plane of a stack). The labels
    may sit on another grid — a run on a decimated stack returns coarser ones —
    and are resampled onto the image's pixels.
    """
    image = np.asarray(image)
    rows, cols = world_box_to_pixels(top_left_um, bottom_right_um, image_scale, image_translate, image.shape)
    crop = np.asarray(image[rows, cols])
    # The labels over exactly the image pixels kept, whatever their own grid.
    kept_low = np.array([rows.start, cols.start]) * np.asarray(image_scale[-2:], float) + np.asarray(image_translate[-2:], float)
    kept_high = np.array([rows.stop, cols.stop]) * np.asarray(image_scale[-2:], float) + np.asarray(image_translate[-2:], float)
    lrows, lcols = world_box_to_pixels(kept_low, kept_high, labels_scale, labels_translate, np.shape(labels))
    masks = np.asarray(np.asarray(labels)[lrows, lcols])
    if masks.size == 0 or crop.size == 0:
        return crop, np.zeros(crop.shape, dtype=np.uint16)
    return crop, relabel(resample_nearest(masks, crop.shape))


def check_pair(image: np.ndarray, masks: np.ndarray) -> str:
    """Why this crop cannot go into a training set, or ``""`` when it can."""
    if image.ndim != 2 or masks.shape != image.shape:
        return f"the crop is {image.shape} and its labels {masks.shape}; both have to be the same 2D shape"
    if min(image.shape) < 32:
        return f"the crop is only {image.shape[1]} × {image.shape[0]} px; draw a larger rectangle"
    if max(image.shape) > MAX_SIDE_PX:
        return (f"the crop is {image.shape[1]} × {image.shape[0]} px, over {MAX_SIDE_PX} px a side; "
                "draw smaller rectangles round what you corrected")
    n = int(len(np.unique(masks)) - (1 if (masks == 0).any() else 0))
    if n < MIN_OBJECTS:
        return f"the crop holds {n} object(s); at least {MIN_OBJECTS} are needed for Cellpose to train on it"
    return ""


# ---------------------------------------------------------------------------
# The training set: a folder of image / mask pairs
# ---------------------------------------------------------------------------


@dataclass
class Pair:
    """One crop of a training set, as stored."""

    image: Path
    masks: Path
    info: dict = field(default_factory=dict)

    @property
    def n_objects(self) -> int:
        return int(self.info.get("objects", 0))


class TrainingSet:
    """A folder of crops and their corrected labels, in Cellpose's own layout.

    ``0001.tif`` is the image, ``0001_masks.tif`` its labels, and ``0001.json``
    where it came from (sample, channel, box, voxel size, when). The folder can
    be handed to the Cellpose GUI or ``python -m cellpose --train`` unchanged.
    """

    def __init__(self, folder: str | Path):
        self.folder = Path(folder)

    def pairs(self) -> list[Pair]:
        if not self.folder.is_dir():
            return []
        out = []
        for image in sorted(self.folder.glob("*.tif")):
            if image.stem.endswith(MASK_SUFFIX):
                continue
            masks = image.with_name(f"{image.stem}{MASK_SUFFIX}.tif")
            if not masks.exists():
                continue
            meta = image.with_suffix(".json")
            info = json.loads(meta.read_text(encoding="utf-8")) if meta.exists() else {}
            out.append(Pair(image, masks, info))
        return out

    def summary(self) -> str:
        pairs = self.pairs()
        if not pairs:
            return f"{self.folder.name}: empty"
        objects = sum(p.n_objects for p in pairs)
        samples = len({p.info.get("source", "") for p in pairs})
        return f"{self.folder.name}: {len(pairs)} crop(s), {objects:,} object(s) from {samples} sample(s)"

    def add(self, image: np.ndarray, masks: np.ndarray, **info: Any) -> Pair:
        """Store one crop. Raises ``ValueError`` saying why when it cannot be used."""
        import tifffile

        problem = check_pair(np.asarray(image), np.asarray(masks))
        if problem:
            raise ValueError(problem)
        self.folder.mkdir(parents=True, exist_ok=True)
        taken = [int(p.image.stem) for p in self.pairs() if p.image.stem.isdigit()]
        stem = f"{(max(taken) + 1) if taken else 1:04d}"
        image_path = self.folder / f"{stem}.tif"
        masks_path = self.folder / f"{stem}{MASK_SUFFIX}.tif"
        masks = relabel(masks)
        tifffile.imwrite(image_path, np.asarray(image))
        tifffile.imwrite(masks_path, masks)
        info = {**info, "objects": int(masks.max()), "shape": list(np.shape(image)),
                "added": datetime.now().isoformat(timespec="seconds")}
        image_path.with_suffix(".json").write_text(json.dumps(info, indent=1, default=str), encoding="utf-8")
        return Pair(image_path, masks_path, info)

    def remove(self, pair: Pair) -> None:
        for path in (pair.image, pair.masks, pair.image.with_suffix(".json")):
            path.unlink(missing_ok=True)

    def load(self) -> tuple[list[np.ndarray], list[np.ndarray], list[Pair]]:
        import tifffile

        pairs = self.pairs()
        return ([tifffile.imread(p.image) for p in pairs], [tifffile.imread(p.masks) for p in pairs], pairs)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


@dataclass
class TrainingSettings:
    """What a run starts from and how long it goes on."""

    #: The model to fine-tune: a name Cellpose knows or a path to weights.
    base_model: str = "cpsam"
    #: Name of the new model, as it will appear in the model list.
    model_name: str = ""
    n_epochs: int = N_EPOCHS
    learning_rate: float = LEARNING_RATE
    weight_decay: float = WEIGHT_DECAY
    test_share: float = TEST_SHARE
    batch_size: int = 1
    use_gpu: bool = True
    #: Seed of the split into training and test crops, so a run can be repeated.
    seed: int = 0


@dataclass
class TrainingResult:
    model_path: Path
    model_name: str
    train_losses: list[float]
    test_losses: list[float]
    n_train: int
    n_test: int
    elapsed_s: float
    base_model: str
    device: str = ""

    def describe(self) -> str:
        last = f"training loss {self.train_losses[-1]:.3f}" if self.train_losses else ""
        tested = [v for v in self.test_losses if v and np.isfinite(v)]
        if tested:
            last += f", test loss {tested[-1]:.3f}"
        return (f"“{self.model_name}” trained from {self.base_model} on {self.n_train} crop(s)"
                f"{f' ({self.n_test} held out)' if self.n_test else ''} in {self.elapsed_s / 60:.1f} min"
                f"{f' on {self.device}' if self.device else ''}; {last}.")


def safe_model_name(name: str, base: str = "cpsam", taken: Sequence[str] = ()) -> str:
    """A file name for the new model: plain characters, not a Cellpose 3 zoo name, not taken."""
    from . import segmentation as sg

    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name).strip()).strip("._") or (
        f"{Path(str(base)).name}_{datetime.now():%Y%m%d_%H%M}")
    if sg.LEGACY_MODEL_PATTERN.match(stem.lower()):
        stem += "_finetuned"
    lowered = {Path(str(t)).name.lower() for t in taken}
    out, k = stem, 2
    while out.lower() in lowered:
        out, k = f"{stem}_{k}", k + 1
    return out


def split(n: int, share: float, seed: int = 0) -> tuple[list[int], list[int]]:
    """Indices to train on and to test on. None are held out from fewer than :data:`MIN_FOR_TEST`."""
    order = list(np.random.default_rng(seed).permutation(n))
    n_test = int(round(n * share)) if n >= MIN_FOR_TEST and share > 0 else 0
    return sorted(order[n_test:]), sorted(order[:n_test])


class _ForwardLog(logging.Handler):
    """Hands Cellpose's training log lines to a progress callback."""

    def __init__(self, progress: Callable[[str], None]):
        super().__init__(logging.INFO)
        self._progress = progress

    def emit(self, record: logging.LogRecord) -> None:
        try:
            text = record.getMessage().strip()
            if text:
                self._progress(text.lstrip(">").strip())
        except Exception:  # pragma: no cover - a progress hook must never stop training
            pass


def _cellpose_train(net, **kwargs):
    from cellpose import train as cp_train

    return cp_train.train_seg(net, **kwargs)


def model_folder() -> Path:
    """Cellpose's own model folder: what the Segmentation panel lists models from."""
    from cellpose import models as cp_models

    folder = Path(str(cp_models.MODEL_DIR))
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def train(
    training_set: TrainingSet,
    settings: TrainingSettings | None = None,
    progress: Callable[[str], None] | None = None,
    trainer: Callable[..., Any] | None = None,
    net: Any = None,
    destination: Path | None = None,
) -> TrainingResult:
    """Fine-tune *settings.base_model* on the set and save it beside Cellpose's models.

    *trainer* replaces ``cellpose.train.train_seg`` and *net* the network it is
    given; *destination* the model folder. All three are for the checks.
    """
    from . import segmentation as sg

    settings = settings or TrainingSettings()
    say = progress or (lambda _text: None)
    images, masks, pairs = training_set.load()
    if not pairs:
        raise ValueError(f"the training set “{training_set.folder}” is empty: add corrected crops first")
    usable = [k for k, m in enumerate(masks) if int(m.max()) >= MIN_OBJECTS]
    if not usable:
        raise ValueError(f"no crop in the set has {MIN_OBJECTS} objects or more")
    images, masks = [images[k] for k in usable], [masks[k] for k in usable]
    train_at, test_at = split(len(images), settings.test_share, settings.seed)

    destination = Path(destination) if destination is not None else model_folder()
    taken = [p.name for p in destination.iterdir()] if destination.is_dir() else []
    name = safe_model_name(settings.model_name, settings.base_model, taken)

    device = ""
    if net is None:
        from cellpose import models as cp_models

        info = sg.compute_device(prefer_gpu=settings.use_gpu)
        device = info.describe()
        say(f"loading {settings.base_model} on {device}")
        base = settings.base_model
        kwargs = {"gpu": bool(settings.use_gpu and info.is_gpu)}
        kwargs["pretrained_model"] = str(base)
        net = cp_models.CellposeModel(**kwargs).net

    say(f"training “{name}” on {len(train_at)} crop(s), testing on {len(test_at)}, {settings.n_epochs} epochs")
    handler = _ForwardLog(say)
    cellpose_log = logging.getLogger("cellpose")
    cellpose_log.addHandler(handler)
    previous_level = cellpose_log.level
    if cellpose_log.getEffectiveLevel() > logging.INFO:
        cellpose_log.setLevel(logging.INFO)
    started = time.perf_counter()
    try:
        with tempfile.TemporaryDirectory() as scratch:
            outcome = (trainer or _cellpose_train)(
                net,
                train_data=[images[k] for k in train_at], train_labels=[masks[k] for k in train_at],
                test_data=[images[k] for k in test_at] or None, test_labels=[masks[k] for k in test_at] or None,
                n_epochs=int(settings.n_epochs), learning_rate=float(settings.learning_rate),
                weight_decay=float(settings.weight_decay), batch_size=int(settings.batch_size),
                min_train_masks=MIN_OBJECTS, save_path=scratch, model_name=name,
            )
            written, train_losses, test_losses = outcome
            destination.mkdir(parents=True, exist_ok=True)
            target = destination / name
            shutil.move(str(written), target)
    finally:
        cellpose_log.removeHandler(handler)
        cellpose_log.setLevel(previous_level)
        sg.release_gpu_memory()

    result = TrainingResult(
        model_path=target, model_name=name,
        train_losses=[float(v) for v in np.ravel(train_losses)],
        test_losses=[float(v) for v in np.ravel(test_losses)] if test_losses is not None else [],
        n_train=len(train_at), n_test=len(test_at), elapsed_s=time.perf_counter() - started,
        base_model=str(settings.base_model), device=device,
    )
    # A record of the run beside the crops it was trained on.
    record = {**asdict(settings), "model_name": name, "model_path": str(target), "n_train": result.n_train,
              "n_test": result.n_test, "elapsed_s": result.elapsed_s, "device": device,
              "train_loss_last": result.train_losses[-1] if result.train_losses else None,
              "trained": datetime.now().isoformat(timespec="seconds"),
              "crops": [str(pairs[k].image.name) for k in usable]}
    runs = training_set.folder / "runs.jsonl"
    with runs.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=str) + "\n")
    logger.info("trained %s", result.describe())
    return result
