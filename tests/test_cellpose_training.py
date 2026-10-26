"""Checks for retraining Cellpose: cropping, the training set, and a run.

No GUI and no Cellpose: the crops are drawn discs, and the training call is a
stand-in that records what it was given and writes a file where Cellpose would.

Run with::

    python tests/test_cellpose_training.py
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from microscopy_viewer import cellpose_training as ct  # noqa: E402

_failures: list[str] = []


def check(condition: bool, message: str) -> None:
    if condition:
        print(f"  ok   {message}")
    else:
        print(f"  FAIL {message}")
        _failures.append(message)


def _discs(shape=(200, 300), n=12, radius=6, seed=0):
    """An image of bright discs and their label map."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    labels = np.zeros(shape, np.int32)
    for k in range(1, n + 1):
        cy, cx = rng.integers(radius + 2, shape[0] - radius - 2), rng.integers(radius + 2, shape[1] - radius - 2)
        labels[((yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2) & (labels == 0)] = k
    image = (labels > 0) * 1000 + rng.normal(100, 5, shape)
    return image.astype(np.uint16), labels


def check_cropping() -> None:
    print("cropping a box given in µm")
    rows, cols = ct.world_box_to_pixels((10.0, 20.0), (30.0, 60.0), (0.5, 0.5), (0.0, 0.0), (200, 300))
    check((rows, cols) == (slice(20, 60), slice(40, 120)), f"µm to pixels with the layer's scale: {rows}, {cols}")
    rows, cols = ct.world_box_to_pixels((30.0, 60.0), (10.0, 20.0), (0.5, 0.5), (0.0, 0.0), (200, 300))
    check((rows, cols) == (slice(20, 60), slice(40, 120)), "a box drawn from the other corner is the same box")
    rows, cols = ct.world_box_to_pixels((-50.0, -50.0), (1e4, 1e4), (1.0, 1.0), (0.0, 0.0), (200, 300))
    check((rows, cols) == (slice(0, 200), slice(0, 300)), "a box past the image is clipped to it")

    image, labels = _discs()
    crop, masks = ct.training_pair(image, (0.5, 0.5), (0, 0), labels, (0.5, 0.5), (0, 0), (10, 10), (60, 110))
    check(crop.shape == (100, 200) and masks.shape == crop.shape, f"same grid: crop {crop.shape}")
    check(np.array_equal(masks > 0, labels[20:120, 20:220] > 0), "same grid: the labels are the ones under the crop")
    check(masks.max() == len(np.unique(masks)) - 1, "labels renumbered 1..n")

    coarse = labels[::2, ::2]                       # a run on a decimated image: labels on a 2x grid
    crop2, masks2 = ct.training_pair(image, (0.5, 0.5), (0, 0), coarse, (1.0, 1.0), (0, 0), (10, 10), (60, 110))
    agree = ((masks2 > 0) == (labels[20:120, 20:220] > 0)).mean()
    check(masks2.shape == crop2.shape and agree > 0.95, f"coarser labels resampled onto the image's pixels ({agree:.0%} agree)")

    shifted, masks3 = ct.training_pair(image, (0.5, 0.5), (5.0, 5.0), labels, (0.5, 0.5), (5.0, 5.0), (15, 15), (65, 115))
    check(np.array_equal(masks3 > 0, labels[20:120, 20:220] > 0), "a translated layer pair crops the same pixels")


def check_refusals() -> None:
    print("crops that cannot be trained on")
    image, labels = _discs(n=3)
    check("at least" in ct.check_pair(image, ct.relabel(labels)), "fewer than five objects is refused, saying why")
    check("larger" in ct.check_pair(image[:10, :10], ct.relabel(labels[:10, :10])), "a sliver is refused")
    big = np.zeros((40, ct.MAX_SIDE_PX + 1), np.uint16)
    check("smaller" in ct.check_pair(big, big), "a crop over the size limit is refused")
    image, labels = _discs()
    check(ct.check_pair(image, ct.relabel(labels)) == "", "a crop of twelve discs is fine")


def check_training_set(folder: Path) -> ct.TrainingSet:
    print("the training set on disk")
    training = ct.TrainingSet(folder / "nuclei")
    check(training.pairs() == [] and "empty" in training.summary(), "a new set is empty")
    for seed in range(6):
        image, labels = _discs(seed=seed)
        training.add(image, labels, source=f"sample_{seed % 3}.ome.tiff", channel="AF405")
    pairs = training.pairs()
    check(len(pairs) == 6, f"six crops stored ({len(pairs)})")
    check(pairs[0].image.name == "0001.tif" and pairs[0].masks.name == "0001_masks.tif",
          "Cellpose's own layout: 0001.tif beside 0001_masks.tif")
    check(pairs[0].info["source"] == "sample_0.ome.tiff" and pairs[0].n_objects == 12, "where it came from is kept")
    check("6 crop(s)" in training.summary() and "3 sample(s)" in training.summary(), training.summary())
    try:
        image, labels = _discs(n=2)
        training.add(image, labels)
        check(False, "a crop with two objects is not stored")
    except ValueError as exc:
        check("object" in str(exc), f"a crop with two objects is not stored: {exc}")
    training.remove(pairs[-1])
    check(len(training.pairs()) == 5, "a crop can be taken out again")
    image, labels = _discs(seed=9)
    added = training.add(image, labels)
    check(added.image.name == "0006.tif", f"numbering continues after the last one ({added.image.name})")
    images, masks, _ = training.load()
    check(len(images) == 6 and images[0].dtype == np.uint16 and masks[0].max() == 12, "the set loads back")
    return training


def check_names() -> None:
    print("the new model's name")
    check(ct.safe_model_name("nuclei", taken=[]) == "nuclei_finetuned", "a Cellpose 3 zoo name is not reused")
    check(ct.safe_model_name("my model / v2") == "my_model_v2", "awkward characters replaced")
    check(ct.safe_model_name("dapi", taken=["dapi", "dapi_2"]) == "dapi_3", "an existing model is never overwritten")
    check(ct.safe_model_name("", base="cpsam").startswith("cpsam_"), "no name: the base model and the date")


def check_split() -> None:
    print("training and test crops")
    train_at, test_at = ct.split(10, 0.2)
    check(len(test_at) == 2 and not set(train_at) & set(test_at), f"10 crops: 8 train, 2 test ({len(train_at)}/{len(test_at)})")
    check(ct.split(10, 0.2) == (train_at, test_at), "the same seed, the same split")
    check(ct.split(4, 0.2) == ([0, 1, 2, 3], []), "fewer than five crops: all train, none held out")


def check_run(training: ct.TrainingSet, folder: Path) -> None:
    print("a run, with a stand-in for Cellpose")
    given = {}
    lines = []

    def fake_trainer(net, **kwargs):
        given.update(kwargs, net=net)
        import logging
        logging.getLogger("cellpose.train").info(">>> epoch 1, train_loss=0.5")
        out = Path(kwargs["save_path"]) / "models" / kwargs["model_name"]
        out.parent.mkdir(parents=True)
        out.write_bytes(b"weights")
        return out, np.array([0.9, 0.5]), np.array([0.0, 0.6])

    models = folder / "models"
    models.mkdir()
    (models / "dapi").write_bytes(b"old")
    settings = ct.TrainingSettings(base_model="cpsam", model_name="dapi", n_epochs=2)
    result = ct.train(training, settings, progress=lines.append, trainer=fake_trainer, net="NET", destination=models)
    check(result.model_name == "dapi_2" and result.model_path == models / "dapi_2", f"saved beside the others as {result.model_name}")
    check((models / "dapi").read_bytes() == b"old", "the model of the same name is untouched")
    check(given["net"] == "NET" and given["n_epochs"] == 2 and given["learning_rate"] == ct.LEARNING_RATE,
          "the settings reach the trainer")
    check(len(given["train_data"]) + len(given["test_data"] or []) == 6 and len(given["test_data"] or []) == 1,
          f"six crops: five to train, one held out ({len(given['train_data'])}/{len(given['test_data'] or [])})")
    check(given["min_train_masks"] == ct.MIN_OBJECTS, "Cellpose's own minimum matches the one used when adding")
    check(any("train_loss" in line for line in lines), "Cellpose's log lines reach the progress callback")
    check(result.train_losses == [0.9, 0.5] and "test loss 0.600" in result.describe(), result.describe())
    runs = [json.loads(l) for l in (training.folder / "runs.jsonl").read_text(encoding="utf-8").splitlines()]
    check(runs and runs[-1]["model_name"] == "dapi_2" and len(runs[-1]["crops"]) == 6, "the run is recorded beside the crops")

    empty = ct.TrainingSet(folder / "nothing")
    try:
        ct.train(empty, settings, trainer=fake_trainer, net="NET", destination=models)
        check(False, "an empty set does not train")
    except ValueError as exc:
        check("empty" in str(exc), f"an empty set does not train: {exc}")


def main() -> int:
    # Not a TemporaryDirectory: Windows can hold a just-written file a moment
    # longer, and failing the run over the clean-up would hide the checks.
    folder = Path(tempfile.mkdtemp())
    try:
        check_cropping()
        check_refusals()
        training = check_training_set(folder)
        check_names()
        check_split()
        check_run(training, folder)
    finally:
        shutil.rmtree(folder, ignore_errors=True)
    print()
    if _failures:
        print(f"{len(_failures)} check(s) failed")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
