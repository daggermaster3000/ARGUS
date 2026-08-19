"""Checks for the segmentation engine. No Qt, no display, no napari.

Cellpose itself is never run here: the models download hundreds of megabytes of
weights on first use, which is not something a check suite should do. What is
exercised instead is everything around it — the µm-to-pixel conversions, the
decimation and the label restore, the per-object table, and the whole
:func:`segment_volume` path through a stub backend. The Cellpose-dependent checks
report what the installed version offers and skip themselves when it is absent,
the same way the ANTs checks do.

Run with::

    python tests/test_segmentation.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from microscopy_viewer import segmentation as seg  # noqa: E402

_failures: list[str] = []


def check(condition: bool, message: str) -> None:
    if condition:
        print(f"  ok   {message}")
    else:
        print(f"  FAIL {message}")
        _failures.append(message)


def _has_cellpose() -> bool:
    return seg.backend_available("cellpose")


def _blobs(shape=(8, 40, 40), radius=4) -> tuple[np.ndarray, np.ndarray]:
    """A stack with two bright spheres, and the label map that describes them."""
    image = np.zeros(shape, dtype=np.float32)
    labels = np.zeros(shape, dtype=np.int32)
    grids = np.ogrid[tuple(slice(0, n) for n in shape)]
    for index, centre in enumerate(((4, 12, 12), (4, 28, 28)), start=1):
        distance = sum((grid - c) ** 2 for grid, c in zip(grids, centre))
        inside = distance <= radius**2
        labels[inside] = index
        image[inside] = 100.0 * index
    return image, labels


class _StubBackend(seg.Backend):
    """A backend that labels nothing, but records exactly what it was handed.

    The point of the seam is that :func:`segment_volume` can be checked without
    the library it wraps; this is what checks it.
    """

    name = "stub"
    install_hint = "the stub is always available"

    def __init__(self):
        self.calls: list[dict] = []

    def available(self) -> bool:
        return True

    def model_choices(self) -> tuple[seg.ModelChoice, ...]:
        return (seg.ModelChoice(label="stub-model", value="stub-model"),)

    def segment(self, image, settings, diameter_px, anisotropy, device, progress=None):
        self.calls.append(
            {
                "shape": tuple(np.asarray(image).shape),
                "diameter_px": diameter_px,
                "anisotropy": anisotropy,
                "settings": settings,
                "device": device,
            }
        )
        masks = np.zeros(np.asarray(image).shape, dtype=np.int32)
        masks[..., :4, :4] = 1  # one object, so the count is not trivially zero
        return masks, {"batch_size": 8}


def test_physical_units() -> None:
    print("physical units -> cellpose units")

    check(
        seg.diameter_in_pixels(5.0, (2.0, 0.5, 0.5)) == 10.0,
        "a 5 µm object at 0.5 µm/px is 10 px across",
    )
    check(
        seg.diameter_in_pixels(5.0, (2.0, 0.25, 0.25)) == 20.0,
        "the same object at 0.25 µm/px is 20 px — the setting stays in µm",
    )
    check(seg.diameter_in_pixels(0.0, (1.0, 1.0, 1.0)) is None, "zero diameter means automatic")

    check(seg.anisotropy_from_voxel((2.0, 0.5, 0.5)) == 4.0, "anisotropy is the Z/XY ratio")
    check(
        seg.anisotropy_from_voxel((1.0, 1.0, 1.0)) is None,
        "an isotropic stack reports no anisotropy rather than 1.0",
    )
    check(
        seg.anisotropy_from_voxel((0.0, 0.5, 0.5)) is None,
        "a missing Z step is reported as unknown, not as a divide by zero",
    )


def test_decimation_plan() -> None:
    print("decimation")

    check(seg.plan_scale((10, 100, 100), 1_000_000) == (1.0, 1.0, 1.0), "a small stack is untouched")

    factors = seg.plan_scale((100, 4000, 4000), 100_000_000)
    check(factors[0] == 1.0, "Z is never decimated — objects are already only a few planes tall")
    check(0 < factors[1] < 1.0 and factors[1] == factors[2], "XY is decimated by one shared factor")
    decimated = 100 * (4000 * factors[1]) * (4000 * factors[2])
    check(decimated <= 100_000_000 * 1.05, f"the plan lands under the limit ({decimated / 1e6:.0f} M)")

    volume = np.random.default_rng(0).random((8, 64, 64)).astype(np.float32)
    smaller = seg.rescale_image(volume, (1.0, 0.5, 0.5))
    check(smaller.shape == (8, 32, 32), "rescaling halves XY and leaves Z alone")

    labels = np.zeros((8, 32, 32), dtype=np.int32)
    labels[2:5, 4:12, 4:12] = 7
    restored = seg.restore_masks(labels, (8, 64, 64))
    check(restored.shape == (8, 64, 64), "labels come back on the original grid")
    check(set(np.unique(restored)) == {0, 7}, "nearest-neighbour restore invents no label values")


def test_object_table() -> None:
    print("the per-object table")

    image, labels = _blobs()
    stats = seg.object_table(labels, image, (2.0, 0.5, 0.5))
    check(len(stats) == 2, f"two objects measured (got {len(stats)})")

    first, second = stats
    voxel_volume = 2.0 * 0.5 * 0.5
    check(
        abs(first.volume_um3 - first.n_voxels * voxel_volume) < 1e-6,
        "volume is the voxel count times the calibrated voxel volume",
    )
    check(abs(first.mean - 100.0) < 1e-6, "intensity is read from the measure channel, not the mask")
    check(abs(second.mean - 200.0) < 1e-6, "each object is measured on its own voxels")
    check(abs(second.integrated - 200.0 * second.n_voxels) < 1e-3, "integrated intensity is the sum")
    check(abs(first.median - 100.0) < 1e-6 and abs(second.maximum - 200.0) < 1e-6, "median and max")

    # The blobs sit at z=4 in a 2 µm/plane stack, and at y=x=12 and 28 in 0.5 µm.
    check(
        abs(first.centroid_um[0] - 8.0) < 0.5 and abs(first.centroid_um[1] - 6.0) < 0.5,
        f"centroids are in µm, not voxels ({first.centroid_um})",
    )
    check(
        abs(second.centroid_um[2] - 14.0) < 0.5,
        f"the second centroid lands where the blob is ({second.centroid_um})",
    )

    # An 8-voxel-wide blob in a 2 × 0.5 × 0.5 µm stack is physically a squashed
    # ellipsoid, so the equivalent diameter is checked against the isotropic case
    # where "a sphere of radius 4" means what it says.
    isotropic = seg.object_table(labels, image, (1.0, 1.0, 1.0))[0]
    diameter = isotropic.equivalent_diameter_um
    check(7.0 < diameter < 9.0, f"equivalent diameter matches an 8-voxel sphere ({diameter:.2f})")


def test_table_edge_cases() -> None:
    print("table edge cases")

    empty = seg.object_table(np.zeros((4, 8, 8), dtype=np.int32), None, (1.0, 1.0, 1.0))
    check(empty == [], "an empty label map produces an empty table, not a row of zeros")

    # Cellpose leaves holes in the numbering when it drops small masks.
    holed = np.zeros((2, 4, 4), dtype=np.int32)
    holed[0, 0, 0] = 1
    holed[1, 2, 2] = 5
    stats = seg.object_table(holed, None, (1.0, 1.0, 1.0))
    check([stat.label for stat in stats] == [1, 5], "holes in the label numbering are skipped")

    flat = np.zeros((6, 6), dtype=np.int32)
    flat[1:4, 1:4] = 1
    plane = seg.object_table(flat, np.ones_like(flat, dtype=float), (0.5, 0.5))
    check(len(plane) == 1 and plane[0].n_voxels == 9, "a 2D image is measured as an area")
    check(plane[0].centroid_um[0] == 0.0, "a 2D object reports no Z centroid rather than guessing one")

    summary = seg.count_summary(stats)
    check(summary["count"] == 2, "the summary counts what the table holds")
    check(seg.count_summary([])["count"] == 0, "an empty summary is a count of zero")


def test_export_reuses_the_workbook_writer() -> None:
    print("export")

    from microscopy_viewer.exports import export_table

    image, labels = _blobs()
    stats = seg.object_table(labels, image, (2.0, 0.5, 0.5))
    frame = seg.object_dataframe(stats)
    check(len(frame) == 2, "the frame has one row per object")
    check("Volume (µm³)" in frame.columns, "columns carry their units")

    with tempfile.TemporaryDirectory() as folder:
        written = export_table(frame, Path(folder) / "objects.xlsx", sheet_name="Objects")
        check(written.exists() and written.stat().st_size > 0, "the object table writes a workbook")


def test_backend_reporting() -> None:
    print("backends")

    try:
        seg.get_backend("nope")
        check(False, "an unknown backend raises")
    except ValueError as exc:
        check("nope" in str(exc), f"an unknown backend names itself ({exc})")

    message = seg.missing_backend_message("cellpose")
    if _has_cellpose():
        check(message is None, "cellpose is installed, so nothing is reported as missing")
        models = seg.available_models("cellpose")
        check(bool(models), f"the installed cellpose offers {models}")
        check(
            seg.default_model("cellpose") in models,
            f"the default model is one it offers ({seg.default_model('cellpose')})",
        )
    else:
        check(
            message is not None and "pip install cellpose" in message,
            "without cellpose the panel is told exactly what to install",
        )
        check(seg.available_models("cellpose") == (), "no models are offered without cellpose")

    device = seg.compute_device()
    check(device.kind in ("cuda", "mps", "cpu"), f"the compute device is reported: {device.describe()}")
    check(
        seg.estimate_batch_size(seg.DeviceInfo()) == 8,
        "the CPU gets cellpose's own batch size rather than a GPU-sized one",
    )
    big = seg.DeviceInfo(kind="cuda", name="test", total_memory_bytes=24_000_000_000,
                         free_memory_bytes=24_000_000_000)
    check(seg.estimate_batch_size(big) > 8, "a large card gets a larger batch than the default 8")


def test_model_filtering() -> None:
    print("which models are offered")

    # Cellpose 4 cannot load a v3 zoo file, and v3 cannot load a v4 one. Both fail
    # minutes in with a shape mismatch, so they are filtered out of the list.
    check(not seg._model_loadable("cyto2torch_0", 4), "a v3 weights file is not offered on cellpose 4")
    check(seg._model_loadable("cyto2torch_0", 3), "and is offered on cellpose 3")
    check(not seg._model_loadable("nuclei", 4), "nor is the v3 nuclei model")
    check(not seg._model_loadable("cpsam", 3), "cpsam is not offered on cellpose 3")
    check(seg._model_loadable("cpsam_v2", 4), "but the v4 models are, on v4")

    # The filter must not swallow a model the user trained and named themselves.
    for name in ("nuclei_finetuned", "my_cyto_2026", "fish_dapi_v3", "cpdino_finetune"):
        check(seg._model_loadable(name, 4), f"a model of your own called “{name}” is still offered")

    check(not seg._model_loadable("size_cyto3torch_0.npy", 4), "size models are never offered")
    check(not seg._model_loadable("size_nuclei_torch_0.npy", 3), "on either version")


def test_model_choices_are_labelled() -> None:
    print("the model list")

    class _Choices(_StubBackend):
        name = "choices"

        def model_choices(self):
            return (
                seg.ModelChoice(label="cpsam", value="cpsam", source="builtin"),
                seg.ModelChoice(
                    label="fish_nuclei", value="D:/models/fish_nuclei", source="user",
                    detail="D:/models/fish_nuclei",
                ),
            )

    backend = _Choices()
    seg.register_backend(backend)

    choices = seg.model_choices("choices")
    check(len(choices) == 2, "builtins and trained models arrive in one list")
    check(
        choices[1].label == "fish_nuclei" and choices[1].value == "D:/models/fish_nuclei",
        "a trained model shows as its name and loads by its full path",
    )
    check("your own trained model" in choices[1].describe(), "and says where it came from")
    check(
        seg.available_models("choices") == ("cpsam", "D:/models/fish_nuclei"),
        "the values are what gets passed to the backend",
    )


def test_run_through_a_stub_backend() -> None:
    print("the run, through a stub backend")

    stub = _StubBackend()
    seg.register_backend(stub)

    image = np.random.default_rng(1).random((6, 40, 40)).astype(np.float32)
    settings = seg.SegmentationSettings(backend="stub", model="stub-model", diameter_um=5.0)
    result = seg.segment_volume(image, (2.0, 0.5, 0.5), settings=settings)

    check(result.masks.shape == image.shape, "the labels come back on the grid that went in")
    check(result.masks.dtype == np.int32, "labels are integers, ready for a Labels layer")
    check(result.n_objects == 1, f"the object count is read off the label map ({result.n_objects})")
    check(result.voxel_size_um == (2.0, 0.5, 0.5), "the result carries the voxel size for the layer scale")
    check(result.elapsed_s >= 0.0 and result.model == "stub-model", "the run records what produced it")

    call = stub.calls[-1]
    check(call["diameter_px"] == 10.0, "the diameter reaches the backend in pixels, not µm")
    check(
        call["anisotropy"] is None,
        "anisotropy is only passed in 3D mode — stitching does its own thing between planes",
    )

    volumetric = seg.SegmentationSettings(backend="stub", mode=seg.MODE_3D, diameter_um=5.0)
    seg.segment_volume(image, (2.0, 0.5, 0.5), settings=volumetric)
    check(stub.calls[-1]["anisotropy"] == 4.0, "3D mode is told the Z/XY ratio")

    try:
        seg.segment_volume(np.zeros((2, 2, 2, 2)), settings=seg.SegmentationSettings(backend="stub"))
        check(False, "a 4D array is refused")
    except ValueError as exc:
        check("2D or 3D" in str(exc), f"a 4D array is refused with a readable message ({exc})")


def test_decimation_inside_a_run() -> None:
    print("decimation inside a run")

    stub = _StubBackend()
    seg.register_backend(stub)

    image = np.zeros((4, 200, 200), dtype=np.float32)
    settings = seg.SegmentationSettings(backend="stub", diameter_um=4.0, max_voxels=40_000)
    result = seg.segment_volume(image, (2.0, 0.5, 0.5), settings=settings)

    call = stub.calls[-1]
    check(call["shape"][0] == 4, "Z survives decimation untouched")
    check(call["shape"][1] < 200, f"XY is decimated before segmentation ({call['shape']})")
    check(result.masks.shape == image.shape, "the labels are still returned on the full grid")
    check(result.decimated, "the result says it decimated")
    check(
        any("decimated" in warning for warning in result.warnings),
        "and says so in a warning the panel shows",
    )

    # The diameter has to follow the decimation, or every object is the wrong size.
    check(
        call["diameter_px"] is not None and call["diameter_px"] < 8.0,
        f"the diameter is converted after decimation, not before ({call['diameter_px']})",
    )


def test_gpu_request_is_reported() -> None:
    print("the GPU")

    stub = _StubBackend()
    seg.register_backend(stub)

    result = seg.segment_volume(
        np.zeros((2, 8, 8), dtype=np.float32),
        (1.0, 1.0, 1.0),
        settings=seg.SegmentationSettings(backend="stub", use_gpu=True),
    )
    device = seg.compute_device(prefer_gpu=True)
    check(result.device == device.describe(), f"the run records the device it used: {result.device}")
    if device.is_gpu:
        check(not result.warnings, "with a GPU present nothing is warned about")
        check(stub.calls[-1]["device"].is_gpu, "the backend is handed the GPU device")
    else:
        check(
            any("CPU" in warning for warning in result.warnings),
            "asking for a GPU that is not there is warned about rather than silently ignored",
        )

    off = seg.segment_volume(
        np.zeros((2, 8, 8), dtype=np.float32),
        (1.0, 1.0, 1.0),
        settings=seg.SegmentationSettings(backend="stub", use_gpu=False),
    )
    check(not stub.calls[-1]["device"].is_gpu, "unticking the GPU box means the CPU is used")
    check(not off.warnings, "and not asking for a GPU is not warned about")


def test_cellpose_call_shape() -> None:
    print("the cellpose call")

    if not _has_cellpose():
        print("  skip cellpose is not installed")
        return

    backend = seg.get_backend("cellpose")
    version = backend.major_version()
    check(version >= 3, f"cellpose {version}.x is installed")

    import inspect

    from cellpose import models as cp_models

    parameters = inspect.signature(cp_models.CellposeModel.eval).parameters
    for name in ("do_3D", "anisotropy", "stitch_threshold", "z_axis", "diameter", "batch_size"):
        check(name in parameters, f"model.eval still takes {name}")
    if version >= 4:
        check(
            "cpsam" in seg.available_models("cellpose"),
            "cellpose 4 offers the cpsam generalist",
        )
    else:
        check("nuclei" in seg.available_models("cellpose"), "cellpose 3 offers the model zoo")


def main() -> int:
    for test in (
        test_physical_units,
        test_decimation_plan,
        test_object_table,
        test_table_edge_cases,
        test_export_reuses_the_workbook_writer,
        test_backend_reporting,
        test_model_filtering,
        test_model_choices_are_labelled,
        test_run_through_a_stub_backend,
        test_decimation_inside_a_run,
        test_gpu_request_is_reported,
        test_cellpose_call_shape,
    ):
        test()
        print()

    if _failures:
        print(f"{len(_failures)} check(s) failed:")
        for failure in _failures:
            print(f"  - {failure}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
