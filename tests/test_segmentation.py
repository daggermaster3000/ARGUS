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


def test_unsupported_models_are_named() -> None:
    print("models the installed cellpose cannot load")

    class _Mixed(_StubBackend):
        name = "mixed"

        def major_version(self):
            return 4

        def unsupported_models(self):
            return ("cyto2torch_0", "nuclei")

    seg.register_backend(_Mixed())
    note = seg.unsupported_models_message("mixed")
    check(note is not None, "local models the version refuses are reported, not silently dropped")
    check("cyto2torch_0" in note, f"and named ({note[:60]}…)")
    check('pip install "cellpose<4"' in note, "with the command that would make them usable")

    class _Clean(_StubBackend):
        name = "clean"

    seg.register_backend(_Clean())
    check(seg.unsupported_models_message("clean") is None, "nothing is said when nothing is refused")


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


def test_duplicate_openmp_is_allowed() -> None:
    """Importing the package must defuse the OpenMP abort before torch loads.

    torch ships its own Intel OpenMP runtime while numpy and scipy use conda's
    MKL copy. The second one to initialise prints "OMP: Error #15" and calls
    abort(), which kills the process outright — this suite could not run at all
    until the switch was set.
    """
    print("the OpenMP duplicate guard")
    import os

    from microscopy_viewer import runtime

    check(
        os.environ.get(runtime.OPENMP_DUPLICATE_VAR) == "TRUE",
        f"importing the package sets {runtime.OPENMP_DUPLICATE_VAR}",
    )

    # Reaching this line at all means torch and numpy coexisted in one process.
    check(seg.compute_device() is not None, "the device can be probed without aborting")

    # An explicit setting is the user's call and must survive.
    previous = os.environ.get(runtime.OPENMP_DUPLICATE_VAR)
    os.environ[runtime.OPENMP_DUPLICATE_VAR] = "FALSE"
    try:
        check(not runtime.allow_duplicate_openmp(), "an explicit setting is left alone")
        check(os.environ[runtime.OPENMP_DUPLICATE_VAR] == "FALSE", "and keeps its value")
    finally:
        if previous is None:
            os.environ.pop(runtime.OPENMP_DUPLICATE_VAR, None)
        else:
            os.environ[runtime.OPENMP_DUPLICATE_VAR] = previous

    os.environ.pop(runtime.OPENMP_DUPLICATE_VAR, None)
    try:
        check(runtime.allow_duplicate_openmp(), "an unset variable is set")
    finally:
        os.environ[runtime.OPENMP_DUPLICATE_VAR] = previous or "TRUE"


def test_max_projection_mode() -> None:
    """A projected run flattens Z, and everything downstream follows it."""
    print("segmenting a maximum projection")

    backend = _StubBackend()
    seg.register_backend(backend)

    # A stack whose signal lives on one plane only, so a projection is provably
    # different from taking the first slice.
    image = np.zeros((6, 20, 24), dtype=np.float32)
    image[3, 2, 2] = 500.0

    flat = seg.max_projection(image)
    check(flat.shape == (20, 24), f"max_projection drops Z ({flat.shape})")
    check(float(flat[2, 2]) == 500.0, "and keeps the brightest voxel, not the first plane")
    already = np.zeros((8, 8), dtype=np.float32)
    check(seg.max_projection(already).shape == (8, 8), "a 2D array passes through untouched")

    arr, voxel = seg.project_for_mode(image, (2.0, 0.5, 0.25), seg.MODE_MIP)
    check(arr.shape == (20, 24), "project_for_mode flattens the array")
    check(voxel == (0.5, 0.25), f"and drops the Z voxel with it ({voxel})")
    untouched, kept = seg.project_for_mode(image, (2.0, 0.5, 0.25), seg.MODE_3D)
    check(untouched.shape == image.shape and kept == (2.0, 0.5, 0.25), "other modes are left alone")

    settings = seg.SegmentationSettings(backend="stub", mode=seg.MODE_MIP, diameter_um=2.0)
    check(settings.is_projection, "the mode reports itself as a projection")
    check(not settings.do_3d, "and is not volumetric")

    result = seg.segment_volume(image, (2.0, 0.5, 0.25), settings=settings)
    check(result.projected, "the result records that it projected")
    check(result.masks.shape == (20, 24), f"labels come back 2D ({result.masks.shape})")
    check(backend.calls[-1]["shape"] == (20, 24), "the backend was handed the flattened image")
    check(backend.calls[-1]["anisotropy"] is None, "a flattened image has no anisotropy")
    # The conversion averages the two lateral voxel sizes, so 2 µm over
    # (0.5, 0.25) is 2 / 0.375. Z is not in it, which is the point.
    check(
        abs(backend.calls[-1]["diameter_px"] - 2.0 / 0.375) < 1e-9,
        f"the diameter used XY only, not Z ({backend.calls[-1]['diameter_px']:.3f} px)",
    )
    isotropic = seg.SegmentationSettings(backend="stub", mode=seg.MODE_MIP, diameter_um=2.0)
    seg.segment_volume(image, (2.0, 0.25, 0.25), settings=isotropic)
    check(
        backend.calls[-1]["diameter_px"] == 8.0,
        f"on square pixels that is simply 2 µm / 0.25 = 8 px ({backend.calls[-1]['diameter_px']})",
    )
    check(
        any("maximum projection" in w for w in result.warnings),
        f"and it says so out loud ({result.warnings})",
    )

    # A 2D image in projection mode is a no-op, not an error.
    plain = seg.segment_volume(np.zeros((10, 10), np.float32), (0.5, 0.5), settings=settings)
    check(not plain.projected, "a 2D input was never projected")
    check(plain.masks.shape == (10, 10), "and comes back unchanged")


def test_projected_measurements_are_areas() -> None:
    """2D labels measure an area, and the column has to say so."""
    print("areas versus volumes")

    masks = np.zeros((20, 24), dtype=np.int32)
    masks[:4, :5] = 1  # 20 px
    stats = seg.object_table(masks, np.full((20, 24), 7.0), (0.5, 0.25))
    check(len(stats) == 1, "one object")
    stat = stats[0]
    check(stat.n_voxels == 20, f"twenty pixels ({stat.n_voxels})")
    check(
        abs(stat.volume_um3 - 20 * 0.5 * 0.25) < 1e-9,
        f"the size is an area: 20 px x 0.5 x 0.25 = 2.5 µm² ({stat.volume_um3})",
    )
    check(abs(stat.mean - 7.0) < 1e-9, "intensities are measured on the flattened signal")
    check(stat.centroid_um[0] == 0.0, "a 2D object has no Z centroid")
    check(stat.as_row()["centroid_z_um"] == 0.0, "and the exported row reports it as zero")

    headers = seg.object_headers(2)
    check(headers["volume_um3"] == "Area (µm²)", f"2D headers say area ({headers['volume_um3']})")
    check(headers["n_voxels"] == "Pixels", "and pixels, not voxels")
    check(seg.object_headers(3)["volume_um3"] == "Volume (µm³)", "3D headers are unchanged")
    check(seg.object_headers()["volume_um3"] == "Volume (µm³)", "and 3D is the default")

    # The heading is chosen by the caller, never guessed from the data: a real 3D
    # run can have every object sitting at z=0.
    frame_2d = seg.object_dataframe(stats, ndim=2)
    check("Area (µm²)" in frame_2d.columns, f"the export follows ndim ({list(frame_2d.columns)[:3]})")
    frame_3d = seg.object_dataframe(stats, ndim=3)
    check("Volume (µm³)" in frame_3d.columns, "and is not inferred from the stats")


def test_per_plane_needs_no_z_axis() -> None:
    """Cellpose refuses z_axis in plain 2D mode; the modes that ask for it must not send it."""
    print("2D modes on a stack")

    seen: list[dict] = []

    class _Recorder(seg.Backend):
        name = "recorder"
        install_hint = "-"

        def available(self) -> bool:
            return True

        def models(self):
            return ("m",)

        def model_choices(self):
            return (seg.ModelChoice(label="m", value="m"),)

        def segment(self, image, settings, diameter_px, anisotropy, device, progress=None):
            seen.append({"mode": settings.mode, "min_size": settings.min_size})
            masks = np.zeros(np.asarray(image).shape, dtype=np.int32)
            masks[..., :3, :3] = 1
            return masks, {}

    seg.register_backend(_Recorder())
    volume = np.zeros((4, 16, 16), dtype=np.float32)

    for mode in (seg.MODE_STITCH, seg.MODE_SLICES, seg.MODE_3D, seg.MODE_MIP):
        settings = seg.SegmentationSettings(backend="recorder", mode=mode)
        result = seg.segment_volume(volume, (2.0, 0.5, 0.5), settings=settings)
        check(result.masks.shape[-2:] == (16, 16), f"{mode} returns labels on the grid")

    check(
        seg.filter_ndim(seg.MODE_3D, 3) == 3,
        "3D filters objects by volume",
    )
    for mode in (seg.MODE_STITCH, seg.MODE_SLICES, seg.MODE_MIP):
        check(
            seg.filter_ndim(mode, 3) == 2,
            f"{mode} filters 2D masks even though the input is a stack",
        )
    check(seg.filter_ndim(seg.MODE_3D, 2) == 2, "a 2D image is 2D whatever the mode says")


def test_minimum_size_in_microns() -> None:
    """The minimum size is a diameter in µm, converted where the voxel size is known."""
    print("minimum size in µm")

    # A 4 µm disc at 0.5 µm/px: area pi*2^2 = 12.57 µm², over 0.25 µm² per px.
    pixels = seg.min_size_in_pixels(4.0, (0.5, 0.5), 2)
    check(pixels == round(np.pi * 4.0 / 0.25), f"a 4 µm disc is {pixels} px at 0.5 µm/px")
    finer = seg.min_size_in_pixels(4.0, (0.25, 0.25), 2)
    # Four times the area per the same physical disc, give or take the rounding
    # each value does on its own (50.27 -> 50, 201.06 -> 201).
    check(abs(finer - 4 * pixels) <= 1, f"at half the pixel size it is four times the count ({finer})")

    voxels = seg.min_size_in_pixels(4.0, (2.0, 0.5, 0.5), 3)
    expected = round((4.0 / 3.0) * np.pi * 8.0 / (2.0 * 0.5 * 0.5))
    check(voxels == expected, f"a 4 µm sphere is {voxels} voxels ({expected} expected)")

    check(seg.min_size_in_pixels(0.0, (0.5, 0.5), 2) is None, "zero means leave min_size alone")
    check(seg.min_size_in_pixels(-1.0, (0.5, 0.5), 2) is None, "so does a negative")
    check(seg.min_size_in_pixels(4.0, (0.0, 0.0), 2) is None, "an uncalibrated layer is left alone")

    # It is the inverse of the diameter the table reports.
    masks = np.zeros((64, 64), dtype=np.int32)
    yy, xx = np.mgrid[0:64, 0:64]
    masks[((yy - 32) ** 2 + (xx - 32) ** 2) <= 16**2] = 1  # radius 16 px = 8 µm at 0.5
    stat = seg.object_table(masks, None, (0.5, 0.5))[0]
    check(
        abs(stat.equivalent_diameter_um - 16.0) < 0.1,
        f"the table calls that disc {stat.equivalent_diameter_um:.1f} µm across",
    )
    threshold = seg.min_size_in_pixels(stat.equivalent_diameter_um, (0.5, 0.5), 2)
    check(
        abs(threshold - stat.n_voxels) / stat.n_voxels < 0.01,
        f"and the same number as a threshold is its own pixel count ({threshold} vs {stat.n_voxels})",
    )

    # End to end: the converted value is what reaches the backend.
    got: list[int] = []

    class _Recorder(seg.Backend):
        name = "minrec"
        install_hint = "-"

        def available(self) -> bool:
            return True

        def models(self):
            return ("m",)

        def model_choices(self):
            return (seg.ModelChoice(label="m", value="m"),)

        def segment(self, image, settings, diameter_px, anisotropy, device, progress=None):
            got.append(int(settings.min_size))
            return np.zeros(np.asarray(image).shape, dtype=np.int32), {}

    seg.register_backend(_Recorder())
    settings = seg.SegmentationSettings(backend="minrec", mode=seg.MODE_STITCH, min_diameter_um=4.0)
    seg.segment_volume(np.zeros((4, 32, 32), np.float32), (2.0, 0.5, 0.5), settings=settings)
    check(got and got[-1] == pixels, f"the backend was given {got[-1] if got else None} px, not the µm")

    untouched = seg.SegmentationSettings(backend="minrec", mode=seg.MODE_STITCH, min_size=42)
    seg.segment_volume(np.zeros((4, 32, 32), np.float32), (2.0, 0.5, 0.5), settings=untouched)
    check(got[-1] == 42, "with no µm threshold the raw min_size passes through untouched")


def test_streams_exist_for_libraries() -> None:
    """pythonw.exe gives a process no stdout; tqdm inside cellpose assumes one."""
    print("stdout under a windowed interpreter")

    import io

    from microscopy_viewer import runtime

    check(sys.stdout is not None, "importing the package leaves a usable stdout")
    check(sys.stderr is not None, "and a usable stderr")

    # Simulate the windowed case: both streams gone, as pythonw.exe leaves them.
    # Nothing may be printed while they are gone — print() silently does nothing
    # when sys.stdout is None, which would hide every check in here — so the
    # findings are collected and reported once the real streams are back.
    saved = (sys.stdout, sys.stderr, sys.__stdout__, sys.__stderr__)
    found: list[tuple[bool, str]] = []
    try:
        sys.stdout = sys.stderr = sys.__stdout__ = sys.__stderr__ = None

        replaced = runtime.ensure_std_streams()
        found.append((set(replaced) == {"stdout", "stderr"}, f"both are replaced ({replaced})"))
        found.append((sys.stdout is not None and sys.stderr is not None, "and are writable again"))

        # The exact call that died inside cellpose's stitch3D.
        try:
            sys.stdout.write("progress")
            sys.stderr.write("progress")
            wrote = True
        except Exception:
            wrote = False
        found.append((wrote, "a library can write to them without an AttributeError"))
        found.append((
            getattr(sys.stdout, runtime.NULL_SINK_FLAG, False),
            "the placeholder is marked, so logging does not attach a handler to it",
        ))
        found.append((sys.__stdout__ is not None, "the pristine aliases are filled in too"))

        # A real stream must never be swapped out from under anything.
        sys.stderr = None
        sys.stdout = io.StringIO()
        mine = sys.stdout
        again = runtime.ensure_std_streams()
        found.append((again == ("stderr",), f"only the missing one is replaced ({again})"))
        found.append((sys.stdout is mine, "an existing stream is left exactly as it was"))
    finally:
        sys.stdout, sys.stderr, sys.__stdout__, sys.__stderr__ = saved

    for ok, message in found:
        check(ok, message)


def test_per_plane_labels_do_not_collide() -> None:
    """Each plane is segmented on its own, so its labels must be made unique.

    The loop lives on :class:`CellposeBackend` rather than in
    :func:`segment_volume`, because it is cellpose specifically that refuses a
    stack in 2D mode — so this drives that method with a stand-in model.
    """
    print("per-plane labels are unique across the stack")

    seen: list[tuple[int, ...]] = []

    class _Model:
        """Stands in for a cellpose model: same two masks on every plane."""

        def eval(self, image, **kwargs):
            array = np.asarray(image)
            seen.append(tuple(array.shape))
            masks = np.zeros(array.shape, dtype=np.int32)
            masks[:3, :3] = 1
            masks[5:8, 5:8] = 2
            return masks, None, None

    backend = seg.CellposeBackend()
    stack = np.zeros((4, 16, 16), dtype=np.float32)
    masks, info = backend._segment_planes(_Model(), stack, {"batch_size": 8})

    check(seen == [(16, 16)] * 4, f"the model is called once per plane, in 2D ({seen})")
    check(info.get("per_plane") is True, "the result says it went plane by plane")
    check(masks.shape == (4, 16, 16), f"the stack comes back whole ({masks.shape})")
    check(int(masks.max()) == 8, f"two objects on each of four planes is eight ({masks.max()})")
    for index in range(4):
        labels = sorted(int(v) for v in set(np.unique(masks[index])) - {0})
        check(len(labels) == 2, f"plane {index} still has its two objects ({labels})")
    spans = {
        int(label): sum(1 for z in range(4) if label in np.unique(masks[z]))
        for label in sorted(set(np.unique(masks)) - {0})
    }
    check(set(spans.values()) == {1}, f"no label spans two planes ({spans})")

    # z_axis and the stitch threshold must not reach a 2D call: that is the whole
    # point — cellpose rejects them when it is doing 2D processing.
    captured: dict = {}

    class _Strict:
        def eval(self, image, **kwargs):
            captured.update(kwargs)
            return np.zeros(np.asarray(image).shape, np.int32), None, None

    backend._segment_planes(
        _Strict(), stack, {"batch_size": 8, "z_axis": 0, "stitch_threshold": 0.25, "anisotropy": 2.0}
    )
    for key in ("z_axis", "stitch_threshold", "anisotropy"):
        check(key not in captured, f"{key} is not passed to a 2D call")
    check(captured.get("do_3D") is False, "and the call is explicitly 2D")


def test_plane_progress_is_reported() -> None:
    """Each plane is counted off as it goes, so a long run is not a frozen window."""
    print("per-plane progress")

    class _Model:
        def eval(self, image, **kwargs):
            return np.zeros(np.asarray(image).shape, np.int32), None, None

    backend = seg.CellposeBackend()
    seen: list[str] = []
    backend._segment_planes(
        _Model(), np.zeros((5, 8, 8), np.float32), {"batch_size": 8}, seen.append
    )
    check(seen == [f"plane {n} of 5" for n in range(1, 6)], f"one line per plane ({seen})")

    # It must survive a caller that does not want progress at all.
    backend._segment_planes(_Model(), np.zeros((3, 8, 8), np.float32), {"batch_size": 8})
    check(True, "and no callback is fine")


def test_stitch_goes_through_the_plane_loop() -> None:
    """Stitching segments plane by plane here, then joins them with stitch_planes.

    Doing it here rather than inside ``model.eval`` is what makes the planes
    countable and what lets the gap be opened past cellpose's fixed one plane.
    """
    print("stitching drives the plane loop")

    calls: list[tuple[int, ...]] = []

    class _Model:
        def eval(self, image, **kwargs):
            array = np.asarray(image)
            calls.append(tuple(array.shape))
            masks = np.zeros(array.shape, np.int32)
            masks[:6, :6] = 1
            return masks, None, None

    backend = seg.CellposeBackend()
    # segment() builds its own model; give it this one instead of loading cpsam.
    backend._model = lambda settings, device: _Model()
    backend._sized_model = lambda settings, device: None

    seen: list[str] = []
    settings = seg.SegmentationSettings(
        mode=seg.MODE_STITCH, stitch_threshold=0.3, stitch_gap_planes=1
    )
    masks, info = backend.segment(
        np.zeros((4, 16, 16), np.float32),
        settings=settings,
        diameter_px=10.0,
        anisotropy=None,
        device=seg.DeviceInfo(),
        progress=seen.append,
    )

    check(calls == [(16, 16)] * 4, f"the model saw one plane at a time ({calls})")
    check(tuple(masks.shape) == (4, 16, 16), f"a stack comes back out ({masks.shape})")
    # The same block in every plane is one object, not four.
    check(int(masks.max()) == 1, f"the four planes stitched into one object ({int(masks.max())})")
    check(info.get("stitched") is True, "the result says it was stitched")
    check(info.get("stitch_gap_planes") == 1, "and reports the gap it used")
    check(any("plane 1 of 4" in line for line in seen), f"planes are counted off ({seen[:2]})")
    check(any("stitching" in line for line in seen), f"and the stitch is announced ({seen[-1:]})")

    # The gap reaches the stitcher: a plane the model missed must not split the
    # object in two when the gap allows reaching over it.
    class _Gappy:
        """Finds the object in every plane but the third."""

        def __init__(self):
            self.plane = 0

        def eval(self, image, **kwargs):
            masks = np.zeros(np.asarray(image).shape, np.int32)
            if self.plane != 2:
                masks[:6, :6] = 1
            self.plane += 1
            return masks, None, None

    for gap, expected, label in ((1, 2, "splits in two"), (2, 1, "stays one object")):
        backend._model = lambda settings, device, _g=_Gappy(): _g
        out, _info = backend.segment(
            np.zeros((5, 16, 16), np.float32),
            settings=seg.SegmentationSettings(
                mode=seg.MODE_STITCH, stitch_threshold=0.3, stitch_gap_planes=gap
            ),
            diameter_px=10.0,
            anisotropy=None,
            device=seg.DeviceInfo(),
        )
        check(
            int(out.max()) == expected,
            f"a missing plane at gap {gap} {label} (got {int(out.max())})",
        )


def test_stitching_reaches_across_a_missing_plane() -> None:
    """The defect this exists for: one missed plane must not split an object.

    Cellpose's own stitch compares plane i with plane i+1 and nothing else, so a
    nucleus absent from a single plane comes back as two objects and no stitch
    threshold can rejoin them — the two halves are never compared.
    """
    print("stitching across a missing plane")

    planes = np.zeros((5, 20, 20), np.int32)
    for index in (0, 1, 3, 4):
        planes[index, 4:12, 4:12] = 1  # each plane numbered from 1, independently

    adjacent = seg.stitch_planes(planes, stitch_threshold=0.25, max_gap=1)
    check(int(adjacent.max()) == 2, f"neighbours-only splits it in two ({int(adjacent.max())})")

    bridged = seg.stitch_planes(planes, stitch_threshold=0.25, max_gap=2)
    check(int(bridged.max()) == 1, f"reaching one plane further rejoins it ({int(bridged.max())})")
    check(
        sorted(np.unique(bridged).tolist()) == [0, 1],
        "and the survivor is numbered from 1 with no gaps",
    )
    # The empty plane stays empty: bridging joins labels, it does not fill in.
    check(not bridged[2].any(), "the plane the object was missing from is still empty")

    # Lowering the threshold cannot help, which is the point: the two halves are
    # never compared at all.
    for threshold in (0.05, 0.25, 0.9):
        split = seg.stitch_planes(planes, stitch_threshold=threshold, max_gap=1)
        check(int(split.max()) == 2, f"threshold {threshold} still leaves it split")


def test_bridging_only_repairs_broken_chains() -> None:
    """A gap must not reroute matches that already worked, or merge neighbours."""
    print("what bridging will not do")

    # Two objects side by side, both present in every plane. A generous gap must
    # leave them as two: neither chain is broken, so neither may bridge.
    planes = np.zeros((4, 20, 40), np.int32)
    for index in range(4):
        planes[index, 4:12, 2:10] = 1
        planes[index, 4:12, 22:30] = 2
    joined = seg.stitch_planes(planes, stitch_threshold=0.25, max_gap=3)
    check(int(joined.max()) == 2, f"two continuous objects stay two ({int(joined.max())})")

    # Two objects at the same place but separated by more depth than the gap
    # allows: they must stay separate.
    tall = np.zeros((8, 20, 20), np.int32)
    tall[0:2, 4:12, 4:12] = 1
    tall[6:8, 4:12, 4:12] = 1
    check(
        int(seg.stitch_planes(tall, 0.25, max_gap=2).max()) == 2,
        "a gap wider than allowed is not bridged",
    )
    check(
        int(seg.stitch_planes(tall, 0.25, max_gap=5).max()) == 1,
        "and is bridged once the gap is wide enough (plane 1 to plane 6 is 5)",
    )

    # Objects that do not overlap in XY are never joined, however close in Z.
    apart = np.zeros((3, 20, 40), np.int32)
    apart[0, 4:12, 2:10] = 1
    apart[2, 4:12, 28:36] = 1
    check(
        int(seg.stitch_planes(apart, 0.25, max_gap=3).max()) == 2,
        "no XY overlap, no join",
    )


def test_the_gap_is_in_micrometres() -> None:
    """Planes are not a fixed distance, so the setting cannot be in planes."""
    print("the stitch gap in micrometres")
    check(seg.stitch_gap_in_planes(1.5, 0.3) == 5, "1.5 um is 5 planes at 0.3 um/plane")
    check(seg.stitch_gap_in_planes(1.5, 2.0) == 1, "and only neighbours at 2 um/plane")
    check(seg.stitch_gap_in_planes(0.0, 0.3) == 1, "zero means neighbours only")
    check(seg.stitch_gap_in_planes(1.5, 0.0) == 1, "an unknown voxel size falls back safely")

    # And it reaches the run, converted against the volume actually segmented.
    class _Recorder(seg.Backend):
        name = "gap-recorder"

        def __init__(self):
            self.seen = None

        def available(self) -> bool:
            return True

        def segment(self, image, settings, diameter_px, anisotropy, device, progress=None):
            self.seen = int(settings.stitch_gap_planes)
            return np.zeros(np.asarray(image).shape, np.int32), {}

    backend = _Recorder()
    seg.register_backend(backend)
    try:
        result = seg.segment_volume(
            np.zeros((8, 16, 16), np.float32),
            voxel_size_um=(0.3, 0.3, 0.3),
            settings=seg.SegmentationSettings(
                backend="gap-recorder", mode=seg.MODE_STITCH, use_gpu=False, stitch_gap_um=1.5
            ),
        )
        check(backend.seen == 5, f"1.5 um became 5 planes for the backend (got {backend.seen})")
        check(result.stitch_gap_planes == 5, "and the result says what it used")

        seg.segment_volume(
            np.zeros((8, 16, 16), np.float32),
            voxel_size_um=(2.0, 0.3, 0.3),
            settings=seg.SegmentationSettings(
                backend="gap-recorder", mode=seg.MODE_STITCH, use_gpu=False, stitch_gap_um=1.5
            ),
        )
        check(backend.seen == 1, f"the same setting is 1 plane on a coarse stack ({backend.seen})")

        # Not a stitching run: the gap is meaningless and must stay at 1.
        flat = seg.segment_volume(
            np.zeros((16, 16), np.float32),
            voxel_size_um=(0.3, 0.3),
            settings=seg.SegmentationSettings(
                backend="gap-recorder", mode=seg.MODE_STITCH, use_gpu=False, stitch_gap_um=1.5
            ),
        )
        check(flat.stitch_gap_planes == 1, "a 2D image has no planes to bridge")
    finally:
        seg._BACKENDS.pop("gap-recorder", None)


def test_maximum_diameter_filter() -> None:
    """The ceiling Cellpose has not got: applied after, measured like the table."""
    print("maximum diameter filter")
    masks = np.zeros((4, 20, 20), dtype=np.int32)
    masks[1:3, 2:6, 2:6] = 1        # 32 voxels -> 3.94 um across
    masks[1:3, 10:18, 10:18] = 2    # 128 voxels -> 6.25 um across
    voxel = (1.0, 1.0, 1.0)

    diameters = {
        stat.label: stat.equivalent_diameter_um
        for stat in seg.object_table(masks, voxel_size_um=voxel)
    }
    check(abs(diameters[1] - 3.94) < 0.01, f"small object is {diameters[1]:.2f} um across")
    check(abs(diameters[2] - 6.25) < 0.01, f"large object is {diameters[2]:.2f} um across")

    # The threshold means what the table column means, so a number between the
    # two reported diameters separates them.
    kept, dropped = seg.filter_by_diameter(masks, voxel, max_diameter_um=5.0)
    check(dropped == 1, "one object over 5 um was dropped")
    check(sorted(np.unique(kept).tolist()) == [0, 1], "one object survives, numbered 1")
    check(kept[2, 4, 4] == 1, "the small object is the survivor")

    # Dropping the first label renumbers the rest rather than leaving a hole, so
    # the highest label and the object count still agree.
    kept, dropped = seg.filter_by_diameter(masks, voxel, min_diameter_um=5.0)
    check(dropped == 1, "one object under 5 um was dropped")
    check(int(kept.max()) == 1, "the survivor was renumbered from 2 to 1")
    check(kept[2, 12, 12] == 1, "and it is the large one")

    # Nothing asked for, nothing touched.
    same, dropped = seg.filter_by_diameter(masks, voxel)
    check(dropped == 0 and same is masks, "no thresholds leaves the array alone")

    # A 2D label map is measured as a disc, not a sphere.
    flat = np.zeros((20, 20), dtype=np.int32)
    flat[2:6, 2:6] = 1              # 16 px -> 4.51 um across as a disc
    kept, dropped = seg.filter_by_diameter(flat, (1.0, 1.0), max_diameter_um=4.0)
    check(dropped == 1, "a 4.5 um disc is dropped by a 4 um ceiling")
    kept, dropped = seg.filter_by_diameter(flat, (1.0, 1.0), max_diameter_um=5.0)
    check(dropped == 0, "and kept by a 5 um one")


def test_maximum_diameter_inside_a_run() -> None:
    """segment_volume applies the ceiling and says how many it removed."""
    print("maximum diameter through a run")
    masks = np.zeros((4, 20, 20), dtype=np.int32)
    masks[1:3, 2:6, 2:6] = 1
    masks[1:3, 10:18, 10:18] = 2

    class _Fixed(seg.Backend):
        name = "fixed-masks"

        def available(self) -> bool:
            return True

        def segment(self, image, settings, diameter_px, anisotropy, device, progress=None):
            return masks.copy(), {}

    seg.register_backend(_Fixed())
    try:
        image = np.zeros((4, 20, 20), dtype=np.float32)
        result = seg.segment_volume(
            image,
            voxel_size_um=(1.0, 1.0, 1.0),
            settings=seg.SegmentationSettings(
                backend="fixed-masks", use_gpu=False, max_diameter_um=5.0
            ),
        )
        check(result.n_objects == 1, f"one object survives the ceiling (got {result.n_objects})")
        check(result.dropped_oversize == 1, "the result records one object dropped")
        check(
            any("over 5 um across" in text.replace("µ", "u") for text in result.warnings),
            f"and says so: {result.warnings}",
        )

        untouched = seg.segment_volume(
            image,
            voxel_size_um=(1.0, 1.0, 1.0),
            settings=seg.SegmentationSettings(backend="fixed-masks", use_gpu=False),
        )
        check(untouched.n_objects == 2, "with no ceiling both objects are kept")
        check(untouched.dropped_oversize == 0, "and nothing is reported as dropped")
    finally:
        seg._BACKENDS.pop("fixed-masks", None)


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
        test_unsupported_models_are_named,
        test_run_through_a_stub_backend,
        test_decimation_inside_a_run,
        test_gpu_request_is_reported,
        test_cellpose_call_shape,
        test_duplicate_openmp_is_allowed,
        test_max_projection_mode,
        test_projected_measurements_are_areas,
        test_per_plane_needs_no_z_axis,
        test_minimum_size_in_microns,
        test_streams_exist_for_libraries,
        test_per_plane_labels_do_not_collide,
        test_plane_progress_is_reported,
        test_stitch_goes_through_the_plane_loop,
        test_stitching_reaches_across_a_missing_plane,
        test_bridging_only_repairs_broken_chains,
        test_the_gap_is_in_micrometres,
        test_maximum_diameter_filter,
        test_maximum_diameter_inside_a_run,
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
