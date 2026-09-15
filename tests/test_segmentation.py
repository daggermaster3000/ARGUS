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

    def segment(
        self, image, settings, diameter_px, anisotropy, device, progress=None, channel_axis=None
    ):
        image = np.asarray(image)
        self.calls.append(
            {
                "shape": tuple(image.shape),
                "diameter_px": diameter_px,
                "anisotropy": anisotropy,
                "settings": settings,
                "device": device,
                "channel_axis": channel_axis,
                "image": image,
            }
        )
        if channel_axis is not None:
            # Real backends return labels without the channel axis.
            image = np.take(image, 0, axis=channel_axis)
        masks = np.zeros(image.shape, dtype=np.int32)
        masks[..., :4, :4] = 1  # one object, so the count is not trivially zero
        return masks, {"batch_size": 8}


def _disc(image, labels, centre, radius, label):
    """Draw a filled disc — a convex shape, and what a false positive looks like."""
    grids = np.ogrid[tuple(slice(0, n) for n in image.shape)]
    inside = sum((grid - c) ** 2 for grid, c in zip(grids, centre)) <= radius**2
    labels[inside] = label
    image[inside] = 100.0 * label
    return inside


def _has_skimage() -> bool:
    return seg.shape_error() is None


def _close(first: float, second: float, tolerance: float = 1e-9) -> bool:
    return abs(float(first) - float(second)) <= tolerance


def test_shape_descriptors() -> None:
    print("shape descriptors from scikit-image")

    if not _has_skimage():
        print("  skip scikit-image is not installed")
        return

    labels = np.zeros((80, 80), dtype=np.int32)
    image = np.zeros((80, 80), dtype=np.float32)
    _disc(image, labels, (20, 20), 12, 1)  # a clean disc
    labels[50:75, 50:75] = 2  # a square with a bite out of it
    labels[62:75, 62:75] = 0

    shapes = seg.shape_properties(labels)
    check(sorted(shapes) == [1, 2], f"one entry per label (got {sorted(shapes)})")
    check(
        set(shapes[1]) == set(seg.SHAPE_COLUMNS),
        f"every descriptor is reported: {sorted(shapes[1])}",
    )
    check(
        shapes[1]["solidity"] > shapes[2]["solidity"],
        f"the disc is more solid than the dented square "
        f"({shapes[1]['solidity']:.3f} vs {shapes[2]['solidity']:.3f})",
    )
    check(
        shapes[1]["circularity"] > shapes[2]["circularity"],
        f"and rounder ({shapes[1]['circularity']:.3f} vs {shapes[2]['circularity']:.3f})",
    )
    check(
        shapes[1]["circularity"] <= 1.0,
        f"circularity is capped at 1, not overshot by the digitised outline "
        f"({shapes[1]['circularity']:.3f})",
    )
    check(
        shapes[1]["eccentricity"] < 0.3,
        f"a disc has almost no eccentricity ({shapes[1]['eccentricity']:.3f})",
    )

    # A plate mask is (1, y, x). Measured as a volume every object would be one
    # voxel thick and its convex hull degenerate, so the axis has to come off.
    stacked = seg.shape_properties(labels[np.newaxis])
    check(
        all(
            _close(stacked[label][name], shapes[label][name])
            for label in shapes
            for name in ("solidity", "circularity")
        ),
        "a single-plane stack measures the same as the 2D image it is",
    )


def test_round_objects_are_filtered() -> None:
    print("filtering out the round false positives")

    if not _has_skimage():
        print("  skip scikit-image is not installed")
        return

    labels = np.zeros((80, 80), dtype=np.int32)
    image = np.zeros((80, 80), dtype=np.float32)
    _disc(image, labels, (20, 20), 12, 1)
    labels[50:75, 50:75] = 2
    labels[62:75, 62:75] = 0
    shapes = seg.shape_properties(labels)
    cut = (shapes[1]["solidity"] + shapes[2]["solidity"]) / 2

    dropped = seg.round_labels(shapes, cut)
    check(dropped == [1], f"the disc is above the cut and the dented square is not ({dropped})")
    check(
        seg.round_labels(shapes, 1.0) == [],
        "a cut of 1.0 drops nothing, so the shapes can be looked at before choosing one",
    )

    filtered, dropped, measured = seg.filter_round_objects(labels, cut)
    check(int(filtered.max()) == 2, "the surviving object keeps its own label, unrenumbered")
    check(not (filtered == 1).any(), "the dropped object is gone from the mask")
    check((filtered == 2).sum() == (labels == 2).sum(), "the survivor is untouched")
    check(len(measured) == 2, "the measurements of everything come back, dropped included")
    check(seg.count_labels(filtered) == 1, "one object is left")
    check(
        seg.count_labels(labels) == 2 and seg.count_labels(np.zeros((4, 4), np.int32)) == 0,
        "counting labels does not go by the highest id",
    )

    # Cellpose leaves holes in its numbering; the count has to survive them.
    holed = np.zeros((10, 10), dtype=np.int32)
    holed[0:3, 0:3] = 7
    check(seg.count_labels(holed) == 1, "one object numbered 7 counts as one, not seven")


def test_the_filter_runs_through_segment_volume() -> None:
    print("the solidity setting reaches the run")

    if not _has_skimage():
        print("  skip scikit-image is not installed")
        return

    class _TwoShapes(_StubBackend):
        """Returns one disc and one ragged object, whatever it is handed."""

        def segment(self, image, settings, diameter_px, anisotropy, device, progress=None,
                    channel_axis=None):
            shape = np.asarray(image).shape
            masks = np.zeros(shape, dtype=np.int32)
            plane = np.zeros(shape[-2:], dtype=np.int32)
            _disc(np.zeros(shape[-2:], dtype=np.float32), plane, (20, 20), 12, 1)
            plane[50:75, 50:75] = 2
            plane[62:75, 62:75] = 0
            masks[...] = plane
            return masks, {}

    seg.register_backend(_TwoShapes())
    image = np.zeros((80, 80), dtype=np.float32)

    unfiltered = seg.segment_volume(
        image, (1.0, 1.0), settings=seg.SegmentationSettings(backend="stub")
    )
    check(unfiltered.n_objects == 2, f"both objects survive with the filter off ({unfiltered.n_objects})")
    check(not unfiltered.shapes, "and nothing is measured, so nothing is paid for")

    shapes = seg.shape_properties(unfiltered.masks)
    cut = (shapes[1]["solidity"] + shapes[2]["solidity"]) / 2
    result = seg.segment_volume(
        image,
        (1.0, 1.0),
        settings=seg.SegmentationSettings(backend="stub", max_solidity=cut),
    )
    check(result.n_objects == 1, f"the round one is gone ({result.n_objects} left)")
    check(result.dropped_labels == [1], f"and is reported as dropped ({result.dropped_labels})")
    check(result.n_dropped == 1, "the count is on the result")
    check(
        any("too round" in warning for warning in result.warnings),
        f"the run says so rather than quietly losing objects: {result.warnings}",
    )

    stats = seg.object_table(result.masks, image, (1.0, 1.0), shapes=result.shapes)
    check(len(stats) == 1, "the object table holds the survivor only")
    check(
        _close(stats[0].solidity, shapes[2]["solidity"]),
        "and carries its solidity, measured once and reused",
    )
    row = stats[0].as_row()
    check(
        all(name in row for name in seg.SHAPE_COLUMNS),
        f"the exported row carries every shape column: {sorted(row)}",
    )
    check(
        "median_solidity" in seg.count_summary(stats),
        "the summary reports the median solidity when it was measured",
    )

    plain = seg.object_table(result.masks, image, (1.0, 1.0))
    check(
        not np.isfinite(plain[0].solidity),
        "without measurements the shape columns are blank rather than a made-up 0",
    )

    # A run on a single-plane stack, which is what plate data is.
    stacked = seg.segment_volume(
        image[np.newaxis],
        (1.0, 1.0, 1.0),
        settings=seg.SegmentationSettings(backend="stub", max_solidity=cut),
    )
    check(
        stacked.masks.shape == (1, 80, 80) and stacked.n_objects == 1,
        f"a (1, y, x) stack filters the same and keeps its axis ({stacked.masks.shape})",
    )
    seg.register_backend(_StubBackend())


def test_intensities_belong_to_the_objects() -> None:
    print("intensities are measured on the object's own voxels")

    masks = np.zeros((8, 8), dtype=np.int32)
    masks[1:3, 1:3] = 1
    masks[5:7, 5:7] = 2
    signal = np.zeros((8, 8), dtype=float)
    signal[1:3, 1:3] = 10.0
    signal[5:7, 5:7] = 100.0

    stats = {stat.label: stat for stat in seg.object_table(masks, signal, (1.0, 1.0))}
    check(
        stats[1].mean == 10.0 and stats[2].mean == 100.0,
        f"each object gets its own voxels and nobody else's ({stats[1].mean}, {stats[2].mean})",
    )
    check(
        stats[2].integrated == 400.0,
        f"the integral is the sum over those voxels ({stats[2].integrated})",
    )

    # The measurement is one flat index into both arrays, which on a mismatch
    # returns numbers for the wrong voxels without complaining. It must not.
    try:
        seg.object_table(masks, np.zeros((4, 4)), (1.0, 1.0))
        check(False, "a signal on a different grid is refused")
    except ValueError as exc:
        check(
            "grid the objects are on" in str(exc),
            f"a signal on a different grid is refused rather than mis-indexed: {exc}",
        )
    try:
        seg.object_table(masks, signal, (1.0, 1.0), extra_signals={"GFP": np.zeros((9, 9))})
        check(False, "and so is an extra channel on a different grid")
    except ValueError as exc:
        check("'GFP'" in str(exc), f"and the refusal names the channel: {exc}")


def test_every_channel_can_be_measured() -> None:
    print("one segmentation, every channel")

    masks = np.zeros((6, 6), dtype=np.int32)
    masks[1:3, 1:3] = 1
    masks[4:6, 4:6] = 2
    dapi = np.full((6, 6), 5.0)
    green = np.zeros((6, 6))
    green[1:3, 1:3] = 20.0
    green[4:6, 4:6] = 40.0

    stats = seg.object_table(masks, dapi, (1.0, 1.0), extra_signals={"Green488": green})
    first = stats[0]
    check(first.mean == 5.0, "the unqualified columns are still the measured channel")
    check(
        first.extra["Mean intensity (Green488)"] == 20.0,
        f"and the extra channel gets columns of its own ({first.extra})",
    )
    check(
        stats[1].extra["Mean intensity (Green488)"] == 40.0,
        "measured per object, not per image",
    )
    check(
        stats[1].extra["Integrated intensity (Green488)"] == 160.0,
        f"with the whole set of statistics ({stats[1].extra})",
    )

    frame = seg.object_dataframe(stats)
    check(
        "Mean intensity (Green488)" in frame.columns,
        f"the column reaches the table: {list(frame.columns)[-5:]}",
    )
    check(
        list(frame.columns).index("Mean intensity")
        < list(frame.columns).index("Mean intensity (Green488)"),
        "the segmented channel's columns stay where they were, so old readers still work",
    )
    check(
        len(frame.columns) == len(seg.OBJECT_COLUMNS) + 5,
        f"five columns per extra channel and no more ({len(frame.columns)})",
    )

    plain = seg.object_dataframe(seg.object_table(masks, dapi, (1.0, 1.0)))
    check(
        list(plain.columns) == [seg.OBJECT_HEADERS.get(c, c) for c in seg.OBJECT_COLUMNS],
        "a run that did not ask for them writes exactly the table it used to",
    )


def test_median_prefilter() -> None:
    print("median filter before segmenting")

    plane = np.zeros((64, 64), dtype=np.uint16)
    plane[20:44, 20:44] = 1000  # an object with a sharp edge
    noisy = plane.copy()
    noisy[5, 5] = 60000  # a hot pixel in the background
    noisy[30, 30] = 0  # a dead pixel inside the object

    filtered = seg.denoise_median(noisy, 1)
    check(int(filtered[5, 5]) == 0, "a hot pixel in the background is removed")
    check(int(filtered[30, 30]) == 1000, "a dead pixel inside an object is filled in")
    # A straight edge, which a median leaves exactly where it was. (Corners are a
    # different matter: a 3x3 median rounds them off, which is the filter working.)
    step = np.zeros((64, 64), dtype=np.uint16)
    step[:, 32:] = 1000
    edge = seg.denoise_median(step, 1)
    check(
        int(edge[:, :32].max()) == 0 and int(edge[:, 32:].min()) == 1000,
        "a straight edge does not move — the point of a median rather than a blur",
    )
    check(seg.denoise_median(noisy, 0) is noisy, "radius 0 is a no-op, not a copy")

    volume = np.zeros((5, 64, 64), dtype=np.uint16)
    volume[2, 10, 10] = 60000
    smoothed = seg.denoise_median(volume, 1)
    check(smoothed.shape == volume.shape, "a stack keeps its shape")
    check(int(smoothed.max()) == 0, "the hot pixel goes")
    layered = np.zeros((3, 8, 8), dtype=np.uint16)
    layered[1] = 500
    check(
        int(seg.denoise_median(layered, 1)[1].max()) == 500
        and int(seg.denoise_median(layered, 1)[0].max()) == 0,
        "Z is not filtered, so a bright plane does not bleed into its neighbours",
    )


def test_the_median_setting_reaches_the_run() -> None:
    print("the median setting reaches the run")

    stub = _StubBackend()
    seg.register_backend(stub)

    image = np.zeros((48, 48), dtype=np.float32)
    image[10, 10] = 9999.0

    plain = seg.segment_volume(image, (1.0, 1.0), settings=seg.SegmentationSettings(backend="stub"))
    check(
        float(np.asarray(stub.calls[-1]["image"]).max()) == 9999.0,
        "with the filter off the backend gets the pixels as they were",
    )
    check(plain.median_radius_px == 0, "and the result says no filtering was done")

    result = seg.segment_volume(
        image, (1.0, 1.0), settings=seg.SegmentationSettings(backend="stub", median_radius_px=1)
    )
    check(
        float(np.asarray(stub.calls[-1]["image"]).max()) == 0.0,
        "with it on the backend gets the filtered channel",
    )
    check(result.median_radius_px == 1, "the radius is reported on the result")
    check(
        result.masks.shape == image.shape,
        "the labels still come back on the grid that went in",
    )

    # Both stains are filtered, or the two channels stop matching.
    nuclei = np.zeros((48, 48), dtype=np.float32)
    nuclei[30, 30] = 9999.0
    seg.segment_volume(
        image,
        (1.0, 1.0),
        settings=seg.SegmentationSettings(backend="stub", median_radius_px=1),
        nuclei=nuclei,
    )
    payload = np.asarray(stub.calls[-1]["image"])
    check(
        stub.calls[-1]["channel_axis"] == payload.ndim - 1 and float(payload.max()) == 0.0,
        "a nuclear channel is filtered alongside the segmented one",
    )
    seg.register_backend(_StubBackend())


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


def test_broken_install_is_not_reported_as_missing() -> None:
    """"Installed but will not import" and "not installed" need different advice."""
    print("a backend that is installed but fails to import")

    class _Broken(_StubBackend):
        name = "broken"
        install_hint = "pip install broken"

        def available(self) -> bool:
            return False

        @staticmethod
        def import_error():
            return OSError("[WinError 1114] ... Error loading c10.dll")

    seg.register_backend(_Broken())
    message = seg.missing_backend_message("broken")
    check("could not be imported" in message, f"the real failure is named ({message[:60]}…)")
    check("c10.dll" in message, "including the error itself, so it can be searched for")
    check("pip install broken" not in message, "and it does not tell you to install what you have")

    class _Absent(_Broken):
        name = "absent"

        @staticmethod
        def import_error():
            return ImportError("No module named 'absent'")

    seg.register_backend(_Absent())
    check(
        seg.missing_backend_message("absent") == "pip install broken",
        "a genuinely missing package still gets the install hint",
    )


def test_torch_dll_preload() -> None:
    """Torch's DLLs are claimed before Qt can take the process down the wrong path."""
    print("the torch DLL preload")
    from microscopy_viewer import runtime

    result = runtime.preload_torch_libraries()
    check(result is None or result.name == "lib", f"returns torch's lib directory or None ({result})")
    if sys.platform == "win32" and _has_cellpose():
        check(result is not None and result.exists(), "on Windows with torch present, it ran")
    # Idempotent: the viewer imports the package more than once in a session.
    check(runtime.preload_torch_libraries() == result, "calling it twice is harmless")


def test_two_channel_run() -> None:
    """A nuclear stain rides alongside the segmented channel, cell channel first."""
    print("segmenting a cell channel together with a nuclear one")

    stub = _StubBackend()
    seg.register_backend(stub)

    rng = np.random.default_rng(7)
    cyto = rng.random((4, 32, 32)).astype(np.float32)
    nuclei = rng.random((4, 32, 32)).astype(np.float32)
    settings = seg.SegmentationSettings(backend="stub", diameter_um=5.0)

    result = seg.segment_volume(cyto, (2.0, 0.5, 0.5), settings=settings, nuclei=nuclei)
    call = stub.calls[-1]
    check(call["channel_axis"] == 3, f"the channel axis is last and named ({call['channel_axis']})")
    check(call["shape"] == (4, 32, 32, 2), f"both stains reach the backend ({call['shape']})")
    check(
        np.array_equal(call["image"][..., 0], cyto) and np.array_equal(call["image"][..., 1], nuclei),
        "the segmented channel comes first, the nuclei second",
    )
    check(result.masks.shape == cyto.shape, "labels come back without the channel axis")
    check(result.used_nuclear_channel, "the result records that two channels were used")

    # Single channel is unchanged: no channel axis, purely spatial.
    single = seg.segment_volume(cyto, (2.0, 0.5, 0.5), settings=settings)
    check(stub.calls[-1]["channel_axis"] is None, "a one-channel run passes no channel axis")
    check(stub.calls[-1]["shape"] == (4, 32, 32), "and hands over the spatial array alone")
    check(not single.used_nuclear_channel, "and says so")

    # A mismatched nuclear channel is reported rather than crashing the run.
    odd = seg.segment_volume(cyto, (2.0, 0.5, 0.5), settings=settings, nuclei=nuclei[:, :16, :16])
    check(stub.calls[-1]["channel_axis"] is None, "a mismatched nuclear channel is dropped")
    check(
        any("same grid" in warning for warning in odd.warnings),
        f"and the reason is reported ({odd.warnings})",
    )

    # The pairing survives decimation, which happens before the two are stacked.
    big = seg.SegmentationSettings(backend="stub", diameter_um=5.0, max_voxels=1_000)
    seg.segment_volume(cyto, (2.0, 0.5, 0.5), settings=big, nuclei=nuclei)
    call = stub.calls[-1]
    check(
        call["shape"][-1] == 2 and call["shape"][:3] == stub.calls[-1]["image"].shape[:3],
        f"both channels are decimated together ({call['shape']})",
    )
    check(call["shape"][1] < 32, f"and really were decimated ({call['shape']})")


def test_single_plane_stack_is_segmented_as_2d() -> None:
    """A ``(1, y, x)`` layer is a 2D image, and must not go down the stitch path.

    Plate and slide-scanner stores keep their Z axis even when it is one plane
    deep. Handing that to Cellpose as a volume makes it stitch a degenerate
    z-stack, which on real plate data found 16 objects where the same crop
    segmented flat found 679.
    """
    print("a single-plane stack is segmented as a 2D image")

    stub = _StubBackend()
    seg.register_backend(stub)

    image = np.random.default_rng(3).random((1, 40, 40)).astype(np.float32)
    settings = seg.SegmentationSettings(backend="stub", diameter_um=5.0)
    result = seg.segment_volume(image, (1.0, 0.5, 0.5), settings=settings)

    call = stub.calls[-1]
    check(len(call["shape"]) == 2, f"the backend is handed a 2D image ({call['shape']})")
    check(call["diameter_px"] == 10.0, "the XY voxel size still drives the diameter")
    check(result.masks.shape == image.shape, "the labels come back on the (1, y, x) grid")
    check(result.voxel_size_um == (1.0, 0.5, 0.5), "the voxel size is reported in full")


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
        test_intensities_belong_to_the_objects,
        test_every_channel_can_be_measured,
        test_median_prefilter,
        test_the_median_setting_reaches_the_run,
        test_shape_descriptors,
        test_round_objects_are_filtered,
        test_the_filter_runs_through_segment_volume,
        test_decimation_plan,
        test_object_table,
        test_table_edge_cases,
        test_export_reuses_the_workbook_writer,
        test_backend_reporting,
        test_model_filtering,
        test_model_choices_are_labelled,
        test_run_through_a_stub_backend,
        test_broken_install_is_not_reported_as_missing,
        test_torch_dll_preload,
        test_two_channel_run,
        test_single_plane_stack_is_segmented_as_2d,
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
