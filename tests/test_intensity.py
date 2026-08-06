"""Checks for the ROI intensity comparison maths. No display required.

Run with::

    python tests/test_intensity.py

Same plain-script convention as tests/test_readers.py.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from microscopy_viewer import intensity as ix  # noqa: E402

_failures: list[str] = []


def check(condition: bool, message: str) -> None:
    if condition:
        print(f"  ok   {message}")
    else:
        print(f"  FAIL {message}")
        _failures.append(message)


def close(actual, expected, tol=1e-6) -> bool:
    return actual is not None and abs(float(actual) - float(expected)) <= tol * max(1.0, abs(expected))


def _rect(top, left, bottom, right) -> np.ndarray:
    return np.array([[top, left], [bottom, left], [bottom, right], [top, right]], dtype=float)


# ---------------------------------------------------------------------------


def test_polygon_mask() -> None:
    print("polygon rasterisation")
    # Pixel centres decide membership, so rows 2..7 and columns 3..9 give
    # exactly 5 x 6 pixels rather than the 6 x 7 a boundary-inclusive fill gives.
    mask = ix.polygon_mask(_rect(2, 3, 7, 9), (20, 20))
    check(mask.dtype == bool, f"mask is boolean ({mask.dtype})")
    check(mask.sum() == 5 * 6, f"a 5x6 rectangle covers {mask.sum()} pixels (expected 30)")
    check(mask[2, 3] and not mask[1, 3], "the rectangle sits at the requested offset")
    check(not mask[7, 3] and not mask[2, 9], "the far edges are excluded, not double counted")

    triangle = np.array([[0, 0], [10, 0], [0, 10]], dtype=float)
    area = ix.polygon_mask(triangle, (12, 12)).sum()
    check(40 <= area <= 55, f"a right triangle of legs 10 covers {area} px (~50)")

    # A ROI reaching past the image edge must clip rather than raise.
    clipped = ix.polygon_mask(_rect(-5, -5, 4, 4), (10, 10))
    check(clipped.sum() == 16, f"an out-of-bounds ROI clips to the image ({clipped.sum()} px)")

    check(ix.polygon_mask(np.array([[1.0, 1.0], [2.0, 2.0]]), (5, 5)).sum() == 0, "a 2-point shape is empty")


def test_ellipse_to_polygon() -> None:
    print("ellipse outline conversion")
    corners = np.array([[0, 0], [20, 0], [20, 40], [0, 40]], dtype=float)
    outline = ix.ellipse_to_polygon(corners, n_points=256)
    check(outline.shape == (256, 2), f"outline shape {outline.shape}")
    centre = outline.mean(axis=0)
    check(close(centre[0], 10, tol=1e-3) and close(centre[1], 20, tol=1e-3), f"centred at {centre}")
    area = ix.polygon_mask(outline, (24, 44)).sum()
    expected = np.pi * 10 * 20
    check(abs(area - expected) / expected < 0.05, f"area {area} within 5% of {expected:.0f}")


def test_world_coordinate_round_trip() -> None:
    print("world coordinates keep ROIs spatially matched")
    shape = _rect(10, 10, 20, 20)
    # A ROI drawn on a 0.5 um/px layer, measured on a 0.25 um/px layer.
    world = ix.world_vertices(shape, scale=(0.5, 0.5), translate=(0.0, 0.0))
    check(np.allclose(world, shape * 0.5), f"layer -> world {world[0]}")

    fine = ix.world_to_data(world, scale=(0.25, 0.25), translate=(0.0, 0.0))
    check(np.allclose(fine, shape * 2.0), "the same region is twice as many pixels on a finer grid")

    shifted = ix.world_to_data(world, scale=(0.5, 0.5), translate=(1.0, 0.0))
    check(np.allclose(shifted[:, 0], shape[:, 0] - 2.0), "a layer offset shifts the ROI back correctly")


def test_extract_plane() -> None:
    print("plane extraction from a 4D stack")
    rng = np.random.default_rng(0)
    data = rng.integers(0, 1000, size=(3, 5, 16, 20)).astype(np.uint16)
    condition = ix.ConditionSpec(
        name="c", layer_name="l", data=data, axes="TZYX",
        scale=(1, 1, 1, 1), translate=(0, 0, 0, 0), current_step=(2, 3, 0, 0), dtype=data.dtype,
    )

    plane, description = ix.extract_plane(condition, "Current slice")
    check(plane.shape == (16, 20), f"slice shape {plane.shape}")
    check(np.array_equal(plane, data[2, 3]), "the slice matches the slider position")
    check("T=2" in description and "Z=3" in description, f"described as {description!r}")

    plane, description = ix.extract_plane(condition, "Maximum projection")
    check(np.array_equal(plane, data[2].max(axis=0)), "max projection reduces over Z at the current T")
    check("max over Z" in description, f"described as {description!r}")

    plane, _ = ix.extract_plane(condition, "Mean projection")
    check(np.allclose(plane, data[2].mean(axis=0)), "mean projection reduces over Z at the current T")

    flat = ix.ConditionSpec(
        name="c", layer_name="l", data=data[0, 0], axes="YX",
        scale=(1, 1), translate=(0, 0), current_step=(0, 0), dtype=data.dtype,
    )
    plane, _ = ix.extract_plane(flat, "Maximum projection")
    check(plane.shape == (16, 20), "a 2D layer is returned unchanged")


def test_projection_helpers() -> None:
    print("projection layer helpers")
    stack = np.zeros((2, 6, 8, 8), dtype=np.uint16)
    volume = ix.ConditionSpec(
        name="c", layer_name="cells.ims :: GFP", data=stack, axes="TZYX",
        scale=(1, 2, 0.5, 0.5), translate=(0, 0, 0, 0), current_step=(0, 0, 0, 0), dtype=stack.dtype,
    )
    check(ix.has_z_axis(volume), "a 4D stack has a projectable Z axis")

    flat = ix.ConditionSpec(
        name="c", layer_name="flat.tif", data=np.zeros((8, 8), np.uint16), axes="YX",
        scale=(1, 1), translate=(0, 0), current_step=(0, 0), dtype=np.uint16,
    )
    check(not ix.has_z_axis(flat), "a 2D layer has none")

    single = ix.ConditionSpec(
        name="c", layer_name="one.ims", data=np.zeros((1, 8, 8), np.uint16), axes="ZYX",
        scale=(1, 1, 1), translate=(0, 0, 0), current_step=(0, 0, 0), dtype=np.uint16,
    )
    check(not ix.has_z_axis(single), "a single-plane Z axis is not worth projecting")

    check(
        ix.projection_layer_name("cells :: GFP", "Maximum projection") == "cells :: GFP [MIP]",
        ix.projection_layer_name("cells :: GFP", "Maximum projection"),
    )
    check(
        ix.projection_layer_name("cells", "Mean projection") == "cells [Mean Z]",
        ix.projection_layer_name("cells", "Mean projection"),
    )
    check(
        ix.projection_layer_name("cells", "Current slice") == "cells [slice]",
        ix.projection_layer_name("cells", "Current slice"),
    )

    # The projected plane is what a MIP layer would hold.
    rng = np.random.default_rng(3)
    stack = rng.integers(0, 5000, size=(2, 6, 8, 8)).astype(np.uint16)
    volume.data = stack
    plane, description = ix.extract_plane(volume, "Maximum projection")
    check(plane.shape == (8, 8), f"projected plane is 2D {plane.shape}")
    check(np.array_equal(plane, stack[0].max(axis=0)), "it is the max over Z at the current T")
    check(plane.dtype == stack.dtype, f"dtype preserved ({plane.dtype})")
    check("max over Z" in description, f"described as {description!r}")


def test_statistics_and_background() -> None:
    print("per-ROI statistics and background correction")
    signal = np.full(100, 500.0)
    signal[:10] = 700.0
    background = np.full(50, 100.0)
    background[:25] = 90.0
    background[25:] = 110.0  # std of exactly 10 (ddof=1 over a balanced split)

    sig = ix.measure_values(signal, roi_name="ROI 1", condition="A", layer_name="L")
    check(sig.n_pixels == 100, f"pixel count {sig.n_pixels}")
    check(close(sig.mean, 520.0), f"mean {sig.mean}")
    check(close(sig.median, 500.0), f"median {sig.median}")
    check(close(sig.integrated, 52000.0), f"integrated {sig.integrated}")
    check(close(sig.maximum, 700.0), f"max {sig.maximum}")

    bg = ix.measure_values(background, roi_name="bg", condition="A", layer_name="L", is_background=True)
    check(close(bg.mean, 100.0), f"background mean {bg.mean}")

    ix.apply_background(sig, bg)
    check(close(sig.corrected_mean, 420.0), f"mean - background = {sig.corrected_mean}")
    check(close(sig.signal_to_background, 5.2), f"signal/background = {sig.signal_to_background}")
    expected_snr = 420.0 / bg.std
    check(close(sig.snr, expected_snr), f"SNR {sig.snr} (expected {expected_snr:.4f})")

    # A flat background has no spread, so SNR is undefined rather than infinite.
    flat_bg = ix.measure_values(np.full(20, 50.0), roi_name="bg", condition="A", layer_name="L")
    other = ix.measure_values(np.full(20, 200.0), roi_name="ROI", condition="A", layer_name="L")
    ix.apply_background(other, flat_bg)
    check(other.snr is None, f"SNR is None for a zero-variance background (got {other.snr})")
    check(close(other.signal_to_background, 4.0), f"ratio still defined ({other.signal_to_background})")


def test_saturation_flagging() -> None:
    print("saturation flagging")
    raw = np.array([100, 200, 65535, 65535, 65535], dtype=np.uint16)
    corrected = raw.astype(float) - 50.0
    stats = ix.measure_values(
        corrected, roi_name="R", condition="A", layer_name="L", offset=50.0,
        saturation_level=ix.default_saturation_level(np.uint16), raw_for_saturation=raw,
    )
    check(stats.saturated_pixels == 3, f"three clipped pixels found ({stats.saturated_pixels})")
    check(close(stats.saturated_percent, 60.0), f"{stats.saturated_percent}% flagged")
    check(close(stats.mean, corrected.mean()), "statistics use the offset-corrected values")

    check(ix.default_saturation_level(np.uint8) == 255, "uint8 saturates at 255")
    check(ix.default_saturation_level(np.float32) is None, "floats have no implicit saturation level")

    # A 12-bit camera never reaches the uint16 maximum, so the level is settable.
    twelve_bit = np.full(10, 4095, dtype=np.uint16)
    stats = ix.measure_values(
        twelve_bit.astype(float), roi_name="R", condition="A", layer_name="L",
        saturation_level=4095, raw_for_saturation=twelve_bit,
    )
    check(stats.saturated_pixels == 10, f"a custom level catches 12-bit clipping ({stats.saturated_pixels})")


def test_separation_metrics() -> None:
    print("non-parametric separation metrics")
    rng = np.random.default_rng(7)

    identical_a = rng.normal(100, 10, 4000)
    identical_b = rng.normal(100, 10, 4000)
    same = ix.mann_whitney_auc(identical_a, identical_b)
    check(abs(same.auc - 0.5) < 0.05, f"identical distributions give AUC ~0.5 (got {same.auc:.3f})")
    check(same.overlap > 0.85, f"and a high overlap ({same.overlap:.3f})")

    separated = ix.mann_whitney_auc(rng.normal(100, 5, 4000), rng.normal(200, 5, 4000))
    check(separated.auc > 0.99, f"well separated distributions give AUC ~1 (got {separated.auc:.3f})")
    check(separated.overlap < 0.1, f"and a low overlap ({separated.overlap:.3f})")
    check(separated.p_value is not None and separated.p_value < 1e-10, f"p = {separated.p_value}")

    # Direction: AUC is P(b > a), so swapping the arguments mirrors it about 0.5.
    reversed_pair = ix.mann_whitney_auc(rng.normal(200, 5, 2000), rng.normal(100, 5, 2000))
    check(reversed_pair.auc < 0.01, f"reversing the order mirrors the AUC ({reversed_pair.auc:.3f})")

    # Heavily skewed, non-Gaussian data must still work.
    skewed = ix.mann_whitney_auc(rng.exponential(1.0, 3000), rng.exponential(3.0, 3000))
    check(skewed.auc > 0.6, f"skewed distributions are still separated ({skewed.auc:.3f})")

    # By hand over the 16 pairs: 12 strictly greater plus 2 ties = 13/16... in
    # full, b=3 scores 2.5, b=4 scores 3.5, b=5 and b=6 score 4 each => 14/16.
    ranks_only = ix._auc_from_ranks(np.array([1.0, 2, 3, 4]), np.array([3.0, 4, 5, 6]))
    check(close(ranks_only, 0.875, tol=1e-3), f"the numpy rank fallback agrees ({ranks_only})")
    scipy_auc = ix.mann_whitney_auc(np.array([1.0, 2, 3, 4]), np.array([3.0, 4, 5, 6])).auc
    check(close(scipy_auc, ranks_only, tol=1e-9), f"scipy and the fallback agree ({scipy_auc})")
    tied = ix._auc_from_ranks(np.array([1.0, 1, 1]), np.array([1.0, 1, 1]))
    check(close(tied, 0.5), f"all-ties give exactly 0.5 ({tied})")

    check(ix.mann_whitney_auc(np.array([]), np.array([1.0])).auc is None, "an empty group yields no AUC")


def test_subsampling_is_deterministic() -> None:
    print("subsampling of very large ROIs")
    values = np.arange(500_000, dtype=float)
    first = ix._subsample(values, limit=1000)
    second = ix._subsample(values, limit=1000)
    check(first.size == 1000, f"capped at the limit ({first.size})")
    check(np.array_equal(first, second), "repeated runs subsample identically")
    check(ix._subsample(np.arange(10.0), limit=1000).size == 10, "small arrays pass through untouched")


def test_shared_bins() -> None:
    print("shared histogram bins")
    edges = ix.shared_bins([np.array([0.0, 5.0]), np.array([-2.0, 9.0])], bins=10)
    check(edges is not None and len(edges) == 11, f"11 edges for 10 bins ({len(edges)})")
    check(close(edges[0], -2.0) and close(edges[-1], 9.0), f"span {edges[0]}..{edges[-1]} covers every sample")
    check(ix.shared_bins([np.array([])]) is None, "no data gives no bins")
    constant = ix.shared_bins([np.array([3.0, 3.0])], bins=4)
    check(constant is not None and constant[-1] > constant[0], "a constant sample still yields a usable range")


def test_run_comparison() -> None:
    print("full comparison run")
    # Two conditions on the same grid: a bright square on a dim background.
    plane_a = np.full((60, 60), 100, dtype=np.uint16)
    plane_a[10:30, 10:30] = 400
    plane_b = np.full((60, 60), 100, dtype=np.uint16)
    plane_b[10:30, 10:30] = 800

    def condition(name, plane):
        return ix.ConditionSpec(
            name=name, layer_name=f"{name}.ims", data=plane, axes="YX",
            scale=(1.0, 1.0), translate=(0.0, 0.0), current_step=(0, 0), dtype=plane.dtype,
        )

    rois = [
        ix.RoiSpec(name="Cell", vertices_world=_rect(10, 10, 30, 30)),
        ix.RoiSpec(name="Background", vertices_world=_rect(40, 40, 55, 55), is_background=True),
    ]
    result = ix.run_comparison(
        [condition("A", plane_a), condition("B", plane_b)], rois, mode="Current slice"
    )

    check(len(result.stats) == 4, f"two ROIs x two conditions = {len(result.stats)} rows")
    check(result.conditions() == ["A", "B"], f"conditions {result.conditions()}")
    check(result.roi_names(include_background=False) == ["Cell"], "the background ROI is excluded on request")

    by_key = {(s.condition, s.roi_name): s for s in result.stats}
    cell_a, cell_b = by_key[("A", "Cell")], by_key[("B", "Cell")]
    check(close(cell_a.mean, 400.0), f"condition A cell mean {cell_a.mean}")
    check(close(cell_b.mean, 800.0), f"condition B cell mean {cell_b.mean}")
    check(close(cell_a.background_mean, 100.0), f"background picked up ({cell_a.background_mean})")
    check(close(cell_a.corrected_mean, 300.0), f"A corrected {cell_a.corrected_mean}")
    check(close(cell_b.corrected_mean, 700.0), f"B corrected {cell_b.corrected_mean}")
    check(close(cell_a.signal_to_background, 4.0), f"A signal/background {cell_a.signal_to_background}")
    check(by_key[("A", "Background")].corrected_mean is None, "the background ROI is not self-corrected")
    check(cell_a.n_pixels == 400, f"a 20x20 ROI holds {cell_a.n_pixels} pixels")

    check(("Cell", "A") in result.samples, "pixel samples kept for plotting")
    check(result.samples[("Cell", "A")].size == 400, "sample size matches the ROI")

    separation = ix.mann_whitney_auc(result.samples[("Cell", "A")], result.samples[("Cell", "B")])
    check(close(separation.auc, 1.0), f"the two conditions separate completely (AUC {separation.auc})")

    # The camera offset must move the means but not the differences.
    offset_result = ix.run_comparison(
        [condition("A", plane_a)], rois, mode="Current slice", offset=50.0
    )
    offset_cell = {(s.condition, s.roi_name): s for s in offset_result.stats}[("A", "Cell")]
    check(close(offset_cell.mean, 350.0), f"offset applied to the mean ({offset_cell.mean})")
    check(close(offset_cell.corrected_mean, 300.0), "background subtraction is unaffected by the offset")
    check(
        close(offset_cell.signal_to_background, 350.0 / 50.0),
        f"but the ratio changes as it should ({offset_cell.signal_to_background})",
    )


def _condition(name, plane):
    return ix.ConditionSpec(
        name=name, layer_name=f"{name}.ims", data=plane, axes="YX",
        scale=(1.0, 1.0), translate=(0.0, 0.0), current_step=(0, 0), dtype=plane.dtype,
    )


def test_per_condition_rois() -> None:
    print("per-condition ROIs")
    # The two samples sit in different corners, as separate acquisitions do.
    plane_a = np.full((80, 80), 100, dtype=np.uint16)
    plane_a[10:30, 10:30] = 500          # sample A, top-left
    plane_b = np.full((80, 80), 200, dtype=np.uint16)
    plane_b[50:70, 50:70] = 1000         # sample B, bottom-right

    rois = [
        ix.RoiSpec(name="ROI 1", vertices_world=_rect(10, 10, 30, 30), condition="A", label="Sample"),
        ix.RoiSpec(name="ROI 2", vertices_world=_rect(50, 50, 70, 70), condition="B", label="Sample"),
        ix.RoiSpec(name="BG A", vertices_world=_rect(60, 5, 75, 20), condition="A",
                   is_background=True, label="Background"),
        ix.RoiSpec(name="BG B", vertices_world=_rect(5, 60, 20, 75), condition="B",
                   is_background=True, label="Background"),
    ]

    check(len(ix.rois_for(rois, "A")) == 2, f"two ROIs apply to A ({len(ix.rois_for(rois, 'A'))})")
    check(ix.background_for(rois, "A").name == "BG A", "A uses its own background")
    check(ix.background_for(rois, "B").name == "BG B", "B uses its own background")

    result = ix.run_comparison([_condition("A", plane_a), _condition("B", plane_b)], rois)
    by_key = {(s.condition, s.roi_name): s for s in result.stats}
    check(("A", "Sample") in by_key and ("B", "Sample") in by_key, f"keys {sorted(by_key)}")

    # Each sample was measured only on its own condition, at its own location.
    check(close(by_key[("A", "Sample")].mean, 500.0), f"A sample mean {by_key[('A', 'Sample')].mean}")
    check(close(by_key[("B", "Sample")].mean, 1000.0), f"B sample mean {by_key[('B', 'Sample')].mean}")
    check(by_key[("A", "Sample")].roi_source == "ROI 1", "the source shape is recorded")
    check(by_key[("B", "Sample")].roi_source == "ROI 2", "for each condition separately")

    # Backgrounds differ per condition, so corrections differ too.
    check(close(by_key[("A", "Sample")].background_mean, 100.0), "A background is 100")
    check(close(by_key[("B", "Sample")].background_mean, 200.0), "B background is 200")
    check(close(by_key[("A", "Sample")].corrected_mean, 400.0), "A corrected = 500 - 100")
    check(close(by_key[("B", "Sample")].corrected_mean, 800.0), "B corrected = 1000 - 200")

    # Both conditions land under one label, so the histogram can overlay them.
    check(("Sample", "A") in result.samples and ("Sample", "B") in result.samples,
          f"samples keyed by label: {sorted(result.samples)}")
    check(result.samples[("Sample", "A")].size == 400, "A sample has 400 px")

    # A ROI belonging to one condition must not be measured on the other.
    a_sources = {s.roi_source for s in result.stats if s.condition == "A"}
    check("ROI 2" not in a_sources and "BG B" not in a_sources, f"A only saw its own ROIs: {a_sources}")


def test_shared_and_per_condition_mix() -> None:
    print("mixing shared and per-condition ROIs")
    plane_a = np.full((60, 60), 100, dtype=np.uint16)
    plane_b = np.full((60, 60), 400, dtype=np.uint16)
    rois = [
        # Shared: no condition, so measured on both.
        ix.RoiSpec(name="Shared", vertices_world=_rect(10, 10, 30, 30)),
        # A shared background, overridden for B only.
        ix.RoiSpec(name="BG shared", vertices_world=_rect(40, 40, 55, 55), is_background=True),
        ix.RoiSpec(name="BG B", vertices_world=_rect(0, 40, 15, 55), condition="B", is_background=True),
    ]
    check(ix.background_for(rois, "A").name == "BG shared", "A falls back to the shared background")
    check(ix.background_for(rois, "B").name == "BG B", "B prefers its own background")

    result = ix.run_comparison([_condition("A", plane_a), _condition("B", plane_b)], rois)
    labels = {(s.condition, s.roi_name) for s in result.stats}
    check(("A", "Shared") in labels and ("B", "Shared") in labels, f"the shared ROI hit both: {sorted(labels)}")
    check(("A", "BG B") not in labels, "B's background was not measured on A")

    # A ROI with no explicit label keeps its own name, as before.
    check(any(s.roi_name == "Shared" for s in result.stats), "unlabelled ROIs keep their name")


def test_normalization() -> None:
    print("normalisation against each condition's background")
    # Same true signal-over-background, different absolute brightness.
    plane_a = np.full((60, 60), 100, dtype=np.uint16)
    plane_a[10:30, 10:30] = 300          # 3x background
    plane_b = np.full((60, 60), 400, dtype=np.uint16)
    plane_b[10:30, 10:30] = 1200         # also 3x background

    rois = [
        ix.RoiSpec(name="Cell", vertices_world=_rect(10, 10, 30, 30)),
        ix.RoiSpec(name="BG", vertices_world=_rect(40, 40, 55, 55), is_background=True),
    ]
    conditions = [_condition("A", plane_a), _condition("B", plane_b)]

    raw = ix.run_comparison(conditions, rois, normalization="None")
    raw_by = {s.condition: s for s in raw.stats if not s.is_background}
    check(raw.samples[("Cell", "A")].mean() != raw.samples[("Cell", "B")].mean(),
          "unnormalised, the two conditions differ")
    check(raw_by["A"].normalized_mean is None, "no normalised mean when normalisation is off")

    divided = ix.run_comparison(conditions, rois, normalization="Divide by background")
    div_by = {s.condition: s for s in divided.stats if not s.is_background}
    check(close(div_by["A"].normalized_mean, 3.0), f"A normalises to {div_by['A'].normalized_mean}")
    check(close(div_by["B"].normalized_mean, 3.0), f"B normalises to {div_by['B'].normalized_mean}")
    check(
        close(divided.samples[("Cell", "A")].mean(), divided.samples[("Cell", "B")].mean()),
        "dividing by background makes the pixel distributions coincide",
    )
    check(close(div_by["A"].mean, 300.0), "the raw mean column is left untouched")
    check(div_by["A"].normalization == "Divide by background", "the choice is recorded on each row")

    subtracted = ix.run_comparison(conditions, rois, normalization="Subtract background")
    sub_by = {s.condition: s for s in subtracted.stats if not s.is_background}
    check(close(sub_by["A"].normalized_mean, 200.0), f"A subtracts to {sub_by['A'].normalized_mean}")
    check(close(sub_by["B"].normalized_mean, 800.0), f"B subtracts to {sub_by['B'].normalized_mean}")
    check(close(subtracted.samples[("Cell", "A")].mean(), 200.0), "pixels shifted by the background")

    # Separation should collapse once the conditions are normalised.
    before = ix.mann_whitney_auc(raw.samples[("Cell", "A")], raw.samples[("Cell", "B")])
    after = ix.mann_whitney_auc(divided.samples[("Cell", "A")], divided.samples[("Cell", "B")])
    check(close(before.auc, 1.0), f"raw AUC separates completely ({before.auc})")
    check(close(after.auc, 0.5), f"normalised AUC shows no difference ({after.auc})")

    check(ix.normalize_values(np.array([5.0]), 0.0, "Divide by background")[0] == 5.0,
          "a zero background leaves values alone instead of dividing")
    check(ix.normalize_values(np.array([5.0]), None, "Divide by background")[0] == 5.0,
          "no background means no normalisation")

    labels = {ix.normalized_axis_label(n) for n in ix.NORMALIZATIONS}
    check(len(labels) == 3, f"each normalisation has its own axis label ({labels})")


def test_series_listing() -> None:
    """Every ROI-on-condition pair is its own curve, whatever the labels are."""
    print("histogram series enumeration")
    plane_a = np.full((60, 60), 100, dtype=np.uint16)
    plane_a[10:30, 10:30] = 500
    plane_b = np.full((60, 60), 100, dtype=np.uint16)
    plane_b[35:55, 35:55] = 900
    conditions = [_condition("A", plane_a), _condition("B", plane_b)]

    # Per-condition ROIs that were never given a shared label — the common case
    # when someone just draws one ROI per sample.
    rois = [
        ix.RoiSpec(name="ROI 1", vertices_world=_rect(10, 10, 30, 30), condition="A"),
        ix.RoiSpec(name="ROI 2", vertices_world=_rect(35, 35, 55, 55), condition="B"),
        ix.RoiSpec(name="BG", vertices_world=_rect(0, 45, 8, 58), is_background=True),
    ]
    result = ix.run_comparison(conditions, rois)

    series = result.series()
    check(len(series) == 2, f"two signal curves regardless of labels ({series})")
    check(("ROI 1", "A") in series and ("ROI 2", "B") in series, f"one per ROI/condition: {series}")
    check(all(key in result.samples for key in series), "each has pixel samples to plot")
    check(all(name != "BG" for name, _cond in series), "the background is left out by default")
    check(len(result.series(include_background=True)) == 4, "backgrounds available on request")

    names = [ix.series_name(roi, cond) for roi, cond in series]
    check(names == ["ROI 1 — A", "ROI 2 — B"], f"display names {names}")

    # Comparing two ROIs drawn on different conditions must work directly.
    first, second = (result.samples[key] for key in series)
    separation = ix.mann_whitney_auc(first, second)
    check(close(separation.auc, 1.0), f"the two ROIs are separable ({separation.auc})")

    # A shared ROI still yields one curve per condition.
    shared = ix.run_comparison(
        conditions,
        [ix.RoiSpec(name="Both", vertices_world=_rect(10, 10, 30, 30))],
    )
    check(
        shared.series() == [("Both", "A"), ("Both", "B")],
        f"a shared ROI gives one curve per condition ({shared.series()})",
    )


def test_run_comparison_warnings() -> None:
    print("warnings")
    plane = np.full((30, 30), 65535, dtype=np.uint16)
    spec = ix.ConditionSpec(
        name="A", layer_name="a.ims", data=plane, axes="YX",
        scale=(1.0, 1.0), translate=(0.0, 0.0), current_step=(0, 0), dtype=plane.dtype,
    )
    result = ix.run_comparison([spec], [ix.RoiSpec(name="R", vertices_world=_rect(5, 5, 15, 15))])
    check(any("saturated" in w for w in result.warnings), f"saturation warned: {result.warnings}")
    check(
        any("no background ROI applies to it" in w for w in result.warnings),
        f"missing background warned per condition: {result.warnings}",
    )

    # Asking for normalisation without any background must say so rather than
    # silently returning raw values.
    unnormalised = ix.run_comparison(
        [spec], [ix.RoiSpec(name="R", vertices_world=_rect(5, 5, 15, 15))],
        normalization="Divide by background",
    )
    check(
        any("no background ROI was found" in w for w in unnormalised.warnings),
        f"impossible normalisation warned: {unnormalised.warnings}",
    )

    empty = ix.run_comparison([], [])
    check(len(empty.stats) == 0 and empty.warnings, "an empty run warns instead of raising")


def test_stats_export() -> None:
    print("statistics export")
    stats = [
        ix.measure_values(np.array([1.0, 2, 3]), roi_name="R", condition="A", layer_name="L"),
        ix.measure_values(np.array([4.0, 5, 6]), roi_name="R", condition="B", layer_name="L"),
    ]
    frame = ix.stats_dataframe(stats)
    check(len(frame) == 2, f"two rows ({len(frame)})")
    check("Condition" in frame.columns and "Mean" in frame.columns, f"labelled columns: {list(frame.columns)[:4]}")

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "stats.csv"
        frame.to_csv(path, index=False)
        text = path.read_text(encoding="utf-8")
        check(path.stat().st_size > 0, "CSV written")
        check("Condition" in text.splitlines()[0], "header row present")
        check(len(text.strip().splitlines()) == 3, "header plus two data rows")

    check(len(ix.stats_dataframe([])) == 0, "an empty table still produces a frame")


def main() -> int:
    for test in (
        test_polygon_mask,
        test_ellipse_to_polygon,
        test_world_coordinate_round_trip,
        test_extract_plane,
        test_projection_helpers,
        test_statistics_and_background,
        test_saturation_flagging,
        test_separation_metrics,
        test_subsampling_is_deterministic,
        test_shared_bins,
        test_run_comparison,
        test_per_condition_rois,
        test_shared_and_per_condition_mix,
        test_normalization,
        test_series_listing,
        test_run_comparison_warnings,
        test_stats_export,
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
