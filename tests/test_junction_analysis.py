"""Checks for counting and scoring the cells in the junction region.

No GUI and no Cellpose: the cells are drawn, so which are positive in which
channel is known, and the segmentation is handed in as a function.

Run with::

    python tests/test_junction_analysis.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from microscopy_viewer import fusion as fu  # noqa: E402
from microscopy_viewer import junction_analysis as ja  # noqa: E402

_failures: list[str] = []


def check(condition: bool, message: str) -> None:
    if condition:
        print(f"  ok   {message}")
    else:
        print(f"  FAIL {message}")
        _failures.append(message)


def _field(seed: int = 0):
    """A grid of 8 × 12 nuclei, 5 px across, on a 1 µm grid.

    Returns the labels, the channel planes and the labels made bright in each
    channel: GFP in every seventh cell, red in every eleventh.
    """
    rng = np.random.default_rng(seed)
    shape = (120, 180)
    labels = np.zeros(shape, dtype=np.int32)
    number = 0
    for row in range(8):
        for col in range(12):
            number += 1
            y, x = 10 + row * 14, 10 + col * 14
            labels[y:y + 5, x:x + 5] = number
    dapi = np.where(labels > 0, 3000.0, 300.0) + rng.normal(0, 20, shape)
    gfp = 200.0 + rng.normal(0, 10, shape)
    red = 150.0 + rng.normal(0, 10, shape)
    green_cells = [n for n in range(1, number + 1) if n % 7 == 0]
    red_cells = [n for n in range(1, number + 1) if n % 11 == 0]
    gfp[np.isin(labels, green_cells)] += 900.0
    red[np.isin(labels, red_cells)] += 700.0
    images = {"AF555": red, "AF488": gfp, "AF405": dapi}
    return labels, images, green_cells, red_cells


def test_statistics_per_label() -> None:
    print("per-cell statistics")
    labels = np.zeros((4, 6), dtype=np.int32)
    labels[0, :4] = 1
    labels[2, :5] = 3  # the numbering has a hole
    image = np.zeros((4, 6))
    image[0, :4] = (1, 2, 3, 10)
    image[2, :5] = (5, 5, 5, 5, 5)
    stats = ja.label_stats(labels, image, [1, 2, 3])
    check(list(stats["n"]) == [4, 0, 5], "pixels are counted per label")
    check(stats["mean"][0] == 4.0 and stats["mean"][2] == 5.0, "the mean is the mean")
    check(stats["max"][0] == 10.0 and stats["integrated"][0] == 16.0, "with the maximum and the sum")
    check(abs(stats["SD"][0] - np.std([1, 2, 3, 10])) < 1e-9, "and the standard deviation")
    check(stats["median"][0] in (2.0, 3.0) and stats["P95"][0] == 10.0, "quantiles come from the label's own pixels")
    check(stats["CV"][2] == 0.0, "a flat cell has no variation")
    check(np.isnan(stats["mean"][1]), "a label that is not there is a gap, not a zero")


def test_thresholds() -> None:
    print("threshold rules")
    rng = np.random.default_rng(2)
    values = np.concatenate((rng.normal(100, 5, 200), [400, 420, 450]))
    centre, spread = ja.robust_centre_and_spread(values)
    check(abs(centre - 100) < 2 and 3 < spread < 7, f"three bright cells do not move the centre or the spread ({centre:.0f}, {spread:.1f})")

    cells, rule = ja.threshold_for("g", values, (50.0, 4.0), ja.CountOptions(k=3))
    check(rule == ja.METHOD_CELLS and 110 < cells < 125, f"above the cells: median + 3 SD ({cells:.0f})")
    background, rule = ja.threshold_for("g", values, (50.0, 4.0), ja.CountOptions(method=ja.METHOD_BACKGROUND, k=3))
    check(rule == ja.METHOD_BACKGROUND and background == 62.0, "above the background: its median + 3 of its SDs")
    otsu, rule = ja.threshold_for("g", values, (50.0, 4.0), ja.CountOptions(method=ja.METHOD_OTSU))
    check(rule == ja.METHOD_OTSU and 110 < otsu < 400, f"Otsu falls between the two groups ({otsu:.0f})")
    fixed, rule = ja.threshold_for("g", values, (50.0, 4.0), ja.CountOptions(method=ja.METHOD_FIXED, fixed={"g": 250.0}))
    check((fixed, rule) == (250.0, ja.METHOD_FIXED), "a fixed value is used as given")
    _value, rule = ja.threshold_for("other", values, (50.0, 4.0), ja.CountOptions(method=ja.METHOD_FIXED, fixed={"g": 250.0}))
    check(rule == ja.METHOD_CELLS, "a channel with no fixed value falls back, and says so")
    _value, rule = ja.threshold_for("g", values, (float("nan"), float("nan")), ja.CountOptions(method=ja.METHOD_BACKGROUND))
    check(rule == ja.METHOD_CELLS, "so does a background that could not be measured")


def test_cells_are_scored_per_channel() -> None:
    print("scoring the cells")
    labels, images, green_cells, red_cells = _field()
    band = np.zeros(labels.shape, dtype=bool)
    band[:, :90] = True  # the left half: columns of cells 1-6
    inside = sorted(int(n) for n in np.unique(labels[:, :84]) if n)
    table, counts = ja.measure_cells(labels, labels, images, (1.0, 1.0), band, nuclear="AF405", origin=(100, 200))

    check(sorted(table["Label"]) == inside, f"only cells whose centroid is in the region are counted ({len(table)})")
    by_channel = {count.channel: count for count in counts}
    green_in = sorted(set(green_cells) & set(inside))
    red_in = sorted(set(red_cells) & set(inside))
    check(sorted(table.loc[table["AF488 positive"], "Label"]) == green_in, f"the GFP cells are the positive ones ({green_in})")
    check(sorted(table.loc[table["AF555 positive"], "Label"]) == red_in, f"and the red ones in their channel ({red_in})")
    check(by_channel["AF488"].n_positive == len(green_in) and by_channel["AF488"].n_cells == len(inside),
          "the channel's count agrees with the table")
    check(abs(by_channel["AF488"].percent - 100 * len(green_in) / len(inside)) < 1e-9, "as a share of the cells in the region")
    check(by_channel["AF405"].reference and "AF405 positive" not in table, "the nuclear stain is reported but not scored")
    check(200 < by_channel["AF488"].threshold < 1100, f"the threshold sits between negative and positive ({by_channel['AF488'].threshold:.0f})")
    check(abs(by_channel["AF488"].background - 200) < 5, "the background is the tissue between the cells")

    both = sorted(set(green_in) & set(red_in))
    named = table.set_index("Label")["Positive for"]
    check(all(named[n] == "AF555 + AF488" for n in both) and named[inside[0]] == "none",
          "each cell is named by the channels it is positive for")
    check(table["Nucleus area (µm²)"].eq(25.0).all(), "sizes are the nucleus's own")
    check(table["Centroid Y (µm)"].min() >= 100 and table["Centroid X (µm)"].min() >= 200,
          "positions are in the whole image, not in the crop")
    columns = {f"AF488 {metric}" for metric in ja.CELL_METRICS} | {"AF488 SDs over background", "AF488 fold over background"}
    check(columns <= set(table.columns), "every metric is there for every channel")
    bright = table.set_index("Label").loc[green_in[0]]
    check(bright["AF488 SDs over background"] > 20 and bright["AF488 fold over background"] > 4,
          "a positive cell is many SDs, and several fold, over background")


def test_the_threshold_can_come_from_the_whole_sample() -> None:
    print("threshold from the region, or from the whole sample")
    labels = np.zeros((80, 200), dtype=np.int32)
    green = np.full(labels.shape, 20.0)
    band = np.zeros(labels.shape, dtype=bool)
    band[:, :60] = True
    number = 0
    for row in range(4):
        for col in range(10):
            number += 1
            y, x = 8 + 20 * row, 8 + 20 * col
            labels[y:y + 5, x:x + 5] = number
            # Every cell in the region is bright; the rest of the sample is not.
            green[y:y + 5, x:x + 5] = 500 + 5 * number if x < 60 else 100 + number
    images = {"AF405": np.full(labels.shape, 300.0), "AF488": green}
    in_region = sorted(int(n) for n in np.unique(labels[:, :60]) if n)

    table, counts = ja.measure_cells(labels, labels, images, (1.0, 1.0), band, nuclear="AF405")
    region = next(count for count in counts if count.channel == "AF488")
    check(region.n_positive < len(in_region) / 2 and region.n_threshold_cells == len(in_region),
          f"a region full of positive cells sets its own bar too high ({region.n_positive} of {len(in_region)})")

    options = ja.CountOptions(threshold_from=ja.FROM_SAMPLE)
    table, counts = ja.measure_cells(labels, labels, images, (1.0, 1.0), band, options, nuclear="AF405")
    sample = next(count for count in counts if count.channel == "AF488")
    check(sorted(table["Label"]) == in_region, "the cells counted are still the region's")
    check(sample.n_positive == len(in_region) and sample.n_threshold_cells == 40,
          f"against all the sample's cells every one of them is positive ({sample.n_positive}, from {sample.n_threshold_cells})")
    check(sample.threshold < region.threshold and sample.threshold_from == ja.FROM_SAMPLE,
          f"the threshold is the sample's ({sample.threshold:.0f} < {region.threshold:.0f})")
    check(list(table["AF488 mean"]) == [float(green[labels == n].mean()) for n in in_region],
          "and each cell's figures are its own")


def test_growing_the_nuclei_changes_what_is_measured_not_the_size() -> None:
    print("measuring around the nucleus")
    from microscopy_viewer import segmentation as sg

    labels, images, green_cells, _red = _field()
    # Put the GFP in a ring round the nucleus instead of under it.
    gfp = np.full(labels.shape, 200.0)
    grown = sg.expand_labels(labels, (1.0, 1.0), 3.0)
    gfp[np.isin(grown, green_cells) & (labels == 0)] += 900.0
    images = {"AF488": gfp, "AF405": images["AF405"]}
    band = np.ones(labels.shape, dtype=bool)

    nucleus_only, _ = ja.measure_cells(labels, labels, images, (1.0, 1.0), band, nuclear="AF405")
    with_ring, _ = ja.measure_cells(labels, grown, images, (1.0, 1.0), band, nuclear="AF405")
    check(int(nucleus_only["AF488 positive"].sum()) == 0, "a cytoplasmic label is missed under the nucleus alone")
    check(sorted(with_ring.loc[with_ring["AF488 positive"], "Label"]) == green_cells, "and found when the nucleus is grown")
    check(with_ring["Nucleus area (µm²)"].eq(25.0).all(), "while the size reported is still the nucleus")


def test_a_sample_end_to_end_and_the_workbook() -> None:
    print("a sample, and a folder")
    import tifffile

    from microscopy_viewer import ims_store, loaders

    labels, images, green_cells, red_cells = _field()
    stack = np.stack([np.clip(images[name], 1, 65535).astype(np.uint16)[None] for name in ("AF555", "AF488", "AF405")])

    def segment(plane, voxel, settings, progress=None):
        # The crop the engine is handed is the region's bounding box plus a margin.
        return labels[segment.window]

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        paths = []
        for index in range(2):
            path = root / f"assembloid_s{index}.ome.tiff"
            tifffile.imwrite(str(path), stack, ome=True, metadata={
                "axes": "CZYX", "Channel": {"Name": ["AF555", "AF488", "AF405"]},
                "PhysicalSizeX": 1.0, "PhysicalSizeY": 1.0, "PhysicalSizeZ": 4.0})
            paths.append(path)

        band = np.zeros(labels.shape, dtype=np.uint8)
        band[30:110, 40:150] = 1
        line = np.array([[30.0, 40.0], [30.0, 150.0]])
        ims_store.save_labels(paths[0], fu.DEFAULT_KEY, band, (1.0, 1.0),
                              attrs={"line_yx_um": line, "status": "ok", "flags": ""})
        margin = int(ja.CROP_MARGIN_UM)
        segment.window = (slice(30 - margin, 110 + margin), slice(40 - margin, 150 + margin))

        qc = root / "junction_analysis" / "QC"
        options = ja.CountOptions(expand_um=0.0, find_junction=False, save_cells=True, qc_dir=qc)
        result = ja.analyse_sample(paths[0], options, segment=segment)
        check(result.ok, f"the sample is counted ({result.error or 'ok'})")
        check(result.qc == "ok" and result.picture == qc / "ok" / "assembloid_s0.png" and result.picture.is_file(),
              "and pictured in the QC folder of its status")
        from PIL import Image

        with Image.open(result.picture) as picture:
            check(picture.size[0] >= 640 and picture.size[1] > 120, f"the picture is the counted crop, captioned {picture.size}")
        expected = sorted(int(n) for n in np.unique(labels) if n and band[tuple(np.round(np.argwhere(labels == n).mean(0)).astype(int))])
        check(sorted(result.cells["Label"]) == expected, f"with the cells whose centre is in the stored region ({result.n_cells})")
        check(result.band_source == f"stored ({fu.DEFAULT_KEY})" and result.status == "ok",
              f"with no hand-drawn region, the automatic one is used, with its status ({result.band_source})")
        check(abs(result.junction_length_um - 110.0) < 1.0, f"the junction length comes with it ({result.junction_length_um:.0f} µm)")
        cells = result.cells
        beside = cells[ja.BESIDE_COLUMN].to_numpy()
        depth = cells[ja.LINE_COLUMN].to_numpy()
        expected_depth = cells["Centroid Y (µm)"].to_numpy() - 30.0
        check(beside.any() and np.allclose(depth[beside], expected_depth[beside], atol=1.5),
              "each cell has its distance to the line, positive into the band")
        check(ja.EDGE_COLUMN in cells and np.isfinite(cells[ja.EDGE_COLUMN]).all(),
              "and to the tissue edge")
        plain = ja.analyse_sample(paths[0], ja.CountOptions(expand_um=0.0, find_junction=False, distances=False),
                                  segment=segment)
        check(plain.ok and ja.LINE_COLUMN not in plain.cells, "distances can be left out")
        check(abs(result.band_area_mm2 - 80 * 110 / 1e6) < 1e-9, "and the region's area")
        green = next(count for count in result.channels if count.channel == "AF488")
        check(green.n_positive == len(set(green_cells) & set(expected)), "GFP-positive cells are counted")
        check(abs(green.per_mm2 - green.n_positive / result.band_area_mm2) < 1e-6, "per mm² of region")
        check(abs(green.per_mm_junction - green.n_positive / 0.110) < 1e-6, "and per mm of junction")
        # 30 px of margin each way, stopped by the top and the right of the image.
        check(result.labels.shape == (120, 170) and result.origin == (0, 10),
              f"the cells come back on the cropped grid, with where it sits ({result.origin})")
        check(not loaders.is_open(paths[0]), "and the file is given back")

        stored, attrs = ims_store.load_labels(paths[0], ja.CELLS_KEY)
        check(stored is not None and stored.shape == labels.shape and sorted(np.unique(stored)[1:]) == expected,
              "the counted cells are stored on the image's grid")
        check(sorted(np.asarray(attrs["positive AF488"]).tolist()) == sorted(set(green_cells) & set(expected)),
              "with each channel's positive cells")

        # A region drawn by hand is preferred to the automatic one.
        manual = np.zeros(labels.shape, dtype=np.uint8)
        manual[30:70, 40:150] = 1
        ims_store.save_labels(paths[0], fu.MANUAL_KEY, manual, (1.0, 1.0), attrs={"line_yx_um": line})
        window = segment.window
        segment.window = (slice(0, 70 + margin), slice(40 - margin, 150 + margin))
        drawn = ja.analyse_sample(paths[0], ja.CountOptions(expand_um=0.0, find_junction=False), segment=segment)
        segment.window = window
        check(drawn.ok and drawn.band_source == f"stored ({fu.MANUAL_KEY})"
              and abs(drawn.band_area_mm2 - 40 * 110 / 1e6) < 1e-9,
              f"a region drawn by hand is counted in preference ({drawn.band_source})")
        check(drawn.region_by == "by hand" and drawn.finder_status == "ok",
              f"it says who drew it, and keeps what the finder said ({drawn.region_by}, {drawn.finder_status})")

        # By keyword: the first stored junction region whose name has it; a
        # label map that is not a junction region (no line) is passed over.
        second = np.zeros(labels.shape, dtype=np.uint8)
        second[30:90, 40:150] = 1
        ims_store.save_labels(paths[0], f"{fu.MANUAL_KEY} quillan", second, (1.0, 1.0), attrs={"line_yx_um": line})
        ims_store.save_labels(paths[0], "manual count AF488", manual, (1.0, 1.0), attrs={"channel": "AF488"})
        check(ja.region_candidates(paths[0], ja.CountOptions(region_keyword="MANUAL"))
              == [fu.MANUAL_KEY, f"{fu.MANUAL_KEY} quillan", fu.DEFAULT_KEY],
              "the keyword finds the hand-drawn regions in name order, then the automatic one")
        segment.window = (slice(0, 90 + margin), slice(40 - margin, 150 + margin))
        keyed = ja.analyse_sample(paths[0], ja.CountOptions(
            expand_um=0.0, find_junction=False, region_keyword="quillan",
            review={"assembloid_s0": {"Status": "check", "Decision": "redrawn", "Note": "redrawn by Q"}}), segment=segment)
        segment.window = window
        check(keyed.ok and keyed.band_source == f"stored ({fu.MANUAL_KEY} quillan)"
              and abs(keyed.band_area_mm2 - 60 * 110 / 1e6) < 1e-9,
              f"another keyword picks the other annotator's region ({keyed.band_source})")
        row = ja.counts_dataframe([keyed]).iloc[0]
        check(row["Region drawn by"] == "by hand" and row["Review decision"] == "redrawn"
              and row["Review status"] == "check" and row["Finder status"] == "ok",
              "the table carries who drew the region, the review, and the finder's verdict")
        ims_store.delete_labels(paths[0], f"{fu.MANUAL_KEY} quillan")
        ims_store.delete_labels(paths[0], "manual count AF488")
        ims_store.delete_labels(paths[0], fu.MANUAL_KEY)
        nothing = ja.region_candidates(paths[0], ja.CountOptions(region_keyword="manual"))
        check(nothing == [fu.DEFAULT_KEY], f"with none drawn, only the automatic one is left ({nothing})")

        # The whole sample segmented, for its threshold: what comes back is
        # still the region's cells on the region's crop.
        handed: list[tuple[int, ...]] = []

        def segment_all(plane, voxel, settings, progress=None):
            handed.append(plane.shape)
            return labels if plane.shape == labels.shape else labels[segment.window]

        everywhere = ja.analyse_sample(paths[0], ja.CountOptions(
            expand_um=0.0, find_junction=False, threshold_from=ja.FROM_SAMPLE), segment=segment_all)
        check(everywhere.ok and handed == [(120, 170), labels.shape],
              f"the region is segmented for the count and the whole section for the threshold ({handed})")
        check(sorted(everywhere.cells["Label"]) == expected and everywhere.labels.shape == (120, 170)
              and everywhere.origin == (0, 10), "and the cells counted, and their crop, are the region's")
        check(everywhere.channels[0].n_threshold_cells == int(labels.max())
              and everywhere.channels[0].threshold_from == ja.FROM_SAMPLE,
              f"with the threshold from every cell ({everywhere.channels[0].n_threshold_cells})")
        handed.clear()
        fixed = ja.analyse_sample(paths[0], ja.CountOptions(
            expand_um=0.0, find_junction=False, threshold_from=ja.FROM_SAMPLE, method=ja.METHOD_BACKGROUND),
            segment=lambda plane, voxel, settings, progress=None: handed.append(plane.shape) or labels[segment.window])
        check(fixed.ok and handed == [(120, 170)], "a rule that takes nothing from the cells segments only the region")

        # A section larger than a tile is segmented piece by piece, slide skipped.
        tile, ja.TILE_PX = ja.TILE_PX, 64
        try:
            pieces: list[tuple[int, ...]] = []

            def segment_tiles(plane, voxel, settings, progress=None):
                pieces.append(plane.shape)
                return (plane > 300).astype(np.int32)

            specs = loaders.load_path(paths[0])
            names = [str(spec.channel_name or spec.name) for spec in specs]
            values = ja.sample_cell_values(specs, names, names.index("AF405"), (1.0, 1.0),
                                           ja.CountOptions(expand_um=0.0), segment=segment_tiles)
            loaders.release(paths[0])
        finally:
            ja.TILE_PX = tile
        check(len(pieces) > 1 and max(max(piece) for piece in pieces) <= 64, f"in tiles ({len(pieces)})")
        check(set(values) == set(names) and len({len(v) for v in values.values()}) == 1 and len(values["AF488"]) > 0,
              "giving every channel a value for each cell found")

        # A GPU fault ends the run: the card will not answer for the next sample either.
        from microscopy_viewer import segmentation as sg

        check(sg.is_gpu_fault(RuntimeError("could not load the model “cpsam_v2”: CUDA error: an illegal memory access was encountered"))
              and not sg.is_gpu_fault(RuntimeError("CUDA out of memory. Tried to allocate 2 GiB"))
              and not sg.is_gpu_fault(ValueError("no junction found")),
              "a GPU fault is told from running out of memory and from an ordinary failure")
        calls: list[int] = []

        def faulting(plane, voxel, settings, progress=None):
            calls.append(1)
            raise RuntimeError("CUDA error: an illegal memory access was encountered")

        told: list[str] = []
        stopped = ja.run([paths[0], paths[0], paths[0]], options, progress=told.append, segment=faulting)
        check(len(stopped) == 1 and stopped[0].gpu_fault and len(calls) == 1,
              f"the run stops at the fault instead of failing every sample ({len(stopped)})")
        check("STOPPED" in told[-1] and "2 sample(s) were not attempted" in told[-1], "and says how many were not attempted")

        seen: list[str] = []
        results = ja.run(paths, options, progress=seen.append, segment=segment)
        check(results[0].labels is None and results[0].cells is not None,
              "a batch keeps the tables and not the label maps")
        check(ja.run(paths[:1], options, segment=segment)[0].labels is not None, "a single sample keeps its cells to draw")
        check(results[0].ok and not results[1].ok, "a folder run goes on past a sample with no region")
        check("no stored region" in results[1].error, f"and says what was missing ({results[1].error})")
        check(any("AF488+" in line for line in seen), "progress carries the counts")
        check(results[1].qc == "fail" and (qc / "fail" / "assembloid_s1.png").is_file(),
              "the sample that failed is pictured too, in fail")
        check(len(list(qc.glob("*/*.png"))) == 2, "one picture per sample, a rerun replacing the earlier one")

        frame = ja.counts_dataframe(results)
        check(list(frame.columns) == list(ja.COUNT_COLUMNS), "the counts table has its columns")
        check(len(frame) == 3 + 1, "one row per channel, and one for the sample that failed")
        reference = frame[frame["Channel"] == "AF405"].iloc[0]
        check(np.isnan(reference["Positive"]) and "reference" in reference["Rule"], "the reference stain has no count")
        combos = ja.combinations_dataframe(results)
        check(int(combos["Cells"].sum()) == results[0].n_cells, "the combinations add up to the cells")
        written = ja.export(results, root / "counts.xlsx", options)
        import pandas as pd

        sheets = pd.read_excel(written, sheet_name=None)
        check(set(sheets) == {"Counts", "Combinations", "Cells", "Settings"}, f"the workbook has its four sheets ({sorted(sheets)})")
        check(len(sheets["Cells"]) == results[0].n_cells and "AF488 SD" in sheets["Cells"].columns,
              "with every cell and its figures")

        # More cells than a sheet holds: the table goes beside the workbook whole.
        from microscopy_viewer import exports

        limit, exports.EXCEL_MAX_ROWS = exports.EXCEL_MAX_ROWS, results[0].n_cells - 1
        try:
            written = ja.export(results, root / "many.xlsx", options)
        finally:
            exports.EXCEL_MAX_ROWS = limit
        sheets = pd.read_excel(written, sheet_name=None)
        beside = root / "many_cells.csv"
        check(beside.exists() and len(pd.read_csv(beside)) == results[0].n_cells,
              "a table too long for a sheet is written whole to a csv beside the workbook")
        check(set(sheets) == {"Counts", "Combinations", "Cells", "Settings"}
              and sheets["Cells"]["Written to"].iloc[0] == beside.name,
              "and its sheet says where it went")
        check(len(sheets["Counts"]) == 3 + 1, "the other sheets are written as before")


def main() -> int:
    for test in (
        test_statistics_per_label,
        test_thresholds,
        test_cells_are_scored_per_channel,
        test_the_threshold_can_come_from_the_whole_sample,
        test_growing_the_nuclei_changes_what_is_measured_not_the_size,
        test_a_sample_end_to_end_and_the_workbook,
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
