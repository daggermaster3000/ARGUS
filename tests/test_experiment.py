"""Checks for the folder-at-a-time experiment tools and the in-file store.

Two things are worth checking without a GUI and without the real 8 GB datasets:
that ROIs and label maps survive a round trip through an HDF5 container without
disturbing what was already in it, and that a batch run picks the right channel,
reports per-file failures instead of aborting, and stops when asked.

The HDF5 checks build their own container — a few groups that look like an
Imaris file — so they run anywhere. Cellpose is never called: the batch goes
through a stub backend, the same seam ``test_segmentation`` uses.

Run with::

    python tests/test_experiment.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from microscopy_viewer import experiment as ex  # noqa: E402
from microscopy_viewer import regions as rg  # noqa: E402
from microscopy_viewer import ims_store as store  # noqa: E402
from microscopy_viewer import segmentation as seg  # noqa: E402

_failures: list[str] = []


def check(condition: bool, message: str) -> None:
    if condition:
        print(f"  ok   {message}")
    else:
        print(f"  FAIL {message}")
        _failures.append(message)


def _container(directory: Path, name: str = "sample.ims") -> Path:
    """An HDF5 file shaped enough like an Imaris one to write beside."""
    import h5py

    path = directory / name
    with h5py.File(str(path), "w") as handle:
        group = handle.create_group("DataSet/ResolutionLevel 0/TimePoint 0/Channel 0")
        group.create_dataset("Data", data=np.ones((2, 8, 8), dtype=np.uint16))
        handle.create_group("DataSetInfo/Image").attrs["Name"] = "test"
    return path


# ---------------------------------------------------------------------------
# The store
# ---------------------------------------------------------------------------


def test_rois_round_trip() -> None:
    print("ROIs in the file")
    with tempfile.TemporaryDirectory() as directory:
        path = _container(Path(directory))
        ok, reason = store.can_write(path)
        check(ok, f"a plain HDF5 file is writable ({reason})")

        rois = [
            store.StoredRoi("cerebellum left", [[0.0, 0.0], [0.0, 50.0], [50.0, 50.0]]),
            store.StoredRoi("cerebellum right", [[0.0, 60.0], [0.0, 110.0], [50.0, 110.0]]),
        ]
        check(store.save_rois(path, rois) == 2, "two ROIs written")

        back = store.load_rois(path)
        check([roi.name for roi in back] == [roi.name for roi in rois], "names survive the trip")
        check(
            np.allclose(back[0].vertices_um, rois[0].vertices_um),
            "and so do the vertices, to the micrometre",
        )
        # A name with a slash in it would be an illegal HDF5 path; keys are
        # positional for exactly that reason.
        odd = [store.StoredRoi("left/right", [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0]])]
        store.save_rois(path, odd)
        check([roi.name for roi in store.load_rois(path)] == ["left/right"],
              "a name containing a slash is still stored")
        check(len(store.load_rois(path)) == 1, "and saving replaces the previous set")

        # A shape with too few vertices encloses nothing and is not stored.
        store.save_rois(path, [store.StoredRoi("line", [[0.0, 0.0], [1.0, 1.0]])])
        check(store.load_rois(path) == [], "a two-point 'ROI' is not written")


def test_labels_round_trip_without_disturbing_the_image() -> None:
    print("label maps in the file")
    import h5py

    with tempfile.TemporaryDirectory() as directory:
        path = _container(Path(directory))
        original = np.ones((2, 8, 8), dtype=np.uint16)

        masks = np.zeros((2, 8, 8), dtype=np.int32)
        masks[:, 1:4, 1:4] = 1
        masks[:, 5:7, 5:7] = 2

        key = store.save_labels(path, "dapi labels", masks, (2.0, 0.3, 0.3), {"model": "cpsam"})
        check(key == "dapi labels", f"stored under the name it was given ({key})")
        check(store.list_labels(path) == ["dapi labels"], "and listed back")

        loaded, attrs = store.load_labels(path, key)
        check(np.array_equal(loaded, masks), "the label map comes back identical")
        check(int(attrs["n_objects"]) == 2, "with its object count")
        check(str(attrs["model"]) == "cpsam", "and the settings it was made with")
        check(
            np.allclose(np.asarray(attrs["voxel_size_um"]), (2.0, 0.3, 0.3)),
            "and the voxel size, so it can be put back on a layer",
        )

        # The whole point of writing into the file is that the file still works.
        with h5py.File(str(path), "r") as handle:
            image = handle["DataSet/ResolutionLevel 0/TimePoint 0/Channel 0/Data"]
            check(np.array_equal(np.asarray(image), original), "the image data is untouched")
            check("DataSetInfo" in handle, "and so is the metadata group")
            check(handle["ARGUS"].attrs["format_version"] == store.FORMAT_VERSION,
                  "our group carries a format stamp")

        check(store.summary(path) == "1 label map(s)", f"summary reads right ({store.summary(path)})")


def test_a_file_that_is_not_a_container() -> None:
    print("refusing what cannot be written")
    with tempfile.TemporaryDirectory() as directory:
        plain = Path(directory) / "notes.txt"
        plain.write_text("not HDF5")
        check(not store.is_container(plain), "a text file is not a container")
        ok, reason = store.can_write(plain)
        check(not ok and "HDF5" in reason, f"and says why ({reason})")
        check(store.load_rois(plain) == [], "reading one yields nothing rather than raising")
        check(store.list_labels(plain) == [], "same for labels")

        missing = Path(directory) / "gone.ims"
        check(store.can_write(missing)[0] is False, "so is a file that is not there")


# ---------------------------------------------------------------------------
# Listing and previewing
# ---------------------------------------------------------------------------


def test_listing_a_folder() -> None:
    print("listing a folder")
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        _container(root, "fish_2.ims")
        _container(root, "fish_1.ims")
        (root / "day2").mkdir()
        _container(root / "day2", "fish_3.ims")
        (root / "notes.txt").write_text("ignored")
        # What macOS writes beside every file on an exFAT drive.
        (root / "._fish_1.ims").write_bytes(b"\x00\x05\x16\x07")

        flat = ex.list_files(root, recursive=False)
        check([path.name for path in flat] == ["fish_1.ims", "fish_2.ims"],
              f"sorted, and only the datasets ({[p.name for p in flat]})")

        deep = ex.list_files(root, recursive=True)
        check(len(deep) == 3, f"subfolders included when asked ({len(deep)})")
        check(ex.list_files(root / "nowhere") == [], "a folder that is not there lists nothing")


def test_thumbnail_merges_channels() -> None:
    print("thumbnails")
    # A bright left half in green, a bright right half in magenta.
    left = np.zeros((32, 32), dtype=np.float32)
    left[:, :16] = 1000.0
    right = np.zeros((32, 32), dtype=np.float32)
    right[:, 16:] = 1000.0

    merged = ex.merge_channels([(left, (0.0, 1.0, 0.0)), (right, (1.0, 0.0, 1.0))], max_pixels=32)
    check(merged.shape == (32, 32, 3), f"one RGB image out ({merged.shape})")
    check(merged.dtype == np.uint8, "as 8-bit, ready for a QImage")
    check(
        merged[0, 0, 1] > 200 and merged[0, 0, 0] < 50,
        f"the left half is green ({merged[0, 0].tolist()})",
    )
    check(
        merged[0, 31, 0] > 200 and merged[0, 31, 2] > 200,
        f"the right half is magenta ({merged[0, 31].tolist()})",
    )

    # A flat image has no contrast to stretch and must not come back as noise.
    flat = ex.merge_channels([(np.full((8, 8), 5.0), (1.0, 1.0, 1.0))])
    check(int(flat.max()) == 0, "a flat image stretches to black rather than to garbage")

    # Downsampling never lands under the requested size: scaling a preview back
    # up is what makes a thumbnail blurry.
    small = ex._downsample(np.zeros((500, 500)), 220)
    check(min(small.shape) >= 220, f"a 500 px plane stays at least 220 across ({small.shape})")


# ---------------------------------------------------------------------------
# The batch
# ---------------------------------------------------------------------------


def test_channel_is_picked_by_name() -> None:
    print("picking the channel")
    names = ["Confocal - NeuroD1-488", "Confocal - DAPI", "Confocal - calretinin-647"]
    check(ex.pick_channel(names, "dapi") == 1, "matched by name, case-insensitively")
    check(ex.pick_channel(names, "647") == 2, "and by any fragment of it")
    check(ex.pick_channel(names, 0) == 0, "an integer is an index")
    check(ex.pick_channel(names, 9) is None, "an index past the end finds nothing")
    # Falling back to channel 0 would segment the wrong channel in silence, which
    # over thirty files is a whole experiment's worth of wrong numbers.
    check(ex.pick_channel(names, "gfp") is None, "a name that matches nothing is not channel 0")
    check(ex.pick_channel(names, None) is None, "and nothing asked for is nothing found")


class _FakeSpec:
    """Enough of a LayerSpec for the batch to read a channel off."""

    def __init__(self, name, data, scale):
        self.name = name
        self.channel_name = name
        self.data = data
        self.scale = scale
        self.multiscale = False


def _patch_reader(monkey: dict, specs_by_path) -> None:
    monkey["specs"] = ex._read_specs
    monkey["release"] = ex._release
    ex._read_specs = lambda path: specs_by_path[Path(path).name]
    ex._release = lambda path: None


def _unpatch_reader(monkey: dict) -> None:
    ex._read_specs = monkey["specs"]
    ex._release = monkey["release"]


class _StubBackend(seg.Backend):
    """Labels a fixed block, and records the shape it was handed."""

    name = "stub-batch"

    def __init__(self):
        self.calls: list[tuple] = []

    def available(self) -> bool:
        return True

    def segment(self, image, settings, diameter_px, anisotropy, device, progress=None):
        array = np.asarray(image)
        self.calls.append(array.shape)
        masks = np.zeros(array.shape, dtype=np.int32)
        masks[..., 1:3, 1:3] = 1
        return masks, {}


def test_batch_runs_over_files_and_keeps_going() -> None:
    print("the batch")
    backend = _StubBackend()
    seg.register_backend(backend)
    monkey: dict = {}
    try:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            good = _container(root, "fish_1.ims")
            also = _container(root, "fish_2.ims")

            volume = np.zeros((2, 8, 8), dtype=np.float32)
            specs = {
                "fish_1.ims": [
                    _FakeSpec("Confocal - GFP", volume, (1.0, 0.5, 0.5)),
                    _FakeSpec("Confocal - DAPI", volume, (1.0, 0.5, 0.5)),
                ],
                # Channel order deliberately different, and no DAPI in the second.
                "fish_2.ims": [_FakeSpec("Confocal - GFP", volume, (1.0, 0.5, 0.5))],
            }
            _patch_reader(monkey, specs)

            messages: list[str] = []
            options = ex.BatchOptions(
                channel="dapi",
                settings=seg.SegmentationSettings(backend="stub-batch", use_gpu=False),
            )
            outcomes = ex.run_batch([good, also], options, progress=messages.append)

            check(len(outcomes) == 2, "every file gets an outcome")
            check(outcomes[0].ok and outcomes[0].n_objects == 1, "the first one segmented")
            check(
                outcomes[0].channel_name == "Confocal - DAPI",
                f"on the channel named, not the first one ({outcomes[0].channel_name})",
            )
            # One bad file must not cost the other twenty-nine.
            check(not outcomes[1].ok, "the file without that channel failed")
            check("no channel matching" in outcomes[1].error, f"and says why ({outcomes[1].error})")
            check(outcomes[0].saved, "the labels went into the file")
            check(store.list_labels(good) == ["Confocal - DAPI labels"],
                  f"under a name naming the channel ({store.list_labels(good)})")
            check(store.list_labels(also) == [], "and nothing was written for the failure")

            stored, attrs = store.load_labels(good, "Confocal - DAPI labels")
            check(int(stored.max()) == 1, "the stored map is the one segmented")
            check(str(attrs["channel"]) == "Confocal - DAPI", "tagged with its channel")

            check(any("fish_1" in text for text in messages), "progress names the file it is on")
            check(len(outcomes[0].stats) == 1, "and the objects were measured")
    finally:
        _unpatch_reader(monkey)
        seg._BACKENDS.pop("stub-batch", None)


def test_batch_stops_when_asked() -> None:
    print("stopping a batch")
    seg.register_backend(_StubBackend())
    monkey: dict = {}
    try:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [_container(root, f"fish_{index}.ims") for index in range(4)]
            volume = np.zeros((2, 8, 8), dtype=np.float32)
            _patch_reader(
                monkey,
                {path.name: [_FakeSpec("DAPI", volume, (1.0, 0.5, 0.5))] for path in paths},
            )

            seen: list[str] = []

            def _stop() -> bool:
                return len(seen) >= 2

            def _progress(text: str) -> None:
                if "reading" in text:
                    seen.append(text)

            outcomes = ex.run_batch(
                paths,
                ex.BatchOptions(
                    channel="dapi",
                    save_to_file=False,
                    settings=seg.SegmentationSettings(backend="stub-batch", use_gpu=False),
                ),
                progress=_progress,
                should_cancel=_stop,
            )
            check(len(outcomes) == 2, f"it stopped after the file it was on ({len(outcomes)})")
            check(all(outcome.ok for outcome in outcomes), "and what it did finish is complete")
            check(store.list_labels(paths[0]) == [], "with save_to_file off, nothing was written")
    finally:
        _unpatch_reader(monkey)
        seg._BACKENDS.pop("stub-batch", None)


def test_roi_restriction_blanks_the_outside() -> None:
    print("restricting to the ROIs")
    image = np.ones((3, 20, 20), dtype=np.float32)
    rois = [store.StoredRoi("brain", [[0.0, 0.0], [0.0, 5.0], [5.0, 5.0], [5.0, 0.0]])]

    # 1 µm/px, so the 5 µm outline covers the first five pixels each way.
    mask = ex.roi_mask(rois, (20, 20), (1.0, 1.0))
    check(mask is not None and mask[2, 2] and not mask[10, 10], "the outline rasterises where it is")

    masked = ex.apply_roi_mask(image, mask)
    check(masked.shape == image.shape, "the array keeps its shape, so labels land on the source grid")
    check(masked[0, 2, 2] == 1.0 and masked[0, 10, 10] == 0.0, "inside kept, outside blanked")
    check(masked[2, 2, 2] == 1.0, "and a 2D outline applies through every plane")

    # Half the voxel size means the same micrometres cover twice the pixels.
    finer = ex.roi_mask(rois, (20, 20), (0.5, 0.5))
    check(bool(finer[8, 8]), "the outline is in micrometres, not pixels")

    check(ex.roi_mask([], (20, 20), (1.0, 1.0)) is None, "no ROIs means no mask")
    check(
        ex.apply_roi_mask(image, None) is image,
        "and no mask leaves the image alone rather than copying it",
    )


def test_the_batch_tables() -> None:
    print("batch tables")
    outcome = ex.BatchOutcome(
        path=Path("fish_1.ims"),
        name="fish_1",
        n_objects=2,
        channel_name="DAPI",
        label_key="DAPI labels",
        saved=True,
        elapsed_s=3.2,
        ndim=3,
    )
    masks = np.zeros((2, 8, 8), dtype=np.int32)
    masks[:, 1:3, 1:3] = 1
    outcome.stats = seg.object_table(masks, voxel_size_um=(1.0, 1.0, 1.0))

    samples = ex.batch_dataframe([outcome])
    check(list(samples.columns)[:4] == ["Sample", "Genotype", "Channel", "Objects"],
          f"one row per sample ({list(samples.columns)[:4]})")
    check(samples["Saved as"].iloc[0] == "DAPI labels", "naming where the labels went")

    objects = ex.objects_dataframe([outcome])
    check("Sample" in objects.columns, "every object carries its sample")
    check(len(objects) == len(outcome.stats), "and there is one row per object")
    check("Volume (µm³)" in objects.columns, "3D labels are labelled as volumes")

    flat = ex.BatchOutcome(path=Path("x.ims"), name="x", ndim=2, stats=outcome.stats)
    check("Area (µm²)" in ex.objects_dataframe([flat]).columns,
          "a projected run's table says area instead")


def test_the_regions_sheet() -> None:
    print("the regions sheet")
    # Two regions side by side in a 20 x 20 µm field, and four objects: two in
    # the left one, one in the right, one in neither.
    masks = np.zeros((1, 20, 20), dtype=np.int32)
    masks[0, 2:4, 2:4] = 1
    masks[0, 6:8, 3:5] = 2
    masks[0, 3:5, 14:16] = 3
    masks[0, 16:18, 16:18] = 4
    stats = seg.object_table(masks, voxel_size_um=(1.0, 1.0, 1.0))

    left = store.StoredRoi("forebrain", [[0.0, 0.0], [0.0, 10.0], [10.0, 10.0], [10.0, 0.0]])
    right = store.StoredRoi("hindbrain", [[0.0, 10.0], [0.0, 20.0], [10.0, 20.0], [10.0, 10.0]])

    outcome = ex.BatchOutcome(
        path=Path("fish01_mut_20x.ims"),
        name="fish01_mut_20x",
        genotype="mut",
        n_objects=4,
        channel_name="DAPI",
        ndim=3,
        stats=stats,
        region_rois=[left, right],
    )

    frame = ex.regions_dataframe([outcome])
    check(len(frame) == 3, f"a row per region plus the leftovers ({len(frame)})")
    by_region = {row["Region"]: row for _index, row in frame.iterrows()}
    check(int(by_region["forebrain"]["Objects"]) == 2, "two objects in the left region")
    check(int(by_region["hindbrain"]["Objects"]) == 1, "one in the right")
    check(
        int(by_region[rg.UNASSIGNED]["Objects"]) == 1,
        "and the one outside both is reported rather than dropped",
    )
    # The property that makes the sheet trustworthy: nothing is counted twice
    # and nothing disappears.
    check(int(frame["Objects"].sum()) == 4, "the region counts sum to the sample total")

    check(
        abs(float(by_region["forebrain"]["Region area (µm²)"]) - 100.0) < 1e-6,
        "the region's area is its own, in µm²",
    )
    check(
        abs(float(by_region["forebrain"]["Objects per mm²"]) - 2 / (100.0 / 1e6)) < 1e-3,
        "and the density is per mm², not per µm²",
    )
    check(set(frame["Genotype"]) == {"mut"}, "every row carries the genotype")
    check(
        np.isfinite(float(by_region["forebrain"]["Mean diameter (µm)"])),
        "with the morphometrics of the objects in it",
    )
    check(
        "Total Volume (µm³)" in frame.columns,
        f"3D labels give volumes ({[c for c in frame.columns if 'Total' in c]})",
    )
    check(bool(by_region[rg.UNASSIGNED]["Outside every region"]), "the leftover row is flagged")
    check(not bool(by_region["forebrain"]["Outside every region"]), "and a real region is not")


def test_a_sample_with_no_regions_still_gets_a_row() -> None:
    print("a sample nobody has outlined")
    masks = np.zeros((1, 8, 8), dtype=np.int32)
    masks[0, 1:3, 1:3] = 1
    outcome = ex.BatchOutcome(
        path=Path("fish_2.ims"),
        name="fish_2",
        n_objects=1,
        ndim=2,
        stats=seg.object_table(masks, voxel_size_um=(1.0, 1.0, 1.0)),
    )
    frame = ex.regions_dataframe([outcome])
    # A missing row reads as a sample that failed; this one segmented fine.
    check(len(frame) == 1, "it is still in the sheet")
    check(frame["Region"].iloc[0] == ex.NO_REGIONS, f"saying so ({frame['Region'].iloc[0]})")
    check(int(frame["Objects"].iloc[0]) == 1, "with all of its objects")
    check("Total Area (µm²)" in frame.columns, "and a 2D run's sizes are areas")


def test_the_genotype_comes_off_the_file_name() -> None:
    print("genotypes in the batch tables")
    backend = _StubBackend()
    seg.register_backend(backend)
    monkey: dict = {}
    try:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mutant = _container(root, "fish01_mut_20x.ims")
            wildtype = _container(root, "fish02_wt_20x.ims")
            volume = np.zeros((2, 8, 8), dtype=np.float32)
            _patch_reader(
                monkey,
                {
                    "fish01_mut_20x.ims": [_FakeSpec("Confocal - DAPI", volume, (1.0, 0.5, 0.5))],
                    "fish02_wt_20x.ims": [_FakeSpec("Confocal - DAPI", volume, (1.0, 0.5, 0.5))],
                },
            )
            options = ex.BatchOptions(
                channel="dapi",
                settings=seg.SegmentationSettings(backend="stub-batch", use_gpu=False),
            )
            outcomes = ex.run_batch([mutant, wildtype], options)
            check([o.genotype for o in outcomes] == ["mut", "wt"],
                  f"read off each file name ({[o.genotype for o in outcomes]})")

            samples = ex.batch_dataframe(outcomes)
            check(list(samples["Genotype"]) == ["mut", "wt"], "and carried into the samples sheet")
            objects = ex.objects_dataframe(outcomes)
            check("Genotype" in objects.columns, "and onto every object")
            check("Region" in objects.columns, "which also says which region it fell in")
    finally:
        _unpatch_reader(monkey)
        seg._BACKENDS.pop("stub-batch", None)


def test_a_write_that_does_not_survive_is_reported() -> None:
    print("verifying the write")
    backend = _StubBackend()
    seg.register_backend(backend)
    monkey: dict = {}
    original = store.save_labels
    try:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = _container(root, "fish_1.ims")
            volume = np.zeros((2, 8, 8), dtype=np.float32)
            _patch_reader(monkey, {"fish_1.ims": [_FakeSpec("DAPI", volume, (1.0, 0.5, 0.5))]})

            # A write that returns a key without leaving anything behind: what a
            # full disk or a file opened read-only underneath us looks like.
            store.save_labels = lambda *args, **kwargs: "DAPI labels"
            ex.ims_store.save_labels = store.save_labels
            options = ex.BatchOptions(
                channel="dapi",
                settings=seg.SegmentationSettings(backend="stub-batch", use_gpu=False),
            )
            outcomes = ex.run_batch([path], options)
            check(not outcomes[0].saved, "the outcome does not claim to have saved")
            check(not outcomes[0].ok, "the file is reported as a failure")
            check("did not survive" in outcomes[0].error,
                  f"saying what went wrong ({outcomes[0].error})")
            check(ex.batch_dataframe(outcomes)["Saved as"].iloc[0] == "",
                  "and the sheet does not name a label map that is not there")
    finally:
        store.save_labels = original
        ex.ims_store.save_labels = original
        _unpatch_reader(monkey)
        seg._BACKENDS.pop("stub-batch", None)


def test_labels_are_stored_in_the_narrowest_type_that_holds_them() -> None:
    print("label storage")
    with tempfile.TemporaryDirectory() as directory:
        path = _container(Path(directory))
        masks = np.zeros((2, 8, 8), dtype=np.int32)
        masks[0, 1:3, 1:3] = 300  # too big for uint8, fits uint16
        store.save_labels(path, "labels", masks, (1.0, 1.0, 1.0))
        stored, _attrs = store.load_labels(path, "labels")
        check(stored.dtype == np.uint16, f"int32 masks are narrowed ({stored.dtype})")
        check(np.array_equal(stored, masks), "without changing a single label")
        check(store.label_shape(path, "labels") == (2, 8, 8), "and the shape can be read alone")
        check(store.label_shape(path, "absent") == (), "a missing map has no shape")

def main() -> int:
    for test in (
        test_rois_round_trip,
        test_labels_round_trip_without_disturbing_the_image,
        test_a_file_that_is_not_a_container,
        test_listing_a_folder,
        test_thumbnail_merges_channels,
        test_channel_is_picked_by_name,
        test_batch_runs_over_files_and_keeps_going,
        test_batch_stops_when_asked,
        test_roi_restriction_blanks_the_outside,
        test_the_batch_tables,
        test_the_regions_sheet,
        test_a_sample_with_no_regions_still_gets_a_row,
        test_the_genotype_comes_off_the_file_name,
        test_a_write_that_does_not_survive_is_reported,
        test_labels_are_stored_in_the_narrowest_type_that_holds_them,
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
