"""Checks for batch segmentation over an HCS plate. No Qt, no display, no GPU.

Cellpose is never run here — a stub backend stands in for it, the same way
``test_segmentation.py`` does — so what is being checked is the part that is ours:
walking the plate, matching a channel whose label changes from cycle to cycle,
writing NGFF label groups back into the store, skipping what is already done, and
reading the result back through the ordinary reader as a Labels layer.

Run with::

    python tests/test_batch.py
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from microscopy_viewer import batch  # noqa: E402
from microscopy_viewer import segmentation as seg  # noqa: E402

_failures: list[str] = []


def check(condition: bool, message: str) -> None:
    if condition:
        print(f"  ok   {message}")
    else:
        print(f"  FAIL {message}")
        _failures.append(message)


def _has_zarr() -> bool:
    try:
        import zarr  # noqa: F401
    except ImportError:
        return False
    return True


class _StubBackend(seg.Backend):
    """Labels a fixed corner block, and remembers what it was handed."""

    name = "batch-stub"
    install_hint = "the stub is always available"

    def __init__(self):
        self.calls: list[dict] = []

    def available(self) -> bool:
        return True

    def model_choices(self):
        return (seg.ModelChoice(label="stub-model", value="stub-model"),)

    def segment(
        self, image, settings, diameter_px, anisotropy, device, progress=None, channel_axis=None
    ):
        image = np.asarray(image)
        self.calls.append({"shape": tuple(image.shape), "channel_axis": channel_axis})
        if channel_axis is not None:
            image = np.take(image, 0, axis=channel_axis)
        masks = np.zeros(image.shape, dtype=np.int32)
        masks[..., :6, :6] = 1
        masks[..., 10:14, 10:14] = 2
        return masks, {}


def _stub_settings(**kwargs) -> batch.BatchSettings:
    defaults = dict(
        segmentation=seg.SegmentationSettings(backend="batch-stub", model="stub-model"),
        channel=batch.ChannelPick(batch.BY_WAVELENGTH, "A01_C01"),
        write_tables=False,
    )
    defaults.update(kwargs)
    return batch.BatchSettings(**defaults)


def _make_plate(directory: Path) -> Path:
    from make_sample_data import write_fractal_plate

    return write_fractal_plate(directory / "plate.zarr")


# ---------------------------------------------------------------------------


def test_channel_matching() -> None:
    print("matching a channel across cycles")

    cycle1 = (
        batch.ChannelInfo(0, "Ab1_DAPI", "A01_C01"),
        batch.ChannelInfo(1, "Green488-bCAT", "A02_C02"),
    )
    cycle7 = (
        batch.ChannelInfo(0, "Ab7_DAPI", "A01_C01"),
        batch.ChannelInfo(1, "Green488-COL1", "A02_C02"),
    )

    pick = batch.ChannelPick(batch.BY_WAVELENGTH, "A01_C01")
    check(
        pick.resolve(cycle1).label == "Ab1_DAPI" and pick.resolve(cycle7).label == "Ab7_DAPI",
        "a wavelength id finds the nuclear channel in every cycle",
    )
    check(
        batch.ChannelPick(batch.BY_LABEL, "dapi").resolve(cycle7).index == 0,
        "a label is matched case-insensitively and as a substring",
    )
    check(
        batch.ChannelPick(batch.BY_LABEL, "Ab1_DAPI").resolve(cycle7) is None,
        "the cycle-1 label does not match cycle 7 — which is why wavelength is the default",
    )
    check(
        batch.ChannelPick(batch.BY_INDEX, "1").resolve(cycle1).wavelength_id == "A02_C02",
        "an index picks by position",
    )
    check(
        batch.ChannelPick(batch.BY_WAVELENGTH, "A09_C09").resolve(cycle1) is None,
        "a channel the image does not have resolves to nothing rather than to the first one",
    )
    check(not batch.ChannelPick().is_set, "an empty pick reports itself unset")

    exact = (batch.ChannelInfo(0, "DAPI_2", ""), batch.ChannelInfo(1, "DAPI", ""))
    check(
        batch.ChannelPick(batch.BY_LABEL, "DAPI").resolve(exact).index == 1,
        "an exact label beats a substring when the plate holds both",
    )


def test_survey(directory: Path) -> None:
    print("surveying a plate")

    plate = _make_plate(directory)
    survey = batch.survey_plate(plate)

    check(len(survey.jobs) == 8, f"four wells of two cycles is eight images (got {len(survey.jobs)})")
    check(survey.wells == ("B/02", "B/03", "C/02", "C/03"), "wells come back in plate order")
    check(survey.acquisitions == (1, 2), "acquisition ids are read from the well metadata")

    job = survey.jobs[0]
    check(job.component == "B/02/0", "the component is the path inside the store")
    check(job.axes == "CZYX", "the axes come from the NGFF axis list")
    check(job.shape == (2, 1, 32, 48), "the shape is the full-resolution level")
    check(
        job.voxel_size_um(0) == (1.0, 0.25, 0.25),
        "the voxel size is the level-0 scale, in (z, y, x)",
    )
    check(
        job.voxel_size_um(1) == (1.0, 0.5, 0.5),
        "a lower pyramid level reports its own, coarser scale",
    )
    check(
        job.level(9).path == job.levels[-1].path,
        "asking for a level the plate does not have gives the smallest one",
    )
    check(
        len(survey.channels()) == 2,
        "the channel list is the union across cycles, keyed by wavelength",
    )
    check(job.describe() == "B/02 · cycle 1", "an image describes itself by well and cycle")

    chosen = batch.select_jobs(survey, wells=["B/02"], acquisitions=[1])
    check(
        [j.component for j in chosen] == ["B/02/0"],
        "selecting one well and one cycle gives one image",
    )
    check(
        len(batch.select_jobs(survey, acquisitions=[2])) == 4,
        "selecting a cycle alone gives one image per well",
    )

    try:
        batch.survey_plate(plate / "B" / "02")
        check(False, "a group that is not a plate is refused")
    except ValueError as exc:
        check("not an OME-Zarr plate" in str(exc), f"a non-plate group is refused: {exc}")


def test_labels_are_written(directory: Path) -> None:
    print("writing labels back into the plate")

    plate = _make_plate(directory)
    survey = batch.survey_plate(plate)
    job = survey.jobs[0]

    masks = np.zeros((1, 32, 48), dtype=np.int32)
    masks[0, 2:10, 2:10] = 1
    masks[0, 20:30, 20:30] = 2

    check(not batch.has_labels(job, "nuclei"), "a fresh plate has no label sets")
    written = batch.write_labels(job, "nuclei", masks)
    check(written.is_dir(), f"the label group is on disk at {written.name}")
    check(batch.has_labels(job, "nuclei"), "the label set is found again afterwards")

    import zarr

    group = zarr.open_group(str(written), mode="r")
    entry = dict(group.attrs)["multiscales"][0]
    check(
        [axis["name"] for axis in entry["axes"]] == ["z", "y", "x"],
        "the label axes are the image's axes without the channel",
    )
    check(
        len(entry["datasets"]) == len(job.levels),
        "the label pyramid has as many levels as the image it came from",
    )
    check(
        dict(group.attrs)["image-label"]["source"]["image"] == "../../",
        "the image-label block points back at the image",
    )

    level0 = np.asarray(group["0"])
    check(level0.shape == masks.shape, "level 0 is on the grid that went in")
    check(level0.dtype == np.uint32, "labels are stored as uint32, so a big well cannot overflow")
    check(int(level0.max()) == 2, "the label values are unchanged")

    level1 = np.asarray(group["1"])
    check(level1.shape == (1, 16, 24), "level 1 is the image's own level-1 shape")
    check(
        set(np.unique(level1)) <= {0, 1, 2},
        "a downsampled level holds label values, not averages of them",
    )

    index = dict(zarr.open_group(str(job.path / "labels"), mode="r").attrs)
    check(index.get("labels") == ["nuclei"], "the labels group lists what is in it")

    try:
        batch.write_labels(job, "nuclei", masks)
        check(False, "writing over an existing label set is refused")
    except FileExistsError:
        check(True, "writing over an existing label set is refused without overwrite")

    batch.write_labels(job, "nuclei", masks * 0 + 5, overwrite=True)
    check(
        int(np.asarray(zarr.open_group(str(written), mode="r")["0"]).max()) == 5,
        "overwrite replaces the masks",
    )
    batch.write_labels(job, "cells", masks)
    check(
        dict(zarr.open_group(str(job.path / "labels"), mode="r").attrs)["labels"]
        == ["nuclei", "cells"],
        "a second label set is added to the index rather than replacing it",
    )


def test_labels_load_as_a_labels_layer(directory: Path) -> None:
    print("reading the result back")

    from microscopy_viewer.loaders import ome_zarr

    plate = _make_plate(directory)
    survey = batch.survey_plate(plate)
    job = survey.jobs[0]
    masks = np.zeros((1, 32, 48), dtype=np.int32)
    masks[0, 4:12, 4:12] = 3
    batch.write_labels(job, "nuclei", masks)

    specs = ome_zarr.read(job.path)
    kinds = [spec.layer_type for spec in specs]
    check(kinds == ["image", "image", "labels"], f"two channels and one label set (got {kinds})")

    label_spec = specs[-1]
    image_spec = specs[0]
    check(label_spec.name.endswith(":: nuclei"), f"the label layer is named after the set: {label_spec.name}")
    check(
        label_spec.axes == image_spec.axes and label_spec.scale == image_spec.scale,
        "the labels come back on the same axes and scale as the channels",
    )
    check(
        "colormap" not in label_spec.to_kwargs(),
        "a Labels layer is not handed a colormap, which add_labels would refuse",
    )
    check(
        label_spec.to_kwargs()["multiscale"] is True,
        "the label pyramid is passed through as a pyramid",
    )


def test_labels_reach_the_plate_mosaic(directory: Path) -> None:
    print("segmentations on the plate mosaic")

    from microscopy_viewer.loaders import ome_zarr

    plate = _make_plate(directory)
    survey = batch.survey_plate(plate)
    # Segment one well of one cycle, the way a run that stopped early leaves it.
    job = batch.select_jobs(survey, wells=["B/02"], acquisitions=[1])[0]
    masks = np.zeros((1, 32, 48), dtype=np.int32)
    masks[0, 4:12, 4:12] = 1
    batch.write_labels(job, "nuclei", masks)

    specs = ome_zarr.read(plate)
    labels = [spec for spec in specs if spec.layer_type == "labels"]
    images = [spec for spec in specs if spec.layer_type == "image"]

    check(len(labels) == 1, f"the one label set becomes one mosaic layer (got {len(labels)})")
    check(
        labels[0].data[0].shape == images[0].data[0].shape,
        f"the label mosaic is on the plate grid: {labels[0].data[0].shape} "
        f"vs {images[0].data[0].shape}",
    )
    check(
        "cycle 1 :: nuclei" in labels[0].name,
        f"and it names its cycle and label set: {labels[0].name}",
    )
    check(
        labels[0].metadata.extra.get("Wells segmented") == "1",
        "the layer says how much of the plate was segmented",
    )

    # The masks belong to well B/02, which is the top-left cell of the mosaic;
    # everywhere else has to be blank rather than a repeat of the same tile.
    mosaic = np.asarray(labels[0].data[0])
    check(int(mosaic[:32, :48].max()) == 1, "the segmented well carries its masks")
    check(int(mosaic[:32, 48:].max()) == 0, "the well beside it is blank, not a copy")
    check(int(mosaic[32:, :].max()) == 0, "so is the row below it")


def test_a_run(directory: Path) -> None:
    print("a whole run")

    stub = _StubBackend()
    seg.register_backend(stub)
    plate = _make_plate(directory)
    survey = batch.survey_plate(plate)
    jobs = batch.select_jobs(survey, acquisitions=[1])

    settings = _stub_settings(nuclei=batch.ChannelPick(batch.BY_WAVELENGTH, "A02_C02"))
    report = batch.run_batch(jobs, settings, survey=survey)

    check(report.counted(batch.STATUS_DONE) == 4, "every selected image ran")
    check(report.total_objects == 8, "two objects per image were counted")
    check(
        stub.calls[0]["channel_axis"] == stub.calls[0]["shape"].index(2)
        if 2 in stub.calls[0]["shape"]
        else False,
        "the nuclear channel reaches the backend on a channel axis",
    )
    check(
        all(batch.has_labels(job, "nuclei") for job in jobs),
        "each image has its label set afterwards",
    )
    check(
        not any(batch.has_labels(job, "nuclei") for job in batch.select_jobs(survey, acquisitions=[2])),
        "the cycle that was not selected was left alone",
    )

    again = batch.run_batch(jobs, settings, survey=survey)
    check(
        again.counted(batch.STATUS_SKIPPED) == 4,
        "a second run skips what is already there, so an interrupted plate can be resumed",
    )
    check(
        batch.run_batch(jobs, _stub_settings(overwrite=True), survey=survey).counted(
            batch.STATUS_DONE
        )
        == 4,
        "overwrite runs them again",
    )

    missing = _stub_settings(channel=batch.ChannelPick(batch.BY_WAVELENGTH, "A09_C09"))
    failed = batch.run_batch(jobs, missing, survey=survey)
    check(
        failed.counted(batch.STATUS_FAILED) == 4,
        "an image without the wanted channel is reported, not crashed on",
    )
    check(
        "no channel matching" in failed.outcomes[0].message,
        f"and the reason says which channel: {failed.outcomes[0].message}",
    )


def test_one_bad_image_does_not_end_the_run(directory: Path) -> None:
    print("one bad image out of many")

    class _Exploding(_StubBackend):
        name = "batch-stub"

        def segment(self, image, settings, diameter_px, anisotropy, device, progress=None,
                    channel_axis=None):
            if not self.calls:
                self.calls.append({})
                raise RuntimeError("out of memory on the first image")
            return super().segment(
                image, settings, diameter_px, anisotropy, device, progress, channel_axis
            )

    seg.register_backend(_Exploding())
    plate = _make_plate(directory)
    survey = batch.survey_plate(plate)
    jobs = batch.select_jobs(survey, acquisitions=[1])
    report = batch.run_batch(jobs, _stub_settings(), survey=survey)

    check(report.counted(batch.STATUS_FAILED) == 1, "the image that raised is marked failed")
    check(report.counted(batch.STATUS_DONE) == 3, "the rest of the plate still ran")
    check(
        "out of memory" in report.outcomes[0].message,
        f"the failure carries the message: {report.outcomes[0].message}",
    )
    seg.register_backend(_StubBackend())


def test_tables(directory: Path) -> None:
    print("object tables and the plate summary")

    try:
        import pandas  # noqa: F401
    except ImportError:
        print("  skip pandas is not installed")
        return

    seg.register_backend(_StubBackend())
    plate = _make_plate(directory)
    survey = batch.survey_plate(plate)
    jobs = batch.select_jobs(survey, wells=["B/02"], acquisitions=[1])
    tables = directory / "tables"
    settings = _stub_settings(write_tables=True, table_dir=tables)
    report = batch.run_batch(jobs, settings, survey=survey)

    per_image = tables / "B_02_0.csv"
    check(per_image.exists(), "the per-image object table is written")
    check(
        report.summary_path is not None and report.summary_path.exists(),
        "the plate summary is written",
    )
    rows = batch.summary_rows(report)
    check(
        rows[0]["well"] == "B/02" and rows[0]["acquisition"] == 1 and rows[0]["n_objects"] == 2,
        "the summary row names the well, the cycle and the count",
    )
    check(
        "median_diameter_um" in rows[0],
        "the summary carries the size distribution, not only the count",
    )

    default = _stub_settings(write_tables=True)
    check(
        default.table_path(jobs[0]).parent.name == f"{plate.stem}_nuclei_objects",
        "without a folder the tables land beside the plate, named after the label set",
    )


def test_which_channel_was_measured(directory: Path) -> None:
    print("the table says what it measured")

    seg.register_backend(_StubBackend())
    plate = _make_plate(directory)
    survey = batch.survey_plate(plate)
    jobs = batch.select_jobs(survey, wells=["B/02"], acquisitions=[1])

    plain = batch.run_batch(jobs, _stub_settings(), survey=survey)
    check(
        plain.outcomes[0].measured_channel == "Ab1_DAPI",
        f"an unqualified run records the segmented channel ({plain.outcomes[0].measured_channel})",
    )
    check(plain.outcomes[0].measured_extra == (), "and no others")

    elsewhere = batch.run_batch(
        jobs,
        _stub_settings(
            overwrite=True, measure=batch.ChannelPick(batch.BY_WAVELENGTH, "A02_C02")
        ),
        survey=survey,
    )
    check(
        elsewhere.outcomes[0].measured_channel == "Green488-x1",
        f"measuring elsewhere is recorded as such ({elsewhere.outcomes[0].measured_channel})",
    )

    # A 4i plate names the same stain differently every cycle, so a pick that
    # matched in cycle 1 can match nothing in cycle 7. Falling back in silence
    # would put the segmented channel's numbers under the reporter's heading.
    absent = batch.run_batch(
        jobs,
        _stub_settings(overwrite=True, measure=batch.ChannelPick(batch.BY_WAVELENGTH, "A09_C09")),
        survey=survey,
    )
    outcome = absent.outcomes[0]
    check(outcome.ok, "an unmatched measure channel does not fail the image")
    check(
        outcome.measured_channel == "Ab1_DAPI",
        "it falls back to the segmented channel, as it always did",
    )
    check(
        "no channel matching" in outcome.message.lower()
        and "segmented channel" in outcome.message,
        f"but says so, which is the whole difference: {outcome.message}",
    )


def test_measuring_every_channel(directory: Path) -> None:
    print("every channel in one run")

    try:
        import pandas as pd
    except ImportError:
        print("  skip pandas is not installed")
        return

    seg.register_backend(_StubBackend())
    plate = _make_plate(directory)
    survey = batch.survey_plate(plate)
    jobs = batch.select_jobs(survey, wells=["B/02"], acquisitions=[1])

    settings = _stub_settings(measure_all_channels=True, write_tables=True, table_dir=directory / "tables")
    report = batch.run_batch(jobs, settings, survey=survey)
    outcome = report.outcomes[0]

    check(outcome.ok, f"the run finishes ({outcome.message})")
    check(
        outcome.measured_extra == ("Green488-x1",),
        f"the other channel of the image was measured too ({outcome.measured_extra})",
    )

    frame = pd.read_csv(outcome.table_path)
    check(
        "Mean intensity (Green488-x1)" in frame.columns,
        f"and reaches the written table: {list(frame.columns)[-3:]}",
    )
    check(
        "Mean intensity" in frame.columns,
        "beside the segmented channel's own columns, which keep their old names",
    )
    check(
        frame["Mean intensity (Green488-x1)"].notna().all(),
        "with a number for every object",
    )
    check(
        not frame["Mean intensity"].equals(frame["Mean intensity (Green488-x1)"]),
        "and the two channels really are measured separately",
    )

    summary = pd.read_csv(report.summary_path)
    check(
        summary["measured_channel"].iloc[0] == "Ab1_DAPI",
        f"the plate summary names the channel too ({summary['measured_channel'].iloc[0]})",
    )
    check(
        summary["extra_channels"].iloc[0] == "Green488-x1",
        "and the extra ones, so a folder of tables can be read months later",
    )


def test_anndata_beside_the_tables(directory: Path) -> None:
    print("an .h5ad beside each table")

    try:
        import anndata as ad
    except ImportError:
        print("  skip anndata is not installed")
        return

    seg.register_backend(_StubBackend())
    plate = _make_plate(directory)
    survey = batch.survey_plate(plate)
    jobs = batch.select_jobs(survey, wells=["B/02"], acquisitions=[1])

    settings = _stub_settings(
        write_tables=True, write_anndata=True, table_dir=directory / "tables"
    )
    report = batch.run_batch(jobs, settings, survey=survey)
    outcome = report.outcomes[0]
    check(outcome.ok, f"the run finishes ({outcome.message})")

    csv = outcome.table_path
    h5ad = csv.with_suffix(".h5ad")
    check(csv.exists(), "the CSV is still written")
    check(h5ad.exists(), f"and an .h5ad beside it ({h5ad.name})")

    adata = ad.read_h5ad(h5ad)
    check(adata.n_obs == outcome.n_objects, f"one observation per object ({adata.n_obs})")
    check("spatial" in adata.obsm, "with the centroids in obsm['spatial'] for squidpy")
    check(
        adata.uns["microscopy_viewer"]["image"] == "B/02/0",
        f"and the image it came from ({adata.uns['microscopy_viewer']['image']})",
    )

    # Written from the measured numbers, not by reading the CSV back: a float that
    # has been through a text file is not the float that was measured.
    import pandas as pd

    frame = pd.read_csv(csv)
    feature = list(adata.var_names)[0]
    check(
        float(adata.X[0, 0]) == float(np.float32(frame[feature].iloc[0])),
        f"the matrix holds the measured values ({feature})",
    )

    plain = _stub_settings(overwrite=True, write_tables=True, table_dir=directory / "plain")
    plain_report = batch.run_batch(jobs, plain, survey=survey)
    check(
        not plain_report.outcomes[0].table_path.with_suffix(".h5ad").exists(),
        "a run that did not ask for one does not write it",
    )

    # One file for the whole run, rather than one per image.
    every = batch.select_jobs(survey, acquisitions=[1])
    one = _stub_settings(
        overwrite=True,
        write_tables=True,
        write_anndata=True,
        anndata_single_file=True,
        table_dir=directory / "one",
    )
    one_report = batch.run_batch(every, one, survey=survey)
    check(one_report.anndata_path is not None, "a combined file is written")
    combined = ad.read_h5ad(one_report.anndata_path)
    check(
        combined.n_obs == one_report.total_objects,
        f"holding every object of the run ({combined.n_obs} of {one_report.total_objects})",
    )
    check(
        combined.obs["image"].nunique() == len(every),
        f"tagged with the image each came from ({combined.obs['image'].nunique()} images)",
    )
    check(
        len(set(combined.obs_names)) == combined.n_obs,
        "with unique names, which label 1 in four wells is not",
    )
    check(
        not list((directory / "one").glob("*_0.h5ad")),
        "and no per-image files, since one file is what was asked for",
    )
    check(
        len(list((directory / "one").glob("*_0.csv"))) == len(every),
        "the per-image CSVs are still written either way",
    )
    check(
        all(outcome.frame is None for outcome in one_report.outcomes),
        "the frames it held to build the file are let go afterwards",
    )


def main() -> int:
    if not _has_zarr():
        print("zarr is not installed; the batch checks need it")
        return 0

    test_channel_matching()
    directory = Path(tempfile.mkdtemp(prefix="mv-batch-"))
    try:
        for name, test in (
            ("survey", test_survey),
            ("write", test_labels_are_written),
            ("read", test_labels_load_as_a_labels_layer),
            ("mosaic", test_labels_reach_the_plate_mosaic),
            ("run", test_a_run),
            ("resilience", test_one_bad_image_does_not_end_the_run),
            ("tables", test_tables),
            ("measured", test_which_channel_was_measured),
            ("allchannels", test_measuring_every_channel),
            ("anndata", test_anndata_beside_the_tables),
        ):
            case = directory / name
            case.mkdir()
            test(case)
    finally:
        shutil.rmtree(directory, ignore_errors=True)

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
