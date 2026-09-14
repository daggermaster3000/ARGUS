"""Checks for batch maximum projection and the TIFF / Imaris writers. No Qt.

The writers are checked by reading back what they wrote with this project's own
readers, which is the strongest check available without Fiji or Imaris to hand:
the readers were written against real files from those programs, so a file they
parse correctly is one built the way those files are built.

Run with::

    python tests/test_projection.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from microscopy_viewer import projection as pj  # noqa: E402
from microscopy_viewer import writers  # noqa: E402

_failures: list[str] = []


def check(condition: bool, message: str) -> None:
    if condition:
        print(f"  ok   {message}")
    else:
        print(f"  FAIL {message}")
        _failures.append(message)


def _read_back(path):
    """Load a file, materialise it, and give the handle back.

    The Imaris reader keeps its ``h5py.File`` open for the life of the process so
    its lazy arrays stay readable. That is right in the viewer and wrong in a
    temporary folder, which Windows then refuses to delete — so these checks pull
    the pixels into memory (they are tiny) and release the file straight away.
    """
    from microscopy_viewer.loaders import load_path, release

    specs = load_path(path)
    for spec in specs:
        spec.data = np.asarray(spec.data[0] if spec.multiscale else spec.data)
        spec.multiscale = False
    release(path)
    return specs


def _planes(timepoints=1, channels=2, height=16, width=20) -> np.ndarray:
    rng = np.random.default_rng(0)
    return (rng.random((timepoints, channels, height, width)) * 4000).astype(np.uint16)


# ---------------------------------------------------------------------------
# Choosing channels
# ---------------------------------------------------------------------------


def test_channel_selection() -> None:
    print("choosing channels")
    names = ["Confocal - GFP", "Confocal - DAPI", "Confocal - 647"]

    check(pj.select_channels(names, ())[0] == [0, 1, 2], "nothing asked for means all of them")
    check(pj.select_channels(names, ("dapi",))[0] == [1], "matched by name, case-insensitively")
    check(pj.select_channels(names, ("647", "gfp"))[0] == [0, 2], "several names, in file order")
    check(pj.select_channels(names, (2,))[0] == [2], "an integer is an index")

    # Order is the file's, not the request's: channel 0 of the output has to be
    # channel 0 of the file or the colours end up on the wrong stains.
    check(
        pj.select_channels(names, ("647", "gfp"))[0] == sorted(pj.select_channels(names, ("gfp", "647"))[0]),
        "asking in a different order gives the same order out",
    )

    indices, unmatched = pj.select_channels(names, ("dapi", "mCherry"))
    check(indices == [1], "what matched is still used")
    check(unmatched == ["mCherry"], f"and what did not is reported ({unmatched})")
    check(pj.select_channels(names, (9,))[1] == ["9"], "an index past the end is reported too")


# ---------------------------------------------------------------------------
# Projecting
# ---------------------------------------------------------------------------


def test_projection_collapses_z_only() -> None:
    print("projecting over Z")
    check(pj.z_axis_of("ZYX", 3) == 0, "a 3D stack projects over its first axis")
    check(pj.z_axis_of("TZYX", 4) == 1, "a time series over its second")
    check(pj.z_axis_of("", 3) == 0, "with no axis string, a 3D array is taken for ZYX")
    check(pj.z_axis_of("", 4) == 1, "and a 4D one for TZYX")
    check(pj.z_axis_of("TYX", 3) is None, "a stack with no Z has nothing to project")
    # Believing the axis string in both directions matters: guessing a Z here
    # would project a time-lapse over time and return a single frame.
    series = np.arange(3 * 5 * 6, dtype=np.uint16).reshape(3, 5, 6)
    check(
        np.array_equal(pj.project_array(series, "TYX"), series),
        "a TYX stack comes through with its timepoints intact",
    )

    volume = np.arange(4 * 5 * 6, dtype=np.uint16).reshape(4, 5, 6)
    flat = pj.project_array(volume, "ZYX")
    check(flat.shape == (1, 5, 6), f"a stack comes back with a time axis ({flat.shape})")
    check(np.array_equal(flat[0], volume.max(axis=0)), "and holds the brightest voxel of each column")

    series = np.arange(3 * 4 * 5 * 6, dtype=np.uint16).reshape(3, 4, 5, 6)
    moving = pj.project_array(series, "TZYX")
    check(moving.shape == (3, 5, 6), f"a time series keeps its timepoints ({moving.shape})")
    check(np.array_equal(moving, series.max(axis=1)), "each projected over Z alone, never over time")

    flatter = pj.project_array(np.zeros((5, 6), np.uint16), "YX")
    check(flatter.shape == (1, 5, 6), "a plane is already its own projection")


def test_projection_is_lazy() -> None:
    """The reduction must go through dask, not through a materialised array.

    A 271 x 2040 x 2040 channel is 2.2 GB in memory and about 8 MB projected. If
    this ever starts calling ``np.asarray`` on the input first, a folder of them
    stops fitting in RAM -- and it would still pass every other check here.
    """
    print("projecting without loading the stack")
    try:
        import dask.array as da
    except Exception:
        check(True, "dask is not installed; skipped")
        return

    volume = da.random.randint(0, 4000, size=(8, 32, 32), chunks=(2, 16, 16), dtype=np.uint16)
    computed: list[int] = []
    original = volume.max

    def _counting_max(*args, **kwargs):
        computed.append(1)
        return original(*args, **kwargs)

    volume.max = _counting_max  # type: ignore[method-assign]
    result = pj.project_array(volume, "ZYX")
    check(computed == [1], "the reduction was asked of the dask array itself")
    check(result.shape == (1, 32, 32), f"and only the plane came back ({result.shape})")
    check(isinstance(result, np.ndarray), "materialised once, at the end")


# ---------------------------------------------------------------------------
# Naming and not clobbering
# ---------------------------------------------------------------------------


def test_output_naming() -> None:
    print("where the output goes")
    source = Path("/data/day1/fish_1.ims")

    beside = pj.output_path(source, pj.ProjectionOptions(fmt=".tif"))
    check(beside.name == "fish_1_MIP.tif", f"suffixed beside the source ({beside.name})")
    check(beside.parent == source.parent, "in the source's own folder when none is given")

    elsewhere = pj.output_path(source, pj.ProjectionOptions(output_dir=Path("/out"), fmt=".ims"))
    check(elsewhere == Path("/out/fish_1_MIP.ims"), f"or in the folder given ({elsewhere})")

    plain = pj.output_path(source, pj.ProjectionOptions(fmt="tif"))
    check(plain.suffix == ".tif", "a format without its dot still works")


def test_a_projection_never_eats_its_source() -> None:
    print("not overwriting the source")
    with tempfile.TemporaryDirectory() as directory:
        folder = Path(directory)
        source = folder / "fish_1.ims"
        writers.write_ims(source, _planes(), (2.0, 0.3, 0.3), ["GFP", "mCherry"])

        # No suffix, no output folder, same format: the output path *is* the input.
        outcome = pj.project_file(source, pj.ProjectionOptions(fmt=".ims", suffix=""))
        check(not outcome.ok, "writing onto the source is refused")
        check("overwrite its own source" in outcome.error, f"and says why ({outcome.error[:50]})")
        check(source.stat().st_size > 0, "the source is still there")

        # An existing output is left alone unless overwriting was asked for.
        first = pj.project_file(source, pj.ProjectionOptions(output_dir=folder, fmt=".tif"))
        check(first.ok and not first.skipped, "the first run writes")
        again = pj.project_file(source, pj.ProjectionOptions(output_dir=folder, fmt=".tif"))
        check(again.skipped, "the second run skips what is already there")
        forced = pj.project_file(
            source, pj.ProjectionOptions(output_dir=folder, fmt=".tif", overwrite=True)
        )
        check(forced.ok and not forced.skipped, "unless overwriting was asked for")


# ---------------------------------------------------------------------------
# The writers
# ---------------------------------------------------------------------------


def test_imaris_attributes_are_character_arrays() -> None:
    """Imaris stores every attribute as |S1. Strings are invisible to it.

    Getting this wrong does not fail: it writes a file whose every attribute is
    garbage, so the image has no name, no channels and no calibration. It is
    checked directly because a round trip through a tolerant reader can hide it.
    """
    print("Imaris attribute encoding")
    import h5py

    encoded = writers._char("2040")
    check(encoded.dtype == np.dtype("S1"), f"single characters, not a string ({encoded.dtype})")
    check(
        b"".join(encoded.tolist()).decode() == "2040",
        f"and they spell what went in ({b''.join(encoded.tolist())!r})",
    )
    check(writers._char("").shape == (0,), "an empty value is an empty array, not a crash")

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "x.ims"
        writers.write_ims(path, _planes(), (2.0, 0.3, 0.3), ["GFP", "mCherry"])
        with h5py.File(str(path), "r") as handle:
            raw = handle["DataSetInfo/Image"].attrs["X"]
            check(raw.dtype == np.dtype("S1"), f"as written into the file ({raw.dtype})")
            check(b"".join(raw.tolist()).decode() == "20", "spelling the real width")


def test_writers_round_trip() -> None:
    print("writing and reading back")
    planes = _planes(timepoints=2, channels=2)
    names = ["Confocal - GFP", "Confocal - mCherry"]
    colors = [(0.0, 1.0, 0.0), (1.0, 0.0, 0.0)]
    voxel = (2.0, 0.325, 0.325)
    extent = (100.0, 999.0, 200.0, 999.0, 300.0, 999.0)  # only the origin is used

    with tempfile.TemporaryDirectory() as directory:
        for suffix in (".tif", ".ims"):
            path = Path(directory) / f"mip{suffix}"
            writers.write_stack(path, planes, voxel, names, colors, extent, "fish_1")
            specs = _read_back(path)

            check(len(specs) == 2, f"{suffix}: both channels came back ({len(specs)})")
            check(
                [spec.channel_name for spec in specs] == names,
                f"{suffix}: with their names ({[s.channel_name for s in specs]})",
            )
            for index, spec in enumerate(specs):
                check(
                    np.array_equal(np.asarray(spec.data), planes[:, index]),
                    f"{suffix}: channel {index} is pixel-for-pixel what went in",
                )
            lateral = tuple(round(float(v), 4) for v in specs[0].scale)[-2:]
            check(lateral == (0.325, 0.325), f"{suffix}: the pixel size survived ({lateral})")

        # Imaris carries what TIFF has no room for.
        specs = _read_back(Path(directory) / "mip.ims")
        meta = specs[0].metadata
        check(
            meta.stage_extent is not None and abs(meta.stage_extent[0] - 100.0) < 1e-6,
            f"ims: the stage origin came through ({meta.stage_extent})",
        )
        check(abs(float(meta.z_step_um) - 2.0) < 1e-6, f"ims: and the Z step ({meta.z_step_um})")
        check(
            [tuple(round(c, 3) for c in spec.color) for spec in specs] == colors,
            f"ims: and the channel colours ({[s.color for s in specs]})",
        )


def test_the_extent_spans_the_array_written() -> None:
    """Imaris stores no pixel size; readers divide the extent by the voxel count.

    So the extent has to describe the array being written, not the array it came
    from. Copying a source's far corner onto a cropped or decimated image puts a
    plausible stage position on a silently wrong calibration.
    """
    print("extents describe what was written")
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "crop.ims"
        # A 20 x 16 image, but an extent copied from a 2040-wide field.
        writers.write_ims(
            path,
            _planes(height=16, width=20),
            (2.0, 0.325, 0.325),
            ["GFP", "mCherry"],
            stage_extent=(1000.0, 1621.0, 2000.0, 2621.0, 50.0, 131.0),
        )
        spec = _read_back(path)[0]
        lateral = tuple(round(float(v), 4) for v in spec.scale)[-2:]
        check(lateral == (0.325, 0.325), f"the pixel size is the real one ({lateral})")
        check(
            abs(spec.metadata.stage_extent[0] - 1000.0) < 1e-6,
            "while the stage origin is still the source's",
        )


# ---------------------------------------------------------------------------
# The batch
# ---------------------------------------------------------------------------


def test_batch_keeps_going_and_can_be_stopped() -> None:
    print("the batch")
    with tempfile.TemporaryDirectory() as directory:
        folder = Path(directory)
        out = folder / "out"
        sources = []
        for index in range(4):
            source = folder / f"fish_{index}.ims"
            writers.write_ims(source, _planes(), (2.0, 0.3, 0.3), ["GFP", "mCherry"])
            sources.append(source)
        broken = folder / "broken.ims"
        broken.write_bytes(b"not an HDF5 file at all")
        sources.insert(2, broken)

        messages: list[str] = []
        outcomes = pj.run_batch(
            sources, pj.ProjectionOptions(output_dir=out, fmt=".tif"), progress=messages.append
        )
        check(len(outcomes) == 5, f"every file gets an outcome ({len(outcomes)})")
        good = [outcome for outcome in outcomes if outcome.ok]
        check(len(good) == 4, f"the four readable ones were projected ({len(good)})")
        bad = [outcome for outcome in outcomes if not outcome.ok]
        check(len(bad) == 1 and bad[0].name == "broken", "and the unreadable one is reported")
        check(
            len(list(out.glob("*.tif"))) == 4,
            f"one output per readable file ({len(list(out.glob('*.tif')))})",
        )
        check(any("broken" in text for text in messages), "progress names the file that failed")
        check("1 had problems" in pj.summarise(outcomes), f"and the summary says so ({pj.summarise(outcomes)})")

        # Stopping leaves what it finished intact.
        out2 = folder / "out2"
        seen: list[str] = []
        outcomes = pj.run_batch(
            sources,
            pj.ProjectionOptions(output_dir=out2, fmt=".tif"),
            progress=lambda text: seen.append(text),
            should_cancel=lambda: len(seen) >= 4,
        )
        check(len(outcomes) < 5, f"it stopped early ({len(outcomes)} of 5)")
        check(all(o.ok or o.error for o in outcomes), "and what it did finish is complete")


def test_batch_selects_channels_per_file() -> None:
    print("channels across a batch")
    with tempfile.TemporaryDirectory() as directory:
        folder = Path(directory)
        out = folder / "out"
        # Same two stains, opposite channel order -- the case an index gets wrong.
        writers.write_ims(folder / "a.ims", _planes(), (2.0, 0.3, 0.3), ["GFP", "mCherry"])
        writers.write_ims(folder / "b.ims", _planes(), (2.0, 0.3, 0.3), ["mCherry", "GFP"])

        outcomes = pj.run_batch(
            sorted(folder.glob("*.ims")),
            pj.ProjectionOptions(output_dir=out, fmt=".tif", channels=("mcherry",)),
        )
        check(all(outcome.ok for outcome in outcomes), "both files projected")
        check(
            [outcome.channels for outcome in outcomes] == [["mCherry"], ["mCherry"]],
            f"the same stain out of both, whatever its position ({[o.channels for o in outcomes]})",
        )

        missing = pj.run_batch(
            [folder / "a.ims"],
            pj.ProjectionOptions(output_dir=out, fmt=".tif", suffix="_x", channels=("dapi",)),
        )
        check(not missing[0].ok, "a channel no file has is an error, not an empty file")
        check("no channel matching" in missing[0].error, f"named plainly ({missing[0].error[:40]})")


def main() -> int:
    for test in (
        test_channel_selection,
        test_projection_collapses_z_only,
        test_projection_is_lazy,
        test_output_naming,
        test_a_projection_never_eats_its_source,
        test_imaris_attributes_are_character_arrays,
        test_writers_round_trip,
        test_the_extent_spans_the_array_written,
        test_batch_keeps_going_and_can_be_stopped,
        test_batch_selects_channels_per_file,
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
