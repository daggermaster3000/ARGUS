"""Checks for the plate explorer engine. No Qt, no display.

What is being checked is what the panel shows and what it opens: which images a
plate holds, which of them already carry a segmentation, that the miniature comes
off the bottom of the pyramid rather than the top, that the layers built for a
chosen image carry its labels and are named the way a table can find them, and
that the table folders beside a plate are the ones listed.

Run with::

    python tests/test_explorer.py
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from microscopy_viewer import batch, explorer  # noqa: E402

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


def _plate(directory: Path) -> Path:
    from make_sample_data import write_fractal_plate

    return write_fractal_plate(directory / "plate.zarr")


# ---------------------------------------------------------------------------


def test_listing(directory: Path) -> None:
    print("what is in the plate")

    survey = explorer.survey_plate(_plate(directory))
    rows = explorer.describe_plate(survey)
    check(len(rows) == len(survey.jobs), f"a row per image ({len(rows)})")

    first = rows[0]
    check(first["well"] == "B/02", f"the well is named as a path ({first['well']})")
    check(first["acquisition"] == 1, f"and the cycle as its acquisition id ({first['acquisition']})")
    check(first["channels"] == 2, f"the channel count is read ({first['channels']})")
    check(
        "Ab1_DAPI" in first["channel_names"],
        f"and so are the names, for the tooltip: {first['channel_names']}",
    )
    check(
        first["shape"] == "2 x 1 x 32 x 48",
        f"the size is the full-resolution one ({first['shape']})",
    )
    check(first["segmentations"] == "", "an image nothing has segmented says so with a blank")

    # The cycles of one well are separate images, not fields of one.
    b02 = [row for row in rows if row["well"] == "B/02"]
    check(len(b02) == 2, f"each 4i cycle is its own row ({len(b02)})")
    check(
        {row["acquisition"] for row in b02} == {1, 2},
        "and the rows are told apart by their cycle",
    )


def test_segmentations_show_up(directory: Path) -> None:
    print("segmentations already in the store")

    survey = explorer.survey_plate(_plate(directory))
    job = survey.jobs[0]
    check(explorer.label_sets(job) == (), "nothing is claimed before anything is written")

    masks = np.zeros(job.level(0).shape[1:], dtype=np.int32)
    masks[0, 4:12, 4:12] = 1
    batch.write_labels(job, "nuclei", masks, level=0)

    check(explorer.label_sets(job) == ("nuclei",), f"a written set is listed: {explorer.label_sets(job)}")
    row = explorer.describe_job(job)
    check(row["n_segmentations"] == 1, "and counted")
    check(row["segmentations"] == "nuclei", f"by name: {row['segmentations']}")
    check(
        explorer.label_sets(survey.jobs[1]) == (),
        "while the next image is unaffected — the label set lives inside its own image",
    )


def test_the_miniature(directory: Path) -> None:
    print("the miniature")

    survey = explorer.survey_plate(_plate(directory))
    job = survey.jobs[0]
    full = job.level(0).shape
    smallest = job.level(len(job.levels) - 1).shape

    plane = explorer.thumbnail(job)
    check(plane.ndim == 2, f"a preview is one plane, whatever the image is ({plane.ndim}D)")
    check(
        max(plane.shape) <= max(smallest[-2:]),
        f"read from the bottom of the pyramid ({plane.shape} against a full size of {full[-2:]})",
    )
    check(
        max(plane.shape) <= explorer.THUMBNAIL_PX,
        f"and subsampled to something a list can draw ({plane.shape})",
    )

    other = explorer.thumbnail(job, channel_index=1)
    check(other.shape == plane.shape, "every channel previews at the same size")
    check(
        not np.array_equal(other, plane),
        "and a different channel really is a different picture",
    )
    check(
        explorer.thumbnail(job, channel_index=99).shape == plane.shape,
        "a channel index past the end is clamped rather than raising",
    )

    grey = explorer.stretch(plane)
    check(grey.dtype == np.uint8, f"the stretch gives 8-bit grey ({grey.dtype})")
    check(grey.max() > grey.min(), "with something actually visible in it")
    flat = explorer.stretch(np.full((4, 4), 7.0))
    check(
        flat.max() == flat.min() == 0,
        "a blank field comes back black rather than dividing by zero",
    )


def test_opening(directory: Path) -> None:
    print("building the layers")

    plate = _plate(directory)
    survey = explorer.survey_plate(plate)
    job = survey.jobs[0]
    masks = np.zeros(job.level(0).shape[1:], dtype=np.int32)
    masks[0, 4:12, 4:12] = 1
    batch.write_labels(job, "nuclei", masks, level=0)

    specs, problems = explorer.specs_for([job], plate_path=plate)
    check(not problems, f"a readable image reports no problem ({problems})")
    names = [spec.name for spec in specs]
    check(len(specs) == 3, f"a layer per channel plus the label set ({len(specs)}): {names}")
    check(
        sum(1 for spec in specs if spec.layer_type == "labels") == 1,
        "the label set comes back as a Labels layer, not an image",
    )
    check(
        all(name.startswith("B/02 :: cycle 1") for name in names),
        f"named by well and cycle, which is what a table matches on: {names}",
    )

    from microscopy_viewer import analysis

    check(
        analysis.match_layer("B_02_0.csv", names) is not None,
        "so a batch run's own table finds its layer",
    )

    plain, _problems = explorer.specs_for([job], plate_path=plate, with_labels=False)
    check(len(plain) == 2, f"asked for channels only, the labels are left out ({len(plain)})")

    both, _problems = explorer.specs_for(survey.jobs[:2], plate_path=plate)
    check(
        len({spec.name for spec in both}) == len(both),
        "two cycles of one well do not collide — the cycle is in the name",
    )


def test_finding_the_tables(directory: Path) -> None:
    print("the analysis folders beside the plate")

    plate = directory / "AssayPlate.zarr"
    plate.mkdir()
    (plate / "B").mkdir()

    for name in ("AssayPlate_nuclei_objects", "AssayPlate_membranes_objects"):
        folder = directory / name
        folder.mkdir()
        (folder / "B_02_0.csv").write_text("Label,Area\n1,5\n", encoding="utf-8")
        (folder / "nuclei_summary.csv").write_text("well,n\nB/02,1\n", encoding="utf-8")
    other = directory / "somebody elses tables"
    other.mkdir()
    (other / "whatever.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    empty = directory / "notes"
    empty.mkdir()
    (empty / "readme.txt").write_text("nothing here", encoding="utf-8")

    folders = explorer.analysis_folders(plate)
    names = [folder.name for folder in folders]
    check(len(folders) == 3, f"folders holding tables are found ({names})")
    check(
        empty.name not in names,
        "a folder with no table in it is not offered",
    )
    check(
        names[0].startswith("AssayPlate") and names[1].startswith("AssayPlate"),
        f"the plate's own output comes first: {names}",
    )
    check(
        names[-1] == other.name,
        f"and an unrelated neighbour last rather than not at all: {names}",
    )
    check(
        plate.name not in names,
        "the plate itself is not listed as a folder of tables",
    )

    tables = explorer.analysis_tables(folders[0])
    check(len(tables) == 2, f"both tables in the folder ({[t.name for t in tables]})")
    check(
        tables[-1].stem.endswith("_summary"),
        f"the run summary sorts last — it is one row per image, not per object: {tables[-1].name}",
    )
    check(explorer.analysis_folders(directory / "nowhere") == [], "a missing plate lists nothing")


def test_tables_find_their_image(directory: Path) -> None:
    print("matching a table to the image it measured")

    plate = _plate(directory)
    survey = explorer.survey_plate(plate)
    folder = directory / "plate_nuclei_objects"
    folder.mkdir()
    for stem in ("B_02_0", "B_03_1", "C_02_0"):
        (folder / f"{stem}.csv").write_text("Label,Area\n1,5\n", encoding="utf-8")
    (folder / "nuclei_summary.csv").write_text("well,n\nB/02,1\n", encoding="utf-8")

    check(explorer.table_key("B/02/0") == "B_02_0", "a component flattens to a table stem")
    check(explorer.component_of("B_02_0.csv") == "B/02/0", "and reads back out of the file name")

    index = explorer.table_index([folder])
    check(len(index) == 4, f"every table is indexed once ({sorted(index)})")

    def job_for(component):
        return next(job for job in survey.jobs if job.component == component)

    own = explorer.tables_for(job_for("B/02/0"), index)
    check([path.name for path in own] == ["B_02_0.csv"], f"the image gets its own table ({own})")
    check(
        explorer.tables_for(job_for("C/02/0"), index)[0].name == "C_02_0.csv",
        "and a different well gets a different one, not the first in the folder",
    )
    check(
        explorer.tables_for(job_for("B/02/1"), index) == [],
        "a cycle nothing measured claims nothing",
    )
    check(
        explorer.tables_for(job_for("C/03/0"), index) == [],
        "and neither does a well nothing measured — a near miss is still a miss",
    )

    # The cycle-1 table does describe the cells in cycle 2: a 4i plate images the
    # same cells every cycle. It is still not this image's table, so it is offered
    # separately rather than mixed in.
    related = explorer.related_tables(job_for("B/02/1"), index)
    check(
        [path.name for path in related] == ["B_02_0.csv"],
        f"another cycle of the same well is offered separately ({related})",
    )
    check(
        explorer.related_tables(job_for("B/02/0"), index) == [],
        "and the image's own table is not offered to it twice",
    )
    check(
        explorer.related_tables(job_for("C/02/0"), index) == [],
        "a well whose only table is its own has nothing related",
    )

    # A pipeline that assumes one image per well writes the well name alone.
    (folder / "C_03.csv").write_text("Label,Area\n1,5\n", encoding="utf-8")
    well_index = explorer.table_index([folder])
    check(
        explorer.tables_for(job_for("C/03/0"), well_index)[0].name == "C_03.csv",
        "a table named after the well alone still finds the well's image",
    )

    rows = explorer.describe_plate(explorer.survey_plate(plate))
    measured = {row["component"] for row in rows if row["n_tables"]}
    check(
        measured == {"B/02/0", "B/03/1", "C/02/0", "C/03/0", "C/03/1"},
        f"describe_plate finds the folder beside the plate by itself ({sorted(measured)})",
    )
    check(
        "B/02/1" not in measured,
        "and marks only the images a table was actually written for",
    )
    check(
        {"C/03/0", "C/03/1"} <= measured,
        "a well-level table marks every cycle of that well — it names no one of them",
    )
    row = explorer.describe_job(job_for("B/02/0"), index=index)
    check(row["n_tables"] == 1, "given an index, the row counts the tables")
    check(
        row["tables"] == folder.name,
        f"and names the folder they came from, which is what tells two runs apart: {row['tables']}",
    )
    check(
        row["table_paths"] == own,
        "with the paths kept, so the panel can open one without searching again",
    )


def main() -> int:
    if not _has_zarr():
        print("zarr is not installed; the explorer checks need it")
        return 0

    directory = Path(tempfile.mkdtemp(prefix="mv-explorer-"))
    try:
        for name, test in (
            ("listing", test_listing),
            ("labels", test_segmentations_show_up),
            ("miniature", test_the_miniature),
            ("opening", test_opening),
            ("tables", test_finding_the_tables),
            ("matching", test_tables_find_their_image),
        ):
            case = directory / name
            case.mkdir()
            test(case)
            print()
    finally:
        shutil.rmtree(directory, ignore_errors=True)

    if _failures:
        print(f"{len(_failures)} check(s) failed:")
        for failure in _failures:
            print(f"  - {failure}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
