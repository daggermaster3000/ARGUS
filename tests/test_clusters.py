"""Checks for painting a clustering back onto the nuclei. No Qt, no display.

What is being checked is the mapping and the bookkeeping: that an object gets the
cluster its row says and not a neighbour's, that an object the clustering never
saw is left as background rather than guessed at, that what produced the
clustering travels with it into the store, and that removing clusterings removes
only clusterings — a segmentation is an hour of GPU and must survive a tidy-up
meant for something that takes forty seconds.

Run with::

    python tests/test_clusters.py
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from microscopy_viewer import batch, clusters  # noqa: E402

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


def _segmented(directory: Path):
    """A plate with a nuclei label set in every cycle-1 image."""
    survey = batch.survey_plate(_plate(directory))
    jobs = [job for job in survey.jobs if job.acquisition == 1]
    for index, job in enumerate(jobs):
        masks = np.zeros(job.level(0).shape[1:], dtype=np.int32)
        # Four objects per image, in a row, so a cluster map is easy to read back.
        for number in range(1, 5):
            masks[0, 4 + number * 4 : 7 + number * 4, 4:10] = number
        batch.write_labels(job, "nuclei", masks, level=0, overwrite=True)
    return batch.survey_plate(survey.path), jobs


# ---------------------------------------------------------------------------


def test_painting() -> None:
    print("label -> cluster")

    masks = np.zeros((1, 10, 10), dtype=np.int32)
    masks[0, 1:3, 1:3] = 1
    masks[0, 5:7, 5:7] = 2
    masks[0, 8:9, 8:9] = 7  # a label the clustering never saw

    painted, assigned = clusters.paint_clusters(masks, {1: 3, 2: 1})
    check(assigned == 2, f"two objects were assigned ({assigned})")
    check(
        set(np.unique(painted[masks == 1])) == {3},
        "an object takes its own cluster, whole",
    )
    check(set(np.unique(painted[masks == 2])) == {1}, "and so does the next one")
    check(
        set(np.unique(painted[masks == 7])) == {0},
        "an object the clustering never saw is left as background, not guessed into "
        "some 'other' cluster that does not exist",
    )
    check(
        np.array_equal(painted == 0, masks == 0) is False,
        "which means the painted footprint is smaller than the mask, as it should be",
    )
    check(painted.shape == masks.shape, "the grid is unchanged")
    check(painted.dtype == np.uint16, f"written small: {painted.dtype}")

    empty, none = clusters.paint_clusters(np.zeros((0,), dtype=np.int32), {1: 1})
    check(empty.size == 0 and none == 0, "an empty mask paints nothing rather than raising")

    stray, count = clusters.paint_clusters(masks, {99: 4})
    check(
        count == 0 and int(stray.max()) == 0,
        "a label that is not in the mask assigns nothing",
    )


def test_reading_assignments() -> None:
    print("a table of assignments")

    import pandas as pd

    frame = pd.DataFrame(
        {
            "label": [1, 2, 3, 1, 2],
            "cluster": ["b", "a", "b", "a", "a"],
            "image": ["B/02/0"] * 3 + ["C/05/0"] * 2,
        }
    )
    found = clusters.assignments_from_frame(frame)
    check(sorted(found) == ["B/02/0", "C/05/0"], f"split per image: {sorted(found)}")
    check(found["B/02/0"] == {1: 2, 2: 1, 3: 2}, f"cluster names become ids from 1: {found['B/02/0']}")
    check(
        found["C/05/0"][1] == 1,
        "and the same name is the same id in every image, or the plate would be "
        "coloured differently well by well",
    )

    names = clusters.cluster_names(frame["cluster"])
    check(names == ["a", "b"], f"the names come back in the order the ids were given: {names}")
    check(
        clusters.cluster_names(pd.Series(["10", "2", "1"])) == ["1", "2", "10"],
        "numbered clusters sort as numbers, so 2 comes before 10",
    )

    single = pd.DataFrame({"Label": [1, 2], "leiden": ["x", "y"]})
    out = clusters.assignments_from_frame(single, component="G/07/0")
    check(
        out == {"G/07/0": {1: 1, 2: 2}},
        f"a table with no image column takes the one it was given: {out}",
    )
    check(
        clusters.assignments_from_frame(single, "G/07/0")["G/07/0"][2] == 2,
        "and other spellings of the columns are accepted",
    )

    try:
        clusters.assignments_from_frame(pd.DataFrame({"a": [1]}))
        check(False, "a table with neither column is refused")
    except ValueError as exc:
        check("label column" in str(exc), f"a table with neither column is refused: {exc}")


def test_writing_into_the_plate(directory: Path) -> None:
    print("into the store")

    survey, jobs = _segmented(directory)
    job = jobs[0]
    run = clusters.ClusterRun(
        method="leiden",
        names=("a", "b"),
        source="nuclei",
        features=("Voxels", "Solidity"),
        resolution=0.6,
        n_objects=4,
        colors=("#1f77b4", "#ff7f0e"),
    )
    target, assigned = clusters.write_clusters(
        job, {1: 1, 2: 2, 3: 1}, run, name="phenotypes", source_labels="nuclei"
    )

    check(target.is_dir(), f"a label group is written ({target.name})")
    check(assigned == 3, f"three objects painted ({assigned})")

    from microscopy_viewer import explorer

    sets = explorer.label_sets(job)
    check("nuclei" in sets and "phenotypes" in sets, f"both label sets are listed: {sets}")

    import zarr

    painted = np.asarray(zarr.open(str(target), mode="r")["0"])
    masks = np.asarray(zarr.open(str(job.path / "labels" / "nuclei"), mode="r")["0"])
    check(painted.shape == masks.shape, "on the same grid as the nuclei")
    check(
        set(np.unique(painted[masks == 1])) == {1} and set(np.unique(painted[masks == 2])) == {2},
        "with each nucleus carrying its own cluster",
    )
    check(
        set(np.unique(painted[masks == 4])) == {0},
        "and the object that was not in the clustering left blank",
    )
    check(
        np.array_equal(masks, np.asarray(zarr.open(str(job.path / "labels" / "nuclei"), mode="r")["0"])),
        "the nuclei themselves are untouched — this is a second label set, not an edit",
    )

    # What produced it.
    group = zarr.open_group(str(target), mode="r")
    back = clusters.read_run(group)
    check(back is not None, "the clustering block is written")
    check(back.method == "leiden", f"with the method ({back.method})")
    check(back.source == "nuclei", f"and what it was painted onto ({back.source})")
    check(back.names == ("a", "b"), f"and the cluster names ({back.names})")
    check(back.features == ("Voxels", "Solidity"), "and the features it was computed from")
    check(back.created != "", f"and when ({back.created})")
    check("leiden" in back.describe(), f"which reads back as a sentence: {back.describe()}")

    marker = dict(group.attrs.get("image-label") or {})
    check(
        [entry["label-value"] for entry in marker.get("colors", [])] == [1, 2],
        "NGFF's own colours are written too, one per cluster",
    )
    check(
        marker["colors"][0]["rgba"] == [31, 119, 180, 255],
        f"as 0-255 RGBA, which is what the spec says: {marker['colors'][0]['rgba']}",
    )
    check(
        [entry["cluster"] for entry in marker.get("properties", [])] == ["a", "b"],
        "and the names, as label properties",
    )

    try:
        clusters.write_clusters(job, {1: 1}, run, name="x", source_labels="nothing")
        check(False, "painting onto a label set that is not there is refused")
    except FileNotFoundError as exc:
        check("nothing" in str(exc), f"painting onto a missing label set is refused: {exc}")


def test_the_plate_and_removing_them(directory: Path) -> None:
    print("a whole plate, and taking it back out")

    survey, jobs = _segmented(directory)
    run = clusters.ClusterRun(method="k-means", names=("a", "b"), source="nuclei", n_objects=8)
    assignments = {job.component: {1: 1, 2: 2} for job in jobs}
    assignments["Z/99/0"] = {1: 1}  # an image this plate does not have

    rows = clusters.write_plate_clusters(
        survey, assignments, run, name="phenotypes", source_labels="nuclei"
    )
    written = [row for row in rows if row[1] is not None]
    check(len(written) == len(jobs), f"every real image was written ({len(written)})")
    check(
        any(row[3] for row in rows if row[1] is None),
        "and the one that is not in the plate is reported rather than raising",
    )

    from microscopy_viewer import explorer

    found = clusters.clusterings_of(jobs[0])
    check(list(found) == ["phenotypes"], f"it is found as a clustering: {list(found)}")
    check(found["phenotypes"].method == "k-means", "with its method")
    check(
        "nuclei" not in clusters.clusterings_of(jobs[0]),
        "and the segmentation is not mistaken for one — it carries no clustering block",
    )

    removed = clusters.remove_clusterings(survey)
    check(len(removed) == len(jobs), f"every clustering is removed ({len(removed)})")
    check(
        all("nuclei" in explorer.label_sets(job) for job in jobs),
        "the segmentations survive, which is the whole point of only removing marked sets",
    )
    check(
        all("phenotypes" not in explorer.label_sets(job) for job in jobs),
        "and the clusterings are gone from the listing, not just from the disk",
    )
    check(clusters.remove_clusterings(survey) == [], "removing again removes nothing")


def test_it_loads_back_into_napari(directory: Path) -> None:
    print("and comes back as a Labels layer with its colours")

    survey, jobs = _segmented(directory)
    job = jobs[0]
    run = clusters.ClusterRun(
        method="leiden", names=("a", "b"), source="nuclei", colors=("#1f77b4", "#ff7f0e")
    )
    clusters.write_clusters(job, {1: 1, 2: 2}, run, name="phenotypes", source_labels="nuclei")

    from microscopy_viewer import explorer

    specs, problems = explorer.specs_for([job], plate_path=survey.path)
    check(not problems, f"the image still reads ({problems})")

    painted = [spec for spec in specs if spec.name.endswith("phenotypes")]
    check(len(painted) == 1, f"the clustering is one layer ({[s.name for s in specs]})")
    spec = painted[0]
    check(spec.layer_type == "labels", "loaded as Labels, not as an image")
    check(
        len(spec.label_colors) == 2,
        f"carrying the colours it was written with ({spec.label_colors})",
    )
    check(
        tuple(round(v, 3) for v in spec.label_colors[1][:3]) == (0.122, 0.467, 0.706),
        f"converted from NGFF's 0-255 to napari's 0-1: {spec.label_colors[1]}",
    )
    check(
        "leiden" in spec.metadata.extra.get("Clustering", ""),
        f"and saying what made it: {spec.metadata.extra.get('Clustering')}",
    )

    nuclei = [s for s in specs if s.name.endswith("nuclei")][0]
    check(
        nuclei.label_colors == {},
        "while a plain segmentation carries none, and napari colours it as it always has",
    )

    kwargs = spec.to_kwargs()
    check(
        "colormap" in kwargs,
        "a labels layer with its own colours does pass a colormap to add_labels",
    )
    check(
        "contrast_limits" not in kwargs and "blending" not in kwargs,
        "but still none of the arguments add_labels would refuse",
    )
    check("colormap" not in nuclei.to_kwargs(), "and a plain one passes none")


def main() -> int:
    if not _has_zarr():
        print("zarr is not installed; the clustering checks need it")
        return 0

    test_painting()
    print()
    try:
        test_reading_assignments()
    except ImportError:
        print("  skip pandas is not installed")
    print()

    directory = Path(tempfile.mkdtemp(prefix="mv-clusters-"))
    try:
        for name, test in (
            ("write", test_writing_into_the_plate),
            ("plate", test_the_plate_and_removing_them),
            ("load", test_it_loads_back_into_napari),
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
