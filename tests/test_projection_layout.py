"""Batch projection: where the projections go, and overviews left out."""

from __future__ import annotations

import sys
from pathlib import Path

import h5py

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import make_sample_data as msd  # noqa: E402
from microscopy_viewer import projection as pj  # noqa: E402


def _place(path, x0, y0, size):
    with h5py.File(path, "r+") as handle:
        image = handle["DataSetInfo"]["Image"]
        for key, value in (("ExtMin0", x0), ("ExtMax0", x0 + size), ("ExtMin1", y0), ("ExtMax1", y0 + size)):
            msd._write_attr(image, key, value)
    return path


def test_projections_keep_the_subfolders_or_pool_into_one():
    root, out = Path("/exp"), Path("/exp/MIP")
    source = root / "day1" / "DMSO" / "fish1.ims"
    kept = pj.ProjectionOptions(output_dir=out, input_root=root, keep_structure=True)
    pooled = pj.ProjectionOptions(output_dir=out, input_root=root, keep_structure=False)
    beside = pj.ProjectionOptions(output_dir=None, input_root=root, keep_structure=True)
    assert pj.output_path(source, kept) == out / "day1" / "DMSO" / "fish1_MIP.tif"
    assert pj.output_path(source, pooled) == out / "fish1_MIP.tif"
    assert pj.output_path(source, beside) == root / "day1" / "DMSO" / "fish1_MIP.tif"


def test_pooled_files_of_the_same_name_do_not_overwrite_each_other():
    root = Path("/exp")
    files = [root / "DMSO" / "fish1.ims", root / "drug" / "fish1.ims", root / "drug" / "fish2.ims"]
    pooled = pj.ProjectionOptions(output_dir=root / "MIP", input_root=root, keep_structure=False)
    targets = pj.output_paths(files, pooled)
    assert [t.name for t in targets.values()] == ["fish1_DMSO_MIP.tif", "fish1_drug_MIP.tif", "fish2_MIP.tif"]
    kept = pj.ProjectionOptions(output_dir=root / "MIP", input_root=root, keep_structure=True)
    assert len(set(pj.output_paths(files, kept).values())) == 3


def test_overviews_are_left_out_of_the_batch(tmp_path):
    fields = [_place(msd.write_ims_2d(tmp_path / f"slide_F{i:04d}.ims", shape=(1, 1, 48, 64)), 1000 * i, 0, 1000)
              for i in range(3)]
    fish = _place(msd.write_ims(tmp_path / "fish1.ims", shape=(1, 5, 32, 32), n_channels=1), 200, 200, 200)
    stacks, skipped = pj.without_overviews([*fields, fish])
    assert stacks == [fish]
    assert sorted(skipped) == sorted(fields)


def test_a_batch_writes_into_the_mirrored_folders(tmp_path):
    for folder in ("DMSO", "drug"):
        (tmp_path / folder).mkdir()
        msd.write_ims(tmp_path / folder / "fish1.ims", shape=(1, 4, 32, 32), n_channels=1)
    files = sorted(tmp_path.glob("*/fish1.ims"))
    options = pj.ProjectionOptions(output_dir=tmp_path / "MIP", input_root=tmp_path, keep_structure=True)
    outcomes = pj.run_batch(files, options)
    assert all(o.ok and not o.skipped for o in outcomes)
    assert (tmp_path / "MIP" / "DMSO" / "fish1_MIP.tif").exists()
    assert (tmp_path / "MIP" / "drug" / "fish1_MIP.tif").exists()
