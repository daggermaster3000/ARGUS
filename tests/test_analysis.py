"""Checks for the per-region analysis workbook.

The geometry is checked against shapes whose answers are known by hand; the file
side against a small Imaris-shaped container carrying a label map and outlines,
the same way ``test_experiment`` builds its own.

Run with::

    python tests/test_analysis.py
"""

from __future__ import annotations

import math
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from microscopy_viewer import analysis as an  # noqa: E402
from microscopy_viewer import ims_store as store  # noqa: E402
from microscopy_viewer import regions as rg  # noqa: E402
from microscopy_viewer.loaders import release  # noqa: E402

_failures: list[str] = []


def check(condition: bool, message: str) -> None:
    if condition:
        print(f"  ok   {message}")
    else:
        print(f"  FAIL {message}")
        _failures.append(message)


def _box(top, bottom, left, right) -> np.ndarray:
    return np.array([[top, left], [top, right], [bottom, right], [bottom, left]], dtype=float)


def test_region_shape() -> None:
    print("region morphometrics")
    square = an.region_shape(rg.Region("square", _box(0, 10, 0, 10)), depth_um=2.0)
    check(abs(square.area_um2 - 100.0) < 1e-9, f"area ({square.area_um2})")
    check(abs(square.perimeter_um - 40.0) < 1e-9, f"perimeter ({square.perimeter_um})")
    check(abs(square.circularity - math.pi / 4) < 1e-9, f"circularity ({square.circularity:.4f})")
    check(abs(square.solidity - 1.0) < 1e-9, f"a square is convex ({square.solidity})")
    check(abs(square.aspect_ratio - 1.0) < 1e-9, f"and not elongated ({square.aspect_ratio})")
    check((square.centroid_y_um, square.centroid_x_um) == (5.0, 5.0), "centroid in the middle")
    check(abs(square.volume_um3 - 200.0) < 1e-9, f"volume is area × depth ({square.volume_um3})")

    wide = an.region_shape(rg.Region("wide", _box(0, 10, 0, 40)))
    check(abs(wide.aspect_ratio - 4.0) < 1e-9, f"a 4:1 rectangle has aspect 4 ({wide.aspect_ratio})")
    check(abs(abs(wide.orientation_deg)) < 1e-6, f"lying along X ({wide.orientation_deg})")
    # A rectangle's equal-moment ellipse has axes sqrt(3) times its half-sides × 2.
    check(abs(wide.major_axis_um - 40.0 * 2 / math.sqrt(3)) < 1e-6, f"major axis ({wide.major_axis_um:.3f})")

    # An L shape: 3 of the 4 cells of its 2×2 hull.
    ell = np.array([[0, 0], [0, 20], [10, 20], [10, 10], [20, 10], [20, 0]], dtype=float)
    shape = an.region_shape(rg.Region("L", ell))
    check(abs(shape.solidity - 0.75 / (1 - 0.125)) < 1e-9, f"L solidity against its hull ({shape.solidity:.4f})")
    reversed_shape = an.region_shape(rg.Region("L", ell[::-1]))
    check(
        abs(reversed_shape.major_axis_um - shape.major_axis_um) < 1e-9,
        "winding does not change the axes",
    )


def test_channel_stats() -> None:
    print("channel statistics")
    volume = np.zeros((3, 10, 10), dtype=np.uint16)
    volume[:, :5, :] = 100
    volume[:, 5:, :] = 300
    masks = {
        "top": np.zeros((10, 10), dtype=bool),
        "all": np.ones((10, 10), dtype=bool),
    }
    masks["top"][:5, :] = True
    stats = {stat.region: stat for stat in an.channel_stats(volume, masks, "GFP")}
    check(stats["top"].n_voxels == 150, f"counted through every plane ({stats['top'].n_voxels})")
    check(stats["top"].mean == 100.0 and stats["top"].std == 0.0, "a flat region is flat")
    check(stats["top"].cv == 0.0, "with zero CV")
    check(stats["all"].mean == 200.0, f"mean over both halves ({stats['all'].mean})")
    check(stats["all"].integrated == 150 * 100 + 150 * 300, "integrated does not overflow uint16")
    check(stats["all"].percentiles[5.0] == 100.0 and stats["all"].percentiles[95.0] == 300.0, "tails")

    empty = an.summarise_values("none", "GFP", np.zeros(0))
    check(empty.n_voxels == 0 and np.isnan(empty.mean), "an empty region has no mean")


def test_projection() -> None:
    print("intensity projection")
    stack = np.zeros((2, 3, 4, 5), dtype=np.uint16)
    stack[0, 1, 2, 3] = 7
    stack[1, 2, 0, 0] = 9
    flat = an.project(stack)
    check(flat.shape == (4, 5), f"every leading axis collapsed ({flat.shape})")
    check(flat[2, 3] == 7 and flat[0, 0] == 9 and flat.sum() == 16, "brightest value kept per pixel")
    check(an.project(flat) is not None and an.project(flat).shape == (4, 5), "a plane stays a plane")


def test_label_key_pick() -> None:
    print("label map choice")
    keys = ["DAPI labels", "GFP labels"]
    check(an.pick_label_key(keys) == "DAPI labels", "default is the first")
    check(an.pick_label_key(keys, "gfp") == "GFP labels", "matched by substring")
    check(an.pick_label_key(keys, "rfp") == "", "a miss is a miss, not the first")
    check(an.pick_label_key([], "") == "", "nothing stored, nothing picked")


def test_whole_workbook() -> None:
    print("analysing files")
    from make_sample_data import write_ims

    with tempfile.TemporaryDirectory() as directory:
        folder = Path(directory)
        first = write_ims(folder / "fish_wt_1.ims", shape=(1, 4, 64, 80))
        second = folder / "fish_mut_2.ims"
        shutil.copy(first, second)
        bare = folder / "fish_wt_3.ims"
        shutil.copy(first, bare)

        # 0.13 µm pixels: the image is 8.32 × 10.4 µm.
        rois = [
            store.StoredRoi("left", _box(0.0, 8.0, 0.0, 5.0)),
            store.StoredRoi("right", _box(0.0, 8.0, 5.0, 10.0)),
        ]
        masks = np.zeros((4, 64, 80), dtype=np.int32)
        masks[1:3, 10:20, 5:15] = 1   # left
        masks[1:3, 30:40, 50:60] = 2  # right
        masks[2:4, 40:50, 55:65] = 3  # right
        for path in (first, second):
            store.save_rois(path, rois)
            store.save_labels(
                path, "GFP labels", masks, (0.5, 0.13, 0.13), attrs={"channel": "GFP"}
            )

        messages: list[str] = []
        outcomes = an.analyse([first, second, bare, folder / "missing.ims"], progress=messages.append)
        for path in (first, second, bare):
            release(path)

        check(len(outcomes) == 4, f"one outcome per file ({len(outcomes)})")
        good = outcomes[0]
        check(not good.error, f"the first file analysed ({good.error})")
        check(good.n_objects == 3, f"three objects read back ({good.n_objects})")
        check(good.label_key == "GFP labels", "from the stored map")
        check([s.region for s in good.shapes] == ["left", "right"], "one shape per region")
        regions_seen = {stat.region for stat in good.channel_stats}
        check(
            regions_seen == {"left", "right", an.ALL_REGIONS},
            f"intensities per region and their union ({sorted(regions_seen)})",
        )
        channels_seen = {stat.channel for stat in good.channel_stats}
        check(len(channels_seen) == 2, f"every channel described ({sorted(channels_seen)})")
        check(bool(good.stats) and good.stats[0].mean > 0, "per-object intensities measured")

        check(outcomes[2].n_objects == 0 and not outcomes[2].error, "a file with nothing stored still runs")
        check(any("no stored label map" in w for w in outcomes[2].warnings), "and says why it has no objects")
        check(bool(outcomes[3].error), "a missing file is reported, not raised")
        check(any("fish_mut_2" in m for m in messages), "progress names each file")

        sheets = an.workbook_sheets(outcomes)
        check(
            list(sheets) == [
                "Samples", "Regions", "Objects",
                "Region features", "Region intensities", "PCA matrix", "Region outlines",
                "Cell shapes",
            ],
            f"the batch sheets and then the new ones ({list(sheets)})",
        )
        features = sheets["Region features"]
        check(
            len(features) == 2 + 2 + 1,
            f"one row per sample and region, one for the bare sample ({len(features)})",
        )
        left = features[(features["Sample"] == "fish_wt_1") & (features["Region"] == "left")].iloc[0]
        check(left["Objects"] == 1, f"left has one object ({left['Objects']})")
        right = features[(features["Sample"] == "fish_wt_1") & (features["Region"] == "right")].iloc[0]
        check(right["Objects"] == 2, f"right has two ({right['Objects']})")
        check(abs(left["Region area (µm²)"] - 40.0) < 0.1, f"region area ({left['Region area (µm²)']})")
        check(abs(left["Region volume (µm³)"] - 40.0 * 2.0) < 0.1, "volume over the 2 µm stack")
        intensity_columns = [c for c in features.columns if c.endswith(" Mean")]
        check(len(intensity_columns) == 2, f"a mean column per channel ({intensity_columns})")

        long = sheets["Region intensities"]
        check(len(long) == 2 * 3 * 2 + 2, f"long sheet rows ({len(long)})")
        left_px = long[(long['Sample'] == 'fish_wt_1') & (long['Region'] == 'left')]['Voxels'].iloc[0]
        check(left_px < 64 * 80, f"region intensities come from one projected plane ({left_px} px)")

        outlines = sheets["Region outlines"]
        check(len(outlines) == 2 * 2 * 4, f"four vertices per outline, per sample ({len(outlines)})")
        first_left = outlines[(outlines["Sample"] == "fish_wt_1") & (outlines["Region"] == "left")]
        check(
            first_left[["Y (µm)", "X (µm)"]].to_numpy().tolist() == _box(0.0, 8.0, 0.0, 5.0).tolist(),
            "the vertices are the stored outline, in µm",
        )

        shapes = sheets["Cell shapes"]
        check(len(shapes) == 3 * 2, f"a shape row per cell, per sample ({len(shapes)})")
        one = shapes[(shapes["Sample"] == "fish_wt_1") & (shapes["Label"] == 1)].iloc[0]
        check(abs(one["Footprint area (µm²)"] - 100 * 0.13 * 0.13) < 1e-6, "footprint area in µm²")
        check(abs(one["Aspect ratio"] - 1.0) < 0.1, f"a square cell is not elongated ({one['Aspect ratio']})")
        three = shapes[(shapes["Sample"] == "fish_wt_1") & (shapes["Label"] == 3)].iloc[0]
        check(
            abs(three["Footprint area (µm²)"] - 100 * 0.13 * 0.13) < 1e-6,
            "a 3D cell is measured by its footprint, not its volume",
        )
        outline = good.cell_outlines[1]
        check(outline.shape == (an.CONTOUR_POINTS, 2), f"outline resampled ({outline.shape})")
        centre = outline.mean(axis=0)
        check(
            abs(centre[0] - 14.5 * 0.13) < 0.05 and abs(centre[1] - 9.5 * 0.13) < 0.05,
            f"outline sits where the cell is ({centre})",
        )

        pca = sheets["PCA matrix"]
        check(len(pca) == 3, f"one PCA row per analysed sample ({len(pca)})")
        check(list(pca.columns[:2]) == ["Sample", "Genotype"], "identified first")
        check(any(c.startswith("left | ") for c in pca.columns), "features named by region")
        numeric = pca.drop(columns=["Sample", "Genotype"])
        check(all(numeric[c].dtype.kind in "fiu" for c in numeric.columns), "and all numeric")

        from microscopy_viewer.exports import export_sheets

        written = export_sheets(sheets, folder / "analysis.xlsx")
        import pandas as pd

        back = pd.read_excel(written, sheet_name=None)
        check(set(back) == set(sheets), f"the workbook has every sheet ({sorted(back)})")

        from microscopy_viewer import analysis_plots as ap

        report, notes = ap.write_report(outcomes, folder, "my experiment")
        check(report.parent == folder, "the report is a folder inside the parent")
        check(report.name.startswith("my_experiment_analysis_"), f"named after the experiment ({report.name})")
        made = sorted(item.name for item in report.iterdir())
        check(
            made == sorted([
                f"{report.name}.xlsx", "cell_outlines.npz", "pca_cells.png",
                "violin_cell_count.png", "violin_region_area.png",
            ]),
            f"workbook and three figures ({made})",
        )
        check(notes == [], f"every figure was drawn ({notes})")
        stored = an.load_cell_outlines(report / "cell_outlines.npz")
        check(len(stored) == 6, f"every cell outline saved ({len(stored)})")
        check(
            np.allclose(stored[("fish_mut_2", 2)], outcomes[1].cell_outlines[2]),
            "and read back unchanged",
        )
        again, _notes = ap.write_report(outcomes, folder, "my experiment")
        check(again != report and again.exists(), "a second run never overwrites the first")


def test_plot_helpers() -> None:
    print("plot helpers")
    from microscopy_viewer import analysis_plots as ap

    order = ap.genotype_order(["mut", "", "wt", "het", None])
    check(order == ["wt", "het", "mut", ap.UNKNOWN_GENOTYPE], f"reference first, unknown last ({order})")
    styles = ap.genotype_styles(order)
    check(len({color for color, _m in styles.values()}) == 4, "one colour per genotype")
    check(len({marker for _c, marker in styles.values()}) == 4, "and one marker each")

    rng = np.random.default_rng(1)
    base = rng.normal(size=(200, 1))
    matrix = np.hstack([base, 2 * base + 0.01 * rng.normal(size=(200, 1)), np.ones((200, 1))])
    scores, explained, kept = ap.pca(matrix)
    check(kept == [0, 1], f"a constant column is dropped ({kept})")
    check(explained[0] > 0.99, f"two correlated columns are one component ({explained[0]:.3f})")
    check(scores.shape == (200, 2), "two scores per row")
    empty, _e, none = ap.pca(np.ones((5, 3)))
    check(empty.shape[0] == 0 and none == [], "nothing varies, nothing to project")


def main() -> int:
    for test in (
        test_region_shape, test_channel_stats, test_projection, test_label_key_pick, test_whole_workbook, test_plot_helpers
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
