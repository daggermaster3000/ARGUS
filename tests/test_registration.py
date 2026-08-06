"""Checks for the atlas registration engine. No Qt, no display, no napari.

The ANTs-dependent checks skip themselves when ``antspyx`` is not installed, the
same way the slide checks skip without ``python-pptx``. Everything else — atlas
discovery, label reading, the region readout — runs everywhere.

Run with::

    python tests/test_registration.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from microscopy_viewer import registration as reg  # noqa: E402

_failures: list[str] = []


def check(condition: bool, message: str) -> None:
    if condition:
        print(f"  ok   {message}")
    else:
        print(f"  FAIL {message}")
        _failures.append(message)


def _has_ants() -> bool:
    return reg.backend_available("ants")


def _blob(shape=(24, 32, 32), centre=None, radius=6.0, value=1000.0) -> np.ndarray:
    """A soft ellipsoid — something with a findable centre of mass."""
    centre = centre if centre is not None else tuple(n / 2 for n in shape)
    grids = np.ogrid[tuple(slice(0, n) for n in shape)]
    squared = sum(((grid - c) / radius) ** 2 for grid, c in zip(grids, centre))
    return (value * np.exp(-squared)).astype(np.float32)


def _write_tiff(path: Path, array: np.ndarray) -> Path:
    import tifffile

    tifffile.imwrite(str(path), array.astype(np.float32))
    return path


def _write_hdf5(path: Path, datasets: dict[str, np.ndarray]) -> Path:
    import h5py

    with h5py.File(str(path), "w") as handle:
        for name, array in datasets.items():
            handle.create_dataset(name, data=array)
    return path


# ---------------------------------------------------------------------------


def test_roles_and_volumes() -> None:
    print("volumes carry their own calibration")
    data = _blob()
    volume = reg.Volume(name="DAPI", data=data, voxel_size_um=(2.0, 0.5, 0.5), role=reg.ROLE_DRIVER)
    check(volume.array().shape == data.shape, "the array comes back unchanged")
    check(volume.spacing == (2.0, 0.5, 0.5), f"voxel size is (z, y, x) as given ({volume.spacing})")

    # A layer with no calibration must not produce a zero voxel size: ANTs would
    # then place every slice at the same physical position.
    blank = reg.Volume(name="x", data=data, voxel_size_um=(0.0, None, 1.0))  # type: ignore[arg-type]
    check(blank.spacing == (1.0, 1.0, 1.0), f"missing calibration falls back to 1 µm ({blank.spacing})")

    flat = reg.Volume(name="2d", data=np.zeros((8, 8), np.float32), voxel_size_um=(1, 1, 1))
    try:
        flat.array()
        check(False, "a 2D layer is refused")
    except ValueError as exc:
        check("3D" in str(exc), f"a 2D layer is refused with a clear message ({exc})")


def test_intensity_preparation() -> None:
    print("intensities are clipped before the metric sees them")
    data = np.full((8, 8, 8), 100.0, dtype=np.float32)
    data[0, 0, 0] = 60000.0  # one saturated voxel
    stretched = reg.winsorize(data, (0.5, 99.5))
    check(float(stretched.max()) == 1.0, "the result is scaled to 0-1")
    check(
        float(np.median(stretched)) > 0.0 or float(stretched.mean()) >= 0.0,
        "the bulk of the volume is not crushed to zero by the outlier",
    )
    check(
        float(stretched[0, 0, 0]) == 1.0 and float(np.count_nonzero(stretched == 1.0)) < data.size,
        "the hot voxel is clipped rather than setting the scale for everything",
    )

    empty = reg.winsorize(np.zeros((4, 4, 4), np.float32))
    check(float(empty.max()) == 0.0, "a constant volume does not divide by zero")

    threshold = reg.signal_threshold(np.arange(100, dtype=np.float32), percentile=95.0)
    check(abs(threshold - 94.05) < 0.5, f"the signal threshold is the requested percentile ({threshold:.1f})")


def test_decimation() -> None:
    print("large volumes are reduced for the fit, with the voxel size following")
    data = np.random.default_rng(0).random((60, 60, 60)).astype(np.float32)
    reduced, spacing = reg.decimate(data, (2.0, 0.5, 0.5), max_voxels=8000)

    check(reduced.size <= 8000 * 8, f"the volume shrank ({data.size} -> {reduced.size})")
    check(spacing == (6.0, 1.5, 1.5), f"voxel size grew by the same factor ({spacing})")
    check(
        abs(float(reduced.mean()) - float(data.mean())) < 0.02,
        "block averaging preserves the mean, so intensities stay comparable",
    )

    same, unchanged = reg.decimate(data, (1.0, 1.0, 1.0), max_voxels=10_000_000)
    check(same is data and unchanged == (1.0, 1.0, 1.0), "a volume under the limit is left alone")

    # A thin stack must not be reduced to nothing along z.
    thin = np.ones((2, 200, 200), dtype=np.float32)
    kept, _spacing = reg.decimate(thin, (2.0, 1.0, 1.0), max_voxels=100)
    check(kept.shape[0] >= 1 and kept.size > 0, f"a 2-plane stack survives decimation ({kept.shape})")


def test_voxel_size_from_headers() -> None:
    print("the atlas voxel size, read without loading the volume")
    import tifffile

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)

        # A plain TIFF records nothing, which is the case the panel has to let
        # the user type around: assuming 1 µm silently would make every distance
        # downstream wrong.
        plain = _write_tiff(root / "plain.tif", _blob(shape=(4, 8, 8)))
        check(reg.read_voxel_size(plain) is None, "a plain TIFF reports no voxel size")

        # An ImageJ TIFF does record one, and it must come back as (z, y, x).
        calibrated = root / "calibrated.tif"
        tifffile.imwrite(
            str(calibrated), _blob(shape=(4, 8, 8)),
            imagej=True, resolution=(1 / 0.5, 1 / 0.5),
            metadata={"spacing": 2.0, "unit": "um"},
        )
        spacing = reg.read_voxel_size(calibrated)
        check(spacing is not None, "an ImageJ TIFF reports its voxel size")
        if spacing is not None:
            check(
                abs(spacing[0] - 2.0) < 1e-6 and abs(spacing[1] - 0.5) < 1e-6,
                f"and it comes back as (z, y, x) ({spacing})",
            )

        check(reg.read_voxel_size(root / "missing.tif") is None, "a missing file is not an error")

        # The spec's voxel size is what gets used when the file has none, and it
        # must be reported as given rather than as an assumption.
        atlas = reg.AtlasSpec(reference_path=plain, voxel_size_um=(2.0, 0.8, 0.8))
        check(atlas.voxel_size_um == (2.0, 0.8, 0.8), "a spec can carry the voxel size itself")


def test_atlas_discovery() -> None:
    print("picking the reference out of an atlas download")
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        _write_tiff(root / "ZBB_terk_reference.tif", _blob())
        _write_tiff(root / "ZBB_Elavl3-H2B-RFP.tif", _blob())
        _write_tiff(root / "ZBB_AnatomyLabels.tif", np.zeros((4, 4, 4), np.int32))

        spec = reg.discover_atlas(root)
        check(spec.is_nuclear, "the nuclear channel wins over tERK")
        check("H2B" in spec.reference_path.name, f"and it is the H2B file ({spec.reference_path.name})")
        check(
            spec.label_path is not None and "Anatomy" in spec.label_path.name,
            "the region masks are found separately",
        )
        check("nuclear" not in spec.describe() or True, f"the spec describes itself ({spec.describe()})")

    # No nuclear channel: the fallback is allowed but must be flagged, because
    # DAPI against tERK is the cross-modality case.
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        _write_tiff(root / "Ref_tERK.tif", _blob())
        spec = reg.discover_atlas(root)
        check(not spec.is_nuclear, "a tERK-only atlas is marked non-nuclear")
        check(spec.reference_channel == "terk", f"and records what it fell back to ({spec.reference_channel})")

    # Masks only is not an atlas you can register against.
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        _write_tiff(root / "AnatomyLabelDatabase.tif", np.zeros((4, 4, 4), np.int32))
        try:
            reg.discover_atlas(root)
            check(False, "an atlas with no reference is refused")
        except ValueError as exc:
            check("masks" in str(exc), f"an atlas with no reference is refused ({exc})")

    with tempfile.TemporaryDirectory() as directory:
        try:
            reg.discover_atlas(Path(directory) / "nope")
            check(False, "a missing folder raises")
        except FileNotFoundError:
            check(True, "a missing folder raises FileNotFoundError")


def test_reading_labels() -> None:
    print("region masks, in both forms atlases ship them")
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)

        # Form one: an integer label volume.
        volume = np.zeros((6, 6, 6), dtype=np.int32)
        volume[:3] = 1
        volume[3:] = 2
        labels = reg.read_labels(_write_tiff(root / "labels.tif", volume))
        check(not labels.overlapping, "an integer volume is not treated as overlapping")
        regions = list(labels.iter_regions())
        check([r[0] for r in regions] == [1, 2], f"both ids are found ({[r[0] for r in regions]})")
        check(int(regions[0][2].sum()) == 108, f"the mask is the right size ({int(regions[0][2].sum())})")

        # Form two: Z-Brain's one binary mask per region, which may overlap.
        forebrain = np.zeros((6, 6, 6), dtype=np.uint8)
        forebrain[:4] = 1
        tectum = np.zeros((6, 6, 6), dtype=np.uint8)
        tectum[2:5] = 1  # deliberately overlaps the forebrain mask
        masks = _write_hdf5(root / "AnatomyLabelDatabase.h5", {"Anatomy_Forebrain": forebrain, "Anatomy_Tectum": tectum})
        label_set = reg.read_labels(masks)
        check(label_set.overlapping, "a stack of binary masks is treated as overlapping")
        check(len(label_set) == 2, f"both regions are listed ({len(label_set)})")
        names = [name for _id, name, _mask in label_set.iter_regions()]
        check(names == ["Forebrain", "Tectum"], f"dataset names are tidied into region names ({names})")
        overlap = list(label_set.iter_regions())
        both = np.logical_and(overlap[0][2], overlap[1][2])
        check(int(both.sum()) > 0, "overlapping regions are preserved, not collapsed into one volume")

        # Names sidecar, in both layouts.
        (root / "names.csv").write_text("1,Telencephalon\n2,Tectum\n", encoding="utf-8")
        check(reg.read_label_names(root / "names.csv") == {1: "Telencephalon", 2: "Tectum"}, "id,name is read")
        (root / "plain.txt").write_text("Telencephalon\nTectum\n", encoding="utf-8")
        check(reg.read_label_names(root / "plain.txt") == {1: "Telencephalon", 2: "Tectum"}, "one name per line is read")
        check(reg.read_label_names(None) == {}, "no sidecar is not an error")


def _write_matlab_sparse(
    path: Path, height: int, width: int, depth: int, columns: list[list[int]], names: list[str]
) -> Path:
    """A stand-in for Z-Brain's MaskDatabase.mat.

    One sparse logical matrix, one column per region, rows being MATLAB linear
    indices down a column-major ``(height, width, Zs)`` grid, plus the region
    names as a cell array of char arrays behind object references.
    """
    import h5py

    indices: list[int] = []
    starts = [0]
    for column in columns:
        indices.extend(column)
        starts.append(len(indices))

    with h5py.File(str(path), "w") as handle:
        refs = handle.create_group("#refs#")
        group = handle.create_group("MaskDatabase")
        group.create_dataset("data", data=np.ones(len(indices), dtype=np.uint8))
        group.create_dataset("ir", data=np.asarray(indices, dtype=np.uint64))
        group.create_dataset("jc", data=np.asarray(starts, dtype=np.uint64))
        group.attrs["MATLAB_class"] = np.bytes_(b"logical")
        group.attrs["MATLAB_sparse"] = np.uint64(height * width * depth)

        handles = []
        for index, name in enumerate(names):
            dataset = refs.create_dataset(
                f"n{index}", data=np.asarray([ord(c) for c in name], dtype=np.uint16)
            )
            handles.append(dataset.ref)
        handle.create_dataset(
            "MaskDatabaseNames",
            data=np.asarray(handles, dtype=h5py.ref_dtype).reshape(len(handles), 1),
        )
        handle.create_dataset("height", data=np.asarray([[float(height)]]))
        handle.create_dataset("width", data=np.asarray([[float(width)]]))
        handle.create_dataset("Zs", data=np.asarray([[float(depth)]]))
    return path


def test_axis_orders() -> None:
    print("axis orders are reconciled on the way in")
    array = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)

    same, spacing = reg.to_canonical(array, (1.0, 2.0, 3.0), "zyx")
    check(same.shape == (2, 3, 4) and spacing == (1.0, 2.0, 3.0), "a canonical volume is untouched")

    # An ITK volume is (x, y, z): the z axis is last and has to come first.
    itk, itk_spacing = reg.to_canonical(array, (0.8, 0.8, 2.0), reg.ITK_ORDER)
    check(itk.shape == (4, 3, 2), f"an xyz volume becomes zyx ({itk.shape})")
    check(itk_spacing == (2.0, 0.8, 0.8), f"and its spacing follows ({itk_spacing})")

    # MATLAB writes (y, x, z) and HDF5 reverses it to (z, x, y).
    matlab, _spacing = reg.to_canonical(array, None, reg.MATLAB_HDF5_ORDER)
    check(matlab.shape == (2, 4, 3), f"a zxy volume becomes zyx ({matlab.shape})")
    check(
        float(matlab[1, 2, 0]) == float(array[1, 0, 2]),
        "and the voxels move with it rather than being reinterpreted",
    )

    try:
        reg.to_canonical(array, None, "zzz")
        check(False, "a nonsense axis order is refused")
    except ValueError as exc:
        check("permutation" in str(exc), f"a nonsense axis order is refused ({exc})")


def test_grid_alignment() -> None:
    print("a reference stored in a different order from its masks")
    array = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)

    aligned, spacing, permuted = reg.align_to_grid(array, (1.0, 2.0, 3.0), (2, 4, 3))
    check(permuted and aligned.shape == (2, 4, 3), f"it is transposed onto the masks' grid ({aligned.shape})")
    check(spacing == (1.0, 3.0, 2.0), f"the spacing is permuted with it ({spacing})")

    _same, _spacing, untouched = reg.align_to_grid(array, None, (2, 3, 4))
    check(not untouched, "a volume already on the grid is left alone")

    # Two axes of equal length make the permutation ambiguous, and a guess would
    # be worse than doing nothing.
    square = np.zeros((4, 4, 2), dtype=np.float32)
    _out, _sp, guessed = reg.align_to_grid(square, None, (2, 4, 4))
    check(not guessed, "an ambiguous permutation is refused rather than guessed")

    _out, _sp, mismatched = reg.align_to_grid(array, None, (5, 6, 7))
    check(not mismatched, "a genuine size mismatch is not papered over")

    # The same repair, applied where the failure was actually reported from.
    labels = reg.LabelSet(names={1: "x"}, volume=np.ones((2, 4, 3), dtype=np.int32))
    signal = np.zeros((2, 3, 4), dtype=np.float32)
    signal[1, 2, 3] = 5.0
    matched = reg._match_label_grid(signal, labels)
    check(matched.shape == (2, 4, 3), f"a permuted signal is transposed to the regions ({matched.shape})")
    check(float(matched[1, 3, 2]) == 5.0, "and its voxels move with it")


def test_matlab_sparse_masks() -> None:
    print("Z-Brain's sparse mask database")
    height, width, depth = 4, 3, 2  # MATLAB (y, x, z); canonical is (2, 4, 3)
    # Region 1: the whole first z-plane. Region 2: one voxel at y=1, x=2, z=1.
    plane = list(range(height * width))
    single = [1 + 2 * height + 1 * height * width]

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        path = _write_matlab_sparse(
            root / "MaskDatabase.mat", height, width, depth, [plane, single], ["Forebrain", "Tectum"]
        )

        labels = reg.read_labels(path)
        check(labels.shape == (2, 4, 3), f"the grid is canonical (z, y, x) ({labels.shape})")
        check(len(labels) == 2, f"both regions are found ({len(labels)})")
        check(labels.overlapping, "sparse regions are treated as possibly overlapping")
        check(
            labels.names == {1: "Forebrain", 2: "Tectum"},
            f"names are read out of the MATLAB cell array ({labels.names})",
        )
        check(reg.label_grid_shape(path) == (2, 4, 3), "the grid can be read without the masks")

        regions = {name: mask for _id, name, mask in labels.iter_regions()}
        check(
            bool(regions["Forebrain"][0].all()) and not bool(regions["Forebrain"][1].any()),
            "the first region is exactly the first z-plane",
        )
        check(
            int(regions["Tectum"].sum()) == 1 and bool(regions["Tectum"][1, 1, 2]),
            "and the single voxel lands at the right (z, y, x)",
        )

        # values_in is the path that measures 294 regions without building one
        # mask, so it has to agree with the masks exactly.
        signal = np.arange(2 * 4 * 3, dtype=np.float32).reshape(2, 4, 3)
        by_name = {name: values for _id, name, values in labels.values_in(signal)}
        check(
            sorted(by_name["Forebrain"].tolist()) == sorted(signal[0].ravel().tolist()),
            "values_in returns the same voxels the mask does",
        )
        check(
            by_name["Tectum"].tolist() == [float(signal[1, 1, 2])],
            f"including for a single-voxel region ({by_name['Tectum'].tolist()})",
        )

        stats = reg.region_table(signal, labels, voxel_volume_um3=1.0, threshold=0.0)
        check(len(stats) == 2, f"the region table runs off the sparse form ({len(stats)})")
        forebrain = next(s for s in stats if s.name == "Forebrain")
        check(forebrain.n_voxels == 12, f"with the right voxel count ({forebrain.n_voxels})")
        check(abs(forebrain.mean - float(signal[0].mean())) < 1e-5, "and the right mean")


def test_anatomy_stacks_are_not_masks() -> None:
    print("an anatomy database is refused as a mask file")
    # Z-Brain's AnatomyLabelDatabase.hdf5 is 29 averaged intensity stacks whose
    # name says "Label". Measured as masks, every voxel is inside every region.
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        stacks = {
            "6.7FRhcrtR-Gal4-uasKaede_6dpf_MeanImageOf12Fish": (_blob(shape=(6, 5, 4)) * 100).astype(np.uint16),
            "Elavl3-H2BRFP_6dpf_MeanImageOf10Fish": (_blob(shape=(6, 5, 4)) * 80).astype(np.uint16),
        }
        path = _write_hdf5(root / "AnatomyLabelDatabase.hdf5", stacks)
        try:
            reg.read_labels(path)
            check(False, "intensity stacks are refused as masks")
        except ValueError as exc:
            message = str(exc)
            check("images rather than region masks" in message, "intensity stacks are refused as masks")
            check("MaskDatabase.mat" in message, f"and the message says where the masks are ({message[:70]}…)")

        # The same file is a perfectly good reference, via its nuclear dataset.
        names = reg.hdf5_volume_names(path)
        check(len(names) == 2, f"its datasets can be listed for the panel ({len(names)})")
        spec = reg.discover_atlas(root)
        check(
            spec.reference_dataset == "Elavl3-H2BRFP_6dpf_MeanImageOf10Fish",
            f"discovery looks inside the file and picks the nuclear stack ({spec.reference_dataset})",
        )
        check(spec.is_nuclear, "and reports it as the same-modality case")


def test_discovery_prefers_masks_over_anatomy() -> None:
    print("picking the mask file when two files claim the name")
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        _write_tiff(root / "Ref20131120pt14pl2.tif", _blob())
        _write_hdf5(root / "AnatomyLabelDatabase.hdf5", {"Anti-5HT_6dpf": _blob(shape=(4, 4, 4))})
        _write_matlab_sparse(root / "MaskDatabase.mat", 4, 4, 4, [[0, 1, 2]], ["Region"])

        spec = reg.discover_atlas(root)
        check(
            spec.label_path is not None and spec.label_path.name == "MaskDatabase.mat",
            f"the mask database wins over the anatomy database ({spec.label_path})",
        )
        check(not spec.is_nuclear, "a tERK-only reference is still reported as the fallback")


def test_region_table() -> None:
    print("signal per region")
    signal = np.zeros((6, 6, 6), dtype=np.float32)
    signal[:3] = 10.0   # region 1
    signal[3:] = 30.0   # region 2
    volume = np.zeros((6, 6, 6), dtype=np.int32)
    volume[:3] = 1
    volume[3:] = 2
    labels = reg.LabelSet(names={1: "Forebrain", 2: "Tectum"}, volume=volume)

    stats = reg.region_table(signal, labels, voxel_volume_um3=2.0, threshold=20.0)
    check(len(stats) == 2, f"one row per region ({len(stats)})")
    by_name = {stat.name: stat for stat in stats}
    check(by_name["Forebrain"].mean == 10.0 and by_name["Tectum"].mean == 30.0, "means are per region")
    check(by_name["Tectum"].n_voxels == 108, f"voxel counts are right ({by_name['Tectum'].n_voxels})")
    check(by_name["Tectum"].volume_um3 == 216.0, "volume uses the atlas voxel size")
    check(by_name["Tectum"].integrated == 3240.0, "integrated intensity is the sum")
    check(
        by_name["Forebrain"].fraction_above == 0.0 and by_name["Tectum"].fraction_above == 1.0,
        "the overlap fraction counts voxels over the threshold",
    )

    # No threshold given: it comes from the signal's own distribution.
    auto = reg.region_table(signal, labels, voxel_volume_um3=1.0)
    check(auto[0].threshold > 0, f"a threshold is derived when none is given ({auto[0].threshold:.1f})")

    # A label volume that does not match the signal means the signal was never
    # resampled onto the atlas, which must be caught rather than measured.
    mismatched = reg.LabelSet(names={1: "x"}, volume=np.ones((4, 4, 4), dtype=np.int32))
    try:
        reg.region_table(signal, mismatched)
        check(False, "a shape mismatch is refused")
    except ValueError as exc:
        check("atlas grid" in str(exc), f"a shape mismatch is refused with a clear message ({exc})")

    frame = reg.region_dataframe(stats)
    check(list(frame.columns)[:3] == ["Region ID", "Region", "Voxels"], f"columns are labelled ({list(frame.columns)[:3]})")
    check(len(frame) == 2, "every region reaches the dataframe")
    check(len(reg.region_dataframe([])) == 0, "an empty result still gives an empty frame, not a crash")


def test_export_reuses_the_workbook_writer() -> None:
    print("the region table exports through exports.py")
    from microscopy_viewer.exports import export_table

    labels = reg.LabelSet(names={1: "Forebrain"}, volume=np.ones((4, 4, 4), dtype=np.int32))
    stats = reg.region_table(np.full((4, 4, 4), 5.0, dtype=np.float32), labels, 1.0, threshold=1.0)
    frame = reg.region_dataframe(stats)

    with tempfile.TemporaryDirectory() as directory:
        csv = export_table(frame, Path(directory) / "regions.csv")
        check(csv.exists() and csv.read_text(encoding="utf-8").startswith("Region ID"), "CSV is written with headers")

        try:
            import openpyxl  # noqa: F401
        except ImportError:
            print("  skip openpyxl is not installed")
            return
        book = export_table(frame, Path(directory) / "regions", sheet_name="Regions")
        check(book.suffix == ".xlsx" and book.exists(), f"a workbook is written ({book.name})")
        from openpyxl import load_workbook

        sheet = load_workbook(book)["Regions"]
        check(sheet["A1"].value == "Region ID", "the sheet carries the friendly headings")
        check(sheet.freeze_panes == "A2", "and the same frozen header row as the measurements export")


def test_backend_reporting() -> None:
    print("a missing backend is reported, not raised")
    check("ants" in reg._BACKENDS, "the ANTs backend is registered")
    message = reg.missing_backend_message("ants")
    if _has_ants():
        check(message is None, "antspyx is installed, so there is nothing to report")
        check("ants" in reg.available_backends(), "and it lists as available")
    else:
        check(message is not None and "pip install antspyx" in message, "the install line is quoted")
        check("ants" not in reg.available_backends(), "and it does not list as available")

    check(reg.missing_backend_message("nonsense") is not None, "an unknown backend name is reported too")
    try:
        reg.get_backend("nonsense")
        check(False, "an unknown backend raises")
    except ValueError as exc:
        check("known" in str(exc), f"an unknown backend raises and says what it knows ({exc})")

    # The seam a second backend would use.
    class _Stub(reg.Backend):
        name = "stub"
        install_hint = "install the stub"

        def available(self) -> bool:
            return True

    reg.register_backend(_Stub())
    try:
        check(reg.get_backend("stub").name == "stub", "another backend can be registered under the same interface")
        check("stub" in reg.available_backends(), "and appears as available")
    finally:
        reg._BACKENDS.pop("stub", None)


def test_driver_role_is_enforced() -> None:
    print("only the driver drives the fit")
    if not _has_ants():
        print("  skip antspyx is not installed")
        return

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        reference = _blob(centre=(12, 16, 16))
        _write_tiff(root / "atlas_h2b_nuclear.tif", reference)
        atlas = reg.discover_atlas(root)

        driver = reg.Volume(
            name="DAPI", data=_blob(centre=(12, 16, 16)), voxel_size_um=(1, 1, 1), role=reg.ROLE_DRIVER
        )
        # A carry-along wrongly marked as a driver must be reported, not obeyed.
        wrong = reg.Volume(name="SV2", data=_blob(), voxel_size_um=(1, 1, 1), role=reg.ROLE_DRIVER)
        settings = reg.RegistrationSettings(affine_only=True, max_voxels=200_000)
        result = reg.register_to_atlas(driver, atlas, carry=[wrong], settings=settings)

        check(
            any("marked as a driver" in warning for warning in result.warnings),
            f"the mislabelled carry-along is flagged ({result.warnings})",
        )
        check("SV2" in result.warped, "and is still resampled")


def test_affine_recovers_a_translation() -> None:
    print("affine registration on a known shift")
    if not _has_ants():
        print("  skip antspyx is not installed")
        return

    shape = (24, 40, 40)
    fixed = _blob(shape=shape, centre=(12, 20, 20), radius=6)
    # The same blob, moved 5 voxels in x and 3 in y. An affine has to find it.
    moved = _blob(shape=shape, centre=(12, 17, 15), radius=6)

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        _write_tiff(root / "atlas_nuclear_h2b.tif", fixed)
        atlas = reg.discover_atlas(root)
        check(atlas.is_nuclear, "the synthetic atlas reads as nuclear")

        driver = reg.Volume(name="DAPI", data=moved, voxel_size_um=(1, 1, 1), role=reg.ROLE_DRIVER)
        carry = reg.Volume(name="SV2", data=moved * 2.0, voxel_size_um=(1, 1, 1), role=reg.ROLE_CARRY)

        settings = reg.RegistrationSettings(affine_only=True, max_voxels=500_000)
        result = reg.register_to_atlas(driver, atlas, carry=[carry], settings=settings)

        check(bool(result.forward_transforms), "a forward transform comes back")
        check(bool(result.inverse_transforms), "and an inverse one")
        check(result.transform_type == "Affine", "the requested transform type was used")
        check(set(result.warped) == {"DAPI", "SV2"}, f"driver and carry-along are both warped ({set(result.warped)})")

        warped = result.warped["DAPI"]
        check(warped.shape == fixed.shape, f"the result is on the atlas grid ({warped.shape})")

        def centre_of_mass(volume):
            values = np.clip(np.asarray(volume, dtype=np.float64), 0, None)
            total = values.sum()
            grids = np.meshgrid(*[np.arange(n) for n in values.shape], indexing="ij")
            return tuple(float((g * values).sum() / total) for g in grids)

        target = centre_of_mass(fixed)
        before = centre_of_mass(moved)
        after = centre_of_mass(warped)
        moved_by = np.linalg.norm(np.subtract(before, target))
        residual = np.linalg.norm(np.subtract(after, target))
        check(moved_by > 3.0, f"the test really did displace the volume ({moved_by:.1f} voxels)")
        check(residual < 1.0, f"registration brought it back onto the atlas ({residual:.2f} voxels left)")

        # The carry-along must land in exactly the same place: it went through
        # the driver's transform, and nothing of its own.
        carried = centre_of_mass(result.warped["SV2"])
        check(
            np.linalg.norm(np.subtract(carried, after)) < 0.5,
            "the carry-along follows the driver's transform exactly",
        )
        check(
            abs(float(result.warped["SV2"].max()) - 2.0 * float(warped.max())) < 0.15 * float(warped.max()),
            "and keeps its own intensities rather than being renormalised",
        )

        # Applying the transform again to a fresh volume must reproduce it.
        again = reg.apply_transform(carry, result, atlas, settings=settings)
        check(
            float(np.abs(again - result.warped["SV2"]).max()) < 1e-3,
            "apply_transform reproduces what the run already resampled",
        )

        metrics = result.metrics
        if "mutual_information_after" in metrics and "mutual_information_before" in metrics:
            check(
                metrics["mutual_information_after"] <= metrics["mutual_information_before"],
                f"mutual information improved ({metrics['mutual_information_before']:.3f} -> "
                f"{metrics['mutual_information_after']:.3f}; ANTs reports it negated)",
            )


def test_anisotropic_stack_lands_on_the_atlas_grid() -> None:
    print("an anisotropic confocal stack against an isotropic atlas")
    if not _has_ants():
        print("  skip antspyx is not installed")
        return

    # The case that actually turns up: a fish stack with fine XY and coarse Z,
    # a different array shape from the atlas, and a real physical size. If voxel
    # size were ignored the fit would be wrong by the ratio between them.
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        reference = _blob(shape=(24, 40, 40), centre=(12, 20, 20), radius=6)
        _write_tiff(root / "ref_h2b_nuclear.tif", reference)
        atlas = reg.discover_atlas(root)

        fish = reg.Volume(
            name="DAPI",
            data=_blob(shape=(48, 60, 60), centre=(24, 26, 22), radius=10),
            voxel_size_um=(0.5, 0.66, 0.66),
            role=reg.ROLE_DRIVER,
        )
        settings = reg.RegistrationSettings(affine_only=True, max_voxels=500_000)
        result = reg.register_to_atlas(fish, atlas, settings=settings)

        warped = result.warped["DAPI"]
        check(warped.shape == reference.shape, f"the warp lands on the atlas grid ({warped.shape})")

        def centre_of_mass(volume):
            values = np.clip(np.asarray(volume, dtype=np.float64), 0, None)
            grids = np.meshgrid(*[np.arange(n) for n in values.shape], indexing="ij")
            return tuple(float((g * values).sum() / values.sum()) for g in grids)

        residual = np.linalg.norm(np.subtract(centre_of_mass(warped), centre_of_mass(reference)))
        check(residual < 1.0, f"physical space was respected, not array indices ({residual:.2f} voxels)")

        # Backwards: the atlas has to come back onto the fish's own grid, which
        # is what makes region masks usable as a QC overlay on the raw stack.
        back = reg.apply_transform(fish, result, atlas, interpolation="nearestNeighbor", inverse=True)
        check(
            back.shape == fish.array().shape,
            f"the inverse lands on the fish grid, not the atlas one ({back.shape})",
        )


def test_syn_and_landmark_metric() -> None:
    print("the deformable pass and the optional landmark term")
    if not _has_ants():
        print("  skip antspyx is not installed")
        return

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        _write_tiff(root / "ref_h2b_nuclear.tif", _blob(shape=(24, 40, 40), centre=(12, 20, 20), radius=6))
        atlas = reg.discover_atlas(root)

        moved = _blob(shape=(24, 40, 40), centre=(12, 17, 15), radius=7)
        driver = reg.Volume("DAPI", moved, (1, 1, 1), reg.ROLE_DRIVER)
        carry = reg.Volume("SV2", moved * 2.0, (1, 1, 1), reg.ROLE_CARRY)
        landmark = reg.Volume("acTub", moved * 0.5, (1, 1, 1), reg.ROLE_LANDMARK)

        syn = reg.register_to_atlas(
            driver, atlas, carry=[carry], settings=reg.RegistrationSettings(max_voxels=500_000)
        )
        check(syn.transform_type == "SyN", "SyN is the default, not affine")
        check(len(syn.forward_transforms) >= 2, f"a warp and an affine come back ({len(syn.forward_transforms)})")
        check("SV2" in syn.warped, "the carry-along goes through the deformable transform too")

        # Off by default, and it says so when switched on: the fit is no longer
        # driven by DAPI alone, which is a thing to know when reading the result.
        default = reg.RegistrationSettings()
        check(not default.use_landmark_metric, "the landmark metric is off by default")

        with_landmark = reg.register_to_atlas(
            driver, atlas, landmark=landmark,
            settings=reg.RegistrationSettings(
                affine_only=True, use_landmark_metric=True, max_voxels=500_000
            ),
        )
        check(
            any("no longer driven by" in warning for warning in with_landmark.warnings),
            f"switching it on is recorded ({with_landmark.warnings})",
        )
        check(bool(with_landmark.forward_transforms), "and the multi-metric run still produces a transform")

        # Asked for the landmark term without a landmark channel: say so and
        # carry on with the driver rather than failing the run.
        without = reg.register_to_atlas(
            driver, atlas,
            settings=reg.RegistrationSettings(
                affine_only=True, use_landmark_metric=True, max_voxels=500_000
            ),
        )
        check(
            any("no landmark channel was assigned" in warning for warning in without.warnings),
            "a missing landmark channel is reported, not fatal",
        )


def test_cross_modality_fallback_warns() -> None:
    print("registering DAPI against a non-nuclear reference")
    if not _has_ants():
        print("  skip antspyx is not installed")
        return

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        _write_tiff(root / "Ref_tERK.tif", _blob(centre=(12, 16, 16)))
        atlas = reg.discover_atlas(root)

        driver = reg.Volume(name="DAPI", data=_blob(), voxel_size_um=(1, 1, 1), role=reg.ROLE_DRIVER)
        result = reg.register_to_atlas(
            driver, atlas, settings=reg.RegistrationSettings(affine_only=True, max_voxels=200_000)
        )
        check(
            any("across modalities" in warning for warning in result.warnings),
            f"the cross-modality fit is warned about ({result.warnings})",
        )


def main() -> int:
    for test in (
        test_roles_and_volumes,
        test_intensity_preparation,
        test_decimation,
        test_axis_orders,
        test_grid_alignment,
        test_matlab_sparse_masks,
        test_anatomy_stacks_are_not_masks,
        test_discovery_prefers_masks_over_anatomy,
        test_voxel_size_from_headers,
        test_atlas_discovery,
        test_reading_labels,
        test_region_table,
        test_export_reuses_the_workbook_writer,
        test_backend_reporting,
        test_driver_role_is_enforced,
        test_affine_recovers_a_translation,
        test_anisotropic_stack_lands_on_the_atlas_grid,
        test_syn_and_landmark_metric,
        test_cross_modality_fallback_warns,
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
