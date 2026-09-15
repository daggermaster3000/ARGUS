"""Checks for the spatial dashboard engine. No Qt, no browser.

Two halves. The first needs only anndata and runs anywhere: reading a file,
slicing one image out of a plate, subsampling, and the command line that starts
the page. The second needs scanpy and squidpy and skips without them, because the
dashboard is meant to be installable in an environment of its own and the viewer's
own environment need not carry it.

Run with::

    python tests/test_dashboard.py
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from microscopy_viewer import dashboard as db  # noqa: E402

_failures: list[str] = []


def check(condition: bool, message: str) -> None:
    if condition:
        print(f"  ok   {message}")
    else:
        print(f"  FAIL {message}")
        _failures.append(message)


def _has(name: str) -> bool:
    try:
        __import__(name)
    except ImportError:
        return False
    return True


def _plate_file(directory: Path, images: int = 3, per_image: int = 120) -> Path:
    """A small combined file shaped like the real one: features, spatial, image."""
    from microscopy_viewer import analysis
    import pandas as pd

    rng = np.random.default_rng(0)
    tables = []
    for index in range(images):
        n = per_image
        # Two blobs, so there is something for a clustering to find and something
        # for a neighbourhood statistic to say is not random.
        centre = rng.integers(0, 2, size=n)
        x = np.where(centre == 0, rng.normal(50, 12, n), rng.normal(250, 12, n))
        y = np.where(centre == 0, rng.normal(50, 12, n), rng.normal(250, 12, n))
        frame = pd.DataFrame(
            {
                "Label": np.arange(1, n + 1),
                "Voxels": np.where(centre == 0, rng.normal(400, 40, n), rng.normal(900, 60, n)),
                "Solidity": np.where(centre == 0, rng.normal(0.9, 0.02, n), rng.normal(0.75, 0.03, n)),
                "Mean intensity": rng.normal(500, 50, n),
                "Centroid Z (µm)": np.zeros(n),
                "Centroid Y (µm)": y,
                "Centroid X (µm)": x,
            }
        )
        path = directory / f"B_0{index + 2}_0.csv"
        frame.to_csv(path, index=False, encoding="utf-8")
        tables.append(path)
    return analysis.write_combined_anndata(tables, directory / "plate.h5ad")


# ---------------------------------------------------------------------------


def test_starting_it() -> None:
    print("what starts the page")

    check(db.app_path().exists(), f"the page is next to the module ({db.app_path().name})")

    argv = db.command("objects.h5ad", port=8600)
    check(argv[1:4] == ["-m", "streamlit", "run"], f"run through the module, not the exe: {argv[1:4]}")
    check(
        sys.executable in argv[0],
        "with this interpreter, so it finds the environment the packages are in",
    )
    check("--server.headless" in argv, "headless, so streamlit does not open its own browser twice")
    check(
        "--browser.gatherUsageStats" in argv,
        "and with the usage prompt off — it otherwise stops at a terminal question",
    )
    check(
        argv.index("--") < argv.index("--file"),
        f"the file is passed after --, where streamlit hands it to the script: {argv[-3:]}",
    )
    check("--file" not in db.command(None), "no file is a valid start; the page asks for one")

    port = db.free_port(9700)
    check(9700 <= port < 9720, f"a free port is found in the range ({port})")
    import socket

    with socket.socket() as taken:
        taken.bind(("127.0.0.1", port))
        taken.listen(1)
        check(db.free_port(port) != port, "and a port in use is stepped over rather than handed out")

    missing = db.missing_packages()
    hint = db.install_hint(missing)
    if missing:
        check(all(name in hint for name in missing), f"the hint names what is missing: {missing}")
        check("pip install" in hint, "and says what to type")
    else:
        check(hint == "", "nothing missing, nothing to say")


def test_reading_and_slicing(directory: Path) -> None:
    print("one image out of a plate")

    if not _has("anndata"):
        print("  skip anndata is not installed")
        return

    path = _plate_file(directory)
    adata = db.read(path)
    facts = db.describe(adata)
    check(facts["objects"] == 360, f"every object of every image ({facts['objects']})")
    check(facts["images"] == 3, f"and every image ({facts['images']})")
    check(facts["spatial"] == 2, "with 2D coordinates")

    names = db.images(adata)
    check(names == ["B/02/0", "B/03/0", "B/04/0"], f"the images are named by component: {names}")

    one = db.select_image(adata, "B/03/0")
    check(one.n_obs == 120, f"one image is one image ({one.n_obs})")
    check(
        set(one.obs[db.IMAGE_KEY].astype(str)) == {"B/03/0"},
        "and carries nothing from the others — coordinates are per image, so mixing "
        "them would join objects that are merely in the same corner of different wells",
    )
    # A view raises the moment anything writes to it, and everything downstream does.
    one.obs["scratch"] = 1
    check("scratch" in one.obs, "it is a copy, so the analysis can write into it")

    check(db.select_image(adata, None).n_obs == 360, "no image chosen means all of them")
    try:
        db.select_image(adata, "Z/99/0")
        check(False, "an image that is not there is refused")
    except ValueError as exc:
        check("no objects" in str(exc), f"an image that is not there is refused: {exc}")

    small = db.subsample(one, 40)
    check(small.n_obs == 40, f"subsampling takes the count asked for ({small.n_obs})")
    check(
        list(small.obs_names) == sorted(small.obs_names, key=list(one.obs_names).index),
        "keeping plate order, which is what makes the scatter readable",
    )
    check(
        len(set(small.obs_names)) == 40 and not np.array_equal(
            np.asarray(small.obsm["spatial"]), np.asarray(one.obsm["spatial"][:40])
        ),
        "drawn from the whole table rather than off the top of it",
    )
    check(db.subsample(one, 500).n_obs == 120, "a table that already fits is left alone")

    check(
        db.values_of(one, "Solidity").shape == (120,),
        "a feature can be read out by name",
    )
    check(db.values_of(one, "label").shape == (120,), "and so can an obs column")
    try:
        db.values_of(one, "nothing")
        check(False, "an unknown column is refused")
    except KeyError:
        check(True, "an unknown column is refused by name")


def test_the_analysis(directory: Path) -> None:
    print("grouping and the spatial statistics")

    if not (_has("scanpy") and _has("squidpy")):
        print("  skip scanpy/squidpy are not installed — the dashboard runs in its own env")
        return

    path = _plate_file(directory, images=1, per_image=300)
    adata = db.select_image(db.read(path), "B/02/0")

    comps = db.prepare(adata)
    check("X_pca" in adata.obsm, f"the features are scaled and reduced ({comps} components)")
    check(
        abs(float(np.nanmean(adata.X))) < 0.1,
        f"scaled, because integrated intensity is six figures and solidity is below one "
        f"(mean {float(np.nanmean(adata.X)):.3f})",
    )

    method = db.cluster(adata, resolution=1.0)
    groups = list(adata.obs[db.CLUSTER_KEY].cat.categories)
    check(len(groups) >= 2, f"the two populations are separated ({method}, {len(groups)} groups)")

    graph = db.build_graph(adata, "delaunay")
    check("spatial_connectivities" in adata.obsp, f"a graph is built ({graph})")
    degrees = np.asarray(adata.obsp["spatial_connectivities"].sum(axis=1)).ravel()
    check(
        5.0 < float(np.median(degrees)) < 7.0,
        f"a triangulation gives about six neighbours each ({float(np.median(degrees)):.1f})",
    )
    check(
        "nearest neighbours" in db.build_graph(adata, "knn", n_neighs=8),
        "knn is available too, and says so",
    )
    db.build_graph(adata, "delaunay")

    zscore, counts = db.neighbourhood_enrichment(adata, n_perms=100)
    check(zscore.shape == (len(groups), len(groups)), f"one z per pair of groups {zscore.shape}")
    check(
        float(np.min(np.diag(zscore))) > 0,
        f"the two blobs each keep their own company, which is what the data says "
        f"(diagonal {np.round(np.diag(zscore), 1).tolist()})",
    )
    check(int(counts.sum()) > 0, "and the raw neighbour counts come back too")

    moran = db.spatial_autocorrelation(adata, n_perms=50)
    check(len(moran) == adata.n_vars, f"Moran's I for every feature ({len(moran)})")
    check("I" in moran.columns, f"with the statistic itself: {list(moran.columns)[:3]}")
    check(
        float(moran["I"].max()) > 0.2,
        f"and the blob-separating features score high ({float(moran['I'].max()):.2f})",
    )

    bins = db.bin_column(adata, "Solidity", bins=4)
    check(
        adata.obs[db.CLUSTER_KEY].nunique() == 4,
        f"binning a measurement is the other way to group ({bins})",
    )
    check(
        all("Solidity" in str(name) for name in adata.obs[db.CLUSTER_KEY].cat.categories),
        f"named after the column, so the plot legend explains itself: "
        f"{list(adata.obs[db.CLUSTER_KEY].cat.categories)}",
    )


def test_blanks_do_not_poison_the_pca(directory: Path) -> None:
    print("a feature one image did not measure")

    if not _has("scanpy"):
        print("  skip scanpy is not installed")
        return
    import anndata as ad
    import pandas as pd

    rng = np.random.default_rng(1)
    matrix = rng.normal(size=(60, 4)).astype(np.float32)
    matrix[:, 2] = np.nan  # a channel this image did not have
    adata = ad.AnnData(
        X=matrix,
        obs=pd.DataFrame(index=[str(i) for i in range(60)]),
        var=pd.DataFrame(index=["a", "b", "missing", "d"]),
    )
    adata.obsm["spatial"] = rng.normal(size=(60, 2))

    db.prepare(adata, n_comps=3)
    check(
        bool(np.isfinite(adata.obsm["X_pca"]).all()),
        "a blank column does not turn the whole PCA into NaN — which is what an "
        "outer-joined plate file would otherwise do the moment two cycles differ",
    )


def main() -> int:
    directory = Path(tempfile.mkdtemp(prefix="mv-dashboard-"))
    try:
        test_starting_it()
        print()
        for name, test in (
            ("slice", test_reading_and_slicing),
            ("analysis", test_the_analysis),
            ("blanks", test_blanks_do_not_poison_the_pca),
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
