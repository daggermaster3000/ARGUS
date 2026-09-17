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


def test_running_it_standalone() -> None:
    print("starting it on its own")

    argv = db.module_command("objects.h5ad", port=8600)
    check(
        argv[1:3] == ["-m", "microscopy_viewer.dashboard"],
        f"the shortcut runs this package, not streamlit directly: {argv[1:3]}",
    )
    check("objects.h5ad" in argv and "8600" in argv, f"with the file and the port: {argv[3:]}")
    check(
        "--no-browser" not in argv,
        "and opens a browser by default, which is the whole point of a shortcut",
    )
    check(
        "--no-browser" in db.module_command("x.h5ad", open_browser=False),
        "unless it is told not to",
    )
    check(
        db.module_command(None)[3] == "--port",
        f"no file is still a valid start: {db.module_command(None)[1:]}",
    )

    # The page is served to this machine and no further. Streamlit's own default
    # binds every interface, which on a university network puts an unauthenticated
    # page holding the object tables in front of anyone who can route to it.
    streamlit_argv = db.command("x.h5ad", port=8600)
    check(
        "--server.address" in streamlit_argv
        and streamlit_argv[streamlit_argv.index("--server.address") + 1] == "127.0.0.1",
        "it binds to loopback only, not to every interface",
    )


def test_the_warm_up() -> None:
    print("paying the compiler up front")

    if not _has("scanpy"):
        print("  skip scanpy is not installed")
        return

    first = db.warm_up()
    check(first >= 0.0, f"the warm-up runs ({first:.1f} s)")
    second = db.warm_up()
    check(
        second == 0.0,
        "and only once per process — two callers racing it would compile twice",
    )

    # What it bought: scanpy's connectivity kernels cost twelve to fifteen seconds
    # to compile whatever the size of the data, so a small analysis afterwards is
    # a hundred times faster than the same one cold.
    import anndata as ad

    rng = np.random.default_rng(0)
    toy = ad.AnnData(X=rng.normal(size=(800, 8)).astype(np.float32))
    toy.obsm["X_pca"] = np.asarray(toy.X)
    import time as _time

    started = _time.perf_counter()
    db.neighbours(toy, n_neighbors=10)
    elapsed = _time.perf_counter() - started
    check(elapsed < 3.0, f"a real graph afterwards takes under three seconds ({elapsed:.1f} s)")


def test_the_neighbour_graph_is_reused() -> None:
    print("not building the same graph twice")

    if not _has("scanpy"):
        print("  skip scanpy is not installed")
        return
    import anndata as ad

    rng = np.random.default_rng(2)
    adata = ad.AnnData(X=rng.normal(size=(600, 8)).astype(np.float32))
    adata.obsm["X_pca"] = np.asarray(adata.X)

    first = db.neighbours(adata, n_neighbors=10)
    check("neighbours" in first and "already built" not in first, f"the first call builds it: {first}")
    again = db.neighbours(adata, n_neighbors=10)
    check("already built" in again, f"the second reuses it: {again}")
    check(
        "already built" not in db.neighbours(adata, n_neighbors=20),
        "but a different k is a different graph and is built",
    )
    check(
        "already built" not in db.neighbours(adata, n_neighbors=20, force=True),
        "and force rebuilds whatever is there",
    )

    check(
        db.EXACT_NEIGHBOURS_LIMIT > 0,
        "there is a size past which the approximate search takes over",
    )


def test_colours_are_shared() -> None:
    print("one colour per group, everywhere")

    groups = [str(i) for i in range(17)]
    mapping = db.colour_map(groups)
    check(len(mapping) == 17, "a colour for every group")
    check(
        len(set(mapping.values())) == 17,
        f"all of them distinct ({len(set(mapping.values()))}) — matplotlib's own cycle is "
        "ten long, so a plate that clusters into seventeen drew 0 and 10 the same blue",
    )
    check(mapping["0"] != mapping["10"], "which is exactly the pair that used to clash")
    check(
        all(colour.startswith("#") and len(colour) == 7 for colour in mapping.values()),
        "hex, which is what both matplotlib and Vega take",
    )
    check(
        db.colour_map(groups) == mapping,
        "and the same every time, or a plot redrawn would recolour itself",
    )
    check(
        db.colour_map(["a", "b"])["a"] == db.colour_map(["a", "z"])["a"],
        "assigned by position, so two plots of the same groups in the same order agree",
    )

    many = db.colour_map([str(i) for i in range(len(db.PALETTE) + 5)])
    check(
        list(many.values()).count(db.OVERFLOW_COLOUR) == 5,
        "past the end of the palette the extras go grey, rather than repeating a "
        "colour that already means something else",
    )

    check(
        db.colours_for(["1", "2", "nope"], mapping)
        == [mapping["1"], mapping["2"], db.OVERFLOW_COLOUR],
        "looking values up gives one colour per value, and grey for a stranger",
    )

    domain, scheme = db.scale_for(mapping)
    check(
        domain == groups and scheme == [mapping[g] for g in groups],
        "and a Vega scale comes out in the mapping's own order",
    )


def test_colours_follow_the_categories(directory: Path) -> None:
    print("the colour map comes off the data")

    if not _has("anndata"):
        print("  skip anndata is not installed")
        return
    import pandas as pd

    path = _plate_file(directory, images=2, per_image=40)
    adata = db.stratified_subsample(db.read(path), 40)
    adata.obs[db.CLUSTER_KEY] = pd.Categorical(
        ["a", "b"] * (adata.n_obs // 2), categories=["a", "b"]
    )

    mapping = db.group_colours(adata)
    check(list(mapping) == ["a", "b"], f"in the category order, not alphabetical luck: {list(mapping)}")
    check(
        db.group_colours(adata, "well") != mapping or adata.obs["well"].nunique() == 2,
        "a different column gets its own map",
    )
    check(
        len(db.group_colours(adata, "well")) == adata.obs["well"].nunique(),
        "with one entry per well",
    )


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


def test_the_whole_plate(directory: Path) -> None:
    print("every well at once")

    if not _has("anndata"):
        print("  skip anndata is not installed")
        return

    check(db.well_of("G/07/0") == "G/07", "an image names its well")
    check(
        db.well_of("G/07/3") == db.well_of("G/07/0"),
        "and the seven 4i cycles of one well are one well, not seven conditions",
    )
    check(db.row_column("G/07/0") == ("G", "07"), "which lays out as a plate")
    check(db.well_of("odd") == "odd", "something that is not a well path is left alone")

    # Wildly unequal images, which is what a real plate is: this one runs from 16
    # objects in one well to 46 394 in another.
    path = _plate_file(directory, images=3, per_image=120)
    adata = db.read(path)
    import anndata as ad

    small = adata[adata.obs[db.IMAGE_KEY].astype(str) != "B/04/0"].copy()
    tiny = adata[adata.obs[db.IMAGE_KEY].astype(str) == "B/04/0"][:9].copy()
    uneven = ad.concat([small, tiny])

    picked = db.stratified_subsample(uneven, per_image=50)
    counts = picked.obs[db.IMAGE_KEY].value_counts()
    check(
        set(counts.index) == {"B/02/0", "B/03/0", "B/04/0"},
        f"every image is represented, however small ({dict(counts)})",
    )
    check(int(counts.max()) == 50, f"a big image is capped at the quota ({int(counts.max())})")
    check(
        int(counts["B/04/0"]) == 9,
        "and an image smaller than the quota keeps everything it has — the point is to "
        "compare wells, not to let the biggest one draw the map",
    )
    check(
        "well" in picked.obs and "row" in picked.obs and "column" in picked.obs,
        f"the plate position is added, for colouring by it: {list(picked.obs.columns)}",
    )

    flat = db.subsample(uneven, 50)
    check(
        flat.obs[db.IMAGE_KEY].nunique() < 3 or int(flat.obs[db.IMAGE_KEY].value_counts().max()) > 20,
        "a flat sample of the same data does not spread over the images, which is why "
        "the stratified one exists",
    )


def test_composition_and_the_plate_grid(directory: Path) -> None:
    print("what each well is made of")

    if not _has("anndata"):
        print("  skip anndata is not installed")
        return
    import pandas as pd

    path = _plate_file(directory, images=3, per_image=60)
    adata = db.stratified_subsample(db.read(path), 60)
    # A grouping that does not need scanpy: the two blobs are large and small.
    adata.obs[db.CLUSTER_KEY] = pd.Categorical(
        np.where(db.values_of(adata, "Voxels") > 650, "big", "small"), categories=["big", "small"]
    )

    shares = db.composition(adata, by="well")
    check(shares.shape == (3, 2), f"a row per well, a column per group {shares.shape}")
    check(
        bool(np.allclose(shares.sum(axis=1), 1.0)),
        "shares, not counts — the wells hold different numbers of objects and a count "
        "table is a table of how full each well was",
    )
    check(
        0.3 < float(shares["big"].mean()) < 0.7,
        f"and the halves come out about even, as the fixture made them "
        f"({float(shares['big'].mean()):.2f})",
    )

    grid = db.plate_grid(shares, "big")
    check(list(grid.index) == ["B"], f"laid out with rows down ({list(grid.index)})")
    check(list(grid.columns) == ["02", "03", "04"], f"and columns across ({list(grid.columns)})")
    check(
        float(grid.loc["B", "02"]) == float(shares.loc["B/02", "big"]),
        "with each well in its own place",
    )

    # A plate is rarely full, and an absent well is not a well holding none of this.
    sparse = shares.drop(index="B/03")
    holey = db.plate_grid(sparse, "big")
    check(
        bool(np.isnan(holey.to_numpy(dtype=float)).any()) if "03" in holey.columns else True,
        "a well the plate does not have is blank rather than zero",
    )

    try:
        db.plate_grid(shares, "nothing")
        check(False, "an unknown group is refused")
    except KeyError:
        check(True, "an unknown group is refused by name")


def test_the_embedding(directory: Path) -> None:
    print("the UMAP")

    if not (_has("scanpy") and _has("umap")):
        print("  skip scanpy/umap-learn are not installed")
        return

    path = _plate_file(directory, images=3, per_image=120)
    adata = db.stratified_subsample(db.read(path), 100)
    # Captured before embedding: embed() scales the matrix in place, so afterwards
    # every feature is in standard deviations and a threshold in voxels finds
    # nothing at all.
    big = db.values_of(adata, "Voxels") > 650
    how = db.embed(adata, n_neighbors=10, min_dist=0.3)

    check("X_umap" in adata.obsm, f"there is an embedding ({how})")
    check(adata.obsm["X_umap"].shape == (adata.n_obs, 2), "two dimensions, one row per object")
    check(bool(np.isfinite(adata.obsm["X_umap"]).all()), "and no NaN in it")
    check("X_pca" in adata.obsm, "run on the components, so a duplicated column is not counted twice")

    # The fixture is two well-separated blobs; an embedding that does not pull
    # them apart is an embedding that is not working.
    check(
        int(big.sum()) > 10 and int((~big).sum()) > 10,
        f"both populations are in the sample ({int(big.sum())} large, {int((~big).sum())} small)",
    )
    centres = [adata.obsm["X_umap"][big].mean(axis=0), adata.obsm["X_umap"][~big].mean(axis=0)]
    apart = float(np.linalg.norm(centres[0] - centres[1]))
    spread = float(np.std(adata.obsm["X_umap"]))
    check(apart > spread, f"the two populations land apart ({apart:.1f} against a spread of {spread:.1f})")


def test_well_views(directory: Path) -> None:
    print("every well, cached in the file")

    if not (_has("anndata") and _has("scanpy")):
        print("  skip anndata/scanpy are not installed")
        return
    import pandas as pd

    path = _plate_file(directory, images=4, per_image=150)
    source = db.read(path)
    whole = db.stratified_subsample(source, 100)
    db.prepare(whole)
    db.cluster(whole, resolution=1.0)

    frame, meta = db.well_views(whole, per_well=40, source=source)
    check(
        set(frame.columns) >= {"image", "well", "x", "y", "cluster", "label"},
        f"the view is points, not a picture: {sorted(frame.columns)}",
    )
    check(frame["image"].nunique() == 4, f"one panel per image ({frame['image'].nunique()})")
    check(
        int(frame.groupby("image").size().max()) <= 40,
        "each capped at the quota so the grid stays quick",
    )
    check(
        meta["totals"]["B/02/0"] == 150,
        f"the totals are the well's real size, not the sample's — a tooltip saying a "
        f"well holds forty objects when it holds thousands is worse than none "
        f"({meta['totals']['B/02/0']})",
    )
    check(
        len(meta["colors"]) == len(meta["clusters"]),
        "with a colour per cluster, so the panels match the rest of the page",
    )
    check(
        frame["well"].iloc[0] == db.well_of(frame["image"].iloc[0]),
        "and the well is derived, for the tooltip",
    )

    # One clustering for the plate, or the panels could not be compared.
    check(
        set(frame["cluster"]) <= set(meta["clusters"]),
        "every point's cluster is one of the plate's, not a per-well grouping",
    )

    # Round trip through the file.
    db.store_well_views(source, frame, meta)
    check(db.WELL_VIEWS_KEY in source.uns, "stored in uns")
    target = db.save_well_views(directory / "with_views.h5ad", source)
    check(target.exists(), f"written ({target.name})")

    reopened = db.read(target)
    back, back_meta = db.load_well_views(reopened)
    check(back is not None and len(back) == len(frame), f"read back whole ({len(back)})")
    check(
        bool(np.allclose(np.sort(back["x"].to_numpy()), np.sort(frame["x"].to_numpy()))),
        "with the same coordinates",
    )
    check(
        back_meta["clusters"] == meta["clusters"] and back_meta["colors"] == meta["colors"],
        "and the same clusters and colours, so the grid looks the same next time",
    )
    check(back_meta["method"] == meta["method"], f"and the method ({back_meta['method']})")
    check(back_meta["totals"] == meta["totals"], "and the per-well totals")
    check(reopened.n_obs == source.n_obs, "the objects themselves are untouched by the write")

    empty, nothing = db.load_well_views(db.read(path))
    check(empty is None and nothing == {}, "a file without them says so rather than raising")

    try:
        db.well_views(source, per_well=10)
        check(False, "a file that has not been clustered is refused")
    except ValueError as exc:
        check("clustered" in str(exc), f"a file that has not been clustered is refused: {exc}")


def main() -> int:
    directory = Path(tempfile.mkdtemp(prefix="mv-dashboard-"))
    try:
        test_starting_it()
        print()
        test_running_it_standalone()
        print()
        test_the_warm_up()
        print()
        test_the_neighbour_graph_is_reused()
        print()
        test_colours_are_shared()
        print()
        for name, test in (
            ("slice", test_reading_and_slicing),
            ("analysis", test_the_analysis),
            ("plate", test_the_whole_plate),
            ("composition", test_composition_and_the_plate_grid),
            ("umap", test_the_embedding),
            ("colours", test_colours_follow_the_categories),
            ("wellviews", test_well_views),
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
