"""The spatial dashboard: what it computes, and how the viewer starts it.

A segmentation run ends with an ``.h5ad`` — one row per object, morphology and
intensity in ``X``, the centroid in ``obsm["spatial"]``. What people want next is
not another table but the questions squidpy answers: which objects sit next to
which, whether a phenotype clusters in space or is scattered through the well,
how far its influence reaches. That is a dashboard rather than a panel, because
it is exploratory — twenty plots looked at once and then not again.

This module is the part with no Streamlit in it: loading, filtering, clustering
and the squidpy calls, each one testable on its own. The page that draws them is
:mod:`microscopy_viewer.dashboard_app`, and :func:`launch` starts it.

**The plate is a second question.** Everything spatial is per image, but the
question *which wells differ from which* is not spatial at all — it is asked of
all the objects at once, in feature space, which is what :func:`embed` and
:func:`composition` are for. The two do not mix: a UMAP says two wells hold
different phenotypes, a neighbourhood graph says where in one well they sit.

**Coordinates are per image.** ``obsm["spatial"]`` holds positions within one
well, so two wells overlap in that space. Every spatial statistic here is
computed on one image at a time, and :func:`select_image` is how the page gets
one — a neighbourhood graph built across a whole plate would join objects in
different wells because they happen to sit at the same corner of each.

**These are not genes.** The columns are morphology and intensity, tens of them
rather than twenty thousand, already on comparable scales and with no counts to
normalise. So the preparation is a z-score and a PCA, not the log1p-and-
highly-variable-genes recipe a scanpy tutorial starts with: applying that here
would take the log of a solidity.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .utils import get_logger

logger = get_logger("dashboard")

#: Column the image is recorded under in a combined file. Kept in step with
#: :data:`microscopy_viewer.analysis.IMAGE_KEY`.
IMAGE_KEY = "image"

#: Column the computed clusters are written to.
CLUSTER_KEY = "cluster"

#: Objects above which a run is subsampled before the graph work. A neighbourhood
#: enrichment on fifty thousand objects is minutes of permutation; on eight
#: thousand it is seconds, and the answer is the same shape.
DEFAULT_MAX_OBJECTS = 20_000

#: Principal components kept before clustering. The features are tens, not
#: thousands, and past this the components are measurement noise.
DEFAULT_COMPONENTS = 15

#: Neighbours for the expression graph the clustering runs on.
DEFAULT_NEIGHBOURS = 15

#: Objects taken per image when the whole plate is embedded. Per image rather
#: than overall: this plate runs from 16 objects in one well to 46 394 in another,
#: and a flat sample of the lot would be a picture of the big wells with the small
#: ones invisible in it.
DEFAULT_PER_IMAGE = 750

#: UMAP neighbours and minimum distance. The defaults umap-learn ships, which are
#: the ones every published figure used, so a plot made here is comparable to one
#: made in a notebook.
DEFAULT_UMAP_NEIGHBOURS = 15
DEFAULT_MIN_DIST = 0.5

#: How the spatial graph is built. Delaunay is the honest default for segmented
#: objects: nuclei touch their neighbours, and a fixed radius in micrometres
#: either misses them in a sparse field or joins half the well in a dense one.
GRAPH_MODES = ("delaunay", "knn", "radius")


# ---------------------------------------------------------------------------
# Loading and slicing
# ---------------------------------------------------------------------------


def missing_packages() -> list[str]:
    """Which of the dashboard's own dependencies are not installed."""
    missing = []
    for name in ("streamlit", "anndata", "scanpy", "squidpy"):
        try:
            __import__(name)
        except ImportError:
            missing.append(name)
    return missing


def install_hint(missing: Sequence[str] | None = None) -> str:
    """What to type to make the dashboard work."""
    names = list(missing) if missing is not None else missing_packages()
    if not names:
        return ""
    return (
        f"The dashboard needs {', '.join(names)}.\n\n"
        'pip install "microscopy-viewer[dashboard]"\n\n'
        "or: pip install " + " ".join(names)
    )


def read(path: str | Path):
    """Read an ``.h5ad`` written by the analysis panel or a batch run."""
    import anndata as ad

    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"{source} does not exist")
    adata = ad.read_h5ad(source)
    logger.info("dashboard: read %s (%d x %d)", source.name, adata.n_obs, adata.n_vars)
    return adata


def images(adata) -> list[str]:
    """The images in a file, in plate order. Empty for a single-image file."""
    if IMAGE_KEY not in adata.obs:
        return []
    return sorted({str(value) for value in adata.obs[IMAGE_KEY]})


def describe(adata) -> dict[str, Any]:
    """A summary line's worth of facts about a file."""
    spatial = adata.obsm.get("spatial")
    return {
        "objects": int(adata.n_obs),
        "features": int(adata.n_vars),
        "images": len(images(adata)),
        "spatial": None if spatial is None else int(np.asarray(spatial).shape[1]),
        "obs": [str(name) for name in adata.obs.columns],
    }


def select_image(adata, image: str | None):
    """One image's objects, or everything when *image* is None.

    A copy, because everything downstream writes into ``obs`` and ``obsp``, and a
    view of an AnnData raises the moment it is written to.
    """
    if not image or IMAGE_KEY not in adata.obs:
        return adata.copy()
    mask = adata.obs[IMAGE_KEY].astype(str) == str(image)
    chosen = adata[mask.to_numpy()].copy()
    if chosen.n_obs == 0:
        raise ValueError(f"no objects belong to {image!r}")
    return chosen


def subsample(adata, limit: int = DEFAULT_MAX_OBJECTS, seed: int = 0):
    """At most *limit* objects, drawn at random, or the whole thing when it fits.

    Random rather than the first N: a table is written in label order, which runs
    roughly top-left to bottom-right, so the first N objects are a corner of the
    well and their neighbourhood statistics describe that corner.
    """
    if adata.n_obs <= int(limit):
        return adata
    chosen = np.random.default_rng(seed).choice(adata.n_obs, size=int(limit), replace=False)
    chosen.sort()  # keeps obs in plate order, which makes the plots readable
    logger.info("dashboard: subsampled %d objects to %d", adata.n_obs, limit)
    return adata[chosen].copy()


def feature_names(adata) -> list[str]:
    return [str(name) for name in adata.var_names]


def obs_columns(adata, numeric_only: bool = False) -> list[str]:
    import pandas as pd

    return [
        str(name)
        for name in adata.obs.columns
        if not numeric_only or pd.api.types.is_numeric_dtype(adata.obs[name])
    ]


def values_of(adata, name: str) -> np.ndarray:
    """A column by name, whether it is a feature or something in ``obs``."""
    if name in adata.var_names:
        column = adata[:, name].X
        return np.asarray(column.todense() if hasattr(column, "todense") else column).ravel()
    if name in adata.obs:
        return np.asarray(adata.obs[name].to_numpy())
    raise KeyError(f"{name!r} is neither a feature nor an obs column")


# ---------------------------------------------------------------------------
# Phenotypes
# ---------------------------------------------------------------------------


def prepare(adata, n_comps: int = DEFAULT_COMPONENTS, features: Sequence[str] | None = None):
    """Z-score the features and run a PCA, in place. Returns the components kept.

    No log1p and no highly-variable selection: those belong to counts, and these
    are micrometres and grey levels. Scaling is still needed — integrated
    intensity is six figures and solidity is below one, so an unscaled PCA would
    be a PCA of whichever column has the largest units.
    """
    import scanpy as sc

    if features:
        keep = [name for name in features if name in set(adata.var_names)]
        if not keep:
            raise ValueError("none of the chosen features are in this file")
        adata._inplace_subset_var(list(keep))

    matrix = np.asarray(adata.X, dtype=np.float64)
    # A feature that is blank for this image cannot be scaled, and a NaN loose in
    # the matrix silently turns the whole PCA into NaN.
    finite = np.isfinite(matrix)
    if not finite.all():
        import warnings

        with warnings.catch_warnings():
            # A column that is blank for every object of this image is the case
            # being handled, not a surprise; nanmean says so and is then ignored.
            warnings.simplefilter("ignore", RuntimeWarning)
            column_means = np.nanmean(np.where(finite, matrix, np.nan), axis=0)
        column_means = np.where(np.isfinite(column_means), column_means, 0.0)
        matrix = np.where(finite, matrix, column_means)
        adata.X = matrix.astype(np.float32)

    sc.pp.scale(adata, max_value=10.0)
    comps = int(max(2, min(int(n_comps), adata.n_vars - 1, adata.n_obs - 1)))
    sc.pp.pca(adata, n_comps=comps)
    return comps


def cluster(
    adata,
    resolution: float = 1.0,
    n_neighbors: int = DEFAULT_NEIGHBOURS,
    key: str = CLUSTER_KEY,
) -> str:
    """Group the objects by what they look like, into ``obs[key]``.

    Leiden when it is installed, k-means otherwise: squidpy's neighbourhood
    statistics need a categorical label per object and do not care which method
    produced it, so a missing leidenalg should cost resolution, not the feature.
    """
    import scanpy as sc

    if "X_pca" not in adata.obsm:
        prepare(adata)
    sc.pp.neighbors(adata, n_neighbors=int(min(n_neighbors, max(2, adata.n_obs - 1))))
    try:
        sc.tl.leiden(adata, resolution=float(resolution), key_added=key, flavor="igraph",
                     n_iterations=2, directed=False)
        method = "leiden"
    except Exception:
        logger.info("leiden is unavailable; clustering with k-means", exc_info=True)
        from sklearn.cluster import KMeans

        # Resolution is not a cluster count and the two cannot be made to agree,
        # but it is the knob the page offers, so it is mapped onto one rather than
        # ignored. Scaled so the default lands near what leiden gives on this kind
        # of data -- a dozen-ish groups -- instead of the two that a gentler
        # mapping produces, which is not an answer to anything.
        k = int(max(2, min(25, round(12 * float(resolution)))))
        labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(adata.obsm["X_pca"])
        adata.obs[key] = [str(value) for value in labels]
        method = f"k-means (k={k})"
    adata.obs[key] = adata.obs[key].astype("category")
    logger.info("dashboard: %s found %d cluster(s)", method, adata.obs[key].nunique())
    adata.uns[f"{key}_method"] = method
    return method


def clustered_with_leiden(adata, key: str = CLUSTER_KEY) -> bool:
    """Whether the grouping came from leiden rather than the k-means fallback."""
    return str(adata.uns.get(f"{key}_method", "")).startswith("leiden")


def bin_column(adata, column: str, bins: int = 4, key: str = CLUSTER_KEY) -> str:
    """Turn a measurement into categories, as an alternative to clustering.

    Quartiles of solidity are a perfectly good grouping to ask a neighbourhood
    question about, and they are one the user can explain, which a leiden cluster
    on fifteen principal components is not.
    """
    import pandas as pd

    values = pd.Series(values_of(adata, column).astype(float))
    try:
        categories = pd.qcut(values, q=int(bins), duplicates="drop")
    except ValueError:
        categories = pd.cut(values, bins=int(bins))
    labels = [f"{column} q{i + 1}" for i in range(len(categories.cat.categories))]
    adata.obs[key] = pd.Categorical(
        categories.cat.rename_categories(labels).astype(str),
        categories=labels,
    )
    adata.uns[f"{key}_method"] = f"{len(labels)} bins of {column}"
    return str(adata.uns[f"{key}_method"])


# ---------------------------------------------------------------------------
# The whole plate
# ---------------------------------------------------------------------------


def well_of(image: str) -> str:
    """``G/07/0`` -> ``G/07``: the well an image belongs to.

    Worth separating from the image because the seven 4i cycles of one well are
    the same cells, so colouring a plate-wide embedding by image would show seven
    points for every object and invite them to be read as seven conditions.
    """
    parts = str(image).strip("/").split("/")
    return "/".join(parts[:2]) if len(parts) >= 2 else str(image)


def row_column(image: str) -> tuple[str, str]:
    """``G/07/0`` -> ``("G", "07")``, for laying a result out as a plate."""
    parts = str(image).strip("/").split("/")
    return (parts[0], parts[1]) if len(parts) >= 2 else (str(image), "")


def stratified_subsample(adata, per_image: int = DEFAULT_PER_IMAGE, seed: int = 0):
    """At most *per_image* objects from each image, drawn at random.

    Equal footing rather than proportional representation: the point of a
    plate-wide embedding is to compare wells, and a well with forty-six thousand
    objects would otherwise draw the map that a well with sixteen is then judged
    against. Wells smaller than the quota keep everything they have.
    """
    if IMAGE_KEY not in adata.obs:
        return subsample(adata, per_image, seed=seed)

    rng = np.random.default_rng(seed)
    labels = adata.obs[IMAGE_KEY].astype(str).to_numpy()
    chosen: list[np.ndarray] = []
    for name in sorted(set(labels)):
        where = np.flatnonzero(labels == name)
        if where.size > int(per_image):
            where = rng.choice(where, size=int(per_image), replace=False)
        chosen.append(where)
    picked = np.sort(np.concatenate(chosen)) if chosen else np.empty(0, dtype=int)
    logger.info(
        "dashboard: %d object(s) from %d image(s), at most %d each",
        picked.size,
        len(set(labels)),
        per_image,
    )
    out = adata[picked].copy()
    out.obs["well"] = [well_of(name) for name in out.obs[IMAGE_KEY].astype(str)]
    out.obs["row"] = [row_column(name)[0] for name in out.obs[IMAGE_KEY].astype(str)]
    out.obs["column"] = [row_column(name)[1] for name in out.obs[IMAGE_KEY].astype(str)]
    for name in ("well", "row", "column"):
        out.obs[name] = out.obs[name].astype("category")
    return out


def embed(
    adata,
    n_neighbors: int = DEFAULT_UMAP_NEIGHBOURS,
    min_dist: float = DEFAULT_MIN_DIST,
    seed: int = 0,
) -> str:
    """Lay the objects out by what they look like, into ``obsm["X_umap"]``.

    Runs on the principal components rather than the raw columns, which is what
    keeps the neighbour search honest when two features are the same measurement
    twice — mean and median intensity of one channel are nearly the same column,
    and a distance computed over both counts it twice.
    """
    import scanpy as sc

    if "X_pca" not in adata.obsm:
        prepare(adata)
    sc.pp.neighbors(adata, n_neighbors=int(min(n_neighbors, max(2, adata.n_obs - 1))))
    sc.tl.umap(adata, min_dist=float(min_dist), random_state=int(seed))
    logger.info("dashboard: UMAP of %d object(s)", adata.n_obs)
    return f"UMAP, {n_neighbors} neighbours, min_dist {min_dist:g}"


def composition(adata, key: str = CLUSTER_KEY, by: str = "well"):
    """What share of each well's objects fall in each group. Rows sum to 1.

    The plate-level readout: a UMAP shows that phenotypes exist, and this shows
    which wells have them. Shares rather than counts, because the wells hold
    wildly different numbers of objects and a count table is a table of how full
    each well was.
    """
    import pandas as pd

    column = by if by in adata.obs else IMAGE_KEY
    counts = pd.crosstab(adata.obs[column], adata.obs[key])
    totals = counts.sum(axis=1).replace(0, np.nan)
    return counts.div(totals, axis=0).fillna(0.0)


def plate_grid(shares, group: str):
    """One group's share laid out as the plate: rows down, columns across.

    A plate is a physical object and the answer is usually physical too — an edge
    effect, a column of controls, one row that did not take. A bar chart of
    forty-four wells hides that; a grid does not.
    """
    import pandas as pd

    if group not in shares.columns:
        raise KeyError(f"{group!r} is not one of the groups")
    rows: dict[str, dict[str, float]] = {}
    for name, value in shares[group].items():
        row, column = row_column(str(name))
        rows.setdefault(row, {})[column] = float(value)
    frame = pd.DataFrame(rows).T
    frame = frame.reindex(sorted(frame.index))
    return frame.reindex(columns=sorted(frame.columns, key=lambda c: (len(c), c)))


# ---------------------------------------------------------------------------
# Space
# ---------------------------------------------------------------------------


def build_graph(
    adata,
    mode: str = "delaunay",
    n_neighs: int = 6,
    radius_um: float = 30.0,
) -> str:
    """Join each object to its neighbours in the image. Returns what it did."""
    import squidpy as sq

    if "spatial" not in adata.obsm:
        raise ValueError("this file has no obsm['spatial']; it cannot be analysed spatially")
    if str(mode) == "delaunay":
        sq.gr.spatial_neighbors(adata, delaunay=True, coord_type="generic")
        return "Delaunay triangulation"
    if str(mode) == "radius":
        sq.gr.spatial_neighbors(adata, radius=float(radius_um), coord_type="generic")
        return f"every object within {radius_um:g} um"
    sq.gr.spatial_neighbors(adata, n_neighs=int(n_neighs), coord_type="generic")
    return f"{int(n_neighs)} nearest neighbours"


def neighbourhood_enrichment(adata, key: str = CLUSTER_KEY, n_perms: int = 200, seed: int = 0):
    """Which groups sit next to which, against a shuffled null. ``(z, count)``."""
    import squidpy as sq

    sq.gr.nhood_enrichment(adata, cluster_key=key, n_perms=int(n_perms), seed=seed, show_progress_bar=False)
    result = adata.uns[f"{key}_nhood_enrichment"]
    return np.asarray(result["zscore"]), np.asarray(result["count"])


def co_occurrence(adata, key: str = CLUSTER_KEY, interval: int = 30):
    """How the chance of a neighbour of each group changes with distance."""
    import squidpy as sq

    sq.gr.co_occurrence(adata, cluster_key=key, interval=int(interval), show_progress_bar=False)
    result = adata.uns[f"{key}_co_occurrence"]
    return np.asarray(result["occ"]), np.asarray(result["interval"])


def ripley(adata, key: str = CLUSTER_KEY, mode: str = "L", n_simulations: int = 50):
    """Ripley's statistic: is each group clustered in space, or scattered?"""
    import squidpy as sq

    sq.gr.ripley(adata, cluster_key=key, mode=mode, n_simulations=int(n_simulations))
    return adata.uns[f"{key}_ripley_{mode}"]


def spatial_autocorrelation(adata, n_perms: int = 100, n_jobs: int = 1):
    """Moran's I per feature: which measurements vary smoothly across the well.

    The one statistic here that needs no grouping at all, and often the most
    useful: a high Moran's I on a reporter channel says the signal comes in
    patches rather than cell by cell.
    """
    import squidpy as sq

    sq.gr.spatial_autocorr(
        adata, mode="moran", n_perms=int(n_perms), n_jobs=int(n_jobs), show_progress_bar=False
    )
    return adata.uns["moranI"]


def centrality(adata, key: str = CLUSTER_KEY):
    """Degree, closeness and clustering coefficient per group."""
    import squidpy as sq

    sq.gr.centrality_scores(adata, cluster_key=key, show_progress_bar=False)
    return adata.uns[f"{key}_centrality_scores"]


# ---------------------------------------------------------------------------
# Starting the page
# ---------------------------------------------------------------------------


def app_path() -> Path:
    """The Streamlit script."""
    return Path(__file__).with_name("dashboard_app.py")


def command(path: str | Path | None = None, port: int = 8501, python: str | None = None) -> list[str]:
    """The command line that starts the dashboard.

    ``python -m streamlit`` rather than the ``streamlit`` executable: the
    executable is only on PATH when the environment's Scripts directory is, which
    on Windows it frequently is not, while the module is importable wherever the
    package is installed.
    """
    argv = [
        str(python or sys.executable),
        "-m",
        "streamlit",
        "run",
        str(app_path()),
        "--server.port",
        str(int(port)),
        "--server.headless",
        "true",
        # Without this the first run stops at a terminal prompt asking for an
        # email address, and the window never opens.
        "--browser.gatherUsageStats",
        "false",
    ]
    if path is not None:
        argv += ["--", "--file", str(path)]
    return argv


def free_port(start: int = 8501, tries: int = 20) -> int:
    """A port nothing is listening on, so a second dashboard does not fail silently."""
    import socket

    for offset in range(int(tries)):
        port = int(start) + offset
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    return int(start)


def launch(
    path: str | Path | None = None,
    port: int | None = None,
    python: str | None = None,
    open_browser: bool = True,
) -> tuple[subprocess.Popen, str]:
    """Start the dashboard as its own process. Returns ``(process, url)``.

    A separate process on purpose. Streamlit runs its own event loop and would
    fight Qt's inside the viewer; and the spatial work is minutes of CPU that has
    no business blocking the window the images are in.
    """
    missing = missing_packages()
    if missing:
        raise RuntimeError(install_hint(missing))

    chosen = int(port) if port else free_port()
    argv = command(path, port=chosen, python=python)
    url = f"http://localhost:{chosen}"

    creation = 0
    if os.name == "nt":
        # Detached, so closing the viewer does not take the dashboard with it and
        # no console window appears in front of the image.
        creation = getattr(subprocess, "CREATE_NO_WINDOW", 0)

    logger.info("dashboard: %s", " ".join(argv))
    process = subprocess.Popen(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        creationflags=creation,
        cwd=str(Path(__file__).resolve().parent.parent),
    )
    if open_browser:
        import threading
        import webbrowser

        # Streamlit takes a couple of seconds to bind the port; opening the
        # browser straight away gives the user a connection-refused page.
        threading.Timer(3.0, lambda: webbrowser.open(url)).start()
    return process, url


def streamlit_available() -> bool:
    return shutil.which("streamlit") is not None or not missing_packages()
