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

**One colour per group, everywhere.** A cluster is the same colour in the UMAP,
in the well, in the composition bar and in the box plot, because reading these
means carrying a colour from one panel to the next. :func:`colour_map` is the
single place that decides, and its palette is forty long rather than matplotlib's
ten — a plate that clusters into seventeen drew clusters 0 and 10 in the same
blue before it existed.

**A feature's colours are z-scores unless told otherwise.** :func:`prepare`
scales the matrix in place, which is what the clustering needs and is not what
someone colouring by "Mean intensity" expects to see. :func:`colour_scale` is the
choice between the two, and between this image's range and the plate's.

**Every well can be drawn at once.** :func:`well_views` reduces a clustered plate
to a few hundred points per well and :func:`store_well_views` keeps them in the
file, so the plate display is a grid of live panels rather than a wall of
pictures — the objects are still there to be pointed at. The clustering is what
costs minutes and is what the caching saves.

**A clustering can go back into the plate.** :func:`assignment_frame` and
:func:`cluster_run` prepare what :mod:`microscopy_viewer.clusters` needs to paint
each object's phenotype onto the nucleus it was measured from, as another label
set beside the segmentation — which is how a phenotype stops being a row in a
table and becomes something you can see the position of.

**The neighbour search is the whole cost.** Everything else here is seconds; the
k-nearest-neighbour graph the clustering and the UMAP are both built on is not,
and which library computes it matters more than anything else on this page. See
:func:`neighbours`.

**These are not genes.** The columns are morphology and intensity, tens of them
rather than twenty thousand, already on comparable scales and with no counts to
normalise. So the preparation is a z-score and a PCA, not the log1p-and-
highly-variable-genes recipe a scanpy tutorial starts with: applying that here
would take the log of a solidity.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

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

#: Objects above which the approximate neighbour search is used instead of the
#: exact one. Below it scikit-learn is both faster and exact; above it the
#: quadratic term catches up with pynndescent's fixed compilation cost. Measured
#: at 30 000 objects in 15 dimensions: 8 s exact against 35 s approximate.
EXACT_NEIGHBOURS_LIMIT = 60_000

#: How the spatial graph is built. Delaunay is the honest default for segmented
#: objects: nuclei touch their neighbours, and a fixed radius in micrometres
#: either misses them in a sparse field or joins half the well in a dense one.
GRAPH_MODES = ("delaunay", "knn", "radius")


#: Colours groups are drawn in, in order. Forty of them rather than matplotlib's
#: ten: a plate that clusters into seventeen drew cluster 0 and cluster 10 in the
#: same blue, in every plot, with nothing to say they were different.
#:
#: Ordered greedily by CIELAB distance -- each colour is the one furthest from
#: everything already in the list -- so the first twenty are at least 22 deltaE
#: apart, which is well past telling-apart. A palette that merely *has* forty
#: entries is not enough when the tenth and the eleventh are two similar blues.
PALETTE: tuple[str, ...] = (
    "#1f77b4", "#ff7f0e", "#2ca02c", "#843c39", "#e7cb94",
    "#ce6dbd", "#9edae5", "#bcbd22", "#d62728", "#637939",
    "#ff9896", "#7f7f7f", "#a1d99b", "#c5b0d5", "#6b6ecf",
    "#7b4173", "#e7ba52", "#8c6d31", "#393b79", "#6baed6",
    "#b5cf6b", "#c49c94", "#17becf", "#fdae6b", "#c7c7c7",
    "#d6616b", "#de9ed6", "#9c9ede", "#e6550d", "#31a354",
    "#f7b6d2", "#8ca252", "#aec7e8", "#cedb9c", "#9467bd",
    "#74c476", "#756bb1", "#fd8d3c", "#c7e9c0", "#a55194",
)

#: Drawn for a group the palette has run out of colours for. Grey on purpose: an
#: unnamed colour repeated is worse than an obvious "and the rest".
OVERFLOW_COLOUR = "#bbbbbb"


def colour_map(categories: Sequence[Any]) -> dict[str, str]:
    """``{category: hex}``, assigned by position and stable for a given order.

    Every plot on the page takes its colours from one of these, so a cluster is
    the same colour in the UMAP, in the well, in the composition bar and in the
    box plot. Reading a figure means carrying a colour from one panel to the next,
    and a page where that does not hold is a page that cannot be read.
    """
    names = [str(value) for value in categories]
    return {
        name: PALETTE[index] if index < len(PALETTE) else OVERFLOW_COLOUR
        for index, name in enumerate(names)
    }


def colours_for(values: Sequence[Any], mapping: dict[str, str]) -> list[str]:
    """One colour per value, looked up in *mapping*."""
    return [mapping.get(str(value), OVERFLOW_COLOUR) for value in values]


def scale_for(mapping: dict[str, str]) -> tuple[list[str], list[str]]:
    """``(domain, range)`` for a Vega colour scale, in the mapping's own order."""
    return list(mapping), [mapping[name] for name in mapping]


def group_colours(adata, key: str = CLUSTER_KEY) -> dict[str, str]:
    """The colour map for a categorical column of *adata*, in its category order."""
    column = adata.obs[key]
    categories = list(column.cat.categories) if hasattr(column, "cat") else sorted(set(column))
    return colour_map(categories)


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


#: How a feature's colour scale is computed on the page.
#:
#: ``scaled``  the z-scored values the clustering was run on — what the page drew
#:             before this existed, and the right thing when looking at what the
#:             clustering saw. Clipped at 10 standard deviations by the scaling.
#: ``image``   the raw measurement, scaled to this image's own 1-99 %.
#: ``cycle``   the raw measurement, scaled to every well of this cycle.
#: ``file``    the raw measurement, scaled to every object in the file.
COLOUR_SCALES = ("cycle", "file", "image", "scaled")

#: Offered first, and for the same reason the viewer's panel defaults to it: a
#: plate is run to compare wells, and a well scaled against itself cannot be.
DEFAULT_COLOUR_SCALE = "cycle"

COLOUR_SCALE_LABELS = {
    "scaled": "z-scored, this image (what the clustering saw)",
    "image": "raw, this image",
    "cycle": "raw, this cycle — every well",
    "file": "raw, every object in the file",
}


def raw_values(source, adata, name: str) -> np.ndarray:
    """*name* for the objects of *adata*, read off the untouched *source*.

    :func:`prepare` z-scores in place, so by the time anything is drawn the matrix
    holds standard deviations rather than grey levels. The file as it was read is
    still the file as it was read, and the objects are matched back to it by name.
    """
    if name not in source.var_names:
        return values_of(adata, name)
    index = source.obs_names.get_indexer(adata.obs_names)
    column = values_of(source, name)
    out = np.full(adata.n_obs, np.nan, dtype=float)
    found = index >= 0
    out[found] = np.asarray(column, dtype=float)[index[found]]
    return out


def colour_scale(source, adata, name: str, scale: str = "scaled"):
    """``(values, low, high, note)`` for colouring by *name* at the chosen scale."""
    if scale == "scaled" or name not in source.var_names:
        values = values_of(adata, name).astype(float)
        low, high = (float(v) for v in np.nanpercentile(values, [1, 99]))
        return values, low, high, COLOUR_SCALE_LABELS.get("scaled", scale)

    values = raw_values(source, adata, name)
    if scale == "image":
        pool = values
        covers = 1
    else:
        from .analysis import SCOPE_CYCLE, SCOPE_PLATE, scope_components

        everything = images(source) or [""]
        mine = str(adata.obs[IMAGE_KEY].iloc[0]) if IMAGE_KEY in adata.obs else ""
        wanted = (
            everything
            if scale == "file"
            else scope_components(mine, everything, SCOPE_CYCLE if mine else SCOPE_PLATE)
        )
        if IMAGE_KEY in source.obs and wanted:
            mask = source.obs[IMAGE_KEY].astype(str).isin(wanted).to_numpy()
            pool = values_of(source, name).astype(float)[mask]
        else:
            pool = values_of(source, name).astype(float)
        covers = len(wanted)

    finite = pool[np.isfinite(pool)]
    if finite.size == 0:
        return values, 0.0, 1.0, "nothing to scale against"
    low, high = (float(v) for v in np.percentile(finite, [1, 99]))
    if high <= low:
        high = low + 1.0
    note = COLOUR_SCALE_LABELS.get(scale, scale)
    if covers > 1:
        note += f" ({covers} images, {finite.size:,} objects)"
    return values, low, high, note


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

    started = time.perf_counter()
    sc.pp.scale(adata, max_value=10.0)
    comps = int(max(2, min(int(n_comps), adata.n_vars - 1, adata.n_obs - 1)))
    sc.pp.pca(adata, n_comps=comps)
    logger.info(
        "prepare: %d x %d scaled, %d component(s), %.1f s",
        adata.n_obs,
        adata.n_vars,
        comps,
        time.perf_counter() - started,
    )
    return comps


def neighbours(adata, n_neighbors: int = DEFAULT_NEIGHBOURS, force: bool = False) -> str:
    """Build the k-nearest-neighbour graph, and say how. Reused if one already fits.

    This is the slow step, and almost all of the slowness is not the arithmetic.
    scanpy's default backend is pynndescent, whose kernels numba compiles on first
    use — about thirty-five seconds, paid on *every* launch, because the parallel
    kernels cannot be written to the numba cache. The work itself takes two.

    Below :data:`EXACT_NEIGHBOURS_LIMIT` objects, scikit-learn's exact search is
    handed to scanpy instead: no compilation, no approximation, and four times
    faster end to end on a plate-sized sample. Above it the exact search's
    quadratic term wins out and pynndescent is the right tool again.
    """
    import scanpy as sc

    wanted = int(min(n_neighbors, max(2, adata.n_obs - 1)))
    existing = adata.uns.get("neighbors")
    if not force and existing and "connectivities" in adata.obsp:
        already = int((existing.get("params") or {}).get("n_neighbors", 0))
        if already == wanted:
            logger.info("neighbours: reusing the graph already built (k=%d)", wanted)
            return f"{wanted} neighbours (already built)"

    started = time.perf_counter()
    how = "approximate (pynndescent)"
    transformer = None
    if adata.n_obs <= EXACT_NEIGHBOURS_LIMIT:
        try:
            from sklearn.neighbors import KNeighborsTransformer

            transformer = KNeighborsTransformer(
                n_neighbors=wanted, algorithm="auto", n_jobs=-1, metric="euclidean"
            )
            how = "exact (scikit-learn)"
        except ImportError:  # pragma: no cover - sklearn arrives with scanpy
            logger.info("scikit-learn is unavailable; using the approximate search")

    if transformer is not None:
        sc.pp.neighbors(adata, n_neighbors=wanted, transformer=transformer)
    else:
        sc.pp.neighbors(adata, n_neighbors=wanted)
    elapsed = time.perf_counter() - started
    logger.info(
        "neighbours: %s, k=%d over %d object(s) in %.1f s", how, wanted, adata.n_obs, elapsed
    )
    return f"{wanted} neighbours, {how}, {elapsed:.1f} s"


_warmed = threading.Lock()
_warm_done = False


def warm_up() -> float:
    """Force the just-in-time compilation now, on a handful of points.

    scanpy builds its connectivities through numba-compiled kernels, and the first
    call in a process pays twelve to fifteen seconds to compile them whatever the
    size of the data — the same cost for three thousand objects as for thirty
    thousand. Paying it on sixty fake points while the user is still choosing a
    file is the difference between a page that takes fifteen seconds to answer and
    one that takes one.
    """
    global _warm_done

    with _warmed:
        if _warm_done:
            return 0.0
        _warm_done = True

    started = time.perf_counter()
    try:
        import anndata as ad
        import scanpy as sc

        rng = np.random.default_rng(0)
        toy = ad.AnnData(X=rng.normal(size=(60, 4)).astype(np.float32))
        toy.obsm["X_pca"] = np.asarray(toy.X)
        neighbours(toy, n_neighbors=5)
        sc.tl.leiden(toy, key_added="c", flavor="igraph", n_iterations=1, directed=False)
    except Exception:  # noqa: BLE001 - a warm-up that fails costs nothing but time
        logger.debug("warm-up did not complete", exc_info=True)
    elapsed = time.perf_counter() - started
    logger.info("warm-up: compiled in %.1f s; the first real run will not pay this", elapsed)
    return elapsed


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
    neighbours(adata, n_neighbors)
    started = time.perf_counter()
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
    logger.info(
        "cluster: %s found %d group(s) in %.1f s",
        method,
        adata.obs[key].nunique(),
        time.perf_counter() - started,
    )
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
    neighbours(adata, n_neighbors)
    started = time.perf_counter()
    sc.tl.umap(adata, min_dist=float(min_dist), random_state=int(seed))
    elapsed = time.perf_counter() - started
    logger.info("umap: %d object(s) laid out in %.1f s", adata.n_obs, elapsed)
    return f"UMAP, {n_neighbors} neighbours, min_dist {min_dist:g} ({elapsed:.1f} s)"


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


#: Where the per-well views are kept inside the file.
WELL_VIEWS_KEY = "well_views"

#: Objects kept per well for the plate display. Forty-four panels of this many
#: points is a page that draws and hovers instantly; the whole plate would be half
#: a million points held in the browser.
DEFAULT_VIEW_POINTS = 500


def well_views(
    adata,
    per_well: int = DEFAULT_VIEW_POINTS,
    key: str = CLUSTER_KEY,
    seed: int = 0,
    source=None,
):
    """One spatial view per well, as points rather than as a picture.

    Takes a plate that has already been clustered — so the clusters mean the same
    thing in every panel, which is the only reason the wells can be laid out side
    by side and compared at all — and reduces it to what a small panel can draw:
    a few hundred objects per well with their position and their phenotype.

    Points and not a rendering, because a picture cannot be pointed at. The
    expensive part of this is the clustering, and that is what the caching saves;
    redrawing a few hundred points is free.

    *source* is the file the plate was sampled out of, used only to count how many
    objects each well really holds. Counting them in *adata* instead would report
    the size of the sample, and a tooltip saying a well holds 400 objects when it
    holds 46 394 is worse than one saying nothing.
    """
    import pandas as pd

    if "spatial" not in adata.obsm:
        raise ValueError("this file has no obsm['spatial']; there is nothing to lay out")
    if key not in adata.obs:
        raise ValueError(f"nothing has been clustered into obs[{key!r}] yet")

    xy = np.asarray(adata.obsm["spatial"], dtype=float)
    images = (
        adata.obs[IMAGE_KEY].astype(str).to_numpy()
        if IMAGE_KEY in adata.obs
        else np.array([""] * adata.n_obs)
    )
    clusters = adata.obs[key].astype(str).to_numpy()
    labels = (
        adata.obs["label"].to_numpy()
        if "label" in adata.obs
        else np.arange(adata.n_obs)
    )

    # How big each well really is, from the whole file when it was given.
    totals: dict[str, int] = {}
    if source is not None and IMAGE_KEY in source.obs:
        counts = source.obs[IMAGE_KEY].astype(str).value_counts()
        totals = {str(name): int(value) for name, value in counts.items()}

    rng = np.random.default_rng(seed)
    chosen: list[np.ndarray] = []
    for image in sorted(set(images)):
        where = np.flatnonzero(images == image)
        totals.setdefault(str(image), int(where.size))
        if where.size > int(per_well):
            where = np.sort(rng.choice(where, size=int(per_well), replace=False))
        chosen.append(where)
    picked = np.concatenate(chosen) if chosen else np.empty(0, dtype=int)

    names = [str(name) for name in _categories(adata.obs[key])]
    frame = pd.DataFrame(
        {
            "image": images[picked],
            "well": [well_of(name) for name in images[picked]],
            "x": xy[picked, 0],
            "y": xy[picked, 1],
            "cluster": clusters[picked],
            "label": np.asarray(labels)[picked],
        }
    )
    logger.info(
        "well views: %d point(s) over %d well(s), at most %d each",
        len(frame),
        len(totals),
        per_well,
    )
    return frame, {
        "method": str(adata.uns.get(f"{key}_method", "clustering")),
        "clusters": names,
        "colors": [colour_map(names)[name] for name in names],
        "per_well": per_well,
        "totals": totals,
    }


def _categories(column) -> list:
    categories = getattr(getattr(column, "cat", None), "categories", None)
    if categories is not None:
        return list(categories)
    return sorted({str(value) for value in column})


def store_well_views(adata, frame, meta: dict) -> None:
    """Put the views into ``uns`` as flat arrays, which is what h5ad round-trips.

    Flat columns rather than a dict per well: a nested structure of forty-four
    small dicts survives a write and a read only if every reader agrees on the
    nesting, and a table does not need them to.
    """
    adata.uns[WELL_VIEWS_KEY] = {
        "image": np.asarray(frame["image"], dtype=object).astype(str),
        "x": np.asarray(frame["x"], dtype=float),
        "y": np.asarray(frame["y"], dtype=float),
        "cluster": np.asarray(frame["cluster"], dtype=object).astype(str),
        "label": np.asarray(frame["label"]).astype(np.int64),
        "clusters": [str(name) for name in meta.get("clusters", [])],
        "colors": [str(colour) for colour in meta.get("colors", [])],
        "method": str(meta.get("method", "")),
        "per_well": int(meta.get("per_well", 0)),
        "totals_image": [str(name) for name in (meta.get("totals") or {})],
        "totals_count": [int(value) for value in (meta.get("totals") or {}).values()],
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def load_well_views(adata):
    """``(frame, meta)`` from a file that has them, or ``(None, {})``."""
    import pandas as pd

    block = adata.uns.get(WELL_VIEWS_KEY)
    if not isinstance(block, (dict, Mapping)) or "x" not in block:
        return None, {}
    try:
        frame = pd.DataFrame(
            {
                "image": [str(value) for value in block["image"]],
                "x": np.asarray(block["x"], dtype=float),
                "y": np.asarray(block["y"], dtype=float),
                "cluster": [str(value) for value in block["cluster"]],
                "label": np.asarray(block["label"]).astype(int),
            }
        )
    except Exception:
        logger.exception("the stored well views could not be read")
        return None, {}
    frame["well"] = [well_of(name) for name in frame["image"]]
    totals = dict(
        zip(
            [str(name) for name in block.get("totals_image", [])],
            [int(value) for value in block.get("totals_count", [])],
        )
    )
    meta = {
        "method": str(block.get("method", "")),
        "clusters": [str(name) for name in block.get("clusters", [])],
        "colors": [str(colour) for colour in block.get("colors", [])],
        "per_well": int(block.get("per_well", 0) or 0),
        "created": str(block.get("created", "")),
        "totals": totals,
    }
    return frame, meta


def save_well_views(path: str | Path, adata) -> Path:
    """Write the file back with its views in it. Returns where it went.

    The whole ``.h5ad`` is rewritten, which for a plate is a hundred megabytes and
    a few seconds; anndata has no way to add one ``uns`` entry in place, and a
    sidecar file would be one more thing to keep beside the data and lose.
    """
    target = Path(str(path))
    started = time.perf_counter()
    adata.write_h5ad(target)
    logger.info(
        "well views saved into %s (%.0f MB, %.1f s)",
        target.name,
        target.stat().st_size / 1024 / 1024,
        time.perf_counter() - started,
    )
    return target


def assignment_frame(adata, key: str = CLUSTER_KEY):
    """``label``, ``cluster`` and ``image`` per object, ready to be painted back.

    The three columns a clustering needs to find its way home: which object, what
    it was called, and which image it lives in.
    """
    import pandas as pd

    if "label" not in adata.obs:
        raise ValueError(
            "this file has no obs['label'], so its objects cannot be matched to a mask"
        )
    frame = pd.DataFrame(
        {
            "label": adata.obs["label"].to_numpy(),
            "cluster": adata.obs[key].astype(str).to_numpy(),
        }
    )
    frame["image"] = (
        adata.obs[IMAGE_KEY].astype(str).to_numpy()
        if IMAGE_KEY in adata.obs
        else ""
    )
    # Ordered, so the ids painted into the plate match the palette on the page.
    if hasattr(adata.obs[key], "cat"):
        frame["cluster"] = pd.Categorical(
            frame["cluster"], categories=[str(c) for c in adata.obs[key].cat.categories]
        )
    return frame


def cluster_run(adata, key: str = CLUSTER_KEY, source_labels: str = "", **extra):
    """The provenance record for a clustering about to be written into a plate."""
    from .clusters import ClusterRun, cluster_names

    names = cluster_names(adata.obs[key])
    return ClusterRun(
        method=str(adata.uns.get(f"{key}_method", "clustering")),
        names=tuple(names),
        source=str(source_labels),
        features=tuple(str(name) for name in adata.var_names),
        n_objects=int(adata.n_obs),
        colors=tuple(colour_map(names)[name] for name in names),
        extra=dict(extra),
    )


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
        # Loopback only. Streamlit's default binds every interface, which puts the
        # object tables -- and on a university network, potentially the internet --
        # in front of anyone who can reach this machine's address. Nothing here
        # has any authentication in front of it.
        "--server.address",
        "127.0.0.1",
        # Without this the first run stops at a terminal prompt asking for an
        # email address, and the window never opens.
        "--browser.gatherUsageStats",
        "false",
    ]
    if path is not None:
        argv += ["--", "--file", str(path)]
    return argv


def module_command(
    path: str | Path | None = None,
    port: int = 8504,
    python: str | None = None,
    open_browser: bool = True,
) -> list[str]:
    """The command the shortcut runs, and the one the viewer's button runs too.

    ``-m microscopy_viewer.dashboard`` rather than ``-m streamlit run …``: the
    banner, the log formatting and the warm-up all live in :func:`main`, and a
    dashboard started from the viewer should be the same program as one started
    from the shortcut rather than a second arrangement that drifts from it.
    """
    argv = [str(python or sys.executable), "-m", "microscopy_viewer.dashboard"]
    if path is not None:
        argv.append(str(path))
    argv += ["--port", str(int(port))]
    if not open_browser:
        argv.append("--no-browser")
    return argv


def free_port(start: int = 8504, tries: int = 20) -> int:
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
    show_console: bool = True,
) -> tuple[subprocess.Popen, str]:
    """Start the dashboard as its own process. Returns ``(process, url)``.

    A separate process on purpose. Streamlit runs its own event loop and would
    fight Qt's inside the viewer; and the spatial work is minutes of CPU that has
    no business blocking the window the images are in.

    *show_console* opens a terminal of its own and lets the output go to it. On by
    default, and worth keeping on: the first neighbour search on a plate takes the
    better part of a minute, and a page that is merely sitting there is
    indistinguishable from a page that has hung unless something is saying what it
    is doing. The window closing is also how you know the server stopped.
    """
    missing = missing_packages()
    if missing:
        raise RuntimeError(install_hint(missing))

    chosen = int(port) if port else free_port()
    argv = module_command(path, port=chosen, python=python, open_browser=open_browser)
    url = f"http://localhost:{chosen}"

    creation = 0
    streams: dict[str, Any] = {}
    if os.name == "nt":
        if show_console:
            # Its own window, so the log is readable and outlives the viewer.
            creation = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
        else:
            creation = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    if not show_console:
        # Captured rather than inherited, so a hidden dashboard cannot write over
        # whatever the parent is printing.
        streams = {"stdout": subprocess.PIPE, "stderr": subprocess.STDOUT}

    logger.info("dashboard: %s", " ".join(argv))
    process = subprocess.Popen(
        argv,
        creationflags=creation,
        cwd=str(Path(__file__).resolve().parent.parent),
        env=_child_environment(),
        **streams,
    )
    # The browser is opened by the child, which knows when it is actually
    # listening; opening it from here would race the server's own start-up.
    return process, url


def _child_environment() -> dict[str, str]:
    """The environment the dashboard process runs in.

    Python buffers stdout when it is a pipe rather than a terminal, and a console
    that shows nothing for forty seconds and then everything at once is worse than
    no console at all.
    """
    environment = dict(os.environ)
    environment["PYTHONUNBUFFERED"] = "1"
    environment.setdefault("PYTHONIOENCODING", "utf-8")
    return environment


def streamlit_available() -> bool:
    return shutil.which("streamlit") is not None or not missing_packages()


# ---------------------------------------------------------------------------
# Starting it without the viewer
# ---------------------------------------------------------------------------


def configure_logging(level: int = logging.INFO) -> None:
    """Send this package's log to stdout, timestamped.

    The dashboard's own console is the only place the timings appear, and they are
    the point of having one: which step is slow, and whether anything is happening
    at all.
    """
    root = logging.getLogger("microscopy_viewer")
    root.setLevel(level)
    if any(isinstance(handler, logging.StreamHandler) for handler in root.handlers):
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s  %(name)-28s %(message)s", "%H:%M:%S"))
    root.addHandler(handler)


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m microscopy_viewer.dashboard [file.h5ad]``.

    Runs Streamlit in *this* process rather than starting another one: started
    from a shortcut there is already a console, and a second process inside it
    would put the log one layer further from the window that is showing it.
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="microscopy-viewer-dashboard",
        description="The spatial dashboard: squidpy statistics on an object table.",
    )
    parser.add_argument("file", nargs="?", default=None, help="an .h5ad to open")
    parser.add_argument("--port", type=int, default=None, help="port to serve on")
    parser.add_argument("--no-browser", action="store_true", help="do not open a browser")
    arguments = parser.parse_args(list(argv) if argv is not None else None)

    configure_logging()
    missing = missing_packages()
    if missing:
        print(install_hint(missing))
        return 2

    port = int(arguments.port) if arguments.port else free_port()
    url = f"http://localhost:{port}"
    print(f"Microscopy Viewer — spatial dashboard\n{url}\n")
    if arguments.file:
        print(f"opening {arguments.file}")
    print("Close this window to stop the server.\n", flush=True)

    # Started here as well as by the page: the page only runs when a browser
    # connects, and the seconds before that are seconds the compiler could be
    # using. Whichever gets there first does it; the other returns at once.
    threading.Thread(target=warm_up, name="warm-up", daemon=True).start()

    if not arguments.no_browser:
        import webbrowser

        # Streamlit takes a couple of seconds to bind the port; opening the
        # browser straight away gives a connection-refused page. The warm-up is
        # started by the page itself, which every way of running this goes through.
        threading.Timer(3.0, lambda: webbrowser.open(url)).start()

    from streamlit.web import cli as stcli

    sys.argv = command(arguments.file, port=port)[2:]  # drop "python -m"
    try:
        return int(stcli.main(standalone_mode=False) or 0)
    except SystemExit as exit_code:  # click raises this on a clean stop
        return int(exit_code.code or 0)
    except KeyboardInterrupt:
        print("\nstopped")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
