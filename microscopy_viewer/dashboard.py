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

**There is one clustering.** :func:`run_clustering` is the only place a cluster is
decided; it samples the plate, clusters and embeds that sample, gives every other
object a cluster from its neighbours, and leaves all of it on the file. Every plot
in the application reads that back through :func:`clustered_frame`, so a colour
means the same thing on the UMAP, in the well display, in one well's spatial
statistics and in the label set written into the plate. It is recomputed only when
the parameters change.

**Most objects were never clustered.** A plate is sampled a few hundred objects a
well, which on this data is two per cent of them; the other ninety-eight have no
phenotype at all. :func:`assign_all` gives them one from their neighbours in the
same feature space, and marks it as the prediction it is, so that a label layer
written into the plate is not ninety-eight per cent empty.

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
from dataclasses import dataclass
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

    if by in adata.obs:
        grouping = adata.obs[by].astype(str)
    elif by == "well" and IMAGE_KEY in adata.obs:
        # Derived rather than required: a file clustered by run_clustering has the
        # image on it and the well is a reading of that, not a second fact.
        grouping = pd.Series(
            [well_of(str(name)) for name in adata.obs[IMAGE_KEY]], index=adata.obs.index
        )
    else:
        grouping = adata.obs[IMAGE_KEY].astype(str)
    counts = pd.crosstab(grouping, adata.obs[key])
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


#: Objects kept per well for the plate display. Forty-four panels of this many
#: points is a page that draws and hovers instantly; the whole plate would be half
#: a million points held in the browser.
DEFAULT_VIEW_POINTS = 500


#: Where the one clustering lives inside the file.
CLUSTERING_KEY = "clustering"

#: Column every object's cluster is written to, and the one saying how it got it.
ORIGIN_COLUMN = "cluster_origin"


@dataclass(frozen=True)
class ClusteringParams:
    """Everything that decides what the clustering comes out as.

    Compared as a whole to decide whether the stored one is still the one asked
    for: change any of these and the clusters are different clusters, and a page
    showing the old ones beside the new parameters would be lying.
    """

    sample_per_well: int = 750
    resolution: float = 0.6
    n_neighbors: int = DEFAULT_UMAP_NEIGHBOURS
    min_dist: float = DEFAULT_MIN_DIST
    n_comps: int = DEFAULT_COMPONENTS
    assign_all: bool = True

    def as_attrs(self) -> dict[str, Any]:
        return {
            "sample_per_well": int(self.sample_per_well),
            "resolution": float(self.resolution),
            "n_neighbors": int(self.n_neighbors),
            "min_dist": float(self.min_dist),
            "n_comps": int(self.n_comps),
            "assign_all": bool(self.assign_all),
        }

    @classmethod
    def from_attrs(cls, block: Mapping) -> "ClusteringParams":
        return cls(
            sample_per_well=int(block.get("sample_per_well", 750)),
            resolution=float(block.get("resolution", 0.6)),
            n_neighbors=int(block.get("n_neighbors", DEFAULT_UMAP_NEIGHBOURS)),
            min_dist=float(block.get("min_dist", DEFAULT_MIN_DIST)),
            n_comps=int(block.get("n_comps", DEFAULT_COMPONENTS)),
            assign_all=bool(block.get("assign_all", True)),
        )

    def describe(self) -> str:
        return (
            f"{self.sample_per_well} per well, resolution {self.resolution:g}, "
            f"{self.n_neighbors} neighbours, min_dist {self.min_dist:g}"
        )


def run_clustering(source, params: ClusteringParams | None = None, progress=None) -> dict:
    """Cluster the plate once and write the result into *source*. Returns the record.

    This is the only clustering in the application. Everything else — the UMAP,
    the well display, the spatial statistics of one well, the label sets written
    back into the plate — reads what this leaves behind, so that a cluster has one
    meaning across the whole page rather than one meaning per tab.

    What it leaves behind is a cluster for *every* object in ``obs``, an origin
    saying whether that cluster was computed or predicted, the embedding of the
    sampled objects, and the palette. The sample is what gets clustered and
    embedded — a UMAP of half a million objects is minutes and a page that cannot
    be drawn — and :func:`assign_all` carries the answer to the rest.
    """
    params = params or ClusteringParams()

    def say(text: str) -> None:
        logger.info("clustering: %s", text)
        if progress is not None:
            progress(text)

    say(f"sampling {params.sample_per_well} objects per image")
    reference = stratified_subsample(source, params.sample_per_well)
    say(f"scaling and reducing {reference.n_obs:,} objects")
    prepare(reference, n_comps=params.n_comps)
    say("finding the clusters")
    method = cluster(reference, resolution=params.resolution)
    say("laying them out")
    embed(reference, n_neighbors=params.n_neighbors, min_dist=params.min_dist)

    names = [str(name) for name in _categories(reference.obs[CLUSTER_KEY])]
    palette = colour_map(names)

    if params.assign_all:
        say("giving every object a cluster")
        frame = assign_all(reference, source, progress=progress)
        clusters = frame["cluster"].astype(str).to_numpy()
        origins = frame["origin"].to_numpy()
    else:
        known = {
            str(name): str(value)
            for name, value in zip(reference.obs_names, reference.obs[CLUSTER_KEY])
        }
        clusters = np.array(
            [known.get(str(name), "") for name in source.obs_names], dtype=object
        )
        origins = np.where(clusters == "", "", ORIGIN_CLUSTERED)

    import pandas as pd

    source.obs[CLUSTER_KEY] = pd.Categorical(
        [str(value) for value in clusters], categories=names
    )
    source.obs[ORIGIN_COLUMN] = [str(value) for value in origins]

    umap = np.asarray(reference.obsm["X_umap"], dtype=float)
    record = {
        **params.as_attrs(),
        "method": str(method),
        "clusters": names,
        "colors": [palette[name] for name in names],
        "umap_names": [str(name) for name in reference.obs_names],
        "umap_x": umap[:, 0],
        "umap_y": umap[:, 1],
        "n_clustered": int(reference.n_obs),
        "n_assigned": int(sum(1 for value in origins if value == ORIGIN_ASSIGNED)),
        "n_objects": int(source.n_obs),
        "features": [str(name) for name in reference.var_names],
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    source.uns[CLUSTERING_KEY] = record
    say(
        f"done: {method} into {len(names)} clusters, "
        f"{record['n_clustered']:,} clustered, {record['n_assigned']:,} assigned"
    )
    return record


def load_clustering(source) -> dict:
    """The stored clustering record, or ``{}`` when the file has none."""
    block = source.uns.get(CLUSTERING_KEY)
    if not isinstance(block, (dict, Mapping)) or "clusters" not in block:
        return {}
    if CLUSTER_KEY not in source.obs:
        return {}
    record = {
        "method": str(block.get("method", "")),
        "clusters": [str(name) for name in block.get("clusters", [])],
        "colors": [str(colour) for colour in block.get("colors", [])],
        "created": str(block.get("created", "")),
        "n_clustered": int(block.get("n_clustered", 0) or 0),
        "n_assigned": int(block.get("n_assigned", 0) or 0),
        "n_objects": int(block.get("n_objects", 0) or 0),
        "features": [str(name) for name in block.get("features", [])],
        "params": ClusteringParams.from_attrs(block),
    }
    names = [str(name) for name in block.get("umap_names", [])]
    if names:
        record["umap"] = {
            "names": names,
            "x": np.asarray(block.get("umap_x", []), dtype=float),
            "y": np.asarray(block.get("umap_y", []), dtype=float),
        }
    return record


def clustering_palette(record: Mapping) -> dict[str, str]:
    """``{cluster: hex}`` from a stored record, falling back to the shared palette."""
    names = [str(name) for name in record.get("clusters", [])]
    colours = [str(colour) for colour in record.get("colors", [])]
    if len(colours) == len(names) and names:
        return dict(zip(names, colours))
    return colour_map(names)


def clustered_frame(source, image: str | None = None, limit: int | None = None, seed: int = 0):
    """The objects and their clusters, ready to plot. Everything derives from this.

    *image* narrows to one well and cycle; *limit* subsamples what comes back, for
    a plot that has to stay quick. The clusters are the ones :func:`run_clustering`
    left on the file, so a colour means the same thing here as in every other
    plot on the page.
    """
    import pandas as pd

    if CLUSTER_KEY not in source.obs:
        raise ValueError("this file has not been clustered yet")

    mask = np.ones(source.n_obs, dtype=bool)
    if image:
        if IMAGE_KEY not in source.obs:
            raise ValueError("this file holds one image; there is nothing to narrow to")
        mask = (source.obs[IMAGE_KEY].astype(str) == str(image)).to_numpy()
        if not mask.any():
            raise ValueError(f"no objects belong to {image!r}")

    where = np.flatnonzero(mask)
    if limit is not None and where.size > int(limit):
        where = np.sort(
            np.random.default_rng(seed).choice(where, size=int(limit), replace=False)
        )

    spatial = source.obsm.get("spatial")
    frame = pd.DataFrame(
        {
            "object": [str(name) for name in source.obs_names[where]],
            "cluster": source.obs[CLUSTER_KEY].astype(str).to_numpy()[where],
        }
    )
    if spatial is not None:
        coordinates = np.asarray(spatial, dtype=float)[where]
        frame["x"] = coordinates[:, 0]
        frame["y"] = coordinates[:, 1]
    for column in ("label", IMAGE_KEY, ORIGIN_COLUMN):
        if column in source.obs:
            frame[column] = source.obs[column].astype(str).to_numpy()[where] \
                if column != "label" else source.obs[column].to_numpy()[where]
    if IMAGE_KEY in frame:
        frame["well"] = [well_of(name) for name in frame[IMAGE_KEY]]
    frame["_row"] = where
    return frame


def save_clustering(path: str | Path, source) -> Path:
    """Write the file back with its clustering in it."""
    return save_h5ad(path, source)


def _categories(column) -> list:
    categories = getattr(getattr(column, "cat", None), "categories", None)
    if categories is not None:
        return list(categories)
    return sorted({str(value) for value in column})


def save_h5ad(path: str | Path, adata) -> Path:
    """Write the file back, clustering and all. Returns where it went.

    The whole ``.h5ad`` is rewritten, which for a plate is a hundred megabytes and
    a few seconds; anndata has no way to add one ``uns`` entry in place, and a
    sidecar file would be one more thing to keep beside the data and lose.
    """
    target = Path(str(path))
    started = time.perf_counter()
    adata.write_h5ad(target)
    logger.info(
        "saved %s (%.0f MB, %.1f s)",
        target.name,
        target.stat().st_size / 1024 / 1024,
        time.perf_counter() - started,
    )
    return target


#: Neighbours voting when an unsampled object is given a cluster.
DEFAULT_ASSIGN_NEIGHBOURS = 15

#: How an object came by its cluster, written into the assignment table and into
#: the plate's metadata. The difference matters: one was computed, the other is a
#: prediction from the objects around it in feature space.
ORIGIN_CLUSTERED = "clustered"
ORIGIN_ASSIGNED = "assigned"


def project(reference, source) -> np.ndarray:
    """*source*'s objects in *reference*'s own principal components.

    :func:`prepare` z-scores and then rotates; both steps have to be repeated
    exactly, not refitted, or the projected objects land in a different space from
    the ones the clusters were found in. scanpy keeps the column means and
    deviations in ``var`` and the rotation in ``varm['PCs']``, which is everything
    needed.

    *source* must hold the **unscaled** measurements — the file as it was read.
    Handing it one that has already been through :func:`prepare` scales it twice
    and puts every object in the wrong place, silently, so that is refused.
    """
    if "std" in getattr(source, "var", {}) and "PCs" in source.varm:
        raise ValueError(
            "this source has already been scaled; project() needs the raw measurements, "
            "which is the file as it was read"
        )
    if "PCs" not in reference.varm:
        raise ValueError("the reference has no PCA to project into; run prepare() first")

    columns = [str(name) for name in reference.var_names]
    missing = [name for name in columns if name not in set(source.var_names)]
    if missing:
        raise ValueError(f"the file is missing {len(missing)} of the clustered features")

    matrix = source[:, columns].X
    matrix = np.asarray(matrix.todense() if hasattr(matrix, "todense") else matrix, dtype=np.float64)

    mean = np.asarray(reference.var.get("mean", np.zeros(len(columns))), dtype=float)
    deviation = np.asarray(reference.var.get("std", np.ones(len(columns))), dtype=float)
    deviation = np.where(np.isfinite(deviation) & (deviation > 0), deviation, 1.0)

    scaled = (np.nan_to_num(matrix, nan=0.0) - mean) / deviation
    # scanpy's scale() clips at ten deviations; an object outside the sample can
    # be further out than anything inside it, and without the same clip one
    # outlier would dominate the distance to every neighbour.
    np.clip(scaled, -10.0, 10.0, out=scaled)
    return scaled @ np.asarray(reference.varm["PCs"], dtype=float)


def assign_all(
    reference,
    source,
    key: str = CLUSTER_KEY,
    n_neighbors: int = DEFAULT_ASSIGN_NEIGHBOURS,
    progress=None,
):
    """A cluster for every object in *source*, and how each one got it.

    The clustering runs on a sample — a few hundred objects a well out of tens of
    thousands — so most objects have no cluster at all. Painting only those into
    the plate leaves a label layer that is mostly empty and badly misrepresents
    how common a phenotype is. This gives the rest one by asking their nearest
    neighbours in the same feature space.

    That is a **prediction**, not a measurement, and the ``origin`` column says
    which is which for every object so a count can be taken over one or the other.
    """
    import pandas as pd

    if key not in reference.obs:
        raise ValueError(f"nothing has been clustered into obs[{key!r}] yet")

    names = [str(name) for name in _categories(reference.obs[key])]
    known = {str(name): str(value) for name, value in zip(reference.obs_names, reference.obs[key])}

    index = source.obs_names.astype(str)
    clusters = np.array([known.get(name, "") for name in index], dtype=object)
    unknown = clusters == ""
    if progress is not None:
        progress(f"{int(unknown.sum()):,} of {source.n_obs:,} objects to assign")

    if unknown.any():
        from sklearn.neighbors import KNeighborsClassifier

        classifier = KNeighborsClassifier(
            n_neighbors=int(min(n_neighbors, max(1, reference.n_obs))), n_jobs=-1
        )
        classifier.fit(
            np.asarray(reference.obsm["X_pca"], dtype=float),
            reference.obs[key].astype(str).to_numpy(),
        )
        started = time.perf_counter()
        predicted = classifier.predict(project(reference, source[unknown]))
        clusters[unknown] = predicted
        logger.info(
            "assigned %d object(s) by %d-neighbour vote in %.1f s",
            int(unknown.sum()),
            classifier.n_neighbors,
            time.perf_counter() - started,
        )

    frame = pd.DataFrame(
        {
            "label": (
                source.obs["label"].to_numpy()
                if "label" in source.obs
                else np.arange(source.n_obs)
            ),
            "cluster": pd.Categorical([str(value) for value in clusters], categories=names),
            "image": (
                source.obs[IMAGE_KEY].astype(str).to_numpy()
                if IMAGE_KEY in source.obs
                else ""
            ),
            "origin": np.where(unknown, ORIGIN_ASSIGNED, ORIGIN_CLUSTERED),
        }
    )
    logger.info(
        "every object has a cluster: %d computed, %d assigned",
        int((~unknown).sum()),
        int(unknown.sum()),
    )
    return frame


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


def cluster_run(
    adata,
    key: str = CLUSTER_KEY,
    source_labels: str = "",
    frame=None,
    **extra,
):
    """The provenance record for a clustering about to be written into a plate.

    *frame* is the assignment table when there is one, so the record can say how
    many objects had their cluster computed and how many were predicted.
    """
    from .clusters import ClusterRun, cluster_names

    names = cluster_names(adata.obs[key])
    n_objects = int(adata.n_obs)
    n_clustered, n_assigned = n_objects, 0
    if frame is not None and "origin" in getattr(frame, "columns", ()):
        counts = frame["origin"].value_counts()
        n_objects = int(len(frame))
        n_clustered = int(counts.get(ORIGIN_CLUSTERED, 0))
        n_assigned = int(counts.get(ORIGIN_ASSIGNED, 0))
    return ClusterRun(
        method=str(adata.uns.get(f"{key}_method", "clustering")),
        names=tuple(names),
        source=str(source_labels),
        features=tuple(str(name) for name in adata.var_names),
        n_objects=n_objects,
        n_clustered=n_clustered,
        n_assigned=n_assigned,
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
