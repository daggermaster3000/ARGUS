"""Explore an analysis workbook: UMAP of the regions, and free scatter plots.

Reads the *Region features* sheet the Analysis panel writes (one row per sample
and brain region, every column a number) and offers:

* **UMAP** of the regions over the chosen feature groups — outline shape,
  objects, and the channel intensities — with hover showing sample, genotype,
  region and the headline numbers.
* **Scatter** of any two or three columns against each other, 2D or 3D,
  coloured by sample, genotype or region.
* **Cells**: UMAP or PCA of the segmented cells over their size, intensity and
  shape, plus a free scatter of cell measurements, coloured by sample,
  genotype, region or cluster, over the cells of the chosen regions only (the
  cerebellum alone, say). Cells can be clustered (k-means, Gaussian
  mixture or HDBSCAN) over the same feature groups; the clusters carry over to
  the atlas.
* **Atlas**: one region's outline (the cerebellum, say) registered across the
  samples into a mean shape, with a heatmap of where the cells sit in it,
  averaged per genotype and compared between them. The cells can be drawn on
  it as dots or as their own outlines, coloured by cluster or genotype.
* **Explain**: SHAP values of a random forest that tells the clusters, or the
  genotypes, apart from the cells' measurements — which features matter, and
  which way. Genotype is scored with whole samples held out, so a model cannot
  pass by recognising the fish.
* **Plots**: one variable of the regions, the cells or the cell clusters as a
  box or violin plot per genotype, with every sample as a dot, a one-way
  ANOVA (or Welch / Kruskal-Wallis) and Tukey post-hoc pairs. Any number of
  genotype groups.
* **Table** of exactly the rows being plotted.

In 2D, each region can be drawn as its own outline instead of a dot, taken from
the *Region outlines* sheet (workbooks written before that sheet existed only
offer dots). Cells can likewise be drawn as their own outlines, from the
``cell_outlines.npz`` the analysis writes beside the workbook.

Run with::

    streamlit run apps/region_explorer.py
    streamlit run apps/region_explorer.py -- path/to/report.xlsx

Needs ``streamlit``, ``plotly``, ``scikit-learn`` and ``umap-learn``; the
Explain tab also ``shap``.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from microscopy_viewer import analysis_plots as ap  # noqa: E402

from explorer_common import (  # noqa: E402  shared with simple_explorer.py
    CELL_CHANNEL_STATS,
    CELL_HOVER,
    CELL_IDENTITY,
    CELL_INTENSITIES,
    CELL_POSITION,
    CELL_SHAPE,
    CELL_TESTS,
    HOVER,
    IDENTITY,
    INK,
    MIXED,
    NEUTRAL,
    REGION_COLORS,
    SHEET,
    SYMBOLS,
    TESTS,
    anova_table,
    cell_groups,
    compare_groups,
    comparison_figure,
    genotype_of_sample,
    hover_columns,
    mixed_model_test,
    mixed_pairs,
    mixed_table,
    default_variable,
    numeric_columns,
    plottable,
    posthoc_pairs,
    read_cells,
    WB_AREA,
    group_choices,
    group_columns,
    group_order,
    normalise,
    normalise_choices,
    read_features,
    with_group,
    stars,
    _alpha,
    _mixed_fit,
)


#: Columns that say where an outline is or which way it points, rather than
#: what it is like. Off by default: two fish mounted differently would otherwise
#: separate on mounting.
POSITION = ("Centroid Y (µm)", "Centroid X (µm)", "Orientation (°)")
SHAPE = (
    "Region area (µm²)", "Region volume (µm³)", "Perimeter (µm)", "Circularity",
    "Solidity", "Major axis (µm)", "Minor axis (µm)", "Aspect ratio",
    "Eccentricity", "Bounding height (µm)", "Bounding width (µm)",
    "WB area (µm²)",
)
INTENSITY_STATS = ("Mean", "Median", "SD", "CV", "P5", "P95", "P99", "Integrated")



OUTLINES = "Region outlines"

#: Most cell outlines drawn at once. Each is its own filled path in the browser;
#: past a few thousand the page stops being responsive.
MAX_CELL_GLYPHS = 3000

#: Plot area of an outline figure, in pixels, and the margins around it. Fixed,
#: because an outline's aspect ratio depends on how many pixels a data unit is.
PLOT_W, PLOT_H = 760, 540
MARGIN = {"l": 70, "r": 200, "t": 50, "b": 60}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------




@st.cache_data(show_spinner=False)
def read_outlines(source: bytes | str) -> dict[tuple[str, str], list[np.ndarray]]:
    """``(sample, region) -> [outline, ...]``; empty for older workbooks."""
    handle = io.BytesIO(source) if isinstance(source, bytes) else source
    try:
        frame = pd.read_excel(handle, sheet_name=OUTLINES)
    except ValueError:
        return {}
    outlines: dict[tuple[str, str], list[np.ndarray]] = {}
    frame = frame.sort_values(["Sample", "Region", "Part", "Vertex"])
    for (sample, region, _part), group in frame.groupby(["Sample", "Region", "Part"], sort=False):
        points = group[["X (µm)", "Y (µm)"]].to_numpy(dtype=float)
        # Image rows grow downwards; plot y grows upwards.
        points[:, 1] *= -1
        outlines.setdefault((str(sample), str(region)), []).append(points)
    return outlines




@st.cache_data(show_spinner="Reading cell outlines…")
def read_cell_outlines(source: bytes | str) -> dict[tuple[str, int], list[np.ndarray]]:
    from microscopy_viewer.analysis import load_cell_outlines

    stored = load_cell_outlines(io.BytesIO(source) if isinstance(source, bytes) else source)
    flipped = {}
    for key, points in stored.items():
        xy = np.asarray(points, dtype=float)[:, ::-1].copy()
        xy[:, 1] *= -1  # image rows grow downwards
        flipped[key] = [xy]
    return flipped







@st.cache_data(show_spinner="Embedding cells…")
def embed(matrix: np.ndarray, method: str, dims: int, neighbors: int, min_dist: float, seed: int):
    if method == "PCA":
        from sklearn.decomposition import PCA

        model = PCA(n_components=dims, random_state=seed)
        points = model.fit_transform(matrix)
        return points, [f"PC{i + 1} ({r * 100:.1f}%)" for i, r in enumerate(model.explained_variance_ratio_)]
    return run_umap(matrix, neighbors, min_dist, dims, seed), [f"UMAP {i + 1}" for i in range(dims)]




def channel_columns(frame: pd.DataFrame) -> dict[str, list[str]]:
    """``channel -> its intensity columns``, read off the column names."""
    channels: dict[str, list[str]] = {}
    for column in numeric_columns(frame):
        for stat in INTENSITY_STATS:
            suffix = f" {stat}"
            if column.endswith(suffix) and column not in SHAPE:
                channel = column[: -len(suffix)]
                if channel and not channel.startswith(("Mean object", "Median object", "SD object")):
                    channels.setdefault(channel, []).append(column)
                break
    return channels


def feature_groups(frame: pd.DataFrame) -> dict[str, list[str]]:
    numeric = numeric_columns(frame)
    channels = channel_columns(frame)
    in_channels = {c for columns in channels.values() for c in columns}
    shape = [c for c in numeric if c in SHAPE]
    position = [c for c in numeric if c in POSITION]
    objects = [c for c in numeric if c not in in_channels and c not in shape and c not in position]
    groups = {"Region shape": shape, "Objects": objects, "Position": position}
    for channel, columns in channels.items():
        groups[f"Intensity · {channel}"] = columns
    return groups


# -- one variable, compared between genotypes ---------------------------------





















# -- clustering ---------------------------------------------------------------

CLUSTER_METHODS = ("Off", "K-means", "Gaussian mixture", "HDBSCAN")
#: Cells HDBSCAN calls noise, and clusters past the eighth: they share grey
#: rather than get a ninth, generated hue.
UNCLUSTERED = "Noise / other"
#: Cells outside the region that was clustered: lighter, so they recede.
NOT_CLUSTERED = "Not clustered"
FAINT = "#c9c8c3"


@st.cache_data(show_spinner="Clustering cells…")
def cluster_cells(matrix: np.ndarray, method: str, k: int, min_size: int, seed: int) -> np.ndarray:
    """Integer cluster per row; -1 is noise (HDBSCAN only)."""
    if method == "K-means":
        from sklearn.cluster import KMeans

        return KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(matrix)
    if method == "Gaussian mixture":
        from sklearn.mixture import GaussianMixture

        return GaussianMixture(n_components=k, random_state=seed).fit_predict(matrix)
    from sklearn.cluster import HDBSCAN

    return HDBSCAN(min_cluster_size=min_size).fit_predict(matrix)


#: What each feature group brings to a clustering, for its tooltip.
GROUP_HELP = {
    "Size & intensity": "Volume, area, diameter and the segmented channel's intensity. "
                        "Clusters on these mostly split big/bright from small/dim cells.",
    "Channel intensities": "Every channel's mean, SD, max and integrated intensity per "
                           "cell — the one to tick to group cells by marker expression.",
    "Cell shape": "Outline shape seen from above: circularity, solidity, elongation. "
                  "Separates round nuclei from elongated or irregular ones.",
    "Position": "Where the cell is in the image. Off by default: fish mounted "
                "differently would cluster by mounting.",
}


@st.cache_data(show_spinner="Scoring cluster counts…")
def score_cluster_counts(matrix: np.ndarray, method: str, seed: int) -> pd.DataFrame:
    """Silhouette (and BIC for a mixture) for 2 to 8 clusters.

    On at most 3000 cells: the silhouette compares every pair of cells.
    """
    from sklearn.metrics import silhouette_score

    rows = np.random.default_rng(seed).permutation(len(matrix))[:3000]
    sample = matrix[rows]
    out = []
    for k in range(2, len(REGION_COLORS) + 1):
        if k >= len(sample):
            break
        labels = cluster_cells(sample, method, k, 0, seed)
        entry = {"Clusters": k, "Silhouette": float(silhouette_score(sample, labels))}
        if method == "Gaussian mixture":
            from sklearn.mixture import GaussianMixture

            model = GaussianMixture(n_components=k, random_state=seed).fit(sample)
            entry["BIC"] = float(model.bic(sample))
        out.append(entry)
    return pd.DataFrame(out)


def cluster_names(labels) -> list[str]:
    """``C1`` for the largest cluster, ``C2`` the next, …; the rest UNCLUSTERED.

    Named by size so the colours do not change with the arbitrary numbers the
    clustering hands out.
    """
    labels = np.asarray(labels)
    ids, counts = np.unique(labels[labels >= 0], return_counts=True)
    ranked = ids[np.argsort(-counts, kind="stable")][: len(REGION_COLORS)]
    name = {int(c): f"C{i + 1}" for i, c in enumerate(ranked)}
    return [name.get(int(v), UNCLUSTERED) for v in labels]


def cluster_order(values) -> list[str]:
    found = set(values)
    rest = (UNCLUSTERED, NOT_CLUSTERED)
    named = sorted((v for v in found if v not in rest), key=lambda v: int(v[1:]))
    return named + [v for v in rest if v in found]


def cluster_colors(order) -> dict[str, str]:
    special = {UNCLUSTERED: NEUTRAL, NOT_CLUSTERED: FAINT}
    return {c: special.get(c) or REGION_COLORS[int(c[1:]) - 1] for c in order}


def cluster_distribution_figure(clustered: pd.DataFrame, region: str, measure: str):
    """Per genotype, how a region's cells split between the clusters.

    One bar per genotype and cluster at the mean over that genotype's samples,
    with each sample as a dot on it: with a handful of fish per genotype, the
    spread between them is the thing to read the difference against.
    """
    order = cluster_order(clustered["Cluster"])
    genotypes_here = ap.genotype_order(clustered["Genotype"])
    styles = ap.genotype_styles(genotypes_here)
    counts = pd.crosstab([clustered["Genotype"], clustered["Sample"]], clustered["Cluster"])
    counts = counts.reindex(columns=order, fill_value=0)
    share = measure.startswith("Share")
    values = counts.div(counts.sum(axis=1), axis=0) * 100 if share else counts
    unit = "% of the sample's cells" if share else "cells"
    figure = go.Figure()
    width = 0.8 / max(len(genotypes_here), 1)
    for k, genotype in enumerate(genotypes_here):
        if genotype not in values.index.get_level_values(0):
            continue
        per_sample = values.xs(genotype, level=0)
        color = styles[genotype][0]
        offset = (k - (len(genotypes_here) - 1) / 2) * width
        figure.add_trace(go.Bar(
            x=np.arange(len(order)) + offset, y=per_sample.mean(axis=0).to_numpy(),
            name=genotype, width=width * 0.9, legendgroup=genotype, customdata=order,
            marker={"color": _alpha(color, 0.55), "line": {"color": color, "width": 1.5}},
            hovertemplate=f"{genotype} · %{{customdata}}: %{{y:.3g}} {unit} "
                          f"(mean of {len(per_sample)})<extra></extra>",
        ))
        jitter = np.random.default_rng(k).uniform(-width * 0.25, width * 0.25, len(per_sample))
        for c_index, cluster in enumerate(order):
            figure.add_trace(go.Scatter(
                x=c_index + offset + jitter, y=per_sample[cluster].to_numpy(), mode="markers",
                legendgroup=genotype, showlegend=False, text=list(per_sample.index),
                marker={"size": 8, "color": color, "line": {"color": "white", "width": 1.5}},
                hovertemplate=f"%{{text}} · {cluster}: %{{y:.3g}} {unit}<extra>{genotype}</extra>",
            ))
    figure.update_layout(
        barmode="overlay",
        title=f"How each genotype's {region} cells split between the clusters",
        xaxis={"title": "Cluster", "tickvals": list(range(len(order))), "ticktext": order,
               "showgrid": False, "zeroline": False, "automargin": True},
        yaxis={"title": unit, "automargin": True, "rangemode": "tozero"},
        legend_title_text="Genotype", height=440, margin={"l": 10, "r": 10, "t": 50, "b": 10},
    )
    return figure


@st.cache_data(show_spinner=False)
def read_samples_sheet(source: bytes | str) -> pd.DataFrame:
    try:
        frame = pd.read_excel(io.BytesIO(source) if isinstance(source, bytes) else source,
                              sheet_name="Samples")
    except ValueError:
        return pd.DataFrame(columns=["Sample", "File", "Saved as"])
    frame["Sample"] = frame["Sample"].astype(str)
    return frame


def locate_sample(sample: str, recorded, report: Path | None) -> Path | None:
    """The sample's ``.ims``: where the workbook says, else beside the report folder.

    A report folder sits inside the experiment folder, so a drive mounted under
    a new name still finds its files one level up.
    """
    candidates = [Path(recorded)] if isinstance(recorded, str) and recorded else []
    if report is not None:
        candidates += [report.parent / f"{sample}.ims", report.parent.parent / f"{sample}.ims"]
    return next((c for c in candidates if c.is_file()), None)


def write_clusters_panel(source, cells: pd.DataFrame, region: str, method: str, k: int,
                         min_size: int, features: list[str]) -> None:
    """The button that writes the clusters back into each sample's ``.ims``."""
    from microscopy_viewer import analysis as an

    clustered = cells[cells["Cluster"] != NOT_CLUSTERED]
    order = cluster_order(clustered["Cluster"])
    number = {c: i + 1 for i, c in enumerate(order)}
    colors = cluster_colors(order)
    sheet = read_samples_sheet(source)
    report = Path(source).parent if isinstance(source, str) else None
    key = f"{region} clusters"
    with st.expander("Write the clusters into the .ims files"):
        st.caption(
            f"Adds a label map “{key}” to each sample: every clustered {region} cell "
            f"keeps its outline and takes its cluster's number (C1 = 1, …"
            + (f", {UNCLUSTERED} = {number[UNCLUSTERED]}" if UNCLUSTERED in number else "")
            + "); every other cell is background. The viewer draws it in these same "
            "colours. Running again replaces it; the original label map is not touched."
        )
        targets = []
        for sample in list(dict.fromkeys(clustered["Sample"])):
            row = sheet[sheet["Sample"] == sample]
            recorded = row["File"].iloc[0] if len(row) and "File" in row else None
            source_key = str(row["Saved as"].iloc[0]) if len(row) and "Saved as" in row else ""
            targets.append((sample, locate_sample(sample, recorded, report), source_key))
        lost = [s for s, path, src in targets if path is None or not src or src == "nan"]
        if lost:
            st.caption("Not found, or no label map recorded, for: " + ", ".join(lost))
        ready = [(s, p, src) for s, p, src in targets if p is not None and src and src != "nan"]
        if st.button(f"Write into {len(ready)} file(s)", key="clu-write", disabled=not ready):
            settings = {"clustered_region": region, "cluster_method": method,
                        "cluster_features": ", ".join(features)}
            settings["clusters" if method != "HDBSCAN" else "smallest_cluster"] = (
                int(k) if method != "HDBSCAN" else int(min_size))
            results = []
            for sample, path, source_key in ready:
                mine = clustered[clustered["Sample"] == sample]
                mapping = {int(lab): number[c] for lab, c in zip(mine["Label"], mine["Cluster"])}
                try:
                    written, placed = an.write_cluster_labels(
                        path, source_key, mapping, key,
                        colors={number[c]: colors[c] for c in order},
                        names={number[c]: c for c in order}, attrs=settings,
                    )
                    results.append({"Sample": sample, "Written": written, "Cells": placed,
                                    "File": str(path), "Problem": ""})
                except Exception as exc:
                    results.append({"Sample": sample, "Written": "", "Cells": 0,
                                    "File": str(path), "Problem": str(exc)})
            outcome = pd.DataFrame(results)
            failed = outcome["Problem"].astype(bool).sum()
            (st.warning if failed else st.success)(
                f"Wrote “{key}” into {len(outcome) - failed} of {len(outcome)} file(s)."
                + (" A file open in the viewer may need closing first." if failed else "")
            )
            st.dataframe(outcome, hide_index=True, width="stretch")


def cluster_stack_figure(clustered: pd.DataFrame, region: str, measure: str):
    """Every sample's mix of clusters as one stacked bar, grouped by genotype.

    Each genotype's samples are followed by their mean, so the mean is read
    next to what went into it.
    """
    order = cluster_order(clustered["Cluster"])
    colors = cluster_colors(order)
    genotypes_here = ap.genotype_order(clustered["Genotype"])
    counts = pd.crosstab([clustered["Genotype"], clustered["Sample"]], clustered["Cluster"])
    counts = counts.reindex(columns=order, fill_value=0)
    share = measure.startswith("Share")
    values = counts.div(counts.sum(axis=1), axis=0) * 100 if share else counts
    unit = "% of the sample's cells" if share else "cells"
    groups, bars, stacks = [], [], []
    for genotype in genotypes_here:
        if genotype not in values.index.get_level_values(0):
            continue
        per_sample = values.xs(genotype, level=0)
        for sample, row in per_sample.iterrows():
            groups.append(genotype)
            bars.append(str(sample))
            stacks.append(row)
        groups.append(genotype)
        # Unique per genotype: plotly places a repeated category where it
        # first saw it.
        bars.append(f"{genotype} mean")
        stacks.append(per_sample.mean(axis=0))
    table = pd.DataFrame(stacks).reset_index(drop=True)
    figure = go.Figure()
    for cluster in order:
        figure.add_trace(go.Bar(
            x=[groups, bars], y=table[cluster].to_numpy(), name=str(cluster),
            marker={"color": colors[cluster], "line": {"color": "white", "width": 1}},
            customdata=bars,
            hovertemplate=f"%{{customdata}} · {cluster}: %{{y:.3g}} {unit}<extra></extra>",
        ))
    figure.update_layout(
        barmode="stack", bargap=0.25,
        title=f"Each sample's mix of {region} clusters",
        xaxis={"automargin": True, "tickangle": 0},
        yaxis={"title": unit, "automargin": True},
        # Top to bottom in the legend as in the bars.
        legend_title_text="Cluster", legend={"traceorder": "reversed"},
        height=460, margin={"l": 10, "r": 10, "t": 50, "b": 10},
    )
    return figure


def color_map(frame: pd.DataFrame, by: str) -> tuple[dict[str, str], list[str]]:
    """Fixed colours per category, shared with the report's PNG figures."""
    if by == "Cluster":
        order = cluster_order(frame["Cluster"])
        return cluster_colors(order), order
    if by == "Genotype":
        order = ap.genotype_order(frame["Genotype"])
        return {g: c for g, (c, _m) in ap.genotype_styles(order).items()}, order
    if by == "Sample":
        genotype_of = dict(zip(frame["Sample"], frame["Genotype"]))
        colors = ap.sample_colors(list(dict.fromkeys(frame["Sample"])), genotype_of)
        return colors, list(colors)
    order = list(dict.fromkeys(frame[by]))
    return {v: REGION_COLORS[i % len(REGION_COLORS)] for i, v in enumerate(order)}, order


def symbol_map(values) -> dict[str, str]:
    order = list(dict.fromkeys(values))
    return {v: SYMBOLS[i % len(SYMBOLS)] for i, v in enumerate(order)}


def prepare_matrix(frame: pd.DataFrame, columns: list[str], max_missing: float):
    """Standardised feature matrix, with what was dropped and why."""
    from sklearn.preprocessing import StandardScaler

    data = frame[columns].astype(float)
    missing = data.isna().mean()
    too_sparse = [c for c in columns if missing[c] > max_missing]
    data = data.drop(columns=too_sparse)
    data = data.fillna(data.median())
    constant = [c for c in data.columns if not np.isfinite(data[c].std()) or data[c].std() == 0]
    data = data.drop(columns=constant)
    if data.shape[1] == 0:
        return None, too_sparse, constant, []
    return StandardScaler().fit_transform(data.to_numpy()), too_sparse, constant, list(data.columns)


@st.cache_data(show_spinner="Running UMAP…")
def run_umap(matrix: np.ndarray, neighbors: int, min_dist: float, dims: int, seed: int) -> np.ndarray:
    import umap

    reducer = umap.UMAP(
        n_neighbors=neighbors, min_dist=min_dist, n_components=dims, random_state=seed
    )
    return reducer.fit_transform(matrix)


# -- SHAP ---------------------------------------------------------------------

#: Feature-value ramp in the beeswarm: one hue, light for low, dark for high.
VALUE_SCALE = [[0.0, "#9ec5f4"], [0.5, "#2a78d6"], [1.0, "#0d366b"]]


@st.cache_data(show_spinner="Training the model and computing SHAP values…")
def explain_cells(matrix: np.ndarray, target: np.ndarray, groups: np.ndarray | None,
                  n_explain: int, seed: int):
    """Cross-validated score, which rows were explained, and their SHAP values.

    A random forest predicts *target* from *matrix*. With *groups* (the sample
    of each cell) the score holds whole groups out, so cells of one fish never
    sit on both sides of a split. SHAP values come back as (rows, features,
    classes), in the model's class order.
    """
    import shap
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict

    model = RandomForestClassifier(
        n_estimators=200, max_depth=8, min_samples_leaf=3, class_weight="balanced",
        n_jobs=-1, random_state=seed,
    )
    score = float("nan")
    if groups is not None and len(set(groups)) >= 2:
        folds = GroupKFold(n_splits=min(5, len(set(groups))))
        predicted = cross_val_predict(model, matrix, target, cv=folds, groups=groups)
        score = balanced_accuracy_score(target, predicted)
    elif groups is None:
        smallest = int(np.min(np.unique(target, return_counts=True)[1]))
        if smallest >= 2:
            folds = StratifiedKFold(n_splits=min(5, smallest), shuffle=True, random_state=seed)
            predicted = cross_val_predict(model, matrix, target, cv=folds)
            score = balanced_accuracy_score(target, predicted)
    model.fit(matrix, target)
    rows = np.random.default_rng(seed).permutation(len(matrix))[: int(n_explain)]
    rows.sort()
    values = shap.TreeExplainer(model).shap_values(matrix[rows])
    if isinstance(values, list):  # older shap: one array per class
        values = np.stack(values, axis=-1)
    values = np.asarray(values, dtype=float)
    if values.ndim == 2:
        values = values[:, :, None]
    return score, rows, values, list(model.classes_)


# -- atlas helpers ------------------------------------------------------------

#: Sequential ramp for cell density: one hue, light (few cells) to dark.
DENSITY_SCALE = [
    [0.0, "#f4f8fd"], [0.2, "#b7d3f6"], [0.4, "#6da7ec"],
    [0.6, "#2a78d6"], [0.8, "#1c5cab"], [1.0, "#0d366b"],
]
#: The same, in grey: under cells coloured by cluster or genotype, whose hues
#: would otherwise be lost against a blue ramp.
DENSITY_SCALE_GREY = [
    [0.0, "#f7f7f5"], [0.25, "#d9d8d3"], [0.5, "#b0afa9"], [0.75, "#76756f"], [1.0, "#3a3a37"],
]
#: Diverging ramp for a difference: blue lower, neutral grey equal, red higher.
DIFFERENCE_SCALE = [
    [0.0, "#1c5cab"], [0.25, "#86b6ef"], [0.5, "#f0efec"], [0.75, "#ee9191"], [1.0, "#b8302f"],
]
#: The template outline: a neutral mid grey that reads on light and dark.
TEMPLATE_COLOR = "#8a8984"


@st.cache_data(show_spinner="Registering outlines…")
def register_region(outlines: dict, cells: dict, scale: bool, reflect: bool, warp: bool,
                    smoothing: float):
    """The atlas, its template and registered contours, per-sample fit, and the
    cells carried in."""
    from microscopy_viewer import shape_atlas as sa

    atlas = sa.build_atlas(outlines, scale=scale, reflect=reflect, warp=warp, smoothing=smoothing)
    mapped = {s: atlas.map_points(s, p) for s, p in cells.items() if s in atlas.transforms}
    fits = {
        s: {"Scale": t.scale, "Rotation (°)": t.angle, "Mirrored": t.mirrored,
            "Residual (µm)": atlas.residual[s]}
        for s, t in atlas.transforms.items()
    }
    return atlas, atlas.template, atlas.registered, fits, mapped




def closed(points: np.ndarray) -> np.ndarray:
    return np.vstack([points, points[:1]])


def equal_axes(figure, count: int) -> None:
    """Every subplot at one µm per pixel both ways, without axes clutter.

    Positions in the template mean nothing on their own, so there are no tick
    labels; :func:`scale_bar` gives the size instead.
    """
    for i in range(1, count + 1):
        suffix = "" if i == 1 else str(i)
        figure.layout[f"yaxis{suffix}"].update(scaleanchor=f"x{suffix}", scaleratio=1)
    for update in (figure.update_xaxes, figure.update_yaxes):
        update(showgrid=False, zeroline=False, showticklabels=False, ticks="")


def scale_bar(figure, template: np.ndarray, row=None, col=None) -> None:
    """A round-length bar under the template's lower left, labelled in µm."""
    low, high = template.min(axis=0), template.max(axis=0)
    width = float(high[0] - low[0])
    length = next((v for v in (500, 200, 100, 50, 20, 10, 5) if v <= width * 0.5), 5)
    y = float(low[1]) - 0.04 * float(high[1] - low[1])
    x0 = float(low[0])
    figure.add_trace(
        go.Scatter(x=[x0, x0 + length], y=[y, y], mode="lines+text", showlegend=False,
                   line={"color": TEMPLATE_COLOR, "width": 3}, text=["", f"{length} µm"],
                   textposition="middle right", hoverinfo="skip"),
        row=row, col=col,
    )


#: Colour of cells when they are not coloured by anything.
PLAIN = "Cells"


def overlay_traces(figure, groups: dict, colors: dict, shown: set, row, col) -> None:
    """Cells of one panel: ``group -> {"points": (N, 2), "outlines": [(M, 2), ...]}``.

    Outlines when there are any, dots otherwise. One trace per group; each
    group is in the legend once, and toggling it there hides it in every panel.
    """
    for group, entry in groups.items():
        color = colors.get(group, NEUTRAL)
        plain = group == PLAIN
        first = group not in shown
        shown.add(group)
        common = dict(name=str(group), legendgroup=str(group), showlegend=first and not plain)
        if entry["outlines"]:
            xs: list = []
            ys: list = []
            for poly in entry["outlines"]:
                xs.extend([*poly[:, 0], poly[0, 0], None])
                ys.extend([*poly[:, 1], poly[0, 1], None])
            figure.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines", fill="toself", hoverinfo="skip",
                line={"color": "#1f1f1e" if plain else color, "width": 1},
                fillcolor="rgba(255,255,255,0.8)" if plain else _alpha(color, 0.6), **common,
            ), row=row, col=col)
        else:
            points = np.asarray(entry["points"]).reshape(-1, 2)
            figure.add_trace(go.Scatter(
                x=points[:, 0], y=points[:, 1], mode="markers",
                marker={"size": 7, "color": "rgba(255,255,255,0.85)" if plain else color,
                        "line": {"width": 1, "color": "#1f1f1e" if plain else "white"}},
                hovertemplate=f"{group}<extra></extra>", **common,
            ), row=row, col=col)


def density_figure(grid, panels: dict, template: np.ndarray, *, colorscale, zmin, zmax,
                   unit: str, cells: dict | None = None, colors: dict | None = None,
                   height: int = 620, columns: int = 0):
    """One heatmap per panel on a shared colour scale, the template drawn over it.

    *cells* maps a panel to its cells for :func:`overlay_traces`.
    """
    from plotly.subplots import make_subplots

    names = list(panels)
    columns = columns or len(names)
    rows = int(np.ceil(len(names) / columns))
    figure = make_subplots(rows=rows, cols=columns, subplot_titles=names,
                           horizontal_spacing=0.03, vertical_spacing=0.08)
    outline = closed(template)
    shown: set = set()
    for i, name in enumerate(names):
        row, col = i // columns + 1, i % columns + 1
        figure.add_trace(
            go.Heatmap(
                x=grid.x, y=grid.y, z=panels[name], coloraxis="coloraxis",
                hovertemplate=f"x %{{x:.0f}} µm<br>y %{{y:.0f}} µm<br>%{{z:.3g}} {unit}<extra>{name}</extra>",
            ),
            row=row, col=col,
        )
        if cells and cells.get(name):
            overlay_traces(figure, cells[name], colors or {}, shown, row, col)
        figure.add_trace(
            go.Scatter(x=outline[:, 0], y=outline[:, 1], mode="lines", showlegend=False,
                       line={"color": TEMPLATE_COLOR, "width": 2}, hoverinfo="skip"),
            row=row, col=col,
        )
        scale_bar(figure, template, row, col)
    figure.update_layout(
        coloraxis={"colorscale": colorscale, "cmin": zmin, "cmax": zmax,
                   "colorbar": {"title": {"text": unit, "side": "right"}}},
        height=height * rows, margin={"l": 10, "r": 10, "t": 50, "b": 40},
        plot_bgcolor="rgba(0,0,0,0)",
        legend={"orientation": "h", "x": 0, "y": 0, "yanchor": "top", "itemsizing": "constant"},
    )
    equal_axes(figure, len(names))
    return figure


def styled(figure, height: int = 620):
    figure.update_traces(marker={"size": 9, "line": {"width": 1, "color": "white"}})
    figure.update_layout(
        height=height, legend={"itemsizing": "constant"},
        margin={"l": 10, "r": 10, "t": 40, "b": 10},
    )
    return figure


def region_key(row) -> tuple:
    return (str(row["Sample"]), str(row["Region"]))


def cell_key(row) -> tuple:
    return (str(row["Sample"]), int(row["Label"]))


def outline_figure(
    plot, x, y, color_by, colors, order, outlines, size, true_scale, title="",
    key_of=region_key, ident=("Sample", "Genotype", "Region"), extra=None,
):
    """Each row drawn as its own outline, centred on its (x, y).

    *outlines* maps ``key_of(row)`` to a list of (N, 2) X/Y outlines — the
    parts of one region, or the single outline of one cell. *ident* are the
    columns named in the hover, the first in bold; *extra* the numbers after them.

    One filled trace per colour, outlines separated by gaps, so a few thousand
    regions stay one draw call each. Hover comes from an invisible marker at the
    centre — plotly does not report which part of a filled trace was hovered.

    *size* is the glyph size as a fraction of the plot's shorter side. With
    *true_scale* every outline keeps its real size relative to the others (the
    largest region is *size* across); otherwise each is scaled to fill *size*,
    which shows shape better when regions differ a lot in area.

    The axes can be in any units — area against a count, say — so glyphs are
    sized in screen pixels, not data units: the figure has a fixed plot area
    and explicit ranges, which fixes how many data units a pixel is on each
    axis, and each outline is stretched by that per axis to come out undistorted.
    """
    import plotly.graph_objects as go

    xs, ys = plot[x].to_numpy(float), plot[y].to_numpy(float)
    glyph_px = size * min(PLOT_W, PLOT_H)
    # Leave room for half a glyph (plus a little) at every edge.
    pad_x = (np.ptp(xs) or 1.0) * (glyph_px / PLOT_W + 0.04)
    pad_y = (np.ptp(ys) or 1.0) * (glyph_px / PLOT_H + 0.04)
    x_range = (float(xs.min() - pad_x), float(xs.max() + pad_x))
    y_range = (float(ys.min() - pad_y), float(ys.max() + pad_y))
    per_px_x = (x_range[1] - x_range[0]) / PLOT_W
    per_px_y = (y_range[1] - y_range[0]) / PLOT_H
    keys = [key_of(row) for _, row in plot.iterrows()]
    plot = plot.assign(_key=keys)
    extents = {}
    for key in set(keys):
        parts = outlines.get(key)
        if parts:
            points = np.vstack(parts)
            extents[key] = float(np.max(np.ptp(points, axis=0))) or 1.0
    largest = max(extents.values(), default=1.0)

    figure = go.Figure()
    if extra is None:
        extra = [c for c in hover_columns(plot) if c not in ident]
    ident = list(ident)
    for category in order:
        members = plot[plot[color_by] == category]
        if members.empty:
            continue
        color = colors.get(category, "#8a8983")
        gx: list = []
        gy: list = []
        for _, row in members.iterrows():
            key = row["_key"]
            parts = outlines.get(key)
            if not parts:
                continue
            centre = np.vstack(parts).mean(axis=0)
            pixels = glyph_px / (largest if true_scale else extents[key])
            for part in parts:
                shifted = (part - centre) * pixels * np.array([per_px_x, per_px_y])
                gx.extend([*(shifted[:, 0] + row[x]), shifted[0, 0] + row[x], None])
                gy.extend([*(shifted[:, 1] + row[y]), shifted[0, 1] + row[y], None])
        figure.add_trace(go.Scatter(
            x=gx, y=gy, mode="lines", fill="toself", name=str(category),
            legendgroup=str(category), line={"color": color, "width": 1.5},
            fillcolor=_alpha(color, 0.35), hoverinfo="skip",
        ))
        custom = members[[*ident, *extra]].to_numpy()
        lines = "<br>".join(
            [f"{c}: %{{customdata[{i}]}}" for i, c in enumerate(ident) if i]
            + [f"{c}: %{{customdata[{i + len(ident)}]:.4g}}" for i, c in enumerate(extra)]
        )
        figure.add_trace(go.Scatter(
            x=members[x], y=members[y], mode="markers", legendgroup=str(category),
            showlegend=False, marker={"size": 14, "color": color, "opacity": 0.01},
            customdata=custom,
            hovertemplate=f"<b>%{{customdata[0]}}</b><br>{lines}<extra></extra>",
        ))
    missing = sum(key not in outlines for key in keys)
    figure.update_layout(
        title=title, xaxis_title=x, yaxis_title=y, legend_title_text=color_by,
        width=MARGIN["l"] + PLOT_W + MARGIN["r"],
        height=MARGIN["t"] + PLOT_H + MARGIN["b"],
        margin=MARGIN, legend={"x": 1.02, "xanchor": "left", "y": 1},
        xaxis={"range": x_range, "autorange": False},
        yaxis={"range": y_range, "autorange": False},
    )
    return figure, missing




def glyph_controls(key: str, available: bool, missing_note: str = ""):
    """The 'draw as' switch and its settings. Returns (outlines?, size, true scale)."""
    if not available:
        st.caption(missing_note or "Outlines need a workbook with the “Region outlines” sheet.")
        return False, 0.0, False
    drawn = st.radio("Draw as", ("Dots", "Outlines"), horizontal=True, key=f"{key}-draw")
    if drawn == "Dots":
        return False, 0.0, False
    size = st.slider("Outline size", 0.01, 0.3, 0.06, 0.01, key=f"{key}-size")
    true_scale = st.checkbox("Keep relative sizes", value=True, key=f"{key}-true")
    return True, float(size), bool(true_scale)




# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------


st.set_page_config(page_title="Region explorer", layout="wide")
st.title("Region explorer")

with st.sidebar:
    st.header("Workbook")
    uploaded = st.file_uploader("Analysis workbook (.xlsx)", type=["xlsx"])
    default_path = sys.argv[1] if len(sys.argv) > 1 else ""
    typed = st.text_input("…or a path on this machine", value=default_path)
    uploaded_cells = st.file_uploader(
        "Cell outlines (.npz)", type=["npz"],
        help="Found by itself beside a workbook given as a path.",
    )

source: bytes | str | None = None
cell_source = None
if uploaded is not None:
    source = uploaded.getvalue()
elif typed.strip():
    candidate = Path(typed.strip()).expanduser()
    if candidate.is_dir():
        books = sorted(candidate.glob("*.xlsx"))
        candidate = books[-1] if books else candidate
    if candidate.is_file():
        source = str(candidate)
        beside = candidate.parent / ap.CELL_OUTLINES_FILE
        cell_source: bytes | str | None = str(beside) if beside.is_file() else None
    else:
        st.sidebar.error(f"No workbook at {candidate}")

if source is None:
    st.info(
        "Load the workbook an Analysis run wrote — the .xlsx in its "
        "`<experiment>_analysis_<date>` folder. It needs the “Region features” sheet."
    )
    st.stop()

try:
    features = read_features(source)
except ValueError as exc:
    st.error(f"Could not read the “{SHEET}” sheet: {exc}")
    st.stop()
outlines = read_outlines(source)
#: The condition columns the analysis wrote, beside the genotype.
conditions = group_columns(features)[1:]
if uploaded_cells is not None:
    cell_source = uploaded_cells.getvalue()

with st.sidebar:
    st.header("Rows")
    regions = st.multiselect("Regions", list(dict.fromkeys(features["Region"])),
                             default=list(dict.fromkeys(features["Region"])))
    genotypes = st.multiselect("Genotypes", ap.genotype_order(features["Genotype"]),
                               default=ap.genotype_order(features["Genotype"]))
    samples = st.multiselect("Samples", list(dict.fromkeys(features["Sample"])),
                             default=list(dict.fromkeys(features["Sample"])))

rows = features[
    features["Region"].isin(regions)
    & features["Genotype"].isin(genotypes)
    & features["Sample"].isin(samples)
].reset_index(drop=True)

st.caption(
    f"{len(rows)} region row(s) from {rows['Sample'].nunique()} sample(s), "
    f"{rows['Genotype'].nunique()} genotype(s), {rows['Region'].nunique()} region(s)."
)
if rows.empty:
    st.warning("No rows left after filtering.")
    st.stop()

# Every cell view — Cells, the clustering, the Atlas — sees only cells in the
# size range kept here.
cells_all = read_cells(source)
SIZE_COLUMNS = ("Equivalent diameter (µm)", "Area (µm²)", "Footprint area (µm²)", "Volume (µm³)")
if not cells_all.empty:
    size_columns = [c for c in SIZE_COLUMNS if c in cells_all and cells_all[c].notna().any()]
    if size_columns:
        with st.sidebar:
            st.header("Cells")
            size_column = st.selectbox("Size measure", size_columns, key="size-column")
            values = cells_all[size_column].astype(float)
            low, high = float(values.min()), float(values.max())
            if high > low:
                kept = st.slider(
                    "Keep cells from … to …", low, high, (low, high),
                    step=float((high - low) / 500), format="%.1f", key=f"size-{size_column}",
                    help="Drops debris below and merged clumps above, everywhere cells "
                         "are shown, clustered or mapped.",
                )
                keep = values.between(*kept)
                st.caption(f"{int(keep.sum()):,} of {len(cells_all):,} cells kept.")
                cells_all = cells_all[keep].reset_index(drop=True)

cell_outlines = read_cell_outlines(cell_source) if cell_source is not None else {}
#: ``(sample, label) -> cluster name``, filled in by the Cells tab for the atlas.
cluster_of: dict[tuple[str, int], str] = {}

umap_tab, scatter_tab, cells_tab, atlas_tab, explain_tab, plots_tab, table_tab = st.tabs(
    ["UMAP Regions", "Scatter Regions", "Cells", "Atlas", "Explain", "Plots", "Table"]
)

# -- UMAP ---------------------------------------------------------------------

with umap_tab:
    groups = feature_groups(rows)
    left, right = st.columns([1, 3])
    with left:
        st.subheader("Features")
        chosen: list[str] = []
        for name, columns in groups.items():
            if not columns:
                continue
            on = st.checkbox(f"{name} ({len(columns)})", value=name != "Position", key=f"g-{name}")
            if on:
                chosen.extend(columns)
        with st.expander("Fine-tune columns"):
            chosen = st.multiselect("Columns used", numeric_columns(rows), default=chosen)
        max_missing = st.slider("Drop columns missing in more than", 0, 100, 50, 5,
                                format="%d%%") / 100.0

        st.subheader("UMAP")
        dims = st.radio("Dimensions", (2, 3), horizontal=True, key="umap-dims")
        limit = max(2, len(rows) - 1)
        neighbors = st.slider("Neighbours", 2, max(2, min(100, limit)), min(15, limit))
        min_dist = st.slider("Minimum distance", 0.0, 1.0, 0.1, 0.05)
        seed = st.number_input("Seed", value=0, step=1)
        color_by = st.selectbox("Colour by", ("Genotype", *conditions, "Sample", "Region"),
                                key="umap-color")
        symbol_by = st.selectbox("Shape by", ("Region", "Genotype", *conditions, "None"),
                                 key="umap-symbol")
        if int(dims) == 2:
            as_outlines, glyph, true_scale = glyph_controls("umap", bool(outlines))
        else:
            as_outlines = False
            st.caption("Outlines are drawn in 2D only.")

    with right:
        if len(rows) < 4:
            st.warning("UMAP needs at least four rows.")
        elif not chosen:
            st.warning("Choose at least one feature group.")
        else:
            matrix, sparse, constant, used = prepare_matrix(rows, chosen, max_missing)
            if matrix is None:
                st.warning("Every chosen column was empty or constant.")
            else:
                embedding = run_umap(matrix, int(neighbors), float(min_dist), int(dims), int(seed))
                plot = rows.copy()
                axes = [f"UMAP {i + 1}" for i in range(int(dims))]
                for i, axis in enumerate(axes):
                    plot[axis] = embedding[:, i]
                colors, order = color_map(plot, color_by)
                symbol = None if symbol_by == "None" else symbol_by
                common = dict(
                    color=color_by, color_discrete_map=colors,
                    category_orders={color_by: order},
                    symbol=symbol,
                    symbol_map=symbol_map(plot[symbol]) if symbol else None,
                    hover_name="Sample", hover_data=hover_columns(plot),
                    title=f"UMAP of {len(plot)} regions on {len(used)} features",
                )
                missing = 0
                if int(dims) == 3:
                    figure = px.scatter_3d(plot, x=axes[0], y=axes[1], z=axes[2], **common)
                    figure = styled(figure, 700)
                    figure.update_traces(marker={"size": 5})
                elif as_outlines:
                    figure, missing = outline_figure(
                        plot, axes[0], axes[1], color_by, colors, order, outlines,
                        glyph, true_scale, common["title"],
                    )
                else:
                    figure = styled(px.scatter(plot, x=axes[0], y=axes[1], **common))
                st.plotly_chart(
                    figure, width="content" if as_outlines and int(dims) == 2 else "stretch",
                    theme="streamlit",
                )
                if missing:
                    st.caption(f"{missing} region(s) have no stored outline and are not drawn.")
                with st.expander(f"{len(used)} features used"):
                    st.write(", ".join(used))
                    if sparse:
                        st.write(f"Dropped as too sparse: {', '.join(sparse)}")
                    if constant:
                        st.write(f"Dropped as constant: {', '.join(constant)}")
                st.download_button(
                    "Download embedding (CSV)",
                    plot[["Sample", "Genotype", "Region", *axes]].to_csv(index=False),
                    file_name="umap_regions.csv",
                )

# -- scatter ------------------------------------------------------------------

with scatter_tab:
    numeric = numeric_columns(rows)
    left, right = st.columns([1, 3])
    with left:
        mode = st.radio("Plot", ("2D", "3D"), horizontal=True)

        def _pick(label, index, key):
            return st.selectbox(label, numeric, index=min(index, len(numeric) - 1), key=key)

        preferred = [c for c in ("Region area (µm²)", "Objects", "Objects per mm²") if c in numeric]
        start = [numeric.index(c) for c in preferred] + [0, 1, 2]
        x = _pick("X", start[0], "sx")
        y = _pick("Y", start[1], "sy")
        z = _pick("Z", start[2], "sz") if mode == "3D" else None
        color_by = st.selectbox("Colour by", ("Sample", "Genotype", *conditions, "Region"),
                                key="sc-color")
        symbol_by = st.selectbox("Shape by", ("Genotype", *conditions, "Region", "None"),
                                 key="sc-symbol")
        log_x = st.checkbox("Log X")
        log_y = st.checkbox("Log Y")
        log_z = st.checkbox("Log Z") if mode == "3D" else False
        if mode == "2D" and not (log_x or log_y):
            sc_outlines, sc_glyph, sc_true = glyph_controls("sc", bool(outlines))
        else:
            sc_outlines = False
            st.caption("Outlines are drawn in 2D on linear axes only.")

    with right:
        if not numeric:
            st.warning("No numeric columns.")
        else:
            wanted = [c for c in (x, y, z) if c]
            plot = rows.dropna(subset=wanted)
            dropped = len(rows) - len(plot)
            colors, order = color_map(plot, color_by)
            symbol = None if symbol_by == "None" else symbol_by
            common = dict(
                color=color_by, color_discrete_map=colors,
                category_orders={color_by: order},
                symbol=symbol,
                symbol_map=symbol_map(plot[symbol]) if symbol else None,
                hover_name="Sample", hover_data=hover_columns(plot),
            )
            if mode == "3D":
                figure = px.scatter_3d(plot, x=x, y=y, z=z, log_x=log_x, log_y=log_y,
                                       log_z=log_z, **common)
                figure = styled(figure, 700)
                figure.update_traces(marker={"size": 5})
            elif sc_outlines:
                figure, missing = outline_figure(
                    plot, x, y, color_by, colors, order, outlines, sc_glyph, sc_true
                )
                if missing:
                    st.caption(f"{missing} region(s) have no stored outline and are not drawn.")
            else:
                figure = styled(px.scatter(plot, x=x, y=y, log_x=log_x, log_y=log_y, **common))
            st.plotly_chart(
                figure, width="content" if sc_outlines else "stretch", theme="streamlit"
            )
            if dropped:
                st.caption(f"{dropped} row(s) without a value in the chosen columns are not shown.")

# -- cells --------------------------------------------------------------------

with cells_tab:
    if cells_all.empty:
        st.info("This workbook has no “Objects” sheet — no cells to show.")
    else:
        cell_regions = list(dict.fromkeys(cells_all["Region"]))
        left, right = st.columns([1, 3])
        with left:
            st.subheader("Cells")
            picked_regions = st.multiselect(
                "From regions", cell_regions, default=cell_regions, key="cell-regions",
                help="Only cells of these regions are embedded and clustered — pick the "
                     "cerebellum alone, say, to see how its cells differ among themselves.",
            )
        cells = cells_all[
            cells_all["Region"].isin(picked_regions)
            & cells_all["Genotype"].isin(genotypes)
            & cells_all["Sample"].isin(samples)
        ].reset_index(drop=True)
        if cells.empty:
            right.warning("No cells in the chosen regions, genotypes and samples.")
        else:
            with left:
                cap = st.slider("Cells to use", 100, max(100, min(50000, len(cells))),
                                min(5000, max(100, len(cells))), 100,
                                help="A random subset, the same for every setting. UMAP on "
                                     "tens of thousands of cells takes a minute.")
                cell_seed = st.number_input("Sampling seed", value=0, step=1, key="cell-seed")
                view = st.radio("View", ("Embedding", "Scatter"), horizontal=True)
                clustering = st.session_state.get("clu-method", "Off") != "Off"
                color_cells = st.selectbox(
                    "Colour by", ("Sample", "Genotype", *[c for c in conditions if c in cells], "Region")
                    + (("Cluster",) if clustering else ()),
                    key="cell-color",
                )
                numeric_cells = [
                    c for c in cells.columns
                    if c not in CELL_IDENTITY and pd.api.types.is_numeric_dtype(cells[c])
                ]
                if view == "Embedding":
                    method = st.radio("Method", ("UMAP", "PCA"), horizontal=True)
                    cdims = st.radio("Dimensions", (2, 3), horizontal=True, key="cell-dims")
                    chosen_cells: list[str] = []
                    for name, columns in cell_groups(cells).items():
                        if columns and st.checkbox(f"{name} ({len(columns)})",
                                                   value=name != "Position", key=f"cg-{name}"):
                            chosen_cells.extend(columns)
                    cneighbors = st.slider("Neighbours", 2, 100, 15, key="cell-nb") if method == "UMAP" else 15
                    cmin_dist = st.slider("Minimum distance", 0.0, 1.0, 0.1, 0.05,
                                          key="cell-md") if method == "UMAP" else 0.1
                    two_d = int(cdims) == 2
                else:
                    cmode = st.radio("Plot", ("2D", "3D"), horizontal=True, key="cell-mode")
                    pick = [c for c in ("Equivalent diameter (µm)", "Mean intensity", "Circularity")
                            if c in numeric_cells] + numeric_cells[:3]
                    cx = st.selectbox("X", numeric_cells, index=numeric_cells.index(pick[0]), key="cx")
                    cy = st.selectbox("Y", numeric_cells, index=numeric_cells.index(pick[1]), key="cy")
                    cz = (st.selectbox("Z", numeric_cells, index=numeric_cells.index(pick[2]), key="cz")
                          if cmode == "3D" else None)
                    two_d = cmode == "2D"
                if two_d:
                    cell_draw, cell_glyph, cell_true = glyph_controls(
                        "cell", bool(cell_outlines),
                        "Cell outlines need the cell_outlines.npz from the report folder — "
                        "give the workbook as a path, or upload the file.",
                    )
                else:
                    cell_draw = False

                st.subheader("Clusters")
                cluster_method = st.selectbox(
                    "Clustering", CLUSTER_METHODS, key="clu-method",
                    help="Groups the cells of one region by the features ticked below, "
                         "standardised first; every cell passing the filters is used, not "
                         "only the sampled ones. The clusters colour the cells here and in "
                         "the Atlas tab.\n\n"
                         "- **K-means**: round, similar-sized groups. The default; fast and "
                         "stable. You choose how many.\n"
                         "- **Gaussian mixture**: like k-means but groups may be elongated "
                         "or of different spread — better when one feature varies much more "
                         "within a group than another. You choose how many.\n"
                         "- **HDBSCAN**: finds the number itself from where the cells are "
                         "dense, and leaves cells that fit nowhere as noise. Good for spotting "
                         "a small distinct population; results shift with the smallest "
                         "cluster size.",
                )
                cluster_features: list[str] = []
                cluster_k, cluster_min = 4, 15
                cluster_region = None
                if cluster_method != "Off":
                    here = list(dict.fromkeys(cells["Region"]))
                    wanted = st.session_state.get("atlas-region", "CB")
                    cluster_region = st.selectbox(
                        "Cluster the cells of", here,
                        index=here.index(wanted) if wanted in here else 0, key="clu-region",
                        help="Only this region's cells are clustered; the others are "
                             "shown as “not clustered”. Follows the Atlas region at first.",
                    )
                    for name, columns in cell_groups(cells).items():
                        if columns and st.checkbox(f"{name} ({len(columns)})",
                                                   value=name != "Position", key=f"clu-{name}",
                                                   help=GROUP_HELP.get(name)):
                            cluster_features.extend(columns)
                    if cluster_method == "HDBSCAN":
                        cluster_min = st.slider(
                            "Smallest cluster", 5, 200, 15, key="clu-min",
                            help="The fewest cells a group needs to count as a cluster; "
                                 "HDBSCAN then finds how many there are. Smaller finds more, "
                                 "finer clusters (and more noise); larger merges them. Start "
                                 "near 1–2 % of the cells, and prefer a size where the "
                                 "clusters stay put when you nudge it. Cells in no cluster "
                                 "are grey “Noise / other”.",
                        )
                    else:
                        cluster_k = st.slider(
                            "Clusters", 2, len(REGION_COLORS), 4, key="clu-k",
                            help="How many groups to split the cells into. There is no "
                                 "right answer in the data alone — pick the fewest that "
                                 "separate things you can name:\n\n"
                                 "- open **Choosing the number of clusters** below for a "
                                 "score per count (higher silhouette is better; for Gaussian "
                                 "mixture, lower BIC);\n"
                                 "- check **Cluster profiles**: two clusters that differ in "
                                 "nothing you care about should be one;\n"
                                 "- a good count gives similar clusters when you change the "
                                 "features or the seed slightly.\n\n"
                                 "At most eight, one colour each.",
                        )

            cluster_used: list[str] = []
            if cluster_method != "Off":
                matrix = None
                members = cells["Region"] == cluster_region
                clustered = cells[members]
                if cluster_features and len(clustered) > max(cluster_k, 2):
                    matrix, _sparse, _constant, cluster_used = prepare_matrix(
                        clustered, cluster_features, 0.5
                    )
                if matrix is None:
                    left.warning(f"Not enough {cluster_region} cells, or no feature group with "
                                 "values, to cluster.")
                else:
                    found = cluster_cells(matrix, cluster_method, int(cluster_k), int(cluster_min), 0)
                    names = pd.Series(NOT_CLUSTERED, index=cells.index, dtype=object)
                    names[members] = cluster_names(found)
                    cells = cells.assign(Cluster=names.to_numpy())
                    clustered = cells[members]
                    cluster_of.update(zip(zip(clustered["Sample"], clustered["Label"].astype(int)),
                                          clustered["Cluster"]))
            if cluster_method in ("K-means", "Gaussian mixture") and matrix is not None:
                with left.expander("Choosing the number of clusters"):
                    st.caption("Scores each count from 2 to 8 on these cells and features. "
                               "Silhouette (−1 to 1): how much closer cells are to their own "
                               "cluster than to the next; above ~0.25 is some structure, "
                               "above 0.5 clear. A peak, or where it stops dropping, is a "
                               "good candidate."
                               + (" BIC: lower is better; look for the elbow."
                                  if cluster_method == "Gaussian mixture" else ""))
                    if st.checkbox("Score them", key="clu-score"):
                        scores = score_cluster_counts(matrix, cluster_method, 0)
                        chart = go.Figure(go.Scatter(
                            x=scores["Clusters"], y=scores["Silhouette"], mode="lines+markers",
                            name="Silhouette", line={"color": "#2a78d6", "width": 2},
                            marker={"size": 8},
                            hovertemplate="%{x} clusters: silhouette %{y:.3f}<extra></extra>",
                        ))
                        chart.add_vline(x=int(cluster_k), line={"color": NEUTRAL, "dash": "dot"})
                        chart.update_layout(height=220, margin={"l": 10, "r": 10, "t": 10, "b": 10},
                                            xaxis={"title": "clusters", "dtick": 1},
                                            yaxis={"title": "silhouette"}, showlegend=False)
                        st.plotly_chart(chart, width="stretch", theme="streamlit")
                        best = int(scores.loc[scores["Silhouette"].idxmax(), "Clusters"])
                        note = f"Best silhouette at {best}."
                        if "BIC" in scores:
                            low = int(scores.loc[scores["BIC"].idxmin(), "Clusters"])
                            note += f" Lowest BIC at {low}."
                        st.caption(note + " Dotted line: the count in use.")
            if color_cells == "Cluster" and "Cluster" not in cells:
                color_cells = "Sample"

            with right:
                if len(cells) > cap:
                    subset = cells.sample(n=int(cap), random_state=int(cell_seed)).reset_index(drop=True)
                else:
                    subset = cells
                hover = {c: True for c in ("Sample", "Genotype", "Region", "Cluster", "Label") if c in subset}
                hover.update({c: ":.4g" for c in CELL_HOVER if c in subset})
                extra = [c for c in CELL_HOVER if c in subset]
                ident = [c for c in ("Sample", "Genotype", "Region", "Cluster", "Label") if c in subset]
                colors, order = color_map(subset, color_cells)
                common = dict(
                    color=color_cells, color_discrete_map=colors,
                    category_orders={color_cells: order},
                    hover_name="Sample", hover_data=hover,
                )
                figure = None
                cell_plot = None
                if view == "Embedding":
                    cz = None
                    if len(subset) < 4 or not chosen_cells:
                        st.warning("Choose at least one feature group (and have four cells or more).")
                    else:
                        matrix, sparse, constant, used = prepare_matrix(subset, chosen_cells, 0.5)
                        if matrix is None:
                            st.warning("Every chosen column was empty or constant.")
                        else:
                            points, names = embed(matrix, method, int(cdims),
                                                  int(min(cneighbors, len(subset) - 1)),
                                                  float(cmin_dist), 0)
                            cell_plot = subset.assign(**{n: points[:, i] for i, n in enumerate(names)})
                            title = f"{method} of {len(cell_plot):,} cells on {len(used)} features"
                            cx, cy = names[0], names[1]
                            cz = names[2] if len(names) > 2 else None
                            st.caption("Features: " + ", ".join(used))
                else:
                    cell_plot = subset.dropna(subset=[c for c in (cx, cy, cz) if c])
                    title = f"{len(cell_plot):,} cells"
                if cell_plot is not None:
                    plot = cell_plot
                    if cell_draw:
                        shown = plot
                        if len(plot) > MAX_CELL_GLYPHS:
                            shown = plot.sample(n=MAX_CELL_GLYPHS, random_state=0)
                            st.caption(f"Outlines drawn for {MAX_CELL_GLYPHS:,} of {len(plot):,} cells.")
                        figure, missing = outline_figure(
                            shown, cx, cy, color_cells, colors, order, cell_outlines,
                            cell_glyph, cell_true, title, key_of=cell_key, ident=ident, extra=extra,
                        )
                        if missing:
                            st.caption(f"{missing} cell(s) have no stored outline.")
                    elif cz is not None and not two_d:
                        figure = px.scatter_3d(plot, x=cx, y=cy, z=cz, title=title, **common)
                        figure.update_traces(marker={"size": 3})
                        figure.update_layout(height=700)
                    else:
                        figure = px.scatter(plot, x=cx, y=cy, title=title, render_mode="webgl", **common)
                        figure.update_traces(marker={"size": 6, "opacity": 0.8})
                        figure.update_layout(height=620)
                if figure is not None:
                    st.plotly_chart(figure, width="content" if cell_draw else "stretch",
                                    theme="streamlit")
                if "Cluster" in cells and cluster_used:
                    controls = st.columns(2)
                    dist_style = controls[0].radio(
                        "Cluster distribution", ("Grouped", "Stacked"), horizontal=True,
                        key="clu-dist-style",
                        help="Grouped compares one cluster between genotypes, with every "
                             "fish as a dot. Stacked shows each fish's whole mix of clusters "
                             "in one bar.",
                    )
                    dist_measure = controls[1].radio(
                        "As", ("Share of cells (%)", "Cells"), horizontal=True, key="clu-dist",
                    )
                    clustered_only = cells[cells["Cluster"] != NOT_CLUSTERED]
                    if dist_style == "Grouped":
                        figure = cluster_distribution_figure(clustered_only, cluster_region, dist_measure)
                        note = ("Bars: mean over each genotype's samples, each sample weighed "
                                "the same. Dots: the samples themselves.")
                    else:
                        figure = cluster_stack_figure(clustered_only, cluster_region, dist_measure)
                        note = ("One bar per sample, grouped by genotype; the last bar of "
                                "each group is its samples averaged, each weighed the same.")
                    st.plotly_chart(figure, width="stretch", theme="streamlit")
                    st.caption(note)
                    write_clusters_panel(source, cells, cluster_region, cluster_method,
                                         cluster_k, cluster_min, cluster_used)
                    with st.expander(f"Cluster profiles ({cluster_region})", expanded=False):
                        clustered = cells[cells["Cluster"] != NOT_CLUSTERED]
                        order = cluster_order(clustered["Cluster"])
                        data = clustered[cluster_used].astype(float)
                        spread = data.std().replace(0, np.nan)
                        profile = ((data.groupby(clustered["Cluster"]).mean() - data.mean())
                                   / spread).reindex(order)
                        sizes = clustered["Cluster"].value_counts().reindex(order)
                        st.caption(f"Mean of each feature per cluster, in standard deviations "
                                   f"from the mean of all {cluster_region} cells.")
                        profile.insert(0, "Cells", sizes)
                        st.dataframe(
                            profile.style.format("{:+.2f}", subset=cluster_used).background_gradient(
                                cmap="RdBu_r", vmin=-2, vmax=2, subset=cluster_used
                            ),
                            width="stretch",
                        )
                        st.caption("Share of each genotype's cells in each cluster (%).")
                        share = pd.crosstab(clustered["Cluster"], clustered["Genotype"],
                                            normalize="columns") * 100
                        st.dataframe(share.reindex(order).style.format("{:.1f}"), width="stretch")

# -- atlas --------------------------------------------------------------------

with atlas_tab:
    from microscopy_viewer import shape_atlas as sa

    atlas_regions = list(dict.fromkeys(region for (_s, region) in outlines))
    if not atlas_regions:
        st.info("The atlas needs the “Region outlines” sheet, which this workbook lacks.")
    else:
        left, right = st.columns([1, 3])
        with left:
            st.subheader("Mean shape")
            region = st.selectbox(
                "Region", atlas_regions, key="atlas-region",
                index=atlas_regions.index("CB") if "CB" in atlas_regions else 0,
                help="Its outline in every sample is registered into one mean shape.",
            )
            fit = st.radio(
                "Registration", ("Shape (move, turn, scale)", "Rigid (move, turn)"),
                help="Shape brings every outline to the same size, so the template is the "
                     "mean shape at the mean size. Rigid keeps each sample's own size.",
            )
            reflect = st.checkbox("Allow mirroring", value=False,
                                  help="For samples mounted the other way up.")
            warp = st.checkbox(
                "Bend each outline onto the mean (thin-plate spline)", value=True,
                help="Cells follow their own outline's shape into the template, so a cell "
                     "at the edge lands at the template's edge. Off: the move/turn/scale only.",
            )
            smoothing = st.slider("Bend smoothing", 0.0, 50.0, 0.0, 1.0, disabled=not warp,
                                  help="0 bends the outline exactly onto the template; higher "
                                       "values bend it less, ignoring small wiggles.")
            st.subheader("Heatmap")
            sigma = st.slider("Smoothing radius (µm)", 2.0, 60.0, 15.0, 1.0)
            units = st.radio("Per sample", (sa.PER_AREA, sa.SHARE),
                             help="Share compares where the cells are regardless of how many "
                                  "each sample has.")
            st.subheader("Cells")
            draw = st.radio("Draw the cells", ("Hidden", "Dots", "Outlines"), horizontal=True,
                            key="atlas-draw")
            if draw == "Outlines" and not cell_outlines:
                st.caption("Cell outlines need the cell_outlines.npz from the report folder; "
                           "drawing dots instead.")
            clusters_found = cluster_order(set(cluster_of.values()))
            clustered_region = st.session_state.get("clu-region")
            if clusters_found and clustered_region and clustered_region != region:
                st.caption(f"The clusters are of the {clustered_region} cells; {region} cells "
                           f"show as “{NOT_CLUSTERED}”. Pick {region} under Clusters in the "
                           f"Cells tab to cluster them.")
            atlas_color = st.selectbox(
                "Colour the cells by",
                (["Cluster"] if clusters_found else []) + ["Genotype", "Nothing"],
                key="atlas-color",
            )
            heat_of = "All cells"
            if clusters_found:
                heat_of = st.selectbox("Heatmap of", ["All cells", *clusters_found], key="atlas-heat")
            else:
                st.caption("Turn clustering on in the Cells tab to colour the cells by "
                           "cluster, or map one cluster.")

        chosen = [s for s in samples if (s, region) in outlines]
        genotype_of = dict(zip(features["Sample"], features["Genotype"]))
        chosen = [s for s in chosen if genotype_of.get(s, ap.UNKNOWN_GENOTYPE) in genotypes]
        with right:
            if len(chosen) < 2:
                st.warning(f"Registration needs the {region} outline of at least two samples.")
            else:
                shapes = {s: sa.largest_part(outlines[(s, region)]) for s in chosen}
                several = [s for s in chosen if len(outlines[(s, region)]) > 1]
                cell_points = {}
                cell_labels = {}
                if not cells_all.empty:
                    inside = cells_all[cells_all["Region"] == region]
                    for s in chosen:
                        mine = inside[inside["Sample"] == s]
                        # Same flip as the outlines: plot y grows upwards.
                        cell_points[s] = np.column_stack(
                            [mine["Centroid X (µm)"].to_numpy(float),
                             -mine["Centroid Y (µm)"].to_numpy(float)]
                        )
                        cell_labels[s] = mine["Label"].astype(int).to_numpy()
                try:
                    atlas, template, registered, fits, mapped = register_region(
                        shapes, cell_points, fit.startswith("Shape"), reflect, warp, smoothing
                    )
                except (ValueError, np.linalg.LinAlgError) as exc:
                    st.error(f"Could not register the {region} outlines: {exc}")
                    st.stop()
                if several:
                    st.caption(f"{', '.join(several)}: {region} has several parts; the largest is used.")

                order = [g for g in ap.genotype_order([genotype_of.get(s) for s in chosen])]
                styles = ap.genotype_styles(order)
                by_genotype = {g: [s for s in chosen if genotype_of.get(s) == g] for g in order}

                # Registered outlines, their genotype means and the template.
                figure = go.Figure()
                for g in order:
                    color = styles[g][0]
                    for k, s in enumerate(by_genotype[g]):
                        line = closed(registered[s])
                        figure.add_trace(go.Scatter(
                            x=line[:, 0], y=line[:, 1], mode="lines", name=s,
                            legendgroup=g, legendgrouptitle_text=g if k == 0 else None,
                            line={"color": _alpha(color, 0.45), "width": 1.5},
                            hovertemplate=f"{s}<extra>{g}</extra>",
                        ))
                    mean = np.mean([registered[s] for s in by_genotype[g]], axis=0)
                    line = closed(mean)
                    figure.add_trace(go.Scatter(
                        x=line[:, 0], y=line[:, 1], mode="lines", name=f"{g} mean",
                        legendgroup=g, line={"color": color, "width": 3},
                        hovertemplate=f"mean of {len(by_genotype[g])} {g}<extra></extra>",
                    ))
                line = closed(template)
                figure.add_trace(go.Scatter(
                    x=line[:, 0], y=line[:, 1], mode="lines", name="Template (all samples)",
                    line={"color": TEMPLATE_COLOR, "width": 3, "dash": "dash"},
                    hovertemplate="template<extra></extra>",
                ))
                figure.update_layout(
                    title=f"{region} of {len(chosen)} samples, registered", height=620,
                    margin={"l": 10, "r": 10, "t": 50, "b": 10}, legend={"groupclick": "toggleitem"},
                )
                scale_bar(figure, template)
                equal_axes(figure, 1)
                shape_col, fit_col = st.columns([3, 2])
                with shape_col:
                    st.plotly_chart(figure, width="stretch", theme="streamlit")
                with fit_col:
                    fit_table = pd.DataFrame(
                        [{"Sample": s, "Genotype": genotype_of.get(s),
                          "Cells": len(cell_points.get(s, ())), **fits[s]} for s in chosen]
                    )
                    st.dataframe(
                        fit_table.style.format({"Scale": "{:.3f}", "Rotation (°)": "{:.1f}",
                                                "Residual (µm)": "{:.1f}"}),
                        hide_index=True, width="stretch",
                    )
                    st.caption(
                        "Residual: RMS distance of each registered outline from the "
                        "template before bending — a large one is a sample whose outline "
                        "differs in shape, or was drawn differently."
                    )

                if not any(len(p) for p in mapped.values()):
                    st.info(f"No cells were detected in {region} in these samples.")
                else:
                    def group_of(sample: str, label: int) -> str:
                        if atlas_color == "Cluster":
                            return cluster_of.get((sample, int(label)), NOT_CLUSTERED)
                        if atlas_color == "Genotype":
                            return genotype_of.get(sample, ap.UNKNOWN_GENOTYPE)
                        return PLAIN

                    groups = {s: np.array([group_of(s, lab) for lab in cell_labels.get(s, ())], dtype=object)
                              for s in mapped}
                    if atlas_color == "Cluster":
                        overlay_colors = cluster_colors(clusters_found + [UNCLUSTERED, NOT_CLUSTERED])
                    else:
                        overlay_colors = {g: c for g, (c, _m) in styles.items()}

                    # Each cell's own outline, carried in with its sample's
                    # transform: all of a sample's outlines in one go.
                    cell_polys: dict[tuple[str, int], np.ndarray] = {}
                    if draw == "Outlines" and cell_outlines:
                        for s in mapped:
                            labels_here = [int(lab) for lab in cell_labels.get(s, ())
                                           if cell_outlines.get((s, int(lab)))]
                            if not labels_here:
                                continue
                            polys = [cell_outlines[(s, lab)][0] for lab in labels_here]
                            moved = atlas.map_points(s, np.vstack(polys))
                            bounds = np.cumsum([0] + [len(p) for p in polys])
                            for lab, a, b in zip(labels_here, bounds[:-1], bounds[1:]):
                                cell_polys[(s, lab)] = moved[a:b]

                    def overlay_for(members) -> dict:
                        out: dict = {}
                        for s in members:
                            for point, lab, g in zip(mapped.get(s, ()), cell_labels.get(s, ()),
                                                     groups.get(s, ())):
                                entry = out.setdefault(g, {"points": [], "outlines": []})
                                entry["points"].append(point)
                                poly = cell_polys.get((s, int(lab)))
                                if poly is not None:
                                    entry["outlines"].append(poly)
                        order_here = (cluster_order(out) if atlas_color == "Cluster" else list(out))
                        return {g: out[g] for g in order_here}

                    ramp = (DENSITY_SCALE_GREY if draw != "Hidden" and atlas_color != "Nothing"
                            else DENSITY_SCALE)
                    heat_points = mapped
                    if heat_of != "All cells":
                        heat_points = {
                            s: p[np.array([cluster_of.get((s, int(lab))) == heat_of
                                           for lab in cell_labels.get(s, ())], dtype=bool)]
                            if len(p) else p
                            for s, p in mapped.items()
                        }
                    grid = sa.density_maps(template, heat_points, sigma=sigma, units=units)
                    unit = "cells / 1000 µm²" if units == sa.PER_AREA else "% / 1000 µm²"
                    panels = {f"{g} (n={len(by_genotype[g])})": grid.mean_of(by_genotype[g])
                              for g in order}
                    top = max((float(np.nanmax(v)) for v in grid.maps.values() if np.isfinite(v).any()),
                              default=0.0)
                    top_mean = max((float(np.nanmax(v)) for v in panels.values() if np.isfinite(v).any()),
                                   default=0.0)
                    overlay = None
                    if draw != "Hidden":
                        overlay = {f"{g} (n={len(by_genotype[g])})": overlay_for(by_genotype[g])
                                   for g in order}
                    st.plotly_chart(
                        density_figure(grid, panels, template, colorscale=ramp,
                                       zmin=0.0, zmax=top_mean or 1.0, unit=unit, cells=overlay,
                                       colors=overlay_colors),
                        width="stretch", theme="streamlit",
                    )
                    st.caption(
                        (f"Heatmap of cluster {heat_of} only. " if heat_of != "All cells" else "")
                        + f"Mean over the samples of each genotype, each sample weighed the "
                        f"same; smoothed over {sigma:g} µm. Densities are per µm² of the "
                        f"template" + (", which the bending stretches or squeezes a little "
                                       "from each sample's own area." if warp else ".")
                    )

                    if len(order) >= 2:
                        pair = st.columns(2)
                        base = pair[0].selectbox("Compare", order, index=0, key="atlas-base")
                        other = pair[1].selectbox("with", [g for g in order if g != base],
                                                  index=0, key="atlas-other")
                        difference = grid.mean_of(by_genotype[other]) - grid.mean_of(by_genotype[base])
                        span = float(np.nanmax(np.abs(difference))) if np.isfinite(difference).any() else 1.0
                        st.plotly_chart(
                            density_figure(grid, {f"{other} − {base}": difference}, template,
                                           colorscale=DIFFERENCE_SCALE, zmin=-span or -1.0,
                                           zmax=span or 1.0, unit=unit),
                            width="stretch", theme="streamlit",
                        )
                        st.caption(f"Red: more cells in {other} than in {base} there; "
                                   f"blue: fewer. With few samples per genotype, read this "
                                   f"next to the per-sample maps below.")

                    with st.expander("Every sample"):
                        per_sample = {f"{s} ({genotype_of.get(s)})": grid.maps[s]
                                      for g in order for s in by_genotype[g] if s in grid.maps}
                        sample_cells = None
                        if draw != "Hidden":
                            sample_cells = {f"{s} ({genotype_of.get(s)})": overlay_for([s])
                                            for g in order for s in by_genotype[g] if s in mapped}
                        st.plotly_chart(
                            density_figure(grid, per_sample, template, colorscale=ramp,
                                           zmin=0.0, zmax=top or 1.0, unit=unit, cells=sample_cells,
                                           colors=overlay_colors, height=480,
                                           columns=min(4, len(per_sample))),
                            width="stretch", theme="streamlit",
                        )

                    mapped_rows = pd.concat(
                        [pd.DataFrame({"Sample": s, "Genotype": genotype_of.get(s),
                                       "Label": cell_labels[s],
                                       **({"Cluster": [cluster_of.get((s, int(lab)), NOT_CLUSTERED)
                                                       for lab in cell_labels[s]]}
                                          if clusters_found else {}),
                                       "Template X (µm)": p[:, 0], "Template Y (µm)": -p[:, 1]})
                         for s, p in mapped.items() if len(p)],
                        ignore_index=True,
                    )
                    downloads = st.columns(2)
                    downloads[0].download_button(
                        "Cells in template space (CSV)", mapped_rows.to_csv(index=False),
                        file_name=f"{region}_cells_in_template.csv",
                    )
                    downloads[1].download_button(
                        "Template outline (CSV)",
                        pd.DataFrame({"X (µm)": template[:, 0], "Y (µm)": -template[:, 1]}).to_csv(index=False),
                        file_name=f"{region}_template.csv",
                    )

# -- explain ------------------------------------------------------------------

with explain_tab:
    try:
        import shap  # noqa: F401
    except ImportError:
        st.info("The Explain tab needs the `shap` package: `pip install shap`.")
    else:
        if cells_all.empty:
            st.info("This workbook has no “Objects” sheet — no cells to explain.")
        else:
            left, right = st.columns([1, 3])
            with left:
                st.subheader("Explain")
                targets = (["Cluster"] if cluster_of else []) + ["Genotype"]
                target_name = st.radio("What tells the cells apart", targets, key="shap-target",
                                       help="Clusters come from the Cells tab.")
                if not cluster_of:
                    st.caption("Turn clustering on in the Cells tab to explain the clusters.")
                explain_regions = list(dict.fromkeys(cells_all["Region"]))
                if target_name == "Cluster":
                    explain_region = st.session_state.get("clu-region")
                    st.caption(f"The clustered cells: {explain_region}.")
                else:
                    wanted = st.session_state.get("atlas-region", "CB")
                    explain_region = st.selectbox(
                        "Cells of", explain_regions,
                        index=explain_regions.index(wanted) if wanted in explain_regions else 0,
                        key="shap-region",
                    )
                pool = cells_all[
                    (cells_all["Region"] == explain_region)
                    & cells_all["Genotype"].isin(genotypes)
                    & cells_all["Sample"].isin(samples)
                ].reset_index(drop=True)
                if target_name == "Cluster":
                    pool = pool.assign(Cluster=[
                        cluster_of.get((sm, int(lab)))
                        for sm, lab in zip(pool["Sample"], pool["Label"])
                    ])
                    pool = pool[pool["Cluster"].notna() & (pool["Cluster"] != UNCLUSTERED)]
                    pool = pool.reset_index(drop=True)
                explain_features: list[str] = []
                for name, columns in cell_groups(pool).items():
                    if columns and st.checkbox(f"{name} ({len(columns)})",
                                               value=name != "Position", key=f"shap-{name}"):
                        explain_features.extend(columns)
                n_explain = st.slider("Cells to explain", 100, 3000, 1000, 100, key="shap-n",
                                      help="SHAP values are computed for a random subset; the "
                                           "model is trained on every cell.")
                top_n = st.slider("Features shown", 5, 30, 12, key="shap-top")

            with right:
                labels = pool[target_name].astype(str) if len(pool) else pd.Series(dtype=str)
                classes_present = labels.unique()
                if len(pool) < 20 or len(classes_present) < 2 or not explain_features:
                    st.warning("Needs at least 20 cells of two or more classes, and a feature group.")
                else:
                    data = pool[explain_features].astype(float)
                    data = data.loc[:, data.notna().any() & (data.std() > 0)]
                    used = list(data.columns)
                    data = data.fillna(data.median())
                    groups = pool["Sample"].to_numpy() if target_name == "Genotype" else None
                    result = None
                    try:
                        result = explain_cells(
                            data.to_numpy(), labels.to_numpy(), groups, int(min(n_explain, len(pool))), 0
                        )
                    except ValueError as exc:
                        st.error(f"Could not train the model: {exc}")
                    if result is not None:
                        score, explained, values, classes = result

                        if target_name == "Cluster":
                            order = cluster_order(classes)
                            colors = cluster_colors(order)
                        else:
                            order = ap.genotype_order(classes)
                            colors = {g: c for g, (c, _m) in ap.genotype_styles(order).items()}
                        index = {c: i for i, c in enumerate(classes)}
                        chance = 1.0 / len(classes)
                        st.metric(
                            "Balanced accuracy, cross-validated",
                            "—" if not np.isfinite(score) else f"{score:.0%}",
                            None if not np.isfinite(score) else f"{score - chance:+.0%} vs chance ({chance:.0%})",
                        )
                        st.caption(
                            ("Whole samples held out in turn: the model is scored on fish it has "
                             "never seen. " if groups is not None else
                             "Clusters were found from these same features, so a high score is "
                             "expected; the point is which features the boundaries use. ")
                            + f"{len(pool):,} {explain_region} cells, {len(used)} features."
                        )
                        if groups is not None and pool["Sample"].nunique() < 6:
                            st.caption(f"Only {pool['Sample'].nunique()} samples: read the "
                                       "features as leads, not findings.")

                        # Global importance: mean |SHAP| per feature, per class.
                        importance = np.abs(values).mean(axis=0)  # features × classes
                        binary = len(classes) == 2
                        total = importance[:, 0] if binary else importance.sum(axis=1)
                        top = np.argsort(total)[::-1][: int(top_n)]
                        figure = go.Figure()
                        if binary:
                            figure.add_trace(go.Bar(
                                y=[used[i] for i in top], x=total[top], orientation="h",
                                marker={"color": "#2a78d6"}, name="mean |SHAP|",
                                hovertemplate="%{y}: %{x:.3g}<extra></extra>",
                            ))
                        else:
                            for c in order:
                                figure.add_trace(go.Bar(
                                    y=[used[i] for i in top], x=importance[top, index[c]],
                                    orientation="h", name=str(c),
                                    marker={"color": colors.get(c, NEUTRAL),
                                            "line": {"color": "white", "width": 1}},
                                    hovertemplate=f"{c} · %{{y}}: %{{x:.3g}}<extra></extra>",
                                ))
                        figure.update_layout(
                            barmode="stack", title="Which features the model leans on",
                            xaxis_title="mean |SHAP value| (change in predicted probability)",
                            yaxis={"autorange": "reversed", "automargin": True},
                            xaxis={"automargin": True}, legend={"traceorder": "normal"},
                            height=max(320, 28 * len(top) + 120),
                            margin={"l": 10, "r": 10, "t": 50, "b": 10},
                        )
                        st.plotly_chart(figure, width="stretch", theme="streamlit")

                        # Beeswarm for one class: each explained cell, by feature.
                        if binary:
                            focus = order[-1]
                            st.caption(f"Positive SHAP values push a cell towards {focus}, "
                                       f"negative towards {order[0]}.")
                        else:
                            focus = st.selectbox("Class", order, key="shap-class")
                        shown = values[:, :, index[focus]]
                        subset = data.iloc[explained]
                        rng = np.random.default_rng(0)
                        bees = go.Figure()
                        for rank, i in enumerate(top):
                            feature = subset.iloc[:, i].to_numpy()
                            spread = np.ptp(feature)
                            tone = (feature - feature.min()) / spread if spread > 0 else np.zeros_like(feature)
                            bees.add_trace(go.Scatter(
                                x=shown[:, i], y=rank + rng.uniform(-0.3, 0.3, len(feature)),
                                mode="markers", showlegend=False,
                                marker={"size": 5, "color": tone, "colorscale": VALUE_SCALE,
                                        "cmin": 0, "cmax": 1, "opacity": 0.8,
                                        "colorbar": {"title": {"text": "feature value"},
                                                     "tickvals": [0, 1], "ticktext": ["low", "high"]}
                                        if rank == 0 else None,
                                        "showscale": rank == 0},
                                customdata=feature,
                                hovertemplate=f"{used[i]} = %{{customdata:.4g}}<br>SHAP %{{x:.3g}}<extra></extra>",
                            ))
                        bees.add_vline(x=0, line={"color": NEUTRAL, "width": 1})
                        bees.update_layout(
                            title=f"How each feature moves cells towards {focus}",
                            xaxis_title=f"SHAP value for {focus}",
                            yaxis={"tickvals": list(range(len(top))), "ticktext": [used[i] for i in top],
                                   "autorange": "reversed", "showgrid": False, "zeroline": False,
                                   "automargin": True},
                            xaxis={"automargin": True},
                            height=max(320, 30 * len(top) + 120),
                            margin={"l": 10, "r": 10, "t": 50, "b": 10},
                        )
                        st.plotly_chart(bees, width="stretch", theme="streamlit")

                        # Dependence: one feature's value against its SHAP value.
                        feature_name = st.selectbox("Feature", [used[i] for i in top], key="shap-feature")
                        fi = used.index(feature_name)
                        true_class = labels.iloc[explained].to_numpy()
                        dep = go.Figure()
                        for c in order:
                            mask = true_class == c
                            if not mask.any():
                                continue
                            dep.add_trace(go.Scatter(
                                x=subset.iloc[:, fi].to_numpy()[mask], y=shown[mask, fi],
                                mode="markers", name=str(c),
                                marker={"size": 7, "color": colors.get(c, NEUTRAL), "opacity": 0.75,
                                        "line": {"width": 1, "color": "white"}},
                                hovertemplate=f"{c}<br>{feature_name} %{{x:.4g}}<br>SHAP %{{y:.3g}}<extra></extra>",
                            ))
                        dep.add_hline(y=0, line={"color": NEUTRAL, "width": 1})
                        dep.update_layout(
                            title=f"{feature_name}: value against its push towards {focus}",
                            xaxis_title=feature_name, yaxis_title=f"SHAP value for {focus}",
                            legend_title_text=f"Actual {target_name.lower()}", height=480,
                            xaxis={"automargin": True}, yaxis={"automargin": True},
                            margin={"l": 10, "r": 10, "t": 50, "b": 10},
                        )
                        st.plotly_chart(dep, width="stretch", theme="streamlit")

                        table = pd.DataFrame(importance, index=used, columns=[str(c) for c in classes])
                        table = table[[str(c) for c in order]].assign(**{"Total": total})
                        st.download_button(
                            "Mean |SHAP| per feature (CSV)",
                            table.sort_values("Total", ascending=False).to_csv(),
                            file_name=f"shap_{target_name.lower()}_{explain_region}.csv",
                        )

# -- plots --------------------------------------------------------------------

with plots_tab:
    plot_cells = read_cells(source)
    left, right = st.columns([1, 3])
    with left:
        st.subheader("Plot")
        measures = ["Regions"]
        if not plot_cells.empty:
            measures.append("Cells")
        if cluster_of:
            measures.append("Cell clusters")
        measure = st.radio(
            "Measure", measures, key="plot-measure",
            help="Regions: one row per sample and region, from the Region features sheet. "
                 "Cells: the segmented cells. Cell clusters: how much of each sample "
                 "belongs to one cluster (clustering is set in the Cells tab).",
        )
        frame = pd.DataFrame()
        dot_label = "Sample"
        note = ""
        per_cell = False
        if measure == "Regions":
            region_list = list(dict.fromkeys(rows["Region"]))
            region = st.selectbox("Region", region_list, key="plot-region")
            frame = rows[rows["Region"] == region]
            numbers = plottable(frame, IDENTITY)
            note = f"One dot per sample, {region}."
        elif measure == "Cells":
            pool = plot_cells[
                plot_cells["Genotype"].isin(genotypes) & plot_cells["Sample"].isin(samples)
            ]
            region_list = ["(every region)"] + list(dict.fromkeys(pool["Region"]))
            region = st.selectbox("Region", region_list, key="plot-cell-region")
            if region != "(every region)":
                pool = pool[pool["Region"] == region]
            if cluster_of:
                labelled = pool.assign(Cluster=[
                    cluster_of.get((s, int(lab)), NOT_CLUSTERED)
                    for s, lab in zip(pool["Sample"], pool["Label"])
                ])
                found = cluster_order(set(labelled["Cluster"]) - {NOT_CLUSTERED})
                if found:
                    which = st.selectbox("Cluster", ["(all cells)", *found], key="plot-cluster")
                    if which != "(all cells)":
                        pool = labelled[labelled["Cluster"] == which]
            numbers = plottable(pool, CELL_IDENTITY)
            level = st.radio(
                "One dot per", ("Sample (mean of its cells)", "Cell"), key="plot-level",
                help="What a dot is. Either way the test never treats the cells of one "
                     "fish as independent measurements: showing every cell switches the "
                     "test to a mixed model with the sample as a random effect, or you "
                     "can summarise each sample first.",
            )
            per_cell = level == "Cell"
            frame = pool
            note = f"{len(pool):,} cells" + ("" if region == "(every region)" else f" in {region}")
        else:
            clustered_region = st.session_state.get("clu-region", "")
            by_sample = plot_cells[
                plot_cells["Genotype"].isin(genotypes) & plot_cells["Sample"].isin(samples)
            ].assign(Cluster=lambda f: [cluster_of.get((s, int(lab)), NOT_CLUSTERED)
                                        for s, lab in zip(f["Sample"], f["Label"])])
            by_sample = by_sample[by_sample["Cluster"] != NOT_CLUSTERED]
            found = cluster_order(by_sample["Cluster"])
            which = st.selectbox("Cluster", found, key="plot-cluster-only")
            as_share = st.radio("As", ("Share of the sample's cells (%)", "Cells"),
                                key="plot-cluster-as") .startswith("Share")
            counts = pd.crosstab(by_sample["Sample"], by_sample["Cluster"])
            values = (counts.div(counts.sum(axis=1), axis=0) * 100) if as_share else counts
            column = f"{which}: {'% of the sample' if as_share else 'cells'}"
            frame = pd.DataFrame({
                "Sample": values.index,
                "Genotype": [genotype_of_sample(features, s) for s in values.index],
                column: values[which].to_numpy() if which in values else np.nan,
            })
            numbers = [column]
            note = f"One dot per sample, {clustered_region} cells."

        variable = (st.selectbox("Variable", numbers, index=default_variable(numbers),
                                 key="plot-variable") if numbers else None)
        if variable is not None and len(numbers) > 1:
            by = st.selectbox("Normalize to", normalise_choices(numbers, variable),
                              key="plot-normalise",
                              help=f"Divide the variable by another, sample by sample. "
                                   f"{WB_AREA} is the whole brain's area.")
            frame, variable = normalise(frame, variable, by)
        choices = group_choices(features)
        group_by = (st.selectbox("Group by", choices, key="plot-group-by",
                                 help="What the boxes are. Conditions come from the "
                                      "Groups box of the viewer's Analysis panel; "
                                      "“Genotype × …” puts the genotypes side by side "
                                      "inside each condition.")
                    if len(choices) > 1 else "Genotype")
        frame, group = with_group(frame, features, group_by)
        kind = st.radio("Shape", ("Box", "Violin"), horizontal=True, key="plot-kind")
        if per_cell:
            test = st.selectbox(
                "Test", CELL_TESTS, key="plot-cell-test",
                help="Cells of one fish are not independent, so a plain test over cells "
                     "is not offered.\n\n"
                     "- **Mixed model**: every cell is used, with a random intercept per "
                     "sample, so fish that differ overall do not count as many independent "
                     "measurements. Genotype is tested by likelihood ratio.\n"
                     "- **Sample means**: each sample becomes one number first, and the "
                     "usual test runs on those — fewer assumptions, and what most papers "
                     "report.",
            )
        else:
            test = st.selectbox(
                "Test", TESTS, key="plot-test",
                help="One-way ANOVA compares the group means, assuming similar spreads and "
                     "roughly normal residuals. Welch's drops the equal-spread assumption. "
                     "Kruskal-Wallis compares ranks and assumes neither — the safe choice for "
                     "small or skewed groups. With two groups an ANOVA is a t-test.",
            )
        show_dots = st.checkbox("Show the dots", value=True, key="plot-dots")
        bars = st.radio("Significance bars", ("Significant only", "All pairs", "Off"),
                        key="plot-bars",
                        help="A bar joins two groups and carries the p value of that "
                             "comparison: the post-hoc pair when there are three groups or "
                             "more, the test itself when there are two.")
        bar_label = st.radio("Bars say", ("Stars", "p value"), horizontal=True, key="plot-bar-label",
                             help="Stars: * < 0.05, ** < 0.01, *** < 0.001; ns otherwise.")

    with right:
        if not numbers or variable is None:
            st.warning("Nothing numeric to plot here.")
        else:
            data = frame.copy()
            if measure == "Cells" and not per_cell:
                data = (data.groupby(["Sample", group], as_index=False)[variable]
                        .mean(numeric_only=True))
            data = data.dropna(subset=[variable])
            # What the test sees: every cell for the mixed model, one number per
            # sample otherwise, never the cells as independent measurements.
            tested = data
            if per_cell and test != MIXED:
                tested = (data.groupby(["Sample", group], as_index=False)[variable]
                          .mean(numeric_only=True))
            dot_label = "Sample"
            if measure == "Cells" and per_cell:
                dot_label = "Sample"
            order = group_order(data[group])
            values = {g: tested.loc[tested[group] == g, variable].to_numpy(dtype=float)
                      for g in order}
            if len(order) < 2 or data.empty:
                st.warning("Needs at least two groups with values.")
            else:
                if per_cell and test == MIXED:
                    result = mixed_model_test(tested, variable, group)
                    pairs = mixed_pairs(tested, variable, group) if len(order) > 2 else pd.DataFrame()
                else:
                    plain = test.split(" · ")[-1]
                    grouped = {g: tested.loc[tested[group] == g, variable].to_numpy(dtype=float)
                               for g in order}
                    result = compare_groups(grouped, plain)
                    pairs = posthoc_pairs(grouped, plain) if len(order) > 2 else pd.DataFrame()
                if len(order) == 2 and np.isfinite(result["p"]):
                    # Two groups: the test itself is the only comparison there is.
                    pairs = pd.DataFrame([{"Group 1": order[0], "Group 2": order[1],
                                           "p": result["p"]}])
                title = f"{variable} — {note}"
                figure = comparison_figure(data, variable, group, kind, title,
                                           dot_label=dot_label, pairs=pairs, show_dots=show_dots,
                                           bars=bars, bar_label=bar_label)
                st.plotly_chart(figure, width="stretch", theme="streamlit")

                statistic = {"Kruskal-Wallis": "H", MIXED: "χ²"}.get(
                    test.split(" · ")[-1] if " · " in test else test, "F")
                columns = st.columns(3)
                short = "Mixed model" if test == MIXED else test.split(" · ")[-1]
                columns[0].metric(f"{statistic} · {short}",
                                  "—" if not np.isfinite(result["statistic"]) else f"{result['statistic']:.3g}")
                columns[1].metric("p", "—" if not np.isfinite(result["p"]) else f"{result['p']:.3g}",
                                  stars(result["p"]) or None)
                columns[2].metric(result["effect_name"],
                                  "—" if not np.isfinite(result["effect"]) else f"{result['effect']:.3f}",
                                  help="Share of the variation that lies between the groups.")
                if per_cell and test != MIXED:
                    st.caption(f"The dots are cells; the test ran on the {len(tested)} "
                               "sample means behind them.")
                for line in result["notes"]:
                    st.caption(line)

                table = result.get("table")
                if table is None:
                    table = anova_table(values, test.split(" · ")[-1], result)
                if table is not None and not table.empty:
                    numeric = [c for c in table.columns if c not in ("Source", "Term", "df")]
                    st.caption("The mixed model, term by term. “Estimate” is the shift from "
                               "the first group, in the variable's units."
                               if test == MIXED else f"The {short} table.")
                    st.dataframe(
                        table.style.format({c: "{:.4g}" for c in numeric}, na_rep=""),
                        hide_index=True, width="stretch",
                    )

                summary = pd.DataFrame([{
                    group: g,
                    "Samples": int(tested.loc[tested[group] == g, "Sample"].nunique()),
                    "n": len(values[g]),
                    "Mean": float(np.mean(values[g])) if len(values[g]) else np.nan,
                    "SD": float(np.std(values[g], ddof=1)) if len(values[g]) > 1 else np.nan,
                    "Median": float(np.median(values[g])) if len(values[g]) else np.nan,
                } for g in order])
                st.dataframe(summary.style.format({"Mean": "{:.4g}", "SD": "{:.4g}",
                                                   "Median": "{:.4g}"}),
                             hide_index=True, width="stretch")
                if not pairs.empty and len(order) > 2:
                    marked = pairs.assign(**{"": [stars(float(p)) for p in pairs["p"]]})
                    method = ("Holm on the mixed model's contrasts." if test == MIXED
                              else "Holm on Mann-Whitney." if short == "Kruskal-Wallis"
                              else "Tukey HSD.")
                    st.caption("Every pair, corrected for the number of comparisons — " + method)
                    st.dataframe(marked.style.format({"p": "{:.3g}", "Difference": "{:.4g}"}),
                                 hide_index=True, width="stretch")
                st.download_button("These values (CSV)", data.to_csv(index=False),
                                   file_name=f"{variable}_by_genotype.csv".replace("/", "-"))

# -- table --------------------------------------------------------------------

with table_tab:
    st.dataframe(rows, width="stretch", hide_index=True)
    st.download_button("Download these rows (CSV)", rows.to_csv(index=False),
                       file_name="region_features_filtered.csv")
