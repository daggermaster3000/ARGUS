"""The spatial dashboard page. Run it with ``streamlit run dashboard_app.py``.

Everything it computes lives in :mod:`microscopy_viewer.dashboard`; this file is
the page — widgets, layout, and the plots. Started from the viewer by the
**Spatial dashboard** button in the Measurement analysis panel, or by hand::

    streamlit run microscopy_viewer/dashboard_app.py -- --file G_07_0.h5ad

The order of the page is the order of the argument: what is in the file, what the
objects look like, how they are grouped, and only then where those groups sit and
whether that arrangement means anything. The spatial statistics are last because
they are the slow ones and because they are meaningless until the grouping above
them is one the user believes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import streamlit as st

# Run by streamlit, this file is __main__ in its own directory rather than part of
# the package, so the package has to be findable before it can be imported.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from microscopy_viewer import dashboard as db  # noqa: E402

st.set_page_config(page_title="Spatial dashboard", layout="wide")

PLOT_HEIGHT = 5.0


def _argument_file() -> str:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--file", default="")
    known, _rest = parser.parse_known_args()
    return known.file


@st.cache_resource(show_spinner="Reading the file…")
def _read(path: str, mtime: float):
    """Cached on the path *and* its modification time, so a re-run reloads it."""
    return db.read(path)


@st.cache_resource(show_spinner="Grouping the objects…")
def _analysed(
    path: str,
    mtime: float,
    image: str,
    limit: int,
    mode: str,
    n_neighs: int,
    radius: float,
    grouping: str,
    resolution: float,
    bins: int,
):
    """One image, grouped and joined up. Cached because everything below reuses it."""
    adata = db.select_image(_read(path, mtime), image or None)
    adata = db.subsample(adata, limit)
    if grouping == "cluster":
        method = db.cluster(adata, resolution=resolution)
    else:
        db.prepare(adata)
        method = db.bin_column(adata, grouping, bins=bins)
    graph = db.build_graph(adata, mode=mode, n_neighs=n_neighs, radius_um=radius)
    return adata, method, graph


def _figure(width: float = 6.0, height: float = PLOT_HEIGHT):
    from matplotlib.figure import Figure

    figure = Figure(figsize=(width, height), dpi=110)
    figure.patch.set_alpha(0.0)
    return figure


def main() -> None:
    st.title("Spatial dashboard")

    default = _argument_file()
    with st.sidebar:
        st.header("File")
        path = st.text_input(
            "AnnData (.h5ad)",
            value=default,
            help="Written by the Measurement analysis panel, or by a batch run.",
        )
        uploaded = st.file_uploader("…or drop one here", type=["h5ad"])
        if uploaded is not None:
            target = Path(st.session_state.get("_upload_dir", ".")) / uploaded.name
            target.write_bytes(uploaded.getbuffer())
            path = str(target)

    if not path or not Path(path).exists():
        st.info(
            "Point this at an `.h5ad` written by the viewer — the Measurement analysis "
            "panel's **Export as AnnData**, or a batch run with the AnnData option ticked."
        )
        st.stop()

    mtime = Path(path).stat().st_mtime
    adata = _read(path, mtime)
    facts = db.describe(adata)

    columns = st.columns(4)
    columns[0].metric("Objects", f"{facts['objects']:,}")
    columns[1].metric("Features", facts["features"])
    columns[2].metric("Images", facts["images"] or 1)
    columns[3].metric("Spatial", f"{facts['spatial']}D" if facts["spatial"] else "none")

    if not facts["spatial"]:
        st.error(
            "This file has no `obsm['spatial']`, so there is nothing spatial to do with it. "
            "Re-export it from the viewer, which writes the centroids."
        )
        st.stop()

    # -- what to look at ------------------------------------------------------
    with st.sidebar:
        st.header("Image")
        available = db.images(adata)
        image = ""
        if available:
            image = st.selectbox(
                "Well and cycle",
                available,
                help=(
                    "One image at a time. The coordinates are positions within a well, so "
                    "two wells overlap in that space and a graph across the plate would "
                    "join objects that are merely in the same corner of different wells."
                ),
            )
        else:
            st.caption("One image in this file.")

        limit = st.slider(
            "Objects at most", 2000, 60000, db.DEFAULT_MAX_OBJECTS, step=2000,
            help="Subsampled at random above this. A permutation test on fifty thousand "
                 "objects is minutes; on eight thousand it is seconds and the same shape.",
        )

        st.header("Grouping")
        grouping_choice = st.radio(
            "Objects are grouped by",
            ["Clustering on the features", "Bins of one measurement"],
            help="The neighbourhood statistics need a label per object. A cluster finds "
                 "phenotypes; bins of one column are cruder and far easier to explain.",
        )
        resolution, bins, grouping = 1.0, 4, "cluster"
        if grouping_choice.startswith("Clustering"):
            resolution = st.slider("Resolution", 0.2, 2.0, 1.0, step=0.1)
        else:
            grouping = st.selectbox(
                "Measurement", db.feature_names(adata) + db.obs_columns(adata, numeric_only=True)
            )
            bins = st.slider("Bins", 2, 8, 4)

        st.header("Neighbours")
        mode = st.selectbox(
            "Graph", db.GRAPH_MODES,
            help="Delaunay joins each object to the ones it actually abuts, which is what "
                 "segmented nuclei are. A fixed radius misses them in a sparse field and "
                 "joins half the well in a dense one.",
        )
        n_neighs = st.slider("Neighbours (knn)", 3, 20, 6) if mode == "knn" else 6
        radius = st.slider("Radius (um)", 5.0, 200.0, 30.0) if mode == "radius" else 30.0

    try:
        adata, method, graph = _analysed(
            path, mtime, image, limit, mode, n_neighs, radius, grouping, resolution, bins
        )
    except Exception as exc:  # noqa: BLE001 - shown on the page, not in a terminal
        st.exception(exc)
        st.stop()

    groups = list(adata.obs[db.CLUSTER_KEY].cat.categories)
    st.caption(
        f"**{adata.n_obs:,} objects**"
        + (f" from {image}" if image else "")
        + f" · grouped by {method} into {len(groups)} · joined by {graph}"
    )

    look, phenotype, space = st.tabs(["Where they are", "What they are", "Does it mean anything"])

    # -- where ----------------------------------------------------------------
    with look:
        left, right = st.columns([3, 2])
        with left:
            st.subheader("The well")
            colour = st.selectbox(
                "Colour by",
                [db.CLUSTER_KEY] + db.feature_names(adata) + db.obs_columns(adata, numeric_only=True),
            )
            figure = _figure(7.0, 6.0)
            axes = figure.add_subplot(111)
            xy = np.asarray(adata.obsm["spatial"])
            if colour == db.CLUSTER_KEY:
                for group in groups:
                    mask = (adata.obs[db.CLUSTER_KEY] == group).to_numpy()
                    axes.scatter(xy[mask, 0], xy[mask, 1], s=4, linewidths=0, label=str(group))
                axes.legend(markerscale=3, fontsize=7, frameon=False, loc="upper right")
            else:
                values = db.values_of(adata, colour).astype(float)
                low, high = np.nanpercentile(values, [1, 99])
                dots = axes.scatter(
                    xy[:, 0], xy[:, 1], c=values, s=4, linewidths=0,
                    cmap="viridis", vmin=low, vmax=high,
                )
                figure.colorbar(dots, ax=axes, shrink=0.75, label=colour)
            axes.set_xlabel("x (um)", fontsize=8)
            axes.set_ylabel("y (um)", fontsize=8)
            axes.set_aspect("equal")
            axes.invert_yaxis()  # image convention: y runs down
            axes.tick_params(labelsize=7)
            st.pyplot(figure)

        with right:
            st.subheader("How many of each")
            counts = adata.obs[db.CLUSTER_KEY].value_counts().sort_index()
            st.bar_chart(counts)
            st.dataframe(
                counts.rename("objects").to_frame().assign(
                    share=lambda frame: (frame["objects"] / frame["objects"].sum()).map("{:.1%}".format)
                )
            )

    # -- what -----------------------------------------------------------------
    with phenotype:
        st.subheader("What tells the groups apart")
        feature = st.selectbox("Measurement", db.feature_names(adata), key="phenotype_feature")
        left, right = st.columns(2)

        with left:
            figure = _figure()
            axes = figure.add_subplot(111)
            data = [db.values_of(adata, feature)[(adata.obs[db.CLUSTER_KEY] == g).to_numpy()]
                    for g in groups]
            # The ticks are set afterwards rather than passed in: matplotlib
            # renamed boxplot's `labels` to `tick_labels` in 3.9 and removed the
            # old spelling, and this has to draw on both sides of that.
            axes.boxplot(data, showfliers=False)
            axes.set_xticks(range(1, len(groups) + 1), [str(g) for g in groups], rotation=45)
            axes.set_ylabel(feature, fontsize=8)
            axes.tick_params(labelsize=7)
            st.pyplot(figure)
            st.caption(
                "Scaled values: the features were z-scored before clustering, so zero is "
                "the image's mean and the units are standard deviations."
            )

        with right:
            pair = st.selectbox(
                "against", [f for f in db.feature_names(adata) if f != feature], key="pair"
            )
            figure = _figure()
            axes = figure.add_subplot(111)
            x, y = db.values_of(adata, feature), db.values_of(adata, pair)
            for group in groups:
                mask = (adata.obs[db.CLUSTER_KEY] == group).to_numpy()
                axes.scatter(x[mask], y[mask], s=4, linewidths=0, alpha=0.5, label=str(group))
            axes.set_xlabel(feature, fontsize=8)
            axes.set_ylabel(pair, fontsize=8)
            axes.legend(markerscale=3, fontsize=7, frameon=False)
            axes.tick_params(labelsize=7)
            st.pyplot(figure)

    # -- does it mean anything ------------------------------------------------
    with space:
        st.caption(
            "These are the slow ones — a permutation test per plot. Each runs when you "
            "open it and is remembered afterwards."
        )

        with st.expander("Neighbourhood enrichment — which groups sit next to which", expanded=True):
            st.markdown(
                "The z-score of how often two groups are neighbours against a null where the "
                "labels are shuffled over the same graph. **Positive** means the two are found "
                "together more than chance; **negative** means they avoid each other. The "
                "diagonal is a group next to itself, which is the usual way a phenotype says "
                "it comes in patches."
            )
            perms = st.slider("Permutations", 50, 1000, 200, step=50)
            if st.button("Compute", key="nhood"):
                with st.spinner("Shuffling…"):
                    zscore, _count = db.neighbourhood_enrichment(adata, n_perms=perms)
                figure = _figure(6.0, 5.0)
                axes = figure.add_subplot(111)
                limit_z = float(np.nanmax(np.abs(zscore))) or 1.0
                image_plot = axes.imshow(zscore, cmap="RdBu_r", vmin=-limit_z, vmax=limit_z)
                axes.set_xticks(range(len(groups)), [str(g) for g in groups], rotation=45, fontsize=7)
                axes.set_yticks(range(len(groups)), [str(g) for g in groups], fontsize=7)
                figure.colorbar(image_plot, ax=axes, shrink=0.8, label="z-score")
                st.pyplot(figure)

        with st.expander("Ripley's L — clustered in space, or scattered?"):
            st.markdown(
                "How many neighbours each group has within a growing radius, against what "
                "complete randomness would give. **Above** the grey line is clustered, "
                "**below** is more evenly spread than chance."
            )
            if st.button("Compute", key="ripley"):
                with st.spinner("Counting…"):
                    result = db.ripley(adata, mode="L")
                figure = _figure(6.0, 4.5)
                axes = figure.add_subplot(111)
                table = result["L_stat"]
                for group in groups:
                    part = table[table[db.CLUSTER_KEY] == group]
                    axes.plot(part["bins"], part["stats"], label=str(group), linewidth=1.2)
                if "sims_stat" in result:
                    sims = result["sims_stat"]
                    axes.plot(sims["bins"], sims["stats"], color="0.6", linestyle="--",
                              linewidth=1.0, label="random")
                axes.set_xlabel("radius (um)", fontsize=8)
                axes.set_ylabel("Ripley's L", fontsize=8)
                axes.legend(fontsize=7, frameon=False)
                axes.tick_params(labelsize=7)
                st.pyplot(figure)

        with st.expander("Co-occurrence — how far the company of a group reaches"):
            st.markdown(
                "The chance of finding each group within a distance of another, divided by "
                "its overall share. Above 1 means enriched at that distance; where the curve "
                "falls back to 1 is how far the effect reaches."
            )
            if st.button("Compute", key="cooc"):
                with st.spinner("Walking outwards…"):
                    occ, interval = db.co_occurrence(adata)
                anchor = st.session_state.get("cooc_anchor", groups[0])
                index = groups.index(anchor) if anchor in groups else 0
                figure = _figure(6.0, 4.5)
                axes = figure.add_subplot(111)
                for position, group in enumerate(groups):
                    axes.plot(interval[:-1], occ[index, position, :], label=str(group), linewidth=1.2)
                axes.axhline(1.0, color="0.6", linestyle="--", linewidth=1.0)
                axes.set_xlabel("distance (um)", fontsize=8)
                axes.set_ylabel(f"p(group | {anchor}) / p(group)", fontsize=8)
                axes.legend(fontsize=7, frameon=False)
                axes.tick_params(labelsize=7)
                st.pyplot(figure)
                st.selectbox("Centred on", groups, key="cooc_anchor")

        with st.expander("Moran's I — which measurements vary in patches"):
            st.markdown(
                "Spatial autocorrelation per feature, needing no grouping at all. A high "
                "value says neighbouring objects resemble each other in that measurement — "
                "a reporter that comes in patches rather than cell by cell."
            )
            if st.button("Compute", key="moran"):
                with st.spinner("Correlating…"):
                    table = db.spatial_autocorrelation(adata)
                st.dataframe(table.head(25))
                figure = _figure(6.0, 4.5)
                axes = figure.add_subplot(111)
                top = table.head(15).iloc[::-1]
                axes.barh(range(len(top)), top["I"])
                axes.set_yticks(range(len(top)), list(top.index), fontsize=7)
                axes.set_xlabel("Moran's I", fontsize=8)
                axes.tick_params(labelsize=7)
                st.pyplot(figure)

        with st.expander("Centrality — how each group sits in the graph"):
            if st.button("Compute", key="centrality"):
                st.dataframe(db.centrality(adata))


if __name__ == "__main__":
    main()
