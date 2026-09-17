"""The spatial dashboard page. Run it with ``streamlit run dashboard_app.py``.

Everything it computes lives in :mod:`microscopy_viewer.dashboard`; this file is
the page — widgets, layout, and the plots. Started from the viewer by the
**Spatial dashboard** button in the Measurement analysis panel, or by hand::

    streamlit run microscopy_viewer/dashboard_app.py -- --file G_07_0.h5ad

The scatters are Altair rather than matplotlib, so a point can be hovered and say
which well and which cluster it belongs to — the question every one of these
plots provokes. Altair ships with Streamlit, so this costs no dependency. The
heatmaps and line plots stay matplotlib, which draws them better and which
nothing is gained by hovering.

Colour is decided in one place, :func:`microscopy_viewer.dashboard.colour_map`,
and every plot takes it from there: a cluster is the same colour in the UMAP, in
the well, in the composition bar and in the box plot.

**Well display** is the plate laid out: every well's spatial view side by side,
from one clustering so the colours mean the same thing in each. Computed once and
kept in the file, as points rather than as pictures — a picture cannot be pointed
at, and hovering is how a panel says which well it is.

There are two questions here, and they do not mix. **Across the plate** asks
which wells hold which phenotypes, in feature space, using every well at once.
The other three tabs ask where those phenotypes sit inside one well, which is
spatial and has to be one image at a time because the coordinates are per image.

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

import altair as alt
import numpy as np
import streamlit as st

# Run by streamlit, this file is __main__ in its own directory rather than part of
# the package, so the package has to be findable before it can be imported.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from microscopy_viewer import dashboard as db  # noqa: E402

st.set_page_config(page_title="Spatial dashboard", layout="wide")

# Streamlit reruns this file on every interaction, so this is called many times;
# configure_logging only attaches a handler once. The timings it prints are the
# only sign, in the console the shortcut opens, that a forty-second neighbour
# search is working rather than hung.
db.configure_logging()


@st.cache_resource(show_spinner=False)
def _warm_up_once() -> bool:
    """Compile the numba kernels in the background, while the page is being read.

    Streamlit reruns this file on every interaction; the cache is what makes this
    happen once per server rather than once per click. A daemon thread, so it
    never holds the process open.
    """
    import threading

    threading.Thread(target=db.warm_up, name="warm-up", daemon=True).start()
    return True


_warm_up_once()

PLOT_HEIGHT = 5.0

#: Points drawn in an interactive scatter. Vega holds the data in the page, so a
#: hoverable plot of half a million points is a browser tab that stops responding;
#: this is where hovering stays instant.
HOVER_POINT_LIMIT = 20_000

# Vega refuses more than five thousand rows unless told otherwise, which is well
# below a single well.
alt.data_transformers.disable_max_rows()


def _points(adata, x, y, colour: str, extra: list[str] | None = None):
    """A DataFrame of what to plot, with the columns the tooltip will show."""
    import pandas as pd

    frame = pd.DataFrame({"x": np.asarray(x, dtype=float), "y": np.asarray(y, dtype=float)})
    frame["object"] = list(adata.obs_names)
    for name in [colour] + list(extra or []):
        if name and name not in frame.columns:
            try:
                frame[name] = db.values_of(adata, name)
            except KeyError:
                continue
    return frame


def _scatter(
    frame,
    colour: str,
    mapping: dict | None,
    x_title: str,
    y_title: str,
    tooltips: list[str],
    height: int = 460,
    equal: bool = False,
    flip_y: bool = False,
    limits: tuple[float, float] | None = None,
):
    """An Altair scatter that says what a point is when you point at it.

    *mapping* fixes the colours to the ones every other plot on the page uses; a
    continuous column passes ``None`` and gets a viridis scale instead.
    """
    shown = frame
    note = ""
    if len(frame) > HOVER_POINT_LIMIT:
        shown = frame.sample(HOVER_POINT_LIMIT, random_state=0).sort_index()
        note = (
            f"{HOVER_POINT_LIMIT:,} of {len(frame):,} points drawn — the page holds the "
            "data for hovering, and all of them would stop the tab responding."
        )

    if mapping is not None:
        domain, scheme = db.scale_for(mapping)
        encoding = alt.Color(
            f"{colour}:N",
            scale=alt.Scale(domain=domain, range=scheme),
            legend=alt.Legend(title=colour, symbolSize=80),
        )
    else:
        if limits is None:
            values = shown[colour].astype(float)
            limits = tuple(float(v) for v in np.nanpercentile(values, [1, 99]))
        low, high = limits
        encoding = alt.Color(
            f"{colour}:Q",
            scale=alt.Scale(scheme="viridis", domain=[float(low), float(high)], clamp=True),
            legend=alt.Legend(title=colour),
        )

    chart = (
        alt.Chart(shown)
        .mark_circle(size=14, opacity=0.75)
        .encode(
            x=alt.X("x:Q", title=x_title, scale=alt.Scale(zero=False, nice=False)),
            y=alt.Y(
                "y:Q",
                title=y_title,
                scale=alt.Scale(zero=False, nice=False, reverse=flip_y),
            ),
            color=encoding,
            tooltip=[alt.Tooltip(name, title=name) for name in tooltips if name in shown.columns],
        )
        .properties(height=height)
        .interactive()
    )
    return chart, note


def _tooltip_columns(adata, colour: str) -> list[str]:
    """What a point should say about itself: where it is, and what it is."""
    wanted = ["object", db.CLUSTER_KEY, "well", db.IMAGE_KEY, "label", colour]
    seen: list[str] = []
    for name in wanted:
        if name and name not in seen and (name == "object" or name in adata.obs or name in adata.var_names):
            seen.append(name)
    return seen


def _argument_file() -> str:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--file", default="")
    known, _rest = parser.parse_known_args()
    return known.file


@st.cache_resource(show_spinner="Reading the file…")
def _read(path: str, mtime: float):
    """Cached on the path *and* its modification time, so a re-run reloads it."""
    return db.read(path)


@st.cache_resource(show_spinner="Building the neighbourhood graph…")
def _analysed(
    path: str,
    mtime: float,
    image: str,
    limit: int,
    mode: str,
    n_neighs: int,
    radius: float,
    grouping: str,
    bins: int,
    stamp: str,
):
    """One image, joined up and ready for the spatial statistics.

    It does **not** cluster. The clusters come from the one clustering on the
    file, so this image's groups are the plate's groups — which is the only way a
    colour can mean the same thing here as on the UMAP. *stamp* is when that
    clustering was computed, and is in the cache key so that re-clustering
    invalidates this.
    """
    adata = db.select_image(_read(path, mtime), image or None)
    adata = db.subsample(adata, limit)
    if grouping == "cluster":
        record = db.load_clustering(_read(path, mtime))
        method = f"{record.get('method', 'the clustering')} (plate-wide)"
    else:
        # No scaling first: quartiles of a z-score are the quartiles of the raw
        # column anyway, and leaving it raw means the bin edges are in the units
        # the legend will name.
        method = db.bin_column(adata, grouping, bins=bins)
    graph = db.build_graph(adata, mode=mode, n_neighs=n_neighs, radius_um=radius)
    return adata, method, graph


def _figure(width: float = 6.0, height: float = PLOT_HEIGHT):
    from matplotlib.figure import Figure

    figure = Figure(figsize=(width, height), dpi=110)
    figure.patch.set_alpha(0.0)
    return figure


def _write_back(adata, path: str, groups: list) -> None:
    """Offer to paint this clustering into the plate it came from.

    Only offered when the file says which plate that was: an ``.h5ad`` carries the
    source table's path, and the plate is the ``.zarr`` those tables were written
    beside. Guessing at a plate and writing label sets into it is not a thing to
    do on a maybe.
    """
    from pathlib import Path as _Path

    with st.expander("Write these clusters into the plate", expanded=False):
        st.markdown(
            "Paints each object's cluster onto the nucleus it was measured from, as "
            "another NGFF label set beside the segmentation. The nuclei are not "
            "touched — this is a second label set, and deleting it leaves them as "
            "they were.\n\n"
            "Open the plate in the viewer afterwards and the clustering is there, in "
            "the colours on this page, with the method recorded on it."
        )

        every = st.radio(
            "Which objects",
            ["Every object", "Only the ones that were clustered"],
            help=(
                "The clustering runs on a sample — a few hundred objects a well — so "
                "most objects have no cluster of their own. Writing only those leaves a "
                "label layer that is almost entirely empty and makes a phenotype look "
                "far rarer than it is.\n\n"
                "“Every object” gives the rest a cluster from their nearest neighbours in "
                "the same feature space. That is a prediction rather than a measurement, "
                "and how many of each is recorded on the layer."
            ),
        ).startswith("Every")

        plate = st.text_input(
            "Plate (.zarr)",
            value=str(_guess_plate(path) or ""),
            help="The store the segmentation lives in.",
            key="writeback_plate",
        )
        columns = st.columns(3)
        name = columns[0].text_input("Label set to write", value="clusters", key="writeback_name")
        source = columns[1].text_input(
            "Painted onto",
            value=str(_guess_source(path) or "nuclei"),
            help="The label set the objects were segmented into — the one whose label "
                 "numbers the table's 'label' column refers to.",
            key="writeback_source",
        )
        replace = columns[2].checkbox(
            "Replace every clustering",
            value=False,
            help="Remove every label set in the plate that carries a clustering marker "
                 "before writing. Segmentations are never touched: one is an hour of GPU "
                 "and this is forty seconds.",
            key="writeback_replace",
        )

        if not st.button("Write into the plate", key="writeback_go"):
            return
        if not plate or not _Path(plate).exists():
            st.error("That plate does not exist.")
            return

        from microscopy_viewer import batch as mvbatch
        from microscopy_viewer import clusters as mvclusters

        progress = st.empty()
        try:
            survey = mvbatch.survey_plate(plate)
            if every:
                with st.spinner("Giving every object a cluster…"):
                    frame = db.assign_all(
                        adata,
                        _read(path, _Path(path).stat().st_mtime),
                        progress=lambda text: progress.write(text),
                    )
            else:
                frame = db.assignment_frame(adata)
            assignments = mvclusters.assignments_from_frame(frame)
            run = db.cluster_run(adata, source_labels=source, frame=frame, plate=str(plate))
        except Exception as exc:  # noqa: BLE001 - shown on the page
            st.exception(exc)
            return

        rows = mvclusters.write_plate_clusters(
            survey,
            assignments,
            run,
            name=name or mvclusters.DEFAULT_NAME,
            source_labels=source,
            replace_existing=bool(replace),
            progress=lambda text: progress.write(text),
        )
        progress.empty()

        import pandas as pd

        written = [row for row in rows if row[1] is not None]
        report = pd.DataFrame(
            [
                {"image": component, "objects painted": assigned,
                 "written": "yes" if target is not None else "no", "note": note}
                for component, target, assigned, note in rows
            ]
        )
        if written:
            st.success(
                f"Wrote “{name}” into {len(written)} of {len(rows)} image(s); "
                f"{sum(row[2] for row in written):,} objects painted. "
                f"Recorded as: {run.describe()}."
            )
        else:
            st.error("Nothing was written — see the notes below.")
        st.dataframe(report)


def _guess_plate(path: str):
    """The .zarr the tables beside this .h5ad were written for, if it is findable."""
    from pathlib import Path as _Path

    here = _Path(str(path)).parent
    for folder in (here, here.parent):
        try:
            for entry in folder.iterdir():
                if entry.is_dir() and entry.suffix.lower() == ".zarr":
                    return entry
        except OSError:
            continue
    return None


def _guess_source(path: str) -> str:
    """The label set the tables came from: ``…_nuclei_objects`` was run on ``nuclei``."""
    from pathlib import Path as _Path

    stem = _Path(str(path)).parent.name
    if stem.endswith("_objects"):
        middle = stem[: -len("_objects")]
        return middle.rsplit("_", 1)[-1] or "nuclei"
    return "nuclei"


#: Panels per row in the plate display, and how big each one is.
WELL_COLUMNS = 6
WELL_PANEL = 190


def _well_display_tab(path: str, mtime: float) -> None:
    """Every well's spatial view, laid out as a plate, from the one clustering.

    Derived rather than stored: the clustering is what costs minutes and it is
    already on the file, so the panels are a subsample of it and are recomputed in
    a moment. Storing them separately was one more thing that could disagree with
    the clustering it came from.
    """
    source = _read(path, mtime)
    record = db.load_clustering(source)
    if not record:
        st.info(
            "Nothing is clustered yet. Set the clustering in the sidebar and press "
            "**Cluster** — this tab reads it like every other."
        )
        return

    groups = [str(name) for name in record["clusters"]]
    palette = db.clustering_palette(record)

    options = st.columns(4)
    per_well = options[0].number_input(
        "Objects per panel", 100, 2000, db.DEFAULT_VIEW_POINTS, step=50, key="wv_points",
        help="Drawn per well. The clustering is not redone — this only decides how many "
             "of its objects each panel shows.",
    )
    columns = options[1].slider("Panels per row", 2, 10, WELL_COLUMNS, key="wv_cols")
    size = options[2].slider("Panel size", 110, 320, WELL_PANEL, step=10, key="wv_size")
    shared = options[3].checkbox(
        "One scale for every well", value=True, key="wv_shared",
        help="On, the panels are on the same micrometre scale, so a small well looks "
             "small. Off, each fills its own panel.",
    )

    frame = _well_frame(path, mtime, int(per_well), str(record.get("created", "")))
    totals = source.obs[db.IMAGE_KEY].astype(str).value_counts().to_dict()
    frame = frame.assign(
        objects_in_well=[int(totals.get(str(image), 0)) for image in frame[db.IMAGE_KEY]]
    )

    st.caption(
        f"**{frame[db.IMAGE_KEY].nunique()} wells**, {len(frame):,} points drawn of "
        f"{record['n_objects']:,} · {record['method']} into {len(groups)} clusters · "
        "the same clustering as every other tab"
    )

    chosen = st.multiselect(
        "Show clusters", groups, default=groups, key="wv_clusters",
        help="Narrow to one phenotype to see where it sits in every well at once — "
             "which is the question this layout exists for.",
    )
    shown = frame[frame["cluster"].astype(str).isin(chosen)] if chosen else frame
    domain = [name for name in groups if name in set(chosen or groups)]

    chart = (
        alt.Chart(shown)
        .mark_circle(size=9, opacity=0.8)
        .encode(
            x=alt.X("x:Q", title=None, axis=None, scale=alt.Scale(zero=False, nice=False)),
            y=alt.Y("y:Q", title=None, axis=None,
                    scale=alt.Scale(zero=False, nice=False, reverse=True)),
            color=alt.Color(
                "cluster:N",
                scale=alt.Scale(domain=domain, range=[palette[n] for n in domain]),
                legend=alt.Legend(title="cluster", symbolSize=80),
            ),
            tooltip=[
                alt.Tooltip("well:N"),
                alt.Tooltip(f"{db.IMAGE_KEY}:N", title="image"),
                alt.Tooltip("cluster:N"),
                alt.Tooltip("label:Q", title="object"),
                alt.Tooltip("objects_in_well:Q", title="objects in well"),
                alt.Tooltip("x:Q", format=".0f"),
                alt.Tooltip("y:Q", format=".0f"),
            ],
        )
        .properties(width=int(size), height=int(size))
        .facet(
            facet=alt.Facet(f"{db.IMAGE_KEY}:N", title=None,
                            sort=sorted(frame[db.IMAGE_KEY].unique())),
            columns=int(columns),
        )
    )
    if not shared:
        chart = chart.resolve_scale(x="independent", y="independent")
    st.altair_chart(chart)
    st.caption(
        "Point at any object in any panel: it says its well, its cluster and how many "
        "objects that well holds in total. Every panel is coloured by the one "
        "clustering, which is what makes them comparable."
        + ("" if shared else " Each panel is on its own scale, so the wells are not "
           "comparable in size.")
    )


@st.cache_data(show_spinner="Laying out the wells…")
def _well_frame(path: str, mtime: float, per_well: int, stamp: str):
    """Points per well, drawn from the stored clustering. *stamp* invalidates it."""
    import pandas as pd

    source = _read(path, mtime)
    parts = []
    for image in db.images(source) or [""]:
        part = db.clustered_frame(source, image=image or None, limit=int(per_well))
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def _plate_tab(path: str, mtime: float, colour_scale: str = "scaled") -> None:
    """Every well at once: the stored UMAP, and what each well is made of.

    Draws the clustering the sidebar computed rather than one of its own. There
    used to be a clustering here, another for each image and a third for the well
    display, which meant cluster 3 was three different things on one page.
    """
    source = _read(path, mtime)
    record = db.load_clustering(source)
    if not record:
        st.info(
            "Nothing is clustered yet. Set the clustering in the sidebar and press "
            "**Cluster** — this tab and every other read it."
        )
        return
    if "umap" not in record:
        st.warning("This clustering was stored without an embedding. Re-cluster to get one.")
        return

    groups = [str(name) for name in record["clusters"]]
    palette = db.clustering_palette(record)
    st.markdown(
        "**Which wells hold which phenotypes.** Every well at once, in feature space "
        "rather than in the well — this is the question the spatial tabs cannot ask, "
        "because the coordinates in them are per image."
    )
    st.caption(
        f"**{record['n_objects']:,} objects** from {source.obs[db.IMAGE_KEY].nunique()} images · "
        f"{record['method']} into {len(groups)} clusters · "
        f"{record['n_clustered']:,} clustered, {record['n_assigned']:,} assigned by neighbours"
        + (f" · {record['created']}" if record.get("created") else "")
    )

    # The embedding covers the objects that were clustered; the rest have a
    # cluster but no place on the map, which is what an embedding is.
    import pandas as pd

    umap = record["umap"]
    embedded = pd.DataFrame({"object": umap["names"], "x": umap["x"], "y": umap["y"]})
    columns = ["cluster", "well", db.IMAGE_KEY, "label", db.ORIGIN_COLUMN]
    lookup = db.clustered_frame(source)
    embedded = embedded.merge(
        lookup[[c for c in ["object", *columns] if c in lookup.columns]], on="object", how="left"
    )

    left, right = st.columns(2)
    with left:
        st.subheader("The map")
        colour_by = st.selectbox(
            "Colour by",
            ["cluster", "well", "row", "column"] + db.feature_names(source),
            key="umap_colour",
        )
        frame = embedded.copy()
        if colour_by in ("row", "column"):
            part = 0 if colour_by == "row" else 1
            frame[colour_by] = [db.row_column(str(name))[part] for name in frame[db.IMAGE_KEY]]
        categorical = colour_by in ("cluster", "well", "row", "column")
        mapping = None
        limits, scale_note = None, ""
        if categorical:
            levels = groups if colour_by == "cluster" else sorted(set(frame[colour_by].astype(str)))
            mapping = palette if colour_by == "cluster" else db.colour_map(levels)
        else:
            values, low, high, scale_note = db.colour_scale(
                source, source[[str(n) for n in frame["object"]]], colour_by, colour_scale
            )
            frame[colour_by] = values
            limits = (low, high)

        tooltips = [c for c in ("object", "cluster", "well", db.IMAGE_KEY, "label",
                                db.ORIGIN_COLUMN, colour_by) if c in frame.columns]
        chart, note = _scatter(
            frame, colour_by, mapping, "UMAP 1", "UMAP 2", tooltips, height=520, limits=limits
        )
        st.altair_chart(chart)
        hints = [
            "Point at an object to see its well and its cluster — which is how you find "
            "out whether a corner of this map is one well or many."
        ]
        if scale_note:
            hints.append(f"Colour scale: {scale_note}.")
        if note:
            hints.append(note)
        if mapping is not None and db.OVERFLOW_COLOUR in mapping.values():
            spare = sum(1 for c in mapping.values() if c == db.OVERFLOW_COLOUR)
            hints.append(f"{spare} of the {len(mapping)} {colour_by}s share grey.")
        st.caption(" ".join(hints))
        st.caption(
            "Distances between clusters on a UMAP mean nothing; only what is together "
            "and what is apart does. Two wells landing in different places is a real "
            "difference — but it can be a difference in staining or focus as easily as "
            "in biology, so check a couple in the image before believing it. Only the "
            f"{record['n_clustered']:,} clustered objects are on the map; the rest have "
            "a cluster but no position in it."
        )

    with right:
        st.subheader("What each well is made of")
        shares = db.composition(source, by="well")
        stacked = shares.reset_index()
        stacked = stacked.melt(
            id_vars=stacked.columns[0], var_name="cluster", value_name="share"
        )
        stacked.columns = ["well", "cluster", "share"]
        stacked["cluster"] = stacked["cluster"].astype(str)
        domain = [name for name in groups if name in set(stacked["cluster"])]
        st.altair_chart(
            alt.Chart(stacked)
            .mark_bar()
            .encode(
                x=alt.X("well:N", title=None, sort=list(shares.index)),
                y=alt.Y("share:Q", stack="normalize", axis=alt.Axis(format="%")),
                color=alt.Color(
                    "cluster:N",
                    scale=alt.Scale(domain=domain, range=[palette[n] for n in domain]),
                    legend=alt.Legend(title="cluster"),
                ),
                tooltip=[
                    alt.Tooltip("well:N"),
                    alt.Tooltip("cluster:N"),
                    alt.Tooltip("share:Q", format=".1%"),
                ],
            )
            .properties(height=260)
        )
        st.caption(
            "The share of each well's objects in each cluster — every object, not only "
            "the sampled ones, and shares rather than counts because the wells hold "
            "wildly different numbers of objects."
        )

        st.subheader("As a plate")
        which = st.selectbox("Share of cluster", groups, key="plate_group")
        swatch = palette.get(str(which), db.OVERFLOW_COLOUR)
        st.markdown(
            f'<span style="display:inline-block;width:12px;height:12px;'
            f'background:{swatch};border-radius:2px;margin-right:6px;"></span>'
            f"cluster {which}, the same colour it is everywhere else",
            unsafe_allow_html=True,
        )
        grid = db.plate_grid(shares, which)
        values = grid.to_numpy(dtype=float)
        from matplotlib import colormaps

        shaded = colormaps["magma"].with_extremes(bad="0.85")
        figure = _figure(6.0, 3.6)
        axes = figure.add_subplot(111)
        heat = axes.imshow(values, cmap=shaded, vmin=0.0, vmax=float(np.nanmax(values) or 1.0))
        axes.set_xticks(range(len(grid.columns)), list(grid.columns), fontsize=7)
        axes.set_yticks(range(len(grid.index)), list(grid.index), fontsize=7)
        figure.colorbar(heat, ax=axes, shrink=0.8, label=f"share in {which}")
        st.pyplot(figure)
        st.caption(
            "A plate is a physical object and the answer often is too — an edge effect, "
            "a column of controls, a row that did not take. Grey cells are wells this "
            "plate does not have, which is not the same as a well holding none."
        )

    _write_back(source, path, groups)

    with st.expander("The numbers"):
        st.dataframe(shares.style.format("{:.1%}"))


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

        st.header("Clustering")
        st.caption(
            "One clustering for the whole page. Every plot, the well display and the "
            "label set written into the plate all read it, so a colour means the same "
            "thing everywhere."
        )
        stored = db.load_clustering(adata)
        current = stored.get("params", db.ClusteringParams())
        wanted = db.ClusteringParams(
            sample_per_well=int(
                st.number_input("Objects per image", 100, 5000, int(current.sample_per_well),
                                step=50, key="cl_sample",
                                help="Sampled from each image and clustered. Every other "
                                     "object then takes the cluster of its neighbours.")
            ),
            resolution=float(
                st.slider("Resolution", 0.2, 2.0, float(current.resolution), step=0.1,
                          key="cl_res")
            ),
            n_neighbors=int(
                st.slider("UMAP neighbours", 5, 50, int(current.n_neighbors), key="cl_neigh")
            ),
            min_dist=float(
                st.slider("UMAP min_dist", 0.0, 1.0, float(current.min_dist), step=0.05,
                          key="cl_dist")
            ),
            assign_all=bool(
                st.checkbox("Give every object a cluster", value=bool(current.assign_all),
                            key="cl_all",
                            help="The objects the clustering did not sample take the "
                                 "cluster of their neighbours in the same feature space. "
                                 "Off, they have none, and the label set written into the "
                                 "plate is mostly empty.")
            ),
        )

        stale = bool(stored) and wanted != current
        if not stored:
            st.info("Not clustered yet.")
        elif stale:
            st.warning("The settings have changed since this was computed.")
        else:
            st.caption(
                f"{stored['method']} into {len(stored['clusters'])} clusters · "
                f"{stored['n_clustered']:,} clustered, {stored['n_assigned']:,} assigned"
                + (f" · {stored['created']}" if stored.get("created") else "")
            )

        recluster = st.button(
            ("Cluster" if not stored else "Re-cluster"),
            type=("primary" if (stale or not stored) else "secondary"),
            key="cl_go",
        )
        keep = st.checkbox(
            "Save into the file", value=True, key="cl_save",
            help="Writes the clustering into the .h5ad, so it is still there next time "
                 "and the label sets written into the plate match what is on screen.",
        )

        if recluster:
            progress = st.empty()
            try:
                db.run_clustering(adata, wanted, progress=lambda text: progress.write(text))
                if keep:
                    progress.write("writing the file…")
                    db.save_clustering(path, adata)
                progress.empty()
                _read.clear()
                st.rerun()
            except Exception as exc:  # noqa: BLE001 - shown in the sidebar
                progress.empty()
                st.exception(exc)

        st.header("Grouping")
        grouping_choice = st.radio(
            "The spatial statistics group by",
            ["The clustering", "Bins of one measurement"],
            help="The neighbourhood statistics need a label per object. The clustering "
                 "above is the usual one; bins of a single column are cruder and far "
                 "easier to explain in a figure legend.",
        )
        grouping, bins = "cluster", 4
        if not grouping_choice.startswith("The clustering"):
            grouping = st.selectbox(
                "Measurement", db.feature_names(adata) + db.obs_columns(adata, numeric_only=True)
            )
            bins = st.slider("Bins", 2, 8, 4)

        st.header("Colour scale")
        colour_scale = st.selectbox(
            "Features are coloured on",
            db.COLOUR_SCALES,
            format_func=lambda key: db.COLOUR_SCALE_LABELS[key],
            help=(
                "Clustering needs the features z-scored — integrated intensity is six "
                "figures and solidity is below one — so that is what the matrix holds by "
                "the time anything is drawn, and colouring by “Mean intensity” shows "
                "standard deviations rather than grey levels.\n\n"
                "The raw options read the measurement back off the file as it was loaded. "
                "“This cycle” puts every well of a staining round on one scale, which is "
                "how a plate is normally read: a well that is genuinely brighter then "
                "looks brighter."
            ),
        )

        st.header("Neighbours")
        mode = st.selectbox(
            "Graph", db.GRAPH_MODES,
            help="Delaunay joins each object to the ones it actually abuts, which is what "
                 "segmented nuclei are. A fixed radius misses them in a sparse field and "
                 "joins half the well in a dense one.",
        )
        n_neighs = st.slider("Neighbours (knn)", 3, 20, 6) if mode == "knn" else 6
        radius = st.slider("Radius (um)", 5.0, 200.0, 30.0) if mode == "radius" else 30.0

    if grouping == "cluster" and not stored:
        st.info(
            "Nothing is clustered yet. Set the clustering in the sidebar and press "
            "**Cluster** — every tab on this page reads it."
        )
        st.stop()

    try:
        adata, method, graph = _analysed(
            path, mtime, image, limit, mode, n_neighs, radius, grouping, bins,
            str(stored.get("created", "")),
        )
    except Exception as exc:  # noqa: BLE001 - shown on the page, not in a terminal
        st.exception(exc)
        st.stop()

    groups = list(adata.obs[db.CLUSTER_KEY].cat.categories)
    # From the stored record, so the well display, the UMAP and the plate's own
    # label set are all painted the same.
    colours = db.clustering_palette(stored) if grouping == "cluster" else db.group_colours(adata)
    colours = {name: colours.get(name, db.OVERFLOW_COLOUR) for name in map(str, groups)}
    st.caption(
        f"**{adata.n_obs:,} objects**"
        + (f" from {image}" if image else "")
        + f" · grouped by {method} into {len(groups)} · joined by {graph}"
    )

    plate, wells, look, phenotype, space = st.tabs(
        ["Across the plate", "Well display", "Where they are", "What they are",
         "Does it mean anything"]
    )

    with wells:
        _well_display_tab(path, mtime)

    # -- every well at once ---------------------------------------------------
    with plate:
        _plate_tab(path, mtime, colour_scale)
    # -- where ----------------------------------------------------------------
    with look:
        left, right = st.columns([3, 2])
        with left:
            st.subheader("The well")
            colour = st.selectbox(
                "Colour by",
                [db.CLUSTER_KEY] + db.feature_names(adata) + db.obs_columns(adata, numeric_only=True),
            )
            xy = np.asarray(adata.obsm["spatial"])
            points = _points(adata, xy[:, 0], xy[:, 1], colour, extra=_tooltip_columns(adata, colour))
            limits, scale_note = None, ""
            if colour != db.CLUSTER_KEY:
                values, low, high, scale_note = db.colour_scale(
                    _read(path, mtime), adata, colour, colour_scale
                )
                points[colour] = values
                limits = (low, high)
            chart, note = _scatter(
                points,
                colour,
                colours if colour == db.CLUSTER_KEY else None,
                "x (um)",
                "y (um)",
                _tooltip_columns(adata, colour) + ["x", "y"],
                height=520,
                # Image convention: y runs down the well, as it does in the viewer.
                flip_y=True,
                limits=limits,
            )
            st.altair_chart(chart)
            hint = "Point at an object to see which cluster it is in and what it measures. "
            if scale_note:
                hint += f"Colour scale: {scale_note}. "
            st.caption(hint + note)

        with right:
            st.subheader("How many of each")
            counts = adata.obs[db.CLUSTER_KEY].value_counts().sort_index()
            import pandas as pd

            tally = counts.rename("objects").reset_index()
            tally.columns = [db.CLUSTER_KEY, "objects"]
            tally[db.CLUSTER_KEY] = tally[db.CLUSTER_KEY].astype(str)
            domain, scheme = db.scale_for(colours)
            st.altair_chart(
                alt.Chart(tally)
                .mark_bar()
                .encode(
                    x=alt.X(f"{db.CLUSTER_KEY}:N", sort=domain, title="cluster"),
                    y=alt.Y("objects:Q"),
                    color=alt.Color(
                        f"{db.CLUSTER_KEY}:N",
                        scale=alt.Scale(domain=domain, range=scheme),
                        legend=None,
                    ),
                    tooltip=["cluster", "objects"],
                )
                .properties(height=200),
            )
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
            drawn = axes.boxplot(data, showfliers=False, patch_artist=True)
            # Painted to match: a box plot beside a scatter of the same groups in
            # different colours is two plots the reader has to reconcile by hand.
            for patch, group in zip(drawn["boxes"], groups):
                patch.set_facecolor(colours.get(str(group), db.OVERFLOW_COLOUR))
                patch.set_alpha(0.85)
                patch.set_edgecolor("0.3")
            for median in drawn["medians"]:
                median.set_color("0.2")
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
            x, y = db.values_of(adata, feature), db.values_of(adata, pair)
            frame = _points(adata, x, y, db.CLUSTER_KEY, extra=_tooltip_columns(adata, feature))
            chart, note = _scatter(
                frame,
                db.CLUSTER_KEY,
                colours,
                feature,
                pair,
                _tooltip_columns(adata, db.CLUSTER_KEY) + ["x", "y"],
                height=380,
            )
            st.altair_chart(chart)
            if note:
                st.caption(note)

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
                # The tick labels carry the group's colour, so a cell in this matrix
                # can be traced back to the two clusters in the scatter above.
                for ticks in (axes.get_xticklabels(), axes.get_yticklabels()):
                    for tick, group in zip(ticks, groups):
                        tick.set_color(colours.get(str(group), db.OVERFLOW_COLOUR))
                        tick.set_fontweight("bold")
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
                    axes.plot(
                        part["bins"], part["stats"], label=str(group), linewidth=1.2,
                        color=colours.get(str(group), db.OVERFLOW_COLOUR),
                    )
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
                    axes.plot(
                        interval[:-1], occ[index, position, :], label=str(group), linewidth=1.2,
                        color=colours.get(str(group), db.OVERFLOW_COLOUR),
                    )
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
