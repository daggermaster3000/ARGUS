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

**Wells** is the plate laid out: every well's spatial view side by side,
from one clustering so the colours mean the same thing in each. Computed once and
kept in the file, as points rather than as pictures — a picture cannot be pointed
at, and hovering is how a panel says which well it is.

There are two questions here, and they do not mix. **Plate** and **Wells** ask
which wells hold which phenotypes, in feature space, using every well at once.
**Space**, **Features** and **Statistics** ask where those phenotypes sit inside
one well, which is spatial and has to be one image at a time because the
coordinates are per image.

The layout follows one skeleton, deliberately, so that the page reads as one tool
rather than six:

* **The sidebar holds only what changes every tab at once** — the file, the
  colours, which rows, how they are grouped — in the order those choices are
  made. A control that steers one tab lives in that tab however global it feels
  while writing it: the neighbourhood graph is in **Statistics**, beside the
  statistics that are statements about it.
* **One caption under the title says what survived**, so it is true for every tab
  and always in view.
* **A tab answers one question**, named in one word, with its controls in a left
  column and the result in a right one three times as wide. The controls read top
  to bottom in the order the decision is made.
* **Table is always last**: the objects as rows, with a CSV. It is the way out of
  the page and the check on everything the plots claim.
* **One vocabulary for states.** ``st.info`` says what to load or compute,
  ``st.warning`` says what to change, ``st.error`` says what broke and quotes it,
  ``st.caption`` reports a partial result. Never an empty chart, and never a
  traceback for something the data can produce.
* **Every chart goes out through :func:`chart`**, which applies the house style —
  every filled mark edged in black — and every plot's values are downloadable
  from under it. Colour is :mod:`microscopy_viewer.dashboard`'s, through
  :func:`microscopy_viewer.dashboard.paint`, so the sidebar's pickers reach every
  plot and there is no hex literal in this file.

The tab order is the order of the argument: what is in the file, what the objects
look like, how they are grouped, and only then where those groups sit and whether
that arrangement means anything. The spatial statistics are last because they are
the slow ones and because they are meaningless until the grouping above them is
one the reader believes.
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

# ---------------------------------------------------------------------------
# The house style
# ---------------------------------------------------------------------------
# Heights are fixed numbers rather than left to the renderer, so that a tab does
# not change shape when its data does and two panels meant to be read together
# are the same size. Colour comes from :mod:`microscopy_viewer.dashboard`, which
# is the one place it is decided; there is no hex literal in this file.

#: The plot a tab is about.
HEIGHT_MAIN = 520
#: A second plot beside or under the main one.
HEIGHT_SECOND = 380
#: A small supporting chart — a tally, a share.
HEIGHT_SMALL = 220
#: Matplotlib figures, in inches, for the ones Vega draws worse.
FIGURE_HEIGHT = 5.0
FIGURE_WIDTH = 6.0

#: Width of the black edge every filled mark gets, and the point count above
#: which it is thinned. At 1 px a dense scatter turns into a black sheet.
OUTLINE_WIDTH = 1.0
OUTLINE_THIN = 0.4
OUTLINE_DENSE = 2_000

#: Points drawn in an interactive scatter. Vega holds the data in the page, so a
#: hoverable plot of half a million points is a browser tab that stops responding;
#: this is where hovering stays instant.
HOVER_POINT_LIMIT = 20_000


def outlined(figure, points: int = 0):
    """The house style, applied in one place: every filled mark edged in black.

    An outline is what lets a pale group read against the page and against the
    mark behind it, and it is the difference between a figure that survives being
    printed and one that does not. Lines and heatmaps are left alone — an outline
    round a line is just a thicker line, and round a heatmap cell it is a grid.

    *points* is how many marks the chart holds; above :data:`OUTLINE_DENSE` the
    edge is thinned, because at one pixel a scatter of thousands is a black
    sheet with some colour behind it.

    Text and axis colours are deliberately **not** set here. The page follows the
    reader's light or dark theme, and pinning ink to a light-theme grey is how a
    chart ends up unreadable for half its users.
    """
    width = OUTLINE_THIN if points > OUTLINE_DENSE else OUTLINE_WIDTH
    return (
        figure.configure_circle(stroke=db.OUTLINE, strokeWidth=width)
        .configure_point(stroke=db.OUTLINE, strokeWidth=width)
        .configure_bar(stroke=db.OUTLINE, strokeWidth=OUTLINE_WIDTH)
        .configure_area(stroke=db.OUTLINE, strokeWidth=OUTLINE_WIDTH)
        .configure_view(strokeWidth=0)
    )


def chart(figure, points: int = 0, **kwargs) -> None:
    """Draw an Altair chart. The only way a chart reaches the page.

    Going through one function is what keeps the style one style: a raw
    ``st.altair_chart`` elsewhere would be the one plot without black edges, and
    nobody would notice until a figure went into a paper.
    """
    st.altair_chart(outlined(figure, points), **kwargs)


def show(figure) -> None:
    """Draw a matplotlib figure. The counterpart of :func:`chart`."""
    st.pyplot(figure)


def values_csv(frame, label: str, name: str, key: str) -> None:
    """The numbers behind a plot, downloadable from under it.

    Every plot owes the reader this: a picture is an argument, and the way to
    check an argument is to have the values it was drawn from.
    """
    st.download_button(
        label,
        frame.to_csv(index=False).encode("utf-8"),
        file_name=name,
        mime="text/csv",
        key=key,
    )

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
            scale=alt.Scale(scheme=db.SEQUENTIAL, domain=[float(low), float(high)], clamp=True),
            legend=alt.Legend(title=colour),
        )

    drawn = (
        alt.Chart(shown)
        .mark_circle(size=14, opacity=0.9)
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
    if equal:
        # Where the axes are both a distance in µm, the picture has to keep the
        # well's proportions: a round organoid drawn in a wide panel is an oval,
        # and the shape is what the plot is being read for.
        span_x = float(shown["x"].max() - shown["x"].min()) or 1.0
        span_y = float(shown["y"].max() - shown["y"].min()) or 1.0
        drawn = drawn.properties(height=height, width=max(240, min(900, height * span_x / span_y)))
    return drawn, note


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


@st.cache_resource(show_spinner="Reading the objects…")
def _prepared(
    path: str,
    mtime: float,
    image: str,
    limit: int,
    grouping: str,
    bins: int,
    stamp: str,
):
    """One image's objects and the group each belongs to.

    It does **not** cluster. The clusters come from the one clustering on the
    file, so this image's groups are the plate's groups — which is the only way a
    colour can mean the same thing here as on the UMAP. *stamp* is when that
    clustering was computed, and is in the cache key so that re-clustering
    invalidates this.

    It does not build a neighbourhood graph either. Four of the six tabs never
    need one, and a Delaunay triangulation of fifty thousand objects is not
    something to pay for while looking at a UMAP. :func:`_joined` adds it for the
    one tab whose statistics are about who is next to whom.
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
    return adata, method


@st.cache_resource(show_spinner="Building the neighbourhood graph…")
def _joined(
    path: str,
    mtime: float,
    image: str,
    limit: int,
    grouping: str,
    bins: int,
    stamp: str,
    mode: str,
    n_neighs: int,
    radius: float,
):
    """The same objects with a neighbourhood graph on them, for the statistics.

    A copy of what :func:`_prepared` cached, because the graph is written into
    ``obsp`` and two graph settings sharing one cached object would each see the
    other's.
    """
    adata, method = _prepared(path, mtime, image, limit, grouping, bins, stamp)
    joined = adata.copy()
    graph = db.build_graph(joined, mode=mode, n_neighs=n_neighs, radius_um=radius)
    return joined, method, graph


def _figure(width: float = FIGURE_WIDTH, height: float = FIGURE_HEIGHT):
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
            key="plate-write-plate",
        )
        columns = st.columns(3)
        name = columns[0].text_input("Label set to write", value="clusters", key="plate-write-name")
        source = columns[1].text_input(
            "Painted onto",
            value=str(_guess_source(path) or "nuclei"),
            help="The label set the objects were segmented into — the one whose label "
                 "numbers the table's 'label' column refers to.",
            key="plate-write-source",
        )
        replace = columns[2].checkbox(
            "Replace every clustering",
            value=False,
            help="Remove every label set in the plate that carries a clustering marker "
                 "before writing. Segmentations are never touched: one is an hour of GPU "
                 "and this is forty seconds.",
            key="plate-write-replace",
        )

        if not st.button("Write into the plate", key="plate-write-go"):
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
            st.error(f"The clusters could not be prepared for writing: {exc}")
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


def _well_display_tab(path: str, mtime: float, colours: dict[str, str] | None = None) -> None:
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
    palette = colours or db.clustering_palette(record)

    st.caption(
        "Every well side by side, each panel the spatial view of one well, all of them "
        "coloured by the one clustering — which is what makes them comparable."
    )

    left, right = st.columns([1, 3])
    with left:
        st.subheader("The plate")
        chosen = st.multiselect(
            "Show clusters", groups, default=groups, key="wells-clusters",
            help="Narrow to one phenotype to see where it sits in every well at once — "
                 "which is the question this layout exists for.",
        )
        per_well = st.number_input(
            "Objects per panel", 100, 2000, db.DEFAULT_VIEW_POINTS, step=50,
            key="wells-points",
            help="Drawn per well. The clustering is not redone — this only decides how "
                 "many of its objects each panel shows.",
        )
        columns = st.slider("Panels per row", 2, 10, WELL_COLUMNS, key="wells-columns")
        size = st.slider("Panel size", 110, 320, WELL_PANEL, step=10, key="wells-size")
        shared = st.checkbox(
            "One scale for every well", value=True, key="wells-shared",
            help="On, the panels are on the same micrometre scale, so a small well looks "
                 "small. Off, each fills its own panel.",
        )

    frame = _well_frame(path, mtime, int(per_well), str(record.get("created", "")))
    totals = source.obs[db.IMAGE_KEY].astype(str).value_counts().to_dict()
    frame = frame.assign(
        objects_in_well=[int(totals.get(str(image), 0)) for image in frame[db.IMAGE_KEY]]
    )

    # An emptied list means nothing was chosen — not everything. Falling back to
    # all of them would redraw the plot the reader just asked to narrow.
    shown = frame[frame["cluster"].astype(str).isin([str(n) for n in chosen])]
    domain = [name for name in groups if name in {str(n) for n in chosen}]
    if shown.empty:
        with right:
            st.warning(
                "No clusters chosen. Put one back in the list on the left to draw the "
                "plate."
            )
        return

    drawn = (
        alt.Chart(shown)
        .mark_circle(size=9, opacity=0.8)
        .encode(
            x=alt.X("x:Q", title=None, axis=None, scale=alt.Scale(zero=False, nice=False)),
            y=alt.Y("y:Q", title=None, axis=None,
                    scale=alt.Scale(zero=False, nice=False, reverse=True)),
            color=alt.Color(
                "cluster:N",
                scale=alt.Scale(
                    domain=domain,
                    range=[palette.get(n, db.OVERFLOW_COLOUR) for n in domain],
                ),
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
        drawn = drawn.resolve_scale(x="independent", y="independent")

    with right:
        st.caption(
            f"{frame[db.IMAGE_KEY].nunique()} well(s), {len(shown):,} of "
            f"{record['n_objects']:,} object(s) drawn · {record['method']} into "
            f"{len(groups)} group(s)."
        )
        chart(drawn, points=len(shown))
        st.caption(
            "Point at any object in any panel: it says its well, its cluster and how "
            "many objects that well holds in total."
            + ("" if shared else " Each panel is on its own scale, so the wells are not "
               "comparable in size.")
        )
        values_csv(shown, "These points (CSV)", "well_display_points.csv", "wells-csv")


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


def _plate_tab(
    path: str,
    mtime: float,
    colour_scale: str = "scaled",
    colours: dict[str, str] | None = None,
) -> None:
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
    palette = colours or db.clustering_palette(record)
    st.caption(
        "Which wells hold which phenotypes: every well at once, in feature space rather "
        "than in the well — the question the spatial tabs cannot ask, because their "
        f"coordinates are per image. {record['n_objects']:,} object(s) from "
        f"{source.obs[db.IMAGE_KEY].nunique()} image(s), {record['n_clustered']:,} "
        f"clustered and {record['n_assigned']:,} assigned by neighbours"
        + (f", {record['created']}" if record.get("created") else "")
        + "."
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

    left, right = st.columns([1, 3])
    with left:
        st.subheader("The map")
        colour_by = st.selectbox(
            "Colour by",
            ["cluster", "well", "row", "column"] + db.feature_names(source),
            key="plate-colour",
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

        st.subheader("As a plate")
        which = st.selectbox("Share of cluster", groups, key="plate-group")

    tooltips = [c for c in ("object", "cluster", "well", db.IMAGE_KEY, "label",
                            db.ORIGIN_COLUMN, colour_by) if c in frame.columns]

    with right:
        drawn, note = _scatter(
            frame, colour_by, mapping, "UMAP 1", "UMAP 2", tooltips,
            height=HEIGHT_MAIN, limits=limits,
        )
        chart(drawn, points=len(frame))
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
        values_csv(frame, "These points (CSV)", "umap_points.csv", "plate-umap-csv")

        st.subheader("What each well is made of")
        shares = db.composition(source, by="well")
        stacked = shares.reset_index()
        stacked = stacked.melt(
            id_vars=stacked.columns[0], var_name="cluster", value_name="share"
        )
        stacked.columns = ["well", "cluster", "share"]
        stacked["cluster"] = stacked["cluster"].astype(str)
        domain = [name for name in groups if name in set(stacked["cluster"])]
        chart(
            alt.Chart(stacked)
            .mark_bar()
            .encode(
                x=alt.X("well:N", title=None, sort=list(shares.index)),
                y=alt.Y("share:Q", stack="normalize", axis=alt.Axis(format="%")),
                color=alt.Color(
                    "cluster:N",
                    scale=alt.Scale(
                        domain=domain,
                        range=[palette.get(n, db.OVERFLOW_COLOUR) for n in domain],
                    ),
                    legend=alt.Legend(title="cluster"),
                ),
                tooltip=[
                    alt.Tooltip("well:N"),
                    alt.Tooltip("cluster:N"),
                    alt.Tooltip("share:Q", format=".1%"),
                ],
            )
            .properties(height=HEIGHT_SMALL)
        )
        st.caption(
            "The share of each well's objects in each cluster — every object, not only "
            "the sampled ones, and shares rather than counts because the wells hold "
            "wildly different numbers of objects."
        )

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

        # A share is an amount, so a one-hue sequential ramp, and a well the
        # plate does not have is blank rather than the bottom of the scale: no
        # well is not the same claim as a well with none of this cluster.
        shaded = colormaps[db.SEQUENTIAL].with_extremes(bad=db.GRID)
        figure = _figure(FIGURE_WIDTH, 3.6)
        axes = figure.add_subplot(111)
        heat = axes.imshow(values, cmap=shaded, vmin=0.0, vmax=float(np.nanmax(values) or 1.0))
        axes.set_xticks(range(len(grid.columns)), list(grid.columns), fontsize=7)
        axes.set_yticks(range(len(grid.index)), list(grid.index), fontsize=7)
        axes.set_title(f"Share of cluster {which}, by well", fontsize=9)
        figure.colorbar(heat, ax=axes, shrink=0.8, label=f"share in {which}")
        show(figure)
        st.caption(
            "A plate is a physical object and the answer often is too — an edge effect, "
            "a column of controls, a row that did not take. Blank cells are wells this "
            "plate does not have, which is not the same as a well holding none."
        )
        values_csv(
            shares.reset_index(), "These shares (CSV)", "well_composition.csv",
            "plate-shares-csv",
        )

    _write_back(source, path, groups)

    with st.expander(f"The numbers ({len(shares)} wells)"):
        st.caption("One row per well, one column per cluster, as shares of that well.")
        st.dataframe(shares.style.format("{:.1%}"), width="stretch")


#: Rows shown in the Table tab at once. The download is the whole thing; the
#: table on screen is for looking, and a browser asked to lay out fifty thousand
#: rows stops responding.
TABLE_ROWS = 2_000


def _table_tab(adata, image: str, colours: dict[str, str]) -> None:
    """The objects as rows, and a CSV of them.

    Every other tab is an argument about these numbers. This is the way to check
    one — and the way out of the page, into a spreadsheet or a script, without
    going back to the viewer to export again.
    """
    import pandas as pd

    frame = pd.DataFrame(index=[str(name) for name in adata.obs_names])
    frame.index.name = "object"
    for column in (db.CLUSTER_KEY, db.ORIGIN_COLUMN, db.IMAGE_KEY, "label"):
        if column in adata.obs:
            frame[column] = adata.obs[column].astype(str).to_numpy()
    spatial = adata.obsm.get("spatial")
    if spatial is not None:
        coordinates = np.asarray(spatial, dtype=float)
        frame["x (µm)"] = coordinates[:, 0]
        frame["y (µm)"] = coordinates[:, 1]
    for name in db.feature_names(adata):
        frame[name] = db.values_of(adata, name)
    frame = frame.reset_index()

    left, right = st.columns([1, 3])
    with left:
        st.subheader("Rows")
        wanted = st.multiselect(
            "Clusters", list(colours), default=list(colours), key="table-clusters",
            help="Filtering removes rows; nothing is selected in. All of them to start.",
        )
        columns = st.multiselect(
            "Columns",
            list(frame.columns),
            default=list(frame.columns)[: min(10, len(frame.columns))],
            key="table-columns",
            help="The download carries the columns chosen here, so a table meant for "
                 "a script can be narrowed before it is written.",
        )
        st.caption(
            "The clusters are the plate-wide ones. The measurements are scaled — "
            "z-scores, as everything was clustered on — not grey levels."
        )

    shown = frame
    if db.CLUSTER_KEY in shown.columns:
        shown = shown[shown[db.CLUSTER_KEY].isin([str(name) for name in wanted])]
    shown = shown[[name for name in columns if name in shown.columns]]

    with right:
        if not columns:
            st.warning("No columns chosen. Pick one in the list on the left.")
            return
        if shown.empty:
            st.warning(
                "No rows left. Put a cluster back in the list on the left, or widen "
                "**Objects at most** in the sidebar."
            )
            return
        st.caption(
            f"{len(shown):,} of {len(frame):,} object(s)"
            + (f" from {image}" if image else "")
            + f", {len(shown.columns)} of {len(frame.columns)} column(s). One row is one "
            "object; **object** is the name it carries in the file, and the one the "
            "written label sets use."
        )
        st.dataframe(
            shown.head(TABLE_ROWS),
            hide_index=True,
            width="stretch",
        )
        if len(shown) > TABLE_ROWS:
            st.caption(
                f"The first {TABLE_ROWS:,} rows are shown. The download is all "
                f"{len(shown):,}."
            )
        values_csv(
            shown, "These rows (CSV)",
            f"{(image or 'objects').replace('/', '_')}_objects.csv", "table-csv",
        )


#: Set to the file whose first clustering raised, so a failure shows the error
#: once instead of being retried on every re-run of the page.
_FAILED_KEY = "_clustering_failed"

#: Where the user's own colours live, as ``{kind: {value: hex}}``.
COLOUR_KEY = "_colour_overrides"

#: How many colour pickers to offer. Past this the sidebar is longer than the
#: page and nobody is hand-picking anyway; the caption says how many of how many.
COLOUR_PICKERS = 24


def _colour_pickers(slot, base: dict[str, str]) -> dict[str, str]:
    """Draw the cluster colour pickers into *slot* and return the painted map.

    The defaults are the clustering's own colours, stored on the file, so the
    pickers start where every plot already is. A pick is kept only when it
    differs from the default — so improving a default later still reaches someone
    who never picked — and it goes back out through
    :func:`microscopy_viewer.dashboard.paint`, which is what every plot on the
    page reads.
    """
    overrides: dict[str, dict[str, str]] = dict(st.session_state.get(COLOUR_KEY) or {})
    chosen = dict(overrides.get(db.CLUSTER_KEY) or {})

    with slot:
        with st.expander(f"Cluster colours ({len(base)})", expanded=False):
            st.caption(
                "A cluster is the same colour in every plot, in the well display and in "
                "the label set written into the plate. Change one here and it changes in "
                "all of them at once."
            )
            shown = list(base)[:COLOUR_PICKERS]
            for name in shown:
                picked = st.color_picker(
                    f"Cluster {name}",
                    value=chosen.get(name, base[name]),
                    key=f"colour-{db.CLUSTER_KEY}-{name}",
                )
                if picked.lower() == str(base[name]).lower():
                    chosen.pop(name, None)
                else:
                    chosen[name] = picked
            if len(base) > len(shown):
                st.caption(
                    f"{len(shown)} of {len(base)} clusters shown. The rest keep the "
                    "palette's own colours."
                )
            if chosen and st.button("Back to the defaults", key="colour-reset"):
                chosen = {}
                for name in shown:
                    st.session_state.pop(f"colour-{db.CLUSTER_KEY}-{name}", None)
                st.session_state[COLOUR_KEY] = {}
                st.rerun()

    overrides[db.CLUSTER_KEY] = chosen
    st.session_state[COLOUR_KEY] = overrides
    return db.paint(db.CLUSTER_KEY, base, overrides)


def _cluster_now(adata, path: str, params, keep: bool, progress) -> bool:
    """Run the one clustering and store it. ``True`` when it worked.

    The caller reruns the page afterwards rather than drawing from the object in
    hand: everything below reads the clustering back off the file, and a page
    half-drawn from memory and half from disk is how the two drift apart.
    """
    try:
        db.run_clustering(adata, params, progress=lambda text: progress.write(text))
        if keep:
            progress.write("writing the file…")
            db.save_clustering(path, adata)
        progress.empty()
        _read.clear()
        return True
    except Exception as exc:  # noqa: BLE001 - shown on the page, not in a terminal
        progress.empty()
        st.error(f"The clustering failed: {exc}")
        return False


def main() -> None:
    st.title("Spatial dashboard")

    default = _argument_file()
    with st.sidebar:
        st.header("File")
        path = st.text_input(
            "AnnData (.h5ad)",
            value=default,
            key="file-path",
            help="Written by the Measurement analysis panel, or by a batch run.",
        )
        uploaded = st.file_uploader("…or drop one here", type=["h5ad"], key="file-upload")
        if uploaded is not None:
            target = Path(st.session_state.get("_upload_dir", ".")) / uploaded.name
            target.write_bytes(uploaded.getbuffer())
            path = str(target)

    if path and not Path(path).exists():
        # Beside the box it was typed in, which is where someone who mistyped a
        # path is looking. The page-level message below is for having loaded
        # nothing at all, which is a different situation.
        st.sidebar.error(f"There is no file at {path}")
    if not path or not Path(path).exists():
        st.info(
            "Point this at an `.h5ad` written by the viewer — the Measurement analysis "
            "panel's **Export as AnnData**, or a batch run with the AnnData option ticked."
        )
        st.stop()

    mtime = Path(path).stat().st_mtime
    adata = _read(path, mtime)
    facts = db.describe(adata)

    if not facts["spatial"]:
        st.error(
            "This file has no `obsm['spatial']`, so there is nothing spatial to do with it. "
            "Re-export it from the viewer, which writes the centroids."
        )
        st.stop()

    # -- what to look at ------------------------------------------------------
    # The sidebar holds only what changes every tab at once, in the order the
    # choices are made: which file, how it is coloured, which rows, how they are
    # grouped. A control that steers one tab lives in that tab, however global it
    # feels while writing it — the neighbourhood graph is the example, and it is
    # in "Statistics" with the statistics that use it.
    with st.sidebar:
        st.header("Colours")
        # Filled in below, once there is a clustering to colour. A container
        # holds its place, so the section stays second in the sidebar rather than
        # appearing underneath the thing it has to know about first.
        colour_slot = st.container()

        st.header("Image")
        available = db.images(adata)
        image = ""
        if available:
            image = st.selectbox(
                "Well and cycle",
                available,
                key="rows-image",
                help=(
                    "One image at a time. The coordinates are positions within a well, so "
                    "two wells overlap in that space and a graph across the plate would "
                    "join objects that are merely in the same corner of different wells."
                ),
            )
        else:
            st.caption("One image in this file.")

        # Clamped to the file: a slider that can be dragged to sixty thousand
        # objects on a file holding nine hundred is a question with no answer.
        biggest = int(max(adata.obs[db.IMAGE_KEY].astype(str).value_counts().max(), 1)) \
            if db.IMAGE_KEY in adata.obs else int(adata.n_obs)
        ceiling = int(min(60_000, max(2_000, biggest)))
        limit = st.slider(
            "Objects at most", 1_000, ceiling, int(min(db.DEFAULT_MAX_OBJECTS, ceiling)),
            step=1_000, key="rows-limit",
            help="Subsampled at random above this, per image. A permutation test on fifty "
                 "thousand objects is minutes; on eight thousand it is seconds and the same "
                 "shape.",
        )
        st.caption(f"The largest image in this file holds {biggest:,} object(s).")

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
                                step=50, key="cluster-sample",
                                help="Sampled from each image and clustered. Every other "
                                     "object then takes the cluster of its neighbours.")
            ),
            resolution=float(
                st.slider("Resolution", 0.2, 2.0, float(current.resolution), step=0.1,
                          key="cluster-resolution")
            ),
            n_neighbors=int(
                st.slider("UMAP neighbours", 5, 50, int(current.n_neighbors), key="cluster-neighbours")
            ),
            min_dist=float(
                st.slider("UMAP min_dist", 0.0, 1.0, float(current.min_dist), step=0.05,
                          key="cluster-min-dist")
            ),
            assign_all=bool(
                st.checkbox("Give every object a cluster", value=bool(current.assign_all),
                            key="cluster-assign-all",
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
            key="cluster-go",
        )
        keep = st.checkbox(
            "Save into the file", value=True, key="cluster-save",
            help="Writes the clustering into the .h5ad, so it is still there next time "
                 "and the label sets written into the plate match what is on screen.",
        )

        if recluster:
            st.session_state.pop(_FAILED_KEY, None)
            if _cluster_now(adata, path, wanted, keep, st.empty()):
                st.rerun()

        st.header("Grouping")
        grouping_choice = st.radio(
            "The spatial statistics group by",
            ["The clustering", "Bins of one measurement"],
            key="group-by",
            help="The neighbourhood statistics need a label per object. The clustering "
                 "above is the usual one; bins of a single column are cruder and far "
                 "easier to explain in a figure legend.",
        )
        grouping, bins = "cluster", 4
        if not grouping_choice.startswith("The clustering"):
            grouping = st.selectbox(
                "Measurement",
                db.feature_names(adata) + db.obs_columns(adata, numeric_only=True),
                key="group-measure",
            )
            bins = st.slider("Bins", 2, 8, 4, key="group-bins")

        st.header("Colour scale")
        colour_scale = st.selectbox(
            "Features are coloured on",
            db.COLOUR_SCALES,
            format_func=lambda key: db.COLOUR_SCALE_LABELS[key],
            key="colour-scale",
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


    if not stored:
        # A file the viewer has just exported has no clustering, and every tab on
        # this page reads one — so without it there is nothing here at all. It is
        # computed on the first view rather than waiting for the button: the page
        # is opened to look at the clusters, and an empty page with a button on it
        # is a worse answer than a minute of work that is then stored on the file
        # and never repeated. The button is for changing it.
        if st.session_state.get(_FAILED_KEY) == str(path):
            st.error(
                "The first clustering failed — the error is above. Change the settings "
                "in the sidebar and press **Cluster** to try again."
            )
            st.stop()
        st.info(
            f"Clustering this file for the first time — {wanted.describe()}. "
            "It is then stored on the file and every tab reads it; change it in the "
            "sidebar and press **Re-cluster**."
        )
        if _cluster_now(adata, path, wanted, keep, st.empty()):
            st.rerun()
        st.session_state[_FAILED_KEY] = str(path)
        st.stop()

    stamp = str(stored.get("created", ""))
    try:
        adata, method = _prepared(path, mtime, image, limit, grouping, bins, stamp)
    except Exception as exc:  # noqa: BLE001 - shown on the page, not in a terminal
        st.error(f"The objects of this image could not be read: {exc}")
        st.stop()

    groups = list(adata.obs[db.CLUSTER_KEY].cat.categories)
    # From the stored record, so the well display, the UMAP and the plate's own
    # label set are all painted the same, and then through paint() so the
    # sidebar's pickers reach every one of them.
    base = db.clustering_palette(stored) if grouping == "cluster" else db.group_colours(adata)
    base = {name: base.get(name, db.OVERFLOW_COLOUR) for name in map(str, groups)}
    colours = _colour_pickers(colour_slot, base)

    st.caption(
        f"{adata.n_obs:,} of {facts['objects']:,} object(s)"
        + (f" from {image}" if image else "")
        + f", {facts['features']} feature(s) measured · grouped by {method} "
        f"into {len(groups)} group(s)."
    )

    plate_tab, wells_tab, space_tab, features_tab, stats_tab, table_tab = st.tabs(
        ["Plate", "Wells", "Space", "Features", "Statistics", "Table"]
    )

    with wells_tab:
        _well_display_tab(path, mtime, colours)

    # -- every well at once ---------------------------------------------------
    with plate_tab:
        _plate_tab(path, mtime, colour_scale, colours)
    # -- where ----------------------------------------------------------------
    with space_tab:
        st.caption(
            "Where the groups sit inside one well. The coordinates are positions "
            "within an image, so this is one image at a time — the well chosen in the "
            "sidebar."
        )
        left, right = st.columns([1, 3])
        with left:
            st.subheader("The well")
            colour = st.selectbox(
                "Colour by",
                [db.CLUSTER_KEY] + db.feature_names(adata) + db.obs_columns(adata, numeric_only=True),
                key="space-colour",
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
            keep_shape = st.checkbox(
                "Keep the well's proportions", value=True, key="space-equal",
                help="On, one µm across is one µm down, so a round organoid is round. "
                     "Off, the plot fills the width, which shows more detail and the "
                     "wrong shape.",
            )

        with right:
            drawn, note = _scatter(
                points,
                colour,
                colours if colour == db.CLUSTER_KEY else None,
                "x (µm)",
                "y (µm)",
                _tooltip_columns(adata, colour) + ["x", "y"],
                height=HEIGHT_MAIN,
                equal=keep_shape,
                # Image convention: y runs down the well, as it does in the viewer.
                flip_y=True,
                limits=limits,
            )
            chart(drawn, points=len(points))
            hint = "Point at an object to see which cluster it is in and what it measures. "
            if scale_note:
                hint += f"Colour scale: {scale_note}. "
            st.caption(hint + note)

            st.subheader("How many of each")
            counts = adata.obs[db.CLUSTER_KEY].value_counts().sort_index()

            tally = counts.rename("objects").reset_index()
            tally.columns = [db.CLUSTER_KEY, "objects"]
            tally[db.CLUSTER_KEY] = tally[db.CLUSTER_KEY].astype(str)
            tally["share"] = tally["objects"] / tally["objects"].sum()
            domain, scheme = db.scale_for(colours)
            st.caption(
                f"One bar per group, in the order the clustering numbered them. "
                f"{int(tally['objects'].sum()):,} object(s) in {len(tally)} group(s)."
            )
            chart(
                alt.Chart(tally)
                .mark_bar()
                .encode(
                    x=alt.X(f"{db.CLUSTER_KEY}:N", sort=domain, title="cluster"),
                    y=alt.Y("objects:Q", title="objects"),
                    color=alt.Color(
                        f"{db.CLUSTER_KEY}:N",
                        scale=alt.Scale(domain=domain, range=scheme),
                        legend=None,
                    ),
                    tooltip=[
                        alt.Tooltip(f"{db.CLUSTER_KEY}:N", title="cluster"),
                        alt.Tooltip("objects:Q", format=",d"),
                        alt.Tooltip("share:Q", format=".1%"),
                    ],
                )
                .properties(height=HEIGHT_SMALL),
            )
            left_numbers, right_numbers = st.columns([3, 2])
            with left_numbers:
                st.dataframe(
                    tally.style.format({"objects": "{:,.0f}", "share": "{:.1%}"}),
                    hide_index=True,
                    width="stretch",
                )
            with right_numbers:
                values_csv(
                    tally, "These counts (CSV)",
                    f"{(image or 'image').replace('/', '_')}_counts.csv", "space-counts-csv",
                )
                values_csv(
                    points, "These points (CSV)",
                    f"{(image or 'image').replace('/', '_')}_points.csv", "space-points-csv",
                )

    # -- what -----------------------------------------------------------------
    with features_tab:
        st.caption(
            "What tells the groups apart: the distribution of one measurement per "
            "group, and two measurements against each other."
        )
        left, right = st.columns([1, 3])

        with left:
            st.subheader("Measurements")
            feature = st.selectbox(
                "Measurement", db.feature_names(adata), key="features-measure"
            )
            pair = st.selectbox(
                "Against",
                [f for f in db.feature_names(adata) if f != feature],
                key="features-against",
            )

        with right:
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
                # The house style's black edge, in matplotlib: the same figure
                # read the same way whichever library drew it.
                patch.set_edgecolor(db.OUTLINE)
            for median in drawn["medians"]:
                median.set_color(db.OUTLINE)
            axes.set_xticks(range(1, len(groups) + 1), [str(g) for g in groups], rotation=45)
            axes.set_ylabel(feature, fontsize=8)
            axes.set_title(f"{feature} by group", fontsize=9)
            axes.tick_params(labelsize=7)
            show(figure)
            st.caption(
                "Scaled values: the features were z-scored before clustering, so zero is "
                "the image's mean and the units are standard deviations. Whiskers to 1.5 "
                "IQR, outliers not drawn."
            )

            x, y = db.values_of(adata, feature), db.values_of(adata, pair)
            frame = _points(adata, x, y, db.CLUSTER_KEY, extra=_tooltip_columns(adata, feature))
            drawn_pair, note = _scatter(
                frame,
                db.CLUSTER_KEY,
                colours,
                feature,
                pair,
                _tooltip_columns(adata, db.CLUSTER_KEY) + ["x", "y"],
                height=HEIGHT_SECOND,
            )
            chart(drawn_pair, points=len(frame))
            st.caption(
                f"{feature} against {pair}, one dot per object, coloured by group."
                + (f" {note}" if note else "")
            )
            values_csv(
                frame.rename(columns={"x": feature, "y": pair}),
                "These values (CSV)",
                f"{feature}_vs_{pair}.csv".replace(" ", "_").replace("/", "_"),
                "features-csv",
            )

    # -- does it mean anything ------------------------------------------------
    with stats_tab:
        st.caption(
            "Whether the arrangement means anything. These are the slow ones — a "
            "permutation test per plot — so each runs when you ask for it and is "
            "remembered afterwards."
        )
        graph_controls, graph_result = st.columns([1, 3])
        with graph_controls:
            st.subheader("Neighbours")
            st.caption(
                "Who counts as next to whom. Every statistic below is a statement "
                "about this graph, so it is here rather than in the sidebar."
            )
            mode = st.selectbox(
                "Graph", db.GRAPH_MODES, key="stats-graph",
                help="Delaunay joins each object to the ones it actually abuts, which is "
                     "what segmented nuclei are. A fixed radius misses them in a sparse "
                     "field and joins half the well in a dense one.",
            )
            n_neighs = st.slider(
                "Neighbours (knn)", 3, 20, 6, key="stats-knn", disabled=mode != "knn",
                help="Used by the knn graph only.",
            )
            radius = st.slider(
                "Radius (µm)", 5.0, 200.0, 30.0, key="stats-radius", disabled=mode != "radius",
                help="Used by the fixed-radius graph only.",
            )

        try:
            adata, method, graph = _joined(
                path, mtime, image, limit, grouping, bins, stamp, mode, n_neighs, radius
            )
        except Exception as exc:  # noqa: BLE001 - shown on the page, not in a terminal
            graph_result.error(f"The neighbourhood graph could not be built: {exc}")
            st.stop()

        with graph_result:
            st.caption(
                f"{adata.n_obs:,} object(s)"
                + (f" from {image}" if image else "")
                + f", joined by {graph}, grouped by {method} into {len(groups)} group(s)."
            )

        with st.expander("Neighbourhood enrichment — which groups sit next to which", expanded=True):
            st.markdown(
                "The z-score of how often two groups are neighbours against a null where the "
                "labels are shuffled over the same graph. **Positive** means the two are found "
                "together more than chance; **negative** means they avoid each other. The "
                "diagonal is a group next to itself, which is the usual way a phenotype says "
                "it comes in patches."
            )
            perms = st.slider("Permutations", 50, 1000, 200, step=50, key="stats-perms")
            if st.button("Compute", key="stats-nhood"):
                with st.spinner("Shuffling…"):
                    zscore, _count = db.neighbourhood_enrichment(adata, n_perms=perms)
                figure = _figure(6.0, 5.0)
                axes = figure.add_subplot(111)
                # Diverging and symmetric about zero, so the neutral middle really
                # is "no more than chance" rather than wherever the data's middle
                # happened to fall.
                low_z, high_z = db.diverging_limits(zscore)
                image_plot = axes.imshow(zscore, cmap=db.DIVERGING, vmin=low_z, vmax=high_z)
                axes.set_xticks(range(len(groups)), [str(g) for g in groups], rotation=45, fontsize=7)
                axes.set_yticks(range(len(groups)), [str(g) for g in groups], fontsize=7)
                # The tick labels carry the group's colour, so a cell in this matrix
                # can be traced back to the two clusters in the scatter above.
                for ticks in (axes.get_xticklabels(), axes.get_yticklabels()):
                    for tick, group in zip(ticks, groups):
                        tick.set_color(colours.get(str(group), db.OVERFLOW_COLOUR))
                        tick.set_fontweight("bold")
                figure.colorbar(image_plot, ax=axes, shrink=0.8, label="z-score")
                axes.set_title("Neighbourhood enrichment (z-score)", fontsize=9)
                show(figure)
                st.caption(
                    f"Blue: the two groups are neighbours less often than chance; red: "
                    f"more often. {perms} permutations of the labels over the same graph."
                )

        with st.expander("Ripley's L — clustered in space, or scattered?"):
            st.markdown(
                "How many neighbours each group has within a growing radius, against what "
                "complete randomness would give. **Above** the grey line is clustered, "
                "**below** is more evenly spread than chance."
            )
            if st.button("Compute", key="stats-ripley"):
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
                    # The null is a line, not a group: the neutral grey, which
                    # reads on either theme and is never given to a real group.
                    axes.plot(sims["bins"], sims["stats"], color=db.NEUTRAL, linestyle="--",
                              linewidth=1.5, label="random")
                axes.set_xlabel("radius (µm)", fontsize=8)
                axes.set_ylabel("Ripley's L", fontsize=8)
                axes.set_title("Ripley's L by group", fontsize=9)
                axes.legend(fontsize=7, frameon=False, title="cluster", title_fontsize=7)
                axes.tick_params(labelsize=7)
                show(figure)
                st.caption(
                    "Above the dashed line the group is clustered in space; below it, "
                    "more evenly spread than chance."
                )

        with st.expander("Co-occurrence — how far the company of a group reaches"):
            st.markdown(
                "The chance of finding each group within a distance of another, divided by "
                "its overall share. Above 1 means enriched at that distance; where the curve "
                "falls back to 1 is how far the effect reaches."
            )
            if st.button("Compute", key="stats-cooc"):
                with st.spinner("Walking outwards…"):
                    occ, interval = db.co_occurrence(adata)
                anchor = st.session_state.get("stats-cooc-anchor", groups[0])
                index = groups.index(anchor) if anchor in groups else 0
                figure = _figure(6.0, 4.5)
                axes = figure.add_subplot(111)
                for position, group in enumerate(groups):
                    axes.plot(
                        interval[:-1], occ[index, position, :], label=str(group), linewidth=1.2,
                        color=colours.get(str(group), db.OVERFLOW_COLOUR),
                    )
                axes.axhline(1.0, color=db.NEUTRAL, linestyle="--", linewidth=1.5)
                axes.set_xlabel("distance (µm)", fontsize=8)
                axes.set_ylabel(f"p(group | {anchor}) / p(group)", fontsize=8)
                axes.set_title(f"Co-occurrence with cluster {anchor}", fontsize=9)
                axes.legend(fontsize=7, frameon=False, title="cluster", title_fontsize=7)
                axes.tick_params(labelsize=7)
                show(figure)
                st.caption(
                    "Above 1 the group is enriched at that distance; where the curve "
                    "falls back to 1 is how far the company of the centred group reaches."
                )
                st.selectbox("Centred on", groups, key="stats-cooc-anchor")

        with st.expander("Moran's I — which measurements vary in patches"):
            st.markdown(
                "Spatial autocorrelation per feature, needing no grouping at all. A high "
                "value says neighbouring objects resemble each other in that measurement — "
                "a reporter that comes in patches rather than cell by cell."
            )
            if st.button("Compute", key="stats-moran"):
                with st.spinner("Correlating…"):
                    table = db.spatial_autocorrelation(adata)
                st.caption(
                    "One row per measurement, the most patchy first. Moran's I runs from "
                    "0 (neighbours tell you nothing) towards 1 (neighbours match)."
                )
                st.dataframe(
                    table.head(25).style.format("{:.4g}"), width="stretch"
                )
                values_csv(
                    table.reset_index(), "These values (CSV)", "morans_i.csv", "stats-moran-csv"
                )
                figure = _figure(6.0, 4.5)
                axes = figure.add_subplot(111)
                top = table.head(15).iloc[::-1]
                axes.barh(range(len(top)), top["I"], color=db.NEUTRAL,
                          edgecolor=db.OUTLINE, linewidth=0.8)
                axes.set_yticks(range(len(top)), list(top.index), fontsize=7)
                axes.set_xlabel("Moran's I", fontsize=8)
                axes.set_title("The fifteen most patchy measurements", fontsize=9)
                axes.tick_params(labelsize=7)
                show(figure)
                st.caption(
                    "Grey on purpose: these are measurements, not groups, so the bars "
                    "carry no group colour."
                )

        with st.expander("Centrality — how each group sits in the graph"):
            st.markdown(
                "Graph measures per group: how close its objects are to everything "
                "else, how often paths run through them, and how tightly its own "
                "members are joined to one another."
            )
            if st.button("Compute", key="stats-centrality"):
                with st.spinner("Walking the graph…"):
                    scores = db.centrality(adata)
                st.caption("One row per group, named as the clusters are named elsewhere.")
                st.dataframe(scores.style.format("{:.4g}"), width="stretch")
                values_csv(
                    scores.reset_index(), "These values (CSV)", "centrality.csv",
                    "stats-centrality-csv",
                )

    # -- the way out ----------------------------------------------------------
    with table_tab:
        _table_tab(adata, image, colours)


if __name__ == "__main__":
    main()
