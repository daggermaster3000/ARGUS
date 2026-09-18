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
  genotype or region.
* **Table** of exactly the rows being plotted.

In 2D, each region can be drawn as its own outline instead of a dot, taken from
the *Region outlines* sheet (workbooks written before that sheet existed only
offer dots). Cells can likewise be drawn as their own outlines, from the
``cell_outlines.npz`` the analysis writes beside the workbook.

Run with::

    streamlit run apps/region_explorer.py
    streamlit run apps/region_explorer.py -- path/to/report.xlsx

Needs ``streamlit``, ``plotly``, ``scikit-learn`` and ``umap-learn``.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from microscopy_viewer import analysis_plots as ap  # noqa: E402

SHEET = "Region features"
IDENTITY = ("Sample", "Genotype", "Region", "Label map")

#: Columns that say where an outline is or which way it points, rather than
#: what it is like. Off by default: two fish mounted differently would otherwise
#: separate on mounting.
POSITION = ("Centroid Y (µm)", "Centroid X (µm)", "Orientation (°)")
SHAPE = (
    "Region area (µm²)", "Region volume (µm³)", "Perimeter (µm)", "Circularity",
    "Solidity", "Major axis (µm)", "Minor axis (µm)", "Aspect ratio",
    "Eccentricity", "Bounding height (µm)", "Bounding width (µm)",
)
INTENSITY_STATS = ("Mean", "Median", "SD", "CV", "P5", "P95", "P99", "Integrated")

#: Region colours when colouring by region: the reference categorical order.
REGION_COLORS = ap.GENOTYPE_COLORS + ap.EXTRA_COLORS
SYMBOLS = ("circle", "square", "diamond", "cross", "x", "triangle-up", "triangle-down", "star")

HOVER = ("Objects", "Region area (µm²)", "Objects per mm²")
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
def read_features(source: bytes | str) -> pd.DataFrame:
    handle = io.BytesIO(source) if isinstance(source, bytes) else source
    frame = pd.read_excel(handle, sheet_name=SHEET)
    for column in ("Sample", "Region", "Label map"):
        if column in frame:
            frame[column] = frame[column].astype(str)
    frame["Genotype"] = [ap.genotype_label(value) for value in frame.get("Genotype", "")]
    return frame


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


@st.cache_data(show_spinner="Reading cells…")
def read_cells(source: bytes | str) -> pd.DataFrame:
    """The Objects sheet joined with Cell shapes; empty if either is missing."""
    try:
        objects = pd.read_excel(io.BytesIO(source) if isinstance(source, bytes) else source,
                                sheet_name="Objects")
    except ValueError:
        return pd.DataFrame()
    try:
        shapes = pd.read_excel(io.BytesIO(source) if isinstance(source, bytes) else source,
                               sheet_name="Cell shapes")
    except ValueError:
        shapes = pd.DataFrame(columns=["Sample", "Label"])
    for frame in (objects, shapes):
        frame["Sample"] = frame["Sample"].astype(str)
    shapes = shapes.drop(columns=["Genotype"], errors="ignore")
    cells = objects.merge(shapes, on=["Sample", "Label"], how="left")
    cells["Genotype"] = [ap.genotype_label(value) for value in cells.get("Genotype", "")]
    cells["Region"] = cells.get("Region", "").astype(str)
    return cells


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


#: Cell columns that say where a cell is, not what it is like.
CELL_POSITION = ("Centroid Z (µm)", "Centroid Y (µm)", "Centroid X (µm)", "Orientation (°)")
CELL_SHAPE = (
    "Footprint area (µm²)", "Perimeter (µm)", "Circularity", "Solidity",
    "Major axis (µm)", "Minor axis (µm)", "Aspect ratio", "Eccentricity",
)
CELL_IDENTITY = ("Sample", "Genotype", "Channel", "Region", "Label")
CELL_HOVER = ("Volume (µm³)", "Area (µm²)", "Equivalent diameter (µm)", "Mean intensity",
              "Circularity")


def cell_groups(cells: pd.DataFrame) -> dict[str, list[str]]:
    numeric = [
        c for c in cells.columns
        if c not in CELL_IDENTITY and pd.api.types.is_numeric_dtype(cells[c])
    ]
    shape = [c for c in numeric if c in CELL_SHAPE]
    position = [c for c in numeric if c in CELL_POSITION]
    measured = [c for c in numeric if c not in shape and c not in position]
    return {"Size & intensity": measured, "Cell shape": shape, "Position": position}


@st.cache_data(show_spinner="Embedding cells…")
def embed(matrix: np.ndarray, method: str, dims: int, neighbors: int, min_dist: float, seed: int):
    if method == "PCA":
        from sklearn.decomposition import PCA

        model = PCA(n_components=dims, random_state=seed)
        points = model.fit_transform(matrix)
        return points, [f"PC{i + 1} ({r * 100:.1f}%)" for i, r in enumerate(model.explained_variance_ratio_)]
    return run_umap(matrix, neighbors, min_dist, dims, seed), [f"UMAP {i + 1}" for i in range(dims)]


def numeric_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column for column in frame.columns
        if column not in IDENTITY and pd.api.types.is_numeric_dtype(frame[column])
    ]


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


def color_map(frame: pd.DataFrame, by: str) -> tuple[dict[str, str], list[str]]:
    """Fixed colours per category, shared with the report's PNG figures."""
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


def _alpha(color: str, alpha: float) -> str:
    color = color.lstrip("#")
    r, g, b = (int(color[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


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


def hover_columns(frame: pd.DataFrame) -> dict:
    hover = {"Sample": True, "Genotype": True, "Region": True}
    for column in HOVER:
        if column in frame:
            hover[column] = ":.4g"
    return hover


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

umap_tab, scatter_tab, cells_tab, table_tab = st.tabs(["UMAP", "Scatter", "Cells", "Table"])

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
        color_by = st.selectbox("Colour by", ("Genotype", "Sample", "Region"), key="umap-color")
        symbol_by = st.selectbox("Shape by", ("Region", "Genotype", "None"), key="umap-symbol")
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
        color_by = st.selectbox("Colour by", ("Sample", "Genotype", "Region"), key="sc-color")
        symbol_by = st.selectbox("Shape by", ("Genotype", "Region", "None"), key="sc-symbol")
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
    cells_all = read_cells(source)
    if cells_all.empty:
        st.info("This workbook has no “Objects” sheet — no cells to show.")
    else:
        cell_outlines = read_cell_outlines(cell_source) if cell_source is not None else {}
        cells = cells_all[
            cells_all["Region"].isin(regions)
            & cells_all["Genotype"].isin(genotypes)
            & cells_all["Sample"].isin(samples)
        ].reset_index(drop=True)
        left, right = st.columns([1, 3])
        with left:
            st.subheader("Cells")
            cap = st.slider("Cells to use", 100, max(100, min(50000, len(cells))),
                            min(5000, max(100, len(cells))), 100,
                            help="A random subset, the same for every setting. UMAP on "
                                 "tens of thousands of cells takes a minute.")
            cell_seed = st.number_input("Sampling seed", value=0, step=1, key="cell-seed")
            view = st.radio("View", ("Embedding", "Scatter"), horizontal=True)
            color_cells = st.selectbox("Colour by", ("Sample", "Genotype", "Region"), key="cell-color")
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

        with right:
            if len(cells) > cap:
                subset = cells.sample(n=int(cap), random_state=int(cell_seed)).reset_index(drop=True)
            else:
                subset = cells
            hover = {c: True for c in ("Sample", "Genotype", "Region", "Label") if c in subset}
            hover.update({c: ":.4g" for c in CELL_HOVER if c in subset})
            extra = [c for c in CELL_HOVER if c in subset]
            ident = [c for c in ("Sample", "Genotype", "Region", "Label") if c in subset]
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

# -- table --------------------------------------------------------------------

with table_tab:
    st.dataframe(rows, width="stretch", hide_index=True)
    st.download_button("Download these rows (CSV)", rows.to_csv(index=False),
                       file_name="region_features_filtered.csv")
