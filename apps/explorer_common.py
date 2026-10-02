"""What the two explorer apps share: reading a workbook, and comparing groups.

The statistics live here rather than in each app, so a p value is computed one
way whichever app printed it — :mod:`region_explorer`, the full one, and
:mod:`simple_explorer`, the plain one.

No Streamlit UI beyond the two cached readers; no Qt.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from typing import Sequence  # noqa: E402

from microscopy_viewer import analysis_plots as ap  # noqa: E402


SHEET = "Region features"


def start_path() -> str:
    """The workbook to open first: from the URL, else the command line.

    The desktop launcher passes a dropped workbook as ``?workbook=…``, which
    reaches an explorer that was already running; ``streamlit run app.py -- x``
    still works too.
    """
    try:
        wanted = st.query_params.get("workbook")
    except Exception:
        wanted = None
    if wanted:
        return str(wanted)
    return sys.argv[1] if len(sys.argv) > 1 else ""


IDENTITY = ("Sample", "Genotype", "Region", "Label map")


#: Ink for annotations over the plot.
INK = ap.INK


#: Region colours when colouring by region: the reference categorical order.
REGION_COLORS = ap.GENOTYPE_COLORS + ap.EXTRA_COLORS


SYMBOLS = ("circle", "square", "diamond", "cross", "x", "triangle-up", "triangle-down", "star")


#: Acronym of the whole brain, the area every region is normalised to.
WB = "WB"
#: The whole brain's area, repeated on each of its sample's rows.
WB_AREA = "WB area (µm²)"
#: "Normalize to" choice that leaves the variable as measured.
NOT_NORMALISED = "(nothing)"
#: Sheet of region outlines, one row per vertex.
OUTLINES_SHEET = "Region outlines"
#: Region row of a sample that had no outlines: the whole image, not a region.
NO_REGIONS = "(no regions stored)"


@st.cache_data(show_spinner=False)
def read_features(source: bytes | str) -> pd.DataFrame:
    handle = io.BytesIO(source) if isinstance(source, bytes) else source
    frame = pd.read_excel(handle, sheet_name=SHEET)
    for column in ("Sample", "Region", "Label map"):
        if column in frame:
            frame[column] = frame[column].astype(str)
    frame["Genotype"] = [ap.genotype_label(value) for value in frame.get("Genotype", "")]
    try:
        outlines = pd.read_excel(io.BytesIO(source) if isinstance(source, bytes) else source,
                                 sheet_name=OUTLINES_SHEET)
    except ValueError:  # workbooks from before the outlines were saved
        outlines = pd.DataFrame()
    return fill_groups(add_whole_brain(frame, outlines))


def fill_groups(frame: pd.DataFrame) -> pd.DataFrame:
    """Condition columns with their blanks named, so a plot gets a "(none)" group."""
    for column in group_columns(frame):
        if column != "Genotype":
            frame[column] = frame[column].fillna(NO_VALUE).astype(str).replace({"": NO_VALUE})
    return frame


def add_whole_brain(features: pd.DataFrame, outlines: pd.DataFrame | None = None) -> pd.DataFrame:
    """Add each sample's whole-brain (WB) area to every one of its rows.

    It is there to normalise to (see :func:`normalise`). The whole brain is, in
    order of preference:

    * a region drawn and named ``WB`` — the brain as the person outlining it saw it;
    * the union of the sample's outlines, so nested or overlapping regions (a
      nucleus drawn inside its lobe) are not counted twice;
    * the sum of the region areas, for workbooks written without outlines.
    """
    area = "Region area (µm²)"
    frame = features.copy()
    if frame.empty or area not in frame or "Sample" not in frame:
        return frame
    drawn = frame[frame["Region"] != NO_REGIONS] if "Region" in frame else frame
    union = _outline_union_areas(outlines) if outlines is not None and not outlines.empty else {}
    wb_area: dict[str, float] = {}
    for sample, rows in drawn.groupby("Sample", sort=False):
        explicit = rows.loc[rows["Region"] == WB, area].dropna() if "Region" in rows else []
        if len(explicit):
            wb_area[sample] = float(explicit.iloc[0])
        elif str(sample) in union:
            wb_area[sample] = union[str(sample)]
        else:
            wb_area[sample] = float(rows[area].sum(min_count=1))
    frame[WB_AREA] = frame["Sample"].map(wb_area).astype(float)
    if "Region" in frame:
        frame.loc[frame["Region"] == NO_REGIONS, WB_AREA] = np.nan
    return frame


#: Columns that name a row rather than group samples.
NOT_GROUPS = {"Sample", "Region", "Label map", "Channel", "Label", "Cluster", "Part", "Vertex"}
#: Joins the two labels of a combined group, "wt · DMSO".
JOIN = " · "
#: A sample with no value in a group column.
NO_VALUE = "(none)"


def group_columns(features: pd.DataFrame) -> list[str]:
    """Genotype, then every condition column the analysis wrote (text, per sample)."""
    found = ["Genotype"] if "Genotype" in features else []
    for column in features.columns:
        if column in NOT_GROUPS or column in found or pd.api.types.is_numeric_dtype(features[column]):
            continue
        # A group labels a whole sample: one value on every row of it.
        if (features.groupby("Sample")[column].nunique(dropna=False) <= 1).all():
            found.append(column)
    return found


def group_choices(features: pd.DataFrame) -> list[str]:
    """What the samples can be grouped by: each column, and genotype × each condition."""
    columns = group_columns(features)
    combined = [f"Genotype × {c}" for c in columns if c != "Genotype"] if "Genotype" in columns else []
    return columns + combined


def with_group(frame: pd.DataFrame, features: pd.DataFrame, choice: str) -> tuple[pd.DataFrame, str]:
    """*frame* carrying the column *choice* groups by; returns it and that column's name.

    Group columns missing from *frame* (a table built per sample, say) are
    looked up by sample in *features*. A combined choice gets a column of its
    own, "wt · DMSO".
    """
    parts = [p.strip() for p in choice.split("×")]
    frame = frame.copy()
    for column in parts:
        if column not in frame and column in features:
            lookup = features.drop_duplicates("Sample").set_index("Sample")[column]
            frame[column] = frame["Sample"].map(lookup)
        if column in frame and column != "Genotype":
            frame[column] = frame[column].fillna(NO_VALUE).astype(str).replace({"": NO_VALUE, "nan": NO_VALUE})
    if len(parts) == 1:
        return frame, parts[0]
    frame[choice] = [JOIN.join(str(v) for v in values) for values in zip(*(frame[p] for p in parts))]
    return frame, choice


def group_order(values) -> list[str]:
    """Groups in plotting order: reference groups first; combined ones condition by condition.

    ``wt · DMSO, mut · DMSO, wt · drug, mut · drug`` — the genotypes side by
    side inside each condition, which is the comparison usually wanted.
    """
    def ordered(labels):
        # Samples without a value come last, like an unknown genotype.
        order = ap.genotype_order(labels)
        return [g for g in order if g != NO_VALUE] + [g for g in order if g == NO_VALUE]

    found = [str(v) for v in dict.fromkeys(values) if pd.notna(v)]
    if found and all(JOIN in v for v in found):
        split = [v.split(JOIN, 1) for v in found]
        first = ordered([a for a, _ in split])
        second = ordered([b for _, b in split])
        return [f"{a}{JOIN}{b}" for b in second for a in first if f"{a}{JOIN}{b}" in found]
    return ordered(found)


def group_colors(order: Sequence[str], column: str = "Genotype") -> dict[str, str]:
    """A colour per group of *column*, the user's choice where they made one.

    Combined groups ("wt · DMSO") take their genotype's colour.
    """
    if order and all(JOIN in g for g in order):
        genotypes = genotype_palette([g.split(JOIN, 1)[0] for g in order])
        return {g: genotypes[g.split(JOIN, 1)[0]] for g in order}
    styles = ap.genotype_styles(list(order))
    return paint(column, {g: styles[g][0] for g in order})


def genotype_palette(values) -> dict[str, str]:
    """Genotype -> colour: the report's fixed colours, then the user's choices."""
    order = ap.genotype_order(values)
    return paint("Genotype", {g: c for g, (c, _m) in ap.genotype_styles(order).items()})


def category_colors(frame: pd.DataFrame, by: str) -> tuple[dict[str, str], list[str]]:
    """``(colours, order)`` for colouring *frame* by the column *by*.

    Genotypes keep the report's colours, samples are shades of their genotype,
    regions follow the order they were drawn in, conditions the group order the
    Compare plot uses — and the user's picks win in every case.
    """
    if by == "Genotype":
        order = ap.genotype_order(frame["Genotype"])
        return genotype_palette(frame["Genotype"]), order
    if by == "Sample":
        genotype_of = dict(zip(frame["Sample"], frame["Genotype"]))
        colors = paint("Sample", ap.sample_colors(list(dict.fromkeys(frame["Sample"])), genotype_of))
        return colors, list(colors)
    if by == "Region":
        order = [str(v) for v in dict.fromkeys(frame[by])]
        return paint("Region", {v: REGION_COLORS[i % len(REGION_COLORS)] for i, v in enumerate(order)}), order
    order = group_order(frame[by])
    return group_colors(order, by), order


# -- the user's colours ---------------------------------------------------------

#: Session key of the user's colours: ``kind -> value -> "#rrggbb"``.
COLORS_KEY = "group-colors"


def _colors_file() -> Path:
    from microscopy_viewer.runtime import app_data_dir

    return app_data_dir() / "explorer_colors.json"


def chosen_colors() -> dict[str, dict[str, str]]:
    """The colours the user picked, by kind ("Genotype", "Cluster", a condition…).

    Loaded once per session from the file they are remembered in, so ``wt`` keeps
    the colour chosen for it across workbooks and days.
    """
    if COLORS_KEY not in st.session_state:
        import json

        try:
            stored = json.loads(_colors_file().read_text(encoding="utf-8"))
            stored = {str(k): {str(v): str(c) for v, c in d.items()} for k, d in stored.items()}
        except (OSError, ValueError, AttributeError):
            stored = {}
        st.session_state[COLORS_KEY] = stored
    return st.session_state[COLORS_KEY]


def _remember(colors: dict[str, dict[str, str]]) -> None:
    import json

    try:
        target = _colors_file()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(colors, indent=1), encoding="utf-8")
    except OSError:
        pass


def paint(kind: str, mapping: dict) -> dict:
    """*mapping* (value -> colour) with the user's colours for *kind* laid over it.

    Every palette in the apps goes through here, which is what makes a colour
    picked once apply to every plot.
    """
    try:
        mine = chosen_colors().get(kind, {})
    except Exception:  # outside a Streamlit run (tests, scripts)
        mine = {}
    return {value: mine.get(str(value), color) for value, color in mapping.items()}


def color_controls(kinds: dict[str, list], defaults) -> None:
    """Sidebar pickers: one colour per value of the chosen kind.

    *kinds* maps a kind to its values; *defaults(kind, values)* returns the colours
    the apps would use without any choice, so each picker starts there.
    """
    kinds = {k: [str(v) for v in values] for k, values in kinds.items() if len(values)}
    if not kinds:
        return
    with st.expander("Colours", expanded=False):
        kind = st.selectbox("Colour the", list(kinds), key="colors-kind",
                            help="Pick a colour for each group. It is used by every plot, "
                                 "and remembered for next time.")
        values = kinds[kind]
        colors = chosen_colors()
        start = defaults(kind, values)
        mine = dict(colors.get(kind, {}))
        changed = False
        for value in values[:40]:
            current = mine.get(value, start.get(value, "#888888"))
            if not str(current).startswith("#"):
                current = _hex(current)
            picked = st.color_picker(value, current, key=f"color-{kind}-{value}")
            if picked.lower() != str(start.get(value, "")).lower() or value in mine:
                if mine.get(value) != picked:
                    mine[value] = picked
                    changed = True
        if len(values) > 40:
            st.caption(f"First 40 of {len(values)} shown.")
        if st.button("Reset these colours", key=f"colors-reset-{kind}"):
            mine = {}
            changed = True
            for value in values:
                st.session_state.pop(f"color-{kind}-{value}", None)
        if changed:
            colors[kind] = mine
            _remember(colors)
            st.rerun()


def default_colors(kind: str, values, features: pd.DataFrame | None = None) -> dict[str, str]:
    """The colours *values* of *kind* get before the user picks any."""
    values = [str(v) for v in values]
    if kind == "Genotype":
        return {g: c for g, (c, _m) in ap.genotype_styles(ap.genotype_order(values)).items()}
    if kind == "Region":
        return {v: REGION_COLORS[i % len(REGION_COLORS)] for i, v in enumerate(values)}
    if kind == "Cluster":
        return {v: REGION_COLORS[(int(v[1:]) - 1) % len(REGION_COLORS)] for v in values
                if v[1:].isdigit()}
    if kind == "Sample" and features is not None:
        genotype_of = dict(zip(features["Sample"].astype(str), features["Genotype"]))
        return ap.sample_colors(values, genotype_of)
    order = group_order(values)
    styles = ap.genotype_styles(order)
    return {g: styles[g][0] for g in order}


def sidebar_colors(features: pd.DataFrame, clusters: int = 0) -> None:
    """The Colours expander for an app: genotype, every condition, regions, samples, clusters."""
    kinds: dict[str, list] = {"Genotype": ap.genotype_order(features["Genotype"])}
    for column in group_columns(features)[1:]:
        kinds[column] = group_order(features[column])
    kinds["Region"] = list(dict.fromkeys(features["Region"])) if "Region" in features else []
    if clusters:
        kinds["Cluster"] = [f"C{i}" for i in range(1, clusters + 1)]
    kinds["Sample"] = list(dict.fromkeys(features["Sample"]))
    color_controls(kinds, lambda kind, values: default_colors(kind, values, features))


def _hex(color: str) -> str:
    """A colour Streamlit's picker accepts, from a name or rgb() string."""
    try:
        from matplotlib.colors import to_hex

        return to_hex(color)
    except Exception:
        return "#888888"


def normalise_choices(numbers: Sequence[str], variable: str | None) -> list[str]:
    """What a variable can be divided by: nothing, the whole brain, or any other number."""
    others = [c for c in numbers if c != variable]
    first = [WB_AREA] if WB_AREA in others else []
    return [NOT_NORMALISED, *first, *(c for c in others if c not in first)]


def normalise(frame: pd.DataFrame, variable: str, by: str) -> tuple[pd.DataFrame, str]:
    """*variable* divided by *by*, row by row, as a new column; returns it and its name.

    A zero denominator gives no value rather than an infinite one.
    """
    if by == NOT_NORMALISED or by not in frame:
        return frame, variable
    column = f"{variable} / {by}"
    denominator = frame[by].astype(float).where(frame[by] != 0)
    return frame.assign(**{column: frame[variable].astype(float) / denominator}), column


def _outline_union_areas(outlines: pd.DataFrame, resolution: int = 2000) -> dict[str, float]:
    """Area (µm²) covered by any of each sample's outlines.

    Rasterised on a grid of at most *resolution* pixels along the brain's longer
    side, which puts the error well under a tenth of a percent for a brain.
    """
    from microscopy_viewer.intensity import polygon_mask

    needed = {"Sample", "Region", "Part", "Vertex", "Y (µm)", "X (µm)"}
    if not needed <= set(outlines.columns):
        return {}
    areas: dict[str, float] = {}
    ordered = outlines.sort_values(["Sample", "Region", "Part", "Vertex"])
    for sample, rows in ordered.groupby("Sample", sort=False):
        points = rows[["Y (µm)", "X (µm)"]].to_numpy(dtype=float)
        if len(points) < 3:
            continue
        origin = points.min(axis=0)
        extent = points.max(axis=0) - origin
        step = max(float(extent.max()) / resolution, 1e-9)
        shape = tuple(int(np.ceil(n / step)) + 2 for n in extent)
        covered = np.zeros(shape, dtype=bool)
        for _, part in rows.groupby(["Region", "Part"], sort=False):
            vertices = (part[["Y (µm)", "X (µm)"]].to_numpy(dtype=float) - origin) / step
            covered |= polygon_mask(vertices, shape)
        areas[str(sample)] = float(covered.sum()) * step * step
    return areas


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
    keys = ["Sample", "Label"]
    # The labels of the sample (genotype, conditions) are on every sheet; keep
    # the Objects sheet's copy rather than merging them in as _x / _y.
    shapes = shapes.drop(columns=[c for c in shapes.columns if c in objects and c not in keys])
    cells = objects.merge(shapes, on=keys, how="left")
    try:
        channels = pd.read_excel(io.BytesIO(source) if isinstance(source, bytes) else source,
                                 sheet_name=CELL_INTENSITIES)
    except ValueError:  # workbooks from before every channel was measured per cell
        channels = pd.DataFrame()
    if not channels.empty:
        channels["Sample"] = channels["Sample"].astype(str)
        channels = channels.drop(columns=[c for c in channels.columns if c in cells and c not in keys])
        cells = cells.merge(channels, on=keys, how="left")
    cells["Genotype"] = [ap.genotype_label(value) for value in cells.get("Genotype", "")]
    cells["Region"] = cells.get("Region", "").astype(str)
    return fill_groups(cells)


#: Sheet of every channel's intensity inside every cell.
CELL_INTENSITIES = "Cell intensities"


CELL_CHANNEL_STATS = (" Mean", " SD", " Max", " Integrated")


#: Numbers put in the hover of a region's point.
HOVER = ("Objects", "Region area (µm²)", "Objects per mm²")

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
    channels = [c for c in numeric if c.endswith(CELL_CHANNEL_STATS) and c not in shape]
    measured = [c for c in numeric if c not in shape and c not in position and c not in channels]
    return {"Size & intensity": measured, "Channel intensities": channels,
            "Cell shape": shape, "Position": position}


#: Tried first as the variable to plot, in order.
PREFERRED = (
    "Objects per mm²", "Region area (µm²)", "Equivalent diameter (µm)", "Area (µm²)",
    "Mean intensity", "Circularity",
)


def plottable(frame: pd.DataFrame, exclude: Sequence[str] = ()) -> list[str]:
    """Numeric columns worth offering: ones that actually vary.

    A column that is the same everywhere (an all-zero Z centroid, say) has no
    comparison in it and makes the tests degenerate, so it is left out.
    """
    out = []
    for column in numeric_columns(frame):
        if column in exclude:
            continue
        values = frame[column].astype(float)
        if values.notna().any() and float(np.nanmax(values) - np.nanmin(values)) != 0:
            out.append(column)
    return out


def default_variable(columns: Sequence[str], skip: Sequence[str] = ()) -> int:
    """Index of the first of :data:`PREFERRED` present, else the first column."""
    for wanted in PREFERRED:
        if wanted in columns and wanted not in skip:
            return list(columns).index(wanted)
    for index, column in enumerate(columns):
        if column not in skip:
            return index
    return 0


def numeric_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column for column in frame.columns
        if column not in IDENTITY and pd.api.types.is_numeric_dtype(frame[column])
    ]


TESTS = ("One-way ANOVA", "Welch's ANOVA", "Kruskal-Wallis")


#: Tests over individual cells. Cells of one fish are not independent — take
#: twice as many cells from the same six fish and a plain ANOVA's p value
#: collapses, though nothing was learnt about fish. Either summarise each
#: sample first, or model the sample as a random effect.
CELL_TESTS = ("Mixed model (sample as random effect)",
              "Sample means · One-way ANOVA",
              "Sample means · Welch's ANOVA",
              "Sample means · Kruskal-Wallis")


MIXED = CELL_TESTS[0]


def _mixed_fit(frame: pd.DataFrame, value: str, group: str, unit: str, formula: str):
    import statsmodels.formula.api as smf

    data = pd.DataFrame({
        "y": frame[value].astype(float).to_numpy(),
        "g": frame[group].astype(str).to_numpy(),
        "u": frame[unit].astype(str).to_numpy(),
    }).dropna()
    # Standardised while fitting: intensities in the millions and diameters in
    # micrometres otherwise need different optimiser settings.
    spread = float(data["y"].std()) or 1.0
    data["y"] = (data["y"] - data["y"].mean()) / spread
    centre = float(frame[value].astype(float).mean())
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit = smf.mixedlm(formula, data, groups=data["u"]).fit(reml=False, method="lbfgs")
    shaky = any("converge" in str(w.message).lower() or "singular" in str(w.message).lower()
                for w in caught)
    return fit, data, spread, centre, shaky


def mixed_model_test(frame: pd.DataFrame, value: str, group: str, unit: str = "Sample") -> dict:
    """Does *group* shift *value*, with the cells of one sample kept together?

    A linear mixed model ``value ~ genotype`` with a random intercept per
    sample: the cells of a fish are allowed to be alike, so the comparison is
    driven by how many fish there are, not how many cells were segmented.
    Genotype is tested by likelihood ratio against the same model without it.
    """
    from scipy import stats

    out = {"test": MIXED, "statistic": float("nan"), "p": float("nan"),
           "effect": float("nan"), "effect_name": "ICC", "notes": []}
    groups = [g for g, rows in frame.groupby(group) if len(rows) >= 2]
    units = frame[unit].nunique()
    if len(groups) < 2 or units < 3:
        out["notes"].append("Needs two groups and at least three samples.")
        return out
    try:
        full, data, spread, centre, shaky = _mixed_fit(frame, value, group, unit, "y ~ C(g)")
        null = _mixed_fit(frame, value, group, unit, "y ~ 1")[0]
    except Exception as exc:
        out["notes"].append(f"The mixed model did not fit: {exc}")
        return out
    # Never below zero: the optimiser can end a hair short on the fuller model,
    # which means the same thing as no improvement at all.
    ratio = max(float(2 * (full.llf - null.llf)), 0.0)
    degrees = int(len(groups) - 1)
    out["statistic"] = ratio
    out["p"] = float(stats.chi2.sf(ratio, degrees))
    between = float(np.asarray(full.cov_re).ravel()[0])
    out["effect"] = float(between / (between + float(full.scale))) if between + full.scale > 0 else float("nan")
    reference = str(sorted(frame[group].astype(str).unique())[0])
    out["table"] = mixed_table(full, spread, centre, reference)
    out["notes"].append(
        f"Mixed model on {len(data):,} cells from {units} samples: the group as a fixed "
        f"effect, sample as a random intercept, likelihood-ratio χ²({degrees}). "
        "ICC is how much of the variation is between samples rather than within them."
    )
    if shaky:
        out["notes"].append(
            "The model did not settle cleanly — usually the samples barely differ from "
            "one another, which makes the random effect hard to estimate. Compare it "
            "with the sample-means test."
        )
    if units < 6:
        out["notes"].append("Fewer than six samples: the p value is approximate.")
    return out


def anova_table(values: dict[str, np.ndarray], test: str, result: dict) -> pd.DataFrame:
    """The table the test would be written up as: sources, df, and the statistic."""
    groups = {k: np.asarray(v, dtype=float) for k, v in values.items() if len(v) >= 1}
    if len(groups) < 2 or not np.isfinite(result.get("p", float("nan"))):
        return pd.DataFrame()
    total_n = sum(len(v) for v in groups.values())
    k = len(groups)
    if test == "Kruskal-Wallis":
        return pd.DataFrame([
            {"Source": "Genotype", "df": k - 1, "H": result["statistic"], "p": result["p"]},
            {"Source": "Residual", "df": total_n - k, "H": np.nan, "p": np.nan},
        ])
    if test == "Welch's ANOVA":
        from statsmodels.stats.oneway import anova_oneway

        fit = anova_oneway(list(groups.values()), use_var="unequal", welch_correction=True)
        return pd.DataFrame([
            {"Source": "Genotype (Welch)", "df": float(fit.df_num), "F": result["statistic"],
             "p": result["p"]},
            {"Source": "Residual (adjusted)", "df": float(fit.df_denom), "F": np.nan, "p": np.nan},
        ])
    grand = np.concatenate(list(groups.values()))
    between = float(sum(len(v) * (v.mean() - grand.mean()) ** 2 for v in groups.values()))
    within = float(sum(((v - v.mean()) ** 2).sum() for v in groups.values()))
    rows = [
        {"Source": "Genotype (between)", "SS": between, "df": k - 1, "MS": between / (k - 1),
         "F": result["statistic"], "p": result["p"]},
        {"Source": "Residual (within)", "SS": within, "df": total_n - k,
         "MS": within / (total_n - k) if total_n > k else np.nan, "F": np.nan, "p": np.nan},
        {"Source": "Total", "SS": between + within, "df": total_n - 1, "MS": np.nan,
         "F": np.nan, "p": np.nan},
    ]
    return pd.DataFrame(rows)


def mixed_table(fit, spread: float, centre: float, reference: str) -> pd.DataFrame:
    """The mixed model written out: each genotype's shift, and the variances.

    Coefficients come back in the variable's own units — the fit is done on
    standardised values so that intensities and micrometres need the same
    optimiser settings.
    """
    rows = []
    for term in fit.params.index:
        if term == "Group Var":
            continue
        name = ("Intercept (" + reference + ")" if term == "Intercept"
                else term.replace("C(g)[T.", "").replace("]", "") + " − " + reference)
        # Undo the standardising: the intercept also gets the mean back, the
        # differences between genotypes only the scale.
        estimate = float(fit.params[term]) * spread + (centre if term == "Intercept" else 0.0)
        intercept = term == "Intercept"
        rows.append({
            "Term": name,
            "Estimate": estimate,
            "SE": float(fit.bse[term]) * spread,
            # The intercept's own z and p test it against zero on the
            # standardised scale, which means nothing here.
            "z": np.nan if intercept else float(fit.tvalues[term]),
            "p": np.nan if intercept else float(fit.pvalues[term]),
        })
    between = float(np.asarray(fit.cov_re).ravel()[0]) * spread**2
    residual = float(fit.scale) * spread**2
    rows.append({"Term": "Variance between samples", "Estimate": between, "SE": np.nan,
                 "z": np.nan, "p": np.nan})
    rows.append({"Term": "Variance within samples", "Estimate": residual, "SE": np.nan,
                 "z": np.nan, "p": np.nan})
    return pd.DataFrame(rows)


def mixed_pairs(frame: pd.DataFrame, value: str, group: str, unit: str = "Sample") -> pd.DataFrame:
    """Every pair of groups from the same mixed model, Holm-corrected."""
    names = [g for g, rows in frame.groupby(group) if len(rows) >= 2]
    if len(names) < 3:
        return pd.DataFrame()
    rows, raw = [], []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            pair = frame[frame[group].isin([a, b])]
            try:
                fit = _mixed_fit(pair, value, group, unit, "y ~ C(g)")[0]
                term = next(k for k in fit.params.index if k.startswith("C(g)"))
                p = float(fit.pvalues[term])
            except Exception:
                p = float("nan")
            raw.append(p)
            rows.append({"Group 1": a, "Group 2": b})
    order = np.argsort([1.0 if not np.isfinite(p) else p for p in raw])
    adjusted, running = np.empty(len(raw)), 0.0
    for rank, index in enumerate(order):  # Holm
        running = max(running, (len(raw) - rank) * (raw[index] if np.isfinite(raw[index]) else 1.0))
        adjusted[index] = min(running, 1.0)
    return pd.DataFrame(rows).assign(p=adjusted)


def compare_groups(values: dict[str, np.ndarray], test: str) -> dict:
    """Run *test* over the groups, with an effect size and the assumption checks.

    Returns statistic, p, effect size and notes; ``p`` is NaN when a test
    cannot be run (a group with fewer than two values, say).
    """
    from scipy import stats

    groups = [np.asarray(v, dtype=float) for v in values.values() if len(v) >= 1]
    usable = [g for g in groups if len(g) >= 2]
    out = {"test": test, "statistic": float("nan"), "p": float("nan"),
           "effect": float("nan"), "effect_name": "η²", "notes": []}
    if len(groups) < 2 or len(usable) < 2:
        out["notes"].append("Needs two groups with at least two values each.")
        return out
    if np.ptp(np.concatenate(groups)) == 0:
        out["notes"].append("Every value is the same, so there is nothing to compare.")
        return out
    try:
        if test == "Kruskal-Wallis":
            statistic, p = stats.kruskal(*groups)
            total = sum(len(g) for g in groups)
            out["effect_name"] = "ε²"
            out["effect"] = (max(float((statistic - len(groups) + 1) / (total - len(groups))), 0.0)
                             if total > len(groups) else float("nan"))
        elif test == "Welch's ANOVA":
            from statsmodels.stats.oneway import anova_oneway

            fit = anova_oneway(groups, use_var="unequal", welch_correction=True)
            statistic, p = float(fit.statistic), float(fit.pvalue)
        else:
            statistic, p = stats.f_oneway(*groups)
    except ValueError as exc:  # constant or degenerate input
        out["notes"].append(f"The test could not run: {exc}")
        return out
    out["statistic"], out["p"] = float(statistic), float(p)
    if test != "Kruskal-Wallis":
        grand = np.concatenate(groups)
        between = sum(len(g) * (g.mean() - grand.mean()) ** 2 for g in groups)
        total_ss = float(((grand - grand.mean()) ** 2).sum())
        out["effect"] = float(between / total_ss) if total_ss > 0 else float("nan")
        spread = stats.levene(*usable).pvalue if len(usable) > 1 else float("nan")
        if np.isfinite(spread) and spread < 0.05:
            out["notes"].append(
                f"Levene p = {spread:.3g}: the groups' spreads differ, so prefer Welch's ANOVA."
            )
        residuals = np.concatenate([g - g.mean() for g in usable])
        if 3 <= len(residuals) <= 5000:
            normal = float(stats.shapiro(residuals).pvalue)
            if normal < 0.05:
                out["notes"].append(
                    f"Shapiro-Wilk p = {normal:.3g} on the residuals: not very normal; "
                    "Kruskal-Wallis makes fewer assumptions."
                )
    if min(len(g) for g in groups) < 3:
        out["notes"].append("A group with fewer than three values: read the p value loosely.")
    return out


def posthoc_pairs(values: dict[str, np.ndarray], test: str) -> pd.DataFrame:
    """Every pair of groups, Tukey-corrected (Dunn-style ranks for Kruskal-Wallis)."""
    usable = {k: np.asarray(v, dtype=float) for k, v in values.items() if len(v) >= 2}
    if len(usable) < 3:
        return pd.DataFrame()
    if test == "Kruskal-Wallis":
        from scipy import stats

        ranked = {k: v for k, v in usable.items()}
        rows = []
        names = list(ranked)
        raw = []
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                p = float(stats.mannwhitneyu(ranked[a], ranked[b]).pvalue)
                raw.append(p)
                rows.append({"Group 1": a, "Group 2": b, "p (raw)": p})
        order = np.argsort(raw)
        adjusted = np.empty(len(raw))
        running = 0.0
        for rank, index in enumerate(order):  # Holm
            running = max(running, (len(raw) - rank) * raw[index])
            adjusted[index] = min(running, 1.0)
        frame = pd.DataFrame(rows)
        frame["p (Holm)"] = adjusted
        return frame.drop(columns=["p (raw)"]).rename(columns={"p (Holm)": "p"})
    from statsmodels.stats.multicomp import pairwise_tukeyhsd

    labels = np.concatenate([[k] * len(v) for k, v in usable.items()])
    data = np.concatenate(list(usable.values()))
    result = pairwise_tukeyhsd(data, labels)
    frame = pd.DataFrame(result.summary().data[1:], columns=result.summary().data[0])
    return frame.rename(columns={"group1": "Group 1", "group2": "Group 2",
                                 "p-adj": "p", "meandiff": "Difference"})[
        ["Group 1", "Group 2", "Difference", "p"]
    ]


def stars(p: float) -> str:
    if not np.isfinite(p):
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


def comparison_figure(frame: pd.DataFrame, value: str, group: str, kind: str, title: str,
                      dot_label: str = "Sample", pairs: pd.DataFrame | None = None,
                      show_dots: bool = True, bars: str = "Significant only",
                      bar_label: str = "Stars"):
    """Box or violin per group, every row a dot, pairs joined by significance bars.

    *pairs* holds ``Group 1``, ``Group 2`` and ``p`` — the post-hoc pairs, or
    the one comparison when there are two groups. *bars* is "Significant only",
    "All pairs" or "Off"; *bar_label* "Stars" or "p value".
    """
    order = group_order(frame[group])
    colors = group_colors(order, group)
    figure = go.Figure()
    for position, name in enumerate(order):
        values = frame.loc[frame[group] == name, value].astype(float).dropna()
        if values.empty:
            continue
        color = colors[name]
        # Numeric positions throughout, so the dots can be jittered and the
        # significance brackets drawn between two of them.
        shared = dict(x=[position] * len(values), y=values, name=str(name),
                      legendgroup=str(name), width=0.5,
                      marker={"color": color}, line={"color": color}, showlegend=False)
        if kind == "Violin":
            figure.add_trace(go.Violin(points=False, box_visible=False, meanline_visible=True,
                                       fillcolor=_alpha(color, 0.6), opacity=1.0,
                                       hoverinfo="skip", **shared))
        else:
            figure.add_trace(go.Box(boxpoints=False, fillcolor=_alpha(color, 0.6),
                                    hoverinfo="skip", **shared))
    if show_dots:
        for position, name in enumerate(order):
            rows = frame[frame[group] == name].dropna(subset=[value])
            if rows.empty:
                continue
            jitter = np.random.default_rng(0).uniform(-0.12, 0.12, len(rows))
            figure.add_trace(go.Scatter(
                x=position + jitter, y=rows[value].astype(float), mode="markers",
                showlegend=False,
                text=rows[dot_label] if dot_label in rows else [str(name)] * len(rows),
                marker={"size": 9, "color": colors[name], "opacity": 0.9,
                        "line": {"width": 1.5, "color": "white"}},
                hovertemplate="%{text}<br>%{y:.4g}<extra>" + str(name) + "</extra>",
            ))
    top = float(frame[value].astype(float).max())
    bottom = float(frame[value].astype(float).min())
    span = (top - bottom) or 1.0
    highest = top
    if pairs is not None and not pairs.empty and bars != "Off":
        wanted = pairs.copy()
        wanted["_gap"] = [abs(order.index(str(a)) - order.index(str(b)))
                          for a, b in zip(wanted["Group 1"], wanted["Group 2"])]
        if bars == "Significant only":
            wanted = wanted[wanted["p"].astype(float) < 0.05]
        # Neighbours first, so short bars sit under the long ones that span them.
        wanted = wanted.sort_values(["_gap", "p"]).head(8)
        cap = span * 0.02
        for drawn, (_, row) in enumerate(wanted.iterrows()):
            a, b = order.index(str(row["Group 1"])), order.index(str(row["Group 2"]))
            height = top + span * (0.10 + 0.11 * drawn)
            highest = max(highest, height)
            figure.add_shape(
                type="path", line={"color": NEUTRAL, "width": 1.5},
                path=(f"M {a},{height - cap} L {a},{height} L {b},{height} L {b},{height - cap}"),
            )
            p = float(row["p"])
            text = stars(p) if bar_label == "Stars" else (f"p = {p:.3g}" if p >= 1e-4 else f"p = {p:.1e}")
            figure.add_annotation(x=(a + b) / 2, y=height, text=text, showarrow=False,
                                  yshift=9, font={"size": 12, "color": INK})
        if len(wanted):
            figure.update_yaxes(range=[bottom - span * 0.08, highest + span * 0.12])
    figure.update_layout(
        title=title, xaxis={"title": group, "tickmode": "array",
                            "tickvals": list(range(len(order))), "ticktext": order,
                            "range": [-0.6, len(order) - 0.4], "automargin": True},
        yaxis={"title": value, "automargin": True}, height=520,
        margin={"l": 10, "r": 10, "t": 60, "b": 10}, showlegend=False,
    )
    return figure


NEUTRAL = "#8a8983"

#: Outline of every mark in every plot: violins, boxes, bars, dots, shapes.
OUTLINE = "#000000"


def outlined(figure):
    """*figure* with the house style: every mark outlined in black.

    Violins and boxes keep their group's fill with a black edge; bars and dots get
    a black rim; filled shapes (region and cell outlines) a thin black edge. Lines
    that are lines — trends, templates, contours — and heatmaps are left alone.
    Applied to every chart through :func:`chart`, so no plot can miss it.
    """
    for trace in figure.data:
        kind = trace.type
        if kind in ("violin", "box"):
            trace.update(line={"color": OUTLINE, "width": 1.5})
            if kind == "violin":
                trace.update(meanline={"color": OUTLINE})
        elif kind in ("bar", "histogram"):
            trace.update(marker_line_color=OUTLINE, marker_line_width=1)
        elif kind in ("scatter", "scattergl", "scatter3d"):
            fill = getattr(trace, "fill", None)
            if fill not in (None, "none", ""):
                trace.update(line={"color": OUTLINE, "width": 0.8})
            elif "markers" in (trace.mode or "markers"):
                count = len(trace.x) if trace.x is not None else 0
                trace.update(marker_line_color=OUTLINE,
                             marker_line_width=0.5 if count > 2000 else 1)
    return figure


def chart(figure, **kwargs):
    """``st.plotly_chart`` with the house style applied."""
    return st.plotly_chart(outlined(figure), **kwargs)


def genotype_of_sample(features: pd.DataFrame, sample: str) -> str:
    match = features.loc[features["Sample"] == sample, "Genotype"]
    return str(match.iloc[0]) if len(match) else ap.UNKNOWN_GENOTYPE


def _alpha(color: str, alpha: float) -> str:
    color = color.lstrip("#")
    r, g, b = (int(color[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def hover_columns(frame: pd.DataFrame) -> dict:
    hover = {"Sample": True, "Genotype": True, "Region": True}
    for column in HOVER:
        if column in frame:
            hover[column] = ":.4g"
    return hover


# -- embeddings ---------------------------------------------------------------

#: Fewest rows an embedding is drawn for: two points have no shape to show.
MIN_EMBED_ROWS = 3


def embed_umap(matrix: np.ndarray, neighbors: int, min_dist: float, dims: int,
               seed: int) -> tuple[np.ndarray, list[str], str]:
    """UMAP of *matrix*, or PCA when UMAP cannot run on it.

    A handful of rows — three samples, say — breaks UMAP's spectral start
    ("k >= N"), so small sets start from random positions instead, and if UMAP
    still fails the rows are placed by PCA. Returns the points (*dims* columns),
    the axis names and a note saying what was done instead ("" when UMAP ran).
    """
    import warnings

    matrix = np.asarray(matrix, dtype=float)
    rows = len(matrix)
    reason = ""
    try:
        import umap

        reducer = umap.UMAP(
            n_neighbors=max(2, min(int(neighbors), rows - 1)), min_dist=min_dist,
            n_components=dims, random_state=seed,
            # The spectral start needs more rows than dimensions + 1.
            init="spectral" if rows > dims + 2 else "random",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            points = reducer.fit_transform(matrix)
        if points.shape == (rows, dims) and np.isfinite(points).all():
            return points, [f"UMAP {i + 1}" for i in range(dims)], ""
        reason = "it gave no usable positions"
    except Exception as exc:  # too few rows, or umap-learn missing
        reason = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
    from sklearn.decomposition import PCA

    # PCA gives at most as many axes as rows and features; the rest stay at 0.
    k = max(1, min(dims, rows, matrix.shape[1]))
    model = PCA(n_components=k, random_state=seed)
    points = np.zeros((rows, dims))
    points[:, :k] = model.fit_transform(matrix)
    ratios = np.nan_to_num(model.explained_variance_ratio_)
    names = [f"PC{i + 1} ({ratios[i] * 100:.1f}%)" if i < k else f"PC{i + 1} (none)"
             for i in range(dims)]
    note = f"UMAP could not run on {rows} rows ({reason}); showing PCA instead."
    return points, names, note
