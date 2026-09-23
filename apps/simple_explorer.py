"""The plain explorer: measured numbers, plotted and tested. No embeddings.

Everything here is a quantity somebody can name — region area, cells per mm²,
cell diameter, a channel's mean intensity — compared between genotypes. The
bigger app, :mod:`region_explorer`, adds UMAP, clustering, SHAP and the region
atlas; this one is for the plot that goes in the figure and the number that
goes in the caption.

Three tabs:

* **Compare** — one variable as a box or violin plot per genotype, every
  sample a dot, with an ANOVA (or Welch / Kruskal-Wallis) and, for cells, a
  mixed model that does not treat the cells of one fish as independent.
* **Relate** — any two variables against each other, coloured by genotype.
* **Table** — the rows behind the plots, and a CSV of them.

The statistics are the same code the big app runs (``explorer_common``), so
the two never disagree about a p value.

Run with::

    streamlit run apps/simple_explorer.py
    streamlit run apps/simple_explorer.py -- path/to/report.xlsx

Needs ``streamlit``, ``plotly``, ``scipy`` and ``statsmodels``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from microscopy_viewer import analysis_plots as ap  # noqa: E402

from explorer_common import (  # noqa: E402
    CELL_IDENTITY,
    CELL_TESTS,
    IDENTITY,
    MIXED,
    SHEET,
    TESTS,
    anova_table,
    compare_groups,
    comparison_figure,
    mixed_model_test,
    mixed_pairs,
    default_variable,
    numeric_columns,
    plottable,
    posthoc_pairs,
    read_cells,
    read_features,
    stars,
)

#: Sizes a cell can be filtered on, first one found wins.
SIZE_COLUMNS = ("Equivalent diameter (µm)", "Area (µm²)", "Footprint area (µm²)", "Volume (µm³)")


def genotype_colors(values) -> tuple[dict[str, str], list[str]]:
    order = ap.genotype_order(values)
    return {g: c for g, (c, _m) in ap.genotype_styles(order).items()}, order


# ---------------------------------------------------------------------------
# The workbook
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Region report", layout="wide")
st.title("Region report")

with st.sidebar:
    st.header("Workbook")
    uploaded = st.file_uploader("Analysis workbook (.xlsx)", type=["xlsx"])
    typed = st.text_input("…or a path on this machine",
                          value=sys.argv[1] if len(sys.argv) > 1 else "")

source: bytes | str | None = None
if uploaded is not None:
    source = uploaded.getvalue()
elif typed.strip():
    candidate = Path(typed.strip()).expanduser()
    if candidate.is_dir():
        books = sorted(candidate.glob("*.xlsx"))
        candidate = books[-1] if books else candidate
    if candidate.is_file():
        source = str(candidate)
    else:
        st.sidebar.error(f"No workbook at {candidate}")

if source is None:
    st.info(
        "Load the workbook an Analysis run wrote — the .xlsx in its "
        "`<experiment>_analysis_<date>` folder."
    )
    st.stop()

try:
    features = read_features(source)
except ValueError as exc:
    st.error(f"Could not read the “{SHEET}” sheet: {exc}")
    st.stop()
cells_all = read_cells(source)

with st.sidebar:
    st.header("Rows")
    genotypes = st.multiselect("Genotypes", ap.genotype_order(features["Genotype"]),
                               default=ap.genotype_order(features["Genotype"]))
    samples = st.multiselect("Samples", list(dict.fromkeys(features["Sample"])),
                             default=list(dict.fromkeys(features["Sample"])))
    size_column, size_range = None, None
    if not cells_all.empty:
        st.header("Cells")
        found = [c for c in SIZE_COLUMNS if c in cells_all and cells_all[c].notna().any()]
        if found:
            size_column = st.selectbox("Size measure", found)
            values = cells_all[size_column].astype(float)
            low, high = float(values.min()), float(values.max())
            if high > low:
                size_range = st.slider("Keep cells from … to …", low, high, (low, high),
                                       step=float((high - low) / 500), format="%.1f",
                                       help="Drops debris below and merged clumps above.")
                keep = values.between(*size_range)
                st.caption(f"{int(keep.sum()):,} of {len(cells_all):,} cells kept.")
                cells_all = cells_all[keep].reset_index(drop=True)

rows = features[
    features["Genotype"].isin(genotypes) & features["Sample"].isin(samples)
].reset_index(drop=True)
if not cells_all.empty:
    cells_all = cells_all[
        cells_all["Genotype"].isin(genotypes) & cells_all["Sample"].isin(samples)
    ].reset_index(drop=True)
if rows.empty:
    st.warning("No rows left after filtering.")
    st.stop()
st.caption(
    f"{rows['Sample'].nunique()} sample(s), {rows['Genotype'].nunique()} genotype(s), "
    f"{rows['Region'].nunique()} region(s)"
    + (f", {len(cells_all):,} cells." if not cells_all.empty else ".")
)

compare_tab, relate_tab, table_tab = st.tabs(["Compare", "Relate", "Table"])


# ---------------------------------------------------------------------------
# Compare: one variable between genotypes
# ---------------------------------------------------------------------------

with compare_tab:
    left, right = st.columns([1, 3])
    with left:
        st.subheader("Compare")
        measures = ["Regions"] + ([] if cells_all.empty else ["Cells"])
        measure = st.radio("Measure", measures, key="measure",
                           help="Regions: one row per sample and region. Cells: the "
                                "segmented cells of one region.")
        per_cell = False
        if measure == "Regions":
            region = st.selectbox("Region", list(dict.fromkeys(rows["Region"])), key="region")
            frame = rows[rows["Region"] == region]
            numbers = plottable(frame, IDENTITY)
            note = f"one dot per sample, {region}"
        else:
            choices = ["(every region)"] + list(dict.fromkeys(cells_all["Region"]))
            region = st.selectbox("Region", choices, key="cell-region")
            frame = cells_all if region == "(every region)" else cells_all[cells_all["Region"] == region]
            numbers = plottable(frame, CELL_IDENTITY)
            per_cell = st.radio(
                "One dot per", ("Sample (mean of its cells)", "Cell"), key="level",
                help="Either way the cells of one fish are never treated as independent "
                     "measurements: showing every cell switches the test to a mixed model "
                     "with the sample as a random effect.",
            ) == "Cell"
            note = f"{len(frame):,} cells" + ("" if region == "(every region)" else f" in {region}")
        variable = (st.selectbox("Variable", numbers, index=default_variable(numbers),
                                 key="variable") if numbers else None)
        kind = st.radio("Shape", ("Box", "Violin"), horizontal=True, key="kind")
        test = st.selectbox("Test", CELL_TESTS if per_cell else TESTS, key="test",
                            help="ANOVA compares means and assumes similar spreads; Welch's "
                                 "drops that assumption; Kruskal-Wallis compares ranks and "
                                 "assumes least. With two groups an ANOVA is a t-test.")
        bars = st.radio("Significance bars", ("Significant only", "All pairs", "Off"), key="bars")
        bar_label = st.radio("Bars say", ("Stars", "p value"), horizontal=True, key="bar-label")

    with right:
        if variable is None:
            st.warning("Nothing numeric to plot here.")
        else:
            data = frame.dropna(subset=[variable])
            if measure == "Cells" and not per_cell:
                data = (data.groupby(["Sample", "Genotype"], as_index=False)[variable]
                        .mean(numeric_only=True))
            tested = data
            if per_cell and test != MIXED:
                tested = (data.groupby(["Sample", "Genotype"], as_index=False)[variable]
                          .mean(numeric_only=True))
            order = ap.genotype_order(data["Genotype"])
            values = {g: tested.loc[tested["Genotype"] == g, variable].to_numpy(dtype=float)
                      for g in order}
            if len(order) < 2 or data.empty:
                st.warning("Needs at least two genotypes with values.")
            else:
                if per_cell and test == MIXED:
                    result = mixed_model_test(tested, variable, "Genotype")
                    pairs = mixed_pairs(tested, variable, "Genotype") if len(order) > 2 else pd.DataFrame()
                else:
                    plain = test.split(" · ")[-1]
                    result = compare_groups(values, plain)
                    pairs = posthoc_pairs(values, plain) if len(order) > 2 else pd.DataFrame()
                if len(order) == 2 and np.isfinite(result["p"]):
                    pairs = pd.DataFrame([{"Group 1": order[0], "Group 2": order[1],
                                           "p": result["p"]}])
                figure = comparison_figure(data, variable, "Genotype", kind,
                                           f"{variable} — {note}", pairs=pairs,
                                           bars=bars, bar_label=bar_label)
                st.plotly_chart(figure, width="stretch", theme="streamlit")

                short = "Mixed model" if test == MIXED else test.split(" · ")[-1]
                statistic = {"Kruskal-Wallis": "H", "Mixed model": "χ²"}.get(short, "F")
                columns = st.columns(3)
                columns[0].metric(f"{statistic} · {short}",
                                  "—" if not np.isfinite(result["statistic"]) else f"{result['statistic']:.3g}")
                columns[1].metric("p", "—" if not np.isfinite(result["p"]) else f"{result['p']:.3g}",
                                  stars(result["p"]) or None)
                columns[2].metric(result["effect_name"],
                                  "—" if not np.isfinite(result["effect"]) else f"{result['effect']:.3f}")
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
                    st.dataframe(table.style.format({c: "{:.4g}" for c in numeric}, na_rep=""),
                                 hide_index=True, width="stretch")
                summary = pd.DataFrame([{
                    "Genotype": g,
                    "Samples": int(tested.loc[tested["Genotype"] == g, "Sample"].nunique()),
                    "n": len(values[g]),
                    "Mean": float(np.mean(values[g])) if len(values[g]) else np.nan,
                    "SD": float(np.std(values[g], ddof=1)) if len(values[g]) > 1 else np.nan,
                    "Median": float(np.median(values[g])) if len(values[g]) else np.nan,
                } for g in order])
                st.dataframe(summary.style.format({"Mean": "{:.4g}", "SD": "{:.4g}",
                                                   "Median": "{:.4g}"}),
                             hide_index=True, width="stretch")
                if not pairs.empty and len(order) > 2:
                    st.caption("Every pair, corrected for the number of comparisons.")
                    st.dataframe(
                        pairs.assign(**{"": [stars(float(p)) for p in pairs["p"]]})
                        .style.format({"p": "{:.3g}", "Difference": "{:.4g}"}),
                        hide_index=True, width="stretch",
                    )
                st.download_button("These values (CSV)", data.to_csv(index=False),
                                   file_name=f"{variable}_by_genotype.csv".replace("/", "-"))


# ---------------------------------------------------------------------------
# Relate: two variables against each other
# ---------------------------------------------------------------------------

with relate_tab:
    left, right = st.columns([1, 3])
    with left:
        st.subheader("Relate")
        measures = ["Regions"] + ([] if cells_all.empty else ["Cells"])
        what = st.radio("Measure", measures, key="rel-measure")
        if what == "Regions":
            region = st.selectbox("Region", ["(every region)"] + list(dict.fromkeys(rows["Region"])),
                                  key="rel-region")
            frame = rows if region == "(every region)" else rows[rows["Region"] == region]
            numbers = plottable(frame, IDENTITY)
            colour_choices = ["Genotype", "Region", "Sample"]
        else:
            region = st.selectbox("Region",
                                  ["(every region)"] + list(dict.fromkeys(cells_all["Region"])),
                                  key="rel-cell-region")
            frame = cells_all if region == "(every region)" else cells_all[cells_all["Region"] == region]
            numbers = plottable(frame, CELL_IDENTITY)
            colour_choices = ["Genotype", "Region", "Sample"]
            if len(frame) > 5000:
                frame = frame.sample(n=5000, random_state=0)
                st.caption("5,000 cells drawn at random.")
        x = (st.selectbox("X", numbers, index=default_variable(numbers), key="rel-x")
             if numbers else None)
        y = (st.selectbox("Y", numbers, index=default_variable(numbers, skip=[x]), key="rel-y")
             if numbers else None)
        colour = st.selectbox("Colour by", colour_choices, key="rel-colour")
        trend = st.checkbox("Straight-line fit per colour", value=False, key="rel-trend")

    with right:
        if not numbers or x is None or y is None:
            st.warning("Nothing numeric to plot here.")
        else:
            plot = frame.dropna(subset=[x, y])
            colors, order = genotype_colors(plot["Genotype"])
            common = dict(color=colour, hover_name="Sample",
                          hover_data={c: True for c in ("Genotype", "Region") if c in plot})
            if colour == "Genotype":
                common.update(color_discrete_map=colors, category_orders={"Genotype": order})
            figure = px.scatter(plot, x=x, y=y, trendline="ols" if trend else None, **common)
            figure.update_traces(marker={"size": 9, "line": {"width": 1, "color": "white"}},
                                 selector={"mode": "markers"})
            figure.update_layout(height=620, margin={"l": 10, "r": 10, "t": 30, "b": 10})
            st.plotly_chart(figure, width="stretch", theme="streamlit")
            if len(plot) > 2:
                from scipy import stats

                fit = stats.pearsonr(plot[x].astype(float), plot[y].astype(float))
                spearman = stats.spearmanr(plot[x].astype(float), plot[y].astype(float))
                columns = st.columns(2)
                columns[0].metric("Pearson r", f"{fit.statistic:.3f}", f"p = {fit.pvalue:.3g}")
                columns[1].metric("Spearman ρ", f"{spearman.statistic:.3f}",
                                  f"p = {spearman.pvalue:.3g}")
                st.caption("Over every point drawn. Cells of one sample are not independent, "
                           "so a correlation over cells describes these cells, not the fish.")


# ---------------------------------------------------------------------------
# Table
# ---------------------------------------------------------------------------

with table_tab:
    which = st.radio("Show", ["Regions"] + ([] if cells_all.empty else ["Cells"]),
                     horizontal=True, key="table-which")
    shown = rows if which == "Regions" else cells_all
    st.dataframe(shown, width="stretch", hide_index=True)
    st.download_button(f"Download these {which.lower()} (CSV)", shown.to_csv(index=False),
                       file_name=f"{which.lower()}_filtered.csv")
