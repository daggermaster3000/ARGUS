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


IDENTITY = ("Sample", "Genotype", "Region", "Label map")


#: Ink for annotations over the plot.
INK = ap.INK


#: Region colours when colouring by region: the reference categorical order.
REGION_COLORS = ap.GENOTYPE_COLORS + ap.EXTRA_COLORS


SYMBOLS = ("circle", "square", "diamond", "cross", "x", "triangle-up", "triangle-down", "star")


@st.cache_data(show_spinner=False)
def read_features(source: bytes | str) -> pd.DataFrame:
    handle = io.BytesIO(source) if isinstance(source, bytes) else source
    frame = pd.read_excel(handle, sheet_name=SHEET)
    for column in ("Sample", "Region", "Label map"):
        if column in frame:
            frame[column] = frame[column].astype(str)
    frame["Genotype"] = [ap.genotype_label(value) for value in frame.get("Genotype", "")]
    return frame


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
    try:
        channels = pd.read_excel(io.BytesIO(source) if isinstance(source, bytes) else source,
                                 sheet_name=CELL_INTENSITIES)
    except ValueError:  # workbooks from before every channel was measured per cell
        channels = pd.DataFrame()
    if not channels.empty:
        channels["Sample"] = channels["Sample"].astype(str)
        channels = channels.drop(columns=["Genotype"], errors="ignore")
        cells = cells.merge(channels, on=["Sample", "Label"], how="left")
    cells["Genotype"] = [ap.genotype_label(value) for value in cells.get("Genotype", "")]
    cells["Region"] = cells.get("Region", "").astype(str)
    return cells


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
        out["notes"].append("Needs two genotypes and at least three samples.")
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
        f"Mixed model on {len(data):,} cells from {units} samples: genotype as a fixed "
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
    order = ap.genotype_order(frame[group]) if group == "Genotype" else list(dict.fromkeys(frame[group]))
    styles = ap.genotype_styles(order) if group == "Genotype" else {}
    colors = {g: (styles[g][0] if styles else REGION_COLORS[i % len(REGION_COLORS)])
              for i, g in enumerate(order)}
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
            figure.add_trace(go.Violin(points=False, box_visible=True, meanline_visible=True,
                                       fillcolor=_alpha(color, 0.35), opacity=0.9,
                                       hoverinfo="skip", **shared))
        else:
            figure.add_trace(go.Box(boxpoints=False, fillcolor=_alpha(color, 0.35),
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
