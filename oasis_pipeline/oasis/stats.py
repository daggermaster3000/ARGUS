"""The primary outcome: WT3 against GD1 in AF488⁺ cells crossed per mm of junction interface.

Per section: y = crossed AF488⁺ cells (``cpself.DEFAULT``, the self-trained Cellpose method at 5 noise SD) ÷ the
junction interface length (mm). Linear mixed model of log(1 + y) (statsmodels ``mixedlm``, REML): genotype (reference
GD1) and experiment as fixed effects, slide as a random intercept. The genotype coefficient, exponentiated, is the
fold (WT3 ÷ GD1) on the log(1 + y) scale, with its 95 % confidence interval and P value.

The same model for every method-threshold of ``cpself`` (signal 3, 5, 8 noise SD; size filters 20, 10, 5, 3 µm²) as
a sensitivity analysis, and for the cells in the junction region per mm.

Reads ``rerun/counting/cellpose_counts.csv`` (``cpself.count``) and ``data/design/analysed_sections.csv``; writes
``rerun/statistics/primary_outcome.csv`` (and prints the primary result). No Qt.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from . import cpself, paths

OUTCOMES = {"crossed": "crossed AF488⁺ cells per mm of junction interface",
            "region": "AF488⁺ cells in the junction region per mm of junction interface"}


def sections() -> pd.DataFrame:
    s = pd.read_csv(paths.SECTIONS_CSV)
    s["section"] = s["id"]
    s["experiment"] = s["experiment"].astype(str)
    return s[["section", "genotype", "experiment", "slide", "junction length (µm)"]]


def fold(table: pd.DataFrame, column: str) -> dict:
    """The genotype fold (WT3 ÷ GD1) of log(1 + column per mm), mixed model with slide as a random intercept."""
    import statsmodels.formula.api as smf

    t = table.dropna(subset=[column, "junction length (µm)"]).copy()
    t = t[t["junction length (µm)"] > 0]
    t["ly"] = np.log1p(t[column] / (t["junction length (µm)"] / 1000))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = smf.mixedlm("ly ~ C(genotype, Treatment('GD1')) + experiment", t, groups=t["slide"]).fit(reml=True)
    term = next(p for p in fit.params.index if p.startswith("C(genotype"))
    lo, hi = fit.conf_int().loc[term]
    return {"n sections": len(t), "n slides": t["slide"].nunique(), "fold WT3/GD1": float(np.exp(fit.params[term])),
            "CI low": float(np.exp(lo)), "CI high": float(np.exp(hi)), "P": float(fit.pvalues[term]),
            "GD1 mean per mm": float(np.expm1(t.loc[t["genotype"] == "GD1", "ly"].mean())),
            "WT3 mean per mm": float(np.expm1(t.loc[t["genotype"] == "WT3", "ly"].mean()))}


def main(names: list[str] | None = None, workers: int = 1) -> int:
    counts = pd.read_csv(paths.CELLPOSE_COUNTS)
    table = counts.merge(sections(), on="section", how="inner")
    if table["genotype"].nunique() < 2 or table["slide"].nunique() < 3:
        print(f"stats: {table['section'].nunique()} sections, genotypes {sorted(table['genotype'].unique())}: "
              "too few for the model (run the cellpose step on every section)", flush=True)
        return 0
    rows = []
    for method, g in table.groupby("method", sort=False):
        for column, label in OUTCOMES.items():
            rows.append({"method": method, "outcome": label, "primary": method == cpself.DEFAULT and column == "crossed",
                         **fold(g, column)})
    out = pd.DataFrame(rows).sort_values(["primary", "outcome", "method"], ascending=[False, True, True])
    paths.STATISTICS.mkdir(parents=True, exist_ok=True)
    target = paths.STATISTICS / "primary_outcome.csv"
    out.round(6).to_csv(target, index=False)
    p = out.iloc[0]
    print(f"primary outcome ({p['method']}, {p['outcome']}): WT3/GD1 = {p['fold WT3/GD1']:.2f} "
          f"(95 % CI {p['CI low']:.2f}–{p['CI high']:.2f}), P = {p['P']:.2g}, {p['n sections']} sections on "
          f"{p['n slides']} slides -> {target}", flush=True)
    return 0
