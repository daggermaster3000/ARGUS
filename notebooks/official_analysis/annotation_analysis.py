"""Analysis of the blinded square counts (``annotation_squares.py``): how well two people agree, and the automated count
against their consensus.

Reads ``annotation_v2/key.csv`` and every ``annotation_v2/counts/square_counts_*.csv`` (the annotators' exports; one
file each). The automated count of a square is the number of counting-method objects (``af488_mask/<section>/
objects.parquet``, mask ``kmad8``; also ``kmad5``) whose centroid lies in the 120 µm frame.

1. **Inter-rater agreement** per square: Lin's CCC on counts and on log(1 + count), ICC(2,1), Bland–Altman on
   log(1 + count); per cell: dots of the two annotators matched within :data:`MATCH_UM` (one-to-one, nearest first),
   giving each annotator's share of the other's cells.
2. **Automated against the consensus** (the mean of the annotators): the ratio Σ automated ÷ Σ consensus overall, per
   genotype and per compartment, with 95 % CIs from :data:`BOOT` bootstrap resamples of *sections* (the squares of a
   section are not independent); the WT3 ÷ GD1 ratio of those ratios (the genotype bias) with its CI.
3. **The genotype difference on the same squares**: WT3 ÷ GD1 of the mean count per square by the consensus and by the
   automated count (crossed and junction squares), with section-bootstrap CIs.

    python notebooks/official_analysis/annotation_analysis.py      # prints the tables, writes annotation_v2/results/

No Qt.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import annotation_squares as aq  # noqa: E402

OUT = aq.OUT
MATCH_UM = 4.0                    #: two annotators' dots within this are the same cell
BOOT = 2000
SEED = 1
MASKS_AUTO = ("kmad8", "kmad5")


def load_counts() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per square and annotator the count (done squares only), and every dot (µm in the frame)."""
    files = sorted((OUT / "counts").glob("square_counts_*.csv"))
    if len(files) < 2:
        raise SystemExit(f"need two annotators' CSVs in {OUT / 'counts'}, found {len(files)}")
    t = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    squares = t[(t["kind"] == "square") & (t["done"] == 1)][["annotator", "code", "n_dots"]]
    dots = t[t["kind"] == "dot"][["annotator", "code", "x_um", "y_um"]]
    return squares, dots


def automated(key: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, g in key.groupby("name"):
        o = pd.read_parquet(aq.MASKS / name / "objects.parquet")
        for _, k in g.iterrows():
            y0, x0, s = k["frame_top_um"], k["frame_left_um"], k["frame_um"]
            inside = (o["Centroid Y (µm)"] >= y0) & (o["Centroid Y (µm)"] < y0 + s) & \
                     (o["Centroid X (µm)"] >= x0) & (o["Centroid X (µm)"] < x0 + s)
            rows.append({"code": k["code"], **{f"auto_{m}": int((inside & (o["mask"] == m)).sum()) for m in MASKS_AUTO}})
    return pd.DataFrame(rows)


def ccc(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(2 * np.cov(a, b, bias=True)[0, 1] / (a.var() + b.var() + (a.mean() - b.mean()) ** 2))


def icc21(x: np.ndarray) -> float:
    """ICC(2,1): two-way random effects, absolute agreement, single rater (x: squares × raters)."""
    n, k = x.shape
    gm = x.mean()
    msr = k * ((x.mean(1) - gm) ** 2).sum() / (n - 1)
    msc = n * ((x.mean(0) - gm) ** 2).sum() / (k - 1)
    sse = ((x - x.mean(1, keepdims=True) - x.mean(0, keepdims=True) + gm) ** 2).sum()
    mse = sse / ((n - 1) * (k - 1))
    return float((msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n))


def match(a: np.ndarray, b: np.ndarray, r: float) -> int:
    """One-to-one matches between two dot sets within *r* (greedy, nearest pairs first)."""
    if not len(a) or not len(b):
        return 0
    d = np.hypot(a[:, None, 0] - b[None, :, 0], a[:, None, 1] - b[None, :, 1])
    pairs = sorted((d[i, j], i, j) for i, j in zip(*np.nonzero(d <= r)))
    ua, ub, n = set(), set(), 0
    for _, i, j in pairs:
        if i not in ua and j not in ub:
            ua.add(i); ub.add(j); n += 1
    return n


def boot_sections(df: pd.DataFrame, stat, n=BOOT, seed=SEED):
    """Percentile 95 % CI of stat(df) over bootstrap resamples of sections."""
    rng = np.random.default_rng(seed)
    names = df["name"].unique()
    groups = {nm: g for nm, g in df.groupby("name")}
    vals = []
    for _ in range(n):
        pick = rng.choice(names, len(names), replace=True)
        v = stat(pd.concat([groups[p] for p in pick], ignore_index=True))
        if np.isfinite(v):
            vals.append(v)
    return np.percentile(vals, [2.5, 97.5]) if vals else (np.nan, np.nan)


def ratio(df, col, ref="consensus"):
    return df[col].sum() / df[ref].sum() if df[ref].sum() > 0 else np.nan


def main() -> int:
    key = pd.read_csv(OUT / "key.csv")
    squares, dots = load_counts()
    who = sorted(squares["annotator"].unique())[:2]
    wide = squares.pivot_table(index="code", columns="annotator", values="n_dots").dropna()[who]
    t = key.merge(wide.reset_index(), on="code").merge(automated(key), on="code")
    t["consensus"] = t[who].mean(axis=1)
    res = OUT / "results"
    res.mkdir(exist_ok=True)
    t.to_csv(res / "squares.csv", index=False)
    print(f"{len(t)} squares counted by both ({who[0]}, {who[1]}), from {t['name'].nunique()} sections\n")

    # 1. inter-rater
    a, b = t[who[0]].to_numpy(float), t[who[1]].to_numpy(float)
    la, lb = np.log1p(a), np.log1p(b)
    diff = la - lb
    matched = sum(match(dots[(dots.annotator == who[0]) & (dots.code == c)][["x_um", "y_um"]].to_numpy(float),
                        dots[(dots.annotator == who[1]) & (dots.code == c)][["x_um", "y_um"]].to_numpy(float), MATCH_UM)
                  for c in t["code"])
    print("1. Inter-rater agreement")
    print(f"   totals: {who[0]} {int(a.sum())}, {who[1]} {int(b.sum())} cells")
    print(f"   CCC counts {ccc(a, b):.2f}, CCC log(1+n) {ccc(la, lb):.2f}, ICC(2,1) {icc21(np.column_stack([a, b])):.2f}, "
          f"Spearman {pd.Series(a).corr(pd.Series(b), method='spearman'):.2f}")
    print(f"   Bland–Altman on log(1+n): bias {np.exp(diff.mean()):.2f}×, 95 % LoA {np.exp(diff.mean() - 1.96 * diff.std(ddof=1)):.2f}–"
          f"{np.exp(diff.mean() + 1.96 * diff.std(ddof=1)):.2f}×")
    print(f"   cells matched within {MATCH_UM:g} µm: {matched}; that is {matched / max(a.sum(), 1):.0%} of {who[0]}'s and "
          f"{matched / max(b.sum(), 1):.0%} of {who[1]}'s dots\n")

    # 2. automated against the consensus
    print("2. Automated ÷ consensus (Σ over squares; 95 % CI, section bootstrap)")
    rows = []
    for m in MASKS_AUTO:
        col = f"auto_{m}"
        for label, part in [("all", t)] + [(g, t[t.genotype == g]) for g in ("WT3", "GD1")] + \
                           [(c, t[t.compartment == c]) for c in aq.COMPARTMENTS]:
            r = ratio(part, col)
            lo, hi = boot_sections(part, lambda d: ratio(d, col))
            rows.append({"mask": m, "subset": label, "auto ÷ consensus": round(r, 2), "CI": f"{lo:.2f}–{hi:.2f}"})
        bias = lambda d: ratio(d[d.genotype == "WT3"], col) / ratio(d[d.genotype == "GD1"], col)  # noqa: E731
        lo, hi = boot_sections(t, bias)
        rows.append({"mask": m, "subset": "WT3 ÷ GD1 (genotype bias)", "auto ÷ consensus": round(bias(t), 2),
                     "CI": f"{lo:.2f}–{hi:.2f}"})
    agree = pd.DataFrame(rows)
    print(agree.to_string(index=False), "\n")
    agree.to_csv(res / "automated_vs_consensus.csv", index=False)

    # 3. the genotype difference on the same squares
    print("3. WT3 ÷ GD1, mean count per square (95 % CI, section bootstrap)")
    rows = []
    for comp in ("crossed", "junction", "home"):
        part = t[t.compartment == comp]
        for col in ("consensus", *[f"auto_{m}" for m in MASKS_AUTO]):
            f = lambda d: d[d.genotype == "WT3"][col].mean() / d[d.genotype == "GD1"][col].mean()  # noqa: E731
            lo, hi = boot_sections(part, f)
            rows.append({"compartment": comp, "count": col, "WT3 ÷ GD1": round(f(part), 2), "CI": f"{lo:.2f}–{hi:.2f}"})
    fold = pd.DataFrame(rows)
    print(fold.to_string(index=False))
    fold.to_csv(res / "genotype_fold_on_squares.csv", index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
