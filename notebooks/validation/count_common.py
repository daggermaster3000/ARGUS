"""What the count notebooks share: the cells, where they are, the clicks, the rules.

Used by ``counts_in_band``, ``counts_crossed``, ``positivity_metrics``,
``band_choice`` and ``manual_vs_auto`` (all in this folder). Nothing here
segments: the cells come from ``count_cells.py`` (``junction_eval/cells``), and
everything else — which band a cell is in, how far it is from each line and from
the tissue edge, whether it is positive under a rule — is worked out from them.

The geometry of a sample is cached beside its cells
(``junction_eval/cell_tables/<name>.parquet``) and rebuilt when the cells are
measured again or an annotation is added.

**Bands.** ``first`` is the first annotator's hand-drawn region
(``manual annotated junction region``), ``second`` Quillan's (``… quillan``),
``automatic`` the junction finder's with its defaults (from the cached overview,
with the hand-drawn band's width, gap and rim). For each band a cell has:

* ``in <band>`` — its centroid is in the band;
* ``from <band> (um)`` — signed distance to the band's line: + on the side the
  band reaches into, − on the other;
* ``beside <band>`` — false beyond either end of the line, where the side is
  not defined.

And ``edge (um)``: distance to the tissue's outline (both organoids, holes
filled), + inside; good to one overview pixel (about 5 µm).
"""

from __future__ import annotations

import json
import ssl
import sys
from pathlib import Path

# Some Python installations on Windows cannot read the Windows certificate store; nothing
# here uses the network, but reading an image imports a library that builds an SSL context.
try:
    ssl.create_default_context()
except ssl.SSLError:  # pragma: no cover - machine-specific
    ssl.SSLContext._load_windows_store_certs = lambda self, storename, purpose: None

import numpy as np
import pandas as pd
from scipy import ndimage, stats
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import count_cells as cc  # noqa: E402
import junction_tuning as jt  # noqa: E402

from microscopy_viewer import fusion as fu  # noqa: E402
from microscopy_viewer import ims_store  # noqa: E402
from microscopy_viewer import junction as jn  # noqa: E402
from microscopy_viewer import junction_analysis as ja  # noqa: E402

OUT = jt.OUT
TABLES = OUT / "cell_tables"
ANNOTATORS = {"first": fu.MANUAL_KEY, "second": jt.SECOND_KEY}
BANDS = ("first", "second", "automatic")
#: Quillan's hand count: every AF488 cell near the line or past it. All of them
#: are AF488, whatever channel the stored count names.
MANUAL_COUNTS = "Manual counts Quillan"
MATCH_UM = 7.0
GAP_UM = fu.DEFAULT_GAP_UM
#: Cells this far past the line, on the side where a channel is not at home, and
#: this far inside the tissue, are its negative control (see :func:`negatives`).
DEEP_UM = 300.0
DEEP_EDGE_UM = 30.0
K = 3.0
GEOMETRY_VERSION = 1

#: The reference measurement: nucleus grown by 3 µm, raw channel, mean.
REFERENCE = dict(expand=3.0, filtered="raw", statistic="mean")


def column(channel, expand=3.0, filtered="raw", statistic="mean"):
    return cc.column(channel, expand, filtered, statistic)


def ring_column(channel, filtered="raw", statistic="mean"):
    return cc.ring_column(channel, filtered, statistic)


# ---------------------------------------------------------------------------
# Samples and design
# ---------------------------------------------------------------------------


def design() -> pd.DataFrame:
    """Genotype, experiment and slide of every file (``Image_File_List.csv``), by sample name."""
    files = pd.read_csv(jt.PICKS.parent / "Image_File_List.csv")
    files["name"] = files["File_Name"].str.replace(".ome.tiff", "", regex=False)
    out = files.set_index("name")[["Genotype", "Experiment", "Slide"]]
    out["slide"] = out["Genotype"] + " e" + out["Experiment"].astype(str) + " s" + out["Slide"].astype(str)
    return out


def measured(name: str) -> bool:
    target = cc.OUT / f"{name}.npz"
    if not target.exists():
        return False
    with np.load(target) as data:
        return "version" in data.files and int(data["version"]) >= cc.VERSION


def samples() -> pd.DataFrame:
    """The annotated samples (excluded ones left out), with their design and whether they are measured."""
    out = jt.annotated_samples()
    out["measured"] = [measured(n) for n in out["name"]]
    return out.join(design(), on="name")


def short(name: str) -> str:
    return str(name).split("Asssembloid_")[-1]


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def _side(points_um, line_um):
    """Raw signed distance of points to a line (+ on its normals' side), and whether beside it."""
    curve = fu.fit_spline(np.asarray(line_um, dtype=float).reshape(-1, 2), 2.0)
    normal = fu.normals(curve)
    distance, nearest = cKDTree(curve).query(points_um)
    side = np.einsum("ij,ij->i", points_um - curve[nearest], normal[nearest])
    return distance * np.sign(side), (nearest > 0) & (nearest < len(curve) - 1), fu.line_length(curve)


def _automatic(name):
    """The finder's band (full resolution) and line, with its defaults; ``(None, None)`` when none found."""
    overview, _band, _line, base = jt.load_case(name)
    options = jn.JunctionOptions(save=False, **base)
    result = jn.find_junction(overview, options)
    if result.line_um is None or result.band is None:
        return None, None
    band = jn.full_resolution_band(result, overview, options) > 0
    return (band if band.any() else None), result.line_um


def _edge(name, y_px, x_px):
    overview = jt.load_case(name)[0]
    filled = jn.tissue_of(overview)
    um, step = overview.um, overview.step
    inside = ndimage.distance_transform_edt(filled) * um - um / 2
    outside = ndimage.distance_transform_edt(~filled) * um - um / 2
    signed = np.where(filled, inside, -outside)
    coords = np.vstack([(y_px - (step - 1) / 2) / step, (x_px - (step - 1) / 2) / step])
    return ndimage.map_coordinates(signed, coords, order=1, mode="nearest").astype(np.float32)


def _geometry(name, path, df, voxel, shape):
    """The band columns, the edge, and per band its line length."""
    rr = np.clip(np.round(df["y"].to_numpy()).astype(int), 0, shape[0] - 1)
    cc_ = np.clip(np.round(df["x"].to_numpy()).astype(int), 0, shape[1] - 1)
    points = np.column_stack([df["y"].to_numpy() * voxel[0], df["x"].to_numpy() * voxel[1]])
    geo, lengths, lines = {}, {}, {}
    keys = ims_store.list_labels(path)
    for band_name in BANDS:
        if band_name == "automatic":
            band, line = _automatic(name)
            if band is None:
                continue
        else:
            if ANNOTATORS[band_name] not in keys:
                continue
            labels, attrs = ims_store.load_labels(path, ANNOTATORS[band_name])
            band = np.asarray(labels).reshape(np.shape(labels)[-2:]) > 0
            line = np.asarray(attrs["line_yx_um"], dtype=float).reshape(-1, 2)
        inside = band[rr, cc_]
        raw, beside, length = _side(points, line)
        flip = np.sign(np.median(np.sign(raw[inside & beside]))) if (inside & beside).any() else 1.0
        geo[f"in {band_name}"] = inside
        geo[f"from {band_name} (um)"] = (raw * (flip or 1.0)).astype(np.float32)
        geo[f"beside {band_name}"] = beside
        lengths[band_name] = length
        lines[band_name] = (np.asarray(line, float), float(flip or 1.0))
    geo["edge (um)"] = _edge(name, df["y"].to_numpy(), df["x"].to_numpy())
    return pd.DataFrame(geo, index=df.index), lengths, lines


def load_sample(name: str, path: str):
    """One sample's cells with their geometry, and a dict of facts about it. Cached."""
    source = cc.OUT / f"{name}.npz"
    d = np.load(source)
    keep = [k for k in d.files if k not in ("voxel_yx", "shape_yx", "channels", "version")]
    df = pd.DataFrame({k: d[k] for k in keep})
    voxel = d["voxel_yx"].astype(float)
    shape = tuple(int(v) for v in d["shape_yx"])
    df["y (um)"], df["x (um)"] = df["y"] * voxel[0], df["x"] * voxel[1]
    keys = sorted(k for k in ims_store.list_labels(path) if k in ANNOTATORS.values())
    # The finder's settings too: a changed default draws another automatic band.
    finder = repr(sorted(vars(jn.JunctionOptions(save=False)).items()))
    stamp = {"version": GEOMETRY_VERSION, "cells_mtime": source.stat().st_mtime, "keys": keys, "finder": finder}
    cache, sidecar = TABLES / f"{name}.parquet", TABLES / f"{name}.json"
    info = json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.exists() else {}
    if cache.exists() and info.get("stamp") == stamp:
        geo = pd.read_parquet(cache)
        lengths, lines = info["lengths"], {k: (np.asarray(v[0]), v[1]) for k, v in info["lines"].items()}
    else:
        geo, lengths, lines = _geometry(name, path, df, voxel, shape)
        TABLES.mkdir(parents=True, exist_ok=True)
        # Written aside and moved into place: several notebooks may build the same cache at once.
        import os
        import uuid

        part = f".{uuid.uuid4().hex}.part"
        geo.to_parquet(cache.with_name(cache.name + part), index=False)
        os.replace(cache.with_name(cache.name + part), cache)
        sidecar.with_name(sidecar.name + part).write_text(
            json.dumps({"stamp": stamp, "lengths": lengths,
                        "lines": {k: [v[0].tolist(), v[1]] for k, v in lines.items()}}), encoding="utf-8")
        os.replace(sidecar.with_name(sidecar.name + part), sidecar)
    df = pd.concat([df, geo.set_index(df.index)], axis=1)
    meta = {"name": name, "path": path, "cells": len(df), "voxel": voxel, "shape": shape,
            "channels": [str(c) for c in d["channels"]], "lengths (um)": lengths, "lines": lines,
            "bands": [b for b in BANDS if f"in {b}" in df]}
    return df, meta


def load_all(progress: bool = True):
    """Every measured sample: ``(cells, meta)`` — a dict of DataFrames and one row per sample."""
    table = samples()
    cells, metas = {}, []
    for row in table[table["measured"]].itertuples():
        cells[row.name], m = load_sample(row.name, row.path)
        metas.append({**m, "Genotype": row.Genotype, "Experiment": row.Experiment, "slide": row.slide})
    meta = pd.DataFrame(metas).set_index("name")
    if progress:
        print(f"{len(cells)} of {len(table)} annotated samples measured, {int(meta['cells'].sum()):,} cells; "
              f"bands: " + ", ".join(f"{b} {int(meta['bands'].apply(lambda x: b in x).sum())}" for b in BANDS))
    return cells, meta


def label_channel(meta) -> str:
    channels = meta["channels"].iloc[0]
    return channels[jn.pick_channel(channels, "", jn.LABEL_HINTS)]


# ---------------------------------------------------------------------------
# Where a channel is at home, its negatives, and the regions counted in
# ---------------------------------------------------------------------------


#: How far from the line the two sides are compared to find a channel's home.
HOME_UM = 300.0
#: The part of each side's distribution that decides the home side: the top,
#: where the labelled cells are — the median is the background.
HOME_QUANTILE = 95.0


def home_score(df, channel, band):
    """``(band side, other side)``: the 95th percentile of *channel*'s ring contrast within
    :data:`HOME_UM` of the line on each side. NaN where a side has fewer than 20 cells."""
    values = contrast(df, channel)
    signed, beside = df[f"from {band} (um)"].to_numpy(), df[f"beside {band}"].to_numpy()
    near = beside & np.isfinite(values) & (np.abs(signed) <= HOME_UM)
    out = []
    for side in (near & (signed > GAP_UM), near & (signed < -GAP_UM)):
        out.append(float(np.percentile(values[side], HOME_QUANTILE)) if side.sum() >= 20 else np.nan)
    return tuple(out)


def home_is_band_side(df, channel, band) -> bool:
    """See :func:`_home_is_band_side`; worked out once per sample, channel and band."""
    memo = df.attrs.setdefault("home", {})       # string keys: attrs go into parquet metadata as JSON
    key = f"{channel}|{band}"
    if key not in memo:
        memo[key] = _home_is_band_side(df, channel, band)
    return memo[key]


def _home_is_band_side(df, channel, band) -> bool:
    """Whether *channel* is at home on the side its band reaches into.

    Decided by the labelled cells, not the background: the side whose brightest
    cells (95th percentile of cell-over-ring contrast, within 300 µm of the line)
    stand out more. A median over all cells mostly measures the background, which
    differs between organoids for reasons of its own. Where a side has too few
    cells, the band's side is taken as *not* home — the junction analysis's own
    assumption — and a warning is logged.
    """
    band_side, other = home_score(df, channel, band)
    if not (np.isfinite(band_side) and np.isfinite(other)):
        import logging

        logging.getLogger("count_common").warning(
            "%s: too few cells on one side of the %s line to tell where it is at home", channel, band)
        return _assumed_home_is_band_side(channel)
    if abs(band_side - other) < HOME_MARGIN * max(band_side, other):
        # Too close to call from the data: the design decides.
        return _assumed_home_is_band_side(channel)
    return bool(band_side > other)


#: Two sides whose scores are closer than this share are too close to call.
HOME_MARGIN = 0.03


def _assumed_home_is_band_side(channel) -> bool:
    """The design: the band reaches into the organoid *without* the label (GFP / 488),
    which is the one the other stain marks."""
    is_label = any(hint in str(channel).lower() for hint in jn.LABEL_HINTS)
    return not is_label


def away_distance(df, channel, band):
    """Signed distance from the line, + into the organoid where *channel* is not at home."""
    signed = df[f"from {band} (um)"].to_numpy()
    return -signed if home_is_band_side(df, channel, band) else signed


def negatives(df, channel, band, deep_um=None):
    """The negative control: cells deep in the organoid where *channel* is not at home.

    More than :data:`DEEP_UM` past the line on that side, beside it, and more
    than :data:`DEEP_EDGE_UM` inside the tissue. Rare migrants that far out are
    tolerated by taking a high percentile, not the maximum — but this is also
    where "crossed, any distance" counts, so a sample with many far migrants
    raises its own bar there; counts within 300 µm share no cells with it.
    """
    away = away_distance(df, channel, band)
    deep = DEEP_UM if deep_um is None else deep_um
    return (away > deep) & df[f"beside {band}"].to_numpy() & (df["edge (um)"].to_numpy() > DEEP_EDGE_UM)


def region(df, which, band, channel, max_um=None, near_um=50.0):
    """Cells counted in: ``"band"``, ``"crossed"`` (past the line into the other organoid,
    up to *max_um*), or ``"near + crossed"`` (within *near_um* of the line either side, or crossed)."""
    if which == "band":
        return df[f"in {band}"].to_numpy()
    away = away_distance(df, channel, band)
    beside = df[f"beside {band}"].to_numpy()
    crossed = beside & (away > GAP_UM) & (away <= (max_um if max_um is not None else np.inf))
    if which == "crossed":
        return crossed
    if which == "near + crossed":
        return crossed | (beside & (np.abs(away) <= near_um))
    raise ValueError(which)


# ---------------------------------------------------------------------------
# Positivity rules: each returns a boolean per cell of the sample
# ---------------------------------------------------------------------------


def _median_ksd(values, population, k):
    centre, spread = ja.robust_centre_and_spread(population)
    return values > centre + k * spread


def contrast(df, channel, filtered="raw"):
    """Cell over its local background: the cell's mean (3 µm growth) over its ring's.

    A cell packed in among its neighbours has no ring — every pixel 3–6 µm out
    belongs to another cell — and there the background is the channel's local
    median over 45 µm instead (the raw mean less the median-filtered one), so a
    packed cell can still be called positive.
    """
    cell = df[column(channel, 3.0, "raw", "mean")].to_numpy(dtype=float)
    ring = df[ring_column(channel, "raw", "mean")].to_numpy(dtype=float)
    local = cell - df[column(channel, 3.0, "median", "mean")].to_numpy(dtype=float)
    background = np.where(np.isfinite(ring) & (ring > 0), ring, local)
    return cell / np.where(background > 0, background, np.nan)


def _gmm_positive(values, p=0.95, seed=0):
    from sklearn.mixture import GaussianMixture

    logs = np.log1p(np.clip(values, 0, None))
    ok = np.isfinite(logs)
    sample = logs[ok]
    if sample.size > 50_000:
        sample = np.random.default_rng(seed).choice(sample, 50_000, replace=False)
    gm = GaussianMixture(2, random_state=seed).fit(sample.reshape(-1, 1))
    means = gm.means_.ravel()
    bright, dim = int(np.argmax(means)), int(np.argmin(means))
    out = np.zeros(len(values), bool)
    # Above the dim population's centre too: a wide bright component also claims the dim tail.
    out[ok] = (gm.predict_proba(logs[ok].reshape(-1, 1))[:, bright] > p) & (logs[ok] > means[dim])
    return out


def _otsu_positive(values):
    from skimage.filters import threshold_otsu

    logs = np.log1p(np.clip(values, 0, None))
    ok = np.isfinite(logs)
    return np.where(ok, logs > threshold_otsu(logs[ok]), False)


def rule_band_ksd(df, channel, band, k=3.0, filtered="raw", statistic="mean"):
    """median + k SD of the band's cells (the junction analysis's rule)."""
    values = df[column(channel, 3.0, filtered, statistic)].to_numpy()
    return _median_ksd(values, values[df[f"in {band}"].to_numpy()], k)


def rule_negative_percentile(df, channel, band, q=99.0, value="intensity"):
    """Above the q-th percentile of the negative control (deep in the other organoid).

    Where fewer than 50 cells lie :data:`DEEP_UM` past the line (a small
    organoid), the control moves nearer — 200, then 100 µm — rather than calling
    nothing positive; failing that, nothing is called and a warning is logged.
    """
    values = contrast(df, channel) if value == "contrast" else df[column(channel)].to_numpy()
    for deep in (DEEP_UM, 200.0, 100.0):
        neg = negatives(df, channel, band, deep) & np.isfinite(values)
        if neg.sum() >= 50:
            return values > np.percentile(values[neg], q)
    import logging

    logging.getLogger("count_common").warning("%s: no negative control on the %s line; nothing called", channel, band)
    return np.zeros(len(df), bool)


def rule_contrast(df, channel, band, ratio=1.5):
    """Cell at least *ratio* × its own ring."""
    return np.nan_to_num(contrast(df, channel), nan=0.0) >= ratio


def rule_not_autofluorescent(df, channel, band, others, base):
    """*base*, without cells that are as bright, for their ring, in another channel (autofluorescence)."""
    own = np.nan_to_num(contrast(df, channel), nan=0.0)
    keep = base.copy()
    for other in others:
        keep &= ~(np.nan_to_num(contrast(df, other), nan=0.0) >= own)
    return keep


def rules(channels, label):
    """The rules compared, by name: a function of ``(df, band)`` giving positive per cell for *label*."""
    others = [c for c in channels if c != label]
    out = {
        "band median + 3 SD (current)": lambda df, b: rule_band_ksd(df, label, b, 3.0),
        "band median + 7 SD, median filter": lambda df, b: rule_band_ksd(df, label, b, 7.0, "median"),
        "negative control 99th pct": lambda df, b: rule_negative_percentile(df, label, b, 99.0),
        "negative control 99.9th pct": lambda df, b: rule_negative_percentile(df, label, b, 99.9),
        "ring contrast ≥ 1.5": lambda df, b: rule_contrast(df, label, b, 1.5),
        "ring contrast ≥ 2": lambda df, b: rule_contrast(df, label, b, 2.0),
        "ring contrast > negative 99th pct": lambda df, b: rule_negative_percentile(df, label, b, 99.0, "contrast"),
        "GMM on log intensity (p > 0.95)": lambda df, b: _gmm_positive(df[column(label)].to_numpy()),
        "Otsu on log intensity": lambda df, b: _otsu_positive(df[column(label)].to_numpy()),
    }
    out["ring contrast > negative 99th pct, no autofluorescence"] = lambda df, b: rule_not_autofluorescent(
        df, label, b, others, rule_negative_percentile(df, label, b, 99.0, "contrast"))
    return out


# ---------------------------------------------------------------------------
# Quillan's clicks
# ---------------------------------------------------------------------------


def load_dots(cells, meta):
    """Every click, matched one to one to the nearest cell within :data:`MATCH_UM` (nearest first).

    Adds ``clicked`` to each sample's cells. Returns one row per dot with the
    matched cell (−1 for none) and the dot's own band membership and distances.
    """
    rows = []
    for name, df in cells.items():
        df["clicked"] = False
        path = meta.loc[name, "path"]
        if MANUAL_COUNTS not in ims_store.list_labels(path):
            continue
        voxel = np.asarray(meta.loc[name, "voxel"], float)
        lab, attrs = ims_store.load_labels(path, MANUAL_COUNTS)
        lab = np.asarray(lab).reshape(np.shape(lab)[-2:])
        ids = np.unique(lab[lab > 0])
        centres = np.asarray(ndimage.center_of_mass(lab > 0, lab, ids)).reshape(-1, 2)
        dots_um = centres * voxel
        cell_um = df[["y (um)", "x (um)"]].to_numpy()
        pairs = []
        for k, near in enumerate(cKDTree(cell_um).query_ball_point(dots_um, MATCH_UM)):
            pairs += [(float(np.hypot(*(cell_um[j] - dots_um[k]))), k, j) for j in near]
        used_dot, used_cell, matched = set(), set(), np.full(len(ids), -1)
        for _dist, k, j in sorted(pairs):
            if k not in used_dot and j not in used_cell:
                used_dot.add(k); used_cell.add(j); matched[k] = j
        clicked = np.zeros(len(df), bool)
        clicked[matched[matched >= 0]] = True
        df["clicked"] = clicked
        d = pd.DataFrame({"name": name, "dot": ids, "y (um)": dots_um[:, 0], "x (um)": dots_um[:, 1],
                          "cell": matched, "matched": matched >= 0,
                          "stored as": ims_store._text(attrs.get("channel", ""))})
        shape = meta.loc[name, "shape"]
        rr = np.clip(np.round(centres[:, 0]).astype(int), 0, shape[0] - 1)
        cc_ = np.clip(np.round(centres[:, 1]).astype(int), 0, shape[1] - 1)
        keys = ims_store.list_labels(path)
        for band_name, (line, flip) in meta.loc[name, "lines"].items():
            if band_name in ANNOTATORS and ANNOTATORS[band_name] in keys:
                labels, _ = ims_store.load_labels(path, ANNOTATORS[band_name])
                d[f"in {band_name}"] = (np.asarray(labels).reshape(np.shape(labels)[-2:]) > 0)[rr, cc_]
            raw, beside, _ = _side(dots_um, line)
            d[f"from {band_name} (um)"], d[f"beside {band_name}"] = raw * flip, beside
        d["edge (um)"] = _edge(name, centres[:, 0], centres[:, 1])
        rows.append(d)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def ccc(a, b):
    """Lin's concordance correlation coefficient."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 3:
        return np.nan
    return 2 * np.cov(a, b, bias=True)[0, 1] / (a.var() + b.var() + (a.mean() - b.mean()) ** 2)


def compare_groups(frame, columns, by="Genotype", unit="sample", groups=None):
    """Groups compared per column: medians, n, and a test. *unit* ``"slide"`` averages per slide first.

    Two groups (*groups*, or the two present): their ratio and Mann-Whitney p.
    More than two: Kruskal-Wallis p over all of them — never two picked by
    alphabet with the rest dropped.
    """
    if unit == "slide":
        frame = frame.groupby("slide").agg({**{c: "mean" for c in columns}, by: "first"})
    groups = list(groups) if groups is not None else sorted(frame[by].dropna().unique())
    out = {}
    for c in columns:
        parts = [frame.loc[frame[by] == g, c].dropna() for g in groups]
        row = {f"{g} median": x.median() for g, x in zip(groups, parts)}
        row["n"] = " vs ".join(str(len(x)) for x in parts)
        filled = [x for x in parts if len(x)]
        if len(groups) == 2 and len(filled) == 2:
            a, b = parts[0].median(), parts[1].median()
            row[f"{groups[1]} / {groups[0]}"] = b / a if a != 0 else (np.inf if b > 0 else np.nan)
            row["test"], row["p"] = "Mann-Whitney", stats.mannwhitneyu(*parts, alternative="two-sided").pvalue
        elif len(filled) > 2:
            row["test"], row["p"] = "Kruskal-Wallis", stats.kruskal(*filled).pvalue
        else:
            row["test"], row["p"] = "", np.nan
        out[c] = row
    return pd.DataFrame(out).T
