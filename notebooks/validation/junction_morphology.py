"""Does the shape of the section, or where the seeds sit in it, say where the junction is?

For every annotated sample (cached overviews, see ``junction_tuning``): the
tissue outline and its concavities (how far each outline point lies inside the
convex hull), the tissue's width along the axis from one seed to the other,
where the hand-drawn junction sits on that axis, and three junctions made from
shape alone, scored against the hand-drawn line as the finder is.

Results to ``junction_eval/morphology.csv`` (one row per sample) and
``morphology.pkl`` (the geometry, for pictures), read by
``01_junction_evaluation.ipynb``.

    python notebooks/validation/junction_morphology.py
"""
from __future__ import annotations

import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

import junction_tuning as jt  # noqa: E402
from microscopy_viewer import junction as jn  # noqa: E402

TABLE = jt.OUT / "morphology.csv"
GEOMETRY = jt.OUT / "morphology.pkl"
#: An outline point's concavity is the deepest within this arc either side of it (µm).
ARC_UM = 50.0
#: A concavity counts as a notch if it is at least this deep and stands out this much (µm).
NOTCH_UM = 20.0
#: Width profile: bins along the seed axis, as a share of the distance between the seeds.
BIN = 0.02


def outline(filled):
    """The largest outline of *filled*, as ordered (row, col) points about 1 px apart."""
    from skimage.measure import find_contours

    contours = find_contours(np.pad(filled, 1).astype(float), 0.5)
    return max(contours, key=len) - 1.0


def windowed_max(values, half):
    """Each value replaced by the largest within *half* points either side (the outline is closed)."""
    n = len(values)
    if half <= 0 or n == 0:
        return values.copy()
    ring = np.concatenate([values[-half:], values, values[:half]])
    from scipy.ndimage import maximum_filter1d

    return maximum_filter1d(ring, 2 * half + 1)[half: half + n]


def notches(depth, um):
    """Indices of the outline's notches: local maxima of concavity depth (closed outline)."""
    from scipy.signal import find_peaks

    n = len(depth)
    doubled = np.concatenate([depth, depth])
    peaks, _ = find_peaks(doubled, height=NOTCH_UM, prominence=NOTCH_UM, distance=max(1, int(100.0 / um)))
    return np.unique(peaks % n)


def segments_cross(a, b, c, d) -> bool:
    def side(p, q, r):
        return np.sign((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))
    return side(a, b, c) != side(a, b, d) and side(c, d, a) != side(c, d, b)


def neck_chord(points, depth, peaks, filled, hot_at, cold_at):
    """The pair of notches that best pinches the tissue between the seeds: (a, b) in px, or None.

    The chord must cross the segment between the seed centres and lie on the
    tissue; of those, the one with the largest (depth a + depth b) / length.
    """
    best, score = None, -np.inf
    for i in range(len(peaks)):
        for j in range(i + 1, len(peaks)):
            a, b = points[peaks[i]], points[peaks[j]]
            if not segments_cross(a, b, hot_at, cold_at):
                continue
            length = float(np.hypot(*(b - a)))
            if length < 5:
                continue
            t = np.linspace(0, 1, 50)[:, None]
            along = np.round(a + t * (b - a)).astype(int)
            along = np.clip(along, 0, np.array(filled.shape) - 1)
            if filled[along[:, 0], along[:, 1]].mean() < 0.85:
                continue
            s = (depth[peaks[i]] + depth[peaks[j]]) / length
            if s > score:
                best, score = (a, b), s
    return best


def across(at, direction, span):
    """A straight line through *at* along *direction*, *span* px each way (px points)."""
    return np.array([at - span * direction, at + span * direction])


def measure(name):
    from scipy import ndimage
    from skimage.morphology import convex_hull_image

    overview, manual_band, manual_line, base = jt.load_case(name)
    um = overview.um
    options = jn.JunctionOptions(save=False, **base)
    result = jn.find_junction(overview, options)
    row = {"name": name, "status": result.status, "seam price": result.metrics.get("Seam price", np.nan),
           "solidity": result.metrics.get("Solidity", np.nan),
           "labelled share": result.metrics.get("Labelled share of tissue", np.nan)}
    filled = jn._largest(result.filled)
    if (not filled.any() or result.seeds is None
            or not (result.seeds[0] & filled).any() or not (result.seeds[1] & filled).any()):
        row["note"] = "no seeds on the main piece of tissue"
        return row, None
    region = ndimage.binary_dilation(result.filled, iterations=max(1, int(np.ceil(base["margin_um"] / um))))
    score = lambda line_um: jt.line_distance(manual_line, line_um, region, um)[0]  # noqa: E731

    # The outline and its concavities.
    points = outline(filled)
    hull = convex_hull_image(filled)
    hull_depth = ndimage.distance_transform_edt(hull) * um
    depth = ndimage.map_coordinates(hull_depth, points.T, order=1)
    arc = max(1, int(round(ARC_UM / um)))
    local = windowed_max(depth, arc)
    peaks = notches(depth, um)
    row["notches"] = len(peaks)
    row["deepest notch (um)"] = float(depth.max())

    # Where a junction meets the outline: its first and last points on the tissue.
    near_tissue = ndimage.binary_dilation(filled, iterations=2)

    def ends_of(line_um, prefix):
        dense_px = jt.dense(line_um) / um
        rc = np.clip(np.round(dense_px).astype(int), 0, np.array(filled.shape) - 1)
        on_ = near_tissue[rc[:, 0], rc[:, 1]]
        ends = dense_px[on_][[0, -1]] if on_.sum() >= 2 else np.zeros((0, 2))
        for k, end in enumerate(ends):
            nearest = int(np.argmin(np.hypot(*(points - end).T)))
            row[f"{prefix}end {k} concavity (um)"] = float(local[nearest])
            row[f"{prefix}end {k} concavity percentile"] = float((local < local[nearest]).mean() * 100)
            if len(peaks):
                around = np.abs(peaks - nearest)
                row[f"{prefix}end {k} to notch (um)"] = float(np.minimum(around, len(points) - around).min() * um)
        return dense_px, on_

    dense, on = ends_of(manual_line, "")
    if result.line_um is not None:
        ends_of(np.asarray(result.line_um), "finder ")
    if len(peaks):
        idx = np.arange(len(points))
        gap = np.abs(idx[:, None] - peaks[None, :])
        row["any point to notch (um, median)"] = float(np.median(np.minimum(gap, len(points) - gap).min(axis=1)) * um)

    # The seed axis and the width of the tissue along it.
    hot_at = np.argwhere(result.seeds[0] & filled).mean(axis=0)
    cold_at = np.argwhere(result.seeds[1] & filled).mean(axis=0)
    axis = cold_at - hot_at
    length = float(np.hypot(*axis))
    u = axis / length
    v = np.array([-u[1], u[0]])
    pix = np.argwhere(filled)
    t = (pix - hot_at) @ u / length
    edges = np.arange(np.floor(t.min() / BIN) * BIN, t.max() + BIN, BIN)
    counts, _ = np.histogram(t, edges)
    width_um = counts * um / (BIN * length)            # px² per bin / bin length in px → px, × um
    centres = (edges[:-1] + edges[1:]) / 2
    between = (centres > 0.05) & (centres < 0.95) & (counts > 0)
    t_neck = float(centres[between][np.argmin(width_um[between])]) if between.any() else np.nan
    row.update({"seed distance (um)": length * um, "hot seed depth (um)": float(
        ndimage.distance_transform_edt(filled)[tuple(np.round(hot_at).astype(int))] * um),
        "cold seed depth (um)": float(ndimage.distance_transform_edt(filled)[tuple(np.round(cold_at).astype(int))] * um)})
    on_t = lambda line_px: float(np.median(((line_px - hot_at) @ u) / length))  # noqa: E731
    manual_px = dense[on]
    row["t junction"] = on_t(manual_px) if len(manual_px) else np.nan
    row["t neck"] = t_neck
    row["neck width / median width"] = (float(width_um[between].min() / np.median(width_um[between]))
                                        if between.any() else np.nan)
    if result.line_um is not None:
        row["t finder"] = on_t(np.asarray(result.line_um) / um)
    # The cumulative share of tissue up to the junction, from the labelled seed's end.
    row["tissue share before junction"] = float((t < row["t junction"]).mean()) if np.isfinite(row["t junction"]) else np.nan
    # Angle of the junction to the seed axis (90 = square across it).
    if len(manual_px) >= 3:
        centred = manual_px - manual_px.mean(axis=0)
        direction = np.linalg.svd(centred, full_matrices=False)[2][0]
        row["junction angle to axis (deg)"] = float(np.degrees(np.arccos(min(1.0, abs(direction @ u)))))

    # Junctions from shape alone, scored as the finder is.
    span = 2 * max(filled.shape)
    lines = {}
    if result.line_um is not None:
        lines["finder"] = np.asarray(result.line_um)
    chord = neck_chord(points, depth, peaks, filled, hot_at, cold_at) if len(peaks) >= 2 else None
    if chord is not None:
        a, b = chord
        lines["notch chord"] = np.array([a - 0.3 * (b - a), a, b, b + 0.3 * (b - a)]) * um
    if np.isfinite(t_neck):
        lines["square at the narrowest"] = across(hot_at + t_neck * axis, v, span) * um
    lines["square halfway"] = across(hot_at + 0.5 * axis, v, span) * um
    for key, line in lines.items():
        row[f"{key} error (um)"] = score(line)
    geometry = {"filled": filled, "hull": hull, "points": points, "depth": depth, "peaks": peaks,
                "hot": hot_at, "cold": cold_at, "seeds": result.seeds, "manual": manual_line, "lines": lines,
                "um": um, "profile": (centres, width_um), "stain": overview.channels[overview.nuclear]}
    return row, geometry


def main() -> None:
    import pandas as pd

    tune, held = jt.split(jt.annotated_samples(keep_excluded=True))
    names = tune + held
    with ProcessPoolExecutor(10) as pool:
        out = list(pool.map(measure, names))
    rows = [{**r, "set": "held out" if r["name"] in held else "tuning"} for r, _g in out]
    pd.DataFrame(rows).to_csv(TABLE, index=False)
    with GEOMETRY.open("wb") as f:
        pickle.dump({r["name"]: g for r, g in out}, f)
    print("written", TABLE, GEOMETRY)


if __name__ == "__main__":
    main()
