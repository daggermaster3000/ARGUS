"""The junction as a minimum cut of the nuclei graph, against the pixel cut and the hand-drawn lines.

Used by ``01_nuclei_graph_cut.ipynb``; run on its own to (re)compute the cache::

    python notebooks/junction_graph/nuclei_graph.py        # every hand-drawn section not done yet
    python notebooks/junction_graph/nuclei_graph.py 4      # just the first four, to time it

The junction finder cuts a 5 µm pixel grid between two seeds, each pixel as dear
to cut as the nuclear stain is bright round it (a dark line between the organoids
is cheap). Here the same cut is made on the **nuclei** instead: every nucleus
Cellpose found is a node, joined to its neighbours, and the cut runs between
nuclei. Where two organoids meet the nuclei are sparser (the dark line *is* fewer
nuclei) and the label changes from one organoid's to the other's, so what an edge
costs to cut can be read off the graph itself:

* **the graph** — a **Gabriel graph** (two nuclei are joined when no other
  nucleus lies in the circle with them as diameter: neighbours that see each
  other directly, no long edges past a third nucleus), or the Delaunay
  triangulation it is taken from, for comparison. Edges longer than
  :data:`MAX_EDGE_UM` are dropped (across holes and tears);
* **what an edge costs**: uniform; ``1/length`` or ``1/length²`` (a long edge,
  across sparse nuclei, is cheap); ``(local length)^-p`` — the edge's length over
  the typical edge within :data:`SURROUND_UM`, to the power p (3, 6, 10), as the
  pixel finder prices the stain against its surroundings to the power 6; times the two nuclei's **degree** (few
  neighbours: the edge of a gap); times the **local density** (nuclei within
  :data:`DENSITY_UM`); times a **label change** factor (the cells' own AF488 − AF555,
  each scaled 0–1, smoothed over :data:`LABEL_UM`: an edge across which it
  changes is cheaper, down to :data:`LABEL_FLOOR`).

Everything else is the pixel finder's, so only the cut differs: the same
overview, tissue and **seeds** (``find_junction`` with the defaults; nuclei inside
a seed are tied to it), and the same line from the two halves
(``trace_interface``; each overview pixel takes the side of its nearest nucleus).
The pixel finder's straightening (``straighten_cut``) is not applied: it cuts
again from the seeds on its own price map, so on a flat price it gives the same
line whatever the graph cut was — the graph's own cut is compared as it is.

Scored as ``02_junction_seeding``: the mean symmetric distance to the hand-drawn
line over the tissue, capped at 300 µm (``junction_tuning.line_distance``,
``capped``). The nuclei are those already segmented: the newest cells-analysis run
of each section, else ``count_cells.py``'s cache. Nothing is segmented here.

Output in ``junction_eval/graph/``: ``results.jsonl`` (one row per section and
variant), ``<name>.npz`` per section (nuclei, Gabriel edges, the sides and lines
of the main variants, the pixel line) for the notebook's pictures, and
``crossing.jsonl`` (edges crossing the hand-drawn line against the rest).
"""

from __future__ import annotations

import os

THREADS = 2
for _variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_variable, str(THREADS))

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parent / "validation"))
import junction_tuning as jt  # noqa: E402

from microscopy_viewer import junction as jn  # noqa: E402
from microscopy_viewer import junction_analysis as ja  # noqa: E402

DATA = Path(r"D:\Slidescanner extracted")
OUT = jt.OUT / "graph"
RESULTS = OUT / "results.jsonl"
CROSSING = OUT / "crossing.jsonl"
SECTIONS = jt.OUT / "annotated_all.csv"
COUNT_CELLS = jt.OUT / "cells"

#: Longest edge kept, in µm: longer ones cross holes, tears and the slide.
MAX_EDGE_UM = 60.0
#: Radius over which a nucleus's neighbours are counted for the local density, in µm.
DENSITY_UM = 25.0
#: Radius over which each cell's label is smoothed before its change is read, in µm
#: (as the pixel finder's ``LABEL_CHANGE_UM``).
LABEL_UM = 50.0
#: Radius of the neighbourhood an edge's length is compared with for the
#: ``local`` costs, in µm (as the pixel finder's ``LINE_SURROUND_UM``).
SURROUND_UM = 60.0
#: Lowest the label-change factor takes an edge's cost (as ``LABEL_CHANGE_FLOOR``).
LABEL_FLOOR = 0.3
WORKERS = 6

#: The variants: graph, and what an edge costs to cut (factors multiplied).
VARIANTS = {
    "Gabriel, uniform": ("gabriel", ()),
    "Gabriel, 1/length": ("gabriel", ("length",)),
    "Gabriel, 1/length²": ("gabriel", ("length2",)),
    "Gabriel, 1/length², degree": ("gabriel", ("length2", "degree")),
    "Gabriel, 1/length², density": ("gabriel", ("length2", "density")),
    "Gabriel, 1/length², label change": ("gabriel", ("length2", "label")),
    "Gabriel, 1/length², degree, label change": ("gabriel", ("length2", "degree", "label")),
    "Gabriel, uniform, label change": ("gabriel", ("label",)),
    "Gabriel, (local length)^-3": ("gabriel", ("local3",)),
    "Gabriel, (local length)^-6": ("gabriel", ("local6",)),
    "Gabriel, (local length)^-10": ("gabriel", ("local10",)),
    "Gabriel, (local length)^-6, label change": ("gabriel", ("local6", "label")),
    "Delaunay, 1/length²": ("delaunay", ("length2",)),
    "Delaunay, 1/length², label change": ("delaunay", ("length2", "label")),
}
#: The variants whose sides and lines are kept for the pictures.
KEEP = ("Gabriel, 1/length²", "Gabriel, (local length)^-6, label change")


# ---------------------------------------------------------------------------
# The nuclei
# ---------------------------------------------------------------------------


def nuclei_of(name: str, saved: dict) -> tuple[np.ndarray, dict[str, np.ndarray], str]:
    """Each nucleus's centre (µm, ``(y, x)``) and its means per channel, from the newest segmentation."""
    import pandas as pd

    if name in saved:
        cells = pd.read_csv(saved[name].file("cells_file"))
        points = cells[["Centroid Y (µm)", "Centroid X (µm)"]].to_numpy(float)
        means = {c[:-5]: cells[c].to_numpy(float) for c in cells.columns
                 if c.endswith(" mean") and not c.endswith(" ring mean")}
        return points, means, "cells analysis"
    d = np.load(COUNT_CELLS / f"{name}.npz")
    voxel = d["voxel_yx"].astype(float)
    points = np.column_stack([d["y"] * voxel[0], d["x"] * voxel[1]])
    means = {str(c): d[f"{c}|3|raw|mean"].astype(float) for c in d["channels"] if f"{c}|3|raw|mean" in d.files}
    return points, means, "count_cells"


# ---------------------------------------------------------------------------
# The graph
# ---------------------------------------------------------------------------


def delaunay_edges(points: np.ndarray) -> np.ndarray:
    """Every edge of the Delaunay triangulation, once, as ``(a, b)`` with a < b."""
    from scipy.spatial import Delaunay

    tri = Delaunay(points).simplices
    edges = np.vstack([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [0, 2]]])
    edges.sort(axis=1)
    return np.unique(edges, axis=0)


def gabriel(points: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """The Delaunay edges no other nucleus lies inside of (the circle on the edge as diameter)."""
    from scipy.spatial import cKDTree

    a, b = points[edges[:, 0]], points[edges[:, 1]]
    middle, radius = (a + b) / 2, np.hypot(*(a - b).T) / 2
    nearest, _ = cKDTree(points).query(middle, k=3)
    # The two ends are at exactly the radius; anything nearer is a third nucleus inside.
    inside = np.sort(nearest, axis=1)
    third = np.where(np.isclose(inside[:, 0], radius) & np.isclose(inside[:, 1], radius), inside[:, 2], inside[:, 0])
    return edges[third >= radius * (1 - 1e-9)]


def label_share(points: np.ndarray, means: dict[str, np.ndarray]):
    """Each nucleus's AF488 − AF555 (each scaled 0–1 over the cells), averaged over :data:`LABEL_UM` round it."""
    from scipy.spatial import cKDTree

    names = list(means)
    label = jn.pick_channel(names, "", jn.LABEL_HINTS)
    other = jn.pick_channel(names, "", jn.OTHER_HINTS)

    def scaled(values):
        low, high = np.nanpercentile(values, (10, 98))
        return np.clip((values - low) / max(high - low, 1e-9), 0, 1)

    if label is None:
        return None
    share = scaled(means[names[label]])
    second = means[names[other]] if other is not None and other != label else None
    if second is not None:
        share = share - scaled(second)
    share = np.nan_to_num(share)
    groups = cKDTree(points).query_ball_point(points, LABEL_UM, workers=THREADS)
    return np.array([share[g].mean() for g in groups])


def capacities(points, edges, factors, smoothed=None) -> np.ndarray:
    """What cutting each edge costs (≥ 0), from *factors* (see :data:`VARIANTS`)."""
    from scipy.spatial import cKDTree

    length = np.hypot(*(points[edges[:, 0]] - points[edges[:, 1]]).T)
    typical = float(np.median(length))
    cap = np.ones(len(edges))
    if "length" in factors:
        cap *= typical / np.maximum(length, 1e-6)
    if "length2" in factors:
        cap *= (typical / np.maximum(length, 1e-6)) ** 2
    local = [f for f in factors if f.startswith("local")]
    if local:
        # Each edge against the edges round it, raised to a power — as the pixel
        # finder divides the stain by its 60 µm surroundings and raises it to 6.
        power = float(local[0][len("local"):])
        incident = np.zeros(len(points))
        order = np.argsort(np.concatenate([edges[:, 0], edges[:, 1]]), kind="stable")
        ends = np.concatenate([edges[:, 0], edges[:, 1]])[order]
        lengths = np.concatenate([length, length])[order]
        starts = np.searchsorted(ends, np.arange(len(points)))
        stops = np.searchsorted(ends, np.arange(len(points)), side="right")
        for node in np.flatnonzero(stops > starts):
            incident[node] = np.median(lengths[starts[node]:stops[node]])
        incident[stops == starts] = typical
        groups = cKDTree(points).query_ball_point(points, SURROUND_UM, workers=THREADS)
        surround = np.array([incident[g].mean() for g in groups])
        relative = length / np.sqrt(surround[edges[:, 0]] * surround[edges[:, 1]])
        cap *= np.clip(relative, 0.3, 3.0) ** -power
    if "degree" in factors:
        degree = np.bincount(edges.ravel(), minlength=len(points)).astype(float)
        cap *= np.minimum(degree[edges[:, 0]], degree[edges[:, 1]]) / max(float(np.median(degree)), 1.0)
    if "density" in factors:
        count = np.array([len(g) for g in cKDTree(points).query_ball_point(points, DENSITY_UM, workers=THREADS)], float)
        cap *= np.sqrt(count[edges[:, 0]] * count[edges[:, 1]]) / max(float(np.median(count)), 1.0)
    if "label" in factors and smoothed is not None:
        change = np.abs(smoothed[edges[:, 0]] - smoothed[edges[:, 1]]) / np.maximum(length, 1e-6)
        top = float(np.percentile(change, 99))
        cap *= np.clip(1.0 - change / max(top, 1e-12), LABEL_FLOOR, 1.0)
    return cap


def cut(n: int, edges: np.ndarray, cap: np.ndarray, hot: np.ndarray, cold: np.ndarray) -> np.ndarray | None:
    """Minimum cut of the graph between the nuclei in *hot* and in *cold*: True on the *hot* side."""
    from scipy import sparse
    from scipy.sparse import csgraph

    if not hot.any() or not cold.any():
        return None
    source, sink = n, n + 1
    weight = np.maximum((cap / max(float(np.median(cap)), 1e-12) * 1000).astype(np.int64), 1).astype(np.int32)
    huge = np.int32(2 ** 30)
    hot_nodes, cold_nodes = np.flatnonzero(hot), np.flatnonzero(cold)
    rows = np.concatenate([edges[:, 0], edges[:, 1], np.full(hot_nodes.size, source), cold_nodes])
    cols = np.concatenate([edges[:, 1], edges[:, 0], hot_nodes, np.full(cold_nodes.size, sink)])
    data = np.concatenate([weight, weight, np.full(hot_nodes.size, huge), np.full(cold_nodes.size, huge)])
    graph = sparse.csr_matrix((data, (rows, cols)), shape=(n + 2, n + 2), dtype=np.int32)
    flow = csgraph.maximum_flow(graph, source, sink).flow
    residual = graph - flow
    residual.data = np.maximum(residual.data, 0)
    residual.eliminate_zeros()
    reached = csgraph.breadth_first_order(residual, source, return_predecessors=False)
    side = np.zeros(n + 2, bool)
    side[reached] = True
    return side[:n]


def halves_of(points: np.ndarray, side: np.ndarray, filled: np.ndarray, um: float) -> np.ndarray | None:
    """The tissue as two halves, each overview pixel taking the side of its nearest nucleus."""
    from scipy.spatial import cKDTree

    rows, cols = np.nonzero(filled)
    _d, nearest = cKDTree(points).query(np.column_stack([rows, cols]) * um, workers=THREADS)
    labelled = np.zeros(filled.shape, bool)
    labelled[rows, cols] = side[nearest]
    halves = np.zeros(filled.shape, np.uint8)
    halves[jn._largest(labelled & filled)] = 1
    halves[jn._largest(~labelled & filled)] = 2
    if not (halves == 1).any() or not (halves == 2).any():
        return None
    return halves


# ---------------------------------------------------------------------------
# One section
# ---------------------------------------------------------------------------


def side_of_line(points: np.ndarray, line: np.ndarray):
    """Each point's signed distance to *line* (µm; the sign says which side) and the distance itself."""
    from scipy.spatial import cKDTree

    dense = jt.dense(line)
    distance, nearest = cKDTree(dense).query(points)
    ahead = dense[np.minimum(nearest + 1, len(dense) - 1)] - dense[np.maximum(nearest - 1, 0)]
    offset = points - dense[nearest]
    cross = ahead[:, 0] * offset[:, 1] - ahead[:, 1] * offset[:, 0]
    return np.sign(cross) * distance, distance


def crossing_stats(name, points, edges, smoothed, manual_line) -> dict:
    """The premise: are the Gabriel edges across the hand-drawn line longer, or across a label change?"""
    signed, _distance = side_of_line(points, manual_line)
    a, b = edges[:, 0], edges[:, 1]
    middle = (points[a] + points[b]) / 2
    _s, mid_distance = side_of_line(middle, manual_line)
    crossing = (np.sign(signed[a]) != np.sign(signed[b])) & (mid_distance < 30)
    near = (mid_distance >= 30) & (mid_distance < 300)
    length = np.hypot(*(points[a] - points[b]).T)
    degree = np.bincount(edges.ravel(), minlength=len(points)).astype(float)
    out = {"name": name, "crossing": int(crossing.sum()), "near": int(near.sum()),
           "length crossing (µm)": float(np.median(length[crossing])) if crossing.any() else np.nan,
           "length near (µm)": float(np.median(length[near])) if near.any() else np.nan,
           "degree crossing": float(np.median(np.minimum(degree[a], degree[b])[crossing])) if crossing.any() else np.nan,
           "degree near": float(np.median(np.minimum(degree[a], degree[b])[near])) if near.any() else np.nan}
    if smoothed is not None:
        change = np.abs(smoothed[a] - smoothed[b]) / np.maximum(length, 1e-6)
        out["label change crossing (/µm)"] = float(np.median(change[crossing])) if crossing.any() else np.nan
        out["label change near (/µm)"] = float(np.median(change[near])) if near.any() else np.nan
    return out


_SAVED: dict = {}


def saved_runs() -> dict:
    """The segmented samples of every cells-analysis run, read once per process."""
    if not _SAVED:
        _SAVED.update(ja.segmented_samples(DATA))
    return _SAVED


def run_section(name: str) -> tuple[list[dict], dict]:
    from scipy import ndimage

    started = time.perf_counter()
    saved = saved_runs()
    overview, manual_band, manual_line, base = jt.load_case(name)
    options = jn.JunctionOptions(save=False, **base)
    pixel = jn.find_junction(overview, options)
    um = overview.um
    region = ndimage.binary_dilation(pixel.filled, iterations=max(1, int(np.ceil(base["margin_um"] / um))))
    rows = []

    def score(variant, line_um, seconds, extra=None):
        row = {"name": name, "variant": variant, "seconds": round(seconds, 2), **(extra or {})}
        if line_um is None:
            row["found"] = False
        else:
            mean_dist, hd95, within100 = jt.line_distance(manual_line, line_um, region, um)
            row.update(found=True, mean_dist=mean_dist, hd95=hd95, within100=within100)
        row["capped"] = jt.capped(row)
        rows.append(row)

    score("pixel cut (the finder's defaults)", pixel.line_um, time.perf_counter() - started,
          {"seam_price": pixel.metrics.get("Seam price", np.nan),
           "no_seam_fallback": bool(pixel.metrics.get("No-seam fallback", 0.0))})
    keep = {"pixel_line": np.zeros((0, 2)) if pixel.line_um is None else pixel.line_um, "manual_line": manual_line,
            "um": um}
    if pixel.seeds is None:
        return rows, keep

    points, means, source = nuclei_of(name, saved)
    ok = np.isfinite(points).all(axis=1)
    rc = np.round(points / um).astype(int)
    inside = ok & (rc[:, 0] >= 0) & (rc[:, 1] >= 0) & (rc[:, 0] < region.shape[0]) & (rc[:, 1] < region.shape[1])
    inside[inside] = region[rc[inside, 0], rc[inside, 1]]
    points, rc = points[inside], rc[inside]
    means = {k: v[inside] for k, v in means.items()}
    hot = pixel.seeds[0][rc[:, 0], rc[:, 1]]
    cold = pixel.seeds[1][rc[:, 0], rc[:, 1]]
    smoothed = label_share(points, means)

    graphs = {}
    every = delaunay_edges(points)
    length = np.hypot(*(points[every[:, 0]] - points[every[:, 1]]).T)
    graphs["delaunay"] = every[length <= MAX_EDGE_UM]
    graphs["gabriel"] = gabriel(points, every)
    graphs["gabriel"] = graphs["gabriel"][np.hypot(*(points[graphs["gabriel"][:, 0]] -
                                                     points[graphs["gabriel"][:, 1]]).T) <= MAX_EDGE_UM]
    crossing = crossing_stats(name, points, graphs["gabriel"], smoothed, manual_line)
    common = {"nuclei": int(len(points)), "source": source, "hot nuclei": int(hot.sum()), "cold nuclei": int(cold.sum())}
    keep.update(points=points.astype(np.float32), gabriel=graphs["gabriel"].astype(np.int32),
                smoothed=np.zeros(0) if smoothed is None else smoothed.astype(np.float32))
    for variant, (graph, factors) in VARIANTS.items():
        begun = time.perf_counter()
        edges = graphs[graph]
        side = cut(len(points), edges, capacities(points, edges, factors, smoothed), hot, cold)
        halves = halves_of(points, side, pixel.filled, um) if side is not None else None
        took = time.perf_counter() - begun
        line = jn.trace_interface(halves, um) if halves is not None else None
        extra = {**common, "edges": int(len(edges)),
                 "cut edges": int((side[edges[:, 0]] != side[edges[:, 1]]).sum()) if side is not None else 0}
        score(variant, line * um if line is not None else None, took, extra)
        if variant in KEEP and side is not None:
            key = KEEP.index(variant)
            keep[f"side_{key}"] = side
            keep[f"line_{key}"] = np.zeros((0, 2)) if line is None else line * um
    keep["crossing"] = crossing
    return rows, keep


def run_and_save(name: str) -> str:
    cv_priority()
    started = time.perf_counter()
    rows, keep = run_section(name)
    crossing = keep.pop("crossing", None)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"{name}.npz", **{k: np.asarray(v) for k, v in keep.items()})
    return json.dumps({"rows": rows, "crossing": crossing, "seconds": time.perf_counter() - started}, default=float)


def cv_priority() -> None:
    try:
        import ctypes

        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
    except Exception:
        pass


def done() -> set[str]:
    if not RESULTS.is_file():
        return set()
    return {json.loads(line)["name"] for line in RESULTS.read_text(encoding="utf-8").splitlines() if line.strip()}


def main(argv: list[str]) -> int:
    import pandas as pd
    from concurrent.futures import ProcessPoolExecutor, as_completed

    names = [n for n in pd.read_csv(SECTIONS)["name"] if n not in done()]
    if argv:
        names = names[: int(argv[0])]
    print(f"{len(names)} section(s) -> {OUT}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=WORKERS) as pool, open(RESULTS, "a", encoding="utf-8") as results, \
            open(CROSSING, "a", encoding="utf-8") as crossings:
        futures = {pool.submit(run_and_save, n): n for n in names}
        for index, future in enumerate(as_completed(futures), start=1):
            name = futures[future]
            try:
                out = json.loads(future.result())
            except Exception as exc:
                print(f"[{index}/{len(names)}] {name}: FAILED {exc!r}", flush=True)
                continue
            for row in out["rows"]:
                results.write(json.dumps(row, default=float) + "\n")
            if out["crossing"]:
                crossings.write(json.dumps(out["crossing"], default=float) + "\n")
            results.flush()
            crossings.flush()
            best = min((r for r in out["rows"] if r["variant"] != "pixel cut (the finder's defaults)"),
                       key=lambda r: r["capped"], default=None)
            pixel = out["rows"][0]["capped"]
            print(f"[{index}/{len(names)}] {name}: pixel {pixel:.0f} µm, best graph "
                  f"{best['capped'] if best else float('nan'):.0f} µm, {out['seconds']:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
