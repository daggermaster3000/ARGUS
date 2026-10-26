"""Tuning the junction finder against junctions drawn by hand.

Used by ``junction_evaluation.ipynb``; run on its own to (re)do the search::

    python notebooks/validation/junction_tuning.py --trials 60

Three parts:

* **A cache of overviews.** Each annotated sample's overview (what the finder
  works on), its hand-drawn band on the same grid and the line it was drawn
  from, so a trial does not read a 1 GB scan again.
* **A scorer** that runs the finder with a given set of settings on every
  sample, in parallel, and measures the line against the hand-drawn one.
* **Bayesian optimisation** of the settings that shape the line: a Gaussian
  process fitted to the trials so far proposes the next ones where it expects
  the most improvement. Only the *tuning* samples guide it; the *held-out*
  ones are scored too but never seen by the search, so their score says
  whether what it found holds on samples it was not tuned on.

Every trial's line for every sample is kept, which is what lets the notebook
also try rules that pick one line among several candidates.
"""

from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from microscopy_viewer import fusion as fu  # noqa: E402
from microscopy_viewer import junction as jn  # noqa: E402

PICKS = Path(r"D:\Slidescanner extracted\picks_20261020_0742.csv")
OUT = PICKS.parent / "junction_eval"
CACHE = OUT / "overviews"
TRIALS = OUT / "trials.jsonl"

#: Annotated samples left out of every evaluation, and why.
EXCLUDE = {
    "2025_09_26_Asssembloid_028-027_s20": "organoids separated: no junction to find",
}

#: Where a second person's hand-drawn region is stored, for the agreement between annotators.
SECOND_KEY = fu.MANUAL_KEY + " quillan"

#: A line further off than this counts as this far: one disaster should not
#: outweigh everything else, and "no junction found" scores it too.
CAP_UM = 300.0

#: What is tuned: (name, low, high, log scale). Module constants of junction.py
#: are set in the worker before each run; the others are JunctionOptions.
PARAMS = [
    ("bend_um", 100.0, 5000.0, True),
    ("tear_price", 0.0, 1.5, False),
    ("LINE_POWER", 2.0, 10.0, False),
    ("LINE_SMOOTH_UM", 4.0, 20.0, False),
    ("LINE_SURROUND_UM", 30.0, 150.0, False),
    ("JUNCTION_SMOOTH_UM", 50.0, 300.0, False),
]
CONSTANTS = {"LINE_POWER", "LINE_SMOOTH_UM", "LINE_SURROUND_UM", "JUNCTION_SMOOTH_UM"}


def defaults() -> dict:
    options = jn.JunctionOptions()
    return {name: float(getattr(jn, name) if name in CONSTANTS else getattr(options, name)) for name, *_ in PARAMS}


# ---------------------------------------------------------------------------
# The cache
# ---------------------------------------------------------------------------


def annotated_samples(keep_excluded: bool = False):
    """The picks carrying a hand-drawn region, with their prescreen categories.

    Those in :data:`EXCLUDE` are left out unless *keep_excluded*: :func:`split`
    wants them, so that leaving a sample out does not reshuffle the held-out set.
    """
    import pandas as pd

    from microscopy_viewer import ims_store

    picks = pd.read_csv(PICKS, encoding="utf-8-sig")
    picks["name"] = [Path(p).name.replace(".ome.tiff", "") for p in picks["path"]]
    picks["manual"] = [fu.MANUAL_KEY in ims_store.list_labels(p) for p in picks["path"]]
    picks = picks[picks["manual"]]
    if not keep_excluded:
        picks = picks[~picks["name"].isin(EXCLUDE)]
    return picks.reset_index(drop=True)


def build_cache(samples) -> None:
    """Overview, hand-drawn band and line, and the band settings, per sample (once)."""
    from microscopy_viewer import ims_store, loaders

    CACHE.mkdir(parents=True, exist_ok=True)
    for row in samples.itertuples():
        target = CACHE / f"{row.name}.npz"
        if target.exists():
            continue
        labels, attrs = ims_store.load_labels(row.path, fu.MANUAL_KEY)
        number = lambda key, default: float(ims_store._text(attrs[key])) if key in attrs else default
        specs = loaders.load_path(Path(row.path))
        try:
            overview = jn.build_overview(specs, jn.JunctionOptions(save=False))
        finally:
            loaders.release(Path(row.path))
        shape = overview.channels.shape[1:]
        band = (np.asarray(labels)[:: overview.step, :: overview.step] > 0)[: shape[0], : shape[1]]
        np.savez_compressed(
            target, channels=overview.channels, names=np.array(overview.names), um=overview.um,
            voxel_yx=np.array(overview.voxel_yx), shape_yx=np.array(overview.shape_yx), step=overview.step,
            nuclear=overview.nuclear, label=overview.label, other=-1 if overview.other is None else overview.other,
            focus=overview.focus, saturated=overview.saturated,
            nuclear_mean=overview.nuclear_mean if overview.nuclear_mean is not None else np.zeros(0),
            manual_band=band, manual_line=np.asarray(attrs["line_yx_um"], dtype=float).reshape(-1, 2),
            width_um=number("band_width_um", fu.DEFAULT_WIDTH_UM), gap_um=number("start_gap_um", fu.DEFAULT_GAP_UM),
            margin_um=number("past_tissue_edge_um", fu.TISSUE_MARGIN_UM),
        )
        print(f"cached {row.name}", flush=True)


_CASES: dict = {}


def load_case(name: str):
    """The cached overview and hand-drawn junction of one sample.

    A worker keeps its last few: with thirty workers, keeping every sample in
    each would hold tens of gigabytes.
    """
    if name in _CASES:
        _CASES[name] = _CASES.pop(name)        # most recently used goes last
    else:
        while len(_CASES) >= 3:
            _CASES.pop(next(iter(_CASES)))
        d = np.load(CACHE / f"{name}.npz")
        overview = jn.Overview(
            channels=d["channels"], names=[str(n) for n in d["names"]], um=float(d["um"]),
            voxel_yx=tuple(float(v) for v in d["voxel_yx"]), shape_yx=tuple(int(v) for v in d["shape_yx"]),
            step=int(d["step"]), nuclear=int(d["nuclear"]), label=int(d["label"]),
            other=None if int(d["other"]) < 0 else int(d["other"]), focus=float(d["focus"]),
            saturated=float(d["saturated"]), nuclear_mean=d["nuclear_mean"] if d["nuclear_mean"].size else None,
        )
        base = {k: float(d[k]) for k in ("width_um", "gap_um", "margin_um")}
        _CASES[name] = (overview, d["manual_band"], d["manual_line"], base)
    return _CASES[name]


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def dice(a, b):
    total = a.sum() + b.sum()
    return float(2.0 * (a & b).sum() / total) if total else float("nan")


def dense(line_um, step_um=2.0):
    line = np.asarray(line_um, dtype=float)
    pieces = [np.linspace(a, b, max(2, int(np.ceil(np.hypot(*(b - a)) / step_um)) + 1)) for a, b in zip(line[:-1], line[1:])]
    return np.vstack(pieces)


def on_tissue(points_um, region, um):
    rc = np.round(points_um / um).astype(int)
    ok = (rc[:, 0] >= 0) & (rc[:, 1] >= 0) & (rc[:, 0] < region.shape[0]) & (rc[:, 1] < region.shape[1])
    keep = np.zeros(len(points_um), bool)
    keep[ok] = region[rc[ok, 0], rc[ok, 1]]
    return points_um[keep]


def line_distance(line_a, line_b, region, um):
    """Mean symmetric distance (µm) between two lines over *region*, and its 95th percentile."""
    from scipy.spatial import cKDTree

    a = on_tissue(dense(line_a), region, um)
    b = on_tissue(dense(line_b), region, um)
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan"), float("nan")
    d_ab = cKDTree(b).query(a)[0]
    d_ba = cKDTree(a).query(b)[0]
    both = np.concatenate([d_ab, d_ba])
    return float(both.mean()), float(np.percentile(both, 95)), float((d_ab <= 100).mean())


def run_case(task):
    """One sample under one set of settings: the line, and how far it is from the hand-drawn one."""
    from scipy import ndimage

    name, params = task
    overview, manual_band, manual_line, base = load_case(name)
    saved = {k: getattr(jn, k) for k in CONSTANTS}
    try:
        for key in CONSTANTS & set(params):
            setattr(jn, key, float(params[key]))
        options = jn.JunctionOptions(save=False, **base,
                                     **{k: v if isinstance(v, str) else float(v) for k, v in params.items() if k not in CONSTANTS})
        started = time.perf_counter()
        result = jn.find_junction(overview, options)
        took = time.perf_counter() - started
    finally:
        for key, value in saved.items():
            setattr(jn, key, value)
    out = {"name": name, "seconds": took, "status": result.status,
           # How the tissue was split (0 cut, 1 walk) and what the cut said about itself.
           "walked": result.metrics.get("Split", 0.0), "cut_price": result.metrics.get("Cut price", float("nan")),
           "sharpest_turn": result.metrics.get("Sharpest turn (deg)", float("nan"))}
    if result.line_um is None or result.band is None:
        return {**out, "found": False}
    um = overview.um
    region = ndimage.binary_dilation(result.filled, iterations=max(1, int(np.ceil(base["margin_um"] / um))))
    mean_dist, hd95, within100 = line_distance(manual_line, result.line_um, region, um)
    return {**out, "found": True, "mean_dist": mean_dist, "hd95": hd95, "within100": within100,
            "dice": dice(result.band & region, manual_band & region), "line": np.round(result.line_um, 1).tolist()}


def capped(row) -> float:
    if not row.get("found") or not np.isfinite(row.get("mean_dist", np.nan)):
        return CAP_UM
    return min(float(row["mean_dist"]), CAP_UM)


def score(rows, names) -> float:
    """What the search minimises: the mean capped distance over *names*."""
    by = {r["name"]: r for r in rows}
    return float(np.mean([capped(by[n]) for n in names if n in by]))


# ---------------------------------------------------------------------------
# Bayesian optimisation
# ---------------------------------------------------------------------------


def to_unit(params: dict) -> np.ndarray:
    out = []
    for name, low, high, log in PARAMS:
        v = float(params[name])
        out.append((np.log(v) - np.log(low)) / (np.log(high) - np.log(low)) if log else (v - low) / (high - low))
    return np.clip(np.array(out), 0, 1)


def from_unit(x) -> dict:
    out = {}
    for (name, low, high, log), u in zip(PARAMS, x):
        out[name] = float(np.exp(np.log(low) + u * (np.log(high) - np.log(low)))) if log else float(low + u * (high - low))
    return out


def propose(xs, ys, count, rng):
    """*count* new points where a Gaussian process expects the most improvement.

    Expected improvement over a few thousand random candidates; for a batch,
    each pick is added to the fit as if it scored what the process predicts
    ("kriging believer"), so the next pick goes somewhere else.
    """
    from scipy.stats import norm
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

    xs, ys = [np.asarray(x) for x in xs], list(ys)
    picks = []
    for _ in range(count):
        kernel = ConstantKernel(1.0) * Matern(length_scale=np.full(len(PARAMS), 0.3), nu=2.5) + WhiteKernel(0.05)
        gp = GaussianProcessRegressor(kernel, normalize_y=True, n_restarts_optimizer=3, random_state=0)
        gp.fit(np.array(xs), np.array(ys))
        candidates = rng.random((4000, len(PARAMS)))
        best = np.argsort(ys)[:3]
        local = np.clip(np.array(xs)[best][rng.integers(0, len(best), 1000)] + rng.normal(0, 0.08, (1000, len(PARAMS))), 0, 1)
        candidates = np.vstack([candidates, local])
        mu, sigma = gp.predict(candidates, return_std=True)
        improve = min(ys) - mu
        z = improve / np.maximum(sigma, 1e-9)
        ei = improve * norm.cdf(z) + sigma * norm.pdf(z)
        x = candidates[int(np.argmax(ei))]
        picks.append(x)
        xs.append(x)
        ys.append(float(gp.predict(x[None])[0]))
    return picks


def load_trials() -> list[dict]:
    if not TRIALS.exists():
        return []
    return [json.loads(line) for line in TRIALS.read_text(encoding="utf-8").splitlines() if line.strip()]


def split(samples, held_out_share=1 / 3, seed=0):
    """Tuning and held-out names, each category spread over both.

    Pass every annotated sample (``annotated_samples(keep_excluded=True)``): the
    split is drawn on all of them and the excluded ones are dropped afterwards,
    so excluding a sample does not move any other between the two sets.
    """
    rng = np.random.default_rng(seed)
    key = samples[["Ripped", "Folded", "Quality"]].astype(str).agg("/".join, axis=1)
    tune, held = [], []
    for _, group in samples.groupby(key):
        names = list(group["name"])
        rng.shuffle(names)
        cut = int(round(len(names) * held_out_share))
        held += names[:cut]
        tune += names[cut:]
    return sorted(set(tune) - set(EXCLUDE)), sorted(set(held) - set(EXCLUDE))


def search(trials: int = 60, batch: int = 4, start: int = 12, workers: int | None = None, seed: int = 0) -> None:
    samples = annotated_samples()
    build_cache(samples)
    names = list(samples["name"])
    tune, held = split(annotated_samples(keep_excluded=True), seed=seed)
    done = load_trials()
    rng = np.random.default_rng(seed + len(done))
    workers = workers or max(1, (os.cpu_count() or 2) - 2)
    print(f"{len(tune)} samples to tune on, {len(held)} held out; {len(done)} trials already done", flush=True)
    with ProcessPoolExecutor(workers) as pool:
        while len(done) < trials:
            if not done:
                first = [defaults()]                                   # the current settings, as a reference
                points = [to_unit(first[0])] + list(rng.random((start - 1, len(PARAMS))))
            elif len(done) < start:
                points = list(rng.random((min(batch, start - len(done)), len(PARAMS))))
            else:
                xs = [to_unit(t["params"]) for t in done]
                ys = [t["tune_score"] for t in done]
                points = propose(xs, ys, min(batch, trials - len(done)), rng)
            proposals = [from_unit(x) for x in points]
            if not done:
                proposals[0] = defaults()
            started = time.perf_counter()
            tasks = [(name, params) for params in proposals for name in names]
            results = list(pool.map(run_case, tasks, chunksize=1))
            for k, params in enumerate(proposals):
                rows = results[k * len(names): (k + 1) * len(names)]
                trial = {"trial": len(done), "params": params, "tune_score": score(rows, tune),
                         "held_score": score(rows, held), "rows": rows}
                done.append(trial)
                with TRIALS.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(trial) + "\n")
            best = min(done, key=lambda t: t["tune_score"])
            print(f"{len(done):3d} trials ({time.perf_counter() - started:.0f} s for {len(proposals)}): "
                  f"best tuning score {best['tune_score']:.1f} µm (held out {best['held_score']:.1f}) at trial {best['trial']}",
                  flush=True)


# ---------------------------------------------------------------------------
# Picking one line among several
# ---------------------------------------------------------------------------


def line_features(name: str, lines: list) -> dict:
    """What can be said about each candidate line of one sample without the hand-drawn one.

    * ``length`` — µm over the tissue.
    * ``bending`` — sum over the line of (turn / 90°)², per mm: how much it bends.
    * ``darkness`` — mean of the finder's own price (the default line cost) along it:
      low on a clear dark junction.
    * ``consensus`` — median distance (µm) to the other candidates: low when most
      settings agree on it.
    """
    from scipy import ndimage

    overview, _band, _line, base = load_case(name)
    um = overview.um
    filled = jn.tissue_of(overview)
    region = ndimage.binary_dilation(filled, iterations=max(1, int(np.ceil(base["margin_um"] / um))))
    stain = overview.nuclear_mean if overview.nuclear_mean is not None else overview.channels[overview.nuclear]
    price = jn.line_cost(stain, filled, um)
    features = {"length": [], "bending": [], "darkness": [], "consensus": []}
    for line in lines:
        points = on_tissue(dense(np.asarray(line)), region, um)
        if len(points) < 3:
            for key in features:
                features[key].append(np.nan)
            continue
        length = float(np.hypot(*np.diff(points, axis=0).T).sum())
        sparse = points[:: max(1, int(round(20.0 / 2.0)))]          # every ~20 µm
        v = np.diff(sparse, axis=0)
        heading = np.unwrap(np.arctan2(v[:, 0], v[:, 1]))
        bend = float(((np.diff(heading) / (np.pi / 2)) ** 2).sum()) / max(length / 1000.0, 1e-6)
        rc = np.clip(np.round(points / um).astype(int), 0, np.array(price.shape) - 1)
        features["length"].append(length)
        features["bending"].append(bend)
        features["darkness"].append(float(price[rc[:, 0], rc[:, 1]].mean()))
        features["consensus"].append(np.nan)
    n = len(lines)
    distances = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(i + 1, n):
            distances[i, j] = distances[j, i] = line_distance(lines[i], lines[j], region, um)[0]
    for i in range(n):
        others = np.delete(distances[i], i)
        features["consensus"][i] = float(np.nanmedian(others)) if np.isfinite(others).any() else np.nan
    return {key: np.array(values, dtype=float) for key, values in features.items()}


RULES = {
    "consensus": lambda f: f["consensus"],
    "darkest": lambda f: f["darkness"],
    "smoothest": lambda f: f["bending"],
    "shortest": lambda f: f["length"],
    "darkest + consensus": lambda f: _rank(f["darkness"]) + _rank(f["consensus"]),
}


def _rank(values):
    order = np.argsort(np.where(np.isfinite(values), values, np.inf))
    ranks = np.empty(len(values))
    ranks[order] = np.arange(len(values))
    return ranks


def pick(features: dict, rule: str) -> int:
    values = np.asarray(RULES[rule](features), dtype=float)
    return int(np.nanargmin(np.where(np.isfinite(values), values, np.nan))) if np.isfinite(values).any() else -1


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--trials", type=int, default=60)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=None)
    arguments = parser.parse_args()
    search(arguments.trials, arguments.batch, workers=arguments.workers)
