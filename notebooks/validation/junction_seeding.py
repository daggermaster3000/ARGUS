"""Seeding and other changes to the junction finder, on every section with a hand-drawn junction.

1. Finds every section under ``D:\\Slidescanner extracted`` that carries a hand-drawn
   junction region (not only the 31 of the first pick list), with its pre-screen
   classes, and caches its overview (``junction_tuning.build_cache``).
2. Split: the first 30 keep the tuning / held-out split of junction_evaluation; the
   new ones are split the same way (a third held out, stratified by Ripped / Folded /
   Quality), on their own, so the old split never moves.
3. Runs every variant in VARIANTS on every section and scores it against the
   hand-drawn line (``junction_tuning.run_case``) -> ``junction_eval/seeding.jsonl``.
4. Seed diagnostics per section with the defaults -> ``junction_eval/seeding_diagnostics.csv``.

    python notebooks/validation/junction_seeding.py
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

import junction_tuning as jt  # noqa: E402
from microscopy_viewer import fusion as fu  # noqa: E402
from microscopy_viewer import ims_store  # noqa: E402
from microscopy_viewer import junction as jn  # noqa: E402

SCANS = Path(r"D:\Slidescanner extracted")
PRESCREEN = Path(r"C:\Users\qfavey\Downloads\oasis_prescreen_211026.csv")
SAMPLES = jt.OUT / "annotated_all.csv"
RESULTS = jt.OUT / "seeding.jsonl"
DIAGNOSTICS = jt.OUT / "seeding_diagnostics.csv"

#: Overrides of the defaults, by name. Grouped: seeds, then the other settings.
VARIANTS = {
    "default": {},
    "seeds: all labelled patches": {"seed_rule": "patches"},
    "seeds: all patches, each pulled back": {"seed_rule": "patches_pulled"},
    "seeds: dominant (lead 0.15)": {"seed_rule": "dominant", "SEED_LEAD": 0.15},
    "seeds: dominant (lead 0.25)": {"seed_rule": "dominant"},
    "seeds: dominant (lead 0.35)": {"seed_rule": "dominant", "SEED_LEAD": 0.35},
    "seeds: second seed without AF555": {"seed_from_other_channel": False},
    "bend 400": {"bend_um": 400.0},
    "bend 2000": {"bend_um": 2000.0},
    "bend 3000": {"bend_um": 3000.0},
    "tears priced": {"tear_price": 1.0},
    "cells' edge 0.5": {"edge_weight": 0.5},
    "no-seam fallback": {"no_seam_price": 0.4, "no_seam_edge": 0.5},
    "tissue from the nuclear stain only": {"tissue_all_channels": False},
    "all patches + tears priced": {"seed_rule": "patches", "tear_price": 1.0},
    "all patches + bend 2000": {"seed_rule": "patches", "bend_um": 2000.0},
}
#: Module constants a variant may set (the worker sets them before the run).
CONSTANTS = {"SEED_LEAD"}


def annotated_sections() -> pd.DataFrame:
    """Every section with a hand-drawn junction, with its pre-screen classes and split."""
    rows = []
    for path in sorted(SCANS.glob("*.ome.tiff")):
        try:
            if fu.MANUAL_KEY in ims_store.list_labels(path):
                rows.append({"path": str(path), "name": path.name.replace(".ome.tiff", "")})
        except Exception:
            pass
    table = pd.DataFrame(rows)
    pre = pd.read_csv(PRESCREEN, encoding="utf-8-sig")
    pre["name"] = [Path(p).name.replace(".ome.tiff", "") for p in pre["path"]]
    table = table.merge(pre[["name", "Ripped", "Folded", "Quality", "Shape", "Pass"]], on="name", how="left")
    old_tune, old_held = jt.split(jt.annotated_samples(keep_excluded=True))
    old = set(old_tune) | set(old_held) | set(jt.EXCLUDE)
    new = table[~table["name"].isin(old)]
    new_tune, new_held = _split(new, seed=1)
    table["set"] = np.where(table["name"].isin(set(old_held) | set(new_held)), "held out", "tuning")
    table["batch"] = np.where(table["name"].isin(old), "first 31", "new")
    table = table[~table["name"].isin(jt.EXCLUDE)].reset_index(drop=True)
    table.to_csv(SAMPLES, index=False)
    return table


def _split(frame, seed, share=1 / 3):
    rng = np.random.default_rng(seed)
    key = frame[["Ripped", "Folded", "Quality"]].astype(str).agg("/".join, axis=1)
    tune, held = [], []
    for _, group in frame.groupby(key):
        names = list(group["name"])
        rng.shuffle(names)
        cut = int(round(len(names) * share))
        held += names[:cut]
        tune += names[cut:]
    return tune, held


def run_one(task):
    """One section, one variant (module constants set in this worker for the run)."""
    name, variant, params = task
    saved = {k: getattr(jn, k) for k in CONSTANTS}
    try:
        for k in CONSTANTS & set(params):
            setattr(jn, k, params[k])
        options = {k: v for k, v in params.items() if k not in CONSTANTS}
        row = jt.run_case((name, {**jt.defaults(), **options}))
    except Exception as exc:
        row = {"name": name, "found": False, "error": f"{type(exc).__name__}: {exc}"}
    finally:
        for k, v in saved.items():
            setattr(jn, k, v)
    row.pop("line", None)
    return {**row, "variant": variant, "capped": jt.capped(row)}


def diagnose(name):
    """With the defaults: what the seeds look like, and what the finder says about the section."""
    from scipy import ndimage

    ov, manual_band, manual_line, base = jt.load_case(name)
    options = jn.JunctionOptions(save=False, width_um=base["width_um"], gap_um=base["gap_um"])
    result = jn.find_junction(ov, options)
    out = {"name": name, "status": result.status, "flags": "; ".join(result.flags),
           **{k: result.metrics.get(k, np.nan) for k in ("Seam price", "Density contrast", "Labelled share of tissue",
                                                           "Solidity", "Tissue area (mm2)", "Hole share")}}
    filled = result.filled
    if result.seeds is not None and filled.any():
        density = result.density
        normal = jn._normalised(density, filled)
        pieces, _ = ndimage.label((normal > 0.5) & filled)
        sizes = np.bincount(pieces.ravel())[1:]
        out["AF488 patches"] = int((sizes >= jn.SEED_MIN_SHARE * filled.sum()).sum())
        out["AF488 patches area (share of tissue)"] = float(sizes[sizes >= jn.SEED_MIN_SHARE * filled.sum()].sum() / filled.sum())
        out["labelled seed (share of tissue)"] = float(result.seeds[0].sum() / filled.sum())
        out["other seed (share of tissue)"] = float(result.seeds[1].sum() / filled.sum())
        out["second seed rule"] = result.second_seed
        hot_at = np.argwhere(result.seeds[0]).mean(0) * ov.um
        cold_at = np.argwhere(result.seeds[1]).mean(0) * ov.um
        out["seed distance (um)"] = float(np.hypot(*(hot_at - cold_at)))
        # Which side of the hand-drawn line each seed is on: a seed on the wrong side is wrong from the start.
        if manual_band.any() and result.halves is not None:
            out["labelled seed in the hand-drawn band"] = float((result.seeds[0] & manual_band).sum() / max(result.seeds[0].sum(), 1))
            out["other seed in the hand-drawn band"] = float((result.seeds[1] & manual_band).sum() / max(result.seeds[1].sum(), 1))
    return out


def main() -> None:
    table = annotated_sections()
    print(f"{len(table)} sections with a hand-drawn junction ({(table['batch'] == 'new').sum()} new); "
          f"held out: {(table['set'] == 'held out').sum()}", flush=True)
    jt.build_cache(table)
    names = list(table["name"])
    tasks = [(n, v, p) for v, p in VARIANTS.items() for n in names]
    with ProcessPoolExecutor(12) as pool:
        rows = list(pool.map(run_one, tasks, chunksize=4))
        diags = list(pool.map(diagnose, names))
    meta = table.set_index("name")[["set", "batch", "Ripped", "Folded", "Quality", "Shape"]]
    with RESULTS.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({**r, **meta.loc[r["name"]].to_dict()}, default=str) + "\n")
    pd.DataFrame(diags).merge(table[["name", "set", "batch"]], on="name").to_csv(DIAGNOSTICS, index=False)
    print("written", RESULTS, DIAGNOSTICS)


if __name__ == "__main__":
    main()
