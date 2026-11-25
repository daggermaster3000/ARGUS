"""The object map: every AF488 mask object (``object_features.py``) placed by a UMAP of its shape, intensity and
nucleus features, for ``object_map.html`` (look at the outlines, select groups such as neurites, filter them out).

Per object it keeps the features, the section's design, and — for the objects in the blinded count squares — how many
annotators dotted them (one-to-one, a dot and an object centroid within :data:`MATCH_UM`, per annotator), so the page
can show whether a group the user excludes is one people did or did not count.

The UMAP takes :data:`UMAP_FEATURES` (skewed ones log-transformed, :data:`LOG`), each scaled by its median and
interquartile range; ``n_neighbors`` :data:`NEIGHBOURS`, ``min_dist`` :data:`MIN_DIST`, seed :data:`SEED`.

Written into the section browser folder: ``object_map.js`` (``window.OBJECT_MAP``: columns as base64 little-endian
float32 / int32, the outlines as one base64 byte string with offsets) and ``object_map.html``.

    python notebooks/official_analysis/object_map.py

No Qt.
"""
from __future__ import annotations

import base64
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import annotation_analysis as aa  # noqa: E402
import object_features as of  # noqa: E402
import section_browser as sb  # noqa: E402

# ---------------------------------------------------------------- Settings
UMAP_FEATURES = ("area", "perimeter", "circularity", "solidity", "extent", "eccentricity", "aspect", "thickness",
                 "skeleton", "width", "ends", "branches", "mean", "max", "signal", "snr", "contrast", "on_nucleus",
                 "nuclei", "nucleus_cover")
LOG = {"area", "perimeter", "aspect", "thickness", "skeleton", "width", "ends", "branches", "mean", "max", "signal",
       "snr", "contrast", "nuclei"}
#: shown on the page (colour, filters, tooltip), in this order
SHOWN = UMAP_FEATURES + ("major", "minor", "integrated", "signal_max", "away")
NEIGHBOURS, MIN_DIST, SEED = 30, 0.1, 42
MATCH_UM = 6.0                    #: an annotator's dot and an object centroid within this are the same cell
# --------------------------------------------------------------------------


def b64(a: np.ndarray, dtype: str) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype=dtype).tobytes()).decode("ascii")


def dotted(t: pd.DataFrame) -> tuple[np.ndarray, dict]:
    """Per object: -1 outside every count square, else the number of annotators who dotted it; and the dots' totals."""
    out = np.full(len(t), -1, np.int32)
    key = pd.read_csv(sb.SQUARES_KEY)
    counts = pd.concat([pd.read_csv(f) for f in sorted(sb.SQUARES_COUNTS.glob("square_counts_*.csv"))], ignore_index=True)
    shown = key.set_index("code")["shown"]
    seen = aa.seen_squares(counts, shown)
    dots = counts[counts["kind"] == "dot"]
    who = sorted(counts["annotator"].unique())
    totals = {w: 0 for w in who}
    by_sid = {sid: g for sid, g in t.groupby("section")}
    for _, k in key.iterrows():
        g = by_sid.get(sb.short(k["name"]))
        if g is None:
            continue
        x0, y0, s = k["frame_left_um"], k["frame_top_um"], k["frame_um"]
        inside = g[(g["x"] >= x0) & (g["x"] < x0 + s) & (g["y"] >= y0) & (g["y"] < y0 + s)]
        out[inside.index] = 0
        xy = inside[["x", "y"]].to_numpy(float)
        for w in who:
            if not ((seen["annotator"] == w) & (seen["code"] == k["code"])).any():
                continue
            d = dots[(dots["annotator"] == w) & (dots["code"] == k["code"])][["x_um", "y_um"]].to_numpy(float) + [x0, y0]
            totals[w] += len(d)
            if not len(d) or not len(xy):
                continue
            dist = np.hypot(d[:, None, 0] - xy[None, :, 0], d[:, None, 1] - xy[None, :, 1])
            used_d, used_o = set(), set()
            for _, i, j in sorted((dist[i, j], i, j) for i, j in zip(*np.nonzero(dist <= MATCH_UM))):
                if i not in used_d and j not in used_o:
                    used_d.add(i); used_o.add(j)
                    out[inside.index[j]] += 1
    return out, {"annotators": len(who), "dots": totals}


def main() -> int:
    import umap

    started = time.perf_counter()
    data = json.loads((sb.BROWSER / "data.json").read_text(encoding="utf-8"))
    sections = [s for s in data["sections"]]
    sec_index = {s["id"]: i for i, s in enumerate(sections)}
    parts = [pd.read_parquet(f) for f in sorted(of.OUT.glob("*.parquet"))]
    t = pd.concat(parts, ignore_index=True)
    t = t[t["section"].isin(sec_index)].reset_index(drop=True)
    print(f"{len(t)} objects from {t['section'].nunique()} sections", flush=True)
    t["contrast"] = t["contrast"].fillna(t["contrast"].median())
    X = np.column_stack([np.log1p(t[f].to_numpy(float)) if f in LOG else t[f].to_numpy(float) for f in UMAP_FEATURES])
    q1, med, q3 = np.nanpercentile(X, [25, 50, 75], axis=0)
    X = (X - med) / np.where(q3 - q1 > 0, q3 - q1, 1.0)
    X = np.nan_to_num(X)
    emb = umap.UMAP(n_neighbors=NEIGHBOURS, min_dist=MIN_DIST, random_state=SEED, low_memory=True,
                    verbose=False).fit_transform(X).astype(np.float32)
    print(f"UMAP {time.perf_counter() - started:.0f} s", flush=True)
    dots, tot = dotted(t)
    # the outlines as one byte string, with each object's start
    raw = [base64.b64decode(o) if o else b"" for o in t["outline"]]
    offsets = np.cumsum([0] + [len(r) for r in raw]).astype(np.int32)
    payload = {
        "generated": time.strftime("%Y-%m-%d %H:%M"), "n": len(t), "key": of.KEY, "outline_step_um": of.OUTLINE_STEP_UM,
        "features": list(SHOWN), "umap_features": list(UMAP_FEATURES),
        "sections": [{"id": s["id"], "genotype": s["genotype"], "experiment": str(s["experiment"]), "slide": s["slide"],
                      "length_um": s["length_um"], "crossed_8": (s["counts"].get("gauss-8") or {}).get("crossed"),
                      "n_8": (s["counts"].get("gauss-8") or {}).get("n")} for s in sections],
        "squares": tot,
        "cols": {
            "sec": b64(t["section"].map(sec_index).to_numpy(), "<i4"), "index": b64(t["index"].to_numpy(), "<i4"),
            "u": b64(emb[:, 0], "<f4"), "v": b64(emb[:, 1], "<f4"),
            "side": b64(t["side"].to_numpy(), "<i4"), "reg": b64(t["reg"].to_numpy(), "<i4"), "dotted": b64(dots, "<i4"),
            "x": b64(t["x"].to_numpy(), "<f4"), "y": b64(t["y"].to_numpy(), "<f4"),
            **{f: b64(t[f].to_numpy(float), "<f4") for f in SHOWN},
        },
        "outlines": base64.b64encode(b"".join(raw)).decode("ascii"), "outline_offsets": b64(offsets, "<i4"),
    }
    text = json.dumps(payload, separators=(",", ":"), allow_nan=False)
    (sb.BROWSER / "object_map.js").write_text("window.OBJECT_MAP = " + text + ";\n", encoding="utf-8")
    shutil.copy(HERE / "object_map.html", sb.BROWSER / "object_map.html")
    print(f"object_map.js: {len(text) / 1e6:.0f} MB, {time.perf_counter() - started:.0f} s; in squares {(dots >= 0).sum()} "
          f"objects, dotted by >= 1 annotator {(dots >= 1).sum()}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
