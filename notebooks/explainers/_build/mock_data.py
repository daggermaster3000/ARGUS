"""Mock DATA for developing explainer_template.html before the real builder runs.

Writes ``_build/mock_data.json`` (next to this file; not ``explainer_data.json``, which the real builder writes) with the schema of
the shared spec, filled with synthetic but plausible content: two fused discs
with a dark line between them, cells, a positivity table, a gallery.

    python notebooks/explainers/_build/mock_data.py [out.json]

Nothing here is a measurement. The real data comes from ``build_explainer.py``.
"""

from __future__ import annotations

import base64
import gzip
import io
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_flow
from skimage.measure import find_contours
from skimage.segmentation import expand_labels

HERE = Path(__file__).resolve().parent
RNG = np.random.default_rng(7)

PAL = dict(teal=(42, 157, 143), coral=(231, 111, 81), navy=(38, 70, 83), amber=(233, 185, 73),
           ink=(31, 36, 48), violet=(141, 108, 171), paper=(250, 247, 242), tissue=(216, 222, 226))


# ---------------------------------------------------------------- encoding
def enc(arr, dtype=None):
    a = np.ascontiguousarray(arr if dtype is None else np.asarray(arr).astype(dtype))
    if a.dtype == np.float64:
        a = a.astype(np.float32)
    if a.dtype == bool:
        a = a.astype(np.uint8)
    raw = a.astype(a.dtype.newbyteorder("<")).tobytes()
    return {"shape": list(a.shape), "dtype": str(a.dtype), "gz": base64.b64encode(gzip.compress(raw, 6)).decode()}


def dataurl(img: Image.Image, fmt="PNG", quality=85):
    buf = io.BytesIO()
    if fmt == "JPEG":
        img.convert("RGB").save(buf, "JPEG", quality=quality)
        mime = "jpeg"
    else:
        img.save(buf, "PNG", optimize=True)
        mime = "png"
    return f"data:image/{mime};base64," + base64.b64encode(buf.getvalue()).decode()


def norm01(a):
    lo, hi = np.percentile(a, 1), np.percentile(a, 99.5)
    return np.clip((a - lo) / (hi - lo + 1e-9), 0, 1).astype(np.float32)


def otsu(values, bins=256):
    hist, edges = np.histogram(values, bins)
    mids = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    m0 = np.cumsum(hist * mids) / np.maximum(w0, 1)
    m1 = (np.sum(hist * mids) - np.cumsum(hist * mids)) / np.maximum(w1, 1)
    between = w0 * w1 * (m0 - m1) ** 2
    return float(mids[np.argmax(between)])


def disk(r):
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return x * x + y * y <= r * r


def largest(mask):
    lab, n = ndi.label(mask)
    if n == 0:
        return mask
    sizes = ndi.sum(mask, lab, range(1, n + 1))
    return lab == (1 + int(np.argmax(sizes)))


def polyline_um(points_px, um):
    return [[round(float(y) * um, 2), round(float(x) * um, 2)] for y, x in points_px]


# ---------------------------------------------------------------- sample
def make_sample(h, w, um, *, c1, r1, c2, r2, line, bright=(0.8, 0.6), hole=None, tear=None, gap_um, width_um):
    """Two discs (in pixels), a dark curved line x = line(y); everything the explainer shows."""
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    d1 = np.hypot((yy - c1[0]) / r1[0], (xx - c1[1]) / r1[1])
    d2 = np.hypot((yy - c2[0]) / r2[0], (xx - c2[1]) / r2[1])
    inside = (d1 < 1) | (d2 < 1)
    jx = line(yy)
    left = xx < jx
    dist_line = (xx - jx) / max(1.0, 10.0 / um)  # in units of ~10 µm
    nuclear = np.where(left, bright[0], bright[1]) * inside
    nuclear *= 1 - 0.5 * np.exp(-dist_line ** 2 / 2)
    if hole is not None:
        nuclear[np.hypot(yy - hole[0], xx - hole[1]) < hole[2]] *= 0.05
    if tear is not None:
        ty, tx0, tx1 = tear
        band = (np.abs(yy - (ty + 0.15 * (xx - tx0))) < 1.0) & (xx > tx0) & (xx < tx1)
        nuclear[band & inside] *= 0.1
    nuclear += RNG.normal(0, 0.06, nuclear.shape) * inside
    # debris speck and background
    nuclear[h - 4:h - 2, 2:4] = 0.7
    nuclear += 0.05 + RNG.normal(0, 0.015, nuclear.shape)
    valid = np.ones((h, w), bool)
    valid[: h // 6, w - w // 9:] = False
    nuclear[~valid] = 0
    nuclear = np.clip(nuclear, 0, None)

    # labelled cells: patchy on the left, a dozen crossed, a diffuse cloud on the right
    label = np.zeros((h, w))
    n_cells = int(inside.sum() * 0.08)
    ys = RNG.integers(0, h, n_cells * 4)
    xs = RNG.integers(0, w, n_cells * 4)
    keep = inside[ys, xs] & left[ys, xs] & (RNG.random(len(ys)) < np.clip(1.2 - xs / jx[ys, xs], 0.1, 1))
    label[ys[keep][:n_cells], xs[keep][:n_cells]] += RNG.uniform(0.6, 1.0, min(n_cells, keep.sum()))
    right_pts = np.argwhere(inside & ~left & (xx - jx < 12 * 10 / um))
    pick = right_pts[RNG.choice(len(right_pts), 12, replace=False)]
    label[pick[:, 0], pick[:, 1]] += 0.8
    cloud = 0.45 * np.exp(-(((yy - c2[0] - r2[0] * 0.3) / (r2[0] * 0.3)) ** 2 + ((xx - c2[1]) / (r2[1] * 0.3)) ** 2))
    label = label + cloud * inside + RNG.normal(0, 0.03, label.shape) + 0.05
    label = np.clip(label, 0, None) * valid

    other = np.zeros((h, w))
    pts = np.argwhere(inside & ~left)
    sel = pts[RNG.random(len(pts)) < 0.12 * np.clip((pts[:, 1] - jx[pts[:, 0], pts[:, 1]]) / (r2[1]), 0.05, 1.5)]
    other[sel[:, 0], sel[:, 1]] = RNG.uniform(0.5, 1, len(sel))
    other = other + RNG.normal(0, 0.03, other.shape) + 0.05
    other = np.clip(other, 0, None) * valid

    nuclear, label, other = norm01(nuclear), norm01(label), norm01(other)

    # step 1: tissue
    smooth = ndi.gaussian_filter(nuclear, 10 / um)
    vals = smooth[valid]
    ot = otsu(vals)
    bg, fg = np.median(vals[vals <= ot]), np.median(vals[vals > ot])
    cut = float(bg + 0.2 * (fg - bg))
    counts, edges = np.histogram(vals, 48)
    thr = (smooth > cut) & valid
    opened = ndi.binary_opening(thr, structure=ndi.generate_binary_structure(2, 1))
    closed = ndi.binary_closing(opened, structure=disk(2))
    lab, n = ndi.label(closed)
    sizes = ndi.sum(closed, lab, range(1, n + 1))
    kept = np.isin(lab, 1 + np.flatnonzero(sizes >= 0.1 * sizes.max()))
    filled = ndi.binary_fill_holes(kept)
    tissue = kept

    # step 2: density
    size = int(round(45 / um)) | 1
    med = ndi.median_filter(label, size)
    prom = np.clip(label - med, 0, None)
    prom = np.clip(prom / (np.percentile(prom[tissue], 99.5) + 1e-9), 0, 1)

    def nblur(x, s):
        amount = ndi.gaussian_filter(np.where(tissue, x, 0), s)
        weight = ndi.gaussian_filter(tissue.astype(float), s)
        return np.where(filled, amount / np.maximum(weight, 1e-3), 0)

    density = nblur(prom, 100 / um)
    density_off = nblur(np.clip(label / np.percentile(label[tissue], 99.5), 0, 1), 100 / um)
    omed = ndi.median_filter(other, size)
    oprom = np.clip(other - omed, 0, None)
    oprom = np.clip(oprom / (np.percentile(oprom[tissue], 99.5) + 1e-9), 0, 1)
    odens = nblur(oprom, 100 / um)

    def nrm(d):
        lo, hi = np.percentile(d[filled], 10), np.percentile(d[filled], 98)
        return np.clip((d - lo) / (hi - lo + 1e-9), 0, 1) * filled

    nd, no = nrm(density), nrm(odens)
    core = (nd > 0.5)
    core_l = largest(core)
    ocore = largest((no > 0.5) & ~core_l & (no > nd))
    # pull back: keep the part further from the other's centre than half
    def pull(a, b):
        cy, cx = np.argwhere(b).mean(0)
        d = np.hypot(yy - cy, xx - cx)
        lo, hi = d[a].min(), d[a].max()
        return largest(a & (d > lo + 0.5 * (hi - lo)))

    hot, cold = pull(core_l, ocore), pull(ocore, core_l)
    seed_steps = [
        {"name": "labelled density, normalised: above 0.5", "mask": enc(core, np.uint8)},
        {"name": "labelled core: the largest piece", "mask": enc(core_l, np.uint8)},
        {"name": "other organoid's core (its own label)", "mask": enc(ocore, np.uint8)},
        {"name": "both pulled back from each other", "mask": enc(hot | cold, np.uint8)},
    ]

    # step 4: price
    stain = np.where(filled, nuclear, np.median(nuclear[filled]))
    fine = ndi.gaussian_filter(stain, 8 / um)
    coarse = ndi.gaussian_filter(stain, 60 / um)
    ratio = np.clip(fine / np.maximum(coarse, 1e-6), 0.3, 1)
    cost = ratio ** 6
    cost_powers = {str(p): enc(ratio ** p, np.float32) for p in (2, 6, 10)}

    # floods (Dijkstra, one step costs 1 / price: slow on the dark line)
    def flood(seed):
        import heapq
        t = np.full((h, w), np.inf)
        heap = []
        for y, x in np.argwhere(seed):
            t[y, x] = 0
            heap.append((0.0, int(y), int(x)))
        heapq.heapify(heap)
        while heap:
            d, y, x = heapq.heappop(heap)
            if d > t[y, x]:
                continue
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and filled[ny, nx]:
                    step = (1.4142 if dy and dx else 1.0) * 2 / (cost[y, x] + cost[ny, nx] + 0.02)
                    if d + step < t[ny, nx]:
                        t[ny, nx] = d + step
                        heapq.heappush(heap, (d + step, ny, nx))
        return np.where(np.isfinite(t), t, -1).astype(np.float32)

    arrival_hot, arrival_cold = flood(hot), flood(cold)

    # step 5: min cut (8-neighbour, as the package does)
    idx = -np.ones((h, w), int)
    pix = np.argwhere(filled)
    idx[filled] = np.arange(len(pix))
    S, T = len(pix), len(pix) + 1
    rows, cols, caps = [], [], []
    for dy, dx, f in ((0, 1, 1.0), (1, 0, 1.0), (1, 1, 0.7071), (1, -1, 0.7071)):
        a = idx[0:h - dy, max(0, -dx):w - max(0, dx)]
        b = idx[dy:h, max(0, dx):w + min(0, dx)]
        ca = cost[0:h - dy, max(0, -dx):w - max(0, dx)]
        cb = cost[dy:h, max(0, dx):w + min(0, dx)]
        ok = (a >= 0) & (b >= 0)
        c = np.maximum(1, (0.5 * (ca + cb) * 1000 * f).astype(int))[ok]
        rows += [a[ok], b[ok]]
        cols += [b[ok], a[ok]]
        caps += [c, c]
    for seedmask, node in ((hot, S), (cold, T)):
        ids = idx[seedmask & filled]
        big = np.full(len(ids), 2 ** 30)
        if node == S:
            rows += [np.full(len(ids), S)]; cols += [ids]
        else:
            rows += [ids]; cols += [np.full(len(ids), T)]
        caps += [big]
    n_nodes = len(pix) + 2
    G = csr_matrix((np.concatenate(caps).astype(np.int32), (np.concatenate(rows), np.concatenate(cols))),
                   shape=(n_nodes, n_nodes))
    G.sum_duplicates()
    flow = maximum_flow(G, S, T).flow
    resid = (G - flow).tocsr()
    resid.data[resid.data < 0] = 0
    resid.eliminate_zeros()
    seen = np.zeros(n_nodes, bool)
    stack = [S]
    seen[S] = True
    while stack:
        u = stack.pop()
        for v in resid.indices[resid.indptr[u]:resid.indptr[u + 1]]:
            if not seen[v]:
                seen[v] = True
                stack.append(v)
    side1 = np.zeros((h, w), bool)
    side1[pix[:, 0], pix[:, 1]] = seen[:len(pix)]
    halves0 = np.where(filled, np.where(largest(side1 & filled), 1, 2), 0).astype(np.uint8)

    def boundary_line(halves):
        b = (halves == 1) & (np.roll(halves, -1, 1) == 2) | (halves == 1) & (np.roll(halves, 1, 1) == 2)
        b |= (halves == 1) & (np.roll(halves, -1, 0) == 2) | (halves == 1) & (np.roll(halves, 1, 0) == 2)
        ys_, xs_ = np.nonzero(b)
        rows_ = np.unique(ys_)
        pts = np.array([[r, xs_[ys_ == r].mean()] for r in rows_])
        return pts

    pts0 = boundary_line(halves0)
    cuts, lines = {"0": enc(halves0)}, {}
    k = max(3, int(round(150 / um)))
    sm = np.column_stack([pts0[:, 0], np.convolve(np.pad(pts0[:, 1], k // 2, mode="edge"), np.ones(k) / k, "valid")[: len(pts0)]])
    step = max(1, int(round(40 / um)))
    lines["0"] = polyline_um(np.vstack([sm[::step], sm[-1:]]), um)
    for bend, deg in (("400", 4), ("1000", 2)):
        coef = np.polyfit(pts0[:, 0], pts0[:, 1], deg)
        fx = np.polyval(coef, yy)
        halves = np.where(filled, np.where(xx < fx, 1, 2), 0).astype(np.uint8)
        cuts[bend] = enc(halves)
        ys_line = np.arange(pts0[0, 0], pts0[-1, 0] + 1, step)
        lines[bend] = polyline_um(np.column_stack([ys_line, np.polyval(coef, ys_line)]), um)
    rows_t = np.nonzero(filled.any(1))[0]
    rows_t = [r for r in rows_t if filled[r, int(np.clip(line(np.array([r]))[0], 0, w - 1))]]
    truth = polyline_um([[r, float(line(np.array([float(r)]))[0])] for r in rows_t[::2]], um)

    # step 7: band from the 1000 line
    lp = np.array(lines["1000"]) / um
    dense = np.vstack([np.linspace(lp[i], lp[i + 1], 20) for i in range(len(lp) - 1)])
    from scipy.spatial import cKDTree
    dist, _ = cKDTree(dense).query(np.column_stack([yy.ravel(), xx.ravel()]))
    dist = dist.reshape(h, w) * um
    side2 = np.asarray(cuts_halves := np.where(xx < np.polyval(np.polyfit(pts0[:, 0], pts0[:, 1], 2), yy), 1, 2)) == 2
    rim = ndi.binary_dilation(filled, disk(max(1, int(round(10 / um))))) & ~filled
    band = side2 & (filled | rim) & (dist >= gap_um) & (dist <= gap_um + width_um)
    band = largest(band)

    return dict(
        um=um, shape=[h, w], nuclear=nuclear, label=label, other=other, smooth=smooth,
        hist={"edges": [round(float(e), 5) for e in edges], "counts": [int(c) for c in counts], "otsu": ot, "cut": cut},
        tissue_steps=[{"name": n, "mask": enc(m, np.uint8)} for n, m in (
            ("above the threshold", thr), ("opened: specks removed", opened), ("closed: small gaps filled", closed),
            ("big pieces kept", kept), ("holes filled", filled))],
        sigmas={str(s): enc(ndi.gaussian_filter(nuclear, s / um), np.float32) for s in (5, 10, 20)},
        median_local=med, prominence=prom, prominence_off=np.clip(label / np.percentile(label[tissue], 99.5), 0, 1),
        density=density, density_off=density_off,
        seed_steps=seed_steps, hot=hot, cold=cold, cost=cost, cost_powers=cost_powers,
        arrival_hot=arrival_hot, arrival_cold=arrival_cold, cuts=cuts, lines=lines, truth=truth,
        band=band, tissue=filled, rim_um=10, gap_um=gap_um, width_um=width_um,
        _filled=filled, _dist=dist, _side2=side2, _halves=cuts_halves, _line=line,
    )


def pack_sample(s):
    out = {}
    for k, v in s.items():
        if k.startswith("_"):
            continue
        if isinstance(v, np.ndarray):
            out[k] = enc(v.astype(np.uint8) if v.dtype == bool else v.astype(np.float32) if v.dtype.kind == "f" else v)
        else:
            out[k] = v
    return out


def composite(s):
    n, l, o = s["nuclear"], s["label"], s["other"]
    rgb = np.zeros(n.shape + (3,))
    for ch, col, g in ((n, (0.35, 0.55, 0.95), 0.9), (l, PAL["teal"], 1.3), (o, PAL["coral"], 1.0)):
        c = np.array(col, float)
        c = c / 255 if c.max() > 1 else c
        rgb += np.clip(ch * g, 0, 1)[..., None] * c[None, None]
    img = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))
    return img.resize((n.shape[1] * 3, n.shape[0] * 3), Image.BILINEAR)


# ---------------------------------------------------------------- cells crop
def make_cells():
    H = W = 240
    um = 0.5
    lab = np.zeros((H, W), np.int32)
    centres = []
    tries = 0
    while len(centres) < 70 and tries < 5000:
        tries += 1
        c = RNG.uniform(10, H - 10, 2)
        if all(np.hypot(*(c - d)) > 17 for d in centres):
            centres.append(c)
    yy, xx = np.mgrid[0:H, 0:W]
    for i, (cy, cx) in enumerate(centres, 1):
        ry, rx, a = RNG.uniform(6, 9), RNG.uniform(6, 9), RNG.uniform(0, np.pi)
        dy, dx = yy - cy, xx - cx
        u = dx * np.cos(a) + dy * np.sin(a)
        v = -dx * np.sin(a) + dy * np.cos(a)
        lab[((u / rx) ** 2 + (v / ry) ** 2 < 1) & (lab == 0)] = i
    grown = expand_labels(lab, 6)
    ring = expand_labels(lab, 12)
    ring[grown > 0] = 0
    dapi = ndi.gaussian_filter((lab > 0) * RNG.uniform(0.6, 1, (H, W)), 1.2) + RNG.normal(0, 0.03, (H, W)) + 0.05
    pos = {i: RNG.random() < 0.15 for i in range(1, len(centres) + 1)}
    haze = 0.25 * np.exp(-((yy - 60) ** 2 + (xx - 190) ** 2) / (2 * 40 ** 2))
    af = haze + RNG.normal(0, 0.02, (H, W)) + 0.08
    for i, p in pos.items():
        if p:
            af += 0.7 * ndi.gaussian_filter((grown == i).astype(float), 2)
    af = np.clip(af, 0, 1)
    dapi = np.clip(dapi, 0, 1)

    def tint(a, col):
        c = np.array(col) / 255
        return Image.fromarray((np.clip(a, 0, 1)[..., None] * c * 255).astype(np.uint8))

    comp = np.clip(dapi[..., None] * np.array([0.35, 0.55, 0.95]) * 0.8 + af[..., None] * np.array(PAL["teal"]) / 255 * 1.4, 0, 1)
    table = []
    for i, (cy, cx) in enumerate(centres, 1):
        m = grown == i
        r = ring == i
        mean = float(af[m].mean()) * 1000
        rmean = float(af[r].mean()) * 1000 if r.any() else float("nan")
        table.append(dict(id=i, y=round(float(cy), 1), x=round(float(cx), 1), area_um2=round(float((lab == i).sum() * um * um), 1),
                          mean=round(mean, 1), ring=None if np.isnan(rmean) else round(rmean, 1),
                          contrast=None if np.isnan(rmean) else round(mean / rmean, 2),
                          filtered_mean=round(mean - float(haze[m].mean()) * 1000 - 80, 1),
                          in_band=bool(cx > 60), from_line_um=round(float(cx - 60) * um + 40, 1), edge_um=round(float(cy) * um + 15, 1),
                          clicked=bool(pos[i] and RNG.random() < 0.7)))
    return dict(name="mock_sample_s3 (crop)", um_per_px=um, size_px=[H, W], origin_um=[1200.0, 2400.0],
                dapi=dataurl(tint(dapi, (200, 215, 255))), af488=dataurl(tint(af, PAL["teal"] if False else (120, 255, 230))),
                composite=dataurl(Image.fromarray((comp * 255).astype(np.uint8))),
                nuclei=enc(lab), grown=enc(grown.astype(np.int32)), ring=enc(ring.astype(np.int32)), table=table)


# ---------------------------------------------------------------- positivity
def make_positivity(real):
    um = real["um"]
    filled, dist, side2 = real["_filled"], real["_dist"], real["_side2"]
    h, w = filled.shape
    edge = ndi.distance_transform_edt(filled) * um - ndi.distance_transform_edt(~filled) * um
    signed = np.where(side2, dist, -dist)
    zone = (filled | (edge > -10)) & (dist <= 600)
    pts = np.argwhere(zone)
    pick = pts[RNG.choice(len(pts), 6000)]
    y = (pick[:, 0] + RNG.random(len(pick))) * um
    x = (pick[:, 1] + RNG.random(len(pick))) * um
    fl = signed[pick[:, 0], pick[:, 1]]
    ed = edge[pick[:, 0], pick[:, 1]]
    home = fl < 0
    p_pos = np.where(home, 0.06, 0.03 * np.exp(-np.maximum(fl, 0) / 120)) * (1 + 1.5 * (ed < 30))
    true = RNG.random(len(pick)) < p_pos
    haze = 120 + 60 * np.exp(-((y - y.mean()) ** 2 + (x - x.max()) ** 2) / (2 * 300 ** 2)) + 30 * home
    base = RNG.lognormal(np.log(1), 0.25, len(pick)) * haze
    raw = base + true * RNG.uniform(150, 600, len(pick))
    ring = haze * RNG.lognormal(0, 0.12, len(pick))
    filtered = raw - haze * RNG.lognormal(0, 0.05, len(pick))
    contrast = raw / ring
    in_band = real["band"][pick[:, 0], pick[:, 1]]
    clicked = (true & (RNG.random(len(pick)) < 0.6) & (np.abs(fl) < 350)) | (RNG.random(len(pick)) < 0.002)
    cols = dict(y_um=y, x_um=x, raw_mean=raw, filtered_mean=filtered, ring=ring, contrast=contrast,
                in_band=in_band, from_line_um=fl, edge_um=ed, clicked=clicked)

    def outline(mask):
        cs = find_contours(np.pad(mask.astype(float), 1), 0.5)
        c = max(cs, key=len) - 1
        return [[round(float(a) * um, 1), round(float(b) * um, 1)] for a, b in c[::2]]

    def ksd(v, k):
        b = v[in_band]
        m = np.median(b)
        return float(m + k * 1.4826 * np.median(np.abs(b - m)))

    return dict(
        name=real["name"], line=real["lines"]["1000"], band_outline=outline(real["band"]), tissue_outline=outline(filled),
        cols={k: enc(v, np.uint8 if v.dtype == bool else np.float32) for k, v in cols.items()},
        rules={"median + 3 SD (current)": ksd(raw, 3), "median + 7 SD, median filter": ksd(filtered, 7),
               "ring contrast ≥ 1.5": 1.5, "negative control 99.9th pct": float(np.percentile(raw[fl > 300], 99.9))},
    )


def make_profile():
    d = np.arange(-300, 801, 25).astype(float)
    out = {}
    for rule, home, away, decay in (("median + 3 SD (current)", 3.5, 1.4, 900), ("median + 7 SD, median filter", 2.0, 0.5, 130),
                                    ("ring contrast ≥ 1.5", 2.6, 0.7, 180), ("negative control 99.9th pct", 2.2, 0.8, 300)):
        med = np.where(d < 0, home, away * np.exp(-d / decay) + 0.05)
        per = [list(np.round(np.clip(med * RNG.lognormal(0, 0.35) + RNG.normal(0, 0.08, len(d)), 0, None), 3)) for _ in range(10)]
        out[rule] = {"median": list(np.round(med, 3)), "per_sample": per}
    return {"distance_um": list(d), "rules": out}


# ---------------------------------------------------------------- gallery
def gallery_img(kind, seed, auto_off=0.0, manual=True, auto=True, second=False):
    r = np.random.default_rng(seed)
    W, H = 360, 260
    img = Image.new("RGB", (W, H), (14, 18, 24))
    d = ImageDraw.Draw(img, "RGBA")
    c1 = (130 + r.integers(-15, 15), 130)
    c2 = (230 + r.integers(-15, 15), 130 + r.integers(-20, 20))
    if kind != "empty":
        rad = 80 if kind != "separated" else 60
        if kind == "separated":
            c1, c2 = (100, 130), (265, 130)
        d.ellipse([c1[0] - rad, c1[1] - rad, c1[0] + rad, c1[1] + rad], fill=(40, 70, 95))
        d.ellipse([c2[0] - rad, c2[1] - rad, c2[0] + rad, c2[1] + rad], fill=(36, 62, 84))
        for _ in range(250):
            x, y = r.normal(c1[0] - 20, 45), r.normal(c1[1], 45)
            d.ellipse([x - 1.5, y - 1.5, x + 1.5, y + 1.5], fill=PAL["teal"] + (220,))
        for _ in range(250):
            x, y = r.normal(c2[0] + 20, 45), r.normal(c2[1], 45)
            d.ellipse([x - 1.5, y - 1.5, x + 1.5, y + 1.5], fill=PAL["coral"] + (200,))
    if kind == "ripped":
        d.line([(120, 40), (160, 120), (150, 220)], fill=(14, 18, 24), width=9)
    if kind == "folded":
        d.polygon([(90, 90), (200, 70), (210, 110), (100, 130)], fill=(90, 130, 160, 200))
    if kind == "debris":
        for _ in range(30):
            x, y = r.uniform(20, 340), r.uniform(20, 240)
            d.ellipse([x - 4, y - 3, x + 4, y + 3], fill=(60, 80, 100))
    if kind in ("separated", "empty", "debris"):
        return dataurl(img, "JPEG")
    jx = (c1[0] + c2[0]) / 2
    if manual:
        pts = [(jx + 6 * np.sin(t / 30), t) for t in range(40, 225, 6)]
        for i in range(0, len(pts) - 1, 2):
            d.line([pts[i], pts[i + 1]], fill=(250, 247, 242), width=3)
    if second:
        d.line([(jx + 10 + 6 * np.sin(t / 30), t) for t in range(40, 225, 6)], fill=PAL["violet"], width=2)
    if auto:
        d.line([(jx + auto_off * np.sin(t / 60) + auto_off * 0.4, t) for t in range(50, 215, 5)], fill=PAL["amber"], width=3)
    return dataurl(img, "JPEG")


def make_gallery():
    items = []
    orgs = [
        ("Pass", dict(Pass="yes", Shape="Fused pair", Ripped="no", Folded="no", Quality="good"), "fused", "Two organoids, one neck, a clean section: the kind the pipeline is built for."),
        ("Damaged", dict(Pass="yes", Shape="Damaged", Ripped="no", Folded="no", Quality="good"), "ripped", "Fused, but the outline is broken: kept, but its junction is drawn by hand."),
        ("Separated", dict(Pass="no", Shape="Separated", Ripped="no", Folded="no", Quality="good"), "separated", "Two organoids that never touched: no junction to find."),
        ("Empty / debris", dict(Pass="no", Shape="Empty/debris", Ripped="no", Folded="no", Quality="poor"), "debris", "Specks and fragments, no organoid."),
        ("Ripped", dict(Pass="no", Shape="Fused pair", Ripped="yes", Folded="no", Quality="good"), "ripped", "A tear runs through the tissue."),
        ("Folded", dict(Pass="no", Shape="Fused pair", Ripped="no", Folded="yes", Quality="good"), "folded", "A flap of the section lies on itself: doubled brightness."),
        ("Out of focus", dict(Pass="no", Shape="Fused pair", Ripped="no", Folded="no", Quality="poor"), "fused", "Blurred: single nuclei cannot be told apart."),
        ("Pass (second)", dict(Pass="yes", Shape="Fused pair", Ripped="no", Folded="no", Quality="good"), "fused", "Another good pair; the labelled organoid is the dimmer one."),
    ]
    for i, (title, cls, kind, why) in enumerate(orgs):
        items.append(dict(id=f"org{i}", category="organoid", title=title, classes=cls, status=None, flags=[], error_um=None,
                          why=why, img=gallery_img(kind, i, manual=False, auto=False), legend="teal: AF488 · coral: AF555 · blue: nuclei"))
    fails = [
        ("Tear taken as junction", "ripped", 180, "fail", ["large holes or torn"], 210.0, "A tear is empty, the darkest and so the cheapest thing to cut along; the cut follows it instead of the junction."),
        ("Damaged shape", "ripped", 120, "check", ["ragged outline"], 140.0, "Without a clean neck and a continuous dark line, the cut is a guess."),
        ("Cut off by the scan edge", "fused", 60, "check", ["cut off by the scan edge"], 95.0, "Part of the organoid is outside the scan, so the seeds and the cut see only part of it."),
        ("Not fused / no junction", "separated", 0, "fail", ["organoids not fused"], None, "Two separate pieces: there is nothing to cut between."),
        ("Wrong 'ok'", "fused", 150, "ok", [], 345.0, "Nothing was flagged, yet the line runs 345 µm from where both annotators drew it. Only a look at the QC picture catches this."),
        ("Annotators disagree", "ripped", 30, "check", ["labelled cells on both sides (junction uncertain)"], 128.0, "Two people drew the junction 128 µm apart on this damaged section: there is no single right answer to match."),
    ]
    for i, (title, kind, off, status, flags, err, why) in enumerate(fails):
        items.append(dict(id=f"fail{i}", category="failure", title=title,
                          classes=dict(Pass="yes", Shape="Damaged" if kind == "ripped" else "Fused pair", Ripped="no", Folded="no", Quality="good"),
                          status=status, flags=flags, error_um=err, why=why,
                          img=gallery_img(kind, 20 + i, auto_off=off, second=(i == 5)),
                          legend="white dashed: hand-drawn · amber: automatic" + (" · violet: second annotator" if i == 5 else "")))
    for i in range(2):
        items.append(dict(id=f"good{i}", category="good_junction", title=f"Good junction {i + 1}",
                          classes=dict(Pass="yes", Shape="Fused pair", Ripped="no", Folded="no", Quality="good"),
                          status="ok", flags=[], error_um=[14.0, 22.0][i], why="The automatic line sits on the hand-drawn one.",
                          img=gallery_img("fused", 40 + i, auto_off=4), legend="white dashed: hand-drawn · amber: automatic"))
    return items


NUMBERS = {
    "triage": {"samples": 30, "failed_gt_100um": 8, "sent_by_check_status": 21, "check_but_fine": 14,
               "sent_by_rule": 7, "rule_true_failures": 7, "missed": 1, "missed_sample": "034-033_s17",
               "rule": "status is fail, or Shape is Damaged"},
    "positivity": {"cells": 920000, "samples": 30, "clicked_samples": 12,
                   "best": {"rule": "median + 7 SD, median filter", "false_pos_pct": 0.03, "step_at_line": 2.4, "migrated_per_mm": 11, "f1": 0.38, "spearman": 0.73},
                   "current": {"rule": "median + 3 SD (current)", "false_pos_pct": 0.54, "step_at_line": 2.0, "migrated_per_mm": 80, "f1": 0.13, "spearman": 0.03}},
    "results": {"band_per_mm_median": {"current": 85, "calibrated": 12, "ring_contrast": 53},
                "genotype_wt3_over_gd1": {"calibrated_samples": "4.0x (p 0.001)", "calibrated_slides": "2.7x (p 0.020)", "current_samples": "1.4x (p 0.55)"},
                "crossed_per_mm_within_300um": 11},
    "annotators": {"samples_both": 27, "line_distance_median_um": 15, "dice": 0.95, "af488_loa_pp": 0.85},
}


def main(out=None):
    out = Path(out) if out else HERE / "mock_data.json"
    syn = make_sample(48, 72, 10.0, c1=(24, 24), r1=(21, 21), c2=(24, 50), r2=(20, 20),
                      line=lambda y: 37 + 0.25 * (y - 24) + 0.01 * (y - 24) ** 2, hole=(18, 15, 2.5), gap_um=10, width_um=100)
    real = make_sample(150, 230, 5.0, c1=(72, 70), r1=(62, 66), c2=(80, 160), r2=(66, 62),
                       line=lambda y: 112 + 0.18 * (y - 75) - 0.002 * (y - 75) ** 2, bright=(0.65, 0.85), tear=(40, 30, 80),
                       hole=(100, 150, 4), gap_um=2, width_um=200)
    real["name"] = "mock_041-040_s7"
    psyn = pack_sample(syn)
    preal = pack_sample(real)
    preal["name"] = real["name"]
    preal["composite"] = dataurl(composite(real), "JPEG")
    jt = real["_line"]
    rows = [r for r in range(8, 145, 6)]
    preal["manual_line"] = polyline_um([[r, jt(np.array([float(r)]))[0] + RNG.normal(0, 1)] for r in rows if real["_filled"][r].any()], 5.0)
    preal["second_line"] = polyline_um([[r, jt(np.array([float(r)]))[0] + 3 + RNG.normal(0, 1)] for r in rows if real["_filled"][r].any()], 5.0)
    preal["band"] = enc(real["band"], np.uint8)
    real["lines"] = preal["lines"]
    positivity = make_positivity(real)
    toy = np.ones((7, 11), np.float32)
    slant = toy.copy()
    for r in range(7):
        slant[r, 3 + (r * 4) // 6] = 0.1
    speck = toy.copy()
    speck[3, 5] = 0.1
    gappy = slant.copy()
    gappy[3, 3 + 2] = 1
    gappy[4, 3 + 2] = 1
    gappy[3, 5] = 1
    data = {
        "meta": {"built": datetime.now().isoformat(timespec="seconds"), "source": "MOCK DATA (notebooks/explainers/_build/mock_data.py) — not a measurement"},
        "synthetic": psyn, "real": preal,
        "toy": {"presets": {"uniform": enc(toy), "slanted": enc(slant), "speck": enc(speck), "gappy": enc(gappy)}},
        "cells": make_cells(), "positivity": positivity, "profile": make_profile(), "gallery": make_gallery(), "numbers": NUMBERS,
    }
    text = json.dumps(data, allow_nan=False, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out} ({len(text) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
