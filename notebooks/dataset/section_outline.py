"""Which AF488⁺ objects lie in the section: the run's outline of the section and the rule that keeps an object.

The counting masks are cut inside a coarse tissue mask (``classical_vs_cellpose.tissue_mask``: the nuclear stain's
Otsu at 1/8, holes filled, widened ~5 µm), which also takes in debris, bubble rims, pieces of a neighbouring
organoid and bright strips at the edge of the scan. An object counts only when its centroid is

- inside the run's own outline of the section (``junction.tissue_of`` on the overview, the outline the junction finder
  and the cells' ``Distance to the tissue edge (µm)`` use: debris under 10 % of the largest piece dropped, holes
  filled), or at most :data:`TOLERANCE_UM` outside it (the outline is good to about one overview pixel, ~5 µm, and a
  section cut by the edge of the scan has its outline a few µm short), and
- in a piece of that outline the junction line passes through (within :data:`LINE_REACH_UM`): the assembloid, not a
  second organoid or a strip lying elsewhere in the scan. Without a line, the largest piece.

On the 614 sections of the run this drops 323 of 113,419 objects at 8 SD (0.3 %); between 10 and 20 µm outside the
outline there was a single object, so the tolerance does not decide much.

The outline is cached per section as ``af488_mask/<section>/outline.npz`` (``filled`` packed, ``shape``, ``step``,
``um``). No Qt.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

TOLERANCE_UM = 10.0               #: an object's centroid may lie this far outside the outline
LINE_REACH_UM = 50.0              #: a piece of the outline counts when the junction line passes within this
OUTLINE_FILE = "outline.npz"


def build_outline(path: Path) -> dict:
    """The run's outline of the section in *path* (``junction.tissue_of`` on the default overview)."""
    from microscopy_viewer import junction as jn
    from microscopy_viewer import loaders

    specs = loaders.load_path(path)
    try:
        overview = jn.build_overview(specs, jn.JunctionOptions(save=False))
    finally:
        loaders.release(path)
    filled = jn.tissue_of(overview)
    return {"filled": filled, "step": int(overview.step), "um": float(overview.um)}


def save_outline(folder: Path, outline: dict) -> None:
    np.savez_compressed(folder / OUTLINE_FILE, filled=np.packbits(outline["filled"]),
                        shape=np.asarray(outline["filled"].shape), step=outline["step"], um=outline["um"])


def load_outline(folder: Path) -> dict:
    z = np.load(folder / OUTLINE_FILE)
    shape = tuple(int(v) for v in z["shape"])
    filled = np.unpackbits(z["filled"])[: shape[0] * shape[1]].reshape(shape).astype(bool)
    return {"filled": filled, "step": int(z["step"]), "um": float(z["um"])}


def outline_for(folder: Path, path: Path) -> dict:
    """The cached outline of *folder*, made (and cached) from the scan *path* when missing."""
    if (folder / OUTLINE_FILE).exists():
        return load_outline(folder)
    outline = build_outline(path)
    folder.mkdir(parents=True, exist_ok=True)
    save_outline(folder, outline)
    return outline


def line_um(info: dict) -> np.ndarray | None:
    """The junction line of a run's ``cells/<section>.json`` (µm, y x), or None."""
    line = info.get("line")
    if not line:
        return None
    return np.asarray(json.loads(line) if isinstance(line, str) else line, dtype=float)


def edge_distance(outline: dict, yx_um: np.ndarray, voxel: float) -> np.ndarray:
    """Signed distance (µm) of each point to the outline, + inside (as the cells' tissue-edge column)."""
    from microscopy_viewer import junction as jn

    yx_um = np.asarray(yx_um, dtype=float).reshape(-1, 2)
    if not len(yx_um):
        return np.zeros(0, np.float32)
    grid = SimpleNamespace(um=outline["um"], step=outline["step"])
    return jn.edge_distance(grid, yx_um[:, 0] / voxel, yx_um[:, 1] / voxel, outline["filled"])


def assembloid_pieces(outline: dict, line: np.ndarray | None, voxel: float) -> tuple[np.ndarray, np.ndarray]:
    """The outline's pieces (labels) and which of them are the assembloid's (the line passes within reach)."""
    from scipy import ndimage

    filled, step, um = outline["filled"], outline["step"], outline["um"]
    pieces, n = ndimage.label(filled)
    keep = np.zeros(n + 1, bool)
    if n == 0:
        return pieces, keep
    if line is None or len(line) < 2:
        keep[int(np.argmax(np.bincount(pieces.ravel())[1:])) + 1] = True
        return pieces, keep
    points = [np.linspace(a, b, max(2, int(np.hypot(*(b - a)) / 2.0))) for a, b in zip(line[:-1], line[1:])]
    p = np.concatenate(points) / voxel / step
    hit = np.zeros(filled.shape, bool)
    hit[np.clip(np.round(p[:, 0]).astype(int), 0, filled.shape[0] - 1),
        np.clip(np.round(p[:, 1]).astype(int), 0, filled.shape[1] - 1)] = True
    hit = ndimage.binary_dilation(hit, iterations=max(1, int(round(LINE_REACH_UM / um))))
    keep[np.unique(pieces[hit & (pieces > 0)])] = True
    keep[0] = False
    if not keep.any():                                  # the line misses the outline: the largest piece
        keep[int(np.argmax(np.bincount(pieces.ravel())[1:])) + 1] = True
    return pieces, keep


def in_section(outline: dict, line: np.ndarray | None, yx_um: np.ndarray, voxel: float) -> np.ndarray:
    """Whether each centroid (µm, y x) counts: within :data:`TOLERANCE_UM` of the outline, in the assembloid's piece."""
    from scipy import ndimage

    yx_um = np.asarray(yx_um, dtype=float).reshape(-1, 2)
    if not len(yx_um):
        return np.zeros(0, bool)
    filled, step = outline["filled"], outline["step"]
    near = edge_distance(outline, yx_um, voxel) >= -TOLERANCE_UM
    pieces, keep = assembloid_pieces(outline, line, voxel)
    # each point's piece: that of the nearest outline pixel (a point just outside belongs to the piece beside it)
    _, (ri, ci) = ndimage.distance_transform_edt(~filled, return_indices=True)
    r = np.clip(np.round(yx_um[:, 0] / voxel / step - (step - 1) / (2 * step)).astype(int), 0, filled.shape[0] - 1)
    c = np.clip(np.round(yx_um[:, 1] / voxel / step - (step - 1) / (2 * step)).astype(int), 0, filled.shape[1] - 1)
    piece = pieces[ri[r, c], ci[r, c]]
    return near & keep[piece]


def drop_outside(mask: np.ndarray, outline: dict, line: np.ndarray | None, voxel: float) -> tuple[np.ndarray, int]:
    """*mask* without its objects (8-connected) that do not count; and how many were dropped."""
    from scipy import ndimage
    from skimage.measure import label

    labelled = label(mask, connectivity=2)
    n = int(labelled.max())
    if n == 0:
        return mask, 0
    ids = np.arange(1, n + 1)
    centroids = np.asarray(ndimage.center_of_mass(mask, labelled, ids), dtype=float).reshape(-1, 2) * voxel
    ok = in_section(outline, line, centroids, voxel)
    if ok.all():
        return mask, 0
    keep = np.concatenate([[False], ok])
    return keep[labelled], int((~ok).sum())
