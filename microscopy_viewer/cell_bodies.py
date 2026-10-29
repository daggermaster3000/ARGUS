"""Positive cells that are one labelled body, counted once.

The positivity rule calls nuclei: a bright labelled cell lying over two or
three nuclei — or one cell whose nucleus Cellpose cut in two — lights up the
few micrometres round each, and is counted two or three times. Here those are
merged (``notebooks/validation/05_watershed_cells.ipynb``, 29 Oct 2026: on 39
slide scans the count roughly halves, and against Quillan's clicks F1 rises
from 0.34 to 0.43 and the per-sample agreement CCC from 0.33 to 0.69):

1. the **stain foreground**: the channel less its local median, smoothed by
   :data:`SMOOTH_UM`, at least :data:`FG_K` robust SDs above the tissue's centre
   (:func:`stain_foreground`);
2. each cell's **territory**: its nucleus grown over that foreground by a
   watershed seeded by every nucleus, on the foreground's distance transform —
   touching cells split at the narrowing between them — never more than
   :data:`REACH_UM` from the nucleus;
3. the **bodies**: the classic distance-transform watershed on the stain alone.
   In each piece of foreground, the maxima of the depth inside the stain at
   least :data:`H_UM` higher than the dip between them are the bodies, and the
   piece is split between them; a thick body with no neck stays one, two bodies
   joined by a thin bridge become two;
4. each positive cell belongs to the body holding most of its territory, and
   **positive cells sharing a body are one object**. Who is positive does not
   change, only how many there are.

Only the pieces of stain touching at least two positive cells are worked on,
so a section takes seconds. Pure numpy / scipy / scikit-image: no Qt, no napari.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

#: Smoothing of the channel less its local median before the foreground is cut, in µm.
SMOOTH_UM = 0.5
#: Robust SDs above the tissue's centre a pixel needs to be foreground.
FG_K = 3.0
#: Furthest a territory reaches from its nucleus, in µm.
REACH_UM = 20.0
#: How deep a dip in the stain must be, in µm, to part two bodies (the h of the h-maxima).
#: 0.5–3 µm all gave F1 0.42–0.43 against the clicks; 1 µm the best.
H_UM = 1.0


def _robust(values: np.ndarray) -> tuple[float, float]:
    """Median, and the SD the MAD stands for."""
    data = np.asarray(values, dtype=np.float64)
    data = data[np.isfinite(data)]
    if not data.size:
        return float("nan"), float("nan")
    centre = float(np.median(data))
    return centre, float(np.median(np.abs(data - centre)) * 1.4826)


def stain_foreground(filtered: np.ndarray, tissue: np.ndarray | None, voxel_um: float,
                     k: float = FG_K) -> np.ndarray:
    """Pixels clearly labelled: the smoothed channel-less-local-median *k* SDs over the tissue's centre."""
    from scipy import ndimage

    smooth = ndimage.gaussian_filter(np.asarray(filtered, dtype=np.float32), SMOOTH_UM / float(voxel_um))
    tissue = np.ones(smooth.shape, dtype=bool) if tissue is None else np.asarray(tissue, dtype=bool)
    centre, spread = _robust(smooth[tissue][::20])
    if not (np.isfinite(centre) and np.isfinite(spread) and spread > 0):
        return np.zeros(smooth.shape, dtype=bool)
    return (smooth > centre + k * spread) & tissue


def positive_objects(nuclei: np.ndarray, foreground: np.ndarray, positive: Sequence[int], voxel_um: float,
                     h_um: float = H_UM, reach_um: float = REACH_UM) -> dict[int, int]:
    """The object each positive cell belongs to: ``{label: object}``.

    *nuclei* is the label map of every nucleus (all of them seed the
    territories, so a positive cell does not grow into its neighbour's body);
    *positive* the labels called positive. An object is named after the
    smallest label in it; a cell alone in its body keeps its own label.
    """
    from scipy import ndimage
    from skimage.morphology import h_maxima
    from skimage.segmentation import watershed

    positive = np.unique(np.asarray(positive, dtype=np.int64))
    objects = {int(label): int(label) for label in positive}
    if positive.size < 2:
        return objects
    nuclei = np.asarray(nuclei)
    foreground = np.asarray(foreground, dtype=bool)
    is_positive = np.isin(nuclei, positive)
    # A territory never leaves the stain touching its own nucleus, so cells can
    # only share a body inside one group of (stain + positive nuclei).
    groups, _n = ndimage.label(foreground | is_positive)
    owner = groups[is_positive]
    held = nuclei[is_positive].astype(np.int64)
    group_of = dict(zip(held.tolist(), owner.tolist()))
    members: dict[int, list[int]] = {}
    for label, group in group_of.items():
        members.setdefault(group, []).append(label)
    boxes = ndimage.find_objects(groups)
    reach_px = reach_um / float(voxel_um)
    margin = int(np.ceil(reach_px)) + 2
    for group, cells in members.items():
        if len(cells) < 2:
            continue
        box = boxes[group - 1]
        box = tuple(slice(max(s.start - margin, 0), min(s.stop + margin, size))
                    for s, size in zip(box, groups.shape))
        seeds = nuclei[box].astype(np.int32)
        stain = foreground[box] & (groups[box] == group)
        # 2. territories: every nucleus grown over the stain, the narrowings parting them.
        near = ndimage.distance_transform_edt(seeds == 0) <= reach_px
        mask = (stain & near) | (seeds > 0)
        territory = watershed(-ndimage.distance_transform_edt(mask), markers=seeds, mask=mask)
        # 3. bodies: the h-maxima of the depth inside the stain alone, a watershed between them.
        depth = np.where(stain, ndimage.distance_transform_edt(stain) * float(voxel_um), 0.0)
        markers, _m = ndimage.label(h_maxima(depth, h_um))
        basins = watershed(-depth, markers, mask=stain)
        # 4. each positive cell to the body holding most of its territory.
        chosen = np.isin(territory, cells) & (basins > 0)
        if not chosen.any():
            continue
        pairs = np.column_stack([territory[chosen].astype(np.int64), basins[chosen].astype(np.int64)])
        unique, counts = np.unique(pairs, axis=0, return_counts=True)
        order = np.lexsort((-counts, unique[:, 0]))
        first = np.unique(unique[order, 0], return_index=True)[1]
        body_of = dict(zip(unique[order][first, 0].tolist(), unique[order][first, 1].tolist()))
        named: dict[int, int] = {}
        for label in sorted(body_of):
            body = body_of[label]
            named.setdefault(body, label)
            objects[label] = named[body]
    return objects


def count_objects(objects: dict[int, int], labels: Sequence[int]) -> int:
    """How many objects the positive cells *labels* make up."""
    return len({objects.get(int(label), int(label)) for label in labels})
