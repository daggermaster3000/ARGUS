"""The section's outline on a 5 µm overview, and distances to it; picking channels by name.

The overview keeps every *step*-th pixel of each channel's maximum over z (step = 5 µm / pixel size). The section's
outline (:func:`tissue_of`) is where any channel, smoothed by 10 µm, stands a fifth of the way from the slide
background to the tissue (Otsu's split of the smoothed values), opened by 10 µm and closed by 15 µm, pieces under a
tenth of the largest dropped, holes filled. This is the outline the junction was found in and the cells' distance to
the tissue edge is measured to.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

OVERVIEW_UM = 5.0                 #: overview pixel size
NUCLEAR_HINTS = ("dapi", "hoechst", "405")
LABEL_HINTS = ("gfp", "488")
OTHER_HINTS = ("mcherry", "tdtomato", "rfp", "555", "561", "568", "594")
TISSUE_ALL_CHANNELS = True        #: the outline from every channel, not the nuclear stain alone
MEDIAN_UM = 45.0                  #: local median window (cells.less_local_median)


@dataclass
class JunctionOptions:
    """Which channels are which ("" = by the hints above)."""

    nuclear_channel: int | str = ""
    label_channel: int | str = ""
    other_channel: int | str = ""
    save: bool = False


@dataclass
class Overview:
    """One section, flattened over z and decimated."""

    channels: np.ndarray          #: (C, y, x) float32
    names: list[str]
    um: float                     #: µm per overview pixel
    voxel_yx: tuple[float, float]
    shape_yx: tuple[int, int]
    step: int
    nuclear: int = 0
    label: int = 0
    other: int | None = None


def _by_name(names: Sequence[str], wanted) -> int | None:
    if wanted is None:
        return None
    if isinstance(wanted, (int, np.integer)) and not isinstance(wanted, bool):
        index = int(wanted)
        return index if 0 <= index < len(names) else None
    needle = str(wanted).strip().lower()
    if not needle:
        return None
    for index, name in enumerate(names):
        if needle in str(name).lower():
            return index
    return None


def pick_channel(names: Sequence[str], wanted: int | str, hints: Sequence[str]) -> int | None:
    """Index of the channel *wanted* names, else the first one a hint matches."""
    if wanted not in ("", None):
        return _by_name(names, wanted)
    for hint in hints:
        index = _by_name(names, hint)
        if index is not None:
            return index
    return None


def pick_other_channel(names: Sequence[str], wanted: int | str, nuclear: int, label: int) -> int | None:
    taken = {nuclear, label}
    if wanted not in ("", None):
        index = pick_channel(names, wanted, ())
        return index if index is not None and index not in taken else None
    for hint in OTHER_HINTS:
        index = pick_channel(names, hint, ())
        if index is not None and index not in taken:
            return index
    left = [index for index in range(len(names)) if index not in taken]
    return left[0] if left else None


def build_overview(specs: Sequence[Any], options: JunctionOptions | None = None,
                   target_um: float = OVERVIEW_UM) -> Overview:
    """Every channel of one section: maximum over z of every *step*-th pixel."""
    import tifffile

    options = options or JunctionOptions()
    names = [str(spec.channel_name or spec.name) for spec in specs]
    nuclear = pick_channel(names, options.nuclear_channel, NUCLEAR_HINTS)
    label = pick_channel(names, options.label_channel, LABEL_HINTS)
    if nuclear is None or label is None:
        raise ValueError(f"no nuclear or labelled-cell channel among: {', '.join(names)}")
    other = pick_other_channel(names, options.other_channel, nuclear, label)
    voxel_yx = (float(specs[0].scale[-2]), float(specs[0].scale[-1]))
    pixel = max(voxel_yx) or 1.0
    step = max(1, int(round(target_um / pixel)))
    planes = []
    with tifffile.TiffFile(str(specs[0].path)) as tif:
        for spec in specs:
            best = None
            for index in spec.pages:
                small = tif.pages[index].asarray()[::step, ::step].astype(np.float32)
                best = small if best is None else np.maximum(best, small)
            planes.append(best)
    return Overview(channels=np.stack(planes), names=names, um=pixel * step, voxel_yx=voxel_yx,
                    shape_yx=tuple(specs[0].shape_yx), step=step, nuclear=nuclear, label=label, other=other)


def _disk(radius_px: float) -> np.ndarray:
    radius = max(int(round(radius_px)), 1)
    grid = np.arange(-radius, radius + 1)
    return (grid[:, None] ** 2 + grid[None, :] ** 2) <= radius * radius


def _above_background(plane: np.ndarray, valid: np.ndarray, um: float) -> np.ndarray | None:
    """Where one channel, smoothed by 10 µm, stands a fifth of the way from background to tissue."""
    from scipy import ndimage
    from skimage.filters import threshold_otsu

    smooth = ndimage.gaussian_filter(np.asarray(plane, dtype=np.float32), 10.0 / um)
    values = smooth[valid]
    if values.size < 100 or float(values.max()) <= float(values.min()):
        return None
    midpoint = threshold_otsu(values)
    below, above = values[values < midpoint], values[values >= midpoint]
    if below.size == 0 or above.size == 0:
        return None
    cut = float(np.median(below)) + 0.2 * (float(np.median(above)) - float(np.median(below)))
    return (smooth > cut) & valid


def tissue_mask(nuclear: np.ndarray, valid: np.ndarray, um: float, others: Sequence[np.ndarray] = ()) -> np.ndarray:
    """Where the tissue is: any channel above the slide, opened, closed, small pieces dropped (holes kept)."""
    from scipy import ndimage

    mask = _above_background(nuclear, valid, um)
    if mask is None:
        return np.zeros(np.shape(nuclear), dtype=bool)
    for plane in others:
        more = _above_background(plane, valid, um)
        if more is not None:
            mask |= more
    mask = ndimage.binary_opening(mask, _disk(10.0 / um))
    mask = ndimage.binary_closing(mask, _disk(15.0 / um))
    labels, count = ndimage.label(mask)
    if count == 0:
        return mask
    sizes = np.bincount(labels.ravel())[1:]
    keep = np.flatnonzero(sizes >= 0.1 * sizes.max()) + 1          # debris dropped; a second organoid kept
    return np.isin(labels, keep)


def _others(channels: np.ndarray, nuclear: int, use: bool) -> list[np.ndarray]:
    return [plane for k, plane in enumerate(channels) if k != nuclear] if use else []


def tissue_of(overview: Overview, all_channels: bool = TISSUE_ALL_CHANNELS) -> np.ndarray:
    """The section's outline, holes filled, on the overview grid."""
    from scipy import ndimage

    valid = (overview.channels > 0).all(axis=0)
    tissue = tissue_mask(overview.channels[overview.nuclear], valid, overview.um,
                         _others(overview.channels, overview.nuclear, all_channels))
    return ndimage.binary_fill_holes(tissue)


def edge_distance(overview, y_px: np.ndarray, x_px: np.ndarray, filled: np.ndarray | None = None) -> np.ndarray:
    """Distance (µm) from each point (scan pixels) to the section's outline: positive inside, negative outside."""
    from scipy import ndimage

    if filled is None:
        filled = tissue_of(overview)
    um, step = overview.um, overview.step
    inside = ndimage.distance_transform_edt(filled) * um - um / 2
    outside = ndimage.distance_transform_edt(~filled) * um - um / 2
    signed = np.where(filled, inside, -outside)
    coords = np.vstack([(np.asarray(y_px, float) - (step - 1) / 2) / step,
                        (np.asarray(x_px, float) - (step - 1) / 2) / step])
    return ndimage.map_coordinates(signed, coords, order=1, mode="nearest").astype(np.float32)
