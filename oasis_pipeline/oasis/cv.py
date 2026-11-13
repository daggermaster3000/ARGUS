"""The AF488 mask: the counting method's image steps.

AF488 (maximum over z) is smoothed (Gaussian, 0.5 µm), its local background (Gaussian, 20 µm, computed at 1/4
resolution) is taken away, and the result is cut inside the tissue at the median + k robust SD (1.4826 × MAD) of the
tissue's pixels (every 20th), k = 3, 5, 8; objects under 20 µm² are removed and holes under 10 µm² filled. The
tissue here is the nuclear stain's Otsu mask at 1/8, closed, holes filled, widened by two steps.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import junction as jn
from . import scan

LABEL_HINT = "488"                #: the labelled-cell channel
SMOOTH_UM = 0.5                   #: smoothing before the threshold (Gaussian σ)
BACKGROUND_UM = 20.0              #: local background (Gaussian σ)
K_MADS = (3.0, 5.0, 8.0)          #: robust SDs above the background: kmad3, kmad5, kmad8 (the counting method)
MIN_OBJECT_UM2 = 20.0             #: smaller objects are removed
MAX_HOLE_UM2 = 10.0               #: smaller holes are filled
METHODS = tuple(f"kmad{k:g}" for k in K_MADS)


def below_normal_priority() -> None:
    """Let other work have the machine first (Windows); does nothing elsewhere."""
    try:
        import ctypes

        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
    except Exception:
        pass


def tissue_mask(nuclear: np.ndarray, step: int = 8) -> np.ndarray:
    """The section, from the nuclear stain at 1/*step*: Otsu, closed, holes filled, widened, at full size."""
    from scipy import ndimage
    from skimage.filters import threshold_otsu

    small = np.asarray(nuclear[::step, ::step], dtype=np.float32)
    mask = small > threshold_otsu(small)
    mask = ndimage.binary_closing(mask, iterations=3)
    mask = ndimage.binary_fill_holes(mask)
    mask = ndimage.binary_dilation(mask, iterations=2)
    full = np.repeat(np.repeat(mask, step, axis=0), step, axis=1)
    return full[: nuclear.shape[0], : nuclear.shape[1]]


def local_background(image: np.ndarray, sigma_px: float, step: int = 4) -> np.ndarray:
    """A Gaussian of *sigma_px* over *image*, computed at 1/*step* and scaled back up."""
    from scipy import ndimage

    height, width = image.shape
    small = image[: height // step * step, : width // step * step]
    small = small.reshape(height // step, step, width // step, step).mean(axis=(1, 3))
    small = ndimage.gaussian_filter(small, sigma_px / step)
    full = ndimage.zoom(small, step, order=1)
    out = np.empty(image.shape, dtype=np.float32)
    out[: full.shape[0], : full.shape[1]] = full[: height, : width]
    if full.shape[0] < height:
        out[full.shape[0]:, :] = out[full.shape[0] - 1: full.shape[0], :]
    if full.shape[1] < width:
        out[:, full.shape[1]:] = out[:, full.shape[1] - 1: full.shape[1]]
    return out


#: 4-connected, as scikit-image's remove_small_objects / remove_small_holes by default
_CROSS = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool)


def remove_small_objects(mask: np.ndarray, size_px: int) -> np.ndarray:
    """*mask* without its objects (4-connected) of *size_px* pixels or fewer.

    Written out rather than taken from scikit-image, whose versions disagree on the boundary (0.25 removes objects
    under *size_px*, 0.26 those up to and including it); the counts were made with the 0.26 rule, kept here."""
    from scipy import ndimage

    labels, n = ndimage.label(mask, structure=_CROSS)
    if n == 0:
        return mask.astype(bool, copy=True)
    sizes = np.bincount(labels.ravel())
    keep = sizes > size_px
    keep[0] = False
    return keep[labels]


def remove_small_holes(mask: np.ndarray, size_px: int) -> np.ndarray:
    """*mask* with its holes (4-connected background) of *size_px* pixels or fewer filled (the 0.26 rule, as above)."""
    from scipy import ndimage

    labels, n = ndimage.label(~mask, structure=_CROSS)
    if n == 0:
        return mask.astype(bool, copy=True)
    sizes = np.bincount(labels.ravel())
    fill = sizes <= size_px
    fill[0] = False
    return mask | fill[labels]


def classical_masks(af488: np.ndarray, tissue: np.ndarray, voxel: float) -> tuple[dict[str, np.ndarray], dict]:
    """The AF488 masks at every k of :data:`K_MADS`, and their thresholds."""
    from scipy import ndimage

    smooth = ndimage.gaussian_filter(af488, SMOOTH_UM / voxel)
    signal = smooth - local_background(smooth, BACKGROUND_UM / voxel)
    del smooth
    values = signal[tissue][::20]
    centre = float(np.median(values))
    spread = float(1.4826 * np.median(np.abs(values - centre)))
    thresholds = {f"kmad{k:g}": centre + k * spread for k in K_MADS}
    min_px = int(round(MIN_OBJECT_UM2 / voxel ** 2))
    hole_px = int(round(MAX_HOLE_UM2 / voxel ** 2))
    masks = {}
    for method in METHODS:
        mask = (signal > thresholds[method]) & tissue
        masks[method] = remove_small_holes(remove_small_objects(mask, min_px), hole_px)
    return masks, {"thresholds": thresholds, "background_centre": centre, "noise_sd": spread}


def read_planes(path: Path):
    """AF488 (float32) and the nuclear stain, maximum over z, and the pixel size (µm)."""
    specs = scan.load_path(path)
    names = [str(spec.channel_name or spec.name) for spec in specs]
    label_index = next(i for i, name in enumerate(names) if LABEL_HINT in name)
    nuclear_index = jn.pick_channel(names, "", jn.NUCLEAR_HINTS)
    voxel = float(specs[0].scale[-1])
    return scan.plane(specs[label_index]).astype(np.float32), scan.plane(specs[nuclear_index]), voxel
