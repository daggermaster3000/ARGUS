"""A mean outline of one brain region across samples, and where the cells sit in it.

Every sample's outline of a region (the cerebellum, say) is resampled to the same
number of points along its edge and registered onto the others by generalized
Procrustes analysis: translation, rotation and — unless sizes are to be kept —
scale, with the starting point of each closed contour found by trying every
cyclic shift. The average of the registered contours is the *template*: the mean
shape of the region.

Each sample's cells are then carried into the template with the same transform
and, optionally, a thin-plate spline that bends that sample's registered outline
exactly onto the template, so a cell near the edge of a small cerebellum lands
near the edge of the template too. Density maps over the template can then be
averaged per genotype and compared.

Coordinates are image micrometres throughout, as the analysis workbook stores
them: columns are ``(x, y)`` with y growing downwards. No Qt, no Streamlit.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: Points each outline is resampled to. Enough for a smooth brain region, few
#: enough that trying every cyclic shift stays instant.
DEFAULT_POINTS = 128
#: Procrustes rounds. It settles in three or four.
ITERATIONS = 8


# ---------------------------------------------------------------------------
# Contours
# ---------------------------------------------------------------------------


def signed_area(points: np.ndarray) -> float:
    """Shoelace area; the sign says which way the contour runs."""
    x, y = points[:, 0], points[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def resample_closed(points: np.ndarray, n: int = DEFAULT_POINTS) -> np.ndarray:
    """*n* points evenly spaced along the closed contour through *points*.

    The result always runs the same way round (positive shoelace area), so two
    outlines drawn in opposite directions can still be matched point for point.
    """
    points = np.asarray(points, dtype=float)
    if len(points) >= 2 and np.allclose(points[0], points[-1]):
        points = points[:-1]
    if len(points) < 3:
        raise ValueError("an outline needs at least three points")
    if signed_area(points) < 0:
        points = points[::-1]
    closed = np.vstack([points, points[:1]])
    steps = np.hypot(*np.diff(closed, axis=0).T)
    along = np.concatenate([[0.0], np.cumsum(steps)])
    wanted = np.linspace(0.0, along[-1], n, endpoint=False)
    return np.column_stack(
        [np.interp(wanted, along, closed[:, 0]), np.interp(wanted, along, closed[:, 1])]
    )


def largest_part(parts: list[np.ndarray]) -> np.ndarray:
    """The part of a multi-part outline with the largest area."""
    return max(parts, key=lambda part: abs(signed_area(np.asarray(part, dtype=float))))


# ---------------------------------------------------------------------------
# Similarity transforms
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Similarity:
    """``p -> scale * p @ rotation.T + shift`` for rows of (x, y)."""

    scale: float = 1.0
    rotation: np.ndarray = field(default_factory=lambda: np.eye(2))
    shift: np.ndarray = field(default_factory=lambda: np.zeros(2))

    def apply(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=float).reshape(-1, 2)
        return self.scale * points @ self.rotation.T + self.shift

    @property
    def angle(self) -> float:
        """Rotation in degrees."""
        return float(np.degrees(np.arctan2(self.rotation[1, 0], self.rotation[0, 0])))

    @property
    def mirrored(self) -> bool:
        return bool(np.linalg.det(self.rotation) < 0)


def fit_similarity(
    source: np.ndarray, target: np.ndarray, *, scale: bool = True, reflect: bool = False
) -> Similarity:
    """Least-squares transform taking *source* onto *target* (Umeyama)."""
    mu_s, mu_t = source.mean(axis=0), target.mean(axis=0)
    a, b = source - mu_s, target - mu_t
    u, sigma, vt = np.linalg.svd(b.T @ a)
    d = np.ones(2)
    if not reflect and np.linalg.det(u @ vt) < 0:
        d[-1] = -1.0
    rotation = u @ np.diag(d) @ vt
    variance = float((a**2).sum())
    factor = float((sigma * d).sum() / variance) if scale and variance > 0 else 1.0
    return Similarity(factor, rotation, mu_t - factor * mu_s @ rotation.T)


def align_contour(
    contour: np.ndarray, target: np.ndarray, *, scale: bool = True, reflect: bool = False
) -> tuple[np.ndarray, Similarity]:
    """Best cyclic start and transform of *contour* onto *target*.

    Returns the contour re-ordered so its point *i* matches the target's point
    *i*, and the transform (applying to the original coordinates). A mirrored
    fit also reverses the order, since mirroring flips the direction of travel.
    """
    best = None
    orders = [contour, contour[::-1]] if reflect else [contour]
    for ordered in orders:
        for shift in range(len(ordered)):
            candidate = np.roll(ordered, -shift, axis=0)
            transform = fit_similarity(candidate, target, scale=scale, reflect=reflect)
            error = float(((transform.apply(candidate) - target) ** 2).sum())
            if best is None or error < best[0]:
                best = (error, candidate, transform)
    return best[1], best[2]


# ---------------------------------------------------------------------------
# Thin-plate spline
# ---------------------------------------------------------------------------


class ThinPlateWarp:
    """Bends *source* landmarks onto *target* landmarks; smooth in between."""

    def __init__(self, source: np.ndarray, target: np.ndarray, smoothing: float = 0.0):
        from scipy.interpolate import RBFInterpolator

        self._map = RBFInterpolator(
            source, target, kernel="thin_plate_spline", smoothing=float(smoothing)
        )

    def apply(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=float).reshape(-1, 2)
        if not len(points):
            return points
        return self._map(points)


# ---------------------------------------------------------------------------
# The atlas
# ---------------------------------------------------------------------------


@dataclass
class Atlas:
    """The template outline and, per sample, how to get into it."""

    template: np.ndarray
    #: ``sample -> contour`` in template space after the similarity transform
    #: only; point *i* corresponds to template point *i*.
    registered: dict[str, np.ndarray]
    transforms: dict[str, Similarity]
    warps: dict[str, ThinPlateWarp]
    #: ``sample -> RMS distance (µm)`` of the registered contour from the template.
    residual: dict[str, float]

    @property
    def samples(self) -> list[str]:
        return list(self.registered)

    def map_points(self, sample: str, points: np.ndarray) -> np.ndarray:
        """Image-µm *points* of *sample* carried into template space."""
        moved = self.transforms[sample].apply(points)
        warp = self.warps.get(sample)
        return warp.apply(moved) if warp is not None else moved

    def mean_of(self, samples) -> np.ndarray:
        """Mean registered contour of a subset, e.g. one genotype."""
        chosen = [self.registered[s] for s in samples if s in self.registered]
        return np.mean(chosen, axis=0) if chosen else np.empty((0, 2))


def centroid_size(contour: np.ndarray) -> float:
    return float(np.sqrt(((contour - contour.mean(axis=0)) ** 2).sum(axis=1).mean()))


def build_atlas(
    outlines: dict[str, np.ndarray],
    *,
    n_points: int = DEFAULT_POINTS,
    scale: bool = True,
    reflect: bool = False,
    warp: bool = True,
    smoothing: float = 0.0,
    reference: str | None = None,
) -> Atlas:
    """Register *outlines* (``sample -> (N, 2)`` image µm) into a mean shape.

    With *scale* every outline is brought to a common size, so the template is
    the mean *shape*, drawn at the samples' mean size; without it sizes are kept
    and only position and rotation are removed. The template is left in the
    position and orientation of *reference* (the first sample by default), so it
    reads like the images do.
    """
    if not outlines:
        raise ValueError("no outlines to register")
    contours = {name: resample_closed(points, n_points) for name, points in outlines.items()}
    names = list(contours)
    reference = reference if reference in contours else names[0]
    size = float(np.mean([centroid_size(c) for c in contours.values()]))

    template = contours[reference].copy()
    anchor = template.mean(axis=0)
    for _ in range(ITERATIONS):
        registered = {}
        for name in names:
            ordered, transform = align_contour(contours[name], template, scale=scale, reflect=reflect)
            registered[name] = transform.apply(ordered)
        mean = np.mean(list(registered.values()), axis=0)
        # Pin the frame down, or it drifts between rounds: the reference's
        # place and turn, and the samples' mean size.
        fixed = fit_similarity(mean, template, scale=False)
        mean = fixed.apply(mean)
        mean = mean - mean.mean(axis=0)
        if scale:
            mean *= size / max(centroid_size(mean), 1e-12)
        mean += anchor
        moved = float(np.sqrt(((mean - template) ** 2).sum(axis=1).mean()))
        template = mean
        if moved < 1e-3:
            break

    registered, transforms, warps, residual = {}, {}, {}, {}
    for name in names:
        ordered, transform = align_contour(contours[name], template, scale=scale, reflect=reflect)
        placed = transform.apply(ordered)
        registered[name] = placed
        transforms[name] = transform
        residual[name] = float(np.sqrt(((placed - template) ** 2).sum(axis=1).mean()))
        if warp:
            warps[name] = ThinPlateWarp(placed, template, smoothing=smoothing)
    return Atlas(template, registered, transforms, warps, residual)


# ---------------------------------------------------------------------------
# Density over the template
# ---------------------------------------------------------------------------

#: Per-sample density units.
PER_AREA = "Cells per 1000 µm²"
SHARE = "Share of the sample's cells (% per 1000 µm²)"


@dataclass
class DensityGrid:
    x: np.ndarray  # pixel centres along x, µm
    y: np.ndarray  # pixel centres along y, µm
    inside: np.ndarray  # (len(y), len(x)) bool, pixel centre inside the template
    maps: dict[str, np.ndarray]  # sample -> (len(y), len(x)), NaN outside

    def mean_of(self, samples) -> np.ndarray:
        chosen = [self.maps[s] for s in samples if s in self.maps]
        if not chosen:
            return np.full(self.inside.shape, np.nan)
        return np.mean(chosen, axis=0)


def inside_polygon(polygon: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Grid mask of the pixel centres inside *polygon* (even-odd rule)."""
    gx, gy = np.meshgrid(x, y)
    px, py = gx.ravel(), gy.ravel()
    inside = np.zeros(px.shape, dtype=bool)
    vx, vy = polygon[:, 0], polygon[:, 1]
    jx, jy = np.roll(vx, 1), np.roll(vy, 1)
    for x0, y0, x1, y1 in zip(vx, vy, jx, jy):
        crosses = (y0 > py) != (y1 > py)
        with np.errstate(divide="ignore", invalid="ignore"):
            at = (x1 - x0) * (py - y0) / (y1 - y0) + x0
        inside ^= crosses & (px < at)
    return inside.reshape(gx.shape)


def density_maps(
    template: np.ndarray,
    points: dict[str, np.ndarray],
    *,
    sigma: float = 10.0,
    pixel: float | None = None,
    units: str = PER_AREA,
    margin: float = 0.05,
) -> DensityGrid:
    """Gaussian-smoothed cell density of each sample over the template.

    *points* maps sample -> (N, 2) template-space µm. *sigma* is the smoothing
    radius in µm. Each sample's map is on its own, so averaging them weighs
    every sample the same however many cells it has. Pixels outside the
    template are NaN.
    """
    from scipy.ndimage import gaussian_filter

    low, high = template.min(axis=0), template.max(axis=0)
    pad = (high - low) * margin
    low, high = low - pad, high + pad
    if pixel is None:
        pixel = float(max(high - low)) / 160.0
    pixel = max(float(pixel), 1e-6)
    edges_x = np.arange(low[0], high[0] + pixel, pixel)
    edges_y = np.arange(low[1], high[1] + pixel, pixel)
    x = 0.5 * (edges_x[:-1] + edges_x[1:])
    y = 0.5 * (edges_y[:-1] + edges_y[1:])
    inside = inside_polygon(template, x, y)

    maps = {}
    for sample, where in points.items():
        where = np.asarray(where, dtype=float).reshape(-1, 2)
        counts, _, _ = np.histogram2d(where[:, 1], where[:, 0], bins=[edges_y, edges_x])
        smooth = gaussian_filter(counts, sigma=max(float(sigma), 0.0) / pixel, mode="constant")
        per_um2 = smooth / (pixel * pixel)
        if units == SHARE:
            value = per_um2 * 1000.0 * 100.0 / len(where) if len(where) else per_um2 * 0.0
        else:
            value = per_um2 * 1000.0
        maps[sample] = np.where(inside, value, np.nan)
    return DensityGrid(x, y, inside, maps)
