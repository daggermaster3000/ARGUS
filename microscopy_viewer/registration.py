"""Register a confocal brain onto a reference atlas, and read signal out per region.

The problem this solves: a 7 dpf zebrafish brain imaged in three channels has one
channel that says *where the brain is* — the nuclear stain — and one that says
what you actually care about. Those are not the same channel, and the second one
must never be allowed to drive the fit. Anti-SV2 is punctate and regional; asked
to match an atlas it will happily warp a synaptic domain onto whatever the atlas
has in that corner. So:

* **DAPI drives the registration, alone.** It is the only channel with signal
  everywhere in the brain, which is what a global affine plus SyN needs.
* **Acetylated tubulin is a landmark and a QC channel.** Tracts are the thing to
  eyeball when deciding whether a fit worked. It can be added as a second metric
  term (:attr:`RegistrationSettings.use_landmark_metric`) but is off by default,
  because a tract-rich channel pulls the warp towards tracts at the expense of
  everything between them.
* **SV2 is carried.** It is resampled through the transform DAPI produced and
  never contributes a metric.

Registration happens in *physical* space: every array arrives with its voxel size
in µm, taken from the calibrated scale the reader already put on the layer, so an
anisotropic confocal stack and a 0.798 × 0.798 × 2 µm atlas line up without anyone
resampling anything by hand first.

Nothing in this module imports Qt or napari, and the backend is imported lazily,
so the whole thing is testable headless and ``antspyx`` stays off the startup
path. ANTs is the backend that exists today; :class:`Backend` is the seam that
lets ``itk-elastix`` be dropped in beside it, which matters because antspyx is
the fussier of the two to install.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import math
import numpy as np

from .utils import get_logger

logger = get_logger("registration")

# ---------------------------------------------------------------------------
# Roles
# ---------------------------------------------------------------------------

#: Drives the fit. The nuclear channel, and nothing else.
ROLE_DRIVER = "Registration driver"
#: Eyeballed afterwards, optionally a second metric term. Never on its own.
ROLE_LANDMARK = "Landmark / QC"
#: Resampled through the driver's transform. Contributes nothing to the fit.
ROLE_CARRY = "Carry-along"

ROLES = (ROLE_DRIVER, ROLE_LANDMARK, ROLE_CARRY)

#: Channel-name fragments that identify the nuclear stain, i.e. the one channel
#: allowed to drive a fit. Matched with separators stripped, the same way
#: :func:`~microscopy_viewer.loaders.layer_spec.is_brightfield` matches, so
#: "Confocal - dapi", "DAPI_405" and "Hoechst" all hit.
DRIVER_KEYWORDS = ("dapi", "hoechst", "sytox", "draq5", "topro", "nuclear", "nucleus", "h2b")

#: Fragments identifying a tract or landmark channel. QC and, if asked for, a
#: second metric term — never a driver on its own.
LANDMARK_KEYWORDS = ("actub", "acetylatedtubulin", "tubulin", "tuj", "neurofilament", "terk", "elavl3")


def guess_role(channel_name: str | None) -> str:
    """The role a channel name suggests, defaulting to carry-along.

    Defaulting to carry-along is the safe direction: a channel nobody recognises
    gets resampled and measured, but is never quietly promoted to steering the
    registration.
    """
    if not channel_name:
        return ROLE_CARRY
    squashed = re.sub(r"[^a-z0-9]", "", str(channel_name).lower())
    if any(word in squashed for word in DRIVER_KEYWORDS):
        return ROLE_DRIVER
    if any(word in squashed for word in LANDMARK_KEYWORDS):
        return ROLE_LANDMARK
    return ROLE_CARRY

#: Filename fragments that identify a nuclear reference in an atlas download.
#: A nuclear-to-nuclear registration is same-modality, which is the easy case;
#: everything else here is a fallback.
NUCLEAR_KEYWORDS = (
    "h2b", "nuclear", "nucleus", "dapi", "hoechst", "sytox", "draq5", "topro", "to-pro", "dsred-nuc",
)

#: Fragments identifying the pan-neuronal reference most larval atlases are built
#: on. Usable, but cross-modality against DAPI, so it is only taken when no
#: nuclear channel is present — and it is warned about when it is.
FALLBACK_KEYWORDS = ("terk", "t-erk", "elavl3", "huc", "reference", "ref")

#: Fragments identifying the region masks that come with an atlas. Deliberately
#: not "atlas": half the reference volumes in these downloads are called
#: ``atlas_something``, and treating that as a mask leaves a folder with no
#: usable reference at all.
LABEL_KEYWORDS = ("mask", "label", "anatomy", "segmentation", "region")

#: Volume formats an atlas download realistically arrives in.
VOLUME_SUFFIXES = (".nrrd", ".nii", ".nii.gz", ".tif", ".tiff", ".h5", ".hdf5", ".mha", ".mhd")

#: Region masks may also arrive as a MATLAB v7.3 file — Z-Brain's MaskDatabase.mat.
LABEL_SUFFIXES = VOLUME_SUFFIXES + (".mat",)

#: Everything in this module works in ``(z, y, x)``, matching the array order the
#: readers produce and the order ``layer.scale`` is in. Atlas files do not agree
#: with each other about this, so each source declares what its own order is and
#: is permuted into that convention once, on the way in. Getting it wrong does not
#: raise anything — it silently asks the optimiser to find a 90° rotation, which
#: it will not — so it is worth being explicit about.
#:
#: The three that turn up in one Z-Brain download alone:
#: ``Ref20131120pt14pl2.nrrd`` is ``xyz``, ``AnatomyLabelDatabase.hdf5`` is
#: ``zxy`` (MATLAB writes ``y x z`` and HDF5 reverses it), and
#: ``MaskDatabase.mat`` is ``yxz``.
CANONICAL_ORDER = "zyx"

#: Axis order of an ITK-read volume (NRRD, NIfTI, MHA): fastest axis first.
ITK_ORDER = "xyz"
#: Axis order of a 3D array written by MATLAB and read back through h5py.
MATLAB_HDF5_ORDER = "zxy"
#: Axis order of a MATLAB array reshaped column-major from its own linear indices.
MATLAB_LINEAR_ORDER = "yxz"

#: Percentile above which a voxel counts as "signal" for the per-region overlap
#: fraction, when the caller does not name a threshold.
DEFAULT_THRESHOLD_PERCENTILE = 95.0


# ---------------------------------------------------------------------------
# What goes in
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Volume:
    """One channel: the array, and how big its voxels are.

    ``voxel_size_um`` is ``(z, y, x)`` — the same order as the array axes and as
    ``layer.scale``, so the widget can pass a layer's own calibration straight
    through without reordering anything or re-reading the file.
    """

    name: str
    data: Any
    voxel_size_um: tuple[float, float, float]
    role: str = ROLE_CARRY

    def array(self) -> np.ndarray:
        """The pixels as a real numpy array. Computes dask, so call it off-thread."""
        values = np.asarray(self.data)
        if values.ndim != 3:
            raise ValueError(
                f"{self.name}: registration needs a 3D volume, got {values.ndim}D with shape {values.shape}"
            )
        return values

    @property
    def spacing(self) -> tuple[float, float, float]:
        """Voxel size with anything missing or nonsensical replaced by 1 µm."""
        return tuple(
            float(value) if value and float(value) > 0 else 1.0 for value in self.voxel_size_um
        )  # type: ignore[return-value]


@dataclass(frozen=True)
class AtlasSpec:
    """Which atlas files to use, and what the reference channel actually is."""

    reference_path: Path
    #: Free text for the record and the log: ``"nuclear"``, ``"tERK"``, …
    reference_channel: str = ""
    label_path: Path | None = None
    label_names_path: Path | None = None
    #: Used when the reference file carries no spacing of its own.
    voxel_size_um: tuple[float, float, float] | None = None
    #: False means the reference is not a nuclear stain, so DAPI is being
    #: registered across modalities. Warned about, never silently accepted.
    is_nuclear: bool = True
    #: Dataset inside the reference file, when it holds more than one volume.
    #: Z-Brain ships 29 averaged stacks in one HDF5, of which exactly one — the
    #: H2B-RFP nuclear average — is the right thing to register DAPI against.
    reference_dataset: str = ""
    #: Overrides the axis order guessed for the reference file. See
    #: :data:`CANONICAL_ORDER`.
    axis_order: str = ""

    def describe(self) -> str:
        modality = "nuclear" if self.is_nuclear else f"non-nuclear ({self.reference_channel or 'unknown'})"
        name = self.reference_path.name
        if self.reference_dataset:
            name = f"{name}::{self.reference_dataset}"
        return f"{name} [{modality}]"


@dataclass
class RegistrationSettings:
    """Knobs for one run. Defaults are what a 7 dpf brain onto a larval atlas wants."""

    backend: str = "ants"
    #: Affine only. Minutes instead of tens of minutes — the sanity pass you do
    #: first to check the stack is not mirrored before paying for SyN.
    affine_only: bool = False
    #: Add the landmark channel as a second metric term. Off: a tract channel
    #: pulls the deformation onto tracts and lets the space between them drift.
    use_landmark_metric: bool = False
    landmark_weight: float = 0.3
    #: Registration is done at or below this many voxels. SyN on a full 2040² ×
    #: 100 stack is hours; the transform is smooth, so fitting it on a decimated
    #: volume and applying it at atlas resolution loses nothing that matters.
    max_voxels: int = 40_000_000
    #: Intensity percentiles clipped before fitting. A confocal stack with a
    #: handful of saturated nuclei otherwise spends its whole dynamic range on
    #: them, and mutual information degrades accordingly.
    winsorize: tuple[float, float] = (0.5, 99.5)
    #: Axes to flip before registering. SyN cannot undo a mirrored stack — it
    #: converges to a confident, wrong answer — so handedness is fixed up front.
    flip_axes: tuple[int, ...] = ()
    random_seed: int = 12345

    @property
    def transform_type(self) -> str:
        return "Affine" if self.affine_only else "SyN"


# ---------------------------------------------------------------------------
# What comes out
# ---------------------------------------------------------------------------


@dataclass
class RegistrationResult:
    """One run: the transforms, what they were applied to, and how it went."""

    #: Transform files taking the fish into atlas space, ANTs order.
    forward_transforms: list[str] = field(default_factory=list)
    #: The same mapping backwards, for bringing atlas labels onto the fish.
    inverse_transforms: list[str] = field(default_factory=list)
    #: Channel name -> volume resampled onto the atlas grid.
    warped: dict[str, np.ndarray] = field(default_factory=dict)
    atlas_voxel_size_um: tuple[float, float, float] = (1.0, 1.0, 1.0)
    #: Quality numbers, notably the mutual information before and after.
    metrics: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    elapsed_s: float = 0.0
    backend: str = ""
    transform_type: str = ""

    @property
    def atlas_voxel_volume_um3(self) -> float:
        z, y, x = self.atlas_voxel_size_um
        return float(z) * float(y) * float(x)


@dataclass
class RegionStat:
    """SV2 signal inside one atlas region."""

    region_id: int
    name: str
    n_voxels: int
    volume_um3: float
    mean: float
    median: float
    std: float
    integrated: float
    #: Fraction of the region's voxels above *threshold* — the overlap number.
    fraction_above: float
    threshold: float

    def as_row(self) -> dict[str, Any]:
        return {
            "region_id": self.region_id,
            "region": self.name,
            "n_voxels": self.n_voxels,
            "volume_um3": self.volume_um3,
            "mean": self.mean,
            "median": self.median,
            "std": self.std,
            "integrated": self.integrated,
            "fraction_above": self.fraction_above,
            "threshold": self.threshold,
        }


#: Column order and headings for the exported region table.
REGION_COLUMNS = (
    "region_id", "region", "n_voxels", "volume_um3", "mean", "median", "std",
    "integrated", "fraction_above", "threshold",
)

REGION_LABELS = {
    "region_id": "Region ID",
    "region": "Region",
    "n_voxels": "Voxels",
    "volume_um3": "Volume (µm³)",
    "mean": "Mean intensity",
    "median": "Median intensity",
    "std": "Std",
    "integrated": "Integrated intensity",
    "fraction_above": "Fraction above threshold",
    "threshold": "Threshold",
}


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


class Backend:
    """What a registration library has to provide.

    Kept to three methods on purpose. ANTs is what runs today; elastix has the
    same three concepts under different names, so swapping it in is writing one
    more subclass rather than touching :func:`register_to_atlas`.
    """

    name = ""
    install_hint = ""

    def available(self) -> bool:
        raise NotImplementedError

    def register(
        self,
        fixed: np.ndarray,
        fixed_spacing: tuple[float, float, float],
        moving: np.ndarray,
        moving_spacing: tuple[float, float, float],
        settings: RegistrationSettings,
        extra_metric: tuple[np.ndarray, np.ndarray] | None = None,
        progress: Callable[[str], None] | None = None,
    ) -> tuple[list[str], list[str], dict[str, float]]:
        """Fit *moving* onto *fixed*. Returns ``(forward, inverse, metrics)``."""
        raise NotImplementedError

    def apply(
        self,
        moving: np.ndarray,
        moving_spacing: tuple[float, float, float],
        reference: np.ndarray,
        reference_spacing: tuple[float, float, float],
        transforms: Sequence[str],
        interpolation: str = "linear",
    ) -> np.ndarray:
        """Resample *moving* onto *reference*'s grid through *transforms*."""
        raise NotImplementedError


class AntsBackend(Backend):
    """antspyx: affine then SyN, mutual information throughout.

    Mutual information rather than cross-correlation even for the nuclear-to-
    nuclear case: it costs little when the modalities match and it is the only
    thing that works when the atlas has no nuclear channel and DAPI ends up being
    matched against tERK.
    """

    name = "ants"
    install_hint = (
        "Atlas registration needs the antspyx package:\n\n"
        "    python -m pip install antspyx\n\n"
        "It is a large, platform-specific wheel; on Apple Silicon it may need to be built "
        "from source."
    )

    def available(self) -> bool:
        try:
            import ants  # noqa: F401
        except Exception:
            return False
        return True

    @staticmethod
    def _image(array: np.ndarray, spacing: tuple[float, float, float]):
        import ants

        return ants.from_numpy(
            np.ascontiguousarray(np.asarray(array, dtype=np.float32)),
            spacing=tuple(float(v) for v in spacing),
        )

    def register(
        self,
        fixed,
        fixed_spacing,
        moving,
        moving_spacing,
        settings,
        extra_metric=None,
        progress=None,
    ):
        import ants

        fixed_image = self._image(fixed, fixed_spacing)
        moving_image = self._image(moving, moving_spacing)

        kwargs: dict[str, Any] = {
            "fixed": fixed_image,
            "moving": moving_image,
            "type_of_transform": settings.transform_type,
            "aff_metric": "mattes",
            "syn_metric": "mattes",
            "random_seed": int(settings.random_seed),
        }
        if extra_metric is not None:
            # A second metric term evaluated on the landmark channel, weighted
            # below the driver so tracts inform the warp without steering it.
            landmark_fixed, landmark_moving = extra_metric
            kwargs["multivariate_extras"] = [
                (
                    "MI",
                    self._image(landmark_fixed, fixed_spacing),
                    self._image(landmark_moving, moving_spacing),
                    float(settings.landmark_weight),
                    32,
                )
            ]

        if progress is not None:
            progress(f"running {settings.transform_type} registration…")
        outcome = ants.registration(**kwargs)

        forward = [str(path) for path in outcome.get("fwdtransforms", [])]
        inverse = [str(path) for path in outcome.get("invtransforms", [])]

        metrics: dict[str, float] = {}
        try:
            warped = outcome.get("warpedmovout")
            metrics["mutual_information_before"] = float(
                ants.image_mutual_information(fixed_image, moving_image)
            )
            if warped is not None:
                metrics["mutual_information_after"] = float(
                    ants.image_mutual_information(fixed_image, warped)
                )
        except Exception:
            # Purely diagnostic; never fail a good registration over a QC number.
            logger.debug("could not compute mutual information", exc_info=True)

        return forward, inverse, metrics

    def apply(
        self,
        moving,
        moving_spacing,
        reference,
        reference_spacing,
        transforms,
        interpolation="linear",
    ):
        import ants

        resampled = ants.apply_transforms(
            fixed=self._image(reference, reference_spacing),
            moving=self._image(moving, moving_spacing),
            transformlist=list(transforms),
            interpolator=interpolation,
        )
        return np.asarray(resampled.numpy())


_BACKENDS: dict[str, Backend] = {AntsBackend.name: AntsBackend()}


def register_backend(backend: Backend) -> None:
    """Add a backend to the registry. The hook ``itk-elastix`` would use."""
    _BACKENDS[backend.name] = backend


def get_backend(name: str = "ants") -> Backend:
    backend = _BACKENDS.get(str(name).lower())
    if backend is None:
        known = ", ".join(sorted(_BACKENDS)) or "none"
        raise ValueError(f"unknown registration backend {name!r} (known: {known})")
    return backend


def available_backends() -> list[str]:
    """Backends whose library is actually importable right now."""
    return sorted(name for name, backend in _BACKENDS.items() if backend.available())


def backend_available(name: str = "ants") -> bool:
    try:
        return get_backend(name).available()
    except ValueError:
        return False


def missing_backend_message(name: str = "ants") -> str | None:
    """The sentence to show when the backend is not installed, or ``None``.

    Mirrors how the slide export reports a missing ``python-pptx``: the panel
    still builds, it just says plainly what to install.
    """
    try:
        backend = get_backend(name)
    except ValueError as exc:
        return str(exc)
    return None if backend.available() else backend.install_hint


# ---------------------------------------------------------------------------
# Reading atlas files
# ---------------------------------------------------------------------------


def _suffixes(path: Path) -> str:
    """Lower-case suffix, treating ``.nii.gz`` as one."""
    name = path.name.lower()
    return ".nii.gz" if name.endswith(".nii.gz") else path.suffix.lower()


def to_canonical(
    array: np.ndarray, spacing: Sequence[float] | None, order: str
) -> tuple[np.ndarray, tuple[float, float, float] | None]:
    """Permute a volume and its spacing from *order* into ``(z, y, x)``.

    The one place axis conventions are reconciled. Everything downstream — the
    metric, the voxel sizes, the region masks, the scale on the layers that come
    back — assumes ``(z, y, x)``, so a file that disagrees is turned around here
    and nowhere else.
    """
    order = (order or CANONICAL_ORDER).lower()
    if sorted(order) != sorted(CANONICAL_ORDER):
        raise ValueError(f"axis order must be a permutation of {CANONICAL_ORDER!r}, got {order!r}")
    if array.ndim != 3:
        return array, tuple(spacing) if spacing is not None else None  # type: ignore[return-value]

    permutation = tuple(order.index(axis) for axis in CANONICAL_ORDER)
    if permutation == (0, 1, 2):
        return array, tuple(float(v) for v in spacing) if spacing is not None else None  # type: ignore[return-value]

    permuted = np.transpose(array, permutation)
    if spacing is None:
        return permuted, None
    return permuted, tuple(float(spacing[index]) for index in permutation)  # type: ignore[return-value]


def _is_matlab_hdf5(handle) -> bool:
    """Whether an HDF5 file was written by MATLAB, which reverses its axes.

    MATLAB v7.3 files carry a ``#refs#`` group and ``MATLAB_class`` attributes.
    Z-Brain's stacks are stored 1406 × 621 × 138 in MATLAB and therefore read
    back as 138 × 621 × 1406 — z, x, y rather than z, y, x, which is a
    transposition no optimiser is going to find on its own.
    """
    try:
        if "#refs#" in handle:
            return True
        for node in handle.values():
            if "MATLAB_class" in getattr(node, "attrs", {}):
                return True
    except Exception:
        logger.debug("could not test for a MATLAB HDF5 signature", exc_info=True)
    return False


def read_volume(
    path: str | Path, dataset: str | None = None, axis_order: str = ""
) -> tuple[np.ndarray, tuple[float, float, float] | None]:
    """Read an atlas volume as ``(array in (z, y, x), voxel size in µm or None)``.

    Each format is read by whatever library is already a dependency where one
    exists — tifffile and h5py are core here — falling back to the backend's own
    reader for the neuroimaging formats, which is why ``.nrrd`` works without
    ``pynrrd`` when antspyx is installed. *dataset* names the volume to take when
    the file holds several.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"atlas file not found: {path}")
    suffix = _suffixes(path)

    if suffix in (".tif", ".tiff"):
        import tifffile

        with tifffile.TiffFile(str(path)) as handle:
            array = handle.asarray()
            spacing = _tiff_spacing(handle)
        return to_canonical(np.asarray(array), spacing, axis_order or CANONICAL_ORDER)

    if suffix in (".h5", ".hdf5", ".mat"):
        array, spacing, guessed = _read_hdf5_volume(path, dataset)
        return to_canonical(array, spacing, axis_order or guessed)

    if suffix == ".nrrd":
        try:
            import nrrd

            array, header = nrrd.read(str(path))
            # pynrrd hands back the array in the file's own axis order, and the
            # spacing helper already returns (z, y, x) for it.
            array, _spacing = to_canonical(np.asarray(array), None, axis_order or ITK_ORDER)
            return array, _nrrd_spacing(header)
        except ImportError:
            logger.debug("pynrrd is not installed; falling back to the backend reader")

    array, spacing = _read_with_backend(path)
    return to_canonical(array, spacing, axis_order or ITK_ORDER)


def read_voxel_size(path: str | Path) -> tuple[float, float, float] | None:
    """The voxel size of an atlas file, from its header alone.

    Reading the header rather than the volume: the panel wants to show the voxel
    size the moment a file is picked, and an atlas reference is hundreds of
    megabytes. Returns ``None`` when the format carries no spacing — plain TIFF
    and HDF5 usually do not, which is why the panel lets it be typed in.
    """
    path = Path(path)
    if not path.exists():
        return None
    suffix = _suffixes(path)

    try:
        if suffix in (".tif", ".tiff"):
            import tifffile

            with tifffile.TiffFile(str(path)) as handle:
                return _tiff_spacing(handle)

        if suffix == ".nrrd":
            try:
                import nrrd

                return _nrrd_spacing(nrrd.read_header(str(path)))
            except ImportError:
                pass

        if suffix in (".nii", ".nii.gz"):
            try:
                import nibabel

                zooms = tuple(float(v) for v in nibabel.load(str(path)).header.get_zooms()[:3])
                return (zooms[2], zooms[1], zooms[0]) if len(zooms) == 3 else None
            except ImportError:
                pass

        if suffix not in (".h5", ".hdf5", ".mat"):
            import ants

            info = ants.image_header_info(str(path))
            spacing = tuple(float(v) for v in info.get("spacing", ()))
            if len(spacing) == 3:
                # ITK reports spacing fastest-axis-first; ours is (z, y, x).
                return (spacing[2], spacing[1], spacing[0])
    except Exception:
        logger.debug("could not read the voxel size of %s from its header", path, exc_info=True)
    return None


def _read_with_backend(path: Path) -> tuple[np.ndarray, tuple[float, float, float] | None]:
    """Last resort: let ANTs (or nibabel) read a format we have no reader for."""
    try:
        import ants

        image = ants.image_read(str(path))
        spacing = tuple(float(v) for v in image.spacing)
        return np.asarray(image.numpy()), spacing if len(spacing) == 3 else None
    except Exception:
        logger.debug("ants could not read %s", path, exc_info=True)

    try:
        import nibabel

        image = nibabel.load(str(path))
        zooms = tuple(float(v) for v in image.header.get_zooms()[:3])
        return np.asarray(image.get_fdata()), zooms if len(zooms) == 3 else None
    except Exception as exc:
        raise ValueError(
            f"no reader available for {path.name}. Install antspyx, pynrrd or nibabel, "
            "or convert the atlas to TIFF."
        ) from exc


def _tiff_spacing(handle) -> tuple[float, float, float] | None:
    """Voxel size from an ImageJ TIFF's own metadata, when it recorded one."""
    try:
        meta = getattr(handle, "imagej_metadata", None) or {}
        page = handle.pages[0]
        tags = page.tags
        z = float(meta.get("spacing", 0.0)) or 0.0
        unit_scale = 1.0
        if str(meta.get("unit", "")).lower() in ("micron", "um", "µm", "micrometer"):
            unit_scale = 1.0
        x = y = 0.0
        for tag_name, target in (("XResolution", "x"), ("YResolution", "y")):
            tag = tags.get(tag_name)
            if tag is None:
                continue
            value = tag.value
            pixels_per_unit = float(value[0]) / float(value[1]) if isinstance(value, tuple) else float(value)
            if pixels_per_unit > 0:
                if target == "x":
                    x = 1.0 / pixels_per_unit
                else:
                    y = 1.0 / pixels_per_unit
        if x > 0 and y > 0 and z > 0:
            return (z * unit_scale, y * unit_scale, x * unit_scale)
    except Exception:
        logger.debug("could not read TIFF spacing", exc_info=True)
    return None


def _nrrd_spacing(header: Mapping[str, Any]) -> tuple[float, float, float] | None:
    """Voxel size from an NRRD header, from either of the two ways it is stored."""
    try:
        directions = header.get("space directions")
        if directions is not None:
            sizes = [
                float(np.linalg.norm(np.asarray(row, dtype=float)))
                for row in directions
                if row is not None and np.asarray(row).dtype != object
            ]
            if len(sizes) >= 3:
                return (sizes[2], sizes[1], sizes[0])
        spacings = header.get("spacings")
        if spacings is not None and len(spacings) >= 3:
            return (float(spacings[2]), float(spacings[1]), float(spacings[0]))
    except Exception:
        logger.debug("could not read NRRD spacing", exc_info=True)
    return None


def _hdf5_volume_keys(handle) -> list[str]:
    """Every 3D dataset in an HDF5 file, by path."""
    found: list[str] = []

    def visit(name, node) -> None:
        if getattr(node, "shape", None) is not None and len(node.shape) == 3:
            found.append(name)

    handle.visititems(visit)
    return sorted(found)


def _read_hdf5_volume(
    path: Path, dataset: str | None = None
) -> tuple[np.ndarray, tuple[float, float, float] | None, str]:
    """One volume out of an HDF5 file, with the axis order it is stored in."""
    import h5py

    with h5py.File(str(path), "r") as handle:
        order = MATLAB_HDF5_ORDER if _is_matlab_hdf5(handle) else CANONICAL_ORDER
        keys = _hdf5_volume_keys(handle)
        if not keys:
            raise ValueError(f"{path.name} holds no 3D dataset")
        if dataset:
            if dataset not in handle:
                available = ", ".join(keys[:6])
                raise ValueError(f"{path.name} has no dataset {dataset!r} (has: {available}…)")
            chosen = dataset
        else:
            chosen = keys[0]
            if len(keys) > 1:
                logger.info("%s holds %d volumes; using %s", path.name, len(keys), chosen)
        return np.asarray(handle[chosen][()]), None, order


def hdf5_volume_names(path: str | Path) -> list[str]:
    """Every 3D dataset in an HDF5 file, for the panel to offer as a reference."""
    path = Path(path)
    if not path.exists() or _suffixes(path) not in (".h5", ".hdf5", ".mat"):
        return []
    try:
        import h5py

        with h5py.File(str(path), "r") as handle:
            return [key for key in _hdf5_volume_keys(handle) if not key.startswith("#refs#")]
    except Exception:
        logger.debug("could not list the datasets in %s", path, exc_info=True)
        return []


# ---------------------------------------------------------------------------
# Region labels
# ---------------------------------------------------------------------------


@dataclass
class LabelSet:
    """The atlas's regions, in whichever of the three forms downloads use.

    Z-Brain ships its 294 regions as one sparse logical matrix in a MATLAB v7.3
    file, one column per region over a flattened 1406 × 621 × 138 grid. Its
    regions overlap — a voxel belongs to a major subdivision *and* to a neuropil
    inside it — so they can never be collapsed into a single integer volume.
    Other atlases ship exactly such an integer volume, or one binary mask per
    region as separate datasets. All three are supported.

    Nothing is held in memory across regions. A single Z-Brain mask is 120
    million voxels; the sparse form is measured through its own indices without
    ever building the mask at all.
    """

    names: dict[int, str] = field(default_factory=dict)
    #: Integer label volume, when the atlas is in that form.
    volume: np.ndarray | None = None
    #: Source file and dataset keys, when it is one binary mask per dataset.
    path: Path | None = None
    mask_keys: list[str] = field(default_factory=list)
    shape: tuple[int, ...] = ()
    #: Dataset holding a MATLAB sparse matrix, one column per region.
    sparse_key: str = ""
    #: Axis order the mask indices are expressed in, before canonicalisation.
    axis_order: str = CANONICAL_ORDER

    def __post_init__(self) -> None:
        # An integer label volume knows its own grid. Leaving ``shape`` empty
        # would let a caller build a set whose grid silently reads as unknown,
        # and the grid is what every alignment check keys on.
        if not self.shape and self.volume is not None:
            self.shape = tuple(int(n) for n in np.asarray(self.volume).shape)

    @property
    def overlapping(self) -> bool:
        """True when regions may overlap, i.e. anything but an integer volume."""
        return self.volume is None

    def __len__(self) -> int:
        if self.volume is not None:
            return len(self.names)
        if self.sparse_key:
            return len(self.names)
        return len(self.mask_keys)

    def iter_regions(self) -> Iterator[tuple[int, str, np.ndarray]]:
        """Yield ``(id, name, boolean mask)`` one region at a time.

        For the sparse form this materialises each mask, which costs a full
        volume of ``bool`` per region — :meth:`values_in` is the way to measure
        one without paying that.
        """
        if self.volume is not None:
            for region_id in sorted(int(v) for v in np.unique(self.volume) if int(v) != 0):
                name = self.names.get(region_id, f"Region {region_id}")
                yield region_id, name, self.volume == region_id
            return

        if self.sparse_key:
            for region_id, name, indices in self._iter_sparse_indices():
                mask = np.zeros(math.prod(int(n) for n in self.shape), dtype=bool)
                mask[indices] = True
                yield region_id, name, mask.reshape(self.shape)
            return

        if self.path is None:
            return
        import h5py

        with h5py.File(str(self.path), "r") as handle:
            for index, key in enumerate(self.mask_keys, start=1):
                mask, _spacing = to_canonical(np.asarray(handle[key][()]), None, self.axis_order)
                yield index, self.names.get(index, key), mask > 0

    def _iter_sparse_indices(self) -> Iterator[tuple[int, str, np.ndarray]]:
        """Yield ``(id, name, flat indices)`` into a canonical ``(z, y, x)`` volume.

        The stored indices run down a MATLAB column-major ``(y, x, z)`` grid, so
        they are turned into subscripts and re-flattened for the canonical array
        rather than the volume being rebuilt and transposed.
        """
        import h5py

        depth, height, width = self.shape  # canonical (z, y, x)
        with h5py.File(str(self.path), "r") as handle:
            group = handle[self.sparse_key]
            row_index = group["ir"]
            column_start = np.asarray(group["jc"][()]).astype(np.int64)
            for column in range(len(column_start) - 1):
                start, stop = int(column_start[column]), int(column_start[column + 1])
                region_id = column + 1
                name = self.names.get(region_id, f"Region {region_id}")
                if stop <= start:
                    yield region_id, name, np.empty(0, dtype=np.int64)
                    continue
                linear = np.asarray(row_index[start:stop]).astype(np.int64)
                # MATLAB stores (y, x, z) column-major: y varies fastest.
                y = linear % height
                x = (linear // height) % width
                z = linear // (height * width)
                yield region_id, name, (z * height + y) * width + x

    def values_in(self, signal: np.ndarray) -> Iterator[tuple[int, str, np.ndarray]]:
        """Yield ``(id, name, the signal's values inside that region)``.

        The sparse form indexes the flattened signal directly, so measuring 294
        Z-Brain regions never allocates a mask.
        """
        values = np.asarray(signal)
        if self.sparse_key:
            flat = values.reshape(-1)
            for region_id, name, indices in self._iter_sparse_indices():
                yield region_id, name, flat[indices]
            return
        for region_id, name, mask in self.iter_regions():
            yield region_id, name, values[np.asarray(mask)]


def _read_matlab_sparse_labels(path: Path, names_path: str | Path | None) -> LabelSet | None:
    """Z-Brain's ``MaskDatabase.mat``: 294 regions as one sparse logical matrix.

    Returns ``None`` when the file holds no such matrix, so the caller can fall
    through to the other forms.
    """
    import h5py

    with h5py.File(str(path), "r") as handle:
        key = next(
            (
                name
                for name in handle
                if isinstance(handle[name], h5py.Group)
                and {"data", "ir", "jc"} <= set(handle[name])
                and "outline" not in str(name).lower()
            ),
            None,
        )
        if key is None:
            return None

        # MATLAB records the grid the columns were flattened from beside them.
        def scalar(name: str) -> int | None:
            node = handle.get(name)
            return int(np.asarray(node[()]).ravel()[0]) if node is not None else None

        height, width, depth = scalar("height"), scalar("width"), scalar("Zs")
        rows = int(handle[key].attrs.get("MATLAB_sparse", 0))
        if not (height and width and depth):
            raise ValueError(
                f"{path.name} holds a sparse mask matrix but no height/width/Zs to reshape it by"
            )
        if rows and rows != height * width * depth:
            raise ValueError(
                f"{path.name}: the sparse matrix has {rows} rows but height x width x Zs is "
                f"{height * width * depth}"
            )

        names = read_label_names(names_path) or _read_matlab_names(handle)
        count = int(np.asarray(handle[key]["jc"][()]).size - 1)
        names = names or {index: f"Region {index}" for index in range(1, count + 1)}
        logger.info("%s: %d sparse regions on a %d x %d x %d grid", path.name, count, depth, height, width)
        return LabelSet(
            names=names,
            path=path,
            sparse_key=key,
            # Canonical (z, y, x); the indices are converted in _iter_sparse_indices.
            shape=(depth, height, width),
            axis_order=MATLAB_LINEAR_ORDER,
        )


def _read_matlab_names(handle) -> dict[int, str]:
    """Region names from a MATLAB cell array of char arrays."""
    node = next((handle[key] for key in handle if "name" in str(key).lower()), None)
    if node is None or not hasattr(node, "shape"):
        return {}
    names: dict[int, str] = {}
    try:
        references = np.asarray(node[()]).ravel()
        for index, reference in enumerate(references, start=1):
            try:
                codes = np.asarray(handle[reference][()]).ravel()
                names[index] = "".join(chr(int(code)) for code in codes).strip()
            except Exception:
                continue
    except Exception:
        logger.debug("could not read MATLAB region names", exc_info=True)
    return names


def read_labels(
    path: str | Path, names_path: str | Path | None = None, max_masks: int | None = None
) -> LabelSet:
    """Load an atlas's regions, in whichever form the download ships them."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"atlas label file not found: {path}")

    if _suffixes(path) in (".h5", ".hdf5", ".mat"):
        import h5py

        sparse = _read_matlab_sparse_labels(path, names_path)
        if sparse is not None:
            return sparse

        with h5py.File(str(path), "r") as handle:
            order = MATLAB_HDF5_ORDER if _is_matlab_hdf5(handle) else CANONICAL_ORDER
            keys = _hdf5_volume_keys(handle)
            if not keys:
                raise ValueError(f"{path.name} holds no 3D dataset")
            stored = tuple(int(n) for n in handle[keys[0]].shape)
            first = np.asarray(handle[keys[0]][: min(4, stored[0])])

        distinct = int(np.unique(first).size)
        # One dataset that is not binary is an integer label volume that happens
        # to live in HDF5; many datasets is the one-mask-per-region form.
        if len(keys) == 1 and distinct > 2:
            volume, _spacing, guessed = _read_hdf5_volume(path)
            volume, _ = to_canonical(volume, None, guessed)
            return LabelSet(
                names=read_label_names(names_path), volume=volume, path=path, shape=volume.shape
            )
        if len(keys) > 1 and distinct > 2:
            # Z-Brain's AnatomyLabelDatabase.hdf5 is 29 averaged *intensity*
            # stacks, not masks. Measured as masks every voxel counts as inside
            # every region, so this has to be refused rather than averaged over.
            raise ValueError(
                f"{path.name} holds {len(keys)} volumes whose values are not 0/1, so they are "
                f"images rather than region masks (for example “{keys[0]}”). If this is Z-Brain, "
                "the region masks are in MaskDatabase.mat and this file is the anatomy stacks — "
                "one of which makes a good registration reference."
            )
        if max_masks is not None:
            keys = keys[:max_masks]
        names = read_label_names(names_path) or {
            index: _tidy_region_name(key) for index, key in enumerate(keys, start=1)
        }
        shape, _spacing = to_canonical(np.empty(stored, dtype=bool), None, order)
        return LabelSet(
            names=names, path=path, mask_keys=keys, shape=shape.shape, axis_order=order
        )

    volume, _spacing = read_volume(path)
    volume = np.asarray(volume)
    if volume.dtype.kind == "f":
        volume = np.rint(volume).astype(np.int32)
    return LabelSet(names=read_label_names(names_path), volume=volume, path=path, shape=volume.shape)


def _tidy_region_name(key: str) -> str:
    """A readable region name from an HDF5 dataset path."""
    name = str(key).rsplit("/", 1)[-1]
    name = re.sub(r"^(Anatomy|Mask|Label)[_\- ]*", "", name, flags=re.IGNORECASE)
    return name.replace("_", " ").strip() or str(key)


def read_label_names(path: str | Path | None) -> dict[int, str]:
    """Region names from a ``.csv``/``.txt`` sidecar, keyed by region id.

    Accepts the two shapes these files come in: ``id,name`` per line, or one name
    per line in region order.
    """
    if path is None:
        return {}
    path = Path(path)
    if not path.exists():
        logger.warning("region name file not found: %s", path)
        return {}

    names: dict[int, str] = {}
    order = 0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        parts = [part.strip() for part in re.split(r"[,;\t]", text) if part.strip()]
        if len(parts) >= 2 and parts[0].lstrip("-").isdigit():
            names[int(parts[0])] = parts[1]
            continue
        order += 1
        if order == 1 and parts and parts[0].lower() in ("id", "index", "label"):
            continue  # a header row
        names[order] = parts[0] if parts else text
    return names


# ---------------------------------------------------------------------------
# Finding the atlas in a download
# ---------------------------------------------------------------------------


def _score_reference(name: str) -> tuple[int, str]:
    """How good a filename looks as a registration reference. Higher is better."""
    lowered = name.lower()
    if any(keyword in lowered for keyword in LABEL_KEYWORDS):
        return (-1, "")  # region masks are not a reference
    for keyword in NUCLEAR_KEYWORDS:
        if keyword in lowered:
            return (2, keyword)
    for keyword in FALLBACK_KEYWORDS:
        if keyword in lowered:
            return (1, keyword)
    return (0, "")


def _pick_labels(candidates: Sequence[Path], reference: Path, dataset: str) -> Path | None:
    """The region-mask file, preferring one that says "mask" over one that says "label".

    Z-Brain has both: ``MaskDatabase.mat`` holds the 294 regions, and
    ``AnatomyLabelDatabase.hdf5`` holds averaged anatomy stacks despite having
    "label" in its name. Picking on the first keyword match gets that backwards.
    """
    ranked: list[tuple[int, Path]] = []
    for path in candidates:
        lowered = path.name.lower()
        # Only skip the reference file when the whole file is the reference; a
        # dataset inside it may sit beside masks in the same file.
        if path == reference and not dataset:
            continue
        if "mask" in lowered:
            ranked.append((0, path))
        elif any(word in lowered for word in LABEL_KEYWORDS):
            ranked.append((1, path))
    if not ranked:
        return None
    ranked.sort(key=lambda item: (item[0], str(item[1])))
    return ranked[0][1]


def discover_atlas(root: str | Path) -> AtlasSpec:
    """Work out which files in an atlas download to register against.

    A nuclear reference is preferred and taken whenever one is present, because
    DAPI against a nuclear channel is the same-modality case. Falling back to the
    pan-neuronal reference (tERK and friends) is allowed but recorded on the spec
    and logged, since a cross-modality fit is the one that needs looking at
    before its output is believed.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"atlas folder not found: {root}")
    if root.is_file():
        return AtlasSpec(reference_path=root, reference_channel="(given)", is_nuclear=True)

    candidates = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and _suffixes(path) in LABEL_SUFFIXES
    )
    if not candidates:
        raise ValueError(f"no atlas volumes found under {root} (looked for {', '.join(LABEL_SUFFIXES)})")

    best: tuple[int, str, Path, str] | None = None
    for path in candidates:
        score, keyword = _score_reference(path.name)
        if score >= 0 and (best is None or score > best[0]):
            best = (score, keyword, path, "")

        # A multi-volume file has to be looked inside: Z-Brain's nuclear
        # reference is one of 29 datasets in a file whose own name says nothing
        # about it, and registering DAPI to a nuclear average rather than to
        # tERK is the difference between a same-modality fit and a hard one.
        for name in hdf5_volume_names(path):
            inner_score, inner_keyword = _score_reference(name)
            if inner_score >= 2 and (best is None or inner_score > best[0]):
                best = (inner_score, inner_keyword, path, name)

    if best is None:
        raise ValueError(
            f"every volume under {root} looks like region masks; none is usable as a reference"
        )

    score, keyword, reference, dataset = best
    labels = _pick_labels(candidates, reference, dataset)
    names_file = next(
        (
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in (".csv", ".txt")
            and any(word in path.name.lower() for word in ("name", "label", "region", "anatomy"))
        ),
        None,
    )

    spec = AtlasSpec(
        reference_path=reference,
        reference_channel=keyword or "unknown",
        label_path=labels,
        label_names_path=names_file,
        is_nuclear=score >= 2,
        reference_dataset=dataset,
    )
    if spec.is_nuclear:
        logger.info("atlas reference: %s (nuclear, matched %r)", spec.describe(), keyword)
    else:
        logger.warning(
            "no nuclear reference found under %s; falling back to %s. DAPI will be registered "
            "across modalities — check the result before trusting it.",
            root, reference.name,
        )
    return spec


# ---------------------------------------------------------------------------
# Preparing volumes
# ---------------------------------------------------------------------------


def winsorize(array: np.ndarray, limits: tuple[float, float] = (0.5, 99.5)) -> np.ndarray:
    """Clip to percentiles and rescale to 0-1.

    Registration metrics care about the shape of the intensity histogram. A few
    saturated nuclei, or the camera offset, otherwise take up most of the range
    and leave the actual structure squeezed into a handful of levels.
    """
    values = np.asarray(array, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros(values.shape, dtype=np.float32)
    low, high = (float(v) for v in np.percentile(finite, list(limits)))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low, high = float(finite.min()), float(finite.max())
    if high <= low:
        return np.zeros(values.shape, dtype=np.float32)
    return np.clip((values - low) / (high - low), 0.0, 1.0).astype(np.float32)


def decimate(
    array: np.ndarray, spacing: tuple[float, float, float], max_voxels: int
) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Block-average down to at most *max_voxels*, adjusting the voxel size to match.

    Averaging over whole blocks rather than striding: striding drops thin
    structures outright, and thin structures are most of what a brain is. The
    deformation being fitted is smooth at this scale, so the transform found on
    the decimated volume applies unchanged at full resolution.
    """
    values = np.asarray(array)
    if max_voxels <= 0 or values.size <= max_voxels or values.ndim != 3:
        return values, tuple(float(v) for v in spacing)  # type: ignore[return-value]

    factor = int(np.ceil((values.size / float(max_voxels)) ** (1.0 / 3.0)))
    factor = max(1, factor)
    if factor == 1:
        return values, tuple(float(v) for v in spacing)  # type: ignore[return-value]

    trimmed = values[
        : values.shape[0] // factor * factor,
        : values.shape[1] // factor * factor,
        : values.shape[2] // factor * factor,
    ]
    if 0 in trimmed.shape:
        # An axis shorter than the factor — usually a thin z-stack. Leave it be
        # rather than reducing it to nothing.
        return values, tuple(float(v) for v in spacing)  # type: ignore[return-value]

    depth, height, width = (n // factor for n in trimmed.shape)
    reduced = np.empty((depth, height, width), dtype=np.float32)
    # One output plane at a time. Reshaping the whole stack and averaging in one
    # call is the obvious way to write this, and on a 134 x 2040 x 2040 stack it
    # asks numpy for several gigabytes of float32 at once; this asks for one
    # slab, which is tens of megabytes.
    for index in range(depth):
        slab = np.asarray(trimmed[index * factor : (index + 1) * factor], dtype=np.float32)
        reduced[index] = slab.reshape(factor, height, factor, width, factor).mean(axis=(0, 2, 4))
    new_spacing = tuple(float(v) * factor for v in spacing)
    logger.info(
        "decimated %s to %s by %dx for registration", values.shape, reduced.shape, factor
    )
    return reduced, new_spacing  # type: ignore[return-value]


def _flip(array: np.ndarray, axes: Sequence[int]) -> np.ndarray:
    valid = tuple(int(axis) for axis in axes if 0 <= int(axis) < array.ndim)
    return np.flip(array, axis=valid).copy() if valid else array


def prepare(
    volume: Volume, settings: RegistrationSettings
) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Flip, winsorise and decimate one volume ready for the metric."""
    values = _flip(volume.array(), settings.flip_axes)
    values, spacing = decimate(values, volume.spacing, settings.max_voxels)
    return winsorize(values, settings.winsorize), spacing


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


def register_to_atlas(
    driver: Volume,
    atlas: AtlasSpec,
    carry: Sequence[Volume] = (),
    landmark: Volume | None = None,
    settings: RegistrationSettings | None = None,
    progress: Callable[[str], None] | None = None,
) -> RegistrationResult:
    """Fit the driver channel onto the atlas and carry the others through.

    Only *driver* is ever passed to the metric — *carry* is resampled with the
    transform that comes out, and *landmark* contributes only if
    :attr:`RegistrationSettings.use_landmark_metric` says so. That asymmetry is
    the whole point: a regional channel like anti-SV2 given to the optimiser will
    produce a beautiful, meaningless alignment of its own domains onto whatever
    is nearby in the atlas.

    Safe to run off the main thread; it holds no Qt or napari objects.
    """
    settings = settings or RegistrationSettings()
    started = time.time()
    result = RegistrationResult(backend=settings.backend, transform_type=settings.transform_type)

    if driver.role != ROLE_DRIVER:
        result.warnings.append(
            f"{driver.name} is assigned as “{driver.role}” but is being used to drive the fit."
        )
    if not atlas.is_nuclear:
        result.warnings.append(
            f"The atlas reference {atlas.reference_path.name} is not a nuclear channel "
            f"({atlas.reference_channel or 'unknown'}), so DAPI is being registered across "
            "modalities. Check the overlay before using the region table."
        )

    backend = get_backend(settings.backend)
    if not backend.available():
        raise RuntimeError(missing_backend_message(settings.backend) or "backend unavailable")

    if progress is not None:
        progress(f"reading the atlas reference ({atlas.reference_path.name})…")
    reference, reference_spacing = read_volume(
        atlas.reference_path, atlas.reference_dataset or None, atlas.axis_order
    )

    # An HDF5 of stacks carries no clue about its axis order, and a reference
    # left transposed asks the optimiser to find a 90° rotation — which it does
    # not, it just converges somewhere wrong. The masks state their grid, so when
    # they are present they settle it.
    if not atlas.axis_order and atlas.label_path is not None:
        grid = label_grid_shape(atlas.label_path)
        reference, reference_spacing, permuted = align_to_grid(reference, reference_spacing, grid)
        if permuted:
            logger.info(
                "transposed %s onto the %s grid the region masks are defined on",
                atlas.reference_path.name, grid,
            )
            result.warnings.append(
                f"{atlas.reference_path.name} was stored in a different axis order from "
                f"{Path(atlas.label_path).name}; it was transposed onto the masks' {grid} grid."
            )

    if reference_spacing is None:
        reference_spacing = atlas.voxel_size_um or (1.0, 1.0, 1.0)
        size = f"{reference_spacing[0]:g} × {reference_spacing[1]:g} × {reference_spacing[2]:g} µm"
        if atlas.voxel_size_um is None:
            # Nothing to go on at all. A wrong reference voxel size scales the
            # whole fit, so this is the warning that matters most on this panel.
            result.warnings.append(
                f"{atlas.reference_path.name} carries no voxel size and none was given; "
                f"assuming {size}. If that is wrong, every distance in the result is wrong."
            )
        else:
            result.warnings.append(
                f"{atlas.reference_path.name} carries no voxel size; using the {size} "
                "given for the atlas."
            )
    result.atlas_voxel_size_um = tuple(float(v) for v in reference_spacing)  # type: ignore[assignment]

    fixed = winsorize(np.asarray(reference), settings.winsorize)
    fixed_small, fixed_spacing = decimate(fixed, result.atlas_voxel_size_um, settings.max_voxels)

    if progress is not None:
        progress(f"preparing {driver.name}…")
    moving, moving_spacing = prepare(driver, settings)

    extra: tuple[np.ndarray, np.ndarray] | None = None
    if settings.use_landmark_metric:
        if landmark is None:
            result.warnings.append(
                "A landmark metric was requested but no landmark channel was assigned; "
                "registering on the driver alone."
            )
        else:
            # The landmark term needs a counterpart in the atlas. There is only
            # one reference volume, so the same fixed image is reused: the term
            # then rewards the landmark channel matching overall brain shape,
            # which is the most it can honestly contribute.
            landmark_moving, _spacing = prepare(landmark, settings)
            extra = (fixed_small, landmark_moving)
            result.warnings.append(
                f"{landmark.name} is contributing to the metric at weight "
                f"{settings.landmark_weight:g}; the fit is no longer driven by "
                f"{driver.name} alone."
            )

    logger.info(
        "registering %s (%s) onto %s with %s via %s",
        driver.name, "x".join(str(n) for n in moving.shape), atlas.describe(),
        settings.transform_type, settings.backend,
    )
    forward, inverse, metrics = backend.register(
        fixed_small, fixed_spacing, moving, moving_spacing, settings, extra, progress
    )
    result.forward_transforms = forward
    result.inverse_transforms = inverse
    result.metrics = metrics
    if not forward:
        raise RuntimeError("the backend returned no transform; registration failed")

    # Everything is resampled onto the full-resolution atlas grid, not the
    # decimated one the fit ran on: the transform is independent of the grid it
    # was found on, and the region masks are at atlas resolution.
    if progress is not None:
        progress(f"resampling {driver.name}…")
    result.warped[driver.name] = backend.apply(
        _flip(driver.array(), settings.flip_axes), driver.spacing,
        np.asarray(reference), result.atlas_voxel_size_um, forward, "linear",
    )

    for volume in carry:
        if progress is not None:
            progress(f"resampling {volume.name}…")
        if volume.role == ROLE_DRIVER:
            result.warnings.append(
                f"{volume.name} is marked as a driver but was passed as a carry-along; "
                "it was resampled, not fitted."
            )
        result.warped[volume.name] = backend.apply(
            _flip(volume.array(), settings.flip_axes), volume.spacing,
            np.asarray(reference), result.atlas_voxel_size_um, forward, "linear",
        )

    result.elapsed_s = time.time() - started
    logger.info(
        "registration finished in %.1f s; warped %d channel(s); MI %s -> %s",
        result.elapsed_s, len(result.warped),
        _format_metric(metrics.get("mutual_information_before")),
        _format_metric(metrics.get("mutual_information_after")),
    )
    return result


def _format_metric(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def apply_transform(
    volume: Volume,
    result: RegistrationResult,
    atlas: AtlasSpec,
    interpolation: str = "linear",
    inverse: bool = False,
    settings: RegistrationSettings | None = None,
) -> np.ndarray:
    """Resample one volume through an existing registration.

    ``inverse=True`` goes the other way, atlas into fish space — which is how the
    region masks are brought back onto the original stack for QC without touching
    the data being measured. Label volumes must be resampled with
    ``interpolation="nearestNeighbor"``, or region boundaries get interpolated
    into values that name no region at all.
    """
    settings = settings or RegistrationSettings()
    backend = get_backend(settings.backend)
    if not backend.available():
        raise RuntimeError(missing_backend_message(settings.backend) or "backend unavailable")

    transforms = result.inverse_transforms if inverse else result.forward_transforms
    if not transforms:
        raise ValueError("this result carries no transform in that direction")

    reference, spacing = read_volume(
        atlas.reference_path, atlas.reference_dataset or None, atlas.axis_order
    )
    if spacing is None:
        spacing = atlas.voxel_size_um or result.atlas_voxel_size_um

    if inverse:
        # Going backwards, the fish stack is the grid being resampled onto.
        target = np.zeros(volume.array().shape, dtype=np.float32)
        return backend.apply(
            np.asarray(reference), tuple(float(v) for v in spacing),
            target, volume.spacing, transforms, interpolation,
        )
    return backend.apply(
        _flip(volume.array(), settings.flip_axes), volume.spacing,
        np.asarray(reference), tuple(float(v) for v in spacing), transforms, interpolation,
    )


# ---------------------------------------------------------------------------
# Region readout
# ---------------------------------------------------------------------------


def signal_threshold(
    signal: np.ndarray, percentile: float = DEFAULT_THRESHOLD_PERCENTILE
) -> float:
    """Intensity above which a voxel counts as signal, from the volume's own distribution."""
    values = np.asarray(signal, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0
    return float(np.percentile(finite, float(percentile)))


def label_grid_shape(path: str | Path) -> tuple[int, int, int] | None:
    """The canonical ``(z, y, x)`` grid a label file is defined on, read cheaply.

    Used to settle the axis order of the reference volume. Two files from the
    same atlas must describe the same grid, and the masks are the ones that say
    outright what theirs is — Z-Brain records height, width and Zs beside them —
    whereas an HDF5 of stacks says nothing at all and cannot always be told apart
    from a file written in a different order.
    """
    path = Path(path)
    if not path.exists():
        return None
    if _suffixes(path) not in (".h5", ".hdf5", ".mat"):
        return None
    try:
        import h5py

        with h5py.File(str(path), "r") as handle:
            height = handle.get("height")
            width = handle.get("width")
            depth = handle.get("Zs")
            if height is not None and width is not None and depth is not None:
                return (
                    int(np.asarray(depth[()]).ravel()[0]),
                    int(np.asarray(height[()]).ravel()[0]),
                    int(np.asarray(width[()]).ravel()[0]),
                )
            keys = _hdf5_volume_keys(handle)
            if not keys:
                return None
            order = MATLAB_HDF5_ORDER if _is_matlab_hdf5(handle) else CANONICAL_ORDER
            stored = tuple(int(n) for n in handle[keys[0]].shape)
            permutation = tuple(order.index(axis) for axis in CANONICAL_ORDER)
            return tuple(stored[index] for index in permutation)  # type: ignore[return-value]
    except Exception:
        logger.debug("could not read the label grid of %s", path, exc_info=True)
    return None


def align_to_grid(
    array: np.ndarray,
    spacing: Sequence[float] | None,
    target: Sequence[int] | None,
) -> tuple[np.ndarray, tuple[float, float, float] | None, bool]:
    """Transpose *array* onto *target*'s axis order when the two are permutations.

    Returns ``(array, spacing, permuted)``. Only unambiguous cases are touched:
    when two axes are the same length there is no way to tell which way round
    they belong, and guessing would be worse than leaving it alone.
    """
    spacing_out = tuple(float(v) for v in spacing) if spacing is not None else None
    if target is None or array.ndim != 3:
        return array, spacing_out, False  # type: ignore[return-value]
    target = tuple(int(n) for n in target)
    if tuple(array.shape) == target:
        return array, spacing_out, False  # type: ignore[return-value]
    if sorted(array.shape) != sorted(target) or len(set(target)) != len(target):
        return array, spacing_out, False  # type: ignore[return-value]

    permutation = tuple(array.shape.index(size) for size in target)
    permuted = np.transpose(array, permutation)
    if spacing_out is not None:
        spacing_out = tuple(spacing_out[index] for index in permutation)  # type: ignore[assignment]
    return permuted, spacing_out, True  # type: ignore[return-value]


def _match_label_grid(signal: np.ndarray, labels: LabelSet) -> np.ndarray:
    """Put *signal* on the same grid as the labels, or explain why it cannot be.

    A mismatch that is a pure permutation is an axis-order disagreement between
    two atlas files, and it is repaired here — with a warning, because it means
    one of the files declared its order wrongly and the fit itself may have been
    run against a transposed reference. Anything else is a real mismatch: the
    signal was never resampled onto the atlas grid, and measuring it would
    produce numbers for regions it was never aligned to.
    """
    target = tuple(int(n) for n in labels.shape)
    if not target or signal.shape == target:
        return signal

    if sorted(signal.shape) == sorted(target) and len(set(target)) == len(target):
        permutation = tuple(signal.shape.index(size) for size in target)
        logger.warning(
            "the warped signal is %s but the regions are %s; transposing by %s. "
            "The two atlas files disagree about axis order — check the overlay.",
            signal.shape, target, permutation,
        )
        return np.transpose(signal, permutation)

    raise ValueError(
        f"the regions are {target} but the warped signal is {signal.shape}; the signal was not "
        "resampled onto the atlas grid. The region masks and the reference volume have to come "
        "from the same atlas and be on the same grid."
    )


def region_table(
    signal: np.ndarray,
    labels: LabelSet,
    voxel_volume_um3: float = 1.0,
    threshold: float | None = None,
) -> list[RegionStat]:
    """Per-region statistics of *signal*, which must already be in atlas space.

    The signal is warped onto the atlas rather than the labels onto the fish, so
    the masks stay exactly as the atlas defines them — no interpolated boundary,
    no region that exists only because of resampling.
    """
    values = np.asarray(signal, dtype=np.float32)
    if threshold is None:
        threshold = signal_threshold(values)

    values = _match_label_grid(values, labels)

    stats: list[RegionStat] = []
    for region_id, name, inside in labels.values_in(values):
        inside = np.asarray(inside)
        inside = inside[np.isfinite(inside)]
        if inside.size == 0:
            continue
        stats.append(
            RegionStat(
                region_id=int(region_id),
                name=str(name),
                n_voxels=int(inside.size),
                volume_um3=float(inside.size) * float(voxel_volume_um3),
                mean=float(inside.mean()),
                median=float(np.median(inside)),
                std=float(inside.std()),
                integrated=float(inside.sum()),
                fraction_above=float(np.count_nonzero(inside > threshold)) / float(inside.size),
                threshold=float(threshold),
            )
        )

    logger.info("read %d region(s) out of the warped signal", len(stats))
    return stats


def region_dataframe(stats: Sequence[RegionStat]):
    """A :class:`pandas.DataFrame` of region statistics with friendly column names."""
    import pandas as pd

    if not stats:
        return pd.DataFrame(columns=[REGION_LABELS[c] for c in REGION_COLUMNS])
    frame = pd.DataFrame([stat.as_row() for stat in stats])
    ordered = [c for c in REGION_COLUMNS if c in frame.columns]
    ordered += [c for c in frame.columns if c not in ordered]
    return frame[ordered].rename(columns=REGION_LABELS)
