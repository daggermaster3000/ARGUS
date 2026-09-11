"""Segment cells and nuclei with Cellpose, on the GPU when there is one.

The shape of this module mirrors :mod:`microscopy_viewer.registration`: no Qt, no
napari, and the heavy library is imported lazily inside the backend, so the whole
thing is testable headless and ``cellpose`` — which drags in torch — stays off the
startup path.

Three things here are more than a thin wrapper around ``model.eval``:

* **The GPU is found and used, and said out loud.** Cellpose falls back to the CPU
  silently, and a 3D stack that takes tens of seconds on a CUDA card takes tens of
  minutes without one. :func:`compute_device` reports what was actually selected,
  and the panel shows it before anything is run rather than after.
* **Sizes are in µm, not pixels.** Cellpose's ``diameter`` is in XY pixels and its
  ``anisotropy`` is the Z/XY voxel ratio; both are derived here from the calibrated
  voxel size the reader already put on the layer, so the same 5 µm nucleus setting
  works on a 0.3 µm/px confocal stack and a 1.2 µm/px one.
* **Big volumes are decimated in XY and the masks put back on the full grid.** A
  558 Mvoxel stack does not fit any consumer card. The label map that comes back is
  always on the grid that went in, so it drops straight onto the source layer.

Cellpose 4 (``cpsam``, ``cpsam_v2``, the ``cpdino`` pair) and Cellpose 3 (``cyto3``,
``nuclei``, the rest of the zoo) have different call signatures — v4 dropped
``channels`` and rebuilt the model list. Both are supported, and
:func:`model_choices` reports what is really available: what the installed version
ships, models the user trained in the Cellpose GUI, and weights sitting in the
cellpose model folder — minus the ones the installed version cannot load.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from .utils import get_logger

logger = get_logger("segmentation")

# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

#: One pass over the volume, flows computed in 3D. Slowest, and the only mode that
#: understands an object present in some planes and absent in others.
MODE_3D = "3D"
#: Segment each plane in 2D, then stitch planes whose masks overlap into one label.
#: Much faster than 3D, and often better on an anisotropic stack where a nucleus is
#: only three or four planes tall.
MODE_STITCH = "2D + stitch"
#: Segment each plane independently; labels do not carry between planes. For
#: counting per plane, or checking a diameter setting quickly.
MODE_SLICES = "2D per plane"

MODES = (MODE_STITCH, MODE_3D, MODE_SLICES)

#: Cellpose 4's single generalist model, and the default when v4 is installed.
CPSAM_MODEL = "cpsam"
#: The v3 zoo, in the order worth trying them, used when the installed cellpose
#: does not publish its own list. ``nuclei`` is the one for a DAPI channel;
#: ``cyto3`` is the generalist for a membrane or cytoplasmic stain.
V3_MODELS = (
    "nuclei",
    "cyto3",
    "cyto2",
    "cyto",
    "tissuenet_cp3",
    "livecell_cp3",
    "yeast_PhC_cp3",
    "yeast_BF_cp3",
    "bact_phase_cp3",
    "bact_fluor_cp3",
    "deepbacs_cp3",
)

#: Weights files written by cellpose 3 and earlier for its own builtins. Cellpose 4
#: is a single architecture (SAM) and cannot load any of them, so on v4 they are
#: kept out of the list rather than offered and then failing halfway into a run.
#: Matched as whole names rather than as fragments, so a model of your own called
#: ``nuclei_finetuned`` is still offered — only the zoo files are filtered.
LEGACY_MODEL_PATTERN = re.compile(
    r"^(?:"
    r"cyto[0-9]*|nuclei|cp|cpx|tn[0-9]|lc[0-9]"          # the v3 builtin names
    r"|(?:tissuenet|livecell|yeast_[a-z]+|bact_[a-z]+|deepbacs)_cp[0-9]"
    r")(?:torch_[0-9]+)?$"                                # ... and their weights files
)

#: Size models, which sit beside the weights and are not segmentation models.
SIZE_MODEL_PREFIX = "size_"

#: Files in the cellpose model directory that are not weights.
NON_MODEL_SUFFIXES = (".txt", ".json", ".csv", ".log", ".yml", ".yaml", ".png")

#: Objects smaller than this many pixels are dropped by Cellpose itself.
DEFAULT_MIN_SIZE = 15

#: Voxels above which the volume is decimated in XY before segmentation. Sized for
#: a card with a few GB free; the result reports what it decided.
DEFAULT_MAX_VOXELS = 200_000_000

#: Columns of the per-object table, in display order.
OBJECT_COLUMNS = (
    "label",
    "n_voxels",
    "volume_um3",
    "equivalent_diameter_um",
    "centroid_z_um",
    "centroid_y_um",
    "centroid_x_um",
    "solidity",
    "circularity",
    "eccentricity",
    "extent",
    "mean",
    "median",
    "std",
    "max",
    "integrated",
)

#: Shape descriptors read from scikit-image, and what they are for.
#:
#: ``solidity`` is the object's area divided by the area of its convex hull: 1.0
#: for anything convex, lower the more ragged or dented the outline. It is what
#: :attr:`SegmentationSettings.max_solidity` filters on, because a debris speck or
#: a bubble that Cellpose has labelled comes out as a clean convex disc while a
#: real nucleus in a packed organoid is pressed out of shape by its neighbours.
SHAPE_COLUMNS = ("solidity", "circularity", "eccentricity", "extent")

OBJECT_HEADERS = {
    "label": "Label",
    "solidity": "Solidity",
    "circularity": "Circularity",
    "eccentricity": "Eccentricity",
    "extent": "Extent",
    "n_voxels": "Voxels",
    "volume_um3": "Volume (µm³)",
    "equivalent_diameter_um": "Equivalent diameter (µm)",
    "centroid_z_um": "Centroid Z (µm)",
    "centroid_y_um": "Centroid Y (µm)",
    "centroid_x_um": "Centroid X (µm)",
    "mean": "Mean intensity",
    "median": "Median intensity",
    "std": "Std",
    "max": "Max intensity",
    "integrated": "Integrated intensity",
}


# ---------------------------------------------------------------------------
# What goes in and what comes out
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SegmentationSettings:
    """Everything the run is steered by, in units a microscopist has to hand."""

    backend: str = "cellpose"
    #: Model name (``cpsam`` on v4, ``nuclei`` / ``cyto3`` / … on v3), or empty to
    #: take whatever the installed version offers.
    model: str = ""
    #: A trained model of your own. Takes precedence over :attr:`model`.
    custom_model_path: str = ""
    mode: str = MODE_STITCH
    #: Expected object diameter **in µm**. Zero means let Cellpose decide, which on
    #: v3 is its size model and on v4 is the scale the network was trained at.
    diameter_um: float = 0.0
    #: Cellpose's flow error threshold: lower keeps fewer, rounder objects.
    flow_threshold: float = 0.4
    #: Mask probability cut: lower finds more and larger objects.
    cellprob_threshold: float = 0.0
    #: IoU above which masks in neighbouring planes become one object. Only used
    #: in :data:`MODE_STITCH`.
    stitch_threshold: float = 0.25
    min_size: int = DEFAULT_MIN_SIZE
    #: Per-channel percentile normalisation. Off only if the data is already scaled.
    normalize: bool = True
    #: Ask for the GPU. It is still checked for before being used.
    use_gpu: bool = True
    #: Patches per forward pass. Zero derives it from free video memory.
    batch_size: int = 0
    max_voxels: int = DEFAULT_MAX_VOXELS
    #: Override the Z/XY ratio instead of taking it from the voxel size. Rarely
    #: wanted; it exists because a badly written file can carry a wrong Z step.
    anisotropy: float | None = None
    #: Drop objects whose solidity is **above** this, and measure the shape of the
    #: rest. Zero switches the whole step off, and 1.0 measures without dropping
    #: anything — which is how you look at the numbers before choosing a cut.
    max_solidity: float = 0.0

    def resolved_model(self) -> str:
        """What to hand the backend: a path if one is set, else the model name."""
        if self.custom_model_path:
            return str(self.custom_model_path)
        return self.model or default_model(self.backend)

    @property
    def do_3d(self) -> bool:
        return self.mode == MODE_3D


@dataclass
class SegmentationResult:
    """The label map, on the grid that went in, and what produced it."""

    masks: np.ndarray
    n_objects: int
    voxel_size_um: tuple[float, float, float]
    model: str
    mode: str
    device: str
    #: Diameter actually used by Cellpose, in XY pixels of the analysed volume.
    diameter_px: float | None = None
    anisotropy: float | None = None
    #: Per-axis decimation applied before segmentation; ``(1, 1, 1)`` when none was.
    scale_factors: tuple[float, float, float] = (1.0, 1.0, 1.0)
    elapsed_s: float = 0.0
    warnings: list[str] = field(default_factory=list)
    #: Whether a nuclear channel was passed alongside the segmented one.
    used_nuclear_channel: bool = False
    #: Labels removed by the solidity filter, and the shape of everything measured.
    dropped_labels: list[int] = field(default_factory=list)
    shapes: dict[int, dict[str, float]] = field(default_factory=dict)

    @property
    def n_dropped(self) -> int:
        return len(self.dropped_labels)

    @property
    def voxel_volume_um3(self) -> float:
        z, y, x = self.voxel_size_um
        return float(z) * float(y) * float(x)

    @property
    def decimated(self) -> bool:
        return any(abs(float(f) - 1.0) > 1e-6 for f in self.scale_factors)


@dataclass(frozen=True)
class ModelChoice:
    """One entry in the model list.

    Split from a plain name because a model can be three different things: a name
    the library resolves itself (``cpsam``, ``nuclei``), a model the user trained
    and registered through the Cellpose GUI, or a weights file sitting in the
    cellpose model directory. The panel shows :attr:`label` and passes
    :attr:`value`, which for the last two is a full path.
    """

    label: str
    value: str
    #: ``builtin``, ``user`` (registered in the Cellpose GUI) or ``downloaded``.
    source: str = "builtin"
    detail: str = ""

    def describe(self) -> str:
        if self.source == "builtin":
            return f"{self.label} — ships with cellpose"
        if self.source == "user":
            return f"{self.label} — your own trained model\n{self.detail or self.value}"
        return f"{self.label} — in the cellpose model folder\n{self.detail or self.value}"


@dataclass(frozen=True)
class ObjectStat:
    """One segmented object, measured in calibrated units."""

    label: int
    n_voxels: int
    volume_um3: float
    equivalent_diameter_um: float
    centroid_um: tuple[float, float, float]
    mean: float
    median: float
    std: float
    maximum: float
    integrated: float
    #: Shape descriptors, or NaN when they were not measured. See
    #: :data:`SHAPE_COLUMNS` and :func:`shape_properties`.
    solidity: float = float("nan")
    circularity: float = float("nan")
    eccentricity: float = float("nan")
    extent: float = float("nan")

    def as_row(self) -> dict[str, Any]:
        z, y, x = self.centroid_um
        return {
            "label": self.label,
            "n_voxels": self.n_voxels,
            "volume_um3": self.volume_um3,
            "equivalent_diameter_um": self.equivalent_diameter_um,
            "centroid_z_um": z,
            "centroid_y_um": y,
            "centroid_x_um": x,
            "solidity": self.solidity,
            "circularity": self.circularity,
            "eccentricity": self.eccentricity,
            "extent": self.extent,
            "mean": self.mean,
            "median": self.median,
            "std": self.std,
            "max": self.maximum,
            "integrated": self.integrated,
        }


# ---------------------------------------------------------------------------
# The compute device
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DeviceInfo:
    """Which processor Cellpose will actually run on.

    Separate from :mod:`microscopy_viewer.gpu`, which answers a different question:
    that module reads the OpenGL context napari draws through, this one reads the
    torch device the network runs on. On a machine with two cards they need not be
    the same piece of hardware.
    """

    kind: str = "cpu"  # "cuda", "mps" or "cpu"
    name: str = "CPU"
    total_memory_bytes: int = 0
    free_memory_bytes: int = 0
    index: int = 0

    @property
    def is_gpu(self) -> bool:
        return self.kind in ("cuda", "mps")

    def describe(self) -> str:
        if self.kind == "cpu":
            return "CPU — no GPU available to torch"
        if self.total_memory_bytes:
            return f"{self.name} — {self.total_memory_bytes / 1e9:.1f} GB, {self.kind.upper()}"
        return f"{self.name} — {self.kind.upper()}"


def compute_device(prefer_gpu: bool = True) -> DeviceInfo:
    """The device a run would use right now.

    Asked before a run rather than reported after one: Cellpose falls back to the
    CPU without raising, and the difference between the two is tens of seconds
    against tens of minutes, so it is worth showing on the panel while there is
    still a chance to install a CUDA build of torch.
    """
    try:
        import torch
    except Exception:
        logger.info("torch is not installed; segmentation would run on the CPU")
        return DeviceInfo()

    if prefer_gpu:
        try:
            if torch.cuda.is_available():
                index = int(torch.cuda.current_device())
                properties = torch.cuda.get_device_properties(index)
                try:
                    free, total = torch.cuda.mem_get_info(index)
                except Exception:
                    free, total = 0, int(getattr(properties, "total_memory", 0))
                return DeviceInfo(
                    kind="cuda",
                    name=str(getattr(properties, "name", f"CUDA device {index}")),
                    total_memory_bytes=int(total),
                    free_memory_bytes=int(free),
                    index=index,
                )
        except Exception:
            logger.info("could not interrogate CUDA; falling back", exc_info=True)
        try:
            if torch.backends.mps.is_available():
                # Apple Silicon. No memory query exists, and it shares system RAM.
                return DeviceInfo(kind="mps", name="Apple GPU (Metal)")
        except Exception:
            pass
    return DeviceInfo()


def estimate_batch_size(device: DeviceInfo | None = None) -> int:
    """Patches per forward pass, from free video memory.

    Cellpose's own default of 8 leaves a 24 GB card mostly idle; overshooting it on
    a 4 GB one is an out-of-memory error part-way through a long run, which is the
    worse failure, so this stays deliberately conservative.
    """
    device = device if device is not None else compute_device()
    if not device.is_gpu:
        return 8
    free = device.free_memory_bytes or device.total_memory_bytes
    if not free:
        return 8
    # Roughly 200 MB per 256x256 patch in flight, including activations.
    return int(max(8, min(64, (free * 0.6) // (200 * 1024 * 1024))))


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


def _model_loadable(name: str, major_version: int) -> bool:
    """Whether a weights file called *name* stands a chance under this cellpose.

    Cellpose 4 is a single architecture and cannot read any of the pre-v4 zoo
    weights; cellpose 3 cannot read the v4 ones. Both failures happen at load time
    with an unhelpful tensor shape mismatch, so the list is filtered instead — and
    only for names that are recognisably the other version's, never for a model
    the user trained and named themselves.
    """
    lowered = str(name).lower().removesuffix(".npy")
    if lowered.startswith(SIZE_MODEL_PREFIX):
        return False  # a size model, not something to segment with
    legacy = bool(LEGACY_MODEL_PATTERN.match(lowered))
    modern = lowered.startswith(CPSAM_MODEL) or lowered.startswith("cpdino")
    return not legacy if major_version >= 4 else not modern


class Backend:
    """What a segmentation library has to provide.

    The same seam as :class:`microscopy_viewer.registration.Backend`, and for the
    same reason: StarDist and micro-SAM answer the same question behind different
    imports, so adding one should be a subclass rather than an edit to
    :func:`segment_volume`.
    """

    name = ""
    install_hint = ""

    def available(self) -> bool:
        raise NotImplementedError

    def model_choices(self) -> tuple[ModelChoice, ...]:
        """Everything the panel may offer: builtins, trained models, loose weights."""
        return ()

    def models(self) -> tuple[str, ...]:
        """Just the values, for callers that only want names."""
        return tuple(choice.value for choice in self.model_choices())

    def segment(
        self,
        image: np.ndarray,
        settings: SegmentationSettings,
        diameter_px: float | None,
        anisotropy: float | None,
        device: DeviceInfo,
        progress: Callable[[str], None] | None = None,
        channel_axis: int | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Label *image*. Returns ``(masks, info)``.

        ``channel_axis`` is set when *image* carries more than one stain — the
        cell channel first, the nuclear channel second — and is None for the
        single-channel case, where *image* is purely spatial.
        """
        raise NotImplementedError


class CellposeBackend(Backend):
    """Cellpose 3 and 4, through one call path.

    v4 replaced the model zoo with a single generalist (``cpsam``) and dropped the
    ``channels`` argument; v3 has the zoo, a size model that estimates diameters,
    and has to be told which channels to read. The differences are all here.
    """

    name = "cellpose"
    install_hint = (
        "Segmentation needs the cellpose package:\n\n"
        "    python -m pip install cellpose\n\n"
        "For GPU segmentation install a CUDA build of torch first, from "
        "https://pytorch.org — the default wheel is CPU-only, and on a 3D stack "
        "that is the difference between tens of seconds and tens of minutes."
    )

    def available(self) -> bool:
        return self.import_error() is None

    @staticmethod
    def import_error() -> BaseException | None:
        """Why ``import cellpose`` fails, or None when it works.

        Kept apart from :meth:`available` because "not installed" and "installed
        but will not load" want different sentences: torch failing to load its
        own DLLs is a common second case on Windows, and telling someone to
        install a package they already have sends them the wrong way.
        """
        try:
            import cellpose  # noqa: F401
        except Exception as exc:
            return exc
        return None

    @staticmethod
    def major_version() -> int:
        try:
            import cellpose

            return int(str(cellpose.version).split(".")[0])
        except Exception:
            return 0

    def _builtin_names(self) -> tuple[str, ...]:
        """What the installed cellpose says it ships with.

        Read from the library rather than hard-coded: v3 publishes its whole zoo in
        ``MODEL_NAMES`` and v4 publishes the one generalist, and which of those is
        installed is not this module's business to remember.
        """
        try:
            from cellpose import models as cp_models

            names = tuple(str(name) for name in getattr(cp_models, "MODEL_NAMES", ()) or ())
        except Exception:
            names = ()
        if names:
            return names
        return (CPSAM_MODEL,) if self.major_version() >= 4 else V3_MODELS

    def _model_directory(self) -> Path | None:
        try:
            from cellpose import models as cp_models

            directory = Path(str(cp_models.MODEL_DIR))
        except Exception:
            return None
        return directory if directory.is_dir() else None

    def _registered_models(self) -> list[str]:
        """Models added through the Cellpose GUI, which keeps a list of paths."""
        try:
            from cellpose import models as cp_models

            return [str(entry) for entry in cp_models.get_user_models() if str(entry).strip()]
        except Exception:
            logger.debug("could not read the cellpose user model list", exc_info=True)
            return []

    def model_choices(self) -> tuple[ModelChoice, ...]:
        if not self.available():
            return ()

        major = self.major_version()
        builtins = self._builtin_names()
        choices = [ModelChoice(label=name, value=name, source="builtin") for name in builtins]
        seen = {name.lower() for name in builtins}

        for entry in self._registered_models():
            path = Path(entry)
            key = path.name.lower() if path.name else entry.lower()
            if key in seen:
                continue
            seen.add(key)
            choices.append(
                ModelChoice(label=path.name or entry, value=entry, source="user", detail=entry)
            )

        directory = self._model_directory()
        if directory is not None:
            for path in sorted(directory.iterdir()):
                if not path.is_file() or path.suffix.lower() in NON_MODEL_SUFFIXES:
                    continue
                if path.name.lower() in seen:
                    continue
                if not _model_loadable(path.name, major):
                    # A pre-v4 zoo file under cellpose 4, or cpsam under cellpose 3.
                    # Offering it would fail on the architecture, minutes in.
                    logger.debug("skipping %s: not loadable by cellpose %s", path.name, major)
                    continue
                seen.add(path.name.lower())
                choices.append(
                    ModelChoice(
                        label=path.name,
                        value=str(path),
                        source="downloaded",
                        detail=str(path),
                    )
                )
        return tuple(choices)

    def _model(self, settings: SegmentationSettings, device: DeviceInfo):
        from cellpose import models as cp_models

        wanted = settings.resolved_model()
        use_gpu = bool(settings.use_gpu and device.is_gpu)
        kwargs: dict[str, Any] = {"gpu": use_gpu}
        if use_gpu and device.kind == "cuda":
            # Pin the card that was reported on the panel rather than letting
            # cellpose pick, so what is shown is what runs.
            try:
                import torch

                kwargs["device"] = torch.device(f"cuda:{device.index}")
            except Exception:
                pass

        major = self.major_version()
        if wanted and Path(wanted).exists():
            kwargs["pretrained_model"] = str(wanted)
        elif major >= 4:
            kwargs["pretrained_model"] = wanted or CPSAM_MODEL
        else:
            kwargs["model_type"] = wanted or "cyto3"

        try:
            return cp_models.CellposeModel(**kwargs)
        except Exception as exc:
            # The usual cause is a model from the other major version: the error
            # cellpose raises is a tensor shape mismatch, which says nothing about
            # why. Name the real problem instead.
            label = Path(wanted).name or wanted
            if not _model_loadable(label, major):
                other = 3 if major >= 4 else 4
                raise RuntimeError(
                    f"“{label}” is a Cellpose {other} model and cellpose {major} cannot load it. "
                    f"Pick one of: {', '.join(self._builtin_names())}."
                ) from exc
            raise RuntimeError(f"could not load the model “{label}”: {exc}") from exc

    def segment(
        self,
        image,
        settings,
        diameter_px,
        anisotropy,
        device,
        progress=None,
        channel_axis=None,
    ):
        model = self._model(settings, device)
        batch_size = int(settings.batch_size) or estimate_batch_size(device)

        kwargs: dict[str, Any] = {
            "batch_size": batch_size,
            "normalize": bool(settings.normalize),
            "flow_threshold": float(settings.flow_threshold),
            "cellprob_threshold": float(settings.cellprob_threshold),
            "min_size": int(settings.min_size),
            "do_3D": settings.mode == MODE_3D,
        }
        if diameter_px:
            kwargs["diameter"] = float(diameter_px)

        # The spatial part of the image, with any channel axis discounted: it is
        # what decides whether this is a stack.
        spatial_ndim = np.asarray(image).ndim - (0 if channel_axis is None else 1)
        if spatial_ndim == 3:
            # Say which axis is Z rather than letting cellpose guess: on a stack
            # with few planes it can take Z for a channel axis.
            kwargs["z_axis"] = 0
            if settings.mode == MODE_3D and anisotropy:
                kwargs["anisotropy"] = float(anisotropy)
            if settings.mode == MODE_STITCH:
                kwargs["stitch_threshold"] = float(settings.stitch_threshold)
        kwargs["channel_axis"] = None if channel_axis is None else int(channel_axis)

        if self.major_version() < 4:
            # v3 is told which stain is which by index, counting from 1 with 0
            # meaning "grayscale": [0, 0] is one channel, [1, 2] is "segment the
            # first channel, use the second as its nuclei". v4 infers it from the
            # array and warns about the argument existing at all.
            kwargs["channels"] = [0, 0] if channel_axis is None else [1, 2]

        if progress is not None:
            progress(f"cellpose {settings.mode}, batch {batch_size}")
        outcome = model.eval(image, **kwargs)
        masks = np.asarray(outcome[0])

        info: dict[str, Any] = {"batch_size": batch_size}
        if len(outcome) > 3 and outcome[3] is not None:
            # v3's size model reports the diameter it estimated.
            try:
                info["estimated_diameter_px"] = float(np.mean(np.asarray(outcome[3], dtype=float)))
            except Exception:
                pass
        return masks, info


_BACKENDS: dict[str, Backend] = {CellposeBackend.name: CellposeBackend()}


def register_backend(backend: Backend) -> None:
    """Add a backend to the registry. The hook StarDist would use."""
    _BACKENDS[backend.name] = backend


def get_backend(name: str = "cellpose") -> Backend:
    backend = _BACKENDS.get(str(name).lower())
    if backend is None:
        known = ", ".join(sorted(_BACKENDS)) or "none"
        raise ValueError(f"unknown segmentation backend {name!r} (known: {known})")
    return backend


def available_backends() -> list[str]:
    """Backends whose library is actually importable right now."""
    return sorted(name for name, backend in _BACKENDS.items() if backend.available())


def backend_available(name: str = "cellpose") -> bool:
    try:
        return get_backend(name).available()
    except ValueError:
        return False


def missing_backend_message(name: str = "cellpose") -> str | None:
    """The sentence to show when the backend is not installed, or ``None``.

    Mirrors the atlas panel and the slide export: the panel still builds, it just
    says plainly what to install.
    """
    try:
        backend = get_backend(name)
    except ValueError as exc:
        return str(exc)
    if backend.available():
        return None

    # An import that fails for any reason other than the package being absent is
    # a different problem with a different fix, so say which one it is.
    failure = getattr(backend, "import_error", lambda: None)()
    if failure is not None and not isinstance(failure, ImportError):
        return (
            f"{name} is installed but could not be imported:\n\n"
            f"    {type(failure).__name__}: {failure}\n\n"
            "On Windows this is usually torch failing to load its own DLLs. Check that "
            "the torch build matches the installed CUDA runtime, and that the viewer is "
            "started through microscopy_viewer rather than by importing Qt first."
        )
    return backend.install_hint


def model_choices(name: str = "cellpose") -> tuple[ModelChoice, ...]:
    """Everything the backend can be pointed at: builtins, trained models, weights.

    This is what the panel fills its list from. A user who trained a model in the
    Cellpose GUI, or dropped weights into ``~/.cellpose/models``, finds it here
    without having to know where the file is.
    """
    try:
        return get_backend(name).model_choices()
    except ValueError:
        return ()


def model_directory(name: str = "cellpose") -> Path | None:
    """Where the backend keeps its weights, for a file dialog to open on."""
    backend = _BACKENDS.get(str(name).lower())
    getter = getattr(backend, "_model_directory", None)
    return getter() if getter is not None else None


def available_models(name: str = "cellpose") -> tuple[str, ...]:
    """Model values the installed backend offers; empty when it is not installed."""
    try:
        return get_backend(name).models()
    except ValueError:
        return ()


def default_model(name: str = "cellpose") -> str:
    """The model to start on.

    The nuclear model where a zoo exists. Otherwise a builtin whose weights are
    already downloaded, in preference to the newest one: the alternative is that
    pressing Segment for the first time silently fetches a gigabyte before it does
    anything, which reads as a hang.
    """
    models = available_models(name)
    if not models:
        return CPSAM_MODEL
    if "nuclei" in models:
        return "nuclei"
    directory = model_directory(name)
    if directory is not None:
        cached = {path.name.lower() for path in directory.iterdir() if path.is_file()}
        for model in models:
            if model.lower() in cached:
                return model
    return models[0]


# ---------------------------------------------------------------------------
# Physical units -> cellpose units
# ---------------------------------------------------------------------------


def anisotropy_from_voxel(voxel_size_um: Sequence[float]) -> float | None:
    """Z/XY ratio, or ``None`` when the volume is isotropic enough not to matter."""
    sizes = tuple(float(v) for v in tuple(voxel_size_um)[-3:])
    if len(sizes) < 3:
        return None
    z, y, x = sizes
    lateral = (abs(y) + abs(x)) / 2.0
    if lateral <= 0 or z <= 0:
        return None
    ratio = z / lateral
    return None if abs(ratio - 1.0) < 0.05 else float(ratio)


def diameter_in_pixels(diameter_um: float, voxel_size_um: Sequence[float]) -> float | None:
    """Cellpose's ``diameter``, from a diameter in µm and the XY voxel size.

    Cellpose measures in XY pixels of the array it is handed, which is why this is
    computed after any decimation rather than before: a 5 µm nucleus is 16 px at
    0.3 µm/px and 8 px once the volume has been halved.
    """
    if not diameter_um or diameter_um <= 0:
        return None
    lateral = [abs(float(v)) for v in tuple(voxel_size_um)[-2:] if float(v) > 0]
    if not lateral:
        return None
    return float(diameter_um / (sum(lateral) / len(lateral)))


def plan_scale(
    shape: Sequence[int],
    max_voxels: int = DEFAULT_MAX_VOXELS,
) -> tuple[float, ...]:
    """Per-axis factors bringing *shape* under *max_voxels*. XY only.

    Z is left alone on purpose. A confocal stack is already coarse in Z — often
    four or five planes through a nucleus — and decimating it further merges
    objects the stitching pass would otherwise have kept apart. The lateral axes
    are where the redundancy is.
    """
    dims = tuple(int(n) for n in shape)
    if not dims:
        return ()
    voxels = 1
    for n in dims:
        voxels *= max(int(n), 1)
    if max_voxels <= 0 or voxels <= max_voxels:
        return tuple(1.0 for _ in dims)
    lateral = min(1.0, max(float(np.sqrt(max_voxels / voxels)), 0.05))
    return tuple(1.0 for _ in dims[:-2]) + (lateral,) * min(2, len(dims))


def _zoom(array: np.ndarray, factors: Sequence[float], order: int) -> np.ndarray:
    from scipy import ndimage

    return ndimage.zoom(array, tuple(float(f) for f in factors), order=order, prefilter=order > 1)


def rescale_image(array: np.ndarray, factors: Sequence[float]) -> np.ndarray:
    """Decimate *array* with linear interpolation, or return it untouched."""
    array = np.asarray(array)
    if all(abs(float(f) - 1.0) < 1e-6 for f in factors):
        return array
    return _zoom(array.astype(np.float32, copy=False), factors, order=1)


def restore_masks(masks: np.ndarray, shape: Sequence[int]) -> np.ndarray:
    """Put a decimated label map back on the original grid.

    Nearest neighbour, so labels stay labels — and then trimmed or padded to the
    exact shape, since a zoom factor that is not a clean ratio lands a voxel either
    side of where it is wanted.
    """
    masks = np.asarray(masks)
    target = tuple(int(n) for n in shape)
    if tuple(masks.shape) == target:
        return masks
    factors = [t / max(s, 1) for t, s in zip(target, masks.shape)]
    grown = _zoom(masks, factors, order=0)
    if tuple(grown.shape) == target:
        return grown
    out = np.zeros(target, dtype=grown.dtype)
    region = tuple(slice(0, min(a, b)) for a, b in zip(target, grown.shape))
    out[region] = grown[region]
    return out


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Shape, and throwing away what is the wrong shape
# ---------------------------------------------------------------------------


def shape_error() -> str | None:
    """Why shape descriptors cannot be measured, or ``None`` when they can.

    scikit-image is not a dependency of the viewer but it *is* one of cellpose's,
    so on any machine that can segment it is already there. Saying which of those
    two is missing beats a bare ImportError out of the middle of a plate run.
    """
    try:
        from skimage.measure import regionprops_table  # noqa: F401
    except ImportError as exc:
        return (
            "Shape measurements need scikit-image, which is normally installed "
            f"alongside cellpose:\n\n    python -m pip install scikit-image\n\n({exc})"
        )
    return None


def _spatial_masks(masks: np.ndarray) -> np.ndarray:
    """The label map with leading singleton axes dropped.

    A plate mask arrives as ``(1, y, x)``. Measured as a volume, every object is a
    slab one voxel thick, its convex hull is degenerate and its solidity is not
    the number anyone means — so the axis comes off before regionprops sees it.
    """
    array = np.asarray(masks)
    while array.ndim > 2 and array.shape[0] == 1:
        array = array[0]
    return array


def shape_properties(masks: np.ndarray) -> dict[int, dict[str, float]]:
    """Per-object shape descriptors, keyed by label, from scikit-image.

    Returns :data:`SHAPE_COLUMNS` for every label present. ``circularity`` is
    derived — ``4 pi A / P**2``, 1.0 for a perfect disc — because scikit-image does
    not offer it directly; ``eccentricity`` and the perimeter it needs are 2D-only
    measurements, so a genuine volume comes back with those two left out.
    """
    message = shape_error()
    if message is not None:
        raise ImportError(message)
    from skimage.measure import regionprops_table

    labels = _spatial_masks(masks)
    flat_2d = labels.ndim == 2
    wanted = ["label", "area", "solidity", "extent"]
    if flat_2d:
        wanted += ["perimeter", "eccentricity"]

    table = regionprops_table(labels, properties=tuple(wanted))
    ids = np.asarray(table["label"], dtype=np.int64)
    area = np.asarray(table["area"], dtype=float)
    if flat_2d:
        perimeter = np.asarray(table["perimeter"], dtype=float)
        # A single-pixel object has zero perimeter; guard rather than divide by it.
        circularity = np.where(
            perimeter > 0, 4.0 * np.pi * area / np.maximum(perimeter, 1e-9) ** 2, np.nan
        )
        # Digitised outlines make the ratio overshoot slightly on small discs.
        circularity = np.minimum(circularity, 1.0)
        eccentricity = np.asarray(table["eccentricity"], dtype=float)
    else:
        circularity = np.full(ids.shape, np.nan)
        eccentricity = np.full(ids.shape, np.nan)

    solidity = np.asarray(table["solidity"], dtype=float)
    extent = np.asarray(table["extent"], dtype=float)
    return {
        int(label): {
            "solidity": float(solidity[index]),
            "circularity": float(circularity[index]),
            "eccentricity": float(eccentricity[index]),
            "extent": float(extent[index]),
        }
        for index, label in enumerate(ids)
    }


def round_labels(shapes: dict[int, dict[str, float]], max_solidity: float) -> list[int]:
    """Labels whose solidity is above *max_solidity*, in order.

    Above, not below: the objects being thrown out here are the ones that are *too*
    clean — a speck of debris or an out-of-focus bead is a convex disc, while a real
    nucleus packed against its neighbours is dented by them. A threshold of 1.0
    therefore drops nothing, which is the way to measure without filtering.
    """
    cut = float(max_solidity)
    return sorted(
        label
        for label, values in shapes.items()
        if np.isfinite(values.get("solidity", np.nan)) and values["solidity"] > cut
    )


def drop_labels(masks: np.ndarray, labels: Sequence[int]) -> np.ndarray:
    """A copy of *masks* with *labels* set to background.

    The surviving objects keep their ids rather than being renumbered, so a label
    in the object table is still the label in the image and in any table written
    before the filter ran.
    """
    array = np.asarray(masks)
    if not labels:
        return array
    highest = int(array.max()) if array.size else 0
    keep = np.ones(highest + 2, dtype=bool)
    for label in labels:
        if 0 <= int(label) <= highest:
            keep[int(label)] = False
    keep[0] = False
    return np.where(keep[array], array, 0).astype(array.dtype, copy=False)


def count_labels(masks: np.ndarray) -> int:
    """How many distinct objects are in a label map.

    Not ``masks.max()``: Cellpose leaves holes in its numbering when it drops an
    object below ``min_size``, and the solidity filter leaves more, so the highest
    id stopped being the count some time ago.
    """
    flat = np.asarray(masks).reshape(-1)
    if flat.size == 0:
        return 0
    highest = int(flat.max())
    if highest <= 0:
        return 0
    return int(np.count_nonzero(np.bincount(flat, minlength=highest + 1)[1:]))


def filter_round_objects(
    masks: np.ndarray, max_solidity: float
) -> tuple[np.ndarray, list[int], dict[int, dict[str, float]]]:
    """Measure every object and drop the ones rounder than *max_solidity*.

    Returns the filtered masks, the labels removed, and the shape of everything
    that was measured — the measurements are kept for the object table so that
    regionprops, which is the expensive part, runs once.
    """
    shapes = shape_properties(masks)
    dropped = round_labels(shapes, max_solidity)
    return drop_labels(masks, dropped), dropped, shapes


def segment_volume(
    image: np.ndarray,
    voxel_size_um: Sequence[float] = (1.0, 1.0, 1.0),
    settings: SegmentationSettings | None = None,
    progress: Callable[[str], None] | None = None,
    nuclei: np.ndarray | None = None,
) -> SegmentationResult:
    """Segment one 2D or 3D array and return its labels on the same grid.

    *image* is ``(z, y, x)`` or ``(y, x)``, and *voxel_size_um* is in the same
    order — the calibrated scale off the layer, not a guess.

    *nuclei* is an optional second stain on the same grid. Given one, Cellpose
    segments whole cells from *image* — a membrane or cytoplasmic stain — while
    using the nuclear channel to tell touching cells apart and to guarantee one
    mask per nucleus. It is the difference between "the fluorescent blobs" and
    "the cells", and it is what the two-channel Cellpose models were trained on.
    """
    settings = settings or SegmentationSettings()
    message = missing_backend_message(settings.backend)
    if message is not None:
        raise RuntimeError(message)
    backend = get_backend(settings.backend)

    array = np.asarray(image)
    if array.ndim not in (2, 3):
        raise ValueError(f"segmentation needs a 2D or 3D array, got {array.ndim}D")

    # A stack one plane deep is a 2D image, and segmenting it as a volume is not
    # the same thing: the stitching path treats the single plane as a degenerate
    # z-stack and returns a fraction of the objects a plain 2D run finds. Plate
    # and slide-scanner data reaches us shaped this way all the time, so drop the
    # axis for the run and put it back on the labels.
    singleton_z = array.ndim == 3 and array.shape[0] == 1
    if singleton_z:
        array = array[0]
        voxel_size_um = tuple(voxel_size_um)[-2:] or (1.0, 1.0)

    warnings: list[str] = []

    nuclear = None if nuclei is None else np.asarray(nuclei)
    if nuclear is not None and singleton_z and nuclear.ndim == 3 and nuclear.shape[0] == 1:
        nuclear = nuclear[0]
    if nuclear is not None and nuclear.shape != array.shape:
        warnings.append(
            f"The nuclear channel is {tuple(nuclear.shape)} and the segmented channel is "
            f"{tuple(array.shape)}; they have to be on the same grid, so it was ignored."
        )
        nuclear = None

    device = compute_device(prefer_gpu=settings.use_gpu)
    if settings.use_gpu and not device.is_gpu:
        warnings.append(
            "No GPU available to torch — running on the CPU, which on a 3D stack "
            "takes tens of minutes rather than tens of seconds."
        )

    voxel = tuple(float(v) for v in tuple(voxel_size_um)[-array.ndim:])
    if len(voxel) < array.ndim:
        voxel = (1.0,) * (array.ndim - len(voxel)) + voxel

    factors = plan_scale(array.shape, settings.max_voxels)
    working = rescale_image(array, factors)
    working_voxel = tuple(v / f for v, f in zip(voxel, factors))
    if any(abs(float(f) - 1.0) > 1e-6 for f in factors):
        warnings.append(
            f"Volume decimated to {tuple(working.shape)} for segmentation "
            f"({int(np.prod(array.shape)) / 1e6:.0f} Mvoxels is over the "
            f"{settings.max_voxels / 1e6:.0f} Mvoxel limit); the labels come back on "
            "the original grid."
        )

    diameter_px = diameter_in_pixels(settings.diameter_um, working_voxel)
    anisotropy = settings.anisotropy
    if anisotropy is None and working.ndim == 3:
        anisotropy = anisotropy_from_voxel(working_voxel)

    # Cellpose reads the two stains from one array: the cell channel first, the
    # nuclear channel second. The channel axis goes last, where it cannot be
    # mistaken for Z, and every shape decision above stays on the spatial array.
    payload = working
    channel_axis = None
    if nuclear is not None:
        payload = np.stack([working, rescale_image(nuclear, factors)], axis=-1)
        channel_axis = payload.ndim - 1

    if progress is not None:
        channels = "" if nuclear is None else " with a nuclear channel"
        progress(f"segmenting {tuple(working.shape)}{channels} on {device.describe()}")

    started = time.perf_counter()
    masks, info = backend.segment(
        payload,
        settings=settings,
        diameter_px=diameter_px,
        anisotropy=anisotropy if settings.mode == MODE_3D else None,
        device=device,
        progress=progress,
        channel_axis=channel_axis,
    )
    elapsed = time.perf_counter() - started

    masks = np.asarray(masks)
    if tuple(masks.shape) != tuple(array.shape):
        masks = restore_masks(masks, array.shape)
    masks = masks.astype(np.int32, copy=False)
    if singleton_z:
        # Back onto the grid the caller handed in, so the labels line up with the
        # layer they came from.
        masks = masks[np.newaxis]

    # Shape comes last, on the masks as the caller will get them, so the numbers in
    # the table describe the objects that are actually in the layer.
    shapes: dict[int, dict[str, float]] = {}
    dropped: list[int] = []
    if settings.max_solidity > 0:
        if progress is not None:
            progress("measuring object shape")
        try:
            masks, dropped, shapes = filter_round_objects(masks, settings.max_solidity)
        except ImportError as exc:
            warnings.append(str(exc).replace("\n\n", " "))
        else:
            if dropped:
                warnings.append(
                    f"{len(dropped)} object(s) above solidity {settings.max_solidity:g} "
                    "were dropped as too round to be cells."
                )

    estimated = info.get("estimated_diameter_px")
    if diameter_px is None and estimated:
        diameter_px = float(estimated)

    padded_voxel = ((1.0,) * (3 - len(voxel)) + tuple(voxel))[-3:]
    padded_factors = ((1.0,) * (3 - len(factors)) + tuple(float(f) for f in factors))[-3:]

    result = SegmentationResult(
        masks=masks,
        n_objects=count_labels(masks),
        voxel_size_um=padded_voxel,  # type: ignore[arg-type]
        model=settings.resolved_model(),
        mode=settings.mode,
        device=device.describe(),
        diameter_px=diameter_px,
        anisotropy=anisotropy,
        scale_factors=padded_factors,  # type: ignore[arg-type]
        elapsed_s=elapsed,
        warnings=warnings,
        used_nuclear_channel=nuclear is not None,
        dropped_labels=dropped,
        shapes=shapes,
    )
    logger.info(
        "segmented %s in %.1f s: %d object(s), model %s, %s",
        tuple(array.shape), elapsed, result.n_objects, result.model, device.describe(),
    )
    return result


# ---------------------------------------------------------------------------
# Measuring what came out
# ---------------------------------------------------------------------------


def _equivalent_diameter(volume: float, ndim: int) -> float:
    """Diameter of the sphere (or disc) of the same volume (or area)."""
    if volume <= 0:
        return 0.0
    if ndim >= 3:
        return float(2.0 * (3.0 * volume / (4.0 * np.pi)) ** (1.0 / 3.0))
    return float(2.0 * np.sqrt(volume / np.pi))


def object_table(
    masks: np.ndarray,
    signal: np.ndarray | None = None,
    voxel_size_um: Sequence[float] = (1.0, 1.0, 1.0),
    shapes: dict[int, dict[str, float]] | None = None,
) -> list[ObjectStat]:
    """Per-object size, position and intensity.

    Measured through the indices of the labelled voxels rather than a loop over
    labels: one pass, and the memory it needs is proportional to the segmented
    fraction of the volume rather than to the volume. On a stack where nuclei are
    2 % of the voxels that is the difference between a table and a swap storm.
    """
    labels = np.asarray(masks)
    flat = labels.reshape(-1)
    indices = np.flatnonzero(flat)
    if indices.size == 0:
        return []

    ids = flat[indices].astype(np.int64, copy=False)
    highest = int(ids.max())
    counts = np.bincount(ids, minlength=highest + 1)

    voxel = tuple(float(v) for v in tuple(voxel_size_um)[-labels.ndim:])
    if len(voxel) < labels.ndim:
        voxel = (1.0,) * (labels.ndim - len(voxel)) + voxel
    voxel_volume = float(np.prod(voxel))

    coordinates = np.unravel_index(indices, labels.shape)
    centroids: list[np.ndarray] = []
    for axis, size in enumerate(voxel):
        totals = np.bincount(
            ids, weights=coordinates[axis].astype(np.float64), minlength=highest + 1
        )
        centroids.append(np.divide(totals, np.maximum(counts, 1)) * size)
    while len(centroids) < 3:  # a 2D image has no Z centroid; it is reported as 0
        centroids.insert(0, np.zeros(highest + 1))

    if signal is not None:
        values = np.asarray(signal).reshape(-1)[indices].astype(np.float64)
        sums = np.bincount(ids, weights=values, minlength=highest + 1)
        squares = np.bincount(ids, weights=values * values, minlength=highest + 1)
        means = np.divide(sums, np.maximum(counts, 1))
        stds = np.sqrt(np.maximum(np.divide(squares, np.maximum(counts, 1)) - means**2, 0.0))
        maxima = np.full(highest + 1, -np.inf)
        np.maximum.at(maxima, ids, values)
        maxima[~np.isfinite(maxima)] = 0.0
        medians = _medians(ids, values, highest)
    else:
        sums = means = stds = maxima = medians = np.zeros(highest + 1)

    shape_by_label = shapes or {}
    blank = {name: float("nan") for name in SHAPE_COLUMNS}

    stats: list[ObjectStat] = []
    for label in range(1, highest + 1):
        n_voxels = int(counts[label])
        if n_voxels == 0:  # a label cellpose dropped; the numbering has holes
            continue
        volume = n_voxels * voxel_volume
        shape = shape_by_label.get(label, blank)
        stats.append(
            ObjectStat(
                label=label,
                n_voxels=n_voxels,
                volume_um3=volume,
                equivalent_diameter_um=_equivalent_diameter(volume, labels.ndim),
                centroid_um=(
                    float(centroids[-3][label]),
                    float(centroids[-2][label]),
                    float(centroids[-1][label]),
                ),
                mean=float(means[label]),
                median=float(medians[label]),
                std=float(stds[label]),
                maximum=float(maxima[label]),
                integrated=float(sums[label]),
                solidity=float(shape.get("solidity", float("nan"))),
                circularity=float(shape.get("circularity", float("nan"))),
                eccentricity=float(shape.get("eccentricity", float("nan"))),
                extent=float(shape.get("extent", float("nan"))),
            )
        )
    return stats


def _medians(ids: np.ndarray, values: np.ndarray, highest: int) -> np.ndarray:
    """Median per label, from one sort of the labelled voxels."""
    order = np.argsort(ids, kind="stable")
    sorted_ids = ids[order]
    sorted_values = values[order]
    medians = np.zeros(highest + 1)
    boundaries = np.flatnonzero(np.diff(sorted_ids)) + 1
    starts = np.concatenate(([0], boundaries))
    stops = np.concatenate((boundaries, [sorted_ids.size]))
    for start, stop in zip(starts, stops):
        medians[int(sorted_ids[start])] = float(np.median(sorted_values[start:stop]))
    return medians


def object_dataframe(stats: Sequence[ObjectStat]):
    """The per-object table as a DataFrame, ready for the workbook writer."""
    import pandas as pd

    frame = pd.DataFrame([stat.as_row() for stat in stats], columns=list(OBJECT_COLUMNS))
    return frame.rename(columns=OBJECT_HEADERS)


def count_summary(stats: Sequence[ObjectStat]) -> dict[str, float]:
    """Count and size distribution, for the one line under the table."""
    if not stats:
        return {"count": 0.0}
    volumes = np.array([stat.volume_um3 for stat in stats], dtype=float)
    diameters = np.array([stat.equivalent_diameter_um for stat in stats], dtype=float)
    totals = {
        "count": float(len(stats)),
        "median_volume_um3": float(np.median(volumes)),
        "median_diameter_um": float(np.median(diameters)),
        "total_volume_um3": float(volumes.sum()),
    }
    solidity = np.array([stat.solidity for stat in stats], dtype=float)
    if np.any(np.isfinite(solidity)):
        totals["median_solidity"] = float(np.nanmedian(solidity))
    return totals
