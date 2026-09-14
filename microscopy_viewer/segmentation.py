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
from dataclasses import dataclass, field, replace
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
#: Flatten the stack with a maximum projection and segment that one image. The
#: usual choice for sparse, well-separated objects in a thin stack: it is the
#: fastest mode by far and gives one label per object rather than one per plane.
#: Objects that overlap along Z merge into one, which is the trade being made.
MODE_MIP = "2D on max projection"

MODES = (MODE_STITCH, MODE_3D, MODE_SLICES, MODE_MIP)

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

#: How far stitching reaches across planes where an object was missed, in µm of
#: depth. A nucleus is rarely absent from more than a fraction of a micrometre of
#: an otherwise continuous run, and on a 5 µm nucleus this is well under the
#: distance to the next one along Z. Zero restores Cellpose's own behaviour of
#: comparing neighbouring planes only.
DEFAULT_STITCH_GAP_UM = 1.5

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
    "mean",
    "median",
    "std",
    "max",
    "integrated",
)

OBJECT_HEADERS = {
    "label": "Label",
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

#: Headers that mean something different for a 2D label map. The measurement is
#: already right — :func:`object_table` multiplies by whatever voxel size it is
#: given — but "Volume (µm³)" over an area in µm² is a wrong label on a right
#: number, which is worse than no label. Projected runs make 2D the usual case.
OBJECT_HEADERS_2D = {
    "n_voxels": "Pixels",
    "volume_um3": "Area (µm²)",
}


def object_headers(ndim: int = 3) -> dict[str, str]:
    """Column headings for a label map of *ndim* dimensions."""
    headers = dict(OBJECT_HEADERS)
    if ndim < 3:
        headers.update(OBJECT_HEADERS_2D)
    return headers


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
    #: Smallest object to keep, as an equivalent diameter in **µm** — the same
    #: quantity the object table reports, so a number read off the table can be
    #: typed straight back in. Zero leaves :attr:`min_size` alone.
    min_diameter_um: float = 0.0
    #: Largest object to keep, as an equivalent diameter in **µm**. Cellpose has
    #: no ceiling of its own — ``min_size`` has no counterpart — so this is
    #: applied to the finished label map by :func:`filter_by_diameter`, measured
    #: exactly the way the object table measures it. Zero means no limit.
    max_diameter_um: float = 0.0
    #: Cellpose's flow error threshold: lower keeps fewer, rounder objects.
    flow_threshold: float = 0.4
    #: Mask probability cut: lower finds more and larger objects.
    cellprob_threshold: float = 0.0
    #: IoU above which masks in neighbouring planes become one object. Only used
    #: in :data:`MODE_STITCH`.
    stitch_threshold: float = 0.25
    #: How far, in **µm of depth**, stitching may reach across planes where an
    #: object was missed. Cellpose compares neighbouring planes only, so one
    #: plane in which a nucleus was not found splits it into two objects that no
    #: threshold can rejoin. Zero restores that behaviour exactly.
    stitch_gap_um: float = DEFAULT_STITCH_GAP_UM
    #: The above in planes, derived from the Z voxel size by :func:`segment_volume`
    #: once the working volume is known. Not something to set by hand.
    stitch_gap_planes: int = 1
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

    def resolved_model(self) -> str:
        """What to hand the backend: a path if one is set, else the model name."""
        if self.custom_model_path:
            return str(self.custom_model_path)
        return self.model or default_model(self.backend)

    @property
    def do_3d(self) -> bool:
        return self.mode == MODE_3D

    @property
    def is_projection(self) -> bool:
        """Whether a 3D stack is flattened to one plane before segmenting."""
        return self.mode == MODE_MIP


@dataclass
class SegmentationResult:
    """The label map, on the grid that went in, and what produced it."""

    masks: np.ndarray
    n_objects: int
    voxel_size_um: tuple[float, float, float]
    model: str
    mode: str
    device: str
    #: True when a 3D stack was flattened first, so ``masks`` is 2D although the
    #: image was not. Callers measuring intensities have to project too.
    projected: bool = False
    #: Objects removed after segmentation by the maximum-diameter filter.
    dropped_oversize: int = 0
    #: Planes the stitch was allowed to reach across. 1 is neighbours only.
    stitch_gap_planes: int = 1
    #: Diameter actually used by Cellpose, in XY pixels of the analysed volume.
    diameter_px: float | None = None
    anisotropy: float | None = None
    #: Per-axis decimation applied before segmentation; ``(1, 1, 1)`` when none was.
    scale_factors: tuple[float, float, float] = (1.0, 1.0, 1.0)
    elapsed_s: float = 0.0
    warnings: list[str] = field(default_factory=list)

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

    def unsupported_models(self) -> tuple[str, ...]:
        """Models present locally that this backend version cannot load."""
        return ()

    def segment(
        self,
        image: np.ndarray,
        settings: SegmentationSettings,
        diameter_px: float | None,
        anisotropy: float | None,
        device: DeviceInfo,
        progress: Callable[[str], None] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Label *image*. Returns ``(masks, info)``."""
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
        try:
            import cellpose  # noqa: F401
        except Exception:
            return False
        return True

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
                    # Cellpose refuses these outright — "This model does not appear
                    # to be a CP4 model" — so they are reported by
                    # :meth:`unsupported_models` rather than offered and then failing.
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

    def unsupported_models(self) -> tuple[str, ...]:
        """Weights present on this machine that the installed cellpose refuses.

        Almost always the cyto / cyto2 / cyto3 / nuclei zoo under cellpose 4:
        cellpose rejects them with *"This model does not appear to be a CP4 model.
        CP3 models are not compatible with CP4"*. Hiding them without saying so
        looks like the panel lost them, so the names come back here for the panel
        to name — with what to install to get them.
        """
        directory = self._model_directory()
        if directory is None or not self.available():
            return ()
        major = self.major_version()
        names = []
        for path in sorted(directory.iterdir()):
            if not path.is_file() or path.suffix.lower() in NON_MODEL_SUFFIXES:
                continue
            if path.name.lower().startswith(SIZE_MODEL_PREFIX):
                continue
            if not _model_loadable(path.name, major):
                names.append(path.name)
        return tuple(names)

    def _sized_model(self, settings: SegmentationSettings, device: DeviceInfo):
        """Cellpose 3's ``Cellpose`` wrapper, which estimates the diameter itself.

        Only reachable on v3, only for a zoo model, and only when no diameter was
        given: it is the size model that makes "automatic" mean something there.
        Plain :class:`CellposeModel` would silently fall back to the model's
        training diameter instead, which is wrong for anything not imaged at that
        magnification. v4 has no size model — its diameter argument is optional
        because the network is scale-tolerant.
        """
        if self.major_version() >= 4 or settings.diameter_um > 0 or settings.custom_model_path:
            return None
        wanted = settings.resolved_model()
        if not wanted or Path(wanted).exists() or wanted not in self._builtin_names():
            return None

        from cellpose import models as cp_models

        factory = getattr(cp_models, "Cellpose", None)
        if factory is None:
            return None
        kwargs: dict[str, Any] = {
            "gpu": bool(settings.use_gpu and device.is_gpu),
            "model_type": wanted,
        }
        try:
            return factory(**kwargs)
        except Exception:
            logger.info("no size model for %s; using its training diameter", wanted, exc_info=True)
            return None

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

    def _segment_planes(self, model, image, kwargs, progress=None, offset_labels=True):
        """Segment a stack one plane at a time and stack the results back up.

        The mode that wants this is "2D per plane", where planes are deliberately
        independent. Doing the loop here rather than handing cellpose the volume
        is not a workaround for the z_axis rule so much as an honest reading of
        it: cellpose is being asked for 2D segmentation, so it is given 2D images.

        Labels are made unique across the stack. Restarting from 1 on every plane
        would collide in the label map, so one object would appear to span planes
        it was never found in — the exact thing this mode exists not to do.

        ``offset_labels=False`` leaves every plane numbered from 1, which is what
        :func:`cellpose.utils.stitch3D` expects: it matches labels between
        neighbouring planes by overlap and renumbers them itself. Offsetting first
        defeats it — it finds no matches and every plane's objects stay separate.
        """
        per_plane = {key: value for key, value in kwargs.items()
                     if key not in ("z_axis", "stitch_threshold", "anisotropy")}
        per_plane["do_3D"] = False

        planes: list[np.ndarray] = []
        highest = 0
        total = int(image.shape[0])
        for index, plane in enumerate(image):
            if progress is not None:
                progress(f"plane {index + 1} of {total}")
            outcome = model.eval(plane, **per_plane)
            masks = np.asarray(outcome[0]).astype(np.int32, copy=True)
            if offset_labels:
                if highest and masks.size:
                    masks[masks > 0] += highest
                if masks.size:
                    highest = max(highest, int(masks.max()))
            planes.append(masks)

        stacked = np.stack(planes) if planes else np.zeros(image.shape, dtype=np.int32)
        return stacked, {"batch_size": int(kwargs.get("batch_size", 0)), "per_plane": True}

    def segment(
        self,
        image,
        settings,
        diameter_px,
        anisotropy,
        device,
        progress=None,
    ):
        model = self._sized_model(settings, device) or self._model(settings, device)
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
        stack = np.asarray(image).ndim == 3
        # Cellpose only accepts ``z_axis`` when it is going to treat the array as a
        # volume — do_3D, or stitching with a real threshold. Handed a 3D array in
        # plain 2D mode it refuses outright: "2D image processing selected, but
        # z_axis is not None". That covers "2D per plane", and also "2D + stitch"
        # with the threshold wound down to 0, which asks for exactly the same thing.
        volumetric = stack and settings.mode == MODE_3D
        stitching = stack and settings.mode == MODE_STITCH and float(settings.stitch_threshold) > 0
        if volumetric:
            # Say which axis is Z rather than letting cellpose guess: on a stack
            # with few planes it can take Z for a channel axis.
            kwargs["z_axis"] = 0
            kwargs["channel_axis"] = None
            if anisotropy:
                kwargs["anisotropy"] = float(anisotropy)
        elif stitching:
            # Segment the planes here rather than inside model.eval, and stitch
            # them with :func:`stitch_planes`. At a gap of one plane that is the
            # same join cellpose makes internally — checked voxel for voxel — but
            # the planes can be counted off as they go, and the gap can be opened
            # up, which is the only way to rejoin an object missing from a plane.
            planes, info = self._segment_planes(
                model, np.asarray(image), kwargs, progress, offset_labels=False
            )
            gap = max(1, int(settings.stitch_gap_planes))
            joined = stitch_planes(
                planes,
                stitch_threshold=float(settings.stitch_threshold),
                max_gap=gap,
                progress=progress,
            )
            return joined, {**info, "stitched": True, "stitch_gap_planes": gap}
        elif stack:
            if progress is not None:
                progress(f"cellpose per plane, {np.asarray(image).shape[0]} planes")
            return self._segment_planes(model, np.asarray(image), kwargs, progress)
        if self.major_version() < 4:
            # v3 reads one grayscale channel when told [0, 0]; v4 warns about the
            # argument existing at all.
            kwargs["channels"] = [0, 0]

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
    return None if backend.available() else backend.install_hint


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


def unsupported_models_message(name: str = "cellpose") -> str | None:
    """One sentence naming local models this cellpose cannot load, or ``None``.

    The zoo (``cyto``, ``cyto2``, ``cyto3``, ``nuclei``) belongs to Cellpose 3 and
    the architecture changed in 4, so there is no way to offer both from one
    install. Which one is wanted is a real choice, not a bug to route around, so
    the panel states it and names the command that switches.
    """
    try:
        backend = get_backend(name)
    except ValueError:
        return None
    missing = backend.unsupported_models()
    if not missing:
        return None
    major = getattr(backend, "major_version", lambda: 0)()
    listed = ", ".join(missing[:4]) + ("…" if len(missing) > 4 else "")
    if major >= 4:
        return (
            f"{len(missing)} model(s) in your cellpose folder ({listed}) are Cellpose 3 models, "
            f"which Cellpose {major} cannot load. For the cyto / cyto2 / cyto3 / nuclei zoo: "
            'python -m pip install "cellpose<4"'
        )
    return (
        f"{len(missing)} model(s) in your cellpose folder ({listed}) need Cellpose 4: "
        "python -m pip install --upgrade cellpose"
    )


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


def max_projection(array: np.ndarray) -> np.ndarray:
    """Flatten a ``(z, y, x)`` stack to ``(y, x)`` by taking the brightest voxel.

    A 2D array comes back untouched, so callers can apply this unconditionally.

    Exposed rather than buried inside :func:`segment_volume` because anything
    measuring intensities against projected labels has to project its own channel
    the same way — a 2D label map and a 3D signal do not line up.
    """
    array = np.asarray(array)
    if array.ndim < 3:
        return array
    return array.max(axis=0)


def project_for_mode(
    array: np.ndarray, voxel_size_um: Sequence[float], mode: str
) -> tuple[np.ndarray, tuple[float, ...]]:
    """Apply *mode*'s flattening to an array and its voxel size together.

    Dropping the Z voxel alongside the Z axis is the point: keeping a three-entry
    voxel size for a 2D image is how a diameter in µm silently becomes wrong.
    """
    voxel = tuple(float(v) for v in voxel_size_um)
    if mode != MODE_MIP or np.asarray(array).ndim < 3:
        return np.asarray(array), voxel
    return max_projection(array), voxel[-2:]


def min_size_in_pixels(
    min_diameter_um: float, voxel_size_um: Sequence[float], ndim: int
) -> int | None:
    """Cellpose's ``min_size`` from a minimum equivalent diameter in µm.

    ``min_size`` counts pixels in 2D and voxels in 3D, so a threshold that means
    anything physical has to be converted against the voxel size — and against the
    right number of dimensions, which is the mode's, not the array's: stitching
    and per-plane modes both filter 2D masks even though the input is a stack.

    This is the exact inverse of :func:`_equivalent_diameter`, so "minimum 5 µm"
    drops what the table would report as under 5 µm across — exactly so when the
    filter and the table count the same dimensions, which is every mode except
    ``2D + stitch``. There cellpose screens each plane before stitching, so an
    object assembled from several just-surviving planes can still be reported
    smaller than the threshold.
    """
    if not min_diameter_um or min_diameter_um <= 0:
        return None
    voxel = [abs(float(v)) for v in tuple(voxel_size_um)[-ndim:] if float(v) > 0]
    if len(voxel) < ndim:
        return None
    radius = float(min_diameter_um) / 2.0
    extent = (4.0 / 3.0) * np.pi * radius**3 if ndim >= 3 else np.pi * radius**2
    return max(0, int(round(extent / float(np.prod(voxel)))))


# ---------------------------------------------------------------------------
# Stitching planes into objects
# ---------------------------------------------------------------------------


def stitch_gap_in_planes(gap_um: float, z_voxel_um: float) -> int:
    """How many planes :attr:`SegmentationSettings.stitch_gap_um` spans.

    In µm rather than planes because a plane is not a fixed distance: four planes
    is 1.2 µm on a 0.3 µm/plane stack and 8 µm on a 2 µm/plane one, and the second
    would happily bridge two different cells stacked on top of each other.
    """
    if gap_um <= 0 or z_voxel_um <= 0:
        return 1
    return max(1, int(round(float(gap_um) / float(z_voxel_um))))


class _Union:
    """Union-find over ``(plane, label)`` keys. Small, and it keeps the stitch flat."""

    def __init__(self) -> None:
        self._parent: dict[tuple[int, int], tuple[int, int]] = {}

    def find(self, key: tuple[int, int]) -> tuple[int, int]:
        parent = self._parent
        parent.setdefault(key, key)
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def join(self, a: tuple[int, int], b: tuple[int, int]) -> None:
        root_a, root_b = self.find(a), self.find(b)
        if root_a != root_b:
            self._parent[root_b] = root_a


def _pair_overlaps(first: np.ndarray, second: np.ndarray):
    """``(label_a, label_b, intersection)`` for every overlapping mask pair.

    The pair is encoded into one integer and counted with a single sort rather
    than with ``np.unique(..., axis=0)`` over a two-column array, which lexsorts
    and is several times slower. This runs once per plane pair, so it is the
    inner loop of the whole stitch.
    """
    flat_a = first.reshape(-1)
    flat_b = second.reshape(-1)
    both = (flat_a > 0) & (flat_b > 0)
    if not both.any():
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.int64)
    left = flat_a[both].astype(np.int64)
    right = flat_b[both].astype(np.int64)
    stride = int(right.max()) + 1
    codes, counts = np.unique(left * stride + right, return_counts=True)
    return codes // stride, codes % stride, counts


def _areas(plane: np.ndarray) -> np.ndarray:
    """Pixel count per label, indexed by label."""
    flat = plane.reshape(-1)
    return np.bincount(flat[flat > 0].astype(np.int64))


def _keep_only(plane: np.ndarray, labels: set) -> np.ndarray:
    """*plane* with every label outside *labels* erased."""
    if not labels:
        return np.zeros_like(plane)
    lookup = np.zeros(int(plane.max()) + 1, dtype=plane.dtype)
    index = np.fromiter((label for label in labels if 0 < label < lookup.size), dtype=np.int64)
    if index.size:
        lookup[index] = index
    return lookup[plane]


def stitch_planes(
    planes: np.ndarray,
    stitch_threshold: float = 0.25,
    max_gap: int = 1,
    progress: Callable[[str], None] | None = None,
) -> np.ndarray:
    """Join per-plane masks into 3D objects, tolerating planes where one is missing.

    Every plane must be numbered from 1 independently — what
    :meth:`CellposeBackend._segment_planes` produces with ``offset_labels=False``.

    With ``max_gap=1`` this compares neighbouring planes only and reproduces
    :func:`cellpose.utils.stitch3D` exactly. That is the behaviour worth
    replacing: **Cellpose only ever compares plane i with plane i+1, so a single
    plane in which a nucleus was missed splits it into two objects, and no stitch
    threshold can put it back together** — the two halves never get compared at
    all. On a 20-plane confocal crop that turned 551 nuclei into 907.

    A ``max_gap`` above 1 lets a chain that has been broken reach further, under
    two rules:

    * **Only broken chains bridge.** A label that already matched in the next
      plane is not offered a longer jump, and neither is one that already has a
      predecessor. Bridging can therefore only repair a gap, never reroute a
      match that already worked.
    * **Best overlap wins, one match each way.** Pairs are taken in descending
      order of intersection-over-union, and each label may be joined to one label
      per direction, so two nuclei that touch in a plane do not collapse into one.

    Measured against the same volume segmented in full 3D — the mode that does
    understand an object missing from a plane — raising the gap improves both
    directions at once: objects wrongly split fell from 73 to 24, and objects
    wrongly merged from 52 to 31. Bridging repairs splits rather than trading
    them for merges.
    """
    stack = np.asarray(planes)
    if stack.ndim != 3 or stack.shape[0] < 2:
        return stack.astype(np.int32, copy=False)

    union = _Union()
    has_next: set[tuple[int, int]] = set()
    has_prev: set[tuple[int, int]] = set()
    threshold = float(stitch_threshold)
    total = int(stack.shape[0])
    furthest = max(1, int(max_gap))

    for gap in range(1, furthest + 1):
        if progress is not None:
            progress(f"stitching {total} planes, gap {gap} of {furthest}")
        for index in range(total - gap):
            other = index + gap
            first, second = stack[index], stack[other]
            if gap > 1:
                # Only labels whose chain is broken may reach across a gap. This
                # is also what keeps the longer passes cheap: after the first
                # pass most labels are already linked, so these planes are nearly
                # empty by the time they are compared.
                loose_a = {int(v) for v in np.unique(first) if v and (index, int(v)) not in has_next}
                loose_b = {int(v) for v in np.unique(second) if v and (other, int(v)) not in has_prev}
                if not loose_a or not loose_b:
                    continue
                first = _keep_only(first, loose_a)
                second = _keep_only(second, loose_b)

            left, right, inter = _pair_overlaps(first, second)
            if not left.size:
                continue
            area_a, area_b = _areas(first), _areas(second)
            scores = inter / np.maximum(area_a[left] + area_b[right] - inter, 1)

            taken_a: set[int] = set()
            taken_b: set[int] = set()
            for position in np.argsort(scores)[::-1]:
                if scores[position] < threshold:
                    break  # sorted, so nothing further can clear the threshold
                label_a, label_b = int(left[position]), int(right[position])
                if label_a in taken_a or label_b in taken_b:
                    continue
                taken_a.add(label_a)
                taken_b.add(label_b)
                union.join((index, label_a), (other, label_b))
                has_next.add((index, label_a))
                has_prev.add((other, label_b))

    return _renumber(stack, union)


def _renumber(planes: np.ndarray, union: _Union) -> np.ndarray:
    """Give every connected chain one label, numbered from 1 with no gaps."""
    out = np.zeros(planes.shape, dtype=np.int32)
    numbering: dict[tuple[int, int], int] = {}
    for index, plane in enumerate(planes):
        labels = np.unique(plane)
        if not labels.size:
            continue
        lookup = np.zeros(int(labels.max()) + 1, dtype=np.int32)
        for label in labels:
            if not label:
                continue
            root = union.find((index, int(label)))
            if root not in numbering:
                numbering[root] = len(numbering) + 1
            lookup[int(label)] = numbering[root]
        out[index] = lookup[plane]
    return out


def filter_ndim(mode: str, array_ndim: int) -> int:
    """How many dimensions cellpose's ``min_size`` counts over, for *mode*."""
    if array_ndim < 3:
        return 2
    return 3 if mode == MODE_3D else 2


def filter_by_diameter(
    masks: np.ndarray,
    voxel_size_um: Sequence[float],
    max_diameter_um: float = 0.0,
    min_diameter_um: float = 0.0,
) -> tuple[np.ndarray, int]:
    """Drop finished objects outside a size band. Returns ``(masks, dropped)``.

    This is the ceiling Cellpose has not got: ``min_size`` screens small masks
    while they are being made, and nothing screens large ones. A cluster of
    touching nuclei that came back as one 40 µm object is the usual thing to
    remove, and it can only be removed once the object exists — in
    :data:`MODE_STITCH` it does not even exist until the planes are joined.

    Size is the **equivalent diameter of the label map that is returned**, the
    same number :func:`object_table` puts in its *Equivalent diameter* column,
    so a threshold read off the table means the same thing typed back in. On a
    3D label map that is the diameter of the sphere of equal volume; on a 2D one,
    the disc of equal area.

    *min_diameter_um* is accepted for symmetry and is off by default — the
    minimum normally goes to Cellpose as ``min_size``, which is cheaper because
    it never builds the objects it rejects.

    Survivors are renumbered from 1 with no gaps, so the count of objects and the
    highest label agree afterwards.
    """
    labels = np.asarray(masks)
    if not labels.size or (max_diameter_um <= 0 and min_diameter_um <= 0):
        return labels, 0

    flat = labels.reshape(-1)
    indices = np.flatnonzero(flat)
    if indices.size == 0:
        return labels, 0

    ids = flat[indices].astype(np.int64, copy=False)
    highest = int(ids.max())
    counts = np.bincount(ids, minlength=highest + 1)

    voxel = tuple(float(v) for v in tuple(voxel_size_um)[-labels.ndim:])
    if len(voxel) < labels.ndim:
        voxel = (1.0,) * (labels.ndim - len(voxel)) + voxel
    voxel_volume = float(np.prod(voxel)) or 1.0

    # The inverse of :func:`_equivalent_diameter`, done once on the threshold
    # rather than per object: comparing voxel counts avoids a cube root per label.
    def _voxels_for(diameter: float) -> float:
        radius = float(diameter) / 2.0
        extent = (4.0 / 3.0) * np.pi * radius**3 if labels.ndim >= 3 else np.pi * radius**2
        return extent / voxel_volume

    keep = counts > 0
    keep[0] = False
    if max_diameter_um > 0:
        keep &= counts <= _voxels_for(max_diameter_um)
    if min_diameter_um > 0:
        keep &= counts >= _voxels_for(min_diameter_um)

    dropped = int(np.count_nonzero((counts > 0)[1:]) - np.count_nonzero(keep))
    if dropped == 0:
        return labels, 0

    # One lookup table rather than a pass per label: renumber and drop together.
    lookup = np.zeros(highest + 1, dtype=np.int32)
    lookup[keep] = np.arange(1, int(np.count_nonzero(keep)) + 1, dtype=np.int32)
    return lookup[labels], dropped


def segment_volume(
    image: np.ndarray,
    voxel_size_um: Sequence[float] = (1.0, 1.0, 1.0),
    settings: SegmentationSettings | None = None,
    progress: Callable[[str], None] | None = None,
) -> SegmentationResult:
    """Segment one 2D or 3D array and return its labels on the same grid.

    *image* is ``(z, y, x)`` or ``(y, x)``, and *voxel_size_um* is in the same
    order — the calibrated scale off the layer, not a guess.
    """
    settings = settings or SegmentationSettings()
    message = missing_backend_message(settings.backend)
    if message is not None:
        raise RuntimeError(message)
    backend = get_backend(settings.backend)

    array = np.asarray(image)
    if array.ndim not in (2, 3):
        raise ValueError(f"segmentation needs a 2D or 3D array, got {array.ndim}D")

    warnings: list[str] = []

    # Flatten before anything else: the decimation budget, the voxel size and the
    # diameter conversion all have to see the array that is actually segmented.
    projected = settings.is_projection and array.ndim == 3
    if projected:
        planes = int(array.shape[0])
        array, voxel_size_um = project_for_mode(array, voxel_size_um, settings.mode)
        warnings.append(
            f"Segmented the maximum projection of {planes} planes; the labels are 2D. "
            "Objects overlapping along Z merge into one."
        )

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

    # A minimum size in µm becomes a pixel count here, where both the decimated
    # voxel size and the mode are known.
    effective = settings
    min_ndim = filter_ndim(settings.mode, working.ndim)
    converted = min_size_in_pixels(settings.min_diameter_um, working_voxel, min_ndim)
    if converted is not None:
        effective = replace(settings, min_size=converted)
        logger.info(
            "minimum size %.2f µm -> %d %s",
            settings.min_diameter_um, converted, "voxels" if min_ndim >= 3 else "pixels",
        )

    # The stitch gap is in µm of depth; it becomes a plane count here, against the
    # Z voxel size of the volume actually being segmented.
    gap_planes = 1
    if settings.mode == MODE_STITCH and working.ndim == 3:
        gap_planes = stitch_gap_in_planes(settings.stitch_gap_um, working_voxel[0])
        effective = replace(effective, stitch_gap_planes=gap_planes)
        if gap_planes > 1:
            logger.info(
                "stitch gap %.2f µm -> %d plane(s) at %.3f µm/plane",
                settings.stitch_gap_um, gap_planes, working_voxel[0],
            )

    if progress is not None:
        progress(f"segmenting {tuple(working.shape)} on {device.describe()}")

    started = time.perf_counter()
    masks, info = backend.segment(
        working,
        settings=effective,
        diameter_px=diameter_px,
        anisotropy=anisotropy if settings.mode == MODE_3D else None,
        device=device,
        progress=progress,
    )
    elapsed = time.perf_counter() - started

    masks = np.asarray(masks)
    if tuple(masks.shape) != tuple(array.shape):
        masks = restore_masks(masks, array.shape)
    masks = masks.astype(np.int32, copy=False)

    # The ceiling is applied here, on the full grid: it is measured against the
    # voxel size the table will use, so "drop anything over 20 µm" removes
    # exactly the rows the table would show as over 20 µm across.
    dropped = 0
    if settings.max_diameter_um > 0:
        masks, dropped = filter_by_diameter(masks, voxel, settings.max_diameter_um)
        if dropped:
            warnings.append(
                f"Dropped {dropped} object(s) over {settings.max_diameter_um:g} µm across."
            )
            logger.info("maximum diameter %.2f µm dropped %d object(s)",
                        settings.max_diameter_um, dropped)

    estimated = info.get("estimated_diameter_px")
    if diameter_px is None and estimated:
        diameter_px = float(estimated)

    padded_voxel = ((1.0,) * (3 - len(voxel)) + tuple(voxel))[-3:]
    padded_factors = ((1.0,) * (3 - len(factors)) + tuple(float(f) for f in factors))[-3:]

    result = SegmentationResult(
        masks=masks,
        n_objects=int(masks.max()) if masks.size else 0,
        dropped_oversize=dropped,
        stitch_gap_planes=gap_planes,
        voxel_size_um=padded_voxel,  # type: ignore[arg-type]
        model=settings.resolved_model(),
        mode=settings.mode,
        device=device.describe(),
        projected=projected,
        diameter_px=diameter_px,
        anisotropy=anisotropy,
        scale_factors=padded_factors,  # type: ignore[arg-type]
        elapsed_s=elapsed,
        warnings=warnings,
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

    stats: list[ObjectStat] = []
    for label in range(1, highest + 1):
        n_voxels = int(counts[label])
        if n_voxels == 0:  # a label cellpose dropped; the numbering has holes
            continue
        volume = n_voxels * voxel_volume
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


def object_dataframe(stats: Sequence[ObjectStat], ndim: int = 3):
    """The per-object table as a DataFrame, ready for the workbook writer.

    *ndim* is the label map's, and only picks the column headings — pass 2 for a
    projected run so the size column says area rather than volume. It is not
    inferred from the stats: objects in a genuine 3D run can all sit at z=0, and
    guessing wrong puts a wrong unit on a right number.
    """
    import pandas as pd

    frame = pd.DataFrame([stat.as_row() for stat in stats], columns=list(OBJECT_COLUMNS))
    return frame.rename(columns=object_headers(ndim))


def count_summary(stats: Sequence[ObjectStat]) -> dict[str, float]:
    """Count and size distribution, for the one line under the table."""
    if not stats:
        return {"count": 0.0}
    volumes = np.array([stat.volume_um3 for stat in stats], dtype=float)
    diameters = np.array([stat.equivalent_diameter_um for stat in stats], dtype=float)
    return {
        "count": float(len(stats)),
        "median_volume_um3": float(np.median(volumes)),
        "median_diameter_um": float(np.median(diameters)),
        "total_volume_um3": float(volumes.sum()),
    }
