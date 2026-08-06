"""File loading: dispatch a path to the right reader and return :class:`LayerSpec` objects.

Add a new format by writing a module with ``can_read(path)`` and ``read(path)``,
then listing it in :data:`_READERS`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Sequence

from ..utils import get_logger
from . import ims, ome_zarr, tiff
from .layer_spec import COLORMAP_CYCLE, LayerSpec

logger = get_logger("loaders")

#: Readers in priority order. Imaris is checked first because ``.ims`` files are
#: the primary format for this viewer.
_READERS: tuple[tuple[str, Callable[[Path], bool], Callable[[Path], list[LayerSpec]]], ...] = (
    ("Imaris", ims.can_read, ims.read),
    ("TIFF/OME-TIFF", tiff.can_read, tiff.read),
    ("OME-Zarr", ome_zarr.can_read, ome_zarr.read),
)

#: Patterns for the "Open Images" file dialog.
FILE_DIALOG_FILTER = ";;".join(
    [
        "All supported (*.ims *.tif *.tiff *.btf *.tf8 *.lsm *.stk *.zarr)",
        "Imaris (*.ims *.imaris)",
        "TIFF / OME-TIFF (*.tif *.tiff *.btf *.tf8 *.lsm *.stk)",
        "OME-Zarr (*.zarr)",
        "All files (*)",
    ]
)

SUPPORTED_SUFFIXES = (
    ".ims", ".imaris", ".tif", ".tiff", ".btf", ".tf8", ".lsm", ".stk", ".zarr", ".ngff",
)


def is_supported(path: str | Path) -> bool:
    """Cheap suffix test, used to decide whether to intercept a drag-and-drop."""
    candidate = Path(path)
    if candidate.suffix.lower() in SUPPORTED_SUFFIXES:
        return True
    return candidate.is_dir() and (candidate / ".zattrs").exists()


class LoadError(Exception):
    """Raised when a file matched a reader but could not be read."""

    def __init__(self, path: Path, message: str):
        super().__init__(f"{path.name}: {message}")
        self.path = path
        self.message = message


def load_path(path: str | Path) -> list[LayerSpec]:
    """Read a single file or store, returning one :class:`LayerSpec` per channel."""
    candidate = Path(path)
    if not candidate.exists():
        raise LoadError(candidate, "file not found")

    for label, matches, reader in _READERS:
        try:
            matched = matches(candidate)
        except OSError:
            matched = False
        if not matched:
            continue
        logger.info("loading %s with the %s reader", candidate.name, label)
        try:
            specs = reader(candidate)
        except LoadError:
            raise
        except Exception as exc:
            raise LoadError(candidate, f"{label} reader failed: {exc}") from exc
        if not specs:
            raise LoadError(candidate, f"{label} reader produced no image layers")
        return specs

    raise LoadError(candidate, f"unsupported format (suffix {candidate.suffix or 'none'!r})")


def load_paths(paths: Iterable[str | Path]) -> tuple[list[LayerSpec], list[LoadError]]:
    """Load several paths, collecting failures instead of aborting the whole batch."""
    specs: list[LayerSpec] = []
    errors: list[LoadError] = []
    for path in paths:
        try:
            specs.extend(load_path(path))
        except LoadError as exc:
            logger.error("%s", exc)
            errors.append(exc)
        except Exception as exc:  # defensive: a reader raising something unexpected
            logger.exception("unexpected failure loading %s", path)
            errors.append(LoadError(Path(path), str(exc)))
    return specs, errors


def expand_inputs(paths: Sequence[str | Path]) -> list[Path]:
    """Turn command line / drop payload entries into a flat list of loadable paths.

    Plain directories are scanned one level deep for supported files, which makes
    dropping a folder of TIFFs onto the shortcut do the obvious thing.
    """
    out: list[Path] = []
    for entry in paths:
        candidate = Path(entry)
        if candidate.is_dir() and candidate.suffix.lower() not in (".zarr", ".ngff") and not (
            candidate / ".zattrs"
        ).exists():
            out.extend(sorted(child for child in candidate.iterdir() if is_supported(child)))
        else:
            out.append(candidate)
    return out


__all__ = [
    "COLORMAP_CYCLE",
    "FILE_DIALOG_FILTER",
    "LayerSpec",
    "LoadError",
    "SUPPORTED_SUFFIXES",
    "expand_inputs",
    "is_supported",
    "load_path",
    "load_paths",
]
