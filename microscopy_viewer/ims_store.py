"""Keep ROIs and label maps inside the ``.ims`` file they belong to.

An experiment is a folder of Imaris files, and the things derived from them — the
outline somebody drew round each brain, the nuclei Cellpose found — are per-file
results. Putting them in a sidecar folder works right up until the files are
moved, copied to a collaborator, or renamed, at which point the derived data is
orphaned silently. An ``.ims`` is an HDF5 file, HDF5 is a container, and there is
room in it.

**What this writes, and where.** Everything goes under one top-level group,
``/ARGUS``, beside Imaris's own ``/DataSet`` and ``/DataSetInfo``. Nothing
Imaris wrote is read back, modified or deleted. Imaris ignores groups it does not
recognise, so a file with an ``/ARGUS`` group still opens and behaves normally.

**What this deliberately does not do** is write Imaris Surfaces or Spots objects.
Those live in the undocumented ``Scene8`` structure; producing one that Imaris
will load means guessing at a private format inside a file holding irreplaceable
acquisition data, and getting it wrong corrupts the file rather than failing. So
the regions saved here are visible to this viewer and not to Imaris. If Imaris
has to see them, export instead.

**Handles.** :mod:`microscopy_viewer.loaders.ims` holds a read handle on every
file it opened, and HDF5 refuses to open a file for writing while one is open for
reading — in the same process as much as across processes. :func:`writable`
releases the reader's handle before opening for write, which invalidates any
layer still backed by that file, so the caller has to take those off screen
first. :func:`can_write` says in advance whether that is going to be necessary.
"""

from __future__ import annotations

import datetime as _dt
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

from .utils import get_logger

logger = get_logger("ims_store")

#: The one group this module writes into. Everything below is ours.
ARGUS_GROUP = "ARGUS"
ROI_GROUP = f"{ARGUS_GROUP}/ROIs"
LABEL_GROUP = f"{ARGUS_GROUP}/Labels"

#: Bumped if the layout below ever changes incompatibly.
FORMAT_VERSION = 1

#: Suffixes worth trying to open. An ``.h5`` written by something else is still
#: a perfectly good container for this.
HDF5_SUFFIXES = (".ims", ".imaris", ".h5", ".hdf5")


@dataclass
class StoredRoi:
    """One named outline, in world micrometres, as saved in the file."""

    name: str
    #: (N, 2) world Y/X vertices in µm — the same coordinates
    #: :class:`microscopy_viewer.regions.Region` uses, so the two interchange.
    vertices_um: np.ndarray
    shape_type: str = "polygon"
    #: Free-form notes kept with the shape: which channel it was drawn on, say.
    attrs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.vertices_um = np.asarray(self.vertices_um, dtype=float).reshape(-1, 2)


# ---------------------------------------------------------------------------
# Opening
# ---------------------------------------------------------------------------


def is_container(path: str | Path) -> bool:
    """Whether *path* looks like an HDF5 file this can write into."""
    candidate = Path(path)
    if candidate.suffix.lower() not in HDF5_SUFFIXES or not candidate.is_file():
        return False
    try:
        import h5py

        return bool(h5py.is_hdf5(str(candidate)))
    except Exception:
        return False


def can_write(path: str | Path) -> tuple[bool, str]:
    """``(ok, reason)`` — whether a write would succeed right now.

    Asked before offering the button rather than discovered halfway through a
    batch of thirty files.
    """
    candidate = Path(path)
    if not is_container(candidate):
        return False, f"{candidate.name} is not an HDF5/Imaris file."
    try:
        if not candidate.stat().st_mode & 0o200:
            return False, f"{candidate.name} is read-only."
    except OSError as exc:
        return False, f"{candidate.name}: {exc}"
    return True, ""


@contextmanager
def writable(path: str | Path, release_reader: bool = True) -> Iterator[Any]:
    """Open *path* for writing, giving back the reader's handle first.

    The release is the whole point of this being a context manager rather than a
    bare ``h5py.File(..., "r+")``: without it the open fails with *"file is
    already open for read-only"* on any file the viewer has loaded, which is
    every file the user has just been drawing on.
    """
    import h5py

    from .loaders import ims as ims_reader

    target = Path(path)
    if release_reader:
        released = ims_reader.release(target)
        if released:
            logger.info(
                "closed %d reader handle(s) on %s to write into it", released, target.name
            )

    handle = h5py.File(str(target), "r+")
    try:
        yield handle
    finally:
        handle.close()


def _group(handle, path: str):
    """``require_group`` with the format stamp, so a reader knows what it has."""
    group = handle.require_group(path)
    root = handle.require_group(ARGUS_GROUP)
    root.attrs["format_version"] = FORMAT_VERSION
    root.attrs["written_by"] = "ARGUS microscopy viewer"
    root.attrs["written_at"] = _dt.datetime.now().isoformat(timespec="seconds")
    return group


def _text(value: Any) -> str:
    """HDF5 attribute text, decoded from whatever h5py hands back."""
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    if isinstance(value, np.ndarray):
        return "".join(_text(item) for item in value.ravel())
    return str(value)


# ---------------------------------------------------------------------------
# ROIs
# ---------------------------------------------------------------------------


def save_rois(path: str | Path, rois: Sequence[StoredRoi], replace: bool = True) -> int:
    """Write *rois* into the file. Returns how many were written.

    With *replace* the previous set is removed first, which is what the panel
    wants: the shapes on screen are the truth, and a shape that was deleted there
    should not survive in the file.
    """
    entries = [roi for roi in rois if roi.vertices_um.shape[0] >= 3]
    with writable(path) as handle:
        if replace and ROI_GROUP in handle:
            del handle[ROI_GROUP]
        group = _group(handle, ROI_GROUP)
        for index, roi in enumerate(entries):
            # Indexed keys, not names: a region can be called "cerebellum left"
            # and HDF5 names cannot contain "/", which anatomical names do
            # sometimes. The name is an attribute.
            key = f"roi_{index:03d}"
            if key in group:
                del group[key]
            dataset = group.create_dataset(key, data=roi.vertices_um.astype(np.float64))
            dataset.attrs["name"] = roi.name
            dataset.attrs["shape_type"] = roi.shape_type
            dataset.attrs["units"] = "um"
            dataset.attrs["axes"] = "YX"
            for attribute, value in roi.attrs.items():
                try:
                    dataset.attrs[str(attribute)] = value
                except Exception:
                    dataset.attrs[str(attribute)] = str(value)
    logger.info("wrote %d ROI(s) into %s", len(entries), Path(path).name)
    return len(entries)


def load_rois(path: str | Path) -> list[StoredRoi]:
    """Read back what :func:`save_rois` wrote. Empty when there is nothing."""
    import h5py

    if not is_container(path):
        return []
    rois: list[StoredRoi] = []
    try:
        with h5py.File(str(path), "r") as handle:
            group = handle.get(ROI_GROUP)
            if group is None:
                return []
            for key in sorted(group):
                dataset = group[key]
                rois.append(
                    StoredRoi(
                        name=_text(dataset.attrs.get("name", key)),
                        vertices_um=np.asarray(dataset[()], dtype=float),
                        shape_type=_text(dataset.attrs.get("shape_type", "polygon")),
                    )
                )
    except OSError as exc:
        # Usually the reader holding the file; reading through its own handle
        # would need the dataset path, and this is not worth that coupling.
        logger.info("could not read ROIs from %s: %s", Path(path).name, exc)
        return []
    return rois


def has_rois(path: str | Path) -> bool:
    return bool(load_rois(path))


# ---------------------------------------------------------------------------
# Label maps
# ---------------------------------------------------------------------------


def save_labels(
    path: str | Path,
    key: str,
    masks: np.ndarray,
    voxel_size_um: Sequence[float] = (),
    attrs: dict[str, Any] | None = None,
) -> str:
    """Store a label map under ``/ARGUS/Labels/<key>``. Returns the key used.

    Compressed, because a label map is mostly zeros and mostly runs — gzip takes
    a 500 Mvoxel int32 volume down by an order of magnitude, and the alternative
    is doubling the size of every file in the experiment.
    """
    labels = np.asarray(masks)
    safe = sanitise_key(key)
    with writable(path) as handle:
        group = _group(handle, LABEL_GROUP)
        if safe in group:
            del group[safe]
        dataset = group.create_dataset(
            safe,
            data=labels,
            compression="gzip",
            compression_opts=4,
            # Chunked by plane: the usual read is one plane at a time, and a
            # whole-volume chunk would have to be decompressed to see any of it.
            chunks=_chunks(labels.shape),
        )
        dataset.attrs["voxel_size_um"] = np.asarray(
            tuple(float(v) for v in voxel_size_um), dtype=np.float64
        )
        dataset.attrs["n_objects"] = int(labels.max()) if labels.size else 0
        dataset.attrs["written_at"] = _dt.datetime.now().isoformat(timespec="seconds")
        for attribute, value in (attrs or {}).items():
            try:
                dataset.attrs[str(attribute)] = value
            except Exception:
                dataset.attrs[str(attribute)] = str(value)
    logger.info("wrote labels %r (%s) into %s", safe, labels.shape, Path(path).name)
    return safe


def _chunks(shape: Sequence[int]) -> tuple[int, ...]:
    dims = tuple(int(n) for n in shape)
    if len(dims) < 2:
        return dims
    plane = tuple(min(n, 256) for n in dims[-2:])
    return (1,) * (len(dims) - 2) + plane


def sanitise_key(key: str) -> str:
    """A name usable as an HDF5 dataset name: no separators, never empty."""
    cleaned = "".join(character if character.isalnum() or character in "-_. " else "_"
                      for character in str(key)).strip()
    return cleaned or "labels"


def list_labels(path: str | Path) -> list[str]:
    """Names of the label maps stored in the file."""
    import h5py

    if not is_container(path):
        return []
    try:
        with h5py.File(str(path), "r") as handle:
            group = handle.get(LABEL_GROUP)
            return sorted(group) if group is not None else []
    except OSError:
        return []


def load_labels(path: str | Path, key: str) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Read one stored label map back, with its attributes."""
    import h5py

    if not is_container(path):
        return None, {}
    try:
        with h5py.File(str(path), "r") as handle:
            group = handle.get(LABEL_GROUP)
            if group is None or key not in group:
                return None, {}
            dataset = group[key]
            attrs = {name: dataset.attrs[name] for name in dataset.attrs}
            return np.asarray(dataset[()]), attrs
    except OSError as exc:
        logger.info("could not read labels from %s: %s", Path(path).name, exc)
        return None, {}


def summary(path: str | Path) -> str:
    """One line for the panel: what this file already carries."""
    rois = len(load_rois(path))
    labels = list_labels(path)
    parts = []
    if rois:
        parts.append(f"{rois} ROI(s)")
    if labels:
        parts.append(f"{len(labels)} label map(s)")
    return ", ".join(parts)
