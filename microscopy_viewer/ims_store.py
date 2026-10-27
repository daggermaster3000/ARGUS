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

**Sources that are not HDF5.** A TIFF or OME-TIFF has nowhere to put any of
this: rewriting a gigabyte of somebody's acquisition to append a label map is
exactly the risk the paragraph above declines to take with ``Scene8``. Those get
a sidecar instead — ``<file name>.argus.h5`` beside the image, holding the same
``/ARGUS`` group — and :func:`store_path` is the one place that decides which of
the two a path means. The sidecar has to travel with the image, which is the
orphaning problem the first paragraph describes; it is the price of the format.
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

#: Appended to the whole file name of a source that is not HDF5, so
#: ``fish.ome.tiff`` keeps its results in ``fish.ome.tiff.argus.h5``. The whole
#: name rather than the stem: ``fish.tif`` and ``fish.ims`` in one folder would
#: otherwise share a sidecar.
SIDECAR_SUFFIX = ".argus.h5"


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


def uses_sidecar(path: str | Path) -> bool:
    """Whether results for *path* go beside it rather than inside it.

    True for an image this program reads that is not HDF5. Anything else — a
    text file, a spreadsheet — gets no sidecar and is refused as before.
    """
    from .loaders import is_supported

    candidate = Path(path)
    return candidate.suffix.lower() not in HDF5_SUFFIXES and is_supported(candidate)


def store_path(path: str | Path) -> Path:
    """The HDF5 file holding the ``/ARGUS`` group for *path*.

    The file itself when it is HDF5, its sidecar otherwise. The sidecar need not
    exist yet: reading from one that does not is "nothing stored", and writing
    creates it.
    """
    candidate = Path(path)
    if not uses_sidecar(candidate):
        return candidate
    return candidate.with_name(candidate.name + SIDECAR_SUFFIX)


def _readable_store(path: str | Path) -> Path | None:
    """The store for *path* when there is one to read, else ``None``."""
    store = store_path(path)
    return store if is_container(store) else None


def can_write(path: str | Path) -> tuple[bool, str]:
    """``(ok, reason)`` — whether a write would succeed right now.

    Asked before offering the button rather than discovered halfway through a
    batch of thirty files.
    """
    candidate = Path(path)
    if uses_sidecar(candidate):
        return _can_write_sidecar(candidate)
    if not is_container(candidate):
        return False, f"{candidate.name} is not an HDF5/Imaris file."
    try:
        if not candidate.stat().st_mode & 0o200:
            return False, f"{candidate.name} is read-only."
    except OSError as exc:
        return False, f"{candidate.name}: {exc}"
    return True, ""


def _can_write_sidecar(source: Path) -> tuple[bool, str]:
    """:func:`can_write` for a source whose results go in a sidecar.

    The image itself may be read-only — it is never touched. What has to be
    writable is the sidecar, or the folder when there is no sidecar yet.
    """
    import os

    if not source.exists():
        return False, f"{source.name} does not exist."
    store = store_path(source)
    if store.exists():
        if not is_container(store):
            return False, f"{store.name} exists and is not an HDF5 file."
        if not os.access(store, os.W_OK):
            return False, f"{store.name} is read-only."
        return True, ""
    if not os.access(store.parent, os.W_OK):
        return False, f"cannot create {store.name}: the folder is read-only."
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
    if uses_sidecar(target):
        # Nothing to release: no reader holds the sidecar, and the image it sits
        # beside is not opened here at all, so its layers stay valid.
        handle = h5py.File(str(store_path(target)), "a")
        try:
            yield handle
        finally:
            handle.close()
        return

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


#: Outlines a panel other than Brain regions stores beside the hand-drawn
#: regions: the junction finder's band (``junction.BAND_ROI_NAME``). They are
#: not anatomy, so the Brain regions layer does not show them (:func:`anatomy`),
#: and saving the regions on screen does not delete them.
OTHER_PANELS_ROIS = ("Fusion band",)


def anatomy(rois: Sequence[StoredRoi]) -> list[StoredRoi]:
    """*rois* without the outlines other panels store (:data:`OTHER_PANELS_ROIS`)."""
    return [roi for roi in rois if roi.name not in OTHER_PANELS_ROIS]


def save_rois(path: str | Path, rois: Sequence[StoredRoi], replace: bool = True) -> int:
    """Write *rois* into the file. Returns how many were written.

    With *replace* the previous set is removed first, which is what the panel
    wants: the shapes on screen are the truth, and a shape that was deleted there
    should not survive in the file. Outlines stored by another panel
    (:data:`OTHER_PANELS_ROIS`, e.g. the junction band) survive a replace unless
    *rois* carries one of the same name — they are not on that panel's screen.
    """
    entries = [roi for roi in rois if roi.vertices_um.shape[0] >= 3]
    if replace:
        names = {roi.name for roi in entries}
        entries += [roi for roi in load_rois(path)
                    if roi.name in OTHER_PANELS_ROIS and roi.name not in names]
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
    logger.info("wrote %d ROI(s) into %s", len(entries), store_path(path).name)
    return len(entries)


def load_rois(path: str | Path) -> list[StoredRoi]:
    """Read back what :func:`save_rois` wrote. Empty when there is nothing."""
    import h5py

    store = _readable_store(path)
    if store is None:
        return []
    rois: list[StoredRoi] = []
    try:
        with h5py.File(str(store), "r") as handle:
            group = handle.get(ROI_GROUP)
            if group is None:
                return []
            for key in sorted(group):
                dataset = group[key]
                standard = {"name", "shape_type", "units", "axes"}
                rois.append(
                    StoredRoi(
                        name=_text(dataset.attrs.get("name", key)),
                        vertices_um=np.asarray(dataset[()], dtype=float),
                        shape_type=_text(dataset.attrs.get("shape_type", "polygon")),
                        # Whatever else was stored with it — who drew it, say.
                        attrs={k: _text(v) for k, v in dataset.attrs.items() if k not in standard},
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
    labels = _compact(np.asarray(masks))
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
    logger.info("wrote labels %r (%s) into %s", safe, labels.shape, store_path(path).name)
    return safe


def delete_labels(path: str | Path, key: str) -> bool:
    """Remove one stored label map. Returns whether there was one to remove."""
    safe = sanitise_key(key)
    if safe not in list_labels(path):
        return False
    with writable(path) as handle:
        del handle[LABEL_GROUP][safe]
    logger.info("removed labels %r from %s", safe, store_path(path).name)
    return True


def _compact(labels: np.ndarray) -> np.ndarray:
    """The same label map in the narrowest integer type that still holds it.

    Cellpose hands back ``int32`` whether it found four objects or forty
    thousand, and a whole-brain stack is a few hundred million voxels: storing
    those as ``uint16`` halves what has to be compressed and what the file grows
    by. Only ever narrows, and only when every label fits — the alternative is
    relabelling objects by overflow, which would be silent and wrong.
    """
    if not np.issubdtype(labels.dtype, np.integer) or labels.size == 0:
        return labels
    highest = int(labels.max())
    if labels.min() < 0:
        return labels
    for candidate in (np.uint8, np.uint16, np.uint32):
        if highest <= np.iinfo(candidate).max:
            return labels.astype(candidate, copy=False)
    return labels


def label_shape(path: str | Path, key: str) -> tuple[int, ...]:
    """Shape of a stored label map, without reading it.

    What a write is verified against: a 4 GB volume read back to be compared
    would cost as much as producing it did, and a dataset that did not survive
    being written is the wrong shape or absent, not subtly different.
    """
    import h5py

    store = _readable_store(path)
    if store is None:
        return ()
    try:
        with h5py.File(str(store), "r") as handle:
            group = handle.get(LABEL_GROUP)
            if group is None or key not in group:
                return ()
            return tuple(int(n) for n in group[key].shape)
    except OSError as exc:
        logger.info("could not read the shape of %r in %s: %s", key, Path(path).name, exc)
        return ()


def label_attrs(path: str | Path, key: str) -> dict[str, Any]:
    """The attributes of a stored label map, without reading the map. Empty when absent."""
    import h5py

    store = _readable_store(path)
    if store is None:
        return {}
    try:
        with h5py.File(str(store), "r") as handle:
            group = handle.get(LABEL_GROUP)
            if group is None or key not in group:
                return {}
            dataset = group[key]
            return {name: dataset.attrs[name] for name in dataset.attrs}
    except OSError as exc:
        logger.info("could not read the attributes of %r in %s: %s", key, Path(path).name, exc)
        return {}


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

    store = _readable_store(path)
    if store is None:
        return []
    try:
        with h5py.File(str(store), "r") as handle:
            group = handle.get(LABEL_GROUP)
            return sorted(group) if group is not None else []
    except OSError:
        return []


def load_labels(path: str | Path, key: str) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Read one stored label map back, with its attributes."""
    import h5py

    store = _readable_store(path)
    if store is None:
        return None, {}
    try:
        with h5py.File(str(store), "r") as handle:
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
