"""Writing into a sample that is open in the viewer.

HDF5 will not open for writing a file that is open for reading, and the reader
keeps each ``.ims`` open for the life of the process. So a panel that writes
into a sample — regions, manual cell counts — takes its layers off screen,
releases the handle, writes, and opens it again, keeping the view where it was.
"""

from __future__ import annotations

from pathlib import Path

from ..utils import get_logger

logger = get_logger("sample_files")


#: A layer read out of a sample's store (a stored label map): the sample's path.
SAMPLE_KEY = "argus_sample"
#: A layer a panel made *for* a sample — drawn lines, counted cells, a
#: segmentation, dots — : the sample's path. It goes when the sample goes.
DERIVED_KEY = "argus_derived_from"
#: A layer that belongs to whatever is on screen and to no file — the QC
#: picture under review. It goes whenever the sample on screen changes.
TRANSIENT_KEY = "argus_transient"


def source_of(layer) -> str:
    """The file a layer was read from, or ``""``."""
    meta = layer.metadata.get("mv_metadata")
    return str(getattr(meta, "file_path", "") or "") if meta is not None else ""


def sample_of(layer, derived: bool = True) -> str:
    """The sample a layer belongs to: read from it, read out of its store, or
    (with *derived*) made from it by a panel. ``""`` for none."""
    metadata = getattr(layer, "metadata", None) or {}
    found = source_of(layer) or str(metadata.get(SAMPLE_KEY, "") or "")
    if not found and derived:
        found = str(metadata.get(DERIVED_KEY, "") or "")
    return found


def tie_to_sample(layer, source) -> None:
    """Mark a layer a panel made as belonging to *source*, so it leaves with it.

    With no source — an image not read from a file — the layer is marked as
    belonging to whatever is on screen instead, which is just as gone when
    another sample is opened.
    """
    try:
        if source:
            layer.metadata[DERIVED_KEY] = str(source)
        else:
            layer.metadata[TRANSIENT_KEY] = True
    except Exception:
        logger.debug("could not tie %s to its sample", getattr(layer, "name", layer), exc_info=True)


def clear_samples(viewer, paths=None) -> int:
    """Take samples off screen with everything that belongs to them.

    *paths* are the samples to clear, or ``None`` for every one: their channels,
    the label maps read out of them, and every layer a panel made for them; and
    in either case the layers that belong to no file (:data:`TRANSIENT_KEY`).
    The reader handles are given back. Returns the number of layers removed.
    """
    from ..loaders import release

    def key(path) -> str:
        try:
            return str(Path(path).resolve())
        except OSError:
            return str(path)

    wanted = None if paths is None else {key(path) for path in paths}
    sources: set[str] = set()
    removed = 0
    for layer in list(viewer.layers):
        sample = sample_of(layer)
        transient = bool((getattr(layer, "metadata", None) or {}).get(TRANSIENT_KEY))
        if sample and (wanted is None or key(sample) in wanted):
            sources.add(key(sample))
        elif not transient:
            continue
        viewer.layers.remove(layer)
        removed += 1
    for source in sources:
        try:
            release(Path(source))
        except Exception:
            logger.debug("could not release %s", source, exc_info=True)
    return removed


def close_sample(viewer, path) -> int:
    """Take a file's layers off screen and release the reader's handle. Returns the count.

    Both halves are needed before writing. Removing the layers drops the dask
    graphs, but the reader holds its ``h5py.File`` open for the life of the
    process, and HDF5 will not open for writing what is open for reading.

    A sample whose results go in a sidecar (a TIFF) is left exactly as it is:
    the write never opens the image, so there is nothing to close or reopen.
    """
    from .. import ims_store
    from ..loaders import ims as ims_reader

    if ims_store.uses_sidecar(path):
        return 0

    target = str(Path(path).resolve())
    removed = 0
    for layer in list(viewer.layers):
        source = source_of(layer)
        if not source:
            continue
        try:
            if str(Path(source).resolve()) != target:
                continue
        except OSError:
            continue
        viewer.layers.remove(layer)
        removed += 1
    ims_reader.release(path)
    return removed


def view_state(viewer) -> dict:
    """Camera and slider position, so reopening a sample does not move the view."""
    camera = viewer.camera
    return {
        "center": tuple(camera.center),
        "zoom": float(camera.zoom),
        "angles": tuple(camera.angles),
        "step": tuple(viewer.dims.current_step),
    }


def restore_view(viewer, state: dict) -> None:
    """Put the camera and sliders back where :func:`view_state` found them."""
    try:
        camera = viewer.camera
        camera.center = state["center"]
        camera.zoom = state["zoom"]
        camera.angles = state["angles"]
        step = state["step"]
        if len(step) == viewer.dims.ndim:
            viewer.dims.current_step = step
    except Exception:
        logger.debug("could not restore the view", exc_info=True)
