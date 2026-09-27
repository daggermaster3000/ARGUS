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


def source_of(layer) -> str:
    """The file a layer was read from, or ``""``."""
    meta = layer.metadata.get("mv_metadata")
    return str(getattr(meta, "file_path", "") or "") if meta is not None else ""


def close_sample(viewer, path) -> int:
    """Take a file's layers off screen and release the reader's handle. Returns the count.

    Both halves are needed before writing. Removing the layers drops the dask
    graphs, but the reader holds its ``h5py.File`` open for the life of the
    process, and HDF5 will not open for writing what is open for reading.
    """
    from ..loaders import ims as ims_reader

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
