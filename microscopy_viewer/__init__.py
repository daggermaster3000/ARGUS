"""Microscopy Viewer — a customised napari viewer for rapid image inspection.

Public entry points::

    from microscopy_viewer import launch
    launch(["cells.ims"])            # open the GUI with a file loaded

or from a shell::

    python -m microscopy_viewer cells.ims
"""

from __future__ import annotations

from .runtime import configure_numba_cache, preload_torch_libraries

# Must run before anything imports napari: see runtime.configure_numba_cache.
configure_numba_cache()
# Must run before anything imports Qt: see runtime.preload_torch_libraries.
preload_torch_libraries()

__version__ = "1.0.0"
__all__ = ["MicroscopyViewer", "launch", "main", "__version__"]


def __getattr__(name: str):
    """Import the GUI lazily so ``import microscopy_viewer`` needs no Qt."""
    if name in ("MicroscopyViewer", "launch"):
        from . import app

        return getattr(app, name)
    if name == "main":
        from .__main__ import main

        return main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
