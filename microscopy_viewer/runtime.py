"""Environment fixes that must happen before napari (and therefore numba) is imported.

Keep this module free of third-party imports: it runs from
``microscopy_viewer/__init__.py``, ahead of everything else.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

#: Environment variable numba reads, once, when ``numba.core.config`` is imported.
NUMBA_CACHE_VAR = "NUMBA_CACHE_DIR"


def app_data_dir() -> Path:
    """Per-user directory for logs and caches, where the platform expects it.

    ``LOCALAPPDATA`` is always set on Windows and ``XDG_CACHE_HOME`` is the
    explicit override on Linux, so either one wins when present. Otherwise this
    falls back to the platform's own cache location rather than dropping a
    directory in the middle of the user's home.
    """
    override = os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_CACHE_HOME")
    if override:
        return Path(override) / "MicroscopyViewer"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "MicroscopyViewer"
    return Path.home() / ".cache" / "MicroscopyViewer"


def preload_torch_libraries() -> Path | None:
    """Load torch's core DLL before Qt does, on Windows.

    Importing PyQt *before* torch leaves torch unable to load its own libraries::

        OSError: [WinError 1114] A dynamic link library (DLL) initialization
        routine failed. Error loading "…\\torch\\lib\\c10.dll"

    which reaches the user as "Segmentation needs the cellpose package" — cellpose
    imports torch, the import raises, and the panel cannot tell that apart from
    the package being absent. Claiming the DLL directory and loading ``c10.dll``
    first is enough to avoid it, and costs milliseconds: torch itself is not
    imported here, only its library directory registered.

    Returns the directory used, or ``None`` when this does not apply.
    """
    if sys.platform != "win32" or not hasattr(os, "add_dll_directory"):
        return None
    try:
        import importlib.util

        spec = importlib.util.find_spec("torch")
    except (ImportError, ValueError):  # a broken or absent install
        return None
    locations = list(getattr(spec, "submodule_search_locations", None) or []) if spec else []
    if not locations:
        return None

    directory = Path(locations[0]) / "lib"
    core = directory / "c10.dll"
    if not core.exists():
        return None
    try:
        os.add_dll_directory(str(directory))
        import ctypes

        ctypes.CDLL(str(core))
    except OSError:
        # Nothing is lost: this is the state the process would have been in.
        return None
    return directory


def configure_numba_cache() -> Path | None:
    """Point numba's JIT cache at a writable per-user directory.

    napari JIT-compiles part of its colormap handling with numba, which caches the
    result next to the installed package. When napari lives in a read-only prefix
    (a system-wide conda install, for instance) that cache cannot be written, and
    importing ``napari.utils.colormaps`` stalls for minutes on every launch while
    numba retries. Redirecting the cache makes the compile happen once.

    Returns the directory used, or ``None`` if the variable was already set or the
    directory could not be created.
    """
    if os.environ.get(NUMBA_CACHE_VAR):
        return None
    directory = app_data_dir() / "numba_cache"
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    os.environ[NUMBA_CACHE_VAR] = str(directory)
    return directory
