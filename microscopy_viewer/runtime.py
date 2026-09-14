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

#: Intel OpenMP's "let a second copy load" switch, read once when the runtime
#: initialises.
OPENMP_DUPLICATE_VAR = "KMP_DUPLICATE_LIB_OK"


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


def allow_duplicate_openmp() -> bool:
    """Stop a second Intel OpenMP runtime from aborting the process.

    Cellpose pulls in torch, which ships its own ``libiomp5md.dll``; numpy and
    scipy are already using the one from conda's MKL. When the second copy
    initialises, Intel's runtime prints *OMP: Error #15* and calls ``abort()`` —
    a C-level kill, not a Python exception, so nothing upstream can catch it and
    the whole process dies mid-run.

    Whether it fires depends on load order. Importing napari first happens to
    arrange the libraries acceptably, which is why the GUI escapes it, but a bare
    ``import microscopy_viewer.segmentation`` — the test suite, or any script —
    aborts reliably.

    The two supported cures are a single OpenMP build across the whole
    environment, or this switch. Rebuilding the environment is not something an
    import should attempt, so this sets the switch. The documented risk is
    performance loss and, in principle, wrong results from two runtimes sharing
    thread state; in practice it is what torch-on-conda installations everywhere
    run with. An explicit setting is always left alone.

    Returns True if this call set the variable.
    """
    if os.environ.get(OPENMP_DUPLICATE_VAR):
        return False
    os.environ[OPENMP_DUPLICATE_VAR] = "TRUE"
    return True


#: Marker on the sinks installed below, so logging can tell a real console from
#: the placeholder and not bother attaching a handler to /dev/null.
NULL_SINK_FLAG = "mv_null_sink"


def ensure_std_streams() -> tuple[str, ...]:
    """Give the process somewhere to write when ``pythonw.exe`` gave it nothing.

    A GUI process launched by the desktop shortcut runs under ``pythonw.exe``,
    which has no console: ``sys.stdout`` and ``sys.stderr`` are both ``None``.
    Our own code checks for that, but libraries do not. Cellpose draws a ``tqdm``
    progress bar from ``utils.stitch3D``, tqdm writes to ``sys.stdout``, and the
    run dies with ``AttributeError: 'NoneType' object has no attribute 'write'``
    — from inside the segmentation, after the slow part, with nothing to show for
    it. "2D + stitch" is where it bites because stitching is what calls tqdm.

    The streams are pointed at the null device rather than a file: what would be
    captured is progress-bar redraws, and real diagnostics already go through
    :func:`microscopy_viewer.utils.setup_logging` to the log file. Anything the
    process was given for real is left alone.

    Returns the names of the streams that had to be replaced.
    """
    replaced: list[str] = []
    for name in ("stdout", "stderr"):
        if getattr(sys, name, None) is not None:
            continue
        try:
            sink = open(os.devnull, "w", encoding="utf-8", buffering=1)
        except OSError:  # pragma: no cover - no null device to open
            continue
        setattr(sink, NULL_SINK_FLAG, True)
        setattr(sys, name, sink)
        # Libraries reach for the pristine originals too; those are None as well.
        if getattr(sys, f"__{name}__", None) is None:
            setattr(sys, f"__{name}__", sink)
        replaced.append(name)
    return tuple(replaced)
