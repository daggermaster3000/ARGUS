"""Local disk cache for volumes that live on slow storage.

Source data usually sits on a NAS, where re-reading a stack every time the 3D
view is entered or a projection is recomputed dominates the wait. This writes the
array once to a local ``.npy`` and memory-maps it, so later reads come off the
local disk and the operating system's page cache instead of the network.

The cache is keyed by the source file's identity (path, size, modification time)
plus the array's own shape, dtype and role, so a changed file never resolves to a
stale entry. Nothing here is required for correctness: every failure degrades to
"use the original array".
"""

from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path
from typing import Any, Callable, Iterable

import math
import numpy as np

from .runtime import app_data_dir
from .utils import get_logger

logger = get_logger("volume_cache")

#: Total size the cache is allowed to reach before the oldest entries go.
DEFAULT_DISK_LIMIT = 32 * 1024**3  # 32 GB

#: Arrays smaller than this are not worth a round trip through the disk.
MIN_CACHE_BYTES = 32 * 1024**2  # 32 MB

#: Environment overrides.
DISK_LIMIT_ENV_VAR = "MICROSCOPY_VIEWER_CACHE_LIMIT"
DISABLE_ENV_VAR = "MICROSCOPY_VIEWER_NO_CACHE"

#: Rows written per pass when filling the cache file, so a huge stack never has
#: to be held in RAM in one piece.
SLAB_ROWS = 1


def enabled() -> bool:
    """False when the user has switched caching off."""
    return os.environ.get(DISABLE_ENV_VAR, "").strip().lower() not in ("1", "true", "yes")


def disk_limit() -> int:
    raw = os.environ.get(DISK_LIMIT_ENV_VAR)
    if raw:
        try:
            value = int(float(raw))
            if value > 0:
                return value
        except ValueError:
            logger.warning("ignoring invalid %s=%r", DISK_LIMIT_ENV_VAR, raw)
    return DEFAULT_DISK_LIMIT


def cache_root() -> Path:
    directory = app_data_dir() / "volume_cache"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def source_fingerprint(path: Any) -> str:
    """Identify a source file by path, size and modification time.

    Content hashing is out of the question for multi-gigabyte files on a network
    share, and size plus mtime is enough to notice a file being replaced.
    """
    if not path:
        return "unknown"
    try:
        stat = Path(path).stat()
        return f"{Path(path).as_posix()}|{stat.st_size}|{int(stat.st_mtime)}"
    except OSError:
        return str(path)


def cache_key(source: Any, role: str, shape: Iterable[int], dtype: Any) -> str:
    """Stable key for one cached array."""
    parts = [
        source_fingerprint(source),
        str(role),
        "x".join(str(int(n)) for n in shape),
        np.dtype(dtype).str,
    ]
    return hashlib.sha1("::".join(parts).encode("utf-8")).hexdigest()


def layer_key(source: Any, layer_name: str, level: int, shape: Iterable[int], dtype: Any) -> str:
    """Key for one pyramid level of one layer.

    The 3D view and time-series playback share it on purpose: both want the same
    array on local disk, so keying them separately would pull the same bytes
    across the network twice and store two copies of them. Whichever asks first
    does the copy; the other finds it already there.
    """
    return cache_key(source, f"{layer_name}|level{int(level)}", shape, dtype)


def cache_path(key: str) -> Path:
    return cache_root() / f"{key}.npy"


def load(key: str, shape: Iterable[int], dtype: Any) -> np.memmap | None:
    """Return the memory-mapped cache entry, or ``None`` if it is unusable."""
    path = cache_path(key)
    if not path.exists():
        return None
    try:
        array = np.load(path, mmap_mode="r")
    except Exception:
        logger.warning("discarding unreadable cache entry %s", path.name, exc_info=True)
        _remove(path)
        return None

    if tuple(array.shape) != tuple(int(n) for n in shape) or array.dtype != np.dtype(dtype):
        logger.warning("cache entry %s does not match the requested array; discarding", path.name)
        _remove(path)
        return None

    # Refresh the modification time so pruning treats this as recently used.
    try:
        os.utime(path, None)
    except OSError:
        pass
    return array


def store(
    array,
    key: str,
    progress: Callable[[int, int], None] | None = None,
) -> np.memmap | None:
    """Write *array* into the cache and return it memory-mapped.

    Written slab by slab along the leading axis so a stack far larger than RAM
    can be cached. A partially written file is removed rather than left behind to
    be mistaken for a complete entry.
    """
    if not enabled():
        return None

    shape = tuple(int(n) for n in array.shape)
    dtype = np.dtype(array.dtype)
    total_bytes = math.prod(int(n) for n in shape) * dtype.itemsize if shape else 0
    if total_bytes < MIN_CACHE_BYTES:
        return None

    path = cache_path(key)
    temporary = path.with_suffix(".npy.part")
    try:
        prune(reserve=total_bytes)
        handle = np.lib.format.open_memmap(temporary, mode="w+", dtype=dtype, shape=shape)
        try:
            leading = shape[0] if shape else 0
            step = max(SLAB_ROWS, 1)
            for start in range(0, leading, step):
                stop = min(start + step, leading)
                handle[start:stop] = np.asarray(array[start:stop])
                if progress is not None:
                    progress(stop, leading)
            handle.flush()
        finally:
            del handle
        os.replace(temporary, path)
    except Exception:
        logger.warning("could not cache %s", key, exc_info=True)
        _remove(temporary)
        return None

    logger.info("cached %s (%.1f MB) as %s", shape, total_bytes / 1e6, path.name)
    return load(key, shape, dtype)


def entries() -> list[tuple[Path, int, float]]:
    """``(path, size, mtime)`` for everything currently cached, newest first."""
    out: list[tuple[Path, int, float]] = []
    for path in cache_root().glob("*.npy"):
        try:
            stat = path.stat()
        except OSError:
            continue
        out.append((path, stat.st_size, stat.st_mtime))
    out.sort(key=lambda item: item[2], reverse=True)
    return out


def total_size() -> int:
    return sum(size for _path, size, _mtime in entries())


def prune(reserve: int = 0, limit: int | None = None) -> int:
    """Delete the least recently used entries until *reserve* bytes would fit.

    Returns the number of bytes freed.
    """
    limit = disk_limit() if limit is None else limit
    listed = entries()
    total = sum(size for _path, size, _mtime in listed)
    freed = 0
    # entries() is newest first, so walk it backwards to drop the oldest.
    for path, size, _mtime in reversed(listed):
        if total + reserve <= limit:
            break
        if _remove(path):
            total -= size
            freed += size
    if freed:
        logger.info("pruned %.1f MB from the volume cache", freed / 1e6)
    return freed


def clear() -> None:
    """Remove the whole cache."""
    try:
        shutil.rmtree(cache_root())
    except OSError:
        logger.warning("could not clear the volume cache", exc_info=True)


def _remove(path: Path) -> bool:
    try:
        path.unlink()
        return True
    except OSError:
        return False
