"""GPU budget and volume-cache checks. No display required.

Run with::

    python tests/test_rendering.py
"""

from __future__ import annotations

import gc
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from microscopy_viewer import gpu, rendering, volume_cache  # noqa: E402

_failures: list[str] = []


def check(condition: bool, message: str) -> None:
    if condition:
        print(f"  ok   {message}")
    else:
        print(f"  FAIL {message}")
        _failures.append(message)


class _Level:
    def __init__(self, shape, dtype=np.uint16):
        self.shape = shape
        self.dtype = np.dtype(dtype)


# ---------------------------------------------------------------------------


class _FakeLayer:
    """Just enough of a napari Image layer for :func:`rendering.plain_level`."""

    def __init__(self, levels, scale, level=0, name="layer"):
        self.data = levels
        self.scale = scale
        self.name = name
        self.multiscale = True
        self._data_level = level


def test_voxel_counts_do_not_overflow() -> None:
    print("voxel counts past 2**31")

    # numpy's default integer is 32-bit on Windows below numpy 2, so np.prod on a
    # shape this size wraps to a negative number and every size check downstream
    # of it passes. That handed napari a volume it could not upload.
    check(
        rendering.displayed_voxels((512, 2048, 2048)) == 2_147_483_648,
        f"a 2.1 Gvoxel stack counts as itself ({rendering.displayed_voxels((512, 2048, 2048))})",
    )
    check(
        rendering.displayed_voxels((4096, 4096, 4096)) == 68_719_476_736,
        f"and so does a 68 Gvoxel one ({rendering.displayed_voxels((4096, 4096, 4096))})",
    )
    check(
        rendering.displayed_voxels((10, 512, 2048, 2048)) == 2_147_483_648,
        "leading axes are still ignored — napari slices them before the upload",
    )


def test_plain_level_for_plugins() -> None:
    print("a plain copy of one pyramid level")

    levels = [_Level((1200, 1600)), _Level((600, 800)), _Level((300, 400))]
    layer = _FakeLayer(levels, scale=(0.25, 0.25), level=0)

    data, scale, level = rendering.plain_level(layer)
    check(level == 0 and data is levels[0], "by default it takes the level on screen")
    check(scale == (0.25, 0.25), f"level 0 keeps the layer's own scale ({scale})")

    layer._data_level = 2
    data, scale, level = rendering.plain_level(layer)
    check(level == 2 and data is levels[2], "following the viewer down the pyramid")
    check(
        scale == (1.0, 1.0),
        f"a level shrunk 4x is stretched 4x so it sits on its parent ({scale})",
    )

    data, scale, level = rendering.plain_level(layer, level=1)
    check(level == 1 and scale == (0.5, 0.5), f"an explicit level is honoured ({scale})")
    check(
        rendering.plain_level(layer, level=99)[2] == 2,
        "a level the pyramid does not have is clamped to the coarsest",
    )

    # Anisotropic levels, which is what a plate pyramid is: Y and X halve, Z does not.
    volume = [_Level((40, 1200, 1600)), _Level((40, 600, 800))]
    _data, scale, _level = rendering.plain_level(
        _FakeLayer(volume, scale=(2.0, 0.25, 0.25), level=1)
    )
    check(
        scale == (2.0, 0.5, 0.5),
        f"only the axes that were downsampled are stretched ({scale})",
    )

    try:
        rendering.plain_level(_FakeLayer([], scale=(1.0,)))
        check(False, "an empty layer is refused")
    except ValueError:
        check(True, "an empty layer is refused rather than indexed into")


def test_budget_scales_with_vram() -> None:
    print("the voxel budget follows the GPU, not a constant")
    os.environ.pop(gpu.BUDGET_ENV_VAR, None)

    # An RTX 4090 with ~21 GB free, which is what this machine reports.
    big = gpu.GpuLimits(
        renderer="NVIDIA GeForce RTX 4090", max_3d_texture=16384,
        total_vram_bytes=25_153_536 * 1024, available_vram_bytes=22_257_944 * 1024, queried=True,
    )
    budget = gpu.voxel_budget(big, bytes_per_voxel=2)
    check(budget > 1_000_000_000, f"a 24 GB card gets over a billion voxels ({budget:,})")
    check(
        budget > 8 * gpu.FALLBACK_VOXEL_BUDGET,
        f"far more than the old fixed {gpu.FALLBACK_VOXEL_BUDGET:,} ({budget:,})",
    )

    # A 16-bit volume fits twice as many voxels as a 32-bit one.
    check(
        gpu.voxel_budget(big, bytes_per_voxel=4) == budget // 2,
        "the budget halves when each voxel is twice the size",
    )

    small = gpu.GpuLimits(max_3d_texture=2048, available_vram_bytes=512 * 1024**2, queried=True)
    modest = gpu.voxel_budget(small, bytes_per_voxel=2)
    check(modest >= gpu.MIN_VOXEL_BUDGET, f"a small GPU still gets the floor ({modest:,})")

    unknown = gpu.GpuLimits()
    check(
        gpu.voxel_budget(unknown) == gpu.FALLBACK_VOXEL_BUDGET,
        "an unqueryable GPU falls back to the safe constant",
    )

    os.environ[gpu.BUDGET_ENV_VAR] = "5000"
    try:
        check(gpu.voxel_budget(big) == 5000, "the environment override wins")
    finally:
        os.environ.pop(gpu.BUDGET_ENV_VAR, None)


def test_texture_axis_limit() -> None:
    print("per-axis 3D texture limit")
    limits = gpu.GpuLimits(max_3d_texture=2048, available_vram_bytes=8 * 1024**3, queried=True)
    check(gpu.fits_texture_limits((100, 2048, 2048), limits), "2048 on an axis is allowed")
    check(not gpu.fits_texture_limits((100, 4096, 2048), limits), "4096 is not, on a 2048 limit")

    # A volume can be small enough overall yet too long in one direction.
    levels = [_Level((4, 4096, 4096)), _Level((4, 2048, 2048)), _Level((4, 1024, 1024))]
    budget = 10**12  # effectively unlimited
    check(
        rendering.choose_level(levels, budget, max_axis=2048) == 1,
        "an over-long level is skipped even when it fits the budget",
    )
    check(
        rendering.choose_level(levels, budget, max_axis=None) == 0,
        "without a texture limit the finest level is used",
    )
    check(
        rendering.choose_level(levels, budget, max_axis=512) == 2,
        "a tight texture limit steps down further",
    )


def test_real_stack_now_uses_full_resolution() -> None:
    """The 17x2040x2040 deconvolved stacks must render at level 0 on this GPU."""
    print("a real z-stack at full resolution")
    levels = [
        _Level((17, 2040, 2040)), _Level((17, 1020, 1020)), _Level((17, 510, 510)),
        _Level((17, 255, 255)), _Level((17, 128, 128)), _Level((6, 64, 64)),
    ]
    big = gpu.GpuLimits(max_3d_texture=16384, available_vram_bytes=22_257_944 * 1024, queried=True)
    budget = gpu.voxel_budget(big, bytes_per_voxel=2)
    check(rendering.choose_level(levels, budget, big.max_3d_texture) == 0, "level 0 chosen")

    # 256 x 2048 x 2048 is 1.07G voxels, inside the ~1.42G budget.
    deep = [_Level((256, 2048, 2048)), _Level((128, 1024, 1024))]
    check(rendering.choose_level(deep, budget, big.max_3d_texture) == 0, "a 256-plane 2048² stack fits")

    # Twice that is 2.1G voxels, past the budget, so it steps down one level.
    deeper = [_Level((512, 2048, 2048)), _Level((256, 1024, 1024))]
    check(
        rendering.choose_level(deeper, budget, big.max_3d_texture) == 1,
        "a 512-plane stack steps down rather than failing the upload",
    )

    # Something genuinely enormous still steps down rather than failing.
    huge = [_Level((4096, 4096, 4096)), _Level((2048, 2048, 2048)), _Level((512, 512, 512))]
    chosen = rendering.choose_level(huge, budget, big.max_3d_texture)
    check(chosen > 0, f"an oversized volume steps down (level {chosen})")


def test_cache_round_trip() -> None:
    print("volume cache round trip")
    os.environ.pop(volume_cache.DISABLE_ENV_VAR, None)
    with tempfile.TemporaryDirectory() as directory:
        os.environ["LOCALAPPDATA"] = directory
        try:
            volume_cache.clear()
            rng = np.random.default_rng(0)
            # 42 MB, comfortably over MIN_CACHE_BYTES so it is actually written.
            volume = rng.integers(0, 4000, size=(80, 512, 512)).astype(np.uint16)
            check(
                volume.nbytes > volume_cache.MIN_CACHE_BYTES,
                f"the fixture is worth caching ({volume.nbytes / 1e6:.0f} MB)",
            )

            key = volume_cache.cache_key("Z:/nas/file.ims", "layer|level0", volume.shape, volume.dtype)
            check(volume_cache.load(key, volume.shape, volume.dtype) is None, "nothing cached yet")

            stored = volume_cache.store(volume, key)
            check(stored is not None, "the volume was cached")
            check(isinstance(stored, np.memmap), f"it comes back memory-mapped ({type(stored).__name__})")
            check(np.array_equal(np.asarray(stored), volume), "contents survive the round trip")
            check(stored.dtype == volume.dtype and stored.shape == volume.shape, "shape and dtype preserved")

            again = volume_cache.load(key, volume.shape, volume.dtype)
            check(again is not None, "a second load hits the cache")
            check(np.array_equal(np.asarray(again), volume), "and returns the same data")

            # A mismatched request must not silently return the wrong array.
            check(
                volume_cache.load(key, (1, 2, 3), volume.dtype) is None,
                "a shape mismatch invalidates the entry",
            )

            # Small arrays are not worth caching.
            tiny_key = volume_cache.cache_key("x", "tiny", (2, 2), np.uint16)
            check(volume_cache.store(np.zeros((2, 2), np.uint16), tiny_key) is None, "tiny arrays are skipped")

            # Windows will not delete a file that is still mapped, so drop the
            # references before the temporary directory is torn down.
            del stored, again
            gc.collect()
        finally:
            os.environ.pop("LOCALAPPDATA", None)


def test_cache_keys_track_the_source() -> None:
    print("cache keys follow the source file")
    with tempfile.TemporaryDirectory() as directory:
        source = Path(directory) / "data.ims"
        source.write_bytes(b"a" * 2048)
        first = volume_cache.cache_key(source, "role", (4, 4), np.uint16)

        # Same file, same key.
        check(first == volume_cache.cache_key(source, "role", (4, 4), np.uint16), "stable for one file")
        # Different role, shape or dtype -> different entry.
        check(first != volume_cache.cache_key(source, "other", (4, 4), np.uint16), "role changes the key")
        check(first != volume_cache.cache_key(source, "role", (8, 8), np.uint16), "shape changes the key")
        check(first != volume_cache.cache_key(source, "role", (4, 4), np.uint8), "dtype changes the key")

        # Rewriting the file must invalidate it: size and mtime both feed the key.
        os.utime(source, (0, 0))
        stale = volume_cache.cache_key(source, "role", (4, 4), np.uint16)
        source.write_bytes(b"b" * 4096)
        os.utime(source, (10_000, 10_000))
        check(stale != volume_cache.cache_key(source, "role", (4, 4), np.uint16), "a modified file re-keys")

        check(volume_cache.cache_key(None, "role", (4, 4), np.uint16), "a missing source still yields a key")


def test_cache_pruning() -> None:
    print("cache pruning")
    with tempfile.TemporaryDirectory() as directory:
        os.environ["LOCALAPPDATA"] = directory
        try:
            volume_cache.clear()
            root = volume_cache.cache_root()
            sizes = []
            for index in range(4):
                path = root / f"{index:040x}.npy"
                np.save(path, np.zeros((256, 256), np.uint16))  # 128 KB each
                os.utime(path, (index, index))  # oldest first
                sizes.append(path.stat().st_size)

            check(len(volume_cache.entries()) == 4, f"four entries present ({len(volume_cache.entries())})")
            check(volume_cache.total_size() == sum(sizes), "total size adds up")

            # Allow only two entries' worth; the two oldest must go.
            freed = volume_cache.prune(reserve=0, limit=sizes[0] * 2)
            remaining = {path.name for path, _size, _mtime in volume_cache.entries()}
            check(freed > 0, f"pruning freed {freed} bytes")
            check(len(remaining) == 2, f"two entries survive ({len(remaining)})")
            check(f"{3:040x}.npy" in remaining, "the newest entry is kept")
            check(f"{0:040x}.npy" not in remaining, "the oldest entry is gone")

            volume_cache.clear()
            check(not volume_cache.entries(), "clear empties the cache")
        finally:
            os.environ.pop("LOCALAPPDATA", None)


def test_cache_can_be_disabled() -> None:
    print("cache opt-out")
    os.environ[volume_cache.DISABLE_ENV_VAR] = "1"
    try:
        check(not volume_cache.enabled(), "the environment variable switches it off")
        check(
            volume_cache.store(np.zeros((64, 512, 512), np.uint16), "key") is None,
            "and nothing is written",
        )
    finally:
        os.environ.pop(volume_cache.DISABLE_ENV_VAR, None)
    check(volume_cache.enabled(), "on again once it is unset")


def main() -> int:
    for test in (
        test_budget_scales_with_vram,
        test_voxel_counts_do_not_overflow,
        test_plain_level_for_plugins,
        test_texture_axis_limit,
        test_real_stack_now_uses_full_resolution,
        test_cache_round_trip,
        test_cache_keys_track_the_source,
        test_cache_pruning,
        test_cache_can_be_disabled,
    ):
        test()
        print()

    if _failures:
        print(f"{len(_failures)} check(s) failed:")
        for failure in _failures:
            print(f"  - {failure}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
