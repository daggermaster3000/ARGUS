"""What the GPU can actually hold, so 3D views are not capped by a guess.

A volume is uploaded to the GPU whole, so the useful limits are the per-axis
maximum 3D texture size and the free video memory. Both are read from the live
OpenGL context; when that cannot be done the conservative fallbacks below apply.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from .utils import get_logger

logger = get_logger("gpu")

# OpenGL enums, spelled out because vispy does not expose all of them by name.
_GL_MAX_3D_TEXTURE_SIZE = 0x8073
#: NVIDIA's NVX_gpu_memory_info; values are in KiB.
_GL_GPU_MEM_TOTAL_KB = 0x9047
_GL_GPU_MEM_AVAILABLE_KB = 0x9049
#: ATI_meminfo: returns four values, the first being free texture memory in KiB.
_GL_TEXTURE_FREE_MEMORY_ATI = 0x87FC

#: Used when the GPU cannot be interrogated. 2048 is the smallest 3D texture size
#: any GL 3.x part is required to support, so it is always safe.
FALLBACK_MAX_3D_TEXTURE = 2048
FALLBACK_VOXEL_BUDGET = 128_000_000

#: Fraction of *available* video memory a single volume may occupy. The rest is
#: left for napari's own textures, the framebuffer and everything else on the card.
VRAM_FRACTION = 0.25

#: Multiplier on the stored size, covering the staging copy made during upload and
#: any widening the driver does to a supported internal format.
UPLOAD_OVERHEAD = 2.0

#: Never go below this, even on a very small GPU: it is roughly a 512^3 volume.
MIN_VOXEL_BUDGET = 128_000_000

#: Environment override, in voxels. Set it to pin the budget explicitly.
BUDGET_ENV_VAR = "MICROSCOPY_VIEWER_3D_VOXEL_BUDGET"


@dataclass
class GpuLimits:
    """Capabilities of the OpenGL context napari is rendering through."""

    renderer: str = ""
    vendor: str = ""
    max_3d_texture: int = FALLBACK_MAX_3D_TEXTURE
    total_vram_bytes: int = 0
    available_vram_bytes: int = 0
    queried: bool = False

    def describe(self) -> str:
        if not self.queried:
            return "GPU limits unknown — using conservative defaults"
        parts = [self.renderer or "unknown GPU", f"max 3D texture {self.max_3d_texture}"]
        if self.available_vram_bytes:
            parts.append(f"{self.available_vram_bytes / 1e9:.1f} GB video memory free")
        return ", ".join(parts)


_CACHED: GpuLimits | None = None


def _read_parameter(gl, enum: int):
    try:
        return gl.glGetParameter(enum)
    except Exception:
        return None


def _first_int(value) -> int:
    """Some of these queries return a 4-tuple; take the leading value."""
    if value is None:
        return 0
    if isinstance(value, (list, tuple)):
        return int(value[0]) if value else 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _query(gl) -> GpuLimits:
    limits = GpuLimits(queried=True)
    limits.renderer = str(_read_parameter(gl, gl.GL_RENDERER) or "")
    limits.vendor = str(_read_parameter(gl, gl.GL_VENDOR) or "")

    max_3d = _first_int(_read_parameter(gl, _GL_MAX_3D_TEXTURE_SIZE))
    limits.max_3d_texture = max_3d if max_3d > 0 else FALLBACK_MAX_3D_TEXTURE

    total_kb = _first_int(_read_parameter(gl, _GL_GPU_MEM_TOTAL_KB))
    available_kb = _first_int(_read_parameter(gl, _GL_GPU_MEM_AVAILABLE_KB))
    if not available_kb:  # AMD reports free memory through a different extension
        available_kb = _first_int(_read_parameter(gl, _GL_TEXTURE_FREE_MEMORY_ATI))
    limits.total_vram_bytes = total_kb * 1024
    limits.available_vram_bytes = available_kb * 1024
    return limits


def query_limits(refresh: bool = False) -> GpuLimits:
    """Read the GPU's limits, reusing the previous answer unless *refresh*.

    Prefers the context napari is already rendering in; if there is none, a
    throwaway hidden canvas is created just long enough to ask.
    """
    global _CACHED
    if _CACHED is not None and not refresh:
        return _CACHED

    limits = GpuLimits()
    try:
        from vispy import gloo
        from vispy.gloo import gl

        canvas = gloo.get_current_canvas()
        if canvas is not None:
            limits = _query(gl)
        else:
            from vispy import app as vispy_app

            temporary = vispy_app.Canvas(show=False)
            try:
                temporary.create_native()
                with temporary:
                    limits = _query(gl)
            finally:
                temporary.close()
    except Exception:
        logger.info("could not query the GPU; using conservative defaults", exc_info=True)
        limits = GpuLimits()

    logger.info("GPU: %s", limits.describe())
    _CACHED = limits
    return limits


def voxel_budget(limits: GpuLimits | None = None, bytes_per_voxel: int = 2) -> int:
    """How many voxels of a single volume may be uploaded at once.

    Derived from free video memory rather than a fixed constant: on a card with
    tens of gigabytes a fixed few hundred megabytes would needlessly force a
    coarse pyramid level, which is exactly the blocky 3D view we are avoiding.
    """
    override = os.environ.get(BUDGET_ENV_VAR)
    if override:
        try:
            value = int(float(override))
            if value > 0:
                return value
        except ValueError:
            logger.warning("ignoring invalid %s=%r", BUDGET_ENV_VAR, override)

    limits = limits if limits is not None else query_limits()
    available = limits.available_vram_bytes or limits.total_vram_bytes
    if not available:
        return FALLBACK_VOXEL_BUDGET

    usable = available * VRAM_FRACTION
    per_voxel = max(int(bytes_per_voxel), 1) * UPLOAD_OVERHEAD
    return max(int(usable / per_voxel), MIN_VOXEL_BUDGET)


def fits_texture_limits(shape, limits: GpuLimits | None = None) -> bool:
    """Whether a volume's spatial axes are each within the 3D texture limit."""
    limits = limits if limits is not None else query_limits()
    spatial = tuple(int(n) for n in tuple(shape)[-3:])
    return all(n <= limits.max_3d_texture for n in spatial)
