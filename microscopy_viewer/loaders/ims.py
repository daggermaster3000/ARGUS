"""Reader for Bitplane Imaris ``.ims`` files (HDF5 container).

Layout this reader understands::

    /DataSetInfo/Image           attrs: X Y Z ExtMin0..2 ExtMax0..2 Unit Name RecordingDate
    /DataSetInfo/Channel 0..N    attrs: Name Color ColorRange LSMEmissionWavelength ...
    /DataSetInfo/TimeInfo        attrs: DatasetTimePoints TimePoint1..N
    /DataSet/ResolutionLevel R/TimePoint T/Channel C/Data   (Z, Y, X, chunk-padded)

Arrays are wrapped in dask so large multi-timepoint stacks stay lazy, and the
resolution levels are handed to napari as a multiscale pyramid when present.
"""

from __future__ import annotations

import datetime as _dt
import re
import threading
from pathlib import Path
from typing import Any

import dask.array as da
import h5py
import numpy as np

from ..metadata import AcquisitionMetadata, ChannelMetadata
from ..utils import (
    MICRON,
    describe_dimensionality,
    get_logger,
    length_to_micron,
    parse_float,
    parse_float_list,
)
from .layer_spec import LayerSpec, channel_appearance

logger = get_logger("ims")

#: Open ``h5py.File`` handles are kept alive for the process lifetime because the
#: dask graphs read from them lazily long after the reader has returned.
_OPEN_FILES: list[h5py.File] = []

_DATE_PATTERNS = ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S")


def can_read(path: Path) -> bool:
    return path.suffix.lower() in (".ims", ".imaris")


# ---------------------------------------------------------------------------
# HDF5 attribute helpers
# ---------------------------------------------------------------------------


def _decode(value: Any) -> Any:
    """Imaris stores strings as arrays of single bytes; turn those back into text."""
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace").strip()
    if isinstance(value, np.ndarray):
        if value.dtype.kind == "S":
            return b"".join(value.ravel().tolist()).decode("utf-8", "replace").strip()
        if value.dtype.kind == "U":
            return "".join(value.ravel().tolist()).strip()
        if value.size == 1:
            return value.ravel()[0].item()
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _attrs(node) -> dict[str, Any]:
    """All attributes of an HDF5 node, decoded to plain Python values."""
    if node is None:
        return {}
    return {str(key): _decode(value) for key, value in node.attrs.items()}


def _attr(node, key: str, default=None) -> Any:
    if node is None or key not in node.attrs:
        return default
    return _decode(node.attrs[key])


def _sorted_children(group, prefix: str) -> list[tuple[int, Any]]:
    """Children named ``"<prefix> <n>"`` sorted by their trailing integer."""
    out: list[tuple[int, Any]] = []
    if group is None:
        return out
    for key in group.keys():
        match = re.fullmatch(rf"{re.escape(prefix)}\s*(\d+)", str(key))
        if match:
            out.append((int(match.group(1)), group[key]))
    out.sort(key=lambda item: item[0])
    return out


def _parse_date(text: str) -> _dt.datetime | None:
    text = (text or "").strip()
    if not text:
        return None
    for pattern in _DATE_PATTERNS:
        try:
            return _dt.datetime.strptime(text, pattern)
        except ValueError:
            continue
    return None


# ---------------------------------------------------------------------------
# Metadata extraction
# ---------------------------------------------------------------------------


def _voxel_size(image_info, sizes: dict[str, int]) -> dict[str, float | None]:
    """Voxel size in µm, derived from the dataset extents divided by voxel counts.

    ``ExtMin{i}``/``ExtMax{i}`` are indexed X=0, Y=1, Z=2 and expressed in the
    unit given by the ``Unit`` attribute (µm in practice, but we convert anyway).
    """
    unit = str(_attr(image_info, "Unit", MICRON) or MICRON)
    out: dict[str, float | None] = {"X": None, "Y": None, "Z": None}
    for index, axis in enumerate("XYZ"):
        low = parse_float(_attr(image_info, f"ExtMin{index}"))
        high = parse_float(_attr(image_info, f"ExtMax{index}"))
        count = sizes.get(axis)
        if low is None or high is None or not count:
            continue
        extent = length_to_micron(abs(high - low), unit)
        if extent is None or extent <= 0:
            continue
        out[axis] = extent / float(count)
    return out


def _stage_extent(image_info) -> tuple[float, float, float, float, float, float] | None:
    """Absolute stage extents as ``(x0, x1, y0, y1, z0, z1)`` in µm.

    The same ``ExtMin{i}``/``ExtMax{i}`` attributes the voxel size comes from, kept
    undifferenced. An overview mosaic and the fields imaged from it are written in
    one stage frame, so these coordinates are what lets a sample be marked on the
    overview at the place it was acquired.
    """
    unit = str(_attr(image_info, "Unit", MICRON) or MICRON)
    values: list[float] = []
    for index in range(3):
        for key in (f"ExtMin{index}", f"ExtMax{index}"):
            converted = length_to_micron(parse_float(_attr(image_info, key)), unit)
            if converted is None:
                return None
            values.append(converted)
    return (values[0], values[1], values[2], values[3], values[4], values[5])


def _time_interval(time_info) -> float | None:
    """Mean interval between the recorded timepoint timestamps, in seconds."""
    if time_info is None:
        return None
    stamps: list[_dt.datetime] = []
    for index in range(1, 4):  # first few timepoints are enough for the interval
        parsed = _parse_date(str(_attr(time_info, f"TimePoint{index}", "")))
        if parsed is None:
            break
        stamps.append(parsed)
    if len(stamps) < 2:
        return None
    deltas = [(b - a).total_seconds() for a, b in zip(stamps, stamps[1:])]
    deltas = [d for d in deltas if d > 0]
    return float(np.mean(deltas)) if deltas else None


def _read_channel(meta: AcquisitionMetadata, index: int, node) -> ChannelMetadata:
    """Populate ``meta``'s channel *index* from a ``/DataSetInfo/Channel N`` group.

    Harvesting into the real metadata object (rather than a throwaway) keeps
    file-level values that Imaris sometimes stores per channel, such as the
    objective or numerical aperture.
    """
    raw = _attrs(node)
    channel = meta.channel(index)
    channel.name = str(raw.pop("Name", "") or "").strip()

    colour = parse_float_list(raw.pop("Color", None))
    if len(colour) >= 3:
        channel.color = (colour[0], colour[1], colour[2])

    limits = parse_float_list(raw.pop("ColorRange", None))
    if len(limits) >= 2 and limits[1] > limits[0]:
        channel.contrast_limits = (limits[0], limits[1])

    # Rendering hints that would only clutter the metadata table.
    for noisy in ("ColorMode", "ColorOpacity", "GammaCorrection", "ColorTable", "ColorTableLength"):
        raw.pop(noisy, None)

    meta.harvest(raw, channel_index=index)
    return channel


def _build_metadata(handle: h5py.File, path: Path, sizes: dict[str, int]) -> AcquisitionMetadata:
    info = handle.get("DataSetInfo")
    image_info = info.get("Image") if info is not None else None
    time_info = info.get("TimeInfo") if info is not None else None

    meta = AcquisitionMetadata(file_path=path, file_format="Imaris (.ims)")
    meta.image_name = str(_attr(image_info, "Name", "") or path.stem)

    voxel = _voxel_size(image_info, sizes)
    meta.pixel_size_x_um = voxel["X"]
    meta.pixel_size_y_um = voxel["Y"]
    meta.z_step_um = voxel["Z"]
    meta.stage_extent = _stage_extent(image_info)
    meta.time_interval_s = _time_interval(time_info)

    # Channels first, so that file-level exposure/power values can fan out to them.
    for index, node in _sorted_children(info, "Channel"):
        _read_channel(meta, index, node)

    image_raw = _attrs(image_info)
    for noisy in ("X", "Y", "Z", "Unit", "Name", "NumberOfChannels", "ResampleDimensionX",
                  "ResampleDimensionY", "ResampleDimensionZ"):
        image_raw.pop(noisy, None)
    for index in range(3):
        image_raw.pop(f"ExtMin{index}", None)
        image_raw.pop(f"ExtMax{index}", None)
    meta.harvest(image_raw)

    # Sweep the remaining DataSetInfo groups for objective / NA / laser settings.
    if info is not None:
        for key in info.keys():
            if str(key) in ("Image", "TimeInfo") or re.fullmatch(r"Channel\s*\d+", str(key)):
                continue
            node = info[key]
            if not hasattr(node, "attrs"):
                continue
            meta.harvest({f"{key}/{name}": value for name, value in _attrs(node).items()})

    if time_info is not None:
        first = str(_attr(time_info, "TimePoint1", "") or "")
        if first and not meta.acquisition_date:
            meta.acquisition_date = first
    return meta


# ---------------------------------------------------------------------------
# Pixel data
# ---------------------------------------------------------------------------


def _real_size(dataset, image_info, axis: str, reference: tuple[int, int] | None = None) -> int:
    """Unpadded voxel count along *axis* for one ``Data`` dataset.

    Imaris pads the stored array up to the HDF5 chunk size, so the true extent
    has to come from the ``ImageSize*`` attributes. Plenty of real files omit
    those, in which case ``/DataSetInfo/Image`` gives the full-resolution size —
    but that is only right for level 0. For coarser levels *reference* carries
    level 0's ``(real, stored)`` size so the true size can be derived from how
    much this level was downsampled; without it a 1024-wide level of a 2040-wide
    image would keep 4 columns of padding.
    """
    axis_index = {"Z": 0, "Y": 1, "X": 2}[axis]
    stored = int(dataset.shape[axis_index])

    value = parse_float(_attr(dataset, f"ImageSize{axis}"))
    if value is None:
        if reference is not None:
            real_0, stored_0 = reference
            if stored_0 > 0:
                # Imaris halves each level, so the real size scales with the
                # stored size; round up so a partial voxel is not cut off.
                value = -(-real_0 * stored // stored_0)
        else:
            value = parse_float(_attr(image_info, axis))

    if value is None or value <= 0:
        return stored
    return int(min(int(value), stored))


def _lazy(dataset, lock: threading.Lock, crop: tuple[int, int, int]) -> da.Array:
    """Wrap one ``Data`` dataset as a cropped dask array."""
    chunks = dataset.chunks or "auto"
    array = da.from_array(dataset, chunks=chunks, lock=lock, name=f"ims-{dataset.name}-{id(dataset)}")
    z, y, x = crop
    return array[:z, :y, :x]


def _level_stack(
    level_group,
    image_info,
    lock: threading.Lock,
    n_channels: int,
    reference: tuple[tuple[int, int], ...] | None = None,
) -> tuple[list[da.Array], int, tuple[tuple[int, int], ...]]:
    """Per-channel ``(T, Z, Y, X)`` dask arrays for one resolution level.

    *reference* is level 0's per-axis ``(real, stored)`` size, used to undo chunk
    padding on coarser levels of files that omit the ``ImageSize*`` attributes.
    Returns the stacks, the timepoint count, and this level's own ``(real,
    stored)`` sizes.
    """
    timepoints = _sorted_children(level_group, "TimePoint")
    if not timepoints:
        raise ValueError(f"no TimePoint groups under {level_group.name}")

    per_channel: list[list[da.Array]] = [[] for _ in range(n_channels)]
    crop: tuple[int, ...] | None = None
    stored: tuple[int, ...] = ()

    for _, tp_group in timepoints:
        channels = _sorted_children(tp_group, "Channel")
        if not channels:
            raise ValueError(f"no Channel groups under {tp_group.name}")
        for channel_index, ch_group in channels:
            if channel_index >= n_channels:
                continue
            dataset = ch_group.get("Data")
            if dataset is None:
                raise ValueError(f"no Data dataset under {ch_group.name}")
            if crop is None:
                crop = tuple(
                    _real_size(
                        dataset,
                        image_info,
                        axis,
                        None if reference is None else reference[axis_index],
                    )
                    for axis_index, axis in enumerate("ZYX")
                )
                stored = tuple(int(n) for n in dataset.shape[:3])
            per_channel[channel_index].append(_lazy(dataset, lock, crop))

    stacks = []
    for channel_index, arrays in enumerate(per_channel):
        if not arrays:
            raise ValueError(f"channel {channel_index} missing from {level_group.name}")
        stacks.append(da.stack(arrays, axis=0))
    return stacks, len(timepoints), tuple(zip(crop, stored))


def _squeeze_plan(shape: tuple[int, ...], axes: str) -> list[int]:
    """Axis indices worth keeping: singleton ``T``/``Z`` axes give useless sliders.

    The plan is computed once from the full-resolution level and reused for every
    pyramid level, otherwise a level whose Z has collapsed to 1 would end up with
    a different number of dimensions than its parent.
    """
    return [i for i, axis in enumerate(axes) if not (axis in "TZ" and shape[i] == 1)]


def _apply_squeeze(array, keep: list[int], ndim: int):
    if len(keep) == ndim:
        return array
    return array[tuple(slice(None) if i in keep else 0 for i in range(ndim))]


def _is_downsample(new: tuple[int, ...], old: tuple[int, ...]) -> bool:
    """True when every axis of *new* is no larger than *old* and at least one shrank."""
    return all(n <= o for n, o in zip(new, old)) and any(n < o for n, o in zip(new, old))


def read(path: Path) -> list[LayerSpec]:
    """Read *path* and return one :class:`LayerSpec` per channel."""
    path = Path(path)
    handle = h5py.File(str(path), "r")
    _OPEN_FILES.append(handle)
    lock = threading.Lock()

    dataset_group = handle.get("DataSet")
    if dataset_group is None:
        handle.close()
        _OPEN_FILES.remove(handle)
        raise ValueError(f"{path.name} is HDF5 but has no /DataSet group — not an Imaris image")

    info = handle.get("DataSetInfo")
    image_info = info.get("Image") if info is not None else None

    levels = _sorted_children(dataset_group, "ResolutionLevel")
    if not levels:
        raise ValueError(f"{path.name} has no ResolutionLevel groups under /DataSet")

    n_channels = len(_sorted_children(info, "Channel")) if info is not None else 0
    if n_channels == 0:
        first_tp = _sorted_children(levels[0][1], "TimePoint")
        n_channels = len(_sorted_children(first_tp[0][1], "Channel")) if first_tp else 1

    # Build every level, then keep only a strictly-decreasing prefix so napari's
    # multiscale bookkeeping stays valid even for oddly-written files.
    per_level: list[list[da.Array]] = []
    n_timepoints = 1
    reference: tuple[tuple[int, int], ...] | None = None
    for _, level_group in levels:
        try:
            stacks, n_timepoints, sizes = _level_stack(
                level_group, image_info, lock, n_channels, reference
            )
        except ValueError as exc:
            logger.warning("skipping %s: %s", level_group.name, exc)
            break
        if reference is None:
            reference = sizes  # level 0 sets the yardstick for the coarser levels
        if per_level and not all(
            _is_downsample(new.shape[1:], old.shape[1:])
            for new, old in zip(stacks, per_level[-1])
        ):
            # Not a valid pyramid step, so stop here rather than hand napari a
            # level list it would mis-scale. Logged because it silently costs the
            # remaining levels, which is otherwise invisible.
            logger.warning(
                "%s: ignoring %s and coarser — shape %s does not downsample %s",
                path.name, level_group.name, stacks[0].shape[1:], per_level[-1][0].shape[1:],
            )
            break
        per_level.append(stacks)
    if not per_level:
        raise ValueError(f"{path.name}: no readable resolution level")

    full = per_level[0]
    sizes = {"X": full[0].shape[3], "Y": full[0].shape[2], "Z": full[0].shape[1], "T": n_timepoints}
    meta = _build_metadata(handle, path, sizes)
    for index in range(n_channels):
        meta.channel(index)  # make sure a block exists even for undocumented channels

    axes = "TZYX"
    scale = (
        meta.time_interval_s or 1.0,
        meta.z_step_um or 1.0,
        meta.pixel_size_y_um or 1.0,
        meta.pixel_size_x_um or 1.0,
    )
    units = ("s", MICRON, MICRON, MICRON)

    meta.axes = axes
    meta.shape = tuple(int(sizes[a]) for a in "TZYX")
    meta.dtype = str(full[0].dtype)
    meta.dimensionality = describe_dimensionality(
        "TZYXC", (sizes["T"], sizes["Z"], sizes["Y"], sizes["X"], n_channels)
    )

    multiscale = len(per_level) > 1
    keep = _squeeze_plan(full[0].shape, axes)
    layer_axes = "".join(axes[i] for i in keep)
    layer_scale = tuple(scale[i] for i in keep)
    layer_units = tuple(units[i] for i in keep)

    specs: list[LayerSpec] = []
    for index in range(n_channels):
        channel = meta.channel(index)
        arrays = [_apply_squeeze(level[index], keep, len(axes)) for level in per_level]
        data: Any = arrays if multiscale else arrays[0]
        colormap, color, blending = channel_appearance(
            channel.display_name, channel.color, index, n_channels
        )

        base = meta.image_name or path.stem
        name = f"{base} :: {channel.display_name}" if n_channels > 1 else base
        specs.append(
            LayerSpec(
                data=data,
                name=name,
                axes=layer_axes,
                scale=layer_scale,
                units=layer_units,
                metadata=meta,
                channel_index=index,
                colormap=colormap,
                color=color,
                blending=blending,
                multiscale=multiscale,
                contrast_limits=channel.contrast_limits,
            )
        )

    logger.info(
        "read %s: %d channel(s), axes=%s shape=%s levels=%d",
        path.name, n_channels, axes, meta.shape, len(per_level),
    )
    return specs
