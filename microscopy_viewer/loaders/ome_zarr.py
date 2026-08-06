"""Optional reader for OME-Zarr / NGFF stores.

Implemented directly against :mod:`zarr` and the NGFF ``.zattrs`` conventions so
no extra dependency is required. Handles the ``multiscales`` pyramid, the ``axes``
list (NGFF v0.3+) and the ``omero`` rendering block when present.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import dask.array as da
import zarr

from ..metadata import AcquisitionMetadata
from ..utils import (
    MICRON,
    describe_dimensionality,
    get_logger,
    guess_axes,
    length_to_micron,
    normalise_axes,
    parse_float,
    time_to_second,
)
from .layer_spec import LayerSpec, channel_appearance

logger = get_logger("ome_zarr")


def can_read(path: Path) -> bool:
    if path.suffix.lower() in (".zarr", ".ngff"):
        return True
    if path.is_dir() and ((path / ".zattrs").exists() or (path / "zarr.json").exists()):
        return True
    return False


def _group_class():
    """``zarr.Group`` moved between zarr 2 and 3; resolve it at call time."""
    return getattr(zarr, "Group", None) or zarr.hierarchy.Group  # type: ignore[attr-defined]


def _open_group(path: Path):
    try:
        node = zarr.open(str(path), mode="r")
    except Exception as exc:
        raise ValueError(f"{path.name} is not a readable Zarr store: {exc}") from exc
    if not isinstance(node, _group_class()):
        raise ValueError(f"{path.name} is a bare Zarr array, not an OME-Zarr image group")
    return node


def _multiscales(group) -> dict[str, Any] | None:
    entries = dict(group.attrs).get("multiscales")
    if isinstance(entries, list) and entries:
        return entries[0]
    return None


def _axes_from_ngff(entry: dict[str, Any], ndim: int) -> tuple[str, list[str]]:
    """Axis letters plus their declared units, from the NGFF ``axes`` list."""
    axes_spec = entry.get("axes")
    if not isinstance(axes_spec, list) or len(axes_spec) != ndim:
        return "", []
    letters, units = [], []
    for axis in axes_spec:
        if isinstance(axis, dict):
            letters.append(str(axis.get("name", "q")))
            units.append(str(axis.get("unit", "")))
        else:
            letters.append(str(axis))
            units.append("")
    return normalise_axes("".join(letters)), units


def _dataset_scale(dataset: dict[str, Any]) -> list[float] | None:
    """The ``scale`` coordinate transformation of one pyramid level."""
    for transform in dataset.get("coordinateTransformations", []) or []:
        if isinstance(transform, dict) and transform.get("type") == "scale":
            values = transform.get("scale")
            if isinstance(values, list):
                return [float(v) for v in values]
    return None


def _apply_ngff_scale(
    meta: AcquisitionMetadata, axes: str, scale: list[float], units: list[str]
) -> None:
    """Translate a level-0 NGFF scale vector into calibrated metadata fields."""
    unit_by_axis = dict(zip(axes, units))
    for index, axis in enumerate(axes):
        if index >= len(scale):
            break
        raw = scale[index]
        unit = unit_by_axis.get(axis, "")
        if axis == "X":
            meta.pixel_size_x_um = length_to_micron(raw, unit or MICRON)
        elif axis == "Y":
            meta.pixel_size_y_um = length_to_micron(raw, unit or MICRON)
        elif axis == "Z":
            meta.z_step_um = length_to_micron(raw, unit or MICRON)
        elif axis == "T":
            meta.time_interval_s = time_to_second(raw, unit or "s")


def _parse_omero(group, meta: AcquisitionMetadata) -> None:
    """Channel names, colours and display windows from the ``omero`` attribute block."""
    omero = dict(group.attrs).get("omero")
    if not isinstance(omero, dict):
        return
    if omero.get("name"):
        meta.image_name = str(omero["name"])
    for index, channel in enumerate(omero.get("channels", []) or []):
        if not isinstance(channel, dict):
            continue
        block = meta.channel(index)
        block.name = str(channel.get("label") or channel.get("name") or "").strip()
        colour = str(channel.get("color") or "")
        if len(colour) == 6:
            try:
                block.color = tuple(int(colour[i : i + 2], 16) / 255.0 for i in (0, 2, 4))  # type: ignore[assignment]
            except ValueError:
                pass
        window = channel.get("window")
        if isinstance(window, dict):
            low = parse_float(window.get("start"))
            high = parse_float(window.get("end"))
            if low is not None and high is not None and high > low:
                block.contrast_limits = (low, high)
        wavelength = parse_float(channel.get("emissionWave"))
        if wavelength:
            block.emission_nm = wavelength


def read(path: Path) -> list[LayerSpec]:
    """Read an OME-Zarr store and return one :class:`LayerSpec` per channel."""
    path = Path(path)
    group = _open_group(path)
    entry = _multiscales(group)

    if entry is None:
        # Not NGFF: fall back to the first array in the group.
        names = [key for key, value in group.arrays()]
        if not names:
            raise ValueError(f"{path.name} has no arrays and no multiscales metadata")
        levels = [da.from_zarr(group[names[0]])]
        axes_units: list[str] = []
        axes = guess_axes(levels[0].shape)
        level_scale = None
    else:
        datasets = entry.get("datasets") or []
        if not datasets:
            raise ValueError(f"{path.name}: multiscales entry has no datasets")
        levels = []
        for dataset in datasets:
            component = str(dataset.get("path", ""))
            try:
                levels.append(da.from_zarr(group[component]))
            except KeyError:
                logger.warning("%s: multiscales references missing array %r", path.name, component)
        if not levels:
            raise ValueError(f"{path.name}: none of the multiscales datasets could be opened")
        axes, axes_units = _axes_from_ngff(entry, levels[0].ndim)
        if not axes:
            axes = guess_axes(levels[0].shape)
            axes_units = [""] * len(axes)
        level_scale = _dataset_scale(datasets[0])

    meta = AcquisitionMetadata(
        image_name=path.stem,
        file_path=path,
        file_format="OME-Zarr",
    )
    if level_scale:
        _apply_ngff_scale(meta, axes, level_scale, axes_units or [""] * len(axes))
    _parse_omero(group, meta)
    meta.harvest(
        {key: value for key, value in dict(group.attrs).items() if key not in ("multiscales", "omero")}
    )

    shape = tuple(int(n) for n in levels[0].shape)
    n_channels = 1
    per_channel: list[list[Any]] = [levels]
    layer_axes = axes
    if "C" in axes:
        axis = axes.index("C")
        n_channels = shape[axis]
        layer_axes = axes[:axis] + axes[axis + 1 :]
        per_channel = []
        for index in range(n_channels):
            selector = tuple(index if i == axis else slice(None) for i in range(len(axes)))
            per_channel.append([level[selector] for level in levels])

    for index in range(n_channels):
        meta.channel(index)
    meta.axes = axes
    meta.shape = shape
    meta.dtype = str(levels[0].dtype)
    meta.dimensionality = describe_dimensionality(axes, shape)

    scale = tuple(
        {
            "X": meta.pixel_size_x_um or 1.0,
            "Y": meta.pixel_size_y_um or 1.0,
            "Z": meta.z_step_um or 1.0,
            "T": meta.time_interval_s or 1.0,
        }.get(axis, 1.0)
        for axis in layer_axes
    )
    units = tuple(MICRON if axis in "XYZ" else ("s" if axis == "T" else "") for axis in layer_axes)

    multiscale = len(levels) > 1
    specs: list[LayerSpec] = []
    for index, arrays in enumerate(per_channel):
        block = meta.channel(index)
        name = f"{meta.image_name} :: {block.display_name}" if n_channels > 1 else meta.image_name
        colormap, color, blending = channel_appearance(
            block.display_name, block.color, index, n_channels
        )
        specs.append(
            LayerSpec(
                data=arrays if multiscale else arrays[0],
                name=name,
                axes=layer_axes,
                scale=scale,
                units=units,
                metadata=meta,
                channel_index=index,
                colormap=colormap,
                color=color,
                blending=blending,
                multiscale=multiscale,
                contrast_limits=block.contrast_limits,
            )
        )
    logger.info("read %s: axes=%s shape=%s levels=%d", path.name, axes, shape, len(levels))
    return specs
