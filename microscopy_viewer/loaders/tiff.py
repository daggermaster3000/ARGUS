"""Readers for plain TIFF, ImageJ TIFF and OME-TIFF, built on :mod:`tifffile`.

Calibration is looked for in three places, in decreasing order of trust:
OME-XML, the ImageJ description block, and finally the baseline TIFF
resolution tags.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import dask.array as da
import numpy as np
import tifffile

from ..metadata import AcquisitionMetadata, apply_channel_names
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

logger = get_logger("tiff")

SUFFIXES = (".tif", ".tiff", ".ome.tif", ".ome.tiff", ".btf", ".tf8", ".lsm", ".stk")


def can_read(path: Path) -> bool:
    name = path.name.lower()
    return any(name.endswith(suffix) for suffix in SUFFIXES)


# ---------------------------------------------------------------------------
# OME-XML
# ---------------------------------------------------------------------------


def _strip_ns(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _find(root: ET.Element, name: str) -> ET.Element | None:
    for element in root.iter():
        if _strip_ns(element.tag) == name:
            return element
    return None


def _find_all(root: ET.Element, name: str) -> list[ET.Element]:
    return [element for element in root.iter() if _strip_ns(element.tag) == name]


def _parse_ome(xml: str, meta: AcquisitionMetadata) -> None:
    """Fill *meta* from an OME-XML document."""
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        logger.warning("OME-XML could not be parsed; falling back to TIFF tags")
        return

    image = _find(root, "Image")
    if image is not None:
        meta.acquisition_date = meta.acquisition_date or _text(image, "AcquisitionDate")
        name = image.get("Name")
        if name:
            meta.image_name = name

    pixels = _find(root, "Pixels")
    if pixels is not None:
        meta.pixel_size_x_um = length_to_micron(
            parse_float(pixels.get("PhysicalSizeX")), pixels.get("PhysicalSizeXUnit", MICRON)
        )
        meta.pixel_size_y_um = length_to_micron(
            parse_float(pixels.get("PhysicalSizeY")), pixels.get("PhysicalSizeYUnit", MICRON)
        )
        meta.z_step_um = length_to_micron(
            parse_float(pixels.get("PhysicalSizeZ")), pixels.get("PhysicalSizeZUnit", MICRON)
        )
        meta.time_interval_s = time_to_second(
            parse_float(pixels.get("TimeIncrement")), pixels.get("TimeIncrementUnit", "s")
        )

    objective = _find(root, "ObjectiveSettings")
    lens = _find(root, "Objective")
    if lens is not None:
        magnification = parse_float(lens.get("NominalMagnification"))
        parts = [lens.get("Model") or lens.get("Manufacturer") or ""]
        if magnification:
            parts.append(f"{magnification:g}x")
        immersion = lens.get("Immersion")
        if immersion and immersion.lower() not in ("other", "unknown"):
            parts.append(immersion)
        meta.objective = " ".join(p for p in parts if p).strip()
        meta.numerical_aperture = parse_float(lens.get("LensNA"))
    if meta.numerical_aperture is None and objective is not None:
        meta.numerical_aperture = parse_float(objective.get("CorrectionCollar"))

    for index, channel in enumerate(_find_all(root, "Channel")):
        block = meta.channel(index)
        block.name = (channel.get("Name") or channel.get("Fluor") or "").strip()
        block.excitation_nm = length_to_micron(
            parse_float(channel.get("ExcitationWavelength")),
            channel.get("ExcitationWavelengthUnit", "nm"),
        )
        block.emission_nm = length_to_micron(
            parse_float(channel.get("EmissionWavelength")),
            channel.get("EmissionWavelengthUnit", "nm"),
        )
        # length_to_micron returns µm; wavelengths are reported in nm.
        if block.excitation_nm is not None:
            block.excitation_nm *= 1e3
        if block.emission_nm is not None:
            block.emission_nm *= 1e3
        illumination = channel.get("IlluminationType")
        if illumination and illumination.lower() not in ("other", "unknown"):
            block.extra.setdefault("Illumination", illumination)

    # Exposure lives on Plane elements; take the first one seen per channel.
    for plane in _find_all(root, "Plane"):
        exposure = parse_float(plane.get("ExposureTime"))
        if exposure is None:
            continue
        channel_index = int(parse_float(plane.get("TheC")) or 0)
        block = meta.channel(channel_index)
        if block.exposure_ms is None:
            seconds = time_to_second(exposure, plane.get("ExposureTimeUnit", "s"))
            block.exposure_ms = None if seconds is None else seconds * 1e3

    for laser in _find_all(root, "Laser"):
        wavelength = parse_float(laser.get("Wavelength"))
        power = parse_float(laser.get("Power"))
        label = f"{wavelength:g} nm" if wavelength else (laser.get("Model") or "laser")
        for block in meta.channels or [meta.channel(0)]:
            if block.laser_name == "":
                block.laser_name = label
            if block.laser_power is None and power is not None:
                block.laser_power = power
                block.laser_power_unit = "mW"


def _text(parent: ET.Element, name: str) -> str:
    for child in parent:
        if _strip_ns(child.tag) == name and child.text:
            return child.text.strip()
    return ""


# ---------------------------------------------------------------------------
# ImageJ and baseline TIFF tags
# ---------------------------------------------------------------------------


def _parse_imagej(tf: tifffile.TiffFile, meta: AcquisitionMetadata) -> None:
    """Fill *meta* from ImageJ's description block plus the resolution tags."""
    ij = tf.imagej_metadata or {}
    unit = str(ij.get("unit") or "")

    spacing = parse_float(ij.get("spacing"))
    if spacing is not None:
        meta.z_step_um = length_to_micron(spacing, unit or MICRON)

    interval = parse_float(ij.get("finterval"))
    if interval is not None:
        meta.time_interval_s = time_to_second(interval, "s")

    labels = ij.get("Labels")
    if isinstance(labels, (list, tuple)) and labels:
        apply_channel_names(meta, [str(label) for label in labels])

    # ImageJ dumps the acquisition log into "Info" as key=value lines.
    info = ij.get("Info")
    if isinstance(info, str) and info:
        pairs: dict[str, Any] = {}
        for line in info.splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                pairs[key.strip()] = value.strip()
        if pairs:
            meta.harvest(pairs)

    _parse_resolution_tags(tf, meta, unit)


def _parse_resolution_tags(tf: tifffile.TiffFile, meta: AcquisitionMetadata, unit_hint: str = "") -> None:
    """Derive XY pixel size from ``XResolution``/``YResolution``/``ResolutionUnit``."""
    if meta.pixel_size_x_um is not None and meta.pixel_size_y_um is not None:
        return
    page = tf.pages[0]
    tags = getattr(page, "tags", {})

    unit_name = unit_hint
    resolution_unit = tags.get("ResolutionUnit")
    if not unit_name and resolution_unit is not None:
        # 1 = none, 2 = inch, 3 = centimetre
        unit_name = {1: "", 2: "inch", 3: "cm"}.get(int(getattr(resolution_unit.value, "value", resolution_unit.value)), "")
    if not unit_name:
        return

    for axis, tag_name in (("x", "XResolution"), ("y", "YResolution")):
        tag = tags.get(tag_name)
        if tag is None:
            continue
        value = tag.value
        if isinstance(value, tuple) and len(value) == 2 and value[1]:
            per_unit = value[0] / value[1]
        else:
            per_unit = parse_float(value)
        if not per_unit:
            continue
        # The tag holds pixels per unit; invert to get the size of one pixel.
        size = length_to_micron(1.0 / per_unit, unit_name)
        if size is None or size <= 0:
            continue
        if axis == "x" and meta.pixel_size_x_um is None:
            meta.pixel_size_x_um = size
        elif axis == "y" and meta.pixel_size_y_um is None:
            meta.pixel_size_y_um = size


# ---------------------------------------------------------------------------
# Pixel data
# ---------------------------------------------------------------------------


def _series_arrays(tf: tifffile.TiffFile, series) -> tuple[list[Any], bool]:
    """Dask arrays for *series*, as a pyramid when the file stores sub-resolutions."""
    n_levels = len(getattr(series, "levels", []) or [])
    try:
        if n_levels > 1:
            arrays = [da.from_zarr(level.aszarr()) for level in series.levels]
            if all(_shrinks(a.shape, b.shape) for a, b in zip(arrays[1:], arrays)):
                return arrays, True
            return [arrays[0]], False
        return [da.from_zarr(series.aszarr())], False
    except Exception:
        logger.info("zarr view unavailable for %s; reading eagerly", tf.filename, exc_info=True)
        return [np.asarray(series.asarray())], False


def _shrinks(new: tuple[int, ...], old: tuple[int, ...]) -> bool:
    return len(new) == len(old) and all(n <= o for n, o in zip(new, old)) and any(
        n < o for n, o in zip(new, old)
    )


def _split_channels(arrays: list[Any], axes: str) -> tuple[list[list[Any]], str, int]:
    """Split a channel axis out into one array list per channel."""
    if "C" not in axes:
        return [arrays], axes, 1
    axis = axes.index("C")
    count = int(arrays[0].shape[axis])
    remaining = axes[:axis] + axes[axis + 1 :]
    per_channel = []
    for index in range(count):
        selector = tuple(index if i == axis else slice(None) for i in range(len(axes)))
        per_channel.append([array[selector] for array in arrays])
    return per_channel, remaining, count


#: Lazy dask graphs read from the file handle long after :func:`read` returns, so
#: the open ``TiffFile`` objects are parked here for the process lifetime.
_OPEN_FILES: list[tifffile.TiffFile] = []


def read(path: Path) -> list[LayerSpec]:
    """Read *path* and return one :class:`LayerSpec` per channel."""
    path = Path(path)
    tf = tifffile.TiffFile(str(path))
    _OPEN_FILES.append(tf)
    try:
        series = tf.series[0]
        raw_axes = str(series.axes or "")
        arrays, multiscale = _series_arrays(tf, series)
        shape = tuple(int(n) for n in arrays[0].shape)

        axes = normalise_axes(raw_axes) if len(raw_axes) == len(shape) else guess_axes(shape)
        if len(axes) != len(shape):  # last resort, keep dimensions aligned
            axes = guess_axes(shape)

        meta = AcquisitionMetadata(
            image_name=path.stem,
            file_path=path,
            file_format="OME-TIFF" if tf.is_ome else ("ImageJ TIFF" if tf.is_imagej else "TIFF"),
        )
        if tf.is_ome and tf.ome_metadata:
            _parse_ome(tf.ome_metadata, meta)
            _parse_resolution_tags(tf, meta)
        elif tf.is_imagej:
            _parse_imagej(tf, meta)
        else:
            _parse_resolution_tags(tf, meta)
        meta.image_name = meta.image_name or path.stem

        per_channel, layer_axes, n_channels = _split_channels(arrays, axes)
        for index in range(n_channels):
            meta.channel(index)

        meta.axes = axes
        meta.shape = shape
        meta.dtype = str(arrays[0].dtype)
        meta.dimensionality = describe_dimensionality(axes, shape)

        scale = tuple(_axis_scale(axis, meta) for axis in layer_axes)
        units = tuple(_axis_unit(axis) for axis in layer_axes)

        specs: list[LayerSpec] = []
        for index, channel_arrays in enumerate(per_channel):
            channel = meta.channel(index)
            base = meta.image_name
            name = f"{base} :: {channel.display_name}" if n_channels > 1 else base
            colormap, color, blending = channel_appearance(
                channel.display_name, channel.color, index, n_channels
            )
            specs.append(
                LayerSpec(
                    data=channel_arrays if multiscale else channel_arrays[0],
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
                    contrast_limits=channel.contrast_limits,
                )
            )
        logger.info("read %s: axes=%s shape=%s channels=%d", path.name, axes, shape, n_channels)
        return specs
    except Exception:
        tf.close()
        _OPEN_FILES.remove(tf)
        raise


def _axis_scale(axis: str, meta: AcquisitionMetadata) -> float:
    if axis == "X":
        return meta.pixel_size_x_um or 1.0
    if axis == "Y":
        return meta.pixel_size_y_um or 1.0
    if axis == "Z":
        return meta.z_step_um or 1.0
    if axis == "T":
        return meta.time_interval_s or 1.0
    return 1.0


def _axis_unit(axis: str) -> str:
    if axis in "XYZ":
        return MICRON
    if axis == "T":
        return "s"
    return ""
