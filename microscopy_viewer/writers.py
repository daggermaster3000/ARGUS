"""Write image stacks back out as TIFF or Imaris files.

The readers in :mod:`microscopy_viewer.loaders` turn files into arrays; this goes
the other way, for the things this program produces that are images rather than
tables or slides — a batch of maximum projections, to begin with.

Two formats, for two different afterlives. **TIFF** is what everything else
reads: ImageJ-flavoured, so Fiji opens it as a calibrated hyperstack with the
channels already separated. **Imaris** is what this lab's other software reads,
and keeping a derived image in the same format as its source is what lets it sit
in the same folder and be opened the same way.

Writing ``.ims`` is the fiddly half, so what it does is worth stating:

* Imaris stores every attribute as an **array of single characters**, not as an
  HDF5 string. A file whose attributes are real strings looks empty to it.
  :func:`_char` is the whole of that difference.
* The array is stored padded up to the chunk size and the true extent lives in
  ``ImageSizeX/Y/Z``. Nothing is padded here — the sizes are simply declared,
  which the readers (ours and Imaris's) take from those attributes anyway.
* **Stage extents are carried through.** ``ExtMin``/``ExtMax`` are absolute stage
  coordinates, and they are what :mod:`microscopy_viewer.overview` uses to place
  each sample on the mosaic. A projection that dropped them would still open, and
  would silently fall off the overview slide.

What this does not write is a resolution pyramid: one level, which is right for a
projection a few megapixels across and would not be for a whole volume.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .utils import MICRON, get_logger

logger = get_logger("writers")

#: Output formats this module can write, as file-dialog-ready suffixes.
TIFF_SUFFIXES = (".tif", ".tiff")
IMS_SUFFIXES = (".ims",)
SUPPORTED_SUFFIXES = TIFF_SUFFIXES + IMS_SUFFIXES

#: Default colours for channels whose source recorded none, in acquisition order.
FALLBACK_COLORS = (
    (0.0, 0.0, 1.0),
    (0.0, 1.0, 0.0),
    (1.0, 0.0, 0.0),
    (1.0, 1.0, 0.0),
)


def write_stack(
    path: str | Path,
    planes: np.ndarray,
    voxel_um: Sequence[float] = (1.0, 1.0, 1.0),
    channel_names: Sequence[str] = (),
    channel_colors: Sequence[Sequence[float] | None] = (),
    stage_extent: Sequence[float] | None = None,
    source_name: str = "",
    time_interval_s: float = 0.0,
) -> Path:
    """Write *planes* to *path*, picking the writer from the suffix.

    *planes* is ``(T, C, Y, X)``. A single timepoint and a single channel are
    still given their axes rather than squeezed, so the callers never have to
    branch on how many of either there are.
    """
    target = Path(path)
    suffix = target.suffix.lower()
    if suffix in IMS_SUFFIXES:
        return write_ims(
            target, planes, voxel_um, channel_names, channel_colors, stage_extent,
            source_name, time_interval_s,
        )
    if suffix in TIFF_SUFFIXES:
        return write_tiff(target, planes, voxel_um, channel_names)
    raise ValueError(
        f"cannot write {target.name}: expected one of {', '.join(SUPPORTED_SUFFIXES)}"
    )


def _as_tczyx(planes: np.ndarray) -> np.ndarray:
    """Force an array to ``(T, C, Y, X)``, adding the axes it is missing."""
    array = np.asarray(planes)
    if array.ndim == 2:
        return array[None, None, ...]
    if array.ndim == 3:
        return array[None, ...]
    if array.ndim == 4:
        return array
    raise ValueError(f"expected a 2D, 3D or 4D array, got {array.ndim}D")


# ---------------------------------------------------------------------------
# TIFF
# ---------------------------------------------------------------------------


def write_tiff(
    path: str | Path,
    planes: np.ndarray,
    voxel_um: Sequence[float] = (1.0, 1.0, 1.0),
    channel_names: Sequence[str] = (),
) -> Path:
    """Write an ImageJ-flavoured TIFF that opens as a calibrated hyperstack.

    ``imagej=True`` and an explicit ``axes`` is what makes Fiji separate the
    channels rather than reading a three-channel image as RGB, and what puts the
    pixel size in the Image ▸ Properties box instead of leaving it at 1 pixel.
    """
    import tifffile

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    array = _as_tczyx(planes)

    voxel = tuple(float(v) for v in tuple(voxel_um)[-2:]) or (1.0, 1.0)
    y_um, x_um = (voxel + (1.0, 1.0))[:2]
    # TIFF resolution is pixels per unit, so it is the reciprocal of a pixel size.
    resolution = (1.0 / x_um if x_um else 1.0, 1.0 / y_um if y_um else 1.0)

    metadata: dict[str, Any] = {"axes": "TCYX", "unit": "um"}
    if channel_names:
        # Fiji shows these as the slice labels, which is how a channel keeps its
        # name in a format that has no concept of one.
        metadata["Labels"] = [str(name) for name in channel_names] * int(array.shape[0])

    tifffile.imwrite(
        str(target),
        array,
        imagej=True,
        resolution=resolution,
        photometric="minisblack",  # never let a 3-channel image be taken for RGB
        metadata=metadata,
    )
    logger.info("wrote %s %s to %s", array.shape, array.dtype, target.name)
    return target


# ---------------------------------------------------------------------------
# Imaris
# ---------------------------------------------------------------------------


def _char(value: Any) -> np.ndarray:
    """One attribute, in the form Imaris stores them: an array of single chars.

    This is not a stylistic choice. Imaris writes every attribute as ``|S1`` and
    reads them back the same way; an attribute written as an ordinary HDF5 string
    is simply not seen, so a file full of them opens with no name, no channels
    and no calibration.
    """
    text = "" if value is None else str(value)
    # From the buffer, not from ``list(text.encode())``: that yields a list of
    # integers, which numpy renders as their decimal digits truncated to one
    # character each, so "2040" is stored as the letters of "50 48 52 48".
    return np.frombuffer(text.encode("utf-8"), dtype="S1").copy()


def _set(node, **attributes: Any) -> None:
    for key, value in attributes.items():
        node.attrs[key] = _char(value)


def _extent_for(
    stage_extent: Sequence[float] | None,
    shape_zyx: tuple[int, int, int],
    voxel_um: Sequence[float],
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    """``((x0, x1), (y0, y1), (z0, z1))`` in µm for the DataSetInfo/Image attrs.

    Only the *origin* comes from the source. The span is always the voxel size
    times the number of voxels actually being written, because that is what the
    readers divide to recover the pixel size — Imaris stores no pixel size of its
    own. Copying the source's far corner as well would put the right stage
    position on an image of the wrong scale the moment anything is cropped or
    decimated, and a wrong calibration is worse than a missing one.

    The Z range collapses to the single plane a projection is, starting where the
    original stack started.
    """
    depth, height, width = (int(n) for n in shape_zyx)
    z_um, y_um, x_um = (tuple(float(v) for v in voxel_um) + (1.0, 1.0, 1.0))[:3]

    x0 = y0 = z0 = 0.0
    if stage_extent is not None and len(tuple(stage_extent)) >= 6:
        values = tuple(float(v) for v in tuple(stage_extent)[:6])
        x0, y0, z0 = values[0], values[2], values[4]
    return (
        (x0, x0 + x_um * width),
        (y0, y0 + y_um * height),
        (z0, z0 + z_um * depth),
    )


def write_ims(
    path: str | Path,
    planes: np.ndarray,
    voxel_um: Sequence[float] = (1.0, 1.0, 1.0),
    channel_names: Sequence[str] = (),
    channel_colors: Sequence[Sequence[float] | None] = (),
    stage_extent: Sequence[float] | None = None,
    source_name: str = "",
    time_interval_s: float = 0.0,
) -> Path:
    """Write an Imaris file with one resolution level.

    *planes* is ``(T, C, Y, X)``; the Z axis of the result is one plane, which is
    what a projection is. The structure written is the one Imaris 5.5 files use,
    which is what :mod:`microscopy_viewer.loaders.ims` reads.
    """
    import h5py

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    array = _as_tczyx(planes)
    timepoints, channels, height, width = (int(n) for n in array.shape)
    depth = 1

    z_um, y_um, x_um = (tuple(float(v) for v in voxel_um) + (1.0, 1.0, 1.0))[:3]
    (x0, x1), (y0, y1), (z0, z1) = _extent_for(stage_extent, (depth, height, width), (z_um, y_um, x_um))
    stamp = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

    with h5py.File(str(target), "w") as handle:
        _set(
            handle,
            DataSetDirectoryName="DataSet",
            DataSetInfoDirectoryName="DataSetInfo",
            ImarisDataSet="ImarisDataSet",
            ImarisVersion="5.5.0",
            ThumbnailDirectoryName="Thumbnail",
        )
        handle.attrs["NumberOfDataSets"] = np.array([1], dtype=np.uint32)

        for time_index in range(timepoints):
            for channel_index in range(channels):
                plane = np.asarray(array[time_index, channel_index])
                group = handle.require_group(
                    f"DataSet/ResolutionLevel 0/TimePoint {time_index}/Channel {channel_index}"
                )
                data = group.create_dataset(
                    "Data",
                    data=plane[None, ...],  # (Z, Y, X) with a single Z
                    chunks=(1, min(height, 256), min(width, 256)),
                    compression="gzip",
                    compression_opts=2,
                )
                finite = plane[np.isfinite(plane)] if plane.size else plane
                low = float(finite.min()) if finite.size else 0.0
                high = float(finite.max()) if finite.size else 0.0
                _set(
                    data,
                    ImageSizeX=width,
                    ImageSizeY=height,
                    ImageSizeZ=depth,
                )
                _set(
                    group,
                    ImageSizeX=width,
                    ImageSizeY=height,
                    ImageSizeZ=depth,
                    HistogramMin=f"{low:.3f}",
                    HistogramMax=f"{high:.3f}",
                    HistogramMin1024=f"{low:.3f}",
                    HistogramMax1024=f"{high:.3f}",
                )

        image = handle.require_group("DataSetInfo/Image")
        _set(
            image,
            X=width,
            Y=height,
            Z=depth,
            Unit=MICRON,
            Description="Maximum projection written by ARGUS",
            Name=source_name or target.name,
            RecordingDate=stamp,
            ExtMin0=f"{x0:.6g}",
            ExtMin1=f"{y0:.6g}",
            ExtMin2=f"{z0:.6g}",
            ExtMax0=f"{x1:.6g}",
            ExtMax1=f"{y1:.6g}",
            ExtMax2=f"{z1:.6g}",
            NumberOfChannels=channels,
        )

        for channel_index in range(channels):
            node = handle.require_group(f"DataSetInfo/Channel {channel_index}")
            name = (
                str(channel_names[channel_index])
                if channel_index < len(channel_names)
                else f"Channel {channel_index + 1}"
            )
            color = None
            if channel_index < len(channel_colors):
                color = channel_colors[channel_index]
            if color is None or len(tuple(color)) < 3:
                color = FALLBACK_COLORS[channel_index % len(FALLBACK_COLORS)]
            red, green, blue = (float(c) for c in tuple(color)[:3])
            values = np.asarray(array[:, channel_index])
            _set(
                node,
                Name=name,
                Description="",
                Color=f"{red:.3f} {green:.3f} {blue:.3f}",
                ColorMode="BaseColor",
                ColorOpacity="1.000",
                ColorRange=f"{float(values.min()):.3f} {float(values.max()):.3f}"
                if values.size
                else "0.000 1.000",
                GammaCorrection="1.000",
            )

        # Timepoint stamps carry the interval: Imaris records no frame rate of
        # its own, so a reader recovers it by differencing these. Writing the
        # same stamp for every frame is how a time-lapse projection comes back
        # with no time calibration at all.
        started = _dt.datetime.now()
        stamps = [
            (started + _dt.timedelta(seconds=float(time_interval_s) * index)).strftime(
                "%Y-%m-%d %H:%M:%S.%f"
            )[:-3]
            for index in range(timepoints)
        ]

        time_info = handle.require_group("DataSetInfo/TimeInfo")
        _set(
            time_info,
            DatasetTimePoints=timepoints,
            FileTimePoints=timepoints,
            AcquisitionStart=stamps[0],
        )
        for time_index, moment in enumerate(stamps, start=1):
            _set(time_info, **{f"TimePoint{time_index}": moment})

        _write_times(handle, stamps)

    logger.info("wrote %s %s to %s", array.shape, array.dtype, target.name)
    return target


def _write_times(handle, stamps: Sequence[str]) -> None:
    """The ``/DataSetTimes`` tables, in the compound layout Imaris writes.

    Not strictly needed to read the pixels back, but a file missing them is a
    file that differs from every other ``.ims`` in the folder, and the point of
    writing Imaris at all is that the output behaves like the input.
    """
    times = handle.require_group("DataSetTimes")
    timepoints = len(stamps)
    time_dtype = np.dtype(
        [("ID", "<i8"), ("Birth", "<i8"), ("Death", "<i8"), ("IDTimeBegin", "<i8")]
    )
    begin_dtype = np.dtype([("ID", "<i8"), ("ObjectTimeBegin", "S256")])
    times.create_dataset(
        "Time",
        data=np.array(
            [(index, 0, 0, index) for index in range(timepoints)], dtype=time_dtype
        ),
    )
    times.create_dataset(
        "TimeBegin",
        data=np.array(
            [(index, stamps[index].encode("utf-8")) for index in range(timepoints)],
            dtype=begin_dtype,
        ),
    )
