"""Generate small synthetic datasets for exercising the readers.

Usage::

    python tests/make_sample_data.py [output_dir]

Writes one file per supported format, each with plausible acquisition metadata,
so the viewer can be checked without access to real microscope data.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def _blobs(shape: tuple[int, ...], seed: int = 0) -> np.ndarray:
    """A stack of soft Gaussian blobs, bright enough to see at default contrast."""
    rng = np.random.default_rng(seed)
    *lead, height, width = shape
    yy, xx = np.mgrid[0:height, 0:width]
    frames = []
    for index in range(int(np.prod(lead)) if lead else 1):
        frame = np.zeros((height, width), dtype=np.float64)
        for _ in range(8):
            cy, cx = rng.uniform(0, height), rng.uniform(0, width)
            sigma = rng.uniform(min(height, width) / 25, min(height, width) / 8)
            frame += np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma**2))
        frame += 0.02 * rng.standard_normal(frame.shape)
        frames.append(frame * (0.6 + 0.4 * np.sin(index / 3.0)))
    stack = np.stack(frames) if lead else frames[0][np.newaxis]
    stack = np.clip(stack, 0, None)
    stack = (stack / (stack.max() or 1) * 4000).astype(np.uint16)
    return stack.reshape(shape)


# ---------------------------------------------------------------------------
# Imaris
# ---------------------------------------------------------------------------


def _write_attr(node, key: str, value) -> None:
    """Imaris stores every attribute as an array of single-byte characters."""
    text = str(value)
    node.attrs.create(key, np.frombuffer(text.encode("ascii", "replace"), dtype="S1"))


#: (name, "r g b", excitation nm, emission nm, laser power, exposure ms)
DEFAULT_CHANNELS = [
    ("GFP", "0 1 0", 488, 509, 3.2, 120.0),
    ("mCherry", "1 0 0.2", 561, 610, 5.5, 200.0),
    ("DAPI", "0 0.4 1", 405, 461, 2.0, 80.0),
]


def write_ims(path: Path, shape=(3, 8, 128, 160), n_channels: int = 2, channels=None) -> Path:
    """A two-channel, multi-timepoint Imaris file with two resolution levels."""
    import h5py

    n_t, n_z, n_y, n_x = shape
    with h5py.File(path, "w") as handle:
        info = handle.create_group("DataSetInfo")

        image = info.create_group("Image")
        for key, value in (
            ("X", n_x), ("Y", n_y), ("Z", n_z),
            ("ExtMin0", 0.0), ("ExtMin1", 0.0), ("ExtMin2", 0.0),
            # 0.13 µm pixels, 0.5 µm z-step.
            ("ExtMax0", n_x * 0.13), ("ExtMax1", n_y * 0.13), ("ExtMax2", n_z * 0.5),
            ("Unit", "um"),
            ("Name", path.stem),
            ("RecordingDate", "2026-07-20 09:41:12.000"),
            ("NumberOfChannels", n_channels),
            ("LensPower", "63"),
            ("NumericalAperture", "1.40"),
            ("Objective", "Plan-Apochromat 63x/1.40 Oil"),
        ):
            _write_attr(image, key, value)

        channel_settings = list(channels or DEFAULT_CHANNELS)
        for index in range(n_channels):
            name, colour, excitation, emission, power, exposure = channel_settings[
                index % len(channel_settings)
            ]
            group = info.create_group(f"Channel {index}")
            for key, value in (
                ("Name", name),
                ("Color", colour),
                ("ColorRange", "0 4000"),
                ("LSMExcitationWavelength", excitation),
                ("LSMEmissionWavelength", emission),
                ("LaserPower", power),
                ("ExposureTime [ms]", exposure),
            ):
                _write_attr(group, key, value)

        time_info = info.create_group("TimeInfo")
        _write_attr(time_info, "DatasetTimePoints", n_t)
        for index in range(n_t):
            seconds = 12 + index * 30  # 30 s interval
            _write_attr(time_info, f"TimePoint{index + 1}", f"2026-07-20 09:41:{seconds:02d}.000")

        data_group = handle.create_group("DataSet")
        for level in range(2):
            factor = 2**level
            level_shape = (n_z, max(n_y // factor, 1), max(n_x // factor, 1))
            level_group = data_group.create_group(f"ResolutionLevel {level}")
            for t in range(n_t):
                tp_group = level_group.create_group(f"TimePoint {t}")
                for c in range(n_channels):
                    volume = _blobs(level_shape, seed=100 * level + 10 * t + c)
                    ch_group = tp_group.create_group(f"Channel {c}")
                    dataset = ch_group.create_dataset("Data", data=volume, chunks=True)
                    _write_attr(dataset, "ImageSizeX", level_shape[2])
                    _write_attr(dataset, "ImageSizeY", level_shape[1])
                    _write_attr(dataset, "ImageSizeZ", level_shape[0])
    return path


def write_ims_2d(path: Path, shape=(1, 1, 96, 128)) -> Path:
    """A single-channel, single-plane Imaris file — the plain XY case."""
    return write_ims(path, shape=shape, n_channels=1)


def write_ims_brightfield(path: Path, shape=(1, 4, 96, 128)) -> Path:
    """Brightfield plus fluorescence, with a *green* colour stored for brightfield.

    Acquisition software routinely assigns a transmitted-light channel whatever
    colour is next in its cycle. The reader is expected to ignore that and show
    the channel in grayscale, so this sample deliberately sets a misleading one.
    """
    return write_ims(
        path,
        shape=shape,
        n_channels=2,
        channels=[
            ("Brightfield", "0 1 0", 0, 0, 0.0, 15.0),
            ("GFP", "0 1 0", 488, 509, 3.2, 120.0),
        ],
    )


# ---------------------------------------------------------------------------
# TIFF
# ---------------------------------------------------------------------------


def write_ome_tiff(path: Path, shape=(4, 6, 2, 96, 128)) -> Path:
    """An OME-TIFF (TZCYX) with pixel sizes, exposures and channel names."""
    import tifffile

    n_t, n_z, n_c, n_y, n_x = shape
    data = _blobs((n_t * n_z * n_c, n_y, n_x), seed=7).reshape(shape)
    metadata = {
        "axes": "TZCYX",
        "PhysicalSizeX": 0.108,
        "PhysicalSizeXUnit": "µm",
        "PhysicalSizeY": 0.108,
        "PhysicalSizeYUnit": "µm",
        "PhysicalSizeZ": 0.35,
        "PhysicalSizeZUnit": "µm",
        "TimeIncrement": 15.0,
        "TimeIncrementUnit": "s",
        "AcquisitionDate": "2026-07-21T14:05:00",
        "Channel": {"Name": ["EGFP", "Alexa 594"][:n_c]},
        # tifffile requires one entry per plane for every Plane attribute.
        "Plane": {"ExposureTime": [0.05] * (n_t * n_z * n_c)},
    }
    tifffile.imwrite(str(path), data, metadata=metadata, ome=True)
    return path


def write_imagej_tiff(path: Path, shape=(10, 128, 128)) -> Path:
    """An ImageJ-style Z stack carrying spacing and resolution tags."""
    import tifffile

    data = _blobs(shape, seed=21)
    tifffile.imwrite(
        str(path),
        data,
        imagej=True,
        resolution=(1 / 0.2, 1 / 0.2),  # pixels per µm
        metadata={"axes": "ZYX", "spacing": 0.8, "unit": "um", "finterval": 0.0},
    )
    return path


def write_plain_tiff(path: Path, shape=(200, 240)) -> Path:
    """A bare 2D TIFF with no calibration at all — the uncalibrated fallback."""
    import tifffile

    tifffile.imwrite(str(path), _blobs(shape, seed=33))
    return path


# ---------------------------------------------------------------------------
# OME-Zarr
# ---------------------------------------------------------------------------


def write_ome_zarr(path: Path, shape=(2, 5, 96, 128)) -> Path:
    """A minimal NGFF store: two channels, two pyramid levels, CZYX axes."""
    import zarr

    n_c, n_z, n_y, n_x = shape
    root = zarr.open_group(str(path), mode="w")
    datasets = []
    for level in range(2):
        factor = 2**level
        level_shape = (n_c, n_z, max(n_y // factor, 1), max(n_x // factor, 1))
        array = _blobs((n_c * n_z, level_shape[2], level_shape[3]), seed=level).reshape(level_shape)
        root.create_dataset(str(level), data=array, chunks=(1, 1, 64, 64), overwrite=True)
        datasets.append(
            {
                "path": str(level),
                "coordinateTransformations": [
                    {"type": "scale", "scale": [1.0, 0.4, 0.15 * factor, 0.15 * factor]}
                ],
            }
        )

    root.attrs["multiscales"] = [
        {
            "version": "0.4",
            "name": path.stem,
            "axes": [
                {"name": "c", "type": "channel"},
                {"name": "z", "type": "space", "unit": "micrometer"},
                {"name": "y", "type": "space", "unit": "micrometer"},
                {"name": "x", "type": "space", "unit": "micrometer"},
            ],
            "datasets": datasets,
        }
    ]
    root.attrs["omero"] = {
        "name": path.stem,
        "channels": [
            {"label": "GFP", "color": "00FF00", "window": {"start": 0, "end": 4000}, "emissionWave": 509},
            {"label": "RFP", "color": "FF0000", "window": {"start": 0, "end": 4000}, "emissionWave": 610},
        ][:n_c],
    }
    return path


# ---------------------------------------------------------------------------


def write_all(directory: Path) -> list[Path]:
    """Write one sample per format into *directory* and return the paths."""
    directory.mkdir(parents=True, exist_ok=True)
    written = [
        write_ims(directory / "sample_4d_2ch.ims"),
        write_ims_2d(directory / "sample_2d.ims"),
        write_ims_brightfield(directory / "sample_brightfield.ims"),
        write_ome_tiff(directory / "sample_tzcyx.ome.tif"),
        write_imagej_tiff(directory / "sample_zstack_imagej.tif"),
        write_plain_tiff(directory / "sample_plain.tif"),
    ]
    try:
        written.append(write_ome_zarr(directory / "sample.zarr"))
    except Exception as exc:  # zarr is optional
        print(f"skipped OME-Zarr sample: {exc}")
    return written


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "sample_data"
    for created in write_all(target):
        print(created)
