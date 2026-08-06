"""Acquisition metadata model plus the generic key-matching used by every reader.

Vendors spell the same concept many different ways, so readers dump whatever raw
key/value pairs they can find into :meth:`AcquisitionMetadata.harvest`, which maps
them onto the canonical fields via the synonym tables below.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .utils import (
    MICRON,
    format_number,
    length_to_micron,
    parse_float,
    time_to_second,
)

# Canonical field -> lower-cased substrings that identify it in vendor metadata.
# Order matters: the first matching field wins, so put specific keys first.
_SYNONYMS: list[tuple[str, tuple[str, ...]]] = [
    ("exposure_ms", ("exposuretime", "exposure", "integrationtime", "dwelltime")),
    ("excitation_nm", ("excitationwavelength", "lsmexcitationwavelength", "exwavelength", "excitation")),
    ("emission_nm", ("emissionwavelength", "lsmemissionwavelength", "emwavelength", "emission")),
    ("numerical_aperture", ("numericalaperture", "lensna", "lsmnumericalaperture", "objectivena")),
    # Objective must be tested before laser power: "LensPower" is a magnification.
    ("objective", ("objectivename", "objective", "lensname", "lenspower", "microscopeobjective")),
    ("laser_power", ("laserpower", "lsmpower", "laserintensity", "laseroutput", "excitationpower")),
    ("acquisition_date", ("recordingdate", "acquisitiondate", "acquisitiontime", "datetime", "creationdate")),
    ("z_step_um", ("zstep", "zspacing", "physicalsizez", "slicespacing")),
    ("time_interval_s", ("timeinterval", "timeincrement", "frameinterval", "finterval", "cycletime")),
    ("laser_name", ("lasername", "lightsource", "laserline")),
]

_UNIT_HINTS = {
    "exposure_ms": "ms",
    "excitation_nm": "nm",
    "emission_nm": "nm",
    "z_step_um": MICRON,
    "time_interval_s": "s",
}


def _match_field(key: str) -> str | None:
    """Return the canonical field name a raw metadata *key* maps onto, if any."""
    flat = key.lower().replace(" ", "").replace("_", "").replace("-", "")
    for name, needles in _SYNONYMS:
        for needle in needles:
            if needle.replace(" ", "") in flat:
                return name
    return None


@dataclass
class ChannelMetadata:
    """Per-channel acquisition settings."""

    index: int
    name: str = ""
    excitation_nm: float | None = None
    emission_nm: float | None = None
    laser_power: float | None = None
    laser_power_unit: str = "%"
    exposure_ms: float | None = None
    laser_name: str = ""
    color: tuple[float, float, float] | None = None
    contrast_limits: tuple[float, float] | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def display_name(self) -> str:
        return self.name or f"Channel {self.index}"

    def rows(self) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        if self.laser_name:
            out.append(("Laser", self.laser_name))
        if self.excitation_nm is not None:
            out.append(("Excitation", f"{format_number(self.excitation_nm)} nm"))
        if self.emission_nm is not None:
            out.append(("Emission", f"{format_number(self.emission_nm)} nm"))
        if self.laser_power is not None:
            out.append(("Laser power", f"{format_number(self.laser_power)} {self.laser_power_unit}"))
        if self.exposure_ms is not None:
            out.append(("Exposure time", f"{format_number(self.exposure_ms)} ms"))
        if self.contrast_limits is not None:
            low, high = self.contrast_limits
            out.append(("Intensity range", f"{format_number(low)} – {format_number(high)}"))
        for key, value in sorted(self.extra.items()):
            out.append((key, str(value)))
        return out


@dataclass
class AcquisitionMetadata:
    """Everything the metadata dock knows how to display about one dataset."""

    image_name: str = ""
    file_path: Path | None = None
    file_format: str = ""
    acquisition_date: str = ""
    objective: str = ""
    numerical_aperture: float | None = None
    pixel_size_x_um: float | None = None
    pixel_size_y_um: float | None = None
    z_step_um: float | None = None
    time_interval_s: float | None = None
    #: Where on the stage the image was taken, as ``(x0, x1, y0, y1, z0, z1)`` in
    #: µm, when the file records it. Absolute rather than a size: an overview
    #: mosaic and the fields acquired from it share one stage frame, and that is
    #: what makes it possible to say where on the overview a sample came from.
    stage_extent: tuple[float, float, float, float, float, float] | None = None
    axes: str = ""
    shape: tuple[int, ...] = ()
    dimensionality: str = ""
    dtype: str = ""
    channels: list[ChannelMetadata] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    # -- construction helpers -------------------------------------------------

    def channel(self, index: int) -> ChannelMetadata:
        """Return (creating if needed) the metadata block for channel *index*."""
        for existing in self.channels:
            if existing.index == index:
                return existing
        created = ChannelMetadata(index=index)
        self.channels.append(created)
        self.channels.sort(key=lambda c: c.index)
        return created

    @property
    def channel_names(self) -> list[str]:
        return [c.display_name for c in self.channels]

    @property
    def stage_position_um(self) -> tuple[float, float] | None:
        """Centre of the field on the stage as ``(x, y)`` in µm, when known."""
        if self.stage_extent is None or len(self.stage_extent) < 4:
            return None
        x0, x1, y0, y1 = (float(v) for v in self.stage_extent[:4])
        return (0.5 * (x0 + x1), 0.5 * (y0 + y1))

    @property
    def is_calibrated(self) -> bool:
        """True when a real voxel size was found, so measurements can use µm."""
        return self.pixel_size_x_um is not None and self.pixel_size_y_um is not None

    def harvest(self, raw: Mapping[str, Any], channel_index: int | None = None) -> None:
        """Fold arbitrary vendor key/value pairs into the canonical fields.

        Values that map onto a known field are converted to the field's unit.
        Anything unrecognised is preserved in ``extra`` so nothing is silently lost.
        """
        target_extra = self.channel(channel_index).extra if channel_index is not None else self.extra
        for key, value in raw.items():
            if value is None or value == "":
                continue
            name = _match_field(str(key))
            if name is None:
                target_extra.setdefault(str(key), value)
                continue
            self._assign(name, key, value, channel_index)

    def _assign(self, name: str, raw_key: str, value: Any, channel_index: int | None) -> None:
        """Write one recognised value onto the right field, converting units."""
        per_channel = {"exposure_ms", "excitation_nm", "emission_nm", "laser_power", "laser_name"}
        unit = _unit_from_key(raw_key) or _UNIT_HINTS.get(name)

        if name in per_channel:
            targets: Iterable[ChannelMetadata]
            if channel_index is not None:
                targets = [self.channel(channel_index)]
            elif self.channels:
                targets = self.channels  # file-level value applies to every channel
            else:
                targets = [self.channel(0)]
            for target in targets:
                self._assign_channel(target, name, value, unit)
            return

        if name == "acquisition_date":
            if not self.acquisition_date:
                self.acquisition_date = str(value).strip()
        elif name == "objective":
            text = str(value).strip()
            if text and not self.objective:
                # "LensPower" style keys hold a bare magnification number.
                self.objective = f"{text}x" if text.replace(".", "", 1).isdigit() else text
        elif name == "numerical_aperture":
            if self.numerical_aperture is None:
                self.numerical_aperture = parse_float(value)
        elif name == "z_step_um":
            if self.z_step_um is None:
                self.z_step_um = length_to_micron(parse_float(value), unit or MICRON)
        elif name == "time_interval_s":
            if self.time_interval_s is None:
                self.time_interval_s = time_to_second(parse_float(value), unit or "s")

    @staticmethod
    def _assign_channel(target: ChannelMetadata, name: str, value: Any, unit: str | None) -> None:
        if name == "laser_name":
            if not target.laser_name:
                target.laser_name = str(value).strip()
            return
        if name == "exposure_ms":
            if target.exposure_ms is None:
                seconds = time_to_second(parse_float(value), unit or "ms")
                target.exposure_ms = None if seconds is None else seconds * 1e3
            return
        if name in ("excitation_nm", "emission_nm"):
            if getattr(target, name) is None:
                nanometres = length_to_micron(parse_float(value), unit or "nm")
                if nanometres is not None:
                    setattr(target, name, nanometres * 1e3)  # µm -> nm
            return
        if name == "laser_power" and target.laser_power is None:
            target.laser_power = parse_float(value)
            if unit:
                target.laser_power_unit = unit

    # -- display --------------------------------------------------------------

    def summary_rows(self) -> list[tuple[str, str]]:
        """Flat ``(label, value)`` rows for the non-channel part of the dock."""
        rows: list[tuple[str, str]] = [("Image", self.image_name)]
        if self.file_path is not None:
            rows.append(("File", str(self.file_path)))
        if self.file_format:
            rows.append(("Format", self.file_format))
        if self.acquisition_date:
            rows.append(("Acquisition date", self.acquisition_date))
        if self.dimensionality:
            rows.append(("Dimensionality", self.dimensionality))
        if self.shape:
            rows.append(("Shape", f"{tuple(self.shape)}  ({self.axes})"))
        if self.dtype:
            rows.append(("Data type", self.dtype))
        if self.objective:
            rows.append(("Objective", self.objective))
        if self.numerical_aperture is not None:
            rows.append(("Numerical aperture", format_number(self.numerical_aperture)))
        if self.pixel_size_x_um is not None:
            if self.pixel_size_y_um is not None and abs(self.pixel_size_x_um - self.pixel_size_y_um) > 1e-9:
                rows.append(
                    (
                        "Pixel size",
                        f"{format_number(self.pixel_size_x_um)} × "
                        f"{format_number(self.pixel_size_y_um)} {MICRON}",
                    )
                )
            else:
                rows.append(("Pixel size", f"{format_number(self.pixel_size_x_um)} {MICRON}"))
        if self.z_step_um is not None:
            rows.append(("Z-step size", f"{format_number(self.z_step_um)} {MICRON}"))
        if self.time_interval_s is not None:
            rows.append(("Time interval", f"{format_number(self.time_interval_s)} s"))
        position = self.stage_position_um
        if position is not None:
            rows.append(
                (
                    "Stage position",
                    f"X {format_number(position[0])}, Y {format_number(position[1])} {MICRON}",
                )
            )
        if self.channels:
            rows.append(("Channel names", ", ".join(self.channel_names)))
        if not self.is_calibrated:
            rows.append(("Calibration", "not in file — measurements shown in pixels"))
        return rows

    def extra_rows(self) -> list[tuple[str, str]]:
        return [(key, str(value)) for key, value in sorted(self.extra.items())]


def _unit_from_key(key: str) -> str | None:
    """Extract a trailing unit from keys like ``ExposureTime [ms]`` or ``Z step (um)``."""
    text = str(key)
    for opener, closer in (("[", "]"), ("(", ")")):
        if opener in text and text.rstrip().endswith(closer):
            candidate = text[text.rindex(opener) + 1 : text.rindex(closer)].strip()
            if candidate and len(candidate) <= 8:
                return candidate
    return None


def apply_channel_names(meta: AcquisitionMetadata, names: Sequence[str]) -> None:
    """Set channel display names from an ordered sequence, keeping blanks untouched."""
    for index, name in enumerate(names):
        if name:
            meta.channel(index).name = str(name).strip()
