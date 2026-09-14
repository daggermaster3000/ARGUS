"""Turn napari Shapes into calibrated distance and area measurements.

Shapes layers created by :func:`new_roi_layer` inherit the scale of the image
they annotate, so vertex coordinates multiplied by that scale are already in
micrometres whenever the file provided a voxel size.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

import numpy as np

from .metadata import AcquisitionMetadata
from .utils import (
    MICRON,
    MICRON_SQ,
    ellipse_axes,
    get_logger,
    polyline_length,
    ramanujan_perimeter,
    shoelace_area,
)

logger = get_logger("measurements")

#: Key under which a Shapes layer records which image layer it annotates.
TARGET_KEY = "mv_target_layer"
#: Key marking a layer as one of ours, so the dock can list it.
ROI_KEY = "mv_roi_layer"
#: Feature column holding the user-editable ROI name.
NAME_FEATURE = "name"

#: Shape types that enclose an area; everything else is length-only.
CLOSED_SHAPES = ("rectangle", "ellipse", "polygon")


@dataclass
class Measurement:
    """One reported quantity for one ROI."""

    roi_name: str
    measurement_type: str
    value: float
    unit: str
    image_name: str
    channel: str
    timestamp: str
    shape_type: str = ""
    calibrated: bool = True
    slice_position: str = ""
    n_vertices: int = 0
    roi_layer: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def as_row(self) -> dict[str, Any]:
        """Flat dict for the results table and the Excel export."""
        row = asdict(self)
        row.pop("extra")
        row.update({str(key): value for key, value in self.extra.items()})
        return row


#: Column order for the results table and the spreadsheet.
COLUMNS = (
    "roi_name",
    "measurement_type",
    "value",
    "unit",
    "image_name",
    "channel",
    "timestamp",
    "shape_type",
    "calibrated",
    "slice_position",
    "n_vertices",
    "roi_layer",
)

COLUMN_LABELS = {
    "roi_name": "ROI name",
    "measurement_type": "Type",
    "value": "Value",
    "unit": "Unit",
    "image_name": "Image",
    "channel": "Channel",
    "timestamp": "Timestamp",
    "shape_type": "Shape",
    "calibrated": "Calibrated",
    "slice_position": "Slice",
    "n_vertices": "Vertices",
    "roi_layer": "ROI layer",
}


def _scaled(vertices: np.ndarray, scale: Sequence[float]) -> np.ndarray:
    """Multiply vertex coordinates by the layer scale to get physical units."""
    vertices = np.asarray(vertices, dtype=float)
    if vertices.ndim == 1:
        vertices = vertices[np.newaxis, :]
    factors = np.asarray(scale, dtype=float)[-vertices.shape[1] :]
    return vertices * factors


def _varying_columns(vertices: np.ndarray) -> np.ndarray:
    """Vertices restricted to the axes that actually vary.

    A line drawn across Z in a 4D stack varies in Z, Y and X; keeping all three
    makes its length a true 3D distance. Constant axes (the T slider position,
    say) would only add zeros.
    """
    if vertices.shape[1] <= 2:
        return vertices
    varies = np.ptp(vertices, axis=0) > 1e-12
    if int(varies.sum()) >= 2:
        return vertices[:, varies]
    return vertices[:, -2:]


def _planar_columns(vertices: np.ndarray) -> np.ndarray:
    """Exactly two columns, chosen as the axes with the largest extent.

    Area formulas are two-dimensional, so an ROI spanning three varying axes is
    projected onto the plane it mostly lies in rather than being computed wrongly.
    """
    if vertices.shape[1] <= 2:
        return vertices
    extents = np.ptp(vertices, axis=0)
    keep = np.sort(np.argsort(extents)[-2:])
    return vertices[:, keep]


def _slice_label(vertices: np.ndarray, axes: str) -> str:
    """Describe the non-varying leading axes, e.g. ``T=3, Z=12``."""
    if vertices.shape[1] <= 2 or not axes:
        return ""
    labels = []
    leading = axes[: vertices.shape[1]][:-2] if len(axes) >= vertices.shape[1] else ""
    for index, axis in enumerate(leading):
        column = vertices[:, index]
        if np.ptp(column) < 1e-9:
            labels.append(f"{axis}={column[0]:g}")
    return ", ".join(labels)


def _layer_context(shapes_layer, viewer=None) -> tuple[str, str, AcquisitionMetadata | None, str]:
    """Resolve the image name, channel, metadata and axes for a Shapes layer."""
    meta: AcquisitionMetadata | None = shapes_layer.metadata.get("mv_metadata")
    image_name = shapes_layer.metadata.get("mv_image_name", "")
    channel = shapes_layer.metadata.get("mv_channel_name", "")
    axes = shapes_layer.metadata.get("mv_axes", "")

    target = shapes_layer.metadata.get(TARGET_KEY)
    if viewer is not None and target and target in viewer.layers:
        image_layer = viewer.layers[target]
        meta = image_layer.metadata.get("mv_metadata", meta)
        image_name = image_layer.name
        channel = image_layer.metadata.get("mv_channel_name", channel)
        axes = image_layer.metadata.get("mv_axes", axes)
    return image_name or "—", channel or "", meta, axes


def roi_names(shapes_layer) -> list[str]:
    """Current ROI names, filling in ``ROI n`` defaults for unnamed shapes."""
    count = len(shapes_layer.data)
    features = shapes_layer.features
    if NAME_FEATURE in features and len(features[NAME_FEATURE]) == count:
        values = [str(value) if str(value) not in ("", "nan", "None") else "" for value in features[NAME_FEATURE]]
    else:
        values = [""] * count
    return [value or f"ROI {index + 1}" for index, value in enumerate(values)]


def ensure_name_feature(shapes_layer) -> None:
    """Make sure the layer has a ``name`` feature column matching its shape count.

    napari resizes ``features`` when shapes are added or removed, but new rows
    arrive empty, so this refills them with sequential defaults.

    Names are also forced to be unique. napari seeds a new shape's features from
    the layer's feature defaults, which copies the previous shape's name, so
    drawing several ROIs otherwise leaves them all called ``ROI 1``. Duplicates
    break every lookup that goes by name — renaming, background selection, the
    exported tables — so a repeat is treated as unset rather than deliberate.
    """
    count = len(shapes_layer.data)
    features = shapes_layer.features
    existing = list(features[NAME_FEATURE]) if NAME_FEATURE in features else []
    values: list[str] = []
    taken: set[str] = set()
    for index in range(count):
        current = str(existing[index]) if index < len(existing) else ""
        if current in ("", "nan", "None") or current in taken:
            current = f"ROI {index + 1}"
            suffix = index + 1
            while current in taken:
                suffix += 1
                current = f"ROI {suffix}"
        taken.add(current)
        values.append(current)
    if NAME_FEATURE in features and list(map(str, existing)) == values:
        return
    try:
        shapes_layer.features = {NAME_FEATURE: np.array(values, dtype=object)}
    except Exception:  # pragma: no cover - napari version differences
        logger.debug("could not set ROI name features", exc_info=True)


def rename_roi(shapes_layer, index: int, name: str) -> None:
    """Write a new name for one ROI back into the layer's feature table."""
    names = roi_names(shapes_layer)
    if not (0 <= index < len(names)):
        return
    names[index] = name.strip() or f"ROI {index + 1}"
    shapes_layer.features = {NAME_FEATURE: np.array(names, dtype=object)}
    if hasattr(shapes_layer, "refresh_text"):
        shapes_layer.refresh_text()


def measure_layer(shapes_layer, viewer=None) -> list[Measurement]:
    """Measure every shape in *shapes_layer*.

    Closed shapes report an area and a perimeter; lines and paths report a
    length. Values use the layer scale, so they are in µm when the source file
    supplied a voxel size.
    """
    image_name, channel, meta, axes = _layer_context(shapes_layer, viewer)
    calibrated = bool(meta.is_calibrated) if meta is not None else False
    length_unit = MICRON if calibrated else "px"
    area_unit = MICRON_SQ if calibrated else "px²"
    timestamp = _dt.datetime.now().isoformat(timespec="seconds")
    names = roi_names(shapes_layer)
    shape_types = list(shapes_layer.shape_type)

    out: list[Measurement] = []
    for index, raw_vertices in enumerate(shapes_layer.data):
        vertices = np.asarray(raw_vertices, dtype=float)
        if vertices.size == 0:
            continue
        shape_type = str(shape_types[index]) if index < len(shape_types) else "polygon"
        scaled = _scaled(vertices, shapes_layer.scale)
        planar = _planar_columns(scaled)
        path = _varying_columns(scaled)

        common = {
            "roi_name": names[index],
            "image_name": image_name,
            "channel": channel,
            "timestamp": timestamp,
            "shape_type": shape_type,
            "calibrated": calibrated,
            "slice_position": _slice_label(vertices, axes),
            "n_vertices": int(vertices.shape[0]),
            "roi_layer": shapes_layer.name,
        }

        if shape_type == "ellipse":
            semi_a, semi_b = ellipse_axes(planar)
            out.append(Measurement(measurement_type="Area", value=float(np.pi * semi_a * semi_b), unit=area_unit, **common))
            out.append(Measurement(measurement_type="Perimeter", value=ramanujan_perimeter(semi_a, semi_b), unit=length_unit, **common))
            out.append(Measurement(measurement_type="Major axis", value=2.0 * max(semi_a, semi_b), unit=length_unit, **common))
            out.append(Measurement(measurement_type="Minor axis", value=2.0 * min(semi_a, semi_b), unit=length_unit, **common))
        elif shape_type in CLOSED_SHAPES:
            out.append(Measurement(measurement_type="Area", value=shoelace_area(planar), unit=area_unit, **common))
            out.append(Measurement(measurement_type="Perimeter", value=polyline_length(planar, closed=True), unit=length_unit, **common))
        else:  # line, path
            out.append(Measurement(measurement_type="Length", value=polyline_length(path), unit=length_unit, **common))
            if shape_type == "line" and path.shape[0] == 2:
                delta = np.abs(path[1] - path[0])
                out.append(Measurement(measurement_type="ΔY", value=float(delta[0]), unit=length_unit, **common))
                out.append(Measurement(measurement_type="ΔX", value=float(delta[-1]), unit=length_unit, **common))
    return out


def measure_all(viewer) -> list[Measurement]:
    """Measure every ROI Shapes layer in the viewer, in layer order."""
    out: list[Measurement] = []
    for layer in viewer.layers:
        if _is_shapes(layer):
            ensure_name_feature(layer)
            out.extend(measure_layer(layer, viewer))
    return out


def _is_shapes(layer) -> bool:
    return type(layer).__name__ == "Shapes"


#: Metadata flag a Shapes layer can carry to say it is not a measurement ROI
#: layer. The Brain regions panel sets it: its outlines are anatomy, named by
#: hand, and this panel renames every shape it finds to "ROI 1", "ROI 2", … —
#: which quietly overwrites the names before anyone can type them.
NOT_A_ROI = "mv_not_a_roi"


def shapes_layers(viewer) -> list[Any]:
    """Shapes layers this panel measures: all of them bar the opted-out ones."""
    return [
        layer
        for layer in viewer.layers
        if _is_shapes(layer) and not layer.metadata.get(NOT_A_ROI, False)
    ]


def new_roi_layer(viewer, image_layer=None, name: str | None = None):
    """Create a Shapes layer aligned to *image_layer* and select the polygon tool.

    Matching ``ndim`` and ``scale`` to the image is what makes the resulting
    coordinates directly convertible to micrometres.
    """
    from napari.layers import Image

    from .loaders.layer_spec import units_like

    if image_layer is None:
        image_layer = next(
            (layer for layer in reversed(viewer.layers) if isinstance(layer, Image)), None
        )

    kwargs: dict[str, Any] = {
        "name": name or "ROIs",
        "ndim": 2,
        "edge_color": "yellow",
        "face_color": "transparent",
        "edge_width": 2,
        "features": {NAME_FEATURE: np.empty(0, dtype=object)},
        "text": {
            "string": "{name}",
            "size": 9,
            "color": "yellow",
            "anchor": "upper_left",
            "translation": [-6, 0],
        },
    }
    if image_layer is not None:
        kwargs["ndim"] = int(image_layer.ndim)
        kwargs["scale"] = tuple(float(s) for s in image_layer.scale)
        # Without this the layer defaults to dimensionless "pixel" units, which
        # makes the whole layer list inconsistent and sends the scale bar back to
        # reading pixels over a calibrated image.
        kwargs.update(units_like(image_layer, int(image_layer.ndim)))
        kwargs["name"] = name or f"ROIs — {image_layer.name}"
        kwargs["metadata"] = {
            ROI_KEY: True,
            TARGET_KEY: image_layer.name,
            "mv_metadata": image_layer.metadata.get("mv_metadata"),
            "mv_image_name": image_layer.name,
            "mv_channel_name": image_layer.metadata.get("mv_channel_name", ""),
            "mv_axes": image_layer.metadata.get("mv_axes", ""),
        }
    else:
        kwargs["metadata"] = {ROI_KEY: True}

    layer = viewer.add_shapes(**kwargs)
    layer.mode = "add_polygon"
    return layer
