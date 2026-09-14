"""Optional reader for OME-Zarr / NGFF stores.

Implemented directly against :mod:`zarr` and the NGFF ``.zattrs`` conventions so
no extra dependency is required. Handles the ``multiscales`` pyramid, the ``axes``
list (NGFF v0.3+) and the ``omero`` rendering block when present.

Three store layouts are recognised:

* a plain image group, read as one layer per channel;
* an HCS ``plate`` group, assembled into a lazy well-grid mosaic;
* a ``bioformats2raw.layout`` group, read as one layer set per series.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
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
from .layer_spec import LayerSpec, apply_squeeze, channel_appearance, squeeze_plan

logger = get_logger("ome_zarr")

#: Files that mark a folder as a Zarr group. zarr 3 writes ``zarr.json`` for every
#: group; zarr 2 writes ``.zattrs`` only when the group has attributes of its own,
#: so a plate row — which has none — is marked by ``.zgroup`` alone.
ZARR_MARKERS = (".zattrs", ".zgroup", "zarr.json")


def can_read(path: Path) -> bool:
    if path.suffix.lower() in (".zarr", ".ngff"):
        return True
    if path.is_dir() and any((path / marker).exists() for marker in ZARR_MARKERS):
        return True
    return False


def _group_class():
    """``zarr.Group`` moved between zarr 2 and 3; resolve it at call time."""
    return getattr(zarr, "Group", None) or zarr.hierarchy.Group  # type: ignore[attr-defined]


def open_group(path: Path):
    try:
        node = zarr.open(str(path), mode="r")
    except Exception as exc:
        raise ValueError(f"{path.name} is not a readable Zarr store: {exc}") from exc
    if not isinstance(node, _group_class()):
        raise ValueError(f"{path.name} is a bare Zarr array, not an OME-Zarr image group")
    return node


def ngff_attrs(node) -> dict[str, Any]:
    """Group attributes with the NGFF keys flattened.

    NGFF up to v0.4 writes ``multiscales``, ``omero``, ``plate`` and ``well`` at
    the top of ``.zattrs``; v0.5 nests the same keys under an ``ome`` block. Both
    are flattened here so the rest of the module only has one shape to know about.
    """
    attrs = dict(node.attrs)
    nested = attrs.get("ome")
    if isinstance(nested, dict):
        merged = dict(attrs)
        merged.update(nested)
        return merged
    return attrs


def child_group(group, component: str):
    """A subgroup by path, or None when it is absent or is an array."""
    try:
        node = group[component]
    except (KeyError, TypeError):
        return None
    return node if isinstance(node, _group_class()) else None


def _multiscales(group) -> dict[str, Any] | None:
    entries = ngff_attrs(group).get("multiscales")
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


#: Image names converters write when they have nothing better to write. Fractal
#: puts ``TBD`` in every image of a plate, so taking it at face value turns a
#: 46-well plate into several hundred layers with the same name; where the path
#: through the store says which well and cycle this is, that is worth more.
PLACEHOLDER_NAMES = frozenset({"tbd", "untitled", "unnamed", "image", "none", "n/a", "-"})


def _parse_omero(group, meta: AcquisitionMetadata) -> None:
    """Channel names, colours and display windows from the ``omero`` attribute block."""
    omero = ngff_attrs(group).get("omero")
    if not isinstance(omero, dict):
        return
    name = str(omero.get("name") or "").strip()
    if name and name.lower() not in PLACEHOLDER_NAMES:
        meta.image_name = name
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


# ---------------------------------------------------------------------------
# One image group
# ---------------------------------------------------------------------------


def _image_levels(group, label: str) -> tuple[list[Any], str, list[str], list[float] | None]:
    """Pyramid levels of an image group, plus its axes, axis units and level-0 scale."""
    entry = _multiscales(group)

    if entry is None:
        # Not NGFF: fall back to the first array in the group.
        names = [key for key, value in group.arrays()]
        if not names:
            raise ValueError(f"{label} has no arrays and no multiscales metadata")
        levels = [da.from_zarr(group[names[0]])]
        return levels, guess_axes(levels[0].shape), [], None

    datasets = entry.get("datasets") or []
    if not datasets:
        raise ValueError(f"{label}: multiscales entry has no datasets")
    levels = []
    for dataset in datasets:
        component = str(dataset.get("path", ""))
        try:
            levels.append(da.from_zarr(group[component]))
        except KeyError:
            logger.warning("%s: multiscales references missing array %r", label, component)
    if not levels:
        raise ValueError(f"{label}: none of the multiscales datasets could be opened")
    axes, axes_units = _axes_from_ngff(entry, levels[0].ndim)
    if not axes:
        axes = guess_axes(levels[0].shape)
        axes_units = [""] * len(axes)
    return levels, axes, axes_units, _dataset_scale(datasets[0])


def _image_metadata(
    path: Path,
    group,
    image_name: str,
    axes: str,
    axes_units: list[str],
    level_scale: list[float] | None,
    file_format: str = "OME-Zarr",
    keep_name: bool = False,
) -> AcquisitionMetadata:
    """Calibrated metadata for one image group.

    *keep_name* holds on to the name the caller passed instead of letting the
    ``omero`` block replace it. That matters for a plate: every image of a
    Fractal-converted plate carries the same placeholder ``omero`` name — ``TBD``
    — so taking it would throw away the one thing that identifies the layer, which
    is the well and the acquisition the caller worked out.
    """
    meta = AcquisitionMetadata(image_name=image_name, file_path=path, file_format=file_format)
    if level_scale:
        _apply_ngff_scale(meta, axes, level_scale, axes_units or [""] * len(axes))
    _parse_omero(group, meta)
    if keep_name:
        meta.image_name = image_name
    meta.harvest(
        {
            key: value
            for key, value in ngff_attrs(group).items()
            if key not in ("multiscales", "omero", "ome", "plate", "well", "image-label", "labels")
        }
    )
    return meta


def _build_specs(levels: list[Any], axes: str, meta: AcquisitionMetadata) -> list[LayerSpec]:
    """Split the channel axis out and turn each channel into a :class:`LayerSpec`."""
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

    # Plate and slide-scanner stores keep a Z axis one plane deep. Carrying it
    # onto the layer costs a slider that cannot move and makes every downstream
    # panel treat a 2D image as a volume, so drop it here as the Imaris reader does.
    layer_shape = tuple(shape[i] for i, axis in enumerate(axes) if axis != "C")
    keep = squeeze_plan(layer_shape, layer_axes)
    if len(keep) != len(layer_axes):
        per_channel = [
            [apply_squeeze(level, keep, len(layer_axes)) for level in arrays]
            for arrays in per_channel
        ]
        layer_axes = "".join(layer_axes[i] for i in keep)

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
    return specs


#: Subgroup of an image that holds segmentations, per the NGFF image-label spec.
LABELS_GROUP = "labels"

#: How many label sets to load from one image. A store written by a pipeline that
#: runs several segmentations accumulates them, and loading twenty at once would
#: bury the channels they belong to.
MAX_LABEL_SETS = 8


def _label_names(group) -> list[str]:
    """Names of the label sets an image group carries, in the order it lists them."""
    node = child_group(group, LABELS_GROUP)
    if node is None:
        return []
    listed = ngff_attrs(node).get(LABELS_GROUP)
    if isinstance(listed, list):
        return [str(name) for name in listed]
    return sorted(key for key, _value in node.groups())


def _label_group(group, name: str):
    """One label set of an image group, or None."""
    node = child_group(group, LABELS_GROUP)
    return None if node is None else child_group(node, name)


def _read_labels(path: Path, group, image_name: str) -> list[LayerSpec]:
    """Segmentations stored beside the image, as Labels layers.

    NGFF keeps them in a ``labels`` subgroup, each one a multiscale image of its
    own on the same grid as the pixels. Reading them here is what makes a batch
    run visible: the masks come back into the viewer aligned with the channel
    they were computed from, without the user having to find the path.
    """
    specs: list[LayerSpec] = []
    for name in _label_names(group)[:MAX_LABEL_SETS]:
        label_group = _label_group(group, name)
        if label_group is None:
            logger.warning("%s: label set %r is listed but not in the store", image_name, name)
            continue
        label_name = f"{image_name} :: {name}"
        try:
            levels, axes, axes_units, level_scale = _image_levels(label_group, label_name)
            meta = _image_metadata(
                path, label_group, label_name, axes, axes_units, level_scale, keep_name=True
            )
            built = _build_specs(levels, axes, meta)
        except Exception:  # a broken label set must not cost you the image
            logger.exception("%s: label set %r could not be read", image_name, name)
            continue
        for spec in built:
            spec.layer_type = "labels"
        specs.extend(built)
    return specs


def _read_image(
    path: Path,
    group,
    image_name: str,
    file_format: str = "OME-Zarr",
    keep_name: bool = False,
) -> list[LayerSpec]:
    """Read one OME-Zarr image group into one :class:`LayerSpec` per channel."""
    levels, axes, axes_units, level_scale = _image_levels(group, image_name)
    meta = _image_metadata(
        path, group, image_name, axes, axes_units, level_scale, file_format, keep_name=keep_name
    )
    specs = _build_specs(levels, axes, meta)
    # Named off the metadata rather than the caller's label so a label set sits
    # next to the channels it came from in the layer list: "TBD :: nuclei" beside
    # "TBD :: Ab1_DAPI", not under a different name for the same image.
    specs.extend(_read_labels(path, group, meta.image_name))
    logger.info(
        "read %s: axes=%s shape=%s levels=%d",
        image_name,
        axes,
        tuple(int(n) for n in levels[0].shape),
        len(levels),
    )
    return specs


# ---------------------------------------------------------------------------
# bioformats2raw series
# ---------------------------------------------------------------------------


def _series_components(group, attrs: dict[str, Any]) -> list[str]:
    """Component paths of a ``bioformats2raw.layout`` store, or [] when it is not one.

    The converter writes each series as a numbered subgroup and records the order
    in the ``OME`` group's ``series`` attribute; fall back to numeric sorting of
    the subgroups when that attribute is missing.
    """
    if attrs.get("bioformats2raw.layout") != 3:
        return []

    ome = child_group(group, "OME")
    if ome is not None:
        listed = ngff_attrs(ome).get("series")
        if isinstance(listed, list) and listed:
            return [str(entry) for entry in listed]

    numbered = [str(key) for key, value in group.groups() if str(key).isdigit()]
    return sorted(numbered, key=int)


def _series_names(path: Path, count: int) -> list[str]:
    """Image names from the converter's ``OME/METADATA.ome.xml``, when it is there."""
    xml_path = path / "OME" / "METADATA.ome.xml"
    if not xml_path.exists():
        return []
    try:
        root = ET.parse(str(xml_path)).getroot()
    except Exception:
        logger.debug("could not parse %s", xml_path, exc_info=True)
        return []
    names = [
        str(element.get("Name") or "").strip()
        for element in root.iter()
        if element.tag.rsplit("}", 1)[-1] == "Image"
    ]
    return names if len(names) >= count else []


def _read_series(path: Path, group, components: list[str]) -> list[LayerSpec]:
    """Read every series of a bioformats2raw store, labelled by name or index."""
    names = _series_names(path, len(components))
    specs: list[LayerSpec] = []
    for index, component in enumerate(components):
        image = child_group(group, component)
        if image is None:
            logger.warning("%s: series %r is missing", path.name, component)
            continue
        label = names[index] if index < len(names) and names[index] else f"series {component}"
        try:
            specs.extend(_read_image(path, image, f"{path.stem} :: {label}"))
        except Exception as exc:
            logger.warning("%s: series %r could not be read: %s", path.name, component, exc)
    if not specs:
        raise ValueError(f"{path.name}: no series of this bioformats2raw store could be read")
    logger.info("read %s: %d bioformats2raw series", path.name, len(components))
    return specs


# ---------------------------------------------------------------------------
# HCS plates
# ---------------------------------------------------------------------------


def _well_images(well_group, label: str) -> list[tuple[int | None, str, Any]]:
    """``(acquisition, path, group)`` for each image a well lists.

    The acquisition id is what separates a 4i plate from a multi-field one: images
    of the same well that differ in acquisition are the *same* field imaged again
    after restaining, not a different part of the well. Wells that list no images
    fall back to their subgroups, which is how a hand-built store still opens.
    """
    block = ngff_attrs(well_group).get("well", {})
    listed = block.get("images") if isinstance(block, dict) else None
    if not isinstance(listed, list) or not listed:
        listed = [
            {"path": key}
            for key, _value in sorted(well_group.groups(), key=lambda item: _sort_key(item[0]))
        ]

    images: list[tuple[int | None, str, Any]] = []
    for entry in listed:
        if isinstance(entry, dict):
            component = str(entry.get("path", ""))
            acquisition = entry.get("acquisition")
        else:
            component, acquisition = str(entry), None
        image = child_group(well_group, component) if component else None
        if image is None:
            logger.warning("%s: image %r is missing", label, component)
            continue
        images.append((None if acquisition is None else int(acquisition), component, image))
    return images


def _acquisition_ids(cells: dict[tuple[int, int], list[tuple[int | None, str, Any]]]) -> list:
    """Every acquisition in the plate, in order, with ``None`` for a plate without any."""
    found: list = []
    for entries in cells.values():
        for acquisition, _component, _group in entries:
            if acquisition not in found:
                found.append(acquisition)
    return sorted(found, key=lambda value: (value is None, value))


def _plate_cells(
    group, plate: dict[str, Any]
) -> tuple[int, int, dict[tuple[int, int], list[tuple[int | None, str, Any]]], str]:
    """Well positions on the plate mapped to the images they hold.

    Returns the grid size, the cells keyed by position within that grid, and a
    label naming the region of the plate the grid covers. Each cell holds the
    well's images as ``(acquisition, path, group)``; splitting those into
    acquisitions is :func:`_read_plate`'s job.
    """
    rows = [str(entry.get("name", index)) for index, entry in enumerate(plate.get("rows") or [])]
    columns = [
        str(entry.get("name", index)) for index, entry in enumerate(plate.get("columns") or [])
    ]
    cells: dict[tuple[int, int], list[tuple[int | None, str, Any]]] = {}
    missing: list[str] = []

    for well in plate.get("wells") or []:
        if not isinstance(well, dict):
            continue
        component = str(well.get("path", ""))
        if not component:
            continue
        # rowIndex/columnIndex are required from v0.4 on; older stores carry only
        # the "row/column" path, which is looked up in the declared name lists.
        row_index = well.get("rowIndex")
        column_index = well.get("columnIndex")
        if row_index is None or column_index is None:
            parts = component.strip("/").split("/")
            if len(parts) != 2 or parts[0] not in rows or parts[1] not in columns:
                logger.warning("plate well %r has no usable position", component)
                continue
            row_index, column_index = rows.index(parts[0]), columns.index(parts[1])

        well_group = child_group(group, component)
        if well_group is None:
            # A plate is routinely written with its full layout in the metadata
            # and only the acquired wells on disk, so this is normal rather than
            # a fault: count them and say so once.
            missing.append(component)
            continue
        images = _well_images(well_group, f"plate well {component}")
        if images:
            cells[(int(row_index), int(column_index))] = images

    if missing:
        shown = ", ".join(missing[:6]) + ("…" if len(missing) > 6 else "")
        logger.info(
            "plate: %d of %d wells in the metadata are not in the store (%s)",
            len(missing),
            len(missing) + len(cells),
            shown,
        )
    if not cells:
        return 0, 0, {}, ""

    # Trim to the wells that are really there. A plate is commonly written with
    # the full layout in its metadata and only the acquired wells on disk, and
    # padding those out to the nominal 6x10 would build a mosaic tens of times
    # the size of the data for no gain.
    first_row, last_row = min(key[0] for key in cells), max(key[0] for key in cells)
    first_column = min(key[1] for key in cells)
    last_column = max(key[1] for key in cells)
    trimmed = {
        (row - first_row, column - first_column): images
        for (row, column), images in cells.items()
    }

    def well_name(row: int, column: int) -> str:
        row_name = rows[row] if row < len(rows) else str(row)
        column_name = columns[column] if column < len(columns) else str(column)
        return f"{row_name}/{column_name}"

    region = well_name(first_row, first_column)
    if (first_row, first_column) != (last_row, last_column):
        region = f"{region} – {well_name(last_row, last_column)}"
    return last_row - first_row + 1, last_column - first_column + 1, trimmed, region


def _grid(tiles: list[list[Any]], shape: tuple[int, ...], dtype, chunks) -> Any:
    """Block a 2D grid of equally shaped arrays along their last two axes.

    Empty cells become lazy zeros, so a partly filled plate still assembles.
    """
    filled = [
        [tile if tile is not None else da.zeros(shape, dtype=dtype, chunks=chunks) for tile in row]
        for row in tiles
    ]
    return da.block(filled)


def _mosaic(
    cells: dict[tuple[int, int], list[list[Any]]],
    n_rows: int,
    n_columns: int,
    field_rows: int,
    field_columns: int,
    level: int,
    reference: Any,
) -> Any:
    """One pyramid level of the plate, as a grid of wells each holding a grid of fields."""
    shape = tuple(int(n) for n in reference.shape)
    dtype, chunks = reference.dtype, reference.chunksize

    def field(arrays: list[Any] | None) -> Any:
        if arrays is None or level >= len(arrays):
            return None
        candidate = arrays[level]
        if tuple(int(n) for n in candidate.shape) != shape:
            logger.warning(
                "plate field shape %s does not match %s at level %d; left blank",
                tuple(int(n) for n in candidate.shape),
                shape,
                level,
            )
            return None
        return candidate

    plate_grid = []
    for row in range(n_rows):
        tiles = []
        for column in range(n_columns):
            fields = cells.get((row, column)) or []
            well_grid = [
                [
                    field(fields[index] if index < len(fields) else None)
                    for index in range(start, start + field_columns)
                ]
                for start in range(0, field_rows * field_columns, field_columns)
            ]
            tiles.append(_grid(well_grid, shape, dtype, chunks))
        plate_grid.append(tiles)
    return da.block(plate_grid)


def _plate_name(plate: dict[str, Any], path: Path) -> str:
    """What to call the plate: its own name, its acquisition's, or the folder's."""
    if plate.get("name"):
        return str(plate["name"])
    for acquisition in plate.get("acquisitions") or []:
        if isinstance(acquisition, dict) and acquisition.get("name"):
            return str(acquisition["name"])
    return path.stem


def _mosaic_pyramid(
    field_levels: dict,
    n_rows: int,
    n_columns: int,
    fields_per_well: int,
    reference_levels: list[Any],
) -> list[Any]:
    """Every pyramid level of the plate, as a grid of wells each holding its fields."""
    field_columns = max(1, math.ceil(math.sqrt(fields_per_well)))
    field_rows = max(1, math.ceil(fields_per_well / field_columns))
    return [
        _mosaic(field_levels, n_rows, n_columns, field_rows, field_columns, index, level)
        for index, level in enumerate(reference_levels)
    ]


def _plate_label_specs(
    path: Path,
    cells: dict,
    n_rows: int,
    n_columns: int,
    fields_per_well: int,
    image_name: str,
) -> list[LayerSpec]:
    """Segmentations of the plate, assembled on the same grid as the channels.

    A batch run leaves a ``labels`` group inside every image it segmented, and a
    plate is normally segmented well by well — so each label set is built into a
    mosaic of its own here, with the wells that have not been segmented left
    blank. Without this the masks exist but can only be looked at one well at a
    time, which is not how anyone reads a plate.
    """
    names: list[str] = []
    for images in cells.values():
        for image in images:
            for name in _label_names(image):
                if name not in names:
                    names.append(name)
    if not names:
        return []

    specs: list[LayerSpec] = []
    for name in names[:MAX_LABEL_SETS]:
        label_name = f"{image_name} :: {name}"
        reference = None
        pyramids: dict = {}
        for position, images in cells.items():
            # A field with no masks keeps its slot, so the tiles of a well that was
            # only partly segmented still land where they belong.
            found: list[Any] = []
            for image in images:
                group = _label_group(image, name)
                if group is None:
                    found.append(None)
                    continue
                try:
                    read = _image_levels(group, label_name)
                except ValueError as exc:
                    logger.warning("%s: %s", label_name, exc)
                    found.append(None)
                    continue
                if reference is None:
                    reference = (group, *read)
                found.append(read[0])
            if any(entry is not None for entry in found):
                pyramids[position] = found
        if reference is None:
            continue

        group, levels, axes, axes_units, level_scale = reference
        mosaic = _mosaic_pyramid(pyramids, n_rows, n_columns, fields_per_well, levels)
        meta = _image_metadata(
            path,
            group,
            label_name,
            axes,
            axes_units,
            level_scale,
            file_format="OME-Zarr (HCS plate)",
            keep_name=True,
        )
        meta.extra.setdefault("Label set", name)
        meta.extra.setdefault("Wells segmented", str(len(pyramids)))
        built = _build_specs(mosaic, axes, meta)
        for spec in built:
            spec.layer_type = "labels"
        specs.extend(built)
        logger.info("%s: label set %r assembled from %d well(s)", image_name, name, len(pyramids))
    return specs


def _read_plate_acquisition(
    path: Path,
    cells: dict,
    n_rows: int,
    n_columns: int,
    region: str,
    image_name: str,
    acquisition: int | None,
) -> list[LayerSpec]:
    """One acquisition of the plate: a mosaic per channel, plus any label sets."""
    reference_group = cells[sorted(cells)[0]][0]
    levels, axes, axes_units, level_scale = _image_levels(reference_group, image_name)
    if not axes.endswith("YX"):
        logger.warning(
            "%s: plate fields have axes %s, which cannot be tiled; reading the first field only",
            path.name,
            axes,
        )
        return _read_image(path, reference_group, image_name, keep_name=True)

    # Read every field's pyramid up front, then block each level into one mosaic.
    field_levels: dict = {}
    for position, fields in cells.items():
        pyramids = []
        for field_group in fields:
            try:
                pyramids.append(_image_levels(field_group, image_name)[0])
            except ValueError as exc:
                logger.warning("%s: skipping a field of well %s: %s", path.name, position, exc)
        if pyramids:
            field_levels[position] = pyramids
    if not field_levels:
        raise ValueError(f"{path.name}: none of the plate fields could be read")

    fields_per_well = max(len(pyramids) for pyramids in field_levels.values())
    mosaic = _mosaic_pyramid(field_levels, n_rows, n_columns, fields_per_well, levels)

    meta = _image_metadata(
        path,
        reference_group,
        image_name,
        axes,
        axes_units,
        level_scale,
        file_format="OME-Zarr (HCS plate)",
        keep_name=True,
    )
    meta.extra.setdefault("Plate layout", f"{n_rows} rows x {n_columns} columns")
    if region:
        meta.extra.setdefault("Wells assembled", f"{len(field_levels)} ({region})")
    meta.extra.setdefault("Fields per well", str(fields_per_well))
    if acquisition is not None:
        meta.extra.setdefault("Acquisition", str(acquisition))

    specs = _build_specs(mosaic, axes, meta)
    specs.extend(_plate_label_specs(path, cells, n_rows, n_columns, fields_per_well, image_name))
    logger.info(
        "read %s: %d well(s) (%s), %d field(s) per well, levels=%d",
        image_name,
        len(field_levels),
        region or "no wells",
        fields_per_well,
        len(mosaic),
    )
    return specs


def _read_plate(path: Path, group, plate: dict[str, Any]) -> list[LayerSpec]:
    """Assemble an HCS plate into one lazy mosaic layer per channel.

    Acquisitions are **channels, not places.** A 4i plate images the same wells
    once per staining cycle, and the cycles are listed inside the well exactly as
    several fields would be. Tiling them the way fields are tiled would lay the
    same cells out seven times side by side and build a mosaic nine times the size
    of the plate. Each acquisition is assembled on the plate grid of its own
    instead, and comes back as a further set of channels — registered on top of
    the others, because that is where they were imaged.
    """
    n_rows, n_columns, cells, region = _plate_cells(group, plate)
    if not cells:
        raise ValueError(f"{path.name}: plate metadata lists no readable wells")

    plate_name = _plate_name(plate, path)
    acquisitions = _acquisition_ids(cells)
    specs: list[LayerSpec] = []
    for acquisition in acquisitions:
        subset = {}
        for position, images in cells.items():
            groups = [
                image
                for entry_acquisition, _component, image in images
                if entry_acquisition == acquisition
            ]
            if groups:
                subset[position] = groups
        if not subset:
            continue
        image_name = plate_name
        if acquisition is not None and len(acquisitions) > 1:
            image_name = f"{plate_name} :: cycle {acquisition}"
        try:
            built = _read_plate_acquisition(
                path, subset, n_rows, n_columns, region, image_name, acquisition
            )
        except ValueError as exc:
            logger.warning("%s: %s", image_name, exc)
            continue
        for spec in built:
            spec.metadata.extra.setdefault("Plate name", plate_name)
            spec.metadata.extra.setdefault("Acquisitions", str(len(acquisitions)))
        specs.extend(built)

    if not specs:
        raise ValueError(f"{path.name}: none of the plate wells could be read")
    logger.info(
        "read %s: HCS plate %dx%d wells, %d acquisition(s), %d layer(s)",
        path.name,
        n_rows,
        n_columns,
        len(acquisitions),
        len(specs),
    )
    return specs


def _read_well(path: Path, group, well: dict[str, Any], label: str) -> list[LayerSpec]:
    """Read one well group, one layer set per acquisition and field.

    Opening one well rather than the whole plate is what you do when a plate is
    too big to look at at once, and dropping the well folder on the window is the
    obvious way to ask for it. The cycle goes in the layer name because the
    ``omero`` name of a converted plate is the same placeholder in every image of
    it, and without the cycle the seven stainings are seven layers called the same
    thing.
    """
    images = _well_images(group, label)
    if not images:
        raise ValueError(f"{label}: well metadata lists no readable images")

    # "G/06" rather than the whole store path: the well is what identifies these
    # layers, and prefixing every one of them with the plate's name — which on a
    # converted plate is forty characters — only pushes the useful part off the
    # end of the layer list.
    label = label.rpartition(" :: ")[2] or label

    acquisitions = _acquisition_ids({(0, 0): images})
    per_acquisition: dict = {}
    for acquisition, _component, _image in images:
        per_acquisition[acquisition] = per_acquisition.get(acquisition, 0) + 1

    specs: list[LayerSpec] = []
    for acquisition, component, image in images:
        parts = [label]
        if acquisition is not None and len(acquisitions) > 1:
            parts.append(f"cycle {acquisition}")
        if per_acquisition[acquisition] > 1:
            parts.append(f"field {component}")
        try:
            specs.extend(_read_image(path, image, " :: ".join(parts), keep_name=True))
        except ValueError as exc:
            logger.warning("%s: image %r could not be read: %s", label, component, exc)
    if not specs:
        raise ValueError(f"{label}: none of the well's images could be read")
    logger.info(
        "read %s: well with %d image(s) in %d acquisition(s)",
        label,
        len(images),
        len(acquisitions),
    )
    return specs


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

#: How far the reader will descend through plain groups looking for images, and
#: how many it will return from such a descent. A store handed to us without any
#: layout metadata — a plate row, a converter's output folder, a hand-built
#: hierarchy — is still worth opening, but not at the cost of walking a whole
#: plate one field at a time.
MAX_DESCENT_DEPTH = 3
MAX_DESCENT_IMAGES = 64


def _sort_key(name: str) -> tuple[int, int, str]:
    """Sort ``2`` before ``10``, and numbered groups before named ones."""
    text = str(name)
    return (0, int(text), "") if text.isdigit() else (1, 0, text)


def _read_children(path: Path, group, label: str, depth: int) -> list[LayerSpec]:
    """Descend into subgroups when a group carries no image data of its own."""
    children = sorted(group.groups(), key=lambda item: _sort_key(item[0]))
    if depth >= MAX_DESCENT_DEPTH or not children:
        contents = ", ".join(str(key) for key, _ in children)
        raise ValueError(
            f"{label} has no arrays and no multiscales metadata"
            + (f"; its subgroups ({contents}) hold none either" if contents else "")
        )

    specs: list[LayerSpec] = []
    for key, child in children:
        try:
            specs.extend(_read_node(path, child, f"{label} :: {key}", depth + 1))
        except ValueError as exc:
            logger.debug("%s: skipping subgroup %r: %s", label, key, exc)
        if len(specs) >= MAX_DESCENT_IMAGES:
            logger.warning(
                "%s: stopping after %d layers; open a subgroup directly for the rest",
                label,
                len(specs),
            )
            break
    if not specs:
        contents = ", ".join(str(key) for key, _ in children)
        raise ValueError(f"{label} has no image data, and neither do its subgroups ({contents})")
    return specs


def _read_node(path: Path, group, label: str, depth: int = 0) -> list[LayerSpec]:
    """Read whichever NGFF layout *group* turns out to be."""
    attrs = ngff_attrs(group)

    plate = attrs.get("plate")
    if isinstance(plate, dict):
        return _read_plate(path, group, plate)

    well = attrs.get("well")
    if isinstance(well, dict):
        return _read_well(path, group, well, label)

    series = _series_components(group, attrs)
    if series:
        return _read_series(path, group, series)

    if _multiscales(group) is not None or any(True for _ in group.arrays()):
        return _read_image(path, group, label)

    return _read_children(path, group, label, depth)


def _store_label(path: Path) -> str:
    """A name that says where in the store *path* is.

    Opening a well or a field means opening a folder called ``1`` or ``0``, which
    on its own names nothing. Walking up to the store root and keeping the path
    from there turns that into ``AssayPlate :: B/03/0``.
    """
    root = path
    for parent in path.parents:
        if parent.suffix.lower() in (".zarr", ".ngff"):
            root = parent
            break
        # A row of a plate is a group with no attributes of its own — only a
        # ``.zgroup`` — so testing for ``.zattrs`` alone stops the walk one level
        # too early and a well ends up named "06".
        if not any((parent / marker).exists() for marker in ZARR_MARKERS):
            break
        root = parent
    if root == path:
        return path.stem
    return f"{root.stem} :: {path.relative_to(root).as_posix()}"


def read(path: Path) -> list[LayerSpec]:
    """Read an OME-Zarr store and return one :class:`LayerSpec` per channel."""
    path = Path(path)
    return _read_node(path, open_group(path), _store_label(path))
