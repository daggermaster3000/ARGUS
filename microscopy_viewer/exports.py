"""Export paths: canvas snapshots for slides, and measurements as a spreadsheet."""

from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .measurements import COLUMN_LABELS, COLUMNS, Measurement
from .metadata import AcquisitionMetadata
from .utils import get_logger

logger = get_logger("exports")

SNAPSHOT_FILTER = ";;".join(
    ["PNG image (*.png)", "TIFF image (*.tif *.tiff)", "JPEG image (*.jpg *.jpeg)"]
)
WORKBOOK_FILTER = "Excel workbook (*.xlsx)"


def default_stem(prefix: str) -> str:
    """Timestamped filename stem so repeated exports never overwrite each other."""
    return f"{prefix}_{_dt.datetime.now():%Y%m%d_%H%M%S}"


# ---------------------------------------------------------------------------
# Snapshots
# ---------------------------------------------------------------------------


def export_snapshot(
    viewer,
    path: str | Path,
    canvas_only: bool = True,
    scale: float = 2,
    include_scale_bar: bool = True,
) -> Path:
    """Save the current napari view as an image file.

    ``scale`` oversamples the canvas so the result stays sharp when dropped into
    a PowerPoint slide. It is rounded to a whole number: napari multiplies the
    canvas size in place as an integer array, so a fractional factor raises a
    casting error. A TIFF suffix writes the pixel data losslessly and, when the
    image is calibrated, tags the file with the on-screen resolution.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    oversample = max(1, int(round(float(scale))))

    previous = viewer.scale_bar.visible
    if include_scale_bar and canvas_only and not previous:
        viewer.scale_bar.visible = True
    try:
        frame = viewer.screenshot(canvas_only=canvas_only, flash=False, scale=oversample)
    finally:
        viewer.scale_bar.visible = previous

    frame = np.asarray(frame)
    suffix = path.suffix.lower()
    if suffix in (".tif", ".tiff"):
        import tifffile

        resolution = _screen_resolution(viewer, oversample)
        kwargs = {"photometric": "rgb"}
        if resolution is not None:
            kwargs.update({"resolution": resolution, "resolutionunit": "CENTIMETER"})
        tifffile.imwrite(str(path), frame, **kwargs)
    else:
        import imageio.v3 as iio

        if suffix in (".jpg", ".jpeg") and frame.shape[-1] == 4:
            frame = frame[..., :3]  # JPEG cannot carry an alpha channel
        iio.imwrite(str(path), frame)

    logger.info("snapshot written to %s (%s px)", path, "x".join(str(n) for n in frame.shape[:2]))
    return path


def _screen_resolution(viewer, oversample: float) -> tuple[float, float] | None:
    """Pixels per centimetre of the saved image, if the canvas zoom is known.

    ``viewer.camera.zoom`` is screen pixels per world unit, and world units are
    micrometres for calibrated data, so the product gives pixels per µm.
    """
    try:
        zoom = float(viewer.camera.zoom) * float(oversample)
    except (TypeError, ValueError, AttributeError):
        return None
    if zoom <= 0:
        return None
    per_cm = zoom * 1e4  # pixels per µm -> pixels per cm
    return (per_cm, per_cm)


# ---------------------------------------------------------------------------
# Measurements workbook
# ---------------------------------------------------------------------------


def measurements_dataframe(measurements: Sequence[Measurement]):
    """A :class:`pandas.DataFrame` of measurements with friendly column names."""
    import pandas as pd

    rows = [m.as_row() for m in measurements]
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=[COLUMN_LABELS[c] for c in COLUMNS])

    ordered = [c for c in COLUMNS if c in frame.columns]
    ordered += [c for c in frame.columns if c not in ordered]
    frame = frame[ordered]
    return frame.rename(columns=COLUMN_LABELS)


def export_measurements(
    measurements: Sequence[Measurement],
    path: str | Path,
    metadata: Sequence[AcquisitionMetadata] = (),
) -> Path:
    """Write measurements to ``.xlsx``.

    Sheet ``Measurements`` holds one row per reported quantity. When acquisition
    metadata is supplied a second sheet records it, so a shared spreadsheet still
    says which settings produced the numbers.
    """
    path = Path(path)
    if path.suffix.lower() != ".xlsx":
        path = path.with_suffix(".xlsx")
    path.parent.mkdir(parents=True, exist_ok=True)

    import pandas as pd

    frame = measurements_dataframe(measurements)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        frame.to_excel(writer, sheet_name="Measurements", index=False)
        _autofit(writer.sheets["Measurements"], frame)

        if metadata:
            meta_frame = _metadata_frame(pd, metadata)
            meta_frame.to_excel(writer, sheet_name="Acquisition", index=False)
            _autofit(writer.sheets["Acquisition"], meta_frame)

    logger.info("wrote %d measurement rows to %s", len(frame), path)
    return path


def export_table(frame, path: str | Path, sheet_name: str = "Results") -> Path:
    """Write any already-built dataframe to ``.xlsx`` or ``.csv``.

    The measurements workbook knows the shape of a :class:`Measurement`; this is
    for tables that do not have one — the per-region signal readout, in
    particular. Same column autofit and frozen header, so the two exports look
    like they came from the same program, which they did.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix.lower() == ".csv":
        frame.to_csv(path, index=False)
    else:
        if path.suffix.lower() != ".xlsx":
            path = path.with_suffix(".xlsx")
        import pandas as pd

        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            frame.to_excel(writer, sheet_name=sheet_name, index=False)
            _autofit(writer.sheets[sheet_name], frame)

    logger.info("wrote %d row(s) to %s", len(frame), path)
    return path


def export_sheets(frames: dict[str, Any], path: str | Path) -> Path:
    """Write several dataframes to one workbook, one sheet each.

    :func:`export_table` opens the file fresh, so calling it twice leaves only
    the second table. A summary and the rows behind it belong in one file — the
    per-region counts and the objects they were counted from, in particular — and
    this is what puts them there.

    Falls back to a sheet-suffixed ``.csv`` per frame when a CSV path is given,
    since CSV has no notion of a sheet.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix.lower() == ".csv":
        written = path
        for index, (name, frame) in enumerate(frames.items()):
            target = path if index == 0 else path.with_name(f"{path.stem}_{name.lower()}.csv")
            frame.to_csv(target, index=False)
            written = path
        return written

    if path.suffix.lower() != ".xlsx":
        path = path.with_suffix(".xlsx")
    import pandas as pd

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, frame in frames.items():
            sheet = str(name)[:31]  # Excel's own limit
            frame.to_excel(writer, sheet_name=sheet, index=False)
            _autofit(writer.sheets[sheet], frame)

    logger.info("wrote %d sheet(s) to %s", len(frames), path)
    return path


def _metadata_frame(pd, metadata: Sequence[AcquisitionMetadata]):
    """One long-format table of every dataset's acquisition settings."""
    rows = []
    seen: set[int] = set()
    for meta in metadata:
        if meta is None or id(meta) in seen:
            continue
        seen.add(id(meta))
        for label, value in meta.summary_rows():
            rows.append({"Image": meta.image_name, "Scope": "Dataset", "Property": label, "Value": value})
        for channel in meta.channels:
            for label, value in channel.rows():
                rows.append(
                    {
                        "Image": meta.image_name,
                        "Scope": channel.display_name,
                        "Property": label,
                        "Value": value,
                    }
                )
        for label, value in meta.extra_rows():
            rows.append({"Image": meta.image_name, "Scope": "Other", "Property": label, "Value": value})
    if not rows:
        return pd.DataFrame(columns=["Image", "Scope", "Property", "Value"])
    return pd.DataFrame(rows)


def _autofit(worksheet, frame) -> None:
    """Widen columns to fit their contents, capped so long paths stay readable."""
    from openpyxl.utils import get_column_letter

    for index, column in enumerate(frame.columns, start=1):
        widest = len(str(column))
        for value in frame[column].head(500):
            widest = max(widest, len(str(value)))
        worksheet.column_dimensions[get_column_letter(index)].width = min(max(widest + 2, 10), 60)
    worksheet.freeze_panes = "A2"
