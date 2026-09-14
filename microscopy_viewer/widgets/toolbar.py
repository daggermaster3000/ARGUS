"""The compact button bar docked above the napari canvas."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ..contrast import auto_contrast, reset_contrast
from ..exports import SNAPSHOT_FILTER, default_stem, export_snapshot
from ..loaders import FILE_DIALOG_FILTER
from ..utils import get_logger

logger = get_logger("toolbar")

#: Above this many pixels, flattening is confirmed first: a plugin handed the copy
#: reads all of it, and a plate mosaic level is tens of gigabytes.
_FLATTEN_WARN_PIXELS = 500_000_000


class ViewerToolbar(QWidget):
    """One row of buttons for the actions used most often during inspection.

    The widget owns no state: every button delegates to the
    :class:`~microscopy_viewer.app.MicroscopyViewer` that created it.
    """

    def __init__(self, app, parent: QWidget | None = None):
        super().__init__(parent)
        self._app = app
        self._viewer = app.viewer

        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 2, 4, 2)
        outer.setSpacing(2)

        row = QHBoxLayout()
        row.setSpacing(4)
        outer.addLayout(row)

        self._buttons: dict[str, QPushButton] = {}
        for key, label, tooltip, handler in (
            ("open", "Open Images", "Open one or more microscopy datasets (Ctrl+O)", self.open_images),
            ("snapshot", "Export Snapshot", "Save the current view as PNG or TIFF (Ctrl+S)", self.export_snapshot),
            ("measurements", "Export Measurements", "Write all ROI measurements to an Excel workbook (Ctrl+E)", self.export_measurements),
            ("slide", "Export Slide", "Build a PowerPoint figure: one row per dataset, one column per channel, plus the merge (Ctrl+P)", self.export_slide),
            ("movie", "Export Movie", "Write the time series as a .mov for PowerPoint (Ctrl+Shift+M)", self.export_movie),
            ("play", "Play / Pause", "Play the time series (Ctrl+Space)", self.toggle_play),
            ("auto", "Auto Contrast", "Stretch contrast to the visible data (Ctrl+Shift+A)", self.auto_contrast),
            ("reset", "Reset Contrast", "Restore the full intensity range", self.reset_contrast),
            ("ndisplay", "2D / 3D (MIP)", "Switch between slice view and 3D maximum-intensity projection (Ctrl+D)", self.toggle_ndisplay),
            ("scalebar", "Toggle Scale Bar", "Show or hide the calibrated scale bar (Ctrl+B)", self.toggle_scale_bar),
            ("metadata", "Show/Hide Metadata", "Toggle the acquisition metadata panel (Ctrl+M)", self.toggle_metadata),
            ("flatten", "Flatten for Plugins", "Copy the level on screen out of a pyramid layer as a plain layer, which is what most napari plugins can actually read (Ctrl+Shift+L)", self.flatten_layer),
        ):
            button = QPushButton(label)
            button.setToolTip(tooltip)
            button.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
            button.clicked.connect(handler)
            row.addWidget(button)
            self._buttons[key] = button
        row.addStretch(1)

        self._status = QLabel("")
        self._status.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._status.setWordWrap(False)
        outer.addWidget(self._status)

    # -- helpers --------------------------------------------------------------

    def set_status(self, text: str) -> None:
        self._status.setText(text)

    def _last_directory(self) -> str:
        return str(self._app.last_directory or Path.home())

    # -- actions --------------------------------------------------------------

    def open_images(self) -> None:
        """Pick one or more datasets and add them to the viewer."""
        paths, _selected = QFileDialog.getOpenFileNames(
            self, "Open microscopy images", self._last_directory(), FILE_DIALOG_FILTER
        )
        if not paths:
            return
        self._app.open_paths(paths)

    def export_snapshot(self) -> None:
        """Save the canvas to an image file chosen by the user."""
        if not len(self._viewer.layers):
            QMessageBox.information(self, "Microscopy Viewer", "Open an image first.")
            return
        suggested = str(Path(self._last_directory()) / f"{default_stem('snapshot')}.png")
        path, _selected = QFileDialog.getSaveFileName(
            self, "Export snapshot", suggested, SNAPSHOT_FILTER
        )
        if not path:
            return
        try:
            written = export_snapshot(self._viewer, path)
        except Exception as exc:
            logger.exception("snapshot export failed")
            QMessageBox.critical(self, "Microscopy Viewer", f"Snapshot failed:\n{exc}")
            return
        self._app.last_directory = Path(written).parent
        self.set_status(f"Snapshot saved to {written}")

    def flatten_layer(self) -> None:
        """Copy one pyramid level out as an ordinary layer, for plugins to work on.

        Plugins that take ``napari.types.ImageData`` are handed the layer's data
        as-is, and for a multiscale layer that is ``MultiScaleData`` — a sequence
        of levels rather than an array. napari-simpleitk-image-processing,
        napari-segment-blobs-and-things and the rest raise on it, and every
        OME-Zarr layer here is multiscale, so this makes them usable at all.
        """
        from napari.layers import Image

        from ..rendering import displayed_voxels, plain_level

        layer = self._selected_image()
        if layer is None:
            QMessageBox.information(
                self, "Microscopy Viewer", "Select an image layer in the layer list first."
            )
            return
        if not getattr(layer, "multiscale", False) or len(layer.data) < 2:
            self.set_status(f"“{layer.name}” is already a plain layer; plugins can read it.")
            return

        try:
            data, scale, level = plain_level(layer)
        except Exception as exc:
            logger.exception("could not flatten %s", layer.name)
            QMessageBox.critical(self, "Microscopy Viewer", f"Could not flatten that layer:\n{exc}")
            return

        shape = tuple(int(n) for n in data.shape)
        # A plugin calls np.asarray on what it is given, so the copy is about to be
        # read into memory in full. Say so before that happens rather than after.
        pixels = displayed_voxels(shape)
        if pixels > _FLATTEN_WARN_PIXELS:
            gigabytes = pixels * int(np.dtype(data.dtype).itemsize) / 1e9
            answer = QMessageBox.question(
                self,
                "Flatten for plugins",
                f"Level {level} of “{layer.name}” is {'×'.join(str(n) for n in shape)}. "
                f"A plugin reading it will pull about {gigabytes:.1f} GB into memory.\n\n"
                "Zoom in first and the viewer will be on a finer, smaller level. Continue?",
                QMessageBox.Ok | QMessageBox.Cancel,
                QMessageBox.Cancel,
            )
            if answer != QMessageBox.Ok:
                return

        name = f"{layer.name} [level {level}]"
        if name in self._viewer.layers:
            del self._viewer.layers[name]
        try:
            self._viewer.add_image(
                data,
                name=name,
                scale=scale,
                colormap=layer.colormap,
                blending=layer.blending,
                contrast_limits=tuple(layer.contrast_limits),
                metadata=dict(layer.metadata),
            )
        except Exception as exc:
            logger.exception("could not add the flattened layer")
            QMessageBox.critical(self, "Microscopy Viewer", f"Could not add that layer:\n{exc}")
            return
        self.set_status(
            f"Added “{name}” ({'×'.join(str(n) for n in shape)}) — a plain layer plugins can read."
        )

    def _selected_image(self):
        """The selected Image layer, falling back to the last one in the list."""
        from napari.layers import Image

        selected = [
            layer for layer in getattr(self._viewer.layers, "selection", ()) if isinstance(layer, Image)
        ]
        if selected:
            return selected[-1]
        images = [layer for layer in self._viewer.layers if isinstance(layer, Image)]
        return images[-1] if images else None

    def export_measurements(self) -> None:
        self._app.measurements_widget.export()

    def export_slide(self) -> None:
        """Open the slide builder: samples down the table, channels across it.

        No guard on there being layers open: the dialog's *Add folder* is batch
        mode, which is the case where nothing has been loaded yet.
        """
        from .slide_dialog import SlideExportDialog

        dialog = SlideExportDialog(self._viewer, self._app.last_directory, parent=self)
        if not dialog.exec_() or dialog.written is None:
            return
        self._app.last_directory = Path(dialog.written).parent
        self.set_status(f"Slide saved to {dialog.written}")

    def export_movie(self) -> None:
        """Write the time series out as a movie.

        Goes through the panel when it is there, so the range being played is
        the range that gets exported; otherwise the dialog is opened directly,
        which is what happens if the panel failed to build.
        """
        panel = getattr(self._app, "timeseries_widget", None)
        if panel is not None:
            panel.export_movie()
            return

        from .movie_dialog import MovieExportDialog

        dialog = MovieExportDialog(self._viewer, self._app.last_directory, parent=self)
        if not dialog.exec_() or dialog.written is None:
            return
        self._app.last_directory = Path(dialog.written).parent
        self.set_status(f"Movie saved to {dialog.written}")

    def toggle_play(self) -> None:
        """Start or stop time-series playback."""
        panel = getattr(self._app, "timeseries_widget", None)
        if panel is None:
            self.set_status("The time-series panel is not available.")
            return
        panel.toggle_play()
        self.set_status("Playing…" if panel.playing else "Paused.")

    def auto_contrast(self) -> None:
        count = auto_contrast(self._viewer)
        self.set_status(
            f"Auto contrast applied to {count} layer(s)." if count else "No visible image layers to adjust."
        )

    def reset_contrast(self) -> None:
        count = reset_contrast(self._viewer)
        self.set_status(f"Contrast reset on {count} layer(s)." if count else "No image layers to reset.")

    def toggle_ndisplay(self) -> None:
        """Flip between 2D slices and a 3D maximum-intensity projection.

        Sets ``mip`` explicitly rather than relying on the layer default, so the
        button does what its label says even if a layer was left on another
        rendering mode.
        """
        from napari.layers import Image

        to_3d = self._viewer.dims.ndisplay == 2
        if to_3d and self._viewer.dims.ndim < 3:
            self.set_status("This dataset is 2D — there is nothing to project.")
            return

        self._viewer.dims.ndisplay = 3 if to_3d else 2
        if to_3d:
            for layer in self._viewer.layers:
                if isinstance(layer, Image):
                    layer.rendering = "mip"
        self.set_status(
            "3D maximum-intensity projection." if to_3d else "2D slice view."
        )

    def toggle_scale_bar(self) -> None:
        visible = not self._viewer.scale_bar.visible
        self._viewer.scale_bar.visible = visible
        self.set_status(f"Scale bar {'shown' if visible else 'hidden'}.")

    def toggle_metadata(self) -> None:
        self._app.toggle_metadata_panel()
