"""The compact button bar docked above the napari canvas."""

from __future__ import annotations

from pathlib import Path

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
            ("mip", "Batch MIP", "Maximum-project a whole folder of stacks to TIFF or Imaris files (Ctrl+Shift+P)", self.batch_projection),
            ("movie", "Export Movie", "Write the time series as a .mov for PowerPoint (Ctrl+Shift+M)", self.export_movie),
            ("play", "Play / Pause", "Play the time series (Ctrl+Space)", self.toggle_play),
            ("auto", "Auto Contrast", "Stretch contrast to the visible data (Ctrl+Shift+A)", self.auto_contrast),
            ("reset", "Reset Contrast", "Restore the full intensity range", self.reset_contrast),
            ("ndisplay", "2D / 3D (MIP)", "Switch between slice view and 3D maximum-intensity projection (Ctrl+D)", self.toggle_ndisplay),
            ("scalebar", "Toggle Scale Bar", "Show or hide the calibrated scale bar (Ctrl+B)", self.toggle_scale_bar),
            ("metadata", "Show/Hide Metadata", "Toggle the acquisition metadata panel (Ctrl+M)", self.toggle_metadata),
            ("tour", "Tour", "Walk through an experiment step by step, pointing at each control (Esc stops it)", self.start_tour),
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

    def batch_projection(self) -> None:
        """Flatten a folder of stacks, without opening any of them.

        No guard on there being layers open: like the slide builder's batch mode,
        the usual case is a folder nobody has loaded.
        """
        from .projection_dialog import ProjectionDialog

        dialog = ProjectionDialog(self._app.last_directory, parent=self)
        dialog.exec_()
        written = [outcome for outcome in dialog.outcomes if outcome.ok and not outcome.skipped]
        if written:
            self._app.last_directory = Path(written[0].written).parent
            self.set_status(f"Projected {len(written)} file(s) to {Path(written[0].written).parent}")

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

    def start_tour(self) -> None:
        self._app.start_tour()

    def toggle_metadata(self) -> None:
        self._app.toggle_metadata_panel()
