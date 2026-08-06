"""Dock widget for drawing ROIs and reading off calibrated measurements."""

from __future__ import annotations

from pathlib import Path

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .. import measurements as mm
from ..exports import WORKBOOK_FILTER, default_stem, export_measurements
from ..utils import format_number, get_logger

logger = get_logger("measurements_widget")

#: Drawing tools offered by the combo box, as (label, napari Shapes mode).
TOOLS = (
    ("Line (distance)", "add_line"),
    ("Polyline (path length)", "add_path"),
    ("Polygon (area)", "add_polygon"),
    ("Rectangle (area)", "add_rectangle"),
    ("Ellipse (area)", "add_ellipse"),
    ("Select / edit", "select"),
)


class MeasurementsWidget(QWidget):
    """Create ROI layers, list their measurements, and export them to Excel.

    The table is recomputed whenever a shape is added, edited or removed. The
    ROI-name column is editable and writes straight back into the Shapes layer's
    feature table, so names survive into the spreadsheet.
    """

    def __init__(self, viewer, parent: QWidget | None = None):
        super().__init__(parent)
        self._viewer = viewer
        self._rows: list[mm.Measurement] = []
        self._connected: set[int] = set()
        self._updating = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        controls = QGridLayout()
        controls.setHorizontalSpacing(6)
        controls.addWidget(QLabel("ROI layer:"), 0, 0)
        self._layer_box = QComboBox()
        self._layer_box.setToolTip("Which Shapes layer new ROIs are drawn into")
        self._layer_box.currentIndexChanged.connect(self._on_layer_chosen)
        controls.addWidget(self._layer_box, 0, 1)
        new_button = QPushButton("New ROI layer")
        new_button.setToolTip("Add a Shapes layer aligned to the active image")
        new_button.clicked.connect(self.new_roi_layer)
        controls.addWidget(new_button, 0, 2)

        controls.addWidget(QLabel("Tool:"), 1, 0)
        self._tool_box = QComboBox()
        for label, _mode in TOOLS:
            self._tool_box.addItem(label)
        self._tool_box.setCurrentIndex(2)  # polygon is the most common ROI
        self._tool_box.currentIndexChanged.connect(self._on_tool_chosen)
        controls.addWidget(self._tool_box, 1, 1, 1, 2)
        layout.addLayout(controls)

        self._status = QLabel("No ROIs yet — add a ROI layer, then draw on the image.")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        self._table = QTableWidget(0, len(mm.COLUMNS))
        self._table.setHorizontalHeaderLabels([mm.COLUMN_LABELS[c] for c in mm.COLUMNS])
        # Off for the same reason as the metadata tree: napari's item padding and
        # the stripe pitch disagree, leaving apparent blank rows.
        self._table.setAlternatingRowColors(False)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.verticalHeader().setVisible(False)
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.itemChanged.connect(self._on_item_changed)
        layout.addWidget(self._table, stretch=1)

        buttons = QHBoxLayout()
        self._all_layers = QCheckBox("All ROI layers")
        self._all_layers.setChecked(True)
        self._all_layers.setToolTip("Measure every ROI layer, not only the selected one")
        self._all_layers.stateChanged.connect(lambda _state: self.recompute())
        buttons.addWidget(self._all_layers)
        recompute = QPushButton("Recompute")
        recompute.clicked.connect(self.recompute)
        buttons.addWidget(recompute)
        export = QPushButton("Export to Excel…")
        export.clicked.connect(self.export)
        buttons.addWidget(export)
        buttons.addStretch(1)
        layout.addLayout(buttons)

        self._connect_viewer()
        self.refresh_layer_list()
        self.recompute()

    # -- wiring ---------------------------------------------------------------

    def _connect_viewer(self) -> None:
        try:
            self._viewer.layers.events.inserted.connect(self._on_layers_changed)
            self._viewer.layers.events.removed.connect(self._on_layers_changed)
            self._viewer.layers.selection.events.active.connect(self._on_layers_changed)
        except Exception:  # pragma: no cover - napari event API drift
            logger.warning("could not connect measurement auto-refresh", exc_info=True)

    def _on_layers_changed(self, event=None) -> None:
        self.refresh_layer_list()
        self._watch_shapes_layers()
        self.recompute()

    def _watch_shapes_layers(self) -> None:
        """Subscribe to shape edits on every ROI layer exactly once."""
        for layer in mm.shapes_layers(self._viewer):
            if id(layer) in self._connected:
                continue
            self._connected.add(id(layer))
            for event_name in ("data", "set_data", "highlight"):
                emitter = getattr(layer.events, event_name, None)
                if emitter is not None:
                    try:
                        emitter.connect(self._on_shapes_edited)
                        break
                    except Exception:  # pragma: no cover
                        logger.debug("could not watch %s.%s", layer.name, event_name, exc_info=True)

    def _on_shapes_edited(self, event=None) -> None:
        if self._updating:
            return
        self.recompute()

    # -- layer management -----------------------------------------------------

    def refresh_layer_list(self) -> None:
        """Repopulate the ROI-layer combo, preserving the current choice."""
        previous = self._layer_box.currentText()
        names = [layer.name for layer in mm.shapes_layers(self._viewer)]
        self._updating = True
        try:
            self._layer_box.clear()
            self._layer_box.addItems(names)
            if previous in names:
                self._layer_box.setCurrentIndex(names.index(previous))
            elif names:
                self._layer_box.setCurrentIndex(len(names) - 1)
        finally:
            self._updating = False

    def _selected_shapes_layer(self):
        name = self._layer_box.currentText()
        if name and name in self._viewer.layers:
            candidate = self._viewer.layers[name]
            if type(candidate).__name__ == "Shapes":
                return candidate
        layers = mm.shapes_layers(self._viewer)
        return layers[-1] if layers else None

    def new_roi_layer(self) -> None:
        """Add a Shapes layer matched to the active image and start drawing."""
        from napari.layers import Image

        active = self._viewer.layers.selection.active
        image = active if isinstance(active, Image) else None
        try:
            layer = mm.new_roi_layer(self._viewer, image)
        except Exception as exc:
            logger.exception("could not create ROI layer")
            QMessageBox.warning(self, "Microscopy Viewer", f"Could not create a ROI layer:\n{exc}")
            return
        self.refresh_layer_list()
        index = self._layer_box.findText(layer.name)
        if index >= 0:
            self._layer_box.setCurrentIndex(index)
        self._on_tool_chosen()
        if image is None:
            self._status.setText(
                "ROI layer added, but no image was selected — measurements will be in pixels."
            )

    def _on_layer_chosen(self, _index: int = 0) -> None:
        if self._updating:
            return
        layer = self._selected_shapes_layer()
        if layer is not None:
            self._viewer.layers.selection.active = layer
        self.recompute()

    def _on_tool_chosen(self, _index: int = 0) -> None:
        layer = self._selected_shapes_layer()
        if layer is None:
            return
        mode = TOOLS[max(self._tool_box.currentIndex(), 0)][1]
        try:
            self._viewer.layers.selection.active = layer
            layer.mode = mode
        except Exception as exc:  # pragma: no cover - unsupported mode name
            logger.warning("could not set shapes mode %s: %s", mode, exc)

    # -- results --------------------------------------------------------------

    def recompute(self) -> None:
        """Re-measure the relevant ROI layers and refill the table."""
        self._watch_shapes_layers()
        try:
            if self._all_layers.isChecked():
                self._rows = mm.measure_all(self._viewer)
            else:
                layer = self._selected_shapes_layer()
                if layer is None:
                    self._rows = []
                else:
                    mm.ensure_name_feature(layer)
                    self._rows = mm.measure_layer(layer, self._viewer)
        except Exception as exc:
            logger.exception("measurement failed")
            self._rows = []
            self._status.setText(f"Measurement failed: {exc}")
            self._fill_table()
            return

        self._fill_table()
        if not self._rows:
            self._status.setText("No ROIs yet — add a ROI layer, then draw on the image.")
            return
        uncalibrated = sum(1 for row in self._rows if not row.calibrated)
        rois = len({(row.roi_layer, row.roi_name) for row in self._rows})
        message = f"{rois} ROI(s), {len(self._rows)} measurement(s)."
        if uncalibrated:
            message += " Some are in pixels — the source file had no voxel size."
        self._status.setText(message)

    def _fill_table(self) -> None:
        self._updating = True
        try:
            self._table.setRowCount(len(self._rows))
            for row_index, measurement in enumerate(self._rows):
                row = measurement.as_row()
                for column_index, key in enumerate(mm.COLUMNS):
                    value = row.get(key, "")
                    text = format_number(value) if key == "value" else str(value)
                    item = QTableWidgetItem(text)
                    if key == "roi_name":
                        item.setToolTip("Double-click to rename this ROI")
                        item.setData(Qt.UserRole, row_index)
                    else:
                        item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                    if key == "value":
                        item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                    self._table.setItem(row_index, column_index, item)
            self._table.resizeColumnsToContents()
        finally:
            self._updating = False

    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        """Push an edited ROI name back into the Shapes layer's features."""
        if self._updating or item.column() != mm.COLUMNS.index("roi_name"):
            return
        row_index = item.data(Qt.UserRole)
        if row_index is None or not (0 <= int(row_index) < len(self._rows)):
            return
        measurement = self._rows[int(row_index)]
        layer_name = measurement.roi_layer
        if layer_name not in self._viewer.layers:
            return
        layer = self._viewer.layers[layer_name]
        names = mm.roi_names(layer)
        try:
            shape_index = names.index(measurement.roi_name)
        except ValueError:
            return
        self._updating = True
        try:
            mm.rename_roi(layer, shape_index, item.text())
        finally:
            self._updating = False
        self.recompute()

    # -- export ---------------------------------------------------------------

    def export(self) -> None:
        """Ask for a path and write the current table to a workbook."""
        if not self._rows:
            QMessageBox.information(
                self, "Microscopy Viewer", "There are no measurements to export yet."
            )
            return
        suggested = str(Path.home() / "Documents" / f"{default_stem('measurements')}.xlsx")
        path, _selected = QFileDialog.getSaveFileName(
            self, "Export measurements", suggested, WORKBOOK_FILTER
        )
        if not path:
            return
        metadata = []
        for layer in self._viewer.layers:
            meta = layer.metadata.get("mv_metadata")
            if meta is not None:
                metadata.append(meta)
        try:
            written = export_measurements(self._rows, path, metadata)
        except Exception as exc:
            logger.exception("measurement export failed")
            QMessageBox.critical(self, "Microscopy Viewer", f"Export failed:\n{exc}")
            return
        self._status.setText(f"Exported {len(self._rows)} measurement(s) to {written}")
