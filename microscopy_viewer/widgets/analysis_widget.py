"""Dock panel: open an object table, colour the labels by it, plot it.

The Segmentation and Batch panels write one row per object and then stop. This is
what happens next: open the table, pick a column, and the segmentation is redrawn
as that measurement — nuclei shaded by area, by mean intensity, by solidity —
with the same colours under a scatter plot of any two columns.

Three things it is built around.

**A row is an object.** Clicking a row selects that label in the viewer and
centres nothing — the point is that the identity survives the round trip through
Excel, so a suspicious row can be looked at.

**The colour scale is clipped by default.** Object tables have a handful of huge
outliers, usually two nuclei merged into one, and scaling to the true maximum
leaves everything else the same dark blue. 1-99 % is the default and the numbers
either side of it are shown.

**The table is not copied into Qt.** 18 000 rows is normal and a ``QTableWidget``
of that size takes seconds to build; a model over the DataFrame reads the cell
that is on screen and nothing else.

matplotlib and pandas are imported lazily, the way the intensity panel does it,
so this module costs nothing at startup.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from qtpy.QtCore import QAbstractTableModel, QModelIndex, Qt
from qtpy.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from .. import analysis
from ..exports import default_stem
from ..utils import format_number, get_logger

logger = get_logger("analysis_widget")

TABLE_FILTER = "Measurement tables (*.csv *.tsv *.txt *.xlsx *.xls);;All files (*)"

#: Metadata key the applied colouring is recorded under, so a second Apply knows
#: what to put back and the panel can say what a layer is currently showing.
COLOURING_KEY = "mv_label_colouring"


def _figure_canvas():
    """The Qt matplotlib canvas, whichever backend name this version exposes."""
    try:
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    except ImportError:  # matplotlib < 3.5
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
    return FigureCanvasQTAgg


class FrameModel(QAbstractTableModel):
    """A read-only Qt model over a pandas DataFrame.

    Qt asks for the cells it is about to draw and nothing else, so a table of any
    size opens instantly; building a ``QTableWidgetItem`` per cell would mean
    288 000 objects for one well.
    """

    def __init__(self, frame=None, parent=None):
        super().__init__(parent)
        self._frame = frame

    def set_frame(self, frame) -> None:
        self.beginResetModel()
        self._frame = frame
        self.endResetModel()

    @property
    def frame(self):
        return self._frame

    def rowCount(self, parent=QModelIndex()) -> int:  # noqa: B008 - Qt's signature
        if parent.isValid() or self._frame is None:
            return 0
        return int(len(self._frame))

    def columnCount(self, parent=QModelIndex()) -> int:  # noqa: B008 - Qt's signature
        if parent.isValid() or self._frame is None:
            return 0
        return int(len(self._frame.columns))

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid() or self._frame is None or role != Qt.DisplayRole:
            return None
        value = self._frame.iat[index.row(), index.column()]
        if isinstance(value, (float, np.floating)):
            return format_number(value)
        return str(value)

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole or self._frame is None:
            return None
        if orientation == Qt.Horizontal:
            return str(self._frame.columns[section])
        return str(self._frame.index[section])

    def column_name(self, index: int) -> str:
        return "" if self._frame is None else str(self._frame.columns[index])


class MeasurementAnalysisWidget(QWidget):
    """Open an object table, colour the labels by a column, plot two of them."""

    def __init__(self, viewer, parent: QWidget | None = None):
        super().__init__(parent)
        self._viewer = viewer
        self._frame = None
        self._path: Path | None = None
        self._label_column: str | None = None
        self._updating = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(self._build_file_box())

        # The table and the plot share the height: a dock is never tall enough for
        # both at their natural size, and which one matters depends on the moment.
        splitter = QSplitter(Qt.Vertical)
        splitter.addWidget(self._build_table_box())
        splitter.addWidget(self._build_colour_box())
        splitter.addWidget(self._build_plot_box())
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(2, 4)
        layout.addWidget(splitter, stretch=1)

        self._status = QLabel("Open a table written by a segmentation run.")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        self._connect_viewer()
        self.refresh_layers()

    # -- construction ---------------------------------------------------------

    def _build_file_box(self) -> QGroupBox:
        box = QGroupBox("Table")
        form = QFormLayout(box)
        form.setLabelAlignment(Qt.AlignRight)

        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        self._path_edit = QLineEdit()
        self._path_edit.setPlaceholderText("…/G_07_0.csv")
        self._path_edit.setToolTip(
            "A per-object table: the CSV a batch run writes, the workbook the "
            "Segmentation panel exports, or anything else with one row per object."
        )
        self._path_edit.returnPressed.connect(self.load)
        row_layout.addWidget(self._path_edit, stretch=1)
        browse = QPushButton("…")
        browse.setMaximumWidth(30)
        browse.clicked.connect(self.browse)
        row_layout.addWidget(browse)
        reload_button = QPushButton("Reload")
        reload_button.setToolTip("Read the file again — after a run has rewritten it.")
        reload_button.clicked.connect(self.load)
        row_layout.addWidget(reload_button)
        form.addRow("File", row)

        self._table_summary = QLabel("Nothing loaded.")
        self._table_summary.setWordWrap(True)
        form.addRow("Contents", self._table_summary)
        return box

    def _build_table_box(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(2, 2, 2, 2)
        outer.addWidget(QLabel("One row per object. Click a row to select that label."))

        self._model = FrameModel(parent=self)
        self._table = QTableView()
        self._table.setModel(self._model)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setAlternatingRowColors(False)
        self._table.setSortingEnabled(False)
        self._table.verticalHeader().setVisible(False)
        self._table.setMinimumHeight(140)
        self._table.clicked.connect(self._on_row_clicked)
        outer.addWidget(self._table, stretch=1)
        return page

    def _build_colour_box(self) -> QGroupBox:
        box = QGroupBox("Colour the labels")
        form = QFormLayout(box)
        form.setLabelAlignment(Qt.AlignRight)

        self._layer_box = QComboBox()
        self._layer_box.setToolTip(
            "The Labels layer the table describes. Guessed from the file name — "
            "“G_07_0.csv” is the table for well G/07 — and changeable here."
        )
        form.addRow("Labels layer", self._layer_box)

        self._column_box = QComboBox()
        self._column_box.setToolTip("The measurement the colours come from.")
        self._column_box.currentTextChanged.connect(self._on_column_changed)
        form.addRow("Colour by", self._column_box)

        self._colormap_box = QComboBox()
        self._colormap_box.addItems(list(analysis.COLORMAPS))
        self._colormap_box.setToolTip(
            "Perceptually uniform maps first: a measurement painted in a map with "
            "false edges in it is a measurement misread."
        )
        form.addRow("Colormap", self._colormap_box)

        limits = QWidget()
        limits_layout = QHBoxLayout(limits)
        limits_layout.setContentsMargins(0, 0, 0, 0)
        self._low_percentile = QDoubleSpinBox()
        self._high_percentile = QDoubleSpinBox()
        for spin, value, tip in (
            (self._low_percentile, analysis.DEFAULT_LOW_PERCENTILE, "Bottom of the colour scale."),
            (self._high_percentile, analysis.DEFAULT_HIGH_PERCENTILE, "Top of the colour scale."),
        ):
            spin.setRange(0.0, 100.0)
            spin.setDecimals(1)
            spin.setSingleStep(0.5)
            spin.setSuffix(" %")
            spin.setValue(value)
            spin.setToolTip(
                tip
                + " Clipping the tails matters here: a couple of merged nuclei with ten "
                "times the area of the rest will otherwise flatten everything else to one "
                "colour."
            )
            spin.valueChanged.connect(self._update_range_label)
            limits_layout.addWidget(spin)
        form.addRow("Percentiles", limits)

        self._range_label = QLabel("—")
        self._range_label.setWordWrap(True)
        form.addRow("Scale", self._range_label)

        buttons = QWidget()
        button_layout = QHBoxLayout(buttons)
        button_layout.setContentsMargins(0, 0, 0, 0)
        apply_button = QPushButton("Apply to the layer")
        apply_button.clicked.connect(self.apply_colours)
        button_layout.addWidget(apply_button)
        reset_button = QPushButton("Reset")
        reset_button.setToolTip("Put the layer's ordinary random label colours back.")
        reset_button.clicked.connect(self.reset_colours)
        button_layout.addWidget(reset_button)
        button_layout.addStretch(1)
        form.addRow("", buttons)
        return box

    def _build_plot_box(self) -> QGroupBox:
        box = QGroupBox("Scatter")
        outer = QVBoxLayout(box)

        row = QHBoxLayout()
        self._x_box = QComboBox()
        self._y_box = QComboBox()
        for label, combo in (("x", self._x_box), ("y", self._y_box)):
            row.addWidget(QLabel(label))
            combo.setMinimumWidth(110)
            combo.currentTextChanged.connect(lambda _text: self.draw_plot())
            row.addWidget(combo, stretch=1)
        row.addStretch(1)
        outer.addLayout(row)

        from matplotlib.figure import Figure

        canvas_class = _figure_canvas()
        self._figure = Figure(figsize=(4.0, 3.2), layout="constrained")
        self._canvas = canvas_class(self._figure)
        self._canvas.setMinimumHeight(240)
        outer.addWidget(self._canvas, stretch=1)

        self._plot_note = QLabel("—")
        self._plot_note.setWordWrap(True)
        outer.addWidget(self._plot_note)

        save = QPushButton("Save the plot as PNG…")
        save.clicked.connect(self.save_figure)
        outer.addWidget(save)
        return box

    # -- viewer ---------------------------------------------------------------

    def _connect_viewer(self) -> None:
        try:
            self._viewer.layers.events.inserted.connect(self._on_layers_changed)
            self._viewer.layers.events.removed.connect(self._on_layers_changed)
        except Exception:  # pragma: no cover - a viewer stub without events
            logger.debug("could not subscribe to layer events", exc_info=True)

    def _on_layers_changed(self, event=None) -> None:
        if not self._updating:
            self.refresh_layers()

    def _label_layers(self) -> list:
        from napari.layers import Labels

        return [layer for layer in self._viewer.layers if isinstance(layer, Labels)]

    def refresh_layers(self) -> None:
        """Rebuild the Labels-layer list, keeping the choice already made."""
        self._updating = True
        try:
            previous = self._layer_box.currentData()
            self._layer_box.clear()
            names = [str(layer.name) for layer in self._label_layers()]
            for name in names:
                self._layer_box.addItem(name, name)
                self._layer_box.setItemData(self._layer_box.count() - 1, name, Qt.ToolTipRole)
            index = self._layer_box.findData(previous)
            if index < 0 and self._path is not None:
                guess = analysis.match_layer(self._path.name, names)
                index = self._layer_box.findData(guess) if guess else -1
            self._layer_box.setCurrentIndex(max(index, 0))
        finally:
            self._updating = False

    def _selected_layer(self):
        name = str(self._layer_box.currentData() or "")
        if not name:
            return None
        try:
            return self._viewer.layers[name]
        except (KeyError, ValueError):
            return None

    # -- loading --------------------------------------------------------------

    def browse(self) -> None:
        start = self._path_edit.text() or str(Path.home() / "Documents")
        path, _selected = QFileDialog.getOpenFileName(
            self, "Open a measurement table", start, TABLE_FILTER
        )
        if path:
            self._path_edit.setText(path)
            self.load()

    def load(self) -> None:
        """Read the table and fill everything that depends on it."""
        text = self._path_edit.text().strip().strip('"')
        if not text:
            self._status.setText("Choose a table first.")
            return
        try:
            frame = analysis.read_table(text)
        except Exception as exc:
            logger.exception("could not read %s", text)
            self._status.setText(f"Could not read that table: {exc}")
            QMessageBox.critical(self, "Microscopy Viewer", f"Could not read that table:\n{exc}")
            return

        self._frame = frame
        self._path = Path(text)
        self._label_column = analysis.label_column(frame)
        self._model.set_frame(frame)
        self._table.resizeColumnsToContents()

        columns = analysis.numeric_columns(frame, exclude=[self._label_column or ""])
        self._updating = True
        try:
            for combo in (self._column_box, self._x_box, self._y_box):
                previous = combo.currentText()
                combo.clear()
                combo.addItems(columns)
                index = combo.findText(previous)
                combo.setCurrentIndex(index if index >= 0 else 0)
            if len(columns) > 1 and self._y_box.currentIndex() == 0:
                self._y_box.setCurrentIndex(1)
        finally:
            self._updating = False

        self.refresh_layers()
        note = f"{len(frame)} row(s), {len(frame.columns)} column(s)"
        if self._label_column:
            note += f"; labels in “{self._label_column}”"
        else:
            note += "; no label column, so the layer cannot be coloured from this table"
        self._table_summary.setText(note)
        self._update_range_label()
        self.draw_plot()
        self._status.setText(f"Loaded {self._path.name}. Pick a column and apply it to the layer.")

    # -- colouring ------------------------------------------------------------

    def _current_values(self):
        """``(labels, values)`` for the chosen column, or ``(None, None)``."""
        column = self._column_box.currentText()
        if self._frame is None or not column or column not in self._frame.columns:
            return None, None
        if not self._label_column:
            return None, None
        return self._frame[self._label_column].to_numpy(), self._frame[column].to_numpy()

    def _update_range_label(self) -> None:
        labels, values = self._current_values()
        if values is None:
            self._range_label.setText("—")
            return
        low, high = analysis.value_range(
            values, self._low_percentile.value(), self._high_percentile.value()
        )
        self._range_label.setText(
            f"{format_number(low)} … {format_number(high)} — {analysis.describe_column(values)}"
        )

    def _on_column_changed(self, _text: str) -> None:
        if not self._updating:
            self._update_range_label()

    def apply_colours(self) -> None:
        """Paint the Labels layer with the chosen column."""
        layer = self._selected_layer()
        if layer is None:
            self._status.setText("Open the segmentation as a Labels layer first.")
            return
        labels, values = self._current_values()
        if values is None:
            self._status.setText(
                "This table has no label column, so there is nothing to match the objects by."
            )
            return

        column = self._column_box.currentText()
        try:
            mapping, (low, high) = analysis.label_colors(
                labels,
                values,
                colormap=self._colormap_box.currentText(),
                low_percentile=self._low_percentile.value(),
                high_percentile=self._high_percentile.value(),
            )
            from napari.utils.colormaps import DirectLabelColormap

            layer.colormap = DirectLabelColormap(color_dict=mapping)
        except Exception as exc:
            logger.exception("could not colour %s", layer.name)
            self._status.setText(f"Could not colour that layer: {exc}")
            QMessageBox.critical(self, "Microscopy Viewer", f"Could not colour that layer:\n{exc}")
            return

        try:
            layer.metadata[COLOURING_KEY] = {
                "column": column,
                "table": str(self._path) if self._path else "",
                "low": low,
                "high": high,
                "colormap": self._colormap_box.currentText(),
            }
        except Exception:  # pragma: no cover - a layer with an odd metadata dict
            logger.debug("could not record the colouring on the layer", exc_info=True)

        painted = sum(1 for key in mapping if isinstance(key, int) and key > 0)
        self._status.setText(
            f"“{layer.name}” coloured by {column}: {painted} object(s), "
            f"{format_number(low)} … {format_number(high)}. "
            "Objects with no row in the table are left transparent."
        )
        self.draw_plot()

    def reset_colours(self) -> None:
        """Put napari's ordinary label colours back."""
        layer = self._selected_layer()
        if layer is None:
            return
        try:
            from napari.utils.colormaps import label_colormap

            layer.colormap = label_colormap(49, seed=0.5)
        except Exception:
            logger.exception("could not reset the label colours")
            self._status.setText("Could not reset that layer's colours.")
            return
        try:
            layer.metadata.pop(COLOURING_KEY, None)
        except Exception:  # pragma: no cover
            logger.debug("could not clear the colouring record", exc_info=True)
        self._status.setText(f"“{layer.name}” back to its ordinary label colours.")

    # -- the plot -------------------------------------------------------------

    def draw_plot(self) -> None:
        """Scatter two columns, coloured the same way the labels are."""
        self._figure.clear()
        axes = self._figure.add_subplot(111)

        x_name, y_name = self._x_box.currentText(), self._y_box.currentText()
        if self._frame is None or not x_name or not y_name:
            axes.set_axis_off()
            axes.text(0.5, 0.5, "Open a table", ha="center", va="center", fontsize=9)
            self._canvas.draw_idle()
            self._plot_note.setText("—")
            return

        x = self._frame[x_name].to_numpy(dtype=float)
        y = self._frame[y_name].to_numpy(dtype=float)
        _labels, values = self._current_values()

        sample = analysis.scatter_sample(len(x))
        if sample is not None:
            x, y = x[sample], y[sample]
            if values is not None:
                values = np.asarray(values, dtype=float)[sample]

        colors = None
        if values is not None:
            low, high = analysis.value_range(
                values, self._low_percentile.value(), self._high_percentile.value()
            )
            colors = analysis.colormap_colors(
                analysis.normalise(values, low, high), self._colormap_box.currentText()
            )

        axes.scatter(
            x,
            y,
            s=6,
            c=colors if colors is not None else "#4c72b0",
            linewidths=0,
            alpha=0.6 if len(x) > 2000 else 0.9,
        )
        axes.set_xlabel(x_name, fontsize=9)
        axes.set_ylabel(y_name, fontsize=9)
        axes.tick_params(labelsize=8)
        self._canvas.draw_idle()

        note = f"{len(x)} point(s)"
        if sample is not None:
            note += f" — a random sample of {len(self._frame)}, drawn so the plot stays usable"
        if values is not None:
            note += f"; coloured by {self._column_box.currentText()}"
        self._plot_note.setText(note)

    def save_figure(self) -> None:
        if self._frame is None:
            self._status.setText("Nothing to save — open a table first.")
            return
        suggested = str(Path.home() / "Documents" / f"{default_stem('scatter')}.png")
        path, _selected = QFileDialog.getSaveFileName(
            self, "Save the plot", suggested, "PNG image (*.png)"
        )
        if not path:
            return
        try:
            self._figure.savefig(path, dpi=200)
        except Exception as exc:
            logger.exception("could not save the plot")
            QMessageBox.critical(self, "Microscopy Viewer", f"Could not save that file:\n{exc}")
            return
        self._status.setText(f"Plot written to {path}.")

    # -- table interaction ----------------------------------------------------

    def _on_row_clicked(self, index) -> None:
        """Select the clicked object's label in the viewer.

        The identity of an object is what survives the trip through a CSV, so a row
        that looks wrong in the table can be found in the image without counting.
        """
        if self._frame is None or not self._label_column or not index.isValid():
            return
        layer = self._selected_layer()
        if layer is None:
            return
        try:
            label = int(self._frame[self._label_column].iat[index.row()])
            layer.selected_label = label
            layer.show_selected_label = True
        except Exception:
            logger.debug("could not select a label from the table", exc_info=True)
            return
        self._status.setText(
            f"Label {label} selected in “{layer.name}” — untick “show selected” on the "
            "layer to see the rest again."
        )
