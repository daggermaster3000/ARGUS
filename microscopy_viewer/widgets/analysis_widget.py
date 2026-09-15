"""Dock panel: open an object table, colour the labels by it, plot it.

The Segmentation and Batch panels write one row per object and then stop. This is
what happens next: open the table, pick a column, and the segmentation is redrawn
as that measurement — nuclei shaded by area, by mean intensity, by solidity —
with the same colours under a scatter plot of any two columns.

Three things it is built around.

**A row is an object.** Clicking a row selects that label in the viewer *and
sends the camera to it*, because the identity surviving the round trip through
Excel is only useful if the object can then be found. Sort by any column and the
first row is the largest, the roundest or the brightest object in the well, one
click from being on screen.

**The colour scale is clipped by default.** Object tables have a handful of huge
outliers, usually two nuclei merged into one, and scaling to the true maximum
leaves everything else the same dark blue. 1-99 % is the default and the numbers
either side of it are shown.

**The table is not copied into Qt.** 18 000 rows is normal and a ``QTableWidget``
of that size takes seconds to build; a model over the DataFrame reads the cell
that is on screen and nothing else.

**The tables find you.** A batch run writes its object tables into a folder
beside the plate, so when a plate is scanned in the File explorer this panel
lists those folders and the tables in them. The file box still takes any path;
the lists are there so that finding the run you just did is two clicks rather
than a walk through a file dialog.

**And out again.** *Export as AnnData* writes the table as ``.h5ad`` for squidpy
and the rest of the single-cell stack, and *Spatial dashboard* opens one in a
browser with the neighbourhood statistics already wired up.

matplotlib and pandas are imported lazily, the way the intensity panel does it,
so this module costs nothing at startup.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from qtpy.QtCore import QAbstractTableModel, QModelIndex, Qt
from qtpy.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
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

#: How a region is picked out of the plot. Off by default, because a selector
#: swallows the drag that otherwise pans the axes.
SELECTION_MODES = ("off", "rectangle", "lasso")

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


def _as_tuple(values) -> tuple[float, ...]:
    """A layer's scale or translate as plain floats, empty when there is none.

    napari hands these back as numpy arrays, so the usual ``value or default``
    guard raises rather than defaulting — an array of more than one element has no
    truth value.
    """
    if values is None:
        return ()
    try:
        return tuple(float(value) for value in np.atleast_1d(np.asarray(values)))
    except (TypeError, ValueError):
        return ()


class FrameModel(QAbstractTableModel):
    """A read-only Qt model over a pandas DataFrame.

    Qt asks for the cells it is about to draw and nothing else, so a table of any
    size opens instantly; building a ``QTableWidgetItem`` per cell would mean
    288 000 objects for one well.
    """

    def __init__(self, frame=None, parent=None):
        super().__init__(parent)
        self._frame = frame
        # View row -> frame row. Sorting builds one of these rather than
        # reordering the DataFrame, because the plot, the region selection and the
        # label colouring all address objects by their position in the table; a
        # sort that moved the rows would silently repaint the wrong nuclei.
        self._order: np.ndarray | None = None

    def set_frame(self, frame) -> None:
        self.beginResetModel()
        self._frame = frame
        self._order = None
        self.endResetModel()

    def source_row(self, view_row: int) -> int:
        """The row of the DataFrame showing at *view_row*."""
        row = int(view_row)
        if self._order is None:
            return row
        return int(self._order[row]) if 0 <= row < self._order.size else row

    def view_row(self, source_row: int) -> int:
        """Where the DataFrame's *source_row* is currently showing."""
        row = int(source_row)
        if self._order is None:
            return row
        found = np.flatnonzero(self._order == row)
        return int(found[0]) if found.size else row

    def sort(self, column: int, order=Qt.AscendingOrder) -> None:
        """Order the view by *column*. Blanks sort last whichever way it is read.

        A column of measurements has NaNs in it — an object too small for
        scikit-image to fit a hull to — and a blank that floats to the top on a
        descending sort buries the answer the sort was asked for.
        """
        if self._frame is None or not (0 <= column < len(self._frame.columns)):
            return
        name = self._frame.columns[column]
        ascending = order == Qt.AscendingOrder
        try:
            ranked = self._frame[name].sort_values(
                ascending=ascending, kind="stable", na_position="last"
            )
            new_order = np.asarray(
                [self._frame.index.get_loc(key) for key in ranked.index], dtype=int
            )
        except Exception:
            logger.exception("could not sort on %r", name)
            return
        self.layoutAboutToBeChanged.emit()
        self._order = new_order
        self.layoutChanged.emit()

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
        value = self._frame.iat[self.source_row(index.row()), index.column()]
        if isinstance(value, (float, np.floating)):
            return format_number(value)
        return str(value)

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole or self._frame is None:
            return None
        if orientation == Qt.Horizontal:
            return str(self._frame.columns[section])
        return str(self._frame.index[self.source_row(section)])

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

        # Selection state. ``_plot_rows`` maps a point in the scatter back to its
        # row in the table, which is not the identity when the plot is showing a
        # subsample of a very large one.
        self._scatter = None
        self._plot_rows = np.empty(0, dtype=int)
        self._plot_x = np.empty(0, dtype=float)
        self._plot_y = np.empty(0, dtype=float)
        self._plot_colors = None
        self._selected_rows = np.empty(0, dtype=int)
        self._selector = None

        #: Analysis folders beside the plate the File explorer last scanned.
        self._folders: list[Path] = []
        self._listed: list[Path] = []
        #: The dashboard process, once one has been started from here.
        self._dashboard_process = None

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

        beside = QWidget()
        beside_layout = QHBoxLayout(beside)
        beside_layout.setContentsMargins(0, 0, 0, 0)
        self._folder_box = QComboBox()
        self._folder_box.setToolTip(
            "Folders of measurement tables sitting beside the plate scanned in the File "
            "explorer. A batch run writes one per label set: “…_nuclei_objects”."
        )
        self._folder_box.currentIndexChanged.connect(self._folder_chosen)
        beside_layout.addWidget(self._folder_box, stretch=2)
        self._beside_table_box = QComboBox()
        self._beside_table_box.setToolTip(
            "The tables in that folder — one per image, and the run's summary last.\n\n"
            "Choosing one opens it."
        )
        self._beside_table_box.currentIndexChanged.connect(self._listed_table_chosen)
        beside_layout.addWidget(self._beside_table_box, stretch=3)
        form.addRow("Beside the plate", beside)
        self._beside_row = beside

        self._table_summary = QLabel("Nothing loaded.")
        self._table_summary.setWordWrap(True)
        form.addRow("Contents", self._table_summary)
        return box

    def _build_table_box(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(2, 2, 2, 2)
        outer.addWidget(
            QLabel("One row per object. Click a header to sort, a row to go to that object.")
        )

        self._model = FrameModel(parent=self)
        self._table = QTableView()
        self._table.setModel(self._model)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setAlternatingRowColors(False)
        self._table.setSortingEnabled(True)
        self._table.horizontalHeader().setSortIndicatorShown(True)
        self._table.horizontalHeader().setToolTip(
            "Click to sort by this measurement; click again to reverse it. Blanks sort "
            "last either way, so a descending sort really does start at the largest."
        )
        self._table.verticalHeader().setVisible(False)
        self._table.setMinimumHeight(140)
        self._table.clicked.connect(self._on_row_clicked)
        outer.addWidget(self._table, stretch=1)

        row = QHBoxLayout()
        self._follow = QCheckBox("go to the object")
        self._follow.setChecked(True)
        self._follow.setToolTip(
            "Centre the viewer on the clicked object and zoom in on it.\n\n"
            "Untick to select the label without moving the camera — useful when you are "
            "already framed on something and only want to step through rows."
        )
        row.addWidget(self._follow)
        row.addStretch(1)
        self._export_button = QPushButton("Export as AnnData…")
        self._export_button.setToolTip(
            "Write this one table as .h5ad: measurements in X, the label and centroids in "
            "obs, and the centroid in obsm[\"spatial\"] — which is what squidpy builds its "
            "neighbourhood graph from."
        )
        self._export_button.clicked.connect(self.export_anndata)
        row.addWidget(self._export_button)

        self._export_folder_button = QPushButton("…the whole folder as one")
        self._export_folder_button.setToolTip(
            "Combine every table in the folder chosen under “Beside the plate” into a "
            "single .h5ad, with the well and cycle in obs[\"image\"].\n\n"
            "A plate is one experiment, and a folder of forty-four files is forty-four "
            "files to concatenate before anything can be asked about the plate as a whole."
        )
        self._export_folder_button.clicked.connect(self.export_folder_anndata)
        row.addWidget(self._export_folder_button)

        self._dashboard_button = QPushButton("Spatial dashboard…")
        self._dashboard_button.setToolTip(
            "Open the squidpy dashboard on an .h5ad — neighbourhood enrichment, Ripley's L, "
            "co-occurrence and Moran's I, in a browser tab.\n\n"
            "It runs as its own process, with a terminal of its own showing what it is "
            "doing: the spatial statistics are minutes of CPU that have no business "
            "blocking the window the images are in, and a page that is merely thinking "
            "looks exactly like one that has hung."
        )
        self._dashboard_button.clicked.connect(self.open_dashboard)
        row.addWidget(self._dashboard_button)
        outer.addLayout(row)
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

        tools = QHBoxLayout()
        tools.addWidget(QLabel("select"))
        self._select_box = QComboBox()
        self._select_box.addItems(list(SELECTION_MODES))
        self._select_box.setToolTip(
            "Drag on the plot to pick out a group of objects. “Rectangle” for a band of "
            "one measurement, “Lasso” to draw round a cluster.\n\n"
            "What is selected stays its own colour and everything else fades, in the plot "
            "and in the image at once — which is how you find out where a cluster in the "
            "numbers actually sits in the well."
        )
        self._select_box.currentTextChanged.connect(self._install_selector)
        tools.addWidget(self._select_box)
        clear = QPushButton("Clear")
        clear.setToolTip("Drop the selection and put every object back.")
        clear.clicked.connect(self.clear_selection)
        tools.addWidget(clear)
        tools.addStretch(1)
        self._selection_label = QLabel("nothing selected")
        tools.addWidget(self._selection_label)
        outer.addLayout(tools)

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

    # -- tables beside the plate ---------------------------------------------

    def plate_changed(self, survey) -> None:
        """List the analysis folders sitting beside the plate the explorer scanned.

        Nothing is opened: which of five label sets someone wants is not something
        to guess at, and reading an 18 000-row table they did not ask for would
        blank the one they are working on.
        """
        from .. import explorer

        try:
            self._folders = explorer.analysis_folders(survey.path)
        except Exception:
            logger.exception("could not list the folders beside %s", survey.path)
            self._folders = []

        self._updating = True
        try:
            self._folder_box.clear()
            for folder in self._folders:
                # The plate's own stem is on the front of every one of these and is
                # forty characters long; what tells them apart is what follows it.
                shown = folder.name
                if shown.lower().startswith(survey.path.stem.lower()):
                    shown = shown[len(survey.path.stem) :].lstrip("_-") or folder.name
                self._folder_box.addItem(shown, folder)
                self._folder_box.setItemData(
                    self._folder_box.count() - 1, str(folder), Qt.ToolTipRole
                )
        finally:
            self._updating = False

        if self._folders:
            self._folder_chosen()
            self._status.setText(
                f"{len(self._folders)} folder(s) of tables beside {survey.path.name}."
            )
        else:
            self._beside_table_box.clear()
            self._status.setText(f"No table folders beside {survey.path.name}.")

    def _folder_chosen(self) -> None:
        """Fill the table list for the chosen folder."""
        from .. import explorer

        folder = self._folder_box.currentData()
        self._updating = True
        try:
            self._beside_table_box.clear()
            self._listed = explorer.analysis_tables(folder) if folder else []
            for table in self._listed:
                self._beside_table_box.addItem(table.name, table)
            # Nothing is loaded until a table is picked, so start on no row rather
            # than silently pointing at the first one.
            self._beside_table_box.setCurrentIndex(-1)
        finally:
            self._updating = False

    def _listed_table_chosen(self) -> None:
        if self._updating:
            return
        table = self._beside_table_box.currentData()
        if table is None:
            return
        self._path_edit.setText(str(table))
        self.load()

    def open_table(self, path) -> None:
        """Open *path*, whoever asked for it.

        The File explorer hands over the table it matched to the image on screen;
        this is the same route the file box takes, so a table opened that way
        behaves exactly like one typed in — including finding its layer.
        """
        self._path_edit.setText(str(path))
        self.load()

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

        # A selection made before the colour column changed still holds: the rows
        # are the same rows, so re-apply the fade rather than dropping it.
        self._apply_selection_to_layer()
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
        self._scatter = None
        self._selector = None
        if self._frame is None or not x_name or not y_name:
            axes.set_axis_off()
            axes.text(0.5, 0.5, "Open a table", ha="center", va="center", fontsize=9)
            self._canvas.draw_idle()
            self._plot_note.setText("—")
            return

        x = self._frame[x_name].to_numpy(dtype=float)
        y = self._frame[y_name].to_numpy(dtype=float)
        _labels, values = self._current_values()

        rows = np.arange(len(x))
        sample = analysis.scatter_sample(len(x))
        if sample is not None:
            rows = np.asarray(sample)
            x, y = x[rows], y[rows]
            if values is not None:
                values = np.asarray(values, dtype=float)[rows]

        colors = None
        if values is not None:
            low, high = analysis.value_range(
                values, self._low_percentile.value(), self._high_percentile.value()
            )
            colors = analysis.colormap_colors(
                analysis.normalise(values, low, high), self._colormap_box.currentText()
            )

        if colors is None:
            # A single colour still has to be per-point: the selection fades the
            # points it did not choose by rewriting their alpha.
            colors = np.tile(np.array([0.298, 0.447, 0.690, 1.0]), (len(x), 1))
        base_alpha = 0.6 if len(x) > 2000 else 0.9
        colors = np.array(colors, dtype=float, copy=True)
        colors[:, 3] *= base_alpha

        self._plot_rows = rows
        self._plot_x, self._plot_y = x, y
        self._plot_colors = colors
        self._scatter = axes.scatter(x, y, s=6, c=colors, linewidths=0)
        axes.set_xlabel(x_name, fontsize=9)
        axes.set_ylabel(y_name, fontsize=9)
        axes.tick_params(labelsize=8)

        # Selecting on stale axes selects the wrong points, so the selector is
        # rebuilt with them.
        self._install_selector(self._select_box.currentText())
        self._apply_selection_to_plot()

        note = f"{len(x)} point(s)"
        if sample is not None:
            note += f" — a random sample of {len(self._frame)}, drawn so the plot stays usable"
        if values is not None:
            note += f"; coloured by {self._column_box.currentText()}"
        self._plot_note.setText(note)

    # -- selecting a region ---------------------------------------------------

    def _install_selector(self, mode: str) -> None:
        """Attach the chosen selector to the current axes, replacing any other."""
        previous, self._selector = self._selector, None
        if previous is not None:
            try:
                previous.set_active(False)
                previous.disconnect_events()
            except Exception:  # pragma: no cover - already torn down with the figure
                logger.debug("could not detach the previous selector", exc_info=True)

        if self._scatter is None or str(mode) == "off":
            return
        axes = self._scatter.axes
        try:
            from matplotlib.widgets import LassoSelector, RectangleSelector

            if str(mode) == "rectangle":
                # Held on the widget: matplotlib keeps only a weak reference, so a
                # selector that is not stored is garbage-collected and does nothing.
                self._selector = RectangleSelector(
                    axes,
                    self._on_rectangle,
                    useblit=True,
                    button=[1],
                    props={"facecolor": "none", "edgecolor": "#d62728", "linewidth": 1.2},
                )
            else:
                self._selector = LassoSelector(
                    axes, self._on_lasso, useblit=True, props={"color": "#d62728", "linewidth": 1.2}
                )
        except Exception:
            logger.exception("could not install the %s selector", mode)
            self._status.setText(f"Region selection ({mode}) is not available in this matplotlib.")

    def _on_rectangle(self, press, release) -> None:
        if press is None or release is None:
            return
        if None in (press.xdata, press.ydata, release.xdata, release.ydata):
            return  # a drag that started or ended outside the axes
        mask = analysis.points_in_rectangle(
            self._plot_x, self._plot_y, press.xdata, release.xdata, press.ydata, release.ydata
        )
        self._set_selection(mask)

    def _on_lasso(self, vertices) -> None:
        mask = analysis.points_in_polygon(self._plot_x, self._plot_y, vertices)
        self._set_selection(mask)

    def _set_selection(self, mask) -> None:
        """Record which rows are selected, then show it in the plot and the image."""
        mask = np.asarray(mask, dtype=bool)
        self._selected_rows = self._plot_rows[mask] if mask.any() else np.empty(0, dtype=int)
        self._apply_selection_to_plot()
        self._apply_selection_to_layer()

        count = int(self._selected_rows.size)
        if not count:
            self._selection_label.setText("nothing selected")
            self._status.setText("Nothing inside that region.")
            return
        self._selection_label.setText(f"{count} selected")
        column = self._column_box.currentText()
        message = f"{count} of {len(self._frame)} object(s) selected"
        if column and self._label_column:
            values = self._frame[column].to_numpy(dtype=float)[self._selected_rows]
            message += f"; {column} {analysis.describe_column(values)}"
        self._status.setText(message + ".")

    def clear_selection(self) -> None:
        """Drop the selection and put every object back to full strength."""
        self._selected_rows = np.empty(0, dtype=int)
        self._apply_selection_to_plot()
        self._apply_selection_to_layer()
        self._selection_label.setText("nothing selected")
        self._status.setText("Selection cleared.")

    def selected_labels(self) -> list[int]:
        """The label ids currently selected, in table order."""
        if self._frame is None or not self._label_column or self._selected_rows.size == 0:
            return []
        chosen = self._frame[self._label_column].to_numpy()[self._selected_rows]
        return [int(value) for value in chosen]

    def _apply_selection_to_plot(self) -> None:
        """Fade the points outside the selection, leaving the chosen ones as they were."""
        if self._scatter is None or self._plot_colors is None:
            return
        colors = np.array(self._plot_colors, dtype=float, copy=True)
        if self._selected_rows.size:
            keep = np.isin(self._plot_rows, self._selected_rows)
            colors[~keep, 3] *= analysis.DIM_ALPHA
        try:
            self._scatter.set_facecolors(colors)
            self._canvas.draw_idle()
        except Exception:  # pragma: no cover - figure torn down mid-update
            logger.debug("could not redraw the selection", exc_info=True)

    def _apply_selection_to_layer(self) -> None:
        """Fade the labels outside the selection, if the layer is coloured by a column.

        Only touches a layer this panel has painted: rewriting the colours of a
        layer someone else set up would be a surprise, and there would be nothing
        to put back afterwards.
        """
        layer = self._selected_layer()
        if layer is None:
            return
        try:
            record = dict(layer.metadata.get(COLOURING_KEY) or {})
        except Exception:  # pragma: no cover - an odd metadata dict
            record = {}
        if not record:
            return

        labels, values = self._current_values()
        if values is None:
            return
        try:
            mapping, _range = analysis.label_colors(
                labels,
                values,
                colormap=record.get("colormap", self._colormap_box.currentText()),
                low=record.get("low"),
                high=record.get("high"),
            )
            mapping = analysis.dim_unselected(mapping, self.selected_labels())
            from napari.utils.colormaps import DirectLabelColormap

            layer.colormap = DirectLabelColormap(color_dict=mapping)
        except Exception:
            logger.exception("could not show the selection on %s", layer.name)

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
        if self._frame is None or not index.isValid():
            return
        row = self._model.source_row(index.row())

        layer = self._selected_layer()
        label = None
        if layer is not None and self._label_column:
            try:
                label = int(self._frame[self._label_column].iat[row])
                layer.selected_label = label
                layer.show_selected_label = True
            except Exception:
                logger.debug("could not select a label from the table", exc_info=True)
                label = None

        moved = self.go_to_row(row) if self._follow.isChecked() else False
        if label is None:
            self._status.setText(
                "Centred on that object." if moved else "That row names no label to select."
            )
            return
        message = f"Label {label}"
        message += f" selected in “{layer.name}”" if layer is not None else ""
        message += " and centred." if moved else " — untick “show selected” to see the rest again."
        self._status.setText(message)

    def go_to_row(self, row: int) -> bool:
        """Put the object at *row* in the middle of the canvas, zoomed in on it.

        The centroids in the table are in micrometres, and a calibrated layer's
        world coordinates are micrometres too, so the centroid *is* a camera
        position — no conversion, and it stays right when the viewer is showing a
        coarser pyramid level.
        """
        if self._frame is None:
            return False
        centre = analysis.object_centroid(self._frame, row)
        if centre is None:
            return False

        layer = self._selected_layer()
        # Not ``getattr(...) or ()``: a layer's translate is a numpy array, and
        # asking an array of more than one element whether it is truthy raises.
        offset = _as_tuple(getattr(layer, "translate", None))
        if len(offset) >= len(centre):
            centre = tuple(c + o for c, o in zip(centre, offset[-len(centre) :]))

        try:
            # napari's camera centre is always (z, y, x); a 2D table gives (y, x).
            padded = (0.0,) * (3 - len(centre)) + tuple(centre)
            self._viewer.camera.center = padded[-3:]
            self._viewer.camera.zoom = analysis.zoom_for(
                analysis.object_diameter(self._frame, row), self._canvas_edge()
            )
        except Exception:
            logger.exception("could not move the camera to row %d", row)
            return False

        self._step_to_plane(layer, centre)
        return True

    def _canvas_edge(self) -> float:
        """The shorter edge of the canvas in pixels, for working out the zoom."""
        try:
            size = self._viewer.window._qt_viewer.canvas.size
            edge = min(float(size[0]), float(size[1]))
            if edge > 1:
                return edge
        except Exception:
            logger.debug("could not measure the canvas; using a default", exc_info=True)
        return 600.0

    def _step_to_plane(self, layer, centre) -> None:
        """Move the Z slider to the plane the object is in, if there is one.

        Centring the camera on a volume does not change which slice is displayed,
        so without this the camera is over an object that is not on screen.
        """
        if layer is None or len(centre) < 3 or self._viewer.dims.ndisplay != 2:
            return
        try:
            scale = _as_tuple(getattr(layer, "scale", None))
            if len(scale) < 3 or scale[-3] <= 0:
                return
            axis = self._viewer.dims.ndim - 3
            if axis < 0:
                return
            plane = int(round(float(centre[0]) / scale[-3]))
            limit = int(self._viewer.dims.nsteps[axis]) - 1
            self._viewer.dims.set_current_step(axis, max(0, min(plane, limit)))
        except Exception:
            logger.debug("could not step to the object's plane", exc_info=True)

    # -- out to the single-cell stack -----------------------------------------

    # -- the dashboard --------------------------------------------------------

    def _dashboard_candidate(self) -> Path | None:
        """The ``.h5ad`` the dashboard should open, guessed from what is loaded.

        The combined file for the plate if there is one, else the file beside the
        table that is open. Only a default — the dialog is still shown, because
        guessing which of several analyses someone means is not something to do
        silently.
        """
        folder = self._folder_box.currentData()
        if folder is not None:
            combined = Path(folder).parent / f"{Path(folder).name}.h5ad"
            if combined.exists():
                return combined
        if self._path is not None:
            beside = analysis.anndata_path(self._path)
            if beside.exists():
                return beside
            return beside.parent / beside.name
        return None

    def open_dashboard(self) -> None:
        """Start the squidpy dashboard on an ``.h5ad``, in a browser."""
        from .. import dashboard

        missing = dashboard.missing_packages()
        if missing:
            self._status.setText(dashboard.install_hint(missing).replace("\n\n", " "))
            QMessageBox.information(self, "Spatial dashboard", dashboard.install_hint(missing))
            return

        candidate = self._dashboard_candidate()
        path, _selected = QFileDialog.getOpenFileName(
            self,
            "Open in the spatial dashboard",
            str(candidate or Path.home()),
            "AnnData (*.h5ad);;All files (*)",
        )
        if not path:
            return

        try:
            process, url = dashboard.launch(path)
        except Exception as exc:  # noqa: BLE001 - shown, not raised
            logger.exception("could not start the dashboard")
            self._status.setText(f"Could not start the dashboard: {exc}")
            QMessageBox.critical(self, "Spatial dashboard", str(exc))
            return

        # Held so the process is not garbage-collected mid-start, and so a second
        # press can see that one is already running.
        self._dashboard_process = process
        self._status.setText(
            f"Dashboard starting at {url} on {Path(path).name} — it opens in your browser in "
            "a few seconds. A terminal opens with it: that is where it says which step it is "
            "on and how long each took. Closing the viewer leaves it running; closing the "
            "terminal stops it."
        )

    def export_folder_anndata(self) -> None:
        """Combine every table in the chosen folder into one ``.h5ad``."""
        from .. import explorer

        folder = self._folder_box.currentData()
        if folder is None:
            self._status.setText(
                "No folder of tables chosen — scan a plate in the File explorer first, "
                "or use “Export as AnnData…” for the table that is open."
            )
            return
        tables = [
            table for table in explorer.analysis_tables(folder)
            if not table.stem.endswith("_summary")
        ]
        if not tables:
            self._status.setText(f"No object tables in {Path(folder).name}.")
            return

        default = Path(folder).parent / f"{Path(folder).name}.h5ad"
        path, _selected = QFileDialog.getSaveFileName(
            self,
            f"Combine {len(tables)} table(s) into one AnnData",
            str(default),
            "AnnData (*.h5ad);;All files (*)",
        )
        if not path:
            return

        # A plate of half a million objects takes a minute; say so before the
        # window stops repainting rather than afterwards.
        self._status.setText(f"Combining {len(tables)} table(s)…")
        self._export_folder_button.setEnabled(False)
        QApplication.processEvents()
        try:
            written = analysis.write_combined_anndata(tables, path)
        except ImportError:
            self._status.setText(
                "anndata is not installed — pip install \"microscopy-viewer[analysis]\", "
                "or pip install anndata."
            )
            return
        except Exception as exc:
            logger.exception("could not combine %s", folder)
            self._status.setText(f"Could not combine those tables: {exc}")
            QMessageBox.critical(self, "Microscopy Viewer", f"Could not combine:\n{exc}")
            return
        finally:
            self._export_folder_button.setEnabled(True)

        import anndata as ad

        combined = ad.read_h5ad(written, backed="r")
        self._status.setText(
            f"Wrote {written.name}: {combined.n_obs} object(s) from {len(tables)} image(s) "
            f"x {combined.n_vars} feature(s), the image in obs[\"image\"]."
        )

    def export_anndata(self) -> None:
        """Write the loaded table as ``.h5ad`` for squidpy and the rest of that stack."""
        if self._frame is None:
            self._status.setText("Open a table first.")
            return
        default = (
            analysis.anndata_path(self._path)
            if self._path is not None
            else Path(default_stem("objects")).with_suffix(".h5ad")
        )
        path, _selected = QFileDialog.getSaveFileName(
            self, "Export as AnnData", str(default), "AnnData (*.h5ad);;All files (*)"
        )
        if not path:
            return
        try:
            written = analysis.write_anndata(
                self._frame,
                path,
                label_column=self._label_column,
                source=self._path,
            )
        except ImportError:
            self._status.setText(
                "anndata is not installed — pip install \"microscopy-viewer[analysis]\", "
                "or pip install anndata."
            )
            return
        except Exception as exc:
            logger.exception("could not export %s", path)
            self._status.setText(f"Could not export that table: {exc}")
            QMessageBox.critical(self, "Microscopy Viewer", f"Could not export:\n{exc}")
            return

        features = len(analysis.feature_columns(self._frame, self._label_column))
        self._status.setText(
            f"Wrote {written.name}: {len(self._frame)} object(s) x {features} feature(s), "
            "centroids in obsm[\"spatial\"]."
        )
