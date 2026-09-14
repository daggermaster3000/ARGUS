"""Dock panel: compare ROI intensities and pixel distributions across conditions.

Self-contained — it is pulled in only by its entry in
:mod:`microscopy_viewer.widgets.registry`, and nothing else in the app imports
it, so matplotlib stays off the startup path.

The heavy work lives in :mod:`microscopy_viewer.intensity`; this module snapshots
the viewer state on the main thread, hands it to a napari ``thread_worker``, and
renders what comes back.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .. import intensity as ix
from ..loaders.layer_spec import units_like
from .. import measurements as mm
from ..exports import default_stem
from ..utils import format_number, get_logger

logger = get_logger("intensity_widget")

ROI_LAYER_NAME = "Comparison ROIs"

ROLE_SIGNAL = "Signal"
ROLE_BACKGROUND = "Background"

#: Shape types that enclose an area. Lines and paths cannot be filled, so they
#: are ignored with a warning rather than silently contributing nothing.
AREA_SHAPES = ("rectangle", "polygon", "ellipse")


def _figure_canvas():
    """The Qt matplotlib canvas, whichever backend name this version exposes."""
    try:
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    except ImportError:  # matplotlib < 3.5
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
    return FigureCanvasQTAgg


class IntensityComparisonWidget(QWidget):
    """Assign layers to conditions, measure shared ROIs, compare distributions."""

    def __init__(self, viewer, parent: QWidget | None = None):
        super().__init__(parent)
        self._viewer = viewer
        self._result: ix.ComparisonResult | None = None
        self._worker = None
        self._updating = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self._tabs = QTabWidget()
        self._tabs.addTab(self._build_setup_tab(), "Setup")
        self._tabs.addTab(self._build_stats_tab(), "Statistics")
        self._tabs.addTab(self._build_histogram_tab(), "Histogram")
        layout.addWidget(self._tabs, stretch=1)

        self._status = QLabel("Assign layers to conditions, draw ROIs, then Measure.")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        self._connect_viewer()
        self.refresh_layers()

    # -- construction ---------------------------------------------------------

    def _build_setup_tab(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(2, 2, 2, 2)

        conditions = QGroupBox("Conditions")
        box = QVBoxLayout(conditions)
        self._condition_table = QTableWidget(0, 2)
        self._condition_table.setHorizontalHeaderLabels(["Condition", "Image layer"])
        self._condition_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self._condition_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self._condition_table.verticalHeader().setVisible(False)
        self._condition_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._condition_table.setAlternatingRowColors(False)
        self._condition_table.setMinimumHeight(120)
        box.addWidget(self._condition_table)

        buttons = QHBoxLayout()
        for label, tip, handler in (
            ("Add", "Add an empty condition row", self.add_condition),
            ("Remove", "Remove the selected condition", self.remove_condition),
            ("One per layer", "Create one condition for every open image layer", self.fill_from_layers),
        ):
            button = QPushButton(label)
            button.setToolTip(tip)
            button.clicked.connect(handler)
            buttons.addWidget(button)
        buttons.addStretch(1)
        box.addLayout(buttons)
        outer.addWidget(conditions)

        rois = QGroupBox("ROIs")
        roi_form = QVBoxLayout(rois)
        note = QLabel(
            "A ROI set to “All conditions” is applied to every one of them, staying "
            "spatially matched even when pixel sizes differ. When the samples sit in "
            "different places, give each condition its own ROI and share a "
            "“Compare as” label."
        )
        note.setWordWrap(True)
        roi_form.addWidget(note)

        roi_buttons = QHBoxLayout()
        create = QPushButton("Create / select ROI layer")
        create.clicked.connect(self.create_roi_layer)
        roi_buttons.addWidget(create)
        for label, mode in (("Rectangle", "add_rectangle"), ("Polygon", "add_polygon")):
            button = QPushButton(label)
            button.setToolTip(f"Draw a {label.lower()} ROI")
            button.clicked.connect(lambda _checked=False, m=mode: self._set_shape_mode(m))
            roi_buttons.addWidget(button)
        roi_buttons.addStretch(1)
        roi_form.addLayout(roi_buttons)

        self._roi_table = QTableWidget(0, 4)
        self._roi_table.setHorizontalHeaderLabels(["ROI", "Applies to", "Role", "Compare as"])
        self._roi_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._roi_table.verticalHeader().setVisible(False)
        self._roi_table.setAlternatingRowColors(False)
        self._roi_table.setMinimumHeight(110)
        self._roi_table.setToolTip(
            "Assign each ROI to one condition (or all of them), mark backgrounds, and "
            "give ROIs on different conditions the same 'Compare as' label to compare them"
        )
        roi_form.addWidget(self._roi_table)

        per_condition = QPushButton("One ROI set per condition")
        per_condition.setToolTip(
            "Assign the drawn ROIs to the conditions in order — for samples that sit "
            "in different places, so each is measured and normalised on its own"
        )
        per_condition.clicked.connect(self.assign_one_per_condition)
        roi_form.addWidget(per_condition)

        form = QFormLayout()
        self._normalization_box = QComboBox()
        self._normalization_box.addItems(ix.NORMALIZATIONS)
        self._normalization_box.setToolTip(
            "Rescale each condition's pixels against its own background before "
            "plotting and comparing"
        )
        form.addRow("Normalise:", self._normalization_box)

        self._mode_box = QComboBox()
        self._mode_box.addItems(ix.PLANE_MODES)
        self._mode_box.setCurrentText("Maximum projection")
        self._mode_box.setToolTip("Which plane of a 3D/4D stack to measure")
        form.addRow("Measure on:", self._mode_box)

        self._project_button = QPushButton("Create projection layers")
        self._project_button.setToolTip(
            "Flatten each condition's stack into a 2D layer using the mode above, "
            "so ROIs can be drawn on what you actually measure"
        )
        self._project_button.clicked.connect(self.create_projection_layers)
        form.addRow("", self._project_button)

        self._offset_box = QDoubleSpinBox()
        self._offset_box.setRange(-1e6, 1e6)
        self._offset_box.setDecimals(2)
        self._offset_box.setToolTip("Camera offset subtracted from every pixel before statistics")
        form.addRow("Camera offset:", self._offset_box)

        self._saturation_box = QDoubleSpinBox()
        self._saturation_box.setRange(0, 1e12)
        self._saturation_box.setDecimals(0)
        self._saturation_box.setSpecialValueText("auto (data type max)")
        self._saturation_box.setValue(0)
        self._saturation_box.setToolTip(
            "Pixels at or above this value count as clipped. 0 uses the data type maximum."
        )
        form.addRow("Saturation level:", self._saturation_box)
        roi_form.addLayout(form)
        outer.addWidget(rois)

        self._measure_button = QPushButton("Measure")
        self._measure_button.setToolTip("Measure every ROI on every condition")
        self._measure_button.clicked.connect(self.measure)
        outer.addWidget(self._measure_button)
        outer.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(page)
        return scroll

    def _build_stats_tab(self) -> QWidget:
        page = QWidget()
        box = QVBoxLayout(page)
        box.setContentsMargins(2, 2, 2, 2)

        self._stats_table = QTableWidget(0, len(ix.STAT_COLUMNS))
        self._stats_table.setHorizontalHeaderLabels([ix.STAT_LABELS[c] for c in ix.STAT_COLUMNS])
        self._stats_table.setAlternatingRowColors(False)
        self._stats_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._stats_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._stats_table.verticalHeader().setVisible(False)
        box.addWidget(self._stats_table, stretch=1)

        self._warnings = QLabel("")
        self._warnings.setWordWrap(True)
        self._warnings.setStyleSheet("color: #ffb454;")
        box.addWidget(self._warnings)

        export = QPushButton("Export statistics to CSV…")
        export.clicked.connect(self.export_csv)
        box.addWidget(export)
        return page

    def _build_histogram_tab(self) -> QWidget:
        from matplotlib.figure import Figure

        page = QWidget()
        box = QVBoxLayout(page)
        box.setContentsMargins(2, 2, 2, 2)

        # Every measured ROI-on-condition pair is its own curve. Listing them
        # individually is what makes per-condition ROIs comparable: two ROIs drawn
        # on different samples are simply two entries to tick.
        box.addWidget(QLabel("Curves to overlay:"))
        self._series_list = QListWidget()
        self._series_list.setMaximumHeight(110)
        self._series_list.setToolTip("Tick the ROI/condition curves to overlay")
        self._series_list.itemChanged.connect(self._on_series_toggled)
        box.addWidget(self._series_list)

        select = QHBoxLayout()
        for label, handler in (("All", lambda: self._check_all(True)), ("None", lambda: self._check_all(False))):
            button = QPushButton(label)
            button.setMaximumWidth(60)
            button.clicked.connect(handler)
            select.addWidget(button)
        select.addStretch(1)
        box.addLayout(select)

        controls = QFormLayout()
        self._bins_box = QSpinBox()
        self._bins_box.setRange(8, 1024)
        self._bins_box.setValue(128)
        self._bins_box.valueChanged.connect(self._draw_histogram)
        controls.addRow("Bins:", self._bins_box)

        self._log_y = QCheckBox("Logarithmic y axis")
        self._log_y.stateChanged.connect(self._draw_histogram)
        controls.addRow("", self._log_y)

        pair = QVBoxLayout()
        pair.setContentsMargins(0, 0, 0, 0)
        self._pair_a = QComboBox()
        self._pair_b = QComboBox()
        for combo in (self._pair_a, self._pair_b):
            combo.currentIndexChanged.connect(self._update_separation)
            pair.addWidget(combo)
        controls.addRow("Compare:", self._wrap(pair))
        box.addLayout(controls)

        self._separation = QLabel("")
        self._separation.setWordWrap(True)
        self._separation.setTextInteractionFlags(Qt.TextSelectableByMouse)
        box.addWidget(self._separation)

        canvas_class = _figure_canvas()
        # Small fonts and a generous minimum height: in a narrow dock the axis
        # labels are the first thing to be clipped by constrained layout.
        self._figure = Figure(figsize=(4.0, 3.2), layout="constrained")
        self._canvas = canvas_class(self._figure)
        self._canvas.setMinimumHeight(280)
        box.addWidget(self._canvas, stretch=1)

        save = QPushButton("Save histogram as PNG…")
        save.clicked.connect(self.save_figure)
        box.addWidget(save)
        return page

    @staticmethod
    def _wrap(layout) -> QWidget:
        holder = QWidget()
        holder.setLayout(layout)
        return holder

    # -- viewer wiring --------------------------------------------------------

    def _connect_viewer(self) -> None:
        try:
            self._viewer.layers.events.inserted.connect(self._on_layers_changed)
            self._viewer.layers.events.removed.connect(self._on_layers_changed)
        except Exception:  # pragma: no cover - napari event API drift
            logger.warning("could not connect layer tracking", exc_info=True)

    def _on_layers_changed(self, event=None) -> None:
        self.refresh_layers()

    def _image_layer_names(self) -> list[str]:
        from napari.layers import Image

        return [layer.name for layer in self._viewer.layers if isinstance(layer, Image)]

    def refresh_layers(self) -> None:
        """Keep the layer combos in step with the viewer, preserving choices."""
        names = self._image_layer_names()
        self._updating = True
        try:
            for row in range(self._condition_table.rowCount()):
                combo = self._condition_table.cellWidget(row, 1)
                if combo is None:
                    continue
                previous = combo.currentText()
                combo.clear()
                combo.addItems(names)
                if previous in names:
                    combo.setCurrentIndex(names.index(previous))
        finally:
            self._updating = False
        self.refresh_rois()

    def _roi_assignments(self) -> dict[str, tuple[str, str, str]]:
        """Current ``roi name -> (applies to, role, label)`` from the table."""
        out: dict[str, tuple[str, str, str]] = {}
        for row in range(self._roi_table.rowCount()):
            item = self._roi_table.item(row, 0)
            applies = self._roi_table.cellWidget(row, 1)
            role = self._roi_table.cellWidget(row, 2)
            label = self._roi_table.item(row, 3)
            if item is None or applies is None or role is None:
                continue
            out[item.text()] = (
                applies.currentText(),
                role.currentText(),
                label.text() if label is not None else "",
            )
        return out

    def refresh_rois(self) -> None:
        """Rebuild the ROI assignment table, preserving what the user has set."""
        layer = self._roi_layer()
        if layer is not None:
            # Fills in names for newly drawn shapes and de-duplicates them, so a
            # ROI can be identified unambiguously.
            mm.ensure_name_feature(layer)
        names = mm.roi_names(layer) if layer is not None else []
        conditions = [name for name, _layer in self.condition_rows()]
        previous = self._roi_assignments()

        self._updating = True
        try:
            self._roi_table.setRowCount(len(names))
            for row, name in enumerate(names):
                applies_to, role, label = previous.get(name, (ix.ALL_CONDITIONS, ROLE_SIGNAL, ""))

                item = QTableWidgetItem(name)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self._roi_table.setItem(row, 0, item)

                applies_box = QComboBox()
                applies_box.addItem(ix.ALL_CONDITIONS)
                applies_box.addItems(conditions)
                applies_box.setCurrentIndex(
                    conditions.index(applies_to) + 1 if applies_to in conditions else 0
                )
                applies_box.currentIndexChanged.connect(self._on_assignment_changed)
                self._roi_table.setCellWidget(row, 1, applies_box)

                role_box = QComboBox()
                role_box.addItems([ROLE_SIGNAL, ROLE_BACKGROUND])
                role_box.setCurrentIndex(1 if role == ROLE_BACKGROUND else 0)
                role_box.currentIndexChanged.connect(self._on_assignment_changed)
                self._roi_table.setCellWidget(row, 2, role_box)

                label_item = QTableWidgetItem(label or name)
                label_item.setToolTip(
                    "ROIs sharing this label are compared with each other, even when "
                    "drawn on different conditions"
                )
                self._roi_table.setItem(row, 3, label_item)
        finally:
            self._updating = False

    def _on_assignment_changed(self, *_args) -> None:
        if self._updating:
            return
        self._status.setText("ROI assignment changed — press Measure to update.")

    def assign_one_per_condition(self) -> None:
        """Hand out the drawn ROIs to the conditions in order.

        Signal ROIs are dealt round-robin across the conditions and given a shared
        ``Compare as`` label so they still line up in the table and the histogram;
        background ROIs are dealt the same way so each condition normalises
        against its own.
        """
        conditions = [name for name, _layer in self.condition_rows()]
        if not conditions:
            self._status.setText("Add conditions first.")
            return
        if self._roi_table.rowCount() == 0:
            self._status.setText("Draw the ROIs first, then assign them.")
            return

        signal_index = 0
        background_index = 0
        self._updating = True
        try:
            for row in range(self._roi_table.rowCount()):
                applies_box = self._roi_table.cellWidget(row, 1)
                role_box = self._roi_table.cellWidget(row, 2)
                if applies_box is None or role_box is None:
                    continue
                is_background = role_box.currentText() == ROLE_BACKGROUND
                if is_background:
                    condition = conditions[background_index % len(conditions)]
                    background_index += 1
                    label = "Background"
                else:
                    # Deal ROIs across the conditions; every full round is one
                    # comparable sample, so those ROIs share a label.
                    condition = conditions[signal_index % len(conditions)]
                    label = f"Sample {signal_index // len(conditions) + 1}"
                    signal_index += 1
                applies_box.setCurrentIndex(applies_box.findText(condition))
                self._roi_table.setItem(row, 3, QTableWidgetItem(label))
        finally:
            self._updating = False
        self._status.setText(
            f"{signal_index} signal and {background_index} background ROI(s) assigned "
            f"across {len(conditions)} condition(s)."
        )

    # -- conditions -----------------------------------------------------------

    def add_condition(self, _checked: bool = False, name: str = "", layer_name: str = "") -> None:
        """Append a condition row, defaulting its name to the chosen layer."""
        names = self._image_layer_names()
        row = self._condition_table.rowCount()
        self._condition_table.insertRow(row)

        combo = QComboBox()
        combo.addItems(names)
        if layer_name in names:
            combo.setCurrentIndex(names.index(layer_name))
        elif row < len(names):
            combo.setCurrentIndex(row)
        self._condition_table.setCellWidget(row, 1, combo)

        label = name or (combo.currentText().split(" :: ")[-1] if combo.count() else f"Condition {row + 1}")
        item = QTableWidgetItem(label)
        item.setToolTip("Double-click to rename this condition")
        self._condition_table.setItem(row, 0, item)
        # The ROI table's "Applies to" combos list the conditions, so they have
        # to be rebuilt whenever the condition set changes.
        self.refresh_rois()

    def remove_condition(self) -> None:
        rows = sorted({index.row() for index in self._condition_table.selectedIndexes()}, reverse=True)
        if not rows and self._condition_table.rowCount():
            rows = [self._condition_table.rowCount() - 1]
        for row in rows:
            self._condition_table.removeRow(row)
        self.refresh_rois()

    def fill_from_layers(self) -> None:
        """Replace the table with one condition per open image layer."""
        self._condition_table.setRowCount(0)
        for name in self._image_layer_names():
            self.add_condition(name=name.split(" :: ")[-1], layer_name=name)
        self._status.setText(f"{self._condition_table.rowCount()} condition(s) created from open layers.")

    def condition_rows(self) -> list[tuple[str, str]]:
        """``(condition name, layer name)`` for each populated row."""
        rows: list[tuple[str, str]] = []
        for row in range(self._condition_table.rowCount()):
            item = self._condition_table.item(row, 0)
            combo = self._condition_table.cellWidget(row, 1)
            if combo is None or not combo.currentText():
                continue
            name = (item.text().strip() if item else "") or combo.currentText()
            rows.append((name, combo.currentText()))
        return rows

    # -- ROI layer ------------------------------------------------------------

    def _roi_layer(self):
        if ROI_LAYER_NAME in self._viewer.layers:
            candidate = self._viewer.layers[ROI_LAYER_NAME]
            if type(candidate).__name__ == "Shapes":
                return candidate
        return None

    def create_roi_layer(self):
        """Add the shared ROI Shapes layer, matched to the first condition's image."""
        existing = self._roi_layer()
        if existing is not None:
            self._viewer.layers.selection.active = existing
            self.refresh_rois()
            return existing

        from napari.layers import Image

        rows = self.condition_rows()
        image = None
        if rows and rows[0][1] in self._viewer.layers:
            candidate = self._viewer.layers[rows[0][1]]
            image = candidate if isinstance(candidate, Image) else None
        if image is None:
            active = self._viewer.layers.selection.active
            image = active if isinstance(active, Image) else None

        layer = mm.new_roi_layer(self._viewer, image, name=ROI_LAYER_NAME)
        layer.events.data.connect(lambda _event=None: self.refresh_rois())
        self.refresh_rois()
        self._status.setText("ROI layer added — draw a rectangle or polygon on the image.")
        return layer

    def _set_shape_mode(self, mode: str) -> None:
        layer = self.create_roi_layer()
        if layer is None:
            return
        try:
            self._viewer.layers.selection.active = layer
            layer.mode = mode
        except Exception:  # pragma: no cover - unsupported mode name
            logger.warning("could not set shapes mode %s", mode, exc_info=True)

    # -- measurement ----------------------------------------------------------

    def _build_specs(self) -> tuple[list[ix.ConditionSpec], list[ix.RoiSpec], list[str]]:
        """Snapshot the viewer into plain data the worker thread can use safely."""
        from napari.layers import Image

        problems: list[str] = []
        conditions: list[ix.ConditionSpec] = []
        for name, layer_name in self.condition_rows():
            if layer_name not in self._viewer.layers:
                problems.append(f"Layer “{layer_name}” is no longer open.")
                continue
            layer = self._viewer.layers[layer_name]
            if not isinstance(layer, Image):
                problems.append(f"“{layer_name}” is not an image layer.")
                continue
            # Always measure the full-resolution level, never a pyramid preview.
            data = layer.data[0] if layer.multiscale else layer.data
            ndim = int(getattr(data, "ndim", 0))
            offset = self._viewer.dims.ndim - ndim
            step = tuple(
                int(self._viewer.dims.current_step[axis + offset])
                if 0 <= axis + offset < len(self._viewer.dims.current_step)
                else 0
                for axis in range(ndim)
            )
            conditions.append(
                ix.ConditionSpec(
                    name=name,
                    layer_name=layer_name,
                    data=data,
                    axes=str(layer.metadata.get("mv_axes", "")),
                    scale=tuple(float(s) for s in layer.scale),
                    translate=tuple(float(t) for t in layer.translate),
                    current_step=step,
                    dtype=np.dtype(getattr(data, "dtype", np.float32)),
                )
            )

        rois: list[ix.RoiSpec] = []
        layer = self._roi_layer()
        if layer is not None:
            mm.ensure_name_feature(layer)
            names = mm.roi_names(layer)
            assignments = self._roi_assignments()
            known = {name for name, _layer in self.condition_rows()}
            for index, shape in enumerate(layer.data):
                shape_type = str(layer.shape_type[index])
                roi_name = names[index] if index < len(names) else f"ROI {index + 1}"
                if shape_type not in AREA_SHAPES:
                    problems.append(f"{roi_name} is a {shape_type} and encloses no area — skipped.")
                    continue
                vertices = ix.world_vertices(shape, layer.scale, layer.translate)
                if shape_type == "ellipse":
                    vertices = ix.ellipse_to_polygon(vertices)

                applies_to, role, label = assignments.get(
                    roi_name, (ix.ALL_CONDITIONS, ROLE_SIGNAL, "")
                )
                condition = None if applies_to == ix.ALL_CONDITIONS else applies_to
                if condition is not None and condition not in known:
                    problems.append(
                        f"{roi_name} is assigned to “{condition}”, which is no longer a condition — "
                        "measured on every condition instead."
                    )
                    condition = None
                rois.append(
                    ix.RoiSpec(
                        name=roi_name,
                        vertices_world=vertices,
                        is_background=(role == ROLE_BACKGROUND),
                        condition=condition,
                        label=label or roi_name,
                    )
                )
        return conditions, rois, problems

    def create_projection_layers(self) -> None:
        """Flatten each condition's stack into a real 2D layer.

        A 3D/MIP *view* cannot be drawn on — napari's shape tools only work in
        2D — so comparing intensities on a projection needs an actual projected
        layer. This produces one per condition and repoints the condition table
        at them, so ROIs are drawn on exactly the pixels that get measured.
        """
        if self._condition_table.rowCount() == 0:
            self.fill_from_layers()

        conditions, _rois, problems = self._build_specs()
        if not conditions:
            self._status.setText("Add at least one condition first.")
            return

        mode = self._mode_box.currentText()
        targets = [condition for condition in conditions if ix.has_z_axis(condition)]
        skipped = [c.layer_name for c in conditions if c not in targets]
        if not targets:
            self._status.setText(
                "Every selected layer is already 2D — draw ROIs on them directly."
            )
            return

        self._project_button.setEnabled(False)
        self._status.setText(f"Projecting {len(targets)} layer(s)…")

        def _project():
            return [(condition, *ix.extract_plane(condition, mode)) for condition in targets]

        def _done(results):
            self._add_projection_layers(results, mode, skipped, problems)

        try:
            from napari.qt.threading import thread_worker
        except Exception:  # pragma: no cover - no Qt threading available
            _done(_project())
            return

        worker = thread_worker(_project)()
        worker.returned.connect(_done)
        worker.errored.connect(self._on_project_error)
        worker.finished.connect(lambda: self._project_button.setEnabled(True))
        worker.start()

    def _on_project_error(self, exc) -> None:
        logger.exception("projection failed", exc_info=exc)
        self._project_button.setEnabled(True)
        self._status.setText(f"Projection failed: {exc}")
        QMessageBox.critical(self, "Microscopy Viewer", f"Could not build the projection:\n{exc}")

    def _add_projection_layers(self, results, mode: str, skipped: list[str], problems: list[str]) -> None:
        """Add the projected planes as layers and point the conditions at them."""
        import copy

        created: dict[str, str] = {}
        for condition, plane, description in results:
            name = ix.projection_layer_name(condition.layer_name, mode)
            source = self._viewer.layers[condition.layer_name] if condition.layer_name in self._viewer.layers else None

            if name in self._viewer.layers:
                self._viewer.layers[name].data = plane
                created[condition.layer_name] = name
                continue

            # Copy the metadata so the projected layer can describe itself without
            # rewriting the source layer's record.
            meta = None
            source_meta = source.metadata.get("mv_metadata") if source is not None else None
            if source_meta is not None:
                meta = copy.copy(source_meta)
                meta.dimensionality = f"XY ({description})"

            kwargs = {
                "name": name,
                "scale": tuple(condition.scale[-2:]),
                "metadata": {
                    "mv_metadata": meta,
                    "mv_axes": "YX",
                    "mv_channel_index": source.metadata.get("mv_channel_index") if source else None,
                    "mv_channel_name": source.metadata.get("mv_channel_name", "") if source else "",
                    "mv_projection": mode,
                    "mv_source_layer": condition.layer_name,
                },
            }
            if source is not None:
                kwargs["colormap"] = source.colormap
                kwargs["blending"] = source.blending
                kwargs["contrast_limits"] = tuple(source.contrast_limits)
                kwargs["visible"] = source.visible
                # A layer added without units is dimensionless, and one of those
                # is enough for napari to stop using units at all — which puts
                # the scale bar back to pixels over a calibrated image.
                kwargs.update(units_like(source, 2))
            try:
                self._viewer.add_image(plane, **kwargs)
            except Exception as exc:
                logger.exception("could not add projection layer %s", name)
                problems.append(f"{name}: {exc}")
                continue
            created[condition.layer_name] = name

        if created:
            self._repoint_conditions(created)
            # The new layers are already flat, so measuring them is a plain read.
            self._mode_box.setCurrentText("Current slice")

        self.refresh_layers()
        message = f"Created {len(created)} projection layer(s) — draw ROIs on them."
        if skipped:
            message += f" {len(skipped)} layer(s) were already 2D and were left alone."
        self._status.setText(message)
        if problems:
            self._warnings.setText("\n".join(problems))

    def _repoint_conditions(self, created: dict[str, str]) -> None:
        """Switch each condition row from its source layer to the projected one."""
        names = self._image_layer_names()
        self._updating = True
        try:
            for row in range(self._condition_table.rowCount()):
                combo = self._condition_table.cellWidget(row, 1)
                if combo is None:
                    continue
                target = created.get(combo.currentText())
                if target is None:
                    continue
                combo.clear()
                combo.addItems(names)
                if target in names:
                    combo.setCurrentIndex(names.index(target))
        finally:
            self._updating = False

    def measure(self) -> None:
        """Run the measurement off the main thread and refresh the results."""
        if self._worker is not None:
            self._status.setText("A measurement is already running.")
            return

        conditions, rois, problems = self._build_specs()
        if not conditions:
            self._status.setText("Add at least one condition first.")
            return
        if not rois:
            self._status.setText("Draw at least one rectangle or polygon ROI first.")
            return

        saturation = self._saturation_box.value() or None
        arguments = {
            "conditions": conditions,
            "rois": rois,
            "mode": self._mode_box.currentText(),
            "offset": float(self._offset_box.value()),
            "saturation_level": saturation,
            "normalization": self._normalization_box.currentText(),
        }

        self._measure_button.setEnabled(False)
        self._status.setText(
            f"Measuring {len(rois)} ROI(s) across {len(conditions)} condition(s)…"
        )

        try:
            from napari.qt.threading import thread_worker
        except Exception:  # pragma: no cover - no Qt threading available
            self._finish(ix.run_comparison(**arguments), problems)
            return

        @thread_worker
        def _run():
            return ix.run_comparison(**arguments)

        worker = _run()
        worker.returned.connect(lambda result: self._finish(result, problems))
        worker.errored.connect(self._on_error)
        worker.finished.connect(self._clear_worker)
        self._worker = worker
        worker.start()

    def _clear_worker(self) -> None:
        self._worker = None
        self._measure_button.setEnabled(True)

    def _on_error(self, exc) -> None:
        logger.exception("intensity comparison failed", exc_info=exc)
        self._measure_button.setEnabled(True)
        self._worker = None
        self._status.setText(f"Measurement failed: {exc}")
        QMessageBox.critical(self, "Microscopy Viewer", f"Intensity comparison failed:\n{exc}")

    def _finish(self, result: ix.ComparisonResult, problems: list[str]) -> None:
        """Apply a completed run to the table, the combos and the plot."""
        self._result = result
        self._measure_button.setEnabled(True)
        self._fill_stats_table(result)

        messages = list(problems) + list(result.warnings)
        self._warnings.setText("\n".join(messages))

        conditions = result.conditions()
        series = [ix.series_name(roi, condition) for roi, condition in result.series()]
        self._updating = True
        try:
            self._populate_series(result)
            self._refill(self._pair_a, series, index=0)
            self._refill(self._pair_b, series, index=min(1, max(len(series) - 1, 0)))
        finally:
            self._updating = False

        self._draw_histogram()
        summary = f"{len(result.stats)} measurement(s) over {len(conditions)} condition(s)."
        if result.normalization != "None":
            summary += f" Normalised: {result.normalization.lower()}."
        if result.plane_description:
            summary += f" Plane: {result.plane_description}."
        if messages:
            summary += f" {len(messages)} warning(s) — see the Statistics tab."
        self._status.setText(summary)

    @staticmethod
    def _refill(combo: QComboBox, values: list[str], index: int | None = None) -> None:
        previous = combo.currentText()
        combo.clear()
        combo.addItems(values)
        if previous in values:
            combo.setCurrentIndex(values.index(previous))
        elif index is not None and 0 <= index < len(values):
            combo.setCurrentIndex(index)

    def _fill_stats_table(self, result: ix.ComparisonResult) -> None:
        self._stats_table.setRowCount(len(result.stats))
        for row, entry in enumerate(result.stats):
            values = entry.as_row()
            for column, key in enumerate(ix.STAT_COLUMNS):
                value = values.get(key)
                if value is None:
                    text = "—"
                elif isinstance(value, str):
                    text = value
                elif key in ("n_pixels", "saturated_pixels"):
                    text = f"{int(value):,}"
                else:
                    text = format_number(value)
                item = QTableWidgetItem(text)
                if not isinstance(value, str):
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                if key == "saturated_pixels" and value:
                    item.setForeground(Qt.red)
                self._stats_table.setItem(row, column, item)
        self._stats_table.resizeColumnsToContents()

    # -- plotting -------------------------------------------------------------

    def _populate_series(self, result: ix.ComparisonResult) -> None:
        """Rebuild the curve list, keeping whatever the user had ticked."""
        previously = {}
        for row in range(self._series_list.count()):
            item = self._series_list.item(row)
            previously[item.text()] = item.checkState() == Qt.Checked

        self._series_list.clear()
        for roi_name, condition in result.series():
            name = ix.series_name(roi_name, condition)
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            # Everything is shown by default: the point of the panel is to see
            # the ROIs side by side without hunting for them first.
            item.setCheckState(Qt.Checked if previously.get(name, True) else Qt.Unchecked)
            item.setData(Qt.UserRole, (roi_name, condition))
            self._series_list.addItem(item)

    def _check_all(self, checked: bool) -> None:
        self._updating = True
        try:
            for row in range(self._series_list.count()):
                self._series_list.item(row).setCheckState(Qt.Checked if checked else Qt.Unchecked)
        finally:
            self._updating = False
        self._draw_histogram()

    def _on_series_toggled(self, _item=None) -> None:
        if self._updating:
            return
        self._draw_histogram()

    def _all_series(self) -> dict[str, np.ndarray]:
        """Every measured curve, keyed by display name."""
        if self._result is None:
            return {}
        return {
            ix.series_name(roi_name, condition): self._result.samples[(roi_name, condition)]
            for roi_name, condition in self._result.series()
            if (roi_name, condition) in self._result.samples
        }

    def _selected_samples(self) -> tuple[str, dict[str, np.ndarray]]:
        """The ticked curves, keyed by display name, plus a title for the plot."""
        available = self._all_series()
        if not available:
            return "", {}

        chosen: dict[str, np.ndarray] = {}
        for row in range(self._series_list.count()):
            item = self._series_list.item(row)
            if item.checkState() == Qt.Checked and item.text() in available:
                chosen[item.text()] = available[item.text()]
        if not chosen:
            return "", {}

        rois = {key.split(" — ")[0] for key in chosen}
        title = f"ROI: {next(iter(rois))}" if len(rois) == 1 else f"{len(chosen)} curves"
        return title, chosen

    def _draw_histogram(self, *_args) -> None:
        """Redraw the overlaid density curves for the selected ROI."""
        if self._updating:
            return
        self._figure.clear()
        # clear() drops the layout engine, and without it the x axis label is
        # laid out past the bottom of the canvas and clipped away.
        self._figure.set_layout_engine("constrained")
        axes = self._figure.add_subplot(111)
        title, samples = self._selected_samples()

        if not samples:
            message = "No measurements yet" if self._result is None else "No curves selected"
            axes.text(0.5, 0.5, message, ha="center", va="center", transform=axes.transAxes)
            axes.set_axis_off()
            self._canvas.draw_idle()
            self._update_separation()
            return

        bins = ix.shared_bins(samples.values(), self._bins_box.value())
        for name, values in samples.items():
            if bins is None or values.size == 0:
                continue
            # step curves rather than bars: several filled histograms overlaid
            # hide each other, and density normalisation makes unequal ROI sizes
            # comparable.
            axes.hist(
                values,
                bins=bins,
                density=True,
                histtype="step",
                linewidth=1.5,
                label=f"{name} (n={values.size:,})",
            )

        normalization = self._result.normalization if self._result is not None else "None"
        axes.set_xlabel(ix.normalized_axis_label(normalization), fontsize=8)
        axes.set_ylabel("Density", fontsize=8)
        axes.set_title(title, fontsize=9)
        axes.tick_params(labelsize=7)
        if self._log_y.isChecked():
            axes.set_yscale("log")
        axes.legend(fontsize=6 if len(samples) > 4 else 7, frameon=False)
        self._canvas.draw_idle()
        self._update_separation()

    def _update_separation(self, *_args) -> None:
        """Recompute the AUC / overlap for the two curves chosen for comparison."""
        if self._updating:
            return
        available = self._all_series()
        first, second = self._pair_a.currentText(), self._pair_b.currentText()
        if not available or first not in available or second not in available:
            self._separation.setText("")
            return
        if first == second:
            self._separation.setText("Pick two different curves to compare.")
            return
        result = ix.mann_whitney_auc(available[first], available[second])
        result.condition_a, result.condition_b = first, second
        self._separation.setText(f"{first}  vs  {second}:\n{result.summary()}")

    # -- export ---------------------------------------------------------------

    def export_csv(self) -> None:
        """Write the statistics table to CSV."""
        if self._result is None or not self._result.stats:
            QMessageBox.information(self, "Microscopy Viewer", "Run a measurement first.")
            return
        suggested = str(Path.home() / "Documents" / f"{default_stem('roi_intensity')}.csv")
        path, _selected = QFileDialog.getSaveFileName(
            self, "Export statistics", suggested, "CSV file (*.csv)"
        )
        if not path:
            return
        try:
            frame = ix.stats_dataframe(self._result.stats)
            frame.to_csv(path, index=False)
        except Exception as exc:
            logger.exception("CSV export failed")
            QMessageBox.critical(self, "Microscopy Viewer", f"Export failed:\n{exc}")
            return
        self._status.setText(f"Wrote {len(frame)} row(s) to {path}")

    def save_figure(self) -> None:
        """Save the current histogram as a PNG."""
        if self._result is None:
            QMessageBox.information(self, "Microscopy Viewer", "Run a measurement first.")
            return
        suggested = str(Path.home() / "Documents" / f"{default_stem('roi_histogram')}.png")
        path, _selected = QFileDialog.getSaveFileName(
            self, "Save histogram", suggested, "PNG image (*.png)"
        )
        if not path:
            return
        try:
            self._figure.savefig(path, dpi=200)
        except Exception as exc:
            logger.exception("figure export failed")
            QMessageBox.critical(self, "Microscopy Viewer", f"Saving the figure failed:\n{exc}")
            return
        self._status.setText(f"Histogram saved to {path}")
