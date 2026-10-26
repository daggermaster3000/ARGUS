"""Dock panel: count the cells in the junction region and score them per channel.

The work is in :mod:`microscopy_viewer.junction_analysis`. This module picks the
sample, runs the count off the main thread — it segments nuclei, which takes
seconds on a GPU and longer without — and shows what came back three ways: a
table in the panel, the cells drawn over the image with one layer per channel
they are positive for, and a workbook.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from qtpy.QtCore import QObject, Qt, Signal
from qtpy.QtWidgets import (
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
    QPlainTextEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .. import fusion as fu
from .. import junction as jn
from .. import junction_analysis as ja
from ..exports import WORKBOOK_FILTER, default_stem
from ..utils import format_number, get_logger
from .sample_files import source_of, tie_to_sample

logger = get_logger("junction_analysis_widget")

#: Layer names. The all-cells layer, and a suffix for one layer per channel.
CELLS_LAYER = "Junction cells"
#: Marks the layers this panel drew, whatever channel they are named after.
DRAWN_KEY = "argus_junction_cells"
POSITIVE_SUFFIX = "+ cells"

#: Columns of the panel's table; the workbook has these and the rest.
TABLE_COLUMNS = (
    "Sample", "Channel", "Cells in region", "Positive", "Positive (%)",
    "Positive per mm²", "Positive per mm of junction", "Threshold",
    "Background median", "Junction status",
)

#: How the positive cells are drawn: a red outline this many pixels thick.
POSITIVE_COLOR = "red"
POSITIVE_CONTOUR = 2


class _Relay(QObject):
    """Carries worker-thread progress text onto the GUI thread."""

    message = Signal(str)


class JunctionAnalysisWidget(QWidget):
    """Count the cells in the junction region; say which channels they are positive for."""

    def __init__(self, app, parent: QWidget | None = None):
        super().__init__(parent)
        self._app = app
        self._viewer = app.viewer
        self._worker = None
        self._cancelled = False
        self._results: list[ja.JunctionCounts] = []
        self._qc_dir: Path | None = None
        self._options = ja.CountOptions()
        self._relay = _Relay()
        self._relay.message.connect(self._log)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        note = QLabel(
            "Counts the nuclei in the junction region and scores every cell in each "
            "channel. Make the region first, in the Fusion region panel — or let this "
            "find the junction itself."
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        layout.addWidget(self._build_settings_box())
        layout.addLayout(self._build_buttons())

        self._table = QTableWidget(0, len(TABLE_COLUMNS), self)
        self._table.setHorizontalHeaderLabels(list(TABLE_COLUMNS))
        self._table.verticalHeader().setVisible(False)
        # napari's stylesheet renders alternating rows as blank stripes.
        self._table.setAlternatingRowColors(False)
        self._table.setSortingEnabled(True)
        layout.addWidget(self._table, stretch=2)

        self._log_view = QPlainTextEdit()
        self._log_view.setReadOnly(True)
        self._log_view.setMaximumHeight(110)
        self._log_view.setPlaceholderText("Progress appears here.")
        layout.addWidget(self._log_view, stretch=1)

        self._status = QLabel("Open a sample and Count, or select samples in Experiment setup and Run.")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        try:
            self._viewer.layers.events.inserted.connect(self._on_layers_changed)
            self._viewer.layers.events.removed.connect(self._on_layers_changed)
        except Exception:  # pragma: no cover - a viewer stub without events
            logger.debug("could not subscribe to layer events", exc_info=True)
        self.refresh_layers()

    # -- construction ---------------------------------------------------------

    def _build_settings_box(self) -> QGroupBox:
        box = QGroupBox("Cells and what counts as positive")
        form = QFormLayout(box)
        form.setLabelAlignment(Qt.AlignRight)

        self._image_combo = QComboBox()
        self._image_combo.setToolTip("Any channel of the sample to count in; all of its channels are measured.")
        form.addRow("Sample", self._image_combo)

        self._region_edit = QLineEdit(fu.MANUAL_KEY)
        self._region_edit.setToolTip(
            "The stored region to count in — the name the Fusion region panel saved it "
            "under. A “Fusion region” layer on screen is used in preference, saved or not.\n"
            f"“{fu.MANUAL_KEY}” (a region drawn or corrected by hand) falls back to the "
            f"automatic “{fu.DEFAULT_KEY}” for samples that have none, and the sheet says "
            "which one each sample was counted in."
        )
        form.addRow("Region", self._region_edit)

        self._method = QComboBox()
        self._method.addItems(list(ja.METHODS))
        self._method.setToolTip(
            "How each channel's threshold is set. It is worked out per sample and per "
            "channel, and written beside every count.\n"
            "• Above the cells: median + k robust SDs of the cells' own values. For a "
            "region where most cells are negative, as with cells that had to migrate to "
            "be there. Adapts to staining and to autofluorescence all cells share.\n"
            "• Above the background: median + k robust SDs of the tissue between the "
            "cells. Stricter about what is signal; blind to a haze every cell carries.\n"
            "• Otsu: the best split of the cells into two groups. Right when there are "
            "two groups; with one it splits it anyway.\n"
            "• Fixed value: one number for every sample — the only rule under which "
            "samples are compared on the same scale."
        )
        self._method.currentTextChanged.connect(self._update_sensitivity)
        form.addRow("Positive when", self._method)

        self._threshold_from = QComboBox()
        self._threshold_from.addItems(list(ja.THRESHOLD_FROM))
        self._threshold_from.setToolTip(
            "Which cells the median and SD (or the Otsu split) are taken from. The cells "
            "counted are those in the region either way.\n"
            "• The cells in the region: only the region is segmented — fast. The bar is "
            "set by the region's own cells, so a region where a large share is positive "
            "raises it and undercounts.\n"
            "• All cells of the sample: the whole section is segmented, both organoids, "
            "and the threshold comes from all of its cells. Holds when the region is full "
            "of positive cells; minutes per slide scan instead of seconds.\n"
            "Not used by “Above the background” and “Fixed value”, which take nothing "
            "from the cells."
        )
        form.addRow("Threshold from", self._threshold_from)

        self._k = QDoubleSpinBox()
        self._k.setRange(0.5, 20.0)
        self._k.setSingleStep(0.5)
        self._k.setValue(3.0)
        self._k.setSuffix(" SD")
        self._k.setToolTip(
            "How far above the centre a cell has to be. SDs are robust (from the MAD), so "
            "a few bright cells do not raise the bar on themselves. 3 is the usual "
            "starting point; raise it if dim cells are being called positive."
        )
        form.addRow("k", self._k)

        self._fixed = QDoubleSpinBox()
        self._fixed.setRange(0.0, 1e6)
        self._fixed.setDecimals(1)
        self._fixed.setValue(1000.0)
        self._fixed.setToolTip("The threshold for “Fixed value”, in the image's own intensity units, used for every scored channel.")
        form.addRow("Fixed threshold", self._fixed)

        self._statistic = QComboBox()
        self._statistic.addItems(list(ja.STATISTICS))
        self._statistic.setToolTip(
            "Which figure of each cell is compared with the threshold. The mean is the "
            "steady choice; P95 or max pick up a label that fills only part of the cell, "
            "at the price of being noisier."
        )
        form.addRow("Compare the cell's", self._statistic)

        self._expand = QDoubleSpinBox()
        self._expand.setRange(0.0, 50.0)
        self._expand.setSingleStep(0.5)
        self._expand.setValue(3.0)
        self._expand.setSuffix(" µm")
        self._expand.setSpecialValueText("nucleus only")
        self._expand.setToolTip(
            "Each nucleus is grown by this far and the cell is measured over that: a "
            "label in the cytoplasm is not under the nucleus. Neighbours split the gap "
            "between them. Sizes in the table stay the nucleus's own."
        )
        form.addRow("Measure around nuclei", self._expand)

        self._find = QCheckBox("Find the junction when no region is stored")
        self._find.setChecked(True)
        self._find.setToolTip("With the settings on the Fusion region panel. Off skips samples that have no region yet.")
        form.addRow("", self._find)

        self._save_cells = QCheckBox("Store the counted cells with the sample")
        self._save_cells.setToolTip(
            "As the label map “junction cells”, with each channel's threshold and its "
            "positive cells. A TIFF's go in its sidecar; an .ims open in the viewer is "
            "skipped, since it would have to be closed to be written into."
        )
        form.addRow("", self._save_cells)

        self._update_sensitivity(self._method.currentText())
        return box

    def _build_buttons(self) -> QHBoxLayout:
        row = QHBoxLayout()
        self._count_button = QPushButton("Count in this sample")
        self._count_button.setToolTip(
            "Segment the nuclei in the region with the Segmentation panel's model and "
            "diameter, score them, and draw them: all cells as outlines, and one layer "
            "per channel with the cells positive in it outlined in red."
        )
        self._count_button.clicked.connect(self.count_current)
        row.addWidget(self._count_button)

        self._run_button = QPushButton("Run on the selected samples")
        self._run_button.setToolTip(
            "Every sample selected in Experiment setup (all when none is). Nothing has to "
            "be open. The workbook is written to “junction_analysis” in the experiment "
            "folder when the run ends.\n"
            "Every run, this one and a single count, leaves a QC picture per sample in "
            "“junction_analysis/QC”, sorted into the folders ok, check and fail."
        )
        self._run_button.clicked.connect(self.run_batch)
        row.addWidget(self._run_button)

        self._stop_button = QPushButton("Stop")
        self._stop_button.setEnabled(False)
        self._stop_button.clicked.connect(self.stop)
        row.addWidget(self._stop_button)

        export = QPushButton("Export…")
        export.setToolTip("Counts, combinations of channels, every cell with all its figures, and the settings, in one workbook.")
        export.clicked.connect(self.export)
        row.addWidget(export)

        qc = QPushButton("QC…")
        qc.setToolTip(
            "Open “junction_analysis/QC”: a picture of every counted region — the stains, "
            "the region, a ring round each positive cell — in the folder of its status: "
            "ok, check (junction flagged, or no cells) or fail."
        )
        qc.clicked.connect(self.open_qc_folder)
        row.addWidget(qc)
        return row

    def qc_folder(self, fallback: Path | None = None) -> Path | None:
        """Where a run sorts its pictures: “junction_analysis/QC” in the experiment folder."""
        experiment = getattr(self._app, "experiment_widget", None)
        folder = (experiment.folder() if experiment is not None else None) or fallback
        return Path(folder) / "junction_analysis" / "QC" if folder else None

    def open_qc_folder(self) -> None:
        from qtpy.QtCore import QUrl
        from qtpy.QtGui import QDesktopServices

        folder = self._qc_dir or self.qc_folder()
        if folder is None or not Path(folder).is_dir():
            self._status.setText("No QC pictures yet — count first.")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

    def _update_sensitivity(self, method: str) -> None:
        self._k.setEnabled(method in (ja.METHOD_CELLS, ja.METHOD_BACKGROUND))
        self._fixed.setEnabled(method == ja.METHOD_FIXED)
        self._threshold_from.setEnabled(method in ja.CELL_RULES)

    # -- state ----------------------------------------------------------------

    def _on_layers_changed(self, event=None) -> None:
        self.refresh_layers()

    def refresh_layers(self) -> None:
        from napari.layers import Image

        previous = self._image_combo.currentText()
        names = [str(layer.name) for layer in self._viewer.layers
                 if isinstance(layer, Image) and source_of(layer)]
        self._image_combo.blockSignals(True)
        self._image_combo.clear()
        self._image_combo.addItems(names)
        index = self._image_combo.findText(previous)
        self._image_combo.setCurrentIndex(index if index >= 0 else 0)
        self._image_combo.blockSignals(False)

    def image_layer(self):
        name = self._image_combo.currentText()
        return self._viewer.layers[name] if name and name in self._viewer.layers else None

    def _log(self, text: str) -> None:
        self._log_view.appendPlainText(text)
        self._status.setText(text)

    def options(self, channels: list[str] | None = None) -> ja.CountOptions:
        """The panel's settings, with the Segmentation and Fusion panels' beside them."""
        settings = None
        segmentation = getattr(self._app, "segmentation_widget", None)
        if segmentation is not None:
            try:
                settings = segmentation.settings()
            except Exception:
                logger.debug("could not read the segmentation settings", exc_info=True)
        junction = None
        fusion = getattr(self._app, "fusion_widget", None)
        if fusion is not None:
            try:
                junction = fusion.junction_options(save=False)
            except Exception:
                logger.debug("could not read the fusion settings", exc_info=True)
        return ja.CountOptions(
            band_key=self._region_edit.text().strip() or fu.MANUAL_KEY,
            method=self._method.currentText(),
            k=float(self._k.value()),
            statistic=self._statistic.currentText(),
            threshold_from=self._threshold_from.currentText(),
            # One number for every channel: named per channel once they are known.
            fixed={name: float(self._fixed.value()) for name in (channels or [])},
            expand_um=float(self._expand.value()),
            find_junction=self._find.isChecked(),
            save_cells=self._save_cells.isChecked(),
            settings=settings,
            junction=junction,
        )

    def _channel_names(self, image) -> list[str]:
        """Names of every channel of the sample *image* belongs to."""
        from napari.layers import Image

        source = source_of(image)
        return [
            str(layer.metadata.get("mv_channel_name", "") or layer.name)
            for layer in self._viewer.layers
            if isinstance(layer, Image) and source_of(layer) == source
        ]

    def _busy(self, running: bool) -> None:
        self._count_button.setEnabled(not running)
        self._run_button.setEnabled(not running)
        self._stop_button.setEnabled(running)

    def stop(self) -> None:
        self._cancelled = True
        self._log("Stopping after the current sample…")

    # -- running --------------------------------------------------------------

    def count_current(self) -> None:
        """Count in the open sample, and draw the cells over it."""
        if self._worker is not None:
            self._status.setText("A count is already running.")
            return
        image = self.image_layer()
        if image is None:
            self._status.setText("Open a sample first.")
            return
        path = Path(source_of(image))
        band = None
        if fu_layer := self._region_layer(image):
            band = np.asarray(fu_layer.data) > 0
        options = self.options(self._channel_names(image))
        options.qc_dir = self.qc_folder(path.parent)
        self._start([path], options, band=band, draw_on=str(image.name))

    def _region_layer(self, image):
        """The Fusion region layer on screen, when it fits this image."""
        from .fusion_widget import REGION_LAYER_NAME

        if REGION_LAYER_NAME not in self._viewer.layers:
            return None
        layer = self._viewer.layers[REGION_LAYER_NAME]
        data = image.data[0] if getattr(image, "multiscale", False) else image.data
        if tuple(layer.data.shape[-2:]) != tuple(data.shape[-2:]) or not np.any(layer.data):
            return None
        return layer

    def run_batch(self) -> None:
        """Count in every sample selected in Experiment setup."""
        experiment = getattr(self._app, "experiment_widget", None)
        if experiment is None:
            self._status.setText("The Experiment setup panel is not available.")
            return
        if self._worker is not None:
            self._status.setText("A count is already running.")
            return
        entries = [entry for entry in experiment.selected_entries() if entry.readable]
        if not entries:
            self._status.setText("No samples. Choose a folder in Experiment setup and scan it first.")
            return
        names = list(dict.fromkeys(name for entry in entries for name in entry.channel_names))
        folder = experiment.folder() or entries[0].path.parent
        workbook = Path(folder) / "junction_analysis" / f"{default_stem('junction_counts')}.xlsx"
        options = self.options(names)
        options.qc_dir = self.qc_folder(folder)
        self._start([entry.path for entry in entries], options, workbook=workbook)

    def run_paths(self, paths) -> bool:
        """Count in the given samples, as a batch does. Returns whether it started."""
        paths = [Path(path) for path in paths]
        if not paths:
            return False
        if self._worker is not None:
            self._status.setText("A count is already running.")
            return False
        experiment = getattr(self._app, "experiment_widget", None)
        folder = (experiment.folder() if experiment is not None else None) or paths[0].parent
        options = self.options()
        options.qc_dir = self.qc_folder(folder)
        workbook = Path(folder) / "junction_analysis" / f"{default_stem('junction_counts_redrawn')}.xlsx"
        self._start(paths, options, workbook=workbook)
        return True

    def _start(self, paths, options, band=None, draw_on: str = "", workbook: Path | None = None) -> None:
        from napari.qt.threading import thread_worker

        self._options = options
        self._qc_dir = options.qc_dir
        self._cancelled = False
        self._log_view.clear()
        self._busy(True)
        self._log(f"Counting in {len(paths)} sample(s): {options.method.lower()}, "
                  f"comparing each cell's {options.statistic}.")
        relay = self._relay

        @thread_worker
        def _run():
            if band is not None:
                result = ja.analyse_sample(paths[0], options, band=band, progress=relay.message.emit)
                relay.message.emit(result.error or ja.summary_line(result))
                return [result]
            return ja.run(paths, options, progress=relay.message.emit,
                          should_cancel=lambda: self._cancelled)

        def _done(results) -> None:
            self._results = list(results)
            self._fill_table()
            failed = [r for r in self._results if not r.ok]
            fault = any(r.gpu_fault for r in self._results)
            total = sum(r.n_cells for r in self._results)
            message = f"{total} cell(s) in {len(self._results) - len(failed)} region(s)."
            if failed:
                message += f" {len(failed)} failed: " + "; ".join(
                    f"{r.name} ({r.error})" for r in failed[:2])
            if draw_on and self._results and self._results[0].ok:
                self._draw(self._results[0], draw_on)
            if workbook is not None and self._results:
                try:
                    workbook.parent.mkdir(parents=True, exist_ok=True)
                    written = ja.export(self._results, workbook, self._options)
                    message += f" Workbook: {written}"
                except Exception as exc:
                    logger.exception("could not write the junction counts")
                    message += f" Could not write the workbook: {exc}"
            if fault:
                missing = len(paths) - len(self._results)
                message += (f" STOPPED on a GPU fault, {missing} sample(s) not attempted: restart the "
                            "viewer and run again. Nothing else should be using the GPU meanwhile.")
            if options.qc_dir is not None and self._results:
                tally = ", ".join(f"{sum(r.qc == status for r in self._results)} {status}"
                                  for status in jn.STATUSES)
                message += f" QC: {tally} — pictures in {options.qc_dir}"
            self._log(message)

        def _finished() -> None:
            self._worker = None
            self._busy(False)

        worker = _run()
        worker.returned.connect(_done)
        worker.errored.connect(lambda exc: self._log(f"Failed: {exc}"))
        worker.finished.connect(_finished)
        self._worker = worker
        worker.start()

    # -- results --------------------------------------------------------------

    def _fill_table(self) -> None:
        frame = ja.counts_dataframe(self._results)
        table = self._table
        table.setSortingEnabled(False)
        table.setRowCount(len(frame))
        for row in range(len(frame)):
            for column, key in enumerate(TABLE_COLUMNS):
                value = frame.iloc[row][key]
                if isinstance(value, str):
                    text = value
                elif value is None or (isinstance(value, float) and not np.isfinite(value)):
                    text = "—"
                else:
                    text = format_number(value)
                item = QTableWidgetItem(str(text))
                item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                table.setItem(row, column, item)
        table.setSortingEnabled(True)
        table.resizeColumnsToContents()

    def _draw(self, result: ja.JunctionCounts, image_name: str) -> None:
        """All the cells as outlines, and a filled layer per channel of its positives."""
        from ..loaders.layer_spec import units_like

        if result.labels is None or image_name not in self._viewer.layers:
            return
        image = self._viewer.layers[image_name]
        voxel = result.voxel_yx
        offset = tuple(float(t) for t in image.translate)[-2:]
        translate = (offset[0] + result.origin[0] * voxel[0], offset[1] + result.origin[1] * voxel[1])
        common = {"scale": voxel, "translate": translate, **units_like(image, 2)}

        def put(name: str, data: np.ndarray, **kwargs):
            if name in self._viewer.layers:
                self._viewer.layers.remove(name)
            layer = self._viewer.add_labels(data, name=name, **common, **kwargs)
            tie_to_sample(layer, str(result.path))
            layer.metadata[DRAWN_KEY] = True
            return layer

        # The cells of the last count only: a channel scored then and not now
        # would otherwise leave its layer behind.
        self.remove_layers()

        try:
            cells = put(CELLS_LAYER, result.labels, opacity=0.9)
            cells.contour = 1  # outlines: the signal underneath stays visible
            scored = [count for count in result.channels if not count.reference]
            for count in scored:
                wanted = result.positive_labels(count.channel)
                mask = np.where(np.isin(result.regions, wanted), 1, 0).astype(np.uint8)
                layer = put(f"{count.channel}{POSITIVE_SUFFIX}", mask, opacity=1.0)
                # Red outlines, the same for every channel: the layer's name
                # says which, and a red ring reads as "this one" over any stain.
                layer.contour = POSITIVE_CONTOUR
                self._tint(layer, POSITIVE_COLOR)
        except Exception:
            logger.exception("could not draw the junction cells")

    def remove_layers(self) -> int:
        """Take the counted cells off screen: all of them, every channel's."""
        removed = 0
        for layer in list(self._viewer.layers):
            if layer.metadata.get(DRAWN_KEY):
                self._viewer.layers.remove(layer)
                removed += 1
        return removed

    @staticmethod
    def _tint(layer, color: str) -> None:
        """One colour for the whole layer, when this napari can do that."""
        try:
            from napari.utils.colormaps import DirectLabelColormap

            layer.colormap = DirectLabelColormap(color_dict={None: color, 0: "transparent"})
        except Exception:
            logger.debug("no single-colour labels in this napari; keeping the default", exc_info=True)

    def export(self, path: str | None = None):
        """Write the counts, the combinations, every cell and the settings out."""
        if not self._results:
            self._status.setText("Nothing to export — count first.")
            return None
        if not path:
            experiment = getattr(self._app, "experiment_widget", None)
            folder = (experiment.folder() if experiment is not None else None) or Path.home()
            suggested = str(Path(folder) / f"{default_stem('junction_counts')}.xlsx")
            path, _selected = QFileDialog.getSaveFileName(self, "Export the junction counts", suggested, WORKBOOK_FILTER)
            if not path:
                return None
        try:
            written = ja.export(self._results, path, self._options)
        except Exception as exc:
            logger.exception("junction export failed")
            QMessageBox.critical(self, "Microscopy Viewer", f"Could not write that file:\n{exc}")
            return None
        self._log(f"Workbook written to {written}.")
        return written
