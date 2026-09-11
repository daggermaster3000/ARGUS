"""Dock panel: segment a whole HCS plate in one run.

The Segmentation panel is where settings are chosen — one channel of one image,
looked at, adjusted, run again. This panel is where those settings are *used*: it
walks an OME-Zarr plate, runs the same engine over every well and every 4i cycle
you tick, and writes the masks back into the plate as NGFF ``labels`` groups.

Two things are worth knowing before running one.

**The Cellpose settings come from the Segmentation panel.** Model, diameter,
mode, thresholds, GPU — all of it is read from there when Run is pressed, and
shown here as a single line so there is no second copy to keep in step. Get one
image right there, then come here and do the other three hundred.

**Channels are matched, not indexed.** In a 4i plate the channel label changes
every cycle (``Ab1_DAPI``, ``Ab2_DAPI``, …) while ``wavelength_id`` does not, so
the picker matches on whichever key holds across the plate. Choosing "DAPI" in
cycle 1 finds the right channel in cycle 7.

The run happens on a worker thread and yields one image at a time, so the window
stays usable, the table fills as it goes, and Stop takes effect at the next image
boundary with everything already finished safely on disk.
"""

from __future__ import annotations

from pathlib import Path

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .. import batch
from .. import segmentation as sg
from ..utils import get_logger

logger = get_logger("batch_widget")

#: Entry for "no second channel", in the nuclei picker.
NO_CHANNEL = "— none —"
#: Entry for "measure on whatever was segmented", in the measure picker.
SAME_CHANNEL = "— the segmented channel —"

RESULT_COLUMNS = ("Image", "Objects", "Seconds", "Status", "Note")


class BatchSegmentationWidget(QWidget):
    """Pick a plate, pick wells and channels, segment the lot."""

    def __init__(self, app, parent: QWidget | None = None):
        super().__init__(parent)
        self._app = app
        self._survey: batch.PlateSurvey | None = None
        self._worker = None
        self._outcomes: list[batch.ImageOutcome] = []
        self._running_jobs: tuple[batch.ImageJob, ...] = ()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self._backend_notice = QLabel("")
        self._backend_notice.setWordWrap(True)
        self._backend_notice.setVisible(False)
        layout.addWidget(self._backend_notice)

        # The setup is four group boxes deep and does not fit a narrow dock, so it
        # scrolls; the run controls and the results table stay put underneath.
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        page = QWidget()
        inner = QVBoxLayout(page)
        inner.setContentsMargins(0, 0, 0, 0)
        inner.addWidget(self._build_plate_box())
        inner.addWidget(self._build_images_box())
        inner.addWidget(self._build_channels_box())
        inner.addWidget(self._build_output_box())
        inner.addWidget(self._build_cellpose_box())
        inner.addStretch(1)
        scroll.setWidget(page)
        layout.addWidget(scroll, stretch=1)

        layout.addLayout(self._build_run_row())
        self._progress = QProgressBar()
        self._progress.setVisible(False)
        layout.addWidget(self._progress)
        layout.addWidget(self._build_results_table(), stretch=1)

        self._status = QLabel("Choose a plate and press Scan.")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        self._check_backend()
        self.refresh_settings_summary()
        self._update_enabled()

    # -- construction ---------------------------------------------------------

    def _build_plate_box(self) -> QGroupBox:
        box = QGroupBox("Plate")
        form = QFormLayout(box)
        form.setLabelAlignment(Qt.AlignRight)

        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        self._plate_edit = QLineEdit()
        self._plate_edit.setPlaceholderText("…/AssayPlate.zarr")
        self._plate_edit.setToolTip(
            "The .zarr folder of an OME-Zarr plate — the one holding the row folders."
        )
        row_layout.addWidget(self._plate_edit, stretch=1)
        browse = QPushButton("…")
        browse.setMaximumWidth(30)
        browse.clicked.connect(self._browse_plate)
        row_layout.addWidget(browse)
        form.addRow("Store", row)

        buttons = QWidget()
        button_layout = QHBoxLayout(buttons)
        button_layout.setContentsMargins(0, 0, 0, 0)
        scan = QPushButton("Scan the plate")
        scan.setToolTip("Reads metadata only — no pixels — so this is quick even on a big plate.")
        scan.clicked.connect(self.scan)
        button_layout.addWidget(scan)
        loaded = QPushButton("Use the loaded plate")
        loaded.setToolTip("Take the path from the layer that is open in the viewer.")
        loaded.clicked.connect(self._use_loaded_plate)
        button_layout.addWidget(loaded)
        button_layout.addStretch(1)
        form.addRow("", buttons)

        self._plate_summary = QLabel("No plate scanned yet.")
        self._plate_summary.setWordWrap(True)
        form.addRow("Contents", self._plate_summary)
        return box

    def _build_images_box(self) -> QGroupBox:
        box = QGroupBox("Images to run")
        outer = QVBoxLayout(box)

        lists = QHBoxLayout()
        wells_column = QVBoxLayout()
        wells_column.addWidget(QLabel("Wells"))
        self._wells_list = QListWidget()
        self._wells_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self._wells_list.setToolTip("Ctrl-click and shift-click to choose. Everything by default.")
        self._wells_list.itemSelectionChanged.connect(self._update_selection_summary)
        wells_column.addWidget(self._wells_list)
        lists.addLayout(wells_column, stretch=2)

        cycles_column = QVBoxLayout()
        cycles_column.addWidget(QLabel("Acquisitions"))
        self._cycles_list = QListWidget()
        self._cycles_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self._cycles_list.setToolTip(
            "The 4i cycles of the plate. One cycle is usually enough: the same cells are "
            "imaged in every one, so segmenting them all gives you the same objects seven times."
        )
        self._cycles_list.itemSelectionChanged.connect(self._update_selection_summary)
        cycles_column.addWidget(self._cycles_list)
        lists.addLayout(cycles_column, stretch=1)
        outer.addLayout(lists)

        buttons = QHBoxLayout()
        for text, handler in (
            ("All wells", lambda: self._wells_list.selectAll()),
            ("No wells", lambda: self._wells_list.clearSelection()),
        ):
            button = QPushButton(text)
            button.clicked.connect(handler)
            buttons.addWidget(button)
        buttons.addStretch(1)
        outer.addLayout(buttons)

        self._selection_summary = QLabel("—")
        self._selection_summary.setWordWrap(True)
        outer.addWidget(self._selection_summary)
        return box

    def _build_channels_box(self) -> QGroupBox:
        box = QGroupBox("Channels")
        form = QFormLayout(box)
        form.setLabelAlignment(Qt.AlignRight)

        self._segment_box = QComboBox()
        self._segment_box.setToolTip(
            "The channel segmented, matched by its wavelength id where the plate records one — "
            "the key that stays put while the label changes from cycle to cycle."
        )
        form.addRow("Segment", self._segment_box)

        self._nuclei_box = QComboBox()
        self._nuclei_box.setToolTip(
            "Optional second stain handed to Cellpose alongside the segmented one. Segment a "
            "membrane or cytoplasmic channel and give it the nuclear stain here to get whole "
            "cells rather than blobs."
        )
        form.addRow("Nuclei", self._nuclei_box)

        self._measure_box = QComboBox()
        self._measure_box.setToolTip(
            "The channel the per-object intensities are read from. Segment on DAPI and measure "
            "on the reporter to get signal per nucleus."
        )
        form.addRow("Measure", self._measure_box)
        return box

    def _build_output_box(self) -> QGroupBox:
        box = QGroupBox("Output")
        form = QFormLayout(box)
        form.setLabelAlignment(Qt.AlignRight)

        self._label_name = QLineEdit(batch.DEFAULT_LABEL_NAME)
        self._label_name.setToolTip(
            "Name of the label set written into each image, at <image>/labels/<name>. Give a "
            "second run a different name to keep both."
        )
        self._label_name.editingFinished.connect(self._refresh_wells)
        form.addRow("Label set", self._label_name)

        self._level = QSpinBox()
        self._level.setRange(0, 8)
        self._level.setToolTip(
            "Which pyramid level to segment. 0 is full resolution. Each step up halves the "
            "image and quarters the time; the masks are written on the level they were "
            "computed at, with the pyramid below it."
        )
        self._level.valueChanged.connect(self._update_level_note)
        form.addRow("Pyramid level", self._level)

        self._level_note = QLabel("—")
        self._level_note.setWordWrap(True)
        form.addRow("", self._level_note)

        self._overwrite = QCheckBox("Replace a label set that is already there")
        self._overwrite.setToolTip(
            "Off, an image that already has this label set is skipped — which is what makes an "
            "interrupted run safe to start again."
        )
        self._overwrite.toggled.connect(self._update_selection_summary)
        form.addRow("Existing labels", self._overwrite)

        self._write_tables = QCheckBox("Write per-object tables and a plate summary")
        self._write_tables.setChecked(True)
        form.addRow("Tables", self._write_tables)

        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        self._table_dir = QLineEdit()
        self._table_dir.setPlaceholderText("beside the plate")
        row_layout.addWidget(self._table_dir, stretch=1)
        browse = QPushButton("…")
        browse.setMaximumWidth(30)
        browse.clicked.connect(self._browse_tables)
        row_layout.addWidget(browse)
        form.addRow("Tables in", row)
        return box

    def _build_cellpose_box(self) -> QGroupBox:
        box = QGroupBox("Cellpose")
        outer = QVBoxLayout(box)
        note = QLabel(
            "Taken from the Segmentation panel when the run starts — set the model and the "
            "diameter on one image there, then run the plate here."
        )
        note.setWordWrap(True)
        outer.addWidget(note)

        row = QHBoxLayout()
        self._settings_summary = QLabel("—")
        self._settings_summary.setWordWrap(True)
        row.addWidget(self._settings_summary, stretch=1)
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self.refresh_settings_summary)
        row.addWidget(refresh)
        outer.addLayout(row)
        return box

    def _build_run_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        self._run_button = QPushButton("Run the batch")
        self._run_button.clicked.connect(self.run)
        row.addWidget(self._run_button)
        self._stop_button = QPushButton("Stop")
        self._stop_button.setEnabled(False)
        self._stop_button.setToolTip(
            "Stops after the image being worked on. Everything already written stays written."
        )
        self._stop_button.clicked.connect(self.stop)
        row.addWidget(self._stop_button)
        row.addStretch(1)
        self._open_button = QPushButton("Open the selected result")
        self._open_button.setEnabled(False)
        self._open_button.setToolTip("Load that image and its labels into the viewer.")
        self._open_button.clicked.connect(self.open_selected)
        row.addWidget(self._open_button)
        return row

    def _build_results_table(self) -> QTableWidget:
        table = QTableWidget(0, len(RESULT_COLUMNS), self)
        table.setHorizontalHeaderLabels(list(RESULT_COLUMNS))
        table.verticalHeader().setVisible(False)
        table.setAlternatingRowColors(False)
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table.itemSelectionChanged.connect(
            lambda: self._open_button.setEnabled(bool(table.selectedItems()))
        )
        self._results_table = table
        return table

    # -- plate ----------------------------------------------------------------

    def _browse_plate(self) -> None:
        start = self._plate_edit.text() or str(getattr(self._app, "last_directory", Path.home()))
        path = QFileDialog.getExistingDirectory(self, "Choose an OME-Zarr plate", start)
        if path:
            self._plate_edit.setText(path)
            self.scan()

    def _browse_tables(self) -> None:
        start = self._table_dir.text() or self._plate_edit.text() or str(Path.home())
        path = QFileDialog.getExistingDirectory(self, "Where to write the tables", start)
        if path:
            self._table_dir.setText(path)

    def _use_loaded_plate(self) -> None:
        """Take the plate path off a layer that is already open.

        The readers record the file the layer came from, so the plate the user is
        looking at is usually the plate they want to run — and it saves finding a
        deeply nested store in a file dialog for the second time.
        """
        for layer in reversed(list(self._app.viewer.layers)):
            meta = (getattr(layer, "metadata", {}) or {}).get("mv_metadata")
            candidate = getattr(meta, "file_path", None)
            if not candidate:
                continue
            path = Path(candidate)
            for parent in [path, *path.parents]:
                if parent.suffix.lower() in (".zarr", ".ngff"):
                    self._plate_edit.setText(str(parent))
                    self.scan()
                    return
        self._status.setText("No open layer came from a Zarr store.")

    def scan(self) -> None:
        """Read the plate's metadata and fill the wells, cycles and channel lists."""
        text = self._plate_edit.text().strip().strip('"')
        if not text:
            self._status.setText("Choose the .zarr folder of a plate first.")
            return
        try:
            survey = batch.survey_plate(Path(text))
        except Exception as exc:
            logger.exception("plate scan failed")
            self._survey = None
            self._plate_summary.setText("—")
            self._status.setText(f"That is not a plate this can run: {exc}")
            self._update_enabled()
            return

        self._survey = survey
        n_wells = len(survey.wells)
        self._plate_summary.setText(
            f"{survey.name}: {len(survey.jobs)} image(s), {n_wells} well(s), "
            f"{len(survey.acquisitions)} acquisition(s)."
        )
        self._refresh_wells()
        self._refresh_cycles()
        self._refresh_channels()
        self._update_level_note()
        self._status.setText("Choose wells, cycles and channels, then run.")
        self._update_enabled()

    def _refresh_wells(self) -> None:
        """Rebuild the well list, marking the ones that already have this label set."""
        if self._survey is None:
            return
        previous = {item.data(Qt.UserRole) for item in self._wells_list.selectedItems()}
        name = self._label_name.text().strip() or batch.DEFAULT_LABEL_NAME
        done = {job.well for job in self._survey.jobs if batch.has_labels(job, name)}

        self._wells_list.clear()
        for well in self._survey.wells:
            text = f"{well} — has {name}" if well in done else well
            item = QListWidgetItem(text)
            item.setData(Qt.UserRole, well)
            self._wells_list.addItem(item)
            item.setSelected(well in previous if previous else True)
        self._update_selection_summary()

    def _refresh_cycles(self) -> None:
        if self._survey is None:
            return
        self._cycles_list.clear()
        acquisitions = self._survey.acquisitions
        for acquisition in acquisitions:
            text = "—" if acquisition is None else f"cycle {acquisition}"
            item = QListWidgetItem(text)
            item.setData(Qt.UserRole, acquisition)
            self._cycles_list.addItem(item)
            # One cycle by default: on a 4i plate every cycle images the same
            # cells, so selecting them all would segment the same nuclei
            # seven times over and take seven times as long.
            item.setSelected(acquisition == acquisitions[0])
        self._update_selection_summary()

    def _refresh_channels(self) -> None:
        if self._survey is None:
            return
        channels = self._survey.channels()
        for box, extra in (
            (self._segment_box, None),
            (self._nuclei_box, NO_CHANNEL),
            (self._measure_box, SAME_CHANNEL),
        ):
            previous = box.currentData()
            box.clear()
            if extra is not None:
                box.addItem(extra, None)
            for channel in channels:
                pick = self._pick_for(channel)
                box.addItem(channel.describe(), pick)
                box.setItemData(box.count() - 1, f"matched by {pick.key}", Qt.ToolTipRole)
            index = box.findData(previous)
            box.setCurrentIndex(index if index >= 0 else 0)

    @staticmethod
    def _pick_for(channel: batch.ChannelInfo) -> batch.ChannelPick:
        """Match on the key that survives the plate: wavelength, then label, then position."""
        if channel.wavelength_id:
            return batch.ChannelPick(batch.BY_WAVELENGTH, channel.wavelength_id)
        if channel.label:
            return batch.ChannelPick(batch.BY_LABEL, channel.label)
        return batch.ChannelPick(batch.BY_INDEX, str(channel.index))

    # -- state ----------------------------------------------------------------

    def selected_jobs(self) -> tuple[batch.ImageJob, ...]:
        """The images the current selection names."""
        if self._survey is None:
            return ()
        wells = [item.data(Qt.UserRole) for item in self._wells_list.selectedItems()]
        cycles = [item.data(Qt.UserRole) for item in self._cycles_list.selectedItems()]
        return batch.select_jobs(
            self._survey,
            wells=wells if wells else None,
            acquisitions=cycles if cycles else None,
        )

    def _update_selection_summary(self) -> None:
        jobs = self.selected_jobs()
        if not jobs:
            self._selection_summary.setText("Nothing selected.")
            self._update_enabled()
            return
        name = self._label_name.text().strip() or batch.DEFAULT_LABEL_NAME
        already = sum(1 for job in jobs if batch.has_labels(job, name))
        text = f"{len(jobs)} image(s) selected."
        if already and not self._overwrite.isChecked():
            text += f" {already} already have “{name}” and will be skipped."
        elif already:
            text += f" {already} already have “{name}” and will be replaced."
        self._selection_summary.setText(text)
        self._update_enabled()

    def _update_level_note(self) -> None:
        if self._survey is None or not self._survey.jobs:
            self._level_note.setText("—")
            return
        job = self._survey.jobs[0]
        level = job.level(self._level.value())
        spatial = [
            size for axis, size in zip(job.axes, level.shape) if axis in "YX"
        ]
        _z, y, _x = job.voxel_size_um(self._level.value())
        extent = "×".join(str(int(size)) for size in spatial)
        note = f"{extent} px at {y:.4g} µm/px"
        if self._level.value() >= len(job.levels):
            note += f" — the plate has {len(job.levels)} level(s), so the smallest is used"
        self._level_note.setText(note)

    def refresh_settings_summary(self) -> None:
        """Show the Cellpose settings this run would use, from the Segmentation panel."""
        settings = self.segmentation_settings()
        diameter = "automatic" if not settings.diameter_um else f"{settings.diameter_um:g} µm"
        device = sg.compute_device(prefer_gpu=settings.use_gpu)
        text = f"{settings.resolved_model()}, {settings.mode}, diameter {diameter}"
        if settings.max_solidity > 0:
            text += f", solidity ≤ {settings.max_solidity:g}"
        self._settings_summary.setText(f"{text} — {device.describe()}")

    def segmentation_settings(self) -> sg.SegmentationSettings:
        """The per-image settings, from the Segmentation panel when there is one."""
        panel = getattr(self._app, "segmentation_widget", None)
        if panel is not None and hasattr(panel, "settings"):
            try:
                return panel.settings()
            except Exception:  # pragma: no cover - a half-built panel
                logger.debug("could not read the segmentation panel settings", exc_info=True)
        return sg.SegmentationSettings()

    def settings(self) -> batch.BatchSettings:
        table_dir = self._table_dir.text().strip()
        return batch.BatchSettings(
            segmentation=self.segmentation_settings(),
            channel=self._segment_box.currentData() or batch.ChannelPick(),
            nuclei=self._nuclei_box.currentData() or batch.ChannelPick(),
            measure=self._measure_box.currentData() or batch.ChannelPick(),
            label_name=self._label_name.text().strip() or batch.DEFAULT_LABEL_NAME,
            level=int(self._level.value()),
            overwrite=self._overwrite.isChecked(),
            write_tables=self._write_tables.isChecked(),
            table_dir=Path(table_dir) if table_dir else None,
        )

    def _check_backend(self) -> None:
        message = sg.missing_backend_message()
        if message is None:
            self._backend_notice.setVisible(False)
            return
        self._backend_notice.setText(f"<b>Segmentation is unavailable.</b><br>{message}")
        self._backend_notice.setVisible(True)

    def _update_enabled(self) -> None:
        runnable = (
            self._worker is None
            and self._survey is not None
            and bool(self.selected_jobs())
            and sg.missing_backend_message() is None
        )
        self._run_button.setEnabled(runnable)

    # -- the run --------------------------------------------------------------

    def run(self) -> None:
        """Start the batch on a worker thread."""
        if self._worker is not None:
            self._status.setText("A batch is already running.")
            return
        message = sg.missing_backend_message()
        if message is not None:
            self._check_backend()
            self._status.setText(message.replace("\n\n", " "))
            return

        jobs = self.selected_jobs()
        if not jobs:
            self._status.setText("Nothing selected.")
            return
        settings = self.settings()
        if not settings.channel.is_set:
            self._status.setText("Choose the channel to segment.")
            return

        if not self._confirm(jobs, settings):
            return

        self.refresh_settings_summary()
        self._outcomes = []
        self._running_jobs = jobs
        self._results_table.setRowCount(0)
        self._progress.setRange(0, len(jobs))
        self._progress.setValue(0)
        self._progress.setVisible(True)
        self._status.setText(f"Running {len(jobs)} image(s)…")

        def _progress(text: str) -> None:
            # The worker thread must not touch Qt; the log is where it can speak.
            logger.info("batch: %s", text)

        try:
            from napari.qt.threading import thread_worker
        except Exception:  # pragma: no cover - no Qt threading available
            for outcome in batch.iter_batch(jobs, settings, progress=_progress):
                self._on_outcome(outcome)
            self._on_finished(settings)
            return

        @thread_worker
        def _work():
            for outcome in batch.iter_batch(jobs, settings, progress=_progress):
                yield outcome

        worker = _work()
        worker.yielded.connect(self._on_outcome)
        worker.errored.connect(self._on_error)
        worker.finished.connect(lambda: self._on_finished(settings))
        self._worker = worker
        self._run_button.setEnabled(False)
        self._stop_button.setEnabled(True)
        worker.start()

    def _confirm(self, jobs, settings: batch.BatchSettings) -> bool:
        """Say plainly what is about to be written, and where, before writing it.

        A batch writes into the user's data store, which is not something to start
        on a mis-click — and on a full plate it is hours of GPU time.
        """
        already = sum(1 for job in jobs if batch.has_labels(job, settings.label_name))
        lines = [
            f"Segment {len(jobs)} image(s) and write the masks into the plate as "
            f"“{settings.label_name}”.",
            "",
            f"Store: {self._plate_edit.text().strip()}",
            f"Channel: {settings.channel.describe()}",
            f"Cellpose: {self._settings_summary.text()}",
        ]
        if already:
            lines.append("")
            lines.append(
                f"{already} image(s) already have that label set and will be "
                + ("replaced." if settings.overwrite else "skipped.")
            )
        box = QMessageBox(
            QMessageBox.Question,
            "Run the batch",
            "\n".join(lines),
            QMessageBox.Ok | QMessageBox.Cancel,
            self,
        )
        box.setDefaultButton(QMessageBox.Cancel)
        return box.exec_() == QMessageBox.Ok

    def stop(self) -> None:
        """Ask the run to stop at the next image boundary."""
        if self._worker is None:
            return
        self._status.setText("Stopping after the image being worked on…")
        try:
            self._worker.quit()
        except Exception:  # pragma: no cover - worker already gone
            logger.debug("could not stop the worker", exc_info=True)

    def _on_outcome(self, outcome: batch.ImageOutcome) -> None:
        self._outcomes.append(outcome)
        row = self._results_table.rowCount()
        self._results_table.insertRow(row)
        values = (
            outcome.job.describe(),
            str(outcome.n_objects) if outcome.ok else "",
            f"{outcome.elapsed_s:.0f}",
            outcome.status,
            outcome.message,
        )
        for column, value in enumerate(values):
            item = QTableWidgetItem(value)
            item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            if column == 0:
                item.setData(Qt.UserRole, outcome.job.path)
            self._results_table.setItem(row, column, item)
        self._results_table.scrollToBottom()
        self._progress.setValue(len(self._outcomes))
        self._status.setText(
            f"{len(self._outcomes)} of {len(self._running_jobs)} done — "
            f"{sum(o.n_objects for o in self._outcomes)} object(s) so far."
        )

    def _on_error(self, exc) -> None:
        logger.exception("batch failed", exc_info=exc)
        self._status.setText(f"The batch failed: {exc}")
        QMessageBox.critical(self, "Microscopy Viewer", f"The batch failed:\n{exc}")

    def _on_finished(self, settings: batch.BatchSettings) -> None:
        self._worker = None
        self._stop_button.setEnabled(False)
        self._progress.setVisible(False)

        report = batch.BatchReport(outcomes=list(self._outcomes))
        report.elapsed_s = sum(outcome.elapsed_s for outcome in self._outcomes)
        report.cancelled = len(self._outcomes) < len(self._running_jobs)
        if settings.write_tables and self._survey is not None:
            try:
                report.summary_path = batch.write_summary(report, settings, self._survey)
            except Exception:
                logger.exception("could not write the batch summary")

        text = report.describe()
        if report.summary_path is not None:
            text += f" Summary: {report.summary_path}"
        self._status.setText(text)
        logger.info("batch panel: %s", text)
        self._refresh_wells()
        self._update_enabled()

    def open_selected(self) -> None:
        """Load the image behind the selected row, labels and all."""
        items = self._results_table.selectedItems()
        if not items:
            return
        first = self._results_table.item(items[0].row(), 0)
        path = first.data(Qt.UserRole) if first is not None else None
        if path is None:
            return
        try:
            self._app.open_paths([path])
        except Exception as exc:
            logger.exception("could not open %s", path)
            self._status.setText(f"Could not open that image: {exc}")
