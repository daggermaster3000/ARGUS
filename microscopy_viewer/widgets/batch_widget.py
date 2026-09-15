"""Dock panel: segment a whole HCS plate in one run.

The Segmentation panel is where settings are chosen — one channel of one image,
looked at, adjusted, run again. This panel is where those settings are *used*: it
walks an OME-Zarr plate, runs the same engine over every well and every 4i cycle
you tick, and writes the masks back into the plate as NGFF ``labels`` groups.

Two things are worth knowing before running one.

**The plate comes from the File explorer panel.** Scan it there and this panel
fills itself in. One store box for the window rather than one per panel: two
paths to keep in step is two lists that can disagree about what is in the plate.

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

from dataclasses import replace
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
from .plate_picker import WellAcquisitionPicker

logger = get_logger("batch_widget")

#: Entry for "no second channel", in the nuclei picker.
NO_CHANNEL = "— none —"
#: Entry for "measure on whatever was segmented", in the measure picker.
SAME_CHANNEL = "— the segmented channel —"

RESULT_COLUMNS = ("Image", "Objects", "Seconds", "Status", "Note")

#: What the panel says before a plate has been scanned. The scanning lives in the
#: File explorer panel now, and a panel that only says "nothing selected" gives
#: nobody any idea where to go.
NO_PLATE = "Scan a plate in the File explorer panel; this one fills itself in."


class BatchSegmentationWidget(QWidget):
    """Pick a plate, pick wells and channels, segment the lot."""

    def __init__(self, app, parent: QWidget | None = None):
        super().__init__(parent)
        self._app = app
        self._survey: batch.PlateSurvey | None = None
        self._worker = None
        self._outcomes: list[batch.ImageOutcome] = []
        self._running_jobs: tuple[batch.ImageJob, ...] = ()

        # The Segmentation panel is handed the viewer, not the app, so it cannot
        # reach this one to keep the shared median setting in step. Leave it a way.
        try:
            self._app.viewer._mv_batch_widget = self
        except Exception:  # pragma: no cover - a viewer stub that refuses attributes
            logger.debug("could not register the batch panel on the viewer", exc_info=True)

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
        inner.addWidget(self._build_images_box())
        inner.addWidget(self._build_channels_box())
        inner.addWidget(self._build_preprocessing_box())
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

        self._status = QLabel(NO_PLATE)
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        self._check_backend()
        self.refresh_settings_summary()
        self._update_enabled()

    # -- construction ---------------------------------------------------------

    def _build_images_box(self) -> QGroupBox:
        box = QGroupBox("Images to run")
        outer = QVBoxLayout(box)

        self._picker = WellAcquisitionPicker(all_cycles=False, note=self._well_note)
        self._picker.selectionChanged.connect(self._update_selection_summary)
        outer.addWidget(self._picker)

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

        self._measure_all = QCheckBox("every channel, in columns of its own")
        self._measure_all.setToolTip(
            "Measure all four stains of each image, not only the one above, adding a set of "
            "intensity columns per channel named after it.\n\n"
            "One segmentation, four answers: it is the only way to ask whether the nuclei "
            "DAPI picked out are the ones carrying the reporter. Costs one extra read of "
            "each channel — seconds per image, not minutes."
        )
        self._measure_all.toggled.connect(self.refresh_settings_summary)
        form.addRow("", self._measure_all)
        return box

    def _build_preprocessing_box(self) -> QGroupBox:
        """What is done to a channel before Cellpose is given it.

        The median filter is the same setting as the Segmentation panel's, shown
        again here because a plate run is where it costs real time and this is the
        panel that run is started from. The two spin boxes are kept in step, so
        there is one value behind them rather than two that disagree.
        """
        box = QGroupBox("Before segmenting")
        form = QFormLayout(box)
        form.setLabelAlignment(Qt.AlignRight)

        self._median = QSpinBox()
        self._median.setRange(0, 10)
        self._median.setValue(0)
        self._median.setSuffix(" px")
        self._median.setSpecialValueText("off")
        self._median.setToolTip(
            "Median-filter each channel before segmenting it, with a square window of 2r+1 "
            "px — shot noise and hot pixels go, edges stay put.\n\n"
            "Only what Cellpose sees is filtered: the masks are written on the original grid "
            "and the object intensities still come from the raw channel. Costs about 12 s per "
            "12000 x 12000 image at radius 1 and 70 s at radius 3, on top of the run itself."
        )
        self._median.valueChanged.connect(self._median_changed)
        form.addRow("Median filter", self._median)

        self._median_cost = QLabel("—")
        self._median_cost.setWordWrap(True)
        form.addRow("", self._median_cost)
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

        self._write_anndata = QCheckBox("and an .h5ad beside each one")
        self._write_anndata.setToolTip(
            "Also write each object table as AnnData, which is what squidpy, scanpy and "
            "the rest of that stack read.\n\n"
            "Measurements go in X, the label and centroids in obs, and the centroid in "
            "obsm[\"spatial\"] — the array a neighbourhood graph is built from. Written "
            "from the measured numbers rather than by reading the CSV back, so the "
            "spatial analysis does not start from rounded floats.\n\n"
            "Needs the anndata package."
        )
        self._write_tables.toggled.connect(self._write_anndata.setEnabled)
        form.addRow("", self._write_anndata)

        self._anndata_one_file = QCheckBox("as one file for the whole run")
        self._anndata_one_file.setToolTip(
            "Write a single .h5ad for the run instead of one per image, with the well and "
            "cycle in obs[\"image\"] and the object names made unique by it.\n\n"
            "A plate is one experiment: a folder of forty-four files is forty-four files to "
            "concatenate before anything can be asked about the plate as a whole. The "
            "per-image CSVs are still written either way."
        )
        self._anndata_one_file.setEnabled(False)
        self._write_anndata.toggled.connect(self._anndata_one_file.setEnabled)
        form.addRow("", self._anndata_one_file)

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

    def _browse_tables(self) -> None:
        plate = str(self._survey.path) if self._survey is not None else ""
        start = self._table_dir.text() or plate or str(Path.home())
        path = QFileDialog.getExistingDirectory(self, "Where to write the tables", start)
        if path:
            self._table_dir.setText(path)

    def plate_changed(self, survey) -> None:
        """Take the plate the File explorer scanned and fill every list from it.

        Called by the explorer rather than reached for, so this panel has no plate
        of its own to get out of step with the one on screen.
        """
        self._survey = survey
        self._picker.set_survey(survey)
        self._refresh_channels()
        self._update_level_note()
        self._status.setText(
            f"{survey.name}: {len(survey.jobs)} image(s). "
            "Choose wells, cycles and channels, then run."
        )
        self._update_enabled()

    def _well_note(self, job: batch.ImageJob) -> str:
        """Marker beside a well that already carries the label set about to be written."""
        name = self._label_name.text().strip() or batch.DEFAULT_LABEL_NAME
        return f" — has {name}" if batch.has_labels(job, name) else ""

    def _refresh_wells(self) -> None:
        """Redraw the well list after the label name changed."""
        self._picker.set_note(self._well_note)

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
        return self._picker.selected_jobs()

    def _update_selection_summary(self) -> None:
        jobs = self.selected_jobs()
        if not jobs:
            self._selection_summary.setText(
                "Nothing selected." if self._survey is not None else NO_PLATE
            )
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
        self._update_median_cost()
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

    def median_radius(self) -> int:
        return int(self._median.value())

    def set_median_radius(self, radius: int) -> None:
        if int(radius) != int(self._median.value()):
            self._median.setValue(int(radius))

    def _median_changed(self, radius: int) -> None:
        panel = getattr(self._app, "segmentation_widget", None)
        if panel is not None and hasattr(panel, "set_median_radius"):
            panel.set_median_radius(int(radius))
        self._update_median_cost()
        self.refresh_settings_summary()

    def _update_median_cost(self) -> None:
        """Say what the filter will add to the run, in minutes, before it starts.

        The filter is per image and scales with the window, so on a plate it is the
        difference between half an hour and an afternoon; a number here beats
        finding that out at image 40 of 46.
        """
        radius = self.median_radius()
        jobs = self.selected_jobs()
        if radius <= 0 or not jobs:
            self._median_cost.setText("—")
            return
        # Measured on this machine: 12 s for a 144 Mpx plane at radius 1, and the cost
        # grows roughly with the window area (40 s at radius 2, 69 s at radius 3).
        level = self._survey.jobs[0].level(self._level.value()) if self._survey else None
        pixels = 1.0
        if level is not None:
            for axis, size in zip(self._survey.jobs[0].axes, level.shape):
                if axis in "YX":
                    pixels *= int(size)
        seconds = 12.0 * (pixels / 144e6) * ((2 * radius + 1) ** 2 / 9.0)
        self._median_cost.setText(
            f"about {seconds:.0f} s per image, {seconds * len(jobs) / 60:.0f} min over "
            f"{len(jobs)} image(s)"
        )

    def refresh_settings_summary(self) -> None:
        """Show the Cellpose settings this run would use, from the Segmentation panel."""
        settings = self.segmentation_settings()
        diameter = "automatic" if not settings.diameter_um else f"{settings.diameter_um:g} µm"
        device = sg.compute_device(prefer_gpu=settings.use_gpu)
        text = f"{settings.resolved_model()}, {settings.mode}, diameter {diameter}"
        if settings.max_solidity > 0:
            text += f", solidity ≤ {settings.max_solidity:g}"
        if self.median_radius() > 0:
            text += f", median r={self.median_radius()} px"
        if self._measure_all.isChecked():
            text += ", every channel measured"
        self._settings_summary.setText(f"{text} — {device.describe()}")

    def segmentation_settings(self) -> sg.SegmentationSettings:
        """The per-image settings, from the Segmentation panel when there is one."""
        settings = sg.SegmentationSettings()
        panel = getattr(self._app, "segmentation_widget", None)
        if panel is not None and hasattr(panel, "settings"):
            try:
                settings = panel.settings()
            except Exception:  # pragma: no cover - a half-built panel
                logger.debug("could not read the segmentation panel settings", exc_info=True)
        # This panel owns the median radius during a plate run: the two spin boxes
        # track each other, but the one the user reached for last is here.
        return replace(settings, median_radius_px=self.median_radius())

    def settings(self) -> batch.BatchSettings:
        table_dir = self._table_dir.text().strip()
        return batch.BatchSettings(
            segmentation=self.segmentation_settings(),
            channel=self._segment_box.currentData() or batch.ChannelPick(),
            nuclei=self._nuclei_box.currentData() or batch.ChannelPick(),
            measure=self._measure_box.currentData() or batch.ChannelPick(),
            measure_all_channels=self._measure_all.isChecked(),
            label_name=self._label_name.text().strip() or batch.DEFAULT_LABEL_NAME,
            level=int(self._level.value()),
            overwrite=self._overwrite.isChecked(),
            write_tables=self._write_tables.isChecked(),
            write_anndata=self._write_anndata.isChecked(),
            anndata_single_file=self._anndata_one_file.isChecked(),
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
            f"Store: {self._survey.path if self._survey is not None else '—'}",
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
            if settings.write_anndata and settings.anndata_single_file:
                # Also drops the frames the run has been holding for this.
                report.anndata_path = batch.write_combined_anndata(
                    report, settings, self._survey
                )

        # The explorer lists the tables beside the plate off an index it built when
        # it scanned; a run that has just written more of them makes that stale.
        explorer_panel = (getattr(self._app, "panels", {}) or {}).get("file_explorer")
        rescan = getattr(explorer_panel, "tables_changed", None)
        if rescan is not None:
            try:
                rescan()
            except Exception:
                logger.exception("could not refresh the explorer's table list")

        text = report.describe()
        if report.summary_path is not None:
            text += f" Summary: {report.summary_path}"
        if report.anndata_path is not None:
            text += f" AnnData: {report.anndata_path.name}"
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
