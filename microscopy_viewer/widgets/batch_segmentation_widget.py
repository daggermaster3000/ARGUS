"""The Segmentation panel's Batch tab: Cellpose over every selected sample.

Lives beside the settings it runs with — model, mode, diameters, device are the
Setup tab's, read when the run starts, so there is one set of controls for one
set of parameters. The samples are the ones selected in *Experiment setup*;
nothing has to be open.

Each file's label map goes back into the file it came from (see
:mod:`microscopy_viewer.ims_store`). HDF5 will not open a file for writing while
it is open for reading, so a sample on screen is closed before its file is
written, and the experiment panel's scan is refused while a batch is running.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from qtpy.QtCore import QObject, Qt, Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .. import experiment as ex
from ..exports import WORKBOOK_FILTER, default_stem, export_sheets
from ..utils import get_logger

logger = get_logger("batch_segmentation_widget")


class _Relay(QObject):
    """Carries worker-thread progress text onto the GUI thread."""

    message = Signal(str)


def channel_spec(text: str):
    """A channel box's contents as an index or a name; empty means none."""
    cleaned = str(text).strip()
    if not cleaned:
        return None
    return int(cleaned) if cleaned.isdigit() else cleaned


class BatchSegmentationTab(QWidget):
    """Segment the samples selected in Experiment setup, writing labels into them."""

    def __init__(self, app, settings: Callable[[], object], parent: QWidget | None = None):
        super().__init__(parent)
        self._app = app
        self._settings = settings
        self._worker = None
        self._cancelled = False
        self._outcomes: list[ex.BatchOutcome] = []
        self._relay = _Relay()
        self._relay.message.connect(self._log)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        note = QLabel(
            "Runs on the samples selected in Experiment setup (all of them when none "
            "is selected), with the model, mode, diameters and device on the Setup tab."
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)

        self._channel_edit = QLineEdit("dapi")
        self._channel_edit.setToolTip(
            "Which channel to segment, matched against the channel names in each file — "
            "“dapi” finds it wherever it sits, which matters because the channel order "
            "is not the same in every acquisition.\n"
            "A bare number is taken as an index instead. A name that matches nothing "
            "skips that file and says so, rather than quietly segmenting channel 0."
        )
        form.addRow("Segment channel", self._channel_edit)

        self._measure_edit = QLineEdit("")
        self._measure_edit.setToolTip(
            "Optional: the channel per-object intensities are read from. Segment on DAPI, "
            "measure on the reporter."
        )
        form.addRow("Measure channel", self._measure_edit)

        self._restrict = QCheckBox("Only inside the stored ROIs")
        self._restrict.setToolTip(
            "Blank everything outside each file's stored ROIs before segmenting. This is "
            "what drawing them was for: Cellpose does not know that the skin and the yolk "
            "are not brain, and finds plenty of objects in both.\n"
            "A file with no stored ROIs is segmented whole."
        )
        form.addRow("Restrict", self._restrict)

        self._save_labels = QCheckBox("Write the labels into each .ims file")
        self._save_labels.setChecked(True)
        self._save_labels.setToolTip(
            "Stored under /ARGUS/Labels inside the file, beside the image data rather "
            "than in a sidecar, so the result travels with the sample.\n"
            "Imaris ignores the group and opens the file normally — but it does not show "
            "these as its own Surfaces either."
        )
        form.addRow("Results", self._save_labels)
        layout.addLayout(form)

        row = QHBoxLayout()
        self._run_button = QPushButton("Run on selected")
        self._run_button.setToolTip(
            "Segment every sample selected in Experiment setup with the settings on the "
            "Setup tab. Nothing selected means all of them."
        )
        self._run_button.clicked.connect(self.run)
        row.addWidget(self._run_button)

        self._stop_button = QPushButton("Stop")
        self._stop_button.setEnabled(False)
        self._stop_button.clicked.connect(self.stop)
        row.addWidget(self._stop_button)

        export = QPushButton("Export…")
        export.setToolTip(
            "Write the per-sample counts, the per-region counts and every object to one "
            "workbook. The genotype column is read out of each file name.\n"
            "The Analysis panel writes the same sheets and more, straight from the files."
        )
        export.clicked.connect(self.export)
        row.addWidget(export)
        row.addStretch(1)
        layout.addLayout(row)

        self._log_view = QPlainTextEdit()
        self._log_view.setReadOnly(True)
        self._log_view.setPlaceholderText("Progress appears here.")
        layout.addWidget(self._log_view, stretch=1)

        self._status = QLabel("")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

    # -- helpers --------------------------------------------------------------

    def _experiment(self):
        return getattr(self._app, "experiment_widget", None)

    def is_running(self) -> bool:
        return self._worker is not None

    def _log(self, text: str) -> None:
        self._log_view.appendPlainText(text)
        self._status.setText(text)

    def _busy(self, running: bool) -> None:
        self._run_button.setEnabled(not running)
        self._stop_button.setEnabled(running)

    def stop(self) -> None:
        """Ask the batch to stop after the file it is on."""
        self._cancelled = True
        self._log("Stopping after the current file…")

    def _should_cancel(self) -> bool:
        return self._cancelled

    def batch_settings(self):
        """The Setup tab's settings, read now rather than copied at startup."""
        from .. import segmentation as sg

        try:
            return self._settings()
        except Exception:
            logger.debug("could not read the segmentation settings", exc_info=True)
            return sg.SegmentationSettings()

    def options(self) -> ex.BatchOptions | None:
        channel = channel_spec(self._channel_edit.text())
        if channel is None:
            return None
        return ex.BatchOptions(
            channel=channel,
            measure_channel=channel_spec(self._measure_edit.text()),
            settings=self.batch_settings(),
            save_to_file=self._save_labels.isChecked(),
            restrict_to_rois=self._restrict.isChecked(),
        )

    # -- running --------------------------------------------------------------

    def run(self) -> None:
        """Segment every selected sample and write the labels back into the files."""
        experiment = self._experiment()
        if experiment is None:
            self._status.setText("The Experiment setup panel is not available.")
            return
        if self._worker is not None or experiment.is_running():
            self._status.setText("Something is already running — stop it first.")
            return
        entries = [entry for entry in experiment.selected_entries() if entry.readable]
        if not entries:
            self._status.setText(
                "No samples. Choose a folder in Experiment setup and scan it first."
            )
            return
        options = self.options()
        if options is None:
            self._status.setText("Say which channel to segment.")
            return

        paths = [entry.path for entry in entries]
        if options.save_to_file:
            # Writing needs the files to itself, and they may well be the ones
            # just used to draw the ROIs.
            experiment._close_paths(paths)

        settings = options.settings
        self._outcomes = []
        self._log_view.clear()
        self._cancelled = False
        self._busy(True)
        self._log(
            f"Segmenting {len(paths)} sample(s) on “{options.channel}” with "
            f"{settings.resolved_model()}, {settings.mode}."
        )

        relay = self._relay

        from napari.qt.threading import thread_worker

        @thread_worker
        def _run():
            return ex.run_batch(
                paths, options, progress=relay.message.emit, should_cancel=self._should_cancel
            )

        worker = _run()
        worker.returned.connect(self._on_done)
        worker.errored.connect(self._on_error)
        worker.finished.connect(self._clear_worker)
        self._worker = worker
        worker.start()

    def _clear_worker(self) -> None:
        self._worker = None
        self._cancelled = False
        self._busy(False)

    def _on_error(self, exc) -> None:
        logger.exception("batch segmentation worker failed: %s", exc)
        self._log(f"Failed: {exc}")

    def _on_done(self, outcomes) -> None:
        self._outcomes = list(outcomes)
        done = [outcome for outcome in self._outcomes if outcome.ok]
        failed = [outcome for outcome in self._outcomes if not outcome.ok]
        total = sum(outcome.n_objects for outcome in done)
        seconds = sum(outcome.elapsed_s for outcome in self._outcomes)

        summary = f"{total} object(s) across {len(done)} sample(s) in {seconds:.0f} s."
        # Say how many label maps reached the files, not just how many objects
        # were found. The two can differ — a read-only file, a full disk — and
        # the difference is only noticed weeks later, when the counts are wanted
        # and the .ims files turn out to be empty.
        saved = [outcome for outcome in self._outcomes if outcome.saved]
        if self._save_labels.isChecked():
            summary += f" Labels written into {len(saved)} of {len(self._outcomes)} file(s)."
            if len(saved) < len(done):
                summary += " Open a sample in Experiment setup to check."
        else:
            summary += " Nothing written into the files — “Write the labels” is off."
        if failed:
            summary += " Failed: " + ", ".join(
                f"{outcome.name} ({outcome.error})" for outcome in failed[:2]
            )
            if len(failed) > 2:
                summary += f" and {len(failed) - 2} more"
        if self._cancelled:
            summary += " Stopped early."
        self._log(summary)
        experiment = self._experiment()
        if experiment is not None:
            experiment._refresh_entries([outcome.path for outcome in self._outcomes])
        logger.info("batch finished: %s", summary)

    # -- export ---------------------------------------------------------------

    def export(self, path: str | None = None):
        """Write the batch summary and every object it found to one workbook."""
        if not self._outcomes:
            self._status.setText("Nothing to export — run a batch first.")
            return None
        if not path:
            experiment = self._experiment()
            folder = (experiment.folder() if experiment is not None else None) or Path.home()
            suggested = str(Path(folder) / f"{default_stem('batch_segmentation')}.xlsx")
            path, _selected = QFileDialog.getSaveFileName(
                self, "Export batch results", suggested, WORKBOOK_FILTER
            )
            if not path:
                return None
        sheets = {
            "Samples": ex.batch_dataframe(self._outcomes),
            "Regions": ex.regions_dataframe(self._outcomes),
            "Objects": ex.objects_dataframe(self._outcomes),
        }
        try:
            written = export_sheets(sheets, path)
        except Exception as exc:
            logger.exception("batch export failed")
            QMessageBox.critical(self, "Microscopy Viewer", f"Export failed:\n{exc}")
            return None
        self._log(f"Batch results written to {written}.")
        return written
