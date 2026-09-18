"""Dock panel: describe every region of the selected samples, for PCA.

Works on the samples selected in the *Experiment setup* panel — nothing needs to
be open. Each file's stored label map and outlines are read back, every channel
is summarised inside every outline, and the lot goes to one workbook: the three
sheets the batch export writes, plus per-region features, per-region channel
intensities, and a one-row-per-sample matrix ready for PCA. Every run leaves a
dated report folder in the experiment folder with that workbook and its figures
(see :mod:`microscopy_viewer.analysis_plots`).

The numbers come from :mod:`microscopy_viewer.analysis`; this is only its view.
"""

from __future__ import annotations

from pathlib import Path

from qtpy.QtCore import QObject, Qt, Signal
from qtpy.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .. import analysis as an
from .. import analysis_plots as ap
from ..utils import get_logger

logger = get_logger("analysis_widget")

#: Columns of the per-sample summary table.
SUMMARY_COLUMNS = ("Sample", "Regions", "Objects", "Channels", "Notes")


class _Relay(QObject):
    """Carries worker-thread progress text onto the GUI thread."""

    message = Signal(str)


class AnalysisWidget(QWidget):
    """Per-region morphometrics and channel intensities across samples."""

    def __init__(self, app, parent: QWidget | None = None):
        super().__init__(parent)
        self._app = app
        self._worker = None
        self._cancelled = False
        self._outcomes: list[an.AnalysisOutcome] = []
        #: Folder the most recent report was written into.
        self.last_report: Path | None = None
        self._relay = _Relay()
        self._relay.message.connect(self._log)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(self._build_settings_box())
        layout.addWidget(self._build_results_box(), stretch=1)

        self._status = QLabel(
            "Select samples in the Experiment setup panel, then analyse them. "
            "Nothing has to be open."
        )
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

    # -- construction ---------------------------------------------------------

    def _build_settings_box(self) -> QGroupBox:
        box = QGroupBox("What to read")
        form = QFormLayout(box)
        form.setLabelAlignment(Qt.AlignRight)

        self._label_edit = QLineEdit("")
        self._label_edit.setPlaceholderText("first stored label map")
        self._label_edit.setToolTip(
            "Which label map stored in each file to count, matched against its name "
            "(“dapi” finds “DAPI labels”). Empty takes the first one.\n"
            "A file with no matching map still gets its region shapes and channel "
            "intensities; its object columns are left blank."
        )
        form.addRow("Label map", self._label_edit)

        self._measure_edit = QLineEdit("")
        self._measure_edit.setPlaceholderText("the channel it was segmented on")
        self._measure_edit.setToolTip(
            "Channel the per-object intensities are read from — a name or an index. "
            "Empty uses the channel recorded with the label map."
        )
        form.addRow("Object intensity", self._measure_edit)

        self._channels_edit = QLineEdit("")
        self._channels_edit.setPlaceholderText("all channels")
        self._channels_edit.setToolTip(
            "Channels to describe inside each region, comma separated — names or "
            "indices. Empty describes every channel in the file."
        )
        form.addRow("Channels", self._channels_edit)

        self._level_spin = QSpinBox()
        self._level_spin.setRange(0, 8)
        self._level_spin.setToolTip(
            "Pyramid level the region intensities are read at. 0 is full resolution "
            "and exact; 1 reads an eighth of the data with nearly the same means and "
            "percentiles. A file with fewer levels uses its coarsest."
        )
        form.addRow("Intensity level", self._level_spin)

        self._outlines_check = QCheckBox("Trace every cell's outline")
        self._outlines_check.setChecked(True)
        self._outlines_check.setToolTip(
            "Outline each segmented cell as seen from above and measure its shape — "
            "the Cell shapes sheet and cell_outlines.npz, which the region explorer "
            "app draws. A few seconds per ten thousand cells."
        )
        form.addRow("Cells", self._outlines_check)

        row = QHBoxLayout()
        self._run_button = QPushButton("Analyse selected")
        self._run_button.setToolTip(
            "Analyse the samples selected in Experiment setup — all of them when "
            "none is selected."
        )
        self._run_button.clicked.connect(self.run)
        row.addWidget(self._run_button)

        self._stop_button = QPushButton("Stop")
        self._stop_button.setEnabled(False)
        self._stop_button.clicked.connect(self.stop)
        row.addWidget(self._stop_button)

        self._export_button = QPushButton("Save report to…")
        self._export_button.setToolTip(
            "Write the last run's report folder somewhere else. Every run already "
            "writes one into the experiment folder: the workbook (batch sheets plus "
            "region features, region intensities, PCA matrix), violin plots of cell "
            "count and region area, and a PCA of the cells."
        )
        self._export_button.clicked.connect(self.export)
        row.addWidget(self._export_button)
        row.addStretch(1)
        form.addRow(row)
        return box

    def _build_results_box(self) -> QGroupBox:
        box = QGroupBox("Results")
        outer = QVBoxLayout(box)

        self._table = QTableWidget(0, len(SUMMARY_COLUMNS))
        self._table.setHorizontalHeaderLabels(list(SUMMARY_COLUMNS))
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.horizontalHeader().setSectionResizeMode(
            len(SUMMARY_COLUMNS) - 1, QHeaderView.Stretch
        )
        outer.addWidget(self._table, stretch=1)

        self._log_view = QPlainTextEdit()
        self._log_view.setReadOnly(True)
        self._log_view.setMaximumHeight(120)
        self._log_view.setPlaceholderText("Progress appears here.")
        outer.addWidget(self._log_view)
        return box

    # -- helpers --------------------------------------------------------------

    def _log(self, text: str) -> None:
        self._log_view.appendPlainText(text)
        self._status.setText(text)

    def _busy(self, running: bool) -> None:
        self._run_button.setEnabled(not running)
        self._export_button.setEnabled(not running)
        self._stop_button.setEnabled(running)

    def stop(self) -> None:
        self._cancelled = True
        self._log("Stopping after the current file…")

    def _should_cancel(self) -> bool:
        return self._cancelled

    @staticmethod
    def _channel_spec(text: str):
        cleaned = str(text).strip()
        if not cleaned:
            return None
        return int(cleaned) if cleaned.isdigit() else cleaned

    def options(self) -> an.AnalysisOptions:
        """The settings on the panel, as the analysis takes them."""
        channels = [
            spec
            for spec in (self._channel_spec(part) for part in self._channels_edit.text().split(","))
            if spec is not None
        ]
        return an.AnalysisOptions(
            label_key=self._label_edit.text().strip(),
            measure_channel=self._channel_spec(self._measure_edit.text()),
            intensity_level=int(self._level_spin.value()),
            channels=channels,
            cell_outlines=self._outlines_check.isChecked(),
        )

    def sample_paths(self) -> list[Path]:
        """Readable samples selected in the Experiment setup panel."""
        panel = getattr(self._app, "experiment_widget", None)
        if panel is None:
            return []
        return [entry.path for entry in panel.selected_entries() if entry.readable]

    # -- running --------------------------------------------------------------

    def run(self) -> None:
        if self._worker is not None:
            self._status.setText("An analysis is already running — stop it first.")
            return
        paths = self.sample_paths()
        if not paths:
            self._status.setText(
                "No samples. Choose a folder in the Experiment setup panel and scan it first."
            )
            return

        options = self.options()
        self._outcomes = []
        self._table.setRowCount(0)
        self._log_view.clear()
        self._cancelled = False
        self._busy(True)
        self._log(f"Analysing {len(paths)} sample(s)…")

        relay = self._relay

        from napari.qt.threading import thread_worker

        parent, name = self.report_target()

        @thread_worker
        def _run():
            outcomes = an.analyse(
                paths, options, progress=relay.message.emit, should_cancel=self._should_cancel
            )
            report = None
            if any(outcome.ok for outcome in outcomes):
                relay.message.emit("Writing the report folder…")
                try:
                    report = ap.write_report(outcomes, parent, name)
                except Exception as exc:
                    logger.exception("could not write the analysis report")
                    report = exc
            return outcomes, report

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
        logger.exception("analysis worker failed: %s", exc)
        self._log(f"Failed: {exc}")

    def report_target(self) -> tuple[Path, str]:
        """Where a run's report folder goes, and the name it starts with."""
        panel = getattr(self._app, "experiment_widget", None)
        text = panel._folder_edit.text().strip() if panel is not None else ""
        if text and Path(text).is_dir():
            return Path(text), Path(text).name
        return Path.home(), "experiment"

    def _on_done(self, result) -> None:
        outcomes, report = result
        self._outcomes = list(outcomes)
        self._fill_table(self._outcomes)
        done = [outcome for outcome in self._outcomes if outcome.ok]
        failed = len(self._outcomes) - len(done)
        regions = sum(len(outcome.shapes) for outcome in done)
        summary = f"Analysed {len(done)} sample(s), {regions} region(s) in all."
        if failed:
            summary += f" {failed} failed — see the table."
        if self._cancelled:
            summary += " Stopped early."
        self._log(summary)
        self._report_written(report)

    def _report_written(self, report) -> None:
        if report is None:
            return
        if isinstance(report, Exception):
            self._log(f"Could not write the report: {report}")
            return
        folder, notes = report
        self.last_report = folder
        self._log(f"Report written to {folder}.")
        for note in notes:
            self._log(f"  not drawn — {note}")

    def _fill_table(self, outcomes) -> None:
        self._table.setRowCount(len(outcomes))
        for row, outcome in enumerate(outcomes):
            notes = outcome.error or "; ".join(outcome.warnings)
            values = (
                outcome.name,
                str(len(outcome.shapes)),
                str(outcome.n_objects) if outcome.label_key else "—",
                str(len({stat.channel for stat in outcome.channel_stats})),
                notes,
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column in (1, 2, 3):
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                if outcome.error:
                    item.setForeground(Qt.red)
                self._table.setItem(row, column, item)

    # -- export ---------------------------------------------------------------

    def export(self, parent: str | None = None):
        """Write the last run's report folder under *parent*. Returns the folder."""
        if not self._outcomes:
            self._status.setText("Nothing to save — analyse first.")
            return None
        default_parent, name = self.report_target()
        if not parent:
            parent = QFileDialog.getExistingDirectory(
                self, "Save the analysis report into", str(default_parent)
            )
            if not parent:
                return None
        try:
            report = ap.write_report(self._outcomes, parent, name)
        except Exception as exc:
            logger.exception("analysis export failed")
            QMessageBox.critical(self, "Microscopy Viewer", f"Export failed:\n{exc}")
            return None
        self._report_written(report)
        return report[0]
