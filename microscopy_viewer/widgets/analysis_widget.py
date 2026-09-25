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

from qtpy.QtCore import QObject, Qt, QTimer, Signal
from qtpy.QtGui import QColor, QFont
from qtpy.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
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
from .. import grouping as gr
from ..utils import get_logger

logger = get_logger("analysis_widget")

#: Columns of the per-sample summary table.
SUMMARY_COLUMNS = ("Sample", "Regions", "Objects", "Channels", "Notes")

#: Columns of the table of group-column rules.
RULE_COLUMNS = ("Column", "Read from", "Find", "Value")

#: A new column's rule: the commonest layout is one folder per condition.
NEW_COLUMN = gr.ColumnRule("Condition", source=gr.FOLDER, rule=gr.WHOLE)

#: Samples listed in the preview; the workbook gets all of them either way.
PREVIEW_ROWS = 200


def _add_choices(combo: QComboBox, choices) -> None:
    """Fill *combo*, each choice explained in its tooltip."""
    for index, choice in enumerate(choices):
        combo.addItem(choice)
        combo.setItemData(index, gr.EXPLAIN.get(choice, ""), Qt.ToolTipRole)
    combo.setToolTip(gr.EXPLAIN.get(combo.currentText(), ""))
    combo.currentTextChanged.connect(lambda text: combo.setToolTip(gr.EXPLAIN.get(text, "")))


def _fit_height(table: QTableWidget, max_rows: int = 0) -> None:
    """Make *table* as tall as its rows (up to *max_rows*, then it scrolls)."""
    rows = table.rowCount() if not max_rows else min(table.rowCount(), max_rows)
    height = table.horizontalHeader().height() + 2 * table.frameWidth()
    height += sum(table.rowHeight(row) for row in range(rows)) or table.verticalHeader().defaultSectionSize()
    if table.horizontalScrollBarPolicy() != Qt.ScrollBarAlwaysOff:
        height += table.horizontalScrollBar().sizeHint().height()
    table.setFixedHeight(height)


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
        layout.addWidget(self._build_groups_box())
        # After the groups: they are part of what a run writes.
        layout.addLayout(self._build_actions())
        layout.addWidget(self._build_results_box(), stretch=1)

        self._status = QLabel(
            "Select samples in the Experiment setup panel, then analyse them. "
            "Nothing has to be open."
        )
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        # Rebuilding the preview reads nothing from disk, but typing a pattern
        # would still redo it on every key: wait for a pause.
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(150)
        self._preview_timer.timeout.connect(self.refresh_preview)
        for rule in gr.load_rules():
            self._add_rule_row(rule)
        panel = getattr(app, "experiment_widget", None)
        if panel is not None and hasattr(panel, "samples_changed"):
            panel.samples_changed.connect(self._rules_changed)
        self.refresh_preview()

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

        self._cell_channels_check = QCheckBox("Measure every channel in every cell")
        self._cell_channels_check.setChecked(True)
        self._cell_channels_check.setToolTip(
            "Each cell's mean, SD, maximum and integrated intensity in every channel "
            "listed above (all of them when empty) — the Cell intensities sheet. Off: "
            "only the channel the cells were segmented on, in the Objects sheet."
        )
        form.addRow("", self._cell_channels_check)
        return box

    def _build_actions(self) -> QHBoxLayout:
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
        return row

    def _build_groups_box(self) -> QGroupBox:
        box = QGroupBox("Groups: genotype and conditions")
        outer = QVBoxLayout(box)

        hint = QLabel(
            "Each column of the workbook is read from the file name or the folder "
            "the file is in. Samples that share a name get the values that tell them "
            "apart added to it, e.g. <i>fish1 (DMSO)</i> and <i>fish1 (drug)</i>."
        )
        hint.setWordWrap(True)
        outer.addWidget(hint)

        self._rules_table = QTableWidget(0, len(RULE_COLUMNS))
        self._rules_table.setHorizontalHeaderLabels(list(RULE_COLUMNS))
        self._rules_table.verticalHeader().setVisible(False)
        self._rules_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._rules_table.setSelectionMode(QAbstractItemView.SingleSelection)
        # Every column shares the panel's width: a rule has to be readable
        # whole, without scrolling sideways to find its value.
        self._rules_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._rules_table.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._rules_table.setToolTip(
            "Read from: the file name, the folder the file is in, the folder above "
            "that, or the whole path below the experiment folder.\n"
            "Find: a genotype word (wt, mut, …), a numbered part of the name (split "
            "at “_”), one of the words you list, all of it, or a regex.\n"
            "Hover over a choice for more."
        )
        outer.addWidget(self._rules_table)

        row = QHBoxLayout()
        add = QPushButton("+ Add column")
        add.setToolTip("Another group column, e.g. the treatment read from the folder name.")
        add.clicked.connect(lambda: self._add_rule_row(gr.ColumnRule(**vars(NEW_COLUMN)), focus=True))
        row.addWidget(add)
        self._remove_rule_button = QPushButton("Remove column")
        self._remove_rule_button.setToolTip("Remove the selected column. Genotype always stays.")
        self._remove_rule_button.clicked.connect(self._remove_rule_row)
        row.addWidget(self._remove_rule_button)
        row.addStretch(1)
        outer.addLayout(row)

        self._parts_label = QLabel("")
        self._parts_label.setWordWrap(True)
        self._parts_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        outer.addWidget(self._parts_label)

        self._preview = QTableWidget(0, 0)
        self._preview.verticalHeader().setVisible(False)
        self._preview.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._preview.setToolTip("What the workbook will say for the selected samples.")
        outer.addWidget(self._preview)

        self._preview_note = QLabel("")
        self._preview_note.setWordWrap(True)
        outer.addWidget(self._preview_note)
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

    # -- group columns --------------------------------------------------------

    def _add_rule_row(self, rule: gr.ColumnRule, focus: bool = False) -> None:
        table = self._rules_table
        row = table.rowCount()
        table.insertRow(row)

        name = QLineEdit(gr.GENOTYPE if row == 0 else rule.name)
        name.setFrame(False)
        if row == 0:
            # Every plot and test groups by genotype: its column can be read any
            # way, but it is always there and always called that.
            name.setReadOnly(True)
            name.setToolTip("Always present: the explorer apps group by it.")
        name.textChanged.connect(self._rules_changed)
        table.setCellWidget(row, 0, name)

        source = QComboBox()
        _add_choices(source, gr.SOURCES)
        source.setCurrentText(rule.source)
        source.currentTextChanged.connect(self._rules_changed)
        table.setCellWidget(row, 1, source)

        how = QComboBox()
        _add_choices(how, gr.RULES)
        how.setCurrentText(rule.rule)
        table.setCellWidget(row, 2, how)

        value = QLineEdit(rule.argument)
        value.setFrame(False)
        value.textChanged.connect(self._rules_changed)
        table.setCellWidget(row, 3, value)

        def _rule_picked(text: str, _value=value) -> None:
            needs = text in (gr.FIELD, gr.ONE_OF, gr.PATTERN)
            _value.setEnabled(needs)
            _value.setPlaceholderText(gr.ARGUMENT_HINTS.get(text, ""))
            _value.setToolTip(gr.EXPLAIN.get(text, ""))
            self._rules_changed()

        how.currentTextChanged.connect(_rule_picked)
        _rule_picked(how.currentText())
        table.resizeRowsToContents()
        _fit_height(table)
        if focus:
            table.selectRow(row)
            name.setFocus()
            name.selectAll()

    def _remove_rule_row(self) -> None:
        rows = {index.row() for index in self._rules_table.selectedIndexes()}
        row = max(rows) if rows else self._rules_table.rowCount() - 1
        if row <= 0:
            self._status.setText("The genotype column always stays; change how it is read instead.")
            return
        self._rules_table.removeRow(row)
        _fit_height(self._rules_table)
        self._rules_changed()

    def rules(self) -> list[gr.ColumnRule]:
        """The group columns as set on the panel, genotype first."""
        table = self._rules_table
        rules = []
        for row in range(table.rowCount()):
            rules.append(gr.ColumnRule(
                name=table.cellWidget(row, 0).text(),
                source=table.cellWidget(row, 1).currentText(),
                rule=table.cellWidget(row, 2).currentText(),
                argument=table.cellWidget(row, 3).text(),
            ))
        return rules or gr.default_rules()

    def _rules_changed(self, *_args) -> None:
        self._preview_timer.start()

    def experiment_root(self) -> Path | None:
        panel = getattr(self._app, "experiment_widget", None)
        text = panel._folder_edit.text().strip() if panel is not None else ""
        return Path(text) if text and Path(text).is_dir() else None

    def sample_labels(self, paths=None) -> list[gr.SampleLabel]:
        """Name and group columns of every sample, as the workbook will have them."""
        paths = self.sample_paths() if paths is None else paths
        return gr.label_samples(paths, self.rules(), self.experiment_root())

    def refresh_preview(self) -> None:
        """Show what the rules make of the selected samples, and remember them."""
        rules = self.rules()
        gr.save_rules(rules)
        paths = self.sample_paths()
        labels = gr.label_samples(paths, rules, self.experiment_root())
        headers = ["Sample", *gr.column_names(rules)]

        if paths:
            first = paths[0]
            numbered = " · ".join(f"<b>{i}</b> {part}" for i, part in enumerate(gr.parts(first.stem), 1))
            self._parts_label.setText(
                f"Parts of <i>{first.stem}</i>: {numbered}"
                f"&nbsp;&nbsp;&nbsp;Folder: <i>{first.parent.name}</i>"
            )
        else:
            self._parts_label.setText("Scan a folder in Experiment setup to preview the columns.")

        table = self._preview
        shown = labels[:PREVIEW_ROWS]
        table.clear()
        table.setColumnCount(len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.setRowCount(len(shown))
        renamed = missing = 0
        grey = QColor(140, 140, 140)
        for row, label in enumerate(shown):
            values = [label.name, label.genotype, *label.conditions.values()]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value or "—")
                if not value:
                    item.setForeground(grey)
                item.setToolTip(str(label.path))
                if column == 0 and label.name != label.path.stem:
                    font = QFont(item.font())
                    font.setBold(True)
                    item.setFont(font)
                    item.setToolTip(f"{label.path}\nRenamed: another sample is also called "
                                    f"“{label.path.stem}”.")
                table.setItem(row, column, item)
        for label in labels:
            renamed += label.name != label.path.stem
            missing += any(not v for v in (label.genotype, *label.conditions.values()))
        table.resizeColumnsToContents()
        _fit_height(table, max_rows=8)

        notes = [f"{len(labels)} sample(s)"]
        if len(labels) > PREVIEW_ROWS:
            notes[0] += f", first {PREVIEW_ROWS} shown"
        if renamed:
            notes.append(f"{renamed} renamed so every name is unique (in bold)")
        if missing:
            notes.append(f"{missing} with an empty column (—)")
        self._preview_note.setText(". ".join(notes) + "." if labels else "")

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
            cell_channels=self._cell_channels_check.isChecked(),
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
        labels = self.sample_labels(paths)
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
            gr.apply(outcomes, labels)
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
        root = self.experiment_root()
        if root is not None:
            return root, root.name
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
        # With the columns as they are now: they may have been changed since the run.
        gr.apply(self._outcomes, self.sample_labels([outcome.path for outcome in self._outcomes]))
        try:
            report = ap.write_report(self._outcomes, parent, name)
        except Exception as exc:
            logger.exception("analysis export failed")
            QMessageBox.critical(self, "Microscopy Viewer", f"Export failed:\n{exc}")
            return None
        self._report_written(report)
        return report[0]
