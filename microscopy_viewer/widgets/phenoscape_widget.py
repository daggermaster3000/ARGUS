"""Dock panel: run the phenoscapes 4i pipeline, step by step, and open what it made.

Phenoscapes is the processing pipeline of the brain organoid morphodynamics
paper (quadbio/morphodynamics_human_brain_organoid): mask, align the cycles with
elastix, crop and denoise, subtract the elution cycles, segment with Cellpose,
and tabulate every stain per cell. This panel drives it; the steps and their
folders are in :mod:`microscopy_viewer.phenoscape`.

**It runs in another Python.** Phenoscapes pins numpy 1.26, cellpose 3 and
itk-elastix, and calls matplotlib and scipy functions later releases removed, so
it cannot be imported where napari is. Point **Python** at an environment that
has it (``pip install -r requirements.txt`` from the phenoscapes folder) and
press **Check**; **Code** is the folder holding the ``phenoscapes`` package, for
when it is cloned rather than installed. **Download…** fetches it.

**The run is a child process**, so the window stays usable, the log fills as it
goes, and Stop ends it outright. Each step writes its own folder under the
output directory, exactly as the original command line does, so a step can be
rerun alone — segment again with another diameter without realigning — and the
original can pick up where this left off.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from qtpy.QtCore import QProcess, QProcessEnvironment, QSettings, Qt, QUrl
from qtpy.QtGui import QDesktopServices
from qtpy.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .. import phenoscape as ph
from ..utils import get_logger

logger = get_logger("phenoscape_widget")

#: Lines of log kept on screen. The whole log is in ``logs/<sample>.log``.
LOG_LINES = 5_000

STATUS_COLUMN = 1
#: What a step's row says.
WAITING, RUNNING, DONE, FAILED = "queued", "running…", "done", "failed"


def _ints(text: str) -> list[int]:
    """``"1, 6, 11"`` → ``[1, 6, 11]``. Anything that is not a number is dropped."""
    return [int(part) for part in text.replace(";", ",").split(",") if part.strip().lstrip("-").isdigit()]


def _ints_text(values) -> str:
    return ", ".join(str(v) for v in (values or []))


class _ConfigDialog(QDialog):
    """The whole config as YAML, for the keys the form does not show."""

    def __init__(self, text: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Phenoscapes config")
        self.resize(560, 640)
        layout = QVBoxLayout(self)
        note = QLabel(
            "Every key phenoscapes reads, in its own format. Keys left out are filled "
            "from the defaults."
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        self.editor = QPlainTextEdit(text)
        self.editor.setLineWrapMode(QPlainTextEdit.NoWrap)
        layout.addWidget(self.editor, stretch=1)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.config: dict | None = None

    def _accept(self) -> None:
        try:
            self.config = ph.parse_config_text(self.editor.toPlainText())
        except Exception as exc:  # noqa: BLE001 - shown, and the dialog stays open
            QMessageBox.warning(self, "Phenoscapes config", f"Not valid YAML: {exc}")
            return
        self.accept()


class PhenoscapeWidget(QWidget):
    """Choose an environment, a sample and the steps; run them; open the results."""

    def __init__(self, app, parent: QWidget | None = None):
        super().__init__(parent)
        self._app = app
        self._viewer = app.viewer
        self._config = ph.default_config()
        self._process: QProcess | None = None
        self._buffer = ""
        self._mode = ""  # "check" or "run"
        self._check_output: list[str] = []
        self._failed: list[str] = []
        self._settings = QSettings("ARGUS", "phenoscape")
        self._syncing = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        page = QWidget()
        inner = QVBoxLayout(page)
        inner.setContentsMargins(0, 0, 0, 0)
        inner.addWidget(self._build_environment_box())
        inner.addWidget(self._build_data_box())
        inner.addWidget(self._build_settings_box())
        inner.addWidget(self._build_steps_box())
        inner.addWidget(self._build_outputs_box())
        inner.addStretch(1)
        scroll.setWidget(page)
        layout.addWidget(scroll, stretch=3)

        layout.addLayout(self._build_run_row())
        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumBlockCount(LOG_LINES)
        self._log.setLineWrapMode(QPlainTextEdit.NoWrap)
        self._log.setPlaceholderText("What the pipeline prints appears here.")
        layout.addWidget(self._log, stretch=2)

        self._status = QLabel("")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        self._restore()
        self._config_to_form()
        self._refresh_samples()
        self._update_enabled()

    # -- construction ---------------------------------------------------------

    def _path_row(self, edit: QLineEdit, browse) -> QWidget:
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(edit, stretch=1)
        button = QPushButton("…")
        button.setMaximumWidth(30)
        button.clicked.connect(browse)
        h.addWidget(button)
        return row

    def _build_environment_box(self) -> QGroupBox:
        box = QGroupBox("Phenoscapes")
        form = QFormLayout(box)
        form.setLabelAlignment(Qt.AlignRight)

        self._python = QLineEdit(ph.default_python())
        self._python.setToolTip(
            "The Python that runs the pipeline: python.exe of an environment with "
            "phenoscapes' requirements installed. Not napari's own, as a rule — "
            "phenoscapes pins numpy 1.26 and needs itk-elastix and smo."
        )
        form.addRow("Python", self._path_row(self._python, self._browse_python))

        self._source = QLineEdit()
        self._source.setPlaceholderText("installed in that Python")
        self._source.setToolTip(
            "The folder that holds the phenoscapes package folder — 4i_processing in a "
            "clone of the repository. Put on PYTHONPATH for the run. Leave empty when "
            "phenoscapes is pip-installed in that Python."
        )
        form.addRow("Code", self._path_row(self._source, self._browse_source))

        row = QHBoxLayout()
        self._check_button = QPushButton("Check")
        self._check_button.setToolTip("Try importing phenoscapes and its dependencies in that Python.")
        self._check_button.clicked.connect(self.check_environment)
        row.addWidget(self._check_button)
        download = QPushButton("Download…")
        download.setToolTip(
            "Fetch the phenoscapes package from GitHub into a folder of your choice and "
            "use it as Code. Its requirements.txt comes with it; installing them is up to you."
        )
        download.clicked.connect(self.download_source)
        row.addWidget(download)
        row.addStretch(1)
        form.addRow("", row)

        self._env_status = QLabel("Not checked.")
        self._env_status.setWordWrap(True)
        form.addRow("", self._env_status)
        return box

    def _build_data_box(self) -> QGroupBox:
        box = QGroupBox("Data")
        form = QFormLayout(box)
        form.setLabelAlignment(Qt.AlignRight)

        self._input = QLineEdit()
        self._input.setPlaceholderText("the folder holding 'stitched'")
        self._input.setToolTip(
            "Laid out as phenoscapes expects: <this folder>/stitched/<sample>/*.tif, one "
            "TIFF per cycle and channel."
        )
        self._input.editingFinished.connect(self._refresh_samples)
        form.addRow("Input", self._path_row(self._input, self._browse_input))

        self._samples = QListWidget()
        self._samples.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self._samples.setMaximumHeight(110)
        self._samples.setToolTip("The samples to run, one after the other. Select several with Ctrl.")
        self._samples.itemSelectionChanged.connect(self._samples_changed)
        form.addRow("Samples", self._samples)

        self._sample_note = QLabel("—")
        self._sample_note.setWordWrap(True)
        form.addRow("", self._sample_note)

        self._metadata = QLineEdit()
        self._metadata.setPlaceholderText("stain table (.csv)")
        self._metadata.setToolTip(
            "CSV with cycle_id, channel_id and stain, one row per cycle and channel. A "
            "cycle with an empty stain is a background (elution) cycle, which is what "
            "background subtraction subtracts."
        )
        self._metadata.textChanged.connect(self._update_enabled)
        form.addRow("Stains", self._path_row(self._metadata, self._browse_metadata))

        self._output = QLineEdit()
        self._output.setPlaceholderText("where the step folders go")
        self._output.editingFinished.connect(self._refresh_outputs)
        form.addRow("Output", self._path_row(self._output, self._browse_output))
        return box

    def _build_settings_box(self) -> QGroupBox:
        box = QGroupBox("Settings")
        form = QFormLayout(box)
        form.setLabelAlignment(Qt.AlignRight)

        self._ref_cycle = QSpinBox()
        self._ref_cycle.setRange(0, 999)
        self._ref_cycle.setToolTip("The cycle every other is aligned onto, and whose nuclei are segmented.")
        form.addRow("Reference cycle", self._ref_cycle)

        self._nuclei_channel = QSpinBox()
        self._nuclei_channel.setRange(0, 99)
        self._nuclei_channel.setToolTip("The channel number of the nuclear stain, in every cycle.")
        form.addRow("Nuclei channel", self._nuclei_channel)

        self._nuclei_stain = QLineEdit()
        self._nuclei_stain.setToolTip("What the nuclear stain is called in the feature tables.")
        form.addRow("Nuclei stain", self._nuclei_stain)

        self._stain_channels = QLineEdit()
        self._stain_channels.setToolTip("The antibody channels, comma-separated.")
        form.addRow("Stain channels", self._stain_channels)

        self._align_refs = QLineEdit()
        self._align_refs.setToolTip(
            "Cycles that become the new alignment reference once aligned, so a long "
            "experiment is not aligned across twenty cycles of drift in one jump."
        )
        form.addRow("Alignment refs", self._align_refs)

        self._processes = QSpinBox()
        self._processes.setRange(1, 64)
        self._processes.setToolTip("Processes for crop and denoise. Each holds one image.")
        form.addRow("Processes", self._processes)

        self._model = QComboBox()
        self._model.setEditable(True)
        self._model.addItems(["cyto2", "cyto", "nuclei", "cyto3"])
        form.addRow("Cellpose model", self._model)

        self._diameter = QSpinBox()
        self._diameter.setRange(0, 500)
        self._diameter.setSuffix(" px")
        form.addRow("Diameter", self._diameter)

        self._flow = QDoubleSpinBox()
        self._flow.setRange(-10.0, 10.0)
        self._flow.setSingleStep(0.1)
        form.addRow("Flow threshold", self._flow)

        self._cellprob = QDoubleSpinBox()
        self._cellprob.setRange(-10.0, 10.0)
        self._cellprob.setSingleStep(0.5)
        form.addRow("Cell prob. threshold", self._cellprob)

        self._stat = QComboBox()
        self._stat.addItems(["median", "mean", "max", "min"])
        self._stat.setToolTip("The per-cell statistic that goes into the AnnData.")
        form.addRow("AnnData statistic", self._stat)

        for widget in (self._ref_cycle, self._nuclei_channel, self._processes, self._diameter):
            widget.valueChanged.connect(self._form_to_config)
        for widget in (self._flow, self._cellprob):
            widget.valueChanged.connect(self._form_to_config)
        for widget in (self._nuclei_stain, self._stain_channels, self._align_refs):
            widget.editingFinished.connect(self._form_to_config)
        for widget in (self._model, self._stat):
            widget.currentTextChanged.connect(self._form_to_config)

        row = QHBoxLayout()
        for label, slot, tip in (
            ("Load…", self._load_config, "Read a phenoscapes config.yaml."),
            ("Save…", self._save_config, "Write this config as a phenoscapes config.yaml."),
            ("Every key…", self._edit_config, "Edit the whole config as YAML."),
            ("Defaults", self._reset_config, "Back to phenoscapes' own defaults."),
        ):
            button = QPushButton(label)
            button.setToolTip(tip)
            button.clicked.connect(slot)
            row.addWidget(button)
        form.addRow("", row)
        return box

    def _build_steps_box(self) -> QGroupBox:
        box = QGroupBox("Steps")
        outer = QVBoxLayout(box)
        note = QLabel(
            "Ticked steps make up the pipeline, in this order. Each reads what the one "
            "before it wrote, so a step run alone needs its predecessors' folders on disk."
        )
        note.setWordWrap(True)
        outer.addWidget(note)

        self._steps = QTreeWidget()
        self._steps.setColumnCount(2)
        self._steps.setHeaderLabels(["Step", "Status"])
        self._steps.setRootIsDecorated(False)
        self._steps.setSelectionMode(QAbstractItemView.SingleSelection)
        for step in ph.STEPS:
            item = QTreeWidgetItem([step.title, ""])
            item.setData(0, Qt.UserRole, step.key)
            item.setToolTip(0, f"{step.about}\n\nWrites: <output>/{step.output}")
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(0, Qt.Unchecked)
            self._steps.addTopLevelItem(item)
        self._steps.resizeColumnToContents(0)
        self._steps.setMinimumHeight(self._steps.sizeHintForRow(0) * (len(ph.STEPS) + 2))
        self._steps.itemChanged.connect(self._step_toggled)
        self._steps.itemSelectionChanged.connect(self._update_enabled)
        outer.addWidget(self._steps)
        return box

    def _build_outputs_box(self) -> QGroupBox:
        box = QGroupBox("Results")
        outer = QVBoxLayout(box)
        row = QHBoxLayout()
        row.addWidget(QLabel("Sample"))
        self._output_sample = QComboBox()
        self._output_sample.currentTextChanged.connect(self._refresh_outputs)
        row.addWidget(self._output_sample, stretch=1)
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self._refresh_outputs)
        row.addWidget(refresh)
        outer.addLayout(row)

        self._outputs = QListWidget()
        self._outputs.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self._outputs.setMinimumHeight(140)
        self._outputs.itemSelectionChanged.connect(self._update_enabled)
        self._outputs.itemDoubleClicked.connect(lambda _item: self.open_selected())
        outer.addWidget(self._outputs)

        row = QHBoxLayout()
        self._open_button = QPushButton("Open in viewer")
        self._open_button.setToolTip(
            "TIFFs open as layers — masks and the segmentation as labels, images named "
            "after their stain. PNGs open as pictures. Tables and .h5ad files open in "
            "the program the system gives them."
        )
        self._open_button.clicked.connect(self.open_selected)
        row.addWidget(self._open_button)
        folder = QPushButton("Show folder")
        folder.clicked.connect(self._show_output_folder)
        row.addWidget(folder)
        row.addStretch(1)
        outer.addLayout(row)
        return box

    def _build_run_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        self._run_button = QPushButton("Run ticked steps")
        self._run_button.setToolTip("Every ticked step, in order, for every selected sample.")
        self._run_button.clicked.connect(lambda: self.run(only_selected=False))
        row.addWidget(self._run_button)
        self._run_one_button = QPushButton("Run selected step")
        self._run_one_button.setToolTip(
            "Only the highlighted step, for every selected sample — to rerun one stage "
            "with new settings without redoing the ones before it."
        )
        self._run_one_button.clicked.connect(lambda: self.run(only_selected=True))
        row.addWidget(self._run_one_button)
        self._stop_button = QPushButton("Stop")
        self._stop_button.setToolTip(
            "Ends the run now, worker processes included. What a step had half-written stays on disk."
        )
        self._stop_button.clicked.connect(self.stop)
        row.addWidget(self._stop_button)
        row.addStretch(1)
        return row

    # -- persistence ------------------------------------------------------------

    def _fields(self) -> dict[str, QLineEdit]:
        return {
            "python": self._python,
            "source": self._source,
            "input": self._input,
            "metadata": self._metadata,
            "output": self._output,
        }

    def _restore(self) -> None:
        for key, edit in self._fields().items():
            value = self._settings.value(key, "")
            if value:
                edit.setText(str(value))

    def _remember(self) -> None:
        for key, edit in self._fields().items():
            self._settings.setValue(key, edit.text().strip())

    # -- the config ---------------------------------------------------------------

    def _config_to_form(self) -> None:
        c = self._config
        self._syncing = True
        try:
            self._ref_cycle.setValue(int(c["reference_cycle"]))
            self._nuclei_channel.setValue(int(c["channel_nuclei"]))
            self._nuclei_stain.setText(str(c["stain_nuclei"]))
            self._stain_channels.setText(_ints_text(c["channel_stains"]))
            self._align_refs.setText(_ints_text(c["alignment"]["reference_cycles"]))
            self._processes.setValue(int(c["cropping_denoising"]["n_processes"]))
            self._model.setCurrentText(str(c["cellpose"]["model"]))
            self._diameter.setValue(int(c["cellpose"]["diameter"] or 0))
            self._flow.setValue(float(c["cellpose"]["flow_threshold"]))
            self._cellprob.setValue(float(c["cellpose"]["cellprob_threshold"]))
            self._stat.setCurrentText(str(c["analysis_pipeline"]["stat"]))
            for index in range(self._steps.topLevelItemCount()):
                item = self._steps.topLevelItem(index)
                on = ph.is_enabled(c, ph.get_step(item.data(0, Qt.UserRole)))
                item.setCheckState(0, Qt.Checked if on else Qt.Unchecked)
        finally:
            self._syncing = False

    def _form_to_config(self, *_args) -> None:
        if self._syncing:
            return
        c = self._config
        c["reference_cycle"] = self._ref_cycle.value()
        c["channel_nuclei"] = self._nuclei_channel.value()
        c["stain_nuclei"] = self._nuclei_stain.text().strip() or "DAPI"
        c["channel_stains"] = _ints(self._stain_channels.text())
        c["alignment"]["reference_cycles"] = _ints(self._align_refs.text()) or None
        c["cropping_denoising"]["n_processes"] = self._processes.value()
        c["cellpose"]["model"] = self._model.currentText().strip() or "cyto2"
        c["cellpose"]["diameter"] = self._diameter.value()
        c["cellpose"]["flow_threshold"] = self._flow.value()
        c["cellpose"]["cellprob_threshold"] = self._cellprob.value()
        c["analysis_pipeline"]["stat"] = self._stat.currentText()
        # The original keeps these two in step with the reference cycle by hand.
        c["refined_masking"]["cycle_dapi"] = c["reference_cycle"]
        c["bg_subtraction"]["cycle_dapi"] = c["reference_cycle"]
        self._update_sample_note()

    def _step_toggled(self, item: QTreeWidgetItem, column: int) -> None:
        if self._syncing or column != 0:
            return
        step = ph.get_step(item.data(0, Qt.UserRole))
        ph.set_enabled(self._config, step, item.checkState(0) == Qt.Checked)
        self._update_enabled()

    def _load_config(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Phenoscapes config", "", "YAML (*.yaml *.yml)")
        if not path:
            return
        try:
            self._config = ph.load_config(path)
        except Exception as exc:  # noqa: BLE001 - shown
            QMessageBox.warning(self, "Phenoscapes config", f"Could not read {path}: {exc}")
            return
        if self._config.get("metadata") and not self._metadata.text().strip():
            self._metadata.setText(str(self._config["metadata"]))
        self._config_to_form()
        self._update_enabled()
        self._status.setText(f"Config loaded from {path}.")

    def _save_config(self) -> None:
        self._form_to_config()
        path, _ = QFileDialog.getSaveFileName(self, "Phenoscapes config", "config.yaml", "YAML (*.yaml)")
        if path:
            config = dict(self._config, metadata=self._metadata.text().strip() or None)
            ph.save_config(config, path)
            self._status.setText(f"Config written to {path}.")

    def _edit_config(self) -> None:
        self._form_to_config()
        dialog = _ConfigDialog(ph.config_text(self._config), self)
        if dialog.exec_() and dialog.config is not None:
            self._config = dialog.config
            self._config_to_form()
            self._update_enabled()

    def _reset_config(self) -> None:
        self._config = ph.default_config()
        self._config_to_form()
        self._update_enabled()

    # -- browsing -------------------------------------------------------------------

    def _browse_python(self) -> None:
        pattern = "Python (python.exe pythonw.exe)" if os.name == "nt" else "Python (python*)"
        path, _ = QFileDialog.getOpenFileName(self, "Python with phenoscapes", "", pattern)
        if path:
            self._python.setText(path)
            self._env_status.setText("Not checked.")
            self._remember()

    def _browse_source(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Folder holding the phenoscapes package")
        if path:
            if Path(path).name == "phenoscapes" and (Path(path) / "cli.py").is_file():
                # The package folder itself was picked; what goes on the path is its parent.
                path = str(Path(path).parent)
            self._source.setText(path)
            self._env_status.setText("Not checked.")
            self._remember()

    def _browse_input(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Folder holding 'stitched'")
        if path:
            if Path(path).name == "stitched":
                path = str(Path(path).parent)
            self._input.setText(path)
            if not self._output.text().strip():
                self._output.setText(str(Path(path) / "processed"))
            self._refresh_samples()
            self._remember()

    def _browse_metadata(self) -> None:
        start = self._input.text().strip()
        path, _ = QFileDialog.getOpenFileName(self, "Stain table", start, "CSV (*.csv)")
        if path:
            self._metadata.setText(path)
            self._remember()
            self._update_enabled()

    def _browse_output(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Output folder", self._output.text().strip())
        if path:
            self._output.setText(path)
            self._remember()
            self._refresh_outputs()

    # -- samples and results --------------------------------------------------------

    def selected_samples(self) -> list[str]:
        return [item.text() for item in self._samples.selectedItems()]

    def _refresh_samples(self) -> None:
        before = set(self.selected_samples())
        samples = ph.list_samples(self._input.text().strip())
        self._samples.blockSignals(True)
        self._samples.clear()
        for name in samples:
            item = QListWidgetItem(name)
            self._samples.addItem(item)
            item.setSelected(name in before or len(samples) == 1)
        self._samples.blockSignals(False)

        current = self._output_sample.currentText()
        self._output_sample.blockSignals(True)
        self._output_sample.clear()
        self._output_sample.addItems(samples)
        if current in samples:
            self._output_sample.setCurrentText(current)
        self._output_sample.blockSignals(False)
        self._samples_changed()
        self._refresh_outputs()

    def _samples_changed(self) -> None:
        chosen = self.selected_samples()
        if len(chosen) == 1 and self._output_sample.findText(chosen[0]) >= 0:
            self._output_sample.setCurrentText(chosen[0])
        self._update_sample_note()
        self._update_enabled()

    def _update_sample_note(self) -> None:
        folder = self._input.text().strip()
        if not folder:
            self._sample_note.setText("Choose the input folder.")
            return
        if not ph.stitched_dir(folder).is_dir():
            self._sample_note.setText(f"No 'stitched' folder in {folder}.")
            return
        if self._samples.count() == 0:
            self._sample_note.setText("'stitched' holds no sample folders with TIFFs.")
            return
        chosen = self.selected_samples()
        if not chosen:
            self._sample_note.setText(f"{self._samples.count()} sample(s). Select the ones to run.")
            return
        total, readable = ph.count_images(folder, chosen[0], self._config.get("regex"))
        text = f"{chosen[0]}: {total} TIFF(s), {readable} with a cycle and channel in the name."
        if readable < total:
            text += (" The rest are ignored by every step — set 'regex' under Every key… if "
                     "the names do not follow the stitching's pattern.")
        self._sample_note.setText(text)

    def _refresh_outputs(self, *_args) -> None:
        self._outputs.clear()
        sample = self._output_sample.currentText()
        out = self._output.text().strip()
        if not sample or not out or not Path(out).is_dir():
            self._update_enabled()
            return
        stains = ph.stain_names(self._metadata.text().strip())
        for output in ph.sample_outputs(out, sample, self._config):
            label = output.name
            if output.kind == "image":
                key = ph.channel_of(output.path, self._config.get("regex"))
                if key is not None:
                    stain = self._stain_for(key, stains)
                    label = f"{ph.get_step(output.step).title}: cycle {key[0]} · {stain}"
            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, output)
            item.setToolTip(str(output.path))
            self._outputs.addItem(item)
        self._update_enabled()

    def _stain_for(self, key: tuple[int, int], stains: dict) -> str:
        if key[1] == self._config.get("channel_nuclei"):
            return str(self._config.get("stain_nuclei") or "nuclei")
        return stains.get(key, f"channel {key[1]} (background)")

    def _show_output_folder(self) -> None:
        out = self._output.text().strip()
        if out and Path(out).is_dir():
            QDesktopServices.openUrl(QUrl.fromLocalFile(out))

    def open_selected(self) -> None:
        """Open the highlighted results: TIFFs and PNGs as layers, the rest outside."""
        items = self._outputs.selectedItems()
        if not items:
            return
        from qtpy.QtWidgets import QApplication

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            for item in items:
                output: ph.Output = item.data(Qt.UserRole)
                try:
                    self._open(output, item.text())
                except Exception as exc:  # noqa: BLE001 - one bad file, not the rest
                    logger.exception("could not open %s", output.path)
                    self._status.setText(f"Could not open {output.path.name}: {exc}")
        finally:
            QApplication.restoreOverrideCursor()

    def _open(self, output: ph.Output, title: str) -> None:
        if output.kind in ("table", "anndata"):
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(output.path)))
            return
        sample = self._output_sample.currentText()
        name = f"{sample} · {title.split(': ', 1)[-1]}"
        if output.kind == "picture":
            from skimage import io

            self._viewer.add_image(io.imread(str(output.path)), name=name, rgb=True)
            return
        import tifffile

        data = tifffile.imread(str(output.path))
        if output.kind == "labels":
            self._viewer.add_labels(data.astype("uint32", copy=False), name=name)
        else:
            self._viewer.add_image(data, name=name, blending="additive")

    # -- running ----------------------------------------------------------------------

    def _ticked(self) -> list[str]:
        return ph.enabled_steps(self._config)

    def _highlighted(self) -> str | None:
        items = self._steps.selectedItems()
        return items[0].data(0, Qt.UserRole) if items else None

    def _running(self) -> bool:
        return self._process is not None and self._process.state() != QProcess.NotRunning

    def _update_enabled(self, *_args) -> None:
        running = self._running()
        ready = bool(self.selected_samples()) and bool(self._metadata.text().strip())
        self._run_button.setEnabled(not running and ready and bool(self._ticked()))
        highlighted = self._highlighted()
        self._run_one_button.setEnabled(
            not running and ready and highlighted is not None
            and ph.is_enabled(self._config, ph.get_step(highlighted))
        )
        if highlighted is not None and not ph.is_enabled(self._config, ph.get_step(highlighted)):
            self._run_one_button.setToolTip("Tick the step first: unticked, it is not part of the pipeline.")
        else:
            self._run_one_button.setToolTip(
                "Only the highlighted step, for every selected sample — to rerun one stage "
                "with new settings without redoing the ones before it."
            )
        self._stop_button.setEnabled(running and self._mode == "run")
        self._check_button.setEnabled(not running)
        self._open_button.setEnabled(bool(self._outputs.selectedItems()))

    def _start(self, argv: list[str], mode: str) -> None:
        process = QProcess(self)
        env = QProcessEnvironment()
        for key, value in ph.child_environment(self._source.text().strip()).items():
            env.insert(key, value)
        process.setProcessEnvironment(env)
        process.setProcessChannelMode(QProcess.MergedChannels)
        process.readyReadStandardOutput.connect(self._read_output)
        process.finished.connect(self._finished)
        process.errorOccurred.connect(self._process_error)
        self._process = process
        self._mode = mode
        self._buffer = ""
        self._check_output = []
        process.start(argv[0], argv[1:])
        self._update_enabled()

    def check_environment(self) -> None:
        """Ask the chosen Python whether phenoscapes and its dependencies import."""
        python = self._python.text().strip()
        if not python:
            return
        self._remember()
        self._env_status.setText("Checking… (importing cellpose and itk takes a while)")
        self._start(ph.check_command(python), "check")

    def download_source(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Where to put the phenoscapes package")
        if not folder:
            return
        from qtpy.QtWidgets import QApplication

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            ph.download_source(folder)
        except Exception as exc:  # noqa: BLE001 - shown
            QMessageBox.warning(self, "Phenoscapes", f"Download failed: {exc}")
            return
        finally:
            QApplication.restoreOverrideCursor()
        self._source.setText(folder)
        self._remember()
        requirements = Path(folder) / "phenoscapes" / "requirements.txt"
        self._env_status.setText(
            f"Downloaded to {Path(folder) / 'phenoscapes'}. Install its requirements in the "
            f"Python above — pip install -r \"{requirements}\" — then press Check."
        )

    def run(self, only_selected: bool = False) -> None:
        if self._running():
            return
        self._form_to_config()
        if only_selected:
            highlighted = self._highlighted()
            steps = [highlighted] if highlighted else []
        else:
            steps = self._ticked()
        request = ph.Request(
            samples=tuple(self.selected_samples()),
            dir_input=self._input.text().strip(),
            dir_output=self._output.text().strip(),
            metadata_file=self._metadata.text().strip(),
            config=ph.complete_config(self._config),
            steps=tuple(steps),
        )
        problems = request.problems()
        if problems:
            QMessageBox.warning(self, "Phenoscapes", "\n".join(problems))
            return
        self._remember()
        request_file = ph.write_request(request, request.dir_output)

        for index in range(self._steps.topLevelItemCount()):
            item = self._steps.topLevelItem(index)
            item.setText(STATUS_COLUMN, WAITING if item.data(0, Qt.UserRole) in steps else "")
        self._log.clear()
        self._failed = []
        self._status.setText(
            f"Running {len(steps)} step(s) on {len(request.samples)} sample(s). The log is "
            f"also written to {Path(request.dir_output) / 'logs'}."
        )
        self._start(ph.run_command(self._python.text().strip(), request_file), "run")

    def stop(self) -> None:
        if not self._running():
            return
        pid = int(self._process.processId())
        if os.name == "nt" and pid:
            # The crop-and-denoise pool is a set of grandchildren; killing the
            # runner alone leaves them working, holding files open.
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(pid)],
                           capture_output=True, check=False)
        else:
            self._process.kill()
        self._status.setText("Stopped.")

    def _read_output(self) -> None:
        if self._process is None:
            return
        chunk = bytes(self._process.readAllStandardOutput()).decode("utf-8", errors="replace")
        self._buffer += chunk.replace("\r\n", "\n")
        *lines, self._buffer = self._buffer.split("\n")
        for line in lines:
            self._handle_line(line.rstrip("\r"))

    def _handle_line(self, line: str) -> None:
        if self._mode == "check":
            self._check_output.append(line)
            return
        parsed = ph.parse_mark(line)
        if parsed is None:
            # tqdm redraws with carriage returns; the last state is the one worth keeping.
            self._log.appendPlainText(line.split("\r")[-1])
            return
        kind, fields = parsed
        if kind == "step-start":
            self._set_step_status(fields[1], f"{RUNNING} ({fields[0]})")
            self._status.setText(f"{fields[0]}: {ph.get_step(fields[1]).title}…")
        elif kind == "step-done":
            self._set_step_status(fields[1], f"{DONE} ({fields[0]}, {fields[2]} s)")
        elif kind == "step-failed":
            self._set_step_status(fields[1], f"{FAILED} ({fields[0]})")
            self._failed.append(f"{fields[0]} · {ph.get_step(fields[1]).title}: {fields[2]}")
            self._log.appendPlainText(f"!! {fields[0]}: {fields[1]} failed — {fields[2]}")
        elif kind == "sample-done":
            self._refresh_outputs()

    def _set_step_status(self, key: str, text: str) -> None:
        for index in range(self._steps.topLevelItemCount()):
            item = self._steps.topLevelItem(index)
            if item.data(0, Qt.UserRole) == key:
                item.setText(STATUS_COLUMN, text)

    def _finished(self, code: int, _status=None) -> None:
        if self._buffer:
            self._handle_line(self._buffer)
            self._buffer = ""
        mode, self._mode = self._mode, ""
        if mode == "check":
            ok, missing, version = ph.parse_check("\n".join(self._check_output))
            if not ok and not missing:
                self._env_status.setText(
                    "That Python did not answer. Is it a python executable?\n"
                    + "\n".join(self._check_output[-5:])
                )
            elif missing:
                self._env_status.setText(
                    f"Python {version}. Missing: " + "; ".join(missing)
                    + ". Install phenoscapes' requirements.txt in it, or set Code to the "
                    "folder holding the phenoscapes package."
                )
            else:
                self._env_status.setText(f"Python {version}: phenoscapes and everything it needs import.")
        else:
            if self._failed:
                self._status.setText("Finished with failures: " + "; ".join(self._failed))
            elif code == 0:
                self._status.setText("Finished.")
            elif not self._status.text().startswith("Stopped"):
                self._status.setText(f"The run ended with code {code}; the log says why.")
            for index in range(self._steps.topLevelItemCount()):
                item = self._steps.topLevelItem(index)
                if item.text(STATUS_COLUMN) in (WAITING,) or item.text(STATUS_COLUMN).startswith(RUNNING):
                    item.setText(STATUS_COLUMN, "not run")
            self._refresh_outputs()
        self._update_enabled()

    def _process_error(self, error) -> None:
        if error == QProcess.FailedToStart:
            target = self._env_status if self._mode == "check" else self._status
            target.setText(f"Could not start {self._python.text().strip()!r}. Is the path right?")
            self._mode = ""
            self._update_enabled()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt's name
        self.stop()
        super().closeEvent(event)
