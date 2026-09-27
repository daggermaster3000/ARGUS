"""Dialog: maximum-project a whole folder of stacks to TIFF or Imaris files.

Shaped like :mod:`microscopy_viewer.widgets.slide_dialog` — pick inputs, pick
where the output goes, watch it run, stop it if it is the wrong thing. Nothing is
added to the viewer; the work happens in
:mod:`microscopy_viewer.projection` on a ``thread_worker``.

The channel list is read from the files rather than typed, and the boxes are
ticked by name. That is what makes one run work across a folder whose
acquisitions put the channels in different orders.
"""

from __future__ import annotations

from pathlib import Path

from qtpy.QtCore import QObject, Qt, Signal
from qtpy.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .. import projection as pj
from ..utils import get_logger

logger = get_logger("projection_dialog")

#: Files read when looking for channel names. Enough to see the whole set
#: without opening thirty stacks to fill in a list of tick boxes.
NAME_SAMPLE = 3


class _Relay(QObject):
    """Carries worker-thread text onto the GUI thread."""

    message = Signal(str)
    channels = Signal(object)


class ProjectionDialog(QDialog):
    """Pick a folder, pick channels, write a maximum projection of each file."""

    def __init__(self, start_directory=None, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Batch maximum projection")
        self.setMinimumWidth(560)

        self._paths: list[Path] = []
        #: Overview files the last scan left out.
        self.skipped: list[Path] = []
        self._worker = None
        self._cancelled = False
        self.outcomes: list[pj.ProjectionOutcome] = []
        self._start_directory = str(start_directory or Path.home())

        self._relay = _Relay()
        self._relay.message.connect(self._log)
        self._relay.channels.connect(self._fill_channels)

        layout = QVBoxLayout(self)
        layout.addWidget(self._build_input_box())
        layout.addWidget(self._build_channel_box(), stretch=1)
        layout.addWidget(self._build_output_box())
        self._update_structure()

        self._log_view = QPlainTextEdit()
        self._log_view.setReadOnly(True)
        self._log_view.setMaximumHeight(120)
        self._log_view.setPlaceholderText("Progress appears here.")
        layout.addWidget(self._log_view)

        self._status = QLabel("Choose the folder holding the stacks to project.")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        self._buttons = QDialogButtonBox()
        self._run_button = self._buttons.addButton("Project", QDialogButtonBox.AcceptRole)
        self._close_button = self._buttons.addButton("Close", QDialogButtonBox.RejectRole)
        self._run_button.clicked.connect(self.run)
        self._close_button.clicked.connect(self._on_close)
        layout.addWidget(self._buttons)

    # -- construction ---------------------------------------------------------

    def _build_input_box(self) -> QGroupBox:
        box = QGroupBox("Stacks to project")
        outer = QVBoxLayout(box)

        row = QHBoxLayout()
        self._input_edit = QLineEdit()
        self._input_edit.setPlaceholderText("No folder chosen")
        self._input_edit.setReadOnly(True)
        row.addWidget(self._input_edit, stretch=1)
        choose = QPushButton("Choose…")
        choose.clicked.connect(self.choose_input)
        row.addWidget(choose)
        outer.addLayout(row)

        self._recursive = QCheckBox("Include subfolders")
        self._recursive.setChecked(True)
        self._recursive.toggled.connect(lambda _on: self.rescan())
        outer.addWidget(self._recursive)

        self._skip_overviews = QCheckBox("Skip overviews")
        self._skip_overviews.setChecked(True)
        self._skip_overviews.setToolTip(
            "Leave out the overview: the fields of an _F#### mosaic and single-plane "
            "maps the samples were taken from. They are one plane already, so a "
            "projection would only copy them. Found as the Experiment setup panel "
            "finds them, from each file's shape and stage position."
        )
        self._skip_overviews.toggled.connect(lambda _on: self.rescan())
        outer.addWidget(self._skip_overviews)
        return box

    def _build_channel_box(self) -> QGroupBox:
        box = QGroupBox("Channels")
        outer = QVBoxLayout(box)

        self._all_channels = QCheckBox("Every channel")
        self._all_channels.setChecked(True)
        self._all_channels.setToolTip(
            "Project every channel each file has. Untick to choose by name."
        )
        self._all_channels.toggled.connect(self._on_all_toggled)
        outer.addWidget(self._all_channels)

        self._channel_list = QListWidget()
        self._channel_list.setSelectionMode(QAbstractItemView.NoSelection)
        self._channel_list.setMaximumHeight(110)
        self._channel_list.setEnabled(False)
        self._channel_list.setToolTip(
            "Read from the first few files in the folder. Channels are matched by "
            "name, not by position, so one run works across acquisitions that put "
            "them in different orders.\n"
            "A name that matches nothing in a given file is reported against that "
            "file rather than passing unnoticed."
        )
        outer.addWidget(self._channel_list)
        return box

    def _build_output_box(self) -> QGroupBox:
        box = QGroupBox("Output")
        form = QFormLayout(box)
        form.setLabelAlignment(Qt.AlignRight)

        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        self._output_edit = QLineEdit()
        self._output_edit.setPlaceholderText("Beside each source file")
        self._output_edit.setReadOnly(True)
        row_layout.addWidget(self._output_edit, stretch=1)
        choose = QPushButton("Choose…")
        choose.clicked.connect(self.choose_output)
        row_layout.addWidget(choose)
        clear = QPushButton("×")
        clear.setMaximumWidth(28)
        clear.setToolTip("Write beside each source file instead.")
        clear.clicked.connect(lambda: self._output_edit.clear())
        row_layout.addWidget(clear)
        form.addRow("Folder", row)

        self._keep_structure = QCheckBox("Keep the subfolder structure")
        self._keep_structure.setChecked(True)
        self._keep_structure.setToolTip(
            "On: each projection goes into the same subfolder of the output folder as "
            "its stack sits in, e.g. MIP/DMSO/fish1_MIP.tif.\n"
            "Off: every projection in the output folder itself. Files that share a "
            "name then get their folder added, e.g. fish1_DMSO_MIP.tif."
        )
        self._output_edit.textChanged.connect(self._update_structure)
        form.addRow("Layout", self._keep_structure)

        self._format_box = QComboBox()
        for suffix in pj.OUTPUT_FORMATS:
            self._format_box.addItem(
                {".tif": "TIFF (.tif) — opens anywhere",
                 ".ims": "Imaris (.ims) — same format as the source"}.get(suffix, suffix),
                userData=suffix,
            )
        self._format_box.setToolTip(
            "TIFF is ImageJ-flavoured, so Fiji opens it as a calibrated hyperstack "
            "with the channels separated.\n"
            "Imaris keeps the projection in the same format as the stack it came "
            "from, carrying the channel names, colours and stage position with it."
        )
        form.addRow("Format", self._format_box)

        self._suffix_edit = QLineEdit(pj.DEFAULT_SUFFIX)
        self._suffix_edit.setToolTip(
            "Added to each file's name. It is what keeps a projection written "
            "beside its source from landing on top of it."
        )
        form.addRow("Name suffix", self._suffix_edit)

        self._overwrite = QCheckBox("Overwrite projections that already exist")
        self._overwrite.setToolTip(
            "Off by default, so re-running after adding files to a folder costs "
            "nothing and cannot destroy a projection you have since edited."
        )
        form.addRow("Existing", self._overwrite)
        return box

    # -- inputs ---------------------------------------------------------------

    def choose_input(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "Folder of stacks to project", self._start_directory
        )
        if not folder:
            return
        self._input_edit.setText(folder)
        self._start_directory = folder
        if not self._output_edit.text().strip():
            self._output_edit.setText(str(Path(folder) / "MIP"))
        self.rescan()

    def choose_output(self) -> None:
        start = self._output_edit.text().strip() or self._start_directory
        folder = QFileDialog.getExistingDirectory(self, "Where to write the projections", start)
        if folder:
            self._output_edit.setText(folder)

    def _update_structure(self, *_args) -> None:
        # Beside each source there is no structure to keep: it is the source's own.
        self._keep_structure.setEnabled(bool(self._output_edit.text().strip()))

    def rescan(self) -> None:
        """List the folder, drop overviews, and read channel names from the first few files."""
        from .. import experiment as ex

        folder = self._input_edit.text().strip()
        if not folder:
            return
        paths = ex.list_files(folder, self._recursive.isChecked())
        # Never the projections of an earlier run, when they were written inside
        # the folder being projected.
        output = self._output_edit.text().strip()
        if output:
            out = Path(output).resolve()
            paths = [p for p in paths if out not in Path(p).resolve().parents]
        self._paths = []
        self.skipped = []
        self._channel_list.clear()
        if not paths:
            self._status.setText(f"No readable stacks in {Path(folder).name}.")
            return
        skip = self._skip_overviews.isChecked()
        self._status.setText(
            f"{len(paths)} file(s) found. "
            + ("Looking for overviews and reading channel names…" if skip
               else "Reading channel names…")
        )

        from napari.qt.threading import thread_worker

        relay = self._relay

        @thread_worker
        def _read():
            stacks, skipped = pj.without_overviews(paths) if skip else (list(paths), [])
            return stacks, skipped, pj.channel_names(stacks[:NAME_SAMPLE], limit=NAME_SAMPLE)

        worker = _read()
        worker.returned.connect(relay.channels.emit)
        worker.errored.connect(self._on_error)
        worker.start()

    def _fill_channels(self, found) -> None:
        stacks, skipped, names = found
        self._paths = list(stacks)
        self.skipped = list(skipped)
        self._channel_list.clear()
        for name in names or ():
            item = QListWidgetItem(str(name))
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            self._channel_list.addItem(item)
        text = (f"{len(self._paths)} stack(s), {self._channel_list.count()} channel(s): "
                + ", ".join(str(name) for name in (names or ())))
        if skipped:
            shown = ", ".join(Path(p).name for p in skipped[:3])
            text += (f". {len(skipped)} overview file(s) skipped: {shown}"
                     + ("…" if len(skipped) > 3 else ""))
        self._status.setText(text)

    def _on_all_toggled(self, checked: bool) -> None:
        self._channel_list.setEnabled(not checked)

    def chosen_channels(self) -> tuple[str, ...]:
        """The ticked channel names, or empty for every channel."""
        if self._all_channels.isChecked():
            return ()
        return tuple(
            self._channel_list.item(row).text()
            for row in range(self._channel_list.count())
            if self._channel_list.item(row).checkState() == Qt.Checked
        )

    def options(self) -> pj.ProjectionOptions:
        folder = self._output_edit.text().strip()
        source = self._input_edit.text().strip()
        return pj.ProjectionOptions(
            input_root=Path(source) if source else None,
            keep_structure=self._keep_structure.isChecked(),
            output_dir=Path(folder) if folder else None,
            channels=self.chosen_channels(),
            fmt=str(self._format_box.currentData() or ".tif"),
            suffix=self._suffix_edit.text().strip(),
            overwrite=self._overwrite.isChecked(),
        )

    # -- running --------------------------------------------------------------

    def _log(self, text: str) -> None:
        self._log_view.appendPlainText(text)
        self._status.setText(text)

    def _on_error(self, exc) -> None:
        logger.exception("batch projection failed: %s", exc)
        self._log(f"Failed: {exc}")
        self._finish_run()

    def run(self) -> None:
        if self._worker is not None:  # the button is Stop while a run is going
            self._cancelled = True
            self._log("Stopping after the current file…")
            return
        if not self._paths:
            self._status.setText("Choose a folder of stacks first.")
            return

        options = self.options()
        if not options.suffix and options.output_dir is None:
            self._status.setText(
                "With no suffix and no output folder a projection would overwrite its "
                "own source. Give it one or the other."
            )
            return
        if not self._all_channels.isChecked() and not options.channels:
            self._status.setText("No channels ticked.")
            return

        self._log_view.clear()
        if self.skipped:
            self._log_view.appendPlainText(
                f"Skipping {len(self.skipped)} overview file(s): "
                + ", ".join(Path(p).name for p in self.skipped))
        self._cancelled = False
        self.outcomes = []
        self._run_button.setText("Stop")
        self._close_button.setEnabled(False)
        chosen = ", ".join(options.channels) if options.channels else "every channel"
        self._log(f"Projecting {len(self._paths)} file(s), {chosen}, to {options.fmt}…")

        from napari.qt.threading import thread_worker

        relay = self._relay
        paths = list(self._paths)

        @thread_worker
        def _run():
            return pj.run_batch(
                paths,
                options,
                progress=relay.message.emit,
                should_cancel=lambda: self._cancelled,
            )

        worker = _run()
        worker.returned.connect(self._on_done)
        worker.errored.connect(self._on_error)
        worker.finished.connect(self._finish_run)
        self._worker = worker
        worker.start()

    def _finish_run(self) -> None:
        self._worker = None
        self._run_button.setText("Project")
        self._close_button.setEnabled(True)

    def _on_done(self, outcomes) -> None:
        self.outcomes = list(outcomes)
        summary = pj.summarise(self.outcomes)
        if self._cancelled:
            summary += " Stopped early."
        self._log(summary)
        logger.info("batch projection: %s", summary)

    def _on_close(self) -> None:
        if self._worker is not None:
            self._cancelled = True
            self._log("Stopping after the current file…")
            return
        self.reject()
