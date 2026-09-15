"""The two plate controls that more than one panel needs.

Choosing a plate and choosing wells out of it were written into the Batch
segmentation panel first, and then wanted again by the File explorer. Two copies
of a well list is two lists that can disagree about what is in the plate, so they
live here and both panels use the same ones.

:class:`PlateSourceBox` is the only place a plate is chosen. It scans, and emits
the survey; whoever is listening decides what to do with it.

:class:`WellAcquisitionPicker` is the wells-and-cycles pair of lists, and knows
nothing about what the selection is for — the explorer opens it, the batch panel
segments it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .. import batch
from ..utils import get_logger

logger = get_logger("plate_picker")

#: Suffixes that mark a folder as a Zarr store, when walking up from a layer's path.
STORE_SUFFIXES = (".zarr", ".ngff")


class PlateSourceBox(QGroupBox):
    """Choose a ``.zarr`` plate, scan it, and hand out the survey.

    Scanning reads metadata only — no pixels — so it is quick even on a plate of
    a few hundred gigabytes, and it is what fills every list downstream of it.
    """

    #: A plate was scanned. Carries the :class:`~microscopy_viewer.batch.PlateSurvey`.
    surveyed = Signal(object)
    #: A plate was chosen and could not be read. Carries the reason.
    failed = Signal(str)

    def __init__(self, app, title: str = "Plate", parent: QWidget | None = None):
        super().__init__(title, parent)
        self._app = app
        self._survey: batch.PlateSurvey | None = None

        form = QFormLayout(self)
        form.setLabelAlignment(Qt.AlignRight)

        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        self._edit = QLineEdit()
        self._edit.setPlaceholderText("…/AssayPlate.zarr")
        self._edit.setToolTip(
            "The .zarr folder of an OME-Zarr plate — the one holding the row folders."
        )
        self._edit.returnPressed.connect(self.scan)
        row_layout.addWidget(self._edit, stretch=1)
        browse = QPushButton("…")
        browse.setMaximumWidth(30)
        browse.clicked.connect(self.browse)
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
        loaded.clicked.connect(self.use_loaded_plate)
        button_layout.addWidget(loaded)
        button_layout.addStretch(1)
        form.addRow("", buttons)

        self._summary = QLabel("No plate scanned yet.")
        self._summary.setWordWrap(True)
        form.addRow("Contents", self._summary)

    # -- state ----------------------------------------------------------------

    @property
    def survey(self) -> batch.PlateSurvey | None:
        return self._survey

    def path(self) -> str:
        return self._edit.text().strip().strip('"')

    def set_path(self, path: str | Path) -> None:
        self._edit.setText(str(path))

    # -- actions --------------------------------------------------------------

    def browse(self) -> None:
        start = self.path() or str(getattr(self._app, "last_directory", Path.home()))
        chosen = QFileDialog.getExistingDirectory(self, "Choose an OME-Zarr plate", start)
        if chosen:
            self._edit.setText(chosen)
            self.scan()

    def use_loaded_plate(self) -> None:
        """Take the plate path off a layer that is already open.

        The readers record the file the layer came from, so the plate on screen is
        usually the plate wanted — and it saves finding a deeply nested store in a
        file dialog for the second time.
        """
        for layer in reversed(list(self._app.viewer.layers)):
            meta = (getattr(layer, "metadata", {}) or {}).get("mv_metadata")
            candidate = getattr(meta, "file_path", None)
            if not candidate:
                continue
            path = Path(candidate)
            for parent in [path, *path.parents]:
                if parent.suffix.lower() in STORE_SUFFIXES:
                    self._edit.setText(str(parent))
                    self.scan()
                    return
        self.failed.emit("No open layer came from a Zarr store.")

    def scan(self) -> None:
        """Read the plate's metadata and emit the survey."""
        text = self.path()
        if not text:
            self.failed.emit("Choose the .zarr folder of a plate first.")
            return
        try:
            survey = batch.survey_plate(Path(text))
        except Exception as exc:  # noqa: BLE001 - shown, not raised
            logger.exception("plate scan failed")
            self._survey = None
            self._summary.setText("—")
            self.failed.emit(f"That is not a plate this can read: {exc}")
            return

        self._survey = survey
        self._summary.setText(
            f"{survey.name}: {len(survey.jobs)} image(s), {len(survey.wells)} well(s), "
            f"{len(survey.acquisitions)} acquisition(s)."
        )
        self.surveyed.emit(survey)


class WellAcquisitionPicker(QWidget):
    """Wells on the left, 4i cycles on the right, both multi-select.

    *all_cycles* decides what a fresh plate starts with. The explorer opens what
    is ticked, so starting with one cycle keeps a first click from building seven
    sets of layers; the batch panel segments what is ticked, and one cycle is
    right there too, for the same reason in reverse — every cycle images the same
    cells, so segmenting them all finds the same nuclei seven times.
    """

    selectionChanged = Signal()

    def __init__(
        self,
        all_cycles: bool = False,
        note: Callable[[batch.ImageJob], str] | None = None,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self._survey: batch.PlateSurvey | None = None
        self._all_cycles = bool(all_cycles)
        self._note = note

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        lists = QHBoxLayout()
        wells_column = QVBoxLayout()
        wells_column.addWidget(QLabel("Wells"))
        self._wells = QListWidget()
        self._wells.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self._wells.setToolTip("Ctrl-click and shift-click to choose. Everything by default.")
        self._wells.itemSelectionChanged.connect(self.selectionChanged)
        wells_column.addWidget(self._wells)
        lists.addLayout(wells_column, stretch=2)

        cycles_column = QVBoxLayout()
        cycles_column.addWidget(QLabel("Acquisitions"))
        self._cycles = QListWidget()
        self._cycles.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self._cycles.setToolTip(
            "The 4i cycles of the plate. The same cells are imaged in every one, so a "
            "cycle is a set of extra stains of the same objects rather than a new field."
        )
        self._cycles.itemSelectionChanged.connect(self.selectionChanged)
        cycles_column.addWidget(self._cycles)
        lists.addLayout(cycles_column, stretch=1)
        outer.addLayout(lists)

        buttons = QHBoxLayout()
        for text, handler in (
            ("All wells", self._wells.selectAll),
            ("No wells", self._wells.clearSelection),
            ("All cycles", self._cycles.selectAll),
        ):
            button = QPushButton(text)
            button.clicked.connect(handler)
            buttons.addWidget(button)
        buttons.addStretch(1)
        outer.addLayout(buttons)

    # -- state ----------------------------------------------------------------

    def set_survey(self, survey: batch.PlateSurvey | None) -> None:
        """Rebuild both lists for *survey*, keeping the well selection where it fits."""
        self._survey = survey
        self._refresh_wells()
        self._refresh_cycles()
        self.selectionChanged.emit()

    def set_note(self, note: Callable[[batch.ImageJob], str] | None) -> None:
        """Change the per-well annotation and redraw the well list.

        The batch panel marks the wells that already carry the label set it is
        about to write, which changes as soon as the label name is edited.
        """
        self._note = note
        self._refresh_wells()

    def _refresh_wells(self) -> None:
        previous = {item.data(Qt.UserRole) for item in self._wells.selectedItems()}
        self._wells.blockSignals(True)
        self._wells.clear()
        if self._survey is not None:
            for well in self._survey.wells:
                suffix = ""
                if self._note is not None:
                    jobs = [job for job in self._survey.jobs if job.well == well]
                    suffix = self._note(jobs[0]) if jobs else ""
                item = QListWidgetItem(f"{well}{suffix}")
                item.setData(Qt.UserRole, well)
                self._wells.addItem(item)
                item.setSelected(well in previous if previous else True)
        self._wells.blockSignals(False)

    def _refresh_cycles(self) -> None:
        self._cycles.blockSignals(True)
        self._cycles.clear()
        if self._survey is not None:
            acquisitions = self._survey.acquisitions
            for acquisition in acquisitions:
                text = "—" if acquisition is None else f"cycle {acquisition}"
                item = QListWidgetItem(text)
                item.setData(Qt.UserRole, acquisition)
                self._cycles.addItem(item)
                item.setSelected(self._all_cycles or acquisition == acquisitions[0])
        self._cycles.blockSignals(False)

    # -- what is chosen -------------------------------------------------------

    def selected_wells(self) -> list:
        return [item.data(Qt.UserRole) for item in self._wells.selectedItems()]

    def selected_acquisitions(self) -> list:
        return [item.data(Qt.UserRole) for item in self._cycles.selectedItems()]

    def selected_jobs(self) -> tuple[batch.ImageJob, ...]:
        """The images the current selection names, in plate order."""
        if self._survey is None:
            return ()
        wells = self.selected_wells()
        cycles = self.selected_acquisitions()
        return batch.select_jobs(
            self._survey,
            wells=wells if wells else None,
            acquisitions=cycles if cycles else None,
        )
