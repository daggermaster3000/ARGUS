"""Dock panel: a folder of samples, previewed as thumbnails and processed at once.

Two steps of one job:

1. **Pick the folder and look at it.** Every readable dataset gets a thumbnail
   read off a middle pyramid level, so a bad mount or an empty stub is visible
   before anything is opened. Nothing is added to the viewer by scanning.
2. **Put ROIs on the samples.** An outline drawn in the viewer — the same Shapes
   layer the *Brain regions* panel uses — is written into the ``.ims`` files
   themselves, either into the one sample it was drawn on or into every sample
   selected. Opening a sample brings back the outlines and label maps stored in
   it, and opening the next one clears them again.
Segmenting the selected samples lives in the *Segmentation* panel's Batch tab,
beside the settings it runs with; it reads its samples from this panel's
selection. Analysing them lives in the *Analysis* panel.

The preview scan runs on a ``thread_worker``, and refuses to start while a
batch is writing into the files.

**Why samples get closed.** HDF5 will not open a file for writing while it is
open for reading, so a file cannot be written to while the viewer is showing it.
The panel takes a sample's layers off screen and releases the reader's handle
before writing into it, and says so when it does.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from qtpy.QtCore import QObject, QSize, Qt, Signal
from qtpy.QtGui import QIcon, QImage, QPixmap
from qtpy.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .. import experiment as ex
from .. import ims_store
from .. import regions as rg
from ..user import credit
from ..utils import get_logger

logger = get_logger("experiment_widget")

#: Side of a thumbnail in the grid, in screen pixels.
ICON_SIZE = 128

#: Role the sample's path is stashed under on its list item.
PATH_ROLE = Qt.UserRole + 1

#: Layer metadata key naming the file a layer was read out of, for layers that
#: are not image channels (a stored label map has no acquisition metadata) but
#: still belong to that sample and have to go when it does.
from .sample_files import SAMPLE_KEY, clear_samples, sample_of  # noqa: E402


def _layer_source(layer) -> str:
    """The file a layer was read from or out of — not what a panel made from it.

    What closing a file *for writing* has to take off screen: the layers holding
    its handle. Panel-made layers (counts, lines) stay through a write.
    """
    return sample_of(layer, derived=False)


class _Relay(QObject):
    """Carries worker-thread text and images onto the GUI thread.

    Same reason as the segmentation panel's: a widget may only be touched from
    the thread that owns it, and a signal is the supported way across.
    """

    message = Signal(str)
    preview = Signal(int, object)


class ExperimentWidget(QWidget):
    """Browse a folder of samples, put ROIs on them, segment all of them."""

    #: The samples :meth:`selected_entries` returns may have changed: a new scan,
    #: or a different selection.
    samples_changed = Signal()
    #: A thumbnail arrived for the file at this path.
    preview_ready = Signal(str)

    def __init__(self, app, parent: QWidget | None = None):
        super().__init__(parent)
        self._app = app
        self._viewer = app.viewer
        self._entries: list[ex.SampleEntry] = []
        #: Overviews found by the last scan, and the files they are made of: those
        #: sit at the end of the grid, marked, and out of "all samples".
        self._overviews: list = []
        self._overview_ids: set[int] = set()
        self._overview_window = None
        self._picker_window = None
        self._picker_paths: set[str] = set()
        self._item_index: dict = {}
        #: Paths selected from a table before the scan reached them.
        self._pending_selection: set[str] = set()
        #: ``((folder, recursive), paths)`` of the last listing, for the picker.
        self._listing: tuple = (None, [])
        self._worker = None
        self._cancelled = False
        #: ``(selected, opened)`` paths waiting for the scan :meth:`restore` started.
        self._restoring = None
        self._relay = _Relay()
        self._relay.message.connect(self._log)
        self._relay.preview.connect(self._set_preview)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        layout.addWidget(self._build_folder_box())
        layout.addWidget(self._build_samples_box(), stretch=1)
        layout.addWidget(self._build_roi_box())
        layout.addWidget(self._build_log())

        self._status = QLabel("Choose the folder this experiment was acquired into.")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

    # -- construction ---------------------------------------------------------

    def _build_folder_box(self) -> QGroupBox:
        box = QGroupBox("Experiment folder")
        outer = QVBoxLayout(box)

        row = QHBoxLayout()
        self._folder_edit = QLineEdit()
        self._folder_edit.setPlaceholderText("No folder chosen")
        self._folder_edit.setReadOnly(True)
        row.addWidget(self._folder_edit, stretch=1)

        self._choose_button = QPushButton("Choose…")
        self._choose_button.clicked.connect(self.choose_folder)
        row.addWidget(self._choose_button)
        outer.addLayout(row)

        options = QHBoxLayout()
        self._recursive = QCheckBox("Include subfolders")
        self._recursive.setChecked(True)
        self._recursive.setToolTip(
            "Acquisitions are usually filed one folder per day, so the experiment is "
            "the parent of several. Off to take one folder literally."
        )
        options.addWidget(self._recursive)

        self._scan_button = QPushButton("Scan")
        self._scan_button.clicked.connect(self.scan)
        options.addWidget(self._scan_button)

        self._stop_button = QPushButton("Stop")
        self._stop_button.setEnabled(False)
        self._stop_button.clicked.connect(self.stop)
        options.addWidget(self._stop_button)
        options.addStretch(1)
        outer.addLayout(options)
        return box

    def _build_samples_box(self) -> QGroupBox:
        box = QGroupBox("Samples")
        outer = QVBoxLayout(box)

        self._grid = QListWidget()
        self._grid.setViewMode(QListWidget.IconMode)
        self._grid.setIconSize(QSize(ICON_SIZE, ICON_SIZE))
        self._grid.setGridSize(QSize(ICON_SIZE + 24, ICON_SIZE + 46))
        self._grid.setResizeMode(QListWidget.Adjust)
        self._grid.setMovement(QListWidget.Static)
        self._grid.setWordWrap(True)
        self._grid.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self._grid.itemSelectionChanged.connect(self.samples_changed)
        self._grid.itemDoubleClicked.connect(lambda _item: self.open_selected())
        outer.addWidget(self._grid)

        row = QHBoxLayout()
        for label, tooltip, handler in (
            ("Open", "Add the selected samples to the viewer (double-click does this too).",
             self.open_selected),
            ("Close", "Take the selected samples off screen and release their files, "
                      "which is what lets anything be written into them.", self.close_selected),
            ("Load labels", "Read a label map stored in the selected sample into the viewer, "
                            "so what a batch wrote can be looked at rather than taken on "
                            "trust. Opening a sample does this for all of them.",
             self.load_labels_from_sample),
            ("Select all", "", self._grid.selectAll),
        ):
            button = QPushButton(label)
            if tooltip:
                button.setToolTip(tooltip)
            button.clicked.connect(handler)
            row.addWidget(button)
        self._overview_button = QPushButton("Overview")
        self._overview_button.setToolTip(
            "The folder's overview with every sample outlined on it. Overviews — an "
            "_F#### mosaic, or a single low-magnification image the samples were "
            "taken from — are kept out of the sample grid."
        )
        self._overview_button.setEnabled(False)
        self._overview_button.clicked.connect(self.show_overview)
        row.addWidget(self._overview_button)
        pick = QPushButton("Pick from table…")
        pick.setToolTip(
            "Choose samples off a prescreen or QC table (CSV or Excel, a row per sample): "
            "keep some categories (Pass: yes), draw n of others (10 ripped, 10 folded, 10 "
            "of each shape), see them as thumbnails, select them here or open them one "
            "after the other."
        )
        pick.clicked.connect(self.show_picker)
        row.addWidget(pick)
        row.addStretch(1)
        outer.addLayout(row)
        return box

    def _build_roi_box(self) -> QGroupBox:
        box = QGroupBox("ROIs")
        outer = QVBoxLayout(box)

        note = QLabel(
            "Draw outlines on the “{}” layer — the Brain regions panel makes it — "
            "then write them into the files.".format(rg.REGION_LAYER_NAME)
        )
        note.setWordWrap(True)
        outer.addWidget(note)

        row = QHBoxLayout()
        to_selected = self._write_rois_button = QPushButton("Write to selected")
        to_selected.setToolTip(
            "Write the outlines now on screen into every selected sample's .ims file.\n"
            "The vertices are in micrometres from each image's own origin, so this is "
            "right when the samples are mounted and framed alike and wrong when they "
            "are not — for those, draw on each one and write to it alone."
        )
        to_selected.clicked.connect(self.write_rois_to_selected)
        row.addWidget(to_selected)

        load = QPushButton("Load from sample")
        load.setToolTip("Read the ROIs stored in the selected sample back onto the canvas.")
        load.clicked.connect(self.load_rois_from_sample)
        row.addWidget(load)
        row.addStretch(1)
        outer.addLayout(row)
        return box

    def _build_log(self) -> QPlainTextEdit:
        self._log_view = QPlainTextEdit()
        self._log_view.setReadOnly(True)
        self._log_view.setMaximumHeight(100)
        self._log_view.setPlaceholderText("Progress appears here.")
        return self._log_view

    # -- small helpers --------------------------------------------------------

    def _log(self, text: str) -> None:
        self._log_view.appendPlainText(text)
        self._status.setText(text)

    def _busy(self, running: bool) -> None:
        self._scan_button.setEnabled(not running)
        self._stop_button.setEnabled(running)

    def is_running(self) -> bool:
        return self._worker is not None

    def _batch_running(self) -> bool:
        panel = getattr(self._app, "segmentation_widget", None)
        batch = getattr(panel, "batch", None)
        return bool(batch is not None and batch.is_running())

    def folder(self) -> Path | None:
        """The experiment folder, if one is chosen."""
        text = self._folder_edit.text().strip()
        return Path(text) if text else None

    def stop(self) -> None:
        """Ask the running worker to stop after the file it is on."""
        self._cancelled = True
        self._log("Stopping after the current file…")

    def _should_cancel(self) -> bool:
        return self._cancelled

    def selected_entries(self) -> list[ex.SampleEntry]:
        """The selected samples, or all of them when nothing is selected.

        "All of them" leaves out the files taken for overviews; selecting one —
        a 10x image that only looked like an overview — includes it.
        """
        rows = [self._grid.row(item) for item in self._grid.selectedItems()]
        if not rows:
            return [e for e in self._entries if id(e) not in self._overview_ids]
        return [self._entries[row] for row in sorted(rows) if row < len(self._entries)]

    def selected_paths(self) -> list[str]:
        """The files selected in the grid — none when nothing is, unlike
        :meth:`selected_entries`."""
        return [str(item.data(PATH_ROLE)) for item in self._grid.selectedItems()]

    def restore(self, folder, selected=(), opened=()) -> None:
        """Come back to *folder* as it was left: scanned, with *selected* selected.

        *opened* are the files the viewer has already put back on screen; the
        label maps stored in them come up once the scan has found them, as they
        do when a sample is opened from here.
        """
        self._folder_edit.setText(str(folder))
        self._restoring = ([str(path) for path in selected], [str(path) for path in opened])
        self.scan()
        if self._worker is None:
            # Nothing to scan, so nothing will come back to finish this.
            self._restoring = None

    def _finish_restore(self) -> None:
        selected, opened = self._restoring
        self._restoring = None
        self._grid.clearSelection()
        first = None
        for row in range(self._grid.count()):
            item = self._grid.item(row)
            if str(item.data(PATH_ROLE)) in selected:
                item.setSelected(True)
                first = first or item
        if first is not None:
            self._grid.scrollToItem(first)
        entries = [entry for entry in self._entries
                   if str(entry.path) in opened and entry.readable]
        if entries:
            self._load_stored_labels(entries)
            regions = getattr(self._app, "regions_widget", None)
            if regions is not None:
                regions.raise_layer()

    # -- scanning -------------------------------------------------------------

    def choose_folder(self) -> None:
        start = str(self._app.last_directory or Path.home())
        folder = QFileDialog.getExistingDirectory(self, "Choose the experiment folder", start)
        if not folder:
            return
        self._folder_edit.setText(folder)
        self._app.last_directory = Path(folder)
        self.scan()

    def scan(self) -> None:
        """List the folder and draw a thumbnail for every sample in it."""
        folder = self._folder_edit.text().strip()
        if not folder:
            self._status.setText("Choose a folder first.")
            return
        if self._worker is not None or self._batch_running():
            self._status.setText("Something is already running — stop it first.")
            return

        paths = ex.list_files(folder, self._recursive.isChecked())
        self._listing = ((str(Path(folder)), self._recursive.isChecked()), list(paths))
        # An open picker matches its table now, not when the scan is done.
        self._rematch_picker()
        self._overviews, self._overview_ids = [], set()
        if not paths:
            self._grid.clear()
            self._item_index = {}
            self._entries = []
            self._status.setText(f"No readable datasets in {Path(folder).name}.")
            return

        self._entries = []
        self._grid.clear()
        self._item_index = {}
        self._log_view.clear()
        self._cancelled = False
        self._busy(True)
        self._log(f"Scanning {len(paths)} file(s) in {folder}…")

        relay = self._relay

        from napari.qt.threading import thread_worker

        @thread_worker
        def _run():
            found: list[ex.SampleEntry] = []
            for index, path in enumerate(paths):
                if self._should_cancel():
                    break
                relay.message.emit(f"Reading {path.name} ({index + 1} of {len(paths)})…")
                entry = ex.describe_file(path)
                found.append(entry)
                # Yield the description first so the grid fills in immediately;
                # the thumbnail costs a second a file and arrives after.
                yield ("entry", index, entry)
                if entry.readable:
                    relay.preview.emit(index, ex.thumbnail(path))
            return found

        worker = _run()
        worker.yielded.connect(self._on_scan_yield)
        worker.returned.connect(self._on_scan_done)
        worker.errored.connect(self._on_error)
        worker.finished.connect(self._clear_worker)
        self._worker = worker
        worker.start()

    def _on_scan_yield(self, payload) -> None:
        kind, index, entry = payload
        if kind != "entry":
            return
        # Yields arrive in order, one per file, so the row index and the list
        # index are the same by construction.
        self._entries.append(entry)

        item = QListWidgetItem(f"{entry.name}\n{entry.describe()}")
        item.setToolTip(f"{entry.path}\n{entry.describe()}")
        item.setData(PATH_ROLE, str(entry.path))
        item.setTextAlignment(Qt.AlignHCenter | Qt.AlignTop)
        if not entry.readable:
            item.setForeground(Qt.red)
        self._grid.addItem(item)
        if str(entry.path) in self._pending_selection:
            self._pending_selection.discard(str(entry.path))
            item.setSelected(True)

    def _set_preview(self, index: int, rgb) -> None:
        """Put a thumbnail on the item at *index*, if the grid still has one."""
        if rgb is None or index >= self._grid.count():
            return
        item = self._grid.item(index)
        pixmap = _to_pixmap(np.asarray(rgb))
        if pixmap is not None:
            item.setIcon(QIcon(pixmap))
            self.preview_ready.emit(str(item.data(PATH_ROLE)))

    def _on_scan_done(self, found) -> None:
        from .. import overview as ov

        samples, self._overviews = ov.find_overviews(list(found))
        kept = {id(entry) for entry in samples}
        # Overviews go to the end of the grid rather than out of it: the guess can
        # be wrong — a single-plane 10x image with 20x stacks taken inside it looks
        # exactly like an overview — and a sample that vanished could not be
        # opened, analysed or segmented at all.
        self._overview_ids = {id(entry) for entry in found if id(entry) not in kept}
        rows = min(len(found), self._grid.count())
        # Taking a tile out drops its selection; what was selected during the
        # scan (picks from a table, say) is put back after.
        selected = {str(item.data(PATH_ROLE)) for item in self._grid.selectedItems()}
        self._grid.blockSignals(True)
        items = [self._grid.takeItem(0) for _ in range(rows)]
        order = ([i for i in range(rows) if id(found[i]) in kept]
                 + [i for i in range(rows) if id(found[i]) not in kept])
        for i in order:
            self._grid.addItem(items[i])
            if id(found[i]) in self._overview_ids:
                self._mark_overview(items[i], found[i])
            if str(items[i].data(PATH_ROLE)) in selected:
                items[i].setSelected(True)
        self._grid.blockSignals(False)
        self._pending_selection = set()
        self._entries = [found[i] for i in order] + list(found[rows:])
        self._overview_button.setEnabled(bool(self._overviews))
        self._overview_button.setText(
            f"Overview ({len(self._overviews)})" if len(self._overviews) > 1 else "Overview")
        self.samples_changed.emit()
        readable = [entry for entry in self._entries if entry.readable]
        with_rois = [entry for entry in readable if entry.n_rois]
        with_labels = [entry for entry in readable if entry.label_keys]
        summary = (f"{len(readable) - len(self._overview_ids & {id(e) for e in readable})} "
                   f"sample(s) of {len(self._entries)} file(s).")
        if with_rois:
            summary += f" {len(with_rois)} already carry ROIs."
        if with_labels:
            summary += f" {len(with_labels)} already carry label maps."
        broken = [entry for entry in self._entries if not entry.readable]
        if broken:
            summary += f" {len(broken)} could not be read."
        if self._overviews:
            fields = sum(len(o.fields) for o in self._overviews)
            summary += (f" {len(self._overviews)} overview(s) ({fields} file(s)) found: shown "
                        "in their own window and at the end of the grid, marked “overview?”. "
                        "They are left out of “all samples” unless selected.")
        self._log(summary)
        if self._restoring is not None:
            # Coming back to last time's folder: the overview is a button away,
            # not a window opened over the viewer at every launch.
            self._finish_restore()
        elif self._overviews:
            self.show_overview()

    # -- overviews ------------------------------------------------------------

    def _mark_overview(self, item, entry) -> None:
        """Show a file taken for an overview as such, without hiding it."""
        item.setText(f"{entry.name}\noverview?  {entry.describe()}")
        font = item.font()
        font.setItalic(True)
        item.setFont(font)
        item.setForeground(Qt.gray)
        item.setToolTip(
            f"{entry.path}\n{entry.describe()}\n\nTaken for an overview: a single plane "
            "with other samples imaged inside it, a mosaic field, or named “overview”. "
            "Left out of “all samples”; select it to open, analyse or segment it like any "
            "other sample."
        )

    def show_overview(self) -> None:
        """Open the window with the overviews and every sample outlined on them."""
        if not self._overviews:
            self._status.setText("No overview in this folder.")
            return
        from .overview_window import OverviewWindow

        if self._overview_window is not None:
            try:
                self._overview_window.close()
            except RuntimeError:
                pass
        self._overview_window = OverviewWindow(
            self._overviews,
            read=lambda entry, size: ex.thumbnail(entry.path, size),
            on_select=self.select_entry,
            on_open=self._open_entry,
            parent=self,
        )
        self._overview_window.show()
        self._overview_window.raise_()

    def select_entry(self, entry) -> None:
        """Select *entry* in the grid, alone, and scroll to it."""
        if entry not in self._entries:
            return
        row = self._entries.index(entry)
        self._grid.clearSelection()
        item = self._grid.item(row)
        if item is not None:
            item.setSelected(True)
            self._grid.scrollToItem(item)

    def _open_entry(self, entry) -> None:
        self.select_entry(entry)
        self.open_selected()

    # -- picking from a table -------------------------------------------------

    def show_picker(self) -> None:
        """Open (or bring back) the window that picks samples off a table."""
        from .sample_picker_window import SamplePickerWindow

        window = self._picker_window
        if window is None:
            window = self._picker_window = SamplePickerWindow(
                paths=self.folder_paths,
                icon_of=self.icon_of,
                select=self.select_paths,
                open_path=self.open_path,
                folder=self.folder,
                parent=self,
            )
            self._picker_paths = {str(path) for path in self.folder_paths()}
            self.samples_changed.connect(self._rematch_picker)
            self.preview_ready.connect(window.on_preview)
        window.show()
        window.raise_()
        window.activateWindow()

    def _rematch_picker(self) -> None:
        # Only another folder changes which files there are; a new selection does not.
        window = self._picker_window
        paths = {str(path) for path in self.folder_paths()}
        if window is not None and paths != self._picker_paths:
            self._picker_paths = paths
            window.rematch()
            window.update_picks()

    def icon_of(self, path) -> QIcon | None:
        """The thumbnail the scan made of *path*, if it is in the grid."""
        item = self._items_by_path().get(str(path))
        return item.icon() if item is not None else None

    def _items_by_path(self) -> dict:
        """Grid items by path, rebuilt only when the grid changed.

        A table picks hundreds of samples at a time; looking each up by walking
        a thousand-tile grid made every tick in the picker take seconds.
        """
        index = self._item_index
        if len(index) != self._grid.count():
            index = self._item_index = {
                str(self._grid.item(row).data(PATH_ROLE)): self._grid.item(row)
                for row in range(self._grid.count())
            }
        return index

    def select_paths(self, paths) -> int:
        """Select exactly *paths* in the grid; returns how many were found."""
        index = self._items_by_path()
        items = [index[str(path)] for path in paths if str(path) in index]
        self._grid.blockSignals(True)
        try:
            self._grid.clearSelection()
            for item in items:
                item.setSelected(True)
        finally:
            self._grid.blockSignals(False)
        first = min(items, key=self._grid.row) if items else None
        if first is not None:
            self._grid.scrollToItem(first)
        self.samples_changed.emit()
        found = len(self._grid.selectedItems())
        # Picks the scan has not reached yet are selected as their tiles arrive.
        self._pending_selection = ({str(path) for path in paths} - set(index)) if self._worker is not None else set()
        self._status.setText(f"{found} sample(s) selected from the table."
                             + (f" {len(self._pending_selection)} more as the scan reaches them."
                                if self._pending_selection else ""))
        return found

    def open_path(self, path) -> None:
        """Open one sample in place of what is on screen, scanned yet or not."""
        if self.select_paths([path]):
            self.open_selected()
            return
        # The scan has not reached it: a header read is enough to open it.
        self.open_entries([ex.describe_file(path)])

    def folder_paths(self) -> list[Path]:
        """Every dataset in the folder, listed without reading any: what a table
        is matched against, so it does not wait for the scan."""
        folder = self.folder()
        if folder is None:
            return []
        key = (str(folder), self._recursive.isChecked())
        if self._listing[0] != key:
            self._listing = (key, ex.list_files(folder, self._recursive.isChecked()))
        return list(self._listing[1])

    def _clear_worker(self) -> None:
        self._worker = None
        self._cancelled = False
        # A scan that failed or was stopped: the next one is an ordinary scan.
        self._restoring = None
        self._busy(False)

    def _on_error(self, exc) -> None:
        logger.exception("experiment panel worker failed: %s", exc)
        self._log(f"Failed: {exc}")

    # -- opening and closing samples -----------------------------------------

    def open_selected(self) -> None:
        """Show the selected samples, replacing whatever sample is on screen.

        Replacing rather than adding is the point of stepping through a folder:
        the panel is a way to look at thirty samples one after another, and
        accumulating them would rebuild the layer list this panel exists to
        avoid. The selection *is* what is shown, so opening two shows exactly
        those two.

        Everything that belongs to the outgoing sample goes with it — its
        channels, the label maps read out of it, and its brain-region outlines.
        Outlines left behind would be drawn over the next fish, would stop its
        own stored outlines from loading, and would be counted against its
        objects. Outlines not yet saved are offered for saving first.

        What the incoming samples carry comes up with them: their stored
        outlines (through the Brain regions panel) and their stored label maps.
        """
        self.open_entries(self.selected_entries())

    def open_entries(self, entries) -> None:
        """Show *entries*, replacing whatever sample is on screen (see :meth:`open_selected`)."""
        entries = [entry for entry in entries if entry.readable]
        if not entries:
            self._status.setText("Nothing readable selected.")
            return
        if len(entries) > 4:
            answer = QMessageBox.question(
                self,
                "Microscopy Viewer",
                f"Show all {len(entries)} samples at once? That is "
                f"{sum(entry.n_channels for entry in entries)} layers, and the point of this "
                "panel is not having to.",
            )
            if answer != QMessageBox.Yes:
                return

        regions = getattr(self._app, "regions_widget", None)
        pending = self._settle_regions(regions)
        if pending is None:
            return
        counts = getattr(self._app, "positive_counter_widget", None)
        if counts is not None and hasattr(counts, "settle_unsaved") and not counts.settle_unsaved(self):
            return

        closed = self._close_all_samples()
        notes: list[str] = []
        if pending:
            target, rois = pending
            try:
                ims_store.save_rois(target, rois)
                notes.append(f"Saved {len(rois)} region(s) into {target.name} first.")
                self._refresh_entries([target])
            except Exception as exc:
                logger.exception("could not save the outgoing regions into %s", target)
                QMessageBox.warning(
                    self,
                    "Microscopy Viewer",
                    f"Could not save the regions into {target.name}:\n{exc}\n\n"
                    "They are still on the canvas; the new sample was not opened.",
                )
                return
        if regions is not None:
            regions.clear_regions()

        opened = self._app.open_paths([entry.path for entry in entries])
        loaded = self._load_stored_labels(entries)
        if regions is not None:
            regions.raise_layer()

        # Name the sample. Imaris records the acquiring machine's own path as the
        # image name, so every file in a folder can produce identically named
        # layers — and once opening replaces rather than adds, the layer list
        # stops being the thing that tells you which sample is on screen.
        shown = ", ".join(entry.name for entry in entries[:3])
        if len(entries) > 3:
            shown += f" and {len(entries) - 3} more"
        message = f"Showing {shown} — {opened} layer(s)."
        n_regions = len(regions.collect_regions()) if regions is not None else 0
        if n_regions:
            message += f" {n_regions} stored region(s)."
        if loaded:
            message += f" {loaded} stored label map(s)."
        if closed:
            message += f" Replaced the {closed} layer(s) that were on screen."
        self._log(" ".join([message, *notes]))

    def _settle_regions(self, regions):
        """Decide what happens to unsaved outlines before the sample goes.

        Returns ``None`` to abandon the switch, ``()`` to go ahead, or
        ``(path, rois)`` to go ahead and write *rois* into *path* once the
        sample is closed — the write needs the file to itself.
        """
        if regions is None or not regions.has_unsaved_regions():
            return ()
        target = regions.sample_path()
        writable = target is not None and ims_store.can_write(target)[0]

        box = QMessageBox(self)
        box.setIcon(QMessageBox.Question)
        box.setWindowTitle("Microscopy Viewer")
        where = f" into {target.name}" if target is not None else ""
        box.setText(
            "The brain regions on the canvas have not been saved"
            + (f" into {target.name}." if target is not None else ".")
        )
        box.setInformativeText(
            "They are cleared when another sample is opened."
            + ("" if writable else " There is no writable sample to save them into.")
        )
        save = box.addButton(f"Save{where}", QMessageBox.AcceptRole) if writable else None
        discard = box.addButton("Discard", QMessageBox.DestructiveRole)
        cancel = box.addButton(QMessageBox.Cancel)
        box.setDefaultButton(save or cancel)
        box.exec()
        clicked = box.clickedButton()
        if save is not None and clicked is save:
            rois = [
                ims_store.StoredRoi(name=region.name, vertices_um=region.vertices_world,
                                attrs=credit())
                for region in regions.collect_regions()
            ]
            return (Path(target), rois)
        if clicked is discard:
            return ()
        return None

    def _load_stored_labels(self, entries) -> int:
        """Put every label map stored in *entries* on screen. Returns how many."""
        loaded = 0
        for entry in entries:
            for key in ims_store.list_labels(entry.path):
                if self._add_labels_layer(entry, key) is not None:
                    loaded += 1
        return loaded

    def close_selected(self) -> int:
        """Take the selected samples off screen and release their files."""
        return self._close_paths([entry.path for entry in self.selected_entries()])

    def _close_all_samples(self) -> int:
        """Clear the screen for the next sample: every sample, and everything of theirs.

        Their channels and stored label maps, and every layer a panel made for
        them — the junction line and region, counted cells, hand counts, a
        segmentation — plus the QC picture. Left behind, those would be drawn
        over the next sample, counted against it, or saved into it.
        """
        removed = clear_samples(self._viewer)
        if removed:
            self._log(f"Closed {removed} layer(s) of the samples that were on screen.")
        return removed

    def _close_paths(self, paths) -> int:
        """Remove every layer backed by these files, then give the handles back.

        Both halves are needed. Removing the layers drops the dask graphs, but the
        reader keeps its ``h5py.File`` open for the lifetime of the process, and
        that alone is enough to make a write fail.

        Every reader's handles, not only Imaris's: a batch gives each file's
        handle back as it finishes with it, TIFF included.
        """
        from ..loaders import release

        wanted = {str(Path(path).resolve()) for path in paths}
        removed = 0
        for layer in list(self._viewer.layers):
            source = _layer_source(layer)
            if not source:
                continue
            try:
                resolved = str(Path(source).resolve())
            except OSError:
                continue
            if resolved in wanted:
                self._viewer.layers.remove(layer)
                removed += 1
        released = sum(release(path) for path in wanted)
        if removed or released:
            self._log(f"Closed {removed} layer(s) and released {released} file handle(s).")
        return removed

    # -- ROIs -----------------------------------------------------------------

    def _shapes_layer(self):
        """The Shapes layer ROIs are read from: the selected one, else the panel's."""
        from napari.layers import Shapes

        for layer in self._viewer.layers.selection:
            if isinstance(layer, Shapes):
                return layer
        try:
            return self._viewer.layers[rg.REGION_LAYER_NAME]
        except (KeyError, ValueError):
            return None

    def current_rois(self) -> list[ims_store.StoredRoi]:
        """The outlines on screen, in micrometres, ready to be stored."""
        layer = self._shapes_layer()
        if layer is None or not len(getattr(layer, "data", ())):
            return []
        regions = self._regions_of(layer)
        return [
            ims_store.StoredRoi(name=region.name, vertices_um=region.vertices_world,
                                attrs=credit())
            for region in regions
        ]

    def _regions_of(self, layer) -> list[rg.Region]:
        try:
            types = [str(kind) for kind in layer.shape_type]
        except Exception:
            types = ["polygon"] * len(layer.data)
        names: list[str] = []
        try:
            names = [str(value) for value in layer.features["name"]]
        except Exception:
            names = []
        return rg.regions_from_shapes(
            list(layer.data),
            types,
            names,
            scale=tuple(float(value) for value in layer.scale),
            translate=tuple(float(value) for value in layer.translate),
        )

    def write_rois_to_selected(self) -> None:
        """Store the outlines on screen into every selected sample's file."""
        rois = self.current_rois()
        if not rois:
            self._status.setText(
                f"No outlines to write. Draw them on a Shapes layer — “{rg.REGION_LAYER_NAME}” "
                "is the one the Brain regions panel makes."
            )
            return
        entries = [entry for entry in self.selected_entries() if entry.readable]
        if not entries:
            self._status.setText("Select the samples to write to.")
            return

        paths = [entry.path for entry in entries]
        self._close_paths(paths)  # nothing can be written while it is open for reading
        outcomes = ex.apply_rois(paths, rois, progress=self._log)

        failed = [outcome for outcome in outcomes if not outcome.ok]
        written = len(outcomes) - len(failed)
        source = self._shapes_layer()
        # Name the layer they came from: this reads whichever Shapes layer is
        # selected, and writing the wrong outlines into thirty files in silence
        # is the failure worth spending a few words to avoid.
        origin = f" from “{getattr(source, 'name', '?')}”" if source is not None else ""
        message = f"Wrote {len(rois)} ROI(s){origin} into {written} file(s)."
        if failed:
            message += " Failed on " + ", ".join(
                f"{outcome.name} ({outcome.error})" for outcome in failed[:2]
            )
        self._log(message)
        self._refresh_entries(paths)

    def load_rois_from_sample(self) -> None:
        """Put the ROIs stored in the selected sample back onto the canvas."""
        entries = [entry for entry in self.selected_entries() if entry.readable]
        if not entries:
            self._status.setText("Select a sample to read ROIs from.")
            return
        entry = entries[0]
        # Onto the Brain regions layer: anatomy only, not the junction band.
        rois = ims_store.anatomy(ims_store.load_rois(entry.path))
        if not rois:
            self._log(f"{entry.name} has no stored ROIs.")
            return

        regions = getattr(self._app, "regions_widget", None)
        if regions is not None:
            # The panel's own layer: names drawn on the canvas, and kept out of
            # the Measurements panel, which would rename every shape.
            regions.set_regions(rois)
            regions.raise_layer()
            self._log(f"Loaded {len(rois)} ROI(s) from {entry.name}.")
            return

        layer = self._shapes_layer()
        if layer is None:
            from ..loaders.layer_spec import world_units

            layer = self._viewer.add_shapes(
                name=rg.REGION_LAYER_NAME,
                edge_color="#ffcc00",
                edge_width=2,
                **world_units(self._viewer, 2),
            )
        try:
            import pandas as pd

            layer.data = [roi.vertices_um for roi in rois]
            layer.features = pd.DataFrame({"name": [roi.name for roi in rois]})
        except Exception as exc:
            logger.exception("could not load the ROIs onto the canvas")
            self._log(f"Could not show the ROIs: {exc}")
            return
        self._log(f"Loaded {len(rois)} ROI(s) from {entry.name}.")

    def _refresh_entries(self, paths) -> None:
        """Re-read what the given files now carry, and relabel their items."""
        wanted = {str(Path(path)) for path in paths}
        for index, entry in enumerate(self._entries):
            if str(entry.path) not in wanted or index >= self._grid.count():
                continue
            entry.n_rois = len(ims_store.load_rois(entry.path))
            entry.label_keys = ims_store.list_labels(entry.path)
            item = self._grid.item(index)
            item.setText(f"{entry.name}\n{entry.describe()}")
            if id(entry) in self._overview_ids:
                self._mark_overview(item, entry)

    # -- stored label maps ----------------------------------------------------

    def load_labels_from_sample(self) -> None:
        """Put a label map stored in the selected sample back into the viewer.

        Without this the batch is a black box: it says it wrote 3 889 objects
        into the file and there is no way to look at them, which is
        indistinguishable from its not having written anything. Reading them
        back is also the only check that what is in the file is the thing the
        run produced.

        Read whole rather than lazily, because the reader's handle is released
        every time anything is written into that file and a lazy layer would
        then raise on its next draw.
        """
        entries = [entry for entry in self.selected_entries() if entry.readable]
        if not entries:
            self._status.setText("Select a sample to read labels from.")
            return
        entry = entries[0]
        keys = ims_store.list_labels(entry.path)
        if not keys:
            self._log(
                f"{entry.name} carries no stored label maps. Run the batch with "
                "“Write the labels into each file” ticked."
            )
            return

        key = keys[0]
        if len(keys) > 1:
            from qtpy.QtWidgets import QInputDialog

            key, chosen = QInputDialog.getItem(
                self, "Load labels", f"Label map stored in {entry.name}:", keys, 0, False
            )
            if not chosen:
                return

        layer = self._add_labels_layer(entry, key)
        if layer is None:
            self._log(f"Could not read “{key}” out of {entry.name}.")
            return
        masks = layer.data
        self._log(
            f"Loaded “{key}” from {entry.name}: {int(np.max(masks)) if masks.size else 0} "
            f"object(s), {' × '.join(str(int(n)) for n in masks.shape)}."
        )

    def _add_labels_layer(self, entry, key: str):
        """Read one stored label map into a Labels layer tied to its sample."""
        masks, attrs = ims_store.load_labels(entry.path, key)
        if masks is None:
            return None

        scale = tuple(float(v) for v in np.asarray(attrs.get("voxel_size_um", ())).ravel())
        if len(scale) != int(masks.ndim):
            scale = tuple(float(v) for v in entry.voxel_um)[-int(masks.ndim):] or None

        from ..loaders.layer_spec import world_units

        name = f"{entry.name} — {key}"
        if name in self._viewer.layers:
            self._viewer.layers.remove(name)
        kwargs = {"name": name, "metadata": {SAMPLE_KEY: str(entry.path)}}
        if scale:
            kwargs["scale"] = scale
        kwargs.update(world_units(self._viewer, int(masks.ndim)) or {})
        colormap = _stored_colormap(attrs)
        if colormap is not None:
            kwargs["colormap"] = colormap
        return self._viewer.add_labels(np.asarray(masks), **kwargs)


def _stored_colormap(attrs):
    """The colours a label map was written with (a cluster map from the region
    explorer, say), as a napari colormap; ``None`` for an ordinary label map."""
    import json

    text = ims_store._text(attrs.get("label_colors", "")) if "label_colors" in attrs else ""
    try:
        colors = {int(k): v for k, v in json.loads(text).items()} if text else {}
    except (ValueError, AttributeError):
        return None
    if not colors:
        return None
    from napari.utils.colormaps import DirectLabelColormap

    return DirectLabelColormap(color_dict={None: "transparent", 0: "transparent", **colors})


def _to_pixmap(rgb: np.ndarray):
    """A contiguous RGB array as a QPixmap.

    The copy is not optional: ``QImage`` wraps the buffer it is given without
    owning it, and the numpy array goes out of scope as soon as this returns.
    """
    array = np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8))
    if array.ndim != 3 or array.shape[2] != 3:
        return None
    height, width, _ = array.shape
    image = QImage(array.data, width, height, 3 * width, QImage.Format_RGB888).copy()
    return QPixmap.fromImage(image)
