"""Dock panel: a folder of samples, previewed as thumbnails and processed at once.

Three things in one panel, because they are three steps of one job:

1. **Pick the folder and look at it.** Every readable dataset gets a thumbnail
   read off a middle pyramid level, so a bad mount or an empty stub is visible
   before anything is opened. Nothing is added to the viewer by scanning.
2. **Put ROIs on the samples.** An outline drawn in the viewer — the same Shapes
   layer the *Brain regions* panel uses — is written into the ``.ims`` files
   themselves, either into the one sample it was drawn on or into every sample
   selected.
3. **Run Cellpose over the folder**, with the settings from the *Segmentation*
   panel, writing each label map back into its own file.

Everything heavy happens on a ``thread_worker``. Two of them, actually: the
preview scan and the batch run, which cannot both be going at once.

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
    QVBoxLayout,
    QWidget,
)

from .. import experiment as ex
from .. import ims_store
from .. import regions as rg
from ..exports import WORKBOOK_FILTER, default_stem, export_sheets
from ..utils import get_logger

logger = get_logger("experiment_widget")

#: Side of a thumbnail in the grid, in screen pixels.
ICON_SIZE = 128

#: Role the sample's path is stashed under on its list item.
PATH_ROLE = Qt.UserRole + 1


class _Relay(QObject):
    """Carries worker-thread text and images onto the GUI thread.

    Same reason as the segmentation panel's: a widget may only be touched from
    the thread that owns it, and a signal is the supported way across.
    """

    message = Signal(str)
    preview = Signal(int, object)


class ExperimentWidget(QWidget):
    """Browse a folder of samples, put ROIs on them, segment all of them."""

    def __init__(self, app, parent: QWidget | None = None):
        super().__init__(parent)
        self._app = app
        self._viewer = app.viewer
        self._entries: list[ex.SampleEntry] = []
        self._worker = None
        self._cancelled = False
        self._outcomes: list[ex.BatchOutcome] = []
        self._relay = _Relay()
        self._relay.message.connect(self._log)
        self._relay.preview.connect(self._set_preview)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        layout.addWidget(self._build_folder_box())
        layout.addWidget(self._build_samples_box(), stretch=1)
        layout.addWidget(self._build_roi_box())
        layout.addWidget(self._build_batch_box())

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

        choose = QPushButton("Choose…")
        choose.clicked.connect(self.choose_folder)
        row.addWidget(choose)
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
        self._grid.itemDoubleClicked.connect(lambda _item: self.open_selected())
        outer.addWidget(self._grid)

        row = QHBoxLayout()
        for label, tooltip, handler in (
            ("Open", "Add the selected samples to the viewer (double-click does this too).",
             self.open_selected),
            ("Close", "Take the selected samples off screen and release their files, "
                      "which is what lets anything be written into them.", self.close_selected),
            ("Select all", "", self._grid.selectAll),
        ):
            button = QPushButton(label)
            if tooltip:
                button.setToolTip(tooltip)
            button.clicked.connect(handler)
            row.addWidget(button)
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
        to_selected = QPushButton("Write to selected")
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

    def _build_batch_box(self) -> QGroupBox:
        box = QGroupBox("Batch segmentation")
        outer = QVBoxLayout(box)

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
        outer.addLayout(form)

        row = QHBoxLayout()
        self._run_button = QPushButton("Run on selected")
        self._run_button.setToolTip(
            "Segment every selected sample with the settings currently on the Segmentation "
            "panel — model, mode, diameters, device. Nothing selected means all of them."
        )
        self._run_button.clicked.connect(self.run_batch)
        row.addWidget(self._run_button)

        export = QPushButton("Export…")
        export.setToolTip("Write the per-sample counts and every object to one workbook.")
        export.clicked.connect(self.export)
        row.addWidget(export)
        row.addStretch(1)
        outer.addLayout(row)

        self._log_view = QPlainTextEdit()
        self._log_view.setReadOnly(True)
        self._log_view.setMaximumHeight(120)
        self._log_view.setPlaceholderText("Progress appears here.")
        outer.addWidget(self._log_view)
        return box

    # -- small helpers --------------------------------------------------------

    def _log(self, text: str) -> None:
        self._log_view.appendPlainText(text)
        self._status.setText(text)

    def _busy(self, running: bool) -> None:
        self._scan_button.setEnabled(not running)
        self._run_button.setEnabled(not running)
        self._stop_button.setEnabled(running)

    def stop(self) -> None:
        """Ask the running worker to stop after the file it is on."""
        self._cancelled = True
        self._log("Stopping after the current file…")

    def _should_cancel(self) -> bool:
        return self._cancelled

    def selected_entries(self) -> list[ex.SampleEntry]:
        """The selected samples, or all of them when nothing is selected."""
        rows = [self._grid.row(item) for item in self._grid.selectedItems()]
        if not rows:
            return list(self._entries)
        return [self._entries[row] for row in sorted(rows) if row < len(self._entries)]

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
        if self._worker is not None:
            self._status.setText("Something is already running — stop it first.")
            return

        paths = ex.list_files(folder, self._recursive.isChecked())
        if not paths:
            self._grid.clear()
            self._entries = []
            self._status.setText(f"No readable datasets in {Path(folder).name}.")
            return

        self._entries = []
        self._grid.clear()
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

    def _set_preview(self, index: int, rgb) -> None:
        """Put a thumbnail on the item at *index*, if the grid still has one."""
        if rgb is None or index >= self._grid.count():
            return
        item = self._grid.item(index)
        pixmap = _to_pixmap(np.asarray(rgb))
        if pixmap is not None:
            item.setIcon(QIcon(pixmap))

    def _on_scan_done(self, found) -> None:
        self._entries = list(found)
        readable = [entry for entry in self._entries if entry.readable]
        with_rois = [entry for entry in readable if entry.n_rois]
        with_labels = [entry for entry in readable if entry.label_keys]
        summary = f"{len(readable)} sample(s) of {len(self._entries)} file(s)."
        if with_rois:
            summary += f" {len(with_rois)} already carry ROIs."
        if with_labels:
            summary += f" {len(with_labels)} already carry label maps."
        broken = [entry for entry in self._entries if not entry.readable]
        if broken:
            summary += f" {len(broken)} could not be read."
        self._log(summary)

    def _clear_worker(self) -> None:
        self._worker = None
        self._cancelled = False
        self._busy(False)

    def _on_error(self, exc) -> None:
        logger.exception("experiment panel worker failed: %s", exc)
        self._log(f"Failed: {exc}")

    # -- opening and closing samples -----------------------------------------

    def open_selected(self) -> None:
        """Add the selected samples to the viewer."""
        entries = [entry for entry in self.selected_entries() if entry.readable]
        if not entries:
            self._status.setText("Nothing readable selected.")
            return
        if len(entries) > 4:
            answer = QMessageBox.question(
                self,
                "Microscopy Viewer",
                f"Open all {len(entries)} samples? That is {sum(entry.n_channels for entry in entries)} "
                "layers, and the point of this panel is not having to.",
            )
            if answer != QMessageBox.Yes:
                return
        opened = self._app.open_paths([entry.path for entry in entries])
        self._log(f"Opened {opened} layer(s) from {len(entries)} sample(s).")

    def close_selected(self) -> int:
        """Take the selected samples off screen and release their files."""
        return self._close_paths([entry.path for entry in self.selected_entries()])

    def _close_paths(self, paths) -> int:
        """Remove every layer backed by these files, then give the handles back.

        Both halves are needed. Removing the layers drops the dask graphs, but the
        reader keeps its ``h5py.File`` open for the lifetime of the process, and
        that alone is enough to make a write fail.
        """
        from ..loaders import ims as ims_reader

        wanted = {str(Path(path).resolve()) for path in paths}
        removed = 0
        for layer in list(self._viewer.layers):
            meta = layer.metadata.get("mv_metadata")
            source = str(getattr(meta, "file_path", "") or "") if meta is not None else ""
            if not source:
                continue
            try:
                resolved = str(Path(source).resolve())
            except OSError:
                continue
            if resolved in wanted:
                self._viewer.layers.remove(layer)
                removed += 1
        released = sum(ims_reader.release(path) for path in wanted)
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
            ims_store.StoredRoi(name=region.name, vertices_um=region.vertices_world)
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
        rois = ims_store.load_rois(entry.path)
        if not rois:
            self._log(f"{entry.name} has no stored ROIs.")
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
            self._grid.item(index).setText(f"{entry.name}\n{entry.describe()}")

    # -- the batch ------------------------------------------------------------

    def _batch_settings(self):
        """The segmentation settings, taken from the Segmentation panel.

        Read live rather than duplicated here: two sets of controls for one set of
        parameters is how a batch ends up run with settings nobody chose.
        """
        from .. import segmentation as sg

        panel = getattr(self._app, "segmentation_widget", None)
        if panel is None:
            return sg.SegmentationSettings()
        try:
            return panel.settings()
        except Exception:
            logger.debug("could not read the segmentation settings", exc_info=True)
            return sg.SegmentationSettings()

    @staticmethod
    def _channel_spec(text: str):
        """A channel box's contents as an index or a name; empty means none."""
        cleaned = str(text).strip()
        if not cleaned:
            return None
        return int(cleaned) if cleaned.isdigit() else cleaned

    def run_batch(self) -> None:
        """Segment every selected sample and write the labels back into the files."""
        if self._worker is not None:
            self._status.setText("Something is already running — stop it first.")
            return
        entries = [entry for entry in self.selected_entries() if entry.readable]
        if not entries:
            self._status.setText("Nothing readable selected.")
            return

        channel = self._channel_spec(self._channel_edit.text())
        if channel is None:
            self._status.setText("Say which channel to segment.")
            return

        settings = self._batch_settings()
        options = ex.BatchOptions(
            channel=channel,
            measure_channel=self._channel_spec(self._measure_edit.text()),
            settings=settings,
            save_to_file=self._save_labels.isChecked(),
            restrict_to_rois=self._restrict.isChecked(),
        )

        paths = [entry.path for entry in entries]
        if options.save_to_file:
            # Writing needs the files to itself, and they may well be the ones
            # just used to draw the ROIs.
            self._close_paths(paths)

        self._outcomes = []
        self._log_view.clear()
        self._cancelled = False
        self._busy(True)
        self._log(
            f"Segmenting {len(paths)} sample(s) on “{channel}” with {settings.resolved_model()}, "
            f"{settings.mode}."
        )

        relay = self._relay

        from napari.qt.threading import thread_worker

        @thread_worker
        def _run():
            return ex.run_batch(
                paths,
                options,
                progress=relay.message.emit,
                should_cancel=self._should_cancel,
            )

        worker = _run()
        worker.returned.connect(self._on_batch_done)
        worker.errored.connect(self._on_error)
        worker.finished.connect(self._clear_worker)
        self._worker = worker
        worker.start()

    def _on_batch_done(self, outcomes) -> None:
        self._outcomes = list(outcomes)
        done = [outcome for outcome in self._outcomes if outcome.ok]
        failed = [outcome for outcome in self._outcomes if not outcome.ok]
        total = sum(outcome.n_objects for outcome in done)
        seconds = sum(outcome.elapsed_s for outcome in self._outcomes)

        summary = (
            f"{total} object(s) across {len(done)} sample(s) in {seconds:.0f} s."
        )
        if failed:
            summary += " Failed: " + ", ".join(
                f"{outcome.name} ({outcome.error})" for outcome in failed[:2]
            )
            if len(failed) > 2:
                summary += f" and {len(failed) - 2} more"
        if self._cancelled:
            summary += " Stopped early."
        self._log(summary)
        self._refresh_entries([outcome.path for outcome in self._outcomes])
        logger.info("batch finished: %s", summary)

    # -- export ---------------------------------------------------------------

    def export(self) -> None:
        """Write the batch summary and every object it found to one workbook."""
        if not self._outcomes:
            self._status.setText("Nothing to export — run a batch first.")
            return
        suggested = str(
            Path(self._folder_edit.text().strip() or Path.home())
            / f"{default_stem('batch_segmentation')}.xlsx"
        )
        path, _selected = QFileDialog.getSaveFileName(
            self, "Export batch results", suggested, WORKBOOK_FILTER
        )
        if not path:
            return
        sheets = {
            "Samples": ex.batch_dataframe(self._outcomes),
            "Objects": ex.objects_dataframe(self._outcomes),
        }
        try:
            written = export_sheets(sheets, path)
        except Exception as exc:
            logger.exception("batch export failed")
            QMessageBox.critical(self, "Microscopy Viewer", f"Export failed:\n{exc}")
            return
        self._log(f"Batch results written to {written}.")


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
