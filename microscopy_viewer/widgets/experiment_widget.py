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
from ..utils import get_logger

logger = get_logger("experiment_widget")

#: Side of a thumbnail in the grid, in screen pixels.
ICON_SIZE = 128

#: Role the sample's path is stashed under on its list item.
PATH_ROLE = Qt.UserRole + 1

#: Layer metadata key naming the file a layer was read out of, for layers that
#: are not image channels (a stored label map has no acquisition metadata) but
#: still belong to that sample and have to go when it does.
SAMPLE_KEY = "argus_sample"


def _layer_source(layer) -> str:
    """The file a layer belongs to: its acquisition's, or the one it was read from."""
    meta = layer.metadata.get("mv_metadata")
    source = str(getattr(meta, "file_path", "") or "") if meta is not None else ""
    return source or str(layer.metadata.get(SAMPLE_KEY, "") or "")


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
        if self._worker is not None or self._batch_running():
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
        entries = [entry for entry in self.selected_entries() if entry.readable]
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
                ims_store.StoredRoi(name=region.name, vertices_um=region.vertices_world)
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
        """Close every layer that came from a file, whichever panel opened it."""
        sources = []
        for layer in self._viewer.layers:
            source = _layer_source(layer)
            if source and source not in sources:
                sources.append(source)
        return self._close_paths(sources) if sources else 0

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
            self._grid.item(index).setText(f"{entry.name}\n{entry.describe()}")

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
                "“Write the labels into each .ims file” ticked."
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
