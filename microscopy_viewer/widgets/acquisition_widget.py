"""Dock panel: a movie of an open z-stack being acquired, channel by channel.

Pick a dataset that is open in the viewer, tick and order its channels, and the
panel renders the stack's maximum-intensity projection building up plane by
plane — one channel after another, each beside a merge — and writes it as an
MP4 for a talk. See :mod:`microscopy_viewer.acquisition` for how it is drawn.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from qtpy.QtCore import QObject, QSize, Qt, Signal
from qtpy.QtGui import QColor, QIcon, QImage, QPixmap
from qtpy.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .. import acquisition as aq
from .. import slides as sl
from ..utils import get_logger

logger = get_logger("acquisition_widget")

MOVIE_FILTER = "MP4 video (*.mp4);;QuickTime movie (*.mov);;Animated GIF (*.gif)"


class _Relay(QObject):
    """Carries worker-thread progress onto the GUI thread."""

    progress = Signal(int, int)
    status = Signal(str)


class AcquisitionWidget(QWidget):
    """Choose a dataset and its channels; preview and export the acquisition movie."""

    def __init__(self, app, parent: QWidget | None = None):
        super().__init__(parent)
        self._app = app
        self._viewer = app.viewer
        self._samples: list[sl.SampleSlide] = []
        self._worker = None
        self._cancelled = False
        self._relay = _Relay()
        self._relay.progress.connect(self._on_progress)
        self._relay.status.connect(self._set_status)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        intro = QLabel(
            "Plays a z-stack back the way it was acquired: each channel's maximum "
            "projection builds up plane by plane, then the next channel, beside a "
            "merge that gains each one."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        layout.addWidget(self._build_source_box())
        layout.addWidget(self._build_look_box())

        buttons = QHBoxLayout()
        self._preview_button = QPushButton("Preview")
        self._preview_button.setToolTip("Draw the last frame: every channel's finished projection and the merge.")
        self._preview_button.clicked.connect(self.preview)
        buttons.addWidget(self._preview_button)
        self._export_button = QPushButton("Export movie…")
        self._export_button.setToolTip("Render every frame and write an MP4 (or .mov, .gif).")
        self._export_button.clicked.connect(self.export)
        buttons.addWidget(self._export_button)
        self._stop_button = QPushButton("Stop")
        self._stop_button.setEnabled(False)
        self._stop_button.clicked.connect(self.stop)
        buttons.addWidget(self._stop_button)
        buttons.addStretch(1)
        layout.addLayout(buttons)

        self._progress = QProgressBar()
        self._progress.setVisible(False)
        layout.addWidget(self._progress)

        self._preview = QLabel()
        self._preview.setAlignment(Qt.AlignCenter)
        self._preview.setMinimumHeight(0)
        layout.addWidget(self._preview)

        self._status = QLabel("")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)
        layout.addStretch(1)

        events = self._viewer.layers.events
        events.inserted.connect(self.refresh)
        events.removed.connect(self.refresh)
        self.refresh()

    # -- construction ---------------------------------------------------------

    def _build_source_box(self) -> QGroupBox:
        box = QGroupBox("What to film")
        outer = QVBoxLayout(box)

        row = QHBoxLayout()
        row.addWidget(QLabel("Dataset"))
        self._sample_combo = QComboBox()
        self._sample_combo.setToolTip("An image open in the viewer. Its displayed timepoint is used.")
        self._sample_combo.currentIndexChanged.connect(self._fill_channels)
        row.addWidget(self._sample_combo, stretch=1)
        outer.addLayout(row)

        self._channel_list = QListWidget()
        self._channel_list.setDragDropMode(QAbstractItemView.InternalMove)
        self._channel_list.setIconSize(QSize(14, 14))
        self._channel_list.setToolTip(
            "Ticked channels are filmed, top to bottom. Drag to change the order."
        )
        self._channel_list.itemChanged.connect(self._update_summary)
        self._channel_list.model().rowsMoved.connect(self._update_summary)
        outer.addWidget(self._channel_list)
        hint = QLabel("Tick the channels to film; drag them into the order they are acquired in.")
        hint.setWordWrap(True)
        outer.addWidget(hint)
        return box

    def _build_look_box(self) -> QGroupBox:
        box = QGroupBox("Movie")
        form = QFormLayout(box)
        form.setLabelAlignment(Qt.AlignRight)

        self._layout_combo = QComboBox()
        self._layout_combo.addItems(list(aq.LAYOUTS))
        self._layout_combo.setToolTip(
            "Channels + merge: a panel per channel beside the merge.\n"
            "Merge only: one panel, the channels adding up in it."
        )
        form.addRow("Layout", self._layout_combo)

        self._contrast_combo = QComboBox()
        self._contrast_combo.addItems(list(aq.CONTRASTS))
        self._contrast_combo.setToolTip(
            "Fit to the projection: each channel stretched to its finished maximum "
            "projection, which is brighter than any one plane.\n"
            "As displayed: the contrast the layer has in the viewer."
        )
        form.addRow("Contrast", self._contrast_combo)

        self._seconds_spin = QDoubleSpinBox()
        self._seconds_spin.setRange(0.5, 120.0)
        self._seconds_spin.setValue(4.0)
        self._seconds_spin.setSuffix(" s")
        self._seconds_spin.setToolTip("How long the sweep through the stack lasts for each channel.")
        form.addRow("Per channel", self._seconds_spin)

        self._hold_spin = QDoubleSpinBox()
        self._hold_spin.setRange(0.0, 60.0)
        self._hold_spin.setValue(2.0)
        self._hold_spin.setSuffix(" s")
        self._hold_spin.setToolTip("How long the finished merge stays on screen at the end.")
        form.addRow("Hold at end", self._hold_spin)

        self._fps_spin = QSpinBox()
        self._fps_spin.setRange(5, 60)
        self._fps_spin.setValue(30)
        self._fps_spin.setSuffix(" fps")
        form.addRow("Frame rate", self._fps_spin)

        self._size_spin = QSpinBox()
        self._size_spin.setRange(240, 2160)
        self._size_spin.setSingleStep(60)
        self._size_spin.setValue(720)
        self._size_spin.setSuffix(" px")
        self._size_spin.setToolTip(
            "Longest edge of each panel. The stack is read from the coarsest "
            "pyramid level that still fills it, so smaller is faster."
        )
        form.addRow("Panel size", self._size_spin)

        self._reverse_check = QCheckBox("Sweep from the last plane to the first")
        form.addRow("Direction", self._reverse_check)
        self._labels_check = QCheckBox("Channel names and depth")
        self._labels_check.setChecked(True)
        form.addRow("Show", self._labels_check)
        self._scale_check = QCheckBox("Scale bar on the merge")
        self._scale_check.setChecked(True)
        form.addRow("", self._scale_check)

        self._summary = QLabel("")
        self._summary.setWordWrap(True)
        form.addRow(self._summary)

        for spin in (self._seconds_spin, self._hold_spin, self._fps_spin, self._size_spin):
            spin.valueChanged.connect(self._update_summary)
        self._layout_combo.currentIndexChanged.connect(self._update_summary)
        return box

    # -- what is open ---------------------------------------------------------

    def refresh(self, *_args) -> None:
        """Re-list the open datasets, keeping the current one if it is still there."""
        try:
            self._refresh()
        except RuntimeError:  # the panel was deleted while the viewer closes
            pass

    def _refresh(self) -> None:
        previous = self._sample_combo.currentText()
        try:
            self._samples = [s for s in sl.collect_samples(self._viewer) if s.channels]
        except Exception:
            logger.debug("could not list the open datasets", exc_info=True)
            self._samples = []
        self._sample_combo.blockSignals(True)
        self._sample_combo.clear()
        for sample in self._samples:
            self._sample_combo.addItem(sample.name)
        index = self._sample_combo.findText(previous)
        self._sample_combo.setCurrentIndex(index if index >= 0 else 0)
        self._sample_combo.blockSignals(False)
        self._fill_channels()

    def _fill_channels(self, *_args) -> None:
        self._channel_list.blockSignals(True)
        self._channel_list.clear()
        sample = self._current_sample()
        for channel in sample.channels if sample else []:
            item = QListWidgetItem(channel.label)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsDragEnabled)
            item.setCheckState(Qt.Checked if channel.in_merge else Qt.Unchecked)
            item.setData(Qt.UserRole, channel.layer_name)
            item.setIcon(_swatch(channel.color))
            self._channel_list.addItem(item)
        self._channel_list.blockSignals(False)
        self._channel_list.setFixedHeight(
            max(1, self._channel_list.count()) * self._channel_list.sizeHintForRow(0)
            + 2 * self._channel_list.frameWidth() + 4
            if self._channel_list.count() else 40
        )
        if not self._samples:
            self._set_status("Open a z-stack to film it.")
        self._update_summary()

    def _current_sample(self) -> sl.SampleSlide | None:
        index = self._sample_combo.currentIndex()
        return self._samples[index] if 0 <= index < len(self._samples) else None

    def chosen_layers(self) -> list[str]:
        """Layer names of the ticked channels, in the order they are listed."""
        names = []
        for row in range(self._channel_list.count()):
            item = self._channel_list.item(row)
            if item.checkState() == Qt.Checked:
                names.append(item.data(Qt.UserRole))
        return names

    def spec(self) -> aq.AcquisitionSpec:
        """The movie as set on the panel, read fresh from the viewer.

        Collected again rather than reusing the list: the timepoint and the
        layers' contrast may have changed since it was drawn.
        """
        sample = self._current_sample()
        if sample is None:
            raise aq.AcquisitionError("Open a z-stack first.")
        fresh = next((s for s in sl.collect_samples(self._viewer)
                      if (s.source or s.name) == (sample.source or sample.name)), sample)
        by_layer = {c.layer_name: c for c in fresh.channels}
        channels = [by_layer[name] for name in self.chosen_layers() if name in by_layer]
        if not channels:
            raise aq.AcquisitionError("Tick at least one channel.")
        meta = None
        try:
            meta = self._viewer.layers[channels[0].layer_name].metadata.get("mv_metadata")
        except (KeyError, ValueError):
            pass
        return aq.AcquisitionSpec(
            channels=channels,
            layout=self._layout_combo.currentText(),
            contrast=self._contrast_combo.currentText(),
            seconds_per_channel=float(self._seconds_spin.value()),
            hold_s=float(self._hold_spin.value()),
            fps=float(self._fps_spin.value()),
            panel_pixels=int(self._size_spin.value()),
            labels=self._labels_check.isChecked(),
            scale_bar=self._scale_check.isChecked(),
            pixel_size_um=fresh.pixel_size_um,
            z_step_um=getattr(meta, "z_step_um", None) if meta is not None else None,
            reverse=self._reverse_check.isChecked(),
        )

    def _update_summary(self, *_args) -> None:
        count = len(self.chosen_layers())
        if not count:
            self._summary.setText("No channel ticked.")
            return
        seconds = self._seconds_spin.value()
        total = count * seconds + self._hold_spin.value()
        frames = int(round(count * seconds * self._fps_spin.value())) + int(
            round(self._hold_spin.value() * self._fps_spin.value()))
        panels = count + 1 if self._layout_combo.currentText() == aq.LAYOUT_GRID else 1
        rows, columns = aq.grid_shape(panels)
        self._summary.setText(
            f"{count} channel(s) × {seconds:g} s + {self._hold_spin.value():g} s hold = "
            f"<b>{total:g} s</b>, {frames} frames; {columns} × {rows} panel(s) of up to "
            f"{self._size_spin.value()} px."
        )

    # -- running --------------------------------------------------------------

    def _set_status(self, text: str) -> None:
        self._status.setText(text)

    def _busy(self, running: bool) -> None:
        self._preview_button.setEnabled(not running)
        self._export_button.setEnabled(not running)
        self._stop_button.setEnabled(running)
        self._progress.setVisible(running)

    def stop(self) -> None:
        self._cancelled = True
        self._set_status("Stopping…")

    def _should_cancel(self) -> bool:
        return self._cancelled

    def _on_progress(self, done: int, total: int) -> None:
        self._progress.setMaximum(max(1, total))
        self._progress.setValue(done)
        self._status.setText(f"Frame {done} of {total}")

    def _start(self, job, on_done) -> None:
        if self._worker is not None:
            return
        from napari.qt.threading import thread_worker

        self._cancelled = False
        self._busy(True)
        self._progress.setValue(0)
        worker = thread_worker(job)()
        worker.returned.connect(on_done)
        worker.errored.connect(self._on_error)
        worker.finished.connect(self._finished)
        self._worker = worker
        worker.start()

    def _finished(self) -> None:
        self._worker = None
        self._busy(False)

    def _on_error(self, exc) -> None:
        from ..movie import MovieCancelled

        if isinstance(exc, (aq.Cancelled, MovieCancelled)):
            self._set_status("Stopped; nothing was written.")
            return
        logger.error("acquisition movie failed: %s", exc)
        self._set_status(f"Failed: {exc}")

    def preview(self) -> None:
        try:
            spec = self.spec()
        except aq.AcquisitionError as exc:
            self._set_status(str(exc))
            return
        self._set_status("Drawing the preview…")

        def _job():
            return aq.preview(spec, should_cancel=self._should_cancel)

        def _done(frame):
            self._show_preview(frame)
            self._set_status("Preview: the last frame of the movie.")

        self._start(_job, _done)

    def _show_preview(self, frame) -> None:
        if frame is None:
            return
        frame = np.ascontiguousarray(frame)
        height, width = frame.shape[:2]
        image = QImage(frame.data, width, height, 3 * width, QImage.Format_RGB888).copy()
        target = max(120, self.width() - 16)
        self._preview.setPixmap(QPixmap.fromImage(image).scaledToWidth(target, Qt.SmoothTransformation))

    def export(self, path: str | None = None):
        """Render and write the movie; to *path* if given, else ask where."""
        try:
            spec = self.spec()
        except aq.AcquisitionError as exc:
            self._set_status(str(exc))
            return None
        if not path:
            sample = self._current_sample()
            start = Path(getattr(self._app, "last_directory", None) or Path.home())
            name = f"{sample.name if sample else 'stack'}_acquisition.mp4"
            path, _ = QFileDialog.getSaveFileName(self, "Export the acquisition movie",
                                                  str(start / name), MOVIE_FILTER)
            if not path:
                return None
        if Path(path).suffix.lower() not in (".mp4", ".mov", ".gif", ".m4v"):
            path = str(Path(path).with_suffix(".mp4"))
        relay = self._relay

        def _job():
            return aq.export(spec, path, on_progress=relay.progress.emit,
                             should_cancel=self._should_cancel, on_status=relay.status.emit)

        def _done(written):
            self._set_status(f"Movie written to {written}.")
            try:
                self._app.last_directory = Path(written).parent
            except Exception:
                pass

        self._start(_job, _done)
        return path


def _swatch(color) -> QIcon:
    pixmap = QPixmap(14, 14)
    pixmap.fill(QColor.fromRgbF(*[max(0.0, min(1.0, float(c))) for c in color[:3]]))
    return QIcon(pixmap)
