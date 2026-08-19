"""Dialog for writing the open time series out as a movie."""

from __future__ import annotations

from pathlib import Path

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from .. import movie, timeseries
from ..utils import get_logger

logger = get_logger("movie_dialog")

#: Oversampling factors offered, with what each is for.
SCALES = (
    (1, "1× — screen resolution"),
    (2, "2× — sharp in PowerPoint"),
    (3, "3× — poster"),
)


class MovieExportDialog(QDialog):
    """Pick a range, a rate and a file, then render the canvas frame by frame.

    Rendering happens on the GUI thread on purpose: every frame is a screenshot
    of the canvas, so the canvas has to be free to draw it. The dialog stays
    responsive because :func:`~microscopy_viewer.movie.capture_frames` pumps the
    event loop while it waits for each slice to load, and **Cancel** becomes
    **Stop**, which finishes the frame in flight and writes nothing.
    """

    def __init__(self, viewer, last_directory=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export movie")
        self._viewer = viewer
        self._axis = timeseries.viewer_time_axis(viewer)
        self._count = timeseries.timepoint_count(viewer, self._axis)
        self._cancelled = False
        self._running = False
        self.written: Path | None = None

        directory = Path(last_directory) if last_directory else Path.home()
        self._suggested = directory / f"{movie.default_stem()}.mov"

        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        if self._axis is None or self._count < 2:
            layout.addWidget(
                QLabel(
                    "None of the open images have a time axis, so there is nothing to play.\n"
                    "Open a time series first."
                )
            )
            buttons = QDialogButtonBox(QDialogButtonBox.Close)
            buttons.rejected.connect(self.reject)
            buttons.accepted.connect(self.reject)
            layout.addWidget(buttons)
            return

        layout.addLayout(self._build_form())

        self._info = QLabel("")
        self._info.setWordWrap(True)
        layout.addWidget(self._info)

        self._progress = QProgressBar()
        self._progress.setVisible(False)
        layout.addWidget(self._progress)

        self._buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self._buttons.button(QDialogButtonBox.Ok).setText("Export")
        self._buttons.accepted.connect(self._export)
        self._buttons.rejected.connect(self._on_reject)
        layout.addWidget(self._buttons)

        self._refresh_info()

    # -- what the caller may set ----------------------------------------------

    @property
    def usable(self) -> bool:
        """False when there is no time axis, so the dialog is only a message."""
        return self._axis is not None and self._count >= 2

    def set_range(self, start: int, stop: int) -> None:
        """Preselect the timepoints to write — the range the panel is playing."""
        if not self.usable:
            return
        self._start.setValue(int(start))
        self._stop.setValue(int(stop))

    # -- construction ---------------------------------------------------------

    def _build_form(self) -> QGridLayout:
        form = QGridLayout()
        form.setHorizontalSpacing(8)
        row = 0

        form.addWidget(QLabel("File:"), row, 0)
        path_row = QHBoxLayout()
        self._path = QLineEdit(str(self._suggested))
        self._path.textChanged.connect(self._refresh_info)
        path_row.addWidget(self._path, stretch=1)
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        path_row.addWidget(browse)
        form.addLayout(path_row, row, 1, 1, 3)
        row += 1

        form.addWidget(QLabel("Timepoints:"), row, 0)
        self._start = QSpinBox()
        self._start.setRange(0, self._count - 1)
        self._start.valueChanged.connect(self._refresh_info)
        form.addWidget(self._start, row, 1)
        form.addWidget(QLabel("to"), row, 2, alignment=Qt.AlignRight)
        self._stop = QSpinBox()
        self._stop.setRange(0, self._count - 1)
        self._stop.setValue(self._count - 1)
        self._stop.valueChanged.connect(self._refresh_info)
        form.addWidget(self._stop, row, 3)
        row += 1

        form.addWidget(QLabel("Every nth:"), row, 0)
        self._step = QSpinBox()
        self._step.setRange(1, max(1, self._count))
        self._step.setToolTip("Write every nth timepoint — a way to shorten a long series")
        self._step.valueChanged.connect(self._refresh_info)
        form.addWidget(self._step, row, 1)

        form.addWidget(QLabel("Frames/s:"), row, 2, alignment=Qt.AlignRight)
        self._fps = QDoubleSpinBox()
        self._fps.setRange(0.5, 120.0)
        self._fps.setDecimals(1)
        self._fps.setValue(10.0)
        self._fps.valueChanged.connect(self._refresh_info)
        form.addWidget(self._fps, row, 3)
        row += 1

        form.addWidget(QLabel("Resolution:"), row, 0)
        self._scale = QComboBox()
        for factor, label in SCALES:
            self._scale.addItem(label, factor)
        self._scale.setCurrentIndex(1)
        form.addWidget(self._scale, row, 1, 1, 3)
        row += 1

        form.addWidget(QLabel("Quality:"), row, 0)
        self._quality = QSpinBox()
        self._quality.setRange(0, 10)
        self._quality.setValue(8)
        self._quality.setToolTip("Encoder quality: higher is a larger file. 8 is visually lossless enough for slides")
        form.addWidget(self._quality, row, 1)

        self._timestamp = QCheckBox("Burn in the time")
        self._timestamp.setToolTip(
            "Draw the elapsed acquisition time into the corner of every frame"
        )
        interval = self._interval()
        if interval is None:
            self._timestamp.setToolTip(
                "The file records no time interval, so frames are stamped with their timepoint number"
            )
        form.addWidget(self._timestamp, row, 2, 1, 2)
        return form

    # -- helpers --------------------------------------------------------------

    def _interval(self) -> float | None:
        for layer in self._viewer.layers:
            if timeseries.has_timeline(layer):
                interval = timeseries.time_interval(layer)
                if interval:
                    return float(interval)
        return None

    def _spec(self) -> movie.MovieSpec:
        return movie.MovieSpec(
            path=Path(self._path.text().strip()),
            fps=float(self._fps.value()),
            start=int(self._start.value()),
            stop=int(self._stop.value()),
            step=int(self._step.value()),
            scale=int(self._scale.currentData()),
            quality=int(self._quality.value()),
            timestamp=self._timestamp.isChecked(),
            interval_s=self._interval(),
        )

    def _browse(self) -> None:
        path, _selected = QFileDialog.getSaveFileName(
            self, "Export movie", self._path.text(), movie.MOVIE_FILTER
        )
        if path:
            self._path.setText(path)

    def _refresh_info(self) -> None:
        spec = self._spec()
        frames = len(spec.indices)
        parts = [f"{frames} frame(s), {spec.duration_s:.1f} s at {spec.fps:g} fps."]

        suffix = spec.path.suffix.lower()
        if suffix == ".gif":
            parts.append("GIF: no codec needed, but large and 256 colours.")
        else:
            ok, message = movie.encoder_available()
            if not ok:
                parts.append(message)
            elif suffix == ".mov":
                parts.append("H.264 in a QuickTime container — plays inside a PowerPoint slide.")
            elif suffix in movie.VIDEO_SUFFIXES:
                parts.append("H.264 video.")
            else:
                parts.append(f"{suffix or 'No'} is not a movie format — use .mov, .mp4 or .gif.")
        self._info.setText(" ".join(parts))

    # -- running --------------------------------------------------------------

    def _on_reject(self) -> None:
        """Cancel closes the dialog; Stop interrupts a run in progress."""
        if self._running:
            self._cancelled = True
            return
        self.reject()

    def _export(self) -> None:
        spec = self._spec()
        if not str(spec.path).strip():
            QMessageBox.information(self, "Export movie", "Choose where to write the movie.")
            return
        if spec.path.exists() and not self._confirm_overwrite(spec.path):
            return

        self._running = True
        self._cancelled = False
        self._progress.setVisible(True)
        self._progress.setRange(0, len(spec.indices))
        self._buttons.button(QDialogButtonBox.Ok).setEnabled(False)
        self._buttons.button(QDialogButtonBox.Cancel).setText("Stop")

        def _progress(done: int, total: int) -> None:
            self._progress.setValue(done)
            self._info.setText(f"Rendering frame {done} of {total}…")

        try:
            self.written = movie.export_movie(
                self._viewer,
                spec,
                on_progress=_progress,
                should_cancel=lambda: self._cancelled,
            )
        except movie.MovieCancelled:
            self._reset("Stopped. Nothing was written.")
            return
        except movie.MovieExportError as exc:
            self._reset(str(exc))
            QMessageBox.warning(self, "Export movie", str(exc))
            return
        except Exception as exc:
            logger.exception("movie export failed")
            self._reset(str(exc))
            QMessageBox.critical(self, "Export movie", f"The movie could not be written:\n{exc}")
            return

        self._running = False
        self.accept()

    def _confirm_overwrite(self, path: Path) -> bool:
        answer = QMessageBox.question(
            self,
            "Export movie",
            f"{path.name} already exists. Overwrite it?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        return answer == QMessageBox.Yes

    def _reset(self, message: str) -> None:
        self._running = False
        self._cancelled = False
        self._progress.setVisible(False)
        self._buttons.button(QDialogButtonBox.Ok).setEnabled(True)
        self._buttons.button(QDialogButtonBox.Cancel).setText("Cancel")
        self._info.setText(message)
