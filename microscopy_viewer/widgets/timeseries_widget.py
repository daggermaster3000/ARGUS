"""Dock widget for playing a time series back and writing it out as a movie.

napari's own slider plays a time lapse, but at a rate that drifts with how long
each frame took to read, and with no way to say which part of the series to
watch. This drives the same slider from a clock instead: the requested rate is
kept and frames are dropped when a read runs late, which is what makes playback
look smooth rather than merely fast.
"""

from __future__ import annotations

import time
from pathlib import Path

from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .. import timeseries
from ..utils import get_logger

logger = get_logger("timeseries_widget")

#: Slowest the timer is allowed to tick, so a high frame rate cannot starve the
#: rest of the GUI of event-loop time.
MIN_INTERVAL_MS = 8


class TimeSeriesWidget(QWidget):
    """Transport controls for the time axis, plus the cache status and export.

    The widget owns the playback clock and nothing else: the frames themselves
    come from napari, the local caching from
    :class:`~microscopy_viewer.timeseries.TimelineManager`, and the movie from
    :mod:`microscopy_viewer.movie`.
    """

    def __init__(self, app, parent: QWidget | None = None):
        super().__init__(parent)
        self._app = app
        self._viewer = app.viewer
        self._playing = False
        self._last_frame_at = 0.0
        self._refreshing = False

        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.PreciseTimer)
        self._timer.timeout.connect(self._tick)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)
        layout.addLayout(self._build_transport())
        layout.addLayout(self._build_range())

        self._status = QLabel("")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        buttons = QHBoxLayout()
        self._cache_button = QPushButton("Cache locally")
        self._cache_button.setToolTip(
            "Copy this time series to local disk so playback stops reading the network"
        )
        self._cache_button.clicked.connect(self.cache_now)
        buttons.addWidget(self._cache_button)
        export = QPushButton("Export movie…")
        export.setToolTip("Write the range below as a .mov, .mp4 or .gif (Ctrl+Shift+M)")
        export.clicked.connect(self.export_movie)
        buttons.addWidget(export)
        buttons.addStretch(1)
        layout.addLayout(buttons)
        layout.addStretch(1)

        self._connect_viewer()
        self.refresh()

    # -- construction ---------------------------------------------------------

    def _build_transport(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(4)
        for label, tooltip, handler in (
            ("|◀", "First timepoint", self.go_first),
            ("◀", "Previous timepoint", self.step_back),
        ):
            button = QPushButton(label)
            button.setToolTip(tooltip)
            button.setFixedWidth(32)
            button.clicked.connect(handler)
            row.addWidget(button)

        self._play_button = QPushButton("▶ Play")
        self._play_button.setToolTip("Play the range below (Ctrl+Space)")
        self._play_button.clicked.connect(self.toggle_play)
        row.addWidget(self._play_button)

        for label, tooltip, handler in (
            ("▶", "Next timepoint", self.step_forward),
            ("▶|", "Last timepoint", self.go_last),
        ):
            button = QPushButton(label)
            button.setToolTip(tooltip)
            button.setFixedWidth(32)
            button.clicked.connect(handler)
            row.addWidget(button)

        row.addWidget(QLabel("fps:"))
        self._fps = QDoubleSpinBox()
        self._fps.setRange(0.5, 60.0)
        self._fps.setDecimals(1)
        self._fps.setValue(10.0)
        self._fps.setFixedWidth(70)
        self._fps.setToolTip("Frames per second. Frames are dropped rather than slowed if a read runs late")
        self._fps.valueChanged.connect(self._on_fps_changed)
        row.addWidget(self._fps)

        self._loop = QCheckBox("Loop")
        self._loop.setChecked(True)
        row.addWidget(self._loop)
        row.addStretch(1)
        return row

    def _build_range(self) -> QGridLayout:
        grid = QGridLayout()
        grid.setHorizontalSpacing(6)
        grid.addWidget(QLabel("Play from:"), 0, 0)
        self._start = QSpinBox()
        self._start.setToolTip("First timepoint of the played range, and of an exported movie")
        self._start.valueChanged.connect(self._on_range_changed)
        grid.addWidget(self._start, 0, 1)
        grid.addWidget(QLabel("to"), 0, 2, alignment=Qt.AlignRight)
        self._stop = QSpinBox()
        self._stop.valueChanged.connect(self._on_range_changed)
        grid.addWidget(self._stop, 0, 3)
        full = QPushButton("Full range")
        full.clicked.connect(self.use_full_range)
        grid.addWidget(full, 0, 4)
        grid.setColumnStretch(5, 1)
        return grid

    def _connect_viewer(self) -> None:
        try:
            self._viewer.dims.events.current_step.connect(self._on_step_changed)
            self._viewer.layers.events.inserted.connect(lambda _e=None: self.refresh())
            self._viewer.layers.events.removed.connect(lambda _e=None: self.refresh())
        except Exception:  # pragma: no cover - napari event API drift
            logger.debug("could not connect the time-series handlers", exc_info=True)

    # -- state ----------------------------------------------------------------

    @property
    def _manager(self):
        return getattr(self._app, "timeline_manager", None)

    @property
    def axis(self) -> int | None:
        return timeseries.viewer_time_axis(self._viewer)

    @property
    def count(self) -> int:
        return timeseries.timepoint_count(self._viewer, self.axis)

    @property
    def playing(self) -> bool:
        return self._playing

    def range(self) -> tuple[int, int]:
        return timeseries.clamp_range(self._start.value(), self._stop.value(), self.count)

    # -- transport ------------------------------------------------------------

    def toggle_play(self) -> None:
        self.pause() if self._playing else self.play()

    def play(self) -> None:
        if self.count < 2:
            self._status.setText("No time series is open — nothing to play.")
            return
        start, stop = self.range()
        if stop <= start:
            self._status.setText("The played range is a single timepoint.")
            return
        self._playing = True
        self._last_frame_at = time.monotonic()
        self._play_button.setText("⏸ Pause")
        self._timer.start(self._interval_ms())
        manager = self._manager
        if manager is not None:
            # Pressing play is the point at which copying the series locally is
            # worth the network traffic — not when the file was opened, where it
            # would compete with the first slice and the first 3D upload.
            manager.start_caching()
            manager.set_playhead(timeseries.current_index(self._viewer, self.axis), 1)

    def pause(self) -> None:
        self._playing = False
        self._timer.stop()
        self._play_button.setText("▶ Play")
        self.refresh()

    def _interval_ms(self) -> int:
        return max(MIN_INTERVAL_MS, int(round(1000.0 / max(0.5, float(self._fps.value())))))

    def _on_fps_changed(self, _value=None) -> None:
        if self._playing:
            self._timer.start(self._interval_ms())

    def _tick(self) -> None:
        """One timer beat: advance by however many frames the clock says."""
        axis = self.axis
        if axis is None:
            self.pause()
            return
        if not self._frames_ready():
            # The previous frame is still loading. Queueing another slice now
            # would only lengthen the backlog; the elapsed time keeps counting,
            # so the next tick that does land jumps straight to where playback
            # should be by then.
            return

        now = time.monotonic()
        advance = timeseries.frames_to_advance(now - self._last_frame_at, float(self._fps.value()))
        start, stop = self.range()
        current = timeseries.current_index(self._viewer, axis)
        nxt = timeseries.next_index(current, start, stop, advance, self._loop.isChecked())
        if nxt is None:
            self.pause()
            return

        self._last_frame_at = now
        timeseries.set_index(self._viewer, nxt, axis)
        manager = self._manager
        if manager is not None:
            manager.set_playhead(nxt, 1)

    def _frames_ready(self) -> bool:
        """Whether every layer has finished loading the slice it was last asked for."""
        try:
            return all(bool(getattr(layer, "loaded", True)) for layer in self._viewer.layers)
        except Exception:  # pragma: no cover - defensive
            return True

    def step_forward(self) -> None:
        self._step(1)

    def step_back(self) -> None:
        self._step(-1)

    def _step(self, delta: int) -> None:
        axis = self.axis
        if axis is None:
            return
        self.pause()
        start, stop = self.range()
        current = timeseries.current_index(self._viewer, axis)
        nxt = timeseries.next_index(current, start, stop, delta, loop=True)
        if nxt is not None:
            timeseries.set_index(self._viewer, nxt, axis)

    def go_first(self) -> None:
        self.pause()
        timeseries.set_index(self._viewer, self.range()[0], self.axis)

    def go_last(self) -> None:
        self.pause()
        timeseries.set_index(self._viewer, self.range()[1], self.axis)

    def use_full_range(self) -> None:
        count = self.count
        if count < 1:
            return
        self._refreshing = True
        self._start.setValue(0)
        self._stop.setValue(count - 1)
        self._refreshing = False
        self.refresh()

    def _on_range_changed(self, _value=None) -> None:
        if self._refreshing:
            return
        self._refresh_status()

    def _on_step_changed(self, _event=None) -> None:
        self._refresh_status()

    # -- caching and export ---------------------------------------------------

    def cache_now(self) -> None:
        """Force the open time series onto local disk, even if caching is off."""
        manager = self._manager
        if manager is None:
            self._status.setText("Local caching is not available in this session.")
            return
        started = manager.cache_all(force=True)
        if started:
            self._status.setText("Caching timepoints locally — playback stays usable meanwhile.")
        else:
            self.refresh()

    def export_movie(self) -> None:
        from .movie_dialog import MovieExportDialog

        self.pause()
        dialog = MovieExportDialog(self._viewer, self._app.last_directory, parent=self)
        dialog.set_range(*self.range())
        if not dialog.exec_() or dialog.written is None:
            return
        self._app.last_directory = Path(dialog.written).parent
        message = f"Movie saved to {dialog.written}"
        self._status.setText(message)
        self._app.toolbar.set_status(message)

    # -- display --------------------------------------------------------------

    def refresh(self) -> None:
        """Bring the spin boxes and the status line in line with what is open."""
        count = self.count
        enabled = count > 1
        for widget in (self._play_button, self._start, self._stop, self._fps, self._cache_button):
            widget.setEnabled(enabled)

        self._refreshing = True
        for box in (self._start, self._stop):
            box.setRange(0, max(0, count - 1))
        if enabled and self._stop.value() == 0:
            self._stop.setValue(count - 1)
        self._refreshing = False
        self._refresh_status()

    def _refresh_status(self) -> None:
        count = self.count
        if count < 2:
            self._status.setText("No time series open. A file with more than one timepoint enables playback.")
            return

        axis = self.axis
        index = timeseries.current_index(self._viewer, axis)
        parts = [f"t {index} of {count - 1}"]

        layer = next(
            (candidate for candidate in self._viewer.layers if timeseries.has_timeline(candidate)),
            None,
        )
        if layer is not None:
            stamp = timeseries.format_timestamp(
                timeseries.elapsed_seconds(index, timeseries.time_interval(layer))
            )
            if stamp:
                parts.append(stamp)
        manager = self._manager
        if manager is not None and layer is not None:
            parts.append(manager.describe(layer))
        self._status.setText(" · ".join(parts))

    def closeEvent(self, event):  # pragma: no cover - Qt teardown
        self.pause()
        super().closeEvent(event)
