"""Notice when the window has stopped responding, and cover it with the animation.

Every slow thing the viewer does — pulling a stack off the NAS, running a
segmentation model, writing a deck — happens on the Qt main thread, and while it
runs that thread cannot paint. The window goes white, stops redrawing, and
Windows eventually writes *(Not Responding)* into its title bar. Nothing is
actually wrong; there is simply nobody left to draw.

This watches for that and puts the loading animation over the window while it
lasts. Two pieces, because a blocked thread cannot animate anything:

* a **heartbeat**, a timer on the main thread that writes the time and where the
  window is into a small shared record. While the thread is stuck the heartbeat
  stops, which is the signal — and the record still holds the last known geometry,
  which is where the animation has to go.
* a **watchdog thread**, which is not blocked, comparing that timestamp against
  the clock and driving :mod:`microscopy_viewer.busy_window` — a separate process
  whose only job is to play the GIF. It is spawned once and kept hidden, so
  showing it during a freeze costs a line on a pipe rather than a process start.

The watchdog thread only ever touches plain data and a pipe, never a Qt object:
calling into Qt from a second thread while the first is inside a long call is how
a hang becomes a crash.

Nothing here is required. A child that will not start, a pipe that breaks, any
failure at all is logged once and the viewer carries on exactly as it did before.
"""

from __future__ import annotations

import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from .utils import get_logger

logger = get_logger("busy")

#: How long the main thread has to be silent before the animation appears. Long
#: enough that routine work — a layer being added, a dialog opening — never
#: flashes it up, short enough to beat the eye's own "is this broken?" timer.
BUSY_AFTER_S = 1.2

#: How often the main thread says it is alive, and how often the watchdog looks.
#: The heartbeat is cheap: a clock read and the window's geometry.
HEARTBEAT_MS = 200
POLL_S = 0.15


def is_stalled(beat: float, showing: bool, after: float, now: float | None = None) -> bool:
    """Whether the animation belongs on screen, given the last heartbeat.

    Split out from the loop so the rule can be checked without waiting in real
    time: silent for longer than *after*, and the window was there to cover.
    """
    now = time.monotonic() if now is None else now
    return bool(showing) and (now - float(beat)) >= float(after)


@dataclass
class Heartbeat:
    """What the main thread last managed to say about itself.

    Plain data behind a lock, because it is written from the Qt thread and read
    from the watchdog. The geometry is part of it for a reason: once the thread is
    stuck it can no longer be asked where the window is.
    """

    beat: float = field(default_factory=time.monotonic)
    geometry: tuple[int, int, int, int] = (0, 0, 0, 0)
    #: Whether the window was visible, unminimised and in front when last seen. An
    #: overlay over an app somebody has alt-tabbed away from is just litter on
    #: their screen.
    showing: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)

    def update(self, geometry: tuple[int, int, int, int], showing: bool) -> None:
        with self.lock:
            self.beat = time.monotonic()
            self.geometry = geometry
            self.showing = showing

    def read(self) -> tuple[float, tuple[int, int, int, int], bool]:
        with self.lock:
            return self.beat, self.geometry, self.showing


class OverlayProcess:
    """The child process that holds the animation, and the pipe to it."""

    def __init__(self) -> None:
        self._process: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._visible = False

    @property
    def alive(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def start(self) -> bool:
        """Spawn the child, hidden and ready. ``False`` if it could not start."""
        if self.alive:
            return True
        try:
            # ``sys.executable`` is pythonw.exe when the shortcut launched us, so
            # the child inherits the same "no console" behaviour for free.
            flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            self._process = subprocess.Popen(
                [sys.executable, "-m", "microscopy_viewer.busy_window"],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                cwd=str(Path(__file__).resolve().parent.parent),
                creationflags=flags if sys.platform == "win32" else 0,
            )
        except Exception:
            logger.info("could not start the busy overlay; freezes will look as they always did", exc_info=True)
            self._process = None
            return False
        logger.info("busy overlay ready (pid %s)", self._process.pid)
        return True

    def _send(self, line: str) -> bool:
        with self._lock:
            if not self.alive or self._process is None or self._process.stdin is None:
                return False
            try:
                self._process.stdin.write((line + "\n").encode("utf-8"))
                self._process.stdin.flush()
            except Exception:
                # The child died, or the pipe went. Nothing to recover: the viewer
                # is unaffected, it just stops showing an animation.
                logger.debug("lost the busy overlay pipe", exc_info=True)
                return False
            return True

    def show(self, geometry: tuple[int, int, int, int], message: str) -> None:
        if self._visible:
            return
        x, y, width, height = geometry
        # Newlines would be read as a second command; nothing else needs escaping.
        message = " ".join(str(message).split())
        self._visible = self._send(f"show {x} {y} {width} {height} {message}")

    def hide(self) -> None:
        if not self._visible:
            return
        self._send("hide")
        self._visible = False

    def stop(self) -> None:
        """Ask the child to quit, and do not wait long for it to agree."""
        self._visible = False
        if not self.alive or self._process is None:
            return
        self._send("quit")
        try:
            self._process.wait(timeout=2)
        except Exception:
            logger.debug("busy overlay did not quit; killing it", exc_info=True)
            try:
                self._process.kill()
            except Exception:
                pass
        self._process = None


class BusyWatchdog:
    """Ties the heartbeat, the watchdog thread and the overlay process together.

    The window is only ever touched from the Qt timer. ``message`` is what the
    overlay says while it is up; the child counts the seconds itself.
    """

    def __init__(self, window, message: str = "Working…", after: float = BUSY_AFTER_S) -> None:
        self._window = window
        self._message = message
        self._after = float(after)
        self._heart = Heartbeat()
        self._overlay = OverlayProcess()
        self._timer = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    # -- the main thread's end ------------------------------------------------

    def _pulse(self) -> None:
        """Runs on the Qt thread. Stops running the moment that thread is busy."""
        from qtpy.QtWidgets import QApplication

        try:
            geometry = (
                int(self._window.x()),
                int(self._window.y()),
                int(self._window.width()),
                int(self._window.height()),
            )
            # Any window of ours being active counts, not this one: the longest
            # freezes happen behind a modal dialog — an export, a segmentation —
            # and that dialog is what holds the focus while they do.
            showing = bool(
                self._window.isVisible()
                and not self._window.isMinimized()
                and QApplication.activeWindow() is not None
            )
        except Exception:
            # The window has been destroyed; there is nothing left to watch.
            geometry, showing = (0, 0, 0, 0), False
        self._heart.update(geometry, showing)

    # -- the watchdog's end ---------------------------------------------------

    def _watch(self) -> None:
        while not self._stop.wait(POLL_S):
            beat, geometry, showing = self._heart.read()
            if is_stalled(beat, showing, self._after):
                self._overlay.show(geometry, self._message)
            else:
                self._overlay.hide()
        self._overlay.hide()

    # -- lifecycle ------------------------------------------------------------

    def start(self) -> bool:
        """Begin watching. ``False`` when the overlay process refused to start."""
        from qtpy.QtCore import QTimer

        if not self._overlay.start():
            return False

        self._pulse()
        self._timer = QTimer()
        self._timer.timeout.connect(self._pulse)
        self._timer.start(HEARTBEAT_MS)

        self._thread = threading.Thread(target=self._watch, name="busy-watchdog", daemon=True)
        self._thread.start()
        logger.info("watching for freezes longer than %.1f s", self._after)
        return True

    def stop(self) -> None:
        self._stop.set()
        if self._timer is not None:
            try:
                self._timer.stop()
            except Exception:
                logger.debug("the heartbeat timer was already gone", exc_info=True)
            self._timer = None
        if self._thread is not None:
            self._thread.join(timeout=1)
            self._thread = None
        self._overlay.stop()


def start(window, message: str = "Working…", after: float = BUSY_AFTER_S) -> BusyWatchdog | None:
    """A running :class:`BusyWatchdog` for *window*, or ``None`` if it cannot run.

    Failure is never fatal and never loud: without this the viewer freezes the way
    every Qt application freezes, which is what it did before.
    """
    if window is None:
        return None
    try:
        watchdog = BusyWatchdog(window, message=message, after=after)
        return watchdog if watchdog.start() else None
    except Exception:
        logger.debug("could not start the freeze watchdog", exc_info=True)
        return None
