"""Application-wide drag-and-drop so files dropped on the window use our readers.

napari has its own drop handling, but it routes files through plugin readers that
know nothing about ``.ims``. An application-level event filter lets us claim drops
we can handle and leave everything else to napari.
"""

from __future__ import annotations

import os
import sys
from typing import Callable, Sequence

from qtpy.QtCore import QEvent, QObject, QTimer
from qtpy.QtWidgets import QApplication

from .loaders import is_supported
from .utils import get_logger

logger = get_logger("dragdrop")


class FileDropFilter(QObject):
    """Intercepts drops of supported microscopy files anywhere in the application.

    Only drops where at least one path looks loadable are claimed; a drop onto a
    text field, or of an unsupported file type, falls through untouched.
    """

    def __init__(self, callback: Callable[[Sequence[str]], None], parent: QObject | None = None):
        super().__init__(parent)
        self._callback = callback

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802 (Qt naming)
        event_type = event.type()
        if event_type not in (QEvent.DragEnter, QEvent.DragMove, QEvent.Drop):
            return False

        paths = self._paths(event)
        if not paths:
            return False

        if event_type in (QEvent.DragEnter, QEvent.DragMove):
            event.acceptProposedAction()
            return True

        event.acceptProposedAction()
        # Return control to Qt before loading: the drag source stays blocked
        # until the drop handler returns, and large files take a while.
        QTimer.singleShot(0, lambda captured=list(paths): self._load(captured))
        return True

    @staticmethod
    def _paths(event: QEvent) -> list[str]:
        """Local file paths in the drop payload that one of our readers accepts."""
        mime = getattr(event, "mimeData", None)
        if mime is None:
            return []
        data = mime()
        if data is None or not data.hasUrls():
            return []
        paths = [url.toLocalFile() for url in data.urls()]
        return [path for path in paths if path and is_supported(path)]

    def _load(self, paths: list[str]) -> None:
        try:
            self._callback(paths)
        except Exception:  # pragma: no cover - the callback reports its own errors
            logger.exception("drop handling failed for %s", paths)


def install(callback: Callable[[Sequence[str]], None]) -> FileDropFilter | None:
    """Install the filter on the running :class:`QApplication`."""
    app = QApplication.instance()
    if app is None:
        logger.warning("no QApplication yet; drag-and-drop not installed")
        return None
    handler = FileDropFilter(callback, parent=app)
    app.installEventFilter(handler)
    logger.info("drag-and-drop handler installed")
    return handler


class FileOpenCatcher(QObject):
    """Opens files handed to the application by macOS.

    Files dropped on the app icon in the Dock or Finder, or opened with *Open
    With*, never reach the command line on macOS: they arrive as ``FileOpen``
    events. The first ones come while the window is still being built, so they
    are held until :meth:`attach` gives somewhere to send them.
    """

    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)
        self._callback: Callable[[Sequence[str]], None] | None = None
        self._pending: list[str] = []
        # macOS also replays the command line as FileOpen events: the launcher
        # script itself, and any paths given there, which are opened already.
        self._from_command_line = {_normalise(arg) for arg in sys.argv}

    def attach(self, callback: Callable[[Sequence[str]], None]) -> None:
        """Start delivering to *callback*, including anything already received."""
        self._callback = callback
        if self._pending:
            QTimer.singleShot(0, self._flush)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802 (Qt naming)
        if event.type() != QEvent.FileOpen:
            return False
        path = event.file()
        if not path:
            return False
        if _normalise(path) in self._from_command_line:
            return True
        # One event per file: several dropped together are opened as one batch.
        if not self._pending and self._callback is not None:
            QTimer.singleShot(0, self._flush)
        self._pending.append(path)
        return True

    def _flush(self) -> None:
        if self._callback is None or not self._pending:
            return
        paths, self._pending = self._pending, []
        logger.info("opening %d file(s) handed over by the system", len(paths))
        try:
            self._callback(paths)
        except Exception:  # pragma: no cover - the callback reports its own errors
            logger.exception("opening %s failed", paths)


def _normalise(path: str) -> str:
    try:
        return os.path.realpath(path)
    except (OSError, ValueError):
        return path


def catch_file_open_events() -> FileOpenCatcher | None:
    """Start holding macOS ``FileOpen`` events on the running :class:`QApplication`."""
    app = QApplication.instance()
    if app is None:
        return None
    catcher = FileOpenCatcher(parent=app)
    app.installEventFilter(catcher)
    return catcher
