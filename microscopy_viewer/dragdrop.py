"""Application-wide drag-and-drop so files dropped on the window use our readers.

napari has its own drop handling, but it routes files through plugin readers that
know nothing about ``.ims``. An application-level event filter lets us claim drops
we can handle and leave everything else to napari.
"""

from __future__ import annotations

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
