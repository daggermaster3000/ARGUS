"""Keep the main window small enough to fit, and move about, on one laptop screen.

A ``QMainWindow`` can never be made smaller than the minimum sizes of its docks
added together. Several panels here are tall forms (segmentation, registration)
and the toolbar is one long row of buttons, so on a single MacBook screen the
window's minimum came out larger than the screen itself: macOS then puts it
partly off screen, refuses to resize it, and every attempt to drag an edge makes
the dock layout fight the window manager, which is what makes it feel frozen.

The fix is to stop the panels from dictating the window's size at all: each one
is put inside a scroll area, so a dock can be squeezed to any size and its
contents scroll instead. The window is then fitted to the screen it opens on,
because napari restores whatever size it last had, possibly on a bigger monitor.
"""

from __future__ import annotations

from qtpy.QtCore import QEvent, QObject, QRect, QSize, Qt, QTimer
from qtpy.QtGui import QGuiApplication
from qtpy.QtWidgets import QFrame, QScrollArea, QSizePolicy, QStyle, QWidget


class _PanelScrollArea(QScrollArea):
    """A scroll area that asks for the size of its panel, not Qt's small default.

    The dock layout shares space out by size hint, and ``QScrollArea``'s is
    capped at a couple of dozen lines of text, which would leave a panel such as
    the experiment grid squashed on first open. It still never asks for more
    than half the screen, and its *minimum* stays tiny, which is the point.
    """

    def sizeHint(self):  # noqa: N802 - Qt naming
        hint = super().sizeHint()
        inner = self.widget()
        if inner is None:
            return hint
        wanted = inner.sizeHint()
        area = screen_rect(self)
        width, height = wanted.width(), wanted.height()
        if not area.isEmpty():
            width = min(width, area.width() // 2)
            height = min(height, area.height() // 2)
        return QSize(max(hint.width(), width), max(hint.height(), height))


def scrollable(widget: QWidget, *, vertical: bool = True) -> QScrollArea:
    """Put *widget* in a frameless scroll area so it no longer sets a minimum size.

    With ``vertical=False`` the area only scrolls sideways and keeps the height
    of *widget*: that is the toolbar, a single row that should never grow a
    vertical scroll bar.
    """
    area = _PanelScrollArea()
    area.setWidgetResizable(True)
    area.setFrameShape(QFrame.NoFrame)
    area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded if vertical else Qt.ScrollBarAlwaysOff)
    area.setWidget(widget)
    if not vertical:
        height = widget.sizeHint().height()
        style = area.style()
        # Leave room for the scroll bar, unless it is drawn over the content
        # (macOS's transient bars take no space).
        if not style.styleHint(QStyle.SH_ScrollBar_Transient, None, area):
            height += style.pixelMetric(QStyle.PM_ScrollBarExtent, None, area)
        area.setFixedHeight(height)
        area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    return area


def reveal(widget: QWidget) -> None:
    """Scroll every scroll area holding *widget* so that it is in view."""
    parent = widget.parentWidget()
    while parent is not None:
        if isinstance(parent, QScrollArea) and parent.widget() is not None:
            parent.ensureWidgetVisible(widget)
        parent = parent.parentWidget()


def screen_rect(widget: QWidget) -> QRect:
    """Available geometry (no menu bar or Dock) of the screen *widget* is mostly on."""
    screen = None
    try:
        handle = widget.window().windowHandle()
        screen = handle.screen() if handle is not None else None
    except RuntimeError:
        screen = None
    if screen is None:
        try:
            centre = widget.mapToGlobal(widget.rect().center())
            screen = QGuiApplication.screenAt(centre)
        except RuntimeError:
            screen = None
    if screen is None:
        screen = QGuiApplication.primaryScreen()
    return screen.availableGeometry() if screen is not None else QRect()


def fit_to_screen(window: QWidget | None) -> None:
    """Move and shrink a top-level *window* so its frame is on its screen."""
    if window is None or window.isFullScreen() or window.isMaximized():
        return
    area = screen_rect(window)
    if area.isEmpty():
        return
    frame = window.frameGeometry()
    if area.contains(frame):
        return
    # The frame (title bar) is extra to the client size.
    extra_w = frame.width() - window.width()
    extra_h = frame.height() - window.height()
    width = min(frame.width(), area.width())
    height = min(frame.height(), area.height())
    window.resize(max(width - extra_w, window.minimumWidth()), max(height - extra_h, window.minimumHeight()))
    x = min(max(frame.x(), area.left()), area.right() + 1 - width)
    y = min(max(frame.y(), area.top()), area.bottom() + 1 - height)
    window.move(x, y)


class _ScreenKeeper(QObject):
    """Re-fits a window whenever it could have ended up bigger than its screen.

    That is on leaving full screen or maximised, when Qt restores the normal
    geometry from before (for napari, possibly one saved on another monitor),
    and on moving to another screen, which includes an external monitor being
    unplugged from under it.
    """

    def __init__(self, window: QWidget):
        super().__init__(window)
        self._window = window
        window.installEventFilter(self)
        self._watch_screen()

    def _watch_screen(self) -> None:
        handle = self._window.windowHandle()
        if handle is not None and not getattr(self, "_watching", False):
            handle.screenChanged.connect(self._later)
            self._watching = True

    def _later(self, *_args) -> None:
        # After the platform has finished changing the window's state.
        QTimer.singleShot(0, self._fit)

    def _fit(self) -> None:
        try:
            fit_to_screen(self._window)
        except RuntimeError:  # the window has been deleted
            pass

    def eventFilter(self, obj, event):  # noqa: N802 - Qt naming
        if obj is self._window:
            kind = event.type()
            if kind == QEvent.Show:
                self._watch_screen()
                self._later()
            elif kind == QEvent.WindowStateChange:
                self._later()
        return False


def keep_on_screen(window: QWidget | None) -> None:
    """Fit *window* to its screen now and whenever it might outgrow it again."""
    if window is None:
        return
    keeper = window.findChild(_ScreenKeeper)
    if keeper is None:
        keeper = _ScreenKeeper(window)
    fit_to_screen(window)
    # Again once the docks have been laid out: they settle a moment later.
    keeper._later()
