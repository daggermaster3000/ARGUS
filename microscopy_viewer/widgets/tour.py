"""The guided tour overlay: a spotlight on one control and a bubble explaining it.

Drawn as a child of the main window covering all of it. Everything is dimmed
except the control of the current step, which stays clickable — the overlay's
mask has a hole there — so the tour shows where to click and lets you click it.
Everything else is blocked while the tour runs, so a stray click cannot start
something the tour is not talking about.

It can be left at any point: **End tour**, or **Esc**. Panels living in their own
floating windows are highlighted where they are by a separate ring window, since
the overlay can only draw inside the main window.

The bubble is kept on the visible part of the screen: on macOS the main window is
easily taller than the screen (menu bar, Dock, a size restored from a bigger
monitor), so the window is first fitted to the screen and the bubble is then
clamped to what can actually be seen.

The script is :mod:`microscopy_viewer.onboarding`.
"""

from __future__ import annotations

from qtpy.QtCore import QEvent, QPoint, QRect, Qt, QTimer, Signal
from qtpy.QtGui import QColor, QGuiApplication, QKeySequence, QPainter, QPen, QRegion
from qtpy.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QShortcut,
    QVBoxLayout,
    QWidget,
)

from .. import onboarding as ob
from ..utils import get_logger

logger = get_logger("tour")

DIM = QColor(0, 0, 0, 150)
ACCENT = QColor("#2a78d6")
#: Space between the highlighted control and the ring drawn round it.
PADDING = 6
BUBBLE_WIDTH = 340
#: Width of the ring drawn round a control in another window.
RING = 3


def _screen_rect(widget) -> QRect:
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


def _fit_to_screen(window) -> None:
    """Move and shrink a top-level *window* so its frame is on its screen."""
    if window is None or window.isFullScreen() or window.isMaximized():
        return
    area = _screen_rect(window)
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


class _Ring(QWidget):
    """A frameless, click-through outline over a control in another window."""

    def __init__(self):
        super().__init__(
            None,
            Qt.ToolTip | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint
            | Qt.WindowTransparentForInput | Qt.WindowDoesNotAcceptFocus,
        )
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)

    def show_around(self, rect: QRect) -> None:
        """Outline the global *rect*."""
        outer = rect.adjusted(-RING - 2, -RING - 2, RING + 2, RING + 2)
        if self.geometry() != outer:
            self.setGeometry(outer)
        # Only the outline is part of the window, so clicks land on the control
        # even where the platform ignores the transparent-for-input hint.
        local = QRect(0, 0, outer.width(), outer.height())
        self.setMask(QRegion(local).subtracted(QRegion(local.adjusted(RING + 1, RING + 1, -RING - 1, -RING - 1))))
        if not self.isVisible():
            self.show()
        self.raise_()
        self.update()

    def paintEvent(self, event):  # noqa: N802 - Qt naming
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QPen(ACCENT, RING))
        painter.setBrush(Qt.NoBrush)
        painter.drawRoundedRect(self.rect().adjusted(2, 2, -2, -2), 6, 6)
        painter.end()


class TourOverlay(QWidget):
    """Runs the tour over *app*'s main window. Emits ``finished(completed)``."""

    finished = Signal(bool)

    def __init__(self, app, steps=ob.TOUR):
        window = app.viewer.window._qt_window
        super().__init__(window)
        self._app = app
        self._window = window
        self._steps = tuple(steps)
        self._index = 0
        self._target = None
        self._hole = QRect()
        self._ended = False
        self._ring = _Ring()

        self.setAttribute(Qt.WA_StyledBackground, False)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setGeometry(window.rect())
        window.installEventFilter(self)

        self._bubble = QFrame(self)
        self._bubble.setObjectName("tourBubble")
        self._bubble.setStyleSheet(
            "#tourBubble { background: #ffffff; border: 2px solid #2a78d6; border-radius: 8px; }"
            "#tourBubble QLabel { color: #1f1f1e; background: transparent; }"
            "#tourBubble QLabel#tourCounter { color: #6b6a65; }"
            "#tourBubble QPushButton { padding: 4px 10px; }"
        )
        self._bubble.setFixedWidth(BUBBLE_WIDTH)
        layout = QVBoxLayout(self._bubble)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(8)

        self._counter = QLabel("")
        self._counter.setObjectName("tourCounter")
        layout.addWidget(self._counter)

        self._title = QLabel("")
        font = self._title.font()
        font.setBold(True)
        font.setPointSize(font.pointSize() + 2)
        self._title.setFont(font)
        self._title.setWordWrap(True)
        layout.addWidget(self._title)

        self._body = QLabel("")
        self._body.setWordWrap(True)
        layout.addWidget(self._body)

        row = QHBoxLayout()
        self._end_button = QPushButton("End tour")
        self._end_button.setToolTip("Stop the tour (Esc). The Tour button starts it again.")
        self._end_button.clicked.connect(self.end)
        row.addWidget(self._end_button)
        row.addStretch(1)
        self._back_button = QPushButton("Back")
        self._back_button.clicked.connect(self.back)
        row.addWidget(self._back_button)
        self._next_button = QPushButton("Next")
        self._next_button.setDefault(True)
        self._next_button.clicked.connect(self.next)
        row.addWidget(self._next_button)
        layout.addLayout(row)

        # Esc anywhere in the window, not only when the overlay has focus: the
        # highlighted control may well have taken it.
        self._escape = QShortcut(QKeySequence(Qt.Key_Escape), window)
        self._escape.setContext(Qt.WindowShortcut)
        self._escape.activated.connect(self.end)

        # Docks move, get resized and get tabbed; follow the target.
        self._timer = QTimer(self)
        self._timer.setInterval(250)
        self._timer.timeout.connect(self._update_geometry)

    # -- control --------------------------------------------------------------

    @property
    def index(self) -> int:
        return self._index

    @property
    def target(self):
        return self._target

    def start(self, index: int = 0) -> None:
        try:
            _fit_to_screen(self._window)
        except Exception:
            logger.debug("could not fit the window to the screen", exc_info=True)
        self.setGeometry(self._window.rect())
        self.show()
        self.raise_()
        self._timer.start()
        self.go(index)

    def next(self) -> None:
        if self._index >= len(self._steps) - 1:
            self._close(completed=True)
        else:
            self.go(self._index + 1)

    def back(self) -> None:
        if self._index > 0:
            self.go(self._index - 1)

    def end(self) -> None:
        """Stop now. Safe to call more than once."""
        self._close(completed=False)

    def go(self, index: int) -> None:
        self._index = max(0, min(int(index), len(self._steps) - 1))
        step = self._steps[self._index]
        self._show_panel(step)
        self._target = self._find_target(step)

        last = self._index == len(self._steps) - 1
        self._counter.setText(f"{self._index + 1} of {len(self._steps)}")
        self._title.setText(step.title)
        self._body.setText(step.body)
        self._back_button.setEnabled(self._index > 0)
        self._next_button.setText("Finish" if last else "Next")
        self._end_button.setVisible(not last)
        self._bubble.adjustSize()
        self._update_geometry()
        self._next_button.setFocus()

    def _close(self, completed: bool) -> None:
        if self._ended:
            return
        self._ended = True
        self._timer.stop()
        self._escape.setEnabled(False)
        self._window.removeEventFilter(self)
        self._ring.hide()
        self._ring.deleteLater()
        self.hide()
        self.finished.emit(bool(completed))
        self.deleteLater()

    # -- steps ----------------------------------------------------------------

    def _show_panel(self, step) -> None:
        if not step.panel:
            return
        dock = getattr(self._app, "docks", {}).get(step.panel)
        if dock is not None:
            try:
                dock.setVisible(True)
                dock.raise_()
                if dock.isFloating():
                    _fit_to_screen(dock)
            except Exception:
                logger.debug("could not bring %s forward", step.panel, exc_info=True)
        widget = getattr(self._app, "panels", {}).get(step.panel)
        tabs = getattr(widget, "_tabs", None)
        if step.tab and tabs is not None:
            for i in range(tabs.count()):
                if tabs.tabText(i) == step.tab:
                    tabs.setCurrentIndex(i)
                    break

    def _find_target(self, step):
        widget = ob.resolve(self._app, step.target) if step.target else None
        if widget is None or not isinstance(widget, QWidget):
            if step.target:
                logger.info("tour target %s is not available; centring the step", step.target)
            return None
        return widget

    # -- geometry and drawing -------------------------------------------------

    def eventFilter(self, obj, event):  # noqa: N802 - Qt naming
        if obj is self._window and event.type() in (QEvent.Resize, QEvent.Move, QEvent.Show):
            self.setGeometry(self._window.rect())
            self._update_geometry()
        return super().eventFilter(obj, event)

    def _target_global_rect(self) -> QRect:
        """The target in screen coordinates, or empty if it cannot be seen."""
        target = self._target
        if target is None:
            return QRect()
        try:
            if not target.isVisible() or target.width() <= 0:
                return QRect()
            return QRect(target.mapToGlobal(QPoint(0, 0)), target.size())
        except RuntimeError:  # the widget was deleted under us
            self._target = None
            return QRect()

    def _in_main_window(self) -> bool:
        try:
            return self._target is not None and self._target.window() is self._window
        except RuntimeError:
            return False

    def _visible_area(self) -> QRect:
        """The part of the overlay that is on screen, in overlay coordinates."""
        screen = _screen_rect(self._window)
        if screen.isEmpty():
            return self.rect()
        local = QRect(self.mapFromGlobal(screen.topLeft()), screen.size())
        visible = local.intersected(self.rect())
        return visible if not visible.isEmpty() else self.rect()

    def _update_geometry(self) -> None:
        if self._ended:
            return
        rect = self._target_global_rect()
        if not rect.isEmpty() and self._in_main_window():
            local = QRect(self.mapFromGlobal(rect.topLeft()), rect.size())
            self._hole = local.adjusted(-PADDING, -PADDING, PADDING, PADDING).intersected(self._visible_area())
            self._ring.hide()
        else:
            # In a floating panel (its own window) or nowhere: the overlay
            # cannot reach it, so the ring window marks it and the bubble sits
            # in the middle of the screen.
            self._hole = QRect()
            if rect.isEmpty():
                self._ring.hide()
            else:
                self._ring.show_around(rect.adjusted(-PADDING, -PADDING, PADDING, PADDING))
        region = QRegion(self.rect())
        if not self._hole.isEmpty():
            region = region.subtracted(QRegion(self._hole))
        self.setMask(region)
        self._place_bubble()
        self.raise_()
        self.update()

    def _place_bubble(self) -> None:
        bubble = self._bubble
        # Height for the fixed width: wrapped labels report their one-line
        # height from sizeHint, which cuts the text off.
        height = bubble.layout().totalHeightForWidth(BUBBLE_WIDTH)
        if height <= 0:
            height = bubble.sizeHint().height()
        area = self._visible_area().adjusted(12, 12, -12, -12)
        height = min(height, area.height())
        bubble.setFixedHeight(height)
        hole = self._hole
        if hole.isEmpty():
            x = min(max(area.center().x() - bubble.width() // 2, area.left()), area.right() - bubble.width())
            y = min(max(area.center().y() - height // 2, area.top()), area.bottom() - height)
            bubble.move(max(x, area.left()), max(y, area.top()))
            return
        width, height = bubble.width(), bubble.height()
        gap = 14
        candidates = (
            QPoint(hole.right() + gap, hole.top()),              # right
            QPoint(hole.left() - gap - width, hole.top()),       # left
            QPoint(hole.left(), hole.bottom() + gap),            # below
            QPoint(hole.left(), hole.top() - gap - height),      # above
        )
        for point in candidates:
            placed = QRect(point, bubble.size())
            if area.contains(placed):
                bubble.move(point)
                return
        # Nothing fits cleanly: take the side with most room and clamp.
        point = candidates[0] if hole.center().x() < area.center().x() else candidates[1]
        x = min(max(point.x(), area.left()), area.right() - width)
        y = min(max(point.y(), area.top()), area.bottom() - height)
        placed = QRect(QPoint(x, y), bubble.size())
        if placed.intersects(hole):
            y = hole.bottom() + gap if hole.bottom() + gap + height <= area.bottom() else area.top()
        bubble.move(max(x, area.left()), max(y, area.top()))

    def paintEvent(self, event):  # noqa: N802 - Qt naming
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), DIM)  # the hole is masked out already
        if not self._hole.isEmpty():
            painter.setPen(QPen(ACCENT, 3))
            painter.setBrush(Qt.NoBrush)
            painter.drawRoundedRect(self._hole.adjusted(-2, -2, 2, 2), 6, 6)
        painter.end()

    # Clicks on the dimmed part are swallowed; the hole passes them through.
    def mousePressEvent(self, event):  # noqa: N802 - Qt naming
        event.accept()

    def mouseReleaseEvent(self, event):  # noqa: N802 - Qt naming
        event.accept()

    def wheelEvent(self, event):  # noqa: N802 - Qt naming
        event.accept()


def start_tour(app, index: int = 0) -> TourOverlay:
    """Start the tour over *app*, replacing one already running."""
    running = getattr(app, "_tour", None)
    if running is not None:
        try:
            running.end()
        except RuntimeError:
            pass
    overlay = TourOverlay(app)
    app._tour = overlay

    def _done(completed: bool) -> None:
        ob.mark_seen(finished=completed)
        if getattr(app, "_tour", None) is overlay:
            app._tour = None
        status = "Tour finished." if completed else "Tour stopped — the Tour button starts it again."
        toolbar = getattr(app, "toolbar", None)
        if toolbar is not None:
            toolbar.set_status(status)

    overlay.finished.connect(_done)
    overlay.start(index)
    return overlay
