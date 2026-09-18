"""The guided tour overlay: a spotlight on one control and a bubble explaining it.

Drawn as a child of the main window covering all of it. Everything is dimmed
except the control of the current step, which stays clickable — the overlay's
mask has a hole there — so the tour shows where to click and lets you click it.
Everything else is blocked while the tour runs, so a stray click cannot start
something the tour is not talking about.

It can be left at any point: **End tour**, or **Esc**. Panels living in their own
floating windows are highlighted where they are, since the geometry is taken in
screen coordinates.

The script is :mod:`microscopy_viewer.onboarding`.
"""

from __future__ import annotations

from qtpy.QtCore import QEvent, QPoint, QRect, Qt, QTimer, Signal
from qtpy.QtGui import QColor, QKeySequence, QPainter, QPen, QRegion
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

    def _target_rect(self) -> QRect:
        target = self._target
        if target is None:
            return QRect()
        try:
            if not target.isVisible() or target.width() <= 0:
                return QRect()
            top_left = self.mapFromGlobal(target.mapToGlobal(QPoint(0, 0)))
        except RuntimeError:  # the widget was deleted under us
            self._target = None
            return QRect()
        rect = QRect(top_left, target.size())
        return rect.adjusted(-PADDING, -PADDING, PADDING, PADDING).intersected(self.rect())

    def _update_geometry(self) -> None:
        if self._ended:
            return
        self._hole = self._target_rect()
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
        bubble.setFixedHeight(height)
        area = self.rect().adjusted(12, 12, -12, -12)
        hole = self._hole
        if hole.isEmpty():
            bubble.move(area.center() - bubble.rect().center())
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
        point = candidates[0] if hole.center().x() < self.width() / 2 else candidates[1]
        x = min(max(point.x(), area.left()), area.right() - width)
        y = min(max(point.y(), area.top()), area.bottom() - height)
        placed = QRect(QPoint(x, y), bubble.size())
        if placed.intersects(hole):
            y = hole.bottom() + gap if hole.bottom() + gap + height <= area.bottom() else area.top()
        bubble.move(x, y)

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
