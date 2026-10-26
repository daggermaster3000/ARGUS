"""The loading animation: one arc going round a ring.

The look of the plain CSS spinner — a faint ring and a bright arc that turns on
it, stretching and closing up as it goes — drawn by the widget itself, since a
Qt stylesheet cannot animate. No image file, so it is sharp at any size and
display scaling, and cannot be missing.

A frame is two strokes: the ring and the arc. That is a few hundredths of a
millisecond, at :data:`FRAME_MS` between frames, and the timer stops whenever
the widget is hidden.

The picture is a function of the clock and nothing else:
:meth:`Loader.paintEvent` asks what time it is and draws that instant. That
matters at startup, where the main thread is busy building the window and only
lets Qt paint now and then — an animation that advanced one step per paint
would crawl and stutter, while this one is simply in the right place whenever
it gets to draw.
"""

from __future__ import annotations

import math
import time

from qtpy.QtCore import QRectF, Qt, QTimer
from qtpy.QtGui import QColor, QPainter, QPen
from qtpy.QtWidgets import QWidget

#: Time between frames: 60 a second, which is what makes a thin line turning
#: look fluid rather than stepped.
FRAME_MS = 16

BACKGROUND = "#101014"
TRACK = "#2a2c36"
ARC = "#6ea8ff"

#: Seconds for the arc to stretch and close up once, and for the whole figure
#: to turn once. Not multiples of each other, so the arc never seems to pause.
BREATH_S = 1.4
TURN_S = 2.0

#: The arc at its shortest and longest, as a share of the circle.
SHORTEST, LONGEST = 0.04, 0.72

#: Diameter of the ring and thickness of its line, as shares of the widget.
RING = 0.30
LINE = 0.022


def _ease(t: float) -> float:
    """Slow at both ends, quick in the middle (0 to 1)."""
    return 0.5 - 0.5 * math.cos(math.pi * min(max(t, 0.0), 1.0))


def arc_at(now: float) -> tuple[float, float]:
    """Where the arc is at *now* seconds: ``(start, length)`` as shares of a turn.

    Each breath has two halves. In the first the head runs ahead and the arc
    grows; in the second the tail catches up and it shrinks. The tail has then
    moved on by what the arc grew, which is added to the steady turn — so the
    motion is continuous from one breath to the next.
    """
    cycle, phase = divmod(now / BREATH_S, 1.0)
    grown = LONGEST - SHORTEST
    if phase < 0.5:
        head = _ease(phase * 2.0)
        tail = 0.0
    else:
        head = 1.0
        tail = _ease(phase * 2.0 - 1.0)
    start = now / TURN_S + (cycle + tail) * grown
    length = SHORTEST + (head - tail) * grown
    return start % 1.0, length


class Loader(QWidget):
    """A self-painting loading spinner of a fixed size."""

    def __init__(self, size: int = 220, parent: QWidget | None = None):
        super().__init__(parent)
        self.setFixedSize(size, size)
        # Every frame covers the whole widget, so Qt need not clear it first.
        self.setAttribute(Qt.WA_OpaquePaintEvent, True)
        self._started = time.monotonic()
        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.PreciseTimer)
        self._timer.timeout.connect(self.update)

    def start(self) -> None:
        """Begin from the first frame."""
        self._started = time.monotonic()
        self._timer.start(FRAME_MS)
        self.update()

    def stop(self) -> None:
        self._timer.stop()

    def hideEvent(self, event) -> None:  # noqa: N802 - Qt's name
        # Nothing to draw, so nothing should be waking up sixty times a second.
        self._timer.stop()
        super().hideEvent(event)

    def paintEvent(self, _event) -> None:  # noqa: N802 - Qt's name
        side = float(min(self.width(), self.height()))
        diameter = side * RING
        width = max(side * LINE, 2.0)
        rect = QRectF((self.width() - diameter) / 2.0, (self.height() - diameter) / 2.0, diameter, diameter)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), QColor(BACKGROUND))
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(QColor(TRACK), width))
        painter.drawEllipse(rect)

        start, length = arc_at(time.monotonic() - self._started)
        painter.setPen(QPen(QColor(ARC), width, Qt.SolidLine, Qt.RoundCap))
        # Qt counts anticlockwise from three o'clock, in sixteenths of a degree;
        # the arc starts at twelve and runs clockwise.
        painter.drawArc(rect, int(round((90.0 - start * 360.0) * 16)), int(round(-length * 360.0 * 16)))
        painter.end()
