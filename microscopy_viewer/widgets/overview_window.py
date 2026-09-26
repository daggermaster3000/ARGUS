"""A window showing an experiment's overview, with every sample outlined on it.

The Experiment setup panel keeps overviews out of its sample grid — a stitched
map of the slide is not a sample — and opens this instead: the overview
picture, each sample's stage footprint drawn on it as a numbered box with its
name, so it is plain which fish each file is. Clicking a box selects that
sample in the grid; double-clicking opens it.

The geometry is :mod:`microscopy_viewer.overview`'s; this is only its view.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from qtpy.QtCore import QObject, QRectF, Qt, QTimer, Signal
from qtpy.QtGui import QBrush, QColor, QFont, QImage, QPainter, QPen, QPixmap, QTransform
from qtpy.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from .. import overview as ov
from ..utils import get_logger

logger = get_logger("overview_window")

#: Box colours, cycled: bright enough on a grey slide, distinct side by side.
COLORS = ("#ff5a5f", "#2ec4ff", "#ffd23f", "#7cff6b", "#ff8bf2", "#ff9f1c", "#b388ff", "#00e5c0")


def color_of(number: int) -> str:
    return COLORS[(number - 1) % len(COLORS)]


def boxes_on(mosaic: ov.Mosaic, samples: Sequence) -> list[tuple[int, object, tuple[float, float, float, float]]]:
    """``(number, sample, (x, y, w, h) in image pixels)`` for every sample with a position."""
    rows, columns = mosaic.image.shape[:2]
    out = []
    for number, sample in enumerate(samples, start=1):
        box = ov.box_from_extent(getattr(sample, "stage_extent", None))
        if box is None:
            continue
        left, top, width, height = mosaic.fractions_of(box)
        out.append((number, sample, (left * columns, top * rows, width * columns, height * rows)))
    return out


def annotated(mosaic: ov.Mosaic, samples: Sequence) -> np.ndarray:
    """The overview with each sample's box and "number name" burnt in, for saving."""
    from PIL import Image, ImageDraw

    from ..acquisition import _font

    image = Image.fromarray(np.ascontiguousarray(mosaic.image))
    draw = ImageDraw.Draw(image, "RGBA")
    font = _font(max(12, image.width // 70))
    line = max(2, image.width // 500)
    for number, sample, (x, y, w, h) in boxes_on(mosaic, samples):
        rgb = QColor(color_of(number))
        ink = (rgb.red(), rgb.green(), rgb.blue(), 255)
        draw.rectangle((x, y, x + w, y + h), outline=ink, width=line)
        text = f"{number} {getattr(sample, 'name', '')}"
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        tx, ty = x, max(0, y - (bottom - top) - 3 * line)
        draw.rectangle((tx, ty, tx + right - left + 2 * line, ty + bottom - top + 2 * line), fill=(0, 0, 0, 170))
        draw.text((tx + line - left, ty + line - top), text, font=font, fill=ink)
    return np.asarray(image)


class _Relay(QObject):
    drawn = Signal(int, object)


class _View(QGraphicsView):
    """Zooms with the wheel, pans by dragging."""

    def __init__(self, scene, parent=None):
        super().__init__(scene, parent)
        self.setRenderHint(QPainter.Antialiasing, True)
        self.setRenderHint(QPainter.SmoothPixmapTransform, True)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setBackgroundBrush(QBrush(QColor(30, 30, 30)))

    def wheelEvent(self, event):  # noqa: N802 - Qt naming
        factor = 1.2 if event.angleDelta().y() > 0 else 1 / 1.2
        self.scale(factor, factor)

    def fit(self):
        rect = self.scene().itemsBoundingRect()
        if not rect.isEmpty():
            self.fitInView(rect, Qt.KeepAspectRatio)


class _Box(QGraphicsRectItem):
    def __init__(self, rect: QRectF, number: int, sample, on_click, on_open):
        super().__init__(rect)
        self.number, self.sample = number, sample
        self._on_click, self._on_open = on_click, on_open
        pen = QPen(QColor(color_of(number)))
        pen.setWidthF(2.0)
        pen.setCosmetic(True)  # the same thickness at any zoom
        self.setPen(pen)
        self.setBrush(QBrush(QColor(0, 0, 0, 0)))
        self.setAcceptHoverEvents(True)
        self.setToolTip(f"{number}  {getattr(sample, 'name', '')}\n{getattr(sample, 'path', '')}\n"
                        "Click to select it in Experiment setup, double-click to open it.")
        self.setCursor(Qt.PointingHandCursor)

    def hoverEnterEvent(self, event):  # noqa: N802
        self.setBrush(QBrush(QColor(255, 255, 255, 50)))
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):  # noqa: N802
        self.setBrush(QBrush(QColor(0, 0, 0, 0)))
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event):  # noqa: N802
        if self._on_click is not None:
            self._on_click(self.sample)
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):  # noqa: N802
        if self._on_open is not None:
            self._on_open(self.sample)
        super().mouseDoubleClickEvent(event)


class OverviewWindow(QDialog):
    """The folder's overviews, one at a time, with their samples outlined."""

    def __init__(
        self,
        overviews: Sequence[ov.FolderOverview],
        read: Callable[[object, int], np.ndarray | None],
        on_select: Callable[[object], None] | None = None,
        on_open: Callable[[object], None] | None = None,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Overview")
        self.setModal(False)
        self.resize(1000, 720)
        self._overviews = list(overviews)
        self._read = read
        self._on_select = on_select
        self._on_open = on_open
        self._mosaics: dict[int, ov.Mosaic | None] = {}
        self._boxes: list[_Box] = []
        self._relay = _Relay()
        self._relay.drawn.connect(self._drawn)
        self._worker = None

        layout = QVBoxLayout(self)
        top = QHBoxLayout()
        top.addWidget(QLabel("Overview"))
        self._combo = QComboBox()
        for overview in self._overviews:
            kind = f"{len(overview.fields)} fields" if overview.is_mosaic else "one image"
            self._combo.addItem(f"{overview.name}  ({kind}, {len(overview.samples)} sample(s))")
        self._combo.currentIndexChanged.connect(self._show)
        top.addWidget(self._combo, stretch=1)
        layout.addLayout(top)

        splitter = QSplitter(Qt.Horizontal)
        self._scene = QGraphicsScene(self)
        self._view = _View(self._scene)
        splitter.addWidget(self._view)
        self._list = QListWidget()
        self._list.setToolTip("Click to find the sample on the overview; double-click to open it.")
        self._list.itemClicked.connect(self._list_clicked)
        self._list.itemDoubleClicked.connect(self._list_opened)
        splitter.addWidget(self._list)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 1)
        self._list.setMaximumWidth(320)
        splitter.setSizes([780, 220])
        layout.addWidget(splitter, stretch=1)

        bottom = QHBoxLayout()
        self._status = QLabel("")
        bottom.addWidget(self._status, stretch=1)
        fit = QPushButton("Fit")
        fit.clicked.connect(self._view.fit)
        bottom.addWidget(fit)
        save = QPushButton("Save image…")
        save.setToolTip("Write the overview with the boxes and names as a PNG, for a slide.")
        save.clicked.connect(self.save_image)
        bottom.addWidget(save)
        close = QPushButton("Close")
        close.clicked.connect(self.close)
        bottom.addWidget(close)
        layout.addLayout(bottom)

        self._show(0)

    def showEvent(self, event):  # noqa: N802 - Qt naming
        super().showEvent(event)
        # Fitting before the window has its size fits into nothing.
        QTimer.singleShot(0, self._view.fit)

    # -- drawing --------------------------------------------------------------

    @property
    def current(self) -> ov.FolderOverview | None:
        index = self._combo.currentIndex()
        return self._overviews[index] if 0 <= index < len(self._overviews) else None

    def _show(self, index: int) -> None:
        if index in self._mosaics:
            self._draw(index)
            return
        self._scene.clear()
        self._boxes = []
        self._list.clear()
        self._status.setText("Stitching the overview…")
        overview = self._overviews[index]
        read, relay = self._read, self._relay

        from napari.qt.threading import thread_worker

        @thread_worker
        def _job():
            try:
                return ov.render_overview(overview, read)
            except Exception:
                logger.exception("could not draw the overview %s", overview.name)
                return None

        worker = _job()
        worker.returned.connect(lambda mosaic, _i=index: relay.drawn.emit(_i, mosaic))
        self._worker = worker
        worker.start()

    def _drawn(self, index: int, mosaic) -> None:
        self._mosaics[index] = mosaic
        if index == self._combo.currentIndex():
            self._draw(index)

    def _draw(self, index: int) -> None:
        mosaic = self._mosaics.get(index)
        overview = self._overviews[index]
        self._scene.clear()
        self._boxes = []
        self._list.clear()
        if mosaic is None:
            self._status.setText("The overview could not be read.")
            return
        image = np.ascontiguousarray(mosaic.image)
        rows, columns = image.shape[:2]
        qimage = QImage(image.data, columns, rows, 3 * columns, QImage.Format_RGB888).copy()
        self._scene.addPixmap(QPixmap.fromImage(qimage))

        font = QFont()
        font.setPixelSize(13)
        font.setBold(True)
        for number, sample, (x, y, w, h) in boxes_on(mosaic, overview.samples):
            box = _Box(QRectF(x, y, w, h), number, sample, self._on_select, self._on_open)
            self._scene.addItem(box)
            self._boxes.append(box)
            label = QGraphicsSimpleTextItem(f"{number} {getattr(sample, 'name', '')}")
            label.setFont(font)
            label.setBrush(QBrush(QColor(color_of(number))))
            label.setPen(QPen(QColor(0, 0, 0, 200), 0.6))
            # Readable at any zoom: the label keeps its size on screen and sits
            # just above the box's top-left corner.
            label.setFlag(QGraphicsSimpleTextItem.ItemIgnoresTransformations, True)
            label.setPos(x, y)
            label.setTransform(QTransform.fromTranslate(0, -label.boundingRect().height() - 2))
            label.setToolTip(box.toolTip())
            self._scene.addItem(label)

            item = QListWidgetItem(f"{number}   {getattr(sample, 'name', '')}")
            item.setForeground(QBrush(QColor(color_of(number))))
            item.setToolTip(str(getattr(sample, "path", "")))
            item.setData(Qt.UserRole, number)
            self._list.addItem(item)

        self._view.fit()
        placed = len(self._boxes)
        self._status.setText(
            f"{placed} sample(s) on {overview.name}, at {mosaic.um_per_px:.1f} µm/px. "
            "Wheel to zoom, drag to pan."
        )

    # -- interaction ----------------------------------------------------------

    def _box(self, number: int) -> _Box | None:
        return next((box for box in self._boxes if box.number == number), None)

    def _list_clicked(self, item) -> None:
        box = self._box(item.data(Qt.UserRole))
        if box is None:
            return
        self._view.centerOn(box)
        for other in self._boxes:
            other.setBrush(QBrush(QColor(255, 255, 255, 60 if other is box else 0)))
        if self._on_select is not None:
            self._on_select(box.sample)

    def _list_opened(self, item) -> None:
        box = self._box(item.data(Qt.UserRole))
        if box is not None and self._on_open is not None:
            self._on_open(box.sample)

    def save_image(self, path: str | None = None):
        index = self._combo.currentIndex()
        mosaic = self._mosaics.get(index)
        if mosaic is None:
            return None
        overview = self._overviews[index]
        if not path:
            start = Path(getattr(overview.fields[0], "path", Path.home())).parent
            path, _ = QFileDialog.getSaveFileName(
                self, "Save the overview", str(start / f"{overview.name}_samples.png"), "PNG image (*.png)"
            )
            if not path:
                return None
        from PIL import Image

        Image.fromarray(annotated(mosaic, overview.samples)).save(path)
        self._status.setText(f"Saved {path}.")
        return path
