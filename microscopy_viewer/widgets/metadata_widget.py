"""Dock widget showing acquisition metadata for the active layer."""

from __future__ import annotations

from qtpy.QtCore import Qt
from qtpy.QtGui import QFont
from qtpy.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..metadata import AcquisitionMetadata
from ..utils import get_logger

logger = get_logger("metadata_widget")


def metadata_for_layer(viewer, layer) -> AcquisitionMetadata | None:
    """Metadata of *layer*, or of the image a Shapes layer annotates."""
    if layer is None:
        return None
    meta = layer.metadata.get("mv_metadata")
    if meta is not None:
        return meta
    target = layer.metadata.get("mv_target_layer")
    if target and target in viewer.layers:
        return viewer.layers[target].metadata.get("mv_metadata")
    return None


class MetadataWidget(QWidget):
    """Read-only tree of acquisition settings that follows the active layer.

    The tree is rebuilt whenever the layer selection changes or layers are added
    or removed, so switching between open datasets updates it with no user action.
    """

    def __init__(self, viewer, parent: QWidget | None = None):
        super().__init__(parent)
        self._viewer = viewer
        self._current: AcquisitionMetadata | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        self._heading = QLabel("No image selected")
        heading_font = QFont(self._heading.font())
        heading_font.setBold(True)
        self._heading.setFont(heading_font)
        self._heading.setWordWrap(True)
        layout.addWidget(self._heading)

        self._tree = QTreeWidget()
        self._tree.setColumnCount(2)
        self._tree.setHeaderLabels(["Property", "Value"])
        # Alternating row colours are left off deliberately: napari's stylesheet
        # gives tree items extra padding, and the stripes are then drawn at a
        # different pitch than the rows, which reads as a blank row between every
        # entry.
        self._tree.setAlternatingRowColors(False)
        self._tree.setUniformRowHeights(True)
        self._tree.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self._tree.setTextElideMode(Qt.ElideMiddle)
        self._tree.setRootIsDecorated(True)
        layout.addWidget(self._tree, stretch=1)

        buttons = QHBoxLayout()
        copy_button = QPushButton("Copy to clipboard")
        copy_button.setToolTip("Copy the visible metadata as tab-separated text")
        copy_button.clicked.connect(self._copy)
        buttons.addWidget(copy_button)
        refresh_button = QPushButton("Refresh")
        refresh_button.clicked.connect(self.refresh)
        buttons.addWidget(refresh_button)
        buttons.addStretch(1)
        layout.addLayout(buttons)

        self._connect()
        self.refresh()

    # -- wiring ---------------------------------------------------------------

    def _connect(self) -> None:
        """Subscribe to the viewer events that should trigger a rebuild."""
        try:
            self._viewer.layers.selection.events.active.connect(self._on_event)
            self._viewer.layers.events.inserted.connect(self._on_event)
            self._viewer.layers.events.removed.connect(self._on_event)
            self._viewer.layers.events.reordered.connect(self._on_event)
        except Exception:  # pragma: no cover - napari event API drift
            logger.warning("could not connect metadata auto-refresh", exc_info=True)

    def _on_event(self, event=None) -> None:
        self.refresh()

    # -- rendering ------------------------------------------------------------

    def refresh(self) -> None:
        """Rebuild the tree from the active layer's metadata."""
        layer = self._viewer.layers.selection.active
        meta = metadata_for_layer(self._viewer, layer)
        self._current = meta
        self._tree.clear()

        if meta is None:
            self._heading.setText(
                f"No metadata for “{layer.name}”" if layer is not None else "No image selected"
            )
            return

        title = meta.image_name or "Image"
        channel = layer.metadata.get("mv_channel_name", "") if layer is not None else ""
        self._heading.setText(f"{title} — {channel}" if channel else title)

        dataset = self._section("Dataset", expanded=True)
        for label, value in meta.summary_rows():
            self._row(dataset, label, value)

        for block in meta.channels:
            rows = block.rows()
            if not rows:
                continue
            highlight = channel and block.display_name == channel
            node = self._section(
                f"Channel: {block.display_name}", expanded=bool(highlight) or len(meta.channels) == 1
            )
            for label, value in rows:
                self._row(node, label, value)

        extra = meta.extra_rows()
        if extra:
            node = self._section(f"Other file metadata ({len(extra)})", expanded=False)
            for label, value in extra:
                self._row(node, label, value)

        self._tree.resizeColumnToContents(0)

    def _section(self, title: str, expanded: bool) -> QTreeWidgetItem:
        item = QTreeWidgetItem(self._tree, [title, ""])
        font = QFont(item.font(0))
        font.setBold(True)
        item.setFont(0, font)
        item.setFirstColumnSpanned(True)
        item.setExpanded(expanded)
        return item

    @staticmethod
    def _row(parent: QTreeWidgetItem, label: str, value: str) -> QTreeWidgetItem:
        item = QTreeWidgetItem(parent, [str(label), str(value)])
        item.setToolTip(1, str(value))
        return item

    # -- actions --------------------------------------------------------------

    def _copy(self) -> None:
        """Copy the whole tree as tab-separated text for pasting into notes."""
        from qtpy.QtWidgets import QApplication

        lines: list[str] = []

        def walk(item: QTreeWidgetItem, depth: int) -> None:
            lines.append("\t".join(["  " * depth + item.text(0), item.text(1)]).rstrip())
            for index in range(item.childCount()):
                walk(item.child(index), depth + 1)

        for index in range(self._tree.topLevelItemCount()):
            walk(self._tree.topLevelItem(index), 0)

        clipboard = QApplication.clipboard()
        if clipboard is not None:
            clipboard.setText("\n".join(lines))
