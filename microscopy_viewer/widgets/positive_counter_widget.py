"""Dock panel: count the positive cells of each channel by hand.

The table lists the sample's channels. Pick one, switch **Count** on, and click
every cell that is positive in it: each click puts a dot on that channel's
counter, a Labels layer named ``<channel> positive manual``. Pick the next channel and
carry on; every channel keeps its own dots, in its own colour. The table shows
each count as it grows, and how many of the dots lie in the junction region
when the sample has one.

**Save into sample** stores every counter as a label map beside the sample's
other label maps, where it is read back by **Load saved**, by the Analysis
panel, and by a script. The arithmetic is in
:mod:`microscopy_viewer.positive_counter`; the dots themselves are the Cell
counter panel's (:mod:`microscopy_viewer.cell_counter`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .. import cell_counter as cc
from .. import fusion as fu
from .. import positive_counter as pc
from ..exports import WORKBOOK_FILTER, default_stem
from ..utils import get_logger
from .cell_counter_widget import CLICK_SLOP, COUNTER_KEY, geometry_for
from .sample_files import close_sample, restore_view, source_of, tie_to_sample, view_state

logger = get_logger("positive_counter_widget")

TABLE_COLUMNS = ("Channel", "Positive", "In region")


@dataclass
class _Counter:
    """One channel's dots: the layer, and what is needed to add to it."""

    channel: str
    layer: object
    next: int = 1
    history: list[int] = field(default_factory=list)
    #: Every dot, by label: its centre in pixels. Kept up to date by the panel's
    #: own edits, so a click costs a lookup and not a scan of the whole map.
    centres: dict[int, np.ndarray] = field(default_factory=dict)


class PositiveCounterWidget(QWidget):
    """Hand counts of positive cells, a counter per channel, saved as label maps."""

    def __init__(self, app, parent: QWidget | None = None):
        super().__init__(parent)
        self._app = app
        self._viewer = app.viewer
        self._counters: dict[str, _Counter] = {}
        self._geometry: cc.Geometry | None = None
        self._channels: list[str] = []
        self._source = ""
        self._counting = False
        self._filling = False
        self._region_cache: tuple[str, np.ndarray | None, str] | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        intro = QLabel(
            "Pick a channel, switch Count on, and click every cell that is positive in "
            "it. Each channel keeps its own dots; they are saved with the sample as "
            "label maps."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)
        layout.addWidget(self._build_sample_box())
        layout.addWidget(self._build_count_box())
        layout.addWidget(self._build_save_box())
        self._status = QLabel("")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)
        layout.addStretch(1)

        #: Set while the panel edits a counter itself, so the layer's own
        #: change events are not taken for an edit made with napari's tools.
        self._editing = False
        #: Dots put down or taken off since the counts were last saved or loaded.
        self._unsaved = False
        self._stale: set[str] = set()
        self._region_layer_cache: tuple[int, np.ndarray | None] | None = None
        self._recount_timer = QTimer(self)
        self._recount_timer.setSingleShot(True)
        self._recount_timer.setInterval(400)
        self._recount_timer.timeout.connect(self._rebuild_stale)

        events = self._viewer.layers.events
        events.inserted.connect(self.refresh)
        events.removed.connect(self._on_removed)
        self.refresh()

    # -- construction ---------------------------------------------------------

    def _build_sample_box(self) -> QGroupBox:
        box = QGroupBox("Sample and channel")
        outer = QVBoxLayout(box)
        self._image_combo = QComboBox()
        self._image_combo.setToolTip("Any channel of the sample to count in. All of its channels are listed below.")
        self._image_combo.currentIndexChanged.connect(self._sample_changed)
        outer.addWidget(self._image_combo)

        self._table = QTableWidget(0, len(TABLE_COLUMNS))
        self._table.setHorizontalHeaderLabels(list(TABLE_COLUMNS))
        self._table.verticalHeader().setVisible(False)
        self._table.setAlternatingRowColors(False)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._table.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._table.setToolTip(
            "The row selected is the channel being counted. “In region” is the dots "
            "inside the junction region — the Fusion region layer on screen, else the "
            "region stored with the sample."
        )
        self._table.currentCellChanged.connect(self._row_changed)
        outer.addWidget(self._table)

        self._solo = QCheckBox("Show only the channel being counted")
        self._solo.setToolTip(
            "Hide the sample's other channels while a channel is counted, so positive "
            "is judged on that stain alone. Untick to see them all again."
        )
        self._solo.toggled.connect(self._apply_solo)
        outer.addWidget(self._solo)
        self._region_label = QLabel("")
        self._region_label.setWordWrap(True)
        outer.addWidget(self._region_label)
        return box

    def _build_count_box(self) -> QGroupBox:
        box = QGroupBox("Count")
        outer = QVBoxLayout(box)
        self._count_label = QLabel("No channel chosen")
        # A stylesheet, not setFont: napari's own stylesheet would override the font.
        self._count_label.setStyleSheet("font-size: 20px; font-weight: bold;")
        self._count_label.setAlignment(Qt.AlignCenter)
        outer.addWidget(self._count_label)

        self._count_button = QPushButton("Count: off")
        self._count_button.setCheckable(True)
        self._count_button.setToolTip(
            "On: a click puts a dot on a cell of the selected channel, Shift-click takes "
            "one off. Dragging still pans, and Ctrl+Z undoes."
        )
        self._count_button.toggled.connect(self.set_counting)
        outer.addWidget(self._count_button)
        hint = QLabel("Click: positive cell · Shift-click: remove · drag: pan · Ctrl+Z: undo")
        hint.setWordWrap(True)
        hint.setAlignment(Qt.AlignCenter)
        outer.addWidget(hint)

        form = QFormLayout()
        self._diameter_spin = QDoubleSpinBox()
        self._diameter_spin.setRange(0.2, 100.0)
        self._diameter_spin.setValue(cc.DEFAULT_DIAMETER_UM)
        self._diameter_spin.setSuffix(" µm")
        self._diameter_spin.setToolTip("Size of each dot: about a nucleus.")
        form.addRow("Dot diameter", self._diameter_spin)
        outer.addLayout(form)

        row = QHBoxLayout()
        self._undo_button = QPushButton("Undo last")
        self._undo_button.setToolTip("Take off the last dot put on the selected channel.")
        self._undo_button.clicked.connect(self.undo_last)
        row.addWidget(self._undo_button)
        self._clear_button = QPushButton("Clear channel")
        self._clear_button.setToolTip("Remove every dot of the selected channel.")
        self._clear_button.clicked.connect(self.clear_channel)
        row.addWidget(self._clear_button)
        outer.addLayout(row)
        return box

    def _build_save_box(self) -> QGroupBox:
        box = QGroupBox("Keep")
        outer = QVBoxLayout(box)
        row = QHBoxLayout()
        self._save_button = QPushButton("Save into sample")
        self._save_button.setToolTip(
            "Store each channel's dots with the sample as a label map named "
            "“<channel> positive manual” — inside an .ims, in the .argus.h5 beside a TIFF."
        )
        self._save_button.clicked.connect(self.save)
        row.addWidget(self._save_button)
        self._load_button = QPushButton("Load saved")
        self._load_button.setToolTip("Carry on with the hand counts stored with this sample.")
        self._load_button.clicked.connect(self.load_saved)
        row.addWidget(self._load_button)
        outer.addLayout(row)
        self._export_button = QPushButton("Export the table…")
        self._export_button.setToolTip("The counts per channel, with the counts in the region, as a workbook or CSV.")
        self._export_button.clicked.connect(self.export)
        outer.addWidget(self._export_button)
        return box

    # -- the sample -----------------------------------------------------------

    def _images(self) -> list:
        from napari.layers import Image

        return [layer for layer in self._viewer.layers
                if isinstance(layer, Image) and int(getattr(layer, "ndim", 0)) >= 2
                and not getattr(layer, "rgb", False)]

    def refresh(self, *_args) -> None:
        try:
            previous = self._image_combo.currentText()
            self._image_combo.blockSignals(True)
            self._image_combo.clear()
            for layer in self._images():
                self._image_combo.addItem(layer.name)
            index = self._image_combo.findText(previous)
            self._image_combo.setCurrentIndex(index if index >= 0 else 0)
            self._image_combo.blockSignals(False)
            self._sample_changed()
        except RuntimeError:  # the panel is being deleted with the viewer
            pass

    def _on_removed(self, event=None) -> None:
        try:
            gone = [channel for channel, counter in self._counters.items()
                    if counter.layer not in self._viewer.layers]
            for channel in gone:
                del self._counters[channel]
            self._region_cache = None
            self.refresh()
            self.update_table()
        except RuntimeError:
            pass

    def image_layer(self):
        name = self._image_combo.currentText()
        return self._viewer.layers[name] if name in self._viewer.layers else None

    def _sample_layers(self) -> list:
        """The image layers of the chosen sample, one per channel."""
        image = self.image_layer()
        if image is None:
            return []
        source = source_of(image)
        if not source:
            return [image]
        return [layer for layer in self._images() if source_of(layer) == source]

    @staticmethod
    def _channel_name(layer) -> str:
        return str(layer.metadata.get("mv_channel_name", "") or layer.name)

    def _sample_changed(self, *_args) -> None:
        image = self.image_layer()
        source = source_of(image) if image is not None else ""
        channels = [self._channel_name(layer) for layer in self._sample_layers()]
        if source != self._source or channels != self._channels:
            if source != self._source:
                # Another sample: its counters are other layers, if it has any.
                self._counters = {}
                self._region_cache = None
                self.set_counting(False)
            self._source, self._channels = source, channels
            self._geometry = None
            if image is not None:
                try:
                    self._geometry = geometry_for(image, three_d=False)
                except ValueError as exc:
                    self._status.setText(str(exc))
            self._adopt_existing()
            self._fill_table()
        self._update_buttons()

    def _adopt_existing(self) -> None:
        """Take over counter layers already on screen for this sample."""
        from napari.layers import Labels

        for layer in self._viewer.layers:
            mark = layer.metadata.get(COUNTER_KEY) if isinstance(layer, Labels) else None
            if not isinstance(mark, dict) or mark.get("source") != self._source:
                continue
            channel = mark.get("positive_channel")
            if channel in self._channels and channel not in self._counters:
                self._watch(_Counter(channel=channel, layer=layer))

    def _fill_table(self) -> None:
        self._filling = True
        try:
            row = max(self._table.currentRow(), 0)
            self._table.setRowCount(len(self._channels))
            for index, channel in enumerate(self._channels):
                for column in range(len(TABLE_COLUMNS)):
                    item = QTableWidgetItem(channel if column == 0 else "")
                    if column:
                        item.setTextAlignment(Qt.AlignCenter)
                    self._table.setItem(index, column, item)
            # Header + rows + frame: no empty space under a three-row table.
            height = self._table.horizontalHeader().height() + 2 * self._table.frameWidth()
            height += sum(self._table.rowHeight(i) for i in range(self._table.rowCount()))
            self._table.setFixedHeight(max(height, 60))
            if self._channels:
                self._table.setCurrentCell(min(row, len(self._channels) - 1), 0)
        finally:
            self._filling = False
        self.update_table()

    def _update_buttons(self) -> None:
        ready = self._geometry is not None and bool(self._channels)
        self._count_button.setEnabled(ready)
        self._undo_button.setEnabled(ready)
        self._clear_button.setEnabled(ready)
        self._export_button.setEnabled(ready)
        self._save_button.setEnabled(ready and bool(self._source))
        self._load_button.setEnabled(ready and bool(self._source))

    # -- the channel being counted --------------------------------------------

    def active_channel(self) -> str | None:
        row = self._table.currentRow()
        return self._channels[row] if 0 <= row < len(self._channels) else None

    def select_channel(self, channel: str) -> None:
        if channel in self._channels:
            self._table.setCurrentCell(self._channels.index(channel), 0)

    def _row_changed(self, *_args) -> None:
        if self._filling:
            return
        self._apply_solo()
        if self._counting:
            self._arm()
        self.update_table()

    def _apply_solo(self, *_args) -> None:
        """Show the counted channel alone, or every channel again."""
        active = self.active_channel()
        only = self._solo.isChecked() and active is not None
        for layer in self._sample_layers():
            layer.visible = (self._channel_name(layer) == active) if only else True

    def counter(self, channel: str | None = None, create: bool = False) -> _Counter | None:
        """The counter of *channel* (the selected one by default)."""
        channel = channel or self.active_channel()
        if channel is None:
            return None
        found = self._counters.get(channel)
        if found is not None or not create:
            return found
        return self._put_counter(channel, None)

    def _put_counter(self, channel: str, data: np.ndarray | None) -> _Counter | None:
        """A counter layer for *channel*, holding *data* or nothing."""
        geometry = self._geometry
        if geometry is None:
            self._status.setText("Open a sample to count in.")
            return None
        if data is None:
            data = np.zeros(geometry.shape, dtype=geometry.dtype())
        elif tuple(np.shape(data)) != geometry.shape:
            self._status.setText(
                f"The saved counts of {channel} are {np.shape(data)} and this image is "
                f"{geometry.shape}; they belong to another image."
            )
            return None
        existing = self._counters.pop(channel, None)
        if existing is not None and existing.layer in self._viewer.layers:
            self._viewer.layers.remove(existing.layer)
        kwargs = {}
        try:
            from ..loaders.layer_spec import world_units

            # The same units as the image, or napari drops units for every layer.
            kwargs.update(world_units(self._viewer, geometry.ndim) or {})
        except Exception:
            logger.debug("no world units", exc_info=True)
        layer = self._viewer.add_labels(
            np.asarray(data), name=pc.key_for(channel), scale=geometry.scale,
            translate=geometry.translate, opacity=1.0,
            metadata={COUNTER_KEY: {"source": self._source, "positive_channel": channel}}, **kwargs,
        )
        tie_to_sample(layer, self._source)
        try:
            from napari.utils.colormaps import DirectLabelColormap

            colour = pc.colour_for(self._channels.index(channel))
            layer.colormap = DirectLabelColormap(color_dict={None: colour, 0: "transparent"})
        except Exception:
            logger.debug("no single-colour labels in this napari", exc_info=True)
        counter = _Counter(channel=channel, layer=layer)
        self._watch(counter)
        return counter

    def _watch(self, counter: _Counter) -> None:
        self._rebuild(counter)
        counter.layer.mouse_drag_callbacks.append(self._on_mouse)
        channel = counter.channel
        counter.layer.events.paint.connect(lambda *_args, _c=channel: self._edited_outside(_c))
        counter.layer.events.data.connect(lambda *_args, _c=channel: self._edited_outside(_c))
        # napari's Ctrl+Z / Ctrl+Shift+Z write into the map and announce only a
        # refresh, which also fires for reasons that change nothing. Wrapping
        # the two calls catches exactly the edits they make.
        for name in ("undo", "redo"):
            original = getattr(counter.layer, name, None)
            if callable(original) and not getattr(original, "_mv_watched", False):
                def wrapped(*args, _original=original, _c=channel, **kwargs):
                    result = _original(*args, **kwargs)
                    self._edited_outside(_c)
                    return result
                wrapped._mv_watched = True
                setattr(counter.layer, name, wrapped)
        self._counters[counter.channel] = counter

    def _rebuild(self, counter: _Counter) -> None:
        """Read every dot back off the layer: on loading, and after outside edits."""
        labels, centres = pc.dot_centres(np.asarray(counter.layer.data))
        counter.centres = {int(label): centre for label, centre in zip(labels, centres)}
        counter.next = int(labels.max()) + 1 if labels.size else 1
        counter.history = [label for label in counter.history if label in counter.centres]

    def _edited_outside(self, channel: str) -> None:
        """A counter changed under napari's brush, eraser or Ctrl+Z: recount it soon."""
        if self._editing:
            return
        self._unsaved = True
        self._stale.add(channel)
        self._recount_timer.start()

    def _rebuild_stale(self) -> None:
        for channel in list(self._stale):
            counter = self._counters.get(channel)
            if counter is not None and counter.layer in self._viewer.layers:
                self._rebuild(counter)
        self._stale.clear()
        self.update_table()

    def _set(self, counter: _Counter, indices, value: int) -> None:
        """Edit a counter as the panel does: undoable, and without a recount."""
        self._editing = True
        try:
            counter.layer.data_setitem(indices, value)
        finally:
            self._editing = False
        self._unsaved = True

    # -- counting -------------------------------------------------------------

    def set_counting(self, on: bool) -> None:
        self._counting = bool(on) and self._geometry is not None and self.active_channel() is not None
        self._count_button.blockSignals(True)
        self._count_button.setChecked(self._counting)
        self._count_button.blockSignals(False)
        channel = self.active_channel() or ""
        self._count_button.setText(f"Count: on — click the {channel}+ cells" if self._counting else "Count: off")
        if self._counting:
            self._arm()

    def _arm(self) -> None:
        """Make the selected channel's counter the layer that receives clicks."""
        counter = self.counter(create=True)
        if counter is None:
            return
        # Clicks only reach the selected layer, and in pan/zoom mode a drag still
        # moves the view instead of painting.
        self._viewer.layers.selection.active = counter.layer
        counter.layer.mode = "pan_zoom"
        self._count_button.setText(f"Count: on — click the {counter.channel}+ cells")
        if self._viewer.dims.ndisplay == 3:
            self._status.setText("Counting works in 2D view: press 2D / 3D (MIP) to go back.")
        self.update_table()

    def _on_mouse(self, layer, event):
        counter = self.counter()
        if not self._counting or counter is None or layer is not counter.layer or event.type != "mouse_press":
            return
        if getattr(event, "button", 1) != 1:
            return
        start_position = tuple(event.position)
        start_pixel = np.asarray(getattr(event, "pos", (0, 0)), dtype=float)
        remove = "Shift" in {str(getattr(m, "name", m)) for m in (event.modifiers or ())}
        yield
        while event.type == "mouse_move":
            yield
        moved = np.abs(np.asarray(getattr(event, "pos", start_pixel), dtype=float) - start_pixel).max()
        if moved > CLICK_SLOP:
            return  # a pan, not a click
        if self._viewer.dims.ndisplay == 3:
            self._status.setText("Counting works in 2D view.")
            return
        centre = cc.to_data(start_position, self._geometry)
        if remove:
            self.remove_at(centre)
        else:
            self.add_at(centre)

    def add_at(self, centre, channel: str | None = None) -> int:
        """Put a dot of *channel* at *centre* (pixels). Returns its label, 0 if none."""
        counter = self.counter(channel, create=True)
        if counter is None:
            return 0
        shape = self._geometry.shape
        if not all(-0.5 <= c < n - 0.5 for c, n in zip(centre, shape)):
            self._status.setText("That is outside the image.")
            return 0
        data = np.asarray(counter.layer.data)
        clicked = tuple(min(n - 1, max(0, int(round(c)))) for c, n in zip(centre, shape))
        if data[clicked]:
            # A second click on a counted cell is a slip, not another cell.
            self._status.setText("Already counted — Shift-click removes a dot.")
            return 0
        indices = cc.dot_indices(centre, self._diameter_spin.value(), self._geometry, data)
        if not len(indices[0]):
            self._status.setText("There is already a dot there.")
            return 0
        label = counter.next
        self._set(counter, indices, label)
        counter.next += 1
        counter.history.append(label)
        counter.centres[label] = np.array([float(np.mean(axis)) for axis in indices])
        self.update_table()
        return label

    def remove_at(self, centre, channel: str | None = None) -> int:
        """Take off the dot of *channel* under or nearest *centre*. Returns its label."""
        counter = self.counter(channel)
        if counter is None:
            return 0
        diameter_px = self._diameter_spin.value() / min(self._geometry.scale)
        data = np.asarray(counter.layer.data)
        label = cc.label_near(data, centre, radius_px=max(2.0, diameter_px))
        if not label:
            self._status.setText("No dot there to remove.")
            return 0
        self._set(counter, cc.indices_of(data, label, centre, reach_px=max(4.0, 3 * diameter_px)), 0)
        if label in counter.history:
            counter.history.remove(label)
        counter.centres.pop(label, None)
        self.update_table()
        return label

    def undo_last(self, *_args) -> None:
        counter = self.counter()
        if counter is None or not counter.history:
            self._status.setText("Nothing to undo on this channel — Ctrl+Z undoes any edit.")
            return
        label = counter.history.pop()
        centre = counter.centres.pop(label, None)
        data = np.asarray(counter.layer.data)
        if centre is not None:
            reach = 2.0 * self._diameter_spin.value() / min(self._geometry.scale) + 4.0
            found = cc.indices_of(data, label, centre, reach_px=reach)
        else:
            found = np.nonzero(data == label)
        if len(found[0]):
            self._set(counter, found, 0)
        self.update_table()

    def clear_channel(self, *_args) -> None:
        counter = self.counter()
        if counter is None or not np.any(np.asarray(counter.layer.data)):
            return
        answer = QMessageBox.question(self, "Manual positive count",
                                      f"Remove every {counter.channel}+ dot?",
                                      QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if answer != QMessageBox.Yes:
            return
        self._editing = True
        try:
            counter.layer.data = np.zeros_like(np.asarray(counter.layer.data))
        finally:
            self._editing = False
        counter.history, counter.next, counter.centres = [], 1, {}
        self._unsaved = True
        self.update_table()

    # -- the region and the table ---------------------------------------------

    def region(self) -> tuple[np.ndarray | None, str]:
        """The junction region on the counters' grid, and where it came from."""
        geometry = self._geometry
        if geometry is None:
            return None, ""
        try:
            from .fusion_widget import REGION_LAYER_NAME

            if REGION_LAYER_NAME in self._viewer.layers:
                data = self._viewer.layers[REGION_LAYER_NAME].data
                # Whether it holds anything is checked once per array, not per click.
                cached = self._region_layer_cache
                if cached is None or cached[0] != id(data):
                    array = np.asarray(data)
                    usable = tuple(array.shape[-2:]) == tuple(geometry.shape[-2:]) and bool(array.any())
                    self._region_layer_cache = cached = (id(data), array if usable else None)
                if cached[1] is not None:
                    return cached[1], "the Fusion region on screen"
        except Exception:
            logger.debug("could not read the Fusion region layer", exc_info=True)
        if not self._source:
            return None, ""
        if self._region_cache is None or self._region_cache[0] != self._source:
            mask, origin = None, ""
            try:
                from .. import ims_store

                # The hand-drawn region where there is one, else the automatic band.
                for key in fu.region_keys(fu.MANUAL_KEY):
                    stored, _attrs = ims_store.load_labels(self._source, key)
                    if stored is not None and tuple(np.shape(stored)[-2:]) == tuple(geometry.shape[-2:]) and np.any(stored):
                        mask = np.asarray(stored)
                        origin = f"the region “{key}” stored with the sample"
                        break
            except Exception:
                logger.debug("could not read the stored region", exc_info=True)
            self._region_cache = (self._source, mask, origin)
        return self._region_cache[1], self._region_cache[2]

    def tallies(self) -> tuple[list[pc.Tally], np.ndarray | None, str]:
        region, origin = self.region()
        out = []
        for channel in self._channels:
            counter = self._counters.get(channel)
            if counter is not None:
                centres = np.array(list(counter.centres.values())) if counter.centres else np.zeros((0, 2))
                out.append(pc.Tally(channel=channel, count=len(counter.centres),
                                    in_region=pc.centres_in_region(centres, region)))
            else:
                out.append(pc.Tally(channel=channel, count=0, in_region=None if region is None else 0))
        return out, region, origin

    def update_table(self) -> None:
        try:
            tallies, region, origin = self.tallies()
        except Exception:
            logger.exception("could not count the dots")
            return
        for row, item in enumerate(tallies):
            if row >= self._table.rowCount():
                break
            self._table.item(row, 1).setText(str(item.count))
            self._table.item(row, 2).setText("—" if item.in_region is None else str(item.in_region))
        active = self.active_channel()
        chosen = next((item for item in tallies if item.channel == active), None)
        if chosen is None:
            self._count_label.setText("No channel chosen")
        else:
            text = f"{chosen.count} {chosen.channel}+"
            if chosen.in_region is not None:
                text += f" · {chosen.in_region} in region"
            self._count_label.setText(text)
        self._region_label.setText(f"Region: {origin}." if region is not None
                                   else "No junction region for this sample: totals only.")

    # -- keeping --------------------------------------------------------------

    def load_saved(self, *_args) -> None:
        from .. import ims_store

        if not self._source:
            return
        keys = ims_store.list_labels(self._source)
        loaded = []
        for channel in self._channels:
            key = pc.key_for(channel)
            if key not in keys:
                continue
            data, _attrs = ims_store.load_labels(self._source, key)
            if data is not None and self._put_counter(channel, np.asarray(data)) is not None:
                loaded.append(channel)
        self._unsaved = False
        self.update_table()
        if self._counting:
            self._arm()
        self._status.setText(
            f"Loaded the hand counts of {', '.join(loaded)}." if loaded
            else f"{Path(self._source).name} has no hand counts stored."
        )

    def save(self, *_args) -> list[str]:
        """Store every channel's dots with the sample. Returns the keys written."""
        from .. import ims_store

        if not self._source:
            self._status.setText("This image was not read from a file, so there is nowhere to save.")
            return []
        if not self._counters:
            self._status.setText("Nothing counted yet.")
            return []
        path = Path(self._source)
        ok, reason = ims_store.can_write(path)
        if not ok:
            self._status.setText(f"Cannot write: {reason}")
            return []
        geometry = self._geometry
        payload = {channel: np.asarray(counter.layer.data).copy() for channel, counter in self._counters.items()}
        image_name = self._image_combo.currentText()
        view = view_state(self._viewer)
        was_counting = self._counting
        # An .ims comes off screen to be written into; a TIFF's sidecar does not.
        reopened = close_sample(self._viewer, path)
        written: list[str] = []
        try:
            for channel, data in payload.items():
                written.append(ims_store.save_labels(
                    path, pc.key_for(channel), data, geometry.scale,
                    attrs={"channel": channel, "source": "manual positive count",
                           "dot_diameter_um": float(self._diameter_spin.value())},
                ))
        except Exception as exc:
            logger.exception("could not save the hand counts into %s", path)
            self._status.setText(f"Could not write into {path.name}: {exc}")
        finally:
            if reopened and hasattr(self._app, "open_paths"):
                self._app.open_paths([path])
                restore_view(self._viewer, view)
            index = self._image_combo.findText(image_name)
            if index >= 0:
                self._image_combo.setCurrentIndex(index)
            for counter in self._counters.values():
                if counter.layer in self._viewer.layers:
                    layers = self._viewer.layers
                    layers.move(layers.index(counter.layer), len(layers))
            self.set_counting(was_counting)
        if written:
            experiment = getattr(self._app, "experiment_widget", None)
            if experiment is not None and hasattr(experiment, "_refresh_entries"):
                experiment._refresh_entries([path])
            self._unsaved = False
            self._status.setText(
                f"Saved {len(written)} label map(s) with {path.name}: {', '.join(written)}."
            )
        return written

    def has_unsaved(self) -> bool:
        """Whether there are dots on screen that are not stored with the sample."""
        return self._unsaved and any(counter.centres for counter in self._counters.values()
                                     if counter.layer in self._viewer.layers)

    def settle_unsaved(self, parent=None) -> bool:
        """Before the sample goes: save the hand counts, drop them, or stay.

        Returns whether to go ahead. Asked only when there is something to lose.
        """
        if not self.has_unsaved():
            return True
        name = Path(self._source).name if self._source else "this image"
        box = QMessageBox(parent or self)
        box.setIcon(QMessageBox.Question)
        box.setWindowTitle("Manual positive count")
        box.setText(f"The hand counts on {name} have not been saved.")
        box.setInformativeText("They are removed with the sample when another one is opened.")
        save = box.addButton("Save them", QMessageBox.AcceptRole) if self._source else None
        discard = box.addButton("Discard", QMessageBox.DestructiveRole)
        box.addButton(QMessageBox.Cancel)
        box.setDefaultButton(save or discard)
        box.exec()
        clicked = box.clickedButton()
        if save is not None and clicked is save:
            return bool(self.save())
        if clicked is discard:
            self._unsaved = False
            return True
        return False

    def dataframe(self):
        tallies, region, _origin = self.tallies()
        area = None
        if region is not None and self._geometry is not None:
            area = float(np.count_nonzero(region)) * float(np.prod(self._geometry.scale[-2:])) / 1e6
        sample = Path(self._source).name if self._source else self._image_combo.currentText()
        return pc.counts_dataframe(sample, tallies, area, float(self._diameter_spin.value()), self._source)

    def export(self, path: str | None = None):
        """Write the table of counts. Asks where unless *path* is a string."""
        from ..exports import export_table

        # Qt's clicked passes a checked flag: only a real path counts as one.
        target = path if isinstance(path, str) and path else None
        if target is None:
            folder = Path(self._source).parent if self._source else Path.home()
            suggested = str(folder / f"{default_stem('manual_positive_counts')}.xlsx")
            target, _selected = QFileDialog.getSaveFileName(self, "Export the manual counts", suggested, WORKBOOK_FILTER)
            if not target:
                return None
        try:
            written = export_table(self.dataframe(), target, sheet_name="Manual counts")
        except Exception as exc:
            logger.exception("manual count export failed")
            self._status.setText(f"Could not write that file: {exc}")
            return None
        self._status.setText(f"Table written to {written}.")
        return written
