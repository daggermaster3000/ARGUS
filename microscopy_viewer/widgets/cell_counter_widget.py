"""Dock panel: count cells by hand, into a label map the analysis can read.

Choose the image, make a counter, switch **Count** on and click each cell: every
click puts a dot — a small disc with a label of its own — on a Labels layer.
Shift-click takes a dot off; dragging still pans; Ctrl+Z undoes like any
labels edit. **Save into sample** stores the map inside the sample's ``.ims``
beside any segmentation, where the Analysis panel reads it (type its name as
the label map). See :mod:`microscopy_viewer.cell_counter`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .. import cell_counter as cc
from ..utils import get_logger
from .sample_files import close_sample, restore_view, source_of, view_state

logger = get_logger("cell_counter_widget")

#: Metadata key marking a layer as a counter, and what it counts on.
COUNTER_KEY = "mv_counter"
MODE_2D = "One plane over the stack (2D)"
MODE_3D = "Every plane (3D)"
#: Suffix of the projection layers the panel adds.
MIP_SUFFIX = "MIP"
#: Screen pixels a press may move and still be a click, not a pan.
CLICK_SLOP = 4


class CellCounterWidget(QWidget):
    """Manual cell counting on a Labels layer, saved for the analysis."""

    def __init__(self, app, parent: QWidget | None = None):
        super().__init__(parent)
        self._app = app
        self._viewer = app.viewer
        self._layer = None
        self._geometry: cc.Geometry | None = None
        self._next = 1
        self._history: list[int] = []
        self._counting = False
        self._count = 0
        self._worker = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        intro = QLabel(
            "Click each cell to put a dot on it. The dots are a label map — one "
            "object per cell — that is saved into the sample and read by the "
            "Analysis panel like a segmentation."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)
        layout.addWidget(self._build_image_box())
        layout.addWidget(self._build_count_box())
        layout.addWidget(self._build_save_box())
        self._status = QLabel("")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)
        layout.addStretch(1)

        self._recount_timer = QTimer(self)
        self._recount_timer.setSingleShot(True)
        self._recount_timer.setInterval(250)
        self._recount_timer.timeout.connect(self._recount)

        events = self._viewer.layers.events
        events.inserted.connect(self.refresh)
        events.removed.connect(self._on_removed)
        self.refresh()

    # -- construction ---------------------------------------------------------

    def _build_image_box(self) -> QGroupBox:
        box = QGroupBox("Count on")
        form = QFormLayout(box)
        form.setLabelAlignment(Qt.AlignRight)
        self._image_combo = QComboBox()
        self._image_combo.setToolTip(
            "The channel the cells are counted on. The counts are saved into its file, "
            "and the analysis measures each dot's intensity in this channel."
        )
        self._image_combo.currentIndexChanged.connect(self._image_changed)
        form.addRow("Image", self._image_combo)
        self._mode_combo = QComboBox()
        self._mode_combo.addItems([MODE_2D, MODE_3D])
        self._mode_combo.setToolTip(
            "2D: one map over the whole stack. A dot shows on every plane, so a cell is "
            "not counted twice while scrolling through Z, and the analysis measures it "
            "on the maximum projection.\n3D: a dot belongs to the plane it was put on. "
            f"Offered for stacks up to {cc.MAX_3D_VOXELS // 1_000_000} million voxels."
        )
        form.addRow("Dots", self._mode_combo)
        row = QHBoxLayout()
        self._new_button = QPushButton("New counter")
        self._new_button.setToolTip("An empty counter layer over the chosen image.")
        self._new_button.clicked.connect(self.new_counter)
        row.addWidget(self._new_button)
        self._load_button = QPushButton("Load saved counts")
        self._load_button.setToolTip("Carry on with counts saved into this sample earlier.")
        self._load_button.clicked.connect(self.load_saved)
        row.addWidget(self._load_button)
        form.addRow(row)
        self._mip_button = QPushButton("Show maximum projection")
        self._mip_button.setToolTip(
            "Add the sample's maximum projection, every channel, to count on: every "
            "cell of the stack in one picture."
        )
        self._mip_button.clicked.connect(self.show_projection)
        form.addRow(self._mip_button)
        return box

    def _build_count_box(self) -> QGroupBox:
        box = QGroupBox("Count")
        outer = QVBoxLayout(box)
        self._count_label = QLabel("No counter yet")
        # A stylesheet, not setFont: napari's own stylesheet would override the font.
        self._count_label.setStyleSheet("font-size: 22px; font-weight: bold;")
        self._count_label.setAlignment(Qt.AlignCenter)
        outer.addWidget(self._count_label)

        self._count_button = QPushButton("Count: off")
        self._count_button.setCheckable(True)
        self._count_button.setToolTip(
            "On: a click puts a dot on a cell, Shift-click takes one off. Dragging "
            "still pans, and Ctrl+Z undoes."
        )
        self._count_button.toggled.connect(self.set_counting)
        outer.addWidget(self._count_button)
        hint = QLabel("Click: add a cell · Shift-click: remove · drag: pan · Ctrl+Z: undo")
        hint.setWordWrap(True)
        hint.setAlignment(Qt.AlignCenter)
        outer.addWidget(hint)

        form = QFormLayout()
        self._diameter_spin = QDoubleSpinBox()
        self._diameter_spin.setRange(0.2, 100.0)
        self._diameter_spin.setValue(cc.DEFAULT_DIAMETER_UM)
        self._diameter_spin.setSuffix(" µm")
        self._diameter_spin.setToolTip(
            "Size of each dot. About a nucleus: big enough to see, small enough not "
            "to touch its neighbours. The analysis measures intensity inside it."
        )
        form.addRow("Dot diameter", self._diameter_spin)
        outer.addLayout(form)

        row = QHBoxLayout()
        self._undo_button = QPushButton("Undo last")
        self._undo_button.clicked.connect(self.undo_last)
        row.addWidget(self._undo_button)
        self._clear_button = QPushButton("Clear all")
        self._clear_button.clicked.connect(self.clear)
        row.addWidget(self._clear_button)
        outer.addLayout(row)
        return box

    def _build_save_box(self) -> QGroupBox:
        box = QGroupBox("Use in the analysis")
        form = QFormLayout(box)
        self._key_edit = QLineEdit(cc.DEFAULT_KEY)
        self._key_edit.setToolTip("The name the counts are stored under in the sample.")
        form.addRow("Save as", self._key_edit)
        self._save_button = QPushButton("Save into sample")
        self._save_button.setToolTip(
            "Store the dots inside the sample's .ims, beside any segmentation. The "
            "sample is closed and reopened around the write."
        )
        self._save_button.clicked.connect(self.save)
        form.addRow(self._save_button)
        note = QLabel("Then, in the Analysis panel, type this name as the <i>Label map</i>.")
        note.setWordWrap(True)
        form.addRow(note)
        return box

    # -- layers ---------------------------------------------------------------

    def _images(self) -> list:
        from napari.layers import Image

        return [layer for layer in self._viewer.layers
                if isinstance(layer, Image) and not layer.name.endswith(MIP_SUFFIX)]

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
            self._image_changed()
        except RuntimeError:  # the panel is being deleted with the viewer
            pass

    def _on_removed(self, event=None) -> None:
        try:
            if self._layer is not None and self._layer not in self._viewer.layers:
                self._forget_layer()
            self.refresh()
        except RuntimeError:
            pass

    def image_layer(self):
        name = self._image_combo.currentText()
        return self._viewer.layers[name] if name in self._viewer.layers else None

    def _image_changed(self, *_args) -> None:
        image = self.image_layer()
        three_d_ok = False
        if image is not None:
            try:
                geometry = self._geometry_for(image, three_d=True)
                three_d_ok = geometry.voxels <= cc.MAX_3D_VOXELS
            except ValueError:
                three_d_ok = False
        item = self._mode_combo.model().item(1)
        if item is not None:
            item.setEnabled(three_d_ok)
        if not three_d_ok:
            self._mode_combo.setCurrentIndex(0)
        self._update_buttons()

    def _geometry_for(self, image, three_d: bool) -> cc.Geometry:
        data = image.data[0] if getattr(image, "multiscale", False) else image.data
        pyramid = image.metadata.get("mv_pyramid")
        if pyramid:
            data = pyramid[0]
        axes = str(image.metadata.get("mv_axes", "") or "")
        # The finest level's shape; the layer's scale may belong to a coarser
        # level the 3D renderer swapped in, so the base scale is used if stored.
        scale = image.metadata.get("mv_pyramid_scale") or image.scale
        return cc.geometry_of(np.shape(data), scale, image.translate, axes, three_d)

    def _update_buttons(self) -> None:
        has_image = self.image_layer() is not None
        has_layer = self._layer is not None
        self._new_button.setEnabled(has_image)
        self._mip_button.setEnabled(has_image)
        self._load_button.setEnabled(has_image and bool(self._source()))
        for button in (self._count_button, self._undo_button, self._clear_button):
            button.setEnabled(has_layer)
        self._save_button.setEnabled(has_layer and bool(self._source()))

    def _source(self) -> str:
        image = self.image_layer()
        return source_of(image) if image is not None else ""

    # -- the counter layer ----------------------------------------------------

    def _layer_name(self) -> str:
        image = self.image_layer()
        if image is None:
            sample = "image"
        else:
            sample = Path(source_of(image)).stem if source_of(image) else image.name
        return f"{sample} — {self._key_edit.text().strip() or cc.DEFAULT_KEY}"

    def new_counter(self, *_args) -> None:
        """Put an empty counter over the chosen image.

        Takes no data: Qt's ``clicked`` passes a ``checked`` flag, which must not
        be mistaken for saved counts.
        """
        self._put_counter(None)

    def _put_counter(self, data: np.ndarray | None) -> None:
        """Put a counter holding *data* (or nothing) over the chosen image."""
        image = self.image_layer()
        if image is None:
            self._status.setText("Open an image to count on.")
            return
        three_d = self._mode_combo.currentText() == MODE_3D
        if data is not None:
            three_d = np.ndim(data) == 3
        try:
            geometry = self._geometry_for(image, three_d)
        except ValueError as exc:
            self._status.setText(str(exc))
            return
        if data is None:
            data = np.zeros(geometry.shape, dtype=geometry.dtype())
        elif tuple(np.shape(data)) != geometry.shape:
            self._status.setText(
                f"The saved counts are {np.shape(data)} and this image is {geometry.shape}; "
                "they belong to another image."
            )
            return
        if self._layer is not None and self._layer in self._viewer.layers:
            if self._has_dots() and not self._confirm("Replace the counter on screen? Its dots "
                                                       "are lost unless saved."):
                return
            self._viewer.layers.remove(self._layer)
        name = self._layer_name()
        if name in self._viewer.layers:
            self._viewer.layers.remove(name)
        kwargs = {}
        try:
            from ..loaders.layer_spec import world_units

            kwargs.update(world_units(self._viewer, geometry.ndim) or {})
        except Exception:
            logger.debug("no world units", exc_info=True)
        layer = self._viewer.add_labels(
            np.asarray(data), name=name, scale=geometry.scale, translate=geometry.translate,
            metadata={COUNTER_KEY: {"source": self._source(), "image": image.name}}, **kwargs,
        )
        self._adopt(layer, geometry)
        self._status.setText(
            f"Counter ready on {image.name}. Switch Count on and click the cells."
        )

    def _adopt(self, layer, geometry: cc.Geometry) -> None:
        self._layer = layer
        self._geometry = geometry
        self._history = []
        data = np.asarray(layer.data)
        self._next = int(data.max()) + 1 if data.size else 1
        layer.mouse_drag_callbacks.append(self._on_mouse)
        layer.events.paint.connect(self._schedule_recount)
        layer.events.data.connect(self._schedule_recount)
        self._recount()
        self._update_buttons()
        if self._counting:
            self.set_counting(True)

    def _forget_layer(self) -> None:
        self._layer = None
        self._geometry = None
        self._history = []
        self._count_button.setChecked(False)
        self._count_label.setText("No counter yet")
        self._update_buttons()

    def _has_dots(self) -> bool:
        return self._layer is not None and bool(np.any(np.asarray(self._layer.data)))

    def _confirm(self, text: str) -> bool:
        answer = QMessageBox.question(self, "Cell counter", text,
                                      QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        return answer == QMessageBox.Yes

    # -- counting -------------------------------------------------------------

    def set_counting(self, on: bool) -> None:
        self._counting = bool(on) and self._layer is not None
        self._count_button.blockSignals(True)
        self._count_button.setChecked(self._counting)
        self._count_button.blockSignals(False)
        self._count_button.setText("Count: on — click the cells" if self._counting else "Count: off")
        if self._counting:
            # Clicks only reach the selected layer, and in pan/zoom mode a drag
            # still moves the view instead of painting.
            self._viewer.layers.selection.active = self._layer
            self._layer.mode = "pan_zoom"
            if self._viewer.dims.ndisplay == 3:
                self._status.setText("Counting works in 2D view: press 2D / 3D (MIP) to go back.")

    def _on_mouse(self, layer, event):
        if not self._counting or layer is not self._layer or event.type != "mouse_press":
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

    def add_at(self, centre) -> int:
        """Put a new dot at *centre* (pixel coordinates). Returns its label, 0 if none."""
        if self._layer is None:
            return 0
        if not all(-0.5 <= c < n - 0.5 for c, n in zip(centre, self._geometry.shape)):
            self._status.setText("That is outside the image.")
            return 0
        data = np.asarray(self._layer.data)
        clicked = tuple(min(n - 1, max(0, int(round(c)))) for c, n in zip(centre, self._geometry.shape))
        if data[clicked]:
            # A second click on a counted cell is a slip, not another cell.
            self._status.setText("Already counted — Shift-click removes a dot.")
            return 0
        indices = cc.dot_indices(centre, self._diameter_spin.value(), self._geometry, data)
        if not len(indices[0]):
            self._status.setText("There is already a dot there.")
            return 0
        label = self._next
        self._layer.data_setitem(indices, label)
        self._next += 1
        self._history.append(label)
        self._shown_count(+1)
        return label

    def remove_at(self, centre) -> int:
        """Take off the dot under (or nearest within a dot of) *centre*. Returns its label."""
        if self._layer is None:
            return 0
        diameter_px = self._diameter_spin.value() / min(self._geometry.scale)
        data = np.asarray(self._layer.data)
        label = cc.label_near(data, centre, radius_px=max(2.0, diameter_px))
        if not label:
            self._status.setText("No dot there to remove.")
            return 0
        indices = cc.indices_of(data, label, centre, reach_px=max(4.0, 3 * diameter_px))
        self._layer.data_setitem(indices, 0)
        if label in self._history:
            self._history.remove(label)
        self._shown_count(-1)
        return label

    def undo_last(self) -> None:
        if not self._history or self._layer is None:
            self._status.setText("Nothing to undo here — Ctrl+Z undoes any edit.")
            return
        label = self._history.pop()
        data = np.asarray(self._layer.data)
        found = np.nonzero(data == label)
        if len(found[0]):
            self._layer.data_setitem(found, 0)
            self._shown_count(-1)

    def clear(self) -> None:
        if self._layer is None or not self._has_dots():
            return
        if not self._confirm("Remove every dot from the counter?"):
            return
        self._layer.data = np.zeros_like(np.asarray(self._layer.data))
        self._history = []
        self._next = 1
        self._recount()

    def _shown_count(self, change: int) -> None:
        # Immediate, then confirmed by a real count once the edits settle.
        self._count = max(0, self._count + change)
        self._count_label.setText(f"{self._count} cell{'s' if self._count != 1 else ''}")

    def _schedule_recount(self, *_args) -> None:
        self._recount_timer.start()

    def _recount(self) -> None:
        if self._layer is None:
            return
        data = np.asarray(self._layer.data)
        self._count = cc.count(data)
        self._count_label.setText(f"{self._count} cell{'s' if self._count != 1 else ''}")

    # -- the projection -------------------------------------------------------

    def show_projection(self) -> None:
        """Add the maximum projection of every channel of the chosen image's sample."""
        from .. import slides as sl
        from ..acquisition import stack_of

        image = self.image_layer()
        if image is None:
            return
        source = source_of(image)
        samples = sl.collect_samples(self._viewer)
        sample = next((s for s in samples if s.source and s.source == source), None) or next(
            (s for s in samples if any(c.layer_name == image.name for c in s.channels)), None)
        if sample is None:
            return
        channels = [c for c in sample.channels
                    if f"{c.layer_name} {MIP_SUFFIX}" not in self._viewer.layers]
        if not channels:
            self._status.setText("The projection is already on screen.")
            return
        self._status.setText("Projecting the stack…")
        self._mip_button.setEnabled(False)
        width = max(c.full_width for c in channels)

        from napari.qt.threading import thread_worker

        @thread_worker
        def _job():
            out = []
            for channel in channels:
                stack = stack_of(channel, width)  # the finest level
                plane = None
                for z in range(stack.depth):
                    current = stack.plane(z)
                    plane = current if plane is None else np.maximum(plane, current)
                out.append((channel, plane))
            return out

        def _done(result):
            self._mip_button.setEnabled(True)
            geometry = self._geometry_for(image, three_d=False)
            for channel, plane in result:
                layer = self._viewer.layers[channel.layer_name] if channel.layer_name in self._viewer.layers else None
                kwargs = {"name": f"{channel.layer_name} {MIP_SUFFIX}", "scale": geometry.scale,
                          "translate": geometry.translate, "blending": "additive"}
                if layer is not None:
                    kwargs["colormap"] = layer.colormap
                self._viewer.add_image(plane, **kwargs)
                if layer is not None:
                    layer.visible = False
            if self._layer is not None:
                # Dots on top of the picture they mark.
                layers = self._viewer.layers
                layers.move(layers.index(self._layer), len(layers))
            self._status.setText("Maximum projection added; the stack's own layers are hidden.")

        worker = _job()
        worker.returned.connect(_done)
        worker.errored.connect(lambda exc: (self._mip_button.setEnabled(True),
                                            self._status.setText(f"Could not project: {exc}")))
        self._worker = worker
        worker.start()

    # -- storing --------------------------------------------------------------

    def load_saved(self) -> None:
        from .. import ims_store

        source = self._source()
        key = self._key_edit.text().strip() or cc.DEFAULT_KEY
        if not source:
            return
        keys = ims_store.list_labels(source)
        wanted = key if key in keys else next((k for k in keys if k.lower() == key.lower()), None)
        if wanted is None:
            self._status.setText(f"{Path(source).name} has no counts saved as “{key}”"
                                 + (f" (it has: {', '.join(keys)})." if keys else "."))
            return
        data, _attrs = ims_store.load_labels(source, wanted)
        if data is None:
            self._status.setText(f"Could not read “{wanted}”.")
            return
        self._put_counter(np.asarray(data))
        if self._layer is not None:
            self._status.setText(f"Loaded {self._count} saved cell(s) from {Path(source).name}.")

    def save(self) -> str | None:
        """Store the dots in the sample's ``.ims``. Returns the key used."""
        from .. import ims_store

        if self._layer is None:
            return None
        source = self._source()
        if not source:
            self._status.setText("This image was not read from a file, so there is nowhere to save.")
            return None
        path = Path(source)
        ok, reason = ims_store.can_write(path)
        if not ok:
            self._status.setText(f"Cannot write: {reason}")
            return None
        key = self._key_edit.text().strip() or cc.DEFAULT_KEY
        data = np.asarray(self._layer.data)
        image = self.image_layer()
        channel = str(image.metadata.get("mv_channel_name", "") or image.name) if image is not None else ""
        geometry = self._geometry
        view = view_state(self._viewer)
        was_counting = self._counting
        reopened = close_sample(self._viewer, path)
        try:
            written = ims_store.save_labels(
                path, key, data, geometry.scale,
                attrs={"channel": channel, "source": "manual count",
                       "dot_diameter_um": float(self._diameter_spin.value())},
            )
        except Exception as exc:
            logger.exception("could not save the counts into %s", path)
            self._status.setText(f"Could not write into {path.name}: {exc}")
            written = None
        finally:
            if reopened and hasattr(self._app, "open_paths"):
                self._app.open_paths([path])
                restore_view(self._viewer, view)
            if self._layer is not None and self._layer in self._viewer.layers:
                layers = self._viewer.layers
                layers.move(layers.index(self._layer), len(layers))
            index = self._image_combo.findText(image.name if image is not None else "")
            if index >= 0:
                self._image_combo.setCurrentIndex(index)
            self.set_counting(was_counting)
        if written:
            experiment = getattr(self._app, "experiment_widget", None)
            if experiment is not None and hasattr(experiment, "_refresh_entries"):
                experiment._refresh_entries([path])
            self._status.setText(
                f"Saved {cc.count(data)} cell(s) into {path.name} as “{written}”. In the "
                f"Analysis panel, type “{written}” as the label map."
            )
        return written
