"""Dock panel: draw a line along a junction, get the band beside it as a label.

The assembloid workflow from the Fiji macro, cut down to its geometry: a line is
drawn through the interface between the two organoids, a spline is put through
it, and a band of fixed width beside it becomes a Labels layer — which is then
stored with the sample like any other label map.

The work is in :mod:`microscopy_viewer.fusion`. This module owns the two layers
(the line, and the region made from it), rebuilds the region whenever the line
or a setting changes, and saves it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .. import fusion as fu
from ..utils import get_logger
from .sample_files import close_sample, restore_view, source_of, view_state

logger = get_logger("fusion_widget")

#: The Shapes layer the junction is drawn on, and the Labels layer made from it.
LINE_LAYER_NAME = "Fusion line"
REGION_LAYER_NAME = "Fusion region"

#: Shape types that are a line rather than an area.
LINE_SHAPES = ("path", "line")


class FusionWidget(QWidget):
    """Draw the junction, expand it into a band, save the band as a label map."""

    def __init__(self, app, parent: QWidget | None = None):
        super().__init__(parent)
        self._app = app
        self._viewer = app.viewer
        self._updating = False
        self._watched = None
        #: What the region on screen was built from; ``None`` when there is none.
        self._labels: np.ndarray | None = None
        self._curve: np.ndarray | None = None
        self._voxel: tuple[float, float] = (1.0, 1.0)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        note = QLabel(
            "Draw a line along the junction between the two organoids. The band "
            "beside it becomes a label, rebuilt as the line or the settings change."
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        layout.addWidget(self._build_line_box())
        layout.addWidget(self._build_band_box())
        layout.addWidget(self._build_save_box())

        self._status = QLabel("Pick the image, then Draw the junction line.")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)
        layout.addStretch(1)

        try:
            self._viewer.layers.events.inserted.connect(self._on_layers_changed)
            self._viewer.layers.events.removed.connect(self._on_layers_changed)
        except Exception:  # pragma: no cover - a viewer stub without events
            logger.debug("could not subscribe to layer events", exc_info=True)
        self.refresh_layers()

    # -- construction ---------------------------------------------------------

    def _build_line_box(self) -> QGroupBox:
        box = QGroupBox("Junction")
        form = QFormLayout(box)
        form.setLabelAlignment(Qt.AlignRight)

        self._image_combo = QComboBox()
        self._image_combo.setToolTip(
            "The image the region is made for: it gives the pixel grid and the "
            "calibration, and the region is saved with the file it was read from."
        )
        self._image_combo.currentIndexChanged.connect(lambda _index: self.update_region())
        form.addRow("Image", self._image_combo)

        row = QWidget()
        buttons = QHBoxLayout(row)
        buttons.setContentsMargins(0, 0, 0, 0)
        draw = QPushButton("Draw the junction line")
        draw.setToolTip(
            "Click along the interface, from one end to the other; double-click (or "
            "Esc) to finish. Drawing a new line replaces the old one. Afterwards the "
            "vertices can be moved with the layer's own select-vertices tool."
        )
        draw.clicked.connect(self.start_drawing)
        buttons.addWidget(draw)
        clear = QPushButton("Clear")
        clear.setToolTip("Remove the line and the region made from it.")
        clear.clicked.connect(self.clear)
        buttons.addWidget(clear)
        form.addRow("", row)

        self._spline = QCheckBox("Fit a spline through the line")
        self._spline.setChecked(True)
        self._spline.setToolTip(
            "Join the clicked points with a smooth curve, as Fiji's Fit Spline does, "
            "so the band has no corner at every click. Off uses the straight segments."
        )
        self._spline.toggled.connect(lambda _on: self.update_region())
        form.addRow("", self._spline)
        return box

    def _build_band_box(self) -> QGroupBox:
        box = QGroupBox("Region")
        form = QFormLayout(box)
        form.setLabelAlignment(Qt.AlignRight)

        self._width = self._spin(1.0, 10000.0, fu.DEFAULT_WIDTH_UM, 10.0)
        self._width.setToolTip("How far the band reaches away from the line.")
        form.addRow("Band width", self._width)

        self._side = QComboBox()
        self._side.addItems(list(fu.SIDES))
        self._side.setToolTip(
            "Which side of the line the band goes, as seen walking along it from "
            "the first point to the last. Switch it if the band comes out on the "
            "wrong organoid — this is the macro's “flip side”.\n"
            "“Both sides” makes two bands: label 1 on the left, label 2 on the right."
        )
        self._side.currentIndexChanged.connect(lambda _index: self.update_region())
        form.addRow("Side", self._side)

        self._gap = self._spin(0.0, 1000.0, fu.DEFAULT_GAP_UM, 1.0)
        self._gap.setToolTip("Distance left between the line and the near edge of the band.")
        form.addRow("Start gap", self._gap)

        self._cap = self._spin(0.0, 10000.0, 0.0, 10.0)
        self._cap.setSpecialValueText("band width")
        self._cap.setToolTip(
            "How far the two outer corners are pushed out along the line, which "
            "splays the ends of the band. Zero uses the band width."
        )
        form.addRow("Outer cap length", self._cap)

        rebuild = QPushButton("Rebuild the region")
        rebuild.setToolTip(
            "Make the region again from the line as it is now. It is rebuilt by "
            "itself when the line or a setting changes; this is for when it was not."
        )
        rebuild.clicked.connect(self.update_region)
        form.addRow("", rebuild)
        return box

    def _build_save_box(self) -> QGroupBox:
        box = QGroupBox("Save")
        form = QFormLayout(box)
        form.setLabelAlignment(Qt.AlignRight)

        self._key_edit = QLineEdit(fu.DEFAULT_KEY)
        self._key_edit.setToolTip(
            "The name the region is stored under with the sample, beside any "
            "segmentation. Saving again under the same name replaces it."
        )
        form.addRow("Save as", self._key_edit)

        self._auto_save = QCheckBox("Save whenever the region changes")
        self._auto_save.setToolTip(
            "Store the region with the sample every time it is rebuilt.\n"
            "Leave it off for an .ims: that file has to be closed and reopened for "
            "each write. A TIFF's region goes in its sidecar and costs nothing."
        )
        form.addRow("", self._auto_save)

        save = QPushButton("Save the region with the sample")
        save.setToolTip(
            "Inside an .ims; beside a TIFF, in “<name>.argus.h5”. Stored under "
            "/ARGUS/Labels with the line it was made from."
        )
        save.clicked.connect(self.save)
        form.addRow("", save)
        return box

    def _spin(self, low: float, high: float, value: float, step: float) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(low, high)
        spin.setDecimals(1)
        spin.setSingleStep(step)
        spin.setSuffix(" µm")
        spin.setValue(value)
        spin.setKeyboardTracking(False)  # rebuild on Enter, not on every digit
        spin.valueChanged.connect(lambda _value: self.update_region())
        return spin

    # -- layers ---------------------------------------------------------------

    def _on_layers_changed(self, event=None) -> None:
        if not self._updating:
            self.refresh_layers()

    def refresh_layers(self) -> None:
        """Rebuild the image list, keeping the choice already made."""
        from napari.layers import Image

        self._updating = True
        try:
            previous = self._image_combo.currentText()
            names = [
                str(layer.name) for layer in self._viewer.layers
                if isinstance(layer, Image) and int(getattr(layer, "ndim", 0)) >= 2
            ]
            self._image_combo.blockSignals(True)
            self._image_combo.clear()
            self._image_combo.addItems(names)
            index = self._image_combo.findText(previous)
            self._image_combo.setCurrentIndex(index if index >= 0 else 0)
            self._image_combo.blockSignals(False)
        finally:
            self._updating = False
        self._watch(self.line_layer())

    def image_layer(self):
        name = self._image_combo.currentText()
        return self._viewer.layers[name] if name and name in self._viewer.layers else None

    def line_layer(self, create: bool = False):
        """The Shapes layer the junction is drawn on, created on demand."""
        if LINE_LAYER_NAME in self._viewer.layers:
            return self._viewer.layers[LINE_LAYER_NAME]
        if not create:
            return None

        from ..loaders.layer_spec import world_units
        from ..measurements import NOT_A_ROI

        self._updating = True
        try:
            layer = self._viewer.add_shapes(
                name=LINE_LAYER_NAME,
                ndim=2,
                edge_color="#ffcc00",
                edge_width=self._line_width(),
                face_color="transparent",
                # Vertices are world micrometres, like the brain regions: the
                # layer's own scale is 1 and its unit the micrometre.
                **(world_units(self._viewer, 2) or {"units": ("um", "um")}),
                metadata={NOT_A_ROI: True},
            )
        finally:
            self._updating = False
        self._watch(layer)
        return layer

    def _line_width(self) -> float:
        """An edge width that is visible on this image, in world units."""
        image = self.image_layer()
        if image is None:
            return 2.0
        scale = [float(s) for s in image.scale][-2:]
        return 6.0 * max(scale)

    def _watch(self, layer) -> None:
        """Rebuild the region whenever the line is edited."""
        if layer is self._watched:
            return
        if self._watched is not None:
            try:
                self._watched.events.data.disconnect(self._on_line_changed)
            except Exception:
                pass
        self._watched = layer
        if layer is not None:
            layer.events.data.connect(self._on_line_changed)

    def _on_line_changed(self, event=None) -> None:
        if self._updating:
            return
        layer = self.line_layer()
        # While the line is still being clicked out it grows a vertex at a time
        # and trails one under the cursor; wait until it is finished.
        if layer is not None and getattr(layer, "_is_creating", False):
            return
        self._keep_last_line(layer)
        self.update_region()

    def _keep_last_line(self, layer) -> None:
        """One junction per image: a newly drawn line replaces the earlier ones."""
        if layer is None:
            return
        lines = [i for i, kind in enumerate(layer.shape_type) if kind in LINE_SHAPES]
        if len(lines) <= 1:
            return
        self._updating = True
        try:
            layer.selected_data = set(lines[:-1])
            layer.remove_selected()
        finally:
            self._updating = False

    def start_drawing(self) -> None:
        """Select the line layer and put it in polyline mode."""
        if self.image_layer() is None:
            self._status.setText("Open an image first.")
            return
        layer = self.line_layer(create=True)
        self._viewer.layers.selection.active = layer
        for mode in ("add_path", "add_polyline"):
            try:
                layer.mode = mode
                break
            except Exception:
                continue
        self._status.setText(
            "Click along the junction; double-click to finish. The region appears "
            "when the line is finished."
        )

    def clear(self) -> None:
        self._updating = True
        try:
            for name in (LINE_LAYER_NAME, REGION_LAYER_NAME):
                if name in self._viewer.layers:
                    self._viewer.layers.remove(name)
        finally:
            self._updating = False
        self._watch(None)
        self._labels = self._curve = None
        self._status.setText("Cleared. Draw the junction line to start again.")

    # -- the region -----------------------------------------------------------

    def drawn_line(self) -> np.ndarray | None:
        """The junction as drawn, in world µm ``(y, x)``, or ``None``."""
        layer = self.line_layer()
        if layer is None:
            return None
        lines = [
            np.asarray(data, dtype=float)[:, -2:]
            for data, kind in zip(layer.data, layer.shape_type)
            if kind in LINE_SHAPES and len(data) >= 2
        ]
        if not lines:
            return None
        scale = np.asarray([float(s) for s in layer.scale][-2:])
        translate = np.asarray([float(t) for t in layer.translate][-2:])
        return lines[-1] * scale + translate

    def _grid(self, image) -> tuple[tuple[int, int], tuple[float, float], tuple[float, float]]:
        data = image.data[0] if getattr(image, "multiscale", False) else image.data
        shape = tuple(int(n) for n in data.shape[-2:])
        voxel = tuple(float(s) for s in image.scale)[-2:]
        translate = tuple(float(t) for t in image.translate)[-2:]
        return shape, voxel, translate

    def update_region(self) -> bool:
        """Rebuild the region from the line and the settings. Returns whether it did."""
        if self._updating:
            return False
        image = self.image_layer()
        line = self.drawn_line()
        if image is None or line is None:
            return False
        shape, voxel, translate = self._grid(image)
        try:
            labels, curve = fu.fusion_region(
                line, shape, voxel, translate,
                width_um=float(self._width.value()),
                gap_um=float(self._gap.value()),
                cap_um=float(self._cap.value()),
                side=self._side.currentText(),
                spline=self._spline.isChecked(),
            )
        except Exception as exc:
            logger.exception("could not build the fusion region")
            self._status.setText(f"Could not build the region: {exc}")
            return False

        self._labels, self._curve, self._voxel = labels, curve, voxel
        self._show(labels, image, voxel, translate)
        area = float(np.count_nonzero(labels)) * voxel[0] * voxel[1]
        message = (
            f"Region: {area / 1e6:.3f} mm² beside a {fu.line_length(curve):.0f} µm "
            f"junction ({self._side.currentText().lower()})."
        )
        if not labels.any():
            message = "The band falls outside the image — check the side and the width."
        self._status.setText(message)
        if self._auto_save.isChecked() and labels.any():
            self.save()
        return True

    def _show(self, labels: np.ndarray, image, voxel, translate) -> None:
        from ..loaders.layer_spec import units_like

        self._updating = True
        try:
            if REGION_LAYER_NAME in self._viewer.layers:
                layer = self._viewer.layers[REGION_LAYER_NAME]
                if tuple(layer.data.shape) == tuple(labels.shape):
                    layer.data = labels
                    layer.scale, layer.translate = voxel, translate
                    return
                self._viewer.layers.remove(layer)
            self._viewer.add_labels(
                labels, name=REGION_LAYER_NAME, scale=voxel, translate=translate,
                opacity=0.4, **units_like(image, 2),
            )
            # Keep the line on top and still the layer being edited.
            line = self.line_layer()
            if line is not None:
                layers = self._viewer.layers
                layers.move(layers.index(line), len(layers))
                layers.selection.active = line
        except Exception:
            logger.exception("could not show the fusion region")
        finally:
            self._updating = False

    # -- saving ---------------------------------------------------------------

    def save(self) -> str | None:
        """Store the region with the sample. Returns the key used."""
        from .. import ims_store

        if self._labels is None or not self._labels.any():
            self._status.setText("No region to save — draw the junction line first.")
            return None
        image = self.image_layer()
        source = source_of(image) if image is not None else ""
        if not source:
            self._status.setText("This image was not read from a file, so there is nowhere to save.")
            return None
        path = Path(source)
        ok, reason = ims_store.can_write(path)
        if not ok:
            self._status.setText(f"Cannot write: {reason}")
            return None

        key = self._key_edit.text().strip() or fu.DEFAULT_KEY
        attrs = {
            "source": "fusion line",
            "band_width_um": float(self._width.value()),
            "start_gap_um": float(self._gap.value()),
            "cap_length_um": float(self._cap.value()) or float(self._width.value()),
            "side": self._side.currentText(),
            "spline": bool(self._spline.isChecked()),
            "junction_length_um": fu.line_length(self._curve),
            # The line as drawn, so the region can be rebuilt with other settings.
            "line_yx_um": np.asarray(self.drawn_line(), dtype=np.float64),
        }
        image_name = str(image.name)
        view = view_state(self._viewer)
        self._updating = True
        # An .ims comes off screen to be written into; a TIFF's sidecar does not.
        reopened = close_sample(self._viewer, path)
        written = None
        try:
            written = ims_store.save_labels(path, key, self._labels, self._voxel, attrs=attrs)
        except Exception as exc:
            logger.exception("could not save the fusion region with %s", path)
            self._status.setText(f"Could not write into {path.name}: {exc}")
        finally:
            if reopened and hasattr(self._app, "open_paths"):
                self._app.open_paths([path])
                restore_view(self._viewer, view)
            self._updating = False
        self.refresh_layers()
        index = self._image_combo.findText(image_name)
        if index >= 0:
            self._image_combo.blockSignals(True)
            self._image_combo.setCurrentIndex(index)
            self._image_combo.blockSignals(False)
        if written:
            experiment = getattr(self._app, "experiment_widget", None)
            if experiment is not None and hasattr(experiment, "_refresh_entries"):
                experiment._refresh_entries([path])
            self._status.setText(
                f"Saved the region as “{written}” in {ims_store.store_path(path).name}."
            )
        return written
