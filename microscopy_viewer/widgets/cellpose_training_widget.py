"""The Train tab of the Segmentation panel: retrain Cellpose on corrected labels.

Correct a segmentation by hand inside a few rectangles, add the rectangles to a
training set, train a new model from the one in use, and segment with it — the
new model appears in the Setup tab's model list. The work is in
:mod:`microscopy_viewer.cellpose_training`; this tab picks the layers, cuts the
crops on the main thread (a few rectangles of a lazily read scan, which is
quick) and runs the training on a ``thread_worker``.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from qtpy.QtCore import QObject, Qt, Signal
from qtpy.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .. import cellpose_training as ct
from ..utils import get_logger
from .sample_files import source_of

logger = get_logger("cellpose_training_widget")

#: The Shapes layer the training rectangles are drawn on.
REGIONS_LAYER = "Training rectangles"


class _Relay(QObject):
    """Progress text from the training thread to the GUI thread."""

    message = Signal(str)


def _plane(layer, viewer) -> tuple[np.ndarray, tuple[float, float], tuple[float, float]]:
    """The 2D plane of *layer* on screen, with its (y, x) scale and translate.

    Full resolution, never a pyramid preview. A stack gives the plane at the
    viewer's current position; a time series its current time point too.
    """
    data = layer.data[0] if getattr(layer, "multiscale", False) else layer.data
    scale = tuple(float(s) for s in layer.scale)
    translate = tuple(float(t) for t in layer.translate)
    if data.ndim > 2:
        try:
            position = layer.world_to_data(viewer.dims.point)
        except Exception:
            position = [0] * data.ndim
        index = tuple(int(np.clip(round(float(p)), 0, n - 1)) for p, n in zip(position[:-2], data.shape[:-2]))
        data = data[index]
    return data, scale[-2:], translate[-2:]


class TrainingTab(QWidget):
    """Corrected crops into a training set, and a new model out of it."""

    def __init__(self, viewer, panel, parent: QWidget | None = None):
        super().__init__(parent)
        self._viewer = viewer
        #: The Segmentation panel: its Setup tab says which channel and model.
        self._panel = panel
        self._worker = None
        self._relay = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)

        intro = QLabel(
            "Teach Cellpose your data: segment, correct the labels by hand inside a few "
            "rectangles, add them to a training set, train. <b>Inside a rectangle every "
            "nucleus has to be labelled</b> — one left out is taught as background."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)
        layout.addWidget(self._build_crops_box())
        layout.addWidget(self._build_set_box())
        layout.addWidget(self._build_train_box())
        layout.addStretch(1)

        try:
            viewer.layers.events.inserted.connect(self.refresh_layers)
            viewer.layers.events.removed.connect(self.refresh_layers)
        except Exception:  # pragma: no cover - a viewer stub without events
            pass
        self.refresh_layers()
        self._show_set()

    # -- construction ---------------------------------------------------------

    def _build_crops_box(self) -> QGroupBox:
        box = QGroupBox("1. Corrected crops")
        form = QFormLayout(box)
        form.setLabelAlignment(Qt.AlignRight)

        self._image_note = QLabel("")
        self._image_note.setWordWrap(True)
        form.addRow("Image", self._image_note)

        self._labels_box = QComboBox()
        self._labels_box.setToolTip(
            "The corrected Labels layer: usually the one a Segment run made, painted over "
            "with the Labels tools (paint the missed nuclei with a new label, erase the "
            "wrong ones, fill to split a merged pair)."
        )
        form.addRow("Labels", self._labels_box)

        draw = QPushButton("Draw rectangles")
        draw.setToolTip(
            f"Adds a “{REGIONS_LAYER}” layer in rectangle mode. Draw round the parts you "
            "corrected, a few hundred µm across: each becomes one training crop."
        )
        draw.clicked.connect(self.draw_rectangles)
        add = QPushButton("Add to the training set")
        add.setToolTip("Cut each rectangle out of the image and the labels and store the pair.")
        add.clicked.connect(self.add_rectangles)
        row = QWidget()
        buttons = QHBoxLayout(row)
        buttons.setContentsMargins(0, 0, 0, 0)
        buttons.addWidget(draw)
        buttons.addWidget(add)
        form.addRow("", row)
        return box

    def _build_set_box(self) -> QGroupBox:
        box = QGroupBox("2. Training set")
        form = QFormLayout(box)
        form.setLabelAlignment(Qt.AlignRight)

        row = QWidget()
        line = QHBoxLayout(row)
        line.setContentsMargins(0, 0, 0, 0)
        self._folder = QLineEdit(str(ct.default_root() / "nuclei"))
        self._folder.setToolTip(
            "A folder of crops (0001.tif) and their labels (0001_masks.tif): the layout the "
            "Cellpose GUI and command line read too. One folder per kind of object and stain."
        )
        self._folder.editingFinished.connect(self._show_set)
        browse = QPushButton("…")
        browse.setFixedWidth(28)
        browse.clicked.connect(self._browse)
        line.addWidget(self._folder, stretch=1)
        line.addWidget(browse)
        form.addRow("Folder", row)

        self._set_summary = QLabel("")
        self._set_summary.setWordWrap(True)
        form.addRow("", self._set_summary)

        open_button = QPushButton("Open the folder")
        open_button.clicked.connect(self._open_folder)
        form.addRow("", open_button)
        return box

    def _build_train_box(self) -> QGroupBox:
        box = QGroupBox("3. Train")
        form = QFormLayout(box)
        form.setLabelAlignment(Qt.AlignRight)

        self._base_note = QLabel("")
        self._base_note.setWordWrap(True)
        form.addRow("From", self._base_note)

        self._name = QLineEdit()
        self._name.setPlaceholderText("e.g. cpsam_assembloid_dapi")
        self._name.setToolTip(
            "Name of the new model, as it will appear in the model list. An existing model "
            "is never overwritten: a number is added instead."
        )
        form.addRow("New model", self._name)

        self._epochs = QSpinBox()
        self._epochs.setRange(1, 2000)
        self._epochs.setValue(ct.N_EPOCHS)
        self._epochs.setToolTip(
            "Passes over the training set. Cellpose's default of 100 suits a few dozen crops; "
            "watch the test loss in the status line — once it stops falling, more epochs "
            "only learn the crops by heart."
        )
        form.addRow("Epochs", self._epochs)

        self._rate = QDoubleSpinBox()
        self._rate.setDecimals(7)
        self._rate.setRange(1e-7, 1e-2)
        self._rate.setSingleStep(1e-6)
        self._rate.setValue(ct.LEARNING_RATE)
        self._rate.setToolTip(
            "How far each step moves the network. 1e-5 is Cellpose's choice for fine-tuning "
            "cpsam; much higher forgets what it knew."
        )
        form.addRow("Learning rate", self._rate)

        self._decay = QDoubleSpinBox()
        self._decay.setDecimals(3)
        self._decay.setRange(0.0, 1.0)
        self._decay.setSingleStep(0.01)
        self._decay.setValue(ct.WEIGHT_DECAY)
        self._decay.setToolTip("Pull towards small weights, against learning the crops by heart.")
        form.addRow("Weight decay", self._decay)

        self._train_button = QPushButton("Train")
        self._train_button.clicked.connect(self.train)
        form.addRow("", self._train_button)
        return box

    # -- state ----------------------------------------------------------------

    def refresh_layers(self, event=None) -> None:
        from napari.layers import Labels

        previous = self._labels_box.currentData()
        self._labels_box.clear()
        for layer in self._viewer.layers:
            if isinstance(layer, Labels):
                self._labels_box.addItem(str(layer.name), str(layer.name))
        index = self._labels_box.findData(previous)
        if index < 0 and self._labels_box.count():
            # The newest labels: what the last Segment run made.
            index = self._labels_box.count() - 1
        self._labels_box.setCurrentIndex(max(index, 0))
        self._update_notes()

    def _update_notes(self) -> None:
        image = self._image_layer()
        self._image_note.setText(
            f"{image.name} — the Setup tab's “Segment” channel" if image is not None
            else "pick the channel in the Setup tab's “Segment”")
        settings = self._panel.settings()
        self._base_note.setText(f"{Path(settings.resolved_model()).name} — the Setup tab's model")

    def showEvent(self, event) -> None:  # noqa: N802 - Qt's name
        self._update_notes()
        super().showEvent(event)

    def _image_layer(self):
        name = str(self._panel._channel_box.currentData() or "")
        return self._panel._layer_named(name)

    def _labels_layer(self):
        return self._panel._layer_named(str(self._labels_box.currentData() or ""))

    def training_set(self) -> ct.TrainingSet:
        return ct.TrainingSet(Path(self._folder.text().strip() or ct.default_root() / "nuclei"))

    def _show_set(self) -> None:
        self._set_summary.setText(self.training_set().summary())

    def _browse(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Training set folder", self._folder.text())
        if folder:
            self._folder.setText(folder)
            self._show_set()

    def _open_folder(self) -> None:
        folder = self.training_set().folder
        folder.mkdir(parents=True, exist_ok=True)
        try:
            os.startfile(str(folder))  # type: ignore[attr-defined]
        except AttributeError:  # not Windows
            import subprocess
            import sys

            subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", str(folder)])

    def _say(self, text: str) -> None:
        self._panel._status.setText(text)

    # -- crops ----------------------------------------------------------------

    def rectangles_layer(self, create: bool = False):
        try:
            return self._viewer.layers[REGIONS_LAYER]
        except (KeyError, ValueError):
            pass
        if not create:
            return None
        from ..loaders.layer_spec import world_units
        from ..measurements import NOT_A_ROI

        return self._viewer.add_shapes(
            name=REGIONS_LAYER, face_color="#00000000", edge_color="#00e5ff", edge_width=3,
            # In world µm, like the regions layer, and out of the Measurements panel.
            **(world_units(self._viewer, 2) or {"units": ("um", "um")}),
            metadata={NOT_A_ROI: True},
        )

    def draw_rectangles(self) -> None:
        layer = self.rectangles_layer(create=True)
        self._viewer.layers.selection.active = layer
        try:
            layer.mode = "add_rectangle"
        except Exception:
            logger.debug("could not switch to rectangle mode", exc_info=True)
        self._say("Draw a rectangle round each corrected area, then “Add to the training set”.")

    def add_rectangles(self) -> None:
        image_layer, labels_layer = self._image_layer(), self._labels_layer()
        rectangles = self.rectangles_layer()
        if image_layer is None or labels_layer is None:
            self._say("Pick the image (Setup tab, “Segment”) and the corrected Labels layer first.")
            return
        boxes = [np.asarray(v, float)[:, -2:] for v in (rectangles.data if rectangles is not None else [])]
        if not boxes:
            self._say("Draw at least one rectangle first (“Draw rectangles”).")
            return
        training = self.training_set()
        added, refused = [], []
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            image, image_scale, image_translate = _plane(image_layer, self._viewer)
            labels, labels_scale, labels_translate = _plane(labels_layer, self._viewer)
            for k, box in enumerate(boxes, start=1):
                crop, masks = ct.training_pair(
                    image, image_scale, image_translate, labels, labels_scale, labels_translate,
                    box.min(axis=0), box.max(axis=0))
                try:
                    pair = training.add(
                        crop, masks, source=source_of(image_layer) or str(image_layer.name),
                        channel=str(image_layer.name), labels=str(labels_layer.name),
                        box_um=[box.min(axis=0).tolist(), box.max(axis=0).tolist()],
                        voxel_um=list(image_scale))
                    added.append(pair)
                except ValueError as exc:
                    refused.append(f"rectangle {k}: {exc}")
        finally:
            QApplication.restoreOverrideCursor()
        if added and rectangles is not None:
            # Gone once stored, so the next Add does not store them twice.
            rectangles.data = []
        self._show_set()
        text = f"Added {len(added)} crop(s), {sum(p.n_objects for p in added):,} object(s). {training.summary()}."
        if refused:
            text += " Not added — " + "; ".join(refused)
        self._say(text)

    # -- training -------------------------------------------------------------

    def train(self) -> None:
        if self._worker is not None:
            self._say("A training run is already going.")
            return
        training = self.training_set()
        if not training.pairs():
            self._say("The training set is empty: add corrected crops first.")
            return
        panel_settings = self._panel.settings()
        settings = ct.TrainingSettings(
            base_model=panel_settings.resolved_model() or "cpsam",
            model_name=self._name.text().strip(),
            n_epochs=int(self._epochs.value()), learning_rate=float(self._rate.value()),
            weight_decay=float(self._decay.value()), use_gpu=panel_settings.use_gpu,
        )
        self._train_button.setEnabled(False)
        self._say(f"Training from {Path(settings.base_model).name} on {training.summary()}…")

        self._relay = _Relay()
        self._relay.message.connect(self._say)
        relay = self._relay

        def _progress(text: str) -> None:
            logger.info("training: %s", text)
            relay.message.emit(f"Training: {text}")

        from napari.qt.threading import thread_worker

        @thread_worker
        def _run():
            return ct.train(training, settings, progress=_progress)

        worker = _run()
        worker.returned.connect(self._finished)
        worker.errored.connect(self._failed)
        worker.finished.connect(self._clear)
        self._worker = worker
        worker.start()

    def _clear(self) -> None:
        self._worker = None
        self._train_button.setEnabled(True)

    def _failed(self, exc) -> None:
        logger.exception("training failed", exc_info=exc)
        self._say(f"Training failed: {exc}")
        QMessageBox.critical(self, "Microscopy Viewer", f"Training failed:\n{exc}")

    def _finished(self, result: ct.TrainingResult) -> None:
        # The new model into the Setup tab's list, and chosen, so the next Segment uses it.
        self._panel._refresh_models()
        box = self._panel._model_box
        index = box.findData(str(result.model_path))
        if index < 0:
            index = box.findText(result.model_name)
        if index >= 0:
            box.setCurrentIndex(index)
        self._update_notes()
        self._say(result.describe() + " It is now the Setup tab's model: segment again to compare.")
