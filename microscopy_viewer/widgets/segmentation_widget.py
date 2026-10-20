"""Dock panel: segment cells or nuclei with Cellpose and measure what came out.

Self-contained — pulled in only by its entry in
:mod:`microscopy_viewer.widgets.registry`, so ``cellpose`` and torch stay off the
startup path and a machine without them still gets every other panel.

The work lives in :mod:`microscopy_viewer.segmentation`; this module picks the
channel, snapshots it on the main thread, hands the snapshot to a napari
``thread_worker`` — a 3D stack takes minutes even on a GPU and the window has to
stay usable — and turns the label map that comes back into a Labels layer and a
table.

Two settings are worth reading the tooltips for. **Diameter is in µm**, converted
to Cellpose's pixels using the layer's calibrated scale, so the same number works
across objectives. And **the device is shown before the run**, not after: Cellpose
falls back to the CPU without complaining, and that is the difference between tens
of seconds and tens of minutes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from qtpy.QtCore import QObject, Qt, Signal
from qtpy.QtWidgets import (
    QCheckBox,
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
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .. import segmentation as sg
from ..exports import default_stem, export_table
from ..utils import format_number, get_logger

logger = get_logger("segmentation_widget")

#: Suffix on the Labels layer a run produces, so a second run on another channel
#: is distinguishable and the source stays obvious in the layer list.
LABELS_SUFFIX = "labels"
#: The same objects grown for intensity (see ``Expand for intensity``).
EXPANDED_LABELS_SUFFIX = "expanded labels"
#: The same, when the growth followed a membrane channel: these are cells, and
#: calling them an expansion would undersell what is on screen.
CELL_LABELS_SUFFIX = "cells"

MODEL_FILTER = "Cellpose models (*);;All files (*)"

#: Shown in the "measure" box for "whatever was segmented".
SAME_CHANNEL = "— the segmented channel —"
#: Entry for "segment one channel on its own", the default.
NO_NUCLEI = "— none —"
#: Entry for "grow by distance rather than along a stain", the default.
NO_MEMBRANE = "— none —"
#: Shown in the training tab when the viewer holds no Labels layer to learn from.
NO_MASKS = "— no corrected masks in the viewer —"


def _short_name(layer) -> str:
    """A readable label for a layer whose name is an acquisition path.

    Imaris records the acquiring machine's own path as the image name, so a layer
    arrives called ``D:\\Transfer\\2026-08-05\\fish_4.ims :: Confocal - dapi``,
    which is three quarters useless in a narrow dock.
    """
    name = str(getattr(layer, "name", ""))
    head, separator, tail = name.partition(" :: ")
    stem = Path(head.replace("\\", "/")).stem or head
    return f"{stem} :: {tail}" if separator else stem


class _TrainingVoice(QObject):
    """Carries cellpose's training log from the worker thread to the panel.

    A plain callback cannot touch a widget from the worker thread. A signal can:
    Qt queues it onto the thread the receiver lives on. Without it the panel
    says nothing for the length of a run, which is the state people kill.
    """

    said = Signal(str)


class SegmentationWidget(QWidget):
    """Pick a channel, segment it with Cellpose, measure the objects."""

    def __init__(self, viewer, parent: QWidget | None = None):
        super().__init__(parent)
        self._viewer = viewer
        self._worker = None
        self._trainer = None
        self._result: sg.SegmentationResult | None = None
        self._stats: list[sg.ObjectStat] = []
        self._updating = False
        self._voice = _TrainingVoice(self)
        self._voice.said.connect(self._training_said)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self._backend_notice = QLabel("")
        self._backend_notice.setWordWrap(True)
        self._backend_notice.setVisible(False)
        layout.addWidget(self._backend_notice)

        self._tabs = QTabWidget()
        self._tabs.addTab(self._build_setup_tab(), "Setup")
        self._tabs.addTab(self._build_objects_tab(), "Objects")
        self._tabs.addTab(self._build_training_tab(), "Training")
        layout.addWidget(self._tabs, stretch=1)

        self._status = QLabel("Pick a channel, set the object diameter in µm, then Segment.")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        self._connect_viewer()
        self.refresh_layers()
        self._check_backend()
        self.refresh_device()

    # -- construction ---------------------------------------------------------

    def _build_setup_tab(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(2, 2, 2, 2)

        outer.addWidget(self._build_channel_box())
        outer.addWidget(self._build_model_box())
        outer.addWidget(self._build_device_box())

        self._run_button = QPushButton("Segment")
        self._run_button.clicked.connect(self.run)
        outer.addWidget(self._run_button)
        outer.addStretch(1)
        return page

    def _build_channel_box(self) -> QGroupBox:
        box = QGroupBox("Channels")
        form = QFormLayout(box)
        form.setLabelAlignment(Qt.AlignRight)

        self._channel_box = QComboBox()
        self._channel_box.setToolTip(
            "The channel segmented. A nuclear stain with the nuclear model is the reliable "
            "case; a membrane or cytoplasmic stain wants a cyto model."
        )
        form.addRow("Segment", self._channel_box)

        self._nuclei_box = QComboBox()
        self._nuclei_box.setToolTip(
            "Optional second stain, handed to Cellpose alongside the segmented one. Segment a "
            "membrane or cytoplasmic channel and give it the nuclear stain here to get whole "
            "cells: the nuclei separate cells that touch, and every mask has one nucleus. "
            "Leave it at “none” to segment a single channel on its own."
        )
        form.addRow("Nuclei", self._nuclei_box)

        self._membrane_box = QComboBox()
        self._membrane_box.setToolTip(
            "Optional membrane or cortical stain used to turn the segmented nuclei into "
            "cells. It is not segmented and never reaches Cellpose: each nucleus is grown "
            "outwards afterwards and stops where the stain is bright or where it meets a "
            "neighbour, so a cell follows its own membrane instead of a fixed radius.\n\n"
            "Segment DAPI, give the membrane here, and set “Expand for intensity” to how "
            "far a cell may reach from its nucleus — that distance still caps the growth, "
            "and without it nothing is grown at all. Every cell keeps the label of its "
            "nucleus, so the two layers line up object for object.\n\n"
            "This is the seeded watershed of the Pelkmans lab's "
            "abbott-segmentation-tasks. Leave it at “none” to grow by distance alone."
        )
        form.addRow("Membrane", self._membrane_box)

        self._measure_box = QComboBox()
        self._measure_box.setToolTip(
            "The channel the per-object intensities are read from. Segment on DAPI and measure "
            "on the reporter to get signal per nucleus."
        )
        form.addRow("Measure", self._measure_box)

        refresh = QPushButton("Refresh from the layer list")
        refresh.clicked.connect(self.refresh_layers)
        form.addRow("", refresh)
        return box

    def _build_model_box(self) -> QGroupBox:
        box = QGroupBox("Cellpose")
        form = QFormLayout(box)
        form.setLabelAlignment(Qt.AlignRight)

        model_row = QWidget()
        model_layout = QHBoxLayout(model_row)
        model_layout.setContentsMargins(0, 0, 0, 0)
        self._model_box = QComboBox()
        self._model_box.setToolTip(
            "Cellpose 4 ships one generalist model (cpsam); cellpose 3 has the zoo — “nuclei” "
            "for a nuclear stain, “cyto3” for everything else. Models you trained in the "
            "Cellpose GUI, and weights sitting in ~/.cellpose/models, are listed too."
        )
        model_layout.addWidget(self._model_box, stretch=1)
        rescan = QPushButton("↻")
        rescan.setMaximumWidth(30)
        rescan.setToolTip("Look for models again — after training one, or dropping weights in.")
        rescan.clicked.connect(self._refresh_models)
        model_layout.addWidget(rescan)
        form.addRow("Model", model_row)

        self._custom_edit, custom_row = self._path_row(
            "A model you trained yourself. Overrides the model above when it is set."
        )
        form.addRow("Custom model", custom_row)

        self._mode_box = QComboBox()
        self._mode_box.addItems(list(sg.MODES))
        self._mode_box.setToolTip(
            "“2D + stitch” segments each plane and joins overlapping masks between planes — "
            "much faster, and usually better on an anisotropic stack. “3D” computes flows in "
            "3D and is the only mode that handles an object interrupted between planes."
        )
        self._mode_box.currentTextChanged.connect(self._update_mode_sensitivity)
        form.addRow("Mode", self._mode_box)

        self._diameter = QDoubleSpinBox()
        self._diameter.setRange(0.0, 500.0)
        self._diameter.setDecimals(2)
        self._diameter.setSingleStep(0.5)
        self._diameter.setSuffix(" µm")
        self._diameter.setSpecialValueText("automatic")
        self._diameter.setToolTip(
            "Expected object diameter, in µm. Converted to Cellpose's pixels with the layer's "
            "own voxel size, so the same number holds across objectives. Zero lets Cellpose "
            "decide, which is worth overriding — the diameter is the setting that matters most."
        )
        form.addRow("Diameter", self._diameter)

        self._flow_threshold = QDoubleSpinBox()
        self._flow_threshold.setRange(0.0, 3.0)
        self._flow_threshold.setSingleStep(0.1)
        self._flow_threshold.setValue(sg.SegmentationSettings.flow_threshold)
        self._flow_threshold.setToolTip(
            "Flow error threshold: lower keeps fewer and rounder objects, higher keeps more "
            "ragged ones. Ignored in 3D mode."
        )
        form.addRow("Flow threshold", self._flow_threshold)

        self._cellprob = QDoubleSpinBox()
        self._cellprob.setRange(-6.0, 6.0)
        self._cellprob.setSingleStep(0.5)
        self._cellprob.setValue(sg.SegmentationSettings.cellprob_threshold)
        self._cellprob.setToolTip(
            "Mask probability cut. Lower it to find more and larger objects in a dim stack; "
            "raise it when background is being labelled."
        )
        form.addRow("Cell probability", self._cellprob)

        self._stitch = QDoubleSpinBox()
        self._stitch.setRange(0.0, 1.0)
        self._stitch.setSingleStep(0.05)
        self._stitch.setValue(sg.SegmentationSettings.stitch_threshold)
        self._stitch.setToolTip(
            "How much two masks in neighbouring planes have to overlap to become one object. "
            "Only used in “2D + stitch”."
        )
        form.addRow("Stitch threshold", self._stitch)

        self._min_size = QSpinBox()
        self._min_size.setRange(0, 100_000)
        self._min_size.setValue(sg.DEFAULT_MIN_SIZE)
        self._min_size.setSuffix(" px")
        self._min_size.setToolTip("Objects smaller than this are dropped by Cellpose itself.")
        form.addRow("Minimum size", self._min_size)

        self._max_solidity = QDoubleSpinBox()
        self._max_solidity.setRange(0.0, 1.0)
        self._max_solidity.setDecimals(3)
        self._max_solidity.setSingleStep(0.005)
        self._max_solidity.setValue(0.0)
        self._max_solidity.setSpecialValueText("off")
        self._max_solidity.setToolTip(
            "Drop objects whose solidity — area over the area of the convex hull — is above "
            "this. It is a way of throwing out false positives that come back as clean round "
            "discs: debris, beads and out-of-focus blobs are convex, while a real nucleus "
            "packed against its neighbours is dented by them.\n\n"
            "Set it to 1.000 to measure the shape of every object without dropping any, look "
            "at the Solidity column in the Objects tab, and pick the cut from what is there. "
            "Needs scikit-image, which comes with cellpose."
        )
        form.addRow("Max solidity", self._max_solidity)

        self._expand = QDoubleSpinBox()
        self._expand.setRange(0.0, 100.0)
        self._expand.setDecimals(1)
        self._expand.setSingleStep(0.5)
        self._expand.setValue(0.0)
        self._expand.setSuffix(" µm")
        self._expand.setSpecialValueText("off")
        self._expand.setToolTip(
            "Grow every object by up to this far and measure intensities there as well. A "
            "reporter in the cytoplasm is not under the nucleus that was segmented; growing "
            "the nucleus reaches it. Neighbours split the gap between them (a watershed), so "
            "no object takes another's.\n\n"
            "Every channel is then measured three ways, each with its own columns:\n"
            "• the object itself (the nucleus) — the columns there always were;\n"
            "• “expansion” — only what growing added, the ring round the nucleus;\n"
            "• “nucleus + expansion” — the two together.\n\n"
            "Size, position and shape stay those of the object as segmented. The grown "
            "objects are added as a second labels layer. The extra columns are in the "
            "exported table; the Objects tab shows the object's own."
        )
        form.addRow("Expand for intensity", self._expand)

        self._membrane_smooth = QDoubleSpinBox()
        self._membrane_smooth.setRange(0.0, 20.0)
        self._membrane_smooth.setDecimals(1)
        self._membrane_smooth.setSingleStep(0.5)
        self._membrane_smooth.setValue(0.0)
        self._membrane_smooth.setSuffix(" µm")
        self._membrane_smooth.setSpecialValueText("off")
        self._membrane_smooth.setToolTip(
            "Blur the membrane channel before growing along it. Only used when a membrane "
            "channel is picked.\n\n"
            "A membrane stain is a pixel or two wide and noisy, and a single bright speck "
            "inside a cell stops that cell's growth where the speck is. Blurring by about "
            "the membrane's own width is usually enough to stop that without moving the "
            "membrane. Intensities are still measured on the raw channel."
        )
        form.addRow("Smooth the membrane", self._membrane_smooth)

        self._membrane_compactness = QDoubleSpinBox()
        self._membrane_compactness.setRange(0.0, 1000.0)
        self._membrane_compactness.setDecimals(1)
        self._membrane_compactness.setSingleStep(10.0)
        self._membrane_compactness.setValue(0.0)
        self._membrane_compactness.setSpecialValueText("off")
        self._membrane_compactness.setToolTip(
            "Bias the cells towards round shapes. Only used when a membrane channel is "
            "picked.\n\n"
            "Zero lets the stain decide the whole outline, which is right when the membrane "
            "is continuous. Raise it when the stain has gaps and cells leak through them "
            "into their neighbours — the growth then prefers to stay near its own nucleus. "
            "It is skimage's compact watershed, and the one advanced parameter the "
            "abbott-segmentation-tasks seeded watershed exposes."
        )
        form.addRow("Cell compactness", self._membrane_compactness)

        self._median = QSpinBox()
        self._median.setRange(0, 10)
        self._median.setValue(0)
        self._median.setSuffix(" px")
        self._median.setSpecialValueText("off")
        self._median.setToolTip(
            "Median-filter the channels before segmenting them, with a square window of "
            "2r+1 px. It removes shot noise and hot pixels without moving edges, which is "
            "what a Gaussian would do.\n\n"
            "Only what Cellpose sees is filtered: the labels come back on the original grid "
            "and intensities are still measured on the raw channel. It is not cheap — on a "
            "12000 x 12000 well, radius 1 costs about 12 s and radius 3 about 70 s, per image."
        )
        self._median.valueChanged.connect(self._median_changed)
        form.addRow("Median filter", self._median)

        self._normalize = QCheckBox("Percentile-normalise before segmenting")
        self._normalize.setChecked(True)
        self._normalize.setToolTip(
            "Cellpose is trained on normalised input; turn this off only for data that is "
            "already scaled."
        )
        form.addRow("Normalise", self._normalize)

        self._add_layer = QCheckBox("Add the labels to the viewer")
        self._add_layer.setChecked(True)
        form.addRow("Results", self._add_layer)

        self._update_mode_sensitivity(self._mode_box.currentText())
        return box

    def _build_device_box(self) -> QGroupBox:
        box = QGroupBox("Compute")
        form = QFormLayout(box)
        form.setLabelAlignment(Qt.AlignRight)

        self._use_gpu = QCheckBox("Use the GPU when one is available")
        self._use_gpu.setChecked(True)
        self._use_gpu.toggled.connect(lambda _on: self.refresh_device())
        form.addRow("GPU", self._use_gpu)

        device_row = QWidget()
        device_layout = QHBoxLayout(device_row)
        device_layout.setContentsMargins(0, 0, 0, 0)
        self._device_label = QLabel("—")
        self._device_label.setWordWrap(True)
        device_layout.addWidget(self._device_label, stretch=1)
        recheck = QPushButton("Re-check")
        recheck.setToolTip("Ask torch again — useful after freeing video memory elsewhere.")
        recheck.clicked.connect(self.refresh_device)
        device_layout.addWidget(recheck)
        form.addRow("Device", device_row)

        self._batch_size = QSpinBox()
        self._batch_size.setRange(0, 256)
        self._batch_size.setValue(0)
        self._batch_size.setSpecialValueText("automatic")
        self._batch_size.setToolTip(
            "Patches per forward pass. Automatic sizes it from free video memory; lower it by "
            "hand if a run runs out of memory part-way through."
        )
        form.addRow("Batch size", self._batch_size)

        self._max_voxels = QSpinBox()
        self._max_voxels.setRange(1, 2000)
        self._max_voxels.setValue(int(sg.DEFAULT_MAX_VOXELS // 1_000_000))
        self._max_voxels.setSuffix(" M voxels")
        self._max_voxels.setToolTip(
            "Volumes above this are decimated laterally before segmentation — Z is left alone, "
            "since that is where objects are already only a few planes tall. The labels come "
            "back on the original grid either way."
        )
        form.addRow("Segment at most", self._max_voxels)
        return box

    def _build_objects_tab(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(2, 2, 2, 2)

        outer.addWidget(
            QLabel(
                "One row per object, in calibrated units: volumes in µm³ and centroids in µm, "
                "from the layer's own voxel size."
            )
        )
        self._object_table = QTableWidget(0, len(sg.OBJECT_COLUMNS), self)
        self._object_table.setHorizontalHeaderLabels(
            [sg.OBJECT_HEADERS[column] for column in sg.OBJECT_COLUMNS]
        )
        self._object_table.verticalHeader().setVisible(False)
        # napari's stylesheet renders alternating rows as blank stripes.
        self._object_table.setAlternatingRowColors(False)
        self._object_table.setSortingEnabled(True)
        outer.addWidget(self._object_table, stretch=1)

        self._summary = QLabel("No objects yet.")
        self._summary.setWordWrap(True)
        outer.addWidget(self._summary)

        buttons = QHBoxLayout()
        export = QPushButton("Export the object table…")
        export.clicked.connect(self.export_objects)
        buttons.addWidget(export)
        buttons.addStretch(1)
        outer.addLayout(buttons)
        return page

    def _build_training_tab(self) -> QWidget:
        """Fine-tune the chosen model on masks that have been corrected by hand.

        The channel and the starting model are the ones chosen in **Setup**
        rather than chosen again here. Two places to say which model is in play
        is two places that can disagree, and the answer to "which did it train?"
        has to be the one on screen.
        """
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(2, 2, 2, 2)

        blurb = QLabel(
            "Correct a Labels layer by hand, then train the model in Setup on it. "
            "Cellpose trains in 2D, so a 3D stack is one example per plane and "
            "planes with too few objects are left out."
        )
        blurb.setWordWrap(True)
        outer.addWidget(blurb)

        box = QGroupBox("Training data")
        form = QFormLayout(box)
        form.setLabelAlignment(Qt.AlignRight)

        self._train_from = QLabel("—")
        self._train_from.setWordWrap(True)
        self._train_from.setToolTip(
            "The channel and the model chosen in Setup. Change them there — a "
            "second copy here could disagree with the one the run uses."
        )
        form.addRow("Learns from", self._train_from)

        self._masks_box = QComboBox()
        self._masks_box.setToolTip(
            "The Labels layer holding the corrected masks. Segment first, correct "
            "the labels in napari, then train on them."
        )
        form.addRow("Corrected masks", self._masks_box)

        self._train_min_masks = QSpinBox()
        self._train_min_masks.setRange(1, 1000)
        self._train_min_masks.setValue(sg.DEFAULT_MIN_TRAIN_MASKS)
        self._train_min_masks.setToolTip(
            "Planes carrying fewer objects than this are not trained on. Most of a "
            "stack is empty at the top and bottom, and those planes teach the "
            "network that this kind of image holds nothing."
        )
        form.addRow("Fewest objects", self._train_min_masks)
        outer.addWidget(box)

        settings_box = QGroupBox("The run")
        run_form = QFormLayout(settings_box)
        run_form.setLabelAlignment(Qt.AlignRight)

        self._train_name = QLineEdit()
        self._train_name.setPlaceholderText("named after the model and the time")
        self._train_name.setToolTip(
            "What to call the new model. Left empty it is named after what it was "
            "trained from and when, so two fine-tunings of one model are two rows "
            "rather than the same row twice."
        )
        run_form.addRow("New model", self._train_name)

        self._train_epochs = QSpinBox()
        self._train_epochs.setRange(1, 5000)
        self._train_epochs.setValue(sg.DEFAULT_EPOCHS)
        self._train_epochs.setToolTip(
            "Passes over the training set. A hundred is cellpose's own default and "
            "enough for a handful of corrected planes."
        )
        run_form.addRow("Epochs", self._train_epochs)

        self._train_rate = QDoubleSpinBox()
        self._train_rate.setDecimals(6)
        self._train_rate.setRange(0.000001, 0.01)
        self._train_rate.setSingleStep(0.000005)
        self._train_rate.setValue(sg.DEFAULT_LEARNING_RATE)
        self._train_rate.setToolTip(
            "How far each step moves the weights. Cellpose's fine-tuning default is "
            "1e-5; higher pulls the model off the general solution it already has, "
            "which is the thing being kept."
        )
        run_form.addRow("Learning rate", self._train_rate)
        outer.addWidget(settings_box)

        self._train_button = QPushButton("Train")
        self._train_button.clicked.connect(self.train)
        outer.addWidget(self._train_button)

        self._train_status = QLabel("Pick a corrected Labels layer, then Train.")
        self._train_status.setWordWrap(True)
        outer.addWidget(self._train_status)
        outer.addStretch(1)
        return page

    def _labels_layers(self) -> list:
        from napari.layers import Labels

        return [layer for layer in self._viewer.layers if isinstance(layer, Labels)]

    def refresh_training(self) -> None:
        """Rebuild the masks list and restate what Setup has chosen."""
        previous = self._masks_box.currentData()
        self._masks_box.clear()
        layers = self._labels_layers()
        if not layers:
            self._masks_box.addItem(NO_MASKS, "")
        for layer in layers:
            self._masks_box.addItem(_short_name(layer), str(layer.name))
            self._masks_box.setItemData(
                self._masks_box.count() - 1, str(layer.name), Qt.ToolTipRole
            )
        index = self._masks_box.findData(previous) if previous else -1
        self._masks_box.setCurrentIndex(max(index, 0))

        channel = str(self._channel_box.currentData() or "") or "no channel"
        model = str(self._model_box.currentData() or "") or "no model"
        self._train_from.setText(f"{Path(model).name or model}, on {channel}")
        self._train_button.setEnabled(bool(layers) and sg.can_train())

    def _path_row(self, tooltip: str) -> tuple[QLineEdit, QWidget]:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        edit = QLineEdit()
        edit.setToolTip(tooltip)
        layout.addWidget(edit, stretch=1)
        browse = QPushButton("…")
        browse.setMaximumWidth(30)
        browse.clicked.connect(lambda: self._browse_into(edit))
        layout.addWidget(browse)
        return edit, row

    def _browse_into(self, edit: QLineEdit) -> None:
        # Start where cellpose keeps its models, which is where a trained one lands
        # and is not somewhere anyone remembers the path to.
        start = edit.text() or str(sg.model_directory() or Path.home())
        path, _selected = QFileDialog.getOpenFileName(
            self, "Choose a Cellpose model", start, MODEL_FILTER
        )
        if path:
            edit.setText(path)

    def _update_mode_sensitivity(self, mode: str) -> None:
        """Grey out what the chosen mode does not read, rather than lying about it."""
        self._stitch.setEnabled(mode == sg.MODE_STITCH)
        self._flow_threshold.setEnabled(mode != sg.MODE_3D)

    # -- viewer state ---------------------------------------------------------

    def _connect_viewer(self) -> None:
        try:
            self._viewer.layers.events.inserted.connect(self._on_layers_changed)
            self._viewer.layers.events.removed.connect(self._on_layers_changed)
        except Exception:  # pragma: no cover - a viewer stub without events
            logger.debug("could not subscribe to layer events", exc_info=True)

    def _on_layers_changed(self, event=None) -> None:
        if not self._updating:
            self.refresh_layers()

    def _image_layers(self) -> list:
        from napari.layers import Image

        return [layer for layer in self._viewer.layers if isinstance(layer, Image)]

    def refresh_layers(self) -> None:
        """Rebuild the channel combos, keeping the choices already made."""
        self._updating = True
        try:
            layers = self._image_layers()
            names = [str(layer.name) for layer in layers]
            labels = [_short_name(layer) for layer in layers]

            for box, extra in (
                (self._channel_box, None),
                (self._nuclei_box, NO_NUCLEI),
                (self._membrane_box, NO_MEMBRANE),
                (self._measure_box, SAME_CHANNEL),
            ):
                previous = box.currentData()
                box.clear()
                if extra is not None:
                    box.addItem(extra, "")
                for name, label in zip(names, labels):
                    box.addItem(label, name)
                    box.setItemData(box.count() - 1, name, Qt.ToolTipRole)
                index = box.findData(previous)
                box.setCurrentIndex(index if index >= 0 else 0)
        finally:
            self._updating = False
        self._refresh_models()
        self.refresh_training()

    def _refresh_models(self) -> None:
        """Fill the model list from the installed Cellpose, keeping the selection.

        Three kinds end up in here: what cellpose ships, what the user trained and
        registered in the Cellpose GUI, and loose weights in the cellpose model
        folder. Which is which shows in the tooltip, since a name on its own does
        not say where it came from.
        """
        previous = self._model_box.currentData()
        choices = list(sg.model_choices())
        self._model_box.clear()
        if not choices:
            choices = [sg.ModelChoice(label=sg.CPSAM_MODEL, value=sg.CPSAM_MODEL)]
        for choice in choices:
            self._model_box.addItem(choice.label, choice.value)
            self._model_box.setItemData(self._model_box.count() - 1, choice.describe(), Qt.ToolTipRole)
        index = self._model_box.findData(previous) if previous else -1
        if index < 0:
            index = max(self._model_box.findData(sg.default_model()), 0)
        self._model_box.setCurrentIndex(index)

    def refresh_device(self) -> None:
        """Show which device a run would use, before it is run."""
        device = sg.compute_device(prefer_gpu=self._use_gpu.isChecked())
        self._device_label.setText(device.describe())
        if not device.is_gpu:
            self._device_label.setToolTip(
                "Cellpose falls back to the CPU silently. If this machine has an NVIDIA card, "
                "the CPU-only torch wheel is usually why — install a CUDA build from pytorch.org."
            )
        else:
            self._device_label.setToolTip("")

    def _check_backend(self) -> None:
        """Say plainly what to install when Cellpose is missing, and disable Segment."""
        message = sg.missing_backend_message()
        if message is None:
            self._backend_notice.setVisible(False)
            self._run_button.setEnabled(True)
            return
        self._backend_notice.setText(f"<b>Segmentation is unavailable.</b><br>{message}")
        self._backend_notice.setVisible(True)
        self._run_button.setEnabled(False)

    # -- settings and snapshotting -------------------------------------------

    def settings(self) -> sg.SegmentationSettings:
        return sg.SegmentationSettings(
            # The data, not the text: a trained model shows as its file name and
            # loads by its full path.
            model=str(self._model_box.currentData() or self._model_box.currentText()),
            custom_model_path=self._custom_edit.text().strip(),
            mode=self._mode_box.currentText(),
            diameter_um=float(self._diameter.value()),
            flow_threshold=float(self._flow_threshold.value()),
            cellprob_threshold=float(self._cellprob.value()),
            stitch_threshold=float(self._stitch.value()),
            min_size=int(self._min_size.value()),
            max_solidity=float(self._max_solidity.value()),
            median_radius_px=int(self._median.value()),
            expand_um=float(self._expand.value()),
            membrane_smooth_um=float(self._membrane_smooth.value()),
            membrane_compactness=float(self._membrane_compactness.value()),
            normalize=self._normalize.isChecked(),
            use_gpu=self._use_gpu.isChecked(),
            batch_size=int(self._batch_size.value()),
            max_voxels=int(self._max_voxels.value()) * 1_000_000,
        )

    # The batch panel shows the same setting, so that a plate run can be filtered
    # from the panel it is started in; these keep the two spin boxes as one value
    # rather than two that drift apart.

    def median_radius(self) -> int:
        return int(self._median.value())

    def set_median_radius(self, radius: int) -> None:
        if int(radius) != int(self._median.value()):
            self._median.setValue(int(radius))

    def _median_changed(self, radius: int) -> None:
        panel = getattr(self._viewer, "_mv_batch_widget", None)
        if panel is not None and hasattr(panel, "set_median_radius"):
            panel.set_median_radius(int(radius))

    def _layer_named(self, name: str):
        if not name:
            return None
        try:
            return self._viewer.layers[name]
        except (KeyError, ValueError):
            return None

    def _snapshot(self, layer) -> tuple[np.ndarray | None, tuple[float, ...], str]:
        """Turn one layer into a plain array the worker thread can use safely.

        Always the full-resolution level, never a pyramid preview, and always the
        layer's own calibrated scale — the reader derived both from the file.
        """
        if layer is None:
            return None, (), "No channel selected."
        data = layer.data[0] if getattr(layer, "multiscale", False) else layer.data
        ndim = int(getattr(data, "ndim", 0))
        scale = tuple(float(s) for s in layer.scale)

        if ndim == 4:
            # A time series: segment the timepoint on screen rather than refusing,
            # since that is the volume being looked at.
            steps = self._viewer.dims.current_step
            index = int(steps[0]) if len(steps) else 0
            index = max(0, min(index, int(data.shape[0]) - 1))
            data = data[index]
            scale = scale[1:]
            ndim = 3
        if ndim not in (2, 3):
            return None, (), f"“{layer.name}” is {ndim}D; segmentation needs a 2D or 3D image."

        voxel = scale[-ndim:] if len(scale) >= ndim else (1.0,) * ndim
        return np.asarray(data), tuple(float(v) for v in voxel), ""

    # -- the run --------------------------------------------------------------

    def run(self) -> None:
        """Segment off the main thread and load what comes back."""
        if self._worker is not None:
            self._status.setText("A segmentation is already running.")
            return

        message = sg.missing_backend_message()
        if message is not None:
            # Status text rather than a modal box: the notice at the top of the
            # panel already says this and Segment is disabled, so a dialog here
            # would be the third time of asking.
            self._status.setText(message.replace("\n\n", " "))
            self._check_backend()
            return

        source_name = str(self._channel_box.currentData() or "")
        source = self._layer_named(source_name)
        image, voxel, problem = self._snapshot(source)
        if image is None:
            self._status.setText(problem)
            return

        problems: list[str] = []
        nuclei_name = str(self._nuclei_box.currentData() or "")
        nuclei = None
        if nuclei_name and nuclei_name != source_name:
            nuclei, _nuclei_voxel, nuclei_problem = self._snapshot(self._layer_named(nuclei_name))
            if nuclei is None:
                problems.append(nuclei_problem or f"“{nuclei_name}” could not be read.")
        elif nuclei_name == source_name:
            problems.append(
                "The nuclear channel is the channel being segmented; it was segmented on its own."
            )

        membrane_name = str(self._membrane_box.currentData() or "")
        membrane = None
        if membrane_name and membrane_name != source_name:
            membrane, _membrane_voxel, membrane_problem = self._snapshot(
                self._layer_named(membrane_name)
            )
            if membrane is None:
                problems.append(membrane_problem or f"“{membrane_name}” could not be read.")
        elif membrane_name == source_name:
            problems.append(
                "The membrane channel is the channel being segmented, so there was nothing "
                "to grow along; the objects were grown by distance instead."
            )

        measure_name = str(self._measure_box.currentData() or "") or source_name
        measure_layer = self._layer_named(measure_name)
        signal, _signal_voxel, signal_problem = self._snapshot(measure_layer)
        if signal is None or signal.shape != image.shape:
            if measure_name != source_name:
                problems.append(
                    signal_problem
                    or f"“{measure_name}” does not match the segmented channel's shape; "
                    "intensities were measured on the segmented channel instead."
                )
            signal = image
            measure_name = source_name

        settings = self.settings()
        self._run_button.setEnabled(False)
        with_nuclei = "" if nuclei is None else f" with nuclei from {nuclei_name}"
        with_membrane = "" if membrane is None else f", cells along {membrane_name}"
        self._status.setText(
            f"Segmenting {_short_name(source)} ({'×'.join(str(n) for n in image.shape)})"
            f"{with_nuclei}{with_membrane} on {self._device_label.text()}…"
        )

        def _progress(text: str) -> None:
            # Runs on the worker thread, where Qt calls are not safe; the log is
            # the one place it can say anything from there.
            logger.info("segmentation: %s", text)

        def _work():
            result = sg.segment_volume(
                image,
                voxel,
                settings=settings,
                progress=_progress,
                nuclei=nuclei,
                membrane=membrane,
            )
            # The shapes are already measured when the filter ran; measuring them
            # again for the table would double the cost of the slowest step.
            stats = sg.object_table(
                result.masks,
                signal,
                result.voxel_size_um[-image.ndim:],
                shapes=result.shapes,
                expanded=result.expanded_masks,
            )
            return result, stats

        try:
            from napari.qt.threading import thread_worker
        except Exception:  # pragma: no cover - no Qt threading available
            self._finish(_work(), source_name, measure_name, problems)
            return

        @thread_worker
        def _run():
            return _work()

        worker = _run()
        worker.returned.connect(
            lambda outcome: self._finish(outcome, source_name, measure_name, problems)
        )
        worker.errored.connect(self._on_error)
        worker.finished.connect(self._clear_worker)
        self._worker = worker
        worker.start()

    def _clear_worker(self) -> None:
        self._worker = None
        self._run_button.setEnabled(True)

    # -- training -------------------------------------------------------------

    def training_examples(self) -> tuple[list, list[str], str]:
        """``(examples, skipped, problem)`` from what is on screen.

        Separate from :meth:`train` so the whole of the reading, pairing and
        refusing can be driven and checked without starting a run that takes
        minutes.
        """
        masks_name = str(self._masks_box.currentData() or "")
        if not masks_name:
            return [], [], "No Labels layer to train on. Segment, correct the masks, then train."

        image, _voxel, problem = self._snapshot(self._layer_named(
            str(self._channel_box.currentData() or "")
        ))
        if image is None:
            return [], [], problem or "No channel selected."

        labels, _labels_voxel, labels_problem = self._snapshot(self._layer_named(masks_name))
        if labels is None:
            return [], [], labels_problem or f"“{masks_name}” could not be read."

        try:
            examples, skipped = sg.training_examples(
                image, labels, min_masks=int(self._train_min_masks.value())
            )
        except ValueError as error:
            return [], [], str(error)
        if not examples:
            kept = "; ".join(skipped[:3])
            return [], skipped, (
                "No plane carried enough corrected objects to train on. "
                f"Lower “Fewest objects”, or correct more masks. {kept}"
            )
        return examples, skipped, ""

    def _training_said(self, text: str) -> None:
        """One line of cellpose's training log, on the main thread."""
        self._train_status.setText(text)

    def train(self) -> None:
        """Fine-tune the Setup model on the corrected masks, off the main thread."""
        if self._trainer is not None:
            self._train_status.setText("A training run is already going.")
            return
        if not sg.can_train():
            self._train_status.setText(
                sg.missing_backend_message() or "The installed cellpose cannot train a model."
            )
            return

        examples, skipped, problem = self.training_examples()
        if problem:
            self._train_status.setText(problem)
            return

        settings = sg.TrainingSettings(
            model=self.settings().resolved_model(),
            name=self._train_name.text().strip(),
            epochs=int(self._train_epochs.value()),
            learning_rate=float(self._train_rate.value()),
            min_masks=int(self._train_min_masks.value()),
            use_gpu=bool(self.settings().use_gpu),
        )
        device = sg.compute_device(settings.use_gpu)
        n_objects = sum(int(example.n_objects) for example in examples)
        left_out = f", {len(skipped)} plane(s) left out" if skipped else ""
        for note in skipped:
            logger.info("training: skipped %s", note)

        self._train_button.setEnabled(False)
        self._train_status.setText(
            f"Training {settings.resolved_name()} from {Path(settings.model).name or settings.model} "
            f"on {len(examples)} plane(s), {n_objects:,} object(s){left_out}, "
            f"on {device.describe()}…"
        )

        voice = self._voice

        def _work():
            return sg.train_model(
                examples, settings, device, progress=lambda text: voice.said.emit(str(text))
            )

        try:
            from napari.qt.threading import thread_worker
        except Exception:  # pragma: no cover - no Qt threading available
            self._finish_training(_work(), skipped)
            self._clear_trainer()
            return

        @thread_worker
        def _run():
            return _work()

        worker = _run()
        worker.returned.connect(lambda outcome: self._finish_training(outcome, skipped))
        worker.errored.connect(self._on_training_error)
        worker.finished.connect(self._clear_trainer)
        self._trainer = worker
        worker.start()

    def _clear_trainer(self) -> None:
        self._trainer = None
        self._train_button.setEnabled(True)

    def _on_training_error(self, exc) -> None:
        logger.exception("training failed", exc_info=exc)
        self._trainer = None
        self._train_button.setEnabled(True)
        self._train_status.setText(f"Training failed: {exc}")

    def _finish_training(self, result: sg.TrainingResult, skipped) -> None:
        """Take the new model, and make it the one that would be used next.

        Selected rather than only listed: training it and then segmenting with
        the old one is the mistake the panel is in a position to prevent.
        """
        self._refresh_models()
        index = self._model_box.findData(result.path)
        if index < 0:
            index = self._model_box.findData(result.name)
        if index < 0:
            # Not in the list: the weights went somewhere the scan does not look.
            # Point the custom-model box at them, which is the other way in.
            self._custom_edit.setText(result.path)
            where = " It is not in the model list, so Custom model now points at it."
        else:
            self._model_box.setCurrentIndex(index)
            self._custom_edit.clear()
            where = " It is selected in Setup."
        lines = [result.describe() + "." + where]
        if skipped:
            lines.append(f"Left out: {'; '.join(skipped[:4])}")
        self._train_status.setText(" ".join(lines))
        logger.info("training: %s -> %s", result.describe(), result.path)

    def _on_error(self, exc) -> None:
        logger.exception("segmentation failed", exc_info=exc)
        self._worker = None
        self._run_button.setEnabled(True)
        self._status.setText(f"Segmentation failed: {exc}")
        QMessageBox.critical(self, "Microscopy Viewer", f"Segmentation failed:\n{exc}")

    def _finish(self, outcome, source_name: str, measure_name: str, problems: list[str]) -> None:
        result, stats = outcome
        self._result = result
        self._stats = stats
        self._run_button.setEnabled(True)

        if self._add_layer.isChecked():
            self._add_labels_layer(result, source_name)
        self._fill_object_table(stats)

        summary = (
            f"{result.n_objects} object(s) in {result.elapsed_s:.0f} s "
            f"({result.model}, {result.mode}, {result.device}"
        )
        if result.diameter_px:
            summary += f", diameter {result.diameter_px:.0f} px"
        if result.used_nuclear_channel:
            summary += ", two channels"
        if result.median_radius_px:
            summary += f", median r={result.median_radius_px} px"
        summary += ")."
        if result.expanded_masks is not None:
            if result.used_membrane_channel:
                summary += (
                    f" Cells grown along the membrane, up to {result.expand_um:g} µm from "
                    "each nucleus: intensities also measured in the cytoplasm and in "
                    "the whole cell (see the exported table)."
                )
            else:
                summary += (
                    f" Grown {result.expand_um:g} µm: intensities also measured in the "
                    "expansion and in nucleus + expansion (see the exported table)."
                )
        if result.n_dropped:
            summary += (
                f" {result.n_dropped} dropped above solidity "
                f"{self._max_solidity.value():g}."
            )
        if measure_name and measure_name != source_name:
            summary += f" Intensities measured on {measure_name}."
        messages = list(problems) + list(result.warnings)
        if messages:
            summary += " " + " ".join(messages[:3])
        self._status.setText(summary)
        self._update_summary(stats)
        logger.info("segmentation finished: %s", summary)

    def _add_labels_layer(self, result: sg.SegmentationResult, source_name: str) -> None:
        """Put the label map into the viewer at the source layer's own scale."""
        self._updating = True
        try:
            base = _short_name(self._layer_named(source_name)) if source_name else "image"
            layer_name = f"{base} {LABELS_SUFFIX}"
            if layer_name in self._viewer.layers:
                self._viewer.layers.remove(layer_name)
            scale = result.voxel_size_um[-result.masks.ndim:]
            grown_suffix = (
                CELL_LABELS_SUFFIX if result.used_membrane_channel else EXPANDED_LABELS_SUFFIX
            )
            # Both names are cleared: a run with a membrane channel after one
            # without should not leave the previous grown set on screen beside
            # its replacement, where the two would be taken for one.
            for stale in (EXPANDED_LABELS_SUFFIX, CELL_LABELS_SUFFIX):
                if f"{base} {stale}" in self._viewer.layers:
                    self._viewer.layers.remove(f"{base} {stale}")
            grown_name = f"{base} {grown_suffix}"
            if result.expanded_masks is not None:
                # Underneath the objects themselves, and faint: the nucleus is
                # what was found, the ring round it is where else it was measured.
                self._viewer.add_labels(
                    result.expanded_masks,
                    name=grown_name,
                    scale=tuple(float(v) for v in scale),
                    opacity=0.35,
                )
            self._viewer.add_labels(
                result.masks,
                name=layer_name,
                scale=tuple(float(v) for v in scale),
            )
        except Exception:
            logger.exception("could not add the labels layer")
        finally:
            self._updating = False
        self.refresh_layers()

    # -- results --------------------------------------------------------------

    def _fill_object_table(self, stats) -> None:
        table = self._object_table
        table.setSortingEnabled(False)
        table.setRowCount(len(stats))
        for row, stat in enumerate(stats):
            values = stat.as_row()
            for column, key in enumerate(sg.OBJECT_COLUMNS):
                value = values.get(key, "")
                text = value if isinstance(value, str) else format_number(value)
                item = QTableWidgetItem(str(text))
                item.setFlags(Qt.ItemIsEnabled)
                table.setItem(row, column, item)
        table.setSortingEnabled(True)
        table.resizeColumnsToContents()

    def _update_summary(self, stats) -> None:
        totals = sg.count_summary(stats)
        if not totals.get("count"):
            self._summary.setText("No objects found. Try a larger cell probability, or a diameter.")
            return
        text = (
            f"{int(totals['count'])} object(s); median volume "
            f"{format_number(totals['median_volume_um3'])} µm³, median equivalent diameter "
            f"{format_number(totals['median_diameter_um'])} µm."
        )
        if "median_solidity" in totals:
            text += f" Median solidity {format_number(totals['median_solidity'])}."
        self._summary.setText(text)

    def export_objects(self) -> None:
        """Write the object table out, through the same writer the measurements use."""
        if not self._stats:
            self._status.setText("Nothing to export — run a segmentation first.")
            return
        suggested = str(Path.home() / "Documents" / f"{default_stem('segmentation')}.xlsx")
        path, _selected = QFileDialog.getSaveFileName(
            self, "Export the object table", suggested,
            "Excel workbook (*.xlsx);;CSV file (*.csv)",
        )
        if not path:
            return
        try:
            written = export_table(sg.object_dataframe(self._stats), path, sheet_name="Objects")
        except Exception as exc:
            logger.exception("object export failed")
            QMessageBox.critical(self, "Microscopy Viewer", f"Could not write that file:\n{exc}")
            return
        self._status.setText(f"Object table written to {written}.")
