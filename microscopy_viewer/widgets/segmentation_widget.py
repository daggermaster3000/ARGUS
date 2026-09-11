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
from qtpy.QtCore import Qt
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

MODEL_FILTER = "Cellpose models (*);;All files (*)"

#: Shown in the "measure" box for "whatever was segmented".
SAME_CHANNEL = "— the segmented channel —"
#: Entry for "segment one channel on its own", the default.
NO_NUCLEI = "— none —"


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


class SegmentationWidget(QWidget):
    """Pick a channel, segment it with Cellpose, measure the objects."""

    def __init__(self, viewer, parent: QWidget | None = None):
        super().__init__(parent)
        self._viewer = viewer
        self._worker = None
        self._result: sg.SegmentationResult | None = None
        self._stats: list[sg.ObjectStat] = []
        self._updating = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self._backend_notice = QLabel("")
        self._backend_notice.setWordWrap(True)
        self._backend_notice.setVisible(False)
        layout.addWidget(self._backend_notice)

        self._tabs = QTabWidget()
        self._tabs.addTab(self._build_setup_tab(), "Setup")
        self._tabs.addTab(self._build_objects_tab(), "Objects")
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
            normalize=self._normalize.isChecked(),
            use_gpu=self._use_gpu.isChecked(),
            batch_size=int(self._batch_size.value()),
            max_voxels=int(self._max_voxels.value()) * 1_000_000,
        )

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
        self._status.setText(
            f"Segmenting {_short_name(source)} ({'×'.join(str(n) for n in image.shape)})"
            f"{with_nuclei} on {self._device_label.text()}…"
        )

        def _progress(text: str) -> None:
            # Runs on the worker thread, where Qt calls are not safe; the log is
            # the one place it can say anything from there.
            logger.info("segmentation: %s", text)

        def _work():
            result = sg.segment_volume(
                image, voxel, settings=settings, progress=_progress, nuclei=nuclei
            )
            # The shapes are already measured when the filter ran; measuring them
            # again for the table would double the cost of the slowest step.
            stats = sg.object_table(
                result.masks,
                signal,
                result.voxel_size_um[-image.ndim:],
                shapes=result.shapes,
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
        summary += ")."
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
