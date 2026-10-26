"""Dock panel: draw a line along a junction, get the band beside it as a label.

The assembloid workflow from the Fiji macro, cut down to its geometry: a line is
drawn through the interface between the two organoids, a spline is put through
it, and a band of fixed width beside it becomes a Labels layer — which is then
stored with the sample like any other label map.

The work is in :mod:`microscopy_viewer.fusion`. This module owns the two layers
(the line, and the region made from it), rebuilds the region whenever the line
or a setting changes, and saves it.

The line does not have to be drawn. :mod:`microscopy_viewer.junction` proposes
one from where the labelled cells are, and scores the sample while it is at it:
**Find it automatically** puts that proposal into the line layer, where it can
be corrected like a drawn one, and **Run on the selected samples** does a whole
folder, storing each band and writing a QC report.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from qtpy.QtCore import QObject, Qt, Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .. import fusion as fu
from ..utils import get_logger
from .sample_files import close_sample, restore_view, source_of, tie_to_sample, view_state

logger = get_logger("fusion_widget")

#: The Shapes layer the junction is drawn on, and the Labels layer made from it.
LINE_LAYER_NAME = "Fusion line"
REGION_LAYER_NAME = "Fusion region"

#: The junction search's cost settings as the panel opens, in step with
#: ``junction.JunctionOptions`` (not imported here: it pulls in scipy and skimage).
BEND_DEFAULT = 1000.0
TEARS_DEFAULT = False
EDGE_DEFAULT = 0.0
TISSUE_ALL_DEFAULT = True
#: Samples a run handles at once (``junction.RUN_WORKERS``).
WORKERS_DEFAULT = max(1, min(8, (os.cpu_count() or 2) - 2))

#: Shape types that are a line rather than an area.
LINE_SHAPES = ("path", "line")


def channel_spec(text: str) -> int | str:
    """A channel box's contents: a number is an index, anything else a name."""
    cleaned = str(text).strip()
    return int(cleaned) if cleaned.isdigit() else cleaned


class _Relay(QObject):
    """Carries worker-thread progress text onto the GUI thread."""

    message = Signal(str)


class FusionWidget(QWidget):
    """Draw the junction, expand it into a band, save the band as a label map."""

    #: Emitted with the sample's file when a region has been stored with it.
    region_saved = Signal(str)

    def __init__(self, app, parent: QWidget | None = None):
        super().__init__(parent)
        self._app = app
        self._viewer = app.viewer
        self._updating = False
        self._watched = None
        #: What the region on screen was built from; ``None`` when there is none.
        self._labels: np.ndarray | None = None
        #: ``(sample, overview, tissue)`` for the band that follows the tissue.
        self._tissue_cache: tuple | None = None
        self._curve: np.ndarray | None = None
        self._voxel: tuple[float, float] = (1.0, 1.0)
        self._worker = None
        self._cancelled = False
        self._relay = _Relay()
        self._relay.message.connect(self._log)

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
        layout.addWidget(self._build_batch_box())

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

        self._follow_tissue = QCheckBox("Follow the tissue")
        self._follow_tissue.setChecked(True)
        self._follow_tissue.setToolTip(
            "On: the band is the organoid's own tissue within the band width of the "
            "line, as the automatic run makes it — its ends follow the organoid's "
            "outline, out to the margin below. The outline is read off the nuclear "
            "stain once per sample, which takes a few seconds the first time.\n"
            "Off: the macro's strip, with straight ends and the outer cap."
        )
        self._follow_tissue.toggled.connect(lambda _on: self.update_region())
        form.addRow("", self._follow_tissue)

        self._margin = self._spin(0.0, 500.0, fu.TISSUE_MARGIN_UM, 5.0)
        self._margin.setToolTip(
            "How far the band reaches past the edge of the tissue, along the organoid's "
            "side. Cells migrating out over the surface of the organoid sit just outside "
            "the outline the nuclear stain gives, and would be missed at 0.\n"
            "Used when the band follows the tissue, here and in the automatic run."
        )
        self._margin.valueChanged.connect(lambda _value: self._follow_tissue.isChecked() and self.update_region())
        form.addRow("Past the tissue edge", self._margin)

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

        self._key_edit = QLineEdit(fu.MANUAL_KEY)
        self._key_edit.setToolTip(
            "The name the region is stored under with the sample, beside any "
            "segmentation. Saving again under the same name replaces it.\n"
            f"The automatic run stores its own under “{fu.DEFAULT_KEY}”, so a region "
            "drawn here never overwrites it; Junction analysis and the manual count use "
            "this one where a sample has it."
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

    def _build_batch_box(self) -> QGroupBox:
        box = QGroupBox("Automatic, with QC")
        outer = QVBoxLayout(box)

        how = QLabel(
            "Which organoid is which comes from the labelled cells; where exactly "
            "they meet is read off the dark line between them in the nuclear stain."
        )
        how.setWordWrap(True)
        outer.addWidget(how)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)
        self._label_channel = QLineEdit("")
        self._label_channel.setPlaceholderText("automatic: GFP / 488")
        self._label_channel.setToolTip(
            "The channel carrying the labelled cells that mark one of the two "
            "organoids — part of its name, or its number. Empty takes the first "
            "channel whose name contains “gfp” or “488”.\n"
            "It has to mark one organoid throughout. A sparse label that sits "
            "along one edge only (mCherry in the 2025-09 slides) splits its own "
            "organoid in two instead."
        )
        form.addRow("Labelled cells", self._label_channel)
        self._nuclear_channel = QLineEdit("")
        self._nuclear_channel.setPlaceholderText("automatic: DAPI / Hoechst / 405")
        self._nuclear_channel.setToolTip(
            "The nuclear stain: it gives the outline of the tissue and the line "
            "between the organoids. Empty takes the first channel whose name "
            "contains “dapi”, “hoechst” or “405”."
        )
        form.addRow("Nuclear stain", self._nuclear_channel)
        self._other_channel = QLineEdit("")
        self._other_channel.setPlaceholderText("automatic: mCherry / 555, else the remaining channel")
        self._other_channel.setToolTip(
            "The channel labelling the other organoid — part of its name, or its number. "
            "It is only used to find where that organoid is: the densest patch of its "
            "cells is the second seed, as the densest patch of the labelled cells is the "
            "first. It does not have to mark the organoid throughout; a label along one "
            "edge is enough.\n"
            "Empty takes mCherry / 555 / 561 when there is one, else whichever channel is "
            "left. When that channel holds no cells, the second seed is the tissue with "
            "few labelled cells furthest from them, as before."
        )
        form.addRow("Other organoid", self._other_channel)
        self._seed_other = QCheckBox("Find the other organoid from its own label")
        self._seed_other.setChecked(True)
        self._seed_other.setToolTip(
            "On: the second seed is the dense core of the other organoid's label. Off: it "
            "is the tissue far from the labelled cells, whatever the other channel shows."
        )
        form.addRow("", self._seed_other)
        self._follow_line = QCheckBox("Follow the dark line in the nuclear stain")
        self._follow_line.setChecked(True)
        self._follow_line.setToolTip(
            "Two organoids pressed together leave a thin dark line between them in "
            "the nuclear stain, and the junction is drawn along it.\n"
            "Turn it off for sections with many cracks or folds: those are dark "
            "lines too, and the junction can be pulled onto one. Off, the junction "
            "is the shortest cut across the tissue between the two organoids, which "
            "is right at a neck and approximate anywhere else."
        )
        form.addRow("", self._follow_line)
        self._tissue_all = QCheckBox("Find the tissue in every channel")
        self._tissue_all.setChecked(TISSUE_ALL_DEFAULT)
        self._tissue_all.setToolTip(
            "The section's outline is where any channel stands above the slide, not only "
            "the nuclear stain. One organoid of a pair often has a faint nuclear stain; "
            "from that alone it is dropped, and the junction stops short of the outline.\n"
            "Turn it off if a channel lights up the slide around the tissue (mounting "
            "medium, debris): the outline then grows onto it."
        )
        form.addRow("", self._tissue_all)
        self._bend = self._spin(0.0, 5000.0, BEND_DEFAULT, 50.0)
        self._bend.setSpecialValueText("free")
        self._bend.setToolTip(
            "What bending the junction costs: a sharp 90° corner costs as much as cutting "
            "this much plain tissue. The same turn spread along a longer stretch costs less, "
            "so the line still curves round an organoid but no longer turns a corner to "
            "follow a crack or a fold.\n"
            "Higher keeps the junction straighter; 0 (“free”) lets the dark line alone decide."
        )
        form.addRow("Bending cost", self._bend)
        self._tears = QCheckBox("Do not cut along tears")
        self._tears.setChecked(TEARS_DEFAULT)
        self._tears.setToolTip(
            "A tear in the section is empty, so in the nuclear stain it is the darkest thing "
            "there is — and the junction search, which looks for a dark line, would run down "
            "it. On: holes in the tissue and stain that is all but gone cost as much to cut "
            "along as plain tissue. The junction is dim, not empty, and keeps its discount."
        )
        form.addRow("", self._tears)
        self._edge = QDoubleSpinBox()
        self._edge.setRange(0.0, 5.0)
        self._edge.setDecimals(2)
        self._edge.setSingleStep(0.25)
        self._edge.setValue(EDGE_DEFAULT)
        self._edge.setSpecialValueText("off")
        self._edge.setToolTip(
            "Pull the junction to where the labelled cells change: cutting costs this much "
            "more where the density of labelled cells is at either organoid's own level, and "
            "nothing where it is halfway between. Keeps the line from sliding towards one "
            "pole of the pair, or down a tear across which the cells do not change. "
            "The other organoid's own label, when there is one, is used the same way."
        )
        form.addRow("Follow the cells' edge", self._edge)

        median_row = QHBoxLayout()
        self._use_median = QCheckBox("Take away the local median over")
        self._use_median.setChecked(True)
        self._use_median.setToolTip(
            "Before the labelled cells are counted into a density map (which says where "
            "each organoid is), the local median of the channel is taken away: a cell is "
            "a point brighter than its surroundings, and diffuse signal — "
            "autofluorescence, a fold, a thick part of the section — goes with the median.\n"
            "Off: the channel is counted as it is, so a bright but cell-free patch counts "
            "as cells. “Show the filter” puts both on the canvas to compare."
        )
        median_row.addWidget(self._use_median)
        self._median_size = QDoubleSpinBox()
        self._median_size.setRange(10.0, 500.0)
        self._median_size.setDecimals(0)
        self._median_size.setSingleStep(5.0)
        self._median_size.setSuffix(" µm")
        self._median_size.setValue(45.0)
        self._median_size.setToolTip(
            "Side of the median window. Wider than a cell (a few cells across) so a cell "
            "stands above it; narrow enough to follow a fold or a bright patch."
        )
        self._use_median.toggled.connect(self._median_size.setEnabled)
        median_row.addWidget(self._median_size)
        show_median = QPushButton("Show the filter")
        show_median.setToolTip(
            "On the sample open here: the labelled channel as the junction search sees it, "
            "the median taken away from it, what is left (the cells it counts), and the "
            "density of labelled cells with and without the median, as layers to flick "
            "between. Nothing is changed or saved."
        )
        show_median.clicked.connect(self.show_median_filter)
        median_row.addWidget(show_median)
        median_row.addStretch(1)
        form.addRow("Cells", median_row)
        self._into_other = QCheckBox("Band on the organoid the cells migrate into")
        self._into_other.setChecked(True)
        self._into_other.setToolTip(
            "On: the band is on the unlabelled organoid, where cells that crossed "
            "the junction are counted. Off: on the labelled organoid itself."
        )
        form.addRow("", self._into_other)
        self._clip = QCheckBox("Keep only the part of the band on tissue")
        self._clip.setChecked(True)
        self._clip.setToolTip(
            "Used by “Run on the selected samples”: the stored band stops at the "
            "edge of the section, so its area is tissue area. The band shown for a "
            "single image is not clipped."
        )
        form.addRow("", self._clip)
        outer.addLayout(form)

        find = QPushButton("Find the junction on this image")
        find.setToolTip(
            "Propose the junction from where the labelled (GFP) cells are, and put it "
            "in the line layer, where it can be corrected like a drawn one.\n"
            "The band shown is the organoid's own tissue within the band width of "
            "the junction, so its ends follow the organoid's outline; editing the "
            "line or a setting afterwards rebuilds it as for a drawn line.\n"
            "The side is set to the organoid the cells migrate into. What was "
            "flagged about the sample is shown below."
        )
        find.clicked.connect(self.find_automatically)
        outer.addWidget(find)

        row = QHBoxLayout()
        self._run_button = QPushButton("Run on the selected samples")
        self._run_button.setToolTip(
            "Every sample selected in Experiment setup (all of them when none is): "
            "find the junction, store the band with the sample, and write a QC "
            "report into “junction_QC” in the experiment folder: a workbook, picture "
            "sheets, and a picture per sample sorted into the folders ok, check and fail.\n"
            "The band is stored as the label map named above and as an outline "
            "called “Fusion band”, so Segmentation ▸ Batch can be restricted to it "
            "and the Analysis panel counts inside it."
        )
        self._run_button.clicked.connect(self.run_batch)
        row.addWidget(self._run_button)
        self._workers = QSpinBox()
        self._workers.setRange(1, max(1, os.cpu_count() or 1))
        self._workers.setValue(WORKERS_DEFAULT)
        self._workers.setSuffix(" at once")
        self._workers.setToolTip(
            "How many samples are worked on at the same time, each in its own process. "
            "About 5 s a sample one after another; eight at once is several times faster. "
            "Lower it if the computer becomes slow or runs short of memory; 1 = one after another."
        )
        row.addWidget(self._workers)
        self._stop_button = QPushButton("Stop")
        self._stop_button.setEnabled(False)
        self._stop_button.clicked.connect(self.stop)
        row.addWidget(self._stop_button)
        report = QPushButton("Open the QC folder")
        report.setToolTip(
            "“junction_QC” in the experiment folder: junction_QC.xlsx with one row "
            "per sample, a picture per sample in the folder of its status (ok, check, "
            "fail), and contact sheets to flick through."
        )
        report.clicked.connect(self.open_qc_folder)
        row.addWidget(report)
        outer.addLayout(row)

        self._log_view = QPlainTextEdit()
        self._log_view.setReadOnly(True)
        self._log_view.setMaximumHeight(140)
        self._log_view.setPlaceholderText("One line per sample: ok, check or fail, and why.")
        outer.addWidget(self._log_view)
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
        if self._updating:
            return
        if REGION_LAYER_NAME not in self._viewer.layers:
            # Taken off with its sample: what it was built from must not be
            # saved into the next one.
            self._labels = self._curve = None
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
            image = self.image_layer()
            tie_to_sample(layer, source_of(image) if image is not None else "")
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

    def prepare_for_drawing(self, hints=()) -> None:
        """Pick the image to draw on and put the line tool in hand.

        *hints* are fragments of a channel name, tried in order — the nuclear
        stain, where the line between two organoids shows. What the Junction
        review panel calls on each sample it opens.
        """
        self.refresh_layers()
        names = [self._image_combo.itemText(i) for i in range(self._image_combo.count())]
        for hint in hints:
            match = next((name for name in names if str(hint).lower() in name.lower()), None)
            if match is not None:
                self._image_combo.setCurrentIndex(names.index(match))
                break
        self.start_drawing()

    def flip_side(self) -> None:
        """Put the band on the other side of the line; the region follows."""
        other = {fu.SIDE_LEFT: fu.SIDE_RIGHT, fu.SIDE_RIGHT: fu.SIDE_LEFT}.get(self._side.currentText(), fu.SIDE_LEFT)
        index = self._side.findText(other)
        if index >= 0:
            self._side.setCurrentIndex(index)
        if self.drawn_line() is not None:
            self.update_region()

    def status_text(self) -> str:
        return self._status.text()

    def clear(self) -> None:
        """Remove the line and the region, and the cells counted in that region."""
        self._updating = True
        try:
            for name in (LINE_LAYER_NAME, REGION_LAYER_NAME):
                if name in self._viewer.layers:
                    self._viewer.layers.remove(name)
        finally:
            self._updating = False
        # Counted in the region that is going: they no longer describe anything.
        analysis = getattr(self._app, "junction_analysis_widget", None)
        if analysis is not None and hasattr(analysis, "remove_layers"):
            analysis.remove_layers()
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
        followed = False
        if self._follow_tissue.isChecked():
            along = self._band_along_tissue(image, line - np.asarray(translate), shape)
            if along is not None and along.any():
                labels, followed = along, True

        self._labels, self._curve, self._voxel = labels, curve, voxel
        self._show(labels, image, voxel, translate)
        area = float(np.count_nonzero(labels)) * voxel[0] * voxel[1]
        message = (
            f"Region: {area / 1e6:.3f} mm² beside a {fu.line_length(curve):.0f} µm "
            f"junction ({self._side.currentText().lower()}"
            + (", following the tissue)." if followed else ").")
        )
        if not labels.any():
            message = "The band falls outside the image — check the side and the width."
        self._status.setText(message)
        if self._auto_save.isChecked() and labels.any():
            self.save()
        return True

    def _band_along_tissue(self, image, line_um: np.ndarray, shape) -> np.ndarray | None:
        """The drawn line's band, following the tissue; ``None`` when it cannot be made.

        The section's outline comes off an overview of the sample, made once
        per sample and kept: moving a point of the line then costs a few
        operations on the overview, not a read of the image.
        """
        from qtpy.QtWidgets import QApplication

        from .. import junction as jn

        source = source_of(image) or str(image.name)
        cached = self._tissue_cache
        if cached is None or cached[0] != source:
            self._status.setText("Reading the tissue outline off the nuclear stain…")
            QApplication.setOverrideCursor(Qt.WaitCursor)
            try:
                QApplication.processEvents()
                overview = jn.build_overview(self._sample_specs(image), self.junction_options(save=False))
                cached = self._tissue_cache = (source, overview, jn.tissue_of(overview))
            except Exception as exc:
                logger.exception("could not read the tissue outline")
                self._status.setText(f"Could not follow the tissue ({exc}); the band is a plain strip.")
                self._tissue_cache = (source, None, None)
                return None
            finally:
                QApplication.restoreOverrideCursor()
        _source, overview, filled = cached
        if overview is None or tuple(overview.shape_yx) != tuple(shape) or not filled.any():
            return None
        try:
            return jn.band_along_tissue(overview, filled, line_um, self._side.currentText(),
                                        float(self._width.value()), float(self._gap.value()),
                                        float(self._margin.value()))
        except Exception:
            logger.exception("could not make the band along the tissue")
            return None

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
            region = self._viewer.add_labels(
                labels, name=REGION_LAYER_NAME, scale=voxel, translate=translate,
                opacity=0.4, **units_like(image, 2),
            )
            tie_to_sample(region, source_of(image))
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

    # -- automatic ------------------------------------------------------------

    def _log(self, text: str) -> None:
        self._log_view.appendPlainText(text)
        self._status.setText(text)

    def junction_options(self, **overrides):
        """The band settings on this panel, as options for the automatic run."""
        from .. import junction as jn

        values = dict(
            width_um=float(self._width.value()),
            gap_um=float(self._gap.value()),
            cap_um=float(self._cap.value()),
            # The automatic band keeps its own name; the one above is for regions drawn here.
            label_key=fu.DEFAULT_KEY,
            label_channel=channel_spec(self._label_channel.text()),
            nuclear_channel=channel_spec(self._nuclear_channel.text()),
            other_channel=channel_spec(self._other_channel.text()),
            seed_from_other_channel=self._seed_other.isChecked(),
            follow_nuclear_line=self._follow_line.isChecked(),
            tissue_all_channels=self._tissue_all.isChecked(),
            bend_um=float(self._bend.value()),
            tear_price=1.0 if self._tears.isChecked() else 0.0,
            edge_weight=float(self._edge.value()),
            use_median=self._use_median.isChecked(),
            median_um=float(self._median_size.value()),
            margin_um=float(self._margin.value()),
            into_unlabelled=self._into_other.isChecked(),
            clip_to_tissue=self._clip.isChecked(),
        )
        values.update(overrides)
        return jn.JunctionOptions(**values)

    def _sample_specs(self, image) -> list:
        """Every channel of the sample *image* belongs to, as reader-like specs."""
        from types import SimpleNamespace

        from napari.layers import Image

        source = source_of(image)
        layers = [
            layer for layer in self._viewer.layers
            if isinstance(layer, Image) and (layer is image or (source and source_of(layer) == source))
        ]
        return [
            SimpleNamespace(
                data=layer.data, scale=tuple(float(v) for v in layer.scale),
                multiscale=bool(getattr(layer, "multiscale", False)),
                channel_name=str(layer.metadata.get("mv_channel_name", "") or layer.name),
                name=str(layer.name),
            )
            for layer in layers
        ]

    def show_median_filter(self) -> None:
        """Put the median filter's steps on the canvas, for the sample open here.

        On the overview the junction search works on (a few µm per pixel), so
        what is shown is what it sees. The layers are tied to the sample and go
        with it.
        """
        from qtpy.QtWidgets import QApplication
        from scipy import ndimage

        from .. import junction as jn

        image = self.image_layer()
        if image is None:
            self._status.setText("Open a sample first.")
            return
        self._status.setText("Making the median filter's preview…")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            QApplication.processEvents()
            options = self.junction_options(save=False)
            overview = jn.build_overview(self._sample_specs(image), options)
            um = overview.um
            channel = overview.channels[overview.label]
            valid = (overview.channels > 0).all(axis=0)
            tissue = jn.tissue_mask(overview.channels[overview.nuclear], valid, um,
                                    jn._others(overview.channels, overview.nuclear, options.tissue_all_channels))
            filled = ndimage.binary_fill_holes(tissue)
            size = max(3, int(round(float(self._median_size.value()) / um)) | 1)
            background = ndimage.median_filter(channel, size)
            left = jn.cell_prominence(channel, um, True, float(self._median_size.value()))
            with_median = jn.cell_density(channel, tissue, filled, um, median=True,
                                          median_um=float(self._median_size.value()))
            without = jn.cell_density(channel, tissue, filled, um, median=False)
        except Exception as exc:
            logger.exception("could not make the median preview")
            self._status.setText(f"Could not show the filter: {exc}")
            return
        finally:
            QApplication.restoreOverrideCursor()

        from ..loaders.layer_spec import units_like

        name = overview.names[overview.label]
        scale = tuple(float(v) * overview.step for v in overview.voxel_yx)
        translate = tuple(float(t) for t in image.translate)[-2:]
        top = float(np.percentile(channel[tissue], 99.5)) if tissue.any() else float(channel.max())

        def top_of(density):
            inside = density[filled]
            return float(np.percentile(inside, 99.5)) if inside.size else 1.0
        layers = [
            (f"Median preview: {name}", channel, (0.0, top), "gray"),
            (f"Median preview: local median ({self._median_size.value():.0f} µm)", background, (0.0, top), "gray"),
            (f"Median preview: {name} less the median", left, (0.0, top), "green"),
            # Each density on its own scale: what matters is where it is high, not how high.
            ("Median preview: cell density, no median", without, (0.0, top_of(without)), "magma"),
            ("Median preview: cell density, median taken away", with_median, (0.0, top_of(with_median)), "magma"),
        ]
        source = source_of(image)
        for layer in [layer for layer in self._viewer.layers if layer.name.startswith("Median preview:")]:
            self._viewer.layers.remove(layer)
        for index, (title, data, limits, colormap) in enumerate(layers):
            layer = self._viewer.add_image(
                np.asarray(data, dtype=np.float32), name=title, scale=scale, translate=translate,
                contrast_limits=limits if limits[1] > limits[0] else None, colormap=colormap,
                visible=index == len(layers) - 1, blending="additive", **units_like(image, 2),
            )
            tie_to_sample(layer, source)
        used = "with" if self._use_median.isChecked() else "without"
        self._status.setText(
            f"Median filter shown on {name}: five layers “Median preview: …”, the density "
            f"{used} it on top. The search now runs {used} it — tick or untick "
            "“Take away the local median” to change that."
        )

    def find_automatically(self) -> None:
        """Propose the junction of the open image and put it in the line layer."""
        from .. import junction as jn

        image = self.image_layer()
        if image is None:
            self._status.setText("Open an image first.")
            return
        specs = self._sample_specs(image)
        options = self.junction_options(save=False)
        self._status.setText("Looking for the junction…")
        try:
            overview, result = jn.analyse_specs(specs, options)
        except Exception as exc:
            logger.exception("automatic junction failed")
            self._status.setText(f"Could not find the junction: {exc}")
            return
        flags = "; ".join(result.flags) or "no flags"
        # A single sample gets its QC picture as a batch's samples do.
        try:
            source = source_of(image)
            qc_dir = self.qc_folder() or (Path(source).parent / "junction_QC" if source else None)
            if qc_dir is not None:
                name = jn.fu_name(Path(source)) if source else str(image.name)
                picture = jn.file_picture(qc_dir, name, result.status, jn.qc_picture(overview, result, name))
                flags += f" (QC picture: {picture.parent.name}/{picture.name})"
        except Exception:
            logger.exception("could not write the QC picture")
        if result.line_um is None:
            self._log(f"{image.name}: {result.status} — {flags}")
            return

        translate = np.asarray([float(t) for t in image.translate][-2:])
        layer = self.line_layer(create=True)
        self._updating = True
        try:
            layer.data = []
            layer.add_paths([np.asarray(result.line_um, dtype=float) + translate])
            index = self._side.findText(result.side)
            if index >= 0:
                self._side.blockSignals(True)
                self._side.setCurrentIndex(index)
                self._side.blockSignals(False)
        finally:
            self._updating = False
        # The band as the automatic run makes it — the organoid's own tissue
        # within the band width, ends following its outline — rather than the
        # strip a drawn line gets. Moving the line, or changing a setting,
        # rebuilds it as a strip from the line.
        shape, voxel, grid_translate = self._grid(image)
        labels = jn.full_resolution_band(result, overview, options)
        if labels.shape == shape and labels.any():
            self._labels, self._curve, self._voxel = labels, fu.fit_spline(result.line_um), voxel
            self._show(labels, image, voxel, grid_translate)
            if self._auto_save.isChecked():
                self.save()
        else:
            self.update_region()
        self._log(f"{image.name}: {result.status} — {flags}")

    def qc_folder(self) -> Path | None:
        """Where a run writes its report: “junction_QC” in the experiment folder."""
        experiment = getattr(self._app, "experiment_widget", None)
        folder = experiment.folder() if experiment is not None else None
        return Path(folder) / "junction_QC" if folder else None

    def open_qc_folder(self) -> None:
        from qtpy.QtCore import QUrl
        from qtpy.QtGui import QDesktopServices

        folder = self.qc_folder()
        if folder is None or not folder.is_dir():
            self._status.setText("No QC report yet — choose a folder in Experiment setup and run.")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

    def stop(self) -> None:
        self._cancelled = True
        self._log("Stopping: no new samples start; those already running finish…")

    def run_batch(self) -> None:
        """Junction, band and QC for every sample selected in Experiment setup."""
        from .. import ims_store
        from .. import junction as jn

        experiment = getattr(self._app, "experiment_widget", None)
        if experiment is None:
            self._status.setText("The Experiment setup panel is not available.")
            return
        if self._worker is not None:
            self._status.setText("A run is already going — stop it first.")
            return
        entries = [entry for entry in experiment.selected_entries() if entry.readable]
        if not entries:
            self._status.setText("No samples. Choose a folder in Experiment setup and scan it first.")
            return
        paths = [entry.path for entry in entries]
        # An .ims is written into, so it cannot stay open; a TIFF's sidecar can.
        inside = [path for path in paths if not ims_store.uses_sidecar(path)]
        if inside:
            experiment._close_paths(inside)

        qc_dir = self.qc_folder() or paths[0].parent / "junction_QC"
        options = self.junction_options(save=True, qc_dir=qc_dir)
        self._log_view.clear()
        self._cancelled = False
        self._run_button.setEnabled(False)
        self._stop_button.setEnabled(True)
        self._log(f"Finding the junction of {len(paths)} sample(s); QC goes to {qc_dir}.")

        relay = self._relay
        workers = int(self._workers.value())

        from napari.qt.threading import thread_worker

        @thread_worker
        def _run():
            return jn.run(paths, options, progress=relay.message.emit,
                          should_cancel=lambda: self._cancelled, workers=workers)

        def _done(outcomes) -> None:
            counts = {status: sum(o.status == status for o in outcomes)
                      for status in (jn.STATUS_OK, jn.STATUS_CHECK, jn.STATUS_FAIL)}
            saved = sum(1 for o in outcomes if o.saved)
            self._log(
                f"Done: {counts[jn.STATUS_OK]} ok, {counts[jn.STATUS_CHECK]} to check, "
                f"{counts[jn.STATUS_FAIL]} failed. Band stored for {saved} of {len(outcomes)}. "
                f"Report: {qc_dir / 'junction_QC.xlsx'}"
            )
            if hasattr(experiment, "_refresh_entries"):
                experiment._refresh_entries([o.path for o in outcomes])

        def _finished() -> None:
            self._worker = None
            self._run_button.setEnabled(True)
            self._stop_button.setEnabled(False)

        worker = _run()
        worker.returned.connect(_done)
        worker.errored.connect(lambda exc: self._log(f"Failed: {exc}"))
        worker.finished.connect(_finished)
        self._worker = worker
        worker.start()

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
            "follows_tissue": bool(self._follow_tissue.isChecked()),
            "past_tissue_edge_um": float(self._margin.value()) if self._follow_tissue.isChecked() else 0.0,
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
            self.region_saved.emit(str(path))
        return written
