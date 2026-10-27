"""Dock panel: the cells — segment the whole assembloid, count in the junction region, score per channel.

One panel and one **Run**: every sample is segmented whole (both organoids),
its junction region's cells are counted, every cell is scored positive or not
per channel and placed (distance to the junction line and to the tissue edge),
and the counts, the per-sample metrics and every cell go into the workbook.
The choices that change the answer are on top — the samples, the model and the
diameter, the region and the rule — and every finer setting is under
**Settings**, which opens like a drop-down. Two more sections open the same
way: training a Cellpose model, and **Segment only** — Cellpose without the
junction or any scoring, writing the label maps the Analysis panel reads.

The work is done by :class:`JunctionAnalysisWidget` with the Cellpose settings
of :class:`SegmentationWidget`: this panel builds both and lays their controls
out. They stay reachable as ``app.segmentation_widget`` and
``app.junction_analysis_widget``, which other panels borrow from.
"""

from __future__ import annotations

from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..utils import get_logger
from . import junction_analysis_widget as jaw
from .session_state import NOSAVE_PROPERTY

logger = get_logger("cells_widget")

#: Which samples a run analyses.
SAMPLES_OPEN = "The image on screen"
SAMPLES_EXPERIMENT = jaw.FROM_EXPERIMENT
SAMPLES_REVIEW = jaw.FROM_REVIEW
SAMPLES = (SAMPLES_OPEN, SAMPLES_EXPERIMENT, SAMPLES_REVIEW)


class Collapsible(QWidget):
    """A heading that opens and closes the section under it, like a drop-down."""

    def __init__(self, title: str, body: QWidget, opened: bool = False, parent: QWidget | None = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        self.button = QToolButton()
        self.button.setText(title)
        self.button.setCheckable(True)
        self.button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.button.setStyleSheet("QToolButton { border: none; font-weight: bold; }")
        self.button.toggled.connect(self.set_open)
        layout.addWidget(self.button)
        self.body = body
        layout.addWidget(body)
        self.set_open(opened)

    def set_open(self, opened: bool) -> None:
        self.button.blockSignals(True)
        self.button.setChecked(opened)
        self.button.blockSignals(False)
        self.button.setArrowType(Qt.DownArrow if opened else Qt.RightArrow)
        self.body.setVisible(opened)

    def is_open(self) -> bool:
        return self.button.isChecked()


def _form(parent: QWidget | None = None) -> tuple[QWidget, QFormLayout]:
    holder = QWidget(parent)
    form = QFormLayout(holder)
    form.setContentsMargins(0, 0, 0, 0)
    form.setLabelAlignment(Qt.AlignRight)
    return holder, form


def _group(title: str, *rows) -> QGroupBox:
    """A titled box of ``(label, widget)`` rows; a ``None`` label takes the whole width."""
    box = QGroupBox(title)
    form = QFormLayout(box)
    form.setLabelAlignment(Qt.AlignRight)
    for label, widget in rows:
        if widget is None:
            continue
        if label is None:
            form.addRow(widget)
        else:
            form.addRow(label, widget)
    return box


class CellsWidget(QWidget):
    """Segment the whole assembloid, count in the junction region, score every cell — one Run."""

    def __init__(self, app, parent: QWidget | None = None):
        super().__init__(parent)
        from .segmentation_widget import SegmentationWidget

        self._app = app
        # The parts that do the work. Built first, then their controls are laid
        # out here; their own layouts stay behind, hidden.
        self._parts = QWidget(self)
        self._parts.setVisible(False)
        self.segmentation = SegmentationWidget(app.viewer, parent=self._parts, app=app)
        self.junction = jaw.JunctionAnalysisWidget(app, parent=self._parts)
        app.segmentation_widget = self.segmentation
        app.junction_analysis_widget = self.junction
        seg, ja = self.segmentation, self.junction
        # The analysis's own "Samples" choice is set from this panel's,
        # so it is not a setting of its own to keep.
        ja._source.setProperty(NOSAVE_PROPERTY, True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(seg._backend_notice)
        note = QLabel(
            "Segments every cell of each sample, counts those in the junction region, "
            "scores every cell positive or not per channel, and places it (distance to "
            "the junction line and to the tissue edge). One Run; the workbook is written "
            "when it ends."
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        # -- what to run on ------------------------------------------------------
        top = QGroupBox("Run on")
        top_layout = QVBoxLayout(top)
        holder, form = _form()
        self._samples = QComboBox()
        self._samples.addItems(list(SAMPLES))
        self._samples.setToolTip(
            "• The image on screen: the sample that is open; its cells are drawn over it.\n"
            "• Selected in Experiment setup: every sample selected there (all when none is); "
            "nothing has to be open.\n"
            "• Sent on by a junction review: the samples of a review.csv whose decision is "
            "ticked; the review's status, decision and note go beside each count."
        )
        form.addRow("Samples", self._samples)
        top_layout.addWidget(holder)

        # The review file, for samples sent on by a review.
        self._review, form = _form()
        form.addRow("Review file", ja._review_row)
        form.addRow("Count the", ja._decisions_row)
        form.addRow("", ja._review_summary)
        top_layout.addWidget(self._review)
        self._open_sample, form = _form()
        form.addRow("Sample", ja._image_combo)
        top_layout.addWidget(self._open_sample)

        holder, form = _form()
        form.addRow("Model", seg._model_box.parentWidget())
        form.addRow("Diameter", seg._diameter)
        form.addRow("Hand-drawn region: name contains", ja._region_edit)
        form.addRow("Positive when", ja._method)
        form.addRow("k", ja._k)
        top_layout.addWidget(holder)
        layout.addWidget(top)

        # -- buttons ------------------------------------------------------------
        row = QHBoxLayout()
        self._run_button = QPushButton("Run")
        self._run_button.clicked.connect(self.run)
        row.addWidget(self._run_button)
        self._stop_button = QPushButton("Stop")
        self._stop_button.setToolTip("Stop after the sample being analysed.")
        self._stop_button.clicked.connect(self.stop)
        row.addWidget(self._stop_button)
        self._export_button = QPushButton("Export…")
        self._export_button.setToolTip(
            "The counts and metrics per sample and channel, the combinations of channels, "
            "the region's cells, the section's cells (when still in memory) and the "
            "settings, in one workbook. A run over samples writes it by itself."
        )
        self._export_button.clicked.connect(lambda: ja.export())
        row.addWidget(self._export_button)
        self._qc_button = QPushButton("QC…")
        self._qc_button.setToolTip("Open the QC pictures of the counted regions, sorted into ok, check and fail.")
        self._qc_button.clicked.connect(ja.open_qc_folder)
        row.addWidget(self._qc_button)
        layout.addLayout(row)

        # -- settings, closed until asked for -------------------------------------
        settings = QWidget()
        settings_layout = QVBoxLayout(settings)
        settings_layout.setContentsMargins(8, 0, 0, 0)
        settings_layout.addWidget(_group(
            "Cellpose",
            (None, seg._model_note),
            ("Custom model", seg._custom_edit.parentWidget()),
            ("Mode", seg._mode_box),
            ("Flow threshold", seg._flow_threshold),
            ("Cell probability", seg._cellprob),
            ("Stitch threshold", seg._stitch),
            ("Stitch gap", seg._stitch_gap),
            ("Minimum diameter", seg._min_diameter),
            ("Maximum diameter", seg._max_diameter),
            ("Normalise", seg._normalize),
        ))
        settings_layout.addWidget(_group(
            "Compute",
            ("GPU", seg._use_gpu),
            ("Device", seg._device_label.parentWidget()),
            ("Batch size", seg._batch_size),
            ("Segment at most", seg._max_voxels),
        ))
        settings_layout.addWidget(_group(
            "Positive cells and the section",
            ("Threshold from", ja._threshold_from),
            ("Fixed threshold", ja._fixed),
            ("Compare the cell's", ja._statistic),
            ("Measure around nuclei", ja._expand),
            ("", ja._find),
            ("", ja._region_only),
            ("", ja._positives_only),
            ("", ja._save_cells),
        ))
        self._settings = Collapsible("Settings", settings)
        layout.addWidget(self._settings)

        self._training = Collapsible("Train a Cellpose model", seg.training)
        layout.addWidget(self._training)
        self._segment_only = Collapsible("Segment only: label maps for the Analysis panel", self._segment_only_body())
        layout.addWidget(self._segment_only)

        # -- results ------------------------------------------------------------
        ja._table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(ja._table, stretch=2)
        layout.addWidget(ja._log_view, stretch=1)
        layout.addWidget(ja._progress)
        self._status = QLabel("")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)
        for label in (ja._status, seg._status, seg.batch._status):
            label.setParent(self._parts)

        self._samples.currentTextChanged.connect(lambda _text: self._update())
        self._update()

        # Run / Stop follow the analysis.
        self._timer = QTimer(self)
        self._timer.setInterval(400)
        self._timer.timeout.connect(self._poll)
        self._timer.start()

    def _segment_only_body(self) -> QWidget:
        """Cellpose alone, on the same samples: label maps and one row per object, no scoring."""
        seg = self.segmentation
        batch = seg.batch
        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(8, 0, 0, 0)
        note = QLabel(
            "Segments one channel and measures every object, without the junction or any "
            "scoring — for the Analysis panel, which reads the label maps stored with each "
            "sample. Uses the model, diameter and Settings above."
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        self._open_channels, form = _form()
        form.addRow("Segment", seg._channel_box)
        form.addRow("Measure", seg._measure_box)
        layout.addWidget(self._open_channels)
        self._batch_channels, form = _form()
        form.addRow("Segment channel", batch._channel_edit)
        form.addRow("Measure channel", batch._measure_edit)
        layout.addWidget(self._batch_channels)
        layout.addWidget(_group(
            "Results",
            ("Expand for intensity", seg._expand),
            ("Results", seg._add_layer),
            ("", seg._save_to_sample),
            ("Restrict", batch._restrict),
            ("Files", batch._save_labels),
        ))
        row = QHBoxLayout()
        self._segment_button = QPushButton("Segment only")
        self._segment_button.clicked.connect(self.segment_only)
        row.addWidget(self._segment_button)
        self._segment_export = QPushButton("Export objects…")
        self._segment_export.clicked.connect(self.export_objects)
        row.addWidget(self._segment_export)
        row.addStretch(1)
        layout.addLayout(row)
        self._objects_page = seg._tabs.widget(1)  # the Objects tab: one row per object
        layout.addWidget(self._objects_page)
        layout.addWidget(batch._log_view)
        return body

    # -- state ------------------------------------------------------------------

    def samples(self) -> str:
        return self._samples.currentText()

    def _update(self) -> None:
        """Show what the samples choice uses; hide the rest."""
        samples = self.samples()
        # The analysis's own sample choice follows this one.
        source = SAMPLES_REVIEW if samples == SAMPLES_REVIEW else SAMPLES_EXPERIMENT
        if self.junction._source.currentText() != source:
            self.junction._source.setCurrentText(source)
        self._review.setVisible(samples == SAMPLES_REVIEW)
        self._open_sample.setVisible(samples == SAMPLES_OPEN)
        self._open_channels.setVisible(samples == SAMPLES_OPEN)
        self._batch_channels.setVisible(samples != SAMPLES_OPEN)
        self._run_button.setText("Analyse this sample" if samples == SAMPLES_OPEN else "Analyse the samples")
        self._poll()

    def _segmenting(self) -> bool:
        seg = self.segmentation
        return bool(seg._worker is not None or (seg.batch is not None and seg.batch.is_running()))

    def running(self) -> bool:
        return self.junction._worker is not None or self._segmenting()

    def _poll(self) -> None:
        busy = self.running()
        blocked = self.segmentation._backend_notice.isVisible()
        self._run_button.setEnabled(not busy and not blocked)
        self._segment_button.setEnabled(not busy and not blocked)
        # A single image's segmentation cannot be stopped half way; a batch or the analysis can.
        self._stop_button.setEnabled(self.junction._worker is not None
                                     or (self.segmentation.batch is not None and self.segmentation.batch.is_running()))
        if self._segmenting():
            label = self.segmentation._status if self.segmentation._worker is not None else self.segmentation.batch._status
        else:
            label = self.junction._status
        if self._status.text() != label.text():
            self._status.setText(label.text())

    # -- actions ------------------------------------------------------------------

    def run(self) -> None:
        """Analyse the open sample, or the samples chosen."""
        self._update()
        if self.samples() == SAMPLES_OPEN:
            self.junction.count_current()
        else:
            self.junction.run_batch()
        self._poll()

    def segment_only(self) -> None:
        """Cellpose alone: the open image, or every sample selected in Experiment setup."""
        self._update()
        if self.samples() == SAMPLES_OPEN:
            self.segmentation.run()
        elif self.samples() == SAMPLES_REVIEW:
            self.junction._status.setText("Segment only runs on the image on screen or the Experiment selection.")
        elif self.segmentation.batch is not None:
            self.segmentation.batch.run()
        self._poll()

    def export_objects(self) -> None:
        if self.samples() == SAMPLES_OPEN:
            self.segmentation.export_objects()
        elif self.segmentation.batch is not None:
            self.segmentation.batch.export()

    def stop(self) -> None:
        if self.segmentation.batch is not None and self.segmentation.batch.is_running():
            self.segmentation.batch.stop()
        if self.junction._worker is not None:
            self.junction.stop()

    # -- what other panels and the session use --------------------------------------

    def settings(self):
        """The Cellpose settings, for whoever borrows them."""
        return self.segmentation.settings()
