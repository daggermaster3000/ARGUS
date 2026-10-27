"""Dock panel: find the cells — segment and measure, or count them in the junction region.

One panel for what used to be two (Segmentation, with its Batch and Train tabs,
and Junction analysis). The choices that change what a run does are on top —
the task, the samples, the channels, the model and the diameter, and for the
junction count the region and the rule — and every finer setting is under
**Settings**, which opens like a drop-down. Cellpose's settings are there once,
for both tasks.

The work is still done by :class:`SegmentationWidget`,
:class:`BatchSegmentationTab` and :class:`JunctionAnalysisWidget`: this panel
builds them, lays their controls out, and sends **Run** to the one the task and
the samples call for. They stay reachable as ``app.segmentation_widget`` and
``app.junction_analysis_widget``, which other panels borrow settings from.
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
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..utils import get_logger
from . import junction_analysis_widget as jaw
from .session_state import NOSAVE_PROPERTY

logger = get_logger("cells_widget")

#: What a run does.
TASK_SEGMENT = "Segment and measure"
TASK_JUNCTION = "Count in the junction region"
TASKS = (TASK_SEGMENT, TASK_JUNCTION)

#: Which samples it does it on.
SAMPLES_OPEN = "The image on screen"
SAMPLES_EXPERIMENT = jaw.FROM_EXPERIMENT
SAMPLES_REVIEW = jaw.FROM_REVIEW


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
    """Segment and measure cells, or count them in the junction region — one panel."""

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
        seg, ja, batch = self.segmentation, self.junction, self.segmentation.batch
        # The junction panel's own "Samples" choice is set from this panel's,
        # so it is not a setting of its own to keep.
        ja._source.setProperty(NOSAVE_PROPERTY, True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(seg._backend_notice)

        # -- what to do ---------------------------------------------------------
        top = QGroupBox("What to do")
        top_layout = QVBoxLayout(top)
        holder, form = _form()
        self._task = QComboBox()
        self._task.addItems(list(TASKS))
        self._task.setToolTip(
            "• Segment and measure: Cellpose on one channel, every object measured; labels "
            "can be saved with each sample.\n"
            "• Count in the junction region: the nuclei in each sample's junction band, "
            "each cell scored positive or not in every channel, with its distance to the "
            "line and to the tissue edge."
        )
        form.addRow("Task", self._task)
        self._samples = QComboBox()
        self._samples.setToolTip(
            "• The image on screen: the sample that is open.\n"
            "• Selected in Experiment setup: every sample selected there (all when none is); "
            "nothing has to be open.\n"
            "• Sent on by a junction review (junction count only): the samples of a "
            "review.csv whose decision is ticked."
        )
        form.addRow("Samples", self._samples)
        top_layout.addWidget(holder)

        # The review file, for samples sent on by a review.
        self._review, form = _form()
        form.addRow("Review file", ja._review_row)
        form.addRow("Count the", ja._decisions_row)
        form.addRow("", ja._review_summary)
        top_layout.addWidget(self._review)

        # Channels: layers of the open image, or names matched in each file.
        self._open_channels, form = _form()
        form.addRow("Segment", seg._channel_box)
        form.addRow("Measure", seg._measure_box)
        top_layout.addWidget(self._open_channels)
        self._batch_channels, form = _form()
        form.addRow("Segment channel", batch._channel_edit)
        form.addRow("Measure channel", batch._measure_edit)
        top_layout.addWidget(self._batch_channels)
        self._open_sample, form = _form()
        form.addRow("Sample", ja._image_combo)
        top_layout.addWidget(self._open_sample)

        holder, form = _form()
        form.addRow("Model", seg._model_box.parentWidget())
        form.addRow("Diameter", seg._diameter)
        top_layout.addWidget(holder)

        # The junction count's own choices.
        self._junction_top, form = _form()
        form.addRow("Hand-drawn region: name contains", ja._region_edit)
        form.addRow("Positive when", ja._method)
        form.addRow("k", ja._k)
        form.addRow("", ja._whole)
        top_layout.addWidget(self._junction_top)
        layout.addWidget(top)

        # -- buttons ------------------------------------------------------------
        row = QHBoxLayout()
        self._run_button = QPushButton("Run")
        self._run_button.clicked.connect(self.run)
        row.addWidget(self._run_button)
        self._stop_button = QPushButton("Stop")
        self._stop_button.clicked.connect(self.stop)
        row.addWidget(self._stop_button)
        self._export_button = QPushButton("Export…")
        self._export_button.clicked.connect(self.export)
        row.addWidget(self._export_button)
        self._qc_button = QPushButton("QC…")
        self._qc_button.setToolTip("Open the junction count's QC pictures, sorted into ok, check and fail.")
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
        self._segment_settings = _group(
            "Segment and measure",
            ("Expand for intensity", seg._expand),
            ("Results", seg._add_layer),
            ("", seg._save_to_sample),
            ("Restrict", batch._restrict),
            ("Files", batch._save_labels),
        )
        settings_layout.addWidget(self._segment_settings)
        self._junction_settings = _group(
            "Junction count",
            ("Threshold from", ja._threshold_from),
            ("Fixed threshold", ja._fixed),
            ("Compare the cell's", ja._statistic),
            ("Measure around nuclei", ja._expand),
            ("", ja._find),
            ("", ja._keep_section),
            ("", ja._save_cells),
        )
        settings_layout.addWidget(self._junction_settings)
        self._settings = Collapsible("Settings", settings)
        layout.addWidget(self._settings)

        self._training = Collapsible("Train a Cellpose model", seg.training)
        layout.addWidget(self._training)

        # -- results ------------------------------------------------------------
        self._results = QTabWidget()
        self._results.setProperty(NOSAVE_PROPERTY, True)
        self._objects_page = seg._tabs.widget(1)  # the Objects tab: one row per object
        self._results.addTab(self._objects_page, "Objects")
        batch_page = QWidget()
        batch_layout = QVBoxLayout(batch_page)
        batch_layout.setContentsMargins(2, 2, 2, 2)
        batch_layout.addWidget(batch._log_view, stretch=1)
        self._results.addTab(batch_page, "Batch log")
        counts_page = QWidget()
        counts_layout = QVBoxLayout(counts_page)
        counts_layout.setContentsMargins(2, 2, 2, 2)
        counts_layout.addWidget(ja._table, stretch=2)
        counts_layout.addWidget(ja._log_view, stretch=1)
        self._results.addTab(counts_page, "Junction counts")
        self._results.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self._results, stretch=1)

        self._status = QLabel("")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)
        self._statuses = {"open": seg._status, "batch": batch._status, "junction": ja._status}
        for label in self._statuses.values():
            label.setParent(self._parts)

        self._task.currentTextChanged.connect(lambda _text: self._update())
        self._samples.currentTextChanged.connect(lambda _text: self._update())
        self._fill_samples()
        self._update()

        # Run / Stop follow whichever part is working.
        self._timer = QTimer(self)
        self._timer.setInterval(400)
        self._timer.timeout.connect(self._poll)
        self._timer.start()

    # -- state ------------------------------------------------------------------

    def task(self) -> str:
        return self._task.currentText()

    def samples(self) -> str:
        return self._samples.currentText()

    def _fill_samples(self) -> None:
        wanted = [SAMPLES_OPEN, SAMPLES_EXPERIMENT] + ([SAMPLES_REVIEW] if self.task() == TASK_JUNCTION else [])
        current = self._samples.currentText()
        if [self._samples.itemText(i) for i in range(self._samples.count())] == wanted:
            return
        self._samples.blockSignals(True)
        self._samples.clear()
        self._samples.addItems(wanted)
        index = self._samples.findText(current)
        self._samples.setCurrentIndex(index if index >= 0 else 0)
        self._samples.blockSignals(False)

    def _update(self) -> None:
        """Show what the task and the samples use; hide the rest."""
        self._fill_samples()
        junction, samples = self.task() == TASK_JUNCTION, self.samples()
        is_open = samples == SAMPLES_OPEN
        # The junction part's own sample choice follows this one.
        source = SAMPLES_REVIEW if junction and samples == SAMPLES_REVIEW else SAMPLES_EXPERIMENT
        if self.junction._source.currentText() != source:
            self.junction._source.setCurrentText(source)
        self._review.setVisible(junction and samples == SAMPLES_REVIEW)
        self._open_channels.setVisible(not junction and is_open)
        self._batch_channels.setVisible(not junction and not is_open)
        self._open_sample.setVisible(junction and is_open)
        self._junction_top.setVisible(junction)
        self._segment_settings.setVisible(not junction)
        self._junction_settings.setVisible(junction)
        self._qc_button.setVisible(junction)
        self._run_button.setText(self._run_text())
        self._results.setCurrentIndex(2 if junction else (0 if is_open else 1))
        self._poll()

    def _run_text(self) -> str:
        if self.task() == TASK_JUNCTION:
            return "Count in this sample" if self.samples() == SAMPLES_OPEN else "Count in the samples"
        return "Segment this image" if self.samples() == SAMPLES_OPEN else "Segment the samples"

    def _active_status(self) -> QLabel:
        if self.task() == TASK_JUNCTION:
            return self._statuses["junction"]
        return self._statuses["open" if self.samples() == SAMPLES_OPEN else "batch"]

    def running(self) -> bool:
        return bool(self.segmentation._worker is not None
                    or (self.segmentation.batch is not None and self.segmentation.batch.is_running())
                    or self.junction._worker is not None)

    def _poll(self) -> None:
        busy = self.running()
        blocked = self.segmentation._backend_notice.isVisible()
        self._run_button.setEnabled(not busy and not blocked)
        self._stop_button.setEnabled(busy and not (self.segmentation._worker is not None
                                                   and self.task() == TASK_SEGMENT and self.samples() == SAMPLES_OPEN))
        text = self._active_status().text()
        if self._status.text() != text:
            self._status.setText(text)

    # -- actions ------------------------------------------------------------------

    def run(self) -> None:
        """Send the run to the part the task and the samples call for."""
        self._update()
        junction, samples = self.task() == TASK_JUNCTION, self.samples()
        if junction:
            if samples == SAMPLES_OPEN:
                self.junction.count_current()
            else:
                self.junction.run_batch()
        elif samples == SAMPLES_OPEN:
            self.segmentation.run()
        elif self.segmentation.batch is not None:
            self.segmentation.batch.run()
        self._poll()

    def stop(self) -> None:
        if self.segmentation.batch is not None and self.segmentation.batch.is_running():
            self.segmentation.batch.stop()
        if self.junction._worker is not None:
            self.junction.stop()

    def export(self) -> None:
        if self.task() == TASK_JUNCTION:
            self.junction.export()
        elif self.samples() == SAMPLES_OPEN:
            self.segmentation.export_objects()
        elif self.segmentation.batch is not None:
            self.segmentation.batch.export()

    # -- what other panels and the session use --------------------------------------

    def settings(self):
        """The Cellpose settings, for whoever borrows them."""
        return self.segmentation.settings()
