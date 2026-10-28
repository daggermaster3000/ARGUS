"""The analysis of the cells: segment the whole assembloid, count in the junction region, score per channel.

The work is in :mod:`microscopy_viewer.junction_analysis`. This module picks the
samples — the open one, those selected in Experiment setup, or those a junction
review sent on (its ``review.csv``) — runs the count off the main thread — it segments nuclei, which takes
seconds on a GPU and longer without — and shows what came back three ways: a
table in the panel, the cells drawn over the image with one layer per channel
they are positive for, and a workbook.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from qtpy.QtCore import QObject, Qt, QTimer, Signal
from qtpy.QtGui import QColor, QFontDatabase
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
    QPlainTextEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .. import fusion as fu
from .. import junction as jn
from .. import junction_analysis as ja
from .. import junction_review as jr
from ..exports import WORKBOOK_FILTER, default_stem
from ..utils import format_number, get_logger
from .sample_files import source_of, tie_to_sample
from .session_state import NAME_PREFIX

logger = get_logger("junction_analysis_widget")

#: Layer names: the region's cells; with the whole section, every nucleus and
#: the same grown; and a suffix for one layer per channel of its positive cells.
CELLS_LAYER = "Junction cells"
NUCLEI_LAYER = "Nuclei (whole section)"
GROWN_LAYER = "Nuclei grown by {um} µm"

#: What re-scoring does, for its button.
RESCORE_TIP = (
    "Score the samples again with the positivity settings as they are now — rule, k, ring "
    "contrast, threshold from — without segmenting anything: from the cells the newest run "
    "saved for each (cells_analysis/<run>/cells), whatever it was segmented with. A region "
    "redrawn since is counted in. Seconds a sample; a new results folder with everything. "
    "Samples never segmented are left out and named."
)

#: What the results folder holds, for the button that opens it.
RESULTS_TIP = (
    "Open this run's folder, “cells_analysis/<date>_<time>” in the experiment folder:\n"
    "• counts.xlsx — counts and metrics per sample and channel, combinations, the region's cells, settings;\n"
    "• cells/<sample>.csv — every cell of the whole section;\n"
    "• region/ok, check, fail/ — a picture of each counted region, positive cells ringed;\n"
    "• positive cells/ — each whole section, its positive cells ringed and outlined in their stain;\n"
    "• labels/ — each whole section's label maps: every cell, the same grown, each channel's "
    "positive cells in their own shape;\n"
    "• README.txt — what is where, and how the run was made."
)
#: Marks the layers this panel drew, whatever channel they are named after.
DRAWN_KEY = "argus_junction_cells"
POSITIVE_SUFFIX = "+ cells"
#: Suffix of the layer of each channel's positive cells in their own shape.
SHAPE_SUFFIX = " positive shape"

#: Columns of the panel's table; the workbook has these and the rest.
TABLE_COLUMNS = (
    "Sample", "Channel", "Cells in region", "Positive", "Positive (%)",
    "Cells in the whole section", "Positive in the whole section (%)", "Enrichment in the region",
    "Positive per mm²", "Positive per mm of junction", "Threshold",
    "Region drawn by", "Finder status", "Finder second search", "Review decision",
)

#: Where the samples of a run come from.
FROM_EXPERIMENT = "Selected in Experiment setup"
FROM_REVIEW = "Sent on by a junction review"
#: The word that marks a region as drawn by hand (see ``CountOptions.region_keyword``).
REGION_KEYWORD = "manual"

#: The points layer's outline argument and attribute: "border" from napari
#: 0.5, "edge" before.
try:
    from napari.layers import Points as _Points

    BORDER, BORDER_ATTR = ("border_color", "border_color") if hasattr(_Points, "border_color") \
        else ("edge_color", "edge_color")
except Exception:  # pragma: no cover - napari missing
    BORDER, BORDER_ATTR = "border_color", "border_color"

#: Colour of a sample's name in the table: it opens the sample when clicked.
LINK_COLOR = "#5aa9ff"
#: Size of the ring drawn round a positive cell from its table, in µm.
RING_UM = 14.0

#: How the positive cells are drawn: a red outline this many pixels thick.
POSITIVE_COLOR = "red"
POSITIVE_CONTOUR = 2


class _Relay(QObject):
    """Carries worker-thread progress text onto the GUI thread."""

    message = Signal(str)


class JunctionAnalysisWidget(QWidget):
    """Count the cells in the junction region; say which channels they are positive for."""

    def __init__(self, app, parent: QWidget | None = None):
        super().__init__(parent)
        self._app = app
        self._viewer = app.viewer
        self._worker = None
        self._cancelled = False
        self._results: list[ja.JunctionCounts] = []
        self._run_dir: Path | None = None
        self._options = ja.CountOptions()
        self._relay = _Relay()
        self._relay.message.connect(self._log)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        note = QLabel(
            "Counts the nuclei in the junction region and scores every cell in each "
            "channel. Make the region first, in the Fusion region panel — or let this "
            "find the junction itself."
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        layout.addWidget(self._build_samples_box())
        layout.addWidget(self._build_settings_box())
        layout.addLayout(self._build_buttons())

        self._table = QTableWidget(0, len(TABLE_COLUMNS), self)
        self._table.setHorizontalHeaderLabels(list(TABLE_COLUMNS))
        self._table.verticalHeader().setVisible(False)
        # napari's stylesheet renders alternating rows as blank stripes.
        self._table.setAlternatingRowColors(False)
        self._table.setSortingEnabled(True)
        self._table.setToolTip(
            "Click a sample's name (or double-click its row) to open it with its results drawn "
            "over it: every positive cell, per channel. Right-click for its pictures and its cells."
        )
        self._table.cellClicked.connect(lambda row, column: column == 0 and self.open_row(row))
        self._table.cellDoubleClicked.connect(lambda row, _column: self.open_row(row))
        self._table.setContextMenuPolicy(Qt.CustomContextMenu)
        self._table.customContextMenuRequested.connect(self._row_menu)
        self._pending_show: ja.JunctionCounts | None = None
        layout.addWidget(self._table, stretch=2)

        self._log_view = QPlainTextEdit()
        self._log_view.setReadOnly(True)
        self._log_view.setMaximumHeight(110)
        self._log_view.setPlaceholderText("Progress appears here.")
        layout.addWidget(self._log_view, stretch=1)

        # One animated line while a run goes: bar, where it is, time so far and left.
        self._progress = QLabel("")
        self._progress.setFont(QFontDatabase.systemFont(QFontDatabase.FixedFont))
        self._progress.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._progress.setToolTip(
            "How far the run is and about how long it has left. The estimate is the time so far "
            "scaled by what is left, so it settles once the first sample's tiles are under way."
        )
        self._progress.setVisible(False)
        layout.addWidget(self._progress)
        self._tracker: ja.RunProgress | None = None
        self._tick = 0
        self._ticker = QTimer(self)
        self._ticker.setInterval(250)
        self._ticker.timeout.connect(self._animate)

        self._status = QLabel("Open a sample and Count, or select samples in Experiment setup and Run.")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        try:
            self._viewer.layers.events.inserted.connect(self._on_layers_changed)
            self._viewer.layers.events.removed.connect(self._on_layers_changed)
        except Exception:  # pragma: no cover - a viewer stub without events
            logger.debug("could not subscribe to layer events", exc_info=True)
        self.refresh_layers()

    # -- construction ---------------------------------------------------------

    def _build_samples_box(self) -> QGroupBox:
        box = QGroupBox("Samples to run on")
        form = QFormLayout(box)
        form.setLabelAlignment(Qt.AlignRight)

        self._source = QComboBox()
        self._source.addItems([FROM_EXPERIMENT, FROM_REVIEW])
        self._source.setToolTip(
            "Which samples “Run” counts in.\n"
            "• Selected in Experiment setup: the samples selected there (all when none is).\n"
            "• Sent on by a junction review: the samples of a review.csv (written by the "
            "Junction review panel) whose decision is ticked below. The review's status, "
            "decision and note go into the workbook beside each count."
        )
        self._source.currentTextChanged.connect(self._update_source)
        form.addRow("Samples", self._source)

        row = QHBoxLayout()
        self._review_edit = QLineEdit()
        self._review_edit.setPlaceholderText("review.csv of a junction review")
        self._review_edit.setToolTip(
            "The review.csv in a QC folder (junction_QC, or a run's cells_analysis/<run>/region). Each "
            "sample is found from the file the review recorded, or by name in the "
            "experiment folder."
        )
        self._review_edit.editingFinished.connect(self._update_review_summary)
        row.addWidget(self._review_edit, stretch=1)
        browse = QPushButton("…")
        browse.setToolTip("Choose the review file")
        browse.clicked.connect(lambda: self._browse_review())
        row.addWidget(browse)
        self._review_row = QWidget()
        self._review_row.setLayout(row)
        row.setContentsMargins(0, 0, 0, 0)
        form.addRow("Review file", self._review_row)

        decisions = QHBoxLayout()
        decisions.setContentsMargins(0, 0, 0, 0)
        self._decisions: dict[str, QCheckBox] = {}
        for decision, label, on in ((jr.ACCEPT, "accepted", True), (jr.REDRAWN, "redrawn", True),
                                    (jr.UNDECIDED, "not reviewed yet", False)):
            check = QCheckBox(label)
            check.setChecked(on)
            check.toggled.connect(lambda _on: self._update_review_summary())
            decisions.addWidget(check)
            self._decisions[decision] = check
        decisions.addStretch(1)
        self._decisions_row = QWidget()
        self._decisions_row.setLayout(decisions)
        self._decisions_row.setToolTip(
            "Which decisions send a sample on. Excluded samples, and those still waiting "
            "to be redrawn, are never counted."
        )
        form.addRow("Count the", self._decisions_row)

        self._review_summary = QLabel("")
        self._review_summary.setWordWrap(True)
        form.addRow("", self._review_summary)
        self._update_source(self._source.currentText())
        return box

    def _build_settings_box(self) -> QGroupBox:
        box = QGroupBox("Cells and what counts as positive")
        form = QFormLayout(box)
        form.setLabelAlignment(Qt.AlignRight)

        self._image_combo = QComboBox()
        self._image_combo.setToolTip("Any channel of the sample to count in; all of its channels are measured.")
        form.addRow("Sample", self._image_combo)

        self._region_edit = QLineEdit(REGION_KEYWORD)
        self._region_edit.setToolTip(
            "Which region each sample is counted in, in this order:\n"
            "1. a junction region drawn by hand: the first stored one (in name order) whose "
            "name contains this word, whoever drew it;\n"
            f"2. else the junction finder's own region, “{fu.DEFAULT_KEY}”;\n"
            "3. else, when ticked below, the junction is found on the spot.\n"
            "A “Fusion region” layer on screen is used in preference for a single count. "
            "The table and workbook say, per sample, which region was used and who drew it, "
            "and what the junction finder said about the sample (status, flags, and whether "
            "its second search for sections without a dark line made the line)."
        )
        form.addRow("Hand-drawn region: name contains", self._region_edit)

        self._method = QComboBox()
        self._method.addItems(list(ja.METHODS))
        self._method.setToolTip(
            "How each channel's threshold is set. It is worked out per sample and per "
            "channel, and written beside every count.\n"
            "• Above the cells: median + k robust SDs of the cells' own values. For a "
            "region where most cells are negative, as with cells that had to migrate to "
            "be there. Adapts to staining and to autofluorescence all cells share.\n"
            "• Above the background: median + k robust SDs of the tissue between the "
            "cells. Stricter about what is signal; blind to a haze every cell carries.\n"
            "• Otsu: the best split of the cells into two groups. Right when there are "
            "two groups; with one it splits it anyway.\n"
            "• Fixed value: one number for every sample — the only rule under which "
            "samples are compared on the same scale.\n"
            "• Ring contrast: the cell's mean over the mean of a ring 3–6 µm round its "
            "nucleus — the cell against its own surroundings, so a haze or a thick part "
            "of the section does not make a cell positive. At least 1.5× by default "
            "(the validation's best local rule: 0.07 % false positives deep in the "
            "other organoid). A packed cell with no free ring is compared with the "
            "tissue between the cells near it."
        )
        self._method.currentTextChanged.connect(self._update_sensitivity)
        form.addRow("Positive when", self._method)

        self._threshold_from = QComboBox()
        self._threshold_from.addItems(list(ja.THRESHOLD_FROM))
        self._threshold_from.setToolTip(
            "Which cells the median and SD (or the Otsu split) are taken from. The same "
            "threshold then scores the region's cells and every other cell of the section.\n"
            "• The cells in the region: the bar is set by the region's own cells, so a "
            "region where a large share is positive raises it and undercounts.\n"
            "• All cells of the sample: the threshold comes from every cell of the "
            "section, both organoids. Holds when the region is full of positive cells.\n"
            "Not used by “Above the background” and “Fixed value”, which take nothing "
            "from the cells."
        )
        form.addRow("Threshold from", self._threshold_from)

        self._k = QDoubleSpinBox()
        self._k.setRange(0.5, 20.0)
        self._k.setSingleStep(0.5)
        self._k.setValue(ja.K_SD)
        # Named for the default of 28 Oct 2026 (k 7, median filter): a session saved
        # under the old default (k 3, plain mean) does not bring it back.
        self._k.setObjectName(NAME_PREFIX + "positivity.k (median filter default)")
        self._k.setSuffix(" SD")
        self._k.setToolTip(
            "How far above the centre a cell has to be. SDs are robust (from the MAD), so "
            "a few bright cells do not raise the bar on themselves. 7, on the mean less the "
            "local median, came closest to the hand counts (F1 0.38, Spearman 0.79 per mm of "
            "junction); 3 calls about four times as many cells, most of them in haze. "
            "Raise it if dim cells are called positive, lower it if clear ones are missed."
        )
        form.addRow("k", self._k)

        self._ring_ratio = QDoubleSpinBox()
        self._ring_ratio.setRange(1.0, 20.0)
        self._ring_ratio.setSingleStep(0.1)
        self._ring_ratio.setDecimals(2)
        self._ring_ratio.setValue(ja.RING_RATIO)
        self._ring_ratio.setSuffix(" ×")
        self._ring_ratio.setToolTip(
            "For “Ring contrast”: how many times brighter than its 3–6 µm ring a cell has "
            "to be. 1.5 is the validation's choice; raise it if hazy cells are called, "
            "lower it if clearly labelled cells next to other labelled cells are missed."
        )
        form.addRow("Ring contrast at least", self._ring_ratio)

        self._fixed = QDoubleSpinBox()
        self._fixed.setRange(0.0, 1e6)
        self._fixed.setDecimals(1)
        self._fixed.setValue(1000.0)
        self._fixed.setToolTip("The threshold for “Fixed value”, in the image's own intensity units, used for every scored channel.")
        form.addRow("Fixed threshold", self._fixed)

        self._statistic = QComboBox()
        self._statistic.addItems(list(ja.STATISTICS))
        self._statistic.setObjectName(NAME_PREFIX + "positivity.statistic (median filter default)")
        self._statistic.setToolTip(
            "Which figure of each cell is compared with the threshold.\n"
            f"• {ja.FILTERED} (the default): the cell's mean less the median of the "
            f"tissue {ja.MEDIAN_UM:g} µm round it, so a cell in a bright haze or a dim corner "
            "is set against its own surroundings. Best against the hand counts.\n"
            "• mean: the cell's mean as it is; a haze over part of the section raises "
            "every cell in it.\n"
            "• P95 or max pick up a label that fills only part of the cell, at the price "
            "of being noisier."
        )
        form.addRow("Compare the cell's", self._statistic)

        self._expand = QDoubleSpinBox()
        self._expand.setRange(0.0, 50.0)
        self._expand.setSingleStep(0.5)
        self._expand.setValue(3.0)
        self._expand.setSuffix(" µm")
        self._expand.setSpecialValueText("nucleus only")
        self._expand.setToolTip(
            "Each nucleus is grown by this far and the cell is measured over that: a "
            "label in the cytoplasm is not under the nucleus. Neighbours split the gap "
            "between them. Sizes in the table stay the nucleus's own."
        )
        form.addRow("Measure around nuclei", self._expand)

        self._find = QCheckBox("Find the junction when no region is stored")
        self._find.setChecked(True)
        self._find.setToolTip("With the settings on the Fusion region panel. Off skips samples that have no region yet.")
        form.addRow("", self._find)

        self._region_only = QCheckBox("Segment only the junction region (faster; nothing outside it)")
        self._region_only.setToolTip(
            "Off (the usual): the whole section is segmented, both organoids, in tiles with "
            "bare slide skipped; the region's cells are taken from that same pass, and every "
            "other cell is measured and scored against the region's thresholds, with its "
            "distance to the junction line and to the tissue edge. About 40 s per slide scan on a free GPU.\n"
            "On: only the region's bounding box is segmented — seconds — and the table has "
            "nothing about the rest of the section."
        )
        form.addRow("", self._region_only)
        self._positives_only = QCheckBox("Of the rest of the section, keep only the positive cells")
        self._positives_only.setToolTip(
            "Every cell of the section is measured either way, and the counts are the same. "
            "Off keeps a row for every one of them (about 100,000 a section), for profiles "
            "against the distance to the line; on keeps the region's cells and those "
            "positive for a channel. Each sample's cells are written to "
            "“junction_analysis/cells/<sample>.csv”."
        )
        self._region_only.toggled.connect(lambda on: self._positives_only.setEnabled(not on))
        form.addRow("", self._positives_only)
        self._shapes = QCheckBox("Outline the positive cells in their own stain")
        self._shapes.setChecked(True)
        self._shapes.setToolTip(
            "Each positive cell is also drawn in its real shape: the pixels of its channel "
            f"standing {ja.SHAPE_K:g} SDs over the tissue round them, joined to the cell they "
            "touch — cell bodies off the nucleus and processes, which the nucleus does not show. "
            "A layer per channel (“<channel> positive shape”), an outline in the section "
            "picture, and “labels/<sample> <channel> shape.tif”. Drawing only: which cells "
            "are positive does not change. A few seconds a channel per sample."
        )
        self._region_only.toggled.connect(lambda on: self._shapes.setEnabled(not on))
        form.addRow("", self._shapes)

        self._reuse = QCheckBox("Skip samples already segmented: re-score them from their saved cells")
        self._reuse.setChecked(True)
        self._reuse.setToolTip(
            "A sample segmented by an earlier run — with the same model, diameter, Cellpose "
            "thresholds and growth, every cell kept — is not segmented again: its saved cells "
            "(cells_analysis/<run>/cells) are scored with the rule set now, in seconds. Only "
            "samples not segmented yet, or segmented with other settings, go through Cellpose. "
            "Every sample is in the table and the new run folder either way."
        )
        form.addRow("", self._reuse)

        self._save_cells = QCheckBox("Store the counted cells with the sample")
        self._save_cells.setToolTip(
            "As the label map “junction cells”, with each channel's threshold and its "
            "positive cells. A TIFF's go in its sidecar; an .ims open in the viewer is "
            "skipped, since it would have to be closed to be written into."
        )
        form.addRow("", self._save_cells)

        self._update_sensitivity(self._method.currentText())
        return box

    def _build_buttons(self) -> QHBoxLayout:
        row = QHBoxLayout()
        self._count_button = QPushButton("Count in this sample")
        self._count_button.setToolTip(
            "Segment the nuclei in the region with the Segmentation panel's model and "
            "diameter, score them, and draw them: all cells as outlines, and one layer "
            "per channel with the cells positive in it outlined in red."
        )
        self._count_button.clicked.connect(lambda: self.count_current())
        row.addWidget(self._count_button)

        self._run_button = QPushButton("Run on the samples")
        self._run_button.setToolTip(
            "The samples chosen under “Samples to run on”: those selected in Experiment "
            "setup, or those a junction review sent on. Nothing has to be open.\n"
            "Every run, this one and a single sample, writes everything to one new folder, "
            "“cells_analysis/<date>_<time>” in the experiment folder: counts.xlsx, cells/, "
            "region/ (sorted into ok, check and fail), positive cells/ and a README."
        )
        self._run_button.clicked.connect(lambda: self.run_batch())
        row.addWidget(self._run_button)

        self._rescore_button = QPushButton("Re-score positivity")
        self._rescore_button.setToolTip(RESCORE_TIP)
        self._rescore_button.clicked.connect(lambda: self.run_batch(rescore_only=True))
        row.addWidget(self._rescore_button)

        self._stop_button = QPushButton("Stop")
        self._stop_button.setEnabled(False)
        self._stop_button.clicked.connect(self.stop)
        row.addWidget(self._stop_button)

        export = QPushButton("Export…")
        export.setToolTip("Counts, combinations of channels, every cell with all its figures, and the settings, in one workbook.")
        export.clicked.connect(self.export)
        row.addWidget(export)

        results = QPushButton("Results folder…")
        results.setToolTip(RESULTS_TIP)
        results.clicked.connect(self.open_results_folder)
        row.addWidget(results)
        return row

    def results_base(self, fallback: Path | None = None) -> Path | None:
        """The folder runs are written under: the experiment folder, else *fallback*."""
        experiment = getattr(self._app, "experiment_widget", None)
        folder = (experiment.folder() if experiment is not None else None) or fallback
        return Path(folder) if folder else None

    def new_run(self, options: ja.CountOptions, fallback: Path | None, label: str = ""):
        """*options* writing into a new run folder, and that folder (``None`` with nowhere to write)."""
        base = self.results_base(fallback)
        if base is None:
            return options, None
        folder = ja.run_folder(base, label)
        return ja.results_options(options, folder), folder

    def results_folder(self) -> Path | None:
        """The last run's folder, or the newest one in the experiment folder."""
        if self._run_dir is not None and Path(self._run_dir).is_dir():
            return Path(self._run_dir)
        base = self.results_base()
        return ja.latest_run(base) if base is not None else None

    def open_results_folder(self) -> None:
        from qtpy.QtCore import QUrl
        from qtpy.QtGui import QDesktopServices

        folder = self.results_folder()
        if folder is None:
            self._status.setText("No results yet — run first.")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

    # The QC button of older layouts.
    open_qc_folder = open_results_folder

    def _update_sensitivity(self, method: str) -> None:
        self._k.setEnabled(method in (ja.METHOD_CELLS, ja.METHOD_BACKGROUND))
        self._ring_ratio.setEnabled(method == ja.METHOD_RING)
        self._statistic.setEnabled(method != ja.METHOD_RING)
        self._fixed.setEnabled(method == ja.METHOD_FIXED)
        self._threshold_from.setEnabled(method in ja.CELL_RULES)

    # -- state ----------------------------------------------------------------

    def _on_layers_changed(self, event=None) -> None:
        self.refresh_layers()
        pending = self._pending_show
        if pending is not None and self._worker is None:
            image = self._image_of(pending.path)
            if image is not None:
                self._pending_show = None
                self.show_result(pending, image)

    def refresh_layers(self) -> None:
        from napari.layers import Image

        previous = self._image_combo.currentText()
        names = [str(layer.name) for layer in self._viewer.layers
                 if isinstance(layer, Image) and source_of(layer)]
        self._image_combo.blockSignals(True)
        self._image_combo.clear()
        self._image_combo.addItems(names)
        index = self._image_combo.findText(previous)
        self._image_combo.setCurrentIndex(index if index >= 0 else 0)
        self._image_combo.blockSignals(False)

    def image_layer(self):
        name = self._image_combo.currentText()
        return self._viewer.layers[name] if name and name in self._viewer.layers else None

    def _log(self, text: str) -> None:
        self._log_view.appendPlainText(text)
        if self._tracker is not None and self._worker is not None:
            self._tracker.update(text)
        self._status.setText(text)

    def options(self, channels: list[str] | None = None) -> ja.CountOptions:
        """The panel's settings, with the Segmentation and Fusion panels' beside them."""
        settings = None
        segmentation = getattr(self._app, "segmentation_widget", None)
        if segmentation is not None:
            try:
                settings = segmentation.settings()
            except Exception:
                logger.debug("could not read the segmentation settings", exc_info=True)
        junction = None
        fusion = getattr(self._app, "fusion_widget", None)
        if fusion is not None:
            try:
                junction = fusion.junction_options(save=False)
            except Exception:
                logger.debug("could not read the fusion settings", exc_info=True)
        return ja.CountOptions(
            band_key=fu.MANUAL_KEY,
            region_keyword=self._region_edit.text().strip() or REGION_KEYWORD,
            method=self._method.currentText(),
            k=float(self._k.value()),
            ring_ratio=float(self._ring_ratio.value()),
            statistic=self._statistic.currentText(),
            threshold_from=self._threshold_from.currentText(),
            # One number for every channel: named per channel once they are known.
            fixed={name: float(self._fixed.value()) for name in (channels or [])},
            expand_um=float(self._expand.value()),
            find_junction=self._find.isChecked(),
            whole_section=not self._region_only.isChecked(),
            keep_section=not self._positives_only.isChecked(),
            shapes=self._shapes.isChecked(),
            save_cells=self._save_cells.isChecked(),
            settings=settings,
            junction=junction,
        )

    def _channel_names(self, image) -> list[str]:
        """Names of every channel of the sample *image* belongs to."""
        from napari.layers import Image

        source = source_of(image)
        return [
            str(layer.metadata.get("mv_channel_name", "") or layer.name)
            for layer in self._viewer.layers
            if isinstance(layer, Image) and source_of(layer) == source
        ]

    def _animate(self) -> None:
        if self._tracker is not None:
            self._tick += 1
            self._progress.setText(self._tracker.line(self._tick))

    def _busy(self, running: bool) -> None:
        self._count_button.setEnabled(not running)
        self._run_button.setEnabled(not running)
        self._rescore_button.setEnabled(not running)
        self._stop_button.setEnabled(running)

    def stop(self) -> None:
        self._cancelled = True
        self._log("Stopping after the current sample…")

    # -- running --------------------------------------------------------------

    def _saved(self, base: Path | None, options: ja.CountOptions, rescore_only: bool) -> dict:
        """The samples segmented already under *base*, when they are to be reused."""
        if base is None or not options.whole_section or not (rescore_only or self._reuse.isChecked()):
            return {}
        try:
            return ja.segmented_samples(base)
        except Exception:
            logger.exception("could not list the samples segmented under %s", base)
            return {}

    def _plan(self, paths, saved: dict, options: ja.CountOptions, rescore_only: bool):
        """*paths* to run, and a line saying which are segmented and which re-scored; ``None`` when none is left."""
        reused = [p for p in paths if jn.fu_name(Path(p)) in saved
                  and (rescore_only or saved[jn.fu_name(Path(p))].matches(options))]
        if rescore_only:
            missing = [jn.fu_name(Path(p)) for p in paths if p not in reused]
            if not reused:
                self._status.setText("None of these samples has been segmented yet — Analyse them first.")
                return None, ""
            line = f"Re-scoring {len(reused)} sample(s) from their saved cells, no segmentation."
            if missing:
                line += f" Not segmented yet, left out: {', '.join(missing[:5])}" + ("…" if len(missing) > 5 else "")
            return reused, line
        if not reused:
            return list(paths), ""
        return list(paths), (f"{len(paths) - len(reused)} sample(s) to segment; {len(reused)} segmented "
                             "already, re-scored from their saved cells.")

    def count_current(self, rescore_only: bool = False) -> None:
        """Count in the open sample, and draw the cells over it."""
        if self._worker is not None:
            self._status.setText("A count is already running.")
            return
        image = self.image_layer()
        if image is None:
            self._status.setText("Open a sample first.")
            return
        path = Path(source_of(image))
        band = None
        if fu_layer := self._region_layer(image):
            band = np.asarray(fu_layer.data) > 0
        options = self.options(self._channel_names(image))
        # Every nucleus of the section and its grown region, to draw.
        options.full_labels = options.whole_section
        saved = self._saved(self.results_base(path.parent), options, rescore_only)
        paths, line = self._plan([path], saved, options, rescore_only)
        if paths is None:
            return
        options, folder = self.new_run(options, path.parent, label=jn.fu_name(path))
        self._start(paths, options, band=band, draw_on=str(image.name),
                    workbook=folder / ja.WORKBOOK_NAME if folder else None,
                    segmented=saved, reuse_any=rescore_only, plan=line)

    def _region_layer(self, image):
        """The Fusion region layer on screen, when it fits this image."""
        from .fusion_widget import REGION_LAYER_NAME

        if REGION_LAYER_NAME not in self._viewer.layers:
            return None
        layer = self._viewer.layers[REGION_LAYER_NAME]
        data = image.data[0] if getattr(image, "multiscale", False) else image.data
        if tuple(layer.data.shape[-2:]) != tuple(data.shape[-2:]) or not np.any(layer.data):
            return None
        return layer

    # -- samples from a review -----------------------------------------------

    def _update_source(self, source: str) -> None:
        review = source == FROM_REVIEW
        for widget in (self._review_row, self._decisions_row, self._review_summary):
            widget.setVisible(review)
        if review and not self._review_edit.text().strip():
            default = self._default_review()
            if default is not None:
                self._review_edit.setText(str(default))
        self._update_review_summary()

    def _default_review(self) -> Path | None:
        """The experiment's newest review file, if it has one."""
        experiment = getattr(self._app, "experiment_widget", None)
        folder = experiment.folder() if experiment is not None else None
        if not folder:
            return None
        found = [jr.source_folder(folder, source) / jr.REVIEW_FILE for _label, source in jr.SOURCES]
        found = [path for path in found if path.is_file()]
        return max(found, key=lambda path: path.stat().st_mtime) if found else None

    def _browse_review(self) -> None:
        start = self._review_edit.text().strip() or str(self._default_review() or Path.home())
        path, _selected = QFileDialog.getOpenFileName(self, "Choose a junction review", start, "Review (*.csv)")
        if path:
            self._review_edit.setText(path)
            self._update_review_summary()

    def review_decisions(self) -> tuple[str, ...]:
        return tuple(decision for decision, check in self._decisions.items() if check.isChecked())

    def reviewed_samples(self, review_file: str | Path | None = None, include=None):
        """The samples the review file sends on, or ``None`` with the reason in the status line."""
        target = review_file or self._review_edit.text().strip()
        if not target:
            self._review_summary.setText("Choose the review file.")
            return None
        try:
            return jr.samples_from_review(target, include=include if include is not None else self.review_decisions())
        except Exception as exc:
            self._review_summary.setText(f"Could not read it: {exc}")
            return None

    def _update_review_summary(self) -> None:
        if self._source.currentText() != FROM_REVIEW:
            return
        picked = self.reviewed_samples()
        if picked is not None:
            self._review_summary.setText(_review_line(picked))

    def run_review(self, review_file: str | Path, include=None, rescore_only: bool = False) -> bool:
        """Count in the samples *review_file* sends on. Returns whether it started."""
        if self._worker is not None:
            self._status.setText("A count is already running.")
            return False
        picked = self.reviewed_samples(review_file, include)
        if picked is None or not picked.paths:
            self._status.setText("The review sends no sample on. " + (_review_line(picked) if picked else ""))
            return False
        paths = list(picked.paths.values())
        experiment = getattr(self._app, "experiment_widget", None)
        folder = (experiment.folder() if experiment is not None else None) or paths[0].parent
        options = self.options()
        options.review = picked.rows
        saved = self._saved(Path(folder), options, rescore_only)
        paths, line = self._plan(paths, saved, options, rescore_only)
        if paths is None:
            return False
        options, run = self.new_run(options, Path(folder), label="re-scored" if rescore_only else "reviewed")
        self._start(paths, options, workbook=run / ja.WORKBOOK_NAME, segmented=saved,
                    reuse_any=rescore_only, plan=line)
        self._log(_review_line(picked))
        return True

    def run_batch(self, rescore_only: bool = False) -> None:
        """Count in the samples chosen under “Samples”: the Experiment selection, or a review's.

        With *rescore_only*, only the samples segmented already, from their saved cells.
        """
        if self._source.currentText() == FROM_REVIEW:
            self.run_review(self._review_edit.text().strip(), rescore_only=rescore_only)
            return
        experiment = getattr(self._app, "experiment_widget", None)
        if experiment is None:
            self._status.setText("The Experiment setup panel is not available.")
            return
        if self._worker is not None:
            self._status.setText("A count is already running.")
            return
        entries = [entry for entry in experiment.selected_entries() if entry.readable]
        if not entries:
            self._status.setText("No samples. Choose a folder in Experiment setup and scan it first.")
            return
        names = list(dict.fromkeys(name for entry in entries for name in entry.channel_names))
        folder = experiment.folder() or entries[0].path.parent
        options = self.options(names)
        saved = self._saved(Path(folder), options, rescore_only)
        paths, line = self._plan([entry.path for entry in entries], saved, options, rescore_only)
        if paths is None:
            return
        options, run = self.new_run(options, Path(folder), label="re-scored" if rescore_only else "")
        self._start(paths, options, workbook=run / ja.WORKBOOK_NAME, segmented=saved,
                    reuse_any=rescore_only, plan=line)

    def _start(self, paths, options, band=None, draw_on: str = "", workbook: Path | None = None,
               segmented: dict | None = None, reuse_any: bool = False, plan: str = "") -> None:
        from napari.qt.threading import thread_worker

        self._options = options
        self._run_dir = workbook.parent if workbook is not None else None
        self._cancelled = False
        self._log_view.clear()
        self._busy(True)
        self._tracker = ja.RunProgress(len(paths))
        self._tick = 0
        self._progress.setVisible(True)
        self._animate()
        self._ticker.start()
        self._log(f"{'Re-scoring' if reuse_any else 'Analysing'} {len(paths)} sample(s)"
                  + (", the whole section" if options.whole_section else ", the region only")
                  + f": {options.method.lower()}, comparing each cell's {options.statistic}.")
        if plan:
            self._log(plan)
        relay = self._relay
        segmented = segmented or {}

        @thread_worker
        def _run():
            if band is not None:
                saved = segmented.get(jn.fu_name(Path(paths[0])))
                if saved is not None and (reuse_any or saved.matches(options)):
                    result = ja.rescore(saved, options, band=band, progress=relay.message.emit)
                else:
                    result = ja.analyse_sample(paths[0], options, band=band, progress=relay.message.emit)
                relay.message.emit(result.error or ja.summary_line(result))
                return [result]
            return ja.run(paths, options, progress=relay.message.emit,
                          should_cancel=lambda: self._cancelled, segmented=segmented, reuse_any=reuse_any)

        returned: list[int] = []

        def _done(results) -> None:
            self._results = list(results)
            returned.append(len(self._results))
            self._fill_table()
            failed = [r for r in self._results if not r.ok]
            fault = any(r.gpu_fault for r in self._results)
            total = sum(r.n_cells for r in self._results)
            message = f"{total} cell(s) in {len(self._results) - len(failed)} region(s)."
            section = sum(r.n_cells_section for r in self._results)
            if section:
                message += f" {section} in the whole sections."
            if failed:
                message += f" {len(failed)} failed: " + "; ".join(
                    f"{r.name} ({r.error})" for r in failed[:2])
            if draw_on and self._results and self._results[0].ok and draw_on in self._viewer.layers:
                self.show_result(self._results[0], self._viewer.layers[draw_on])
            if workbook is not None and self._results:
                try:
                    workbook.parent.mkdir(parents=True, exist_ok=True)
                    ja.export(self._results, workbook, self._options)
                    ja.write_readme(workbook.parent, self._results, self._options)
                    message += f" Results: {workbook.parent}"
                except Exception as exc:
                    logger.exception("could not write the junction counts")
                    message += f" Could not write the workbook: {exc}"
            if fault:
                missing = len(paths) - len(self._results)
                message += (f" STOPPED on a GPU fault, {missing} sample(s) not attempted: restart the "
                            "viewer and run again. Nothing else should be using the GPU meanwhile.")
            if options.qc_dir is not None and self._results:
                tally = ", ".join(f"{sum(r.qc == status for r in self._results)} {status}"
                                  for status in jn.STATUSES)
                message += f" (QC: {tally})"
            self._log(message)

        def _finished() -> None:
            self._worker = None
            self._busy(False)
            self._ticker.stop()
            tracker, self._tracker = self._tracker, None
            if tracker is not None:
                done = returned[0] if returned else 0
                spent = ja.duration(tracker.elapsed())
                if self._cancelled or done < len(paths):
                    self._progress.setText(f"■ {ja.progress_bar(done / len(paths))} stopped · "
                                           f"{done}/{len(paths)} sample(s) in {spent}")
                else:
                    self._progress.setText(f"✓ {ja.progress_bar(1.0)} 100% · {len(paths)} sample(s) in {spent}")

        worker = _run()
        worker.returned.connect(_done)
        worker.errored.connect(lambda exc: self._log(f"Failed: {exc}"))
        worker.finished.connect(_finished)
        self._worker = worker
        worker.start()

    # -- results --------------------------------------------------------------

    def _fill_table(self) -> None:
        frame = ja.counts_dataframe(self._results)
        table = self._table
        table.setSortingEnabled(False)
        table.setRowCount(len(frame))
        for row in range(len(frame)):
            for column, key in enumerate(TABLE_COLUMNS):
                value = frame.iloc[row][key]
                if isinstance(value, str):
                    text = value
                elif value is None or (isinstance(value, float) and not np.isfinite(value)):
                    text = "—"
                else:
                    text = format_number(value)
                item = QTableWidgetItem(str(text))
                item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                # Which sample the row is, whatever order the table is sorted in.
                item.setData(Qt.UserRole, str(frame.iloc[row]["File"]))
                if column == 0:
                    font = item.font()
                    font.setUnderline(True)
                    item.setFont(font)
                    item.setForeground(QColor(LINK_COLOR))
                table.setItem(row, column, item)
        table.setSortingEnabled(True)
        table.resizeColumnsToContents()

    # -- opening a processed sample --------------------------------------------

    def result_of_row(self, row: int) -> ja.JunctionCounts | None:
        item = self._table.item(row, 0)
        if item is None:
            return None
        wanted = item.data(Qt.UserRole)
        return next((result for result in self._results if str(result.path) == wanted), None)

    def open_row(self, row: int) -> None:
        result = self.result_of_row(row)
        if result is not None:
            self.open_result(result)

    def _image_of(self, path) -> object | None:
        """An image layer on screen read from *path*: the nuclear stain's when there is one."""
        from napari.layers import Image

        wanted = str(Path(path)).lower()
        layers = [layer for layer in self._viewer.layers
                  if isinstance(layer, Image) and str(Path(source_of(layer) or ".")).lower() == wanted]
        nuclear = [layer for layer in layers if any(h in layer.name.lower() for h in jn.NUCLEAR_HINTS)]
        return (nuclear or layers or [None])[0]

    def open_result(self, result: ja.JunctionCounts) -> None:
        """Open *result*'s sample, unless it is on screen already, and draw its results over it."""
        image = self._image_of(result.path)
        if image is None:
            self._log(f"Opening {result.name}…")
            self._pending_show = result
            experiment = getattr(self._app, "experiment_widget", None)
            try:
                if experiment is not None and hasattr(experiment, "open_path"):
                    experiment.open_path(result.path)
                else:
                    self._app.open_paths([result.path])
            except Exception as exc:
                logger.exception("could not open %s", result.path)
                self._log(f"Could not open {result.name}: {exc}")
                self._pending_show = None
                return
            image = self._image_of(result.path)
            if image is None:
                return  # it is still loading: drawn when its layers arrive (_on_layers_changed)
        self._pending_show = None
        self.show_result(result, image)

    def show_result(self, result: ja.JunctionCounts, image) -> None:
        """Draw *result* over *image*: its label maps when they are kept, else rings from its table."""
        from dataclasses import replace

        if result.section_labels is None and ja.labels_files(result) is not None:
            # Read back for this drawing only: the result in the table stays small.
            nuclei_file, grown_file = ja.labels_files(result)
            self._log(f"{result.name}: reading its label maps…")
            table = result.section_cells
            if table is None and result.cells_file is not None and Path(result.cells_file).is_file():
                import pandas as pd

                table = pd.read_csv(result.cells_file)
            shapes = {channel: ja.read_labels(path) for channel, path in result.shape_files.items()
                      if Path(path).is_file()}
            result = replace(result, section_labels=ja.read_labels(nuclei_file),
                             section_regions=ja.read_labels(grown_file), section_cells=table,
                             section_shapes=shapes or result.section_shapes)
        if result.section_labels is not None or result.labels is not None:
            self._draw(result, str(image.name))
            self._log(f"{result.name}: its cells drawn.")
            return
        table, source = result.section_cells, "the section"
        if table is None and result.cells_file is not None and Path(result.cells_file).is_file():
            import pandas as pd

            table, source = pd.read_csv(result.cells_file), Path(result.cells_file).name
        if table is None:
            table, source = result.cells, "the region"
        if table is None or not len(table):
            self._log(f"{result.name}: no cells to draw ({result.error or 'none kept'}).")
            return
        self._draw_rings(result, image, table)
        self._log(f"{result.name}: positive cells from {source}, ringed in red.")

    def _draw_rings(self, result: ja.JunctionCounts, image, table) -> None:
        """A points layer per channel of its positive cells, and one of the region's cells (hidden)."""
        from ..loaders.layer_spec import units_like

        self.remove_layers()
        voxel = result.voxel_yx
        offset = tuple(float(v) for v in image.translate)[-2:]
        common = {"scale": voxel, "translate": offset, **units_like(image, 2)}
        coords = np.column_stack([table["Centroid Y (µm)"].to_numpy() / voxel[0],
                                  table["Centroid X (µm)"].to_numpy() / voxel[1]])
        size = RING_UM / float(voxel[0])

        def put(name: str, chosen: np.ndarray, **kwargs):
            if name in self._viewer.layers:
                self._viewer.layers.remove(name)
            layer = self._viewer.add_points(coords[chosen], name=name, size=size, face_color="transparent",
                                            **{BORDER: POSITIVE_COLOR}, **common, **kwargs)
            tie_to_sample(layer, str(result.path))
            layer.metadata[DRAWN_KEY] = True
            return layer

        for count in result.channels:
            column = f"{count.channel} positive"
            if count.reference or column not in table:
                continue
            put(f"{count.channel}{POSITIVE_SUFFIX}", table[column].to_numpy(dtype=bool))
        if ja.IN_REGION_COLUMN in table:
            region = put(CELLS_LAYER, table[ja.IN_REGION_COLUMN].to_numpy(dtype=bool), visible=False)
            self._set_border(region, "white")

    @staticmethod
    def _set_border(layer, color: str) -> None:
        try:
            setattr(layer, BORDER_ATTR, color)
        except Exception:
            logger.debug("could not colour the rings", exc_info=True)

    def _row_menu(self, position) -> None:
        from qtpy.QtCore import QUrl
        from qtpy.QtGui import QDesktopServices
        from qtpy.QtWidgets import QMenu

        row = self._table.rowAt(position.y())
        result = self.result_of_row(row) if row >= 0 else None
        if result is None:
            return
        menu = QMenu(self)
        menu.addAction(f"Open {result.name} with its results", lambda: self.open_result(result))
        for label, target in (("Show the region picture", result.picture),
                              ("Show the whole-section picture", result.section_picture),
                              ("Open its cells (CSV)", result.cells_file)):
            action = menu.addAction(label, lambda target=target: QDesktopServices.openUrl(
                QUrl.fromLocalFile(str(target))))
            action.setEnabled(target is not None and Path(target).is_file())
        menu.addAction("Show the results folder", self.open_results_folder)
        menu.exec_(self._table.viewport().mapToGlobal(position))

    def _draw(self, result: ja.JunctionCounts, image_name: str) -> None:
        """All the cells as outlines, and a filled layer per channel of its positives."""
        from ..loaders.layer_spec import units_like

        if (result.labels is None and result.section_labels is None) or image_name not in self._viewer.layers:
            return
        image = self._viewer.layers[image_name]
        voxel = result.voxel_yx
        offset = tuple(float(t) for t in image.translate)[-2:]
        translate = (offset[0] + result.origin[0] * voxel[0], offset[1] + result.origin[1] * voxel[1])
        common = {"scale": voxel, "translate": translate, **units_like(image, 2)}

        def put(name: str, data: np.ndarray, **kwargs):
            if name in self._viewer.layers:
                self._viewer.layers.remove(name)
            layer = self._viewer.add_labels(data, name=name, **kwargs)
            tie_to_sample(layer, str(result.path))
            layer.metadata[DRAWN_KEY] = True
            return layer

        # The cells of the last count only: a channel scored then and not now
        # would otherwise leave its layer behind.
        self.remove_layers()

        try:
            scored = [count for count in result.channels if not count.reference]
            if result.section_labels is not None:
                # The whole section: every nucleus, the same grown, and each
                # channel's positive cells wherever they are.
                whole = {**common, "translate": offset}
                nuclei = put(NUCLEI_LAYER, result.section_labels, opacity=0.8, **whole)
                nuclei.contour = 1
                grown = put(GROWN_LAYER.format(um=format_number(self._options.expand_um)),
                            result.section_regions, opacity=0.5, visible=False, **whole)
                grown.contour = 1
                table = result.section_cells
                for count in scored:
                    column = f"{count.channel} positive"
                    if table is None or column not in table:
                        continue
                    wanted = table.loc[table[column], "Label"].to_numpy(dtype=np.int64)
                    # The nucleus's outline: the cell as segmented, not the area it was measured over.
                    mask = np.isin(result.section_labels, wanted).astype(np.uint8)
                    layer = put(f"{count.channel}{POSITIVE_SUFFIX}", mask, opacity=1.0, **whole)
                    layer.contour = POSITIVE_CONTOUR
                    self._tint(layer, POSITIVE_COLOR)
                for count in scored:
                    shapes = result.section_shapes.get(count.channel)
                    if shapes is not None:
                        # The cell as its stain draws it, one colour per cell, outlined.
                        shape = put(f"{count.channel}{SHAPE_SUFFIX}", shapes, opacity=0.9, **whole)
                        shape.contour = 1
                if result.labels is not None:
                    region = put(CELLS_LAYER, result.labels, opacity=0.9, visible=False, **common)
                    region.contour = 1
                return
            cells = put(CELLS_LAYER, result.labels, opacity=0.9, **common)
            cells.contour = 1  # outlines: the signal underneath stays visible
            for count in scored:
                wanted = result.positive_labels(count.channel)
                mask = np.where(np.isin(result.labels, wanted), 1, 0).astype(np.uint8)
                layer = put(f"{count.channel}{POSITIVE_SUFFIX}", mask, opacity=1.0, **common)
                # Red outlines, the same for every channel: the layer's name
                # says which, and a red ring reads as "this one" over any stain.
                layer.contour = POSITIVE_CONTOUR
                self._tint(layer, POSITIVE_COLOR)
        except Exception:
            logger.exception("could not draw the junction cells")

    def remove_layers(self) -> int:
        """Take the counted cells off screen: all of them, every channel's."""
        removed = 0
        for layer in list(self._viewer.layers):
            if layer.metadata.get(DRAWN_KEY):
                self._viewer.layers.remove(layer)
                removed += 1
        return removed

    @staticmethod
    def _tint(layer, color: str) -> None:
        """One colour for the whole layer, when this napari can do that."""
        try:
            from napari.utils.colormaps import DirectLabelColormap

            layer.colormap = DirectLabelColormap(color_dict={None: color, 0: "transparent"})
        except Exception:
            logger.debug("no single-colour labels in this napari; keeping the default", exc_info=True)

    def export(self, path: str | None = None):
        """Write the counts, the combinations, every cell and the settings out."""
        if not self._results:
            self._status.setText("Nothing to export — count first.")
            return None
        if not path:
            experiment = getattr(self._app, "experiment_widget", None)
            folder = (experiment.folder() if experiment is not None else None) or Path.home()
            suggested = str(Path(folder) / f"{default_stem('junction_counts')}.xlsx")
            path, _selected = QFileDialog.getSaveFileName(self, "Export the junction counts", suggested, WORKBOOK_FILTER)
            if not path:
                return None
        try:
            written = ja.export(self._results, path, self._options)
        except Exception as exc:
            logger.exception("junction export failed")
            QMessageBox.critical(self, "Microscopy Viewer", f"Could not write that file:\n{exc}")
            return None
        self._log(f"Workbook written to {written}.")
        return written


def _review_line(picked) -> str:
    """One line: what a review sends on, and what it leaves behind."""
    said = {jr.ACCEPT: "accepted", jr.REDRAWN: "redrawn", jr.EXCLUDE: "excluded",
            jr.REDRAW: "still to redraw", jr.UNDECIDED: "not reviewed"}
    left = ", ".join(f"{n} {said.get(d, d)}" for d, n in sorted(picked.left_out.items()))
    line = f"{len(picked.paths)} sample(s) to count"
    if left:
        line += f"; left out: {left}"
    if picked.missing:
        line += f"; {len(picked.missing)} not found: " + ", ".join(picked.missing[:3]) + ("…" if len(picked.missing) > 3 else "")
    return line + "."
