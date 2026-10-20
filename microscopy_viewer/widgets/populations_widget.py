"""Dock panel: the population analysis of a whole plate, in three steps.

The Measurement analysis panel answers questions about one image. This one
answers the question a plate was run to ask — *what fraction of the cells in
these wells is positive, and does it differ between them* — off the combined
``.h5ad`` a batch run or that panel already wrote.

Three sub-tabs, in the order the decisions have to be made in:

**Prepare** — which samples are in, and which objects are cells. A gate is one
low-and-high bound on one feature; the histogram underneath is the feature's
distribution over the objects currently selected, with the bounds drawn on it, so
a threshold is set by looking rather than by guessing. The table says what each
gate cost, because a filter whose cost is invisible is a filter that quietly
removes the population of interest.

**Positive** — what positive means: ``median + 3 sd`` of a chosen measurement, the
multiplier and the statistic adjustable, pooled over one of the application's own
normalisation scopes. The table shows the threshold each sample was actually
compared with.

**Explore** — the answer per sample and per well, plotted and exportable, and the
button that writes the whole definition back into the file.

**None of the arithmetic is in here.** It is all in
:mod:`microscopy_viewer.populations`, which has no Qt in it and which the
Streamlit page imports too — a threshold that meant one thing in this panel and
another on that page would be worse than having only one of them. This module
reads widgets, calls that module, and draws what it is handed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from qtpy.QtCore import Qt
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
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QTableView,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .. import analysis, dashboard, populations
from ..docks import emphasise
from ..utils import get_logger

# The model is the one the Measurement analysis panel uses. A second read-only
# model over a DataFrame would be the same forty lines with a different bug in it.
from .analysis_widget import FrameModel, _figure_canvas

logger = get_logger("populations_widget")

#: The two ways of looking at a gate. One feature binned, or two against each
#: other — the second is the one that shows whether a bound separates two clouds
#: or runs through the middle of one.
VIEW_HISTOGRAM = "histogram"
VIEW_SCATTER = "scatter"

#: Dot size in the two-feature view, and the size above which they are made
#: smaller and fainter. Matches the Measurement analysis panel's scatter.
DOT_SIZE = 6
DOT_FADE_ABOVE = 2_000

FILE_FILTER = "AnnData (*.h5ad);;All files (*)"

#: What the Explore tab can draw. Few on purpose: this panel is for getting to an
#: answer and exporting it, and the dashboard page is where a plate gets explored
#: twenty plots at a time.
PLOTS = (
    "percent positive per sample",
    "percent positive per well",
    "objects kept and excluded",
    "signal, positives apart",
)

#: The label for "do not normalise", in the one combo that chooses between raw
#: signal and each of :data:`microscopy_viewer.analysis.SCOPES`.
RAW_LABEL = "raw signal"


class PopulationAnalysisWidget(QWidget):
    """Gate a plate, define what positive means, and read off the percentages."""

    def __init__(self, app, parent: QWidget | None = None):
        super().__init__(parent)
        self._app = app
        self._adata = None
        self._path: Path | None = None
        self._gates: list[populations.Gate] = []
        self._calls: populations.PositiveCalls | None = None
        self._summary = None
        self._keep: np.ndarray | None = None
        # Set while a refresh is filling the boxes, so that the signals that fire
        # as each one is populated do not each trigger a full recompute.
        self._updating = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(self._build_file_box())

        self._tabs = QTabWidget()
        self._tabs.addTab(self._build_prepare_tab(), "1. Prepare")
        self._tabs.addTab(self._build_positive_tab(), "2. Positive")
        self._tabs.addTab(self._build_explore_tab(), "3. Explore")
        layout.addWidget(self._tabs, stretch=1)

        self._status = QLabel("Open the combined .h5ad for a plate.")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        survey = getattr(app, "plate_survey", None)
        if survey is not None:
            self.plate_changed(survey)

    # -- construction ---------------------------------------------------------

    def _build_file_box(self) -> QGroupBox:
        box = QGroupBox("Plate")
        form = QFormLayout(box)
        form.setLabelAlignment(Qt.AlignRight)

        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        self._path_edit = QLineEdit()
        self._path_edit.setPlaceholderText("…/plate_nuclei_objects.h5ad")
        self._path_edit.setToolTip(
            "The combined .h5ad for a plate: one row per object, with the well and "
            "cycle in obs[\"image\"].\n\n"
            "Written by a batch run asked for a single file, or by the Measurement "
            "analysis panel's Combine button."
        )
        self._path_edit.returnPressed.connect(self.load)
        row_layout.addWidget(self._path_edit, stretch=1)
        browse = QPushButton("…")
        browse.setMaximumWidth(30)
        browse.clicked.connect(self.browse)
        row_layout.addWidget(browse)
        load = QPushButton("Load")
        load.clicked.connect(self.load)
        row_layout.addWidget(load)
        form.addRow("File", row)

        self._beside_box = QComboBox()
        self._beside_box.setToolTip(
            "Combined files sitting beside the plate scanned in the File explorer, "
            "newest first. Choosing one opens it."
        )
        self._beside_box.currentIndexChanged.connect(self._beside_chosen)
        form.addRow("Beside the plate", self._beside_box)

        self._file_summary = QLabel("Nothing loaded.")
        self._file_summary.setWordWrap(True)
        form.addRow("Contents", self._file_summary)
        return box

    def _build_prepare_tab(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(2, 2, 2, 2)

        samples = QGroupBox("Samples in the analysis")
        samples_layout = QVBoxLayout(samples)
        self._sample_list = QListWidget()
        self._sample_list.setToolTip(
            "One entry per obs[\"image\"] — a well and a cycle. Untick the wells that "
            "failed and the ones that belong to another experiment: a percentage "
            "pooled over those is a percentage of nothing."
        )
        self._sample_list.setMaximumHeight(140)
        self._sample_list.itemChanged.connect(self._selection_changed)
        samples_layout.addWidget(self._sample_list)
        buttons = QHBoxLayout()
        for label, state in (("All", Qt.Checked), ("None", Qt.Unchecked)):
            button = QPushButton(label)
            button.clicked.connect(lambda _checked=False, value=state: self._set_all_samples(value))
            buttons.addWidget(button)
        buttons.addStretch(1)
        self._sample_count = QLabel("—")
        buttons.addWidget(self._sample_count)
        samples_layout.addLayout(buttons)
        outer.addWidget(samples)

        gates = QGroupBox("Exclude cells and artifacts")
        gates_layout = QVBoxLayout(gates)
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)
        self._gate_feature = QComboBox()
        self._gate_feature.setToolTip(
            "The feature to gate on. Size cuts debris and fused pairs; solidity cuts "
            "the objects that are not one cell."
        )
        self._gate_feature.currentTextChanged.connect(self._gate_feature_changed)
        form.addRow("Feature", self._gate_feature)

        bounds = QWidget()
        bounds_layout = QHBoxLayout(bounds)
        bounds_layout.setContentsMargins(0, 0, 0, 0)
        self._gate_low = self._number_box()
        self._gate_high = self._number_box()
        for label, spin in (("keep from", self._gate_low), ("to", self._gate_high)):
            bounds_layout.addWidget(QLabel(label))
            spin.valueChanged.connect(self._bounds_changed)
            bounds_layout.addWidget(spin, stretch=1)
        suggest = QPushButton("Suggest")
        suggest.setToolTip(
            "Put the bounds at the 0.5 and 99.5 percentiles of what is selected — the "
            "tails, which is where the debris and the fused objects are."
        )
        suggest.clicked.connect(self.suggest_bounds)
        bounds_layout.addWidget(suggest)
        form.addRow("Bounds", bounds)

        self._gate_join = QComboBox()
        for join in populations.JOINS:
            self._gate_join.addItem(populations.JOIN_LABELS[join], join)
        self._gate_join.setToolTip(
            "How this gate combines with the result of the gates above it.\n\n"
            "“and” narrows: an object has to pass this one as well. “or” widens: it "
            "may pass this one instead, which is how you keep two kinds of object "
            "with one rule, or put two bounds on one feature. “xor” keeps the objects "
            "that pass an odd number of the gates — over two gates, one or the other "
            "but not both, which is how you ask what two markers disagree about.\n\n"
            "The gates are applied top to bottom with no brackets and no precedence: "
            "“A or B and C” here means “(A or B) and C”. The rule as it will be "
            "applied is written out under the table.\n\n"
            "Ignored on the first gate, which has nothing to combine with."
        )
        form.addRow("Join", self._gate_join)

        view = QWidget()
        view_layout = QHBoxLayout(view)
        view_layout.setContentsMargins(0, 0, 0, 0)
        self._gate_view = QComboBox()
        self._gate_view.addItem("distribution", VIEW_HISTOGRAM)
        self._gate_view.addItem("against", VIEW_SCATTER)
        self._gate_view.setToolTip(
            "“distribution” is the feature on its own, binned, with the bounds across "
            "it — the view for choosing one bound.\n\n"
            "“against” plots it against a second measurement, with the bounds of both "
            "drawn on and every object coloured by whether the whole rule keeps it. "
            "That is the view for seeing whether a gate is cutting through a cloud or "
            "between two, which a histogram of either feature alone cannot show."
        )
        self._gate_view.currentIndexChanged.connect(self._gate_view_changed)
        view_layout.addWidget(self._gate_view)
        # Set here as well as on change: the combo starts on the histogram, and
        # setting the index it is already on fires nothing, so the second axis
        # would be left enabled in the view that has no second axis.
        self._gate_view.setCurrentIndex(0)
        self._gate_y = QComboBox()
        self._gate_y.setToolTip("The second measurement, on the vertical axis.")
        self._gate_y.currentTextChanged.connect(lambda _text: self.draw_gate_plot())
        view_layout.addWidget(self._gate_y, stretch=1)
        form.addRow("View", view)
        gates_layout.addLayout(form)

        from matplotlib.figure import Figure

        canvas_class = _figure_canvas()
        self._gate_figure = Figure(figsize=(4.0, 2.2), layout="constrained")
        self._gate_canvas = canvas_class(self._gate_figure)
        self._gate_canvas.setMinimumHeight(170)
        self._gate_canvas.setToolTip(
            "The selected objects, with the gates' bounds drawn on them — one feature "
            "binned, or two against each other, as chosen in “View”.\n\n"
            "In the distribution the end bins hold everything beyond them. In the "
            "two-feature view a grey point is an object the rule excludes, the lines "
            "are the bounds of the gates on these two axes, and the shaded corner is "
            "where both of them keep an object."
        )
        gates_layout.addWidget(self._gate_canvas, stretch=1)

        self._gate_plot_note = QLabel("—")
        self._gate_plot_note.setWordWrap(True)
        self._gate_plot_note.setToolTip(
            "What is in the picture, and what is not: how many points were drawn, "
            "whether they are a sample, and any gate whose feature is on neither axis."
        )
        gates_layout.addWidget(self._gate_plot_note)

        actions = QHBoxLayout()
        add = QPushButton("Add gate")
        add.setToolTip(
            "Add these bounds as another gate at the bottom of the list.\n\n"
            "Two gates on the same feature are allowed and are the point of “or”: "
            "“Voxels in [0, 100] or Voxels in [900, inf]” keeps the debris and the "
            "fused objects and nothing in between."
        )
        add.clicked.connect(self.add_gate)
        actions.addWidget(add)
        update = QPushButton("Update selected")
        update.setToolTip(
            "Replace the selected gate with what is in the boxes, keeping its place "
            "in the list — which matters, because the gates are applied in order."
        )
        update.clicked.connect(self.update_gate)
        actions.addWidget(update)
        remove = QPushButton("Remove")
        remove.setToolTip("Drop the selected gate.")
        remove.clicked.connect(self.remove_gate)
        actions.addWidget(remove)
        actions.addStretch(1)
        gates_layout.addLayout(actions)

        self._gate_table = QTableWidget(0, 6)
        self._gate_table.setHorizontalHeaderLabels(
            ["Feature", "join", "from", "to", "removed", "blank"]
        )
        self._gate_table.setToolTip(
            "The gates, in the order they are applied — top to bottom, each joined to "
            "the result so far by its own “join”. The first row's join is blank "
            "because it has nothing to combine with.\n\n"
            "“removed” is what that gate takes out applied **on its own** to the whole "
            "selection, not in sequence — in sequence the second gate looks cheap "
            "because the first already took what they share. With “or” or “xor” in the "
            "rule these do not add up to the total kept, and the line under the table "
            "says so.\n\n"
            "“blank” is objects with no value for that feature: a 4i cycle that did "
            "not measure it. A gate cannot judge those, so they count as a pass for "
            "“and” and as a fail for “or” and “xor” — either way they leave the "
            "running answer alone.\n\n"
            "Untick a row to switch that gate off; click one to load it into the boxes."
        )
        self._gate_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._gate_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._gate_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._gate_table.setMaximumHeight(130)
        self._gate_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._gate_table.itemChanged.connect(self._gate_toggled)
        self._gate_table.itemSelectionChanged.connect(self._gate_row_chosen)
        gates_layout.addWidget(self._gate_table)

        self._gate_status = QLabel("—")
        self._gate_status.setWordWrap(True)
        gates_layout.addWidget(self._gate_status)
        outer.addWidget(gates, stretch=1)
        return page

    def _build_positive_tab(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(2, 2, 2, 2)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)

        self._rule_feature = QComboBox()
        self._rule_feature.setToolTip("The measurement an object is positive for.")
        form.addRow("Measurement", self._rule_feature)

        self._statistic_box = QComboBox()
        for name in populations.STATISTICS:
            self._statistic_box.addItem(populations.STATISTIC_LABELS[name], name)
        self._statistic_box.setToolTip(
            "median + k x sd is the rule to use. The median rather than the mean "
            "because the positives are themselves inside the distribution the "
            "threshold comes from, and they drag a mean upwards.\n\n"
            "A percentile assumes the positive fraction instead of measuring it."
        )
        self._statistic_box.currentIndexChanged.connect(self._statistic_changed)
        form.addRow("Threshold", self._statistic_box)

        self._k_box = QDoubleSpinBox()
        self._k_box.setRange(0.0, 20.0)
        self._k_box.setSingleStep(0.5)
        self._k_box.setDecimals(2)
        self._k_box.setValue(populations.DEFAULT_K)
        self._k_box.setToolTip("Standard deviations above the centre. Three is the rule.")
        form.addRow("k", self._k_box)

        self._percentile_box = QDoubleSpinBox()
        self._percentile_box.setRange(50.0, 100.0)
        self._percentile_box.setDecimals(2)
        self._percentile_box.setValue(populations.DEFAULT_PERCENTILE)
        form.addRow("Percentile", self._percentile_box)

        self._signal_box = QComboBox()
        self._signal_box.addItem(RAW_LABEL, "")
        for scope in analysis.SCOPES:
            self._signal_box.addItem(
                f"normalised over {analysis.SCOPE_LABELS[scope]}", scope
            )
        self._signal_box.setToolTip(
            "Raw grey levels, or the measurement mapped onto the 1-99 % range of the "
            "chosen scope — the same “normalise over” the viewer's colour scales use.\n\n"
            "Normalising per image before a threshold pooled over something wider is "
            "what makes one threshold fair across images acquired at different "
            "exposures. Normalising over the same scope the threshold is pooled over "
            "calls the same objects; it only changes the scale the threshold is read on."
        )
        form.addRow("Signal", self._signal_box)

        self._scope_box = QComboBox()
        for scope in analysis.SCOPES:
            self._scope_box.addItem(analysis.SCOPE_LABELS[scope], scope)
        self._scope_box.setCurrentIndex(analysis.SCOPES.index(analysis.DEFAULT_SCOPE))
        self._scope_box.setToolTip(
            "What the threshold is pooled over, and the decision that matters most.\n\n"
            "Per image lets every well define its own normality, which hides exactly "
            "the difference between wells a plate was run to find. Per cycle gives one "
            "staining round one threshold across the plate, which is how a plate is "
            "usually read."
        )
        form.addRow("Pooled over", self._scope_box)
        outer.addLayout(form)

        apply_button = QPushButton("Apply the rule")
        apply_button.setToolTip(
            "Compute the thresholds and call every object. The gates are applied first: "
            "debris is the thing with the extreme intensities, and leaving it in the "
            "pool moves the threshold by the amount the gates exist to prevent."
        )
        apply_button.clicked.connect(self.apply_rule)
        outer.addWidget(apply_button)

        self._rule_status = QLabel("Choose a measurement and apply.")
        self._rule_status.setWordWrap(True)
        outer.addWidget(self._rule_status)

        outer.addWidget(QLabel("The threshold each sample was compared with:"))
        self._threshold_model = FrameModel(parent=self)
        self._threshold_table = QTableView()
        self._threshold_table.setModel(self._threshold_model)
        self._threshold_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        outer.addWidget(self._threshold_table, stretch=1)

        self._statistic_changed()
        return page

    def _build_explore_tab(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(2, 2, 2, 2)

        self._headline = QLabel("—")
        emphasise(self._headline)
        self._headline.setWordWrap(True)
        outer.addWidget(self._headline)

        row = QHBoxLayout()
        row.addWidget(QLabel("plot"))
        self._plot_box = QComboBox()
        self._plot_box.addItems(list(PLOTS))
        self._plot_box.currentTextChanged.connect(lambda _text: self.draw_plot())
        row.addWidget(self._plot_box, stretch=1)
        self._per_well = QCheckBox("pool the cycles of a well")
        self._per_well.setToolTip(
            "A 4i well is imaged every cycle and those are the same cells, so one "
            "condition is otherwise drawn seven times."
        )
        self._per_well.toggled.connect(lambda _on: self.draw_plot())
        row.addWidget(self._per_well)
        outer.addLayout(row)

        from matplotlib.figure import Figure

        canvas_class = _figure_canvas()
        self._figure = Figure(figsize=(4.0, 3.0), layout="constrained")
        self._canvas = canvas_class(self._figure)
        self._canvas.setMinimumHeight(220)
        outer.addWidget(self._canvas, stretch=1)

        self._summary_model = FrameModel(parent=self)
        self._summary_table = QTableView()
        self._summary_table.setModel(self._summary_model)
        self._summary_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._summary_table.setSortingEnabled(True)
        outer.addWidget(self._summary_table, stretch=1)

        actions = QHBoxLayout()
        export = QPushButton("Export the summary as CSV…")
        export.setToolTip("The table as it is, so a figure and the numbers beside it agree.")
        export.clicked.connect(self.export_summary)
        actions.addWidget(export)
        save = QPushButton("Write into the .h5ad…")
        save.setToolTip(
            "Put the gates, the rule and the per-object calls back on the file, so a "
            "percentage reopened next month still says what it was a percentage of."
        )
        save.clicked.connect(self.save_into_file)
        actions.addWidget(save)
        outer.addLayout(actions)
        return page

    @staticmethod
    def _number_box() -> QDoubleSpinBox:
        """A spin box wide enough for an integrated intensity and a solidity both."""
        spin = QDoubleSpinBox()
        spin.setRange(-1e12, 1e12)
        spin.setDecimals(4)
        # Otherwise every keystroke of a six-figure number is a recompute over the
        # whole plate, and the box cannot be typed into.
        spin.setKeyboardTracking(False)
        return spin

    # -- the plate ------------------------------------------------------------

    def plate_changed(self, survey) -> None:
        """List the combined files beside the plate the File explorer scanned.

        Nothing is opened: which of several analyses someone means is not a thing
        to guess at, and reading half a million objects they did not ask for would
        drop the gates they had already set.
        """
        try:
            found = populations.population_files(survey.path)
        except Exception:
            logger.exception("could not list the combined files beside %s", survey.path)
            found = []

        self._updating = True
        try:
            self._beside_box.clear()
            self._beside_box.addItem("—", None)
            for path in found:
                self._beside_box.addItem(path.name, path)
        finally:
            self._updating = False
        self._status.setText(
            f"{len(found)} combined file(s) beside {survey.path.name}."
            if found
            else f"No combined .h5ad beside {survey.path.name} — the Measurement "
            "analysis panel's Combine button writes one."
        )

    def _beside_chosen(self, _index: int) -> None:
        if self._updating:
            return
        path = self._beside_box.currentData()
        if path is None:
            return
        self._path_edit.setText(str(path))
        self.load()

    def browse(self) -> None:
        path, _filter = QFileDialog.getOpenFileName(
            self, "Open a combined .h5ad", self._path_edit.text(), FILE_FILTER
        )
        if path:
            self._path_edit.setText(path)
            self.load()

    def load(self) -> None:
        """Read the file, and put back any analysis already stored on it."""
        text = self._path_edit.text().strip()
        if not text:
            return
        try:
            adata = dashboard.read(text)
        except Exception as error:
            logger.exception("could not read %s", text)
            self._status.setText(f"Could not read {Path(text).name}: {error}")
            return

        self._adata = adata
        self._path = Path(text)
        self._calls = None
        self._summary = None
        self._keep = None

        stored = populations.load_populations(adata)
        self._gates = list(stored.get("gates", []))
        self._refresh_boxes(stored)
        self._recompute_gates()
        if stored:
            self._status.setText(
                f"{self._path.name}: a population analysis from {stored['created']} is on "
                f"this file — {stored['n_positive']:,} of {stored['n_kept']:,} kept objects "
                f"positive. Its gates and its rule are back in the boxes."
            )
        else:
            self._status.setText(
                f"{self._path.name}: {adata.n_obs:,} objects, {adata.n_vars} features, "
                f"{len(populations.sample_labels(adata))} samples."
            )

    def _refresh_boxes(self, stored: dict | None = None) -> None:
        """Fill the feature and sample lists from the file, and restore a stored rule."""
        adata = self._adata
        if adata is None:
            return
        names = populations.measurable_names(adata)
        samples = populations.sample_labels(adata)
        rule = (stored or {}).get("rule")
        chosen = set((stored or {}).get("samples", samples)) or set(samples)

        self._updating = True
        try:
            self._file_summary.setText(
                f"{adata.n_obs:,} objects, {adata.n_vars} features, {len(samples)} samples"
            )
            for box in (self._gate_feature, self._rule_feature, self._gate_y):
                box.clear()
                box.addItems(names)
            # A different measurement from the first by default: two axes showing
            # one feature against itself is a diagonal line and no information.
            if len(names) > 1:
                self._gate_y.setCurrentIndex(1)

            self._sample_list.clear()
            for name in samples:
                item = QListWidgetItem(name)
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Checked if name in chosen else Qt.Unchecked)
                self._sample_list.addItem(item)

            if rule is not None and rule.feature in names:
                self._rule_feature.setCurrentText(rule.feature)
                index = self._statistic_box.findData(rule.statistic)
                if index >= 0:
                    self._statistic_box.setCurrentIndex(index)
                self._k_box.setValue(float(rule.k))
                self._percentile_box.setValue(float(rule.percentile))
                self._signal_box.setCurrentIndex(
                    max(0, self._signal_box.findData(rule.normalise_scope if rule.normalised else ""))
                )
                scope = self._scope_box.findData(rule.scope)
                if scope >= 0:
                    self._scope_box.setCurrentIndex(scope)
            self._fill_gate_table()
        finally:
            self._updating = False
        self._statistic_changed()
        self._update_sample_count()
        self._gate_feature_changed()

    # -- prepare --------------------------------------------------------------

    def selected_samples(self) -> list[str]:
        """The ticked samples, in the file's own order."""
        return [
            self._sample_list.item(row).text()
            for row in range(self._sample_list.count())
            if self._sample_list.item(row).checkState() == Qt.Checked
        ]

    def _set_all_samples(self, state) -> None:
        self._updating = True
        try:
            for row in range(self._sample_list.count()):
                self._sample_list.item(row).setCheckState(state)
        finally:
            self._updating = False
        self._selection_changed()

    def _update_sample_count(self) -> None:
        chosen = len(self.selected_samples())
        self._sample_count.setText(f"{chosen} of {self._sample_list.count()} in")

    def _selection_changed(self, _item=None) -> None:
        if self._updating:
            return
        self._update_sample_count()
        self._recompute_gates()

    def _gate_view_changed(self, _index: int = 0) -> None:
        """Enable the second axis only for the view that has one."""
        scatter = self._gate_view.currentData() == VIEW_SCATTER
        self._gate_y.setEnabled(scatter)
        self.draw_gate_plot()

    def draw_gate_plot(self) -> None:
        """Draw whichever view is chosen. Everything calls this, not either one."""
        scatter = self._gate_view.currentData() == VIEW_SCATTER
        # Disabled rather than hidden, so the row does not change height when the
        # view changes.
        self._gate_y.setEnabled(scatter)
        if scatter:
            self.draw_scatter()
        else:
            self.draw_histogram()

    def draw_scatter(self) -> None:
        """Two features, with the gates' bounds on them and the rule in the colour.

        The bounds are drawn from the gates that exist, and the pair in the spin
        boxes is drawn too — dashed, like the histogram's — so a bound being
        dragged can be seen against the cloud before it is added.
        """
        self._gate_figure.clear()
        axes = self._gate_figure.add_subplot(111)
        x_name = self._gate_feature.currentText()
        y_name = self._gate_y.currentText()

        scatter = None
        if self._adata is not None and x_name and y_name:
            try:
                scatter = populations.feature_scatter(
                    self._adata, x_name, y_name, self._gates, within=self._selection_mask()
                )
            except Exception:
                logger.exception("could not build the two-feature view")
                scatter = None

        if scatter is None or len(scatter.x) == 0:
            axes.text(
                0.5,
                0.5,
                "pick two measurements the file has"
                if x_name != y_name
                else "pick a different second measurement",
                ha="center",
                va="center",
                color=dashboard.INK,
                transform=axes.transAxes,
            )
            axes.set_axis_off()
            self._gate_canvas.draw_idle()
            self._gate_plot_note.setText("—")
            return

        kept = np.asarray(scatter.kept, dtype=bool)
        faint = len(scatter.x) > DOT_FADE_ABOVE
        # Grey for the excluded, which is what grey means everywhere here: not a
        # group. The kept objects take the first palette colour, as they do in
        # every other plot in this panel.
        for mask, colour, label, alpha in (
            (~kept, dashboard.NEUTRAL, "excluded", 0.45 if faint else 0.6),
            (kept, dashboard.PALETTE[0], "kept", 0.6 if faint else 0.9),
        ):
            if not mask.any():
                continue
            axes.scatter(
                scatter.x[mask],
                scatter.y[mask],
                s=DOT_SIZE,
                c=colour,
                alpha=alpha,
                linewidths=0,
                label=label,
            )

        # The gates that exist, solid; the pair being edited, dashed.
        for low, high in scatter.x_bounds:
            for bound in (low, high):
                if np.isfinite(bound):
                    axes.axvline(bound, color=dashboard.NEUTRAL, linewidth=1.4)
        for low, high in scatter.y_bounds:
            for bound in (low, high):
                if np.isfinite(bound):
                    axes.axhline(bound, color=dashboard.NEUTRAL, linewidth=1.4)
        for bound in (self._gate_low.value(), self._gate_high.value()):
            axes.axvline(bound, color=dashboard.NEUTRAL, linestyle="--", linewidth=1.4)

        # Where a gate on each axis both keep an object: the rectangle the pair of
        # them reads as, shaded rather than outlined so it cannot be mistaken for
        # a bound of its own.
        if scatter.x_bounds and scatter.y_bounds:
            (x_low, x_high), (y_low, y_high) = scatter.x_bounds[0], scatter.y_bounds[0]
            limits_x, limits_y = axes.get_xlim(), axes.get_ylim()
            axes.axvspan(
                max(x_low, limits_x[0]), min(x_high, limits_x[1]),
                ymin=0, ymax=1, color=dashboard.PALETTE[0], alpha=0.06, linewidth=0,
            )
            axes.axhspan(
                max(y_low, limits_y[0]), min(y_high, limits_y[1]),
                color=dashboard.PALETTE[0], alpha=0.06, linewidth=0,
            )
            axes.set_xlim(limits_x)
            axes.set_ylim(limits_y)

        axes.set_xlabel(scatter.x_feature, color=dashboard.INK)
        axes.set_ylabel(scatter.y_feature, color=dashboard.INK)
        axes.tick_params(colors=dashboard.INK, labelsize=8)
        axes.grid(True, color=dashboard.GRID, linewidth=0.6)
        axes.set_axisbelow(True)
        axes.legend(fontsize=7, frameon=False, labelcolor=dashboard.INK, loc="best")
        axes.set_title(
            f"{scatter.n_kept:,} of {len(scatter.x):,} drawn objects kept",
            color=dashboard.INK,
            fontsize=9,
        )
        self._gate_canvas.draw_idle()
        self._gate_plot_note.setText(scatter.describe())

    def _gate_feature_changed(self, _text: str = "") -> None:
        """Put the suggested bounds for a newly chosen feature in the boxes.

        A gate already set on that feature wins: coming back to it should show
        what was chosen, not overwrite it with a default.
        """
        if self._updating or self._adata is None:
            return
        feature = self._gate_feature.currentText()
        existing = next((gate for gate in self._gates if gate.feature == feature), None)
        if existing is not None:
            self._set_bounds(existing.low, existing.high)
        else:
            self.suggest_bounds()
        self.draw_gate_plot()

    def _bounds_changed(self, _value: float = 0.0) -> None:
        if not self._updating:
            self.draw_gate_plot()

    def _set_bounds(self, low: float, high: float) -> None:
        was = self._updating
        self._updating = True
        try:
            self._gate_low.setValue(float(low))
            self._gate_high.setValue(float(high))
        finally:
            self._updating = was

    def suggest_bounds(self) -> None:
        """Bounds at the tails of what is currently selected."""
        histogram = self._histogram()
        if histogram is None:
            return
        self._set_bounds(histogram.suggested_low, histogram.suggested_high)
        self.draw_gate_plot()

    def _selection_mask(self) -> np.ndarray | None:
        if self._adata is None:
            return None
        return populations.sample_mask(self._adata, self.selected_samples())

    def _histogram(self):
        feature = self._gate_feature.currentText()
        if self._adata is None or not feature:
            return None
        try:
            return populations.feature_histogram(
                self._adata, feature, within=self._selection_mask()
            )
        except Exception:
            logger.exception("could not bin %r", feature)
            return None

    def draw_histogram(self) -> None:
        """The chosen feature's distribution, with the bounds drawn on it."""
        self._gate_figure.clear()
        axes = self._gate_figure.add_subplot(111)
        histogram = self._histogram()
        if histogram is None or histogram.n_finite == 0:
            axes.text(
                0.5,
                0.5,
                "nothing to bin" if histogram is None else "no values for this feature here",
                ha="center",
                va="center",
                color=dashboard.INK,
                transform=axes.transAxes,
            )
            axes.set_axis_off()
            self._gate_canvas.draw_idle()
            self._gate_plot_note.setText("—")
            return

        widths = np.diff(histogram.edges)
        axes.bar(
            histogram.edges[:-1],
            histogram.counts,
            width=widths,
            align="edge",
            color=dashboard.PALETTE[0],
            edgecolor=dashboard.OUTLINE,
            linewidth=0.2,
        )
        for bound in (self._gate_low.value(), self._gate_high.value()):
            axes.axvline(bound, color=dashboard.NEUTRAL, linestyle="--", linewidth=1.4)
        axes.set_xlabel(histogram.feature, color=dashboard.INK)
        axes.set_ylabel("objects", color=dashboard.INK)
        axes.tick_params(colors=dashboard.INK, labelsize=8)
        axes.grid(True, axis="y", color=dashboard.GRID, linewidth=0.6)
        axes.set_axisbelow(True)
        blank = f", {histogram.n_blank:,} blank" if histogram.n_blank else ""
        axes.set_title(
            f"{histogram.n_finite:,} objects{blank}", color=dashboard.INK, fontsize=9
        )
        self._gate_canvas.draw_idle()
        self._gate_plot_note.setText(
            f"{histogram.n_finite:,} object(s) binned"
            + (f", {histogram.n_blank:,} with no value for this feature" if histogram.n_blank else "")
            + "; every object is binned, so this is not a sample."
        )

    def _gate_from_boxes(self) -> populations.Gate | None:
        """The gate the editor currently describes, or None if it cannot make one."""
        feature = self._gate_feature.currentText()
        if self._adata is None or not feature:
            return None
        low, high = self._gate_low.value(), self._gate_high.value()
        if high < low:
            low, high = high, low
        return populations.Gate(
            feature, low, high, True, str(self._gate_join.currentData())
        )

    def add_gate(self) -> None:
        """Add the bounds in the boxes as another gate, at the end of the list.

        Appends rather than replacing a gate on the same feature, which is what
        this did when every gate was ANDed and a second bound on one feature
        could only ever contradict the first. With ``or`` it is the useful case:
        two windows on one measurement keep the debris and the fused objects and
        nothing in between.
        """
        gate = self._gate_from_boxes()
        if gate is None:
            return
        self._gates.append(gate)
        self._recompute_gates()

    def update_gate(self) -> None:
        """Replace the selected gate, keeping its position in the order."""
        row = self._gate_table.currentRow()
        gate = self._gate_from_boxes()
        if gate is None or not (0 <= row < len(self._gates)):
            self._gate_status.setText("Select a gate in the table to update it.")
            return
        # Its enabled state is the row's, not the editor's: updating the bounds of
        # a gate that is switched off should not switch it back on.
        self._gates[row] = populations.Gate(
            gate.feature, gate.low, gate.high, self._gates[row].enabled, gate.join
        )
        self._recompute_gates()

    def remove_gate(self) -> None:
        row = self._gate_table.currentRow()
        if 0 <= row < len(self._gates):
            self._gates.pop(row)
            self._recompute_gates()

    def _gate_toggled(self, item) -> None:
        """Switch a gate off from its row rather than deleting it.

        Half of using a gate is finding out what it costs, and that means being
        able to take it out and put it back without retyping its bounds.
        """
        if self._updating or item.column() != 0:
            return
        row = item.row()
        if not (0 <= row < len(self._gates)):
            return
        gate = self._gates[row]
        self._gates[row] = populations.Gate(
            gate.feature, gate.low, gate.high, item.checkState() == Qt.Checked, gate.join
        )
        self._recompute_gates()

    def _gate_row_chosen(self) -> None:
        row = self._gate_table.currentRow()
        if self._updating or not (0 <= row < len(self._gates)):
            return
        gate = self._gates[row]
        self._updating = True
        try:
            self._gate_feature.setCurrentText(gate.feature)
            index = self._gate_join.findData(gate.join)
            if index >= 0:
                self._gate_join.setCurrentIndex(index)
        finally:
            self._updating = False
        self._set_bounds(gate.low, gate.high)
        self.draw_gate_plot()

    def _fill_gate_table(self, report: populations.GateReport | None = None) -> None:
        was = self._updating
        self._updating = True
        try:
            self._gate_table.setRowCount(len(self._gates))
            for row, gate in enumerate(self._gates):
                cost = report.costs[row] if report is not None and row < len(report.costs) else None
                name = QTableWidgetItem(gate.feature)
                name.setFlags(name.flags() | Qt.ItemIsUserCheckable)
                name.setCheckState(Qt.Checked if gate.enabled else Qt.Unchecked)
                self._gate_table.setItem(row, 0, name)
                for column, text in (
                    # Blank on the first row: it is joined to nothing, and showing
                    # "and" there reads as a rule that was chosen.
                    (1, "" if row == 0 else str(gate.join)),
                    (2, f"{gate.low:.4g}"),
                    (3, f"{gate.high:.4g}"),
                    (
                        4,
                        "missing"
                        if cost is not None and cost.missing
                        else ("—" if cost is None else f"{cost.n_removed:,}"),
                    ),
                    (5, "—" if cost is None else f"{cost.n_blank:,}"),
                ):
                    self._gate_table.setItem(row, column, QTableWidgetItem(text))
        finally:
            self._updating = was

    def _recompute_gates(self) -> None:
        """Re-apply the gates and say what they cost. Invalidates any calls made."""
        if self._adata is None:
            return
        selection = self._selection_mask()
        try:
            report = populations.gate_report(self._adata, self._gates, within=selection)
            self._keep = populations.gate_mask(self._adata, self._gates, within=selection)
        except Exception as error:
            logger.exception("could not apply the gates")
            self._gate_status.setText(f"Could not apply the gates: {error}")
            return

        self._fill_gate_table(report)
        # The rule on its own line: it is the one thing a table of rows cannot
        # show, because the joins are between the rows rather than in them.
        percent = 100.0 * report.n_kept / report.n_objects if report.n_objects else float("nan")
        lines = [
            f"{report.n_kept:,} of {report.n_objects:,} objects kept ({percent:.1f} %)",
            f"Rule, applied in order: {report.expression}",
        ]
        if report.combined:
            lines.append(
                "“removed” is each gate on its own, so with or / xor in the rule the "
                "column does not add up to the total kept."
            )
        if report.missing:
            lines.append("Not in this file: " + ", ".join(report.missing))
        self._gate_status.setText("\n".join(lines))
        self.draw_gate_plot()
        # The thresholds were computed over the previous gating, so they are no
        # longer the thresholds of what is on screen.
        if self._calls is not None:
            self._calls = None
            self._summary = None
            self._summary_model.set_frame(None)
            self._threshold_model.set_frame(None)
            self._rule_status.setText("The gates changed — apply the rule again.")
            self._headline.setText("—")
            self.draw_plot()

    # -- the rule -------------------------------------------------------------

    def _statistic_changed(self, _index: int = 0) -> None:
        """Show only the number the chosen statistic uses."""
        statistic = self._statistic_box.currentData()
        self._k_box.setEnabled(statistic != populations.STAT_PERCENTILE)
        self._percentile_box.setEnabled(statistic == populations.STAT_PERCENTILE)

    def rule(self) -> populations.PositiveRule:
        """The rule the boxes currently describe."""
        normalise_scope = self._signal_box.currentData() or ""
        return populations.PositiveRule(
            feature=self._rule_feature.currentText(),
            statistic=str(self._statistic_box.currentData()),
            k=float(self._k_box.value()),
            percentile=float(self._percentile_box.value()),
            scope=str(self._scope_box.currentData()),
            normalised=bool(normalise_scope),
            normalise_scope=str(normalise_scope or analysis.SCOPE_IMAGE),
        )

    def apply_rule(self) -> None:
        """Compute the thresholds over the gated objects and call every object."""
        if self._adata is None:
            self._rule_status.setText("Open a file first.")
            return
        rule = self.rule()
        if not rule.feature:
            self._rule_status.setText("Choose a measurement.")
            return
        if self._keep is None:
            self._recompute_gates()
        keep = self._keep
        if keep is None or not keep.any():
            self._rule_status.setText("Nothing is selected — tick a sample in Prepare.")
            return

        try:
            calls = populations.call_positives(self._adata, rule, keep=keep)
            summary = populations.sample_summary(self._adata, calls, keep=keep)
        except Exception as error:
            logger.exception("could not apply %s", rule.describe())
            self._rule_status.setText(f"Could not apply the rule: {error}")
            return

        self._calls = calls
        self._summary = summary
        self._threshold_model.set_frame(calls.thresholds)
        self._summary_model.set_frame(summary)

        n_kept = int(keep.sum())
        percent = 100.0 * calls.n_positive / n_kept if n_kept else float("nan")
        lines = [
            rule.describe(),
            f"{calls.n_positive:,} of {n_kept:,} kept objects positive ({percent:.1f} %).",
        ]
        lines += list(calls.notes)
        self._rule_status.setText("\n".join(lines))
        self._headline.setText(f"{percent:.1f} % positive")
        self.draw_plot()
        self._tabs.setCurrentIndex(2)

    # -- explore --------------------------------------------------------------

    def draw_plot(self) -> None:
        """Draw the chosen view of the summary. Every colour comes from the dashboard."""
        self._figure.clear()
        axes = self._figure.add_subplot(111)
        summary = self._summary
        if summary is None or len(summary) == 0:
            axes.text(
                0.5,
                0.5,
                "apply a rule in “2. Positive”",
                ha="center",
                va="center",
                color=dashboard.INK,
                transform=axes.transAxes,
            )
            axes.set_axis_off()
            self._canvas.draw_idle()
            return

        kind = self._plot_box.currentText()
        if kind == "percent positive per well" or (
            self._per_well.isChecked() and kind == "percent positive per sample"
        ):
            self._draw_wells(axes, populations.well_summary(summary))
        elif kind == "percent positive per sample":
            self._draw_samples(axes, summary)
        elif kind == "objects kept and excluded":
            self._draw_counts(axes, summary)
        else:
            self._draw_signal(axes)
        self._canvas.draw_idle()

    def _draw_samples(self, axes, summary) -> None:
        names = [str(name) for name in summary["sample"]]
        wells = sorted({str(name) for name in summary["well"]})
        palette = dashboard.paint("well", dashboard.colour_map(wells))
        axes.bar(
            range(len(names)),
            summary["percent positive"].to_numpy(dtype=float),
            color=[palette.get(str(well), dashboard.OVERFLOW_COLOUR) for well in summary["well"]],
            edgecolor=dashboard.OUTLINE,
            linewidth=0.3,
        )
        self._label_bars(axes, names, "positive (%)")

    def _draw_wells(self, axes, wells) -> None:
        names = [str(name) for name in wells["well"]]
        palette = dashboard.paint("well", dashboard.colour_map(names))
        axes.bar(
            range(len(names)),
            wells["percent positive"].to_numpy(dtype=float),
            color=[palette[name] for name in names],
            edgecolor=dashboard.OUTLINE,
            linewidth=0.3,
        )
        self._label_bars(axes, names, "positive (%)")

    def _draw_counts(self, axes, summary) -> None:
        names = [str(name) for name in summary["sample"]]
        kept = summary["kept"].to_numpy(dtype=float)
        axes.bar(
            range(len(names)), kept, color=dashboard.PALETTE[0],
            edgecolor=dashboard.OUTLINE, linewidth=0.3, label="kept",
        )
        axes.bar(
            range(len(names)),
            summary["excluded"].to_numpy(dtype=float),
            bottom=kept,
            color=dashboard.NEUTRAL,
            edgecolor=dashboard.OUTLINE,
            linewidth=0.3,
            label="excluded by the gates",
        )
        axes.legend(fontsize=7, frameon=False, labelcolor=dashboard.INK)
        self._label_bars(axes, names, "objects")

    def _draw_signal(self, axes) -> None:
        """The signal of the kept objects, with the positives drawn on top.

        One histogram rather than one per sample: the question it answers is
        whether the threshold sits in a gap or through the middle of a single
        mode, and that is visible on the pooled distribution.
        """
        calls, keep = self._calls, self._keep
        if calls is None or keep is None:
            return
        values = np.asarray(calls.values, dtype=float)[keep]
        positive = np.asarray(calls.positive, dtype=bool)[keep]
        finite = np.isfinite(values)
        if not finite.any():
            axes.set_axis_off()
            return
        # Through the same helper the Measurement analysis panel paints with, so
        # "the range of this measurement" means one thing in the application
        # rather than one thing per plot. Its percentiles are this panel's, which
        # are tighter than a colour scale's for the same reason the suggested
        # bounds are: a threshold is read against the tails.
        low, high = analysis.value_range(
            values[finite],
            populations.SUGGEST_LOW_PERCENTILE,
            populations.SUGGEST_HIGH_PERCENTILE,
        )
        edges = np.linspace(low, high, populations.DEFAULT_BINS + 1)
        axes.hist(
            [np.clip(values[finite & ~positive], low, high), np.clip(values[finite & positive], low, high)],
            bins=edges,
            stacked=True,
            color=[dashboard.PALETTE[0], dashboard.PALETTE[1]],
            label=["negative", "positive"],
        )
        for threshold in sorted(set(calls.thresholds["threshold"].dropna())):
            axes.axvline(threshold, color=dashboard.NEUTRAL, linestyle="--", linewidth=1.2)
        axes.legend(fontsize=7, frameon=False, labelcolor=dashboard.INK)
        axes.set_xlabel(calls.rule.feature, color=dashboard.INK)
        axes.set_ylabel("objects", color=dashboard.INK)
        axes.tick_params(colors=dashboard.INK, labelsize=8)
        axes.grid(True, axis="y", color=dashboard.GRID, linewidth=0.6)
        axes.set_axisbelow(True)

    def _label_bars(self, axes, names: list[str], ylabel: str) -> None:
        axes.set_xticks(range(len(names)))
        axes.set_xticklabels(names, rotation=90, fontsize=7, color=dashboard.INK)
        axes.set_ylabel(ylabel, color=dashboard.INK)
        axes.tick_params(colors=dashboard.INK, labelsize=8)
        axes.grid(True, axis="y", color=dashboard.GRID, linewidth=0.6)
        axes.set_axisbelow(True)

    # -- out ------------------------------------------------------------------

    def export_summary(self) -> None:
        """Write the per-sample table as CSV, exactly as it is on screen."""
        if self._summary is None or len(self._summary) == 0:
            self._status.setText("Nothing to export yet — apply a rule first.")
            return
        default = str(
            (self._path.with_name(f"{self._path.stem}_populations.csv"))
            if self._path is not None
            else "populations.csv"
        )
        path, _filter = QFileDialog.getSaveFileName(
            self, "Export the summary", default, "CSV (*.csv);;All files (*)"
        )
        if not path:
            return
        try:
            self._summary.to_csv(path, index=False, encoding="utf-8")
        except Exception as error:
            logger.exception("could not write %s", path)
            QMessageBox.warning(self, "Export", f"Could not write the file:\n{error}")
            return
        self._status.setText(f"Wrote {Path(path).name}.")

    def save_into_file(self) -> None:
        """Run the whole definition and write it back into an ``.h5ad``.

        Re-runs rather than saving what is on screen: the file has to carry the
        gates, the rule and the per-object calls as one consistent set, and the
        only way to be sure they are is to produce them together.
        """
        if self._adata is None:
            return
        rule = self.rule()
        if not rule.feature:
            self._status.setText("Choose a measurement in “2. Positive” first.")
            return
        default = str(
            self._path.with_name(f"{self._path.stem}_populations.h5ad")
            if self._path is not None
            else "populations.h5ad"
        )
        path, _filter = QFileDialog.getSaveFileName(
            self, "Write the analysis into an .h5ad", default, FILE_FILTER
        )
        if not path:
            return
        try:
            record = populations.run_populations(
                self._adata, self._gates, rule, samples=self.selected_samples()
            )
            written = populations.save_populations(path, self._adata)
        except Exception as error:
            logger.exception("could not write %s", path)
            QMessageBox.warning(self, "Write", f"Could not write the file:\n{error}")
            return
        self._status.setText(
            f"Wrote {written.name}: {record['n_positive']:,} of {record['n_kept']:,} kept "
            f"objects positive ({record['percent_positive']:.1f} %), with the gates and "
            f"the rule on it."
        )
