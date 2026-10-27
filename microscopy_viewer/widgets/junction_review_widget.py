"""Dock panel: review the QC pictures, then redraw the junctions that need it.

Two halves, used one after the other.

**Review.** The pictures of a run's QC folder are listed, worst first, and the
selected one is shown on the canvas — where it can be zoomed, which a thumbnail
in a panel cannot. A key or a button decides it: accept, redraw, exclude. The
decision is written to ``review.csv`` at once and the next picture comes up.

**Redraw.** The samples marked *redraw* are a queue. One is opened in place of
the picture with the Fusion region panel put in drawing mode on its nuclear
stain; the line is drawn, the band appears, and *Save and next* stores it with
the sample and opens the following one. The buttons that drawing needs are
repeated here because the Fusion region panel is a tab beside this one and
cannot be seen at the same time.

The bookkeeping is in :mod:`microscopy_viewer.junction_review`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtGui import QKeySequence
from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QShortcut,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from .. import junction as jn
from .. import junction_review as jr
from ..utils import get_logger
from .sample_files import source_of

logger = get_logger("junction_review_widget")

#: The layer a QC picture is shown in.
PICTURE_LAYER_NAME = "QC picture"

#: Rows the sample list shows at the least.
LIST_ROWS = 18

FILTER_ALL = "All"
FILTER_UNDECIDED = "To review"

#: How a decision is written beside a sample's name in the list.
MARKS = {jr.UNDECIDED: "", jr.ACCEPT: "✓ accepted", jr.REDRAW: "✎ to redraw",
         jr.EXCLUDE: "✗ excluded", jr.REDRAWN: "✓ redrawn"}


class JunctionReviewWidget(QWidget):
    """Decide each QC picture, then draw the junctions marked for it."""

    def __init__(self, app, parent: QWidget | None = None):
        super().__init__(parent)
        self._app = app
        self._viewer = app.viewer
        self._items: list[jr.ReviewItem] = []
        self._shown: list[jr.ReviewItem] = []
        #: The sample open for drawing, when there is one.
        self._drawing: jr.ReviewItem | None = None
        self._filling = False
        #: Set while the list is rebuilt under an open sample: no picture then.
        self._quiet = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(self._build_folder_box())
        layout.addWidget(self._build_review_box(), stretch=1)
        layout.addWidget(self._build_redraw_box())
        self._status = QLabel("Choose the QC folder of a run, or Reload after one has finished.")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        for key, slot in (("A", lambda: self.decide(jr.ACCEPT)), ("R", lambda: self.decide(jr.REDRAW)),
                          ("X", lambda: self.decide(jr.EXCLUDE)), ("U", lambda: self.decide(jr.UNDECIDED)),
                          ("Right", lambda: self.step(1)), ("Left", lambda: self.step(-1))):
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.setContext(Qt.WidgetWithChildrenShortcut)
            shortcut.activated.connect(slot)

        fusion = getattr(app, "fusion_widget", None)
        if fusion is not None and hasattr(fusion, "region_saved"):
            fusion.region_saved.connect(self._on_region_saved)

    # -- construction ---------------------------------------------------------

    def _build_folder_box(self) -> QGroupBox:
        box = QGroupBox("QC folder")
        outer = QVBoxLayout(box)
        self._source = QComboBox()
        self._source.addItems([label for label, _relative in jr.SOURCES])
        self._source.setToolTip(
            "Which run's pictures to review, in the experiment folder. The junction "
            "pictures show the whole section with the line and the band, which is what "
            "a line is judged on; the junction analysis pictures show only the counted region."
        )
        self._source.currentIndexChanged.connect(lambda _index: self.use_source())
        outer.addWidget(self._source)

        row = QHBoxLayout()
        self._folder_edit = QLineEdit()
        self._folder_edit.setPlaceholderText("…/junction_QC")
        self._folder_edit.setToolTip("The folder holding ok, check and fail. Decisions are kept in review.csv inside it.")
        self._folder_edit.editingFinished.connect(self.reload)
        row.addWidget(self._folder_edit, stretch=1)
        browse = QPushButton("Browse…")
        browse.clicked.connect(self.browse)
        row.addWidget(browse)
        reload_button = QPushButton("Reload")
        reload_button.setToolTip("Read the folder again — after a run, or after the experiment folder changed.")
        reload_button.clicked.connect(self.use_source)
        row.addWidget(reload_button)
        outer.addLayout(row)
        return box

    def _build_review_box(self) -> QGroupBox:
        box = QGroupBox("Review")
        outer = QVBoxLayout(box)

        row = QHBoxLayout()
        self._status_filter = QComboBox()
        self._status_filter.addItems([FILTER_ALL, jn.STATUS_FAIL, jn.STATUS_CHECK, jn.STATUS_OK])
        self._status_filter.setToolTip("Show only the samples a run filed under this status.")
        self._status_filter.currentIndexChanged.connect(lambda _index: self.refill())
        row.addWidget(QLabel("Status"))
        row.addWidget(self._status_filter)
        self._decision_filter = QComboBox()
        self._decision_filter.addItems([FILTER_UNDECIDED, FILTER_ALL, *jr.DECISIONS])
        self._decision_filter.setToolTip("“To review” hides a sample once it is decided, so the list shrinks to nothing.")
        self._decision_filter.currentIndexChanged.connect(lambda _index: self.refill())
        row.addWidget(QLabel("Show"))
        row.addWidget(self._decision_filter, stretch=1)
        outer.addLayout(row)

        self._list = QListWidget()
        self._list.setAlternatingRowColors(False)
        self._list.currentRowChanged.connect(self._on_row_changed)
        # Long enough to see a run of samples at once, and it takes whatever
        # height the dock has beyond the controls.
        rows = max(self._list.fontMetrics().height() + 6, 18)
        self._list.setMinimumHeight(LIST_ROWS * rows)
        self._list.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        outer.addWidget(self._list, stretch=1)

        self._note = QLineEdit()
        self._note.setPlaceholderText("Note for this sample (kept in review.csv)")
        self._note.editingFinished.connect(self._on_note_edited)
        outer.addWidget(self._note)

        row = QHBoxLayout()
        for text, decision, tip in (
            ("Accept (A)", jr.ACCEPT, "The junction is where it should be."),
            ("Redraw (R)", jr.REDRAW, "The sample is good and the line is not: add it to the queue below."),
            ("Exclude (X)", jr.EXCLUDE, "Do not use this sample. Say why in the note."),
            ("Undo (U)", jr.UNDECIDED, "Back to undecided."),
        ):
            button = QPushButton(text)
            button.setToolTip(tip + " The keys work while this panel has the focus; ← and → move through the list.")
            button.clicked.connect(lambda _checked=False, d=decision: self.decide(d))
            row.addWidget(button)
        outer.addLayout(row)
        return box

    def _build_redraw_box(self) -> QGroupBox:
        box = QGroupBox("Redraw")
        outer = QVBoxLayout(box)
        self._queue_label = QLabel("Nothing to redraw yet.")
        self._queue_label.setWordWrap(True)
        outer.addWidget(self._queue_label)

        row = QHBoxLayout()
        self._open_next = QPushButton("Open the next to redraw")
        self._open_next.setToolTip(
            "Open the next sample marked “redraw” in place of what is on screen, with "
            "the line tool ready on its nuclear stain. Click along the junction and "
            "double-click to finish; the band appears."
        )
        self._open_next.clicked.connect(self.open_next)
        row.addWidget(self._open_next)
        open_this = QPushButton("Open this one")
        open_this.setToolTip("Open the sample selected in the list for drawing, whatever was decided about it.")
        open_this.clicked.connect(self.open_selected)
        row.addWidget(open_this)
        outer.addLayout(row)

        build = QPushButton("Build the band from the drawn line")
        build.setToolTip(
            "Build the band from the line drawn on the canvas, with the Fusion region "
            "panel's band settings (width, gap, side, following the tissue). Nothing is "
            "searched for: the line is yours, as drawn or as edited since.\n"
            "With no sample open, opens the one selected in the list first; with no line "
            "drawn, the line tool is made ready instead."
        )
        build.clicked.connect(self.build_junction)
        outer.addWidget(build)

        row = QHBoxLayout()
        again = QPushButton("Draw again")
        again.setToolTip("Clear the line and the band and start the line over.")
        again.clicked.connect(self.draw_again)
        row.addWidget(again)
        flip = QPushButton("Other side")
        flip.setToolTip("Put the band on the other side of the line.")
        flip.clicked.connect(self.flip_side)
        row.addWidget(flip)
        self._save_next = QPushButton("Save and next")
        self._save_next.setToolTip(
            "Store the band with the sample, as the Fusion region panel's Save does and "
            "with its width, mark the sample redrawn, and open the next one in the queue."
        )
        self._save_next.clicked.connect(self.save_and_next)
        row.addWidget(self._save_next)
        outer.addLayout(row)

        self._advance = QCheckBox("Open the next one after saving")
        self._advance.setChecked(True)
        outer.addWidget(self._advance)

        count = QPushButton("Count the redrawn samples")
        count.setToolTip(
            "Run the Junction analysis panel on the samples redrawn here, with its "
            "settings. Their stored region is now the one drawn by hand. The result is a "
            "workbook of those samples only. To count every accepted and redrawn sample, "
            "choose “Sent on by a junction review” in the Junction analysis panel."
        )
        count.clicked.connect(self.count_redrawn)
        outer.addWidget(count)
        return box

    # -- the folder -----------------------------------------------------------

    def _experiment_folder(self) -> Path | None:
        experiment = getattr(self._app, "experiment_widget", None)
        try:
            return experiment.folder() if experiment is not None else None
        except Exception:
            return None

    def qc_folder(self) -> Path | None:
        text = self._folder_edit.text().strip()
        return Path(text) if text else None

    def use_source(self) -> None:
        """Point at the chosen run's QC folder in the experiment folder, and read it."""
        folder = self._experiment_folder()
        if folder is not None:
            self._folder_edit.setText(str(jr.source_folder(folder, jr.SOURCES[self._source.currentIndex()][1])))
        self.reload()

    def browse(self) -> None:
        start = str(self.qc_folder() or self._experiment_folder() or Path.home())
        chosen = QFileDialog.getExistingDirectory(self, "The QC folder (holding ok, check and fail)", start)
        if chosen:
            self._folder_edit.setText(chosen)
            self.reload()

    def _sample_folder(self, qc_folder: Path) -> Path:
        """Where the samples are: the experiment folder, or above the QC folder."""
        experiment = self._experiment_folder()
        if experiment is not None:
            return experiment
        for parent in qc_folder.parents:
            if jr.find_samples(parent):
                return parent
        return qc_folder.parent

    def reload(self) -> None:
        folder = self.qc_folder()
        if folder is None or not folder.is_dir():
            self._items = []
            self.refill()
            self._status.setText("No such folder. Run the junction first, or choose its QC folder.")
            return
        self._items = jr.scan(folder, self._sample_folder(folder))
        self.refill()
        missing = sum(1 for item in self._items if item.path is None)
        text = jr.summary(self._items) + "."
        if missing:
            text += f" {missing} have no sample file beside them and cannot be opened."
        self._status.setText(text if self._items else "No pictures in ok, check or fail in that folder.")

    # -- the list -------------------------------------------------------------

    def _visible(self, item: jr.ReviewItem) -> bool:
        status = self._status_filter.currentText()
        if status != FILTER_ALL and item.status != status:
            return False
        wanted = self._decision_filter.currentText()
        if wanted == FILTER_ALL:
            return True
        return item.decision == (jr.UNDECIDED if wanted == FILTER_UNDECIDED else wanted)

    @staticmethod
    def _label(item: jr.ReviewItem) -> str:
        mark = MARKS.get(item.decision, "")
        return f"[{item.status}]  {item.name}" + (f"   {mark}" if mark else "")

    def refill(self, keep: jr.ReviewItem | None = None, row: int | None = None) -> None:
        """Rebuild the list for the filters, staying on *keep* or at *row*."""
        self._filling = True
        try:
            self._shown = [item for item in self._items if self._visible(item)]
            self._list.clear()
            for item in self._shown:
                entry = QListWidgetItem(self._label(item))
                entry.setToolTip(str(item.picture))
                self._list.addItem(entry)
        finally:
            self._filling = False
        if keep is not None and keep in self._shown:
            target = self._shown.index(keep)
        else:
            target = min(row if row is not None else 0, len(self._shown) - 1)
        if self._shown:
            self._list.setCurrentRow(target)
        self._update_queue()

    def current(self) -> jr.ReviewItem | None:
        row = self._list.currentRow()
        return self._shown[row] if 0 <= row < len(self._shown) else None

    def _on_row_changed(self, row: int) -> None:
        if self._filling:
            return
        item = self.current()
        self._note.blockSignals(True)
        self._note.setText(item.note if item is not None else "")
        self._note.blockSignals(False)
        if item is not None and not self._quiet:
            self.show_picture(item)

    def step(self, by: int) -> None:
        if self._shown:
            self._list.setCurrentRow(int(np.clip(self._list.currentRow() + by, 0, len(self._shown) - 1)))

    def _on_note_edited(self) -> None:
        item = self.current()
        if item is not None and item.note != self._note.text():
            item.note = self._note.text()
            self._store()

    def _store(self) -> None:
        folder = self.qc_folder()
        if folder is None:
            return
        try:
            jr.save_decisions(folder, self._items)
        except Exception as exc:
            logger.exception("could not write the review")
            self._status.setText(f"Could not write {jr.REVIEW_FILE}: {exc}")

    def decide(self, decision: str) -> None:
        """Record *decision* on the selected sample and move to the next."""
        item = self.current()
        if item is None:
            return
        row = self._list.currentRow()
        jr.decide(item, decision, self._note.text())
        self._store()
        if self._visible(item):
            self._list.item(row).setText(self._label(item))
            self._update_queue()
            self.step(1)
        else:
            # It left the list; the one after it has moved up into its row.
            self.refill(row=row)
        self._status.setText(jr.summary(self._items) + ".")

    # -- the picture ----------------------------------------------------------

    def _remove_picture(self) -> None:
        if PICTURE_LAYER_NAME in self._viewer.layers:
            self._viewer.layers.remove(PICTURE_LAYER_NAME)

    def show_picture(self, item: jr.ReviewItem) -> None:
        """Put the sample's QC picture on the canvas, filling it."""
        from PIL import Image

        from ..measurements import NOT_A_ROI

        try:
            with Image.open(item.picture) as stored:
                pixels = np.asarray(stored.convert("RGB"))
        except Exception as exc:
            self._status.setText(f"Could not read {item.picture.name}: {exc}")
            return
        try:
            if self._viewer.dims.ndisplay != 2:
                self._viewer.dims.ndisplay = 2
            if PICTURE_LAYER_NAME in self._viewer.layers:
                layer = self._viewer.layers[PICTURE_LAYER_NAME]
                same = tuple(layer.data.shape) == tuple(pixels.shape)
                layer.data = pixels
                if not same:
                    self._viewer.reset_view()
            else:
                # Under a sample that is open, other layers would show through
                # nothing: the picture is opaque and goes on top.
                for layer in self._viewer.layers:
                    layer.visible = False
                from .sample_files import TRANSIENT_KEY

                self._viewer.add_image(pixels, name=PICTURE_LAYER_NAME, rgb=True,
                                       metadata={NOT_A_ROI: True, TRANSIENT_KEY: True})
                self._viewer.reset_view()
        except Exception:
            logger.exception("could not show the QC picture")

    # -- redrawing ------------------------------------------------------------

    def _update_queue(self) -> None:
        queue = jr.redraw_queue(self._items)
        done = sum(1 for item in self._items if item.decision == jr.REDRAWN)
        text = f"{len(queue)} to redraw, {done} redrawn." if (queue or done) else "Nothing to redraw yet."
        if self._drawing is not None:
            text += f" Open now: {self._drawing.name}."
        self._queue_label.setText(text)
        self._open_next.setEnabled(bool(queue))

    def _fusion(self):
        fusion = getattr(self._app, "fusion_widget", None)
        if fusion is None:
            self._status.setText("The Fusion region panel is not available.")
        return fusion

    def _close_samples(self) -> None:
        """Take every open sample off screen with everything of theirs, and the picture."""
        from .sample_files import clear_samples

        self._remove_picture()
        clear_samples(self._viewer)

    def open_for_drawing(self, item: jr.ReviewItem) -> bool:
        """Open *item*'s sample alone, ready for its junction to be drawn."""
        fusion = self._fusion()
        if fusion is None:
            return False
        if item.path is None or not Path(item.path).is_file():
            self._status.setText(f"{item.name}: its file was not found, so it cannot be opened.")
            return False
        counts = getattr(self._app, "positive_counter_widget", None)
        if counts is not None and hasattr(counts, "settle_unsaved") and not counts.settle_unsaved(self):
            return False
        fusion.clear()
        self._close_samples()
        for layer in self._viewer.layers:
            layer.visible = True
        if not self._app.open_paths([item.path]):
            self._status.setText(f"{item.name} could not be opened.")
            return False
        try:
            self._viewer.reset_view()
        except Exception:
            logger.debug("could not reset the view", exc_info=True)
        self._drawing = item
        fusion.prepare_for_drawing(jn.NUCLEAR_HINTS)
        self._update_queue()
        self._status.setText(
            f"{item.name}: click along the junction on the canvas, double-click to finish. "
            "“Other side” if the band is on the wrong organoid, then “Save and next”."
        )
        return True

    def open_next(self) -> None:
        queue = jr.redraw_queue(self._items)
        if not queue:
            self._status.setText("Nothing is marked to redraw.")
            return
        # After the one being drawn, when there is one, so skipping is possible.
        start = queue.index(self._drawing) + 1 if self._drawing in queue else 0
        self.open_for_drawing(queue[start % len(queue)])

    def open_selected(self) -> None:
        item = self.current()
        if item is None:
            self._status.setText("Select a sample in the list first.")
            return
        self.open_for_drawing(item)

    def draw_again(self) -> None:
        fusion = self._fusion()
        if fusion is not None:
            fusion.clear()
            fusion.prepare_for_drawing(jn.NUCLEAR_HINTS)

    def build_junction(self) -> None:
        """Build the band from the line drawn on the open sample — no search for a line of its own.

        The band is the Fusion region panel's for a drawn line: its width, gap
        and side, following the tissue when that is on. Nothing is looked for
        automatically; with no line drawn, nothing is built.
        """
        fusion = self._fusion()
        if fusion is None:
            return
        if self._drawing is None or fusion.image_layer() is None:
            item = self.current()
            if item is None:
                self._status.setText("Select a sample in the list first.")
                return
            if not self.open_for_drawing(item):
                return
        name = self._drawing.name
        layer = fusion.line_layer()
        if layer is not None and getattr(layer, "_is_creating", False):
            self._status.setText(f"{name}: finish the line first — double-click its last point.")
            return
        if fusion.drawn_line() is None:
            self._status.setText(f"{name}: no line drawn yet. Click along the junction on the canvas "
                                 "and double-click to finish, then build.")
            fusion.prepare_for_drawing(jn.NUCLEAR_HINTS)
            return
        self._status.setText(f"{name}: building the band from the drawn line…")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            QApplication.processEvents()
            built = fusion.update_region()
        finally:
            QApplication.restoreOverrideCursor()
        if not built:
            self._status.setText(f"{name}: could not build the band — {fusion.status_text()}")
            return
        self._status.setText(f"{name}: {fusion.status_text()} Wrong side? “Other side”. Then “Save and next”.")

    def flip_side(self) -> None:
        fusion = self._fusion()
        if fusion is not None:
            fusion.flip_side()

    def _on_region_saved(self, source: str) -> None:
        """A region was stored, here or in the Fusion region panel: tick it off."""
        try:
            saved = Path(source).resolve()
        except OSError:
            return
        for item in self._items:
            if item.path is not None and Path(item.path).resolve() == saved and item.decision in (jr.REDRAW, jr.UNDECIDED):
                jr.decide(item, jr.REDRAWN, item.note)
                self._store()
                # The sample is on screen: the list moves on without its picture.
                self._quiet = True
                try:
                    self.refill(keep=self.current(), row=self._list.currentRow())
                finally:
                    self._quiet = False
                break

    def save_and_next(self) -> None:
        fusion = self._fusion()
        if fusion is None:
            return
        drawing = self._drawing
        if drawing is None:
            self._status.setText("No sample is open for drawing — “Open the next to redraw” first.")
            return
        if not fusion.save():
            self._status.setText(f"{drawing.name}: nothing saved — {fusion.status_text()}")
            return
        queue = jr.redraw_queue(self._items)
        if self._advance.isChecked() and queue:
            self.open_for_drawing(queue[0])
        else:
            self._drawing = None
            self._update_queue()
            self._status.setText(f"{drawing.name} saved. " + ("" if queue else "The queue is empty. ")
                                 + jr.summary(self._items) + ".")

    def count_redrawn(self) -> None:
        """Count the redrawn samples, through the Junction analysis panel's review route."""
        paths = [item.path for item in self._items if item.decision == jr.REDRAWN and item.path is not None]
        folder = self.qc_folder()
        if not paths or folder is None:
            self._status.setText("No redrawn samples yet.")
            return
        analysis = getattr(self._app, "junction_analysis_widget", None)
        if analysis is None or not hasattr(analysis, "run_review"):
            self._status.setText("The Junction analysis panel is not available.")
            return
        # The decisions are on disk already (written after each one): the review file is the list.
        if analysis.run_review(folder / jr.REVIEW_FILE, include=(jr.REDRAWN,)):
            self._status.setText(f"Counting {len(paths)} redrawn sample(s) — progress is in the Junction analysis panel.")
