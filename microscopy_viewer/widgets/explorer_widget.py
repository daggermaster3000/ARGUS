"""Dock panel: browse a plate and open the part of it you actually want.

A converted 4i plate is a few hundred images in one folder. Dropping it on the
window builds a mosaic of every cycle — correct, and far more than anyone asked
for when the question is "what does well G/07 look like, and did the run pick up
its nuclei?". This panel answers that question: scan the plate, see every image
in it with the segmentations it already carries, look at a miniature to check the
well is not empty, then open the handful you want with their labels on top.

The tables find their images here too. A batch run names each object table after
the image it measured, so the list says which images have been measured as well as
which have been segmented, and the tables for the image in front of you are one
click from the Measurement analysis panel.

It is also the only place a plate is chosen. The Batch segmentation panel used to
carry its own copy of the store box, which meant two paths to keep in step and
two lists that could disagree; now this panel scans and every other panel is told
— the batch picker fills itself in, and the measurement analysis panel lists the
table folders sitting beside the store.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtGui import QImage, QPixmap
from qtpy.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .. import batch, explorer
from ..utils import get_logger
from .plate_picker import PlateSourceBox, WellAcquisitionPicker

logger = get_logger("explorer_widget")

#: Columns of the image list.
IMAGE_COLUMNS = ("Well", "Cycle", "Field", "Channels", "Size", "Segmentations", "Tables")

#: Marker on a table written for another acquisition of the same well. A 4i plate
#: images the same cells every cycle, so such a table does describe these objects
#: — but it was measured somewhere else and should not look like this image's own.
OTHER_CYCLE = "  (from {component})"

#: Images above which opening is refused without a second look. Each one is a
#: layer per channel plus a layer per label set, so a whole plate at four
#: channels is over a thousand layers and a window that will not redraw.
MANY_IMAGES = 24


class FileExplorerWidget(QWidget):
    """Scan a plate, pick images out of it, open them with their segmentations."""

    def __init__(self, app, parent: QWidget | None = None):
        super().__init__(parent)
        self._app = app
        self._survey: batch.PlateSurvey | None = None
        self._rows: list[batch.ImageJob] = []
        self._thumbnail: np.ndarray | None = None
        #: ``{stem: [table, ...]}`` for the folders beside the plate, built once a
        #: scan, because filling a column cannot cost a directory walk per row.
        self._table_index: dict[str, list[Path]] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self._source = PlateSourceBox(app)
        self._source.surveyed.connect(self._on_surveyed)
        self._source.failed.connect(self._on_failed)
        layout.addWidget(self._source)

        layout.addWidget(self._build_images_box(), stretch=1)
        layout.addWidget(self._build_preview_box())
        layout.addWidget(self._build_tables_box())
        layout.addLayout(self._build_open_row())

        self._status = QLabel("Choose a plate and press Scan.")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        self._update_enabled()

    # -- construction ---------------------------------------------------------

    def _build_images_box(self) -> QGroupBox:
        box = QGroupBox("Images")
        outer = QVBoxLayout(box)

        self._picker = WellAcquisitionPicker(all_cycles=False)
        self._picker.selectionChanged.connect(self._refresh_table)
        outer.addWidget(self._picker)

        self._table = QTableWidget(0, len(IMAGE_COLUMNS))
        self._table.setHorizontalHeaderLabels(list(IMAGE_COLUMNS))
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setToolTip(
            "Every image the wells and cycles above name. Select the rows to open; "
            "the miniature follows whichever row you touched last."
        )
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self._table.itemSelectionChanged.connect(self._on_row_selected)
        outer.addWidget(self._table, stretch=1)

        self._table_note = QLabel("—")
        self._table_note.setWordWrap(True)
        outer.addWidget(self._table_note)
        return box

    def _build_preview_box(self) -> QGroupBox:
        box = QGroupBox("Miniature")
        outer = QHBoxLayout(box)

        self._preview = QLabel("—")
        self._preview.setAlignment(Qt.AlignCenter)
        self._preview.setMinimumSize(explorer.THUMBNAIL_PX, explorer.THUMBNAIL_PX)
        self._preview.setToolTip(
            "Read from the smallest level of the image's own pyramid, so it costs a few "
            "hundred kilobytes rather than the 288 MB the full-resolution channel would.\n\n"
            "A Z stack is projected at maximum, which is what makes an organoid visible in "
            "one plane."
        )
        outer.addWidget(self._preview)

        side = QVBoxLayout()
        side.addWidget(QLabel("Channel"))
        self._channel_box = QComboBox()
        self._channel_box.setToolTip("Which channel the miniature shows.")
        self._channel_box.currentIndexChanged.connect(self._refresh_preview)
        side.addWidget(self._channel_box)
        self._preview_note = QLabel("—")
        self._preview_note.setWordWrap(True)
        side.addWidget(self._preview_note)
        side.addStretch(1)
        outer.addLayout(side, stretch=1)
        return box

    def _build_tables_box(self) -> QGroupBox:
        box = QGroupBox("Object tables for this image")
        outer = QVBoxLayout(box)

        self._tables_list = QListWidget()
        self._tables_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self._tables_list.setMaximumHeight(90)
        self._tables_list.setToolTip(
            "Tables found beside the plate that were written for the image selected above.\n\n"
            "A run names each table after the image it measured, so this is a match on the "
            "well and cycle rather than a guess."
        )
        self._tables_list.itemDoubleClicked.connect(lambda _item: self.open_table())
        self._tables_list.itemSelectionChanged.connect(self._update_enabled)
        outer.addWidget(self._tables_list)

        row = QHBoxLayout()
        self._open_table_button = QPushButton("Open in Measurement analysis")
        self._open_table_button.setToolTip(
            "Load the selected table in the Measurement analysis panel, where it colours "
            "the labels of this image by any column."
        )
        self._open_table_button.clicked.connect(self.open_table)
        row.addWidget(self._open_table_button)
        row.addStretch(1)
        self._tables_note = QLabel("—")
        self._tables_note.setWordWrap(True)
        row.addWidget(self._tables_note)
        outer.addLayout(row)
        return box

    def _build_open_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        self._with_labels = QCheckBox("with segmentations")
        self._with_labels.setChecked(True)
        self._with_labels.setToolTip(
            "Load the NGFF label sets stored inside each image alongside its channels, "
            "on the same grid and already aligned."
        )
        row.addWidget(self._with_labels)

        self._open_button = QPushButton("Open in the viewer")
        self._open_button.clicked.connect(self.open_selected)
        row.addWidget(self._open_button)
        row.addStretch(1)
        return row

    # -- the plate ------------------------------------------------------------

    def _on_failed(self, message: str) -> None:
        self._status.setText(message)
        self._update_enabled()

    def _on_surveyed(self, survey: batch.PlateSurvey) -> None:
        self._survey = survey
        self._refresh_table_index()
        self._picker.set_survey(survey)
        self._status.setText(
            f"{len(survey.jobs)} image(s). Choose wells and cycles, then open what you want."
        )
        self._announce(survey)
        self._update_enabled()

    def _refresh_table_index(self) -> None:
        """Index the object tables beside the plate, once per scan."""
        if self._survey is None:
            self._table_index = {}
            return
        try:
            folders = explorer.analysis_folders(self._survey.path)
            self._table_index = explorer.table_index(folders)
        except Exception:
            logger.exception("could not index the tables beside %s", self._survey.path)
            self._table_index = {}
        else:
            logger.info(
                "%d table(s) in %d folder(s) beside %s",
                sum(len(v) for v in self._table_index.values()),
                len(folders),
                self._survey.path.name,
            )

    def _announce(self, survey: batch.PlateSurvey) -> None:
        """Tell the other panels which plate is open.

        Broadcast rather than wired one panel to another: a panel opts in by
        having a ``plate_changed`` method, and one that does not care needs no
        change here when it is added.
        """
        try:
            self._app.plate_survey = survey
        except Exception:  # pragma: no cover - an app stub that refuses attributes
            logger.debug("could not record the survey on the app", exc_info=True)
        for identifier, panel in (getattr(self._app, "panels", {}) or {}).items():
            if panel is self:
                continue
            handler = getattr(panel, "plate_changed", None)
            if handler is None:
                continue
            try:
                handler(survey)
            except Exception:
                logger.exception("panel %s could not take the new plate", identifier)

    # -- the image list -------------------------------------------------------

    def _refresh_table(self) -> None:
        jobs = self._picker.selected_jobs()
        self._rows = list(jobs)
        self._table.setRowCount(len(jobs))
        for row, job in enumerate(jobs):
            described = explorer.describe_job(job, index=self._table_index)
            values = (
                described["well"],
                "" if described["acquisition"] == "" else f"cycle {described['acquisition']}",
                described["field"],
                f"{described['channels']}",
                described["shape"],
                described["segmentations"] or "—",
                described["tables"] or "—",
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                if column == 0:
                    item.setData(Qt.UserRole, row)
                self._table.setItem(row, column, item)
            self._table.item(row, IMAGE_COLUMNS.index("Channels")).setToolTip(
                described["channel_names"] or "—"
            )
            self._table.item(row, IMAGE_COLUMNS.index("Tables")).setToolTip(
                "\n".join(str(path) for path in described["table_paths"]) or "No table for this image"
            )

        segmented = sum(1 for job in jobs if explorer.label_sets(job))
        measured = sum(1 for job in jobs if explorer.tables_for(job, self._table_index))
        note = f"{len(jobs)} image(s)"
        if jobs:
            note += f", {segmented} already segmented, {measured} with an object table"
        self._table_note.setText(note + ".")
        if jobs and not self._table.selectedItems():
            self._table.selectRow(0)
        self._refresh_channels()
        self._refresh_tables()
        self._update_enabled()

    def _current_job(self) -> batch.ImageJob | None:
        rows = {index.row() for index in self._table.selectedIndexes()}
        if not rows:
            return None
        row = self._table.currentRow()
        chosen = row if row in rows else min(rows)
        return self._rows[chosen] if 0 <= chosen < len(self._rows) else None

    def selected_jobs(self) -> tuple[batch.ImageJob, ...]:
        """The images whose rows are selected; the whole list when none are."""
        rows = sorted({index.row() for index in self._table.selectedIndexes()})
        if not rows:
            return tuple(self._rows)
        return tuple(self._rows[row] for row in rows if 0 <= row < len(self._rows))

    def _refresh_channels(self) -> None:
        job = self._current_job()
        previous = self._channel_box.currentIndex()
        self._channel_box.blockSignals(True)
        self._channel_box.clear()
        if job is not None:
            for channel in job.channels:
                self._channel_box.addItem(channel.describe(), channel.index)
        self._channel_box.setCurrentIndex(
            previous if 0 <= previous < self._channel_box.count() else 0
        )
        self._channel_box.blockSignals(False)
        self._refresh_preview()

    def _on_row_selected(self) -> None:
        self._refresh_channels()
        self._refresh_tables()
        self._update_enabled()

    def _refresh_tables(self) -> None:
        """List the tables written for the image the miniature is showing."""
        self._tables_list.clear()
        job = self._current_job()
        if job is None:
            self._tables_note.setText("—")
            return

        own = explorer.tables_for(job, self._table_index)
        for path in own:
            item = QListWidgetItem(f"{path.parent.name} / {path.name}")
            item.setData(Qt.UserRole, path)
            item.setToolTip(str(path))
            self._tables_list.addItem(item)

        related = explorer.related_tables(job, self._table_index)
        for path in related:
            component = explorer.component_of(path)
            item = QListWidgetItem(
                f"{path.parent.name} / {path.name}" + OTHER_CYCLE.format(component=component)
            )
            item.setData(Qt.UserRole, path)
            item.setToolTip(
                f"{path}\n\nMeasured on {component}, not on this image. The same cells are "
                "imaged every 4i cycle, so the objects are the same ones — but the "
                "segmentation it refers to lives in the image it was run on."
            )
            self._tables_list.addItem(item)

        if own:
            self._tables_list.setCurrentRow(0)
            self._tables_note.setText(f"{len(own)} for this image.")
        elif related:
            self._tables_note.setText(
                f"None for this image; {len(related)} for another cycle of {job.well}."
            )
        else:
            self._tables_note.setText("None found beside the plate.")

    def open_table(self) -> None:
        """Hand the selected table to the Measurement analysis panel."""
        item = self._tables_list.currentItem()
        if item is None:
            self._status.setText("Choose a table first.")
            return
        path = item.data(Qt.UserRole)
        panel = (getattr(self._app, "panels", {}) or {}).get("measurement_analysis")
        opener = getattr(panel, "open_table", None)
        if opener is None:
            self._status.setText("The Measurement analysis panel is not available.")
            return
        try:
            opener(path)
        except Exception as exc:  # noqa: BLE001 - shown, not raised
            logger.exception("could not open %s", path)
            self._status.setText(f"Could not open that table: {exc}")
            return
        dock = (getattr(self._app, "docks", {}) or {}).get("measurement_analysis")
        if dock is not None:
            try:
                dock.setVisible(True)
                dock.raise_()
            except Exception:  # pragma: no cover - a dock that has never been shown
                logger.debug("could not raise the analysis dock", exc_info=True)
        self._status.setText(f"Opened {path.name} in Measurement analysis.")

    # -- the miniature --------------------------------------------------------

    def _refresh_preview(self) -> None:
        job = self._current_job()
        if job is None:
            self._preview.setText("—")
            self._preview.setPixmap(QPixmap())
            self._preview_note.setText("—")
            return
        index = self._channel_box.currentData()
        try:
            plane = explorer.thumbnail(job, None if index is None else int(index))
        except Exception as exc:  # noqa: BLE001 - a preview is never worth an error box
            logger.exception("%s: no miniature", job.component)
            self._preview.setText("—")
            self._preview.setPixmap(QPixmap())
            self._preview_note.setText(f"No miniature: {exc}")
            return

        grey = np.ascontiguousarray(explorer.stretch(plane))
        # Held on the widget: QImage wraps the buffer rather than copying it, and a
        # temporary would be freed while Qt was still drawing from it.
        self._thumbnail = grey
        height, width = grey.shape
        image = QImage(grey.data, width, height, width, QImage.Format_Grayscale8)
        self._preview.setPixmap(QPixmap.fromImage(image))

        level = job.level(max(0, len(job.levels) - 1))
        names = explorer.label_sets(job)
        self._preview_note.setText(
            f"{job.describe()}\n{width}x{height} px, from level "
            f"{len(job.levels) - 1} of {len(job.levels)} ({' x '.join(str(int(n)) for n in level.shape)})"
            + (f"\nSegmentations: {', '.join(names)}" if names else "\nNo segmentation yet")
        )

    # -- opening --------------------------------------------------------------

    def _update_enabled(self) -> None:
        self._open_button.setEnabled(bool(self._rows))
        self._open_table_button.setEnabled(self._tables_list.currentItem() is not None)

    def open_selected(self) -> None:
        """Build the layers for the selected images and add them to the viewer."""
        jobs = self.selected_jobs()
        if not jobs:
            self._status.setText("Nothing selected.")
            return
        if len(jobs) > MANY_IMAGES and not self._confirm(jobs):
            return

        self._status.setText(f"Opening {len(jobs)} image(s)…")
        self._app.toolbar.set_status(f"Opening {len(jobs)} image(s) from the plate…")
        try:
            specs, problems = explorer.specs_for(
                jobs,
                plate_path=Path(self._source.path()) if self._source.path() else None,
                with_labels=self._with_labels.isChecked(),
            )
        except Exception as exc:  # noqa: BLE001 - shown, not raised
            logger.exception("could not build the layers")
            self._status.setText(f"Could not open those images: {exc}")
            return

        added, failures = self._app.add_specs(specs)
        images = sum(1 for spec in specs if spec.layer_type != "labels")
        message = f"Opened {added} layer(s) from {len(jobs)} image(s) — {images} channel(s)"
        message += f", {added - images} label set(s)." if added > images else "."
        for problem in problems + [f"{error.name}: {error.message}" for error in failures]:
            logger.warning("explorer: %s", problem)
        if problems or failures:
            count = len(problems) + len(failures)
            message += f" {count} failed: {(problems or ['see the log'])[0]}"
        self._status.setText(message)

    def _confirm(self, jobs) -> bool:
        from qtpy.QtWidgets import QMessageBox

        channels = sum(len(job.channels) for job in jobs)
        answer = QMessageBox.question(
            self,
            "Open all of those?",
            f"{len(jobs)} images is about {channels} channel layers, plus a layer for every "
            "segmentation they carry.\n\nThe layer list will be long and the window slow to "
            "redraw. Open them anyway?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        return answer == QMessageBox.Yes

    # -- what other panels call ----------------------------------------------

    def tables_changed(self) -> None:
        """Re-index the tables beside the plate and redraw what shows them.

        Called by the Batch segmentation panel when a run finishes: the index is
        built once per scan, so a run that has just written twenty tables would
        otherwise leave the Tables column claiming there are none.
        """
        self._refresh_table_index()
        self._refresh_table()

    def plate_changed(self, survey: batch.PlateSurvey) -> None:
        """Follow a plate another panel scanned. Present for symmetry; nothing does yet."""
        if survey is not self._survey:
            self._source.set_path(survey.path)
            self._on_surveyed(survey)
