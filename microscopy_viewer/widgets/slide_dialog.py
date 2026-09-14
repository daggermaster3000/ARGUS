"""The dialog behind the toolbar's *Export Slide* button.

Two tables: which datasets go on the slide, and what each channel is called.
The channel labels are the reason this is a dialog rather than a one-click
export — a file says "Alexa 568", a figure needs to say "anti-CD31", and the
person exporting is the only one who knows which is which.

*Add folder* is batch mode. It reads a whole directory straight into slide rows
without adding anything to the viewer, and the deck is split across as many
slides as it takes to keep the panels big enough to see.
"""

from __future__ import annotations

from pathlib import Path

from qtpy.QtCore import Qt
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from .. import slides
from ..loaders import FILE_DIALOG_FILTER
from ..utils import get_logger

logger = get_logger("slide_dialog")


def _format_limit(value: float) -> str:
    """A limit as a box wants to show it: no trailing ``.0`` on whole numbers."""
    return f"{value:g}"


def _parse_limits(low: str, high: str) -> tuple[float, float] | None:
    """``(low, high)`` from two typed boxes, or ``None`` if they are not usable.

    Anything unparseable is treated as an empty box rather than an error: the
    fallback is the range the channel is displayed at, which is a sane picture,
    and a modal complaint about a half-typed number would be worse than that.
    """
    try:
        pair = (float(low.strip()), float(high.strip()))
    except (AttributeError, TypeError, ValueError):
        return None
    return pair if pair[1] > pair[0] else None


class SlideExportDialog(QDialog):
    """Choose samples, name the stainings, and write the PowerPoint deck."""

    def __init__(self, viewer, last_directory: Path | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export PowerPoint slide")
        self.setMinimumWidth(680)

        self._viewer = viewer
        self._last_directory = last_directory
        self.written: Path | None = None
        self._cancelled = False
        self._exporting = False

        collected = slides.collect_samples(viewer) if viewer is not None else []
        # Overview fields are pulled out before anything else sees them: twenty-five
        # low-magnification views of the same slide are one picture, not
        # twenty-five rows.
        self._samples, self._overview = slides.split_overview(collected)
        self._from_viewer = len(self._samples)
        self._included: list[bool] = [True] * len(self._samples)
        self._names: list[str] = [sample.name for sample in self._samples]
        self._columns: list = []
        # Channel labels and merge membership survive a folder being added, so
        # typing "anti-CD31" is not undone by loading more files.
        self._labels: dict[str, str] = {}
        self._merge: dict[str, bool] = {}
        # Typed contrast limits, one pair per channel column. Only consulted in
        # manual mode; kept regardless, so switching modes to look at something
        # else and back does not lose what was typed.
        self._limits: dict[str, tuple[float, float]] = {}

        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        header = QHBoxLayout()
        header.addWidget(QLabel("<b>Samples</b> — one table row each"))
        header.addStretch(1)
        folder_button = QPushButton("Add folder…")
        folder_button.setToolTip(
            "Batch mode: read every supported file in a folder without opening it in the viewer."
        )
        folder_button.clicked.connect(self.add_folder)
        header.addWidget(folder_button)
        files_button = QPushButton("Add files…")
        files_button.clicked.connect(self.add_files)
        header.addWidget(files_button)
        layout.addLayout(header)

        self._sample_table = QTableWidget(0, 3, self)
        self._sample_table.setHorizontalHeaderLabels(["Include", "Row label", "Channels"])
        self._sample_table.verticalHeader().setVisible(False)
        # napari's stylesheet renders alternating rows as blank stripes.
        self._sample_table.setAlternatingRowColors(False)
        self._sample_table.setSelectionMode(QAbstractItemView.NoSelection)
        self._sample_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self._sample_table.setMinimumHeight(150)
        layout.addWidget(self._sample_table, 1)

        layout.addWidget(
            QLabel(
                "<b>Channels</b> — the label becomes the coloured column heading. "
                "Rename it to the staining or antibody."
            )
        )
        self._channel_table = QTableWidget(0, 5, self)
        self._channel_table.setHorizontalHeaderLabels(
            ["Colour", "Label (staining / antibody)", "In merge", "Min", "Max"]
        )
        self._channel_table.verticalHeader().setVisible(False)
        self._channel_table.setAlternatingRowColors(False)
        self._channel_table.setSelectionMode(QAbstractItemView.NoSelection)
        self._channel_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self._channel_table.setMaximumHeight(160)
        layout.addWidget(self._channel_table)

        layout.addLayout(self._build_options())

        self._status = QLabel("")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        self._buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        self._save = self._buttons.button(QDialogButtonBox.Save)
        self._save.setText("Export…")
        self._cancel = self._buttons.button(QDialogButtonBox.Cancel)
        self._buttons.accepted.connect(self.export)
        self._buttons.rejected.connect(self._on_cancel)
        layout.addWidget(self._buttons)

        self._refresh_tables()

    # -- tables ---------------------------------------------------------------

    def _refresh_tables(self) -> None:
        """Rebuild both tables from ``self._samples``, keeping every edit."""
        self._harvest()
        self._columns = slides.channel_columns(self._samples)
        for key, label, _color in self._columns:
            self._labels.setdefault(key, label)
            self._merge.setdefault(key, self._merge_default(key))
        self._fill_sample_table()
        self._fill_channel_table()
        self._update_summary()

    def _harvest(self) -> None:
        """Copy what is currently typed or ticked out of the widgets."""
        for row in range(min(self._sample_table.rowCount(), len(self._samples))):
            include = self._sample_table.item(row, 0)
            name = self._sample_table.item(row, 1)
            if include is not None:
                self._included[row] = include.checkState() == Qt.Checked
            if name is not None and name.text().strip():
                self._names[row] = name.text().strip()

        for row in range(min(self._channel_table.rowCount(), len(self._columns))):
            key = self._columns[row][0]
            label = self._channel_table.item(row, 1)
            tick = self._channel_table.item(row, 2)
            if label is not None and label.text().strip():
                self._labels[key] = label.text().strip()
            if tick is not None:
                self._merge[key] = tick.checkState() == Qt.Checked

            low = self._channel_table.item(row, 3)
            high = self._channel_table.item(row, 4)
            pair = _parse_limits(
                low.text() if low is not None else "",
                high.text() if high is not None else "",
            )
            if pair is None:
                self._limits.pop(key, None)
            else:
                self._limits[key] = pair

    def _fill_sample_table(self) -> None:
        table = self._sample_table
        table.setRowCount(len(self._samples))
        for row, sample in enumerate(self._samples):
            tick = QTableWidgetItem()
            tick.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            tick.setCheckState(Qt.Checked if self._included[row] else Qt.Unchecked)
            table.setItem(row, 0, tick)

            name = QTableWidgetItem(self._names[row])
            name.setToolTip(sample.source or sample.name)
            table.setItem(row, 1, name)

            summary = QTableWidgetItem(
                ", ".join(self._labels.get(c.key, c.label) for c in sample.channels)
            )
            summary.setFlags(Qt.ItemIsEnabled)
            table.setItem(row, 2, summary)
        table.resizeColumnToContents(0)

    def _fill_channel_table(self) -> None:
        table = self._channel_table
        # The limit cells are only editable in manual mode: greyed out they say
        # the numbers exist and what the mode does, which an empty column does not.
        combo = getattr(self, "_contrast", None)
        manual = combo is not None and combo.currentText() == slides.CONTRAST_MANUAL
        table.setRowCount(len(self._columns))
        for row, (key, label, color) in enumerate(self._columns):
            swatch = QTableWidgetItem("")
            swatch.setFlags(Qt.ItemIsEnabled)
            swatch.setBackground(QColor(*(int(round(c * 255)) for c in color)))
            table.setItem(row, 0, swatch)

            text = QTableWidgetItem(self._labels.get(key, label))
            # Shown in the colour it will be printed in, so a mismatch is obvious.
            text.setForeground(QColor(*slides.text_color(color)))
            table.setItem(row, 1, text)

            tick = QTableWidgetItem()
            tick.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            tick.setCheckState(Qt.Checked if self._merge.get(key, True) else Qt.Unchecked)
            table.setItem(row, 2, tick)

            pair = self._limits.get(key)
            for column, value in ((3, pair[0] if pair else None), (4, pair[1] if pair else None)):
                cell = QTableWidgetItem("" if value is None else _format_limit(value))
                cell.setFlags(
                    Qt.ItemIsEnabled | Qt.ItemIsEditable | Qt.ItemIsSelectable
                    if manual
                    else Qt.NoItemFlags
                )
                cell.setToolTip(
                    "Intensity mapped to black (Min) and to full colour (Max).\n"
                    "Applied to this channel on every sample in the deck.\n"
                    "Leave empty to keep the range the channel is displayed at."
                )
                table.setItem(row, column, cell)
        table.resizeColumnToContents(0)
        table.resizeColumnToContents(2)
        table.resizeColumnToContents(3)
        table.resizeColumnToContents(4)

    def _merge_default(self, key: str) -> bool:
        """Whether this channel starts out included in the merge."""
        for sample in self._samples:
            for channel in sample.channels:
                if channel.key == key:
                    return channel.in_merge
        return True

    def _build_options(self) -> QFormLayout:
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)

        self._title = QLineEdit()
        self._title.setPlaceholderText("Slide title (optional)")
        form.addRow("Title", self._title)

        self._mode = QComboBox()
        self._mode.addItems(list(slides.PLANE_MODES))
        self._mode.setToolTip("Z-stacks are flattened this way; 2D images are unaffected.")
        form.addRow("Z projection", self._mode)

        self._contrast = QComboBox()
        self._contrast.addItems(list(slides.CONTRAST_MODES))
        # Files read in batch were never on screen, so "as displayed" would mean
        # whatever range the file happens to store — often not a usable one.
        self._contrast.setCurrentText(
            slides.CONTRAST_AS_DISPLAYED if self._from_viewer else slides.CONTRAST_AUTO
        )
        self._contrast.setToolTip(
            "As displayed: the layer's current contrast limits.\n"
            "Auto per image: the 0.5–99.5 percentile of each projected image.\n"
            "Manual limits: the Min and Max typed into the channel table above,\n"
            "one pair per channel, used for every sample in the deck."
        )
        self._contrast.currentTextChanged.connect(self._on_contrast_changed)
        form.addRow("Contrast", self._contrast)

        row = QHBoxLayout()
        self._rows_per_slide = QSpinBox()
        self._rows_per_slide.setRange(1, slides.MAX_ROWS_PER_SLIDE)
        self._rows_per_slide.setValue(slides.DEFAULT_ROWS_PER_SLIDE)
        self._rows_per_slide.setToolTip(
            "Samples per slide. More rows means smaller panels; the rest go on further slides."
        )
        self._rows_per_slide.valueChanged.connect(self._update_summary)
        row.addWidget(self._rows_per_slide)

        self._resolution = QSpinBox()
        self._resolution.setRange(200, 4000)
        self._resolution.setSingleStep(100)
        self._resolution.setValue(slides.DEFAULT_MAX_PIXELS)
        self._resolution.setSuffix(" px")
        self._resolution.setToolTip(
            "Longest edge of each embedded image. Larger means a bigger file and a slower read."
        )
        row.addWidget(QLabel("Image size"))
        row.addWidget(self._resolution)

        self._scale_bar = QCheckBox("Scale bar")
        self._scale_bar.setChecked(True)
        row.addWidget(self._scale_bar)
        row.addStretch(1)
        form.addRow("Rows per slide", row)

        self._overview_slide = QCheckBox("Stitch the overview and mark where each sample was imaged")
        self._overview_slide.setToolTip(
            "Overview fields (…_F0000, _F0001, …) are stitched from their stage coordinates\n"
            "into one locator slide at the front of the deck. Each sample gets a numbered box\n"
            "at the position it was acquired from; the numbers are the deck's row order."
        )
        self._overview_slide.setChecked(True)
        self._overview_slide.stateChanged.connect(self._update_summary)
        form.addRow("Overview", self._overview_slide)

        self._zoom_slides = QCheckBox("Show each closeup beside the field it was taken from")
        self._zoom_slides.setToolTip(
            "A dataset acquired inside the field of another one — a 40x taken from a 20x —\n"
            "gets its own slide: that field on the left with a box around the part the\n"
            "closeup covers, and the closeup's channels beside it. The pairing comes from\n"
            "the stage coordinates. A sample nothing else contains is shown on the overview."
        )
        self._zoom_slides.setChecked(True)
        self._zoom_slides.stateChanged.connect(self._update_summary)
        form.addRow("Closeups", self._zoom_slides)

        return form

    def _update_summary(self) -> None:
        if self._exporting:
            return
        count = sum(1 for included in self._included if included)
        if not count:
            self._status.setText(
                "Nothing selected — tick a sample, or use <b>Add folder…</b> for batch mode."
            )
            self._save.setEnabled(False)
            return
        per_slide = max(1, int(self._rows_per_slide.value()))
        deck = -(-count // per_slide)  # ceiling division
        note = ""
        if self._overview:
            self._overview_slide.setEnabled(True)
            if self._overview_slide.isChecked():
                deck += 1
                note = f" Overview: {len(self._overview)} field(s) stitched onto the first slide."
            else:
                note = f" {len(self._overview)} overview field(s) found but not used."
        else:
            # Nothing to stitch: leave the tick visible but inert rather than
            # hiding it, so its absence is not mistaken for the feature missing.
            self._overview_slide.setEnabled(False)

        # Counted from the stage coordinates, which are already in hand: nothing
        # is read or stitched to say how long the deck will be.
        if self._zoom_slides.isChecked():
            chosen = [sample for sample, ok in zip(self._samples, self._included) if ok]
            tiles = self._overview if self._overview_slide.isChecked() else ()
            pairs = slides.find_closeups(chosen, tiles=tiles)
            if pairs:
                deck += len(pairs)
                note += f" {len(pairs)} closeup slide(s)."
        self._status.setText(
            f"{count} sample(s) x {len(self._columns)} channel(s) "
            f"→ {deck} slide(s) at {per_slide} per slide.{note}"
        )
        self._save.setEnabled(True)

    # -- adding samples -------------------------------------------------------

    def add_folder(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self, "Add every supported file in a folder", str(self._last_directory or Path.home())
        )
        if directory:
            self._last_directory = Path(directory)
            self._load([directory])

    def add_files(self) -> None:
        paths, _selected = QFileDialog.getOpenFileNames(
            self, "Add datasets", str(self._last_directory or Path.home()), FILE_DIALOG_FILTER
        )
        if paths:
            self._last_directory = Path(paths[0]).parent
            self._load(paths)

    def _load(self, paths) -> None:
        """Read datasets into slide rows, skipping any that are already listed."""
        self._harvest()
        self._status.setText("Reading…")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        QApplication.instance().processEvents()
        try:
            found, errors = slides.samples_from_paths(paths)
        except Exception as exc:
            logger.exception("could not read the batch")
            QMessageBox.critical(self, "Microscopy Viewer", f"Could not read those files:\n{exc}")
            return
        finally:
            QApplication.restoreOverrideCursor()

        found, tiles = slides.split_overview(found)
        known_tiles = {tile.source for tile in self._overview if tile.source}
        self._overview.extend(tile for tile in tiles if tile.source not in known_tiles)

        known = {sample.source for sample in self._samples if sample.source}
        added = 0
        for sample in found:
            if sample.source and sample.source in known:
                continue
            known.add(sample.source)
            self._samples.append(sample)
            self._included.append(True)
            self._names.append(sample.name)
            added += 1

        self._refresh_tables()
        note = f"Added {added} sample(s)."
        if tiles:
            note += f" {len(tiles)} overview field(s) kept aside for the locator slide."
        if added < len(found):
            note += f" {len(found) - added} were already listed."
        if errors:
            note += f" {len(errors)} file(s) could not be read."
            logger.warning("batch load skipped: %s", "; ".join(errors[:10]))
        self._status.setText(f"{note} {self._status.text()}" if added else note)
        if added:
            self._update_summary()

    # -- reading the tables ---------------------------------------------------

    def selected_samples(self) -> list:
        self._harvest()
        chosen = []
        for row, sample in enumerate(self._samples):
            if not self._included[row]:
                continue
            sample.name = self._names[row]
            chosen.append(sample)
        return chosen

    def _on_contrast_changed(self, mode: str) -> None:
        """Enable the limit cells for manual mode, seeding them the first time.

        Seeded from what each channel is displayed at, because that is a range
        somebody has already looked at — a pair of empty boxes is a worse start
        than a pair of numbers to nudge.
        """
        self._harvest()
        if mode == slides.CONTRAST_MANUAL and not self._limits:
            self._limits = slides.suggested_limits(self._samples)
        self._fill_channel_table()
        self._update_summary()

    def labels(self) -> dict[str, str]:
        self._harvest()
        return dict(self._labels)

    def contrast_limits(self) -> dict[str, tuple[float, float]]:
        """The typed limits, or nothing at all outside manual mode."""
        self._harvest()
        if self._contrast.currentText() != slides.CONTRAST_MANUAL:
            return {}
        return dict(self._limits)

    def merge_keys(self) -> list[str]:
        self._harvest()
        return [key for key, included in self._merge.items() if included]

    # -- the export -----------------------------------------------------------

    def export(self) -> None:
        chosen = self.selected_samples()
        if not chosen:
            QMessageBox.information(self, "Microscopy Viewer", "Tick at least one sample.")
            return

        # The headings come straight from the channel labels, so renaming here is
        # all that is needed for the coloured column titles to follow.
        slides.apply_labels(chosen, self.labels())
        slides.apply_merge_selection(chosen, self.merge_keys())
        # Cleared rather than skipped outside manual mode: the samples are the
        # dialog's own objects and may carry limits from an earlier export.
        slides.apply_contrast_limits(chosen, self.contrast_limits())

        directory = self._last_directory or Path.home()
        suggested = str(Path(directory) / f"{slides.default_stem()}.pptx")
        path, _selected = QFileDialog.getSaveFileName(
            self, "Export slide", suggested, slides.PRESENTATION_FILTER
        )
        if not path:
            return

        # Rendering re-reads every channel, which takes tens of seconds per sample
        # when the data is on a network share. Keep the dialog painting, say where
        # it has got to, and leave Cancel live so a whole folder can be stopped.
        self._exporting = True
        self._cancelled = False
        self._save.setEnabled(False)
        self._cancel.setText("Stop")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self.written = slides.export_slide(
                chosen,
                path,
                title=self._title.text().strip(),
                mode=self._mode.currentText(),
                max_pixels=int(self._resolution.value()),
                scale_bar=self._scale_bar.isChecked(),
                contrast=self._contrast.currentText(),
                rows_per_slide=int(self._rows_per_slide.value()),
                overview_tiles=self._overview if self._overview_slide.isChecked() else (),
                zoom_slides=self._zoom_slides.isChecked(),
                progress=self._on_progress,
                should_cancel=lambda: self._cancelled,
            )
        except slides.ExportCancelled as exc:
            logger.info("slide export cancelled: %s", exc)
            self._status.setText(f"Stopped — nothing was written ({exc}).")
            return
        except ImportError:
            logger.exception("python-pptx is missing")
            QMessageBox.critical(
                self,
                "Microscopy Viewer",
                "Writing PowerPoint slides needs the python-pptx package:\n\n"
                "    python -m pip install python-pptx",
            )
            return
        except Exception as exc:
            logger.exception("slide export failed")
            self._status.setText(f"Export failed: {exc}")
            QMessageBox.critical(self, "Microscopy Viewer", f"Could not write the slide:\n{exc}")
            return
        finally:
            QApplication.restoreOverrideCursor()
            self._exporting = False
            self._save.setEnabled(True)
            self._cancel.setText("Cancel")

        self.accept()

    def _on_cancel(self) -> None:
        """Cancel stops the export while one is running, and closes otherwise."""
        if self._exporting:
            self._cancelled = True
            self._status.setText("Stopping after the current sample…")
            return
        self.reject()

    def _on_progress(self, index: int, total: int, name: str) -> None:
        # A negative index is work that is not one of the numbered samples — the
        # overview, or a closeup slide. Those pass a phrase rather than a name.
        if index < 0:
            self._status.setText(f"{name}…")
        else:
            self._status.setText(f"Rendering {index + 1} of {total}: {name}…")
        application = QApplication.instance()
        if application is not None:
            application.processEvents()
