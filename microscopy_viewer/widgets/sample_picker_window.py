"""A window to pick samples off a table of categories, with their thumbnails.

Opened from Experiment setup (*Pick from table…*). Load a prescreen or QC
table (CSV or Excel, a row per sample), and optionally a file list saying what
each file is (genotype, experiment, slide — ``Image_File_List``), joined on by
file name; every column that sorts samples into a few groups is listed with its
values, each with two ticks:

* **Keep** — samples with this value may be picked. Untick *Pass: no* to leave
  failing samples out.
* **Draw** — this group gets *n* random picks. Tick *Ripped: yes*, *Folded:
  yes* and each *Shape* to get ten of each to annotate.

The picks show on the right as the Experiment setup thumbnails. They can be
selected in the Experiment setup grid (so its other buttons, batches and
panels work on them), opened one after the other for annotating, and saved as
a list with the seed that drew them.

The choosing is :mod:`microscopy_viewer.sample_picker`'s; this is its view.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Callable

import pandas as pd
from qtpy.QtCore import QSize, Qt, QTimer
from qtpy.QtGui import QIcon
from qtpy.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .. import sample_picker as sp
from ..utils import get_logger

logger = get_logger("sample_picker_window")

COL_NAME, COL_COUNT, COL_KEEP, COL_DRAW = range(4)
ICON_SIZE = 112
#: At most this many picks get a thumbnail drawn here rather than waiting for
#: the scan: enough for a draw of ten per group, not a second scan of the folder.
OWN_THUMBNAILS = 60
ROW_ROLE = Qt.UserRole + 1
VALUE_ROLE = Qt.UserRole + 2


class SamplePickerWindow(QWidget):
    """Keep some categories, draw *n* of others, see and use the picks."""

    def __init__(
        self,
        paths: Callable[[], list],
        icon_of: Callable[[Path], QIcon | None],
        select: Callable[[list], None],
        open_path: Callable[[Path], None],
        folder: Callable[[], Path | None],
        parent: QWidget | None = None,
    ):
        super().__init__(parent, Qt.Window)
        self.setWindowTitle("Pick samples from a table")
        self.resize(1100, 700)
        self._paths, self._icon_of, self._select = paths, icon_of, select
        self._open_path, self._folder = open_path, folder
        self._table: pd.DataFrame | None = None
        #: The table as loaded, and the file list joined onto it (either may be missing).
        self._raw: pd.DataFrame | None = None
        self._files: pd.DataFrame | None = None
        self._files_note = ""
        self._matched = pd.Series(dtype=object)
        self._columns: list[str] = []
        self._texts: dict[str, pd.Series] = {}
        #: Thumbnails this window drew, by path, and the grid tiles of each path.
        self._icons: dict[str, QIcon] = {}
        self._items: dict[str, list] = {}
        #: Bumped on every refill, so a thumbnail run for older picks stops.
        self._generation = 0
        self._picks = sp.Picks()
        self._opened: set = set()
        self._filling = False
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(0)
        self._timer.timeout.connect(self.update_picks)

        layout = QVBoxLayout(self)
        row = QHBoxLayout()
        row.addWidget(QLabel("Table"))
        self._path_edit = QLineEdit()
        self._path_edit.setReadOnly(True)
        self._path_edit.setPlaceholderText("A CSV or Excel file with a row per sample")
        row.addWidget(self._path_edit, stretch=1)
        load = QPushButton("Load table…")
        load.clicked.connect(self.choose_table)
        row.addWidget(load)
        layout.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(QLabel("File list"))
        self._files_edit = QLineEdit()
        self._files_edit.setReadOnly(True)
        self._files_edit.setPlaceholderText(
            "Optional: what each file is — genotype, experiment, slide (e.g. Image_File_List)")
        row.addWidget(self._files_edit, stretch=1)
        add = QPushButton("Add file list…")
        add.setToolTip(
            "A CSV or Excel list of the files with their genotype, experiment, slide …\n"
            "Its columns are added to the table's rows by file name, so you can keep or draw "
            "by genotype. Loaded alone, it is the table.")
        add.clicked.connect(self.choose_file_list)
        row.addWidget(add)
        remove = QPushButton("Remove")
        remove.setToolTip("Use the table without the file list.")
        remove.clicked.connect(self.clear_file_list)
        row.addWidget(remove)
        layout.addLayout(row)
        self._match_label = QLabel("")
        self._match_label.setWordWrap(True)
        layout.addWidget(self._match_label)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_left())
        splitter.addWidget(self._build_right())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([380, 720])
        layout.addWidget(splitter, stretch=1)

    # -- construction ---------------------------------------------------------

    def _build_left(self) -> QWidget:
        box = QWidget()
        outer = QVBoxLayout(box)
        outer.setContentsMargins(0, 0, 0, 0)
        self._tree = QTreeWidget()
        self._tree.setHeaderLabels(["Category", "Kept", "Keep", "Draw"])
        self._tree.headerItem().setToolTip(COL_COUNT, "Kept samples with this value, of all in the table.")
        self._tree.headerItem().setToolTip(COL_KEEP, "Samples with this value may be picked.")
        self._tree.headerItem().setToolTip(COL_DRAW, "Draw n picks from this group.")
        self._tree.setColumnWidth(COL_NAME, 170)
        self._tree.setColumnWidth(COL_COUNT, 70)
        self._tree.setColumnWidth(COL_KEEP, 44)
        self._tree.itemChanged.connect(self._on_item_changed)
        self._tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self._tree.customContextMenuRequested.connect(self._tree_menu)
        self._tree.setToolTip("Right-click a value to keep only it, or all of its column again.")
        outer.addWidget(self._tree, stretch=1)

        row = QHBoxLayout()
        row.addWidget(QLabel("Draw"))
        self._n = QSpinBox()
        self._n.setRange(0, 10000)
        self._n.setValue(10)
        # 0 is shown as "all": every kept sample of each ticked group, no draw.
        self._n.setSpecialValueText("all")
        self._n.setToolTip(
            "How many samples to draw from each group ticked under Draw. Set it to "
            "“all” (below 1) to take every kept sample of those groups instead — e.g. tick "
            "Draw on Pass: yes for every passing sample.")
        self._n.valueChanged.connect(lambda _v: self._timer.start())
        row.addWidget(self._n)
        row.addWidget(QLabel("per group, seed"))
        self._seed = QSpinBox()
        self._seed.setRange(0, 999999)
        self._seed.valueChanged.connect(lambda _v: self._timer.start())
        row.addWidget(self._seed)
        again = QPushButton("New draw")
        again.setToolTip("Draw again with the next seed.")
        again.clicked.connect(lambda: self._seed.setValue(self._seed.value() + 1))
        row.addWidget(again)
        outer.addLayout(row)
        self._share = QCheckBox("A pick counts for every group it is in")
        self._share.setToolTip(
            "Off: every ticked group gets its own n samples — 10 per group over four "
            "groups is 40 samples.\nOn: a sample drawn for one group also counts for the "
            "others it belongs to (a ripped, folded sample fills both), so fewer samples "
            "cover every group at least n times."
        )
        self._share.toggled.connect(lambda _on: self._timer.start())
        outer.addWidget(self._share)
        clear = QPushButton("Untick all draws")
        clear.clicked.connect(self._clear_draws)
        outer.addWidget(clear)
        self._summary = QLabel("Load a table to start.")
        self._summary.setWordWrap(True)
        outer.addWidget(self._summary)
        return box

    def _build_right(self) -> QWidget:
        box = QWidget()
        outer = QVBoxLayout(box)
        outer.setContentsMargins(0, 0, 0, 0)
        self._grid = QListWidget()
        self._grid.setViewMode(QListWidget.IconMode)
        self._grid.setIconSize(QSize(ICON_SIZE, ICON_SIZE))
        self._grid.setGridSize(QSize(ICON_SIZE + 40, ICON_SIZE + 58))
        self._grid.setResizeMode(QListWidget.Adjust)
        self._grid.setMovement(QListWidget.Static)
        self._grid.setWordWrap(True)
        # Sample names share a long prefix; the part that tells them apart is the end.
        self._grid.setTextElideMode(Qt.ElideMiddle)
        self._grid.itemDoubleClicked.connect(self._open_item)
        outer.addWidget(self._grid, stretch=1)

        row = QHBoxLayout()
        select = QPushButton("Select in Experiment setup")
        select.setToolTip("Select the picks in the Experiment setup grid, so batches and panels run on them.")
        select.clicked.connect(self.select_picks)
        row.addWidget(select)
        self._next = QPushButton("Open next")
        self._next.setToolTip(
            "Open the next pick not opened yet, in place of the sample on screen "
            "(double-click a thumbnail to open that one)."
        )
        self._next.clicked.connect(self.open_next)
        row.addWidget(self._next)
        save = QPushButton("Save list…")
        save.setToolTip("Write the picks, what they were drawn for and the seed to a CSV.")
        # Not connected directly: clicked passes its checked state, which would land in *path*.
        save.clicked.connect(lambda: self.save_list())
        row.addWidget(save)
        row.addStretch(1)
        outer.addLayout(row)
        return box

    # -- the table ------------------------------------------------------------

    def choose_table(self) -> None:
        folder = self._folder()
        start = str(folder) if folder else str(Path.home())
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose the table", start, "Tables (*.csv *.tsv *.txt *.xlsx *.xls);;All files (*)")
        if path:
            self.load_table(path)

    def load_table(self, path) -> bool:
        try:
            table = sp.read_table(path)
        except Exception as exc:
            logger.exception("could not read %s", path)
            self._match_label.setText(f"Could not read {Path(path).name}: {exc}")
            return False
        self._path_edit.setText(str(path))
        self._raw = table
        self._use_tables()
        return True

    def choose_file_list(self) -> None:
        folder = self._folder()
        start = str(folder) if folder else str(Path.home())
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose the file list", start, "Tables (*.csv *.tsv *.txt *.xlsx *.xls);;All files (*)")
        if path:
            self.load_file_list(path)

    def load_file_list(self, path) -> bool:
        """Join a list of the files with their genotype, experiment, slide … onto the table."""
        try:
            files = sp.read_table(path)
        except Exception as exc:
            logger.exception("could not read %s", path)
            self._match_label.setText(f"Could not read {Path(path).name}: {exc}")
            return False
        self._files_edit.setText(str(path))
        self._files = files
        self._use_tables()
        return True

    def clear_file_list(self) -> None:
        self._files = None
        self._files_edit.clear()
        self._use_tables()

    def _use_tables(self) -> None:
        """The table to pick from: the loaded one with the file list joined on, or either alone."""
        self._files_note = ""
        if self._raw is not None and self._files is not None:
            table, found = sp.join_file_list(self._raw, self._files)
            self._files_note = f" {found} of {len(table)} rows found in the file list."
            if not found:
                self._files_note += " Its file names match none of the table's — is it the same experiment?"
        else:
            table = self._raw if self._raw is not None else self._files
        self._table = table
        if table is None:
            self._columns, self._texts = [], {}
            self._tree.clear()
            self._grid.clear()
            self._match_label.setText("")
            self._summary.setText("Load a table to start.")
            return
        self._columns = sp.category_columns(table)
        # The category values as text, once: every tick counts over them.
        self._texts = {column: table[column].map(sp._text) for column in self._columns}
        self._opened.clear()
        self.rematch()
        self._fill_tree()
        self.update_picks()

    def rematch(self) -> None:
        """Match the rows to the files of the experiment folder again (after a scan)."""
        if self._table is None:
            return
        self._matched = sp.match_paths(self._table, self._paths())
        found = int(self._matched.notna().sum())
        missing = len(self._table) - found
        text = f"{len(self._table)} rows, {found} matched to samples of the experiment folder."
        if missing:
            text += (f" {missing} matched no file and are left out — choose the folder they "
                     "are in in Experiment setup, or give full paths in a “path” column.")
        self._match_label.setText(text + self._files_note)

    def _fill_tree(self) -> None:
        self._filling = True
        try:
            self._tree.clear()
            for column in self._columns:
                parent = QTreeWidgetItem([column])
                font = parent.font(COL_NAME)
                font.setBold(True)
                parent.setFont(COL_NAME, font)
                parent.setFlags(Qt.ItemIsEnabled)
                self._tree.addTopLevelItem(parent)
                for value, _n in sp.values_of(self._table, column):
                    child = QTreeWidgetItem([value, ""])
                    child.setData(COL_NAME, ROW_ROLE, column)
                    child.setData(COL_NAME, VALUE_ROLE, value)
                    child.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
                    child.setCheckState(COL_KEEP, Qt.Checked)
                    child.setCheckState(COL_DRAW, Qt.Unchecked)
                    parent.addChild(child)
                parent.setExpanded(True)
        finally:
            self._filling = False

    def _children(self):
        for i in range(self._tree.topLevelItemCount()):
            parent = self._tree.topLevelItem(i)
            for j in range(parent.childCount()):
                yield parent.child(j)

    def _on_item_changed(self, _item, _column) -> None:
        if not self._filling:
            self._timer.start()

    def _tree_menu(self, point) -> None:
        """Right-click on a value: keep only it, or every value of its column."""
        from qtpy.QtWidgets import QMenu

        item = self._tree.itemAt(point)
        if item is None or item.data(COL_NAME, ROW_ROLE) is None:
            return
        column, value = item.data(COL_NAME, ROW_ROLE), item.data(COL_NAME, VALUE_ROLE)
        menu = QMenu(self._tree)
        only = menu.addAction(f"Keep only {column}: {value}")
        every = menu.addAction(f"Keep every {column}")
        chosen = menu.exec_(self._tree.viewport().mapToGlobal(point))
        if chosen is only:
            self.keep_only(column, [value])
        elif chosen is every:
            self.keep_only(column, None)

    def keep_only(self, column: str, values) -> None:
        """Keep only *values* of *column* (``None``: all of them); other columns are left as they are."""
        self._filling = True
        try:
            for child in self._children():
                if child.data(COL_NAME, ROW_ROLE) == column:
                    keep = values is None or child.data(COL_NAME, VALUE_ROLE) in values
                    child.setCheckState(COL_KEEP, Qt.Checked if keep else Qt.Unchecked)
        finally:
            self._filling = False
        self.update_picks()

    def _clear_draws(self) -> None:
        self._filling = True
        try:
            for child in self._children():
                child.setCheckState(COL_DRAW, Qt.Unchecked)
        finally:
            self._filling = False
        self._timer.start()

    # -- the ticks ------------------------------------------------------------

    def keep(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for child in self._children():
            column, value = child.data(COL_NAME, ROW_ROLE), child.data(COL_NAME, VALUE_ROLE)
            out.setdefault(column, [])
            if child.checkState(COL_KEEP) == Qt.Checked:
                out[column].append(value)
        return out

    def groups(self) -> list[tuple[str, str]]:
        return [(child.data(COL_NAME, ROW_ROLE), child.data(COL_NAME, VALUE_ROLE))
                for child in self._children() if child.checkState(COL_DRAW) == Qt.Checked]

    def set_ticks(self, keep: dict | None = None, draw=()) -> None:
        """Tick by hand: *keep* ``{column: [values]}`` (others in that column unticked),
        *draw* ``[(column, value)]``."""
        keep = keep or {}
        draw = set(draw)
        self._filling = True
        try:
            for child in self._children():
                column, value = child.data(COL_NAME, ROW_ROLE), child.data(COL_NAME, VALUE_ROLE)
                if column in keep:
                    child.setCheckState(COL_KEEP, Qt.Checked if value in keep[column] else Qt.Unchecked)
                child.setCheckState(COL_DRAW, Qt.Checked if (column, value) in draw else Qt.Unchecked)
        finally:
            self._filling = False
        self.update_picks()

    def eligible(self) -> pd.Series:
        return sp.keep_mask(self._table, self.keep()) & self._matched.notna()

    # -- the picks ------------------------------------------------------------

    def update_picks(self) -> None:
        if self._table is None:
            return
        keep = self.keep()
        matched = self._matched.notna().to_numpy()
        texts = self._texts
        ticked = {c: texts[c].isin(set(v)).to_numpy() for c, v in keep.items()}
        self._filling = True
        try:
            for child in self._children():
                column, value = child.data(COL_NAME, ROW_ROLE), child.data(COL_NAME, VALUE_ROLE)
                # Kept by every other column, so the number is what this tick adds.
                mask = matched.copy()
                for other, rows in ticked.items():
                    if other != column:
                        mask &= rows
                values = (texts[column] == value).to_numpy()
                child.setText(COL_COUNT, f"{int((mask & values).sum())} / {int(values.sum())}")
        finally:
            self._filling = False

        eligible = self.eligible()
        groups = self.groups()
        self._picks = sp.draw(self._table, eligible, groups, self._n.value(), self._seed.value(),
                              share=self._share.isChecked())
        self._fill_grid()
        text = f"{int(eligible.sum())} kept, {len(self._picks.rows)} picked."
        n = self._n.value()
        short = []
        for name, (got, have) in self._picks.filled.items():
            note = ""
            if n <= 0:
                note = " (all)"
            elif got < n:
                note = (f" (only {have} kept)" if have < n
                        else " (the rest were drawn for an earlier group)")
            short.append(f"{name} → {got}{note}")
        if short:
            text += " " + "; ".join(short) + "."
        else:
            text += " Nothing ticked to draw, so every kept sample is picked."
        self._summary.setText(text)

    def _fill_grid(self) -> None:
        self._grid.setUpdatesEnabled(False)
        try:
            self._fill_grid_items()
        finally:
            self._grid.setUpdatesEnabled(True)
        self._draw_thumbnails()

    def _fill_grid_items(self) -> None:
        self._grid.clear()
        self._items = {}
        for row in self._picks.rows:
            path = self._matched.get(row)
            name = Path(path).name if path is not None else str(row)
            name = name[: len(sp._key(name))]
            groups = self._picks.groups.get(row, [])
            item = QListWidgetItem(name +("\n" + "\n".join(groups) if groups else ""))
            item.setData(ROW_ROLE, row)
            item.setToolTip(f"{path}\n" + "\n".join(
                f"{c}: {sp._text(self._table.at[row, c])}" for c in self._columns))
            item.setTextAlignment(Qt.AlignHCenter | Qt.AlignTop)
            icon = self._icon(path) if path is not None else None
            if icon is not None:
                item.setIcon(icon)
            if row in self._opened:
                self._mark_opened(item)
            self._grid.addItem(item)
            if path is not None:
                self._items.setdefault(str(path), []).append(item)

    # -- thumbnails -----------------------------------------------------------

    def _icon(self, path) -> QIcon | None:
        """The pick's thumbnail: drawn here before, or by the Experiment setup scan."""
        key = str(path)
        if key in self._icons:
            return self._icons[key]
        icon = self._icon_of(Path(path))
        return icon if icon is not None and not icon.isNull() else None

    def on_preview(self, path: str) -> None:
        """The scan drew a thumbnail: show it on that pick, if it is one."""
        if path in self._items and path not in self._icons:
            icon = self._icon(path)
            if icon is not None:
                self._set_icon(path, icon)

    def _set_icon(self, path: str, icon: QIcon) -> None:
        for item in self._items.get(path, []):
            try:
                item.setIcon(icon)
            except RuntimeError:  # the grid was refilled meanwhile
                pass

    def _draw_thumbnails(self) -> None:
        """Draw the picks' thumbnails the scan has not reached, a few at a time.

        Only the picks: tens of files rather than the folder, so the picker does
        not wait for the scan.
        """
        missing = [path for path, items in self._items.items()
                   if items and items[0].icon().isNull()][:OWN_THUMBNAILS]
        self._generation += 1
        if not missing:
            return
        generation = self._generation
        from napari.qt.threading import thread_worker

        from .. import experiment as ex

        @thread_worker
        def _run():
            for path in missing:
                if generation != self._generation:
                    return
                if path in self._icons:
                    continue
                yield path, ex.thumbnail(path)

        worker = _run()
        worker.yielded.connect(self._on_thumbnail)
        worker.start()

    def _on_thumbnail(self, payload) -> None:
        from .experiment_widget import _to_pixmap

        path, rgb = payload
        pixmap = _to_pixmap(rgb) if rgb is not None else None
        if pixmap is None:
            return
        icon = QIcon(pixmap)
        self._icons[path] = icon
        self._set_icon(path, icon)

    def picked_paths(self) -> list[Path]:
        return [Path(self._matched.get(row)) for row in self._picks.rows
                if self._matched.get(row) is not None]

    def select_picks(self) -> None:
        paths = self.picked_paths()
        found = self._select(paths)
        text = f"{found} of {len(paths)} pick(s) selected in Experiment setup."
        if found is not None and found < len(paths):
            text += " The scan has not reached the rest yet; they are selected as it does."
        self._summary.setText(text)

    @staticmethod
    def _mark_opened(item) -> None:
        font = item.font()
        font.setItalic(True)
        item.setFont(font)
        item.setForeground(Qt.gray)

    def _open_item(self, item) -> None:
        row = item.data(ROW_ROLE)
        path = self._matched.get(row)
        if path is None:
            return
        self._opened.add(row)
        self._mark_opened(item)
        self._grid.setCurrentItem(item)
        self._open_path(Path(path))

    def open_next(self) -> None:
        for i in range(self._grid.count()):
            item = self._grid.item(i)
            if item.data(ROW_ROLE) not in self._opened:
                self._grid.scrollToItem(item)
                name = item.text().splitlines()[0]
                self._open_item(item)
                left = sum(1 for row in self._picks.rows if row not in self._opened)
                self._summary.setText(f"Opened {name}; {left} not opened yet.")
                return
        self._summary.setText("Every pick has been opened.")

    def save_list(self, path=None) -> Path | None:
        if self._table is None or not self._picks.rows:
            self._summary.setText("Nothing picked to save.")
            return None
        if path is None:
            folder = self._folder() or Path(self._path_edit.text()).parent
            default = Path(folder) / f"picks_{datetime.now():%Y%m%d_%H%M}.csv"
            path, _ = QFileDialog.getSaveFileName(self, "Save the picks", str(default), "CSV (*.csv)")
            if not path:
                return None
        path = Path(path)
        sp.picks_table(self._table, self._picks, self._matched).to_csv(path, index=False, encoding="utf-8-sig")
        self._summary.setText(f"Saved {len(self._picks.rows)} pick(s) to {path}.")
        return path
