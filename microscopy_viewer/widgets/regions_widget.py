"""Dock panel: outline brain regions, name them, count objects in each.

The panel owns one Shapes layer — "Brain regions" — where each shape is one
named region. Names live in the layer's ``features`` table and are drawn on the
canvas, so the outline and its name travel together: saved with the layer,
reloaded with it, and visible while drawing rather than only in this panel.

Counting goes through :mod:`microscopy_viewer.regions`, which measures object
centroids against the outlines in world micrometres. The measurement itself runs
on a ``thread_worker`` — a whole-brain label map is hundreds of millions of
voxels and the window has to stay usable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .. import regions as rg
from ..exports import WORKBOOK_FILTER, default_stem, export_sheets
from ..utils import format_number, get_logger

logger = get_logger("regions_widget")

#: Feature column the region name is kept in on the Shapes layer.
NAME_FEATURE = "name"

#: Shown in the intensity combo for "do not measure intensities".
NO_SIGNAL = "— none —"


def _short_name(layer) -> str:
    """A readable label for a layer whose name is an acquisition path."""
    name = str(getattr(layer, "name", ""))
    head, separator, tail = name.partition(" :: ")
    stem = Path(head.replace("\\", "/")).stem or head
    return f"{stem} :: {tail}" if separator else stem


class RegionsWidget(QWidget):
    """Draw named regions and count segmented objects inside each of them."""

    def __init__(self, app, parent: QWidget | None = None):
        super().__init__(parent)
        # Takes the app so it can reopen a sample after writing into it, but a
        # bare viewer still works — the checks build it that way, and everything
        # except Save to file needs nothing more than the viewer.
        self._app = app
        self._viewer = getattr(app, "viewer", app)
        self._worker = None
        self._counts: list[rg.RegionCount] = []
        self._stats: list = []
        self._regions: list[rg.Region] = []
        self._translate: tuple[float, ...] = (0.0, 0.0)
        self._label_ndim = 3
        self._updating = False
        #: The file the outlines on the canvas were read from, so reopening it
        #: refreshes them rather than being treated as somebody else's work.
        self._regions_source: Path | None = None
        #: Set while this panel is itself closing and reopening a sample, so its
        #: own write does not come straight back as an automatic load.
        self._suspend_autoload = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        layout.addWidget(self._build_source_box())
        layout.addWidget(self._build_regions_box(), stretch=1)
        layout.addWidget(self._build_store_box())
        layout.addWidget(self._build_results_box(), stretch=1)

        self._status = QLabel(
            "Draw an outline per region, name it, then count. "
            "Names are drawn on the canvas and saved with the layer."
        )
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        self._connect_viewer()
        self.refresh_layers()
        self.refresh_regions()

    # -- construction ---------------------------------------------------------

    def _build_source_box(self) -> QGroupBox:
        box = QGroupBox("Source")
        form = QFormLayout(box)
        form.setLabelAlignment(Qt.AlignRight)

        self._labels_box = QComboBox()
        self._labels_box.setToolTip(
            "The label map whose objects are counted. Any Labels layer will do — "
            "one the segmentation panel produced, or one loaded from disk."
        )
        form.addRow("Objects", self._labels_box)

        self._signal_box = QComboBox()
        self._signal_box.setToolTip(
            "Optional: an image channel to read per-object intensities from, so the "
            "table can report a mean intensity per region as well as a count."
        )
        form.addRow("Measure", self._signal_box)

        refresh = QPushButton("Refresh from the layer list")
        refresh.clicked.connect(self.refresh_layers)
        form.addRow("", refresh)
        return box

    def _build_regions_box(self) -> QGroupBox:
        box = QGroupBox("Regions")
        outer = QVBoxLayout(box)

        row = QHBoxLayout()
        add = QPushButton("Add region")
        add.setToolTip(
            "Start a new outline. The region layer is selected and put into polygon "
            "mode; double-click to close the shape, then type its name below."
        )
        add.clicked.connect(self.add_region)
        row.addWidget(add)

        suggest = QPushButton("Suggest names")
        suggest.setToolTip(
            "Fill any unnamed outlines with " + ", ".join(rg.SUGGESTED_REGIONS[:3]) + ", … "
            "in the order they were drawn. Every name stays editable."
        )
        suggest.clicked.connect(self.suggest_names)
        row.addWidget(suggest)

        remove = QPushButton("Delete selected")
        remove.setToolTip("Remove the outlines selected in the table.")
        remove.clicked.connect(self.delete_selected)
        row.addWidget(remove)
        row.addStretch(1)
        outer.addLayout(row)

        self._region_table = QTableWidget(0, 2)
        self._region_table.setHorizontalHeaderLabels(["Region", "Area (µm²)"])
        self._region_table.verticalHeader().setVisible(False)
        self._region_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._region_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self._region_table.itemChanged.connect(self._on_name_edited)
        self._region_table.setToolTip("Double-click a name to change it.")
        outer.addWidget(self._region_table)
        return box

    def _build_store_box(self) -> QGroupBox:
        """Saving the outlines into the sample they were drawn on."""
        box = QGroupBox("Store in the sample's file")
        outer = QVBoxLayout(box)

        note = QLabel(
            "Outlines are written into the .ims itself, under /ARGUS/ROIs, so they "
            "travel with the sample. Writing needs the file to itself, so the image "
            "is closed and reopened around the write."
        )
        note.setWordWrap(True)
        outer.addWidget(note)

        row = QHBoxLayout()
        save = QPushButton("Save to file")
        save.setToolTip(
            "Write these outlines into the open sample's own .ims file." "\n"
            "For a whole folder at once, use the Experiment setup panel."
        )
        save.clicked.connect(self.save_to_file)
        row.addWidget(save)

        load = QPushButton("Load from file")
        load.setToolTip("Read back the outlines stored in the open sample's .ims file.")
        load.clicked.connect(self.load_from_file)
        row.addWidget(load)
        row.addStretch(1)
        outer.addLayout(row)
        return box

    def _build_results_box(self) -> QGroupBox:
        box = QGroupBox("Counts")
        outer = QVBoxLayout(box)

        row = QHBoxLayout()
        self._count_button = QPushButton("Count objects")
        self._count_button.clicked.connect(self.count)
        row.addWidget(self._count_button)

        export = QPushButton("Export…")
        export.setToolTip(
            "Write the per-region counts and the per-object table, with each object's "
            "region, to one workbook."
        )
        export.clicked.connect(self.export)
        row.addWidget(export)
        row.addStretch(1)
        outer.addLayout(row)

        self._result_table = QTableWidget(0, len(rg.REGION_COLUMNS))
        self._result_table.setHorizontalHeaderLabels(
            [rg.REGION_HEADERS[column] for column in rg.REGION_COLUMNS]
        )
        self._result_table.verticalHeader().setVisible(False)
        self._result_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._result_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        outer.addWidget(self._result_table)
        return box

    # -- the region layer -----------------------------------------------------

    def region_layer(self, create: bool = False):
        """The Shapes layer regions are drawn on, created on demand."""
        try:
            return self._viewer.layers[rg.REGION_LAYER_NAME]
        except (KeyError, ValueError):
            pass
        if not create:
            return None

        from ..loaders.layer_spec import world_units
        from ..measurements import NOT_A_ROI

        layer = self._viewer.add_shapes(
            name=rg.REGION_LAYER_NAME,
            face_color="#ffffff20",
            edge_color="#ffcc00",
            edge_width=2,
            # Outlines are stored in world micrometres, so the layer's own scale
            # is 1 and its unit is the micrometre. Saying so keeps the layer list
            # consistent — a layer left on the default "pixel" is dimensionless,
            # and one of those is enough to put the scale bar back to pixels.
            # On an empty viewer there is nothing to copy, and micrometres are
            # what this module means by a region either way; ``match_units``
            # corrects it if the images that arrive later turn out uncalibrated.
            **(world_units(self._viewer, 2) or {"units": ("um", "um")}),
            # Kept out of the Measurements panel. It renames every shape it finds
            # to "ROI 1", "ROI 2", … on every recompute, which would overwrite
            # the anatomical names as fast as they were typed.
            metadata={NOT_A_ROI: True},
        )
        self._install_name_feature(layer)
        return layer

    def _install_name_feature(self, layer) -> None:
        """Give the layer a ``name`` feature and draw it on the canvas.

        Wrapped because napari's features and text APIs have moved between
        versions; a viewer that cannot draw the names is still perfectly able to
        count with them, so a failure here is logged and not raised.
        """
        try:
            import pandas as pd

            if NAME_FEATURE not in getattr(layer, "features", pd.DataFrame()).columns:
                layer.features = pd.DataFrame(
                    {NAME_FEATURE: [""] * len(layer.data)}, index=range(len(layer.data))
                )
        except Exception:
            logger.debug("could not add the name feature", exc_info=True)
            return
        try:
            layer.text = {
                "string": "{" + NAME_FEATURE + "}",
                "size": 10,
                "color": "#ffcc00",
                "anchor": "center",
            }
        except Exception:
            logger.debug("could not label the shapes on the canvas", exc_info=True)

    def _names_of(self, layer) -> list[str]:
        """One name per shape, padded when the feature table has fallen behind."""
        count = len(getattr(layer, "data", ()))
        values: list[str] = []
        try:
            column = layer.features[NAME_FEATURE]
            values = ["" if value is None else str(value) for value in column]
        except Exception:
            values = []
        values = [value if value != "nan" else "" for value in values][:count]
        return values + [""] * (count - len(values))

    def _write_names(self, layer, names) -> None:
        try:
            import pandas as pd

            layer.features = pd.DataFrame({NAME_FEATURE: list(names)}, index=range(len(names)))
        except Exception:
            logger.debug("could not write the region names", exc_info=True)
            return
        try:
            layer.refresh_text()
        except Exception:
            logger.debug("could not refresh the canvas labels", exc_info=True)

    # -- viewer wiring --------------------------------------------------------

    def _connect_viewer(self) -> None:
        try:
            self._viewer.layers.events.inserted.connect(self._on_layers_changed)
            self._viewer.layers.events.removed.connect(self._on_layers_changed)
        except Exception:
            logger.debug("could not follow the layer list", exc_info=True)

    def _on_layers_changed(self, event=None) -> None:
        if self._updating:
            return
        self.match_units()
        self.refresh_layers()
        self.refresh_regions()

    def match_units(self) -> bool:
        """Put the region layer's units back in step with the images.

        The layer picks up the images' unit when it is created, but it can be
        created first — on an empty viewer, before anything is open. Opening a
        calibrated image then leaves one dimensionless layer in the list, which
        is all it takes for napari to give up on units and put the scale bar back
        to pixels. So it is matched again whenever the layer list changes.
        """
        from ..loaders.layer_spec import units_like, world_units

        layer = self.region_layer()
        if layer is None:
            return False
        wanted = world_units(self._viewer, int(layer.ndim), exclude=layer)
        if not wanted:
            # Nothing calibrated is open. If the images are on pixels, follow
            # them there — being the only dimensioned layer is just as
            # inconsistent as being the only dimensionless one.
            images = [other for other in self._viewer.layers if other is not layer]
            wanted = units_like(images[0], int(layer.ndim)) if images else {}
        if not wanted:
            return False
        current = tuple(str(unit) for unit in getattr(layer, "units", ()) or ())
        if current == tuple(str(unit) for unit in wanted["units"]):
            return False
        try:
            layer.units = wanted["units"]
        except Exception:  # pragma: no cover - napari API drift
            logger.debug("could not match the region layer units", exc_info=True)
            return False
        logger.info("region layer units matched to %s", wanted["units"])
        return True

    def _layers_of(self, kind: str) -> list:
        from napari.layers import Image, Labels

        wanted = Labels if kind == "labels" else Image
        return [layer for layer in self._viewer.layers if isinstance(layer, wanted)]

    def refresh_layers(self) -> None:
        """Repopulate the source combos, keeping the current pick where possible."""
        self._updating = True
        try:
            for box, layers, extra in (
                (self._labels_box, self._layers_of("labels"), None),
                (self._signal_box, self._layers_of("image"), NO_SIGNAL),
            ):
                previous = box.currentData()
                box.clear()
                if extra is not None:
                    box.addItem(extra, userData="")
                for layer in layers:
                    box.addItem(_short_name(layer), userData=layer.name)
                index = box.findData(previous)
                box.setCurrentIndex(max(index, 0))
        finally:
            self._updating = False

    def _layer_named(self, name: str):
        if not name:
            return None
        try:
            return self._viewer.layers[name]
        except (KeyError, ValueError):
            return None

    # -- region editing -------------------------------------------------------

    def add_region(self) -> None:
        """Select the region layer and put it into polygon-drawing mode."""
        layer = self.region_layer(create=True)
        self._updating = True
        try:
            self._viewer.layers.selection = {layer}
            layer.mode = "add_polygon"
        except Exception:
            logger.debug("could not switch the region layer into drawing mode", exc_info=True)
        finally:
            self._updating = False
        self._status.setText(
            "Click around the region and double-click to close it, then name it in the table."
        )
        self.refresh_regions()

    def suggest_names(self) -> None:
        """Name any still-unnamed outline from :data:`regions.SUGGESTED_REGIONS`."""
        layer = self.region_layer()
        if layer is None or not len(layer.data):
            self._status.setText("Draw an outline first — there is nothing to name.")
            return
        names = self._names_of(layer)
        spare = [name for name in rg.SUGGESTED_REGIONS if name not in names]
        filled = 0
        for index, name in enumerate(names):
            if name.strip():
                continue
            names[index] = spare.pop(0) if spare else f"region {index + 1}"
            filled += 1
        self._write_names(layer, names)
        self.refresh_regions()
        self._status.setText(
            f"Named {filled} outline(s)." if filled else "Every outline already has a name."
        )

    def delete_selected(self) -> None:
        """Remove the outlines whose rows are selected in the region table."""
        layer = self.region_layer()
        if layer is None:
            return
        rows = sorted({index.row() for index in self._region_table.selectedIndexes()})
        if not rows:
            self._status.setText("Select a row first.")
            return
        names = self._names_of(layer)
        keep = [i for i in range(len(layer.data)) if i not in rows]
        self._updating = True
        try:
            layer.data = [layer.data[i] for i in keep]
            self._write_names(layer, [names[i] for i in keep])
        finally:
            self._updating = False
        self.refresh_regions()
        self._status.setText(f"Removed {len(rows)} region(s).")

    def refresh_regions(self) -> None:
        """Rebuild the region table from the layer, and its areas."""
        layer = self.region_layer()
        self._regions = self.collect_regions()

        self._updating = True
        try:
            self._region_table.setRowCount(len(self._regions))
            for row, region in enumerate(self._regions):
                name_item = QTableWidgetItem(region.name)
                name_item.setFlags(name_item.flags() | Qt.ItemIsEditable)
                self._region_table.setItem(row, 0, name_item)
                area = QTableWidgetItem(format_number(region.area_um2))
                area.setFlags(area.flags() & ~Qt.ItemIsEditable)
                area.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self._region_table.setItem(row, 1, area)
        finally:
            self._updating = False

        if layer is not None:
            overlaps = rg.overlapping_pairs(self._regions)
            if overlaps:
                pairs = ", ".join(f"{a} / {b}" for a, b in overlaps[:3])
                self._status.setText(
                    f"{len(overlaps)} region(s) overlap ({pairs}). An object in two of them is "
                    "counted in the first one listed."
                )

    def _on_name_edited(self, item) -> None:
        if self._updating or item is None or item.column() != 0:
            return
        layer = self.region_layer()
        if layer is None:
            return
        names = self._names_of(layer)
        row = item.row()
        if row >= len(names):
            return
        names[row] = item.text().strip()
        self._write_names(layer, names)
        self._regions = self.collect_regions()

    def collect_regions(self) -> list[rg.Region]:
        """The current outlines as world-space :class:`regions.Region` objects."""
        layer = self.region_layer()
        if layer is None or not len(getattr(layer, "data", ())):
            return []
        try:
            types = [str(kind) for kind in layer.shape_type]
        except Exception:
            types = ["polygon"] * len(layer.data)
        return rg.regions_from_shapes(
            list(layer.data),
            types,
            self._names_of(layer),
            scale=tuple(float(value) for value in layer.scale),
            translate=tuple(float(value) for value in layer.translate),
        )

    # -- storing in the sample's own file --------------------------------------

    def sample_path(self):
        """The file the regions were drawn over, or ``None``.

        The selected image layer wins, so a viewer holding two samples writes to
        the one being looked at rather than to whichever loaded first.
        """
        from napari.layers import Image

        candidates = list(self._viewer.layers.selection) + list(self._viewer.layers)
        for layer in candidates:
            if not isinstance(layer, Image):
                continue
            meta = layer.metadata.get("mv_metadata")
            source = str(getattr(meta, "file_path", "") or "") if meta is not None else ""
            if source:
                return Path(source)
        return None

    def _close_sample(self, path) -> int:
        """Take a file's layers off screen and release the reader's handle.

        Both halves are needed before writing. Removing the layers drops the dask
        graphs, but the reader holds its ``h5py.File`` open for the life of the
        process, and HDF5 will not open for writing what is open for reading.
        """
        from ..loaders import ims as ims_reader

        target = str(Path(path).resolve())
        removed = 0
        for layer in list(self._viewer.layers):
            meta = layer.metadata.get("mv_metadata")
            source = str(getattr(meta, "file_path", "") or "") if meta is not None else ""
            if not source:
                continue
            try:
                if str(Path(source).resolve()) != target:
                    continue
            except OSError:
                continue
            self._viewer.layers.remove(layer)
            removed += 1
        ims_reader.release(path)
        return removed

    def _view_state(self) -> dict:
        """Camera and slider position, so reopening a sample does not move the view."""
        camera = self._viewer.camera
        return {
            "center": tuple(camera.center),
            "zoom": float(camera.zoom),
            "angles": tuple(camera.angles),
            "step": tuple(self._viewer.dims.current_step),
        }

    def _restore_view(self, state: dict) -> None:
        """Put the view back, and the outlines back on top of the image.

        Both matter. Removing every image layer collapses the world extent to the
        outlines alone, which moves the camera; and layers reopened after the
        region layer are drawn over it, so the outlines would come back hidden
        underneath the image they describe.
        """
        try:
            camera = self._viewer.camera
            camera.center = state["center"]
            camera.zoom = state["zoom"]
            camera.angles = state["angles"]
            step = state["step"]
            if len(step) == self._viewer.dims.ndim:
                self._viewer.dims.current_step = step
        except Exception:
            logger.debug("could not restore the view", exc_info=True)
        layer = self.region_layer()
        if layer is None:
            return
        try:
            index = self._viewer.layers.index(layer)
            if index != len(self._viewer.layers) - 1:
                self._viewer.layers.move(index, len(self._viewer.layers))
        except Exception:
            logger.debug("could not raise the region layer", exc_info=True)

    def on_files_opened(self, specs) -> None:
        """Show the regions stored in a sample that has just been opened.

        Regions live inside the ``.ims`` they were drawn on, so opening that file
        again should put them back on the canvas — having to remember to press
        *Load from file* is how a saved annotation looks lost.

        Two things it will not do. It will not overwrite outlines already on the
        canvas that came from somewhere else, because those may be unsaved work.
        And when several of the files just opened carry regions it loads none of
        them: there is one region layer, the outlines are per-sample, and quietly
        showing one sample's regions over another's image would be worse than
        showing none.
        """
        if self._suspend_autoload:
            return

        from .. import ims_store

        sources: list[Path] = []
        for spec in specs or ():
            meta = getattr(spec, "metadata", None)
            source = str(getattr(meta, "file_path", "") or "") if meta is not None else ""
            if not source:
                continue
            path = Path(source)
            if path not in sources:
                sources.append(path)

        carrying = [(path, ims_store.load_rois(path)) for path in sources]
        carrying = [(path, rois) for path, rois in carrying if rois]
        if not carrying:
            return

        if len(carrying) > 1:
            names = ", ".join(path.stem for path, _rois in carrying[:3])
            self._status.setText(
                f"{len(carrying)} of the samples just opened carry stored regions ({names}). "
                "Select one and press “Load from file” — there is only one region layer, so "
                "they cannot all be shown at once."
            )
            return

        path, rois = carrying[0]
        existing = self.collect_regions()
        if existing and self._regions_source != path:
            self._status.setText(
                f"{path.name} carries {len(rois)} stored region(s), left alone because there are "
                "already outlines on the canvas. “Load from file” replaces them."
            )
            return

        self.set_regions(rois)
        self._regions_source = path
        self._status.setText(f"Loaded {len(rois)} region(s) stored in {path.name}.")
        logger.info("auto-loaded %d region(s) from %s", len(rois), path.name)

    def save_to_file(self) -> None:
        """Write the outlines into the open sample's own ``.ims``."""
        from .. import ims_store

        regions = self.collect_regions()
        if not regions:
            self._status.setText("No outlines to save — draw one first.")
            return
        path = self.sample_path()
        if path is None:
            self._status.setText(
                "No sample open to save into. Open the image these regions belong to, "
                "or use the Experiment setup panel to write to a whole folder."
            )
            return
        ok, reason = ims_store.can_write(path)
        if not ok:
            self._status.setText(f"Cannot write: {reason}")
            return

        rois = [
            ims_store.StoredRoi(name=region.name, vertices_um=region.vertices_world)
            for region in regions
        ]
        # The image has to come off screen for the write and go back afterwards.
        # Reopening is not a reset: the viewer is not empty, so the camera stays
        # where it was.
        view = self._view_state()
        self._suspend_autoload = True
        reopen = self._close_sample(path)
        try:
            written = ims_store.save_rois(path, rois)
        except Exception as exc:
            logger.exception("could not write the regions into %s", path)
            self._status.setText(f"Could not write into {path.name}: {exc}")
            if reopen and hasattr(self._app, "open_paths"):
                self._app.open_paths([path])
                self._restore_view(view)
            self._suspend_autoload = False
            return

        self._regions_source = path
        message = f"Saved {written} region(s) into {path.name}."
        if reopen:
            if hasattr(self._app, "open_paths"):
                self._app.open_paths([path])
                self._restore_view(view)
                message += f" {reopen} layer(s) were closed and reopened to write into it."
            else:
                message += f" {reopen} layer(s) were closed to write into it."
        self._suspend_autoload = False
        self._status.setText(message)
        logger.info("%s", message)

    def load_from_file(self) -> None:
        """Read the outlines stored in the open sample back onto the canvas."""
        from .. import ims_store

        path = self.sample_path()
        if path is None:
            self._status.setText("No sample open to read from.")
            return
        rois = ims_store.load_rois(path)
        if not rois:
            self._status.setText(f"{path.name} carries no stored regions.")
            return
        self.set_regions(rois)
        self._regions_source = path
        self._status.setText(f"Loaded {len(rois)} region(s) from {path.name}.")

    def set_regions(self, rois) -> None:
        """Replace the outlines on the canvas with *rois*, names and all."""
        layer = self.region_layer(create=True)
        self._updating = True
        try:
            layer.data = [np.asarray(roi.vertices_um, dtype=float) for roi in rois]
            self._write_names(layer, [roi.name for roi in rois])
        except Exception:
            logger.exception("could not put the stored regions on the canvas")
        finally:
            self._updating = False
        self.refresh_regions()

    # -- counting -------------------------------------------------------------

    def count(self) -> None:
        """Measure the label map and attribute every object to a region."""
        if self._worker is not None:
            self._status.setText("A count is already running.")
            return

        regions = self.collect_regions()
        if not regions:
            self._status.setText("No regions drawn yet — press “Add region” first.")
            return
        unnamed = [region for region in regions if region.name.startswith("region ")]

        labels_layer = self._layer_named(str(self._labels_box.currentData() or ""))
        if labels_layer is None:
            self._status.setText("Pick a label map to count.")
            return

        masks = np.asarray(
            labels_layer.data[0] if getattr(labels_layer, "multiscale", False) else labels_layer.data
        )
        voxel = tuple(float(value) for value in labels_layer.scale)[-masks.ndim:]
        translate = tuple(float(value) for value in labels_layer.translate)

        signal = None
        signal_layer = self._layer_named(str(self._signal_box.currentData() or ""))
        problem = ""
        if signal_layer is not None:
            candidate = np.asarray(
                signal_layer.data[0]
                if getattr(signal_layer, "multiscale", False)
                else signal_layer.data
            )
            if candidate.shape == masks.shape:
                signal = candidate
            else:
                problem = (
                    f"“{_short_name(signal_layer)}” is {candidate.shape} and the labels are "
                    f"{masks.shape}; intensities were not measured."
                )

        self._label_ndim = int(masks.ndim)
        self._translate = translate
        self._regions = regions
        self._count_button.setEnabled(False)
        self._status.setText(f"Counting objects in {len(regions)} region(s)…")

        from napari.qt.threading import thread_worker

        from .. import segmentation as sg

        def _work():
            stats = sg.object_table(masks, signal, voxel)
            return stats, rg.count_objects(stats, regions, translate)

        @thread_worker
        def _run():
            return _work()

        worker = _run()
        worker.returned.connect(lambda outcome: self._finish(outcome, problem, unnamed))
        worker.errored.connect(self._on_error)
        worker.finished.connect(self._clear_worker)
        self._worker = worker
        worker.start()

    def _clear_worker(self) -> None:
        self._worker = None
        self._count_button.setEnabled(True)

    def _on_error(self, exc) -> None:
        logger.exception("region counting failed: %s", exc)
        self._status.setText(f"Counting failed: {exc}")

    def _finish(self, outcome, problem: str, unnamed) -> None:
        stats, counts = outcome
        self._stats = stats
        self._counts = counts
        self._fill_results(counts)

        total = sum(count.n_objects for count in counts)
        named = [count for count in counts if count.region != rg.UNASSIGNED]
        inside = sum(count.n_objects for count in named)
        summary = (
            f"{inside} of {total} object(s) fell inside {len(named)} region(s)."
        )
        outside = total - inside
        if outside:
            summary += f" {outside} outside every outline."
        if unnamed:
            summary += f" {len(unnamed)} outline(s) still unnamed."
        if problem:
            summary += " " + problem
        self._status.setText(summary)
        logger.info("region counts: %s", summary)

    def _fill_results(self, counts) -> None:
        self._result_table.setRowCount(len(counts))
        for row, count in enumerate(counts):
            values = count.as_row()
            for column, key in enumerate(rg.REGION_COLUMNS):
                value = values[key]
                text = value if isinstance(value, str) else format_number(value)
                item = QTableWidgetItem(text)
                if not isinstance(value, str):
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self._result_table.setItem(row, column, item)

    # -- export ---------------------------------------------------------------

    def export(self) -> None:
        """Write the counts and the per-object attribution to one workbook."""
        if not self._counts:
            self._status.setText("Nothing to export — count first.")
            return
        suggested = str(Path.home() / f"{default_stem('region_counts')}.xlsx")
        path, _selected = QFileDialog.getSaveFileName(
            self, "Export region counts", suggested, WORKBOOK_FILTER
        )
        if not path:
            return
        sheets = {"Regions": rg.counts_dataframe(self._counts)}
        if self._stats:
            # The counts and the objects behind them, in one file: a surprising
            # number in the summary is only checkable against the rows that made it.
            sheets["Objects"] = rg.objects_dataframe(
                self._stats, self._regions, self._translate, ndim=self._label_ndim
            )
        # Sample and genotype on every row, so one sample's sheets can be pasted
        # under another's without losing which fish they came from. Both are read
        # off the file the labels were counted on; a name encoding no genotype
        # leaves the column blank rather than having a guess put in it.
        sample = self.sample_path()
        if sample is not None:
            from .. import naming

            genotype = naming.genotype_from_name(sample.name)
            for frame in sheets.values():
                frame.insert(0, "Genotype", genotype)
                frame.insert(0, "Sample", sample.stem)

        try:
            written = export_sheets(sheets, path)
        except Exception as exc:
            logger.exception("region export failed")
            QMessageBox.critical(self, "Microscopy Viewer", f"Export failed:\n{exc}")
            return
        self._status.setText(
            f"Region counts written to {written} ({len(sheets)} sheet(s))."
        )
