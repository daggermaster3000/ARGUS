"""Dock panel: register the open brain onto a reference atlas and read signal out per region.

Self-contained — pulled in only by its entry in
:mod:`microscopy_viewer.widgets.registry`, so ``antspyx`` stays off the startup
path and a machine without it still gets every other panel.

The heavy work lives in :mod:`microscopy_viewer.registration`; this module
assigns roles to the open layers, snapshots them on the main thread, hands the
snapshot to a napari ``thread_worker`` — SyN takes minutes and the window has to
stay usable — and turns what comes back into layers and a table.

The role assignment is the part worth understanding. Only the channel marked
*Registration driver* is fitted; everything else rides on the transform that
produces. Roles are guessed from the channel names the reader already recorded,
so a file whose channels are called "dapi", "Actub" and "sv2" arrives correctly
assigned, but every guess is a combo box that can be overridden.
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
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .. import registration as rg
from ..exports import default_stem, export_table
from ..utils import format_number, get_logger

logger = get_logger("registration_widget")

#: Suffix on layers produced by a run, so a second run is distinguishable from
#: the first and the sources stay obvious in the layer list.
WARPED_SUFFIX = "→ atlas"

ATLAS_FILTER = "Atlas volumes (*.nrrd *.nii *.nii.gz *.tif *.tiff *.h5 *.hdf5);;All files (*)"
LABEL_FILTER = "Region masks (*.h5 *.hdf5 *.nrrd *.nii *.nii.gz *.tif *.tiff);;All files (*)"
NAMES_FILTER = "Region names (*.csv *.txt);;All files (*)"


def _short_name(layer) -> str:
    """A readable label for a layer whose name is an acquisition path.

    Imaris records the acquiring machine's own path as the image name, so a layer
    arrives called ``D:\\Transfer\\2026-08-05\\fish_4.ims :: Confocal - dapi``,
    which is three quarters useless in a narrow dock. The table shows the file's
    stem and the channel; the full name is the tooltip, and every lookup still
    keys on the real one.
    """
    name = str(getattr(layer, "name", ""))
    head, separator, tail = name.partition(" :: ")
    stem = Path(head.replace("\\", "/")).stem or head
    return f"{stem} :: {tail}" if separator else stem


class RegistrationWidget(QWidget):
    """Assign channel roles, register onto an atlas, read signal per region."""

    def __init__(self, viewer, parent: QWidget | None = None):
        super().__init__(parent)
        self._viewer = viewer
        self._worker = None
        self._result: rg.RegistrationResult | None = None
        self._stats: list[rg.RegionStat] = []
        self._updating = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self._backend_notice = QLabel("")
        self._backend_notice.setWordWrap(True)
        self._backend_notice.setVisible(False)
        layout.addWidget(self._backend_notice)

        self._tabs = QTabWidget()
        self._tabs.addTab(self._build_setup_tab(), "Setup")
        self._tabs.addTab(self._build_regions_tab(), "Regions")
        layout.addWidget(self._tabs, stretch=1)

        self._status = QLabel("Assign a nuclear channel as the driver, pick an atlas, then Register.")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        self._connect_viewer()
        self.refresh_layers()
        self._check_backend()

    # -- construction ---------------------------------------------------------

    def _build_setup_tab(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(2, 2, 2, 2)

        outer.addWidget(
            QLabel(
                "<b>Channels</b> — only the <i>driver</i> is fitted. Everything else is "
                "resampled through its transform."
            )
        )
        self._layer_table = QTableWidget(0, 3, self)
        self._layer_table.setHorizontalHeaderLabels(["Layer", "Channel", "Role"])
        self._layer_table.verticalHeader().setVisible(False)
        # napari's stylesheet renders alternating rows as blank stripes.
        self._layer_table.setAlternatingRowColors(False)
        self._layer_table.setSelectionMode(QAbstractItemView.NoSelection)
        self._layer_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self._layer_table.setMinimumHeight(120)
        outer.addWidget(self._layer_table, stretch=1)

        refresh = QPushButton("Refresh from the layer list")
        refresh.clicked.connect(self.refresh_layers)
        outer.addWidget(refresh)

        outer.addWidget(self._build_atlas_box())
        outer.addWidget(self._build_settings_box())

        self._run_button = QPushButton("Register")
        self._run_button.clicked.connect(self.run)
        outer.addWidget(self._run_button)
        return page

    def _build_atlas_box(self) -> QGroupBox:
        box = QGroupBox("Atlas")
        form = QFormLayout(box)
        form.setLabelAlignment(Qt.AlignRight)

        detect = QPushButton("Detect from a folder…")
        detect.setToolTip(
            "Scan an atlas download and pick the reference, the region masks and the name list.\n"
            "A nuclear reference is preferred; falling back to tERK is reported."
        )
        detect.clicked.connect(self.detect_atlas)
        form.addRow("", detect)

        self._reference_edit, reference_row = self._path_row(
            "The volume the driver channel is registered to.", ATLAS_FILTER
        )
        form.addRow("Reference", reference_row)

        self._labels_edit, labels_row = self._path_row(
            "Region masks: an integer label volume, or one binary mask per region in an HDF5.",
            LABEL_FILTER,
        )
        form.addRow("Region masks", labels_row)

        self._names_edit, names_row = self._path_row(
            "Optional list of region names, as “id,name” or one name per line.", NAMES_FILTER
        )
        form.addRow("Region names", names_row)

        # Plain TIFF and HDF5 carry no voxel size, and most atlas downloads are
        # one of the two. Without it every distance downstream is wrong — the
        # region volumes, the scale of the warped layers, and the fit itself —
        # so it is typed here rather than silently assumed to be 1 µm.
        voxel_row = QWidget()
        voxel_layout = QHBoxLayout(voxel_row)
        voxel_layout.setContentsMargins(0, 0, 0, 0)
        self._voxel_boxes = {}
        for axis, label in (("z", "Z"), ("y", "Y"), ("x", "X")):
            spin = QDoubleSpinBox()
            spin.setRange(0.001, 1000.0)
            spin.setDecimals(3)
            spin.setValue(1.0)
            spin.setPrefix(f"{label} ")
            spin.setSuffix(" µm")
            spin.setToolTip(
                "Voxel size of the atlas reference. Filled in automatically when the file "
                "records one; type it in when it does not."
            )
            voxel_layout.addWidget(spin)
            self._voxel_boxes[axis] = spin
        form.addRow("Atlas voxel", voxel_row)

        self._reference_edit.editingFinished.connect(self._read_reference_voxel_size)

        self._atlas_note = QLabel("")
        self._atlas_note.setWordWrap(True)
        form.addRow("", self._atlas_note)
        return box

    def _path_row(self, tooltip: str, file_filter: str) -> tuple[QLineEdit, QWidget]:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        edit = QLineEdit()
        edit.setToolTip(tooltip)
        layout.addWidget(edit, stretch=1)
        browse = QPushButton("…")
        browse.setMaximumWidth(30)
        browse.clicked.connect(lambda: self._browse_into(edit, file_filter))
        layout.addWidget(browse)
        return edit, row

    def _build_settings_box(self) -> QGroupBox:
        box = QGroupBox("Registration")
        form = QFormLayout(box)
        form.setLabelAlignment(Qt.AlignRight)

        self._backend_box = QComboBox()
        self._backend_box.addItems(sorted(rg._BACKENDS))
        self._backend_box.currentTextChanged.connect(lambda _text: self._check_backend())
        form.addRow("Backend", self._backend_box)

        self._affine_only = QCheckBox("Affine only (fast sanity pass)")
        self._affine_only.setToolTip(
            "Minutes instead of tens of minutes. Run this first to check the stack is not "
            "mirrored before paying for the deformable pass."
        )
        form.addRow("Transform", self._affine_only)

        landmark_row = QWidget()
        landmark_layout = QHBoxLayout(landmark_row)
        landmark_layout.setContentsMargins(0, 0, 0, 0)
        self._use_landmark = QCheckBox("Include the landmark channel in the metric")
        self._use_landmark.setToolTip(
            "Off by default. A tract-rich channel pulls the warp onto tracts and lets the "
            "space between them drift, so the fit stops being driven by the nuclear channel alone."
        )
        landmark_layout.addWidget(self._use_landmark)
        self._landmark_weight = QDoubleSpinBox()
        self._landmark_weight.setRange(0.05, 1.0)
        self._landmark_weight.setSingleStep(0.05)
        self._landmark_weight.setValue(rg.RegistrationSettings.landmark_weight)
        self._landmark_weight.setPrefix("weight ")
        landmark_layout.addWidget(self._landmark_weight)
        form.addRow("Landmark", landmark_row)

        self._max_voxels = QSpinBox()
        self._max_voxels.setRange(1, 500)
        self._max_voxels.setValue(int(rg.RegistrationSettings.max_voxels // 1_000_000))
        self._max_voxels.setSuffix(" M voxels")
        self._max_voxels.setToolTip(
            "The fit runs at or below this size. The transform is smooth, so it is found on a "
            "decimated volume and applied at full atlas resolution."
        )
        form.addRow("Fit at most", self._max_voxels)

        flip_row = QWidget()
        flip_layout = QHBoxLayout(flip_row)
        flip_layout.setContentsMargins(0, 0, 0, 0)
        self._flip_boxes = {}
        for axis, label in ((0, "Z"), (1, "Y"), (2, "X")):
            tick = QCheckBox(label)
            tick.setToolTip(
                "Flip before registering. A deformable registration cannot undo a mirrored "
                "stack — it converges to a confident, wrong answer — so handedness is fixed here."
            )
            flip_layout.addWidget(tick)
            self._flip_boxes[axis] = tick
        flip_layout.addStretch(1)
        form.addRow("Flip axes", flip_row)

        self._add_layers = QCheckBox("Add the warped channels to the viewer")
        self._add_layers.setChecked(True)
        form.addRow("Results", self._add_layers)
        return box

    def _build_regions_tab(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(2, 2, 2, 2)

        outer.addWidget(
            QLabel(
                "Signal per atlas region, measured on the warped channels in atlas space so the "
                "masks stay exactly as the atlas defines them."
            )
        )
        self._region_table = QTableWidget(0, len(rg.REGION_COLUMNS), self)
        self._region_table.setHorizontalHeaderLabels(
            [rg.REGION_LABELS[column] for column in rg.REGION_COLUMNS]
        )
        self._region_table.verticalHeader().setVisible(False)
        self._region_table.setAlternatingRowColors(False)
        self._region_table.setSortingEnabled(True)
        outer.addWidget(self._region_table, stretch=1)

        buttons = QHBoxLayout()
        export = QPushButton("Export the region table…")
        export.clicked.connect(self.export_regions)
        buttons.addWidget(export)
        buttons.addStretch(1)
        outer.addLayout(buttons)
        return page

    # -- viewer state ---------------------------------------------------------

    def _connect_viewer(self) -> None:
        try:
            self._viewer.layers.events.inserted.connect(self._on_layers_changed)
            self._viewer.layers.events.removed.connect(self._on_layers_changed)
        except Exception:  # pragma: no cover - a viewer stub without events
            logger.debug("could not subscribe to layer events", exc_info=True)

    def _on_layers_changed(self, event=None) -> None:
        if not self._updating:
            self.refresh_layers()

    def _image_layers(self) -> list:
        from napari.layers import Image

        return [
            layer
            for layer in self._viewer.layers
            if isinstance(layer, Image) and WARPED_SUFFIX not in str(layer.name)
        ]

    def refresh_layers(self) -> None:
        """Rebuild the role table, keeping any role the user has already changed."""
        chosen = self.roles()
        self._updating = True
        try:
            layers = self._image_layers()
            table = self._layer_table
            table.setRowCount(len(layers))
            for row, layer in enumerate(layers):
                name = str(layer.name)
                channel = str(layer.metadata.get("mv_channel_name", "") or "")

                item = QTableWidgetItem(_short_name(layer))
                item.setFlags(Qt.ItemIsEnabled)
                item.setToolTip(name)
                # The real layer name is what every lookup keys on; only the
                # display is shortened.
                item.setData(Qt.UserRole, name)
                table.setItem(row, 0, item)

                channel_item = QTableWidgetItem(channel or "—")
                channel_item.setFlags(Qt.ItemIsEnabled)
                table.setItem(row, 1, channel_item)

                combo = QComboBox()
                combo.addItems(list(rg.ROLES))
                # A role already chosen wins over the guess: re-guessing on every
                # layer event would undo the user's correction.
                combo.setCurrentText(chosen.get(name) or rg.guess_role(channel or name))
                table.setCellWidget(row, 2, combo)
            table.resizeColumnToContents(1)
            table.resizeColumnToContents(2)
        finally:
            self._updating = False
        self._update_summary()

    def roles(self) -> dict[str, str]:
        """Layer name -> role, as the table currently reads."""
        out: dict[str, str] = {}
        for row in range(self._layer_table.rowCount()):
            item = self._layer_table.item(row, 0)
            combo = self._layer_table.cellWidget(row, 2)
            if item is not None and combo is not None:
                out[item.data(Qt.UserRole) or item.text()] = combo.currentText()
        return out

    def _update_summary(self) -> None:
        if self._worker is not None:
            return
        drivers = [name for name, role in self.roles().items() if role == rg.ROLE_DRIVER]
        if not drivers:
            self._status.setText(
                "No driver assigned. Mark the nuclear channel (DAPI) as “Registration driver”."
            )
        elif len(drivers) > 1:
            self._status.setText(
                f"{len(drivers)} channels are marked as the driver — exactly one can be. "
                "The rest should be landmark or carry-along."
            )
        else:
            carried = sum(1 for role in self.roles().values() if role == rg.ROLE_CARRY)
            self._status.setText(f"{drivers[0]} will drive the fit; {carried} channel(s) carried through.")

    def _check_backend(self) -> None:
        """Say plainly what to install when the backend is missing, and disable Run."""
        message = rg.missing_backend_message(self._backend_box.currentText() or "ants")
        if message is None:
            self._backend_notice.setVisible(False)
            self._run_button.setEnabled(True)
            return
        self._backend_notice.setText(f"<b>Registration is unavailable.</b><br>{message}")
        self._backend_notice.setVisible(True)
        self._run_button.setEnabled(False)

    # -- atlas ----------------------------------------------------------------

    def _browse_into(self, edit: QLineEdit, file_filter: str) -> None:
        start = edit.text().strip() or str(Path.home())
        path, _selected = QFileDialog.getOpenFileName(self, "Select a file", start, file_filter)
        if path:
            edit.setText(path)

    def detect_atlas(self) -> None:
        """Scan a download and fill the three paths in."""
        directory = QFileDialog.getExistingDirectory(self, "Select the atlas folder", str(Path.home()))
        if not directory:
            return
        try:
            spec = rg.discover_atlas(directory)
        except Exception as exc:
            logger.exception("atlas discovery failed")
            QMessageBox.warning(self, "Microscopy Viewer", f"Could not read that atlas folder:\n{exc}")
            return

        self._reference_edit.setText(str(spec.reference_path))
        self._labels_edit.setText(str(spec.label_path) if spec.label_path else "")
        self._names_edit.setText(str(spec.label_names_path) if spec.label_names_path else "")
        self._read_reference_voxel_size()
        detected = self._atlas_note.text()

        if spec.is_nuclear:
            self._atlas_note.setText(
                f"Nuclear reference found ({spec.reference_channel}) — same modality as DAPI. "
                f"{detected}"
            )
        else:
            self._atlas_note.setText(
                f"<b>No nuclear reference found.</b> Falling back to "
                f"{spec.reference_path.name} ({spec.reference_channel}), so DAPI will be "
                "registered across modalities. Check the overlay before trusting the region "
                f"table. {detected}"
            )

    def _read_reference_voxel_size(self) -> None:
        """Fill the voxel-size boxes from the reference file, when it records one."""
        reference = self._reference_edit.text().strip()
        if not reference:
            return
        spacing = rg.read_voxel_size(reference)
        if spacing is None:
            self._atlas_note.setText(
                f"{Path(reference).name} records no voxel size — type the atlas voxel size "
                "below, or every distance in the result will be wrong."
            )
            return
        for axis, value in zip(("z", "y", "x"), spacing):
            self._voxel_boxes[axis].setValue(float(value))
        self._atlas_note.setText(
            f"{Path(reference).name}: voxel size read from the file "
            f"({spacing[0]:g} × {spacing[1]:g} × {spacing[2]:g} µm)."
        )

    def atlas_voxel_size(self) -> tuple[float, float, float]:
        return tuple(float(self._voxel_boxes[axis].value()) for axis in ("z", "y", "x"))  # type: ignore[return-value]

    def atlas_spec(self) -> rg.AtlasSpec:
        """The atlas as the path boxes and the voxel size currently read."""
        reference = self._reference_edit.text().strip()
        if not reference:
            raise ValueError("Choose an atlas reference volume first.")
        labels = self._labels_edit.text().strip()
        names = self._names_edit.text().strip()
        score, keyword = rg._score_reference(Path(reference).name)
        return rg.AtlasSpec(
            reference_path=Path(reference),
            reference_channel=keyword or "unknown",
            label_path=Path(labels) if labels else None,
            label_names_path=Path(names) if names else None,
            voxel_size_um=self.atlas_voxel_size(),
            is_nuclear=score >= 2,
        )

    def settings(self) -> rg.RegistrationSettings:
        return rg.RegistrationSettings(
            backend=self._backend_box.currentText() or "ants",
            affine_only=self._affine_only.isChecked(),
            use_landmark_metric=self._use_landmark.isChecked(),
            landmark_weight=float(self._landmark_weight.value()),
            max_voxels=int(self._max_voxels.value()) * 1_000_000,
            flip_axes=tuple(axis for axis, tick in self._flip_boxes.items() if tick.isChecked()),
        )

    # -- snapshotting ---------------------------------------------------------

    def _volume_from_layer(self, layer, role: str) -> tuple[rg.Volume | None, str]:
        """Turn one layer into a :class:`~microscopy_viewer.registration.Volume`.

        Always the full-resolution level, never a pyramid preview, and always the
        layer's own calibrated scale — the reader already derived both from the
        file, so nothing here re-reads or re-measures anything.
        """
        data = layer.data[0] if getattr(layer, "multiscale", False) else layer.data
        ndim = int(getattr(data, "ndim", 0))
        scale = tuple(float(s) for s in layer.scale)

        if ndim == 4:
            # A time series: register the timepoint on screen rather than
            # refusing, since that is the volume being looked at.
            index = int(self._viewer.dims.current_step[0]) if len(self._viewer.dims.current_step) else 0
            index = max(0, min(index, int(data.shape[0]) - 1))
            data = data[index]
            scale = scale[1:]
            ndim = 3
        if ndim != 3:
            return None, f"“{layer.name}” is {ndim}D; registration needs a 3D stack."

        voxel = tuple(scale[-3:]) if len(scale) >= 3 else (1.0, 1.0, 1.0)
        return (
            rg.Volume(name=str(layer.name), data=data, voxel_size_um=voxel, role=role),  # type: ignore[arg-type]
            "",
        )

    def _build_volumes(self) -> tuple[rg.Volume | None, rg.Volume | None, list[rg.Volume], list[str]]:
        """Snapshot the viewer into plain data the worker thread can use safely."""
        problems: list[str] = []
        driver: rg.Volume | None = None
        landmark: rg.Volume | None = None
        carry: list[rg.Volume] = []

        assignments = self.roles()
        for layer in self._image_layers():
            role = assignments.get(str(layer.name), rg.ROLE_CARRY)
            volume, problem = self._volume_from_layer(layer, role)
            if volume is None:
                problems.append(problem)
                continue
            if role == rg.ROLE_DRIVER:
                if driver is None:
                    driver = volume
                else:
                    problems.append(
                        f"“{layer.name}” is a second driver; {driver.name} is being used and "
                        "this one was carried through instead."
                    )
                    carry.append(rg.Volume(volume.name, volume.data, volume.voxel_size_um, rg.ROLE_CARRY))
            elif role == rg.ROLE_LANDMARK:
                landmark = volume
                carry.append(volume)  # still resampled, for the QC overlay
            else:
                carry.append(volume)
        return driver, landmark, carry, problems

    # -- the run --------------------------------------------------------------

    def run(self) -> None:
        """Register off the main thread and load what comes back."""
        if self._worker is not None:
            self._status.setText("A registration is already running.")
            return

        message = rg.missing_backend_message(self._backend_box.currentText() or "ants")
        if message is not None:
            # Status text rather than a modal box: the notice at the top of the
            # panel already says this, and Run is disabled, so a dialog here would
            # be the third time of asking.
            self._status.setText(message.replace("\n\n", " "))
            self._check_backend()
            return

        try:
            atlas = self.atlas_spec()
        except ValueError as exc:
            self._status.setText(str(exc))
            return

        driver, landmark, carry, problems = self._build_volumes()
        if driver is None:
            self._status.setText(
                "No driver assigned. Mark the nuclear channel (DAPI) as “Registration driver”."
            )
            return

        settings = self.settings()
        self._run_button.setEnabled(False)
        self._status.setText(f"Registering {driver.name} onto {atlas.reference_path.name}…")

        def _progress(text: str) -> None:
            # Runs on the worker thread; Qt only tolerates a text assignment here,
            # which is all this does.
            logger.info("registration: %s", text)

        def _work():
            result = rg.register_to_atlas(
                driver, atlas, carry=carry, landmark=landmark,
                settings=settings, progress=_progress,
            )
            stats = self._region_stats(result, atlas)
            return result, stats

        try:
            from napari.qt.threading import thread_worker
        except Exception:  # pragma: no cover - no Qt threading available
            self._finish(_work(), problems)
            return

        @thread_worker
        def _run():
            return _work()

        worker = _run()
        worker.returned.connect(lambda outcome: self._finish(outcome, problems))
        worker.errored.connect(self._on_error)
        worker.finished.connect(self._clear_worker)
        self._worker = worker
        worker.start()

    def _region_stats(self, result: rg.RegistrationResult, atlas: rg.AtlasSpec) -> list[rg.RegionStat]:
        """Per-region signal for every carried channel's warp. Worker-thread safe."""
        if atlas.label_path is None:
            return []
        signal_name = self._signal_channel(result)
        if signal_name is None:
            return []
        labels = rg.read_labels(atlas.label_path, atlas.label_names_path)
        return rg.region_table(
            result.warped[signal_name], labels, result.atlas_voxel_volume_um3
        )

    def _signal_channel(self, result: rg.RegistrationResult) -> str | None:
        """Which warped channel the region table describes: a carry-along, by preference."""
        assignments = self.roles()
        for name in result.warped:
            if assignments.get(name) == rg.ROLE_CARRY:
                return name
        return next(iter(result.warped), None)

    def _clear_worker(self) -> None:
        self._worker = None
        self._run_button.setEnabled(True)

    def _on_error(self, exc) -> None:
        logger.exception("registration failed", exc_info=exc)
        self._worker = None
        self._run_button.setEnabled(True)
        self._status.setText(f"Registration failed: {exc}")
        QMessageBox.critical(self, "Microscopy Viewer", f"Registration failed:\n{exc}")

    def _finish(self, outcome, problems: list[str]) -> None:
        result, stats = outcome
        self._result = result
        self._stats = stats
        self._run_button.setEnabled(True)

        if self._add_layers.isChecked():
            self._add_result_layers(result)
        self._fill_region_table(stats)

        messages = list(problems) + list(result.warnings)
        summary = (
            f"Registered in {result.elapsed_s:.0f} s "
            f"({result.transform_type}, {len(result.warped)} channel(s) warped"
        )
        after = result.metrics.get("mutual_information_after")
        before = result.metrics.get("mutual_information_before")
        if after is not None and before is not None:
            summary += f", MI {before:.3f} → {after:.3f}"
        summary += ")."
        if stats:
            summary += f" {len(stats)} region(s) measured."
        elif self._labels_edit.text().strip():
            summary += " No regions measured — check the mask file."
        if messages:
            summary += " " + " ".join(messages[:3])
        self._status.setText(summary)
        logger.info("registration finished: %s", summary)

    def _add_result_layers(self, result: rg.RegistrationResult) -> None:
        """Put the warped channels into the viewer at the atlas's own scale."""
        self._updating = True
        try:
            for name, volume in result.warped.items():
                layer_name = f"{name} {WARPED_SUFFIX}"
                if layer_name in self._viewer.layers:
                    self._viewer.layers.remove(layer_name)
                kwargs = {
                    "name": layer_name,
                    "scale": tuple(float(v) for v in result.atlas_voxel_size_um),
                    "blending": "additive",
                }
                source = self._viewer.layers[name] if name in self._viewer.layers else None
                if source is not None:
                    kwargs["colormap"] = source.colormap
                self._viewer.add_image(np.asarray(volume), **kwargs)
        except Exception:
            logger.exception("could not add the warped layers")
        finally:
            self._updating = False
        self.refresh_layers()

    # -- results --------------------------------------------------------------

    def _fill_region_table(self, stats) -> None:
        table = self._region_table
        table.setSortingEnabled(False)
        table.setRowCount(len(stats))
        for row, stat in enumerate(stats):
            values = stat.as_row()
            for column, key in enumerate(rg.REGION_COLUMNS):
                value = values.get(key, "")
                text = value if isinstance(value, str) else format_number(value)
                item = QTableWidgetItem(str(text))
                item.setFlags(Qt.ItemIsEnabled)
                table.setItem(row, column, item)
        table.setSortingEnabled(True)
        table.resizeColumnsToContents()

    def export_regions(self) -> None:
        """Write the region table out, through the same writer the measurements use."""
        if not self._stats:
            self._status.setText("Nothing to export — run a registration with region masks first.")
            return
        suggested = str(Path.home() / "Documents" / f"{default_stem('atlas_regions')}.xlsx")
        path, _selected = QFileDialog.getSaveFileName(
            self, "Export the region table", suggested,
            "Excel workbook (*.xlsx);;CSV file (*.csv)",
        )
        if not path:
            return
        try:
            written = export_table(rg.region_dataframe(self._stats), path, sheet_name="Regions")
        except Exception as exc:
            logger.exception("region export failed")
            QMessageBox.critical(self, "Microscopy Viewer", f"Could not write that file:\n{exc}")
            return
        self._status.setText(f"Region table written to {written}.")
