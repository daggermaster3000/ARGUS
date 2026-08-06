"""End-to-end GUI check: build the viewer, load samples, measure, export.

Runs against a real Qt application but never needs a visible screen::

    python tests/smoke_gui.py

Set ``QT_QPA_PLATFORM=offscreen`` to run it on a machine with no display.
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SAMPLES = Path(__file__).resolve().parent / "sample_data"

_failures: list[str] = []


#: Signatures of "this machine has no usable OpenGL context right now": the
#: offscreen Qt platform, and a Remote Desktop session that has been disconnected
#: (which tears down the session's GPU context under a running process).
_NO_GL_MARKERS = (
    "no OpenGL context",
    "Failed to create context",
    "'NoneType' object has no attribute 'setsize'",
)


def _is_missing_gl(exc: BaseException) -> bool:
    return any(marker in str(exc) for marker in _NO_GL_MARKERS)


def check(condition: bool, message: str) -> None:
    if condition:
        print(f"  ok   {message}", flush=True)
    else:
        print(f"  FAIL {message}", flush=True)
        _failures.append(message)


def main() -> int:
    if not SAMPLES.exists():
        print("sample data missing — run tests/make_sample_data.py first")
        return 2

    from microscopy_viewer import measurements as mm
    from microscopy_viewer.app import MicroscopyViewer
    from microscopy_viewer.contrast import auto_contrast, reset_contrast
    from microscopy_viewer.exports import export_snapshot
    from microscopy_viewer.utils import MICRON

    print("building the viewer", flush=True)
    app = MicroscopyViewer(show=False)
    check(app.viewer is not None, "napari viewer created")
    check(app.toolbar is not None, "toolbar built")
    check(app.metadata_widget is not None, "metadata dock built")
    check(app.measurements_widget is not None, "measurements dock built")
    check(app._drop_filter is not None, "drag-and-drop filter installed")
    print(flush=True)

    print("opening samples", flush=True)
    paths = [
        SAMPLES / "sample_4d_2ch.ims",
        SAMPLES / "sample_tzcyx.ome.tif",
        SAMPLES / "sample_plain.tif",
    ]
    added = app.open_paths(paths)
    check(added == 5, f"five layers added from three files (got {added})")
    check(len(app.viewer.layers) == 5, f"viewer holds five layers ({len(app.viewer.layers)})")
    check(app.viewer.dims.ndim == 4, f"viewer is 4D for the widest dataset ({app.viewer.dims.ndim})")
    check(app.viewer.scale_bar.unit == MICRON, f"scale bar unit = {app.viewer.scale_bar.unit!r}")
    names = [layer.name for layer in app.viewer.layers]
    check(any("GFP" in name for name in names), f"channel names used in layers: {names[:2]}")
    print(flush=True)

    print("metadata panel follows the active layer", flush=True)
    ims_layer = app.viewer.layers[0]
    app.viewer.layers.selection.active = ims_layer
    app.metadata_widget.refresh()
    tree = app.metadata_widget._tree
    check(tree.topLevelItemCount() >= 2, f"tree has sections ({tree.topLevelItemCount()})")
    labels = [tree.topLevelItem(i).text(0) for i in range(tree.topLevelItemCount())]
    check(any("Channel" in label for label in labels), f"a channel section is present: {labels}")

    rows = {}
    dataset = tree.topLevelItem(0)
    for index in range(dataset.childCount()):
        rows[dataset.child(index).text(0)] = dataset.child(index).text(1)
    check("Pixel size" in rows and "0.13" in rows["Pixel size"], f"pixel size row = {rows.get('Pixel size')!r}")
    check("Z-step size" in rows and "0.5" in rows["Z-step size"], f"z-step row = {rows.get('Z-step size')!r}")
    check("Objective" in rows, f"objective row = {rows.get('Objective')!r}")
    check("Numerical aperture" in rows, f"NA row = {rows.get('Numerical aperture')!r}")

    tiff_layer = app.viewer.layers[2]
    app.viewer.layers.selection.active = tiff_layer
    heading_before = app.metadata_widget._heading.text()
    app.viewer.layers.selection.active = ims_layer
    heading_after = app.metadata_widget._heading.text()
    check(heading_before != heading_after, "heading changes when the active layer changes")
    print(flush=True)

    print("contrast controls", flush=True)
    limits_before = tuple(ims_layer.contrast_limits)
    ims_layer.contrast_limits = (0, 1)
    changed = auto_contrast(app.viewer)
    check(changed >= 1, f"auto contrast touched {changed} layer(s)")
    check(tuple(ims_layer.contrast_limits) != (0, 1), f"limits updated to {tuple(ims_layer.contrast_limits)}")
    reset = reset_contrast(app.viewer)
    check(reset == 5, f"reset contrast touched every image layer ({reset})")
    check(limits_before is not None, "original limits captured")
    print(flush=True)

    print("ROI drawing and measurement", flush=True)
    app.viewer.layers.selection.active = ims_layer
    app.measurements_widget.new_roi_layer()
    shapes = mm.shapes_layers(app.viewer)
    check(len(shapes) == 1, f"one ROI layer created ({len(shapes)})")
    roi = shapes[0]
    check(roi.ndim == ims_layer.ndim, f"ROI layer matches image ndim ({roi.ndim})")
    check(tuple(roi.scale) == tuple(ims_layer.scale), f"ROI layer inherits scale {tuple(roi.scale)}")

    # A 20 x 40 px rectangle on t=0, z=2: 0.13 µm pixels -> 2.6 x 5.2 µm = 13.52 µm².
    rectangle = np.array(
        [[0, 2, 0, 0], [0, 2, 20, 0], [0, 2, 20, 40], [0, 2, 0, 40]], dtype=float
    )
    roi.add_rectangles(rectangle)
    mm.ensure_name_feature(roi)
    app.measurements_widget.recompute()
    results = {m.measurement_type: m for m in app.measurements_widget._rows}
    check("Area" in results, f"an area was measured ({list(results)})")
    if "Area" in results:
        expected = 20 * 0.13 * 40 * 0.13
        check(
            abs(results["Area"].value - expected) < 1e-6,
            f"area = {results['Area'].value:.4f} µm² (expected {expected:.4f})",
        )
        check(results["Area"].unit == "µm²", f"unit = {results['Area'].unit}")
        check(results["Area"].image_name == ims_layer.name, f"image = {results['Area'].image_name}")
        check(results["Area"].channel != "", f"channel = {results['Area'].channel!r}")
        check("Z=2" in results["Area"].slice_position, f"slice = {results['Area'].slice_position!r}")
    check(app.measurements_widget._table.rowCount() == len(app.measurements_widget._rows), "table matches results")

    mm.rename_roi(roi, 0, "Nucleus 1")
    app.measurements_widget.recompute()
    check(
        all(m.roi_name == "Nucleus 1" for m in app.measurements_widget._rows),
        f"rename propagated: {[m.roi_name for m in app.measurements_widget._rows]}",
    )
    print(flush=True)

    print("3D / MIP shows real resolution", flush=True)
    from microscopy_viewer.rendering import _LEVEL, _PYRAMID

    # ims_layer is a 2-level Imaris pyramid: 128x160 at level 0, 64x80 at level 1.
    check(ims_layer.multiscale, "the Imaris layer is multiscale")
    check(len(ims_layer.data) == 2, f"two levels in 2D ({len(ims_layer.data)})")
    world_2d = ims_layer.extent.world.copy()
    scale_2d = tuple(ims_layer.scale)

    app.toolbar.toggle_ndisplay()
    check(app.viewer.dims.ndisplay == 3, f"viewer switched to 3D ({app.viewer.dims.ndisplay})")
    check(ims_layer.rendering == "mip", f"rendering set to MIP ({ims_layer.rendering})")
    # Without the fix napari would render len(data)-1, i.e. the 64x80 level.
    check(len(ims_layer.data) == 1, f"pyramid collapsed to one level ({len(ims_layer.data)})")
    check(
        tuple(ims_layer.data[0].shape[-2:]) == (128, 160),
        f"3D shows full resolution {tuple(ims_layer.data[0].shape[-2:])}, not the 64x80 level",
    )
    check(ims_layer.metadata[_LEVEL] == 0, f"level 0 chosen ({ims_layer.metadata[_LEVEL]})")
    check(tuple(ims_layer.scale) == scale_2d, "scale unchanged when level 0 is used")

    app.toolbar.toggle_ndisplay()
    check(app.viewer.dims.ndisplay == 2, "viewer switched back to 2D")
    check(len(ims_layer.data) == 2, f"full pyramid restored ({len(ims_layer.data)})")
    check(ims_layer.metadata[_LEVEL] is None, "layer marked as back on the full pyramid")
    check(tuple(ims_layer.scale) == scale_2d, "scale restored")
    check(np.allclose(ims_layer.extent.world, world_2d), "world extent unchanged across the round trip")

    # Force a budget so small that the coarse level must be used, and confirm the
    # scale compensates so world coordinates still line up.
    from microscopy_viewer.rendering import MultiscaleDepthManager

    tight = MultiscaleDepthManager(app.viewer, budget=1000)
    app.viewer.dims.ndisplay = 3
    tight.apply()
    check(ims_layer.metadata[_LEVEL] == 1, f"tight budget steps down to level 1 ({ims_layer.metadata[_LEVEL]})")
    check(tuple(ims_layer.data[0].shape[-2:]) == (64, 80), f"coarse level in use {tuple(ims_layer.data[0].shape[-2:])}")
    check(
        np.allclose(np.array(ims_layer.scale)[-2:], np.array(scale_2d)[-2:] * 2),
        f"voxel size doubled to match {tuple(ims_layer.scale)[-2:]}",
    )
    app.viewer.dims.ndisplay = 2
    tight.apply()
    check(len(ims_layer.data) == 2 and tuple(ims_layer.scale) == scale_2d, "restored after the tight-budget run")
    check(_PYRAMID in ims_layer.metadata, "the original pyramid is still held on the layer")
    print(flush=True)

    print("exports", flush=True)
    with tempfile.TemporaryDirectory() as directory:
        out = Path(directory)
        # Canvas screenshots need a live OpenGL context. That is absent under
        # QT_QPA_PLATFORM=offscreen and on a Remote Desktop session that has been
        # disconnected, so treat it as a skip rather than failing the whole run.
        try:
            png = export_snapshot(app.viewer, out / "shot.png", scale=1.0)
            gl_available = True
        except Exception as exc:
            if not _is_missing_gl(exc):
                raise
            gl_available = False
            print(f"  skip snapshot checks — no OpenGL context ({exc})", flush=True)

        if gl_available:
            check(png.exists() and png.stat().st_size > 1000, f"PNG snapshot written ({png.stat().st_size} bytes)")
            tif = export_snapshot(app.viewer, out / "shot.tif", scale=1.0)
            check(tif.exists() and tif.stat().st_size > 1000, f"TIFF snapshot written ({tif.stat().st_size} bytes)")

            import tifffile

            with tifffile.TiffFile(tif) as handle:
                check(handle.pages[0].shape[0] > 0, f"TIFF snapshot is a real image {handle.pages[0].shape}")

        from microscopy_viewer.exports import export_measurements

        book = export_measurements(
            app.measurements_widget._rows,
            out / "measurements.xlsx",
            [layer.metadata.get("mv_metadata") for layer in app.viewer.layers],
        )
        import openpyxl

        loaded = openpyxl.load_workbook(book)
        sheet = loaded["Measurements"]
        check(sheet.max_row == len(app.measurements_widget._rows) + 1, f"workbook has a header + data rows ({sheet.max_row})")
        acquisition = loaded["Acquisition"]
        check(acquisition.max_row > 5, f"acquisition sheet populated ({acquisition.max_row} rows)")
    print(flush=True)

    print("toolbar actions", flush=True)
    visible = app.viewer.scale_bar.visible
    app.toolbar.toggle_scale_bar()
    check(app.viewer.scale_bar.visible != visible, "scale bar toggled")
    app.toolbar.toggle_scale_bar()
    # isHidden rather than isVisible: this viewer was built with show=False, so
    # every child widget reports itself invisible regardless of the dock state.
    app.toggle_metadata_panel()
    check(app._metadata_dock.isHidden(), "metadata dock hidden")
    app.toggle_metadata_panel()
    check(not app._metadata_dock.isHidden(), "metadata dock shown again")
    check("slide" in app.toolbar._buttons, "the slide export button is on the toolbar")
    print(flush=True)

    print("PowerPoint slide export", flush=True)
    from microscopy_viewer import slides
    from microscopy_viewer.widgets.slide_dialog import SlideExportDialog

    dialog = SlideExportDialog(app.viewer, None)
    check(dialog._sample_table.rowCount() > 0, f"the dialog lists {dialog._sample_table.rowCount()} sample(s)")
    check(dialog._channel_table.rowCount() > 0, f"and {dialog._channel_table.rowCount()} channel(s)")
    check(bool(dialog.labels()), "channel labels are pre-filled from the file")
    chosen = dialog.selected_samples()
    check(len(chosen) == dialog._sample_table.rowCount(), "every sample starts ticked")
    dialog.close()

    with tempfile.TemporaryDirectory() as directory:
        written = slides.export_slide(
            chosen, Path(directory) / "smoke_slide.pptx", title="Smoke test", max_pixels=200
        )
        check(
            written.exists() and written.stat().st_size > 10_000,
            f"a slide was written ({written.stat().st_size} bytes)",
        )

        from pptx import Presentation

        shapes = Presentation(str(written)).slides[0].shapes
        table = [shape.table for shape in shapes if shape.has_table][0]
        check(len(table.rows) == len(chosen) + 1, f"a header row plus one per sample ({len(table.rows)})")
        check(table.cell(0, len(table.columns) - 1).text == "Merge", "the last column is the merge")
        check(len([s for s in shapes if s.shape_type == 13]) > 0, "pictures were placed on the slide")
    print(flush=True)

    print("batch mode: a folder straight into the dialog", flush=True)
    batch = SlideExportDialog(None, SAMPLES)
    check(not batch.selected_samples(), "an empty dialog starts with nothing to export")
    before = batch._sample_table.rowCount()
    batch._load([SAMPLES])
    added = batch._sample_table.rowCount()
    check(added > before, f"the folder added {added - before} sample(s) without touching the viewer")
    check(batch._channel_table.rowCount() > 0, "and populated the channel table")
    check("slide" in batch._status.text(), f"the summary says how many slides ({batch._status.text()})")

    # A second pass over the same folder must not duplicate rows.
    batch._load([SAMPLES])
    check(batch._sample_table.rowCount() == added, "loading the same folder again adds nothing")

    # An edited label must survive more files being added.
    key = batch._columns[0][0]
    batch._channel_table.item(0, 1).setText("anti-CD31")
    batch._load([SAMPLES / "sample_zstack_imagej.tif"])
    check(batch.labels()[key] == "anti-CD31", "a typed channel label survives a further load")

    batch._rows_per_slide.setValue(2)
    with tempfile.TemporaryDirectory() as directory:
        deck = slides.export_slide(
            batch.selected_samples(), Path(directory) / "batch.pptx",
            max_pixels=120, rows_per_slide=2, contrast=slides.CONTRAST_AUTO,
        )
        count = len(Presentation(str(deck)).slides)
        expected = -(-len(batch.selected_samples()) // 2)
        check(count == expected, f"{count} slide(s) at two rows each, as expected ({expected})")
    batch.close()
    print(flush=True)

    print("drag-and-drop path handling", flush=True)
    before = len(app.viewer.layers)
    app.open_paths([str(SAMPLES / "sample_zstack_imagej.tif")])
    check(len(app.viewer.layers) == before + 1, f"dropped file added one layer ({len(app.viewer.layers)})")

    from microscopy_viewer.loaders import expand_inputs, is_supported

    check(is_supported(SAMPLES / "sample_plain.tif"), "TIFF recognised as supported")
    check(not is_supported(SAMPLES / "nope.docx"), "unrelated file rejected")
    expanded = expand_inputs([SAMPLES])
    check(len(expanded) >= 5, f"dropping the folder expands to {len(expanded)} files")
    print(flush=True)

    print("panel manifest", flush=True)
    from microscopy_viewer.widgets.registry import iter_panels

    identifiers = [spec.identifier for spec in iter_panels()]
    check("metadata" in identifiers, f"metadata panel registered ({identifiers})")
    check("measurements" in identifiers, "measurements panel registered")
    check("intensity_comparison" in identifiers, "intensity comparison panel registered")
    check("atlas_registration" in identifiers, "atlas registration panel registered")
    for identifier in identifiers:
        check(identifier in app.panels, f"{identifier} built and tracked in app.panels")
        check(identifier in app.docks, f"{identifier} has a dock")
    check(app.metadata_widget is not None, "metadata_widget attribute still populated")
    check(app.measurements_widget is not None, "measurements_widget attribute still populated")
    check(app.intensity_widget is not None, "intensity_widget attribute populated")
    check(app.registration_widget is not None, "registration_widget attribute populated")
    print(flush=True)

    print("atlas registration panel", flush=True)
    from microscopy_viewer import registration as rg

    panel = app.registration_widget
    panel.refresh_layers()
    assigned = panel.roles()
    check(bool(assigned), f"the role table lists {len(assigned)} layer(s)")
    check(
        all(role in rg.ROLES for role in assigned.values()),
        f"every layer got a role from the manifest ({sorted(set(assigned.values()))})",
    )
    # Nothing in the sample data is a nuclear channel, so nothing may be promoted
    # to driving a registration by accident.
    check(
        rg.ROLE_DRIVER not in assigned.values(),
        "no driver is guessed when no channel looks nuclear",
    )
    check(rg.guess_role("Confocal - dapi") == rg.ROLE_DRIVER, "a DAPI channel is guessed as the driver")
    check(rg.guess_role("Confocal - Actub") == rg.ROLE_LANDMARK, "acetylated tubulin is guessed as a landmark")
    check(rg.guess_role("Confocal - sv2") == rg.ROLE_CARRY, "an unrecognised channel stays a carry-along")

    # Refusals have to be quiet status text, not modal dialogs: an offscreen run
    # would hang on one, and so would a user who just wanted to look at the panel.
    panel.run()
    check("driver" in panel._status.text().lower() or "antspyx" in panel._status.text(),
          f"running with nothing set up explains itself ({panel._status.text()[:60]}…)")
    panel.export_regions()
    check("Nothing to export" in panel._status.text(), "exporting with no result is refused politely")

    first_layer = next(iter(assigned))
    volume, problem = panel._volume_from_layer(app.viewer.layers[first_layer], rg.ROLE_CARRY)
    if volume is not None:
        check(len(volume.spacing) == 3, f"a snapshot carries a 3-axis voxel size ({volume.spacing})")
        check(
            all(size > 0 for size in volume.spacing),
            "and no zero voxel size, which would collapse the stack in physical space",
        )
    else:
        check("3D" in problem, f"a non-3D layer is reported rather than snapshotted ({problem})")

    if rg.backend_available("ants"):
        check(panel._run_button.isEnabled(), "antspyx is installed, so Register is live")
    else:
        check(not panel._run_button.isEnabled(), "without antspyx, Register is disabled")
        check(not panel._backend_notice.isHidden(), "and the panel says what to install")
    print(flush=True)

    print("ROI intensity comparison panel", flush=True)
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import QTableWidgetItem

    from microscopy_viewer import intensity as ix

    panel = app.intensity_widget
    panel.fill_from_layers()
    rows = panel.condition_rows()
    check(len(rows) >= 3, f"one condition per image layer ({len(rows)})")
    check(all(name and layer for name, layer in rows), f"every row is populated: {rows[:2]}")

    # Two conditions is the minimum the panel needs; keep the first two layers.
    while panel._condition_table.rowCount() > 2:
        panel._condition_table.selectRow(panel._condition_table.rowCount() - 1)
        panel.remove_condition()
    check(len(panel.condition_rows()) == 2, f"trimmed to two conditions ({len(panel.condition_rows())})")

    print("  -- projection layers --", flush=True)
    from qtpy.QtCore import QCoreApplication

    source_name = panel.condition_rows()[0][1]
    source = app.viewer.layers[source_name]
    check(source.ndim == 4, f"the source layer is a 4D stack ({source.ndim}D)")
    before = len(app.viewer.layers)

    panel._mode_box.setCurrentText("Maximum projection")
    panel.create_projection_layers()
    for _ in range(600):
        QCoreApplication.processEvents()
        if len(app.viewer.layers) > before:
            break
        time.sleep(0.01)

    mip_name = ix.projection_layer_name(source_name, "Maximum projection")
    check(mip_name in app.viewer.layers, f"a MIP layer was added ({mip_name})")
    mip = app.viewer.layers[mip_name]
    check(mip.ndim == 2, f"the MIP layer is 2D so shapes can be drawn on it ({mip.ndim}D)")
    check(tuple(mip.data.shape) == (128, 160), f"full XY resolution kept {tuple(mip.data.shape)}")
    check(tuple(mip.scale) == tuple(source.scale)[-2:], f"calibration carried over {tuple(mip.scale)}")
    check(mip.metadata.get("mv_projection") == "Maximum projection", "projection recorded on the layer")
    check(mip.metadata.get("mv_source_layer") == source_name, "source layer recorded")
    check(mip.metadata.get("mv_axes") == "YX", "axes updated to YX")

    # The projection must be the true max over Z at the displayed timepoint.
    step = int(app.viewer.dims.current_step[0])
    expected = np.asarray(source.data[0][step]).max(axis=0)
    check(np.array_equal(np.asarray(mip.data), expected), "pixels equal the max over Z")

    check(
        panel.condition_rows()[0][1] == mip_name,
        f"the condition now points at the MIP layer ({panel.condition_rows()[0][1]})",
    )
    check(panel._mode_box.currentText() == "Current slice", "mode reset now that the layer is flat")

    # Running it twice must update in place rather than pile up duplicates.
    count = len(app.viewer.layers)
    panel._mode_box.setCurrentText("Maximum projection")
    panel.create_projection_layers()
    for _ in range(400):
        QCoreApplication.processEvents()
        if panel._project_button.isEnabled():
            break
        time.sleep(0.01)
    check(len(app.viewer.layers) == count, f"re-projecting adds no duplicate layers ({len(app.viewer.layers)})")
    print(flush=True)

    roi_layer = panel.create_roi_layer()
    check(roi_layer is not None and roi_layer.name == "Comparison ROIs", "shared ROI layer created")
    check(panel.create_roi_layer() is roi_layer, "calling again reuses the same layer")

    # The conditions now point at the 2D MIP layers, so the ROI layer is 2D too.
    check(roi_layer.ndim == 2, f"the ROI layer matches the projected layers ({roi_layer.ndim}D)")
    signal_roi = np.array([[10, 10], [30, 10], [30, 30], [10, 30]], dtype=float)
    background_roi = np.array([[60, 60], [80, 60], [80, 80], [60, 80]], dtype=float)
    roi_layer.add_rectangles(signal_roi)
    roi_layer.add_rectangles(background_roi)
    mm.ensure_name_feature(roi_layer)
    panel.refresh_rois()
    names = mm.roi_names(roi_layer)
    check(len(names) == 2, f"two ROIs drawn ({names})")

    check(panel._roi_table.rowCount() == 2, f"the ROI table lists both ROIs ({panel._roi_table.rowCount()})")
    check(
        panel._roi_table.cellWidget(0, 1).currentText() == ix.ALL_CONDITIONS,
        "ROIs default to applying to every condition",
    )
    # Mark the second ROI as the background, shared across conditions.
    panel._roi_table.cellWidget(1, 2).setCurrentText("Background")

    conditions, rois, problems = panel._build_specs()
    check(len(conditions) == 2, f"two condition specs built ({len(conditions)})")
    check(len(rois) == 2, f"two ROI specs built ({len(rois)})")
    check(not problems, f"no setup problems reported ({problems})")
    check(sum(1 for roi in rois if roi.is_background) == 1, "exactly one ROI marked as background")
    check(
        conditions[0].data is not app.viewer.layers[conditions[0].layer_name].data
        or not app.viewer.layers[conditions[0].layer_name].multiscale,
        "multiscale layers are measured at full resolution, not a pyramid preview",
    )

    result = ix.run_comparison(conditions, rois, mode="Current slice")
    check(len(result.stats) == 4, f"two ROIs x two conditions ({len(result.stats)})")
    signal_stats = [s for s in result.stats if not s.is_background]
    check(all(s.n_pixels == 400 for s in signal_stats), f"20x20 ROI -> {[s.n_pixels for s in signal_stats]}")
    check(all(s.background_mean is not None for s in signal_stats), "background subtraction applied")

    panel._finish(result, problems)
    check(panel._stats_table.rowCount() == 4, f"statistics table filled ({panel._stats_table.rowCount()})")
    # One curve per ROI/condition, all ticked so the comparison is visible at once.
    check(panel._series_list.count() == 2, f"two curves listed ({panel._series_list.count()})")
    check(
        all(
            panel._series_list.item(row).checkState() == Qt.Checked
            for row in range(panel._series_list.count())
        ),
        "every curve starts ticked",
    )
    check(len(panel._selected_samples()[1]) == 2, "both curves feed the plot")
    check(panel._pair_a.count() == 2 and panel._pair_b.count() == 2, "curve pair combos populated")
    check(panel._pair_a.currentText() != panel._pair_b.currentText(), "two different curves preselected")
    check(len(panel._figure.axes) == 1, "histogram drawn onto the embedded canvas")
    check(len(panel._figure.axes[0].patches) > 0, "curves actually plotted")
    check(panel._separation.text() != "", f"separation metric shown: {panel._separation.text()[:60]}")

    # Unticking a curve drops it from the plot.
    panel._series_list.item(0).setCheckState(Qt.Unchecked)
    check(len(panel._selected_samples()[1]) == 1, "unticking removes a curve")
    panel._check_all(True)
    check(len(panel._selected_samples()[1]) == 2, "All restores them")

    panel._log_y.setChecked(True)
    panel._draw_histogram()
    check(panel._figure.axes[0].get_yscale() == "log", "log-y toggle applies to the plot")
    panel._log_y.setChecked(False)

    # The projection modes must not raise on a 4D layer.
    for mode in ix.PLANE_MODES:
        panel._mode_box.setCurrentText(mode)
        plane, description = ix.extract_plane(conditions[0], mode)
        check(plane.ndim == 2, f"{mode} yields a 2D plane {plane.shape} ({description})")
    panel._mode_box.setCurrentText("Current slice")

    print("  -- per-condition ROIs and normalisation --", flush=True)
    conditions_named = [name for name, _layer in panel.condition_rows()]
    # Give each condition its own signal ROI, under a shared comparison label.
    panel._roi_table.cellWidget(0, 2).setCurrentText("Signal")
    panel._roi_table.cellWidget(1, 2).setCurrentText("Signal")
    panel.assign_one_per_condition()
    assigned = panel._roi_assignments()
    applied = {values[0] for values in assigned.values()}
    check(applied == set(conditions_named), f"one ROI per condition ({applied})")
    labels = {values[2] for values in assigned.values()}
    check(labels == {"Sample 1"}, f"both share a comparison label ({labels})")

    _conditions, per_rois, _problems = panel._build_specs()
    check(all(roi.condition is not None for roi in per_rois), "specs carry the condition assignment")
    check(all(roi.label == "Sample 1" for roi in per_rois), "specs carry the shared label")
    for name in conditions_named:
        check(len(ix.rois_for(per_rois, name)) == 1, f"exactly one ROI applies to {name}")

    per_result = ix.run_comparison(_conditions, per_rois, mode="Current slice")
    check(len(per_result.stats) == 2, f"one measurement per condition ({len(per_result.stats)})")
    check(
        {s.roi_name for s in per_result.stats} == {"Sample 1"},
        "both rows share the label so they can be overlaid",
    )
    check(
        len({s.roi_source for s in per_result.stats}) == 2,
        "but they came from two different shapes",
    )
    panel._finish(per_result, [])
    # The point of the exercise: two ROIs drawn on two different conditions must
    # still show up as two overlaid curves.
    check(panel._series_list.count() == 2, f"both per-condition ROIs listed ({panel._series_list.count()})")
    check(len(panel._selected_samples()[1]) == 2, "and both are plotted together")
    check(len(panel._figure.axes[0].patches) > 0, "histogram drawn for per-condition ROIs")
    check(panel._pair_a.count() == 2, "either can be picked for the AUC")
    panel._update_separation()
    check("vs" in panel._separation.text(), f"separation computed across them: {panel._separation.text()[:50]}")

    # Restore a shared background so normalisation has something to work with.
    panel._roi_table.cellWidget(0, 1).setCurrentText(ix.ALL_CONDITIONS)
    panel._roi_table.cellWidget(1, 1).setCurrentText(ix.ALL_CONDITIONS)
    panel._roi_table.cellWidget(1, 2).setCurrentText("Background")
    # Give the background its own label; sharing one with a signal ROI would make
    # both write to the same (label, condition) sample key.
    panel._roi_table.setItem(1, 3, QTableWidgetItem("Background"))
    panel._normalization_box.setCurrentText("Divide by background")
    _conditions, norm_rois, _problems = panel._build_specs()
    norm_result = ix.run_comparison(
        _conditions, norm_rois, mode="Current slice", normalization="Divide by background"
    )
    signal_rows = [s for s in norm_result.stats if not s.is_background]
    check(all(s.normalized_mean is not None for s in signal_rows), "normalised means computed")
    check(all(s.normalization == "Divide by background" for s in signal_rows), "choice recorded per row")
    panel._finish(norm_result, [])
    check(
        panel._figure.axes[0].get_xlabel() == ix.normalized_axis_label("Divide by background"),
        f"the axis label reflects the normalisation ({panel._figure.axes[0].get_xlabel()!r})",
    )
    panel._normalization_box.setCurrentText("None")
    print(flush=True)

    with tempfile.TemporaryDirectory() as directory:
        csv_path = Path(directory) / "stats.csv"
        ix.stats_dataframe(norm_result.stats).to_csv(csv_path, index=False)
        header = csv_path.read_text(encoding="utf-8").splitlines()[0]
        check("Normalised mean" in header, f"CSV carries the normalised column: {header[:70]}")
        ix.stats_dataframe(result.stats).to_csv(csv_path, index=False)
        check(csv_path.stat().st_size > 0, f"stats CSV written ({csv_path.stat().st_size} bytes)")
        png_path = Path(directory) / "hist.png"
        panel._figure.savefig(png_path, dpi=100)
        check(png_path.stat().st_size > 1000, f"histogram PNG written ({png_path.stat().st_size} bytes)")
    print(flush=True)

    print("threaded measurement keeps the UI responsive", flush=True)
    from qtpy.QtCore import QCoreApplication

    panel._result = None
    panel.measure()
    check(not panel._measure_button.isEnabled(), "the Measure button is disabled while running")
    for _ in range(600):  # up to ~6 s
        QCoreApplication.processEvents()
        if panel._result is not None and panel._worker is None:
            break
        time.sleep(0.01)
    check(panel._result is not None, "the worker delivered a result back to the main thread")
    check(panel._worker is None, "the worker was cleared when it finished")
    check(panel._measure_button.isEnabled(), "the Measure button is re-enabled afterwards")
    check(panel._stats_table.rowCount() == 4, f"table refreshed from the worker ({panel._stats_table.rowCount()})")
    print(flush=True)

    app.viewer.close()

    if _failures:
        print(f"{len(_failures)} check(s) failed:")
        for failure in _failures:
            print(f"  - {failure}")
        return 1
    print("all GUI checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
