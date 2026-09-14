"""Assembles the customised napari viewer: docks, toolbar, shortcuts, file opening."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

from .utils import MICRON, get_logger, setup_logging

logger = get_logger("app")

WINDOW_TITLE = "Microscopy Viewer"


def suppress_shimmed_plugin_dialog() -> set[str]:
    """Stop napari's "Installed Plugin Warning" from blocking every launch.

    When any installed plugin still uses the deprecated plugin engine, napari
    opens a *modal* dialog during window construction and waits for a click. For
    a viewer started by double-clicking a desktop shortcut that is fatal: the
    window appears but nothing loads until someone finds and dismisses the box.

    This marks the plugins that are already installed as "already warned about",
    which is exactly what napari's own *Only warn me about newly installed
    plugins* checkbox does — so a plugin installed later still raises the warning.
    The change is saved in napari's own settings file, so pass
    ``suppress_plugin_warning=False`` (or ``--warn-shimmed-plugins``) to leave
    napari's behaviour alone.

    Returns the plugin names that were silenced.
    """
    try:
        from napari.settings import get_settings
        from npe2 import plugin_manager

        settings = get_settings()
        shimmed = set(plugin_manager.get_shimmed_plugins())
        already = set(settings.plugins.already_warned_shimmed_plugins)
        settings.plugins.only_new_shimmed_plugins_warning = True
        new = shimmed - already
        if new:
            settings.plugins.already_warned_shimmed_plugins = already | shimmed
            logger.info("silenced napari's shimmed-plugin dialog for: %s", ", ".join(sorted(new)))
        return new
    except Exception:
        # Never let a settings quirk stop the viewer from opening.
        logger.debug("could not adjust napari's shimmed-plugin setting", exc_info=True)
        return set()


class MicroscopyViewer:
    """Owns a napari viewer plus the widgets and behaviour layered on top of it.

    Construction is deliberately ordered: the viewer exists first, then the docks
    (the toolbar needs to reference the measurements widget), then the shortcuts
    and the drag-and-drop filter.
    """

    def __init__(
        self,
        show: bool = True,
        suppress_plugin_warning: bool = True,
        progress: Callable[[str], None] | None = None,
    ):
        import napari

        # Called at each step of the build. It is what puts the panel currently
        # being made onto the splash, and it is also what lets the splash paint:
        # none of this returns to the event loop on its own.
        self._progress = progress or (lambda _message: None)

        setup_logging()
        self._progress("Starting napari…")
        if suppress_plugin_warning:
            suppress_shimmed_plugin_dialog()
        self.viewer = napari.Viewer(title=WINDOW_TITLE, show=show)
        self.last_directory: Path | None = None
        self._metadata_dock = None
        self._load_errors: list[str] = []
        #: Set by :func:`launch` when it is watching for freezes; ``None`` otherwise.
        self.busy_watchdog = None

        self.viewer.scale_bar.visible = True
        self._set_scale_bar_unit("px")
        self.viewer.scale_bar.colored = False

        self._build_docks()
        self._bind_shortcuts()
        self._install_dragdrop()

        # Must exist before any image is added: it registers each pyramid the
        # first time it sees the layer.
        from .rendering import MultiscaleDepthManager
        from .timeseries import TimelineManager

        self.depth_manager = MultiscaleDepthManager(self.viewer, on_status=self.toolbar.set_status)
        # Caches time series onto local disk so playback stops reading the NAS.
        # It writes its cached arrays back into the same ``mv_pyramid`` list the
        # depth manager restores from, so a trip through 3D keeps them.
        self.timeline_manager = TimelineManager(self.viewer, on_status=self.toolbar.set_status)
        self._progress("Ready")

    # -- construction ---------------------------------------------------------

    def _build_docks(self) -> None:
        """Build the toolbar, then every panel declared in the manifest.

        Panels are built independently: one that fails to construct (a missing
        optional dependency, say) is logged and skipped rather than taking the
        whole window down with it.
        """
        from .widgets import ViewerToolbar
        from .widgets.registry import iter_panels

        self._progress("Building the toolbar…")
        self.toolbar = ViewerToolbar(self)
        self.viewer.window.add_dock_widget(
            self.toolbar, name="Tools", area="top", tabify=False
        )

        self.panels: dict[str, object] = {}
        self.docks: dict[str, object] = {}
        specs = iter_panels()

        # Declare the attributes up front so a failed panel leaves an explicit
        # None rather than a missing attribute for callers to trip over.
        for spec in specs:
            if spec.attribute:
                setattr(self, spec.attribute, None)

        for spec in specs:
            try:
                # Panels are where the wait is: one of them importing torch or
                # cellpose is seconds on its own, so each is named as it is built.
                self._progress(f"Building the {spec.title} panel…")
                widget = spec.factory(self)
                dock = self.viewer.window.add_dock_widget(
                    widget, name=spec.title, area=spec.area, **spec.dock_kwargs
                )
            except Exception:
                logger.exception("could not build the %r panel", spec.identifier)
                continue
            self.panels[spec.identifier] = widget
            self.docks[spec.identifier] = dock
            if spec.attribute:
                setattr(self, spec.attribute, widget)
            if spec.start_hidden:
                dock.setVisible(False)

        self._tabify_panels(specs)
        self._metadata_dock = self.docks.get("metadata")

    def _tabify_panels(self, specs) -> None:
        """Stack panels that asked to share a tab bar with another panel."""
        window = getattr(self.viewer.window, "_qt_window", None)
        if window is None:
            return
        for spec in specs:
            target = self.docks.get(spec.tabify_with or "")
            dock = self.docks.get(spec.identifier)
            if target is None or dock is None or target is dock:
                continue
            try:
                window.tabifyDockWidget(target, dock)
                target.raise_()  # keep the original panel in front
            except Exception:
                logger.debug("could not tabify %s", spec.identifier, exc_info=True)

    def _set_scale_bar_unit(self, unit: str) -> None:
        """Tell the scale bar what a world unit is, on whichever napari this is.

        Up to napari 0.6 the overlay carried its own ``unit``. From 0.8 it has no
        such field: the bar reads ``layer.units`` instead, which the readers now
        supply through :class:`~microscopy_viewer.loaders.layer_spec.LayerSpec`.
        Setting the old attribute anyway would either raise or, worse, silently
        stick an ignored value on the model, so it is only set where it is real.
        """
        scale_bar = self.viewer.scale_bar
        fields = getattr(type(scale_bar), "model_fields", None) or getattr(
            type(scale_bar), "__fields__", {}
        )
        if "unit" not in fields:
            return
        try:
            scale_bar.unit = unit
        except Exception:  # pragma: no cover - napari API drift
            logger.debug("could not set the scale bar unit", exc_info=True)

    def _bind_shortcuts(self) -> None:
        """Keyboard equivalents for the toolbar buttons."""
        bindings = (
            ("Control-O", self.toolbar.open_images),
            ("Control-S", self.toolbar.export_snapshot),
            ("Control-E", self.toolbar.export_measurements),
            ("Control-P", self.toolbar.export_slide),
            ("Control-Shift-A", self.toolbar.auto_contrast),
            ("Control-D", self.toolbar.toggle_ndisplay),
            ("Control-B", self.toolbar.toggle_scale_bar),
            ("Control-M", self.toggle_metadata_panel),
            ("Control-R", self.toolbar.reset_contrast),
            ("Control-Space", self.toolbar.toggle_play),
            ("Control-Shift-M", self.toolbar.export_movie),
        )
        for key, handler in bindings:
            try:
                self.viewer.bind_key(key, lambda _viewer, _handler=handler: _handler(), overwrite=True)
            except Exception:  # pragma: no cover - key already claimed by napari
                logger.debug("could not bind %s", key, exc_info=True)

    def _install_dragdrop(self) -> None:
        from . import dragdrop

        self._drop_filter = dragdrop.install(self.open_paths)

    # -- file opening ---------------------------------------------------------

    def open_paths(self, paths: Sequence[str | Path]) -> int:
        """Load every path and add the resulting layers. Returns the layer count."""
        from .loaders import expand_inputs, load_paths

        candidates = expand_inputs(list(paths))
        if not candidates:
            self._notify("Nothing to open", "None of the dropped items look like image files.")
            return 0

        self.toolbar.set_status(f"Loading {len(candidates)} file(s)…")
        _process_events()

        was_empty = len(self.viewer.layers) == 0
        specs, errors = load_paths(candidates)

        added = 0
        for spec in specs:
            try:
                self.viewer.add_image(spec.data, **spec.to_kwargs())
                added += 1
            except Exception as exc:
                logger.exception("could not add layer %s", spec.name)
                errors.append(_FakeError(spec.name, str(exc)))

        if added:
            self._after_open(specs, reset_view=was_empty)
            self.last_directory = Path(candidates[0]).parent

        if errors:
            self._report(errors)
        if added:
            status = f"Opened {added} layer(s)."
            if errors:
                status += f" {len(errors)} item(s) failed."
        else:
            status = "No images were opened."
        self.toolbar.set_status(status)
        logger.info("%s (from %d input path(s))", status, len(candidates))
        return added

    def _after_open(self, specs, reset_view: bool) -> None:
        """Label the dimension sliders, set the scale bar unit, and frame the data."""
        widest = max(specs, key=lambda spec: len(spec.axes), default=None)
        if widest is not None and len(widest.axes) == self.viewer.dims.ndim:
            self.viewer.dims.axis_labels = tuple(widest.axes)

        calibrated = any(spec.metadata.is_calibrated for spec in specs)
        self._set_scale_bar_unit(MICRON if calibrated else "px")

        # New layers arrive as full pyramids; if the viewer is already in 3D they
        # need collapsing to a single level straight away.
        self.depth_manager.apply()
        # Starts the local copy of any time series that was just opened.
        self.timeline_manager.apply()

        if reset_view:
            self.viewer.reset_view()
            from .contrast import auto_contrast

            auto_contrast(self.viewer)

        self.metadata_widget.refresh()
        self.measurements_widget.refresh_layer_list()
        if self.timeseries_widget is not None:
            self.timeseries_widget.refresh()

        # Brain regions are stored inside the .ims they were drawn on, so opening
        # that sample again should put them back on the canvas. Done here rather
        # than off the layer-inserted event because that fires once per channel,
        # and this needs the file list exactly once per open.
        regions = getattr(self, "regions_widget", None)
        if regions is not None:
            try:
                regions.on_files_opened(specs)
            except Exception:
                logger.exception("could not restore the stored brain regions")

    # -- panels ---------------------------------------------------------------

    def toggle_metadata_panel(self) -> None:
        """Show or hide the acquisition metadata dock.

        Uses ``isHidden`` rather than ``isVisible``: a child of a window that has
        not been shown yet always reports itself invisible, which would make the
        toggle stick in one direction.
        """
        if self._metadata_dock is None:
            return
        show = self._metadata_dock.isHidden()
        self._metadata_dock.setVisible(show)
        self.toolbar.set_status(f"Metadata panel {'shown' if show else 'hidden'}.")

    # -- messaging ------------------------------------------------------------

    def _notify(self, title: str, message: str) -> None:
        from qtpy.QtWidgets import QMessageBox

        QMessageBox.information(self.viewer.window._qt_window, title, message)

    def _report(self, errors) -> None:
        """Summarise load failures in one dialog rather than one per file."""
        from qtpy.QtWidgets import QMessageBox

        lines = [str(error) for error in errors]
        logger.warning("%d file(s) failed to load", len(lines))
        QMessageBox.warning(
            self.viewer.window._qt_window,
            "Some files could not be opened",
            "\n".join(lines[:12]) + ("\n…" if len(lines) > 12 else ""),
        )


class _FakeError:
    """Adapter so layer-creation failures print like reader failures."""

    def __init__(self, name: str, message: str):
        self.name = name
        self.message = message

    def __str__(self) -> str:
        return f"{self.name}: {self.message}"


def _process_events() -> None:
    """Let the status label repaint before a long synchronous load."""
    from qtpy.QtWidgets import QApplication

    app = QApplication.instance()
    if app is not None:
        app.processEvents()


def launch(
    paths: Sequence[str | Path] = (),
    block: bool = True,
    suppress_plugin_warning: bool = True,
    splash: bool = True,
    busy_overlay: bool = True,
) -> MicroscopyViewer:
    """Create the viewer, open any given paths, and optionally run the Qt loop.

    ``block=False`` is used by tests and by interactive sessions that already
    have an event loop running.

    With *splash* the main window stays hidden until it is built and the loading
    animation is shown in its place, because napari otherwise puts an empty white
    window on screen for the whole of the build. If the splash cannot be made the
    window opens straight away as before, so nothing depends on it.

    *busy_overlay* keeps watching after that: whenever the main thread stops
    responding for longer than :data:`microscopy_viewer.busy.BUSY_AFTER_S`, the
    same animation is put over the window by a second process until it comes back.
    See :mod:`microscopy_viewer.busy` for why that has to be another process.
    """
    import napari
    from napari.qt import get_qapp
    from qtpy.QtWidgets import QApplication

    from .splash import start as start_splash

    banner = None
    if splash:
        # The splash needs a QApplication, and napari's is the one the viewer
        # will run on: made here rather than by ``napari.Viewer`` a moment later.
        get_qapp()
        banner = start_splash(WINDOW_TITLE)

    app = MicroscopyViewer(
        show=banner is None,
        suppress_plugin_warning=suppress_plugin_warning,
        progress=banner.pump if banner is not None else None,
    )
    if paths:
        if banner is not None:
            banner.pump(f"Opening {len(paths)} file(s)…")
        app.open_paths(paths)
    if banner is not None:
        # Shown only now, with everything on it: the point of the splash is that
        # nobody watches an empty window being filled in.
        app.viewer.window.show()
        banner.finish(getattr(app.viewer.window, "_qt_window", None))
    if busy_overlay:
        from . import busy

        window = getattr(app.viewer.window, "_qt_window", None)
        app.busy_watchdog = busy.start(window)
        if app.busy_watchdog is not None:
            # Stopping it on quit closes the child process politely; the child
            # also exits by itself when our stdin pipe dies, so a crash still
            # cleans up.
            application = QApplication.instance()
            if application is not None:
                application.aboutToQuit.connect(app.busy_watchdog.stop)

    if block:
        napari.run()
    return app
