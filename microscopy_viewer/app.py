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
        from .window_fit import scrollable

        self._progress("Building the toolbar…")
        self.toolbar = ViewerToolbar(self)
        # Scrolls sideways rather than holding the window as wide as every
        # button put end to end.
        self.viewer.window.add_dock_widget(
            scrollable(self.toolbar, vertical=False), name="Tools", area="top", tabify=False
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
                # In a scroll area so no panel can make the window taller or
                # wider than the screen (see :mod:`microscopy_viewer.window_fit`).
                dock = self.viewer.window.add_dock_widget(
                    scrollable(widget), name=spec.title, area=spec.area, **spec.dock_kwargs
                )
                self._add_vertical_stretch(dock, widget, spec)
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
        self._scroll_layer_controls()

    def _scroll_layer_controls(self) -> None:
        """Let napari's own layer-controls dock shrink too.

        It holds a form per layer type, and on a laptop screen its height alone
        plus the layer list and our panels already made the left column taller
        than the screen.
        """
        from .window_fit import scrollable

        try:
            dock = self.viewer.window._qt_viewer.dockLayerControls
            controls = dock.widget()
            if controls is None:
                return
            dock.setWidget(scrollable(controls))
            # By now the old contents' minimum is pinned on the dock itself;
            # put back the 50 px napari gives every dock.
            dock.setMinimumSize(50, 50)
            dock.layout().invalidate()
            dock.updateGeometry()
        except Exception:  # pragma: no cover - napari API drift
            logger.debug("could not make the layer controls scroll", exc_info=True)

    @staticmethod
    def _add_vertical_stretch(dock, widget, spec) -> None:
        """Push a side panel's controls to the top, as napari does unwrapped.

        napari adds the stretch to the widget it is given, which is now the
        scroll area; the panel inside would otherwise spread its rows out over
        the whole height of the dock.
        """
        if spec.area not in ("left", "right") or not spec.dock_kwargs.get("add_vertical_stretch", True):
            return
        add = getattr(dock, "_maybe_add_vertical_stretch", None)
        if add is None:
            return
        try:
            add(widget)
        except Exception:  # pragma: no cover - napari API drift
            logger.debug("could not add stretch to %s", spec.identifier, exc_info=True)

    def fit_window_to_screen(self) -> None:
        """Shrink and move the main window onto the screen it is on, now and later.

        napari restores the size the window last had, which may have been on a
        bigger monitor; on a laptop screen that leaves the title bar or the
        resize corner out of reach. It is fitted again on leaving full screen and
        on changing screen, where the same thing happens.
        """
        from .window_fit import keep_on_screen

        try:
            keep_on_screen(getattr(self.viewer.window, "_qt_window", None))
        except Exception:
            logger.debug("could not fit the window to the screen", exc_info=True)

    def share_dock_space(self) -> None:
        """Give the panels a useful share of the window on first open.

        Wrapped in scroll areas they no longer insist on a size of their own, so
        Qt would otherwise hand the experiment grid a sliver under the layer
        list. Sizes are fractions of the window so they suit any screen.
        """
        from qtpy.QtCore import Qt

        window = getattr(self.viewer.window, "_qt_window", None)
        if window is None:
            return
        qt_viewer = getattr(self.viewer.window, "_qt_viewer", None)
        # Every dock in a column is given its share at once: sizing one alone
        # lets Qt take the space back from whichever neighbour it likes.
        columns = (
            # (dock, share of the window's height), share of its width
            (
                (
                    (getattr(qt_viewer, "dockLayerControls", None), 0.25),
                    (getattr(qt_viewer, "dockLayerList", None), 0.2),
                    (self.docks.get("experiment"), 0.55),
                ),
                0.25,
            ),
            (
                (
                    (self.docks.get("metadata"), 0.25),
                    (self.docks.get("measurements"), 0.75),
                ),
                0.3,
            ),
        )
        height, width = window.height(), window.width()
        timeseries = self.docks.get("timeseries")
        panel = self.panels.get("timeseries")
        if timeseries is not None and panel is not None and timeseries.isVisible():
            # The transport bar: as tall as its controls and no taller, since
            # every pixel it takes comes off the canvas.
            try:
                title = timeseries.height() - timeseries.widget().height()
                window.resizeDocks([timeseries], [panel.sizeHint().height() + title], Qt.Vertical)
            except Exception:
                logger.debug("could not size the time series dock", exc_info=True)
        for column, width_share in columns:
            shown = [(dock, share) for dock, share in column if dock is not None and dock.isVisible()]
            if not shown:
                continue
            docks = [dock for dock, _ in shown]
            try:
                window.resizeDocks(docks, [int(height * share) for _, share in shown], Qt.Vertical)
                window.resizeDocks(docks[:1], [int(width * width_share)], Qt.Horizontal)
            except Exception:
                logger.debug("could not size the docks", exc_info=True)

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
            ("Control-Shift-P", self.toolbar.batch_projection),
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

    # -- the tour -------------------------------------------------------------

    def start_tour(self, index: int = 0):
        """Run the guided tour from step *index*."""
        from .widgets.tour import start_tour

        return start_tour(self, index)

    def maybe_start_tour(self):
        """Start the tour unless this version of it has been seen already."""
        from .onboarding import has_seen

        if has_seen():
            return None
        try:
            return self.start_tour()
        except Exception:
            logger.exception("could not start the guided tour")
            return None

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
    tour: bool = True,
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

    With *tour* the guided tour starts once the window is up, the first time this
    version of the tour has not been seen (see :mod:`microscopy_viewer.onboarding`).
    """
    import napari
    from napari.qt import get_qapp
    from qtpy.QtCore import QTimer
    from qtpy.QtWidgets import QApplication

    from .splash import start as start_splash

    from .dragdrop import catch_file_open_events

    # napari's QApplication, made here rather than by ``napari.Viewer`` a moment
    # later: the splash needs it, and so does catching files macOS hands over
    # (dropped on the app icon) while the window is still being built.
    get_qapp()
    file_opener = catch_file_open_events()

    banner = None
    if splash:
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
    if file_opener is not None:
        file_opener.attach(app.open_paths)
    if banner is not None:
        # Shown only now, with everything on it: the point of the splash is that
        # nobody watches an empty window being filled in.
        app.viewer.window.show()
        banner.finish(getattr(app.viewer.window, "_qt_window", None))
    app.fit_window_to_screen()
    # Once the window has its final size, which the fit settles a moment later.
    QTimer.singleShot(50, app.share_dock_space)
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

    if tour:
        # After the event loop has laid the window out: the tour measures it.
        QTimer.singleShot(800, app.maybe_start_tour)

    if block:
        napari.run()
    return app
