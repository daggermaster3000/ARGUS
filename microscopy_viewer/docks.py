"""Window plumbing: scroll-wrapped panels, a window that fits its screen, a Panels menu.

Nine dock panels on one window is a layout problem before it is anything else,
and the failure it produces does not look like a layout problem. It looks like a
broken window: one that opens taller than the screen, cannot be dragged smaller,
has its resize corner somewhere below the dock, and fights every edge drag.

The cause is a rule of Qt's: **a QMainWindow cannot shrink below the sum of its
docks' minimum sizes**, and a tab stack contributes the largest minimum in it.
Measured on this viewer before any of this existed::

    screen available : 2560 x 1392
    window minimum   : 1258 x 1517      <- 125 px taller than the screen

    Measurement analysis   min 545x1083   <- the tallest of six tabbed panels
    Segmentation           min 644x 892
    File explorer          min 319x 801
    Atlas registration     min 734x 752
    ROI intensity          min 204x 681
    + the toolbar (51), Time series (128) and the title bars

Everything here exists to take those numbers apart:

* :func:`scrollable` wraps a panel so its dock's minimum is a scroll area's
  minimum — tens of pixels — instead of the form's natural height. The panel is
  not made smaller; it scrolls.
* :func:`fit_to_screen` deals with the other half, which is that napari restores
  a window size saved on whatever monitor it was last used on.
* :func:`size_docks` gives the columns explicit proportions, because a panel that
  no longer demands its height will otherwise be handed a sliver of one.
* :func:`panels_menu` starts the panels hidden and opens them one at a time,
  because nine tabs in a column is clutter even when it fits.

No napari imports here: these take a ``QMainWindow`` and ``QDockWidget``, so they
can be tested against a bare Qt window with no viewer at all.
"""

from __future__ import annotations

from typing import Callable, Iterable, Sequence

from .utils import get_logger

logger = get_logger("docks")

#: What a dock is allowed to shrink to once its panel scrolls. Small enough that
#: the sum of a column of them is never what stops the window fitting.
DOCK_MINIMUM = (50, 50)

#: A scroll-wrapped panel asks for at most this fraction of the screen, so one
#: tall form cannot decide the size of the whole window.
SCREEN_FRACTION = 2


def screen_rect(widget):
    """The available geometry of the screen *widget* is on, or an empty rect."""
    from qtpy.QtCore import QRect

    try:
        handle = widget.screen()
    except Exception:  # pragma: no cover - no platform screen (offscreen tests)
        handle = None
    if handle is None:
        from qtpy.QtWidgets import QApplication

        handle = QApplication.primaryScreen()
    if handle is None:  # pragma: no cover - no screen at all
        return QRect()
    return handle.availableGeometry()


def _panel_scroll_area_class():
    """The QScrollArea subclass, built lazily so importing this needs no Qt."""
    from qtpy.QtCore import QSize
    from qtpy.QtWidgets import QScrollArea

    class PanelScrollArea(QScrollArea):
        """A scroll area that asks for the size its panel wants, within reason.

        Without ``sizeHint`` a scroll area asks for almost nothing, and Qt is
        happy to oblige: panels open as a 93-pixel sliver with everything inside
        them scrolled away, which reads as a broken panel rather than a small
        one. With it the panel gets the size it wants — capped at half the
        screen, so one long form does not go back to deciding the window size.
        """

        def sizeHint(self):  # noqa: N802 - Qt's name
            hint = super().sizeHint()
            inner = self.widget()
            if inner is None:
                return hint
            want = inner.sizeHint()
            width, height = want.width(), want.height()
            area = screen_rect(self)
            if not area.isEmpty():
                width = min(width, area.width() // SCREEN_FRACTION)
                height = min(height, area.height() // SCREEN_FRACTION)
            return QSize(max(hint.width(), width), max(hint.height(), height))

    return PanelScrollArea


def scrollable(widget, vertical: bool = True):
    """*widget* in a scroll area, ready to be given to ``add_dock_widget``.

    The panel keeps its own size; what changes is that the **dock** no longer
    insists on it, which is the whole point — see this module's own docstring for
    the numbers.

    *vertical* off is for a one-row toolbar: it keeps the row's exact height and
    scrolls sideways instead, so a window narrower than the toolbar does not grow
    a second row or clip the last button.
    """
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import QFrame, QSizePolicy, QStyle

    area = _panel_scroll_area_class()()
    area.setWidgetResizable(True)
    area.setFrameShape(QFrame.NoFrame)
    area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    area.setVerticalScrollBarPolicy(
        Qt.ScrollBarAsNeeded if vertical else Qt.ScrollBarAlwaysOff
    )
    area.setWidget(widget)

    if not vertical:
        height = widget.sizeHint().height()
        style = area.style()
        # A platform with overlay scroll bars needs no room for one; one with
        # permanent bars does, or the bar eats the bottom of the buttons.
        if not style.styleHint(QStyle.SH_ScrollBar_Transient, None, area):
            height += style.pixelMetric(QStyle.PM_ScrollBarExtent, None, area)
        area.setFixedHeight(height)
        area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    return area


def relax(dock) -> None:
    """Let *dock* shrink: clear the minimum its old widget pinned on it.

    A dock remembers the minimum size of the widget it was given, and swapping in
    a scroll area does not clear it. napari's own layer-controls dock is the one
    that matters — its minimum grows with the selected layer's type and stays
    there — but it is true of any dock whose widget is replaced.
    """
    try:
        dock.setMinimumSize(*DOCK_MINIMUM)
        layout = dock.layout()
        if layout is not None:
            layout.invalidate()
    except Exception:  # pragma: no cover - a dock without a layout
        logger.debug("could not relax a dock", exc_info=True)


def pack_to_top(dock, panel) -> None:
    """Keep *panel*'s controls at the top of its dock rather than spread out.

    napari adds a vertical stretch to the widget it is handed; handed a scroll
    area, the stretch lands in the scroll area and the panel inside it spreads to
    fill, with its rows drifting apart. This puts the stretch on the panel, where
    napari would have put it.
    """
    helper = getattr(dock, "_maybe_add_vertical_stretch", None)
    if helper is None:
        return
    try:
        helper(panel)
    except Exception:  # pragma: no cover - napari changed the helper
        logger.debug("could not pack the panel to the top", exc_info=True)


def settle(work: Callable) -> None:
    """Let Qt finish its layout pass, then do *work*. No timers.

    Both of these jobs — fitting the window to the screen and giving the dock
    columns their proportions — have to happen *after* Qt has laid the window
    out, or they are computed against geometry that is about to change.

    The obvious way to arrange that is a zero or 50 ms ``QTimer``. Do not. A
    timer whose slot is a Python callable and which has not fired by the time the
    interpreter shuts down is called by Qt while Python is dismantling the
    modules that callable closed over: an access violation at process exit, exit
    code 0xC0000005, with no traceback and *after* every check in the suite has
    passed and printed itself green. Parenting the timer to the window does not
    help in a process that exits without destroying the window, and neither does
    stopping pending timers from ``atexit`` — both were tried.

    Draining the event queue instead gets the same laid-out window, synchronously,
    and leaves nothing behind to fire later.
    """
    from qtpy.QtWidgets import QApplication

    application = QApplication.instance()
    if application is not None:
        application.processEvents()
    work()


def fit_to_screen(window) -> bool:
    """Shrink and move *window* until it is on its screen. ``True`` if it moved.

    napari restores ``window_size`` and ``window_position`` from its settings,
    which were saved on whatever monitor it was last used on — a 4K desktop
    screen, say, for a window now opening on a laptop. The window then opens
    partly off-screen with its resize corner out of reach, which is the state
    that reads as "the window is frozen".

    Fullscreen and maximised windows are left alone: their geometry is the
    platform's business, and the stale normal geometry underneath is dealt with
    when they leave that state — see :func:`keep_fitting`.
    """
    try:
        if window.isFullScreen() or window.isMaximized():
            return False
        area = screen_rect(window)
        if area.isEmpty():
            return False
        frame = window.frameGeometry()
        if area.contains(frame):
            return False

        # The frame is the window plus its decorations; resize() sets the inner
        # size, so the decorations have to come off the target first or the
        # window ends up exactly its title bar too tall, every time.
        extra_width = frame.width() - window.width()
        extra_height = frame.height() - window.height()
        width = min(frame.width(), area.width())
        height = min(frame.height(), area.height())
        window.resize(
            max(width - extra_width, window.minimumWidth()),
            max(height - extra_height, window.minimumHeight()),
        )
        frame = window.frameGeometry()
        window.move(
            min(max(frame.x(), area.left()), area.right() + 1 - frame.width()),
            min(max(frame.y(), area.top()), area.bottom() + 1 - frame.height()),
        )
        logger.info(
            "fitted the window to %dx%d available at (%d, %d)",
            area.width(), area.height(), area.left(), area.top(),
        )
        return True
    except Exception:  # pragma: no cover - platform oddity
        logger.debug("could not fit the window to the screen", exc_info=True)
        return False


def keep_fitting(window):
    """Fit *window* now and whenever it could stop fitting. Returns the filter.

    .. warning::

       **Not installed by default, and not recommended yet.** A permanently
       installed event filter on napari's main window is still installed while
       the interpreter shuts down, and Qt goes on delivering events to it after
       Python has begun dismantling this module: an access violation at process
       exit, 0xC0000005, with no traceback, *after* every check has passed and
       printed itself green. Detaching on ``Close`` and again from ``atexit``
       made it intermittent rather than gone. Measured, not guessed: starting
       from everything disabled and re-enabling one piece at a time, this was the
       only piece that crashed, and it crashed with the resizing stubbed out —
       so it is the filter, not the fit.

       :func:`fit_to_screen` on its own is clean and is what the viewer calls. It
       deals with the case that actually bites — a window size restored from a
       larger monitor — and misses only the live cases: dragging the window to
       another screen, and leaving fullscreen.

    Three moments, and each of them has been the one that was missed:

    * **shown** — the restored size arrives before anything is laid out;
    * **window state changed** — leaving fullscreen or un-maximising restores the
      *stale* normal geometry, which is exactly the one that did not fit;
    * **screen changed** — dragged from the desktop monitor to the laptop's.

    Each schedules the fit through a zero timer rather than doing it inline: the
    platform has not finished the state change when the event arrives, so a
    resize inside the handler is computed against the old geometry and undone.

    The returned event filter has to be kept alive by the caller — Qt holds only
    a borrowed reference, and a filter that is garbage collected silently stops
    filtering.

    **Everything connects through the filter object, never through a bare
    lambda.** ``screenChanged`` belongs to the platform window, which outlives the
    teardown of the widgets: a lambda connected to it has no QObject receiver for
    Qt to disconnect, so at interpreter exit it is called with a window that has
    already been destroyed — an access violation, after every test on the page
    has passed and reported itself green. A bound method of a ``QObject``
    parented to the window is disconnected with it.
    """
    from qtpy.QtCore import QEvent, QObject

    class Fitter(QObject):
        """Watches the window, and detaches itself when the window closes.

        **The detaching is the part that matters.** An event filter left on
        napari's main window is still installed while the interpreter shuts
        down, and Qt goes on delivering events to it after Python has begun
        dismantling the module its methods live in: an access violation at
        process exit, 0xC0000005, with no traceback and after every check has
        passed and printed itself green. The filter is cheap to install and
        cheap to remove; leaving it installed is what is expensive.
        """

        def __init__(self, parent):
            super().__init__(parent)
            self._attached = True

        def eventFilter(self, watched, event):  # noqa: N802 - Qt's name
            kind = event.type()
            if kind in (QEvent.Show, QEvent.WindowStateChange):
                self.soon()
            elif kind in (QEvent.Close, QEvent.DeferredDelete):
                self.detach()
            return False

        def detach(self) -> None:
            if not self._attached:
                return
            self._attached = False
            try:
                window.removeEventFilter(self)
                handle = window.windowHandle()
                if handle is not None:
                    handle.screenChanged.disconnect(self.on_screen_changed)
            except Exception:  # pragma: no cover - already torn down
                logger.debug("could not detach the window fitter", exc_info=True)

        def on_screen_changed(self, _screen=None) -> None:
            self.soon()

        def soon(self) -> None:
            if not self._attached:
                return
            # Through settle(): the platform has not finished the state change
            # when the event arrives, so a fit computed inside the handler is
            # computed against the old geometry and undone.
            settle(self.fit)

        def fit(self) -> None:
            fit_to_screen(window)

        def connect_handle(self) -> None:
            handle = window.windowHandle()
            if handle is not None:
                handle.screenChanged.connect(self.on_screen_changed)

    fitter = Fitter(window)
    window.installEventFilter(fitter)
    # And once more at interpreter exit, for a process that never closes its
    # window — a test harness, or a crash on the way out.
    import atexit

    atexit.register(fitter.detach)

    if window.windowHandle() is not None:
        fitter.connect_handle()
    else:
        # No platform window yet; it appears when the window is first shown, and
        # the Show event above is where this gets connected instead.
        settle(fitter.connect_handle)

    fitter.soon()
    return fitter


def size_docks(window, columns: Sequence[tuple[Sequence, Sequence[float], str]]) -> None:
    """Give whole columns of docks explicit proportions, once the window is laid out.

    Needed *because* the panels scroll: a dock that no longer demands its height
    is a dock Qt will happily give 93 pixels to. ``resizeDocks`` on one dock is
    not enough either — Qt takes the space from whichever neighbour it likes, so
    a whole column has to be sized in one call.

    *columns* is ``[(docks, weights, "vertical"|"horizontal"), …]``; weights are
    fractions of the window's height or width and are normalised here.
    """
    from qtpy.QtCore import Qt

    def apply() -> None:
        for docks, weights, direction in columns:
            alive = [dock for dock in docks if dock is not None]
            if len(alive) != len(weights) or not alive:
                continue
            vertical = str(direction).startswith("v")
            span = window.height() if vertical else window.width()
            total = float(sum(weights)) or 1.0
            sizes = [max(60, int(span * float(weight) / total)) for weight in weights]
            try:
                window.resizeDocks(
                    alive, sizes, Qt.Vertical if vertical else Qt.Horizontal
                )
            except Exception:  # pragma: no cover - a dock closed meanwhile
                logger.debug("could not size a dock column", exc_info=True)

    # After the window has been laid out: sizing docks before Qt has done its
    # own pass is sizing them against numbers that are about to change.
    settle(apply)


def size_bottom_bar(window, dock, panel) -> None:
    """Size a bottom dock to the height of the row of controls in it.

    A transport bar is one row of buttons. Scroll-wrapped it demands nothing, and
    Qt reads that as permission to give it a quarter of the window — pushing the
    canvas up and leaving the bar mostly empty grey.
    """
    from qtpy.QtCore import Qt

    if dock is None or panel is None:
        return

    def apply() -> None:
        try:
            wanted = panel.sizeHint().height()
            bar = dock.titleBarWidget()
            if bar is not None:
                wanted += bar.sizeHint().height()
            window.resizeDocks([dock], [max(60, int(wanted))], Qt.Vertical)
        except Exception:  # pragma: no cover - the dock closed meanwhile
            logger.debug("could not size the bottom bar", exc_info=True)

    settle(apply)


def hide_rather_than_destroy(dock) -> None:
    """Make napari's dock close button hide the dock instead of deleting it.

    napari wires that button to ``remove_dock_widget``, which destroys the dock
    and its widget. Every menu action, shortcut and attribute pointing at the
    panel is then pointing at a deleted C++ object, and the next use of it takes
    the window down. Hiding is what the button is understood to mean.

    Set on the dock, not on the title bar's button: the title bar is rebuilt
    whenever the dock is floated or moved between areas, and the button with it.
    """
    try:
        dock.destroyOnClose = dock.hide
    except Exception:  # pragma: no cover - not a napari dock
        logger.debug("could not make a dock hide on close", exc_info=True)


def panels_menu(
    window,
    entries: Iterable[tuple[str, object]],
    title: str = "&Panels",
    before=None,
) -> object | None:
    """A menu of panel toggles, in the menu bar, owned by us.

    *entries* is ``[(title, dock), …]``; *window* is napari's ``Window``, whose
    ``main_menu`` is the real ``QMenuBar``.

    **Our own menu, not an entry added to one of napari's.** napari's File, View,
    Window and the rest are app-model ``QModelMenu``s rebuilt from the command
    registry, and anything added to them by hand vanishes on the next rebuild —
    which happens when a plugin loads, among other things.

    Each entry is the dock's own ``toggleViewAction``, so the tick follows the
    dock whoever hid it: the menu, the close button, a restored session. Showing
    a panel also raises it, or a panel tabbed behind another opens *behind* its
    tab-mate and looks like nothing happened.
    """
    from qtpy.QtCore import QObject
    from qtpy.QtWidgets import QMenu

    class _Raiser(QObject):
        """Raises one dock when its menu entry is ticked.

        A ``QObject`` rather than a lambda, and parented to the menu, for the
        same reason as the window fitter: Qt disconnects a bound method when its
        receiver is destroyed, and has nothing to disconnect for a lambda. One
        left connected is called during interpreter teardown with a dock that no
        longer exists, which is an access violation *after* everything has
        passed — the hardest kind of failure to attribute.
        """

        def __init__(self, dock, parent):
            super().__init__(parent)
            self._dock = dock

        def toggled(self, checked: bool) -> None:
            if checked:
                self._dock.raise_()

    bar = getattr(window, "main_menu", None)
    if bar is None:
        logger.debug("no menu bar to add the Panels menu to")
        return None

    menu = QMenu(title, bar)
    anchor = None
    if before is not None:
        anchor = getattr(before, "menuAction", lambda: None)()
    if anchor is not None:
        bar.insertMenu(anchor, menu)
    else:
        bar.addMenu(menu)

    for name, dock in entries:
        if dock is None:
            continue
        hide_rather_than_destroy(dock)
        action = dock.toggleViewAction()
        action.setText(str(name))
        raiser = _Raiser(dock, menu)
        action.triggered.connect(raiser.toggled)
        menu.addAction(action)
    return menu


def reveal(widget) -> None:
    """Scroll *widget* into view in every scroll area above it, and show its dock.

    A control inside a scroll area can be perfectly visible to the code and
    scrolled out of sight on screen. Anything that points at a control — a
    status message that says which box to fill, a guided tour — has to ask for
    it to be brought into view first.

    The order is the part that is easy to get wrong, twice over:

    * **Open the docks first, then scroll.** A hidden widget has no laid-out
      geometry, so a scroll asked for while the dock is closed is computed
      against nothing and silently does nothing.
    * **Scroll to the widget itself, not to the child on the way up.** Walking
      the parents, the thing immediately below a scroll area is its *viewport*,
      and asking for the viewport to be made visible is asking for what is
      already visible. ``ensureWidgetVisible`` takes any descendant.
    """
    from qtpy.QtWidgets import QApplication, QDockWidget, QScrollArea

    chain = []
    node = widget
    while node is not None:
        chain.append(node)
        node = node.parentWidget()

    for node in chain:
        if isinstance(node, QDockWidget):
            node.setVisible(True)
            node.raise_()

    # Let the docks that were just opened lay themselves out, or the scroll
    # below is computed against geometry that does not exist yet.
    application = QApplication.instance()
    if application is not None:
        application.processEvents()

    for node in chain:
        if isinstance(node, QScrollArea):
            try:
                node.ensureWidgetVisible(widget)
            except Exception:  # pragma: no cover - detached widget
                logger.debug("could not scroll a widget into view", exc_info=True)


def emphasise(label, size: int = 22, weight: str = "bold") -> None:
    """Make *label* a heading. Through a stylesheet, which is what survives.

    napari's own stylesheet is applied after the widget is built and overrides
    ``setFont``, so a heading set with a ``QFont`` comes out the same size as
    everything else — which looks like the call did nothing.
    """
    try:
        label.setStyleSheet(f"font-size: {int(size)}px; font-weight: {weight};")
    except Exception:  # pragma: no cover - not a label
        logger.debug("could not emphasise a label", exc_info=True)


def relax_napari_docks(viewer) -> list[str]:
    """Clear the minimum sizes napari's own left-hand docks pin on the window.

    Those docks grow a minimum from the widget inside them — the layer-controls
    dock's grows with the selected layer's type, so a window that fits with an
    image open can stop fitting when a segmentation is added, which is a very
    confusing bug to be handed. Clearing the pinned minimum lets them shrink
    again.

    It does **not** put them in scroll areas, which is what the obvious version
    of this function did. Replacing the widget inside a dock napari owns means
    the original is reparented into a scroll area while napari still deletes it
    itself during teardown, and the process dies of an access violation at
    interpreter exit — *after* every check has passed and printed itself green.
    It also was not needed: the left column adds up to about 450 px, and it was
    the six tabbed panels on the right, 1083 px at their tallest, that the window
    could not shrink past.
    """
    done: list[str] = []
    qt_viewer = getattr(viewer.window, "_qt_viewer", None)
    if qt_viewer is None:
        return done
    for name in ("dockLayerControls", "dockLayerList"):
        dock = getattr(qt_viewer, name, None)
        if dock is None:
            continue
        relax(dock)
        done.append(name)
    return done
