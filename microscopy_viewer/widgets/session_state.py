"""Reads the window into a session, and puts a saved session back onto it.

The Qt half of :mod:`microscopy_viewer.session`. Panel settings are read off the
input widgets themselves rather than declared panel by panel, so a control added
to a panel is remembered without anything else having to change.
"""

from __future__ import annotations

from pathlib import Path

from qtpy.QtCore import QEvent, QObject
from qtpy.QtWidgets import (
    QAbstractItemView,
    QAbstractSpinBox,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QLineEdit,
    QRadioButton,
    QSlider,
    QSpinBox,
    QTabWidget,
    QWidget,
)

from .. import session
from ..utils import get_logger
from .sample_files import restore_view, source_of, view_state

logger = get_logger("session_state")

#: The kinds of control that hold a setting, in the order they are looked for.
_KINDS = (
    QSpinBox,
    QDoubleSpinBox,
    QCheckBox,
    QRadioButton,
    QComboBox,
    QLineEdit,
    QSlider,
    QGroupBox,
    QTabWidget,
)


def _controls(panel: QWidget) -> dict[str, QWidget]:
    """Every setting-holding control in *panel*, under a key that is stable.

    The key is the control's kind and its position among that kind, which only
    changes when the panel's code does; :func:`_fingerprint` catches that.
    Controls inside a table or list are left out: those rows come and go with
    the data, and belong to it rather than to the panel.
    """
    found: dict[str, QWidget] = {}
    for kind in _KINDS:
        index = 0
        for widget in panel.findChildren(kind):
            if _inside_view_or_editor(widget, panel):
                continue
            if isinstance(widget, QGroupBox) and not widget.isCheckable():
                continue
            if isinstance(widget, QLineEdit) and widget.isReadOnly():
                continue
            found[f"{kind.__name__}:{index}"] = widget
            index += 1
    return found


def _inside_view_or_editor(widget: QWidget, panel: QWidget) -> bool:
    parent = widget.parentWidget()
    while parent is not None and parent is not panel:
        # The line edit of a spin box or an editable combo is that control's own.
        if isinstance(parent, (QAbstractItemView, QAbstractSpinBox, QComboBox)):
            return True
        parent = parent.parentWidget()
    return False


def _fingerprint(controls: dict[str, QWidget]) -> str:
    """How many of each kind: saved values only go back onto the same form."""
    counts: dict[str, int] = {}
    for key in controls:
        kind = key.split(":")[0]
        counts[kind] = counts.get(kind, 0) + 1
    return ",".join(f"{kind}={count}" for kind, count in sorted(counts.items()))


def _read(widget: QWidget):
    if isinstance(widget, (QSpinBox, QDoubleSpinBox, QSlider)):
        return widget.value()
    if isinstance(widget, (QCheckBox, QRadioButton, QGroupBox)):
        return bool(widget.isChecked())
    if isinstance(widget, QComboBox):
        return widget.currentText()
    if isinstance(widget, QLineEdit):
        return widget.text()
    if isinstance(widget, QTabWidget):
        return widget.currentIndex()
    return None


def _write(widget: QWidget, value) -> None:
    """Set *value* as the user would have: signals fire, so the panel follows."""
    if isinstance(widget, (QSpinBox, QDoubleSpinBox, QSlider)):
        # Ranges often follow the data; a value that no longer fits is dropped
        # rather than clamped to something nobody chose.
        if widget.minimum() <= value <= widget.maximum():
            widget.setValue(value)
    elif isinstance(widget, (QCheckBox, QRadioButton, QGroupBox)):
        widget.setChecked(bool(value))
    elif isinstance(widget, QComboBox):
        index = widget.findText(str(value))
        if index >= 0:
            widget.setCurrentIndex(index)
        elif widget.isEditable():
            widget.setEditText(str(value))
    elif isinstance(widget, QLineEdit):
        if widget.text() != str(value):
            widget.setText(str(value))
            # What typing it and pressing Return would have done.
            widget.editingFinished.emit()
    elif isinstance(widget, QTabWidget):
        if 0 <= int(value) < widget.count():
            widget.setCurrentIndex(int(value))


def capture_settings(panel: QWidget) -> dict:
    """The value of every control in *panel*."""
    controls = _controls(panel)
    values = {}
    for key, widget in controls.items():
        try:
            values[key] = _read(widget)
        except Exception:
            logger.debug("could not read %s", key, exc_info=True)
    return {"form": _fingerprint(controls), "values": values}


def restore_settings(panel: QWidget, saved: dict) -> int:
    """Put saved values back onto *panel*. Returns how many controls were set."""
    controls = _controls(panel)
    if not saved or saved.get("form") != _fingerprint(controls):
        # The panel has changed since: positions no longer mean the same control.
        return 0
    done = 0
    for key, value in (saved.get("values") or {}).items():
        widget = controls.get(key)
        if widget is None or value is None:
            continue
        try:
            _write(widget, value)
            done += 1
        except Exception:
            logger.debug("could not restore %s", key, exc_info=True)
    return done


# -- the whole window ---------------------------------------------------------


def capture(app) -> dict:
    """Everything about *app* worth having back at the next launch."""
    viewer = app.viewer
    docks = getattr(app, "docks", {})
    shown = [identifier for identifier, dock in docks.items() if not dock.isHidden()]
    # Of several panels sharing a tab bar, the one in front is the one painted.
    front = [
        identifier for identifier in shown
        if not docks[identifier].visibleRegion().isEmpty()
    ]

    files: list[str] = []
    for layer in viewer.layers:
        source = source_of(layer)
        if source and source not in files:
            files.append(source)

    state = {
        "panels": shown,
        "front": front,
        "last_directory": str(app.last_directory) if app.last_directory else "",
        "files": files,
        "scale_bar": bool(viewer.scale_bar.visible),
        "settings": {},
    }
    if files:
        state["view"] = {**view_state(viewer), "ndisplay": int(viewer.dims.ndisplay)}

    for identifier, panel in getattr(app, "panels", {}).items():
        try:
            state["settings"][identifier] = capture_settings(panel)
        except Exception:
            logger.debug("could not read the %s panel", identifier, exc_info=True)

    experiment = getattr(app, "experiment_widget", None)
    if experiment is not None:
        folder = experiment.folder()
        state["experiment"] = {
            "folder": str(folder) if folder else "",
            "selected": experiment.selected_paths(),
        }
    return state


def restore(app, state: dict, reopen_files: bool = True, progress=None) -> None:
    """Put *state* back onto *app*. Each part stands alone: one failing is logged."""
    if not state:
        return
    say = progress or (lambda _message: None)

    def attempt(what: str, step) -> None:
        try:
            step()
        except Exception:
            logger.exception("could not restore %s", what)

    def panels() -> None:
        if "panels" not in state:
            return
        wanted = set(state["panels"])
        for identifier, dock in app.docks.items():
            dock.setVisible(identifier in wanted)
        for identifier in state.get("front", ()):
            dock = app.docks.get(identifier)
            if dock is not None and identifier in wanted:
                dock.raise_()

    def directory() -> None:
        last = state.get("last_directory")
        if last and Path(last).is_dir():
            app.last_directory = Path(last)

    reopened: list[str] = []

    def files() -> None:
        paths = session.existing(state.get("files")) if reopen_files else []
        if not paths:
            return
        say(f"Reopening {len(paths)} file(s) from last time…")
        # Marked while it happens, so a file that takes the viewer down is not
        # tried again at every launch (see ``session.load``).
        session.save({**state, session.REOPENING: True})
        try:
            if app.open_paths(paths):
                reopened.extend(paths)
        finally:
            session.save(state)

    def settings() -> None:
        for identifier, saved in (state.get("settings") or {}).items():
            panel = app.panels.get(identifier)
            if panel is not None:
                restore_settings(panel, saved)

    def view() -> None:
        app.viewer.scale_bar.visible = bool(state.get("scale_bar", True))
        saved = state.get("view")
        if not reopened or not saved:
            return
        if saved.get("ndisplay") == 3 and app.viewer.dims.ndim >= 3:
            app.toolbar.toggle_ndisplay()
        restore_view(app.viewer, saved)

    def experiment() -> None:
        saved = state.get("experiment") or {}
        panel = getattr(app, "experiment_widget", None)
        folder = saved.get("folder")
        if panel is None or not folder or not Path(folder).is_dir():
            return
        panel.restore(folder, saved.get("selected") or (), reopened)

    attempt("the open panels", panels)
    attempt("the last directory", directory)
    attempt("the open files", files)
    # After the files: the lists of layers the panels choose from are filled.
    attempt("the panel settings", settings)
    attempt("the view", view)
    attempt("the experiment folder", experiment)


class SessionKeeper(QObject):
    """Saves the session as the main window closes.

    On the close *event* rather than at quit: by the time the application is
    quitting napari has emptied the viewer, and there is nothing left to read.
    """

    def __init__(self, app, window: QWidget):
        super().__init__(window)
        self._app = app
        window.installEventFilter(self)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802 (Qt naming)
        if event.type() == QEvent.Close:
            self.save()
        return False

    def save(self) -> bool:
        try:
            return session.save(capture(self._app))
        except Exception:
            logger.exception("could not save the session")
            return False
