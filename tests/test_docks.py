"""The window plumbing, against a bare Qt window — no napari, no viewer.

Every check here stands for a number measured on the real window before any of
this existed: a minimum size of 1258x1517 on a 1392-px screen, nine docks open at
once, and a close button that deleted the panel behind every menu action pointing
at it.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Imported here, before any Qt: microscopy_viewer/__init__.py preloads torch's
# DLLs, which has to happen before Qt is initialised. Importing it after a
# QApplication exists access-violates on Windows, which as a test failure looks
# like "no output, exit code 9".
from microscopy_viewer import docks as dk  # noqa: E402

_failures: list[str] = []


def check(condition: bool, message: str) -> None:
    if condition:
        print(f"  ok   {message}", flush=True)
    else:
        print(f"  FAIL {message}", flush=True)
        _failures.append(message)


def _has_qt() -> bool:
    try:
        import qtpy  # noqa: F401
    except ImportError:
        return False
    return True


#: Every window these tests make, kept alive until the process exits.
#:
#: A QMainWindow held only by a local is deleted by Python when the function
#: returns, while Qt still owns its docks and their children — which on Windows
#: takes the whole process down with a stack-buffer-overrun and no output at all.
#: As a test failure it looks like the suite printing nothing and exiting 9.
_alive: list = []


#: The QApplication, held here for the life of the process.
#:
#: Returning it and letting the caller drop it is not enough: Python collects it,
#: Qt tears the application down underneath the widgets, and the process dies
#: with a stack-buffer-overrun before the first check runs.
_app = None


def _application():
    global _app
    from qtpy.QtWidgets import QApplication

    if _app is None:
        _app = QApplication.instance() or QApplication([])
    return _app


def _window():
    """A main window that outlives the test that made it."""
    from qtpy.QtWidgets import QMainWindow

    window = QMainWindow()
    _alive.append(window)
    return window


def _tall_panel(rows: int = 40):
    """A form as tall as the real panels: the thing that pinned the window open."""
    from qtpy.QtWidgets import QLabel, QSpinBox, QVBoxLayout, QWidget

    panel = QWidget()
    layout = QVBoxLayout(panel)
    for index in range(rows):
        layout.addWidget(QLabel(f"A fairly long label for row {index}"))
        layout.addWidget(QSpinBox())
    return panel


def test_a_scroll_area_lets_a_dock_shrink() -> None:
    print("a scroll area lets a dock shrink")

    if not _has_qt():
        print("  skip  qtpy is not installed")
        return

    _application()
    from qtpy.QtWidgets import QDockWidget

    panel = _tall_panel()
    _alive.append(panel)
    natural = panel.sizeHint().height()
    check(natural > 600, f"the panel really is tall ({natural} px)")

    window = _window()
    bare = QDockWidget("bare", window)
    other = _tall_panel()
    _alive.append(other)
    bare.setWidget(other)
    window.addDockWidget(dk_area(), bare)
    before = window.minimumSizeHint().height()

    window_two = _window()
    wrapped = QDockWidget("wrapped", window_two)
    wrapped.setWidget(dk.scrollable(panel))
    window_two.addDockWidget(dk_area(), wrapped)
    dk.relax(wrapped)
    after = window_two.minimumSizeHint().height()

    check(
        after < before,
        f"a window of wrapped docks has a smaller minimum than one of bare docks "
        f"({after} vs {before} px)",
    )
    check(
        after < natural,
        f"and smaller than the panel it holds, which is the whole point "
        f"({after} < {natural})",
    )
    check(
        dk.scrollable(_tall_panel()).widget() is not None,
        "the panel is still reachable through the scroll area",
    )

    # The panel asks for its own size, capped, rather than for nothing: a scroll
    # area with no sizeHint is handed a sliver with everything scrolled away.
    area = dk.scrollable(_tall_panel())
    check(
        area.sizeHint().height() > 80,
        f"a wrapped panel still asks for a usable height ({area.sizeHint().height()} px)",
    )
    screen = dk.screen_rect(area)
    if not screen.isEmpty():
        check(
            area.sizeHint().height() <= screen.height() // dk.SCREEN_FRACTION,
            f"but never more than half the screen "
            f"({area.sizeHint().height()} of {screen.height()})",
        )


def dk_area():
    from qtpy.QtCore import Qt

    return Qt.RightDockWidgetArea


def test_a_toolbar_scrolls_sideways() -> None:
    print("a one-row toolbar scrolls sideways")

    if not _has_qt():
        print("  skip  qtpy is not installed")
        return

    _application()
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import QHBoxLayout, QPushButton, QWidget

    row = QWidget()
    layout = QHBoxLayout(row)
    for index in range(14):
        layout.addWidget(QPushButton(f"Button {index}"))
    wanted = row.sizeHint().height()

    area = dk.scrollable(row, vertical=False)
    check(
        area.verticalScrollBarPolicy() == Qt.ScrollBarAlwaysOff,
        "it never scrolls vertically, so the row cannot be cut in half",
    )
    check(
        area.horizontalScrollBarPolicy() == Qt.ScrollBarAsNeeded,
        "and scrolls sideways when the window is narrower than the row",
    )
    check(
        wanted <= area.height() <= wanted + 30,
        f"its height is the row's, plus room for a scroll bar if the platform "
        f"needs one ({area.height()} for a {wanted}px row)",
    )


def test_the_window_is_fitted_to_its_screen() -> None:
    print("fitting the window to the screen")

    if not _has_qt():
        print("  skip  qtpy is not installed")
        return

    _application()
    window = _window()
    window.show()
    area = dk.screen_rect(window)
    if area.isEmpty():
        print("  skip  no screen geometry available")
        return

    # The state napari restores from a bigger monitor: larger than this screen
    # and placed off the left of it.
    window.resize(area.width() + 600, area.height() + 400)
    window.move(area.left() - 300, area.top() - 200)
    check(
        not area.contains(window.frameGeometry()),
        "a window restored from a larger monitor starts off the screen",
    )

    moved = dk.fit_to_screen(window)
    check(moved, "fitting it reports that it moved it")
    frame = window.frameGeometry()
    check(
        frame.width() <= area.width() and frame.height() <= area.height(),
        f"it is no larger than the screen ({frame.width()}x{frame.height()} "
        f"in {area.width()}x{area.height()})",
    )
    check(
        frame.left() >= area.left() and frame.top() >= area.top(),
        f"and no longer off the top or the left ({frame.left()}, {frame.top()})",
    )
    check(
        not dk.fit_to_screen(window),
        "a window that already fits is left alone",
    )
    window.close()


def test_a_dock_hides_rather_than_dying() -> None:
    print("a closed dock hides rather than dying")

    if not _has_qt():
        print("  skip  qtpy is not installed")
        return

    _application()
    from qtpy.QtWidgets import QDockWidget, QLabel

    window = _window()
    dock = QDockWidget("panel", window)
    dock.setWidget(QLabel("content"))
    window.addDockWidget(dk_area(), dock)

    # napari sets this attribute to remove_dock_widget, which destroys the dock
    # and the panel with it, leaving every menu action pointing at a dead object.
    dock.destroyOnClose = lambda: (_ for _ in ()).throw(AssertionError("destroyed"))
    dk.hide_rather_than_destroy(dock)
    check(dock.destroyOnClose == dock.hide, "closing it is hiding it")
    dock.destroyOnClose()
    check(dock.isHidden(), "and the dock is hidden, not deleted")
    check(dock.widget() is not None, "with its panel still there")


def test_the_panels_menu() -> None:
    print("the panels menu")

    if not _has_qt():
        print("  skip  qtpy is not installed")
        return

    _application()
    from qtpy.QtWidgets import QDockWidget, QLabel

    window = _window()
    made = []
    for name in ("One", "Two", "Three"):
        dock = QDockWidget(name, window)
        dock.setWidget(QLabel(name))
        window.addDockWidget(dk_area(), dock)
        dock.setVisible(False)
        made.append((name, dock))

    class _Host:
        """Stands in for napari's Window, whose main_menu is the menu bar."""

        main_menu = window.menuBar()

    menu = dk.panels_menu(_Host(), made)
    check(menu is not None, "a menu is made")
    check(
        [action.text() for action in menu.actions()] == ["One", "Two", "Three"],
        f"one entry per panel: {[a.text() for a in menu.actions()]}",
    )
    check(
        all(not action.isChecked() for action in menu.actions()),
        "unticked, because the panels start closed",
    )
    check(
        all(dock.destroyOnClose == dock.hide for _name, dock in made),
        "every dock in it hides on close rather than dying",
    )

    action = menu.actions()[1]
    action.trigger()
    check(not made[1][1].isHidden(), "triggering an entry opens its panel")
    check(action.isChecked(), "and ticks it")
    made[1][1].hide()
    check(
        not action.isChecked(),
        "hiding the dock by any other route unticks it too — the tick follows "
        "the dock, not the menu",
    )

    check(
        dk.panels_menu(object(), made) is None,
        "a host with no menu bar is reported rather than raised at",
    )


def test_reveal_scrolls_a_control_into_view() -> None:
    print("revealing a control inside a scroll area")

    if not _has_qt():
        print("  skip  qtpy is not installed")
        return

    _application()
    from qtpy.QtWidgets import QDockWidget, QLabel, QVBoxLayout, QWidget

    panel = QWidget()
    layout = QVBoxLayout(panel)
    rows = [QLabel(f"row {index}") for index in range(60)]
    for row in rows:
        layout.addWidget(row)
    target = rows[-1]

    window = _window()
    dock = QDockWidget("panel", window)
    area = dk.scrollable(panel)
    dock.setWidget(area)
    window.addDockWidget(dk_area(), dock)
    dock.setVisible(False)
    window.show()
    window.resize(400, 300)
    from qtpy.QtWidgets import QApplication

    # Forced small and laid out, so there is genuinely something to scroll. A
    # scroll area that has not been laid out has a scroll range of zero, and a
    # check against it passes without testing anything.
    area.setFixedSize(300, 200)
    dock.setVisible(True)
    QApplication.instance().processEvents()
    bar = area.verticalScrollBar()
    check(bar.maximum() > 0, f"the panel is taller than its scroll area ({bar.maximum()} px of range)")
    bar.setValue(0)
    dock.setVisible(False)

    dk.reveal(target)
    QApplication.instance().processEvents()
    check(not dock.isHidden(), "revealing a control opens the dock it is in")
    check(
        bar.value() > 0,
        f"and really scrolls to it rather than leaving it off screen "
        f"(at {bar.value()} of {bar.maximum()})",
    )
    window.close()


def test_emphasise_uses_a_stylesheet() -> None:
    print("a heading that survives napari's stylesheet")

    if not _has_qt():
        print("  skip  qtpy is not installed")
        return

    _application()
    from qtpy.QtWidgets import QLabel

    label = QLabel("Heading")
    dk.emphasise(label, size=18)
    sheet = label.styleSheet()
    check("font-size: 18px" in sheet, f"the size is in the stylesheet: {sheet!r}")
    check("bold" in sheet, "and so is the weight")


def main() -> int:
    for test in (
        test_a_scroll_area_lets_a_dock_shrink,
        test_a_toolbar_scrolls_sideways,
        test_the_window_is_fitted_to_its_screen,
        test_a_dock_hides_rather_than_dying,
        test_the_panels_menu,
        test_reveal_scrolls_a_control_into_view,
        test_emphasise_uses_a_stylesheet,
    ):
        test()
        print()

    if _failures:
        print(f"{len(_failures)} check(s) failed:")
        for failure in _failures:
            print(f"  - {failure}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
