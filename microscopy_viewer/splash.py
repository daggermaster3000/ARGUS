"""The startup splash: an animation shown while the window is being built.

napari puts its main window on screen the moment the viewer object exists, and
the several seconds after that go on building the docks, importing what each
panel needs and waking the GPU. What is on screen for those seconds is an empty
white rectangle, which reads as a hung application rather than a loading one.

So the window is kept hidden until it is ready and this is shown over the gap
instead. Nothing here is required for the viewer to work: any failure means the
splash is skipped and the window opens the way it always did.

The animation is :class:`microscopy_viewer.loader.Loader`, which paints itself;
there is no image file to be missing and no codec for Qt to lack.

The animation only advances while something is processing events, and building a
viewer is one long blocking call, so the caller drives it: :meth:`Splash.pump`
puts a line of text up and lets Qt paint. That makes the splash a progress
report as much as a picture — the panel being built is named as it is built.
"""

from __future__ import annotations

from .utils import get_logger

logger = get_logger("splash")

#: Width of the splash, and the side of the animation drawn in the middle of it.
_WIDTH = 420
_LOADER = 230

#: Padding around the animation, and the height left under it for the caption.
_PAD = 14
_CAPTION = 46

#: Splash colours: the animation's own near-black, with pale text.
_BACKGROUND = "#101014"
_TEXT = "#e8e8ea"
_DIM = "#9a9aa2"


class Splash:
    """A frameless window showing the loading animation while the viewer is built.

    Not a ``QSplashScreen``: that class paints a single pixmap, so an animation
    means pushing every frame into it by hand, and it has no room for the caption.
    A plain frameless widget does both and closes the same way.
    """

    def __init__(self, title: str = "Microscopy Viewer") -> None:
        from qtpy.QtCore import Qt
        from qtpy.QtWidgets import QLabel, QVBoxLayout, QWidget

        self._loader = None
        # Never takes focus: as a busy overlay it appears while somebody is typing
        # somewhere else, and stealing the keyboard for a picture is unforgivable.
        self.widget = QWidget(
            None,
            Qt.SplashScreen
            | Qt.FramelessWindowHint
            | Qt.WindowStaysOnTopHint
            | Qt.WindowDoesNotAcceptFocus,
        )
        self.widget.setAttribute(Qt.WA_DeleteOnClose, False)
        self.widget.setStyleSheet(f"background: {_BACKGROUND};")

        layout = QVBoxLayout(self.widget)
        layout.setContentsMargins(_PAD, _PAD, _PAD, _PAD)
        layout.setSpacing(8)

        self.widget.setFixedWidth(_WIDTH + 2 * _PAD)
        self._load_animation(layout)

        self._title = QLabel(title, self.widget)
        self._title.setAlignment(Qt.AlignCenter)
        self._title.setStyleSheet(f"color: {_TEXT}; font-size: 13px; font-weight: 600;")
        layout.addWidget(self._title)

        self._message = QLabel("Starting…", self.widget)
        self._message.setAlignment(Qt.AlignCenter)
        self._message.setStyleSheet(f"color: {_DIM}; font-size: 11px;")
        layout.addWidget(self._message)

        self.widget.adjustSize()
        self._centre()

    # -- setup ----------------------------------------------------------------

    def _load_animation(self, layout) -> None:
        """Put the animation in, or leave the captions alone if it cannot be made."""
        from qtpy.QtCore import Qt

        try:
            from .loader import Loader

            self._loader = Loader(_LOADER, self.widget)
            layout.addWidget(self._loader, alignment=Qt.AlignHCenter)
            self._loader.start()
        except Exception:
            logger.info("could not build the loading animation; showing the caption alone",
                        exc_info=True)
            self._loader = None

    def _centre(self) -> None:
        """Put the splash in the middle of the screen the cursor is on."""
        from qtpy.QtGui import QCursor, QGuiApplication

        screen = QGuiApplication.screenAt(QCursor.pos()) or QGuiApplication.primaryScreen()
        if screen is None:
            return
        area = screen.availableGeometry()
        size = self.widget.sizeHint()
        self.widget.move(
            area.center().x() - size.width() // 2,
            area.center().y() - size.height() // 2,
        )

    def place_over(self, x: int, y: int, width: int, height: int) -> None:
        """Centre the splash on a rectangle in screen coordinates.

        Used by the busy overlay, which is told where the viewer's window is and
        has to sit on top of it rather than in the middle of whichever screen it
        happens to have started on.
        """
        size = self.widget.sizeHint()
        self.widget.move(
            int(x + width // 2 - size.width() // 2),
            int(y + height // 2 - size.height() // 2),
        )

    def set_message(self, message: str) -> None:
        """The line under the animation, without processing events."""
        self._message.setText(message)

    # -- during the build -----------------------------------------------------

    def show(self) -> "Splash":
        # Restarted rather than resumed: an overlay shown again should open on
        # the animation, not on the frame it happened to stop at.
        if self._loader is not None:
            self._loader.start()
        self.widget.show()
        self.widget.raise_()
        self.pump()
        return self

    def hide(self) -> None:
        """Take the splash down but keep it ready to be shown again."""
        if self._loader is not None:
            self._loader.stop()
        self.widget.hide()

    def pump(self, message: str | None = None) -> None:
        """Show *message* and let Qt paint a frame.

        Building the viewer never returns to the event loop, so without this the
        splash would be one frozen frame — the very thing it exists to replace.
        """
        from qtpy.QtWidgets import QApplication

        if message is not None:
            self._message.setText(message)
        if self._loader is not None:
            # No event loop is turning the timer during the build; ask for a
            # frame here, and the animation draws wherever the clock says it is.
            self._loader.update()
        application = QApplication.instance()
        if application is not None:
            application.processEvents()

    # -- done -----------------------------------------------------------------

    def finish(self, window=None) -> None:
        """Close the splash and bring *window* forward in its place."""
        if self._loader is not None:
            self._loader.stop()
        self.widget.close()
        if window is not None:
            try:
                window.raise_()
                window.activateWindow()
            except Exception:  # a Qt object that has already gone
                logger.debug("could not raise the main window over the splash", exc_info=True)
        self.pump()


def start(title: str = "Microscopy Viewer") -> Splash | None:
    """A shown :class:`Splash`, or ``None`` when one cannot be made.

    Needs a ``QApplication``; the caller makes napari create one first. Any
    failure here is logged and swallowed — a viewer that will not start because
    its loading animation would not play is worse than no animation.
    """
    try:
        from qtpy.QtWidgets import QApplication

        if QApplication.instance() is None:
            logger.debug("no QApplication yet; skipping the splash")
            return None
        return Splash(title).show()
    except Exception:
        logger.debug("could not build the splash", exc_info=True)
        return None
