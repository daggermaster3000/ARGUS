"""The startup splash: an animation shown while the window is being built.

napari puts its main window on screen the moment the viewer object exists, and
the several seconds after that go on building the docks, importing what each
panel needs and waking the GPU. What is on screen for those seconds is an empty
white rectangle, which reads as a hung application rather than a loading one.

So the window is kept hidden until it is ready and this is shown over the gap
instead. Nothing here is required for the viewer to work: a missing animation, a
Qt build without the GIF plugin, or any other failure means the splash is skipped
and the window opens the way it always did.

The animation only advances while something is processing events, and building a
viewer is one long blocking call, so the caller drives it: :meth:`Splash.pump`
puts a line of text up and lets Qt paint. That makes the splash a progress
report as much as a picture — the panel being built is named as it is built.
"""

from __future__ import annotations

from pathlib import Path

from .utils import get_logger

logger = get_logger("splash")

#: The animation, inside the package so an installed copy has it too.
ANIMATION = Path(__file__).resolve().parent / "resources" / "loading.gif"

#: Width the animation is drawn at. The file is 320 px across; blown up much
#: past that a GIF turns to mush, and a splash is not meant to fill the screen.
_WIDTH = 420

#: Padding around the animation, and the height left under it for the caption.
_PAD = 14
_CAPTION = 46

#: Splash colours: near-black behind the animation with pale text, which is what
#: a GIF with a dark background sits on without a bright border around it.
_BACKGROUND = "#141416"
_TEXT = "#e8e8ea"
_DIM = "#9a9aa2"


class Splash:
    """A frameless window playing :data:`ANIMATION` while the viewer is built.

    Not a ``QSplashScreen``: that class paints a single pixmap, so an animation
    means pushing every frame into it by hand, and it has no room for the caption.
    A plain frameless widget does both and closes the same way.
    """

    def __init__(self, title: str = "Microscopy Viewer") -> None:
        from qtpy.QtCore import Qt
        from qtpy.QtGui import QMovie
        from qtpy.QtWidgets import QLabel, QVBoxLayout, QWidget

        self._movie: QMovie | None = None
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

        self._picture = QLabel(self.widget)
        self._picture.setAlignment(Qt.AlignCenter)
        layout.addWidget(self._picture)

        self._title = QLabel(title, self.widget)
        self._title.setAlignment(Qt.AlignCenter)
        self._title.setStyleSheet(f"color: {_TEXT}; font-size: 13px; font-weight: 600;")
        layout.addWidget(self._title)

        self._message = QLabel("Starting…", self.widget)
        self._message.setAlignment(Qt.AlignCenter)
        self._message.setStyleSheet(f"color: {_DIM}; font-size: 11px;")
        layout.addWidget(self._message)

        self._load_animation()
        self.widget.adjustSize()
        self._centre()

    # -- setup ----------------------------------------------------------------

    def _load_animation(self) -> None:
        """Start the GIF, or leave a blank panel of the same size behind."""
        from qtpy.QtCore import QSize
        from qtpy.QtGui import QMovie

        height = int(_WIDTH * 9 / 16)
        if not ANIMATION.is_file():
            logger.info("no splash animation at %s; showing the caption alone", ANIMATION)
            self._picture.setFixedSize(_WIDTH, 1)
            return

        movie = QMovie(str(ANIMATION))
        if not movie.isValid():
            # A Qt build without the GIF image plugin. Not worth a warning box.
            logger.info("Qt cannot play %s; showing the caption alone", ANIMATION.name)
            self._picture.setFixedSize(_WIDTH, 1)
            return

        movie.jumpToFrame(0)
        native = movie.currentPixmap().size()
        if native.width() > 0:
            height = int(round(_WIDTH * native.height() / native.width()))
        movie.setScaledSize(QSize(_WIDTH, height))
        self._picture.setFixedSize(_WIDTH, height)
        self._picture.setMovie(movie)
        movie.start()
        self._movie = movie

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
        if self._movie is not None:
            self._movie.start()
        self.widget.show()
        self.widget.raise_()
        self.pump()
        return self

    def hide(self) -> None:
        """Take the splash down but keep it ready to be shown again."""
        if self._movie is not None:
            self._movie.stop()
        self.widget.hide()

    def pump(self, message: str | None = None) -> None:
        """Show *message* and let Qt paint a frame.

        Building the viewer never returns to the event loop, so without this the
        splash would be one frozen frame — the very thing it exists to replace.
        """
        from qtpy.QtWidgets import QApplication

        if message is not None:
            self._message.setText(message)
        application = QApplication.instance()
        if application is not None:
            application.processEvents()

    # -- done -----------------------------------------------------------------

    def finish(self, window=None) -> None:
        """Close the splash and bring *window* forward in its place."""
        if self._movie is not None:
            self._movie.stop()
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
