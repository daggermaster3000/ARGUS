"""The child process that plays the animation while the viewer is frozen.

A Qt application paints from its main thread and nowhere else, so while that
thread is inside a long call — reading a stack off the NAS, running a Cellpose
model, writing a deck — nothing in that process can draw a frame. That is the
whole reason a frozen window goes white and Windows stamps *(Not Responding)*
across the title bar: there is nobody left to repaint it.

So the animation lives here, in a second process that is doing nothing else. The
viewer sends it one-line commands on stdin and it shows or hides a frameless
always-on-top window over the viewer's own. When the viewer exits, for any reason
including a crash, stdin closes and this exits with it: it can never outlive the
window it belongs to.

Run as::

    python -m microscopy_viewer.busy_window

Commands, one per line:

``show <x> <y> <w> <h> <message>``
    Cover the given screen rectangle and say *message*.
``hide``
    Take it down.
``quit``
    Exit.
"""

from __future__ import annotations

import queue
import sys
import threading

#: How often the Qt side looks for commands the reader thread has queued. Small
#: enough that the animation appears the moment the viewer asks for it.
_POLL_MS = 40

#: How often the elapsed counter under the animation ticks over.
_ELAPSED_MS = 500


def _reader(commands: "queue.Queue[str]") -> None:
    """Feed stdin lines to the Qt side, then a quit when the pipe closes."""
    try:
        for line in sys.stdin:
            commands.put(line.strip())
    except Exception:
        pass
    commands.put("quit")


def main() -> int:
    from qtpy.QtCore import QTimer
    from qtpy.QtWidgets import QApplication

    from .splash import Splash

    application = QApplication(sys.argv[:1])
    splash = Splash("Microscopy Viewer")
    splash.widget.hide()

    commands: "queue.Queue[str]" = queue.Queue()
    threading.Thread(target=_reader, args=(commands,), daemon=True).start()

    state = {"since": 0.0, "message": ""}

    def elapsed() -> None:
        """Count the seconds up, so a long wait is visibly progressing."""
        if not splash.widget.isVisible():
            return
        state["since"] += _ELAPSED_MS / 1000.0
        splash.set_message(f"{state['message']} ({state['since']:.0f} s)")

    def drain() -> None:
        while True:
            try:
                line = commands.get_nowait()
            except queue.Empty:
                return

            command, _, rest = line.partition(" ")
            if command == "quit":
                application.quit()
                return
            if command == "hide":
                splash.hide()
                continue
            if command != "show":
                continue

            parts = rest.split(" ", 4)
            try:
                x, y, width, height = (int(float(value)) for value in parts[:4])
            except (IndexError, ValueError):
                continue
            state["since"] = 0.0
            state["message"] = parts[4] if len(parts) > 4 and parts[4] else "Working…"
            splash.set_message(state["message"])
            splash.place_over(x, y, width, height)
            splash.show()

    poll = QTimer()
    poll.timeout.connect(drain)
    poll.start(_POLL_MS)

    ticker = QTimer()
    ticker.timeout.connect(elapsed)
    ticker.start(_ELAPSED_MS)

    return int(application.exec_() if hasattr(application, "exec_") else application.exec())


if __name__ == "__main__":
    raise SystemExit(main())
