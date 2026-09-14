"""Checks for the freeze watchdog and the process that plays the animation.

The watchdog itself needs no display: it is a timestamp, a rule and a pipe. The
child process is started for real, offscreen, because the thing worth proving is
that it takes its commands and that it dies with its parent.

Run with::

    python tests/test_busy.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from microscopy_viewer import busy  # noqa: E402

_failures: list[str] = []


def check(condition: bool, message: str) -> None:
    if condition:
        print(f"  ok   {message}")
    else:
        print(f"  FAIL {message}")
        _failures.append(message)


class _FakeOverlay:
    """Records what the watchdog thread would have told the child process."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.visible = False

    def start(self) -> bool:
        return True

    def show(self, geometry, message) -> None:
        if not self.visible:
            self.calls.append(f"show {geometry} {message}")
            self.visible = True

    def hide(self) -> None:
        if self.visible:
            self.calls.append("hide")
            self.visible = False

    def stop(self) -> None:
        self.calls.append("stop")


def test_rule() -> None:
    print("when the animation belongs on screen")
    now = 1000.0
    check(not busy.is_stalled(now, True, 1.2, now), "a heartbeat just taken is not a freeze")
    check(not busy.is_stalled(now - 1.0, True, 1.2, now), "nor is a one-second pause")
    check(busy.is_stalled(now - 1.2, True, 1.2, now), "silence for the full delay is")
    check(busy.is_stalled(now - 60.0, True, 1.2, now), "and so is a minute of it")
    # A window nobody is looking at gets no overlay: minimised, hidden, or the
    # user has alt-tabbed to something else entirely.
    check(not busy.is_stalled(now - 60.0, False, 1.2, now), "but not when the window is not on screen")


def test_heartbeat() -> None:
    print("the shared record between the two threads")
    heart = busy.Heartbeat()
    heart.update((10, 20, 300, 400), True)
    beat, geometry, showing = heart.read()
    check(geometry == (10, 20, 300, 400), f"the geometry survives the hand-off ({geometry})")
    check(showing, "so does whether the window was showing")

    # The point of keeping the geometry: once the main thread is stuck it can no
    # longer be asked where the window is, so the last answer has to do.
    time.sleep(0.05)
    stale_beat, stale_geometry, _ = heart.read()
    check(stale_beat == beat, "a heartbeat that never comes leaves the timestamp alone")
    check(stale_geometry == geometry, "and the animation still knows where to go")

    # Written from one thread, read from another, which is the whole reason for
    # the lock: this must not tear or raise.
    def hammer() -> None:
        for index in range(2000):
            heart.update((index, index, index, index), index % 2 == 0)

    writers = [threading.Thread(target=hammer) for _ in range(4)]
    for writer in writers:
        writer.start()
    for _ in range(2000):
        _beat, geometry, _showing = heart.read()
        if len(set(geometry)) != 1:
            check(False, f"a half-written record was read ({geometry})")
            break
    for writer in writers:
        writer.join()
    else:
        check(True, "concurrent updates are never read half-written")


def test_watchdog_drives_the_overlay() -> None:
    print("the watchdog thread")
    watchdog = busy.BusyWatchdog(window=None, message="Working…", after=0.2)
    fake = _FakeOverlay()
    watchdog._overlay = fake

    # Started by hand rather than through start(), which would want a QTimer and
    # therefore a QApplication. The heartbeat is fed directly instead.
    watchdog._heart.update((100, 50, 800, 600), True)
    thread = threading.Thread(target=watchdog._watch, daemon=True)
    thread.start()

    time.sleep(0.15)
    check(not fake.visible, "a live heartbeat keeps the animation away")

    # Stop feeding it: this is what a blocked main thread looks like.
    time.sleep(0.6)
    check(fake.visible, f"silence brings it up ({fake.calls})")
    check(
        fake.calls and fake.calls[0].startswith("show (100, 50, 800, 600)"),
        f"over the window's last known place ({fake.calls[0] if fake.calls else None})",
    )
    shown = len([call for call in fake.calls if call.startswith("show")])
    time.sleep(0.4)
    check(
        len([call for call in fake.calls if call.startswith("show")]) == shown,
        "and it is not told again every time round the loop",
    )

    # The freeze ends: the timer starts firing again and the animation comes
    # down. Beaten in a loop, because one lone heartbeat followed by more silence
    # is simply the next freeze — which is exactly what should happen.
    for _ in range(10):
        watchdog._heart.update((100, 50, 800, 600), True)
        time.sleep(0.05)
    check(not fake.visible, f"a returning heartbeat takes it down ({fake.calls})")

    watchdog._stop.set()
    thread.join(timeout=2)
    check(not thread.is_alive(), "and the thread stops when it is asked to")


def test_child_process() -> None:
    print("the process that holds the animation")
    environment = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    child = subprocess.Popen(
        [sys.executable, "-m", "microscopy_viewer.busy_window"],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        cwd=str(Path(__file__).resolve().parent.parent),
        env=environment,
    )
    try:
        time.sleep(2.0)  # Qt and the GIF take a moment to come up
        check(child.poll() is None, "it starts and stays up")

        for line in (b"show 0 0 800 600 Reading a stack\n", b"hide\n", b"show 10 10 400 300 Working\n"):
            child.stdin.write(line)
        child.stdin.flush()
        time.sleep(0.6)
        check(child.poll() is None, "commands do not upset it")

        # Nonsense must be ignored rather than fatal: the pipe is written from a
        # watchdog thread during a freeze, which is no place for an exception.
        child.stdin.write(b"show not a rectangle\nwaffle\n\n")
        child.stdin.flush()
        time.sleep(0.5)
        check(child.poll() is None, "and neither does nonsense")

        # Closing the pipe is what happens when the viewer dies, crash included.
        child.stdin.close()
        child.wait(timeout=10)
        check(child.returncode == 0, f"it exits when its parent's pipe closes ({child.returncode})")
    finally:
        if child.poll() is None:
            child.kill()
            check(False, "the child had to be killed")


def main() -> int:
    for test in (test_rule, test_heartbeat, test_watchdog_drives_the_overlay, test_child_process):
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
