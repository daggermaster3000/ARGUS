"""Launcher for the Streamlit explorer — the script its desktop shortcut points at.

Double-clicking the shortcut twice must not start two servers, and a second
click while the explorer is already running should just bring it back up. So:

* The explorer always listens on its own fixed port (:data:`PORTS`). If
  something answers Streamlit's health check there, no server is started: a
  browser tab is opened to the running one.
* Otherwise one server is started in the background — no console window, output
  to a log file — and the tab is opened once it answers.
* A lock file makes two launches a split second apart take turns, so both do
  not see "nothing running" and start a server each. The fixed port is the
  second guard: a second server could not bind it anyway.
* Streamlit's own "open a browser" is off, so exactly one tab opens per launch.

A workbook or report folder given on the command line (or dropped on the
shortcut) is passed in the URL, so it opens even when the server was already
running. ``--stop`` ends the background server.

Run directly::

    python launch_explorer.py                  # the region explorer
    python launch_explorer.py report.xlsx      # …opening this workbook
    python launch_explorer.py --simple         # the plain one
    python launch_explorer.py --stop           # stop the background server(s)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import webbrowser
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

#: App script -> the port it always gets. Fixed, so a running one can be found.
APPS = {
    "region": PROJECT_ROOT / "apps" / "region_explorer.py",
    "simple": PROJECT_ROOT / "apps" / "simple_explorer.py",
}
PORTS = {"region": 8765, "simple": 8766}
#: Query parameter carrying a workbook path to an already running app.
WORKBOOK_PARAM = "workbook"
#: How long a new server may take to answer before giving up.
START_TIMEOUT_S = 90.0
#: A lock older than this belongs to a launch that died; take it over.
STALE_LOCK_S = 120.0


def _state_dir() -> Path:
    from microscopy_viewer.runtime import app_data_dir

    folder = app_data_dir()
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def url_of(app: str, workbook: str | None = None) -> str:
    base = f"http://localhost:{PORTS[app]}/"
    if workbook:
        return base + "?" + urllib.parse.urlencode({WORKBOOK_PARAM: workbook})
    return base


def is_running(app: str, timeout: float = 1.0) -> bool:
    """Whether a Streamlit server answers its health check on *app*'s port."""
    try:
        with urllib.request.urlopen(f"http://localhost:{PORTS[app]}/_stcore/health",
                                    timeout=timeout) as reply:
            return reply.status == 200 and reply.read().strip() == b"ok"
    except Exception:
        return False


class _Lock:
    """An exclusive lock file, so two launches in the same second take turns."""

    def __init__(self, path: Path):
        self.path = path

    def __enter__(self):
        deadline = time.monotonic() + START_TIMEOUT_S + 10
        while True:
            try:
                handle = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(handle, str(os.getpid()).encode())
                os.close(handle)
                return self
            except FileExistsError:
                try:
                    if time.time() - self.path.stat().st_mtime > STALE_LOCK_S:
                        self.path.unlink(missing_ok=True)
                        continue
                except FileNotFoundError:
                    continue
                if time.monotonic() > deadline:
                    raise TimeoutError(f"another launch is holding {self.path}")
                time.sleep(0.2)

    def __exit__(self, *_exc):
        self.path.unlink(missing_ok=True)


def _server_command(app: str) -> list[str]:
    python = Path(sys.executable)
    if python.name.lower() == "pythonw.exe":
        # Streamlit writes to the console streams; pythonw has none. The server
        # gets python.exe, started without a window below.
        python = python.with_name("python.exe")
    return [
        str(python), "-m", "streamlit", "run", str(APPS[app]),
        "--server.port", str(PORTS[app]),
        "--server.headless", "true",          # no second tab: the launcher opens it
        "--browser.gatherUsageStats", "false",
    ]


def start_server(app: str) -> subprocess.Popen:
    """Start *app*'s server in the background, detached from this launcher."""
    state = _state_dir()
    log = open(state / f"explorer_{app}.log", "ab")
    kwargs: dict = {"stdout": log, "stderr": subprocess.STDOUT, "stdin": subprocess.DEVNULL,
                    "cwd": str(PROJECT_ROOT)}
    if os.name == "nt":
        kwargs["creationflags"] = 0x08000000 | 0x00000200  # CREATE_NO_WINDOW | NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True  # outlives the launcher and its terminal
    process = subprocess.Popen(_server_command(app), **kwargs)
    (state / f"explorer_{app}.json").write_text(
        json.dumps({"pid": process.pid, "port": PORTS[app]}), encoding="utf-8")
    return process


def wait_until_running(app: str, process: subprocess.Popen | None = None,
                       timeout: float = START_TIMEOUT_S) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if is_running(app, timeout=0.5):
            return True
        if process is not None and process.poll() is not None:
            return False  # it exited: port taken by something else, or a crash
        time.sleep(0.3)
    return False


def ensure_running(app: str) -> tuple[bool, bool]:
    """``(running, started)``: make sure exactly one server for *app* is up."""
    with _Lock(_state_dir() / f"explorer_{app}.lock"):
        if is_running(app):
            return True, False
        process = start_server(app)
        return wait_until_running(app, process), True


def stop(app: str) -> bool:
    """End the background server this launcher started for *app*."""
    record = _state_dir() / f"explorer_{app}.json"
    try:
        pid = int(json.loads(record.read_text(encoding="utf-8"))["pid"])
    except (OSError, ValueError, KeyError):
        return False
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
        else:
            import signal

            os.kill(pid, signal.SIGTERM)
    except (OSError, ProcessLookupError):
        pass
    record.unlink(missing_ok=True)
    return True


def _report(message: str) -> None:
    """Say something even when launched without a console (pythonw, the .app)."""
    if sys.stderr is not None:
        print(message, file=sys.stderr)
    try:
        with open(_state_dir() / "explorer_launcher.log", "a", encoding="utf-8") as log:
            log.write(time.strftime("%Y-%m-%d %H:%M:%S ") + message + "\n")
    except OSError:
        pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("workbook", nargs="?", help="a workbook or report folder to open")
    parser.add_argument("--simple", action="store_true", help="the plain explorer instead of the region explorer")
    parser.add_argument("--stop", action="store_true", help="stop the background server(s) and exit")
    parser.add_argument("--no-browser", action="store_true", help="start the server but open no tab")
    args = parser.parse_args([a for a in (argv if argv is not None else sys.argv[1:])
                              if not a.startswith("-psn_")])

    if args.stop:
        stopped = [app for app in APPS if stop(app)]
        _report(f"stopped: {', '.join(stopped) or 'nothing was running'}")
        return 0

    app = "simple" if args.simple else "region"
    workbook = str(Path(args.workbook).expanduser().resolve()) if args.workbook else None
    try:
        running, started = ensure_running(app)
    except TimeoutError as exc:
        _report(str(exc))
        return 1
    if not running:
        _report(f"the {app} explorer did not start; see {_state_dir() / f'explorer_{app}.log'}")
        return 1
    _report(("started" if started else "already running") + f": {url_of(app)}")
    if not args.no_browser:
        webbrowser.open(url_of(app, workbook), new=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
