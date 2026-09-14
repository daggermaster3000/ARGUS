"""Command line entry point: ``python -m microscopy_viewer [files...]``.

Files given as arguments are opened at startup. This is also how drag-and-drop
onto the desktop shortcut works: Windows appends the dropped paths to the
shortcut's argument list.
"""

from __future__ import annotations

import argparse
import logging
import sys
import traceback
from pathlib import Path

from .utils import get_logger, log_file, setup_logging


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="microscopy_viewer",
        description="Open microscopy images (.ims, TIFF, OME-TIFF, OME-Zarr) in a customised napari viewer.",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="image files or folders to open at startup",
    )
    parser.add_argument("--verbose", action="store_true", help="log debug detail")
    parser.add_argument(
        "--warn-shimmed-plugins",
        action="store_true",
        help="keep napari's modal plugin-engine warning at startup (it blocks loading until dismissed)",
    )
    parser.add_argument(
        "--no-splash",
        action="store_true",
        help="open the window straight away instead of covering the build with the loading animation",
    )
    parser.add_argument(
        "--no-busy-overlay",
        action="store_true",
        help="do not cover the window with the animation while it is busy",
    )
    parser.add_argument(
        "--version", action="store_true", help="print the version and exit"
    )
    return parser.parse_args(argv)


def _fatal(exc: BaseException) -> None:
    """Report a startup failure even when launched without a console.

    The shortcut runs ``pythonw.exe``, which has no stderr, so a traceback would
    otherwise vanish. The log file always gets it; a message box is attempted too.
    """
    logger = get_logger()
    logger.critical("startup failed", exc_info=exc)
    detail = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    message = (
        f"Microscopy Viewer could not start.\n\n{type(exc).__name__}: {exc}\n\n"
        f"Full details were written to:\n{log_file()}"
    )
    try:
        from qtpy.QtWidgets import QApplication, QMessageBox

        app = QApplication.instance() or QApplication(sys.argv[:1])
        box = QMessageBox(QMessageBox.Critical, "Microscopy Viewer", message)
        box.setDetailedText(detail)
        box.exec_()
    except Exception:
        if sys.stderr is not None:
            sys.stderr.write(message + "\n\n" + detail)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(sys.argv[1:] if argv is None else argv))

    if args.version:
        from . import __version__

        print(f"Microscopy Viewer {__version__}")
        return 0

    logger = setup_logging(logging.DEBUG if args.verbose else logging.INFO)
    logger.info("starting Microscopy Viewer with %d argument path(s)", len(args.paths))

    try:
        from .app import launch

        # Quoted drops can arrive with stray whitespace; normalise before loading.
        paths = [Path(p.strip('"').strip()) for p in args.paths if p.strip()]
        launch(
            paths,
            block=True,
            suppress_plugin_warning=not args.warn_shimmed_plugins,
            splash=not args.no_splash,
            busy_overlay=not args.no_busy_overlay,
        )
    except BaseException as exc:  # noqa: BLE001 - last line of defence for a GUI app
        _fatal(exc)
        return 1
    logger.info("Microscopy Viewer closed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
