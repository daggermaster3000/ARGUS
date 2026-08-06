"""Launcher for Microscopy Viewer — the script the desktop shortcut points at.

It exists so the shortcut has a single stable target that works no matter where
the project lives: it puts the project directory on ``sys.path`` and then hands
over to :mod:`microscopy_viewer.__main__`.

Run directly for a console (useful for debugging)::

    python launch_viewer.py cells.ims

The desktop shortcut uses ``pythonw.exe`` instead so no console window appears.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> int:
    from microscopy_viewer.__main__ import main as viewer_main

    return viewer_main(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
