"""What the viewer looked like when it was last closed, kept for the next launch.

The open panels, each panel's settings, the experiment folder with its selected
samples, the files that were on screen and where the view was. One JSON file in
the per-user data directory, written when the window closes and read back by
:func:`microscopy_viewer.app.launch`.

No Qt here: reading the widgets is :mod:`microscopy_viewer.widgets.session_state`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from .utils import get_logger

logger = get_logger("session")

#: Bumped when the layout of the file changes; an older file is ignored.
SESSION_VERSION = 1

#: Set in the file while the last session's images are being reopened. Still
#: being there at the next launch means that reopening took the viewer down, so
#: the images are left out rather than doing the same again on every start.
REOPENING = "reopening"


def session_file() -> Path:
    from .runtime import app_data_dir

    return app_data_dir() / "session.json"


def load(path: Path | None = None) -> dict:
    """The saved session, or ``{}`` when there is none that can be used."""
    target = Path(path) if path is not None else session_file()
    try:
        state = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(state, dict) or state.get("version") != SESSION_VERSION:
        return {}
    if state.pop(REOPENING, False):
        logger.warning("the last launch stopped while reopening its images; not reopening them")
        state["files"] = []
    return state


def save(state: dict, path: Path | None = None) -> bool:
    """Write *state* for the next launch. Returns whether it was written.

    Written beside the file and moved over it, so a launch never reads half of
    one.
    """
    target = Path(path) if path is not None else session_file()
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + ".part")
        partial.write_text(
            json.dumps({**state, "version": SESSION_VERSION}, indent=1, default=_plain),
            encoding="utf-8",
        )
        os.replace(partial, target)
        return True
    except (OSError, TypeError, ValueError):
        logger.debug("could not save the session", exc_info=True)
        return False


def existing(paths) -> list[str]:
    """Those of *paths* still there, in order and once each.

    A sample on a share that is not mounted today is skipped rather than
    reported: nobody asked for it this time.
    """
    kept: list[str] = []
    for path in paths or ():
        try:
            text = str(path)
            if text and text not in kept and Path(text).exists():
                kept.append(text)
        except (OSError, TypeError, ValueError):
            continue
    return kept


def _plain(value):
    """numpy scalars and paths, as something JSON can hold."""
    if isinstance(value, Path):
        return str(value)
    item = getattr(value, "item", None)
    if callable(item):
        return item()
    return str(value)
