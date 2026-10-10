"""Checks for the saved session: what is written when the viewer closes.

Reading the window into it is exercised by ``smoke_gui.py``; this is the part
that needs no Qt.

Run with::

    python tests/test_session.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from microscopy_viewer import session  # noqa: E402

_failures: list[str] = []


def check(condition: bool, message: str) -> None:
    if condition:
        print(f"  ok   {message}")
    else:
        print(f"  FAIL {message}")
        _failures.append(message)


def test_round_trip() -> None:
    print("saving and loading")
    with tempfile.TemporaryDirectory() as directory:
        target = Path(directory) / "nested" / "session.json"
        check(session.load(target) == {}, "no file is an empty session")

        state = {
            "panels": ["experiment", "segmentation"],
            "files": ["/data/a.ims"],
            "view": {"center": (0.0, 1.5, 2.5), "zoom": 2.0},
            "last_directory": Path("/data"),
        }
        check(session.save(state, target), "saved, making the directory")
        loaded = session.load(target)
        check(loaded["panels"] == state["panels"], "the panels come back")
        check(loaded["view"]["center"] == [0.0, 1.5, 2.5], "tuples come back as lists")
        check(loaded["last_directory"] == str(Path("/data")), "a path is written as text")
        check(not list(target.parent.glob("*.part")), "nothing half-written is left behind")

        target.write_text("{not json", encoding="utf-8")
        check(session.load(target) == {}, "a damaged file is an empty session")
        target.write_text(json.dumps({"version": -1, "panels": ["x"]}), encoding="utf-8")
        check(session.load(target) == {}, "so is one from another version")
        target.write_text(json.dumps([1, 2]), encoding="utf-8")
        check(session.load(target) == {}, "and one that is not a session at all")


def test_reopening_guard() -> None:
    print("a launch that died reopening its images")
    with tempfile.TemporaryDirectory() as directory:
        target = Path(directory) / "session.json"
        session.save({"panels": ["experiment"], "files": ["/data/a.ims"],
                      session.REOPENING: True}, target)
        loaded = session.load(target)
        check(loaded["files"] == [], "does not reopen them next time")
        check(loaded["panels"] == ["experiment"], "but keeps the rest")
        check(session.REOPENING not in loaded, "and the mark is not passed on")


def test_existing() -> None:
    print("files that are still there")
    with tempfile.TemporaryDirectory() as directory:
        here = Path(directory) / "a.ims"
        here.write_bytes(b"")
        gone = Path(directory) / "gone.ims"
        check(
            session.existing([str(here), str(gone), str(here), ""]) == [str(here)],
            "missing and repeated paths are dropped",
        )
        check(session.existing(None) == [], "nothing saved is nothing to open")


def main() -> int:
    test_round_trip()
    test_reopening_guard()
    test_existing()
    print()
    if _failures:
        print(f"{len(_failures)} check(s) failed")
        return 1
    print("all session checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
