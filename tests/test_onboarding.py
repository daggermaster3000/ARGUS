"""Checks for the guided tour's script and its seen/not-seen memory.

The overlay itself is exercised by ``smoke_gui.py``; this is the part that needs
no Qt.

Run with::

    python tests/test_onboarding.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from microscopy_viewer import onboarding as ob  # noqa: E402

_failures: list[str] = []


def check(condition: bool, message: str) -> None:
    if condition:
        print(f"  ok   {message}")
    else:
        print(f"  FAIL {message}")
        _failures.append(message)


def test_script() -> None:
    print("the tour script")
    check(len(ob.TOUR) >= 5, f"a tour of several steps ({len(ob.TOUR)})")
    check(not ob.TOUR[0].target, "it opens without a target, centred")
    check(all(step.title and step.body for step in ob.TOUR), "every step says something")
    check(
        any("Esc" in step.body for step in ob.TOUR[:1]),
        "the first step says how to stop",
    )
    for step in ob.TOUR:
        if step.tab:
            check(bool(step.panel), f"“{step.title}” names the panel its tab is on")


def test_resolve() -> None:
    print("finding targets")

    class Leaf:
        pass

    class Root:
        pass

    root = Root()
    root.panel = Leaf()
    root.panel.button = "the button"
    root.panel.buttons = {"open": "open button"}
    check(ob.resolve(root, "panel.button") == "the button", "attribute path")
    check(ob.resolve(root, "panel.buttons.open") == "open button", "dictionary keys too")
    check(ob.resolve(root, "panel.missing.button") is None, "a broken path is None")
    check(ob.resolve(root, "") is root, "an empty path is the root")


def test_seen_state() -> None:
    print("remembering the tour was seen")
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "sub" / "onboarding.json"
        check(not ob.has_seen(path), "nothing recorded means not seen")
        ob.mark_seen(path, finished=False)
        check(ob.has_seen(path), "stopping early counts as seen")
        path.write_text('{"tour_version": 0}', encoding="utf-8")
        check(not ob.has_seen(path), "an older tour version is shown again")
        path.write_text("not json", encoding="utf-8")
        check(not ob.has_seen(path), "a corrupt file means not seen, not a crash")


def main() -> int:
    for test in (test_script, test_resolve, test_seen_state):
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
