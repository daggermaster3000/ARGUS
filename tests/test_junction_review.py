"""Reviewing a run's QC pictures: what is decided, kept, and queued to redraw.

Run with:  python tests/test_junction_review.py

No Qt, no napari: only :mod:`microscopy_viewer.junction_review`.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from microscopy_viewer import junction as jn  # noqa: E402
from microscopy_viewer import junction_review as jr  # noqa: E402

_failures: list[str] = []


def check(condition: bool, message: str) -> None:
    status = "ok  " if condition else "FAIL"
    print(f"  {status} {message}")
    if not condition:
        _failures.append(message)


def _folder(root: Path) -> Path:
    """A QC folder with five pictures, and three of their samples beside it."""
    import tifffile
    from PIL import Image

    qc = root / "junction_QC"
    for name, status in (("a_s1", "ok"), ("a_s2", "check"), ("a_s3", "check"), ("a_s4", "fail"), ("gone", "check")):
        jn.file_picture(qc, name, status, Image.new("RGB", (32, 32), "black"))
    for name in ("a_s1", "a_s2", "a_s3"):
        tifffile.imwrite(str(root / f"{name}.ome.tiff"), np.zeros((4, 4), dtype=np.uint16))
    return qc


def test_scanning_and_deciding() -> None:
    print("reviewing a QC folder")
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        qc = _folder(root)

        items = jr.scan(qc, root)
        check([item.name for item in items] == ["a_s4", "a_s2", "a_s3", "gone", "a_s1"],
              f"the pictures come worst first, then by name ({[item.name for item in items]})")
        check([item.status for item in items] == ["fail", "check", "check", "check", "ok"],
              "each with the status of the folder it is in")
        by_name = {item.name: item for item in items}
        check(by_name["a_s2"].path == root / "a_s2.ome.tiff" and by_name["gone"].path is None,
              "and its sample's file, when there is one beside the QC folder")
        check(all(item.decision == jr.UNDECIDED for item in items) and jr.redraw_queue(items) == [],
              "nothing is decided to begin with")

        jr.decide(by_name["a_s2"], jr.REDRAW, "line through the wrong fold")
        jr.decide(by_name["a_s3"], jr.REDRAW)
        jr.decide(by_name["gone"], jr.REDRAW)
        jr.decide(by_name["a_s4"], jr.EXCLUDE, "torn")
        jr.decide(by_name["a_s1"], jr.ACCEPT)
        check([item.name for item in jr.redraw_queue(items)] == ["a_s2", "a_s3"],
              "the queue is the samples to redraw that can be opened, in review order")
        check(jr.tally(items) == {"": 0, "accept": 1, "redraw": 3, "exclude": 1, "redrawn": 0},
              f"the tally counts every decision ({jr.tally(items)})")
        written = jr.save_decisions(qc, items)
        check(written == qc / jr.REVIEW_FILE and written.is_file(), "the decisions are written into the QC folder")

        again = {item.name: item for item in jr.scan(qc, root)}
        check(again["a_s2"].decision == jr.REDRAW and again["a_s2"].note == "line through the wrong fold"
              and again["a_s2"].reviewed != "", "and read back, note and date with them")
        check(again["a_s4"].decision == jr.EXCLUDE and again["a_s1"].decision == jr.ACCEPT, "every one of them")

        jr.decide(again["a_s2"], jr.REDRAWN)
        check([item.name for item in jr.redraw_queue(list(again.values()))] == ["a_s3"],
              "a sample that has been redrawn leaves the queue")
        jr.decide(again["a_s1"], jr.UNDECIDED)
        check(again["a_s1"].decision == "" and again["a_s1"].reviewed == "", "a decision can be taken back")
        line = jr.summary(list(again.values()))
        check("5 sample(s)" in line and "2 to redraw" in line and "1 redrawn" in line and "1 excluded" in line,
              f"the summary says how far the review has got ({line})")
        try:
            jr.decide(again["a_s1"], "maybe")
            check(False, "an unknown decision is refused")
        except ValueError:
            check(True, "an unknown decision is refused")

        # A rerun moves a picture to another folder; the decision stays with the sample.
        jr.save_decisions(qc, list(again.values()))
        jn.file_picture(qc, "a_s3", "ok")
        moved = {item.name: item for item in jr.scan(qc, root)}
        check(moved["a_s3"].status == "ok" and moved["a_s3"].decision == jr.REDRAW,
              "a picture a rerun filed elsewhere keeps its decision")
        check(jr.scan(root / "nowhere", root) == [] and jr.load_decisions(root / "nowhere") == {},
              "a folder that is not there is simply empty")


def main() -> int:
    for test in (test_scanning_and_deciding,):
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
    sys.exit(main())
